"""CPU structural/algebra regression checks; GPU validation is separate."""
import importlib.util
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_generated_c1_single_entry_and_projection_reuse():
    source=(ROOT/'crates/plkv-kernels/src/cutile/c1.rs').read_text()
    assert source==load('generate_c1').generated()
    lengths = load('generate_c1').LENGTHS
    assert source.count('table.load_tile(const_shape![1], [logical])') == len(lengths)
    assert 'extract(' not in source
    for module in source.split('#[cutile::module]')[1:]:
        projection,score=module.split('pub fn model_small_scores_fp16_storage_preprojected')
        assert 'reduce_sum(kp * q_row' in projection
        assert 'k_projection' not in score and 'projected_query.load_tile' in score
        assert '0.125f32' in score
    example=(ROOT/'crates/plkv-kernels/examples/c1_rtable.rs').read_text()
    pipeline=example.split('if phase == "pipeline"')[1].split('return;')[0]
    assert '"projection"' in pipeline and pipeline.index('"projection"')<pipeline.index('"score"')
    assert 'p15b_model_profile_kernel_8192::model_small_context_fp16_storage_rtable_8192' in example
    for n in lengths:
        assert f'launch_{n}' in example


def test_factoring_paging_and_query_refresh():
    rng=np.random.default_rng(42)
    for n in (1024,2048,4096,8192):
        blocks=n//16;table=(17*np.arange(blocks)+11)%blocks
        assert len(set(table))==blocks and not np.array_equal(table,np.arange(blocks))
        z=rng.normal(size=(n,32)).astype(np.float16).astype(np.float64)
        physical=np.empty_like(z).reshape(blocks,16,32)
        physical[table]=z.reshape(blocks,16,32)
        np.testing.assert_array_equal(physical[table].reshape(n,32),z)
        assert not np.array_equal(physical.reshape(n,32),z)
        q=rng.normal(size=(16,64));kp=rng.normal(size=(4,32,64))
        projected=np.einsum('hd,hld->hl',q,kp[np.arange(16)//4])
        factored=projected@z.T/8
        rebuilt=np.einsum('tl,hld->thd',z,kp[np.arange(16)//4])
        reference=np.einsum('hd,thd->ht',q,rebuilt)/8
        np.testing.assert_allclose(factored,reference,atol=1e-12,rtol=1e-12)
        assert not np.allclose((projected*2)@z.T/8,factored)


def test_independent_oracle_rejects_paging_scaling_and_nan(tmp_path):
    rng=np.random.default_rng(7);n=1024
    q=rng.normal(size=(16,64)).astype(np.float32)*0.1
    z=rng.normal(size=(n,32)).astype(np.float32)*0.1
    kp=rng.normal(size=(32,4,64)).astype(np.float32)*0.1
    vp=kp.copy(); zz=z.astype(np.float16).astype(np.float64)
    k=np.einsum('tl,lhd->thd',zz,kp.astype(float))[:,np.arange(16)//4]
    scores=np.einsum('hd,thd->ht',q.astype(float),k)/8
    probs=np.exp(scores-scores.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True)
    ctx=np.einsum('ht,thd->hd',probs,k)
    d={k:v.ravel().tolist() for k,v in [('q',q),('latent_logical_prequantization',z),('kp',kp),('vp',vp),('scores',scores),('probabilities',probs),('context',ctx)]}
    d['table']=((17*np.arange(64)+11)%64).tolist()
    p=tmp_path/'data.json';oracle=load('validate_c1_oracle')
    p.write_text(json.dumps(d));assert oracle.validate(p)['pass']
    for bad in (float('nan'),d['scores'][0]+1):
        copy={**d,'scores':d['scores'].copy()};copy['scores'][0]=bad
        p.write_text(json.dumps(copy));assert not oracle.validate(p)['pass']
    d['scores']=np.roll(scores,16,axis=1).ravel().tolist()
    p.write_text(json.dumps(d));assert not oracle.validate(p)['pass']
