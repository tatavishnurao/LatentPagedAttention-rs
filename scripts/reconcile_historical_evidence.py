#!/usr/bin/env python3
"""Semantically recompute retained P0/P0.5/P1.5A/P1.5B evidence."""
from __future__ import annotations
import argparse,csv,glob,json,math,statistics
from pathlib import Path

def records(path):
 with open(path,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def med(xs):return statistics.median(xs)
def row(rows,phase,metric,value,unit,stat,n,variant,seq,path,status,notes=''):
 rows.append(dict(phase=phase,metric=metric,value=value,unit=unit,statistic=stat,sample_count=n,variant=variant,sequence_length=seq,input_artifact=path,verification_status=status,notes=notes))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 out=[]; narrative=[]
 # P0 raw
 p='reports/p0_gpu_baseline/abc_samples_1788501169.jsonl'; rs=records(p)
 for v,label in [('full_kv','A0'),('latent_original','B0')]:
  xs=[r['latency_ms'] for r in rs if r['variant']==v and r['timing']=='gpu_event']
  row(out,'P0',f'{label}_CUDA_event_latency',repr(med(xs)),'ms','median',len(xs),label,1024,p,'RAW_VERIFIED')
 a0=med([r['latency_ms'] for r in rs if r['variant']=='full_kv' and r['timing']=='gpu_event']);b0=med([r['latency_ms'] for r in rs if r['variant']=='latent_original' and r['timing']=='gpu_event'])
 row(out,'P0','B0_over_A0',repr(b0/a0),'ratio','ratio_of_medians',300,'A0/B0',1024,p,'SUMMARY_RECOMPUTED')
 # P05 all raw processes and manifest policy
 mp=Path('reports/p05_clean_display_replications/replications_manifest.json'); manifest=json.loads(mp.read_text()); accepted=[]; rejected=0
 for rec in manifest['records']:
  sample=rec['samples_path']; rr=records(sample)
  av=[x['latency_ms'] for x in rr if x['variant']=='full_kv' and x['timing']=='gpu_event'];bv=[x['latency_ms'] for x in rr if x['variant']=='latent_original' and x['timing']=='gpu_event']
  am,bm=med(av),med(bv);ratio=bm/am
  status='RAW_VERIFIED'; row(out,'P0.5',f'attempt_{rec["attempt"]}_B_over_A',repr(ratio),'ratio','ratio_of_raw_medians',f'{len(av)}+{len(bv)}','A0/B0',1024,sample,status,rec['status']+'; '+('; '.join(rec['reasons'])))
  if rec['status']=='ACCEPTED':accepted.append((rec['attempt'],am,bm,ratio))
  else:rejected+=1
 ratios=[x[3] for x in accepted]; diffs=[(x[2]-x[1])*1000 for x in accepted]
 sign_p=2/(2**len(ratios)) if all(x>1 for x in ratios) or all(x<1 for x in ratios) else None
 coverage=1-2/(2**len(ratios))
 for metric,val,unit in [('accepted_count',len(accepted),'process'),('rejected_count',rejected,'process'),('median_B_over_A',med(ratios),'ratio'),('min_B_over_A',min(ratios),'ratio'),('max_B_over_A',max(ratios),'ratio'),('median_paired_difference',med(diffs),'us'),('two_sided_sign_test_p',sign_p,'p'),('minmax_median_interval_coverage',coverage,'fraction')]:row(out,'P0.5',metric,repr(val),unit,'recomputed',len(ratios),'A0/B0',1024,str(mp),'SUMMARY_RECOMPUTED')
 narrative.append(f'P0.5 accepted attempts: {[x[0] for x in accepted]}; rejected={rejected}; median ratio={med(ratios)!r}; range=[{min(ratios)!r},{max(ratios)!r}]; sign p={sign_p}; coverage={coverage}.')
 # P15A raw JSONL medians
 for p in sorted(glob.glob('reports/p15_kernel_scaling/p15_samples_seq*.jsonl')):
  rr=records(p); seq=int(Path(p).name.split('seq')[1].split('_')[0])
  for k in sorted({x['kernel'] for x in rr}):
   xs=[x['latency_ms'] for x in rr if x['kernel']==k]
   row(out,'P1.5A',k,repr(med(xs)),'ms','median',len(xs),k[0],seq,p,'RAW_VERIFIED')
 # P15B raw medians and derived values
 vals={}
 for p in sorted(glob.glob('reports/p15b_rtable/summary_seq*.jsonl')):
  rr=records(p); seq=int(Path(p).name.split('seq')[1].split('_')[0])
  for k in sorted({x['kernel'] for x in rr}):
   xs=[x['latency_ms'] for x in rr if x['kernel']==k]; vals[seq,k]=med(xs)
   row(out,'P1.5B',k,repr(vals[seq,k]),'ms','median',len(xs),k.split('_')[0],seq,p,'RAW_VERIFIED')
 ks=['A0_score','A1_score','B0_score','B1_score','A0_context','A1_context','B0_context','B1_context']
 seqs=sorted({x[0] for x in vals})
 for k in ks:
  for x,y in zip(seqs,seqs[1:]):
   ratio=vals[y,k]/vals[x,k]; row(out,'P1.5B',k+'_doubling_'+str(x)+'_'+str(y),repr(ratio),'ratio','adjacent_ratio','derived',k.split('_')[0],f'{x}-{y}',f'reports/p15b_rtable/summary_seq{x}_*.jsonl; reports/p15b_rtable/summary_seq{y}_*.jsonl','SUMMARY_RECOMPUTED')
  slope=math.log(vals[seqs[-1],k]/vals[seqs[0],k],2)/math.log(seqs[-1]/seqs[0],2); row(out,'P1.5B',k+'_endpoint_slope',repr(slope),'log2 ratio','endpoint',f'{seqs[0]}-{seqs[-1]}',k.split('_')[0],f'{seqs[0]}-{seqs[-1]}','reports/p15b_rtable/summary_seq*.jsonl','SUMMARY_RECOMPUTED','descriptive, not asymptotic')
 for metric,val in [('B1_over_A1_score_8K',vals[8192,'B1_score']/vals[8192,'A1_score']),('B1_over_A1_context_8K',vals[8192,'B1_context']/vals[8192,'A1_context']),('latent_context_reduction_vs_full_8K',1-vals[8192,'B1_context']/vals[8192,'A1_context'])]:row(out,'P1.5B',metric,repr(val),'ratio','derived',10,'A1/B1',8192,'reports/p15b_rtable/summary_seq8192_1788719774.jsonl','SUMMARY_RECOMPUTED')
 fields=list(out[0]);
 with (a.output_dir/'RECOMPUTED_VALUES.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 (a.output_dir/'RECOMPUTATION_REPORT.md').write_text('# Historical Evidence Recomputation\n\n'+ '\n\n'.join(narrative)+f'\n\nP0, P0.5, P1.5A, and P1.5B individual JSONL samples were parsed and medians recomputed. Rows and exact source artifacts are in `RECOMPUTED_VALUES.csv`. NCU binary-report extraction is recorded separately; an out-of-range hit-rate value is excluded from physical interpretation. Historical artifacts do not embed source commits, so raw measurement values can be certified while immutable source identity remains partial/unresolved.\n',encoding='utf-8')
 print(f'rows={len(out)}')
if __name__=='__main__':main()
