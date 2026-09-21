#!/usr/bin/env python3
"""Validate sealed C1 campaign and generate descriptive process-level tables."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics as st


def analyze(base, out, allow_pilot=False):
    for line in (base/'SHA256SUMS').read_text().splitlines():
        digest, rel=line.split('  ',1)
        assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==digest, rel
    manifest=json.loads((base/'manifest.json').read_text())
    assert manifest['complete']
    assert manifest['mode']=='main' or allow_pilot
    pilot=manifest['mode']=='pilot'
    lengths=(1024,) if pilot else (1024,2048,4096,8192)
    processes=1 if pilot else 3
    grouped=defaultdict(list)
    for run in manifest['runs']:
        assert run['status']=='OK'
        d=base/f"seq{run['seq']}_p{run['process']}"
        assert json.loads((d/'independent_oracle.json').read_text())['pass']
        if not pilot:
            assert json.loads((d/'query_refresh.json').read_text())['pass']
        rows=[json.loads(line) for line in (d/'samples.jsonl').read_text().splitlines()]
        assert len(rows)==13*manifest['samples_per_phase']
        keys=set()
        for r in rows:
            key=(r['variant'],r['component'],r['iteration']);assert key not in keys;keys.add(key)
            assert r['seq']==run['seq'] and r['process']==run['process'] and r['latency_ms']>0
            grouped[(r['seq'],r['process'],r['variant'],r['component'])].append(r['latency_ms'])
    out.mkdir(exist_ok=False,parents=True)
    process_rows=[]
    for (n,p,v,c),values in sorted(grouped.items()):
        assert len(values)==manifest['samples_per_phase']
        q=st.quantiles(values,n=4,method='inclusive')
        process_rows.append(dict(seq=n,process=p,variant=v,component=c,median_ms=st.median(values),iqr_ms=q[2]-q[0],min_ms=min(values),max_ms=max(values)))
    with (out/'process_components.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(process_rows[0]));w.writeheader();w.writerows(process_rows)
    table=['# C1 generated results','',f'Source: `{base}`. {processes} independent process(es) per length; median of process medians. Mode: {manifest["mode"]}. Pilot is diagnostic only, excluded from main evidence.','', '| N | A1 ms | B1 ms | C1 ms | paired C1/B1 median [range] | paired C1/A1 median [range] |','|---:|---:|---:|---:|---|---|']
    result=[]
    for n in lengths:
        med={v:[st.median(grouped[n,p,v,'pipeline']) for p in range(processes)] for v in ('A1','B1','C1')}
        ratios={v:[c/b for c,b in zip(med['C1'],med[v])] for v in ('A1','B1')}
        fmt=lambda v:f'{st.median(v):.4f} [{min(v):.4f}, {max(v):.4f}]'
        table.append(f"| {n} | {st.median(med['A1']):.6f} | {st.median(med['B1']):.6f} | {st.median(med['C1']):.6f} | {fmt(ratios['B1'])} | {fmt(ratios['A1'])} |")
        result.append({'seq':n,'process_pipeline_medians_ms':med,'paired_ratios':ratios,'projected_query_bytes':2048,'common_workspace_bytes_per_variant':2*16*n*4+4096})
    table+=['','Within-process IQR and all component medians: `process_components.csv`.', 'All process pipeline medians/ratios: `results.json`. No sample-level inference; no equivalence claim.']
    (out/'results.md').write_text('\n'.join(table)+'\n')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('output',type=Path);p.add_argument('--allow-pilot',action='store_true');a=p.parse_args();analyze(a.campaign,a.output,a.allow_pilot)
