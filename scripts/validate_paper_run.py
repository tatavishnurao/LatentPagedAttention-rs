#!/usr/bin/env python3
"""Validate paper-evidence run directories and recompute timing summaries."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
REQ=['run_manifest.json','raw_samples.csv','summary.json','artifact_checksums.sha256']
FIELDS=['run_id','timestamp_utc','git_commit','dirty_worktree','variant','kernel_names','sequence_length','block_size','logical_page_count','block_table','block_table_is_identity','storage_dtype','arithmetic_dtype','head_configuration','latent_width','warmup_count','iteration_count','process_index','execution_order','timing_method','timing_boundary','gpu_name','compute_capability','driver_version','cuda_version','cutile_version','temperature_before','temperature_after','graphics_clock_before','graphics_clock_after','power_limit','command']
def sha(p):h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();errors=[]
 for d in ([a.directory] if (a.directory/'run_manifest.json').exists() else sorted(x for x in a.directory.iterdir() if x.is_dir())):
  if not all((d/x).exists() for x in REQ):errors.append(f'{d}: required files missing');continue
  m=json.loads((d/'run_manifest.json').read_text());s=json.loads((d/'summary.json').read_text())
  for f in FIELDS:
   if f not in m:errors.append(f'{d}: manifest missing {f}')
  if not m.get('variant'):errors.append(f'{d}: variant identity absent')
  if not m.get('timing_boundary'):errors.append(f'{d}: timing boundary absent')
  if m.get('timing_method') not in ['PROCESS_END_TO_END','CUDA_EVENT_KERNEL','CUDA_EVENT_PIPELINE']:errors.append(f'{d}: invalid timing method')
  if m.get('block_table_is_identity') is not False:errors.append(f'{d}: nonidentity paging not recorded')
  if m.get('block_table') is None:errors.append(f'{d}: block table absent')
  groups={}
  with (d/'raw_samples.csv').open() as f:
   for r in csv.DictReader(f):groups.setdefault(f"{r['variant']}_{r['component']}",[]).append(float(r['latency_ms']))
  for k,xs in groups.items():
   expected=s['kernels'][k]['median_ms'];actual=statistics.median(xs)
   if not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12):errors.append(f'{d}: median mismatch {k}')
  correctness=s.get('correctness',{})
  for pair in ['a1_a0_elementwise','b1_b0_elementwise']:
   c=correctness.get(pair,{})
   if not c or c.get('element_count',0)<=0 or any(c.get(x,-1)!=0 for x in ['nan_count','positive_infinity_count','negative_infinity_count']):errors.append(f'{d}: nonfinite/incomplete {pair}')
  for name, value in correctness.items():
   if isinstance(value, dict) and any(not math.isfinite(float(value[k])) for k in ['max_absolute_error','max_relative_error','mean_absolute_error'] if k in value):errors.append(f'{d}: non-finite correctness metric {name}')
  for line in (d/'artifact_checksums.sha256').read_text().splitlines():
   digest,name=line.split('  ',1);p=d/name
   if not p.exists() or sha(p)!=digest:errors.append(f'{d}: checksum {name}')
 if errors:print('PAPER_RUN_VALIDATION_FAIL');print('\n'.join('- '+x for x in errors));return 1
 print('PAPER_RUN_VALIDATION_OK');return 0
if __name__=='__main__':raise SystemExit(main())
