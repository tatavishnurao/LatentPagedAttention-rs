#!/usr/bin/env python3
"""Run the bounded 1K repair-correctness profiles with auditable artifacts.

No 16K/32K path is accepted. This wrapper snapshots the exact source, retains
raw samples, emits required manifests, and checksums every produced artifact.
"""
from __future__ import annotations
import csv,glob,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENV={**os.environ,"CUDA_TOOLKIT_PATH":"/usr/local/cuda-13.3","LIBCLANG_PATH":"/usr/lib/llvm-18/lib","CUTILE_TILEIRAS_PATH":"/usr/local/cuda-13.3/bin/tileiras"}
PROFILES=(("tiny_active",17),("model_small",1024),("partial_runtime_active",1009))
SOURCES=("crates/plkv-kernels/examples/p15b_rtable.rs","crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs","crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs","Cargo.lock","Cargo.toml")
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def cmdout(cmd):
 try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
 except Exception:return None
def wsl_driver_library():
    try:
        release=open('/proc/sys/kernel/osrelease', encoding='utf-8').read().lower()
    except OSError:
        return None
    if 'microsoft' not in release and 'wsl' not in release:
        return None
    matches=sorted({os.path.dirname(p) for p in glob.glob('/usr/lib/wsl/drivers/*/libnvidia-ptxjitcompiler.so.1') if os.path.isfile(p)})
    if len(matches)!=1:
        raise RuntimeError(f'expected exactly one WSL PTX-JIT directory, found {matches!r}')
    return matches[0]

def gpu():
 s=cmdout(["nvidia-smi","--query-gpu=name,driver_version,temperature.gpu,clocks.gr,clocks.mem,power.limit","--format=csv,noheader,nounits"])
 if not s:return {"gpu_name":None,"driver_version":None,"temperature":None,"graphics_clock":None,"memory_clock":None,"power_limit":None,"reason":"nvidia-smi unavailable"}
 x=[v.strip() for v in s.split(',')]
 def number(value):
  try:return float(value)
  except ValueError:return None
 return dict(gpu_name=x[0],driver_version=x[1],temperature=number(x[2]),graphics_clock=number(x[3]),memory_clock=number(x[4]),power_limit=number(x[5]),reason="management field unavailable" if any(number(v) is None for v in x[2:]) else None)
def checksums(d):
 lines=[]
 for p in sorted(x for x in d.rglob('*') if x.is_file() and x.name!='artifact_checksums.sha256'):lines.append(f'{sha(p)}  {p.relative_to(d).as_posix()}')
 (d/'artifact_checksums.sha256').write_text('\n'.join(lines)+'\n')
def main():
 if any(x in sys.argv for x in ['--seq','16384','32768']):raise SystemExit('This bounded wrapper does not accept 16K/32K.')
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); base=ROOT/'reports'/f'repair_correctness_{stamp}'
 binary=ROOT/'target/release/examples/p15b_rtable'
 build=["cargo","build","--release","-p","plkv-kernels","--features","gpu-cutile","--example","p15b_rtable"]
 subprocess.run(build,cwd=ROOT,env=ENV,check=True)
 base.mkdir()
 wsl_dir=wsl_driver_library()
 if wsl_dir:
  existing=ENV.get('LD_LIBRARY_PATH')
  ENV['LD_LIBRARY_PATH']=':'.join([wsl_dir,'/usr/lib/wsl/lib']+([existing] if existing else []))
 head=cmdout(['git','-C',str(ROOT),'rev-parse','HEAD']);dirty=bool(cmdout(['git','-C',str(ROOT),'status','--porcelain']))
 for index,(profile,active) in enumerate(PROFILES,1):
  d=base/profile;d.mkdir();snap=d/'source_snapshot';snap.mkdir()
  for rel in SOURCES:
   dst=snap/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
  before=gpu(); rawdir=d/'native';rawdir.mkdir(); command=[str(binary),'--seq','1024','--active-seq-len',str(active),'--warmup','1','--iterations','1','--output-dir',str(rawdir)]
  started=datetime.now(timezone.utc).isoformat(); cp=subprocess.run(command,cwd=ROOT,env=ENV,text=True,capture_output=True);(d/'stdout.log').write_text(cp.stdout);(d/'stderr.log').write_text(cp.stderr)
  if cp.returncode:raise SystemExit(f'{profile} failed: {cp.stderr[-2000:]}')
  after=gpu(); summaries=list(rawdir.glob('summary_seq*.json')); samples=list(rawdir.glob('summary_seq*.jsonl'))
  if len(summaries)!=1 or len(samples)!=1:raise SystemExit(f'{profile}: missing native outputs')
  summary=json.loads(summaries[0].read_text());(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  with samples[0].open() as f, (d/'raw_samples.csv').open('w',newline='') as o:
   w=csv.DictWriter(o,fieldnames=['variant','component','iteration','latency_ms']);w.writeheader()
   for line in f:
    r=json.loads(line);v,c=r['kernel'].split('_',1);w.writerow(dict(variant=v,component=c,iteration=r['iteration'],latency_ms=r['latency_ms']))
  block_table=summary['block_table']; cuda=cmdout(['/usr/local/cuda-13.3/bin/nvcc','--version'])
  manifest={"run_id":f"repair-correctness-{stamp}-{profile}","timestamp_utc":started,"git_commit":head,"dirty_worktree":dirty,"source_snapshot":"source_snapshot/","variant":["A0","A1","B0","B1"],"kernel_names":["A0_score","A1_score","A0_context","A1_context","B0_score","B1_score","B0_context","B1_context"],"sequence_length":1024,"active_sequence_length":active,"profile":profile,"block_size":16,"logical_page_count":64,"block_table":block_table,"block_table_is_identity":summary['block_table_is_identity'],"storage_dtype":"FP16","arithmetic_dtype":"FP32","head_configuration":{"query_heads":16,"kv_heads":4,"head_dimension":64},"latent_width":32,"warmup_count":1,"iteration_count":1,"process_index":index,"execution_order":["A0_score","A1_score","A0_context","A1_context","B0_score","B1_score","B0_context","B1_context","A0_pipeline","A1_pipeline","B0_pipeline","B1_pipeline"],"timing_method":"CUDA_EVENT_KERNEL","timing_boundary":"single launch for component phases; three-launch span for pipeline phases","gpu_name":before['gpu_name'],"compute_capability":"8.9 (configuration record; runtime query unavailable)","driver_version":before['driver_version'],"cuda_version":cuda,"cutile_version":"0.2.0 (Cargo.lock)","temperature_before":before['temperature'],"temperature_after":after['temperature'],"graphics_clock_before":before['graphics_clock'],"graphics_clock_after":after['graphics_clock'],"memory_clock_before":before['memory_clock'],"memory_clock_after":after['memory_clock'],"power_limit":before['power_limit'],"command":command,"wsl_nvidia_driver_dir":wsl_dir,"null_reasons":{k:before['reason'] for k,v in before.items() if v is None}}
  (d/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');checksums(d)
 print(base)
if __name__=='__main__':main()
