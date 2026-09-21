#!/usr/bin/env python3
"""Serial, thermally guarded C1 campaign; see docs/C1_PROTOCOL.md."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from run_paper_repair_correctness import wsl_driver_library
from run_p15b_rtable import query_gpu, wait_for_cooldown
from validate_c1_oracle import validate

ROOT = Path(__file__).resolve().parents[1]

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')

def capture(cmd):
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'command':cmd, 'returncode':p.returncode, 'output':p.stdout}

def seal(base):
    files=sorted(p for p in base.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (base/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(base)}\n' for p in files))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    base=ROOT/'reports'/f'c1_{"pilot" if args.pilot else "main"}_{stamp}'
    base.mkdir()
    print(base,flush=True)
    env={**os.environ,'CUDA_TOOLKIT_PATH':'/usr/local/cuda-13.3','LIBCLANG_PATH':'/usr/lib/llvm-18/lib','CUTILE_TILEIRAS_PATH':'/usr/local/cuda-13.3/bin/tileiras','OPENBLAS_NUM_THREADS':'1'}
    wsl=wsl_driver_library()
    if wsl: env['LD_LIBRARY_PATH']=':'.join([wsl,'/usr/lib/wsl/lib',env.get('LD_LIBRARY_PATH','')])
    manifest={'mode':'pilot' if args.pilot else 'main','base_commit':capture(['git','rev-parse','HEAD'])['output'].strip(),'policy':'docs/C1_PROTOCOL.md','runs':[], 'environment_overrides':{k:env[k] for k in ('CUDA_TOOLKIT_PATH','LIBCLANG_PATH','CUTILE_TILEIRAS_PATH','LD_LIBRARY_PATH') if k in env}}
    for name,cmd in [('git_status',['git','status','--porcelain']),('gpu',['nvidia-smi','-q']),('rust',['rustc','--version']),('cuda',['/usr/local/cuda-13.3/bin/nvcc','--version'])]:save(base/f'{name}.json',capture(cmd))
    (base/'source.patch').write_text(capture(['git','diff','HEAD'])['output'])
    sources=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    sources.update(str(p.relative_to(ROOT)) for folder in ('scripts','crates','tests','docs') for p in (ROOT/folder).rglob('*') if p.is_file() and (p.suffix in ('.py','.rs') or p.name.startswith('C1_')))
    for rel in sorted(sources):
        p=ROOT/rel
        if p.is_file():
            dst=base/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    build=['cargo','build','--release','-p','plkv-kernels','--features','gpu-cutile','--example','c1_rtable']
    with (base/'build.log').open('w') as f:
        rc=subprocess.run(build,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    manifest['build_command']=build
    manifest['build_returncode']=rc
    save(base/'manifest.json',manifest)
    if rc: seal(base);raise SystemExit('build failed')
    binary=ROOT/'target/release/examples/c1_rtable'
    manifest['executable_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    plan=[(1024,0)] if args.pilot else [(n,p) for n in (1024,2048,4096,8192) for p in range(3)]
    warmup,samples=(1,3) if args.pilot else (3,12)
    manifest.update(warmup=warmup,samples_per_phase=samples,plan=plan)
    try:
        for n,idx in plan:
            d=base/f'seq{n}_p{idx}';d.mkdir()
            command=[str(binary),'--seq',str(n),'--process',str(idx),'--warmup',str(warmup),'--iterations',str(samples),'--output-dir',str(d)]
            run={'seq':n,'process':idx,'command':command,'status':'STARTING','telemetry':[]}
            manifest['runs'].append(run)
            save(base/'manifest.json',manifest)
            pre=query_gpu();run['telemetry'].append(pre)
            if pre['temperature_c']>=84 and not wait_for_cooldown(82,360,'pre'):
                run['status']='SKIPPED_THERMAL';break
            with (d/'stdout.log').open('w') as f, (d/'telemetry.jsonl').open('w') as telemetry:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        snap=query_gpu();telemetry.write(json.dumps(snap)+'\n');telemetry.flush()
                        if snap['temperature_c']>=86:
                            run['status']='THERMAL_ABORT';process.kill();break
                        time.sleep(0.5)
                finally:
                    if process.poll() is None: process.kill()
                    run['returncode']=process.wait()
            run['telemetry'].append(query_gpu())
            if run['status']=='THERMAL_ABORT':break
            if run['returncode']!=0:
                run['status']='PROCESS_ERROR';break
            oracle=validate(d/'oracle_inputs_outputs.json');save(d/'independent_oracle.json',oracle)
            run['status']='OK' if oracle['pass'] else 'ORACLE_FAILURE'
            save(base/'manifest.json',manifest)
            print(n,idx,run['status'],flush=True)
            if run['status']!='OK':break
            wait_for_cooldown(78,300,'between processes')
    except BaseException as exc:
        manifest['exception']=repr(exc)
        raise
    finally:
        manifest['complete']=len(manifest['runs'])==len(plan) and all(r['status']=='OK' for r in manifest['runs'])
        save(base/'manifest.json',manifest)
        seal(base)
    if not manifest['complete']:raise SystemExit('campaign incomplete; retained '+str(base))

if __name__=='__main__':main()
