#!/usr/bin/env python3
"""E0b: A1 vs C1, 1K and 8K, >=5 cool-start processes each, under the 2026-10-02 policy.

Policy: abort >= 91 C (in-process and external); cooldown < 80 C before each process
(max 300 s, else tag hot-start); 100 ms external telemetry; THROTTLED tagging is done in
analysis. Lengths are interleaved per round so drift does not align with length.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path('<repo>')
BIN = ROOT / 'target/release/examples/c1_rtable'
FIELDS = 'timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active'
ENV = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ['PATH']}",
           CUDA_TOOLKIT_PATH='/usr/local/cuda-13.3', LIBCLANG_PATH='/usr/lib/llvm-18/lib',
           CUTILE_TILEIRAS_PATH='/usr/local/cuda-13.3/bin/tileiras', OPENBLAS_NUM_THREADS='1')
PLAN = {1024: 200, 8192: 60}
PROCESSES = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def temp():
    out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return int(out.strip())


def cooldown():
    start = time.time()
    while (t := temp()) >= 80:
        if time.time() - start >= 300:
            return 'hot-start', t, round(time.time() - start, 1)
        time.sleep(5)
    return 'cool-start', t, round(time.time() - start, 1)


def main():
    stamp = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
    root = Path.home() / 'plkv_runs' / f'e0b_{stamp}'
    root.mkdir(parents=True)
    manifest = {'binary_sha256': subprocess.run(['sha256sum', str(BIN)], capture_output=True, text=True).stdout.split()[0],
                'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                'plan': PLAN, 'processes': PROCESSES, 'runs': []}
    for p in range(PROCESSES):
        for seq, iterations in PLAN.items():
            label = f'seq{seq}_p{p}'
            out = root / label
            condition, start_temp, waited = cooldown()
            tele = (root / f'{label}_telemetry.csv').open('w')
            logger = subprocess.Popen(['nvidia-smi', f'--query-gpu={FIELDS}', '--format=csv,noheader,nounits',
                                       '-lms', '100'], stdout=tele, stderr=subprocess.STDOUT)
            cmd = [str(BIN), '--seq', str(seq), '--process', str(p), '--variants', 'A1,C1',
                   '--warmup', '10', '--iterations', str(iterations), '--output-dir', str(out)]
            t0 = time.time()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=(root / f'{label}_stdout.log').open('w'),
                                    stderr=subprocess.STDOUT)
            killed = {'value': False}

            def watch():
                while proc.poll() is None:
                    try:
                        if temp() >= 91:
                            killed['value'] = True
                            proc.kill()
                    except ValueError:
                        pass
                    time.sleep(0.5)
            w = threading.Thread(target=watch)
            w.start()
            rc = proc.wait()
            w.join()
            logger.terminate()
            logger.wait()
            tele.close()
            run = {'label': label, 'seq': seq, 'process': p, 'start_condition': condition,
                   'start_temp_c': start_temp, 'cooldown_wait_s': waited, 'returncode': rc,
                   'external_thermal_kill': killed['value'], 'wall_s': round(time.time() - t0, 2),
                   'start_unix': t0, 'cmd': cmd}
            manifest['runs'].append(run)
            print(json.dumps({k: v for k, v in run.items() if k != 'cmd'}), flush=True)
            (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            if killed['value'] or rc == 3:
                print('THERMAL ABORT: stopping campaign', flush=True)
                return
    print('CAMPAIGN_DIR', root)


if __name__ == '__main__':
    main()
