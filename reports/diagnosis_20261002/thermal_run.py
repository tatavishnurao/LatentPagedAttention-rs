#!/usr/bin/env python3
"""Run the unchanged c1_rtable binary under 100 ms GPU telemetry.

Policy (user override 2026-10-02): hard abort >= 91 C; cooldown to < 80 C (max 300 s),
otherwise tag hot-start; tag THROTTLED if any timing-window SM clock < 95 % of the
window mode or any thermal/HW slowdown reason bit is set.
"""
import argparse
import collections
import datetime as dt
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path('<repo>')
BIN = ROOT / 'target/release/examples/c1_rtable'
FIELDS = 'timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active'
THERMAL_BITS = {0x8: 'hw_slowdown', 0x20: 'sw_thermal', 0x40: 'hw_thermal', 0x80: 'hw_power_brake'}
POWER_CAP_BIT = 0x4
ABORT_C = 91
ENV = dict(os.environ, CUDA_TOOLKIT_PATH='/usr/local/cuda-13.3',
           LIBCLANG_PATH='/usr/lib/llvm-18/lib',
           CUTILE_TILEIRAS_PATH='/usr/local/cuda-13.3/bin/tileiras',
           OPENBLAS_NUM_THREADS='1')


def query():
    out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    t, c = out.strip().split(',')
    return int(t), int(c)


def cooldown(limit=80, max_s=300):
    start = time.time()
    while True:
        t, _ = query()
        if t < limit:
            return {'start_condition': 'cool-start', 'temp_c': t, 'waited_s': round(time.time() - start, 1)}
        if time.time() - start >= max_s:
            return {'start_condition': 'hot-start', 'temp_c': t, 'waited_s': round(time.time() - start, 1)}
        time.sleep(5)


def parse_ts(s):
    return dt.datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()


def run(seq, iterations, warmup, label, out_root, start_condition=None, skip_cooldown=False):
    out = out_root / label
    out.mkdir(parents=True, exist_ok=False)
    pre = {'start_condition': start_condition or 'hot-start', 'temp_c': query()[0], 'waited_s': 0} \
        if skip_cooldown else cooldown()
    tele_path = out / 'telemetry.csv'
    tele_f = tele_path.open('w')
    logger = subprocess.Popen(['nvidia-smi', f'--query-gpu={FIELDS}', '--format=csv,noheader,nounits',
                               '-lms', '100'], stdout=tele_f, stderr=subprocess.STDOUT)
    time.sleep(0.5)
    cmd = [str(BIN), '--seq', str(seq), '--process', '0', '--warmup', str(warmup),
           '--iterations', str(iterations), '--output-dir', str(out)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=(out / 'stdout.log').open('w'),
                            stderr=subprocess.STDOUT)
    progress = []
    aborted = {'value': False, 'max_temp': 0}
    samples = out / 'samples.jsonl'

    def watch():
        while proc.poll() is None:
            n = sum(1 for _ in samples.open()) if samples.exists() else 0
            progress.append((time.time(), n))
            try:
                last = tele_path.read_text().strip().splitlines()[-1].split(',')
                temp = int(last[1])
                aborted['max_temp'] = max(aborted['max_temp'], temp)
                if temp >= ABORT_C:
                    aborted['value'] = True
                    proc.kill()
            except (IndexError, ValueError):
                pass
            time.sleep(0.05)

    w = threading.Thread(target=watch)
    w.start()
    rc = proc.wait()
    w.join()
    t1 = time.time()
    time.sleep(0.3)
    logger.terminate()
    logger.wait()
    tele_f.close()

    rows = [json.loads(l) for l in samples.open()] if samples.exists() else []
    tele = []
    for line in tele_path.read_text().splitlines():
        p = [x.strip() for x in line.split(',')]
        try:
            tele.append({'t': parse_ts(p[0]), 'temp': int(p[1]), 'sm': int(p[2]), 'mem': int(p[3]),
                         'power': float(p[4]), 'reasons': int(p[5], 16)})
        except (ValueError, IndexError):
            continue
    # timing window: first time samples.jsonl had >=1 row (minus one poll) to process end
    first_row_t = next((t for t, n in progress if n > 0), None)
    win_start = (first_row_t - 0.2) if first_row_t else t0
    window = [x for x in tele if win_start <= x['t'] <= t1]
    summary = {'label': label, 'seq': seq, 'iterations': iterations, 'warmup': warmup, 'cmd': cmd,
               'returncode': rc, 'aborted_thermal': aborted['value'], 'pre': pre,
               'wall_s': round(t1 - t0, 2), 'timing_window_s': round(t1 - win_start, 2),
               'rows': len(rows), 'telemetry_samples_total': len(tele),
               'telemetry_samples_window': len(window)}
    if window:
        sms = [x['sm'] for x in window]
        mode = collections.Counter(sms).most_common(1)[0][0]
        reasons = 0
        for x in window:
            reasons |= x['reasons']
        thermal = [name for bit, name in THERMAL_BITS.items() if reasons & bit]
        low = sum(1 for s in sms if s < 0.95 * mode)
        summary.update({
            'sm_mode_mhz': mode, 'sm_min_mhz': min(sms), 'sm_max_mhz': max(sms),
            'sm_median_mhz': statistics.median(sms), 'frac_below_95pct_mode': round(low / len(sms), 3),
            'mem_mode_mhz': collections.Counter(x['mem'] for x in window).most_common(1)[0][0],
            'temp_start': window[0]['temp'], 'temp_max': max(x['temp'] for x in window),
            'power_median_w': statistics.median(x['power'] for x in window),
            'reasons_or': hex(reasons), 'thermal_bits': thermal,
            'power_cap_bit': bool(reasons & POWER_CAP_BIT),
            'throttled': bool(low) or bool(thermal),
        })
    med = collections.defaultdict(list)
    for r in rows:
        med[(r['variant'], r['component'])].append(r['latency_ms'] * 1000)
    summary['median_us'] = {f'{v}/{c}': round(statistics.median(x), 2) for (v, c), x in sorted(med.items())}
    summary['n'] = {f'{v}/{c}': len(x) for (v, c), x in sorted(med.items())}
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'progress.json').write_text(json.dumps(progress))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, required=True)
    ap.add_argument('--iterations', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out-root', default=str(ROOT / 'reports/diagnosis_20261002/e0'))
    ap.add_argument('--hot', action='store_true', help='skip cooldown, tag hot-start')
    a = ap.parse_args()
    s = run(a.seq, a.iterations, a.warmup, a.label, Path(a.out_root), skip_cooldown=a.hot)
    print(json.dumps({k: v for k, v in s.items() if k not in ('cmd',)}, indent=1))
