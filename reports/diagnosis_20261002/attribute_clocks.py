#!/usr/bin/env python3
"""Attribute each CUDA-event sample row to the SM clock at the time it was written.

Row write times come from 50 ms polling of samples.jsonl line counts (progress.json),
interpolated linearly between polls; clock = nearest 100 ms telemetry sample. Rows whose
nearest telemetry sample is >150 ms away, or whose surrounding progress polls straddle a
clock change, are marked unattributed. Bucket: HIGH if clock >= 0.95 * max window clock,
LOW if clock < 0.5 * max, else MID.
"""
import bisect
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path


def load(run_dir):
    run = Path(run_dir)
    prog = json.loads((run / 'progress.json').read_text())
    rows = [json.loads(l) for l in (run / 'samples.jsonl').open()]
    tele = []
    for line in (run / 'telemetry.csv').read_text().splitlines():
        p = [x.strip() for x in line.split(',')]
        try:
            tele.append((dt.datetime.strptime(p[0], '%Y/%m/%d %H:%M:%S.%f').timestamp(), int(p[2]),
                         int(p[1]), int(p[5], 16)))
        except (ValueError, IndexError):
            pass
    return prog, rows, tele


def attribute(run_dir):
    prog, rows, tele = load(run_dir)
    tt = [t for t, *_ in tele]
    # time at which line count first reached k
    reach = {}
    prev_t, prev_n = None, 0
    for t, n in prog:
        for k in range(prev_n + 1, n + 1):
            if prev_t is None:
                reach[k] = t
            else:
                reach[k] = prev_t + (t - prev_t) * (k - prev_n) / max(n - prev_n, 1)
        prev_t, prev_n = t, max(prev_n, n)
    first = min(reach.values()) if reach else 0
    window = [c for t, c, *_ in tele if t >= first - 0.2]
    cmax = max(window) if window else 0
    out = []
    for i, r in enumerate(rows, start=1):
        t = reach.get(i)
        clock = None
        if t is not None:
            j = bisect.bisect_left(tt, t)
            cands = [k for k in (j - 1, j) if 0 <= k < len(tt)]
            k = min(cands, key=lambda k: abs(tt[k] - t))
            if abs(tt[k] - t) <= 0.15:
                clock = tele[k][1]
        if clock is None:
            bucket = 'UNATTRIBUTED'
        elif clock >= 0.95 * cmax:
            bucket = 'HIGH'
        elif clock < 0.5 * cmax:
            bucket = 'LOW'
        else:
            bucket = 'MID'
        out.append({**r, 'sm_clock_mhz': clock, 'clock_bucket': bucket})
    return out, cmax


def summarize(run_dir):
    rows, cmax = attribute(run_dir)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['clock_bucket'], r['variant'], r['component'])].append(r['latency_ms'] * 1000)
    summary = {'max_window_clock_mhz': cmax,
               'bucket_counts': dict(collections.Counter(r['clock_bucket'] for r in rows)),
               'median_us': {f'{b}/{v}/{c}': {'median': round(statistics.median(x), 2), 'n': len(x)}
                             for (b, v, c), x in sorted(groups.items())}}
    Path(run_dir, 'clock_attributed_samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    Path(run_dir, 'clock_attributed_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    for d in sys.argv[1:]:
        s = summarize(d)
        print(d, s['max_window_clock_mhz'], s['bucket_counts'])
        for k, v in s['median_us'].items():
            if '/pipeline' in k or '/context' in k:
                print(' ', k, v)
