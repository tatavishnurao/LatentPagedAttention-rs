#!/usr/bin/env python3
"""Portable E0b campaign: A1 vs C1 decode pipelines at 1K/8K/32K on a native-Linux GPU host.

Collects >= --min-cool cool-start processes per length (cooldown to < --cool-c, at most
--cooldown-s seconds; a process that starts hotter is still run but tagged hot-start and does
not count toward the target). Lengths are interleaved per round. Every process records
per-sample SM/memory clocks in-process (NVML) and 100 ms external telemetry.

Thermal policy (2026-10-02): hard abort at >= --abort-c (default 91 C) both in-process and
from this driver; THROTTLED tagging happens in analyze_e0b.py.

Standard library only.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(
    os.environ.get("C1_RTABLE_BIN")
    or Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release/examples/c1_rtable"
)
FIELDS = "timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active"
ITERATIONS = {1024: 200, 8192: 60, 32768: 30}


def gpu_temp():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def cooldown(limit_c, max_s):
    start = time.time()
    while (t := gpu_temp()) >= limit_c:
        if time.time() - start >= max_s:
            return "hot-start", t, round(time.time() - start, 1)
        time.sleep(5)
    return "cool-start", t, round(time.time() - start, 1)


def run_process(args, root, seq, process):
    label = f"seq{seq}_p{process}"
    condition, start_temp, waited = cooldown(args.cool_c, args.cooldown_s)
    tele = (root / f"{label}_telemetry.csv").open("w")
    logger = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=tele,
        stderr=subprocess.STDOUT,
    )
    cmd = [
        str(BIN),
        "--seq",
        str(seq),
        "--process",
        str(process),
        "--variants",
        "A1,C1",
        "--warmup",
        "10",
        "--iterations",
        str(ITERATIONS[seq]),
        "--output-dir",
        str(root / label),
    ]
    if process > 0:
        cmd.append("--skip-oracle-export")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=(root / f"{label}_stdout.log").open("w"), stderr=subprocess.STDOUT
    )
    killed = {"value": False}

    def watch():
        while proc.poll() is None:
            try:
                if gpu_temp() >= args.abort_c:
                    killed["value"] = True
                    proc.kill()
            except (subprocess.CalledProcessError, ValueError):
                pass
            time.sleep(0.5)

    watcher = threading.Thread(target=watch)
    watcher.start()
    rc = proc.wait()
    watcher.join()
    logger.terminate()
    logger.wait()
    tele.close()
    return {
        "label": label,
        "seq": seq,
        "process": process,
        "start_condition": condition,
        "start_temp_c": start_temp,
        "cooldown_wait_s": waited,
        "returncode": rc,
        "external_thermal_kill": killed["value"],
        "wall_s": round(time.time() - t0, 2),
        "start_unix": t0,
        "cmd": cmd,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lengths", default="1024,8192,32768")
    parser.add_argument("--min-cool", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=8, help="per length")
    parser.add_argument("--cool-c", type=int, default=80)
    parser.add_argument("--cooldown-s", type=int, default=300)
    parser.add_argument("--abort-c", type=int, default=91)
    args = parser.parse_args()

    root = Path(args.out_dir).resolve()
    if str(root).startswith("/mnt/"):
        raise SystemExit(f"refusing /mnt output path: {root}")
    root.mkdir(parents=True, exist_ok=False)
    lengths = [int(x) for x in args.lengths.split(",")]
    sha = subprocess.run(["sha256sum", str(BIN)], capture_output=True, text=True, check=True)
    manifest = {
        "binary_sha256": sha.stdout.split()[0],
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "args": vars(args),
        "iterations": ITERATIONS,
        "runs": [],
        "env": {k: os.environ.get(k) for k in ("CUDA_TOOLKIT_PATH", "CUTILE_TILEIRAS_PATH")},
    }
    cool = dict.fromkeys(lengths, 0)
    attempts = dict.fromkeys(lengths, 0)
    process = 0
    while any(cool[n] < args.min_cool and attempts[n] < args.max_attempts for n in lengths):
        for seq in lengths:
            if cool[seq] >= args.min_cool or attempts[seq] >= args.max_attempts:
                continue
            run = run_process(args, root, seq, process)
            attempts[seq] += 1
            if run["returncode"] == 0 and run["start_condition"] == "cool-start":
                cool[seq] += 1
            manifest["runs"].append(run)
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(json.dumps({k: v for k, v in run.items() if k != "cmd"}), flush=True)
            if run["external_thermal_kill"] or run["returncode"] == 3:
                print("THERMAL ABORT: stopping campaign", flush=True)
                return
        process += 1
    manifest["cool_start_counts"] = cool
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("CAMPAIGN_DIR", root, "cool_start_counts", cool, flush=True)


if __name__ == "__main__":
    main()
