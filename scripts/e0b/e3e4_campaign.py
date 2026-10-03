#!/usr/bin/env python3
"""E3/E4 campaign on a stable-clock host: E3 timing (padded A3 vs C3 fast vs C3 accurate at
N = 1K/8K/32K), the E4 correctness suite once, and E4 timing (B = 1/8/32).

Each timing configuration collects >= --min-cool cool-start processes (same cooldown, 100 ms
external telemetry and 91 C abort as e0b_campaign.py); hot-start processes are run, tagged and
excluded from the cool-start count. Standard library only.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from e0b_campaign import FIELDS, ROOT, cooldown, gpu_temp

BIN = Path(
    os.environ.get("E3_SPLITK_BIN")
    or Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release/examples/e3_splitk_gpu"
)


def run_logged(args, root, label, cmd):
    """Runs one harness process with telemetry and the thermal watcher; returns a run record."""
    condition, start_temp, waited = cooldown(args.cool_c, args.cooldown_s)
    tele = (root / f"{label}_telemetry.csv").open("w")
    logger = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=tele,
        stderr=subprocess.STDOUT,
    )
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
        "start_condition": condition,
        "start_temp_c": start_temp,
        "cooldown_wait_s": waited,
        "returncode": rc,
        "external_thermal_kill": killed["value"],
        "wall_s": round(time.time() - t0, 2),
        "cmd": cmd,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--e3-lengths", default="1024,8192,32768")
    parser.add_argument("--e4-batches", default="1,8,32")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-cool", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--cool-c", type=int, default=80)
    parser.add_argument("--cooldown-s", type=int, default=300)
    parser.add_argument("--abort-c", type=int, default=91)
    parser.add_argument("--skip-e4-correctness", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir).resolve()
    if str(root).startswith("/mnt/"):
        raise SystemExit(f"refusing /mnt output path: {root}")
    root.mkdir(parents=True, exist_ok=False)
    sha = subprocess.run(["sha256sum", str(BIN)], capture_output=True, text=True, check=True)
    manifest = {
        "binary_sha256": sha.stdout.split()[0],
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "args": vars(args),
        "runs": [],
    }

    def save():
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    common = ["--iterations", str(args.iterations), "--warmup", str(args.warmup)]
    configs = [
        ("e3_timing", f"N{n}", ["--suite", "timing", "--seq", n])
        for n in filter(None, args.e3_lengths.split(","))
    ] + [
        ("e4_timing", f"B{b}", ["--suite", "e4-timing", "--batch", b])
        for b in filter(None, args.e4_batches.split(","))
    ]

    if not args.skip_e4_correctness:
        label = "e4_correctness"
        run = run_logged(args, root, label, [str(BIN), "--suite", "e4"])
        run.update({"kind": "e4_correctness", "config": "all"})
        manifest["runs"].append(run)
        save()
        print(json.dumps({k: v for k, v in run.items() if k != "cmd"}), flush=True)
        if run["returncode"] != 0:
            raise SystemExit("E4 correctness suite failed; not timing")

    for kind, config, suite_args in configs:
        cool, attempts, process = 0, 0, 0
        while cool < args.min_cool and attempts < args.max_attempts:
            label = f"{kind}_{config}_p{process}"
            cmd = [
                str(BIN),
                *suite_args,
                "--process",
                str(process),
                *common,
                "--output-dir",
                str(root / label),
            ]
            run = run_logged(args, root, label, cmd)
            run.update({"kind": kind, "config": config, "process": process})
            attempts += 1
            process += 1
            if run["returncode"] == 0 and run["start_condition"] == "cool-start":
                cool += 1
            manifest["runs"].append(run)
            save()
            print(json.dumps({k: v for k, v in run.items() if k != "cmd"}), flush=True)
            if run["external_thermal_kill"] or run["returncode"] == 3:
                print("THERMAL ABORT: stopping campaign", flush=True)
                return
    print("E3E4_CAMPAIGN_DIR", root, flush=True)


if __name__ == "__main__":
    main()
