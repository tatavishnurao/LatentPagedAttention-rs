#!/usr/bin/env python3
"""E3/E4 campaign on a stable-clock host: E3 timing (padded A3 vs C3 fast vs C3 accurate at
N = 1K/8K/32K), the E4 correctness suite once per E4 context length, and E4 timing over the
grid B x N (default B = 1/8/32, N = 1024; labels B{b}_N{n}).

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


BLOCK_SIZE = 16


def int_list(text, name, multiple_of=1):
    """Parses a comma-separated list of positive integers (order kept, duplicates rejected)."""
    try:
        values = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name}: not a comma-separated integer list: {text!r}") from exc
    if not values:
        raise SystemExit(f"{name}: empty list")
    if any(v <= 0 or v % multiple_of for v in values):
        raise SystemExit(f"{name}: values must be positive multiples of {multiple_of}: {values}")
    if len(set(values)) != len(values):
        raise SystemExit(f"{name}: duplicate values: {values}")
    return values


def plan(args):
    """Returns (e4 correctness runs, timing configs) without running anything.

    Correctness: one `--suite e4 --seq N --batches <e4-batches>` process per N, all before any
    timing. Timing configs: (kind, config label, harness args).
    """
    lengths = int_list(args.e3_lengths, "--e3-lengths", BLOCK_SIZE) if args.e3_lengths else []
    batches = int_list(args.e4_batches, "--e4-batches") if args.e4_batches else []
    seqs = int_list(args.e4_seqs, "--e4-seqs", BLOCK_SIZE) if batches else []
    limit = ["--max-footprint-gib", str(args.max_footprint_gib)]
    correctness = []
    if batches and not args.skip_e4_correctness:
        correctness = [
            (
                f"e4_correctness_N{n}",
                f"N{n}",
                [
                    "--suite",
                    "e4",
                    "--seq",
                    str(n),
                    "--batches",
                    ",".join(map(str, batches)),
                    *limit,
                ],
            )
            for n in seqs
        ]
    configs = [("e3_timing", f"N{n}", ["--suite", "timing", "--seq", str(n)]) for n in lengths]
    configs += [
        (
            "e4_timing",
            f"B{b}_N{n}",
            ["--suite", "e4-timing", "--batch", str(b), "--seq", str(n), *limit],
        )
        for n in seqs
        for b in batches
    ]
    return correctness, configs


class HostCheckFailed(Exception):
    pass


def host_check(command):
    """Runs the --host-check command; returns its output, raises HostCheckFailed on non-zero."""
    if not command:
        return None
    r = subprocess.run(command, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise HostCheckFailed(out or f"host check exited {r.returncode}")
    return out


def run_logged(args, root, label, cmd):
    """Runs one harness process with telemetry and the thermal watcher; returns a run record.
    The host check runs after the cooldown, immediately before the process starts."""
    condition, start_temp, waited = cooldown(args.cool_c, args.cooldown_s)
    checked = host_check(args.host_check)
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
        "host_check": checked,
        "cmd": cmd,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--e3-lengths", default="1024,8192,32768")
    parser.add_argument("--e4-batches", default="1,8,32")
    parser.add_argument("--e4-seqs", default="1024", help="E4 context lengths N (default 1024)")
    parser.add_argument("--max-footprint-gib", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-cool", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--cool-c", type=int, default=80)
    parser.add_argument("--cooldown-s", type=int, default=300)
    parser.add_argument("--abort-c", type=int, default=91)
    parser.add_argument("--skip-e4-correctness", action="store_true")
    parser.add_argument(
        "--host-check",
        default="",
        help="shell command run before every process; non-zero stops the campaign",
    )
    parser.add_argument(
        "--print-plan", action="store_true", help="print the commands and exit (no GPU)"
    )
    args = parser.parse_args()
    correctness, configs = plan(args)
    if args.print_plan:
        print(json.dumps({"e4_correctness": correctness, "timing": configs}, indent=1))
        return

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
    try:
        run_all(args, root, manifest, save, correctness, configs, common)
    except HostCheckFailed as exc:
        manifest["host_check_failed"] = str(exc)
        save()
        print(f"HOST_CHECK_FAILED: stopping campaign before the next process ({exc})", flush=True)
    print("E3E4_CAMPAIGN_DIR", root, flush=True)


def run_all(args, root, manifest, save, correctness, configs, common):
    # Every E4 context length passes correctness before any timing process starts.
    for label, config, suite_args in correctness:
        run = run_logged(args, root, label, [str(BIN), *suite_args])
        run.update({"kind": "e4_correctness", "config": config})
        manifest["runs"].append(run)
        save()
        print(json.dumps({k: v for k, v in run.items() if k != "cmd"}), flush=True)
        if run["returncode"] != 0:
            raise SystemExit(f"E4 correctness suite failed ({label}); not timing")

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


if __name__ == "__main__":
    main()
