"""P1.5A diagnostic driver: per-kernel scaling measurements with thermal safety.

Runs the p15_kernel_scaling example for each sequence length with the
diagnostic iteration budget, monitoring GPU temperature via nvidia-smi.

Safety policy:
  * If temperature >= 84 C before a length, wait for cooldown (max 360 s);
    if still hot, the length is marked SKIPPED_THERMAL.
  * If temperature >= 86 C during a length, the process is killed and the
    length is marked THERMAL_ABORT.
  * Cooldown between lengths until temperature <= 78 C (max 300 s).
  * Clocks/power limits are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/p15_kernel_scaling"
ENV = {
    **os.environ,
    "CUDA_TOOLKIT_PATH": "/usr/local/cuda-13.3",
    "LIBCLANG_PATH": "/usr/lib/llvm-18/lib",
    "CUTILE_TILEIRAS_PATH": "/usr/local/cuda-13.3/bin/tileiras",
}

# (seq, warmup, measured) — diagnostic budget per the P1.5A protocol.
PLAN = (
    (1024, 10, 50),
    (2048, 10, 30),
    (4096, 5, 20),
    (8192, 3, 10),
    (16384, 1, 5),
    (32768, 1, 1),
)

PRE_ABORT_C = 84.0
DURING_ABORT_C = 86.0
COOLDOWN_TARGET_C = 78.0


def query_gpu() -> dict:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,pstate,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    util, mem_util, temp, pstate, sm_clock, power = [part.strip() for part in out.split(",")]
    procs = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "unix_ms": int(time.time() * 1000),
        "gpu_util_pct": int(util),
        "mem_util_pct": int(mem_util),
        "temperature_c": float(temp),
        "pstate": pstate,
        "sm_clock_mhz": int(sm_clock),
        "power_draw_w": power,
        "compute_processes": procs if procs else "none",
    }


def sample_phase(seconds: float, interval: float = 1.0) -> list[dict]:
    samples = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        samples.append(query_gpu())
        time.sleep(interval)
    return samples


def wait_for_cooldown(target_c: float, timeout_s: float, label: str) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snap = query_gpu()
        print(f"  [{label}] temp={snap['temperature_c']:.0f} C target<={target_c:.0f} C", flush=True)
        if snap["temperature_c"] <= target_c:
            return True
        time.sleep(10)
    return False


def build(seq: int) -> None:
    binary = ROOT / "target/release/examples/p15_kernel_scaling"
    if binary.exists():
        return
    print(f"Building p15_kernel_scaling (once for all lengths)...", flush=True)
    log = OUT / "build_p15.log"
    with open(log, "w") as handle:
        result = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "plkv-kernels",
                "--features",
                "gpu-cutile",
                "--example",
                "p15_kernel_scaling",
            ],
            cwd=ROOT,
            env=ENV,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise SystemExit(f"build failed; see {log}")


def run_length(seq: int, warmup: int, iterations: int) -> dict:
    binary = ROOT / "target/release/examples/p15_kernel_scaling"
    record = {
        "seq": seq,
        "warmup": warmup,
        "iterations": iterations,
        "status": "PENDING",
        "pre_thermal": [],
        "during_thermal": [],
        "post_thermal": [],
        "stdout_tail": "",
        "summary_path": None,
    }

    # Pre-run thermal gate.
    record["pre_thermal"] = sample_phase(3.0)
    pre_max = max(sample["temperature_c"] for sample in record["pre_thermal"])
    if pre_max >= PRE_ABORT_C:
        print(f"  seq={seq}: pre-run temp {pre_max:.0f} C >= {PRE_ABORT_C:.0f} C, cooling down...", flush=True)
        if not wait_for_cooldown(PRE_ABORT_C - 2.0, 360.0, f"pre-cooldown {seq}"):
            record["status"] = "SKIPPED_THERMAL"
            return record
        record["pre_thermal"] = sample_phase(3.0)

    stdout_path = OUT / f"run_seq{seq}.stdout.log"
    print(f"  seq={seq}: launching warmup={warmup} iterations={iterations}", flush=True)
    with open(stdout_path, "w") as stdout_file:
        process = subprocess.Popen(
            [
                str(binary),
                "--seq",
                str(seq),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
                "--output-dir",
                str(OUT),
            ],
            cwd=ROOT,
            env=ENV,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )
        peak_temp = 0.0
        aborted = False
        while process.poll() is None:
            time.sleep(1.5)
            try:
                snap = query_gpu()
            except subprocess.CalledProcessError:
                continue
            record["during_thermal"].append(snap)
            peak_temp = max(peak_temp, snap["temperature_c"])
            if snap["temperature_c"] >= DURING_ABORT_C:
                print(f"  seq={seq}: THERMAL_ABORT at {snap['temperature_c']:.0f} C, killing process", flush=True)
                process.kill()
                aborted = True
                break
        process.wait()

    record["post_thermal"] = sample_phase(3.0)
    record["peak_temperature_c"] = peak_temp
    record["stdout_tail"] = stdout_path.read_text()[-4000:]
    for line in record["stdout_tail"].splitlines():
        if line.startswith("P15_SUMMARY_PATH="):
            record["summary_path"] = line.split("=", 1)[1]

    if aborted:
        record["status"] = "THERMAL_ABORT"
    elif process.returncode != 0:
        record["status"] = f"PROCESS_ERROR_{process.returncode}"
    elif "P15_KERNEL_SCALING_OK=1" in record["stdout_tail"]:
        record["status"] = "OK"
    else:
        record["status"] = "MISSING_COMPLETION_MARKER"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, default=0, help="run only this sequence length")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    build(1024)

    manifest = {
        "policy": {
            "pre_abort_c": PRE_ABORT_C,
            "during_abort_c": DURING_ABORT_C,
            "cooldown_target_c": COOLDOWN_TARGET_C,
            "clocks_or_power_modified": False,
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
        },
        "plan": [
            {"seq": seq, "warmup": warmup, "iterations": iterations}
            for seq, warmup, iterations in PLAN
        ],
        "runs": [],
    }

    for seq, warmup, iterations in PLAN:
        if args.only and seq != args.only:
            continue
        print(f"=== seq={seq} ===", flush=True)
        record = run_length(seq, warmup, iterations)
        manifest["runs"].append(record)
        print(f"  seq={seq}: status={record['status']}", flush=True)
        with open(OUT / "thermal_manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2)
        if record["status"] == "THERMAL_ABORT":
            print("  THERMAL_ABORT recorded; attempting cooldown before continuing.", flush=True)
            wait_for_cooldown(COOLDOWN_TARGET_C, 300.0, f"post-abort cooldown {seq}")
        elif record["status"] == "OK":
            wait_for_cooldown(COOLDOWN_TARGET_C, 300.0, f"cooldown {seq}")

    print(f"Thermal manifest: {OUT / 'thermal_manifest.json'}")


if __name__ == "__main__":
    main()
