#!/usr/bin/env python3
"""Time NVML host calls (a separate micro-measurement, not campaign data).

Calls each NVML query REPS times in a loop from the CPU and records the wall time of each
call. No kernel is launched and no CUDA context is created; this measures only the latency
of the driver call that the E3/E4 harness made after every timed sample before fix 1
(`nvmlDeviceGetCurrentClocksEventReasons`), against the cheaper reads it also makes.

Usage: python3 scripts/paper/nvml_latency.py --out reports/nvml_latency_<date>/nvml_latency.json
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

REPS = 50
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_GPU = 0


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} returned NVML error {rc}")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _summary(ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": round(statistics.median(ms), 4),
        "min_ms": round(min(ms), 4),
        "max_ms": round(max(ms), 4),
        "mean_ms": round(statistics.fmean(ms), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()

    lib = ctypes.CDLL("libnvidia-ml.so.1")
    _check(lib.nvmlInit_v2(), "nvmlInit_v2")
    dev = ctypes.c_void_p()
    _check(lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)), "GetHandleByIndex")

    name = ctypes.create_string_buffer(96)
    _check(lib.nvmlDeviceGetName(dev, name, 96), "GetName")
    drv = ctypes.create_string_buffer(80)
    _check(lib.nvmlSystemGetDriverVersion(drv, 80), "GetDriverVersion")

    u32 = ctypes.c_uint()
    u64 = ctypes.c_ulonglong()

    def temperature() -> int:
        _check(lib.nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, ctypes.byref(u32)), "temp")
        return u32.value

    def sm_clock() -> int:
        _check(lib.nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, ctypes.byref(u32)), "clock")
        return u32.value

    reasons_fn = getattr(lib, "nvmlDeviceGetCurrentClocksEventReasons", None)
    reasons_name = "nvmlDeviceGetCurrentClocksEventReasons"
    if reasons_fn is None:  # older drivers
        reasons_fn = lib.nvmlDeviceGetCurrentClocksThrottleReasons
        reasons_name = "nvmlDeviceGetCurrentClocksThrottleReasons"

    calls = {
        reasons_name: lambda: reasons_fn(dev, ctypes.byref(u64)),
        "nvmlDeviceGetClockInfo(SM)": lambda: lib.nvmlDeviceGetClockInfo(
            dev, NVML_CLOCK_SM, ctypes.byref(u32)
        ),
        "nvmlDeviceGetClockInfo(MEM)": lambda: lib.nvmlDeviceGetClockInfo(
            dev, NVML_CLOCK_MEM, ctypes.byref(u32)
        ),
        "nvmlDeviceGetTemperature": lambda: lib.nvmlDeviceGetTemperature(
            dev, NVML_TEMPERATURE_GPU, ctypes.byref(u32)
        ),
    }

    started = dt.datetime.now(dt.timezone.utc)
    temp_before, clock_before = temperature(), sm_clock()
    results = {}
    for label, fn in calls.items():
        per_call = []
        for _ in range(args.reps):
            t0 = time.perf_counter_ns()
            rc = fn()
            t1 = time.perf_counter_ns()
            _check(rc, label)
            per_call.append((t1 - t0) / 1e6)
        results[label] = {**_summary(per_call), "per_call_ms": [round(x, 4) for x in per_call]}
    temp_after, clock_after = temperature(), sm_clock()
    lib.nvmlShutdown()

    out = {
        "what": "NVML host-call latency micro-measurement; separate from the E3/E4 campaign, "
        "not campaign data. No kernel launched, no CUDA context.",
        "host_label": "laptop RTX 4060, WSL2, not a cloud result",
        "date_utc": started.isoformat(timespec="seconds"),
        "host": {
            "uname": " ".join(platform.uname()),
            "gpu": name.value.decode(),
            "driver": drv.value.decode(),
        },
        "gpu_temperature_c": {"before": temp_before, "after": temp_after},
        "sm_clock_mhz": {"before": clock_before, "after": clock_after},
        "reps_per_call": args.reps,
        "timer": "time.perf_counter_ns around one ctypes call, calls run back to back per function",
        "git_commit": _git_commit(),
        "calls": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for label, r in results.items():
        print(f"{label:48s} median {r['median_ms']:.4f} ms  max {r['max_ms']:.4f} ms")
    print(f"temperature {temp_before} -> {temp_after} C; wrote {args.out}")


if __name__ == "__main__":
    main()
