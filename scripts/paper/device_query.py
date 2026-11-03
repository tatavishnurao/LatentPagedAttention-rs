#!/usr/bin/env python3
"""Device facts reported by the CUDA driver (cuDeviceGetAttribute), saved raw.

Loads libcuda.so.1 with ctypes, calls cuInit and reads device attributes. No CUDA context is
created and no kernel is launched. The peak DRAM bandwidth is COMPUTED from two reported values,
2 x memory clock x bus width / 8 (double data rate), and is a theoretical peak, not a measurement.

Attribute enum values and units are those of the installed cuda.h (CUDA 13.3):
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16   number of multiprocessors
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13             typical clock frequency, kHz
  CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36      peak memory clock frequency, kHz
  CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37 global memory bus width, bits
  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38          L2 cache size, bytes
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR = 75/76

usage: device_query.py --out reports/device_query_<date>/device_query.json
"""

import argparse
import ctypes
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

ATTRS = {
    "MULTIPROCESSOR_COUNT": (16, "count"),
    "CLOCK_RATE": (13, "kHz"),
    "MEMORY_CLOCK_RATE": (36, "kHz"),
    "GLOBAL_MEMORY_BUS_WIDTH": (37, "bits"),
    "L2_CACHE_SIZE": (38, "bytes"),
    "COMPUTE_CAPABILITY_MAJOR": (75, ""),
    "COMPUTE_CAPABILITY_MINOR": (76, ""),
}


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} returned CUresult {rc}")


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cu = ctypes.CDLL("libcuda.so.1")
    check(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    name = ctypes.create_string_buffer(256)
    check(cu.cuDeviceGetName(name, 256, dev), "cuDeviceGetName")
    drv = ctypes.c_int()
    check(cu.cuDriverGetVersion(ctypes.byref(drv)), "cuDriverGetVersion")
    total = ctypes.c_size_t()
    check(cu.cuDeviceTotalMem_v2(ctypes.byref(total), dev), "cuDeviceTotalMem_v2")
    raw = {}
    for key, (enum, unit) in ATTRS.items():
        v = ctypes.c_int()
        check(cu.cuDeviceGetAttribute(ctypes.byref(v), enum, dev), key)
        raw[f"CU_DEVICE_ATTRIBUTE_{key}"] = {"enum": enum, "value": v.value, "unit": unit}
    mem_khz = raw["CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE"]["value"]
    bus_bits = raw["CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH"]["value"]
    l2 = raw["CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"]["value"]
    peak = 2 * mem_khz * 1e3 * bus_bits / 8
    out = {
        "what": "device attributes reported by the CUDA driver; no context, no kernel",
        "host_label": "laptop RTX 4060, WSL2, not a cloud result",
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": {"uname": " ".join(platform.uname())},
        "device": name.value.decode(),
        "cuda_driver_api_version": drv.value,
        "total_mem_bytes": total.value,
        "attributes": raw,
        "derived": {
            "l2_cache_mib": l2 / (1 << 20),
            "peak_dram_bandwidth_formula": "2 * MEMORY_CLOCK_RATE[kHz] * 1e3 * BUS_WIDTH[bits] / 8",
            "peak_dram_bandwidth_bytes_per_s": peak,
            "peak_dram_bandwidth_gb_per_s": peak / 1e9,
            "note": "theoretical peak computed from reported values, not a measured bandwidth",
        },
        "git_commit": git_commit(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["value"] for k, v in raw.items()}))
    print(json.dumps(out["derived"]))
    print(f"DEVICE_QUERY_OK {a.out}")


if __name__ == "__main__":
    main()
