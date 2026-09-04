"""Extract the focused P0.5 metrics from Nsight Compute reports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path
from typing import Any

REPORTS = {
    "full_score": "ncu_full_score.ncu-rep",
    "latent_score": "ncu_latent_score.ncu-rep",
    "full_context": "ncu_full_context.ncu-rep",
    "latent_context": "ncu_latent_context.ncu-rep",
}
EXTRA_REPORTS = {role: filename.replace(".ncu-rep", "_extra.ncu-rep") for role, filename in REPORTS.items()}
FOCUSED_METRICS = [
    "device__attribute_l2_cache_size",
    "device__attribute_multiprocessor_count",
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "lts__t_bytes.sum.per_second",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_request_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.per_cycle_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__local_mem_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__waves_per_multiprocessor",
    "sass__inst_executed_register_spilling_mem_local",
    "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_issued.sum",
]


def _number(value: str) -> int | float | str | None:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _raw_report(ncu: Path, report: Path) -> tuple[dict[str, Any], dict[str, str]]:
    completed = subprocess.run(
        [
            str(ncu),
            "--import",
            str(report),
            "--csv",
            "--page",
            "raw",
            "--print-units",
            "base",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    if len(rows) != 3:
        raise ValueError(f"expected header, unit, value rows in {report}; got {len(rows)}")
    header, units, values = rows
    return (
        {name: _number(value) for name, value in zip(header, values, strict=True)},
        {name: unit for name, unit in zip(header, units, strict=True)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda-13.3/bin/ncu"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "ncu": str(args.ncu),
        "metric_note": "Exact Nsight Compute 2026.2.1 metric names; base units.",
        "kernels": {},
    }
    for role, filename in REPORTS.items():
        values, units = _raw_report(args.ncu, args.report_dir / filename)
        extra_values, extra_units = _raw_report(args.ncu, args.report_dir / EXTRA_REPORTS[role])
        for name, value in extra_values.items():
            if name not in values:
                values[name] = value
                units[name] = extra_units[name]
        focused = {
            name: {"value": values[name], "unit": units[name]}
            for name in FOCUSED_METRICS
            if name in values
        }
        stalls = {
            name: {"value": value, "unit": units[name]}
            for name, value in values.items()
            if name.startswith("smsp__average_warps_issue_stalled_")
            and name.endswith("_per_issue_active.ratio")
            and isinstance(value, (int, float))
        }
        ranked_stalls = sorted(
            stalls.items(),
            key=lambda item: float(item[1]["value"]),
            reverse=True,
        )
        dram_read = focused.get("dram__bytes_read.sum", {}).get("value")
        extra_duration_ns = extra_values.get("gpu__time_duration.sum")
        dram_read_per_second = (
            float(dram_read) / (float(extra_duration_ns) * 1.0e-9)
            if isinstance(dram_read, (int, float))
            and isinstance(extra_duration_ns, (int, float))
            else None
        )
        global_load_sectors = focused.get(
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", {}
        ).get("value")
        global_load_requests = focused.get(
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", {}
        ).get("value")
        sectors_per_global_load_request = (
            float(global_load_sectors) / float(global_load_requests)
            if isinstance(global_load_sectors, (int, float))
            and isinstance(global_load_requests, (int, float))
            and global_load_requests
            else None
        )
        result["kernels"][role] = {
            "report": filename,
            "extra_report": EXTRA_REPORTS[role],
            "kernel_name": values["Kernel Name"],
            "focused_metrics": focused,
            "extra_profile_duration_ns": extra_duration_ns,
            "derived_dram_read_bytes_per_second": dram_read_per_second,
            "derived_sectors_per_global_load_request": sectors_per_global_load_request,
            "dominant_warp_stalls": [
                {"metric": name, **metric} for name, metric in ranked_stalls[:5]
            ],
        }

    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
