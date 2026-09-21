"""Run independently replicated P0.5 A/B measurements with GPU telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

TELEMETRY_FIELDS = [
    "timestamp",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw.average",
    "power.draw.instant",
    "memory.used",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
]
NUMERIC_FIELDS = {
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw.average",
    "power.draw.instant",
    "memory.used",
}


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _median(samples: list[dict[str, Any]], field: str) -> float | None:
    values = [sample[field] for sample in samples if sample[field] is not None]
    return statistics.median(values) if values else None


def _maximum(samples: list[dict[str, Any]], field: str) -> float | None:
    values = [sample[field] for sample in samples if sample[field] is not None]
    return max(values) if values else None


def _coefficient_of_variation(samples: list[dict[str, Any]], field: str) -> float | None:
    values = [sample[field] for sample in samples if sample[field] is not None]
    if len(values) < 2 or statistics.mean(values) == 0.0:
        return None
    return statistics.pstdev(values) / statistics.mean(values)


def _phase_marker(stdout: str, name: str) -> int:
    prefix = f"{name}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return int(line.removeprefix(prefix))
    raise ValueError(f"missing phase marker {name}")


def _artifact_path(stdout: str, name: str, repo: Path) -> Path:
    prefix = f"{name}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            path = Path(line.removeprefix(prefix))
            return path if path.is_absolute() else repo / path
    raise ValueError(f"missing artifact path {name}")


def _competing_compute_processes(pmon: str) -> list[str]:
    processes = []
    for line in pmon.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) >= 4 and columns[2] == "C":
            processes.append(line.strip())
    return processes


def _assess(
    pre: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    pmon: str,
    returncode: int,
) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    pre_gpu_median = _median(pre, "utilization.gpu")
    pre_gpu_max = _maximum(pre, "utilization.gpu")
    pre_memory_median = _median(pre, "utilization.memory")
    pre_memory_max = _maximum(pre, "utilization.memory")
    pre_temp_max = _maximum(pre, "temperature.gpu")
    measured_temp_max = _maximum(measured, "temperature.gpu")
    active_pstates = sum(sample["pstate"] in {"P0", "P1", "P2"} for sample in measured)
    active_pstate_fraction = active_pstates / len(measured) if measured else 0.0
    active_clock_cv = _coefficient_of_variation(
        [sample for sample in measured if sample["pstate"] in {"P0", "P1", "P2"}],
        "clocks.current.sm",
    )
    thermal_fields = [
        "clocks_event_reasons.hw_slowdown",
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.sw_thermal_slowdown",
        "clocks_event_reasons.hw_power_brake_slowdown",
    ]
    thermal_events = [
        {"sampled_at_unix_ms": sample["sampled_at_unix_ms"], "field": field}
        for sample in measured
        for field in thermal_fields
        if sample[field] != "Not Active"
    ]
    competitors = _competing_compute_processes(pmon)

    if returncode != 0:
        reasons.append(f"benchmark return code {returncode}")
    if len(pre) < 4:
        reasons.append(f"only {len(pre)} pre-run telemetry samples")
    if len(measured) < 4:
        reasons.append(f"only {len(measured)} measured-phase telemetry samples")
    if pre_gpu_median is None or pre_gpu_median > 25.0:
        reasons.append(f"pre-run GPU utilization median {pre_gpu_median} exceeds 25%")
    if pre_gpu_max is None or pre_gpu_max > 60.0:
        reasons.append(f"pre-run GPU utilization max {pre_gpu_max} exceeds 60%")
    if pre_memory_median is None or pre_memory_median > 45.0:
        reasons.append(f"pre-run memory utilization median {pre_memory_median} exceeds 45%")
    if pre_memory_max is None or pre_memory_max > 55.0:
        reasons.append(f"pre-run memory utilization max {pre_memory_max} exceeds 55%")
    if pre_temp_max is None or pre_temp_max > 84.0:
        reasons.append(f"pre-run temperature max {pre_temp_max} exceeds 84 C")
    if measured_temp_max is None or measured_temp_max > 87.0:
        reasons.append(f"measured temperature max {measured_temp_max} exceeds 87 C")
    if active_pstate_fraction < 0.75:
        reasons.append(f"active P0-P2 fraction {active_pstate_fraction:.3f} is below 0.75")
    if active_clock_cv is None or active_clock_cv > 0.15:
        reasons.append(f"active SM-clock CV {active_clock_cv} exceeds 0.15")
    if thermal_events:
        reasons.append(f"{len(thermal_events)} thermal/hardware slowdown telemetry events")
    if competitors:
        reasons.append(f"{len(competitors)} competing compute processes before run")

    metrics = {
        "pre_sample_count": len(pre),
        "measured_sample_count": len(measured),
        "pre_gpu_utilization_median_percent": pre_gpu_median,
        "pre_gpu_utilization_max_percent": pre_gpu_max,
        "pre_memory_utilization_median_percent": pre_memory_median,
        "pre_memory_utilization_max_percent": pre_memory_max,
        "pre_temperature_max_c": pre_temp_max,
        "measured_temperature_max_c": measured_temp_max,
        "measured_active_pstate_fraction": active_pstate_fraction,
        "measured_active_sm_clock_cv": active_clock_cv,
        "thermal_events": thermal_events,
        "competing_compute_processes": competitors,
    }
    status = "ACCEPTED" if not reasons else "REJECTED_ENVIRONMENTAL_NOISE"
    return status, reasons, metrics


def _monitor(samples: list[dict[str, Any]], process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        values = next(csv.reader([line]))
        if len(values) != len(TELEMETRY_FIELDS):
            continue
        sample: dict[str, Any] = {
            "sampled_at_unix_ms": time.time_ns() // 1_000_000,
        }
        for field, value in zip(TELEMETRY_FIELDS, values, strict=True):
            stripped = value.strip()
            sample[field] = _float(stripped) if field in NUMERIC_FIELDS else stripped
        samples.append(sample)


def _write_telemetry(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sampled_at_unix_ms", *TELEMETRY_FIELDS])
        writer.writeheader()
        writer.writerows(samples)


def run_attempt(args: argparse.Namespace, attempt: int) -> dict[str, Any]:
    attempt_dir = args.output_dir / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    monitor_command = [
        "nvidia-smi",
        f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
        "--format=csv,noheader,nounits",
        "--loop-ms=50",
    ]
    monitor = subprocess.Popen(
        monitor_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitor_thread = threading.Thread(target=_monitor, args=(samples, monitor), daemon=True)
    monitor_thread.start()
    time.sleep(args.pre_seconds)
    pmon = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    process_start_ms = time.time_ns() // 1_000_000
    env = os.environ.copy()
    env.setdefault("CUDA_TOOLKIT_PATH", "/usr/local/cuda-13.3")
    env.setdefault("LIBCLANG_PATH", "/usr/lib/llvm-18/lib")
    env.setdefault("CUTILE_TILEIRAS_PATH", "/usr/local/cuda-13.3/bin/tileiras")
    command = [
        str(args.binary),
        "--variant",
        "full-latent",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--output-dir",
        str(attempt_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=args.repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    process_end_ms = time.time_ns() // 1_000_000
    time.sleep(0.25)
    monitor.terminate()
    monitor_thread.join(timeout=3.0)
    if monitor.poll() is None:
        monitor.kill()
    monitor_stderr = monitor.stderr.read() if monitor.stderr is not None else ""

    (attempt_dir / "benchmark_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (attempt_dir / "benchmark_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (attempt_dir / "monitor_stderr.txt").write_text(monitor_stderr, encoding="utf-8")
    (attempt_dir / "pmon_before.txt").write_text(pmon, encoding="utf-8")
    _write_telemetry(attempt_dir / "gpu_telemetry.csv", samples)

    gpu_start = _phase_marker(completed.stdout, "GPU_TIMING_PHASE_START_UNIX_MS")
    host_end = _phase_marker(completed.stdout, "HOST_TIMING_PHASE_END_UNIX_MS")
    pre = [
        sample
        for sample in samples
        if process_start_ms - int(args.pre_seconds * 1000) <= sample["sampled_at_unix_ms"]
        < process_start_ms
    ]
    measured = [
        sample for sample in samples if gpu_start <= sample["sampled_at_unix_ms"] <= host_end
    ]
    status, reasons, environmental = _assess(pre, measured, pmon, completed.returncode)
    summary_path = _artifact_path(completed.stdout, "SUMMARY_PATH", args.repo)
    sample_path = _artifact_path(completed.stdout, "SAMPLES_PATH", args.repo)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    full_median = summary["full_kv"]["gpu_execution"]["median_ms"]
    latent_median = summary["latent_original"]["gpu_execution"]["median_ms"]
    record = {
        "attempt": attempt,
        "acceptance_criteria_version": "v2_attainable_compositor_floor",
        "status": status,
        "reasons": reasons,
        "command": command,
        "process_start_unix_ms": process_start_ms,
        "process_end_unix_ms": process_end_ms,
        "gpu_timing_start_unix_ms": gpu_start,
        "host_timing_end_unix_ms": host_end,
        "summary_path": str(summary_path.relative_to(args.repo)),
        "samples_path": str(sample_path.relative_to(args.repo)),
        "environmental": environmental,
        "median_a_ms": full_median,
        "median_b_ms": latent_median,
        "b_over_a": latent_median / full_median,
        "difference_us": (latent_median - full_median) * 1000.0,
    }
    (attempt_dir / "run_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--binary", type=Path, default=Path("target/release/examples/p0_gpu_baseline")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/p05_hardware_attribution")
    )
    parser.add_argument("--accepted-runs", type=int, default=7)
    parser.add_argument("--max-attempts", type=int, default=14)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--pre-seconds", type=float, default=2.5)
    parser.add_argument("--between-seconds", type=float, default=8.0)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    if not args.binary.is_absolute():
        args.binary = args.repo / args.binary
    if not args.output_dir.is_absolute():
        args.output_dir = args.repo / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "replications_manifest.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        for record in records:
            record.setdefault("acceptance_criteria_version", "v1_initial")
    else:
        records = []
    existing_attempts = [
        int(path.name.removeprefix("attempt_"))
        for path in args.output_dir.glob("attempt_*")
        if path.is_dir() and path.name.removeprefix("attempt_").isdigit()
    ]
    first_attempt = max(existing_attempts, default=0) + 1
    for attempt in range(first_attempt, args.max_attempts + 1):
        record = run_attempt(args, attempt)
        records.append(record)
        manifest = {
            "acceptance_criteria": {
                "version": "v2_attainable_compositor_floor",
                "pre_gpu_utilization_median_percent_max": 25.0,
                "pre_gpu_utilization_max_percent_max": 60.0,
                "pre_memory_utilization_median_percent_max": 45.0,
                "pre_memory_utilization_max_percent_max": 55.0,
                "pre_temperature_max_c": 84.0,
                "measured_temperature_max_c": 87.0,
                "measured_p0_p2_fraction_min": 0.75,
                "measured_active_sm_clock_cv_max": 0.15,
                "thermal_or_hardware_slowdown_events": 0,
                "competing_compute_processes": 0,
            },
            "target_accepted_runs": args.accepted_runs,
            "records": records,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"attempt={attempt} status={record['status']} "
            f"A_ms={record['median_a_ms']:.9f} B_ms={record['median_b_ms']:.9f} "
            f"B/A={record['b_over_a']:.6f} reasons={record['reasons']}",
            flush=True,
        )
        if sum(item["status"] == "ACCEPTED" for item in records) >= args.accepted_runs:
            break
        time.sleep(args.between_seconds)

    accepted = sum(record["status"] == "ACCEPTED" for record in records)
    print(f"accepted={accepted} attempted={len(records)} manifest={args.output_dir}")
    if accepted < args.accepted_runs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
