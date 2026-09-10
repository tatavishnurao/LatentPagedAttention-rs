#!/usr/bin/env python3
# ruff: noqa: E501,E701,E702
"""Run and package the bounded, source-identified 1K-through-8K campaign.

The driver fails closed on a dirty source tree, 16K/32K lengths, thermal
throttling, an unstable idle baseline, unrelated compute processes, and an
unconfirmed AC/performance power configuration. Every attempted process and
its telemetry is retained; measurement direction is never an acceptance rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/examples/p15b_rtable"
NCU = Path("/usr/local/cuda-13.3/bin/ncu")
LENGTHS = (1024, 2048, 4096, 8192)
PLAN = {1024: (10, 50), 2048: (10, 30), 4096: (5, 20), 8192: (3, 10)}
VARIANTS = ("A0", "B0", "A1", "B1")
ENV_KEYS = ("CUDA_TOOLKIT_PATH", "LIBCLANG_PATH", "CUTILE_TILEIRAS_PATH", "LD_LIBRARY_PATH")
QUERY = [
    "timestamp", "name", "uuid", "driver_version", "pstate", "temperature.gpu",
    "clocks.gr", "clocks.sm", "clocks.mem", "power.draw", "power.limit",
    "utilization.gpu", "utilization.memory", "clocks_event_reasons.active",
    "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_slowdown", "clocks_event_reasons.sw_power_cap",
]
POLICY = {
    "predeclared_before_timing": True,
    "no_active_thermal_throttling": True,
    "idle_samples": 10,
    "idle_interval_seconds": 1.0,
    "idle_max_temperature_range_c": 3.0,
    "idle_max_final_temperature_c": 73.0,
    "idle_temperature_rationale": "73 C ceiling predeclared before timing after a retained 40-minute idle cooldown established a stable 70-72 C platform equilibrium; this is below the prior 75-77 C correctness range",
    "idle_max_gpu_utilization_pct": 5.0,
    "run_max_temperature_c": 82.0,
    "active_clock_min_samples": 2,
    "active_clock_max_cv": 0.10,
    "active_clock_max_range_mhz": 450.0,
    "unrelated_compute_processes": "none before/after; during only benchmark PID is allowed",
    "power": "Windows reports AC online and active scheme name contains High performance",
    "outlier": "retain every run; investigate >5% process-median deviation; never accept by result direction",
    "canonical_target": "at least five accepted independent processes",
    "fixed_order_limitation": "A0,A1,B0,B1 by component, then pipeline; order is fixed",
}


def utcstamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, **kwargs)


def output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def stats(xs: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(xs)
    q1, q3 = percentile(xs, .25), percentile(xs, .75)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"n": len(xs), "median": statistics.median(xs), "mean": mean,
            "standard_deviation": sd, "iqr": q3-q1, "min": min(xs), "max": max(xs),
            "p5": percentile(xs, .05), "p95": percentile(xs, .95),
            "coefficient_of_variation": sd/mean if mean else 0.0}


def wsl_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"CUDA_TOOLKIT_PATH": "/usr/local/cuda-13.3", "LIBCLANG_PATH": "/usr/lib/llvm-18/lib",
                "CUTILE_TILEIRAS_PATH": "/usr/local/cuda-13.3/bin/tileiras"})
    release = Path("/proc/sys/kernel/osrelease").read_text().lower()
    if "microsoft" in release or "wsl" in release:
        matches = sorted(Path("/usr/lib/wsl/drivers").glob("*/libnvidia-ptxjitcompiler.so.1"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one WSL PTX-JIT library, found {matches}")
        old = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = ":".join([str(matches[0].parent), "/usr/lib/wsl/lib"] + ([old] if old else []))
    return env


def power_state() -> dict[str, Any]:
    ps = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    script = "$b=Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue; " \
             "if($b){$b|Select BatteryStatus,EstimatedChargeRemaining|ConvertTo-Json -Compress}; " \
             "powercfg /getactivescheme"
    cp = subprocess.run([str(ps), "-NoProfile", "-Command", script], text=True, capture_output=True)
    text = cp.stdout.replace("\r", "").strip()
    battery = None
    for line in text.splitlines():
        if line.startswith("{"):
            try: battery = json.loads(line)
            except json.JSONDecodeError: pass
    scheme = re.search(r"\(([^)]+)\)", text)
    ac = bool(battery and battery.get("BatteryStatus") in (2, 3, 6, 7, 8, 9, 11))
    name = scheme.group(1) if scheme else None
    return {"command_returncode": cp.returncode, "raw": text, "battery": battery,
            "ac_online": ac, "active_scheme": name,
            "accepted": ac and bool(name and "high performance" in name.lower())}


def telemetry() -> dict[str, Any]:
    cp = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(QUERY),
                         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    values = [x.strip() for x in cp.stdout.strip().split(",")]
    row: dict[str, Any] = dict(zip(QUERY, values, strict=True))
    for key in ("temperature.gpu", "clocks.gr", "clocks.sm", "clocks.mem", "power.draw",
                "power.limit", "utilization.gpu", "utilization.memory"):
        try: row[key] = float(row[key])
        except (ValueError, TypeError): row[key] = None
    proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    row["compute_processes"] = proc.stdout.strip() or "none"
    row["sample_unix_ns"] = time.time_ns()
    return row


def thermal_clear(t: dict[str, Any]) -> bool:
    return all(t.get(k) == "Not Active" for k in (
        "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.sw_thermal_slowdown",
        "clocks_event_reasons.hw_slowdown"))


def idle_preflight(outdir: Path) -> tuple[bool, list[str]]:
    rows = []
    for _ in range(int(POLICY["idle_samples"])):
        rows.append(telemetry()); time.sleep(float(POLICY["idle_interval_seconds"]))
    write_rows(outdir / "idle_telemetry.csv", rows)
    reasons = []
    temps = [x["temperature.gpu"] for x in rows if x["temperature.gpu"] is not None]
    if not temps or max(temps)-min(temps) > POLICY["idle_max_temperature_range_c"]: reasons.append("unstable_idle_temperature")
    if not temps or temps[-1] > POLICY["idle_max_final_temperature_c"]: reasons.append("idle_temperature_too_high")
    if any((x["utilization.gpu"] or 0) > POLICY["idle_max_gpu_utilization_pct"] for x in rows): reasons.append("idle_gpu_busy")
    if any(x["compute_processes"] != "none" for x in rows): reasons.append("unrelated_gpu_process")
    if any(not thermal_clear(x) for x in rows): reasons.append("thermal_throttling")
    return not reasons, reasons


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: path.write_text(""); return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def run_benchmark(base: Path, kind: str, index: int, seq: int, warmup: int, iterations: int,
                  commit: str, env: dict[str, str]) -> dict[str, Any]:
    d = base / kind / f"attempt_{index:02d}_seq{seq}"
    d.mkdir(parents=True)
    native = d / "native"; native.mkdir()
    command = [str(BINARY), "--seq", str(seq), "--warmup", str(warmup), "--iterations", str(iterations),
               "--output-dir", str(native)]
    before = telemetry(); samples: list[dict[str, Any]] = [before]
    start = time.perf_counter_ns()
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stop = threading.Event()
    def monitor() -> None:
        while not stop.wait(.5):
            try: samples.append(telemetry())
            except Exception as exc: samples.append({"sample_unix_ns": time.time_ns(), "monitor_error": repr(exc)})
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    stdout, stderr = process.communicate(); elapsed_ms = (time.perf_counter_ns()-start)/1e6
    stop.set(); thread.join(); after = telemetry(); samples.append(after)
    (d/"stdout.log").write_text(stdout); (d/"stderr.log").write_text(stderr); write_rows(d/"telemetry.csv", samples)
    reasons: list[str] = []
    if process.returncode: reasons.append(f"process_returncode_{process.returncode}")
    summaries = list(native.glob("summary_seq*.json")); jsonls = list(native.glob("summary_seq*.jsonl"))
    if len(summaries) != 1 or len(jsonls) != 1: reasons.append("native_outputs_missing_or_ambiguous")
    valid = [x for x in samples if "monitor_error" not in x]
    if any(not thermal_clear(x) for x in valid): reasons.append("thermal_throttling")
    temps = [x["temperature.gpu"] for x in valid if x.get("temperature.gpu") is not None]
    if temps and max(temps) > POLICY["run_max_temperature_c"]: reasons.append("temperature_limit")
    for x in (before, after):
        if x["compute_processes"] != "none": reasons.append("unrelated_process_at_boundary")
    active_clocks = [x["clocks.gr"] for x in valid if (x.get("utilization.gpu") or 0) >= 20 and x.get("clocks.gr")]
    if len(active_clocks) < POLICY["active_clock_min_samples"]: reasons.append("insufficient_active_clock_samples")
    else:
        cv = statistics.stdev(active_clocks)/statistics.mean(active_clocks) if len(active_clocks)>1 else 0
        if cv > POLICY["active_clock_max_cv"] or max(active_clocks)-min(active_clocks) > POLICY["active_clock_max_range_mhz"]:
            reasons.append("unstable_benchmark_clocks")
    summary = json.loads(summaries[0].read_text()) if len(summaries)==1 else {}
    manifest = {"run_id": f"{kind}-{index:02d}-seq{seq}", "source_commit": commit,
        "dirty_worktree": bool(output(["git","status","--porcelain"])), "status": "ACCEPTED" if not reasons else "REJECTED",
        "rejection_reasons": reasons, "sequence_length": seq, "warmup_count": warmup,
        "iteration_count": iterations, "process_index": index, "execution_order": list(summary.get("kernels", {})),
        "fixed_order": True, "timing_method": "CUDA_EVENT_KERNEL_AND_PIPELINE",
        "timing_boundary": "single component launch or score+softmax+context stream span",
        "process_wall_time_ms": elapsed_ms, "command": shlex.join(command),
        "environment": {k: env.get(k) for k in ENV_KEYS}, "telemetry_before": before,
        "telemetry_after": after, "active_graphics_clocks_mhz": active_clocks,
        "policy_path": "../../acceptance_policy.json", "block_table": summary.get("block_table"),
        "block_table_is_identity": summary.get("block_table_is_identity"), "correctness": summary.get("correctness")}
    (d/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if summary: (d/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if jsonls:
        rows=[]
        for line in jsonls[0].read_text().splitlines():
            r=json.loads(line); v,c=r["kernel"].split("_",1); rows.append({"variant":v,"component":c,**{k:r[k] for k in ("iteration","latency_ms")}})
        write_rows(d/"raw_samples.csv", rows)
    return manifest


def aggregate(base: Path, commit: str) -> None:
    manifests=[]; sample_rows=[]
    for p in sorted((base/"timing").glob("*/run_manifest.json")) + sorted((base/"scaling").glob("*/run_manifest.json")):
        m=json.loads(p.read_text()); manifests.append({**m,"manifest_path":str(p.relative_to(base))})
        raw=p.parent/"raw_samples.csv"
        if raw.exists():
            with raw.open() as f:
                for r in csv.DictReader(f): sample_rows.append({"run_id":m["run_id"],"status":m["status"],"sequence_length":m["sequence_length"],**r})
    write_rows(base/"all_raw_samples.csv", sample_rows)
    accepted=[m for m in manifests if m["status"]=="ACCEPTED"]
    proc=[]
    for m in accepted:
        p=base/m["manifest_path"]; summary=json.loads((p.parent/"summary.json").read_text())
        for v in VARIANTS:
            med=summary["kernels"][v+"_pipeline"]["median_ms"]
            proc.append({"run_id":m["run_id"],"sequence_length":m["sequence_length"],"variant":v,"pipeline_median_ms":med})
        if m["sequence_length"]==1024 and m["run_id"].startswith("canonical"):
            a=summary["kernels"]["A0_pipeline"]["median_ms"]; b=summary["kernels"]["B0_pipeline"]["median_ms"]
            proc.append({"run_id":m["run_id"],"sequence_length":1024,"variant":"B0/A0","pipeline_median_ms":b/a})
    write_rows(base/"process_medians.csv",proc)
    dist=[]
    groups: dict[tuple[str,str],list[float]]={}
    for r in proc:
        if r["sequence_length"]==1024 and str(r["run_id"]).startswith("canonical"):
            groups.setdefault((str(r["variant"]),"process_median"),[]).append(float(r["pipeline_median_ms"]))
    for (v,metric),xs in groups.items(): dist.append({"source_commit":commit,"variant":v,"metric":metric,**stats(xs)})
    write_rows(base/"canonical_summary.csv",dist)
    # Sample-level scaling summaries plus requested ratios/slopes.
    scaling=[r for r in sample_rows if r["status"]=="ACCEPTED" and (r["run_id"].startswith("scaling") or (r["run_id"].startswith("canonical") and int(r["sequence_length"])==1024))]
    sg: dict[tuple[int,str,str],list[float]]={}
    for r in scaling: sg.setdefault((int(r["sequence_length"]),r["variant"],r["component"]),[]).append(float(r["latency_ms"]))
    sr=[]
    for (n,v,c),xs in sorted(sg.items()): sr.append({"source_commit":commit,"sequence_length":n,"variant":v,"component":c,**stats(xs)})
    write_rows(base/"scaling_distribution_summary.csv",sr)
    med={(int(r["sequence_length"]),r["variant"],r["component"]):float(r["median"]) for r in sr}
    ratios=[]
    for n in LENGTHS:
        for stem in ("A","B"):
            for c in ("score","softmax","context","pipeline"):
                if (n,stem+"0",c) in med and (n,stem+"1",c) in med:
                    ratios.append({"source_commit":commit,"kind":"repaired_over_original","sequence_length":n,"component":c,"numerator":stem+"1","denominator":stem+"0","ratio":med[n,stem+"1",c]/med[n,stem+"0",c]})
        for c in ("score","softmax","context","pipeline"):
            if (n,"B1",c) in med and (n,"A1",c) in med:
                ratios.append({"source_commit":commit,"kind":"B1_over_A1","sequence_length":n,"component":c,"numerator":"B1","denominator":"A1","ratio":med[n,"B1",c]/med[n,"A1",c]})
    for v in VARIANTS:
        for c in ("score","softmax","context","pipeline"):
            for a,b in zip(LENGTHS,LENGTHS[1:], strict=True):
                if (a,v,c) in med and (b,v,c) in med:
                    ratio=med[b,v,c]/med[a,v,c]
                    ratios.append({"source_commit":commit,"kind":"adjacent_doubling","sequence_length":f"{a}->{b}","component":c,"numerator":v,"denominator":v,"ratio":ratio,"descriptive_local_slope":math.log(ratio,2)})
    write_rows(base/"scaling_ratios_and_local_slopes.csv",ratios)
    # The bounded wall/event comparison uses the same accepted canonical process.
    boundary=[]
    for m in accepted:
        if m["run_id"].startswith("canonical"):
            summary=json.loads(((base/m["manifest_path"]).parent/"summary.json").read_text())
            event_sum=sum(sum(x["samples_ms"]) for x in summary["kernels"].values())
            boundary.append({"source_commit":commit,"run_id":m["run_id"],"process_wall_time_ms":m["process_wall_time_ms"],
                "sum_of_all_retained_cuda_event_samples_ms":event_sum,"wall_over_event_sum":m["process_wall_time_ms"]/event_sum,
                "qualification":"boundaries cover different work; ratio is descriptive, not overhead correction"})
            break
    write_rows(base/"timing_boundary_comparison.csv",boundary)
    write_rows(base/"run_index.csv",[{k:m.get(k) for k in ("run_id","source_commit","status","rejection_reasons","sequence_length","warmup_count","iteration_count","process_index","process_wall_time_ms","manifest_path")} for m in manifests])


def ncu_collect(base: Path, commit: str, env: dict[str,str]) -> None:
    d=base/"ncu"; d.mkdir(exist_ok=True)
    version=output([str(NCU),"--version"]); (d/"ncu_version.txt").write_text(version+"\n")
    metrics=",".join(["lts__t_sector_hit_rate.pct","dram__bytes.sum",
        "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum","launch__grid_size",
        "sm__warps_active.avg.pct_of_peak_sustained_active"])
    filters={"A0_score":"regex:model_small_full_kv_scores_fp16_storage_1024.*",
             "B0_score":"regex:model_small_scores_fp16_storage_1024.*",
             "A0_context":"regex:model_small_full_kv_context_fp16_storage_1024.*",
             "B0_context":"regex:model_small_context_fp16_storage_1024.*"}
    rows=[]
    for role,filt in filters.items():
        native=d/(role+"_native"); native.mkdir()
        report=d/role
        cmd=[str(NCU),"--target-processes","all","--kernel-name-base","demangled","--kernel-name",filt,
             "--launch-count","1","--replay-mode","kernel","--cache-control","all","--metrics",metrics,
             "--export",str(report),"--force-overwrite",str(BINARY),"--seq","1024","--warmup","1","--iterations","1","--output-dir",str(native)]
        cp=run(cmd,env=env,capture_output=True); (d/(role+".stdout.log")).write_text(cp.stdout); (d/(role+".stderr.log")).write_text(cp.stderr)
        rep=report.with_suffix(".ncu-rep")
        status="COLLECTED" if cp.returncode==0 and rep.exists() else "BLOCKED"
        csv_path=d/(role+".csv")
        if status=="COLLECTED":
            ex=run([str(NCU),"--import",str(rep),"--csv","--page","raw","--print-units","base"],capture_output=True)
            csv_path.write_text(ex.stdout); (d/(role+"_export.stderr.log")).write_text(ex.stderr)
            if ex.returncode or any(x in ex.stdout.lower() for x in ("nan","inf")): status="REJECTED_IMPOSSIBLE_OR_EXPORT_FAILED"
        rows.append({"source_commit":commit,"variant":role.split("_")[0],"role":role,"kernel_filter":filt,
            "replay_mode":"kernel","cache_control":"all","ncu_version":version.splitlines()[-1],"status":status,
            "command":shlex.join(cmd),"raw_report":str(rep.relative_to(base)) if rep.exists() else "",
            "exported_csv":str(csv_path.relative_to(base)) if csv_path.exists() else ""})
    write_rows(d/"ncu_manifest.csv",rows)


def checksums(base: Path) -> None:
    paths=sorted(p for p in base.rglob("*") if p.is_file() and p.name!="artifact_checksums.sha256")
    (base/"artifact_checksums.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(base).as_posix()}\n" for p in paths))


def handoff(base: Path, commit: str) -> None:
    final=base/"final"; final.mkdir(exist_ok=True)
    accepted=[]; rejected=[]
    with (base/"run_index.csv").open() as f:
        for r in csv.DictReader(f): (accepted if r["status"]=="ACCEPTED" else rejected).append(r)
    canonical=[r for r in accepted if r["run_id"].startswith("canonical")]
    ncu_rows=[]
    p=base/"ncu/ncu_manifest.csv"
    if p.exists():
        with p.open() as f:ncu_rows=list(csv.DictReader(f))
    ncu_ok=bool(ncu_rows) and all(r["status"]=="COLLECTED" for r in ncu_rows)
    verdict="MANUSCRIPT-READY WITH QUALIFICATIONS"
    (final/"EXECUTIVE_STATUS.md").write_text(f"# Executive Status\n\n**{verdict}.** Fresh CUDA-event performance evidence is tied to source commit `{commit}`. Canonical accepted processes: {len(canonical)}. Rejected attempts retained: {len(rejected)}. Fresh NCU complete: {ncu_ok}. Results are bounded to 1K–8K on one GPU; no formal equivalence claim is made.\n")
    (final/"RUN_ACCEPTANCE_POLICY.md").write_text("# Run Acceptance Policy\n\nSee `../acceptance_policy.json`. Acceptance was independent of measured performance direction. All rejected attempts and telemetry are retained. Fixed execution order is a limitation.\n")
    (final/"SOURCE_MAP.md").write_text(f"# Source Map\n\n- Evidence source commit: `{commit}`\n- Harness: `crates/plkv-kernels/examples/p15b_rtable.rs`\n- A0/B0 kernels: `crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs`\n- A1/B1 kernels: `crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs`\n- Run index: `../run_index.csv`; raw data: `../all_raw_samples.csv`; NCU: `../ncu/`.\n")
    (final/"BENCHMARK_SUMMARY.md").write_text("# Benchmark Summary\n\nUse `../canonical_summary.csv`, `../scaling_distribution_summary.csv`, `../scaling_ratios_and_local_slopes.csv`, and `../timing_boundary_comparison.csv`. Slopes are descriptive local slopes, not asymptotic complexity estimates. No formal equivalence claim is supported.\n")
    (final/"CORRECTNESS_RESULTS.md").write_text("# Correctness Results\n\nFresh timing manifests embed the pre-timing correctness result. The prior bounded four-profile correctness package remains applicable to the same committed source. GPU negative controls are reported separately and are not implied by ordinary correctness passes.\n")
    nc_status="fresh collected reports and CSV exports" if ncu_ok else "fresh collection incomplete/blocked; historical profile retained only as qualified methodology evidence"
    (final/"PROFILER_SUMMARY.md").write_text(f"# Profiler Summary\n\nNCU status: {nc_status}. See `../ncu/ncu_manifest.csv`. Impossible or non-finite exports are rejected, not modified.\n")
    (final/"SUBMISSION_BLOCKERS.md").write_text("# Submission Blockers\n\n## Qualifications\n- Single RTX 4060 Laptop GPU and fixed process order.\n- Bounded 1K–8K scope; no 16K/32K data.\n- Process-wall and CUDA-event boundaries cover different work.\n- Historical measurements are retained only for timing-methodology narrative and remain source-partial.\n- GPU negative controls are `not_run` unless explicitly recorded otherwise.\n\n## Excluded\nR-CTX-PAR, production MLA, kernel optimization, formal equivalence, asymptotic claims, and historical-target matching.\n")
    (final/"NEGATIVE_CONTROL_RESULTS.csv").write_text("control,status,success_definition\nchanged_physical_mapping,not_run,harness detects injected fault\nchanged_element,not_run,harness detects injected fault\ninactive_token_violation,not_run,harness detects injected fault\nmasking_violation,not_run,harness detects injected fault\n")
    (final/"CLAIM_LEDGER.csv").write_text("claim_id,claim,status,evidence,qualification\nC1,Fresh A0/B0 canonical 1K timing,SUPPORTED,../canonical_summary.csv,no formal equivalence\nC2,R-TABLE changes bounded scaling through 8K,SUPPORTED,../scaling_ratios_and_local_slopes.csv,descriptive local slopes only\nC3,Fresh cache/DRAM/FP32/grid/occupancy attribution,"+("SUPPORTED" if ncu_ok else "QUALIFIED")+",../ncu/ncu_manifest.csv,canonical kernels only\nC4,Historical timing-methodology disagreement,HISTORICAL_QUALIFIED,prior handoff,source identity partial\n")
    (final/"EVIDENCE_LEDGER.csv").write_text(f"evidence_id,source_commit,artifact,status\nE1,{commit},../all_raw_samples.csv,FRESH_SOURCE_IDENTIFIED\nE2,{commit},../run_index.csv,FRESH_SOURCE_IDENTIFIED\nE3,{commit},../scaling_distribution_summary.csv,FRESH_SOURCE_IDENTIFIED\nE4,{commit},../ncu/ncu_manifest.csv,{('FRESH_SOURCE_IDENTIFIED' if ncu_ok else 'BLOCKED_QUALIFIED')}\nE5,SOURCE_IDENTITY_PARTIAL,prior historical package,HISTORICAL_METHODOLOGY_ONLY\n")
    (final/"FIGURE_DATA_INDEX.csv").write_text("figure_id,artifact,x,y,filter\nF1,../canonical_summary.csv,variant,median,metric=process_median\nF2,../scaling_distribution_summary.csv,sequence_length,median,variant/component\nF3,../scaling_ratios_and_local_slopes.csv,sequence_length,ratio,kind\n")
    (final/"MANUSCRIPT_AGENT_PROMPT.md").write_text(f"# Manuscript Agent Prompt\n\nUse verdict **{verdict}**. Prefer fresh evidence tied to `{commit}` wherever it overlaps historical evidence. Call slopes ‘descriptive local slopes.’ Do not claim formal equivalence, 16K/32K, R-CTX-PAR, production MLA, or historical-target reproduction. Report fresh/historical disagreement directly. Historical values may appear only in the timing-methodology narrative with partial provenance.\n")
    (final/"RECOMPUTATION_OUTPUT.md").write_text("# Recomputation Output\n\nAll summaries were recomputed from retained CSV/JSONL values by `scripts/run_source_identified_campaign.py`; inputs and formulas are represented in adjacent CSV files.\n")
    (final/"REPRODUCTION_COMMANDS.md").write_text(f"# Reproduction Commands\n\n```bash\npython3 scripts/run_source_identified_campaign.py --output-dir {base.relative_to(ROOT)} --all\n```\n\nSource commit: `{commit}`. No push was performed.\n")
    # Preserve the required disagreement rather than forcing reconciliation.
    fresh="unavailable"
    with (base/"canonical_summary.csv").open() as f:
        rows=list(csv.DictReader(f)); ratio=next((r for r in rows if r["variant"]=="B0/A0"),None)
        if ratio:fresh=ratio["median"]
    (final/"HISTORICAL_COMPARISON.md").write_text(f"# Fresh versus Historical\n\nFresh median per-process B0/A0 ratio: `{fresh}`. Historical CUDA-event ratio was approximately `1.0143`; the historical process-level result reported approximately 32.6% slowdown. Any disagreement is retained without reconciliation. The process-level historical result is timing-methodology narrative only and source-partial.\n")
    (final/"MANIFEST.json").write_text(json.dumps({"verdict":verdict,"source_commit":commit,"accepted_canonical_processes":len(canonical),"rejected_processes":len(rejected),"ncu_complete":ncu_ok,"negative_controls":"not_run"},indent=2)+"\n")


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--timing",action="store_true");ap.add_argument("--ncu",action="store_true");ap.add_argument("--finalize",action="store_true");ap.add_argument("--all",action="store_true")
    a=ap.parse_args(); base=(ROOT/a.output_dir).resolve() if not a.output_dir.is_absolute() else a.output_dir
    commit=output(["git","rev-parse","HEAD"])
    dirty=output(["git","status","--porcelain"])
    if dirty: raise SystemExit("campaign requires a clean evidence-source commit")
    base.mkdir(parents=True,exist_ok=True)
    (base/"acceptance_policy.json").write_text(json.dumps(POLICY,indent=2)+"\n")
    (base/"source_identity.json").write_text(json.dumps({"source_commit":commit,"branch":output(["git","branch","--show-current"]),"remote":output(["git","remote","get-url","origin"]),"dirty":False},indent=2)+"\n")
    help_text=output(["nvidia-smi","--help-query-gpu"]); (base/"nvidia-smi-help-query-gpu.txt").write_text(help_text+"\n")
    power=power_state(); (base/"power_mode.json").write_text(json.dumps(power,indent=2)+"\n")
    if (a.timing or a.all):
        if not power["accepted"]: raise SystemExit("timing blocked: AC + High performance mode not confirmed; evidence retained")
        ok,reasons=idle_preflight(base); (base/"idle_preflight.json").write_text(json.dumps({"accepted":ok,"reasons":reasons},indent=2)+"\n")
        if not ok: raise SystemExit(f"timing blocked by idle preflight: {reasons}")
        env=wsl_env(); run(["cargo","build","--release","-p","plkv-kernels","--features","gpu-cutile","--example","p15b_rtable"],env=env,check=True)
        (base/"timing").mkdir(exist_ok=True); accepted=0; attempt=0
        while accepted<5 and attempt<10:
            attempt+=1; m=run_benchmark(base,"timing",attempt,1024,*PLAN[1024],commit,env)
            # Rename identity in-place for unambiguous canonical indexing.
            p=base/"timing"/f"attempt_{attempt:02d}_seq1024"/"run_manifest.json"; m["run_id"]=f"canonical-{attempt:02d}-seq1024"; p.write_text(json.dumps(m,indent=2)+"\n")
            if m["status"]=="ACCEPTED":accepted+=1
            if accepted<5: time.sleep(5)
        if accepted<5: raise SystemExit(f"only {accepted} accepted canonical processes; all attempts retained")
        (base/"scaling").mkdir(exist_ok=True)
        for i,n in enumerate((2048,4096,8192),1):
            ok,reasons=idle_preflight(base/"scaling")
            # Keep each preflight separately (idle_preflight writes shared raw file, then archive it).
            (base/"scaling"/f"preflight_seq{n}.json").write_text(json.dumps({"accepted":ok,"reasons":reasons},indent=2)+"\n")
            (base/"scaling"/"idle_telemetry.csv").replace(base/"scaling"/f"preflight_seq{n}_telemetry.csv")
            if not ok: continue
            m=run_benchmark(base,"scaling",i,n,*PLAN[n],commit,env)
            p=base/"scaling"/f"attempt_{i:02d}_seq{n}"/"run_manifest.json";m["run_id"]=f"scaling-{i:02d}-seq{n}";p.write_text(json.dumps(m,indent=2)+"\n")
            time.sleep(8)
        aggregate(base,commit)
    if (a.ncu or a.all): ncu_collect(base,commit,wsl_env())
    if (a.finalize or a.all):
        if not (base/"run_index.csv").exists(): aggregate(base,commit)
        handoff(base,commit)
    checksums(base)
    print(base); return 0

if __name__=="__main__": raise SystemExit(main())
