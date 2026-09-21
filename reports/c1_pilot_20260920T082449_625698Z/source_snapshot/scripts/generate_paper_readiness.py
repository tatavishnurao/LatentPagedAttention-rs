#!/usr/bin/env python3
"""Generate non-destructive paper-readiness provenance and ledger artifacts.

No benchmark, CUDA, NCU, build, authentication, or git-mutating command is run.
Historical results are marked source-identity unresolved unless their inspected
artifact itself carries a source commit.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("p0_gpu_baseline", "p05_hardware_attribution", "p05_clean_display_replications", "p1_sequence_crossover", "p15_kernel_scaling", "p15b_rtable")
FIELDS = ["artifact_id","experiment_phase","path","format","exists","size_bytes","sha256","source_commit","variant","sequence_length","timing_method","raw_or_summary","inspection_status","notes"]

def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def git(*args: str) -> str:
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()

def phase_for(p: Path) -> str:
    s=p.as_posix()
    return next((x for x in PHASES if f"/{x}/" in f"/{s}"),"OTHER")

def fmt(p: Path) -> str: return p.suffix.lstrip(".").upper() or "FILE"
def raw_kind(p: Path) -> str:
    return "RAW" if p.suffix in {".jsonl",".csv",".ncu-rep",".sqlite"} else "SUMMARY_OR_LOG"
def inspect(p: Path) -> str:
    if p.suffix == ".json":
        try: json.loads(p.read_text(encoding="utf-8")); return "PARSED_SCHEMA_NOT_SEMANTICALLY_INSPECTED"
        except Exception: return "PRESENT_PARSE_FAILED"
    return "DISCOVERED_NOT_OPENED"
def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    out=args.output_dir; final=out/"final"; final.mkdir(parents=True,exist_ok=False)
    status=git("status","--short").splitlines(); head=git("rev-parse","HEAD")
    files=[]
    for phase in PHASES:
        d=ROOT/"reports"/phase
        if not d.exists():
            files.append({"artifact_id":phase,"experiment_phase":phase,"path":str(d.relative_to(ROOT)),"format":"DIRECTORY","exists":"false","size_bytes":"","sha256":"","source_commit":"SOURCE_IDENTITY_UNRESOLVED","variant":"","sequence_length":"","timing_method":"","raw_or_summary":"","inspection_status":"MISSING","notes":"known phase directory absent"})
            continue
        for p in sorted(x for x in d.rglob("*") if x.is_file()):
            files.append({"artifact_id":hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:16],"experiment_phase":phase_for(p),"path":str(p.relative_to(ROOT)),"format":fmt(p),"exists":"true","size_bytes":p.stat().st_size,"sha256":digest(p),"source_commit":"SOURCE_IDENTITY_UNRESOLVED","variant":"","sequence_length":"","timing_method":"","raw_or_summary":raw_kind(p),"inspection_status":inspect(p),"notes":"inventory discovery; semantic source association requires reconciliation"})
    write_csv(out/"ARTIFACT_INVENTORY.csv",FIELDS,files)
    (out/"REPOSITORY_STATE.md").write_text(f"# Repository State\n\n- Branch: `{git('branch','--show-current')}`\n- HEAD at audit: `{head}`\n- UTC audit generation: `{datetime.now(timezone.utc).isoformat()}`\n- Dirty/untracked entries preserved without modification:\n\n```text\n"+"\n".join(status)+"\n```\n",encoding="utf-8")
    (out/"CODEBASE_AUDIT.md").write_text("""# Codebase Audit

## Inspected locations
- Original generated A0/B0 kernels: `crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs`.
- Repaired A1/B1 kernels: `crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs`.
- R-TABLE measurement driver/analysis: `scripts/run_p15b_rtable.py`, `scripts/analyze_p15b_rtable.py`.
- Earlier timing, NCU, thermal, reference, and release scripts under `scripts/`.
- Reports indexed in `ARTIFACT_INVENTORY.csv`; parse-only status is not semantic inspection.

## Source-level R-TABLE finding
A0/B0 load a table-shaped tile and extract the requested logical entry. A1/B1 use `table.load_tile(const_shape![1], [logical])` in generated full-KV and latent score/context families. `scripts/validate_rtable_structure.py` checks this contract. This is source-level evidence, not an instruction- or transaction-level measurement.

## Entry points and validation
CPU/Python tests are under `tests/` and `crates/plkv-core`; GPU examples are under `crates/plkv-kernels/examples`; profiling extraction is `scripts/extract_p05_ncu.py`; existing release validation is `scripts/validate_release.sh`.
""",encoding="utf-8")
    variants=[
      ["A0","crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs","full_kv_baseline_kernel_<N>","generated module","FP16 K/V","FP32","whole table tile + extract","full score","shared softmax","serial 16-block context","score N blocks; context 16 blocks","[16,N] scores; [16,64] context"],
      ["B0","crates/plkv-kernels/src/cutile/p1_sequence_kernels.rs","model_profile_kernel_<N>","generated module","FP16 latent","FP32","whole table tile + extract","latent score with repeated projection","shared softmax","serial 16-block context + output projection","score N blocks; context 16 blocks","[16,N] scores; [16,64] context"],
      ["A1","crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs","p15b_full_kv_baseline_kernel_<N>","generated module","FP16 K/V","FP32","one-entry runtime tile load","same as A0","shared softmax","same as A0","same as A0","same as A0"],
      ["B1","crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs","p15b_model_profile_kernel_<N>","generated module","FP16 latent","FP32","one-entry runtime tile load","same as B0","shared softmax","same as B0","same as B0","same as B0"],]
    with (out/"VARIANT_IDENTITY.md").open("w",encoding="utf-8") as f:
      f.write("# Variant Identity\n\n| Variant | Source | Entry family | Selector | Storage | Arithmetic | Table lookup | Score | Softmax | Context | Launch topology | Output |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
      for r in variants:f.write("|"+"|".join(r)+"|\n")
    mi=[]
    for rid,path,var,seq,method in [("P0_EVENT_1K","reports/p0_gpu_baseline/abc_summary_1788501169.json","A/B/C",1024,"CUDA_EVENT_PIPELINE"),("P05_CLEAN_DISPLAY","reports/p05_clean_display_replications/run_level_analysis.json","A0/B0",1024,"CUDA_EVENT_KERNEL"),("P15A_DIAG","reports/p15_kernel_scaling/per_kernel_timings.json","A0/B0", "1024-16384","CUDA_EVENT_KERNEL"),("P15B_DIAG","reports/p15b_rtable/per_kernel_timings.json","A0/B0/A1/B1","1024-8192","CUDA_EVENT_KERNEL")]:
      mi.append({"result_id":rid,"artifact_path":path,"variant":var,"source_commit":"SOURCE_IDENTITY_UNRESOLVED","binary_or_command":"see artifact/report; command association unresolved","timing_boundary":method,"warmups":"artifact-specific","iterations":"artifact-specific","process_count":"artifact-specific","sequence_length":seq,"block_size":16,"head_configuration":"q=16,kv=4","storage_dtype":"FP16","arithmetic_dtype":"FP32","gpu":"RTX 4060 Laptop (report-derived)","driver":"UNRESOLVED","cuda":"13.3 report-derived","cutile":"0.2.0 report-derived","source_identity_status":"SOURCE_IDENTITY_UNRESOLVED"})
    write_csv(out/"MEASUREMENT_IDENTITY.csv",list(mi[0]),mi)
    evidence=[
      ["E01","P0","full/latent event medians","0.137216 / 0.139184","ms","median","300 reported","A0/B0",1024,"reports/p0_gpu_baseline/abc_summary_1788501169.json","reports/p0_gpu_baseline/*samples*.jsonl","SOURCE_IDENTITY_UNRESOLVED","CUDA_EVENT_PIPELINE","REPORT-DERIVED — RAW SAMPLES NOT INSPECTED","historical source association unresolved"],
      ["E02","P0.5","clean-display median B/A","1.008662129","ratio","median","5 accepted processes","A0/B0",1024,"reports/p05_clean_display_replications/run_level_analysis.json","reports/p05_clean_display_replications/attempt_*/ab_samples_*.jsonl","SOURCE_IDENTITY_UNRESOLVED","CUDA_EVENT_KERNEL","REPORT-DERIVED — RAW SAMPLES NOT INSPECTED","p=.0625; do not pool cohorts"],
      ["E03","P0.5","full score/context L2 sector hit","98.49 / 98.23","percent","profiled value","representative launches","A0",1024,"reports/p05_hardware_attribution/ncu_focused_metrics.json","reports/p05_hardware_attribution/*.ncu-rep","SOURCE_IDENTITY_UNRESOLVED","NCU_REPLAY","REPORT-DERIVED — RAW SAMPLES NOT INSPECTED","cache-control unresolved"],
      ["E04","P1.5B","R-TABLE endpoint slopes","A score 2.24→-0.05; B score 2.19→0.67; A ctx 2.16→0.73; B ctx 2.01→0.67","log2 ratio","endpoint diagnostic","bounded samples","A0/B0/A1/B1","1024-8192","reports/p15b_rtable/scaling_exponents.json","reports/p15b_rtable/summary_seq*.jsonl","SOURCE_IDENTITY_UNRESOLVED","CUDA_EVENT_KERNEL","MACHINE-READABLE SUMMARY PARSED","not regression/asymptotic"],
      ["E05","P1.5B","repaired/control reported numerical difference","0","aggregate error","reported","completed 1K-8K","A0/A1,B0/B1","1024-8192","reports/p15b_rtable/correctness.json","reports/p15b_rtable/summary_seq*.jsonl","SOURCE_IDENTITY_UNRESOLVED","CORRECTNESS","MACHINE-READABLE SUMMARY PARSED","not bitwise or explicit finite check"],
      ["E06","P1.5B","corrected 16K / 32K status","thermal abort / not attempted","status","n/a","n/a","A1/B1","16384/32768","reports/p15b_rtable/thermal_manifest.json; reports/p15b_rtable/feasibility_32k.json","","SOURCE_IDENTITY_UNRESOLVED","THERMAL_GUARD","MACHINE-READABLE SUMMARY PARSED","no corrected 16K/32K claim"],
      ["E07","analysis","persistent state ratio","16","ratio","formula","n/a","A/B",1024,"docs/02_memory_model.md","","n/a","ANALYTICAL","SOURCE INSPECTED","persistent state only; excludes total memory"]]
    ef=["evidence_id","phase","metric","value","unit","statistic","sample_count","variant","sequence_length","artifact_path","raw_data_path","source_commit","timing_method","verification_status","limitations"]
    write_csv(final/"EVIDENCE_LEDGER.csv",ef,[dict(zip(ef,x)) for x in evidence])
    claims=[["C01","The workload is synthetic paged latent-cache attention, not complete MLA","SOURCE-LEVEL FACT","E07","source and docs","Introduction/Methods","retain synthetic scope","ACTIVE","no"],["C02","The chosen persistent cache state has a 16x byte ratio","ANALYTICAL","E07","docs/02_memory_model.md","Methods","persistent state only","ACTIVE","no"],["C03","Original A0/B0 are near practical parity at 1K","INFERENCE","E01;E02","P0/P0.5 artifacts","Results","not equality or formal equivalence","ACTIVE","no"],["C04","Original 1K profiles show strong within-profile L2 reuse","INFERENCE","E03","NCU artifacts","Results","cache-control unresolved; no warm-start proof","ACTIVE","no"],["C05","R-TABLE materially changes observed score/context scaling through 8K","INFERENCE","E04;E05","P1.5B source and summaries","Results","endpoint diagnostics only; bounded samples","ACTIVE","no"],["C06","R-TABLE is validated at 16K or 32K","UNSUPPORTED","E06","thermal/feasibility artifacts","None","exclude","EXCLUDED","yes"],["C07","A1/B1 are bitwise identical to A0/B0","UNSUPPORTED","E05","correctness summary","None","raw elementwise finite comparison required","EXCLUDED","yes"]]
    cf=["claim_id","claim_text","classification","supporting_evidence_ids","source_paths","allowed_section","required_qualification","status","blocking"]
    write_csv(final/"CLAIM_LEDGER.csv",cf,[dict(zip(cf,x)) for x in claims])
    (final/"CONTRADICTIONS.md").write_text("# Contradictions and Resolution\n\n- Historical process-level ~32.6% slowdown versus P0 event medians: retain the former only as methodology-failure history; use event timing for steady-state GPU latency.\n- Earlier P0.5 display-connected and later clean-display cohorts: do not pool; the five-process clean-display cohort is the selected main cohort.\n- Report prose may use bitwise wording while machine-readable aggregate errors show zero: use zero reported numerical difference until raw bit-pattern/finite comparison is inspected.\n- 16K P1.5A completed diagnostic phases, but corrected P1.5B 16K did not produce a summary: do not extend repaired claims beyond 8K.\n",encoding="utf-8")
    (final/"STALE_CLAIMS.md").write_text("# Stale Claims\n\n- Remove steady-state interpretation of the historical ~32.6% process-level slowdown.\n- Do not call the five-process cohort formal equivalence.\n- Do not use >100% L2 hit rate as physical evidence.\n- Do not claim bitwise equality, total-memory 16x savings, repaired 16K+, long-context crossover, or external-baseline superiority.\n",encoding="utf-8")
    for name,text in {
      "METRIC_SPECIFICATION.md":"# Metric Specification\n\nNew correctness outputs must include finite counts, NaN/Inf counts, mismatch count, absolute/relative errors, checksums, mapping status, and masking violations. New timing outputs must retain every sample and label `PROCESS_END_TO_END`, `CUDA_EVENT_KERNEL`, or `CUDA_EVENT_PIPELINE`. Slopes are descriptive local/endpoint values only. Impossible profiler metrics are rejected, not clipped.\n",
      "EXPERIMENT_PROTOCOL.md":"# Experiment Protocol\n\nNo GPU run is scheduled by this audit. If a central claim cannot be narrowed, use a unique timestamped directory, predeclare acceptance criteria, retain all raw samples and rejected runs, run correctness first, and use in-process CUDA events for steady-state conclusions. Do not run corrected 16K/32K for graph extension.\n",
      "RUN_ACCEPTANCE_POLICY.md":"# Run Acceptance Policy\n\nPredeclare acceptance before viewing performance: retain all processes; report all-run and accepted summaries; record order and telemetry; investigate >5% process-median deviations without deletion. Five independent accepted processes are the canonical target. A performance direction is never an acceptance criterion.\n",
      "RTABLE_MECHANISM.md":"# R-TABLE Mechanism\n\nLet M=ceil(N/P) and Hq be query heads. Under the confirmed generated call pattern, an original score lookup requests an M-entry table tile while repair requests one entry. If invoked once per page per query head, requested source-level entries are Hq*M^2 original versus Hq*M repaired. Context call frequency must be stated separately from its serial loop. This is source-level requested-element accounting only; it does not prove instruction, DRAM, or total-runtime complexity. Matched through-8K timings are consistent with the mechanism; alternative compiler/cache effects remain possible.\n",
      "SUBMISSION_BLOCKERS.md":"# Submission Blockers\n\n## SUBMISSION-CRITICAL\n- Historical source commits/binaries for central measurements remain `SOURCE_IDENTITY_UNRESOLVED`.\n- Raw samples and source fields for central P0/P0.5/NCU claims require semantic inspection and figure indexing.\n- Repaired-control bitwise/finite-output language is unsupported without retained elementwise comparison.\n\n## QUALIFIED LIMITATION\n- Single GPU, fixed order/DVFS, bounded diagnostic samples, unresolved NCU cache control, lost first-run thermal manifest, and through-8K repair scope.\n\n## OUT-OF-SCOPE FUTURE WORK\n- Corrected 16K/32K, R-CTX-PAR, real model quality, production serving, allocator dynamics, and multi-GPU work.\n",
      "SOURCE_MAP.md":"# Source Map\n\nSee `../ARTIFACT_INVENTORY.csv`, `../MEASUREMENT_IDENTITY.csv`, `EVIDENCE_LEDGER.csv`, and source locations listed in `../CODEBASE_AUDIT.md`. Current HEAD is an audit context, not asserted historical measurement identity.\n",
      "REPRODUCTION_COMMANDS.md":"# Reproduction Commands\n\nStructural validation: `python3 scripts/validate_rtable_structure.py`\n\nAudit regeneration: `[TODO: record exact command with selected output directory]`\n\nCPU/structural validation: `cargo fmt --check`, `cargo check`, `cargo test`, `python3 -m py_compile scripts/*.py`, `[TODO: project Python test command]`. GPU commands must be recorded from the exact run manifest; do not reuse this document as a historical command record.\n",
      "EXECUTIVE_STATUS.md":"# Executive Status\n\nEvidence package is an audit handoff, not submission clearance. Existing scope supports a through-8K controlled characterization if central raw/provenance associations are recovered. No new GPU experiment was run by this audit.\n",
      "CODEBASE_CHANGES.md":"# Codebase Changes\n\nAdded `scripts/validate_rtable_structure.py` and `scripts/generate_paper_readiness.py`. They are non-GPU, non-destructive reproducibility/audit tooling. No kernel algorithm, dimensions, precision, or timing harness was changed.\n",
      "VALIDATION_RESULTS.md":"# Validation Results\n\n[TODO: populate from executed non-GPU commands in the final handoff.]\n",
      "CORRECTNESS_MATRIX.csv":"comparison,tiny,model_small,nonidentity_paging,partial_block,runtime_active_length,gpu_write_handoff,status\nPython_vs_Rust_CPU,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,WHERE_APPLICABLE,NEEDS_RECONCILIATION\nRust_CPU_vs_A0,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,WHERE_APPLICABLE,NEEDS_RECONCILIATION\nRust_CPU_vs_B0,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,REQUIRED,NEEDS_RECONCILIATION\nA0_vs_A1,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,WHERE_APPLICABLE,NEEDS_RECONCILIATION\nB0_vs_B1,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,UNINSPECTED,REQUIRED,NEEDS_RECONCILIATION\n",
      "BENCHMARK_SUMMARY.csv":"result_id,variant,sequence_length,timing_method,status,notes\nP0_EVENT_1K,A0/B0,1024,CUDA_EVENT_PIPELINE,REPORT_DERIVED,raw samples not semantically inspected\nP05_CLEAN_DISPLAY,A0/B0,1024,CUDA_EVENT_KERNEL,REPORT_DERIVED,five accepted cohort not pooled\nP15B_DIAG,A0/B0/A1/B1,1024-8192,CUDA_EVENT_KERNEL,MACHINE_READABLE_SUMMARY_PARSED,bounded diagnostics\n",
      "PROFILER_SUMMARY.csv":"result_id,variant,sequence_length,status,notes\nP05_NCU,A0/B0,1024,REPORT_DERIVED,cache control and historical source identity unresolved\n",
      "FIGURE_DATA_INDEX.csv":"figure_id,panel,claim_supported,artifact_path,table_or_json_path,x_field,y_field,filter,statistic,unit,verification_status\nF1,process_ratio,near parity,REPORT_PATH_TODO,run_level_analysis.json,process_id,b_over_a,accepted clean-display,per-process ratio,ratio,RAW_SELECTION_TODO\nF2,decomposition,score/context diagnosis,reports/p15_kernel_scaling,per_kernel_timings.json,sequence_length,median_ms,kernel and variant,median,ms,RAW_SELECTION_TODO\nF3,rtable,repaired through-8K,reports/p15b_rtable,per_kernel_timings.json,sequence_length,median_ms,A0/A1/B0/B1 score/context,median,ms,MACHINE_SUMMARY_PARSED_RAW_SELECTION_TODO\n",
      "RESEARCH_HANDOFF.md":"# Research Handoff\n\nUse only active claims in `CLAIM_LEDGER.csv`. Figures must be regenerated from fields listed in `FIGURE_DATA_INDEX.csv` after raw-source selection. Preserve `SOURCE_IDENTITY_UNRESOLVED` and report-derived labels until resolved.\n"}.items(): (final/name).write_text(text,encoding="utf-8")
    checks=[]
    for p in sorted(final.iterdir()):
      if p.is_file(): checks.append(f"{digest(p)}  {p.name}")
    (final/"artifact_checksums.sha256").write_text("\n".join(checks)+"\n",encoding="utf-8")
    print(out)
if __name__=="__main__": main()
