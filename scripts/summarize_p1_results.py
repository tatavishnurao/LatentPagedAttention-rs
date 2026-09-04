"""Assemble machine-readable P1 reconnaissance/correctness artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/p1_sequence_crossover"
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072)


def main() -> None:
    records = []
    correctness = []
    for length in LENGTHS:
        root = OUT / f"recon_{length}"
        found = list(root.glob("attempt_*/run_record.json"))
        if found:
            record = json.loads(found[0].read_text())
            record["seq_len"] = length
            summary = json.loads((ROOT / record["summary_path"]).read_text())
            records.append(record)
            correctness.append({"seq_len": length, "status": "EXECUTED_REJECTED_ENVIRONMENT", "metrics": summary["correctness"], "reason": record["reasons"]})
        else:
            probe = OUT / f"probe_{length}"
            summaries = list(probe.glob("ab_summary_*.json"))
            if summaries:
                summary = json.loads(summaries[0].read_text())
                correctness.append({"seq_len": length, "status": "PROBE_ONLY", "metrics": summary["correctness"]})
                if length == 32768:
                    continue
            else:
                records.append({"seq_len": length, "status": "UNSUPPORTED", "reason": "No accepted process; 300-iteration run not completed within safe runtime/thermal budget."})
                correctness.append({"seq_len": length, "status": "UNSUPPORTED", "reason": "No accepted process."})
    # The 32K probe is retained as a feasibility datapoint, never as run-level evidence.
    probe = OUT / "probe_32768"
    summaries = list(probe.glob("ab_summary_*.json"))
    if summaries:
        summary = json.loads(summaries[0].read_text())
        records.append({"seq_len": 32768, "status": "PROBE_ONLY", "a_median_ms": summary["full_kv"]["gpu_execution"]["median_ms"], "b_median_ms": summary["latent_original"]["gpu_execution"]["median_ms"], "b_over_a": summary["latent_original"]["gpu_execution"]["median_ms"] / summary["full_kv"]["gpu_execution"]["median_ms"], "reason": "One iteration only; 300-iteration run was thermally/runtimely unsafe."})
    records.sort(key=lambda x: (x["seq_len"], x.get("status", "")))
    (OUT / "replications_manifest.json").write_text(json.dumps({"protocol": "P1 reconnaissance; accepted criteria unchanged from P0.5", "records": records}, indent=2) + "\n")
    (OUT / "run_level_by_sequence.json").write_text(json.dumps({"statistical_unit": "independent process", "records": records, "note": "No sequence length has >=5 accepted independent processes; rejected values and probe values are not inferential data."}, indent=2) + "\n")
    (OUT / "correctness_by_sequence.json").write_text(json.dumps({"tolerances": "P0.5 unchanged", "records": correctness}, indent=2) + "\n")
    analytical = []
    for length in LENGTHS:
        full_k = length * 4 * 64 * 2
        analytical.append({"seq_len": length, "blocks": length // 16, "full_k_bytes": full_k, "full_v_bytes": full_k, "full_kv_bytes": 2 * full_k, "latent_bytes": length * 32 * 2, "full_k_mib": full_k / 2**20, "full_v_mib": full_k / 2**20, "full_kv_mib": 2 * full_k / 2**20, "latent_mib": length * 32 * 2 / 2**20})
    (OUT / "working_set_scaling.json").write_text(json.dumps(analytical, indent=2) + "\n")
    (OUT / "ncu_sequence_metrics.json").write_text(json.dumps({"status": "NOT_COLLECTED", "reason": "P1 accepted NCU points were not attempted after 32K run reached 87 C and was stopped; P0.5 focused metrics remain the only hardware-counter evidence.", "lengths": list(LENGTHS)}, indent=2) + "\n")
    (OUT / "crossover_analysis.json").write_text(json.dumps({"status": "INCONCLUSIVE", "cache_crossover_observed": False, "reason": "No accepted multi-length process or P1 NCU sequence data; reconnaissance beyond 16K was rejected/unsupported for thermal/runtime reasons.", "one_iteration_probe_32768": {"full_kv_ms": 644.3980712890625, "latent_ms": 724.9879150390625, "b_over_a": 1.1250622050882104}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
