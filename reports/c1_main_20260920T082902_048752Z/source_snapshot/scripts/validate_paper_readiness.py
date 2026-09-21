#!/usr/bin/env python3
"""Validate a generated paper-readiness handoff without executing GPU work."""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path

REQUIRED = {"EXECUTIVE_STATUS.md","CODEBASE_CHANGES.md","VALIDATION_RESULTS.md","EVIDENCE_LEDGER.csv","CLAIM_LEDGER.csv","CONTRADICTIONS.md","METRIC_SPECIFICATION.md","CORRECTNESS_MATRIX.csv","BENCHMARK_SUMMARY.csv","PROFILER_SUMMARY.csv","RTABLE_MECHANISM.md","FIGURE_DATA_INDEX.csv","SOURCE_MAP.md","SUBMISSION_BLOCKERS.md","RESEARCH_HANDOFF.md","REPRODUCTION_COMMANDS.md","artifact_checksums.sha256"}
CSV_FIELDS = {
 "EVIDENCE_LEDGER.csv": {"evidence_id","phase","metric","value","unit","statistic","sample_count","variant","sequence_length","artifact_path","raw_data_path","source_commit","timing_method","verification_status","limitations"},
 "CLAIM_LEDGER.csv": {"claim_id","claim_text","classification","supporting_evidence_ids","source_paths","allowed_section","required_qualification","status","blocking"},
 "FIGURE_DATA_INDEX.csv": {"figure_id","panel","claim_supported","artifact_path","table_or_json_path","x_field","y_field","filter","statistic","unit","verification_status"},
}
def sha(p: Path) -> str:
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main() -> int:
 ap=argparse.ArgumentParser(); ap.add_argument("handoff",type=Path); a=ap.parse_args(); d=a.handoff
 errors=[]
 missing=REQUIRED-{p.name for p in d.iterdir() if p.is_file()}
 if missing: errors.append("missing: "+", ".join(sorted(missing)))
 for name, fields in CSV_FIELDS.items():
  p=d/name
  if p.exists():
   with p.open(newline="",encoding="utf-8") as f:
    r=csv.DictReader(f)
    if not r.fieldnames or not fields.issubset(r.fieldnames): errors.append(f"{name}: missing required columns")
    elif not list(r): errors.append(f"{name}: no data rows")
 sums={}
 p=d/"artifact_checksums.sha256"
 if p.exists():
  for line_number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
   if not line.strip():
    continue
   try:
    digest, name = line.split("  ", 1)
   except ValueError:
    errors.append(f"checksum manifest: malformed line {line_number}")
    continue
   if name in sums:
    errors.append(f"checksum manifest: duplicate entry {name}")
   sums[name] = digest
  actual = {q.name for q in d.iterdir() if q.is_file() and q.name != p.name}
  missing_entries = actual - sums.keys()
  unexpected_entries = sums.keys() - actual
  if missing_entries: errors.append("checksum manifest: missing " + ", ".join(sorted(missing_entries)))
  if unexpected_entries: errors.append("checksum manifest: unexpected " + ", ".join(sorted(unexpected_entries)))
  for name,digest in sums.items():
   q=d/name
   if not q.is_file() or sha(q)!=digest: errors.append(f"checksum mismatch: {name}")
 if errors:
  print("PAPER_READINESS_VALIDATION_FAIL"); print("\n".join("- "+x for x in errors)); return 1
 print(f"PAPER_READINESS_VALIDATION_OK files={len(sums)}")
 return 0
if __name__=="__main__": raise SystemExit(main())
