from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
VALIDATOR = ROOT / "scripts/validate_paper_readiness.py"
REQUIRED = {
    "EXECUTIVE_STATUS.md",
    "CODEBASE_CHANGES.md",
    "VALIDATION_RESULTS.md",
    "EVIDENCE_LEDGER.csv",
    "CLAIM_LEDGER.csv",
    "CONTRADICTIONS.md",
    "METRIC_SPECIFICATION.md",
    "CORRECTNESS_MATRIX.csv",
    "BENCHMARK_SUMMARY.csv",
    "PROFILER_SUMMARY.csv",
    "RTABLE_MECHANISM.md",
    "FIGURE_DATA_INDEX.csv",
    "SOURCE_MAP.md",
    "SUBMISSION_BLOCKERS.md",
    "RESEARCH_HANDOFF.md",
    "REPRODUCTION_COMMANDS.md",
}
CSV_HEADERS = {
    "EVIDENCE_LEDGER.csv": (
        "evidence_id,phase,metric,value,unit,statistic,sample_count,variant,"
        "sequence_length,artifact_path,raw_data_path,source_commit,timing_method,"
        "verification_status,limitations\n"
        "row,phase,metric,value,unit,statistic,1,variant,1,path,path,commit,timing,"
        "status,none\n"
    ),
    "CLAIM_LEDGER.csv": (
        "claim_id,claim_text,classification,supporting_evidence_ids,source_paths,"
        "allowed_section,required_qualification,status,blocking\n"
        "row,claim,class,evidence,source,section,qualification,ACTIVE,no\n"
    ),
    "FIGURE_DATA_INDEX.csv": (
        "figure_id,panel,claim_supported,artifact_path,table_or_json_path,x_field,y_field,"
        "filter,statistic,unit,verification_status\n"
        "row,panel,claim,path,path,x,y,filter,stat,unit,status\n"
    ),
}


def write_checksums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.iterdir()):
        if path.name != "artifact_checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}")
    (directory / "artifact_checksums.sha256").write_text("\n".join(lines) + "\n")


def make_handoff(directory: Path) -> None:
    directory.mkdir()
    for name in REQUIRED:
        directory.joinpath(name).write_text(CSV_HEADERS.get(name, "content\n"))
    write_checksums(directory)


def validate(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(directory)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_readiness_validator_accepts_complete_checksums(tmp_path: Path) -> None:
    handoff = tmp_path / "final"
    make_handoff(handoff)

    result = validate(handoff)

    assert result.returncode == 0, result.stdout + result.stderr


def test_readiness_validator_rejects_omitted_checksum_entry(tmp_path: Path) -> None:
    handoff = tmp_path / "final"
    make_handoff(handoff)
    checksums = handoff / "artifact_checksums.sha256"
    lines = checksums.read_text().splitlines()
    checksums.write_text("\n".join(lines[:-1]) + "\n")

    result = validate(handoff)

    assert result.returncode == 1
    assert "checksum manifest: missing" in result.stdout
