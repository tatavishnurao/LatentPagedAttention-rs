"""Build and run one accepted A/B process for each P1 sequence length."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072)
OUT = ROOT / "reports/p1_sequence_crossover"
ENV = {
    **os.environ,
    "CUDA_TOOLKIT_PATH": "/usr/local/cuda-13.3",
    "LIBCLANG_PATH": "/usr/lib/llvm-18/lib",
    "CUTILE_TILEIRAS_PATH": "/usr/local/cuda-13.3/bin/tileiras",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for length in LENGTHS:
        binary = ROOT / f"target/release/examples/p1_seq_{length}"
        log = OUT / f"build_{length}.log"
        build = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "plkv-kernels",
                "--features",
                "gpu-cutile",
                "--example",
                f"p1_seq_{length}",
            ],
            cwd=ROOT,
            env=ENV,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
        log.write_text(build.stdout, encoding="utf-8")
        row: dict[str, object] = {"seq_len": length, "build_returncode": build.returncode}
        if build.returncode == 0:
            out_dir = OUT / f"recon_{length}"
            run = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/run_p05_ab_replications.py",
                    "--repo",
                    str(ROOT),
                    "--binary",
                    str(binary),
                    "--output-dir",
                    str(out_dir),
                    "--accepted-runs",
                    "1",
                    "--max-attempts",
                    "1",
                    "--warmup",
                    "20",
                    "--iterations",
                    "300",
                    "--pre-seconds",
                    "10",
                    "--between-seconds",
                    "0",
                ],
                cwd=ROOT,
                env=ENV,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
            (out_dir / "runner.log").write_text(run.stdout, encoding="utf-8")
            row["run_returncode"] = run.returncode
            match = re.search(
                r"attempt=\d+ status=(\S+) A_ms=([0-9.]+) B_ms=([0-9.]+) B/A=([0-9.]+)",
                run.stdout,
            )
            if match:
                row.update(
                    {
                        "status": match.group(1),
                        "a_median_ms": float(match.group(2)),
                        "b_median_ms": float(match.group(3)),
                        "b_over_a": float(match.group(4)),
                    }
                )
        rows.append(row)
        (OUT / "reconnaissance_summary.json").write_text(
            json.dumps({"lengths": rows}, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
