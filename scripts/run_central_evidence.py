#!/usr/bin/env python3
"""Auditable launcher for central paper-evidence runs.

The benchmark remains the source of measurements; this launcher supplies the
immutable run envelope required by the evidence protocol. It deliberately does
not accept 16384 or 32768 sequence lengths.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, shlex, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

TIMING = {"PROCESS_END_TO_END", "CUDA_EVENT_KERNEL", "CUDA_EVENT_PIPELINE"}

def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def git(*args: str) -> str | None:
    try: return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError): return None

def wsl_driver_library() -> str | None:
    """Return the unique WSL PTX-JIT directory, or None outside WSL."""
    try:
        release = Path('/proc/sys/kernel/osrelease').read_text().lower()
    except OSError:
        return None
    if 'microsoft' not in release and 'wsl' not in release:
        return None
    matches = sorted({str(Path(p).parent) for p in glob.glob(
        '/usr/lib/wsl/drivers/*/libnvidia-ptxjitcompiler.so.1')
        if os.path.isfile(p)})
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one WSL PTX-JIT directory, found {matches!r}')
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sequence-length", type=int, required=True)
    ap.add_argument("--timing-method", choices=sorted(TIMING), required=True)
    ap.add_argument("--timing-boundary", required=True)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = ap.parse_args()
    if args.sequence_length in (16384, 32768):
        ap.error("16K/32K evidence runs are prohibited")
    if not args.command or args.command[0] == "--":
        ap.error("--command must be followed by an executable")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    child_env = os.environ.copy()
    wsl_dir = wsl_driver_library()
    if wsl_dir:
        paths = [wsl_dir, '/usr/lib/wsl/lib']
        existing = child_env.get('LD_LIBRARY_PATH')
        if existing: paths.append(existing)
        child_env['LD_LIBRARY_PATH'] = ':'.join(paths)
    proc = subprocess.run(args.command, text=True, capture_output=True, env=child_env)
    (out / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        return proc.returncode
    summaries = sorted(out.glob("summary*.json"))
    jsonls = sorted(out.glob("*.jsonl"))
    if len(summaries) != 1 or len(jsonls) != 1:
        raise SystemExit("benchmark must emit exactly one summary JSON and one JSONL sample file")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    rows = []
    for line in jsonls[0].read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if "kernel" in r:
            variant, component = r["kernel"].split("_", 1)
        else:
            variant, component = r.get("variant"), r.get("component", "pipeline")
        rows.append({"variant": variant, "component": component,
                     "iteration": r.get("iteration"), "latency_ms": r.get("latency_ms")})
    with (out / "raw_samples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "component", "iteration", "latency_ms"])
        w.writeheader(); w.writerows(rows)
    env = {"gpu_name": None, "driver_version": None, "cuda_version": None,
           "temperature_before": None, "temperature_after": None,
           "graphics_clock_before": None, "graphics_clock_after": None,
           "power_limit": None}
    manifest = {
        "run_id": f"central-{started.replace(':','').replace('-','').replace('+00:00','Z')}",
        "timestamp_utc": started, "git_commit": git("rev-parse", "HEAD"),
        "dirty_worktree": bool(git("status", "--porcelain")), "variant": args.variant,
        "kernel_names": sorted({r["variant"] + "_" + r["component"] for r in rows}),
        "sequence_length": args.sequence_length, "block_size": summary.get("block_size"),
        "logical_page_count": (args.sequence_length // summary.get("block_size", 1)
                               if summary.get("block_size") else None),
        "block_table": summary.get("block_table"),
        "block_table_is_identity": summary.get("block_table_is_identity"),
        "storage_dtype": "FP16", "arithmetic_dtype": "FP32",
        "head_configuration": {"query_heads": 16, "kv_heads": 4, "head_dimension": 64},
        "latent_width": 32, "warmup_count": args.warmup, "iteration_count": args.iterations,
        "process_index": None, "execution_order": [r["variant"] + "_" + r["component"] for r in rows],
        "timing_method": args.timing_method, "timing_boundary": args.timing_boundary,
        **env, "compute_capability": None, "cutile_version": None,
        "command": [shlex.join(args.command)],
        "wsl_nvidia_driver_dir": wsl_dir,
        "null_reasons": {k: "not exposed by launcher environment" for k, v in env.items() if v is None},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "artifact_checksums.sha256")
    (out / "artifact_checksums.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), encoding="utf-8")
    return 0

if __name__ == "__main__": sys.exit(main())
