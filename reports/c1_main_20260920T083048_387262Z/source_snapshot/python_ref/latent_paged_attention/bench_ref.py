"""Deterministic NumPy reference benchmarks for the four KV-cache variants."""

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .attention_ref import (
    gqa_decode_attention_ref,
    latent_kv_decode_attention_ref,
    paged_gqa_decode_attention_ref,
    paged_latent_kv_decode_attention_ref,
)
from .memory_model import (
    compression_ratio,
    estimate_total_kv_cache_bytes,
    kv_bytes_per_token_gqa,
    kv_bytes_per_token_latent,
)

SMALL_CONFIG = {
    "batch": 1,
    "seq_len": 128,
    "q_heads": 4,
    "kv_heads": 2,
    "head_dim": 32,
    "group_size": 2,
    "latent_dim": 32,
    "block_size": 16,
    "layers": 24,
    "dtype_bytes": 2,
}

MEDIUM_CONFIG = {
    "batch": 1,
    "seq_len": 1024,
    "q_heads": 8,
    "kv_heads": 4,
    "head_dim": 64,
    "group_size": 2,
    "latent_dim": 128,
    "block_size": 16,
    "layers": 28,
    "dtype_bytes": 2,
}

BENCHMARK_CONFIGS = {"small": SMALL_CONFIG, "medium": MEDIUM_CONFIG}
VARIANTS = ("GQA", "Paged GQA", "Latent KV", "Paged Latent KV")
CSV_FIELDS = [
    "variant",
    "config_name",
    "batch",
    "seq_len",
    "q_heads",
    "kv_heads",
    "head_dim",
    "latent_dim",
    "block_size",
    "layers",
    "iterations",
    "kv_bytes_per_token_per_layer",
    "total_kv_mib",
    "compression_ratio_vs_gqa",
    "avg_runtime_ms",
    "output_shape",
    "notes",
]

console = Console()
CSV_PATH_OPTION = typer.Option(None, "--csv", help="Write results to a CSV path.")


def get_benchmark_config(name: str) -> dict:
    """Return a copy of a named benchmark configuration."""
    key = name.lower()
    if key not in BENCHMARK_CONFIGS:
        available = ", ".join(sorted(BENCHMARK_CONFIGS))
        raise ValueError(f"unknown benchmark config {name!r}; expected one of: {available}")
    return dict(BENCHMARK_CONFIGS[key])


def _paged_storage(tokens: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Put token-major data into a deterministic non-contiguous physical layout."""
    seq_len = tokens.shape[0]
    num_blocks = (seq_len + block_size - 1) // block_size
    block_table = np.arange(num_blocks - 1, -1, -1, dtype=np.int64)
    padded = np.zeros((num_blocks * block_size, *tokens.shape[1:]), dtype=np.float32)
    padded[:seq_len] = tokens
    logical_blocks = padded.reshape(num_blocks, block_size, *tokens.shape[1:])
    physical_blocks = np.empty_like(logical_blocks)
    for logical_idx, physical_idx in enumerate(block_table):
        physical_blocks[physical_idx] = logical_blocks[logical_idx]
    return physical_blocks, block_table


def _make_workload(config: dict) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(20260710 + config["seq_len"] + config["q_heads"])
    batch = config["batch"]
    seq_len = config["seq_len"]
    q_heads = config["q_heads"]
    kv_heads = config["kv_heads"]
    head_dim = config["head_dim"]
    latent_dim = config["latent_dim"]
    q = rng.normal(size=(batch, q_heads, head_dim)).astype(np.float32)
    k_cache = rng.normal(size=(batch, seq_len, kv_heads, head_dim)).astype(np.float32)
    v_cache = rng.normal(size=(batch, seq_len, kv_heads, head_dim)).astype(np.float32)
    latent_cache = rng.normal(size=(batch, seq_len, latent_dim)).astype(np.float32)
    k_proj = rng.normal(size=(latent_dim, kv_heads * head_dim)).astype(np.float32)
    v_proj = rng.normal(size=(latent_dim, kv_heads * head_dim)).astype(np.float32)
    k_blocks, block_table = _paged_storage(k_cache[0], config["block_size"])
    v_blocks, _ = _paged_storage(v_cache[0], config["block_size"])
    latent_blocks, _ = _paged_storage(latent_cache[0], config["block_size"])
    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "latent_cache": latent_cache,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "k_blocks": k_blocks,
        "v_blocks": v_blocks,
        "latent_blocks": latent_blocks,
        "block_table": block_table,
    }


def _variant_callables(config: dict, data: dict) -> dict[str, Callable[[], np.ndarray]]:
    common = {
        "group_size": config["group_size"],
    }
    q = data["q"]
    return {
        "GQA": lambda: gqa_decode_attention_ref(
            q, data["k_cache"], data["v_cache"], **common
        ),
        "Paged GQA": lambda: paged_gqa_decode_attention_ref(
            q,
            data["k_blocks"],
            data["v_blocks"],
            data["block_table"],
            config["seq_len"],
            config["block_size"],
            **common,
        ),
        "Latent KV": lambda: latent_kv_decode_attention_ref(
            q,
            data["latent_cache"],
            data["k_proj"],
            data["v_proj"],
            q_heads=config["q_heads"],
            kv_heads=config["kv_heads"],
            head_dim=config["head_dim"],
            **common,
        ),
        "Paged Latent KV": lambda: paged_latent_kv_decode_attention_ref(
            q,
            data["latent_blocks"],
            data["block_table"],
            config["seq_len"],
            config["block_size"],
            data["k_proj"],
            data["v_proj"],
            q_heads=config["q_heads"],
            kv_heads=config["kv_heads"],
            head_dim=config["head_dim"],
            **common,
        ),
    }


def run_reference_benchmark(config_name: str, iters: int) -> list[dict]:
    """Run all variants and return metric rows for a named configuration."""
    if not isinstance(iters, int) or iters <= 0:
        raise ValueError(f"iters must be a positive integer, got {iters!r}")
    config = get_benchmark_config(config_name)
    data = _make_workload(config)
    calls = _variant_callables(config, data)
    gqa_bytes = kv_bytes_per_token_gqa(
        config["kv_heads"], config["head_dim"], config["dtype_bytes"]
    )
    latent_bytes = kv_bytes_per_token_latent(config["latent_dim"], config["dtype_bytes"])
    byte_counts = {"GQA": gqa_bytes, "Paged GQA": gqa_bytes,
                   "Latent KV": latent_bytes, "Paged Latent KV": latent_bytes}
    notes = {
        "GQA": "baseline full KV cache",
        "Paged GQA": (
            "same theoretical KV bytes/token as GQA; paging affects allocation/fragmentation, "
            "not raw bytes/token"
        ),
        "Latent KV": "stores compressed latent cache; reconstruction/projection adds compute",
        "Paged Latent KV": "combines paged layout with latent cache; still reference-level only",
    }
    rows = []
    for variant in VARIANTS:
        call = calls[variant]
        output = call()  # Warm up allocation and validation outside the timed loop.
        start = time.perf_counter()
        for _ in range(iters):
            output = call()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
        total_bytes = estimate_total_kv_cache_bytes(
            config["layers"], config["seq_len"], config["batch"], byte_counts[variant]
        )
        rows.append({
            **{key: config[key] for key in (
                "batch", "seq_len", "q_heads", "kv_heads", "head_dim", "latent_dim",
                "block_size", "layers"
            )},
            "variant": variant,
            "config_name": config_name.lower(),
            "iterations": iters,
            "kv_bytes_per_token_per_layer": byte_counts[variant],
            "total_kv_mib": total_bytes / (1024 ** 2),
            "compression_ratio_vs_gqa": compression_ratio(gqa_bytes, byte_counts[variant]),
            "avg_runtime_ms": elapsed_ms,
            "output_shape": str(tuple(int(dim) for dim in output.shape)),
            "notes": notes[variant],
        })
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    """Write benchmark rows, creating the destination directory if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict]) -> None:
    table = Table(title="NumPy reference runtime")
    columns = ("Variant", "KV bytes/token/layer", "Total KV MiB", "Compression", "Avg ms", "Output")
    for column in columns:
        table.add_column(column, justify="right" if column != "Variant" else "left")
    for row in rows:
        table.add_row(
            row["variant"], str(row["kv_bytes_per_token_per_layer"]),
            f"{row['total_kv_mib']:.3f}", f"{row['compression_ratio_vs_gqa']:.3f}x",
            f"{row['avg_runtime_ms']:.3f}", row["output_shape"],
        )
    console.print(table)
    console.print(
        "Timing is local NumPy reference runtime; it is not GPU or production inference timing."
    )


def benchmark(
    config: str = typer.Option("small", "--config"),
    iters: int = typer.Option(20, "--iters", min=1),
    csv_path: Path | None = CSV_PATH_OPTION,
) -> None:
    """Run the NumPy-only reference benchmark."""
    try:
        rows = run_reference_benchmark(config, iters)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--config/--iters") from error
    _print_table(rows)
    if csv_path is not None:
        write_csv(rows, csv_path)
        console.print(f"Wrote CSV: {csv_path}")


def main() -> None:
    typer.run(benchmark)


if __name__ == "__main__":
    main()
