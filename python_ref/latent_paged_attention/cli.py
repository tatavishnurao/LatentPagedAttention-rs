"""CLI for lightweight KV-cache memory estimates."""

import typer
from rich.console import Console
from rich.table import Table

from .memory_model import (
    compression_ratio,
    estimate_total_kv_cache_bytes,
    kv_bytes_per_token_gqa,
    kv_bytes_per_token_latent,
)

app = typer.Typer(add_completion=False)
console = Console()


def _format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


@app.command()
def main(
    layers: int = typer.Option(..., min=1),
    seq_len: int = typer.Option(..., min=1),
    batch_size: int = typer.Option(..., min=1),
    kv_heads: int = typer.Option(..., min=1),
    head_dim: int = typer.Option(..., min=1),
    latent_dim: int = typer.Option(..., min=1),
    dtype_bytes: int = typer.Option(..., min=1),
) -> None:
    """Print GQA KV vs latent KV memory estimates."""
    gqa_bytes_per_token = kv_bytes_per_token_gqa(kv_heads, head_dim, dtype_bytes)
    latent_bytes_per_token = kv_bytes_per_token_latent(latent_dim, dtype_bytes)

    gqa_total = estimate_total_kv_cache_bytes(layers, seq_len, batch_size, gqa_bytes_per_token)
    latent_total = estimate_total_kv_cache_bytes(
        layers, seq_len, batch_size, latent_bytes_per_token
    )

    table = Table(title="KV Cache Estimate")
    table.add_column("Variant")
    table.add_column("Bytes/token/layer", justify="right")
    table.add_column("Total bytes", justify="right")
    table.add_column("Total MiB", justify="right")
    table.add_row("GQA KV", str(gqa_bytes_per_token), str(gqa_total), _format_mib(gqa_total))
    table.add_row(
        "Latent KV", str(latent_bytes_per_token), str(latent_total), _format_mib(latent_total)
    )

    console.print(table)
    console.print(
        f"Compression ratio (full / latent): {compression_ratio(gqa_total, latent_total):.3f}x"
    )


def run() -> None:
    app()


if __name__ == "__main__":
    run()
