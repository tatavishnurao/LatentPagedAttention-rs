import csv

import pytest
from latent_paged_attention.bench_ref import (
    CSV_FIELDS,
    run_reference_benchmark,
    write_csv,
)
from latent_paged_attention.memory_model import kv_bytes_per_token_gqa


def test_reference_benchmark_returns_four_rows_with_expected_shapes() -> None:
    rows = run_reference_benchmark("small", iters=1)

    assert [row["variant"] for row in rows] == ["GQA", "Paged GQA", "Latent KV", "Paged Latent KV"]
    expected_shape = "(1, 4, 32)"
    assert all(row["avg_runtime_ms"] >= 0 for row in rows)
    assert all(row["output_shape"] == expected_shape for row in rows)


@pytest.mark.parametrize("config_name", ["small", "medium"])
def test_reference_benchmark_memory_relationships_hold(config_name: str) -> None:
    rows = {row["variant"]: row for row in run_reference_benchmark(config_name, iters=1)}
    gqa_bytes = rows["GQA"]["kv_bytes_per_token_per_layer"]

    assert rows["Latent KV"]["kv_bytes_per_token_per_layer"] < gqa_bytes
    assert rows["Paged GQA"]["kv_bytes_per_token_per_layer"] == gqa_bytes
    assert (
        rows["Paged Latent KV"]["kv_bytes_per_token_per_layer"]
        == rows["Latent KV"]["kv_bytes_per_token_per_layer"]
    )
    assert gqa_bytes == kv_bytes_per_token_gqa(rows["GQA"]["kv_heads"], rows["GQA"]["head_dim"], 2)


def test_write_csv_creates_file_and_expected_headers(tmp_path) -> None:
    rows = run_reference_benchmark("small", iters=1)
    path = tmp_path / "nested" / "reference.csv"

    write_csv(rows, path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_FIELDS
        assert len(list(reader)) == 4


def test_unknown_config_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown benchmark config"):
        run_reference_benchmark("unknown", iters=1)
