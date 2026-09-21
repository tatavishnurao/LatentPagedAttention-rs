import pytest
from latent_paged_attention.memory_model import (
    compression_ratio,
    estimate_total_kv_cache_bytes,
    kv_bytes_per_token_gqa,
    kv_bytes_per_token_latent,
)


def test_gqa_bytes_per_token_counts_k_and_v() -> None:
    assert kv_bytes_per_token_gqa(8, 128, 2) == 4096


def test_latent_bytes_per_token() -> None:
    assert kv_bytes_per_token_latent(512, 2) == 1024


def test_total_cache_bytes() -> None:
    assert estimate_total_kv_cache_bytes(28, 4096, 1, 4096) == 469762048


def test_compression_ratio() -> None:
    assert compression_ratio(4096, 1024) == 4.0


def test_latent_memory_is_smaller_than_full_gqa_when_latent_dim_is_smaller() -> None:
    kv_heads = 2
    head_dim = 8
    latent_dim = 6
    dtype_bytes = 2

    full_bytes = kv_bytes_per_token_gqa(kv_heads, head_dim, dtype_bytes)
    latent_bytes = kv_bytes_per_token_latent(latent_dim, dtype_bytes)

    assert latent_dim < 2 * kv_heads * head_dim
    assert latent_bytes < full_bytes


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_inputs_raise(value: int) -> None:
    with pytest.raises(ValueError):
        kv_bytes_per_token_gqa(value, 128, 2)
