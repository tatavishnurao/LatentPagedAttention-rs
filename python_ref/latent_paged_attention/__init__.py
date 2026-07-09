"""Reference models for LatentPagedAttention-rs."""

from .attention_ref import (
    gqa_decode_attention_ref,
    latent_kv_decode_attention_ref,
    paged_gqa_decode_attention_ref,
    paged_latent_kv_decode_attention_ref,
    paged_lookup_ref,
    softmax_stable,
)
from .block_table import PagedBlockTable, logical_blocks_for_seq_len
from .memory_model import (
    compression_ratio,
    estimate_total_kv_cache_bytes,
    kv_bytes_per_token_gqa,
    kv_bytes_per_token_latent,
)

__all__ = [
    "PagedBlockTable",
    "compression_ratio",
    "estimate_total_kv_cache_bytes",
    "gqa_decode_attention_ref",
    "kv_bytes_per_token_gqa",
    "kv_bytes_per_token_latent",
    "latent_kv_decode_attention_ref",
    "logical_blocks_for_seq_len",
    "paged_gqa_decode_attention_ref",
    "paged_latent_kv_decode_attention_ref",
    "paged_lookup_ref",
    "softmax_stable",
]
