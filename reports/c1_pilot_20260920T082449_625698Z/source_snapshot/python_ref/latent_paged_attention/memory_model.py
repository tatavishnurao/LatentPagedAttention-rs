"""Memory accounting helpers for KV cache experiments."""


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def kv_bytes_per_token_gqa(
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    *,
    include_k_and_v: bool = True,
) -> int:
    """Return bytes per token per layer for a standard GQA KV cache."""
    _validate_positive_int("n_kv_heads", n_kv_heads)
    _validate_positive_int("head_dim", head_dim)
    _validate_positive_int("dtype_bytes", dtype_bytes)

    components = 2 if include_k_and_v else 1
    return n_kv_heads * head_dim * dtype_bytes * components


def kv_bytes_per_token_latent(latent_dim: int, dtype_bytes: int) -> int:
    """Return bytes per token per layer for a latent KV representation."""
    _validate_positive_int("latent_dim", latent_dim)
    _validate_positive_int("dtype_bytes", dtype_bytes)
    return latent_dim * dtype_bytes


def compression_ratio(full_kv_bytes: int, latent_kv_bytes: int) -> float:
    """Return full_kv_bytes / latent_kv_bytes."""
    _validate_positive_int("full_kv_bytes", full_kv_bytes)
    _validate_positive_int("latent_kv_bytes", latent_kv_bytes)
    return full_kv_bytes / latent_kv_bytes


def estimate_total_kv_cache_bytes(
    num_layers: int,
    seq_len: int,
    batch_size: int,
    bytes_per_token_per_layer: int,
) -> int:
    """Return total KV cache bytes across layers, sequence length, and batch."""
    _validate_positive_int("num_layers", num_layers)
    _validate_positive_int("seq_len", seq_len)
    _validate_positive_int("batch_size", batch_size)
    _validate_positive_int("bytes_per_token_per_layer", bytes_per_token_per_layer)
    return num_layers * seq_len * batch_size * bytes_per_token_per_layer
