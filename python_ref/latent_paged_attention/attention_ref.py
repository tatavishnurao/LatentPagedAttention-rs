"""Small NumPy references for decode-attention correctness experiments."""

import numpy as np


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along the chosen axis."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _validate_group_mapping(q_heads: int, kv_heads: int, group_size: int) -> None:
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if q_heads != kv_heads * group_size:
        raise ValueError(
            "expected q_heads == kv_heads * group_size, "
            f"got q_heads={q_heads}, kv_heads={kv_heads}, group_size={group_size}"
        )


def paged_lookup_ref(
    cache_blocks: np.ndarray,
    block_table: np.ndarray,
    seq_len: int,
    block_size: int,
) -> np.ndarray:
    """
    Materialize a token-major view from paged storage.

    Args:
        cache_blocks: [num_blocks, block_size, ...]
        block_table: [num_logical_blocks]
        seq_len: number of valid token positions
        block_size: tokens per block
    Returns:
        Array with shape [seq_len, ...]
    """
    if cache_blocks.ndim < 3:
        raise ValueError(
            "cache_blocks must have shape [num_blocks, block_size, ...] with ndim >= 3"
        )
    if block_table.ndim != 1:
        raise ValueError("block_table must be 1D")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if cache_blocks.shape[1] != block_size:
        raise ValueError(
            f"block_size mismatch: cache has {cache_blocks.shape[1]}, argument was {block_size}"
        )
    if block_table.size == 0:
        raise ValueError("block_table must not be empty")
    if np.any(block_table < 0) or np.any(block_table >= cache_blocks.shape[0]):
        raise ValueError("block_table contains out-of-range physical block indices")
    if block_table.size * block_size < seq_len:
        raise ValueError("block_table does not cover the requested seq_len")

    gathered = cache_blocks[block_table]
    token_major = gathered.reshape(-1, *cache_blocks.shape[2:])
    return token_major[:seq_len]


def gqa_decode_attention_ref(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    *,
    group_size: int,
) -> np.ndarray:
    """
    Reference GQA decode attention.

    Args:
        q: [batch, q_heads, head_dim]
        k_cache: [batch, seq_len, kv_heads, head_dim]
        v_cache: [batch, seq_len, kv_heads, head_dim]
        group_size: q_head grouping factor such that kv_head = q_head // group_size
    Returns:
        Context: [batch, q_heads, head_dim]
    """
    _, _, context = gqa_decode_attention_intermediates_ref(
        q, k_cache, v_cache, group_size=group_size
    )
    return context


def gqa_decode_attention_intermediates_ref(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    *,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference GQA decode attention with intermediates.

    Args:
        q: [batch, q_heads, head_dim]
        k_cache: [batch, seq_len, kv_heads, head_dim]
        v_cache: [batch, seq_len, kv_heads, head_dim]
        group_size: q_head grouping factor such that kv_head = q_head // group_size
    Returns:
        scores: [batch, q_heads, seq_len]
        probabilities: [batch, q_heads, seq_len]
        context: [batch, q_heads, head_dim]
    """
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "expected q [batch, q_heads, head_dim], "
            "k_cache/v_cache [batch, seq_len, kv_heads, head_dim]"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must share the same shape")
    if q.shape[0] != k_cache.shape[0]:
        raise ValueError("batch size mismatch between q and cache tensors")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("head_dim mismatch between q and cache tensors")

    batch, q_heads, head_dim = q.shape
    _, seq_len, kv_heads, _ = k_cache.shape
    _validate_group_mapping(q_heads, kv_heads, group_size)

    q_to_kv = np.arange(q_heads) // group_size
    selected_k = k_cache[:, :, q_to_kv, :]
    selected_v = v_cache[:, :, q_to_kv, :]

    scores = np.einsum("bqh,btqh->bqt", q, selected_k) / np.sqrt(float(head_dim))
    probs = softmax_stable(scores, axis=-1)
    context = np.einsum("bqt,btqh->bqh", probs, selected_v)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if not np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    assert context.shape == (batch, q_heads, head_dim)
    assert seq_len == k_cache.shape[1]
    return scores, probs, context


def paged_gqa_decode_attention_ref(
    q: np.ndarray,
    k_blocks: np.ndarray,
    v_blocks: np.ndarray,
    block_table: np.ndarray,
    seq_len: int,
    block_size: int,
    *,
    group_size: int,
) -> np.ndarray:
    """
    Reference paged GQA decode attention.

    The current helper uses a single logical block table shared across the batch
    dimension and broadcasts the reconstructed cache to each batch item.
    """
    logical_k = paged_lookup_ref(k_blocks, block_table, seq_len, block_size)
    logical_v = paged_lookup_ref(v_blocks, block_table, seq_len, block_size)
    batch = q.shape[0]
    k_cache = np.broadcast_to(logical_k[None, ...], (batch, *logical_k.shape))
    v_cache = np.broadcast_to(logical_v[None, ...], (batch, *logical_v.shape))
    return gqa_decode_attention_ref(q, k_cache, v_cache, group_size=group_size)


def latent_kv_decode_attention_ref(
    q: np.ndarray,
    latent_cache: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    group_size: int,
) -> np.ndarray:
    """
    Reference latent-KV decode attention with reconstructed keys/values.

    Args:
        q: [batch, q_heads, head_dim]
        latent_cache: [batch, seq_len, latent_dim]
        k_proj: [latent_dim, kv_heads * head_dim]
        v_proj: [latent_dim, kv_heads * head_dim]
    Returns:
        Context: [batch, q_heads, head_dim]
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, q_heads, head_dim]")
    if latent_cache.ndim != 3:
        raise ValueError("latent_cache must have shape [batch, seq_len, latent_dim]")
    if k_proj.ndim != 2 or v_proj.ndim != 2 or k_proj.shape != v_proj.shape:
        raise ValueError("k_proj and v_proj must have shape [latent_dim, kv_heads * head_dim]")
    if latent_cache.shape[2] != k_proj.shape[0]:
        raise ValueError("latent_dim mismatch between cache and projections")
    if q.shape[1] != q_heads or q.shape[2] != head_dim:
        raise ValueError("q shape does not match q_heads/head_dim arguments")
    if k_proj.shape[1] != kv_heads * head_dim:
        raise ValueError("projection width must equal kv_heads * head_dim")
    _validate_group_mapping(q_heads, kv_heads, group_size)

    batch, seq_len, _ = latent_cache.shape
    k_cache = (latent_cache @ k_proj).reshape(batch, seq_len, kv_heads, head_dim)
    v_cache = (latent_cache @ v_proj).reshape(batch, seq_len, kv_heads, head_dim)
    return gqa_decode_attention_ref(q, k_cache, v_cache, group_size=group_size)


def paged_latent_kv_decode_attention_ref(
    q: np.ndarray,
    latent_blocks: np.ndarray,
    block_table: np.ndarray,
    seq_len: int,
    block_size: int,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    group_size: int,
) -> np.ndarray:
    """Reference paged latent-KV decode attention."""
    logical_latent = paged_lookup_ref(latent_blocks, block_table, seq_len, block_size)
    batch = q.shape[0]
    latent_cache = np.broadcast_to(logical_latent[None, ...], (batch, *logical_latent.shape))
    return latent_kv_decode_attention_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        group_size=group_size,
    )
