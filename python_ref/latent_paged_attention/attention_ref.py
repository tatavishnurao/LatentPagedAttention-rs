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
    _, _, context = paged_gqa_decode_attention_intermediates_ref(
        q,
        k_blocks,
        v_blocks,
        block_table,
        seq_len,
        block_size,
        group_size=group_size,
    )
    return context


def paged_gqa_decode_attention_intermediates_ref(
    q: np.ndarray,
    k_blocks: np.ndarray,
    v_blocks: np.ndarray,
    block_table: np.ndarray,
    seq_len: int,
    block_size: int,
    *,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference paged GQA decode attention with intermediates.

    Args:
        q: [batch, q_heads, head_dim]
        k_blocks: [num_physical_blocks, block_size, kv_heads, head_dim]
        v_blocks: [num_physical_blocks, block_size, kv_heads, head_dim]
        block_table: [num_logical_blocks]
    Returns:
        scores: [batch, q_heads, seq_len]
        probabilities: [batch, q_heads, seq_len]
        context: [batch, q_heads, head_dim]
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, q_heads, head_dim]")
    if k_blocks.ndim != 4 or v_blocks.ndim != 4:
        raise ValueError(
            "k_blocks and v_blocks must have shape "
            "[num_physical_blocks, block_size, kv_heads, head_dim]"
        )
    if k_blocks.shape != v_blocks.shape:
        raise ValueError("k_blocks and v_blocks must share the same shape")
    if block_table.ndim != 1:
        raise ValueError("block_table must be 1D")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if k_blocks.shape[1] != block_size:
        raise ValueError(
            f"block_size mismatch: cache has {k_blocks.shape[1]}, argument was {block_size}"
        )
    if q.shape[2] != k_blocks.shape[3]:
        raise ValueError("head_dim mismatch between q and paged cache tensors")
    if block_table.size * block_size < seq_len:
        raise ValueError("block_table does not cover the requested seq_len")

    logical_k = paged_lookup_ref(k_blocks, block_table, seq_len, block_size)
    logical_v = paged_lookup_ref(v_blocks, block_table, seq_len, block_size)
    batch = q.shape[0]
    k_cache = np.broadcast_to(logical_k[None, ...], (batch, *logical_k.shape))
    v_cache = np.broadcast_to(logical_v[None, ...], (batch, *logical_v.shape))
    return gqa_decode_attention_intermediates_ref(q, k_cache, v_cache, group_size=group_size)


def latent_kv_reconstruction_ref(
    latent_cache: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    *,
    kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct K and V from a latent cache.

    Args:
        latent_cache: [batch, seq_len, latent_dim]
        k_proj: [latent_dim, kv_heads * head_dim]
        v_proj: [latent_dim, kv_heads * head_dim]
    Returns:
        k_cache: [batch, seq_len, kv_heads, head_dim]
        v_cache: [batch, seq_len, kv_heads, head_dim]
    """
    if latent_cache.ndim != 3:
        raise ValueError("latent_cache must have shape [batch, seq_len, latent_dim]")
    if k_proj.ndim != 2 or v_proj.ndim != 2:
        raise ValueError("k_proj and v_proj must have shape [latent_dim, kv_heads * head_dim]")
    if k_proj.shape != v_proj.shape:
        raise ValueError("k_proj and v_proj shapes must match")
    if kv_heads <= 0:
        raise ValueError(f"kv_heads must be > 0, got {kv_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}")

    batch, seq_len, latent_dim = latent_cache.shape
    if batch <= 0 or seq_len <= 0 or latent_dim <= 0:
        raise ValueError("latent_cache dimensions must be > 0")
    if k_proj.shape[0] != latent_dim:
        raise ValueError("latent_dim mismatch between cache and projections")
    projection_width = kv_heads * head_dim
    if k_proj.shape[1] != projection_width:
        raise ValueError("projection width must equal kv_heads * head_dim")

    k_flat = np.asarray(latent_cache @ k_proj, dtype=np.float32)
    v_flat = np.asarray(latent_cache @ v_proj, dtype=np.float32)
    k_cache = k_flat.reshape(batch, seq_len, kv_heads, head_dim)
    v_cache = v_flat.reshape(batch, seq_len, kv_heads, head_dim)
    if not np.all(np.isfinite(k_cache)):
        raise ValueError("reconstructed K cache must be finite")
    if not np.all(np.isfinite(v_cache)):
        raise ValueError("reconstructed V cache must be finite")
    return k_cache, v_cache


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
    _, _, context = latent_kv_decode_attention_intermediates_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        group_size=group_size,
    )
    return context


def latent_kv_decode_attention_intermediates_ref(
    q: np.ndarray,
    latent_cache: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Materialized correctness oracle.

    Reconstruct full K/V using latent_kv_reconstruction_ref,
    then run gqa_decode_attention_intermediates_ref.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, q_heads, head_dim]")
    if q.shape[1] != q_heads or q.shape[2] != head_dim:
        raise ValueError("q shape does not match q_heads/head_dim arguments")
    _validate_group_mapping(q_heads, kv_heads, group_size)

    k_cache, v_cache = latent_kv_reconstruction_ref(
        latent_cache,
        k_proj,
        v_proj,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    return gqa_decode_attention_intermediates_ref(q, k_cache, v_cache, group_size=group_size)


def direct_latent_gqa_decode_attention_intermediates_ref(
    q: np.ndarray,
    latent_cache: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct latent-space GQA without materializing full K/V.

    Returns:
        scores: [batch, q_heads, seq_len]
        probabilities: [batch, q_heads, seq_len]
        context: [batch, q_heads, head_dim]
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, q_heads, head_dim]")
    if latent_cache.ndim != 3:
        raise ValueError("latent_cache must have shape [batch, seq_len, latent_dim]")
    if k_proj.ndim != 2 or v_proj.ndim != 2:
        raise ValueError("k_proj and v_proj must have shape [latent_dim, kv_heads * head_dim]")
    if k_proj.shape != v_proj.shape:
        raise ValueError("k_proj and v_proj shapes must match")
    if q.shape[1] != q_heads or q.shape[2] != head_dim:
        raise ValueError("q shape does not match q_heads/head_dim arguments")

    batch, seq_len, latent_dim = latent_cache.shape
    if batch <= 0 or seq_len <= 0 or latent_dim <= 0:
        raise ValueError("latent_cache dimensions must be > 0")
    if q.shape[0] != batch:
        raise ValueError("batch size mismatch between q and latent_cache tensors")
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("kv_heads and head_dim must be > 0")
    _validate_group_mapping(q_heads, kv_heads, group_size)
    if k_proj.shape[0] != latent_dim:
        raise ValueError("latent_dim mismatch between cache and projections")
    projection_width = kv_heads * head_dim
    if k_proj.shape[1] != projection_width:
        raise ValueError("projection width must equal kv_heads * head_dim")

    k_proj_head_major = np.asarray(k_proj.reshape(latent_dim, kv_heads, head_dim), dtype=np.float32)
    v_proj_head_major = np.asarray(v_proj.reshape(latent_dim, kv_heads, head_dim), dtype=np.float32)
    latent_cache = np.asarray(latent_cache, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    scores = np.empty((batch, q_heads, seq_len), dtype=np.float32)
    probabilities = np.empty_like(scores)
    context = np.empty((batch, q_heads, head_dim), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(float(head_dim)))

    for q_head in range(q_heads):
        kv_head = q_head // group_size
        projected_query_latent = np.einsum(
            "ld,bd->bl", k_proj_head_major[:, kv_head, :], q[:, q_head, :], optimize=True
        ).astype(np.float32, copy=False)
        row_scores = np.einsum("bsl,bl->bs", latent_cache, projected_query_latent, optimize=True)
        row_scores = np.asarray(row_scores * scale, dtype=np.float32)
        row_probs = softmax_stable(row_scores, axis=-1).astype(np.float32, copy=False)
        latent_context = np.einsum("bs,bsl->bl", row_probs, latent_cache, optimize=True)
        row_context = np.einsum(
            "bl,ld->bd", latent_context, v_proj_head_major[:, kv_head, :], optimize=True
        ).astype(np.float32, copy=False)
        scores[:, q_head, :] = row_scores
        probabilities[:, q_head, :] = row_probs
        context[:, q_head, :] = row_context

    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if not np.all(np.isfinite(context)):
        raise ValueError("context must be finite")
    if not np.allclose(np.sum(probabilities, axis=-1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    return scores, probabilities, context


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
