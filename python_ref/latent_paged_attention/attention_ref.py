"""NumPy decode-attention references for shape and correctness checks."""

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def paged_lookup_ref(
    paged_blocks: np.ndarray, block_indices: np.ndarray, seq_len: int
) -> np.ndarray:
    """
    Materialize a token-major view from paged storage.

    Args:
        paged_blocks: [num_blocks, block_size, n_kv_heads, head_dim]
        block_indices: [num_logical_blocks]
        seq_len: number of valid token positions
    Returns:
        Array with shape [seq_len, n_kv_heads, head_dim]
    """
    if paged_blocks.ndim != 4:
        raise ValueError(
            "paged_blocks must have shape [num_blocks, block_size, n_kv_heads, head_dim]"
        )
    if block_indices.ndim != 1:
        raise ValueError("block_indices must be 1D")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    gathered = paged_blocks[block_indices]
    token_major = gathered.reshape(-1, *paged_blocks.shape[2:])
    return token_major[:seq_len]


def gqa_decode_attention_ref(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    Reference GQA decode attention.

    Args:
        query: [n_q_heads, head_dim]
        keys: [seq_len, n_kv_heads, head_dim]
        values: [seq_len, n_kv_heads, head_dim]
    Returns:
        Context: [n_q_heads, head_dim]
    """
    if query.ndim != 2 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError(
            "expected query [n_q_heads, head_dim], "
            "keys/values [seq_len, n_kv_heads, head_dim]"
        )
    if keys.shape != values.shape:
        raise ValueError("keys and values must share the same shape")
    if query.shape[1] != keys.shape[2]:
        raise ValueError("query head_dim must match key/value head_dim")
    if query.shape[0] % keys.shape[1] != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads for GQA grouping")

    n_q_heads, head_dim = query.shape
    n_kv_heads = keys.shape[1]
    q_per_kv = n_q_heads // n_kv_heads

    q_grouped = query.reshape(n_kv_heads, q_per_kv, head_dim)
    scores = np.einsum("kgh,skh->kgs", q_grouped, keys) / np.sqrt(head_dim)
    probs = _softmax(scores, axis=-1)
    context = np.einsum("kgs,skh->kgh", probs, values)
    return context.reshape(n_q_heads, head_dim)


def latent_kv_decode_attention_ref(
    query: np.ndarray,
    latent_cache: np.ndarray,
    k_proj: np.ndarray,
    v_proj: np.ndarray,
) -> np.ndarray:
    """
    Reference latent-KV decode attention with reconstructed keys/values.

    Args:
        query: [n_q_heads, head_dim]
        latent_cache: [seq_len, latent_dim]
        k_proj: [latent_dim, n_kv_heads, head_dim]
        v_proj: [latent_dim, n_kv_heads, head_dim]
    Returns:
        Context: [n_q_heads, head_dim]
    """
    if latent_cache.ndim != 2:
        raise ValueError("latent_cache must have shape [seq_len, latent_dim]")
    if k_proj.shape != v_proj.shape or k_proj.ndim != 3:
        raise ValueError("k_proj and v_proj must have shape [latent_dim, n_kv_heads, head_dim]")
    if latent_cache.shape[1] != k_proj.shape[0]:
        raise ValueError("latent_dim mismatch between cache and projections")
    if query.shape[1] != k_proj.shape[2]:
        raise ValueError("query head_dim must match projection head_dim")

    keys = np.einsum("sl,lkh->skh", latent_cache, k_proj)
    values = np.einsum("sl,lkh->skh", latent_cache, v_proj)
    return gqa_decode_attention_ref(query, keys, values)
