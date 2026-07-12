"""Reference helpers for paged KV-cache mutation."""

import numpy as np


def fp16_storage_roundtrip_f32(values: np.ndarray) -> np.ndarray:
    """Return values after deterministic IEEE FP16 storage as FP32."""
    if not np.isfinite(values).all():
        raise ValueError("FP16 storage input must be finite")
    with np.errstate(over="ignore"):
        stored = values.astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError("FP16 storage conversion overflowed")
    return stored.astype(np.float32)


def fp16_storage_bits(values: np.ndarray) -> np.ndarray:
    """Return IEEE FP16 storage bit patterns as uint16."""
    return fp16_storage_roundtrip_f32(values).astype(np.float16).view(np.uint16)


def resolve_paged_token_location(
    block_table: np.ndarray,
    token_position: int,
    block_size: int,
) -> tuple[int, int, int]:
    """Return logical block, physical block, and in-block offset."""
    if block_table.ndim != 1:
        raise ValueError("block_table must be 1D")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if token_position < 0:
        raise ValueError(f"token_position must be >= 0, got {token_position}")
    logical_block = token_position // block_size
    if logical_block >= block_table.size:
        raise IndexError("block_table does not cover token_position")
    physical_block = int(block_table[logical_block])
    if physical_block < 0:
        raise ValueError("block_table contains a negative physical block")
    return logical_block, physical_block, token_position % block_size


def paged_kv_write_ref(
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    block_table: np.ndarray,
    token_position: int,
    new_k: np.ndarray,
    new_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Write one K/V row into copies of paged physical caches."""
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("K and V caches must have shape [blocks, block_size, width]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes must match")
    if block_table.ndim != 1:
        raise ValueError("block_table must be 1D")
    if new_k.ndim != 1 or new_v.ndim != 1:
        raise ValueError("new_k and new_v must be 1D")
    if new_k.shape != new_v.shape or new_k.shape[0] != k_cache.shape[2]:
        raise ValueError("new K/V vectors must have the cache width")
    logical_block, physical_block, block_offset = resolve_paged_token_location(
        block_table, token_position, k_cache.shape[1]
    )
    del logical_block
    if physical_block >= k_cache.shape[0]:
        raise IndexError("block_table contains an out-of-range physical block")

    updated_k = np.array(k_cache, copy=True)
    updated_v = np.array(v_cache, copy=True)
    updated_k[physical_block, block_offset, :] = new_k
    updated_v[physical_block, block_offset, :] = new_v
    return updated_k, updated_v


def paged_latent_write_ref(
    latent_cache: np.ndarray,
    block_table: np.ndarray,
    token_position: int,
    new_latent: np.ndarray,
) -> np.ndarray:
    """Write one latent row into a copy of a physical paged latent cache."""
    if latent_cache.ndim != 3:
        raise ValueError("latent_cache must have shape [blocks, block_size, latent_dim]")
    if block_table.ndim != 1:
        raise ValueError("block_table must be 1D")
    if new_latent.ndim != 1:
        raise ValueError("new_latent must be 1D")
    if new_latent.shape[0] != latent_cache.shape[2]:
        raise ValueError("new_latent width must match latent_dim")
    if not np.isfinite(latent_cache).all() or not np.isfinite(new_latent).all():
        raise ValueError("latent cache and replacement must be finite")
    _, physical_block, block_offset = resolve_paged_token_location(
        block_table, token_position, latent_cache.shape[1]
    )
    if physical_block >= latent_cache.shape[0]:
        raise IndexError("block_table contains an out-of-range physical block")
    updated = np.array(latent_cache, copy=True)
    updated[physical_block, block_offset, :] = new_latent
    return updated


def paged_latent_write_fp16_storage_ref(
    latent_cache_f32: np.ndarray,
    block_table: np.ndarray,
    token_position: int,
    new_latent_f32: np.ndarray,
) -> np.ndarray:
    """Write one row into an FP16-stored latent cache, returned as FP32."""
    stored_cache = fp16_storage_roundtrip_f32(latent_cache_f32)
    stored_new_latent = fp16_storage_roundtrip_f32(new_latent_f32)
    return paged_latent_write_ref(
        stored_cache, block_table, token_position, stored_new_latent
    )
