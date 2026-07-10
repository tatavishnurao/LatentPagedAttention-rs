import numpy as np
import pytest
from latent_paged_attention.cache_ref import (
    paged_kv_write_ref,
    resolve_paged_token_location,
)


def make_caches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_cache = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    v_cache = (np.arange(4 * 2 * 8, dtype=np.float32) + 1000).reshape(4, 2, 8)
    return k_cache, v_cache, np.asarray([2, 0, 1], dtype=np.int32)


@pytest.mark.parametrize(("token_position", "expected"), [(3, (1, 0, 1)), (4, (2, 1, 0))])
def test_resolve_paged_token_location(token_position: int, expected: tuple[int, int, int]) -> None:
    _, _, block_table = make_caches()
    assert resolve_paged_token_location(block_table, token_position, 2) == expected


@pytest.mark.parametrize(
    ("token_position", "new_k_base", "new_v_base"), [(3, 100, 200), (4, 300, 400)]
)
def test_paged_kv_write_changes_exactly_one_row(
    token_position: int, new_k_base: int, new_v_base: int
) -> None:
    k_cache, v_cache, block_table = make_caches()
    original_k = k_cache.copy()
    original_v = v_cache.copy()
    new_k = np.arange(new_k_base, new_k_base + 8, dtype=np.float32)
    new_v = np.arange(new_v_base, new_v_base + 8, dtype=np.float32)
    updated_k, updated_v = paged_kv_write_ref(
        k_cache, v_cache, block_table, token_position, new_k, new_v
    )
    _, physical_block, offset = resolve_paged_token_location(block_table, token_position, 2)
    np.testing.assert_array_equal(updated_k[physical_block, offset], new_k)
    np.testing.assert_array_equal(updated_v[physical_block, offset], new_v)
    assert np.count_nonzero(updated_k != original_k) == 8
    assert np.count_nonzero(updated_v != original_v) == 8
    assert np.array_equal(k_cache, original_k)
    assert np.array_equal(v_cache, original_v)


def test_paged_kv_write_rejects_invalid_inputs() -> None:
    k_cache, v_cache, block_table = make_caches()
    with pytest.raises(IndexError):
        paged_kv_write_ref(k_cache, v_cache, block_table, 6, np.zeros(8), np.zeros(8))
    with pytest.raises(IndexError):
        paged_kv_write_ref(k_cache, v_cache, np.asarray([4, 0, 1]), 0, np.zeros(8), np.zeros(8))
    with pytest.raises(ValueError):
        paged_kv_write_ref(k_cache, v_cache[:, :, :-1], block_table, 0, np.zeros(8), np.zeros(8))
    with pytest.raises(ValueError):
        paged_kv_write_ref(k_cache, v_cache, block_table, 0, np.zeros(7), np.zeros(8))
    with pytest.raises(ValueError):
        resolve_paged_token_location(block_table, 0, 0)
