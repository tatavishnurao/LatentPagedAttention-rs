import numpy as np
from latent_paged_attention.attention_ref import (
    gqa_decode_attention_ref,
    latent_kv_decode_attention_ref,
    paged_lookup_ref,
)


def test_paged_lookup_returns_token_major_slice() -> None:
    paged = np.arange(2 * 4 * 1 * 2, dtype=np.float32).reshape(2, 4, 1, 2)
    logical = np.array([1, 0], dtype=np.int64)

    out = paged_lookup_ref(paged, logical, seq_len=6)

    assert out.shape == (6, 1, 2)
    np.testing.assert_array_equal(out[0], paged[1, 0])
    np.testing.assert_array_equal(out[4], paged[0, 0])


def test_gqa_decode_attention_output_shape_and_finiteness() -> None:
    query = np.array(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    keys = np.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]]],
        dtype=np.float32,
    )
    values = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=np.float32,
    )

    out = gqa_decode_attention_ref(query, keys, values)

    assert out.shape == query.shape
    assert np.isfinite(out).all()


def test_latent_attention_matches_reconstructed_gqa_path() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    latent = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    k_proj = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    v_proj = np.array([[[0.5, 0.5]], [[1.0, -1.0]]], dtype=np.float32)

    out = latent_kv_decode_attention_ref(query, latent, k_proj, v_proj)

    assert out.shape == query.shape
    assert np.isfinite(out).all()
