import numpy as np
from latent_paged_attention.attention_ref import (
    direct_latent_gqa_decode_attention_intermediates_ref,
    gqa_decode_attention_intermediates_ref,
    gqa_decode_attention_ref,
    latent_kv_decode_attention_intermediates_ref,
    latent_kv_decode_attention_ref,
    latent_kv_reconstruction_ref,
    paged_gqa_decode_attention_intermediates_ref,
    paged_gqa_decode_attention_ref,
    paged_latent_kv_decode_attention_ref,
    paged_lookup_ref,
    softmax_stable,
)


def _make_tiny_case() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(7)
    batch = 1
    seq_len = 5
    q_heads = 4
    kv_heads = 2
    head_dim = 8
    latent_dim = 6

    q = rng.normal(size=(batch, q_heads, head_dim)).astype(np.float32)
    k_cache = rng.normal(size=(batch, seq_len, kv_heads, head_dim)).astype(np.float32)
    v_cache = rng.normal(size=(batch, seq_len, kv_heads, head_dim)).astype(np.float32)
    latent_cache = rng.normal(size=(batch, seq_len, latent_dim)).astype(np.float32)
    k_proj = rng.normal(size=(latent_dim, kv_heads * head_dim)).astype(np.float32)
    v_proj = rng.normal(size=(latent_dim, kv_heads * head_dim)).astype(np.float32)
    return q, k_cache, v_cache, latent_cache, k_proj, v_proj


def test_softmax_stable_normalizes_last_axis() -> None:
    x = np.array([[1000.0, 1001.0, 1002.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    out = softmax_stable(x, axis=-1)

    np.testing.assert_allclose(out.sum(axis=-1), np.ones(2, dtype=np.float32), atol=1e-6)


def test_paged_lookup_reconstructs_expected_token_order() -> None:
    paged = np.arange(3 * 2 * 2 * 3, dtype=np.float32).reshape(3, 2, 2, 3)
    block_table = np.array([2, 0, 1], dtype=np.int64)

    out = paged_lookup_ref(paged, block_table, seq_len=5, block_size=2)

    assert out.shape == (5, 2, 3)
    np.testing.assert_array_equal(out[0], paged[2, 0])
    np.testing.assert_array_equal(out[1], paged[2, 1])
    np.testing.assert_array_equal(out[2], paged[0, 0])
    np.testing.assert_array_equal(out[4], paged[1, 0])


def test_gqa_decode_attention_output_shape_is_correct() -> None:
    q, k_cache, v_cache, _, _, _ = _make_tiny_case()
    out = gqa_decode_attention_ref(q, k_cache, v_cache, group_size=2)

    assert out.shape == q.shape
    assert np.isfinite(out).all()


def test_gqa_decode_attention_intermediates_are_correct() -> None:
    q, k_cache, v_cache, _, _, _ = _make_tiny_case()
    scores, probs, context = gqa_decode_attention_intermediates_ref(
        q, k_cache, v_cache, group_size=2
    )

    assert scores.shape == (1, 4, 5)
    assert probs.shape == (1, 4, 5)
    assert context.shape == q.shape
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones((1, 4)), atol=1e-6)
    np.testing.assert_allclose(context, gqa_decode_attention_ref(q, k_cache, v_cache, group_size=2))

    scale = 1.0 / np.sqrt(float(q.shape[-1]))
    np.testing.assert_allclose(
        scores[0, 0, 0],
        np.dot(q[0, 0], k_cache[0, 0, 0]) * scale,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        scores[0, 2, 0],
        np.dot(q[0, 2], k_cache[0, 0, 1]) * scale,
        atol=1e-6,
    )


def test_gqa_decode_attention_stable_softmax_handles_large_scores() -> None:
    q = np.full((1, 4, 8), 80.0, dtype=np.float32)
    k_cache = np.arange(1 * 8 * 2 * 8, dtype=np.float32).reshape(1, 8, 2, 8)
    v_cache = np.linspace(-1.0, 1.0, 1 * 8 * 2 * 8, dtype=np.float32).reshape(1, 8, 2, 8)

    scores, probs, context = gqa_decode_attention_intermediates_ref(
        q, k_cache, v_cache, group_size=2
    )

    assert np.isfinite(scores).all()
    assert np.isfinite(probs).all()
    assert np.isfinite(context).all()
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones((1, 4)), atol=1e-6)


def test_gqa_decode_attention_rejects_invalid_shapes() -> None:
    q, k_cache, v_cache, _, _, _ = _make_tiny_case()

    with np.testing.assert_raises(ValueError):
        gqa_decode_attention_intermediates_ref(q[:, :3], k_cache, v_cache, group_size=2)
    with np.testing.assert_raises(ValueError):
        gqa_decode_attention_intermediates_ref(q, k_cache[..., :7], v_cache, group_size=2)
    with np.testing.assert_raises(ValueError):
        gqa_decode_attention_intermediates_ref(q, k_cache, v_cache[:, :, :1], group_size=2)


def test_paged_gqa_matches_dense_gqa_for_same_cache() -> None:
    q, k_cache, v_cache, _, _, _ = _make_tiny_case()
    block_size = 2
    seq_len = k_cache.shape[1]
    block_table = np.array([1, 0, 2], dtype=np.int64)

    pad_tokens = block_size * len(block_table) - seq_len
    k_tokens = np.pad(k_cache[0], ((0, pad_tokens), (0, 0), (0, 0)))
    v_tokens = np.pad(v_cache[0], ((0, pad_tokens), (0, 0), (0, 0)))
    k_blocks = np.empty((len(block_table), block_size, *k_tokens.shape[1:]), dtype=np.float32)
    v_blocks = np.empty((len(block_table), block_size, *v_tokens.shape[1:]), dtype=np.float32)
    logical_k_blocks = k_tokens.reshape(len(block_table), block_size, *k_tokens.shape[1:])
    logical_v_blocks = v_tokens.reshape(len(block_table), block_size, *v_tokens.shape[1:])
    for logical_idx, physical_idx in enumerate(block_table):
        k_blocks[physical_idx] = logical_k_blocks[logical_idx]
        v_blocks[physical_idx] = logical_v_blocks[logical_idx]

    dense_out = gqa_decode_attention_ref(q, k_cache, v_cache, group_size=2)
    paged_out = paged_gqa_decode_attention_ref(
        q,
        k_blocks,
        v_blocks,
        block_table,
        seq_len,
        block_size,
        group_size=2,
    )

    np.testing.assert_allclose(paged_out, dense_out, atol=1e-6)


def test_paged_gqa_intermediates_match_dense_gqa_for_same_cache() -> None:
    q, k_cache, v_cache, _, _, _ = _make_tiny_case()
    block_size = 2
    seq_len = k_cache.shape[1]
    block_table = np.array([1, 0, 2], dtype=np.int64)
    pad_tokens = block_size * len(block_table) - seq_len
    k_tokens = np.pad(k_cache[0], ((0, pad_tokens), (0, 0), (0, 0)))
    v_tokens = np.pad(v_cache[0], ((0, pad_tokens), (0, 0), (0, 0)))
    k_blocks = np.empty((len(block_table), block_size, *k_tokens.shape[1:]), dtype=np.float32)
    v_blocks = np.empty((len(block_table), block_size, *v_tokens.shape[1:]), dtype=np.float32)
    logical_k_blocks = k_tokens.reshape(len(block_table), block_size, *k_tokens.shape[1:])
    logical_v_blocks = v_tokens.reshape(len(block_table), block_size, *v_tokens.shape[1:])
    for logical_idx, physical_idx in enumerate(block_table):
        k_blocks[physical_idx] = logical_k_blocks[logical_idx]
        v_blocks[physical_idx] = logical_v_blocks[logical_idx]

    dense_scores, dense_probs, dense_context = gqa_decode_attention_intermediates_ref(
        q, k_cache, v_cache, group_size=2
    )
    paged_scores, paged_probs, paged_context = paged_gqa_decode_attention_intermediates_ref(
        q,
        k_blocks,
        v_blocks,
        block_table,
        seq_len,
        block_size,
        group_size=2,
    )

    assert paged_scores.shape == (1, 4, seq_len)
    assert paged_probs.shape == (1, 4, seq_len)
    assert paged_context.shape == q.shape
    np.testing.assert_allclose(paged_scores, dense_scores, atol=1e-6)
    np.testing.assert_allclose(paged_probs, dense_probs, atol=1e-6)
    np.testing.assert_allclose(paged_context, dense_context, atol=1e-6)
    np.testing.assert_allclose(paged_probs.sum(axis=-1), np.ones((1, 4)), atol=1e-6)
    assert np.isfinite(paged_scores).all()
    assert np.isfinite(paged_probs).all()
    assert np.isfinite(paged_context).all()


def test_paged_gqa_intermediates_reject_invalid_shapes_and_mapping() -> None:
    q, k_cache, v_cache, _, _, _ = _make_tiny_case()
    block_table = np.array([0, 1, 2], dtype=np.int64)

    with np.testing.assert_raises(ValueError):
        paged_gqa_decode_attention_intermediates_ref(
            q,
            k_cache[0],
            v_cache[0],
            block_table,
            seq_len=5,
            block_size=2,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        paged_gqa_decode_attention_intermediates_ref(
            q,
            k_cache[0],
            v_cache[0, :, :1],
            block_table,
            seq_len=5,
            block_size=2,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        paged_gqa_decode_attention_intermediates_ref(
            q[:, :3],
            k_cache[0],
            v_cache[0],
            block_table,
            seq_len=5,
            block_size=2,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        paged_gqa_decode_attention_intermediates_ref(
            q,
            k_cache[0],
            v_cache[0],
            block_table[:2],
            seq_len=5,
            block_size=2,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        paged_gqa_decode_attention_intermediates_ref(
            q,
            k_cache[0],
            v_cache[0],
            block_table,
            seq_len=5,
            block_size=0,
            group_size=2,
        )


def test_latent_kv_decode_attention_output_shape_is_correct() -> None:
    q, _, _, latent_cache, k_proj, v_proj = _make_tiny_case()
    out = latent_kv_decode_attention_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )

    assert out.shape == q.shape
    assert np.isfinite(out).all()


def test_latent_kv_materialized_intermediates_use_reconstruction() -> None:
    q, _, _, latent_cache, k_proj, v_proj = _make_tiny_case()
    scores, probs, context = latent_kv_decode_attention_intermediates_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )

    assert scores.shape == (1, 4, 5)
    assert probs.shape == (1, 4, 5)
    assert context.shape == q.shape
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones((1, 4)), atol=1e-6)
    assert np.isfinite(scores).all()
    assert np.isfinite(probs).all()
    assert np.isfinite(context).all()


def test_direct_latent_gqa_matches_materialized_reference() -> None:
    q, _, _, latent_cache, k_proj, v_proj = _make_tiny_case()
    direct = direct_latent_gqa_decode_attention_intermediates_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )
    materialized = latent_kv_decode_attention_intermediates_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )

    np.testing.assert_allclose(direct[0], materialized[0], atol=1e-6)
    np.testing.assert_allclose(direct[1], materialized[1], atol=1e-6)
    np.testing.assert_allclose(direct[2], materialized[2], atol=1e-6)


def test_direct_latent_gqa_does_not_call_reconstruction_helper(monkeypatch) -> None:
    q, _, _, latent_cache, k_proj, v_proj = _make_tiny_case()

    def _boom(*args, **kwargs):  # noqa: ANN001, ANN002
        raise AssertionError("reconstruction helper should not be called")

    monkeypatch.setattr(
        "latent_paged_attention.attention_ref.latent_kv_reconstruction_ref", _boom
    )
    direct_latent_gqa_decode_attention_intermediates_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )


def test_direct_latent_gqa_rejects_invalid_inputs() -> None:
    q, _, _, latent_cache, k_proj, v_proj = _make_tiny_case()

    with np.testing.assert_raises(ValueError):
        direct_latent_gqa_decode_attention_intermediates_ref(
            q[:, :3], latent_cache, k_proj, v_proj, q_heads=4, kv_heads=2, head_dim=8, group_size=2
        )
    with np.testing.assert_raises(ValueError):
        direct_latent_gqa_decode_attention_intermediates_ref(
            q,
            latent_cache[0],
            k_proj,
            v_proj,
            q_heads=4,
            kv_heads=2,
            head_dim=8,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        direct_latent_gqa_decode_attention_intermediates_ref(
            q,
            latent_cache,
            k_proj[:, None],
            v_proj,
            q_heads=4,
            kv_heads=2,
            head_dim=8,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        direct_latent_gqa_decode_attention_intermediates_ref(
            q,
            latent_cache,
            k_proj,
            v_proj[:, :7],
            q_heads=4,
            kv_heads=2,
            head_dim=8,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        direct_latent_gqa_decode_attention_intermediates_ref(
            q,
            latent_cache,
            k_proj,
            v_proj,
            q_heads=3,
            kv_heads=2,
            head_dim=8,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        direct_latent_gqa_decode_attention_intermediates_ref(
            q,
            latent_cache,
            k_proj,
            v_proj,
            q_heads=4,
            kv_heads=0,
            head_dim=8,
            group_size=2,
        )
    with np.testing.assert_raises(ValueError):
        direct_latent_gqa_decode_attention_intermediates_ref(
            q,
            latent_cache,
            k_proj,
            v_proj,
            q_heads=4,
            kv_heads=2,
            head_dim=0,
            group_size=2,
        )


def test_latent_kv_reconstruction_shapes_and_manual_matmul() -> None:
    latent = np.array(
        [[[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]]],
        dtype=np.float32,
    )
    k_proj = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) * 0.125 - 0.5
    v_proj = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) * -0.25 + 1.25

    k_cache, v_cache = latent_kv_reconstruction_ref(
        latent, k_proj, v_proj, kv_heads=2, head_dim=2
    )

    assert k_cache.shape == (1, 2, 2, 2)
    assert v_cache.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(k_cache.reshape(1, 2, 4), latent @ k_proj, atol=1e-6)
    np.testing.assert_allclose(v_cache.reshape(1, 2, 4), latent @ v_proj, atol=1e-6)
    assert np.isfinite(k_cache).all()
    assert np.isfinite(v_cache).all()
    assert not np.array_equal(k_cache, v_cache)


def test_latent_kv_reconstruction_is_deterministic_and_signed() -> None:
    latent = np.linspace(-1.0, 1.0, 1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    k_proj = np.linspace(0.75, -0.5, 4 * 8, dtype=np.float32).reshape(4, 8)
    v_proj = np.linspace(-0.25, 0.875, 4 * 8, dtype=np.float32).reshape(4, 8)

    first_k, first_v = latent_kv_reconstruction_ref(
        latent, k_proj, v_proj, kv_heads=2, head_dim=4
    )
    second_k, second_v = latent_kv_reconstruction_ref(
        latent, k_proj, v_proj, kv_heads=2, head_dim=4
    )

    np.testing.assert_array_equal(first_k, second_k)
    np.testing.assert_array_equal(first_v, second_v)
    assert np.any(first_k < 0)
    assert np.any(first_v < 0)


def test_latent_kv_reconstruction_rejects_invalid_inputs() -> None:
    latent = np.zeros((1, 2, 3), dtype=np.float32)
    k_proj = np.zeros((3, 4), dtype=np.float32)
    v_proj = np.zeros((3, 4), dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        latent_kv_reconstruction_ref(latent[0], k_proj, v_proj, kv_heads=2, head_dim=2)
    with np.testing.assert_raises(ValueError):
        latent_kv_reconstruction_ref(latent, k_proj[:, None], v_proj, kv_heads=2, head_dim=2)
    with np.testing.assert_raises(ValueError):
        latent_kv_reconstruction_ref(latent, k_proj, v_proj[:, :3], kv_heads=2, head_dim=2)
    with np.testing.assert_raises(ValueError):
        latent_kv_reconstruction_ref(latent, k_proj[:2], v_proj[:2], kv_heads=2, head_dim=2)
    with np.testing.assert_raises(ValueError):
        latent_kv_reconstruction_ref(latent, k_proj, v_proj, kv_heads=2, head_dim=3)
    with np.testing.assert_raises(ValueError):
        latent_kv_reconstruction_ref(latent, k_proj, v_proj, kv_heads=0, head_dim=2)
    with np.testing.assert_raises(ValueError):
        latent_kv_reconstruction_ref(latent, k_proj, v_proj, kv_heads=2, head_dim=0)


def test_latent_kv_decode_attention_uses_reconstruction_helper() -> None:
    q, _, _, latent_cache, k_proj, v_proj = _make_tiny_case()
    k_cache, v_cache = latent_kv_reconstruction_ref(
        latent_cache,
        k_proj,
        v_proj,
        kv_heads=2,
        head_dim=8,
    )

    from_reconstruction = gqa_decode_attention_ref(q, k_cache, v_cache, group_size=2)
    from_latent = latent_kv_decode_attention_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )

    np.testing.assert_allclose(from_latent, from_reconstruction, atol=1e-6)


def test_paged_latent_kv_matches_dense_latent_kv() -> None:
    q, _, _, latent_cache, k_proj, v_proj = _make_tiny_case()
    block_size = 2
    seq_len = latent_cache.shape[1]
    block_table = np.array([2, 0, 1], dtype=np.int64)

    pad_tokens = block_size * len(block_table) - seq_len
    latent_tokens = np.pad(latent_cache[0], ((0, pad_tokens), (0, 0)))
    latent_blocks = np.empty(
        (len(block_table), block_size, latent_tokens.shape[1]), dtype=np.float32
    )
    logical_blocks = latent_tokens.reshape(len(block_table), block_size, latent_tokens.shape[1])
    for logical_idx, physical_idx in enumerate(block_table):
        latent_blocks[physical_idx] = logical_blocks[logical_idx]

    dense_out = latent_kv_decode_attention_ref(
        q,
        latent_cache,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )
    paged_out = paged_latent_kv_decode_attention_ref(
        q,
        latent_blocks,
        block_table,
        seq_len,
        block_size,
        k_proj,
        v_proj,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        group_size=2,
    )

    np.testing.assert_allclose(paged_out, dense_out, atol=1e-6)
