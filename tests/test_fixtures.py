import json

import numpy as np
from latent_paged_attention.attention_ref import latent_kv_reconstruction_ref
from latent_paged_attention.fixtures import (
    block_table_fixture,
    gqa_decode_f32_fixture,
    latent_kv_reconstruction_f32_fixture,
    memory_model_fixture,
    paged_gqa_decode_f32_fixture,
    paged_kv_write_fixture,
    paged_lookup_f32_fixture,
    write_fixtures,
)


def test_memory_fixture_has_expected_keys_and_values() -> None:
    fixture = memory_model_fixture()

    assert {
        "config",
        "gqa_bytes_per_token_per_layer",
        "latent_bytes_per_token_per_layer",
        "gqa_total_kv_bytes",
        "latent_total_kv_bytes",
        "compression_ratio_vs_gqa",
    } <= fixture.keys()
    assert fixture["gqa_bytes_per_token_per_layer"] == 256
    assert fixture["latent_bytes_per_token_per_layer"] == 64
    assert fixture["gqa_total_kv_bytes"] == 786432
    assert fixture["latent_total_kv_bytes"] == 196608
    assert fixture["compression_ratio_vs_gqa"] == 4.0


def test_block_table_fixture_has_expected_locations() -> None:
    fixture = block_table_fixture(5, 2)

    assert fixture["logical_blocks"] == [0, 1, 2]
    assert fixture["token_locations"] == [
        {"token": 0, "logical_block": 0, "physical_block": 0, "offset": 0},
        {"token": 1, "logical_block": 0, "physical_block": 0, "offset": 1},
        {"token": 2, "logical_block": 1, "physical_block": 1, "offset": 0},
        {"token": 3, "logical_block": 1, "physical_block": 1, "offset": 1},
        {"token": 4, "logical_block": 2, "physical_block": 2, "offset": 0},
    ]


def test_generated_fixture_files_are_valid_json(tmp_path) -> None:
    paths = write_fixtures(tmp_path)

    assert {path.name for path in paths} == {
        "memory_model_small.json",
        "block_table_seq5_block2.json",
        "block_table_seq128_block16.json",
        "paged_lookup_f32_seq5_block2_width4.json",
        "paged_kv_write_f32.json",
        "gqa_decode_f32.json",
        "paged_gqa_decode_f32.json",
        "latent_kv_reconstruction_f32.json",
    }
    for path in paths:
        assert json.loads(path.read_text(encoding="utf-8"))


def test_paged_lookup_fixture_crosses_non_identity_physical_blocks() -> None:
    fixture = paged_lookup_f32_fixture()
    assert fixture["block_table"] == [2, 0, 1]
    assert fixture["block_table"] != [0, 1, 2]
    assert fixture["expected_logical_output"] == [
        [20.0, 21.0, 22.0, 23.0],
        [24.0, 25.0, 26.0, 27.0],
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [10.0, 11.0, 12.0, 13.0],
    ]
    assert len(fixture["expected_logical_output"]) == 5
    assert all(len(row) == 4 for row in fixture["expected_logical_output"])


def test_paged_lookup_fixture_is_deterministic() -> None:
    assert paged_lookup_f32_fixture() == paged_lookup_f32_fixture()


def test_paged_kv_write_fixture_has_two_locations_and_is_deterministic() -> None:
    fixture = paged_kv_write_fixture()
    assert fixture["block_table"] == [2, 0, 1]
    assert fixture["gpu_padded_block_table"] == [2, 0, 1, 3]
    assert [(case["physical_block"], case["block_offset"]) for case in fixture["cases"]] == [
        (0, 1),
        (1, 0),
    ]
    assert fixture == paged_kv_write_fixture()


def test_gqa_decode_fixture_layouts_and_probabilities_are_valid() -> None:
    fixture = gqa_decode_f32_fixture()
    assert fixture["q_to_kv"] == [0, 0, 1, 1]
    assert fixture["batch"] == 1
    assert fixture["q_heads"] == 4
    assert fixture["kv_heads"] == 2
    assert fixture["seq_len"] == 8
    assert fixture["head_dim"] == 8

    for case in fixture["cases"]:
        k_token = np.asarray(case["k_token_major"], dtype=np.float32)
        v_token = np.asarray(case["v_token_major"], dtype=np.float32)
        k_head = np.asarray(case["k_head_major"], dtype=np.float32)
        v_head = np.asarray(case["v_head_major"], dtype=np.float32)
        np.testing.assert_array_equal(k_head, np.transpose(k_token, (1, 0, 2)))
        np.testing.assert_array_equal(v_head, np.transpose(v_token, (1, 0, 2)))

        probabilities = np.asarray(case["expected_probabilities"], dtype=np.float32)
        np.testing.assert_allclose(probabilities.sum(axis=-1), np.ones(4), atol=1e-6)
        assert np.isfinite(np.asarray(case["expected_scores"], dtype=np.float32)).all()
        assert np.isfinite(probabilities).all()
        assert np.isfinite(np.asarray(case["expected_context"], dtype=np.float32)).all()

    balanced = np.asarray(fixture["cases"][0]["expected_context"], dtype=np.float32)
    stable = np.asarray(fixture["cases"][1]["expected_context"], dtype=np.float32)
    assert not np.array_equal(balanced, stable)
    assert fixture == gqa_decode_f32_fixture()


def test_paged_gqa_decode_fixture_layouts_and_probabilities_are_valid() -> None:
    fixture = paged_gqa_decode_f32_fixture()
    assert fixture["block_table"] == [2, 0, 3, 1]
    assert fixture["block_table"] != [0, 1, 2, 3]
    assert sorted(fixture["block_table"]) == [0, 1, 2, 3]
    assert fixture["q_to_kv"] == [0, 0, 1, 1]

    contiguous = gqa_decode_f32_fixture()
    for paged_case, contiguous_case in zip(
        fixture["cases"], contiguous["cases"], strict=True
    ):
        k_physical_token = np.asarray(paged_case["k_physical_token_major"], dtype=np.float32)
        v_physical_token = np.asarray(paged_case["v_physical_token_major"], dtype=np.float32)
        k_physical_head = np.asarray(paged_case["k_physical_gpu_head_major"], dtype=np.float32)
        v_physical_head = np.asarray(paged_case["v_physical_gpu_head_major"], dtype=np.float32)
        np.testing.assert_array_equal(k_physical_head, np.transpose(k_physical_token, (0, 2, 1, 3)))
        np.testing.assert_array_equal(v_physical_head, np.transpose(v_physical_token, (0, 2, 1, 3)))

        reconstructed_k = np.asarray(
            [
                k_physical_token[fixture["block_table"][logical_block]]
                for logical_block in range(fixture["num_logical_blocks"])
            ],
            dtype=np.float32,
        ).reshape(fixture["seq_len"], fixture["kv_heads"], fixture["head_dim"])
        reconstructed_v = np.asarray(
            [
                v_physical_token[fixture["block_table"][logical_block]]
                for logical_block in range(fixture["num_logical_blocks"])
            ],
            dtype=np.float32,
        ).reshape(fixture["seq_len"], fixture["kv_heads"], fixture["head_dim"])
        np.testing.assert_array_equal(
            reconstructed_k, np.asarray(contiguous_case["k_token_major"], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            reconstructed_v, np.asarray(contiguous_case["v_token_major"], dtype=np.float32)
        )

        np.testing.assert_allclose(
            np.asarray(paged_case["expected_scores"], dtype=np.float32),
            np.asarray(contiguous_case["expected_scores"], dtype=np.float32),
            atol=1e-6,
        )
        probabilities = np.asarray(paged_case["expected_probabilities"], dtype=np.float32)
        np.testing.assert_allclose(probabilities.sum(axis=-1), np.ones(4), atol=1e-6)
        assert np.isfinite(np.asarray(paged_case["expected_scores"], dtype=np.float32)).all()
        assert np.isfinite(probabilities).all()
        assert np.isfinite(np.asarray(paged_case["expected_context"], dtype=np.float32)).all()

    assert fixture == paged_gqa_decode_f32_fixture()


def test_latent_kv_reconstruction_fixture_layouts_are_valid() -> None:
    fixture = latent_kv_reconstruction_f32_fixture()

    assert fixture["batch"] == 1
    assert fixture["seq_len"] == 8
    assert fixture["latent_dim"] == 8
    assert fixture["kv_heads"] == 2
    assert fixture["head_dim"] == 8
    assert fixture["projection_width"] == 16
    assert fixture["latent_values_per_token"] == 8
    assert fixture["full_kv_values_per_token"] == 32
    assert fixture["theoretical_cache_compression_ratio"] == 4.0

    for case in fixture["cases"]:
        latent = np.asarray(case["latent_cache"], dtype=np.float32)[None, ...]
        k_proj = np.asarray(case["k_projection"], dtype=np.float32)
        v_proj = np.asarray(case["v_projection"], dtype=np.float32)
        k_token = np.asarray(case["expected_k_token_major"], dtype=np.float32)
        v_token = np.asarray(case["expected_v_token_major"], dtype=np.float32)
        k_head = np.asarray(case["expected_k_head_major"], dtype=np.float32)
        v_head = np.asarray(case["expected_v_head_major"], dtype=np.float32)

        assert latent.shape == (1, 8, 8)
        assert k_proj.shape == (8, 16)
        assert v_proj.shape == (8, 16)
        assert k_token.shape == (8, 2, 8)
        assert v_token.shape == (8, 2, 8)
        assert k_head.shape == (2, 8, 8)
        assert v_head.shape == (2, 8, 8)
        assert not np.array_equal(k_proj, v_proj)
        assert not np.array_equal(k_token, v_token)
        np.testing.assert_array_equal(k_head, np.transpose(k_token, (1, 0, 2)))
        np.testing.assert_array_equal(v_head, np.transpose(v_token, (1, 0, 2)))

        expected_k, expected_v = latent_kv_reconstruction_ref(
            latent,
            k_proj,
            v_proj,
            kv_heads=fixture["kv_heads"],
            head_dim=fixture["head_dim"],
        )
        np.testing.assert_allclose(k_token, expected_k[0], atol=1e-6)
        np.testing.assert_allclose(v_token, expected_v[0], atol=1e-6)
        assert np.isfinite(k_token).all()
        assert np.isfinite(v_token).all()

        _, wrong_v = latent_kv_reconstruction_ref(
            latent,
            k_proj,
            k_proj,
            kv_heads=fixture["kv_heads"],
            head_dim=fixture["head_dim"],
        )
        assert not np.allclose(wrong_v[0], v_token, atol=1e-6)

    signed = fixture["cases"][1]
    signed_latent = np.asarray(signed["latent_cache"], dtype=np.float32)
    signed_k_proj = np.asarray(signed["k_projection"], dtype=np.float32)
    signed_products = signed_latent[0, :, None] * signed_k_proj
    assert np.any(signed_products > 0)
    assert np.any(signed_products < 0)
    assert fixture == latent_kv_reconstruction_f32_fixture()
