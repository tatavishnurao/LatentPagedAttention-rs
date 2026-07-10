import json

from latent_paged_attention.fixtures import (
    block_table_fixture,
    memory_model_fixture,
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
    }
    for path in paths:
        assert json.loads(path.read_text(encoding="utf-8"))
