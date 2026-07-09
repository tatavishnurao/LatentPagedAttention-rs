import pytest
from latent_paged_attention.block_table import PagedBlockTable


def test_allocate_and_translate_across_boundary() -> None:
    table = PagedBlockTable(block_size=4)
    allocations = table.allocate_token_positions(6)

    assert allocations[0] == (0, 0, 0)
    assert allocations[3] == (3, 0, 3)
    assert allocations[4] == (4, 1, 0)
    assert table.num_blocks == 2
    assert table.translate(5) == (1, 1)


def test_translate_rejects_unallocated_position() -> None:
    table = PagedBlockTable(block_size=8)
    table.allocate_token_positions(2)

    with pytest.raises(IndexError):
        table.translate(2)


def test_invalid_block_size_rejected() -> None:
    with pytest.raises(ValueError):
        PagedBlockTable(block_size=0)
