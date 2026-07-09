"""Simple paged KV cache block-table model."""

from dataclasses import dataclass, field


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass
class PagedBlockTable:
    """Track logical token positions in fixed-size KV blocks."""

    block_size: int
    logical_to_physical: list[int] = field(default_factory=list)
    next_physical_block: int = 0
    num_tokens: int = 0

    def __post_init__(self) -> None:
        _validate_positive_int("block_size", self.block_size)

    def allocate_token_positions(self, num_tokens: int) -> list[tuple[int, int, int]]:
        """Allocate logical positions and return (token_pos, block_idx, offset)."""
        _validate_positive_int("num_tokens", num_tokens)
        allocated = []
        for _ in range(num_tokens):
            token_pos = self.num_tokens
            logical_block = token_pos // self.block_size
            if logical_block == len(self.logical_to_physical):
                self.logical_to_physical.append(self.next_physical_block)
                self.next_physical_block += 1
            self.num_tokens += 1
            allocated.append((token_pos, *self.translate(token_pos)))
        return allocated

    def translate(self, token_pos: int) -> tuple[int, int]:
        """Translate token position to (physical_block_index, offset)."""
        if not isinstance(token_pos, int):
            raise TypeError(f"token_pos must be an int, got {type(token_pos).__name__}")
        if token_pos < 0:
            raise ValueError(f"token_pos must be >= 0, got {token_pos}")
        if token_pos >= self.num_tokens:
            raise IndexError(
                f"token_pos {token_pos} is out of range for {self.num_tokens} allocated tokens"
            )

        logical_block = token_pos // self.block_size
        offset = token_pos % self.block_size
        return self.logical_to_physical[logical_block], offset

    @property
    def num_blocks(self) -> int:
        return len(self.logical_to_physical)
