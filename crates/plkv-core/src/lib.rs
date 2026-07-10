use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlkvError {
    ZeroValue { name: &'static str },
    ArithmeticOverflow,
    InvalidPagedLookup { reason: &'static str },
    TokenOutOfRange { token_pos: usize, seq_len: usize },
    TokenNotAllocated { token_pos: usize, num_tokens: usize },
}

impl fmt::Display for PlkvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroValue { name } => write!(f, "{name} must be > 0"),
            Self::ArithmeticOverflow => write!(f, "paged lookup size arithmetic overflowed"),
            Self::InvalidPagedLookup { reason } => write!(f, "invalid paged lookup: {reason}"),
            Self::TokenOutOfRange { token_pos, seq_len } => {
                write!(
                    f,
                    "token position {token_pos} is out of range for seq_len {seq_len}"
                )
            }
            Self::TokenNotAllocated {
                token_pos,
                num_tokens,
            } => write!(
                f,
                "token position {token_pos} is out of range for {num_tokens} allocated tokens"
            ),
        }
    }
}

impl std::error::Error for PlkvError {}

fn require_positive(name: &'static str, value: usize) -> Result<(), PlkvError> {
    if value == 0 {
        return Err(PlkvError::ZeroValue { name });
    }
    Ok(())
}

pub fn kv_bytes_per_token_gqa(
    n_kv_heads: usize,
    head_dim: usize,
    dtype_bytes: usize,
    include_k_and_v: bool,
) -> Result<usize, PlkvError> {
    require_positive("n_kv_heads", n_kv_heads)?;
    require_positive("head_dim", head_dim)?;
    require_positive("dtype_bytes", dtype_bytes)?;
    let components = if include_k_and_v { 2 } else { 1 };
    n_kv_heads
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(dtype_bytes))
        .and_then(|value| value.checked_mul(components))
        .ok_or(PlkvError::ArithmeticOverflow)
}

pub fn kv_bytes_per_token_latent(
    latent_dim: usize,
    dtype_bytes: usize,
) -> Result<usize, PlkvError> {
    require_positive("latent_dim", latent_dim)?;
    require_positive("dtype_bytes", dtype_bytes)?;
    latent_dim
        .checked_mul(dtype_bytes)
        .ok_or(PlkvError::ArithmeticOverflow)
}

pub fn compression_ratio(full_kv_bytes: usize, latent_kv_bytes: usize) -> Result<f64, PlkvError> {
    require_positive("full_kv_bytes", full_kv_bytes)?;
    require_positive("latent_kv_bytes", latent_kv_bytes)?;
    Ok(full_kv_bytes as f64 / latent_kv_bytes as f64)
}

pub fn estimate_total_kv_cache_bytes(
    num_layers: usize,
    seq_len: usize,
    batch_size: usize,
    bytes_per_token_per_layer: usize,
) -> Result<usize, PlkvError> {
    require_positive("num_layers", num_layers)?;
    require_positive("seq_len", seq_len)?;
    require_positive("batch_size", batch_size)?;
    require_positive("bytes_per_token_per_layer", bytes_per_token_per_layer)?;
    num_layers
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(batch_size))
        .and_then(|value| value.checked_mul(bytes_per_token_per_layer))
        .ok_or(PlkvError::ArithmeticOverflow)
}

pub fn paged_lookup_f32(
    physical_blocks: &[f32],
    block_table: &[usize],
    seq_len: usize,
    block_size: usize,
    width: usize,
) -> Result<Vec<f32>, PlkvError> {
    require_positive("seq_len", seq_len)?;
    require_positive("block_size", block_size)?;
    require_positive("width", width)?;
    let logical_blocks = seq_len.div_ceil(block_size);
    if block_table.len() < logical_blocks {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table does not cover seq_len",
        });
    }
    let block_stride = block_size
        .checked_mul(width)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if physical_blocks.len() % block_stride != 0 {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "physical storage length is not divisible by block stride",
        });
    }
    let num_physical_blocks = physical_blocks.len() / block_stride;
    let output_len = seq_len
        .checked_mul(width)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let mut output = Vec::with_capacity(output_len);
    for token in 0..seq_len {
        let logical_block = token / block_size;
        let offset = token % block_size;
        let physical_block = block_table[logical_block];
        if physical_block >= num_physical_blocks {
            return Err(PlkvError::InvalidPagedLookup {
                reason: "block table contains an invalid physical block",
            });
        }
        let start = physical_block
            .checked_mul(block_stride)
            .and_then(|value| value.checked_add(offset.checked_mul(width)?))
            .ok_or(PlkvError::ArithmeticOverflow)?;
        let end = start
            .checked_add(width)
            .ok_or(PlkvError::ArithmeticOverflow)?;
        output.extend_from_slice(physical_blocks.get(start..end).ok_or(
            PlkvError::InvalidPagedLookup {
                reason: "physical storage indexing exceeded storage length",
            },
        )?);
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl CacheConfig {
    pub fn new(
        block_size: usize,
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, PlkvError> {
        require_positive("block_size", block_size)?;
        require_positive("num_layers", num_layers)?;
        require_positive("kv_heads", kv_heads)?;
        require_positive("head_dim", head_dim)?;
        Ok(Self {
            block_size,
            num_layers,
            kv_heads,
            head_dim,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTableConfig {
    pub seq_len: usize,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLocation {
    pub logical_block: usize,
    pub physical_block: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTable {
    seq_len: usize,
    block_size: usize,
    logical_to_physical: Vec<usize>,
    next_physical_block: usize,
    num_tokens: usize,
}

impl BlockTable {
    pub fn new_contiguous(seq_len: usize, block_size: usize) -> Result<Self, PlkvError> {
        require_positive("seq_len", seq_len)?;
        require_positive("block_size", block_size)?;
        let num_blocks = seq_len.div_ceil(block_size);
        Ok(Self {
            seq_len,
            block_size,
            logical_to_physical: (0..num_blocks).collect(),
            next_physical_block: num_blocks,
            num_tokens: seq_len,
        })
    }

    pub fn new(block_size: usize) -> Result<Self, PlkvError> {
        require_positive("block_size", block_size)?;
        Ok(Self {
            seq_len: 0,
            block_size,
            logical_to_physical: Vec::new(),
            next_physical_block: 0,
            num_tokens: 0,
        })
    }

    pub fn allocate_tokens(
        &mut self,
        count: usize,
    ) -> Result<Vec<(usize, BlockLocation)>, PlkvError> {
        require_positive("count", count)?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let token_pos = self.num_tokens;
            let logical_block = token_pos / self.block_size;
            if logical_block == self.logical_to_physical.len() {
                self.logical_to_physical.push(self.next_physical_block);
                self.next_physical_block += 1;
            }
            self.num_tokens += 1;
            self.seq_len = self.num_tokens;
            out.push((token_pos, self.location_for_token(token_pos)?));
        }
        Ok(out)
    }

    pub fn location_for_token(&self, token_pos: usize) -> Result<BlockLocation, PlkvError> {
        if token_pos >= self.seq_len {
            return Err(PlkvError::TokenOutOfRange {
                token_pos,
                seq_len: self.seq_len,
            });
        }
        let logical_block = token_pos / self.block_size;
        Ok(BlockLocation {
            logical_block,
            physical_block: self.logical_to_physical[logical_block],
            offset: token_pos % self.block_size,
        })
    }

    pub fn translate(&self, token_pos: usize) -> Result<BlockLocation, PlkvError> {
        if token_pos >= self.num_tokens {
            return Err(PlkvError::TokenNotAllocated {
                token_pos,
                num_tokens: self.num_tokens,
            });
        }
        self.location_for_token(token_pos)
    }

    pub fn physical_block_for_token(&self, token_pos: usize) -> Result<usize, PlkvError> {
        Ok(self.location_for_token(token_pos)?.physical_block)
    }

    pub fn logical_blocks(&self) -> &[usize] {
        &self.logical_to_physical
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_logical_blocks(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn num_blocks(&self) -> usize {
        self.num_logical_blocks()
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::{
        BlockLocation, BlockTable, CacheConfig, PlkvError, compression_ratio,
        estimate_total_kv_cache_bytes, kv_bytes_per_token_gqa, kv_bytes_per_token_latent,
        paged_lookup_f32,
    };

    #[test]
    fn memory_model_matches_small_reference_values() {
        let gqa = kv_bytes_per_token_gqa(2, 32, 2, true).unwrap();
        let latent = kv_bytes_per_token_latent(32, 2).unwrap();
        assert_eq!(gqa, 256);
        assert_eq!(latent, 64);
        assert_eq!(
            estimate_total_kv_cache_bytes(24, 128, 1, gqa).unwrap(),
            786_432
        );
        assert_eq!(
            estimate_total_kv_cache_bytes(24, 128, 1, latent).unwrap(),
            196_608
        );
        assert_eq!(compression_ratio(gqa, latent).unwrap(), 4.0);
    }

    #[test]
    fn memory_model_rejects_zero_values() {
        assert_eq!(
            kv_bytes_per_token_gqa(0, 32, 2, true),
            Err(PlkvError::ZeroValue { name: "n_kv_heads" })
        );
        assert!(kv_bytes_per_token_latent(32, 0).is_err());
        assert!(compression_ratio(1, 0).is_err());
        assert!(estimate_total_kv_cache_bytes(1, 1, 0, 1).is_err());
    }

    #[test]
    fn contiguous_table_covers_exact_boundary() {
        let table = BlockTable::new_contiguous(5, 2).unwrap();
        assert_eq!(table.logical_blocks(), &[0, 1, 2]);
        assert_eq!(table.location_for_token(0).unwrap().offset, 0);
        assert_eq!(table.location_for_token(4).unwrap().logical_block, 2);
        assert_eq!(table.location_for_token(4).unwrap().offset, 0);
    }

    #[test]
    fn contiguous_table_rejects_invalid_values_and_range() {
        assert!(BlockTable::new_contiguous(0, 2).is_err());
        assert!(BlockTable::new_contiguous(5, 0).is_err());
        let table = BlockTable::new_contiguous(5, 2).unwrap();
        assert!(table.location_for_token(5).is_err());
    }

    #[test]
    fn allocated_table_preserves_existing_behavior() {
        let mut table = BlockTable::new(4).unwrap();
        let allocations = table.allocate_tokens(6).unwrap();
        assert_eq!(allocations[0].0, 0);
        assert_eq!(allocations[0].1.physical_block, 0);
        assert_eq!(allocations[3].1.offset, 3);
        assert_eq!(allocations[4].1.physical_block, 1);
        assert_eq!(allocations[4].1.offset, 0);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.translate(5).unwrap().offset, 1);
    }

    #[test]
    fn cache_config_rejects_zero_values() {
        assert!(CacheConfig::new(0, 28, 8, 128).is_err());
    }

    #[derive(Debug, Deserialize)]
    struct MemoryFixture {
        gqa_bytes_per_token_per_layer: usize,
        latent_bytes_per_token_per_layer: usize,
        gqa_total_kv_bytes: usize,
        latent_total_kv_bytes: usize,
        compression_ratio_vs_gqa: f64,
    }

    #[derive(Debug, Deserialize)]
    struct BlockFixture {
        seq_len: usize,
        block_size: usize,
        logical_blocks: Vec<usize>,
        token_locations: Vec<TokenFixture>,
    }

    #[derive(Debug, Deserialize)]
    struct TokenFixture {
        token: usize,
        logical_block: usize,
        physical_block: usize,
        offset: usize,
    }

    #[derive(Debug, Deserialize)]
    struct PagedLookupFixture {
        seq_len: usize,
        block_size: usize,
        width: usize,
        num_physical_blocks: usize,
        block_table: Vec<usize>,
        physical_blocks: Vec<Vec<Vec<f32>>>,
        expected_logical_output: Vec<Vec<f32>>,
    }

    #[test]
    fn memory_model_matches_python_golden_fixture() {
        let fixture: MemoryFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/memory_model_small.json"
        ))
        .unwrap();
        let gqa = kv_bytes_per_token_gqa(2, 32, 2, true).unwrap();
        let latent = kv_bytes_per_token_latent(32, 2).unwrap();

        assert_eq!(gqa, fixture.gqa_bytes_per_token_per_layer);
        assert_eq!(latent, fixture.latent_bytes_per_token_per_layer);
        assert_eq!(
            estimate_total_kv_cache_bytes(24, 128, 1, gqa).unwrap(),
            fixture.gqa_total_kv_bytes
        );
        assert_eq!(
            estimate_total_kv_cache_bytes(24, 128, 1, latent).unwrap(),
            fixture.latent_total_kv_bytes
        );
        assert_eq!(
            compression_ratio(gqa, latent).unwrap(),
            fixture.compression_ratio_vs_gqa
        );
    }

    #[test]
    fn block_table_matches_seq5_python_golden_fixture() {
        let fixture: BlockFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/block_table_seq5_block2.json"
        ))
        .unwrap();
        let table = BlockTable::new_contiguous(fixture.seq_len, fixture.block_size).unwrap();
        assert_eq!(table.logical_blocks(), fixture.logical_blocks);
        for expected in fixture.token_locations {
            assert_eq!(
                table.location_for_token(expected.token).unwrap(),
                BlockLocation {
                    logical_block: expected.logical_block,
                    physical_block: expected.physical_block,
                    offset: expected.offset,
                }
            );
        }
    }

    #[test]
    fn block_table_matches_seq128_python_golden_fixture() {
        let fixture: BlockFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/block_table_seq128_block16.json"
        ))
        .unwrap();
        let table = BlockTable::new_contiguous(fixture.seq_len, fixture.block_size).unwrap();
        assert_eq!(table.logical_blocks(), fixture.logical_blocks);
        for expected in fixture.token_locations {
            let actual = table.location_for_token(expected.token).unwrap();
            assert_eq!(actual.logical_block, expected.logical_block);
            assert_eq!(actual.physical_block, expected.physical_block);
            assert_eq!(actual.offset, expected.offset);
        }
    }

    #[test]
    fn paged_lookup_matches_python_golden_fixture() {
        let fixture: PagedLookupFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/paged_lookup_f32_seq5_block2_width4.json"
        ))
        .unwrap();
        assert_eq!(fixture.physical_blocks.len(), fixture.num_physical_blocks);
        let physical_blocks: Vec<f32> = fixture
            .physical_blocks
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        let expected: Vec<f32> = fixture
            .expected_logical_output
            .iter()
            .flatten()
            .copied()
            .collect();
        assert_eq!(
            paged_lookup_f32(
                &physical_blocks,
                &fixture.block_table,
                fixture.seq_len,
                fixture.block_size,
                fixture.width,
            )
            .unwrap(),
            expected
        );
    }

    #[test]
    fn paged_lookup_rejects_invalid_storage_and_dimensions() {
        assert!(paged_lookup_f32(&[1.0], &[0], 1, 1, 2).is_err());
        assert!(paged_lookup_f32(&[1.0], &[], 1, 1, 1).is_err());
        assert!(paged_lookup_f32(&[1.0], &[1], 1, 1, 1).is_err());
        assert!(paged_lookup_f32(&[1.0], &[0], 1, 0, 1).is_err());
    }
}
