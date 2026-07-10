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
pub struct PagedTokenLocation {
    pub logical_block: usize,
    pub physical_block: usize,
    pub block_offset: usize,
}

pub fn resolve_paged_token_location(
    block_table: &[usize],
    token_position: usize,
    block_size: usize,
    num_physical_blocks: usize,
) -> Result<PagedTokenLocation, PlkvError> {
    require_positive("block_size", block_size)?;
    require_positive("num_physical_blocks", num_physical_blocks)?;
    let logical_block = token_position / block_size;
    let physical_block = *block_table
        .get(logical_block)
        .ok_or(PlkvError::InvalidPagedLookup {
            reason: "block table does not cover token position",
        })?;
    if physical_block >= num_physical_blocks {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table contains an invalid physical block",
        });
    }
    Ok(PagedTokenLocation {
        logical_block,
        physical_block,
        block_offset: token_position % block_size,
    })
}

pub fn paged_kv_write_f32(
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    block_table: &[usize],
    token_position: usize,
    block_size: usize,
    width: usize,
    new_k: &[f32],
    new_v: &[f32],
) -> Result<PagedTokenLocation, PlkvError> {
    require_positive("block_size", block_size)?;
    require_positive("width", width)?;
    if k_cache.len() != v_cache.len() {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "K and V cache lengths must match",
        });
    }
    if new_k.len() != width || new_v.len() != width {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "new K and V vectors must match width",
        });
    }
    let block_stride = block_size
        .checked_mul(width)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if k_cache.len() % block_stride != 0 {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "cache length is not divisible by block stride",
        });
    }
    let location = resolve_paged_token_location(
        block_table,
        token_position,
        block_size,
        k_cache.len() / block_stride,
    )?;
    let start = location
        .physical_block
        .checked_mul(block_stride)
        .and_then(|value| value.checked_add(location.block_offset.checked_mul(width)?))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let end = start
        .checked_add(width)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let k_target = k_cache
        .get_mut(start..end)
        .ok_or(PlkvError::InvalidPagedLookup {
            reason: "K cache indexing exceeded storage length",
        })?;
    let v_target = v_cache
        .get_mut(start..end)
        .ok_or(PlkvError::InvalidPagedLookup {
            reason: "V cache indexing exceeded storage length",
        })?;
    k_target.copy_from_slice(new_k);
    v_target.copy_from_slice(new_v);
    Ok(location)
}

#[derive(Debug, Clone, PartialEq)]
pub struct GqaDecodeResult {
    pub scores: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub context: Vec<f32>,
}

pub fn contiguous_gqa_decode_f32(
    q: &[f32],
    k_head_major: &[f32],
    v_head_major: &[f32],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    group_size: usize,
) -> Result<GqaDecodeResult, PlkvError> {
    require_positive("q_heads", q_heads)?;
    require_positive("kv_heads", kv_heads)?;
    require_positive("seq_len", seq_len)?;
    require_positive("head_dim", head_dim)?;
    require_positive("group_size", group_size)?;
    if q_heads != kv_heads.checked_mul(group_size).ok_or(PlkvError::ArithmeticOverflow)? {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "expected q_heads == kv_heads * group_size",
        });
    }
    let q_len = q_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let kv_len = kv_heads
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if q.len() != q_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "Q length does not match q_heads * head_dim",
        });
    }
    if k_head_major.len() != kv_len || v_head_major.len() != kv_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "K/V length does not match kv_heads * seq_len * head_dim",
        });
    }
    let scores_len = q_heads
        .checked_mul(seq_len)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let context_len = q_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let mut scores = vec![0.0f32; scores_len];
    let mut probabilities = vec![0.0f32; scores_len];
    let mut context = vec![0.0f32; context_len];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for q_head in 0..q_heads {
        let kv_head = q_head / group_size;
        let q_start = q_head * head_dim;
        let scores_start = q_head * seq_len;
        for token in 0..seq_len {
            let k_start = (kv_head * seq_len + token) * head_dim;
            let mut dot = 0.0f32;
            for dim in 0..head_dim {
                dot += q[q_start + dim] * k_head_major[k_start + dim];
            }
            scores[scores_start + token] = dot * scale;
        }

        let row = &scores[scores_start..scores_start + seq_len];
        let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !row_max.is_finite() {
            return Err(PlkvError::InvalidPagedLookup {
                reason: "score row maximum is not finite",
            });
        }
        let mut denom = 0.0f32;
        for token in 0..seq_len {
            let value = (scores[scores_start + token] - row_max).exp();
            probabilities[scores_start + token] = value;
            denom += value;
        }
        if !denom.is_finite() || denom == 0.0 {
            return Err(PlkvError::InvalidPagedLookup {
                reason: "softmax denominator must be finite and non-zero",
            });
        }
        for token in 0..seq_len {
            let value = probabilities[scores_start + token] / denom;
            if !value.is_finite() {
                return Err(PlkvError::InvalidPagedLookup {
                    reason: "probability must be finite",
                });
            }
            probabilities[scores_start + token] = value;
        }

        let context_start = q_head * head_dim;
        for dim in 0..head_dim {
            let mut value = 0.0f32;
            for token in 0..seq_len {
                let v_start = (kv_head * seq_len + token) * head_dim;
                value += probabilities[scores_start + token] * v_head_major[v_start + dim];
            }
            context[context_start + dim] = value;
        }
    }

    Ok(GqaDecodeResult {
        scores,
        probabilities,
        context,
    })
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
        contiguous_gqa_decode_f32, estimate_total_kv_cache_bytes, kv_bytes_per_token_gqa,
        kv_bytes_per_token_latent, paged_kv_write_f32, paged_lookup_f32,
        resolve_paged_token_location,
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

    #[derive(Debug, Deserialize)]
    struct PagedKvWriteFixture {
        block_size: usize,
        width: usize,
        block_table: Vec<usize>,
        cases: Vec<PagedKvCase>,
    }

    #[derive(Debug, Deserialize)]
    struct PagedKvCase {
        token_position: usize,
        logical_block: usize,
        physical_block: usize,
        block_offset: usize,
        initial_k_cache: Vec<Vec<Vec<f32>>>,
        initial_v_cache: Vec<Vec<Vec<f32>>>,
        new_k: Vec<f32>,
        new_v: Vec<f32>,
        expected_k_cache: Vec<Vec<Vec<f32>>>,
        expected_v_cache: Vec<Vec<Vec<f32>>>,
    }

    #[derive(Debug, Deserialize)]
    struct GqaDecodeFixture {
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        seq_len: usize,
        head_dim: usize,
        q_to_kv: Vec<usize>,
        cases: Vec<GqaCase>,
    }

    #[derive(Debug, Deserialize)]
    struct GqaCase {
        q: Vec<Vec<f32>>,
        k_head_major: Vec<Vec<Vec<f32>>>,
        v_head_major: Vec<Vec<Vec<f32>>>,
        expected_scores: Vec<Vec<f32>>,
        expected_probabilities: Vec<Vec<f32>>,
        expected_context: Vec<Vec<f32>>,
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

    #[test]
    fn paged_kv_write_matches_python_fixture_and_preserves_other_rows() {
        let fixture: PagedKvWriteFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/paged_kv_write_f32.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let mut k: Vec<f32> = case
                .initial_k_cache
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let mut v: Vec<f32> = case
                .initial_v_cache
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let original_k = k.clone();
            let original_v = v.clone();
            let location = paged_kv_write_f32(
                &mut k,
                &mut v,
                &fixture.block_table,
                case.token_position,
                fixture.block_size,
                fixture.width,
                &case.new_k,
                &case.new_v,
            )
            .unwrap();
            assert_eq!(location.logical_block, case.logical_block);
            assert_eq!(location.physical_block, case.physical_block);
            assert_eq!(location.block_offset, case.block_offset);
            let expected_k: Vec<f32> = case
                .expected_k_cache
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let expected_v: Vec<f32> = case
                .expected_v_cache
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            assert_eq!(k, expected_k);
            assert_eq!(v, expected_v);
            assert_eq!(
                k.iter().zip(original_k).filter(|(a, b)| *a != b).count(),
                fixture.width
            );
            assert_eq!(
                v.iter().zip(original_v).filter(|(a, b)| *a != b).count(),
                fixture.width
            );
        }
    }

    #[test]
    fn paged_kv_write_rejects_invalid_inputs() {
        assert!(resolve_paged_token_location(&[0], 2, 2, 1).is_err());
        assert!(resolve_paged_token_location(&[2], 0, 2, 2).is_err());
        assert!(
            paged_kv_write_f32(
                &mut [0.0; 4],
                &mut [0.0; 8],
                &[0],
                0,
                2,
                2,
                &[1.0, 2.0],
                &[1.0, 2.0]
            )
            .is_err()
        );
        assert!(
            paged_kv_write_f32(&mut [0.0; 4], &mut [0.0; 4], &[0], 0, 2, 2, &[1.0], &[1.0])
                .is_err()
        );
    }

    #[test]
    fn contiguous_gqa_decode_matches_python_fixture() {
        let fixture: GqaDecodeFixture =
            serde_json::from_str(include_str!("../../../fixtures/reference/gqa_decode_f32.json"))
                .unwrap();
        assert_eq!(fixture.q_to_kv, vec![0, 0, 1, 1]);
        for case in fixture.cases {
            let q: Vec<f32> = case.q.iter().flatten().copied().collect();
            let k: Vec<f32> = case.k_head_major.iter().flatten().flatten().copied().collect();
            let v: Vec<f32> = case.v_head_major.iter().flatten().flatten().copied().collect();
            let expected_scores: Vec<f32> =
                case.expected_scores.iter().flatten().copied().collect();
            let expected_probabilities: Vec<f32> =
                case.expected_probabilities.iter().flatten().copied().collect();
            let expected_context: Vec<f32> =
                case.expected_context.iter().flatten().copied().collect();
            let actual = contiguous_gqa_decode_f32(
                &q,
                &k,
                &v,
                fixture.q_heads,
                fixture.kv_heads,
                fixture.seq_len,
                fixture.head_dim,
                fixture.group_size,
            )
            .unwrap();
            assert_close(&actual.scores, &expected_scores, 1e-5, 1e-5);
            assert_close(&actual.probabilities, &expected_probabilities, 1e-5, 1e-5);
            assert_close(&actual.context, &expected_context, 1e-5, 1e-5);
            for q_head in 0..fixture.q_heads {
                let start = q_head * fixture.seq_len;
                let row_sum: f32 = actual.probabilities[start..start + fixture.seq_len]
                    .iter()
                    .sum();
                assert!(
                    (row_sum - 1.0).abs() <= 1e-5,
                    "probability row sum was {row_sum}"
                );
            }
        }
    }

    #[test]
    fn contiguous_gqa_decode_rejects_invalid_inputs() {
        assert!(contiguous_gqa_decode_f32(&[], &[], &[], 0, 2, 8, 8, 2).is_err());
        assert!(contiguous_gqa_decode_f32(&[0.0; 32], &[0.0; 128], &[0.0; 128], 3, 2, 8, 8, 2).is_err());
        assert!(contiguous_gqa_decode_f32(&[0.0; 31], &[0.0; 128], &[0.0; 128], 4, 2, 8, 8, 2).is_err());
        assert!(contiguous_gqa_decode_f32(&[0.0; 32], &[0.0; 127], &[0.0; 128], 4, 2, 8, 8, 2).is_err());
    }

    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = atol + rtol * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }
}
