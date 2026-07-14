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

pub fn paged_latent_write_f32(
    latent_cache: &mut [f32],
    block_table: &[usize],
    token_position: usize,
    block_size: usize,
    latent_dim: usize,
    new_latent: &[f32],
) -> Result<PagedTokenLocation, PlkvError> {
    require_positive("block_size", block_size)?;
    require_positive("latent_dim", latent_dim)?;
    let block_stride = block_size
        .checked_mul(latent_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if latent_cache.is_empty() || latent_cache.len() % block_stride != 0 {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "latent cache length is not divisible by block stride",
        });
    }
    if new_latent.len() != latent_dim {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "new latent vector does not match latent_dim",
        });
    }
    if !latent_cache.iter().all(|value| value.is_finite())
        || !new_latent.iter().all(|value| value.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "latent write inputs must be finite",
        });
    }
    let location = resolve_paged_token_location(
        block_table,
        token_position,
        block_size,
        latent_cache.len() / block_stride,
    )?;
    let start = location
        .physical_block
        .checked_mul(block_stride)
        .and_then(|value| value.checked_add(location.block_offset.checked_mul(latent_dim)?))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let end = start
        .checked_add(latent_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    latent_cache
        .get_mut(start..end)
        .ok_or(PlkvError::InvalidPagedLookup {
            reason: "latent cache indexing exceeded storage length",
        })?
        .copy_from_slice(new_latent);
    Ok(location)
}

pub fn quantize_f32_to_f16_storage(values: &[f32]) -> Result<Vec<half::f16>, PlkvError> {
    let mut output = Vec::with_capacity(values.len());
    for &value in values {
        if !value.is_finite() {
            return Err(PlkvError::InvalidPagedLookup {
                reason: "FP16 storage input must be finite",
            });
        }
        let stored = half::f16::from_f32(value);
        if !stored.is_finite() {
            return Err(PlkvError::InvalidPagedLookup {
                reason: "FP16 storage conversion overflowed",
            });
        }
        output.push(stored);
    }
    Ok(output)
}

pub fn fp16_storage_to_f32(values: &[half::f16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
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
    if q_heads
        != kv_heads
            .checked_mul(group_size)
            .ok_or(PlkvError::ArithmeticOverflow)?
    {
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

pub fn paged_gqa_decode_f32(
    q: &[f32],
    k_physical_head_major: &[f32],
    v_physical_head_major: &[f32],
    block_table: &[usize],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    group_size: usize,
    block_size: usize,
    num_physical_blocks: usize,
) -> Result<GqaDecodeResult, PlkvError> {
    require_positive("q_heads", q_heads)?;
    require_positive("kv_heads", kv_heads)?;
    require_positive("seq_len", seq_len)?;
    require_positive("head_dim", head_dim)?;
    require_positive("group_size", group_size)?;
    require_positive("block_size", block_size)?;
    require_positive("num_physical_blocks", num_physical_blocks)?;
    let logical_blocks = seq_len.div_ceil(block_size);
    if block_table.len() < logical_blocks {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table does not cover seq_len",
        });
    }
    for &physical_block in &block_table[..logical_blocks] {
        if physical_block >= num_physical_blocks {
            return Err(PlkvError::InvalidPagedLookup {
                reason: "block table contains an invalid physical block",
            });
        }
    }
    let physical_len = num_physical_blocks
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(block_size))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if k_physical_head_major.len() != physical_len || v_physical_head_major.len() != physical_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "K/V physical length does not match [num_physical_blocks, kv_heads, block_size, head_dim]",
        });
    }
    let logical_len = kv_heads
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let mut logical_k = vec![0.0f32; logical_len];
    let mut logical_v = vec![0.0f32; logical_len];
    for token in 0..seq_len {
        let logical_block = token / block_size;
        let block_offset = token % block_size;
        let physical_block = block_table[logical_block];
        for kv_head in 0..kv_heads {
            let physical_row =
                ((physical_block * kv_heads + kv_head) * block_size + block_offset) * head_dim;
            let logical_row = (kv_head * seq_len + token) * head_dim;
            logical_k[logical_row..logical_row + head_dim]
                .copy_from_slice(&k_physical_head_major[physical_row..physical_row + head_dim]);
            logical_v[logical_row..logical_row + head_dim]
                .copy_from_slice(&v_physical_head_major[physical_row..physical_row + head_dim]);
        }
    }
    contiguous_gqa_decode_f32(
        q, &logical_k, &logical_v, q_heads, kv_heads, seq_len, head_dim, group_size,
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatentKvReconstruction {
    pub k_token_major: Vec<f32>,
    pub v_token_major: Vec<f32>,
}

pub fn reconstruct_latent_kv_f32(
    latent_cache: &[f32],
    k_projection: &[f32],
    v_projection: &[f32],
    seq_len: usize,
    latent_dim: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<LatentKvReconstruction, PlkvError> {
    require_positive("seq_len", seq_len)?;
    require_positive("latent_dim", latent_dim)?;
    require_positive("kv_heads", kv_heads)?;
    require_positive("head_dim", head_dim)?;
    let projection_width = kv_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let latent_len = seq_len
        .checked_mul(latent_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let projection_len = latent_dim
        .checked_mul(projection_width)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let output_len = seq_len
        .checked_mul(projection_width)
        .ok_or(PlkvError::ArithmeticOverflow)?;

    if latent_cache.len() != latent_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "latent cache length does not match seq_len * latent_dim",
        });
    }
    if k_projection.len() != projection_len || v_projection.len() != projection_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "K/V projection length does not match latent_dim * projection_width",
        });
    }
    if !latent_cache.iter().all(|value| value.is_finite())
        || !k_projection.iter().all(|value| value.is_finite())
        || !v_projection.iter().all(|value| value.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "latent reconstruction inputs must be finite",
        });
    }

    let mut k_token_major = vec![0.0f32; output_len];
    let mut v_token_major = vec![0.0f32; output_len];
    for token in 0..seq_len {
        for out_dim in 0..projection_width {
            let mut k_sum = 0.0f32;
            let mut v_sum = 0.0f32;
            for latent_idx in 0..latent_dim {
                let latent_value = latent_cache[token * latent_dim + latent_idx];
                let projection_idx = latent_idx * projection_width + out_dim;
                k_sum += latent_value * k_projection[projection_idx];
                v_sum += latent_value * v_projection[projection_idx];
            }
            if !k_sum.is_finite() || !v_sum.is_finite() {
                return Err(PlkvError::InvalidPagedLookup {
                    reason: "latent reconstruction output must be finite",
                });
            }
            let output_idx = token * projection_width + out_dim;
            k_token_major[output_idx] = k_sum;
            v_token_major[output_idx] = v_sum;
        }
    }

    Ok(LatentKvReconstruction {
        k_token_major,
        v_token_major,
    })
}

pub fn direct_latent_gqa_decode_f32(
    q: &[f32],
    latent_cache: &[f32],
    k_projection: &[f32],
    v_projection: &[f32],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    latent_dim: usize,
    head_dim: usize,
    group_size: usize,
) -> Result<GqaDecodeResult, PlkvError> {
    require_positive("q_heads", q_heads)?;
    require_positive("kv_heads", kv_heads)?;
    require_positive("seq_len", seq_len)?;
    require_positive("latent_dim", latent_dim)?;
    require_positive("head_dim", head_dim)?;
    require_positive("group_size", group_size)?;
    if q_heads
        != kv_heads
            .checked_mul(group_size)
            .ok_or(PlkvError::ArithmeticOverflow)?
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "expected q_heads == kv_heads * group_size",
        });
    }
    let q_len = q_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if q.len() != q_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "Q length does not match q_heads * head_dim",
        });
    }
    let latent_len = seq_len
        .checked_mul(latent_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if latent_cache.len() != latent_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "latent cache length does not match seq_len * latent_dim",
        });
    }
    let projection_width = kv_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let projection_len = latent_dim
        .checked_mul(projection_width)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if k_projection.len() != projection_len || v_projection.len() != projection_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "projection length does not match latent_dim * kv_heads * head_dim",
        });
    }
    if !q.iter().all(|value| value.is_finite())
        || !latent_cache.iter().all(|value| value.is_finite())
        || !k_projection.iter().all(|value| value.is_finite())
        || !v_projection.iter().all(|value| value.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "direct latent GQA inputs must be finite",
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
        let mut projected_query_latent = vec![0.0f32; latent_dim];
        for latent_idx in 0..latent_dim {
            let mut value = 0.0f32;
            for dim in 0..head_dim {
                let projection_idx = (latent_idx * kv_heads + kv_head) * head_dim + dim;
                value += q[q_start + dim] * k_projection[projection_idx];
            }
            projected_query_latent[latent_idx] = value;
        }

        let scores_start = q_head * seq_len;
        for token in 0..seq_len {
            let latent_start = token * latent_dim;
            let mut score = 0.0f32;
            for latent_idx in 0..latent_dim {
                score +=
                    latent_cache[latent_start + latent_idx] * projected_query_latent[latent_idx];
            }
            scores[scores_start + token] = score * scale;
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

        let mut latent_context = vec![0.0f32; latent_dim];
        for latent_idx in 0..latent_dim {
            let mut value = 0.0f32;
            for token in 0..seq_len {
                value += probabilities[scores_start + token]
                    * latent_cache[token * latent_dim + latent_idx];
            }
            latent_context[latent_idx] = value;
        }

        let context_start = q_head * head_dim;
        for dim in 0..head_dim {
            let mut value = 0.0f32;
            for latent_idx in 0..latent_dim {
                let projection_idx = (latent_idx * kv_heads + kv_head) * head_dim + dim;
                value += latent_context[latent_idx] * v_projection[projection_idx];
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

pub fn direct_paged_latent_gqa_decode_f32(
    q: &[f32],
    latent_physical: &[f32],
    block_table: &[usize],
    k_projection: &[f32],
    v_projection: &[f32],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    latent_dim: usize,
    head_dim: usize,
    group_size: usize,
    block_size: usize,
    num_physical_blocks: usize,
) -> Result<GqaDecodeResult, PlkvError> {
    require_positive("q_heads", q_heads)?;
    require_positive("kv_heads", kv_heads)?;
    require_positive("seq_len", seq_len)?;
    require_positive("latent_dim", latent_dim)?;
    require_positive("head_dim", head_dim)?;
    require_positive("group_size", group_size)?;
    require_positive("block_size", block_size)?;
    require_positive("num_physical_blocks", num_physical_blocks)?;
    if q_heads
        != kv_heads
            .checked_mul(group_size)
            .ok_or(PlkvError::ArithmeticOverflow)?
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "invalid GQA mapping",
        });
    }
    let q_len = q_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let physical_len = num_physical_blocks
        .checked_mul(block_size)
        .and_then(|v| v.checked_mul(latent_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let projection_len = latent_dim
        .checked_mul(kv_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if q.len() != q_len
        || latent_physical.len() != physical_len
        || k_projection.len() != projection_len
        || v_projection.len() != projection_len
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "direct paged latent tensor length mismatch",
        });
    }
    if block_table
        .len()
        .checked_mul(block_size)
        .ok_or(PlkvError::ArithmeticOverflow)?
        < seq_len
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table does not cover sequence",
        });
    }
    if block_table.iter().any(|&p| p >= num_physical_blocks) {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table index out of range",
        });
    }
    if !q
        .iter()
        .chain(latent_physical)
        .chain(k_projection)
        .chain(v_projection)
        .all(|x| x.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "inputs must be finite",
        });
    }
    let mut scores = vec![0.0; q_heads * seq_len];
    let mut probabilities = scores.clone();
    let mut context = vec![0.0; q_heads * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..q_heads {
        let kv = h / group_size;
        let mut projected = vec![0.0; latent_dim];
        for l in 0..latent_dim {
            for d in 0..head_dim {
                projected[l] +=
                    q[h * head_dim + d] * k_projection[(l * kv_heads + kv) * head_dim + d];
            }
        }
        for t in 0..seq_len {
            let logical = t / block_size;
            let off = t % block_size;
            let base = (block_table[logical] * block_size + off) * latent_dim;
            scores[h * seq_len + t] = latent_physical[base..base + latent_dim]
                .iter()
                .zip(&projected)
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
        }
        let row = &scores[h * seq_len..(h + 1) * seq_len];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0;
        for t in 0..seq_len {
            probabilities[h * seq_len + t] = (scores[h * seq_len + t] - max).exp();
            denom += probabilities[h * seq_len + t];
        }
        for t in 0..seq_len {
            probabilities[h * seq_len + t] /= denom;
        }
        let mut latent_context = vec![0.0; latent_dim];
        for t in 0..seq_len {
            let base = (block_table[t / block_size] * block_size + t % block_size) * latent_dim;
            for l in 0..latent_dim {
                latent_context[l] += probabilities[h * seq_len + t] * latent_physical[base + l];
            }
        }
        for d in 0..head_dim {
            for l in 0..latent_dim {
                context[h * head_dim + d] +=
                    latent_context[l] * v_projection[(l * kv_heads + kv) * head_dim + d];
            }
        }
    }
    if !scores
        .iter()
        .chain(&probabilities)
        .chain(&context)
        .all(|x| x.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "outputs must be finite",
        });
    }
    Ok(GqaDecodeResult {
        scores,
        probabilities,
        context,
    })
}

pub fn paged_latent_write_fp16_storage(
    latent_cache: &mut [half::f16],
    block_table: &[usize],
    token_position: usize,
    block_size: usize,
    latent_dim: usize,
    new_latent_f32: &[f32],
) -> Result<PagedTokenLocation, PlkvError> {
    require_positive("block_size", block_size)?;
    require_positive("latent_dim", latent_dim)?;
    let block_stride = block_size
        .checked_mul(latent_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if latent_cache.is_empty() || latent_cache.len() % block_stride != 0 {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "FP16 latent cache length is not divisible by block stride",
        });
    }
    if new_latent_f32.len() != latent_dim {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "new latent vector does not match latent_dim",
        });
    }
    let replacement = quantize_f32_to_f16_storage(new_latent_f32)?;
    if latent_cache.iter().any(|value| !value.is_finite()) {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "FP16 latent cache must be finite",
        });
    }
    let location = resolve_paged_token_location(
        block_table,
        token_position,
        block_size,
        latent_cache.len() / block_stride,
    )?;
    let start = location
        .physical_block
        .checked_mul(block_stride)
        .and_then(|value| value.checked_add(location.block_offset.checked_mul(latent_dim)?))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let end = start
        .checked_add(latent_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    latent_cache
        .get_mut(start..end)
        .ok_or(PlkvError::InvalidPagedLookup {
            reason: "FP16 latent cache indexing exceeded storage length",
        })?
        .copy_from_slice(&replacement);
    Ok(location)
}

pub fn direct_paged_latent_gqa_decode_fp16_storage_f32_accum(
    q: &[f32],
    latent_physical: &[half::f16],
    block_table: &[usize],
    k_projection: &[f32],
    v_projection: &[f32],
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    latent_dim: usize,
    head_dim: usize,
    group_size: usize,
    block_size: usize,
    num_physical_blocks: usize,
) -> Result<GqaDecodeResult, PlkvError> {
    require_positive("q_heads", q_heads)?;
    require_positive("kv_heads", kv_heads)?;
    require_positive("seq_len", seq_len)?;
    require_positive("latent_dim", latent_dim)?;
    require_positive("head_dim", head_dim)?;
    require_positive("group_size", group_size)?;
    require_positive("block_size", block_size)?;
    require_positive("num_physical_blocks", num_physical_blocks)?;
    if q_heads
        != kv_heads
            .checked_mul(group_size)
            .ok_or(PlkvError::ArithmeticOverflow)?
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "invalid GQA mapping",
        });
    }
    let q_len = q_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let physical_len = num_physical_blocks
        .checked_mul(block_size)
        .and_then(|value| value.checked_mul(latent_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let projection_len = latent_dim
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if q.len() != q_len
        || latent_physical.len() != physical_len
        || k_projection.len() != projection_len
        || v_projection.len() != projection_len
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "FP16 direct paged latent tensor length mismatch",
        });
    }
    if block_table
        .len()
        .checked_mul(block_size)
        .ok_or(PlkvError::ArithmeticOverflow)?
        < seq_len
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table does not cover sequence",
        });
    }
    if block_table
        .iter()
        .any(|&physical| physical >= num_physical_blocks)
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table index out of range",
        });
    }
    if !q.iter().all(|value| value.is_finite())
        || !latent_physical.iter().all(|value| value.is_finite())
        || !k_projection.iter().all(|value| value.is_finite())
        || !v_projection.iter().all(|value| value.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "FP16 direct paged latent inputs must be finite",
        });
    }
    let mut scores = vec![0.0f32; q_heads * seq_len];
    let mut probabilities = scores.clone();
    let mut context = vec![0.0f32; q_heads * head_dim];
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for q_head in 0..q_heads {
        let kv_head = q_head / group_size;
        let mut projected = vec![0.0f32; latent_dim];
        for latent_idx in 0..latent_dim {
            for dim in 0..head_dim {
                projected[latent_idx] += q[q_head * head_dim + dim]
                    * k_projection[(latent_idx * kv_heads + kv_head) * head_dim + dim];
            }
        }
        for token in 0..seq_len {
            let base =
                (block_table[token / block_size] * block_size + token % block_size) * latent_dim;
            scores[q_head * seq_len + token] = (0..latent_dim)
                .map(|latent_idx| {
                    latent_physical[base + latent_idx].to_f32() * projected[latent_idx]
                })
                .sum::<f32>()
                * scale;
        }
        let row_start = q_head * seq_len;
        let row_max = scores[row_start..row_start + seq_len]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0f32;
        for token in 0..seq_len {
            let value = (scores[row_start + token] - row_max).exp();
            probabilities[row_start + token] = value;
            denominator += value;
        }
        for token in 0..seq_len {
            probabilities[row_start + token] /= denominator;
        }
        let mut latent_context = vec![0.0f32; latent_dim];
        for token in 0..seq_len {
            let base =
                (block_table[token / block_size] * block_size + token % block_size) * latent_dim;
            for latent_idx in 0..latent_dim {
                latent_context[latent_idx] +=
                    probabilities[row_start + token] * latent_physical[base + latent_idx].to_f32();
            }
        }
        for dim in 0..head_dim {
            for latent_idx in 0..latent_dim {
                context[q_head * head_dim + dim] += latent_context[latent_idx]
                    * v_projection[(latent_idx * kv_heads + kv_head) * head_dim + dim];
            }
        }
    }
    if !scores
        .iter()
        .chain(&probabilities)
        .chain(&context)
        .all(|value| value.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "FP16 direct paged latent outputs must be finite",
        });
    }
    Ok(GqaDecodeResult {
        scores,
        probabilities,
        context,
    })
}

pub fn direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
    q: &[f32],
    latent_physical: &[half::f16],
    block_table: &[usize],
    k_projection: &[f32],
    v_projection: &[f32],
    q_heads: usize,
    kv_heads: usize,
    max_seq_len: usize,
    active_seq_len: usize,
    latent_dim: usize,
    head_dim: usize,
    group_size: usize,
    block_size: usize,
    num_physical_blocks: usize,
) -> Result<GqaDecodeResult, PlkvError> {
    require_positive("active_seq_len", active_seq_len)?;
    if active_seq_len > max_seq_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "active sequence length exceeds max sequence length",
        });
    }
    let active = direct_paged_latent_gqa_decode_fp16_storage_f32_accum(
        q,
        latent_physical,
        block_table,
        k_projection,
        v_projection,
        q_heads,
        kv_heads,
        active_seq_len,
        latent_dim,
        head_dim,
        group_size,
        block_size,
        num_physical_blocks,
    )?;
    if active_seq_len == max_seq_len {
        return Ok(active);
    }
    let mut scores = vec![f32::MIN; q_heads * max_seq_len];
    let mut probabilities = vec![0.0f32; q_heads * max_seq_len];
    for head in 0..q_heads {
        let active_start = head * active_seq_len;
        let runtime_start = head * max_seq_len;
        scores[runtime_start..runtime_start + active_seq_len]
            .copy_from_slice(&active.scores[active_start..active_start + active_seq_len]);
        probabilities[runtime_start..runtime_start + active_seq_len]
            .copy_from_slice(&active.probabilities[active_start..active_start + active_seq_len]);
    }
    Ok(GqaDecodeResult {
        scores,
        probabilities,
        context: active.context,
    })
}

pub fn paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum(
    q: &[f32],
    k_physical: &[half::f16],
    v_physical: &[half::f16],
    block_table: &[usize],
    q_heads: usize,
    kv_heads: usize,
    max_seq_len: usize,
    active_seq_len: usize,
    head_dim: usize,
    group_size: usize,
    block_size: usize,
    num_physical_blocks: usize,
) -> Result<GqaDecodeResult, PlkvError> {
    require_positive("q_heads", q_heads)?;
    require_positive("kv_heads", kv_heads)?;
    require_positive("max_seq_len", max_seq_len)?;
    require_positive("active_seq_len", active_seq_len)?;
    require_positive("head_dim", head_dim)?;
    require_positive("group_size", group_size)?;
    require_positive("block_size", block_size)?;
    require_positive("num_physical_blocks", num_physical_blocks)?;
    if active_seq_len > max_seq_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "active sequence length exceeds max sequence length",
        });
    }
    if q_heads
        != kv_heads
            .checked_mul(group_size)
            .ok_or(PlkvError::ArithmeticOverflow)?
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "invalid GQA mapping",
        });
    }
    let q_len = q_heads
        .checked_mul(head_dim)
        .ok_or(PlkvError::ArithmeticOverflow)?;
    let physical_len = num_physical_blocks
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(block_size))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(PlkvError::ArithmeticOverflow)?;
    if q.len() != q_len || k_physical.len() != physical_len || v_physical.len() != physical_len {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "FP16 full-KV tensor length mismatch",
        });
    }
    if block_table
        .len()
        .checked_mul(block_size)
        .ok_or(PlkvError::ArithmeticOverflow)?
        < max_seq_len
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table does not cover max sequence",
        });
    }
    if block_table
        .iter()
        .any(|&physical| physical >= num_physical_blocks)
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "block table index out of range",
        });
    }
    if !q.iter().all(|value| value.is_finite())
        || !k_physical.iter().all(|value| value.is_finite())
        || !v_physical.iter().all(|value| value.is_finite())
    {
        return Err(PlkvError::InvalidPagedLookup {
            reason: "FP16 full-KV inputs must be finite",
        });
    }
    let mut scores = vec![f32::MIN; q_heads * max_seq_len];
    let mut probabilities = vec![0.0f32; q_heads * max_seq_len];
    let mut context = vec![0.0f32; q_heads * head_dim];
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    for q_head in 0..q_heads {
        let kv_head = q_head / group_size;
        let row_start = q_head * max_seq_len;
        for token in 0..active_seq_len {
            let logical_block = token / block_size;
            let offset = token % block_size;
            let base = (((block_table[logical_block] * kv_heads + kv_head) * block_size + offset)
                * head_dim) as usize;
            let mut dot = 0.0f32;
            for dim in 0..head_dim {
                dot += q[q_head * head_dim + dim] * k_physical[base + dim].to_f32();
            }
            scores[row_start + token] = dot * scale;
        }
        let row_max = scores[row_start..row_start + active_seq_len]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0f32;
        for token in 0..active_seq_len {
            let numerator = (scores[row_start + token] - row_max).exp();
            probabilities[row_start + token] = numerator;
            denominator += numerator;
        }
        for token in 0..active_seq_len {
            probabilities[row_start + token] /= denominator;
        }
        for token in 0..active_seq_len {
            let logical_block = token / block_size;
            let offset = token % block_size;
            let base = (((block_table[logical_block] * kv_heads + kv_head) * block_size + offset)
                * head_dim) as usize;
            let probability = probabilities[row_start + token];
            for dim in 0..head_dim {
                context[q_head * head_dim + dim] += probability * v_physical[base + dim].to_f32();
            }
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
        contiguous_gqa_decode_f32, direct_latent_gqa_decode_f32,
        direct_paged_latent_gqa_decode_f32, direct_paged_latent_gqa_decode_fp16_storage_f32_accum,
        direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum,
        estimate_total_kv_cache_bytes, kv_bytes_per_token_gqa, kv_bytes_per_token_latent,
        paged_gqa_decode_f32, paged_kv_write_f32, paged_latent_write_f32, paged_lookup_f32,
        quantize_f32_to_f16_storage, reconstruct_latent_kv_f32, resolve_paged_token_location,
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

    #[derive(Debug, Deserialize)]
    struct PagedGqaFixture {
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        seq_len: usize,
        head_dim: usize,
        block_size: usize,
        num_logical_blocks: usize,
        num_physical_blocks: usize,
        block_table: Vec<usize>,
        q_to_kv: Vec<usize>,
        cases: Vec<PagedGqaCase>,
    }

    #[derive(Debug, Deserialize)]
    struct PagedGqaCase {
        q: Vec<Vec<f32>>,
        k_physical_gpu_head_major: Vec<Vec<Vec<Vec<f32>>>>,
        v_physical_gpu_head_major: Vec<Vec<Vec<Vec<f32>>>>,
        expected_scores: Vec<Vec<f32>>,
        expected_probabilities: Vec<Vec<f32>>,
        expected_context: Vec<Vec<f32>>,
    }

    #[derive(Debug, Deserialize)]
    struct LatentKvFixture {
        seq_len: usize,
        latent_dim: usize,
        kv_heads: usize,
        head_dim: usize,
        projection_width: usize,
        theoretical_cache_compression_ratio: f32,
        cases: Vec<LatentKvCase>,
    }

    #[derive(Debug, Deserialize)]
    struct LatentKvCase {
        latent_cache: Vec<Vec<f32>>,
        k_projection: Vec<Vec<f32>>,
        v_projection: Vec<Vec<f32>>,
        expected_k_token_major: Vec<Vec<Vec<f32>>>,
        expected_v_token_major: Vec<Vec<Vec<f32>>>,
        expected_k_head_major: Vec<Vec<Vec<f32>>>,
        expected_v_head_major: Vec<Vec<Vec<f32>>>,
    }

    #[derive(Debug, Deserialize)]
    struct DirectLatentGqaFixture {
        dtype: String,
        batch: usize,
        seq_len: usize,
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        head_dim: usize,
        latent_dim: usize,
        projection_width: usize,
        q_to_kv: Vec<usize>,
        scale: f32,
        latent_values_per_token: usize,
        full_kv_values_per_token: usize,
        theoretical_cache_compression_ratio: f32,
        cases: Vec<DirectLatentGqaCase>,
    }

    #[derive(Debug, Deserialize)]
    struct DirectLatentGqaCase {
        name: String,
        q: Vec<Vec<f32>>,
        latent_cache: Vec<Vec<f32>>,
        k_projection: Vec<Vec<f32>>,
        v_projection: Vec<Vec<f32>>,
        k_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
        v_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
        expected_scores: Vec<Vec<f32>>,
        expected_probabilities: Vec<Vec<f32>>,
        expected_context: Vec<Vec<f32>>,
        materialized_scores: Vec<Vec<f32>>,
        materialized_probabilities: Vec<Vec<f32>>,
        materialized_context: Vec<Vec<f32>>,
    }

    #[derive(Debug, Deserialize)]
    struct PagedLatentWriteFixture {
        seq_len: usize,
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        head_dim: usize,
        latent_dim: usize,
        block_size: usize,
        num_physical_blocks: usize,
        block_table: Vec<usize>,
        cases: Vec<PagedLatentWriteCase>,
    }

    #[derive(Debug, Deserialize)]
    struct PagedLatentWriteCase {
        token_position: usize,
        logical_block: usize,
        physical_block: usize,
        block_offset: usize,
        q: Vec<Vec<f32>>,
        initial_latent_physical_blocks: Vec<Vec<Vec<f32>>>,
        new_latent: Vec<f32>,
        expected_updated_latent_physical_blocks: Vec<Vec<Vec<f32>>>,
        k_projection: Vec<Vec<f32>>,
        v_projection: Vec<Vec<f32>>,
        post_write_scores: Vec<Vec<f32>>,
        post_write_probabilities: Vec<Vec<f32>>,
        post_write_context: Vec<Vec<f32>>,
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
        let fixture: GqaDecodeFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/gqa_decode_f32.json"
        ))
        .unwrap();
        assert_eq!(fixture.q_to_kv, vec![0, 0, 1, 1]);
        for case in fixture.cases {
            let q: Vec<f32> = case.q.iter().flatten().copied().collect();
            let k: Vec<f32> = case
                .k_head_major
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let v: Vec<f32> = case
                .v_head_major
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let expected_scores: Vec<f32> =
                case.expected_scores.iter().flatten().copied().collect();
            let expected_probabilities: Vec<f32> = case
                .expected_probabilities
                .iter()
                .flatten()
                .copied()
                .collect();
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
        assert!(
            contiguous_gqa_decode_f32(&[0.0; 32], &[0.0; 128], &[0.0; 128], 3, 2, 8, 8, 2).is_err()
        );
        assert!(
            contiguous_gqa_decode_f32(&[0.0; 31], &[0.0; 128], &[0.0; 128], 4, 2, 8, 8, 2).is_err()
        );
        assert!(
            contiguous_gqa_decode_f32(&[0.0; 32], &[0.0; 127], &[0.0; 128], 4, 2, 8, 8, 2).is_err()
        );
    }

    #[test]
    fn paged_gqa_decode_matches_python_fixture() {
        let fixture: PagedGqaFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/paged_gqa_decode_f32.json"
        ))
        .unwrap();
        assert_eq!(fixture.block_table, vec![2, 0, 3, 1]);
        assert_eq!(fixture.q_to_kv, vec![0, 0, 1, 1]);
        assert_eq!(fixture.num_logical_blocks, 4);

        for case in fixture.cases {
            let q: Vec<f32> = case.q.iter().flatten().copied().collect();
            let k: Vec<f32> = case
                .k_physical_gpu_head_major
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let v: Vec<f32> = case
                .v_physical_gpu_head_major
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let expected_scores: Vec<f32> =
                case.expected_scores.iter().flatten().copied().collect();
            let expected_probabilities: Vec<f32> = case
                .expected_probabilities
                .iter()
                .flatten()
                .copied()
                .collect();
            let expected_context: Vec<f32> =
                case.expected_context.iter().flatten().copied().collect();
            let actual = paged_gqa_decode_f32(
                &q,
                &k,
                &v,
                &fixture.block_table,
                fixture.q_heads,
                fixture.kv_heads,
                fixture.seq_len,
                fixture.head_dim,
                fixture.group_size,
                fixture.block_size,
                fixture.num_physical_blocks,
            )
            .unwrap();
            assert_close(&actual.scores, &expected_scores, 1e-5, 1e-5);
            assert_close(&actual.probabilities, &expected_probabilities, 1e-5, 1e-5);
            assert_close(&actual.context, &expected_context, 1e-5, 1e-5);
            assert!(actual.scores.iter().all(|value| value.is_finite()));
            assert!(actual.probabilities.iter().all(|value| value.is_finite()));
            assert!(actual.context.iter().all(|value| value.is_finite()));
            for q_head in 0..fixture.q_heads {
                let start = q_head * fixture.seq_len;
                let row_sum: f32 = actual.probabilities[start..start + fixture.seq_len]
                    .iter()
                    .sum();
                assert!((row_sum - 1.0).abs() <= 1e-5);
            }
        }
    }

    #[test]
    fn paged_gqa_decode_rejects_invalid_inputs() {
        let q = vec![0.0f32; 32];
        let physical = vec![0.0f32; 128];
        assert!(
            paged_gqa_decode_f32(&q, &physical, &physical, &[2, 0, 3], 4, 2, 8, 8, 2, 2, 4)
                .is_err()
        );
        assert!(
            paged_gqa_decode_f32(&q, &physical, &physical, &[2, 0, 4, 1], 4, 2, 8, 8, 2, 2, 4)
                .is_err()
        );
        assert!(
            paged_gqa_decode_f32(
                &q,
                &physical[..127],
                &physical,
                &[2, 0, 3, 1],
                4,
                2,
                8,
                8,
                2,
                2,
                4
            )
            .is_err()
        );
        assert!(
            paged_gqa_decode_f32(
                &q,
                &physical,
                &physical[..127],
                &[2, 0, 3, 1],
                4,
                2,
                8,
                8,
                2,
                2,
                4
            )
            .is_err()
        );
        assert!(
            paged_gqa_decode_f32(&q, &physical, &physical, &[2, 0, 3, 1], 4, 2, 8, 8, 2, 0, 4)
                .is_err()
        );
        assert!(
            paged_gqa_decode_f32(&q, &physical, &physical, &[2, 0, 3, 1], 3, 2, 8, 8, 2, 2, 4)
                .is_err()
        );
    }

    #[test]
    fn latent_kv_reconstruction_matches_python_fixture() {
        let fixture: LatentKvFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/latent_kv_reconstruction_f32.json"
        ))
        .unwrap();
        assert_eq!(
            fixture.projection_width,
            fixture.kv_heads * fixture.head_dim
        );
        assert_eq!(fixture.theoretical_cache_compression_ratio, 4.0);

        for case in fixture.cases {
            let latent: Vec<f32> = case.latent_cache.iter().flatten().copied().collect();
            let k_projection: Vec<f32> = case.k_projection.iter().flatten().copied().collect();
            let v_projection: Vec<f32> = case.v_projection.iter().flatten().copied().collect();
            let expected_k: Vec<f32> = case
                .expected_k_token_major
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let expected_v: Vec<f32> = case
                .expected_v_token_major
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let actual = reconstruct_latent_kv_f32(
                &latent,
                &k_projection,
                &v_projection,
                fixture.seq_len,
                fixture.latent_dim,
                fixture.kv_heads,
                fixture.head_dim,
            )
            .unwrap();
            assert_close(&actual.k_token_major, &expected_k, 1e-5, 1e-5);
            assert_close(&actual.v_token_major, &expected_v, 1e-5, 1e-5);
            assert!(actual.k_token_major.iter().all(|value| value.is_finite()));
            assert!(actual.v_token_major.iter().all(|value| value.is_finite()));

            let expected_k_head: Vec<f32> = case
                .expected_k_head_major
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let expected_v_head: Vec<f32> = case
                .expected_v_head_major
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            assert_close(
                &token_major_to_head_major(
                    &actual.k_token_major,
                    fixture.seq_len,
                    fixture.kv_heads,
                    fixture.head_dim,
                ),
                &expected_k_head,
                1e-5,
                1e-5,
            );
            assert_close(
                &token_major_to_head_major(
                    &actual.v_token_major,
                    fixture.seq_len,
                    fixture.kv_heads,
                    fixture.head_dim,
                ),
                &expected_v_head,
                1e-5,
                1e-5,
            );
        }
    }

    #[test]
    fn latent_kv_reconstruction_rejects_invalid_inputs() {
        let latent = vec![0.0f32; 64];
        let projection = vec![0.0f32; 128];
        assert!(reconstruct_latent_kv_f32(&latent, &projection, &projection, 0, 8, 2, 8).is_err());
        assert!(
            reconstruct_latent_kv_f32(&latent[..63], &projection, &projection, 8, 8, 2, 8).is_err()
        );
        assert!(
            reconstruct_latent_kv_f32(&latent, &projection[..127], &projection, 8, 8, 2, 8)
                .is_err()
        );
        assert!(
            reconstruct_latent_kv_f32(&latent, &projection, &projection[..127], 8, 8, 2, 8)
                .is_err()
        );
        let mut non_finite = latent.clone();
        non_finite[0] = f32::NAN;
        assert!(
            reconstruct_latent_kv_f32(&non_finite, &projection, &projection, 8, 8, 2, 8).is_err()
        );
    }

    #[test]
    fn direct_latent_gqa_matches_python_and_materialized_fixtures() {
        let fixture: DirectLatentGqaFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/direct_latent_gqa_decode_f32.json"
        ))
        .unwrap();
        assert_eq!(fixture.dtype, "f32");
        assert_eq!(fixture.batch, 1);
        assert_eq!(fixture.q_to_kv, vec![0, 0, 1, 1]);
        assert_eq!(
            fixture.projection_width,
            fixture.kv_heads * fixture.head_dim
        );
        assert_eq!(fixture.latent_values_per_token, fixture.latent_dim);
        assert_eq!(fixture.full_kv_values_per_token, 32);
        assert_eq!(fixture.theoretical_cache_compression_ratio, 4.0);

        for case in fixture.cases {
            let q: Vec<f32> = case.q.iter().flatten().copied().collect();
            let latent: Vec<f32> = case.latent_cache.iter().flatten().copied().collect();
            let k_projection: Vec<f32> = case.k_projection.iter().flatten().copied().collect();
            let v_projection: Vec<f32> = case.v_projection.iter().flatten().copied().collect();
            let expected_scores: Vec<f32> =
                case.expected_scores.iter().flatten().copied().collect();
            let expected_probabilities: Vec<f32> = case
                .expected_probabilities
                .iter()
                .flatten()
                .copied()
                .collect();
            let expected_context: Vec<f32> =
                case.expected_context.iter().flatten().copied().collect();
            let materialized_scores: Vec<f32> =
                case.materialized_scores.iter().flatten().copied().collect();
            let materialized_probabilities: Vec<f32> = case
                .materialized_probabilities
                .iter()
                .flatten()
                .copied()
                .collect();
            let materialized_context: Vec<f32> = case
                .materialized_context
                .iter()
                .flatten()
                .copied()
                .collect();

            let direct = direct_latent_gqa_decode_f32(
                &q,
                &latent,
                &k_projection,
                &v_projection,
                fixture.q_heads,
                fixture.kv_heads,
                fixture.seq_len,
                fixture.latent_dim,
                fixture.head_dim,
                fixture.group_size,
            )
            .unwrap();
            let reconstructed = reconstruct_latent_kv_f32(
                &latent,
                &k_projection,
                &v_projection,
                fixture.seq_len,
                fixture.latent_dim,
                fixture.kv_heads,
                fixture.head_dim,
            )
            .unwrap();
            let reconstructed_k_head_major = token_major_to_head_major(
                &reconstructed.k_token_major,
                fixture.seq_len,
                fixture.kv_heads,
                fixture.head_dim,
            );
            let reconstructed_v_head_major = token_major_to_head_major(
                &reconstructed.v_token_major,
                fixture.seq_len,
                fixture.kv_heads,
                fixture.head_dim,
            );
            let materialized = contiguous_gqa_decode_f32(
                &q,
                &reconstructed_k_head_major,
                &reconstructed_v_head_major,
                fixture.q_heads,
                fixture.kv_heads,
                fixture.seq_len,
                fixture.head_dim,
                fixture.group_size,
            )
            .unwrap();

            assert_close(&direct.scores, &expected_scores, 5e-5, 1e-5);
            assert_close(&direct.probabilities, &expected_probabilities, 5e-5, 1e-5);
            assert_close(&direct.context, &expected_context, 1e-4, 1e-5);
            assert_close(&direct.scores, &materialized_scores, 5e-5, 1e-5);
            assert_close(
                &direct.probabilities,
                &materialized_probabilities,
                5e-5,
                1e-5,
            );
            assert_close(&direct.context, &materialized_context, 1e-4, 1e-5);
            assert_close(&direct.scores, &materialized.scores, 5e-5, 1e-5);
            assert_close(
                &direct.probabilities,
                &materialized.probabilities,
                5e-5,
                1e-5,
            );
            assert_close(&direct.context, &materialized.context, 1e-4, 1e-5);
            assert!(direct.scores.iter().all(|value| value.is_finite()));
            assert!(direct.probabilities.iter().all(|value| value.is_finite()));
            assert!(direct.context.iter().all(|value| value.is_finite()));
            for q_head in 0..fixture.q_heads {
                let start = q_head * fixture.seq_len;
                let row_sum: f32 = direct.probabilities[start..start + fixture.seq_len]
                    .iter()
                    .sum();
                assert!((row_sum - 1.0).abs() <= 1e-5);
            }

            assert_eq!(reconstructed.k_token_major.len(), 128);
            assert_eq!(reconstructed.v_token_major.len(), 128);
        }
    }

    #[test]
    fn direct_latent_gqa_rejects_invalid_inputs() {
        let q = vec![0.0f32; 32];
        let latent = vec![0.0f32; 64];
        let projection = vec![0.0f32; 128];
        assert!(
            direct_latent_gqa_decode_f32(
                &q[..31],
                &latent,
                &projection,
                &projection,
                4,
                2,
                8,
                8,
                8,
                2
            )
            .is_err()
        );
        assert!(
            direct_latent_gqa_decode_f32(
                &q,
                &latent[..63],
                &projection,
                &projection,
                4,
                2,
                8,
                8,
                8,
                2
            )
            .is_err()
        );
        assert!(
            direct_latent_gqa_decode_f32(
                &q,
                &latent,
                &projection[..127],
                &projection,
                4,
                2,
                8,
                8,
                8,
                2
            )
            .is_err()
        );
        assert!(
            direct_latent_gqa_decode_f32(
                &q,
                &latent,
                &projection,
                &projection[..127],
                4,
                2,
                8,
                8,
                8,
                2
            )
            .is_err()
        );
        assert!(
            direct_latent_gqa_decode_f32(&q, &latent, &projection, &projection, 3, 2, 8, 8, 8, 2)
                .is_err()
        );
        assert!(
            direct_latent_gqa_decode_f32(&q, &latent, &projection, &projection, 4, 2, 8, 8, 8, 0)
                .is_err()
        );
    }

    #[test]
    fn paged_latent_write_matches_python_and_post_write_attention_fixture() {
        let fixture: PagedLatentWriteFixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/paged_latent_write_attention_f32.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let initial = flatten_3d(&case.initial_latent_physical_blocks);
            let expected = flatten_3d(&case.expected_updated_latent_physical_blocks);
            let mut updated = initial.clone();
            let location = paged_latent_write_f32(
                &mut updated,
                &fixture.block_table,
                case.token_position,
                fixture.block_size,
                fixture.latent_dim,
                &case.new_latent,
            )
            .unwrap();
            assert_eq!(location.logical_block, case.logical_block);
            assert_eq!(location.physical_block, case.physical_block);
            assert_eq!(location.block_offset, case.block_offset);
            assert_eq!(updated, expected);
            let result = direct_paged_latent_gqa_decode_f32(
                &flatten_2d(&case.q),
                &updated,
                &fixture.block_table,
                &flatten_2d(&case.k_projection),
                &flatten_2d(&case.v_projection),
                fixture.q_heads,
                fixture.kv_heads,
                fixture.seq_len,
                fixture.latent_dim,
                fixture.head_dim,
                fixture.group_size,
                fixture.block_size,
                fixture.num_physical_blocks,
            )
            .unwrap();
            assert_close(
                &result.scores,
                &flatten_2d(&case.post_write_scores),
                2e-4,
                1e-5,
            );
            assert_close(
                &result.probabilities,
                &flatten_2d(&case.post_write_probabilities),
                1e-4,
                1e-5,
            );
            assert_close(
                &result.context,
                &flatten_2d(&case.post_write_context),
                1e-4,
                1e-5,
            );
        }
        let mut cache = vec![0.0f32; 64];
        assert!(paged_latent_write_f32(&mut cache, &[2, 0, 3, 1], 8, 2, 8, &[0.0; 8]).is_err());
        assert!(paged_latent_write_f32(&mut cache, &[4, 0, 3, 1], 0, 2, 8, &[0.0; 8]).is_err());
        assert!(paged_latent_write_f32(&mut cache, &[2, 0, 3, 1], 0, 2, 8, &[0.0; 7]).is_err());
    }

    #[test]
    fn fp16_runtime_active_lengths_mask_inactive_tokens() {
        let q_heads = 4;
        let kv_heads = 2;
        let group_size = 2;
        let head_dim = 8;
        let latent_dim = 8;
        let block_size = 2;
        let max_seq_len = 8;
        let num_physical_blocks = 4;
        let block_table = vec![2, 0, 3, 1];
        let q: Vec<f32> = (0..q_heads * head_dim)
            .map(|value| value as f32 / 17.0)
            .collect();
        let mut logical = vec![0.0f32; max_seq_len * latent_dim];
        for (index, value) in logical.iter_mut().enumerate() {
            *value = index as f32 / 31.0;
        }
        let mut physical = vec![0.0f32; num_physical_blocks * block_size * latent_dim];
        for (logical_block, &physical_block) in block_table.iter().enumerate() {
            let logical_start = logical_block * block_size * latent_dim;
            let physical_start = physical_block * block_size * latent_dim;
            physical[physical_start..physical_start + block_size * latent_dim]
                .copy_from_slice(&logical[logical_start..logical_start + block_size * latent_dim]);
        }
        let latent_fp16 = quantize_f32_to_f16_storage(&physical).unwrap();
        let k_projection: Vec<f32> = (0..latent_dim * kv_heads * head_dim)
            .map(|value| value as f32 / 101.0)
            .collect();
        let v_projection: Vec<f32> = (0..latent_dim * kv_heads * head_dim)
            .map(|value| (value as f32 + 3.0) / 89.0)
            .collect();

        for active_seq_len in [1, 3, 4, 7, 8] {
            let runtime = direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
                &q,
                &latent_fp16,
                &block_table,
                &k_projection,
                &v_projection,
                q_heads,
                kv_heads,
                max_seq_len,
                active_seq_len,
                latent_dim,
                head_dim,
                group_size,
                block_size,
                num_physical_blocks,
            )
            .unwrap();
            let active = direct_paged_latent_gqa_decode_fp16_storage_f32_accum(
                &q,
                &latent_fp16,
                &block_table,
                &k_projection,
                &v_projection,
                q_heads,
                kv_heads,
                active_seq_len,
                latent_dim,
                head_dim,
                group_size,
                block_size,
                num_physical_blocks,
            )
            .unwrap();
            assert_eq!(runtime.scores.len(), q_heads * max_seq_len);
            assert_eq!(runtime.probabilities.len(), q_heads * max_seq_len);
            assert_eq!(runtime.context, active.context);
            for head in 0..q_heads {
                let active_start = head * active_seq_len;
                let runtime_start = head * max_seq_len;
                assert_close(
                    &runtime.scores[runtime_start..runtime_start + active_seq_len],
                    &active.scores[active_start..active_start + active_seq_len],
                    0.0,
                    0.0,
                );
                assert_close(
                    &runtime.probabilities[runtime_start..runtime_start + active_seq_len],
                    &active.probabilities[active_start..active_start + active_seq_len],
                    0.0,
                    0.0,
                );
                assert!(
                    runtime.probabilities
                        [runtime_start + active_seq_len..runtime_start + max_seq_len]
                        .iter()
                        .all(|&value| value == 0.0)
                );
                let row_sum: f32 = runtime.probabilities
                    [runtime_start..runtime_start + max_seq_len]
                    .iter()
                    .sum();
                assert!((row_sum - 1.0).abs() <= 1e-6);
            }
        }
        assert!(
            direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
                &q,
                &latent_fp16,
                &block_table,
                &k_projection,
                &v_projection,
                q_heads,
                kv_heads,
                max_seq_len,
                0,
                latent_dim,
                head_dim,
                group_size,
                block_size,
                num_physical_blocks,
            )
            .is_err()
        );
    }

    fn flatten_2d(values: &[Vec<f32>]) -> Vec<f32> {
        values.iter().flatten().copied().collect()
    }

    fn flatten_3d(values: &[Vec<Vec<f32>>]) -> Vec<f32> {
        values.iter().flatten().flatten().copied().collect()
    }

    fn token_major_to_head_major(
        token_major: &[f32],
        seq_len: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; token_major.len()];
        for kv_head in 0..kv_heads {
            for token in 0..seq_len {
                for dim in 0..head_dim {
                    let src = (token * kv_heads + kv_head) * head_dim + dim;
                    let dst = (kv_head * seq_len + token) * head_dim + dim;
                    out[dst] = token_major[src];
                }
            }
        }
        out
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
