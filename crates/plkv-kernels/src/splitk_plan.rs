//! Split-K launch planning for the E3 kernels.
//!
//! Each split CTA walks `steps_per_split` steps of `BLOCKS_PER_STEP` logical blocks. The scale
//! validation (docs/E3_RESULTS.md) found that error grows when one CTA accumulates ~1000
//! sequential tensor-core steps (C3 and padded A3, uniform inputs, N = 32768, one split:
//! 2.9e-3 / 3.1e-3) while 128-step CTAs stayed at 5-8e-4. The rule therefore bounds the work per
//! CTA: `splits >= ceil(steps_needed / MAX_STEPS_PER_CTA)`.

/// Logical blocks gathered per E3 step (two blocks = 32 tokens at block size 16).
pub const BLOCKS_PER_STEP: usize = 2;
/// Upper bound on sequential steps per split CTA (128 steps = 4096 tokens at block size 16).
pub const MAX_STEPS_PER_CTA: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitPlan {
    pub splits: usize,
    pub steps_per_split: usize,
    /// Logical blocks the launch reads (`splits * steps_per_split * BLOCKS_PER_STEP`); the tail
    /// beyond the active length is masked in-kernel and must still be valid table entries.
    pub blocks_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    ZeroActiveLength,
    ZeroBlockSize,
    ExceedsTable {
        blocks_read: usize,
        table_blocks: usize,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroActiveLength => write!(f, "active length must be positive"),
            Self::ZeroBlockSize => write!(f, "block size must be positive"),
            Self::ExceedsTable {
                blocks_read,
                table_blocks,
            } => write!(
                f,
                "plan reads {blocks_read} blocks but the table has {table_blocks}"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// Plans a launch over the active range: the smallest split count with at most
/// `MAX_STEPS_PER_CTA` steps per CTA, raised to `min_splits` if requested (never above the
/// number of steps). Fails if the rounded-up range would read past the block table.
pub fn plan_bounded(
    active_seq_len: usize,
    block_size: usize,
    table_blocks: usize,
    min_splits: usize,
) -> Result<SplitPlan, PlanError> {
    if active_seq_len == 0 {
        return Err(PlanError::ZeroActiveLength);
    }
    if block_size == 0 {
        return Err(PlanError::ZeroBlockSize);
    }
    let steps_needed = active_seq_len
        .div_ceil(block_size)
        .div_ceil(BLOCKS_PER_STEP);
    let splits = steps_needed
        .div_ceil(MAX_STEPS_PER_CTA)
        .max(min_splits)
        .min(steps_needed)
        .max(1);
    let steps_per_split = steps_needed.div_ceil(splits);
    let blocks_read = splits * steps_per_split * BLOCKS_PER_STEP;
    if blocks_read > table_blocks {
        return Err(PlanError::ExceedsTable {
            blocks_read,
            table_blocks,
        });
    }
    Ok(SplitPlan {
        splits,
        steps_per_split,
        blocks_read,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_counts_for_documented_lengths() {
        // model_small block size 16: 1K -> 32 steps, 8K -> 256, 32K -> 1024.
        let p1k = plan_bounded(1024, 16, 64, 1).unwrap();
        let p8k = plan_bounded(8192, 16, 512, 1).unwrap();
        let p32k = plan_bounded(32768, 16, 2048, 1).unwrap();
        assert_eq!((p1k.splits, p1k.steps_per_split), (1, 32));
        assert_eq!((p8k.splits, p8k.steps_per_split), (2, 128));
        assert_eq!((p32k.splits, p32k.steps_per_split), (8, 128));
    }

    #[test]
    fn never_exceeds_step_bound() {
        for active in [1usize, 17, 4095, 4096, 4097, 8191, 32768] {
            let p = plan_bounded(active, 16, 2048, 1).unwrap();
            assert!(
                p.steps_per_split <= MAX_STEPS_PER_CTA,
                "active {active}: {p:?}"
            );
            assert!(p.blocks_read * 16 >= active);
        }
    }

    #[test]
    fn min_splits_raises_but_never_exceeds_steps() {
        assert_eq!(plan_bounded(1024, 16, 64, 4).unwrap().splits, 4);
        assert_eq!(plan_bounded(17, 16, 64, 8).unwrap().splits, 1);
    }

    #[test]
    fn rejects_reading_past_table_and_bad_inputs() {
        assert_eq!(
            plan_bounded(1024, 16, 63, 1),
            Err(PlanError::ExceedsTable {
                blocks_read: 64,
                table_blocks: 63
            })
        );
        assert_eq!(plan_bounded(0, 16, 64, 1), Err(PlanError::ZeroActiveLength));
        assert_eq!(plan_bounded(16, 0, 64, 1), Err(PlanError::ZeroBlockSize));
    }
}
