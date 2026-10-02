//! Host-side block-table validation before a paged-decode launch.
//!
//! GPU kernels index physical cache blocks with entries loaded from the block table and do not
//! bounds-check them. Every launch must therefore validate the table on the host first
//! (`docs/E3_SPLITK_DESIGN.md` §0(e)). Duplicate physical blocks are legal for read-only decode
//! (aliasing), so they are reported but not rejected.

use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockTableError {
    ZeroBlockSize,
    TooShort {
        entries: usize,
        required: usize,
    },
    OutOfRange {
        logical: usize,
        physical: i64,
        num_physical_blocks: usize,
    },
}

impl fmt::Display for BlockTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlockSize => write!(f, "block_size must be positive"),
            Self::TooShort { entries, required } => write!(
                f,
                "block table has {entries} entries but {required} logical blocks are read"
            ),
            Self::OutOfRange {
                logical,
                physical,
                num_physical_blocks,
            } => write!(
                f,
                "logical block {logical} maps to physical block {physical}, outside [0, {num_physical_blocks})"
            ),
        }
    }
}

impl std::error::Error for BlockTableError {}

/// Outcome of a successful validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTableReport {
    /// Logical blocks the kernel reads (may exceed the active length: split kernels read every
    /// table entry they are launched over and mask inactive tokens).
    pub blocks_read: usize,
    /// Physical blocks referenced by more than one logical block.
    pub duplicate_physical_blocks: Vec<i64>,
}

/// Validates the first `blocks_read` entries of `table` for a launch that covers
/// `active_seq_len` tokens. `blocks_read` must be at least `ceil(active_seq_len / block_size)`;
/// every read entry must lie in `[0, num_physical_blocks)`.
pub fn validate_block_table(
    table: &[i32],
    active_seq_len: usize,
    block_size: usize,
    blocks_read: usize,
    num_physical_blocks: usize,
) -> Result<BlockTableReport, BlockTableError> {
    if block_size == 0 {
        return Err(BlockTableError::ZeroBlockSize);
    }
    let required = active_seq_len.div_ceil(block_size).max(blocks_read);
    if table.len() < required {
        return Err(BlockTableError::TooShort {
            entries: table.len(),
            required,
        });
    }
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for (logical, &physical) in table[..required].iter().enumerate() {
        if physical < 0 || physical as usize >= num_physical_blocks {
            return Err(BlockTableError::OutOfRange {
                logical,
                physical: i64::from(physical),
                num_physical_blocks,
            });
        }
        if !seen.insert(physical) {
            duplicates.insert(i64::from(physical));
        }
    }
    Ok(BlockTableReport {
        blocks_read: required,
        duplicate_physical_blocks: duplicates.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_non_identity_permutation() {
        let table: Vec<i32> = (0..64).map(|l| (l * 17 + 11) % 64).collect();
        let report = validate_block_table(&table, 1024, 16, 64, 64).unwrap();
        assert_eq!(report.blocks_read, 64);
        assert!(report.duplicate_physical_blocks.is_empty());
    }

    #[test]
    fn partial_final_block_requires_ceil_blocks() {
        // 17 tokens with 16-token blocks read 2 logical blocks.
        assert_eq!(
            validate_block_table(&[3], 17, 16, 0, 4),
            Err(BlockTableError::TooShort {
                entries: 1,
                required: 2
            })
        );
        assert!(validate_block_table(&[3, 1], 17, 16, 0, 4).is_ok());
    }

    #[test]
    fn rejects_out_of_range_and_negative_entries() {
        assert_eq!(
            validate_block_table(&[0, 4], 32, 16, 2, 4),
            Err(BlockTableError::OutOfRange {
                logical: 1,
                physical: 4,
                num_physical_blocks: 4
            })
        );
        assert!(matches!(
            validate_block_table(&[-1, 0], 32, 16, 2, 4),
            Err(BlockTableError::OutOfRange { physical: -1, .. })
        ));
    }

    #[test]
    fn checks_every_block_the_kernel_reads_not_only_active_ones() {
        // Active length covers one block, but a split kernel launched over 4 blocks reads all 4.
        assert!(matches!(
            validate_block_table(&[0, 1, 2, 99], 5, 16, 4, 4),
            Err(BlockTableError::OutOfRange { logical: 3, .. })
        ));
    }

    #[test]
    fn reports_duplicates_without_rejecting() {
        let report = validate_block_table(&[2, 2, 0, 0], 64, 16, 4, 4).unwrap();
        assert_eq!(report.duplicate_physical_blocks, vec![0, 2]);
    }

    #[test]
    fn rejects_zero_block_size() {
        assert_eq!(
            validate_block_table(&[0], 1, 0, 1, 1),
            Err(BlockTableError::ZeroBlockSize)
        );
    }
}
