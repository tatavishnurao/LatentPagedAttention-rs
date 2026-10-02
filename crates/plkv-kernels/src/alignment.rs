//! Host-side guard for kernels compiled with a forced divisibility hint.
//!
//! `CompileOptions::max_divisibility(16)` lets cuTile emit 128-bit vector loads
//! (`docs/MMA_PROBE_PLAN.md`, probe P1). The hint is a promise, not a check: if a tensor's
//! base address or a non-unit stride is not a multiple of 16 bytes, vector loads read the wrong
//! addresses. cuTile's own runtime path derives divisibility from the real pointer, so this guard
//! only matters when the hint is forced. Call [`check_layout`] at launch time with the actual
//! device pointer (`Tensor::device_pointer().cu_deviceptr()`) and strides (`Tensor::strides()`).

use std::fmt;

/// Byte alignment required by `max_divisibility(16)`.
pub const VECTOR_ALIGN_BYTES: u64 = 16;

/// Address, element strides and element size of one kernel tensor argument.
#[derive(Debug, Clone, Copy)]
pub struct TensorLayout<'a> {
    pub name: &'a str,
    pub base_ptr: u64,
    pub strides: &'a [i32],
    pub elem_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignmentError {
    MisalignedBase {
        name: String,
        base_ptr: u64,
        align: u64,
    },
    MisalignedStride {
        name: String,
        dim: usize,
        stride_bytes: i64,
        align: u64,
    },
    NonUnitInnerStride {
        name: String,
        stride: i32,
    },
}

impl fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MisalignedBase {
                name,
                base_ptr,
                align,
            } => write!(
                f,
                "{name}: base pointer {base_ptr:#x} is not {align}-byte aligned"
            ),
            Self::MisalignedStride {
                name,
                dim,
                stride_bytes,
                align,
            } => write!(
                f,
                "{name}: stride of dim {dim} is {stride_bytes} bytes, not a multiple of {align}"
            ),
            Self::NonUnitInnerStride { name, stride } => write!(
                f,
                "{name}: innermost stride is {stride}, vector loads need a contiguous inner dim"
            ),
        }
    }
}

impl std::error::Error for AlignmentError {}

/// Checks that `layout` satisfies `align`-byte vector access: aligned base, unit innermost
/// stride, and every outer stride a multiple of `align` bytes.
pub fn check_layout(layout: &TensorLayout<'_>, align: u64) -> Result<(), AlignmentError> {
    if !layout.base_ptr.is_multiple_of(align) {
        return Err(AlignmentError::MisalignedBase {
            name: layout.name.into(),
            base_ptr: layout.base_ptr,
            align,
        });
    }
    let Some((&inner, outer)) = layout.strides.split_last() else {
        return Ok(());
    };
    if inner != 1 {
        return Err(AlignmentError::NonUnitInnerStride {
            name: layout.name.into(),
            stride: inner,
        });
    }
    for (dim, &stride) in outer.iter().enumerate() {
        let stride_bytes = i64::from(stride) * layout.elem_bytes as i64;
        if stride_bytes.rem_euclid(align as i64) != 0 {
            return Err(AlignmentError::MisalignedStride {
                name: layout.name.into(),
                dim,
                stride_bytes,
                align,
            });
        }
    }
    Ok(())
}

/// Checks every argument; returns all violations rather than the first.
pub fn check_layouts(layouts: &[TensorLayout<'_>], align: u64) -> Result<(), Vec<AlignmentError>> {
    let errors: Vec<_> = layouts
        .iter()
        .filter_map(|l| check_layout(l, align).err())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout<'a>(base_ptr: u64, strides: &'a [i32], elem_bytes: usize) -> TensorLayout<'a> {
        TensorLayout {
            name: "z",
            base_ptr,
            strides,
            elem_bytes,
        }
    }

    #[test]
    fn accepts_aligned_contiguous_fp16_latent() {
        // [N, 32] FP16 latent rows: 64-byte row stride, cudaMalloc-style base.
        assert_eq!(
            check_layout(&layout(0x7f00_0000_0000, &[32, 1], 2), 16),
            Ok(())
        );
    }

    #[test]
    fn rejects_misaligned_host_allocation() {
        // A real allocation viewed at a 2-byte offset, as a sliced FP16 view would be.
        let buffer = Box::new([0u16; 64]); // heap allocation
        let aligned = buffer.as_ptr() as u64;
        let base = aligned + (16 - aligned % 16) % 16;
        let misaligned = base + 2;
        assert_eq!(check_layout(&layout(base, &[32, 1], 2), 16), Ok(()));
        assert_eq!(
            check_layout(&layout(misaligned, &[32, 1], 2), 16),
            Err(AlignmentError::MisalignedBase {
                name: "z".into(),
                base_ptr: misaligned,
                align: 16,
            })
        );
    }

    #[test]
    fn rejects_unaligned_row_stride() {
        // 36 FP16 elements = 72-byte rows: base aligned, rows not.
        assert!(matches!(
            check_layout(&layout(0x1000, &[36, 1], 2), 16),
            Err(AlignmentError::MisalignedStride {
                dim: 0,
                stride_bytes: 72,
                ..
            })
        ));
    }

    #[test]
    fn rejects_non_unit_inner_stride() {
        assert!(matches!(
            check_layout(&layout(0x1000, &[1, 32], 2), 16),
            Err(AlignmentError::NonUnitInnerStride { stride: 32, .. })
        ));
    }

    #[test]
    fn reports_every_violation() {
        let ok = [32, 1];
        let bad = [36, 1];
        let layouts = [
            layout(0x1000, &ok, 2),
            layout(0x1002, &ok, 2),
            layout(0x1000, &bad, 2),
        ];
        assert_eq!(check_layouts(&layouts, 16).unwrap_err().len(), 2);
    }
}
