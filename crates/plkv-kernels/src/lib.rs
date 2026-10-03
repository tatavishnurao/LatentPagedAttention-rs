//! Placeholder boundaries for future GPU kernel work.

pub mod alignment;
pub mod block_table;
pub mod splitk_plan;

#[cfg(feature = "gpu-cutile")]
pub mod cutile;

pub mod decode {
    pub fn status() -> &'static str {
        "placeholder: future Rust/cuTile decode kernels"
    }
}

pub mod paged_cache {
    pub fn status() -> &'static str {
        "placeholder: future paged-cache kernels"
    }
}
