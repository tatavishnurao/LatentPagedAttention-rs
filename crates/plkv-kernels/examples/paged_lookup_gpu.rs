use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
use cutile::tile_kernel::DeviceOp;
use plkv_core::paged_lookup_f32;
use plkv_kernels::cutile::paged_lookup::paged_lookup_kernel;
use serde::Deserialize;

fn main() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../fixtures/reference/paged_lookup_f32_seq5_block2_width4.json"
    ))
    .expect("failed to parse paged lookup fixture");
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
    let cpu = paged_lookup_f32(
        &physical_blocks,
        &fixture.block_table,
        fixture.seq_len,
        fixture.block_size,
        fixture.width,
    )
    .expect("Rust CPU paged lookup failed");
    assert_close(&cpu, &expected, 1e-6, "CPU/Python");

    let physical = api::copy_host_vec_to_device(&Arc::new(physical_blocks))
        .sync()
        .expect("failed to upload physical blocks")
        .reshape(&[
            fixture.num_physical_blocks * fixture.block_size,
            fixture.width,
        ])
        .expect("failed to reshape physical blocks");
    let mut block_table_i32: Vec<i32> = fixture
        .block_table
        .iter()
        .map(|value| i32::try_from(*value).expect("block index does not fit i32"))
        .collect();
    block_table_i32.push(0);
    let block_table = api::copy_host_vec_to_device(&Arc::new(block_table_i32))
        .sync()
        .expect("failed to upload block table");
    let output_len =
        fixture.seq_len.div_ceil(fixture.block_size) * fixture.block_size * fixture.width;
    let output = api::zeros::<f32>(&[output_len / fixture.width, fixture.width])
        .sync()
        .expect("failed to allocate output");
    let (result, _, _) = paged_lookup_kernel::paged_lookup(
        output.partition([fixture.block_size, fixture.width]),
        &physical,
        &block_table,
    )
    .sync()
    .expect("cuTile paged lookup failed");
    let gpu_padded = result
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("failed to read GPU output");
    let gpu = gpu_padded[..fixture.seq_len * fixture.width].to_vec();
    let max_abs_error = max_abs_error(&gpu, &expected);
    assert_close(&gpu, &cpu, 1e-6, "GPU/CPU");
    assert_close(&gpu, &expected, 1e-6, "GPU/Python");

    println!("GPU_NAME={}", gpu_name());
    println!(
        "CUDA_TOOLKIT_PATH={}",
        std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "<unset>".to_string())
    );
    println!("CUTILE_VERSION=0.2.0");
    println!("SEQ_LEN={}", fixture.seq_len);
    println!("BLOCK_SIZE={}", fixture.block_size);
    println!("WIDTH={}", fixture.width);
    println!(
        "BLOCK_TABLE={}",
        fixture
            .block_table
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("MAX_ABS_ERROR={max_abs_error}");
    println!("CPU_PYTHON_MATCH=1");
    println!("GPU_CPU_MATCH=1");
    println!("GPU_PYTHON_MATCH=1");
    println!("PAGED_LOOKUP_GPU_OK=1");
}

#[derive(Debug, Deserialize)]
struct Fixture {
    seq_len: usize,
    block_size: usize,
    width: usize,
    num_physical_blocks: usize,
    block_table: Vec<usize>,
    physical_blocks: Vec<Vec<Vec<f32>>>,
    expected_logical_output: Vec<Vec<f32>>,
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let error = max_abs_error(actual, expected);
    assert!(error <= atol, "{label}: max_abs_error={error}, atol={atol}");
}

fn gpu_name() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "<unavailable>".to_string())
}
