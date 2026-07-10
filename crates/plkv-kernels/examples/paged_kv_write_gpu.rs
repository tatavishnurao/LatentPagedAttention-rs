use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
use cutile::tile_kernel::DeviceOp;
use plkv_core::paged_kv_write_f32;
use plkv_kernels::cutile::paged_kv_write::paged_kv_write_kernel;
use serde::Deserialize;

const ATOL: f32 = 1e-6;

fn main() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../fixtures/reference/paged_kv_write_f32.json"
    ))
    .expect("failed to parse paged KV-write fixture");
    let gpu_name = gpu_name();

    for case in &fixture.cases {
        validate_case(&fixture, case);
    }

    println!("GPU_NAME={gpu_name}");
    println!(
        "CUDA_TOOLKIT_PATH={}",
        std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "<unset>".to_string())
    );
    println!("CUTILE_VERSION=0.2.0");
    println!("NUM_PHYSICAL_BLOCKS={}", fixture.num_physical_blocks);
    println!("BLOCK_SIZE={}", fixture.block_size);
    println!("KV_HEADS={}", fixture.kv_heads);
    println!("HEAD_DIM={}", fixture.head_dim);
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
    println!("ALL_CASES_OK=1");
    println!("PAGED_KV_WRITE_GPU_OK=1");
}

fn validate_case(fixture: &Fixture, case: &Case) {
    let initial_k = flatten(&case.initial_k_cache);
    let initial_v = flatten(&case.initial_v_cache);
    let expected_k = flatten(&case.expected_k_cache);
    let expected_v = flatten(&case.expected_v_cache);
    let mut cpu_k = initial_k.clone();
    let mut cpu_v = initial_v.clone();
    let location = paged_kv_write_f32(
        &mut cpu_k,
        &mut cpu_v,
        &fixture.block_table,
        case.token_position,
        fixture.block_size,
        fixture.width,
        &case.new_k,
        &case.new_v,
    )
    .expect("Rust CPU paged KV write failed");
    assert_eq!(cpu_k, expected_k, "{} CPU K mismatch", case.name);
    assert_eq!(cpu_v, expected_v, "{} CPU V mismatch", case.name);
    assert_eq!(location.logical_block, case.logical_block);
    assert_eq!(location.physical_block, case.physical_block);
    assert_eq!(location.block_offset, case.block_offset);

    let mut padded_table = fixture.gpu_padded_block_table.clone();
    let table = api::copy_host_vec_to_device(&Arc::new(padded_table.clone()))
        .sync()
        .expect("failed to upload padded block table");
    padded_table.clear();
    let token_position = api::copy_host_vec_to_device(&Arc::new(vec![
        i32::try_from(case.token_position).expect("token position does not fit i32"),
    ]))
    .sync()
    .expect("failed to upload token position");
    let new_k = api::copy_host_vec_to_device(&Arc::new(case.new_k.clone()))
        .sync()
        .expect("failed to upload new K");
    let new_v = api::copy_host_vec_to_device(&Arc::new(case.new_v.clone()))
        .sync()
        .expect("failed to upload new V");
    let k_cache = api::copy_host_vec_to_device(&Arc::new(initial_k))
        .sync()
        .expect("failed to upload K cache")
        .reshape(&[
            fixture.num_physical_blocks * fixture.block_size,
            fixture.width,
        ])
        .expect("failed to reshape K cache");
    let v_cache = api::copy_host_vec_to_device(&Arc::new(initial_v))
        .sync()
        .expect("failed to upload V cache")
        .reshape(&[
            fixture.num_physical_blocks * fixture.block_size,
            fixture.width,
        ])
        .expect("failed to reshape V cache");

    let (k_result, v_result, _, _, _, _) = paged_kv_write_kernel::paged_kv_write(
        k_cache.partition([fixture.block_size, fixture.width]),
        v_cache.partition([fixture.block_size, fixture.width]),
        &table,
        &token_position,
        &new_k,
        &new_v,
    )
    .sync()
    .expect("cuTile paged KV write failed");
    let gpu_k = k_result.unpartition().to_host_vec().sync().expect("read K");
    let gpu_v = v_result.unpartition().to_host_vec().sync().expect("read V");

    let k_error = max_abs_error(&gpu_k, &expected_k);
    let v_error = max_abs_error(&gpu_v, &expected_v);
    let k_changed = changed_elements(&gpu_k, &flatten(&case.initial_k_cache));
    let v_changed = changed_elements(&gpu_v, &flatten(&case.initial_v_cache));
    assert!(k_error <= ATOL, "{} GPU K error {k_error}", case.name);
    assert!(v_error <= ATOL, "{} GPU V error {v_error}", case.name);
    assert_eq!(k_changed, fixture.width);
    assert_eq!(v_changed, fixture.width);
    assert_unchanged_except_target(
        &gpu_k,
        &flatten(&case.initial_k_cache),
        case.physical_block,
        case.block_offset,
        fixture.block_size,
        fixture.width,
    );
    assert_unchanged_except_target(
        &gpu_v,
        &flatten(&case.initial_v_cache),
        case.physical_block,
        case.block_offset,
        fixture.block_size,
        fixture.width,
    );

    let roundtrip = run_lookup_roundtrip(fixture, &gpu_k, case);
    assert_eq!(
        roundtrip, case.new_k,
        "{} GPU lookup round trip mismatch",
        case.name
    );

    println!("CASE_NAME={}", case.name);
    println!("TOKEN_POSITION={}", case.token_position);
    println!("LOGICAL_BLOCK={}", case.logical_block);
    println!("PHYSICAL_BLOCK={}", case.physical_block);
    println!("BLOCK_OFFSET={}", case.block_offset);
    println!("K_MAX_ABS_ERROR={k_error}");
    println!("V_MAX_ABS_ERROR={v_error}");
    println!("K_CHANGED_ELEMENTS={k_changed}");
    println!("V_CHANGED_ELEMENTS={v_changed}");
    println!("K_UNCHANGED_REGION_OK=1");
    println!("V_UNCHANGED_REGION_OK=1");
    println!("CPU_PYTHON_MATCH=1");
    println!("GPU_CPU_MATCH=1");
    println!("GPU_PYTHON_MATCH=1");
    println!("KV_WRITE_LOOKUP_ROUNDTRIP_OK=1");
    println!("CASE_OK=1");
}

fn run_lookup_roundtrip(fixture: &Fixture, gpu_k: &[f32], case: &Case) -> Vec<f32> {
    let cache = api::copy_host_vec_to_device(&Arc::new(gpu_k.to_vec()))
        .sync()
        .expect("failed to upload written K cache")
        .reshape(&[
            fixture.num_physical_blocks * fixture.block_size,
            fixture.width,
        ])
        .expect("failed to reshape written K cache");
    let table = api::copy_host_vec_to_device(&Arc::new(fixture.gpu_padded_block_table.clone()))
        .sync()
        .expect("failed to upload round-trip table");
    let output = api::zeros::<f32>(&[
        fixture.block_table.len() * fixture.block_size,
        fixture.width,
    ])
    .sync()
    .expect("failed to allocate round-trip output");
    let (result, _, _) = paged_kv_write_kernel::paged_lookup_width8(
        output.partition([fixture.block_size, fixture.width]),
        &cache,
        &table,
    )
    .sync()
    .expect("GPU lookup round trip failed");
    let logical = result
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("read round trip");
    let start = case.token_position * fixture.width;
    logical[start..start + fixture.width].to_vec()
}

fn flatten(cache: &[Vec<Vec<f32>>]) -> Vec<f32> {
    cache.iter().flatten().flatten().copied().collect()
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

fn changed_elements(actual: &[f32], initial: &[f32]) -> usize {
    actual
        .iter()
        .zip(initial)
        .filter(|(actual, initial)| *actual != *initial)
        .count()
}

fn assert_unchanged_except_target(
    actual: &[f32],
    initial: &[f32],
    physical_block: usize,
    block_offset: usize,
    block_size: usize,
    width: usize,
) {
    let target_start = (physical_block * block_size + block_offset) * width;
    for (index, (actual, initial)) in actual.iter().zip(initial).enumerate() {
        if index < target_start || index >= target_start + width {
            assert_eq!(actual, initial, "cache corruption at element {index}");
        }
    }
}

#[derive(Debug, Deserialize)]
struct Fixture {
    num_physical_blocks: usize,
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    width: usize,
    block_table: Vec<usize>,
    gpu_padded_block_table: Vec<i32>,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
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
