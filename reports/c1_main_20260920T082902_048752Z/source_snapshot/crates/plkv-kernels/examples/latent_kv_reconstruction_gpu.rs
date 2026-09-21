use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
use cutile::tile_kernel::DeviceOp;
use plkv_core::reconstruct_latent_kv_f32;
use plkv_kernels::cutile::latent_kv_reconstruction::latent_kv_reconstruction_kernel;
use serde::Deserialize;

const CPU_ATOL: f32 = 1e-5;
const CPU_RTOL: f32 = 1e-5;
const GPU_ATOL: f32 = 2e-5;
const GPU_RTOL: f32 = 1e-5;

fn main() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../fixtures/reference/latent_kv_reconstruction_f32.json"
    ))
    .expect("failed to parse latent-KV reconstruction fixture");
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
    println!("OPERATION=LATENT_KV_RECONSTRUCTION");
    println!("IMPLEMENTATION=FUSED");
    println!("DTYPE={}", fixture.dtype);
    println!("BATCH={}", fixture.batch);
    println!("SEQ_LEN={}", fixture.seq_len);
    println!("LATENT_DIM={}", fixture.latent_dim);
    println!("PROJECTION_WIDTH={}", fixture.projection_width);
    println!(
        "THEORETICAL_CACHE_COMPRESSION_RATIO={}",
        fixture.theoretical_cache_compression_ratio
    );
    println!("ALL_CASES_OK=1");
    println!("LATENT_KV_RECONSTRUCTION_GPU_OK=1");
}

fn validate_case(fixture: &Fixture, case: &Case) {
    let latent = flatten_2d(&case.latent_cache);
    let k_projection = flatten_2d(&case.k_projection);
    let v_projection = flatten_2d(&case.v_projection);
    let expected_k = flatten_3d(&case.expected_k_token_major);
    let expected_v = flatten_3d(&case.expected_v_token_major);
    let expected_k_head = flatten_3d(&case.expected_k_head_major);
    let expected_v_head = flatten_3d(&case.expected_v_head_major);

    let cpu = reconstruct_latent_kv_f32(
        &latent,
        &k_projection,
        &v_projection,
        fixture.seq_len,
        fixture.latent_dim,
        fixture.kv_heads,
        fixture.head_dim,
    )
    .expect("Rust CPU latent-KV reconstruction failed");
    assert_all_finite(&cpu.k_token_major, "CPU K");
    assert_all_finite(&cpu.v_token_major, "CPU V");
    assert_close(
        &cpu.k_token_major,
        &expected_k,
        CPU_ATOL,
        CPU_RTOL,
        "CPU/Python K",
    );
    assert_close(
        &cpu.v_token_major,
        &expected_v,
        CPU_ATOL,
        CPU_RTOL,
        "CPU/Python V",
    );

    let latent_device = api::copy_host_vec_to_device(&Arc::new(latent.clone()))
        .sync()
        .expect("failed to upload latent cache")
        .reshape(&[fixture.seq_len, fixture.latent_dim])
        .expect("failed to reshape latent cache");
    let k_projection_device = api::copy_host_vec_to_device(&Arc::new(k_projection.clone()))
        .sync()
        .expect("failed to upload K projection")
        .reshape(&[fixture.latent_dim, fixture.projection_width])
        .expect("failed to reshape K projection");
    let v_projection_device = api::copy_host_vec_to_device(&Arc::new(v_projection.clone()))
        .sync()
        .expect("failed to upload V projection")
        .reshape(&[fixture.latent_dim, fixture.projection_width])
        .expect("failed to reshape V projection");
    let k_out = api::zeros::<f32>(&[fixture.seq_len, fixture.projection_width])
        .sync()
        .expect("failed to allocate K output");
    let v_out = api::zeros::<f32>(&[fixture.seq_len, fixture.projection_width])
        .sync()
        .expect("failed to allocate V output");

    let (k_partition, v_partition, _, _, _) =
        latent_kv_reconstruction_kernel::reconstruct_latent_kv(
            k_out.partition([1, fixture.projection_width]),
            v_out.partition([1, fixture.projection_width]),
            &latent_device,
            &k_projection_device,
            &v_projection_device,
        )
        .sync()
        .expect("GPU latent-KV reconstruction failed");

    let gpu_k = k_partition
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("failed to read K output");
    let gpu_v = v_partition
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("failed to read V output");
    assert_all_finite(&gpu_k, "GPU K");
    assert_all_finite(&gpu_v, "GPU V");
    assert_close(&gpu_k, &cpu.k_token_major, GPU_ATOL, GPU_RTOL, "GPU/CPU K");
    assert_close(&gpu_v, &cpu.v_token_major, GPU_ATOL, GPU_RTOL, "GPU/CPU V");
    assert_close(&gpu_k, &expected_k, GPU_ATOL, GPU_RTOL, "GPU/Python K");
    assert_close(&gpu_v, &expected_v, GPU_ATOL, GPU_RTOL, "GPU/Python V");

    let gpu_k_head =
        token_major_to_head_major(&gpu_k, fixture.seq_len, fixture.kv_heads, fixture.head_dim);
    let gpu_v_head =
        token_major_to_head_major(&gpu_v, fixture.seq_len, fixture.kv_heads, fixture.head_dim);
    assert_close(
        &gpu_k_head,
        &expected_k_head,
        GPU_ATOL,
        GPU_RTOL,
        "GPU/Python head-major K",
    );
    assert_close(
        &gpu_v_head,
        &expected_v_head,
        GPU_ATOL,
        GPU_RTOL,
        "GPU/Python head-major V",
    );

    let control_v_projection_device = api::copy_host_vec_to_device(&Arc::new(k_projection.clone()))
        .sync()
        .expect("failed to upload control V projection")
        .reshape(&[fixture.latent_dim, fixture.projection_width])
        .expect("failed to reshape control V projection");
    let control_k_out = api::zeros::<f32>(&[fixture.seq_len, fixture.projection_width])
        .sync()
        .expect("failed to allocate control K output");
    let control_v_out = api::zeros::<f32>(&[fixture.seq_len, fixture.projection_width])
        .sync()
        .expect("failed to allocate control V output");
    let (_, control_v_partition, _, _, _) = latent_kv_reconstruction_kernel::reconstruct_latent_kv(
        control_k_out.partition([1, fixture.projection_width]),
        control_v_out.partition([1, fixture.projection_width]),
        &latent_device,
        &k_projection_device,
        &control_v_projection_device,
    )
    .sync()
    .expect("GPU latent-KV projection control failed");
    let control_v = control_v_partition
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("failed to read projection control V");
    assert!(
        max_abs_error(&control_v, &expected_v) > GPU_ATOL,
        "replacing V projection with K projection did not change V output"
    );
    assert!(
        rows_are_token_dependent(&gpu_k, fixture.seq_len, fixture.projection_width)
            || rows_are_token_dependent(&gpu_v, fixture.seq_len, fixture.projection_width),
        "all reconstructed token rows were identical"
    );
    let wrong_orientation =
        wrong_head_major_orientation(&gpu_k, fixture.seq_len, fixture.kv_heads, fixture.head_dim);
    assert!(
        max_abs_error(&wrong_orientation, &expected_k_head) > GPU_ATOL,
        "incorrect projection orientation matched expected head-major K"
    );

    println!("CASE_NAME={}", case.name);
    println!("SEQ_LEN={}", fixture.seq_len);
    println!("LATENT_DIM={}", fixture.latent_dim);
    println!("KV_HEADS={}", fixture.kv_heads);
    println!("HEAD_DIM={}", fixture.head_dim);
    println!("PROJECTION_WIDTH={}", fixture.projection_width);
    println!("DTYPE={}", fixture.dtype);
    println!("K_MAX_ABS_ERROR={}", max_abs_error(&gpu_k, &expected_k));
    println!("V_MAX_ABS_ERROR={}", max_abs_error(&gpu_v, &expected_v));
    println!(
        "K_HEAD_MAJOR_MAX_ABS_ERROR={}",
        max_abs_error(&gpu_k_head, &expected_k_head)
    );
    println!(
        "V_HEAD_MAJOR_MAX_ABS_ERROR={}",
        max_abs_error(&gpu_v_head, &expected_v_head)
    );
    println!("K_FINITE=1");
    println!("V_FINITE=1");
    println!("CPU_PYTHON_K_MATCH=1");
    println!("CPU_PYTHON_V_MATCH=1");
    println!("GPU_CPU_K_MATCH=1");
    println!("GPU_CPU_V_MATCH=1");
    println!("GPU_PYTHON_K_MATCH=1");
    println!("GPU_PYTHON_V_MATCH=1");
    println!("DISTINCT_KV_PROJECTIONS_CONFIRMED=1");
    println!("TOKEN_DEPENDENT_RECONSTRUCTION_CONFIRMED=1");
    println!("PROJECTION_ORIENTATION_CONFIRMED=1");
    println!("CASE_OK=1");
}

#[derive(Debug, Deserialize)]
struct Fixture {
    dtype: String,
    batch: usize,
    seq_len: usize,
    latent_dim: usize,
    kv_heads: usize,
    head_dim: usize,
    projection_width: usize,
    theoretical_cache_compression_ratio: f32,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    latent_cache: Vec<Vec<f32>>,
    k_projection: Vec<Vec<f32>>,
    v_projection: Vec<Vec<f32>>,
    expected_k_token_major: Vec<Vec<Vec<f32>>>,
    expected_v_token_major: Vec<Vec<Vec<f32>>>,
    expected_k_head_major: Vec<Vec<Vec<f32>>>,
    expected_v_head_major: Vec<Vec<Vec<f32>>>,
}

fn flatten_2d(values: &[Vec<f32>]) -> Vec<f32> {
    values.iter().flatten().copied().collect()
}

fn flatten_3d(values: &[Vec<Vec<f32>>]) -> Vec<f32> {
    values.iter().flatten().flatten().copied().collect()
}

fn assert_all_finite(values: &[f32], label: &str) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contained non-finite values"
    );
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
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

fn wrong_head_major_orientation(
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
                let dst = (kv_head * seq_len + token) * head_dim + (head_dim - 1 - dim);
                out[dst] = token_major[src];
            }
        }
    }
    out
}

fn rows_are_token_dependent(values: &[f32], seq_len: usize, width: usize) -> bool {
    (1..seq_len).any(|token| values[token * width..(token + 1) * width] != values[0..width])
}

fn gpu_name() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| {
            value
                .lines()
                .next()
                .unwrap_or("<unknown>")
                .trim()
                .to_string()
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "<unknown>".to_string())
}
