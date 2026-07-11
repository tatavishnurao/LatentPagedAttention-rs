use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
use cutile::tile_kernel::DeviceOp;
use plkv_core::paged_gqa_decode_f32;
use plkv_kernels::cutile::paged_gqa_decode::paged_gqa_decode_kernel;
use serde::Deserialize;

const CPU_ATOL: f32 = 1e-5;
const CPU_RTOL: f32 = 1e-5;
const SCORES_ATOL: f32 = 2e-5;
const SCORES_RTOL: f32 = 1e-5;
const PROBABILITIES_ATOL: f32 = 5e-5;
const PROBABILITIES_RTOL: f32 = 1e-5;
const CONTEXT_ATOL: f32 = 5e-5;
const CONTEXT_RTOL: f32 = 1e-5;
const ROW_SUM_ATOL: f32 = 5e-5;

fn main() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../fixtures/reference/paged_gqa_decode_f32.json"
    ))
    .expect("failed to parse paged GQA decode fixture");
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
    println!("ATTENTION_TYPE=DIRECT_PAGED_GQA_DECODE");
    println!("DTYPE={}", fixture.dtype);
    println!("BATCH={}", fixture.batch);
    println!("PIPELINE_STAGES=3");
    println!("ALL_CASES_OK=1");
    println!("PAGED_GQA_DECODE_GPU_OK=1");
}

fn validate_case(fixture: &Fixture, case: &Case) {
    let q = flatten_2d(&case.q);
    let k = flatten_4d(&case.k_physical_gpu_head_major);
    let v = flatten_4d(&case.v_physical_gpu_head_major);
    let expected_scores = flatten_2d(&case.expected_scores);
    let expected_probabilities = flatten_2d(&case.expected_probabilities);
    let expected_context = flatten_2d(&case.expected_context);

    let cpu = paged_gqa_decode_f32(
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
    .expect("Rust CPU paged GQA decode failed");
    assert_all_finite(&cpu.scores, "CPU scores");
    assert_all_finite(&cpu.probabilities, "CPU probabilities");
    assert_all_finite(&cpu.context, "CPU context");
    assert_close(
        &cpu.scores,
        &expected_scores,
        CPU_ATOL,
        CPU_RTOL,
        "CPU/Python scores",
    );
    assert_close(
        &cpu.probabilities,
        &expected_probabilities,
        CPU_ATOL,
        CPU_RTOL,
        "CPU/Python probabilities",
    );
    assert_close(
        &cpu.context,
        &expected_context,
        CPU_ATOL,
        CPU_RTOL,
        "CPU/Python context",
    );
    assert_row_sums(
        &cpu.probabilities,
        fixture.q_heads,
        fixture.seq_len,
        ROW_SUM_ATOL,
    );

    let q_device = api::copy_host_vec_to_device(&Arc::new(q))
        .sync()
        .expect("failed to upload Q")
        .reshape(&[fixture.q_heads, fixture.head_dim])
        .expect("failed to reshape Q");
    let k_device = api::copy_host_vec_to_device(&Arc::new(k))
        .sync()
        .expect("failed to upload physical K")
        .reshape(&[
            fixture.num_physical_blocks * fixture.kv_heads * fixture.block_size,
            fixture.head_dim,
        ])
        .expect("failed to reshape physical K");
    let v_device = api::copy_host_vec_to_device(&Arc::new(v))
        .sync()
        .expect("failed to upload physical V")
        .reshape(&[
            fixture.num_physical_blocks * fixture.kv_heads * fixture.block_size,
            fixture.head_dim,
        ])
        .expect("failed to reshape physical V");
    let block_table = api::copy_host_vec_to_device(&Arc::new(
        fixture
            .block_table
            .iter()
            .map(|value| i32::try_from(*value).expect("block index does not fit i32"))
            .collect::<Vec<_>>(),
    ))
    .sync()
    .expect("failed to upload block table");

    let scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
        .sync()
        .expect("failed to allocate scores output");
    let (scores_partition, _, _, _) = paged_gqa_decode_kernel::paged_gqa_scores(
        scores_out.partition([1, fixture.block_size]),
        &q_device,
        &k_device,
        &block_table,
    )
    .sync()
    .expect("GPU paged score kernel failed");
    let scores_tensor = scores_partition.unpartition();

    let probabilities_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
        .sync()
        .expect("failed to allocate probabilities output");
    let (probabilities_partition, _) = paged_gqa_decode_kernel::stable_softmax_8(
        probabilities_out.partition([1, fixture.seq_len]),
        &scores_tensor,
    )
    .sync()
    .expect("GPU stable softmax kernel failed");
    let probabilities_tensor = probabilities_partition.unpartition();

    let context_out = api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
        .sync()
        .expect("failed to allocate context output");
    let (context_partition, _, _, _) = paged_gqa_decode_kernel::paged_gqa_context(
        context_out.partition([1, fixture.head_dim]),
        &probabilities_tensor,
        &v_device,
        &block_table,
    )
    .sync()
    .expect("GPU paged context kernel failed");

    let gpu_scores = scores_tensor
        .to_host_vec()
        .sync()
        .expect("failed to read scores");
    let gpu_probabilities = probabilities_tensor
        .to_host_vec()
        .sync()
        .expect("failed to read probabilities");
    let gpu_context = context_partition
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("failed to read context");

    assert_all_finite(&gpu_scores, "GPU scores");
    assert_all_finite(&gpu_probabilities, "GPU probabilities");
    assert_all_finite(&gpu_context, "GPU context");
    assert_close(
        &gpu_scores,
        &cpu.scores,
        SCORES_ATOL,
        SCORES_RTOL,
        "GPU/CPU scores",
    );
    assert_close(
        &gpu_probabilities,
        &cpu.probabilities,
        PROBABILITIES_ATOL,
        PROBABILITIES_RTOL,
        "GPU/CPU probabilities",
    );
    assert_close(
        &gpu_context,
        &cpu.context,
        CONTEXT_ATOL,
        CONTEXT_RTOL,
        "GPU/CPU context",
    );
    assert_close(
        &gpu_scores,
        &expected_scores,
        SCORES_ATOL,
        SCORES_RTOL,
        "GPU/Python scores",
    );
    assert_close(
        &gpu_probabilities,
        &expected_probabilities,
        PROBABILITIES_ATOL,
        PROBABILITIES_RTOL,
        "GPU/Python probabilities",
    );
    assert_close(
        &gpu_context,
        &expected_context,
        CONTEXT_ATOL,
        CONTEXT_RTOL,
        "GPU/Python context",
    );
    let row_sum_error =
        max_probability_row_sum_error(&gpu_probabilities, fixture.q_heads, fixture.seq_len);
    assert!(
        row_sum_error <= ROW_SUM_ATOL,
        "GPU probability row sum error {row_sum_error} exceeded {ROW_SUM_ATOL}"
    );

    let identity_block_table = api::copy_host_vec_to_device(&Arc::new(vec![0i32, 1, 2, 3]))
        .sync()
        .expect("failed to upload identity block table");
    let identity_scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
        .sync()
        .expect("failed to allocate identity scores output");
    let (identity_scores_partition, _, _, _) = paged_gqa_decode_kernel::paged_gqa_scores(
        identity_scores_out.partition([1, fixture.block_size]),
        &q_device,
        &k_device,
        &identity_block_table,
    )
    .sync()
    .expect("identity GPU paged score kernel failed");
    let identity_scores = identity_scores_partition
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("failed to read identity scores");
    let identity_changed = max_abs_error(&identity_scores, &gpu_scores) > SCORES_ATOL;
    assert!(
        identity_changed,
        "runtime block table had no effect on paged scores"
    );

    println!("CASE_NAME={}", case.name);
    println!("Q_HEADS={}", fixture.q_heads);
    println!("KV_HEADS={}", fixture.kv_heads);
    println!("GROUP_SIZE={}", fixture.group_size);
    println!("SEQ_LEN={}", fixture.seq_len);
    println!("HEAD_DIM={}", fixture.head_dim);
    println!("BLOCK_SIZE={}", fixture.block_size);
    println!("NUM_LOGICAL_BLOCKS={}", fixture.num_logical_blocks);
    println!("NUM_PHYSICAL_BLOCKS={}", fixture.num_physical_blocks);
    println!(
        "BLOCK_TABLE={}",
        fixture
            .block_table
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "Q_TO_KV={}",
        fixture
            .q_to_kv
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("PIPELINE=PAGED_SCORES_GPU__SOFTMAX_GPU__PAGED_CONTEXT_GPU");
    println!(
        "SCORES_MAX_ABS_ERROR={}",
        max_abs_error(&gpu_scores, &expected_scores)
    );
    println!(
        "PROBABILITIES_MAX_ABS_ERROR={}",
        max_abs_error(&gpu_probabilities, &expected_probabilities)
    );
    println!(
        "CONTEXT_MAX_ABS_ERROR={}",
        max_abs_error(&gpu_context, &expected_context)
    );
    println!("MAX_PROBABILITY_ROW_SUM_ERROR={row_sum_error}");
    println!("SCORES_FINITE=1");
    println!("PROBABILITIES_FINITE=1");
    println!("CONTEXT_FINITE=1");
    println!("PROBABILITY_ROWS_SUM_TO_ONE=1");
    println!("CPU_PYTHON_SCORES_MATCH=1");
    println!("CPU_PYTHON_PROBABILITIES_MATCH=1");
    println!("CPU_PYTHON_CONTEXT_MATCH=1");
    println!("GPU_CPU_SCORES_MATCH=1");
    println!("GPU_CPU_PROBABILITIES_MATCH=1");
    println!("GPU_CPU_CONTEXT_MATCH=1");
    println!("GPU_PYTHON_SCORES_MATCH=1");
    println!("GPU_PYTHON_PROBABILITIES_MATCH=1");
    println!("GPU_PYTHON_CONTEXT_MATCH=1");
    println!("NON_IDENTITY_MAPPING_EFFECT_CONFIRMED=1");
    println!("CASE_OK=1");
}

#[derive(Debug, Deserialize)]
struct Fixture {
    dtype: String,
    batch: usize,
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
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    q: Vec<Vec<f32>>,
    k_physical_gpu_head_major: Vec<Vec<Vec<Vec<f32>>>>,
    v_physical_gpu_head_major: Vec<Vec<Vec<Vec<f32>>>>,
    expected_scores: Vec<Vec<f32>>,
    expected_probabilities: Vec<Vec<f32>>,
    expected_context: Vec<Vec<f32>>,
}

fn flatten_2d(values: &[Vec<f32>]) -> Vec<f32> {
    values.iter().flatten().copied().collect()
}

fn flatten_4d(values: &[Vec<Vec<Vec<f32>>>]) -> Vec<f32> {
    values
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .copied()
        .collect()
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: index={index}, actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn assert_all_finite(values: &[f32], label: &str) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label}: encountered non-finite values"
    );
}

fn assert_row_sums(probabilities: &[f32], q_heads: usize, seq_len: usize, atol: f32) {
    let error = max_probability_row_sum_error(probabilities, q_heads, seq_len);
    assert!(
        error <= atol,
        "probability row sum error {error} exceeded {atol}"
    );
}

fn max_probability_row_sum_error(probabilities: &[f32], q_heads: usize, seq_len: usize) -> f32 {
    let mut max_error = 0.0f32;
    for q_head in 0..q_heads {
        let start = q_head * seq_len;
        let row_sum: f32 = probabilities[start..start + seq_len].iter().sum();
        max_error = max_error.max((row_sum - 1.0).abs());
    }
    max_error
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
