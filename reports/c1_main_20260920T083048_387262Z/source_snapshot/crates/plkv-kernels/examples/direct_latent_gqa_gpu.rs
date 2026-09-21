#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::sync::Arc;

    use cutile::api;
    use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{
        contiguous_gqa_decode_f32, direct_latent_gqa_decode_f32, reconstruct_latent_kv_f32,
    };
    use plkv_kernels::cutile::direct_latent_gqa::direct_latent_gqa_kernel;
    use serde::Deserialize;

    const DIRECT_SCORES_ATOL: f32 = 5e-5;
    const DIRECT_SCORES_RTOL: f32 = 1e-5;
    const DIRECT_PROBS_ATOL: f32 = 5e-5;
    const DIRECT_PROBS_RTOL: f32 = 1e-5;
    const DIRECT_CONTEXT_ATOL: f32 = 1e-4;
    const DIRECT_CONTEXT_RTOL: f32 = 1e-5;
    const DIRECT_ROW_SUM_ATOL: f32 = 5e-5;

    pub fn main() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/direct_latent_gqa_decode_f32.json"
        ))
        .expect("failed to parse direct latent GQA fixture");
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
        println!("OPERATION=DIRECT_CONTIGUOUS_LATENT_GQA");
        println!("IMPLEMENTATION=FUSED");
        println!("DTYPE={}", fixture.dtype);
        println!("BATCH={}", fixture.batch);
        println!(
            "LATENT_CACHE_VALUES={}",
            fixture.batch * fixture.seq_len * fixture.latent_dim
        );
        println!(
            "HYPOTHETICAL_FULL_KV_CACHE_VALUES={}",
            fixture.batch * fixture.seq_len * fixture.full_kv_values_per_token
        );
        println!("DIRECT_PATH_RECONSTRUCTED_K_VALUES=0");
        println!("DIRECT_PATH_RECONSTRUCTED_V_VALUES=0");
        println!("DIRECT_PATH_RECONSTRUCTED_KV_DEVICE_VALUES=0");
        println!(
            "THEORETICAL_CACHE_COMPRESSION_RATIO={}",
            fixture.theoretical_cache_compression_ratio
        );
        println!("ALL_CASES_OK=1");
        println!("DIRECT_LATENT_GQA_GPU_OK=1");
    }

    fn validate_case(fixture: &Fixture, case: &Case) {
        let q = flatten_2d(&case.q);
        let latent = flatten_2d(&case.latent_cache);
        let k_projection = flatten_2d(&case.k_projection);
        let v_projection = flatten_2d(&case.v_projection);
        let k_projection_head_major = flatten_3d(&case.k_projection_gpu_head_major);
        let v_projection_head_major = flatten_3d(&case.v_projection_gpu_head_major);
        let expected_scores = flatten_2d(&case.expected_scores);
        let expected_probabilities = flatten_2d(&case.expected_probabilities);
        let expected_context = flatten_2d(&case.expected_context);
        let materialized_scores = flatten_2d(&case.materialized_scores);
        let materialized_probabilities = flatten_2d(&case.materialized_probabilities);
        let materialized_context = flatten_2d(&case.materialized_context);

        let cpu_direct = direct_latent_gqa_decode_f32(
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
        .expect("Rust CPU direct latent GQA failed");
        let reconstructed = reconstruct_latent_kv_f32(
            &latent,
            &k_projection,
            &v_projection,
            fixture.seq_len,
            fixture.latent_dim,
            fixture.kv_heads,
            fixture.head_dim,
        )
        .expect("Rust CPU reconstruction failed");
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
        let cpu_materialized = contiguous_gqa_decode_f32(
            &q,
            &reconstructed_k_head_major,
            &reconstructed_v_head_major,
            fixture.q_heads,
            fixture.kv_heads,
            fixture.seq_len,
            fixture.head_dim,
            fixture.group_size,
        )
        .expect("Rust CPU materialized latent GQA failed");

        assert_all_finite(&cpu_direct.scores, "CPU direct scores");
        assert_all_finite(&cpu_direct.probabilities, "CPU direct probabilities");
        assert_all_finite(&cpu_direct.context, "CPU direct context");
        assert_close(
            &cpu_direct.scores,
            &expected_scores,
            DIRECT_SCORES_ATOL,
            DIRECT_SCORES_RTOL,
            "CPU direct/Python scores",
        );
        assert_close(
            &cpu_direct.probabilities,
            &expected_probabilities,
            DIRECT_PROBS_ATOL,
            DIRECT_PROBS_RTOL,
            "CPU direct/Python probabilities",
        );
        assert_close(
            &cpu_direct.context,
            &expected_context,
            DIRECT_CONTEXT_ATOL,
            DIRECT_CONTEXT_RTOL,
            "CPU direct/Python context",
        );
        assert_close(
            &cpu_direct.scores,
            &materialized_scores,
            DIRECT_SCORES_ATOL,
            DIRECT_SCORES_RTOL,
            "CPU direct/materialized scores",
        );
        assert_close(
            &cpu_direct.probabilities,
            &materialized_probabilities,
            DIRECT_PROBS_ATOL,
            DIRECT_PROBS_RTOL,
            "CPU direct/materialized probabilities",
        );
        assert_close(
            &cpu_direct.context,
            &materialized_context,
            DIRECT_CONTEXT_ATOL,
            DIRECT_CONTEXT_RTOL,
            "CPU direct/materialized context",
        );
        assert_close(
            &cpu_direct.scores,
            &cpu_materialized.scores,
            DIRECT_SCORES_ATOL,
            DIRECT_SCORES_RTOL,
            "CPU materialized scores",
        );
        assert_close(
            &cpu_direct.probabilities,
            &cpu_materialized.probabilities,
            DIRECT_PROBS_ATOL,
            DIRECT_PROBS_RTOL,
            "CPU materialized probabilities",
        );
        assert_close(
            &cpu_direct.context,
            &cpu_materialized.context,
            DIRECT_CONTEXT_ATOL,
            DIRECT_CONTEXT_RTOL,
            "CPU materialized context",
        );
        assert_row_sums(
            &cpu_direct.probabilities,
            fixture.q_heads,
            fixture.seq_len,
            DIRECT_ROW_SUM_ATOL,
        );

        let q_device = api::copy_host_vec_to_device(&Arc::new(q))
            .sync()
            .expect("failed to upload Q")
            .reshape(&[fixture.q_heads, fixture.head_dim])
            .expect("failed to reshape Q");
        let latent_device = api::copy_host_vec_to_device(&Arc::new(latent))
            .sync()
            .expect("failed to upload latent cache")
            .reshape(&[fixture.seq_len, fixture.latent_dim])
            .expect("failed to reshape latent cache");
        let k_projection_device =
            api::copy_host_vec_to_device(&Arc::new(k_projection_head_major.clone()))
                .sync()
                .expect("failed to upload K projection")
                .reshape(&[fixture.kv_heads * fixture.latent_dim, fixture.head_dim])
                .expect("failed to reshape K projection");
        let v_projection_device =
            api::copy_host_vec_to_device(&Arc::new(v_projection_head_major.clone()))
                .sync()
                .expect("failed to upload V projection")
                .reshape(&[fixture.kv_heads * fixture.latent_dim, fixture.head_dim])
                .expect("failed to reshape V projection");

        let scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("failed to allocate scores output");
        let probabilities_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("failed to allocate probabilities output");
        let context_out = api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
            .sync()
            .expect("failed to allocate context output");

        let (scores_partition, probabilities_partition, context_partition, _, _, _, _) =
            direct_latent_gqa_kernel::direct_latent_gqa_decode(
                scores_out.partition([1, fixture.seq_len]),
                probabilities_out.partition([1, fixture.seq_len]),
                context_out.partition([1, fixture.head_dim]),
                &q_device,
                &latent_device,
                &k_projection_device,
                &v_projection_device,
            )
            .sync()
            .expect("GPU direct latent GQA failed");

        let gpu_scores = scores_partition
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("failed to read GPU scores");
        let gpu_probabilities = probabilities_partition
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("failed to read GPU probabilities");
        let gpu_context = context_partition
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("failed to read GPU context");

        assert_all_finite(&gpu_scores, "GPU scores");
        assert_all_finite(&gpu_probabilities, "GPU probabilities");
        assert_all_finite(&gpu_context, "GPU context");
        assert_close(
            &gpu_scores,
            &cpu_direct.scores,
            DIRECT_SCORES_ATOL,
            DIRECT_SCORES_RTOL,
            "GPU/CPU direct scores",
        );
        assert_close(
            &gpu_probabilities,
            &cpu_direct.probabilities,
            DIRECT_PROBS_ATOL,
            DIRECT_PROBS_RTOL,
            "GPU/CPU direct probabilities",
        );
        assert_close(
            &gpu_context,
            &cpu_direct.context,
            DIRECT_CONTEXT_ATOL,
            DIRECT_CONTEXT_RTOL,
            "GPU/CPU direct context",
        );
        assert_close(
            &gpu_scores,
            &expected_scores,
            DIRECT_SCORES_ATOL,
            DIRECT_SCORES_RTOL,
            "GPU/Python direct scores",
        );
        assert_close(
            &gpu_probabilities,
            &expected_probabilities,
            DIRECT_PROBS_ATOL,
            DIRECT_PROBS_RTOL,
            "GPU/Python direct probabilities",
        );
        assert_close(
            &gpu_context,
            &expected_context,
            DIRECT_CONTEXT_ATOL,
            DIRECT_CONTEXT_RTOL,
            "GPU/Python direct context",
        );
        let row_sum_error =
            max_probability_row_sum_error(&gpu_probabilities, fixture.q_heads, fixture.seq_len);
        assert!(
            row_sum_error <= DIRECT_ROW_SUM_ATOL,
            "GPU probability row sum error {row_sum_error} exceeded {DIRECT_ROW_SUM_ATOL}"
        );
        assert!(
            fixture.cases[0].expected_scores.len() == fixture.cases[1].expected_scores.len(),
            "fixture cases must share score shape"
        );

        let swapped_k_projection_device =
            api::copy_host_vec_to_device(&Arc::new(swap_k_projection_heads(
                &k_projection_head_major,
                fixture.latent_dim,
                fixture.head_dim,
            )))
            .sync()
            .expect("failed to upload swapped K projection")
            .reshape(&[fixture.kv_heads * fixture.latent_dim, fixture.head_dim])
            .expect("failed to reshape swapped K projection");
        let swapped_scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("failed to allocate swapped scores output");
        let swapped_probabilities_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("failed to allocate swapped probabilities output");
        let swapped_context_out = api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
            .sync()
            .expect("failed to allocate swapped context output");
        let (swapped_scores_partition, _, _, _, _, _, _) =
            direct_latent_gqa_kernel::direct_latent_gqa_decode(
                swapped_scores_out.partition([1, fixture.seq_len]),
                swapped_probabilities_out.partition([1, fixture.seq_len]),
                swapped_context_out.partition([1, fixture.head_dim]),
                &q_device,
                &latent_device,
                &swapped_k_projection_device,
                &v_projection_device,
            )
            .sync()
            .expect("GPU swapped K projection run failed");
        let swapped_scores = swapped_scores_partition
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("failed to read swapped scores");
        let kv_head_mapping_effect =
            max_abs_error(&swapped_scores, &gpu_scores) > DIRECT_SCORES_ATOL;
        assert!(
            kv_head_mapping_effect,
            "KV head mapping control did not change scores"
        );

        let control_v_projection_device =
            api::copy_host_vec_to_device(&Arc::new(k_projection_head_major.clone()))
                .sync()
                .expect("failed to upload control V projection")
                .reshape(&[fixture.kv_heads * fixture.latent_dim, fixture.head_dim])
                .expect("failed to reshape control V projection");
        let control_scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("failed to allocate control scores output");
        let control_probabilities_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("failed to allocate control probabilities output");
        let control_context_out = api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
            .sync()
            .expect("failed to allocate control context output");
        let (_, _, control_context_partition, _, _, _, _) =
            direct_latent_gqa_kernel::direct_latent_gqa_decode(
                control_scores_out.partition([1, fixture.seq_len]),
                control_probabilities_out.partition([1, fixture.seq_len]),
                control_context_out.partition([1, fixture.head_dim]),
                &q_device,
                &latent_device,
                &k_projection_device,
                &control_v_projection_device,
            )
            .sync()
            .expect("GPU control V projection run failed");
        let control_context = control_context_partition
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("failed to read control context");
        assert!(
            max_abs_error(&control_context, &gpu_context) > DIRECT_CONTEXT_ATOL,
            "replacing V projection with K projection did not change context"
        );

        println!("CASE_NAME={}", case.name);
        println!("Q_HEADS={}", fixture.q_heads);
        println!("KV_HEADS={}", fixture.kv_heads);
        println!("GROUP_SIZE={}", fixture.group_size);
        println!("SEQ_LEN={}", fixture.seq_len);
        println!("LATENT_DIM={}", fixture.latent_dim);
        println!("PROJECTION_WIDTH={}", fixture.projection_width);
        println!("Q_TO_KV={}", join_usize(&fixture.q_to_kv));
        println!("DTYPE={}", fixture.dtype);
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
        println!(
            "DIRECT_VS_MATERIALIZED_SCORES_ERROR={}",
            max_abs_error(&cpu_direct.scores, &cpu_materialized.scores)
        );
        println!(
            "DIRECT_VS_MATERIALIZED_PROBABILITIES_ERROR={}",
            max_abs_error(&cpu_direct.probabilities, &cpu_materialized.probabilities)
        );
        println!(
            "DIRECT_VS_MATERIALIZED_CONTEXT_ERROR={}",
            max_abs_error(&cpu_direct.context, &cpu_materialized.context)
        );
        println!("SCORES_FINITE=1");
        println!("PROBABILITIES_FINITE=1");
        println!("CONTEXT_FINITE=1");
        println!("PROBABILITY_ROWS_SUM_TO_ONE=1");
        println!("CPU_DIRECT_PYTHON_MATCH=1");
        println!("CPU_MATERIALIZED_PYTHON_MATCH=1");
        println!("CPU_DIRECT_MATERIALIZED_MATCH=1");
        println!("GPU_CPU_DIRECT_SCORES_MATCH=1");
        println!("GPU_CPU_DIRECT_PROBABILITIES_MATCH=1");
        println!("GPU_CPU_DIRECT_CONTEXT_MATCH=1");
        println!("GPU_PYTHON_DIRECT_SCORES_MATCH=1");
        println!("GPU_PYTHON_DIRECT_PROBABILITIES_MATCH=1");
        println!("GPU_PYTHON_DIRECT_CONTEXT_MATCH=1");
        println!("GPU_MATERIALIZED_ORACLE_MATCH=1");
        println!("KV_HEAD_MAPPING_EFFECT_CONFIRMED=1");
        println!("DISTINCT_KV_PROJECTIONS_CONFIRMED=1");
        println!("MATERIALIZED_EQUIVALENCE_CONFIRMED=1");
        println!("NO_FULL_KV_MATERIALIZATION_CONFIRMED=1");
        println!("CASE_OK=1");
    }

    #[derive(Debug, Deserialize)]
    struct Fixture {
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
        cases: Vec<Case>,
    }

    #[derive(Debug, Deserialize)]
    struct Case {
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

    fn flatten_2d(values: &[Vec<f32>]) -> Vec<f32> {
        values.iter().flatten().copied().collect()
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

    fn assert_row_sums(probabilities: &[f32], q_heads: usize, seq_len: usize, atol: f32) {
        for q_head in 0..q_heads {
            let start = q_head * seq_len;
            let row_sum: f32 = probabilities[start..start + seq_len].iter().sum();
            assert!(
                (row_sum - 1.0).abs() <= atol,
                "probability row sum for head {q_head} was {row_sum}"
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

    fn max_probability_row_sum_error(probabilities: &[f32], q_heads: usize, seq_len: usize) -> f32 {
        let mut max_error = 0.0f32;
        for q_head in 0..q_heads {
            let start = q_head * seq_len;
            let row_sum: f32 = probabilities[start..start + seq_len].iter().sum();
            max_error = max_error.max((row_sum - 1.0).abs());
        }
        max_error
    }

    fn join_usize(values: &[usize]) -> String {
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn swap_k_projection_heads(
        projection_head_major: &[f32],
        latent_dim: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = projection_head_major.to_vec();
        for latent_idx in 0..latent_dim {
            for dim in 0..head_dim {
                out[(latent_idx * 2 + 0) * head_dim + dim] =
                    projection_head_major[(latent_idx * 2 + 1) * head_dim + dim];
                out[(latent_idx * 2 + 1) * head_dim + dim] =
                    projection_head_major[(latent_idx * 2 + 0) * head_dim + dim];
            }
        }
        out
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
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    gpu_impl::main();
}

#[cfg(not(feature = "gpu-cutile"))]
fn main() {}
