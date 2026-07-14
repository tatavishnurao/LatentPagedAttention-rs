#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::sync::Arc;

    use cutile::api;
    use cutile::half::f16;
    use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{
        direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum, quantize_f32_to_f16_storage,
    };
    use plkv_kernels::cutile::runtime_sequence::runtime_sequence_kernel;
    use serde::Deserialize;

    const SCORE_ATOL: f32 = 2e-4;
    const PROB_ATOL: f32 = 1e-4;
    const CONTEXT_ATOL: f32 = 2e-4;

    pub fn main() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/paged_latent_write_attention_fp16_storage.json"
        ))
        .expect("fixture JSON must parse");
        assert_eq!(fixture.q_heads, 4);
        assert_eq!(fixture.kv_heads, 2);
        assert_eq!(fixture.group_size, 2);
        assert_eq!(fixture.seq_len, 8);
        assert_eq!(fixture.head_dim, 8);
        assert_eq!(fixture.latent_dim, 8);
        assert_eq!(fixture.block_size, 2);
        assert_eq!(fixture.block_table, vec![2, 0, 3, 1]);
        let lengths = [1usize, 3, 4, 7, 8];
        for case in &fixture.cases {
            for &active_seq_len in &lengths {
                run_case(&fixture, case, active_seq_len);
            }
        }
        println!("GPU_NAME={}", gpu_name());
        println!("CUDA_TOOLKIT_PATH=/opt/cuda");
        println!("CUTILE_VERSION=0.2.0");
        println!("OPERATION=RUNTIME_SEQUENCE_DIRECT_PAGED_LATENT_GQA");
        println!("IMPLEMENTATION=THREE_STAGE");
        println!("RUNTIME_ACTIVE_SEQUENCE_LENGTH_CONFIRMED=1");
        println!("PARTIAL_FINAL_BLOCK_MASKING_CONFIRMED=1");
        println!("INACTIVE_PROBABILITIES_ZERO=1");
        println!("ACTIVE_PROBABILITY_ROWS_SUM_TO_ONE=1");
        println!("RUNTIME_BLOCK_TABLE_MAPPING_CONFIRMED=1");
        println!("NO_INACTIVE_TOKEN_CONTEXT_CONTRIBUTION=1");
        println!("PROFILE=tiny");
        println!("TINY_PROFILE_GPU_OK=1");
        println!("RUNTIME_SEQUENCE_GPU_OK=1");
    }

    fn run_case(fixture: &Fixture, case: &Case, active_seq_len: usize) {
        let q = flatten_2d(&case.q);
        let latent_f16 = quantize_f32_to_f16_storage(&flatten_3d(&case.initial_latent_source_f32))
            .expect("FP16 source quantization failed");
        let k = flatten_3d(&case.k_projection_gpu_head_major);
        let v = flatten_3d(&case.v_projection_gpu_head_major);
        let cpu = direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
            &q,
            &latent_f16,
            &fixture.block_table,
            &flatten_2d(&case.k_projection),
            &flatten_2d(&case.v_projection),
            fixture.q_heads,
            fixture.kv_heads,
            fixture.seq_len,
            active_seq_len,
            fixture.latent_dim,
            fixture.head_dim,
            fixture.group_size,
            fixture.block_size,
            fixture.num_physical_blocks,
        )
        .expect("Rust runtime reference failed");

        let q_device = upload_f32(q, &[fixture.q_heads, fixture.head_dim]);
        let latent_device = upload_f16(latent_f16, &[fixture.num_physical_blocks * 2, 8]);
        let table_device = upload_i32(
            fixture
                .block_table
                .iter()
                .map(|&value| value as i32)
                .collect(),
            &[fixture.block_table.len()],
        );
        let active_device = upload_i32(vec![active_seq_len as i32], &[1]);
        let k_device = upload_f32(
            k,
            &[fixture.kv_heads * fixture.latent_dim, fixture.head_dim],
        );
        let v_device = upload_f32(
            v,
            &[fixture.kv_heads * fixture.latent_dim, fixture.head_dim],
        );

        let scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("scores allocation failed");
        let (scores_part, _, _, _, _, _) =
            runtime_sequence_kernel::direct_paged_latent_scores_fp16_runtime(
                scores_out.partition([1, fixture.block_size]),
                &q_device,
                &latent_device,
                &table_device,
                &active_device,
                &k_device,
            )
            .sync()
            .expect("runtime score kernel failed");
        let scores = scores_part.unpartition();
        let probabilities_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("probabilities allocation failed");
        let (probabilities_part, _, _) = runtime_sequence_kernel::stable_softmax_8_runtime(
            probabilities_out.partition([1, fixture.seq_len]),
            &scores,
            &active_device,
        )
        .sync()
        .expect("runtime softmax kernel failed");
        let probabilities = probabilities_part.unpartition();
        let context_out = api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
            .sync()
            .expect("context allocation failed");
        let (context_part, _, _, _, _) =
            runtime_sequence_kernel::direct_paged_latent_context_fp16_runtime(
                context_out.partition([1, fixture.head_dim]),
                &probabilities,
                &latent_device,
                &table_device,
                &v_device,
            )
            .sync()
            .expect("runtime context kernel failed");

        let gpu_scores = scores.to_host_vec().sync().expect("scores readback failed");
        let gpu_probabilities = probabilities
            .to_host_vec()
            .sync()
            .expect("probabilities readback failed");
        let gpu_context = context_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("context readback failed");

        assert_close(&gpu_scores, &cpu.scores, SCORE_ATOL, 1e-5, "scores");
        assert_close(
            &gpu_probabilities,
            &cpu.probabilities,
            PROB_ATOL,
            1e-5,
            "probabilities",
        );
        assert_close(&gpu_context, &cpu.context, CONTEXT_ATOL, 1e-5, "context");
        assert_inactive_zero(
            &gpu_probabilities,
            fixture.q_heads,
            fixture.seq_len,
            active_seq_len,
        );
        let row_sum_error =
            max_probability_row_sum_error(&gpu_probabilities, fixture.q_heads, fixture.seq_len);
        assert!(row_sum_error <= 1e-4);

        let identity_table = upload_i32(vec![0, 1, 2, 3], &[4]);
        let identity_scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("identity scores allocation failed");
        let (identity_scores_part, _, _, _, _, _) =
            runtime_sequence_kernel::direct_paged_latent_scores_fp16_runtime(
                identity_scores_out.partition([1, fixture.block_size]),
                &q_device,
                &latent_device,
                &identity_table,
                &active_device,
                &k_device,
            )
            .sync()
            .expect("identity score kernel failed");
        let identity_scores = identity_scores_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("identity score readback failed");
        assert!(max_abs_error(&gpu_scores, &identity_scores) > SCORE_ATOL);

        println!("CASE_NAME={}", case.name);
        println!("ACTIVE_SEQ_LEN={active_seq_len}");
        println!("MAX_SEQ_LEN={}", fixture.seq_len);
        println!("Q_HEADS={}", fixture.q_heads);
        println!("KV_HEADS={}", fixture.kv_heads);
        println!("GROUP_SIZE={}", fixture.group_size);
        println!("HEAD_DIM={}", fixture.head_dim);
        println!("LATENT_DIM={}", fixture.latent_dim);
        println!("BLOCK_SIZE={}", fixture.block_size);
        println!("BLOCK_TABLE=2,0,3,1");
        println!("PIPELINE=RUNTIME_MASKED_SCORES_GPU__RUNTIME_SOFTMAX_GPU__RUNTIME_CONTEXT_GPU");
        println!(
            "SCORES_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_scores, &cpu.scores)
        );
        println!(
            "PROBABILITIES_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_probabilities, &cpu.probabilities)
        );
        println!(
            "CONTEXT_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_context, &cpu.context)
        );
        println!("MAX_PROBABILITY_ROW_SUM_ERROR={row_sum_error}");
        println!("INACTIVE_PROBABILITIES_ZERO=1");
        println!("ACTIVE_PROBABILITY_ROWS_SUM_TO_ONE=1");
        println!("RUNTIME_BLOCK_TABLE_MAPPING_CONFIRMED=1");
        println!("NO_INACTIVE_TOKEN_CONTEXT_CONTRIBUTION=1");
        println!("CASE_OK=1");
    }

    fn upload_f32(values: Vec<f32>, shape: &[usize]) -> cutile::tensor::Tensor<f32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("f32 upload failed")
            .reshape(shape)
            .expect("f32 reshape failed")
    }
    fn upload_f16(values: Vec<f16>, shape: &[usize]) -> cutile::tensor::Tensor<f16> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("f16 upload failed")
            .reshape(shape)
            .expect("f16 reshape failed")
    }
    fn upload_i32(values: Vec<i32>, shape: &[usize]) -> cutile::tensor::Tensor<i32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("i32 upload failed")
            .reshape(shape)
            .expect("i32 reshape failed")
    }
    fn flatten_2d(values: &[Vec<f32>]) -> Vec<f32> {
        values.iter().flatten().copied().collect()
    }
    fn flatten_3d(values: &[Vec<Vec<f32>>]) -> Vec<f32> {
        values.iter().flatten().flatten().copied().collect()
    }
    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= atol + rtol * expected.abs(),
                "{label} at {index}: actual={actual}, expected={expected}"
            );
        }
    }
    fn assert_inactive_zero(probabilities: &[f32], heads: usize, seq: usize, active: usize) {
        for head in 0..heads {
            assert!(
                probabilities[head * seq + active..(head + 1) * seq]
                    .iter()
                    .all(|&value| value == 0.0)
            );
        }
    }
    fn max_probability_row_sum_error(p: &[f32], heads: usize, seq: usize) -> f32 {
        (0..heads)
            .map(|head| (p[head * seq..(head + 1) * seq].iter().sum::<f32>() - 1.0).abs())
            .fold(0.0, f32::max)
    }
    fn gpu_name() -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "<unknown>".to_string())
    }

    #[derive(Debug, Deserialize)]
    struct Fixture {
        seq_len: usize,
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        head_dim: usize,
        latent_dim: usize,
        block_size: usize,
        num_physical_blocks: usize,
        block_table: Vec<usize>,
        cases: Vec<Case>,
    }

    #[derive(Debug, Deserialize)]
    struct Case {
        name: String,
        q: Vec<Vec<f32>>,
        initial_latent_source_f32: Vec<Vec<Vec<f32>>>,
        k_projection: Vec<Vec<f32>>,
        v_projection: Vec<Vec<f32>>,
        k_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
        v_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
    }
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    gpu_impl::main();
}

#[cfg(not(feature = "gpu-cutile"))]
fn main() {}
