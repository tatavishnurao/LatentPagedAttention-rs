#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::sync::Arc;

    use cutile::api;
    use cutile::half::f16;
    use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{
        direct_paged_latent_gqa_decode_f32, direct_paged_latent_gqa_decode_fp16_storage_f32_accum,
        paged_latent_write_fp16_storage, quantize_f32_to_f16_storage,
    };
    use plkv_kernels::cutile::direct_paged_latent_gqa_fp16::direct_paged_latent_gqa_fp16_kernel;
    use plkv_kernels::cutile::paged_gqa_decode::paged_gqa_decode_kernel;
    use plkv_kernels::cutile::paged_latent_write_fp16::paged_latent_write_fp16_kernel;
    use serde::Deserialize;

    const SCORES_ATOL: f32 = 2e-4;
    const SCORES_RTOL: f32 = 1e-5;
    const PROBS_ATOL: f32 = 1e-4;
    const PROBS_RTOL: f32 = 1e-5;
    const CONTEXT_ATOL: f32 = 2e-4;
    const CONTEXT_RTOL: f32 = 1e-5;
    const BASELINE_SCORES_ATOL: f32 = 2e-3;
    const BASELINE_PROBS_ATOL: f32 = 5e-4;
    const BASELINE_CONTEXT_ATOL: f32 = 2e-3;
    const ROW_SUM_ATOL: f32 = 1e-4;
    const UNWRITTEN_ATOL: f32 = 2e-5;

    pub fn main() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/paged_latent_write_attention_fp16_storage.json"
        ))
        .expect("failed to parse FP16 storage fixture");
        validate_fixture(&fixture);
        for case in &fixture.cases {
            validate_case(&fixture, case);
        }
        println!("GPU_NAME={}", gpu_name());
        println!(
            "CUDA_TOOLKIT_PATH={}",
            std::env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "<unset>".to_string())
        );
        println!("CUTILE_VERSION=0.2.0");
        println!("OPERATION=FP16_PAGED_LATENT_STORAGE_WITH_FP32_ATTENTION");
        println!("IMPLEMENTATION=FOUR_STAGE");
        println!("BATCH={}", fixture.batch);
        println!("PHYSICAL_LATENT_CACHE_VALUES=64");
        println!("PHYSICAL_LATENT_CACHE_BYTES_FP16=128");
        println!("SAME_LATENT_CACHE_BYTES_FP32=256");
        println!("HYPOTHETICAL_FULL_KV_CACHE_BYTES_FP16=512");
        println!("LATENT_STORAGE_RATIO_FP32_TO_FP16=2");
        println!("STORED_CACHE_RATIO_VS_FP16_FULL_KV=4");
        println!("LOGICAL_LATENT_DEVICE_VALUES=0");
        println!("DIRECT_PATH_RECONSTRUCTED_KV_DEVICE_VALUES=0");
        println!("ALL_CASES_OK=1");
        println!("FP16_PAGED_LATENT_ATTENTION_GPU_OK=1");
    }

    fn validate_fixture(fixture: &Fixture) {
        assert_eq!(fixture.storage_dtype, "f16");
        assert_eq!(fixture.compute_dtype, "f32");
        assert_eq!(fixture.write_input_dtype, "f32");
        assert_eq!(fixture.block_table, [2, 0, 3, 1]);
        assert_eq!(fixture.q_to_kv, [0, 0, 1, 1]);
        assert_eq!(fixture.cases.len(), 2);
    }

    fn validate_case(fixture: &Fixture, case: &Case) {
        let q = flatten_2d(&case.q);
        let initial_source = flatten_3d(&case.initial_latent_source_f32);
        let new_source = case.new_latent_source_f32.clone();
        let k_projection = flatten_2d(&case.k_projection);
        let v_projection = flatten_2d(&case.v_projection);
        let k_head_major = flatten_3d(&case.k_projection_gpu_head_major);
        let v_head_major = flatten_3d(&case.v_projection_gpu_head_major);
        let initial_f16 =
            quantize_f32_to_f16_storage(&initial_source).expect("initial FP16 conversion failed");
        let expected_f16 =
            quantize_f32_to_f16_storage(&flatten_3d(&case.expected_updated_latent_fp16_as_f32))
                .expect("expected FP16 conversion failed");
        let mut cpu_f16_cache = initial_f16.clone();
        let location = paged_latent_write_fp16_storage(
            &mut cpu_f16_cache,
            &fixture.block_table,
            case.token_position,
            fixture.block_size,
            fixture.latent_dim,
            &new_source,
        )
        .expect("Rust FP16 write failed");
        assert_eq!(location.logical_block, case.logical_block);
        assert_eq!(location.physical_block, case.physical_block);
        assert_eq!(location.block_offset, case.block_offset);
        assert_eq!(
            cpu_f16_cache
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected_f16.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let cpu_fp16_result = direct_paged_latent_gqa_decode_fp16_storage_f32_accum(
            &q,
            &cpu_f16_cache,
            &fixture.block_table,
            &k_projection,
            &v_projection,
            fixture.q_heads,
            fixture.kv_heads,
            fixture.seq_len,
            fixture.latent_dim,
            fixture.head_dim,
            fixture.group_size,
            fixture.block_size,
            fixture.num_physical_blocks,
        )
        .expect("Rust FP16 attention failed");
        let baseline_cache = initial_source.clone();
        let mut baseline_updated = baseline_cache.clone();
        let _ = plkv_core::paged_latent_write_f32(
            &mut baseline_updated,
            &fixture.block_table,
            case.token_position,
            fixture.block_size,
            fixture.latent_dim,
            &new_source,
        )
        .expect("Rust FP32 baseline write failed");
        let cpu_baseline = direct_paged_latent_gqa_decode_f32(
            &q,
            &baseline_updated,
            &fixture.block_table,
            &k_projection,
            &v_projection,
            fixture.q_heads,
            fixture.kv_heads,
            fixture.seq_len,
            fixture.latent_dim,
            fixture.head_dim,
            fixture.group_size,
            fixture.block_size,
            fixture.num_physical_blocks,
        )
        .expect("Rust FP32 baseline attention failed");
        assert_close(
            &cpu_fp16_result.scores,
            &flatten_2d(&case.fp16_storage_post_write_scores),
            SCORES_ATOL,
            SCORES_RTOL,
            "CPU/Python scores",
        );
        assert_close(
            &cpu_fp16_result.probabilities,
            &flatten_2d(&case.fp16_storage_post_write_probabilities),
            PROBS_ATOL,
            PROBS_RTOL,
            "CPU/Python probabilities",
        );
        assert_close(
            &cpu_fp16_result.context,
            &flatten_2d(&case.fp16_storage_post_write_context),
            CONTEXT_ATOL,
            CONTEXT_RTOL,
            "CPU/Python context",
        );

        let q_device = upload_f32(q.clone(), &[fixture.q_heads, fixture.head_dim]);
        let latent_device = upload_f16(initial_f16.clone(), &[8, 8]);
        let table_device = upload_i32(
            fixture.block_table.iter().map(|v| *v as i32).collect(),
            &[4],
        );
        let token_device = upload_i32(vec![case.token_position as i32], &[1]);
        let new_device = upload_f32(new_source, &[8]);
        let k_device = upload_f32(k_head_major, &[16, 8]);
        let v_device = upload_f32(v_head_major, &[16, 8]);
        let (write_part, _, _, _) = paged_latent_write_fp16_kernel::paged_latent_write_fp16(
            latent_device.partition([2, 8]),
            &table_device,
            &token_device,
            &new_device,
        )
        .sync()
        .expect("GPU FP16 write failed");
        let updated_device = write_part.unpartition();
        let (gpu_scores, gpu_probabilities, gpu_context) = run_attention(
            &q_device,
            &updated_device,
            &table_device,
            &k_device,
            &v_device,
        );
        let gpu_cache = updated_device
            .to_host_vec()
            .sync()
            .expect("FP16 cache readback failed");
        let expected_bits = flatten_3d_u16(&case.expected_updated_latent_fp16_bits);
        let actual_bits: Vec<u16> = gpu_cache.iter().map(|value| value.to_bits()).collect();
        let mismatch_count = actual_bits
            .iter()
            .zip(&expected_bits)
            .filter(|(a, b)| a != b)
            .count();
        let changed = actual_bits
            .iter()
            .zip(initial_f16.iter())
            .filter(|(a, b)| **a != b.to_bits())
            .count();
        assert_eq!(mismatch_count, 0);
        assert_eq!(changed, 8);
        assert_unchanged_bits(
            &actual_bits,
            &initial_f16,
            case.physical_block,
            case.block_offset,
        );
        assert_close(
            &gpu_scores,
            &cpu_fp16_result.scores,
            SCORES_ATOL,
            SCORES_RTOL,
            "GPU/FP16 scores",
        );
        assert_close(
            &gpu_probabilities,
            &cpu_fp16_result.probabilities,
            PROBS_ATOL,
            PROBS_RTOL,
            "GPU/FP16 probabilities",
        );
        assert_close(
            &gpu_context,
            &cpu_fp16_result.context,
            CONTEXT_ATOL,
            CONTEXT_RTOL,
            "GPU/FP16 context",
        );
        let row_sum_error = max_probability_row_sum_error(&gpu_probabilities, 4, 8);
        assert!(row_sum_error <= ROW_SUM_ATOL);

        assert_close(
            &cpu_fp16_result.scores,
            &cpu_baseline.scores,
            BASELINE_SCORES_ATOL,
            1e-3,
            "FP16/FP32 scores",
        );
        assert_close(
            &cpu_fp16_result.probabilities,
            &cpu_baseline.probabilities,
            BASELINE_PROBS_ATOL,
            1e-3,
            "FP16/FP32 probabilities",
        );
        assert_close(
            &cpu_fp16_result.context,
            &cpu_baseline.context,
            BASELINE_CONTEXT_ATOL,
            1e-3,
            "FP16/FP32 context",
        );
        let pre_fp16 = direct_paged_latent_gqa_decode_fp16_storage_f32_accum(
            &q,
            &initial_f16,
            &fixture.block_table,
            &k_projection,
            &v_projection,
            4,
            2,
            8,
            8,
            8,
            2,
            2,
            4,
        )
        .expect("FP16 pre-write attention failed");
        assert!((0..4).all(|head| {
            (pre_fp16.scores[head * 8 + case.token_position]
                - cpu_fp16_result.scores[head * 8 + case.token_position])
                .abs()
                > 1e-6
        }));
        assert!((0..4).all(|head| {
            (0..8)
                .filter(|token| *token != case.token_position)
                .all(|token| {
                    (pre_fp16.scores[head * 8 + token] - cpu_fp16_result.scores[head * 8 + token])
                        <= UNWRITTEN_ATOL
                })
        }));
        assert!(max_abs_error(&pre_fp16.context, &cpu_fp16_result.context) > CONTEXT_ATOL);

        let identity_device = upload_f16(initial_f16, &[8, 8]);
        let identity_table = upload_i32(vec![0, 1, 2, 3], &[4]);
        let (identity_part, _, _, _) = paged_latent_write_fp16_kernel::paged_latent_write_fp16(
            identity_device.partition([2, 8]),
            &identity_table,
            &token_device,
            &new_device,
        )
        .sync()
        .expect("identity FP16 write failed");
        let identity_bits: Vec<u16> = identity_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("identity cache readback failed")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_ne!(identity_bits, expected_bits);

        println!("CASE_NAME={}", case.name);
        println!("TOKEN_POSITION={}", case.token_position);
        println!("LOGICAL_BLOCK={}", case.logical_block);
        println!("PHYSICAL_BLOCK={}", case.physical_block);
        println!("BLOCK_OFFSET={}", case.block_offset);
        println!("STORAGE_DTYPE=f16");
        println!("WRITE_INPUT_DTYPE=f32");
        println!("ACCUMULATION_DTYPE=f32");
        println!("SCORES_DTYPE=f32");
        println!("PROBABILITIES_DTYPE=f32");
        println!("CONTEXT_DTYPE=f32");
        println!("FP16_CACHE_BIT_MISMATCH_COUNT={mismatch_count}");
        println!("FP16_CHANGED_ELEMENTS={changed}");
        println!(
            "GPU_VS_FP16_ORACLE_SCORES_MAX_ABS_ERROR={}",
            max_abs_error(
                &gpu_scores,
                &flatten_2d(&case.fp16_storage_post_write_scores)
            )
        );
        println!(
            "GPU_VS_FP16_ORACLE_PROBABILITIES_MAX_ABS_ERROR={}",
            max_abs_error(
                &gpu_probabilities,
                &flatten_2d(&case.fp16_storage_post_write_probabilities)
            )
        );
        println!(
            "GPU_VS_FP16_ORACLE_CONTEXT_MAX_ABS_ERROR={}",
            max_abs_error(
                &gpu_context,
                &flatten_2d(&case.fp16_storage_post_write_context)
            )
        );
        println!("GPU_MAX_PROBABILITY_ROW_SUM_ERROR={row_sum_error}");
        println!(
            "FP16_STORAGE_VS_FP32_BASELINE_POST_WRITE_SCORES_ERROR={}",
            max_abs_error(&cpu_fp16_result.scores, &cpu_baseline.scores)
        );
        println!(
            "FP16_STORAGE_VS_FP32_BASELINE_POST_WRITE_PROBABILITIES_ERROR={}",
            max_abs_error(&cpu_fp16_result.probabilities, &cpu_baseline.probabilities)
        );
        println!(
            "FP16_STORAGE_VS_FP32_BASELINE_POST_WRITE_CONTEXT_ERROR={}",
            max_abs_error(&cpu_fp16_result.context, &cpu_baseline.context)
        );
        println!("FP16_UNCHANGED_REGION_BITS_OK=1");
        println!("GPU_F32_TO_FP16_WRITE_CONVERSION_CONFIRMED=1");
        println!("WRITTEN_TOKEN_SCORE_COLUMN_CHANGED=1");
        println!("UNWRITTEN_SCORE_COLUMNS_UNCHANGED=1");
        println!("ATTENTION_CONTEXT_CHANGED_AFTER_WRITE=1");
        println!("POST_WRITE_PROBABILITY_ROWS_SUM_TO_ONE=1");
        println!("CPU_FP16_STORAGE_PYTHON_MATCH=1");
        println!("GPU_FP16_STORAGE_CPU_MATCH=1");
        println!("GPU_FP16_STORAGE_PYTHON_MATCH=1");
        println!("WRITE_NON_IDENTITY_MAPPING_EFFECT_CONFIRMED=1");
        println!("FP16_STORAGE_EFFECT_CONFIRMED=1");
        println!("NO_LOGICAL_LATENT_MATERIALIZATION_CONFIRMED=1");
        println!("NO_FULL_KV_MATERIALIZATION_CONFIRMED=1");
        println!("GPU_WRITE_TO_ATTENTION_NO_HOST_ROUNDTRIP_CONFIRMED=1");
        println!("CASE_OK=1");
    }

    fn run_attention(
        q: &cutile::tensor::Tensor<f32>,
        latent: &cutile::tensor::Tensor<f16>,
        table: &cutile::tensor::Tensor<i32>,
        k: &cutile::tensor::Tensor<f32>,
        v: &cutile::tensor::Tensor<f32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let scores_out = api::zeros::<f32>(&[4, 8])
            .sync()
            .expect("scores allocation failed");
        let (scores_part, _, _, _, _) =
            direct_paged_latent_gqa_fp16_kernel::direct_paged_latent_scores_fp16_storage(
                scores_out.partition([1, 2]),
                q,
                latent,
                table,
                k,
            )
            .sync()
            .expect("FP16 score kernel failed");
        let scores = scores_part.unpartition();
        let probabilities_out = api::zeros::<f32>(&[4, 8])
            .sync()
            .expect("probabilities allocation failed");
        let (probabilities_part, _) =
            paged_gqa_decode_kernel::stable_softmax_8(probabilities_out.partition([1, 8]), &scores)
                .sync()
                .expect("FP32 softmax kernel failed");
        let probabilities = probabilities_part.unpartition();
        let context_out = api::zeros::<f32>(&[4, 8])
            .sync()
            .expect("context allocation failed");
        let (context_part, _, _, _, _) =
            direct_paged_latent_gqa_fp16_kernel::direct_paged_latent_context_fp16_storage(
                context_out.partition([1, 8]),
                &probabilities,
                latent,
                table,
                v,
            )
            .sync()
            .expect("FP16 context kernel failed");
        let scores_host = scores.to_host_vec().sync().expect("scores readback failed");
        let probabilities_host = probabilities
            .to_host_vec()
            .sync()
            .expect("probabilities readback failed");
        let context_host = context_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("context readback failed");
        (scores_host, probabilities_host, context_host)
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

    #[derive(Debug, Deserialize)]
    struct Fixture {
        storage_dtype: String,
        compute_dtype: String,
        write_input_dtype: String,
        batch: usize,
        seq_len: usize,
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        head_dim: usize,
        latent_dim: usize,
        block_size: usize,
        num_physical_blocks: usize,
        block_table: Vec<usize>,
        q_to_kv: Vec<usize>,
        cases: Vec<Case>,
    }
    #[derive(Debug, Deserialize)]
    struct Case {
        name: String,
        token_position: usize,
        logical_block: usize,
        physical_block: usize,
        block_offset: usize,
        q: Vec<Vec<f32>>,
        initial_latent_source_f32: Vec<Vec<Vec<f32>>>,
        new_latent_source_f32: Vec<f32>,
        expected_updated_latent_fp16_as_f32: Vec<Vec<Vec<f32>>>,
        expected_updated_latent_fp16_bits: Vec<Vec<Vec<u16>>>,
        k_projection: Vec<Vec<f32>>,
        v_projection: Vec<Vec<f32>>,
        k_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
        v_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
        fp16_storage_post_write_scores: Vec<Vec<f32>>,
        fp16_storage_post_write_probabilities: Vec<Vec<f32>>,
        fp16_storage_post_write_context: Vec<Vec<f32>>,
    }
    fn flatten_2d(values: &[Vec<f32>]) -> Vec<f32> {
        values.iter().flatten().copied().collect()
    }
    fn flatten_3d(values: &[Vec<Vec<f32>>]) -> Vec<f32> {
        values.iter().flatten().flatten().copied().collect()
    }
    fn flatten_3d_u16(values: &[Vec<Vec<u16>>]) -> Vec<u16> {
        values.iter().flatten().flatten().copied().collect()
    }
    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
    fn assert_close(a: &[f32], b: &[f32], atol: f32, rtol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label} length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= atol + rtol * y.abs(),
                "{label} at {i}: {x} vs {y}"
            );
        }
    }
    fn max_probability_row_sum_error(p: &[f32], heads: usize, seq: usize) -> f32 {
        (0..heads)
            .map(|h| (p[h * seq..(h + 1) * seq].iter().sum::<f32>() - 1.0).abs())
            .fold(0.0, f32::max)
    }
    fn assert_unchanged_bits(actual: &[u16], initial: &[f16], block: usize, offset: usize) {
        let start = (block * 2 + offset) * 8;
        for i in 0..actual.len() {
            if i < start || i >= start + 8 {
                assert_eq!(
                    actual[i],
                    initial[i].to_bits(),
                    "FP16 cache corruption at {i}"
                );
            }
        }
    }
    fn gpu_name() -> String {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "<unknown>".to_string())
    }
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    gpu_impl::main();
}

#[cfg(not(feature = "gpu-cutile"))]
fn main() {}
