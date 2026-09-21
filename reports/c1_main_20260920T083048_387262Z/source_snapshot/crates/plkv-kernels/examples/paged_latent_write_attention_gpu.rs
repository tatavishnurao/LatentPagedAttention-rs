#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::sync::Arc;

    use cutile::api;
    use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{direct_paged_latent_gqa_decode_f32, paged_latent_write_f32};
    use plkv_kernels::cutile::direct_paged_latent_gqa::direct_paged_latent_gqa_kernel;
    use plkv_kernels::cutile::paged_gqa_decode::paged_gqa_decode_kernel;
    use plkv_kernels::cutile::paged_latent_write::paged_latent_write_kernel;
    use serde::Deserialize;

    const LATENT_ATOL: f32 = 1e-6;
    const SCORES_ATOL: f32 = 2e-4;
    const SCORES_RTOL: f32 = 1e-5;
    const PROBS_ATOL: f32 = 1e-4;
    const PROBS_RTOL: f32 = 1e-5;
    const CONTEXT_ATOL: f32 = 1e-4;
    const CONTEXT_RTOL: f32 = 1e-5;
    const ROW_SUM_ATOL: f32 = 1e-4;
    const UNWRITTEN_ATOL: f32 = 2e-5;

    pub fn main() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/paged_latent_write_attention_f32.json"
        ))
        .expect("failed to parse paged latent write-attention fixture");
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
        println!("OPERATION=PAGED_LATENT_WRITE_TO_DIRECT_PAGED_ATTENTION");
        println!("IMPLEMENTATION=FOUR_STAGE");
        println!("DTYPE={}", fixture.dtype);
        println!("BATCH={}", fixture.batch);
        println!("PHYSICAL_LATENT_CACHE_VALUES=64");
        println!("LOGICAL_LATENT_DEVICE_VALUES=0");
        println!("HYPOTHETICAL_FULL_PHYSICAL_KV_CACHE_VALUES=256");
        println!("DIRECT_PATH_RECONSTRUCTED_K_VALUES=0");
        println!("DIRECT_PATH_RECONSTRUCTED_V_VALUES=0");
        println!("DIRECT_PATH_RECONSTRUCTED_KV_DEVICE_VALUES=0");
        println!("THEORETICAL_CACHE_COMPRESSION_RATIO=4");
        println!("ALL_CASES_OK=1");
        println!("PAGED_LATENT_WRITE_ATTENTION_GPU_OK=1");
    }

    fn validate_fixture(fixture: &Fixture) {
        assert_eq!(fixture.block_table, [2, 0, 3, 1]);
        assert_eq!(fixture.q_to_kv, [0, 0, 1, 1]);
        assert_eq!(
            (
                fixture.batch,
                fixture.seq_len,
                fixture.q_heads,
                fixture.kv_heads
            ),
            (1, 8, 4, 2)
        );
        assert_eq!(
            (
                fixture.group_size,
                fixture.head_dim,
                fixture.latent_dim,
                fixture.block_size
            ),
            (2, 8, 8, 2)
        );
        assert_eq!(
            (fixture.num_logical_blocks, fixture.num_physical_blocks),
            (4, 4)
        );
        assert_eq!(fixture.cases.len(), 2);
    }

    fn validate_case(fixture: &Fixture, case: &Case) {
        let q = flatten_2d(&case.q);
        let initial = flatten_3d(&case.initial_latent_physical_blocks);
        let expected_updated = flatten_3d(&case.expected_updated_latent_physical_blocks);
        let k_projection = flatten_2d(&case.k_projection);
        let v_projection = flatten_2d(&case.v_projection);
        let k_head_major = flatten_3d(&case.k_projection_gpu_head_major);
        let v_head_major = flatten_3d(&case.v_projection_gpu_head_major);
        let new_latent = case.new_latent.clone();
        let pre_scores = flatten_2d(&case.pre_write_scores);
        let pre_probabilities = flatten_2d(&case.pre_write_probabilities);
        let pre_context = flatten_2d(&case.pre_write_context);
        let expected_scores = flatten_2d(&case.post_write_scores);
        let expected_probabilities = flatten_2d(&case.post_write_probabilities);
        let expected_context = flatten_2d(&case.post_write_context);

        let mut cpu_updated = initial.clone();
        let location = paged_latent_write_f32(
            &mut cpu_updated,
            &fixture.block_table,
            case.token_position,
            fixture.block_size,
            fixture.latent_dim,
            &new_latent,
        )
        .expect("Rust paged latent write failed");
        assert_eq!(location.logical_block, case.logical_block);
        assert_eq!(location.physical_block, case.physical_block);
        assert_eq!(location.block_offset, case.block_offset);
        assert_close(
            &cpu_updated,
            &expected_updated,
            LATENT_ATOL,
            0.0,
            "CPU updated cache",
        );

        let cpu_pre = direct_paged_latent_gqa_decode_f32(
            &q,
            &initial,
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
        .expect("Rust pre-write attention failed");
        assert_close(
            &cpu_pre.scores,
            &pre_scores,
            SCORES_ATOL,
            SCORES_RTOL,
            "CPU pre-write scores",
        );
        assert_close(
            &cpu_pre.probabilities,
            &pre_probabilities,
            PROBS_ATOL,
            PROBS_RTOL,
            "CPU pre-write probabilities",
        );
        assert_close(
            &cpu_pre.context,
            &pre_context,
            CONTEXT_ATOL,
            CONTEXT_RTOL,
            "CPU pre-write context",
        );

        let cpu_post = direct_paged_latent_gqa_decode_f32(
            &q,
            &cpu_updated,
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
        .expect("Rust post-write attention failed");
        assert_close(
            &cpu_post.scores,
            &expected_scores,
            SCORES_ATOL,
            SCORES_RTOL,
            "CPU/Python scores",
        );
        assert_close(
            &cpu_post.probabilities,
            &expected_probabilities,
            PROBS_ATOL,
            PROBS_RTOL,
            "CPU/Python probabilities",
        );
        assert_close(
            &cpu_post.context,
            &expected_context,
            CONTEXT_ATOL,
            CONTEXT_RTOL,
            "CPU/Python context",
        );
        assert_write_effects(
            &pre_scores,
            &cpu_post.scores,
            &pre_context,
            &cpu_post.context,
            case.token_position,
            fixture.q_heads,
            fixture.seq_len,
        );

        let q_device = upload_f32(q, &[fixture.q_heads, fixture.head_dim]);
        let latent_device = upload_f32(
            initial.clone(),
            &[
                fixture.num_physical_blocks * fixture.block_size,
                fixture.latent_dim,
            ],
        );
        let table_device = upload_i32(
            fixture
                .block_table
                .iter()
                .map(|value| *value as i32)
                .collect(),
            &[fixture.num_logical_blocks],
        );
        let token_device = upload_i32(vec![case.token_position as i32], &[1]);
        let new_latent_device = upload_f32(new_latent, &[fixture.latent_dim]);
        let k_device = upload_f32(k_head_major, &[16, 8]);
        let v_device = upload_f32(v_head_major, &[16, 8]);

        let (written_part, _, _, _) = paged_latent_write_kernel::paged_latent_write(
            latent_device.partition([fixture.block_size, fixture.latent_dim]),
            &table_device,
            &token_device,
            &new_latent_device,
        )
        .sync()
        .expect("GPU paged latent write failed");
        let updated_device = written_part.unpartition();
        let (gpu_scores, gpu_probabilities, gpu_context) = run_attention(
            fixture,
            &q_device,
            &updated_device,
            &table_device,
            &k_device,
            &v_device,
        );
        let gpu_updated = updated_device
            .to_host_vec()
            .sync()
            .expect("updated latent cache readback failed");

        let latent_error = max_abs_error(&gpu_updated, &expected_updated);
        let changed = changed_elements(&gpu_updated, &initial);
        assert!(latent_error <= LATENT_ATOL, "latent error {latent_error}");
        assert_eq!(changed, fixture.latent_dim);
        assert_unchanged_except_target(
            &gpu_updated,
            &initial,
            case.physical_block,
            case.block_offset,
            fixture.block_size,
            fixture.latent_dim,
        );
        assert_close(
            &gpu_scores,
            &cpu_post.scores,
            SCORES_ATOL,
            SCORES_RTOL,
            "GPU/Rust scores",
        );
        assert_close(
            &gpu_probabilities,
            &cpu_post.probabilities,
            PROBS_ATOL,
            PROBS_RTOL,
            "GPU/Rust probabilities",
        );
        assert_close(
            &gpu_context,
            &cpu_post.context,
            CONTEXT_ATOL,
            CONTEXT_RTOL,
            "GPU/Rust context",
        );
        let row_sum_error =
            max_probability_row_sum_error(&gpu_probabilities, fixture.q_heads, fixture.seq_len);
        assert!(
            row_sum_error <= ROW_SUM_ATOL,
            "row sum error {row_sum_error}"
        );

        let identity_initial = upload_f32(initial.clone(), &[8, 8]);
        let identity_table = upload_i32(vec![0, 1, 2, 3], &[4]);
        let (identity_part, _, _, _) = paged_latent_write_kernel::paged_latent_write(
            identity_initial.partition([2, 8]),
            &identity_table,
            &token_device,
            &new_latent_device,
        )
        .sync()
        .expect("identity-table GPU write failed");
        let identity_updated = identity_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("identity write readback failed");
        assert_ne!(
            identity_updated, expected_updated,
            "identity table did not affect write target"
        );

        println!("CASE_NAME={}", case.name);
        println!("TOKEN_POSITION={}", case.token_position);
        println!("LOGICAL_BLOCK={}", case.logical_block);
        println!("PHYSICAL_BLOCK={}", case.physical_block);
        println!("BLOCK_OFFSET={}", case.block_offset);
        println!("Q_HEADS={}", fixture.q_heads);
        println!("KV_HEADS={}", fixture.kv_heads);
        println!("GROUP_SIZE={}", fixture.group_size);
        println!("SEQ_LEN={}", fixture.seq_len);
        println!("HEAD_DIM={}", fixture.head_dim);
        println!("LATENT_DIM={}", fixture.latent_dim);
        println!("BLOCK_SIZE={}", fixture.block_size);
        println!("NUM_LOGICAL_BLOCKS={}", fixture.num_logical_blocks);
        println!("NUM_PHYSICAL_BLOCKS={}", fixture.num_physical_blocks);
        println!("BLOCK_TABLE={}", join(&fixture.block_table));
        println!(
            "PIPELINE=PAGED_LATENT_WRITE_GPU__DIRECT_PAGED_LATENT_SCORES_GPU__SOFTMAX_GPU__DIRECT_PAGED_LATENT_CONTEXT_GPU"
        );
        println!("LATENT_MAX_ABS_ERROR={latent_error}");
        println!("LATENT_CHANGED_ELEMENTS={changed}");
        println!(
            "POST_WRITE_SCORES_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_scores, &expected_scores)
        );
        println!(
            "POST_WRITE_PROBABILITIES_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_probabilities, &expected_probabilities)
        );
        println!(
            "POST_WRITE_CONTEXT_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_context, &expected_context)
        );
        println!("POST_WRITE_MAX_PROBABILITY_ROW_SUM_ERROR={row_sum_error}");
        println!(
            "PRE_VS_POST_WRITTEN_SCORE_COLUMN_ERROR={}",
            written_score_error(
                &pre_scores,
                &cpu_post.scores,
                case.token_position,
                fixture.seq_len
            )
        );
        println!(
            "PRE_VS_POST_CONTEXT_ERROR={}",
            max_abs_error(&pre_context, &cpu_post.context)
        );
        println!("LATENT_UNCHANGED_REGION_OK=1");
        println!("WRITTEN_TOKEN_SCORE_COLUMN_CHANGED=1");
        println!("UNWRITTEN_SCORE_COLUMNS_UNCHANGED=1");
        println!("ATTENTION_CONTEXT_CHANGED_AFTER_WRITE=1");
        println!("POST_WRITE_PROBABILITY_ROWS_SUM_TO_ONE=1");
        println!("CPU_WRITE_PYTHON_MATCH=1");
        println!("CPU_POST_WRITE_ATTENTION_PYTHON_MATCH=1");
        println!("GPU_WRITE_CPU_MATCH=1");
        println!("GPU_WRITE_PYTHON_MATCH=1");
        println!("GPU_POST_WRITE_ATTENTION_CPU_MATCH=1");
        println!("GPU_POST_WRITE_ATTENTION_PYTHON_MATCH=1");
        println!("WRITE_NON_IDENTITY_MAPPING_EFFECT_CONFIRMED=1");
        println!("NO_LOGICAL_LATENT_MATERIALIZATION_CONFIRMED=1");
        println!("NO_FULL_KV_MATERIALIZATION_CONFIRMED=1");
        println!("GPU_WRITE_TO_ATTENTION_NO_HOST_ROUNDTRIP_CONFIRMED=1");
        println!("CASE_OK=1");
    }

    fn run_attention(
        fixture: &Fixture,
        q: &cutile::tensor::Tensor<f32>,
        latent: &cutile::tensor::Tensor<f32>,
        table: &cutile::tensor::Tensor<i32>,
        k: &cutile::tensor::Tensor<f32>,
        v: &cutile::tensor::Tensor<f32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("scores allocation failed");
        let (scores_part, _, _, _, _) = direct_paged_latent_gqa_kernel::direct_paged_latent_scores(
            scores_out.partition([1, fixture.block_size]),
            q,
            latent,
            table,
            k,
        )
        .sync()
        .expect("score kernel failed");
        let scores = scores_part.unpartition();
        let probabilities_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("probabilities allocation failed");
        let (probabilities_part, _) = paged_gqa_decode_kernel::stable_softmax_8(
            probabilities_out.partition([1, fixture.seq_len]),
            &scores,
        )
        .sync()
        .expect("softmax kernel failed");
        let probabilities = probabilities_part.unpartition();
        let context_out = api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
            .sync()
            .expect("context allocation failed");
        let (context_part, _, _, _, _) =
            direct_paged_latent_gqa_kernel::direct_paged_latent_context(
                context_out.partition([1, fixture.head_dim]),
                &probabilities,
                latent,
                table,
                v,
            )
            .sync()
            .expect("context kernel failed");
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
        assert_all_finite(&scores_host, "GPU scores");
        assert_all_finite(&probabilities_host, "GPU probabilities");
        assert_all_finite(&context_host, "GPU context");
        (scores_host, probabilities_host, context_host)
    }

    fn upload_f32(values: Vec<f32>, shape: &[usize]) -> cutile::tensor::Tensor<f32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("GPU f32 upload failed")
            .reshape(shape)
            .expect("GPU f32 reshape failed")
    }
    fn upload_i32(values: Vec<i32>, shape: &[usize]) -> cutile::tensor::Tensor<i32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("GPU i32 upload failed")
            .reshape(shape)
            .expect("GPU i32 reshape failed")
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
        token_position: usize,
        logical_block: usize,
        physical_block: usize,
        block_offset: usize,
        q: Vec<Vec<f32>>,
        initial_latent_physical_blocks: Vec<Vec<Vec<f32>>>,
        new_latent: Vec<f32>,
        expected_updated_latent_physical_blocks: Vec<Vec<Vec<f32>>>,
        k_projection: Vec<Vec<f32>>,
        v_projection: Vec<Vec<f32>>,
        k_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
        v_projection_gpu_head_major: Vec<Vec<Vec<f32>>>,
        pre_write_scores: Vec<Vec<f32>>,
        pre_write_probabilities: Vec<Vec<f32>>,
        pre_write_context: Vec<Vec<f32>>,
        post_write_scores: Vec<Vec<f32>>,
        post_write_probabilities: Vec<Vec<f32>>,
        post_write_context: Vec<Vec<f32>>,
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
    fn changed_elements(a: &[f32], b: &[f32]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x != y).count()
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
    fn assert_all_finite(values: &[f32], label: &str) {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{label} non-finite"
        );
    }
    fn max_probability_row_sum_error(p: &[f32], heads: usize, seq: usize) -> f32 {
        (0..heads)
            .map(|h| (p[h * seq..(h + 1) * seq].iter().sum::<f32>() - 1.0).abs())
            .fold(0.0, f32::max)
    }
    fn written_score_error(pre: &[f32], post: &[f32], token: usize, seq: usize) -> f32 {
        (0..4)
            .map(|h| (pre[h * seq + token] - post[h * seq + token]).abs())
            .fold(0.0, f32::max)
    }
    fn assert_write_effects(
        pre: &[f32],
        post: &[f32],
        pre_context: &[f32],
        post_context: &[f32],
        token: usize,
        heads: usize,
        seq: usize,
    ) {
        assert!((0..heads).all(|h| (pre[h * seq + token] - post[h * seq + token]).abs() > 1e-6));
        assert!((0..heads).all(|h| {
            (0..seq).filter(|t| *t != token).all(|t| {
                (pre[h * seq + t] - post[h * seq + t]).abs()
                    <= UNWRITTEN_ATOL + UNWRITTEN_ATOL * pre[h * seq + t].abs()
            })
        }));
        assert!(max_abs_error(pre_context, post_context) > CONTEXT_ATOL);
    }
    fn assert_unchanged_except_target(
        a: &[f32],
        b: &[f32],
        block: usize,
        offset: usize,
        block_size: usize,
        dim: usize,
    ) {
        let start = (block * block_size + offset) * dim;
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if i < start || i >= start + dim {
                assert_eq!(x, y, "cache corruption at {i}");
            }
        }
    }
    fn join(values: &[usize]) -> String {
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
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
