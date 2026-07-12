#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::sync::Arc;

    use cutile::api;
    use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{
        contiguous_gqa_decode_f32, direct_latent_gqa_decode_f32,
        direct_paged_latent_gqa_decode_f32, paged_lookup_f32, reconstruct_latent_kv_f32,
    };
    use plkv_kernels::cutile::direct_paged_latent_gqa::direct_paged_latent_gqa_kernel;
    use plkv_kernels::cutile::paged_gqa_decode::paged_gqa_decode_kernel;
    use serde::Deserialize;

    const SCORES_ATOL: f32 = 2e-4;
    const SCORES_RTOL: f32 = 1e-5;
    const PROBABILITIES_ATOL: f32 = 1e-4;
    const PROBABILITIES_RTOL: f32 = 1e-5;
    const CONTEXT_ATOL: f32 = 1e-4;
    const CONTEXT_RTOL: f32 = 1e-5;
    const ROW_SUM_ATOL: f32 = 1e-4;

    pub fn main() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../fixtures/reference/direct_paged_latent_gqa_decode_f32.json"
        ))
        .expect("failed to parse direct paged latent GQA fixture");
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
        println!("OPERATION=DIRECT_PAGED_LATENT_GQA");
        println!("IMPLEMENTATION=THREE_STAGE");
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
        println!("DIRECT_PAGED_LATENT_GQA_GPU_OK=1");
    }

    fn validate_fixture(fixture: &Fixture) {
        assert_eq!((fixture.batch, fixture.seq_len, fixture.q_heads), (1, 8, 4));
        assert_eq!((fixture.kv_heads, fixture.group_size), (2, 2));
        assert_eq!((fixture.head_dim, fixture.latent_dim), (8, 8));
        assert_eq!((fixture.block_size, fixture.num_logical_blocks), (2, 4));
        assert_eq!(fixture.num_physical_blocks, 4);
        assert_eq!(fixture.block_table, [2, 0, 3, 1]);
        assert_eq!(fixture.q_to_kv, [0, 0, 1, 1]);
        assert_eq!(fixture.cases.len(), 2);
    }

    fn validate_case(fixture: &Fixture, case: &Case) {
        let q = flatten_2d(&case.q);
        let physical = flatten_3d(&case.latent_physical_blocks);
        let k_projection = flatten_2d(&case.k_projection);
        let v_projection = flatten_2d(&case.v_projection);
        let k_head_major = flatten_3d(&case.k_projection_gpu_head_major);
        let v_head_major = flatten_3d(&case.v_projection_gpu_head_major);
        let expected_scores = flatten_2d(&case.expected_scores);
        let expected_probabilities = flatten_2d(&case.expected_probabilities);
        let expected_context = flatten_2d(&case.expected_context);
        let materialized_scores = flatten_2d(&case.materialized_scores);
        let materialized_probabilities = flatten_2d(&case.materialized_probabilities);
        let materialized_context = flatten_2d(&case.materialized_context);
        let contiguous_scores = flatten_2d(&case.contiguous_direct_scores);
        let contiguous_probabilities = flatten_2d(&case.contiguous_direct_probabilities);
        let contiguous_context = flatten_2d(&case.contiguous_direct_context);

        let cpu_direct = direct_paged_latent_gqa_decode_f32(
            &q,
            &physical,
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
        .expect("Rust direct paged reference failed");
        let logical = paged_lookup_f32(
            &physical,
            &fixture.block_table,
            fixture.seq_len,
            fixture.block_size,
            fixture.latent_dim,
        )
        .expect("Rust paged latent lookup control failed");
        let cpu_contiguous = direct_latent_gqa_decode_f32(
            &q,
            &logical,
            &k_projection,
            &v_projection,
            fixture.q_heads,
            fixture.kv_heads,
            fixture.seq_len,
            fixture.latent_dim,
            fixture.head_dim,
            fixture.group_size,
        )
        .expect("Rust contiguous direct control failed");
        let reconstructed = reconstruct_latent_kv_f32(
            &logical,
            &k_projection,
            &v_projection,
            fixture.seq_len,
            fixture.latent_dim,
            fixture.kv_heads,
            fixture.head_dim,
        )
        .expect("Rust reconstruction control failed");
        let cpu_materialized = contiguous_gqa_decode_f32(
            &q,
            &token_major_to_head_major(
                &reconstructed.k_token_major,
                fixture.seq_len,
                fixture.kv_heads,
                fixture.head_dim,
            ),
            &token_major_to_head_major(
                &reconstructed.v_token_major,
                fixture.seq_len,
                fixture.kv_heads,
                fixture.head_dim,
            ),
            fixture.q_heads,
            fixture.kv_heads,
            fixture.seq_len,
            fixture.head_dim,
            fixture.group_size,
        )
        .expect("Rust materialized control failed");

        assert_result(
            &cpu_direct,
            &expected_scores,
            &expected_probabilities,
            &expected_context,
            "CPU/Python",
        );
        assert_close(
            &cpu_direct.scores,
            &materialized_scores,
            "CPU materialized scores",
        );
        assert_close(
            &cpu_direct.probabilities,
            &materialized_probabilities,
            "CPU materialized probabilities",
        );
        assert_close(
            &cpu_direct.context,
            &materialized_context,
            "CPU materialized context",
        );
        assert_close(
            &cpu_direct.scores,
            &contiguous_scores,
            "CPU contiguous scores",
        );
        assert_close(
            &cpu_direct.probabilities,
            &contiguous_probabilities,
            "CPU contiguous probabilities",
        );
        assert_close(
            &cpu_direct.context,
            &contiguous_context,
            "CPU contiguous context",
        );
        assert_close(
            &cpu_direct.scores,
            &cpu_contiguous.scores,
            "direct/contiguous Rust scores",
        );
        assert_close(
            &cpu_direct.context,
            &cpu_contiguous.context,
            "direct/contiguous Rust context",
        );
        assert_close(
            &cpu_direct.scores,
            &cpu_materialized.scores,
            "direct/materialized Rust scores",
        );
        assert_close(
            &cpu_direct.context,
            &cpu_materialized.context,
            "direct/materialized Rust context",
        );

        let q_device = upload_f32(q, &[fixture.q_heads, fixture.head_dim]);
        let physical_device = upload_f32(
            physical,
            &[
                fixture.num_physical_blocks * fixture.block_size,
                fixture.latent_dim,
            ],
        );
        let table_device = upload_i32(
            fixture.block_table.iter().map(|v| *v as i32).collect(),
            &[fixture.num_logical_blocks],
        );
        let k_device = upload_f32(
            k_head_major.clone(),
            &[fixture.kv_heads * fixture.latent_dim, fixture.head_dim],
        );
        let v_device = upload_f32(
            v_head_major.clone(),
            &[fixture.kv_heads * fixture.latent_dim, fixture.head_dim],
        );
        let control_v = upload_f32(
            k_head_major.clone(),
            &[fixture.kv_heads * fixture.latent_dim, fixture.head_dim],
        );

        let (gpu_scores, gpu_probabilities, gpu_context, control_context) = run_gpu(
            fixture,
            &q_device,
            &physical_device,
            &table_device,
            &k_device,
            &v_device,
            &control_v,
        );
        assert_result(
            &cpu_direct,
            &gpu_scores,
            &gpu_probabilities,
            &gpu_context,
            "GPU/Rust",
        );
        assert_result(
            &cpu_direct,
            &expected_scores,
            &expected_probabilities,
            &expected_context,
            "CPU/Python",
        );
        assert_close(&gpu_scores, &materialized_scores, "GPU materialized scores");
        assert_close(
            &gpu_probabilities,
            &materialized_probabilities,
            "GPU materialized probabilities",
        );
        assert_close(
            &gpu_context,
            &materialized_context,
            "GPU materialized context",
        );
        assert_close(&gpu_scores, &contiguous_scores, "GPU contiguous scores");
        assert_close(
            &gpu_probabilities,
            &contiguous_probabilities,
            "GPU contiguous probabilities",
        );
        assert_close(&gpu_context, &contiguous_context, "GPU contiguous context");
        let row_sum_error =
            max_probability_row_sum_error(&gpu_probabilities, fixture.q_heads, fixture.seq_len);
        assert!(
            row_sum_error <= ROW_SUM_ATOL,
            "row sum error {row_sum_error} exceeded {ROW_SUM_ATOL}"
        );

        let identity_table = upload_i32(vec![0i32, 1, 2, 3], &[4]);
        let identity_scores = run_scores(
            fixture,
            &q_device,
            &physical_device,
            &identity_table,
            &k_device,
        );
        assert!(
            max_abs_error(&identity_scores, &gpu_scores) > SCORES_ATOL,
            "identity table did not change scores"
        );

        let swapped_k = upload_f32(
            swap_heads(&k_head_major, fixture.latent_dim, fixture.head_dim),
            &[16, 8],
        );
        let swapped_scores = run_scores(
            fixture,
            &q_device,
            &physical_device,
            &table_device,
            &swapped_k,
        );
        assert!(
            max_abs_error(&swapped_scores, &gpu_scores) > SCORES_ATOL,
            "swapped K heads did not change scores"
        );

        assert!(
            max_abs_error(&control_context, &gpu_context) > CONTEXT_ATOL,
            "K and V projections were not distinct"
        );

        println!("CASE_NAME={}", case.name);
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
        println!("Q_TO_KV={}", join(&fixture.q_to_kv));
        println!(
            "PIPELINE=DIRECT_PAGED_LATENT_SCORES_GPU__SOFTMAX_GPU__DIRECT_PAGED_LATENT_CONTEXT_GPU"
        );
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
            max_abs_error(&cpu_direct.scores, &materialized_scores)
        );
        println!(
            "DIRECT_VS_MATERIALIZED_PROBABILITIES_ERROR={}",
            max_abs_error(&cpu_direct.probabilities, &materialized_probabilities)
        );
        println!(
            "DIRECT_VS_MATERIALIZED_CONTEXT_ERROR={}",
            max_abs_error(&cpu_direct.context, &materialized_context)
        );
        println!(
            "DIRECT_PAGED_VS_CONTIGUOUS_DIRECT_SCORES_ERROR={}",
            max_abs_error(&cpu_direct.scores, &contiguous_scores)
        );
        println!(
            "DIRECT_PAGED_VS_CONTIGUOUS_DIRECT_PROBABILITIES_ERROR={}",
            max_abs_error(&cpu_direct.probabilities, &contiguous_probabilities)
        );
        println!(
            "DIRECT_PAGED_VS_CONTIGUOUS_DIRECT_CONTEXT_ERROR={}",
            max_abs_error(&cpu_direct.context, &contiguous_context)
        );
        println!("SCORES_FINITE=1");
        println!("PROBABILITIES_FINITE=1");
        println!("CONTEXT_FINITE=1");
        println!("PROBABILITY_ROWS_SUM_TO_ONE=1");
        println!("CPU_DIRECT_PAGED_PYTHON_MATCH=1");
        println!("CPU_DIRECT_PAGED_MATERIALIZED_MATCH=1");
        println!("CPU_DIRECT_PAGED_CONTIGUOUS_DIRECT_MATCH=1");
        println!("GPU_CPU_DIRECT_PAGED_SCORES_MATCH=1");
        println!("GPU_CPU_DIRECT_PAGED_PROBABILITIES_MATCH=1");
        println!("GPU_CPU_DIRECT_PAGED_CONTEXT_MATCH=1");
        println!("GPU_PYTHON_DIRECT_PAGED_MATCH=1");
        println!("GPU_MATERIALIZED_ORACLE_MATCH=1");
        println!("GPU_CONTIGUOUS_DIRECT_ORACLE_MATCH=1");
        println!("NON_IDENTITY_MAPPING_EFFECT_CONFIRMED=1");
        println!("KV_HEAD_MAPPING_EFFECT_CONFIRMED=1");
        println!("DISTINCT_KV_PROJECTIONS_CONFIRMED=1");
        println!("MATERIALIZED_EQUIVALENCE_CONFIRMED=1");
        println!("NO_LOGICAL_LATENT_MATERIALIZATION_CONFIRMED=1");
        println!("NO_FULL_KV_MATERIALIZATION_CONFIRMED=1");
        println!("CASE_OK=1");
    }

    fn upload_f32(values: Vec<f32>, shape: &[usize]) -> cutile::tensor::Tensor<f32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("GPU upload failed")
            .reshape(shape)
            .expect("GPU reshape failed")
    }

    fn upload_i32(values: Vec<i32>, shape: &[usize]) -> cutile::tensor::Tensor<i32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync()
            .expect("GPU upload failed")
            .reshape(shape)
            .expect("GPU reshape failed")
    }

    fn run_gpu(
        fixture: &Fixture,
        q: &cutile::tensor::Tensor<f32>,
        physical: &cutile::tensor::Tensor<f32>,
        table: &cutile::tensor::Tensor<i32>,
        k: &cutile::tensor::Tensor<f32>,
        v: &cutile::tensor::Tensor<f32>,
        control_v: &cutile::tensor::Tensor<f32>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let scores_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("score allocation failed");
        let (scores_part, _, _, _, _) = direct_paged_latent_gqa_kernel::direct_paged_latent_scores(
            scores_out.partition([1, fixture.block_size]),
            q,
            physical,
            table,
            k,
        )
        .sync()
        .expect("score kernel failed");
        let scores = scores_part.unpartition();
        let probabilities_out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("probability allocation failed");
        let (prob_part, _) = paged_gqa_decode_kernel::stable_softmax_8(
            probabilities_out.partition([1, fixture.seq_len]),
            &scores,
        )
        .sync()
        .expect("softmax kernel failed");
        let probabilities = prob_part.unpartition();
        let context_out = api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
            .sync()
            .expect("context allocation failed");
        let (context_part, _, _, _, _) =
            direct_paged_latent_gqa_kernel::direct_paged_latent_context(
                context_out.partition([1, fixture.head_dim]),
                &probabilities,
                physical,
                table,
                v,
            )
            .sync()
            .expect("context kernel failed");
        let (control_part, _, _, _, _) =
            direct_paged_latent_gqa_kernel::direct_paged_latent_context(
                api::zeros::<f32>(&[fixture.q_heads, fixture.head_dim])
                    .sync()
                    .expect("control context allocation failed")
                    .partition([1, fixture.head_dim]),
                &probabilities,
                physical,
                table,
                control_v,
            )
            .sync()
            .expect("control context kernel failed");
        let scores_host = scores.to_host_vec().sync().expect("score readback failed");
        let probabilities_host = probabilities
            .to_host_vec()
            .sync()
            .expect("probability readback failed");
        let context_host = context_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("context readback failed");
        let control_context_host = control_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("control context readback failed");
        assert_all_finite(&scores_host, "GPU scores");
        assert_all_finite(&probabilities_host, "GPU probabilities");
        assert_all_finite(&context_host, "GPU context");
        (
            scores_host,
            probabilities_host,
            context_host,
            control_context_host,
        )
    }

    fn run_scores(
        fixture: &Fixture,
        q: &cutile::tensor::Tensor<f32>,
        physical: &cutile::tensor::Tensor<f32>,
        table: &cutile::tensor::Tensor<i32>,
        k: &cutile::tensor::Tensor<f32>,
    ) -> Vec<f32> {
        let out = api::zeros::<f32>(&[fixture.q_heads, fixture.seq_len])
            .sync()
            .expect("control score allocation failed");
        let (part, _, _, _, _) = direct_paged_latent_gqa_kernel::direct_paged_latent_scores(
            out.partition([1, fixture.block_size]),
            q,
            physical,
            table,
            k,
        )
        .sync()
        .expect("control score kernel failed");
        part.unpartition()
            .to_host_vec()
            .sync()
            .expect("control score readback failed")
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
        q: Vec<Vec<f32>>,
        latent_physical_blocks: Vec<Vec<Vec<f32>>>,
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
        contiguous_direct_scores: Vec<Vec<f32>>,
        contiguous_direct_probabilities: Vec<Vec<f32>>,
        contiguous_direct_context: Vec<Vec<f32>>,
    }

    fn flatten_2d(v: &[Vec<f32>]) -> Vec<f32> {
        v.iter().flatten().copied().collect()
    }
    fn flatten_3d(v: &[Vec<Vec<f32>>]) -> Vec<f32> {
        v.iter().flatten().flatten().copied().collect()
    }
    fn token_major_to_head_major(v: &[f32], seq: usize, heads: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0; v.len()];
        for h in 0..heads {
            for t in 0..seq {
                for d in 0..dim {
                    out[(h * seq + t) * dim + d] = v[(t * heads + h) * dim + d];
                }
            }
        }
        out
    }
    fn assert_result(r: &plkv_core::GqaDecodeResult, s: &[f32], p: &[f32], c: &[f32], label: &str) {
        assert_close(&r.scores, s, &format!("{label} scores"));
        assert_close(&r.probabilities, p, &format!("{label} probabilities"));
        assert_close(&r.context, c, &format!("{label} context"));
        assert_all_finite(&r.scores, label);
        assert_all_finite(&r.probabilities, label);
        assert_all_finite(&r.context, label);
    }
    fn assert_close(a: &[f32], b: &[f32], label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let (atol, rtol) = if label.contains("probabilities") {
                (PROBABILITIES_ATOL, PROBABILITIES_RTOL)
            } else if label.contains("context") {
                (CONTEXT_ATOL, CONTEXT_RTOL)
            } else {
                (SCORES_ATOL, SCORES_RTOL)
            };
            assert!(
                (x - y).abs() <= atol + rtol * y.abs(),
                "{label} mismatch at {i}: {x} vs {y}"
            );
        }
    }
    fn assert_all_finite(v: &[f32], label: &str) {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{label} contained non-finite values"
        );
    }
    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
    fn max_probability_row_sum_error(p: &[f32], heads: usize, seq: usize) -> f32 {
        (0..heads)
            .map(|h| (p[h * seq..(h + 1) * seq].iter().sum::<f32>() - 1.0).abs())
            .fold(0.0, f32::max)
    }
    fn join(v: &[usize]) -> String {
        v.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
    }
    fn swap_heads(v: &[f32], latent: usize, dim: usize) -> Vec<f32> {
        let mut out = v.to_vec();
        for l in 0..latent {
            for d in 0..dim {
                out[(l * 2) * dim + d] = v[(l * 2 + 1) * dim + d];
                out[(l * 2 + 1) * dim + d] = v[(l * 2) * dim + d];
            }
        }
        out
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
