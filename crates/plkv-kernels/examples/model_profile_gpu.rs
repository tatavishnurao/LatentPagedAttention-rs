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
    use plkv_kernels::cutile::model_profile::model_profile_kernel;

    const Q_HEADS: usize = 16;
    const KV_HEADS: usize = 4;
    const GROUP_SIZE: usize = 4;
    const HEAD_DIM: usize = 64;
    const LATENT_DIM: usize = 32;
    const BLOCK_SIZE: usize = 16;
    const MAX_SEQ_LEN: usize = 1024;
    const NUM_PHYSICAL_BLOCKS: usize = 64;

    pub fn main() {
        let block_table = model_block_table();
        let q = deterministic_values(Q_HEADS * HEAD_DIM, 0.011, -0.4);
        let logical_latent = deterministic_values(MAX_SEQ_LEN * LATENT_DIM, 0.007, -1.2);
        let latent_physical_f32 = logical_to_physical_latent(&logical_latent, &block_table);
        let latent_f16 =
            quantize_f32_to_f16_storage(&latent_physical_f32).expect("latent quantization failed");
        let k_projection = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.005, -0.7);
        let v_projection = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.006, 0.3);
        let k_head_major = projection_head_major(&k_projection);
        let v_head_major = projection_head_major(&v_projection);

        for active_seq_len in [17usize, 129, 513, 1024] {
            run_case(
                active_seq_len,
                &q,
                &latent_f16,
                &block_table,
                &k_projection,
                &v_projection,
                &k_head_major,
                &v_head_major,
            );
        }

        println!("PROFILE=model_small");
        println!("MODEL_Q_HEADS={Q_HEADS}");
        println!("MODEL_KV_HEADS={KV_HEADS}");
        println!("MODEL_GROUP_SIZE={GROUP_SIZE}");
        println!("MODEL_HEAD_DIM={HEAD_DIM}");
        println!("MODEL_LATENT_DIM={LATENT_DIM}");
        println!("MODEL_BLOCK_SIZE={BLOCK_SIZE}");
        println!("MODEL_MAX_SEQ_LEN={MAX_SEQ_LEN}");
        println!("MODEL_PHYSICAL_BLOCKS={NUM_PHYSICAL_BLOCKS}");
        println!(
            "MODEL_BLOCK_TABLE_CHECKSUM={}",
            checksum_usize(&block_table)
        );
        println!("MODEL_SHAPED_PROFILE_GPU_OK=1");
        println!("MODEL_SHAPED_PARTIAL_BLOCK_OK=1");
        println!("MODEL_SHAPED_MAX_SEQUENCE_OK=1");
        println!("MODEL_SHAPED_NON_IDENTITY_MAPPING_OK=1");
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        active_seq_len: usize,
        q: &[f32],
        latent_f16: &[f16],
        block_table: &[usize],
        k_projection: &[f32],
        v_projection: &[f32],
        k_head_major: &[f32],
        v_head_major: &[f32],
    ) {
        let cpu = direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
            q,
            latent_f16,
            block_table,
            k_projection,
            v_projection,
            Q_HEADS,
            KV_HEADS,
            MAX_SEQ_LEN,
            active_seq_len,
            LATENT_DIM,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_SIZE,
            NUM_PHYSICAL_BLOCKS,
        )
        .expect("CPU model reference failed");

        let q_device = upload_f32(q.to_vec(), &[Q_HEADS, HEAD_DIM]);
        let latent_device = upload_f16(
            latent_f16.to_vec(),
            &[NUM_PHYSICAL_BLOCKS * BLOCK_SIZE, LATENT_DIM],
        );
        let table_device = upload_i32(
            block_table.iter().map(|&value| value as i32).collect(),
            &[64],
        );
        let active_device = upload_i32(vec![active_seq_len as i32], &[1]);
        let k_device = upload_f32(k_head_major.to_vec(), &[KV_HEADS * LATENT_DIM, HEAD_DIM]);
        let v_device = upload_f32(v_head_major.to_vec(), &[KV_HEADS * LATENT_DIM, HEAD_DIM]);

        let scores_out = api::zeros::<f32>(&[Q_HEADS, MAX_SEQ_LEN])
            .sync()
            .expect("scores allocation failed");
        let (scores_part, _, _, _, _, _) = model_profile_kernel::model_small_scores_fp16_storage(
            scores_out.partition([1, BLOCK_SIZE]),
            &q_device,
            &latent_device,
            &table_device,
            &active_device,
            &k_device,
        )
        .sync()
        .expect("model score kernel failed");
        let scores = scores_part.unpartition();
        let probabilities_out = api::zeros::<f32>(&[Q_HEADS, MAX_SEQ_LEN])
            .sync()
            .expect("probability allocation failed");
        let (probabilities_part, _, _) = model_profile_kernel::model_small_softmax_1024_runtime(
            probabilities_out.partition([1, MAX_SEQ_LEN]),
            &scores,
            &active_device,
        )
        .sync()
        .expect("model softmax kernel failed");
        let probabilities = probabilities_part.unpartition();
        let context_out = api::zeros::<f32>(&[Q_HEADS, HEAD_DIM])
            .sync()
            .expect("context allocation failed");
        let (context_part, _, _, _, _) = model_profile_kernel::model_small_context_fp16_storage(
            context_out.partition([1, HEAD_DIM]),
            &probabilities,
            &latent_device,
            &table_device,
            &v_device,
        )
        .sync()
        .expect("model context kernel failed");

        let gpu_scores = scores.to_host_vec().sync().expect("score readback failed");
        let gpu_probabilities = probabilities
            .to_host_vec()
            .sync()
            .expect("probability readback failed");
        let gpu_context = context_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("context readback failed");
        assert_close(&gpu_scores, &cpu.scores, 5e-3, 1e-5, "model scores");
        assert_close(
            &gpu_probabilities,
            &cpu.probabilities,
            2e-4,
            1e-5,
            "model probabilities",
        );
        assert_close(&gpu_context, &cpu.context, 2e-3, 1e-5, "model context");
        assert_inactive_zero(&gpu_probabilities, active_seq_len);
        let row_sum_error = max_probability_row_sum_error(&gpu_probabilities);
        assert!(row_sum_error <= 1e-4);

        let identity = upload_i32((0..64).collect(), &[64]);
        let identity_scores_out = api::zeros::<f32>(&[Q_HEADS, MAX_SEQ_LEN])
            .sync()
            .expect("identity allocation failed");
        let (identity_scores_part, _, _, _, _, _) =
            model_profile_kernel::model_small_scores_fp16_storage(
                identity_scores_out.partition([1, BLOCK_SIZE]),
                &q_device,
                &latent_device,
                &identity,
                &active_device,
                &k_device,
            )
            .sync()
            .expect("identity score kernel failed");
        let identity_scores = identity_scores_part
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("identity readback failed");
        assert!(max_abs_error(&gpu_scores, &identity_scores) > 1e-5);

        println!("PROFILE=model_small");
        println!("ACTIVE_SEQ_LEN={active_seq_len}");
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
        println!("MODEL_SHAPED_CASE_OK=1");
    }

    fn model_block_table() -> Vec<usize> {
        (0..NUM_PHYSICAL_BLOCKS)
            .map(|logical| (logical * 17 + 11) % NUM_PHYSICAL_BLOCKS)
            .collect()
    }
    fn deterministic_values(len: usize, step: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let lane = (index % 257) as f32;
                ((lane * step + offset).sin() * 0.75) + ((index % 13) as f32 - 6.0) * 0.01
            })
            .collect()
    }
    fn logical_to_physical_latent(logical: &[f32], table: &[usize]) -> Vec<f32> {
        let mut physical = vec![0.0f32; logical.len()];
        for (logical_block, &physical_block) in table.iter().enumerate() {
            let logical_start = logical_block * BLOCK_SIZE * LATENT_DIM;
            let physical_start = physical_block * BLOCK_SIZE * LATENT_DIM;
            physical[physical_start..physical_start + BLOCK_SIZE * LATENT_DIM]
                .copy_from_slice(&logical[logical_start..logical_start + BLOCK_SIZE * LATENT_DIM]);
        }
        physical
    }
    fn projection_head_major(canonical: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; canonical.len()];
        for kv in 0..KV_HEADS {
            for latent in 0..LATENT_DIM {
                for dim in 0..HEAD_DIM {
                    let src = (latent * KV_HEADS + kv) * HEAD_DIM + dim;
                    let dst = (kv * LATENT_DIM + latent) * HEAD_DIM + dim;
                    out[dst] = canonical[src];
                }
            }
        }
        out
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
    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= atol + rtol * expected.abs(),
                "{label} at {index}: actual={actual}, expected={expected}"
            );
        }
    }
    fn assert_inactive_zero(probabilities: &[f32], active: usize) {
        for head in 0..Q_HEADS {
            assert!(
                probabilities[head * MAX_SEQ_LEN + active..(head + 1) * MAX_SEQ_LEN]
                    .iter()
                    .all(|&value| value == 0.0)
            );
        }
    }
    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }
    fn max_probability_row_sum_error(p: &[f32]) -> f32 {
        (0..Q_HEADS)
            .map(|head| {
                (p[head * MAX_SEQ_LEN..(head + 1) * MAX_SEQ_LEN]
                    .iter()
                    .sum::<f32>()
                    - 1.0)
                    .abs()
            })
            .fold(0.0, f32::max)
    }
    fn checksum_usize(values: &[usize]) -> u64 {
        values.iter().fold(0u64, |acc, value| {
            acc.wrapping_mul(1_000_003).wrapping_add(*value as u64)
        })
    }
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    gpu_impl::main();
}

#[cfg(not(feature = "gpu-cutile"))]
fn main() {}
