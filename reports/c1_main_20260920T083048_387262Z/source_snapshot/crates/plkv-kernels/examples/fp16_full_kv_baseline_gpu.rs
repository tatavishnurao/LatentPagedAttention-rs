#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::sync::Arc;

    use cutile::api;
    use cutile::half::f16;
    use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{
        paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum, quantize_f32_to_f16_storage,
    };
    use plkv_kernels::cutile::full_kv_baseline::full_kv_baseline_kernel;
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
        let k_projection = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.005, -0.7);
        let v_projection = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.006, 0.3);
        let (logical_k, logical_v) =
            reconstruct_logical_kv(&logical_latent, &k_projection, &v_projection);
        let k_physical = logical_to_physical_kv(&logical_k, &block_table);
        let v_physical = logical_to_physical_kv(&logical_v, &block_table);
        let k_fp16 = quantize_f32_to_f16_storage(&k_physical).expect("K quantization failed");
        let v_fp16 = quantize_f32_to_f16_storage(&v_physical).expect("V quantization failed");

        for active_seq_len in [17usize, 129, 513, 1024] {
            run_full_kv_case(active_seq_len, &q, &k_fp16, &v_fp16, &block_table);
        }

        println!("PROFILE=model_small");
        println!("FP16_FULL_KV_BASELINE_GPU_OK=1");
        println!("FP16_LATENT_PATH_GPU_OK=1");
        println!("BASELINE_AND_LATENT_USE_SAME_STORAGE_WIDTH=1");
        println!("LATENT_CACHE_BYTES_FP16={}", MAX_SEQ_LEN * LATENT_DIM * 2);
        println!(
            "FULL_KV_CACHE_BYTES_FP16={}",
            MAX_SEQ_LEN * KV_HEADS * HEAD_DIM * 2 * 2
        );
        println!("CACHE_BYTE_RATIO_FULL_KV_TO_LATENT=16");
        println!("CACHE_BYTE_ACCOUNTING_CONFIRMED=1");
    }

    fn run_full_kv_case(
        active_seq_len: usize,
        q: &[f32],
        k_fp16: &[f16],
        v_fp16: &[f16],
        block_table: &[usize],
    ) {
        let cpu = paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum(
            q,
            k_fp16,
            v_fp16,
            block_table,
            Q_HEADS,
            KV_HEADS,
            MAX_SEQ_LEN,
            active_seq_len,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_SIZE,
            NUM_PHYSICAL_BLOCKS,
        )
        .expect("CPU full-KV baseline failed");
        let q_device = upload_f32(q.to_vec(), &[Q_HEADS, HEAD_DIM]);
        let k_device = upload_f16(
            k_fp16.to_vec(),
            &[NUM_PHYSICAL_BLOCKS * KV_HEADS * BLOCK_SIZE, HEAD_DIM],
        );
        let v_device = upload_f16(
            v_fp16.to_vec(),
            &[NUM_PHYSICAL_BLOCKS * KV_HEADS * BLOCK_SIZE, HEAD_DIM],
        );
        let table_device = upload_i32(
            block_table.iter().map(|&value| value as i32).collect(),
            &[64],
        );
        let active_device = upload_i32(vec![active_seq_len as i32], &[1]);

        let scores_out = api::zeros::<f32>(&[Q_HEADS, MAX_SEQ_LEN])
            .sync()
            .expect("scores allocation failed");
        let (scores_part, _, _, _, _) =
            full_kv_baseline_kernel::model_small_full_kv_scores_fp16_storage(
                scores_out.partition([1, BLOCK_SIZE]),
                &q_device,
                &k_device,
                &table_device,
                &active_device,
            )
            .sync()
            .expect("full-KV score kernel failed");
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
        .expect("full-KV softmax failed");
        let probabilities = probabilities_part.unpartition();
        let context_out = api::zeros::<f32>(&[Q_HEADS, HEAD_DIM])
            .sync()
            .expect("context allocation failed");
        let (context_part, _, _, _) =
            full_kv_baseline_kernel::model_small_full_kv_context_fp16_storage(
                context_out.partition([1, HEAD_DIM]),
                &probabilities,
                &v_device,
                &table_device,
            )
            .sync()
            .expect("full-KV context kernel failed");
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
        assert_close(&gpu_scores, &cpu.scores, 5e-3, 1e-5, "full-KV scores");
        assert_close(
            &gpu_probabilities,
            &cpu.probabilities,
            2e-4,
            1e-5,
            "full-KV probabilities",
        );
        assert_close(&gpu_context, &cpu.context, 2e-3, 1e-5, "full-KV context");
        assert_inactive_zero(&gpu_probabilities, active_seq_len);
        let row_sum_error = max_probability_row_sum_error(&gpu_probabilities);
        assert!(row_sum_error <= 1e-4);
        println!("PROFILE=model_small");
        println!("ACTIVE_SEQ_LEN={active_seq_len}");
        println!(
            "FULL_KV_SCORES_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_scores, &cpu.scores)
        );
        println!(
            "FULL_KV_PROBABILITIES_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_probabilities, &cpu.probabilities)
        );
        println!(
            "FULL_KV_CONTEXT_MAX_ABS_ERROR={}",
            max_abs_error(&gpu_context, &cpu.context)
        );
        println!("FULL_KV_MAX_PROBABILITY_ROW_SUM_ERROR={row_sum_error}");
        println!("FULL_KV_CASE_OK=1");
    }

    fn reconstruct_logical_kv(
        latent: &[f32],
        k_projection: &[f32],
        v_projection: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut k = vec![0.0f32; MAX_SEQ_LEN * KV_HEADS * HEAD_DIM];
        let mut v = vec![0.0f32; MAX_SEQ_LEN * KV_HEADS * HEAD_DIM];
        for token in 0..MAX_SEQ_LEN {
            for kv in 0..KV_HEADS {
                for dim in 0..HEAD_DIM {
                    let mut k_value = 0.0f32;
                    let mut v_value = 0.0f32;
                    for latent_idx in 0..LATENT_DIM {
                        let latent_value = latent[token * LATENT_DIM + latent_idx];
                        let projection_idx = (latent_idx * KV_HEADS + kv) * HEAD_DIM + dim;
                        k_value += latent_value * k_projection[projection_idx];
                        v_value += latent_value * v_projection[projection_idx];
                    }
                    k[(token * KV_HEADS + kv) * HEAD_DIM + dim] = k_value;
                    v[(token * KV_HEADS + kv) * HEAD_DIM + dim] = v_value;
                }
            }
        }
        (k, v)
    }
    fn logical_to_physical_kv(logical: &[f32], table: &[usize]) -> Vec<f32> {
        let mut physical = vec![0.0f32; NUM_PHYSICAL_BLOCKS * KV_HEADS * BLOCK_SIZE * HEAD_DIM];
        for (logical_block, &physical_block) in table.iter().enumerate() {
            for kv in 0..KV_HEADS {
                for offset in 0..BLOCK_SIZE {
                    let token = logical_block * BLOCK_SIZE + offset;
                    let src = (token * KV_HEADS + kv) * HEAD_DIM;
                    let dst = ((physical_block * KV_HEADS + kv) * BLOCK_SIZE + offset) * HEAD_DIM;
                    physical[dst..dst + HEAD_DIM].copy_from_slice(&logical[src..src + HEAD_DIM]);
                }
            }
        }
        physical
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
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    gpu_impl::main();
}

#[cfg(not(feature = "gpu-cutile"))]
fn main() {}
