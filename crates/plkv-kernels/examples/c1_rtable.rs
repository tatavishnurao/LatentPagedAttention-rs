#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use cutile::api;
    use cutile::cuda_async::device_context::with_default_device_policy;
    use cutile::cuda_core::{Stream, sys};
    use cutile::half::f16;
    use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{
        GqaDecodeResult, direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum,
        paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum, quantize_f32_to_f16_storage,
    };
    use plkv_kernels::cutile::c1::*;
    use plkv_kernels::cutile::p15b_rtable_kernels::*;
    use serde::Serialize;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;
    const Q_HEADS: usize = 16;
    const KV_HEADS: usize = 4;
    const GROUP_SIZE: usize = 4;
    const HEAD_DIM: usize = 64;
    const LATENT_DIM: usize = 32;
    const BLOCK_SIZE: usize = 16;
    struct Inputs {
        q: Tensor<f32>,
        latent: Tensor<f16>,
        k_projection: Tensor<f32>,
        v_projection: Tensor<f32>,
        k_full: Tensor<f16>,
        v_full: Tensor<f16>,
        table: Tensor<i32>,
        active: Tensor<i32>,
    }
    struct Buffers {
        projected: Option<Tensor<f32>>,
        scores: Option<Tensor<f32>>,
        probabilities: Option<Tensor<f32>>,
        context: Option<Tensor<f32>>,
    }
    #[derive(Debug, Serialize)]
    struct ErrorMetrics {
        max_absolute_error: f64,
        mean_absolute_error: f64,
        rmse: f64,
        max_relative_error: f64,
        max_probability_row_sum_error: f64,
    }
    #[derive(Debug, Serialize)]
    struct ComparisonMetrics {
        element_count: usize,
        finite_left_count: usize,
        finite_right_count: usize,
        nan_count: usize,
        positive_infinity_count: usize,
        negative_infinity_count: usize,
        exact_mismatch_count: usize,
        max_absolute_error: f64,
        max_relative_error: f64,
        mean_absolute_error: f64,
        left_checksum: String,
        right_checksum: String,
    }
    struct Events {
        start: sys::CUevent,
        stop: sys::CUevent,
    }
    impl Events {
        fn new() -> Self {
            let (mut s, mut e) = (std::ptr::null_mut(), std::ptr::null_mut());
            unsafe {
                check_cuda(sys::cuEventCreate(&mut s, 0), "event");
                check_cuda(sys::cuEventCreate(&mut e, 0), "event");
            }
            Self { start: s, stop: e }
        }
        fn measure<F: FnOnce()>(&self, st: &Arc<Stream>, f: F) -> f64 {
            unsafe {
                st.synchronize().unwrap();
                check_cuda(sys::cuEventRecord(self.start, st.cu_stream()), "record");
                f();
                check_cuda(sys::cuEventRecord(self.stop, st.cu_stream()), "record");
                check_cuda(sys::cuEventSynchronize(self.stop), "sync");
                let mut ms = 0.;
                check_cuda(
                    sys::cuEventElapsedTime_v2(&mut ms, self.start, self.stop),
                    "elapsed",
                );
                f64::from(ms)
            }
        }
    }
    impl Drop for Events {
        fn drop(&mut self) {
            unsafe {
                let _ = sys::cuEventDestroy_v2(self.start);
                let _ = sys::cuEventDestroy_v2(self.stop);
            }
        }
    }
    // A1/B1 retain their original kernels. C1 reuses B1 softmax/context unchanged.
    type Launch = fn(&str, &str, &mut Buffers, &Inputs, &Arc<Stream>, usize);
    macro_rules! define_launch {
        ($name:ident, $as:path, $ac:path, $bs:path, $bc:path, $sm:path, $proj:path, $cs:path) => {
            fn $name(
                v: &str,
                phase: &str,
                b: &mut Buffers,
                i: &Inputs,
                st: &Arc<Stream>,
                n: usize,
            ) {
                if phase == "pipeline" {
                    if v == "C1" {
                        $name(v, "projection", b, i, st, n);
                    }
                    for component in ["score", "softmax", "context"] {
                        $name(v, component, b, i, st, n);
                    }
                    return;
                }
                match phase {
                    "projection" => {
                        assert_eq!(v, "C1");
                        let out = b.projected.take().unwrap();
                        let (p, _, _) = unsafe {
                            $proj(out.partition([1, 32]), &i.q, &i.k_projection)
                                .async_on(st)
                                .unwrap()
                        };
                        b.projected = Some(p.unpartition());
                    }
                    "score" => {
                        let out = b.scores.take().unwrap();
                        b.scores = Some(match v {
                            "A1" => {
                                let (p, _, _, _, _) = unsafe {
                                    $as(
                                        out.partition([1, 16]),
                                        &i.q,
                                        &i.k_full,
                                        &i.table,
                                        &i.active,
                                    )
                                    .async_on(st)
                                    .unwrap()
                                };
                                p.unpartition()
                            }
                            "B1" => {
                                let (p, _, _, _, _, _) = unsafe {
                                    $bs(
                                        out.partition([1, 16]),
                                        &i.q,
                                        &i.latent,
                                        &i.table,
                                        &i.active,
                                        &i.k_projection,
                                    )
                                    .async_on(st)
                                    .unwrap()
                                };
                                p.unpartition()
                            }
                            "C1" => {
                                let (p, _, _, _, _) = unsafe {
                                    $cs(
                                        out.partition([1, 16]),
                                        b.projected.as_ref().unwrap(),
                                        &i.latent,
                                        &i.table,
                                        &i.active,
                                    )
                                    .async_on(st)
                                    .unwrap()
                                };
                                p.unpartition()
                            }
                            _ => unreachable!(),
                        });
                    }
                    "softmax" => {
                        let out = b.probabilities.take().unwrap();
                        let (p, _, _) = unsafe {
                            $sm(out.partition([1, n]), b.scores.as_ref().unwrap(), &i.active)
                                .async_on(st)
                                .unwrap()
                        };
                        b.probabilities = Some(p.unpartition());
                    }
                    "context" => {
                        let out = b.context.take().unwrap();
                        b.context = Some(if v == "A1" {
                            let (p, _, _, _) = unsafe {
                                $ac(
                                    out.partition([1, 64]),
                                    b.probabilities.as_ref().unwrap(),
                                    &i.v_full,
                                    &i.table,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            p.unpartition()
                        } else {
                            let (p, _, _, _, _) = unsafe {
                                $bc(
                                    out.partition([1, 64]),
                                    b.probabilities.as_ref().unwrap(),
                                    &i.latent,
                                    &i.table,
                                    &i.v_projection,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            p.unpartition()
                        });
                    }
                    _ => unreachable!(),
                }
            }
        };
    }
    fn validate(
        n: usize,
        actual: &GqaDecodeResult,
        reference: &GqaDecodeResult,
    ) -> serde_json::Value {
        let mut rows = Vec::new();
        let mut pass = true;
        for (name, a, r) in [
            ("scores", &actual.scores, &reference.scores),
            (
                "probabilities",
                &actual.probabilities,
                &reference.probabilities,
            ),
            ("context", &actual.context, &reference.context),
        ] {
            assert_eq!(a.len(), r.len());
            let mut max_abs = 0.0f64;
            let mut max_rel = 0.0f64;
            let mut location = 0;
            let mut first_failure = None;
            for (j, (&x, &y)) in a.iter().zip(r).enumerate() {
                let abs = f64::from((x - y).abs());
                if abs > max_abs {
                    max_abs = abs;
                    location = j;
                }
                max_rel = max_rel.max(abs / f64::from(y.abs()).max(1e-12));
                // FP32 tree vs serial reference; absolute threshold inherited from repair validation.
                if (!x.is_finite() || !y.is_finite() || abs > 5e-3) && first_failure.is_none() {
                    first_failure = Some(j);
                }
            }
            pass &= first_failure.is_none();
            rows.push(serde_json::json!({"tensor":name,"max_absolute_error":max_abs,"max_relative_error":max_rel,"max_absolute_location":location,"first_failure":first_failure,"elements":a.len()}));
        }
        let normalization = max_probability_row_sum_error(n, &actual.probabilities);
        pass &= normalization <= 1e-4
            && actual
                .probabilities
                .iter()
                .all(|&x| (0.0..=1.0).contains(&x));
        serde_json::json!({"pass":pass,"tensors":rows,"max_probability_row_sum_error":normalization,"atol":5e-3,"rtol":0,"relative_denominator_floor":1e-12})
    }
    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let arg = |key: &str, default: &str| {
            args.windows(2)
                .find(|w| w[0] == key)
                .map(|w| w[1].clone())
                .unwrap_or(default.into())
        };
        let n: usize = arg("--seq", "1024").parse().unwrap();
        let warmup: usize = arg("--warmup", "3").parse().unwrap();
        let iterations: usize = arg("--iterations", "12").parse().unwrap();
        let process: usize = arg("--process", "0").parse().unwrap();
        let dir = PathBuf::from(arg("--output-dir", "reports/c1_native"));
        fs::create_dir_all(&dir).unwrap();
        let launch: Launch = match n {
            1024 => launch_1024,
            2048 => launch_2048,
            4096 => launch_4096,
            8192 => launch_8192,
            _ => panic!("bounded milestone: only 1K/2K/4K/8K"),
        };
        let st = with_default_device_policy(|p| p.next_stream())
            .unwrap()
            .unwrap();
        st.device().bind_to_thread().unwrap();
        let (inp, cf, cb) = make_inputs(n, n, n / 16, &st, -0.4);
        let variants = ["A1", "B1", "C1"];
        let mut buffers: Vec<_> = variants.iter().map(|_| make_buffers(n, &st)).collect();
        for (v, b) in variants.iter().zip(&mut buffers) {
            launch(v, "pipeline", b, &inp, &st, n);
        }
        unsafe {
            st.synchronize().unwrap();
        }
        let outputs: Vec<_> = buffers.iter().map(|b| read_outputs(b, &st)).collect();
        let checks = serde_json::json!({"A1_reference":validate(n,&outputs[0],&cf),"B1_reference":validate(n,&outputs[1],&cb),"C1_reference":validate(n,&outputs[2],&cb),"C1_B1":validate(n,&outputs[2],&outputs[1]),"C1_B1_bitwise_diagnostic":comparison_metrics(&outputs[2],&outputs[1])});
        serde_json::to_writer_pretty(File::create(dir.join("correctness.json")).unwrap(), &checks)
            .unwrap();
        for key in ["A1_reference", "B1_reference", "C1_reference", "C1_B1"] {
            assert_eq!(checks[key]["pass"], true, "{checks}");
        }
        // Export actual inputs/outputs for an independent FP64, reconstructed-K/V oracle.
        let q = deterministic_values(Q_HEADS * HEAD_DIM, 0.011, -0.4);
        let z = deterministic_values(n * LATENT_DIM, 0.007, -1.2);
        let kp = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.005, -0.7);
        let vp = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.006, 0.3);
        let data = serde_json::json!({"q":q,"latent_logical_prequantization":z,"kp":kp,"vp":vp,"scores":outputs[2].scores,"probabilities":outputs[2].probabilities,"context":outputs[2].context,"table":model_block_table(n/16)});
        serde_json::to_writer(
            File::create(dir.join("oracle_inputs_outputs.json")).unwrap(),
            &data,
        )
        .unwrap();
        // A distinct decode query must refresh the projection; never reuse stale projected q.
        {
            let (changed, _, changed_ref) = make_inputs(n, n, n / 16, &st, 0.2);
            launch("C1", "pipeline", &mut buffers[2], &changed, &st, n);
            let changed_output = read_outputs(&buffers[2], &st);
            let refresh = validate(n, &changed_output, &changed_ref);
            serde_json::to_writer_pretty(
                File::create(dir.join("query_refresh.json")).unwrap(),
                &refresh,
            )
            .unwrap();
            assert_eq!(
                refresh["pass"], true,
                "changed-query projection was not refreshed"
            );
            assert_ne!(outputs[2].scores, changed_output.scores);
            // Restore matched resident inputs before warmup and timing.
            launch("C1", "pipeline", &mut buffers[2], &inp, &st, n);
        }
        let phases = ["pipeline", "score", "softmax", "context", "projection"];
        // JIT and fixed warmup outside event intervals.
        for phase in phases {
            for (v, b) in variants.iter().zip(&mut buffers) {
                if phase == "projection" && *v != "C1" {
                    continue;
                }
                for _ in 0..warmup {
                    launch(v, phase, b, &inp, &st, n);
                }
            }
        }
        unsafe {
            st.synchronize().unwrap();
        }
        let ev = Events::new();
        let mut raw = File::options()
            .write(true)
            .create_new(true)
            .open(dir.join("samples.jsonl"))
            .unwrap();
        for iteration in 0..iterations {
            for phase in phases {
                for offset in 0..3 {
                    let index = (iteration + process + offset) % 3;
                    let v = variants[index];
                    if phase == "projection" && v != "C1" {
                        continue;
                    }
                    let ms =
                        ev.measure(&st, || launch(v, phase, &mut buffers[index], &inp, &st, n));
                    writeln!(raw,"{}",serde_json::json!({"seq":n,"process":process,"iteration":iteration,"order":offset,"variant":v,"component":phase,"latency_ms":ms})).unwrap();
                    raw.flush().unwrap();
                }
            }
        }
        println!("C1_OK=1");
    }
    define_launch!(
        launch_1024,
        p15b_full_kv_baseline_kernel_1024::model_small_full_kv_scores_fp16_storage_rtable_1024,
        p15b_full_kv_baseline_kernel_1024::model_small_full_kv_context_fp16_storage_rtable_1024,
        p15b_model_profile_kernel_1024::model_small_scores_fp16_storage_rtable_1024,
        p15b_model_profile_kernel_1024::model_small_context_fp16_storage_rtable_1024,
        p15b_model_profile_kernel_1024::model_small_softmax_1024_runtime,
        c1_kernel_1024::model_small_project_query_once,
        c1_kernel_1024::model_small_scores_fp16_storage_preprojected
    );
    define_launch!(
        launch_2048,
        p15b_full_kv_baseline_kernel_2048::model_small_full_kv_scores_fp16_storage_rtable_2048,
        p15b_full_kv_baseline_kernel_2048::model_small_full_kv_context_fp16_storage_rtable_2048,
        p15b_model_profile_kernel_2048::model_small_scores_fp16_storage_rtable_2048,
        p15b_model_profile_kernel_2048::model_small_context_fp16_storage_rtable_2048,
        p15b_model_profile_kernel_2048::model_small_softmax_2048_runtime,
        c1_kernel_2048::model_small_project_query_once,
        c1_kernel_2048::model_small_scores_fp16_storage_preprojected
    );
    define_launch!(
        launch_4096,
        p15b_full_kv_baseline_kernel_4096::model_small_full_kv_scores_fp16_storage_rtable_4096,
        p15b_full_kv_baseline_kernel_4096::model_small_full_kv_context_fp16_storage_rtable_4096,
        p15b_model_profile_kernel_4096::model_small_scores_fp16_storage_rtable_4096,
        p15b_model_profile_kernel_4096::model_small_context_fp16_storage_rtable_4096,
        p15b_model_profile_kernel_4096::model_small_softmax_4096_runtime,
        c1_kernel_4096::model_small_project_query_once,
        c1_kernel_4096::model_small_scores_fp16_storage_preprojected
    );
    define_launch!(
        launch_8192,
        p15b_full_kv_baseline_kernel_8192::model_small_full_kv_scores_fp16_storage_rtable_8192,
        p15b_full_kv_baseline_kernel_8192::model_small_full_kv_context_fp16_storage_rtable_8192,
        p15b_model_profile_kernel_8192::model_small_scores_fp16_storage_rtable_8192,
        p15b_model_profile_kernel_8192::model_small_context_fp16_storage_rtable_8192,
        p15b_model_profile_kernel_8192::model_small_softmax_8192_runtime,
        c1_kernel_8192::model_small_project_query_once,
        c1_kernel_8192::model_small_scores_fp16_storage_preprojected
    );
    fn make_inputs(
        seq: usize,
        active_seq_len: usize,
        blocks: usize,
        stream: &Arc<Stream>,
        query_offset: f32,
    ) -> (Inputs, GqaDecodeResult, GqaDecodeResult) {
        let block_table = model_block_table(blocks);
        let q = deterministic_values(Q_HEADS * HEAD_DIM, 0.011, query_offset);
        let logical_latent = deterministic_values(seq * LATENT_DIM, 0.007, -1.2);
        let latent_physical_f32 = logical_to_physical_latent(&logical_latent, &block_table);
        let latent_f16 =
            quantize_f32_to_f16_storage(&latent_physical_f32).expect("latent quantization failed");
        let k_projection = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.005, -0.7);
        let v_projection = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.006, 0.3);
        let k_head_major = projection_head_major(&k_projection);
        let v_head_major = projection_head_major(&v_projection);
        let (logical_k, logical_v) =
            reconstruct_logical_kv(seq, &logical_latent, &k_projection, &v_projection);
        let k_physical = logical_to_physical_kv(blocks, &logical_k, &block_table);
        let v_physical = logical_to_physical_kv(blocks, &logical_v, &block_table);
        let k_full_f16 = quantize_f32_to_f16_storage(&k_physical).expect("K quantization failed");
        let v_full_f16 = quantize_f32_to_f16_storage(&v_physical).expect("V quantization failed");

        let cpu_full = paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum(
            &q,
            &k_full_f16,
            &v_full_f16,
            &block_table,
            Q_HEADS,
            KV_HEADS,
            seq,
            active_seq_len,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_SIZE,
            blocks,
        )
        .expect("CPU full-KV reference failed");
        let cpu_latent = direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
            &q,
            &latent_f16,
            &block_table,
            &k_projection,
            &v_projection,
            Q_HEADS,
            KV_HEADS,
            seq,
            active_seq_len,
            LATENT_DIM,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_SIZE,
            blocks,
        )
        .expect("CPU latent reference failed");

        let inputs = Inputs {
            q: upload_f32(q, &[Q_HEADS, HEAD_DIM], stream),
            latent: upload_f16(latent_f16, &[blocks * BLOCK_SIZE, LATENT_DIM], stream),
            k_projection: upload_f32(k_head_major, &[KV_HEADS * LATENT_DIM, HEAD_DIM], stream),
            v_projection: upload_f32(v_head_major, &[KV_HEADS * LATENT_DIM, HEAD_DIM], stream),
            k_full: upload_f16(
                k_full_f16,
                &[blocks * KV_HEADS * BLOCK_SIZE, HEAD_DIM],
                stream,
            ),
            v_full: upload_f16(
                v_full_f16,
                &[blocks * KV_HEADS * BLOCK_SIZE, HEAD_DIM],
                stream,
            ),
            table: upload_i32(
                block_table.iter().map(|&value| value as i32).collect(),
                &[blocks],
                stream,
            ),
            active: upload_i32(vec![active_seq_len as i32], &[1], stream),
        };
        (inputs, cpu_full, cpu_latent)
    }

    fn make_buffers(seq: usize, stream: &Arc<Stream>) -> Buffers {
        Buffers {
            projected: Some(
                api::zeros::<f32>(&[Q_HEADS, LATENT_DIM])
                    .sync_on(stream)
                    .unwrap(),
            ),
            scores: Some(
                api::zeros::<f32>(&[Q_HEADS, seq])
                    .sync_on(stream)
                    .expect("scores allocation failed"),
            ),
            probabilities: Some(
                api::zeros::<f32>(&[Q_HEADS, seq])
                    .sync_on(stream)
                    .expect("probabilities allocation failed"),
            ),
            context: Some(
                api::zeros::<f32>(&[Q_HEADS, HEAD_DIM])
                    .sync_on(stream)
                    .expect("context allocation failed"),
            ),
        }
    }

    fn read_outputs(buffers: &Buffers, stream: &Arc<Stream>) -> GqaDecodeResult {
        let scores_tensor = buffers.scores.as_ref().expect("scores unavailable");
        let scores_alias = unsafe { scores_tensor.into_shared_alias() };
        let scores = (&scores_alias)
            .to_host_vec()
            .sync_on(stream)
            .expect("score readback failed");
        let probabilities_tensor = buffers
            .probabilities
            .as_ref()
            .expect("probabilities unavailable");
        let probabilities_alias = unsafe { probabilities_tensor.into_shared_alias() };
        let probabilities = (&probabilities_alias)
            .to_host_vec()
            .sync_on(stream)
            .expect("probability readback failed");
        let context_tensor = buffers.context.as_ref().expect("context unavailable");
        let context_alias = unsafe { context_tensor.into_shared_alias() };
        let context = (&context_alias)
            .to_host_vec()
            .sync_on(stream)
            .expect("context readback failed");
        GqaDecodeResult {
            scores,
            probabilities,
            context,
        }
    }

    fn error_metrics(
        seq: usize,
        actual: &GqaDecodeResult,
        expected: &GqaDecodeResult,
    ) -> ErrorMetrics {
        let mut actual_flat = Vec::with_capacity(
            actual.scores.len() + actual.probabilities.len() + actual.context.len(),
        );
        actual_flat.extend_from_slice(&actual.scores);
        actual_flat.extend_from_slice(&actual.probabilities);
        actual_flat.extend_from_slice(&actual.context);
        let mut reference = Vec::with_capacity(actual_flat.len());
        reference.extend_from_slice(&expected.scores);
        reference.extend_from_slice(&expected.probabilities);
        reference.extend_from_slice(&expected.context);
        let absolute: Vec<f64> = actual_flat
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| f64::from((actual - expected).abs()))
            .collect();
        let max_absolute_error = absolute.iter().copied().fold(0.0, f64::max);
        let mean_absolute_error = absolute.iter().sum::<f64>() / absolute.len() as f64;
        let rmse = (absolute.iter().map(|value| value * value).sum::<f64>()
            / absolute.len() as f64)
            .sqrt();
        let max_relative_error = actual_flat
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| {
                f64::from((actual - expected).abs()) / f64::from(expected.abs()).max(1.0e-12)
            })
            .fold(0.0, f64::max);
        ErrorMetrics {
            max_absolute_error,
            mean_absolute_error,
            rmse,
            max_relative_error,
            max_probability_row_sum_error: max_probability_row_sum_error(
                seq,
                &actual.probabilities,
            ),
        }
    }

    fn comparison_metrics(left: &GqaDecodeResult, right: &GqaDecodeResult) -> ComparisonMetrics {
        let mut l =
            Vec::with_capacity(left.scores.len() + left.probabilities.len() + left.context.len());
        l.extend_from_slice(&left.scores);
        l.extend_from_slice(&left.probabilities);
        l.extend_from_slice(&left.context);
        let mut r = Vec::with_capacity(l.len());
        r.extend_from_slice(&right.scores);
        r.extend_from_slice(&right.probabilities);
        r.extend_from_slice(&right.context);
        assert_eq!(l.len(), r.len());
        let finite_left_count = l.iter().filter(|x| x.is_finite()).count();
        let finite_right_count = r.iter().filter(|x| x.is_finite()).count();
        let nan_count = l.iter().chain(&r).filter(|x| x.is_nan()).count();
        let positive_infinity_count = l.iter().chain(&r).filter(|x| **x == f32::INFINITY).count();
        let negative_infinity_count = l
            .iter()
            .chain(&r)
            .filter(|x| **x == f32::NEG_INFINITY)
            .count();
        let exact_mismatch_count = l
            .iter()
            .zip(&r)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let absolute: Vec<f64> = l
            .iter()
            .zip(&r)
            .map(|(a, b)| f64::from((a - b).abs()))
            .collect();
        let checksum = |values: &[f32]| {
            let mut h = 0xcbf29ce484222325u64;
            for value in values {
                for byte in value.to_bits().to_le_bytes() {
                    h ^= u64::from(byte);
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            format!("fnv1a64:{h:016x}")
        };
        ComparisonMetrics {
            element_count: l.len(),
            finite_left_count,
            finite_right_count,
            nan_count,
            positive_infinity_count,
            negative_infinity_count,
            exact_mismatch_count,
            max_absolute_error: absolute.iter().copied().fold(0.0, f64::max),
            max_relative_error: l
                .iter()
                .zip(&r)
                .map(|(a, b)| f64::from((a - b).abs()) / f64::from(b.abs()).max(1e-12))
                .fold(0.0, f64::max),
            mean_absolute_error: absolute.iter().sum::<f64>() / absolute.len() as f64,
            left_checksum: checksum(&l),
            right_checksum: checksum(&r),
        }
    }

    fn max_probability_row_sum_error(seq: usize, probabilities: &[f32]) -> f64 {
        (0..Q_HEADS)
            .map(|head| {
                let start = head * seq;
                f64::from((probabilities[start..start + seq].iter().sum::<f32>() - 1.0).abs())
            })
            .fold(0.0, f64::max)
    }

    fn check_cuda(result: sys::CUresult, operation: &str) {
        assert_eq!(
            result,
            sys::cudaError_enum_CUDA_SUCCESS,
            "{operation} failed with CUDA result {result}"
        );
    }

    fn model_block_table(blocks: usize) -> Vec<usize> {
        (0..blocks)
            .map(|logical| (logical * 17 + 11) % blocks)
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

    fn reconstruct_logical_kv(
        seq: usize,
        latent: &[f32],
        k_projection: &[f32],
        v_projection: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut k = vec![0.0f32; seq * KV_HEADS * HEAD_DIM];
        let mut v = vec![0.0f32; seq * KV_HEADS * HEAD_DIM];
        for token in 0..seq {
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

    fn logical_to_physical_kv(blocks: usize, logical: &[f32], table: &[usize]) -> Vec<f32> {
        let mut physical = vec![0.0f32; blocks * KV_HEADS * BLOCK_SIZE * HEAD_DIM];
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

    fn upload_f32(values: Vec<f32>, shape: &[usize], stream: &Arc<Stream>) -> Tensor<f32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(stream)
            .expect("f32 upload failed")
            .reshape(shape)
            .expect("f32 reshape failed")
    }

    fn upload_f16(values: Vec<f16>, shape: &[usize], stream: &Arc<Stream>) -> Tensor<f16> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(stream)
            .expect("f16 upload failed")
            .reshape(shape)
            .expect("f16 reshape failed")
    }

    fn upload_i32(values: Vec<i32>, shape: &[usize], stream: &Arc<Stream>) -> Tensor<i32> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(stream)
            .expect("i32 upload failed")
            .reshape(shape)
            .expect("i32 reshape failed")
    }
}
#[cfg(feature = "gpu-cutile")]
fn main() {
    gpu_impl::main();
}
#[cfg(not(feature = "gpu-cutile"))]
fn main() {
    panic!("requires gpu-cutile");
}
