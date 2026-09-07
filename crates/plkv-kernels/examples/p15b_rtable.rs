#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use cutile::api;
    use cutile::cuda_async::device_context::with_default_device_policy;
    use cutile::cuda_core::{sys, Stream};
    use cutile::half::f16;
    use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
    use cutile::tile_kernel::DeviceOp;
    use plkv_core::{
        direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum,
        paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum, quantize_f32_to_f16_storage,
        GqaDecodeResult,
    };
    use plkv_kernels::cutile::p15b_rtable_kernels::*;
    use plkv_kernels::cutile::p1_sequence_kernels::*;
    use serde::Serialize;
    use std::collections::BTreeMap;
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    const Q_HEADS: usize = 16;
    const KV_HEADS: usize = 4;
    const GROUP_SIZE: usize = 4;
    const HEAD_DIM: usize = 64;
    const LATENT_DIM: usize = 32;
    const BLOCK_SIZE: usize = 16;
    #[derive(Clone, Copy, Debug, Serialize)]
    enum K {
        A0Score,
        A1Score,
        A0Context,
        A1Context,
        B0Score,
        B1Score,
        B0Context,
        B1Context,
        A0Pipeline,
        A1Pipeline,
        B0Pipeline,
        B1Pipeline,
    }
    impl K {
        fn label(self) -> &'static str {
            match self {
                K::A0Score => "A0_score",
                K::A1Score => "A1_score",
                K::A0Context => "A0_context",
                K::A1Context => "A1_context",
                K::B0Score => "B0_score",
                K::B1Score => "B1_score",
                K::B0Context => "B0_context",
                K::B1Context => "B1_context",
                K::A0Pipeline => "A0_pipeline",
                K::A1Pipeline => "A1_pipeline",
                K::B0Pipeline => "B0_pipeline",
                K::B1Pipeline => "B1_pipeline",
            }
        }
    }
    const PHASES: &[K] = &[
        K::A0Score,
        K::A1Score,
        K::A0Context,
        K::A1Context,
        K::B0Score,
        K::B1Score,
        K::B0Context,
        K::B1Context,
        K::A0Pipeline,
        K::A1Pipeline,
        K::B0Pipeline,
        K::B1Pipeline,
    ];
    type Launch =
        fn(K, &mut Buffers, &mut Buffers, &mut Buffers, &mut Buffers, &Inputs, &Arc<Stream>, usize);
    #[derive(Debug)]
    struct Args {
        seq: usize,
        warmup: usize,
        iterations: usize,
        output_dir: PathBuf,
    }
    impl Args {
        fn parse() -> Self {
            let mut a = std::env::args().skip(1);
            let mut x = Self {
                seq: 1024,
                warmup: 10,
                iterations: 50,
                output_dir: PathBuf::from("reports/p15b_rtable"),
            };
            while let Some(v) = a.next() {
                match v.as_str() {
                    "--seq" => x.seq = a.next().unwrap().parse().unwrap(),
                    "--warmup" => x.warmup = a.next().unwrap().parse().unwrap(),
                    "--iterations" => x.iterations = a.next().unwrap().parse().unwrap(),
                    "--output-dir" => x.output_dir = PathBuf::from(a.next().unwrap()),
                    _ => panic!("unknown argument: {v}"),
                }
            }
            assert!([1024, 2048, 4096, 8192, 16384, 32768].contains(&x.seq));
            assert!(x.iterations > 0);
            x
        }
    }
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
    struct Correctness {
        a0_cpu: ErrorMetrics,
        a1_cpu: ErrorMetrics,
        b0_cpu: ErrorMetrics,
        b1_cpu: ErrorMetrics,
        a1_a0: ErrorMetrics,
        b1_b0: ErrorMetrics,
        pass: bool,
    }
    #[derive(Debug, Serialize)]
    struct Timing {
        warmup: usize,
        iterations: usize,
        median_ms: f64,
        mean_ms: f64,
        min_ms: f64,
        max_ms: f64,
        samples_ms: Vec<f64>,
    }
    #[derive(Debug, Serialize)]
    struct Summary {
        seq: usize,
        correctness: Correctness,
        kernels: BTreeMap<&'static str, Timing>,
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
    pub fn main() {
        let a = Args::parse();
        let st = with_default_device_policy(|p| p.next_stream())
            .unwrap()
            .unwrap();
        st.device().bind_to_thread().unwrap();
        let launch: Launch = match a.seq {
            1024 => launch_1024,
            2048 => launch_2048,
            4096 => launch_4096,
            8192 => launch_8192,
            16384 => launch_16384,
            32768 => launch_32768,
            _ => unreachable!(),
        };
        run(&a, launch, &st)
    }
    fn run(a: &Args, launch: Launch, st: &Arc<Stream>) {
        let n = a.seq;
        let blocks = n / BLOCK_SIZE;
        let (inp, cf, cb) = make_inputs(n, blocks, st);
        let (mut a0, mut a1, mut b0, mut b1) = (
            make_buffers(n, st),
            make_buffers(n, st),
            make_buffers(n, st),
            make_buffers(n, st),
        );
        launch(
            K::A0Pipeline,
            &mut a0,
            &mut a1,
            &mut b0,
            &mut b1,
            &inp,
            st,
            n,
        );
        launch(
            K::A1Pipeline,
            &mut a0,
            &mut a1,
            &mut b0,
            &mut b1,
            &inp,
            st,
            n,
        );
        launch(
            K::B0Pipeline,
            &mut a0,
            &mut a1,
            &mut b0,
            &mut b1,
            &inp,
            st,
            n,
        );
        launch(
            K::B1Pipeline,
            &mut a0,
            &mut a1,
            &mut b0,
            &mut b1,
            &inp,
            st,
            n,
        );
        unsafe { st.synchronize().unwrap() };
        let (o0, o1, o2, o3) = (
            read_outputs(&a0, st),
            read_outputs(&a1, st),
            read_outputs(&b0, st),
            read_outputs(&b1, st),
        );
        let mut c = Correctness {
            a0_cpu: error_metrics(n, &o0, &cf),
            a1_cpu: error_metrics(n, &o1, &cf),
            b0_cpu: error_metrics(n, &o2, &cb),
            b1_cpu: error_metrics(n, &o3, &cb),
            a1_a0: error_metrics(n, &o1, &o0),
            b1_b0: error_metrics(n, &o3, &o2),
            pass: false,
        };
        c.pass = [&c.a0_cpu, &c.a1_cpu, &c.b0_cpu, &c.b1_cpu]
            .iter()
            .all(|m| m.max_absolute_error <= 5e-3 && m.max_probability_row_sum_error <= 1e-4)
            && c.a1_a0.max_absolute_error <= 1e-6
            && c.b1_b0.max_absolute_error <= 1e-6;
        assert!(c.pass, "correctness failure at {n}: {c:?}");
        let ev = Events::new();
        let mut out = BTreeMap::new();
        for &k in PHASES {
            let mut s = Vec::with_capacity(a.iterations);
            for _ in 0..a.warmup {
                launch(k, &mut a0, &mut a1, &mut b0, &mut b1, &inp, st, n)
            }
            unsafe { st.synchronize().unwrap() }
            for _ in 0..a.iterations {
                s.push(ev.measure(st, || {
                    launch(k, &mut a0, &mut a1, &mut b0, &mut b1, &inp, st, n)
                }))
            }
            let mut z = s.clone();
            z.sort_by(f64::total_cmp);
            let med = if z.len() % 2 == 0 {
                (z[z.len() / 2 - 1] + z[z.len() / 2]) / 2.
            } else {
                z[z.len() / 2]
            };
            out.insert(
                k.label(),
                Timing {
                    warmup: a.warmup,
                    iterations: a.iterations,
                    median_ms: med,
                    mean_ms: s.iter().sum::<f64>() / s.len() as f64,
                    min_ms: *z.first().unwrap(),
                    max_ms: *z.last().unwrap(),
                    samples_ms: s,
                },
            );
        }
        let summary = Summary {
            seq: n,
            correctness: c,
            kernels: out,
        };
        fs::create_dir_all(&a.output_dir).unwrap();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let p = a.output_dir.join(format!("summary_seq{n}_{stamp}.json"));
        serde_json::to_writer_pretty(File::create(&p).unwrap(), &summary).unwrap();
        let raw = p.with_extension("jsonl");
        let mut writer = BufWriter::new(File::create(&raw).unwrap());
        for (name, timing) in &summary.kernels {
            for (iteration, &latency_ms) in timing.samples_ms.iter().enumerate() {
                serde_json::to_writer(
                    &mut writer,
                    &serde_json::json!({"kernel": name, "iteration": iteration, "latency_ms": latency_ms}),
                ).unwrap();
                writer.write_all(b"\n").unwrap();
            }
        }
        println!("P15B_SUMMARY_PATH={}", p.display());
        println!("P15B_SAMPLES_PATH={}", raw.display());
        println!("P15B_OK=1");
    }
    macro_rules! define_launch {
        ($name:ident, $a0s:path, $a0c:path, $b0s:path, $b0c:path,
         $a1s:path, $a1c:path, $b1s:path, $b1c:path, $sm:path) => {
            fn $name(
                k: K,
                a0: &mut Buffers,
                a1: &mut Buffers,
                b0: &mut Buffers,
                b1: &mut Buffers,
                i: &Inputs,
                st: &Arc<Stream>,
                n: usize,
            ) {
                match k {
                    K::A0Score | K::A1Score | K::B0Score | K::B1Score => {
                        let (b, is_a, repaired) = match k {
                            K::A0Score => (a0, true, false),
                            K::A1Score => (a1, true, true),
                            K::B0Score => (b0, false, false),
                            K::B1Score => (b1, false, true),
                            _ => unreachable!(),
                        };
                        let out = b.scores.take().unwrap();
                        if is_a && !repaired {
                            let (p, _, _, _, _) = unsafe {
                                $a0s(
                                    out.partition([1, BLOCK_SIZE]),
                                    &i.q,
                                    &i.k_full,
                                    &i.table,
                                    &i.active,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            b.scores = Some(p.unpartition());
                        } else if is_a {
                            let (p, _, _, _, _) = unsafe {
                                $a1s(
                                    out.partition([1, BLOCK_SIZE]),
                                    &i.q,
                                    &i.k_full,
                                    &i.table,
                                    &i.active,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            b.scores = Some(p.unpartition());
                        } else if !repaired {
                            let (p, _, _, _, _, _) = unsafe {
                                $b0s(
                                    out.partition([1, BLOCK_SIZE]),
                                    &i.q,
                                    &i.latent,
                                    &i.table,
                                    &i.active,
                                    &i.k_projection,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            b.scores = Some(p.unpartition());
                        } else {
                            let (p, _, _, _, _, _) = unsafe {
                                $b1s(
                                    out.partition([1, BLOCK_SIZE]),
                                    &i.q,
                                    &i.latent,
                                    &i.table,
                                    &i.active,
                                    &i.k_projection,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            b.scores = Some(p.unpartition());
                        }
                    }
                    K::A0Context | K::A1Context => {
                        let b = if matches!(k, K::A0Context) { a0 } else { a1 };
                        let out = b.context.take().unwrap();
                        let p = b.probabilities.take().unwrap();
                        if matches!(k, K::A0Context) {
                            let (c, _, _, _) = unsafe {
                                $a0c(out.partition([1, HEAD_DIM]), &p, &i.v_full, &i.table)
                                    .async_on(st)
                                    .unwrap()
                            };
                            b.context = Some(c.unpartition());
                        } else {
                            let (c, _, _, _) = unsafe {
                                $a1c(out.partition([1, HEAD_DIM]), &p, &i.v_full, &i.table)
                                    .async_on(st)
                                    .unwrap()
                            };
                            b.context = Some(c.unpartition());
                        }
                        b.probabilities = Some(p);
                    }
                    K::B0Context | K::B1Context => {
                        let b = if matches!(k, K::B0Context) { b0 } else { b1 };
                        let out = b.context.take().unwrap();
                        let p = b.probabilities.take().unwrap();
                        if matches!(k, K::B0Context) {
                            let (c, _, _, _, _) = unsafe {
                                $b0c(
                                    out.partition([1, HEAD_DIM]),
                                    &p,
                                    &i.latent,
                                    &i.table,
                                    &i.v_projection,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            b.context = Some(c.unpartition());
                        } else {
                            let (c, _, _, _, _) = unsafe {
                                $b1c(
                                    out.partition([1, HEAD_DIM]),
                                    &p,
                                    &i.latent,
                                    &i.table,
                                    &i.v_projection,
                                )
                                .async_on(st)
                                .unwrap()
                            };
                            b.context = Some(c.unpartition());
                        }
                        b.probabilities = Some(p);
                    }
                    K::A0Pipeline => {
                        $name(K::A0Score, a0, a1, b0, b1, i, st, n);
                        soft_launch!(a0, i, st, $sm, n);
                        $name(K::A0Context, a0, a1, b0, b1, i, st, n)
                    }
                    K::A1Pipeline => {
                        $name(K::A1Score, a0, a1, b0, b1, i, st, n);
                        soft_launch!(a1, i, st, $sm, n);
                        $name(K::A1Context, a0, a1, b0, b1, i, st, n)
                    }
                    K::B0Pipeline => {
                        $name(K::B0Score, a0, a1, b0, b1, i, st, n);
                        soft_launch!(b0, i, st, $sm, n);
                        $name(K::B0Context, a0, a1, b0, b1, i, st, n)
                    }
                    K::B1Pipeline => {
                        $name(K::B1Score, a0, a1, b0, b1, i, st, n);
                        soft_launch!(b1, i, st, $sm, n);
                        $name(K::B1Context, a0, a1, b0, b1, i, st, n)
                    }
                }
            }
        };
    }
    macro_rules! soft_launch {
        ($b:expr,$i:expr,$st:expr,$f:path,$n:expr) => {{
            let s = $b.scores.take().unwrap();
            let o = $b.probabilities.take().unwrap();
            let (p, _, _) = unsafe {
                $f(o.partition([1, $n]), &s, &$i.active)
                    .async_on($st)
                    .unwrap()
            };
            $b.scores = Some(s);
            $b.probabilities = Some(p.unpartition());
        }};
    }
    define_launch!(
        launch_1024,
        full_kv_baseline_kernel_1024::model_small_full_kv_scores_fp16_storage_1024,
        full_kv_baseline_kernel_1024::model_small_full_kv_context_fp16_storage_1024,
        model_profile_kernel_1024::model_small_scores_fp16_storage_1024,
        model_profile_kernel_1024::model_small_context_fp16_storage_1024,
        p15b_full_kv_baseline_kernel_1024::model_small_full_kv_scores_fp16_storage_rtable_1024,
        p15b_full_kv_baseline_kernel_1024::model_small_full_kv_context_fp16_storage_rtable_1024,
        p15b_model_profile_kernel_1024::model_small_scores_fp16_storage_rtable_1024,
        p15b_model_profile_kernel_1024::model_small_context_fp16_storage_rtable_1024,
        model_profile_kernel_1024::model_small_softmax_1024_runtime
    );
    define_launch!(
        launch_2048,
        full_kv_baseline_kernel_2048::model_small_full_kv_scores_fp16_storage_2048,
        full_kv_baseline_kernel_2048::model_small_full_kv_context_fp16_storage_2048,
        model_profile_kernel_2048::model_small_scores_fp16_storage_2048,
        model_profile_kernel_2048::model_small_context_fp16_storage_2048,
        p15b_full_kv_baseline_kernel_2048::model_small_full_kv_scores_fp16_storage_rtable_2048,
        p15b_full_kv_baseline_kernel_2048::model_small_full_kv_context_fp16_storage_rtable_2048,
        p15b_model_profile_kernel_2048::model_small_scores_fp16_storage_rtable_2048,
        p15b_model_profile_kernel_2048::model_small_context_fp16_storage_rtable_2048,
        model_profile_kernel_2048::model_small_softmax_2048_runtime
    );
    define_launch!(
        launch_4096,
        full_kv_baseline_kernel_4096::model_small_full_kv_scores_fp16_storage_4096,
        full_kv_baseline_kernel_4096::model_small_full_kv_context_fp16_storage_4096,
        model_profile_kernel_4096::model_small_scores_fp16_storage_4096,
        model_profile_kernel_4096::model_small_context_fp16_storage_4096,
        p15b_full_kv_baseline_kernel_4096::model_small_full_kv_scores_fp16_storage_rtable_4096,
        p15b_full_kv_baseline_kernel_4096::model_small_full_kv_context_fp16_storage_rtable_4096,
        p15b_model_profile_kernel_4096::model_small_scores_fp16_storage_rtable_4096,
        p15b_model_profile_kernel_4096::model_small_context_fp16_storage_rtable_4096,
        model_profile_kernel_4096::model_small_softmax_4096_runtime
    );
    define_launch!(
        launch_8192,
        full_kv_baseline_kernel_8192::model_small_full_kv_scores_fp16_storage_8192,
        full_kv_baseline_kernel_8192::model_small_full_kv_context_fp16_storage_8192,
        model_profile_kernel_8192::model_small_scores_fp16_storage_8192,
        model_profile_kernel_8192::model_small_context_fp16_storage_8192,
        p15b_full_kv_baseline_kernel_8192::model_small_full_kv_scores_fp16_storage_rtable_8192,
        p15b_full_kv_baseline_kernel_8192::model_small_full_kv_context_fp16_storage_rtable_8192,
        p15b_model_profile_kernel_8192::model_small_scores_fp16_storage_rtable_8192,
        p15b_model_profile_kernel_8192::model_small_context_fp16_storage_rtable_8192,
        model_profile_kernel_8192::model_small_softmax_8192_runtime
    );
    define_launch!(
        launch_16384,
        full_kv_baseline_kernel_16384::model_small_full_kv_scores_fp16_storage_16384,
        full_kv_baseline_kernel_16384::model_small_full_kv_context_fp16_storage_16384,
        model_profile_kernel_16384::model_small_scores_fp16_storage_16384,
        model_profile_kernel_16384::model_small_context_fp16_storage_16384,
        p15b_full_kv_baseline_kernel_16384::model_small_full_kv_scores_fp16_storage_rtable_16384,
        p15b_full_kv_baseline_kernel_16384::model_small_full_kv_context_fp16_storage_rtable_16384,
        p15b_model_profile_kernel_16384::model_small_scores_fp16_storage_rtable_16384,
        p15b_model_profile_kernel_16384::model_small_context_fp16_storage_rtable_16384,
        model_profile_kernel_16384::model_small_softmax_16384_runtime
    );
    define_launch!(
        launch_32768,
        full_kv_baseline_kernel_32768::model_small_full_kv_scores_fp16_storage_32768,
        full_kv_baseline_kernel_32768::model_small_full_kv_context_fp16_storage_32768,
        model_profile_kernel_32768::model_small_scores_fp16_storage_32768,
        model_profile_kernel_32768::model_small_context_fp16_storage_32768,
        p15b_full_kv_baseline_kernel_32768::model_small_full_kv_scores_fp16_storage_rtable_32768,
        p15b_full_kv_baseline_kernel_32768::model_small_full_kv_context_fp16_storage_rtable_32768,
        p15b_model_profile_kernel_32768::model_small_scores_fp16_storage_rtable_32768,
        p15b_model_profile_kernel_32768::model_small_context_fp16_storage_rtable_32768,
        model_profile_kernel_32768::model_small_softmax_32768_runtime
    );
    fn make_inputs(
        seq: usize,
        blocks: usize,
        stream: &Arc<Stream>,
    ) -> (Inputs, GqaDecodeResult, GqaDecodeResult) {
        let block_table = model_block_table(blocks);
        let q = deterministic_values(Q_HEADS * HEAD_DIM, 0.011, -0.4);
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
            seq,
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
            seq,
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
            active: upload_i32(vec![seq as i32], &[1], stream),
        };
        (inputs, cpu_full, cpu_latent)
    }

    fn make_buffers(seq: usize, stream: &Arc<Stream>) -> Buffers {
        Buffers {
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
fn main() {}
