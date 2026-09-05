// P1.5A diagnostic-only harness: isolated per-kernel CUDA-event timing for the
// unchanged P0/P1 A (full KV) and B (original latent) kernels.
//
// Timing boundary contains ONLY the target kernel launch. Process startup, JIT,
// allocations, input generation/upload, correctness readback and teardown are
// all outside the timed phases. Dependencies are prepared once, untimed, before
// isolated timing: softmax timing reads resident valid scores produced by one
// untimed score launch; context timing reads resident valid probabilities
// produced by one untimed score+softmax launch.

#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

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
    use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_1024;
    use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_2048;
    use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_4096;
    use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_8192;
    use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_16384;
    use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_32768;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_1024;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_2048;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_4096;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_8192;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_16384;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_32768;
    use serde::Serialize;

    const Q_HEADS: usize = 16;
    const KV_HEADS: usize = 4;
    const GROUP_SIZE: usize = 4;
    const HEAD_DIM: usize = 64;
    const LATENT_DIM: usize = 32;
    const BLOCK_SIZE: usize = 16;
    const RANDOM_SEED: u64 = 0;
    const SUPPORTED_SEQ: &[usize] = &[1024, 2048, 4096, 8192, 16384, 32768];

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
    enum KernelId {
        AScore,
        BScore,
        ASoftmax,
        BSoftmax,
        AContext,
        BContext,
        APipeline,
        BPipeline,
    }

    impl KernelId {
        fn label(self) -> &'static str {
            match self {
                KernelId::AScore => "A_score",
                KernelId::BScore => "B_score",
                KernelId::ASoftmax => "A_softmax",
                KernelId::BSoftmax => "B_softmax",
                KernelId::AContext => "A_context",
                KernelId::BContext => "B_context",
                KernelId::APipeline => "A_pipeline",
                KernelId::BPipeline => "B_pipeline",
            }
        }
    }

    const PHASE_ORDER: &[KernelId] = &[
        KernelId::AScore,
        KernelId::BScore,
        KernelId::ASoftmax,
        KernelId::BSoftmax,
        KernelId::AContext,
        KernelId::BContext,
        KernelId::APipeline,
        KernelId::BPipeline,
    ];

    type LaunchFn = fn(KernelId, &mut Buffers, &mut Buffers, &Inputs, &Arc<Stream>, usize);

    #[derive(Debug)]
    struct Args {
        seq: usize,
        warmup: usize,
        iterations: usize,
        output_dir: PathBuf,
    }

    impl Args {
        fn parse() -> Self {
            let mut args = std::env::args().skip(1);
            let mut parsed = Self {
                seq: 1024,
                warmup: 10,
                iterations: 50,
                output_dir: PathBuf::from("reports/p15_kernel_scaling"),
            };
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--seq" => {
                        parsed.seq = args
                            .next()
                            .expect("missing seq")
                            .parse()
                            .expect("invalid seq");
                    }
                    "--warmup" => {
                        parsed.warmup = args
                            .next()
                            .expect("missing warmup")
                            .parse()
                            .expect("invalid warmup");
                    }
                    "--iterations" => {
                        parsed.iterations = args
                            .next()
                            .expect("missing iterations")
                            .parse()
                            .expect("invalid iterations");
                    }
                    "--output-dir" => {
                        parsed.output_dir = PathBuf::from(args.next().expect("missing output dir"));
                    }
                    _ => panic!("unknown argument: {arg}"),
                }
            }
            assert!(
                SUPPORTED_SEQ.contains(&parsed.seq),
                "--seq must be one of {SUPPORTED_SEQ:?}"
            );
            assert!(parsed.iterations > 0, "iterations must be positive");
            parsed
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
        full_kv: ErrorMetrics,
        latent_original: ErrorMetrics,
        pass: bool,
    }

    #[derive(Debug, Serialize)]
    struct Sample {
        kernel: &'static str,
        phase: &'static str,
        iteration: usize,
        latency_ms: f64,
    }

    #[derive(Debug, Serialize)]
    struct KernelTiming {
        warmup: usize,
        iterations: usize,
        median_ms: f64,
        mean_ms: f64,
        min_ms: f64,
        max_ms: f64,
        samples_ms: Vec<f64>,
    }

    #[derive(Debug, Serialize)]
    struct Config {
        profile: &'static str,
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        head_dim: usize,
        latent_dim: usize,
        block_size: usize,
        seq_len: usize,
        logical_blocks: usize,
        num_physical_blocks: usize,
        storage_dtype: &'static str,
        arithmetic_dtype: &'static str,
        random_seed: u64,
        warmup_iterations: usize,
        measured_iterations: usize,
        timing_method: &'static str,
    }

    #[derive(Debug, Serialize)]
    struct Summary {
        seq: usize,
        config: Config,
        correctness: Correctness,
        kernels: std::collections::BTreeMap<&'static str, KernelTiming>,
    }

    struct EventPair {
        start: sys::CUevent,
        stop: sys::CUevent,
    }

    impl EventPair {
        fn new() -> Self {
            let mut start = std::ptr::null_mut();
            let mut stop = std::ptr::null_mut();
            unsafe {
                check_cuda(
                    sys::cuEventCreate(&mut start, sys::CUevent_flags_enum_CU_EVENT_DEFAULT),
                    "cuEventCreate(start)",
                );
                check_cuda(
                    sys::cuEventCreate(&mut stop, sys::CUevent_flags_enum_CU_EVENT_DEFAULT),
                    "cuEventCreate(stop)",
                );
            }
            Self { start, stop }
        }

        fn measure<F: FnOnce()>(&self, stream: &Arc<Stream>, operation: F) -> f64 {
            unsafe {
                stream.synchronize().expect("pre-timing stream sync failed");
                check_cuda(
                    sys::cuEventRecord(self.start, stream.cu_stream()),
                    "cuEventRecord(start)",
                );
                operation();
                check_cuda(
                    sys::cuEventRecord(self.stop, stream.cu_stream()),
                    "cuEventRecord(stop)",
                );
                check_cuda(
                    sys::cuEventSynchronize(self.stop),
                    "cuEventSynchronize(stop)",
                );
                let mut elapsed_ms = 0.0f32;
                check_cuda(
                    sys::cuEventElapsedTime_v2(&mut elapsed_ms, self.start, self.stop),
                    "cuEventElapsedTime_v2",
                );
                f64::from(elapsed_ms)
            }
        }
    }

    impl Drop for EventPair {
        fn drop(&mut self) {
            unsafe {
                let _ = sys::cuEventDestroy_v2(self.start);
                let _ = sys::cuEventDestroy_v2(self.stop);
            }
        }
    }

    pub fn main() {
        let args = Args::parse();
        let stream = with_default_device_policy(|policy| policy.next_stream())
            .expect("device context initialization failed")
            .expect("stream acquisition failed");
        stream
            .device()
            .bind_to_thread()
            .expect("failed to bind CUDA context");

        let launch: LaunchFn = match args.seq {
            1024 => launch_1024,
            2048 => launch_2048,
            4096 => launch_4096,
            8192 => launch_8192,
            16384 => launch_16384,
            32768 => launch_32768,
            other => panic!("unsupported sequence length {other}"),
        };

        run_length(&args, launch, &stream);
    }

    fn run_length(args: &Args, launch: LaunchFn, stream: &Arc<Stream>) {
        let seq = args.seq;
        let blocks = seq / BLOCK_SIZE;
        let (inputs, cpu_full, cpu_latent) = make_inputs(seq, blocks, stream);
        let mut full = make_buffers(seq, stream);
        let mut latent = make_buffers(seq, stream);

        // Dependency preparation + correctness (all untimed):
        // one full A pipeline and one full B pipeline run produce resident,
        // correct scores, probabilities and context tensors. The same run
        // validates outputs against the established CPU references.
        launch(
            KernelId::APipeline,
            &mut full,
            &mut latent,
            &inputs,
            stream,
            seq,
        );
        launch(
            KernelId::BPipeline,
            &mut full,
            &mut latent,
            &inputs,
            stream,
            seq,
        );
        unsafe { stream.synchronize().expect("dependency-prep sync failed") };
        let full_output = read_outputs(&full, stream);
        let latent_output = read_outputs(&latent, stream);
        let correctness = Correctness {
            full_kv: error_metrics(seq, &full_output, &cpu_full),
            latent_original: error_metrics(seq, &latent_output, &cpu_latent),
            pass: false,
        };
        let mut correctness = correctness;
        correctness.pass = correctness.full_kv.max_absolute_error <= 5.0e-3
            && correctness.latent_original.max_absolute_error <= 5.0e-3
            && correctness.full_kv.max_probability_row_sum_error <= 1.0e-4
            && correctness.latent_original.max_probability_row_sum_error <= 1.0e-4;
        assert!(
            correctness.pass,
            "correctness validation failed at seq {seq}"
        );

        // Explicit dependency refresh immediately before timing (untimed):
        // resident valid scores exist before softmax timing; resident valid
        // probabilities exist before context timing. Buffers already hold valid
        // tensors from the correctness pipelines; re-running keeps them fresh
        // and warms the JIT/caches identically for both paths.
        launch(
            KernelId::APipeline,
            &mut full,
            &mut latent,
            &inputs,
            stream,
            seq,
        );
        launch(
            KernelId::BPipeline,
            &mut full,
            &mut latent,
            &inputs,
            stream,
            seq,
        );
        unsafe { stream.synchronize().expect("pre-timing sync failed") };

        let events = EventPair::new();
        let mut kernels = std::collections::BTreeMap::new();
        for &kernel in PHASE_ORDER {
            let mut samples = Vec::with_capacity(args.iterations);
            for _ in 0..args.warmup {
                launch(kernel, &mut full, &mut latent, &inputs, stream, seq);
            }
            unsafe { stream.synchronize().expect("warmup sync failed") };
            for _ in 0..args.iterations {
                let latency_ms = events.measure(stream, || {
                    launch(kernel, &mut full, &mut latent, &inputs, stream, seq);
                });
                samples.push(latency_ms);
            }
            println!(
                "P15_KERNEL={} SEQ={} MEDIAN_MS={}",
                kernel.label(),
                seq,
                median(&samples)
            );
            std::io::stdout().flush().expect("flush failed");
            kernels.insert(
                kernel.label(),
                KernelTiming {
                    warmup: args.warmup,
                    iterations: args.iterations,
                    median_ms: median(&samples),
                    mean_ms: samples.iter().sum::<f64>() / samples.len() as f64,
                    min_ms: samples.iter().copied().fold(f64::INFINITY, f64::min),
                    max_ms: samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    samples_ms: samples,
                },
            );
        }

        let summary = Summary {
            seq,
            config: Config {
                profile: "model_small",
                q_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                group_size: GROUP_SIZE,
                head_dim: HEAD_DIM,
                latent_dim: LATENT_DIM,
                block_size: BLOCK_SIZE,
                seq_len: seq,
                logical_blocks: blocks,
                num_physical_blocks: blocks,
                storage_dtype: "fp16",
                arithmetic_dtype: "fp32",
                random_seed: RANDOM_SEED,
                warmup_iterations: args.warmup,
                measured_iterations: args.iterations,
                timing_method: "CUDA_EVENTS_AROUND_SINGLE_KERNEL_LAUNCH_ON_ONE_STREAM",
            },
            correctness,
            kernels,
        };

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_secs();
        fs::create_dir_all(&args.output_dir).expect("failed to create output directory");
        let summary_path = args
            .output_dir
            .join(format!("p15_summary_seq{seq}_{stamp}.json"));
        let samples_path = args
            .output_dir
            .join(format!("p15_samples_seq{seq}_{stamp}.jsonl"));
        {
            let file = File::create(&samples_path).expect("failed to create samples file");
            let mut writer = BufWriter::new(file);
            for (&kernel, timing) in &summary.kernels {
                for (iteration, &latency_ms) in timing.samples_ms.iter().enumerate() {
                    serde_json::to_writer(
                        &mut writer,
                        &Sample {
                            kernel,
                            phase: "measured",
                            iteration,
                            latency_ms,
                        },
                    )
                    .expect("failed to serialize sample");
                    writer.write_all(b"\n").expect("failed to write newline");
                }
            }
        }
        serde_json::to_writer_pretty(
            BufWriter::new(File::create(&summary_path).expect("failed to create summary")),
            &summary,
        )
        .expect("failed to write summary");

        println!("P15_SUMMARY_PATH={}", summary_path.display());
        println!("P15_SAMPLES_PATH={}", samples_path.display());
        println!("P15_KERNEL_SCALING_OK=1");
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    // Per-length launch dispatch. Generated as explicit functions because the
    // kernel modules are compile-time-specialized per sequence length. Kernel
    // logic, tile shapes and topology are NOT changed by this harness.
    macro_rules! define_launch {
        ($fn_name:ident, $full_scores:path, $full_softmax:path, $full_context:path,
         $latent_scores:path, $latent_softmax:path, $latent_context:path) => {
            fn $fn_name(
                kernel: KernelId,
                full: &mut Buffers,
                latent: &mut Buffers,
                inputs: &Inputs,
                stream: &Arc<Stream>,
                seq: usize,
            ) {
                match kernel {
                    KernelId::AScore => {
                        let out = full.scores.take().expect("missing A scores buffer");
                        let (part, _, _, _, _) = unsafe {
                            $full_scores(
                                out.partition([1, BLOCK_SIZE]),
                                &inputs.q,
                                &inputs.k_full,
                                &inputs.table,
                                &inputs.active,
                            )
                            .async_on(stream)
                            .expect("A score launch failed")
                        };
                        full.scores = Some(part.unpartition());
                    }
                    KernelId::ASoftmax => {
                        let scores = full.scores.take().expect("missing A scores buffer");
                        let out = full
                            .probabilities
                            .take()
                            .expect("missing A probabilities buffer");
                        let (part, _, _) = unsafe {
                            $full_softmax(out.partition([1, seq]), &scores, &inputs.active)
                                .async_on(stream)
                                .expect("A softmax launch failed")
                        };
                        full.probabilities = Some(part.unpartition());
                        full.scores = Some(scores);
                    }
                    KernelId::AContext => {
                        let probs = full
                            .probabilities
                            .take()
                            .expect("missing A probabilities buffer");
                        let out = full.context.take().expect("missing A context buffer");
                        let (part, _, _, _) = unsafe {
                            $full_context(
                                out.partition([1, HEAD_DIM]),
                                &probs,
                                &inputs.v_full,
                                &inputs.table,
                            )
                            .async_on(stream)
                            .expect("A context launch failed")
                        };
                        full.context = Some(part.unpartition());
                        full.probabilities = Some(probs);
                    }
                    KernelId::BScore => {
                        let out = latent.scores.take().expect("missing B scores buffer");
                        let (part, _, _, _, _, _) = unsafe {
                            $latent_scores(
                                out.partition([1, BLOCK_SIZE]),
                                &inputs.q,
                                &inputs.latent,
                                &inputs.table,
                                &inputs.active,
                                &inputs.k_projection,
                            )
                            .async_on(stream)
                            .expect("B score launch failed")
                        };
                        latent.scores = Some(part.unpartition());
                    }
                    KernelId::BSoftmax => {
                        let scores = latent.scores.take().expect("missing B scores buffer");
                        let out = latent
                            .probabilities
                            .take()
                            .expect("missing B probabilities buffer");
                        let (part, _, _) = unsafe {
                            $latent_softmax(out.partition([1, seq]), &scores, &inputs.active)
                                .async_on(stream)
                                .expect("B softmax launch failed")
                        };
                        latent.probabilities = Some(part.unpartition());
                        latent.scores = Some(scores);
                    }
                    KernelId::BContext => {
                        let probs = latent
                            .probabilities
                            .take()
                            .expect("missing B probabilities buffer");
                        let out = latent.context.take().expect("missing B context buffer");
                        let (part, _, _, _, _) = unsafe {
                            $latent_context(
                                out.partition([1, HEAD_DIM]),
                                &probs,
                                &inputs.latent,
                                &inputs.table,
                                &inputs.v_projection,
                            )
                            .async_on(stream)
                            .expect("B context launch failed")
                        };
                        latent.context = Some(part.unpartition());
                        latent.probabilities = Some(probs);
                    }
                    KernelId::APipeline => {
                        $fn_name(KernelId::AScore, full, latent, inputs, stream, seq);
                        $fn_name(KernelId::ASoftmax, full, latent, inputs, stream, seq);
                        $fn_name(KernelId::AContext, full, latent, inputs, stream, seq);
                    }
                    KernelId::BPipeline => {
                        $fn_name(KernelId::BScore, full, latent, inputs, stream, seq);
                        $fn_name(KernelId::BSoftmax, full, latent, inputs, stream, seq);
                        $fn_name(KernelId::BContext, full, latent, inputs, stream, seq);
                    }
                }
            }
        };
    }

    define_launch!(
        launch_1024,
        full_kv_baseline_kernel_1024::model_small_full_kv_scores_fp16_storage_1024,
        model_profile_kernel_1024::model_small_softmax_1024_runtime,
        full_kv_baseline_kernel_1024::model_small_full_kv_context_fp16_storage_1024,
        model_profile_kernel_1024::model_small_scores_fp16_storage_1024,
        model_profile_kernel_1024::model_small_softmax_1024_runtime,
        model_profile_kernel_1024::model_small_context_fp16_storage_1024
    );
    define_launch!(
        launch_2048,
        full_kv_baseline_kernel_2048::model_small_full_kv_scores_fp16_storage_2048,
        model_profile_kernel_2048::model_small_softmax_2048_runtime,
        full_kv_baseline_kernel_2048::model_small_full_kv_context_fp16_storage_2048,
        model_profile_kernel_2048::model_small_scores_fp16_storage_2048,
        model_profile_kernel_2048::model_small_softmax_2048_runtime,
        model_profile_kernel_2048::model_small_context_fp16_storage_2048
    );
    define_launch!(
        launch_4096,
        full_kv_baseline_kernel_4096::model_small_full_kv_scores_fp16_storage_4096,
        model_profile_kernel_4096::model_small_softmax_4096_runtime,
        full_kv_baseline_kernel_4096::model_small_full_kv_context_fp16_storage_4096,
        model_profile_kernel_4096::model_small_scores_fp16_storage_4096,
        model_profile_kernel_4096::model_small_softmax_4096_runtime,
        model_profile_kernel_4096::model_small_context_fp16_storage_4096
    );
    define_launch!(
        launch_8192,
        full_kv_baseline_kernel_8192::model_small_full_kv_scores_fp16_storage_8192,
        model_profile_kernel_8192::model_small_softmax_8192_runtime,
        full_kv_baseline_kernel_8192::model_small_full_kv_context_fp16_storage_8192,
        model_profile_kernel_8192::model_small_scores_fp16_storage_8192,
        model_profile_kernel_8192::model_small_softmax_8192_runtime,
        model_profile_kernel_8192::model_small_context_fp16_storage_8192
    );
    define_launch!(
        launch_16384,
        full_kv_baseline_kernel_16384::model_small_full_kv_scores_fp16_storage_16384,
        model_profile_kernel_16384::model_small_softmax_16384_runtime,
        full_kv_baseline_kernel_16384::model_small_full_kv_context_fp16_storage_16384,
        model_profile_kernel_16384::model_small_scores_fp16_storage_16384,
        model_profile_kernel_16384::model_small_softmax_16384_runtime,
        model_profile_kernel_16384::model_small_context_fp16_storage_16384
    );
    define_launch!(
        launch_32768,
        full_kv_baseline_kernel_32768::model_small_full_kv_scores_fp16_storage_32768,
        model_profile_kernel_32768::model_small_softmax_32768_runtime,
        full_kv_baseline_kernel_32768::model_small_full_kv_context_fp16_storage_32768,
        model_profile_kernel_32768::model_small_scores_fp16_storage_32768,
        model_profile_kernel_32768::model_small_softmax_32768_runtime,
        model_profile_kernel_32768::model_small_context_fp16_storage_32768
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
