#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use std::fs::{self, File};
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    use plkv_kernels::cutile::p1_sequence_kernels::full_kv_baseline_kernel_131072;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_kernel_131072;
    use plkv_kernels::cutile::p1_sequence_kernels::model_profile_preprojected_kernel_131072;
    use serde::Serialize;

    const Q_HEADS: usize = 16;
    const KV_HEADS: usize = 4;
    const GROUP_SIZE: usize = 4;
    const HEAD_DIM: usize = 64;
    const LATENT_DIM: usize = 32;
    const BLOCK_SIZE: usize = 16;
    const MAX_SEQ_LEN: usize = 131072;
    const ACTIVE_SEQ_LEN: usize = 131072;
    const NUM_PHYSICAL_BLOCKS: usize = 8192;
    const RANDOM_SEED: u64 = 0;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Selection {
        All,
        FullLatent,
        Full,
        Latent,
        LatentPreprojected,
    }

    impl Selection {
        fn parse(value: &str) -> Self {
            match value {
                "all" => Self::All,
                "full-latent" => Self::FullLatent,
                "full" => Self::Full,
                "latent" => Self::Latent,
                "latent-preprojected" => Self::LatentPreprojected,
                _ => panic!(
                    "--variant must be one of: all, full-latent, full, latent, latent-preprojected"
                ),
            }
        }
    }

    #[derive(Debug)]
    struct Args {
        selection: Selection,
        warmup: usize,
        iterations: usize,
        output_dir: PathBuf,
    }

    impl Args {
        fn parse() -> Self {
            let mut args = std::env::args().skip(1);
            let mut parsed = Self {
                selection: Selection::All,
                warmup: 20,
                iterations: 100,
                output_dir: PathBuf::from("reports/p0_gpu_baseline"),
            };
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--variant" => {
                        parsed.selection = Selection::parse(&args.next().expect("missing variant"));
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

    #[derive(Debug, Serialize)]
    struct PythonOracleInputs {
        q: Vec<f32>,
        latent_storage_f16_as_f32: Vec<f32>,
        block_table: Vec<usize>,
        k_projection: Vec<f32>,
        v_projection: Vec<f32>,
    }

    #[derive(Debug, Serialize)]
    struct PythonOracleCase<'a> {
        profile: &'static str,
        q_heads: usize,
        kv_heads: usize,
        group_size: usize,
        head_dim: usize,
        latent_dim: usize,
        block_size: usize,
        max_seq_len: usize,
        active_seq_len: usize,
        num_physical_blocks: usize,
        inputs: &'a PythonOracleInputs,
        scores: &'a [f32],
        probabilities: &'a [f32],
        context: &'a [f32],
    }

    struct Buffers {
        scores: Option<Tensor<f32>>,
        probabilities: Option<Tensor<f32>>,
        context: Option<Tensor<f32>>,
    }

    struct PreprojectedBuffers {
        projected_query: Option<Tensor<f32>>,
        attention: Buffers,
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
        latent_preprojected_vs_rust_cpu: ErrorMetrics,
        latent_preprojected_vs_original_gpu: ErrorMetrics,
    }

    #[derive(Debug, Serialize)]
    struct Sample<'a> {
        variant: &'a str,
        timing: &'a str,
        iteration: usize,
        latency_ms: f64,
    }

    #[derive(Debug, Serialize)]
    struct Stats {
        n: usize,
        min_ms: f64,
        mean_ms: f64,
        median_ms: f64,
        standard_deviation_ms: f64,
        p5_ms: f64,
        p50_ms: f64,
        p95_ms: f64,
        p99_ms: f64,
        max_ms: f64,
    }

    #[derive(Debug, Serialize)]
    struct VariantSummary {
        kernel_count: usize,
        persistent_bytes_per_token: usize,
        gpu_execution: Stats,
        host_synchronized: Stats,
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
        max_seq_len: usize,
        active_seq_len: usize,
        num_physical_blocks: usize,
        storage_dtype: &'static str,
        arithmetic_dtype: &'static str,
        random_seed: u64,
        input_generation: &'static str,
        warmup_iterations: usize,
        measured_iterations_per_timing_kind: usize,
    }

    #[derive(Debug, Serialize)]
    struct Summary {
        timing_method_gpu: &'static str,
        timing_method_host: &'static str,
        config: Config,
        correctness: Correctness,
        full_kv: Option<VariantSummary>,
        latent_original: Option<VariantSummary>,
        latent_preprojected: Option<VariantSummary>,
        latent_original_over_full_kv_median: Option<f64>,
        latent_original_relative_overhead: Option<f64>,
        latent_preprojected_over_full_kv_median: Option<f64>,
        latent_preprojected_over_original_median: Option<f64>,
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

        let (inputs, cpu_full, cpu_latent, python_oracle_inputs) = make_inputs(&stream);
        let mut full_buffers = make_buffers(&stream);
        let mut latent_buffers = make_buffers(&stream);
        let mut preprojected_buffers = make_preprojected_buffers(&stream);

        enqueue_full(&mut full_buffers, &inputs, &stream);
        enqueue_latent(&mut latent_buffers, &inputs, &stream);
        enqueue_latent_preprojected(&mut preprojected_buffers, &inputs, &stream);
        unsafe {
            stream
                .synchronize()
                .expect("correctness synchronization failed")
        };
        let full_output = read_outputs(&full_buffers, &stream);
        let latent_output = read_outputs(&latent_buffers, &stream);
        let preprojected_output = read_outputs(&preprojected_buffers.attention, &stream);
        let correctness = Correctness {
            full_kv: error_metrics(&full_output, &cpu_full),
            latent_original: error_metrics(&latent_output, &cpu_latent),
            latent_preprojected_vs_rust_cpu: error_metrics(&preprojected_output, &cpu_latent),
            latent_preprojected_vs_original_gpu: error_metrics(
                &preprojected_output,
                &latent_output,
            ),
        };
        validate_correctness(&correctness);

        for iteration in 0..args.warmup {
            enqueue_selected(
                args.selection,
                iteration,
                &mut full_buffers,
                &mut latent_buffers,
                &mut preprojected_buffers,
                &inputs,
                &stream,
            );
        }
        unsafe { stream.synchronize().expect("warmup synchronization failed") };

        let event_pair = EventPair::new();
        let mut full_gpu = Vec::with_capacity(args.iterations);
        let mut latent_gpu = Vec::with_capacity(args.iterations);
        let mut preprojected_gpu = Vec::with_capacity(args.iterations);
        print_phase_marker("GPU_TIMING_PHASE_START_UNIX_MS");
        for iteration in 0..args.iterations {
            measure_selected_gpu(
                args.selection,
                iteration,
                &event_pair,
                &mut full_gpu,
                &mut latent_gpu,
                &mut preprojected_gpu,
                &mut full_buffers,
                &mut latent_buffers,
                &mut preprojected_buffers,
                &inputs,
                &stream,
            );
        }
        print_phase_marker("GPU_TIMING_PHASE_END_UNIX_MS");

        let mut full_host = Vec::with_capacity(args.iterations);
        let mut latent_host = Vec::with_capacity(args.iterations);
        let mut preprojected_host = Vec::with_capacity(args.iterations);
        print_phase_marker("HOST_TIMING_PHASE_START_UNIX_MS");
        for iteration in 0..args.iterations {
            measure_selected_host(
                args.selection,
                iteration,
                &mut full_host,
                &mut latent_host,
                &mut preprojected_host,
                &mut full_buffers,
                &mut latent_buffers,
                &mut preprojected_buffers,
                &inputs,
                &stream,
            );
        }
        print_phase_marker("HOST_TIMING_PHASE_END_UNIX_MS");

        let full_summary = (!full_gpu.is_empty()).then(|| VariantSummary {
            kernel_count: 3,
            persistent_bytes_per_token: KV_HEADS * HEAD_DIM * 2 * 2,
            gpu_execution: stats(&full_gpu),
            host_synchronized: stats(&full_host),
        });
        let latent_summary = (!latent_gpu.is_empty()).then(|| VariantSummary {
            kernel_count: 3,
            persistent_bytes_per_token: LATENT_DIM * 2,
            gpu_execution: stats(&latent_gpu),
            host_synchronized: stats(&latent_host),
        });
        let preprojected_summary = (!preprojected_gpu.is_empty()).then(|| VariantSummary {
            kernel_count: 4,
            persistent_bytes_per_token: LATENT_DIM * 2,
            gpu_execution: stats(&preprojected_gpu),
            host_synchronized: stats(&preprojected_host),
        });
        let original_over_full = full_summary
            .as_ref()
            .zip(latent_summary.as_ref())
            .map(|(full, latent)| latent.gpu_execution.median_ms / full.gpu_execution.median_ms);
        let preprojected_over_full = full_summary
            .as_ref()
            .zip(preprojected_summary.as_ref())
            .map(|(full, optimized)| {
                optimized.gpu_execution.median_ms / full.gpu_execution.median_ms
            });
        let preprojected_over_original = latent_summary
            .as_ref()
            .zip(preprojected_summary.as_ref())
            .map(|(original, optimized)| {
                optimized.gpu_execution.median_ms / original.gpu_execution.median_ms
            });
        let summary = Summary {
            timing_method_gpu: "CUDA_EVENTS_ON_SINGLE_EXPLICIT_STREAM",
            timing_method_host: "STD_INSTANT_AROUND_ASYNC_LAUNCH_SEQUENCE_PLUS_STREAM_SYNCHRONIZE",
            config: Config {
                profile: "model_small",
                q_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                group_size: GROUP_SIZE,
                head_dim: HEAD_DIM,
                latent_dim: LATENT_DIM,
                block_size: BLOCK_SIZE,
                max_seq_len: MAX_SEQ_LEN,
                active_seq_len: ACTIVE_SEQ_LEN,
                num_physical_blocks: NUM_PHYSICAL_BLOCKS,
                storage_dtype: "fp16",
                arithmetic_dtype: "fp32",
                random_seed: RANDOM_SEED,
                input_generation: "existing deterministic_values formula; no RNG",
                warmup_iterations: args.warmup,
                measured_iterations_per_timing_kind: args.iterations,
            },
            correctness,
            full_kv: full_summary,
            latent_original: latent_summary,
            latent_preprojected: preprojected_summary,
            latent_original_over_full_kv_median: original_over_full,
            latent_original_relative_overhead: original_over_full.map(|value| value - 1.0),
            latent_preprojected_over_full_kv_median: preprojected_over_full,
            latent_preprojected_over_original_median: preprojected_over_original,
        };

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_secs();
        fs::create_dir_all(&args.output_dir).expect("failed to create output directory");
        let artifact_prefix = if args.selection == Selection::FullLatent {
            "ab"
        } else {
            "abc"
        };
        let samples_path = args
            .output_dir
            .join(format!("{artifact_prefix}_samples_{stamp}.jsonl"));
        let summary_path = args
            .output_dir
            .join(format!("{artifact_prefix}_summary_{stamp}.json"));
        let oracle_path = args
            .output_dir
            .join(format!("python_oracle_case_{stamp}.json"));
        write_samples(&samples_path, "full_kv", &full_gpu, &full_host);
        append_samples(&samples_path, "latent_original", &latent_gpu, &latent_host);
        append_samples(
            &samples_path,
            "latent_preprojected",
            &preprojected_gpu,
            &preprojected_host,
        );
        serde_json::to_writer_pretty(
            BufWriter::new(File::create(&summary_path).expect("failed to create summary")),
            &summary,
        )
        .expect("failed to write summary");
        let oracle_case = PythonOracleCase {
            profile: "model_small",
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            group_size: GROUP_SIZE,
            head_dim: HEAD_DIM,
            latent_dim: LATENT_DIM,
            block_size: BLOCK_SIZE,
            max_seq_len: MAX_SEQ_LEN,
            active_seq_len: ACTIVE_SEQ_LEN,
            num_physical_blocks: NUM_PHYSICAL_BLOCKS,
            inputs: &python_oracle_inputs,
            scores: &preprojected_output.scores,
            probabilities: &preprojected_output.probabilities,
            context: &preprojected_output.context,
        };
        serde_json::to_writer(
            BufWriter::new(File::create(&oracle_path).expect("failed to create oracle case")),
            &oracle_case,
        )
        .expect("failed to write oracle case");

        println!("PROFILE=model_small");
        println!("WARMUP_ITERATIONS={}", args.warmup);
        println!("MEASURED_ITERATIONS={}", args.iterations);
        println!("TIMING_GPU=CUDA_EVENTS_ON_SINGLE_EXPLICIT_STREAM");
        println!("TIMING_HOST=ASYNC_LAUNCH_SEQUENCE_PLUS_STREAM_SYNCHRONIZE");
        println!(
            "FULL_KV_GPU_MEDIAN_MS={}",
            summary
                .full_kv
                .as_ref()
                .map_or(f64::NAN, |s| s.gpu_execution.median_ms)
        );
        println!(
            "LATENT_ORIGINAL_GPU_MEDIAN_MS={}",
            summary
                .latent_original
                .as_ref()
                .map_or(f64::NAN, |s| s.gpu_execution.median_ms)
        );
        println!(
            "LATENT_ORIGINAL_OVER_FULL_KV_MEDIAN={}",
            original_over_full.unwrap_or(f64::NAN)
        );
        println!(
            "LATENT_PREPROJECTED_GPU_MEDIAN_MS={}",
            summary
                .latent_preprojected
                .as_ref()
                .map_or(f64::NAN, |s| s.gpu_execution.median_ms)
        );
        println!(
            "LATENT_PREPROJECTED_OVER_FULL_KV_MEDIAN={}",
            preprojected_over_full.unwrap_or(f64::NAN)
        );
        println!(
            "LATENT_PREPROJECTED_OVER_ORIGINAL_MEDIAN={}",
            preprojected_over_original.unwrap_or(f64::NAN)
        );
        println!("SAMPLES_PATH={}", samples_path.display());
        println!("SUMMARY_PATH={}", summary_path.display());
        println!("PYTHON_ORACLE_CASE_PATH={}", oracle_path.display());
        println!("P0_ABC_BENCHMARK_OK=1");
    }

    fn print_phase_marker(name: &str) {
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_millis();
        println!("{name}={unix_ms}");
        std::io::stdout()
            .flush()
            .expect("failed to flush phase marker");
    }

    fn make_inputs(
        stream: &Arc<Stream>,
    ) -> (Inputs, GqaDecodeResult, GqaDecodeResult, PythonOracleInputs) {
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
        let (logical_k, logical_v) =
            reconstruct_logical_kv(&logical_latent, &k_projection, &v_projection);
        let k_physical = logical_to_physical_kv(&logical_k, &block_table);
        let v_physical = logical_to_physical_kv(&logical_v, &block_table);
        let k_full_f16 = quantize_f32_to_f16_storage(&k_physical).expect("K quantization failed");
        let v_full_f16 = quantize_f32_to_f16_storage(&v_physical).expect("V quantization failed");

        let python_oracle_inputs = PythonOracleInputs {
            q: q.clone(),
            latent_storage_f16_as_f32: latent_f16.iter().copied().map(f32::from).collect(),
            block_table: block_table.clone(),
            k_projection: k_projection.clone(),
            v_projection: v_projection.clone(),
        };

        let cpu_full = paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum(
            &q,
            &k_full_f16,
            &v_full_f16,
            &block_table,
            Q_HEADS,
            KV_HEADS,
            MAX_SEQ_LEN,
            ACTIVE_SEQ_LEN,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_SIZE,
            NUM_PHYSICAL_BLOCKS,
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
            MAX_SEQ_LEN,
            ACTIVE_SEQ_LEN,
            LATENT_DIM,
            HEAD_DIM,
            GROUP_SIZE,
            BLOCK_SIZE,
            NUM_PHYSICAL_BLOCKS,
        )
        .expect("CPU latent reference failed");

        let inputs = Inputs {
            q: upload_f32(q, &[Q_HEADS, HEAD_DIM], stream),
            latent: upload_f16(
                latent_f16,
                &[NUM_PHYSICAL_BLOCKS * BLOCK_SIZE, LATENT_DIM],
                stream,
            ),
            k_projection: upload_f32(k_head_major, &[KV_HEADS * LATENT_DIM, HEAD_DIM], stream),
            v_projection: upload_f32(v_head_major, &[KV_HEADS * LATENT_DIM, HEAD_DIM], stream),
            k_full: upload_f16(
                k_full_f16,
                &[NUM_PHYSICAL_BLOCKS * KV_HEADS * BLOCK_SIZE, HEAD_DIM],
                stream,
            ),
            v_full: upload_f16(
                v_full_f16,
                &[NUM_PHYSICAL_BLOCKS * KV_HEADS * BLOCK_SIZE, HEAD_DIM],
                stream,
            ),
            table: upload_i32(
                block_table.iter().map(|&value| value as i32).collect(),
                &[NUM_PHYSICAL_BLOCKS],
                stream,
            ),
            active: upload_i32(vec![ACTIVE_SEQ_LEN as i32], &[1], stream),
        };
        (inputs, cpu_full, cpu_latent, python_oracle_inputs)
    }

    fn make_buffers(stream: &Arc<Stream>) -> Buffers {
        Buffers {
            scores: Some(
                api::zeros::<f32>(&[Q_HEADS, MAX_SEQ_LEN])
                    .sync_on(stream)
                    .expect("scores allocation failed"),
            ),
            probabilities: Some(
                api::zeros::<f32>(&[Q_HEADS, MAX_SEQ_LEN])
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

    fn make_preprojected_buffers(stream: &Arc<Stream>) -> PreprojectedBuffers {
        PreprojectedBuffers {
            projected_query: Some(
                api::zeros::<f32>(&[Q_HEADS, LATENT_DIM])
                    .sync_on(stream)
                    .expect("projected-query allocation failed"),
            ),
            attention: make_buffers(stream),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_selected(
        selection: Selection,
        iteration: usize,
        full: &mut Buffers,
        latent: &mut Buffers,
        preprojected: &mut PreprojectedBuffers,
        inputs: &Inputs,
        stream: &Arc<Stream>,
    ) {
        match selection {
            Selection::FullLatent => {
                if iteration.is_multiple_of(2) {
                    enqueue_full(full, inputs, stream);
                    enqueue_latent(latent, inputs, stream);
                } else {
                    enqueue_latent(latent, inputs, stream);
                    enqueue_full(full, inputs, stream);
                }
            }
            Selection::Full => enqueue_full(full, inputs, stream),
            Selection::Latent => enqueue_latent(latent, inputs, stream),
            Selection::LatentPreprojected => {
                enqueue_latent_preprojected(preprojected, inputs, stream)
            }
            Selection::All => match iteration % 3 {
                0 => {
                    enqueue_full(full, inputs, stream);
                    enqueue_latent(latent, inputs, stream);
                    enqueue_latent_preprojected(preprojected, inputs, stream);
                }
                1 => {
                    enqueue_latent(latent, inputs, stream);
                    enqueue_latent_preprojected(preprojected, inputs, stream);
                    enqueue_full(full, inputs, stream);
                }
                _ => {
                    enqueue_latent_preprojected(preprojected, inputs, stream);
                    enqueue_full(full, inputs, stream);
                    enqueue_latent(latent, inputs, stream);
                }
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn measure_selected_gpu(
        selection: Selection,
        iteration: usize,
        events: &EventPair,
        full_samples: &mut Vec<f64>,
        latent_samples: &mut Vec<f64>,
        preprojected_samples: &mut Vec<f64>,
        full: &mut Buffers,
        latent: &mut Buffers,
        preprojected: &mut PreprojectedBuffers,
        inputs: &Inputs,
        stream: &Arc<Stream>,
    ) {
        let mut measure_full = || {
            full_samples.push(events.measure(stream, || enqueue_full(full, inputs, stream)));
        };
        let mut measure_latent = || {
            latent_samples.push(events.measure(stream, || enqueue_latent(latent, inputs, stream)));
        };
        let mut measure_preprojected = || {
            preprojected_samples.push(events.measure(stream, || {
                enqueue_latent_preprojected(preprojected, inputs, stream)
            }));
        };
        match selection {
            Selection::FullLatent => {
                if iteration.is_multiple_of(2) {
                    measure_full();
                    measure_latent();
                } else {
                    measure_latent();
                    measure_full();
                }
            }
            Selection::Full => measure_full(),
            Selection::Latent => measure_latent(),
            Selection::LatentPreprojected => measure_preprojected(),
            Selection::All => match iteration % 3 {
                0 => {
                    measure_full();
                    measure_latent();
                    measure_preprojected();
                }
                1 => {
                    measure_latent();
                    measure_preprojected();
                    measure_full();
                }
                _ => {
                    measure_preprojected();
                    measure_full();
                    measure_latent();
                }
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn measure_selected_host(
        selection: Selection,
        iteration: usize,
        full_samples: &mut Vec<f64>,
        latent_samples: &mut Vec<f64>,
        preprojected_samples: &mut Vec<f64>,
        full: &mut Buffers,
        latent: &mut Buffers,
        preprojected: &mut PreprojectedBuffers,
        inputs: &Inputs,
        stream: &Arc<Stream>,
    ) {
        let mut measure_full = || {
            full_samples.push(measure_host(stream, || enqueue_full(full, inputs, stream)));
        };
        let mut measure_latent = || {
            latent_samples.push(measure_host(stream, || {
                enqueue_latent(latent, inputs, stream)
            }));
        };
        let mut measure_preprojected = || {
            preprojected_samples.push(measure_host(stream, || {
                enqueue_latent_preprojected(preprojected, inputs, stream)
            }));
        };
        match selection {
            Selection::FullLatent => {
                if iteration.is_multiple_of(2) {
                    measure_full();
                    measure_latent();
                } else {
                    measure_latent();
                    measure_full();
                }
            }
            Selection::Full => measure_full(),
            Selection::Latent => measure_latent(),
            Selection::LatentPreprojected => measure_preprojected(),
            Selection::All => match iteration % 3 {
                0 => {
                    measure_full();
                    measure_latent();
                    measure_preprojected();
                }
                1 => {
                    measure_latent();
                    measure_preprojected();
                    measure_full();
                }
                _ => {
                    measure_preprojected();
                    measure_full();
                    measure_latent();
                }
            },
        }
    }

    fn enqueue_full(buffers: &mut Buffers, inputs: &Inputs, stream: &Arc<Stream>) {
        let scores_out = buffers.scores.take().expect("missing full scores buffer");
        let (scores_part, _, _, _, _) = unsafe {
            full_kv_baseline_kernel_131072::model_small_full_kv_scores_fp16_storage_131072(
                scores_out.partition([1, BLOCK_SIZE]),
                &inputs.q,
                &inputs.k_full,
                &inputs.table,
                &inputs.active,
            )
            .async_on(stream)
            .expect("full-KV score launch failed")
        };
        let scores = scores_part.unpartition();
        let probabilities_out = buffers
            .probabilities
            .take()
            .expect("missing full probabilities buffer");
        let (probabilities_part, _, _) = unsafe {
            model_profile_kernel_131072::model_small_softmax_131072_runtime(
                probabilities_out.partition([1, MAX_SEQ_LEN]),
                &scores,
                &inputs.active,
            )
            .async_on(stream)
            .expect("full-KV softmax launch failed")
        };
        let probabilities = probabilities_part.unpartition();
        let context_out = buffers.context.take().expect("missing full context buffer");
        let (context_part, _, _, _) = unsafe {
            full_kv_baseline_kernel_131072::model_small_full_kv_context_fp16_storage_131072(
                context_out.partition([1, HEAD_DIM]),
                &probabilities,
                &inputs.v_full,
                &inputs.table,
            )
            .async_on(stream)
            .expect("full-KV context launch failed")
        };
        buffers.scores = Some(scores);
        buffers.probabilities = Some(probabilities);
        buffers.context = Some(context_part.unpartition());
    }

    fn enqueue_latent(buffers: &mut Buffers, inputs: &Inputs, stream: &Arc<Stream>) {
        let scores_out = buffers.scores.take().expect("missing latent scores buffer");
        let (scores_part, _, _, _, _, _) = unsafe {
            model_profile_kernel_131072::model_small_scores_fp16_storage_131072(
                scores_out.partition([1, BLOCK_SIZE]),
                &inputs.q,
                &inputs.latent,
                &inputs.table,
                &inputs.active,
                &inputs.k_projection,
            )
            .async_on(stream)
            .expect("latent score launch failed")
        };
        let scores = scores_part.unpartition();
        let probabilities_out = buffers
            .probabilities
            .take()
            .expect("missing latent probabilities buffer");
        let (probabilities_part, _, _) = unsafe {
            model_profile_kernel_131072::model_small_softmax_131072_runtime(
                probabilities_out.partition([1, MAX_SEQ_LEN]),
                &scores,
                &inputs.active,
            )
            .async_on(stream)
            .expect("latent softmax launch failed")
        };
        let probabilities = probabilities_part.unpartition();
        let context_out = buffers
            .context
            .take()
            .expect("missing latent context buffer");
        let (context_part, _, _, _, _) = unsafe {
            model_profile_kernel_131072::model_small_context_fp16_storage_131072(
                context_out.partition([1, HEAD_DIM]),
                &probabilities,
                &inputs.latent,
                &inputs.table,
                &inputs.v_projection,
            )
            .async_on(stream)
            .expect("latent context launch failed")
        };
        buffers.scores = Some(scores);
        buffers.probabilities = Some(probabilities);
        buffers.context = Some(context_part.unpartition());
    }

    fn enqueue_latent_preprojected(
        buffers: &mut PreprojectedBuffers,
        inputs: &Inputs,
        stream: &Arc<Stream>,
    ) {
        let projected_out = buffers
            .projected_query
            .take()
            .expect("missing projected-query buffer");
        let (projected_part, _, _) = unsafe {
            model_profile_preprojected_kernel_131072::model_small_project_query_once_131072(
                projected_out.partition([1, LATENT_DIM]),
                &inputs.q,
                &inputs.k_projection,
            )
            .async_on(stream)
            .expect("query projection launch failed")
        };
        let projected_query = projected_part.unpartition();

        let scores_out = buffers
            .attention
            .scores
            .take()
            .expect("missing preprojected scores buffer");
        let (scores_part, _, _, _, _) = unsafe {
            model_profile_preprojected_kernel_131072::model_small_scores_fp16_storage_preprojected_131072(
                scores_out.partition([1, BLOCK_SIZE]),
                &projected_query,
                &inputs.latent,
                &inputs.table,
                &inputs.active,
            )
            .async_on(stream)
            .expect("preprojected score launch failed")
        };
        let scores = scores_part.unpartition();
        let probabilities_out = buffers
            .attention
            .probabilities
            .take()
            .expect("missing preprojected probabilities buffer");
        let (probabilities_part, _, _) = unsafe {
            model_profile_kernel_131072::model_small_softmax_131072_runtime(
                probabilities_out.partition([1, MAX_SEQ_LEN]),
                &scores,
                &inputs.active,
            )
            .async_on(stream)
            .expect("preprojected softmax launch failed")
        };
        let probabilities = probabilities_part.unpartition();
        let context_out = buffers
            .attention
            .context
            .take()
            .expect("missing preprojected context buffer");
        let (context_part, _, _, _, _) = unsafe {
            model_profile_kernel_131072::model_small_context_fp16_storage_131072(
                context_out.partition([1, HEAD_DIM]),
                &probabilities,
                &inputs.latent,
                &inputs.table,
                &inputs.v_projection,
            )
            .async_on(stream)
            .expect("preprojected context launch failed")
        };

        buffers.projected_query = Some(projected_query);
        buffers.attention.scores = Some(scores);
        buffers.attention.probabilities = Some(probabilities);
        buffers.attention.context = Some(context_part.unpartition());
    }

    fn measure_host<F: FnOnce()>(stream: &Arc<Stream>, operation: F) -> f64 {
        unsafe {
            stream
                .synchronize()
                .expect("pre-host-timing stream sync failed")
        };
        let start = Instant::now();
        operation();
        unsafe {
            stream
                .synchronize()
                .expect("host-timing stream sync failed")
        };
        start.elapsed().as_secs_f64() * 1000.0
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

    fn error_metrics(actual_result: &GqaDecodeResult, expected: &GqaDecodeResult) -> ErrorMetrics {
        let mut actual = Vec::with_capacity(
            actual_result.scores.len()
                + actual_result.probabilities.len()
                + actual_result.context.len(),
        );
        actual.extend_from_slice(&actual_result.scores);
        actual.extend_from_slice(&actual_result.probabilities);
        actual.extend_from_slice(&actual_result.context);
        let mut reference = Vec::with_capacity(actual.len());
        reference.extend_from_slice(&expected.scores);
        reference.extend_from_slice(&expected.probabilities);
        reference.extend_from_slice(&expected.context);
        let absolute: Vec<f64> = actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| f64::from((actual - expected).abs()))
            .collect();
        let max_absolute_error = absolute.iter().copied().fold(0.0, f64::max);
        let mean_absolute_error = absolute.iter().sum::<f64>() / absolute.len() as f64;
        let rmse = (absolute.iter().map(|value| value * value).sum::<f64>()
            / absolute.len() as f64)
            .sqrt();
        let max_relative_error = actual
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
                &actual_result.probabilities,
            ),
        }
    }

    fn validate_correctness(correctness: &Correctness) {
        assert!(correctness.full_kv.max_absolute_error <= 5.0e-3);
        assert!(correctness.latent_original.max_absolute_error <= 5.0e-3);
        assert!(
            correctness
                .latent_preprojected_vs_rust_cpu
                .max_absolute_error
                <= 5.0e-3
        );
        assert!(
            correctness
                .latent_preprojected_vs_original_gpu
                .max_absolute_error
                <= 5.0e-3
        );
        assert!(correctness.full_kv.max_probability_row_sum_error <= 1.0e-4);
        assert!(correctness.latent_original.max_probability_row_sum_error <= 1.0e-4);
        assert!(
            correctness
                .latent_preprojected_vs_rust_cpu
                .max_probability_row_sum_error
                <= 1.0e-4
        );
    }

    fn stats(values: &[f64]) -> Stats {
        assert!(!values.is_empty());
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let variance = sorted
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / sorted.len() as f64;
        Stats {
            n: sorted.len(),
            min_ms: sorted[0],
            mean_ms: mean,
            median_ms: percentile(&sorted, 0.5),
            standard_deviation_ms: variance.sqrt(),
            p5_ms: percentile(&sorted, 0.05),
            p50_ms: percentile(&sorted, 0.5),
            p95_ms: percentile(&sorted, 0.95),
            p99_ms: percentile(&sorted, 0.99),
            max_ms: *sorted.last().expect("nonempty samples"),
        }
    }

    fn percentile(sorted: &[f64], quantile: f64) -> f64 {
        let position = quantile * (sorted.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        if lower == upper {
            sorted[lower]
        } else {
            let fraction = position - lower as f64;
            sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
        }
    }

    fn write_samples(path: &Path, variant: &str, gpu: &[f64], host: &[f64]) {
        let mut writer = BufWriter::new(File::create(path).expect("failed to create samples"));
        write_sample_rows(&mut writer, variant, gpu, host);
    }

    fn append_samples(path: &Path, variant: &str, gpu: &[f64], host: &[f64]) {
        let file = fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("failed to append samples");
        let mut writer = BufWriter::new(file);
        write_sample_rows(&mut writer, variant, gpu, host);
    }

    fn write_sample_rows(writer: &mut impl Write, variant: &str, gpu: &[f64], host: &[f64]) {
        for (iteration, &latency_ms) in gpu.iter().enumerate() {
            serde_json::to_writer(
                &mut *writer,
                &Sample {
                    variant,
                    timing: "gpu_event",
                    iteration,
                    latency_ms,
                },
            )
            .expect("failed to serialize GPU sample");
            writer.write_all(b"\n").expect("failed to write newline");
        }
        for (iteration, &latency_ms) in host.iter().enumerate() {
            serde_json::to_writer(
                &mut *writer,
                &Sample {
                    variant,
                    timing: "host_synchronized",
                    iteration,
                    latency_ms,
                },
            )
            .expect("failed to serialize host sample");
            writer.write_all(b"\n").expect("failed to write newline");
        }
    }

    fn check_cuda(result: sys::CUresult, operation: &str) {
        assert_eq!(
            result,
            sys::cudaError_enum_CUDA_SUCCESS,
            "{operation} failed with CUDA result {result}"
        );
    }

    fn max_probability_row_sum_error(probabilities: &[f32]) -> f64 {
        (0..Q_HEADS)
            .map(|head| {
                let start = head * MAX_SEQ_LEN;
                f64::from(
                    (probabilities[start..start + MAX_SEQ_LEN]
                        .iter()
                        .sum::<f32>()
                        - 1.0)
                        .abs(),
                )
            })
            .fold(0.0, f64::max)
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
