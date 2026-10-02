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
    // In-process NVML reader (dlopen; no link-time dependency). Missing symbols are recorded
    // as unavailable rather than fabricated.
    struct Nvml {
        _lib: libloading::Library,
        device: *mut std::ffi::c_void,
        clock: unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32,
        temperature: unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32,
        reasons: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64) -> i32>,
    }
    impl Nvml {
        fn open() -> Option<Self> {
            unsafe {
                let lib = libloading::Library::new("libnvidia-ml.so.1").ok()?;
                let init: libloading::Symbol<unsafe extern "C" fn() -> i32> =
                    lib.get(b"nvmlInit_v2\0").ok()?;
                if init() != 0 {
                    return None;
                }
                let handle: libloading::Symbol<
                    unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32,
                > = lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
                let mut device = std::ptr::null_mut();
                if handle(0, &mut device) != 0 {
                    return None;
                }
                let clock = *lib.get(b"nvmlDeviceGetClockInfo\0").ok()?;
                let temperature = *lib.get(b"nvmlDeviceGetTemperature\0").ok()?;
                let reasons = lib
                    .get(b"nvmlDeviceGetCurrentClocksEventReasons\0")
                    .or_else(|_| lib.get(b"nvmlDeviceGetCurrentClocksThrottleReasons\0"))
                    .ok()
                    .map(|f| *f);
                Some(Self {
                    _lib: lib,
                    device,
                    clock,
                    temperature,
                    reasons,
                })
            }
        }
        fn sm_clock(&self) -> Option<u32> {
            let mut v = 0;
            // NVML_CLOCK_SM = 1
            (unsafe { (self.clock)(self.device, 1, &mut v) } == 0).then_some(v)
        }
        fn mem_clock(&self) -> Option<u32> {
            let mut v = 0;
            // NVML_CLOCK_MEM = 2
            (unsafe { (self.clock)(self.device, 2, &mut v) } == 0).then_some(v)
        }
        fn temperature(&self) -> Option<u32> {
            let mut v = 0;
            (unsafe { (self.temperature)(self.device, 0, &mut v) } == 0).then_some(v)
        }
        fn reasons(&self) -> Option<u64> {
            let mut v = 0;
            self.reasons
                .and_then(|f| (unsafe { f(self.device, &mut v) } == 0).then_some(v))
        }
    }
    const ABORT_TEMPERATURE_C: u32 = 91;

    fn write_json(dir: &std::path::Path, name: &str, value: &serde_json::Value, pretty: bool) {
        let file = std::io::BufWriter::new(File::create(dir.join(name)).unwrap());
        if pretty {
            serde_json::to_writer_pretty(file, value).unwrap();
        } else {
            serde_json::to_writer(file, value).unwrap();
        }
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let arg = |key: &str, default: &str| {
            args.windows(2)
                .find(|w| w[0] == key)
                .map(|w| w[1].clone())
                .unwrap_or(default.into())
        };
        let flag = |key: &str| args.iter().any(|a| a == key);
        let n: usize = arg("--seq", "1024").parse().unwrap();
        let warmup: usize = arg("--warmup", "3").parse().unwrap();
        let iterations: usize = arg("--iterations", "12").parse().unwrap();
        let process: usize = arg("--process", "0").parse().unwrap();
        let clock_warm_ms: u64 = arg("--clock-warm-ms", "2000").parse().unwrap();
        let keepalive_max_ms: u64 = arg("--keepalive-max-ms", "0").parse().unwrap();
        let skip_oracle_export = flag("--skip-oracle-export");
        let variant_arg = arg("--variants", "A1,B1,C1");
        let dir = PathBuf::from(arg("--output-dir", "/tmp/plkv_c1_native"));
        // Slow 9P/OneDrive writes idle the GPU into P8 before and between timed samples
        // (diagnosis 2026-10-02); require an explicit override to write under /mnt.
        let absolute = std::path::absolute(&dir).unwrap();
        assert!(
            !absolute.starts_with("/mnt") || flag("--allow-slow-output"),
            "--output-dir {absolute:?} is on /mnt; use a Linux filesystem path or --allow-slow-output"
        );
        fs::create_dir_all(&dir).unwrap();
        let launch: Launch = match n {
            128 => launch_128,
            256 => launch_256,
            512 => launch_512,
            1024 => launch_1024,
            2048 => launch_2048,
            4096 => launch_4096,
            8192 => launch_8192,
            16384 => launch_16384,
            32768 => launch_32768,
            _ => panic!("Phase-2 sweep supports 128 through 32768 power-of-two lengths"),
        };
        let st = with_default_device_policy(|p| p.next_stream())
            .unwrap()
            .unwrap();
        st.device().bind_to_thread().unwrap();
        let nvml = Nvml::open();
        let (inp, cf, cb) = make_inputs(n, n, n / 16, &st, -0.4);
        // Correctness always covers A1/B1/C1; timing covers only the selected variants.
        let all_variants = ["A1", "B1", "C1"];
        let mut buffers: Vec<_> = all_variants.iter().map(|_| make_buffers(n, &st)).collect();
        for (v, b) in all_variants.iter().zip(&mut buffers) {
            launch(v, "pipeline", b, &inp, &st, n);
        }
        unsafe {
            st.synchronize().unwrap();
        }
        let outputs: Vec<_> = buffers.iter().map(|b| read_outputs(b, &st)).collect();
        let checks = serde_json::json!({"A1_reference":validate(n,&outputs[0],&cf),"B1_reference":validate(n,&outputs[1],&cb),"C1_reference":validate(n,&outputs[2],&cb),"C1_B1":validate(n,&outputs[2],&outputs[1]),"C1_B1_bitwise_diagnostic":comparison_metrics(&outputs[2],&outputs[1])});
        for key in ["A1_reference", "B1_reference", "C1_reference", "C1_B1"] {
            if checks[key]["pass"] != true {
                write_json(&dir, "correctness.json", &checks, true);
                panic!("{key} failed: {checks}");
            }
        }
        // A distinct decode query must refresh the projection; never reuse stale projected q.
        let refresh = {
            let (changed, _, changed_ref) = make_inputs(n, n, n / 16, &st, 0.2);
            launch("C1", "pipeline", &mut buffers[2], &changed, &st, n);
            let changed_output = read_outputs(&buffers[2], &st);
            let refresh = validate(n, &changed_output, &changed_ref);
            if refresh["pass"] != true || outputs[2].scores == changed_output.scores {
                write_json(&dir, "query_refresh.json", &refresh, true);
                panic!("changed-query projection was not refreshed");
            }
            // Restore matched resident inputs before warmup and timing.
            launch("C1", "pipeline", &mut buffers[2], &inp, &st, n);
            refresh
        };
        let selected: Vec<usize> = variant_arg
            .split(',')
            .map(|v| {
                all_variants
                    .iter()
                    .position(|x| *x == v)
                    .expect("unknown variant")
            })
            .collect();
        let phases = ["pipeline", "score", "softmax", "context", "projection"];
        // JIT and fixed warmup outside event intervals.
        for phase in phases {
            for &index in &selected {
                let v = all_variants[index];
                if phase == "projection" && v != "C1" {
                    continue;
                }
                for _ in 0..warmup {
                    launch(v, phase, &mut buffers[index], &inp, &st, n);
                }
            }
        }
        // Clock warm-up: back-to-back untimed pipelines until the SM clock settles.
        let mut warm_clocks = Vec::new();
        let warm_start = std::time::Instant::now();
        while warm_start.elapsed().as_millis() < u128::from(clock_warm_ms) {
            for _ in 0..8 {
                for &index in &selected {
                    launch(
                        all_variants[index],
                        "pipeline",
                        &mut buffers[index],
                        &inp,
                        &st,
                        n,
                    );
                }
            }
            unsafe {
                st.synchronize().unwrap();
            }
            if let Some(c) = nvml.as_ref().and_then(Nvml::sm_clock) {
                warm_clocks.push(c);
            }
        }
        // Reference clock: mode of the second half of warm-up reads.
        let reference_clock = {
            let tail = &warm_clocks[warm_clocks.len() / 2..];
            let mut counts = std::collections::BTreeMap::new();
            for c in tail {
                *counts.entry(*c).or_insert(0usize) += 1;
            }
            counts.into_iter().max_by_key(|(_, k)| *k).map(|(c, _)| c)
        };
        let keepalive_threshold = reference_clock.map(|c| f64::from(c) * 0.95);
        let ev = Events::new();
        let mut samples: Vec<serde_json::Value> = Vec::new();
        let mut aborted = None;
        let timing_start = std::time::Instant::now();
        'timing: for iteration in 0..iterations {
            for phase in ["empty"].into_iter().chain(phases) {
                for offset in 0..selected.len() {
                    let index = selected[(iteration + process + offset) % selected.len()];
                    let v = if phase == "empty" {
                        "NONE"
                    } else {
                        all_variants[index]
                    };
                    if (phase == "projection" && v != "C1") || (phase == "empty" && offset > 0) {
                        continue;
                    }
                    // Optional keep-alive (off by default; --keepalive-max-ms > 0 enables it).
                    // Disabled because it cannot override temperature-driven DVFS and adds heat
                    // (diagnosis 2026-10-02, E0b).
                    let mut keepalive_launches = 0usize;
                    let mut pre_clock = nvml.as_ref().and_then(Nvml::sm_clock);
                    if let (Some(threshold), Some(_)) = (keepalive_threshold, pre_clock) {
                        let started = std::time::Instant::now();
                        while pre_clock.is_some_and(|c| f64::from(c) < threshold)
                            && started.elapsed().as_millis() < u128::from(keepalive_max_ms)
                        {
                            for _ in 0..8 {
                                launch(
                                    v_or(v, all_variants[index]),
                                    "pipeline",
                                    &mut buffers[index],
                                    &inp,
                                    &st,
                                    n,
                                );
                                keepalive_launches += 1;
                            }
                            unsafe {
                                st.synchronize().unwrap();
                            }
                            pre_clock = nvml.as_ref().and_then(Nvml::sm_clock);
                        }
                    }
                    let ms = if phase == "empty" {
                        ev.measure(&st, || {})
                    } else {
                        ev.measure(&st, || launch(v, phase, &mut buffers[index], &inp, &st, n))
                    };
                    let post_clock = nvml.as_ref().and_then(Nvml::sm_clock);
                    let mem_clock = nvml.as_ref().and_then(Nvml::mem_clock);
                    let keepalive_restored = keepalive_threshold
                        .zip(pre_clock)
                        .map(|(threshold, c)| f64::from(c) >= threshold);
                    let temperature = nvml.as_ref().and_then(Nvml::temperature);
                    let reasons = nvml.as_ref().and_then(Nvml::reasons);
                    samples.push(serde_json::json!({"seq":n,"process":process,"iteration":iteration,"order":offset,"variant":v,"component":phase,"latency_ms":ms,"t_ms":timing_start.elapsed().as_secs_f64()*1e3,"sm_clock_pre_mhz":pre_clock,"sm_clock_post_mhz":post_clock,"reference_clock_mhz":reference_clock,"keepalive_launches":keepalive_launches,"keepalive_restored":keepalive_restored,"mem_clock_post_mhz":mem_clock,"temperature_c":temperature,"clock_event_reasons":reasons}));
                    if temperature.is_some_and(|t| t >= ABORT_TEMPERATURE_C) {
                        aborted = Some(temperature);
                        break 'timing;
                    }
                }
            }
        }
        // All file output happens after the timing loop.
        let mut raw = std::io::BufWriter::new(
            File::options()
                .write(true)
                .create_new(true)
                .open(dir.join("samples.jsonl"))
                .unwrap(),
        );
        for sample in &samples {
            writeln!(raw, "{sample}").unwrap();
        }
        raw.flush().unwrap();
        write_json(&dir, "correctness.json", &checks, true);
        write_json(&dir, "query_refresh.json", &refresh, true);
        write_json(
            &dir,
            "timing_meta.json",
            &serde_json::json!({"nvml_available":nvml.is_some(),"reference_clock_mhz":reference_clock,"warm_clocks_mhz":warm_clocks,"clock_warm_ms":clock_warm_ms,"keepalive_max_ms":keepalive_max_ms,"timed_variants":variant_arg,"iterations":iterations,"warmup":warmup,"aborted_temperature_c":aborted}),
            true,
        );
        if !skip_oracle_export {
            // Export actual inputs/outputs for an independent FP64, reconstructed-K/V oracle.
            let q = deterministic_values(Q_HEADS * HEAD_DIM, 0.011, -0.4);
            let z = deterministic_values(n * LATENT_DIM, 0.007, -1.2);
            let kp = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.005, -0.7);
            let vp = deterministic_values(LATENT_DIM * KV_HEADS * HEAD_DIM, 0.006, 0.3);
            let data = serde_json::json!({"q":q,"latent_logical_prequantization":z,"kp":kp,"vp":vp,"scores":outputs[2].scores,"probabilities":outputs[2].probabilities,"context":outputs[2].context,"table":model_block_table(n/16)});
            write_json(&dir, "oracle_inputs_outputs.json", &data, false);
        }
        if aborted.is_some() {
            eprintln!(
                "THERMAL_ABORT temperature >= {ABORT_TEMPERATURE_C} C; partial samples saved"
            );
            std::process::exit(3);
        }
        println!("C1_OK=1");
    }
    fn v_or<'a>(v: &'a str, fallback: &'a str) -> &'a str {
        if v == "NONE" { fallback } else { v }
    }
    define_launch!(
        launch_128,
        p15b_full_kv_baseline_kernel_128::model_small_full_kv_scores_fp16_storage_rtable_128,
        p15b_full_kv_baseline_kernel_128::model_small_full_kv_context_fp16_storage_rtable_128,
        p15b_model_profile_kernel_128::model_small_scores_fp16_storage_rtable_128,
        p15b_model_profile_kernel_128::model_small_context_fp16_storage_rtable_128,
        p15b_model_profile_kernel_128::model_small_softmax_128_runtime,
        c1_kernel_128::model_small_project_query_once,
        c1_kernel_128::model_small_scores_fp16_storage_preprojected
    );
    define_launch!(
        launch_256,
        p15b_full_kv_baseline_kernel_256::model_small_full_kv_scores_fp16_storage_rtable_256,
        p15b_full_kv_baseline_kernel_256::model_small_full_kv_context_fp16_storage_rtable_256,
        p15b_model_profile_kernel_256::model_small_scores_fp16_storage_rtable_256,
        p15b_model_profile_kernel_256::model_small_context_fp16_storage_rtable_256,
        p15b_model_profile_kernel_256::model_small_softmax_256_runtime,
        c1_kernel_256::model_small_project_query_once,
        c1_kernel_256::model_small_scores_fp16_storage_preprojected
    );
    define_launch!(
        launch_512,
        p15b_full_kv_baseline_kernel_512::model_small_full_kv_scores_fp16_storage_rtable_512,
        p15b_full_kv_baseline_kernel_512::model_small_full_kv_context_fp16_storage_rtable_512,
        p15b_model_profile_kernel_512::model_small_scores_fp16_storage_rtable_512,
        p15b_model_profile_kernel_512::model_small_context_fp16_storage_rtable_512,
        p15b_model_profile_kernel_512::model_small_softmax_512_runtime,
        c1_kernel_512::model_small_project_query_once,
        c1_kernel_512::model_small_scores_fp16_storage_preprojected
    );
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
    define_launch!(
        launch_16384,
        p15b_full_kv_baseline_kernel_16384::model_small_full_kv_scores_fp16_storage_rtable_16384,
        p15b_full_kv_baseline_kernel_16384::model_small_full_kv_context_fp16_storage_rtable_16384,
        p15b_model_profile_kernel_16384::model_small_scores_fp16_storage_rtable_16384,
        p15b_model_profile_kernel_16384::model_small_context_fp16_storage_rtable_16384,
        p15b_model_profile_kernel_16384::model_small_softmax_16384_runtime,
        c1_kernel_16384::model_small_project_query_once,
        c1_kernel_16384::model_small_scores_fp16_storage_preprojected
    );
    define_launch!(
        launch_32768,
        p15b_full_kv_baseline_kernel_32768::model_small_full_kv_scores_fp16_storage_rtable_32768,
        p15b_full_kv_baseline_kernel_32768::model_small_full_kv_context_fp16_storage_rtable_32768,
        p15b_model_profile_kernel_32768::model_small_scores_fp16_storage_rtable_32768,
        p15b_model_profile_kernel_32768::model_small_context_fp16_storage_rtable_32768,
        p15b_model_profile_kernel_32768::model_small_softmax_32768_runtime,
        c1_kernel_32768::model_small_project_query_once,
        c1_kernel_32768::model_small_scores_fp16_storage_preprojected
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
