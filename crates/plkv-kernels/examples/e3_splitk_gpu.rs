//! E3 split-K correctness harness (GPU, correctness only; no timing).
//!
//! Chain: Python oracle (fixtures/reference, FP32) -> Rust CPU FP32 reference (asserted equal on
//! the tiny fixtures) -> Rust CPU FP16-storage runtime reference -> cuTile E3 GPU kernels (C3
//! latent, A3 full-KV). Profiles: tiny (fixture cases, block table [2, 0, 3, 1]) and model_small
//! (Hq=16, Hkv=4, d=64, dl=32, block=16, N=1024, non-identity table (17l+11) mod 64). Active
//! lengths include partial final blocks; several split counts check split invariance; an
//! identity-table run is a negative control. Every launch is preceded by host-side block-table
//! and layout validation.
#[cfg(feature = "gpu-cutile")]
#[path = "timing_common/mod.rs"]
mod timing_common;

#[cfg(feature = "gpu-cutile")]
mod gpu_impl {
    use super::timing_common::{
        ABORT_TEMPERATURE_C, Events, Nvml, clock_fields, clock_warmup, memset_d32,
    };
    use cutile::api;
    use cutile::cuda_async::device_context::with_default_device_policy;
    use cutile::cuda_core::Stream;
    use cutile::cutile_compiler::hints::CompileOptions;
    use cutile::half::f16;
    use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
    use cutile::tile_kernel::{DeviceOp, TileKernel};
    use plkv_core::{
        GqaDecodeResult, direct_paged_latent_gqa_decode_f32,
        direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum,
        paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum, paged_gqa_decode_f32,
        quantize_f32_to_f16_storage,
    };
    use plkv_kernels::alignment::{TensorLayout, VECTOR_ALIGN_BYTES, check_layouts};
    use plkv_kernels::block_table::validate_block_table;
    use plkv_kernels::cutile::e3_splitk::e3_kernels;
    use plkv_kernels::splitk_plan::{MAX_STEPS_PER_CTA, SplitPlan, plan_bounded};
    use serde_json::Value;
    use std::sync::Arc;

    const ATOL: f64 = 5e-3;

    #[derive(Clone, Copy)]
    struct Dims {
        hq: usize,
        hkv: usize,
        d: usize,
        l: usize,
        blk: usize,
        seq: usize,
        dc: usize,
    }

    impl Dims {
        fn g(&self) -> usize {
            self.hq / self.hkv
        }
        fn nb(&self) -> usize {
            self.seq / self.blk
        }
        fn scale(&self) -> f32 {
            1.0 / (self.d as f32).sqrt()
        }
    }

    /// Host-side data for one case.
    struct Case {
        name: String,
        q: Vec<f32>,
        /// Query for the full-KV path (the tiny fixtures are independent datasets).
        q_full: Vec<f32>,
        latent_physical: Vec<f32>,
        k_projection: Vec<f32>,
        v_projection: Vec<f32>,
        kp_head_major: Vec<f32>,
        vp_head_major: Vec<f32>,
        k_physical_head_major: Vec<f32>,
        v_physical_head_major: Vec<f32>,
        table: Vec<usize>,
        oracle_latent_context: Option<Vec<f32>>,
        oracle_full_context: Option<Vec<f32>>,
    }

    struct Device {
        q: Tensor<f32>,
        q_full: Tensor<f32>,
        kp: Tensor<f32>,
        vp: Tensor<f32>,
        latent: Tensor<f16>,
        k_full: Tensor<f16>,
        v_full: Tensor<f16>,
        table: Tensor<i32>,
    }

    /// Borrowed kernel inputs; lets batch elements share one physical block pool per layer.
    #[derive(Clone, Copy)]
    struct Views<'a> {
        q: &'a Tensor<f32>,
        q_full: &'a Tensor<f32>,
        kp: &'a Tensor<f32>,
        vp: &'a Tensor<f32>,
        latent: &'a Tensor<f16>,
        k_full: &'a Tensor<f16>,
        v_full: &'a Tensor<f16>,
        table: &'a Tensor<i32>,
    }

    impl Device {
        fn views(&self) -> Views<'_> {
            Views {
                q: &self.q,
                q_full: &self.q_full,
                kp: &self.kp,
                vp: &self.vp,
                latent: &self.latent,
                k_full: &self.k_full,
                v_full: &self.v_full,
                table: &self.table,
            }
        }
    }

    fn flat(value: &Value) -> Vec<f32> {
        match value {
            Value::Array(items) => items.iter().flat_map(flat).collect(),
            Value::Number(n) => vec![n.as_f64().expect("number") as f32],
            other => panic!("unexpected fixture value {other}"),
        }
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                assert!(x.is_finite() && y.is_finite(), "non-finite value");
                f64::from((x - y).abs())
            })
            .fold(0.0, f64::max)
    }

    fn upload_f32(values: &[f32], shape: &[usize], st: &Arc<Stream>) -> Tensor<f32> {
        api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
            .sync_on(st)
            .expect("upload")
            .reshape(shape)
            .expect("reshape")
    }

    fn upload_f16(values: &[f16], shape: &[usize], st: &Arc<Stream>) -> Tensor<f16> {
        api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
            .sync_on(st)
            .expect("upload")
            .reshape(shape)
            .expect("reshape")
    }

    fn upload_table(table: &[usize], st: &Arc<Stream>) -> Tensor<i32> {
        let values: Vec<i32> = table.iter().map(|&v| v as i32).collect();
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(st)
            .expect("upload")
            .reshape(&[table.len()])
            .expect("reshape")
    }

    fn to_device(dims: Dims, case: &Case, st: &Arc<Stream>) -> Device {
        let latent_f16 = quantize_f32_to_f16_storage(&case.latent_physical).unwrap();
        let k_f16 = quantize_f32_to_f16_storage(&case.k_physical_head_major).unwrap();
        let v_f16 = quantize_f32_to_f16_storage(&case.v_physical_head_major).unwrap();
        let kv_rows = dims.nb() * dims.hkv * dims.blk;
        Device {
            q: upload_f32(&case.q, &[dims.hq, dims.d], st),
            q_full: upload_f32(&case.q_full, &[dims.hq, dims.d], st),
            kp: upload_f32(&case.kp_head_major, &[dims.hkv * dims.l, dims.d], st),
            vp: upload_f32(&case.vp_head_major, &[dims.hkv * dims.l, dims.d], st),
            latent: upload_f16(&latent_f16, &[dims.seq, dims.l], st),
            k_full: upload_f16(&k_f16, &[kv_rows, dims.d], st),
            v_full: upload_f16(&v_f16, &[kv_rows, dims.d], st),
            table: upload_table(&case.table, st),
        }
    }

    /// Host-side launch guard (design §0(e)): table bounds and 16-byte layouts.
    fn validate_launch(dims: Dims, dev: &Device, table: &[usize], active: usize) {
        let table_i32: Vec<i32> = table.iter().map(|&v| v as i32).collect();
        validate_block_table(&table_i32, active, dims.blk, dims.nb(), dims.nb())
            .unwrap_or_else(|e| panic!("block table rejected: {e}"));
        let latent_strides = dev.latent.strides().to_vec();
        let k_strides = dev.k_full.strides().to_vec();
        let v_strides = dev.v_full.strides().to_vec();
        let layouts = [
            TensorLayout {
                name: "latent",
                base_ptr: dev.latent.device_pointer().cu_deviceptr(),
                strides: &latent_strides,
                elem_bytes: 2,
            },
            TensorLayout {
                name: "k_full",
                base_ptr: dev.k_full.device_pointer().cu_deviceptr(),
                strides: &k_strides,
                elem_bytes: 2,
            },
            TensorLayout {
                name: "v_full",
                base_ptr: dev.v_full.device_pointer().cu_deviceptr(),
                strides: &v_strides,
                elem_bytes: 2,
            },
        ];
        if let Err(errors) = check_layouts(&layouts, VECTOR_ALIGN_BYTES) {
            panic!("layout rejected: {errors:?}");
        }
    }

    /// (splits, steps_per_split) covering the whole table: every split reads nb / (2 * splits)
    /// two-block steps.
    fn plan_full(dims: Dims, splits: usize) -> (usize, usize) {
        let steps = dims.nb() / 2;
        assert!(
            dims.nb().is_multiple_of(2) && steps.is_multiple_of(splits),
            "splits must divide nb/2"
        );
        (splits, steps / splits)
    }

    fn gens(values: &[usize]) -> Vec<String> {
        values.iter().map(usize::to_string).collect()
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Mode {
        /// Single FP16 q' in the score GEMM.
        Fast,
        /// q' = hi + lo (two FP16 halves), two score MMAs into one FP32 accumulator.
        Accurate,
    }

    fn run_c3(
        dims: Dims,
        dev: Views,
        active: usize,
        (splits, sps): (usize, usize),
        mode: Mode,
        st: &Arc<Stream>,
    ) -> Vec<f32> {
        assert!(
            sps <= MAX_STEPS_PER_CTA,
            "C3 plan exceeds {MAX_STEPS_PER_CTA} steps per CTA"
        );
        let (hq, l, d, g) = (dims.hq, dims.l, dims.d, dims.g());
        let acc = api::zeros::<f32>(&[splits, hq, l]).sync_on(st).unwrap();
        let m = api::zeros::<f32>(&[splits, hq, 1]).sync_on(st).unwrap();
        let lsum = api::zeros::<f32>(&[splits, hq, 1]).sync_on(st).unwrap();
        let (acc, m, lsum, _, _, _, _, _, _, _) = e3_kernels::c3_split(
            acc.partition([1, hq, l]),
            m.partition([1, hq, 1]),
            lsum.partition([1, hq, 1]),
            dev.q,
            dev.kp,
            dev.latent,
            dev.table,
            active as i32,
            sps as i32,
            dims.scale(),
        )
        .generics(gens(&[
            hq,
            dims.hkv,
            g,
            d,
            l,
            dims.blk,
            2 * dims.blk,
            dims.dc,
            dims.hkv * l,
            usize::from(mode == Mode::Accurate),
        ]))
        .sync_on(st)
        .expect("c3_split launch");
        let (acc, m, lsum) = (acc.unpartition(), m.unpartition(), lsum.unpartition());
        let out = api::zeros::<f32>(&[hq, d]).sync_on(st).unwrap();
        let (out, _, _, _, _, _) = e3_kernels::c3_reduce(
            out.partition([g, d]),
            &acc,
            &m,
            &lsum,
            dev.vp,
            splits as i32,
        )
        .generics(gens(&[hq, g, d, l]))
        .sync_on(st)
        .expect("c3_reduce launch");
        out.unpartition()
            .to_host_vec()
            .sync_on(st)
            .expect("readback")
    }

    fn run_a3(
        dims: Dims,
        dev: Views,
        active: usize,
        (splits, sps): (usize, usize),
        st: &Arc<Stream>,
    ) -> Vec<f32> {
        let (hq, d, g) = (dims.hq, dims.d, dims.g());
        let acc = api::zeros::<f32>(&[splits, hq, d]).sync_on(st).unwrap();
        let m = api::zeros::<f32>(&[splits, hq, 1]).sync_on(st).unwrap();
        let lsum = api::zeros::<f32>(&[splits, hq, 1]).sync_on(st).unwrap();
        let (acc, m, lsum, _, _, _, _, _, _, _) = e3_kernels::a3_split(
            acc.partition([1, g, d]),
            m.partition([1, g, 1]),
            lsum.partition([1, g, 1]),
            dev.q_full,
            dev.k_full,
            dev.v_full,
            dev.table,
            active as i32,
            sps as i32,
            dims.scale(),
        )
        .generics(gens(&[hq, dims.hkv, g, d, dims.blk, 2 * dims.blk]))
        .sync_on(st)
        .expect("a3_split launch");
        let (acc, m, lsum) = (acc.unpartition(), m.unpartition(), lsum.unpartition());
        let out = api::zeros::<f32>(&[hq, d]).sync_on(st).unwrap();
        let (out, _, _, _, _) =
            e3_kernels::a3_reduce(out.partition([g, d]), &acc, &m, &lsum, splits as i32)
                .generics(gens(&[hq, g, d]))
                .sync_on(st)
                .expect("a3_reduce launch");
        out.unpartition()
            .to_host_vec()
            .sync_on(st)
            .expect("readback")
    }

    /// Padded A3 (query rows of a KV group zero-padded to MP = 16, tensor-core shapes), compiled
    /// with the occupancy(4) hint that caps registers at 128. Requires HQ == 16.
    fn run_a3_pad(
        dims: Dims,
        dev: Views,
        active: usize,
        (splits, sps): (usize, usize),
        st: &Arc<Stream>,
    ) -> Vec<f32> {
        const MP: usize = 16;
        assert_eq!(dims.hq, MP, "a3_split_pad needs HQ == 16");
        assert!(
            sps <= MAX_STEPS_PER_CTA,
            "A3 plan exceeds {MAX_STEPS_PER_CTA} steps per CTA"
        );
        let (hq, d, g, hkv) = (dims.hq, dims.d, dims.g(), dims.hkv);
        let acc = api::zeros::<f32>(&[splits, hkv * MP, d])
            .sync_on(st)
            .unwrap();
        let m = api::zeros::<f32>(&[splits, hkv * MP, 1])
            .sync_on(st)
            .unwrap();
        let lsum = api::zeros::<f32>(&[splits, hkv * MP, 1])
            .sync_on(st)
            .unwrap();
        let (acc, m, lsum, _, _, _, _, _, _, _) = e3_kernels::a3_split_pad(
            acc.partition([1, MP, d]),
            m.partition([1, MP, 1]),
            lsum.partition([1, MP, 1]),
            dev.q_full,
            dev.k_full,
            dev.v_full,
            dev.table,
            active as i32,
            sps as i32,
            dims.scale(),
        )
        .generics(gens(&[hq, hkv, g, d, dims.blk, 2 * dims.blk, MP]))
        .compile_options(CompileOptions::new().occupancy(4))
        .sync_on(st)
        .expect("a3_split_pad launch");
        let (acc, m, lsum) = (acc.unpartition(), m.unpartition(), lsum.unpartition());
        let out = api::zeros::<f32>(&[hq, d]).sync_on(st).unwrap();
        let (out, _, _, _, _) =
            e3_kernels::a3_reduce_pad(out.partition([g, d]), &acc, &m, &lsum, splits as i32)
                .generics(gens(&[g, d, MP]))
                .sync_on(st)
                .expect("a3_reduce_pad launch");
        out.unpartition()
            .to_host_vec()
            .sync_on(st)
            .expect("readback")
    }

    fn cpu_refs(dims: Dims, case: &Case, active: usize) -> (GqaDecodeResult, GqaDecodeResult) {
        let latent_f16 = quantize_f32_to_f16_storage(&case.latent_physical).unwrap();
        let k_f16 = quantize_f32_to_f16_storage(&case.k_physical_head_major).unwrap();
        let v_f16 = quantize_f32_to_f16_storage(&case.v_physical_head_major).unwrap();
        let latent = direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
            &case.q,
            &latent_f16,
            &case.table,
            &case.k_projection,
            &case.v_projection,
            dims.hq,
            dims.hkv,
            dims.seq,
            active,
            dims.l,
            dims.d,
            dims.g(),
            dims.blk,
            dims.nb(),
        )
        .expect("latent CPU reference");
        let full = paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum(
            &case.q_full,
            &k_f16,
            &v_f16,
            &case.table,
            dims.hq,
            dims.hkv,
            dims.seq,
            active,
            dims.d,
            dims.g(),
            dims.blk,
            dims.nb(),
        )
        .expect("full CPU reference");
        (latent, full)
    }

    fn tiny_cases(dims: Dims) -> Vec<Case> {
        let latent_fx: Value = serde_json::from_str(
            &std::fs::read_to_string("fixtures/reference/direct_paged_latent_gqa_decode_f32.json")
                .expect("latent fixture"),
        )
        .unwrap();
        let full_fx: Value = serde_json::from_str(
            &std::fs::read_to_string("fixtures/reference/paged_gqa_decode_f32.json")
                .expect("full fixture"),
        )
        .unwrap();
        let table: Vec<usize> = latent_fx["block_table"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let full_table: Vec<usize> = full_fx["block_table"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(table, full_table, "tiny fixtures must share a block table");
        let latent_cases = latent_fx["cases"].as_array().unwrap();
        let full_cases = full_fx["cases"].as_array().unwrap();
        latent_cases
            .iter()
            .zip(full_cases)
            .map(|(lc, fc)| {
                // Oracle chain link: Rust FP32 references must reproduce the Python oracle.
                let q = flat(&lc["q"]);
                let latent_physical = flat(&lc["latent_physical_blocks"]);
                let k_projection = flat(&lc["k_projection"]);
                let v_projection = flat(&lc["v_projection"]);
                let rust_latent = direct_paged_latent_gqa_decode_f32(
                    &q,
                    &latent_physical,
                    &table,
                    &k_projection,
                    &v_projection,
                    dims.hq,
                    dims.hkv,
                    dims.seq,
                    dims.l,
                    dims.d,
                    dims.g(),
                    dims.blk,
                    dims.nb(),
                )
                .unwrap();
                let oracle_latent = flat(&lc["expected_context"]);
                let err = max_abs(&rust_latent.context, &oracle_latent);
                assert!(err <= 1e-5, "Rust FP32 latent ref vs Python oracle: {err}");
                let fq = flat(&fc["q"]);
                let k_hm = flat(&fc["k_physical_gpu_head_major"]);
                let v_hm = flat(&fc["v_physical_gpu_head_major"]);
                let rust_full = paged_gqa_decode_f32(
                    &fq,
                    &k_hm,
                    &v_hm,
                    &table,
                    dims.hq,
                    dims.hkv,
                    dims.seq,
                    dims.d,
                    dims.g(),
                    dims.blk,
                    dims.nb(),
                )
                .unwrap();
                let oracle_full = flat(&fc["expected_context"]);
                let err = max_abs(&rust_full.context, &oracle_full);
                assert!(err <= 1e-5, "Rust FP32 full ref vs Python oracle: {err}");
                Case {
                    name: format!(
                        "tiny/{}+{}",
                        lc["name"].as_str().unwrap(),
                        fc["name"].as_str().unwrap()
                    ),
                    q,
                    q_full: fq,
                    latent_physical,
                    k_projection,
                    v_projection,
                    kp_head_major: flat(&lc["k_projection_gpu_head_major"]),
                    vp_head_major: flat(&lc["v_projection_gpu_head_major"]),
                    k_physical_head_major: k_hm,
                    v_physical_head_major: v_hm,
                    table: table.clone(),
                    oracle_latent_context: Some(oracle_latent),
                    oracle_full_context: Some(oracle_full),
                }
            })
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

    /// model_small inputs, built exactly as in the C1/R-TABLE harness (c1_rtable.rs).
    #[derive(Clone, Copy, PartialEq)]
    enum Inputs {
        /// The repository's deterministic sinusoidal values (as in the C1/R-TABLE harness).
        Uniform,
        /// Heavy-tailed latent values (Pareto-like, tail index 3, clamped at 40) with a few
        /// outlier tokens scaled x12, so scores span a wide range and softmax is sharp.
        HeavyTailed,
    }

    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn heavy_tailed_latent(seq: usize, l: usize, seed: u64) -> Vec<f32> {
        let outliers = [5usize, 1000, 4000, 20000];
        (0..seq * l)
            .map(|i| {
                let h = splitmix(seed ^ i as u64);
                let u = ((h >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
                let tail = (u.powf(-1.0 / 3.0) - 1.0).min(40.0);
                let mut v = (sign * 0.3 * tail) as f32;
                if outliers.contains(&(i / l)) {
                    v *= 12.0;
                }
                v
            })
            .collect()
    }

    fn model_case(dims: Dims, table: Vec<usize>, name: &str) -> Case {
        model_case_with(dims, table, name, Inputs::Uniform, 0)
    }

    fn model_case_with(
        dims: Dims,
        table: Vec<usize>,
        name: &str,
        inputs: Inputs,
        seed: u64,
    ) -> Case {
        let (hq, hkv, d, l, blk, seq) = (dims.hq, dims.hkv, dims.d, dims.l, dims.blk, dims.seq);
        let offset = seed as f32 * 0.37;
        let q = deterministic_values(hq * d, 0.011, -0.4 + offset);
        let logical_latent = match inputs {
            Inputs::Uniform => deterministic_values(seq * l, 0.007, -1.2 + offset),
            Inputs::HeavyTailed => heavy_tailed_latent(seq, l, 0x5eed ^ seed),
        };
        let k_projection = deterministic_values(l * hkv * d, 0.005, -0.7 + offset);
        let v_projection = deterministic_values(l * hkv * d, 0.006, 0.3 + offset);
        let mut latent_physical = vec![0.0f32; logical_latent.len()];
        for (logical, &physical) in table.iter().enumerate() {
            let (src, dst) = (logical * blk * l, physical * blk * l);
            latent_physical[dst..dst + blk * l]
                .copy_from_slice(&logical_latent[src..src + blk * l]);
        }
        let head_major = |canonical: &[f32]| {
            let mut out = vec![0.0f32; canonical.len()];
            for kv in 0..hkv {
                for li in 0..l {
                    for di in 0..d {
                        out[(kv * l + li) * d + di] = canonical[(li * hkv + kv) * d + di];
                    }
                }
            }
            out
        };
        let mut k_logical = vec![0.0f32; seq * hkv * d];
        let mut v_logical = vec![0.0f32; seq * hkv * d];
        for t in 0..seq {
            for kv in 0..hkv {
                for di in 0..d {
                    let (mut kv_k, mut kv_v) = (0.0f32, 0.0f32);
                    for li in 0..l {
                        let z = logical_latent[t * l + li];
                        let idx = (li * hkv + kv) * d + di;
                        kv_k += z * k_projection[idx];
                        kv_v += z * v_projection[idx];
                    }
                    k_logical[(t * hkv + kv) * d + di] = kv_k;
                    v_logical[(t * hkv + kv) * d + di] = kv_v;
                }
            }
        }
        let to_physical = |logical: &[f32]| {
            let mut physical = vec![0.0f32; dims.nb() * hkv * blk * d];
            for (lb, &pb) in table.iter().enumerate() {
                for kv in 0..hkv {
                    for off in 0..blk {
                        let src = ((lb * blk + off) * hkv + kv) * d;
                        let dst = ((pb * hkv + kv) * blk + off) * d;
                        physical[dst..dst + d].copy_from_slice(&logical[src..src + d]);
                    }
                }
            }
            physical
        };
        Case {
            name: name.into(),
            kp_head_major: head_major(&k_projection),
            vp_head_major: head_major(&v_projection),
            k_physical_head_major: to_physical(&k_logical),
            v_physical_head_major: to_physical(&v_logical),
            q_full: q.clone(),
            q,
            latent_physical,
            k_projection,
            v_projection,
            table,
            oracle_latent_context: None,
            oracle_full_context: None,
        }
    }

    fn gpu_temperature() -> Option<u32> {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }

    /// Runs every (active, splits) combination for a case; returns the worst errors.
    fn check_case(
        dims: Dims,
        case: &Case,
        actives: &[usize],
        splits: &[usize],
        st: &Arc<Stream>,
    ) -> (f64, f64, f64) {
        let dev = to_device(dims, case, st);
        let (mut worst_c3, mut worst_a3, mut worst_split) = (0.0f64, 0.0f64, 0.0f64);
        for &active in actives {
            if gpu_temperature().is_some_and(|t| t >= 91) {
                panic!("THERMAL_ABORT: GPU >= 91 C");
            }
            validate_launch(dims, &dev, &case.table, active);
            let (latent_ref, full_ref) = cpu_refs(dims, case, active);
            let mut c3_by_split = Vec::new();
            let mut a3_by_split = Vec::new();
            for &s in splits {
                let plan = plan_full(dims, s);
                let c3 = run_c3(dims, dev.views(), active, plan, Mode::Fast, st);
                let c3_acc = run_c3(dims, dev.views(), active, plan, Mode::Accurate, st);
                let e_c3_acc = max_abs(&c3_acc, &latent_ref.context);
                assert!(e_c3_acc <= ATOL, "C3-accurate mismatch {e_c3_acc} > {ATOL}");
                worst_c3 = worst_c3.max(e_c3_acc);
                let a3 = run_a3(dims, dev.views(), active, plan, st);
                let e_c3 = max_abs(&c3, &latent_ref.context);
                let e_a3 = max_abs(&a3, &full_ref.context);
                let pad = if dims.hq == 16 {
                    let a3p = run_a3_pad(dims, dev.views(), active, plan, st);
                    let e = max_abs(&a3p, &full_ref.context);
                    assert!(e <= ATOL, "A3-pad mismatch {e} > {ATOL}");
                    worst_a3 = worst_a3.max(e);
                    format!(" a3pad_vs_rust={e:.3e}")
                } else {
                    String::new()
                };
                println!(
                    "case={} active={active} splits={s} c3_vs_rust={e_c3:.3e} c3acc_vs_rust={e_c3_acc:.3e} a3_vs_rust={e_a3:.3e}{pad}",
                    case.name
                );
                assert!(e_c3 <= ATOL, "C3 mismatch {e_c3} > {ATOL}");
                assert!(e_a3 <= ATOL, "A3 mismatch {e_a3} > {ATOL}");
                worst_c3 = worst_c3.max(e_c3);
                worst_a3 = worst_a3.max(e_a3);
                c3_by_split.push(c3);
                a3_by_split.push(a3);
            }
            for i in 1..splits.len() {
                worst_split = worst_split
                    .max(max_abs(&c3_by_split[0], &c3_by_split[i]))
                    .max(max_abs(&a3_by_split[0], &a3_by_split[i]));
            }
            if active == dims.seq {
                if let Some(oracle) = &case.oracle_latent_context {
                    let c3 = &c3_by_split[0];
                    println!(
                        "case={} c3_vs_python_oracle_fp32={:.3e} (includes FP16 storage rounding)",
                        case.name,
                        max_abs(c3, oracle)
                    );
                }
                if let Some(oracle) = &case.oracle_full_context {
                    println!(
                        "case={} a3_vs_python_oracle_fp32={:.3e} (includes FP16 storage rounding)",
                        case.name,
                        max_abs(&a3_by_split[0], oracle)
                    );
                }
            }
        }
        (worst_c3, worst_a3, worst_split)
    }

    fn e3_suite(st: &Arc<Stream>) {
        let st = st.clone();

        let tiny = Dims {
            hq: 4,
            hkv: 2,
            d: 8,
            l: 8,
            blk: 2,
            seq: 8,
            dc: 8,
        };
        for case in tiny_cases(tiny) {
            let (c3, a3, split) = check_case(tiny, &case, &[1, 3, 4, 7, 8], &[1, 2], &st);
            println!(
                "TINY_CASE={} worst_c3={c3:.3e} worst_a3={a3:.3e} split_spread={split:.3e}",
                case.name
            );
            assert!(split <= ATOL, "split invariance {split}");
        }
        println!("E3_ORACLE_CHAIN_OK=1");
        println!("E3_TINY_C3_OK=1");
        println!("E3_TINY_A3_OK=1");

        let model = Dims {
            hq: 16,
            hkv: 4,
            d: 64,
            l: 32,
            blk: 16,
            seq: 1024,
            dc: 16,
        };
        let table: Vec<usize> = (0..model.nb())
            .map(|lb| (lb * 17 + 11) % model.nb())
            .collect();
        assert!(table.iter().enumerate().any(|(i, &p)| i != p));
        let case = model_case(model, table, "model_small/nonidentity");
        let (c3, a3, split) =
            check_case(model, &case, &[17, 129, 513, 1021, 1024], &[1, 4, 8], &st);
        println!("MODEL_CASE worst_c3={c3:.3e} worst_a3={a3:.3e} split_spread={split:.3e}");
        assert!(split <= ATOL, "split invariance {split}");
        println!("E3_MODEL_C3_OK=1");
        println!("E3_MODEL_A3_OK=1");
        println!("E3_PARTIAL_BLOCK_OK=1");
        println!("E3_SPLIT_INVARIANCE_OK=1");

        // Negative control: the same physical data read through the identity table must NOT match
        // the non-identity reference, proving the kernels gather through the table. A partial
        // active length is required: at full length both tables cover the same token set and
        // attention is order-invariant, so the outputs legitimately agree.
        let control_active = 129;
        let mut wrong = model_case(model, case.table.clone(), "model_small/identity_control");
        let dev_ok = to_device(model, &wrong, &st);
        let (latent_ref, full_ref) = cpu_refs(model, &wrong, control_active);
        wrong.table = (0..model.nb()).collect();
        let dev_wrong = Device {
            table: upload_table(&wrong.table, &st),
            ..dev_ok
        };
        let c3_wrong = run_c3(
            model,
            dev_wrong.views(),
            control_active,
            plan_full(model, 4),
            Mode::Fast,
            &st,
        );
        let a3_wrong = run_a3(
            model,
            dev_wrong.views(),
            control_active,
            plan_full(model, 4),
            &st,
        );
        let (e_c3, e_a3) = (
            max_abs(&c3_wrong, &latent_ref.context),
            max_abs(&a3_wrong, &full_ref.context),
        );
        println!(
            "identity_table_control active={control_active} c3_err={e_c3:.3e} a3_err={e_a3:.3e} (must exceed {ATOL})"
        );
        assert!(
            e_c3 > ATOL && e_a3 > ATOL,
            "identity-table control did not detect paging"
        );
        println!("E3_NON_IDENTITY_TABLE_OK=1");
        println!("E3_SPLITK_GPU_OK=1");
    }

    fn mean_abs(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()))
            .sum::<f64>()
            / a.len() as f64
    }

    fn thermal_guard() {
        if gpu_temperature().is_some_and(|t| t >= 91) {
            panic!("THERMAL_ABORT: GPU >= 91 C");
        }
    }

    /// FP16 round trip, as the kernels' `convert_tile` f32 -> f16 -> f32 does.
    fn f16r(x: f32) -> f32 {
        f16::from_f32(x).to_f32()
    }

    /// Matched-rounding online softmax (FP64 accumulation). Applies the same FP16 roundings the
    /// kernels apply -- FP16 P per two-block step relative to each split's running max -- with
    /// the same split/step structure, masking (-1e30 sentinel) and log-sum-exp merge. Scores use
    /// `k_rows`, accumulation uses `v_rows` (both may be the same latent rows for C3). Kernel
    /// error measured against this is attributable to accumulation order and tensor-core
    /// arithmetic, not to the chosen roundings.
    #[allow(clippy::too_many_arguments)]
    fn matched_online(
        q_rows: &[Vec<f64>],
        k_rows: &dyn Fn(usize) -> Vec<f64>,
        v_rows: &dyn Fn(usize) -> Vec<f64>,
        width: usize,
        table: &[usize],
        blk: usize,
        active: usize,
        (splits, sps): (usize, usize),
        scale: f64,
    ) -> Vec<Vec<f64>> {
        let heads = q_rows.len();
        let tok = 2 * blk;
        let mut parts = Vec::new();
        for sidx in 0..splits {
            let mut m = vec![-1e30f64; heads];
            let mut l = vec![0f64; heads];
            let mut acc = vec![vec![0f64; width]; heads];
            for step in 0..sps {
                let b0 = (sidx * sps + step) * 2;
                let rows: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..tok)
                    .map(|j| {
                        let phys = table[b0 + j / blk] * blk + j % blk;
                        (b0 * blk + j, k_rows(phys), v_rows(phys))
                    })
                    .collect();
                for h in 0..heads {
                    let scores: Vec<f64> = rows
                        .iter()
                        .map(|(t, kr, _)| {
                            if *t < active {
                                q_rows[h].iter().zip(kr).map(|(a, b)| a * b).sum::<f64>() * scale
                            } else {
                                -1e30
                            }
                        })
                        .collect();
                    let m_new = scores.iter().cloned().fold(m[h], f64::max);
                    let alpha = (m[h] - m_new).exp();
                    let p: Vec<f64> = rows
                        .iter()
                        .zip(&scores)
                        .map(|((t, _, _), s)| if *t < active { (s - m_new).exp() } else { 0.0 })
                        .collect();
                    l[h] = l[h] * alpha + p.iter().sum::<f64>();
                    for a in acc[h].iter_mut() {
                        *a *= alpha;
                    }
                    for ((_, _, vr), pv) in rows.iter().zip(&p) {
                        let p16 = f64::from(f16r(*pv as f32));
                        for (a, x) in acc[h].iter_mut().zip(vr) {
                            *a += p16 * x;
                        }
                    }
                    m[h] = m_new;
                }
            }
            parts.push((m, l, acc));
        }
        (0..heads)
            .map(|h| {
                let big_m = parts.iter().map(|(m, _, _)| m[h]).fold(-1e30f64, f64::max);
                let mut big_l = 0f64;
                let mut out = vec![0f64; width];
                for (m, l, acc) in &parts {
                    let w = (m[h] - big_m).exp();
                    big_l += w * l[h];
                    for (o, a) in out.iter_mut().zip(&acc[h]) {
                        *o += w * a;
                    }
                }
                out.iter().map(|x| x / big_l).collect()
            })
            .collect()
    }

    /// Matched-rounding reference for C3: q' in FP32 as the kernel computes it, then FP16 hi
    /// (plus FP16 lo in accurate mode); FP16-stored latent; FP16 P; V projection in FP64.
    fn matched_c3(
        dims: Dims,
        case: &Case,
        active: usize,
        plan: (usize, usize),
        mode: Mode,
    ) -> Vec<f32> {
        let (hq, hkv, d, l, g) = (dims.hq, dims.hkv, dims.d, dims.l, dims.g());
        let z: Vec<f32> = case.latent_physical.iter().map(|&x| f16r(x)).collect();
        let q_rows: Vec<Vec<f64>> = (0..hq)
            .map(|h| {
                (0..l)
                    .map(|li| {
                        let qp: f32 = (0..d)
                            .map(|di| {
                                case.q[h * d + di] * case.k_projection[(li * hkv + h / g) * d + di]
                            })
                            .sum();
                        let hi = f16r(qp);
                        let lo = f16r(qp - hi);
                        f64::from(hi)
                            + if mode == Mode::Accurate {
                                f64::from(lo)
                            } else {
                                0.0
                            }
                    })
                    .collect()
            })
            .collect();
        let rows = |phys: usize| -> Vec<f64> {
            z[phys * l..(phys + 1) * l]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        };
        let ctx = matched_online(
            &q_rows,
            &rows,
            &rows,
            l,
            &case.table,
            dims.blk,
            active,
            plan,
            f64::from(dims.scale()),
        );
        let mut out = vec![0f32; hq * d];
        for h in 0..hq {
            for di in 0..d {
                let v: f64 = (0..l)
                    .map(|li| {
                        ctx[h][li] * f64::from(case.v_projection[(li * hkv + h / g) * d + di])
                    })
                    .sum();
                out[h * d + di] = v as f32;
            }
        }
        out
    }

    /// Matched-rounding reference for padded A3: FP16 q (padding rows do not affect real rows),
    /// FP16-stored K and V, FP16 P.
    fn matched_a3(dims: Dims, case: &Case, active: usize, plan: (usize, usize)) -> Vec<f32> {
        let (hq, hkv, d, g, blk) = (dims.hq, dims.hkv, dims.d, dims.g(), dims.blk);
        let k: Vec<f32> = case
            .k_physical_head_major
            .iter()
            .map(|&x| f16r(x))
            .collect();
        let v: Vec<f32> = case
            .v_physical_head_major
            .iter()
            .map(|&x| f16r(x))
            .collect();
        let mut out = vec![0f32; hq * d];
        for kv in 0..hkv {
            let q_rows: Vec<Vec<f64>> = (0..g)
                .map(|j| {
                    (0..d)
                        .map(|di| f64::from(f16r(case.q_full[(kv * g + j) * d + di])))
                        .collect()
                })
                .collect();
            // Physical layout (block, kv_head, token, dim); `phys` = block * blk + token.
            let row = |src: &[f32], phys: usize| -> Vec<f64> {
                let (b, t) = (phys / blk, phys % blk);
                let base = ((b * hkv + kv) * blk + t) * d;
                src[base..base + d].iter().map(|&x| f64::from(x)).collect()
            };
            let k_rows = |phys: usize| row(&k, phys);
            let v_rows = |phys: usize| row(&v, phys);
            let ctx = matched_online(
                &q_rows,
                &k_rows,
                &v_rows,
                d,
                &case.table,
                blk,
                active,
                plan,
                f64::from(dims.scale()),
            );
            for j in 0..g {
                for di in 0..d {
                    out[(kv * g + j) * d + di] = ctx[j][di] as f32;
                }
            }
        }
        out
    }

    /// Step 3: correctness at scale with bounded split plans (`plan_bounded`: at most
    /// MAX_STEPS_PER_CTA steps per CTA) for C3 fast, C3 accurate and padded A3 on identical
    /// inputs. Reports, per case, total error vs the exact FP32-arithmetic reference and
    /// kernel-attributable error vs the matched-rounding reference. Gate: C3 accurate total
    /// <= 3e-3 (stop after reporting every case if violated); C3 fast is reported, not gated.
    fn scale_suite(st: &Arc<Stream>) {
        const GATE: f64 = 3e-3;
        let dims = Dims {
            hq: 16,
            hkv: 4,
            d: 64,
            l: 32,
            blk: 16,
            seq: 32768,
            dc: 16,
        };
        let nb = dims.nb();
        let table: Vec<usize> = (0..nb).map(|lb| (lb * 17 + 11) % nb).collect();
        let mut failures = Vec::new();
        for (inputs, label) in [
            (Inputs::Uniform, "uniform"),
            (Inputs::HeavyTailed, "heavy_tailed"),
        ] {
            let case = model_case_with(dims, table.clone(), label, inputs, 0);
            let dev = to_device(dims, &case, st);
            for active in [4096usize, 8192, 32768] {
                thermal_guard();
                validate_launch(dims, &dev, &case.table, active);
                let plan = plan_bounded(active, dims.blk, nb, 1).expect("plan");
                let pl = (plan.splits, plan.steps_per_split);
                let (latent_ref, full_ref) = cpu_refs(dims, &case, active);
                let runs = [
                    (
                        "c3_fast",
                        run_c3(dims, dev.views(), active, pl, Mode::Fast, st),
                        &latent_ref.context,
                        matched_c3(dims, &case, active, pl, Mode::Fast),
                    ),
                    (
                        "c3_accurate",
                        run_c3(dims, dev.views(), active, pl, Mode::Accurate, st),
                        &latent_ref.context,
                        matched_c3(dims, &case, active, pl, Mode::Accurate),
                    ),
                    (
                        "a3pad",
                        run_a3_pad(dims, dev.views(), active, pl, st),
                        &full_ref.context,
                        matched_a3(dims, &case, active, pl),
                    ),
                ];
                for (name, out, exact, matched) in &runs {
                    let (t_max, t_mean) = (max_abs(out, exact), mean_abs(out, exact));
                    let (k_max, k_mean) = (max_abs(out, matched), mean_abs(out, matched));
                    let r_max = max_abs(matched, exact);
                    println!(
                        "SCALE inputs={label} N={active} splits={} steps_per_split={} {name} total_max={t_max:.3e} total_mean={t_mean:.3e} kernel_max={k_max:.3e} kernel_mean={k_mean:.3e} rounding_only_max={r_max:.3e}",
                        plan.splits, plan.steps_per_split
                    );
                    if *name == "c3_accurate" && t_max > GATE {
                        failures.push(format!(
                            "{label} N={active} c3_accurate total_max={t_max:.3e}"
                        ));
                    }
                }
            }
        }
        if !failures.is_empty() {
            println!("SCALE_GATE_FAILED gate={GATE}: {}", failures.join("; "));
            panic!("C3 accurate exceeded the {GATE} gate");
        }
        println!("E3_SCALE_OK=1");
    }

    /// One sequence of an E4 batch: its query, its block table into the shared pool, its active
    /// length and its bounded launch plan (identical across layers).
    struct E4Seq {
        q: Vec<f32>,
        q_dev: Tensor<f32>,
        table: Vec<usize>,
        t_dev: Tensor<i32>,
        active: usize,
        plan: SplitPlan,
    }

    struct E4Setup {
        seq_dims: Dims,
        layers: usize,
        pool_blocks: usize,
        pools: Vec<(Case, Device)>,
        seqs: Vec<E4Seq>,
    }

    /// E4 configuration: L = max(4, ceil(64 / B)) layers of resident per-layer pools (1 MiB of
    /// K+V per sequence per layer, so A3 pools total >= 64 MiB); every sequence has its own
    /// non-identity table into the pool, its own query and active length (b = 0 runs the full
    /// 1024 tokens). Tables and plans are validated on the host here, before any launch.
    fn e4_setup(batch: usize, st: &Arc<Stream>) -> E4Setup {
        const MIN_SPLITS: usize = 4;
        let seq_dims = Dims {
            hq: 16,
            hkv: 4,
            d: 64,
            l: 32,
            blk: 16,
            seq: 1024,
            dc: 16,
        };
        let nb = seq_dims.nb();
        let layers = 4usize.max(64usize.div_ceil(batch));
        let pool_blocks = batch * nb;
        let pool_dims = Dims {
            seq: batch * seq_dims.seq,
            ..seq_dims
        };
        let pool_table: Vec<usize> = (0..pool_blocks)
            .map(|lb| (lb * 1021 + 11) % pool_blocks)
            .collect();
        let pools: Vec<(Case, Device)> = (0..layers)
            .map(|layer| {
                let case = model_case_with(
                    pool_dims,
                    pool_table.clone(),
                    &format!("e4/B{batch}/L{layer}"),
                    Inputs::Uniform,
                    layer as u64 + 1,
                );
                let dev = to_device(pool_dims, &case, st);
                (case, dev)
            })
            .collect();
        let seqs = (0..batch)
            .map(|b| {
                let active = if b == 0 {
                    seq_dims.seq
                } else {
                    1 + (b * 373) % seq_dims.seq
                };
                let table: Vec<usize> = pool_table[b * nb..(b + 1) * nb].to_vec();
                let plan = plan_bounded(active, seq_dims.blk, nb, MIN_SPLITS)
                    .unwrap_or_else(|e| panic!("E4 plan rejected: {e}"));
                let table_i32: Vec<i32> = table.iter().map(|&v| v as i32).collect();
                validate_block_table(
                    &table_i32,
                    active,
                    seq_dims.blk,
                    plan.blocks_read,
                    pool_blocks,
                )
                .unwrap_or_else(|e| panic!("E4 table rejected: {e}"));
                let q =
                    deterministic_values(seq_dims.hq * seq_dims.d, 0.011, -0.4 + 0.13 * b as f32);
                E4Seq {
                    q_dev: upload_f32(&q, &[seq_dims.hq, seq_dims.d], st),
                    t_dev: upload_table(&table, st),
                    q,
                    table,
                    active,
                    plan,
                }
            })
            .collect();
        E4Setup {
            seq_dims,
            layers,
            pool_blocks,
            pools,
            seqs,
        }
    }

    fn e4_views<'a>(dev: &'a Device, seq: &'a E4Seq) -> Views<'a> {
        Views {
            q: &seq.q_dev,
            q_full: &seq.q_dev,
            table: &seq.t_dev,
            ..dev.views()
        }
    }

    /// Analytical bytes one sequence's launch pair reads or writes, per variant:
    /// (c3, a3pad, cache-only minimum c3, cache-only minimum a3).
    fn e4_bytes(plan: SplitPlan, active: usize) -> (u64, u64, u64, u64) {
        let (hq, hkv, d, l, blk) = (16u64, 4u64, 64u64, 32u64, 16u64);
        let (s, br) = (plan.splits as u64, plan.blocks_read as u64);
        let table_bytes = br * 4;
        let c3_partials = 2 * s * hq * (l + 2) * 4; // written by the split kernel, read by reduce
        let c3 = br * blk * l * 2
            + hq * d * 4
            + s * hkv * l * d * 4
            + hkv * l * d * 4
            + c3_partials
            + table_bytes
            + hq * d * 4;
        let a3_partials = 2 * s * hkv * 16 * (d + 2) * 4;
        let a3 = br * blk * hkv * d * 2 * 2
            + s * hkv * hq * d * 4
            + a3_partials
            + table_bytes
            + hq * d * 4;
        let active_blocks = active.div_ceil(16) as u64;
        (
            c3,
            a3,
            active_blocks * blk * l * 2,
            active_blocks * blk * hkv * d * 2 * 2,
        )
    }

    /// Step 4 (E4, correctness only): B in {1, 8, 32}, rotating per-layer pools (see
    /// `e4_setup`); the latent pools are smaller than 64 MiB, so a 64 MiB L2-flush memset runs
    /// before every layer. Every batch element is validated against per-sequence references over
    /// the shared pool (C3 accurate and padded A3). Bytes are analytical, not timing.
    fn e4_suite(st: &Arc<Stream>) {
        let flush = api::zeros::<f32>(&[L2_FLUSH_WORDS]).sync_on(st).unwrap();
        for batch in [1usize, 8, 32] {
            let setup = e4_setup(batch, st);
            let (mut c3_b, mut a3_b, mut min_c3, mut min_a3) = (0u64, 0u64, 0u64, 0u64);
            let (mut worst_c3, mut worst_a3) = (0.0f64, 0.0f64);
            for (case, dev) in &setup.pools {
                thermal_guard();
                memset_d32(flush.device_pointer().cu_deviceptr(), L2_FLUSH_WORDS, st);
                for seq in &setup.seqs {
                    let views = e4_views(dev, seq);
                    let pl = (seq.plan.splits, seq.plan.steps_per_split);
                    let c3 = run_c3(setup.seq_dims, views, seq.active, pl, Mode::Accurate, st);
                    let a3 = run_a3_pad(setup.seq_dims, views, seq.active, pl, st);
                    let seq_case = e4_seq_case(case, seq);
                    let (latent_ref, full_ref) =
                        cpu_refs_pool(setup.seq_dims, &seq_case, seq.active, setup.pool_blocks);
                    let (e_c3, e_a3) = (
                        max_abs(&c3, &latent_ref.context),
                        max_abs(&a3, &full_ref.context),
                    );
                    assert!(e_c3 <= ATOL, "E4 C3 B={batch} err {e_c3}");
                    assert!(e_a3 <= ATOL, "E4 A3 B={batch} err {e_a3}");
                    worst_c3 = worst_c3.max(e_c3);
                    worst_a3 = worst_a3.max(e_a3);
                    let (c, a, mc, ma) = e4_bytes(seq.plan, seq.active);
                    c3_b += c;
                    a3_b += a;
                    min_c3 += mc;
                    min_a3 += ma;
                }
            }
            let mib = |x: u64| x as f64 / (1u64 << 20) as f64;
            let pools = (setup.layers * setup.pool_blocks) as u64;
            println!(
                "E4 B={batch} layers={} pools_resident: a3={:.1}MiB c3={:.1}MiB (+64MiB L2 flush per layer) worst_err c3_accurate={worst_c3:.3e} a3pad={worst_a3:.3e}",
                setup.layers,
                mib(pools * 4 * 16 * 64 * 2 * 2),
                mib(pools * 16 * 32 * 2)
            );
            println!(
                "E4_BYTES B={batch} per_decode_step(all layers): c3={:.2}MiB a3pad={:.2}MiB ratio_a3_over_c3={:.2} | cache-only minimum: c3={:.2}MiB a3={:.2}MiB ratio={:.2}",
                mib(c3_b),
                mib(a3_b),
                a3_b as f64 / c3_b as f64,
                mib(min_c3),
                mib(min_a3),
                min_a3 as f64 / min_c3 as f64
            );
        }
        println!("E4_BATCH_LAYERS_OK=1");
    }

    fn e4_seq_case(case: &Case, seq: &E4Seq) -> Case {
        Case {
            name: case.name.clone(),
            q: seq.q.clone(),
            q_full: seq.q.clone(),
            latent_physical: case.latent_physical.clone(),
            k_projection: case.k_projection.clone(),
            v_projection: case.v_projection.clone(),
            kp_head_major: Vec::new(),
            vp_head_major: Vec::new(),
            k_physical_head_major: case.k_physical_head_major.clone(),
            v_physical_head_major: case.v_physical_head_major.clone(),
            table: seq.table.clone(),
            oracle_latent_context: None,
            oracle_full_context: None,
        }
    }

    /// 64 MiB of 32-bit words: larger than the 32 MiB (RTX 4060 Laptop) and 48 MB (L4) L2.
    const L2_FLUSH_WORDS: usize = 16 * 1024 * 1024;

    // ---------------------------------------------------------------- timing (cloud only) ---
    //
    // Timing protocol, same as the fixed C1 harness: preallocated buffers, async launches
    // bracketed by CUDA events on one stream (split + reduce inside one interval), per-sample
    // NVML clocks, variant order rotated by (iteration + process), JIT and warm-up outside the
    // intervals, correctness checked before timing, all file output after timing, 91 C abort.
    // These suites are intended for a stable-clock host (scripts/e0b/run_e0b_portable.sh);
    // `--iterations 0` exercises setup, correctness and output without taking samples.

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Variant {
        A3Pad,
        C3Fast,
        C3Accurate,
    }

    impl Variant {
        fn name(self) -> &'static str {
            match self {
                Self::A3Pad => "a3pad",
                Self::C3Fast => "c3_fast",
                Self::C3Accurate => "c3_accurate",
            }
        }
    }

    /// Preallocated split partials and output for one (variant, plan).
    struct TBuf {
        acc: Option<Tensor<f32>>,
        m: Option<Tensor<f32>>,
        l: Option<Tensor<f32>>,
        out: Option<Tensor<f32>>,
    }

    fn tbuf(dims: Dims, variant: Variant, splits: usize, st: &Arc<Stream>) -> TBuf {
        let (rows, width) = match variant {
            Variant::A3Pad => (dims.hkv * 16, dims.d),
            _ => (dims.hq, dims.l),
        };
        TBuf {
            acc: Some(
                api::zeros::<f32>(&[splits, rows, width])
                    .sync_on(st)
                    .unwrap(),
            ),
            m: Some(api::zeros::<f32>(&[splits, rows, 1]).sync_on(st).unwrap()),
            l: Some(api::zeros::<f32>(&[splits, rows, 1]).sync_on(st).unwrap()),
            out: Some(api::zeros::<f32>(&[dims.hq, dims.d]).sync_on(st).unwrap()),
        }
    }

    /// Enqueues one variant's split + reduce on `st` without synchronizing.
    fn launch_async(
        dims: Dims,
        dev: Views,
        active: usize,
        (splits, sps): (usize, usize),
        variant: Variant,
        b: &mut TBuf,
        st: &Arc<Stream>,
    ) {
        assert!(sps <= MAX_STEPS_PER_CTA);
        let (hq, hkv, d, l, g) = (dims.hq, dims.hkv, dims.d, dims.l, dims.g());
        let (acc, m, lsum, out) = (
            b.acc.take().unwrap(),
            b.m.take().unwrap(),
            b.l.take().unwrap(),
            b.out.take().unwrap(),
        );
        let (acc, m, lsum, out) = match variant {
            Variant::A3Pad => {
                const MP: usize = 16;
                let (acc, m, lsum, _, _, _, _, _, _, _) = unsafe {
                    e3_kernels::a3_split_pad(
                        acc.partition([1, MP, d]),
                        m.partition([1, MP, 1]),
                        lsum.partition([1, MP, 1]),
                        dev.q_full,
                        dev.k_full,
                        dev.v_full,
                        dev.table,
                        active as i32,
                        sps as i32,
                        dims.scale(),
                    )
                    .generics(gens(&[hq, hkv, g, d, dims.blk, 2 * dims.blk, MP]))
                    .compile_options(CompileOptions::new().occupancy(4))
                    .async_on(st)
                    .expect("a3_split_pad launch")
                };
                let (acc, m, lsum) = (acc.unpartition(), m.unpartition(), lsum.unpartition());
                let (out, _, _, _, _) = unsafe {
                    e3_kernels::a3_reduce_pad(out.partition([g, d]), &acc, &m, &lsum, splits as i32)
                        .generics(gens(&[g, d, MP]))
                        .async_on(st)
                        .expect("a3_reduce_pad launch")
                };
                (acc, m, lsum, out.unpartition())
            }
            Variant::C3Fast | Variant::C3Accurate => {
                let accurate = usize::from(variant == Variant::C3Accurate);
                let (acc, m, lsum, _, _, _, _, _, _, _) = unsafe {
                    e3_kernels::c3_split(
                        acc.partition([1, hq, l]),
                        m.partition([1, hq, 1]),
                        lsum.partition([1, hq, 1]),
                        dev.q,
                        dev.kp,
                        dev.latent,
                        dev.table,
                        active as i32,
                        sps as i32,
                        dims.scale(),
                    )
                    .generics(gens(&[
                        hq,
                        hkv,
                        g,
                        d,
                        l,
                        dims.blk,
                        2 * dims.blk,
                        dims.dc,
                        hkv * l,
                        accurate,
                    ]))
                    .async_on(st)
                    .expect("c3_split launch")
                };
                let (acc, m, lsum) = (acc.unpartition(), m.unpartition(), lsum.unpartition());
                let (out, _, _, _, _, _) = unsafe {
                    e3_kernels::c3_reduce(
                        out.partition([g, d]),
                        &acc,
                        &m,
                        &lsum,
                        dev.vp,
                        splits as i32,
                    )
                    .generics(gens(&[hq, g, d, l]))
                    .async_on(st)
                    .expect("c3_reduce launch")
                };
                (acc, m, lsum, out.unpartition())
            }
        };
        b.acc = Some(acc);
        b.m = Some(m);
        b.l = Some(lsum);
        b.out = Some(out);
    }

    fn readback(b: &TBuf, st: &Arc<Stream>) -> Vec<f32> {
        let t = b.out.as_ref().unwrap();
        let alias = unsafe { t.into_shared_alias() };
        (&alias).to_host_vec().sync_on(st).expect("readback")
    }

    struct TimingArgs {
        process: usize,
        iterations: usize,
        warmup: usize,
        clock_warm_ms: u64,
        dir: std::path::PathBuf,
    }

    fn timing_args(args: &[String]) -> TimingArgs {
        let arg = |key: &str, default: &str| {
            args.windows(2)
                .find(|w| w[0] == key)
                .map(|w| w[1].clone())
                .unwrap_or(default.into())
        };
        let dir = std::path::PathBuf::from(arg("--output-dir", "/tmp/plkv_e3_timing"));
        let absolute = std::path::absolute(&dir).unwrap();
        assert!(
            !absolute.starts_with("/mnt"),
            "--output-dir {absolute:?} is on /mnt; use a native filesystem path"
        );
        std::fs::create_dir_all(&dir).unwrap();
        TimingArgs {
            process: arg("--process", "0").parse().unwrap(),
            iterations: arg("--iterations", "50").parse().unwrap(),
            warmup: arg("--warmup", "10").parse().unwrap(),
            clock_warm_ms: arg("--clock-warm-ms", "2000").parse().unwrap(),
            dir,
        }
    }

    fn write_outputs(
        dir: &std::path::Path,
        samples: &[serde_json::Value],
        meta: &serde_json::Value,
        correctness: &serde_json::Value,
    ) {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::options()
                .write(true)
                .create_new(true)
                .open(dir.join("samples.jsonl"))
                .expect("samples.jsonl must not exist"),
        );
        for s in samples {
            writeln!(f, "{s}").unwrap();
        }
        f.flush().unwrap();
        std::fs::write(
            dir.join("timing_meta.json"),
            serde_json::to_string_pretty(meta).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("correctness.json"),
            serde_json::to_string_pretty(correctness).unwrap(),
        )
        .unwrap();
    }

    /// E3 timing: padded A3 vs C3 fast vs C3 accurate at one N (model_small shape, bounded
    /// plan), one process. Each sample = split + reduce of one variant.
    fn e3_timing_suite(args: &[String], st: &Arc<Stream>) {
        let ta = timing_args(args);
        let n: usize = args
            .windows(2)
            .find(|w| w[0] == "--seq")
            .map(|w| w[1].parse().unwrap())
            .unwrap_or(1024);
        let dims = Dims {
            hq: 16,
            hkv: 4,
            d: 64,
            l: 32,
            blk: 16,
            seq: n,
            dc: 16,
        };
        let nb = dims.nb();
        let table: Vec<usize> = (0..nb).map(|lb| (lb * 17 + 11) % nb).collect();
        let case = model_case(dims, table, "timing");
        let dev = to_device(dims, &case, st);
        validate_launch(dims, &dev, &case.table, n);
        let plan = plan_bounded(n, dims.blk, nb, 1).expect("plan");
        let pl = (plan.splits, plan.steps_per_split);
        let variants = [Variant::A3Pad, Variant::C3Fast, Variant::C3Accurate];
        let mut bufs: Vec<TBuf> = variants
            .iter()
            .map(|&v| tbuf(dims, v, plan.splits, st))
            .collect();
        // Correctness before timing, through the same async launch path.
        let (latent_ref, full_ref) = cpu_refs(dims, &case, n);
        let mut correctness = serde_json::Map::new();
        for (v, b) in variants.iter().zip(bufs.iter_mut()) {
            launch_async(dims, dev.views(), n, pl, *v, b, st);
            let out = readback(b, st);
            let reference = if *v == Variant::A3Pad {
                &full_ref.context
            } else {
                &latent_ref.context
            };
            let err = max_abs(&out, reference);
            assert!(err <= ATOL, "{} pre-timing correctness {err}", v.name());
            correctness.insert(
                v.name().into(),
                serde_json::json!({"max_abs_error": err, "atol": ATOL}),
            );
        }
        // JIT + fixed warm-up, then clock warm-up, all outside the timed intervals.
        for _ in 0..ta.warmup {
            for (v, b) in variants.iter().zip(bufs.iter_mut()) {
                launch_async(dims, dev.views(), n, pl, *v, b, st);
            }
        }
        let nvml = Nvml::open();
        let (reference_clock, warm_clocks) =
            clock_warmup(nvml.as_ref(), st, ta.clock_warm_ms, || {
                for (v, b) in variants.iter().zip(bufs.iter_mut()) {
                    launch_async(dims, dev.views(), n, pl, *v, b, st);
                }
            });
        let ev = Events::new();
        let mut samples = Vec::new();
        let mut aborted = None;
        let t0 = std::time::Instant::now();
        'timing: for it in 0..ta.iterations {
            let empty_pre = nvml.as_ref().and_then(Nvml::sm_clock);
            let empty_ms = ev.measure(st, || {});
            let mut row = clock_fields(nvml.as_ref(), empty_pre);
            row.as_object_mut().unwrap().extend(
                serde_json::json!({"suite":"e3_timing","seq":n,"process":ta.process,"iteration":it,"order":0,"variant":"NONE","component":"empty","latency_ms":empty_ms})
                    .as_object()
                    .unwrap()
                    .clone(),
            );
            samples.push(row);
            for off in 0..variants.len() {
                let idx = (it + ta.process + off) % variants.len();
                let pre = nvml.as_ref().and_then(Nvml::sm_clock);
                let ms = ev.measure(st, || {
                    launch_async(dims, dev.views(), n, pl, variants[idx], &mut bufs[idx], st)
                });
                let mut row = clock_fields(nvml.as_ref(), pre);
                let temp = row["temperature_c"].as_u64();
                row.as_object_mut().unwrap().extend(
                    serde_json::json!({"suite":"e3_timing","seq":n,"process":ta.process,"iteration":it,"order":off,"variant":variants[idx].name(),"component":"pipeline","latency_ms":ms,"t_ms":t0.elapsed().as_secs_f64()*1e3,"splits":plan.splits,"steps_per_split":plan.steps_per_split})
                        .as_object()
                        .unwrap()
                        .clone(),
                );
                samples.push(row);
                if temp.is_some_and(|t| t >= u64::from(ABORT_TEMPERATURE_C)) {
                    aborted = temp;
                    break 'timing;
                }
            }
        }
        let meta = serde_json::json!({"suite":"e3_timing","seq":n,"plan":{"splits":plan.splits,"steps_per_split":plan.steps_per_split,"blocks_read":plan.blocks_read},"nvml_available":nvml.is_some(),"reference_clock_mhz":reference_clock,"warm_clocks_mhz":warm_clocks,"iterations":ta.iterations,"warmup":ta.warmup,"variants":variants.iter().map(|v| v.name()).collect::<Vec<_>>(),"aborted_temperature_c":aborted,"label":"timing; label results with the host (e.g. cloud L4)"});
        write_outputs(
            &ta.dir,
            &samples,
            &meta,
            &serde_json::Value::Object(correctness),
        );
        if aborted.is_some() {
            eprintln!(
                "THERMAL_ABORT temperature >= {ABORT_TEMPERATURE_C} C; partial samples saved"
            );
            std::process::exit(3);
        }
        println!("E3_TIMING_OK=1 samples={}", samples.len());
    }

    /// E4 timing: one decode step = every layer x every batch element for one variant. Each
    /// layer's B launch pairs form one timed interval; a 64 MiB L2-flush memset runs before each
    /// layer outside the interval. The per-step total is the sum of that iteration's layer
    /// intervals (a per-iteration sum, never a sum of medians).
    fn e4_timing_suite(args: &[String], st: &Arc<Stream>) {
        let ta = timing_args(args);
        let batch: usize = args
            .windows(2)
            .find(|w| w[0] == "--batch")
            .map(|w| w[1].parse().unwrap())
            .unwrap_or(1);
        let setup = e4_setup(batch, st);
        let dims = setup.seq_dims;
        let variants = [Variant::A3Pad, Variant::C3Accurate];
        let flush = api::zeros::<f32>(&[L2_FLUSH_WORDS]).sync_on(st).unwrap();
        // Per (variant, sequence) buffers, reused across layers.
        let mut bufs: Vec<Vec<TBuf>> = variants
            .iter()
            .map(|&v| {
                setup
                    .seqs
                    .iter()
                    .map(|s| tbuf(dims, v, s.plan.splits, st))
                    .collect()
            })
            .collect();
        // Correctness before timing: layer 0, every batch element, both variants.
        let (case0, dev0) = &setup.pools[0];
        let mut worst = [0.0f64; 2];
        for (vi, v) in variants.iter().enumerate() {
            for (si, seq) in setup.seqs.iter().enumerate() {
                let pl = (seq.plan.splits, seq.plan.steps_per_split);
                launch_async(
                    dims,
                    e4_views(dev0, seq),
                    seq.active,
                    pl,
                    *v,
                    &mut bufs[vi][si],
                    st,
                );
                let out = readback(&bufs[vi][si], st);
                let seq_case = e4_seq_case(case0, seq);
                let (latent_ref, full_ref) =
                    cpu_refs_pool(dims, &seq_case, seq.active, setup.pool_blocks);
                let reference = if *v == Variant::A3Pad {
                    &full_ref.context
                } else {
                    &latent_ref.context
                };
                let err = max_abs(&out, reference);
                assert!(
                    err <= ATOL,
                    "E4 {} b={si} pre-timing correctness {err}",
                    v.name()
                );
                worst[vi] = worst[vi].max(err);
            }
        }
        let run_layer = |vi: usize, dev: &Device, bufs: &mut Vec<Vec<TBuf>>| {
            for (si, seq) in setup.seqs.iter().enumerate() {
                let pl = (seq.plan.splits, seq.plan.steps_per_split);
                launch_async(
                    dims,
                    e4_views(dev, seq),
                    seq.active,
                    pl,
                    variants[vi],
                    &mut bufs[vi][si],
                    st,
                );
            }
        };
        for _ in 0..ta.warmup {
            for vi in 0..variants.len() {
                for (_, dev) in &setup.pools {
                    run_layer(vi, dev, &mut bufs);
                }
            }
        }
        let nvml = Nvml::open();
        let (reference_clock, warm_clocks) =
            clock_warmup(nvml.as_ref(), st, ta.clock_warm_ms, || {
                for vi in 0..variants.len() {
                    run_layer(vi, &setup.pools[0].1, &mut bufs);
                }
            });
        let ev = Events::new();
        let mut samples = Vec::new();
        let mut aborted = None;
        'timing: for it in 0..ta.iterations {
            for off in 0..variants.len() {
                let vi = (it + ta.process + off) % variants.len();
                let mut step_ms = 0.0;
                for (layer, (_, dev)) in setup.pools.iter().enumerate() {
                    memset_d32(flush.device_pointer().cu_deviceptr(), L2_FLUSH_WORDS, st);
                    let pre = nvml.as_ref().and_then(Nvml::sm_clock);
                    let ms = ev.measure(st, || run_layer(vi, dev, &mut bufs));
                    step_ms += ms;
                    let mut row = clock_fields(nvml.as_ref(), pre);
                    let temp = row["temperature_c"].as_u64();
                    row.as_object_mut().unwrap().extend(
                        serde_json::json!({"suite":"e4_timing","batch":batch,"process":ta.process,"iteration":it,"order":off,"variant":variants[vi].name(),"component":"layer","layer":layer,"latency_ms":ms})
                            .as_object()
                            .unwrap()
                            .clone(),
                    );
                    samples.push(row);
                    if temp.is_some_and(|t| t >= u64::from(ABORT_TEMPERATURE_C)) {
                        aborted = temp;
                        break 'timing;
                    }
                }
                samples.push(serde_json::json!({"suite":"e4_timing","batch":batch,"process":ta.process,"iteration":it,"order":off,"variant":variants[vi].name(),"component":"step_total","latency_ms":step_ms,"layers":setup.layers}));
            }
        }
        let (mut c3_b, mut a3_b) = (0u64, 0u64);
        for seq in &setup.seqs {
            let (c, a, _, _) = e4_bytes(seq.plan, seq.active);
            c3_b += c * setup.layers as u64;
            a3_b += a * setup.layers as u64;
        }
        let meta = serde_json::json!({"suite":"e4_timing","batch":batch,"layers":setup.layers,"pool_blocks":setup.pool_blocks,"l2_flush_bytes":L2_FLUSH_WORDS*4,"bytes_per_step_analytical":{"c3_accurate":c3_b,"a3pad":a3_b},"nvml_available":nvml.is_some(),"reference_clock_mhz":reference_clock,"warm_clocks_mhz":warm_clocks,"iterations":ta.iterations,"warmup":ta.warmup,"variants":variants.iter().map(|v| v.name()).collect::<Vec<_>>(),"aborted_temperature_c":aborted});
        let correctness = serde_json::json!({"layer0_all_batch_elements":{"a3pad_max_abs_error":worst[0],"c3_accurate_max_abs_error":worst[1],"atol":ATOL}});
        write_outputs(&ta.dir, &samples, &meta, &correctness);
        if aborted.is_some() {
            eprintln!(
                "THERMAL_ABORT temperature >= {ABORT_TEMPERATURE_C} C; partial samples saved"
            );
            std::process::exit(3);
        }
        println!("E4_TIMING_OK=1 samples={}", samples.len());
    }

    /// CPU references for one sequence reading a shared pool of `pool_blocks` physical blocks.
    fn cpu_refs_pool(
        dims: Dims,
        case: &Case,
        active: usize,
        pool_blocks: usize,
    ) -> (GqaDecodeResult, GqaDecodeResult) {
        let latent_f16 = quantize_f32_to_f16_storage(&case.latent_physical).unwrap();
        let k_f16 = quantize_f32_to_f16_storage(&case.k_physical_head_major).unwrap();
        let v_f16 = quantize_f32_to_f16_storage(&case.v_physical_head_major).unwrap();
        let latent = direct_paged_latent_gqa_decode_fp16_storage_runtime_f32_accum(
            &case.q,
            &latent_f16,
            &case.table,
            &case.k_projection,
            &case.v_projection,
            dims.hq,
            dims.hkv,
            dims.seq,
            active,
            dims.l,
            dims.d,
            dims.g(),
            dims.blk,
            pool_blocks,
        )
        .expect("latent CPU reference (pool)");
        let full = paged_full_kv_gqa_decode_fp16_storage_runtime_f32_accum(
            &case.q_full,
            &k_f16,
            &v_f16,
            &case.table,
            dims.hq,
            dims.hkv,
            dims.seq,
            active,
            dims.d,
            dims.g(),
            dims.blk,
            pool_blocks,
        )
        .expect("full CPU reference (pool)");
        (latent, full)
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let suite = args
            .windows(2)
            .find(|w| w[0] == "--suite")
            .map(|w| w[1].clone())
            .unwrap_or_else(|| "e3".into());
        let st = with_default_device_policy(|p| p.next_stream())
            .unwrap()
            .unwrap();
        st.device().bind_to_thread().unwrap();
        println!("E3_TEMPERATURE_START_C={:?}", gpu_temperature());
        if suite == "e3" || suite == "all" {
            e3_suite(&st);
        }
        if suite == "scale" || suite == "all" {
            scale_suite(&st);
        }
        if suite == "e4" || suite == "all" {
            e4_suite(&st);
        }
        // Timing suites: not part of "all"; intended for a stable-clock host.
        if suite == "timing" {
            e3_timing_suite(&args, &st);
        }
        if suite == "e4-timing" {
            e4_timing_suite(&args, &st);
        }
        println!("E3_TEMPERATURE_END_C={:?}", gpu_temperature());
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
