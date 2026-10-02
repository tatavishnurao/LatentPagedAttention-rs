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
mod gpu_impl {
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

    /// (splits, steps_per_split) covering only the active range rounded up to whole two-block
    /// steps; never reads past the table. Returns the logical blocks read as well.
    fn plan_active(dims: Dims, active: usize, max_splits: usize) -> (usize, usize, usize) {
        let steps_needed = active.div_ceil(dims.blk).div_ceil(2);
        let splits = max_splits.min(steps_needed).max(1);
        let sps = steps_needed.div_ceil(splits);
        let blocks_read = splits * sps * 2;
        assert!(blocks_read <= dims.nb(), "plan reads past the table");
        (splits, sps, blocks_read)
    }

    fn gens(values: &[usize]) -> Vec<String> {
        values.iter().map(usize::to_string).collect()
    }

    fn run_c3(
        dims: Dims,
        dev: Views,
        active: usize,
        (splits, sps): (usize, usize),
        st: &Arc<Stream>,
    ) -> Vec<f32> {
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
                let c3 = run_c3(dims, dev.views(), active, plan, st);
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
                    "case={} active={active} splits={s} c3_vs_rust={e_c3:.3e} a3_vs_rust={e_a3:.3e}{pad}",
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

    /// Step 3: correctness at scale (model_small shape, N up to 32768) for uniform and
    /// heavy-tailed inputs, several split counts. Reports max and mean |error| against the Rust
    /// FP32-arithmetic reference (FP16 storage). Stops (panics after reporting every case) if
    /// any case exceeds SCALE_TOL.
    fn scale_suite(st: &Arc<Stream>) {
        const SCALE_TOL: f64 = 3e-3;
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
        let mut exceeded = Vec::new();
        for (inputs, label) in [
            (Inputs::Uniform, "uniform"),
            (Inputs::HeavyTailed, "heavy_tailed"),
        ] {
            let case = model_case_with(dims, table.clone(), label, inputs, 0);
            let dev = to_device(dims, &case, st);
            for active in [4096usize, 8192, 32768] {
                thermal_guard();
                validate_launch(dims, &dev, &case.table, active);
                let (latent_ref, full_ref) = cpu_refs(dims, &case, active);
                let (mut worst, mut worst_mean) = ([0.0f64; 3], [0.0f64; 3]);
                for max_splits in [1usize, 8, 64] {
                    let (splits, sps, blocks) = plan_active(dims, active, max_splits);
                    let outs = [
                        run_c3(dims, dev.views(), active, (splits, sps), st),
                        run_a3(dims, dev.views(), active, (splits, sps), st),
                        run_a3_pad(dims, dev.views(), active, (splits, sps), st),
                    ];
                    let refs = [&latent_ref.context, &full_ref.context, &full_ref.context];
                    let mut line = format!(
                        "SCALE inputs={label} active={active} splits={splits} steps_per_split={sps} blocks_read={blocks}"
                    );
                    for (k, name) in ["c3", "a3", "a3pad"].iter().enumerate() {
                        let (mx, mn) = (max_abs(&outs[k], refs[k]), mean_abs(&outs[k], refs[k]));
                        worst[k] = worst[k].max(mx);
                        worst_mean[k] = worst_mean[k].max(mn);
                        line += &format!(" {name}_max={mx:.3e} {name}_mean={mn:.3e}");
                        if mx > SCALE_TOL {
                            exceeded.push(format!(
                                "{label} N={active} splits={splits} {name} max={mx:.3e}"
                            ));
                        }
                    }
                    println!("{line}");
                }
                println!(
                    "SCALE_SUMMARY inputs={label} active={active} worst_max c3={:.3e} a3={:.3e} a3pad={:.3e} worst_mean c3={:.3e} a3={:.3e} a3pad={:.3e}",
                    worst[0], worst[1], worst[2], worst_mean[0], worst_mean[1], worst_mean[2]
                );
            }
        }
        if !exceeded.is_empty() {
            println!(
                "SCALE_TOLERANCE_EXCEEDED tol={SCALE_TOL}: {}",
                exceeded.join("; ")
            );
            panic!("scale tolerance exceeded; FP16 probability path may need FP32 P accumulation");
        }
        println!("E3_SCALE_OK=1");
    }

    /// Step 4 (E4, correctness only): batch B in {1, 8, 32} over a shared physical block pool
    /// per layer, L layers of resident, rotating per-layer pools. A3 pools total >= 64 MiB by
    /// construction (L = max(4, ceil(64 / B)), 1 MiB K+V per sequence per layer); the latent
    /// pools are smaller, so a 64 MiB L2-flush allocation is written before every layer. Each
    /// sequence has its own non-identity table into the pool, its own query and its own active
    /// length; every batch element is validated. Bytes are analytical accounting, not timing.
    fn e4_suite(st: &Arc<Stream>) {
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
        const MAX_SPLITS: usize = 4;
        for batch in [1usize, 8, 32] {
            let layers = 4usize.max(64usize.div_ceil(batch));
            let pool_blocks = batch * nb;
            let pool_dims = Dims {
                seq: batch * seq_dims.seq,
                ..seq_dims
            };
            let pool_table: Vec<usize> = (0..pool_blocks)
                .map(|lb| (lb * 1021 + 11) % pool_blocks)
                .collect();
            let actives: Vec<usize> = (0..batch)
                .map(|b| {
                    if b == 0 {
                        seq_dims.seq
                    } else {
                        1 + (b * 373) % seq_dims.seq
                    }
                })
                .collect();
            // Resident, rotating per-layer pools.
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
            let (mut c3_bytes, mut a3_bytes, mut min_bytes_c3, mut min_bytes_a3) =
                (0u64, 0u64, 0u64, 0u64);
            let (mut worst_c3, mut worst_a3) = (0.0f64, 0.0f64);
            for (case, dev) in &pools {
                thermal_guard();
                // Explicit L2 flush: write a fresh 64 MiB buffer before each layer.
                let flush = api::zeros::<f32>(&[16 * 1024 * 1024]).sync_on(st).unwrap();
                drop(flush);
                for (b, &active) in actives.iter().enumerate() {
                    let table_b: Vec<usize> = case.table[b * nb..(b + 1) * nb].to_vec();
                    let (splits, sps, blocks_read) = plan_active(seq_dims, active, MAX_SPLITS);
                    let table_i32: Vec<i32> = table_b.iter().map(|&v| v as i32).collect();
                    validate_block_table(
                        &table_i32,
                        active,
                        seq_dims.blk,
                        blocks_read,
                        pool_blocks,
                    )
                    .unwrap_or_else(|e| panic!("E4 table rejected: {e}"));
                    let q = deterministic_values(
                        seq_dims.hq * seq_dims.d,
                        0.011,
                        -0.4 + 0.13 * b as f32,
                    );
                    let q_dev = upload_f32(&q, &[seq_dims.hq, seq_dims.d], st);
                    let t_dev = upload_table(&table_b, st);
                    let views = Views {
                        q: &q_dev,
                        q_full: &q_dev,
                        table: &t_dev,
                        ..dev.views()
                    };
                    let c3 = run_c3(seq_dims, views, active, (splits, sps), st);
                    let a3 = run_a3_pad(seq_dims, views, active, (splits, sps), st);
                    // Per-sequence references over the shared pool.
                    let seq_case = Case {
                        name: format!("{}/b{b}", case.name),
                        q: q.clone(),
                        q_full: q,
                        latent_physical: case.latent_physical.clone(),
                        k_projection: case.k_projection.clone(),
                        v_projection: case.v_projection.clone(),
                        kp_head_major: Vec::new(),
                        vp_head_major: Vec::new(),
                        k_physical_head_major: case.k_physical_head_major.clone(),
                        v_physical_head_major: case.v_physical_head_major.clone(),
                        table: table_b,
                        oracle_latent_context: None,
                        oracle_full_context: None,
                    };
                    let ref_dims = Dims {
                        seq: seq_dims.seq,
                        ..seq_dims
                    };
                    let (latent_ref, full_ref) =
                        cpu_refs_pool(ref_dims, &seq_case, active, pool_blocks);
                    let (e_c3, e_a3) = (
                        max_abs(&c3, &latent_ref.context),
                        max_abs(&a3, &full_ref.context),
                    );
                    assert!(e_c3 <= ATOL, "E4 C3 B={batch} b={b} err {e_c3}");
                    assert!(e_a3 <= ATOL, "E4 A3 B={batch} b={b} err {e_a3}");
                    worst_c3 = worst_c3.max(e_c3);
                    worst_a3 = worst_a3.max(e_a3);
                    // Analytical bytes for this launch pair (unique bytes each kernel reads or writes).
                    let (hq, hkv, d, l, blk) = (16u64, 4u64, 64u64, 32u64, 16u64);
                    let (s, br) = (splits as u64, blocks_read as u64);
                    let table_bytes = br * 4;
                    let c3_partials = 2 * s * hq * (l + 2) * 4; // written by split, read by reduce
                    c3_bytes += br * blk * l * 2
                        + hq * d * 4
                        + s * hkv * l * d * 4
                        + hkv * l * d * 4
                        + c3_partials
                        + table_bytes
                        + hq * d * 4;
                    let a3_partials = 2 * s * hkv * 16 * (d + 2) * 4;
                    a3_bytes += br * blk * hkv * d * 2 * 2
                        + s * hkv * hq * d * 4
                        + a3_partials
                        + table_bytes
                        + hq * d * 4;
                    let active_blocks = active.div_ceil(seq_dims.blk) as u64;
                    min_bytes_c3 += active_blocks * blk * l * 2;
                    min_bytes_a3 += active_blocks * blk * hkv * d * 2 * 2;
                }
            }
            let mib = |x: u64| x as f64 / (1u64 << 20) as f64;
            let a3_pool_mib = mib((layers * pool_blocks) as u64 * 4 * 16 * 64 * 2 * 2);
            let c3_pool_mib = mib((layers * pool_blocks) as u64 * 16 * 32 * 2);
            println!(
                "E4 B={batch} layers={layers} pools_resident: a3={a3_pool_mib:.1}MiB c3={c3_pool_mib:.1}MiB (+64MiB L2 flush per layer) worst_err c3={worst_c3:.3e} a3pad={worst_a3:.3e}"
            );
            println!(
                "E4_BYTES B={batch} per_decode_step(all layers): c3={:.2}MiB a3pad={:.2}MiB ratio_a3_over_c3={:.2} | cache-only minimum: c3={:.2}MiB a3={:.2}MiB ratio={:.2}",
                mib(c3_bytes),
                mib(a3_bytes),
                a3_bytes as f64 / c3_bytes as f64,
                mib(min_bytes_c3),
                mib(min_bytes_a3),
                min_bytes_a3 as f64 / min_bytes_c3 as f64
            );
        }
        println!("E4_BATCH_LAYERS_OK=1");
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
