//! Compile-only probe: does cuTile 0.2 lower the latent score GEMM to tensor-core MMA on sm_89?
//!
//! For T in {64, 128, 256, 512} this compiles three kernels with `KernelCompiler` (no GPU, no
//! launch), runs `tileiras --gpu-name <target>` on the bytecode, and counts SASS opcodes with
//! `cuobjdump -sass`:
//! - `scores_qz_f16`:  q' [16,32] x Z^T [32,T]  (FP16 in, FP32 acc) -- the GEMM as specified;
//! - `scores_zq_f16`:  Z [T,32] x q'^T [32,16]  (FP16 in, FP32 acc) -- natural paged-row layout;
//! - `scores_reduce_f32`: current kernel style (FP32 broadcast-multiply + reduce_sum), as control.
#[cfg(feature = "gpu-cutile")]
mod probe {
    use cutile::compile_api::KernelCompiler;
    use cutile::cutile_compiler::hints::CompileOptions;
    use cutile::cutile_compiler::specialization::{SpecializationBits, compute_spec};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    #[cutile::module]
    pub mod mma_probe_module {
        use cutile::core::*;

        #[cutile::entry()]
        pub fn scores_qz_f16<const T: i32>(
            out: &mut Tensor<f32, { [16, T] }>,
            q: &Tensor<f16, { [16, 32] }>,
            zt: &Tensor<f16, { [32, -1] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let q_tile: Tile<f16, { [16, 32] }> = q.load_tile(const_shape![16, 32], [0, 0]);
            let z_part = zt.partition(const_shape![32, T]);
            let z_tile: Tile<f16, { [32, T] }> = z_part.load([0i32, pid.1]);
            let acc: Tile<f32, { [16, T] }> = constant(0.0f32, const_shape![16, T]);
            out.store(mma(q_tile, z_tile, acc));
        }

        #[cutile::entry()]
        pub fn scores_zq_f16<const T: i32>(
            out: &mut Tensor<f32, { [T, 16] }>,
            z: &Tensor<f16, { [-1, 32] }>,
            qt: &Tensor<f16, { [32, 16] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let z_part = z.partition(const_shape![T, 32]);
            let z_tile: Tile<f16, { [T, 32] }> = z_part.load([pid.0, 0i32]);
            let q_tile: Tile<f16, { [32, 16] }> = qt.load_tile(const_shape![32, 16], [0, 0]);
            let acc: Tile<f32, { [T, 16] }> = constant(0.0f32, const_shape![T, 16]);
            out.store(mma(z_tile, q_tile, acc));
        }

        #[cutile::entry()]
        pub fn scores_reduce_f32<const T: i32>(
            out: &mut Tensor<f32, { [1, T] }>,
            q: &Tensor<f32, { [16, 32] }>,
            z: &Tensor<f32, { [-1, 32] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let q_row: Tile<f32, { [1, 32] }> = q.load_tile(const_shape![1, 32], [pid.0, 0]);
            let z_part = z.partition(const_shape![T, 32]);
            let z_tile: Tile<f32, { [T, 32] }> = z_part.load([pid.1, 0i32]);
            let dots: Tile<f32, { [T] }> =
                reduce_sum(z_tile * q_row.broadcast(const_shape![T, 32]), 1i32);
            out.store(dots.reshape(const_shape![1, T]));
        }
    }
    use mma_probe_module::__module_ast_self;

    const N_TOKENS: i32 = 8192; // size of the dynamic token dimension
    // A cudaMalloc-style base address: 256-byte aligned, so DivHint::from_ptr clamps to 16.
    const ALIGNED_BASE_PTR: u64 = 0x7f00_0000_0000;
    /// (parameter, full-tensor shape, element bytes); strides are row-major over the shape.
    type Param<'a> = (&'a str, Vec<i32>, i32);
    type KernelParams<'a> = (&'a str, Vec<Param<'a>>);
    /// P1 hint modes: none (baseline), CompileOptions::max_divisibility(16), spec_args
    /// computed like the runtime JIT does for aligned allocations, and both.
    const HINT_MODES: [&str; 4] = ["none", "maxdiv16", "spec", "spec+maxdiv16"];

    fn row_major_strides(shape: &[i32]) -> Vec<i32> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn sass_counts(cubin: &Path, cuobjdump: &Path) -> BTreeMap<String, usize> {
        let out = Command::new(cuobjdump)
            .arg("-sass")
            .arg(cubin)
            .output()
            .expect("cuobjdump failed to start");
        assert!(out.status.success(), "cuobjdump failed: {out:?}");
        let text = String::from_utf8_lossy(&out.stdout);
        let mut counts = BTreeMap::new();
        for line in text.lines() {
            // SASS lines look like: "        /*0040*/   HMMA.16816.F32 R4, R8, R12, R4 ;"
            let Some(rest) = line.trim_start().strip_prefix("/*") else {
                continue;
            };
            let Some((_, instr)) = rest.split_once("*/") else {
                continue;
            };
            let mut words = instr.split_whitespace();
            let mut op = words.next().unwrap_or("");
            if op.starts_with('@') {
                op = words.next().unwrap_or("");
            }
            if op.is_empty() || op.starts_with("/*") {
                continue;
            }
            let base = op.split('.').next().unwrap().to_string();
            // Keep full opcodes for memory instructions so access width (e.g. LDG.E.U16 vs
            // LDG.E.128) stays visible.
            if matches!(base.as_str(), "LDG" | "STG" | "LDS" | "STS" | "LDGSTS") {
                *counts.entry(format!("op:{op}")).or_insert(0) += 1;
            }
            *counts.entry(base).or_insert(0) += 1;
            *counts.entry("_total".into()).or_insert(0) += 1;
        }
        counts
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let arg = |key: &str, default: &str| {
            args.windows(2)
                .find(|w| w[0] == key)
                .map(|w| w[1].clone())
                .unwrap_or(default.into())
        };
        let target = arg("--target", "sm_89");
        let dir = PathBuf::from(arg("--output-dir", "/tmp/plkv_mma_probe"));
        std::fs::create_dir_all(&dir).unwrap();
        let toolkit =
            PathBuf::from(std::env::var("CUDA_TOOLKIT_PATH").unwrap_or("/usr/local/cuda".into()));
        let tileiras = std::env::var("CUTILE_TILEIRAS_PATH")
            .map(PathBuf::from)
            .unwrap_or(toolkit.join("bin/tileiras"));
        let cuobjdump = toolkit.join("bin/cuobjdump");
        let mut results = Vec::new();
        for t in [64, 128, 256, 512] {
            let n = N_TOKENS;
            let kernels: [KernelParams; 3] = [
                (
                    "scores_qz_f16",
                    vec![
                        ("out", vec![16, n], 4),
                        ("q", vec![16, 32], 2),
                        ("zt", vec![32, n], 2),
                    ],
                ),
                (
                    "scores_zq_f16",
                    vec![
                        ("out", vec![n, 16], 4),
                        ("z", vec![n, 32], 2),
                        ("qt", vec![32, 16], 2),
                    ],
                ),
                (
                    "scores_reduce_f32",
                    vec![
                        ("out", vec![16, n], 4),
                        ("q", vec![16, 32], 4),
                        ("z", vec![n, 32], 4),
                    ],
                ),
            ];
            for (name, params) in &kernels {
                let strides: Vec<Vec<i32>> = params
                    .iter()
                    .map(|(_, shape, _)| row_major_strides(shape))
                    .collect();
                let stride_refs: Vec<(&str, &[i32])> = params
                    .iter()
                    .zip(&strides)
                    .map(|((p, _, _), s)| (*p, s.as_slice()))
                    .collect();
                let specs: Vec<SpecializationBits> = params
                    .iter()
                    .zip(&strides)
                    .map(|((_, shape, bytes), s)| compute_spec(ALIGNED_BASE_PTR, shape, s, *bytes))
                    .collect();
                let spec_refs: Vec<(&str, SpecializationBits)> = params
                    .iter()
                    .zip(&specs)
                    .map(|((p, _, _), spec)| (*p, spec.clone()))
                    .collect();
                for mode in HINT_MODES {
                    let name = *name;
                    let tag = if mode == "none" {
                        format!("{name}_T{t}_{target}")
                    } else {
                        format!("{name}_T{t}_{target}_{}", mode.replace('+', "_"))
                    };
                    let mut compiler =
                        KernelCompiler::new(__module_ast_self, "mma_probe_module", name)
                            .generics(vec![t.to_string()])
                            .strides(&stride_refs)
                            .target(&target);
                    if mode.contains("spec") {
                        compiler = compiler.spec_args(&spec_refs);
                    }
                    if mode.contains("maxdiv16") {
                        compiler = compiler.options(CompileOptions::new().max_divisibility(16));
                    }
                    let artifacts = match compiler.compile() {
                        Ok(a) => a,
                        Err(e) => {
                            results.push(serde_json::json!({"kernel":name,"T":t,"target":target,"hint_mode":mode,"stage":"tile_ir_compile","error":format!("{e:?}")}));
                            continue;
                        }
                    };
                    // Tile IR spells the op `mmaf` (float) or `mmai` (integer); the module
                    // name also contains "mma".
                    let ir = artifacts.ir_text();
                    let ir_has_mma = ir.contains(" mmaf ")
                        || ir.contains(" mmai ")
                        || ir.contains(".mmaf")
                        || ir.contains(".mmai");
                    let bc = dir.join(format!("{tag}.bc"));
                    let cubin = dir.join(format!("{tag}.cubin"));
                    std::fs::write(&bc, artifacts.bytecode().expect("bytecode")).unwrap();
                    std::fs::write(dir.join(format!("{tag}.mlir")), &ir).unwrap();
                    let status = Command::new(&tileiras)
                        .args(["--gpu-name", &target, "--opt-level", "3", "-o"])
                        .arg(&cubin)
                        .arg(&bc)
                        .output()
                        .expect("tileiras failed to start");
                    if !status.status.success() {
                        results.push(serde_json::json!({"kernel":name,"T":t,"target":target,"hint_mode":mode,"stage":"tileiras","ir_has_mma":ir_has_mma,"error":String::from_utf8_lossy(&status.stderr)}));
                        continue;
                    }
                    let counts = sass_counts(&cubin, &cuobjdump);
                    let mma: usize = counts
                        .iter()
                        .filter(|(k, _)| !k.starts_with("op:") && k.contains("MMA"))
                        .map(|(_, v)| v)
                        .sum();
                    let loads: Vec<String> = counts
                        .iter()
                        .filter(|(k, _)| k.starts_with("op:LDG"))
                        .map(|(k, v)| format!("{}x{v}", &k[3..]))
                        .collect();
                    println!(
                        "{tag}: ir_mma={ir_has_mma} sass_mma={mma} HMMA={} total={} global_loads=[{}]",
                        counts.get("HMMA").unwrap_or(&0),
                        counts.get("_total").unwrap_or(&0),
                        loads.join(" ")
                    );
                    results.push(serde_json::json!({"kernel":name,"T":t,"target":target,"hint_mode":mode,"ir_has_mma":ir_has_mma,"sass_mma_instructions":mma,"opcode_counts":counts,"cubin":cubin.file_name().map(|f| f.to_string_lossy().into_owned())}));
                }
            }
        }
        std::fs::write(
            dir.join("mma_probe_summary.json"),
            serde_json::to_string_pretty(&results).unwrap(),
        )
        .unwrap();
        println!("MMA_PROBE_DONE dir={}", dir.display());
    }
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    // The cuTile compiler recurses deeply; mirror the upstream test stack size.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(probe::main)
        .unwrap()
        .join()
        .unwrap();
}
#[cfg(not(feature = "gpu-cutile"))]
fn main() {
    panic!("requires gpu-cutile");
}
