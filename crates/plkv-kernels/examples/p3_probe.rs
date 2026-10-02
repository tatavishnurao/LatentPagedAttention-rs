//! Probe P3 (compile-only, CPU): resource usage and load widths with and without the
//! `max_divisibility(16)` hint for (a) the repository's production A1/B1/C1 score, softmax and
//! context kernels and (b) a minimal block-table-gathered FP16 latent tile feeding `mma`.
//! No GPU, no CUDA context, no launches. Output: `p3_probe_summary.json` plus .mlir/.cubin.
#[cfg(feature = "gpu-cutile")]
#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "gpu-cutile")]
mod probe {
    use super::common::*;
    use cutile::cutile_compiler::ast::Module;
    use plkv_kernels::cutile::{c1, p15b_rtable_kernels as rt};
    use std::path::PathBuf;

    /// E3-shaped minimal kernel: each CTA walks R logical blocks; each block's 16x32 FP16 latent
    /// tile is gathered through the block table and used twice on the tensor cores:
    /// scores S = q' Z^T (16 heads x 16 tokens) and the PV-style update acc += P Z (16 x 32),
    /// with P = exp(S) cast to FP16 as a stand-in for softmax numerators.
    #[cutile::module]
    pub mod gather_mma_module {
        use cutile::core::*;

        #[cutile::entry()]
        pub fn latent_gather_mma<const R: i32>(
            out: &mut Tensor<f32, { [16, 32] }>,
            qp: &Tensor<f16, { [16, 32] }>,
            latent: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [-1] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let q: Tile<f16, { [16, 32] }> = qp.load_tile(const_shape![16, 32], [0, 0]);
            let mut acc: Tile<f32, { [16, 32] }> = constant(0.0f32, const_shape![16, 32]);
            for r in 0i32..R {
                let logical = pid.0 * R + r;
                let entry: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
                let physical = tile_to_scalar(entry.reshape(const_shape![]));
                let z: Tile<f16, { [16, 32] }> =
                    latent.load_tile(const_shape![16, 32], [physical, 0]);
                let zt: Tile<f16, { [32, 16] }> = z.transpose();
                let zero: Tile<f32, { [16, 16] }> = constant(0.0f32, const_shape![16, 16]);
                let s: Tile<f32, { [16, 16] }> = mma(q, zt, zero);
                let p: Tile<f16, { [16, 16] }> = convert_tile(exp(s));
                acc = mma(p, z, acc);
            }
            out.store(acc);
        }

        /// Split variant 1: only the score GEMM S = q' Z^T on the gathered tile (summed over
        /// blocks so the result is kept).
        #[cutile::entry()]
        pub fn latent_gather_scores_only<const R: i32>(
            out: &mut Tensor<f32, { [16, 16] }>,
            qp: &Tensor<f16, { [16, 32] }>,
            latent: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [-1] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let q: Tile<f16, { [16, 32] }> = qp.load_tile(const_shape![16, 32], [0, 0]);
            let mut acc: Tile<f32, { [16, 16] }> = constant(0.0f32, const_shape![16, 16]);
            for r in 0i32..R {
                let logical = pid.0 * R + r;
                let entry: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
                let physical = tile_to_scalar(entry.reshape(const_shape![]));
                let z: Tile<f16, { [16, 32] }> =
                    latent.load_tile(const_shape![16, 32], [physical, 0]);
                let zt: Tile<f16, { [32, 16] }> = z.transpose();
                acc = mma(q, zt, acc);
            }
            out.store(acc);
        }

        /// Split variant 1b: the score GEMM in the "zq" orientation, S^T = Z q'^T, with q'^T
        /// supplied pre-transposed so no register-tile transpose feeds `mma`.
        #[cutile::entry()]
        pub fn latent_gather_scores_zq_only<const R: i32>(
            out: &mut Tensor<f32, { [16, 16] }>,
            qpt: &Tensor<f16, { [32, 16] }>,
            latent: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [-1] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let qt: Tile<f16, { [32, 16] }> = qpt.load_tile(const_shape![32, 16], [0, 0]);
            let mut acc: Tile<f32, { [16, 16] }> = constant(0.0f32, const_shape![16, 16]);
            for r in 0i32..R {
                let logical = pid.0 * R + r;
                let entry: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
                let physical = tile_to_scalar(entry.reshape(const_shape![]));
                let z: Tile<f16, { [16, 32] }> =
                    latent.load_tile(const_shape![16, 32], [physical, 0]);
                acc = mma(z, qt, acc);
            }
            out.store(acc);
        }

        /// Size test: gather two blocks per step and concatenate to 32 tokens, so the score
        /// GEMM output is 512 elements. `TRANSPOSED` = 1 computes S = q' Z^T ([16, 32]) via a
        /// register transpose; 0 computes S^T = Z q'^T ([32, 16]) with q'^T pre-transposed.
        #[cutile::entry()]
        pub fn latent_gather2_scores<const R: i32>(
            out: &mut Tensor<f32, { [32, 16] }>,
            qpt: &Tensor<f16, { [32, 16] }>,
            latent: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [-1] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let qt: Tile<f16, { [32, 16] }> = qpt.load_tile(const_shape![32, 16], [0, 0]);
            let mut acc: Tile<f32, { [32, 16] }> = constant(0.0f32, const_shape![32, 16]);
            for r in 0i32..R {
                let logical = (pid.0 * R + r) * 2i32;
                let ea: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
                let eb: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical + 1i32]);
                let pa = tile_to_scalar(ea.reshape(const_shape![]));
                let pb = tile_to_scalar(eb.reshape(const_shape![]));
                let za: Tile<f16, { [16, 32] }> = latent.load_tile(const_shape![16, 32], [pa, 0]);
                let zb: Tile<f16, { [16, 32] }> = latent.load_tile(const_shape![16, 32], [pb, 0]);
                let z: Tile<f16, { [32, 32] }> = cat(za, zb, 0i32);
                acc = mma(z, qt, acc);
            }
            out.store(acc);
        }

        #[cutile::entry()]
        pub fn latent_gather2_scores_transposed<const R: i32>(
            out: &mut Tensor<f32, { [16, 32] }>,
            qp: &Tensor<f16, { [16, 32] }>,
            latent: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [-1] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let q: Tile<f16, { [16, 32] }> = qp.load_tile(const_shape![16, 32], [0, 0]);
            let mut acc: Tile<f32, { [16, 32] }> = constant(0.0f32, const_shape![16, 32]);
            for r in 0i32..R {
                let logical = (pid.0 * R + r) * 2i32;
                let ea: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
                let eb: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical + 1i32]);
                let pa = tile_to_scalar(ea.reshape(const_shape![]));
                let pb = tile_to_scalar(eb.reshape(const_shape![]));
                let za: Tile<f16, { [16, 32] }> = latent.load_tile(const_shape![16, 32], [pa, 0]);
                let zb: Tile<f16, { [16, 32] }> = latent.load_tile(const_shape![16, 32], [pb, 0]);
                let z: Tile<f16, { [32, 32] }> = cat(za, zb, 0i32);
                let zt: Tile<f16, { [32, 32] }> = z.transpose();
                acc = mma(q, zt, acc);
            }
            out.store(acc);
        }

        /// Split variant 2: only the PV-style GEMM acc += P Z with P read from memory.
        #[cutile::entry()]
        pub fn latent_gather_pv_only<const R: i32>(
            out: &mut Tensor<f32, { [16, 32] }>,
            probs: &Tensor<f16, { [16, -1] }>,
            latent: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [-1] }>,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let mut acc: Tile<f32, { [16, 32] }> = constant(0.0f32, const_shape![16, 32]);
            for r in 0i32..R {
                let logical = pid.0 * R + r;
                let entry: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
                let physical = tile_to_scalar(entry.reshape(const_shape![]));
                let z: Tile<f16, { [16, 32] }> =
                    latent.load_tile(const_shape![16, 32], [physical, 0]);
                let p: Tile<f16, { [16, 16] }> =
                    probs.load_tile(const_shape![16, 16], [0, logical]);
                acc = mma(p, z, acc);
            }
            out.store(acc);
        }
    }

    struct Case {
        label: String,
        ast: fn() -> Module,
        module: String,
        function: String,
        generics: Vec<String>,
        params: Vec<Param>,
    }

    fn production_cases(n: i32) -> Vec<Case> {
        let nb = n / 16;
        macro_rules! case {
            ($label:expr, $ast:path, $module:expr, $function:expr, $params:expr) => {
                Case {
                    label: format!("{}_N{}", $label, n),
                    ast: $ast,
                    module: $module.to_string(),
                    function: $function.to_string(),
                    generics: vec![],
                    params: $params,
                }
            };
        }
        macro_rules! cases_for {
            ($n:literal, $full:ident, $latent:ident, $c1:ident) => {
                vec![
                    case!(
                        "A1_score",
                        rt::$full::__module_ast_self,
                        stringify!($full),
                        concat!("model_small_full_kv_scores_fp16_storage_rtable_", $n),
                        vec![
                            param("out", &[16, n], 4),
                            param("q", &[16, 64], 4),
                            param("k_fp16", &[4 * n, 64], 2),
                            param("table", &[nb], 4),
                            param("active_seq_len", &[1], 4)
                        ]
                    ),
                    case!(
                        "A1_context",
                        rt::$full::__module_ast_self,
                        stringify!($full),
                        concat!("model_small_full_kv_context_fp16_storage_rtable_", $n),
                        vec![
                            param("out", &[16, 64], 4),
                            param("probabilities", &[16, n], 4),
                            param("v_fp16", &[4 * n, 64], 2),
                            param("table", &[nb], 4)
                        ]
                    ),
                    case!(
                        "B1_score",
                        rt::$latent::__module_ast_self,
                        stringify!($latent),
                        concat!("model_small_scores_fp16_storage_rtable_", $n),
                        vec![
                            param("out", &[16, n], 4),
                            param("q", &[16, 64], 4),
                            param("latent_fp16", &[n, 32], 2),
                            param("table", &[nb], 4),
                            param("active_seq_len", &[1], 4),
                            param("k_projection", &[128, 64], 4)
                        ]
                    ),
                    case!(
                        "softmax",
                        rt::$latent::__module_ast_self,
                        stringify!($latent),
                        concat!("model_small_softmax_", $n, "_runtime"),
                        vec![
                            param("out", &[16, n], 4),
                            param("scores", &[16, n], 4),
                            param("active_seq_len", &[1], 4)
                        ]
                    ),
                    case!(
                        "latent_context",
                        rt::$latent::__module_ast_self,
                        stringify!($latent),
                        concat!("model_small_context_fp16_storage_rtable_", $n),
                        vec![
                            param("out", &[16, 64], 4),
                            param("probabilities", &[16, n], 4),
                            param("latent_fp16", &[n, 32], 2),
                            param("table", &[nb], 4),
                            param("v_projection", &[128, 64], 4)
                        ]
                    ),
                    case!(
                        "C1_projection",
                        c1::$c1::__module_ast_self,
                        stringify!($c1),
                        "model_small_project_query_once",
                        vec![
                            param("out", &[16, 32], 4),
                            param("q", &[16, 64], 4),
                            param("k_projection", &[128, 64], 4)
                        ]
                    ),
                    case!(
                        "C1_score",
                        c1::$c1::__module_ast_self,
                        stringify!($c1),
                        "model_small_scores_fp16_storage_preprojected",
                        vec![
                            param("out", &[16, n], 4),
                            param("projected_query", &[16, 32], 4),
                            param("latent_fp16", &[n, 32], 2),
                            param("table", &[nb], 4),
                            param("active_seq_len", &[1], 4)
                        ]
                    ),
                ]
            };
        }
        match n {
            1024 => cases_for!(
                "1024",
                p15b_full_kv_baseline_kernel_1024,
                p15b_model_profile_kernel_1024,
                c1_kernel_1024
            ),
            8192 => cases_for!(
                "8192",
                p15b_full_kv_baseline_kernel_8192,
                p15b_model_profile_kernel_8192,
                c1_kernel_8192
            ),
            _ => unreachable!(),
        }
    }

    fn gather_cases() -> Vec<Case> {
        let n = 8192;
        let mut cases: Vec<Case> = [4, 16]
            .into_iter()
            .map(|r| Case {
                label: format!("gather_mma_R{r}"),
                ast: gather_mma_module::__module_ast_self,
                module: "gather_mma_module".into(),
                function: "latent_gather_mma".into(),
                generics: vec![r.to_string()],
                params: vec![
                    param("out", &[16 * (n / 16 / r), 32], 4),
                    param("qp", &[16, 32], 2),
                    param("latent", &[n, 32], 2),
                    param("table", &[n / 16], 4),
                ],
            })
            .collect();
        for r in [4, 16] {
            cases.push(Case {
                label: format!("gather_scores_only_R{r}"),
                ast: gather_mma_module::__module_ast_self,
                module: "gather_mma_module".into(),
                function: "latent_gather_scores_only".into(),
                generics: vec![r.to_string()],
                params: vec![
                    param("out", &[16 * (n / 16 / r), 16], 4),
                    param("qp", &[16, 32], 2),
                    param("latent", &[n, 32], 2),
                    param("table", &[n / 16], 4),
                ],
            });
            cases.push(Case {
                label: format!("gather_scores_zq_only_R{r}"),
                ast: gather_mma_module::__module_ast_self,
                module: "gather_mma_module".into(),
                function: "latent_gather_scores_zq_only".into(),
                generics: vec![r.to_string()],
                params: vec![
                    param("out", &[16 * (n / 16 / r), 16], 4),
                    param("qpt", &[32, 16], 2),
                    param("latent", &[n, 32], 2),
                    param("table", &[n / 16], 4),
                ],
            });
            for (label, function, out_shape, q_name, q_shape) in [
                (
                    "gather2_scores",
                    "latent_gather2_scores",
                    [32, 16],
                    "qpt",
                    [32, 16],
                ),
                (
                    "gather2_scores_transposed",
                    "latent_gather2_scores_transposed",
                    [16, 32],
                    "qp",
                    [16, 32],
                ),
            ] {
                cases.push(Case {
                    label: format!("{label}_R{r}"),
                    ast: gather_mma_module::__module_ast_self,
                    module: "gather_mma_module".into(),
                    function: function.into(),
                    generics: vec![r.to_string()],
                    params: vec![
                        param("out", &[out_shape[0] * (n / 32 / r), out_shape[1]], 4),
                        param(q_name, &q_shape, 2),
                        param("latent", &[n, 32], 2),
                        param("table", &[n / 16], 4),
                    ],
                });
            }
            cases.push(Case {
                label: format!("gather_pv_only_R{r}"),
                ast: gather_mma_module::__module_ast_self,
                module: "gather_mma_module".into(),
                function: "latent_gather_pv_only".into(),
                generics: vec![r.to_string()],
                params: vec![
                    param("out", &[16 * (n / 16 / r), 32], 4),
                    param("probs", &[16, n], 2),
                    param("latent", &[n, 32], 2),
                    param("table", &[n / 16], 4),
                ],
            });
        }
        cases
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().collect();
        let arg = |key: &str, default: &str| {
            args.windows(2)
                .find(|w| w[0] == key)
                .map(|w| w[1].clone())
                .unwrap_or(default.into())
        };
        let tc = Toolchain::from_env(&arg("--target", "sm_89"));
        let dir = PathBuf::from(arg("--output-dir", "/tmp/plkv_p3_probe"));
        std::fs::create_dir_all(&dir).unwrap();

        // Host-side alignment assert self-check: a misaligned layout must be rejected before
        // any hinted compile is attempted.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // the rejection below is expected
        let misaligned = std::panic::catch_unwind(|| {
            assert_hint_layouts(&[param("latent", &[64, 32], 2)], ALIGNED_BASE_PTR + 2)
        });
        std::panic::set_hook(default_hook);
        assert!(misaligned.is_err(), "misaligned layout was not rejected");
        println!("ALIGNMENT_ASSERT_SELF_TEST_OK (base+2 bytes rejected)");

        let mut cases = production_cases(1024);
        cases.extend(production_cases(8192));
        cases.extend(gather_cases());
        let mut results = Vec::new();
        for case in &cases {
            for hint in ["none", "maxdiv16"] {
                let tag = format!("{}_{}_{hint}", case.label, tc.target);
                let compiled = compile(
                    case.ast,
                    &case.module,
                    &case.function,
                    &case.generics,
                    &case.params,
                    hint,
                    &tc,
                    &dir,
                    &tag,
                );
                let (cubin, ir) = match compiled {
                    Ok(x) => x,
                    Err(e) => {
                        println!("{tag}: ERROR {e}");
                        results.push(serde_json::json!({"case":case.label,"hint":hint,"error":e}));
                        continue;
                    }
                };
                let instrs = sass(&cubin, &tc);
                let counts = opcode_counts(&instrs);
                let res = resource_usage(&cubin, &tc);
                let threads = threads_per_cta(&cubin, &tc).unwrap_or(0);
                let occ = occupancy_sm89(
                    *res.get("REG").unwrap_or(&0),
                    threads,
                    *res.get("SHARED").unwrap_or(&0),
                );
                let loads = global_load_bytes(&instrs, threads.div_ceil(32));
                let get = |k: &str| *counts.get(k).unwrap_or(&0);
                println!(
                    "{tag}: instr={} REG={} LOCAL={} STACK={} SHARED={} threads={threads} occ={:.1}% HMMA={} LDL/STL={}/{} SHFL={} BAR={} loads={}",
                    instrs.len(),
                    res.get("REG").unwrap_or(&0),
                    res.get("LOCAL").unwrap_or(&0),
                    res.get("STACK").unwrap_or(&0),
                    res.get("SHARED").unwrap_or(&0),
                    occ["theoretical_occupancy_pct"].as_f64().unwrap_or(0.0),
                    get("HMMA"),
                    get("LDL"),
                    get("STL"),
                    get("SHFL"),
                    get("BAR"),
                    loads["load_opcodes_per_warp"]
                );
                results.push(serde_json::json!({
                    "case": case.label, "function": case.function, "hint": hint,
                    "instructions": instrs.len(), "resource_usage": res, "threads_per_cta": threads,
                    "occupancy_sm89": occ, "global_loads": loads, "opcode_counts": counts,
                    "ir_has_mma": ir.contains("mmaf"),
                    "cubin": cubin.file_name().map(|f| f.to_string_lossy().into_owned()),
                }));
            }
        }
        std::fs::write(
            dir.join("p3_probe_summary.json"),
            serde_json::to_string_pretty(&results).unwrap(),
        )
        .unwrap();
        println!("P3_PROBE_DONE dir={}", dir.display());
    }
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    // The cuTile compiler recurses deeply; run on a large stack.
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
