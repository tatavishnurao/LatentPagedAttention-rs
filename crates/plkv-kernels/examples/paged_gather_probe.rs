//! Compile-only P2 probe for the real R-TABLE paged-gather latent score/context kernels.
// Kernel bodies are copied verbatim from the production R-TABLE kernels, including their
// f32 literals, so the probe compiles exactly what runs.
#![allow(clippy::excessive_precision)]

#[cfg(feature = "gpu-cutile")]
mod probe {
    use cutile::compile_api::KernelCompiler;
    use cutile::cutile_compiler::hints::CompileOptions;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const N: i32 = 1024;
    #[cutile::module]
    pub mod paged_nohint {
        use cutile::core::*;

        fn physical_block(table: &Tensor<i32, { [64] }>, logical: i32) -> i32 {
            let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
            tile_to_scalar(selected.reshape(const_shape![]))
        }

        fn latent_block_contribution(
            probabilities: &Tensor<f32, { [-1, 1024] }>,
            latent_fp16: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [64] }>,
            head: i32,
            logical_block: i32,
        ) -> Tile<f32, { [32] }> {
            let probs: Tile<f32, { [1, 16] }> =
                probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
            let latent: Tile<f32, { [16, 32] }> = convert_tile(latent_fp16.load_tile(
                const_shape![16, 32],
                [physical_block(table, logical_block), 0],
            ));
            reduce_sum(
                probs
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 32])
                    * latent,
                0i32,
            )
        }

        #[cutile::entry()]
        pub fn model_small_scores_fp16_storage_rtable_1024(
            out: &mut Tensor<f32, { [1, 16] }>,
            q: &Tensor<f32, { [-1, 64] }>,
            latent_fp16: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [64] }>,
            active_seq_len: &Tensor<i32, { [1] }>,
            k_projection: &Tensor<f32, { [-1, 64] }>,
        ) {
            let pid = get_tile_block_id();
            let q_head = pid.0;
            let logical_block = pid.1;
            let kv_head = q_head / 4i32;
            let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
            let physical = physical_block(table, logical_block);
            let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
            let kp: Tile<f32, { [32, 64] }> =
                k_projection.load_tile(const_shape![32, 64], [kv_head, 0]);
            let projected: Tile<f32, { [32] }> =
                reduce_sum(kp * q_row.broadcast(const_shape![32, 64]), 1i32);
            let latent_f16: Tile<f16, { [16, 32] }> =
                latent_fp16.load_tile(const_shape![16, 32], [physical, 0]);
            let latent_f32: Tile<f32, { [16, 32] }> = convert_tile(latent_f16);
            let dots: Tile<f32, { [16] }> = reduce_sum(
                latent_f32
                    * projected
                        .reshape(const_shape![1, 32])
                        .broadcast(const_shape![16, 32]),
                1i32,
            );
            let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
            let token_indices: Tile<i32, { [16] }> =
                iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
            let active_mask: Tile<bool, { [16] }> = cmpi(
                token_indices,
                active_tile.broadcast(const_shape![16]),
                predicate::LessThan,
            );
            out.store(
                select(
                    active_mask,
                    scores,
                    broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
                )
                .reshape(const_shape![1, 16]),
            );
        }

        #[cutile::entry()]
        pub fn model_small_context_fp16_storage_rtable_1024(
            out: &mut Tensor<f32, { [1, 64] }>,
            probabilities: &Tensor<f32, { [-1, 1024] }>,
            latent_fp16: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [64] }>,
            v_projection: &Tensor<f32, { [-1, 64] }>,
        ) {
            let head = get_tile_block_id().0;
            let kv_head = head / 4i32;
            let mut latent_context: Tile<f32, { [32] }> =
                broadcast_scalar(0.0f32, const_shape![32]);
            for logical_block in 0i32..64i32 {
                latent_context = latent_context
                    + latent_block_contribution(
                        probabilities,
                        latent_fp16,
                        table,
                        head,
                        logical_block,
                    );
            }
            let vp: Tile<f32, { [32, 64] }> =
                v_projection.load_tile(const_shape![32, 64], [kv_head, 0]);
            let context: Tile<f32, { [64] }> = reduce_sum(
                latent_context
                    .reshape(const_shape![32, 1])
                    .broadcast(const_shape![32, 64])
                    * vp,
                0i32,
            );
            out.store(context.reshape(const_shape![1, 64]));
        }
    }

    #[cutile::module]
    pub mod paged_source_hint {
        use cutile::core::*;

        fn physical_block(table: &Tensor<i32, { [64] }>, logical: i32) -> i32 {
            let selected: Tile<i32, { [1] }> = table.load_tile(const_shape![1], [logical]);
            tile_to_scalar(selected.reshape(const_shape![]))
        }

        fn latent_block_contribution(
            probabilities: &Tensor<f32, { [-1, 1024] }>,
            latent_fp16: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [64] }>,
            head: i32,
            logical_block: i32,
        ) -> Tile<f32, { [32] }> {
            let probs: Tile<f32, { [1, 16] }> =
                probabilities.load_tile(const_shape![1, 16], [head, logical_block]);
            let latent: Tile<f32, { [16, 32] }> = convert_tile(latent_fp16.load_tile(
                const_shape![16, 32],
                [physical_block(table, logical_block), 0],
            ));
            reduce_sum(
                probs
                    .reshape(const_shape![16, 1])
                    .broadcast(const_shape![16, 32])
                    * latent,
                0i32,
            )
        }

        #[cutile::entry(optimization_hints = (sm_89 = (max_divisibility = 16,),))]
        pub fn model_small_scores_fp16_storage_rtable_1024(
            out: &mut Tensor<f32, { [1, 16] }>,
            q: &Tensor<f32, { [-1, 64] }>,
            latent_fp16: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [64] }>,
            active_seq_len: &Tensor<i32, { [1] }>,
            k_projection: &Tensor<f32, { [-1, 64] }>,
        ) {
            let pid = get_tile_block_id();
            let q_head = pid.0;
            let logical_block = pid.1;
            let kv_head = q_head / 4i32;
            let active_tile: Tile<i32, { [1] }> = active_seq_len.load_tile(const_shape![1], [0]);
            let physical = physical_block(table, logical_block);
            let q_row: Tile<f32, { [1, 64] }> = q.load_tile(const_shape![1, 64], [q_head, 0]);
            let kp: Tile<f32, { [32, 64] }> =
                k_projection.load_tile(const_shape![32, 64], [kv_head, 0]);
            let projected: Tile<f32, { [32] }> =
                reduce_sum(kp * q_row.broadcast(const_shape![32, 64]), 1i32);
            let latent_f16: Tile<f16, { [16, 32] }> =
                latent_fp16.load_tile(const_shape![16, 32], [physical, 0]);
            let latent_f32: Tile<f32, { [16, 32] }> = convert_tile(latent_f16);
            let dots: Tile<f32, { [16] }> = reduce_sum(
                latent_f32
                    * projected
                        .reshape(const_shape![1, 32])
                        .broadcast(const_shape![16, 32]),
                1i32,
            );
            let scores = dots * broadcast_scalar(0.125f32, const_shape![16]);
            let token_indices: Tile<i32, { [16] }> =
                iota(const_shape![16]) + broadcast_scalar(logical_block * 16i32, const_shape![16]);
            let active_mask: Tile<bool, { [16] }> = cmpi(
                token_indices,
                active_tile.broadcast(const_shape![16]),
                predicate::LessThan,
            );
            out.store(
                select(
                    active_mask,
                    scores,
                    broadcast_scalar(-3.4028234663852886e38f32, const_shape![16]),
                )
                .reshape(const_shape![1, 16]),
            );
        }

        #[cutile::entry(optimization_hints = (sm_89 = (max_divisibility = 16,),))]
        pub fn model_small_context_fp16_storage_rtable_1024(
            out: &mut Tensor<f32, { [1, 64] }>,
            probabilities: &Tensor<f32, { [-1, 1024] }>,
            latent_fp16: &Tensor<f16, { [-1, 32] }>,
            table: &Tensor<i32, { [64] }>,
            v_projection: &Tensor<f32, { [-1, 64] }>,
        ) {
            let head = get_tile_block_id().0;
            let kv_head = head / 4i32;
            let mut latent_context: Tile<f32, { [32] }> =
                broadcast_scalar(0.0f32, const_shape![32]);
            for logical_block in 0i32..64i32 {
                latent_context = latent_context
                    + latent_block_contribution(
                        probabilities,
                        latent_fp16,
                        table,
                        head,
                        logical_block,
                    );
            }
            let vp: Tile<f32, { [32, 64] }> =
                v_projection.load_tile(const_shape![32, 64], [kv_head, 0]);
            let context: Tile<f32, { [64] }> = reduce_sum(
                latent_context
                    .reshape(const_shape![32, 1])
                    .broadcast(const_shape![32, 64])
                    * vp,
                0i32,
            );
            out.store(context.reshape(const_shape![1, 64]));
        }
    }

    use paged_nohint::__module_ast_self as nohint_ast;
    use paged_source_hint::__module_ast_self as source_hint_ast;

    fn sass_counts(cubin: &Path, cuobjdump: &Path) -> BTreeMap<String, usize> {
        let out = Command::new(cuobjdump)
            .arg("-sass")
            .arg(cubin)
            .output()
            .unwrap();
        assert!(out.status.success(), "cuobjdump failed: {out:?}");
        let text = String::from_utf8_lossy(&out.stdout);
        let mut counts = BTreeMap::new();
        for line in text.lines() {
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
            if op.is_empty() {
                continue;
            }
            let base = op.split('.').next().unwrap().to_string();
            if matches!(base.as_str(), "LDG" | "STG" | "LDS" | "STS" | "LDGSTS") {
                *counts.entry(format!("op:{op}")).or_insert(0) += 1;
            }
            *counts.entry(base).or_insert(0) += 1;
            *counts.entry("_total".into()).or_insert(0) += 1;
        }
        counts
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_one(
        ast: fn() -> cutile::cutile_compiler::ast::Module,
        module_name: &str,
        kernel: &str,
        mode: &str,
        target: &str,
        dir: &Path,
        tileiras: &Path,
        cuobjdump: &Path,
        builder_hint: bool,
    ) -> serde_json::Value {
        let strides: Vec<(&str, &[i32])> = match kernel {
            "model_small_scores_fp16_storage_rtable_1024" => vec![
                ("out", &[N, 1]),
                ("q", &[64, 1]),
                ("latent_fp16", &[32, 1]),
                ("table", &[1]),
                ("active_seq_len", &[1]),
                ("k_projection", &[64, 1]),
            ],
            "model_small_context_fp16_storage_rtable_1024" => vec![
                ("out", &[64, 1]),
                ("probabilities", &[N, 1]),
                ("latent_fp16", &[32, 1]),
                ("table", &[1]),
                ("v_projection", &[64, 1]),
            ],
            _ => unreachable!(),
        };
        let tag = format!("{kernel}_{target}_{mode}");
        let mut compiler = KernelCompiler::new(ast, module_name, kernel)
            .strides(&strides)
            .target(target);
        if builder_hint {
            compiler = compiler.options(CompileOptions::new().max_divisibility(16));
        }
        let artifacts = compiler.compile().unwrap();
        let ir = artifacts.ir_text();
        let bc = dir.join(format!("{tag}.bc"));
        let cubin = dir.join(format!("{tag}.cubin"));
        std::fs::write(&bc, artifacts.bytecode().unwrap()).unwrap();
        std::fs::write(dir.join(format!("{tag}.mlir")), &ir).unwrap();
        let status = Command::new(tileiras)
            .args(["--gpu-name", target, "--opt-level", "3", "-o"])
            .arg(&cubin)
            .arg(&bc)
            .output()
            .unwrap();
        assert!(status.status.success(), "tileiras failed: {status:?}");
        let counts = sass_counts(&cubin, cuobjdump);
        let loads: Vec<_> = counts
            .iter()
            .filter(|(k, _)| k.starts_with("op:LDG"))
            .map(|(k, v)| format!("{}x{v}", &k[3..]))
            .collect();
        println!(
            "{tag}: HMMA={} total={} loads=[{}]",
            counts.get("HMMA").unwrap_or(&0),
            counts.get("_total").unwrap_or(&0),
            loads.join(" ")
        );
        serde_json::json!({"kernel":kernel,"mode":mode,"target":target,"opcode_counts":counts,"cubin":cubin.file_name().unwrap().to_string_lossy()})
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
        let dir = PathBuf::from(arg("--output-dir", "/tmp/plkv_paged_gather_probe"));
        std::fs::create_dir_all(&dir).unwrap();
        let toolkit =
            PathBuf::from(std::env::var("CUDA_TOOLKIT_PATH").unwrap_or("/usr/local/cuda".into()));
        let tileiras = std::env::var("CUTILE_TILEIRAS_PATH")
            .map(PathBuf::from)
            .unwrap_or(toolkit.join("bin/tileiras"));
        let cuobjdump = toolkit.join("bin/cuobjdump");
        let kernels = [
            "model_small_scores_fp16_storage_rtable_1024",
            "model_small_context_fp16_storage_rtable_1024",
        ];
        let mut results = Vec::new();
        for kernel in kernels {
            results.push(compile_one(
                nohint_ast,
                "paged_nohint",
                kernel,
                "none",
                &target,
                &dir,
                &tileiras,
                &cuobjdump,
                false,
            ));
            results.push(compile_one(
                nohint_ast,
                "paged_nohint",
                kernel,
                "builder_maxdiv16",
                &target,
                &dir,
                &tileiras,
                &cuobjdump,
                true,
            ));
            results.push(compile_one(
                source_hint_ast,
                "paged_source_hint",
                kernel,
                "source_maxdiv16",
                &target,
                &dir,
                &tileiras,
                &cuobjdump,
                false,
            ));
        }
        std::fs::write(
            dir.join("paged_gather_probe_summary.json"),
            serde_json::to_string_pretty(&results).unwrap(),
        )
        .unwrap();
        println!("PAGED_GATHER_PROBE_DONE dir={}", dir.display());
    }
}

#[cfg(feature = "gpu-cutile")]
fn main() {
    probe::main();
}

#[cfg(not(feature = "gpu-cutile"))]
fn main() {
    eprintln!("requires --features gpu-cutile");
}
