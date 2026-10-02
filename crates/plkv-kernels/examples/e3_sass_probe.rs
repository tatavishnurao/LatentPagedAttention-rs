//! Compile-only (CPU) SASS report for the E3 split-K kernels: HMMA count, global load widths,
//! registers, spills, shared memory and theoretical occupancy, for the tiny and model_small
//! profiles. Specialization bits are computed exactly as cuTile's runtime does for aligned
//! allocations (`spec` mode), so the SASS matches what a launch would JIT.
#[cfg(feature = "gpu-cutile")]
#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "gpu-cutile")]
mod probe {
    use super::common::*;
    use plkv_kernels::cutile::e3_splitk::e3_kernels;
    use std::path::PathBuf;

    struct Profile {
        name: &'static str,
        hq: i32,
        hkv: i32,
        d: i32,
        l: i32,
        blk: i32,
        dc: i32,
        seq: i32,
        splits: i32,
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
        let dir = PathBuf::from(arg("--output-dir", "/tmp/plkv_e3_sass"));
        std::fs::create_dir_all(&dir).unwrap();
        let profiles = [
            Profile {
                name: "tiny",
                hq: 4,
                hkv: 2,
                d: 8,
                l: 8,
                blk: 2,
                dc: 8,
                seq: 8,
                splits: 2,
            },
            Profile {
                name: "model_small",
                hq: 16,
                hkv: 4,
                d: 64,
                l: 32,
                blk: 16,
                dc: 16,
                seq: 1024,
                splits: 4,
            },
        ];
        let mut results = Vec::new();
        for p in &profiles {
            let (g, tok, nb, s) = (p.hq / p.hkv, 2 * p.blk, p.seq / p.blk, p.splits);
            let kernels: Vec<(&str, Vec<String>, Vec<Param>)> = vec![
                (
                    "c3_split",
                    [p.hq, p.hkv, g, p.d, p.l, p.blk, tok, p.dc, p.hkv * p.l]
                        .iter()
                        .map(i32::to_string)
                        .collect(),
                    vec![
                        param("out_acc", &[s, p.hq, p.l], 4),
                        param("out_m", &[s, p.hq, 1], 4),
                        param("out_l", &[s, p.hq, 1], 4),
                        param("q", &[p.hq, p.d], 4),
                        param("k_projection", &[p.hkv * p.l, p.d], 4),
                        param("latent", &[p.seq, p.l], 2),
                        param("table", &[nb], 4),
                    ],
                ),
                (
                    "c3_reduce",
                    [p.hq, g, p.d, p.l].iter().map(i32::to_string).collect(),
                    vec![
                        param("out", &[p.hq, p.d], 4),
                        param("partial_acc", &[s, p.hq, p.l], 4),
                        param("partial_m", &[s, p.hq, 1], 4),
                        param("partial_l", &[s, p.hq, 1], 4),
                        param("v_projection", &[p.hkv * p.l, p.d], 4),
                    ],
                ),
                (
                    "a3_split",
                    [p.hq, p.hkv, g, p.d, p.blk, tok]
                        .iter()
                        .map(i32::to_string)
                        .collect(),
                    vec![
                        param("out_acc", &[s, p.hq, p.d], 4),
                        param("out_m", &[s, p.hq, 1], 4),
                        param("out_l", &[s, p.hq, 1], 4),
                        param("q", &[p.hq, p.d], 4),
                        param("k_full", &[nb * p.hkv * p.blk, p.d], 2),
                        param("v_full", &[nb * p.hkv * p.blk, p.d], 2),
                        param("table", &[nb], 4),
                    ],
                ),
                (
                    "a3_reduce",
                    [p.hq, g, p.d].iter().map(i32::to_string).collect(),
                    vec![
                        param("out", &[p.hq, p.d], 4),
                        param("partial_acc", &[s, p.hq, p.d], 4),
                        param("partial_m", &[s, p.hq, 1], 4),
                        param("partial_l", &[s, p.hq, 1], 4),
                    ],
                ),
            ];
            for (name, generics, params) in kernels {
                let tag = format!("{name}_{}_{}", p.name, tc.target);
                let compiled = compile(
                    e3_kernels::__module_ast_self,
                    "e3_kernels",
                    name,
                    &generics,
                    &params,
                    "spec",
                    &tc,
                    &dir,
                    &tag,
                );
                let (cubin, _ir) = match compiled {
                    Ok(x) => x,
                    Err(e) => {
                        println!("{tag}: ERROR {e}");
                        results.push(serde_json::json!({"kernel":name,"profile":p.name,"error":e}));
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
                    "{tag}: instr={} REG={} LOCAL={} STACK={} SHARED={} threads={threads} occ={:.1}% HMMA={} FFMA={} FMUL+FADD={} LDL/STL={}/{} loads={}",
                    instrs.len(),
                    res.get("REG").unwrap_or(&0),
                    res.get("LOCAL").unwrap_or(&0),
                    res.get("STACK").unwrap_or(&0),
                    res.get("SHARED").unwrap_or(&0),
                    occ["theoretical_occupancy_pct"].as_f64().unwrap_or(0.0),
                    get("HMMA"),
                    get("FFMA"),
                    get("FMUL") + get("FADD"),
                    get("LDL"),
                    get("STL"),
                    loads["load_opcodes_per_warp"]
                );
                results.push(serde_json::json!({
                    "kernel": name, "profile": p.name, "generics": generics, "hint": "spec",
                    "instructions": instrs.len(), "resource_usage": res, "threads_per_cta": threads,
                    "occupancy_sm89": occ, "global_loads": loads, "opcode_counts": counts,
                    "cubin": cubin.file_name().map(|f| f.to_string_lossy().into_owned()),
                }));
            }
        }
        std::fs::write(
            dir.join("e3_sass_summary.json"),
            serde_json::to_string_pretty(&results).unwrap(),
        )
        .unwrap();
        println!("E3_SASS_PROBE_DONE dir={}", dir.display());
    }
}

#[cfg(feature = "gpu-cutile")]
fn main() {
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
