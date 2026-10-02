//! Shared helpers for compile-only cuTile probes: Tile IR -> tileiras cubin -> SASS/resources.
//! No GPU or CUDA context is used.
#![allow(dead_code)]

use cutile::compile_api::KernelCompiler;
use cutile::cutile_compiler::ast::Module;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::cutile_compiler::specialization::{SpecializationBits, compute_spec};
use plkv_kernels::alignment::{TensorLayout, VECTOR_ALIGN_BYTES, check_layouts};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// cudaMalloc returns >= 256-byte aligned pointers; DivHint::from_ptr clamps to 16.
pub const ALIGNED_BASE_PTR: u64 = 0x7f00_0000_0000;

/// One tensor parameter: name, full-tensor shape, element bytes (row-major strides).
pub struct Param {
    pub name: &'static str,
    pub shape: Vec<i32>,
    pub elem_bytes: i32,
}

pub fn param(name: &'static str, shape: &[i32], elem_bytes: i32) -> Param {
    Param {
        name,
        shape: shape.to_vec(),
        elem_bytes,
    }
}

pub fn row_major_strides(shape: &[i32]) -> Vec<i32> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub struct Toolchain {
    pub tileiras: PathBuf,
    pub cuobjdump: PathBuf,
    pub target: String,
}

impl Toolchain {
    pub fn from_env(target: &str) -> Self {
        let toolkit =
            PathBuf::from(std::env::var("CUDA_TOOLKIT_PATH").unwrap_or("/usr/local/cuda".into()));
        Self {
            tileiras: std::env::var("CUTILE_TILEIRAS_PATH")
                .map(PathBuf::from)
                .unwrap_or(toolkit.join("bin/tileiras")),
            cuobjdump: toolkit.join("bin/cuobjdump"),
            target: target.into(),
        }
    }
}

/// Host-side alignment assert for the hinted (`max_divisibility(16)`) build: every declared
/// tensor must be 16-byte aligned with 16-byte-multiple outer strides. Panics otherwise, so a
/// forced hint can never be paired with a layout that breaks 128-bit loads.
pub fn assert_hint_layouts(params: &[Param], base_ptr: u64) {
    let strides: Vec<Vec<i32>> = params.iter().map(|p| row_major_strides(&p.shape)).collect();
    let layouts: Vec<TensorLayout> = params
        .iter()
        .zip(&strides)
        .map(|(p, s)| TensorLayout {
            name: p.name,
            base_ptr,
            strides: s,
            elem_bytes: p.elem_bytes as usize,
        })
        .collect();
    if let Err(errors) = check_layouts(&layouts, VECTOR_ALIGN_BYTES) {
        let msgs: Vec<String> = errors.iter().map(ToString::to_string).collect();
        panic!(
            "max_divisibility(16) requested for a misaligned layout: {}",
            msgs.join("; ")
        );
    }
}

/// Compiles one kernel to a cubin. `hint` is "none" or a "+"-joined set of "maxdiv16",
/// "spec" and "occN" (occupancy hint N CTAs/SM).
#[allow(clippy::too_many_arguments)]
pub fn compile(
    ast: fn() -> Module,
    module: &str,
    function: &str,
    generics: &[String],
    params: &[Param],
    hint: &str,
    tc: &Toolchain,
    dir: &Path,
    tag: &str,
) -> Result<(PathBuf, String), String> {
    let strides: Vec<Vec<i32>> = params.iter().map(|p| row_major_strides(&p.shape)).collect();
    let stride_refs: Vec<(&str, &[i32])> = params
        .iter()
        .zip(&strides)
        .map(|(p, s)| (p.name, s.as_slice()))
        .collect();
    let mut compiler = KernelCompiler::new(ast, module, function)
        .generics(generics.to_vec())
        .strides(&stride_refs)
        .target(&tc.target);
    let mut options = CompileOptions::new();
    let mut set_options = false;
    if hint.contains("maxdiv16") {
        assert_hint_layouts(params, ALIGNED_BASE_PTR);
        options = options.max_divisibility(16);
        set_options = true;
    }
    if let Some(occ) = hint
        .split('+')
        .find_map(|h| h.strip_prefix("occ").and_then(|n| n.parse::<i32>().ok()))
    {
        // Entry-level occupancy hint: target CTAs per SM (bounds registers per thread).
        options = options.occupancy(occ);
        set_options = true;
    }
    if set_options {
        compiler = compiler.options(options);
    }
    if hint.contains("spec") {
        let specs: Vec<(&str, SpecializationBits)> = params
            .iter()
            .zip(&strides)
            .map(|(p, s)| {
                (
                    p.name,
                    compute_spec(ALIGNED_BASE_PTR, &p.shape, s, p.elem_bytes),
                )
            })
            .collect();
        compiler = compiler.spec_args(&specs);
    }
    let artifacts = compiler.compile().map_err(|e| format!("tile_ir: {e:?}"))?;
    let ir = artifacts.ir_text();
    let bc = dir.join(format!("{tag}.bc"));
    let cubin = dir.join(format!("{tag}.cubin"));
    std::fs::write(&bc, artifacts.bytecode().map_err(|e| format!("{e:?}"))?).unwrap();
    std::fs::write(dir.join(format!("{tag}.mlir")), &ir).unwrap();
    let out = Command::new(&tc.tileiras)
        .args(["--gpu-name", &tc.target, "--opt-level", "3", "-o"])
        .arg(&cubin)
        .arg(&bc)
        .output()
        .map_err(|e| format!("tileiras: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "tileiras: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok((cubin, ir))
}

/// One SASS instruction: address, predicate-stripped opcode, full text.
pub struct Instr {
    pub addr: u64,
    pub op: String,
    pub text: String,
}

pub fn sass(cubin: &Path, tc: &Toolchain) -> Vec<Instr> {
    let out = Command::new(&tc.cuobjdump)
        .arg("-sass")
        .arg(cubin)
        .output()
        .expect("cuobjdump");
    let mut instrs = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some(rest) = line.trim_start().strip_prefix("/*") else {
            continue;
        };
        let Some((addr, body)) = rest.split_once("*/") else {
            continue;
        };
        let Ok(addr) = u64::from_str_radix(addr.trim(), 16) else {
            continue;
        };
        let body = body
            .split("/*")
            .next()
            .unwrap_or("")
            .trim()
            .trim_end_matches(';');
        let mut words = body.split_whitespace();
        let mut op = words.next().unwrap_or("");
        if op.starts_with('@') {
            op = words.next().unwrap_or("");
        }
        if op.is_empty() {
            continue;
        }
        instrs.push(Instr {
            addr,
            op: op.into(),
            text: body.into(),
        });
    }
    instrs
}

pub fn base_op(op: &str) -> &str {
    op.split('.').next().unwrap_or(op)
}

pub fn opcode_counts(instrs: &[Instr]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for i in instrs {
        *counts.entry(base_op(&i.op).to_string()).or_insert(0) += 1;
        if is_memory(base_op(&i.op)) {
            *counts.entry(format!("op:{}", i.op)).or_insert(0) += 1;
        }
    }
    counts
}

fn is_memory(base: &str) -> bool {
    matches!(
        base,
        "LDG" | "STG" | "LDS" | "STS" | "LDGSTS" | "LDL" | "STL" | "LDSM"
    )
}

/// Bytes moved per lane by a global load opcode (LDG/LDGSTS).
pub fn load_width_bytes(op: &str) -> u64 {
    if op.contains(".128") {
        16
    } else if op.contains(".64") {
        8
    } else if op.contains(".U16") || op.contains(".S16") {
        2
    } else if op.contains(".U8") || op.contains(".S8") {
        1
    } else {
        4
    }
}

/// Backward branches define loop bodies [target, branch].
pub fn loop_regions(instrs: &[Instr]) -> Vec<(u64, u64)> {
    instrs
        .iter()
        .filter(|i| base_op(&i.op) == "BRA")
        .filter_map(|i| {
            let target = i.text.split_whitespace().find_map(|w| {
                u64::from_str_radix(w.trim_start_matches("0x").trim_end_matches(','), 16)
                    .ok()
                    .filter(|_| w.starts_with("0x"))
            })?;
            (target < i.addr).then_some((target, i.addr))
        })
        .collect()
}

/// Static global-load bytes per CTA (all lanes active), split into outside-loop and
/// per-loop-iteration (innermost region containing the load).
pub fn global_load_bytes(instrs: &[Instr], warps: u64) -> serde_json::Value {
    let loops = loop_regions(instrs);
    let mut once = 0u64;
    let mut per_iter: BTreeMap<String, u64> = BTreeMap::new();
    let mut widths: BTreeMap<String, usize> = BTreeMap::new();
    for i in instrs {
        let b = base_op(&i.op);
        if b != "LDG" && b != "LDGSTS" {
            continue;
        }
        *widths.entry(i.op.clone()).or_insert(0) += 1;
        let bytes = load_width_bytes(&i.op) * 32 * warps;
        match loops
            .iter()
            .filter(|(s, e)| *s <= i.addr && i.addr <= *e)
            .min_by_key(|(s, e)| e - s)
        {
            Some((s, e)) => *per_iter.entry(format!("{s:#x}-{e:#x}")).or_insert(0) += bytes,
            None => once += bytes,
        }
    }
    serde_json::json!({"load_opcodes_per_warp":widths,"bytes_outside_loops_per_cta":once,"bytes_per_iteration_per_cta_by_loop":per_iter,"loops":loops.iter().map(|(s,e)| format!("{s:#x}-{e:#x}")).collect::<Vec<_>>()})
}

/// `cuobjdump --dump-resource-usage` fields for the (single) function in a cubin.
pub fn resource_usage(cubin: &Path, tc: &Toolchain) -> BTreeMap<String, u64> {
    let out = Command::new(&tc.cuobjdump)
        .arg("--dump-resource-usage")
        .arg(cubin)
        .output()
        .expect("cuobjdump");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut fields = BTreeMap::new();
    let mut in_function = false;
    for line in text.lines() {
        if line.trim_start().starts_with("Function ") {
            in_function = true;
            continue;
        }
        if !in_function {
            continue;
        }
        for kv in line.split_whitespace() {
            if let Some((k, v)) = kv.split_once(':')
                && let Ok(v) = v.parse()
            {
                fields.insert(k.to_string(), v);
            }
        }
    }
    fields
}

/// Threads per CTA from EIATTR_REQNTID (x*y*z).
pub fn threads_per_cta(cubin: &Path, tc: &Toolchain) -> Option<u64> {
    let out = Command::new(&tc.cuobjdump)
        .arg("-elf")
        .arg(cubin)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.contains("EIATTR_REQNTID") {
            for next in lines.by_ref().take(3) {
                if let Some(v) = next.trim().strip_prefix("Value:") {
                    return Some(
                        v.split_whitespace()
                            .filter_map(|x| {
                                u64::from_str_radix(x.trim_start_matches("0x"), 16).ok()
                            })
                            .product(),
                    );
                }
            }
        }
    }
    None
}

/// Theoretical occupancy on sm_89 (Ada): 48 warps/SM, 64K regs/SM (256-reg warp granule,
/// regs/thread rounded to 8), 24 CTAs/SM, 100 KiB shared/SM with 1 KiB reserved per CTA
/// (128-byte granule). Cross-checked against P05 ncu: 113 regs x 128 threads -> 33.3 %.
pub fn occupancy_sm89(regs: u64, threads: u64, shared: u64) -> serde_json::Value {
    let warps = threads.div_ceil(32).max(1);
    let regs_per_warp = (regs.div_ceil(8) * 8 * 32).div_ceil(256) * 256;
    let by_regs = 65536u64
        .checked_div(regs_per_warp)
        .map_or(24, |w| w / warps);
    let smem_per_cta = (shared + 1024).div_ceil(128) * 128;
    let by_smem = 102_400 / smem_per_cta;
    let by_warps = 48 / warps;
    let ctas = by_regs.min(by_smem).min(by_warps).min(24);
    let limiter = if ctas == by_regs {
        "registers"
    } else if ctas == by_smem {
        "shared_memory"
    } else {
        "warps_or_ctas"
    };
    serde_json::json!({"warps_per_cta":warps,"ctas_per_sm":ctas,"limit_by_regs":by_regs,"limit_by_smem":by_smem,"limit_by_warps":by_warps,"theoretical_occupancy_pct":(ctas*warps) as f64 * 100.0 / 48.0,"limiter":limiter})
}
