//! GPU-free E4 plan check: prints the footprint, active lengths and split plans that
//! `e3_splitk_gpu --suite e4 / e4-timing` would use for the same arguments, and exits non-zero
//! (code 2) if the configuration is refused. Runs no kernel and needs no GPU.
//!
//! `cargo run -p plkv-kernels --example e4_plan -- --seq 8192 --batch 8`
use plkv_kernels::e4_config::{e4_layer_bytes, e4_layouts, parse_e4_args};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layouts = match parse_e4_args(&args).and_then(|a| e4_layouts(&a)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("E4_CONFIG_REFUSED: {e}");
            std::process::exit(2);
        }
    };
    for lay in &layouts {
        let b = e4_layer_bytes(lay);
        let mib = |x: u64| x as f64 / (1u64 << 20) as f64;
        println!("E4_FOOTPRINT {}", lay.footprint.describe());
        println!(
            "E4_PLAN B={} N={} layers={} pool_blocks={} per_layer: a3pad={:.2}MiB c3={:.2}MiB ratio={:.2} cache_only_ratio={:.2}",
            lay.batch,
            lay.seq,
            lay.layers,
            lay.pool_blocks,
            mib(b.a3pad),
            mib(b.c3),
            b.a3pad as f64 / b.c3 as f64,
            b.min_a3 as f64 / b.min_c3 as f64
        );
        for (i, s) in lay.seqs.iter().enumerate() {
            println!(
                "  b={i} active={} splits={} steps_per_split={} blocks_read={}",
                s.active, s.plan.splits, s.plan.steps_per_split, s.plan.blocks_read
            );
        }
    }
    println!("E4_PLAN_OK=1");
}
