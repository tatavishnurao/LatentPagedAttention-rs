//! Host-side E4 (batched, multi-layer decode step) configuration: active lengths, per-sequence
//! block tables into a shared per-layer pool, bounded split plans, memory footprint and
//! analytical bytes. Everything here runs without a GPU so it can be unit-tested; the GPU
//! harness (`examples/e3_splitk_gpu.rs`) builds its launches from [`e4_layout`].
//!
//! Shape is fixed to model_small (Hq 16, Hkv 4, d 64, latent rank 32, block 16, FP16 cache).

use crate::block_table::validate_block_table;
use crate::splitk_plan::{SplitPlan, plan_bounded};
use std::fmt;

pub const HQ: usize = 16;
pub const HKV: usize = 4;
pub const HEAD_DIM: usize = 64;
pub const LATENT_RANK: usize = 32;
pub const BLOCK_SIZE: usize = 16;
/// Default per-sequence context; the original E4 configuration.
pub const DEFAULT_SEQ: usize = 1024;
/// Every E4 launch is planned with at least this many splits (fills more SMs at small B).
pub const MIN_SPLITS: usize = 4;
/// Minimum pool tokens summed over layers: at 1 KiB of full K+V per token this keeps the A3
/// pools at >= 64 MiB (`layers = max(4, ceil(65536 / (B * N)))`, i.e. `max(4, ceil(64 / B))`
/// at N = 1024, the original rule).
pub const MIN_POOL_TOKENS_ALL_LAYERS: usize = 65536;
pub const MIN_LAYERS: usize = 4;
/// 64 MiB L2-flush buffer the harness allocates (16 Mi 32-bit words).
pub const L2_FLUSH_BYTES: u64 = 64 << 20;
/// Default device-footprint limit (suits a 24 GB L4 with headroom).
pub const DEFAULT_MAX_FOOTPRINT_GIB: f64 = 20.0;
/// Default host-RAM limit for the f32 host copies the CPU references need.
pub const DEFAULT_MAX_HOST_FOOTPRINT_GIB: f64 = 24.0;
pub const DEFAULT_BATCHES: [usize; 3] = [1, 8, 32];

/// Full-KV FP16 bytes per token (K and V): Hkv * d * 2 bytes * 2 = 1024.
pub const FULL_KV_BYTES_PER_TOKEN: u64 = (HKV * HEAD_DIM * 2 * 2) as u64;
/// Latent FP16 bytes per token: rank * 2 = 64.
pub const LATENT_BYTES_PER_TOKEN: u64 = (LATENT_RANK * 2) as u64;
/// Host f32 bytes per pool token per layer kept in the harness `Case` (latent + head-major K
/// and V physical copies).
pub const HOST_RETAINED_BYTES_PER_TOKEN: u64 = ((LATENT_RANK + 2 * HKV * HEAD_DIM) * 4) as u64;
/// Host f32 bytes per pool token for transient buffers while one layer is built (logical
/// latent, logical K and V) plus the FP16 copies the CPU reference quantizes once per layer.
pub const HOST_TRANSIENT_BYTES_PER_TOKEN: u64 =
    ((LATENT_RANK + 2 * HKV * HEAD_DIM) * 4 + (LATENT_RANK + 2 * HKV * HEAD_DIM) * 2) as u64;

#[derive(Debug, Clone, PartialEq)]
pub enum E4ConfigError {
    BadArgument(String),
    SeqNotBlockMultiple {
        seq: usize,
    },
    ZeroBatch,
    PoolTableNotPermutation {
        pool_blocks: usize,
    },
    Plan {
        batch: usize,
        seq_index: usize,
        reason: String,
    },
    Table {
        batch: usize,
        seq_index: usize,
        reason: String,
    },
    TooLarge(Box<Footprint>),
}

impl fmt::Display for E4ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadArgument(m) => write!(f, "bad argument: {m}"),
            Self::SeqNotBlockMultiple { seq } => {
                write!(f, "--seq {seq} must be a positive multiple of {BLOCK_SIZE}")
            }
            Self::ZeroBatch => write!(f, "batch must be positive"),
            Self::PoolTableNotPermutation { pool_blocks } => {
                write!(f, "pool table is not a permutation of {pool_blocks} blocks")
            }
            Self::Plan {
                batch,
                seq_index,
                reason,
            } => write!(f, "E4 plan rejected (B={batch}, b={seq_index}): {reason}"),
            Self::Table {
                batch,
                seq_index,
                reason,
            } => write!(f, "E4 table rejected (B={batch}, b={seq_index}): {reason}"),
            Self::TooLarge(fp) => write!(f, "E4 footprint refused: {}", fp.describe()),
        }
    }
}

impl std::error::Error for E4ConfigError {}

/// Active length of sequence `b` in a batch with context `seq`: b = 0 runs the full context,
/// the others follow `1 + ((b * 373) mod 1024) * seq / 1024`, a deterministic spread in
/// [1, seq] that equals the original `1 + (b * 373) mod 1024` at seq = 1024.
pub fn e4_active_len(b: usize, seq: usize) -> usize {
    if b == 0 {
        seq
    } else {
        1 + ((b * 373) % 1024) * seq / 1024
    }
}

/// Resident layers: enough that the full-KV pools total >= 64 MiB, and at least 4.
pub fn e4_layers(batch: usize, seq: usize) -> usize {
    MIN_LAYERS.max(MIN_POOL_TOKENS_ALL_LAYERS.div_ceil(batch * seq))
}

/// Memory footprint of one E4 configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Footprint {
    pub batch: usize,
    pub seq: usize,
    pub layers: usize,
    /// FP16 K+V pools over all layers.
    pub full_kv_bytes: u64,
    /// FP16 latent pools over all layers.
    pub latent_bytes: u64,
    /// Pools + L2 flush buffer (queries, projections, tables and partials are < 1 MiB extra
    /// and not counted).
    pub device_bytes: u64,
    /// Estimated peak host RAM: retained f32 copies for all layers + transients of one layer.
    pub host_bytes: u64,
    pub max_device_gib: f64,
    pub max_host_gib: f64,
}

const GIB: f64 = (1u64 << 30) as f64;

impl Footprint {
    pub fn fits(&self) -> bool {
        self.device_bytes as f64 <= self.max_device_gib * GIB
            && self.host_bytes as f64 <= self.max_host_gib * GIB
    }

    pub fn describe(&self) -> String {
        format!(
            "B={} N={} layers={} full_kv={:.3}GiB latent={:.3}GiB device_total={:.3}GiB (limit {:.1}GiB) host_est={:.3}GiB (limit {:.1}GiB)",
            self.batch,
            self.seq,
            self.layers,
            self.full_kv_bytes as f64 / GIB,
            self.latent_bytes as f64 / GIB,
            self.device_bytes as f64 / GIB,
            self.max_device_gib,
            self.host_bytes as f64 / GIB,
            self.max_host_gib
        )
    }
}

/// Footprint of B sequences of N tokens over `e4_layers(B, N)` layers.
pub fn e4_footprint(batch: usize, seq: usize, max_device_gib: f64, max_host_gib: f64) -> Footprint {
    let layers = e4_layers(batch, seq);
    let tokens_per_layer = (batch * seq) as u64;
    let all = tokens_per_layer * layers as u64;
    let full_kv_bytes = all * FULL_KV_BYTES_PER_TOKEN;
    let latent_bytes = all * LATENT_BYTES_PER_TOKEN;
    Footprint {
        batch,
        seq,
        layers,
        full_kv_bytes,
        latent_bytes,
        device_bytes: full_kv_bytes + latent_bytes + L2_FLUSH_BYTES,
        host_bytes: all * HOST_RETAINED_BYTES_PER_TOKEN
            + tokens_per_layer * HOST_TRANSIENT_BYTES_PER_TOKEN,
        max_device_gib,
        max_host_gib,
    }
}

/// Refuses a configuration whose device or host footprint exceeds its limit.
pub fn check_footprint(fp: Footprint) -> Result<Footprint, E4ConfigError> {
    if fp.fits() {
        Ok(fp)
    } else {
        Err(E4ConfigError::TooLarge(Box::new(fp)))
    }
}

/// One batch element: its block table into the pool, active length and launch plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct E4SeqLayout {
    pub table: Vec<usize>,
    pub active: usize,
    pub plan: SplitPlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct E4Layout {
    pub batch: usize,
    pub seq: usize,
    pub layers: usize,
    pub pool_blocks: usize,
    pub pool_table: Vec<usize>,
    pub seqs: Vec<E4SeqLayout>,
    pub footprint: Footprint,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Builds and validates the E4 host layout: footprint check first (nothing is allocated if it
/// fails), then the non-identity pool table `(lb * 1021 + 11) mod pool_blocks`, then for every
/// sequence its active length, `plan_bounded` plan and block-table validation.
pub fn e4_layout(
    batch: usize,
    seq: usize,
    max_device_gib: f64,
    max_host_gib: f64,
) -> Result<E4Layout, E4ConfigError> {
    if batch == 0 {
        return Err(E4ConfigError::ZeroBatch);
    }
    if seq == 0 || seq % BLOCK_SIZE != 0 {
        return Err(E4ConfigError::SeqNotBlockMultiple { seq });
    }
    let footprint = check_footprint(e4_footprint(batch, seq, max_device_gib, max_host_gib))?;
    let nb = seq / BLOCK_SIZE;
    let pool_blocks = batch * nb;
    if gcd(1021, pool_blocks) != 1 || i32::try_from(pool_blocks).is_err() {
        return Err(E4ConfigError::PoolTableNotPermutation { pool_blocks });
    }
    let pool_table: Vec<usize> = (0..pool_blocks)
        .map(|lb| (lb * 1021 + 11) % pool_blocks)
        .collect();
    let mut seqs = Vec::with_capacity(batch);
    for b in 0..batch {
        let active = e4_active_len(b, seq);
        let table = pool_table[b * nb..(b + 1) * nb].to_vec();
        let plan =
            plan_bounded(active, BLOCK_SIZE, nb, MIN_SPLITS).map_err(|e| E4ConfigError::Plan {
                batch,
                seq_index: b,
                reason: e.to_string(),
            })?;
        let table_i32: Vec<i32> = table.iter().map(|&v| v as i32).collect();
        validate_block_table(
            &table_i32,
            active,
            BLOCK_SIZE,
            plan.blocks_read,
            pool_blocks,
        )
        .map_err(|e| E4ConfigError::Table {
            batch,
            seq_index: b,
            reason: e.to_string(),
        })?;
        seqs.push(E4SeqLayout {
            table,
            active,
            plan,
        });
    }
    Ok(E4Layout {
        batch,
        seq,
        layers: footprint.layers,
        pool_blocks,
        pool_table,
        seqs,
        footprint,
    })
}

/// Analytical bytes one sequence's launch pair (split + reduce) reads or writes, per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct E4Bytes {
    pub c3: u64,
    pub a3pad: u64,
    /// Cache-only minimum over the active tokens (no padding, no partials).
    pub min_c3: u64,
    pub min_a3: u64,
}

impl std::ops::AddAssign for E4Bytes {
    fn add_assign(&mut self, o: Self) {
        self.c3 += o.c3;
        self.a3pad += o.a3pad;
        self.min_c3 += o.min_c3;
        self.min_a3 += o.min_a3;
    }
}

/// Same accounting as the original harness `e4_bytes` (model_small shape).
pub fn e4_bytes(plan: SplitPlan, active: usize) -> E4Bytes {
    let (hq, hkv, d, l, blk) = (
        HQ as u64,
        HKV as u64,
        HEAD_DIM as u64,
        LATENT_RANK as u64,
        BLOCK_SIZE as u64,
    );
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
    let a3 =
        br * blk * hkv * d * 2 * 2 + s * hkv * hq * d * 4 + a3_partials + table_bytes + hq * d * 4;
    let active_blocks = active.div_ceil(BLOCK_SIZE) as u64;
    E4Bytes {
        c3,
        a3pad: a3,
        min_c3: active_blocks * blk * l * 2,
        min_a3: active_blocks * blk * hkv * d * 2 * 2,
    }
}

/// Bytes for one layer of the whole batch.
pub fn e4_layer_bytes(layout: &E4Layout) -> E4Bytes {
    let mut total = E4Bytes::default();
    for s in &layout.seqs {
        total += e4_bytes(s.plan, s.active);
    }
    total
}

/// Achieved bandwidth in GB/s (1e9 bytes per second) for `bytes` moved in `ms` milliseconds.
pub fn achieved_gbps(bytes: u64, ms: f64) -> Option<f64> {
    (ms > 0.0).then(|| bytes as f64 / (ms * 1e-3) / 1e9)
}

/// E4 command-line options shared by `--suite e4` and `--suite e4-timing`.
#[derive(Debug, Clone, PartialEq)]
pub struct E4Args {
    pub seq: usize,
    /// `--batch B` (e4-timing) or `--batches LIST` (e4); default 1,8,32.
    pub batches: Vec<usize>,
    pub max_footprint_gib: f64,
    pub max_host_footprint_gib: f64,
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].as_str())
}

fn parse<T: std::str::FromStr>(key: &str, v: &str) -> Result<T, E4ConfigError> {
    v.parse()
        .map_err(|_| E4ConfigError::BadArgument(format!("{key} {v:?}")))
}

/// Parses `--seq`, `--batch` / `--batches`, `--max-footprint-gib` and
/// `--max-host-footprint-gib`. Defaults reproduce the original E4 suite (N = 1024, B = 1,8,32).
pub fn parse_e4_args(args: &[String]) -> Result<E4Args, E4ConfigError> {
    let seq = match arg_value(args, "--seq") {
        Some(v) => parse("--seq", v)?,
        None => DEFAULT_SEQ,
    };
    if seq == 0 || seq % BLOCK_SIZE != 0 {
        return Err(E4ConfigError::SeqNotBlockMultiple { seq });
    }
    let batches = match (arg_value(args, "--batch"), arg_value(args, "--batches")) {
        (Some(_), Some(_)) => {
            return Err(E4ConfigError::BadArgument(
                "use --batch or --batches, not both".into(),
            ));
        }
        (Some(v), None) => vec![parse("--batch", v)?],
        (None, Some(v)) => v
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| parse("--batches", s))
            .collect::<Result<_, _>>()?,
        (None, None) => DEFAULT_BATCHES.to_vec(),
    };
    if batches.is_empty() || batches.contains(&0) {
        return Err(E4ConfigError::ZeroBatch);
    }
    let gib = |key: &str, default: f64| -> Result<f64, E4ConfigError> {
        let v = match arg_value(args, key) {
            Some(v) => parse(key, v)?,
            None => default,
        };
        if v.is_finite() && v > 0.0 {
            Ok(v)
        } else {
            Err(E4ConfigError::BadArgument(format!(
                "{key} must be positive"
            )))
        }
    };
    Ok(E4Args {
        seq,
        batches,
        max_footprint_gib: gib("--max-footprint-gib", DEFAULT_MAX_FOOTPRINT_GIB)?,
        max_host_footprint_gib: gib("--max-host-footprint-gib", DEFAULT_MAX_HOST_FOOTPRINT_GIB)?,
    })
}

/// Validates every requested batch for the parsed arguments (footprint, tables, plans) on the
/// host, before the harness opens a GPU stream.
pub fn e4_layouts(a: &E4Args) -> Result<Vec<E4Layout>, E4ConfigError> {
    a.batches
        .iter()
        .map(|&b| e4_layout(b, a.seq, a.max_footprint_gib, a.max_host_footprint_gib))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splitk_plan::MAX_STEPS_PER_CTA;

    fn args(s: &str) -> Vec<String> {
        std::iter::once("bin")
            .chain(s.split_whitespace())
            .map(String::from)
            .collect()
    }

    #[test]
    fn defaults_reproduce_original_e4() {
        let a = parse_e4_args(&args("--suite e4")).unwrap();
        assert_eq!(a.seq, 1024);
        assert_eq!(a.batches, vec![1, 8, 32]);
        assert_eq!(a.max_footprint_gib, 20.0);
        // Original rules: 1 + (b*373) % 1024, layers = max(4, ceil(64 / B)).
        for b in 1..32 {
            assert_eq!(e4_active_len(b, 1024), 1 + (b * 373) % 1024);
        }
        for batch in [1usize, 8, 32] {
            assert_eq!(e4_layers(batch, 1024), 4usize.max(64usize.div_ceil(batch)));
        }
        assert_eq!(e4_layers(1, 1024), 64);
        assert_eq!(e4_layers(8, 1024), 8);
        assert_eq!(e4_layers(32, 1024), 4);
    }

    #[test]
    fn parses_seq_batch_and_limits() {
        let a = parse_e4_args(&args(
            "--suite e4-timing --batch 8 --seq 8192 --max-footprint-gib 6.5",
        ))
        .unwrap();
        assert_eq!(
            (a.seq, a.batches.clone(), a.max_footprint_gib),
            (8192, vec![8], 6.5)
        );
        let a = parse_e4_args(&args("--suite e4 --batches 1,32 --seq 32768")).unwrap();
        assert_eq!((a.seq, a.batches), (32768, vec![1, 32]));
    }

    #[test]
    fn rejects_bad_arguments() {
        for bad in [
            "--seq 1000",
            "--seq 0",
            "--seq x",
            "--batch 0",
            "--batch 8 --batches 1,8",
            "--max-footprint-gib -1",
            "--max-footprint-gib nan",
        ] {
            assert!(parse_e4_args(&args(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn active_lengths_spread_over_whole_context() {
        for seq in [1024usize, 8192, 32768] {
            let actives: Vec<usize> = (0..32).map(|b| e4_active_len(b, seq)).collect();
            assert_eq!(actives[0], seq);
            assert!(actives.iter().all(|&a| (1..=seq).contains(&a)));
            // The spread reaches the upper quarter of the context, not only the first 12K.
            assert!(
                actives[1..].iter().any(|&a| a > 3 * seq / 4),
                "{seq}: {actives:?}"
            );
            assert!(
                actives[1..].iter().any(|&a| a < seq / 4),
                "{seq}: {actives:?}"
            );
        }
    }

    #[test]
    fn footprint_matches_hand_computation() {
        // B = 32, N = 32768: 4 layers * 1 Mi tokens * 1 KiB = 4 GiB full KV, 256 MiB latent.
        let fp = e4_footprint(32, 32768, 20.0, 24.0);
        assert_eq!(fp.layers, 4);
        assert_eq!(fp.full_kv_bytes, 4 << 30);
        assert_eq!(fp.latent_bytes, 256 << 20);
        assert_eq!(fp.device_bytes, (4 << 30) + (256 << 20) + (64 << 20));
        assert!(fp.fits());
    }

    #[test]
    fn footprint_refusal_is_an_error_before_any_table() {
        // B = 32, N = 131072 needs 16 GiB of full KV + latent + flush > 16 GiB limit.
        let err = e4_layout(32, 131072, 16.0, 1e6).unwrap_err();
        match &err {
            E4ConfigError::TooLarge(fp) => {
                assert!(!fp.fits());
                assert!(err.to_string().contains("footprint refused"));
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
        // Host limit is enforced too.
        assert!(matches!(
            e4_layout(32, 32768, 20.0, 1.0),
            Err(E4ConfigError::TooLarge(_))
        ));
        // Through the argument path.
        let a = parse_e4_args(&args("--batch 32 --seq 32768 --max-footprint-gib 2")).unwrap();
        assert!(matches!(e4_layouts(&a), Err(E4ConfigError::TooLarge(_))));
    }

    #[test]
    fn grid_plans_and_tables_are_valid() {
        for seq in [1024usize, 8192, 32768] {
            for batch in [1usize, 8, 32] {
                let lay = e4_layout(batch, seq, 20.0, 24.0)
                    .unwrap_or_else(|e| panic!("B={batch} N={seq}: {e}"));
                let mut sorted = lay.pool_table.clone();
                sorted.sort_unstable();
                assert!(sorted.iter().copied().eq(0..lay.pool_blocks), "permutation");
                assert_ne!(lay.pool_table[0], 0, "non-identity");
                for s in &lay.seqs {
                    assert!(s.plan.splits >= MIN_SPLITS.min(s.active.div_ceil(32)));
                    assert!(s.plan.steps_per_split <= MAX_STEPS_PER_CTA);
                    assert!(s.plan.blocks_read <= seq / BLOCK_SIZE);
                    assert!(s.plan.blocks_read * BLOCK_SIZE >= s.active);
                }
            }
        }
    }

    /// The CPU-only check requested for the L4 campaign: `--seq 8192 --batch 8`.
    #[test]
    fn seq8192_batch8_plans_valid_splits() {
        let a = parse_e4_args(&args("--suite e4-timing --seq 8192 --batch 8")).unwrap();
        let lays = e4_layouts(&a).unwrap();
        assert_eq!(lays.len(), 1);
        let lay = &lays[0];
        assert_eq!(
            (lay.batch, lay.seq, lay.layers, lay.pool_blocks),
            (8, 8192, 4, 4096)
        );
        assert_eq!(lay.seqs[0].active, 8192);
        assert_eq!(lay.seqs[1].active, 1 + 373 * 8);
        assert_eq!(lay.seqs[2].active, 1 + 746 * 8);
        // b = 0 (full 8K): 256 steps -> 4 splits of 64 steps, reads all 512 blocks.
        assert_eq!(
            (
                lay.seqs[0].plan.splits,
                lay.seqs[0].plan.steps_per_split,
                lay.seqs[0].plan.blocks_read
            ),
            (4, 64, 512)
        );
        for s in &lay.seqs {
            assert!(s.plan.splits >= 4 && s.plan.steps_per_split <= MAX_STEPS_PER_CTA);
            assert!(s.plan.blocks_read <= 512);
        }
    }

    #[test]
    fn bytes_match_original_accounting_and_ratio() {
        let lay = e4_layout(32, 1024, 20.0, 24.0).unwrap();
        let b = e4_layer_bytes(&lay);
        // Cache-only minimum ratio is exactly the per-token byte ratio, 1024 / 64 = 16.
        assert_eq!(b.min_a3, 16 * b.min_c3);
        assert!(b.a3pad > b.c3);
        let one = e4_bytes(lay.seqs[0].plan, 1024);
        // Full 1K sequence: 64 blocks * 16 tokens * 32 * 2 bytes of latent.
        assert!(one.c3 >= 64 * 16 * 32 * 2);
        assert_eq!(one.min_c3, 64 * 16 * 32 * 2);
        assert_eq!(achieved_gbps(1_000_000_000, 1000.0), Some(1.0));
        assert_eq!(achieved_gbps(1, 0.0), None);
    }
}
