#!/usr/bin/env python3
"""Generate the paper's tables from committed result files (no GPU, standard library only).

Writes Markdown fragments to paper/generated/ and paper/generated/numbers.json, a registry of
every number the tables emit (as formatted) with the file it came from. check_numbers.py
accepts a number in the prose only if it is in that registry, in the whitelist, or verbatim in
a results file. Every value here is read from a file; nothing is typed in by hand.

usage: make_tables.py [--out paper/generated]
"""

import argparse
import json
import lzma
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "reports" / "local_4060_20261101"
E3REP = ROOT / "reports" / "e3_splitk_20261002"
LABEL = "laptop RTX 4060, WSL2, not a cloud result"
PRIMARY_RUNS = {
    "e3": "local4060_b1r_e3_20261101_051132",
    "e4": [
        "local4060_e4n1024v2_20261101_055641",
        "local4060_e4n4096_20261101_054159",
        "local4060_e4n8192_20261101_054853",
    ],
    "e3_prelim": "local4060_b1_e3_20261101_043331",
    "sens": "local4060_e3a3sens_20261101_063414",
    "instrument_failure": "local4060_b2r_e4n1024_20261101_051528",
}
MIB = float(1 << 20)


class Registry:
    """Every number written into a table, with its source file (relative to the repo)."""

    def __init__(self):
        self.numbers = {}

    def add(self, text, source):
        text = str(text)
        self.numbers.setdefault(text, set()).add(str(Path(source).relative_to(ROOT)))
        return text

    def fmt(self, value, spec, source):
        return self.add(format(value, spec), source)

    def dump(self, path):
        data = {k: sorted(v) for k, v in sorted(self.numbers.items())}
        path.write_text(json.dumps(data, indent=1) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def config_sort_key(key):
    kind, cfg = key.split("/")
    m = re.fullmatch(r"(?:B(\d+)_)?N(\d+)", cfg)
    b, n = (int(m.group(1) or 0), int(m.group(2))) if m else (0, 0)
    return (kind, n, b)


# ---- setup ----------------------------------------------------------------------------------
def setup_table(reg):
    run = LOCAL / "runs" / PRIMARY_RUNS["e4"][2]
    env = (run / "environment.txt").read_text().splitlines()
    i = next(k for k, line in enumerate(env) if line.startswith("name, compute_cap"))
    fields = [x.strip() for x in env[i + 1].split(",")]
    args = load(run / "manifest.json")["args"]
    src_env, src_man = run / "environment.txt", run / "manifest.json"
    nvcc = next(line for line in env if line.startswith("Cuda compilation tools"))
    rows = [
        ("GPU", f"{fields[0]} (compute capability {reg.add(fields[1], src_env)}), {LABEL}"),
        ("Memory", reg.add(fields[4], src_env)),
        ("Max SM / memory clock", f"{reg.add(fields[7], src_env)} / {reg.add(fields[8], src_env)}"),
        ("Driver", f"{reg.add(fields[2], src_env)} (WSL2 kernel 6.6.87.2)"),
        ("Toolkit", nvcc.split(",")[1].strip().replace("release ", "CUDA ")),
        ("Shape (model_small)", "Hq 16, Hkv 4 (GQA group 4), head dim 64, latent 32, block 16"),
        (
            "Timing protocol",
            f"{reg.add(args['warmup'], src_man)} warm-up + {reg.add(args['iterations'], src_man)} "
            f"timed iterations per process; >= {reg.add(args['min_cool'], src_man)} cool-start "
            f"processes per configuration, at most {reg.add(args['max_attempts'], src_man)} "
            f"attempts; cool-start below {reg.add(args['cool_c'], src_man)} C, abort at "
            f"{reg.add(args['abort_c'], src_man)} C",
        ),
    ]
    reg.add(fields[0], src_env)
    return table(["Item", "Value"], rows)


# ---- kernel variants ------------------------------------------------------------------------
KERNELS = [
    ("sass", "c3_split_model_small_sm_89.cubin", "C3 split (latent)"),
    ("sass_c3_modes", "c3_split_accurate_model_small_sm_89.cubin", "C3 split, accurate (hi/lo q')"),
    ("sass", "c3_reduce_model_small_sm_89.cubin", "C3 reduce"),
    ("sass", "a3_split_model_small_sm_89.cubin", "A3 split, unpadded (4 query rows)"),
    ("sass_a3_equal_treatment", "a3_split8_model_small_sm_89.cubin", "A3 split, 8-block steps"),
    (
        "sass_a3_equal_treatment",
        "a3_split_pad_model_small_sm_89_spec_occ4.cubin",
        "A3 split, padded to 16 rows + occupancy(4)",
    ),
    (
        "sass_a3_equal_treatment",
        "a3_split_pad_model_small_sm_89.cubin",
        "A3 split, padded, no hint",
    ),
    ("sass_a3_equal_treatment", "a3_reduce_pad_model_small_sm_89.cubin", "A3 reduce (padded)"),
]


def kernel_table(reg):
    rows = []
    for sub, cubin, label in KERNELS:
        src = E3REP / sub / "e3_sass_summary.json"
        rec = next(x for x in load(src) if x["cubin"] == cubin)
        ru, oc = rec["resource_usage"], rec["occupancy_sm89"]
        rows.append(
            (
                label,
                reg.add(rec["opcode_counts"].get("HMMA", 0), src),
                reg.add(ru["REG"], src),
                reg.add(ru["STACK"], src),
                reg.fmt(oc["theoretical_occupancy_pct"], ".1f", src),
            )
        )
    return table(
        [
            "Kernel (model_small, sm_89)",
            "HMMA (static)",
            "Registers",
            "Stack bytes",
            "Theor. occ. %",
        ],
        rows,
    )


# ---- numerics -------------------------------------------------------------------------------
SCALE_RE = re.compile(
    r"SCALE inputs=(\S+) N=(\d+) splits=(\d+) steps_per_split=\d+ (\S+) total_max=(\S+) "
    r"total_mean=\S+ kernel_max=(\S+) kernel_mean=\S+ rounding_only_max=(\S+)"
)


def scale_rows(path):
    return [
        {
            "inputs": m.group(1),
            "n": int(m.group(2)),
            "splits": int(m.group(3)),
            "variant": m.group(4),
            "total": float(m.group(5)),
            "kernel": float(m.group(6)),
            "rounding": float(m.group(7)),
            "total_s": m.group(5),
            "kernel_s": m.group(6),
        }
        for m in map(SCALE_RE.match, Path(path).read_text().splitlines())
        if m
    ]


def numerics_tables(reg):
    src = E3REP / "scale_c3_modes" / "scale_suite.log"
    rows = scale_rows(src)
    out = []
    for inputs in ("uniform", "heavy_tailed"):
        for n in sorted({r["n"] for r in rows}):
            cell = {r["variant"]: r for r in rows if r["inputs"] == inputs and r["n"] == n}
            if not cell:
                continue
            out.append(
                [
                    inputs.replace("_", "-"),
                    f"{reg.add(n, src)} ({reg.add(cell['c3_fast']['splits'], src)})",
                ]
                + [
                    f"{reg.add(cell[v]['total_s'], src)} / {reg.add(cell[v]['kernel_s'], src)}"
                    for v in ("c3_fast", "c3_accurate", "a3pad")
                ]
            )
    scale = table(
        [
            "Inputs",
            "N (splits)",
            "C3 fast: total / kernel",
            "C3 accurate: total / kernel",
            "Padded A3: total / kernel",
        ],
        out,
    )
    emu_src = E3REP / "scale" / "emulate_c3_numerics.out"
    emu = []
    for line in emu_src.read_text().splitlines():
        m = re.match(r"(.+?)\s+max=(\S+) mean=(\S+)", line)
        if m:
            emu.append((m.group(1).strip(), reg.add(m.group(2), emu_src)))
    emulation = table(["Rounding applied (CPU emulation, heavy-tailed, N=8192)", "Max error"], emu)
    return scale, emulation, rows


# ---- timing ---------------------------------------------------------------------------------
def counts(res, runs_by_label):
    clean = len(res["clean_processes"])
    excl = res["excluded_processes"]
    thr = sum(1 for lbl in excl if runs_by_label.get(lbl, {}).get("throttled"))
    hot = sum(
        1 for lbl in excl if runs_by_label.get(lbl, {}).get("start_condition") != "cool-start"
    )
    return clean, thr, hot


def ratio_cell(reg, entry, src):
    if entry.get("geomean_ratio_clean") is None:
        return "—", "—"
    geo = reg.fmt(entry["geomean_ratio_clean"], ".3f", src)
    ci = entry.get("ci95_hierarchical_bootstrap")
    ci_s = f"[{reg.fmt(ci[0], '.3f', src)}, {reg.fmt(ci[1], '.3f', src)}]" if ci else "—"
    return geo, ci_s


def timing_table(reg):
    src = LOCAL / "primary_merged_analysis.json"
    data = load(src)
    runs = {r["label"]: r for r in data["runs"]}
    rows, summary = [], {"eligible": [], "descriptive": [], "byte_ratios": []}
    for key in sorted(data["results"], key=config_sort_key):
        res = data["results"][key]
        e = res["c3_accurate"]
        clean, thr, hot = counts(res, runs)
        geo, ci = ratio_cell(reg, e, src)
        med = e.get("median_ms_clean")
        med_s = "—"
        if med:
            a3_us = reg.fmt(med["a3pad"] * 1000, ".1f", src)
            c3_us = reg.fmt(med["c3_accurate"] * 1000, ".1f", src)
            med_s = f"{a3_us} / {c3_us}"
        ws = reg.fmt(res["working_set_bytes"] / MIB, ".1f", src)
        br = reg.fmt(res["byte_ratio_a3pad_over_c3"], ".1f", src)
        summary["byte_ratios"].append(res["byte_ratio_a3pad_over_c3"])
        if e["claim_eligible"]:
            status = "claim-eligible"
            summary["eligible"].append((key, e["geomean_ratio_clean"]))
        else:
            status = "not enough clean processes"
            summary["descriptive"].append(key)
        kind, cfg = key.split("/")
        rows.append(
            (
                ("E3 " if kind == "e3_timing" else "E4 ") + cfg,
                f"{reg.add(clean, src)} / {reg.add(thr, src)} / {reg.add(hot, src)}",
                med_s,
                geo,
                ci,
                status,
                ws,
                br,
            )
        )
    tab = table(
        [
            "Config",
            "Clean / THR / hot",
            "Median µs per step, A3pad / C3acc",
            "C3acc / A3pad",
            "95 % CI",
            "Status",
            "A3pad MiB per layer",
            "Byte ratio A3pad / C3",
        ],
        rows,
    )
    # Derived numbers used in the prose (computed here, never typed).
    reg.add(len(summary["eligible"]), src)
    reg.add(len(summary["descriptive"]), src)
    reg.add(len(data["results"]), src)
    reg.fmt(min(summary["byte_ratios"]), ".1f", src)
    reg.fmt(max(summary["byte_ratios"]), ".1f", src)
    ratios = [r for _, r in summary["eligible"]]
    reg.fmt(min(ratios), ".3f", src)
    reg.fmt(max(ratios), ".3f", src)
    for _, r in summary["eligible"]:
        reg.fmt(abs(1 - r) * 100, ".1f", src)
        reg.fmt(round(abs(1 - r) * 100), ".0f", src)
    return tab, summary


def sensitivity_table(reg):
    run = LOCAL / "runs" / PRIMARY_RUNS["sens"]
    rows = []
    for fname, baseline in (
        ("variants_analysis_vs_a3.json", "unpadded A3"),
        ("variants_analysis.json", "padded A3"),
    ):
        src = run / fname
        res = load(src)["results"]["e3_timing/N8192"]
        for v, name in (
            ("c3_accurate", "C3 accurate"),
            ("c3_fast", "C3 fast"),
            ("a3", "unpadded A3"),
            ("a3pad", "padded A3"),
        ):
            if v in res and isinstance(res[v], dict) and "claim_eligible" in res[v]:
                geo, ci = ratio_cell(reg, res[v], src)
                rows.append(
                    (f"{name} / {baseline}", geo, ci, reg.add(len(res["clean_processes"]), src))
                )
                if res[v].get("geomean_ratio_clean"):
                    reg.fmt(abs(1 - res[v]["geomean_ratio_clean"]) * 100, ".1f", src)
    return table(["E3 N=8192, one run (secondary)", "Ratio", "95 % CI", "Clean processes"], rows)


def prelim_table(reg):
    src = LOCAL / "runs" / PRIMARY_RUNS["e3_prelim"] / "variants_analysis.json"
    data = load(src)
    runs = {r["label"]: r for r in data["runs"]}
    rows = []
    for key in sorted(data["results"], key=config_sort_key):
        res = data["results"][key]
        clean, thr, hot = counts(res, runs)
        geo, ci = ratio_cell(reg, res["c3_accurate"], src)
        rows.append(
            (
                key.split("/")[1],
                f"{reg.add(clean, src)} / {reg.add(thr, src)} / {reg.add(hot, src)}",
                geo,
                ci,
            )
        )
    return table(
        ["E3 preliminary run (Brave open)", "Clean / THR / hot", "C3acc / A3pad", "95 % CI"], rows
    )


# ---- clock validity -------------------------------------------------------------------------
def read_samples(d):
    p = d / "samples.jsonl.xz"
    text = lzma.open(p, "rt").read() if p.exists() else (d / "samples.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def in_band(rows, component, ref):
    clocks = [
        c
        for r in rows
        if r.get("component") == component
        for c in (r.get("sm_clock_pre_mhz"), r.get("sm_clock_post_mhz"))
        if c is not None
    ]
    return (
        sum(1 for c in clocks if c >= 0.95 * ref) / len(clocks),
        statistics.mode(clocks),
        len(clocks),
    )


CHECKS = [
    ("e4_B1_N1024_dryrun_VOID", "E4 B1_N1024", "none (per-sample reasons query)", "layer"),
    ("e4_B1_N1024_10it_fix1", "E4 B1_N1024", "fix 1, 10 iterations", "layer"),
    ("e3_N1024_50it_prefix", "E3 N=1024", "none (per-sample reasons query)", "pipeline"),
    ("e3_N1024_50it_fix1", "E3 N=1024", "fix 1", "pipeline"),
    ("e3_N8192_50it_prefix", "E3 N=8192", "none (per-sample reasons query)", "pipeline"),
    ("e4_B1_N1024_50it_fix2", "E4 B1_N1024", "fix 2 (saturating re-warm)", "layer"),
    ("e4_B8_N4096_50it_fix2", "E4 B8_N4096", "fix 2 (saturating re-warm)", "layer"),
]


def clock_table(reg):
    rows = []
    for name, cfg, fix, comp in CHECKS:
        d = LOCAL / "instrument_checks" / name
        meta = load(d / "timing_meta.json")
        ref = meta["reference_clock_mhz"]
        frac, modal, n = in_band(read_samples(d), comp, ref)
        src = d / "samples.jsonl.xz"
        rows.append(
            (
                cfg,
                fix,
                reg.add(meta["iterations"], d / "timing_meta.json"),
                reg.add(ref, d / "timing_meta.json"),
                reg.add(modal, src),
                reg.fmt(frac, ".2f", src),
            )
        )
    # The full-length failure of fix 1 (campaign run): in-band per process.
    fail = LOCAL / "runs" / PRIMARY_RUNS["instrument_failure"] / "variants_analysis.json"
    va = load(fail)
    b1 = [
        r
        for r in va["runs"]
        if r.get("config") == "B1_N1024" and r.get("in_band_fraction") is not None
    ]
    if b1:
        man = LOCAL / "runs" / PRIMARY_RUNS["instrument_failure"] / "manifest.json"
        iters = load(man)["args"]["iterations"]
        refs = sorted({r["reference_clock_mhz"] for r in b1})
        lo = min(r["in_band_fraction"] for r in b1)
        hi = max(r["in_band_fraction"] for r in b1)
        rows.append(
            (
                f"E4 B1_N1024 (campaign, {reg.add(len(b1), fail)} processes)",
                "fix 1",
                reg.add(iters, man),
                "/".join(reg.add(x, fail) for x in refs),
                "—",
                f"{reg.fmt(lo, '.3f', fail)}–{reg.fmt(hi, '.3f', fail)}",
            )
        )
    return table(
        ["Process", "Harness fix", "Iterations", "Ref. MHz", "Modal MHz", "In-band fraction"], rows
    )


def dryrun_slowdown(reg):
    """Median layer latency at idle vs reference clock in the VOID dry run (the ~16x figure)."""
    d = LOCAL / "instrument_checks" / "e4_B1_N1024_dryrun_VOID"
    rows = [r for r in read_samples(d) if r.get("component") == "layer" and r["variant"] == "a3pad"]
    ref = load(d / "timing_meta.json")["reference_clock_mhz"]
    lo = statistics.median(r["latency_ms"] for r in rows if r["sm_clock_pre_mhz"] < 1000)
    hi = statistics.median(r["latency_ms"] for r in rows if r["sm_clock_pre_mhz"] >= 0.95 * ref)
    idle = sum(
        1 for r in read_samples(d) if r.get("component") == "layer" and r["sm_clock_pre_mhz"] == 210
    )
    total = sum(1 for r in read_samples(d) if r.get("component") == "layer")
    src = d / "samples.jsonl.xz"
    return {
        "idle_ms": reg.fmt(lo * 1000, ".0f", src),
        "ref_ms": reg.fmt(hi * 1000, ".0f", src),
        "ratio": reg.fmt(lo / hi, ".0f", src),
        "idle_share_pct": reg.fmt(100 * idle / total, ".0f", src),
    }


def t1_slow_output(reg):
    """Trap 1 (2026-10-02, A1/C1 harness): /mnt/c output vs ext4, from the diagnosis E0 runs."""
    e0 = ROOT / "reports" / "diagnosis_20261002" / "e0"
    slow, clean = e0 / "seq1024_run1", e0 / "seq1024_cool2"
    rs, rc = load(slow / "run_summary.json"), load(clean / "run_summary.json")
    cs = load(slow / "clock_attributed_summary.json")
    cc = load(clean / "clock_attributed_summary.json")
    low = cs["median_us"]["LOW/A1/context"]["median"]
    high = cc["median_us"]["HIGH/A1/context"]["median"]
    buckets = cs["bucket_counts"]
    return {
        "slow_wall_s": reg.fmt(rs["wall_s"], ".0f", slow / "run_summary.json"),
        "ext4_wall_s": reg.fmt(rc["wall_s"], ".0f", clean / "run_summary.json"),
        "slow_mode_mhz": reg.add(rs["sm_mode_mhz"], slow / "run_summary.json"),
        "low_share_pct": reg.fmt(
            100 * buckets["LOW"] / sum(buckets.values()),
            ".0f",
            slow / "clock_attributed_summary.json",
        ),
        "a1_context_idle_us": reg.fmt(low, ".1f", slow / "clock_attributed_summary.json"),
        "a1_context_boost_us": reg.fmt(high, ".1f", clean / "clock_attributed_summary.json"),
        "idle_factor": reg.fmt(low / high, ".1f", clean / "clock_attributed_summary.json"),
    }


def nvml_latency(reg):
    """Trap 2: NVML host-call latency (separate micro-measurement, reports/nvml_latency_20261101)."""
    src = ROOT / "reports" / "nvml_latency_20261101" / "nvml_latency.json"
    data = load(src)
    calls = data["calls"]
    reasons = next(v for k, v in calls.items() if "Reasons" in k)
    others = [v["median_ms"] for k, v in calls.items() if "Reasons" not in k]
    return {
        "date_utc": data["date_utc"],
        "driver": reg.add(data["host"]["driver"], src),
        "reps": reg.add(data["reps_per_call"], src),
        "reasons_median_ms": reg.fmt(reasons["median_ms"], ".1f", src),
        "reasons_max_ms": reg.fmt(reasons["max_ms"], ".1f", src),
        "other_median_ms_min": reg.fmt(min(others), ".2f", src),
        "other_median_ms_max": reg.fmt(max(others), ".2f", src),
        "temperature_c": reg.add(data["gpu_temperature_c"]["before"], src),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated"))
    out = Path(ap.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    reg = Registry()
    scale, emulation, _ = numerics_tables(reg)
    timing, summary = timing_table(reg)
    frags = {
        "tab_setup.md": setup_table(reg),
        "tab_kernels.md": kernel_table(reg),
        "tab_numerics_scale.md": scale,
        "tab_numerics_emulation.md": emulation,
        "tab_timing.md": timing,
        "tab_sensitivity.md": sensitivity_table(reg),
        "tab_prelim.md": prelim_table(reg),
        "tab_clock_validity.md": clock_table(reg),
    }
    slow = dryrun_slowdown(reg)
    t1 = t1_slow_output(reg)
    nvml = nvml_latency(reg)
    for name, text in frags.items():
        (out / name).write_text(
            f"<!-- generated by scripts/paper/make_tables.py; do not edit -->\n{text}"
        )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "label": LABEL,
                "claim_eligible": summary["eligible"],
                "descriptive": summary["descriptive"],
                "byte_ratio_range": [min(summary["byte_ratios"]), max(summary["byte_ratios"])],
                "void_dryrun_slowdown": slow,
                "t1_slow_output": t1,
                "t2_nvml_latency": nvml,
            },
            indent=1,
        )
        + "\n"
    )
    reg.dump(out / "numbers.json")
    print(f"MAKE_TABLES_OK fragments={len(frags)} numbers={len(reg.numbers)} out={out}")


if __name__ == "__main__":
    main()
