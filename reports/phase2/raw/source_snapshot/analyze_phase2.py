#!/usr/bin/env python3
"""Aggregate Phase-2 C1 exploratory samples without upgrading them to canonical evidence."""
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/phase2"
RAW = OUT / "raw/c1_exploratory"
SEQS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
VARIANTS = ("A1", "B1", "C1")
COMPONENTS = ("pipeline", "projection", "score", "softmax", "context")
L2 = 33_554_432


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def mad(values: list[float]) -> float:
    m = statistics.median(values)
    return statistics.median(abs(x - m) for x in values)


def load_processes(seq: int) -> dict[int, list[dict]]:
    result = {}
    # Use matched budgets within each length. Long cases use short campaigns because
    # 50-iteration runs showed severe time drift; failed/partial files are retained.
    selected = (1, 2) if seq <= 2048 else ((0, 3, 4) if seq == 4096 else (0, 3))
    expected_lines = 650 if seq <= 2048 else 130
    for process in selected:
        path = RAW / f"seq{seq}_p{process}/samples.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) == expected_lines:
                result[process] = rows
    return result


def stationarity_status(processes: dict[int, list[dict]]) -> str:
    """Flag >20% first/last-quartile pipeline drift; never discard the samples."""
    for rows in processes.values():
        for variant in VARIANTS:
            values = [r["latency_ms"] for r in rows if r["variant"] == variant and r["component"] == "pipeline"]
            q = max(1, len(values) // 4)
            drift = statistics.median(values[-q:]) / statistics.median(values[:q])
            if not 0.8 <= drift <= 1.2:
                return "EXPLORATORY_NONSTATIONARY_BALANCED_POWER"
    return "EXPLORATORY_BALANCED_POWER"


def sample_values(processes: dict[int, list[dict]], variant: str, component: str) -> list[float]:
    return [
        row["latency_ms"] * 1000.0
        for rows in processes.values()
        for row in rows
        if row["variant"] == variant and row["component"] == component
    ]


def process_medians(processes: dict[int, list[dict]], variant: str, component: str) -> list[float]:
    out = []
    for rows in processes.values():
        values = [r["latency_ms"] * 1000.0 for r in rows if r["variant"] == variant and r["component"] == component]
        if values:
            out.append(statistics.median(values))
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_line(path: Path, title: str, series: dict[str, list[tuple[int, float]]], ylabel: str, logx: bool = True) -> None:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 60, 70
    points = [(x, y) for values in series.values() for x, y in values if math.isfinite(y)]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    tx = (lambda x: math.log2(x)) if logx else (lambda x: x)
    xmin, xmax = min(map(tx, xs)), max(map(tx, xs))
    ymin, ymax = min(ys), max(ys)
    pad = max((ymax - ymin) * 0.08, 1e-9)
    ymin, ymax = ymin - pad, ymax + pad
    sx = lambda x: left + (tx(x) - xmin) / (xmax - xmin) * (width - left - right)
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * (height - top - bottom)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>']
    for x in sorted(set(xs)):
        px = sx(x); lines += [f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+5}" stroke="black"/>', f'<text x="{px:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{x}</text>']
    for i in range(6):
        y = ymin + (ymax-ymin)*i/5; py=sy(y)
        lines += [f'<line x1="{left-5}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#ddd"/>', f'<text x="{left-10}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>']
    for idx,(name, values) in enumerate(series.items()):
        color=colors[idx%len(colors)]; pts=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in values if math.isfinite(y))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in values:
            if math.isfinite(y): lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>')
        lines += [f'<line x1="{width-210}" y1="{65+idx*20}" x2="{width-185}" y2="{65+idx*20}" stroke="{color}" stroke-width="3"/>', f'<text x="{width-178}" y="{69+idx*20}" font-family="sans-serif" font-size="12">{name}</text>']
    lines += [f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="13">sequence length (log2 spacing)</text>', f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="13">{ylabel}</text>', '</svg>']
    path.write_text('\n'.join(lines)+'\n')


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    crossover, ablations = [], []
    latency_series = {v: [] for v in VARIANTS}
    ratio_series = {"B1/full": [], "C1/full": []}
    for seq in SEQS:
        processes = load_processes(seq)
        evidence_status = stationarity_status(processes)
        stats = {}
        for variant in VARIANTS:
            for component in COMPONENTS:
                values = sample_values(processes, variant, component)
                if not values:
                    continue
                pmed = process_medians(processes, variant, component)
                row = {
                    "seq_len": seq, "variant": variant, "component": component,
                    "process_count": len(pmed), "sample_count": len(values),
                    "median_us": statistics.median(pmed), "p10_us": quantile(values, .10),
                    "p90_us": quantile(values, .90), "p95_us": quantile(values, .95),
                    "mad_us": mad(values), "min_us": min(values), "max_us": max(values),
                    "evidence_status": evidence_status
                }
                ablations.append(row); stats[(variant, component)] = row
        full = stats[("A1", "pipeline")]["median_us"]
        latent = stats[("B1", "pipeline")]["median_us"]
        c1 = stats[("C1", "pipeline")]["median_us"]
        for variant,value in (("A1",full),("B1",latent),("C1",c1)): latency_series[variant].append((seq,value))
        ratio_series["B1/full"].append((seq,latent/full)); ratio_series["C1/full"].append((seq,c1/full))
        paired_c1_ratios = [c / f for c, f in zip(process_medians(processes, "C1", "pipeline"), process_medians(processes, "A1", "pipeline"))]
        crossover.append({
            "seq_len": seq, "head_dim": 64, "latent_dim": 32, "hq": 16, "hkv": 4,
            "gqa_ratio": 4, "block_size": 16, "full_bytes_token": 1024,
            "latent_bytes_token": 64, "compression_ratio": 16,
            "full_latency_us": full, "latent_latency_us": c1, "ratio": statistics.median(paired_c1_ratios),
            "original_latent_b1_latency_us": latent, "original_latent_b1_ratio": latent/full,
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "commit": commit,
            "working_tree_variant": "phase2_c1_extended_lengths",
            "evidence_status": evidence_status,
        })
    write_csv(OUT/"crossover_results.csv", crossover, list(crossover[0]))
    write_csv(OUT/"ablation_results.csv", ablations, list(ablations[0]))

    cache_rows=[]
    measured={r['seq_len']:r for r in crossover}
    def residency(size):
        ratio=size/L2
        return "L2_RESIDENT" if ratio <= .5 else ("L2_BORDERLINE" if ratio <= 1.5 else "L2_NONRESIDENT")
    for seq in (*SEQS,65536,131072):
        full,latent=seq*1024,seq*64
        cache_rows.append({"seq_len":seq,"l2_bytes":L2,"full_persistent_bytes":full,"latent_persistent_bytes":latent,"full_l2_fraction":full/L2,"latent_l2_fraction":latent/L2,"full_class":residency(full),"latent_class":residency(latent),"c1_full_latency_ratio":measured.get(seq,{}).get('ratio',''),"measurement_status":"EXPLORATORY_BALANCED_POWER" if seq in measured else "MODEL_ONLY_NOT_MEASURED"})
    write_csv(OUT/"cache_residency_results.csv",cache_rows,list(cache_rows[0]))

    ncu=json.loads((ROOT/'reports/p05_hardware_attribution/ncu_focused_metrics.json').read_text())
    profiler=[]
    keys=("gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","lts__t_bytes.sum","lts__t_sector_hit_rate.pct","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum","l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum","smsp__inst_executed.sum","smsp__sass_thread_inst_executed_op_fp32_pred_on.sum","launch__registers_per_thread","launch__shared_mem_per_block","sm__warps_active.avg.pct_of_peak_sustained_active","smsp__issue_active.avg.pct_of_peak_sustained_active","sass__inst_executed_register_spilling_mem_local")
    for kernel,data in ncu['kernels'].items():
        m=data['focused_metrics']; row={"kernel":kernel,"source":"historical_P05_not_C1","kernel_name":data['kernel_name']}
        for key in keys: row[key]=m.get(key,{}).get('value','')
        profiler.append(row)
    write_csv(OUT/"profiler_summary.csv",profiler,list(profiler[0]))

    plots=OUT/'plots'; plots.mkdir(parents=True,exist_ok=True)
    svg_line(plots/'latency_vs_sequence.svg','Exploratory latency (4K+ nonstationary)',latency_series,'pipeline latency (us)')
    svg_line(plots/'ratio_vs_sequence.svg','Exploratory ratio (4K+ nonstationary)',ratio_series,'latency ratio')
    svg_line(plots/'persistent_bytes_vs_sequence.svg','Persistent cache bytes',{"Full KV":[(n,n*1024) for n in SEQS],"Latent":[(n,n*64) for n in SEQS]},'bytes')
    svg_line(plots/'latency_per_token.svg','Exploratory per-token latency (4K+ nonstationary)',{v:[(n,y/n) for n,y in vals] for v,vals in latency_series.items()},'us/token')
    cells=[]
    colors={"FULL":"#D55E00","NEAR PARITY":"#E69F00","LATENT":"#009E73","UNKNOWN":"#999999"}
    for i,row in enumerate(crossover):
        status = "UNKNOWN" if "NONSTATIONARY" in row["evidence_status"] else ("NEAR PARITY" if 0.95 <= row["ratio"] <= 1.05 else ("LATENT" if row["ratio"] < 0.95 else "FULL"))
        x=90+i*85
        cells += [f'<rect x="{x}" y="120" width="75" height="100" fill="{colors[status]}"/>',f'<text x="{x+37.5}" y="245" text-anchor="middle" font-family="sans-serif" font-size="11">{row["seq_len"]}</text>',f'<text x="{x+37.5}" y="175" text-anchor="middle" font-family="sans-serif" font-size="10" fill="white" transform="rotate(-90 {x+37.5} 175)">{status}</text>']
    legend=[]
    for j,label in enumerate(colors):
        legend += [f'<rect x="{120+j*180}" y="300" width="18" height="18" fill="{colors[label]}"/>',f'<text x="{145+j*180}" y="314" font-family="sans-serif" font-size="12">{label}</text>']
    (plots/'phase_map.svg').write_text('\n'.join(['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="380" viewBox="0 0 900 380">','<rect width="100%" height="100%" fill="white"/>','<text x="450" y="40" text-anchor="middle" font-family="sans-serif" font-size="20">Canonical-shape exploratory phase map (near parity ±5%)</text>','<text x="450" y="75" text-anchor="middle" font-family="sans-serif" font-size="12">UNKNOWN marks nonstationary measurements; no parameter dimension was measured.</text>',*cells,*legend,'</svg>'])+'\n')


if __name__ == '__main__':
    main()
