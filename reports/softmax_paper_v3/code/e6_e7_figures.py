"""Figures and summary tables for E6 (real-model attention probes) and E7 (log-prob precision).
Reads data/e6_<model>.json and data/e7_<model>.json; writes figures/e6_*.png, figures/e7_logprobs.png and
data/e6_e7_summary.json. All numbers in the paper's E6/E7 tables come from that summary file."""
import json
import math
import os
import numpy as np
from style import plt, C, INK2, save, dump, DATA

MODELS = ["qwen3-0.6b", "gemma3-1b-it", "smollm2-360m", "vanilla-340m", "softpick-340m"]
LABEL = {"qwen3-0.6b": "Qwen3-0.6B", "gemma3-1b-it": "Gemma-3-1B-it", "smollm2-360m": "SmolLM2-360M",
         "vanilla-340m": "vanilla-340M (softmax)", "softpick-340m": "softpick-340M"}


def load(kind, m):
    p = os.path.join(DATA, f"{kind}_{m}.json")
    return json.load(open(p)) if os.path.exists(p) else None


E6 = {m: load("e6", m) for m in MODELS}
E7 = {m: load("e7", m) for m in MODELS}
summary = {"e6": {}, "e7": {}}

# ---------- per-model E6 summary ----------
for m, d in E6.items():
    if d is None:
        continue
    Ns = [int(n) for n in d["Ns"]]
    rows = []
    for n in Ns:
        s = d["stats"][str(n)]
        glob = [i for i, w in enumerate(s["sliding"]) if not w]  # global (non-sliding) layers
        ent = np.array(s["entropy"])[glob]
        nvis = np.array(s["n_visible"])[glob]
        rows.append(dict(N=n, sink_rate_0p3=s["sink_rate_0p3"], sink_rate_0p8=s["sink_rate_0p8"],
                         mean_first_token_mass=float(np.mean(s["sink"])), max_layer_first_token_mass=float(np.max(s["sink"])),
                         mean_entropy_global=float(ent.mean()), mean_ln_visible=float(np.log(nvis).mean()),
                         entropy_over_ln_visible=float((ent / np.log(nvis)).mean()),
                         mean_row_mass=float(np.mean(s.get("mass", [1.0])))))
    # slope of global-layer mean entropy vs ln N
    x = np.log([r["mean_ln_visible"] and math.exp(r["mean_ln_visible"]) for r in rows])
    y = np.array([r["mean_entropy_global"] for r in rows])
    slope = float(np.polyfit(x, y, 1)[0]) if len(rows) > 1 else None
    summary["e6"][m] = dict(rows=rows, entropy_slope_vs_lnN=slope, needle=d["needle_summary"],
                            n_global_layers=int(sum(1 for w in d["stats"][str(Ns[0])]["sliding"] if not w)),
                            n_layers=len(d["stats"][str(Ns[0])]["sliding"]), bos=d["bos"])

# ---------- figure: sinks ----------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
for i, m in enumerate(MODELS):
    if m not in summary["e6"]:
        continue
    r = summary["e6"][m]["rows"]
    Ns = [x["N"] for x in r]
    axes[0].plot(Ns, [x["sink_rate_0p3"] for x in r], marker="o", ms=4, color=C[i], label=LABEL[m])
    axes[1].plot(Ns, [x["mean_first_token_mass"] for x in r], marker="o", ms=4, color=C[i], label=LABEL[m])
axes[0].set_ylabel("share of heads with mean weight on token 0 > 0.3")
axes[1].set_ylabel("mean attention weight on token 0 (all heads)")
axes[0].set_title("(a) Sink rate (ε = 0.3, Gu et al. definition)")
axes[1].set_title("(b) Mean first-token mass")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length N (tokens, log scale)")
    ax.set_ylim(bottom=0)
axes[1].legend(fontsize=7.8)
fig.suptitle("E6a. Attention sinks vs context length (last 64 queries; 3 Gutenberg passages per point; CPU fp32)", y=1.02, fontsize=11)
save(fig, "e6_sinks")

# ---------- figure: entropy ----------
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for i, m in enumerate(MODELS):
    if m not in summary["e6"]:
        continue
    r = summary["e6"][m]["rows"]
    ax.plot([x["N"] for x in r], [x["entropy_over_ln_visible"] for x in r], marker="o", ms=4, color=C[i], label=LABEL[m])
ax.axhline(1, color=INK2, lw=1, ls=":")
ax.text(300, 0.97, "uniform attention (H = ln N)", fontsize=7.5, color=INK2, va="top")
ax.set_xscale("log", base=2)
ax.set_ylim(0, 1.05)
ax.set_xlabel("context length N (tokens, log scale)")
ax.set_ylabel("mean entropy / ln(keys visible)  (global layers)")
ax.set_title("E6b. Attention entropy relative to uniform")
ax.legend(fontsize=7.5)
save(fig, "e6_entropy")

# ---------- figure: needle ----------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
for i, m in enumerate(MODELS):
    if m not in summary["e6"]:
        continue
    nd = summary["e6"][m]["needle"]["rows"]
    Ns = np.array([x["N"] for x in nd])
    wf = np.array([x["w_fixed"] for x in nd])
    axes[0].plot(Ns, wf, marker="o", ms=4, color=C[i], label=LABEL[m])
    D0 = nd[0]["D_fixed"]
    axes[0].plot(Ns, np.exp(D0) / (np.exp(D0) + Ns - 1), color=C[i], lw=1, ls=":")
    axes[1].plot(Ns, [x["w_best"] for x in nd], marker="o", ms=4, color=C[i], label=LABEL[m])
axes[0].set_title("(a) Fixed head (best at N = 256); dotted: fixed-gap law")
axes[1].set_title("(b) Best head at each N")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("context length N (tokens, log scale)")
    ax.set_ylabel("weight of final query on needle answer tokens")
axes[1].legend(fontsize=7.8)
fig.suptitle("E6c. Planted-needle retrieval weight (3 depths per point; CPU fp32)", y=1.02, fontsize=11)
save(fig, "e6_needle")

# ---------- E7 ----------
for m, d in E7.items():
    if d is None:
        continue
    summary["e7"][m] = d
ms = [m for m in MODELS if m in summary["e7"]]
if ms:
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    meth = [("fp32", "fp32 log_softmax"), ("logsm", "fp32 log(softmax)"), ("bf16in", "bf16 logits → fp32 log_softmax"),
            ("bf16", "bf16 log_softmax")]
    x = np.arange(len(ms))
    w = 0.2
    for j, (k, lab) in enumerate(meth):
        vals = [summary["e7"][m]["buckets"]["0.9"][k]["exact_zeros"] / max(1, summary["e7"][m]["buckets"]["0.9"]["n"]) for m in ms]
        ax.bar(x + (j - 1.5) * w, vals, w, color=C[j], label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in ms], fontsize=8, rotation=12)
    ax.set_ylabel("share of tokens with p > 0.9 whose log p is exactly 0")
    ax.set_title("E7. Exact-zero log-probabilities on real text (3 × 2,048 tokens per model)")
    ax.legend(fontsize=7.5)
    save(fig, "e7_logprobs")

dump(summary, "e6_e7_summary")
for m, s in summary["e6"].items():
    r0, r1 = s["rows"][0], s["rows"][-1]
    nd = s["needle"]["rows"]
    print(f"{m}: N {r0['N']}->{r1['N']}: sink rate {r0['sink_rate_0p3']:.3f}->{r1['sink_rate_0p3']:.3f}; first-token mass "
          f"{r0['mean_first_token_mass']:.3f}->{r1['mean_first_token_mass']:.3f}; H/lnN {r0['entropy_over_ln_visible']:.3f}->"
          f"{r1['entropy_over_ln_visible']:.3f}; entropy slope vs lnN {s['entropy_slope_vs_lnN']:.3f}; needle fixed w "
          f"{nd[0]['w_fixed']:.3f}->{nd[-1]['w_fixed']:.4f} (D {nd[0]['D_fixed']:.2f}->{nd[-1]['D_fixed']:.2f}); best w "
          f"{nd[0]['w_best']:.3f}->{nd[-1]['w_best']:.3f}; top1 {[x['top1_rate'] for x in nd]}")
for m, d in summary["e7"].items():
    b = d["buckets"]["0.9"]
    print(f"E7 {m}: p>0.9 n={b['n']} zeros fp32={b['fp32']['exact_zeros']} logsm={b['logsm']['exact_zeros']} "
          f"bf16in={b['bf16in']['exact_zeros']} bf16={b['bf16']['exact_zeros']}; min p at bf16 zero {d['min_p_at_exact_zero']['bf16']}")
