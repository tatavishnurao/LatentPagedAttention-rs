"""Part B: architecture census of SELECTED families (not a sample of all architectures).

Every cell is backed by a source key (see SOURCES). Values: 1 = present, 0 = absent (checked),
None = unknown / not documented in the sources read, "na" = the site does not exist.
Writes data/census.json, data/census_table.md, figures/census_heatmap.png, figures/census_timeline.png."""
import json
import os
import numpy as np
from style import plt, C, INK2, save, dump, ROOT

SOURCES = {
    "cfg": "official config.json on the Hugging Face Hub (read in this session; sources/configs/)",
    "code": "Hugging Face transformers reference code @3568df44 (or the vendor's own reference code where named)",
    "paper": "technical report / paper full text (arXiv id in the row)",
    "card": "model card README on the Hub",
}

SITES = ["self-attention", "cross-attention", "vocabulary head", "MoE router", "contrastive loss"]
MODS = ["QK-norm", "logit soft-capping", "length/temperature logit scaling", "learned sink logit",
        "attention output gate", "local / sliding-window layers", "linear-attention or SSM layers",
        "sigmoid MoE router", "sigmoid (non-softmax) loss", "z-loss", "vocab logit multiplier"]

# shorthand constructors -------------------------------------------------------------------------
def row(family, group, date, attn, ctx, vocab, experts, router, sites, mods, src, notes=""):
    s = {k: sites.get(k, 0) for k in SITES}
    m = {k: mods.get(k, 0) for k in MODS}
    return dict(family=family, group=group, date=date, attention=attn, context=ctx, vocab=vocab,
                experts=experts, router=router, sites=s, mods=m, sources=src, notes=notes)

DEC = {"self-attention": 1, "vocabulary head": 1, "cross-attention": "na", "MoE router": "na", "contrastive loss": "na"}
def dec(**kw):
    d = dict(DEC); d.update(kw); return d
U = None  # unknown

ROWS = [
    # ------------------------------ dense decoders ------------------------------
    row("GPT-2", "dense", "2019", "MHA", 1024, 50257, "-", "-", dec(),
        {"z-loss": U}, ["cfg:openai-community/gpt2"], "release year as cited in arXiv 2005.14165"),
    row("GPT-3", "dense", "2020-05", "MHA; alternating dense / locally banded sparse", 2048, U, "-", "-", dec(),
        {"local / sliding-window layers": 1, "z-loss": U}, ["paper:2005.14165"]),
    row("Llama 1", "dense", "2023-02", "MHA", 2048, U, "-", "-", dec(), {"z-loss": U},
        ["paper:2302.13971", "paper:2307.09288 (Table 1)"]),
    row("Llama 2", "dense", "2023-07", "MHA (7B,13B); GQA (34B,70B)", 4096, 32000, "-", "-", dec(), {"z-loss": U},
        ["paper:2307.09288"], "official config gated (403) in this session"),
    row("Llama 3.1", "dense", "2024-07", "GQA (8 KV heads)", 131072, 128256, "-", "-", dec(), {"z-loss": U},
        ["cfg:meta-llama/Llama-3.1-8B", "paper:2407.21783"], "rope_type llama3, factor 8: frequency change, no logit multiplier"),
    row("Mistral 7B", "dense", "2023-10", "GQA (8 KV) + sliding window 4096", 32768, 32000, "-", "-", dec(),
        {"local / sliding-window layers": 1, "z-loss": U}, ["cfg:mistralai/Mistral-7B-v0.1", "paper:2310.06825"]),
    row("Qwen2 / Qwen2.5", "dense", "2024-07", "GQA; QKV bias", 131072, 152064, "-", "-", dec(),
        {"length/temperature logit scaling": 1, "z-loss": U}, ["cfg:Qwen/Qwen2-7B", "cfg:Qwen/Qwen2.5-7B", "paper:2407.10671"],
        "YaRN + Dual Chunk Attention for long context (report); YaRN not in the shipped config"),
    row("Qwen3 (dense)", "dense", "2025-05", "GQA; QK-norm; no QKV bias", 40960, 151936, "-", "-", dec(),
        {"QK-norm": 1, "length/temperature logit scaling": 1, "z-loss": U},
        ["cfg:Qwen/Qwen3-8B", "code:modeling_qwen3", "paper:2505.09388", "card:Qwen/Qwen3-8B"],
        "32,768 native; 131,072 with static YaRN factor 4 (card, opt-in)"),
    row("Gemma 1", "dense", "2024-03", "MHA (7B); MQA (2B)", 8192, 256000, "-", "-", dec(), {"z-loss": U},
        ["cfg:google/gemma-7b", "paper:2403.08295"]),
    row("Gemma 2", "dense", "2024-07", "GQA; local 4096 / global alternating", 8192, 256000, "-", "-", dec(),
        {"logit soft-capping": 1, "local / sliding-window layers": 1, "z-loss": U},
        ["cfg:google/gemma-2-9b", "code:modeling_gemma2", "paper:2408.00118"], "soft-cap 50 (attention), 30 (final logits)"),
    row("Gemma 3", "dense", "2025-03", "GQA; 5 local (1024) : 1 global; QK-norm", 131072, 262144, "-", "-", dec(),
        {"QK-norm": 1, "local / sliding-window layers": 1, "z-loss": U},
        ["cfg:google/gemma-3-1b-it", "cfg:google/gemma-3-27b-it", "code:modeling_gemma3", "paper:2503.19786"],
        "soft-capping replaced by QK-norm (report); 1B config has soft-capping null; 1B context 32K"),
    row("Phi-3-mini / Phi-4", "dense", "2024-04", "MHA (Phi-3-mini); GQA 40/10 (Phi-4)", 16384, 100352, "-", "-", dec(),
        {"local / sliding-window layers": U, "z-loss": U},
        ["cfg:microsoft/Phi-3-mini-4k-instruct", "cfg:microsoft/phi-4", "paper:2404.14219"],
        "Phi-3-mini config lists sliding_window 2047; whether it is active at 4K context not checked. Context/vocab shown: Phi-4"),
    row("OLMo 2", "dense", "2024-12", "MHA; QK-norm", 4096, 100352, "-", "-", dec(),
        {"QK-norm": 1, "z-loss": 1}, ["cfg:allenai/OLMo-2-1124-7B", "code:modeling_olmo2", "paper:2501.00656"]),
    row("Falcon", "dense", "2023-11", "MQA (7B); multigroup (40B,180B)", 2048, 65024, "-", "-", dec(), {"z-loss": U},
        ["cfg:tiiuae/falcon-7b", "paper:2311.16867"]),
    row("Command R", "dense", "2024-03", "not verified (config gated)", 131072, U, "-", "-", dec(),
        {"QK-norm": U, "vocab logit multiplier": 1, "z-loss": U},
        ["card:CohereLabs/c4ai-command-r-v01", "code:modeling_cohere"],
        "HF code multiplies vocab logits by logit_scale (default 0.0625); released value not read (gated)"),
    row("GLM-4.5", "MoE", "2025-08", "GQA 96 q / 8 KV; partial RoPE; QK-norm", 131072, 151552, "160 (+1 shared), top-8",
        "sigmoid + loss-free balance bias", dec(**{"MoE router": 1}),
        {"QK-norm": 1, "sigmoid MoE router": 1, "z-loss": U},
        ["cfg:zai-org/GLM-4.5", "code:modeling_glm4_moe", "paper:2508.06471"], "router: sigmoid scores, no softmax (code)"),
    row("gpt-oss", "MoE", "2025-08", "GQA 64 q / 8 KV; alternating banded window 128 / dense", 131072, 201088,
        "128 (120b) / 32 (20b), top-4", "softmax over the selected top-4", dec(**{"MoE router": 1}),
        {"learned sink logit": 1, "local / sliding-window layers": 1, "length/temperature logit scaling": 1, "z-loss": U},
        ["cfg:openai/gpt-oss-20b", "cfg:openai/gpt-oss-120b", "code:modeling_gpt_oss", "paper:2508.10925"],
        "YaRN factor 32 in config (logits x1.813); per-head learned bias in the softmax denominator"),
    # ------------------------------ MoE ------------------------------
    row("Mixtral 8x7B", "MoE", "2024-01", "GQA (8 KV)", 32768, 32000, "8, top-2", "softmax over top-2 logits",
        dec(**{"MoE router": 1}), {"z-loss": U}, ["cfg:mistralai/Mixtral-8x7B-v0.1", "code:modeling_mixtral", "paper:2401.04088"]),
    row("DeepSeek-V2", "MoE", "2024-05", "MLA (kv_lora 512)", 163840, 102400, "160 (+2 shared), top-6",
        "softmax (scoring_func), group-limited greedy", dec(**{"MoE router": 1}),
        {"length/temperature logit scaling": 1, "z-loss": U}, ["cfg:deepseek-ai/DeepSeek-V2", "paper:2405.04434"],
        "YaRN factor 40, mscale_all_dim 0.707: softmax scale x1.590 (DeepSeek code path)"),
    row("DeepSeek-V3", "MoE", "2024-12", "MLA (kv_lora 512)", 163840, 129280, "256 (+1 shared), top-8",
        "sigmoid + bias (noaux_tc), normalised over selected", dec(**{"MoE router": 0}),
        {"length/temperature logit scaling": 1, "sigmoid MoE router": 1, "z-loss": U},
        ["cfg:deepseek-ai/DeepSeek-V3", "code:modeling_deepseek_v3", "paper:2412.19437"],
        "YaRN factor 40, mscale_all_dim 1: softmax scale x1.874"),
    row("Qwen3 MoE", "MoE", "2025-05", "GQA (4 KV); QK-norm", 40960, 151936, "128, top-8, no shared", "softmax, renormalised top-k",
        dec(**{"MoE router": 1}), {"QK-norm": 1, "z-loss": U},
        ["cfg:Qwen/Qwen3-30B-A3B", "code:modeling_qwen3_moe", "paper:2505.09388"]),
    row("Llama 4 (Scout/Maverick)", "MoE", "2025-04", "chunked local (8192) RoPE layers + global NoPE layers every 4th",
        "10M (Scout) / 1M (Maverick)", 202048, "16 / 128 (+1 shared), top-1", "sigmoid of top-1 score",
        dec(**{"MoE router": 0}),
        {"QK-norm": 1, "length/temperature logit scaling": 1, "local / sliding-window layers": 1, "sigmoid MoE router": 1, "z-loss": U},
        ["code:llama-models/models/llama4 @0e0b8c5", "code:modeling_llama4", "card:llama-models MODEL_CARD"],
        "config gated (403): QK-norm, temperature tuning and vocab are HF config defaults, not the released config"),
    row("Kimi K2", "MoE", "2025-07", "MLA, 64 heads", 131072, 163840, "384 (+1 shared), top-8", "sigmoid + bias (noaux_tc)",
        dec(**{"MoE router": 0}), {"length/temperature logit scaling": 1, "sigmoid MoE router": 1, "z-loss": U},
        ["cfg:moonshotai/Kimi-K2-Instruct", "paper:2507.20534"],
        "YaRN factor 32 (config); QK-Clip rescales W_q/W_k during training (not a forward-pass change)"),
    # ------------------------------ hybrid / non-softmax ------------------------------
    row("Mamba", "hybrid", "2023-12", "none (selective SSM)", U, 50280, "-", "-",
        dec(**{"self-attention": 0}), {"linear-attention or SSM layers": 1, "z-loss": U},
        ["cfg:state-spaces/mamba-2.8b-hf", "paper:2312.00752"]),
    row("Mamba-2", "hybrid", "2024-05", "none (SSD)", U, 50277, "-", "-", dec(**{"self-attention": 0}),
        {"linear-attention or SSM layers": 1, "z-loss": U}, ["cfg:state-spaces/mamba2-2.7b", "paper:2405.21060"]),
    row("RWKV-6 (Finch)", "hybrid", "2024-04", "none (matrix-valued recurrence)", U, 65536, "-", "-",
        dec(**{"self-attention": 0}), {"linear-attention or SSM layers": 1, "z-loss": U},
        ["cfg:RWKV/v6-Finch-1B6-HF", "paper:2404.05892"]),
    row("Jamba / Jamba-1.5", "hybrid", "2024-03", "1 attention : 7 Mamba layers; no positional encoding", 262144, 65536,
        "16, top-2, every 2nd layer", "softmax", dec(**{"MoE router": 1}),
        {"linear-attention or SSM layers": 1, "z-loss": U},
        ["paper:2403.19887", "paper:2408.12570", "card:ai21labs/AI21-Jamba-1.5-Mini", "code:configuration_jamba/modeling_jamba"],
        "config gated (403); layer ratios from paper and HF config defaults"),
    row("MiniMax-Text-01", "hybrid", "2025-01", "7 lightning (linear) : 1 softmax attention, GQA 8 KV", "1M train / 4M inference",
        200064, "32, top-2", "softmax", dec(**{"MoE router": 1}),
        {"linear-attention or SSM layers": 1, "attention output gate": 1, "z-loss": U},
        ["cfg:MiniMaxAI/MiniMax-Text-01", "code:modeling_minimax", "paper:2501.08313"],
        "sigmoid output gate is in the lightning-attention block (HF code)"),
    row("Qwen3-Next", "hybrid", "2025-09", "3 Gated DeltaNet : 1 gated softmax attention; QK-norm", 262144, 151936,
        "512 (+1 shared), top-10", "softmax, renormalised top-k", dec(**{"MoE router": 1}),
        {"QK-norm": 1, "attention output gate": 1, "linear-attention or SSM layers": 1, "z-loss": U},
        ["cfg:Qwen/Qwen3-Next-80B-A3B-Instruct", "code:modeling_qwen3_next", "paper:2505.06708 (gating method)"]),
    row("RecurrentGemma (Griffin)", "hybrid", "2024-04", "2 RG-LRU : 1 local MQA attention (window 2048)", U, 256000, "-", "-",
        dec(), {"logit soft-capping": 1, "local / sliding-window layers": 1, "linear-attention or SSM layers": 1, "z-loss": U},
        ["cfg:google/recurrentgemma-2b", "paper:2404.07839", "paper:2402.19427"], "final-logit soft-cap 30 (config)"),
    # ------------------------------ encoder / multimodal ------------------------------
    row("BERT", "encoder/multimodal", "2018-10", "bidirectional MHA", 512, 30522, "-", "-",
        {"self-attention": 1, "vocabulary head": 1, "cross-attention": "na", "MoE router": "na", "contrastive loss": "na"},
        {"z-loss": U}, ["cfg:google-bert/bert-base-uncased", "paper:1810.04805"], "vocabulary softmax = MLM head (training)"),
    row("T5", "encoder/multimodal", "2019-10", "encoder-decoder MHA; relative position bias", 512, 32128, "-", "-",
        {"self-attention": 1, "cross-attention": 1, "vocabulary head": 1, "MoE router": "na", "contrastive loss": "na"},
        {"z-loss": U}, ["cfg:google-t5/t5-base", "paper:1910.10683"]),
    row("ViT", "encoder/multimodal", "2020-10", "bidirectional MHA over patches", "197 tokens (224px/16)", "-", "-", "-",
        {"self-attention": 1, "vocabulary head": "na", "cross-attention": "na", "MoE router": "na", "contrastive loss": "na"},
        {"z-loss": U}, ["cfg:google/vit-base-patch16-224", "paper:2010.11929"], "class-label softmax head instead of a vocabulary"),
    row("CLIP", "encoder/multimodal", "2021-02", "ViT image encoder + causal text encoder", 77, 49408, "-", "-",
        {"self-attention": 1, "vocabulary head": "na", "cross-attention": "na", "MoE router": "na", "contrastive loss": 1},
        {"z-loss": U}, ["cfg:openai/clip-vit-base-patch32", "paper:2103.00020"], "symmetric softmax contrastive loss, learned temperature"),
    row("SigLIP", "encoder/multimodal", "2023-03", "ViT image encoder + text encoder", U, 32000, "-", "-",
        {"self-attention": 1, "vocabulary head": "na", "cross-attention": "na", "MoE router": "na", "contrastive loss": 0},
        {"sigmoid (non-softmax) loss": 1, "z-loss": U}, ["cfg:google/siglip-base-patch16-224", "paper:2303.15343"],
        "pairwise sigmoid loss replaces the softmax contrastive loss"),
    row("Whisper", "encoder/multimodal", "2022-12", "encoder-decoder MHA", "30 s audio / 448 text tokens", 51866, "-", "-",
        {"self-attention": 1, "cross-attention": 1, "vocabulary head": 1, "MoE router": "na", "contrastive loss": "na"},
        {"z-loss": U}, ["cfg:openai/whisper-large-v3", "paper:2212.04356"]),
]


def main():
    dump({"sources": SOURCES, "sites": SITES, "mods": MODS, "rows": ROWS,
          "note": "Census of selected families; not a percentage of all architectures."}, "census")
    # ------------- markdown table -------------
    lines = ["| Family | Date | Attention | Context | Vocab | Experts | Router | Softmax modifications | Sources |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ROWS:
        mods = [k for k, v in r["mods"].items() if v == 1]
        unk = [k for k, v in r["mods"].items() if v is None and k != "z-loss"]
        ms = ", ".join(mods) if mods else "none found"
        if unk:
            ms += " (unknown: " + ", ".join(unk) + ")"
        fmt = lambda v: "unknown" if v is None else (f"{v:,}" if isinstance(v, int) else str(v))
        lines.append(f"| {r['family']} | {r['date']} | {r['attention']} | {fmt(r['context'])} | {fmt(r['vocab'])} | "
                     f"{r['experts']} | {r['router']} | {ms} | {'; '.join(r['sources'])} |")
    with open(os.path.join(ROOT, "data", "census_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # ------------- heatmap -------------
    cols = SITES + MODS
    M = np.full((len(ROWS), len(cols)), np.nan)
    code = {1: 2, 0: 0, None: 1, "na": -1}
    for i, r in enumerate(ROWS):
        for j, c in enumerate(cols):
            v = r["sites"][c] if c in SITES else r["mods"][c]
            M[i, j] = code[v]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f2f2f0", "#ffffff", "#e9e3c9", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(11.5, 11.5))
    ax.imshow(M, cmap=cmap, vmin=-1.5, vmax=2.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = {2: "●", 0: "", 1: "?", -1: "–"}[int(M[i, j])]
            ax.text(j, i, t, ha="center", va="center", fontsize=8, color="white" if M[i, j] == 2 else INK2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f"{r['family']}  ({r['date'][:4]})" for r in ROWS], fontsize=8.5)
    ax.axvline(len(SITES) - 0.5, color="black", lw=1.2)
    groups = [r["group"] for r in ROWS]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="black", lw=0.8)
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True); ax.set_yticks(np.arange(-.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d4", lw=0.4); ax.tick_params(which="minor", length=0)
    ax.set_title("Census of selected families: softmax sites (left) and modifications (right)\n"
                 "● present   blank = absent in sources read   ? = unknown / not documented   – = site does not exist",
                 fontsize=10)
    save(fig, "census_heatmap")
    # ------------- timeline: first appearance within this census -------------
    first = {}
    for r in sorted(ROWS, key=lambda r: r["date"]):
        for k, v in r["mods"].items():
            if v == 1 and k not in first:
                first[k] = (r["date"], r["family"])
        if r["sites"]["contrastive loss"] == 1 and "softmax contrastive loss" not in first:
            first["softmax contrastive loss"] = (r["date"], r["family"])
        if r["sites"]["MoE router"] == 1 and "softmax MoE router" not in first:
            first["softmax MoE router"] = (r["date"], r["family"])
    items = sorted(first.items(), key=lambda kv: kv[1][0])
    def yr(d):
        p = d.split("-"); return int(p[0]) + (int(p[1]) - 0.5) / 12 if len(p) > 1 else int(p[0]) + 0.5
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, (k, (d, fam)) in enumerate(items):
        x = yr(d)
        ax.plot([x], [i], "o", color=C[i % len(C)], ms=7)
        ax.text(x + 0.12, i, f"{k} — {fam} ({d})", va="center", fontsize=8.2)
    ax.set_yticks([]); ax.set_ylim(-1, len(items))
    ax.set_xlim(2017.5, 2028.2)
    ax.set_xlabel("date of the family's report, card or release (year)")
    ax.set_title("First appearance of each softmax modification within this census (not a global first)")
    save(fig, "census_timeline")
    dump({k: {"date": d, "family": f} for k, (d, f) in items}, "census_timeline")
    for k, (d, f) in items:
        print(f"{d}  {k:36s} {f}")


if __name__ == "__main__":
    main()
