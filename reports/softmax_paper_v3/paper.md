# Softmax Numerics and Attention Sinks in LLM Serving

**A reproduction and measurement study**

*Tata Vishnu Vardhan Rao · Anurag University, Hyderabad · November 2026 · **working draft** · code, data and logs published with this page*

> **Draft status (4 Nov 2026).** This draft uses every result available in the session so far. Real-model results cover the matched vanilla/softpick 340M pair only; SmolLM2-360M, Qwen3-0.6B and Gemma-3-1B-it are still running, and E7 (log-prob precision on real text) has not run yet. Sections that depend on them are marked **Pending**. The GPU experiments (B4–B6) are scripts that have not been run.

> **Abstract.** Serving engines compute softmax in every attention head and in the vocabulary head. We re-test four published claims about it: dispersion with growing context and an adaptive-temperature fix, first-token attention sinks, Softmax Collapse, and Softmax Cancellation. We use fine-grid CPU reproductions and the papers' own published code. Three findings: the published collapse cliffs (5.5, 16.6, 36.7) are two-class values, and CPU PyTorch collapses up to 2.1 nats earlier with ≥113 classes; the adaptive-temperature polynomial only acts for row entropies of 0.85–5.95 nats; in a matched 340M pair, the softmax model's sink rate falls from 0.63 to 0.11 between 256 and 4,096 tokens, while softpick shows none. We also locate where vLLM and SGLang compute returned log-probabilities; SGLang's RL on-policy mode uses bfloat16. Code, data and validation logs are released.

## 1 Introduction

A served language model evaluates softmax in every attention head and again in the vocabulary head. Recent papers make four claims about it that matter for serving:
- global attention must disperse as context grows, and an entropy-based adaptive temperature partly counters this [@velickovic, §2–3];
- the first-token attention sink is a learned mechanism, strongest in large, long-context models [@barbero, §3–4];
- standard cross-entropy has a "precision cliff" beyond which the target gradient is exactly zero (Softmax Collapse) [@prieto, §3.1; @barron, §2];
- soft-target cross-entropy leaves a non-zero gradient when student and teacher agree (Softmax Cancellation) [@barron, §3].

Serving engines inherit these behaviours. Eviction policies decide which keys to keep [@streamingllm, §3], kernels fold learned sinks into the normaliser [@flashattn, softmax.py], and RL trainers consume returned log-probabilities [@sglang, sampler.py].

This paper re-tests the four claims and measures the related behaviour in five open models. It proposes no new method. Following the reproducibility-report format used at TMLR and the ML Reproducibility Challenge [@lico, §2], each claim is stated with its source, tested, and given a verdict. Following measurement papers on LLM serving [@etalon, §3.1; @measbias, §4], we report where a standard computation returns a misleading number. Our contributions:

1. **Reproduction of four claims** on CPU PyTorch, with fine grids, float64 and 50-digit references, and the papers' own published code translated line by line (§4).
2. **Real-model measurements** of sink strength, attention entropy, planted-needle retrieval and log-probability precision in five open models. Among them is a matched softmax/softpick pair released by the softpick authors [@softpickckpt]. Every probe is validated bit for bit against the reference attention implementation (§3, §5).
3. **An audit of how vLLM and SGLang compute returned log-probabilities**, with the one configuration where a bfloat16 failure is expected (§5.3).
4. **Scripts for the GPU experiments** that remain: serving-engine log-probs, sink pinning under window eviction with a pre-registered criterion, and CUDA thresholds (§8).

## 2 Claims checked

**Table 1.** Claims, where they are stated, and our verdicts (details in §4–5).

| # | Claim (paraphrased) | Source | What we test | Verdict |
|---|---|---|---|---|
| C1a | With bounded logits, every global-attention coefficient is Θ(1/n) as items n grow | [@velickovic, Lemma 2.1, Thm. 2.2] | Closed form vs float64; five real models | Pending (2 of 5 models): the fixed-gap law does not describe a single head; other heads take over |
| C1b | Adaptive temperature, set from row entropy by a fitted polynomial, sharpens attention at inference | [@velickovic, §3, Fig. 4, Table 1] | The printed polynomial on synthetic rows | Holds inside the fitted regime; inactive above 5.95 nats |
| C2 | Strong first-token sinks are common; a non-normalised attention removes them | [@barbero, Table 1; @gu, §7; @softpick, Fig. 1] | Sink rate and first-token mass in five models | Holds in the matched pair (softmax 62.6% at 256 tokens vs softpick 0%); 3 models pending |
| C3 | Cross-entropy's target gradient rounds to zero beyond a cliff c ≈ 5.5 (bf16), 16.6 (fp32), 36.7 (fp64); the log-space fix removes it | [@barron, §1–2, Fig. 1a] | Built-in vs fixed, three dtypes, C ∈ {2, 113, 1000, 32000, 151936} | Holds for C = 2; earlier cliff for C ≥ 113 on CPU |
| C4 | Soft-target cross-entropy has a non-zero gradient at z_S = z_T; the fix returns 0 | [@barron, §3, Fig. 1b] | Built-in vs fixed vs float64/mpmath reference | Holds |

## 3 Methodology and metrics

**Synthetic reproductions (C1b, C3, C4).** These were run on CPU PyTorch 2.15.0, so every threshold below is a property of CPU kernels and may differ on GPUs (B6, §8).
- **C3:** C classes, the target logit at gap g above all others, built-in `F.cross_entropy`. We scan g on a 0.01 grid, refined to 0.001 near each transition. We report the paper's margin $m = z_t - \mathrm{LSE}_{j\ne t} z_j = g - \ln(C-1)$.
- **The fixes** are line-by-line translations of the paper's JAX listings [@barron, §2–3].
- **C4:** 1,000 classes. The reference is float64 $q-p$ on the same rounded logits, cross-checked against mpmath at 50 digits (maximum difference ≤ 2.6×10⁻¹⁷).
- **C1b:** the adaptive temperature is applied exactly as printed [@velickovic, Fig. 4].

**Real-model probes (C1a, C2).** Five models run in float32 on CPU (Table 2). We registered an attention function with transformers' attention interface. It computes exact softmax (or softpick) attention in query chunks and records statistics as it runs. Before any probe number was used, we checked it against transformers' eager attention: the logits matched bit for bit for all five models. For the matched pair, we also matched an independent forward pass written from the authors' flash-linear-attention code. That check is sensitive: swapping softpick for softmax in the reference changes the logits by up to 30.9. Perplexity on 1,024 held-out tokens (Frankenstein, Project Gutenberg) is 17–40 for all five models, which confirms the weights loaded correctly.

**Table 2.** Models probed (configs and validation: `data/validate_models.json`).

| Model | Layers × heads | Attention | Context probed | BOS token | Perplexity (1,024 tokens) |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 × 16 (8 KV) | global, QK-norm | 256–8,192 | none | 31.6 |
| Gemma-3-1B-it | 26 × 4 (1 KV) | 5 local (window 512) : 1 global, QK-norm | 256–8,192 | `<bos>` | 39.9 |
| SmolLM2-360M | 32 × 15 (5 KV) | global | 256–8,192 | none | 17.0 |
| vanilla-340M | 24 × 16 | global, softmax | 256–4,096 (trained at 4,096) | `<s>` | 17.3 |
| softpick-340M | 24 × 16 | global, softpick | 256–4,096 (trained at 4,096) | `<s>` | 17.1 |

**Metrics.** Probe statistics are taken over the last 64 query positions of each sequence.
- **Sink rate:** share of heads whose mean weight on token 0 exceeds ε = 0.3, the definition of Gu et al. [@gu, §3]. We also give ε = 0.8 [@barbero, Table 1]. Gu et al. average over all queries; we average over the last 64.
- **First-token mass:** mean weight on token 0 over all heads.
- **Entropy ratio:** mean row entropy on global layers divided by ln(keys visible). It equals 1 for uniform attention. For softpick, entropy is computed on the renormalised row.
- **Needle weight:** the weight the final query puts on the answer tokens of a planted sentence. We report it for the best head at each N, and for a fixed head (the best head at N = 256). For the fixed-gap law $w = e^{\Delta}/(e^{\Delta}+N-1)$ [proven], we report $\Delta_N = \mathrm{logit}(w) + \ln(N-1)$, which the law keeps constant.
- **Log-prob precision:** the share of tokens with reference probability > 0.9 whose returned log p is exactly 0.

Natural text comes from three Project Gutenberg books (3 passages per point). Needles are inserted at depths 0.25, 0.5 and 0.75.

## 4 Reproduction results

### 4.1 C3 — Softmax Collapse

![E4a](figures/e4_collapse.png "Figure 1. Target-logit gradient magnitude versus logit gap, C = 113, CPU PyTorch 2.15.0. Solid: built-in F.cross_entropy; dashed: Barron et al.'s fixed cross-entropy (translated); dash-dot: bf16 logits with an fp32 loss. Exact zeros are drawn on the floor. Takeaway: each dtype has a sharp cliff; the fix removes it except where fp16 exponentials underflow.")

**Table 3.** Precision cliff in the paper's margin units: published value vs first exact zero on CPU PyTorch (`data/e4_collapse.json`, `data/e4_fp64.json`).

| dtype | Published [@barron, §1] | C = 2 | C = 113 | C = 1,000 | C = 32,000 / 151,936 | ln(2/ε) |
|---|---|---|---|---|---|---|
| bfloat16 | ≈ 5.5 | 5.547 | 5.564 | 5.562 | 5.534 / 5.507 | 5.545 |
| float16 | — | 7.624 | 7.622 | 7.621 | 7.588 / 7.593 | 7.625 |
| float32 | ≈ 16.6 | 16.636 | 15.943 | 15.944 | 15.943 / 15.943 | 16.636 |
| float64 | ≈ 36.7 | 36.737 | 34.658 | 34.659 | — / 34.658 | 36.737 |

**Verdict: holds, with a class-count condition.**
- **The published cliffs are the two-class values,** ln(2/ε).
- **With 113 or more classes, CPU PyTorch collapses earlier:** by 0.69 nats in float32, at ln(1/ε), and by 2.08 nats in float64. bf16 and fp16 stay within 0.04 of ln(2/ε).
- **The fix works in fp32 and bf16.** It never returns a zero target gradient on [0, 45], with maximum relative error 3.2×10⁻⁷ (fp32) and 9.7×10⁻³ (bf16).
- **In fp16 the fix still returns zero from gap 17.336, for every C we tested,** because each non-target term $e^{z_j-z_t}$ underflows fp16 at that gap.
- **Realistic mixed precision.** With bf16 logits and an fp32 loss (autocast), the bf16 cliff disappears and the fp32 cliff remains. It appears at grid gap 20.688 because bf16 rounds that gap to 20.75.

### 4.2 C4 — Softmax Cancellation

**Verdict: holds.** At $z_S = z_T$ the built-in gradient norm is 9.1×10⁻⁹ in fp32, 1.22×10⁻³ in pure bf16 and 2.27×10⁻⁴ in pure fp16; the fix returns exactly 0 in all three (`data/e4_cancellation.json`).
- **The bf16 floor comes from low-precision storage.** Storing the student's log-softmax in bf16 contributes 1.08×10⁻³; storing the teacher's probabilities in bf16 contributes 3.29×10⁻⁴.
- **The fp32 floor comes from PyTorch's own gradient path.** ‖softmax − exp(log_softmax)‖ is 5.3×10⁻⁹ in fp32.
- **Mixed precision lowers the bf16 floor to 2.2×10⁻⁸.** That is teacher probabilities in fp32 and the student upcast before the loss.

### 4.3 C1b — Adaptive temperature

![E3](figures/e3_random_logits.png "Figure 2. Gaussian-logit rows (20–400 rows per point). (a) Mean entropy vs N. (b) Max weight with and without the paper's adaptive θ. (c) The paper's polynomial β(H), clipped at 1. Takeaway: the polynomial sharpens only rows with entropy between 0.85 and 5.95 nats.")

**Verdict: holds inside the regime it was fitted for; inactive beyond it.**
- **The printed polynomial gives β > 1 only for row entropy H ∈ [0.849, 5.945] nats** (peak β = 2.42 at H = 4.41).
- **Inside that window it sharpens a planted target strongly.** A target 4σ above Gaussian distractors rises from 0.125 to 0.798 of the row at N = 235.
- **From N = 861 the row entropy leaves the window and the correction switches off** (β = 1). With a 6σ target it switches off from N = 1,328.
- **This is consistent with the paper's own description:** an "ad-hoc" fit derived on its max-retrieval task [@velickovic, §3, §5]. A uniform row over 4,096 keys already has ln 4096 ≈ 8.3 nats, so the polynomial would have to be refitted before use in long-context heads.

## 5 Real-model measurements

### 5.1 C2 — Attention sinks

![E6a](figures/e6_sinks.png "Figure 3. Sink rate (share of heads with mean weight on token 0 above 0.3) and mean first-token mass vs context length, last 64 queries, 3 Gutenberg passages per point, CPU fp32. Draft: vanilla-340M and softpick-340M only. Takeaway: the softmax model's sink weakens as context grows; softpick has none at any length.")

**Verdict (matched pair): holds.**
- **The softmax model has strong sinks.** At 256 tokens, 62.6% of vanilla-340M's heads put more than 0.3 of their weight on token 0. That is close to the 63.41% the softpick authors report for their 340M softmax model [@softpick, Fig. 1]. Their averaging may differ from ours, which uses the last 64 queries.
- **The sink weakens as context grows.** The sink rate falls to 60.2%, 52.6%, 36.9% and 10.6% at 512, 1,024, 2,048 and 4,096 tokens. Mean first-token mass falls from 0.382 to 0.088. Gu et al. also report their sink metric decreasing with sequence length [@gu, Fig. 3].
- **Strong sinks (ε = 0.8) are rare.** They cover 0.6% of heads at 256 tokens and none beyond, below the 45.97% reported for Llama 3.1 8B [@barbero, Table 1]. These are much smaller models, so we do not read this as a contradiction.
- **Softpick has no sinks.** Its sink rate is 0.0% at every length, and its first-token mass is at most 0.001 [@softpick, Fig. 1].
- **Pending:** SmolLM2-360M (sink rate 0.733 at 256 tokens and 0.424 at 4,096 in the running log), Qwen3-0.6B and Gemma-3-1B-it.

### 5.2 C1a — Dispersion in real heads

![E6c](figures/e6_needle.png "Figure 4. Final-query weight on the planted answer tokens vs context length (3 depths per point). (a) The head that is best at N = 256, tracked across N; dotted: the fixed-gap law with Δ fitted at N = 256. (b) The best head at each N. Draft: matched 340M pair only. Takeaway: a fixed head disperses faster than the law predicts in the softmax model, but other heads take over.")

**Verdict (matched pair): dispersion holds for a fixed head; retrieval survives through head switching.**
- **The fixed-gap law fails for the softmax model.** Under the law, the fitted gap $\Delta_N$ of a fixed head would stay constant. In vanilla-340M the best head at 256 tokens (layer 21, head 13) falls from weight 0.782 to 0.0019 at 4,096 tokens, and $\Delta_N$ falls from 6.82 to 2.03. That is faster dispersion than the law predicts.
- **Other heads take over.** The best head at each N keeps weight 0.69–0.78, its $\Delta_N$ rises from 6.82 to 9.51, and the model's top-1 prediction is the answer at all 15 needle positions.
- **Softpick's fixed head holds longer.** Layer 20, head 11 keeps 0.69, 0.63, 0.51 and 0.50 up to 2,048 tokens ($\Delta_N$ 6.34 → 7.62), then falls to 0.074 at 4,096. At 4,096 the top-1 answer rate drops to 2 of 3. Softpick weights need not sum to one, so its needle weights are not directly comparable to softmax weights.
- **Attention entropy is not reported in this draft.** The softpick entropies in the current data were computed before rows were renormalised. A corrected re-run is queued.

### 5.3 Log-probability precision and serving engines

**Serving-engine code (read at pinned commits).**
- **vLLM** returns `logits.log_softmax(dim=-1, dtype=torch.float32)` [@vllm, sampler.py].
- **SGLang** casts logits to float32. Greedy requests use `log_softmax`, and standard sampling returns `torch.log(probs)` [@sglang, logits_processor.py and sampler.py].
- **The exception:** with `--rl-on-policy-target` (and deterministic inference, temperature sampling without top-k/top-p), SGLang casts logits to bfloat16 and computes `log_softmax` in bfloat16 [@sglang, sampler.py and server_arguments.mdx].

**Synthetic precision thresholds** (from E4c, `data/e4_forward_logprob.json`; Qwen's 151,936-token vocabulary):
- bfloat16 `log_softmax` returns exactly 0 once the target leads by a gap of 17.44, i.e. paper margin 5.51, which corresponds to p ≳ 0.996.
- fp32 `log_softmax` and fp32 `log(softmax)` do so only at a gap of 27.88 (margin 15.95).

**Pending:** E7 on real text (exact-zero rate among tokens with p > 0.9, per method), and B4 (returned log-probs from running servers, GPU).

## 6 Implications for serving

Each item is labelled *established* (measured here or shown in a cited source) or *inference*.

**Table 4.** Places where a standard computation returns a misleading value.

| Computation | What goes wrong | Where | Status |
|---|---|---|---|
| Collapse cliff quoted as 16.6 (fp32) | With ≥113 classes, CPU PyTorch collapses at margin 15.94 | Training losses | established (Table 3) |
| Adaptive temperature from [@velickovic, Fig. 4] | Inactive above 5.95 nats of row entropy | Long-context heads | established (§4.3) |
| bf16 `log_softmax` of returned log-probs | log p = 0 for p ≳ 0.996 | SGLang RL on-policy mode | established in code and on synthetic logits; real-model rate pending |
| "Keep the first page" eviction heuristics | Sink mass falls with context in vanilla-340M; absent in softpick | Paged KV eviction | inference from the matched pair; test in B5 |

- **KV eviction.** Window-only eviction breaks unless the first tokens are kept [@streamingllm, §3] (established). In the matched pair, the sink's mass shrinks as context grows and is absent with softpick. So the value of pinning the first page may depend on context length and on the attention function (inference; B5 tests it).
- **Long-context retrieval.** One retrieval head does disperse with N, but the model compensates by switching heads. A probe that tracks a single head would overstate the loss of retrieval ability (established in the pair).
- **Log-prob APIs.** Default vLLM and SGLang paths are float32. Users of SGLang's RL on-policy mode should expect exact zeros for confident tokens (established in code; the size of the effect on real text is pending).

## 7 Threats to validity

- **Hardware and kernels.** All thresholds are CPU PyTorch 2.15.0 results. The B6 script checks CUDA.
- **Scale.** The real models are 340M–1B parameters, probed at ≤8,192 tokens. Sink behaviour in 8B–405B models comes from the cited papers.
- **Sample size.** Three passages and three needle depths per context length. No confidence intervals in this draft.
- **Definitions.** Our sink rate averages over the last 64 queries; Gu et al. and the softpick authors average differently.
- **Implementation.** The fixes are our translations of published JAX listings. The 340M models are loaded through a mapping onto Llama, validated against an independent forward pass.

## 8 Artifact availability

Code, raw JSON, stdout logs and validation results are published with this page and in `reports/softmax_paper_v3/` of the author's repository. Validation: `data/validate_models.json`. GPU scripts with a pre-registered criterion for B5: `code/gpu/`. The companion technical report holds the survey, the census of 36 model families and the E1–E5 experiments [@v2report].

## References

{{REFERENCES}}
