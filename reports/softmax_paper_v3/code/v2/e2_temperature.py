"""E2: logit gap / temperature needed to hold the target weight >= alpha, and how far the
logit multipliers that real families ship close the gap.

Toy model: one target at Delta, N-1 distractors at 0, temperature T.
  w >= alpha  <=>  Delta/T >= ln(N-1) + ln(alpha/(1-alpha))          (closed form)
Verified by bisection on the explicit softmax for N <= 65,536.

Family multipliers c(N) on the attention logits (sources in the report, Part B):
  * none: RoPE-frequency changes only (Llama 3.1 'llama3' rope, Gemma 3 linear x8) -> c = 1
  * static YaRN, factor s: logits x (0.1 ln s + 1)^2 (YaRN eq. 15; HF rope utils put the factor
    on cos and sin, i.e. on q and k; DeepSeek code puts mscale^2 on the softmax scale)
      - Qwen3 model card, inference-time, s = 4        -> 1.2964
      - gpt-oss / Kimi K2 config, s = 32               -> 1.8133
      - DeepSeek-V3 config, s = 40, mscale_all_dim = 1 -> 1.8739
  * Llama 4 attention temperature tuning (NoPE layers only, reference code):
      c(pos) = 1 + 0.1 * ln(floor((pos + 1) / 8192) + 1)
  * log-length scaling (Scalable-Softmax form s * ln N, normalised to 1 at N = L): c = ln N / ln L
All curves are calibrated so that the head exactly holds alpha = 0.9 at L = 4096 with c = 1.
This is a toy best case: one target, identical distractors, and a multiplier applied to a head
whose learned gap is unchanged. Deterministic."""
import numpy as np
from style import plt, C, INK2, save, dump

Ns = np.unique(np.round(np.logspace(np.log10(2), np.log10(65536), 60)).astype(int))
alphas = [0.5, 0.9, 0.99]
L = 4096
ALPHA = 0.9


def req(N, a):
    return np.log(N - 1) + np.log(a / (1 - a))


def target_weight(x, N):
    z = np.zeros(N)
    z[0] = x
    z -= z.max()
    e = np.exp(z)
    return e[0] / e.sum()


def bisect(alpha, N, lo=-50.0, hi=100.0):
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if target_weight(mid, N) >= alpha:
            hi = mid
        else:
            lo = mid
    return hi


def yarn(s):
    return (0.1 * np.log(s) + 1.0) ** 2


schemes = {
    "none (RoPE rescaling only)": lambda N: np.ones_like(N, dtype=float),
    "static YaRN s=4 (Qwen3 card)": lambda N: np.full_like(N, yarn(4), dtype=float),
    "YaRN s=32 (gpt-oss, Kimi K2)": lambda N: np.full_like(N, yarn(32), dtype=float),
    "YaRN s=40 (DeepSeek-V3)": lambda N: np.full_like(N, yarn(40), dtype=float),
    "Llama 4 temp. tuning (NoPE)": lambda N: 1 + 0.1 * np.log(np.floor((N + 1) / 8192) + 1),
    "log-length ln N / ln L": lambda N: np.log(N) / np.log(L),
}

out = {"N": Ns.tolist(), "L": L, "alpha_calib": ALPHA, "series": {}, "schemes": {}}
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
ax = axes[0]
for i, a in enumerate(alphas):
    closed = req(Ns, a)
    num = np.array([bisect(a, N) for N in Ns])
    err = float(np.max(np.abs(num - closed)))
    out["series"][str(a)] = {"closed": closed.tolist(), "bisect": num.tolist(), "max_abs_err": err,
                             "at_65536": float(closed[-1])}
    ax.plot(Ns, closed, color=C[i], label=f"α = {a}")
    ax.plot(Ns[::4], num[::4], "o", ms=4.5, mfc="white", mec=C[i], mew=1.4)
    print(f"alpha={a}: max |bisect-closed| = {err:.2e}; required Delta/T at N=65536: {closed[-1]:.3f}")
ax.plot([], [], "o", ms=4.5, mfc="white", mec=INK2, label="bisection, explicit softmax")
ax.set_xscale("log", base=2)
ax.set_xlabel("number of keys N (log scale)")
ax.set_ylabel("required Δ/T (logit units)")
ax.set_title("(a) Required Δ/T for target weight ≥ α")
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
NN = np.unique(np.round(np.logspace(np.log10(L), np.log10(10_485_760), 400)).astype(np.int64)).astype(float)
need = req(NN, ALPHA) / req(L, ALPHA)
ax.plot(NN, need, color="black", lw=2.4, label="required multiplier (α = 0.9)")
D0 = req(L, ALPHA)
probes = [32768, 131072, 1_048_576, 10_485_760]
out["required_multiplier"] = {str(n): float(req(n, ALPHA) / D0) for n in probes}
for i, (name, f) in enumerate(schemes.items()):
    c = f(NN)
    ax.plot(NN, c, color=C[i % len(C)], lw=1.6, ls="-" if i else ":", label=name)
    rows = {}
    for n in probes:
        cn = float(f(np.array([float(n)]))[0])
        nd = float(req(n, ALPHA) / D0)
        w = 1.0 / (1.0 + (n - 1) * np.exp(-cn * D0))
        rows[str(n)] = {"c": cn, "need": nd, "gap_closed": (cn - 1) / (nd - 1), "weight": float(w)}
    out["schemes"][name] = rows
    print(name + ": " + "; ".join(f"N={n}: c={r['c']:.3f} need={r['need']:.3f} closed={r['gap_closed']:.2f} w={r['weight']:.3f}"
                                  for n, r in rows.items()))
ax.set_xscale("log", base=2)
ax.set_xlabel("context length N (tokens, log scale)")
ax.set_ylabel("logit multiplier relative to N = 4,096 (×)")
ax.set_title("(b) Shipped multipliers vs the multiplier needed")
ax.legend(loc="upper left", fontsize=7.2)
fig.suptitle("E2. Holding a selection as context grows (toy model; head calibrated to α = 0.9 at N = 4,096)",
             y=1.02, fontsize=11)
save(fig, "e2_temperature")
dump(out, "e2_temperature")
