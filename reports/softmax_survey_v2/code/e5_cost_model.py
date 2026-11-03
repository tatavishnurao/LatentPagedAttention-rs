"""E5: analytic cost model (a MODEL, not a measurement) for four released configs.

Counts per generated token (decode) and per prompt (prefill), batch 1, bf16 KV cache (2 bytes).
FLOPs: 2 per multiply-accumulate. 'Linear' = all weight matmuls on the token's path (attention
projections, MLP or active experts, router, LM head). 'Attention' = score (QK) and value (PV) matmuls
only. exp() count = number of softmax score elements. Norms, RoPE, activations, conv and gating
nonlinearities are ignored.

Configs (official config.json, see report Part B):
  Llama-3.1-8B            GQA dense, 32 layers, 32 q / 8 kv heads, d=128
  DeepSeek-V3             MLA (kv_lora 512 + rope 64), 61 layers, 128 heads, MoE 8 of 256 + 1 shared;
                          decode counted with the weight-absorbed MLA form (scores over 576 dims,
                          values over 512), prefill with the non-absorbed form (192 / 128 dims)
  Gemma-3-27B             62 layers, 5 local (window 1024) : 1 global, 32 q / 16 kv heads, d=128;
                          the 27B config does not list vocab_size, the 1B config's 262,144 is used
  Qwen3-Next-80B-A3B      48 layers, every 4th is gated softmax attention (16 q / 2 kv, d=256),
                          the other 36 are Gated DeltaNet (16 key heads x128, 32 value heads x128)
                          with a fixed-size state; MoE 10 of 512 + 1 shared
"""
import numpy as np
from style import plt, C, INK2, save, dump

BYTES = 2


def llama31_8b():
    L, H, KV, d, dm, I, V = 32, 32, 8, 128, 4096, 14336, 128256
    lin = L * (dm * H * d + 2 * dm * KV * d + H * d * dm + 3 * dm * I) + dm * V
    layers = [dict(H=H, dqk=d, dv=d, W=None, kv_elems=2 * KV * d)] * L
    return dict(name="Llama-3.1-8B (GQA)", lin_flops=2 * lin, layers=layers, pre_dims=None, fixed_state_bytes=0)


def deepseek_v3():
    L, H, dm, ql, kvl, rope, nope, vd, V = 61, 128, 7168, 1536, 512, 64, 128, 128, 129280
    attn = dm * ql + ql * H * (nope + rope) + dm * (kvl + rope) + H * nope * kvl + H * kvl * vd + H * vd * dm
    dense_mlp, moe_mlp = 3 * dm * 18432, (8 + 1) * 3 * dm * 2048 + dm * 256
    lin = L * attn + 3 * dense_mlp + (L - 3) * moe_mlp + dm * V
    # decode (absorbed): scores over kv_lora+rope dims, values over kv_lora dims; cache = latent + rope key
    layers = [dict(H=H, dqk=kvl + rope, dv=kvl, W=None, kv_elems=kvl + rope)] * L
    return dict(name="DeepSeek-V3 (MLA)", lin_flops=2 * lin, layers=layers,
                pre_dims=dict(dqk=nope + rope, dv=vd), fixed_state_bytes=0)


def gemma3_27b():
    L, H, KV, d, dm, I, V, W = 62, 32, 16, 128, 5376, 21504, 262144, 1024
    lin = L * (dm * H * d + 2 * dm * KV * d + H * d * dm + 3 * dm * I) + dm * V
    layers = [dict(H=H, dqk=d, dv=d, W=(None if (i + 1) % 6 == 0 else W), kv_elems=2 * KV * d) for i in range(L)]
    return dict(name="Gemma-3-27B (5 local : 1 global)", lin_flops=2 * lin, layers=layers, pre_dims=None, fixed_state_bytes=0)


def qwen3_next():
    L, dm, V = 48, 2048, 151936
    H, KV, d = 16, 2, 256
    nk, dk, nv, dv = 16, 128, 32, 128
    attn_lin = dm * H * d * 2 + 2 * dm * KV * d + H * d * dm            # q_proj doubled for the output gate
    dn_lin = dm * (2 * nk * dk + 2 * nv * dv) + dm * 2 * nv + nv * dv * dm  # in_proj_qkvz, in_proj_ba, out_proj
    moe = (10 + 1) * 3 * dm * 512 + dm * 512 + dm
    n_attn = L // 4
    lin = n_attn * attn_lin + (L - n_attn) * dn_lin + L * moe + dm * V
    dn_state_flops = (L - n_attn) * nv * 2 * (3 * dk * dv)  # read, rank-1 update, output: ~3 dk*dv MACs/head
    layers = [dict(H=H, dqk=d, dv=d, W=None, kv_elems=2 * KV * d) for _ in range(n_attn)]
    return dict(name="Qwen3-Next-80B-A3B (3 DeltaNet : 1 attn)", lin_flops=2 * lin + dn_state_flops, layers=layers,
                pre_dims=None, fixed_state_bytes=(L - n_attn) * nv * dk * dv * 4)  # fp32 recurrent state


def span(n, W):
    return np.minimum(n, W) if W else n


def costs(m, n):
    n = np.asarray(n, float)
    dec_exp = sum(l["H"] * span(n, l["W"]) for l in m["layers"])
    dec_attn = sum(2 * l["H"] * (l["dqk"] + l["dv"]) * span(n, l["W"]) for l in m["layers"])
    kv_cache = sum(l["kv_elems"] * span(n, l["W"]) for l in m["layers"]) * BYTES
    pre_exp, pre_attn = 0.0, 0.0
    for l in m["layers"]:
        dqk = m["pre_dims"]["dqk"] if m["pre_dims"] else l["dqk"]
        dv = m["pre_dims"]["dv"] if m["pre_dims"] else l["dv"]
        if l["W"]:
            W = l["W"]
            s = np.where(n <= W, n * (n + 1) / 2, W * (W + 1) / 2 + (n - W) * W)
        else:
            s = n * (n + 1) / 2
        pre_exp = pre_exp + l["H"] * s
        pre_attn = pre_attn + 2 * l["H"] * (dqk + dv) * s
    return dict(dec_exp=dec_exp, dec_attn=dec_attn, kv_bytes=kv_cache + m["fixed_state_bytes"],
                kv_bytes_per_token=sum(l["kv_elems"] for l in m["layers"] if not l["W"]) * BYTES,
                pre_exp=pre_exp, pre_attn=pre_attn, lin=m["lin_flops"])


def crossover(m, f):
    lo, hi = 1.0, 1e12
    if f(m, hi) < m["lin_flops"]:
        return None
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(m, mid) < m["lin_flops"] else (lo, mid)
    return float(hi)


models = [llama31_8b(), deepseek_v3(), gemma3_27b(), qwen3_next()]
n = np.unique(np.round(np.logspace(np.log10(256), np.log10(1_048_576), 120))).astype(float)
out = {"n": n.tolist(), "models": {}}
for m in models:
    c = costs(m, n)
    x_dec = crossover(m, lambda mm, nn: costs(mm, nn)["dec_attn"])
    x_pre = crossover(m, lambda mm, nn: costs(mm, nn)["pre_attn"] / nn)
    at = {k: float(np.asarray(v)) if np.ndim(v) == 0 else None for k, v in costs(m, 32768.0).items()}
    out["models"][m["name"]] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    out["models"][m["name"]].update(dict(crossover_decode_n=x_dec, crossover_prefill_amortised_n=x_pre, at_32768=at))
    print(f"{m['name']}: linear {m['lin_flops']/1e9:.2f} GFLOP/token; growing KV bytes per context token "
          f"{at['kv_bytes_per_token']:.0f} B; at n=32,768: KV+state {at['kv_bytes']/2**30:.3f} GiB, "
          f"decode exp {at['dec_exp']:.3e}, decode attn {at['dec_attn']/1e9:.2f} GFLOP (attn/linear {at['dec_attn']/m['lin_flops']:.2f}); "
          f"decode crossover n={x_dec if x_dec is None else round(x_dec)}; prefill (amortised) crossover n={x_pre if x_pre is None else round(x_pre)}")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
for i, m in enumerate(models):
    d = out["models"][m["name"]]
    axes[0].plot(n, d["dec_exp"], color=C[i], label=m["name"])
    axes[1].plot(n, np.array(d["dec_attn"]) / m["lin_flops"], color=C[i], label=m["name"])
    axes[2].plot(n, np.array(d["kv_bytes"]) / 2**30, color=C[i], label=m["name"])
    if d["crossover_decode_n"] and d["crossover_decode_n"] < n[-1]:
        axes[1].plot([d["crossover_decode_n"]], [1.0], "o", color=C[i], ms=6)
axes[1].axhline(1, color=INK2, lw=1, ls=":")
axes[0].set_ylabel("exp() evaluations per decode token")
axes[1].set_ylabel("decode attention FLOPs / linear FLOPs (×)")
axes[2].set_ylabel("KV cache + recurrent state (GiB, bf16 KV)")
axes[0].set_title("(a) Softmax exp() work per decode token")
axes[1].set_title("(b) Decode: attention vs weight matmuls")
axes[2].set_title("(c) Cache size for one sequence")
for ax in axes:
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length n (tokens, log scale)")
axes[2].legend(fontsize=7.5, loc="upper left")
fig.suptitle("E5. Analytic cost model for four released configs (batch 1; a model, not a measurement)", y=1.02, fontsize=11)
save(fig, "e5_cost_model")
dump(out, "e5_cost_model")
