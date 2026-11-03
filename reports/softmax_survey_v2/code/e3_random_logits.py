"""E3: softmax over i.i.d. Gaussian logits, and the adaptive temperature of Velickovic et al.

(a) Rows z ~ N(0, sigma^2), sigma in {1, 2, 4}; entropy (nats) and max weight vs N (float64).
    Large-N asymptote of the entropy: ln N - sigma^2 / 2.
(b) The paper's adaptive temperature, exactly as printed in its Figure 4 (JAX code):
        H = entropy(softmax(z));  beta = max(polyval([-0.037, 0.481, -2.3, 4.917, -1.791], H), 1)
        if H > 0.5 else 1;  output softmax(beta * z)
    applied to (i) the Gaussian rows and (ii) rows with one planted target at z = sigma*k
    above Gaussian distractors (sigma = 1, k in {4, 6}).
Seed 0. Trials per point: max(20, min(400, 2e6 // N))."""
import numpy as np
from style import plt, C, INK2, save, dump

rng = np.random.default_rng(0)
Ns = np.unique(np.round(np.logspace(np.log10(2), np.log10(65536), 25)).astype(int))
sigmas = [1.0, 2.0, 4.0]
POLY = np.array([-0.037, 0.481, -2.3, 4.917, -1.791])


def softmax(z, beta=1.0):
    z = beta * z
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p):
    return -np.sum(p * np.log(p + 1e-9), axis=-1)  # same 1e-9 as the paper's code


def adaptive(z):
    p = softmax(z)
    H = entropy(p)
    beta = np.where(H > 0.5, np.maximum(np.polyval(POLY, H), 1.0), 1.0)
    return softmax(z, beta[..., None]), beta, H


# where does the polynomial sharpen at all?
Hgrid = np.linspace(0, 12, 120001)
bg = np.maximum(np.polyval(POLY, Hgrid), 1.0) * (Hgrid > 0.5) + 1.0 * (Hgrid <= 0.5)
sharp = Hgrid[bg > 1.0]
poly_info = {"H_min_sharpen": float(sharp.min()), "H_max_sharpen": float(sharp.max()),
             "beta_max": float(bg.max()), "H_at_beta_max": float(Hgrid[bg.argmax()])}
print("adaptive-theta polynomial sharpens (beta>1) only for H in [%.3f, %.3f] nats; max beta %.3f at H=%.3f"
      % (poly_info["H_min_sharpen"], poly_info["H_max_sharpen"], poly_info["beta_max"], poly_info["H_at_beta_max"]))

out = {"N": Ns.tolist(), "poly": POLY.tolist(), "poly_info": poly_info, "gauss": {}, "planted": {}}
for s in sigmas:
    H_mean, M, M_ad, B_ad, T_used = [], [], [], [], []
    for N in Ns:
        T = int(max(20, min(400, 2_000_000 // N)))
        z = rng.normal(0, s, (T, N))
        p = softmax(z)
        pa, beta, _ = adaptive(z)
        H_mean.append(float(entropy(p).mean()))
        M.append(float(p.max(-1).mean()))
        M_ad.append(float(pa.max(-1).mean()))
        B_ad.append(float(beta.mean()))
        T_used.append(T)
    out["gauss"][str(s)] = dict(H_mean=H_mean, max_w=M, max_w_adaptive=M_ad, beta_mean=B_ad, trials=T_used,
                                asymptote=(np.log(Ns) - s ** 2 / 2).tolist())
    i64 = int(np.argmin(abs(Ns - 64)))
    print(f"sigma={s}: H(65536)={H_mean[-1]:.3f} vs lnN-s^2/2={np.log(Ns[-1]) - s**2/2:.3f}; "
          f"max_w N=64 {M[i64]:.3f} -> N=65536 {M[-1]:.4f}; adaptive max_w(65536)={M_ad[-1]:.4f} "
          f"(mean beta {B_ad[-1]:.3f}); trials at 65536 = {T_used[-1]}")

for k in [4.0, 6.0]:
    W, W_ad, B = [], [], []
    for N in Ns:
        T = int(max(20, min(400, 2_000_000 // N)))
        z = rng.normal(0, 1.0, (T, N))
        z[:, 0] = k
        p = softmax(z)
        pa, beta, _ = adaptive(z)
        W.append(float(p[:, 0].mean()))
        W_ad.append(float(pa[:, 0].mean()))
        B.append(float(beta.mean()))
    out["planted"][str(k)] = dict(w=W, w_adaptive=W_ad, beta_mean=B)
    i2 = int(np.argmin(abs(Ns - 256)))
    off = [int(n) for n, b in zip(Ns, B) if b == 1.0 and n > 64]
    print(f"planted target at {k} sigma: w(N={Ns[i2]})={W[i2]:.3f}->adaptive {W_ad[i2]:.3f}; adaptive off (beta=1) from N={off[0] if off else None}; "
          f"w(65536)={W[-1]:.4f}->adaptive {W_ad[-1]:.4f} (mean beta {B[-1]:.3f})")

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
ax = axes[0]
ax.plot(Ns, np.log(Ns), color=INK2, lw=1, ls=":", label="ln N (uniform)")
for i, s in enumerate(sigmas):
    d = out["gauss"][str(s)]
    ax.plot(Ns, d["H_mean"], color=C[i], label=f"σ = {s:g}")
    ax.plot(Ns, d["asymptote"], color=C[i], lw=1, ls="--")
ax.plot([], [], color=INK2, lw=1, ls="--", label="ln N − σ²/2")
ax.set_xscale("log", base=2); ax.set_ylim(bottom=0)
ax.set_xlabel("number of keys N (log scale)"); ax.set_ylabel("row entropy (nats)")
ax.set_title("(a) Entropy of Gaussian-logit rows"); ax.legend(fontsize=7.5, loc="upper left")
ax = axes[1]
for i, s in enumerate(sigmas):
    d = out["gauss"][str(s)]
    ax.plot(Ns, d["max_w"], color=C[i], label=f"σ = {s:g}, T = 1")
    ax.plot(Ns, d["max_w_adaptive"], color=C[i], ls="--", label=f"σ = {s:g}, adaptive θ")
ax.set_xscale("log", base=2); ax.set_ylim(0, 1)
ax.set_xlabel("number of keys N (log scale)"); ax.set_ylabel("max weight in row (0–1)")
ax.set_title("(b) Max weight, with the paper's adaptive θ"); ax.legend(fontsize=7, loc="upper right")
ax = axes[2]
ax.plot(Hgrid, bg, color=C[0])
ax.axhline(1, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("row entropy H (nats)"); ax.set_ylabel("inverse temperature β (×)")
ax.set_title("(c) Paper's polynomial β(H), clipped at 1")
fig.suptitle("E3. Gaussian logits and adaptive temperature (Veličković et al., Fig. 4 polynomial)", y=1.02, fontsize=11)
save(fig, "e3_random_logits")
dump(out, "e3_random_logits")
