"""E1 (illustration, not a finding): dispersion of a fixed logit gap.

One target logit Delta, N-1 distractors at 0, temperature 1.
Closed form: w(N) = e^D / (e^D + N - 1).  Numerical: float64 softmax of the explicit vector.
Deterministic (no randomness)."""
import numpy as np
from style import plt, C, INK2, save, dump

Ns = np.unique(np.round(np.logspace(np.log10(2), np.log10(65536), 60)).astype(int))
deltas = [2, 4, 8, 16]
probe_N = [2048, 4096, 8192, 32768, 65536]


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def w_closed(D, N):
    return np.exp(D) / (np.exp(D) + N - 1)


out = {"N": Ns.tolist(), "series": {}, "probe": {}}
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for i, D in enumerate(deltas):
    analytic = w_closed(D, Ns)
    numeric = []
    for N in Ns:
        z = np.zeros(N)
        z[0] = D
        numeric.append(softmax(z)[0])
    numeric = np.array(numeric)
    err = float(np.max(np.abs(numeric - analytic) / analytic))
    probe = {str(N): float(w_closed(D, N)) for N in probe_N}
    out["series"][str(D)] = {"analytic": analytic.tolist(), "numeric": numeric.tolist(),
                             "max_rel_err": err, "N_half": float(np.exp(D) + 1)}
    out["probe"][str(D)] = probe
    ax.plot(Ns, analytic, color=C[i], label=f"Δ = {D}")
    ax.plot(Ns[::4], numeric[::4], "o", ms=4.5, mfc="white", mec=C[i], mew=1.4)
    print(f"Delta={D}: max rel err = {err:.2e}; w=0.5 at N={np.exp(D)+1:.1f}; "
          + "; ".join(f"w({N})={v:.4f}" for N, v in probe.items()))
ax.plot([], [], "o", ms=4.5, mfc="white", mec=INK2, label="float64 softmax (every 4th N)")
ax.set_xscale("log", base=2)
ax.set_xlabel("number of keys N (log scale)")
ax.set_ylabel("weight on target (dimensionless, 0–1)")
ax.set_title("E1. Fixed logit gap Δ: target weight vs N (T = 1)")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="lower left", fontsize=8.5)
save(fig, "e1_dispersion")
dump(out, "e1_dispersion")
