"""E4: Softmax Collapse and Softmax Cancellation in float32 / bfloat16 / float16 (CPU PyTorch).

All thresholds below are properties of PyTorch's CPU kernels in this environment. GPU kernels,
other PyTorch versions and other frameworks can give different numbers.

E4a  Collapse (hard-label cross-entropy).
     C classes, target logit = gap g, all other logits = 0. The paper's margin is
     m = z_t - LSE_{j!=t} z_j = g - ln(C-1).  Exact target gradient: -(C-1)e^-g / (1 + (C-1)e^-g).
     Grid: g in [0, 45] with step 0.01, then step 0.001 within +-0.25 of every change found.
     Setups:  pure   - logits, loss and gradient in the working dtype
              upcast - logits stored in the dtype, .float() before F.cross_entropy (fp32 loss)
              autocast - logits in the dtype, F.cross_entropy under torch.autocast('cpu', bfloat16)
              fixed  - cross_entropy from Barron et al. (Sec. 2 JAX listing), translated to torch,
                       autodiff in the working dtype, no upcasting
     Predictions printed for comparison: g* = ln(C-1) + ln(1/eps) and ln(C-1) + ln(2/eps).
E4b  Cancellation (soft-target cross-entropy, distillation).
     C = 1000, teacher logits z_T ~ N(0, 3^2) rounded to the dtype; student z_S = round(z_T + eps*xi).
     Setups: all-dtype (teacher probs, loss, grad in dtype), teacher-fp32 (teacher probs in fp32,
     student logits upcast to fp32 for the loss), autocast (bf16 only), fixed (Barron et al. Sec. 3
     JAX listing, translated, in dtype).
     Reference: float64 naive p - q on the same rounded logits; cross-checked against mpmath (50 digits).
E4c  Forward log-probabilities: first gap where log p_target is exactly 0, for F.log_softmax in each
     dtype and for torch.log(torch.softmax(.)) in fp32 (the form SGLang's standard sampling path uses),
     at C = 113 and C = 151,936 (Qwen vocabulary size).
Seed 0."""
import numpy as np
import torch
import torch.nn.functional as F
import mpmath as mp
from style import plt, C as COL, INK2, save, dump

torch.manual_seed(0)
torch.set_num_threads(4)
DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# ---------------- Barron, Gnanapragasam, Tschannen: fixed functions (JAX -> torch) ----------------
def fixed_cross_entropy(z, t):
    """z: (B, K), t: (B,) long. Translation of `cross_entropy` in the paper's Section 2 listing."""
    z_t = torch.gather(z, -1, t[:, None])
    is_t = F.one_hot(t, z.shape[-1]).bool()
    shifted = torch.where(is_t, torch.zeros((), dtype=z.dtype), z - z_t)
    b = shifted.max(-1, keepdim=True).values
    s = torch.where(is_t, torch.zeros((), dtype=z.dtype), torch.exp(shifted - b)).sum(-1)
    return b[..., 0] + torch.log1p(s + torch.expm1(-b[..., 0]))


def fixed_log_softmax(z):
    i_max = z.argmax(-1)
    z_max = torch.gather(z, -1, i_max[:, None])
    is_max = F.one_hot(i_max, z.shape[-1]).bool()
    shifted = torch.where(is_max, torch.zeros((), dtype=z.dtype), z - z_max)
    exp_others = torch.where(is_max, torch.zeros((), dtype=z.dtype), torch.exp(shifted))
    s = exp_others.sum(-1, keepdim=True)
    return shifted - torch.log1p(s)


def fixed_soft_grad(z_S, z_T):
    """Gradient g of soft-target CE wrt z_S, following the paper's Section 3 listing."""
    log_q = fixed_log_softmax(z_S)
    q, p = torch.exp(log_q), torch.exp(fixed_log_softmax(z_T))
    delta = z_S - z_T
    d = delta - (p * delta).sum(-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))
    u = torch.where(s.abs() < 1, p * torch.expm1(s), q - p)
    return u - q * u.sum(-1, keepdim=True)


# ---------------- E4a: collapse ----------------
def grads_target(gaps, Ccls, dt, setup):
    """returns (g_t, sum of grads) in float64 for each gap."""
    B = len(gaps)
    z64 = torch.zeros(B, Ccls, dtype=torch.float64)
    z64[:, 0] = torch.as_tensor(gaps, dtype=torch.float64)
    x = z64.to(dt).clone().requires_grad_(True)
    t = torch.zeros(B, dtype=torch.long)
    if setup == "pure":
        loss = F.cross_entropy(x, t, reduction="sum")
    elif setup == "upcast":
        loss = F.cross_entropy(x.float(), t, reduction="sum")
    elif setup == "autocast":
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = F.cross_entropy(x, t, reduction="sum")
    elif setup == "fixed":
        loss = fixed_cross_entropy(x, t).sum()
    loss.backward()
    g = x.grad.to(torch.float64)
    return g[:, 0].numpy(), g.sum(-1).numpy(), str(loss.dtype)


def first_zero(Ccls, dt, setup, lo=0.0, hi=45.0):
    """Coarse scan (0.01) + fine scan (0.001) around each transition. Returns first gap with g_t == 0,
    last gap with g_t != 0, and whether zeros are contiguous after the first zero."""
    coarse = np.round(np.arange(lo, hi + 1e-9, 0.01), 6)
    chunk = max(1, int(4e6 // Ccls))
    gt = np.concatenate([grads_target(coarse[i:i + chunk], Ccls, dt, setup)[0] for i in range(0, len(coarse), chunk)])
    z = gt == 0
    if not z.any():
        return None, None, None, coarse, gt
    i0 = int(np.argmax(z))
    fine = np.round(np.arange(max(lo, coarse[i0] - 0.25), coarse[i0] + 0.25 + 1e-9, 0.001), 6)
    gf = grads_target(fine, Ccls, dt, setup)[0]
    first = float(fine[np.argmax(gf == 0)]) if (gf == 0).any() else float(coarse[i0])
    nz = np.where(~z)[0]
    last_nonzero = float(coarse[nz[-1]]) if len(nz) else None
    contiguous = bool(z[i0:].all())
    return first, last_nonzero, contiguous, coarse, gt


res_a = {"setups": {}, "classes": [2, 113, 1000, 32000], "grid": "0.01 coarse, 0.001 fine"}
curves = {}
for Ccls in res_a["classes"]:
    for name, dt in DT.items():
        eps = torch.finfo(dt).eps
        for setup in (["pure", "upcast", "fixed"] + (["autocast"] if name != "float32" else [])):
            if Ccls == 32000 and setup in ("upcast", "autocast"):
                continue
            first, last_nz, contig, grid, gt = first_zero(Ccls, dt, setup)
            _, _, ldt = grads_target(np.array([1.0]), Ccls, dt, setup)
            key = f"C{Ccls}|{name}|{setup}"
            p1 = float(np.log(Ccls - 1) + np.log(1 / eps))
            p2 = float(np.log(Ccls - 1) + np.log(2 / eps))
            res_a["setups"][key] = {"first_zero_gap": first, "first_zero_paper_margin": None if first is None else first - float(np.log(Ccls - 1)),
                                    "last_nonzero_gap_coarse": last_nz, "zeros_contiguous": contig,
                                    "pred_ln1_over_eps": p1, "pred_ln2_over_eps": p2, "loss_dtype": ldt}
            print(f"{key:28s} loss dtype {ldt:15s} first g_t==0 at gap {first} "
                  f"(paper margin {None if first is None else round(first - np.log(Ccls - 1), 3)}); "
                  f"pred ln(C-1)+ln(1/eps)={p1:.3f}, +ln(2/eps)={p2:.3f}; last nonzero {last_nz}; contiguous {contig}")
            if Ccls == 113:
                curves[f"{name}|{setup}"] = (grid, gt)

# fixed-version accuracy (C=113) vs float64 closed form on the dtype-rounded gap
acc = {}
for name, dt in DT.items():
    grid, gt = curves[f"{name}|fixed"]
    gq = torch.as_tensor(grid, dtype=torch.float64).to(dt).to(torch.float64).numpy()
    ref = -(112) * np.exp(-gq) / (1 + 112 * np.exp(-gq))
    ok = np.abs(ref) >= torch.finfo(dt).tiny
    rel = np.abs(gt[ok] - ref[ok]) / np.abs(ref[ok])
    gtb = curves[f"{name}|pure"][1]
    relb = np.abs(gtb[ok] - ref[ok]) / np.abs(ref[ok])
    zero_fixed = grid[np.argmax(gt == 0)] if (gt == 0).any() else None
    # mechanism check for the fixed fp16 zero: sigma(r) with r = LSE(z_j - z_t) = ln(C-1) - gap
    acc[name] = {"fixed_max_rel_err_normal": float(rel.max()), "builtin_max_rel_err_normal": float(relb.max()),
                 "fixed_first_zero_gap": None if zero_fixed is None else float(zero_fixed),
                 "smallest_subnormal": float(torch.finfo(dt).smallest_normal) * float(torch.finfo(dt).eps)}
    print(f"C=113 {name}: fixed g_t max rel err (normal range) {rel.max():.2e}; built-in {relb.max():.2e}; "
          f"fixed g_t first exactly 0 at gap {zero_fixed}")
# fp16 fixed zero: where does sigma(r) underflow? r = ln(112) - gap
r = torch.as_tensor(np.log(112) - np.arange(15, 30, 0.001), dtype=torch.float16)
sig = torch.sigmoid(r)
e_m = torch.exp(-torch.as_tensor(np.arange(15, 30, 0.001), dtype=torch.float16))
acc["fp16_sigmoid_r_first_zero_gap"] = float(np.arange(15, 30, 0.001)[np.argmax((sig == 0).numpy())])
acc["fp16_exp_minus_gap_first_zero_gap"] = float(np.arange(15, 30, 0.001)[np.argmax((e_m == 0).numpy())]) if (e_m == 0).any() else None
print("fp16: sigma(ln(112)-gap) first 0 at gap", acc["fp16_sigmoid_r_first_zero_gap"],
      "; exp(-gap) first 0 at gap", acc["fp16_exp_minus_gap_first_zero_gap"])
res_a["fixed_accuracy_C113"] = acc

# ---------------- E4b: cancellation ----------------
CD = 1000
g0 = torch.Generator().manual_seed(0)
zT64 = 3.0 * torch.randn(CD, dtype=torch.float64, generator=g0)
xi = torch.randn(CD, dtype=torch.float64, generator=g0)
epss = np.concatenate([[0.0], np.logspace(-9, 0, 28)])


def builtin_soft(zS, zT, dt, setup):
    x = zS.to(dt).clone().requires_grad_(True)
    if setup == "all-dtype":
        q = torch.softmax(zT.to(dt), -1)
        F.cross_entropy(x[None], q[None], reduction="sum").backward()
    elif setup == "teacher-fp32":
        q = torch.softmax(zT.to(dt).float(), -1)
        F.cross_entropy(x.float()[None], q[None], reduction="sum").backward()
    elif setup == "autocast":
        q = torch.softmax(zT.to(dt).float(), -1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            F.cross_entropy(x[None], q[None], reduction="sum").backward()
    return x.grad.to(torch.float64)


def ref_naive64(zS_r, zT_r):
    return torch.softmax(zS_r, -1) - torch.softmax(zT_r, -1)


def ref_mpmath(zS_r, zT_r):
    mp.mp.dps = 50
    def sm(v):
        m = max(v); e = [mp.e ** (mp.mpf(a) - m) for a in v]; s = mp.fsum(e)
        return [x / s for x in e]
    a, b = sm([mp.mpf(float(v)) for v in zS_r]), sm([mp.mpf(float(v)) for v in zT_r])
    return np.array([float(x - y) for x, y in zip(a, b)])


res_b = {"C": CD, "eps": epss.tolist(), "dtypes": {}, "mpmath_check": {}}
for name, dt in DT.items():
    zT_r = zT64.to(dt).to(torch.float64)
    setups = ["all-dtype", "teacher-fp32"] + (["autocast"] if name == "bfloat16" else [])
    rows = {s: [] for s in setups + ["fixed"]}
    refn = []
    for e in epss:
        zS_r = (zT_r + e * xi).to(dt).to(torch.float64)
        ref = ref_naive64(zS_r, zT_r)
        refn.append(float(ref.norm()))
        for s in setups:
            rows[s].append(float((builtin_soft(zS_r, zT_r, dt, s) - ref).norm()))
        gf = fixed_soft_grad(zS_r.to(dt)[None], zT_r.to(dt)[None])[0].to(torch.float64)
        rows["fixed"].append(float((gf - ref).norm()))
    # where does the eps=0 floor come from? (a) teacher probs stored in dtype, (b) PyTorch's gradient path
    q_store_err = float((torch.softmax(zT_r, -1) - torch.softmax(zT_r.to(dt), -1).to(torch.float64)).norm())
    zf = zT_r.float()
    path_err = float((torch.softmax(zf, -1) - torch.exp(torch.log_softmax(zf, -1))).to(torch.float64).norm())
    # decomposition at eps = 0 (student == teacher): what the built-in all-dtype gradient computes is
    # exp(log_softmax_dtype(z)) - q_dtype; isolate the two stored quantities in float64.
    z_d = zT_r.to(dt)
    lsm_d = torch.log_softmax(z_d, -1).to(torch.float64)
    q_d = torch.softmax(z_d, -1).to(torch.float64)
    p64 = torch.softmax(zT_r, -1)
    decomp = {"only_teacher_probs_in_dtype": float((p64 - q_d).norm()),
              "only_student_logsoftmax_in_dtype": float((torch.exp(lsm_d) - p64).norm()),
              "both_(=built-in_formula)": float((torch.exp(lsm_d) - q_d).norm())}
    print(f"   eps=0 floor decomposition ({name}): " + ", ".join(f"{k}={v:.2e}" for k, v in decomp.items()))
    res_b["dtypes"][name] = {"err": rows, "ref_norm": refn, "teacher_prob_storage_err_l2": q_store_err,
                             "fp32_softmax_vs_exp_logsoftmax_l2": path_err, "eps0_decomposition": decomp}
    j = int(np.argmin(abs(epss - 1e-3)))
    print(f"{name}: eps=0 ||g|| " + ", ".join(f"{s}={rows[s][0]:.3e}" for s in rows)
          + f" | at eps={epss[j]:.1e}: ||g_exact||={refn[j]:.2e}, " + ", ".join(f"{s}={rows[s][j]:.2e}" for s in rows)
          + f" | ||softmax64(zT) - softmax_{name}(zT)||={q_store_err:.2e}; fp32 ||softmax - exp(log_softmax)||={path_err:.2e}")
    for e in [0.0, 1e-6, 1e-3]:
        zS_r = (zT_r + e * xi).to(dt).to(torch.float64)
        a = ref_naive64(zS_r, zT_r).numpy()
        b = ref_mpmath(zS_r.numpy(), zT_r.numpy())
        res_b["mpmath_check"][f"{name}|{e}"] = {"max_abs_diff": float(np.abs(a - b).max()), "mp_norm": float(np.linalg.norm(b))}
    print(f"   mpmath vs float64-naive reference: " + ", ".join(
        f"eps={k.split('|')[1]}: max|diff|={v['max_abs_diff']:.1e} (||ref||={v['mp_norm']:.1e})"
        for k, v in res_b["mpmath_check"].items() if k.startswith(name)))

# ---------------- E4c: forward log-prob zeros ----------------
res_c = {}
for Ccls in [113, 151936]:
    for name, dt in list(DT.items()) + [("float32-log(softmax)", torch.float32)]:
        gaps = np.round(np.arange(0, 45, 0.01), 6)
        out = []
        for i in range(0, len(gaps), max(1, int(2e6 // Ccls))):
            gg = gaps[i:i + max(1, int(2e6 // Ccls))]
            z = torch.zeros(len(gg), Ccls, dtype=torch.float64)
            z[:, 0] = torch.as_tensor(gg)
            z = z.to(dt)
            lp = torch.log(torch.softmax(z, -1))[:, 0] if "log(softmax)" in name else F.log_softmax(z, -1)[:, 0]
            out.append(lp.to(torch.float64).numpy())
        lp = np.concatenate(out)
        first = float(gaps[np.argmax(lp == 0)]) if (lp == 0).any() else None
        res_c[f"C{Ccls}|{name}"] = {"first_exact_zero_gap": first,
                                    "paper_margin": None if first is None else first - float(np.log(Ccls - 1))}
        print(f"E4c C={Ccls} {name}: log p_target first exactly 0 at gap {first} "
              f"(paper margin {None if first is None else round(first - np.log(Ccls - 1), 2)})")

dump(res_a, "e4_collapse"); dump(res_b, "e4_cancellation"); dump(res_c, "e4_forward_logprob")

# ---------------- figures ----------------
FLOOR = 1e-30
def fl(a):
    a = np.abs(np.asarray(a, float)); return np.where(a == 0, FLOOR, a)

fig, ax = plt.subplots(figsize=(8.8, 4.6))
grid = curves["float32|pure"][0]
ax.plot(grid, 112 * np.exp(-grid) / (1 + 112 * np.exp(-grid)), color=INK2, lw=1, ls=":", label="exact (float64)")
for i, name in enumerate(DT):
    ax.plot(grid, fl(curves[f"{name}|pure"][1]), color=COL[i], lw=1.8, label=f"built-in, {name}")
    ax.plot(grid, fl(curves[f"{name}|fixed"][1]), color=COL[i], lw=1.2, ls="--", label=f"Barron et al. fix, {name}")
ax.plot(grid, fl(curves["bfloat16|upcast"][1]), color=COL[4], lw=1.2, ls="-.", label="bf16 logits, fp32 loss")
for i, name in enumerate(DT):
    f0 = res_a["setups"][f"C113|{name}|pure"]["first_zero_gap"]
    if f0:
        ax.axvline(f0, color=COL[i], lw=0.7, ls=":")
ax.set_yscale("log"); ax.set_ylim(FLOOR / 5, 10)
ax.text(0.3, FLOOR * 3, "exact zeros drawn on this floor", fontsize=7, color=INK2)
ax.annotate("fix, float16: each non-target term\ne^(z_j − z_t) underflows float16\n(gap 17.34, same for every C tested)",
            xy=(17.34, 1e-12), xytext=(23.5, 1e-21), fontsize=7, color=COL[2],
            arrowprops=dict(arrowstyle="->", color=COL[2], lw=0.8))
ax.set_xlabel("logit gap g = z_target − z_other (nats); paper margin m = g − ln 112")
ax.set_ylabel("|∂L/∂z_target| (log scale)")
ax.set_title(f"E4a. Softmax Collapse, C = 113, PyTorch {torch.__version__.split('+')[0]} CPU")
ax.legend(fontsize=7.2, loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "e4_collapse")

fig, ax = plt.subplots(figsize=(8.8, 4.4))
x = np.array(epss); x[0] = 1e-10
styles = {"all-dtype": "-", "teacher-fp32": "-.", "autocast": ":", "fixed": "--"}
for i, name in enumerate(DT):
    r = res_b["dtypes"][name]["err"]
    for s, v in r.items():
        ax.plot(x, np.maximum(v, 1e-30), color=COL[i], ls=styles[s], lw=1.6 if s != "fixed" else 1.2,
                marker="o" if s == "all-dtype" else None, ms=3, label=f"{s}, {name}")
ax.plot(x[1:], res_b["dtypes"]["float32"]["ref_norm"][1:], color=INK2, lw=1, ls=":", label="‖exact gradient‖ (fp32 logits)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-31, 10)
ax.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]); ax.set_xticklabels(["ε = 0", "1e-8", "1e-6", "1e-4", "1e-2", "1"])
ax.set_xlabel("student–teacher logit perturbation ε (std of z_S − z_T, logit units)")
ax.set_ylabel("‖g − g_ref‖₂ (1e-30 = exactly 0)")
ax.set_title(f"E4b. Softmax Cancellation, C = {CD}; reference = float64 p − q")
ax.legend(fontsize=6.6, loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "e4_cancellation")

# traceability: bf16 rounding of the gap near the fp32 threshold (explains the 20.688 / 20.75 grid value)
_g = torch.tensor([20.661, 20.6875, 20.688], dtype=torch.float64)
print("bf16 rounding of gaps", _g.tolist(), "->", _g.to(torch.bfloat16).to(torch.float64).tolist(),
      "; bf16 spacing at 20 =", float(torch.finfo(torch.bfloat16).eps) * 16)
