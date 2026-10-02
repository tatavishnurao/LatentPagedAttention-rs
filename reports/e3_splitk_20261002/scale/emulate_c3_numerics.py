"""CPU emulation of the C3 numerics on the heavy-tailed scale case (no GPU).
Isolates FP16 rounding of q' and of P (per 32-token step, online softmax) against the
FP32-arithmetic reference on the same FP16-stored latent. Tensor-core accumulation is not
emulated (its rounding is hardware-specific)."""
import numpy as np
M64 = (1 << 64) - 1
def splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)
def det(n, step, off):
    i = np.arange(n); lane = (i % 257).astype(np.float32)
    return (np.sin(lane * np.float32(step) + np.float32(off)).astype(np.float32) * np.float32(0.75)
            + ((i % 13).astype(np.float32) - 6.0) * np.float32(0.01)).astype(np.float32)
HQ, HKV, D, L, BLK = 16, 4, 64, 32, 16
def heavy(seq):
    out = np.empty(seq * L, np.float32); outl = {5, 1000, 4000, 20000}
    for i in range(seq * L):
        h = splitmix(0x5eed ^ i); u = ((h >> 11) + 0.5) / float(1 << 53)
        sign = 1.0 if h & 1 == 0 else -1.0; tail = min(u ** (-1.0 / 3.0) - 1.0, 40.0)
        v = np.float32(sign * 0.3 * tail)
        if (i // L) in outl: v *= np.float32(12.0)
        out[i] = v
    return out
N = 8192
z = heavy(N).reshape(N, L).astype(np.float16).astype(np.float32)
q = det(HQ * D, 0.011, -0.4).reshape(HQ, D)
kp = det(L * HKV * D, 0.005, -0.7).reshape(L, HKV, D)   # canonical (latent, kv, dim)
vp = det(L * HKV * D, 0.006, 0.3).reshape(L, HKV, D)
g = np.arange(HQ) // (HQ // HKV)
qp = np.einsum('hd,lhd->hl', q, kp[:, g, :]).astype(np.float32)          # [HQ, L] FP32
scale = np.float32(1 / 8)
def run(qp_used, p16, step=32):
    m = np.full(HQ, -1e30, np.float32); lsum = np.zeros(HQ, np.float32); acc = np.zeros((HQ, L), np.float32)
    for t0 in range(0, N, step):
        zt = z[t0:t0 + step]; s = (qp_used @ zt.T) * scale
        mn = np.maximum(m, s.max(1)); p = np.exp(s - mn[:, None]); a = np.exp(m - mn)
        lsum = lsum * a + p.sum(1); pu = p.astype(np.float16).astype(np.float32) if p16 else p
        acc = acc * a[:, None] + pu @ zt; m = mn
    ctx = acc / lsum[:, None]
    return np.einsum('hl,lhd->hd', ctx, vp[:, g, :])
# FP32-arithmetic reference (two-pass softmax, same FP16-stored latent)
s = (qp @ z.T) * scale; p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
ref = np.einsum('hl,lhd->hd', p @ z, vp[:, g, :])
q16 = qp.astype(np.float16).astype(np.float32)
for name, args in [("all FP32 (online softmax only)", (qp, False)), ("FP16 P only", (qp, True)),
                   ("FP16 q' only", (q16, False)), ("FP16 q' + FP16 P (C3 numerics)", (q16, True))]:
    out = run(*args); e = np.abs(out - ref)
    print(f"{name:34s} max={e.max():.3e} mean={e.mean():.3e}")
print("max |score|", float(np.abs(s).max()), " max |q'|", float(np.abs(qp).max()))
