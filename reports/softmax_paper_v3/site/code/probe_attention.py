"""Exact, chunked softmax / softpick attention for Hugging Face models, recording statistics on the fly.

Registered with transformers as attn_implementation="probe" (softmax) and "probe_softpick" (softpick).
Computes the same numbers as transformers' eager_attention_forward (fp32 softmax, additive eager mask,
optional soft-cap), but processes queries in chunks so long contexts fit in memory on CPU. While it runs it
records, per layer, for the last `q_rows` query positions:
  * mean row entropy (nats), over heads and rows (of the renormalised row for softpick)
  * mean row mass (1 for softmax; the softpick row sum)
  * mean weight on key position 0 (per head and averaged)
  * per-head weight that the final query puts on a set of "needle" key positions
  * the number of keys each recorded row can see (to compare with ln N)
validate_against_eager() checks the logits against attn_implementation="eager" before any data is used.
"""
import math
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask

STATE = {"q_rows": 64, "needle": None, "records": {}, "chunk": 256, "enabled": True}


def softpick(x, valid, dim=-1, eps=1e-8):
    """Softpick (Zuhri et al.) exactly as in the authors' naive_softpick.py, with an explicit validity mask
    in place of x.isfinite() because the HF eager mask is a large finite negative number, not -inf."""
    x = torch.where(valid, x, torch.full_like(x, -float("inf")))
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(valid, torch.abs(x_e_1), torch.zeros_like(x_e_1))
    return r_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps)


def _repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def _probe(module, query, key, value, attention_mask, scaling, dropout=0.0, softcap=None, kind="softmax", **kw):
    n_rep = query.shape[1] // key.shape[1]
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)
    B, H, Q, D = query.shape
    K = k.shape[2]
    out = torch.empty(B, H, Q, v.shape[-1], dtype=query.dtype)
    rows0 = max(0, Q - STATE["q_rows"])
    ent_sum = torch.zeros(H, dtype=torch.float64)
    sink_sum = torch.zeros(H, dtype=torch.float64)
    mass_sum = torch.zeros(H, dtype=torch.float64)
    nvis_sum = 0.0
    nrows = 0
    needle_w = None
    for s in range(0, Q, STATE["chunk"]):
        e = min(Q, s + STATE["chunk"])
        sc = torch.matmul(query[:, :, s:e], k.transpose(2, 3)) * scaling
        if softcap is not None:
            sc = torch.tanh(sc / softcap) * softcap
        if attention_mask is not None:
            m = attention_mask[:, :, s:e, :K]
            sc = sc + m
            valid = m > torch.finfo(m.dtype).min / 2
        else:
            valid = torch.ones_like(sc, dtype=torch.bool)
        if kind == "softmax":
            p = F.softmax(sc, dim=-1, dtype=torch.float32)
        else:
            p = softpick(sc.float(), valid)
        out[:, :, s:e] = torch.matmul(p.to(query.dtype), v)
        if STATE["enabled"] and e > rows0:
            a = max(s, rows0)
            pr = p[0, :, a - s:e - s].double()
            mass = pr.sum(-1)  # 1 for softmax; <= 1 for softpick
            pn = pr / mass.clamp_min(1e-300)[..., None]
            ent = -(pn * torch.log(pn.clamp_min(1e-300))).sum(-1)  # entropy of the renormalised row
            ent_sum += ent.sum(-1)
            mass_sum += mass.sum(-1)
            sink_sum += pr[:, :, 0].sum(-1)
            nvis_sum += valid[0, 0, a - s:e - s].sum().item()
            nrows += e - a
            if e == Q and STATE["needle"] is not None:
                needle_w = pr[:, -1, STATE["needle"]].sum(-1)
    if STATE["enabled"]:
        STATE["records"][module.layer_idx] = dict(
            entropy=(ent_sum / nrows).tolist(), sink=(sink_sum / nrows).tolist(), mass=(mass_sum / nrows).tolist(),
            n_visible=nvis_sum / nrows, needle=None if needle_w is None else needle_w.tolist(),
            sliding=getattr(module, "sliding_window", None))
    return out.transpose(1, 2).contiguous(), None


def probe_softmax(module, q, k, v, mask, scaling, dropout=0.0, **kw):
    return _probe(module, q, k, v, mask, scaling, dropout, softcap=kw.get("softcap"), kind="softmax")


def probe_softpick(module, q, k, v, mask, scaling, dropout=0.0, **kw):
    return _probe(module, q, k, v, mask, scaling, dropout, softcap=kw.get("softcap"), kind="softpick")


AttentionInterface.register("probe", probe_softmax)
AttentionInterface.register("probe_softpick", probe_softpick)
AttentionMaskInterface.register("probe", eager_mask)
AttentionMaskInterface.register("probe_softpick", eager_mask)


def softpick_eager(module, q, k, v, mask, scaling, dropout=0.0, **kw):
    """Unchunked reference for validation of the chunked softpick path."""
    n_rep = q.shape[1] // k.shape[1]
    k, v = _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)
    sc = torch.matmul(q, k.transpose(2, 3)) * scaling + mask
    valid = mask > torch.finfo(mask.dtype).min / 2
    p = softpick(sc.float(), valid).to(q.dtype)
    return torch.matmul(p, v).transpose(1, 2).contiguous(), p


AttentionInterface.register("softpick_eager", softpick_eager)
AttentionMaskInterface.register("softpick_eager", eager_mask)
