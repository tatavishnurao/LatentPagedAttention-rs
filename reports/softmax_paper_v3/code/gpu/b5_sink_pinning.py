"""B5 (GPU): do learned sinks make "pin the first KV page" unnecessary under window eviction?

PRE-REGISTERED (written before any run; do not edit after running):
  Setup: context N = 8,192 tokens of public-domain text (Project Gutenberg books, see corpus.py); eviction
  keeps a recent window of W = 1,024 tokens plus P pinned prefix tokens, P in {0, 16, 64} (0, 1, 4 pages of
  16 tokens). Perplexity is measured on the last 4,096 positions and compared with the full cache.
  Positive result: gpt-oss-20b stays within 1% of its full-cache perplexity with P = 0, while
  Llama-3.1-8B degrades by more than 10% with P = 0 and recovers to within 1% with P = 16 or 64.
  Anything else is reported as it is.

Method: eviction is simulated with an attention mask (query t may see keys k < P or t - W < k <= t). Keys keep
their original RoPE positions; StreamingLLM instead re-indexes positions inside the cache, so this is a
mask-level simulation of eviction, not a paged-cache implementation. gpt-oss's own banded (128-token) layers
are left unchanged; the window applies on top of every layer's native mask. The attention function is exact
softmax with gpt-oss's learned per-head sink logit when the module has one, computed in query chunks.

Hardware: Llama-3.1-8B bf16 ~16 GB; gpt-oss-20b loads at ~42 GB in bf16 when the MXFP4 weights are
dequantised. Use an 80 GB A100/H100 for gpt-oss (about 2-4 GPU-hours for both models), or run Llama only.
Usage:  python b5_sink_pinning.py --model meta-llama/Llama-3.1-8B --device cuda
        python b5_sink_pinning.py --model openai/gpt-oss-20b --device cuda
        python b5_sink_pinning.py --smoke            (CPU code check with tiny random models; not a result)"""
import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import AttentionMaskInterface, eager_mask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
EVICT = {"P": None, "W": None, "chunk": 512}


def _rep(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def evict_attention(module, q, k, v, mask, scaling, dropout=0.0, **kw):
    n = q.shape[1] // k.shape[1]
    k, v = _rep(k, n), _rep(v, n)
    B, H, Q, D = q.shape
    K = k.shape[2]
    out = torch.empty(B, H, Q, v.shape[-1], dtype=q.dtype, device=q.device)
    kpos = torch.arange(K, device=q.device)
    sinks = getattr(module, "sinks", None)
    for s in range(0, Q, EVICT["chunk"]):
        e = min(Q, s + EVICT["chunk"])
        sc = (torch.matmul(q[:, :, s:e], k.transpose(2, 3)) * scaling).float()
        if mask is not None:
            sc = sc + mask[:, :, s:e, :K].float()
        if EVICT["W"] is not None:
            qpos = torch.arange(s, e, device=q.device)[:, None] + (K - Q)
            keep = (kpos[None] < EVICT["P"]) | (kpos[None] > qpos - EVICT["W"])
            sc = sc.masked_fill(~keep[None, None], float("-inf"))
        if sinks is not None:
            sk = sinks.float().reshape(1, -1, 1, 1).expand(B, -1, e - s, 1)
            comb = torch.cat([sc, sk], -1)
            p = torch.softmax(comb - comb.max(-1, keepdim=True).values, -1)[..., :-1]
        else:
            p = torch.softmax(sc, -1)
        out[:, :, s:e] = torch.matmul(p.to(v.dtype), v)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("evict", evict_attention)
AttentionMaskInterface.register("evict", eager_mask)


def ppl(model, ids, tail):
    with torch.no_grad():
        lo = model(ids, use_cache=False).logits[0, -tail - 1:-1].float()
    return math.exp(F.cross_entropy(lo, ids[0, -tail:]).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--tail", type=int, default=4096)
    ap.add_argument("--books", default="pg2701,pg1342,pg1661")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    results = []
    if a.smoke:
        runs = []
        for mt in ("llama", "gpt_oss"):
            cfg = AutoConfig.for_model(mt, vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
            if mt == "gpt_oss":
                cfg.num_local_experts, cfg.num_experts_per_tok, cfg.sliding_window = 4, 2, 32
                cfg.layer_types = ["sliding_attention", "full_attention"]
            torch.manual_seed(0)
            runs.append((mt, AutoModelForCausalLM.from_config(cfg, attn_implementation="evict").float().eval(), None))
        a.N, a.W, a.tail, a.device = 384, 64, 128, "cpu"
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="evict").to(a.device).eval()
        runs = [(a.model, m, tok)]
    from corpus import load_text
    for name, model, tok in runs:
        for book in a.books.split(","):
            if tok is None:
                ids = torch.randint(0, 256, (1, a.N), generator=torch.Generator().manual_seed(0))
            else:
                t = tok(load_text(book))["input_ids"]
                ids = torch.tensor(t[:1] + t[5000:5000 + a.N - 1])[None]
            ids = ids.to(a.device)
            EVICT.update(P=None, W=None)
            full = ppl(model, ids, a.tail)
            row = dict(model=name, book=book, N=a.N, W=a.W, tail=a.tail, ppl_full=full, by_P={})
            for P in (0, 16, 64):
                EVICT.update(P=P, W=a.W)
                pp = ppl(model, ids, a.tail)
                row["by_P"][P] = dict(ppl=pp, rel_change=pp / full - 1)
            results.append(row)
            print(json.dumps(row), flush=True)
            if tok is None:
                break
    out = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                       "b5_smoke.json" if a.smoke else f"b5_{a.model.replace('/', '_')}.json")
    json.dump(results, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
