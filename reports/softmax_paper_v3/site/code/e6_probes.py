"""E6: real-model attention probes on CPU (float32, exact attention via probe_attention.py).

Models: qwen3-0.6b, gemma3-1b-it, smollm2-360m (contexts 256..8192) and the matched vanilla/softpick 340M pair
(trained at 4,096 tokens; contexts 256..4096).
For each context length N:
  (a) Statistics on natural text: 3 passages (Pride and Prejudice, Moby Dick, Sherlock Holmes; Project
      Gutenberg), each N tokens starting at token 5,000 of the book, with the model's own BOS rule.
      Per layer, over the last 64 query positions: mean attention entropy (nats); mean weight on position 0;
      share of heads whose mean weight on position 0 exceeds 0.3 (Gu et al.) and 0.8 (Barbero et al.).
  (b) Planted needle: NEEDLE inserted at relative depths 0.25 / 0.5 / 0.75 of Moby Dick filler, followed by CUE.
      For every head, the weight the final query puts on the needle's answer tokens. Reported for the best
      head at each N, and for a fixed head (the best head averaged over depths at N = 256). The fixed-gap law
      w = e^D / (e^D + N - 1) gives D_N = logit(w) + ln(N - 1); a constant D_N across N matches the law.
      Also records whether the model's top-1 next token is the first answer token.
Usage: python e6_probes.py [model ...]   (writes data/e6_<model>.json)"""
import json
import math
import os
import sys
import time
import numpy as np
import torch
import models
import probe_attention as pa
from corpus import load_text

torch.manual_seed(0)
torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
NEEDLE = " The name of the secret city is Vellmora."
ANSWER = " Vellmora"
CUE = " Question: What is the name of the secret city? Answer: The name of the secret city is"
TEXTS = ["pg1342", "pg2701", "pg1661"]
OFFSET = 5000
MAX_N = {"vanilla-340m": 4096, "softpick-340m": 4096}
NS = [256, 512, 1024, 2048, 4096, 8192]


def enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def run(name):
    model, tok = models.load(name, "probe")
    bos = [tok.bos_token_id] if (tok("a")["input_ids"][0] == tok.bos_token_id and tok.bos_token_id is not None) else []
    books = {t: enc(tok, load_text(t)) for t in TEXTS}
    needle, answer, cue = enc(tok, NEEDLE), enc(tok, ANSWER), enc(tok, CUE)
    # locate the answer tokens inside the needle (tokenisation in context)
    ans_off = next(i for i in range(len(needle) - len(answer) + 1) if needle[i:i + len(answer)] == answer)
    ns = [n for n in NS if n <= MAX_N.get(name, 8192)]
    res = dict(model=name, bos=bool(bos), needle_tokens=len(needle), answer_tokens=answer, Ns=ns, stats={}, needle={})
    for N in ns:
        t0 = time.time()
        per_text = []
        for t in TEXTS:
            ids = bos + books[t][OFFSET: OFFSET + N - len(bos)]
            pa.STATE.update(records={}, needle=None)
            with torch.no_grad():
                model(torch.tensor(ids)[None], use_cache=False)
            per_text.append({L: pa.STATE["records"][L] for L in sorted(pa.STATE["records"])})
        layers = sorted(per_text[0])
        res["stats"][N] = dict(
            entropy=[float(np.mean([np.mean(pt[L]["entropy"]) for pt in per_text])) for L in layers],
            sink=[float(np.mean([np.mean(pt[L]["sink"]) for pt in per_text])) for L in layers],
            sink_rate_0p3=float(np.mean([np.mean(np.array(pt[L]["sink"]) > 0.3) for pt in per_text for L in layers])),
            sink_rate_0p8=float(np.mean([np.mean(np.array(pt[L]["sink"]) > 0.8) for pt in per_text for L in layers])),
            mass=[float(np.mean([np.mean(pt[L].get("mass", [1.0])) for pt in per_text])) for L in layers],
            n_visible=[float(np.mean([pt[L]["n_visible"] for pt in per_text])) for L in layers],
            sliding=[per_text[0][L]["sliding"] for L in layers])
        # needle
        depth_res = []
        filler = books["pg2701"][OFFSET:]
        body = N - len(bos) - len(needle) - len(cue)
        for d in (0.25, 0.5, 0.75):
            pos = int(d * body)
            ids = bos + filler[:pos] + needle + filler[pos:body] + cue
            assert len(ids) == N
            a0 = len(bos) + pos + ans_off
            pa.STATE.update(records={}, needle=list(range(a0, a0 + len(answer))))
            with torch.no_grad():
                lo = model(torch.tensor(ids)[None], use_cache=False).logits[0, -1]
            W = np.array([pa.STATE["records"][L]["needle"] for L in sorted(pa.STATE["records"])])  # layers x heads
            depth_res.append(dict(depth=d, W=W.tolist(), top1_is_answer=bool(int(lo.argmax()) == answer[0]),
                                  p_answer=float(torch.softmax(lo.double(), -1)[answer[0]])))
        res["needle"][N] = depth_res
        print(f"{name} N={N}: {time.time() - t0:.0f}s; sink_rate(0.3)={res['stats'][N]['sink_rate_0p3']:.3f}; "
              f"top1 answer {[r['top1_is_answer'] for r in depth_res]}", flush=True)
    # needle summary
    Wm = {N: np.mean([np.array(r["W"]) for r in res["needle"][N]], 0) for N in ns}
    l0, h0 = np.unravel_index(np.argmax(Wm[ns[0]]), Wm[ns[0]].shape)
    summ = []
    for N in ns:
        wf = float(Wm[N][l0, h0])
        wb = float(Wm[N].max())
        lg = lambda w: math.log(max(w, 1e-12) / max(1 - w, 1e-12)) + math.log(N - 1)
        summ.append(dict(N=N, w_fixed=wf, w_best=wb, D_fixed=lg(wf), D_best=lg(wb),
                         top1_rate=float(np.mean([r["top1_is_answer"] for r in res["needle"][N]])),
                         p_answer=float(np.mean([r["p_answer"] for r in res["needle"][N]]))))
    res["needle_summary"] = dict(fixed_head=[int(l0), int(h0)], rows=summ)
    for r in summ:
        print(f"  N={r['N']}: fixed head L{l0}H{h0} w={r['w_fixed']:.4f} D={r['D_fixed']:.2f}; best w={r['w_best']:.4f} "
              f"D={r['D_best']:.2f}; top1={r['top1_rate']:.2f} p(answer)={r['p_answer']:.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, f"e6_{name}.json"), "w"))


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(models.HF_MODELS) + list(models.FLA_MODELS)):
        run(n)
