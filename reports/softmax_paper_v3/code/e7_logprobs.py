"""E7: log-probability precision on real text (CPU).

For each model, 3 passages of 2,048 tokens (Pride and Prejudice, Moby Dick, Sherlock Holmes, starting at
token 20,000) are run through the model in float32. For every next-token position the log-probability of
the actual next token is computed five ways from the same logits:
  ref     float64 log_softmax of the float32 logits (reference)
  fp32    float32 log_softmax                      (vLLM compute_logprobs; SGLang greedy path)
  logsm   float32 log(softmax(.))                  (SGLang standard sampling path: torch.log(probs))
  bf16    logits cast to bfloat16, bfloat16 log_softmax (SGLang rl_on_policy_target path)
  bf16in  logits rounded to bfloat16, then float32 log_softmax (a bf16 model output read in fp32)
Reported per confidence bucket (p_ref > 0.9 / 0.99 / 0.999): number of tokens, exact zeros per method, max and
mean |error| against ref; and the smallest p_ref at which each method returned exactly 0.
Usage: python e7_logprobs.py [model ...]   (writes data/e7_<model>.json)"""
import json
import os
import sys
import numpy as np
import torch
import models
from corpus import load_text

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
TEXTS, OFFSET, N = ["pg1342", "pg2701", "pg1661"], 20000, 2048


def run(name):
    impl = "sdpa" if name in models.HF_MODELS else "eager"
    model, tok = models.load(name, impl)
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and tok("a")["input_ids"][0] == tok.bos_token_id) else []
    cols = {k: [] for k in ["ref", "fp32", "logsm", "bf16", "bf16in"]}
    for t in TEXTS:
        ids = bos + tok(load_text(t), add_special_tokens=False)["input_ids"][OFFSET: OFFSET + N - len(bos)]
        x = torch.tensor(ids)[None]
        with torch.no_grad():
            lo = model(x, use_cache=False).logits[0, :-1].float()
        y = x[0, 1:, None]
        g = lambda a: a.gather(-1, y)[:, 0].double()
        cols["ref"].append(g(torch.log_softmax(lo.double(), -1)))
        cols["fp32"].append(g(torch.log_softmax(lo, -1)))
        cols["logsm"].append(g(torch.log(torch.softmax(lo, -1))))
        cols["bf16"].append(g(torch.log_softmax(lo.bfloat16(), -1)))
        cols["bf16in"].append(g(torch.log_softmax(lo.bfloat16().float(), -1)))
        del lo
    c = {k: torch.cat(v).numpy() for k, v in cols.items()}
    p = np.exp(c["ref"])
    res = dict(model=name, n_tokens=int(len(p)), buckets={}, min_p_at_exact_zero={}, max_p_nonzero={})
    for thr in (0.9, 0.99, 0.999):
        m = p > thr
        res["buckets"][str(thr)] = dict(n=int(m.sum()), **{
            k: dict(exact_zeros=int((c[k][m] == 0).sum()),
                    max_abs_err=float(np.abs(c[k][m] - c["ref"][m]).max()) if m.any() else None,
                    mean_abs_err=float(np.abs(c[k][m] - c["ref"][m]).mean()) if m.any() else None)
            for k in ["fp32", "logsm", "bf16", "bf16in"]})
    for k in ["fp32", "logsm", "bf16", "bf16in"]:
        z = c[k] == 0
        res["min_p_at_exact_zero"][k] = float(p[z].min()) if z.any() else None
        res["max_p_nonzero"][k] = float(p[~z].max())
    res["all_tokens_mean_abs_err"] = {k: float(np.abs(c[k] - c["ref"]).mean()) for k in ["fp32", "logsm", "bf16", "bf16in"]}
    b = res["buckets"]["0.9"]
    print(f"{name}: tokens {res['n_tokens']}, p>0.9: {b['n']}; exact zeros fp32 {b['fp32']['exact_zeros']}, "
          f"log(softmax) {b['logsm']['exact_zeros']}, bf16 {b['bf16']['exact_zeros']}, bf16-logits->fp32 {b['bf16in']['exact_zeros']}; "
          f"bf16 max err {b['bf16']['max_abs_err']:.2e}; smallest p with bf16 log p == 0: {res['min_p_at_exact_zero']['bf16']}",
          flush=True)
    json.dump(res, open(os.path.join(OUT, f"e7_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(models.HF_MODELS) + list(models.FLA_MODELS)):
        run(n)
