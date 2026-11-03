"""E6 + E7: real-model probes on small CPU models (<= 1B parameters).

STATUS IN THIS REPORT: NOT RUN. The session's network policy denied the Hugging Face weight CDN
(us.aws.cdn.hf.co, the Xet bridge that /resolve/ redirects safetensors to), so no weights could be
downloaded. `--smoke` runs every code path on tiny randomly initialised copies of the real configs to
check the code; smoke output is not a result and is never reported as one.

E6, per model and context length N (256 .. --max-n):
  * mean attention entropy per layer (nats) over the last --q-rows query positions and all heads,
    compared with ln N;
  * attention mass on position 0 per layer (sink strength);
  * planted needle: a fixed 8-token string inserted at a random depth in filler text; for every head,
    the attention weight the final query puts on the needle tokens; the best head per N is kept and
    w(N) is fitted to e^D / (e^D + N - 1) by least squares on logit(w) (D reported per model).
E7, per model: next-token log-probabilities on the same text. Logits are taken in float32 from the
  model, then log p is computed (a) float32 log_softmax, (b) logits cast to bfloat16 then bfloat16
  log_softmax, (c) float32 log(softmax) (the form SGLang's standard sampling path uses). For tokens
  with p > 0.9 under (a): count exact zeros and the max |error| against a float64 log_softmax.

Usage:  python e6_e7_probes.py --text-file corpus.txt [--max-n 8192]      (real run)
        python e6_e7_probes.py --smoke                                      (code check only)
Models: Qwen/Qwen3-0.6B, google/gemma-3-1b-it, HuggingFaceTB/SmolLM2-360M. The matched pair
zaydzuhri/vanilla-340M-4096-model / softpick-340M-4096-model uses a custom flash-linear-attention
class and needs its own loader (not implemented here; see report section 9)."""
import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F

MODELS = ["Qwen/Qwen3-0.6B", "google/gemma-3-1b-it", "HuggingFaceTB/SmolLM2-360M"]
NEEDLE = " The vault passcode is 7 3 9 1 ."


def load(repo, smoke, cfg_dir):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if smoke:
        cfg = AutoConfig.from_pretrained(os.path.join(cfg_dir, repo.replace("/", "_") + ".json"))
        for k, v in dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, vocab_size=512, sliding_window=64).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
            cfg.layer_types = cfg.layer_types[:2]
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
        return model.float().eval(), None
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32, attn_implementation="eager")
    return model.eval(), tok


class Stats:
    """forward hooks on attention modules: consume attention weights layer by layer, then drop them."""
    def __init__(self, model, q_rows, needle_slice):
        self.rows, self.needle, self.out, self.h = q_rows, needle_slice, [], []
        for name, mod in model.named_modules():
            if name.endswith("self_attn"):
                self.h.append(mod.register_forward_hook(self.hook))

    def hook(self, mod, args, output):
        w = output[1]  # (B, H, Q, K) with eager attention + output_attentions
        if w is None:
            return output
        w = w[0, :, -self.rows:, :].double()
        ent = -(w * torch.log(w.clamp_min(1e-30))).sum(-1).mean().item()
        sink = w[:, :, 0].mean().item()
        needle = w[:, -1, self.needle].sum(-1).tolist() if self.needle is not None else None
        self.out.append(dict(entropy=ent, sink=sink, needle_per_head=needle))
        return (output[0], None) + tuple(output[2:])

    def close(self):
        for h in self.h:
            h.remove()


def run_e6(model, ids_all, needle_ids, Ns, q_rows, seed):
    rng = np.random.default_rng(seed)
    res = {}
    for N in Ns:
        filler = ids_all[: N - len(needle_ids)]
        pos = int(rng.integers(N // 10, N - N // 10))
        ids = torch.tensor(filler[:pos] + needle_ids + filler[pos:])[None, :N]
        st = Stats(model, q_rows, slice(pos, pos + len(needle_ids)))
        with torch.no_grad():
            model(ids, output_attentions=True, use_cache=False)
        st.close()
        best = max(max(l["needle_per_head"]) for l in st.out)
        res[N] = dict(entropy=[l["entropy"] for l in st.out], sink=[l["sink"] for l in st.out],
                      needle_best=best, needle_pos=pos)
    Nv = np.array(sorted(res)); w = np.clip([res[n]["needle_best"] for n in Nv], 1e-9, 1 - 1e-9)
    # logit(w) = D - ln(N-1)  ->  D = mean(logit(w) + ln(N-1))
    D = float(np.mean(np.log(w / (1 - w)) + np.log(Nv - 1)))
    return res, D


def run_e7(model, ids):
    with torch.no_grad():
        logits = model(torch.tensor(ids)[None], use_cache=False).logits[0, :-1].float()
    tgt = torch.tensor(ids[1:])
    ref = torch.log_softmax(logits.double(), -1).gather(-1, tgt[:, None])[:, 0]
    a = torch.log_softmax(logits, -1).gather(-1, tgt[:, None])[:, 0].double()
    b = torch.log_softmax(logits.bfloat16(), -1).gather(-1, tgt[:, None])[:, 0].double()
    c = torch.log(torch.softmax(logits, -1)).gather(-1, tgt[:, None])[:, 0].double()
    hi = ref.exp() > 0.9
    def summ(x):
        return dict(exact_zeros=int((x[hi] == 0).sum()), max_abs_err=float((x[hi] - ref[hi]).abs().max()) if hi.any() else None)
    return dict(n_tokens=int(len(tgt)), n_high_conf=int(hi.sum()), fp32_log_softmax=summ(a),
                bf16_log_softmax=summ(b), fp32_log_of_softmax=summ(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file")
    ap.add_argument("--max-n", type=int, default=8192)
    ap.add_argument("--q-rows", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config-dir", default=os.path.join(os.path.dirname(__file__), "..", "sources", "configs"))
    args = ap.parse_args()
    torch.manual_seed(0)
    out = {}
    for repo in MODELS:
        model, tok = load(repo, args.smoke, args.config_dir)
        if args.smoke:
            rng = np.random.default_rng(0)
            ids_all = rng.integers(0, 512, 4096).tolist(); needle_ids = [7, 3, 9, 1, 11]; Ns = [64, 128, 256]
        else:
            text = open(args.text_file).read()
            ids_all = tok(text)["input_ids"]
            needle_ids = tok(NEEDLE, add_special_tokens=False)["input_ids"]
            Ns = [n for n in [256, 512, 1024, 2048, 4096, 8192, 16384] if n <= min(args.max_n, len(ids_all))]
        e6, D = run_e6(model, ids_all, needle_ids, Ns, args.q_rows, seed=0)
        e7 = run_e7(model, ids_all[: Ns[-1]])
        out[repo] = dict(e6=e6, fitted_delta=D, e7=e7, smoke=args.smoke)
        print(repo, "smoke" if args.smoke else "", "fitted Delta=%.2f" % D, "| E7:", json.dumps(e7))
    name = "e6_e7_smoke" if args.smoke else "e6_e7_probes"
    with open(os.path.join(os.path.dirname(__file__), "..", "data", name + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
