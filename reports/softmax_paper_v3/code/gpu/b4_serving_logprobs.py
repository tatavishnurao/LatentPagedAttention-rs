"""B4 (GPU): log-probabilities returned by SGLang (default and RL on-policy mode) and vLLM, compared with a
float64 recomputation.

Start three servers for the same model (RTX 4060 8 GB is enough for Qwen/Qwen3-0.6B; run them one at a time):
  python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000
  python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30001 \
      --enable-deterministic-inference --rl-on-policy-target fsdp
  vllm serve Qwen/Qwen3-0.6B --port 8000 --dtype bfloat16
(Flags as documented in SGLang docs/advanced_features/server_arguments.mdx @36273b9. In SGLang's sampler, the
bf16 log_softmax path is taken when rl_on_policy_target is set, deterministic inference is on and sampling is
"simple" (no top-k / top-p / min-p). Greedy requests bypass it, so this script samples at temperature 1.0.)

Then:  python b4_serving_logprobs.py --engine sglang --url http://localhost:30000 --tag sglang_default
       python b4_serving_logprobs.py --engine sglang --url http://localhost:30001 --tag sglang_rl_fsdp
       python b4_serving_logprobs.py --engine vllm   --url http://localhost:8000  --tag vllm
For each of 32 prompts (Gutenberg passages, 512 tokens) the server samples 128 tokens and returns the
log-probability of each sampled token. The script recomputes log p of the same tokens with Hugging Face in
float32 and a float64 log_softmax, and reports, for tokens with reference p > 0.9 / 0.99: the share whose
returned log p is exactly 0, and the max |returned - reference|. Differences also include bf16-vs-fp32 model
numerics, so exact zeros are the primary metric. Requires: requests, transformers, torch.
--offline-check runs the comparison code on a synthetic response (no server); it is a code check only."""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def query(engine, url, model, prompt_ids, n_new, seed):
    import requests
    if engine == "sglang":
        r = requests.post(url + "/generate", json=dict(
            input_ids=prompt_ids, return_logprob=True,
            sampling_params=dict(temperature=1.0, max_new_tokens=n_new, seed=seed, ignore_eos=True)), timeout=600).json()
        lp = r["meta_info"]["output_token_logprobs"]
        return [int(t[1]) for t in lp], [float(t[0]) for t in lp]
    r = requests.post(url + "/v1/completions", json=dict(
        model=model, prompt=prompt_ids, max_tokens=n_new, temperature=1.0, logprobs=0, seed=seed,
        ignore_eos=True, return_tokens_as_token_ids=True), timeout=600).json()
    lp = r["choices"][0]["logprobs"]
    ids = [int(t.split(":")[1]) for t in lp["tokens"]]
    return ids, [float(x) for x in lp["token_logprobs"]]


def reference(model, prompt_ids, out_ids):
    x = torch.tensor(prompt_ids + out_ids)[None].to(model.device)
    with torch.no_grad():
        lo = model(x, use_cache=False).logits[0, len(prompt_ids) - 1:-1].double()
    return torch.log_softmax(lo, -1).gather(-1, torch.tensor(out_ids, device=lo.device)[:, None])[:, 0].cpu().numpy()


def summarise(ref, got):
    ref, got = np.asarray(ref), np.asarray(got)
    p = np.exp(ref)
    out = dict(n_tokens=int(len(ref)))
    for thr in (0.9, 0.99):
        m = p > thr
        out[str(thr)] = dict(n=int(m.sum()), exact_zero_share=float((got[m] == 0).mean()) if m.any() else None,
                             max_abs_diff=float(np.abs(got[m] - ref[m]).max()) if m.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["sglang", "vllm"])
    ap.add_argument("--url")
    ap.add_argument("--tag")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--offline-check", action="store_true")
    a = ap.parse_args()
    if a.offline_check:
        rng = np.random.default_rng(0)
        ref = np.log(rng.uniform(0.5, 1.0, 1000))
        got = ref.astype(np.float32).astype(np.float64)
        got[np.exp(ref) > 0.996] = 0.0  # what a bf16 path would do (E4c threshold), synthetic
        print(json.dumps(summarise(ref, got)))
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from corpus import load_text
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    text = tok(load_text("pg2701"), add_special_tokens=False)["input_ids"]
    R, G = [], []
    for i in range(a.prompts):
        pr = text[5000 + i * 2000: 5000 + i * 2000 + 512]
        out_ids, got = query(a.engine, a.url, a.model, pr, 128, seed=i)
        R.append(reference(model, pr, out_ids))
        G.append(got)
    res = dict(tag=a.tag, engine=a.engine, model=a.model, **summarise(np.concatenate(R), np.concatenate(G)))
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "..", "data", f"b4_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
