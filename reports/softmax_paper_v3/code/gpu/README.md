# GPU experiments (B4–B6): run these on your machine

None of these ran in the cloud session (no GPU there). Each script was checked on CPU: B6 on CPU reproduces
the v2 E4 thresholds exactly, B5's attention with eviction off matches transformers' eager attention bit for
bit (Llama and gpt-oss, including sinks), and B4's comparison code runs on a synthetic response. Send back
the JSON files they write to `data/`.

| Script | What it measures | Hardware | Time (estimate) |
|---|---|---|---|
| `b6_cuda_thresholds.py --device cuda` | E4 collapse and log p = 0 thresholds on CUDA, C = 113 and 151,936 | RTX 4060 | ~1 min |
| `b4_serving_logprobs.py` | Returned log-probs: SGLang default, SGLang `--rl-on-policy-target fsdp`, vLLM; exact-zero share on confident tokens | RTX 4060, Qwen3-0.6B | ~30 min incl. server start-ups |
| `b5_sink_pinning.py` | Perplexity under window eviction with 0/1/4 pinned 16-token pages vs full cache | Llama-3.1-8B: 24 GB+ (bf16); gpt-oss-20b: 80 GB A100/H100 (~42 GB bf16 weights) | 2–4 GPU-hours for both |

Setup (once):

```bash
pip install torch transformers requests
```

```bash
pip install "sglang[all]" vllm
```

```bash
export CORPUS_DIR=$PWD/../../corpus
```

The corpus is four Project Gutenberg books (`pg1342`, `pg2701`, `pg84`, `pg1661`); `code/corpus.py` shows how to fetch them.

B4 needs each server running in turn (commands are in the script's docstring). B5's success criterion is
pre-registered in the script's docstring; do not edit it after running.
