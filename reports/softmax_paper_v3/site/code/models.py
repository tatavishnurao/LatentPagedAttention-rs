"""Model loading for the real-model probes (CPU, float32).

HF models load with transformers. The softpick authors' matched 340M pair (flash-linear-attention
'transformer' class: pre-norm RMSNorm eps 1e-6, NeoX-style RoPE base 10,000, SwiGLU, untied head, MHA with
16 heads of dim 64) is mapped onto transformers' LlamaForCausalLM by renaming weights. fla_reference_logits()
is an independent forward pass written from the fla source, used to check that mapping."""
import glob
import json
import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
import probe_attention  # noqa: F401  (registers attention implementations)

HF_CACHE = os.environ.get("HF_MODELS", os.path.join(os.path.dirname(__file__), "..", "hf_models"))
HF_MODELS = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "gemma3-1b-it": "google/gemma-3-1b-it",
             "smollm2-360m": "HuggingFaceTB/SmolLM2-360M"}
FLA_MODELS = {"vanilla-340m": "zaydzuhri/vanilla-340M-4096-model", "softpick-340m": "zaydzuhri/softpick-340M-4096-model"}


def snapshot(repo):
    d = glob.glob(os.path.join(HF_CACHE, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    if not d:
        raise FileNotFoundError(repo)
    return d[0]


def _fla_config(path):
    c = json.load(open(os.path.join(path, "config.json")))
    hs, nh = c["hidden_size"], c["num_heads"]
    inter = int(hs * c["hidden_ratio"] * 2 / 3)
    inter = 256 * ((inter + 255) // 256)
    return c, LlamaConfig(vocab_size=c["vocab_size"], hidden_size=hs, intermediate_size=inter,
                          num_hidden_layers=c["num_hidden_layers"], num_attention_heads=nh,
                          num_key_value_heads=c["num_kv_heads"] or nh, max_position_embeddings=c["max_position_embeddings"],
                          rms_norm_eps=c["norm_eps"], rope_theta=c["rope_theta"], tie_word_embeddings=False,
                          attention_bias=False, mlp_bias=False, hidden_act="silu",
                          bos_token_id=c["bos_token_id"], eos_token_id=c["eos_token_id"])


def _fla_state(path):
    sd = load_file(os.path.join(path, "model.safetensors"))
    out = {}
    for k, v in sd.items():
        k2 = (k.replace("model.embeddings.", "model.embed_tokens.").replace(".attn.", ".self_attn.")
              .replace(".attn_norm.", ".input_layernorm.").replace(".mlp_norm.", ".post_attention_layernorm."))
        out[k2] = v
    return out


def load(name, impl="probe"):
    """impl: 'eager' or 'probe' (softmax). For softpick-340m, 'probe' means chunked softpick and 'eager' means
    unchunked softpick."""
    if name in HF_MODELS:
        p = snapshot(HF_MODELS[name])
        tok = AutoTokenizer.from_pretrained(p)
        m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, attn_implementation=impl)
        return m.eval(), tok
    p = snapshot(FLA_MODELS[name])
    raw, cfg = _fla_config(p)
    if name == "softpick-340m":
        assert raw["attn_impl"] == "parallel_softpick_attn"
        impl = {"probe": "probe_softpick", "eager": "softpick_eager"}[impl]
    else:
        assert raw["attn_impl"] == "parallel_attn"
    cfg._attn_implementation = impl
    m = LlamaForCausalLM(cfg)
    missing, unexpected = m.load_state_dict(_fla_state(p), strict=False)
    missing = [k for k in missing if "rotary" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    tok = AutoTokenizer.from_pretrained(p)
    return m.float().eval(), tok


@torch.no_grad()
def fla_reference_logits(name, ids):
    """Independent fla-faithful forward (pure torch, written from fla/models/transformer and fla/layers/attn.py)."""
    p = snapshot(FLA_MODELS[name])
    c = json.load(open(os.path.join(p, "config.json")))
    sd = load_file(os.path.join(p, "model.safetensors"))
    H, d = c["num_heads"], c["hidden_size"] // c["num_heads"]
    T = ids.shape[1]
    inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    fr = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    cos, sin = fr.cos().repeat(1, 2), fr.sin().repeat(1, 2)

    def rope(x):  # x: (T, H, d), GPT-NeoX (non-interleaved) rotation as in fla rotary_embedding_ref
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return x * cos[:, None] + torch.cat([-x2, x1], -1) * sin[:, None]

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c["norm_eps"]) * w

    x = sd["model.embeddings.weight"][ids[0]]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for i in range(c["num_hidden_layers"]):
        P = f"model.layers.{i}."
        h = rms(x, sd[P + "attn_norm.weight"])
        q = (h @ sd[P + "attn.q_proj.weight"].T).view(T, H, d)
        k = (h @ sd[P + "attn.k_proj.weight"].T).view(T, H, d)
        v = (h @ sd[P + "attn.v_proj.weight"].T).view(T, H, d)
        q, k = rope(q), rope(k)
        s = torch.einsum("thd,shd->hts", q, k) * d ** -0.5
        if c["attn_impl"] == "parallel_softpick_attn":
            pr = probe_attention.softpick(s, causal[None])
        else:
            pr = torch.softmax(s.masked_fill(~causal[None], -float("inf")), -1)
        o = torch.einsum("hts,shd->thd", pr, v).reshape(T, H * d) @ sd[P + "attn.o_proj.weight"].T
        x = x + o
        h = rms(x, sd[P + "mlp_norm.weight"])
        g, u = h @ sd[P + "mlp.gate_proj.weight"].T, h @ sd[P + "mlp.up_proj.weight"].T
        x = x + (F.silu(g) * u) @ sd[P + "mlp.down_proj.weight"].T
    x = rms(x, sd["model.norm.weight"])
    return (x @ sd["lm_head.weight"].T)[None]
