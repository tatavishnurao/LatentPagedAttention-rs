"""Validation before any probe number is used.
1. probe attention (chunked) vs transformers eager attention: max |logit difference| at 300 tokens, chunk 64
   (so several chunks are exercised).
2. fla 340M pair: mapped LlamaForCausalLM vs an independent fla-faithful forward: max |logit difference|.
3. Perplexity of every model on 1,024 tokens of held-out public-domain text (Frankenstein, Project Gutenberg #84),
   eager attention, to confirm the weights load into a working model."""
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
import models
import probe_attention as pa
from corpus import load_text

torch.manual_seed(0)
torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
res = {}
text = load_text("pg84")
for name in list(models.HF_MODELS) + list(models.FLA_MODELS):
    r = {}
    m_e, tok = models.load(name, "eager")
    ids = torch.tensor(tok(text)["input_ids"][:1025])[None]
    with torch.no_grad():
        lo = m_e(ids[:, :1024], use_cache=False).logits.double()
    nll = F.cross_entropy(lo[0], ids[0, 1:1025])
    r["ppl_1024_eager"] = float(math.exp(nll))
    with torch.no_grad():
        le = m_e(ids[:, :300], use_cache=False).logits
    del m_e
    m_p, _ = models.load(name, "probe")
    pa.STATE["chunk"] = 64
    with torch.no_grad():
        lp = m_p(ids[:, :300], use_cache=False).logits
    pa.STATE["chunk"] = 256
    r["max_abs_logit_diff_probe_vs_eager"] = float((lp - le).abs().max())
    r["logit_scale_max_abs"] = float(le.abs().max())
    if name in models.FLA_MODELS:
        lr = models.fla_reference_logits(name, ids[:, :300])
        r["max_abs_logit_diff_mapped_vs_fla_reference"] = float((le - lr).abs().max())
    r["bos_prepended"] = bool(ids[0, 0].item() == tok.bos_token_id) if tok.bos_token_id is not None else False
    r["first_token"] = tok.convert_ids_to_tokens(int(ids[0, 0]))
    res[name] = r
    print(name, json.dumps(r), flush=True)
    del m_p
json.dump(res, open(os.path.join(OUT, "validate_models.json"), "w"), indent=1)
