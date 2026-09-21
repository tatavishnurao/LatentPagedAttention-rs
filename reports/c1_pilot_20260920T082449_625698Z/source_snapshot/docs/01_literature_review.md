# Literature Review

This is a living document. It is meant to track implementation-relevant ideas and open questions, not to pretend the review is complete.

## PagedAttention / vLLM

PagedAttention established the practical value of paging KV cache storage so decode workloads can allocate memory in block units instead of one contiguous sequence buffer per request. The main win is allocator efficiency and better handling of dynamic batching. Paging does not by itself reduce KV bytes per token.

## DeepSeek MLA

DeepSeek's MLA direction is relevant because it compresses or restructures KV state into a latent representation with a cheaper cache footprint than storing full K and V per token. The exact reconstruction and projection path matters. This repo treats MLA-style latent KV as a study target, not as a drop-in assumption for arbitrary models.

## KIVI

KIVI is relevant as a KV-cache quantization reference point. It shows the general direction for shrinking cache size with low-bit representations, while raising questions about decode-time error accumulation, dequantization cost, and the quality tradeoff under long contexts.

## MHA2MLA / TransMLA

These lines of work are relevant because they explore ways to transform or adapt existing attention/state representations toward MLA-like forms. They matter for compatibility questions: a latent-KV runtime path may require model-side adaptation rather than simple inference-time substitution.

## Grout / cutile-rs

The future kernel plan here is Rust-first where possible, with cuTile-related work as a later dedicated pass if the environment and bindings are stable enough. This repo does not claim that Rust/cuTile integration is already validated. The goal is to reach that stage only after the reference models and measurement harness are solid.
