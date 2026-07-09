# Attention Variants

## MHA

Multi-Head Attention stores and attends over full K and V state for each head.

## GQA

Grouped-Query Attention shares KV heads across multiple query heads. It keeps full K and V semantics, but reduces KV cache size relative to MHA by lowering the number of KV heads.

## Paged GQA

Paged GQA stores the same GQA KV information in fixed-size blocks rather than per-request contiguous slabs. The attention math is still GQA. The storage layout changes.

## Latent-KV Attention

Latent-KV attention stores a compressed or projected latent state per token instead of full K and V tensors. Decode requires a path from the latent state back to attention-usable representations.

## Paged Latent-KV Attention

Paged Latent-KV attention combines fixed-size block allocation with a latent per-token cache representation. This is one of the main combinations this repo intends to study.

## Quantized Paged Latent-KV

This adds low-bit storage on top of paged latent KV. It may further reduce bytes per token, but it increases numerical and implementation risk. It should be evaluated with explicit drift reporting.
