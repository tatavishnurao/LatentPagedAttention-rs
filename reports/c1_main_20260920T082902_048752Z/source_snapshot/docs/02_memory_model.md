# Memory Model

## Standard or GQA KV bytes per token per layer

For a standard KV cache, bytes per token per layer are:

`n_kv_heads * head_dim * dtype_bytes * 2`

The factor of 2 is for storing both K and V.

In GQA, the number of KV heads is smaller than the number of query heads, so this formula uses `n_kv_heads`, not `n_q_heads`.

## Latent KV bytes per token per layer

For a latent representation, bytes per token per layer are:

`latent_dim * dtype_bytes`

This is the memory-side attraction of latent KV: bytes per token may be materially smaller than full K and V storage.

## Total KV cache memory

Total KV cache size is:

`num_layers * seq_len * batch_size * bytes_per_token_per_layer`

This is a simplification. Real systems also pay for allocator overhead, block metadata, alignment, internal fragmentation, and temporary workspace.

## Why paging helps

Paging improves allocation behavior. It can reduce wasted memory caused by large contiguous reservations and make dynamic growth easier to manage. Paging does not reduce the fundamental bytes per token formula by itself.

## Why latent KV helps

Latent KV can directly reduce bytes per token, which is the main path to longer context or higher concurrency under a fixed VRAM budget. The tradeoff is extra decode work: the runtime may need to reconstruct or project usable K and V views from the latent cache.
