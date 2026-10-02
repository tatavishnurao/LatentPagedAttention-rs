# C2 design disposition: no duplicate kernel

## Hypothesis

The proposed C2 algorithm is:

```text
q' = q Wk^T
scores[t] = q' dot z[t]
p = softmax(scores)
z_context = sum_t p[t] z[t]
output = z_context Wv
```

## Source verification

This algorithm is mathematically valid by associativity and linearity for the repository's synthetic `K=ZWk`, `V=ZWv` formulation.

It is also already implemented:

- `crates/plkv-kernels/src/cutile/c1.rs`: `model_small_project_query_once` computes `q'` once/head and the C1 score kernel dots it with FP16 latent rows after FP32 conversion.
- `crates/plkv-kernels/src/cutile/p15b_rtable_kernels.rs`: `model_small_context_*` accumulates `latent_context` across all logical blocks, then loads `v_projection` and projects once.
- `crates/plkv-kernels/examples/c1_rtable.rs`: C1 launches projection, latent score, shared softmax, and the latent-context/value-projection kernel.

No per-token full-dimensional V reconstruction occurs in C1. No persistent reconstructed K/V buffer is allocated.

## Decision

A separate C2 kernel was **not** created because it would be source-equivalent to C1 and would create a misleading duplicate ablation. For Phase 2, “C2 / latent-space accumulation” is therefore recorded as **ALREADY PRESENT IN C1**.

The next genuinely distinct algorithm should target sequence-parallel context reduction (tentative `C2_SPLIT_CONTEXT`), not repeat the same algebra. Such a kernel must preserve the current C1 result and be added rather than overwriting historical kernels.

## Complexity

Current C1/minimum-operation latent arithmetic is approximately:

```text
query projection:  2 Hq L D
score:             2 Hq N L
latent value sum:  2 Hq N L
final V projection:2 Hq L D
```

Total: `4 Hq L (N + D)` plus softmax. A per-token V reconstruction would instead add approximately `2 Hq N L D`, which the current direct path avoids.
