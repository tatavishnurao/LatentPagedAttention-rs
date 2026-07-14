# Architecture

```mermaid
flowchart LR
    token[logical token position] --> table[runtime block table]
    table --> phys[physical latent block]
    write[optional latent write] --> phys
    phys --> load[FP16 latent load]
    load --> cast[FP16 to FP32]
    q[FP32 query] --> projq[FP32 projected query]
    kproj[FP32 K projection] --> projq
    cast --> scores[paged latent scores]
    projq --> scores
    scores --> mask[runtime active-length mask]
    mask --> softmax[masked stable softmax]
    softmax --> lctx[paged latent context]
    cast --> lctx
    vproj[FP32 V projection] --> out[FP32 output projection]
    lctx --> out
```

The direct latent path stores physical latent-cache blocks and resolves logical
tokens through a runtime block table. It does not persist a logical latent cache
or reconstructed K/V tensors on the GPU.
