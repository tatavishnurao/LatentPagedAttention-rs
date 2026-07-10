# cuTile Smoke Test and GPU Paged Lookup

## Hardware

The validation target is an NVIDIA GeForce RTX 4060 Laptop GPU with 8,188 MiB
reported VRAM and compute capability 8.9. The local validation used driver
610.43.02 and CUDA toolkit 13.3 through `/opt/cuda`.

## Dependency policy

cuTile is pinned to the exact released version `0.2.0`. GPU functionality is
behind the `gpu-cutile` Cargo feature, so normal workspace tests remain GPU
independent.

## Smoke test

The vector-add example uploads deterministic `f32` vectors containing ones and
twos, runs a cuTile kernel, synchronizes, reads the result back, and validates
that every value is three. It proves that the released cuTile Rust stack can
JIT-compile and execute a small kernel on the RTX 4060. It does not measure
kernel latency or throughput.

## Paged lookup

The first project-specific kernel supports `seq_len=5`, `block_size=2`, and
`width=4`. Physical storage is flattened from
`[num_physical_blocks, block_size, width]` and the block table maps logical
blocks `[0, 1, 2]` to physical blocks `[2, 0, 1]`. The GPU performs this
remapping while loading each output tile. The block-table input is padded to
four entries because cuTile 0.2.0 requires power-of-two tile dimensions; the
padded entry is not used.

## Correctness chain

Python fixture -> Rust CPU reference -> cuTile GPU kernel -> host comparison.

The GPU example checks both Python and Rust outputs with `atol=1e-6` and
`rtol=0`.

The next GPU primitive is the single-token paged KV write described in
`docs/10_gpu_paged_kv_write.md`.

## Commands

```bash
source scripts/cutile_env.sh
bash scripts/run_cutile_smoke.sh
bash scripts/run_gpu_paged_lookup.sh
bash scripts/run_gpu_paged_lookup.sh
```

The second run checks that the JIT path also succeeds after the first run. Any
cache reuse is an observation about the local environment, not a performance
result.

## What this proves

- A cuTile kernel can JIT-compile and execute on the RTX 4060.
- The GPU performs a non-identity paged-cache lookup correctly.
- GPU output matches the Python fixture and Rust CPU reference.

## What this does not prove

- Attention performance
- GQA correctness
- Latent-KV correctness
- Efficient production paging
- Kernel speed
- Model quality
- End-to-end inference
