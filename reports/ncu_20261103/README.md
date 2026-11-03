# Nsight Compute at Step B8·N8192: not run (2026-11-03)

**`ncu` could not access the GPU performance counters, so no profile exists:** `ncu` 2026.3 from CUDA 13.3 (`/usr/local/cuda-13.3/bin/ncu`) on driver 617.42 under WSL2 stopped on the first profiled kernel with `==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.`

Laptop RTX 4060, WSL2, not a cloud result. As declared in `PLAN.md`, this is recorded and the profile is skipped; no setting was changed to work around it. The same error stopped the 2026-10-02 attempt (`docs/DIAGNOSIS_2026-10-02.md`, E1). Enabling it is a Windows-side setting (NVIDIA Control Panel, Developer, "Allow access to the GPU performance counters to all users", then `wsl --shutdown`), outside this repository.

Probe command (one kernel, base clocks; the harness itself completed, `E4_TIMING_OK=1`, 79 C at the end):

```
ncu --clock-control base --kernel-name regex:c3_reduce --launch-skip 1 --launch-count 1 \
  --section SpeedOfLight -o <scratch>/probe -f target/release/examples/e3_splitk_gpu \
  --suite e4-timing --batch 8 --seq 8192 --iterations 1 --warmup 1 --output-dir <scratch>/run_probe
```

Binary SHA-256: `010db76cb814ceb128210549bc0d75eea991c42fe9de9203aff1fa359b67f9ac` (`target/release/examples/e3_splitk_gpu`).
