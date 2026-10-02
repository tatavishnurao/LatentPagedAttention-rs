# VOID notice (2026-10-02)

The timing samples under `c1_exploratory/` were collected with the GPU at idle SM clocks
(~210 MHz) for N <= 16K and at uncontrolled clocks at 32K. The harness wrote JSON to `/mnt/c`
before and between samples, idling the GPU. Correctness artifacts are unaffected; latencies and
any latent/full ratio derived from them must not be cited. Evidence: `docs/DIAGNOSIS_2026-10-02.md`
(E0) and `reports/diagnosis_20261002/e0/`.
