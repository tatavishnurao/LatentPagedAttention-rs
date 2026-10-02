# C2 correctness disposition

**Status: ALREADY IMPLEMENTED BY C1; no duplicate C2 kernel introduced.**

The C1 pipeline is the proposed minimum-operation latent-space algorithm. Phase-2 extended generated C1 kernels to sequence lengths 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, and 32,768 without changing the algebra.

Each listed length executed the independent `scripts/validate_c1_oracle.py` FP64 explicit reconstructed-K/V oracle on a nonidentity `(17*l+11) mod blocks` page mapping.

| N | score max abs | probability max abs | context max abs | normalization error | pass |
|---:|---:|---:|---:|---:|---|
| 128 | 2.57e-6 | 5.06e-7 | 2.11e-6 | 1.23e-7 | yes |
| 256 | 2.57e-6 | 2.49e-7 | 2.66e-6 | 1.60e-7 | yes |
| 512 | 2.57e-6 | 1.29e-7 | 2.03e-6 | 1.16e-7 | yes |
| 1,024 | 2.75e-6 | 7.29e-8 | 1.74e-6 | 9.37e-8 | yes |
| 2,048 | 2.77e-6 | 3.56e-8 | 1.40e-6 | 6.52e-8 | yes |
| 4,096 | 2.99e-6 | 1.88e-8 | 2.50e-6 | 6.99e-8 | yes |
| 8,192 | 2.99e-6 | 9.24e-9 | 2.15e-6 | 7.04e-8 | yes |
| 16,384 | 2.99e-6 | 4.58e-9 | 2.08e-6 | 8.14e-8 | yes |
| 32,768 | 2.99e-6 | 2.21e-9 | 4.12e-6 | 8.97e-8 | yes |

Raw oracle results are stored beside inputs and samples under `reports/phase2/raw/c1_exploratory/`. The executable also saves `query_refresh.json` and asserts that a changed query changes scores and agrees with the CPU latent reference.

These tests establish correctness for the fixed canonical dimensions and generated lengths. They do not cover the requested latent/head/GQA/block-size parameter matrix.
