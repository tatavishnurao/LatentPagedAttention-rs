# Changelog

## v0.1.0 - 2026-07-14

- Added NumPy, Rust, and cuTile validation for paged latent-cache decode attention.
- Added direct paged latent-space GQA with no logical latent or reconstructed K/V device tensors.
- Added paged latent-cache write-to-attention validation.
- Added FP16 latent-cache storage with FP32 attention arithmetic.
- Added runtime active sequence lengths and partial-final-block masking for the tiny profile.
- Added a synthetic `model_small` GPU profile.
- Added an FP16 full-KV paged baseline with FP32 arithmetic.
- Added reproducibility, architecture, limitations, final report, and release checklist documents.
