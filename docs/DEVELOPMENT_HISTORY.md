# Development History

The public result has three layers: the paged latent path, the FP16 full-KV
baseline, and Python/Rust/GPU validation. The numbered documents below record
the implementation history and remain useful when isolating regressions, but
they are not separate product claims or release milestones.

## Foundation and references

- [Research question](00_research_question.md)
- [Literature review](01_literature_review.md)
- [Memory model](02_memory_model.md)
- [Attention variants](03_attention_variants.md)
- [Benchmark methodology](04_benchmark_methodology.md)
- [No-fake-claims policy](05_no_fake_claims_policy.md)
- [Reference benchmark report](06_reference_benchmark_report.md)

## GPU implementation history

- [RTX 4060 baseline](08_rtx4060_gpu_baseline.md)
- [cuTile smoke test and paged lookup](09_cutile_smoke_and_gpu_paged_lookup.md)
- [Paged K/V write](10_gpu_paged_kv_write.md)
- [Contiguous GQA](11_gpu_contiguous_gqa_decode.md)
- [Paged GQA](12_gpu_paged_gqa_decode.md)
- [Latent-KV reconstruction](13_gpu_latent_kv_reconstruction.md)
- [Direct latent GQA](14_gpu_direct_latent_gqa.md)
- [Direct paged latent GQA](15_gpu_direct_paged_latent_gqa.md)
- [Paged latent write-to-attention](16_gpu_paged_latent_write_attention.md)
- [FP16 paged latent storage](17_gpu_fp16_paged_latent_storage.md)

Use [Reproducibility](REPRODUCIBILITY.md) for the current commands and
[Final Report](FINAL_REPORT.md) for the release evidence.
