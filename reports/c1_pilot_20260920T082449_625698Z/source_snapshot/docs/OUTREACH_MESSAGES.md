# Outreach Messages

Personalize these messages using the recipient's public work. Do not send them
as bulk messages.

## Template 1 - Kernel Engineer

Hi <name>, I saw your work on <specific kernel/project>. I recently released LatentPagedAttention-rs, a correctness-first Rust/cuTile prototype for paged latent-cache decode attention on an RTX 4060. The interesting trade-off is explicit: for a synthetic model-shaped profile it stores 16x fewer persistent FP16 cache bytes than an FP16 full-KV paged baseline, but the current latent read path is about 32.6% slower under synchronized host end-to-end timing. I would value feedback on the kernel design and what you would profile first, especially around direct latent-space scoring, masked softmax, and block-table addressing. Repo: <link>

## Template 2 - Open-Source Maintainer

Hi <name>, I’ve been following <project> because of its work on <specific feature>. I released a small related research prototype: LatentPagedAttention-rs, which combines paged physical cache addressing with direct latent-space GQA and FP16 cache storage. It is not a serving runtime, but it has Python/Rust/cuTile parity checks and an FP16 full-KV baseline. The result is a 16x persistent-cache-byte reduction with a measured latency trade-off. I’d appreciate criticism on the validation methodology or whether the abstractions map cleanly to real serving systems. Repo: <link>

## Template 3 - Hiring Manager or Recruiter

Hi <name>, I’m sharing a recent project that may be relevant to your GPU inference work at <company/team>. LatentPagedAttention-rs is a tagged open-source Rust, Python, and cuTile research release for paged latent-cache decode attention on an RTX 4060. I implemented the reference chain, CPU/GPU parity validation, runtime sequence masking, FP16 cache writes, and an FP16 full-KV baseline. The measured result is a 16x persistent-cache-byte reduction for a synthetic model-shaped profile, with about 32.6% higher synchronized host end-to-end latent-read time. I’ve documented the limitations carefully; it is not a production-serving claim. Repo/release: <link>
