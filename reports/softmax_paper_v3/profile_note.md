# How this paper fits a Fall 2028 ML-systems application (1 page)

*For Tata Vishnu Vardhan Rao. Not part of the paper. Checked 2026-11-04. This note cannot predict admission; it only says what the work shows and how to describe it accurately.*

## What the paper shows a reader

- **That you check published claims before building on them.** Four claims from four papers (three with Google DeepMind authors) are each stated, tested and given a verdict. Two parts did not hold as published: the fp32 collapse cliff for more than two classes, and the adaptive-temperature polynomial at long context. One review hypothesis about the bf16 cancellation floor was also tested and found not to hold.
- **That you can measure real models carefully.** Exact attention probes on five open models were validated bit for bit against the reference implementation before any number was used. You wrote a loader for a non-standard checkpoint and checked it with a sensitivity test.
- **That you read serving code.** You found where vLLM and SGLang compute log-probabilities and which SGLang mode uses bfloat16.

## Labs it speaks to (checked against their own papers, read in full this session)

| Lab | Their paper (read) | Connection |
|---|---|---|
| Junchen Jiang, UChicago (LMCache) | LMCache, arXiv 2510.09665 | KV-cache reuse and eviction; the sink results bear on what an eviction or offload policy must keep (B5) |
| Alexey Tumanov, Georgia Tech | Etalon 2407.07000; Vidur 2405.05465; Revati 2601.00397 | Same genre: "a standard metric or tool reports something misleading" |
| Hao Zhang, UCSD | DistServe 2401.09670 (co-author) | Serving measurement; goodput and SLO framing |
| Mosharaf Chowdhury, Michigan | ML.ENERGY Benchmark 2505.06371 | Benchmark-methodology papers with explicit design principles and limitations |
| Fan Lai, UIUC | not read this session | Check JITServe (2504.20068) before naming it |
| Waterloo and UofT systems groups | **not checked this session** | Do not name a PI until you have read one of their recent papers |

## One research line, three artifacts

**Measurement validity in LLM serving:**
1. **tailbench:** does the load generator report the latency the server produced?
2. **This paper:** does the softmax computation (log-probs, collapse, sinks under eviction) behave the way published claims and default code paths assume?
3. **The KV-cache flagship** from your strategy files: does the measured benefit of KV reuse survive correct measurement?

Present them in that order in the SOP: a tool, a reproduction study, then a measurement study that uses both.

## Wording for the CV and SOP (exact, not inflated)

- CV, before submission: *"Softmax numerics and attention sinks in LLM serving: a reproduction and measurement study." Preprint / technical report, 2026. Code and data public.* Do not write "under review" until it is submitted, or "accepted" until it is.
- SOP sentence: *"I re-tested four published claims about softmax on CPU and in five small open models; two held as stated, one held only for two classes, and one switched off at the context lengths where it was meant to help. I traced where vLLM and SGLang compute returned log-probabilities and found that SGLang's RL on-policy mode computes them in bfloat16."* Adjust it to the final results; every number must match the paper.

## Where to submit (verified 2026-11-04)

- **TMLR** (rolling submissions; jmlr.org/tmlr). The ML Reproducibility Challenge takes TMLR-accepted reproducibility papers. Its 2026 edition was a NeurIPS 2026 track, with eligibility ending 30 Sep 2026 (reproml.org/call_for_papers). Check reproml.org for a 2027 call before planning around it.
- **ICLR 2027 workshops** (ICLR 2027 is 26–30 Apr 2027; workshop calls are not posted yet). Pick a workshop on efficient inference, numerics or reproducibility once the list is out.
- **MLSys 2027:** the main-track deadline (30 Oct 2026) has passed.

## What would make it stronger before December 2027

1. Run B4–B6 on a GPU (scripts in `code/gpu/`). The serving and paged-KV results are what connect it to the labs above.
2. Get one external reader, for example a maintainer comment on an SGLang issue about the RL-mode log-probs, or a student in a target lab.
3. A letter from someone who watched you do this work. A paper by itself does not replace that.
