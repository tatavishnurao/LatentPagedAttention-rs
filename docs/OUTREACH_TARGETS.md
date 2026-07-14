# Outreach Targets

This table uses public project pages, public GitHub profiles, public X/LinkedIn
profiles when available, or official project channels. Do not add private contact
data to this repository.

| name/team | organization/project | public profile | why relevant | recommended channel | priority |
|---|---|---|---|---|---|
| TensorRT-LLM maintainers | NVIDIA TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM | NVIDIA's open LLM inference stack includes paged attention, custom attention kernels, and FP16/FP8 inference work. | GitHub Discussion or issue only if framed as technical feedback | High |
| Juney NVIDIA | NVIDIA TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM/discussions/3987 | Public TensorRT-LLM maintainer discussing open-sourced attention kernels. | GitHub Discussion reply or public GitHub profile | High |
| vLLM maintainers | vLLM | https://github.com/vllm-project/vllm | vLLM introduced PagedAttention and focuses on KV-cache memory management. | GitHub Discussion | High |
| Woosuk Kwon | vLLM / UC Berkeley | https://github.com/WoosukKwon | PagedAttention/vLLM author; directly relevant to paged cache design. | Public GitHub/X/LinkedIn if available | High |
| Zhuohan Li | vLLM | https://vllm.ai/blog/2023-06-20-vllm | vLLM/PagedAttention co-author; relevant to paged KV memory and serving trade-offs. | Public academic or project channel | High |
| Ying Sheng | vLLM / LMSYS | https://arxiv.org/abs/2309.06180 | PagedAttention co-author and serving-systems researcher. | Public academic/project channel | High |
| FlashInfer maintainers | FlashInfer | https://github.com/flashinfer-ai/flashinfer | FlashInfer is an inference kernel library with attention backends and serving integrations. | GitHub Discussion/Issue | High |
| FlashInfer authors/team | FlashInfer | https://arxiv.org/pdf/2501.01005 | Published work on customizable attention engines and kernel-level serving performance. | Public project channel | High |
| SGLang maintainers | SGLang | https://github.com/sgl-project/sglang | High-performance serving framework using advanced cache and inference scheduling techniques. | GitHub Discussion | Medium |
| LMSYS / Mini-SGLang team | SGLang / Mini-SGLang | https://github.com/sgl-project/mini-sglang | Educational serving implementation; useful audience for correctness-first prototypes. | GitHub Discussion | Medium |
| Punica authors/team | Punica | https://github.com/punica-ai/punica | GPU serving work focused on memory-efficient multi-tenant inference and custom CUDA kernels. | GitHub project channel | Medium |
| Zihao Ye | FlashInfer / Punica | https://arxiv.org/abs/2310.18547 | Public author on Punica and FlashInfer-related systems; relevant to serving kernels. | Public academic/project channel | High |
| Yongji Wu | Punica | https://www.yongjiwu.me/assets/pdf/mlsys24-punica.pdf | Punica author; relevant to memory-efficient serving and custom kernels. | Public academic site/channel | Medium |
| llama.cpp maintainers | ggml-org/llama.cpp | https://github.com/ggml-org/llama.cpp | Low-resource inference and GPU backend community; relevant to RTX 4060 reproducibility. | GitHub Discussion | Medium |
| Rust GPU community | Rust GPU ecosystem | https://github.com/Rust-GPU/Rust-CUDA | Rust + GPU systems audience; relevant to Rust-side correctness and GPU integration. | GitHub/community channel | Medium |
| NVIDIA Developer Forums | CUDA / inference kernels | https://forums.developer.nvidia.com/ | Appropriate place for cuTile/CUDA kernel-design feedback. | Public forum post | Medium |

Sources consulted include public GitHub repositories and project pages for
TensorRT-LLM, vLLM, FlashInfer, SGLang, Punica, and llama.cpp, plus public
papers/blogs linked from those projects.
