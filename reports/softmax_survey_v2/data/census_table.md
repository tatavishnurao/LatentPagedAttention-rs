| Family | Date | Attention | Context | Vocab | Experts | Router | Softmax modifications | Sources |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | 2019 | MHA | 1,024 | 50,257 | - | - | none found | cfg:openai-community/gpt2 |
| GPT-3 | 2020-05 | MHA; alternating dense / locally banded sparse | 2,048 | unknown | - | - | local / sliding-window layers | paper:2005.14165 |
| Llama 1 | 2023-02 | MHA | 2,048 | unknown | - | - | none found | paper:2302.13971; paper:2307.09288 (Table 1) |
| Llama 2 | 2023-07 | MHA (7B,13B); GQA (34B,70B) | 4,096 | 32,000 | - | - | none found | paper:2307.09288 |
| Llama 3.1 | 2024-07 | GQA (8 KV heads) | 131,072 | 128,256 | - | - | none found | cfg:meta-llama/Llama-3.1-8B; paper:2407.21783 |
| Mistral 7B | 2023-10 | GQA (8 KV) + sliding window 4096 | 32,768 | 32,000 | - | - | local / sliding-window layers | cfg:mistralai/Mistral-7B-v0.1; paper:2310.06825 |
| Qwen2 / Qwen2.5 | 2024-07 | GQA; QKV bias | 131,072 | 152,064 | - | - | length/temperature logit scaling | cfg:Qwen/Qwen2-7B; cfg:Qwen/Qwen2.5-7B; paper:2407.10671 |
| Qwen3 (dense) | 2025-05 | GQA; QK-norm; no QKV bias | 40,960 | 151,936 | - | - | QK-norm, length/temperature logit scaling | cfg:Qwen/Qwen3-8B; code:modeling_qwen3; paper:2505.09388; card:Qwen/Qwen3-8B |
| Gemma 1 | 2024-03 | MHA (7B); MQA (2B) | 8,192 | 256,000 | - | - | none found | cfg:google/gemma-7b; paper:2403.08295 |
| Gemma 2 | 2024-07 | GQA; local 4096 / global alternating | 8,192 | 256,000 | - | - | logit soft-capping, local / sliding-window layers | cfg:google/gemma-2-9b; code:modeling_gemma2; paper:2408.00118 |
| Gemma 3 | 2025-03 | GQA; 5 local (1024) : 1 global; QK-norm | 131,072 | 262,144 | - | - | QK-norm, local / sliding-window layers | cfg:google/gemma-3-1b-it; cfg:google/gemma-3-27b-it; code:modeling_gemma3; paper:2503.19786 |
| Phi-3-mini / Phi-4 | 2024-04 | MHA (Phi-3-mini); GQA 40/10 (Phi-4) | 16,384 | 100,352 | - | - | none found (unknown: local / sliding-window layers) | cfg:microsoft/Phi-3-mini-4k-instruct; cfg:microsoft/phi-4; paper:2404.14219 |
| OLMo 2 | 2024-12 | MHA; QK-norm | 4,096 | 100,352 | - | - | QK-norm, z-loss | cfg:allenai/OLMo-2-1124-7B; code:modeling_olmo2; paper:2501.00656 |
| Falcon | 2023-11 | MQA (7B); multigroup (40B,180B) | 2,048 | 65,024 | - | - | none found | cfg:tiiuae/falcon-7b; paper:2311.16867 |
| Command R | 2024-03 | not verified (config gated) | 131,072 | unknown | - | - | vocab logit multiplier (unknown: QK-norm) | card:CohereLabs/c4ai-command-r-v01; code:modeling_cohere |
| GLM-4.5 | 2025-08 | GQA 96 q / 8 KV; partial RoPE; QK-norm | 131,072 | 151,552 | 160 (+1 shared), top-8 | sigmoid + loss-free balance bias | QK-norm, sigmoid MoE router | cfg:zai-org/GLM-4.5; code:modeling_glm4_moe; paper:2508.06471 |
| gpt-oss | 2025-08 | GQA 64 q / 8 KV; alternating banded window 128 / dense | 131,072 | 201,088 | 128 (120b) / 32 (20b), top-4 | softmax over the selected top-4 | length/temperature logit scaling, learned sink logit, local / sliding-window layers | cfg:openai/gpt-oss-20b; cfg:openai/gpt-oss-120b; code:modeling_gpt_oss; paper:2508.10925 |
| Mixtral 8x7B | 2024-01 | GQA (8 KV) | 32,768 | 32,000 | 8, top-2 | softmax over top-2 logits | none found | cfg:mistralai/Mixtral-8x7B-v0.1; code:modeling_mixtral; paper:2401.04088 |
| DeepSeek-V2 | 2024-05 | MLA (kv_lora 512) | 163,840 | 102,400 | 160 (+2 shared), top-6 | softmax (scoring_func), group-limited greedy | length/temperature logit scaling | cfg:deepseek-ai/DeepSeek-V2; paper:2405.04434 |
| DeepSeek-V3 | 2024-12 | MLA (kv_lora 512) | 163,840 | 129,280 | 256 (+1 shared), top-8 | sigmoid + bias (noaux_tc), normalised over selected | length/temperature logit scaling, sigmoid MoE router | cfg:deepseek-ai/DeepSeek-V3; code:modeling_deepseek_v3; paper:2412.19437 |
| Qwen3 MoE | 2025-05 | GQA (4 KV); QK-norm | 40,960 | 151,936 | 128, top-8, no shared | softmax, renormalised top-k | QK-norm | cfg:Qwen/Qwen3-30B-A3B; code:modeling_qwen3_moe; paper:2505.09388 |
| Llama 4 (Scout/Maverick) | 2025-04 | chunked local (8192) RoPE layers + global NoPE layers every 4th | 10M (Scout) / 1M (Maverick) | 202,048 | 16 / 128 (+1 shared), top-1 | sigmoid of top-1 score | QK-norm, length/temperature logit scaling, local / sliding-window layers, sigmoid MoE router | code:llama-models/models/llama4 @0e0b8c5; code:modeling_llama4; card:llama-models MODEL_CARD |
| Kimi K2 | 2025-07 | MLA, 64 heads | 131,072 | 163,840 | 384 (+1 shared), top-8 | sigmoid + bias (noaux_tc) | length/temperature logit scaling, sigmoid MoE router | cfg:moonshotai/Kimi-K2-Instruct; paper:2507.20534 |
| Mamba | 2023-12 | none (selective SSM) | unknown | 50,280 | - | - | linear-attention or SSM layers | cfg:state-spaces/mamba-2.8b-hf; paper:2312.00752 |
| Mamba-2 | 2024-05 | none (SSD) | unknown | 50,277 | - | - | linear-attention or SSM layers | cfg:state-spaces/mamba2-2.7b; paper:2405.21060 |
| RWKV-6 (Finch) | 2024-04 | none (matrix-valued recurrence) | unknown | 65,536 | - | - | linear-attention or SSM layers | cfg:RWKV/v6-Finch-1B6-HF; paper:2404.05892 |
| Jamba / Jamba-1.5 | 2024-03 | 1 attention : 7 Mamba layers; no positional encoding | 262,144 | 65,536 | 16, top-2, every 2nd layer | softmax | linear-attention or SSM layers | paper:2403.19887; paper:2408.12570; card:ai21labs/AI21-Jamba-1.5-Mini; code:configuration_jamba/modeling_jamba |
| MiniMax-Text-01 | 2025-01 | 7 lightning (linear) : 1 softmax attention, GQA 8 KV | 1M train / 4M inference | 200,064 | 32, top-2 | softmax | attention output gate, linear-attention or SSM layers | cfg:MiniMaxAI/MiniMax-Text-01; code:modeling_minimax; paper:2501.08313 |
| Qwen3-Next | 2025-09 | 3 Gated DeltaNet : 1 gated softmax attention; QK-norm | 262,144 | 151,936 | 512 (+1 shared), top-10 | softmax, renormalised top-k | QK-norm, attention output gate, linear-attention or SSM layers | cfg:Qwen/Qwen3-Next-80B-A3B-Instruct; code:modeling_qwen3_next; paper:2505.06708 (gating method) |
| RecurrentGemma (Griffin) | 2024-04 | 2 RG-LRU : 1 local MQA attention (window 2048) | unknown | 256,000 | - | - | logit soft-capping, local / sliding-window layers, linear-attention or SSM layers | cfg:google/recurrentgemma-2b; paper:2404.07839; paper:2402.19427 |
| BERT | 2018-10 | bidirectional MHA | 512 | 30,522 | - | - | none found | cfg:google-bert/bert-base-uncased; paper:1810.04805 |
| T5 | 2019-10 | encoder-decoder MHA; relative position bias | 512 | 32,128 | - | - | none found | cfg:google-t5/t5-base; paper:1910.10683 |
| ViT | 2020-10 | bidirectional MHA over patches | 197 tokens (224px/16) | - | - | - | none found | cfg:google/vit-base-patch16-224; paper:2010.11929 |
| CLIP | 2021-02 | ViT image encoder + causal text encoder | 77 | 49,408 | - | - | none found | cfg:openai/clip-vit-base-patch32; paper:2103.00020 |
| SigLIP | 2023-03 | ViT image encoder + text encoder | unknown | 32,000 | - | - | sigmoid (non-softmax) loss | cfg:google/siglip-base-patch16-224; paper:2303.15343 |
| Whisper | 2022-12 | encoder-decoder MHA | 30 s audio / 448 text tokens | 51,866 | - | - | none found | cfg:openai/whisper-large-v3; paper:2212.04356 |
