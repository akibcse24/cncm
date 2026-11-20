---
license: mit
tags:
  - cncm
  - moe
  - causal-lm
  - code
  - reasoning
---

# ctx0an (CNCM-Base-MoE)

Private model repository for **`ctx0an`** — a Compact Causal-Hybrid Next-Concept Model with DeepSeek-style Fine-Grained Mixture-of-Experts (`base_moe`).

## Architecture Summary
- **Total Parameters**: ~221.45M (`215.51M` Stage 2 generator + `5.93M` Stage 1 M-RFSQ Concept Tokenizer)
- **Active Parameters / Token**: ~94.05M (`88.11M` active generator + `5.93M` Stage 1)
- **Backbone**: 12 unique Transformer layers × 2 recursive loops = **24 effective layers**, `hidden_size=768`, GQA (12Q / 3KV), rank-32 per-step `LoopAdapter`s
- **MoE FFN**: 1 always-on Shared Expert (`d=768`) + 8 Routed Experts (`d=768`, Top-2 active, total active width `2304`) with **24 per-loop routers** + Switch/DeepSeek load-balancing loss (`aux=0.01`)
- **Stage 1 Tokenizer**: Strictly causal 1-layer semantic encoder (`enc_layers=1`), M-RFSQ `(8,8,8,8)` (4096 implicit codes), Matryoshka cutoffs `(4, 6, 7, 8)`
- **Text Tokenizer**: Corpus-trained Byte-Level BPE (`V=6400`, `Digits(individual_digits=True)`) preserving 16 core + 11 PG-FIM special tokens
- **Dual Decoding**: Parallel Concept Mode (8 passes / 64 tokens) + Exact-AR Mode + PG-FIM Code Infilling
- **Estimated Edge Size (INT4)**: ~115 MB

## Training Curriculum (Multi-Session Resumable, Kaggle GPU T4 ×2 DDP)
- **Curated Data Mix**:
  - Educational Textbooks: `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`, `fineweb-edu-dedup`)
  - Python & Code: `Nan-Do/code-search-net-python`, `TokenBender/code_instructions_122k_alpaca_style`, `flytech/python-codes-25k`, PG-FIM spans
  - Math & `<think>` Reasoning: `HuggingFaceTB/finemath` (`finemath-4plus`), `meta-math/MetaMathQA`, `openai/gsm8k`
  - Function Calling & Tool JSON: `glaiveai/glaive-function-calling-v2` + strict JSON schemas
- **Phases**: Stage 1 FP32 → Stage 2 Phase 1 Exact-AR Foundation → Stage 2 Phase 2 Joint Concept + `L_align` → Stage 2 Phase 3 SFT Annealing.
