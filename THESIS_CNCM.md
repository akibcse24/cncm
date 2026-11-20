# CNCM: A Compact Causal-Hybrid Next-Concept Model for Edge Reasoning, Code Infilling, and Agentic Execution

**Abstract.** Autoregressive (AR) language models generate text one token at a time, making edge inference memory-bandwidth bound on mobile CPUs and laptop iGPUs. We present **CNCM** (*Compact Causal-Hybrid Next-Concept Model*), a unified dual-mode architecture that shares a single recursive-depth Transformer backbone between two complementary decoding heads: (1) a **Factorized Concept Head** that predicts 64-token chunks in parallel across $K=8$ coarse-to-fine residual levels in 8 backbone passes, and (2) a shared **Exact-AR Head** for strict token-by-token generation, JSON-schema grammar masking, and autoregressive code repair. To eliminate codebook collapse and lookup-table memory, CNCM introduces a strictly causal **Multiscale Residual Finite Scalar Quantization (M-RFSQ)** Stage 1 tokenizer trained with Matryoshka multi-cutoff reconstruction ($k \in \{4,6,7,8\}$). To align Concept Mode with Exact-AR geometry, we introduce **Continuous Scaled RoPE** ($\text{pos}_k(i) = P + \lfloor (i+0.5)L/\ell_k \rfloor$) and a **Text-Level Alignment Loss ($\mathcal{L}_{\text{align}}$)**. On held-out NVIDIA A10/A10G benchmarks, CNCM-Nano (`14.50M` parameters) achieves **`99.93%` Stage 1 reconstruction accuracy**, **`6.40×–7.63×` wall-clock Concept-Mode speedup** over a parameter-matched Pure-AR baseline (`255.8 ms` vs. `1,950.9 ms` for 64 tokens at `B=1`), **`12.29 MB` packed INT4 weight footprint** ($\Delta\text{NLL} = +0.0054$ PTQ; $-0.0338$ after QAT), **`100.0%` 4-turn top-1 token agreement** under INT8 KV compression, **`4e-6`–`6e-6` ONNXRuntime parity**, and **`62.5%` model-only Plan-Guided Fill-in-the-Middle (PG-FIM) parse/compile rate** (`50.0%` unit-test pass@1 at `821.9 ms`, faster than AR FIM's `932.6 ms`). Scaling to **CNCM-Base (`93.90M` dense)** and an experimental **Fine-Grained Mixture-of-Experts (`base_moe`, `221.45M` total / `94.05M` active)** on a 4-domain curated corpus demonstrates `-0.1733` Phase 2 validation NLL (`-75.47` PPL) and `8/8` balanced expert utilization (`0.9984` normalized load entropy).

---

## 1. Introduction

Deploying language models on consumer edge hardware—smartphones, laptops, and embedded devices—is constrained less by raw arithmetic FLOPs than by **memory bandwidth**. Standard token-by-token autoregressive decoding must stream all model weights from DRAM for every generated token.

Block-wise and concept-space language models amortize weight sweeps by predicting multi-token spans in parallel. However, naive latent-space models suffer from three practical bottlenecks:
1. **Discrete Codebook Collapse & Memory Overhead:** VQ-VAE codebooks require large embedding tables and EMA updates that collapse under small batch sizes.
2. **Positional Misalignment:** Resetting target concept positions to `0..ℓ_k-1` inverts relative RoPE distances to the causal prefix, degrading target-to-prefix attention.
3. **Loss of Exact Syntax:** Parallel latent decoding struggles with strict syntactic invariants required for JSON tool calls and executable code infilling.

CNCM addresses all three challenges in a single compact architecture:
- **Zero-Table M-RFSQ Tokenizer (Stage 1):** Uses implicit integer grids (`fsq_levels=(8,8,8,8)`, `4,096` codes) with a 1-layer strictly causal semantic encoder (`enc_layers=1`) and Matryoshka multi-cutoff supervision (`k ∈ {4,6,7,8}`).
- **Continuous Scaled RoPE & $\mathcal{L}_{\text{align}}$:** Places level-$k$ target positions strictly after the prefix at their physical span centers, $\text{pos}_k(i) = P + \lfloor (i+0.5)L/\ell_k \rfloor$, and projects finest-level concept hidden states through the shared Exact-AR head.
- **Dual-Mode Execution & PG-FIM:** Uses Concept Mode for fast prose/reasoning drafting and Exact-AR Mode for grammar-constrained JSON tool calls, self-speculative verification with an EMA circuit breaker, and Plan-Guided Fill-in-the-Middle (`<mlen_k>` + `<kw>...</kw>`) with local-span autoregressive repair.

---

## 2. Architecture

### 2.1 Stage 1: Causal M-RFSQ Concept Tokenizer
Given a length-$L$ token chunk ($L=64$), Stage 1 embeds tokens, applies a 1-layer strictly causal Transformer encoder (`enc_layers=1`) followed by affine-free `GroupNorm` (`pre_quant_norm`), and quantizes residuals across $K=8$ multiscale levels:
$$\ell = (1, 2, 4, 8, 12, 20, 36, 64)$$
Each level downsamples the residual, projects to `d_q = 4` dimensions, applies bounded `tanh` + Straight-Through Estimator (STE) rounding onto `8×8×8×8 = 4,096` implicit integer grid points, and upsamples back. Training uses nested fine-level dropout and Matryoshka reconstruction loss across cutoffs `k ∈ {4, 6, 7, 8}` so coarse levels remain independently decodable for early exit.

### 2.2 Stage 2: Causal-Hybrid Recursive Backbone & Dual Heads
Stage 2 freezes Stage 1 and trains a shared Transformer backbone with **Relaxed Recursive Layer Looping** (e.g., `7` unique layers × `3` loops = `21` effective layers in `nano`; `12 × 2 = 24` effective layers in `base`) augmented by lightweight low-rank `LoopAdapter`s per effective layer.

- **Factorized Concept Head:** Predicts the `4` FSQ digits of each code autoregressively via chained micro-heads ($O(4 \times 8)$ logits instead of a flat `4,096`-way softmax).
- **Exact-AR Head:** Tied to `exact_embed` for standard next-token prediction, JSON-schema logit masking, and local-span PG-FIM repair.
- **Continuous Scaled RoPE:** Prefix tokens occupy positions `0..P-1`, while concept level $k$ occupies `P + floor((i + 0.5) * L / l_k)`, preserving monotonic temporal distance between prefix and target spans.

### 2.3 Experimental Fine-Grained Mixture-of-Experts (`base_moe`)
To scale knowledge capacity without increasing per-token FLOPs, our experimental `base_moe` configuration replaces the dense SwiGLU FFN (`d=2304`) with a DeepSeek-style Fine-Grained MoE layer (`MoEFFN`): **1 always-on Shared Expert** (`d=768`) + **8 Routed Experts** (`d=768`, Top-2 active = `2304` active width), **24 per-loop routers** (`step_moe_routers`), and a Switch/DeepSeek load-balancing auxiliary loss (`moe_aux_loss_weight = 0.01`) implemented with single-sort sparse dispatch.

---

## 3. Experimental Setup

All experiments are evaluated on held-out validation splits using NVIDIA A10 / A10G cloud GPUs and CPU ONNXRuntime, with `25/25` unit tests verified in `tests/test_cncm.py`:

- **CNCM-Nano (`14.50M` params)**: Evaluated on `roneneldan/TinyStories` + synthetic Python/PG-FIM/tool-use corpora (`modal_verify_report.json`, `modal_v2_verify_report.json`).
- **CNCM-Base (`93.90M` dense) vs. `base_moe` (`221.45M` total / `94.05M` active)**: Evaluated on a controlled 4-domain curated corpus (`3.17M` train tokens / `355K` val tokens, `V=6400` Byte-Level BPE with `Digits(individual_digits=True)`, `0%` padding via `uint16` `MemmapTokenDataset`) streaming from:
  - English Textbooks: `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`, `fineweb-edu-dedup`)
  - Python Code: `Nan-Do/code-search-net-python`, `TokenBender/code_instructions_122k_alpaca_style`, PG-FIM spans
  - Math & `<think>` CoT: `HuggingFaceTB/finemath` (`finemath-4plus`), `meta-math/MetaMathQA`
  - Function Calling & JSON: `glaiveai/glaive-function-calling-v2`, strict JSON schemas

---

## 4. Empirical Results

### 4.1 Stage 1 Reconstruction & Concept-Mode Speedup (CNCM-Nano, `14.50M`)

| Metric | Measured Result | Notes |
|---|---:|---|
| **Stage 1 Full-Depth Recon (`k=8`)** | **`99.93%`** | Zero codebook tables; FP32-safe STE |
| **Stage 1 Early-Exit Recon (`k=6` / `k=7`)** | **`39.39%` / `64.81%`** | Matryoshka multi-cutoff supervision |
| **Concept-Mode Wall-Clock Speedup (`B=1`, 64 tok)** | **`6.40×–7.63×`** | `255.8 ms` vs. `1,950.9 ms` Pure-AR |
| **Concept Head Parameter Count** | **`71,072`** (`nano`) | `23.3×` fewer output MACs than flat 4096 |
| **INT8 KV Cache 4-Turn Top-1 Agreement** | **`100.0%`** | `ΔNLL = -9e-5` |
| **Packed INT4 Weight Footprint** | **`12.29 MB`** | `ΔNLL = +0.0054` PTQ; `-0.0338` after QAT |
| **ONNXRuntime Parity** | **`4e-6`–`6e-6`** | Max absolute logit diff vs. PyTorch |

### 4.2 Held-Out Language Modeling & PG-FIM Code Infilling (`modal_v2_verify_report.json`)

| Metric (Strict Held-Out Eval, `allow_template_fallback=False`) | Measured Result |
|---|---:|
| **PG-FIM Model-Only Parse / Compile Rate** (`24` held-out tasks) | **`62.5%`** (vs. AR FIM `0.0%`) |
| **PG-FIM Model-Only Unit-Test Pass@1** | **`50.0%`** (`12/24` passing) |
| **PG-FIM Model-Only Latency / Task** | **`821.9 ms`** (`1.13×` faster than AR FIM `932.6 ms`) |
| **Stage 2 Held-Out Exact-AR NLL / PPL** | **`1.1696` / `3.22`** (vs. Pure-AR `0.903`) |
| **Self-Speculative Speed vs. AR (`exact_greedy_match=True`)** | **`0.695×`** (`2,613 ms`, EMA circuit breaker active) |

### 4.3 Controlled `ctx0an` Base (`93.90M` Dense) vs. `base_moe` (`221.45M` Total / `94.05M` Active)

On NVIDIA A10 (`BF16`, `200` Phase-1 Exact-AR + `200` Phase-2 Joint steps, `3.07M` Stage 2 tokens per variant):

| Metric | Dense Base (`93.90M`) | **`base_moe` (`94.05M` active)** | Delta (MoE − Dense) |
|---|---:|---:|---:|
| **Phase 1 Val Exact-AR NLL** | `6.4818` | **`6.4026`** | **`-0.0792`** |
| **Phase 2 Val Exact-AR NLL / PPL** | `6.1619` / `474.33` | **`5.9886` / `398.86`** | **`-0.1733` / `-75.47`** |
| **Phase 2 Python Val NLL** | `6.1198` | **`5.9460`** | **`-0.1738`** |
| **Phase 2 English / Math / JSON NLL** | `6.6818` / `6.1063` / `5.7397` | **`6.5213` / `5.9724` / `5.5145`** | **`-0.1605` / `-0.1339` / `-0.2252`** |
| **Gate #5 Align NLL / Concept Draft NLL** | `6.6168` / `9.3439` | **`6.3854` / `8.8170`** | **`-0.2314` / `-0.5269`** |
| **Training Throughput (Phase 1 / Phase 2)** | **`14,689` / `6,490` tok/s** | `10,625` / `4,828` tok/s | `0.723×` / `0.744×` |
| **MoE Load Balance & Loop Switching** | — | **`8/8` active (`0.9984` entropy)** | **`82.85%`** Loop 0→1 switch |

---

## 5. Limitations & Future Work

1. **Pure-PyTorch INT4 Inference Overhead:** While packed INT4 reduces model memory to `12.29 MB` (`nano`) / `~46.6 MB` (`base`), pure-PyTorch dequantization is `~2.32×` slower than FP32 on CPU; dedicated ONNX QDQ or native INT4 kernels are required for edge wall-clock speedups.
2. **Self-Speculative Acceptance at Small Scale:** With the EMA circuit breaker, worst-case self-speculative latency is bounded (`0.695×` of AR with `exact_greedy_match=True`), but achieving `>1.0×` speculative speedup requires larger-scale Stage 2 training so Concept Mode drafts match Exact-AR tokens more closely.
3. **Sparse GRPO Rewards at `14.5M`:** Stage 3 GRPO/CISPO executes with finite gradients and frozen Stage 1, but zero-variance rollout groups limit reward signal at small parameter/token budgets.
4. **Short-Budget Recursive Looping:** Under short training budgets (`300–500` steps), non-looped `7×1` converges faster than looped `7×3`; multi-session pretraining (`1B–2B+` tokens) is needed to realize the full capacity of recursive depth.

---

## 6. Conclusion

CNCM demonstrates that a single compact recursive Transformer can unify fast parallel Concept-Mode drafting (`6.40×–7.63×` speedup at `B=1`) with exact autoregressive syntax, `100%` 4-turn INT8 KV agreement, `12.29 MB` INT4 edge packing, and `62.5%` model-only PG-FIM code infilling. Extending the backbone with Fine-Grained MoE (`base_moe`) further improves held-out NLL across all four domains (`-0.1733` NLL) with zero expert collapse (`0.9984` entropy), while Dense `base` remains `~1.35×` faster to train and `~2.4×` smaller for tight edge deployments.
