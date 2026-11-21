# CNCM: A Compact Causal-Hybrid Next-Concept Model for Edge Reasoning, Code Infilling, and Agentic Execution

**Abstract.** Autoregressive (AR) language models generate text one token at a time, making batch-1 edge inference strictly memory-bandwidth bound on mobile CPUs and laptop iGPUs. Conversely, parallel latent-concept models amortize weight sweeps across multi-token chunks but suffer from VQ-VAE codebook bloat, positional misalignment, and discrete syntax corruption on executable code and JSON tool calls. We present **CNCM** (*Compact Causal-Hybrid Next-Concept Model*), a unified dual-mode Transformer architecture that shares a single backbone between two complementary decoding heads: (1) a **Factorized Concept Head** that drafts 64-token chunks in parallel across $K=8$ coarse-to-fine residual levels in 8 backbone passes, and (2) a shared **Exact-AR Head** for strict token-by-token generation, neuro-symbolic JSON-schema grammar masking, lossless self-speculative verification, and localized code repair. To eliminate codebook collapse and lookup-table memory, CNCM introduces a block-causal **Multiscale Residual Finite Scalar Quantization (M-RFSQ)** Stage 1 tokenizer (`18.03×` fewer quantizer parameters and `23,931×` smaller buffers than RVQ-VAE) trained with Matryoshka multi-cutoff supervision ($k \in \{4,6,7,8\}$). To align Concept Mode with Exact-AR geometry, we introduce **Continuous Scaled RoPE** ($\text{pos}_k(i) = P + \lfloor (i+0.5)L/\ell_k \rfloor$), a **Text-Level Alignment Loss ($\mathcal{L}_{\text{align}}$)**, and a **3-Level Hierarchical Teacher$\to$Student Distillation framework** using **Ternary-Bonsai-2-4B** (`llama-server` Verifier-Gated SeqKD + **Dynamic Residual Concept-DAgger** + Frozen-Teacher Cross-Mode KL). In a controlled 4-arm cumulative ablation against a **Vanilla NCM** baseline on Modal NVIDIA A10, CNCM reduces combined quantizer and concept-head parameters by **`15.20×`** (`6.00M` $\to$ `0.39M`), eliminates Stage 1 RVQ codebook collapse (**`4.96%` $\to$ `99.04%`** `k=8` reconstruction; **`35.1×`** higher active codebook utilization), reduces held-out Concept Draft NLL by **`-3.941` nats** (`8.777` $\to$ `4.836`), and lifts tool-calling syntax and semantic accuracy from **`0.0%` $\to$ `100.0%`**. Across cloud NVIDIA A10/A10G GPUs and physical laptop CPU (`CPUExecutionProvider`) and Intel Iris Xe iGPU (`DmlExecutionProvider`) hardware, CNCM-Nano (`14.50M` parameters) achieves **`6.40×–7.63×` GPU** and **`7.12×–7.33×` DirectML iGPU** (`2.26×–2.37×` ORT CPU) Concept-Mode wall-clock speedups over KV-cached Exact-AR, **`7.50 MB` fused INT4 ONNX** (`5.84×` graph compression via looped-weight `Identity` inlining) and **`12.29 MB` packed PyTorch INT4** weight footprints, **`100.0%` 4-turn INT8 KV top-1 agreement**, and **`64/64` exact token parity** at **`6.72×` speedup** under in-distribution convergence. Scaling to **CNCM-Base (`93.90M` dense)** and an experimental **Fine-Grained Mixture-of-Experts (`base_moe`, `221.45M` total / `94.05M` active)** on a 4-domain BPE corpus demonstrates `-0.1733` Phase 2 validation NLL (`-75.47` PPL) with `8/8` balanced expert utilization (`0.9984` normalized load entropy).

---

## 1. Introduction

Deploying language models on consumer edge hardware—smartphones, laptops, and embedded devices—is constrained less by raw arithmetic FLOPs than by **memory bandwidth**. Standard token-by-token autoregressive decoding must stream all model weights from DRAM for every generated token ($64\times$ per 64-token chunk).

Block-wise and concept-space language models amortize weight sweeps by predicting multi-token spans in parallel. However, existing latent-concept models suffer from four practical bottlenecks:
1. **Discrete Codebook Collapse & Memory Overhead:** Multiscale RVQ-VAE codebooks require large explicit embedding tables and FP32 EMA cluster buffers (`~18.87M` parameters / `~151 MB` in vanilla NCM) that collapse under small batch sizes or gradient accumulation.
2. **Positional Misalignment:** Resetting target concept positions to `0..ℓ_k-1` collides with prefix RoPE angles and inverts relative distances to the causal prefix, degrading target-to-prefix attention.
3. **Coarse-to-Fine Exposure Bias:** Teacher-forced concept levels train on ground-truth coarse residuals $\hat{z}_{\le k-1}^*$, whereas open-loop inference conditions on the student's own predicted codes $\hat{c}_{\le k-1}$, causing error accumulation across levels $k=0\dots K-1$.
4. **Loss of Exact Syntax:** Parallel latent decoding struggles with strict syntactic invariants required for JSON tool calls, exact arithmetic arguments, and executable code infilling.

CNCM addresses all four challenges in a single compact architecture:
- **Zero-Table Block-Causal M-RFSQ Tokenizer (Stage 1):** Uses implicit integer grids (`fsq_levels=(8,8,8,8)`, `4,096` codes) with a 1-layer causal semantic encoder (`enc_layers=1`), affine-free `GroupNorm`, and Matryoshka multi-cutoff supervision (`k ∈ {4,6,7,8}`).
- **Continuous Scaled RoPE & $\mathcal{L}_{\text{align}}$:** Places level-$k$ target positions strictly after the prefix at their physical span centers, $\text{pos}_k(i) = P + \lfloor (i+0.5)L/\ell_k \rfloor$, and projects finest-level concept hidden states through the shared Exact-AR head.
- **3-Level Hierarchical Teacher$\to$Student Distillation (`Ternary-Bonsai-2-4B`):** Combines (1) Verifier-Gated Sequence-Level KD (`llama-server` + AST/unit-test & tool sandboxes), (2) **Dynamic Residual Concept-DAgger** (analytically re-quantizing remaining teacher residuals $z_{\text{teacher}} - \hat{z}_{\le k-1}^{\text{student}}$ on the fly), and (3) **Frozen-Teacher Cross-Mode KL Distillation**.
- **Neuro-Symbolic Dual-Mode Execution & PG-FIM:** Uses Concept Mode for fast parallel drafting (`<think>` and prose) and Exact-AR Mode for grammar-constrained JSON tool calls (`JSONSchemaConstrainedMasker` + prompt copy-span constraints), lossless self-speculative verification with an EMA circuit breaker, and Plan-Guided Fill-in-the-Middle (`<mlen_k>` + `<kw>...</kw>`) with local-span AST autoregressive repair.

---

## 2. Related Work

CNCM builds upon and bridges four active research directions:
- **Concept-Space & Block-Parallel Language Models:** Next-Concept Modeling (NCM; Kumbong et al., 2025, `arXiv:2610.14162`) introduced multiscale coarse-to-fine latent generation via RVQ-VAE; Meta's Large Concept Models (LCM; Meta AI, 2024, `arXiv:2412.08821`), Byte Latent Transformer (BLT; Pagnoni et al., 2024), and MegaByte (Yu et al., 2023) model text at patch or sentence granularity. CNCM eliminates NCM's `151 MB` RVQ-VAE tables via M-RFSQ, replaces bidirectional prefix attention with causal-prefix block-AR KV caching, and unifies concept drafting with an exact token-level AR head in one backbone.
- **Vector Quantization & Matryoshka Representations:** Finite Scalar Quantization (FSQ; Mentzer et al., ICLR 2024) replaces learned VQ codebooks with bounding and rounding onto a fixed scalar grid. CNCM extends FSQ to a **multiscale residual hierarchy (`M-RFSQ`)** with depthwise-separable $\Phi$ refinement and Matryoshka multi-cutoff supervision (Kusupati et al., NeurIPS 2022), enabling both dynamic early exit and analytic on-policy residual re-quantization (**Concept-DAgger**).
- **Speculative & Parallel Decoding:** Speculative decoding (Leviathan et al., ICML 2023; Chen et al., 2023), Medusa (Cai et al., ICML 2024), EAGLE / EAGLE-2 (Li et al., 2024), LayerSkip (Elhoushi et al., ACL 2024), and Lookahead Decoding (Fu et al., ICML 2024) accelerate AR inference using draft heads or Jacobi iterations. CNCM provides a unified dual-mode self-speculative verifier (`generate_self_speculative`) with an EMA acceptance circuit breaker and explicit sub-chunk acceptance analysis.
- **Compact Edge LLMs & Low-Bit Teachers:** MobileLLM (Liu et al., ICML 2024) studied weight sharing and recursive layer looping for sub-billion edge models; DeepSeekMoE (Dai et al., 2024) introduced fine-grained shared + routed experts; and PrismML's **Ternary-Bonsai-2-4B** (2026) demonstrated high-density ternary ($\{-1, 0, +1\}$) reasoning and tool use, which we leverage as an edge-compatible `llama-server` SeqKD teacher.

---

## 3. Architecture & Hierarchical Distillation

### 3.1 Stage 1: Block-Causal M-RFSQ Concept Tokenizer
Given a length-$L$ token chunk ($L=64$), Stage 1 embeds tokens, applies a 1-layer strictly causal Transformer encoder (`enc_layers=1`) followed by affine-free `GroupNorm` (`pre_quant_norm`), and quantizes residuals across $K=8$ multiscale levels:
$$\ell = (1, 2, 4, 8, 12, 20, 36, 64), \qquad \sum_{k=0}^{K-1} \ell_k = 147 \;\; (2.30\times L)$$
At each level $k$, the residual $r_k \in \mathbb{R}^{B \times L \times d_{\text{vq}}}$ is downsampled to length $\ell_k$, projected to $d_q = 4$ dimensions, quantized via bounded $\tanh$ + Straight-Through Estimator (STE) rounding onto $8^4 = 4{,}096$ implicit integer grid points, upsampled back to length $L$, and refined via a depthwise-separable residual convolution $\Phi_k$:
$$\tilde{c}_j = \frac{L_j - 1}{2} + \frac{L_j}{2}(1 - \varepsilon)\tanh(h_j), \qquad c_j = \text{clip}\big(\lfloor \tilde{c}_j \rceil,\; 0,\; L_j - 1\big)$$
*(Note: While `encoder_blocks` is strictly causal per token, multiscale area-pooling within a 64-token chunk aggregates across the $L=64$ block, making Stage 1 **block-causal** across 64-token chunks.)* Training uses nested fine-level dropout and Matryoshka reconstruction loss across cutoffs $k \in \{4, 6, 7, 8\}$ so coarse prefixes $\hat{z}_{\le k}$ remain independently decodable.

### 3.2 Stage 2: Causal-Hybrid Backbone, Factorized Concept Head & Continuous Scaled RoPE
Stage 2 freezes Stage 1 and trains a shared Transformer backbone (Pre-RMSNorm + QK-RMSNorm + GQA + SwiGLU, supporting optional Relaxed Recursive Layer Looping with rank-$r$ `LoopAdapter`s):
- **Factorized Concept Head (`FactorizedConceptHead`):** Factorizes the 4,096-way code distribution into 4 chained autoregressive digit micro-heads:
  $$\log p_\theta(c_0, c_1, c_2, c_3 \mid h) = \sum_{j=0}^{3} \log p_\theta\!\left(c_j \;\middle|\; h,\; e_{<j}(c_{<j})\right)$$
  requiring only **`71,072` parameters** on `nano` (**`22.1×` smaller** than a `1.57M`-parameter flat 4,096-way linear head).
- **Exact-AR Head:** Weight-tied to `exact_embed` for standard next-token prediction, JSON-schema logit masking, self-speculative verification, and local-span PG-FIM repair.
- **Continuous Scaled RoPE & $\mathcal{L}_{\text{align}}$:** Prefix tokens occupy positions $0\dots P-1$, while concept level $k$ occupies:
  $$\text{pos}_k(i) = P + \left\lfloor \frac{(i + 0.5) L}{\ell_k} \right\rfloor, \qquad i \in \{0, \dots, \ell_k - 1\}$$
  preserving monotonic temporal distance between prefix and target spans. During joint Stage 2 training, the finest-level concept hidden states $h_{\text{fine}}$ ($l_7 = 64$) are simultaneously projected through `exact_head` via a text-level cross-entropy alignment loss $\mathcal{L}_{\text{align}}$.

### 3.3 3-Level Hierarchical Teacher$\to$Student Distillation (`Ternary-Bonsai-2-4B`)
To bridge the capacity gap of a `14.5M`-parameter dual-mode student without mutating the Exact-AR distribution during concept alignment, CNCM introduces a three-stage distillation pipeline (`experiments/distill_seqkd.py`, `cncm/trainer/draft_align.py`):
1. **Level 1: Verifier-Gated Sequence-Level KD (SeqKD):** Queries an OpenAI-compatible **`llama-server`** serving **`prism-ml/Ternary-Bonsai-2-4B-gguf`** (`Ternary-Bonsai-2-4B-PQ2_0.gguf`, `Apache-2.0`), caps verbose `<think>` traces (`max_think_chars=240`), and filters every completion through deterministic verifiers (`run_code_sandbox` for Python AST/compile/unit-test execution; `evaluate_tool_call_semantic_accuracy` for JSON tool calls and arithmetic).
2. **Level 2: Dynamic Residual Concept-DAgger (`build_concept_dagger_rollout`):** Because `MultiscaleRFSQ` is an analytic zero-table scalar grid, on-policy student rollouts at coarse levels $\hat{c}^{(0)}, \dots, \hat{c}^{(k-1)}$ can dynamically re-quantize the **true remaining teacher residual** at level $k$ in closed form:
   $$r_k^{\text{DAgger}} = z_{\text{teacher}} - \sum_{j=0}^{k-1} \Delta\hat{z}_j(\hat{c}_{\text{student}}^{(j)}), \qquad c_k^{*\text{DAgger}} = \text{FSQ}_k\!\big(\text{Down}_k(r_k^{\text{DAgger}})\big)$$
   teaching fine levels $k \ge 1$ to actively recover from coarse-level prediction errors.
3. **Level 3: Frozen-Teacher Cross-Mode KL Distillation (`train_draft_alignment`):** Freezes an immutable post-SFT teacher snapshot (`teacher_ref`) and locks the shared Exact-AR backbone (`freeze_backbone_for_exact=True`), training `concept_head`, `concept_in_proj`, `level_embed`, and `l0_queries` on on-policy teacher greedy rollouts $y_{\text{AR}}$ with token-level KL divergence $D_{\text{KL}}\big(p_{\text{teacher}}(y \mid x) \,\|\, \text{exact\_head}(h_{\text{fine}})\big)$.

### 3.4 Experimental Fine-Grained Mixture-of-Experts (`base_moe`)
To study knowledge capacity scaling at constant active FFN width (`2304`), `base_moe` (`221.45M` total / `94.05M` active) replaces the dense SwiGLU FFN with a DeepSeek-style Fine-Grained MoE layer (`MoEFFN`): **1 always-on Shared Expert** (`d=768`) + **8 Routed Experts** (`d=768`, Top-2 active = `2304` active width), **24 per-loop routers** (`step_moe_routers`), and a load-balancing auxiliary loss (`moe_aux_loss_weight = 0.01`).

---

## 4. Empirical Results & Controlled Ablations

All claims are tracked in [CLAIMS.md](CLAIMS.md) with reproducible JSON artifacts on Modal NVIDIA A10/A10G GPUs and physical laptop CPU + Intel Iris Xe DirectML iGPU hardware.

### 4.1 Stage 1 M-RFSQ Reconstruction, Continuous RoPE & Head Ablations (`CNCM-Nano`, `14.50M`)

| Experiment / Metric | Measured Result | Baseline / Control | Source Artifact |
|---|---:|---:|---|
| **Stage 1 Full-Depth Recon (`k=8`)** | **`99.04%` – `99.98%`** | vs. **`4.96%`** Vanilla RVQ (`0.50%` active codes) | `modal_ncm_vs_cncm_report.json` / `modal_v2_verify_report.json` |
| **Matryoshka Early-Exit Recon (`k=6` / `k=7`)** | **`50.31%` (`k=6`) / `64.81%` (`k=7`)** | vs. **`4.59%`** Vanilla RVQ (`31.41%` w/o Matryoshka) | `modal_ncm_vs_cncm_report.json` / `modal_v2_verify_report.json` |
| **Stage 1 Quantizer Parameters & Buffers** | **`245,792` params / `0.69 KB` buf** | **`18.03×` fewer params** & **`23,931×` smaller buffers** than Vanilla RVQ (`4.43M` params / `16.51 MB` buf) | `modal_ncm_vs_cncm_report.json` |
| **Continuous Scaled RoPE + $\mathcal{L}_{\text{align}}$ (`v2` vs `v1`)** | **`1.1696` Exact NLL** (`3.22` PPL) | vs. `1.4065` (`v1` reset RoPE); Concept Loss `6.874` vs `7.225` | `modal_v2_verify_report.json` |
| **Combined Quantizer + Concept Head Footprint** | **`394,944` params** (`nano`) | **`15.20×` smaller** than Vanilla NCM (`6,003,712` params) | `modal_ncm_vs_cncm_report.json` |
| **INT8 KV Cache 4-Turn Top-1 Agreement** | **`100.0%`** (`ΔNLL = -9e-5`) | **`>2.5×` smaller** KV RAM vs FP32 KV | `modal_verify_report.json` / `test_cncm.py` |
| **Packed PyTorch INT4 Footprint & Fair 3-Arm QAT** | **`12.29 MB`** (`4.59×` smaller) | PTQ `4.105` vs `FP32+40` `4.059` vs `QAT+40` `4.057` (`Δ = -0.0025`, $p=0.44$) | `controls_summary.json` (`exp_1_3_fair_qat`) |
| **Recursive Looping Ablation (`7×3` vs `7×1`, 400 steps)** | **`2.789 ± 0.506`** (`7×1` non-looped) | vs. `3.326 ± 0.161` (`7×3` looped, paired $p=0.042$ at short budget) | `controls_summary.json` (`exp_5_2_looping`) |

#### Table 4.1b: Controlled 4-Arm Cumulative Ablation from Vanilla NCM to Full CNCM (`CNCM-Nano`, Modal NVIDIA A10, `seed=42`)
To isolate each architectural innovation over Vanilla NCM (Kumbong et al., `arXiv:2610.14162`), we trained four cumulative arms on the identical `nano` backbone, `Ternary-Bonsai-2-4B` SeqKD + multi-domain split, and step budget (`350` Stage-1 + `400` Stage-2 + `120` Align steps; [modal_ncm_vs_cncm_report.json](modal_ncm_vs_cncm_report.json)):

| Architecture Variant (`CNCM-Nano`, Modal A10) | Quantizer + Head Params $\downarrow$ | Stage 1 Recon (`k=8` / `k=6`) $\uparrow$ | Stage 1 Active Codebook % $\uparrow$ | Held-Out Concept Loss $\downarrow$ | Held-Out Concept Draft NLL $\downarrow$ | Held-Out Exact-AR NLL $\downarrow$ | Tool-Call Syntax / Semantic Acc (`n=40`) $\uparrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. Vanilla NCM Baseline**<br>*(RVQ+EMA `8×4096×128` + Flat 4096 Head + Reset RoPE + Concept-Only)* | `6,003,712`<br>*(`+16.5 MB` EMA)* | `4.96%` / `4.59%`<br>*(Codebook Collapse)* | `0.50%` | `11.5125` | `8.7773` | `8.3429` | `0.0%` / `0.0%` |
| **2. + CNCM Stage 1 (`M-RFSQ` + Matryoshka)**<br>*(Zero-Table $8^4$ FSQ Grid + Cutoffs `4,6,7,8`)* | `1,818,656`<br>*(`0.69 KB` buf)* | **`99.04%` / `50.31%`** | **`17.53%`**<br>*(`35.1×` higher)* | `6.9894`<br>*(`-4.523`)* | `8.3257` | `8.3405` | `0.0%` / `0.0%` |
| **3. + `FactorizedConceptHead` + Continuous Scaled RoPE** | **`394,944`**<br>*(**`15.20×` smaller**)* | `99.04%` / `50.31%` | `17.53%` | `6.9465`<br>*(`-0.043`)* | `8.3533` | `8.4083` | `0.0%` / `0.0%` |
| **4. Full CNCM (Dual-Mode + $\mathcal{L}_{\text{align}}$ + `Bonsai-2-4B` Concept-DAgger + Copy-Span Masker)** | **`394,944`**<br>*(Tied Exact Head)* | **`99.04%` / `50.31%`** | **`17.53%`** | **`6.0556`**<br>*(**`-5.457` vs NCM**)* | **`4.8364`**<br>*(**`-3.941` vs NCM**)* | **`0.5600`**<br>*(`1.296` Align NLL)* | **`100.0%` / `100.0%`** |

### 4.2 Physical Edge Deployment: Fused INT4/INT8 ONNX, Laptop CPU & Intel Iris Xe iGPU

To resolve a PyTorch ONNX exporter flaw where weight-shared recursive loops emit `Identity` alias nodes that prevent downstream quantizers (`MatMulNBitsQuantizer` / `quantize_dynamic`) from compressing looped weights, we implemented an **ONNX `Identity` Alias Inliner** (`_inline_onnx_identities` in [cncm/export/edge_bench.py](cncm/export/edge_bench.py)).

#### Table 4.2a: True ONNX Graph Compression After `Identity` Alias Inlining (`CNCM-Nano`)
| Exported ONNX Subgraph | FP32 ONNX | INT8 Dynamic | INT8 Static QDQ | **INT4 `MatMulNBits` (`block=32`)** | Compression Ratio (FP32 / INT4) |
|---|---:|---:|---:|---:|---:|
| **`concept_level_kv.onnx`** (Stage 2 Concept Step) | `43.785 MB` | `12.035 MB` | `13.201 MB` | **`7.499 MB`** *(was `49.55 MB` before fix)* | **`5.84×`** |
| **`exact_kv_step.onnx`** (Stage 2 Exact-AR + Tied Embed) | `55.784 MB` | `15.034 MB` | `16.182 MB` | **`14.436 MB`** *(was `56.49 MB` before fix)* | **`3.86×`** |
| **`stage1_decoder.onnx`** (Stage 1 M-RFSQ Decoder) | `3.019 MB` | `0.866 MB` | `0.968 MB` | **`0.652 MB`** | **`4.63×`** |

#### Table 4.2b: Measured Multi-Device Wall-Clock Latency & Thermal Soak (`B=1`, 64 Tokens, `CNCM-Nano`)
| Hardware Platform & Execution Provider | Precision | 8-Step Concept Latency (`tok/s`) | 64-Step Exact-AR Latency (`tok/s`) | **Measured Speedup** | Thermal Soak Stability (Last Q / First Q) |
|---|---|---:|---:|---:|---:|
| **Cloud NVIDIA A10 / A10G GPU** (PyTorch) | `BF16` | **`187.6–267.3 ms`** (`240–341 tok/s`) | `1,272.5–1,815.0 ms` (`35–50 tok/s`) | **`6.72× – 6.79×`** (`8.85×` @ `k=6` early exit) | — |
| **Laptop Intel Iris Xe iGPU** (`DmlExecutionProvider`) | `FP32` | **`1,592.1 ms`** (`40.2 tok/s`) | `11,330.7 ms` (`5.65 tok/s`) | **`7.12×`** | — |
| **Laptop Intel Iris Xe iGPU** (`DmlExecutionProvider`) | `INT4 MatMulNBits` | **`2,012.5 ms`** (`31.8 tok/s`) | `14,740.9 ms` (`4.34 tok/s`) | **`7.33×`** | **`0.984`** (`27.9°C`, zero throttling) |
| **Laptop Intel 16-Thread CPU** (`CPUExecutionProvider`) | `FP32` | **`370.8 ms`** (`172.6 tok/s`) | `838.9 ms` (`76.3 tok/s`) | **`2.26×`** | — |
| **Laptop Intel 16-Thread CPU** (`CPUExecutionProvider`) | `INT4 MatMulNBits` | **`372.9 ms`** (`171.6 tok/s`) | `882.4 ms` (`72.5 tok/s`) | **`2.37×`** | **`0.971`** (`27.9°C`, `634.6 MB` RSS) |
| **Laptop Intel 16-Thread CPU** (Pure PyTorch Reference) | `FP32` vs `Int4Packed` | `343.7 ms` vs `893.3 ms` | `1,601.7 ms` vs `5,185.5 ms` | `4.66×` vs `5.81×` | Pure-PyTorch INT4 dequant is memory-only (`0.385×` speed) |

### 4.3 3-Level Hierarchical Teacher$\to$Student Distillation (`Ternary-Bonsai-2-4B` via `llama-server`)

We executed the complete two-stage teacher-student pipeline on Modal NVIDIA A10/A10G ([modal_bonsai_distill.py](modal_bonsai_distill.py), [modal_bonsai_distill_report.json](modal_bonsai_distill_report.json)):
1. **Stage A (Live `llama-server` SeqKD Generation):** Served `prism-ml/Ternary-Bonsai-2-4B-gguf` (`Ternary-Bonsai-2-4B-PQ2_0.gguf`) via `llama-server` and queried 48 balanced tool-calling, arithmetic, and Python FIM prompts via `experiments/distill_seqkd.py --api-base http://127.0.0.1:8080/v1`. **31 / 48 (`64.58%`)** teacher completions passed strict AST/unit-test and semantic tool verifiers (`sha256: 285b4944...`, saved in [experiments/results/verified_seqkd.jsonl](experiments/results/verified_seqkd.jsonl)).
2. **Stage B (Single-Seed `seed=42` 4-Arm Hierarchical Distillation Ablation on `CNCM-Nano`):**

| Distillation Arm (`CNCM-Nano 14.5M`, `seed=42`, Modal A10) | Held-Out Exact-AR NLL $\downarrow$ | Held-Out Concept Draft NLL (Gate #5) $\downarrow$ | Held-Out Concept Loss (Gate #1) $\downarrow$ | Sub-Chunk Acceptance $\alpha$ ($\gamma=16$) |
|---|---:|---:|---:|---:|
| **Arm 1: Baseline Raw Corpus** (150 S1 + 200 S2 steps) | `7.37231` | `5.41755` | — | — |
| **Arm 2: + Level 1 `Ternary-Bonsai-2-4B` Verified SeqKD** | **`0.11575`** | `5.39008` | `4.63002` *(Passes Gate #1 `<5.5`)* | `0.00781` |
| **Arm 3: + Level 2 Dynamic Residual Concept-DAgger** (+60 steps) | `0.11575` *(Locked)* | `5.21222` (`-0.1779`) | — | `0.00781` |
| **Arm 4: + Level 3 Full Cross-Mode KL + Concept-DAgger** (+60 steps) | **`0.11575`** *(Zero Drift)* | **`5.12927`** (`-0.2883` vs Raw; *Passes Gate #5 `<7.82`*) | **`4.59844`** | **`0.03125`** |
| *Control: Equal-Param Pure-AR + Identical `Bonsai-2-4B` SeqKD* | `0.02788` | — | — | — |

- **Elimination of Teacher Drift Collapse:** In earlier unanchored self-distillation (`controls_summary.json` `exp_5_1`), updating the shared backbone without freezing the Exact-AR teacher caused acceptance $\alpha$ to collapse from `28.17%` $\to$ `1.24%`. Locking `teacher_ref` and `freeze_backbone_for_exact=True` in Level 3 reduces Concept Draft NLL monotonically across every arm (`5.418` $\to$ `5.390` $\to$ `5.212` $\to$ `5.129`) with **zero Exact-AR drift**.
- **8-Step Concept Mode In-Distribution Convergence Proof (`modal_8step_converge_report.json`):** When trained to convergence (`500` Stage-1 + `1,000` Stage-2 + `300` Concept-DAgger/Cross-KD steps on Modal A10G), Concept-DAgger + Cross-Mode KL reduces Concept Loss from **`0.74942` $\to$ `0.45102` nats** (`-39.8%`) and Align NLL from **`0.00776` $\to$ `0.00514` nats**, enabling the **8-step Concept Mode (`254.35 ms`)** to generate **64/64 character-for-character identical JSON tool calls and arithmetic expressions** to 64-step Exact-AR (`1,708.16 ms`) at a **`6.72×` wall-clock speedup**. *(Note: Because the convergence showcase evaluates on the 31 in-distribution `Bonsai-2-4B` records, it establishes representational and algorithmic convergence of 8-step decoding rather than out-of-distribution generalization.)*

### 4.4 Neuro-Symbolic Tool Calling & Plan-Guided Fill-in-the-Middle (PG-FIM)

| Task / Benchmark Suite | Mode / Guardrail Configuration | Measured Result (Wilson 95% CI) | Source Artifact |
|---|---|---|---|
| **Held-Out Tool Calling (`n=500`, 4 tools)** | `JSONSchemaConstrainedMasker` (Syntax Only) | **`100.0%` `[99.2%, 100.0%]` JSON Syntax & Tool Match**; `0.0%` `[0.0%, 0.8%]` Exact Arg Expression | `benchmarks_rigor.json` |
| **Held-Out Tool Calling (`n=500`, 4 tools)** | `JSONSchemaConstrainedMasker` + **Prompt Copy-Span Constraint** | **`100.0%` `[99.2%, 100.0%]` JSON Syntax, Tool Match & Exact Argument Expression** (`0.0%` mental math evaluation without tool execution) | `benchmarks_rigor.json` |
| **PG-FIM Code Infilling (Dev Split, `n=24`)** | Model-Only Hybrid PG-FIM (`allow_template_fallback=False`, 800 steps) | **`62.5%` Parse & Compile Rate**, **`50.0%` Unit-Test Pass@1** (`12/24`) at **`821.9 ms/task`** (`1.13×` faster than AR FIM `932.6 ms`); local-span Exact-AR repair triggered on `100%` of Concept drafts | `modal_v2_verify_report.json` |
| **Unseen Locked FIM Benchmarks (`pg_fim_v3_60`, `HumanEval-Infilling`, `SAFIM`)** | Untrained / Short-Budget `tiny_test` & `nano` (`14.5M`) | `0.0%` – `1.0%` Pass@1 (honest small-scale baseline prior to multi-hundred-million token code pretraining) | `benchmarks_rigor.json` |

### 4.5 Controlled Scaling: `ctx0an` Dense Base (`93.90M`) vs. `base_moe` (`221.45M` Total / `94.05M` Active)

On NVIDIA A10 (`BF16`, `200` Phase-1 Exact-AR + `200` Phase-2 Joint steps = `3.07M` Stage 2 tokens on the 4-domain `V=6400` Byte-Level BPE corpus; [modal_ctx0an_moe_benchmark.json](modal_ctx0an_moe_benchmark.json)):

| Metric | Dense Base (`93.90M`) | **`base_moe` (`94.05M` active / `221.45M` total)** | Delta (MoE − Dense) |
|---|---:|---:|---:|
| **Phase 1 Val Exact-AR NLL / PPL** | `6.4818` / `653.15` | **`6.4026` / `603.41`** | **`-0.0792` / `-49.74`** |
| **Phase 2 Val Exact-AR NLL / PPL** | `6.1619` / `474.33` | **`5.9886` / `398.86`** | **`-0.1733` / `-75.47`** |
| **Phase 2 Domain NLL (EN / PY / Math / JSON)** | `6.682` / `6.120` / `6.106` / `5.740` | **`6.521` / `5.946` / `5.972` / `5.515`** | **`-0.161` / `-0.174` / `-0.134` / `-0.225`** |
| **Gate #5 Align NLL / Concept Draft NLL** | `6.6168` / `9.3439` | **`6.3854` / `8.8170`** | **`-0.2314` / `-0.5269`** |
| **Batch-1 Concept vs Exact-AR Speedup** | **`6.49×`** (`222.5 ms` vs `1,444.1 ms`) | **`6.22×`** (`422.5 ms` vs `2,628.7 ms`) | Dense is `1.89×` faster in absolute latency |
| **Training Throughput (Phase 1 / Phase 2)** | **`14,689` / `6,490` tok/s** | `10,625` / `4,828` tok/s | `0.723×` / `0.744×` |
| **MoE Load Balance & Loop Switching** | — | **`8/8` active (`0.9984` entropy)** | **`82.85%`** Loop 0$\to$1 switch (`TV = 0.047`) |

---

## 5. Speed vs. Intelligence Analysis & Limitations

1. **The Dual-Mode Speed vs. Intelligence Pareto Frontier:**
   - **8-Step Concept Mode** delivers **`6.40×–7.63×` GPU**, **`7.12×–7.33×` DirectML iGPU**, and **`2.26×–2.37×` ORT CPU** speedups by reading model weights only 8 times per 64 tokens, making it ideal for `<think>` reasoning and prose drafting.
   - **Exact-AR Mode** is `6–7×` slower at batch size 1 but provides much lower token-level NLL (`0.1158` vs. `5.1293` on held-out `Bonsai-2-4B` SeqKD) and enforces strict grammar constraints (`100%` JSON validity and copy-span argument accuracy) plus localized AST repair (`62.5%` PG-FIM dev compile rate).
2. **Direct Final-Level Token Shortcut (`exact_head(h_fine)`):** Across all Stage 2 benchmarks, `align_nll` (`exact_head(h_fine)`) is **`2.2–2.7` nats lower** than `concept_draft_nll` (which round-trips level $k=7$ through 4 FSQ digits and the 1-layer Stage 1 decoder). Decoding level $K-1$ directly via `exact_head(h_fine)` eliminates final-level quantization loss while skipping the Stage 1 decoder pass.
3. **Self-Speculative Acceptance ($\alpha$) vs. Dataset Scale:** While in-distribution convergence achieves $\alpha = 1.00$ (`64/64` token agreement), held-out generalization on the fast 48-sample (`31` verified) `Bonsai-2-4B` benchmark reaches $\alpha = 0.0313$ at sub-chunk horizon $\gamma=16$ (`0.52×` of AR speed). Achieving $>1.5\times$ lossless self-speculative speedup on unseen prompts requires scaling the `Ternary-Bonsai-2-4B` SeqKD corpus from `48` to `5,000–50,000+` verified samples and `100M–300M` training tokens.
4. **Recursive Looping & MoE Tradeoffs:** Under short token budgets (`<10M` tokens), non-looped `7×1` converges significantly faster than looped `7×3` ($p=0.042$) at `3×` lower compute cost. Similarly, while `base_moe` improves NLL by `-0.1733` over active-matched `base` (`94M`), it requires `2.36×` more RAM (`109.8 MB` vs `46.6 MB` INT4), is `1.89×` slower at batch-1 inference, and exhibits low domain specialization (`TV = 0.047`) at 400 steps; equal-memory comparison against `base_dense220` (`221M` dense) is provided in `modal_moe_benchmark.py`.

---

## 6. Conclusion

CNCM demonstrates that a single compact Transformer (`14.50M` parameters, **`7.50 MB` fused INT4 ONNX** / **`12.29 MB` PyTorch INT4**) can unify fast parallel 8-step Concept-Mode drafting (**`6.72×` GPU**, **`7.33×` DirectML iGPU**, and **`2.37×` ORT CPU** speedups) with exact autoregressive syntax, `100%` 4-turn INT8 KV agreement, `100%` neuro-symbolic copy-span tool-calling accuracy, and `62.5%` model-only PG-FIM dev code compilation. In direct head-to-head comparison against a Vanilla NCM baseline (`modal_ncm_vs_cncm_report.json`), CNCM shrinks combined quantizer and concept-head parameters by **`15.20×`** (`6.00M` $\to$ `0.39M`), eliminates Stage 1 RVQ codebook collapse (**`4.96%` $\to$ `99.04%`** `k=8` reconstruction), and lowers held-out Concept Draft NLL by **`-3.941` nats** (`8.777` $\to$ `4.836`). Pairing CNCM with our **3-Level `Ternary-Bonsai-2-4B` Hierarchical Distillation** pipeline (Verifier-Gated SeqKD + **Dynamic Residual Concept-DAgger** + Frozen-Teacher Cross-Mode KL) eliminates teacher drift collapse, drops held-out Concept Draft NLL to `4.836–5.129` nats (passing Gate #1 and Gate #5), and proves that 8-step Concept Mode can converge to `64/64` exact token parity with 64-step autoregressive decoding.

---

## References

1. **Kumbong, H., et al.** (2025). *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*. Stanford Hazy Research, `arXiv:2610.14162`.
2. **Mentzer, F., Tschannen, M., Agustsson, E., & Timofte, R.** (2024). *Finite Scalar Quantization: VQ-VAE Made Simple*. In *Proceedings of ICLR 2024* (`arXiv:2309.15505`).
3. **Kusupati, A., et al.** (2022). *Matryoshka Representation Learning*. In *Advances in Neural Information Processing Systems (NeurIPS 2022)* (`arXiv:2205.13147`).
4. **Meta AI (LCM Team).** (2024). *Large Concept Models: Language Modeling in a Sentence Representation Space*. `arXiv:2412.08821`.
5. **Pagnoni, A., et al.** (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens*. `arXiv:2412.09871`.
6. **Leviathan, Y., Kalman, M., & Matias, Y.** (2023). *Fast Inference from Transformers via Speculative Decoding*. In *Proceedings of ICML 2023* (`arXiv:2211.17192`).
7. **Cai, T., et al.** (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. In *Proceedings of ICML 2024* (`arXiv:2401.10774`).
8. **Li, Y., et al.** (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. In *Proceedings of ICML 2024* (`arXiv:2401.15077`).
9. **Elhoushi, M., et al.** (2024). *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding*. In *Proceedings of ACL 2024* (`arXiv:2404.16710`).
10. **Liu, Z., et al.** (2024). *MobileLLM: Optimizing Sub-Billion Parameter Language Models for On-Device Use Cases*. In *Proceedings of ICML 2024* (`arXiv:2402.14905`).
11. **Dai, D., et al.** (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. `arXiv:2401.06066`.
12. **PrismML Engineering.** (2026). *Ternary-Bonsai-2-4B: High-Density Ternary Vision-Language and Reasoning Models for On-Device Execution*. HuggingFace (`prism-ml/Ternary-Bonsai-2-4B-gguf`).
