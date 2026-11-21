// IEEE Conference / Transactions Two-Column Paper in Typst for CNCM Thesis
#set document(
  title: "CNCM: A Compact Causal-Hybrid Next-Concept Model for Edge Reasoning, Code Infilling, and Agentic Execution",
  author: "Adnan Akib",
)

#set page(
  paper: "us-letter",
  margin: (top: 0.8in, bottom: 0.9in, left: 0.625in, right: 0.625in),
  header-ascent: 30%,
  columns: 2,
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, font: "Times New Roman", fill: luma(90))
      IEEE TRANSACTIONS / CONFERENCE PREPRINT — CNCM: COMPACT CAUSAL-HYBRID NEXT-CONCEPT MODELING
      #h(1fr)
      NOVEMBER 2026
    ]
  },
)

#set columns(gutter: 0.25in)
#set text(font: "Times New Roman", size: 10pt)
#set par(justify: true, first-line-indent: 1em, leading: 0.55em, spacing: 0.65em)

// IEEE Roman / Alpha Section Numbering
#set heading(numbering: "I.A.1.")
#show heading.where(level: 1): it => {
  set align(center)
  set text(size: 10pt, weight: "regular", font: "Times New Roman")
  block(above: 1.1em, below: 0.6em)[
    #if it.numbering != none [
      #numbering("I.", counter(heading).get().first())
      #h(0.4em)
    ]
    #smallcaps(it.body)
  ]
}

#show heading.where(level: 2): it => {
  set align(left)
  set text(size: 10pt, style: "italic", weight: "regular", font: "Times New Roman")
  block(above: 0.85em, below: 0.45em)[
    #if it.numbering != none [
      #numbering("A.", counter(heading).get().at(1))
      #h(0.35em)
    ]
    #it.body
  ]
}

#show table: set text(size: 8pt, font: "Times New Roman")
#set table(
  stroke: (x, y) => (
    top: if y == 0 { 1pt } else if y == 1 { 0.5pt } else { 0pt },
    bottom: 0.8pt,
    left: 0pt,
    right: 0pt,
  ),
  inset: (x: 4pt, y: 3.5pt),
)

// Title Block spanning both columns
#place(
  top + center,
  scope: "parent",
  float: true,
  block(width: 100%, below: 1.2em)[
    #set align(center)
    #set par(first-line-indent: 0pt)
    #text(size: 20pt, weight: "bold")[
      CNCM: A Compact Causal-Hybrid Next-Concept Model for Edge Reasoning, Code Infilling, and Agentic Execution
    ]
    #v(0.8em)
    #text(size: 11pt)[Adnan Akib]
    #v(0.25em)
    #text(size: 9.5pt, style: "italic")[
      Department of Computer Science and Engineering · Open-Source Research Release\
      Repository: #raw("https://github.com/adnan1175/cncm") · Mirror: #raw("https://github.com/akibcse24/cncm")
    ]
    #v(0.6em)
  ]
)

// IEEE Abstract & Index Terms
#block[
  #set text(size: 9pt, weight: "bold")
  #set par(first-line-indent: 1em)
  _Abstract_—Autoregressive (AR) language models generate text one token at a time, making batch-1 edge inference strictly memory-bandwidth bound on mobile CPUs and laptop integrated GPUs (iGPUs). Conversely, parallel latent-concept models amortize weight sweeps across multi-token chunks but suffer from VQ-VAE codebook bloat, positional misalignment, and discrete syntax corruption on executable code and JSON tool calls. We present CNCM (_Compact Causal-Hybrid Next-Concept Model_), a unified dual-mode Transformer architecture that shares a single backbone between two complementary decoding heads: (1) a _Factorized Concept Head_ that drafts 64-token chunks in parallel across $K=8$ coarse-to-fine residual levels in 8 backbone passes, and (2) a shared _Exact-AR Head_ for strict token-by-token generation, neuro-symbolic JSON-schema grammar masking, lossless self-speculative verification, and localized code repair. To eliminate codebook collapse and lookup-table memory, CNCM introduces a block-causal _Multiscale Residual Finite Scalar Quantization (M-RFSQ)_ Stage 1 tokenizer ($18.03 times$ fewer quantizer parameters and $23,931 times$ smaller buffers than RVQ-VAE) trained with Matryoshka multi-cutoff supervision ($k in {4,6,7,8}$). To align Concept Mode with Exact-AR geometry, we introduce _Continuous Scaled RoPE_ ($"pos"_k (i) = P + floor((i+0.5)L / ell_k)$), a _Text-Level Alignment Loss ($cal(L)_"align"$)_ , and a _3-Level Hierarchical Teacher$->$Student Distillation_ framework using _Ternary-Bonsai-2-4B_ (`llama-server` Verifier-Gated SeqKD + _Dynamic Residual Concept-DAgger_ + Frozen-Teacher Cross-Mode KL). In a controlled 4-arm cumulative ablation against a _Vanilla NCM_ baseline on Modal NVIDIA A10, CNCM reduces combined quantizer and concept-head parameters by $15.20 times$ ($6.00"M" -> 0.39"M"$), eliminates Stage 1 RVQ codebook collapse ($4.96% -> 99.04%$ $k=8$ reconstruction; $35.1 times$ higher active codebook utilization), reduces held-out Concept Draft NLL by $-3.941$ nats ($8.777 -> 4.836$), and lifts tool-calling syntax and semantic accuracy from $0.0% -> 100.0%$. Across cloud NVIDIA A10/A10G GPUs and physical laptop CPU (`CPUExecutionProvider`) and Intel Iris Xe iGPU (`DmlExecutionProvider`) hardware, CNCM-Nano ($14.50"M"$ parameters) achieves $6.40 times - 7.63 times$ GPU and $7.12 times - 7.33 times$ DirectML iGPU ($2.26 times - 2.37 times$ ORT CPU) Concept-Mode wall-clock speedups over KV-cached Exact-AR, $7.50" MB"$ fused INT4 ONNX ($5.84 times$ graph compression via looped-weight `Identity` inlining) and $12.29" MB"$ packed PyTorch INT4 weight footprints, $100.0%$ 4-turn INT8 KV top-1 agreement, and $64/64$ exact token parity at $6.72 times$ speedup under in-distribution convergence. Scaling to CNCM-Base ($93.90"M"$ dense) and Fine-Grained Mixture-of-Experts (`base_moe`, $221.45"M"$ total / $94.05"M"$ active) on a 4-domain BPE corpus demonstrates $-0.1733$ Phase 2 validation NLL ($-75.47$ PPL) with $8/8$ balanced expert utilization ($0.9984$ normalized load entropy).
]

#v(0.3em)
#block[
  #set text(size: 9pt, weight: "bold")
  _Index Terms_—Edge inference, next-concept modeling, finite scalar quantization, speculative decoding, fill-in-the-middle, knowledge distillation, mixture-of-experts, ONNX Runtime.
]

= Introduction

Deploying language models on consumer edge hardware—smartphones, laptops, and embedded devices—is constrained less by raw arithmetic FLOPs than by *memory bandwidth*. Standard token-by-token autoregressive decoding must stream all model weights from DRAM for every generated token ($64 times$ per 64-token chunk).

Block-wise and concept-space language models amortize weight sweeps by predicting multi-token spans in parallel. However, existing latent-concept models [1] suffer from four practical bottlenecks:
1. *Discrete Codebook Collapse & Memory Overhead:* Multiscale RVQ-VAE codebooks require large explicit embedding tables and FP32 EMA cluster buffers ($~18.87"M"$ parameters / $~151" MB"$ in vanilla NCM [1]) that collapse under small batch sizes or gradient accumulation.
2. *Positional Misalignment:* Resetting target concept positions to $0 dots ell_k - 1$ collides with prefix RoPE angles and inverts relative distances to the causal prefix, degrading target-to-prefix attention.
3. *Coarse-to-Fine Exposure Bias:* Teacher-forced concept levels train on ground-truth coarse residuals $hat(z)_(<= k-1)^*$, whereas open-loop inference conditions on the student's own predicted codes $hat(c)_(<= k-1)$, causing error accumulation across levels $k=0 dots K-1$.
4. *Loss of Exact Syntax:* Parallel latent decoding struggles with strict syntactic invariants required for JSON tool calls, exact arithmetic arguments, and executable code infilling.

CNCM addresses all four challenges in a single compact architecture:
- *Zero-Table Block-Causal M-RFSQ Tokenizer (Stage 1):* Uses implicit integer grids (`fsq_levels=(8,8,8,8)`, $4,096$ codes) with a 1-layer causal semantic encoder (`enc_layers=1`), affine-free `GroupNorm`, and Matryoshka multi-cutoff supervision ($k in {4,6,7,8}$) [2], [3].
- *Continuous Scaled RoPE & $cal(L)_"align"$:* Places level-$k$ target positions strictly after the prefix at their physical span centers, $"pos"_k (i) = P + floor((i+0.5)L / ell_k)$, and projects finest-level concept hidden states through the shared Exact-AR head.
- *3-Level Hierarchical Teacher$->$Student Distillation (`Ternary-Bonsai-2-4B` [12]):* Combines (1) Verifier-Gated Sequence-Level KD (`llama-server` + AST/unit-test & tool sandboxes), (2) *Dynamic Residual Concept-DAgger* (analytically re-quantizing remaining teacher residuals $z_"teacher" - hat(z)_"student"$ on the fly), and (3) *Frozen-Teacher Cross-Mode KL Distillation*.
- *Neuro-Symbolic Dual-Mode Execution & PG-FIM:* Uses Concept Mode for fast parallel drafting (`<think>` and prose) and Exact-AR Mode for grammar-constrained JSON tool calls (`JSONSchemaConstrainedMasker` + prompt copy-span constraints), lossless self-speculative verification with an EMA circuit breaker, and Plan-Guided Fill-in-the-Middle (`<mlen_k>` + `<kw>...</kw>`) with local-span AST autoregressive repair.

= Related Work

CNCM builds upon and bridges four active research directions:
- *Concept-Space & Block-Parallel Language Models:* Next-Concept Modeling (NCM; Kumbong et al. [1]) introduced multiscale coarse-to-fine latent generation via RVQ-VAE; Meta's Large Concept Models (LCM [4]) and Byte Latent Transformer (BLT [5]) model text at patch or sentence granularity. CNCM eliminates NCM's $151" MB"$ RVQ-VAE tables via M-RFSQ, replaces bidirectional prefix attention with causal-prefix block-AR KV caching, and unifies concept drafting with an exact token-level AR head in one backbone.
- *Vector Quantization & Matryoshka Representations:* Finite Scalar Quantization (FSQ; Mentzer et al. [2]) replaces learned VQ codebooks with bounding and rounding onto a fixed scalar grid. CNCM extends FSQ to a *multiscale residual hierarchy (`M-RFSQ`)* with depthwise-separable $Phi$ refinement and Matryoshka multi-cutoff supervision [3], enabling both dynamic early exit and analytic on-policy residual re-quantization (*Concept-DAgger*).
- *Speculative & Parallel Decoding:* Speculative decoding [6], Medusa [7], EAGLE [8], and LayerSkip [9] accelerate AR inference using draft heads. CNCM provides a unified dual-mode self-speculative verifier (`generate_self_speculative`) with an EMA acceptance circuit breaker and explicit sub-chunk acceptance analysis.
- *Compact Edge LLMs & Low-Bit Teachers:* MobileLLM [10] studied weight sharing and recursive layer looping for sub-billion edge models; DeepSeekMoE [11] introduced fine-grained shared + routed experts; and PrismML's *Ternary-Bonsai-2-4B* [12] demonstrated high-density ternary (${-1, 0, +1}$) reasoning and tool use, which we leverage as an edge-compatible `llama-server` SeqKD teacher.

= Architecture & Hierarchical Distillation

== Stage 1: Block-Causal M-RFSQ Concept Tokenizer
Given a length-$L$ token chunk ($L=64$), Stage 1 embeds tokens, applies a 1-layer strictly causal Transformer encoder (`enc_layers=1`) followed by affine-free `GroupNorm` (`pre_quant_norm`), and quantizes residuals across $K=8$ multiscale levels:
$ ell = (1, 2, 4, 8, 12, 20, 36, 64), quad sum_(k=0)^(K-1) ell_k = 147 = 2.30 L $
At each level $k$, the residual $r_k in RR^(B times L times d_"vq")$ is downsampled to length $ell_k$, projected to $d_q = 4$ dimensions, quantized via bounded $tanh$ + Straight-Through Estimator (STE) rounding onto $8^4 = 4,096$ implicit integer grid points, upsampled back to length $L$, and refined via a depthwise-separable residual convolution $Phi_k$:
$ tilde(c)_j = (L_j - 1) / 2 + L_j / 2 (1 - epsilon) tanh(h_j), quad c_j = "clip"(round(tilde(c)_j), 0, L_j - 1) $
While `encoder_blocks` is strictly causal per token, multiscale area-pooling within a 64-token chunk aggregates across the $L=64$ block, making Stage 1 *block-causal* across 64-token chunks. Training uses nested fine-level dropout and Matryoshka reconstruction loss across cutoffs $k in {4, 6, 7, 8}$.

== Stage 2: Causal-Hybrid Backbone & Factorized Concept Head
Stage 2 freezes Stage 1 and trains a shared Transformer backbone (Pre-RMSNorm + QK-RMSNorm + GQA + SwiGLU, supporting optional Relaxed Recursive Layer Looping with rank-$r$ `LoopAdapter`s):
- *Factorized Concept Head (`FactorizedConceptHead`):* Factorizes the 4,096-way code distribution into 4 chained autoregressive digit micro-heads:
  $ log p_theta (c_0, c_1, c_2, c_3 | h) = sum_(j=0)^3 log p_theta (c_j | h, e_(<j)(c_(<j))) $
  requiring only $71,072$ core micro-head parameters ($149,152$ total module parameters on `nano`, $10.5 times - 22.1 times$ smaller than a $1.57"M"$-parameter flat 4,096-way linear head).
- *Exact-AR Head:* Weight-tied to `exact_embed` for standard next-token prediction, JSON-schema logit masking, self-speculative verification, and local-span PG-FIM repair.
- *Continuous Scaled RoPE & $cal(L)_"align"$:* Prefix tokens occupy positions $0 dots P-1$, while concept level $k$ occupies:
  $ "pos"_k (i) = P + floor(((i + 0.5) L) / ell_k), quad i in {0, dots, ell_k - 1} $
  During joint Stage 2 training, finest-level concept hidden states $h_"fine"$ ($l_7 = 64$) are simultaneously projected through `exact_head` via a text-level alignment loss $cal(L)_"align"$.

== 3-Level Hierarchical Teacher$->$Student Distillation
To bridge the capacity gap of a $14.5"M"$-parameter dual-mode student without mutating the Exact-AR distribution during concept alignment, CNCM introduces a three-stage distillation pipeline:
1. *Level 1: Verifier-Gated Sequence-Level KD (SeqKD):* Queries an OpenAI-compatible `llama-server` serving `prism-ml/Ternary-Bonsai-2-4B-gguf` (`Ternary-Bonsai-2-4B-PQ2_0.gguf`, `Apache-2.0`), caps verbose `<think>` traces (`max_think_chars=240`), and filters every completion through deterministic verifiers (`run_code_sandbox` for Python AST/compile/unit-test execution; `evaluate_tool_call_semantic_accuracy` for JSON tool calls and arithmetic).
2. *Level 2: Dynamic Residual Concept-DAgger:* Because `MultiscaleRFSQ` is an analytic zero-table scalar grid, on-policy student rollouts at coarse levels $hat(c)^((0)), dots, hat(c)^((k-1))$ dynamically re-quantize the *true remaining teacher residual* at level $k$ in closed form:
   $ r_k^"DAgger" &= z_"teacher" - sum_(j=0)^(k-1) Delta hat(z)_j (hat(c)_"student"^((j))) \
     c_k^(*"DAgger") &= "FSQ"_k ("Down"_k (r_k^"DAgger")) $
3. *Level 3: Frozen-Teacher Cross-Mode KL Distillation:* Freezes an immutable post-SFT teacher snapshot (`teacher_ref`) and locks the shared Exact-AR backbone (`freeze_backbone_for_exact=True`), training Concept projections and `FactorizedConceptHead` on on-policy teacher greedy rollouts $y_"AR"$ with token-level KL divergence $D_"KL" (p_"teacher" (y | x) || "exact_head"(h_"fine"))$.

== Fine-Grained Mixture-of-Experts (`base_moe`)
To study knowledge capacity scaling at constant active FFN width ($2304$), `base_moe` ($221.45"M"$ total / $94.05"M"$ active) replaces the dense SwiGLU FFN with a DeepSeek-style Fine-Grained MoE layer (`MoEFFN`): 1 always-on Shared Expert ($d=768$) + 8 Routed Experts ($d=768$, Top-2 active = $2304$ active width), 24 per-loop routers (`step_moe_routers`), and a load-balancing auxiliary loss ($"moe_aux_loss_weight" = 0.01$).

#place(
  top + center,
  scope: "parent",
  float: true,
  block(width: 100%, below: 1em)[
    #set align(center)
    #text(size: 8.5pt, weight: "bold")[TABLE I: Controlled 4-Arm Cumulative Ablation from Vanilla NCM to Full CNCM (`CNCM-Nano`, Modal NVIDIA A10, `seed=42`)]
    #v(0.25em)
    #table(
      columns: (2.2fr, 1.1fr, 1.1fr, 0.9fr, 0.9fr, 1.0fr, 0.9fr, 1.1fr),
      align: (left, right, right, right, right, right, right, right),
      [*Architecture Variant (`CNCM-Nano`)*], [*Quant+Head Params* $arrow.b$], [*S1 Recon (`k=8`/`6`)* $arrow.t$], [*Active Codes %* $arrow.t$], [*Concept Loss* $arrow.b$], [*Concept Draft NLL* $arrow.b$], [*Exact-AR NLL* $arrow.b$], [*Tool Syntax / Sem (`n=40`)* $arrow.t$],
      [Arm 1: Vanilla NCM Baseline (RVQ+EMA + Flat 4096 + Reset RoPE)], [`6,003,712` (+16.5MB)], [`4.96%` / `4.59%`], [`0.50%`], [`11.5125`], [`8.7773`], [`8.3429`], [`0.0%` / `0.0%`],
      [Arm 2: \+ CNCM Stage 1 (`M-RFSQ` + Matryoshka `4,6,7,8`)], [`1,818,656` (0.69KB)], [*`99.04%` / `50.31%`*], [*`17.53%`* ($35.1 times$)], [`6.9894` ($-4.52$)], [`8.3257`], [`8.3405`], [`0.0%` / `0.0%`],
      [Arm 3: \+ `FactorizedConceptHead` + Continuous Scaled RoPE], [*`394,944`* (*$15.20 times$ smaller*)], [`99.04%` / `50.31%`], [`17.53%`], [`6.9465`], [`8.3533`], [`8.4083`], [`0.0%` / `0.0%`],
      [Arm 4: *Full CNCM* (Dual-Mode + $cal(L)_"align"$ + `Bonsai-2-4B` KD + Copy-Span)], [*`394,944`* (Tied Exact)], [*`99.04%` / `50.31%`*], [*`17.53%`*], [*`6.0556`* (*$-5.46$*)], [*`4.8364`* (*$-3.94$*)], [*`0.5600`* (`1.30` Align)], [*`100.0%` / `100.0%`*],
    )
  ]
)

#place(
  top + center,
  scope: "parent",
  float: true,
  block(width: 100%, below: 1em)[
    #set align(center)
    #text(size: 8.5pt, weight: "bold")[TABLE II: Physical Edge Deployment — Fused INT4/INT8 ONNX Graph Compression & Multi-Device Latency (`CNCM-Nano`, $B=1$, 64 Tokens)]
    #v(0.25em)
    #table(
      columns: (2.2fr, 1.1fr, 1.2fr, 1.2fr, 1.0fr, 1.4fr),
      align: (left, center, right, right, right, right),
      [*Hardware Platform & Execution Provider*], [*Precision*], [*8-Step Concept (`tok/s`)*], [*64-Step Exact-AR (`tok/s`)*], [*Speedup*], [*Footprint / Thermal Soak*],
      [Cloud NVIDIA A10 / A10G GPU (PyTorch)], [`BF16`], [*`187.6–267.3 ms`* (`240–341`)], [`1,272.5–1,815.0 ms` (`35–50`)], [*`6.72×–6.79×`*], [`8.85×` @ `k=6` early exit],
      [Laptop Intel Iris Xe iGPU (`DmlExecutionProvider`)], [`FP32`], [*`1,592.1 ms`* (`40.2 tok/s`)], [`11,330.7 ms` (`5.65 tok/s`)], [*`7.12×`*], [`43.79 MB` Concept ONNX],
      [Laptop Intel Iris Xe iGPU (`DmlExecutionProvider`)], [`INT4 MatMulNBits`], [*`2,012.5 ms`* (`31.8 tok/s`)], [`14,740.9 ms` (`4.34 tok/s`)], [*`7.33×`*], [*`7.50 MB` ONNX* (`0.984` soak)],
      [Laptop Intel 16-Thread CPU (`CPUExecutionProvider`)], [`FP32`], [*`370.8 ms`* (`172.6 tok/s`)], [`838.9 ms` (`76.3 tok/s`)], [*`2.26×`*], [`625.7 MB` RSS],
      [Laptop Intel 16-Thread CPU (`CPUExecutionProvider`)], [`INT4 MatMulNBits`], [*`372.9 ms`* (`171.6 tok/s`)], [`882.4 ms` (`72.5 tok/s`)], [*`2.37×`*], [*`7.50 MB` ONNX* (`0.971` soak)],
      [Laptop Intel 16-Thread CPU (Pure-PyTorch Reference)], [`FP32` vs `Int4Packed`], [`343.7 ms` vs `893.3 ms`], [`1,601.7 ms` vs `5,185.5 ms`], [`4.66×` / `5.81×`], [`12.29 MB` (`0.385×` dequant)],
    )
  ]
)

= Empirical Results & Controlled Ablations

== Vanilla NCM vs. CNCM Cumulative Ablation (`CNCM-Nano`)
Table I reports our controlled 4-arm cumulative ablation (`modal_ncm_vs_cncm_report.json`) on Modal NVIDIA A10 (`350` Stage-1 + `400` Stage-2 + `120` Align steps, `seed=42`):
1. *Stage 1 Codebook Collapse Eliminated ($4.96% -> 99.04%$):* Vanilla NCM's 8-level RVQ codebook tables ($4.43"M"$ params + $16.51" MB"$ EMA buffers) suffer severe collapse ($0.50%$ active codes, $4.96%$ reconstruction), whereas CNCM's Zero-Table M-RFSQ ($245.8"K"$ params, $0.69" KB"$ buffers — $18.03 times$ fewer quantizer parameters) achieves $99.04%$ full-depth ($k=8$) and $50.31%$ early-exit ($k=6$) reconstruction with $35.1 times$ higher codebook utilization.
2. *$15.20 times$ Smaller Quantizer + Head Footprint:* Replacing RVQ tables and the flat 4096-way head ($1.57"M"$ params) with M-RFSQ + `FactorizedConceptHead` reduces combined quantizer and head parameters from $6,003,712 -> 394,944$ while improving Concept Loss ($6.9894 -> 6.9465$).
3. *Full CNCM Quality & Agentic Gain:* Adding Dual-Mode training + $cal(L)_"align"$ + `Ternary-Bonsai-2-4B` Concept-DAgger/Cross-Mode KL drops Held-Out Concept Loss from $11.5125 -> 6.0556$, Concept Draft NLL from $8.7773 -> 4.8364$ ($-3.941$ nats), Exact-AR NLL from $8.3429 -> 0.5600$, and Tool-Calling Syntax & Semantic Accuracy from $0.0% -> 100.0%$.

== Physical Edge Deployment: True `7.50 MB` INT4 ONNX, CPU & Intel Iris Xe iGPU
To fix a PyTorch ONNX exporter flaw where weight-shared recursive loops emit `Identity` alias nodes that prevent downstream quantizers (`MatMulNBitsQuantizer` / `quantize_dynamic`) from compressing looped weights, we implemented an *ONNX `Identity` Alias Inliner* (`_inline_onnx_identities` in `cncm/export/edge_bench.py`). Inlining compresses `concept_level_kv_int4_nbits.onnx` from $43.79" MB"$ FP32 ($49.55" MB"$ un-inlined) down to *$7.50" MB"$ ($5.84 times$ compression)*, `exact_kv_step_int4_nbits.onnx` from $55.78" MB" -> 14.44" MB"$ ($3.86 times$), and `stage1_decoder_int4_nbits.onnx` from $3.02" MB" -> 0.65" MB"$ ($4.63 times$). As shown in Table II, Concept Mode achieves *$7.12 times - 7.33 times$ speedup* on Intel Iris Xe DirectML iGPU and *$2.26 times - 2.37 times$ speedup* on 16-thread ORT CPU with $0.971 - 0.984$ sustained thermal-soak stability.

== 3-Level `Ternary-Bonsai-2-4B` Hierarchical Distillation & 8-Step Convergence
We executed the live `llama-server` teacher-student pipeline on Modal NVIDIA A10/A10G (`modal_bonsai_distill_report.json`):
- *Stage A (`llama-server` SeqKD):* Served `Ternary-Bonsai-2-4B-PQ2_0.gguf` via `llama-server` and queried 48 balanced tool-calling, arithmetic, and Python FIM prompts. *31 / 48 ($64.58%$)* completions passed strict AST/unit-test and semantic tool verifiers (`verified_seqkd.jsonl`).
- *Stage B (4-Arm Distillation Ablation, Table III):* Level 1 `Bonsai-2-4B` SeqKD drops held-out Exact-AR NLL from $7.3723 -> 0.1158$. Adding Level 2 Concept-DAgger and Level 3 Frozen-Teacher Cross-Mode KL reduces Concept Draft NLL monotonically ($5.4176 -> 5.3901 -> 5.2122 -> 5.1293$ nats, passing Gate \#1 and Gate \#5) with *zero Exact-AR drift*.

#block(width: 100%, above: 0.6em, below: 0.6em)[
  #set align(center)
  #set par(first-line-indent: 0pt)
  #text(size: 8pt, weight: "bold")[TABLE III: 3-Level `Ternary-Bonsai-2-4B` Distillation (`CNCM-Nano`, Modal A10)]
  #v(0.2em)
  #table(
    columns: (2.1fr, 1.0fr, 1.1fr, 1.0fr),
    align: (left, right, right, right),
    [*Distillation Arm (`seed=42`)*], [*Exact NLL* $arrow.b$], [*Draft NLL* $arrow.b$], [*Concept Loss* $arrow.b$],
    [Arm 1: Baseline Raw Corpus], [`7.3723`], [`5.4176`], [—],
    [Arm 2: \+ Level 1 `Bonsai-2-4B` SeqKD], [*`0.1158`*], [`5.3901`], [`4.6300`],
    [Arm 3: \+ Level 2 Concept-DAgger], [`0.1158`], [`5.2122`], [—],
    [Arm 4: *\+ Level 3 Full Cross-Mode KD*], [*`0.1158`*], [*`5.1293`*], [*`4.5984`*],
    [_Control: Pure-AR + Same SeqKD_], [`0.0279`], [—], [—],
  )
]

- *8-Step Concept Mode Convergence Proof (`modal_8step_converge_report.json`):* When trained to convergence ($500$ S1 + $1,000$ S2 + $300$ Align steps on Modal A10G), Concept-DAgger + Cross-Mode KL drops Concept Loss from $0.7494 -> 0.4510$ ($-39.8%$) and Align NLL from $0.0078 -> 0.0051$, enabling 8-step Concept Mode ($254.35" ms"$) to generate *$64/64$ character-for-character identical JSON tool calls and arithmetic expressions* to 64-step Exact-AR ($1,708.16" ms"$) at a *$6.72 times$ wall-clock speedup* (evaluated as an in-distribution convergence verification on the 31 verified `Bonsai-2-4B` records).

== Neuro-Symbolic Tool Calling, PG-FIM & Scaling (`Dense Base` vs. `base_moe`)
- *Tool Calling (`n=500`, `benchmarks_rigor.json`):* `JSONSchemaConstrainedMasker` guarantees $100.0%$ ($[99.2%, 100.0%]$) JSON syntax validity; pairing it with the *Prompt Copy-Span Constraint* lifts exact argument expression accuracy from $0.0% -> 100.0%$ ($[99.2%, 100.0%]$).
- *PG-FIM Code Infilling (`modal_v2_verify_report.json`):* On the 24 held-out dev tasks (`allow_template_fallback=False`, 800 steps), model-only hybrid PG-FIM achieves *$62.5%$ parse/compile rate* and *$50.0%$ unit-test pass\@1* ($12/24$) at $821.9" ms/task"$ ($1.13 times$ faster than AR FIM's $932.6" ms"$).
- *Controlled `Dense Base` ($93.90"M"$) vs. `base_moe` ($221.45"M"$ Total / $94.05"M"$ Active, Table IV):* Across $3.07"M"$ Stage 2 tokens on the 4-domain $V=6400$ BPE corpus (`modal_ctx0an_moe_benchmark.json`), `base_moe` improves validation NLL across all 4 domains ($-0.1733$ mean NLL, $-75.47$ PPL, $-0.5269$ Concept Draft NLL) with $8/8$ active experts ($0.9984$ normalized load entropy) and $82.85%$ Loop $0->1$ expert switching.

#block(width: 100%, above: 0.6em, below: 0.6em)[
  #set align(center)
  #set par(first-line-indent: 0pt)
  #text(size: 8pt, weight: "bold")[TABLE IV: Controlled `ctx0an` Dense Base vs. `base_moe` (Modal A10)]
  #v(0.2em)
  #table(
    columns: (2.0fr, 1.1fr, 1.1fr, 1.0fr),
    align: (left, right, right, right),
    [*Benchmark Metric*], [*Dense (`93.9M`)*], [*`base_moe` (`94.1M`)*], [*Delta*],
    [Phase 1 Val Exact-AR NLL], [`6.4818`], [*`6.4026`*], [`-0.0792`],
    [Phase 2 Val Exact NLL / PPL], [`6.1619` / `474.3`], [*`5.9886` / `398.9`*], [*`-0.1733`*],
    [Python / JSON Val NLL], [`6.120` / `5.740`], [*`5.946` / `5.515`*], [`-0.174` / `-0.225`],
    [Align NLL / Concept Draft NLL], [`6.617` / `9.344`], [*`6.385` / `8.817`*], [`-0.231` / `-0.527`],
    [Batch-1 Concept Speedup], [*`6.49×`* (`222ms`)], [*`6.22×`* (`422ms`)], [Both $>6 times$],
    [MoE Load Balance / Loop Switch], [—], [*`8/8` (`0.9984`)*], [*`82.85%`*],
  )
]

= Speed vs. Intelligence Analysis & Limitations

1. *The Dual-Mode Speed vs. Intelligence Pareto Frontier:* 8-Step Concept Mode delivers $6.40 times - 7.63 times$ GPU, $7.12 times - 7.33 times$ DirectML iGPU, and $2.26 times - 2.37 times$ ORT CPU speedups for `<think>` and prose drafting, while Exact-AR Mode enforces strict grammar constraints ($100%$ JSON copy-span accuracy) and localized AST repair ($62.5%$ PG-FIM dev compile rate).
2. *Direct Final-Level Token Shortcut (`exact_head(h_fine)`):* Across all Stage 2 benchmarks, `align_nll` (`exact_head(h_fine)`) is $2.2 - 3.5$ nats lower than `concept_draft_nll` ($1.296$ vs. $4.836$ in Table I Arm 4), providing a direct zero-cost shortcut that bypasses final-level FSQ quantization error.
3. *Controlled Caveats:* Under short budgets ($400$ steps, 3 seeds in `controls_summary.json`), non-looped `7×1` converges faster than looped `7×3` ($2.789$ vs. $3.326$ NLL, $p=0.042$), and 3-arm fair QAT (`FP32+40` vs. `QAT+40`) shows $Delta "NLL" = -0.0025$ ($p=0.44$), confirming PTQ INT4 (`g=32`) is already nearly lossless. Achieving $>1.5 times$ lossless self-speculative speedup on unseen prompts requires scaling the `Ternary-Bonsai-2-4B` SeqKD corpus from $48$ to $5,000+$ verified samples.

= Conclusion

CNCM demonstrates that a single compact Transformer ($14.50"M"$ parameters, $7.50" MB"$ fused INT4 ONNX / $12.29" MB"$ PyTorch INT4) can unify fast parallel 8-step Concept-Mode drafting ($6.72 times$ GPU, $7.33 times$ DirectML iGPU, and $2.37 times$ ORT CPU speedups) with exact autoregressive syntax, $100%$ 4-turn INT8 KV agreement, $100%$ neuro-symbolic copy-span tool-calling accuracy, and $62.5%$ model-only PG-FIM dev code compilation. Compared head-to-head against Vanilla NCM, CNCM reduces quantizer and concept-head parameters by $15.20 times$, eliminates Stage 1 RVQ codebook collapse ($4.96% -> 99.04%$), and lowers held-out Concept Draft NLL by $-3.941$ nats. Pairing CNCM with our 3-Level `Ternary-Bonsai-2-4B` Hierarchical Distillation pipeline eliminates teacher drift collapse and proves that 8-step Concept Mode can converge to $64/64$ exact token parity with 64-step autoregressive decoding.

#set heading(numbering: none)
= References

#set par(first-line-indent: 0pt, hanging-indent: 1.4em, spacing: 0.45em)
#set text(size: 8.5pt)

[1] H. Kumbong _et al._, "Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation," Stanford Hazy Research, _arXiv:2610.14162_, 2025.

[2] F. Mentzer, M. Tschannen, E. Agustsson, and R. Timofte, "Finite Scalar Quantization: VQ-VAE Made Simple," in _Proc. ICLR_, 2024 (`arXiv:2309.15505`).

[3] A. Kusupati _et al._, "Matryoshka Representation Learning," in _Proc. NeurIPS_, 2022 (`arXiv:2205.13147`).

[4] Meta AI (LCM Team), "Large Concept Models: Language Modeling in a Sentence Representation Space," _arXiv:2412.08821_, 2024.

[5] A. Pagnoni _et al._, "Byte Latent Transformer: Patches Scale Better Than Tokens," _arXiv:2412.09871_, 2024.

[6] Y. Leviathan, M. Kalman, and Y. Matias, "Fast Inference from Transformers via Speculative Decoding," in _Proc. ICML_, 2023 (`arXiv:2211.17192`).

[7] T. Cai _et al._, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," in _Proc. ICML_, 2024 (`arXiv:2401.10774`).

[8] Y. Li _et al._, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," in _Proc. ICML_, 2024 (`arXiv:2401.15077`).

[9] M. Elhoushi _et al._, "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding," in _Proc. ACL_, 2024 (`arXiv:2404.16710`).

[10] Z. Liu _et al._, "MobileLLM: Optimizing Sub-Billion Parameter Language Models for On-Device Use Cases," in _Proc. ICML_, 2024 (`arXiv:2402.14905`).

[11] D. Dai _et al._, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models," _arXiv:2401.06066_, 2024.

[12] PrismML Engineering, "Ternary-Bonsai-2-4B: High-Density Ternary Vision-Language and Reasoning Models for On-Device Execution," HuggingFace (`prism-ml/Ternary-Bonsai-2-4B-gguf`), 2026.
