# Future CNCM Architecture Updates (`future_cncm_update.md`)

> **Status:** `🔵 Planned / Research Roadmap` — All items below are future architectural upgrades for **CNCM (Paper 1 / v2+)** and are kept strictly separate from verified `✅` claims in `CLAIMS.md` and `ARCHITECTURE.md`.
>
> **Priority Note:** **Paper 2 (`cncm-rs` Edge Runtime)** and **Paper 3 (Edge Agent Runtime)** remain **Zero-Training** and ship first on existing open GGUF models. These CNCM upgrades run in parallel / post-v1.0.

---

## 1. Fast Zero-Backprop Donor Surgery & Plug-In `CNCM-DeltaNet` Sidecar (`Path A → Path B`)

This is the centerpiece upgrade that cuts donor surgery cost by **10×–30×** (from ~30–45 min on a paid 48 GB L40S down to **~5–10 min on a free 15 GB T4**, scaling cleanly up to `1.5B–3B` donors like `Qwen2.5-Coder-1.5B` and `Qwen3-1.7B`) while keeping Exact-AR **100% bit-exact untouched**.

### A. Fast Zero-Backprop Surgery Pipeline (`universal_surgery.py`)
1. **SVD-Projected Frozen-Embedding Stage 1 (`18×` Smaller Tokenizer):**
   - Replace training a brand-new `49K–152K × d_vq` (~27M) Stage 1 embedding table with a closed-form truncated SVD projection ($P \in \mathbb{R}^{H \times d_{\text{vq}}}$, ~2s on CPU) of the donor's already-trained `embed_tokens.weight`.
   - Train only the tiny **~1.5M** `MultiscaleRFSQ` convs + 1-layer decoder (converges in ~300–400 steps, ~2 min on T4).
2. **Offline Multi-Layer Hidden + Top-K Logit Cache:**
   - Run the frozen donor **once** (`torch.no_grad()`, optionally in 4-bit NF4) over 10M–20M tokens, saving fused multi-layer hidden states (**`L16 + L24 + L32`**, EAGLE-3 style) + `top_k=16` teacher logits to a compact `memmap` (~1–2 GB).
   - Stage 2 + Cross-Mode KL then trains with **zero donor forward/backward passes** (**15×–30× faster**).
3. **Closed-Form Step-0 Ridge Warm-Start:**
   - Initialize `FactorizedConceptHead` ($c_0$) via closed-form Ridge regression (`torch.linalg.lstsq`) on cached hidden states before Step 1.
4. **Universal Config-Driven Auto-Surgeon (`universal_surgery.py`):**
   - One metadata-driven CLI supporting `llama`, `smollm2`, `qwen2`, `qwen3`, `mistral`, `gemma` (QKV biases, QK-Norm, tied/untied `lm_head`, RoPE style) with automatic bit-exact parity check (`--donor <repo> --mode sidecar --svd-stage1`).

### B. `CNCM-DeltaNet` — Chunk-64 WY-Parallel & Dual-Phase Read-Only State
Custom linear attention designed specifically for CNCM (zero Triton/CUDA lock-in, full bidirectional target-scale parity):
- **Chunk-64 WY-Parallel Form:** Aligns DeltaNet's chunk size with CNCM's `L = 64` (or `L = 16/32`) via WY/Householder matrix representation so every chunk runs as **100% standard `matmul`** across Pure PyTorch, Kaggle TPU (XLA), DirectML, MPS, and `cncm-rs` SIMD/wgpu.
- **Dual-Phase Read-Only State Contract:**
  - *Phase A (Causal Prefix):* Updates fixed-size FP32 recurrent state $S_{\text{prefix}} \in \mathbb{R}^{H \times d_k \times d_v}$ ($O(1)$ RAM) using L2-normalized $q, k$ and bounded $\beta = 2\sigma(\cdot)$.
  - *Phase B (Bidirectional Target Scales `L0..L7`):* Queries $S_{\text{prefix}}$ **read-only** ($q_i^\top S_{\text{prefix}}$, **0 MB target state growth**) + local bidirectional mixing ($\ell_k \le 64$). Commits via 1 WY pass after decode (**exact train/infer parity** `max_abs_diff < 1e-5`).

### C. Two-Stage Rollout (`Path A → Path B`, Zero Wasted Code)
- **Path A (Ship First — 4-Layer `CNCM-DeltaNet` Plug-In Sidecar, `~10–25 MB` `.sidecar.gguf`):**
  - Donor stays **100% unmodified** (official GGUF). Attach a tiny **4-layer `CNCM-DeltaNet` Sidecar** + SVD Stage 1 + `FactorizedConceptHead` reading fused `L16+L24+L32` donor features.
  - **Runtime wins:** **0 MB extra sidecar KV cache** (`~2–4 MB` fixed state even at 32K), **8× fewer layer-reads while drafting** (`4L × 8` vs `32L × 8`), trivial 4-layer state snapshot rollback in `cncm-rs`, and 1 parallel batched donor verify pass (`M = 16..32`) for **100% lossless** output.
  - **Known trade-offs:** Main donor still uses its own `INT8` KV; best lossless acceptance horizon is `L = 16–32` for hard code/math and `L = 64` for prose.
- **Path B (Natural Extension — 3:1 Hybrid `CNCM-DeltaNet` Backbone):**
  - Reuses **100% of Path A's `CNCMDeltaNetLayer` and `cncm-rs` SIMD kernels** (`depthwise_conv1d_k4` + `delta_state_update`) inside 24 of 32 main trunk layers (1-in-4 Softmax GQA preserved for exact UUID/needle recall) via LoLCATs/MOHAWK distillation or CNCM v2 pretraining.
  - Cuts main-trunk KV RAM by **~75%** (`16K–32K+` on 4 GB phones) and speeds up long-context decode `1.5×–3×+`; can still pair with Path A's 4-layer sidecar on top!

- **বাংলা সারমর্ম:** Paid L40S-এ ৩২ লেয়ার backprop-এর বদলে **SVD Stage 1 (~1.5M) + Cached `L16+L24+L32` Hidden States + 4-Layer `CNCM-DeltaNet` Plug-In Sidecar (Path A)** দিয়ে ফ্রি T4 GPU-তেই ৫–১০ মিনিটে যেকোনো ওপেন GGUF-এ CNCM যোগ করা যাবে (~১৫–২৫ MB `.sidecar.gguf`, 0 extra KV)। পরে একই `CNCM-DeltaNet` কোড ও Rust kernel রিউস করে **Path B (3:1 Hybrid Trunk)**-এ মেইন KV RAM ৭৫% কমানো যাবে।

---

## 2. Variable Chunk Schedule (`L = 16 / 32 / 64`) & Bangla-Optimized Vocab (`V = 8192–16384`)

- **Problem:** Fixed `L = 64` chunks waste compute on short turns/tool calls and stretch a 4-layer sidecar's drafting horizon on dense math/code; small English-heavy vocabularies inflate Bangla UTF-8/BPE tokens by `2×–3×`.
- **Upgrade:**
  - Support **variable chunk horizons** (`L = 16` / `L = 32` for dense code/math & Path A sidecar verify; `L = 64` for prose) via `<chunk_end>`.
  - For from-scratch CNCM / **BLP Workshop** models, train a **Bangla + English + Code Byte-Level BPE** (`V = 8192–16384`, `Digits(individual_digits=True)`).
- **বাংলা সারমর্ম:** `L=16/32` (কোড/ম্যাথ ও Path A sidecar-এর জন্য) এবং `L=64` (সাধারণ লেখার জন্য) ভ্যারিয়েবল চাঙ্ক আর বাংলা-ফ্রেন্ডলি BPE vocab যোগ করলে acceptance rate ও বাংলার semantic density দুটোই বাড়বে।

---

## 3. Learned Early-Exit & Dual-Mode Router

- **Problem:** Static `early_exit_conf = 0.92` thresholds and tag-only `<tool_call>` switching miss inline numerals/identifiers and under-optimize easy prose.
- **Upgrade:** Attach a tiny **2-layer MLP Router Gate** on the sidecar/backbone hidden state predicting per-level `exit_now` (`L2..L5` vs `L7`) and inline `exact_mode` (for numbers/code outside XML tags).
- **বাংলা সারমর্ম:** ফিক্সড threshold বা ট্যাগের বদলে ছোট্ট 2-layer MLP গেট নিজেই ঠিক করবে কখন early-exit করবে আর কখন সংখ্যা/কোডের জন্য Exact-AR-এ যাবে।

---

## 4. Rotation-Aware INT4 / 1.58-Bit Native Weights (`<8 MB` Nano)

- **Problem:** Our controlled +40-step QAT ablation showed no statistically significant gain over +40 FP32 steps (`p = 0.44`), and `nano` packed INT4 sits at `12.29 MB`.
- **Upgrade:** Integrate **Hadamard / SpinQuant online rotations** inside `CNCMBlock` linear projections or full-schedule **1.58-bit ternary (`{-1, 0, +1}`)** native training for ultra-small from-scratch models.
- **বাংলা সারমর্ম:** Hadamard/SpinQuant rotation অথবা পূর্ণ-স্কেল 1.58-bit ternary ট্রেইনিং দিয়ে `nano`-কে **<8 MB**-এ নামানো যাবে প্রায় শূন্য quantization লসে।

---

## 5. Concept-Compressed KV Cache *(Optional Bridge — Only Until Path B Lands)*

- **Role Update:** Still useful for **pure-Softmax donors** on 4 GB phones (compressing older 64-token prefix chunks into `4–8` coarse Stage 1 vectors for `8×–16×` KV RAM savings, especially for Bangla). Once **Path B (3:1 Hybrid `CNCM-DeltaNet` trunk)** is deployed (`~75%` KV reduction structurally), trained Concept-Compressed KV becomes **optional** rather than mandatory.
- **বাংলা সারমর্ম:** Pure-Softmax মডেলের জন্য পুরোনো টোকেন কম্প্রেস করতে কাজে লাগবে, তবে Path B Hybrid `CNCM-DeltaNet` এলে মেইন KV এমনিতেই ৭৫% কমে যাবে—তাই এটা বাধ্যতামূলক থাকবে না।

---

## 6. What We No Longer Need (Superseded / Simplified by Path A → Path B)

| Old / Heavier Plan | Status | Replaced By |
|---|---|---|
| **32-layer full-trunk backprop surgery on paid 48 GB L40S** (`modal_smol360_heal.py`) | **Superseded** | Zero-backprop Offline Hidden Cache + 4-Layer `CNCM-DeltaNet` Sidecar (~5–10 min on free T4) |
| **Per-donor hardcoded surgery scripts** (`surgery_smollm2_to_cncm.py`) | **Superseded** | Single metadata-driven `universal_surgery.py` for any HF/GGUF donor |
| **Training a 27M Stage 1 embedding table from scratch per donor** | **Retired** | Frozen **SVD-Projected Donor Embeddings** (~1.5M trainable Stage 1 params, 18× smaller) |
| **Upcycling 8 Routed MoE Experts (+180M params) during donor surgery** | **Dropped as default** | Keep donor **100% Dense & unmodified** + ~15–25 MB `.sidecar.gguf` (faster & smaller on edge) |
| **Re-exporting & hosting custom 550 MB–1 GB grafted GGUFs per model** | **Retired for Path A** | Users load official unmodified GGUFs + tiny `~10–25 MB` plug-in `.sidecar.gguf` |
| **Full-trunk Stage 1 + Stage 2 unfreeze for donor models** | **Simplified** | Path A trains only the 4L sidecar + head against cached Top-K donor logits (0% forgetting; joint unfreeze only for from-scratch CNCM) |
| **Extra KV cache allocation for the Concept Sidecar** | **Eliminated** | `CNCM-DeltaNet` sidecar uses $O(1)$ fixed state (`~2–4 MB`) even at 32K context |
