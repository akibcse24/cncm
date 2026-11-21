# CNCM — Compact Causal-Hybrid Next-Concept Model
[[https://zenodo.org/badge/DOI/10.5281/zenodo.24419823.svg]](https://doi.org/10.5281/zenodo.24419823)
**Fast, parallel coarse-to-fine text generation and schema-constrained Agentic AI designed for low-end Android phones, consumer iGPU laptops, Apple Silicon, and desktop GPUs.**

CNCM re-engineers **Next-Concept Modeling** ([HazyResearch/ncm](https://github.com/HazyResearch/ncm)) with **6 edge-first architectural components** (see [CLAIMS.md](CLAIMS.md) for the full empirical verification & audit status of every claim):
1. **Multiscale Residual Finite Scalar Quantization (`MultiscaleRFSQ`)** — eliminates ~151 MB of explicit FP32+EMA VQ codebook tables (for `12×4096×384`) using zero-table implicit $8^4 = 4096$ FSQ grids (`99.95%` held-out recon, grad-accum invariant, FP16/BF16/INT8/INT4 quality verified).
2. **Causal-Prefix Block-AR + Dynamic Early Exit** — compact $L = 64, K = 8$ schedule `(1, 2, 4, 8, 12, 20, 36, 64)` (**2.30×** sequence-length blowup vs. NCM's 5.0×–11.2×), causal prefix KV caching (with INT8 KV `ΔNLL = 3e-5`), **6.4×–8.85× measured Concept-Mode speedup vs. AR** on NVIDIA A10 (`B=1`), and zero target-scale KV cache growth.
3. **Relaxed Recursive Layer Looping (`CNCMBlock`)** — Pre-RMSNorm + QK-RMSNorm + GQA + SwiGLU with shared weights + low-rank loop adapters achieving **21–24 looped layer passes** in **14.5M–93.0M** total parameters (**12.29 MB** packed INT4 for `nano`). *(Looping is an option, not a proven win: at `tiny_test` / 400 equal steps the non-looped arm had the better held-out NLL — see `CLAIMS.md` §5, item 5.2.)*
4. **Factorized Joint Concept Head (`FactorizedConceptHead`)** — models 4096-way joint FSQ digit correlation in **71K params** (`nano`) / **123K params** (`mini`) and **~23×–32× fewer output MACs**, enabling exact 1-pass code-space **GRPO / CISPO** RL (code complete; RL is currently experimental at 14.5M scale).
5. **Dual-Mode Agentic AI + Self-Speculative Verification** — one shared backbone switches cleanly between fast parallel **Concept Mode** (`<think>` & prose) and grammar-constrained **Exact-AR Mode** (`<tool_call>...</tool_call>` with `JSONSchemaConstrainedMasker` guaranteeing **100% JSON-schema syntax validity**; semantic argument accuracy depends on SFT scale), plus greedy-equivalent `generate_self_speculative` verification (acceptance-dependent; not an accelerator at short-training scale).
6. **Plan-Guided Fill-in-the-Middle (PG-FIM)** — cursor-aware code and prose infilling (`model.generate_fim`) combining a short Exact-AR plan (`<mlen_k> <kw> kw1 | kw2 </kw>` guided by `CodeSymbolTable` + `KeywordPlanMasker`), parallel Concept Mode middle generation, Phase A 1-pass Exact-AR syntax/keyword span repair (with `allow_template_fallback=False` by default so model-only metrics are never masked), optional Phase B `CopyAwareKeywordCrossAttention` Stage 1 pointer decoding, and Stage 3 Compiler-in-the-Loop GRPO (`train_pg_fim_grpo` + `run_code_sandbox`).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the mathematical specification and [CLAIMS.md](CLAIMS.md) for the verified vs. unverified claim audit.

---

## Quickstart

```bash
# 1. Install minimal pure-PyTorch core (CPU / CUDA / MPS / DirectML)
pip install -r requirements.txt

# 2. Run the full end-to-end demo (hardware profiles, Stage 1+2+GRPO smoke run,
#    dual-mode JSON-constrained calculator agent, real INT4/INT8 quantization & ONNX wrapper check)
python demo_cncm.py
```

---

## Hardware-Targeted Presets

| Preset | Params (S1 + S2) | Vocab | Chunk $L$ / $K$ | Effective Depth | Packed INT4 (`g=32`) | Target Hardware |
|---|---|---|---|---|---|---|
| `tiny_test` | **0.38M** (`0.10M + 0.28M`) | 512 | `16` / `5` | `2 × 2 = 4L` | **0.51 MB** | Fast CPU CI & unit tests |
| `nano` | **14.50M** (`1.51M + 12.99M`) | 4096 | `64` / `8` | `7 × 3 = 21L` | **12.29 MB** (`<15 MB`) | **Low-End Android** (4 GB RAM, XNNPACK/NNAPI) |
| `mini` | **39.33M** (`2.56M + 36.77M`) | 4096 | `64` / `8` | `12 × 2 = 24L` | **29.67 MB** (`<30 MB`) | **Consumer iGPU Laptop** (8–16 GB RAM, DirectML/Vulkan) |
| `base` | **93.04M** (`5.08M + 87.96M`) | 6400 | `64` / `8` | `12 × 2 = 24L` | **66.08 MB** | **Desktop GPU / Apple Silicon / High-End NPU** |

*(Run `python demo_cncm.py --section profile` to print live hardware roofline estimates, KV-cache RAM, and HBM/LPDDR bandwidth savings.)*

---

## Training Pipeline

All trainers use pure PyTorch (zero CUDA-only/Triton-only dependencies), `WarmupCosineScheduler`, padding-free chunk packing, explicit `prefix_mask`, and FP32/INT64-safe atomic checkpoints:

```bash
# Stage 1: Train M-RFSQ Concept Tokenizer (padding-free PackedChunkDataset + nested fine_drop)
python -m cncm.trainer.train_stage1 --preset tiny_test --steps 50 --batch-size 8

# Stage 2: Train Unified Dual-Mode CNCMModel (frozen Stage 1 + 50% cond corruption + Exact-AR loss)
python -m cncm.trainer.train_stage2 --preset tiny_test --mode joint --steps 50 --batch-size 4

# Stage 3: 1-Pass Code-Space GRPO / CISPO RL with verifiable <think>/<tool_call>/answer rewards
python -m cncm.trainer.train_grpo --preset tiny_test --steps 6 --num-generations 4

# Stage 3 (PG-FIM): Compiler-in-the-Loop GRPO with AST/compile/unit-test sandbox rewards
python -m cncm.trainer.train_grpo --preset tiny_test --fim --steps 6 --num-generations 4
```

---

## Plan-Guided Fill-in-the-Middle (PG-FIM) Usage

```python
from cncm import CNCMConceptTokenizer, CNCMModel, CNCMTextTokenizer, get_preset

cfg = get_preset("nano", use_copy_decoder=True)
tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
model = CNCMModel(cfg, CNCMConceptTokenizer(cfg))

res = model.generate_fim(
    prefix="def sum_positive(items):\n    total = 0\n    for x in items:\n        if x > 0:\n",
    suffix="\n    return total\n",
    tokenizer=tokenizer,
    use_plan=True,        # Emits <mlen_k> <kw> ... </kw> under KeywordPlanMasker
    enable_repair=True,   # Phase A 1-pass Exact-AR syntax & keyword span repair
    quantize_kv=True,     # INT8 causal prefix KV cache
)
print("Plan:", res["planned_chunks"], res["keywords"])
print("Infill:\n", res["full_text"])
```

---

## Dual-Mode Agentic AI Usage

```python
from cncm import CNCMConfig, CNCMConceptTokenizer, CNCMModel, CNCMTextTokenizer, get_preset
from cncm.agent import CNCMAgent, create_default_registry

cfg = get_preset("nano")
tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
model = CNCMModel(cfg, CNCMConceptTokenizer(cfg))

agent = CNCMAgent(
    model=model,
    tokenizer=tokenizer,
    registry=create_default_registry(),  # calculator, kv_memory, text_stats, clock_utc
    use_grammar_constraints=True,        # 100% schema-valid JSON in Exact-AR mode
    quantize_kv=True,                    # INT8 causal prefix KV cache
)
trace = agent.run("Calculate 18 * 24 + 9 using the calculator.")
print(trace.summary())
```

---

## Edge Deployment Guide (Android & Consumer iGPU Laptops)

### 1. Real Packed INT4 / INT8 Quantization
```python
from cncm.export import quantize_model_int4, measure_model_size

q_model = quantize_model_int4(model, group_size=32, quantize_embeddings=True)
print(measure_model_size(q_model))  # Packs 2 x 4-bit weights per uint8 + fp16 group scales/zeros
```

### 2. ONNX Graph Export (Opset 17+)
```python
from cncm.export import export_cncm_onnx

# Verifies all 3 pure-PyTorch edge wrappers & emits stage1_decoder.onnx,
# stage2_exact_step.onnx, and stage2_concept_step.onnx when `onnx` is installed:
report = export_cncm_onnx(model, out_dir="exports/nano_onnx", opset=17)
```

### 3. Target Backend Setup
- **Windows Consumer iGPU (Intel Iris Xe / AMD Radeon Vega)**:
  Install `pip install torch-directml` (training/PyTorch inference) or `pip install onnxruntime-directml` (ONNX deployment). `cncm.trainer.train_utils.select_device("auto")` detects DirectML automatically.
- **Low-End Android (4 GB RAM, ARM Cortex-A55/A76)**:
  Export `nano` with `export_cncm_onnx`, apply INT4 weight packing (`quantize_model_int4` -> `12.29 MB` memory footprint; note that pure-PyTorch INT4 is memory-only and requires ONNX QDQ / XNNPACK / NNAPI fused kernels for latency parity), and execute via **ONNX Runtime Mobile** with INT8 prefix KV caching (`<15 MB` weights, projected `~22–30 MB` runtime RAM).
- **Apple Silicon / Desktop GPU**:
  Runs natively on `mps` or `cuda` with PyTorch SDPA (`is_causal=False` explicit 4D masks).

---

## Project Layout

```text
cncm/
├── config.py              # CNCMConfig + tiny_test / nano / mini / base presets
├── text_tokenizer.py      # Lossless byte-fallback + subword tokenizer & chat/tool template
├── quantizer.py           # MultiscaleRFSQ (zero-table FSQ, DepthwiseSep L1 & Phi convs)
├── tokenizer_model.py     # Stage 1 CNCMConceptTokenizer
├── layers.py              # 2D-safe RoPE, 4D SDPA masks, Recursive CNCMBlock, INT8 KV, FactorizedConceptHead
├── model.py               # Unified Dual-Mode CNCMModel (Concept, Block-AR, Exact-AR, Self-Speculative)
├── trainer/               # PackedChunkDataset, BlockARSFTDataset, Stage 1/2 & Code-Space GRPO trainers
├── agent/                 # ToolRegistry (4 built-in tools), JSONSchemaConstrainedMasker, CNCMAgent
└── export/                # Packed INT4/INT8 quantization, hardware roofline profiler, ONNX export
demo_cncm.py               # Runnable end-to-end showcase CLI
ARCHITECTURE.md            # Deep technical specification & vanilla NCM comparison
```

---

## Credits & References

- **Next-Concept Modeling (NCM)**: H. Kumbong et al., Stanford Hazy Research & Scaling Intelligence Lab — [*Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*](https://github.com/HazyResearch/ncm) (arXiv:2610.14162, MIT License).
- **MiniMind-3**: Compact edge-LLM architecture, small-vocabulary efficiency, and multi-stage training lessons from Jingyao Gong's MiniMind series.
- **Finite Scalar Quantization (FSQ)**: Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple* (ICLR 2024).

---

## Latest `ctx0an` Training & MoE Experiment Updates

- **Main Architecture (`base` Dense)**: [train_ctx0an_base.py](train_ctx0an_base.py) defaults to `--preset base` (`93.90M` dense, `24` effective layers via `12 × 2` loops, `V=6400` corpus Byte-Level BPE, `~46.6 MB` INT4).
- **MoE Experiment (`base_moe`)**: [experiment/moe_ctx0an.py](experiment/moe_ctx0an.py) implements DeepSeek-style Fine-Grained MoE (`221.45M` total / `94.05M` active, `1` shared + `8` routed Top-2 experts with `24` per-loop routers, `~109.8 MB` INT4).
- **Kaggle & Cloud Infrastructure**:
  - `MemmapTokenDataset` + `prepare_memmap_corpus` (`uint16` flat binary, `0%` padding, `<10 MB` RAM) + 200× faster Rust `tokenizers` `batch_encode`.
  - `CheckpointManager` (atomic `latest.pt`, `best.pt`, `keep_last_k` pruning, wall-clock guard, optional private HF Hub async sync).
  - T4 FP16 `GradScaler` (Stage 1 strictly FP32), `grad_checkpointing`, and `torchrun` multi-GPU DDP.
- **Verified Modal A10 Benchmark**: See [BENCHMARK_CTX0AN_MOE.md](BENCHMARK_CTX0AN_MOE.md) (`base_moe` won **6/6** quality/alignment checks: `-0.1733` Phase 2 Val NLL, `-0.5269` Concept Draft NLL, `8/8` active experts with `0.9984` load entropy; Dense `base` is `~1.35×` faster and `~2.4×` smaller).
- **Test Suite**: `25/25` unit tests passing in [tests/test_cncm.py](tests/test_cncm.py).
