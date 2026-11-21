# CNCM Empirical Claims Audit & Verification Ledger (`CLAIMS.md`)

This ledger tracks every architectural, performance, and quality claim in `ARCHITECTURE.md` and `README.md` against direct empirical measurements in:
- [modal_verify_report.json](modal_verify_report.json) (NVIDIA A10, 19/19 pytest, ONNXRuntime, Agentic AI, Stage 1/2/3 run)
- [modal_audit_report.json](modal_audit_report.json) (NVIDIA A10G, Section E strict held-out audit: E1–E5, `$0.226`)

---

## 1. Verified Core Claims (Keep & Cite)

| ID | Claim | Status | Empirical Evidence |
|---|---|---|---|
| **V1** | M-RFSQ has **zero codebook/EMA tables** (`-18.87M` params / `-151 MB` vs `12×4096×384` RVQ+EMA) | [x] Verified | `cncm/quantizer.py` + `tests/test_cncm.py` Test 3 + arithmetic |
| **V2** | **2.30×** vs NCM's **5.0×–11.2×** sequence-length blowup | [x] Verified | `147/64 = 2.30` vs `641/128 = 5.01`, `11469/1024 = 11.20` |
| **V3** | Parameter counts: `nano = 14.50M`, `mini = 39.33M`, `base = 93.04M` | [x] Verified | `count_parameters()` + `modal_verify_report.json` |
| **V4** | `nano` KV cache = `10.50 MB` FP16 (`5.51 MB` INT8) at 1024 context | [x] Verified | `21 × 2 × 1024 × 64 × 2 × 2 = 10.50 MB` |
| **V5** | ONNX export (opset 18) matches PyTorch within `4e-6`–`6e-6` in ONNXRuntime | [x] Verified | `modal_verify_report.json` (`all_ort_passed = true`) |
| **V6** | 2D RoPE (`S == H`), bottom-right causal commit (`max_abs_diff = 2.72e-7`), dtype-safe checkpoints | [x] Verified | `tests/test_cncm.py` + `modal_audit_report.json` (`E2`) |
| **V7** | Concept Mode (`K=8`) achieves **6.4×–7.6×** measured speedup vs. same-param Exact-AR (`B=1`, A10/A10G) | [x] Verified | `modal_audit_report.json` (`E1`: `255.8 ms` vs `1950.9 ms` = **`7.63×`**) |
| **V8** | Packed INT4 footprint = **`12.29 MB`** (`nano`, `4.59×` smaller than `56.32 MB` FP32) | [x] Verified | `modal_audit_report.json` (`E3`) & `modal_verify_report.json` |
| **V9** | Stage 1 **held-out** reconstruction accuracy = **`99.93%`** (16,384 held-out tokens, zero leakage) | [x] Verified | `modal_audit_report.json` (`stage1_heldout_phase_a: 0.99927`, `recon_ce = 0.0525`) |
| **V10** | **Grad-accum equivalence** (`1×16` vs `4×4` microbatches) exact to `< 5e-7` on both Stage 1 & Stage 2 | [x] Verified | `modal_audit_report.json` (`E2`: Stage 1 `4.77e-7`, Stage 2 `5.96e-8`) |
| **V11** | **Quantization quality (FP32 / BF16 / FP16 / INT8 / INT4 + QAT)** preserves held-out NLL & `99.93%` recon | [x] Verified for INT8/INT4 parity; **QAT attribution retracted** | `modal_audit_report.json` (`E3`: FP32 `0.8578`, INT8 `0.8582`, PTQ INT4 `0.8631`, 40-step QAT INT4 `0.8240`), but the fair Tier A control (§5, item 1.3) shows the QAT gain is the extra 40 steps: QAT+40 `4.0570` vs FP32+40 `4.0594` (`Δ = -0.0025 [-0.0134, +0.0085]`, `p = 0.44`) |
| **V12** | **INT8 KV cache drift** is negligible across single-turn (`ΔNLL = -9e-5`) and **4-turn multi-turn chat** (`100.0%` top-1 agreement on all 4 turns) | [x] Verified | `modal_audit_report.json` (`E3`) & `modal_thesis_ablations.py` (`A3`: mean logit diff `0.0000 -> 0.0059`, `top1 = 1.0`) |

---

## 2. Contradicted / Negative Results (Corrected in Code & Docs)

| ID | Original Claim | Empirical Finding | Resolution Applied |
|---|---|---|---|
| **C1 / C37** | Self-speculative gives 8–10.7× & stochastic T>0 lossless | Acceptance = `0.0` at 14.5M/1.5M tokens → **`0.239×` (nano)**, **`0.093×` (mini)**; draft was `argmax` for `T>0` | Removed speed claim in `ARCHITECTURE.md` / `README.md`; fixed `T>0` draft sampling in `cncm/model.py`; labeled acceptance-dependent verification |
| **C2 / C19** | PG-FIM gives 7.78× speedup & 100% compile | `allow_template_fallback` masked model-only output. **Model-only** (`modal_audit_report.json` E5): `parse_rate = 0.0%` (same as AR FIM `0.0%` at 500 steps!), and `KeywordPlanMasker` Python overhead made planned PG-FIM `0.88×` wall-clock vs AR FIM (`1040 ms` vs `933 ms`), whereas no-plan Concept FIM was **`4.64×` faster** (`201 ms`) | Set `allow_template_fallback=False` default in `cncm/model.py`; added `model_generated` & `repair_source` fields; reported model-only vs template upper bound separately |
| **C3** | Stage 3 GRPO/CISPO trains the policy | `final_mean_reward = 0.0`, `parse_rate = 0.0`, `compile_rate = 0.0` at 14.5M/1.5M tokens | Marked RL **experimental / no measurable learning yet** in docs; added `zero_variance_group_fraction` diagnostics in `cncm/trainer/train_grpo.py` |
| **C4** | INT4 gives edge speedup | Pure-PyTorch INT4 is **`2.32×` slower** (`593.0 ms` vs `255.8 ms` FP32) due to per-matmul dequantization | Relabeled INT4 as **memory-only (`12.29 MB`)**; noted ONNX QDQ / fused INT4 kernels required for latency parity |
| **C5** | ~100% FSQ grid utilization | `active_code_ratio` (`n_unique/4096`) was `0.114` (capped at `ℓ_k/4096`); true per-level `batch_utilization` (`n_unique/min(numel, 4096)`) is **`52.3%` mean** (`L2=75.3%`, `L3=76.6%`, `L4=68.8%`, `L7=14.9%`) | Exposed `per_level_batch_utilization` in `cncm/quantizer.py` and updated `ARCHITECTURE.md` |
| **C6 / C7** | >40× fewer output MACs / ~0.16M concept head | Exact measured: **`71,072` params** (`nano`), **`123,424` params** (`mini`), **`~23×–32×` fewer output MACs** | Corrected exact numbers in `ARCHITECTURE.md` & `README.md` |
| **C8 / C29** | 100% schema-valid JSON = deterministic agentic fidelity | Grammar guarantees **100% JSON syntax validity**, not semantic argument accuracy (`"15"` for `"17*6+9"`) | Separated syntax validity from semantic accuracy in `ARCHITECTURE.md` & `README.md` |
| **C9 / C15** | Early exit gives 8–10.7× with preserved quality | Stage 1 held-out cutoff recon drops sharply before `k=8` (`k=6: 31.4%`, `k=7: 54.7%`, `k=8: 99.9%`); at 500 Stage 2 steps, confidence sweep `[0.50..0.99]` stayed at `exit_level=8` (`7.52×–7.62×` speedup) | Scoped early-exit to B=1 and documented the `k=1..8` held-out reconstruction dropoff (`E4`) |
| **C10** | 768-token FIM prefix KV < 5 MB | `8.25 MB` in FP16; **`~4.3 MB` in INT8 KV** | Clarified "under 5 MB with INT8 KV" in `ARCHITECTURE.md` |
| **C17** | Phase B Copy Decoder lowers repair rate | At 500 steps on 14.5M `nano`, Phase B beat Phase A on **held-out FIM val loss** (`exact NLL: 0.655 vs 0.688`, `concept loss: 2.203 vs 2.355`, `recon_ce: 0.0385 vs 0.0525`) and `exact_ar_repair_success_rate` (`62.5%` vs `12.5%`), but both still had `0.0%` end-to-end model-only parse rate | Reported Phase B held-out NLL gain honestly; full pass@1 separation still requires longer/larger training |
| **C20 (E1)** | Same-param quality vs Pure-AR baseline | At 500 steps (`14.50M` CNCM vs `14.48M` Pure-AR, 24,576 held-out tokens): Pure-AR `val_exact_nll = 0.903 (PPL 2.47)` vs CNCM `val_exact_nll = 1.407 (PPL 4.08)` (`Δ = +0.504`, because CNCM splits 70%/30% gradient capacity across Concept + Exact heads), while CNCM Concept Mode is **`7.63×` faster** (`255.8 ms` vs `1950.9 ms`) | Documented exact quality-vs-speed tradeoff |

---

## 3. Still Unverified (Requires Additional Hardware or Scale)

| ID | Claim | Minimum Experiment Command / Requirement |
|---|---|---|
| [ ] **C11 (DDP)** | Multi-GPU 2-rank DDP equivalence | Run 2×GPU `torchrun --nproc_per_node=2` for 100 steps vs 1×GPU (grad-accum 1×16 vs 4×4 is already verified in `E2`) |
| [ ] **C13 / C39** | On-device Android & iGPU latency / RAM | Benchmark `stage1_decoder.onnx`, `stage2_exact_step.onnx`, `stage2_concept_step.onnx` via ONNX Runtime Mobile (Android) & `onnxruntime-directml` (Windows iGPU) |
| [ ] **C14** | DirectML / Apple MPS CI parity | Run `pytest -v tests/test_cncm.py` with `device="dml"` / `device="mps"` |
| [x] **C18** | Factorized head (`71K`) vs Flat 4096 head (`1.57M`) NLL | **Measured (A1):** `22.13×` smaller; improves Exact NLL (`0.1272` vs `0.2367`) at `+0.913` Concept loss (`5.945` vs `5.032`) |
| [ ] **C25** | 300M–1B token scaling & HumanEval-Infilling pass@1 | Scale `mini` (39M) or `base` (93M) to 300M+ tokens on code + prose corpus |
| [x] **C26** | Multi-turn 4-turn conversation consistency | **Measured (A3):** `100.0%` top-1 token agreement across all 4 turns under INT8 KV (`mean_diff <= 0.0059`) |
| [ ] **C27** | `CodeSymbolTable` on real HumanEval/MBPP repositories | Run `CodeSymbolTable.extract_symbols` across HumanEval/MBPP corpus |
| [x] **C30** | 21 looped (`7×3`) vs equal-param non-looped (`7×1`) | **Measured (A2, negative result at 300 steps):** non-looped `7×1` beat `7×3` on Exact NLL (`0.0567` vs `0.1272`) and Concept loss (`4.515` vs `5.945`) |

---

## 4. Reproducing This Audit

```bash
# 1. Local unit & invariant verification
python -m pytest -v tests/test_cncm.py

# 2. Tier A controlled experiments (local CPU, 3 seeds, ~44 min total)
python experiments/run_controls.py                    # full suite -> experiments/results/controls_summary.json
python experiments/run_controls.py exp_5_2_looping    # or a single experiment by name

# 3. Held-out benchmark suites (local CPU)
python experiments/eval_benchmarks.py                 # -> experiments/results/benchmarks_rigor.json

# 4. Edge / ONNX Runtime benchmark on this laptop (CPU + Iris Xe via DirectML)
#    results: experiments/results/edge_bench_nano.json (no committed runner script)

# 5. Section E GPU audit on Modal (~$0.23 on NVIDIA A10G)
modal run modal_audit.py --preset nano --stage1-steps 350 --stage2-steps 500 --ar-steps 500
```

---

## 5. Tier A Controlled Experiments (local CPU, 3 seeds, 2026-11-20)

Harness: `experiments/harness.py` (strict 70/15/15 document splits, SHA-256 config hash, JSON ledger),
`experiments/stats.py` (Wilson intervals, 95% t-intervals, paired t-test + paired bootstrap).
Ledger: [`experiments/results/controls_summary.json`](experiments/results/controls_summary.json).
Host: Intel i5-1240P, `torch 2.15.0+cpu`, no CUDA. Seeds `42 / 137 / 2026`. Model: `tiny_test`
(`enc_layers=1`, 408k params for CNCM). All arms in one comparison share steps, batch size, LR schedule and data order.

| # | Question | Result (mean ± 95% CI) | Verdict |
|---|---|---|---|
| **1.1** | Does equal-step training close the CNCM vs Pure-AR quality gap? | 120 steps, 24,576 held-out tokens: CNCM `3.976 ± 0.387` vs parameter-matched Pure-AR `3.946 ± 0.238`. Paired Δ `+0.031`, t-CI `[-0.440, +0.501]`, `p = 0.81` | **No difference detectable** at this scale. The interval is wide, so this is "not significant", *not* "equal" |
| **1.2b** | At matched decode latency, is CNCM the better model? | Latency-matched Pure-AR (45.5k params, 41.3 ms) reaches NLL `3.747 ± 0.118`; CNCM (408k params, 92.5 ms) reaches `3.976 ± 0.387` | **Negative.** A ~9× smaller AR model is better on quality at ~half the latency (CNCM latency is noisy: `92.5 ± 118.9` ms) |
| **1.3** | Is the QAT gain caused by QAT or by the extra 40 steps? | FP32+0 `4.1079`, PTQ-INT4+0 `4.1050`, FP32+40 `4.0594`, INT4-QAT+40 `4.0570`. True QAT effect (QAT+40 − FP32+40) `-0.0025 [-0.0134, +0.0085]`, `p = 0.44` | **The gain is the extra 40 steps.** QAT's own contribution is not measurable at this scale |
| **1.5** | Are `λ_c / λ_e` tuned, and does tuning transfer? | VAL-selected `λ_c=0.5, λ_e=1.0` → TEST `3.914 ± 0.349` vs default `(1.0, 0.3)` `4.069 ± 0.099`. Paired Δ `-0.155 [-0.406, +0.095]`, `p = 0.12`; bootstrap CI `[-0.268, -0.075]` | **Directional only.** Lower concept weight looks better; 3 seeds cannot resolve it |
| **5.1** | Can draft-alignment training rescue self-speculative decoding? | Acceptance `0.282 ± 0.761` → `0.012 ± 0.053`; speedup vs AR `0.262×` → `0.101×`; greedy output still bit-identical to AR | **Negative.** Distillation did not raise acceptance; self-speculation stays a documented negative result |
| **5.2** | Does recursive looping pay off at 4× the budget? | 400 steps, looped (`2 unique × 3 loops`) `3.326 ± 0.161` vs non-looped (`2 × 1`) `2.789 ± 0.506`. Paired Δ `+0.536 [+0.047, +1.026]`, `p = 0.042` | **Negative, significant.** Looping costs quality at equal params and *more* compute → `num_loops=1` is the honest default |
| **5.4** | Does RL work with SFT warm start, G=16, T=1.0? | 80-step SFT warm start + 10 GRPO steps: mean reward `0.0` on every seed and every step, max step reward `0.0`, `policy_updated_seed_rate = 0/3`, final KL `0.0` | **Still no learning signal.** GRPO at this scale should be described as a harness, not a result |

Held-out benchmark suites ([`experiments/results/benchmarks_rigor.json`](experiments/results/benchmarks_rigor.json),
hash-manifested before evaluation in [`experiments/benchmarks/MANIFEST.json`](experiments/benchmarks/MANIFEST.json)):

| Suite | n | parse | compile | pass@1 |
|---|---|---|---|---|
| `pg_fim_v3` fresh test (dev-only old 24-task set excluded) | 60 | 0.0% | 0.0% | 0.0% |
| HumanEval-Infilling (MIT) | 100 | 1.0% [0.2, 5.4] | 1.0% | 0.0% |
| SAFIM python (CC-BY-SA-4.0) | 100 | 0.0% | 0.0% | 0.0% |
| Held-out arithmetic (exact match) | 1000 | — | — | 0.0% |

Tool-calling, 500 held-out prompts: the grammar gives `100%` JSON syntax validity and `100%` tool-name
accuracy, but `0%` semantic expression accuracy (0/500). With the prompt-span copy constraint, expression
copying reaches `100%` while the *evaluated value* stays `0%` — the grammar constrains syntax and copying,
never arithmetic correctness.

**What this means for the paper.** Only V7 (`7.63×` measured on A10G) and V8/V9/V10/V11/V12 survive as
positive results. Every quality-related claim must be stated as "no measurable difference / not measurable at
this scale", and self-speculation, QAT, looping and RL are negative results that belong in a limitations section.

---

## `ctx0an` Base-MoE Controlled A/B Claims (Modal NVIDIA A10 Verified)

- **Controlled A/B Benchmark (`modal_moe_benchmark.py` -> `modal_ctx0an_moe_benchmark.json`, `BENCHMARK_CTX0AN_MOE.md`)**:
  - `base_moe` (`221.45M` total / `94.05M` active) vs. Dense `base` (`93.90M`) on identical 4-domain curated data (`3.17M` train tokens, `V=6400` BPE, shared Stage 1):
    - **Phase 1 Val Exact-AR NLL**: `6.4026` vs. `6.4818` (`-0.0792`)
    - **Phase 2 Val Exact-AR NLL**: `5.9886` (`398.86` PPL) vs. `6.1619` (`474.33` PPL) (`-0.1733`)
    - **Gate #5 Align NLL / Concept Draft NLL**: `-0.2314` / `-0.5269`
    - **MoE Routing Health**: `8/8` experts active (`0.9984` normalized load entropy), `82.85%` Loop 0→1 expert switching.
    - **Throughput & Size Trade-off**: Dense `base` is `~1.35×` faster in training (`14,689` vs `10,625` tok/s Phase 1) and `~2.4×` smaller (`~46.6 MB` vs `~109.8 MB` INT4).
- **Unit Tests**: `25/25` tests passing in `tests/test_cncm.py`.
