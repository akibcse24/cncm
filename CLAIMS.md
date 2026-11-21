# CNCM Experimental Claims Ledger (`CLAIMS.md`)

This document tracks every quantitative claim made for CNCM under strict scientific controls.

## Reporting Policy
1. **Single-Seed Default + Test-Set CIs**: All scripts default to **1 deterministic seed (`42`)** via `experiments/harness.py` to maximize compute spent on data/model scale, reporting **Wilson 95% CIs** for pass rates and **chunk-level bootstrap 95% CIs** for held-out NLL (multi-seed `--seeds 42 1337 2025` remains optionally supported).
2. **Wilson 95% CIs for Pass Rates**: Every pass@1, syntax rate, and semantic accuracy rate must report a **Wilson 95% confidence interval** via `experiments/stats.py`.
3. **Zero Unverified Claims / Honest Negative Results**: Negative results (e.g., CNCM trailing Pure-AR at equal steps on nano scale, `7x1` beating `7x3` looping, or ~0% on HumanEval-Infilling at 14.5M params) are reported verbatim rather than tuned away.
4. **Concept Quality Gate (Item 4.3)**: No claim about Concept Mode generation quality is permitted until `concept_draft_nll < ln(4096) - 0.5` (`< 7.82` nats) AND `exact_nll < unigram_nll - 0.5`.

---

## Claim Status Matrix

| Item | Claim / Experiment | Status | Command to Reproduce | Ledger Artifact |
|---|---|---|---|---|
| **1.1** | Equal-budget (500/500 steps, equal tokens & order) CNCM vs Pure-AR (3 seeds) | Ready (Tier A — requires PyTorch) | `python -m experiments.run_controls --exp 1.1 --preset nano --steps 500` | `experiments/results/exp_1_1_equal_budget_nano.json` |
| **1.2a** | Fair deployed latency: ORT KV-cache (CPU/iGPU) & `torch.compile(max-autotune)` (GPU) | Ready (Tier A + Tier B) | `python -m cncm.export.edge_bench --preset nano --full-soak` | `experiments/results/edge_bench_nano.json` |
| **1.2b** | Latency-matched Pure-AR depth/width sweep vs CNCM Concept Mode | Ready (Tier A) | `python -m experiments.run_controls --exp 1.2b --preset nano` | `experiments/results/exp_1_2b_latency_matched_ar_nano.json` |
| **1.3** | Fair 3-arm QAT (`FP32+40` vs `INT4 QAT+40` vs `PTQ INT4+0`, 3 seeds) | Ready (Tier A) | `python -m experiments.run_controls --exp 1.3 --preset nano` | `experiments/results/exp_1_3_fair_qat_nano.json` |
| **1.4** | Equal-memory `base_dense220` vs `base_moe` A/B | Cost-gated (Tier B) | `python -m experiments.modal_tier_b --exp 1.4 --confirm-paid-gpu` | `experiments/results/tier_b_1_4_dense220_vs_moe.json` |
| **1.5** | Validation-only `lambda_e x lambda_c` grid -> 3-seed test confirmation | Ready (Tier A) | `python -m experiments.run_controls --exp 1.5 --preset nano` | `experiments/results/exp_1_5_lambda_grid_nano.json` |
| **2.1 / 2.2** | Physical laptop CPU + Iris Xe iGPU benchmark (FP32 / INT8 Dynamic / INT8 QDQ / INT4 `MatMulNBits` + 5-min thermal soak) | Ready (Tier A + Tier C Android) | `python -m cncm.export.edge_bench --preset nano --full-soak` | `experiments/results/edge_bench_nano.json` |
| **3.1** | Held-out PG-FIM test set (36 unseen tasks, SHA-256 locked before eval) | **Locked & Verified** | `python -m experiments.eval_suites` | `experiments/manifests/eval_suites_sha256.json` |
| **3.2** | Held-out Arithmetic (`n=1,000`), Tool-Calling (`n=500`), HumanEval-Infilling, SAFIM | **Locked & Verified** | `python -m unittest discover -s tests -v` | `experiments/manifests/eval_suites_sha256.json` |
| **5.1** | Draft-alignment distillation (`alpha`, actual speedup, expected speedup formula) | Ready (Tier A) | `python -m experiments.run_controls --exp 5.1 --preset nano` | `experiments/results/exp_5_1_draft_align_nano.json` |
| **5.2** | Looping ablation (`7x3` vs `7x1` at 2000 steps, 3 seeds; auto-recommends `num_loops=1` if `7x1` wins) | Ready (Tier A) | `python -m experiments.run_controls --exp 5.2 --preset nano --steps 500` | `experiments/results/exp_5_2_looping_ablation_nano.json` |
| **5.3** | Tool-calling syntax rate vs semantic accuracy with copy-span constraint | **Unit Verified** | `cncm.agent.grammar.evaluate_tool_call_semantic_accuracy` | `experiments/manifests/eval_suites_sha256.json` |
| **5.4** | SFT warm-start + partial-credit GRPO (`G=16`, `T=1.0`, skip zero-variance groups) | **Implemented** | `cncm.trainer.train_grpo` (`skip_zero_variance_groups=True`) | `experiments/results/` |
| **5.5** | Dataset & benchmark licensing audit | **Complete** | See `LICENSES_DATA.md` & `THIRD_PARTY_NOTICES.md` | `LICENSES_DATA.md` |
