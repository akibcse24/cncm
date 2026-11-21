"""Tier A Controlled Remediation Experiments (`experiments/run_controls.py`).

Implements every local (Tier A) control and algorithmic remediation experiment
from `remediation_plan.md` using `experiments.harness` (>=3 seeds, mean ± 95% CI,
paired statistical tests, deterministic config hashes, and JSON ledgers):

  - `1.1`: Equal-step & equal-token CNCM vs Pure-AR (500/500, 3 seeds, same session latency)
  - `1.2b`: Latency-matched Pure-AR depth/width sweep vs CNCM Concept Mode
  - `1.3`: Fair 3-arm QAT control (FP32 +40 vs INT4 QAT +40 vs PTQ INT4 +0, 3 seeds)
  - `1.5`: Validation-only `lambda_e x lambda_c` grid search (2 seeds val -> 3 seeds test)
  - `5.1`: Draft-alignment self-distillation (acceptance alpha, speedup, theoretical formula)
  - `5.2`: Looping ablation 7x3 vs 7x1 at 4x budget (2000 steps, 3 seeds)
  - `5.3`: Tool-calling syntax vs semantic accuracy with & without copy-span constraint
  - `5.4`: SFT warm-start + partial-credit GRPO (G=16, T=1.0, skip zero-variance groups)

Usage:
    python -m experiments.run_controls --exp 1.1 --preset nano --steps 500
    python -m experiments.run_controls --exp all --preset tiny_test --steps 20 --seeds 42 1337 2025
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments.harness import DEFAULT_SEEDS, run_seeded_experiment, save_result, set_global_seed
from experiments.stats import (
    expected_speculative_speedup,
    paired_nonparametric_test,
    paired_t_test,
    summarize_seeded_metric,
    wilson_interval,
)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to run training/inference controls. "
            "Install torch in your environment (e.g. `.venv`) and re-run."
        ) from exc


def _build_split_datasets(cfg: Any, tokenizer: Any, seed: int, n_train: int = 256, n_val: int = 64, n_test: int = 64):
    """Build strictly disjoint train / val / test PackedChunkDatasets."""
    from cncm.trainer.dataset import PackedChunkDataset, make_synthetic_corpus

    train_texts = make_synthetic_corpus(num_samples=n_train, seed=seed)
    val_texts = make_synthetic_corpus(num_samples=n_val, seed=seed + 10_000)
    test_texts = make_synthetic_corpus(num_samples=n_test, seed=seed + 20_000)

    train_ds = PackedChunkDataset(train_texts, tokenizer, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    val_ds = PackedChunkDataset(val_texts, tokenizer, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    test_ds = PackedChunkDataset(test_texts, tokenizer, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    return train_ds, val_ds, test_ds


def _eval_exact_nll(model: Any, dataset: Any, device: str, max_batches: int = 16) -> float:
    """Compute mean Exact-AR cross-entropy NLL on a held-out dataset."""
    torch = _require_torch()
    from torch.utils.data import DataLoader

    model.eval()
    dl = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False)
    losses: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            ids = batch["exact_input_ids"].to(device)
            labels = batch["exact_labels"].to(device)
            mask = batch["exact_attention_mask"].to(device)
            out = model.forward_exact(ids, labels=labels, attention_mask=mask)
            losses.append(float(out["loss"].item()))
    return sum(losses) / max(1, len(losses))


def _measure_session_latency(model: Any, cfg: Any, device: str, prefix_len: int = 64, iters: int = 3) -> Dict[str, float]:
    """Measure Concept Mode and Exact-AR 64-tok latency on the same device in one session."""
    torch = _require_torch()
    model.eval()
    pfx = torch.randint(16, cfg.vocab_size, (1, min(prefix_len, cfg.max_position_embeddings // 2)), device=device)

    with torch.no_grad():
        # Warmup (excluded from timing)
        _ = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)
        _ = model.generate_exact(pfx, max_new_tokens=cfg.chunk_size, temperature=0.0)

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)
        c_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model.generate_exact(pfx, max_new_tokens=cfg.chunk_size, temperature=0.0)
        ar_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    return {
        "concept_chunk_ms": round(c_ms, 3),
        "concept_tok_per_sec": round(cfg.chunk_size / max(c_ms / 1000.0, 1e-6), 3),
        "exact_ar_chunk_ms": round(ar_ms, 3),
        "exact_ar_tok_per_sec": round(cfg.chunk_size / max(ar_ms / 1000.0, 1e-6), 3),
        "concept_speedup_vs_ar": round(ar_ms / max(c_ms, 1e-6), 4),
    }


# ---------------------------------------------------------------------------
# 1.1: Equal-Step, Equal-Token CNCM vs Pure-AR (500/500, 3 seeds)
# ---------------------------------------------------------------------------
def run_exp_1_1_equal_budget(
    preset: str = "nano",
    steps: int = 500,
    stage1_steps: int = 100,
    batch_size: int = 8,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train CNCM (joint) and Pure-AR on identical steps, batch size, tokens, and data order."""
    _require_torch()
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.train_stage1 import train_stage1
    from cncm.trainer.train_stage2 import train_stage2

    cfg = get_preset(preset, enc_layers=1)
    tokens_per_run = int(steps * batch_size * (2 * cfg.chunk_size))

    def _one_seed(seed: int) -> Dict[str, Any]:
        tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
        train_ds, _, test_ds = _build_split_datasets(cfg, tok, seed=seed)

        # Stage 1 shared concept tokenizer
        s1 = CNCMConceptTokenizer(cfg)
        train_stage1(cfg, tok, model=s1, dataset=train_ds, steps=stage1_steps, batch_size=batch_size, device=device, seed=seed, verbose=False)

        # Arm A: CNCM Joint (concept + exact)
        set_global_seed(seed)
        cncm_model = CNCMModel(cfg, copy.deepcopy(s1))
        train_stage2(
            cfg, tok, model=cncm_model, dataset=train_ds,
            mode="joint", steps=steps, batch_size=batch_size,
            concept_weight=1.0, exact_weight=0.7, device=device, seed=seed, verbose=False,
        )
        cncm_nll = _eval_exact_nll(cncm_model, test_ds, device)
        cncm_lat = _measure_session_latency(cncm_model, cfg, device)

        # Arm B: Pure-AR (exact_only, same steps, same batch_size, same data seed)
        set_global_seed(seed)
        ar_model = CNCMModel(cfg, copy.deepcopy(s1))
        train_stage2(
            cfg, tok, model=ar_model, dataset=train_ds,
            mode="exact_only", steps=steps, batch_size=batch_size,
            concept_weight=0.0, exact_weight=1.0, device=device, seed=seed, verbose=False,
        )
        ar_nll = _eval_exact_nll(ar_model, test_ds, device)
        ar_lat = _measure_session_latency(ar_model, cfg, device)

        return {
            "cncm_test_exact_nll": round(cncm_nll, 5),
            "pure_ar_test_exact_nll": round(ar_nll, 5),
            "nll_gap_cncm_minus_ar": round(cncm_nll - ar_nll, 5),
            "cncm_concept_tok_per_sec": cncm_lat["concept_tok_per_sec"],
            "pure_ar_tok_per_sec": ar_lat["exact_ar_tok_per_sec"],
            "same_session_speedup": round(
                ar_lat["exact_ar_chunk_ms"] / max(cncm_lat["concept_chunk_ms"], 1e-6), 4
            ),
        }

    res = run_seeded_experiment(
        f"exp_1_1_equal_budget_{preset}",
        _one_seed,
        seeds=seeds,
        config={"preset": preset, "steps": steps, "stage1_steps": stage1_steps, "batch_size": batch_size, "device": device},
        steps=steps,
        total_tokens=tokens_per_run,
    )
    cncm_vals = [r["metrics"]["cncm_test_exact_nll"] for r in res["per_seed"]]
    ar_vals = [r["metrics"]["pure_ar_test_exact_nll"] for r in res["per_seed"]]
    if len(cncm_vals) >= 2:
        res["paired_nll_test"] = paired_t_test(cncm_vals, ar_vals)
        res["paired_nll_nonparam"] = paired_nonparametric_test(cncm_vals, ar_vals)
    return res


# ---------------------------------------------------------------------------
# 1.2b: Latency-Matched Pure-AR Sweep vs CNCM Concept Mode
# ---------------------------------------------------------------------------
def run_exp_1_2b_latency_matched_ar(
    preset: str = "nano",
    steps: int = 200,
    batch_size: int = 8,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Sweep Pure-AR depth/width until tok/s matches CNCM Concept Mode, then compare NLL."""
    _require_torch()
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.train_stage2 import train_stage2

    base_cfg = get_preset(preset, enc_layers=1)
    # Candidate smaller AR shapes (num_unique_layers, num_loops, hidden_size, intermediate_size)
    candidates = [
        {"num_unique_layers": max(1, base_cfg.num_unique_layers // 2), "num_loops": 1, "hidden_size": base_cfg.hidden_size, "intermediate_size": base_cfg.intermediate_size},
        {"num_unique_layers": max(1, base_cfg.num_unique_layers // 3), "num_loops": 1, "hidden_size": base_cfg.hidden_size, "intermediate_size": base_cfg.intermediate_size},
        {"num_unique_layers": 2, "num_loops": 1, "hidden_size": max(96, (base_cfg.hidden_size // 2 // base_cfg.num_attention_heads) * base_cfg.num_attention_heads), "intermediate_size": max(256, base_cfg.intermediate_size // 2)},
        {"num_unique_layers": 1, "num_loops": 1, "hidden_size": max(96, (base_cfg.hidden_size // 2 // base_cfg.num_attention_heads) * base_cfg.num_attention_heads), "intermediate_size": max(256, base_cfg.intermediate_size // 2)},
    ]

    # Step 1: Benchmark CNCM Concept Mode latency & pick closest latency-matched AR candidate
    s1_ref = CNCMConceptTokenizer(base_cfg)
    cncm_ref = CNCMModel(base_cfg, s1_ref).to(device)
    target_tps = _measure_session_latency(cncm_ref, base_cfg, device)["concept_tok_per_sec"]

    sweep_log = []
    best_cand = candidates[0]
    best_rel_err = float("inf")
    for cand in candidates:
        c_cfg = get_preset(preset, enc_layers=1, **cand)
        m = CNCMModel(c_cfg, CNCMConceptTokenizer(c_cfg)).to(device)
        tps = _measure_session_latency(m, c_cfg, device)["exact_ar_tok_per_sec"]
        rel_err = abs(tps - target_tps) / max(target_tps, 1e-6)
        sweep_log.append({"overrides": cand, "ar_tok_per_sec": tps, "target_cncm_tok_per_sec": target_tps, "rel_err": round(rel_err, 4)})
        if rel_err < best_rel_err:
            best_rel_err = rel_err
            best_cand = cand

    matched_cfg = get_preset(preset, enc_layers=1, **best_cand)

    def _one_seed(seed: int) -> Dict[str, Any]:
        tok = CNCMTextTokenizer(vocab_size=base_cfg.vocab_size)
        train_ds, _, test_ds = _build_split_datasets(base_cfg, tok, seed=seed)

        cncm_m = CNCMModel(base_cfg, CNCMConceptTokenizer(base_cfg))
        train_stage2(base_cfg, tok, model=cncm_m, dataset=train_ds, mode="joint", steps=steps, batch_size=batch_size, device=device, seed=seed, verbose=False)
        cncm_nll = _eval_exact_nll(cncm_m, test_ds, device)

        ar_m = CNCMModel(matched_cfg, CNCMConceptTokenizer(matched_cfg))
        train_stage2(matched_cfg, tok, model=ar_m, dataset=train_ds, mode="exact_only", steps=steps, batch_size=batch_size, concept_weight=0.0, exact_weight=1.0, device=device, seed=seed, verbose=False)
        ar_nll = _eval_exact_nll(ar_m, test_ds, device)

        return {
            "cncm_test_nll": round(cncm_nll, 5),
            "latency_matched_ar_test_nll": round(ar_nll, 5),
            "nll_delta_cncm_minus_matched_ar": round(cncm_nll - ar_nll, 5),
        }

    res = run_seeded_experiment(
        f"exp_1_2b_latency_matched_ar_{preset}",
        _one_seed,
        seeds=seeds,
        config={"preset": preset, "steps": steps, "matched_ar_overrides": best_cand, "sweep_log": sweep_log},
        steps=steps,
    )
    res["sweep_log"] = sweep_log
    res["selected_ar_overrides"] = best_cand
    return res


# ---------------------------------------------------------------------------
# 1.3: Fair 3-Arm QAT Control (FP32 +40 vs INT4 QAT +40 vs PTQ INT4 +0)
# ---------------------------------------------------------------------------
def run_exp_1_3_fair_qat(
    preset: str = "nano",
    base_steps: int = 100,
    extra_steps: int = 40,
    batch_size: int = 8,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Start from one checkpoint and run 3 arms; credit QAT strictly with (ii) - (i)."""
    _require_torch()
    from cncm.config import get_preset
    from cncm.export.quantize import quantize_model_int4
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.train_stage2 import train_stage2

    cfg = get_preset(preset, enc_layers=1)

    def _one_seed(seed: int) -> Dict[str, Any]:
        tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
        train_ds, _, test_ds = _build_split_datasets(cfg, tok, seed=seed)

        # Shared starting checkpoint
        base_model = CNCMModel(cfg, CNCMConceptTokenizer(cfg))
        train_stage2(cfg, tok, model=base_model, dataset=train_ds, mode="joint", steps=base_steps, batch_size=batch_size, device=device, seed=seed, verbose=False)

        # Arm (iii): PTQ INT4 (+0 extra steps)
        ptq_model = quantize_model_int4(copy.deepcopy(base_model), group_size=32)
        ptq_nll = _eval_exact_nll(ptq_model, test_ds, device)

        # Arm (i): FP32 +40 steps on the same LR schedule (then PTQ INT4 for quantized deployment comparison)
        fp32_plus = copy.deepcopy(base_model)
        train_stage2(cfg, tok, model=fp32_plus, dataset=train_ds, mode="joint", steps=extra_steps, batch_size=batch_size, qat_finetune=False, device=device, seed=seed + 1, verbose=False)
        fp32_plus_nll = _eval_exact_nll(fp32_plus, test_ds, device)
        fp32_plus_ptq_nll = _eval_exact_nll(quantize_model_int4(copy.deepcopy(fp32_plus), group_size=32), test_ds, device)

        # Arm (ii): INT4 QAT +40 steps on the same LR schedule (then pack INT4)
        qat_plus = copy.deepcopy(base_model)
        train_stage2(cfg, tok, model=qat_plus, dataset=train_ds, mode="joint", steps=extra_steps, batch_size=batch_size, qat_finetune=True, qat_bits=4, device=device, seed=seed + 1, verbose=False)
        qat_int4_nll = _eval_exact_nll(quantize_model_int4(copy.deepcopy(qat_plus), group_size=32), test_ds, device)

        return {
            "arm_i_fp32_plus40_nll": round(fp32_plus_nll, 5),
            "arm_i_fp32_plus40_ptq_int4_nll": round(fp32_plus_ptq_nll, 5),
            "arm_ii_qat_int4_plus40_nll": round(qat_int4_nll, 5),
            "arm_iii_ptq_int4_plus0_nll": round(ptq_nll, 5),
            "true_qat_effect_ii_minus_i_fp32": round(qat_int4_nll - fp32_plus_nll, 5),
            "true_qat_effect_ii_minus_i_ptq": round(qat_int4_nll - fp32_plus_ptq_nll, 5),
        }

    return run_seeded_experiment(
        f"exp_1_3_fair_qat_{preset}",
        _one_seed,
        seeds=seeds,
        config={"preset": preset, "base_steps": base_steps, "extra_steps": extra_steps, "batch_size": batch_size},
        steps=base_steps + extra_steps,
    )


# ---------------------------------------------------------------------------
# 1.5: Validation-Only lambda_e x lambda_c Grid Search
# ---------------------------------------------------------------------------
def run_exp_1_5_lambda_grid(
    preset: str = "nano",
    steps: int = 120,
    batch_size: int = 8,
    grid_seeds: Sequence[int] = (42,),
    confirm_seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Grid lambda_e in {0.1, 0.3, 0.6, 1.0} x lambda_c in {0.5, 1.0} on val split; confirm on test split."""
    _require_torch()
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.train_stage2 import train_stage2

    cfg = get_preset(preset, enc_layers=1)
    lambda_es = (0.1, 0.3, 0.6, 1.0)
    lambda_cs = (0.5, 1.0)

    grid_results: List[Dict[str, Any]] = []
    best_pair = (0.5, 1.0)
    best_val_nll = float("inf")

    for le in lambda_es:
        for lc in lambda_cs:
            val_nlls = []
            for seed in grid_seeds:
                set_global_seed(seed)
                tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
                train_ds, val_ds, _ = _build_split_datasets(cfg, tok, seed=seed)
                m = CNCMModel(cfg, CNCMConceptTokenizer(cfg))
                train_stage2(
                    cfg, tok, model=m, dataset=train_ds, mode="joint",
                    steps=steps, batch_size=batch_size,
                    concept_weight=lc, exact_weight=le, device=device, seed=seed, verbose=False,
                )
                val_nlls.append(_eval_exact_nll(m, val_ds, device))
            mean_val = sum(val_nlls) / len(val_nlls)
            grid_results.append({"lambda_e": le, "lambda_c": lc, "val_nlls": val_nlls, "mean_val_nll": round(mean_val, 5)})
            if mean_val < best_val_nll:
                best_val_nll = mean_val
                best_pair = (le, lc)

    best_le, best_lc = best_pair

    def _confirm_seed(seed: int) -> Dict[str, Any]:
        tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
        train_ds, val_ds, test_ds = _build_split_datasets(cfg, tok, seed=seed)
        m = CNCMModel(cfg, CNCMConceptTokenizer(cfg))
        train_stage2(
            cfg, tok, model=m, dataset=train_ds, mode="joint",
            steps=steps, batch_size=batch_size,
            concept_weight=best_lc, exact_weight=best_le, device=device, seed=seed, verbose=False,
        )
        return {
            "val_exact_nll": round(_eval_exact_nll(m, val_ds, device), 5),
            "test_exact_nll": round(_eval_exact_nll(m, test_ds, device), 5),
        }

    confirm = run_seeded_experiment(
        f"exp_1_5_lambda_grid_{preset}",
        _confirm_seed,
        seeds=confirm_seeds,
        config={"preset": preset, "steps": steps, "selected_lambda_e": best_le, "selected_lambda_c": best_lc, "grid": grid_results},
        steps=steps,
    )
    confirm["grid_val_results"] = grid_results
    confirm["selected_on_val"] = {"lambda_e": best_le, "lambda_c": best_lc, "mean_val_nll": round(best_val_nll, 5)}
    return confirm


# ---------------------------------------------------------------------------
# 5.1: Draft-Alignment Self-Distillation
# ---------------------------------------------------------------------------
def run_exp_5_1_draft_align(
    preset: str = "nano",
    stage2_steps: int = 100,
    align_steps: int = 40,
    batch_size: int = 8,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run draft-alignment distillation and report alpha, actual speedup, and expected-speedup formula."""
    torch = _require_torch()
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.draft_align import train_draft_alignment
    from cncm.trainer.train_stage2 import train_stage2

    cfg = get_preset(preset, enc_layers=1)

    def _measure_alpha(m: CNCMModel, ds: Any) -> Dict[str, float]:
        m.eval()
        accepted_total, drafted_total = 0, 0
        with torch.no_grad():
            for i in range(min(8, len(ds))):
                pfx = ds[i]["prefix_ids"].unsqueeze(0).to(device)
                ar_ids = m.generate_exact(pfx, max_new_tokens=cfg.chunk_size, temperature=0.0)["token_ids"]
                c_ids = m.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)["token_ids"]
                # Count prefix-matched consecutive tokens (exact greedy self-speculative acceptance)
                matches = 0
                for t in range(cfg.chunk_size):
                    if int(c_ids[0, t].item()) == int(ar_ids[0, t].item()):
                        matches += 1
                    else:
                        break
                accepted_total += matches
                drafted_total += cfg.chunk_size
        alpha = accepted_total / float(max(1, drafted_total))
        lat = _measure_session_latency(m, cfg, device, iters=2)
        draft_ratio = lat["concept_chunk_ms"] / max(lat["exact_ar_chunk_ms"] / cfg.chunk_size, 1e-6)
        theo = expected_speculative_speedup(alpha=alpha, gamma=cfg.chunk_size, draft_cycle_cost_ratio=draft_ratio, verify_cycle_cost_ratio=1.0)
        return {
            "alpha": round(alpha, 5),
            "expected_speedup_formula": theo["expected_speedup_vs_ar"],
            "expected_tokens_per_cycle": theo["expected_tokens_per_cycle"],
            "concept_speedup_vs_ar": lat["concept_speedup_vs_ar"],
        }

    def _one_seed(seed: int) -> Dict[str, Any]:
        tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
        train_ds, _, test_ds = _build_split_datasets(cfg, tok, seed=seed)
        m = CNCMModel(cfg, CNCMConceptTokenizer(cfg))
        train_stage2(cfg, tok, model=m, dataset=train_ds, mode="joint", steps=stage2_steps, batch_size=batch_size, device=device, seed=seed, verbose=False)
        before = _measure_alpha(m, test_ds)
        train_draft_alignment(m, train_ds, steps=align_steps, batch_size=batch_size, device=device, seed=seed, verbose=False)
        after = _measure_alpha(m, test_ds)
        return {
            "alpha_before": before["alpha"],
            "alpha_after": after["alpha"],
            "alpha_gain": round(after["alpha"] - before["alpha"], 5),
            "expected_speedup_before": before["expected_speedup_formula"],
            "expected_speedup_after": after["expected_speedup_formula"],
        }

    return run_seeded_experiment(
        f"exp_5_1_draft_align_{preset}",
        _one_seed,
        seeds=seeds,
        config={"preset": preset, "stage2_steps": stage2_steps, "align_steps": align_steps},
        steps=stage2_steps + align_steps,
    )


# ---------------------------------------------------------------------------
# 5.2: Looping Ablation 7x3 vs 7x1 at 4x Budget (2000 steps, 3 seeds)
# ---------------------------------------------------------------------------
def run_exp_5_2_looping_ablation(
    preset: str = "nano",
    steps: int = 2000,
    batch_size: int = 8,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare 7x3 (`num_loops=3`) vs 7x1 (`num_loops=1`) at 4x budget across 3 seeds."""
    _require_torch()
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.train_stage2 import train_stage2

    cfg_loop = get_preset(preset, enc_layers=1, num_loops=3)
    cfg_flat = get_preset(preset, enc_layers=1, num_loops=1)

    def _one_seed(seed: int) -> Dict[str, Any]:
        tok = CNCMTextTokenizer(vocab_size=cfg_loop.vocab_size)
        train_ds, _, test_ds = _build_split_datasets(cfg_loop, tok, seed=seed)

        m_loop = CNCMModel(cfg_loop, CNCMConceptTokenizer(cfg_loop))
        train_stage2(cfg_loop, tok, model=m_loop, dataset=train_ds, mode="joint", steps=steps, batch_size=batch_size, device=device, seed=seed, verbose=False)
        nll_loop = _eval_exact_nll(m_loop, test_ds, device)

        m_flat = CNCMModel(cfg_flat, CNCMConceptTokenizer(cfg_flat))
        train_stage2(cfg_flat, tok, model=m_flat, dataset=train_ds, mode="joint", steps=steps, batch_size=batch_size, device=device, seed=seed, verbose=False)
        nll_flat = _eval_exact_nll(m_flat, test_ds, device)

        return {
            "nll_7x3_looped": round(nll_loop, 5),
            "nll_7x1_nonlooped": round(nll_flat, 5),
            "delta_looped_minus_nonlooped": round(nll_loop - nll_flat, 5),
        }

    res = run_seeded_experiment(
        f"exp_5_2_looping_ablation_{preset}",
        _one_seed,
        seeds=seeds,
        config={"preset": preset, "steps": steps, "num_loops_a": 3, "num_loops_b": 1},
        steps=steps,
    )
    a = [r["metrics"]["nll_7x3_looped"] for r in res["per_seed"]]
    b = [r["metrics"]["nll_7x1_nonlooped"] for r in res["per_seed"]]
    if len(a) >= 2:
        pt = paired_t_test(a, b)
        res["paired_test"] = pt
        res["recommended_default_num_loops"] = 1 if pt["mean_diff"] >= 0.0 else 3
    return res




# ---------------------------------------------------------------------------
# 5.3: Tool-Calling Syntax vs Semantic Accuracy (With & Without Copy-Span)
# ---------------------------------------------------------------------------
def run_exp_5_3_copy_span_semantics(
    preset: str = "nano",
    steps: int = 100,
    eval_n: int = 100,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare JSON tool-calling syntax rate vs semantic accuracy with/without copy-span masking."""
    torch = _require_torch()
    from cncm.agent.grammar import JSONSchemaConstrainedMasker
    from cncm.agent.tools import create_default_registry
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.dataset import PackedChunkDataset, make_synthetic_corpus
    from cncm.trainer.train_stage2 import train_stage2
    from experiments.eval_suites import build_heldout_tool_calling_suite, evaluate_tool_suite_with_ci

    set_global_seed(seed)
    cfg = get_preset(preset, enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    reg = create_default_registry()
    train_ds = PackedChunkDataset(make_synthetic_corpus(128, seed=seed), tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    m = CNCMModel(cfg, CNCMConceptTokenizer(cfg))
    train_stage2(cfg, tok, model=m, dataset=train_ds, mode="joint", steps=steps, batch_size=batch_size, device=device, seed=seed, verbose=False)
    m.eval()

    tasks = [t for t in build_heldout_tool_calling_suite(eval_n) if t["expected_tool"] == "calculator"]
    preds_plain, preds_copy = [], []
    with torch.no_grad():
        for t in tasks:
            pfx = torch.tensor([tok.encode(t["prompt"])[-96:]], dtype=torch.long, device=device)
            masker_plain = JSONSchemaConstrainedMasker(tok, reg, allowed_tools=["calculator"], prompt_text=None)
            out_p = m.generate_exact(pfx, max_new_tokens=40, temperature=0.0, logit_masker=masker_plain)["token_ids"]
            preds_plain.append(tok.decode(out_p[0].tolist()))

            masker_copy = JSONSchemaConstrainedMasker(tok, reg, allowed_tools=["calculator"], prompt_text=t["prompt"])
            out_c = m.generate_exact(pfx, max_new_tokens=40, temperature=0.0, logit_masker=masker_copy)["token_ids"]
            preds_copy.append(tok.decode(out_c[0].tolist()))

    report = {
        "n_calculator_tasks": len(tasks),
        "without_copy_span": evaluate_tool_suite_with_ci(preds_plain, tasks),
        "with_copy_span": evaluate_tool_suite_with_ci(preds_copy, tasks),
    }
    save_result(f"exp_5_3_copy_span_{preset}", report, config={"preset": preset, "steps": steps, "eval_n": eval_n}, seeds=[seed], steps=steps)
    return report


# ---------------------------------------------------------------------------
# 5.4: SFT Warm-Start + Partial-Credit GRPO (G=16, T=1.0, Skip Zero-Var)
# ---------------------------------------------------------------------------
def run_exp_5_4_grpo_warmstart(
    preset: str = "nano",
    sft_steps: int = 80,
    grpo_steps: int = 30,
    num_generations: int = 16,
    temperature: float = 1.0,
    batch_size: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run SFT warm-start followed by partial-credit GRPO with zero-variance group skipping."""
    _require_torch()
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.dataset import BlockARSFTDataset, make_synthetic_sft
    from cncm.trainer.train_grpo import train_grpo
    from cncm.trainer.train_stage2 import train_stage2

    set_global_seed(seed)
    cfg = get_preset(preset, enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    sft_ds = BlockARSFTDataset(make_synthetic_sft(128, seed=seed), tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    m = CNCMModel(cfg, CNCMConceptTokenizer(cfg))
    sft_res = train_stage2(cfg, tok, model=m, dataset=sft_ds, mode="sft", steps=sft_steps, batch_size=batch_size, device=device, seed=seed, verbose=False)
    grpo_res = train_grpo(
        cfg, tok, model=m, steps=grpo_steps, num_generations=num_generations,
        batch_size=min(2, batch_size), temperature=temperature,
        skip_zero_variance_groups=True, device=device, seed=seed, verbose=False,
    )
    report = {
        "sft_final_loss": sft_res.get("final_loss"),
        "grpo_mean_reward_history": grpo_res.get("mean_reward_history", []),
        "zero_variance_fraction": grpo_res.get("zero_variance_fraction"),
        "zero_variance_groups": grpo_res.get("zero_variance_groups"),
        "total_groups": grpo_res.get("total_groups"),
    }
    save_result(
        f"exp_5_4_grpo_warmstart_{preset}",
        report,
        config={"preset": preset, "sft_steps": sft_steps, "grpo_steps": grpo_steps, "G": num_generations, "T": temperature},
        seeds=[seed],
        steps=sft_steps + grpo_steps,
    )
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="CNCM Tier A Controlled Remediation Experiments")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["1.1", "1.2b", "1.3", "1.5", "5.1", "5.2", "5.3", "5.4", "all"],
    )
    parser.add_argument("--preset", type=str, default="nano")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out: Dict[str, Any] = {}
    if args.exp in ("1.1", "all"):
        out["1.1"] = run_exp_1_1_equal_budget(preset=args.preset, steps=args.steps, batch_size=args.batch_size, seeds=args.seeds, device=args.device)
    if args.exp in ("1.2b", "all"):
        out["1.2b"] = run_exp_1_2b_latency_matched_ar(preset=args.preset, steps=min(args.steps, 200), batch_size=args.batch_size, seeds=args.seeds, device=args.device)
    if args.exp in ("1.3", "all"):
        out["1.3"] = run_exp_1_3_fair_qat(preset=args.preset, base_steps=min(args.steps, 100), extra_steps=40, batch_size=args.batch_size, seeds=args.seeds, device=args.device)
    if args.exp in ("1.5", "all"):
        out["1.5"] = run_exp_1_5_lambda_grid(preset=args.preset, steps=min(args.steps, 120), batch_size=args.batch_size, confirm_seeds=args.seeds, device=args.device)
    if args.exp in ("5.1", "all"):
        out["5.1"] = run_exp_5_1_draft_align(preset=args.preset, stage2_steps=min(args.steps, 100), align_steps=40, batch_size=args.batch_size, seeds=args.seeds, device=args.device)
    if args.exp in ("5.2", "all"):
        out["5.2"] = run_exp_5_2_looping_ablation(preset=args.preset, steps=args.steps * 4 if args.exp == "5.2" else args.steps, batch_size=args.batch_size, seeds=args.seeds, device=args.device)

    if args.exp in ("5.3", "all"):
        out["5.3"] = run_exp_5_3_copy_span_semantics(preset=args.preset, steps=min(args.steps, 100), batch_size=args.batch_size, seed=args.seeds[0], device=args.device)
    if args.exp in ("5.4", "all"):
        out["5.4"] = run_exp_5_4_grpo_warmstart(preset=args.preset, sft_steps=min(args.steps, 80), grpo_steps=min(args.steps, 30), batch_size=args.batch_size, seed=args.seeds[0], device=args.device)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
