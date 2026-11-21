"""Controlled Multi-Seed Experiment Suite for CNCM Review Remediation (`run_controls.py`).

Covers:
  - 1.1: Equal-step CNCM vs Parameter-Matched Pure-AR (3 seeds + fresh latency re-measure)
  - 1.2b: Latency-matched smaller Pure-AR baseline (matched to CNCM Concept Mode latency, 3 seeds)
  - 1.3: Fair QAT comparison: FP32 (+40 steps) vs INT4 QAT (+40 steps) vs PTQ INT4 (+0 steps), 3 seeds
  - 1.5: Stage 2 dual-mode loss weight grid (lambda_c x lambda_e) selected on VAL, confirmed on TEST (3 seeds)
  - 5.1: Draft-Alignment Training (before vs after self-speculative acceptance alpha & speedup, 3 seeds)
  - 5.2: Recursive Looping (L x 3) vs Non-Looped (L x 1) at 4x longer training budget (3 seeds)
  - 5.4: Productive GRPO RL (SFT warm-start + shaped partial-credit reward + skip zero-variance groups + G=16, 3 seeds)

All results are reported with 3-seed mean +/- 95% t-CI and paired comparison stats,
saved to `experiments/results/controls_summary.json`.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cncm.config import CNCMConfig, get_preset
from cncm.export import quantize_model_int4
from cncm.model import CNCMModel
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer import (
    BlockARSFTDataset,
    GRPOPromptDataset,
    PackedChunkDataset,
    WarmupCosineScheduler,
    make_synthetic_grpo_prompts,
    set_seed,
    train_grpo,
    train_stage1,
    train_stage2,
)
from cncm.trainer.draft_align import train_draft_alignment
from experiments.harness import RESULTS_DIR, build_strict_splits, eval_lm_split, save_result
from experiments.stats import paired_comparison, seed_summary

SEEDS = (42, 137, 2026)


def _train_pure_ar(
    cfg: CNCMConfig,
    train_ds: PackedChunkDataset,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> CNCMModel:
    """Train a Pure-AR baseline with identical AdamW, WarmupCosineScheduler, steps, and data."""
    set_seed(seed)
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).to(device)
    ar_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("concept_tokenizer.")
        and not n.startswith("concept_head.")
        and not n.startswith("concept_in_proj.")
        and not n.startswith("level_embed.")
        and n != "l0_queries"
    ]
    opt = torch.optim.AdamW(ar_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = WarmupCosineScheduler(opt, total_steps=steps, warmup_ratio=0.08, min_lr_ratio=0.1)
    dl = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, drop_last=False)

    def _inf():
        while True:
            for b in dl:
                yield b

    it = _inf()
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        b = next(it)
        inp = b["exact_input_ids"].to(device)
        lbl = b["exact_labels"].to(device)
        msk = b["exact_attention_mask"].to(device)
        out = model.forward_exact(inp, labels=lbl, attention_mask=msk)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(ar_params, 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def _measure_decode_ms(model: CNCMModel, pfx: torch.Tensor, mode: str, iters: int = 5) -> float:
    L = model.cfg.chunk_size
    if mode == "concept":
        _ = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)
        return ((time.perf_counter() - t0) / iters) * 1000.0
    _ = model.generate_exact(pfx, max_new_tokens=L, temperature=0.0)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.generate_exact(pfx, max_new_tokens=L, temperature=0.0)
    return ((time.perf_counter() - t0) / iters) * 1000.0


def run_exp_1_1_and_1_2b(device: torch.device) -> Dict[str, Any]:
    """1.1 Equal-step CNCM vs Parameter-Matched Pure-AR + 1.2b Latency-Matched Smaller AR (3 seeds)."""
    print("\n=== [1.1 & 1.2b] Equal-Step CNCM vs Param-Matched AR & Latency-Matched Smaller AR (3 seeds) ===")
    cfg_cncm = get_preset("tiny_test", enc_layers=1, matryoshka_cutoffs=(3, 4, 5))
    # Param-matched Pure-AR: add 1 unique layer to match CNCM total params (generator + stage1)
    cfg_ar_param = get_preset("tiny_test", num_unique_layers=cfg_cncm.num_unique_layers + 1)
    # Latency-matched smaller Pure-AR: 1 unique layer x 1 loop, smaller width to match Concept Mode ms
    cfg_ar_lat = get_preset(
        "tiny_test",
        num_unique_layers=1,
        num_loops=1,
        hidden_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=96,
    )

    tok = CNCMTextTokenizer(vocab_size=cfg_cncm.vocab_size)
    splits = build_strict_splits(tok, chunk_size=cfg_cncm.chunk_size, prefix_len=cfg_cncm.chunk_size, num_code_docs=240, num_prose_docs=240, seed=42)
    steps = 120
    bs = 16
    lr = 2.0e-3

    cncm_nlls, ar_param_nlls, ar_lat_nlls = [], [], []
    cncm_ms_list, ar_param_ms_list, ar_lat_ms_list = [], [], []
    cncm_concept_losses = []
    param_counts: Dict[str, int] = {}

    sample_pfx = splits["s2_test"][0]["prefix_ids"].unsqueeze(0).to(device)

    for sd in SEEDS:
        s1 = train_stage1(cfg_cncm, tok, splits["s1_train"], steps=100, batch_size=32, lr=3e-3, device=device, seed=sd, verbose=False)["model"].eval()
        cncm_m = train_stage2(
            cfg_cncm, tok, s1, splits["s2_train"], mode="joint", steps=steps,
            batch_size=bs, lr=lr, device=device, seed=sd, verbose=False, num_workers=0,
        )["model"].eval()
        ar_param_m = _train_pure_ar(cfg_ar_param, splits["s2_train"], steps=steps, batch_size=bs, lr=lr, seed=sd, device=device)
        ar_lat_m = _train_pure_ar(cfg_ar_lat, splits["s2_train"], steps=steps, batch_size=bs, lr=lr, seed=sd, device=device)

        if not param_counts:
            param_counts = {
                "cncm_total": cncm_m.count_parameters()["total"],
                "ar_param_matched": sum(
                    p.numel() for n, p in ar_param_m.named_parameters()
                    if not n.startswith(("concept_tokenizer.", "concept_head.", "concept_in_proj.", "level_embed.")) and n != "l0_queries"
                ),
                "ar_latency_matched": sum(
                    p.numel() for n, p in ar_lat_m.named_parameters()
                    if not n.startswith(("concept_tokenizer.", "concept_head.", "concept_in_proj.", "level_embed.")) and n != "l0_queries"
                ),
            }

        ev_c = eval_lm_split(cncm_m, splits["s2_test"], device=device, eval_concept=True, max_batches=12)
        ev_ap = eval_lm_split(ar_param_m, splits["s2_test"], device=device, eval_concept=False, max_batches=12)
        ev_al = eval_lm_split(ar_lat_m, splits["s2_test"], device=device, eval_concept=False, max_batches=12)

        cncm_nlls.append(ev_c["exact_nll"])
        cncm_concept_losses.append(ev_c["concept_loss"])
        ar_param_nlls.append(ev_ap["exact_nll"])
        ar_lat_nlls.append(ev_al["exact_nll"])

        cncm_ms_list.append(_measure_decode_ms(cncm_m, sample_pfx, "concept"))
        ar_param_ms_list.append(_measure_decode_ms(ar_param_m, sample_pfx, "exact"))
        ar_lat_ms_list.append(_measure_decode_ms(ar_lat_m, sample_pfx, "exact"))

    res = {
        "steps_all_arms": steps,
        "seeds": list(SEEDS),
        "param_counts": param_counts,
        "cncm_exact_nll": seed_summary(cncm_nlls),
        "cncm_concept_loss": seed_summary(cncm_concept_losses),
        "ar_param_matched_exact_nll": seed_summary(ar_param_nlls),
        "ar_latency_matched_exact_nll": seed_summary(ar_lat_nlls),
        "paired_cncm_minus_ar_param": paired_comparison(cncm_nlls, ar_param_nlls),
        "paired_cncm_minus_ar_latency_matched": paired_comparison(cncm_nlls, ar_lat_nlls),
        "latency_ms_cncm_concept": seed_summary(cncm_ms_list),
        "latency_ms_ar_param_matched": seed_summary(ar_param_ms_list),
        "latency_ms_ar_latency_matched": seed_summary(ar_lat_ms_list),
    }
    print(f"  CNCM NLL={res['cncm_exact_nll']['mean']:.4f} | Param-AR={res['ar_param_matched_exact_nll']['mean']:.4f} | Lat-AR={res['ar_latency_matched_exact_nll']['mean']:.4f}")
    return res


def run_exp_1_3_fair_qat(device: torch.device) -> Dict[str, Any]:
    """1.3 Fair QAT Comparison: Same base ckpt -> FP32(+40) vs INT4 QAT(+40) vs PTQ INT4(+0), 3 seeds."""
    print("\n=== [1.3] Fair QAT Comparison: FP32(+40) vs INT4 QAT(+40) vs PTQ INT4(+0) (3 seeds) ===")
    cfg = get_preset("tiny_test", enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    splits = build_strict_splits(tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size, num_code_docs=240, num_prose_docs=240, seed=42)

    base_fp32_nlls, fp32_plus40_nlls, ptq_int4_nlls, qat_int4_nlls = [], [], [], []
    extra_steps = 40

    for sd in SEEDS:
        s1 = train_stage1(cfg, tok, splits["s1_train"], steps=100, batch_size=32, lr=3e-3, device=device, seed=sd, verbose=False)["model"].eval()
        base_m = train_stage2(
            cfg, tok, s1, splits["s2_train"], mode="joint", steps=100,
            batch_size=16, lr=2e-3, device=device, seed=sd, verbose=False, num_workers=0,
        )["model"].eval()

        base_fp32_nlls.append(eval_lm_split(base_m, splits["s2_test"], device=device, eval_concept=False, max_batches=12)["exact_nll"])

        # Arm (iii): PTQ INT4 (+0 steps)
        ptq_m = quantize_model_int4(copy.deepcopy(base_m), group_size=32, quantize_embeddings=True).eval()
        ptq_int4_nlls.append(eval_lm_split(ptq_m, splits["s2_test"], device=device, eval_concept=False, max_batches=12)["exact_nll"])

        # Arm (i): FP32 (+40 extra steps, identical LR schedule & seed)
        fp32_40 = train_stage2(
            cfg, tok, s1, splits["s2_train"], mode="joint", steps=extra_steps,
            batch_size=16, lr=5e-4, qat_finetune=False, model=copy.deepcopy(base_m),
            device=device, seed=sd + 1000, verbose=False, num_workers=0,
        )["model"].eval()
        fp32_plus40_nlls.append(eval_lm_split(fp32_40, splits["s2_test"], device=device, eval_concept=False, max_batches=12)["exact_nll"])

        # Arm (ii): INT4 QAT (+40 extra steps, identical LR schedule & seed) -> pack to INT4
        qat_40 = train_stage2(
            cfg, tok, s1, splits["s2_train"], mode="joint", steps=extra_steps,
            batch_size=16, lr=5e-4, qat_finetune=True, qat_bits=4, model=copy.deepcopy(base_m),
            device=device, seed=sd + 1000, verbose=False, num_workers=0,
        )["model"].eval()
        qat_packed = quantize_model_int4(qat_40, group_size=32, quantize_embeddings=True).eval()
        qat_int4_nlls.append(eval_lm_split(qat_packed, splits["s2_test"], device=device, eval_concept=False, max_batches=12)["exact_nll"])

    res = {
        "extra_steps": extra_steps,
        "seeds": list(SEEDS),
        "base_fp32_0steps": seed_summary(base_fp32_nlls),
        "ptq_int4_0steps": seed_summary(ptq_int4_nlls),
        "fp32_plus40_steps": seed_summary(fp32_plus40_nlls),
        "qat_int4_plus40_steps": seed_summary(qat_int4_nlls),
        "true_qat_effect_qat40_minus_fp32_40": paired_comparison(qat_int4_nlls, fp32_plus40_nlls),
        "qat40_minus_ptq0": paired_comparison(qat_int4_nlls, ptq_int4_nlls),
    }
    print(f"  FP32(+40)={res['fp32_plus40_steps']['mean']:.4f} | QAT-INT4(+40)={res['qat_int4_plus40_steps']['mean']:.4f} | PTQ-INT4(+0)={res['ptq_int4_0steps']['mean']:.4f}")
    return res


def run_exp_1_5_loss_weights(device: torch.device) -> Dict[str, Any]:
    """1.5 Stage 2 Loss Weight Tuning: lambda_c in {0.5, 1.0} x lambda_e in {0.1, 0.3, 0.6, 1.0}."""
    print("\n=== [1.5] Dual-Mode Loss Weight Grid (val selection -> untouched test 3-seed confirmation) ===")
    cfg = get_preset("tiny_test", enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    splits = build_strict_splits(tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size, num_code_docs=240, num_prose_docs=240, seed=42)

    s1_shared = train_stage1(cfg, tok, splits["s1_train"], steps=100, batch_size=32, lr=3e-3, device=device, seed=42, verbose=False)["model"].eval()

    grid_results: List[Dict[str, Any]] = []
    for lc in (0.5, 1.0):
        for le in (0.1, 0.3, 0.6, 1.0):
            val_ex, val_c = [], []
            for sd in SEEDS[:2]:
                m = train_stage2(
                    cfg, tok, s1_shared, splits["s2_train"], mode="joint", steps=80,
                    concept_weight=lc, exact_weight=le, batch_size=16, lr=2e-3,
                    device=device, seed=sd, verbose=False, num_workers=0,
                )["model"].eval()
                ev = eval_lm_split(m, splits["s2_val"], device=device, eval_concept=True, max_batches=10)
                val_ex.append(ev["exact_nll"])
                val_c.append(ev["concept_loss"])
            # Pareto score: normalized sum of Exact NLL + 0.25 * Concept Loss on VAL
            score = (sum(val_ex) / len(val_ex)) + 0.25 * (sum(val_c) / len(val_c))
            grid_results.append({
                "lambda_c": lc,
                "lambda_e": le,
                "val_exact_nll": seed_summary(val_ex),
                "val_concept_loss": seed_summary(val_c),
                "val_combined_score": round(score, 5),
            })

    best = min(grid_results, key=lambda r: r["val_combined_score"])
    # Confirm best vs default (1.0, 0.3) on untouched TEST split with 3 seeds
    def_ex, def_c, best_ex, best_c = [], [], [], []
    for sd in SEEDS:
        m_def = train_stage2(
            cfg, tok, s1_shared, splits["s2_train"], mode="joint", steps=100,
            concept_weight=1.0, exact_weight=0.3, batch_size=16, lr=2e-3,
            device=device, seed=sd, verbose=False, num_workers=0,
        )["model"].eval()
        m_best = train_stage2(
            cfg, tok, s1_shared, splits["s2_train"], mode="joint", steps=100,
            concept_weight=best["lambda_c"], exact_weight=best["lambda_e"], batch_size=16, lr=2e-3,
            device=device, seed=sd, verbose=False, num_workers=0,
        )["model"].eval()
        ev_d = eval_lm_split(m_def, splits["s2_test"], device=device, eval_concept=True, max_batches=12)
        ev_b = eval_lm_split(m_best, splits["s2_test"], device=device, eval_concept=True, max_batches=12)
        def_ex.append(ev_d["exact_nll"])
        def_c.append(ev_d["concept_loss"])
        best_ex.append(ev_b["exact_nll"])
        best_c.append(ev_b["concept_loss"])

    res = {
        "grid_val_results": grid_results,
        "selected_on_val": {"lambda_c": best["lambda_c"], "lambda_e": best["lambda_e"]},
        "default_on_test_exact_nll": seed_summary(def_ex),
        "default_on_test_concept_loss": seed_summary(def_c),
        "selected_on_test_exact_nll": seed_summary(best_ex),
        "selected_on_test_concept_loss": seed_summary(best_c),
        "paired_exact_nll_selected_minus_default": paired_comparison(best_ex, def_ex),
    }
    print(f"  Best on VAL: lc={best['lambda_c']}, le={best['lambda_e']} | Test Exact NLL: {res['selected_on_test_exact_nll']['mean']:.4f} vs default {res['default_on_test_exact_nll']['mean']:.4f}")
    return res


def run_exp_5_1_draft_alignment(device: torch.device) -> Dict[str, Any]:
    """5.1 Draft-Alignment Training: Self-Speculative Acceptance alpha & Speedup Before vs After (3 seeds)."""
    print("\n=== [5.1] Draft-Alignment Training for Self-Speculative Decoding (3 seeds) ===")
    cfg = get_preset("tiny_test", enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    splits = build_strict_splits(tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size, num_code_docs=240, num_prose_docs=240, seed=42)

    alpha_before, alpha_after = [], []
    speedup_before, speedup_after = [], []
    exact_match_after = []

    eval_prefixes = [splits["s2_test"][i]["prefix_ids"].unsqueeze(0).to(device) for i in range(6)]

    def _eval_spec(m: CNCMModel) -> Tuple[float, float, bool]:
        accs, spds = [], []
        all_match = True
        with torch.no_grad():
            for pfx in eval_prefixes:
                t0 = time.perf_counter()
                ar_o = m.generate_exact(pfx, max_new_tokens=cfg.chunk_size, temperature=0.0)
                ar_ms = (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                sp_o = m.generate_self_speculative(pfx, max_new_tokens=cfg.chunk_size, temperature=0.0, draft_horizon=8, min_acceptance_rate=0.0)
                sp_ms = (time.perf_counter() - t0) * 1000.0
                accs.append(float(sp_o["acceptance_rate"]))
                spds.append(ar_ms / max(sp_ms, 1e-6))
                if not torch.equal(ar_o["token_ids"], sp_o["token_ids"]):
                    all_match = False
        return sum(accs) / len(accs), sum(spds) / len(spds), all_match

    for sd in SEEDS:
        s1 = train_stage1(cfg, tok, splits["s1_train"], steps=120, batch_size=32, lr=3e-3, device=device, seed=sd, verbose=False)["model"].eval()
        m = train_stage2(
            cfg, tok, s1, splits["s2_train"], mode="joint", steps=100,
            batch_size=16, lr=2e-3, device=device, seed=sd, verbose=False, num_workers=0,
        )["model"].eval()

        a0, s0, _ = _eval_spec(m)
        alpha_before.append(a0)
        speedup_before.append(s0)

        da_res = train_draft_alignment(
            m, splits["s2_train"], steps=50, batch_size=12, lr=1.2e-3,
            kl_weight=0.5, exact_anchor_weight=0.1, device=device, seed=sd + 500, verbose=False,
        )
        m_aligned = da_res["model"]
        a1, s1_spd, match = _eval_spec(m_aligned)
        alpha_after.append(a1)
        speedup_after.append(s1_spd)
        exact_match_after.append(match)

    res = {
        "seeds": list(SEEDS),
        "acceptance_before": seed_summary(alpha_before),
        "acceptance_after": seed_summary(alpha_after),
        "paired_acceptance_gain": paired_comparison(alpha_after, alpha_before),
        "speedup_vs_ar_before": seed_summary(speedup_before),
        "speedup_vs_ar_after": seed_summary(speedup_after),
        "all_greedy_exact_match": all(exact_match_after),
    }
    print(f"  Acceptance alpha: before={res['acceptance_before']['mean']:.4f} -> after={res['acceptance_after']['mean']:.4f} | speedup: {res['speedup_vs_ar_before']['mean']:.3f}x -> {res['speedup_vs_ar_after']['mean']:.3f}x")
    return res


def run_exp_5_2_looping(device: torch.device) -> Dict[str, Any]:
    """5.2 Recursive Looping (2x3=6L) vs Non-Looped (2x1=2L) at 4x longer budget (400 steps, 3 seeds)."""
    print("\n=== [5.2] Recursive Looping (num_loops=3) vs Non-Looped (num_loops=1) at 4x Budget (3 seeds) ===")
    cfg_loop = get_preset("tiny_test", num_loops=3, enc_layers=1)
    cfg_noloop = get_preset("tiny_test", num_loops=1, enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg_loop.vocab_size)
    splits = build_strict_splits(tok, chunk_size=cfg_loop.chunk_size, prefix_len=cfg_loop.chunk_size, num_code_docs=240, num_prose_docs=240, seed=42)

    steps = 400
    loop_ex, noloop_ex, loop_c, noloop_c = [], [], [], []
    for sd in SEEDS:
        s1 = train_stage1(cfg_loop, tok, splits["s1_train"], steps=100, batch_size=32, lr=3e-3, device=device, seed=sd, verbose=False)["model"].eval()
        m_l = train_stage2(cfg_loop, tok, s1, splits["s2_train"], mode="joint", steps=steps, batch_size=16, lr=2e-3, device=device, seed=sd, verbose=False, num_workers=0)["model"].eval()
        m_nl = train_stage2(cfg_noloop, tok, s1, splits["s2_train"], mode="joint", steps=steps, batch_size=16, lr=2e-3, device=device, seed=sd, verbose=False, num_workers=0)["model"].eval()
        ev_l = eval_lm_split(m_l, splits["s2_test"], device=device, eval_concept=True, max_batches=12)
        ev_nl = eval_lm_split(m_nl, splits["s2_test"], device=device, eval_concept=True, max_batches=12)
        loop_ex.append(ev_l["exact_nll"])
        loop_c.append(ev_l["concept_loss"])
        noloop_ex.append(ev_nl["exact_nll"])
        noloop_c.append(ev_nl["concept_loss"])

    res = {
        "steps": steps,
        "seeds": list(SEEDS),
        "looped_exact_nll": seed_summary(loop_ex),
        "non_looped_exact_nll": seed_summary(noloop_ex),
        "paired_exact_nll_loop_minus_noloop": paired_comparison(loop_ex, noloop_ex),
        "looped_concept_loss": seed_summary(loop_c),
        "non_looped_concept_loss": seed_summary(noloop_c),
        "paired_concept_loop_minus_noloop": paired_comparison(loop_c, noloop_c),
        "winner_exact_nll": "looped" if (sum(loop_ex) < sum(noloop_ex)) else "non_looped",
    }
    print(f"  400 steps: Looped NLL={res['looped_exact_nll']['mean']:.4f} vs Non-Looped NLL={res['non_looped_exact_nll']['mean']:.4f} (winner={res['winner_exact_nll']})")
    return res


def run_exp_5_4_productive_rl(device: torch.device) -> Dict[str, Any]:
    """5.4 Productive GRPO RL: SFT warm-start + shaped rewards + G=16 + T=1.0 (3 seeds)."""
    print("\n=== [5.4] Productive GRPO RL (SFT Warm-Start + G=16 + T=1.0, 3 seeds) ===")
    cfg = get_preset("tiny_test", enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    prompts = make_synthetic_grpo_prompts(num_samples=24, seed=42)
    # Build matching SFT warm-start dataset teaching <think> + <tool_call> + answer format
    sft_pairs: List[Dict[str, Any]] = []
    for p in prompts:
        ans = str(p.get("expected_answer", "0"))
        tool = p.get("expected_tool") or "calculator"
        args_d = p.get("expected_args") or {"expression": "1+1"}
        tc = json.dumps({"name": tool, "arguments": args_d}, separators=(",", ":"))
        target = f"<think>solve</think><tool_call>{tc}</tool_call>{ans}"
        sft_pairs.append({"prompt": p["prompt"], "response": target})

    s1_docs = [x["prompt"] + " " + x["response"] for x in sft_pairs] * 4
    s1_ds = PackedChunkDataset(s1_docs, tok, chunk_size=cfg.chunk_size)
    sft_ds = BlockARSFTDataset(sft_pairs, tok, chunk_size=cfg.chunk_size, max_prefix_len=64)
    grpo_ds = GRPOPromptDataset(prompts, tok, max_prompt_len=64)

    init_rewards, final_rewards, non_zero_var = [], [], []
    policy_updates: List[float] = []
    final_kls: List[float] = []
    seen_rewards: List[float] = []
    for sd in SEEDS:
        s1 = train_stage1(cfg, tok, s1_ds, steps=80, batch_size=16, lr=3e-3, device=device, seed=sd, verbose=False)["model"].eval()
        # SFT warm-start so rollouts land near the format manifold (breaks zero-variance group collapse!)
        m_sft = train_stage2(
            cfg, tok, s1, sft_ds, mode="joint", steps=80,
            batch_size=8, lr=2e-3, device=device, seed=sd, verbose=False, num_workers=0,
        )["model"]

        grpo_res = train_grpo(
            cfg=cfg,
            tokenizer=tok,
            model=m_sft,
            prompts=grpo_ds,
            steps=10,
            num_generations=16,
            batch_size=2,
            temperature=1.0,
            lr=3e-4,
            device=device,
            seed=sd,
            verbose=False,
        )
        r_hist = grpo_res.get("mean_reward_history") or [0.0]
        init_rewards.append(float(r_hist[0]))
        final_rewards.append(float(r_hist[-1]))
        seen_rewards.extend(float(r) for r in r_hist)
        policy_updates.append(1.0 if grpo_res.get("policy_updated") else 0.0)
        final_kls.append(float(grpo_res.get("final_kl", 0.0)))
        non_zero_var.append(1.0 if float(max(r_hist)) > 0.0 else 0.0)

    res = {
        "seeds": list(SEEDS),
        "num_generations_G": 16,
        "temperature": 1.0,
        "sft_warm_start_steps": 80,
        "grpo_steps": 10,
        "initial_mean_reward": seed_summary(init_rewards),
        "final_mean_reward": seed_summary(final_rewards),
        "paired_reward_gain": paired_comparison(final_rewards, init_rewards),
        "positive_reward_seed_rate": round(sum(non_zero_var) / len(non_zero_var), 4),
        "max_step_mean_reward": round(max(seen_rewards), 5) if seen_rewards else 0.0,
        "policy_updated_seed_rate": round(sum(policy_updates) / len(policy_updates), 4),
        "mean_final_kl": round(sum(final_kls) / len(final_kls), 6) if final_kls else None,
    }
    print(f"  GRPO Reward: init={res['initial_mean_reward']['mean']:.4f} -> final={res['final_mean_reward']['mean']:.4f}")
    return res


def main() -> None:
    torch.set_num_threads(min(8, 8))
    device = torch.device("cpu")
    t0 = time.perf_counter()

    runners = {
        "exp_1_1_and_1_2b": run_exp_1_1_and_1_2b,
        "exp_1_3_fair_qat": run_exp_1_3_fair_qat,
        "exp_1_5_loss_weights": run_exp_1_5_loss_weights,
        "exp_5_1_draft_alignment": run_exp_5_1_draft_alignment,
        "exp_5_2_looping": run_exp_5_2_looping,
        "exp_5_4_productive_rl": run_exp_5_4_productive_rl,
    }
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    if wanted:
        unknown = [w for w in wanted if w not in runners]
        if unknown:
            raise SystemExit(f"unknown experiment(s): {unknown}; choose from {list(runners)}")

    summary: Dict[str, Any] = {}
    prev_total = 0.0
    prev = RESULTS_DIR / "controls_summary.json"
    if wanted and prev.exists():
        summary = json.loads(prev.read_text(encoding="utf-8"))
        summary.pop("experiment", None)
        prev_total = float(summary.pop("total_elapsed_sec", 0.0) or 0.0)
    for name, fn in runners.items():
        if wanted and name not in wanted:
            continue
        try:
            summary[name] = fn(device)
        except Exception as exc:  # keep partial suite progress if one arm fails
            import traceback

            traceback.print_exc()
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"}
        # Persist after every experiment so an interrupted suite keeps its results.
        if not wanted:  # only meaningful for a complete-suite run
            summary["total_elapsed_sec"] = round(prev_total + time.perf_counter() - t0, 2)
        save_result("controls_summary", summary)
    out = save_result("controls_summary", summary)
    print(f"\n[CONTROLS COMPLETE] Saved -> {out} ({summary['total_elapsed_sec']}s)")


if __name__ == "__main__":
    main()
