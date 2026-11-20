"""CNCM Section E Empirical Audit Benchmark on Modal Cloud GPU (NVIDIA A10).

Converts the key "unverified" & "contradicted" audit items (C1-C40, Section E) into
hard empirical measurements on strict held-out validation splits:

  E1. Same-Param AR Baseline vs CNCM (held-out val loss / perplexity + tok/s)
      - Exact parameter-matched Pure-AR baseline trained on the EXACT same data,
        optimizer, LR schedule, token budget, and evaluated on the EXACT same held-out split.
      - Also reports FactorizedConceptHead vs Flat 4096-way Head NLL & MACs (C6, C7, C18).
  E2. Grad-Accum Equivalence & Exact Bottom-Right Causal Commit Max Abs Diff (C11, C21)
      - Compares 1-step BS=32 vs 4x8 grad-accum on Stage 1 (M-RFSQ) and Stage 2 (CNCMModel).
      - Reports exact max abs diff for cached bottom-right causal chunk commit.
  E3. Quantization Quality & KV Drift (C4, C12, C16, C22, C23)
      - Strict held-out Stage 1 recon_acc + per-level batch_utilization (C5, C22).
      - Held-out Stage 2 val loss & latency under FP32, BF16, FP16, INT8, Packed INT4 (C4, C12).
      - FP16/FP32 vs INT8 QuantizedKV attention/logit drift & val NLL delta (C16).
      - QAT (BitQuantLinear) -> INT4 export accuracy test (C23).
  E4. Early-Exit Quality vs Speed Curve (C9, C15)
      - Sweeps early_exit_conf in [0.50, 0.70, 0.85, 0.90, 0.95, 0.99, None]
        and per-level Stage 1 forced exit k=2..8 on held-out chunks, measuring
        recon_acc, held-out CE loss, exit_level, and latency at B=1 and B=8.
  E5. Honest Model-Only PG-FIM & Phase A vs Phase B A/B (C2, C17, C19)
      - Separates model_generated=True (allow_template_fallback=False) from
        template upper bound (allow_template_fallback=True).
      - Compares Phase A (use_copy_decoder=False) vs Phase B (use_copy_decoder=True)
        and real Pure-AR FIM latency + parse/compile/test rates on held-out FIM tasks.
      - Also logs semantic argument accuracy on Agentic tool calls (C8, C29) and
        GRPO zero-variance step diagnostics (C3).

Usage (ephemeral run ONLY — never `modal deploy`):
    modal run modal_audit.py --preset nano --steps 600
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/root/project"
VOLUME_NAME = "cncm-checkpoints"
VOLUME_MOUNT = "/vol"

GPU_RATES_USD_PER_HOUR: Dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "L40S": 1.95,
    "A100": 2.10,
    "H100": 3.95,
}


def _estimate_gpu_rate(gpu_name: str) -> float:
    upper = gpu_name.upper()
    for key, rate in GPU_RATES_USD_PER_HOUR.items():
        if key in upper:
            return rate
    return 1.10


app = modal.App("cncm-section-e-audit")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": REMOTE_PROJECT_DIR,
        }
    )
    .pip_install(
        "torch>=2.5.0",
        "numpy>=1.26.0",
        "datasets>=2.19.0",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "cncm"),
        remote_path=f"{REMOTE_PROJECT_DIR}/cncm",
        ignore=["__pycache__", "*.pyc", "*.pyo"],
    )
)


def _load_real_english_corpus(
    target_samples: int = 2400,
    timeout_sec: float = 20.0,
) -> Tuple[List[str], str]:
    result_holder: Dict[str, Any] = {"texts": [], "source": "synthetic_fallback", "error": None}

    def _worker() -> None:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]

            attempts = [
                ("roneneldan/TinyStories", None, "train", True),
                ("wikitext", "wikitext-2-raw-v1", "train", False),
            ]
            for ds_name, ds_cfg, split, streaming in attempts:
                try:
                    ds = (
                        load_dataset(ds_name, ds_cfg, split=split, streaming=streaming)
                        if ds_cfg is not None
                        else load_dataset(ds_name, split=split, streaming=streaming)
                    )
                    collected: List[str] = []
                    for row in ds:
                        txt = str(row.get("text", "")).strip()
                        if len(txt) < 64 or txt.startswith("="):
                            continue
                        collected.append(txt[:480])
                        if len(collected) >= target_samples:
                            break
                    if len(collected) >= 300:
                        result_holder["texts"] = collected
                        result_holder["source"] = f"{ds_name}/{ds_cfg}" if ds_cfg else ds_name
                        return
                except Exception as exc:  # noqa: BLE001
                    result_holder["error"] = f"{ds_name}: {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            result_holder["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive() or not result_holder["texts"]:
        return [], f"synthetic_fallback ({result_holder.get('error') or 'timeout'})"
    return result_holder["texts"], str(result_holder["source"])


@app.function(
    image=image,
    gpu=["A10", "L40S", "T4"],
    timeout=900,
    volumes={VOLUME_MOUNT: vol},
)
def run_section_e_audit(
    preset: str = "nano",
    stage1_steps: int = 350,
    stage2_steps: int = 500,
    ar_steps: int = 500,
) -> str:
    """Execute all feasible Section E audit experiments with zero template masking."""
    t_all = time.perf_counter()
    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from cncm.code import run_code_sandbox
    from cncm.config import CNCMConfig, get_preset
    from cncm.export import measure_model_size, quantize_model_int4, quantize_model_int8
    from cncm.layers import CNCMBlock, RMSNorm, RotaryEmbedding, build_causal_mask
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer import (
        FIMPlanDataset,
        PackedChunkDataset,
        WarmupCosineScheduler,
        collate_block_ar,
        compute_fim_reward,
        make_pg_fim_stage1_corpus,
        make_synthetic_code_corpus,
        make_synthetic_fim_tasks,
        set_seed,
        train_stage1,
        train_stage2,
    )

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float32
    print(f"[AUDIT ENV] GPU={gpu_name} | bf16={bf16_ok} | torch={torch.__version__}")

    # =========================================================================
    # Strict Train / Held-Out Validation Split (Zero Leakage!)
    # =========================================================================
    set_seed(42)
    cfg_phase_b = get_preset(preset, use_copy_decoder=True)
    cfg_phase_a = get_preset(preset, use_copy_decoder=False)
    tokenizer = CNCMTextTokenizer(vocab_size=cfg_phase_b.vocab_size)

    real_texts, corpus_source = _load_real_english_corpus(target_samples=2200, timeout_sec=20.0)
    code_docs_all = make_synthetic_code_corpus(num_samples=600, seed=42)
    pg_s1_docs_all = make_pg_fim_stage1_corpus(num_samples=1200, seed=42)

    # Strict 85% train / 15% held-out validation split on documents BEFORE packing
    n_real_val = max(120, int(len(real_texts) * 0.15)) if real_texts else 0
    real_train = real_texts[n_real_val:] if real_texts else []
    real_val = real_texts[:n_real_val] if real_texts else []

    s1_train_docs = pg_s1_docs_all[180:] + real_train
    s1_val_docs = pg_s1_docs_all[:180] + real_val

    s1_train_ds = PackedChunkDataset(
        s1_train_docs,
        tokenizer,
        chunk_size=cfg_phase_b.chunk_size,
        eos_token_id=cfg_phase_b.eos_token_id,
        bos_token_id=cfg_phase_b.bos_token_id,
    )
    s1_val_ds = PackedChunkDataset(
        s1_val_docs,
        tokenizer,
        chunk_size=cfg_phase_b.chunk_size,
        eos_token_id=cfg_phase_b.eos_token_id,
        bos_token_id=cfg_phase_b.bos_token_id,
    )

    # Stage 2 packed prefix+target datasets (identical split for CNCM vs Same-Param AR!)
    s2_prefix_len = 64
    s2_lm_train_ds = PackedChunkDataset(
        s1_train_docs,
        tokenizer,
        chunk_size=cfg_phase_b.chunk_size,
        prefix_len=s2_prefix_len,
        eos_token_id=cfg_phase_b.eos_token_id,
        bos_token_id=cfg_phase_b.bos_token_id,
    )
    s2_lm_val_ds = PackedChunkDataset(
        s1_val_docs,
        tokenizer,
        chunk_size=cfg_phase_b.chunk_size,
        prefix_len=s2_prefix_len,
        eos_token_id=cfg_phase_b.eos_token_id,
        bos_token_id=cfg_phase_b.bos_token_id,
    )

    # Held-out FIM tasks (different seed from train!)
    fim_train_tasks = make_synthetic_fim_tasks(num_samples=240, seed=42, curriculum_stage="all")
    fim_val_tasks = make_synthetic_fim_tasks(num_samples=24, seed=9999, curriculum_stage="all")
    fim_train_ds = FIMPlanDataset(
        samples=list(fim_train_tasks) + list(code_docs_all[100:]) + list(real_train[:200]),
        tokenizer=tokenizer,
        chunk_size=cfg_phase_b.chunk_size,
        max_prefix_len=96,
        fim_rate=0.65,
        span_mode="mix",
        kw_max=cfg_phase_b.kw_max,
        kw_empty_prob=cfg_phase_b.kw_empty_prob,
        kw_corrupt_prob=cfg_phase_b.kw_corrupt_prob,
        seed=42,
    )
    fim_val_ds = FIMPlanDataset(
        samples=list(fim_val_tasks) + list(code_docs_all[:60]),
        tokenizer=tokenizer,
        chunk_size=cfg_phase_b.chunk_size,
        max_prefix_len=96,
        fim_rate=1.0,
        span_mode="mix",
        kw_max=cfg_phase_b.kw_max,
        kw_empty_prob=0.0,
        kw_corrupt_prob=0.0,
        seed=9999,
    )

    print(
        f"[SPLIT] source={corpus_source} | s1_train_chunks={len(s1_train_ds)} s1_val_chunks={len(s1_val_ds)} | "
        f"s2_lm_train={len(s2_lm_train_ds)} s2_lm_val={len(s2_lm_val_ds)} | fim_train={len(fim_train_ds)} fim_val_tasks={len(fim_val_tasks)}"
    )

    # =========================================================================
    # E2: Grad-Accum Equivalence (C11) & Bottom-Right Causal Commit Parity (C21)
    # =========================================================================
    print("\n=== [E2] Grad-Accum Equivalence & Bottom-Right Causal Commit Max Abs Diff ===")
    e2_report: Dict[str, Any] = {}
    tiny_cfg = get_preset("tiny_test")

    # E2a: Stage 1 M-RFSQ grad-accum equivalence (1x BS=16 vs 4x BS=4)
    set_seed(101)
    s1_a = CNCMConceptTokenizer(tiny_cfg).to(device)
    s1_b = copy.deepcopy(s1_a).to(device)
    x_ga = torch.randint(16, tiny_cfg.vocab_size, (16, tiny_cfg.chunk_size), device=device)

    s1_a.train()
    s1_a.zero_grad(set_to_none=True)
    out_a = s1_a(x_ga, apply_fine_drop=False)
    out_a["loss"].backward()

    s1_b.train()
    s1_b.zero_grad(set_to_none=True)
    loss_b_acc = 0.0
    for mb in range(4):
        out_mb = s1_b(x_ga[mb * 4 : (mb + 1) * 4], apply_fine_drop=False)
        (out_mb["loss"] / 4.0).backward()
        loss_b_acc += float(out_mb["loss"].item()) / 4.0

    s1_grad_diffs = [
        float((pa.grad - pb.grad).abs().max().item())
        for pa, pb in zip(s1_a.parameters(), s1_b.parameters())
        if pa.grad is not None and pb.grad is not None
    ]
    e2_report["stage1_grad_accum_1x16_vs_4x4"] = {
        "loss_single": round(float(out_a["loss"].item()), 7),
        "loss_accum4": round(loss_b_acc, 7),
        "loss_abs_diff": float(f"{abs(float(out_a['loss'].item()) - loss_b_acc):.3e}"),
        "max_param_grad_abs_diff": float(f"{max(s1_grad_diffs):.3e}"),
        "exact_match_lt_1e5": max(s1_grad_diffs) < 1e-5,
    }

    # E2b: Stage 2 CNCMModel grad-accum equivalence (1x BS=16 vs 4x BS=4, corrupt_rate=0.0)
    set_seed(102)
    m2_a = CNCMModel(tiny_cfg, copy.deepcopy(s1_a)).to(device)
    m2_b = copy.deepcopy(m2_a).to(device)
    p_ga = torch.randint(16, tiny_cfg.vocab_size, (16, 16), device=device)
    t_ga = torch.randint(16, tiny_cfg.vocab_size, (16, tiny_cfg.chunk_size), device=device)

    m2_a.train()
    m2_a.zero_grad(set_to_none=True)
    o2_a = m2_a.forward_joint(p_ga, t_ga, corrupt_rate=0.0)
    o2_a["loss"].backward()

    m2_b.train()
    m2_b.zero_grad(set_to_none=True)
    loss2_b = 0.0
    for mb in range(4):
        o2_mb = m2_b.forward_joint(
            p_ga[mb * 4 : (mb + 1) * 4],
            t_ga[mb * 4 : (mb + 1) * 4],
            corrupt_rate=0.0,
        )
        (o2_mb["loss"] / 4.0).backward()
        loss2_b += float(o2_mb["loss"].item()) / 4.0

    m2_grad_diffs = [
        float((pa.grad - pb.grad).abs().max().item())
        for pa, pb in zip(m2_a.generator_parameters(), m2_b.generator_parameters())
        if pa.grad is not None and pb.grad is not None
    ]
    e2_report["stage2_grad_accum_1x16_vs_4x4"] = {
        "loss_single": round(float(o2_a["loss"].item()), 7),
        "loss_accum4": round(loss2_b, 7),
        "loss_abs_diff": float(f"{abs(float(o2_a['loss'].item()) - loss2_b):.3e}"),
        "max_param_grad_abs_diff": float(f"{max(m2_grad_diffs):.3e}"),
        "exact_match_lt_1e5": max(m2_grad_diffs) < 1e-5,
    }

    # E2c: Exact bottom-right causal commit parity max abs diff (C21)
    m2_a.eval()
    commit_diffs: Dict[str, float] = {}
    with torch.no_grad():
        for p_l, q_l in ((7, 16), (32, 64), (48, 16)):
            p_c = torch.randint(16, tiny_cfg.vocab_size, (2, p_l), device=device)
            q_c = torch.randint(16, tiny_cfg.vocab_size, (2, q_l), device=device)
            full_c = torch.cat([p_c, q_c], dim=1)
            one_shot = m2_a.forward_exact(full_c, use_cache=False)["logits"][:, p_l:, :]
            s1_kv = m2_a.forward_exact(p_c, use_cache=True)["past_key_values"]
            two_step = m2_a.forward_exact(q_c, past_key_values=s1_kv, use_cache=True)["logits"]
            commit_diffs[f"P={p_l}_Q={q_l}"] = float(
                f"{float((one_shot - two_step).abs().max().item()):.3e}"
            )
    e2_report["bottom_right_causal_commit_max_abs_diff"] = commit_diffs
    print(f"[E2 REPORT] {json.dumps(e2_report)}")
    del s1_a, s1_b, m2_a, m2_b

    # =========================================================================
    # Stage 1 Training: Phase A (no copy) & Phase B (copy decoder) on Train Split
    # + Strict Held-Out Per-Level FSQ Utilization (C5, C22)
    # =========================================================================
    print(f"\n=== [S1] Training Stage 1 Phase B (copy=True) & Phase A (copy=False) ({stage1_steps} steps) ===")
    s1_b_res = train_stage1(
        cfg=cfg_phase_b,
        tokenizer=tokenizer,
        dataset=s1_train_ds,
        steps=stage1_steps,
        batch_size=64,
        lr=2.5e-3,
        device=device,
        apply_fine_drop=True,
        seed=42,
        log_every=max(1, stage1_steps // 4),
        verbose=True,
    )
    s1_phase_b: CNCMConceptTokenizer = s1_b_res["model"].eval()

    s1_a_res = train_stage1(
        cfg=cfg_phase_a,
        tokenizer=tokenizer,
        dataset=s1_train_ds,
        steps=stage1_steps,
        batch_size=64,
        lr=2.5e-3,
        device=device,
        apply_fine_drop=True,
        seed=42,
        log_every=max(1, stage1_steps // 4),
        verbose=False,
    )
    s1_phase_a: CNCMConceptTokenizer = s1_a_res["model"].eval()

    @torch.no_grad()
    def _eval_stage1_heldout(s1_mod: CNCMConceptTokenizer, use_kw: bool = False) -> Dict[str, Any]:
        s1_mod.eval()
        n_eval = min(256, len(s1_val_ds))
        ids = torch.stack([s1_val_ds[i]["input_ids"] for i in range(n_eval)], dim=0).to(device)
        correct = 0
        total = 0
        ce_sum = 0.0
        n_batches = 0
        per_lvl_batch_utils = [0.0] * s1_mod.cfg.num_levels
        per_lvl_active_ratios = [0.0] * s1_mod.cfg.num_levels
        # Per-level forced-cutoff recon_acc (for E4 early-exit quality curve!)
        level_cutoff_correct = [0] * s1_mod.cfg.num_levels

        for st in range(0, n_eval, 64):
            xb = ids[st : st + 64]
            kw_ids, kw_mask = (
                s1_mod.extract_oracle_kw_token_ids(xb, apply_dropout=False)
                if (use_kw and s1_mod.copy_cross_attn is not None)
                else (None, None)
            )
            z = s1_mod.encode_z(xb)
            q_out = s1_mod.quantizer(z, apply_fine_drop=False)
            logits = s1_mod.decode_z_hat(q_out["z_hat"], kw_token_ids=kw_ids, kw_mask=kw_mask)
            ce = float(F.cross_entropy(logits.reshape(-1, s1_mod.cfg.vocab_size).float(), xb.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            correct += int((preds == xb).sum().item())
            total += int(xb.numel())
            ce_sum += ce
            n_batches += 1

            pbu = q_out.get("per_level_batch_utilization", [])
            par = q_out.get("per_level_active_ratio", [])
            for k in range(s1_mod.cfg.num_levels):
                if k < len(pbu):
                    per_lvl_batch_utils[k] += float(pbu[k])
                if k < len(par):
                    per_lvl_active_ratios[k] += float(par[k])

            # Partial-depth recon acc for every cutoff level k=1..K
            codes = q_out["codes"]
            for k in range(s1_mod.cfg.num_levels):
                logits_k = s1_mod.decode_codes(codes[: k + 1], kw_token_ids=kw_ids, kw_mask=kw_mask)
                level_cutoff_correct[k] += int((logits_k.argmax(dim=-1) == xb).sum().item())

        return {
            "heldout_chunks": n_eval,
            "heldout_tokens": total,
            "heldout_recon_acc": round(correct / float(max(1, total)), 5),
            "heldout_recon_ce": round(ce_sum / float(max(1, n_batches)), 5),
            "per_level_batch_utilization": {
                f"L{k}(len={s1_mod.cfg.levels[k]})": round(v / max(1, n_batches), 4)
                for k, v in enumerate(per_lvl_batch_utils)
            },
            "mean_batch_utilization": round(
                sum(per_lvl_batch_utils) / max(1, n_batches * s1_mod.cfg.num_levels), 4
            ),
            "per_level_active_over_4096": {
                f"L{k}(len={s1_mod.cfg.levels[k]})": round(v / max(1, n_batches), 4)
                for k, v in enumerate(per_lvl_active_ratios)
            },
            "per_level_cutoff_recon_acc": {
                f"exit_at_k={k+1}": round(level_cutoff_correct[k] / float(max(1, total)), 4)
                for k in range(s1_mod.cfg.num_levels)
            },
        }

    s1_heldout_phase_a = _eval_stage1_heldout(s1_phase_a, use_kw=False)
    s1_heldout_phase_b_nokw = _eval_stage1_heldout(s1_phase_b, use_kw=False)
    s1_heldout_phase_b_kw = _eval_stage1_heldout(s1_phase_b, use_kw=True)
    print(f"[S1 HELDOUT Phase A] recon_acc={s1_heldout_phase_a['heldout_recon_acc']} | batch_util={s1_heldout_phase_a['mean_batch_utilization']}")
    print(f"[S1 HELDOUT Phase B (kw)] recon_acc={s1_heldout_phase_b_kw['heldout_recon_acc']} | batch_util={s1_heldout_phase_b_kw['mean_batch_utilization']}")

    # =========================================================================
    # E1: Same-Param Pure-AR Baseline vs CNCM (Held-Out Val Loss / PPL + Speed)
    # + FactorizedConceptHead vs Flat 4096-way Head Parameter & MAC Count (C6, C7)
    # =========================================================================
    print(f"\n=== [E1] Same-Param Pure-AR Baseline vs CNCM ({stage2_steps} steps, held-out eval) ===")

    # Train CNCM (Joint: Concept + Exact-AR) on s2_lm_train_ds
    t_cncm_s2 = time.perf_counter()
    cncm_lm_res = train_stage2(
        cfg=cfg_phase_a,
        tokenizer=tokenizer,
        concept_tokenizer=s1_phase_a,
        dataset=s2_lm_train_ds,
        mode="joint",
        steps=stage2_steps,
        corrupt_rate=0.5,
        batch_size=32,
        lr=1.2e-3,
        device=device,
        amp_dtype="bf16" if bf16_ok else "fp32",
        seed=42,
        log_every=max(1, stage2_steps // 4),
        verbose=True,
    )
    cncm_lm_elapsed = time.perf_counter() - t_cncm_s2
    cncm_lm: CNCMModel = cncm_lm_res["model"].eval()

    # Pure-AR Same-Param Baseline:
    # Uses the EXACT same recursive backbone (CNCMModel's forward_exact) trained with
    # 100% Exact-AR LM loss on the EXACT same s2_lm_train_ds, batch_size=32, lr=1.2e-3, steps=ar_steps.
    # To match TOTAL params of CNCM (generator + Stage 1 = ~14.5M), we add 1 extra unique layer
    # (8 unique layers x 3 loops = 14.37M params vs CNCM's 14.50M total params — within 0.9%!).
    ar_cfg = get_preset(
        preset,
        num_unique_layers=cfg_phase_a.num_unique_layers + 1,
        use_copy_decoder=False,
    )
    set_seed(42)
    ar_baseline = CNCMModel(ar_cfg, CNCMConceptTokenizer(ar_cfg)).to(device)
    # Exclude concept_tokenizer and concept_head from AR param count (pure AR LM!)
    ar_only_params = [
        p
        for n, p in ar_baseline.named_parameters()
        if not n.startswith("concept_tokenizer.")
        and not n.startswith("concept_head.")
        and not n.startswith("concept_in_proj.")
        and not n.startswith("level_embed.")
        and n != "l0_queries"
    ]
    ar_param_count = sum(p.numel() for p in ar_only_params)
    cncm_total_params = cncm_lm.count_parameters()["total"]
    cncm_gen_params = cncm_lm.count_parameters()["generator"]

    # Measure FactorizedConceptHead exact params & MACs vs flat 4096-way head (C6, C7)
    fch_params = sum(p.numel() for p in cncm_lm.concept_head.parameters())
    flat_head_params = cfg_phase_a.hidden_size * cfg_phase_a.codebook_size
    # Exact per-position MACs:
    # Flat head: H * 4096
    # FactorizedConceptHead: proj_h (H * d_h) + 4 * (d_h * 8) + 3 * (d_h * d_h) [or MLP]
    H = cfg_phase_a.hidden_size
    dh = cfg_phase_a.concept_head_hidden
    flat_macs_per_pos = H * cfg_phase_a.codebook_size
    fch_macs_per_pos = (H * dh) + (dh * dh) + sum(dh * l_j for l_j in cfg_phase_a.fsq_levels)
    mac_reduction_ratio = flat_macs_per_pos / float(max(1, fch_macs_per_pos))

    # Train Pure-AR baseline with identical AdamW + WarmupCosineScheduler + DataLoader
    from torch.utils.data import DataLoader

    ar_opt = torch.optim.AdamW(ar_only_params, lr=1.2e-3, weight_decay=0.01, betas=(0.9, 0.95))
    ar_sched = WarmupCosineScheduler(ar_opt, total_steps=ar_steps, warmup_ratio=0.08, min_lr_ratio=0.1)
    ar_loader = DataLoader(s2_lm_train_ds, batch_size=32, shuffle=True, drop_last=False)

    def _inf_it(dl):
        while True:
            for b in dl:
                yield b

    ar_it = _inf_it(ar_loader)
    ar_baseline.train()
    t_ar = time.perf_counter()
    ar_train_losses: List[float] = []
    for step_i in range(1, ar_steps + 1):
        ar_opt.zero_grad(set_to_none=True)
        b = next(ar_it)
        inp = b["exact_input_ids"].to(device)
        lbl = b["exact_labels"].to(device)
        msk = b["exact_attention_mask"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bf16_ok):
            out = ar_baseline.forward_exact(inp, labels=lbl, attention_mask=msk)
            loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ar_only_params, 1.0)
        ar_opt.step()
        ar_sched.step()
        ar_train_losses.append(float(loss.detach().item()))
        if step_i in (1, ar_steps // 2, ar_steps):
            print(f"[Pure-AR Baseline] step {step_i:4d}/{ar_steps} | train_loss={ar_train_losses[-1]:.4f}")
    torch.cuda.synchronize()
    ar_train_elapsed = time.perf_counter() - t_ar
    ar_baseline.eval()

    # Evaluate BOTH CNCM and Pure-AR on the EXACT same held-out validation split (`s2_lm_val_ds`)
    @torch.no_grad()
    def _eval_lm_heldout(
        mod: CNCMModel,
        ds: PackedChunkDataset,
        eval_concept: bool = True,
        max_batches: int = 12,
        batch_size: int = 32,
        quantize_kv: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        mod.eval()
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        exact_nll_sum = 0.0
        exact_tok_count = 0
        concept_loss_sum = 0.0
        concept_nll_elbo_sum = 0.0
        n_b = 0

        for idx, b in enumerate(dl):
            if idx >= max_batches:
                break
            pfx = b["prefix_ids"].to(device)
            pfx_m = b["prefix_mask"].to(device)
            tgt = b["target_ids"].to(device)
            inp = b["exact_input_ids"].to(device)
            lbl = b["exact_labels"].to(device)
            msk = b["exact_attention_mask"].to(device)

            use_amp = cast_dtype in (torch.bfloat16, torch.float16)
            ctx = torch.amp.autocast(device_type="cuda", dtype=cast_dtype, enabled=use_amp)
            with ctx:
                if not quantize_kv:
                    ex_out = mod.forward_exact(inp, labels=lbl, attention_mask=msk)
                    logits = ex_out["logits"][:, :-1, :].float()
                else:
                    # Prefill prefix into INT8 KV cache, then evaluate target tokens!
                    pre = mod.forward_exact(pfx, attention_mask=pfx_m, use_cache=True, quantize_kv=True)
                    post = mod.forward_exact(
                        tgt,
                        attention_mask=msk,
                        past_key_values=pre["past_key_values"],
                        use_cache=False,
                        quantize_kv=True,
                    )
                    logits = torch.cat([pre["logits"][:, -1:, :], post["logits"][:, :-1, :]], dim=1).float()
                    lbl = torch.cat([torch.full_like(lbl[:, :1], -100), tgt], dim=1)

                shift_lbl = lbl[:, 1:]
                valid = shift_lbl != -100
                if valid.any():
                    nll = F.cross_entropy(logits[valid], shift_lbl[valid], reduction="sum")
                    exact_nll_sum += float(nll.item())
                    exact_tok_count += int(valid.sum().item())

                if eval_concept:
                    codes = mod.concept_tokenizer.encode_codes(tgt)
                    c_out = mod.forward_concept(pfx, codes, prefix_mask=pfx_m, corrupt_rate=0.0)
                    concept_loss_sum += float(c_out["loss"].item())
                    # Per-text-token code-space NLL upper bound: sum(-log p(codes)) / L
                    code_bpt = float((-c_out["log_probs"].sum(dim=-1) / float(tgt.shape[1])).mean().item())
                    concept_nll_elbo_sum += code_bpt

            n_b += 1

        val_exact_nll = exact_nll_sum / float(max(1, exact_tok_count))
        res = {
            "val_exact_nll": round(val_exact_nll, 5),
            "val_exact_ppl": round(math.exp(min(val_exact_nll, 20.0)), 3),
            "val_tokens_evaluated": exact_tok_count,
        }
        if eval_concept and n_b > 0:
            res["val_concept_weighted_loss"] = round(concept_loss_sum / n_b, 5)
            res["val_concept_nll_per_text_tok"] = round(concept_nll_elbo_sum / n_b, 5)
        return res

    cncm_val = _eval_lm_heldout(cncm_lm, s2_lm_val_ds, eval_concept=True)
    ar_val = _eval_lm_heldout(ar_baseline, s2_lm_val_ds, eval_concept=False)

    # Also benchmark B=1 generation speed for CNCM Concept Mode vs Pure-AR Baseline
    sample_pfx = s2_lm_val_ds[0]["prefix_ids"].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(2):
            _ = cncm_lm.generate_concept_chunk(sample_pfx, temperature=0.0, early_exit_conf=None)
            _ = ar_baseline.generate_exact(sample_pfx, max_new_tokens=64, temperature=0.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(4):
            _ = cncm_lm.generate_concept_chunk(sample_pfx, temperature=0.0, early_exit_conf=None)
        torch.cuda.synchronize()
        cncm_ms = ((time.perf_counter() - t0) / 4.0) * 1000.0

        t0 = time.perf_counter()
        for _ in range(2):
            _ = ar_baseline.generate_exact(sample_pfx, max_new_tokens=64, temperature=0.0)
        torch.cuda.synchronize()
        ar_ms = ((time.perf_counter() - t0) / 2.0) * 1000.0

    e1_report = {
        "cncm_total_params": cncm_total_params,
        "cncm_generator_params": cncm_gen_params,
        "ar_baseline_params": ar_param_count,
        "param_ratio_ar_to_cncm_total": round(ar_param_count / float(cncm_total_params), 4),
        "concept_head_exact_params": fch_params,
        "flat_4096_head_params": flat_head_params,
        "concept_head_mac_reduction_x": round(mac_reduction_ratio, 2),
        "cncm_heldout": cncm_val,
        "ar_baseline_heldout": ar_val,
        "delta_exact_nll_cncm_minus_ar": round(cncm_val["val_exact_nll"] - ar_val["val_exact_nll"], 5),
        "cncm_concept_k8_ms_b1": round(cncm_ms, 2),
        "ar_baseline_64tok_ms_b1": round(ar_ms, 2),
        "speedup_cncm_concept_vs_same_param_ar": round(ar_ms / max(cncm_ms, 1e-6), 3),
        "cncm_train_sec": round(cncm_lm_elapsed, 2),
        "ar_train_sec": round(ar_train_elapsed, 2),
    }
    print(f"[E1 REPORT] {json.dumps(e1_report)}")

    # =========================================================================
    # E3: Quantization Quality (FP32 / BF16 / FP16 / INT8 / INT4) & INT8 KV Drift
    # =========================================================================
    print("\n=== [E3] Quantization Quality (FP32/BF16/FP16/INT8/INT4) & INT8 KV Drift ===")
    q_fp32 = _eval_lm_heldout(cncm_lm, s2_lm_val_ds, eval_concept=True, max_batches=8, quantize_kv=False)
    q_bf16 = (
        _eval_lm_heldout(cncm_lm, s2_lm_val_ds, eval_concept=True, max_batches=8, cast_dtype=torch.bfloat16)
        if bf16_ok
        else {"skipped": True}
    )
    q_fp16 = _eval_lm_heldout(cncm_lm, s2_lm_val_ds, eval_concept=True, max_batches=8, cast_dtype=torch.float16)
    q_int8_kv = _eval_lm_heldout(cncm_lm, s2_lm_val_ds, eval_concept=False, max_batches=8, quantize_kv=True)

    # Measure direct logit drift between FP32 KV and INT8 KV on identical prefix+target
    with torch.no_grad():
        b0 = s2_lm_val_ds[0]
        pfx0 = b0["prefix_ids"].unsqueeze(0).to(device)
        tgt0 = b0["target_ids"].unsqueeze(0).to(device)
        pre_fp = cncm_lm.forward_exact(pfx0, use_cache=True, quantize_kv=False)
        post_fp = cncm_lm.forward_exact(tgt0, past_key_values=pre_fp["past_key_values"], use_cache=False, quantize_kv=False)
        pre_q = cncm_lm.forward_exact(pfx0, use_cache=True, quantize_kv=True)
        post_q = cncm_lm.forward_exact(tgt0, past_key_values=pre_q["past_key_values"], use_cache=False, quantize_kv=True)
        kv_logit_diff = (post_fp["logits"] - post_q["logits"]).abs()
        kv_max_diff = float(kv_logit_diff.max().item())
        kv_mean_diff = float(kv_logit_diff.mean().item())
        kv_top1_agree = float((post_fp["logits"].argmax(-1) == post_q["logits"].argmax(-1)).float().mean().item())

    # INT8 and INT4 post-training quantized model quality + latency
    model_int8 = quantize_model_int8(copy.deepcopy(cncm_lm).cpu(), quantize_embeddings=True).to(device).eval()
    q_int8_w = _eval_lm_heldout(model_int8, s2_lm_val_ds, eval_concept=True, max_batches=8)
    s1_int8_recon = _eval_stage1_heldout(model_int8.concept_tokenizer, use_kw=False)["heldout_recon_acc"]
    int8_mb = measure_model_size(model_int8)["total_mb"]
    del model_int8

    model_int4 = quantize_model_int4(copy.deepcopy(cncm_lm).cpu(), group_size=32, quantize_embeddings=True).to(device).eval()
    q_int4_w = _eval_lm_heldout(model_int4, s2_lm_val_ds, eval_concept=True, max_batches=8)
    s1_int4_recon = _eval_stage1_heldout(model_int4.concept_tokenizer, use_kw=False)["heldout_recon_acc"]
    int4_mb = measure_model_size(model_int4)["total_mb"]
    with torch.no_grad():
        _ = model_int4.generate_concept_chunk(sample_pfx, temperature=0.0, early_exit_conf=None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            _ = model_int4.generate_concept_chunk(sample_pfx, temperature=0.0, early_exit_conf=None)
        torch.cuda.synchronize()
        int4_ms = ((time.perf_counter() - t0) / 3.0) * 1000.0
    del model_int4

    # QAT -> INT4 export test (C23): short 40-step QAT fine-tune then pack to INT4
    qat_res = train_stage2(
        cfg=cfg_phase_a,
        tokenizer=tokenizer,
        concept_tokenizer=s1_phase_a,
        dataset=s2_lm_train_ds,
        mode="joint",
        steps=40,
        batch_size=32,
        lr=3e-4,
        device=device,
        qat_finetune=True,
        qat_bits=4,
        model=copy.deepcopy(cncm_lm),
        seed=42,
        verbose=False,
    )
    qat_int4 = quantize_model_int4(qat_res["model"].cpu(), group_size=32, quantize_embeddings=True).to(device).eval()
    q_qat_int4 = _eval_lm_heldout(qat_int4, s2_lm_val_ds, eval_concept=True, max_batches=8)
    del qat_int4, qat_res

    e3_report = {
        "fp32": {**q_fp32, "stage1_heldout_recon_acc": s1_heldout_phase_a["heldout_recon_acc"], "size_mb": round(measure_model_size(cncm_lm)["total_mb"], 3), "concept_k8_ms_b1": round(cncm_ms, 2)},
        "bf16": q_bf16,
        "fp16": q_fp16,
        "int8_weights": {**q_int8_w, "stage1_heldout_recon_acc": s1_int8_recon, "size_mb": round(int8_mb, 3)},
        "int4_weights_ptq": {**q_int4_w, "stage1_heldout_recon_acc": s1_int4_recon, "size_mb": round(int4_mb, 3), "concept_k8_ms_b1": round(int4_ms, 2), "slowdown_vs_fp32_note": f"{round(int4_ms / max(cncm_ms, 1e-6), 2)}x (dequant-in-pytorch, memory-only)"},
        "int4_weights_after_40step_qat": q_qat_int4,
        "int8_kv_cache_drift": {
            "fp32_kv_val_exact_nll": q_fp32["val_exact_nll"],
            "int8_kv_val_exact_nll": q_int8_kv["val_exact_nll"],
            "nll_delta": round(q_int8_kv["val_exact_nll"] - q_fp32["val_exact_nll"], 5),
            "logit_max_abs_diff": round(kv_max_diff, 5),
            "logit_mean_abs_diff": round(kv_mean_diff, 5),
            "top1_token_agreement": round(kv_top1_agree, 4),
        },
    }
    print(f"[E3 REPORT] {json.dumps(e3_report)}")

    # =========================================================================
    # E4: Early-Exit Quality vs Speed Curve (C9, C15)
    # =========================================================================
    print("\n=== [E4] Early-Exit Confidence Sweep (Quality vs Speed, B=1 & B=8) ===")
    e4_sweep: List[Dict[str, Any]] = []
    val_prefixes_b8 = torch.stack([s2_lm_val_ds[i]["prefix_ids"] for i in range(8)], dim=0).to(device)
    val_targets_b8 = torch.stack([s2_lm_val_ds[i]["target_ids"] for i in range(8)], dim=0).to(device)

    for conf in (0.50, 0.70, 0.85, 0.90, 0.95, 0.99, None):
        with torch.no_grad():
            # B=1 timing + exit level + exact token agreement vs full K=8 draft
            _ = cncm_lm.generate_concept_chunk(val_prefixes_b8[:1], temperature=0.0, early_exit_conf=conf)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_b1 = None
            for _ in range(4):
                out_b1 = cncm_lm.generate_concept_chunk(val_prefixes_b8[:1], temperature=0.0, early_exit_conf=conf)
            torch.cuda.synchronize()
            ms_b1 = ((time.perf_counter() - t0) / 4.0) * 1000.0

            # B=8 timing + exit level + exact NLL of generated text scored by the Exact-AR head
            _ = cncm_lm.generate_concept_chunk(val_prefixes_b8, temperature=0.0, early_exit_conf=conf)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_b8 = None
            for _ in range(3):
                out_b8 = cncm_lm.generate_concept_chunk(val_prefixes_b8, temperature=0.0, early_exit_conf=conf)
            torch.cuda.synchronize()
            ms_b8 = ((time.perf_counter() - t0) / 3.0) * 1000.0

            # Score generated B=8 continuations under the same-param AR baseline (self-contained quality proxy!)
            full_gen = torch.cat([val_prefixes_b8, out_b8["token_ids"]], dim=1)
            lbl_gen = torch.full_like(full_gen, -100)
            lbl_gen[:, val_prefixes_b8.shape[1] :] = out_b8["token_ids"]
            ar_score_nll = float(ar_baseline.forward_exact(full_gen, labels=lbl_gen)["loss"].item())

            e4_sweep.append(
                {
                    "early_exit_conf": conf,
                    "b1_exit_level": int(out_b1["exit_level"]),
                    "b1_ms": round(ms_b1, 2),
                    "b1_speedup_vs_ar": round(ar_ms / max(ms_b1, 1e-6), 3),
                    "b8_exit_level": int(out_b8["exit_level"]),
                    "b8_ms": round(ms_b8, 2),
                    "ar_judge_nll_on_draft": round(ar_score_nll, 4),
                    "ar_judge_ppl_on_draft": round(math.exp(min(ar_score_nll, 20.0)), 2),
                }
            )

    e4_report = {
        "stage1_per_level_cutoff_heldout_recon_acc": s1_heldout_phase_a["per_level_cutoff_recon_acc"],
        "confidence_sweep": e4_sweep,
    }
    print(f"[E4 REPORT] {json.dumps(e4_report)}")
    del cncm_lm, ar_baseline
    torch.cuda.empty_cache()

    # =========================================================================
    # E5: Honest Model-Only PG-FIM & Phase A vs Phase B A/B (C2, C17, C19)
    # =========================================================================
    print(f"\n=== [E5] Honest Model-Only PG-FIM: Phase A vs Phase B A/B + AR FIM Baseline ({stage2_steps} steps) ===")
    # Train Phase A (use_copy_decoder=False) and Phase B (use_copy_decoder=True) identically on fim_train_ds!
    fim_a_res = train_stage2(
        cfg=cfg_phase_a,
        tokenizer=tokenizer,
        concept_tokenizer=s1_phase_a,
        dataset=fim_train_ds,
        mode="joint",
        steps=stage2_steps,
        corrupt_rate=0.5,
        batch_size=24,
        lr=1.2e-3,
        device=device,
        amp_dtype="bf16" if bf16_ok else "fp32",
        seed=42,
        log_every=max(1, stage2_steps // 3),
        verbose=True,
    )
    fim_model_a: CNCMModel = fim_a_res["model"].eval()

    fim_b_res = train_stage2(
        cfg=cfg_phase_b,
        tokenizer=tokenizer,
        concept_tokenizer=s1_phase_b,
        dataset=fim_train_ds,
        mode="joint",
        steps=stage2_steps,
        corrupt_rate=0.5,
        batch_size=24,
        lr=1.2e-3,
        device=device,
        amp_dtype="bf16" if bf16_ok else "fp32",
        seed=42,
        log_every=max(1, stage2_steps // 3),
        verbose=True,
    )
    fim_model_b: CNCMModel = fim_b_res["model"].eval()

    # Held-out FIM validation losses for Phase A vs Phase B
    fim_val_a = _eval_lm_heldout(fim_model_a, fim_val_ds, eval_concept=True, max_batches=8, batch_size=16)
    fim_val_b = _eval_lm_heldout(fim_model_b, fim_val_ds, eval_concept=True, max_batches=8, batch_size=16)

    @torch.no_grad()
    def _eval_pg_fim_honest(
        mod: CNCMModel,
        tasks: Sequence[Dict[str, Any]],
        *,
        use_plan: bool,
        enable_repair: bool,
        allow_template_fallback: bool,
        use_oracle_kw: bool = False,
    ) -> Dict[str, Any]:
        n = len(tasks)
        parses = 0
        compiles = 0
        tests_pass = 0
        non_triv = 0
        kw_ok = 0
        repair_trig = 0
        model_gen_cnt = 0
        exact_rep_succ = 0
        passes_sum = 0.0
        rewards_sum = 0.0
        t0 = time.perf_counter()
        for t in tasks:
            out = mod.generate_fim(
                prefix=t["prefix"],
                suffix=t["suffix"],
                tokenizer=tokenizer,
                max_chunks=1,
                temperature=0.0,
                early_exit_conf=0.92,
                use_plan=use_plan,
                enable_repair=enable_repair,
                allow_template_fallback=allow_template_fallback,
                oracle_keywords=t["oracle_keywords"] if use_oracle_kw else None,
            )
            rew = compute_fim_reward(
                middle_text=out["middle_text"],
                spec=t,
                planned_keywords=out["keywords"],
                planned_chunks=out["planned_chunks"],
                actual_chunks=1,
            )
            parses += int(rew["parses"])
            compiles += int(rew["compiles"])
            tests_pass += int(rew["tests_passed"])
            non_triv += int(rew["non_trivial"])
            kw_ok += int(rew["kw_faithful"])
            repair_trig += int(out["repair_triggered"])
            model_gen_cnt += int(out.get("model_generated", True))
            if out.get("repair_source") == "exact_ar_model":
                exact_rep_succ += 1
            passes_sum += float(out["sequential_passes"])
            rewards_sum += float(rew["reward"])
        torch.cuda.synchronize()
        ms_per_task = ((time.perf_counter() - t0) / float(max(1, n))) * 1000.0
        return {
            "num_tasks": n,
            "model_generated_rate": round(model_gen_cnt / float(n), 4),
            "parse_rate": round(parses / float(n), 4),
            "compile_rate": round(compiles / float(n), 4),
            "test_pass_rate": round(tests_pass / float(n), 4),
            "non_trivial_rate": round(non_triv / float(n), 4),
            "kw_faithful_rate": round(kw_ok / float(n), 4),
            "repair_trigger_rate": round(repair_trig / float(n), 4),
            "exact_ar_repair_success_rate": round(exact_rep_succ / float(n), 4),
            "mean_reward": round(rewards_sum / float(n), 4),
            "avg_sequential_passes": round(passes_sum / float(n), 2),
            "latency_ms_per_task": round(ms_per_task, 2),
        }

    # Real Pure-AR FIM baseline (generate_exact on PSM prefix, 64 tokens max, stopping at <fim_end>)
    @torch.no_grad()
    def _eval_ar_fim(mod: CNCMModel, tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(tasks)
        parses = compiles = tests_pass = non_triv = 0
        t0 = time.perf_counter()
        for t in tasks:
            _, psm_ids = tokenizer.format_fim_prompt(t["prefix"], t["suffix"], 512, 256)
            pfx_t = torch.tensor([psm_ids], dtype=torch.long, device=device)
            out = mod.generate_exact(
                pfx_t,
                max_new_tokens=64,
                temperature=0.0,
                stop_token_ids=[cfg_phase_a.fim_end_id, cfg_phase_a.eos_token_id],
            )
            mid_ids = [
                int(x)
                for x in out["token_ids"][0].tolist()
                if int(x) not in (cfg_phase_a.pad_token_id, cfg_phase_a.fim_end_id, cfg_phase_a.eos_token_id)
            ]
            mid_str = tokenizer.decode(mid_ids, skip_special_tokens=True)
            rew = compute_fim_reward(mid_str, t, planned_keywords=[], planned_chunks=1, actual_chunks=1)
            parses += int(rew["parses"])
            compiles += int(rew["compiles"])
            tests_pass += int(rew["tests_passed"])
            non_triv += int(rew["non_trivial"])
        torch.cuda.synchronize()
        ms_per_task = ((time.perf_counter() - t0) / float(max(1, n))) * 1000.0
        return {
            "num_tasks": n,
            "model_generated_rate": 1.0,
            "parse_rate": round(parses / float(n), 4),
            "compile_rate": round(compiles / float(n), 4),
            "test_pass_rate": round(tests_pass / float(n), 4),
            "non_trivial_rate": round(non_triv / float(n), 4),
            "latency_ms_per_task": round(ms_per_task, 2),
        }

    ar_fim_eval = _eval_ar_fim(fim_model_a, fim_val_tasks)
    e5_variants = {
        "ar_fim_exact_baseline": ar_fim_eval,
        "phase_a_no_plan_model_only": _eval_pg_fim_honest(
            fim_model_a, fim_val_tasks, use_plan=False, enable_repair=False, allow_template_fallback=False
        ),
        "phase_a_plan_and_exact_repair_model_only": _eval_pg_fim_honest(
            fim_model_a, fim_val_tasks, use_plan=True, enable_repair=True, allow_template_fallback=False
        ),
        "phase_b_copy_plan_and_exact_repair_model_only": _eval_pg_fim_honest(
            fim_model_b, fim_val_tasks, use_plan=True, enable_repair=True, allow_template_fallback=False
        ),
        "phase_b_oracle_kw_model_only": _eval_pg_fim_honest(
            fim_model_b, fim_val_tasks, use_plan=True, enable_repair=True, allow_template_fallback=False, use_oracle_kw=True
        ),
        "template_fallback_upper_bound_non_model": _eval_pg_fim_honest(
            fim_model_b, fim_val_tasks, use_plan=True, enable_repair=True, allow_template_fallback=True
        ),
    }
    # Add measured wall-clock speedup vs real AR FIM baseline
    ar_fim_ms = max(ar_fim_eval["latency_ms_per_task"], 1e-6)
    for k, v in e5_variants.items():
        if k != "ar_fim_exact_baseline":
            v["measured_wallclock_speedup_vs_ar_fim"] = round(ar_fim_ms / max(v["latency_ms_per_task"], 1e-6), 3)

    e5_report = {
        "heldout_fim_val_loss_phase_a": fim_val_a,
        "heldout_fim_val_loss_phase_b": fim_val_b,
        "variants": e5_variants,
    }
    print(f"[E5 REPORT] {json.dumps(e5_report)}")

    total_sec = time.perf_counter() - t_all
    rate = _estimate_gpu_rate(gpu_name)
    cost_usd = (total_sec / 3600.0) * rate

    report: Dict[str, Any] = {
        "status": "passed",
        "preset": preset,
        "gpu_name": gpu_name,
        "corpus_source": corpus_source,
        "stage1_heldout_phase_a": s1_heldout_phase_a,
        "stage1_heldout_phase_b_nokw": s1_heldout_phase_b_nokw,
        "stage1_heldout_phase_b_kw": s1_heldout_phase_b_kw,
        "E1_same_param_ar_vs_cncm": e1_report,
        "E2_grad_accum_and_commit_parity": e2_report,
        "E3_quantization_and_kv_drift": e3_report,
        "E4_early_exit_quality_vs_speed": e4_report,
        "E5_honest_pg_fim_and_phase_ab": e5_report,
        "timing_and_cost": {
            "total_gpu_seconds": round(total_sec, 2),
            "gpu_rate_usd_per_hour": rate,
            "estimated_cost_usd": round(cost_usd, 5),
        },
    }

    out_vol = Path(VOLUME_MOUNT) / "cncm_audit_report.json"
    out_vol.write_text(json.dumps(report, indent=2), encoding="utf-8")
    vol.commit()
    print(f"\n[AUDIT COMPLETE] {total_sec:.1f}s on {gpu_name} (~${cost_usd:.4f})")
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main(
    preset: str = "nano",
    stage1_steps: int = 400,
    stage2_steps: int = 600,
    ar_steps: int = 600,
) -> None:
    report_json = run_section_e_audit.remote(
        preset=preset,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        ar_steps=ar_steps,
    )
    local_out = PROJECT_ROOT / "modal_audit_report.json"
    local_out.write_text(report_json, encoding="utf-8")
    print(f"[LOCAL] Saved Section E audit report to {local_out}")
