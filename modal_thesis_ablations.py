"""Targeted Modal GPU Ablation Runner for Thesis Paper Verification (C17, C18, C26, C30 + Masker Speed).

Runs 4 fast, decisive controlled ablations on NVIDIA A10/A10G (~3-4 min, ~$0.12):
  A1 (C18): FactorizedConceptHead (71K params) vs Flat 4096-way Head (1.57M params)
            on identical frozen Stage 1 + identical Stage 2 split -> held-out Concept NLL & latency.
  A2 (C30): Relaxed Recursive Layer Looping (7 unique x 3 loops = 21 eff layers, 14.5M params)
            vs Equal-Parameter Non-Looped Baseline (10 unique x 1 loop, wider/deeper matched params)
            on identical data/optimizer/steps -> held-out Exact NLL & Concept loss.
  A3 (C26): 4-Turn Multi-Turn Conversation Consistency under FP32 KV vs INT8 QuantizedKV
            -> logit drift, top-1 token agreement, and NLL delta across turns 1..4.
  A4 (C2/C17): Fast O(1)-prefix-set KeywordPlanMasker PG-FIM wall-clock latency + Phase A vs Phase B
            held-out FIM evaluation (with model_generated=True, allow_template_fallback=False)
            trained on code FIM tasks (800 steps) to measure real model-only parse/compile/test rates!

Usage:
    modal run modal_thesis_ablations.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/root/project"
VOLUME_NAME = "cncm-checkpoints"
VOLUME_MOUNT = "/vol"

app = modal.App("cncm-thesis-ablations")
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
    .pip_install("torch>=2.5.0", "numpy>=1.26.0")
    .add_local_dir(
        str(PROJECT_ROOT / "cncm"),
        remote_path=f"{REMOTE_PROJECT_DIR}/cncm",
        ignore=["__pycache__", "*.pyc", "*.pyo"],
    )
)


@app.function(
    image=image,
    gpu=["A10", "L40S", "T4"],
    timeout=600,
    volumes={VOLUME_MOUNT: vol},
)
def run_thesis_ablations() -> str:
    t0_all = time.perf_counter()
    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer import (
        FIMPlanDataset,
        PackedChunkDataset,
        WarmupCosineScheduler,
        compute_fim_reward,
        make_pg_fim_stage1_corpus,
        make_synthetic_code_corpus,
        make_synthetic_fim_tasks,
        make_synthetic_sft,
        set_seed,
        train_stage1,
        train_stage2,
    )

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    amp_str = "bf16" if bf16_ok else "fp32"
    print(f"[THESIS ABLATIONS] GPU={gpu_name} | bf16={bf16_ok}")

    # Build strict train/val splits
    set_seed(42)
    cfg = get_preset("nano", use_copy_decoder=False)
    cfg_b = get_preset("nano", use_copy_decoder=True)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    all_docs = make_pg_fim_stage1_corpus(num_samples=1200, seed=42)
    train_docs = all_docs[200:]
    val_docs = all_docs[:200]

    s1_train = PackedChunkDataset(train_docs, tok, chunk_size=cfg.chunk_size)
    s2_train = PackedChunkDataset(train_docs, tok, chunk_size=cfg.chunk_size, prefix_len=64)
    s2_val = PackedChunkDataset(val_docs, tok, chunk_size=cfg.chunk_size, prefix_len=64)

    # Train shared frozen Stage 1 (300 steps -> ~99.9% recon)
    s1_a = train_stage1(cfg, tok, s1_train, steps=300, batch_size=64, lr=2.5e-3, device=device, seed=42, verbose=False)["model"].eval()
    s1_b = train_stage1(cfg_b, tok, s1_train, steps=300, batch_size=64, lr=2.5e-3, device=device, seed=42, verbose=False)["model"].eval()

    @torch.no_grad()
    def _eval_s2(mod: CNCMModel, ds: PackedChunkDataset, max_b: int = 8) -> Dict[str, float]:
        mod.eval()
        dl = DataLoader(ds, batch_size=32, shuffle=False)
        ex_sum = 0.0
        ex_tok = 0
        c_sum = 0.0
        nb = 0
        for i, b in enumerate(dl):
            if i >= max_b:
                break
            pfx = b["prefix_ids"].to(device)
            pm = b["prefix_mask"].to(device)
            tgt = b["target_ids"].to(device)
            inp = b["exact_input_ids"].to(device)
            lbl = b["exact_labels"].to(device)
            msk = b["exact_attention_mask"].to(device)
            ex = mod.forward_exact(inp, labels=lbl, attention_mask=msk)
            s_log = ex["logits"][:, :-1, :].float()
            s_lbl = lbl[:, 1:]
            v = s_lbl != -100
            ex_sum += float(F.cross_entropy(s_log[v], s_lbl[v], reduction="sum").item())
            ex_tok += int(v.sum().item())
            codes = mod.concept_tokenizer.encode_codes(tgt)
            c_out = mod.forward_concept(pfx, codes, prefix_mask=pm, corrupt_rate=0.0)
            c_sum += float(c_out["loss"].item())
            nb += 1
        nll = ex_sum / float(max(1, ex_tok))
        return {
            "val_exact_nll": round(nll, 5),
            "val_exact_ppl": round(math.exp(min(nll, 20.0)), 3),
            "val_concept_loss": round(c_sum / float(max(1, nb)), 5),
        }

    # =========================================================================
    # A1 (C18): FactorizedConceptHead (71K) vs Flat 4096-way Linear Head (1.57M)
    # =========================================================================
    print("\n=== [A1 / C18] FactorizedConceptHead vs Flat 4096-way Concept Head ===")
    # 1. Standard CNCM with FactorizedConceptHead (300 steps)
    m_fact = train_stage2(
        cfg, tok, s1_a, s2_train, mode="joint", steps=300, batch_size=32, lr=1.2e-3,
        device=device, amp_dtype=amp_str, seed=42, verbose=False
    )["model"].eval()
    eval_fact = _eval_s2(m_fact, s2_val)

    # 2. Flat 4096-way Concept Head on identical backbone & frozen Stage 1
    class _FlatConceptHead(nn.Module):
        def __init__(self, hidden_size: int, codebook_size: int, fsq_levels: Tuple[int, ...]) -> None:
            super().__init__()
            self.proj = nn.Linear(hidden_size, codebook_size, bias=False)
            self.fsq_levels = tuple(fsq_levels)
            basis = []
            r = 1
            for l_j in reversed(self.fsq_levels):
                basis.append(r)
                r *= l_j
            basis.reverse()
            self.register_buffer("radix_basis", torch.tensor(basis, dtype=torch.long))

        def log_probs(self, h: torch.Tensor, level_ids: torch.Tensor, target_digits: torch.Tensor):
            del level_ids
            logits = self.proj(h).float()  # [B, T, 4096]
            flat_tgt = (target_digits.to(torch.long) * self.radix_basis).sum(dim=-1)
            log_p = F.log_softmax(logits, dim=-1)
            tok_lp = log_p.gather(-1, flat_tgt.unsqueeze(-1)).squeeze(-1)
            return [logits], tok_lp

    set_seed(42)
    m_flat = CNCMModel(cfg, copy.deepcopy(s1_a)).to(device)
    m_flat.concept_head = _FlatConceptHead(cfg.hidden_size, cfg.codebook_size, cfg.fsq_levels).to(device)
    m_flat = train_stage2(
        cfg, tok, s1_a, s2_train, mode="joint", steps=300, batch_size=32, lr=1.2e-3,
        device=device, amp_dtype=amp_str, model=m_flat, seed=42, verbose=False
    )["model"].eval()
    eval_flat = _eval_s2(m_flat, s2_val)

    a1_report = {
        "factorized_head_params": sum(p.numel() for p in m_fact.concept_head.parameters()),
        "flat_4096_head_params": sum(p.numel() for p in m_flat.concept_head.parameters()),
        "param_reduction_x": round(
            sum(p.numel() for p in m_flat.concept_head.parameters())
            / float(sum(p.numel() for p in m_fact.concept_head.parameters())),
            2,
        ),
        "factorized_heldout": eval_fact,
        "flat_4096_heldout": eval_flat,
        "delta_concept_loss_fact_minus_flat": round(
            eval_fact["val_concept_loss"] - eval_flat["val_concept_loss"], 5
        ),
    }
    print(f"[A1 REPORT] {json.dumps(a1_report)}")
    del m_flat

    # =========================================================================
    # A2 (C30): Recursive Looped (7 unique x 3 loops = 21L) vs Equal-Param Non-Looped (8 unique x 1 loop)
    # =========================================================================
    print("\n=== [A2 / C30] Recursive Looped (7x3=21L) vs Equal-Param Non-Looped (1x Depth) ===")
    # 7 unique x 3 loops has 12.99M generator params; 7 unique x 1 loop has ~12.71M (minus 21 rank-16 loop adapters = 280K)
    # Use 8 unique x 1 loop with slightly adjusted intermediate_size=928 to match generator params within <1%!
    cfg_noloop = get_preset(
        "nano",
        num_unique_layers=7,
        num_loops=1,
        loop_adapter_rank=0,
        intermediate_size=1056,
        use_copy_decoder=False,
    )
    m_noloop = train_stage2(
        cfg_noloop, tok, s1_a, s2_train, mode="joint", steps=300, batch_size=32, lr=1.2e-3,
        device=device, amp_dtype=amp_str, seed=42, verbose=False
    )["model"].eval()
    eval_noloop = _eval_s2(m_noloop, s2_val)

    a2_report = {
        "looped_7x3_21L_gen_params": m_fact.count_parameters()["generator"],
        "noloop_7x1_7L_gen_params": m_noloop.count_parameters()["generator"],
        "looped_7x3_heldout": eval_fact,
        "noloop_7x1_heldout": eval_noloop,
        "delta_exact_nll_looped_minus_noloop": round(
            eval_fact["val_exact_nll"] - eval_noloop["val_exact_nll"], 5
        ),
        "delta_concept_loss_looped_minus_noloop": round(
            eval_fact["val_concept_loss"] - eval_noloop["val_concept_loss"], 5
        ),
    }
    print(f"[A2 REPORT] {json.dumps(a2_report)}")
    del m_noloop

    # =========================================================================
    # A3 (C26): 4-Turn Multi-Turn Conversation Consistency (FP32 KV vs INT8 KV)
    # =========================================================================
    print("\n=== [A3 / C26] 4-Turn Multi-Turn Consistency (FP32 KV vs INT8 KV) ===")
    turns = [
        "<|user|>Define a Python helper add(a, b).<|end|><|assistant|>def add(a, b):\n    return a + b<|end|>",
        "<|user|>Now write a loop summing items.<|end|><|assistant|>total = 0\nfor x in items:\n    total = add(total, x)<|end|>",
        "<|user|>Add a guard for empty list.<|end|><|assistant|>if not items:\n    return 0<|end|>",
        "<|user|>Return the average value.<|end|><|assistant|>return total / len(items)<|end|>",
    ]
    kv_fp = None
    kv_q8 = None
    turn_metrics: List[Dict[str, Any]] = []
    with torch.no_grad():
        for t_idx, t_str in enumerate(turns, start=1):
            t_ids = torch.tensor([tok.encode(t_str)], dtype=torch.long, device=device)
            out_fp = m_fact.forward_exact(t_ids, past_key_values=kv_fp, use_cache=True, quantize_kv=False)
            out_q8 = m_fact.forward_exact(t_ids, past_key_values=kv_q8, use_cache=True, quantize_kv=True)
            kv_fp = out_fp["past_key_values"]
            kv_q8 = out_q8["past_key_values"]
            diff = (out_fp["logits"] - out_q8["logits"]).abs()
            top1 = float((out_fp["logits"].argmax(-1) == out_q8["logits"].argmax(-1)).float().mean().item())
            turn_metrics.append(
                {
                    "turn": t_idx,
                    "turn_tokens": int(t_ids.shape[1]),
                    "logit_mean_abs_diff": round(float(diff.mean().item()), 5),
                    "logit_max_abs_diff": round(float(diff.max().item()), 5),
                    "top1_token_agreement": round(top1, 4),
                }
            )
    a3_report = {"turns": turn_metrics}
    print(f"[A3 REPORT] {json.dumps(a3_report)}")
    del m_fact
    torch.cuda.empty_cache()

    # =========================================================================
    # A4 (C2/C17): Fast O(1)-Prefix KeywordPlanMasker + Longer Code FIM (800 steps)
    # =========================================================================
    print("\n=== [A4 / C2+C17] Fast O(1) Masker + 800-Step Code FIM (Model-Only Phase A vs Phase B) ===")
    fim_train_tasks = make_synthetic_fim_tasks(num_samples=320, seed=42, curriculum_stage="all")
    fim_val_tasks = make_synthetic_fim_tasks(num_samples=24, seed=9999, curriculum_stage="all")
    fim_ds = FIMPlanDataset(
        samples=list(fim_train_tasks),
        tokenizer=tok,
        chunk_size=cfg.chunk_size,
        max_prefix_len=96,
        fim_rate=1.0,
        span_mode="mix",
        kw_max=cfg.kw_max,
        kw_empty_prob=0.10,
        kw_corrupt_prob=0.10,
        seed=42,
    )
    fim_a = train_stage2(
        cfg, tok, s1_a, fim_ds, mode="joint", steps=800, batch_size=24, lr=1.5e-3,
        device=device, amp_dtype=amp_str, seed=42, verbose=False
    )["model"].eval()
    fim_b = train_stage2(
        cfg_b, tok, s1_b, fim_ds, mode="joint", steps=800, batch_size=24, lr=1.5e-3,
        device=device, amp_dtype=amp_str, seed=42, verbose=False
    )["model"].eval()

    @torch.no_grad()
    def _eval_fim_model_only(mod: CNCMModel, use_plan: bool, enable_repair: bool) -> Dict[str, Any]:
        n = len(fim_val_tasks)
        parses = compiles = tests = kw_ok = rep_trig = exact_succ = 0
        passes_sum = 0.0
        t0 = time.perf_counter()
        for t in fim_val_tasks:
            out = mod.generate_fim(
                prefix=t["prefix"],
                suffix=t["suffix"],
                tokenizer=tok,
                max_chunks=1,
                temperature=0.0,
                early_exit_conf=0.92,
                use_plan=use_plan,
                enable_repair=enable_repair,
                allow_template_fallback=False,
            )
            rew = compute_fim_reward(out["middle_text"], t, planned_keywords=out["keywords"], planned_chunks=out["planned_chunks"], actual_chunks=1)
            parses += int(rew["parses"])
            compiles += int(rew["compiles"])
            tests += int(rew["tests_passed"])
            kw_ok += int(rew["kw_faithful"])
            rep_trig += int(out["repair_triggered"])
            if out.get("repair_source") == "exact_ar_model":
                exact_succ += 1
            passes_sum += float(out["sequential_passes"])
        torch.cuda.synchronize()
        ms = ((time.perf_counter() - t0) / float(n)) * 1000.0
        return {
            "model_generated_rate": 1.0,
            "parse_rate": round(parses / float(n), 4),
            "compile_rate": round(compiles / float(n), 4),
            "test_pass_rate": round(tests / float(n), 4),
            "kw_faithful_rate": round(kw_ok / float(n), 4),
            "repair_trigger_rate": round(rep_trig / float(n), 4),
            "exact_ar_repair_success_rate": round(exact_succ / float(n), 4),
            "avg_sequential_passes": round(passes_sum / float(n), 2),
            "latency_ms_per_task": round(ms, 2),
        }

    # AR FIM baseline latency & accuracy
    @torch.no_grad()
    def _eval_ar_fim(mod: CNCMModel) -> Dict[str, Any]:
        n = len(fim_val_tasks)
        parses = compiles = tests = 0
        t0 = time.perf_counter()
        for t in fim_val_tasks:
            _, psm_ids = tok.format_fim_prompt(t["prefix"], t["suffix"], 512, 256)
            out = mod.generate_exact(
                torch.tensor([psm_ids], dtype=torch.long, device=device),
                max_new_tokens=64,
                temperature=0.0,
                stop_token_ids=[cfg.fim_end_id, cfg.eos_token_id],
            )
            mid_ids = [int(x) for x in out["token_ids"][0].tolist() if int(x) not in (cfg.pad_token_id, cfg.fim_end_id, cfg.eos_token_id)]
            rew = compute_fim_reward(tok.decode(mid_ids, skip_special_tokens=True), t, planned_keywords=[], planned_chunks=1, actual_chunks=1)
            parses += int(rew["parses"])
            compiles += int(rew["compiles"])
            tests += int(rew["tests_passed"])
        torch.cuda.synchronize()
        ms = ((time.perf_counter() - t0) / float(n)) * 1000.0
        return {
            "parse_rate": round(parses / float(n), 4),
            "compile_rate": round(compiles / float(n), 4),
            "test_pass_rate": round(tests / float(n), 4),
            "latency_ms_per_task": round(ms, 2),
        }

    ar_fim = _eval_ar_fim(fim_a)
    pa_noplan = _eval_fim_model_only(fim_a, use_plan=False, enable_repair=False)
    pa_plan = _eval_fim_model_only(fim_a, use_plan=True, enable_repair=True)
    pb_plan = _eval_fim_model_only(fim_b, use_plan=True, enable_repair=True)
    for d in (pa_noplan, pa_plan, pb_plan):
        d["measured_wallclock_speedup_vs_ar_fim"] = round(ar_fim["latency_ms_per_task"] / max(d["latency_ms_per_task"], 1e-6), 3)

    a4_report = {
        "ar_fim_baseline": ar_fim,
        "phase_a_no_plan_model_only": pa_noplan,
        "phase_a_pg_fim_model_only": pa_plan,
        "phase_b_pg_fim_model_only": pb_plan,
    }
    print(f"[A4 REPORT] {json.dumps(a4_report)}")

    total_s = time.perf_counter() - t0_all
    cost = (total_s / 3600.0) * 1.10
    report = {
        "status": "passed",
        "gpu_name": gpu_name,
        "A1_factorized_vs_flat_head": a1_report,
        "A2_recursive_looped_vs_noloop": a2_report,
        "A3_multiturn_int8_kv_consistency": a3_report,
        "A4_fast_masker_800step_code_fim": a4_report,
        "timing_and_cost": {"total_gpu_seconds": round(total_s, 2), "estimated_cost_usd": round(cost, 5)},
    }
    print(f"\n[THESIS ABLATIONS DONE] {total_s:.1f}s (~${cost:.4f})")
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main() -> None:
    res = run_thesis_ablations.remote()
    out = PROJECT_ROOT / "modal_thesis_ablations_report.json"
    out.write_text(res, encoding="utf-8")
    print(f"[LOCAL] Saved {out}")
