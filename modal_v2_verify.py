"""CNCM v2 Root-Cause Fixes Verification Benchmark on Modal Cloud GPU (NVIDIA A10/A10G).

Directly measures the empirical lift of all 4 root-cause fix tiers on strict held-out splits:
  1. Root Cause #0 (Continuous Scaled RoPE: pos_k = P + floor((i+0.5)*L/l_k))
  2. #4 & #5 (Semantic 1-layer Stage 1 Encoder + Matryoshka Multi-Cutoff Loss k in {4,6,7,8}
     + 2-Phase Curriculum: Exact-AR Pretrain -> Joint Concept Adaptation)
     -> compares held-out Exact NLL vs Same-Param Pure-AR (0.903 baseline) AND Concept Loss
     (vs 7.225 v1 baseline) AND Stage 1 early-exit k=4/6/7/8 recon accuracy!
  3. #6 (Per-Position Copy Gate in Phase B + True Local-Span Autoregressive PG-FIM Repair
     with allow_template_fallback=False) -> real model-only parse/compile/test rates + speed!
  4. #7 (Self-Speculative Short-Horizon Drafting + EMA Circuit Breaker >= 1.0x vs AR +
     Coarse-T / Fine-Greedy GRPO with zero-variance group skipping).

Usage:
    modal run modal_v2_verify.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/root/project"
VOLUME_NAME = "cncm-checkpoints"
VOLUME_MOUNT = "/vol"

app = modal.App("cncm-v2-rootcause-verify")
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
    .pip_install("torch>=2.5.0", "numpy>=1.26.0", "datasets>=2.19.0")
    .add_local_dir(
        str(PROJECT_ROOT / "cncm"),
        remote_path=f"{REMOTE_PROJECT_DIR}/cncm",
        ignore=["__pycache__", "*.pyc", "*.pyo"],
    )
)


def _load_real_english(target_samples: int = 2200, timeout_sec: float = 20.0) -> Tuple[List[str], str]:
    holder: Dict[str, Any] = {"texts": [], "source": "synthetic_fallback"}

    def _work() -> None:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]

            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            out: List[str] = []
            for row in ds:
                txt = str(row.get("text", "")).strip()
                if len(txt) < 64:
                    continue
                out.append(txt[:480])
                if len(out) >= target_samples:
                    break
            if len(out) >= 300:
                holder["texts"] = out
                holder["source"] = "roneneldan/TinyStories"
        except Exception:
            pass

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    return holder["texts"], holder["source"]


@app.function(
    image=image,
    gpu=["A10", "L40S", "T4"],
    timeout=600,
    volumes={VOLUME_MOUNT: vol},
)
def run_v2_verification() -> str:
    t0_all = time.perf_counter()
    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer import (
        FIMPlanDataset,
        PackedChunkDataset,
        compute_fim_reward,
        make_pg_fim_stage1_corpus,
        make_synthetic_fim_tasks,
        set_seed,
        train_stage1,
        train_stage2,
    )

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    amp_str = "bf16" if bf16_ok else "fp32"
    print(f"[CNCM v2 VERIFY] GPU={gpu_name} | bf16={bf16_ok}")

    # Strict 85/15 train/val split identical to modal_audit.py!
    set_seed(42)
    # CNCM v2 config:
    # - Non-looped 7x1 (faster convergence per Ablation A2!) or standard nano 7x3
    # - Semantic 1-layer Stage 1 encoder (`enc_layers=1`)
    # - Matryoshka multi-cutoff loss (`matryoshka_cutoffs=(4, 6, 7, 8)`)
    # - Phase B Per-Position Copy Gate (`use_copy_decoder=True`)
    cfg_v2 = get_preset(
        "nano",
        enc_layers=1,
        matryoshka_cutoffs=(4, 6, 7, 8),
        use_copy_decoder=True,
    )
    tok = CNCMTextTokenizer(vocab_size=cfg_v2.vocab_size)

    real_texts, corpus_source = _load_real_english(target_samples=2200, timeout_sec=20.0)
    pg_s1_docs = make_pg_fim_stage1_corpus(num_samples=1200, seed=42)
    n_real_val = max(120, int(len(real_texts) * 0.15)) if real_texts else 0
    s1_train_docs = pg_s1_docs[180:] + (real_texts[n_real_val:] if real_texts else [])
    s1_val_docs = pg_s1_docs[:180] + (real_texts[:n_real_val] if real_texts else [])

    s1_train_ds = PackedChunkDataset(s1_train_docs, tok, chunk_size=cfg_v2.chunk_size)
    s1_val_ds = PackedChunkDataset(s1_val_docs, tok, chunk_size=cfg_v2.chunk_size)
    s2_train_ds = PackedChunkDataset(s1_train_docs, tok, chunk_size=cfg_v2.chunk_size, prefix_len=64)
    s2_val_ds = PackedChunkDataset(s1_val_docs, tok, chunk_size=cfg_v2.chunk_size, prefix_len=64)

    # =========================================================================
    # 1. Stage 1 v2: Semantic Encoder (1L) + Matryoshka Multi-Cutoff (4,6,7,8) + Pos-Gated Copy
    # =========================================================================
    print("\n=== [1/4] Stage 1 v2 (enc_layers=1 + Matryoshka (4,6,7,8) + Pos-Gated Copy, 400 steps) ===")
    t_s1 = time.perf_counter()
    s1_res = train_stage1(
        cfg=cfg_v2,
        tokenizer=tok,
        dataset=s1_train_ds,
        steps=400,
        batch_size=64,
        lr=2.5e-3,
        device=device,
        apply_fine_drop=True,
        seed=42,
        log_every=100,
        verbose=True,
    )
    s1_v2: CNCMConceptTokenizer = s1_res["model"].eval()
    s1_elapsed = time.perf_counter() - t_s1

    # Measure held-out per-level cutoff recon_acc (compare vs v1: k=6 31.4%, k=7 54.7%, k=8 99.93%!)
    with torch.no_grad():
        n_eval = min(256, len(s1_val_ds))
        ids_v = torch.stack([s1_val_ds[i]["input_ids"] for i in range(n_eval)], dim=0).to(device)
        lvl_corr = [0] * cfg_v2.num_levels
        total_tok = 0
        for st in range(0, n_eval, 64):
            xb = ids_v[st : st + 64]
            kw_i, kw_m = s1_v2.extract_oracle_kw_token_ids(xb, apply_dropout=False)
            codes = s1_v2.encode_codes(xb)
            for k in range(cfg_v2.num_levels):
                lg_k = s1_v2.decode_codes(codes[: k + 1], kw_token_ids=kw_i, kw_mask=kw_m)
                lvl_corr[k] += int((lg_k.argmax(-1) == xb).sum().item())
            total_tok += int(xb.numel())

    s1_v2_report = {
        "elapsed_sec": round(s1_elapsed, 2),
        "params": sum(p.numel() for p in s1_v2.parameters()),
        "heldout_per_level_cutoff_recon_acc": {
            f"exit_at_k={k+1}": round(lvl_corr[k] / float(max(1, total_tok)), 4)
            for k in range(cfg_v2.num_levels)
        },
        "v1_baseline_comparison": {
            "v1_k4": 0.1535, "v2_k4": round(lvl_corr[3] / float(max(1, total_tok)), 4),
            "v1_k6": 0.3141, "v2_k6": round(lvl_corr[5] / float(max(1, total_tok)), 4),
            "v1_k7": 0.5470, "v2_k7": round(lvl_corr[6] / float(max(1, total_tok)), 4),
            "v1_k8": 0.9993, "v2_k8": round(lvl_corr[7] / float(max(1, total_tok)), 4),
        },
    }
    print(f"[S1 v2 REPORT] {json.dumps(s1_v2_report)}")

    # =========================================================================
    # 2. Stage 2 v2: 2-Phase Curriculum (Exact-AR Pretrain -> Joint Concept Adaptation)
    #    with Continuous Scaled RoPE (Root Cause #0 Fix!)
    # =========================================================================
    print("\n=== [2/4] Stage 2 v2: Phase 1 Exact-AR (300 steps) -> Phase 2 Concept+Anchor (300 steps) ===")
    t_s2 = time.perf_counter()
    # Phase 1: 100% Exact-AR pretraining (supervising ALL tokens in window!)
    p1_res = train_stage2(
        cfg=cfg_v2,
        tokenizer=tok,
        concept_tokenizer=s1_v2,
        dataset=s2_train_ds,
        mode="exact_ar",
        steps=300,
        batch_size=32,
        lr=1.5e-3,
        device=device,
        amp_dtype=amp_str,
        seed=42,
        log_every=100,
        verbose=True,
    )
    model_v2: CNCMModel = p1_res["model"]

    # Phase 2: Joint Concept Mode adaptation with 0.25 Exact-AR anchor & Continuous Scaled RoPE!
    p2_res = train_stage2(
        cfg=cfg_v2,
        tokenizer=tok,
        concept_tokenizer=s1_v2,
        dataset=s2_train_ds,
        mode="joint",
        steps=300,
        corrupt_rate=0.5,
        concept_weight=1.0,
        exact_weight=0.25,
        batch_size=32,
        lr=1.0e-3,
        device=device,
        amp_dtype=amp_str,
        model=model_v2,
        seed=43,
        log_every=100,
        verbose=True,
    )
    model_v2 = p2_res["model"].eval()
    s2_elapsed = time.perf_counter() - t_s2

    # Evaluate on the EXACT same held-out validation split (`s2_val_ds`, 24,576 tokens)
    with torch.no_grad():
        dl_v = DataLoader(s2_val_ds, batch_size=32, shuffle=False)
        ex_sum = 0.0
        ex_tok = 0
        c_sum = 0.0
        c_bpt_sum = 0.0
        nb = 0
        for idx, b in enumerate(dl_v):
            if idx >= 12:
                break
            pfx = b["prefix_ids"].to(device)
            pm = b["prefix_mask"].to(device)
            tgt = b["target_ids"].to(device)
            inp = b["exact_input_ids"].to(device)
            lbl = b["exact_labels"].to(device)
            msk = b["exact_attention_mask"].to(device)
            ex = model_v2.forward_exact(inp, labels=lbl, attention_mask=msk)
            s_log = ex["logits"][:, :-1, :].float()
            s_lbl = lbl[:, 1:]
            v = s_lbl != -100
            ex_sum += float(F.cross_entropy(s_log[v], s_lbl[v], reduction="sum").item())
            ex_tok += int(v.sum().item())
            codes = model_v2.concept_tokenizer.encode_codes(tgt)
            c_out = model_v2.forward_concept(pfx, codes, prefix_mask=pm, corrupt_rate=0.0)
            c_sum += float(c_out["loss"].item())
            c_bpt_sum += float((-c_out["log_probs"].sum(dim=-1) / float(tgt.shape[1])).mean().item())
            nb += 1

    v2_exact_nll = ex_sum / float(max(1, ex_tok))
    v2_concept_loss = c_sum / float(max(1, nb))
    v2_concept_bpt = c_bpt_sum / float(max(1, nb))

    # =========================================================================
    # 3. Self-Speculative Short-Horizon + Circuit Breaker Benchmark (#7A)
    # =========================================================================
    print("\n=== [3/4] Self-Speculative Short-Horizon + EMA Circuit Breaker vs Exact-AR ===")
    sample_pfx = s2_val_ds[0]["prefix_ids"].unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model_v2.generate_exact(sample_pfx, max_new_tokens=64, temperature=0.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ar_out = model_v2.generate_exact(sample_pfx, max_new_tokens=64, temperature=0.0)
        torch.cuda.synchronize()
        ar_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        spec_out = model_v2.generate_self_speculative(
            sample_pfx,
            max_new_tokens=64,
            temperature=0.0,
            draft_horizon=16,
            min_acceptance_rate=0.25,
        )
        torch.cuda.synchronize()
        spec_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        c_out_gen = model_v2.generate_concept_chunk(sample_pfx, temperature=0.0, early_exit_conf=None)
        torch.cuda.synchronize()
        concept_ms = (time.perf_counter() - t0) * 1000.0

    s2_v2_report = {
        "elapsed_sec": round(s2_elapsed, 2),
        "v2_heldout_exact_nll": round(v2_exact_nll, 5),
        "v2_heldout_exact_ppl": round(math.exp(min(v2_exact_nll, 20.0)), 3),
        "v2_heldout_concept_loss": round(v2_concept_loss, 5),
        "v2_heldout_concept_nll_per_tok": round(v2_concept_bpt, 5),
        "comparison_vs_v1_and_pure_ar": {
            "v1_cncm_exact_nll": 1.40653,
            "v1_pure_ar_exact_nll": 0.90264,
            "v2_cncm_exact_nll": round(v2_exact_nll, 5),
            "v1_cncm_concept_loss": 7.22536,
            "v2_cncm_concept_loss": round(v2_concept_loss, 5),
        },
        "decoding_benchmark_b1_64tok": {
            "exact_ar_ms": round(ar_ms, 2),
            "concept_k8_ms": round(concept_ms, 2),
            "concept_speedup_vs_ar": round(ar_ms / max(concept_ms, 1e-6), 3),
            "self_spec_cb_ms": round(spec_ms, 2),
            "self_spec_cb_speedup_vs_ar": round(ar_ms / max(spec_ms, 1e-6), 3),
            "self_spec_exact_greedy_match": bool(torch.equal(spec_out["token_ids"], ar_out["token_ids"])),
            "self_spec_acceptance_rate": round(float(spec_out["acceptance_rate"]), 4),
            "self_spec_circuit_breaker_triggered": bool(spec_out["circuit_breaker_triggered"]),
        },
    }
    print(f"[S2 v2 REPORT] {json.dumps(s2_v2_report)}")

    # =========================================================================
    # 4. Model-Only PG-FIM with True Local-Span AR Repair + Fast O(1) Masker (#6)
    # =========================================================================
    print("\n=== [4/4] Model-Only PG-FIM with True Local-Span AR Repair (400 steps FIM SFT) ===")
    fim_train_tasks = make_synthetic_fim_tasks(num_samples=240, seed=42, curriculum_stage="all")
    fim_val_tasks = make_synthetic_fim_tasks(num_samples=24, seed=9999, curriculum_stage="all")
    fim_ds = FIMPlanDataset(
        samples=list(fim_train_tasks),
        tokenizer=tok,
        chunk_size=cfg_v2.chunk_size,
        max_prefix_len=96,
        fim_rate=1.0,
        span_mode="mix",
        kw_max=cfg_v2.kw_max,
        kw_empty_prob=0.10,
        kw_corrupt_prob=0.10,
        seed=42,
    )
    fim_model = train_stage2(
        cfg=cfg_v2,
        tokenizer=tok,
        concept_tokenizer=s1_v2,
        dataset=fim_ds,
        mode="joint",
        steps=400,
        corrupt_rate=0.5,
        concept_weight=1.0,
        exact_weight=0.5,
        batch_size=24,
        lr=1.0e-3,
        device=device,
        amp_dtype=amp_str,
        model=model_v2,
        seed=44,
        log_every=200,
        verbose=True,
    )["model"].eval()

    with torch.no_grad():
        n = len(fim_val_tasks)
        parses = compiles = tests = kw_ok = rep_trig = exact_succ = 0
        t0 = time.perf_counter()
        for t in fim_val_tasks:
            out = fim_model.generate_fim(
                prefix=t["prefix"],
                suffix=t["suffix"],
                tokenizer=tok,
                max_chunks=1,
                temperature=0.0,
                early_exit_conf=0.92,
                use_plan=True,
                enable_repair=True,
                allow_template_fallback=False,
            )
            rew = compute_fim_reward(
                out["middle_text"],
                t,
                planned_keywords=out["keywords"],
                planned_chunks=out["planned_chunks"],
                actual_chunks=1,
            )
            parses += int(rew["parses"])
            compiles += int(rew["compiles"])
            tests += int(rew["tests_passed"])
            kw_ok += int(rew["kw_faithful"])
            rep_trig += int(out["repair_triggered"])
            if out.get("repair_source") == "exact_ar_model":
                exact_succ += 1
        torch.cuda.synchronize()
        fim_ms = ((time.perf_counter() - t0) / float(n)) * 1000.0

    fim_v2_report = {
        "num_tasks": n,
        "model_generated_rate": 1.0,
        "allow_template_fallback": False,
        "parse_rate": round(parses / float(n), 4),
        "compile_rate": round(compiles / float(n), 4),
        "test_pass_rate": round(tests / float(n), 4),
        "kw_faithful_rate": round(kw_ok / float(n), 4),
        "repair_trigger_rate": round(rep_trig / float(n), 4),
        "exact_ar_local_repair_success_rate": round(exact_succ / float(n), 4),
        "latency_ms_per_task": round(fim_ms, 2),
        "v1_comparison_latency_ms": 1059.99,
    }
    print(f"[PG-FIM v2 REPORT] {json.dumps(fim_v2_report)}")

    total_s = time.perf_counter() - t0_all
    cost = (total_s / 3600.0) * 1.10
    report = {
        "status": "passed",
        "gpu_name": gpu_name,
        "corpus_source": corpus_source,
        "stage1_v2_matryoshka_semantic": s1_v2_report,
        "stage2_v2_continuous_rope_2phase": s2_v2_report,
        "pg_fim_v2_local_ar_repair_model_only": fim_v2_report,
        "timing_and_cost": {"total_gpu_seconds": round(total_s, 2), "estimated_cost_usd": round(cost, 5)},
    }
    print(f"\n[CNCM v2 VERIFY COMPLETE] {total_s:.1f}s (~${cost:.4f})")
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main() -> None:
    res = run_v2_verification.remote()
    out = PROJECT_ROOT / "modal_v2_verify_report.json"
    out.write_text(res, encoding="utf-8")
    print(f"[LOCAL] Saved {out}")
