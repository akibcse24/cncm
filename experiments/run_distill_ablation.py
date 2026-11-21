"""Controlled 4-Arm Hierarchical Student-Teacher Distillation Benchmark (`experiments/run_distill_ablation.py`).

Evaluates the 3-Level Hierarchical Teacher -> Student Distillation stack for CNCM across
multiple seeds with paired statistical tests and Wilson / t-distribution 95% confidence intervals:

  Arm 1 (`baseline_raw`):
      Stage 1 M-RFSQ + Stage 2 Joint training on raw baseline corpus.
  Arm 2 (`level1_seqkd`):
      Stage 1 M-RFSQ + Stage 2 Joint training on Verifier-Gated Teacher SeqKD corpus
      (AST/compile/unit-test verified Python + tool-calling + concise `<think>` CoT).
  Arm 3 (`level2_concept_dagger`):
      Arm 2 + Dynamic Residual Concept-DAgger (`build_concept_dagger_rollout`: dynamically
      re-quantizing `z_teacher - z_hat_student_{<=k-1}` on on-policy student rollouts).
  Arm 4 (`level3_full_cross_kd`):
      Arm 3 + Frozen Exact-AR Teacher -> Concept Mode Cross-Head KL Distillation
      (`train_draft_alignment` with frozen teacher snapshot & `freeze_backbone_for_exact=True`).
  Control (`pure_ar_seqkd`):
      Equal-parameter Pure-AR model trained on the exact same Verifier-Gated SeqKD corpus
      for identical steps, batch size, and token budget.

Outputs:
  `experiments/results/distill_hierarchical_<preset>.json`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments.eval_suites import (
    build_heldout_arithmetic_suite,
    build_heldout_tool_calling_suite,
    build_pg_fim_heldout_test_bank,
)
from experiments.harness import run_seeded_experiment, set_global_seed
from experiments.stats import expected_speculative_speedup, paired_t_test
from experiments.train_smart_cncm import load_verified_seqkd_texts


def build_verified_teacher_seqkd_corpus(n_samples: int = 320, seed: int = 42) -> List[str]:
    """Synthesize a 100% verifier-checked Teacher SeqKD corpus (Prose, Math CoT, Python PG-FIM, JSON Tools).

    Every code sample is verified via `cncm.code.sandbox.run_code_sandbox` and every tool-call
    sample is verified via `evaluate_tool_call_semantic_accuracy` (via `filter_teacher_records`).
    """
    from cncm.trainer.dataset import make_pg_fim_stage1_corpus, make_synthetic_sft
    from experiments.distill_seqkd import filter_teacher_records

    raw_records: List[Dict[str, Any]] = []

    # 1. Verified Tool-Calling + Concise <think> CoT teacher traces (distinct train seed 9001)
    tool_items = build_heldout_tool_calling_suite(n_items=max(16, n_samples // 3), seed=seed + 9001)
    for item in tool_items:
        t_name = item["expected_tool"]
        t_args = item["expected_args"]
        json_call = json.dumps({"name": t_name, "arguments": t_args}, separators=(",", ":"))
        raw_records.append(
            {
                **item,
                "domain": "tool_call",
                "completion": f"<think>Invoke {t_name}.</think><tool_call>{json_call}</tool_call>",
            }
        )

    # 2. Verified Arithmetic Teacher traces (distinct train seed 9002)
    arith_items = build_heldout_arithmetic_suite(n_items=max(16, n_samples // 3), seed=seed + 9002)
    for item in arith_items:
        expr = item["expression"]
        val = item["expected_value"]
        json_call = json.dumps({"name": "calculator", "arguments": {"expression": expr}}, separators=(",", ":"))
        raw_records.append(
            {
                **item,
                "domain": "math",
                "expected_tool": "calculator",
                "completion": f"<think>Compute {expr} = {val}.</think><tool_call>{json_call}</tool_call>",
            }
        )

    # 3. Verified Python / PG-FIM & structured SFT prose traces
    code_docs = make_pg_fim_stage1_corpus(num_samples=max(32, n_samples // 2), seed=seed + 500)
    for doc in code_docs:
        raw_records.append({"domain": "prose", "completion": doc})

    sft_docs = make_synthetic_sft(num_samples=max(32, n_samples // 3), seed=seed + 700)
    for s_item in sft_docs:
        p_txt = s_item.get("prompt", "")
        c_txt = s_item.get("response", s_item.get("completion", ""))
        raw_records.append({"domain": "prose", "completion": f"{p_txt}\n{c_txt}".strip()})

    filtered = filter_teacher_records(raw_records, max_think_chars=180)
    texts: List[str] = []
    for rec in filtered["accepted"]:
        p = str(rec.get("prompt", "")).strip()
        c = str(rec.get("completion", "")).strip()
        texts.append(f"{p}\n{c}".strip() if p else c)
    return texts


def evaluate_student_model(
    model: Any,
    test_ds: Any,
    device: str,
    gamma: int = 16,
    max_batches: int = 12,
) -> Dict[str, float]:
    """Measure Exact-AR NLL, Concept Loss, Align NLL, Concept Draft NLL, and Speculative Acceptance alpha."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    model.eval()
    dl = DataLoader(test_ds, batch_size=min(8, len(test_ds)), shuffle=False)
    exact_nlls: List[float] = []
    concept_losses: List[float] = []
    align_nlls: List[float] = []
    draft_nlls: List[float] = []

    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= max_batches:
                break
            pfx = b["prefix_ids"].to(device)
            pm = b["prefix_mask"].to(device)
            tgt = b["target_ids"].to(device)
            inp = b["exact_input_ids"].to(device)
            lbl = b["exact_labels"].to(device)
            msk = b["exact_attention_mask"].to(device)

            ex = model.forward_exact(inp, labels=lbl, attention_mask=msk)
            exact_nlls.append(float(ex["loss"].item()))

            codes = model.concept_tokenizer.encode_codes(tgt)
            c_out = model.forward_concept(pfx, codes, prefix_mask=pm, corrupt_rate=0.0)
            concept_losses.append(float(c_out["loss"].item()))

            # Align NLL (finest concept hidden states projected through exact_head)
            t_len = tgt.shape[1]
            h_fine = c_out["h_target"][:, -t_len:, :]
            a_logits = model.exact_head(h_fine).float()
            a_nll = F.cross_entropy(a_logits.reshape(-1, model.cfg.vocab_size), tgt.reshape(-1))
            align_nlls.append(float(a_nll.item()))

            # Concept Draft NLL scored by Exact-AR teacher
            c_draft = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)["token_ids"]
            full_draft = torch.cat([pfx, c_draft], dim=1)
            d_labels = torch.cat([torch.full_like(pfx, -100), c_draft], dim=1)
            d_out = model.forward_exact(full_draft, labels=d_labels, use_cache=False)
            draft_nlls.append(float(d_out["loss"].item()))

        # Self-speculative token acceptance alpha (at sub-chunk horizon gamma and full chunk L)
        acc_gamma, tot_gamma = 0, 0
        acc_full, tot_full = 0, 0
        L = model.cfg.chunk_size
        eff_g = min(gamma, L)
        for i in range(min(12, len(test_ds))):
            pfx = test_ds[i]["prefix_ids"].unsqueeze(0).to(device)
            ar_ids = model.generate_exact(pfx, max_new_tokens=L, temperature=0.0)["token_ids"]
            c_ids = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)["token_ids"]

            # Full-chunk consecutive prefix acceptance
            m_full = 0
            for t in range(L):
                if int(c_ids[0, t].item()) == int(ar_ids[0, t].item()):
                    m_full += 1
                else:
                    break
            acc_full += m_full
            tot_full += L
            acc_gamma += min(m_full, eff_g)
            tot_gamma += eff_g

    alpha_full = acc_full / float(max(1, tot_full))
    alpha_gamma = acc_gamma / float(max(1, tot_gamma))
    # With K=8 passes for L=64 tokens, draft cycle cost ratio per verify pass is ~1.0 (or 0.5 with early exit)
    theo_gamma = expected_speculative_speedup(
        alpha=alpha_gamma,
        gamma=eff_g,
        draft_cycle_cost_ratio=1.0,
        verify_cycle_cost_ratio=1.0,
    )
    mean_ex = sum(exact_nlls) / max(1, len(exact_nlls))
    mean_c = sum(concept_losses) / max(1, len(concept_losses))
    mean_al = sum(align_nlls) / max(1, len(align_nlls))
    mean_dr = sum(draft_nlls) / max(1, len(draft_nlls))

    return {
        "exact_nll": round(mean_ex, 5),
        "exact_ppl": round(math.exp(min(mean_ex, 15.0)), 3),
        "concept_loss": round(mean_c, 5),
        "align_nll": round(mean_al, 5),
        "concept_draft_nll": round(mean_dr, 5),
        "draft_minus_exact_gap": round(mean_dr - mean_ex, 5),
        "alpha_full_chunk": round(alpha_full, 5),
        "alpha_subchunk_gamma": round(alpha_gamma, 5),
        "expected_lossless_speedup_gamma": round(theo_gamma["expected_speedup_vs_ar"], 4),
    }


def run_hierarchical_distill_ablation(
    preset: str = "tiny_test",
    stage1_steps: int = 80,
    stage2_steps: int = 120,
    align_steps: int = 40,
    batch_size: int = 8,
    seeds: Sequence[int] = (42,),
    device: str = "cpu",
    teacher_name: str = "Bonsai-2-4B",
    seqkd_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the 4-arm Hierarchical Distillation (Teacher: Bonsai-2-4B) + Pure-AR SeqKD control (single-seed default `seed=42`)."""
    import torch
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.dataset import PackedChunkDataset, make_synthetic_corpus
    from cncm.trainer.draft_align import train_draft_alignment
    from cncm.trainer.train_stage1 import train_stage1
    from cncm.trainer.train_stage2 import train_stage2

    cfg = get_preset(preset, enc_layers=1)

    def _one_seed(seed: int) -> Dict[str, Any]:
        set_global_seed(seed)
        tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

        # Raw corpus vs Verifier-Gated Teacher SeqKD corpus (from real llama-server JSONL if provided!)
        raw_texts = make_synthetic_corpus(num_samples=256, seed=seed)
        loaded_jsonl_texts = load_verified_seqkd_texts(seqkd_jsonl) if seqkd_jsonl is not None else []
        if loaded_jsonl_texts:
            # Repeat/tile verified real llama-server records to match PackedChunkDataset training window count
            repeats = max(1, (256 // max(1, len(loaded_jsonl_texts))) + 1)
            seqkd_texts = (loaded_jsonl_texts * repeats)[: max(128, len(loaded_jsonl_texts))]
            seqkd_source = f"real_llama_server_jsonl:{seqkd_jsonl} ({len(loaded_jsonl_texts)} verified records)"
        else:
            seqkd_texts = build_verified_teacher_seqkd_corpus(n_samples=256, seed=seed)
            seqkd_source = "programmatic_verified_seqkd"
        test_texts = (
            loaded_jsonl_texts[-max(8, len(loaded_jsonl_texts) // 5) :]
            if len(loaded_jsonl_texts) >= 20
            else build_verified_teacher_seqkd_corpus(n_samples=64, seed=seed + 20_000)
        )

        raw_ds = PackedChunkDataset(raw_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
        seqkd_ds = PackedChunkDataset(seqkd_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
        test_ds = PackedChunkDataset(test_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)

        # Shared Stage 1 on raw vs SeqKD
        s1_raw = train_stage1(
            cfg, tok, dataset=raw_ds, steps=stage1_steps, batch_size=batch_size,
            device=device, seed=seed, verbose=False
        )["model"].eval()
        s1_kd = train_stage1(
            cfg, tok, dataset=seqkd_ds, steps=stage1_steps, batch_size=batch_size,
            device=device, seed=seed, verbose=False
        )["model"].eval()

        # Arm 1: Baseline Raw Corpus (Stage 2 Joint)
        set_global_seed(seed)
        m_arm1 = CNCMModel(cfg, copy.deepcopy(s1_raw))
        train_stage2(
            cfg, tok, concept_tokenizer=m_arm1.concept_tokenizer, model=m_arm1,
            dataset=raw_ds, mode="joint", steps=stage2_steps, batch_size=batch_size,
            concept_weight=1.0, exact_weight=0.7, align_weight=0.3,
            device=device, seed=seed, verbose=False,
        )
        arm1_eval = evaluate_student_model(m_arm1, test_ds, device)

        # Arm 2: + Level 1 Verifier-Gated SeqKD
        set_global_seed(seed)
        m_arm2 = CNCMModel(cfg, copy.deepcopy(s1_kd))
        train_stage2(
            cfg, tok, concept_tokenizer=m_arm2.concept_tokenizer, model=m_arm2,
            dataset=seqkd_ds, mode="joint", steps=stage2_steps, batch_size=batch_size,
            concept_weight=1.0, exact_weight=0.7, align_weight=0.3,
            device=device, seed=seed, verbose=False,
        )
        arm2_eval = evaluate_student_model(m_arm2, test_ds, device)

        # Control: Pure-AR trained on the exact same Verifier-Gated SeqKD corpus & budget
        set_global_seed(seed)
        m_ar_kd = CNCMModel(cfg, copy.deepcopy(s1_kd))
        train_stage2(
            cfg, tok, concept_tokenizer=m_ar_kd.concept_tokenizer, model=m_ar_kd,
            dataset=seqkd_ds, mode="exact_ar", steps=stage2_steps, batch_size=batch_size,
            concept_weight=0.0, exact_weight=1.0, align_weight=0.0,
            device=device, seed=seed, verbose=False,
        )
        ar_kd_eval = evaluate_student_model(m_ar_kd, test_ds, device)

        # Arm 3: + Level 2 Dynamic Residual Concept-DAgger (kl_weight=0.0 isolates Concept-DAgger)
        m_arm3 = copy.deepcopy(m_arm2)
        train_draft_alignment(
            m_arm3,
            seqkd_ds,
            teacher_model=m_arm2,
            steps=align_steps,
            batch_size=batch_size,
            kl_weight=0.0,
            dagger_prob=0.75,
            freeze_backbone_for_exact=True,
            device=device,
            seed=seed,
            verbose=False,
        )
        arm3_eval = evaluate_student_model(m_arm3, test_ds, device)

        # Arm 4: + Level 3 Full Cross-Mode Exact->Concept KD + Concept-DAgger
        m_arm4 = copy.deepcopy(m_arm2)
        train_draft_alignment(
            m_arm4,
            seqkd_ds,
            teacher_model=m_arm2,
            steps=align_steps,
            batch_size=batch_size,
            kl_weight=0.5,
            dagger_prob=0.75,
            freeze_backbone_for_exact=True,
            device=device,
            seed=seed,
            verbose=False,
        )
        arm4_eval = evaluate_student_model(m_arm4, test_ds, device)

        return {
            "seqkd_source": seqkd_source,
            "seqkd_loaded_records": len(loaded_jsonl_texts),
            "arm1_baseline_raw_exact_nll": arm1_eval["exact_nll"],
            "arm2_seqkd_exact_nll": arm2_eval["exact_nll"],
            "pure_ar_seqkd_exact_nll": ar_kd_eval["exact_nll"],
            "arm1_baseline_raw_draft_nll": arm1_eval["concept_draft_nll"],
            "arm2_seqkd_draft_nll": arm2_eval["concept_draft_nll"],
            "arm3_dagger_draft_nll": arm3_eval["concept_draft_nll"],
            "arm4_full_kd_draft_nll": arm4_eval["concept_draft_nll"],
            "arm2_alpha_before": arm2_eval["alpha_full_chunk"],
            "arm3_alpha_dagger": arm3_eval["alpha_full_chunk"],
            "arm4_alpha_full_kd": arm4_eval["alpha_full_chunk"],
            "arm4_alpha_gain_over_arm2": round(arm4_eval["alpha_full_chunk"] - arm2_eval["alpha_full_chunk"], 5),
            "arm4_subchunk_alpha_g16": arm4_eval["alpha_subchunk_gamma"],
            "arm4_expected_lossless_speedup": arm4_eval["expected_lossless_speedup_gamma"],
            "arm4_concept_loss": arm4_eval["concept_loss"],
            "arm2_concept_loss": arm2_eval["concept_loss"],
        }

    res = run_seeded_experiment(
        f"distill_hierarchical_{preset}",
        _one_seed,
        seeds=seeds,
        config={
            "preset": preset,
            "teacher_model": teacher_name,
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "align_steps": align_steps,
            "batch_size": batch_size,
            "device": device,
        },
        steps=stage2_steps + align_steps,
    )

    if len(seeds) >= 2:
        a_before = [r["metrics"]["arm2_alpha_before"] for r in res["per_seed"]]
        a_after = [r["metrics"]["arm4_alpha_full_kd"] for r in res["per_seed"]]
        d_raw = [r["metrics"]["arm1_baseline_raw_exact_nll"] for r in res["per_seed"]]
        d_kd = [r["metrics"]["arm2_seqkd_exact_nll"] for r in res["per_seed"]]
        res["paired_tests"] = {
            "seqkd_vs_raw_exact_nll": paired_t_test(d_kd, d_raw),
            "full_kd_vs_pre_align_alpha": paired_t_test(a_after, a_before),
        }
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="CNCM 4-Arm Hierarchical Student-Teacher Distillation Benchmark (Teacher: Bonsai-2-4B, Single-Seed Default)")
    parser.add_argument("--preset", type=str, default="tiny_test")
    parser.add_argument("--teacher-name", type=str, default="Bonsai-2-4B")
    parser.add_argument("--seqkd-jsonl", type=str, default=None, help="Path to real verified_seqkd.jsonl from distill_seqkd.py")
    parser.add_argument("--stage1-steps", type=int, default=80)
    parser.add_argument("--stage2-steps", type=int, default=120)
    parser.add_argument("--align-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out = run_hierarchical_distill_ablation(
        preset=args.preset,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        align_steps=args.align_steps,
        batch_size=args.batch_size,
        seeds=args.seeds,
        device=args.device,
        teacher_name=args.teacher_name,
        seqkd_jsonl=Path(args.seqkd_jsonl) if args.seqkd_jsonl else None,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
