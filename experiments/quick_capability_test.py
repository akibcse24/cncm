"""One-Command CNCM Capability Test (`experiments/quick_capability_test.py`).

Runs a fast, end-to-end capability check on CNCM (`tiny_test`, `nano`, `mini`, or `base`)
and prints a clean scorecard + side-by-side Concept Mode vs. Exact-AR generations:
  1. Trains (or loads) Stage 1 M-RFSQ + Stage 2 Joint + Draft Alignment (1 seed by default).
  2. Reports held-out Stage 1 reconstruction accuracy and Exact-AR NLL.
  3. Reports self-speculative acceptance rate `alpha` and theoretical expected speedup.
  4. Evaluates on the SHA-256 locked held-out PG-FIM test set (36 unseen tasks) and
     held-out tool-calling suite with Wilson 95% confidence intervals.
  5. Saves the full scorecard to `experiments/results/quick_capability_<preset>.json`.

Usage:
    python -m experiments.quick_capability_test --preset tiny_test --steps 40
    python -m experiments.quick_capability_test --preset nano --steps 200 --align-steps 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from experiments.eval_suites import (
    MANIFEST_PATH,
    build_heldout_tool_calling_suite,
    build_pg_fim_heldout_test_bank,
    evaluate_fim_suite,
    evaluate_tool_suite_with_ci,
    verify_manifest,
)
from experiments.harness import save_result, set_global_seed
from experiments.stats import expected_speculative_speedup


def run_quick_capability_test(
    preset: str = "tiny_test",
    stage1_steps: int = 40,
    stage2_steps: int = 60,
    align_steps: int = 20,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cpu",
    eval_tool_n: int = 50,
    qat_int4_distill: bool = False,
) -> Dict[str, Any]:
    """Run one-command CNCM capability scorecard."""
    import torch  # type: ignore[import-not-found]
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.dataset import PackedChunkDataset, make_synthetic_corpus
    from cncm.trainer.draft_align import train_draft_alignment
    from cncm.trainer.train_stage1 import train_stage1
    from cncm.trainer.train_stage2 import train_stage2

    t0 = time.perf_counter()
    set_global_seed(seed)
    cfg = get_preset(preset, enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    train_texts = make_synthetic_corpus(num_samples=256, seed=seed)
    test_texts = make_synthetic_corpus(num_samples=64, seed=seed + 20_000)
    train_ds = PackedChunkDataset(train_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    test_ds = PackedChunkDataset(test_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)

    # 1. Stage 1
    s1 = CNCMConceptTokenizer(cfg)
    s1_res = train_stage1(
        cfg, tok, model=s1, dataset=train_ds,
        steps=stage1_steps, batch_size=batch_size, device=device, seed=seed, verbose=False,
    )

    # 2. Stage 2 Joint
    model = CNCMModel(cfg, s1)
    s2_res = train_stage2(
        cfg, tok, model=model, dataset=train_ds,
        mode="joint", steps=stage2_steps, batch_size=batch_size,
        concept_weight=1.0, exact_weight=0.5, device=device, seed=seed, verbose=False,
    )

    # 3. Draft Alignment (Exact-AR -> Concept Mode token-level KL + SeqKD)
    da_res = train_draft_alignment(
        model, train_ds, steps=align_steps, batch_size=batch_size,
        device=device, seed=seed, verbose=False,
    )

    # Optional Step 3 of distillation recipe: FP32 -> INT4 QAT logit KL
    qat_res: Dict[str, Any] = {"skipped": True}
    if qat_int4_distill:
        qat_res = train_stage2(
            cfg, tok, model=model, dataset=train_ds,
            mode="joint", steps=20, batch_size=batch_size,
            qat_finetune=True, qat_bits=4,
            self_distill=True, self_distill_weight=0.25, self_distill_temp=2.0,
            device=device, seed=seed + 1, verbose=False,
        )

    # 4. Measure Exact-AR NLL + Self-Speculative Acceptance alpha + Side-by-Side Samples
    model.eval()
    nlls: List[float] = []
    accepted_toks, drafted_toks = 0, 0
    samples: List[Dict[str, str]] = []
    with torch.no_grad():
        for i in range(min(8, len(test_ds))):
            item = test_ds[i]
            ex_ids = item["exact_input_ids"].unsqueeze(0).to(device)
            ex_lbl = item["exact_labels"].unsqueeze(0).to(device)
            ex_msk = item["exact_attention_mask"].unsqueeze(0).to(device)
            nlls.append(float(model.forward_exact(ex_ids, labels=ex_lbl, attention_mask=ex_msk)["loss"].item()))

            pfx = item["prefix_ids"].unsqueeze(0).to(device)
            ar_ids = model.generate_exact(pfx, max_new_tokens=cfg.chunk_size, temperature=0.0)["token_ids"]
            c_ids = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)["token_ids"]
            matches = 0
            for t in range(cfg.chunk_size):
                if int(c_ids[0, t].item()) == int(ar_ids[0, t].item()):
                    matches += 1
                else:
                    break
            accepted_toks += matches
            drafted_toks += cfg.chunk_size
            if i < 2:
                samples.append(
                    {
                        "prefix": tok.decode(pfx[0].tolist())[:120],
                        "exact_ar": tok.decode(ar_ids[0].tolist())[:160],
                        "concept_mode": tok.decode(c_ids[0].tolist())[:160],
                    }
                )

    alpha = accepted_toks / float(max(1, drafted_toks))
    theo = expected_speculative_speedup(alpha=alpha, gamma=cfg.chunk_size, draft_cycle_cost_ratio=8.0, verify_cycle_cost_ratio=1.0)

    # 5. Held-out PG-FIM (36 unseen tasks) & Tool-Calling (syntax vs semantic)
    manifest_ok = verify_manifest(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
    fim_tasks = build_pg_fim_heldout_test_bank()

    def _predict_fim(prefix: str, suffix: str) -> str:
        prompt_ids = torch.tensor([tok.encode(prefix)[-min(cfg.max_position_embeddings // 2, 128) :]], dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = model.generate_concept_chunk(prompt_ids, temperature=0.0, early_exit_conf=None)["token_ids"]
        return tok.decode(out_ids[0].tolist())

    fim_eval = evaluate_fim_suite(fim_tasks, _predict_fim)

    tool_tasks = build_heldout_tool_calling_suite(n_items=eval_tool_n)
    tool_preds: List[str] = []
    with torch.no_grad():
        for t in tool_tasks:
            p_ids = torch.tensor([tok.encode(t["prompt"])[-128:]], dtype=torch.long, device=device)
            out_ids = model.generate_exact(p_ids, max_new_tokens=48, temperature=0.0)["token_ids"]
            tool_preds.append(tok.decode(out_ids[0].tolist()))
    tool_eval = evaluate_tool_suite_with_ci(tool_preds, tool_tasks)

    report: Dict[str, Any] = {
        "preset": preset,
        "seed": seed,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "params": model.count_parameters(),
        "manifest_verified": manifest_ok,
        "stage1_final_loss": s1_res.get("final_loss"),
        "stage2_final_loss": s2_res.get("final_loss"),
        "draft_align_final_kl": da_res.get("final_kl_loss"),
        "qat_int4_distill": {"enabled": qat_int4_distill, "final_loss": qat_res.get("final_loss")},
        "heldout_exact_nll": round(sum(nlls) / max(1, len(nlls)), 5),
        "self_speculative": {"alpha": round(alpha, 5), **theo},
        "pg_fim_heldout_36": fim_eval,
        "tool_calling_eval": tool_eval,
        "sample_generations": samples,
    }
    save_result(
        f"quick_capability_{preset}",
        report,
        config={"preset": preset, "stage1_steps": stage1_steps, "stage2_steps": stage2_steps, "align_steps": align_steps, "qat_int4_distill": qat_int4_distill},
        seeds=[seed],
        steps=stage2_steps,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="One-Command CNCM Capability Test")
    parser.add_argument("--preset", type=str, default="tiny_test")
    parser.add_argument("--stage1-steps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--align-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-tool-n", type=int, default=50)
    parser.add_argument("--qat-int4-distill", action="store_true")
    args = parser.parse_args()

    res = run_quick_capability_test(
        preset=args.preset,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.steps,
        align_steps=args.align_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        eval_tool_n=args.eval_tool_n,
        qat_int4_distill=args.qat_int4_distill,
    )
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
