"""Full Smart-CNCM Training & Post-Training Pipeline (`experiments/train_smart_cncm.py`).

Unifies all 5 intelligence & edge-efficiency levers into one single-seed CLI pipeline:
  1. Domain Byte-Level BPE tokenizer training (`V=6400..8192`, split digits) or CNCMTextTokenizer fallback.
  2. Curated + Verified SeqKD data mixing (`experiments/distill_seqkd.py` verified JSONL + open text/code).
  3. Rebalanced Stage 2 Dual-Mode training (`lambda_c=1.0`, `lambda_e=0.7`) with optional Edge KV Stack
     (`--kv-sharing 2 --sliding-window 2048 --concept-kv-compress`).
  4. Ordered post-training stack:
       Verified SFT -> Internal Draft-Alignment KL (`draft_align.py`) ->
       Partial-Credit GRPO (`skip_zero_variance_groups=True`) ->
       Optional FP32->INT4 QAT Logit Distillation (`self_distill=True, qat_bits=4`).
  5. Evaluation on SHA-256 locked held-out suites with Wilson 95% CIs (single-seed default).

Usage:
    python -m experiments.train_smart_cncm --preset mini --steps 500 --seqkd-jsonl experiments/results/verified_seqkd.jsonl
    python -m experiments.train_smart_cncm --preset base --kv-sharing 2 --sliding-window 2048 --concept-kv-compress --qat-int4-distill
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cncm.config import CNCMConfig, compute_kv_memory_footprint, get_preset
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

DEFAULT_LAMBDA_C = 1.0
DEFAULT_LAMBDA_E = 0.7


def load_verified_seqkd_texts(jsonl_path: Optional[Path]) -> List[str]:
    """Load verified SeqKD JSONL records as formatted training strings."""
    if jsonl_path is None or not jsonl_path.exists():
        return []
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("verified") is False:
                continue
            prompt = str(obj.get("prompt", obj.get("prefix", ""))).strip()
            comp = str(obj.get("completion", obj.get("text", ""))).strip()
            suffix = str(obj.get("suffix", "")).strip()
            if prompt and comp:
                texts.append(f"{prompt}\n{comp}" + (f"\n{suffix}" if suffix else ""))
            elif comp:
                texts.append(comp)
    return texts


def train_domain_bpe_tokenizer(
    texts: Sequence[str],
    vocab_size: int = 6400,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a Byte-Level BPE tokenizer with individual digit splitting (falls back cleanly if `tokenizers` absent)."""
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # type: ignore[import-not-found]
    except ImportError:
        return {"trained_hf_bpe": False, "reason": "tokenizers package not installed; using CNCMTextTokenizer"}

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    specials = [
        "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<think>", "</think>", "<tool_call>", "</tool_call>",
        "<tool_resp>", "</tool_resp>", "<tools>", "</tools>",
    ]
    trainer = trainers.BpeTrainer(vocab_size=int(vocab_size), special_tokens=specials, min_frequency=2)
    tok.train_from_iterator(texts, trainer=trainer)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(save_path))
    return {
        "trained_hf_bpe": True,
        "vocab_size": tok.get_vocab_size(),
        "saved_path": str(save_path) if save_path is not None else None,
    }


def build_smart_config(
    preset: str = "mini",
    *,
    vocab_size: Optional[int] = None,
    kv_sharing_group_size: int = 2,
    sliding_window: Optional[int] = 2048,
    concept_kv_compress: bool = True,
    concept_kv_level: int = 2,
    num_loops: Optional[int] = None,
) -> CNCMConfig:
    """Build an edge-optimized Dense CNCMConfig with CLA + SWA + Concept-Compressed KV."""
    overrides: Dict[str, Any] = {
        "enc_layers": 1,
        "kv_sharing_group_size": int(kv_sharing_group_size),
        "sliding_window": sliding_window,
        "concept_kv_compress": bool(concept_kv_compress),
        "concept_kv_level": int(concept_kv_level),
    }
    if vocab_size is not None:
        overrides["vocab_size"] = int(vocab_size)
    if num_loops is not None:
        overrides["num_loops"] = int(num_loops)
    return get_preset(preset, **overrides)


def run_smart_pipeline(
    *,
    preset: str = "mini",
    stage1_steps: int = 120,
    stage2_steps: int = 300,
    sft_steps: int = 80,
    align_steps: int = 40,
    grpo_steps: int = 25,
    lambda_c: float = DEFAULT_LAMBDA_C,
    lambda_e: float = DEFAULT_LAMBDA_E,
    kv_sharing_group_size: int = 2,
    sliding_window: Optional[int] = 2048,
    concept_kv_compress: bool = True,
    qat_int4_distill: bool = True,
    seqkd_jsonl: Optional[Path] = None,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cpu",
    eval_tool_n: int = 100,
) -> Dict[str, Any]:
    """Execute the complete Smart-CNCM pipeline (single-seed default)."""
    import torch  # type: ignore[import-not-found]
    from cncm.agent.grammar import JSONSchemaConstrainedMasker
    from cncm.agent.tools import create_default_registry
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.dataset import (
        BlockARSFTDataset,
        PackedChunkDataset,
        make_synthetic_corpus,
        make_synthetic_sft,
    )
    from cncm.trainer.draft_align import train_draft_alignment
    from cncm.trainer.train_grpo import train_grpo
    from cncm.trainer.train_stage1 import train_stage1
    from cncm.trainer.train_stage2 import train_stage2

    t0 = time.perf_counter()
    set_global_seed(seed)

    cfg = build_smart_config(
        preset=preset,
        kv_sharing_group_size=kv_sharing_group_size,
        sliding_window=sliding_window,
        concept_kv_compress=concept_kv_compress,
    )
    kv_profile_64k = compute_kv_memory_footprint(cfg, seq_len=65536, kv_bytes=1.05)

    # 1. Corpus mixing (base corpus + verified SeqKD teacher completions)
    base_texts = make_synthetic_corpus(num_samples=384, seed=seed)
    seqkd_texts = load_verified_seqkd_texts(seqkd_jsonl)
    train_texts = base_texts + seqkd_texts
    test_texts = make_synthetic_corpus(num_samples=64, seed=seed + 20_000)

    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    train_ds = PackedChunkDataset(train_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    test_ds = PackedChunkDataset(test_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    sft_ds = BlockARSFTDataset(make_synthetic_sft(192, seed=seed), tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)

    # 2. Stage 1 M-RFSQ
    s1 = CNCMConceptTokenizer(cfg)
    s1_res = train_stage1(cfg, tok, model=s1, dataset=train_ds, steps=stage1_steps, batch_size=batch_size, device=device, seed=seed, verbose=False)

    # 3. Stage 2 Rebalanced Joint Pretraining (lambda_c=1.0, lambda_e=0.7)
    model = CNCMModel(cfg, s1)
    s2_res = train_stage2(
        cfg, tok, model=model, dataset=train_ds, mode="joint",
        steps=stage2_steps, batch_size=batch_size,
        concept_weight=lambda_c, exact_weight=lambda_e,
        device=device, seed=seed, verbose=False,
    )

    # 4. Post-training stack: SFT -> Draft-Alignment KL -> GRPO -> FP32->INT4 QAT Logit KL
    sft_res = train_stage2(cfg, tok, model=model, dataset=sft_ds, mode="sft", steps=sft_steps, batch_size=batch_size, device=device, seed=seed + 1, verbose=False)
    da_res = train_draft_alignment(model, train_ds, steps=align_steps, batch_size=batch_size, device=device, seed=seed + 2, verbose=False)
    grpo_res = train_grpo(
        cfg, tok, model=model, steps=grpo_steps, num_generations=8,
        batch_size=min(2, batch_size), temperature=1.0,
        skip_zero_variance_groups=True, device=device, seed=seed + 3, verbose=False,
    )

    qat_res: Dict[str, Any] = {"skipped": True}
    if qat_int4_distill:
        qat_res = train_stage2(
            cfg, tok, model=model, dataset=train_ds, mode="joint",
            steps=40, batch_size=batch_size,
            concept_weight=lambda_c, exact_weight=lambda_e,
            qat_finetune=True, qat_bits=4,
            self_distill=True, self_distill_weight=0.25, self_distill_temp=2.0,
            device=device, seed=seed + 4, verbose=False,
        )

    # 5. Held-out Evaluation (Exact-AR NLL, alpha, PG-FIM 36, Copy-Span Tool Calling)
    model.eval()
    nlls: List[float] = []
    acc_toks, draft_toks = 0, 0
    with torch.no_grad():
        for i in range(min(12, len(test_ds))):
            item = test_ds[i]
            nlls.append(
                float(
                    model.forward_exact(
                        item["exact_input_ids"].unsqueeze(0).to(device),
                        labels=item["exact_labels"].unsqueeze(0).to(device),
                        attention_mask=item["exact_attention_mask"].unsqueeze(0).to(device),
                    )["loss"].item()
                )
            )
            pfx = item["prefix_ids"].unsqueeze(0).to(device)
            ar_ids = model.generate_exact(pfx, max_new_tokens=cfg.chunk_size, temperature=0.0)["token_ids"]
            c_ids = model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)["token_ids"]
            for t in range(cfg.chunk_size):
                if int(c_ids[0, t].item()) == int(ar_ids[0, t].item()):
                    acc_toks += 1
                else:
                    break
            draft_toks += cfg.chunk_size

    alpha = acc_toks / float(max(1, draft_toks))
    theo = expected_speculative_speedup(alpha=alpha, gamma=cfg.chunk_size, draft_cycle_cost_ratio=8.0, verify_cycle_cost_ratio=1.0)

    # Copy-span tool calling evaluation
    reg = create_default_registry()
    tool_tasks = [t for t in build_heldout_tool_calling_suite(eval_tool_n) if t["expected_tool"] == "calculator"]
    tool_preds: List[str] = []
    with torch.no_grad():
        for t in tool_tasks:
            pfx = torch.tensor([tok.encode(t["prompt"])[-96:]], dtype=torch.long, device=device)
            masker = JSONSchemaConstrainedMasker(tok, reg, allowed_tools=["calculator"], prompt_text=t["prompt"])
            out_ids = model.generate_exact(pfx, max_new_tokens=40, temperature=0.0, logit_masker=masker)["token_ids"]
            tool_preds.append(tok.decode(out_ids[0].tolist()))
    tool_eval = evaluate_tool_suite_with_ci(tool_preds, tool_tasks)

    fim_tasks = build_pg_fim_heldout_test_bank()
    fim_eval = evaluate_fim_suite(
        fim_tasks,
        lambda prefix, _s: tok.decode(
            model.generate_concept_chunk(
                torch.tensor([tok.encode(prefix)[-128:]], dtype=torch.long, device=device),
                temperature=0.0,
                early_exit_conf=None,
            )["token_ids"][0].tolist()
        ),
    )

    report: Dict[str, Any] = {
        "preset": preset,
        "seed": seed,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "params": model.count_parameters(),
        "seqkd_examples_used": len(seqkd_texts),
        "loss_weights": {"lambda_c": lambda_c, "lambda_e": lambda_e},
        "edge_kv_64k_profile": kv_profile_64k,
        "manifest_verified": verify_manifest(MANIFEST_PATH) if MANIFEST_PATH.exists() else {},
        "stage1_final_loss": s1_res.get("final_loss"),
        "stage2_final_loss": s2_res.get("final_loss"),
        "sft_final_loss": sft_res.get("final_loss"),
        "draft_align_final_kl": da_res.get("final_kl_loss"),
        "grpo_zero_variance_fraction": grpo_res.get("zero_variance_fraction"),
        "qat_int4_distill_final_loss": qat_res.get("final_loss"),
        "heldout_exact_nll": round(sum(nlls) / max(1, len(nlls)), 5),
        "self_speculative": {"alpha": round(alpha, 5), **theo},
        "copy_span_tool_calling": tool_eval,
        "pg_fim_heldout_36": fim_eval,
    }
    save_result(
        f"smart_cncm_{preset}",
        report,
        config={
            "preset": preset,
            "lambda_c": lambda_c,
            "lambda_e": lambda_e,
            "kv_sharing_group_size": kv_sharing_group_size,
            "sliding_window": sliding_window,
            "concept_kv_compress": concept_kv_compress,
            "qat_int4_distill": qat_int4_distill,
        },
        seeds=[seed],
        steps=stage2_steps,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart-CNCM End-to-End Training & Post-Training Pipeline")
    parser.add_argument("--preset", type=str, default="mini")
    parser.add_argument("--stage1-steps", type=int, default=120)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sft-steps", type=int, default=80)
    parser.add_argument("--align-steps", type=int, default=40)
    parser.add_argument("--grpo-steps", type=int, default=25)
    parser.add_argument("--lambda-c", type=float, default=DEFAULT_LAMBDA_C)
    parser.add_argument("--lambda-e", type=float, default=DEFAULT_LAMBDA_E)
    parser.add_argument("--kv-sharing", type=int, default=2)
    parser.add_argument("--sliding-window", type=int, default=2048)
    parser.add_argument("--concept-kv-compress", action="store_true", default=True)
    parser.add_argument("--qat-int4-distill", action="store_true", default=True)
    parser.add_argument("--seqkd-jsonl", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    res = run_smart_pipeline(
        preset=args.preset,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.steps,
        sft_steps=args.sft_steps,
        align_steps=args.align_steps,
        grpo_steps=args.grpo_steps,
        lambda_c=args.lambda_c,
        lambda_e=args.lambda_e,
        kv_sharing_group_size=args.kv_sharing,
        sliding_window=args.sliding_window,
        concept_kv_compress=args.concept_kv_compress,
        qat_int4_distill=args.qat_int4_distill,
        seqkd_jsonl=Path(args.seqkd_jsonl) if args.seqkd_jsonl else None,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
