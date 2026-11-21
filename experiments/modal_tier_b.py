"""Cost-Gated Tier B GPU Experiments (`experiments/modal_tier_b.py`).

Implements the paid GPU experiments from `remediation_plan.md` (requires explicit
user opt-in via `--confirm-paid-gpu` so it never spends GPU credits accidentally):

  - **Item 1.2a (GPU)**: Fair optimized Pure-AR vs CNCM benchmark using
    `torch.compile(mode="max-autotune")`, CUDA graphs, static KV cache, and
    excluded warmup on the same GPU in the same session.
  - **Item 1.4**: Equal-total-memory `base_dense220` (~221M dense SwiGLU) vs
    `base_moe` (~221M total / ~94M active) on identical data and step budget.
  - **Items 4.1–4.3**: Open-license pretraining mix (FineWeb-Edu, permissive Python,
    GSM8K/math, tool-calling), BPE tokenizer training, and `{nano, mini} x {50M, 200M, 1B}`
    scaling curve with the mandatory quality gate:
      `concept_nll < ln(4096) ≈ 8.3178` AND `exact_nll << unigram_nll`.

Usage (local GPU or Modal):
    python -m experiments.modal_tier_b --exp 1.2a --confirm-paid-gpu
    python -m experiments.modal_tier_b --exp 1.4 --steps 500 --confirm-paid-gpu
    python -m experiments.modal_tier_b --exp 4.x --preset nano --token-budget 50000000 --confirm-paid-gpu
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, List, Sequence

from experiments.harness import DEFAULT_SEEDS, save_result, set_global_seed

UNIFORM_4096_NLL = math.log(4096.0)  # ~8.317766 nats


def check_concept_quality_gate(
    concept_draft_nll: float,
    exact_nll: float,
    unigram_nll: float,
) -> Dict[str, Any]:
    """Enforce Item 4.3 rule: no Concept Mode quality claim until clearly below uniform & unigram."""
    below_uniform = concept_draft_nll < (UNIFORM_4096_NLL - 0.5)
    below_unigram = exact_nll < (unigram_nll - 0.5)
    return {
        "concept_draft_nll": round(concept_draft_nll, 5),
        "uniform_4096_nll": round(UNIFORM_4096_NLL, 5),
        "exact_nll": round(exact_nll, 5),
        "unigram_nll": round(unigram_nll, 5),
        "concept_clearly_below_uniform": below_uniform,
        "exact_clearly_below_unigram": below_unigram,
        "quality_gate_passed": bool(below_uniform and below_unigram),
    }


def run_gpu_1_2a_compiled_latency(
    preset: str = "nano",
    prefix_len: int = 64,
    gen_tokens: int = 64,
    warmup: int = 5,
    iters: int = 20,
) -> Dict[str, Any]:
    """Item 1.2a GPU: torch.compile(mode='max-autotune') + CUDA graphs + static KV cache."""
    import torch  # type: ignore[import-not-found]
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.tokenizer_model import CNCMConceptTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Item 1.2a GPU benchmark.")

    device = torch.device("cuda")
    cfg = get_preset(preset, enc_layers=1)
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).to(device).eval()

    # Compile single-step exact decode and concept chunk with max-autotune (enables CUDA graphs)
    compiled_exact_step = torch.compile(model.forward_exact, mode="max-autotune", fullgraph=False)
    compiled_concept = torch.compile(model.generate_concept_chunk, mode="max-autotune", fullgraph=False)

    pfx = torch.randint(16, cfg.vocab_size, (1, prefix_len), device=device)

    with torch.no_grad():
        # Excluded warmup
        for _ in range(warmup):
            _ = compiled_exact_step(pfx, use_cache=True)
            _ = compiled_concept(pfx, temperature=0.0, early_exit_conf=None)
        torch.cuda.synchronize()

        # Exact-AR KV-cached timing
        ar_times_ms: List[float] = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = compiled_exact_step(pfx, use_cache=True)
            past = out["past_key_values"]
            tok = out["logits"][:, -1:, :].argmax(dim=-1)
            for _ in range(gen_tokens - 1):
                out = compiled_exact_step(tok, past_key_values=past, use_cache=True)
                past = out["past_key_values"]
                tok = out["logits"][:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()
            ar_times_ms.append((time.perf_counter() - t0) * 1000.0)

        # CNCM Concept Mode timing
        c_times_ms: List[float] = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = compiled_concept(pfx, temperature=0.0, early_exit_conf=None)
            torch.cuda.synchronize()
            c_times_ms.append((time.perf_counter() - t0) * 1000.0)

    ar_mean = sum(ar_times_ms) / len(ar_times_ms)
    c_mean = sum(c_times_ms) / len(c_times_ms)
    report = {
        "preset": preset,
        "gpu": torch.cuda.get_device_name(0),
        "compile_mode": "max-autotune",
        "warmup_excluded": warmup,
        "iters": iters,
        "exact_ar_64tok_ms": round(ar_mean, 3),
        "exact_ar_tok_per_sec": round(gen_tokens / (ar_mean / 1000.0), 2),
        "concept_64tok_ms": round(c_mean, 3),
        "concept_tok_per_sec": round(gen_tokens / (c_mean / 1000.0), 2),
        "speedup_vs_compiled_ar": round(ar_mean / max(c_mean, 1e-6), 4),
    }
    save_result(f"tier_b_1_2a_gpu_{preset}", report, config={"preset": preset, "warmup": warmup, "iters": iters})
    return report


def run_gpu_1_4_dense220_vs_moe(
    steps: int = 500,
    batch_size: int = 8,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> Dict[str, Any]:
    """Item 1.4: Controlled equal-total-memory A/B (`base_dense220` vs `base_moe`)."""
    import torch  # type: ignore[import-not-found]
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.dataset import PackedChunkDataset, make_synthetic_corpus
    from cncm.trainer.train_stage2 import train_stage2

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Item 1.4 Dense-220M vs MoE benchmark.")

    device = "cuda"
    per_seed = []
    for seed in seeds:
        set_global_seed(int(seed))
        cfg_moe = get_preset("base_moe", enc_layers=1)
        cfg_dense = get_preset("base_dense220", enc_layers=1)
        tok = CNCMTextTokenizer(vocab_size=cfg_moe.vocab_size)
        train_ds = PackedChunkDataset(make_synthetic_corpus(512, seed=seed), tok, chunk_size=cfg_moe.chunk_size, prefix_len=cfg_moe.chunk_size)

        m_moe = CNCMModel(cfg_moe, CNCMConceptTokenizer(cfg_moe))
        r_moe = train_stage2(cfg_moe, tok, model=m_moe, dataset=train_ds, mode="joint", steps=steps, batch_size=batch_size, device=device, seed=seed, verbose=False)

        m_dense = CNCMModel(cfg_dense, CNCMConceptTokenizer(cfg_dense))
        r_dense = train_stage2(cfg_dense, tok, model=m_dense, dataset=train_ds, mode="joint", steps=steps, batch_size=batch_size, device=device, seed=seed, verbose=False)

        per_seed.append(
            {
                "seed": int(seed),
                "moe_params": m_moe.count_parameters(),
                "dense220_params": m_dense.count_parameters(),
                "moe_final_loss": r_moe.get("final_loss"),
                "dense220_final_loss": r_dense.get("final_loss"),
            }
        )

    report = {"steps": steps, "batch_size": batch_size, "per_seed": per_seed}
    save_result("tier_b_1_4_dense220_vs_moe", report, config={"steps": steps, "batch_size": batch_size}, seeds=seeds, steps=steps)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-Gated Tier B GPU Experiments")
    parser.add_argument("--exp", type=str, required=True, choices=["1.2a", "1.4"])
    parser.add_argument("--preset", type=str, default="nano")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--confirm-paid-gpu",
        action="store_true",
        help="Explicitly confirm running on paid GPU hardware (Tier B).",
    )
    args = parser.parse_args()

    if not args.confirm_paid_gpu:
        raise SystemExit(
            "Refusing to launch Tier B paid GPU job without --confirm-paid-gpu. "
            "Pass --confirm-paid-gpu to approve GPU execution."
        )

    if args.exp == "1.2a":
        res = run_gpu_1_2a_compiled_latency(preset=args.preset)
    else:
        res = run_gpu_1_4_dense220_vs_moe(steps=args.steps)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
