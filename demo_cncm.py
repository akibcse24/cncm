"""CNCM End-to-End Interactive Demo & Verification CLI.

Showcases all 4 core capabilities of CNCM (Compact Causal-Hybrid Next-Concept Model):
  1. Architecture specs & edge hardware profile (`nano`, `mini`, `base`)
  2. Quick `tiny_test` Stage 1 (M-RFSQ) + Stage 2 (Joint) + Code-Space GRPO smoke run
  3. Dual-Mode Agentic AI execution with deterministic JSON-Schema tool calling
  4. Real packed INT4 / per-channel INT8 quantization (size, speed, accuracy) + ONNX wrapper check

Usage:
    python demo_cncm.py
    python demo_cncm.py --section all --stage1-steps 6 --stage2-steps 6 --grpo-steps 2
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch

from cncm.agent import CNCMAgent, create_default_registry
from cncm.config import get_preset
from cncm.export import (
    export_cncm_onnx,
    format_edge_profile_table,
    measure_model_size,
    profile_edge_targets,
    quantize_model_int4,
    quantize_model_int8,
)
from cncm.model import CNCMModel
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer import (
    BlockARSFTDataset,
    FIMPlanDataset,
    PackedChunkDataset,
    compute_fim_reward,
    make_synthetic_corpus,
    make_synthetic_fim_tasks,
    make_synthetic_sft,
    train_grpo,
    train_stage1,
    train_stage2,
)


def _banner(title: str) -> None:
    print("\n" + "=" * 96)
    print(f"  {title}")
    print("=" * 96)


def run_profile_section(instantiate_nano: bool = True) -> None:
    _banner("1. CNCM Architecture Specs & Multi-Device Edge Hardware Profile")
    for preset_name in ("nano", "mini", "base"):
        cfg = get_preset(preset_name)
        model_for_exact = None
        if instantiate_nano and preset_name == "nano":
            s1 = CNCMConceptTokenizer(cfg)
            model_for_exact = CNCMModel(cfg, s1)
        profile = profile_edge_targets(cfg, model=model_for_exact, context_len=1024)
        print(format_edge_profile_table(profile))
        print()


def run_training_smoke_section(
    stage1_steps: int = 6,
    stage2_steps: int = 6,
    grpo_steps: int = 2,
    device: str = "cpu",
) -> tuple[Any, CNCMTextTokenizer, CNCMModel]:
    _banner("2. End-to-End Training Smoke Run (tiny_test: Stage 1 -> Stage 2 Joint -> GRPO)")
    cfg = get_preset("tiny_test")
    tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    # Stage 1: Multiscale RFSQ Concept Tokenizer
    corpus = make_synthetic_corpus(num_samples=96, seed=42)
    s1_ds = PackedChunkDataset(
        corpus,
        tokenizer,
        chunk_size=cfg.chunk_size,
        eos_token_id=cfg.eos_token_id,
        bos_token_id=cfg.bos_token_id,
    )
    t0 = time.perf_counter()
    s1_res = train_stage1(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset=s1_ds,
        steps=stage1_steps,
        batch_size=8,
        lr=3e-3,
        device=device,
        seed=42,
        log_every=max(1, stage1_steps),
        verbose=False,
    )
    s1_model: CNCMConceptTokenizer = s1_res["model"]
    s1_m: Dict[str, Any] = s1_res["metrics"]
    print(
        f"[Stage 1 M-RFSQ] {stage1_steps} steps in {(time.perf_counter()-t0):.2f}s | "
        f"loss: {s1_m.get('initial_loss', 0.0):.4f} -> {s1_m.get('final_loss', 0.0):.4f} | "
        f"recon_acc={s1_m.get('final_recon_acc', 0.0):.3f} | "
        f"active_code_ratio={s1_m.get('final_active_code_ratio', 0.0):.3f}"
    )

    # Stage 2: Joint Concept Mode + Exact-AR Mode (Stage 1 strictly frozen)
    sft_data = make_synthetic_sft(num_samples=48, seed=42)
    s2_ds = BlockARSFTDataset(
        sft_data,
        tokenizer,
        chunk_size=cfg.chunk_size,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
    )
    t0 = time.perf_counter()
    s2_res = train_stage2(
        cfg=cfg,
        tokenizer=tokenizer,
        concept_tokenizer=s1_model,
        dataset=s2_ds,
        mode="joint",
        steps=stage2_steps,
        corrupt_rate=0.5,
        batch_size=4,
        lr=2e-3,
        device=device,
        seed=42,
        log_every=max(1, stage2_steps),
        verbose=False,
    )
    model: CNCMModel = s2_res["model"]
    s2_m: Dict[str, Any] = s2_res["metrics"]
    print(
        f"[Stage 2 Joint ] {stage2_steps} steps in {(time.perf_counter()-t0):.2f}s | "
        f"loss: {s2_m.get('initial_loss', 0.0):.4f} -> {s2_m.get('final_loss', 0.0):.4f} | "
        f"stage1_frozen={all(not p.requires_grad for p in model.concept_tokenizer.parameters())}"
    )

    # Stage 3: 1-Pass Code-Space GRPO RL
    t0 = time.perf_counter()
    grpo_res = train_grpo(
        cfg=cfg,
        tokenizer=tokenizer,
        model=model,
        prompts=None,
        steps=grpo_steps,
        num_generations=4,
        batch_size=2,
        lr=5e-4,
        device=device,
        seed=42,
        log_every=max(1, grpo_steps),
        verbose=False,
    )
    model = grpo_res["model"]
    gm: Dict[str, Any] = grpo_res["metrics"]
    print(
        f"[Code-Space RL ] {grpo_steps} GRPO steps in {(time.perf_counter()-t0):.2f}s | "
        f"mean_reward={gm.get('final_mean_reward', 0.0):.4f} | "
        f"all_logprobs_finite={gm.get('all_logprobs_finite', True)}"
    )
    return cfg, tokenizer, model


def run_agent_section(tokenizer: CNCMTextTokenizer, model: CNCMModel) -> None:
    _banner("3. Dual-Mode Agentic AI Execution (Concept Mode <-> JSON-Constrained Exact-AR Mode)")
    registry = create_default_registry()
    print(registry.format_prompt_schema())
    print()

    agent = CNCMAgent(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        max_turns=2,
        use_grammar_constraints=True,
        quantize_kv=True,
        early_exit_conf=0.92,
    )
    prompt = "Use calculator to compute 12 * 7 + 5."
    trace = agent.run(prompt, force_tool="calculator")
    print(trace.summary())
    print(f"Mode switches: {json.dumps(trace.mode_switches)}")

    # Also showcase Self-Speculative Verification on the same unified backbone
    prefix = torch.tensor(
        [tokenizer.encode("CNCM edge speed:", add_special_tokens=True)],
        dtype=torch.long,
    )
    spec_out = model.generate_self_speculative(
        prefix,
        max_new_tokens=model.cfg.chunk_size,
        temperature=0.0,
        quantize_kv=True,
    )
    print(
        f"[Self-Speculative] acceptance_rate={spec_out.get('acceptance_rate', 0.0):.2f} | "
        f"concept_steps={spec_out.get('concept_steps', 0)} | "
        f"verify_passes={spec_out.get('verify_passes', 0)}"
    )


def run_pg_fim_section(tokenizer: CNCMTextTokenizer, model: CNCMModel) -> None:
    _banner("4. Plan-Guided Fill-in-the-Middle (PG-FIM: Exact-AR Plan -> Parallel Concept Infill -> Verify/Repair)")
    task = make_synthetic_fim_tasks(num_samples=1, seed=42, curriculum_stage="single_line")[0]
    prefix_code = task["prefix"]
    suffix_code = task["suffix"]

    fim_out = model.generate_fim(
        prefix=prefix_code,
        suffix=suffix_code,
        tokenizer=tokenizer,
        max_chunks=2,
        temperature=0.0,
        early_exit_conf=0.92,
        quantize_kv=True,
        use_plan=True,
        enable_repair=True,
        oracle_keywords=task.get("oracle_keywords"),
    )
    plan_text = tokenizer.decode(fim_out["plan_token_ids"][0].tolist(), skip_special_tokens=False)
    reward_eval = compute_fim_reward(
        fim_out["middle_text"],
        task,
        planned_keywords=fim_out["keywords"],
        planned_chunks=fim_out["planned_chunks"],
        actual_chunks=1,
    )
    print(f"Task: {task.get('name', 'fim_task')} ({task.get('curriculum_stage', 'single_line')})")
    print(f"Emitted Plan: {plan_text} -> chunks={fim_out['planned_chunks']}, keywords={fim_out['keywords']}")
    print(
        f"Infill Status: parses={fim_out['parses']} | keywords_present={fim_out['keywords_present']} | "
        f"repair_triggered={fim_out['repair_triggered']}"
    )
    print(
        f"Sequential Passes: {fim_out['sequential_passes']} "
        f"(plan={fim_out['plan_steps']} + concept={fim_out['concept_steps']} + repair={fim_out['repair_passes']}) | "
        f"Speedup vs AR: {fim_out['speedup_vs_ar']:.2f}x"
    )
    print(
        f"Compiler Sandbox Reward: {reward_eval['reward']:.2f} "
        f"(non_trivial={reward_eval['non_trivial']}, compiles={reward_eval['compiles']}, "
        f"tests_passed={reward_eval['tests_passed']})"
    )


def run_quantization_section(
    tokenizer: CNCMTextTokenizer,
    model: CNCMModel,
    export_onnx_dir: str | None = None,
) -> None:
    _banner("5. Edge Quantization (Packed INT4 / Per-Channel INT8) & ONNX Graph Wrappers")
    model.eval()
    fp32_sz = measure_model_size(model)
    model_int8 = quantize_model_int8(model, quantize_embeddings=True).eval()
    model_int4 = quantize_model_int4(model, group_size=32, quantize_embeddings=True).eval()
    int8_sz = measure_model_size(model_int8)
    int4_sz = measure_model_size(model_int4)

    # Compare latency & logit fidelity on Exact-AR forward pass
    sample_ids = torch.tensor(
        [tokenizer.encode("CNCM runs fast on Android and laptop iGPUs.", add_special_tokens=True)[:16]],
        dtype=torch.long,
    )
    with torch.no_grad():
        t0 = time.perf_counter()
        fp32_logits = model.forward_exact(sample_ids)["logits"]
        fp32_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        int8_logits = model_int8.forward_exact(sample_ids)["logits"]
        int8_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        int4_logits = model_int4.forward_exact(sample_ids)["logits"]
        int4_ms = (time.perf_counter() - t0) * 1000.0

    int8_mae = float((fp32_logits - int8_logits).abs().mean().item())
    int4_mae = float((fp32_logits - int4_logits).abs().mean().item())
    int8_lin_ratio = fp32_sz["linear_bytes"] / max(1, int8_sz["linear_bytes"])
    int4_lin_ratio = fp32_sz["linear_bytes"] / max(1, int4_sz["linear_bytes"])

    print(
        f"{'Variant':<12} {'Total MB':>10} {'Linear MB':>11} {'Linear Ratio':>14} "
        f"{'Forward ms':>11} {'Logit MAE':>11}"
    )
    print("-" * 74)
    print(
        f"{'FP32':<12} {fp32_sz['total_mb']:>10.3f} {fp32_sz['linear_mb']:>11.3f} "
        f"{'1.00x':>14} {fp32_ms:>11.2f} {'0.0000':>11}"
    )
    print(
        f"{'INT8':<12} {int8_sz['total_mb']:>10.3f} {int8_sz['linear_mb']:>11.3f} "
        f"{int8_lin_ratio:>13.2f}x {int8_ms:>11.2f} {int8_mae:>11.5f}"
    )
    print(
        f"{'INT4 (g=32)':<12} {int4_sz['total_mb']:>10.3f} {int4_sz['linear_mb']:>11.3f} "
        f"{int4_lin_ratio:>13.2f}x {int4_ms:>11.2f} {int4_mae:>11.5f}"
    )

    # Verify ONNX export wrappers (and emit .onnx binaries if onnx is installed)
    if export_onnx_dir is not None:
        onnx_res = export_cncm_onnx(model, Path(export_onnx_dir), opset=17)
        print(f"\n[ONNX Export] {onnx_res['message']}")
        print(f"  Wrappers verified: {onnx_res['wrappers_verified']} | Manifest: {onnx_res['manifest']}")
    else:
        with tempfile.TemporaryDirectory(prefix="cncm_onnx_") as tmpdir:
            onnx_res = export_cncm_onnx(model, Path(tmpdir), opset=17)
            print(f"\n[ONNX Export] {onnx_res['message']}")
            print(f"  Wrappers verified: {onnx_res['wrappers_verified']}")
            for gname, ginfo in onnx_res["wrapper_shapes"].items():
                print(f"    - {gname}: in={ginfo['inputs']} -> out={ginfo['outputs']} (finite={ginfo['finite']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="CNCM End-to-End Demo CLI")
    parser.add_argument(
        "--section",
        choices=("all", "profile", "train", "agent", "fim", "quant"),
        default="all",
        help="Which demo section(s) to execute.",
    )
    parser.add_argument("--stage1-steps", type=int, default=6)
    parser.add_argument("--stage2-steps", type=int, default=6)
    parser.add_argument("--grpo-steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--export-onnx-dir", type=str, default=None)
    args = parser.parse_args()

    print("CNCM — Compact Causal-Hybrid Next-Concept Model (End-to-End Demo)")

    if args.section in ("all", "profile"):
        run_profile_section(instantiate_nano=True)

    tokenizer = None
    model = None
    if args.section in ("all", "train"):
        _, tokenizer, model = run_training_smoke_section(
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            grpo_steps=args.grpo_steps,
            device=args.device,
        )
    elif args.section in ("agent", "fim", "quant"):
        cfg = get_preset("tiny_test")
        tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
        model = CNCMModel(cfg, CNCMConceptTokenizer(cfg))

    if args.section in ("all", "agent"):
        assert tokenizer is not None and model is not None
        run_agent_section(tokenizer, model)

    if args.section in ("all", "fim"):
        assert tokenizer is not None and model is not None
        run_pg_fim_section(tokenizer, model)

    if args.section in ("all", "quant"):
        assert tokenizer is not None and model is not None
        run_quantization_section(tokenizer, model, export_onnx_dir=args.export_onnx_dir)

    print("\nCNCM demo completed successfully.")


if __name__ == "__main__":
    main()
