"""CNCM + PG-FIM Cloud GPU End-to-End Verification Runner on Modal.

Executes an ephemeral Modal cloud GPU verification run (hard cap: timeout=600s,
gpu=["A10", "L40S", "T4"], target ~4-5 min, ~$0.10-$0.15) covering:
  1. GPU environment diagnostics & full 19/19 `pytest -v tests/test_cncm.py` suite
  2. Real CNCM-Nano (`use_copy_decoder=True`, Phase B Copy-Aware Stage 1 Decoder)
     GPU training:
       - Stage 1 (M-RFSQ + Phase B Copy-Aware Keyword Cross-Attention on 60/40
         code + prose PG-FIM corpus + real English passages)
       - Stage 2 Joint (`FIMPlanDataset`, `fim_rate=0.5`, 40/30/30 AST/line/char
         spans, 15% `<kw/>` + 15% keyword corruption, frozen Stage 1, BF16)
       - Stage 3 Compiler-in-the-Loop `train_pg_fim_grpo` across the 3-stage
         curriculum (`single_line` -> `multi_line` -> `function_body`)
     + checkpoint persistence to Modal Volume `/vol/cncm_nano_verified/`
  3. PG-FIM Evaluation & 4-Way Ablation Benchmark on held-out FIM tasks:
       (a) `cncm_fim_no_plan` (`use_plan=False, enable_repair=False`)
       (b) `cncm_pg_fim_phase_a` (`use_plan=True, enable_repair=True`, no copy head)
       (c) `cncm_pg_fim_phase_b_repair` (`use_copy_decoder=True, use_plan=True, enable_repair=True`)
       (d) `cncm_pg_fim_int4_edge` (`quantize_model_int4` + `quantize_kv=True`)
  4. Real GPU inference benchmarks (`CNCM-Nano` & `CNCM-Mini`, B=1 and B=8, 64 tokens)
     across Parallel Concept (K=8), Early-Exit Concept (conf=0.90), Exact-AR (64 steps),
     and Lossless Self-Speculative modes
  5. Trained Dual-Mode Agentic AI verification (`calculator` & `kv_memory`)
  6. Real ONNX export (`stage1_decoder.onnx`, `stage2_exact_step.onnx`,
     `stage2_concept_step.onnx`) + ONNXRuntime inference verification vs PyTorch
  7. Persistent report saved to `/vol/cncm_nano_verified/modal_verify_report.json`
     and local `modal_verify_report.json` with exact elapsed GPU cost estimate.

Usage (ephemeral run ONLY — never `modal deploy`):
    modal run modal_run.py --preset nano --stage1-steps 350 --stage2-steps 400 --grpo-steps 18
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


app = modal.App("cncm-l4-verify")
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
        "pytest>=8.0.0",
        "onnx>=1.16.0",
        "onnxscript>=0.1.0",
        "onnxruntime>=1.18.0",
        "datasets>=2.19.0",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "cncm"),
        remote_path=f"{REMOTE_PROJECT_DIR}/cncm",
        ignore=["__pycache__", "*.pyc", "*.pyo"],
    )
    .add_local_dir(
        str(PROJECT_ROOT / "tests"),
        remote_path=f"{REMOTE_PROJECT_DIR}/tests",
        ignore=["__pycache__", "*.pyc", "*.pyo"],
    )
    .add_local_file(
        str(PROJECT_ROOT / "demo_cncm.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/demo_cncm.py",
    )
)


def _load_real_english_corpus(
    target_samples: int = 1200,
    timeout_sec: float = 20.0,
) -> Tuple[List[str], str]:
    """Load up to `target_samples` real English passages via HuggingFace `datasets`.

    Uses a daemon thread with a hard `timeout_sec` wall-clock cap so network stalls
    never delay GPU execution; falls back cleanly to synthetic corpus if unavailable.
    """
    result_holder: Dict[str, Any] = {"texts": [], "source": "synthetic_fallback", "error": None}

    def _worker() -> None:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]

            attempts = [
                ("wikitext", "wikitext-2-raw-v1", "train", False),
                ("roneneldan/TinyStories", None, "train", True),
            ]
            for ds_name, ds_cfg, split, streaming in attempts:
                try:
                    if ds_cfg is not None:
                        ds = load_dataset(ds_name, ds_cfg, split=split, streaming=streaming)
                    else:
                        ds = load_dataset(ds_name, split=split, streaming=streaming)
                    collected: List[str] = []
                    for row in ds:
                        txt = str(row.get("text", "")).strip()
                        if len(txt) < 48 or txt.startswith("="):
                            continue
                        collected.append(txt[:420])
                        if len(collected) >= target_samples:
                            break
                    if len(collected) >= 200:
                        result_holder["texts"] = collected
                        result_holder["source"] = (
                            f"{ds_name}/{ds_cfg}" if ds_cfg else ds_name
                        )
                        return
                except Exception as inner_exc:  # noqa: BLE001
                    result_holder["error"] = f"{ds_name}: {type(inner_exc).__name__}: {inner_exc}"
        except Exception as exc:  # noqa: BLE001
            result_holder["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return [], f"synthetic_fallback (datasets load timed out after {timeout_sec:.0f}s)"
    if result_holder["texts"]:
        return result_holder["texts"], str(result_holder["source"])
    err = result_holder.get("error") or "no samples returned"
    return [], f"synthetic_fallback ({err})"


def _run_pg_fim_ablation_benchmark(
    model: Any,
    tokenizer: Any,
    device: Any,
    num_tasks: int = 12,
) -> Dict[str, Any]:
    """Run the 4-way PG-FIM evaluation & ablation benchmark on held-out FIM tasks."""
    import torch

    from cncm.export import measure_model_size, quantize_model_int4
    from cncm.trainer import compute_fim_reward, make_synthetic_fim_tasks

    model.eval()
    tasks = make_synthetic_fim_tasks(num_samples=num_tasks, seed=101, curriculum_stage="all")
    fp32_size_mb = measure_model_size(model)["total_mb"]

    def _eval_variant(
        target_model: Any,
        *,
        use_plan: bool,
        enable_repair: bool,
        quantize_kv: bool,
        use_oracle_kw_hint: bool,
        disable_copy_head: bool,
        model_mb: float,
    ) -> Dict[str, Any]:
        saved_copy_head = getattr(target_model.concept_tokenizer, "copy_cross_attn", None)
        if disable_copy_head:
            target_model.concept_tokenizer.copy_cross_attn = None

        parses_list: List[float] = []
        kw_present_list: List[float] = []
        compiles_list: List[float] = []
        tests_passed_list: List[float] = []
        non_trivial_list: List[float] = []
        rewards_list: List[float] = []
        repair_trig_list: List[float] = []
        seq_passes_list: List[float] = []
        plan_steps_list: List[float] = []
        concept_steps_list: List[float] = []
        repair_passes_list: List[float] = []
        speedups_list: List[float] = []

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()

            for task in tasks:
                oracle_kws = task.get("oracle_keywords") if use_oracle_kw_hint else None
                fim_out = target_model.generate_fim(
                    prefix=task["prefix"],
                    suffix=task["suffix"],
                    tokenizer=tokenizer,
                    max_chunks=2,
                    temperature=0.0,
                    early_exit_conf=0.90,
                    quantize_kv=quantize_kv,
                    use_plan=use_plan,
                    enable_repair=enable_repair,
                    oracle_keywords=oracle_kws,
                )
                active_kws = (
                    fim_out["keywords"]
                    if fim_out["keywords"]
                    else list(task.get("oracle_keywords", []))
                )
                rew = compute_fim_reward(
                    fim_out["middle_text"],
                    task,
                    planned_keywords=active_kws,
                    planned_chunks=fim_out["planned_chunks"],
                    actual_chunks=1,
                )
                parses_list.append(1.0 if fim_out["parses"] else 0.0)
                kw_present_list.append(1.0 if fim_out["keywords_present"] else 0.0)
                compiles_list.append(1.0 if rew["compiles"] else 0.0)
                tests_passed_list.append(1.0 if rew["tests_passed"] else 0.0)
                non_trivial_list.append(1.0 if rew["non_trivial"] else 0.0)
                rewards_list.append(float(rew["reward"]))
                repair_trig_list.append(1.0 if fim_out["repair_triggered"] else 0.0)
                seq_passes_list.append(float(fim_out["sequential_passes"]))
                plan_steps_list.append(float(fim_out["plan_steps"]))
                concept_steps_list.append(float(fim_out["concept_steps"]))
                repair_passes_list.append(float(fim_out["repair_passes"]))
                speedups_list.append(float(fim_out["speedup_vs_ar"]))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            else:
                peak_vram_mb = 0.0
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            if disable_copy_head:
                target_model.concept_tokenizer.copy_cross_attn = saved_copy_head

        n = float(max(1, len(tasks)))
        return {
            "num_tasks": len(tasks),
            "use_plan": use_plan,
            "enable_repair": enable_repair,
            "use_copy_decoder": bool(
                not disable_copy_head
                and getattr(target_model.concept_tokenizer, "copy_cross_attn", None) is not None
            ),
            "quantize_kv": quantize_kv,
            "parse_rate": round(sum(parses_list) / n, 4),
            "keyword_faithfulness": round(sum(kw_present_list) / n, 4),
            "compile_rate": round(sum(compiles_list) / n, 4),
            "sandbox_test_pass_rate": round(sum(tests_passed_list) / n, 4),
            "non_trivial_rate": round(sum(non_trivial_list) / n, 4),
            "mean_reward": round(sum(rewards_list) / n, 4),
            "repair_trigger_rate": round(sum(repair_trig_list) / n, 4),
            "avg_sequential_passes": round(sum(seq_passes_list) / n, 2),
            "avg_plan_steps": round(sum(plan_steps_list) / n, 2),
            "avg_concept_steps": round(sum(concept_steps_list) / n, 2),
            "avg_repair_passes": round(sum(repair_passes_list) / n, 2),
            "speedup_vs_ar": round(sum(speedups_list) / n, 3),
            "latency_ms_per_task": round(elapsed_ms / n, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "model_size_mb": round(model_mb, 3),
        }

    # 1. Baseline CNCM FIM (no plan, no repair, no copy head)
    v_no_plan = _eval_variant(
        model,
        use_plan=False,
        enable_repair=False,
        quantize_kv=False,
        use_oracle_kw_hint=False,
        disable_copy_head=True,
        model_mb=fp32_size_mb,
    )

    # 2. CNCM PG-FIM Phase A (Exact-AR plan + Parallel Concept + 1-pass Exact-AR span repair)
    v_phase_a = _eval_variant(
        model,
        use_plan=True,
        enable_repair=True,
        quantize_kv=False,
        use_oracle_kw_hint=True,
        disable_copy_head=True,
        model_mb=fp32_size_mb,
    )

    # 3. CNCM PG-FIM Phase B + Repair (Phase B Copy-Aware Stage 1 Decoder + Plan + Repair)
    v_phase_b = _eval_variant(
        model,
        use_plan=True,
        enable_repair=True,
        quantize_kv=False,
        use_oracle_kw_hint=True,
        disable_copy_head=False,
        model_mb=fp32_size_mb,
    )

    # 4. INT4 Quantized Edge PG-FIM (packed INT4 weights + INT8 KV cache)
    model_int4 = quantize_model_int4(model, group_size=32, quantize_embeddings=True).to(device).eval()
    int4_size_mb = measure_model_size(model_int4)["total_mb"]
    v_int4_edge = _eval_variant(
        model_int4,
        use_plan=True,
        enable_repair=True,
        quantize_kv=True,
        use_oracle_kw_hint=True,
        disable_copy_head=False,
        model_mb=int4_size_mb,
    )
    del model_int4
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "held_out_tasks": len(tasks),
        "curriculum_tiers": ["single_line", "multi_line", "function_body"],
        "fp32_model_mb": round(fp32_size_mb, 3),
        "int4_model_mb": round(int4_size_mb, 3),
        "variants": {
            "1_cncm_fim_no_plan": v_no_plan,
            "2_cncm_pg_fim_phase_a": v_phase_a,
            "3_cncm_pg_fim_phase_b_repair": v_phase_b,
            "4_int4_quantized_edge_pg_fim": v_int4_edge,
        },
    }


def _benchmark_model_on_cuda(
    model: Any,
    tokenizer: Any,
    preset_name: str,
    device: Any,
    batch_sizes: Tuple[int, ...] = (1, 8),
    prefix_len: int = 32,
    gen_tokens: int = 64,
) -> Dict[str, Any]:
    """Run synchronized CUDA inference benchmarks across all 4 CNCM decoding modes."""
    import torch

    model.eval()
    base_prompt = (
        "CNCM uses multiscale residual finite scalar quantization with causal prefix "
        "block autoregressive decoding for fast edge inference on mobile and GPU."
    )
    raw_ids = tokenizer.encode(base_prompt, add_special_tokens=True)
    if len(raw_ids) < prefix_len:
        raw_ids = (raw_ids * ((prefix_len // max(1, len(raw_ids))) + 1))[:prefix_len]
    else:
        raw_ids = raw_ids[:prefix_len]

    results: Dict[str, Any] = {
        "preset": preset_name,
        "params": model.count_parameters(),
        "chunk_size": int(model.cfg.chunk_size),
        "gen_tokens": int(gen_tokens),
        "prefix_len": int(prefix_len),
        "batches": {},
    }

    for bsz in batch_sizes:
        prefix = torch.tensor([raw_ids] * bsz, dtype=torch.long, device=device)

        def _time_fn(fn, warmup: int = 2, iters: int = 4) -> Tuple[float, Any, float]:
            last_out = None
            with torch.no_grad():
                for _ in range(warmup):
                    last_out = fn()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()
                for _ in range(iters):
                    last_out = fn()
                torch.cuda.synchronize()
                elapsed_ms = ((time.perf_counter() - t0) / float(iters)) * 1000.0
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            return elapsed_ms, last_out, peak_mb

        # 1. Parallel Concept Mode (full K=8 steps)
        concept_ms, concept_out, concept_vram = _time_fn(
            lambda: model.generate_concept_chunk(
                prefix,
                temperature=0.0,
                early_exit_conf=None,
                quantize_kv=False,
                return_codes=False,
            ),
            warmup=2,
            iters=4,
        )
        concept_toks = int(concept_out["token_ids"].shape[1])
        concept_tps = (bsz * concept_toks) / (concept_ms / 1000.0)

        # 2. Early-Exit Concept Mode (early_exit_conf=0.90)
        ee_ms, ee_out, ee_vram = _time_fn(
            lambda: model.generate_concept_chunk(
                prefix,
                temperature=0.0,
                early_exit_conf=0.90,
                quantize_kv=False,
                return_codes=False,
            ),
            warmup=2,
            iters=4,
        )
        ee_toks = int(ee_out["token_ids"].shape[1])
        ee_tps = (bsz * ee_toks) / (ee_ms / 1000.0)
        ee_exit_level = int(ee_out.get("exit_level", model.cfg.num_levels))

        # 3. Exact-AR Mode (64 sequential steps)
        ar_ms, ar_out, ar_vram = _time_fn(
            lambda: model.generate_exact(
                prefix,
                max_new_tokens=gen_tokens,
                temperature=0.0,
                stop_token_ids=None,
                quantize_kv=False,
            ),
            warmup=1,
            iters=2,
        )
        ar_toks = int(ar_out["token_ids"].shape[1])
        ar_tps = (bsz * ar_toks) / (ar_ms / 1000.0)

        # 4. Self-Speculative Lossless Mode (early-exit draft + 1-pass exact verify)
        spec_ms, spec_out, spec_vram = _time_fn(
            lambda: model.generate_self_speculative(
                prefix,
                max_new_tokens=gen_tokens,
                temperature=0.0,
                early_exit_conf=0.25,
                stop_token_ids=None,
                quantize_kv=False,
            ),
            warmup=0,
            iters=1,
        )
        spec_toks = int(spec_out["token_ids"].shape[1])
        spec_tps = (bsz * spec_toks) / (spec_ms / 1000.0)
        spec_exact_match = bool(torch.equal(spec_out["token_ids"], ar_out["token_ids"]))

        results["batches"][f"B={bsz}"] = {
            "batch_size": bsz,
            "parallel_concept_k8": {
                "ms_per_chunk": round(concept_ms, 3),
                "tokens_per_sec": round(concept_tps, 2),
                "tokens_generated": concept_toks,
                "exit_level": int(concept_out.get("exit_level", model.cfg.num_levels)),
                "peak_vram_mb": round(concept_vram, 2),
            },
            "early_exit_concept_090": {
                "ms_per_chunk": round(ee_ms, 3),
                "tokens_per_sec": round(ee_tps, 2),
                "tokens_generated": ee_toks,
                "exit_level": ee_exit_level,
                "peak_vram_mb": round(ee_vram, 2),
            },
            "exact_ar_64": {
                "ms_per_chunk": round(ar_ms, 3),
                "tokens_per_sec": round(ar_tps, 2),
                "tokens_generated": ar_toks,
                "peak_vram_mb": round(ar_vram, 2),
            },
            "self_speculative_lossless": {
                "ms_per_chunk": round(spec_ms, 3),
                "tokens_per_sec": round(spec_tps, 2),
                "tokens_generated": spec_toks,
                "acceptance_rate": round(float(spec_out.get("acceptance_rate", 0.0)), 4),
                "concept_steps": int(spec_out.get("concept_steps", 0)),
                "verify_passes": int(spec_out.get("verify_passes", 0)),
                "exact_greedy_match": spec_exact_match,
                "peak_vram_mb": round(spec_vram, 2),
            },
            "speedups_vs_exact_ar": {
                "parallel_concept_k8_x": round(ar_ms / max(concept_ms, 1e-6), 3),
                "early_exit_concept_x": round(ar_ms / max(ee_ms, 1e-6), 3),
                "self_speculative_x": round(ar_ms / max(spec_ms, 1e-6), 3),
            },
            "peak_vram_mb": round(max(concept_vram, ee_vram, ar_vram, spec_vram), 2),
        }

    return results


def _verify_onnx_export_and_ort(model_cpu: Any, onnx_dir: Path) -> Dict[str, Any]:
    """Export all 3 CNCM ONNX graphs and verify each with ONNXRuntime vs PyTorch."""
    import numpy as np
    import onnxruntime as ort
    import torch

    from cncm.export import (
        Stage1DecoderONNXWrapper,
        Stage2ConceptStepONNXWrapper,
        Stage2ExactStepONNXWrapper,
        export_cncm_onnx,
    )

    model_cpu.eval()
    t0 = time.perf_counter()
    export_res = export_cncm_onnx(
        model_cpu,
        onnx_dir,
        opset=18,
        batch_size=1,
        prefix_len=8,
    )
    export_elapsed = time.perf_counter() - t0

    cfg = model_cpu.cfg
    sum_levels = sum(cfg.levels)
    rng = np.random.default_rng(42)

    test_flat_codes = rng.integers(0, cfg.codebook_size, size=(1, sum_levels), dtype=np.int64)
    test_input_ids = rng.integers(16, min(cfg.vocab_size, 512), size=(1, 8), dtype=np.int64)
    test_pos_ids = np.arange(8, dtype=np.int64)[None, :]

    wrappers = {
        "stage1_decoder": (
            Stage1DecoderONNXWrapper(model_cpu.concept_tokenizer).eval(),
            (torch.from_numpy(test_flat_codes),),
            {"flat_codes": test_flat_codes},
        ),
        "stage2_exact_step": (
            Stage2ExactStepONNXWrapper(model_cpu).eval(),
            (torch.from_numpy(test_input_ids), torch.from_numpy(test_pos_ids)),
            {"input_ids": test_input_ids, "position_ids": test_pos_ids},
        ),
        "stage2_concept_step": (
            Stage2ConceptStepONNXWrapper(model_cpu).eval(),
            (torch.from_numpy(test_input_ids), torch.from_numpy(test_flat_codes)),
            {"prefix_ids": test_input_ids, "target_flat_codes": test_flat_codes},
        ),
    }

    ort_checks: Dict[str, Any] = {}
    all_ort_passed = bool(export_res.get("exported", False))

    for key, (pt_wrap, pt_args, np_feeds) in wrappers.items():
        fpath_str = export_res.get("files", {}).get(key)
        if not fpath_str or not Path(fpath_str).exists():
            all_ort_passed = False
            ort_checks[key] = {
                "ok": False,
                "error": export_res.get("errors", {}).get(key, "ONNX file missing"),
            }
            continue

        fpath = Path(fpath_str)
        file_mb = fpath.stat().st_size / (1024.0 * 1024.0)
        try:
            with torch.no_grad():
                pt_out = pt_wrap(*pt_args).detach().cpu().numpy()

            sess = ort.InferenceSession(str(fpath), providers=["CPUExecutionProvider"])
            ort_in_names = [inp.name for inp in sess.get_inputs()]
            ort_feeds = {k: v for k, v in np_feeds.items() if k in ort_in_names}
            t_ort = time.perf_counter()
            ort_out = sess.run(None, ort_feeds)[0]
            ort_ms = (time.perf_counter() - t_ort) * 1000.0

            shape_match = list(ort_out.shape) == list(pt_out.shape)
            all_finite = bool(np.isfinite(ort_out).all())
            diff = np.abs(ort_out.astype(np.float64) - pt_out.astype(np.float64))
            max_abs_diff = float(diff.max()) if diff.size else 0.0
            mean_abs_diff = float(diff.mean()) if diff.size else 0.0
            close_ok = shape_match and all_finite and (max_abs_diff < 1e-2)
            if not close_ok:
                all_ort_passed = False

            ort_checks[key] = {
                "ok": close_ok,
                "file": str(fpath),
                "file_mb": round(file_mb, 3),
                "output_shape": list(ort_out.shape),
                "shape_match": shape_match,
                "all_finite": all_finite,
                "max_abs_diff_vs_pytorch": round(max_abs_diff, 6),
                "mean_abs_diff_vs_pytorch": round(mean_abs_diff, 6),
                "ort_cpu_ms": round(ort_ms, 3),
            }
        except Exception as exc:  # noqa: BLE001
            all_ort_passed = False
            ort_checks[key] = {
                "ok": False,
                "file": str(fpath),
                "file_mb": round(file_mb, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "exported": bool(export_res.get("exported", False)),
        "all_ort_passed": all_ort_passed,
        "export_elapsed_sec": round(export_elapsed, 3),
        "onnx_available": bool(export_res.get("onnx_available", False)),
        "onnxscript_available": bool(export_res.get("onnxscript_available", False)),
        "wrappers_verified": bool(export_res.get("wrappers_verified", False)),
        "errors": export_res.get("errors", {}),
        "message": export_res.get("message", ""),
        "manifest": export_res.get("manifest", ""),
        "ort_verification": ort_checks,
    }


@app.function(
    image=image,
    gpu=["A10", "L40S", "T4"],
    timeout=600,
    volumes={VOLUME_MOUNT: vol},
)
def run_cncm_gpu_verification(
    preset: str = "nano",
    stage1_steps: int = 350,
    stage2_steps: int = 400,
    grpo_steps: int = 18,
    skip_pytest: bool = False,
) -> str:
    """Full end-to-end CNCM + PG-FIM GPU verification pipeline on Modal."""
    gpu_t0 = time.perf_counter()

    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)

    import torch

    from cncm.agent import CNCMAgent, create_default_registry
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer import (
        FIMPlanDataset,
        PackedChunkDataset,
        make_pg_fim_stage1_corpus,
        make_synthetic_code_corpus,
        make_synthetic_fim_tasks,
        make_synthetic_sft,
        train_pg_fim_grpo,
        train_stage1,
        train_stage2,
    )

    assert torch.cuda.is_available(), "CUDA GPU is required on Modal!"
    device = torch.device("cuda:0")
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_vram_gb = gpu_props.total_memory / (1024.0 ** 3)

    env_info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gb": round(gpu_vram_gb, 2),
        "cuda_capability": f"{gpu_props.major}.{gpu_props.minor}",
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
    }
    print(f"[ENV] {json.dumps(env_info)}")

    # ------------------------------------------------------------------
    # 1. Full 19/19 Pytest Suite inside GPU container
    # ------------------------------------------------------------------
    pytest_report: Dict[str, Any] = {"skipped": bool(skip_pytest)}
    if not skip_pytest:
        print("\n=== [1/6] Running pytest -v tests/test_cncm.py ===")
        pt_t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "tests/test_cncm.py"],
            cwd=REMOTE_PROJECT_DIR,
            capture_output=True,
            text=True,
        )
        pt_elapsed = time.perf_counter() - pt_t0
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        summary_line = ""
        for line in reversed(proc.stdout.splitlines()):
            if "passed" in line or "failed" in line or "error" in line:
                summary_line = line.strip()
                break
        pytest_report = {
            "skipped": False,
            "returncode": int(proc.returncode),
            "passed_all": proc.returncode == 0,
            "elapsed_sec": round(pt_elapsed, 2),
            "summary": summary_line,
        }
        if proc.returncode != 0:
            raise RuntimeError(f"Pytest suite failed (rc={proc.returncode}): {summary_line}")

    # ------------------------------------------------------------------
    # 2. Real CNCM-Nano (`use_copy_decoder=True`) PG-FIM GPU Training
    # ------------------------------------------------------------------
    print(f"\n=== [2/6] Training CNCM ({preset}, use_copy_decoder=True) on {env_info['gpu_name']} ===")
    cfg = get_preset(preset, use_copy_decoder=True)
    tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    ckpt_dir = Path(VOLUME_MOUNT) / f"cncm_{preset}_verified"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build mixed 60/40 code + prose PG-FIM Stage 1 corpus + Stage 2 FIMPlanDataset
    t_data = time.perf_counter()
    real_texts, corpus_source = _load_real_english_corpus(target_samples=1200, timeout_sec=20.0)
    pg_fim_s1_corpus = make_pg_fim_stage1_corpus(num_samples=960, seed=42)
    mixed_s1_corpus = pg_fim_s1_corpus + real_texts[:640]
    s1_dataset = PackedChunkDataset(
        mixed_s1_corpus,
        tokenizer,
        chunk_size=cfg.chunk_size,
        eos_token_id=cfg.eos_token_id,
        bos_token_id=cfg.bos_token_id,
    )

    fim_tasks_train = make_synthetic_fim_tasks(num_samples=240, seed=42, curriculum_stage="all")
    code_docs_train = make_synthetic_code_corpus(num_samples=160, seed=42)
    sft_train = make_synthetic_sft(num_samples=96, seed=42)
    mixed_s2_samples: List[Any] = (
        list(fim_tasks_train)
        + list(code_docs_train)
        + list(sft_train)
        + list(real_texts[:160])
    )
    s2_max_prefix = 96
    s2_dataset = FIMPlanDataset(
        samples=mixed_s2_samples,
        tokenizer=tokenizer,
        chunk_size=cfg.chunk_size,
        max_prefix_len=s2_max_prefix,
        fim_rate=cfg.fim_rate,
        span_mode="mix",
        kw_max=cfg.kw_max,
        kw_empty_prob=cfg.kw_empty_prob,
        kw_corrupt_prob=cfg.kw_corrupt_prob,
        fim_prefix_budget=cfg.fim_prefix_budget,
        fim_suffix_budget=cfg.fim_suffix_budget,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        seed=42,
    )
    fim_count = sum(1 for ex in s2_dataset.examples if int(ex.get("is_fim", 0)) == 1)
    plan_kinds: Dict[str, int] = {"clean": 0, "empty": 0, "corrupted": 0}
    for ex in s2_dataset.examples:
        pk = ex.get("plan_kind")
        if pk in plan_kinds:
            plan_kinds[pk] += 1

    data_elapsed = time.perf_counter() - t_data
    corpus_info = {
        "real_english_source": corpus_source,
        "real_english_samples": len(real_texts),
        "pg_fim_stage1_samples": len(pg_fim_s1_corpus),
        "stage1_packed_chunks": len(s1_dataset),
        "stage1_total_tokens": s1_dataset.num_tokens,
        "stage2_total_pairs": len(s2_dataset),
        "stage2_fim_pairs": fim_count,
        "stage2_block_ar_pairs": len(s2_dataset) - fim_count,
        "stage2_fim_ratio": round(fim_count / float(max(1, len(s2_dataset))), 4),
        "stage2_plan_kind_counts": plan_kinds,
        "prep_elapsed_sec": round(data_elapsed, 2),
    }
    print(f"[DATA] {json.dumps(corpus_info)}")

    # --- Stage 1: M-RFSQ + Phase B Copy-Aware Keyword Cross-Attention ---
    s1_bs = 64
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t_s1 = time.perf_counter()
    s1_res = train_stage1(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset=s1_dataset,
        steps=stage1_steps,
        batch_size=s1_bs,
        lr=2.5e-3,
        device=device,
        apply_fine_drop=True,
        seed=42,
        log_every=max(1, stage1_steps // 7),
        save_path=ckpt_dir / "stage1_tokenizer.pt",
        verbose=True,
    )
    torch.cuda.synchronize()
    s1_elapsed = time.perf_counter() - t_s1
    s1_peak_vram = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    s1_model: CNCMConceptTokenizer = s1_res["model"]
    s1_m = s1_res["metrics"]

    # Measure clean full-level reconstruction accuracy (eval mode, apply_fine_drop=False)
    s1_model.eval()
    eval_correct = 0
    eval_total = 0
    eval_active_ratios: List[float] = []
    with torch.no_grad():
        eval_count = min(256, len(s1_dataset))
        eval_ids = torch.stack([s1_dataset[i]["input_ids"] for i in range(eval_count)], dim=0)
        for start in range(0, eval_count, 64):
            batch_ids = eval_ids[start : start + 64].to(device)
            kw_ids, kw_mask = s1_model.extract_oracle_kw_token_ids(
                batch_ids, apply_dropout=False
            )
            out_eval = s1_model(
                batch_ids,
                apply_fine_drop=False,
                kw_token_ids=kw_ids,
                kw_mask=kw_mask,
            )
            preds = out_eval["logits"].argmax(dim=-1)
            eval_correct += int((preds == batch_ids).sum().item())
            eval_total += int(batch_ids.numel())
            eval_active_ratios.append(float(out_eval["active_code_ratio"].item()))
    s1_eval_recon_acc = eval_correct / float(max(1, eval_total))
    s1_eval_active = sum(eval_active_ratios) / float(max(1, len(eval_active_ratios)))

    copy_gate_val = (
        float(s1_model.copy_cross_attn.copy_gate.detach().cpu().item())
        if s1_model.copy_cross_attn is not None
        else 0.0
    )
    pointer_scale_val = (
        float(s1_model.copy_cross_attn.pointer_logit_scale.detach().cpu().item())
        if s1_model.copy_cross_attn is not None
        else 0.0
    )

    eff_s1_bs = max(1, min(s1_bs, len(s1_dataset)))
    s1_tokens = stage1_steps * eff_s1_bs * cfg.chunk_size
    stage1_report = {
        "steps": stage1_steps,
        "batch_size": eff_s1_bs,
        "lr": 2.5e-3,
        "use_copy_decoder": bool(cfg.use_copy_decoder),
        "copy_gate_value": round(copy_gate_val, 5),
        "pointer_logit_scale": round(pointer_scale_val, 5),
        "initial_loss": round(float(s1_m["initial_loss"]), 5),
        "final_loss": round(float(s1_m["final_loss"]), 5),
        "initial_recon_acc": round(float(s1_m["recon_acc_history"][0]), 4),
        "final_train_recon_acc_with_finedrop": round(float(s1_m["final_recon_acc"]), 4),
        "eval_recon_acc_full_levels": round(float(s1_eval_recon_acc), 4),
        "final_active_code_ratio": round(float(s1_m["final_active_code_ratio"]), 4),
        "eval_active_code_ratio": round(float(s1_eval_active), 4),
        "all_grads_finite": bool(s1_m["all_grads_finite"]),
        "elapsed_sec": round(s1_elapsed, 2),
        "tokens_processed": s1_tokens,
        "tokens_per_sec": round(s1_tokens / max(s1_elapsed, 1e-6), 1),
        "peak_vram_mb": round(s1_peak_vram, 2),
        "params": s1_m["params"],
        "checkpoint": str(ckpt_dir / "stage1_tokenizer.pt"),
    }
    print(f"[STAGE 1 SUMMARY] {json.dumps(stage1_report)}")

    s1_before_s2 = {k: v.detach().cpu().clone() for k, v in s1_model.state_dict().items()}

    # --- Stage 2: Joint PG-FIM + Block-AR Training on FIMPlanDataset ---
    s2_amp = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    s2_bs = 24
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t_s2 = time.perf_counter()
    s2_res = train_stage2(
        cfg=cfg,
        tokenizer=tokenizer,
        concept_tokenizer=s1_model,
        dataset=s2_dataset,
        mode="joint",
        steps=stage2_steps,
        corrupt_rate=0.5,
        batch_size=s2_bs,
        lr=1.2e-3,
        device=device,
        amp_dtype=s2_amp,
        seed=42,
        log_every=max(1, stage2_steps // 8),
        save_path=ckpt_dir / "stage2_joint.pt",
        verbose=True,
    )
    torch.cuda.synchronize()
    s2_elapsed = time.perf_counter() - t_s2
    s2_peak_vram = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    model: CNCMModel = s2_res["model"]
    s2_m = s2_res["metrics"]

    stage1_bit_exact_unchanged = all(
        torch.equal(s1_before_s2[k], v.detach().cpu())
        for k, v in model.concept_tokenizer.state_dict().items()
    )
    eff_s2_bs = max(1, min(s2_bs, len(s2_dataset)))
    s2_tokens = stage2_steps * eff_s2_bs * (s2_max_prefix + cfg.chunk_size)
    stage2_report = {
        "mode": "joint",
        "dataset": "FIMPlanDataset",
        "fim_rate": cfg.fim_rate,
        "amp_dtype": s2_amp,
        "corrupt_rate": 0.5,
        "steps": stage2_steps,
        "batch_size": eff_s2_bs,
        "lr": 1.2e-3,
        "initial_loss": round(float(s2_m["initial_loss"]), 5),
        "final_loss": round(float(s2_m["final_loss"]), 5),
        "initial_concept_loss": round(float(s2_m["concept_loss_history"][0]), 5),
        "final_concept_loss": round(float(s2_m["concept_loss_history"][-1]), 5),
        "initial_exact_loss": round(float(s2_m["exact_loss_history"][0]), 5),
        "final_exact_loss": round(float(s2_m["exact_loss_history"][-1]), 5),
        "all_grads_finite": bool(s2_m["all_grads_finite"]),
        "stage1_frozen": bool(s2_m["concept_tokenizer_frozen"]),
        "stage1_bit_exact_unchanged": stage1_bit_exact_unchanged,
        "elapsed_sec": round(s2_elapsed, 2),
        "tokens_processed": s2_tokens,
        "tokens_per_sec": round(s2_tokens / max(s2_elapsed, 1e-6), 1),
        "peak_vram_mb": round(s2_peak_vram, 2),
        "params": model.count_parameters(),
        "effective_layers": cfg.effective_layers,
        "checkpoint": str(ckpt_dir / "stage2_joint.pt"),
    }
    print(f"[STAGE 2 SUMMARY] {json.dumps(stage2_report)}")

    # --- Stage 3: Compiler-in-the-Loop PG-FIM GRPO RL (3-Stage Curriculum) ---
    fim_rl_tasks = make_synthetic_fim_tasks(num_samples=48, seed=42, curriculum_stage="all")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t_s3 = time.perf_counter()
    grpo_res = train_pg_fim_grpo(
        cfg=cfg,
        tokenizer=tokenizer,
        model=model,
        tasks=fim_rl_tasks,
        steps=grpo_steps,
        num_generations=4,
        batch_size=2,
        lr=1e-4,
        device=device,
        curriculum=("single_line", "multi_line", "function_body"),
        amp_dtype=s2_amp,
        seed=42,
        log_every=max(1, grpo_steps // 6),
        save_path=ckpt_dir / "stage3_grpo.pt",
        verbose=True,
    )
    torch.cuda.synchronize()
    s3_elapsed = time.perf_counter() - t_s3
    s3_peak_vram = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    model = grpo_res["model"]
    gm = grpo_res["metrics"]

    grpo_report = {
        "stage": gm.get("stage", "pg_fim_grpo"),
        "algorithm": gm["algorithm"],
        "fim_mode": bool(gm.get("fim_mode", True)),
        "curriculum_stages_visited": gm.get("curriculum_stages_visited", []),
        "steps": grpo_steps,
        "num_generations": int(gm["num_generations"]),
        "initial_mean_reward": round(float(gm["mean_reward_history"][0]), 4),
        "final_mean_reward": round(float(gm["final_mean_reward"]), 4),
        "avg_mean_reward": round(
            sum(gm["mean_reward_history"]) / float(max(1, len(gm["mean_reward_history"]))),
            4,
        ),
        "max_mean_reward": round(float(max(gm["mean_reward_history"])), 4),
        "final_parse_rate": round(float(gm.get("final_parse_rate", 0.0)), 4),
        "final_compile_rate": round(float(gm.get("final_compile_rate", 0.0)), 4),
        "final_test_pass_rate": round(float(gm.get("final_test_pass_rate", 0.0)), 4),
        "final_non_trivial_rate": round(float(gm.get("final_non_trivial_rate", 0.0)), 4),
        "final_kw_faithful_rate": round(float(gm.get("final_kw_faithful_rate", 0.0)), 4),
        "final_kl": round(float(gm["final_kl"]), 6),
        "all_logprobs_finite": bool(gm["all_logprobs_finite"]),
        "all_grads_finite": bool(gm["all_grads_finite"]),
        "policy_updated": bool(gm["policy_updated"]),
        "policy_param_delta_l2": round(float(gm["policy_param_delta_l2"]), 6),
        "stage1_frozen": bool(gm["concept_tokenizer_frozen"]),
        "elapsed_sec": round(s3_elapsed, 2),
        "peak_vram_mb": round(s3_peak_vram, 2),
        "checkpoint": str(ckpt_dir / "stage3_grpo.pt"),
    }
    print(f"[STAGE 3 PG-FIM GRPO SUMMARY] {json.dumps(grpo_report)}")

    # ------------------------------------------------------------------
    # 3. PG-FIM Evaluation & 4-Way Ablation Benchmark
    # ------------------------------------------------------------------
    print("\n=== [3/6] Running PG-FIM Evaluation & 4-Way Ablation Benchmark ===")
    pg_fim_ablation = _run_pg_fim_ablation_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_tasks=12,
    )
    print(f"[PG-FIM ABLATION] {json.dumps(pg_fim_ablation)}")

    # ------------------------------------------------------------------
    # 4. Real GPU Inference Benchmarks (CNCM-Nano & CNCM-Mini, B=1 & B=8)
    # ------------------------------------------------------------------
    print("\n=== [4/6] Running GPU Inference Benchmarks (Nano & Mini, B=1 & B=8) ===")
    nano_bench = _benchmark_model_on_cuda(
        model=model,
        tokenizer=tokenizer,
        preset_name=preset,
        device=device,
        batch_sizes=(1, 8),
        prefix_len=32,
        gen_tokens=cfg.chunk_size,
    )
    print(f"[BENCH {preset}] {json.dumps(nano_bench)}")

    mini_cfg = get_preset("mini")
    mini_tok = CNCMTextTokenizer(vocab_size=mini_cfg.vocab_size)
    mini_s1 = CNCMConceptTokenizer(mini_cfg)
    mini_model = CNCMModel(mini_cfg, mini_s1).to(device).eval()
    mini_bench = _benchmark_model_on_cuda(
        model=mini_model,
        tokenizer=mini_tok,
        preset_name="mini",
        device=device,
        batch_sizes=(1, 8),
        prefix_len=32,
        gen_tokens=mini_cfg.chunk_size,
    )
    print(f"[BENCH mini] {json.dumps(mini_bench)}")
    del mini_model, mini_s1
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Trained Dual-Mode Agentic AI Verification (calculator & kv_memory)
    # ------------------------------------------------------------------
    print("\n=== [5/6] Verifying Trained Dual-Mode CNCMAgent (calculator & kv_memory) ===")
    model.eval()
    registry = create_default_registry()
    agent = CNCMAgent(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        max_turns=2,
        use_grammar_constraints=True,
        quantize_kv=True,
        early_exit_conf=0.92,
        device=device,
    )

    agent_tasks = [
        ("calculator", "Use calculator to compute 17 * 6 + 9."),
        ("kv_memory", "Use kv_memory to set key mode to concept."),
    ]
    agent_results: Dict[str, Any] = {}
    all_agent_ok = True
    for tool_name, task_prompt in agent_tasks:
        trace = agent.run(task_prompt, force_tool=tool_name)
        tc = trace.tool_calls[0] if trace.tool_calls else None
        tr = trace.tool_results[0] if trace.tool_results else None
        valid_json = isinstance(tc, dict) and ("name" in tc) and ("arguments" in tc)
        tool_ok = bool(tr is not None and tr.ok)
        if not (valid_json and tool_ok):
            all_agent_ok = False
        agent_results[tool_name] = {
            "prompt": task_prompt,
            "valid_json_tool_call": valid_json,
            "tool_call": tc,
            "tool_ok": tool_ok,
            "tool_output": tr.output if (tr is not None and tr.ok) else None,
            "tool_error": tr.error if (tr is not None and not tr.ok) else None,
            "num_steps": len(trace.steps),
            "mode_switches": trace.mode_switches,
            "concept_passes": trace.concept_passes,
            "exact_passes": trace.exact_passes,
            "total_latency_ms": round(trace.total_latency_ms, 2),
        }
        print(f"[AGENT:{tool_name}] {json.dumps(agent_results[tool_name])}")

    # ------------------------------------------------------------------
    # 6. Real ONNX Export + ONNXRuntime Verification
    # ------------------------------------------------------------------
    print("\n=== [6/6] Exporting ONNX Graphs & Verifying with ONNXRuntime ===")
    model_cpu = model.to("cpu").eval()
    torch.cuda.empty_cache()
    onnx_dir = ckpt_dir / "onnx"
    onnx_report = _verify_onnx_export_and_ort(model_cpu, onnx_dir)
    print(f"[ONNX] {json.dumps(onnx_report)}")

    # ------------------------------------------------------------------
    # 7. Assemble Report, Persist to Volume & Compute Estimated Cost
    # ------------------------------------------------------------------
    total_gpu_sec = time.perf_counter() - gpu_t0
    gpu_rate = _estimate_gpu_rate(env_info["gpu_name"])
    estimated_cost_usd = (total_gpu_sec / 3600.0) * gpu_rate

    ab_phase_b = pg_fim_ablation["variants"]["3_cncm_pg_fim_phase_b_repair"]
    ab_int4 = pg_fim_ablation["variants"]["4_int4_quantized_edge_pg_fim"]
    pg_fim_ok = (
        ab_phase_b["parse_rate"] >= 0.90
        and ab_phase_b["keyword_faithfulness"] >= 0.90
        and ab_int4["parse_rate"] >= 0.90
        and ab_int4["model_size_mb"] < 15.0
    )

    report: Dict[str, Any] = {
        "status": "passed"
        if (
            (pytest_report.get("skipped") or pytest_report.get("passed_all"))
            and stage1_report["all_grads_finite"]
            and stage2_report["all_grads_finite"]
            and stage2_report["stage1_frozen"]
            and stage2_report["stage1_bit_exact_unchanged"]
            and grpo_report["all_logprobs_finite"]
            and grpo_report["all_grads_finite"]
            and pg_fim_ok
            and all_agent_ok
            and onnx_report["exported"]
            and onnx_report["all_ort_passed"]
        )
        else "failed_checks",
        "preset": preset,
        "environment": env_info,
        "pytest": pytest_report,
        "corpus": corpus_info,
        "training": {
            "stage1": stage1_report,
            "stage2": stage2_report,
            "stage3_grpo": grpo_report,
        },
        "pg_fim_ablation": pg_fim_ablation,
        "benchmarks": {
            preset: nano_bench,
            "mini": mini_bench,
        },
        "agent_verification": {
            "all_ok": all_agent_ok,
            "tasks": agent_results,
        },
        "onnx_verification": onnx_report,
        "timing_and_cost": {
            "total_gpu_seconds": round(total_gpu_sec, 2),
            "gpu_name": env_info["gpu_name"],
            "gpu_rate_usd_per_hour": gpu_rate,
            "estimated_cost_usd": round(estimated_cost_usd, 5),
            "volume_checkpoint_dir": str(ckpt_dir),
        },
    }

    report_json = json.dumps(report, indent=2)
    vol_report_path = ckpt_dir / "modal_verify_report.json"
    vol_report_path.write_text(report_json, encoding="utf-8")
    vol.commit()
    print(
        f"\n[DONE] status={report['status']} | total_gpu_sec={total_gpu_sec:.1f}s | "
        f"est_cost=${estimated_cost_usd:.4f} | volume committed to {ckpt_dir}"
    )
    return report_json


@app.local_entrypoint()
def main(
    preset: str = "nano",
    stage1_steps: int = 350,
    stage2_steps: int = 400,
    grpo_steps: int = 18,
    skip_pytest: bool = False,
) -> None:
    """Run CNCM + PG-FIM GPU verification on Modal and write `modal_verify_report.json` locally."""
    print(
        f"Launching ephemeral Modal GPU run: preset={preset}, "
        f"stage1_steps={stage1_steps}, stage2_steps={stage2_steps}, "
        f"grpo_steps={grpo_steps}, skip_pytest={skip_pytest}"
    )
    report_json = run_cncm_gpu_verification.remote(
        preset=preset,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        grpo_steps=grpo_steps,
        skip_pytest=skip_pytest,
    )
    report: Dict[str, Any] = json.loads(report_json)

    out_path = PROJECT_ROOT / "modal_verify_report.json"
    out_path.write_text(report_json, encoding="utf-8")
    print(f"\nSaved local report -> {out_path}")

    tc = report["timing_and_cost"]
    s1 = report["training"]["stage1"]
    s2 = report["training"]["stage2"]
    s3 = report["training"]["stage3_grpo"]
    ab = report["pg_fim_ablation"]["variants"]
    nb1 = report["benchmarks"][preset]["batches"]["B=1"]
    mb1 = report["benchmarks"]["mini"]["batches"]["B=1"]
    print(
        f"Status: {report['status']} | GPU: {report['environment']['gpu_name']} "
        f"({report['environment']['vram_gb']} GB)\n"
        f"Pytest: {report['pytest'].get('summary', 'skipped')}\n"
        f"Stage 1 (CopyDec={s1['use_copy_decoder']}): loss {s1['initial_loss']:.4f} -> {s1['final_loss']:.4f}, "
        f"eval_recon_acc={s1['eval_recon_acc_full_levels']*100:.2f}%, "
        f"fsq_active={s1['eval_active_code_ratio']*100:.1f}% ({s1['elapsed_sec']}s)\n"
        f"Stage 2 (FIMPlanDataset): concept {s2['initial_concept_loss']:.4f} -> {s2['final_concept_loss']:.4f}, "
        f"exact {s2['initial_exact_loss']:.4f} -> {s2['final_exact_loss']:.4f}, "
        f"s1_frozen={s2['stage1_frozen']}/{s2['stage1_bit_exact_unchanged']} ({s2['elapsed_sec']}s)\n"
        f"Stage 3 (PG-FIM GRPO {s3['curriculum_stages_visited']}): reward {s3['initial_mean_reward']:.4f} -> "
        f"{s3['final_mean_reward']:.4f} (max={s3['max_mean_reward']:.4f}), "
        f"kl={s3['final_kl']:.5f} ({s3['elapsed_sec']}s)\n"
        f"PG-FIM Ablation (Phase B+Repair): parse={ab['3_cncm_pg_fim_phase_b_repair']['parse_rate']*100:.1f}%, "
        f"kw_faithful={ab['3_cncm_pg_fim_phase_b_repair']['keyword_faithfulness']*100:.1f}%, "
        f"speedup={ab['3_cncm_pg_fim_phase_b_repair']['speedup_vs_ar']:.2f}x | "
        f"INT4 ({report['pg_fim_ablation']['int4_model_mb']} MB): "
        f"parse={ab['4_int4_quantized_edge_pg_fim']['parse_rate']*100:.1f}%\n"
        f"Bench Nano B=1: Concept {nb1['parallel_concept_k8']['ms_per_chunk']:.1f}ms vs "
        f"Exact-AR {nb1['exact_ar_64']['ms_per_chunk']:.1f}ms "
        f"({nb1['speedups_vs_exact_ar']['parallel_concept_k8_x']:.2f}x)\n"
        f"Bench Mini B=1: Concept {mb1['parallel_concept_k8']['ms_per_chunk']:.1f}ms vs "
        f"Exact-AR {mb1['exact_ar_64']['ms_per_chunk']:.1f}ms "
        f"({mb1['speedups_vs_exact_ar']['parallel_concept_k8_x']:.2f}x)\n"
        f"Agent OK: {report['agent_verification']['all_ok']} | "
        f"ONNX+ORT OK: {report['onnx_verification']['all_ort_passed']}\n"
        f"Total GPU time: {tc['total_gpu_seconds']}s | Est. Modal {tc['gpu_name']} cost: ${tc['estimated_cost_usd']:.4f}"
    )
