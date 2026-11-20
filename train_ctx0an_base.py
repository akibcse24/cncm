"""High-Power Multi-Session Training Pipeline for `ctx0an` (CNCM-Base, ~93.9M params, 24 eff. layers).

Designed for Kaggle 2x NVIDIA T4 (16GB x 2) DDP (`torchrun --nproc_per_node=2 train_ctx0an_base.py`),
strictly safe against Kaggle's session/weekly GPU quota timeouts and 19.5 GB `/kaggle/working` disk cap:

Architecture (`ctx0an` = CNCM `base` preset):
  - 93.9M total params (88.0M Stage 2 generator + 5.9M Stage 1 M-RFSQ tokenizer)
  - 12 unique Transformer layers x 2 recursive loops = 24 effective layers (rank-32 LoopAdapters)
  - Hidden=768, GQA (12Q / 3KV), SwiGLU FFN=2304, Max Context=4096, Vocab=6400 Byte-Level BPE
  - Strictly causal 1-layer Stage 1 semantic encoder (`enc_layers=1`) + Matryoshka cutoffs (4, 6, 7, 8)
  - Continuous Scaled RoPE + Dual-Mode Concept/Exact-AR + Text-Level Alignment Loss (`L_align`)

Curated "Textbooks + Code + CoT Math + Tool JSON" Mixture (Phi / SmolLM2 principle):
  - 45% High-educational textbooks & knowledge (`HuggingFaceTB/smollm-corpus` cosmopedia-v2 + fineweb-edu-dedup)
  - 30% Clean Python code & PG-FIM infilling (`HuggingFaceTB/smollm-corpus` python-edu + flytech/python-codes-25k)
  - 15% Step-by-step `<think>` math reasoning (`meta-math/MetaMathQA` + `openai/gsm8k`)
  - 10% Structured JSON & function calling (`glaiveai/glaive-function-calling-v2` / synthetic agentic schemas)

3-Stage Progressive Curriculum (All Resumable via Rolling `CheckpointManager`):
  1. Stage 1 (FP32-safe): Strictly Causal M-RFSQ Concept Tokenizer + Matryoshka cutoffs
  2. Stage 2 Phase 1 (FP16 + GradScaler, DDP): Pure Exact-AR Foundation Pretraining (all 576 tokens supervised)
  3. Stage 2 Phase 2 (FP16 + GradScaler, DDP): Joint Dual-Mode + `L_align` Concept Adaptation
  4. Stage 2 Phase 3 (FP16 + GradScaler, DDP): Curated Instruction / `<think>` CoT / Tool-Call SFT Annealing
  5. Rank-0 Readiness Gate Evaluation (Gates #1-#5) + Generation Samples + Structured JSON Report

Usage:
    # Dual-GPU Kaggle T4 x2 DDP (recommended):
    torchrun --nproc_per_node=2 train_ctx0an_base.py --max-wall-minutes 325

    # Single-GPU:
    python train_ctx0an_base.py --max-wall-minutes 325

    # Local CPU smoke test:
    python train_ctx0an_base.py --preset tiny_test --smoke --no-hf-download --device cpu --out-dir ./ctx0an_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from cncm.config import get_preset
from cncm.model import CNCMModel
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer import (
    BlockARSFTDataset,
    MemmapTokenDataset,
    cleanup_ddp,
    make_pg_fim_stage1_corpus,
    make_synthetic_code_corpus,
    make_synthetic_corpus,
    make_synthetic_sft,
    measure_code_entropy,
    measure_concept_vs_exact_gap,
    measure_per_domain_stage1_recon,
    measure_tokenizer_fertility,
    prepare_memmap_corpus,
    resolve_hf_token,
    select_device,
    set_seed,
    setup_ddp_if_needed,
    train_stage1,
    train_stage2,
)


def _default_output_root() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists() and os.access(kaggle_working, os.W_OK):
        return kaggle_working / "ctx0an_out"
    return Path.cwd() / "ctx0an_out"


def _select_stage2_amp_dtype(device: torch.device, requested: str = "auto") -> Optional[str]:
    req = str(requested).strip().lower()
    if req == "fp32":
        return None
    if req in ("fp16", "bf16"):
        return req
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(idx)
            if int(major) >= 8 and torch.cuda.is_bf16_supported():
                return "bf16"
        except Exception:
            pass
        return "fp16"  # T4 SM 7.5 native Tensor Cores + GradScaler (avoids slow emulated BF16!)
    return None


def _safe_round(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return round(f, 5) if np.isfinite(f) else None
    except Exception:
        return None


def _make_synthetic_json_docs(num_samples: int = 1024, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    tools = ("calculator", "kv_memory", "text_stats", "clock_utc", "code_search", "sql_query", "unit_convert")
    statuses = ("ok", "cached", "updated", "verified")
    docs: List[str] = []
    for i in range(max(1, int(num_samples))):
        tool = tools[i % len(tools)]
        a = rng.randint(1, 999)
        b = rng.randint(2, 99)
        expr = f"{a} * {b} + {a % 23}"
        ans = a * b + (a % 23)
        payload = {
            "id": f"ctx0an_{i:06d}",
            "tool_call": {
                "name": tool,
                "arguments": {
                    "expression": expr,
                    "key": f"slot_{i % 64}",
                    "limit": rng.randint(5, 50),
                    "strict": bool(i % 2 == 0),
                },
            },
            "tool_response": {
                "status": statuses[i % len(statuses)],
                "result": ans,
                "latency_ms": round(rng.uniform(0.3, 18.5), 2),
                "tags": ["ctx0an", "edge", tool],
            },
        }
        docs.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return docs


def _stream_hf_docs(
    candidates: Sequence[Tuple[str, Optional[str], str, Sequence[str]]],
    target_count: int,
    *,
    min_chars: int = 80,
    max_chars: int = 4096,
    formatter: Optional[Any] = None,
) -> Tuple[List[str], List[str]]:
    """Stream curated documents from HuggingFace datasets with deduplication & graceful fallback."""
    docs: List[str] = []
    seen_hashes: set[int] = set()
    used_sources: List[str] = []

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        return docs, used_sources

    for ds_name, ds_cfg, split, fields in candidates:
        if len(docs) >= target_count:
            break
        before = len(docs)
        try:
            ds = (
                load_dataset(ds_name, ds_cfg, split=split, streaming=True)
                if ds_cfg
                else load_dataset(ds_name, split=split, streaming=True)
            )
            for row in ds:
                if formatter is not None:
                    txt = str(formatter(row)).strip()
                else:
                    txt = ""
                    for f in fields:
                        v = row.get(f)
                        if v:
                            txt = str(v).strip()
                            break
                if len(txt) < min_chars:
                    continue
                clipped = txt[:max_chars]
                h = hash(clipped[:256])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                docs.append(clipped)
                if len(docs) >= target_count:
                    break
        except Exception as exc:
            print(f"[DATA WARNING] Could not stream {ds_name} ({ds_cfg}): {type(exc).__name__}: {exc}")
            continue
        added = len(docs) - before
        if added > 0:
            label = f"{ds_name}/{ds_cfg}" if ds_cfg else ds_name
            used_sources.append(f"{label}({added})")
            print(f"[DATA STREAM] Loaded {added:,} docs from {label}")

    return docs, used_sources


def _format_math_row(row: Dict[str, Any]) -> str:
    q = str(row.get("query") or row.get("question") or "").strip()
    r = str(row.get("response") or row.get("answer") or "").strip()
    if not q or not r:
        return ""
    # Extract final answer after #### or "The answer is:" if present
    final_ans = ""
    if "####" in r:
        parts = r.split("####", 1)
        reasoning, final_ans = parts[0].strip(), parts[1].strip()
    elif "The answer is:" in r:
        parts = r.rsplit("The answer is:", 1)
        reasoning, final_ans = parts[0].strip(), parts[1].strip()
    else:
        reasoning = r
    if final_ans:
        return f"Question: {q}\n<think>{reasoning}</think>\nAnswer: {final_ans}"
    return f"Question: {q}\n<think>{reasoning}</think>"


def _format_code_row(row: Dict[str, Any]) -> str:
    text = row.get("original_string") or row.get("whole_func_string") or row.get("code")
    if text:
        doc = str(row.get("docstring") or "").strip()
        code_str = str(text).strip()
        if doc and doc not in code_str:
            return f'"""{doc}"""\n{code_str}'
        return code_str
    inst = str(row.get("instruction") or row.get("prompt") or "").strip()
    out = str(row.get("output") or "").strip()
    if inst and out:
        return f"# Task: {inst}\n{out}"
    return out or str(row.get("text") or "")


def _format_tool_row(row: Dict[str, Any]) -> str:
    chat = row.get("chat")
    system = row.get("system")
    if chat:
        sys_str = str(system or "").strip()
        return f"{sys_str}\n{str(chat).strip()}" if sys_str else str(chat).strip()
    convs = row.get("conversations")
    if isinstance(convs, list):
        parts = [f"{c.get('from', 'user')}: {c.get('value', '')}" for c in convs if isinstance(c, dict)]
        return "\n".join(parts)
    return ""


def _load_ctx0an_curated_corpora(
    target_en: int,
    target_py: int,
    target_math: int,
    target_json: int,
    val_fraction: float = 0.03,
    allow_hf_download: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[Dict[str, Any]], Dict[str, List[str]]]:
    """Build curated Textbooks + Code + CoT Math + Tool JSON corpora with strict train/val splits."""
    sources: Dict[str, List[str]] = {"english": [], "python": [], "math": [], "json": []}
    en_docs: List[str] = []
    py_docs: List[str] = []
    math_docs: List[str] = []
    json_docs: List[str] = []

    if allow_hf_download:
        # 1. Textbooks & High-Educational Web Prose (SmolLM2 / FineWeb-Edu)
        n_cosmo = int(target_en * 0.55)
        cosmo_docs, cosmo_src = _stream_hf_docs(
            [
                ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", ("text",)),
            ],
            target_count=n_cosmo,
            min_chars=120,
            max_chars=4096,
        )
        en_docs.extend(cosmo_docs)
        sources["english"].extend(cosmo_src)

        rem_en = max(0, target_en - len(en_docs))
        if rem_en > 0:
            fw_docs, fw_src = _stream_hf_docs(
                [
                    ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train", ("text",)),
                    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", ("text",)),
                    ("roneneldan/TinyStories", None, "train", ("text",)),
                ],
                target_count=rem_en,
                min_chars=100,
                max_chars=4096,
            )
            en_docs.extend(fw_docs)
            sources["english"].extend(fw_src)

        # 2. Clean Educational Python & Algorithmic Code (all live-verified ungated direct-text repos)
        n_csn = int(target_py * 0.55)
        csn_docs, csn_src = _stream_hf_docs(
            [
                ("Nan-Do/code-search-net-python", "default", "train", ("original_string", "code")),
            ],
            target_count=n_csn,
            min_chars=80,
            max_chars=4096,
            formatter=_format_code_row,
        )
        py_docs.extend(csn_docs)
        sources["python"].extend(csn_src)

        rem_py = max(0, target_py - len(py_docs))
        if rem_py > 0:
            inst_code_docs, inst_code_src = _stream_hf_docs(
                [
                    ("TokenBender/code_instructions_122k_alpaca_style", "default", "train", ("output",)),
                    ("flytech/python-codes-25k", "default", "train", ("output",)),
                    ("iamtarun/python_code_instructions_18k_alpaca", "default", "train", ("output",)),
                ],
                target_count=rem_py,
                min_chars=60,
                max_chars=4096,
                formatter=_format_code_row,
            )
            py_docs.extend(inst_code_docs)
            sources["python"].extend(inst_code_src)

        # 3. High-Quality Math & Step-by-Step Reasoning (<think> CoT)
        n_finemath = int(target_math * 0.55)
        fm_docs, fm_src = _stream_hf_docs(
            [
                ("HuggingFaceTB/finemath", "finemath-4plus", "train", ("text",)),
            ],
            target_count=n_finemath,
            min_chars=100,
            max_chars=4096,
        )
        math_docs.extend(fm_docs)
        sources["math"].extend(fm_src)

        rem_math = max(0, target_math - len(math_docs))
        if rem_math > 0:
            cot_docs, cot_src = _stream_hf_docs(
                [
                    ("meta-math/MetaMathQA", "default", "train", ("response",)),
                    ("openai/gsm8k", "main", "train", ("answer",)),
                ],
                target_count=rem_math,
                min_chars=60,
                max_chars=3072,
                formatter=_format_math_row,
            )
            math_docs.extend(cot_docs)
            sources["math"].extend(cot_src)

        # 4. Structured Function Calling & JSON Tool-Use
        tool_docs, tool_src = _stream_hf_docs(
            [
                ("glaiveai/glaive-function-calling-v2", "default", "train", ("chat",)),
                ("NousResearch/hermes-function-calling-v1", "func_calling", "train", ("conversations",)),
            ],
            target_count=max(64, int(target_json * 0.70)),
            min_chars=80,
            max_chars=3072,
            formatter=_format_tool_row,
        )
        json_docs.extend(tool_docs)
        sources["json"].extend(tool_src)

    # Deterministic fallbacks & PG-FIM / JSON schema anchors
    if len(en_docs) < min(256, target_en):
        syn = make_synthetic_corpus(num_samples=max(256, target_en), seed=seed)
        en_docs.extend(syn)
        sources["english"].append(f"synthetic({len(syn)})")
    if len(py_docs) < min(256, target_py):
        syn_py = make_synthetic_code_corpus(num_samples=max(256, target_py), seed=seed)
        py_docs.extend(syn_py)
        sources["python"].append(f"synthetic_code({len(syn_py)})")

    # Always mix in PG-FIM sentinel-formatted code docs so Stage 1 & 2 master all 11 PG-FIM tokens
    pg_fim_docs = make_pg_fim_stage1_corpus(num_samples=max(128, min(4000, target_py // 5)), seed=seed)
    py_docs.extend(pg_fim_docs)
    sources["python"].append(f"pg_fim_mix({len(pg_fim_docs)})")

    if len(math_docs) < min(128, target_math):
        syn_math = make_synthetic_corpus(num_samples=max(128, target_math), seed=seed + 7)
        math_docs.extend(syn_math)
        sources["math"].append(f"synthetic_math({len(syn_math)})")

    rem_json = max(256, target_json - len(json_docs))
    syn_json = _make_synthetic_json_docs(num_samples=rem_json, seed=seed)
    json_docs.extend(syn_json)
    sources["json"].append(f"synthetic_json({len(syn_json)})")

    # Build SFT Phase 3 conversations (tool calling + CoT math + reasoning)
    sft_convs: List[Dict[str, Any]] = make_synthetic_sft(num_samples=min(2048, max(128, target_json // 2)), seed=seed)
    for m_doc in math_docs[: min(4000, len(math_docs))]:
        if "Question:" in m_doc and "<think>" in m_doc:
            q_part, think_rest = m_doc.split("<think>", 1)
            q_clean = q_part.replace("Question:", "").strip()
            a_clean = "<think>" + think_rest.strip()
            if q_clean and a_clean:
                sft_convs.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are ctx0an, a compact reasoning and coding assistant. Think step-by-step inside <think>...</think>.",
                            },
                            {"role": "user", "content": q_clean},
                            {"role": "assistant", "content": a_clean},
                        ],
                        "category": "math_cot",
                    }
                )

    rng = random.Random(seed)
    train_domains: Dict[str, List[str]] = {}
    val_domains: Dict[str, List[str]] = {}

    for name, docs in (
        ("english", en_docs),
        ("python", py_docs),
        ("math", math_docs),
        ("json", json_docs),
    ):
        shuffled = list(docs)
        rng.shuffle(shuffled)
        n_val = max(32, min(1000, max(16, int(len(shuffled) * val_fraction))))
        if n_val >= len(shuffled):
            n_val = max(1, len(shuffled) // 5)
        val_domains[name] = shuffled[:n_val]
        train_domains[name] = shuffled[n_val:] if len(shuffled) > n_val else shuffled

    return train_domains, val_domains, sft_convs, sources


def _generate_qualitative_samples(
    model: CNCMModel,
    tokenizer: CNCMTextTokenizer,
    device: torch.device,
) -> Dict[str, str]:
    """Run quick Exact-AR & Concept-Mode sanity generations on `ctx0an`."""
    model.eval()
    prompts = {
        "english_exact": "Explain why recursive Transformer layers help compact models:",
        "python_exact": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n",
        "math_exact": "Question: Compute 18 * 7 + 14.\n<think>",
    }
    samples: Dict[str, str] = {}
    with torch.no_grad():
        for name, prompt in prompts.items():
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids:
                ids = [model.cfg.bos_token_id]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate_exact(x, max_new_tokens=48, temperature=0.0)
            gen_ids = out["token_ids"][0].detach().cpu().tolist()
            samples[name] = prompt + tokenizer.decode(gen_ids, skip_special_tokens=False)

        # Also sample 1 Concept-Mode chunk (64 tokens in 8 parallel passes)
        c_prompt = "def sum_even_numbers(nums):\n    total = 0\n"
        c_ids = tokenizer.encode(c_prompt, add_special_tokens=False) or [model.cfg.bos_token_id]
        cx = torch.tensor([c_ids], dtype=torch.long, device=device)
        c_out = model.generate_concept_chunk(cx, temperature=0.0)
        tok_ids = c_out["token_ids"][0].detach().cpu().tolist()
        samples["python_concept_chunk"] = c_prompt + tokenizer.decode(tok_ids, skip_special_tokens=False)
    return samples


def _cleanup_completed_phase_dir(phase_dir: Path, keep_weights_only: bool = True) -> None:
    """Free disk space after a phase finishes cleanly (critical for 221M MoE under Kaggle's 19.5 GB cap).

    Removes rolling `step_*.pt` duplicates and strips bulky AdamW optimizer tensors from the
    completed phase's `latest.pt` while keeping `step`, `model_state_dict`, and `metrics` intact
    so subsequent resumed sessions instantly recognize the completed phase (`step >= target_steps`).
    """
    if not phase_dir.exists():
        return
    for p in phase_dir.glob("step_*.pt"):
        try:
            p.unlink()
        except OSError:
            pass
    best_p = phase_dir / "best.pt"
    latest_p = phase_dir / "latest.pt"
    if keep_weights_only and latest_p.exists():
        try:
            payload = torch.load(latest_p, map_location="cpu", weights_only=False)
            payload["optimizer_state_dict"] = None
            payload["scaler_state_dict"] = None
            tmp_p = phase_dir / ".latest_compact.tmp"
            torch.save(payload, tmp_p)
            os.replace(tmp_p, latest_p)
            if best_p.exists():
                best_p.unlink()
        except Exception:
            pass


def run_ctx0an_training(args: argparse.Namespace) -> Dict[str, Any]:
    wall_start = time.monotonic()
    t0_all = time.perf_counter()

    dist_info = setup_ddp_if_needed()
    is_distributed = bool(dist_info["is_distributed"])
    rank = int(dist_info["rank"])
    local_rank = int(dist_info["local_rank"])
    world_size = int(dist_info["world_size"])
    is_main = bool(dist_info["is_main"])

    memmap_datasets: List[MemmapTokenDataset] = []

    try:
        set_seed(args.seed + rank)
        if is_distributed and torch.cuda.is_available() and args.device in ("auto", "cuda"):
            dev = torch.device(f"cuda:{local_rank}")
        else:
            dev = select_device(args.device)
        stage2_amp_dtype = _select_stage2_amp_dtype(dev, requested=args.amp_dtype)

        out_root = Path(args.out_dir) if args.out_dir else _default_output_root()
        ckpt_dir = out_root / "checkpoints"
        data_dir = out_root / "memmap_data"
        report_path = Path(args.report_path) if args.report_path else (out_root / "ctx0an_report.json")
        if is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            report_path.parent.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            dist.barrier()

        preset_overrides: Dict[str, Any] = {
            "enc_layers": int(args.enc_layers),
            "use_copy_decoder": bool(args.use_copy_decoder),
        }
        if args.preset != "tiny_test":
            preset_overrides["matryoshka_cutoffs"] = (4, 6, 7, 8)
        else:
            preset_overrides["matryoshka_cutoffs"] = (3, 4, 5)
        if args.num_loops is not None:
            preset_overrides["num_loops"] = int(args.num_loops)

        cfg = get_preset(args.preset, **preset_overrides)
        auto_resume = not args.no_auto_resume
        if args.smoke:
            args.hf_repo_id = None
        elif args.hf_repo_id and resolve_hf_token() is None:
            if is_main:
                print(
                    "[HF SYNC] No HF_TOKEN in environment or Kaggle Secrets; saving all checkpoints "
                    "to local output directory (attach HF_TOKEN in Kaggle Secrets to enable live HF Hub sync)."
                )
            args.hf_repo_id = None
        eff_max_wall = (
            args.max_wall_minutes
            if (args.max_wall_minutes is not None and not args.smoke)
            else (args.max_wall_minutes if (args.max_wall_minutes is not None and args.max_wall_minutes <= 5.0) else None)
        )

        if is_main:
            moe_str = (
                f"MoE({cfg.num_experts} routed Top-{cfg.num_experts_per_tok} + shared, "
                f"d_exp={cfg.moe_expert_intermediate_size}, aux={cfg.moe_aux_loss_weight})"
                if cfg.num_experts > 0
                else "Dense SwiGLU"
            )
            print(
                f"================ [ctx0an Training Init] ================\n"
                f"  model_name       : {args.model_name}\n"
                f"  preset           : {cfg.preset_name} (hidden={cfg.hidden_size}, "
                f"{cfg.num_unique_layers}x{cfg.num_loops}={cfg.effective_layers} eff layers, V={cfg.vocab_size})\n"
                f"  ffn_type         : {moe_str}\n"
                f"  device / DDP     : {dev} | world_size={world_size}\n"
                f"  precision        : Stage1=FP32, Stage2={stage2_amp_dtype or 'fp32'}\n"
                f"  grad_ckpt        : {args.grad_checkpointing} | prefix_len={args.prefix_len}\n"
                f"  wall_guard_min   : {eff_max_wall} | hf_repo={args.hf_repo_id or 'local-only'}\n"
                f"  out_root         : {out_root}\n"
                f"========================================================"
            )

        # 1 & 2. Rank 0 prepares curated corpora, trains/loads BPE tokenizer, and writes uint16 memmaps
        bpe_path = ckpt_dir / f"{args.model_name}_bpe_tokenizer.json"
        train_bin = data_dir / f"{args.model_name}_train.bin"
        val_bin = data_dir / f"{args.model_name}_val.bin"
        sft_jsonl = data_dir / f"{args.model_name}_sft.jsonl"
        meta_path = data_dir / f"{args.model_name}_corpus_meta.json"

        bpe_trained = False
        val_domains_eval: Dict[str, List[str]] = {}
        corpus_sources: Dict[str, List[str]] = {}
        sft_convs: List[Dict[str, Any]] = []

        if is_main:
            if args.smoke:
                t_en, t_py, t_math, t_json = 96, 96, 64, 64
            else:
                t_en, t_py, t_math, t_json = args.en_docs, args.py_docs, args.math_docs, args.json_docs

            can_reuse = (
                auto_resume
                and train_bin.exists()
                and val_bin.exists()
                and meta_path.exists()
            )
            if can_reuse:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                val_domains_eval = meta["val_domains_eval"]
                corpus_sources = meta["corpus_sources"]
                bpe_trained = bool(meta.get("bpe_trained", False))
                if bpe_path.exists():
                    tokenizer = CNCMTextTokenizer.from_hf_json(bpe_path, vocab_size=cfg.vocab_size)
                else:
                    tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
                if sft_jsonl.exists():
                    with sft_jsonl.open("r", encoding="utf-8") as fh:
                        sft_convs = [json.loads(line) for line in fh if line.strip()]
                print(
                    f"[DATA RESUME] Reusing pre-tokenized memmaps in {data_dir} "
                    f"(train={train_bin.stat().st_size / 1e6:.1f} MB, val={val_bin.stat().st_size / 1e6:.1f} MB)"
                )
            else:
                train_domains, val_domains, sft_convs, corpus_sources = _load_ctx0an_curated_corpora(
                    target_en=t_en,
                    target_py=t_py,
                    target_math=t_math,
                    target_json=t_json,
                    val_fraction=0.03,
                    allow_hf_download=not args.no_hf_download,
                    seed=args.seed,
                )
                # Interleave all 4 curated domains deterministically before packing into uint16 memmap
                all_train_texts = (
                    train_domains["english"]
                    + train_domains["python"]
                    + train_domains["math"]
                    + train_domains["json"]
                )
                rng_shuf = random.Random(args.seed)
                rng_shuf.shuffle(all_train_texts)

                all_val_texts = (
                    val_domains["english"]
                    + val_domains["python"]
                    + val_domains["math"]
                    + val_domains["json"]
                )
                val_domains_eval = {
                    "english": val_domains["english"][:256],
                    "python": val_domains["python"][:256],
                    "json": val_domains["json"][:256],
                }
                print(
                    f"[DATA READY] sources={corpus_sources}\n"
                    f"  train_docs={len(all_train_texts):,} | val_docs={len(all_val_texts):,} | sft_convs={len(sft_convs):,}"
                )

                if auto_resume and bpe_path.exists():
                    tokenizer = CNCMTextTokenizer.from_hf_json(bpe_path, vocab_size=cfg.vocab_size)
                    bpe_trained = True
                else:
                    try:
                        bpe_sample = all_train_texts[: min(len(all_train_texts), 45000)]
                        tokenizer = CNCMTextTokenizer.train_bpe(
                            bpe_sample,
                            vocab_size=cfg.vocab_size,
                            save_path=bpe_path,
                            min_frequency=2,
                        )
                        bpe_trained = True
                        print(f"[TOKENIZER] Trained ctx0an Byte-Level BPE (V={cfg.vocab_size}) -> {bpe_path}")
                    except ImportError as exc:
                        print(f"[TOKENIZER WARNING] {exc}; falling back to built-in CNCMTextTokenizer.")
                        tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

                prepare_memmap_corpus(
                    all_train_texts,
                    tokenizer,
                    train_bin,
                    eos_token_id=cfg.eos_token_id,
                    bos_token_id=cfg.bos_token_id,
                    dtype=np.uint16,
                )
                prepare_memmap_corpus(
                    all_val_texts,
                    tokenizer,
                    val_bin,
                    eos_token_id=cfg.eos_token_id,
                    bos_token_id=cfg.bos_token_id,
                    dtype=np.uint16,
                )
                with sft_jsonl.open("w", encoding="utf-8") as fh:
                    for item in sft_convs:
                        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                meta_path.write_text(
                    json.dumps(
                        {
                            "corpus_sources": corpus_sources,
                            "val_domains_eval": val_domains_eval,
                            "bpe_trained": bpe_trained,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"[MEMMAP] Built uint16 memmaps: train={train_bin.stat().st_size / 1e6:.2f} MB "
                    f"({train_bin.stat().st_size // 2:,} tokens), "
                    f"val={val_bin.stat().st_size / 1e6:.2f} MB ({val_bin.stat().st_size // 2:,} tokens)"
                )

        if is_distributed:
            dist.barrier()

        if args.prepare_only:
            if is_main:
                print(
                    f"[PREPARE-ONLY DONE] BPE tokenizer + uint16 memmaps ready in {data_dir} "
                    f"({time.perf_counter() - t0_all:.1f}s)."
                )
            return {
                "model_name": args.model_name,
                "overall_status": "PREPARED",
                "elapsed_sec": round(time.perf_counter() - t0_all, 2),
            }

        if not is_main:
            if bpe_path.exists():
                tokenizer = CNCMTextTokenizer.from_hf_json(bpe_path, vocab_size=cfg.vocab_size)
                bpe_trained = True
            else:
                tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
            if sft_jsonl.exists():
                with sft_jsonl.open("r", encoding="utf-8") as fh:
                    sft_convs = [json.loads(line) for line in fh if line.strip()]

        gate4_fertility: Dict[str, Any] = {}
        if is_main:
            gate4_fertility = measure_tokenizer_fertility(tokenizer, val_domains_eval)
            print(
                f"[GATE #4 BPE FERTILITY] passed={gate4_fertility['gate_all_passed']} | "
                f"{json.dumps(gate4_fertility['domains'])}"
            )

        # 3. Stage 1: Strictly Causal M-RFSQ Concept Tokenizer (FP32-only)
        s1_train_ds = MemmapTokenDataset(train_bin, chunk_size=cfg.chunk_size, prefix_len=0, dtype=np.uint16)
        s1_val_ds = MemmapTokenDataset(val_bin, chunk_size=cfg.chunk_size, prefix_len=0, dtype=np.uint16)
        memmap_datasets.extend([s1_train_ds, s1_val_ds])

        s1_steps = args.stage1_steps if not args.smoke else min(args.stage1_steps, 8)
        s1_bs = args.stage1_batch_size if not args.smoke else min(args.stage1_batch_size, 8)
        s1_dir = ckpt_dir / "stage1"

        if is_main:
            print(
                f"\n=== [Stage 1: ctx0an Causal M-RFSQ Tokenizer] {s1_steps} steps "
                f"(BS={s1_bs}/gpu x {world_size}, tokens={s1_train_ds.num_tokens:,}) ==="
            )
        s1_res = train_stage1(
            cfg=cfg,
            tokenizer=tokenizer,
            dataset=s1_train_ds,
            steps=s1_steps,
            batch_size=s1_bs,
            lr=args.stage1_lr,
            device=dev,
            apply_fine_drop=True,
            amp_dtype=None,  # Strictly FP32
            seed=args.seed,
            log_every=max(1, min(100, s1_steps // 10)),
            save_path=ckpt_dir / f"{args.model_name}_stage1.pt",
            save_dir=s1_dir,
            save_every=args.save_every,
            keep_last_k=args.keep_last_k,
            val_dataset=s1_val_ds,
            eval_every=args.eval_every,
            max_wall_minutes=eff_max_wall,
            hf_repo_id=args.hf_repo_id,
            hf_upload_every=args.hf_upload_every,
            hf_path_prefix="stage1",
            auto_resume=auto_resume,
            start_time_monotonic=wall_start,
            verbose=True,
        )
        s1_model: CNCMConceptTokenizer = s1_res["model"].eval()

        if s1_res["metrics"].get("stopped_early_wall_clock"):
            partial = {
                "model_name": args.model_name,
                "overall_status": "STOPPED_WALL_CLOCK_RESUMABLE",
                "stopped_phase": "stage1",
                "stage1_metrics": s1_res["metrics"],
                "elapsed_sec": round(time.perf_counter() - t0_all, 2),
            }
            if is_main:
                report_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            return partial

        gates23_recon: Dict[str, Any] = {}
        if is_main:
            gates23_recon = measure_per_domain_stage1_recon(
                s1_model=s1_model,
                tokenizer=tokenizer,
                domain_texts_dict=val_domains_eval,
                device=dev,
                max_chunks_per_domain=64 if not args.smoke else 16,
            )
            print(
                f"[GATES #2/#3 STAGE 1 RECON] k{gates23_recon['k_full']}_passed={gates23_recon['gate_k_full_all_passed']} | "
                f"k{gates23_recon['k_prev']}_passed={gates23_recon['gate_k_prev_all_passed']}"
            )
        if is_distributed:
            dist.barrier()

        # 4. Stage 2 Phase 1: Pure Exact-AR Foundation Pretraining (all tokens supervised)
        prefix_len = min(
            int(args.prefix_len),
            max(16, cfg.max_position_embeddings - cfg.total_concept_tokens - cfg.chunk_size),
        )
        s2_train_ds = MemmapTokenDataset(train_bin, chunk_size=cfg.chunk_size, prefix_len=prefix_len, dtype=np.uint16)
        s2_val_ds = MemmapTokenDataset(val_bin, chunk_size=cfg.chunk_size, prefix_len=prefix_len, dtype=np.uint16)
        memmap_datasets.extend([s2_train_ds, s2_val_ds])

        s2_bs = args.stage2_batch_size if not args.smoke else min(args.stage2_batch_size, 2)
        s2_ga = args.grad_accum if not args.smoke else 1
        p1_steps = args.phase1_steps if not args.smoke else min(args.phase1_steps, 4)
        p2_steps = args.phase2_steps if not args.smoke else min(args.phase2_steps, 6)
        p3_steps = args.phase3_sft_steps if not args.smoke else min(args.phase3_sft_steps, 4)

        p1_dir = ckpt_dir / "stage2_phase1_exact_ar"
        p2_dir = ckpt_dir / "stage2_phase2_joint"
        p3_dir = ckpt_dir / "stage2_phase3_sft"

        if is_main:
            tok_per_step = s2_bs * world_size * s2_ga * (prefix_len + cfg.chunk_size)
            print(
                f"\n=== [Stage 2 Phase 1/3: Exact-AR Foundation] {p1_steps} steps "
                f"(BS={s2_bs}/gpu x {world_size} x GA={s2_ga} = {tok_per_step:,} tok/step, "
                f"total~{p1_steps * tok_per_step / 1e6:.1f}M tok) ==="
            )
        p1_res = train_stage2(
            cfg=cfg,
            tokenizer=tokenizer,
            concept_tokenizer=s1_model,
            dataset=s2_train_ds,
            mode="exact_ar",
            steps=p1_steps,
            batch_size=s2_bs,
            grad_accum_steps=s2_ga,
            lr=args.phase1_lr,
            device=dev,
            amp_dtype=stage2_amp_dtype,
            grad_checkpointing=args.grad_checkpointing,
            seed=args.seed,
            log_every=max(1, min(50, p1_steps // 10)),
            save_path=None,
            save_dir=p1_dir,
            save_every=args.save_every,
            keep_last_k=args.keep_last_k,
            val_dataset=s2_val_ds,
            eval_every=args.eval_every,
            max_wall_minutes=eff_max_wall,
            hf_repo_id=args.hf_repo_id,
            hf_upload_every=args.hf_upload_every,
            hf_path_prefix="stage2_phase1_exact_ar",
            auto_resume=auto_resume,
            start_time_monotonic=wall_start,
            verbose=True,
        )
        model: CNCMModel = p1_res["model"]

        if p1_res["metrics"].get("stopped_early_wall_clock"):
            partial = {
                "model_name": args.model_name,
                "overall_status": "STOPPED_WALL_CLOCK_RESUMABLE",
                "stopped_phase": "stage2_phase1_exact_ar",
                "phase1_metrics": p1_res["metrics"],
                "elapsed_sec": round(time.perf_counter() - t0_all, 2),
            }
            if is_main:
                report_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            return partial

        if is_main:
            _cleanup_completed_phase_dir(p1_dir, keep_weights_only=True)
        if is_distributed:
            dist.barrier()

        # 5. Stage 2 Phase 2: Joint Dual-Mode + L_align Concept Adaptation
        if is_main:
            tok_per_step = s2_bs * world_size * s2_ga * (prefix_len + cfg.chunk_size)
            print(
                f"\n=== [Stage 2 Phase 2/3: Joint Dual-Mode + L_align] {p2_steps} steps "
                f"(total~{p2_steps * tok_per_step / 1e6:.1f}M tok, align={args.align_weight}, exact={args.exact_weight}) ==="
            )
        p2_res = train_stage2(
            cfg=cfg,
            tokenizer=tokenizer,
            concept_tokenizer=s1_model,
            dataset=s2_train_ds,
            mode="joint",
            steps=p2_steps,
            corrupt_rate=0.5,
            concept_weight=1.0,
            exact_weight=args.exact_weight,
            align_weight=args.align_weight,
            supervise_prefix_exact=True,
            batch_size=s2_bs,
            grad_accum_steps=s2_ga,
            lr=args.phase2_lr,
            device=dev,
            amp_dtype=stage2_amp_dtype,
            grad_checkpointing=args.grad_checkpointing,
            model=model,
            seed=args.seed + 1,
            log_every=max(1, min(50, p2_steps // 10)),
            save_path=None,
            save_dir=p2_dir,
            save_every=args.save_every,
            keep_last_k=args.keep_last_k,
            val_dataset=s2_val_ds,
            eval_every=args.eval_every,
            max_wall_minutes=eff_max_wall,
            hf_repo_id=args.hf_repo_id,
            hf_upload_every=args.hf_upload_every,
            hf_path_prefix="stage2_phase2_joint",
            auto_resume=auto_resume,
            start_time_monotonic=wall_start,
            verbose=True,
        )
        model = p2_res["model"]

        if p2_res["metrics"].get("stopped_early_wall_clock"):
            partial = {
                "model_name": args.model_name,
                "overall_status": "STOPPED_WALL_CLOCK_RESUMABLE",
                "stopped_phase": "stage2_phase2_joint",
                "phase2_metrics": p2_res["metrics"],
                "elapsed_sec": round(time.perf_counter() - t0_all, 2),
            }
            if is_main:
                report_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            return partial

        if is_main:
            _cleanup_completed_phase_dir(p2_dir, keep_weights_only=True)
        if is_distributed:
            dist.barrier()

        # 6. Stage 2 Phase 3: Curated Instruction / <think> CoT / Tool-Call SFT Annealing
        p3_res: Optional[Dict[str, Any]] = None
        if p3_steps > 0 and sft_convs:
            sft_ds = BlockARSFTDataset(
                sft_convs,
                tokenizer,
                chunk_size=cfg.chunk_size,
                max_prefix_len=prefix_len,
                pad_token_id=cfg.pad_token_id,
                bos_token_id=cfg.bos_token_id,
                eos_token_id=cfg.eos_token_id,
            )
            if is_main:
                print(
                    f"\n=== [Stage 2 Phase 3/3: Reasoning & Tool-Use SFT Annealing] {p3_steps} steps "
                    f"(samples={len(sft_ds):,}, lr={args.phase3_lr}) ==="
                )
            p3_res = train_stage2(
                cfg=cfg,
                tokenizer=tokenizer,
                concept_tokenizer=s1_model,
                dataset=sft_ds,
                mode="sft",
                steps=p3_steps,
                corrupt_rate=0.35,
                concept_weight=1.0,
                exact_weight=0.5,
                align_weight=args.align_weight,
                batch_size=max(1, s2_bs // 2),
                grad_accum_steps=max(1, s2_ga // 2),
                lr=args.phase3_lr,
                device=dev,
                amp_dtype=stage2_amp_dtype,
                grad_checkpointing=args.grad_checkpointing,
                model=model,
                seed=args.seed + 2,
                log_every=max(1, min(25, p3_steps // 10)),
                save_path=None,
                save_dir=p3_dir,
                save_every=args.save_every,
                keep_last_k=args.keep_last_k,
                val_dataset=s2_val_ds,
                eval_every=args.eval_every,
                max_wall_minutes=eff_max_wall,
                hf_repo_id=args.hf_repo_id,
                hf_upload_every=args.hf_upload_every,
                hf_path_prefix="stage2_phase3_sft",
                auto_resume=auto_resume,
                start_time_monotonic=wall_start,
                verbose=True,
            )
            model = p3_res["model"]

        model = model.eval()
        if not is_main:
            return {"model_name": args.model_name, "overall_status": "WORKER_DONE", "rank": rank}

        # Save clean FP32 weights-only export for inference / quantization / HF release
        from cncm.trainer import save_checkpoint

        save_checkpoint(
            ckpt_dir / f"{args.model_name}_weights_fp32.pt",
            model=model,
            optimizer=None,
            scheduler=None,
            step=int(p3_res["metrics"]["completed_steps"]) if p3_res else int(p2_res["metrics"]["completed_steps"]),
            config=cfg,
            extra={"model_name": args.model_name, "preset": cfg.preset_name},
            save_rng=False,
        )

        # 7. Rank-0 Final Evaluation Gates & Qualitative Generation Check
        max_eval_batches = 4 if args.smoke else 16
        gate1_entropy = measure_code_entropy(
            model=model,
            val_dataset=s2_val_ds,
            device=dev,
            max_batches=max_eval_batches,
            batch_size=min(8, max(2, s2_bs)),
            gate_threshold_nats=5.5,
        )
        gate5_gap = measure_concept_vs_exact_gap(
            model=model,
            val_dataset=s2_val_ds,
            device=dev,
            max_batches=max_eval_batches,
            batch_size=min(8, max(2, s2_bs)),
            gate_gap_threshold_nats=0.40,
        )
        samples = _generate_qualitative_samples(model, tokenizer, dev)

        gate_summary = {
            "gate_1_conditional_code_entropy_lt_5_5": gate1_entropy["gate_passed"],
            "gate_2_stage1_k_full_per_domain_recon": gates23_recon["gate_k_full_all_passed"],
            "gate_3_stage1_k_prev_early_exit_gt_75pct": gates23_recon["gate_k_prev_all_passed"],
            "gate_4_bpe_tokenizer_fertility": gate4_fertility["gate_all_passed"],
            "gate_5_concept_vs_exact_gap_lt_0_40": gate5_gap["gate_passed"],
        }

        report: Dict[str, Any] = {
            "model_name": args.model_name,
            "overall_status": "PASS" if all(gate_summary.values()) else "COMPLETED_NEEDS_MORE_TOKENS",
            "all_gates_passed": all(gate_summary.values()),
            "gate_summary": gate_summary,
            "config": {
                "preset": cfg.preset_name,
                "params": model.count_parameters(),
                "vocab_size": cfg.vocab_size,
                "hidden_size": cfg.hidden_size,
                "effective_layers": cfg.effective_layers,
                "enc_layers": cfg.enc_layers,
                "matryoshka_cutoffs": list(cfg.matryoshka_cutoffs),
                "prefix_len": prefix_len,
                "chunk_size": cfg.chunk_size,
                "bpe_trained": bpe_trained,
                "device": str(dev),
                "world_size": world_size,
                "stage2_amp_dtype": stage2_amp_dtype or "fp32",
                "grad_checkpointing": bool(args.grad_checkpointing),
                "memmap_train_tokens": s2_train_ds.num_tokens,
                "memmap_val_tokens": s2_val_ds.num_tokens,
            },
            "corpus_sources": corpus_sources,
            "training_metrics": {
                "stage1_final_loss": _safe_round(s1_res["metrics"].get("final_loss")),
                "stage1_final_recon_acc": _safe_round(s1_res["metrics"].get("final_recon_acc")),
                "phase1_exact_ar_final_loss": _safe_round(p1_res["metrics"].get("final_loss")),
                "phase1_tokens_seen": int(p1_res["metrics"].get("tokens_seen", 0)),
                "phase2_joint_final_loss": _safe_round(p2_res["metrics"].get("final_loss")),
                "phase2_tokens_seen": int(p2_res["metrics"].get("tokens_seen", 0)),
                "phase3_sft_final_loss": _safe_round(p3_res["metrics"].get("final_loss")) if p3_res else None,
            },
            "gates": {
                "gate_1_code_entropy": gate1_entropy,
                "gates_2_3_stage1_recon": gates23_recon,
                "gate_4_tokenizer_fertility": gate4_fertility,
                "gate_5_concept_vs_exact_gap": gate5_gap,
            },
            "qualitative_samples": samples,
            "elapsed_sec": round(time.perf_counter() - t0_all, 2),
            "checkpoints_dir": str(ckpt_dir),
        }

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n================ [{args.model_name} TRAINING SUMMARY] ================")
        for gname, gpass in gate_summary.items():
            print(f"  [{'PASS' if gpass else 'FAIL'}] {gname}")
        print(f"  STATUS: {report['overall_status']} | Elapsed: {report['elapsed_sec']}s")
        print(f"  Report: {report_path}")
        print("=================================================================")
        return report
    finally:
        for ds in memmap_datasets:
            ds.close()
        cleanup_ddp()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="High-Power Multi-Session Trainer for ctx0an (CNCM-Base).")
    parser.add_argument("--model-name", type=str, default="ctx0an")
    parser.add_argument("--preset", type=str, default="base", choices=["tiny_test", "nano", "mini", "base", "base_moe"])
    parser.add_argument("--enc-layers", type=int, default=1)
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--use-copy-decoder", action="store_true")
    parser.add_argument("--prefix-len", type=int, default=512)
    # Curated high-signal document targets (Textbooks + Code-heavy + CoT Math + Tool JSON)
    parser.add_argument("--en-docs", type=int, default=75000)
    parser.add_argument("--py-docs", type=int, default=70000)
    parser.add_argument("--math-docs", type=int, default=35000)
    parser.add_argument("--json-docs", type=int, default=20000)
    # Stage 1 (FP32)
    parser.add_argument("--stage1-steps", type=int, default=2500)
    parser.add_argument("--stage1-batch-size", type=int, default=64)
    parser.add_argument("--stage1-lr", type=float, default=2.0e-3)
    # Stage 2 (FP16 + GradScaler + grad_checkpointing + T4x2 DDP)
    # Effective batch = 6/gpu * 2 gpus * 10 grad_accum = 120 windows = 69,120 tokens/step
    parser.add_argument("--stage2-batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=10)
    parser.add_argument("--phase1-steps", type=int, default=1500)
    parser.add_argument("--phase1-lr", type=float, default=5.0e-4)
    parser.add_argument("--phase2-steps", type=int, default=2500)
    parser.add_argument("--phase2-lr", type=float, default=3.5e-4)
    parser.add_argument("--phase3-sft-steps", type=int, default=400)
    parser.add_argument("--phase3-lr", type=float, default=1.2e-4)
    parser.add_argument("--exact-weight", type=float, default=0.35)
    parser.add_argument("--align-weight", type=float, default=0.35)
    # Hardware, checkpointing & wall-clock safety (keep_last_k=1 fits 221M MoE easily under Kaggle's 19.5 GB cap)
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--grad-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--keep-last-k", type=int, default=1)
    parser.add_argument("--max-wall-minutes", type=float, default=325.0)
    parser.add_argument("--hf-repo-id", type=str, default="ctx0akib/ctx0an")
    parser.add_argument("--hf-upload-every", type=int, default=500)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--no-hf-download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only stream data, train BPE, and build uint16 memmaps.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    run_ctx0an_training(args)


if __name__ == "__main__":
    main()
