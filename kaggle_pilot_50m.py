"""Kaggle 50M-Token Pilot Harness for CNCM (`kaggle_pilot_50m.py`).

Self-contained pre-training validation pilot (optimized for Kaggle 2x NVIDIA T4 15GB,
12-hour hard timeout, 19.5 GB `/kaggle/working` disk cap, and single/multi-GPU DDP) that
gates all 5 readiness criteria before committing compute to the full 300M-token English +
Python + JSON run:

  Gate #1: Conditional Code Entropy < 5.5 nats/code (`measure_code_entropy`)
  Gate #2: Stage 1 Per-Domain Full-Depth Recon (EN > 99%, PY > 99%, JSON > 98%)
  Gate #3: Stage 1 Per-Domain Early-Exit Recon at k=7 (> 75%)
  Gate #4: Corpus-Trained BPE Fertility (`chars/token`: EN > 3.5, PY > 3.0, JSON > 2.5)
  Gate #5: Held-Out Concept vs Exact-AR NLL Gap (< 0.40 nats)

Pipeline:
  1. Loads English + Python + JSON corpora with strict document-level held-out validation splits
     and pre-tokenizes them into zero-padding flat `uint16` memory-mapped `.bin` files
     (`MemmapTokenDataset`, <10 MB RAM, <0.1s reload across resumed Kaggle sessions).
  2. Trains (or reuses on resume) a corpus-specific Byte-Level BPE tokenizer (`V=4096`,
     `Digits(individual_digits=True)`) preserving all 16 core + 11 PG-FIM special tokens,
     and runs Gate #4.
  3. Trains Stage 1 (`enc_layers=1` strictly causal semantic encoder, `matryoshka_cutoffs=(4,6,7,8)`,
     strictly FP32-safe) with rolling checkpointing & evaluates Gates #2 & #3.
  4. Trains Stage 2 with the 2-Phase Curriculum (FP16 + `GradScaler` on T4, BF16 on Ampere+,
     CUDA fused `AdamW`, optional activation gradient checkpointing, and `torchrun` DDP):
       - Phase 1 (25% steps): `mode="exact_ar"` (all tokens supervised, `prefix_len=256`)
       - Phase 2 (75% steps): `mode="joint"` (`align_weight=0.3`, `exact_weight=0.3`,
         `corrupt_rate=0.5`, `supervise_prefix_exact=False`)
     with rolling `CheckpointManager` (`keep_last_k=2`, `best.pt`, `latest.pt`, optional private
     HF Hub background sync, and 11-hour graceful wall-clock stop guard).
  5. Evaluates Gates #1 & #5 on rank 0 and writes a structured PASS/FAIL JSON summary to
     `/kaggle/working/pilot_50m_report.json` (or `--report-path`).

Usage:
    # Single-GPU Kaggle T4/P100/L4 pilot:
    python kaggle_pilot_50m.py

    # Dual-GPU Kaggle T4 x2 DDP pilot:
    torchrun --nproc_per_node=2 kaggle_pilot_50m.py

    # Quick local/smoke test:
    python kaggle_pilot_50m.py --preset tiny_test --smoke --out-dir ./pilot_smoke_out
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
    select_device,
    set_seed,
    setup_ddp_if_needed,
    train_stage1,
    train_stage2,
)


def _default_output_root() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists() and os.access(kaggle_working, os.W_OK):
        return kaggle_working
    return Path.cwd() / "pilot_working"


def _make_synthetic_json_corpus(num_samples: int = 512, seed: int = 42) -> List[str]:
    """Generate diverse, schema-rich JSON tool-call & API payload documents."""
    rng = random.Random(seed)
    tools = ("calculator", "kv_memory", "text_stats", "clock_utc", "code_search", "sql_query")
    statuses = ("ok", "cached", "updated", "verified")
    docs: List[str] = []
    for i in range(max(1, int(num_samples))):
        tool = tools[i % len(tools)]
        a = rng.randint(1, 999)
        b = rng.randint(1, 99)
        payload = {
            "request_id": f"req_{i:05d}",
            "tool_call": {
                "name": tool,
                "arguments": {
                    "expression": f"{a} * {b} + {a % 17}",
                    "key": f"metric_{i % 32}",
                    "limit": rng.randint(5, 50),
                    "strict": bool(i % 2 == 0),
                },
            },
            "tool_response": {
                "status": statuses[i % len(statuses)],
                "result": a * b + (a % 17),
                "latency_ms": round(rng.uniform(0.4, 24.5), 2),
                "tags": ["edge", "cncm", tool],
            },
        }
        docs.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return docs


def _load_domain_corpora(
    target_en: int = 12000,
    target_py: int = 8000,
    target_json: int = 4000,
    val_fraction: float = 0.10,
    allow_hf_download: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    """Load English + Python + JSON corpora with strict document-level train/val splits."""
    sources = {"english": "synthetic", "python": "synthetic", "json": "synthetic"}
    en_docs: List[str] = []
    py_docs: List[str] = []
    json_docs: List[str] = []

    if allow_hf_download:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]

            # 1. English prose (TinyStories or FineWeb-Edu streaming)
            for ds_name, ds_cfg in (
                ("roneneldan/TinyStories", None),
                ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
            ):
                try:
                    ds = (
                        load_dataset(ds_name, ds_cfg, split="train", streaming=True)
                        if ds_cfg
                        else load_dataset(ds_name, split="train", streaming=True)
                    )
                    for row in ds:
                        txt = str(row.get("text", "")).strip()
                        if len(txt) >= 80:
                            en_docs.append(txt[:2048])
                        if len(en_docs) >= target_en:
                            break
                    if len(en_docs) >= min(200, target_en):
                        sources["english"] = f"{ds_name}/{ds_cfg}" if ds_cfg else ds_name
                        break
                except Exception:
                    continue

            # 2. Python code (flytech/python-codes-25k or code_search_net streaming)
            for ds_name, ds_cfg, field in (
                ("flytech/python-codes-25k", None, "output"),
                ("code_search_net", "python", "whole_func_string"),
            ):
                try:
                    ds = (
                        load_dataset(ds_name, ds_cfg, split="train", streaming=True)
                        if ds_cfg
                        else load_dataset(ds_name, split="train", streaming=True)
                    )
                    for row in ds:
                        txt = str(row.get(field, "") or row.get("code", "")).strip()
                        if len(txt) >= 60:
                            py_docs.append(txt[:2048])
                        if len(py_docs) >= target_py:
                            break
                    if len(py_docs) >= min(200, target_py):
                        sources["python"] = f"{ds_name}/{ds_cfg}" if ds_cfg else ds_name
                        break
                except Exception:
                    continue
        except ImportError:
            pass

    # Deterministic fallbacks (and mix-in of PG-FIM / agentic JSON formats)
    if len(en_docs) < min(200, target_en):
        en_docs.extend(make_synthetic_corpus(num_samples=max(256, target_en), seed=seed))
    if len(py_docs) < min(200, target_py):
        py_docs.extend(make_synthetic_code_corpus(num_samples=max(256, target_py), seed=seed))
        py_docs.extend(make_pg_fim_stage1_corpus(num_samples=max(128, target_py // 4), seed=seed))

    # JSON domain: combine synthetic JSON payloads + formatted tool-call SFT JSON blocks
    json_docs.extend(_make_synthetic_json_corpus(num_samples=max(256, target_json), seed=seed))
    for item in make_synthetic_sft(num_samples=min(512, max(64, target_json // 4)), seed=seed):
        json_docs.append(json.dumps(item["messages"], ensure_ascii=False, separators=(",", ":")))

    rng = random.Random(seed)
    train_domains: Dict[str, List[str]] = {}
    val_domains: Dict[str, List[str]] = {}

    for name, docs in (("english", en_docs), ("python", py_docs), ("json", json_docs)):
        shuffled = list(docs)
        rng.shuffle(shuffled)
        n_val = max(16, min(len(shuffled) // 4, int(len(shuffled) * val_fraction)))
        val_domains[name] = shuffled[:n_val]
        train_domains[name] = shuffled[n_val:] if len(shuffled) > n_val else shuffled

    return train_domains, val_domains, sources


def _select_stage2_amp_dtype(device: torch.device, requested: str = "auto") -> Optional[str]:
    """Select Stage 2 AMP precision: native BF16 on Ampere+ (SM>=8), FP16+GradScaler on T4/P100, FP32 on CPU."""
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
        return "fp16"
    return None


def _safe_round_loss(val: Any) -> Optional[float]:
    if val is None:
        return None
    return round(float(val), 5)


def run_pilot(args: argparse.Namespace) -> Dict[str, Any]:
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
        data_cache_dir = out_root / "memmap_data"
        report_path = Path(args.report_path) if args.report_path else (out_root / "pilot_50m_report.json")
        if is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            data_cache_dir.mkdir(parents=True, exist_ok=True)
            report_path.parent.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            dist.barrier()

        # Configure preset with strictly causal 1-layer Stage 1 semantic encoder & Matryoshka cutoffs
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
        eff_max_wall = (
            args.max_wall_minutes
            if (args.max_wall_minutes is not None and not args.smoke)
            else (args.max_wall_minutes if args.max_wall_minutes is not None and args.max_wall_minutes <= 5.0 else None)
        )
        auto_resume = not args.no_auto_resume

        if is_main:
            print(
                f"[PILOT INIT] preset={cfg.preset_name} | device={dev} | world_size={world_size} | "
                f"stage1_amp=fp32 | stage2_amp={stage2_amp_dtype or 'fp32'} | "
                f"grad_ckpt={args.grad_checkpointing} | max_wall_min={eff_max_wall} | "
                f"enc_layers={cfg.enc_layers} (causal) | matryoshka={cfg.matryoshka_cutoffs}"
            )

        # 1 & 2. Rank 0 loads corpora, trains/loads BPE tokenizer, and writes uint16 memmap .bin files.
        bpe_path = ckpt_dir / "cncm_bpe_tokenizer.json"
        train_bin = data_cache_dir / "train_tokens.bin"
        val_bin = data_cache_dir / "val_tokens.bin"
        meta_path = data_cache_dir / "corpus_meta.json"

        bpe_trained = False
        val_domains: Dict[str, List[str]] = {}
        corpus_sources: Dict[str, str] = {}

        if is_main:
            if args.smoke:
                target_en, target_py, target_json = 128, 128, 128
            else:
                target_en, target_py, target_json = args.en_docs, args.py_docs, args.json_docs

            can_reuse_cache = (
                auto_resume
                and train_bin.exists()
                and val_bin.exists()
                and meta_path.exists()
            )
            if can_reuse_cache:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                val_domains = meta["val_domains"]
                corpus_sources = meta["corpus_sources"]
                bpe_trained = bool(meta.get("bpe_trained", False))
                if bpe_path.exists():
                    tokenizer = CNCMTextTokenizer.from_hf_json(bpe_path, vocab_size=cfg.vocab_size)
                else:
                    tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
                print(
                    f"[DATA RESUME] Reused pre-tokenized memmap corpora in {data_cache_dir} "
                    f"(bpe_trained={bpe_trained})"
                )
            else:
                train_domains, val_domains, corpus_sources = _load_domain_corpora(
                    target_en=target_en,
                    target_py=target_py,
                    target_json=target_json,
                    val_fraction=0.10,
                    allow_hf_download=not args.no_hf_download,
                    seed=args.seed,
                )
                all_train_texts = train_domains["english"] + train_domains["python"] + train_domains["json"]
                all_val_texts = val_domains["english"] + val_domains["python"] + val_domains["json"]
                print(
                    f"[DATA] sources={corpus_sources} | train_docs=(en:{len(train_domains['english'])}, "
                    f"py:{len(train_domains['python'])}, json:{len(train_domains['json'])}) | "
                    f"val_docs=(en:{len(val_domains['english'])}, py:{len(val_domains['python'])}, json:{len(val_domains['json'])})"
                )

                if auto_resume and bpe_path.exists():
                    tokenizer = CNCMTextTokenizer.from_hf_json(bpe_path, vocab_size=cfg.vocab_size)
                    bpe_trained = True
                    print(f"[TOKENIZER] Loaded existing corpus Byte-Level BPE from {bpe_path}")
                else:
                    try:
                        tokenizer = CNCMTextTokenizer.train_bpe(
                            all_train_texts,
                            vocab_size=cfg.vocab_size,
                            save_path=bpe_path,
                            min_frequency=2,
                        )
                        bpe_trained = True
                        print(f"[TOKENIZER] Trained corpus Byte-Level BPE (V={cfg.vocab_size}) -> {bpe_path}")
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
                meta_path.write_text(
                    json.dumps(
                        {
                            "corpus_sources": corpus_sources,
                            "val_domains": val_domains,
                            "bpe_trained": bpe_trained,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"[MEMMAP] Wrote uint16 binaries: train={train_bin.stat().st_size / 1e6:.2f} MB, "
                    f"val={val_bin.stat().st_size / 1e6:.2f} MB"
                )

        if is_distributed:
            dist.barrier()

        if not is_main:
            if bpe_path.exists():
                tokenizer = CNCMTextTokenizer.from_hf_json(bpe_path, vocab_size=cfg.vocab_size)
                bpe_trained = True
            else:
                tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

        gate4_fertility: Dict[str, Any] = {}
        if is_main:
            gate4_fertility = measure_tokenizer_fertility(tokenizer, val_domains)
            print(
                f"[GATE #4 FERTILITY] passed={gate4_fertility['gate_all_passed']} | "
                f"{json.dumps(gate4_fertility['domains'])}"
            )

        # 3. Stage 1 Training (Strictly Causal enc_layers=1 + Matryoshka Cutoffs, FP32-only)
        s1_train_ds = MemmapTokenDataset(
            train_bin,
            chunk_size=cfg.chunk_size,
            prefix_len=0,
            dtype=np.uint16,
        )
        s1_val_ds = MemmapTokenDataset(
            val_bin,
            chunk_size=cfg.chunk_size,
            prefix_len=0,
            dtype=np.uint16,
        )
        memmap_datasets.extend([s1_train_ds, s1_val_ds])

        s1_steps = args.stage1_steps if not args.smoke else min(args.stage1_steps, 12)
        s1_bs = args.stage1_batch_size if not args.smoke else min(args.stage1_batch_size, 8)
        s1_save_dir = ckpt_dir / "stage1"
        if is_main:
            print(
                f"\n=== [Stage 1] Training {s1_steps} steps "
                f"(BS={s1_bs}/gpu x {world_size}, memmap_tokens={s1_train_ds.num_tokens:,}, chunks={len(s1_train_ds):,}) ==="
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
            amp_dtype=None,  # Strictly FP32 for Stage 1 FSQ/STE safety
            seed=args.seed,
            log_every=max(1, s1_steps // 10),
            save_path=ckpt_dir / "stage1_causal_matryoshka.pt",
            save_dir=s1_save_dir,
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
                "overall_status": "STOPPED_WALL_CLOCK_RESUMABLE",
                "stopped_phase": "stage1",
                "stage1_metrics": s1_res["metrics"],
                "elapsed_sec": round(time.perf_counter() - t0_all, 2),
                "checkpoints_dir": str(ckpt_dir),
            }
            if is_main:
                report_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
                print(f"[PILOT] Graceful wall-clock stop in Stage 1; resumable checkpoint saved in {s1_save_dir}")
            return partial

        gates23_recon: Dict[str, Any] = {}
        if is_main:
            gates23_recon = measure_per_domain_stage1_recon(
                s1_model=s1_model,
                tokenizer=tokenizer,
                domain_texts_dict=val_domains,
                device=dev,
                max_chunks_per_domain=64 if not args.smoke else 16,
            )
            print(
                f"[GATES #2/#3 STAGE 1 RECON] k{gates23_recon['k_full']}_passed={gates23_recon['gate_k_full_all_passed']} | "
                f"k{gates23_recon['k_prev']}_passed={gates23_recon['gate_k_prev_all_passed']}"
            )
        if is_distributed:
            dist.barrier()

        # 4. Stage 2 Training: 2-Phase Curriculum (25% exact_ar -> 75% joint with align_weight=0.3)
        prefix_len = min(
            int(args.prefix_len),
            max(16, cfg.max_position_embeddings - cfg.total_concept_tokens - cfg.chunk_size),
        )
        s2_train_ds = MemmapTokenDataset(
            train_bin,
            chunk_size=cfg.chunk_size,
            prefix_len=prefix_len,
            dtype=np.uint16,
        )
        s2_val_ds = MemmapTokenDataset(
            val_bin,
            chunk_size=cfg.chunk_size,
            prefix_len=prefix_len,
            dtype=np.uint16,
        )
        memmap_datasets.extend([s2_train_ds, s2_val_ds])

        total_s2_steps = args.stage2_steps if not args.smoke else min(args.stage2_steps, 16)
        s2_bs = args.stage2_batch_size if not args.smoke else min(args.stage2_batch_size, 4)
        phase1_steps = max(1, int(round(total_s2_steps * 0.25)))
        phase2_steps = max(1, total_s2_steps - phase1_steps)

        p1_save_dir = ckpt_dir / "stage2_phase1"
        p2_save_dir = ckpt_dir / "stage2_phase2"

        if is_main:
            print(
                f"\n=== [Stage 2 Phase 1/2: Exact-AR] {phase1_steps} steps "
                f"(BS={s2_bs}/gpu x {world_size}, grad_accum={args.grad_accum}, "
                f"prefix_len={prefix_len}, amp={stage2_amp_dtype or 'fp32'}) ==="
            )
        p1_res = train_stage2(
            cfg=cfg,
            tokenizer=tokenizer,
            concept_tokenizer=s1_model,
            dataset=s2_train_ds,
            mode="exact_ar",
            steps=phase1_steps,
            batch_size=s2_bs,
            grad_accum_steps=args.grad_accum,
            lr=args.stage2_lr,
            device=dev,
            amp_dtype=stage2_amp_dtype,
            grad_checkpointing=args.grad_checkpointing,
            seed=args.seed,
            log_every=max(1, phase1_steps // 5),
            save_path=ckpt_dir / "stage2_phase1_exact_ar.pt",
            save_dir=p1_save_dir,
            save_every=args.save_every,
            keep_last_k=args.keep_last_k,
            val_dataset=s2_val_ds,
            eval_every=args.eval_every,
            max_wall_minutes=eff_max_wall,
            hf_repo_id=args.hf_repo_id,
            hf_upload_every=args.hf_upload_every,
            hf_path_prefix="stage2_phase1",
            auto_resume=auto_resume,
            start_time_monotonic=wall_start,
            verbose=True,
        )
        model: CNCMModel = p1_res["model"]

        if p1_res["metrics"].get("stopped_early_wall_clock"):
            partial = {
                "overall_status": "STOPPED_WALL_CLOCK_RESUMABLE",
                "stopped_phase": "stage2_phase1",
                "stage2_phase1_metrics": p1_res["metrics"],
                "elapsed_sec": round(time.perf_counter() - t0_all, 2),
                "checkpoints_dir": str(ckpt_dir),
            }
            if is_main:
                report_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
                print(f"[PILOT] Graceful wall-clock stop in Stage 2 Phase 1; resumable checkpoint saved in {p1_save_dir}")
            return partial

        if is_main:
            print(
                f"\n=== [Stage 2 Phase 2/2: Joint + L_align] {phase2_steps} steps "
                f"(align_weight={args.align_weight}, exact_weight={args.exact_weight}, "
                f"corrupt_rate=0.5, amp={stage2_amp_dtype or 'fp32'}) ==="
            )
        p2_res = train_stage2(
            cfg=cfg,
            tokenizer=tokenizer,
            concept_tokenizer=s1_model,
            dataset=s2_train_ds,
            mode="joint",
            steps=phase2_steps,
            corrupt_rate=0.5,
            concept_weight=1.0,
            exact_weight=args.exact_weight,
            align_weight=args.align_weight,
            supervise_prefix_exact=args.supervise_prefix_exact,
            batch_size=s2_bs,
            grad_accum_steps=args.grad_accum,
            lr=args.stage2_lr * 0.8,
            device=dev,
            amp_dtype=stage2_amp_dtype,
            grad_checkpointing=args.grad_checkpointing,
            model=model,
            seed=args.seed + 1,
            log_every=max(1, phase2_steps // 10),
            save_path=ckpt_dir / "stage2_phase2_joint_align.pt",
            save_dir=p2_save_dir,
            resume_from=args.resume_from,
            save_every=args.save_every,
            keep_last_k=args.keep_last_k,
            val_dataset=s2_val_ds,
            eval_every=args.eval_every,
            max_wall_minutes=eff_max_wall,
            hf_repo_id=args.hf_repo_id,
            hf_upload_every=args.hf_upload_every,
            hf_path_prefix="stage2_phase2",
            auto_resume=auto_resume,
            start_time_monotonic=wall_start,
            verbose=True,
        )
        model = p2_res["model"].eval()

        if p2_res["metrics"].get("stopped_early_wall_clock"):
            partial = {
                "overall_status": "STOPPED_WALL_CLOCK_RESUMABLE",
                "stopped_phase": "stage2_phase2",
                "stage2_phase2_metrics": p2_res["metrics"],
                "elapsed_sec": round(time.perf_counter() - t0_all, 2),
                "checkpoints_dir": str(ckpt_dir),
            }
            if is_main:
                report_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
                print(f"[PILOT] Graceful wall-clock stop in Stage 2 Phase 2; resumable checkpoint saved in {p2_save_dir}")
            return partial

        if not is_main:
            return {"overall_status": "WORKER_DONE", "rank": rank}

        # 5. Run Gates #1 & #5 on rank 0
        max_eval_batches = 4 if args.smoke else 16
        gate1_entropy = measure_code_entropy(
            model=model,
            val_dataset=s2_val_ds,
            device=dev,
            max_batches=max_eval_batches,
            batch_size=min(16, s2_bs * 2),
            gate_threshold_nats=5.5,
        )
        gate5_gap = measure_concept_vs_exact_gap(
            model=model,
            val_dataset=s2_val_ds,
            device=dev,
            max_batches=max_eval_batches,
            batch_size=min(16, s2_bs * 2),
            gate_gap_threshold_nats=0.40,
        )

        gate_summary = {
            "gate_1_conditional_code_entropy_lt_5_5": gate1_entropy["gate_passed"],
            "gate_2_stage1_k_full_per_domain_recon": gates23_recon["gate_k_full_all_passed"],
            "gate_3_stage1_k_prev_early_exit_gt_75pct": gates23_recon["gate_k_prev_all_passed"],
            "gate_4_bpe_tokenizer_fertility": gate4_fertility["gate_all_passed"],
            "gate_5_concept_vs_exact_gap_lt_0_40": gate5_gap["gate_passed"],
        }
        all_gates_passed = all(gate_summary.values())

        tokens_per_step = s2_bs * world_size * args.grad_accum * (prefix_len + cfg.chunk_size)
        est_stage2_tokens = total_s2_steps * tokens_per_step
        p2_align_hist = p2_res["metrics"].get("align_loss_history", [])

        report: Dict[str, Any] = {
            "overall_status": "PASS" if all_gates_passed else "FAIL",
            "all_gates_passed": all_gates_passed,
            "gate_summary": gate_summary,
            "config": {
                "preset": cfg.preset_name,
                "vocab_size": cfg.vocab_size,
                "enc_layers": cfg.enc_layers,
                "matryoshka_cutoffs": list(cfg.matryoshka_cutoffs),
                "concept_head_hidden": cfg.concept_head_hidden,
                "loop_adapter_rank": cfg.loop_adapter_rank,
                "num_loops": cfg.num_loops,
                "prefix_len": prefix_len,
                "chunk_size": cfg.chunk_size,
                "bpe_trained": bpe_trained,
                "device": str(dev),
                "world_size": world_size,
                "stage1_amp_dtype": "fp32",
                "stage2_amp_dtype": stage2_amp_dtype or "fp32",
                "grad_checkpointing": bool(args.grad_checkpointing),
                "memmap_train_tokens": s2_train_ds.num_tokens,
                "memmap_val_tokens": s2_val_ds.num_tokens,
                "estimated_stage2_tokens": est_stage2_tokens,
            },
            "corpus_sources": corpus_sources,
            "training_metrics": {
                "stage1_final_loss": _safe_round_loss(s1_res["metrics"].get("final_loss")),
                "stage2_phase1_exact_ar_final_loss": _safe_round_loss(p1_res["metrics"].get("final_loss")),
                "stage2_phase2_joint_final_loss": _safe_round_loss(p2_res["metrics"].get("final_loss")),
                "stage2_phase2_final_align_loss": _safe_round_loss(p2_align_hist[-1] if p2_align_hist else None),
                "stage2_phase2_tokens_seen": int(p2_res["metrics"].get("tokens_seen", 0)),
            },
            "gates": {
                "gate_1_code_entropy": gate1_entropy,
                "gates_2_3_stage1_recon": gates23_recon,
                "gate_4_tokenizer_fertility": gate4_fertility,
                "gate_5_concept_vs_exact_gap": gate5_gap,
            },
            "elapsed_sec": round(time.perf_counter() - t0_all, 2),
            "checkpoints_dir": str(ckpt_dir),
        }

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\n================ KAGGLE 50M PILOT GATE SUMMARY ================")
        for gname, gpass in gate_summary.items():
            print(f"  [{'PASS' if gpass else 'FAIL'}] {gname}")
        print(f"  OVERALL: {report['overall_status']} | Report saved to: {report_path}")
        print("===============================================================")
        return report
    finally:
        for ds in memmap_datasets:
            ds.close()
        cleanup_ddp()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNCM 50M-Token Kaggle Pilot Harness & Readiness Gates.")
    parser.add_argument("--preset", type=str, default="nano", choices=["tiny_test", "nano", "mini", "base"])
    parser.add_argument("--enc-layers", type=int, default=1)
    parser.add_argument("--num-loops", type=int, default=None, help="Optional override (e.g. 1 or 3).")
    parser.add_argument("--use-copy-decoder", action="store_true")
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--stage1-steps", type=int, default=2000)
    parser.add_argument("--stage1-batch-size", type=int, default=64)
    parser.add_argument("--stage1-lr", type=float, default=2.5e-3)
    # 50M tokens / (32 * 4 * (256+64)) = 50,000,000 / 40,960 ~= 1221 steps
    parser.add_argument("--stage2-steps", type=int, default=1225)
    parser.add_argument("--stage2-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--stage2-lr", type=float, default=1.2e-3)
    parser.add_argument("--exact-weight", type=float, default=0.3)
    parser.add_argument("--align-weight", type=float, default=0.3)
    parser.add_argument("--supervise-prefix-exact", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--keep-last-k", type=int, default=2)
    parser.add_argument("--max-wall-minutes", type=float, default=660.0)
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-upload-every", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--en-docs", type=int, default=12000)
    parser.add_argument("--py-docs", type=int, default=8000)
    parser.add_argument("--json-docs", type=int, default=4000)
    parser.add_argument("--no-hf-download", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Fast local smoke verification mode.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    run_pilot(args)


if __name__ == "__main__":
    main()
