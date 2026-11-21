"""Shared experiment harness for controlled CNCM studies.

Enforces identical data splits, seeds, config hashing, and JSON ledger logging
under `experiments/results/`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def config_hash(cfg_dict: Dict[str, Any]) -> str:
    raw = json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def save_result(name: str, payload: Dict[str, Any]) -> Path:
    """Write an experiment JSON record with provenance metadata."""
    enriched = {
        "experiment": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        **payload,
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return path


def build_strict_splits(
    tokenizer: Any,
    chunk_size: int,
    prefix_len: int,
    *,
    num_code_docs: int = 360,
    num_prose_docs: int = 360,
    seed: int = 42,
    use_real_english: bool = False,
) -> Dict[str, Any]:
    """Build strictly disjoint train / val / test document splits BEFORE packing.

    70% train / 15% val (for hyperparameter selection) / 15% test (untouched final eval).
    """
    from cncm.trainer.dataset import (
        PackedChunkDataset,
        make_pg_fim_stage1_corpus,
        make_synthetic_corpus,
    )

    code_docs = make_pg_fim_stage1_corpus(num_samples=num_code_docs, seed=seed)
    prose_docs: List[str] = []
    source = "synthetic_prose+pg_fim_code"
    if use_real_english:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]

            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            for row in ds:
                txt = str(row.get("text", "")).strip()
                if len(txt) >= 64:
                    prose_docs.append(txt[:480])
                if len(prose_docs) >= num_prose_docs:
                    break
            if len(prose_docs) >= 60:
                source = "TinyStories+pg_fim_code"
        except Exception:
            prose_docs = []
    if len(prose_docs) < 60:
        prose_docs = make_synthetic_corpus(num_samples=num_prose_docs, seed=seed + 100)

    def _split3(docs: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(docs)
        n_test = max(12, int(n * 0.15))
        n_val = max(12, int(n * 0.15))
        return docs[n_test + n_val :], docs[n_test : n_test + n_val], docs[:n_test]

    c_tr, c_va, c_te = _split3(code_docs)
    p_tr, p_va, p_te = _split3(prose_docs)

    train_docs = c_tr + p_tr
    val_docs = c_va + p_va
    test_docs = c_te + p_te

    return {
        "source": source,
        "s1_train": PackedChunkDataset(train_docs, tokenizer, chunk_size=chunk_size),
        "s1_val": PackedChunkDataset(val_docs, tokenizer, chunk_size=chunk_size),
        "s1_test": PackedChunkDataset(test_docs, tokenizer, chunk_size=chunk_size),
        "s2_train": PackedChunkDataset(train_docs, tokenizer, chunk_size=chunk_size, prefix_len=prefix_len),
        "s2_val": PackedChunkDataset(val_docs, tokenizer, chunk_size=chunk_size, prefix_len=prefix_len),
        "s2_test": PackedChunkDataset(test_docs, tokenizer, chunk_size=chunk_size, prefix_len=prefix_len),
    }


@torch.no_grad()
def eval_lm_split(
    model: Any,
    ds: Any,
    *,
    device: torch.device,
    eval_concept: bool = True,
    max_batches: int = 16,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate exact-AR NLL/PPL and (optionally) concept loss + concept-draft NLL."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    exact_nll_sum = 0.0
    exact_tok_count = 0
    concept_loss_sum = 0.0
    concept_draft_nll_sum = 0.0
    concept_draft_tok = 0
    n_b = 0

    for idx, b in enumerate(dl):
        if idx >= max_batches:
            break
        pfx = b["prefix_ids"].to(device)
        pfx_m = b["prefix_mask"].to(device)
        tgt = b["target_ids"].to(device)
        inp = b["exact_input_ids"].to(device)
        lbl = b["exact_labels"].to(device)
        msk = b["exact_attention_mask"].to(device)

        ex_out = model.forward_exact(inp, labels=lbl, attention_mask=msk)
        logits = ex_out["logits"][:, :-1, :].float()
        shift_lbl = lbl[:, 1:]
        valid = shift_lbl != -100
        if valid.any():
            nll = F.cross_entropy(logits[valid], shift_lbl[valid], reduction="sum")
            exact_nll_sum += float(nll.item())
            exact_tok_count += int(valid.sum().item())

        if eval_concept:
            codes = model.concept_tokenizer.encode_codes(tgt)
            c_out = model.forward_concept(pfx, codes, prefix_mask=pfx_m, corrupt_rate=0.0)
            concept_loss_sum += float(c_out["loss"].item())
            # Also score greedy concept-draft tokens under exact_head (draft quality!)
            draft_out = model.generate_concept_chunk(
                pfx[: min(4, pfx.shape[0])],
                prefix_mask=pfx_m[: min(4, pfx.shape[0])],
                temperature=0.0,
                early_exit_conf=None,
            )
            draft_ids = draft_out["token_ids"]
            p_sub = pfx[: draft_ids.shape[0]]
            full_d = torch.cat([p_sub, draft_ids], dim=1)
            d_ex = model.forward_exact(full_d, use_cache=False)["logits"][:, p_sub.shape[1] - 1 : -1, :].float()
            concept_draft_nll_sum += float(
                F.cross_entropy(d_ex.reshape(-1, d_ex.shape[-1]), draft_ids.reshape(-1), reduction="sum").item()
            )
            concept_draft_tok += int(draft_ids.numel())

        n_b += 1

    val_nll = exact_nll_sum / float(max(1, exact_tok_count))
    res: Dict[str, Any] = {
        "exact_nll": round(val_nll, 5),
        "exact_ppl": round(math.exp(min(val_nll, 20.0)), 3),
        "tokens_evaluated": exact_tok_count,
    }
    if eval_concept and n_b > 0:
        res["concept_loss"] = round(concept_loss_sum / n_b, 5)
        if concept_draft_tok > 0:
            res["concept_draft_nll"] = round(concept_draft_nll_sum / concept_draft_tok, 5)
    return res
