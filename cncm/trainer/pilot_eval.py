"""Conditional Code Entropy & 50M-Token Pilot Gate Diagnostics (`cncm.trainer.pilot_eval`).

Provides reusable evaluation gates to validate tokenizer fertility, Stage 1 per-domain
reconstruction fidelity, Stage 2 conditional code entropy, and Concept-vs-Exact NLL
alignment before committing compute to a 300M-token training run.

Pilot Gates:
  1. `measure_code_entropy`:
       - Held-out per-level conditional code entropy H(c^(k) | prefix, c^(<k)) in nats/code
       - Marginal empirical code entropy of Stage 1 `encode_codes`
       - Gate threshold: mean conditional entropy < 5.5 nats/code
  2. `measure_per_domain_stage1_recon`:
       - Per-domain Stage 1 `recon_acc` at full depth (k=K, e.g. 8) and k=K-1 (e.g. 7)
       - Gates: english > 99%, python > 99%, json > 98% at k=K; > 75% at k=K-1
  3. `measure_tokenizer_fertility`:
       - Chars / token per domain
       - Gates: english > 3.5, python > 3.0, json > 2.5
  4. `measure_concept_vs_exact_gap`:
       - Held-out Exact-AR NLL, L_align NLL, and Concept-draft NLL scored by `exact_head`
       - Gate: |concept_draft_nll - exact_ar_nll| (or concept_draft_nll - exact_ar_nll) < 0.40 nats
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cncm.model import CNCMModel
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer.dataset import PackedChunkDataset, collate_block_ar
from cncm.trainer.train_utils import select_device


def _make_val_loader(
    val_dataset: Dataset,
    batch_size: int,
    pad_token_id: int,
) -> DataLoader:
    sample0 = val_dataset[0]
    use_block = isinstance(sample0, dict) and "prefix_ids" in sample0
    eff_bs = max(1, min(int(batch_size), len(val_dataset)))
    if use_block:
        return DataLoader(
            val_dataset,
            batch_size=eff_bs,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda b: collate_block_ar(b, pad_token_id=pad_token_id),
        )
    return DataLoader(val_dataset, batch_size=eff_bs, shuffle=False, drop_last=False)


def _extract_prefix_target(
    batch: Dict[str, torch.Tensor],
    cfg: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "prefix_ids" in batch and ("target_ids" in batch or "target_chunk_ids" in batch):
        pfx = batch["prefix_ids"].to(device)
        pfx_m = batch.get("prefix_mask")
        if pfx_m is None:
            pfx_m = (pfx != cfg.pad_token_id).long()
        else:
            pfx_m = pfx_m.to(device)
        tgt = batch.get("target_ids", batch.get("target_chunk_ids")).to(device)
    else:
        inp = batch["input_ids"].to(device)
        if inp.shape[1] > cfg.chunk_size:
            pfx = inp[:, : -cfg.chunk_size]
            tgt = inp[:, -cfg.chunk_size :]
        else:
            pfx = torch.full((inp.shape[0], 1), cfg.bos_token_id, dtype=torch.long, device=device)
            tgt = inp[:, : cfg.chunk_size]
        pfx_m = torch.ones_like(pfx, dtype=torch.long)
    return pfx, pfx_m, tgt


@torch.no_grad()
def measure_code_entropy(
    model: CNCMModel,
    val_dataset: Dataset,
    device: Union[str, torch.device] = "auto",
    max_batches: int = 16,
    batch_size: int = 16,
    gate_threshold_nats: float = 5.5,
) -> Dict[str, Any]:
    """Compute held-out per-level conditional code entropy H(c^(k) | prefix, c^(<k)) and marginal code entropy.

    Returns a dict including `mean_conditional_nats_per_code`, `per_level_conditional_nats`,
    `per_level_marginal_nats`, `mean_marginal_nats_per_code`, `bottleneck_hint`, and `gate_passed`.
    """
    dev = select_device(device)
    model = model.to(dev)
    was_training = model.training
    model.eval()

    cfg = model.cfg
    n_levels = cfg.num_levels
    codebook_size = cfg.codebook_size
    dl = _make_val_loader(val_dataset, batch_size=batch_size, pad_token_id=cfg.pad_token_id)

    cond_nll_sums = [0.0] * n_levels
    cond_token_counts = [0] * n_levels
    marginal_counts: List[torch.Tensor] = [
        torch.zeros(codebook_size, dtype=torch.long) for _ in range(n_levels)
    ]
    n_batches = 0

    try:
        for idx, batch in enumerate(dl):
            if idx >= max_batches:
                break
            pfx, pfx_m, tgt = _extract_prefix_target(batch, cfg, dev)
            codes = model.concept_tokenizer.encode_codes(tgt)
            c_out = model.forward_concept(
                prefix_ids=pfx,
                target_codes=codes,
                prefix_mask=pfx_m,
                corrupt_rate=0.0,
            )
            level_losses: Sequence[torch.Tensor] = c_out["level_losses"]
            bsz = int(tgt.shape[0])

            for k in range(n_levels):
                l_k = int(codes[k].shape[1])
                cnt = bsz * l_k
                cond_nll_sums[k] += float(level_losses[k].detach().float().item()) * cnt
                cond_token_counts[k] += cnt

                flat_k = model.concept_tokenizer.digits_to_flat(codes[k]).reshape(-1).cpu()
                bc = torch.bincount(flat_k, minlength=codebook_size)
                marginal_counts[k] += bc[:codebook_size]
            n_batches += 1
    finally:
        if was_training:
            model.train(True)

    per_level_cond: Dict[str, float] = {}
    per_level_marg: Dict[str, float] = {}
    total_cond_sum = 0.0
    total_cond_cnt = 0
    marg_vals: List[float] = []

    for k in range(n_levels):
        key = f"L{k}(len={cfg.levels[k]})"
        c_nats = cond_nll_sums[k] / float(max(1, cond_token_counts[k]))
        per_level_cond[key] = round(c_nats, 4)
        total_cond_sum += cond_nll_sums[k]
        total_cond_cnt += cond_token_counts[k]

        counts_f = marginal_counts[k].float()
        tot_m = float(counts_f.sum().item())
        if tot_m > 0.0:
            probs = counts_f[counts_f > 0] / tot_m
            m_ent = float(-(probs * torch.log(probs)).sum().item())
        else:
            m_ent = 0.0
        per_level_marg[key] = round(m_ent, 4)
        marg_vals.append(m_ent)

    mean_cond = total_cond_sum / float(max(1, total_cond_cnt))
    mean_marg = sum(marg_vals) / float(max(1, len(marg_vals)))
    uniform_nats = math.log(float(codebook_size))

    # Flag whether the bottleneck is Stage 1 marginal entropy or Stage 2 generator prediction
    if mean_cond < gate_threshold_nats:
        bottleneck_hint = "ok_below_threshold"
    elif mean_marg > 0.85 * uniform_nats and (mean_marg - mean_cond) < 0.5:
        bottleneck_hint = "tokenizer_high_entropy_or_weak_conditioning"
    else:
        bottleneck_hint = "generator_undertrained_or_capacity_bound"

    return {
        "batches_evaluated": n_batches,
        "codebook_size": codebook_size,
        "uniform_entropy_nats": round(uniform_nats, 4),
        "mean_conditional_nats_per_code": round(mean_cond, 4),
        "mean_marginal_nats_per_code": round(mean_marg, 4),
        "per_level_conditional_nats": per_level_cond,
        "per_level_marginal_nats": per_level_marg,
        "gate_threshold_nats": float(gate_threshold_nats),
        "gate_passed": bool(mean_cond < gate_threshold_nats),
        "bottleneck_hint": bottleneck_hint,
    }


@torch.no_grad()
def measure_per_domain_stage1_recon(
    s1_model: CNCMConceptTokenizer,
    tokenizer: Any,
    domain_texts_dict: Mapping[str, Sequence[str]],
    device: Union[str, torch.device] = "auto",
    max_chunks_per_domain: int = 128,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate held-out Stage 1 `recon_acc` at full depth (k=K) and k=K-1 per domain.

    Default pilot gates:
      - `english`: k=K > 0.99, k=K-1 > 0.75
      - `python`:  k=K > 0.99, k=K-1 > 0.75
      - `json`:    k=K > 0.98, k=K-1 > 0.75
    """
    dev = select_device(device)
    s1_model = s1_model.to(dev)
    was_training = s1_model.training
    s1_model.eval()
    cfg = s1_model.cfg
    k_full = cfg.num_levels
    k_prev = max(1, cfg.num_levels - 1)

    full_thresholds = {"english": 0.99, "python": 0.99, "json": 0.98}
    prev_threshold = 0.75

    per_domain: Dict[str, Dict[str, Any]] = {}
    all_full_passed = True
    all_prev_passed = True

    try:
        for domain, texts in domain_texts_dict.items():
            clean_texts = [t for t in texts if t and t.strip()]
            if not clean_texts:
                per_domain[domain] = {"error": "empty_domain_texts", "gate_k_full_passed": False, "gate_k_prev_passed": False}
                all_full_passed = False
                all_prev_passed = False
                continue

            ds = PackedChunkDataset(
                clean_texts,
                tokenizer,
                chunk_size=cfg.chunk_size,
                eos_token_id=cfg.eos_token_id,
                bos_token_id=cfg.bos_token_id,
            )
            n_chunks = min(int(max_chunks_per_domain), len(ds))
            ids = torch.stack([ds[i]["input_ids"] for i in range(n_chunks)], dim=0)

            corr_full = 0
            corr_prev = 0
            total_tok = 0
            for st in range(0, n_chunks, max(1, int(batch_size))):
                xb = ids[st : st + batch_size].to(dev)
                codes = s1_model.encode_codes(xb)
                logits_full = s1_model.decode_codes(codes)
                logits_prev = s1_model.decode_codes(codes[:k_prev])
                corr_full += int((logits_full.argmax(dim=-1) == xb).sum().item())
                corr_prev += int((logits_prev.argmax(dim=-1) == xb).sum().item())
                total_tok += int(xb.numel())

            acc_full = corr_full / float(max(1, total_tok))
            acc_prev = corr_prev / float(max(1, total_tok))
            thr_full = full_thresholds.get(domain.lower(), 0.98)
            pass_full = acc_full > thr_full
            pass_prev = acc_prev > prev_threshold
            if not pass_full:
                all_full_passed = False
            if not pass_prev:
                all_prev_passed = False

            per_domain[domain] = {
                "chunks_evaluated": n_chunks,
                "tokens_evaluated": total_tok,
                f"recon_acc_k{k_full}": round(acc_full, 5),
                f"recon_acc_k{k_prev}": round(acc_prev, 5),
                "gate_k_full_threshold": thr_full,
                "gate_k_prev_threshold": prev_threshold,
                "gate_k_full_passed": bool(pass_full),
                "gate_k_prev_passed": bool(pass_prev),
            }
    finally:
        if was_training:
            s1_model.train(True)

    return {
        "k_full": k_full,
        "k_prev": k_prev,
        "domains": per_domain,
        "gate_k_full_all_passed": bool(all_full_passed),
        "gate_k_prev_all_passed": bool(all_prev_passed),
    }


def measure_tokenizer_fertility(
    tokenizer: Any,
    domain_texts_dict: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """Compute `chars / token` fertility per domain.

    Default pilot gates: `english > 3.5`, `python > 3.0`, `json > 2.5`.
    """
    thresholds = {"english": 3.5, "python": 3.0, "json": 2.5}
    per_domain: Dict[str, Dict[str, Any]] = {}
    all_passed = True

    for domain, texts in domain_texts_dict.items():
        total_chars = 0
        total_tokens = 0
        for txt in texts:
            if not txt:
                continue
            total_chars += len(txt)
            ids = tokenizer.encode(txt, add_special_tokens=False)
            total_tokens += len(ids)

        cpt = total_chars / float(max(1, total_tokens))
        thr = thresholds.get(domain.lower(), 2.5)
        passed = cpt > thr
        if not passed:
            all_passed = False
        per_domain[domain] = {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "chars_per_token": round(cpt, 4),
            "gate_threshold": thr,
            "gate_passed": bool(passed),
        }

    return {
        "domains": per_domain,
        "gate_all_passed": bool(all_passed),
    }


@torch.no_grad()
def measure_concept_vs_exact_gap(
    model: CNCMModel,
    val_dataset: Dataset,
    device: Union[str, torch.device] = "auto",
    max_batches: int = 16,
    batch_size: int = 16,
    gate_gap_threshold_nats: float = 0.40,
) -> Dict[str, Any]:
    """Compare held-out Exact-AR NLL, L_align NLL, and Concept-draft NLL scored by `exact_head`.

    Measures:
      1. `exact_ar_nll`: cross-entropy of `exact_head` on ground-truth `target_ids` conditioned on `prefix_ids`.
      2. `align_nll`: cross-entropy of `exact_head(h_fine)` from `forward_concept` against `target_ids`.
      3. `concept_draft_nll_by_exact`: cross-entropy of `exact_head` on tokens generated by
         `generate_concept_chunk(..., temperature=0.0, early_exit_conf=None)`.
      4. `align_minus_exact_gap` & `concept_draft_minus_exact_gap` against `gate_gap_threshold_nats` (0.40).
    """
    dev = select_device(device)
    model = model.to(dev)
    was_training = model.training
    model.eval()

    cfg = model.cfg
    dl = _make_val_loader(val_dataset, batch_size=batch_size, pad_token_id=cfg.pad_token_id)

    exact_nll_sum = 0.0
    exact_tok_cnt = 0
    align_nll_sum = 0.0
    align_tok_cnt = 0
    draft_nll_sum = 0.0
    draft_tok_cnt = 0
    n_batches = 0

    try:
        for idx, batch in enumerate(dl):
            if idx >= max_batches:
                break
            pfx, pfx_m, tgt = _extract_prefix_target(batch, cfg, dev)
            bsz, p_len = pfx.shape
            t_len = int(tgt.shape[1])

            # 1. Exact-AR NLL on ground-truth target tokens
            full_ids = torch.cat([pfx, tgt], dim=1)
            tgt_valid = tgt != cfg.pad_token_id
            if not tgt_valid.any():
                tgt_valid = torch.ones_like(tgt, dtype=torch.bool)
            full_mask = torch.cat([pfx_m.bool(), tgt_valid], dim=1)
            ex_out = model.forward_exact(full_ids, attention_mask=full_mask, use_cache=False)
            ex_logits = ex_out["logits"][:, p_len - 1 : p_len + t_len - 1, :].float()
            ex_lbl = tgt.masked_fill(~tgt_valid, -100)
            valid_ex = ex_lbl != -100
            if valid_ex.any():
                exact_nll_sum += float(F.cross_entropy(ex_logits[valid_ex], ex_lbl[valid_ex], reduction="sum").item())
                exact_tok_cnt += int(valid_ex.sum().item())

            # 2. L_align NLL from finest Concept Mode hidden states
            codes = model.concept_tokenizer.encode_codes(tgt)
            c_out = model.forward_concept(pfx, codes, prefix_mask=pfx_m, corrupt_rate=0.0)
            h_fine = c_out["h_target"][:, -t_len:, :]
            align_logits = model.exact_head(h_fine).float()
            if valid_ex.any():
                align_nll_sum += float(
                    F.cross_entropy(align_logits[valid_ex], ex_lbl[valid_ex], reduction="sum").item()
                )
                align_tok_cnt += int(valid_ex.sum().item())

            # 3. Concept-draft NLL scored by `exact_head`
            draft_out = model.generate_concept_chunk(
                prefix_ids=pfx,
                prefix_mask=pfx_m,
                temperature=0.0,
                early_exit_conf=None,
                return_codes=False,
            )
            draft_ids = draft_out["token_ids"]
            draft_full = torch.cat([pfx, draft_ids], dim=1)
            draft_mask = torch.cat([pfx_m.bool(), torch.ones_like(draft_ids, dtype=torch.bool)], dim=1)
            d_ex = model.forward_exact(draft_full, attention_mask=draft_mask, use_cache=False)
            d_logits = d_ex["logits"][:, p_len - 1 : p_len + draft_ids.shape[1] - 1, :].float()
            draft_nll_sum += float(
                F.cross_entropy(d_logits.reshape(-1, cfg.vocab_size), draft_ids.reshape(-1), reduction="sum").item()
            )
            draft_tok_cnt += int(draft_ids.numel())
            n_batches += 1
    finally:
        if was_training:
            model.train(True)

    exact_nll = exact_nll_sum / float(max(1, exact_tok_cnt))
    align_nll = align_nll_sum / float(max(1, align_tok_cnt))
    draft_nll = draft_nll_sum / float(max(1, draft_tok_cnt))
    align_gap = align_nll - exact_nll
    draft_gap = draft_nll - exact_nll

    return {
        "batches_evaluated": n_batches,
        "exact_ar_nll": round(exact_nll, 5),
        "exact_ar_ppl": round(math.exp(min(exact_nll, 20.0)), 3),
        "align_nll": round(align_nll, 5),
        "align_ppl": round(math.exp(min(align_nll, 20.0)), 3),
        "concept_draft_nll_by_exact": round(draft_nll, 5),
        "concept_draft_ppl_by_exact": round(math.exp(min(draft_nll, 20.0)), 3),
        "align_minus_exact_gap": round(align_gap, 5),
        "concept_draft_minus_exact_gap": round(draft_gap, 5),
        "gate_gap_threshold_nats": float(gate_gap_threshold_nats),
        "gate_passed": bool(abs(align_gap) < gate_gap_threshold_nats and abs(draft_gap) < gate_gap_threshold_nats),
    }


__all__ = [
    "measure_code_entropy",
    "measure_per_domain_stage1_recon",
    "measure_tokenizer_fertility",
    "measure_concept_vs_exact_gap",
]
