"""Draft-Alignment Training for Self-Speculative CNCM Decoding (Review Item 5.1).

Addresses the root cause of low self-speculative acceptance:
Standard Stage 2 trains Concept Mode on ground-truth corpus continuations, while
self-speculative verification accepts Concept Mode drafts ONLY when they match
Exact-AR's own distribution/greedy rollout (`y_ar`).

`train_draft_alignment` performs sequence-level + token-level self-distillation:
  1. Generates on-policy Exact-AR greedy continuations `y_ar` from prefixes.
  2. Encodes `y_ar` with the frozen Stage 1 M-RFSQ tokenizer into `c_ar`.
  3. Trains Concept Mode (`forward_concept(prefix, c_ar)` + `L_align` KL against
     Exact-AR teacher logits on `y_ar`), optionally with a light Exact-AR anchor
     so Exact-AR NLL does not drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cncm.model import CNCMModel
from cncm.trainer.train_utils import WarmupCosineScheduler, set_seed


def train_draft_alignment(
    model: CNCMModel,
    dataset: Dataset,
    *,
    steps: int = 40,
    batch_size: int = 8,
    lr: float = 8e-4,
    kl_weight: float = 0.5,
    exact_anchor_weight: float = 0.1,
    temperature: float = 1.0,
    freeze_backbone_for_exact: bool = False,
    device: Union[str, torch.device] = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Distill Concept Mode onto Exact-AR's own greedy rollouts to raise acceptance alpha."""
    set_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)
    model.freeze_concept_tokenizer()

    if freeze_backbone_for_exact:
        # Train only Concept-specific projections/head while preserving Exact-AR identically
        trainable = [
            p
            for n, p in model.named_parameters()
            if (
                n.startswith("concept_head.")
                or n.startswith("concept_in_proj.")
                or n.startswith("level_embed.")
                or n == "l0_queries"
            )
            and p.requires_grad
        ]
    else:
        trainable = model.generator_parameters()

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = WarmupCosineScheduler(opt, total_steps=steps, warmup_ratio=0.1, min_lr_ratio=0.1)
    dl = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)

    def _inf():
        while True:
            for b in dl:
                yield b

    it = _inf()
    loss_hist: List[float] = []
    concept_hist: List[float] = []
    kl_hist: List[float] = []

    L = model.cfg.chunk_size

    for step in range(1, steps + 1):
        b = next(it)
        pfx = b["prefix_ids"].to(dev)
        pfx_m = b["prefix_mask"].to(dev) if "prefix_mask" in b else None

        # 1. Generate Exact-AR greedy continuation y_ar + teacher logits (no grad)
        model.eval()
        with torch.no_grad():
            ar_out = model.generate_exact(pfx, max_new_tokens=L, temperature=0.0, prefix_mask=pfx_m)
            y_ar = ar_out["token_ids"]  # [B, L]
            c_ar = model.concept_tokenizer.encode_codes(y_ar)
            full_ar = torch.cat([pfx, y_ar], dim=1)
            teacher_logits = model.forward_exact(full_ar, use_cache=False)["logits"][:, pfx.shape[1] - 1 : -1, :].detach()

        # 2. Train Concept Mode on (pfx -> c_ar) + KL(exact_head(h_fine) || teacher_logits)
        model.train()
        model.freeze_concept_tokenizer()
        opt.zero_grad(set_to_none=True)

        c_out = model.forward_concept(pfx, c_ar, prefix_mask=pfx_m, corrupt_rate=0.0)
        c_loss = c_out["loss"]

        h_fine = c_out["h_target"][:, -L:, :]
        student_logits = model.exact_head(h_fine).float() / float(temperature)
        t_probs = F.softmax(teacher_logits.float() / float(temperature), dim=-1)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            t_probs,
            reduction="batchmean",
        ) * (float(temperature) ** 2)

        loss = c_loss + float(kl_weight) * kl_loss

        if not freeze_backbone_for_exact and exact_anchor_weight > 0.0 and "exact_input_ids" in b:
            ex_out = model.forward_exact(
                b["exact_input_ids"].to(dev),
                labels=b["exact_labels"].to(dev),
                attention_mask=b["exact_attention_mask"].to(dev),
            )
            loss = loss + float(exact_anchor_weight) * ex_out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        loss_hist.append(float(loss.detach().item()))
        concept_hist.append(float(c_loss.detach().item()))
        kl_hist.append(float(kl_loss.detach().item()))
        if verbose and (step == 1 or step == steps or step % max(1, steps // 4) == 0):
            print(f"[DraftAlign] step {step:3d}/{steps} | loss={loss_hist[-1]:.4f} c={concept_hist[-1]:.4f} kl={kl_hist[-1]:.4f}")

    model.eval()
    return {
        "model": model,
        "steps": steps,
        "initial_loss": round(loss_hist[0], 5),
        "final_loss": round(loss_hist[-1], 5),
        "initial_concept_loss": round(concept_hist[0], 5),
        "final_concept_loss": round(concept_hist[-1], 5),
        "final_kl_loss": round(kl_hist[-1], 5),
    }
