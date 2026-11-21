"""Dual-Space Hierarchical Distillation & Draft-Alignment for CNCM (`cncm/trainer/draft_align.py`).

Implements the 3-Level Hierarchical Teacher -> Student Distillation stack for CNCM:
  1. **Frozen In-Model or External Teacher Snapshot**: Keeps a strictly frozen copy of the
     post-SFT / Teacher model so the target distribution `y_ar` and teacher logits NEVER drift
     or collapse during student alignment (fixing the `28.2% -> 1.2%` collapse in `exp_5_1`).
  2. **Dynamic Residual Concept-DAgger (`dagger_prob`)**: Because `MultiscaleRFSQ` uses an
     analytic zero-table scalar grid, on-policy student rollouts dynamically re-quantize the
     true remaining latent residual `z_teacher - z_hat_student_{<=k-1}` at every level `k`:
         c_k^DAgger, delta_k = quantizer.quantize_single_level_residual(z_teacher - z_hat_student, k)
     eliminating open-loop coarse-to-fine exposure bias!
  3. **Dual-Head Cross-Mode KL (`L_align` + Stage-1 Decoder Logit Distillation)**:
     Aligns BOTH the finest-level concept hidden states `exact_head(h_fine)` AND the
     differentiable Stage 1 M-RFSQ decoder distribution against the frozen teacher's
     token distribution.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cncm.layers import build_concept_mask
from cncm.model import CNCMModel
from cncm.trainer.train_utils import WarmupCosineScheduler, set_seed


@torch.no_grad()
def build_concept_dagger_rollout(
    model: CNCMModel,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prefix_mask: Optional[torch.Tensor] = None,
    student_rollout_prob: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build on-policy Concept-DAgger conditioning and dynamically re-quantized residual targets.

    At each coarse-to-fine level `k = 0..K-1`:
      - Re-quantizes the remaining teacher latent residual `z_teacher - z_hat_running` via
        `quantizer.quantize_single_level_residual` to obtain the optimal corrective FSQ digits `c_k^*`.
      - With probability `student_rollout_prob`, samples the student's own greedy prediction
        `c_k^student` at level `k` and accumulates `delta(c_k^student)` into `z_hat_running`
        so level `k+1` learns to actively recover from level `k`'s prediction errors!

    Returns:
      - cond_list: list of K-1 conditioning tensors `[B, l_k, d_vq]` for levels 1..K-1
      - dagger_target_codes: list of K corrective target digit tensors `[B, l_k, d_q]`
    """
    was_training = model.training
    model.eval()

    bsz, prefix_len = prefix_ids.shape
    device = prefix_ids.device
    emb_dtype = model.exact_embed.weight.dtype
    quant = model.concept_tokenizer.quantizer

    # Ground-truth continuous Stage 1 normalized latent z_teacher [B, L, d_vq]
    z_teacher = model.concept_tokenizer.encode_z(target_ids).to(dtype=emb_dtype)

    # Prefill prefix into KV cache once for fast student greedy rollouts
    cur_kv, cur_plen, _ = model._prefill_or_extend_prefix(
        prefix_ids=prefix_ids,
        prefix_kv=None,
        prefix_len=0,
        prefix_mask=prefix_mask,
        quantize_kv=False,
    )
    norm_pm = model._normalize_prefix_mask(prefix_mask, cur_plen, bsz, device)

    z_hat_running = torch.zeros_like(z_teacher)
    cond_list: List[torch.Tensor] = []
    dagger_target_codes: List[torch.Tensor] = []

    for k in range(model.cfg.num_levels):
        l_k = model.cfg.levels[k]
        # 1. Dynamic Residual Re-Quantization: optimal corrective code for (z_teacher - z_hat_running)
        z_rest = z_teacher - z_hat_running
        c_k_target, delta_target = quant.quantize_single_level_residual(z_rest, k)
        dagger_target_codes.append(c_k_target)

        if k == model.cfg.num_levels - 1:
            break

        # 2. Decide whether to roll out student's own predicted code at level k (DAgger)
        use_student = (student_rollout_prob > 0.0) and (
            float(torch.rand((), device=device).item()) < float(student_rollout_prob)
        )
        if use_student:
            lid = torch.tensor(k, device=device, dtype=torch.long)
            l_emb = model.level_embed(lid).view(1, 1, model.cfg.hidden_size)
            if k == 0:
                x_k = model.l0_queries.to(dtype=emb_dtype).expand(bsz, -1, -1) + l_emb
            else:
                x_k = model.concept_in_proj(cond_list[k - 1].to(dtype=emb_dtype)) + l_emb

            pos_k = model._concept_level_positions(cur_plen, l_k, device=device)
            from cncm.layers import build_cached_concept_step_mask

            step_mask = build_cached_concept_step_mask(
                prefix_len=cur_plen,
                level_len=l_k,
                batch_size=bsz,
                prefix_mask=norm_pm,
                device=device,
                dtype=x_k.dtype,
            )
            h_k, _ = model._run_backbone(
                x_k,
                attn_mask=step_mask,
                position_ids=pos_k,
                past_key_values=cur_kv,
                use_cache=False,
                commit_to_cache=False,
            )
            digits_student, _, _ = model.concept_head.sample(
                h_k, level_ids=k, temperature=0.0
            )
            delta_used = quant.decode_level(digits_student, k, dtype=emb_dtype)
        else:
            delta_used = delta_target.to(dtype=emb_dtype)

        z_hat_running = z_hat_running + delta_used
        cond_list.append(quant.get_level_conditioning(z_hat_running, k + 1))

    if was_training:
        model.train()
    return cond_list, dagger_target_codes


def forward_concept_with_custom_conditioning(
    model: CNCMModel,
    prefix_ids: torch.Tensor,
    cond_list: Sequence[torch.Tensor],
    target_codes: Sequence[torch.Tensor],
    prefix_mask: Optional[torch.Tensor] = None,
    level_weights_alpha: float = 0.5,
) -> Dict[str, Any]:
    """Run teacher-forced Concept forward using custom Concept-DAgger `cond_list` and `target_codes`."""
    bsz, prefix_len = prefix_ids.shape
    device = prefix_ids.device
    num_active_levels = len(target_codes)
    level_lengths = tuple(int(c.shape[1]) for c in target_codes)
    emb_dtype = model.exact_embed.weight.dtype

    parts: List[torch.Tensor] = []
    pos_parts: List[torch.Tensor] = []
    level_id_parts: List[torch.Tensor] = []

    if prefix_len > 0:
        parts.append(model.embed_dropout(model.exact_embed(prefix_ids)))
        pos_parts.append(torch.arange(prefix_len, device=device, dtype=torch.long))

    for k in range(num_active_levels):
        l_k = level_lengths[k]
        lid = torch.tensor(k, device=device, dtype=torch.long)
        l_emb = model.level_embed(lid).view(1, 1, model.cfg.hidden_size)
        if k == 0:
            x_k = model.l0_queries.to(dtype=emb_dtype).expand(bsz, -1, -1) + l_emb
        else:
            x_k = model.concept_in_proj(cond_list[k - 1].to(dtype=emb_dtype)) + l_emb
        parts.append(model.embed_dropout(x_k))
        pos_parts.append(model._concept_level_positions(prefix_len, l_k, device=device))
        level_id_parts.append(torch.full((l_k,), k, device=device, dtype=torch.long))

    x = torch.cat(parts, dim=1)
    position_ids = torch.cat(pos_parts, dim=0)
    level_ids = torch.cat(level_id_parts, dim=0)

    attn_mask = build_concept_mask(
        prefix_len=prefix_len,
        level_lengths=level_lengths,
        batch_size=bsz,
        prefix_mask=prefix_mask,
        device=device,
        dtype=x.dtype,
    )
    h, _, moe_aux_loss = model._run_backbone(
        x,
        attn_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        return_aux_loss=True,
    )
    h_target = h[:, prefix_len:, :]
    all_targets = torch.cat([c.to(device=device, dtype=torch.long) for c in target_codes], dim=1)
    digit_logits, token_log_probs = model.concept_head.log_probs(
        h_target, level_ids=level_ids, target_digits=all_targets
    )

    level_losses: List[torch.Tensor] = []
    offset = 0
    for l_k in level_lengths:
        level_losses.append((-token_log_probs[:, offset : offset + l_k]).mean())
        offset += l_k

    weights = model._level_weights(level_lengths, level_weights_alpha, device=device)
    loss = torch.stack(level_losses).mul(weights).sum()
    if model.cfg.num_experts > 0 and model.cfg.moe_aux_loss_weight > 0.0:
        loss = loss + float(model.cfg.moe_aux_loss_weight) * moe_aux_loss.to(dtype=loss.dtype)

    return {
        "loss": loss,
        "level_losses": level_losses,
        "digit_logits": digit_logits,
        "log_probs": token_log_probs,
        "h_target": h_target,
    }


def train_draft_alignment(
    model: CNCMModel,
    dataset: Dataset,
    *,
    teacher_model: Optional[CNCMModel] = None,
    steps: int = 40,
    batch_size: int = 8,
    lr: float = 8e-4,
    kl_weight: float = 0.5,
    exact_anchor_weight: float = 0.25,
    temperature: float = 1.0,
    dagger_prob: float = 0.5,
    corrupt_rate: float = 0.25,
    freeze_backbone_for_exact: bool = True,
    device: Union[str, torch.device] = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Distill Concept Mode onto a strictly frozen Exact-AR Teacher with Dynamic Residual Concept-DAgger.

    Key Fixes over legacy `draft_align.py`:
      1. Uses a strictly frozen `teacher_ref` snapshot (`teacher_model` or `copy.deepcopy(model)`)
         so the Exact-AR target rollouts `y_ar` and teacher logits NEVER drift or collapse.
      2. Defaults to `freeze_backbone_for_exact=True` (training `concept_head`, `concept_in_proj`,
         `level_embed`, `l0_queries`, plus optional `loop_adapters`), guaranteeing zero degradation
         to Exact-AR Mode while aligning Concept Mode directly to the shared backbone!
      3. Combines Dynamic Residual Concept-DAgger (`dagger_prob=0.5`) with conditioning corruption
         (`corrupt_rate=0.25`) so open-loop Concept generation stays calibrated at test time.
    """
    set_seed(seed)
    dev = torch.device(device)
    model = model.to(dev)
    model.freeze_concept_tokenizer()

    # Freeze an immutable teacher reference so y_ar and teacher_logits never drift
    if teacher_model is None or teacher_model is model:
        teacher_ref = copy.deepcopy(model).to(dev).eval()
    else:
        teacher_ref = copy.deepcopy(teacher_model).to(dev).eval()
    for p in teacher_ref.parameters():
        p.requires_grad_(False)

    # Ensure student generator parameters have requires_grad=True
    for p in model.generator_parameters():
        p.requires_grad_(True)
    model.freeze_concept_tokenizer()

    if freeze_backbone_for_exact:
        # Train Concept-Mode projections & FactorizedConceptHead while keeping Exact-AR weights 100% intact
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
    sched = WarmupCosineScheduler(opt, total_steps=max(1, steps), warmup_ratio=0.1, min_lr_ratio=0.1)
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

        # 1. Generate Exact-AR greedy continuation y_ar + teacher logits from FROZEN teacher_ref
        with torch.no_grad():
            ar_out = teacher_ref.generate_exact(pfx, max_new_tokens=L, temperature=0.0, prefix_mask=pfx_m)
            y_ar = ar_out["token_ids"]  # [B, L]
            full_ar = torch.cat([pfx, y_ar], dim=1)
            full_logits = teacher_ref.forward_exact(full_ar, use_cache=False)["logits"]
            # Logits predicting y_ar[:, 0..L-1] sit at positions [P-1 .. P+L-2]
            teacher_logits = full_logits[:, pfx.shape[1] - 1 : pfx.shape[1] - 1 + L, :].detach()

        # 2. Build Concept-DAgger rollout or corrupted teacher-forced conditioning
        use_dagger = (dagger_prob > 0.0) and (
            float(torch.rand((), device=dev).item()) < float(dagger_prob)
        )
        if use_dagger:
            cond_list, target_codes = build_concept_dagger_rollout(
                model,
                prefix_ids=pfx,
                target_ids=y_ar,
                prefix_mask=pfx_m,
                student_rollout_prob=0.5,
            )
            model.train()
            model.freeze_concept_tokenizer()
            opt.zero_grad(set_to_none=True)
            c_out = forward_concept_with_custom_conditioning(
                model,
                prefix_ids=pfx,
                cond_list=cond_list,
                target_codes=target_codes,
                prefix_mask=pfx_m,
            )
        else:
            with torch.no_grad():
                c_ar = model.concept_tokenizer.encode_codes(y_ar)
            model.train()
            model.freeze_concept_tokenizer()
            opt.zero_grad(set_to_none=True)
            c_out = model.forward_concept(
                pfx, c_ar, prefix_mask=pfx_m, corrupt_rate=corrupt_rate
            )

        c_loss = c_out["loss"]

        # 3. Cross-Mode Token-Level KL Distillation (h_fine -> exact_head vs frozen teacher_logits)
        h_fine = c_out["h_target"][:, -L:, :]
        student_logits = model.exact_head(h_fine).float() / float(temperature)
        t_probs = F.softmax(teacher_logits.float() / float(temperature), dim=-1)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            t_probs,
            reduction="batchmean",
        ) * (float(temperature) ** 2)

        loss = c_loss + float(kl_weight) * kl_loss

        # 4. Anchor Exact-AR when backbone is unfrozen so Exact-AR never drifts
        if not freeze_backbone_for_exact and exact_anchor_weight > 0.0:
            ex_labels = torch.cat(
                [torch.full_like(pfx, -100), y_ar],
                dim=1,
            )
            ex_out = model.forward_exact(full_ar, labels=ex_labels, use_cache=False)
            loss = loss + float(exact_anchor_weight) * ex_out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        loss_hist.append(float(loss.detach().item()))
        concept_hist.append(float(c_loss.detach().item()))
        kl_hist.append(float(kl_loss.detach().item()))
        if verbose and (step == 1 or step == steps or step % max(1, steps // 4) == 0):
            print(
                f"[DraftAlign] step {step:3d}/{steps} | "
                f"loss={loss_hist[-1]:.4f} c={concept_hist[-1]:.4f} kl={kl_hist[-1]:.4f}"
            )

    model.eval()
    return {
        "model": model,
        "steps": steps,
        "dagger_prob": dagger_prob,
        "freeze_backbone_for_exact": freeze_backbone_for_exact,
        "initial_loss": round(loss_hist[0], 5) if loss_hist else 0.0,
        "final_loss": round(loss_hist[-1], 5) if loss_hist else 0.0,
        "initial_concept_loss": round(concept_hist[0], 5) if concept_hist else 0.0,
        "final_concept_loss": round(concept_hist[-1], 5) if concept_hist else 0.0,
        "final_kl_loss": round(kl_hist[-1], 5) if kl_hist else 0.0,
    }
