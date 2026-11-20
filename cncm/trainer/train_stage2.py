"""Stage 2 training pipeline for CNCM Generator (`CNCMModel`).

Trains the unified Dual-Mode Concept + Exact-AR backbone while keeping Stage 1
`CNCMConceptTokenizer` strictly frozen:
- Supports `mode="joint"` (default), `mode="pretrain"`, and `mode="sft"`.
- Always passes explicit `prefix_mask` (never omitting padding masks!).
- Combines Concept Mode (with 50% linear conditioning corruption) and Exact-AR auxiliary
  supervision on target / `<tool_call>` spans.
- Supports optional **Self-Distillation** (`self_distill=True`) and **QAT Fine-Tuning**
  (`qat_finetune=True`) using STE weight fake-quantization cleanly compatible with FP32/INT4 export.
- Atomic checkpointing preserving exact FP32 / INT64 dtypes (never `.half()`).

CLI usage:
    python -m cncm.trainer.train_stage2 --preset tiny_test --mode joint --steps 30 --batch-size 4
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from cncm.config import CNCMConfig, get_preset
from cncm.model import CNCMModel
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer.dataset import (
    BlockARSFTDataset,
    MemmapTokenDataset,
    PackedChunkDataset,
    collate_block_ar,
    make_synthetic_corpus,
    make_synthetic_sft,
)
from cncm.trainer.train_utils import (
    CheckpointManager,
    WarmupCosineScheduler,
    _parse_dtype,
    amp_context,
    clip_and_check_grad_norm,
    count_parameters,
    get_device_type,
    get_dist_info,
    load_checkpoint,
    make_adamw,
    make_dataloader,
    save_checkpoint,
    select_device,
    set_seed,
)


def _to_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, torch.Tensor):
        return float(val.detach().float().mean().item())
    return float(val)


def _infinite_loader(
    loader: DataLoader,
    sampler: Optional[DistributedSampler] = None,
) -> Iterator[dict[str, torch.Tensor]]:
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _sync_bool_flag(flag: bool, is_distributed: bool, device: torch.device) -> bool:
    if not is_distributed or not (dist.is_available() and dist.is_initialized()):
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int64, device=device)
    dist.broadcast(t, src=0)
    return bool(int(t.item()) != 0)


class _Stage2DDPWrapper(nn.Module):
    """Single-entry forward wrapper so `DistributedDataParallel` hooks fire for all Stage 2 modes."""

    def __init__(self, model: CNCMModel) -> None:
        super().__init__()
        self.module = model

    def forward(self, dispatch: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if dispatch == "exact":
            return self.module.forward_exact(*args, **kwargs)
        if dispatch == "joint":
            return self.module.forward_joint(*args, **kwargs)
        if dispatch == "concept":
            return self.module.forward_concept(*args, **kwargs)
        if dispatch == "sft":
            prefix_ids = kwargs["prefix_ids"]
            prefix_mask = kwargs["prefix_mask"]
            target_ids = kwargs["target_ids"]
            exact_input_ids = kwargs["exact_input_ids"]
            exact_attention_mask = kwargs["exact_attention_mask"]
            exact_labels = kwargs["exact_labels"]
            corrupt_rate = float(kwargs["corrupt_rate"])
            level_weights_alpha = float(kwargs["level_weights_alpha"])
            concept_weight = float(kwargs["concept_weight"])
            exact_weight = float(kwargs["exact_weight"])
            align_weight = float(kwargs["align_weight"])
            dev = prefix_ids.device

            with torch.no_grad():
                target_codes = self.module.concept_tokenizer.encode_codes(target_ids)
            c_out = self.module.forward_concept(
                prefix_ids,
                target_codes,
                prefix_mask=prefix_mask,
                corrupt_rate=corrupt_rate,
                level_weights_alpha=level_weights_alpha,
            )
            c_loss = c_out["loss"]
            if align_weight > 0.0 and "h_target" in c_out:
                t_len = target_ids.shape[1]
                h_fine = c_out["h_target"][:, -t_len:, :]
                a_logits = self.module.exact_head(h_fine).float()
                a_tgt = target_ids.to(device=dev, dtype=torch.long).clone()
                non_pad_a = a_tgt != self.module.cfg.pad_token_id
                if non_pad_a.any():
                    a_tgt = a_tgt.masked_fill(~non_pad_a, -100)
                a_loss = F.cross_entropy(
                    a_logits.reshape(-1, self.module.cfg.vocab_size),
                    a_tgt.reshape(-1),
                    ignore_index=-100,
                )
            else:
                a_loss = torch.zeros_like(c_loss)
            e_out = self.module.forward_exact(
                exact_input_ids,
                labels=exact_labels,
                attention_mask=exact_attention_mask,
            )
            e_loss = e_out["loss"]
            loss = concept_weight * c_loss + exact_weight * e_loss + align_weight * a_loss
            return {
                "loss": loss,
                "concept_loss": c_loss,
                "exact_loss": e_loss,
                "align_loss": a_loss,
                "exact_logits": e_out["logits"],
            }
        raise ValueError(f"Unknown Stage 2 DDP dispatch mode: {dispatch!r}")


class _SymmetricFakeQuantSTE(nn.Module):
    """Per-output-channel symmetric fake-quantization parametrization with STE."""

    def __init__(self, bits: int = 8) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"QAT bits must be 4 or 8, got {bits}")
        self.bits = int(bits)
        self.qmax = (1 << (self.bits - 1)) - 1  # 7 for INT4, 127 for INT8

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim < 2:
            return weight
        w_fp32 = weight.float()
        reduce_dims = tuple(range(1, w_fp32.ndim))
        max_abs = w_fp32.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-6)
        scale = max_abs / float(self.qmax)
        q = torch.clamp(torch.round(w_fp32 / scale), -self.qmax, self.qmax) * scale
        q = q.to(dtype=weight.dtype)
        # Straight-through estimator: forward uses fake-quantized weights, backward passes through.
        return weight + (q - weight).detach()


def _enable_qat_on_generator(model: CNCMModel, bits: int = 8) -> list[nn.Module]:
    """Attach STE QAT fake-quantization to Stage 2 linear layers (skipping frozen Stage 1)."""
    concept_tok_modules = set(model.concept_tokenizer.modules()) if hasattr(model, "concept_tokenizer") else set()
    parametrized: list[nn.Module] = []
    for mod in model.modules():
        if mod in concept_tok_modules:
            continue
        if hasattr(mod, "qat_enabled"):
            setattr(mod, "qat_enabled", True)
            continue
        if mod.__class__.__name__ == "BitQuantLinear":
            continue
        if isinstance(mod, nn.Linear) and mod.weight.requires_grad and not parametrize.is_parametrized(mod, "weight"):
            parametrize.register_parametrization(mod, "weight", _SymmetricFakeQuantSTE(bits=bits))
            parametrized.append(mod)
    return parametrized


def _disable_qat_parametrizations(modules: Sequence[nn.Module]) -> None:
    """Remove temporary QAT parametrizations while keeping the QAT-adapted FP32 weights."""
    for mod in modules:
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)


def _enforce_frozen_concept_tokenizer(model: CNCMModel) -> None:
    """Guarantee Stage 1 concept_tokenizer remains strictly frozen and in eval mode."""
    if hasattr(model, "freeze_concept_tokenizer") and callable(model.freeze_concept_tokenizer):
        model.freeze_concept_tokenizer()
    if hasattr(model, "concept_tokenizer") and model.concept_tokenizer is not None:
        for p in model.concept_tokenizer.parameters():
            p.requires_grad_(False)
        model.concept_tokenizer.eval()


def _prepare_stage2_batch(
    batch: dict[str, torch.Tensor],
    cfg: CNCMConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Normalize either BlockARSFTDataset or PackedChunkDataset/MemmapTokenDataset batches for Stage 2."""
    non_blocking = device.type == "cuda"
    if "prefix_ids" in batch and ("target_ids" in batch or "target_chunk_ids" in batch):
        prefix_ids = batch["prefix_ids"].to(device, non_blocking=non_blocking)
        prefix_mask = batch.get("prefix_mask")
        if prefix_mask is None:
            prefix_mask = (prefix_ids != cfg.pad_token_id).long()
        else:
            prefix_mask = prefix_mask.to(device, non_blocking=non_blocking)
        target_ids = batch.get("target_ids", batch.get("target_chunk_ids")).to(device, non_blocking=non_blocking)
        target_mask = batch.get("target_mask")
        if target_mask is None:
            target_mask = torch.ones_like(target_ids, dtype=torch.long)
        else:
            target_mask = target_mask.to(device, non_blocking=non_blocking)

        exact_input_ids = batch.get("exact_input_ids")
        if exact_input_ids is None:
            exact_input_ids = torch.cat([prefix_ids, target_ids], dim=1)
        else:
            exact_input_ids = exact_input_ids.to(device, non_blocking=non_blocking)

        exact_attention_mask = batch.get("exact_attention_mask")
        if exact_attention_mask is None:
            exact_attention_mask = torch.cat([prefix_mask, target_mask], dim=1)
        else:
            exact_attention_mask = exact_attention_mask.to(device, non_blocking=non_blocking)

        exact_labels = batch.get("exact_labels")
        if exact_labels is None:
            exact_labels = torch.full_like(exact_input_ids, -100)
            p_len = prefix_ids.shape[1]
            exact_labels[:, p_len:] = torch.where(
                target_mask.bool(),
                target_ids,
                torch.full_like(target_ids, -100),
            )
        else:
            exact_labels = exact_labels.to(device, non_blocking=non_blocking)
    else:
        # Adapt Stage-1 style PackedChunkDataset (`input_ids` only) with explicit BOS prefix.
        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        if input_ids.shape[1] > cfg.chunk_size:
            prefix_ids = input_ids[:, : -cfg.chunk_size]
            target_ids = input_ids[:, -cfg.chunk_size :]
        else:
            prefix_ids = torch.full(
                (input_ids.shape[0], 1),
                cfg.bos_token_id,
                dtype=torch.long,
                device=device,
            )
            target_ids = input_ids[:, : cfg.chunk_size]

        prefix_mask = torch.ones_like(prefix_ids, dtype=torch.long)
        target_mask = torch.ones_like(target_ids, dtype=torch.long)
        exact_input_ids = torch.cat([prefix_ids, target_ids], dim=1)
        exact_attention_mask = torch.cat([prefix_mask, target_mask], dim=1)
        exact_labels = torch.full_like(exact_input_ids, -100)
        exact_labels[:, prefix_ids.shape[1] :] = target_ids

    return {
        "prefix_ids": prefix_ids,
        "prefix_mask": prefix_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
        "exact_input_ids": exact_input_ids,
        "exact_attention_mask": exact_attention_mask,
        "exact_labels": exact_labels,
    }


def _compute_self_distill_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    exact_labels: torch.Tensor,
    temperature: float = 1.5,
) -> torch.Tensor:
    """Compute temperature-scaled KL divergence on shifted target positions (`labels != -100`)."""
    # Align with standard causal LM shift: logits[:, :-1] predicts labels[:, 1:]
    s_shift = student_logits[:, :-1, :].float()
    t_shift = teacher_logits[:, :-1, :].float()
    lbl_shift = exact_labels[:, 1:]
    valid = lbl_shift != -100

    if not valid.any():
        return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    s_sel = s_shift[valid] / float(temperature)
    t_sel = t_shift[valid] / float(temperature)
    log_p_s = F.log_softmax(s_sel, dim=-1)
    p_t = F.softmax(t_sel, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (float(temperature) ** 2)
    return kl.to(dtype=student_logits.dtype)


@torch.no_grad()
def _evaluate_stage2_val(
    model: CNCMModel,
    val_loader: DataLoader,
    cfg: CNCMConfig,
    dev: torch.device,
    *,
    mode: str,
    concept_weight: float,
    exact_weight: float,
    align_weight: float,
    level_weights_alpha: float,
    supervise_prefix_exact: bool,
    amp_dtype: Union[str, torch.dtype, None],
    max_batches: int = 16,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for raw_batch in val_loader:
        if n_batches >= max(1, int(max_batches)):
            break
        batch = _prepare_stage2_batch(raw_batch, cfg, dev)
        prefix_ids = batch["prefix_ids"]
        prefix_mask = batch["prefix_mask"]
        target_ids = batch["target_ids"]
        exact_input_ids = batch["exact_input_ids"]
        exact_attention_mask = batch["exact_attention_mask"]
        exact_labels = batch["exact_labels"]

        with amp_context(dev, dtype=amp_dtype, stage=2):
            if mode == "exact_ar":
                full_labels = exact_input_ids.clone()
                if exact_attention_mask is not None:
                    full_labels = full_labels.masked_fill(~exact_attention_mask.bool(), -100)
                loss = model.forward_exact(
                    exact_input_ids,
                    labels=full_labels,
                    attention_mask=exact_attention_mask,
                )["loss"]
            elif mode in ("joint", "pretrain") and (mode == "joint" or exact_weight > 0.0 or align_weight > 0.0):
                loss = model.forward_joint(
                    prefix_ids,
                    target_ids,
                    prefix_mask=prefix_mask,
                    concept_weight=concept_weight,
                    exact_weight=exact_weight,
                    align_weight=align_weight,
                    corrupt_rate=0.0,
                    exact_labels=exact_labels,
                    level_weights_alpha=level_weights_alpha,
                    supervise_prefix_exact=supervise_prefix_exact,
                )["loss"]
            elif mode == "pretrain":
                target_codes = model.concept_tokenizer.encode_codes(target_ids)
                c_loss = model.forward_concept(
                    prefix_ids,
                    target_codes,
                    prefix_mask=prefix_mask,
                    corrupt_rate=0.0,
                    level_weights_alpha=level_weights_alpha,
                )["loss"]
                loss = concept_weight * c_loss
            else:
                target_codes = model.concept_tokenizer.encode_codes(target_ids)
                c_out = model.forward_concept(
                    prefix_ids,
                    target_codes,
                    prefix_mask=prefix_mask,
                    corrupt_rate=0.0,
                    level_weights_alpha=level_weights_alpha,
                )
                c_loss = c_out["loss"]
                e_loss = model.forward_exact(
                    exact_input_ids,
                    labels=exact_labels,
                    attention_mask=exact_attention_mask,
                )["loss"]
                loss = concept_weight * c_loss + exact_weight * e_loss

        total_loss += _to_float(loss)
        n_batches += 1

    if was_training:
        model.train()
        _enforce_frozen_concept_tokenizer(model)

    return total_loss / float(n_batches) if n_batches > 0 else float("nan")


def train_stage2(
    cfg: CNCMConfig,
    tokenizer: CNCMTextTokenizer,
    concept_tokenizer: CNCMConceptTokenizer,
    dataset: Optional[Union[Dataset, Sequence[Any]]] = None,
    mode: str = "joint",
    steps: int = 50,
    corrupt_rate: float = 0.5,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: Union[str, torch.device] = "auto",
    *,
    concept_weight: float = 1.0,
    exact_weight: float = 0.3,
    align_weight: float = 0.3,
    supervise_prefix_exact: bool = False,
    level_weights_alpha: float = 0.5,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 1,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.08,
    min_lr_ratio: float = 0.1,
    self_distill: bool = False,
    self_distill_weight: float = 0.1,
    self_distill_temp: float = 1.5,
    qat_finetune: bool = False,
    qat_bits: int = 8,
    model: Optional[CNCMModel] = None,
    amp_dtype: Union[str, torch.dtype, None] = None,
    grad_checkpointing: bool = False,
    seed: int = 42,
    log_every: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    resume_from: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_every: int = 500,
    keep_last_k: int = 2,
    val_dataset: Optional[Dataset] = None,
    eval_every: int = 500,
    eval_max_batches: int = 16,
    max_wall_minutes: Optional[float] = None,
    hf_repo_id: Optional[str] = None,
    hf_upload_every: int = 1000,
    hf_path_prefix: str = "",
    num_workers: Optional[int] = None,
    auto_resume: bool = True,
    start_time_monotonic: Optional[float] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train Stage 2 `CNCMModel` with frozen Stage 1 `concept_tokenizer`.

    Supports T4 FP16 `GradScaler`, CUDA fused `AdamW`, activation gradient checkpointing,
    multi-GPU DDP (`torchrun`), rolling `CheckpointManager` resume/pruning, HF Hub sync,
    and Kaggle wall-clock graceful stopping.
    """
    if mode not in ("joint", "pretrain", "sft", "exact_ar"):
        raise ValueError(f"mode must be 'joint', 'pretrain', 'sft', or 'exact_ar'; got {mode!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    dist_info = get_dist_info()
    is_distributed = bool(dist_info["is_distributed"])
    rank = int(dist_info["rank"])
    local_rank = int(dist_info["local_rank"])
    world_size = int(dist_info["world_size"])
    is_main = bool(dist_info["is_main"])

    set_seed(seed + rank)
    if is_distributed and torch.cuda.is_available() and str(device) in ("auto", "cuda"):
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = select_device(device)

    safe_max_prefix = min(
        256,
        max(16, cfg.max_position_embeddings - cfg.total_concept_tokens - cfg.chunk_size),
    )

    # Build or wrap dataset
    if dataset is None:
        if mode in ("pretrain", "exact_ar"):
            corpus = make_synthetic_corpus(num_samples=256, seed=seed)
            train_ds: Dataset = PackedChunkDataset(
                corpus,
                tokenizer,
                chunk_size=cfg.chunk_size,
                prefix_len=min(cfg.chunk_size, safe_max_prefix),
                eos_token_id=cfg.eos_token_id,
                bos_token_id=cfg.bos_token_id,
            )
        else:
            sft_data = make_synthetic_sft(num_samples=128, seed=seed)
            train_ds = BlockARSFTDataset(
                sft_data,
                tokenizer,
                chunk_size=cfg.chunk_size,
                max_prefix_len=safe_max_prefix,
                pad_token_id=cfg.pad_token_id,
                bos_token_id=cfg.bos_token_id,
                eos_token_id=cfg.eos_token_id,
            )
    elif isinstance(dataset, Dataset):
        train_ds = dataset
    else:
        seq = list(dataset)
        if seq and isinstance(seq[0], dict):
            train_ds = BlockARSFTDataset(
                seq,
                tokenizer,
                chunk_size=cfg.chunk_size,
                max_prefix_len=safe_max_prefix,
                pad_token_id=cfg.pad_token_id,
                bos_token_id=cfg.bos_token_id,
                eos_token_id=cfg.eos_token_id,
            )
        else:
            train_ds = PackedChunkDataset(
                seq,
                tokenizer,
                chunk_size=cfg.chunk_size,
                prefix_len=min(cfg.chunk_size, safe_max_prefix),
                eos_token_id=cfg.eos_token_id,
                bos_token_id=cfg.bos_token_id,
            )

    if model is None:
        model = CNCMModel(cfg, concept_tokenizer)
    model = model.to(dev)
    _enforce_frozen_concept_tokenizer(model)
    prev_grad_ckpt = bool(getattr(model, "grad_checkpointing", False))
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(grad_checkpointing)

    # Snapshot frozen FP32 teacher before enabling QAT if self-distillation is requested.
    teacher_model: Optional[CNCMModel] = None
    if self_distill:
        teacher_model = copy.deepcopy(model).to(dev)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    qat_modules: list[nn.Module] = []
    if qat_finetune:
        qat_modules = _enable_qat_on_generator(model, bits=qat_bits)

    # Collect ONLY trainable generator parameters (strictly excluding frozen Stage 1)
    stage1_param_ids = {id(p) for p in model.concept_tokenizer.parameters()}
    trainable_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in stage1_param_ids
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in Stage 2 CNCMModel.")

    optimizer = make_adamw(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        device=dev,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    parsed_amp = _parse_dtype(amp_dtype)
    use_fp16_scaler = (get_device_type(dev) == "cuda") and (parsed_amp == torch.float16)
    scaler: Optional[Any] = torch.amp.GradScaler("cuda", enabled=True) if use_fp16_scaler else None

    ckpt_mgr: Optional[CheckpointManager] = None
    if save_dir is not None:
        ckpt_mgr = CheckpointManager(
            save_dir=save_dir,
            keep_last_k=keep_last_k,
            max_wall_minutes=max_wall_minutes,
            hf_repo_id=hf_repo_id,
            hf_upload_every=hf_upload_every,
            is_main_process=is_main,
            start_time_monotonic=start_time_monotonic,
            hf_path_prefix=hf_path_prefix,
        )
    wall_start = float(start_time_monotonic) if start_time_monotonic is not None else time.monotonic()

    # Resolve optional resume checkpoint (explicit path -> save_dir/latest.pt -> HF Hub)
    resume_path: Optional[Path] = None
    if ckpt_mgr is not None and (resume_from is not None or auto_resume):
        if is_main:
            resume_path = ckpt_mgr.resolve_resume_path(resume_from)
        if is_distributed:
            obj_list: list[Any] = [str(resume_path) if resume_path is not None else None]
            dist.broadcast_object_list(obj_list, src=0)
            resume_path = Path(obj_list[0]) if obj_list[0] is not None else None
    elif resume_from is not None and Path(resume_from).exists():
        resume_path = Path(resume_from)

    start_step = 0
    tokens_seen = 0
    data_cursor = 0
    best_val_loss: Optional[float] = None
    loss_history: list[float] = []
    concept_loss_history: list[float] = []
    exact_loss_history: list[float] = []
    align_loss_history: list[float] = []
    distill_loss_history: list[float] = []
    grad_norm_history: list[float] = []
    lr_history: list[float] = []
    val_loss_history: list[dict[str, Any]] = []
    fp16_overflow_steps = 0
    all_grads_finite = True

    if resume_path is not None and resume_path.exists():
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=dev,
            restore_rng=True,
        )
        _enforce_frozen_concept_tokenizer(model)
        start_step = int(payload.get("step", 0))
        tokens_seen = int(payload.get("tokens_seen", 0))
        data_cursor = int(payload.get("data_cursor", 0))
        best_val_loss = payload.get("best_val_loss")
        if ckpt_mgr is not None and best_val_loss is not None:
            ckpt_mgr.best_val_loss = float(best_val_loss)
        if steps > scheduler.total_steps:
            scheduler.total_steps = int(steps)
            scheduler._apply_lr(scheduler.current_step)
        prev_metrics = payload.get("extra", {}).get("metrics", {})
        if isinstance(prev_metrics, dict):
            loss_history = list(prev_metrics.get("loss_history", []))
            concept_loss_history = list(prev_metrics.get("concept_loss_history", []))
            exact_loss_history = list(prev_metrics.get("exact_loss_history", []))
            align_loss_history = list(prev_metrics.get("align_loss_history", []))
            distill_loss_history = list(prev_metrics.get("distill_loss_history", []))
            grad_norm_history = list(prev_metrics.get("grad_norm_history", []))
            lr_history = list(prev_metrics.get("lr_history", []))
            val_loss_history = list(prev_metrics.get("val_loss_history", []))
        if verbose and is_main:
            print(
                f"[Stage 2:{mode}] Resumed from {resume_path} at step {start_step}/{steps} "
                f"(tokens_seen={tokens_seen:,}, data_cursor={data_cursor:,})"
            )

    # Shard dataset & apply deterministic offset slicing for resume/DDP
    active_ds: Dataset = train_ds
    sampler: Optional[DistributedSampler] = None
    is_memmap = isinstance(train_ds, MemmapTokenDataset)
    if is_memmap:
        active_ds = train_ds.with_offset(start_offset=data_cursor, rank=rank, world_size=world_size)
        shuffle_loader = False
    else:
        if data_cursor > 0 and len(train_ds) > 1:
            offset_mod = data_cursor % len(train_ds)
            if offset_mod > 0:
                indices = list(range(offset_mod, len(train_ds))) + list(range(0, offset_mod))
                active_ds = Subset(train_ds, indices)
        if is_distributed:
            sampler = DistributedSampler(
                active_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
            shuffle_loader = False
        else:
            shuffle_loader = True

    sample0 = active_ds[0]
    use_block_collate = isinstance(sample0, dict) and "prefix_ids" in sample0
    collate_fn = partial(collate_block_ar, pad_token_id=cfg.pad_token_id) if use_block_collate else None
    eff_bs = max(1, min(int(batch_size), len(active_ds)))
    loader = make_dataloader(
        active_ds,
        batch_size=eff_bs,
        shuffle=shuffle_loader,
        drop_last=False,
        collate_fn=collate_fn,
        device=dev,
        num_workers=num_workers,
        sampler=sampler,
    )
    data_iter = _infinite_loader(loader, sampler=sampler)

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None and is_main and len(val_dataset) > 0:
        val_sample0 = val_dataset[0]
        val_collate = (
            partial(collate_block_ar, pad_token_id=cfg.pad_token_id)
            if (isinstance(val_sample0, dict) and "prefix_ids" in val_sample0)
            else None
        )
        val_loader = make_dataloader(
            val_dataset,
            batch_size=max(1, min(int(batch_size), len(val_dataset))),
            shuffle=False,
            drop_last=False,
            collate_fn=val_collate,
            device=dev,
            num_workers=0,
        )

    ddp_exec: Optional[DDP] = None
    if is_distributed:
        ddp_wrapper = _Stage2DDPWrapper(model)
        ddp_kwargs: dict[str, Any] = {"find_unused_parameters": True}
        if dev.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        ddp_exec = DDP(ddp_wrapper, **ddp_kwargs)

    completed_step = start_step
    stopped_early_wall_clock = False

    def _build_metrics(cur_step: int, stopped_early: bool) -> dict[str, Any]:
        stage1_frozen_now = all(not p.requires_grad for p in model.concept_tokenizer.parameters())
        return {
            "stage": 2,
            "mode": mode,
            "steps": int(steps),
            "completed_steps": int(cur_step),
            "resumed_from_step": int(start_step),
            "tokens_seen": int(tokens_seen),
            "data_cursor": int(data_cursor),
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
            "stopped_early_wall_clock": bool(stopped_early),
            "initial_loss": loss_history[0] if loss_history else None,
            "final_loss": loss_history[-1] if loss_history else None,
            "loss_history": loss_history,
            "concept_loss_history": concept_loss_history,
            "exact_loss_history": exact_loss_history,
            "align_loss_history": align_loss_history,
            "distill_loss_history": distill_loss_history,
            "grad_norm_history": grad_norm_history,
            "lr_history": lr_history,
            "val_loss_history": val_loss_history,
            "all_grads_finite": all_grads_finite,
            "fp16_scaler_enabled": bool(use_fp16_scaler),
            "fp16_overflow_steps": int(fp16_overflow_steps),
            "grad_checkpointing": bool(grad_checkpointing),
            "concept_tokenizer_frozen": stage1_frozen_now,
            "self_distill": bool(self_distill),
            "qat_finetune": bool(qat_finetune),
            "params": count_parameters(model),
            "device": str(dev),
            "world_size": world_size,
        }

    def _should_stop_wall() -> bool:
        if ckpt_mgr is not None:
            return ckpt_mgr.should_stop()
        if max_wall_minutes is not None:
            return ((time.monotonic() - wall_start) / 60.0) >= float(max_wall_minutes)
        return False

    try:
        for step_idx in range(start_step + 1, steps + 1):
            local_stop = _should_stop_wall() if is_main else False
            if _sync_bool_flag(local_stop, is_distributed, dev):
                stopped_early_wall_clock = True
                snap = _build_metrics(completed_step, stopped_early=True)
                if ckpt_mgr is not None and is_main:
                    ckpt_mgr.save(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=completed_step,
                        config=cfg,
                        scaler=scaler,
                        tokens_seen=tokens_seen,
                        best_val_loss=best_val_loss,
                        data_cursor=data_cursor,
                        metrics=snap,
                        extra={"stage": 2, "mode": mode},
                        save_numbered=False,
                        force_hf_upload=True,
                    )
                    ckpt_mgr.wait_for_uploads()
                elif save_path is not None and is_main:
                    save_checkpoint(
                        save_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=completed_step,
                        config=cfg,
                        scaler=scaler,
                        tokens_seen=tokens_seen,
                        best_val_loss=best_val_loss,
                        data_cursor=data_cursor,
                        extra={"stage": 2, "mode": mode, "metrics": snap},
                    )
                if verbose and is_main:
                    print(
                        f"[Stage 2:{mode}] Wall-clock limit ({max_wall_minutes} min) reached before step {step_idx}; "
                        f"saved checkpoint at step {completed_step} and exiting cleanly."
                    )
                break

            model.train()
            _enforce_frozen_concept_tokenizer(model)
            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            step_concept = 0.0
            step_exact = 0.0
            step_align = 0.0
            step_distill = 0.0

            for mb_idx in range(grad_accum_steps):
                raw_batch = next(data_iter)
                batch = _prepare_stage2_batch(raw_batch, cfg, dev)
                prefix_ids = batch["prefix_ids"]
                prefix_mask = batch["prefix_mask"]
                target_ids = batch["target_ids"]
                exact_input_ids = batch["exact_input_ids"]
                exact_attention_mask = batch["exact_attention_mask"]
                exact_labels = batch["exact_labels"]

                sync_ctx = (
                    ddp_exec.no_sync()
                    if (ddp_exec is not None and mb_idx < grad_accum_steps - 1)
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    with amp_context(dev, dtype=amp_dtype, stage=2):
                        student_exact_logits: Optional[torch.Tensor] = None
                        a_loss = torch.zeros((), device=dev, dtype=torch.float32)

                        if mode == "exact_ar":
                            # Phase 1 Exact-AR Pretraining: supervise ALL non-pad tokens across the full window
                            # (prefix + target) so the shared backbone reaches full Pure-AR LM quality first.
                            full_labels = exact_input_ids.clone()
                            if exact_attention_mask is not None:
                                full_labels = full_labels.masked_fill(~exact_attention_mask.bool(), -100)
                            if ddp_exec is not None:
                                e_out = ddp_exec(
                                    "exact",
                                    exact_input_ids,
                                    labels=full_labels,
                                    attention_mask=exact_attention_mask,
                                )
                            else:
                                e_out = model.forward_exact(
                                    exact_input_ids,
                                    labels=full_labels,
                                    attention_mask=exact_attention_mask,
                                )
                            e_loss = e_out["loss"]
                            c_loss = torch.zeros_like(e_loss)
                            a_loss = torch.zeros_like(e_loss)
                            student_exact_logits = e_out["logits"]
                            loss = e_loss
                        elif mode == "joint":
                            if ddp_exec is not None:
                                out = ddp_exec(
                                    "joint",
                                    prefix_ids,
                                    target_ids,
                                    prefix_mask=prefix_mask,
                                    concept_weight=concept_weight,
                                    exact_weight=exact_weight,
                                    align_weight=align_weight,
                                    corrupt_rate=corrupt_rate,
                                    exact_labels=exact_labels,
                                    level_weights_alpha=level_weights_alpha,
                                    supervise_prefix_exact=supervise_prefix_exact,
                                )
                            else:
                                out = model.forward_joint(
                                    prefix_ids,
                                    target_ids,
                                    prefix_mask=prefix_mask,
                                    concept_weight=concept_weight,
                                    exact_weight=exact_weight,
                                    align_weight=align_weight,
                                    corrupt_rate=corrupt_rate,
                                    exact_labels=exact_labels,
                                    level_weights_alpha=level_weights_alpha,
                                    supervise_prefix_exact=supervise_prefix_exact,
                                )
                            loss = out["loss"]
                            c_loss = out.get("concept_loss", loss)
                            e_loss = out.get("exact_loss", torch.zeros_like(loss))
                            a_loss = out.get("align_loss", torch.zeros_like(loss))
                        elif mode == "pretrain":
                            if exact_weight > 0.0 or align_weight > 0.0:
                                if ddp_exec is not None:
                                    out = ddp_exec(
                                        "joint",
                                        prefix_ids,
                                        target_ids,
                                        prefix_mask=prefix_mask,
                                        concept_weight=concept_weight,
                                        exact_weight=exact_weight,
                                        align_weight=align_weight,
                                        corrupt_rate=corrupt_rate,
                                        exact_labels=exact_labels,
                                        level_weights_alpha=level_weights_alpha,
                                        supervise_prefix_exact=supervise_prefix_exact,
                                    )
                                else:
                                    out = model.forward_joint(
                                        prefix_ids,
                                        target_ids,
                                        prefix_mask=prefix_mask,
                                        concept_weight=concept_weight,
                                        exact_weight=exact_weight,
                                        align_weight=align_weight,
                                        corrupt_rate=corrupt_rate,
                                        exact_labels=exact_labels,
                                        level_weights_alpha=level_weights_alpha,
                                        supervise_prefix_exact=supervise_prefix_exact,
                                    )
                                loss = out["loss"]
                                c_loss = out.get("concept_loss", loss)
                                e_loss = out.get("exact_loss", torch.zeros_like(loss))
                                a_loss = out.get("align_loss", torch.zeros_like(loss))
                            else:
                                with torch.no_grad():
                                    target_codes = model.concept_tokenizer.encode_codes(target_ids)
                                if ddp_exec is not None:
                                    c_out = ddp_exec(
                                        "concept",
                                        prefix_ids,
                                        target_codes,
                                        prefix_mask=prefix_mask,
                                        corrupt_rate=corrupt_rate,
                                        level_weights_alpha=level_weights_alpha,
                                    )
                                else:
                                    c_out = model.forward_concept(
                                        prefix_ids,
                                        target_codes,
                                        prefix_mask=prefix_mask,
                                        corrupt_rate=corrupt_rate,
                                        level_weights_alpha=level_weights_alpha,
                                    )
                                c_loss = c_out["loss"]
                                e_loss = torch.zeros_like(c_loss)
                                a_loss = torch.zeros_like(c_loss)
                                loss = concept_weight * c_loss
                        else:  # mode == "sft"
                            if ddp_exec is not None:
                                sft_out = ddp_exec(
                                    "sft",
                                    prefix_ids=prefix_ids,
                                    prefix_mask=prefix_mask,
                                    target_ids=target_ids,
                                    exact_input_ids=exact_input_ids,
                                    exact_attention_mask=exact_attention_mask,
                                    exact_labels=exact_labels,
                                    corrupt_rate=corrupt_rate,
                                    level_weights_alpha=level_weights_alpha,
                                    concept_weight=concept_weight,
                                    exact_weight=exact_weight,
                                    align_weight=align_weight,
                                )
                                c_loss = sft_out["concept_loss"]
                                e_loss = sft_out["exact_loss"]
                                a_loss = sft_out["align_loss"]
                                student_exact_logits = sft_out["exact_logits"]
                                loss = sft_out["loss"]
                            else:
                                with torch.no_grad():
                                    target_codes = model.concept_tokenizer.encode_codes(target_ids)
                                c_out = model.forward_concept(
                                    prefix_ids,
                                    target_codes,
                                    prefix_mask=prefix_mask,
                                    corrupt_rate=corrupt_rate,
                                    level_weights_alpha=level_weights_alpha,
                                )
                                c_loss = c_out["loss"]
                                if align_weight > 0.0 and "h_target" in c_out:
                                    t_len = target_ids.shape[1]
                                    h_fine = c_out["h_target"][:, -t_len:, :]
                                    a_logits = model.exact_head(h_fine).float()
                                    a_tgt = target_ids.to(device=dev, dtype=torch.long).clone()
                                    non_pad_a = a_tgt != cfg.pad_token_id
                                    if non_pad_a.any():
                                        a_tgt = a_tgt.masked_fill(~non_pad_a, -100)
                                    a_loss = F.cross_entropy(
                                        a_logits.reshape(-1, cfg.vocab_size),
                                        a_tgt.reshape(-1),
                                        ignore_index=-100,
                                    )
                                else:
                                    a_loss = torch.zeros_like(c_loss)
                                e_out = model.forward_exact(
                                    exact_input_ids,
                                    labels=exact_labels,
                                    attention_mask=exact_attention_mask,
                                )
                                e_loss = e_out["loss"]
                                student_exact_logits = e_out["logits"]
                                loss = concept_weight * c_loss + exact_weight * e_loss + align_weight * a_loss

                        d_loss = torch.zeros((), device=dev, dtype=loss.dtype)
                        if self_distill and teacher_model is not None and self_distill_weight > 0.0:
                            if student_exact_logits is None:
                                if ddp_exec is not None:
                                    s_out = ddp_exec(
                                        "exact",
                                        exact_input_ids,
                                        attention_mask=exact_attention_mask,
                                    )
                                else:
                                    s_out = model.forward_exact(
                                        exact_input_ids,
                                        attention_mask=exact_attention_mask,
                                    )
                                student_exact_logits = s_out["logits"]
                            with torch.no_grad():
                                t_out = teacher_model.forward_exact(
                                    exact_input_ids,
                                    attention_mask=exact_attention_mask,
                                )
                            d_loss = _compute_self_distill_kl(
                                student_exact_logits,
                                t_out["logits"],
                                exact_labels=exact_labels,
                                temperature=self_distill_temp,
                            )
                            loss = loss + float(self_distill_weight) * d_loss

                        scaled_loss = loss / float(grad_accum_steps)

                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                step_loss += _to_float(loss) / float(grad_accum_steps)
                step_concept += _to_float(c_loss) / float(grad_accum_steps)
                step_exact += _to_float(e_loss) / float(grad_accum_steps)
                step_align += _to_float(a_loss) / float(grad_accum_steps)
                step_distill += _to_float(d_loss) / float(grad_accum_steps)
                tokens_seen += int(exact_attention_mask.sum().item()) * world_size
                data_cursor += int(prefix_ids.shape[0]) * world_size

            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm, grads_ok = clip_and_check_grad_norm(trainable_params, max_norm=grad_clip)
            if not grads_ok:
                if scaler is not None:
                    fp16_overflow_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    if float(scaler.get_scale()) <= 1.0:
                        all_grads_finite = False
                else:
                    all_grads_finite = False
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            completed_step = step_idx

            loss_history.append(step_loss)
            concept_loss_history.append(step_concept)
            exact_loss_history.append(step_exact)
            align_loss_history.append(step_align)
            distill_loss_history.append(step_distill)
            grad_norm_history.append(grad_norm)
            lr_history.append(current_lr)

            if verbose and is_main and (step_idx == start_step + 1 or step_idx % max(1, log_every) == 0 or step_idx == steps):
                extra_str = f", align={step_align:.4f}" if align_weight > 0.0 and mode != "exact_ar" else ""
                if self_distill:
                    extra_str += f", distill={step_distill:.4f}"
                print(
                    f"[Stage 2:{mode}] step {step_idx:4d}/{steps} | "
                    f"loss={step_loss:.4f} (concept={step_concept:.4f}, exact={step_exact:.4f}{extra_str}) | "
                    f"gnorm={grad_norm:.3f} | lr={current_lr:.2e}"
                )

            val_loss_this_step: Optional[float] = None
            do_eval = (
                val_loader is not None
                and eval_every > 0
                and (step_idx % eval_every == 0 or step_idx == steps)
            )
            if do_eval and is_main and val_loader is not None:
                val_loss_this_step = _evaluate_stage2_val(
                    model=model,
                    val_loader=val_loader,
                    cfg=cfg,
                    dev=dev,
                    mode=mode,
                    concept_weight=concept_weight,
                    exact_weight=exact_weight,
                    align_weight=align_weight,
                    level_weights_alpha=level_weights_alpha,
                    supervise_prefix_exact=supervise_prefix_exact,
                    amp_dtype=amp_dtype,
                    max_batches=eval_max_batches,
                )
                val_loss_history.append({"step": step_idx, "val_loss": val_loss_this_step})
                preview_best = (
                    val_loss_this_step
                    if (best_val_loss is None or val_loss_this_step < best_val_loss)
                    else best_val_loss
                )
                if ckpt_mgr is None:
                    best_val_loss = preview_best
                if verbose:
                    print(
                        f"[Stage 2:{mode} Val] step {step_idx:4d}/{steps} | "
                        f"val_loss={val_loss_this_step:.4f} | best={preview_best:.4f}"
                    )

            do_periodic_save = (
                ckpt_mgr is not None
                and is_main
                and (
                    ckpt_mgr.should_periodic_save(step_idx, save_every)
                    or val_loss_this_step is not None
                )
            )
            if do_periodic_save and ckpt_mgr is not None:
                prev_best = best_val_loss
                if val_loss_this_step is not None and (best_val_loss is None or val_loss_this_step < best_val_loss):
                    best_val_loss = val_loss_this_step
                snap = _build_metrics(completed_step, stopped_early=False)
                save_res = ckpt_mgr.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=completed_step,
                    config=cfg,
                    scaler=scaler,
                    tokens_seen=tokens_seen,
                    best_val_loss=prev_best,
                    data_cursor=data_cursor,
                    val_loss=val_loss_this_step,
                    metrics=snap,
                    extra={"stage": 2, "mode": mode},
                    save_numbered=True,
                    force_hf_upload=(step_idx == steps),
                )
                if save_res.get("best_val_loss") is not None:
                    best_val_loss = float(save_res["best_val_loss"])

            if is_distributed and (do_eval or do_periodic_save):
                dist.barrier()
    finally:
        if qat_modules:
            _disable_qat_parametrizations(qat_modules)
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(prev_grad_ckpt)
        _enforce_frozen_concept_tokenizer(model)
        if ckpt_mgr is not None and is_main:
            ckpt_mgr.wait_for_uploads()

    metrics = _build_metrics(completed_step, stopped_early=stopped_early_wall_clock)

    if ckpt_mgr is not None and is_main and not stopped_early_wall_clock:
        ckpt_mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=completed_step,
            config=cfg,
            scaler=scaler,
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
            data_cursor=data_cursor,
            metrics=metrics,
            extra={"stage": 2, "mode": mode},
            save_numbered=False,
            force_hf_upload=True,
        )
        ckpt_mgr.wait_for_uploads()

    if save_path is not None and is_main and not stopped_early_wall_clock:
        save_checkpoint(
            save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=completed_step,
            config=cfg,
            scaler=scaler,
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
            data_cursor=data_cursor,
            extra={"stage": 2, "mode": mode, "metrics": metrics},
        )

    return {"model": model, "metrics": metrics}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNCM Stage 2 Dual-Mode Generator.")
    parser.add_argument("--preset", type=str, default="tiny_test", choices=["tiny_test", "nano", "mini", "base", "base_moe"])
    parser.add_argument("--mode", type=str, default="joint", choices=["joint", "pretrain", "sft", "exact_ar"])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corrupt-rate", type=float, default=0.5)
    parser.add_argument("--concept-weight", type=float, default=1.0)
    parser.add_argument("--exact-weight", type=float, default=0.3)
    parser.add_argument("--align-weight", type=float, default=0.3)
    parser.add_argument("--supervise-prefix-exact", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp-dtype", type=str, default=None, choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--self-distill", action="store_true")
    parser.add_argument("--qat-finetune", action="store_true")
    parser.add_argument("--qat-bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--stage1-ckpt", type=str, default=None, help="Optional Stage 1 checkpoint path.")
    parser.add_argument("--data", type=str, default=None, help="Optional .json/.jsonl/.txt/.bin data path.")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--keep-last-k", type=int, default=2)
    parser.add_argument("--max-wall-minutes", type=float, default=None)
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-upload-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = get_preset(args.preset)
    tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    concept_tokenizer = CNCMConceptTokenizer(cfg)
    if args.stage1_ckpt:
        load_checkpoint(args.stage1_ckpt, model=concept_tokenizer, map_location="cpu")

    if args.data:
        if str(args.data).lower().endswith(".bin"):
            dataset: Dataset = MemmapTokenDataset(args.data, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
        elif args.mode in ("pretrain", "exact_ar"):
            dataset = PackedChunkDataset(
                args.data, tokenizer, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size
            )
        else:
            dataset = BlockARSFTDataset(args.data, tokenizer, chunk_size=cfg.chunk_size)
    else:
        if args.mode in ("pretrain", "exact_ar"):
            dataset = PackedChunkDataset(
                make_synthetic_corpus(num_samples=args.num_samples, seed=args.seed),
                tokenizer,
                chunk_size=cfg.chunk_size,
                prefix_len=cfg.chunk_size,
            )
        else:
            dataset = BlockARSFTDataset(
                make_synthetic_sft(num_samples=args.num_samples, seed=args.seed),
                tokenizer,
                chunk_size=cfg.chunk_size,
            )

    result = train_stage2(
        cfg=cfg,
        tokenizer=tokenizer,
        concept_tokenizer=concept_tokenizer,
        dataset=dataset,
        mode=args.mode,
        steps=args.steps,
        corrupt_rate=args.corrupt_rate,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        concept_weight=args.concept_weight,
        exact_weight=args.exact_weight,
        align_weight=args.align_weight,
        supervise_prefix_exact=args.supervise_prefix_exact,
        grad_accum_steps=args.grad_accum,
        grad_clip=args.grad_clip,
        amp_dtype=args.amp_dtype,
        grad_checkpointing=args.grad_checkpointing,
        self_distill=args.self_distill,
        qat_finetune=args.qat_finetune,
        qat_bits=args.qat_bits,
        seed=args.seed,
        log_every=args.log_every,
        save_path=args.save_path,
        save_dir=args.save_dir,
        resume_from=args.resume_from,
        save_every=args.save_every,
        eval_every=args.eval_every,
        keep_last_k=args.keep_last_k,
        max_wall_minutes=args.max_wall_minutes,
        hf_repo_id=args.hf_repo_id,
        hf_upload_every=args.hf_upload_every,
        verbose=not args.quiet,
    )
    if not args.quiet:
        m = result["metrics"]
        init_l = m["initial_loss"] if m["initial_loss"] is not None else float("nan")
        fin_l = m["final_loss"] if m["final_loss"] is not None else float("nan")
        print(
            f"[Stage 2 Done] mode={m['mode']} | loss {init_l:.4f} -> {fin_l:.4f} | "
            f"stage1_frozen={m['concept_tokenizer_frozen']}"
        )


if __name__ == "__main__":
    main()
