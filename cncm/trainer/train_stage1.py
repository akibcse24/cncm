"""Stage 1 training pipeline for CNCM Concept Tokenizer (Multiscale RFSQ).

Trains `CNCMConceptTokenizer` on padding-free fixed-length chunks (`PackedChunkDataset`) with:
- Nested fine-level dropout (`apply_fine_drop=True`) so coarse levels remain independently useful.
- Warmup + cosine LR schedule (`WarmupCosineScheduler`).
- Gradient accumulation and gradient clipping (`max_norm=1.0`) with finiteness checks.
- Strict FP32/INT64 safety (never FP16 autocast or `.half()` checkpoint corruption).
- Per-step tracking of exact-token reconstruction accuracy and active FSQ code utilization.

CLI usage:
    python -m cncm.trainer.train_stage1 --preset tiny_test --steps 30 --batch-size 8
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from cncm.config import CNCMConfig, get_preset
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer.dataset import MemmapTokenDataset, PackedChunkDataset, make_synthetic_corpus
from cncm.trainer.train_utils import (
    CheckpointManager,
    WarmupCosineScheduler,
    amp_context,
    clip_and_check_grad_norm,
    count_parameters,
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


@torch.no_grad()
def _evaluate_stage1_val(
    model: CNCMConceptTokenizer,
    val_loader: DataLoader,
    cfg: CNCMConfig,
    dev: torch.device,
    max_batches: int = 16,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    non_blocking = dev.type == "cuda"

    for batch in val_loader:
        if n_batches >= max(1, int(max_batches)):
            break
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
        else:
            input_ids = batch
            attention_mask = None

        if input_ids.shape[1] > cfg.chunk_size:
            input_ids = input_ids[:, : cfg.chunk_size]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : cfg.chunk_size]

        input_ids = input_ids.to(dev, non_blocking=non_blocking)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(dev, non_blocking=non_blocking)

        with amp_context(dev, dtype=None, stage=1):
            out = model(input_ids, attention_mask=attention_mask, apply_fine_drop=False)
            loss = out["loss"]
            logits = out["logits"]
            preds = logits.argmax(dim=-1)
            valid = attention_mask.bool()
            acc = (preds[valid] == input_ids[valid]).float().mean().item() if valid.any() else 0.0

        total_loss += _to_float(loss)
        total_acc += float(acc)
        n_batches += 1

    if was_training:
        model.train()

    if n_batches == 0:
        return {"val_loss": float("nan"), "val_recon_acc": 0.0}
    return {
        "val_loss": total_loss / float(n_batches),
        "val_recon_acc": total_acc / float(n_batches),
    }


def train_stage1(
    cfg: CNCMConfig,
    tokenizer: CNCMTextTokenizer,
    dataset: Optional[Union[Dataset, Sequence[str]]] = None,
    steps: int = 50,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: Union[str, torch.device] = "auto",
    *,
    model: Optional[CNCMConceptTokenizer] = None,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 1,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.08,
    min_lr_ratio: float = 0.1,
    apply_fine_drop: bool = True,
    amp_dtype: Union[str, torch.dtype, None] = None,
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
    """Train Stage 1 `CNCMConceptTokenizer` (strictly FP32-safe) with rolling resume & wall-clock guard."""
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
    non_blocking = dev.type == "cuda"

    if dataset is None:
        corpus = make_synthetic_corpus(num_samples=256, seed=seed)
        train_ds: Dataset = PackedChunkDataset(
            corpus,
            tokenizer,
            chunk_size=cfg.chunk_size,
            eos_token_id=cfg.eos_token_id,
            bos_token_id=cfg.bos_token_id,
        )
    elif isinstance(dataset, Dataset):
        train_ds = dataset
    else:
        train_ds = PackedChunkDataset(
            dataset,
            tokenizer,
            chunk_size=cfg.chunk_size,
            eos_token_id=cfg.eos_token_id,
            bos_token_id=cfg.bos_token_id,
        )

    if model is None:
        model = CNCMConceptTokenizer(cfg)
    model = model.to(dev)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
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
    recon_loss_history: list[float] = []
    commit_loss_history: list[float] = []
    recon_acc_history: list[float] = []
    active_code_ratio_history: list[float] = []
    grad_norm_history: list[float] = []
    lr_history: list[float] = []
    val_loss_history: list[dict[str, Any]] = []
    all_grads_finite = True

    if resume_path is not None and resume_path.exists():
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=dev,
            restore_rng=True,
        )
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
            recon_loss_history = list(prev_metrics.get("recon_loss_history", []))
            commit_loss_history = list(prev_metrics.get("commit_loss_history", []))
            recon_acc_history = list(prev_metrics.get("recon_acc_history", []))
            active_code_ratio_history = list(prev_metrics.get("active_code_ratio_history", []))
            grad_norm_history = list(prev_metrics.get("grad_norm_history", []))
            lr_history = list(prev_metrics.get("lr_history", []))
            val_loss_history = list(prev_metrics.get("val_loss_history", []))
        if verbose and is_main:
            print(
                f"[Stage 1] Resumed from {resume_path} at step {start_step}/{steps} "
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

    eff_bs = max(1, min(int(batch_size), len(active_ds)))
    loader = make_dataloader(
        active_ds,
        batch_size=eff_bs,
        shuffle=shuffle_loader,
        drop_last=False,
        device=dev,
        num_workers=num_workers,
        sampler=sampler,
    )
    data_iter = _infinite_loader(loader, sampler=sampler)

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None and is_main and len(val_dataset) > 0:
        val_loader = make_dataloader(
            val_dataset,
            batch_size=max(1, min(int(batch_size), len(val_dataset))),
            shuffle=False,
            drop_last=False,
            device=dev,
            num_workers=0,
        )

    exec_model: torch.nn.Module = model
    if is_distributed:
        ddp_kwargs: dict[str, Any] = {"find_unused_parameters": True}
        if dev.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        exec_model = DDP(model, **ddp_kwargs)

    completed_step = start_step
    stopped_early_wall_clock = False

    def _build_metrics(cur_step: int, stopped_early: bool) -> dict[str, Any]:
        return {
            "stage": 1,
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
            "recon_loss_history": recon_loss_history,
            "commit_loss_history": commit_loss_history,
            "recon_acc_history": recon_acc_history,
            "final_recon_acc": recon_acc_history[-1] if recon_acc_history else 0.0,
            "active_code_ratio_history": active_code_ratio_history,
            "final_active_code_ratio": active_code_ratio_history[-1] if active_code_ratio_history else 0.0,
            "grad_norm_history": grad_norm_history,
            "lr_history": lr_history,
            "val_loss_history": val_loss_history,
            "all_grads_finite": all_grads_finite,
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
                        tokens_seen=tokens_seen,
                        best_val_loss=best_val_loss,
                        data_cursor=data_cursor,
                        metrics=snap,
                        extra={"stage": 1},
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
                        tokens_seen=tokens_seen,
                        best_val_loss=best_val_loss,
                        data_cursor=data_cursor,
                        extra={"stage": 1, "metrics": snap},
                    )
                if verbose and is_main:
                    print(
                        f"[Stage 1] Wall-clock limit ({max_wall_minutes} min) reached before step {step_idx}; "
                        f"saved checkpoint at step {completed_step} and exiting cleanly."
                    )
                break

            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            step_recon_loss = 0.0
            step_commit_loss = 0.0
            step_recon_acc = 0.0
            step_code_ratio = 0.0

            for mb_idx in range(grad_accum_steps):
                batch = next(data_iter)
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask")
                else:
                    input_ids = batch
                    attention_mask = None

                if input_ids.shape[1] > cfg.chunk_size:
                    input_ids = input_ids[:, : cfg.chunk_size]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, : cfg.chunk_size]

                input_ids = input_ids.to(dev, non_blocking=non_blocking)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                else:
                    attention_mask = attention_mask.to(dev, non_blocking=non_blocking)

                sync_ctx = (
                    exec_model.no_sync()
                    if (is_distributed and mb_idx < grad_accum_steps - 1 and hasattr(exec_model, "no_sync"))
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    # Stage 1 ALWAYS runs in FP32-safe mode (stage=1 blocks FP16).
                    with amp_context(dev, dtype=amp_dtype, stage=1):
                        out = exec_model(
                            input_ids,
                            attention_mask=attention_mask,
                            apply_fine_drop=apply_fine_drop,
                        )
                        loss = out["loss"]
                        scaled_loss = loss / float(grad_accum_steps)

                    scaled_loss.backward()

                with torch.no_grad():
                    logits = out["logits"]
                    preds = logits.argmax(dim=-1)
                    valid = attention_mask.bool()
                    if valid.any():
                        acc = (preds[valid] == input_ids[valid]).float().mean().item()
                    else:
                        acc = 0.0

                    step_loss += _to_float(loss) / float(grad_accum_steps)
                    step_recon_loss += _to_float(out.get("recon_loss")) / float(grad_accum_steps)
                    step_commit_loss += _to_float(out.get("commit_loss")) / float(grad_accum_steps)
                    step_recon_acc += acc / float(grad_accum_steps)
                    step_code_ratio += _to_float(out.get("active_code_ratio")) / float(grad_accum_steps)
                    tokens_seen += int(attention_mask.sum().item()) * world_size
                    data_cursor += int(input_ids.shape[0]) * world_size

            grad_norm, grads_ok = clip_and_check_grad_norm(trainable_params, max_norm=grad_clip)
            if not grads_ok:
                all_grads_finite = False
            else:
                optimizer.step()

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            completed_step = step_idx

            loss_history.append(step_loss)
            recon_loss_history.append(step_recon_loss)
            commit_loss_history.append(step_commit_loss)
            recon_acc_history.append(step_recon_acc)
            active_code_ratio_history.append(step_code_ratio)
            grad_norm_history.append(grad_norm)
            lr_history.append(current_lr)

            if verbose and is_main and (step_idx == start_step + 1 or step_idx % max(1, log_every) == 0 or step_idx == steps):
                print(
                    f"[Stage 1] step {step_idx:4d}/{steps} | "
                    f"loss={step_loss:.4f} (recon={step_recon_loss:.4f}, commit={step_commit_loss:.4f}) | "
                    f"recon_acc={step_recon_acc * 100:5.1f}% | "
                    f"fsq_active={step_code_ratio * 100:5.1f}% | "
                    f"gnorm={grad_norm:.3f} | lr={current_lr:.2e}"
                )

            val_loss_this_step: Optional[float] = None
            do_eval = (
                val_loader is not None
                and eval_every > 0
                and (step_idx % eval_every == 0 or step_idx == steps)
            )
            if do_eval and is_main and val_loader is not None:
                val_res = _evaluate_stage1_val(model, val_loader, cfg, dev, max_batches=eval_max_batches)
                val_loss_this_step = val_res["val_loss"]
                val_loss_history.append(
                    {"step": step_idx, "val_loss": val_loss_this_step, "val_recon_acc": val_res["val_recon_acc"]}
                )
                if ckpt_mgr is None and (best_val_loss is None or val_loss_this_step < best_val_loss):
                    best_val_loss = val_loss_this_step
                if verbose:
                    print(
                        f"[Stage 1 Val] step {step_idx:4d}/{steps} | "
                        f"val_loss={val_loss_this_step:.4f} | val_recon_acc={val_res['val_recon_acc'] * 100:5.1f}%"
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
                    tokens_seen=tokens_seen,
                    best_val_loss=prev_best,
                    data_cursor=data_cursor,
                    val_loss=val_loss_this_step,
                    metrics=snap,
                    extra={"stage": 1},
                    save_numbered=True,
                    force_hf_upload=(step_idx == steps),
                )
                if save_res.get("best_val_loss") is not None:
                    best_val_loss = float(save_res["best_val_loss"])

            if is_distributed and (do_eval or do_periodic_save):
                dist.barrier()
    finally:
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
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
            data_cursor=data_cursor,
            metrics=metrics,
            extra={"stage": 1},
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
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
            data_cursor=data_cursor,
            extra={"stage": 1, "metrics": metrics},
        )

    return {"model": model, "metrics": metrics}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNCM Stage 1 Concept Tokenizer (M-RFSQ).")
    parser.add_argument("--preset", type=str, default="tiny_test", choices=["tiny_test", "nano", "mini", "base", "base_moe"])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data", type=str, default=None, help="Optional .txt, .jsonl, or .bin memmap corpus path.")
    parser.add_argument("--num-samples", type=int, default=256, help="Synthetic corpus size if --data is omitted.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional checkpoint output path.")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional rolling checkpoint directory.")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--keep-last-k", type=int, default=2)
    parser.add_argument("--max-wall-minutes", type=float, default=None)
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-upload-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-fine-drop", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = get_preset(args.preset)
    tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    if args.data:
        if str(args.data).lower().endswith(".bin"):
            dataset: Dataset = MemmapTokenDataset(args.data, chunk_size=cfg.chunk_size, prefix_len=0)
        else:
            dataset = PackedChunkDataset(args.data, tokenizer, chunk_size=cfg.chunk_size)
    else:
        corpus = make_synthetic_corpus(num_samples=args.num_samples, seed=args.seed)
        dataset = PackedChunkDataset(corpus, tokenizer, chunk_size=cfg.chunk_size)

    result = train_stage1(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset=dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum,
        grad_clip=args.grad_clip,
        apply_fine_drop=not args.no_fine_drop,
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
            f"[Stage 1 Done] loss {init_l:.4f} -> {fin_l:.4f} | "
            f"recon_acc={m['final_recon_acc'] * 100:.1f}% | "
            f"fsq_active={m['final_active_code_ratio'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
