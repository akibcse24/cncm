"""Training utilities for CNCM (Compact Causal-Hybrid Next-Concept Model).

Fixes key MiniMind & NCM engineering pitfalls:
- WarmupCosineScheduler: linear warmup + cosine decay to 0.1 * lr (fixing MiniMind's 0-warmup instability).
- select_device & amp_context: portable across CUDA, DirectML (Windows iGPU), MPS, and CPU while
  refusing FP16 overflow on Stage 1 M-RFSQ / Concept Tokenizer.
- save_checkpoint & load_checkpoint: atomic checkpointing that strictly preserves FP32 and INT64
  parameter/buffer dtypes (NEVER casting with `.half()`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import math
import os
import random
import re
import shutil
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts for a module."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def get_device_type(device: Union[torch.device, str]) -> str:
    """Normalize a torch.device or string into 'cuda', 'directml', 'mps', or 'cpu'."""
    dev_str = str(device).lower()
    if dev_str.startswith("cuda"):
        return "cuda"
    if dev_str.startswith("privateuseone") or dev_str in ("directml", "dml"):
        return "directml"
    if dev_str.startswith("mps"):
        return "mps"
    return "cpu"


def select_device(preferred: Union[str, torch.device] = "auto") -> torch.device:
    """Select the best available compute device across CUDA, DirectML, MPS, and CPU.

    Args:
        preferred: "auto", "cuda", "cuda:0", "directml"/"dml", "mps", "cpu", or a torch.device.
    """
    if isinstance(preferred, torch.device):
        return preferred

    pref = str(preferred).strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml  # type: ignore[import-not-found]

            if hasattr(torch_directml, "is_available"):
                if torch_directml.is_available():
                    return torch_directml.device()
            elif hasattr(torch_directml, "device_count") and torch_directml.device_count() > 0:
                return torch_directml.device()
        except Exception:
            pass
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref in ("directml", "dml"):
        try:
            import torch_directml  # type: ignore[import-not-found]

            return torch_directml.device()
        except ImportError as exc:
            raise RuntimeError(
                "DirectML requested but `torch-directml` is not installed. "
                "Install it with `pip install torch-directml` or pass device='cpu'."
            ) from exc

    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.", stacklevel=2)
            return torch.device("cpu")
        return torch.device(pref)

    if pref.startswith("mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            warnings.warn("MPS requested but not available; falling back to CPU.", stacklevel=2)
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device(pref)


def _parse_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).strip().lower()
    mapping: dict[str, Optional[torch.dtype]] = {
        "auto": None,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported AMP dtype: {dtype!r}")
    return mapping[key]


def _has_native_bf16(device: Union[torch.device, str]) -> bool:
    """Return True only on Ampere+ (SM >= 8.0) GPUs with hardware BF16 (excludes slow T4 emulation)."""
    if get_device_type(device) != "cuda" or not torch.cuda.is_available():
        return False
    try:
        idx = device.index if isinstance(device, torch.device) and device.index is not None else torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(idx)
        return int(major) >= 8 and bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


@contextlib.contextmanager
def amp_context(
    device: Union[torch.device, str],
    dtype: Union[str, torch.dtype, None] = None,
    stage: int = 2,
) -> Iterator[None]:
    """Safe autocast context manager across CUDA, DirectML, MPS, and CPU.

    Safety invariants:
    - Stage 1 (Concept Tokenizer / M-RFSQ) NEVER runs in FP16 to prevent STE / norm overflow.
      By default Stage 1 stays in exact FP32 unless BF16 is explicitly requested on supported HW.
    - Stage 2 / GRPO default to BF16 on native Ampere+ GPUs (SM >= 8.0), and FP32 elsewhere unless
      FP16 is explicitly requested (e.g., T4 SM 7.5 with GradScaler).
    """
    dev_type = get_device_type(device)
    parsed = _parse_dtype(dtype)

    if stage == 1:
        if parsed == torch.float16:
            warnings.warn(
                "FP16 AMP is disabled for Stage 1 Concept Tokenizer to prevent overflow; using FP32.",
                stacklevel=2,
            )
            parsed = torch.float32
        elif parsed is None:
            parsed = torch.float32

    if parsed is None:
        if _has_native_bf16(device):
            parsed = torch.bfloat16
        else:
            parsed = torch.float32

    if parsed == torch.float32:
        yield
        return

    if dev_type == "cuda":
        if parsed == torch.bfloat16 and not _has_native_bf16(device):
            yield
            return
        with torch.amp.autocast(device_type="cuda", dtype=parsed):
            yield
        return

    if dev_type == "cpu" and parsed == torch.bfloat16:
        try:
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                yield
            return
        except Exception:
            yield
            return

    # DirectML and MPS do not universally support PyTorch AMP autocast safely; run FP32.
    yield


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay down to `min_lr_ratio * base_lr`.

    Fixes MiniMind's zero-warmup bug while ensuring step 0 always has a non-zero learning rate
    (even on very short smoke-test runs).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.08,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        if warmup_steps is None:
            if self.total_steps <= 1:
                self.warmup_steps = 0
            else:
                self.warmup_steps = max(1, int(round(self.total_steps * warmup_ratio)))
                self.warmup_steps = min(self.warmup_steps, self.total_steps - 1)
        else:
            self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps))
        self.min_lr_ratio = float(min_lr_ratio)

        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.current_step = 0
        self._last_lrs: list[float] = []
        self._apply_lr(0)

    def lr_factor(self, step: int) -> float:
        """Compute multiplicative LR factor for 0-based `step`."""
        s = max(0, int(step))
        if self.warmup_steps > 0 and s < self.warmup_steps:
            return float(s + 1) / float(self.warmup_steps)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, float(s - self.warmup_steps) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_lr(self, step: int) -> list[float]:
        factor = self.lr_factor(step)
        lrs = [base * factor for base in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._last_lrs = lrs
        return lrs

    def step(self, step_idx: Optional[int] = None) -> list[float]:
        """Advance scheduler by 1 step (or jump to `step_idx`) and update optimizer LRs."""
        if step_idx is None:
            self.current_step += 1
        else:
            self.current_step = int(step_idx)
        return self._apply_lr(self.current_step)

    def get_lr(self) -> list[float]:
        return self._apply_lr(self.current_step)

    def get_last_lr(self) -> list[float]:
        return list(self._last_lrs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.min_lr_ratio = float(state["min_lr_ratio"])
        self.base_lrs = [float(x) for x in state["base_lrs"]]
        self.current_step = int(state["current_step"])
        self._apply_lr(self.current_step)


def clip_and_check_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float = 1.0,
) -> tuple[float, bool]:
    """Clip gradient norm and verify every gradient tensor is finite.

    Returns:
        (pre_clip_grad_norm, all_grads_finite)
    """
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0, True

    all_finite = True
    for p in params:
        if not torch.isfinite(p.grad).all():
            all_finite = False
            break

    if not all_finite:
        return float("inf"), False

    if max_norm is not None and max_norm > 0.0:
        norm_tensor = torch.nn.utils.clip_grad_norm_(params, max_norm)
        grad_norm = float(norm_tensor.item() if isinstance(norm_tensor, torch.Tensor) else norm_tensor)
    else:
        total_sq = sum(float(p.grad.detach().float().pow(2).sum().item()) for p in params)
        grad_norm = math.sqrt(total_sq)

    return grad_norm, math.isfinite(grad_norm)


def _serialize_config(config: Any) -> Any:
    if config is None:
        return None
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return config.to_dict()
    return config


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DistributedDataParallel / DataParallel / dispatch wrappers to the underlying module."""
    cur = model
    while hasattr(cur, "module") and isinstance(getattr(cur, "module"), nn.Module):
        cur = getattr(cur, "module")
    return cur


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG states for exact resume."""
    cuda_states: Optional[list[torch.Tensor]] = None
    if torch.cuda.is_available():
        try:
            cuda_states = [s.detach().to(device="cpu", dtype=torch.uint8).clone() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            cuda_states = None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state() if np is not None else None,
        "torch_cpu": torch.get_rng_state().detach().to(device="cpu", dtype=torch.uint8).clone(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(state: Optional[dict[str, Any]]) -> None:
    """Restore Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG states safely."""
    if not state or not isinstance(state, dict):
        return
    if state.get("python") is not None:
        try:
            random.setstate(state["python"])
        except Exception:
            pass
    if np is not None and state.get("numpy") is not None:
        try:
            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if state.get("torch_cpu") is not None:
        try:
            cpu_rng = state["torch_cpu"]
            if isinstance(cpu_rng, torch.Tensor):
                torch.set_rng_state(cpu_rng.detach().to(device="cpu", dtype=torch.uint8))
        except Exception:
            pass
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        try:
            cuda_states = state["torch_cuda"]
            if isinstance(cuda_states, (list, tuple)):
                n_dev = torch.cuda.device_count()
                if len(cuda_states) == n_dev:
                    torch.cuda.set_rng_state_all(
                        [s.detach().to(device="cpu", dtype=torch.uint8) for s in cuda_states]
                    )
                elif len(cuda_states) > 0:
                    for idx in range(min(n_dev, len(cuda_states))):
                        torch.cuda.set_rng_state(
                            cuda_states[idx].detach().to(device="cpu", dtype=torch.uint8),
                            device=idx,
                        )
        except Exception:
            pass


def get_dist_info() -> dict[str, Any]:
    """Return current distributed training topology (works with or without torchrun)."""
    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count() or 1)))
        return {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "is_distributed": world_size > 1,
            "is_main": rank == 0,
        }
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": max(1, world_size),
        "is_distributed": False,
        "is_main": rank == 0,
    }


def setup_ddp_if_needed(backend: Optional[str] = None, timeout_minutes: int = 60) -> dict[str, Any]:
    """Auto-initialize `torch.distributed` when launched via `torchrun` (`WORLD_SIZE > 1`).

    Safe no-op when `WORLD_SIZE == 1` (single-GPU or CPU).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        chosen = backend
        if chosen is None:
            chosen = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=chosen,
            timeout=datetime.timedelta(minutes=max(5, int(timeout_minutes))),
        )
    return get_dist_info()


def cleanup_ddp() -> None:
    """Cleanly tear down `torch.distributed` if initialized."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def make_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    device: Union[torch.device, str] = "cpu",
) -> torch.optim.AdamW:
    """Create `AdamW(..., fused=True)` on CUDA with transparent fallback elsewhere."""
    param_list = list(params)
    if get_device_type(device) == "cuda":
        try:
            return torch.optim.AdamW(
                param_list,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                fused=True,
            )
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.optim.AdamW(
        param_list,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable[..., Any]] = None,
    device: Union[torch.device, str] = "cpu",
    num_workers: Optional[int] = None,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """Build a platform-optimal DataLoader (fast `num_workers=2, pin_memory=True, persistent_workers=True` on Linux/CUDA)."""
    is_cuda = get_device_type(device) == "cuda"
    is_memmap_ds = hasattr(dataset, "_ensure_memmap")
    if num_workers is None:
        # Memmap slices read straight from OS page cache (<0.05ms) and avoid fork-after-CUDA hazards under DDP
        resolved_workers = 0 if is_memmap_ds else (2 if (os.name == "posix" and is_cuda) else 0)
    else:
        resolved_workers = max(0, int(num_workers))

    use_pin = bool(is_cuda)
    use_persistent = resolved_workers > 0
    eff_shuffle = False if sampler is not None else bool(shuffle)

    kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": eff_shuffle,
        "sampler": sampler,
        "drop_last": bool(drop_last),
        "collate_fn": collate_fn,
        "num_workers": resolved_workers,
        "pin_memory": use_pin,
    }
    if use_persistent:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def resolve_hf_token() -> Optional[str]:
    """Read `HF_TOKEN` from environment variables or Kaggle Secrets without logging it."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        secret = UserSecretsClient().get_secret("HF_TOKEN")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def _atomic_copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dst.name}.",
        suffix=".tmp",
        dir=str(dst.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return dst


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    config: Any = None,
    extra: Optional[dict[str, Any]] = None,
    *,
    scaler: Optional[Any] = None,
    tokens_seen: int = 0,
    best_val_loss: Optional[float] = None,
    data_cursor: int = 0,
    save_rng: bool = True,
) -> Path:
    """Atomically save a checkpoint while preserving exact FP32 / INT64 tensor dtypes.

    CRITICAL: Never casts tensors via `.half()`, so Stage 1 FSQ / norm / RoPE / index buffers
    remain bit-exact across save/load round-trips.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Detach and move to CPU WITHOUT altering dtype!
    raw_model = unwrap_model(model)
    raw_sd = raw_model.state_dict()
    state_dict: dict[str, torch.Tensor] = {}
    tensor_dtypes: dict[str, str] = {}
    for name, tensor in raw_sd.items():
        cpu_tensor = tensor.detach().to(device="cpu").clone()
        state_dict[name] = cpu_tensor
        tensor_dtypes[name] = str(cpu_tensor.dtype)

    payload: dict[str, Any] = {
        "format_version": "cncm-ckpt-v2",
        "step": int(step),
        "model_state_dict": state_dict,
        "tensor_dtypes": tensor_dtypes,
        "config": _serialize_config(config),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None and hasattr(scheduler, "state_dict")
            else None
        ),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None and hasattr(scaler, "state_dict")
            else None
        ),
        "tokens_seen": int(tokens_seen),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "data_cursor": int(data_cursor),
        "rng_state": capture_rng_state() if save_rng else None,
        "extra": extra or {},
    }

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=str(target_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return target_path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    *,
    scaler: Optional[Any] = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load an atomic CNCM checkpoint, restoring exact parameter/buffer dtypes and resume state."""
    ckpt_path = Path(path)
    payload: dict[str, Any] = torch.load(
        ckpt_path,
        map_location=map_location,
        weights_only=False,
    )

    if model is not None:
        raw_model = unwrap_model(model)
        sd = payload["model_state_dict"]
        raw_model.load_state_dict(sd, strict=strict)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        # Ensure optimizer state tensors live on the same device as their parameter
        if model is not None:
            for group in optimizer.param_groups:
                for param in group.get("params", ()):
                    st = optimizer.state.get(param)
                    if not isinstance(st, dict):
                        continue
                    p_dev = param.device
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor) and v.device != p_dev:
                            st[k] = v.to(device=p_dev)

    if (
        scheduler is not None
        and payload.get("scheduler_state_dict") is not None
        and hasattr(scheduler, "load_state_dict")
    ):
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    if (
        scaler is not None
        and payload.get("scaler_state_dict") is not None
        and hasattr(scaler, "load_state_dict")
    ):
        scaler.load_state_dict(payload["scaler_state_dict"])

    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])

    return payload


_STEP_CKPT_RE = re.compile(r"^step_(\d+)\.pt$")


class CheckpointManager:
    """Rolling, disk-safe, wall-clock-guarded checkpoint manager with optional private HF Hub sync.

    Designed for Kaggle's 12-hour hard timeout and 19.5 GB `/kaggle/working` disk cap:
    - Atomically updates `latest.pt` on every periodic save (step or time cadence) and `best.pt`
      whenever held-out `val_loss` improves.
    - Prunes older `step_XXXXXX.pt` files keeping only the newest `keep_last_k` + `latest.pt` + `best.pt`.
    - Tracks elapsed wall time against `max_wall_minutes` (default 660 min = 11h) for graceful exit.
    - Optionally syncs `latest.pt`, `best.pt`, and `metrics.json` asynchronously in a background
      thread to a private HuggingFace Hub repo (rank 0 only under DDP).
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        keep_last_k: int = 2,
        max_wall_minutes: Optional[float] = 660.0,
        save_every_minutes: Optional[float] = 20.0,
        hf_repo_id: Optional[str] = None,
        hf_upload_every: int = 1000,
        is_main_process: bool = True,
        start_time_monotonic: Optional[float] = None,
        hf_path_prefix: str = "",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.keep_last_k = max(0, int(keep_last_k))
        self.max_wall_minutes = float(max_wall_minutes) if max_wall_minutes is not None else None
        self.save_every_minutes = float(save_every_minutes) if save_every_minutes is not None else None
        self.hf_repo_id = str(hf_repo_id).strip() if hf_repo_id else None
        self.hf_upload_every = max(0, int(hf_upload_every))
        self.is_main_process = bool(is_main_process)
        self.hf_path_prefix = str(hf_path_prefix).strip().strip("/")

        self.start_time = float(start_time_monotonic) if start_time_monotonic is not None else time.monotonic()
        self.last_save_time = self.start_time
        self.best_val_loss: Optional[float] = None
        self._upload_thread: Optional[threading.Thread] = None
        self._hf_repo_ready: bool = False

        if self.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.save_dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.save_dir / "best.pt"

    @property
    def metrics_path(self) -> Path:
        return self.save_dir / "metrics.json"

    def elapsed_minutes(self) -> float:
        return (time.monotonic() - self.start_time) / 60.0

    def remaining_minutes(self) -> Optional[float]:
        if self.max_wall_minutes is None:
            return None
        return max(0.0, self.max_wall_minutes - self.elapsed_minutes())

    def should_stop(self) -> bool:
        """Return True when the configured Kaggle wall-clock budget has been reached."""
        if self.max_wall_minutes is None:
            return False
        return self.elapsed_minutes() >= self.max_wall_minutes

    def should_periodic_save(self, step: int, save_every: int) -> bool:
        """Return True when either the step interval or wall-time save interval elapsed."""
        if save_every > 0 and step > 0 and (step % save_every == 0):
            return True
        if self.save_every_minutes is not None and self.save_every_minutes > 0.0:
            if (time.monotonic() - self.last_save_time) >= self.save_every_minutes * 60.0:
                return True
        return False

    def resolve_resume_path(
        self,
        explicit_resume: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """Resolve resume checkpoint: explicit path -> local `latest.pt` -> HF Hub `latest.pt`."""
        if explicit_resume is not None:
            exp = Path(explicit_resume)
            if exp.exists():
                return exp

        if self.latest_path.exists():
            return self.latest_path

        if self.hf_repo_id:
            try:
                from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

                token = resolve_hf_token()
                remote_name = f"{self.hf_path_prefix}/latest.pt" if self.hf_path_prefix else "latest.pt"
                downloaded = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=remote_name,
                    local_dir=str(self.save_dir),
                    token=token,
                )
                dl_path = Path(downloaded)
                if dl_path.exists():
                    return dl_path
            except Exception as exc:
                warnings.warn(
                    f"HF Hub resume lookup failed for {self.hf_repo_id!r} ({exc}); starting fresh.",
                    stacklevel=2,
                )

        return None

    def list_numbered_checkpoints(self) -> list[tuple[int, Path]]:
        if not self.save_dir.exists():
            return []
        found: list[tuple[int, Path]] = []
        for entry in self.save_dir.iterdir():
            if not entry.is_file():
                continue
            m = _STEP_CKPT_RE.match(entry.name)
            if m:
                found.append((int(m.group(1)), entry))
        found.sort(key=lambda item: item[0])
        return found

    def prune_old_checkpoints(self) -> list[Path]:
        """Prune older `step_XXXXXX.pt` files keeping only the newest `keep_last_k`."""
        if not self.is_main_process:
            return []
        numbered = self.list_numbered_checkpoints()
        pruned: list[Path] = []
        if len(numbered) > self.keep_last_k:
            to_remove = numbered[: len(numbered) - self.keep_last_k]
            for _, p in to_remove:
                try:
                    p.unlink()
                    pruned.append(p)
                except OSError:
                    pass
        return pruned

    def _upload_files_worker(self, files: list[tuple[Path, str]]) -> None:
        if not self.hf_repo_id:
            return
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            token = resolve_hf_token()
            api = HfApi(token=token)
            if not self._hf_repo_ready:
                api.create_repo(
                    repo_id=self.hf_repo_id,
                    private=True,
                    exist_ok=True,
                )
                self._hf_repo_ready = True
            for local_path, path_in_repo in files:
                if local_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=path_in_repo,
                        repo_id=self.hf_repo_id,
                    )
        except Exception as exc:
            warnings.warn(
                f"Background HF Hub upload to {self.hf_repo_id!r} failed (non-fatal): {exc}",
                stacklevel=2,
            )

    def _maybe_upload_async(self, step: int, *, force: bool = False) -> None:
        if not self.is_main_process or not self.hf_repo_id:
            return
        should_upload = force or (self.hf_upload_every > 0 and step > 0 and (step % self.hf_upload_every == 0))
        if not should_upload:
            return

        if self._upload_thread is not None and self._upload_thread.is_alive():
            if not force:
                return
            self._upload_thread.join(timeout=300.0)

        files_to_upload: list[tuple[Path, str]] = []
        prefix = f"{self.hf_path_prefix}/" if self.hf_path_prefix else ""
        for p, remote in (
            (self.latest_path, f"{prefix}latest.pt"),
            (self.best_path, f"{prefix}best.pt"),
            (self.metrics_path, f"{prefix}metrics.json"),
        ):
            if p.exists():
                files_to_upload.append((p, remote))
        if not files_to_upload:
            return

        thread = threading.Thread(
            target=self._upload_files_worker,
            args=(files_to_upload,),
            daemon=True,
            name="cncm-hf-upload",
        )
        self._upload_thread = thread
        thread.start()

    def wait_for_uploads(self, timeout: Optional[float] = 300.0) -> None:
        """Join any active background HF Hub upload thread before process exit."""
        if self._upload_thread is not None and self._upload_thread.is_alive():
            self._upload_thread.join(timeout=timeout)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        *,
        config: Any = None,
        scaler: Optional[Any] = None,
        tokens_seen: int = 0,
        best_val_loss: Optional[float] = None,
        data_cursor: int = 0,
        val_loss: Optional[float] = None,
        metrics: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
        save_numbered: bool = True,
        force_hf_upload: bool = False,
    ) -> dict[str, Any]:
        """Atomically persist `latest.pt`, optional `step_XXXXXX.pt` (pruned to `keep_last_k`),
        `best.pt` on improved `val_loss`, `metrics.json`, and trigger async HF Hub upload.
        """
        if not self.is_main_process:
            return {"latest": None, "numbered": None, "best": None, "is_best": False}

        if best_val_loss is not None and self.best_val_loss is None:
            self.best_val_loss = float(best_val_loss)

        is_best = False
        if val_loss is not None and math.isfinite(float(val_loss)):
            v = float(val_loss)
            if self.best_val_loss is None or v < self.best_val_loss:
                self.best_val_loss = v
                is_best = True

        merged_extra = dict(extra or {})
        if metrics is not None:
            merged_extra["metrics"] = metrics

        latest = save_checkpoint(
            self.latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=config,
            extra=merged_extra,
            scaler=scaler,
            tokens_seen=tokens_seen,
            best_val_loss=self.best_val_loss,
            data_cursor=data_cursor,
            save_rng=True,
        )

        numbered_path: Optional[Path] = None
        if save_numbered and self.keep_last_k > 0:
            step_file = self.save_dir / f"step_{int(step):06d}.pt"
            numbered_path = _atomic_copy_file(latest, step_file)
            self.prune_old_checkpoints()

        best_saved: Optional[Path] = None
        if is_best:
            best_saved = _atomic_copy_file(latest, self.best_path)

        if metrics is not None:
            _atomic_write_json(self.metrics_path, metrics)

        self.last_save_time = time.monotonic()
        self._maybe_upload_async(step, force=bool(force_hf_upload or is_best))

        return {
            "latest": latest,
            "numbered": numbered_path,
            "best": best_saved,
            "is_best": is_best,
            "best_val_loss": self.best_val_loss,
        }
