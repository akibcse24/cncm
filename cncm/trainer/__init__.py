"""CNCM Training & Reinforcement Learning Pipeline (`cncm.trainer`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cncm.trainer.dataset import (
    BlockARSFTDataset,
    FIMPlanDataset,
    GRPOPromptDataset,
    MemmapTokenDataset,
    PackedChunkDataset,
    collate_block_ar,
    collate_grpo_prompts,
    make_pg_fim_stage1_corpus,
    make_synthetic_code_corpus,
    make_synthetic_corpus,
    make_synthetic_fim_tasks,
    make_synthetic_grpo_prompts,
    make_synthetic_sft,
    prepare_memmap_corpus,
)
from cncm.trainer.pilot_eval import (
    measure_code_entropy,
    measure_concept_vs_exact_gap,
    measure_per_domain_stage1_recon,
    measure_tokenizer_fertility,
)
from cncm.trainer.train_utils import (
    CheckpointManager,
    WarmupCosineScheduler,
    amp_context,
    capture_rng_state,
    cleanup_ddp,
    clip_and_check_grad_norm,
    count_parameters,
    get_device_type,
    get_dist_info,
    load_checkpoint,
    make_adamw,
    make_dataloader,
    resolve_hf_token,
    restore_rng_state,
    save_checkpoint,
    select_device,
    set_seed,
    setup_ddp_if_needed,
    unwrap_model,
)

if TYPE_CHECKING:
    from cncm.trainer.train_grpo import (
        compute_fim_reward,
        compute_verifiable_reward,
        train_grpo,
        train_pg_fim_grpo,
    )
    from cncm.trainer.train_stage1 import train_stage1
    from cncm.trainer.train_stage2 import train_stage2

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "train_stage1": ("cncm.trainer.train_stage1", "train_stage1"),
    "train_stage2": ("cncm.trainer.train_stage2", "train_stage2"),
    "train_grpo": ("cncm.trainer.train_grpo", "train_grpo"),
    "train_pg_fim_grpo": ("cncm.trainer.train_grpo", "train_pg_fim_grpo"),
    "compute_verifiable_reward": ("cncm.trainer.train_grpo", "compute_verifiable_reward"),
    "compute_fim_reward": ("cncm.trainer.train_grpo", "compute_fim_reward"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load CLI trainer modules to avoid `runpy` RuntimeWarnings under `python -m`."""
    if name in _LAZY_EXPORTS:
        import importlib

        mod_name, _ = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        # Overwrite Python's auto-bound submodule attribute (e.g. `cncm.trainer.train_grpo`)
        # with every exported callable from `mod_name` so subsequent imports never receive
        # the module object instead of the function.
        for export_name, (exp_mod_name, exp_attr_name) in _LAZY_EXPORTS.items():
            if exp_mod_name == mod_name:
                globals()[export_name] = getattr(mod, exp_attr_name)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


__all__ = [
    # Datasets & synthetic data
    "PackedChunkDataset",
    "MemmapTokenDataset",
    "prepare_memmap_corpus",
    "BlockARSFTDataset",
    "FIMPlanDataset",
    "GRPOPromptDataset",
    "collate_block_ar",
    "collate_grpo_prompts",
    "make_synthetic_corpus",
    "make_synthetic_sft",
    "make_synthetic_grpo_prompts",
    "make_synthetic_code_corpus",
    "make_pg_fim_stage1_corpus",
    "make_synthetic_fim_tasks",
    # Pilot diagnostics
    "measure_code_entropy",
    "measure_per_domain_stage1_recon",
    "measure_tokenizer_fertility",
    "measure_concept_vs_exact_gap",
    # Utilities
    "CheckpointManager",
    "WarmupCosineScheduler",
    "select_device",
    "get_device_type",
    "amp_context",
    "save_checkpoint",
    "load_checkpoint",
    "capture_rng_state",
    "restore_rng_state",
    "unwrap_model",
    "get_dist_info",
    "setup_ddp_if_needed",
    "cleanup_ddp",
    "make_adamw",
    "make_dataloader",
    "resolve_hf_token",
    "set_seed",
    "count_parameters",
    "clip_and_check_grad_norm",
    # Trainers (lazy-loaded to keep `python -m cncm.trainer.train_*` warning-free)
    "train_stage1",
    "train_stage2",
    "train_grpo",
    "train_pg_fim_grpo",
    "compute_verifiable_reward",
    "compute_fim_reward",
]
