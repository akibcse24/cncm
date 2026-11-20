"""ONNX Graph Export (opset 17+) & Pure-PyTorch Edge Graph Wrappers for CNCM.

Provides 3 deployment-ready graph wrappers matching edge inference runtimes
(ONNX Runtime Mobile / NNAPI / XNNPACK / DirectML / CoreML):

1. ``Stage1DecoderONNXWrapper`` -> ``stage1_decoder.onnx``:
   Maps flat concept codes ``[B, sum(levels)]`` directly to text logits ``[B, L, V]``.
2. ``Stage2ExactStepONNXWrapper`` -> ``stage2_exact_step.onnx``:
   Runs Exact-AR mode (``<tool_call>`` JSON generation or self-speculative verification)
   mapping ``(input_ids [B, S], position_ids [B, S]) -> logits [B, S, V]``.
3. ``Stage2ConceptStepONNXWrapper`` -> ``stage2_concept_step.onnx``:
   Runs parallel Concept Mode coarse-to-fine forward pass mapping
   ``(prefix_ids [B, P], target_flat_codes [B, sum(levels)]) -> packed digit logits``.

All 3 wrapper modules are pure PyTorch and verifiable even when ``onnx`` /
``onnxscript`` are not installed in the local Python environment.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def _flat_to_digits_onnx_safe(flat_codes: Tensor, fsq_levels: Tuple[int, ...]) -> Tensor:
    """Pure tensor-op mixed-radix conversion ``[...,] -> [..., d_q]`` (ONNX safe)."""
    digits: List[Tensor] = []
    rem = flat_codes.to(torch.long)
    # Match standard mixed-radix order (LSB-first or MSB-first handled by concept_tokenizer if available)
    strides: List[int] = []
    stride = 1
    for lvl in reversed(fsq_levels):
        strides.append(stride)
        stride *= int(lvl)
    strides.reverse()

    for s, lvl in zip(strides, fsq_levels):
        d = torch.div(rem, s, rounding_mode="floor") % int(lvl)
        digits.append(d)
    return torch.stack(digits, dim=-1)


def _split_flat_to_level_digits(
    concept_tokenizer: Any,
    flat_codes: Tensor,
    levels: Tuple[int, ...],
    fsq_levels: Tuple[int, ...],
) -> List[Tensor]:
    """Split ``[B, sum(levels)]`` flat codes into per-level ``[B, l_k, d_q]`` digit tensors."""
    chunks = torch.split(flat_codes.to(torch.long), list(levels), dim=1)
    use_native = hasattr(concept_tokenizer, "flat_to_digits") and callable(
        concept_tokenizer.flat_to_digits
    )
    digit_list: List[Tensor] = []
    for chunk in chunks:
        if use_native:
            digit_list.append(concept_tokenizer.flat_to_digits(chunk))
        else:
            digit_list.append(_flat_to_digits_onnx_safe(chunk, fsq_levels))
    return digit_list


def _pack_digit_logits(raw_digit_logits: Any) -> Tensor:
    """Normalize ``forward_concept(...)[digit_logits]`` into a single ONNX-friendly Tensor."""
    if isinstance(raw_digit_logits, Tensor):
        return raw_digit_logits
    if isinstance(raw_digit_logits, (list, tuple)) and len(raw_digit_logits) > 0:
        first = raw_digit_logits[0]
        if isinstance(first, Tensor):
            # Could be list[d_q] of [B, T, L_j] or list[K] of [B, l_k, ...]
            shapes = [t.shape for t in raw_digit_logits]
            if len(set(shapes)) == 1:
                # Uniform shape across digits -> stack along dim=-2 -> [B, T, d_q, L_j]
                return torch.stack(list(raw_digit_logits), dim=-2)
            # If dim=1 varies (per-level [B, l_k, ...]), concat along dim=1; else concat last dim
            if len({t.shape[-1] for t in raw_digit_logits}) == 1 and first.ndim >= 3:
                return torch.cat(list(raw_digit_logits), dim=1)
            return torch.cat(list(raw_digit_logits), dim=-1)
        if isinstance(first, (list, tuple)):
            # Nested list[K][d_q] -> pack each level then concat on token dim=1
            packed_levels = [_pack_digit_logits(lvl) for lvl in raw_digit_logits]
            return torch.cat(packed_levels, dim=1)
    raise TypeError(f"Unsupported digit_logits structure: {type(raw_digit_logits).__name__}")


class Stage1DecoderONNXWrapper(nn.Module):
    """ONNX wrapper for Stage 1 M-RFSQ Concept Decoder (``stage1_decoder.onnx``)."""

    def __init__(self, concept_tokenizer: nn.Module) -> None:
        super().__init__()
        self.concept_tokenizer = concept_tokenizer
        cfg = getattr(concept_tokenizer, "cfg", None)
        self.levels: Tuple[int, ...] = tuple(getattr(cfg, "levels", (1, 2, 4, 8, 12, 20, 36, 64)))
        self.fsq_levels: Tuple[int, ...] = tuple(getattr(cfg, "fsq_levels", (8, 8, 8, 8)))

    def forward(self, flat_codes: Tensor) -> Tensor:
        """Decode flat concept indices ``[B, sum(levels)]`` -> text logits ``[B, L, V]``."""
        digit_codes = _split_flat_to_level_digits(
            self.concept_tokenizer,
            flat_codes,
            self.levels,
            self.fsq_levels,
        )
        return self.concept_tokenizer.decode_codes(digit_codes)


class Stage2ExactStepONNXWrapper(nn.Module):
    """ONNX wrapper for Stage 2 Exact-AR step (``stage2_exact_step.onnx``)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, position_ids: Tensor) -> Tensor:
        """Run Exact-AR forward pass: ``(input_ids [B, S], position_ids [B, S]) -> logits [B, S, V]``."""
        out = self.model.forward_exact(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        )
        return out["logits"]


class Stage2ConceptStepONNXWrapper(nn.Module):
    """ONNX wrapper for Stage 2 Concept Mode step (``stage2_concept_step.onnx``)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        cfg = getattr(model, "cfg", None)
        self.levels: Tuple[int, ...] = tuple(getattr(cfg, "levels", (1, 2, 4, 8, 12, 20, 36, 64)))
        self.fsq_levels: Tuple[int, ...] = tuple(getattr(cfg, "fsq_levels", (8, 8, 8, 8)))

    def forward(self, prefix_ids: Tensor, target_flat_codes: Tensor) -> Tensor:
        """Run parallel Concept Mode pass: ``(prefix_ids [B, P], target_flat_codes [B, sum(levels)])``."""
        digit_codes = _split_flat_to_level_digits(
            self.model.concept_tokenizer,
            target_flat_codes,
            self.levels,
            self.fsq_levels,
        )
        out = self.model.forward_concept(
            prefix_ids=prefix_ids,
            target_codes=digit_codes,
            prefix_mask=None,
            corrupt_rate=0.0,
        )
        return _pack_digit_logits(out["digit_logits"])


def _run_torch_onnx_export(
    wrapper: nn.Module,
    args: Tuple[Tensor, ...],
    file_path: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    opset: int,
    has_onnxscript: bool,
) -> None:
    kwargs: Dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": int(opset),
        "do_constant_folding": True,
    }
    # PyTorch 2.5+ defaults to dynamo=True which requires onnxscript; fall back cleanly to legacy exporter
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters and not has_onnxscript:
        kwargs["dynamo"] = False
    torch.onnx.export(wrapper, args, str(file_path), **kwargs)


@torch.no_grad()
def export_cncm_onnx(
    model: Any,
    out_dir: Union[str, Path],
    opset: int = 17,
    batch_size: int = 1,
    prefix_len: int = 8,
) -> Dict[str, Any]:
    """Verify pure-PyTorch export wrappers and emit 3 ONNX graphs when ``onnx`` is installed.

    Emitted graphs:
      - ``<out_dir>/stage1_decoder.onnx``
      - ``<out_dir>/stage2_exact_step.onnx``
      - ``<out_dir>/stage2_concept_step.onnx``
      - ``<out_dir>/export_manifest.json``

    If ``onnx`` (or ``onnxscript`` when required by the local PyTorch version) is
    not installed, this function still builds and executes all 3 PyTorch wrappers,
    writes ``export_manifest.json``, and returns a clear status report without crashing.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    device = torch.device("cpu")
    for p in model.parameters():
        device = p.device
        break
    else:
        for b in model.buffers():
            device = b.device
            break

    cfg = getattr(model, "cfg", None)
    levels = tuple(getattr(cfg, "levels", (1, 2, 4, 8, 12, 20, 36, 64)))
    sum_levels = sum(levels)
    codebook_size = int(getattr(cfg, "codebook_size", 4096))
    vocab_size = int(getattr(cfg, "vocab_size", 4096))

    # Build the 3 edge graph wrappers
    s1_wrapper = Stage1DecoderONNXWrapper(model.concept_tokenizer).eval()
    exact_wrapper = Stage2ExactStepONNXWrapper(model).eval()
    concept_wrapper = Stage2ConceptStepONNXWrapper(model).eval()

    # Sample inputs for verification and tracing
    sample_flat_codes = torch.zeros((batch_size, sum_levels), dtype=torch.long, device=device) % max(
        1, codebook_size
    )
    sample_exact_ids = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device) % max(
        1, vocab_size
    )
    sample_pos_ids = (
        torch.arange(prefix_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    # Always verify all 3 wrappers with pure PyTorch first
    s1_out = s1_wrapper(sample_flat_codes)
    exact_out = exact_wrapper(sample_exact_ids, sample_pos_ids)
    concept_out = concept_wrapper(sample_exact_ids, sample_flat_codes)

    wrapper_shapes = {
        "stage1_decoder": {
            "inputs": {"flat_codes": list(sample_flat_codes.shape)},
            "outputs": {"logits": list(s1_out.shape)},
            "finite": bool(torch.isfinite(s1_out).all().item()),
        },
        "stage2_exact_step": {
            "inputs": {
                "input_ids": list(sample_exact_ids.shape),
                "position_ids": list(sample_pos_ids.shape),
            },
            "outputs": {"logits": list(exact_out.shape)},
            "finite": bool(torch.isfinite(exact_out).all().item()),
        },
        "stage2_concept_step": {
            "inputs": {
                "prefix_ids": list(sample_exact_ids.shape),
                "target_flat_codes": list(sample_flat_codes.shape),
            },
            "outputs": {"digit_logits": list(concept_out.shape)},
            "finite": bool(torch.isfinite(concept_out).all().item()),
        },
    }

    has_onnx = importlib.util.find_spec("onnx") is not None
    has_onnxscript = importlib.util.find_spec("onnxscript") is not None
    exported_files: Dict[str, str] = {}
    export_errors: Dict[str, str] = {}
    exported = False
    message = ""

    if not has_onnx:
        message = (
            "Optional package 'onnx' is not installed in the current Python environment; "
            "all 3 pure-PyTorch ONNX wrapper modules were verified cleanly. "
            "Install `pip install onnx onnxscript` to emit .onnx binaries."
        )
    else:
        graph_specs = [
            (
                "stage1_decoder",
                s1_wrapper,
                (sample_flat_codes,),
                out_path / "stage1_decoder.onnx",
                ["flat_codes"],
                ["logits"],
                {"flat_codes": {0: "batch"}, "logits": {0: "batch"}},
            ),
            (
                "stage2_exact_step",
                exact_wrapper,
                (sample_exact_ids, sample_pos_ids),
                out_path / "stage2_exact_step.onnx",
                ["input_ids", "position_ids"],
                ["logits"],
                {
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "position_ids": {0: "batch", 1: "seq_len"},
                    "logits": {0: "batch", 1: "seq_len"},
                },
            ),
            (
                "stage2_concept_step",
                concept_wrapper,
                (sample_exact_ids, sample_flat_codes),
                out_path / "stage2_concept_step.onnx",
                ["prefix_ids", "target_flat_codes"],
                ["digit_logits"],
                {
                    "prefix_ids": {0: "batch", 1: "prefix_len"},
                    "target_flat_codes": {0: "batch"},
                    "digit_logits": {0: "batch"},
                },
            ),
        ]
        for key, wrap, args, fpath, in_names, out_names, dyn in graph_specs:
            try:
                _run_torch_onnx_export(
                    wrapper=wrap,
                    args=args,
                    file_path=fpath,
                    input_names=in_names,
                    output_names=out_names,
                    dynamic_axes=dyn,
                    opset=opset,
                    has_onnxscript=has_onnxscript,
                )
                exported_files[key] = str(fpath)
            except Exception as exc:  # noqa: BLE001
                export_errors[key] = f"{type(exc).__name__}: {exc}"

        exported = len(exported_files) == 3 and not export_errors
        if exported:
            message = f"Successfully exported all 3 CNCM ONNX graphs (opset {opset}) to {out_path}."
        else:
            message = (
                f"Exported {len(exported_files)}/3 ONNX graphs; errors={export_errors}. "
                "Pure-PyTorch wrappers remain verified."
            )

    manifest = {
        "preset": getattr(cfg, "preset_name", "custom") if cfg is not None else "custom",
        "opset": int(opset),
        "onnx_available": has_onnx,
        "onnxscript_available": has_onnxscript,
        "exported": exported,
        "wrappers_verified": all(v["finite"] for v in wrapper_shapes.values()),
        "wrapper_shapes": wrapper_shapes,
        "files": exported_files,
        "errors": export_errors,
    }
    manifest_path = out_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if was_training:
        model.train()

    return {
        "exported": exported,
        "onnx_available": has_onnx,
        "onnxscript_available": has_onnxscript,
        "wrappers_verified": manifest["wrappers_verified"],
        "wrapper_shapes": wrapper_shapes,
        "files": exported_files,
        "errors": export_errors,
        "manifest": str(manifest_path),
        "message": message,
    }


__all__ = [
    "Stage1DecoderONNXWrapper",
    "Stage2ExactStepONNXWrapper",
    "Stage2ConceptStepONNXWrapper",
    "export_cncm_onnx",
]
