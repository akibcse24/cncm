"""Real Edge Quantization (Packed INT4 / Per-Channel INT8) & Hardware Profiler for CNCM.

Implements genuine memory-saving quantized modules using pure PyTorch ops
(CPU, CUDA, DirectML, MPS, ONNX-compatible):

- ``Int4PackedLinear``: group-wise asymmetric 4-bit weights packed 2 values per
  ``uint8`` byte + ``float16`` per-group scales/zeros (~5.0 bits/param at
  ``group_size=32``, cutting Linear RAM by ~6.4x vs FP32).
- ``Int8Linear``: per-output-channel symmetric 8-bit weights (``int8`` + ``float16``
  scales, ~4.0x smaller than FP32).
- ``Int8Embedding``: per-row symmetric 8-bit embedding tables (~3.9x smaller than FP32).
- ``quantize_model_int4`` / ``quantize_model_int8``: non-destructive model converters
  keeping ``RMSNorm``, ``GroupNorm``, and tiny FSQ bottleneck projections in FP32.
- ``profile_edge_targets``: computes exact parameter counts, memory footprints,
  KV-cache sizes (FP16 vs INT8), chunk FLOPs, memory-bandwidth reduction vs AR,
  and realistic latency/feasibility across 4 edge/desktop hardware profiles.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_SKIP_SUBSTRINGS: Tuple[str, ...] = (
    "quantizer",
    "rfsq",
    "fsq",
    "pre_quant",
    "post_quant",
    "to_fsq",
    "from_fsq",
    "norm",
)


class Int4PackedLinear(nn.Module):
    """Group-wise asymmetric INT4 Linear layer with real 2-per-byte ``uint8`` packing.

    For weight matrix ``W`` of shape ``[out_features, in_features]``:
    - Weights are quantized into ``[0, 15]`` per group of ``eff_group_size`` inputs:
      ``q = clamp(round((W - zero) / scale), 0, 15)``.
    - Two adjacent 4-bit values are packed into one ``uint8`` byte:
      ``packed = (q_even & 0x0F) | ((q_odd & 0x0F) << 4)``.
    - ``scales`` and ``zeros`` are stored in ``float16`` per ``[out_features, num_groups]``.
    - At ``group_size=32``, total storage is ``0.5 + 2*(2/32) = 0.625`` bytes/param
      (5.0 bits/param, **6.4x smaller than FP32**).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.requested_group_size = int(group_size)

        # Ensure group size is even and <= padded_in_features
        g = max(2, int(group_size))
        if g % 2 != 0:
            g += 1
        if self.in_features < g:
            g = max(2, self.in_features + (self.in_features % 2))
        self.group_size = g

        self.padded_in_features = math.ceil(self.in_features / self.group_size) * self.group_size
        self.num_groups = self.padded_in_features // self.group_size

        self.register_buffer(
            "qweight",
            torch.zeros((self.out_features, self.padded_in_features // 2), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.ones((self.out_features, self.num_groups), dtype=torch.float16),
        )
        self.register_buffer(
            "zeros",
            torch.zeros((self.out_features, self.num_groups), dtype=torch.float16),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, group_size: int = 32) -> "Int4PackedLinear":
        out_features, in_features = linear.weight.shape
        has_bias = linear.bias is not None
        mod = cls(
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            bias=has_bias,
        )
        device = linear.weight.device
        mod = mod.to(device)

        w = linear.weight.detach().to(torch.float32)
        if mod.padded_in_features > in_features:
            pad = mod.padded_in_features - in_features
            w = F.pad(w, (0, pad), value=0.0)

        w_g = w.view(out_features, mod.num_groups, mod.group_size)
        w_min = w_g.amin(dim=-1)
        w_max = w_g.amax(dim=-1)

        # Avoid fp16 subnormal collapse while keeping exact zeros
        span = w_max - w_min
        scales_fp16 = (span / 15.0).clamp(min=1e-5).to(torch.float16)
        zeros_fp16 = w_min.to(torch.float16)

        # Quantize against the exact fp16 scale/zero that will be used at runtime
        scales_f = scales_fp16.to(torch.float32).clamp(min=1e-7)
        zeros_f = zeros_fp16.to(torch.float32)

        q = torch.round((w_g - zeros_f.unsqueeze(-1)) / scales_f.unsqueeze(-1))
        q = q.clamp(0.0, 15.0).to(torch.uint8)
        q_pairs = q.view(out_features, mod.padded_in_features // 2, 2)
        packed = (q_pairs[..., 0] & 0x0F) | ((q_pairs[..., 1] & 0x0F) << 4)

        mod.qweight.copy_(packed)
        mod.scales.copy_(scales_fp16)
        mod.zeros.copy_(zeros_fp16)
        if has_bias and linear.bias is not None and mod.bias is not None:
            mod.bias.copy_(linear.bias.detach().to(torch.float16))
        return mod

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        """Unpack 4-bit nibbles and dequantize weight matrix on the fly."""
        lo = (self.qweight & 0x0F).to(dtype)
        hi = ((self.qweight >> 4) & 0x0F).to(dtype)
        unpacked = torch.stack((lo, hi), dim=-1).reshape(
            self.out_features, self.num_groups, self.group_size
        )
        w = unpacked * self.scales.to(dtype).unsqueeze(-1) + self.zeros.to(dtype).unsqueeze(-1)
        return w.reshape(self.out_features, self.padded_in_features)[:, : self.in_features]

    @property
    def weight(self) -> Tensor:
        """Compatibility property returning dequantized FP32 weight for tied-head consumers."""
        return self.dequantize(dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        w = self.dequantize(dtype=x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def nbytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        total += self.zeros.numel() * self.zeros.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def logical_numel(self) -> int:
        n = self.in_features * self.out_features
        if self.bias is not None:
            n += self.out_features
        return n

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )


class Int8Linear(nn.Module):
    """Per-output-channel symmetric INT8 Linear layer (~4.0x smaller than FP32)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer(
            "qweight",
            torch.zeros((self.out_features, self.in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.ones(self.out_features, dtype=torch.float16),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        out_features, in_features = linear.weight.shape
        has_bias = linear.bias is not None
        mod = cls(in_features=in_features, out_features=out_features, bias=has_bias)
        mod = mod.to(linear.weight.device)

        w = linear.weight.detach().to(torch.float32)
        max_abs = w.abs().amax(dim=-1)
        scales_fp16 = (max_abs / 127.0).clamp(min=1e-5).to(torch.float16)
        scales_f = scales_fp16.to(torch.float32).clamp(min=1e-7)
        q = torch.round(w / scales_f.unsqueeze(-1)).clamp(-127.0, 127.0).to(torch.int8)

        mod.qweight.copy_(q)
        mod.scales.copy_(scales_fp16)
        if has_bias and linear.bias is not None and mod.bias is not None:
            mod.bias.copy_(linear.bias.detach().to(torch.float16))
        return mod

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        return self.qweight.to(dtype) * self.scales.to(dtype).unsqueeze(-1)

    @property
    def weight(self) -> Tensor:
        return self.dequantize(dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        w = self.dequantize(dtype=x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def nbytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def logical_numel(self) -> int:
        n = self.in_features * self.out_features
        if self.bias is not None:
            n += self.out_features
        return n

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class Int8Embedding(nn.Module):
    """Per-row symmetric INT8 Embedding table (~3.9x smaller than FP32)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.register_buffer(
            "qweight",
            torch.zeros((self.num_embeddings, self.embedding_dim), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.ones(self.num_embeddings, dtype=torch.float16),
        )

    @classmethod
    @torch.no_grad()
    def from_embedding(cls, emb: nn.Embedding) -> "Int8Embedding":
        mod = cls(
            num_embeddings=emb.num_embeddings,
            embedding_dim=emb.embedding_dim,
            padding_idx=emb.padding_idx,
        ).to(emb.weight.device)
        w = emb.weight.detach().to(torch.float32)
        max_abs = w.abs().amax(dim=-1)
        scales_fp16 = (max_abs / 127.0).clamp(min=1e-5).to(torch.float16)
        scales_f = scales_fp16.to(torch.float32).clamp(min=1e-7)
        q = torch.round(w / scales_f.unsqueeze(-1)).clamp(-127.0, 127.0).to(torch.int8)
        if emb.padding_idx is not None and 0 <= emb.padding_idx < emb.num_embeddings:
            q[emb.padding_idx].zero_()
        mod.qweight.copy_(q)
        mod.scales.copy_(scales_fp16)
        return mod

    def dequantize(self, dtype: torch.dtype = torch.float32) -> Tensor:
        return self.qweight.to(dtype) * self.scales.to(dtype).unsqueeze(-1)

    @property
    def weight(self) -> Tensor:
        return self.dequantize(dtype=torch.float32)

    def forward(self, input_ids: Tensor) -> Tensor:
        q_rows = F.embedding(input_ids, self.qweight, padding_idx=self.padding_idx)
        s_rows = F.embedding(input_ids, self.scales.unsqueeze(-1))
        return q_rows.to(torch.float32) * s_rows.to(torch.float32)

    def nbytes(self) -> int:
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def logical_numel(self) -> int:
        return self.num_embeddings * self.embedding_dim


def _should_skip(full_name: str, skip_modules: Sequence[str]) -> bool:
    low = full_name.lower()
    return any(tok.lower() in low for tok in skip_modules)


def _replace_modules(
    module: nn.Module,
    mode: str,
    group_size: int,
    skip_modules: Sequence[str],
    quantize_embeddings: bool,
    min_features: int,
    prefix: str = "",
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if _should_skip(full_name, skip_modules):
            continue
        if isinstance(child, nn.Linear):
            out_f, in_f = child.weight.shape
            if min(in_f, out_f) < min_features:
                continue
            if mode == "int4":
                setattr(module, child_name, Int4PackedLinear.from_linear(child, group_size=group_size))
            elif mode == "int8":
                setattr(module, child_name, Int8Linear.from_linear(child))
        elif quantize_embeddings and isinstance(child, nn.Embedding):
            if child.num_embeddings * child.embedding_dim >= 4096:
                setattr(module, child_name, Int8Embedding.from_embedding(child))
        else:
            _replace_modules(
                child,
                mode=mode,
                group_size=group_size,
                skip_modules=skip_modules,
                quantize_embeddings=quantize_embeddings,
                min_features=min_features,
                prefix=full_name,
            )


def quantize_model_int4(
    model: nn.Module,
    group_size: int = 32,
    skip_modules: Optional[Sequence[str]] = None,
    quantize_embeddings: bool = True,
    min_features: int = 16,
    inplace: bool = False,
) -> nn.Module:
    """Convert eligible ``nn.Linear`` layers to packed ``Int4PackedLinear`` (and large
    ``nn.Embedding`` tables to ``Int8Embedding`` when ``quantize_embeddings=True``).

    Preserves ``RMSNorm``, ``GroupNorm``, and sensitive FSQ projections in FP32.
    """
    target = model if inplace else copy.deepcopy(model)
    skips = tuple(DEFAULT_SKIP_SUBSTRINGS) if skip_modules is None else tuple(skip_modules)
    _replace_modules(
        target,
        mode="int4",
        group_size=group_size,
        skip_modules=skips,
        quantize_embeddings=quantize_embeddings,
        min_features=min_features,
    )
    return target


def quantize_model_int8(
    model: nn.Module,
    skip_modules: Optional[Sequence[str]] = None,
    quantize_embeddings: bool = True,
    min_features: int = 16,
    inplace: bool = False,
) -> nn.Module:
    """Convert eligible ``nn.Linear`` layers to per-channel ``Int8Linear`` (and large
    ``nn.Embedding`` tables to ``Int8Embedding`` when ``quantize_embeddings=True``).
    """
    target = model if inplace else copy.deepcopy(model)
    skips = tuple(DEFAULT_SKIP_SUBSTRINGS) if skip_modules is None else tuple(skip_modules)
    _replace_modules(
        target,
        mode="int8",
        group_size=32,
        skip_modules=skips,
        quantize_embeddings=quantize_embeddings,
        min_features=min_features,
    )
    return target


def measure_model_size(model: nn.Module) -> Dict[str, Any]:
    """Compute exact deduplicated byte footprint and layer breakdown of a model."""
    seen: set[Tuple[int, int, int]] = set()
    param_bytes = 0
    buffer_bytes = 0
    raw_param_count = 0

    for p in model.parameters():
        key = (p.data_ptr(), p.numel(), p.element_size())
        if key not in seen:
            seen.add(key)
            param_bytes += p.numel() * p.element_size()
            raw_param_count += p.numel()

    for b in model.buffers():
        key = (b.data_ptr(), b.numel(), b.element_size())
        if key not in seen:
            seen.add(key)
            buffer_bytes += b.numel() * b.element_size()

    linear_bytes = 0
    embedding_bytes = 0
    logical_quant_params = 0
    num_int4 = 0
    num_int8 = 0
    num_int8_emb = 0
    num_fp_linear = 0

    for mod in model.modules():
        if isinstance(mod, Int4PackedLinear):
            num_int4 += 1
            linear_bytes += mod.nbytes()
            logical_quant_params += mod.logical_numel()
        elif isinstance(mod, Int8Linear):
            num_int8 += 1
            linear_bytes += mod.nbytes()
            logical_quant_params += mod.logical_numel()
        elif isinstance(mod, Int8Embedding):
            num_int8_emb += 1
            embedding_bytes += mod.nbytes()
            logical_quant_params += mod.logical_numel()
        elif isinstance(mod, nn.Linear):
            num_fp_linear += 1
            linear_bytes += mod.weight.numel() * mod.weight.element_size()
            if mod.bias is not None:
                linear_bytes += mod.bias.numel() * mod.bias.element_size()
        elif isinstance(mod, nn.Embedding):
            embedding_bytes += mod.weight.numel() * mod.weight.element_size()

    total_bytes = param_bytes + buffer_bytes
    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024.0 * 1024.0),
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "linear_bytes": linear_bytes,
        "linear_mb": linear_bytes / (1024.0 * 1024.0),
        "embedding_bytes": embedding_bytes,
        "embedding_mb": embedding_bytes / (1024.0 * 1024.0),
        "num_params_raw": raw_param_count,
        "num_params_logical": raw_param_count + logical_quant_params,
        "num_int4_layers": num_int4,
        "num_int8_layers": num_int8,
        "num_int8_embeddings": num_int8_emb,
        "num_fp_linear_layers": num_fp_linear,
    }


# ---------------------------------------------------------------------------
# Edge Hardware Profiler
# ---------------------------------------------------------------------------

def _estimate_cfg_params(cfg: Any) -> Dict[str, int]:
    """Analytically estimate Stage 1 and Stage 2 parameter counts matching ``CNCMModel``."""
    v = int(getattr(cfg, "vocab_size", 4096))
    d_vq = int(getattr(cfg, "d_vq", 128))
    fsq_levels = tuple(getattr(cfg, "fsq_levels", (8, 8, 8, 8)))
    d_q = len(fsq_levels)
    levels = tuple(getattr(cfg, "levels", (1, 2, 4, 8, 12, 20, 36, 64)))
    k_levels = len(levels)
    dec_layers = int(getattr(cfg, "dec_layers", 1))
    dec_inter = int(getattr(cfg, "dec_intermediate_size", int(d_vq * 3)))
    tie_tok = bool(getattr(cfg, "tie_tokenizer_embeddings", False))

    h = int(getattr(cfg, "hidden_size", 384))
    u_layers = int(getattr(cfg, "num_unique_layers", 7))
    n_loops = int(getattr(cfg, "num_loops", 3))
    eff_layers = u_layers * n_loops
    rank = int(getattr(cfg, "loop_adapter_rank", 16))
    n_heads = int(getattr(cfg, "num_attention_heads", 6))
    n_kv = int(getattr(cfg, "num_key_value_heads", 2))
    inter = int(getattr(cfg, "intermediate_size", int(h * 8 / 3)))
    ch_hidden = int(getattr(cfg, "concept_head_hidden", 96))
    tie_word = bool(getattr(cfg, "tie_word_embeddings", True))
    head_dim = h // max(1, n_heads)

    # Stage 1: wte + output_head + M-RFSQ (zero codebook tables) + dec_layers
    s1_emb = v * d_vq
    s1_out = 0 if tie_tok else (v * d_vq)
    # Cascade ratio = chunk_size // levels[0]; each stage has dw(d_vq*s) + pw(d_vq*d_vq)
    chunk_size = int(getattr(cfg, "chunk_size", 64))
    ratio = max(1, chunk_size // max(1, levels[0]))
    cascade_stages = 0
    rem = ratio
    for f in (4, 3, 2):
        while rem % f == 0:
            cascade_stages += 1
            rem //= f
    if rem > 1:
        cascade_stages += 1
    s1_rfsq = (
        k_levels * (2 * d_vq * d_q + d_q + d_vq)  # FSQLevel proj_in + proj_out
        + k_levels * (d_vq * 3 + d_vq * d_vq + d_vq)  # DepthwiseSepPhi
        + 2 * cascade_stages * (d_vq * 4 + d_vq * d_vq)  # CascadeDown + CascadeUp
    )
    s1_dec = dec_layers * (4 * d_vq * d_vq + 3 * d_vq * dec_inter + 4 * d_vq) + d_vq
    stage1_params = s1_emb + s1_out + s1_rfsq + s1_dec

    # Stage 2: exact_embed (+ untied head if any) + unique CNCMBlocks + LoopAdapters + heads
    s2_emb = v * h
    s2_exact_head = 0 if tie_word else (v * h)
    attn_params = h * (n_heads * head_dim) + 2 * h * (n_kv * head_dim) + (n_heads * head_dim) * h + 2 * head_dim
    mlp_params = 3 * h * inter
    unique_blocks = u_layers * (attn_params + mlp_params)
    step_norms = eff_layers * (2 * h) + h
    loop_adapters = eff_layers * (2 * h * rank + h) if (n_loops > 1 and rank > 0) else 0
    io_projs = d_vq * h + h + k_levels * h + levels[0] * h
    # FactorizedConceptHead: proj_h + level_emb + digit_embs + per-digit heads
    sum_fsq = sum(fsq_levels)
    concept_head = (
        h * ch_hidden
        + k_levels * ch_hidden
        + sum_fsq * ch_hidden
        + d_q * (ch_hidden * ch_hidden + ch_hidden * max(fsq_levels))
    )
    stage2_params = s2_emb + s2_exact_head + unique_blocks + step_norms + loop_adapters + io_projs + concept_head

    # Active per-token params (accounting for relaxed recursive layer looping)
    active_per_token = eff_layers * (attn_params + mlp_params) + loop_adapters + io_projs + concept_head
    return {
        "stage1_params": int(stage1_params),
        "stage2_params": int(stage2_params),
        "total_params": int(stage1_params + stage2_params),
        "effective_params_per_token": int(active_per_token),
    }


def profile_edge_targets(
    cfg: Any,
    model: Optional[nn.Module] = None,
    context_len: int = 1024,
    group_size: int = 32,
    avg_early_exit_ratio: float = 0.80,
) -> Dict[str, Any]:
    """Compute comprehensive CNCM edge memory, FLOPs, bandwidth, and device latency profile.

    Compares CNCM Concept Mode against an equivalent autoregressive (AR) model of
    identical effective depth and active parameter count across 4 target devices:
    1. Low-End Android (4 GB RAM, ARM Cortex-A55/A76, XNNPACK/NNAPI)
    2. Consumer iGPU Laptop (8–16 GB shared RAM, Intel Iris Xe / AMD Radeon Vega, DirectML/Vulkan)
    3. Apple Silicon / Mid-Range Mobile NPU (8–16 GB unified RAM, CoreML/QNN/MPS)
    4. Desktop GPU (RTX 3060/4060/4090, CUDA/TensorRT)
    """
    est = _estimate_cfg_params(cfg)
    if model is not None:
        sz = measure_model_size(model)
        total_params = int(sz["num_params_logical"])
        s1_mod = getattr(model, "concept_tokenizer", None)
        if s1_mod is not None:
            s1_params = int(measure_model_size(s1_mod)["num_params_logical"])
            s2_params = max(0, total_params - s1_params)
        else:
            s1_params = est["stage1_params"]
            s2_params = max(0, total_params - s1_params)
    else:
        total_params = est["total_params"]
        s1_params = est["stage1_params"]
        s2_params = est["stage2_params"]

    chunk_size = int(getattr(cfg, "chunk_size", 64))
    levels = tuple(getattr(cfg, "levels", (1, 2, 4, 8, 12, 20, 36, 64)))
    num_levels = len(levels)
    sum_levels = sum(levels)
    h = int(getattr(cfg, "hidden_size", 384))
    u_layers = int(getattr(cfg, "num_unique_layers", 6))
    n_loops = int(getattr(cfg, "num_loops", 3))
    eff_layers = u_layers * n_loops
    n_heads = int(getattr(cfg, "num_attention_heads", 6))
    n_kv = int(getattr(cfg, "num_key_value_heads", 2))
    head_dim = h // max(1, n_heads)
    vocab_size = int(getattr(cfg, "vocab_size", 4096))
    d_vq = int(getattr(cfg, "d_vq", 192))

    # Separate embedding params (~INT8) vs linear params (INT4 packed at 0.5 + 4/group_size B/param)
    emb_params = min(total_params, vocab_size * d_vq * 2)
    linear_params = max(0, total_params - emb_params)
    int4_bytes_per_linear = 0.5 + (4.0 / max(2, group_size))  # 0.625 B/param at group_size=32

    fp32_bytes = total_params * 4.0
    bf16_bytes = total_params * 2.0
    int8_bytes = total_params * 1.02  # int8 + fp16 per-channel scales
    int4_bytes = linear_params * int4_bytes_per_linear + emb_params * 1.02

    mb = 1024.0 * 1024.0

    # KV cache RAM for causal prefix of length `context_len`
    # Target scales use block_diagonal attention -> ZERO target KV cache growth!
    kv_elems = 2 * eff_layers * n_kv * head_dim * context_len
    kv_fp16_mb = (kv_elems * 2.0) / mb
    kv_int8_mb = (kv_elems * 1.05) / mb  # int8 + per-head/token fp16 scales

    # Working activation scratchpad (~6–12 MB for edge models)
    act_overhead_mb = max(4.0, (sum_levels * h * 8 * 4.0) / mb + 4.0)
    runtime_ram_int4_mb = (int4_bytes / mb) + kv_int8_mb + act_overhead_mb
    runtime_ram_int8_mb = (int8_bytes / mb) + kv_int8_mb + act_overhead_mb

    # Active FLOPs per forward pass (accounting for relaxed recursive loops!)
    # Each unique layer runs `n_loops` times -> effective active params ~ (s2_params * n_loops)
    active_params_per_token = max(s2_params, est["effective_params_per_token"])

    # FLOPs for a single L-token chunk (64 tokens):
    # - Equivalent AR model: 2 * active_params * L
    # - CNCM Concept Mode (full K steps): 2 * active_params * sum(levels) + Stage1 decode
    # - CNCM Concept Mode (early exit): scales roughly by avg_early_exit_ratio
    ar_chunk_flops = 2.0 * active_params_per_token * chunk_size
    s1_dec_flops = 2.0 * s1_params * chunk_size
    cncm_chunk_flops_full = 2.0 * active_params_per_token * sum_levels + s1_dec_flops
    eff_levels = max(1.0, num_levels * avg_early_exit_ratio)
    eff_sum_levels = sum_levels * avg_early_exit_ratio
    cncm_chunk_flops_ee = 2.0 * active_params_per_token * eff_sum_levels + s1_dec_flops

    # Memory bandwidth traffic per 64-token chunk (the true edge bottleneck!):
    # AR sweeps all weights L=64 times (batch=1 decode); CNCM sweeps weights ONLY K=8 times!
    # Note: Unique weights are read `n_loops` times if larger than L2/SRAM, or 1x if L2-cached!
    # We use conservative uncached weight bytes per step = int4_bytes (for edge INT4).
    ar_bw_bytes_per_chunk = int4_bytes * chunk_size
    cncm_bw_bytes_per_chunk = int4_bytes * num_levels
    cncm_ee_bw_bytes_per_chunk = int4_bytes * eff_levels
    bw_reduction_full = ar_bw_bytes_per_chunk / max(1.0, cncm_bw_bytes_per_chunk)
    bw_reduction_ee = ar_bw_bytes_per_chunk / max(1.0, cncm_ee_bw_bytes_per_chunk)

    # Hardware profiles:
    # (name, ram_gb, bw_gbs, gemv_tflops, gemm_tflops, layer_dispatch_ms, backend, q_label, w_bytes, run_ram_mb)
    # Reflects real GEMV (batch=1 AR) vs batched GEMM (CNCM parallel levels) + per-layer dispatch overhead.
    # For recursive looping (n_loops > 1), weights exceeding L2/L3 cache incur partial re-read traffic.
    loop_bw_factor = 1.0 + 0.45 * max(0, n_loops - 1)
    devices_spec = [
        (
            "Low-End Android (4 GB RAM, ARM Cortex-A55/A76)",
            4.0,
            10.0,     # Sustained single-process LPDDR4X bandwidth (GB/s)
            0.05,     # Batch-1 INT4/INT8 NEON GEMV effective throughput (TFLOPS)
            0.28,     # Parallel-level XNNPACK sdot/i8mm GEMM effective throughput (TFLOPS)
            0.070,    # Per-effective-layer dispatch overhead (ms)
            "XNNPACK / NNAPI / ONNX Runtime Mobile",
            "INT4+INT8KV",
            int4_bytes,
            runtime_ram_int4_mb,
        ),
        (
            "Consumer iGPU Laptop (8-16 GB RAM, Intel Iris Xe / AMD Vega)",
            12.0,
            42.0,     # Sustained shared DDR4/LPDDR5 bandwidth (GB/s)
            0.35,     # Batch-1 GEMV effective throughput (TFLOPS)
            2.20,     # Parallel-level DirectML/Vulkan GEMM throughput (TFLOPS)
            0.045,    # Per-effective-layer DirectML dispatch overhead (ms)
            "DirectML / Vulkan / ONNX Runtime",
            "INT4+INT8KV",
            int4_bytes,
            runtime_ram_int4_mb,
        ),
        (
            "Apple Silicon / Mid-Range Mobile NPU (8-16 GB Unified)",
            16.0,
            95.0,     # Sustained unified LPDDR5 bandwidth (GB/s)
            1.20,     # Batch-1 GEMV effective throughput (TFLOPS)
            9.00,     # Parallel-level CoreML/QNN/MPS GEMM throughput (TFLOPS)
            0.025,    # Per-effective-layer Metal/QNN dispatch overhead (ms)
            "CoreML / MPS / Qualcomm QNN",
            "INT4+INT8KV",
            int4_bytes,
            runtime_ram_int4_mb,
        ),
        (
            "Desktop GPU (RTX 3060/4060/4090, 8-24 GB VRAM)",
            12.0,
            360.0,    # Sustained GDDR6 bandwidth (GB/s)
            6.00,     # Batch-1 GEMV effective throughput (TFLOPS)
            48.0,     # Parallel-level BF16/INT4 Tensor Core GEMM throughput (TFLOPS)
            0.015,    # Per-effective-layer CUDA kernel launch overhead (ms)
            "CUDA / TensorRT / PyTorch SDPA",
            "BF16/INT4",
            bf16_bytes,
            (bf16_bytes / mb) + kv_fp16_mb + act_overhead_mb,
        ),
    ]

    targets: List[Dict[str, Any]] = []
    for (
        name,
        ram_gb,
        bw_gbs,
        gemv_tflops,
        gemm_tflops,
        layer_disp_ms,
        backend,
        q_label,
        w_bytes,
        run_ram_mb,
    ) in devices_spec:
        bw_bps = bw_gbs * 1e9
        gemv_flops_ps = gemv_tflops * 1e12
        gemm_flops_ps = gemm_tflops * 1e12
        step_disp_s = (eff_layers * layer_disp_ms) / 1000.0
        eff_w_bytes = w_bytes * loop_bw_factor

        # Roofline + per-layer dispatch latency:
        # AR runs `chunk_size` sequential GEMV steps; CNCM runs only `K` (or `K_ee`) parallel GEMM steps!
        ar_time_s = (
            max((eff_w_bytes * chunk_size) / bw_bps, ar_chunk_flops / gemv_flops_ps)
            + chunk_size * step_disp_s
        )
        cncm_full_s = (
            max((eff_w_bytes * num_levels) / bw_bps, cncm_chunk_flops_full / gemm_flops_ps)
            + (num_levels + 1) * step_disp_s  # +1 for Stage 1 single-block decode
        )
        cncm_ee_s = (
            max((eff_w_bytes * eff_levels) / bw_bps, cncm_chunk_flops_ee / gemm_flops_ps)
            + (eff_levels + 1) * step_disp_s
        )

        ar_tps = chunk_size / ar_time_s
        cncm_full_tps = chunk_size / cncm_full_s
        cncm_ee_tps = chunk_size / cncm_ee_s

        usable_ram_mb = ram_gb * 1024.0 * 0.35  # Conservative app memory budget (35% of device RAM)
        if run_ram_mb <= usable_ram_mb * 0.25 and cncm_full_tps >= 25.0:
            feasibility = "Excellent (real-time + plenty of headroom)"
        elif run_ram_mb <= usable_ram_mb and cncm_full_tps >= 10.0:
            feasibility = "Good (interactive real-time)"
        elif run_ram_mb <= ram_gb * 1024.0 * 0.65:
            feasibility = "Feasible (background / bursty)"
        else:
            feasibility = "Tight (use smaller preset)"

        targets.append(
            {
                "device": name,
                "backend": backend,
                "precision": q_label,
                "runtime_ram_mb": round(run_ram_mb, 2),
                "ar_tok_per_s": round(ar_tps, 1),
                "cncm_tok_per_s": round(cncm_full_tps, 1),
                "cncm_early_exit_tok_per_s": round(cncm_ee_tps, 1),
                "chunk_latency_ms": round(cncm_full_s * 1000.0, 1),
                "speedup_vs_ar": round(cncm_full_tps / max(1e-6, ar_tps), 2),
                "feasibility": feasibility,
            }
        )

    return {
        "preset": getattr(cfg, "preset_name", getattr(cfg, "name", "custom")),
        "total_params": total_params,
        "stage1_params": s1_params,
        "stage2_params": s2_params,
        "unique_layers": u_layers,
        "num_loops": n_loops,
        "effective_layers": eff_layers,
        "chunk_size": chunk_size,
        "num_levels": num_levels,
        "sum_levels": sum_levels,
        "seq_blowup": round(sum_levels / max(1, chunk_size), 2),
        "disk_ram_mb": {
            "fp32": round(fp32_bytes / mb, 2),
            "bf16": round(bf16_bytes / mb, 2),
            "int8": round(int8_bytes / mb, 2),
            "int4": round(int4_bytes / mb, 2),
        },
        "kv_cache_mb": {
            "context_len": context_len,
            "fp16": round(kv_fp16_mb, 2),
            "int8": round(kv_int8_mb, 2),
            "target_scale_kv_mb": 0.0,
        },
        "runtime_ram_int4_mb": round(runtime_ram_int4_mb, 2),
        "runtime_ram_int8_mb": round(runtime_ram_int8_mb, 2),
        "compute": {
            "ar_chunk_gflops": round(ar_chunk_flops / 1e9, 3),
            "cncm_chunk_gflops": round(cncm_chunk_flops_full / 1e9, 3),
            "cncm_early_exit_gflops": round(cncm_chunk_flops_ee / 1e9, 3),
            "bw_reduction_vs_ar": round(bw_reduction_full, 2),
            "bw_reduction_early_exit": round(bw_reduction_ee, 2),
        },
        "targets": targets,
    }


def format_edge_profile_table(profile: Dict[str, Any]) -> str:
    """Format a ``profile_edge_targets`` dictionary as a concise terminal/markdown summary."""
    d = profile["disk_ram_mb"]
    kv = profile["kv_cache_mb"]
    c = profile["compute"]
    lines = [
        f"Preset: {profile['preset']} | Params: {profile['total_params']/1e6:.2f}M "
        f"(Stage1={profile['stage1_params']/1e6:.2f}M, Stage2={profile['stage2_params']/1e6:.2f}M) | "
        f"Depth: {profile['unique_layers']}x{profile['num_loops']}={profile['effective_layers']} eff layers",
        f"Chunk: L={profile['chunk_size']}, K={profile['num_levels']}, sum(l_k)={profile['sum_levels']} "
        f"({profile['seq_blowup']}x seq blowup vs 5.0-11.2x in vanilla NCM)",
        f"Weights: FP32={d['fp32']:.2f} MB | BF16={d['bf16']:.2f} MB | "
        f"INT8={d['int8']:.2f} MB | INT4={d['int4']:.2f} MB",
        f"KV Cache ({kv['context_len']} ctx): FP16={kv['fp16']:.2f} MB | INT8={kv['int8']:.2f} MB | "
        f"Target-scale KV=0.00 MB",
        f"64-tok Chunk Compute: AR={c['ar_chunk_gflops']:.2f} GFLOPs | "
        f"CNCM={c['cncm_chunk_gflops']:.2f} GFLOPs | "
        f"HBM/LPDDR BW Traffic Reduction={c['bw_reduction_vs_ar']:.1f}x "
        f"({c['bw_reduction_early_exit']:.1f}x w/ early exit)",
        "-" * 96,
        f"{'Target Hardware':<48} {'RAM(MB)':>8} {'AR t/s':>8} {'CNCM t/s':>9} {'Speedup':>8}  Status",
        "-" * 96,
    ]
    for t in profile["targets"]:
        dev = t["device"][:47]
        lines.append(
            f"{dev:<48} {t['runtime_ram_mb']:>8.1f} {t['ar_tok_per_s']:>8.1f} "
            f"{t['cncm_tok_per_s']:>9.1f} {t['speedup_vs_ar']:>7.2f}x  {t['feasibility']}"
        )
    return "\n".join(lines)


__all__ = [
    "Int4PackedLinear",
    "Int8Linear",
    "Int8Embedding",
    "quantize_model_int4",
    "quantize_model_int8",
    "measure_model_size",
    "profile_edge_targets",
    "format_edge_profile_table",
]
