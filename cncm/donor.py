"""Universal Config-Driven Donor Loader, Frozen Runner & Offline Hidden/Top-K Cache.

Supports zero-backprop Path A sidecar surgery across open LLM families without requiring
`transformers` model code or custom per-model scripts:
  - `llama` / `smollm2` (`SmolLM2-135M/360M/1.7B`, `Llama-3.2-1B/3B`)
  - `qwen2` (`Qwen2.5-0.5B/1.5B/3B`, `Qwen2.5-Coder-1.5B-Instruct` — QKV bias + 1M RoPE + 152K vocab)
  - `qwen3` (`Qwen3-0.6B/1.7B/4B` — explicit `head_dim != hidden/heads` + per-head `q_norm`/`k_norm`)
  - `mistral` (`Mistral` / `Ministral` — optional sliding window + untied `lm_head`)
  - `gemma` / `gemma2` (`(1 + w)` RMSNorm + `sqrt(H)` embed scale + GeGLU + logit softcapping)
  - `cncm` (extracts clean frozen dense donor weights directly from local CNCM `.safetensors`/`.pt`)

Offline Cache Guarantees (`OfflineDonorCache`):
  - Saves parameter-free unit-RMS-normalized multi-layer taps (`L16 + L24 + L32`) in `float16`
    (strictly bounded `O(1)` values, eliminating FP16 `inf` overflow on Qwen/Gemma outliers).
  - Automatically chooses `uint16` (`vocab < 65535`) vs `int32` (`vocab >= 65535`, e.g. Qwen 152K)
    so token IDs and top-k indices never wrap around modulo 65536.
  - Saves `topk_idx` and normalized `topk_logprobs` aligned directly to target chunk positions
    (`donor_logits[:, P - 1 + j]` predicting `target_ids[:, j]`), preventing off-by-one shifts.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cncm.layers import (
    LayerKV,
    RotaryEmbedding,
    append_kv_cache,
    apply_rotary_pos_emb,
    build_causal_mask,
    build_sliding_window_causal_mask,
    get_kv_seq_len,
    repeat_kv,
    scaled_dot_product_attn_4d,
)


def default_tap_layers(num_layers: int) -> Tuple[int, int, int]:
    """Return 0-indexed 3-tap layer tuple `(L_mid, L_3q, L_last)` (`(15, 23, 31)` -> `L16+L24+L32` for 32L)."""
    n = max(1, int(num_layers))
    t0 = max(0, min(n - 1, (n // 2) - 1 if n >= 2 else 0))
    t1 = max(0, min(n - 1, ((3 * n) // 4) - 1 if n >= 4 else max(0, n - 1)))
    t2 = n - 1
    return (int(t0), int(t1), int(t2))


def unit_rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free unit RMS normalization (`O(1)` range, zero FP16 overflow, exact cache/live parity)."""
    x_f = x.float()
    return (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype=x.dtype)


@dataclass
class DonorConfig:
    """Normalized architectural metadata for any supported open-weight donor LLM."""

    arch: str = "llama"
    donor_id: str = "unknown"
    vocab_size: int = 49152
    hidden_size: int = 960
    num_layers: int = 32
    num_attention_heads: int = 15
    num_key_value_heads: int = 5
    head_dim: int = 64
    intermediate_size: int = 2560
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    max_position_embeddings: int = 8192
    qkv_bias: bool = False
    o_bias: bool = False
    use_qk_norm: bool = False
    rms_norm_add_unit: bool = False
    embed_scale: float = 1.0
    hidden_act: str = "silu"
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    tap_layers: Tuple[int, int, int] = (15, 23, 31)

    def __post_init__(self) -> None:
        self.arch = str(self.arch).strip().lower()
        if self.head_dim <= 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"Donor head_dim ({self.head_dim}) must be even for RoPE")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if not self.tap_layers or len(self.tap_layers) != 3:
            self.tap_layers = default_tap_layers(self.num_layers)
        else:
            self.tap_layers = tuple(
                max(0, min(self.num_layers - 1, int(idx))) for idx in self.tap_layers
            )  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tap_layers"] = list(self.tap_layers)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DonorConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in data.items() if k in valid}
        if "tap_layers" in kw and kw["tap_layers"] is not None:
            kw["tap_layers"] = tuple(int(x) for x in kw["tap_layers"])
        return cls(**kw)


class _DonorRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.add_unit = bool(add_unit)
        self.weight = nn.Parameter(torch.zeros(dim) if self.add_unit else torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        scale = (1.0 + self.weight.float()) if self.add_unit else self.weight.float()
        return (normed * scale).to(dtype=x.dtype)


class _DonorBlock(nn.Module):
    def __init__(self, cfg: DonorConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.input_layernorm = _DonorRMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, add_unit=cfg.rms_norm_add_unit
        )
        self.post_attention_layernorm = _DonorRMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, add_unit=cfg.rms_norm_add_unit
        )

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=cfg.o_bias)

        self.q_norm: Optional[_DonorRMSNorm] = (
            _DonorRMSNorm(self.head_dim, eps=cfg.rms_norm_eps, add_unit=cfg.rms_norm_add_unit)
            if cfg.use_qk_norm
            else None
        )
        self.k_norm: Optional[_DonorRMSNorm] = (
            _DonorRMSNorm(self.head_dim, eps=cfg.rms_norm_eps, add_unit=cfg.rms_norm_add_unit)
            if cfg.use_qk_norm
            else None
        )

        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        past_kv: Optional[LayerKV] = None,
        use_cache: bool = False,
        quantize_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerKV]]:
        bsz, seq_len, _ = h.shape
        xn = self.input_layernorm(h)

        q = self.q_proj(xn).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(xn).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(xn).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        present_kv: Optional[LayerKV] = None
        if past_kv is not None:
            k_full, v_full, updated = append_kv_cache(past_kv, k_t, v_t, quantize_kv=quantize_kv)
            if use_cache:
                present_kv = updated
        else:
            k_full, v_full = k_t, v_t
            if use_cache:
                from cncm.layers import quantize_kv_pair

                present_kv = quantize_kv_pair(k_full, v_full) if quantize_kv else (k_full, v_full)

        k_rep = repeat_kv(k_full, self.n_rep)
        v_rep = repeat_kv(v_full, self.n_rep)

        if self.cfg.attn_logit_softcapping is not None and self.cfg.attn_logit_softcapping > 0.0:
            cap = float(self.cfg.attn_logit_softcapping)
            scale = 1.0 / math.sqrt(float(self.head_dim))
            scores = torch.matmul(q_t.float(), k_rep.float().transpose(-2, -1)) * scale
            scores = torch.tanh(scores / cap) * cap + attn_mask.float()
            probs = F.softmax(scores, dim=-1).to(dtype=v_rep.dtype)
            attn_out = torch.matmul(probs, v_rep)
        else:
            attn_out = scaled_dot_product_attn_4d(q_t, k_rep, v_rep, attn_mask=attn_mask, dropout_p=0.0)

        attn_flat = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        h = h + self.o_proj(attn_flat)

        xn2 = self.post_attention_layernorm(h)
        gate_raw = self.gate_proj(xn2)
        if self.cfg.hidden_act in ("gelu", "gelu_pytorch_tanh", "geglu"):
            act = F.gelu(gate_raw, approximate="tanh")
        else:
            act = F.silu(gate_raw)
        h = h + self.down_proj(act * self.up_proj(xn2))
        return h, present_kv


class FrozenDonorRunner(nn.Module):
    """Zero-grad universal donor runner with multi-layer `L16+L24+L32` tap extraction & KV cache."""

    def __init__(self, cfg: DonorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(
            head_dim=cfg.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
        )
        self.layers = nn.ModuleList([_DonorBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm = _DonorRMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, add_unit=cfg.rms_norm_add_unit
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def freeze_all(self) -> "FrozenDonorRunner":
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        return self

    @torch.no_grad()
    def forward_with_taps(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[LayerKV]] = None,
        use_cache: bool = False,
        quantize_kv: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run frozen donor forward and capture unit-RMS-normalized `tap_layers` hidden states."""
        bsz, q_len = input_ids.shape
        device = input_ids.device
        past_len = get_kv_seq_len(past_key_values)
        kv_len = past_len + q_len

        h = self.embed_tokens(input_ids)
        if self.cfg.embed_scale != 1.0:
            h = h * float(self.cfg.embed_scale)

        if position_ids is None:
            pos_ids = torch.arange(past_len, past_len + q_len, device=device, dtype=torch.long)
        else:
            pos_ids = position_ids.to(device=device, dtype=torch.long)
        cos, sin = self.rope(pos_ids, dtype=h.dtype)

        full_mask = build_causal_mask(
            q_len=q_len,
            kv_len=kv_len,
            past_len=past_len,
            batch_size=bsz,
            key_padding_mask=attention_mask,
            device=device,
            dtype=h.dtype,
        )
        swa_mask: Optional[torch.Tensor] = None
        if self.cfg.sliding_window is not None and self.cfg.sliding_window > 0:
            swa_mask = build_sliding_window_causal_mask(
                q_len=q_len,
                kv_len=kv_len,
                past_len=past_len,
                batch_size=bsz,
                sliding_window=self.cfg.sliding_window,
                sink_tokens=0,
                key_padding_mask=attention_mask,
                device=device,
                dtype=h.dtype,
            )

        tap_set = set(self.cfg.tap_layers)
        captured: Dict[int, torch.Tensor] = {}
        present_kvs: Optional[List[LayerKV]] = [] if use_cache else None

        for i, block in enumerate(self.layers):
            past_i = past_key_values[i] if (past_key_values is not None and i < len(past_key_values)) else None
            # Gemma-2 uses alternating sliding-window on even layers; Mistral uses SWA on all layers
            use_swa = swa_mask is not None and (
                (i % 2 == 0) if self.cfg.arch == "gemma2" else True
            )
            mask_i = swa_mask if use_swa and swa_mask is not None else full_mask
            h, pres_i = block(
                h,
                cos=cos,
                sin=sin,
                attn_mask=mask_i,
                past_kv=past_i,
                use_cache=use_cache,
                quantize_kv=quantize_kv,
            )
            if present_kvs is not None and pres_i is not None:
                present_kvs.append(pres_i)
            if i in tap_set:
                captured[i] = unit_rms_normalize(h, eps=self.cfg.rms_norm_eps)

        # Stack in exact `cfg.tap_layers` order -> [B, q_len, 3, H_donor]
        unit_taps = torch.stack([captured[idx] for idx in self.cfg.tap_layers], dim=2)
        logits = self.lm_head(self.norm(h))
        if self.cfg.final_logit_softcapping is not None and self.cfg.final_logit_softcapping > 0.0:
            cap = float(self.cfg.final_logit_softcapping)
            logits = torch.tanh(logits.float() / cap) * cap

        return {
            "logits": logits,
            "unit_taps": unit_taps,
            "past_key_values": present_kvs,
        }

    @torch.no_grad()
    def generate_exact(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        quantize_kv: bool = False,
    ) -> torch.Tensor:
        """Exact autoregressive baseline generation on the untouched frozen donor."""
        if max_new_tokens <= 0:
            return prompt_ids
        eos = self.cfg.eos_token_id if eos_token_id is None else int(eos_token_id)
        out = self.forward_with_taps(prompt_ids, use_cache=True, quantize_kv=quantize_kv)
        past_kv = out["past_key_values"]
        next_logits = out["logits"][:, -1, :].float()

        generated: List[torch.Tensor] = [prompt_ids]
        finished = torch.zeros(prompt_ids.shape[0], device=prompt_ids.device, dtype=torch.bool)

        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / float(temperature), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok)
            if eos >= 0:
                finished = finished | (next_tok.squeeze(-1) == eos)
                if bool(finished.all().item()):
                    break
            step_out = self.forward_with_taps(
                next_tok, past_key_values=past_kv, use_cache=True, quantize_kv=quantize_kv
            )
            past_kv = step_out["past_key_values"]
            next_logits = step_out["logits"][:, -1, :].float()

        return torch.cat(generated, dim=1)


# ----------------------------------------------------------------------
# Universal State-Dict Normalization & Auto-Detection
# ----------------------------------------------------------------------
def _normalize_cncm_state_dict_to_hf(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract clean dense donor HF-named weights from a CNCM checkpoint or safetensors dict."""
    out: Dict[str, torch.Tensor] = {}
    out["model.embed_tokens.weight"] = sd["exact_embed.weight"]
    out["model.norm.weight"] = sd["final_norm.weight"]
    if "exact_head.weight" in sd:
        out["lm_head.weight"] = sd["exact_head.weight"]

    layer_ids = sorted(
        {
            int(k.split(".")[1])
            for k in sd.keys()
            if k.startswith("blocks.") and k.split(".")[1].isdigit()
        }
    )
    for i in layer_ids:
        hp = f"model.layers.{i}"
        out[f"{hp}.input_layernorm.weight"] = sd[f"step_attn_norms.{i}.weight"]
        out[f"{hp}.post_attention_layernorm.weight"] = sd[f"step_ffn_norms.{i}.weight"]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[f"{hp}.self_attn.{proj}.weight"] = sd[f"blocks.{i}.attn.{proj}.weight"]
            if f"blocks.{i}.attn.{proj}.bias" in sd:
                out[f"{hp}.self_attn.{proj}.bias"] = sd[f"blocks.{i}.attn.{proj}.bias"]
        if f"blocks.{i}.attn.q_norm.weight" in sd:
            out[f"{hp}.self_attn.q_norm.weight"] = sd[f"blocks.{i}.attn.q_norm.weight"]
            out[f"{hp}.self_attn.k_norm.weight"] = sd[f"blocks.{i}.attn.k_norm.weight"]
        # Dense CNCM uses `blocks.{i}.mlp.gate_proj`, Sparse-Upcycled MoE uses `blocks.{i}.mlp.shared_expert.gate_proj`
        mlp_prefix = (
            f"blocks.{i}.mlp.shared_expert"
            if f"blocks.{i}.mlp.shared_expert.gate_proj.weight" in sd
            else f"blocks.{i}.mlp"
        )
        for ffn_p in ("gate_proj", "up_proj", "down_proj"):
            out[f"{hp}.mlp.{ffn_p}.weight"] = sd[f"{mlp_prefix}.{ffn_p}.weight"]
    return out


def infer_donor_config_from_state_dict(
    sd: Dict[str, torch.Tensor],
    hf_config: Optional[Dict[str, Any]] = None,
    donor_id: str = "local_donor",
) -> DonorConfig:
    """Infer exact `DonorConfig` from an HF state_dict + optional `config.json`."""
    cfg_json = hf_config or {}
    emb_w = sd["model.embed_tokens.weight"]
    vocab_size, hidden_size = int(emb_w.shape[0]), int(emb_w.shape[1])

    layer_ids = sorted(
        {
            int(k.split(".")[2])
            for k in sd.keys()
            if k.startswith("model.layers.") and k.split(".")[2].isdigit()
        }
    )
    num_layers = int(cfg_json.get("num_hidden_layers", len(layer_ids)))
    inter_size = int(
        cfg_json.get("intermediate_size", sd["model.layers.0.mlp.gate_proj.weight"].shape[0])
    )

    q_out = int(sd["model.layers.0.self_attn.q_proj.weight"].shape[0])
    k_out = int(sd["model.layers.0.self_attn.k_proj.weight"].shape[0])
    has_qkv_bias = "model.layers.0.self_attn.q_proj.bias" in sd
    has_o_bias = "model.layers.0.self_attn.o_proj.bias" in sd
    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in sd

    if has_qk_norm:
        head_dim = int(sd["model.layers.0.self_attn.q_norm.weight"].shape[0])
        n_heads = q_out // head_dim
        n_kv_heads = k_out // head_dim
    elif "num_attention_heads" in cfg_json:
        n_heads = int(cfg_json["num_attention_heads"])
        head_dim = int(cfg_json.get("head_dim") or (q_out // n_heads))
        n_kv_heads = int(cfg_json.get("num_key_value_heads", k_out // head_dim))
    else:
        # Infer GCD-consistent head_dim (prefer 64 or 128)
        head_dim = 64 if (q_out % 64 == 0 and k_out % 64 == 0) else math.gcd(q_out, k_out)
        if head_dim > 128 and head_dim % 128 == 0:
            head_dim = 128
        elif head_dim > 64 and head_dim % 64 == 0 and hidden_size <= 1536:
            head_dim = 64
        n_heads = q_out // head_dim
        n_kv_heads = k_out // head_dim

    model_type = str(cfg_json.get("model_type", "")).lower()
    if not model_type:
        if has_qk_norm:
            model_type = "qwen3"
        elif has_qkv_bias:
            model_type = "qwen2"
        else:
            model_type = "llama"

    is_gemma = model_type in ("gemma", "gemma2")
    default_theta = 1_000_000.0 if model_type in ("qwen2", "qwen3") else 100_000.0
    rope_theta = float(cfg_json.get("rope_theta", default_theta))
    rms_eps = float(cfg_json.get("rms_norm_eps", 1e-6 if model_type in ("qwen2", "qwen3", "gemma", "gemma2") else 1e-5))
    max_pos = int(cfg_json.get("max_position_embeddings", 8192))

    has_untied_head = "lm_head.weight" in sd and not torch.equal(
        sd["lm_head.weight"].float(), emb_w.float()
    )
    tie_emb = bool(cfg_json.get("tie_word_embeddings", not has_untied_head)) and (
        "lm_head.weight" not in sd or not has_untied_head
    )

    bos_id = cfg_json.get("bos_token_id", 1)
    eos_id = cfg_json.get("eos_token_id", 2)
    if isinstance(bos_id, list):
        bos_id = bos_id[0] if bos_id else 1
    if isinstance(eos_id, list):
        eos_id = eos_id[0] if eos_id else 2
    pad_id = cfg_json.get("pad_token_id")
    if pad_id is None:
        pad_id = 0

    return DonorConfig(
        arch=model_type,
        donor_id=str(donor_id),
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate_size=inter_size,
        rms_norm_eps=rms_eps,
        rope_theta=rope_theta,
        max_position_embeddings=max_pos,
        qkv_bias=has_qkv_bias,
        o_bias=has_o_bias,
        use_qk_norm=has_qk_norm,
        rms_norm_add_unit=is_gemma,
        embed_scale=math.sqrt(float(hidden_size)) if is_gemma else 1.0,
        hidden_act=str(cfg_json.get("hidden_act") or cfg_json.get("hidden_activation") or ("gelu_pytorch_tanh" if is_gemma else "silu")),
        attn_logit_softcapping=cfg_json.get("attn_logit_softcapping"),
        final_logit_softcapping=cfg_json.get("final_logit_softcapping"),
        sliding_window=cfg_json.get("sliding_window") if bool(cfg_json.get("use_sliding_window", model_type in ("mistral", "gemma2"))) else None,
        tie_word_embeddings=tie_emb,
        bos_token_id=int(bos_id),
        eos_token_id=int(eos_id),
        pad_token_id=int(pad_id),
    )


@torch.no_grad()
def populate_donor_runner(
    cfg: DonorConfig,
    sd: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> FrozenDonorRunner:
    """Copy normalized HF donor weights into `FrozenDonorRunner` and freeze."""
    runner = FrozenDonorRunner(cfg)
    runner.embed_tokens.weight.copy_(sd["model.embed_tokens.weight"].to(dtype=torch.float32))
    runner.norm.weight.copy_(sd["model.norm.weight"].to(dtype=torch.float32))
    if not cfg.tie_word_embeddings and "lm_head.weight" in sd:
        runner.lm_head.weight.copy_(sd["lm_head.weight"].to(dtype=torch.float32))

    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        blk = runner.layers[i]
        blk.input_layernorm.weight.copy_(sd[f"{p}.input_layernorm.weight"].to(dtype=torch.float32))
        blk.post_attention_layernorm.weight.copy_(sd[f"{p}.post_attention_layernorm.weight"].to(dtype=torch.float32))
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj: nn.Linear = getattr(blk, proj_name)
            proj.weight.copy_(sd[f"{p}.self_attn.{proj_name}.weight"].to(dtype=torch.float32))
            if proj.bias is not None and f"{p}.self_attn.{proj_name}.bias" in sd:
                proj.bias.copy_(sd[f"{p}.self_attn.{proj_name}.bias"].to(dtype=torch.float32))
        if blk.q_norm is not None and f"{p}.self_attn.q_norm.weight" in sd:
            blk.q_norm.weight.copy_(sd[f"{p}.self_attn.q_norm.weight"].to(dtype=torch.float32))
        if blk.k_norm is not None and f"{p}.self_attn.k_norm.weight" in sd:
            blk.k_norm.weight.copy_(sd[f"{p}.self_attn.k_norm.weight"].to(dtype=torch.float32))
        for ffn_name in ("gate_proj", "up_proj", "down_proj"):
            ffn_lin: nn.Linear = getattr(blk, ffn_name)
            ffn_lin.weight.copy_(sd[f"{p}.mlp.{ffn_name}.weight"].to(dtype=torch.float32))

    if dtype != torch.float32:
        runner = runner.to(dtype=dtype)
    return runner.freeze_all()


@torch.no_grad()
def verify_donor_parity(
    runner: FrozenDonorRunner,
    raw_hf_sd: Dict[str, torch.Tensor],
    seq_len: int = 16,
) -> Dict[str, float]:
    """Independent pure-functional parity check verifying 100% top-1 logit agreement."""
    cfg = runner.cfg
    device = runner.embed_tokens.weight.device
    torch.manual_seed(2026)
    input_ids = torch.randint(
        0, max(8, min(cfg.vocab_size, 10000)), (2, seq_len), device=device, dtype=torch.long
    )

    runner_logits = runner.forward_with_taps(input_ids)["logits"].float().cpu()

    # Independent FP32 functional forward directly from `raw_hf_sd`
    ids_cpu = input_ids.cpu()
    h = F.embedding(ids_cpu, raw_hf_sd["model.embed_tokens.weight"].float()) * float(cfg.embed_scale)
    bsz, s_len, _ = h.shape
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
    freqs = torch.outer(torch.arange(s_len, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]

    def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        scale = (1.0 + w) if cfg.rms_norm_add_unit else w
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)) * scale

    def _rot(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., cfg.head_dim // 2 :], x[..., : cfg.head_dim // 2]), dim=-1)

    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    q_idx = torch.arange(s_len).unsqueeze(1)
    k_idx = torch.arange(s_len).unsqueeze(0)
    full_causal = torch.where(k_idx <= q_idx, 0.0, -1e9).view(1, 1, s_len, s_len)

    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        xn = _rms(h, raw_hf_sd[f"{p}.input_layernorm.weight"].float())
        qb = raw_hf_sd[f"{p}.self_attn.q_proj.bias"].float() if f"{p}.self_attn.q_proj.bias" in raw_hf_sd else None
        kb = raw_hf_sd[f"{p}.self_attn.k_proj.bias"].float() if f"{p}.self_attn.k_proj.bias" in raw_hf_sd else None
        vb = raw_hf_sd[f"{p}.self_attn.v_proj.bias"].float() if f"{p}.self_attn.v_proj.bias" in raw_hf_sd else None
        ob = raw_hf_sd[f"{p}.self_attn.o_proj.bias"].float() if f"{p}.self_attn.o_proj.bias" in raw_hf_sd else None

        q = F.linear(xn, raw_hf_sd[f"{p}.self_attn.q_proj.weight"].float(), qb).view(bsz, s_len, cfg.num_attention_heads, cfg.head_dim)
        k = F.linear(xn, raw_hf_sd[f"{p}.self_attn.k_proj.weight"].float(), kb).view(bsz, s_len, cfg.num_key_value_heads, cfg.head_dim)
        v = F.linear(xn, raw_hf_sd[f"{p}.self_attn.v_proj.weight"].float(), vb).view(bsz, s_len, cfg.num_key_value_heads, cfg.head_dim)
        if cfg.use_qk_norm:
            q = _rms(q, raw_hf_sd[f"{p}.self_attn.q_norm.weight"].float())
            k = _rms(k, raw_hf_sd[f"{p}.self_attn.k_norm.weight"].float())
        q = (q * cos + _rot(q) * sin).transpose(1, 2)
        k = (k * cos + _rot(k) * sin).transpose(1, 2)
        v = v.transpose(1, 2)
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        mask_i = full_causal
        if cfg.sliding_window is not None and cfg.sliding_window > 0:
            if cfg.arch != "gemma2" or (i % 2 == 0):
                in_win = (k_idx <= q_idx) & (k_idx > (q_idx - cfg.sliding_window))
                mask_i = torch.where(in_win, 0.0, -1e9).view(1, 1, s_len, s_len)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(cfg.head_dim))
        if cfg.attn_logit_softcapping is not None and cfg.attn_logit_softcapping > 0.0:
            cap = float(cfg.attn_logit_softcapping)
            scores = torch.tanh(scores / cap) * cap
        probs = F.softmax(scores + mask_i, dim=-1)
        attn = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, s_len, cfg.num_attention_heads * cfg.head_dim)
        h = h + F.linear(attn, raw_hf_sd[f"{p}.self_attn.o_proj.weight"].float(), ob)

        xn2 = _rms(h, raw_hf_sd[f"{p}.post_attention_layernorm.weight"].float())
        g_raw = F.linear(xn2, raw_hf_sd[f"{p}.mlp.gate_proj.weight"].float())
        act = F.gelu(g_raw, approximate="tanh") if cfg.hidden_act in ("gelu", "gelu_pytorch_tanh", "geglu") else F.silu(g_raw)
        up = F.linear(xn2, raw_hf_sd[f"{p}.mlp.up_proj.weight"].float())
        h = h + F.linear(act * up, raw_hf_sd[f"{p}.mlp.down_proj.weight"].float())

    lm_w = (
        raw_hf_sd["model.embed_tokens.weight"].float()
        if cfg.tie_word_embeddings or "lm_head.weight" not in raw_hf_sd
        else raw_hf_sd["lm_head.weight"].float()
    )
    ref_logits = F.linear(_rms(h, raw_hf_sd["model.norm.weight"].float()), lm_w)
    if cfg.final_logit_softcapping is not None and cfg.final_logit_softcapping > 0.0:
        cap = float(cfg.final_logit_softcapping)
        ref_logits = torch.tanh(ref_logits / cap) * cap

    diff = (runner_logits - ref_logits).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    top1_agree = float((runner_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item())
    metrics = {"max_abs_diff": max_diff, "mean_abs_diff": mean_diff, "top1_agreement": top1_agree}
    if top1_agree < 0.999 or max_diff > 5e-2:
        raise AssertionError(f"Donor parity check failed for {cfg.donor_id}: {metrics}")
    return metrics


def load_universal_donor(
    source: str | Path,
    *,
    hf_token: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    verify_parity: bool = True,
) -> Tuple[FrozenDonorRunner, Dict[str, float]]:
    """Load any local `.safetensors`/`.pt`/directory or HF Hub donor into `FrozenDonorRunner`."""
    from safetensors.torch import load_file

    src_str = str(source)
    src_path = Path(src_str)
    hf_cfg: Optional[Dict[str, Any]] = None
    raw_sd: Dict[str, torch.Tensor]

    if src_path.is_file():
        if src_path.suffix == ".safetensors":
            raw_sd = load_file(str(src_path), device="cpu")
        else:
            ck = torch.load(str(src_path), map_location="cpu", weights_only=False)
            raw_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        cfg_candidate = src_path.parent / "config.json"
        if cfg_candidate.exists():
            hf_cfg = json.loads(cfg_candidate.read_text(encoding="utf-8"))
    elif src_path.is_dir():
        cfg_file = src_path / "config.json"
        if cfg_file.exists():
            hf_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        idx_file = src_path / "model.safetensors.index.json"
        single_sf = src_path / "model.safetensors"
        if single_sf.exists():
            raw_sd = load_file(str(single_sf), device="cpu")
        elif idx_file.exists():
            idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
            shards = sorted(set(idx_data.get("weight_map", {}).values()))
            raw_sd = {}
            for sh in shards:
                raw_sd.update(load_file(str(src_path / sh), device="cpu"))
        else:
            sf_files = sorted(src_path.glob("*.safetensors"))
            if not sf_files:
                raise FileNotFoundError(f"No `.safetensors` found in {src_path}")
            raw_sd = {}
            for sf in sf_files:
                raw_sd.update(load_file(str(sf), device="cpu"))
    else:
        from huggingface_hub import hf_hub_download

        try:
            cfg_dl = hf_hub_download(repo_id=src_str, filename="config.json", token=hf_token)
            hf_cfg = json.loads(Path(cfg_dl).read_text(encoding="utf-8"))
        except Exception:
            hf_cfg = None
        try:
            sf_dl = hf_hub_download(repo_id=src_str, filename="model.safetensors", token=hf_token)
            raw_sd = load_file(sf_dl, device="cpu")
        except Exception:
            idx_dl = hf_hub_download(
                repo_id=src_str, filename="model.safetensors.index.json", token=hf_token
            )
            idx_data = json.loads(Path(idx_dl).read_text(encoding="utf-8"))
            shards = sorted(set(idx_data.get("weight_map", {}).values()))
            raw_sd = {}
            for sh in shards:
                sh_dl = hf_hub_download(repo_id=src_str, filename=sh, token=hf_token)
                raw_sd.update(load_file(sh_dl, device="cpu"))

    if "exact_embed.weight" in raw_sd and "model.embed_tokens.weight" not in raw_sd:
        raw_sd = _normalize_cncm_state_dict_to_hf(raw_sd)
        if hf_cfg is None:
            hf_cfg = {"model_type": "cncm", "rope_theta": 100_000.0, "rms_norm_eps": 1e-5 if raw_sd["model.embed_tokens.weight"].shape[0] == 49152 else 1e-6}

    cfg = infer_donor_config_from_state_dict(raw_sd, hf_config=hf_cfg, donor_id=src_str)
    runner = populate_donor_runner(cfg, raw_sd, dtype=dtype)
    parity = verify_donor_parity(runner, raw_sd) if verify_parity else {}
    return runner, parity


def build_synthetic_donor(
    arch: str = "llama",
    vocab_size: int = 512,
    hidden_size: int = 96,
    num_layers: int = 4,
    seed: int = 42,
) -> Tuple[FrozenDonorRunner, Dict[str, torch.Tensor]]:
    """Build a small deterministic donor for unit-testing every architecture quirk offline."""
    torch.manual_seed(seed)
    arch_l = arch.lower()
    is_qwen2 = arch_l == "qwen2"
    is_qwen3 = arch_l == "qwen3"
    is_gemma2 = arch_l in ("gemma", "gemma2")

    # Test Qwen3 explicit head_dim != hidden_size / num_heads!
    n_heads = 4
    n_kv_heads = 2
    head_dim = 32 if is_qwen3 else (hidden_size // n_heads)
    inter_size = hidden_size * 2

    cfg = DonorConfig(
        arch=arch_l,
        donor_id=f"synthetic_{arch_l}",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate_size=inter_size,
        rms_norm_eps=1e-6 if (is_qwen2 or is_qwen3 or is_gemma2) else 1e-5,
        rope_theta=1_000_000.0 if (is_qwen2 or is_qwen3) else 10_000.0,
        max_position_embeddings=512,
        qkv_bias=is_qwen2,
        o_bias=False,
        use_qk_norm=is_qwen3,
        rms_norm_add_unit=is_gemma2,
        embed_scale=math.sqrt(float(hidden_size)) if is_gemma2 else 1.0,
        hidden_act="gelu_pytorch_tanh" if is_gemma2 else "silu",
        attn_logit_softcapping=50.0 if is_gemma2 else None,
        final_logit_softcapping=30.0 if is_gemma2 else None,
        sliding_window=32 if is_gemma2 else None,
        tie_word_embeddings=not is_qwen2,
    )

    sd: Dict[str, torch.Tensor] = {}
    sd["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size) * 0.02
    sd["model.norm.weight"] = torch.zeros(hidden_size) if is_gemma2 else torch.ones(hidden_size)
    if not cfg.tie_word_embeddings:
        sd["lm_head.weight"] = torch.randn(vocab_size, hidden_size) * 0.02

    for i in range(num_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = (torch.randn(hidden_size) * 0.05) if is_gemma2 else torch.ones(hidden_size)
        sd[f"{p}.post_attention_layernorm.weight"] = (torch.randn(hidden_size) * 0.05) if is_gemma2 else torch.ones(hidden_size)
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden_size) * 0.02
        sd[f"{p}.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden_size) * 0.02
        sd[f"{p}.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, hidden_size) * 0.02
        sd[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden_size, n_heads * head_dim) * 0.02
        if is_qwen2:
            sd[f"{p}.self_attn.q_proj.bias"] = torch.randn(n_heads * head_dim) * 0.02
            sd[f"{p}.self_attn.k_proj.bias"] = torch.randn(n_kv_heads * head_dim) * 0.02
            sd[f"{p}.self_attn.v_proj.bias"] = torch.randn(n_kv_heads * head_dim) * 0.02
        if is_qwen3:
            sd[f"{p}.self_attn.q_norm.weight"] = torch.ones(head_dim)
            sd[f"{p}.self_attn.k_norm.weight"] = torch.ones(head_dim)
        sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter_size, hidden_size) * 0.02
        sd[f"{p}.mlp.up_proj.weight"] = torch.randn(inter_size, hidden_size) * 0.02
        sd[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden_size, inter_size) * 0.02

    runner = populate_donor_runner(cfg, sd)
    return runner, sd


# ----------------------------------------------------------------------
# Offline Multi-Layer Hidden + Top-K Logit Memmap Cache
# ----------------------------------------------------------------------
class OfflineDonorCache:
    """Compact on-disk `memmap` cache of frozen donor multi-layer taps + target Top-K log-probs."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "cache_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing cache_meta.json in {self.cache_dir}")
        self.meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.num_samples = int(self.meta["num_samples"])
        self.prefix_len = int(self.meta["prefix_len"])
        self.chunk_size = int(self.meta["chunk_size"])
        self.seq_len = self.prefix_len + self.chunk_size
        self.hidden_size = int(self.meta["hidden_size"])
        self.top_k = int(self.meta["top_k"])
        self.id_dtype = np.dtype(self.meta["id_dtype"])

        self.tokens_mm = np.memmap(
            self.cache_dir / "tokens.bin",
            dtype=self.id_dtype,
            mode="r",
            shape=(self.num_samples, self.seq_len),
        )
        # Stores unit_taps for prefix positions [0 .. prefix_len - 1] (used by EAGLE-3 shifted Phase A)
        self.taps_mm = np.memmap(
            self.cache_dir / "unit_taps.bin",
            dtype=np.float16,
            mode="r",
            shape=(self.num_samples, self.prefix_len, 3, self.hidden_size),
        )
        self.topk_idx_mm = np.memmap(
            self.cache_dir / "topk_idx.bin",
            dtype=self.id_dtype,
            mode="r",
            shape=(self.num_samples, self.chunk_size, self.top_k),
        )
        self.topk_lp_mm = np.memmap(
            self.cache_dir / "topk_logprobs.bin",
            dtype=np.float16,
            mode="r",
            shape=(self.num_samples, self.chunk_size, self.top_k),
        )

    def __len__(self) -> int:
        return self.num_samples

    def close(self) -> None:
        """Release Win32 file-mapping handles cleanly so temporary cache dirs can be deleted on Windows."""
        for attr in ("tokens_mm", "taps_mm", "topk_idx_mm", "topk_lp_mm"):
            mm = getattr(self, attr, None)
            if mm is not None:
                raw_mmap = getattr(mm, "_mmap", None)
                if raw_mmap is not None:
                    try:
                        raw_mmap.close()
                    except Exception:
                        pass
                setattr(self, attr, None)

    def __enter__(self) -> "OfflineDonorCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_batch(
        self, indices: Sequence[int] | np.ndarray, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        idx = np.asarray(indices, dtype=np.int64)
        toks = torch.from_numpy(np.array(self.tokens_mm[idx], dtype=np.int64)).to(device=device)
        taps = torch.from_numpy(np.array(self.taps_mm[idx], dtype=np.float32)).to(device=device)
        tk_idx = torch.from_numpy(np.array(self.topk_idx_mm[idx], dtype=np.int64)).to(device=device)
        tk_lp = torch.from_numpy(np.array(self.topk_lp_mm[idx], dtype=np.float32)).to(device=device)
        return {
            "prefix_ids": toks[:, : self.prefix_len],
            "target_ids": toks[:, self.prefix_len :],
            "prefix_unit_taps": taps,
            "topk_idx": tk_idx,
            "topk_logprobs": tk_lp,
        }

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        donor: FrozenDonorRunner,
        sequences: torch.Tensor | np.ndarray,
        cache_dir: str | Path,
        *,
        prefix_len: int,
        chunk_size: int,
        top_k: int = 16,
        batch_size: int = 8,
    ) -> "OfflineDonorCache":
        """Run frozen donor once (`torch.no_grad()`) over `[N, prefix_len + chunk_size]` token sequences."""
        if prefix_len < 1 or chunk_size < 1:
            raise ValueError(f"prefix_len ({prefix_len}) and chunk_size ({chunk_size}) must be >= 1")

        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        seq_np = (
            sequences.detach().cpu().numpy()
            if isinstance(sequences, torch.Tensor)
            else np.asarray(sequences)
        )
        n_samples, seq_len = int(seq_np.shape[0]), int(seq_np.shape[1])
        if seq_len != prefix_len + chunk_size:
            raise ValueError(
                f"Expected sequences shape [N, {prefix_len + chunk_size}], got {seq_np.shape}"
            )

        vocab_size = donor.cfg.vocab_size
        eff_topk = min(int(top_k), vocab_size)
        id_dtype_str = "uint16" if vocab_size < 65535 else "int32"
        id_dtype = np.dtype(id_dtype_str)
        hidden = donor.cfg.hidden_size
        device = donor.embed_tokens.weight.device

        tokens_mm = np.memmap(
            out_dir / "tokens.bin", dtype=id_dtype, mode="w+", shape=(n_samples, seq_len)
        )
        taps_mm = np.memmap(
            out_dir / "unit_taps.bin",
            dtype=np.float16,
            mode="w+",
            shape=(n_samples, prefix_len, 3, hidden),
        )
        topk_idx_mm = np.memmap(
            out_dir / "topk_idx.bin",
            dtype=id_dtype,
            mode="w+",
            shape=(n_samples, chunk_size, eff_topk),
        )
        topk_lp_mm = np.memmap(
            out_dir / "topk_logprobs.bin",
            dtype=np.float16,
            mode="w+",
            shape=(n_samples, chunk_size, eff_topk),
        )

        tokens_mm[:] = seq_np.astype(id_dtype, copy=False)

        donor.eval()
        for start in range(0, n_samples, max(1, int(batch_size))):
            end = min(n_samples, start + max(1, int(batch_size)))
            batch_ids = torch.from_numpy(seq_np[start:end].astype(np.int64)).to(device=device)
            # We only need donor forward up to index `prefix_len + chunk_size - 2` inclusive
            # (length `seq_len - 1`) because `logits[:, prefix_len - 1 + j]` predicts `target_ids[:, j]`!
            run_ids = batch_ids[:, :-1]
            res = donor.forward_with_taps(run_ids, use_cache=False)

            # Prefix taps [0 .. prefix_len - 1] (Note: run_ids has length prefix_len + chunk_size - 1 >= prefix_len)
            p_taps = res["unit_taps"][:, :prefix_len, :, :].float().clamp(-65000.0, 65000.0)
            taps_mm[start:end] = p_taps.cpu().numpy().astype(np.float16)

            # Target teacher logits: positions [prefix_len - 1 .. prefix_len + chunk_size - 2] predict target [0 .. chunk_size - 1]
            tgt_logits = res["logits"][:, prefix_len - 1 : prefix_len - 1 + chunk_size, :].float()
            lse = torch.logsumexp(tgt_logits, dim=-1, keepdim=True)
            tk_vals, tk_indices = torch.topk(tgt_logits, k=eff_topk, dim=-1)
            tk_logprobs = (tk_vals - lse).clamp(min=-65000.0, max=0.0)

            topk_idx_mm[start:end] = tk_indices.cpu().numpy().astype(id_dtype)
            topk_lp_mm[start:end] = tk_logprobs.cpu().numpy().astype(np.float16)

        tokens_mm.flush()
        taps_mm.flush()
        topk_idx_mm.flush()
        topk_lp_mm.flush()
        for mm in (tokens_mm, taps_mm, topk_idx_mm, topk_lp_mm):
            if getattr(mm, "_mmap", None) is not None:
                mm._mmap.close()
        del tokens_mm, taps_mm, topk_idx_mm, topk_lp_mm

        meta = {
            "num_samples": n_samples,
            "prefix_len": prefix_len,
            "chunk_size": chunk_size,
            "hidden_size": hidden,
            "vocab_size": vocab_size,
            "top_k": eff_topk,
            "id_dtype": id_dtype_str,
            "tap_layers": list(donor.cfg.tap_layers),
            "donor_config": donor.cfg.to_dict(),
        }
        (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return cls(out_dir)
