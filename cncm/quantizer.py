"""Multiscale Residual Finite Scalar Quantization (MultiscaleRFSQ) for CNCM.

Replaces vanilla NCM's large FP32+EMA VQ codebook tables (18.87M params / 151 MB)
with zero-codebook-table implicit FSQ grids:
  - Zero codebook parameter tables and zero EMA buffers (DDP & grad-accum safe)
  - FP16 / BF16 / INT8 / INT4 friendly (smooth bounded tanh + STE rounding)
  - Depthwise-separable cascaded L1 down/up convolutions (avg-pool / nearest init)
  - Depthwise-separable per-level Phi residual convolutions
  - Nested dropout (fine_drop) + linear conditioning corruption for Stage 2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cncm.config import CNCMConfig


def factorize_stride(ratio: int) -> List[int]:
    """Factorize an integer stride >= 1 into small cascaded factors <= 4."""
    if ratio <= 1:
        return []
    factors: List[int] = []
    rem = int(ratio)
    for f in (4, 3, 2):
        while rem % f == 0:
            factors.append(f)
            rem //= f
    if rem > 1:
        factors.append(rem)
    return factors


class DepthwiseSepCascadeDown(nn.Module):
    """Depthwise-separable cascaded strided Conv1d downsampler (avg-pool init)."""

    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        self.dim = dim
        self.ratio = int(ratio)
        factors = factorize_stride(self.ratio)
        self.stages = nn.ModuleList()
        for s in factors:
            dw = nn.Conv1d(dim, dim, kernel_size=s, stride=s, groups=dim, bias=False)
            pw = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
            with torch.no_grad():
                dw.weight.fill_(1.0 / float(s))
                pw.weight.zero_()
                idx = torch.arange(dim)
                pw.weight[idx, idx, 0] = 1.0
            self.stages.append(nn.Sequential(dw, pw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] -> [B, C, L // ratio]."""
        for stage in self.stages:
            x = stage(x)
        return x


class DepthwiseSepCascadeUp(nn.Module):
    """Depthwise-separable cascaded ConvTranspose1d upsampler (nearest-repeat init)."""

    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        self.dim = dim
        self.ratio = int(ratio)
        factors = list(reversed(factorize_stride(self.ratio)))
        self.stages = nn.ModuleList()
        for s in factors:
            dw = nn.ConvTranspose1d(
                dim, dim, kernel_size=s, stride=s, groups=dim, bias=False
            )
            pw = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
            with torch.no_grad():
                dw.weight.fill_(1.0)
                pw.weight.zero_()
                idx = torch.arange(dim)
                pw.weight[idx, idx, 0] = 1.0
            self.stages.append(nn.Sequential(dw, pw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L // ratio] -> [B, C, L]."""
        for stage in self.stages:
            x = stage(x)
        return x


class DepthwiseSepPhi(nn.Module):
    """Per-level depthwise-separable residual refinement conv: (1-r)*x + r*Conv(x)."""

    def __init__(self, dim: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.ratio = float(ratio)
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        with torch.no_grad():
            self.dw.weight.zero_()
            self.dw.weight[:, 0, 0] = 0.25
            self.dw.weight[:, 0, 1] = 0.50
            self.dw.weight[:, 0, 2] = 0.25
            nn.init.zeros_(self.pw.weight)
            nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] -> [B, C, L]."""
        if self.ratio <= 0.0:
            return x
        return (1.0 - self.ratio) * x + self.ratio * self.pw(self.dw(x))


class FSQLevel(nn.Module):
    """Single-level Finite Scalar Quantizer projection (zero codebook table)."""

    def __init__(self, d_vq: int, fsq_levels: Tuple[int, ...]) -> None:
        super().__init__()
        self.d_vq = int(d_vq)
        self.fsq_levels = tuple(int(v) for v in fsq_levels)
        self.d_q = len(self.fsq_levels)

        self.proj_in = nn.Linear(self.d_vq, self.d_q, bias=True)
        self.proj_out = nn.Linear(self.d_q, self.d_vq, bias=True)

        # Smooth, balanced grid geometry valid for every L_j >= 2 (odd or even)
        levels_f = torch.tensor(self.fsq_levels, dtype=torch.float32)
        center = (levels_f - 1.0) * 0.5
        half_span = (levels_f - 1.0) * 0.5
        max_radius = (levels_f * 0.5) * (1.0 - 1e-3)
        max_digit = torch.tensor([l - 1 for l in self.fsq_levels], dtype=torch.long)

        self.register_buffer("center", center, persistent=True)
        self.register_buffer("half_span", half_span, persistent=True)
        self.register_buffer("max_radius", max_radius, persistent=True)
        self.register_buffer("max_digit", max_digit, persistent=True)

    def quantize(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize [B, l_k, d_vq] -> (u_q [B, l_k, d_vq], digits [B, l_k, d_q])."""
        h = self.proj_in(u)
        # Perform bounding + STE rounding in FP32 for FP16/BF16 numerical safety
        h_f = h.float()
        cont_idx = self.center + self.max_radius * torch.tanh(h_f)
        rounded = torch.round(cont_idx)
        digits = torch.minimum(rounded.to(torch.long).clamp(min=0), self.max_digit)

        # Straight-through estimator with exact unit Jacobian across round()
        quantized_idx = cont_idx + (rounded - cont_idx).detach()
        z_q_norm = ((quantized_idx - self.center) / self.half_span).to(u.dtype)
        u_q = self.proj_out(z_q_norm)
        return u_q, digits

    def dequantize(
        self, digits: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Map integer FSQ digits [..., d_q] back to [..., d_vq]."""
        target_dtype = dtype if dtype is not None else self.proj_out.weight.dtype
        z_q_norm = ((digits.float() - self.center) / self.half_span).to(target_dtype)
        return self.proj_out(z_q_norm)


class MultiscaleRFSQ(nn.Module):
    """Multiscale Residual Finite Scalar Quantizer with nested dropout & corruption."""

    def __init__(self, cfg: CNCMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_vq = cfg.d_vq
        self.chunk_size = cfg.chunk_size
        self.levels = tuple(cfg.levels)
        self.num_levels = len(self.levels)
        self.fsq_levels = tuple(cfg.fsq_levels)
        self.d_q = len(self.fsq_levels)
        self.codebook_size = math.prod(self.fsq_levels)
        self.fine_drop_prob = cfg.fine_drop_prob
        self.fine_drop_min_keep = max(1, min(cfg.fine_drop_min_keep, self.num_levels))
        self.commit_beta = cfg.commit_beta

        # Mixed-radix bases for exact bijection between digit vectors and flat IDs
        basis: List[int] = []
        running = 1
        for l_j in reversed(self.fsq_levels):
            basis.append(running)
            running *= l_j
        basis.reverse()

        self.register_buffer(
            "radix_basis", torch.tensor(basis, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "fsq_levels_tensor",
            torch.tensor(self.fsq_levels, dtype=torch.long),
            persistent=True,
        )

        # Level-0 learned depthwise-separable down/up cascades
        l0_ratio = self.chunk_size // self.levels[0]
        self.down_l1 = DepthwiseSepCascadeDown(self.d_vq, l0_ratio)
        self.up_l1 = DepthwiseSepCascadeUp(self.d_vq, l0_ratio)

        # Per-level FSQ projections and per-level Phi residual convolutions
        self.fsq_modules = nn.ModuleList(
            [FSQLevel(self.d_vq, self.fsq_levels) for _ in range(self.num_levels)]
        )
        self.phi_modules = nn.ModuleList(
            [
                DepthwiseSepPhi(self.d_vq, ratio=cfg.phi_ratio)
                for _ in range(self.num_levels)
            ]
        )

    # ------------------------------------------------------------------
    # Bijective flat <-> digit conversions
    # ------------------------------------------------------------------
    def digits_to_flat(self, digits: torch.Tensor) -> torch.Tensor:
        """Convert FSQ digit indices [..., d_q] into flat code IDs [...] in [0, V-1]."""
        return (digits.to(torch.long) * self.radix_basis).sum(dim=-1)

    def flat_to_digits(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat code IDs [...] into FSQ digit indices [..., d_q]."""
        flat_l = flat.to(torch.long).unsqueeze(-1)
        return torch.div(flat_l, self.radix_basis, rounding_mode="floor") % self.fsq_levels_tensor

    # ------------------------------------------------------------------
    # Internal multiscale resampling helpers (pure PyTorch / ONNX safe)
    # ------------------------------------------------------------------
    def _downsample(self, x_bcl: torch.Tensor, level_idx: int) -> torch.Tensor:
        target_len = self.levels[level_idx]
        if level_idx == 0:
            return self.down_l1(x_bcl)
        if target_len == self.chunk_size:
            return x_bcl
        if target_len == 1:
            return x_bcl.mean(dim=-1, keepdim=True)
        return F.interpolate(x_bcl, size=target_len, mode="area")

    def _upsample(self, x_bcl: torch.Tensor, level_idx: int) -> torch.Tensor:
        if level_idx == 0:
            return self.up_l1(x_bcl)
        cur_len = x_bcl.shape[-1]
        if cur_len == self.chunk_size:
            return x_bcl
        if cur_len == 1:
            return x_bcl.expand(-1, -1, self.chunk_size)
        return F.interpolate(
            x_bcl, size=self.chunk_size, mode="linear", align_corners=False
        )

    # ------------------------------------------------------------------
    # Decode / reconstruction / Stage-2 conditioning helpers
    # ------------------------------------------------------------------
    def decode_level(
        self,
        digits_k: torch.Tensor,
        level_idx: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Decode a single level's digits [B, l_k, d_q] into residual delta [B, L, d_vq]."""
        u_q = self.fsq_modules[level_idx].dequantize(digits_k, dtype=dtype)
        h_up = self._upsample(u_q.transpose(1, 2), level_idx)
        delta = self.phi_modules[level_idx](h_up).transpose(1, 2)
        return delta

    def reconstruct_from_codes(
        self,
        codes_list: Sequence[torch.Tensor],
        up_to_level: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Accumulate z_hat [B, L, d_vq] from a prefix or full list of code levels."""
        n_use = len(codes_list) if up_to_level is None else min(up_to_level, len(codes_list))
        if n_use <= 0:
            raise ValueError("codes_list must contain at least one level tensor")
        z_hat = self.decode_level(codes_list[0], 0, dtype=dtype)
        for k in range(1, n_use):
            z_hat = z_hat + self.decode_level(codes_list[k], k, dtype=dtype)
        return z_hat

    def get_level_conditioning(
        self, z_hat_prev: torch.Tensor, level_idx: int
    ) -> torch.Tensor:
        """Area-pool accumulated z_hat [B, L, d_vq] to length l_{level_idx}."""
        target_len = self.levels[level_idx]
        if target_len == self.chunk_size:
            return z_hat_prev
        if target_len == 1:
            return z_hat_prev.mean(dim=1, keepdim=True)
        return F.interpolate(
            z_hat_prev.transpose(1, 2), size=target_len, mode="area"
        ).transpose(1, 2)

    def corrupt_codes(
        self,
        codes_list: Sequence[torch.Tensor],
        corrupt_rate: float = 0.5,
    ) -> List[torch.Tensor]:
        """Apply NCM linearly decreasing conditioning corruption (mean == corrupt_rate).

        Only conditioning inputs are corrupted; Stage 2 prediction targets stay clean.
        """
        if corrupt_rate <= 0.0:
            return [c for c in codes_list]

        n_cond = max(len(codes_list) - 1, 1)
        corrupted: List[torch.Tensor] = []
        for k, digits in enumerate(codes_list):
            if k >= len(codes_list) - 1:
                corrupted.append(digits)
                continue
            # Linear schedule across levels 0..K-2 with mean exactly equal to corrupt_rate
            if n_cond > 1:
                p_k = corrupt_rate * (1.5 - float(k) / float(n_cond - 1))
            else:
                p_k = corrupt_rate
            p_k = max(0.0, min(0.95, p_k))
            if p_k <= 0.0:
                corrupted.append(digits)
                continue

            mask = torch.rand(digits.shape[:-1], device=digits.device) < p_k
            if not mask.any():
                corrupted.append(digits)
                continue

            # Uniform random digits per FSQ coordinate in [0, L_j - 1]
            rand_u = torch.rand(digits.shape, device=digits.device)
            rand_digits = torch.floor(
                rand_u * self.fsq_levels_tensor.to(rand_u.dtype)
            ).to(torch.long)
            rand_digits = torch.minimum(rand_digits, self.fsq_levels_tensor - 1)
            corrupted.append(torch.where(mask.unsqueeze(-1), rand_digits, digits))
        return corrupted

    def build_stage2_conditioning(
        self,
        target_codes: Sequence[torch.Tensor],
        corrupt_rate: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ) -> List[torch.Tensor]:
        """Build Stage 2 teacher-forced conditioning tensors for levels 1..len-1."""
        cond_codes = (
            self.corrupt_codes(target_codes, corrupt_rate=corrupt_rate)
            if corrupt_rate > 0.0
            else list(target_codes)
        )
        n_levels = len(cond_codes)
        if n_levels <= 1:
            return []

        cond_list: List[torch.Tensor] = []
        z_hat: Optional[torch.Tensor] = None
        for k in range(1, n_levels):
            delta_prev = self.decode_level(cond_codes[k - 1], k - 1, dtype=dtype)
            z_hat = delta_prev if z_hat is None else (z_hat + delta_prev)
            cond_list.append(self.get_level_conditioning(z_hat, k))
        return cond_list

    # ------------------------------------------------------------------
    # Stage 1 forward & encoding
    # ------------------------------------------------------------------
    def encode(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Deterministic multiscale encoding: z [B, L, d_vq] -> list of [B, l_k, d_q]."""
        codes: List[torch.Tensor] = []
        z_hat = torch.zeros_like(z)
        for k in range(self.num_levels):
            z_rest = z - z_hat
            u_k = self._downsample(z_rest.transpose(1, 2), k).transpose(1, 2)
            u_q, digits_k = self.fsq_modules[k].quantize(u_k)
            h_up = self._upsample(u_q.transpose(1, 2), k)
            delta = self.phi_modules[k](h_up).transpose(1, 2)
            z_hat = z_hat + delta
            codes.append(digits_k)
        return codes

    def forward(
        self,
        z: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        apply_fine_drop: bool = True,
    ) -> Dict[str, object]:
        """Quantize z [B, L, d_vq] through all K residual FSQ levels.

        Returns a dict containing:
          - z_hat: [B, L, d_vq] active quantized representation (with nested dropout)
          - z_hat_full: [B, L, d_vq] full-depth quantized representation
          - codes: list[Tensor], each [B, l_k, d_q] int64
          - commit_loss: scalar Tensor (commitment + per-level residual MSE)
          - active_code_ratio: 0-d detached float Tensor (mean unique codes / codebook_size)
          - batch_utilization: 0-d detached float Tensor (normalized by batch positions)
          - cutoff_level: int (1..K)
        """
        if z.shape[1] != self.chunk_size:
            raise ValueError(
                f"Expected sequence length chunk_size={self.chunk_size}, got {z.shape[1]}"
            )

        # Decide nested-dropout cutoff level (1-indexed in [min_keep, K])
        if self.training and apply_fine_drop and self.fine_drop_prob > 0.0:
            if torch.rand((), device=z.device).item() < self.fine_drop_prob:
                cutoff = int(
                    torch.randint(
                        self.fine_drop_min_keep,
                        self.num_levels + 1,
                        (1,),
                        device=z.device,
                    ).item()
                )
            else:
                cutoff = self.num_levels
        else:
            cutoff = self.num_levels

        mask_weight: Optional[torch.Tensor] = None
        mask_denom: Optional[torch.Tensor] = None
        if attention_mask is not None:
            mask_weight = attention_mask.to(dtype=z.dtype).unsqueeze(-1)
            mask_denom = (mask_weight.sum() * self.d_vq).clamp(min=1.0)

        z_hat = torch.zeros_like(z)
        z_hat_active: Optional[torch.Tensor] = None
        z_hat_levels: List[torch.Tensor] = []
        codes: List[torch.Tensor] = []
        commit_loss_acc = z.new_zeros(())
        active_ratios: List[float] = []
        batch_utils: List[float] = []

        for k in range(self.num_levels):
            # Detaching z_hat in residual prevents cross-level STE blowup while keeping
            # direct, bounded STE gradients into every level's FSQ + Phi + encoder.
            z_rest = z - z_hat.detach()
            u_k = self._downsample(z_rest.transpose(1, 2), k).transpose(1, 2)
            u_q, digits_k = self.fsq_modules[k].quantize(u_k)
            h_up = self._upsample(u_q.transpose(1, 2), k)
            delta = self.phi_modules[k](h_up).transpose(1, 2)
            z_hat = z_hat + delta
            z_hat_levels.append(z_hat)
            codes.append(digits_k)

            if k + 1 == cutoff:
                z_hat_active = z_hat

            # Per-level commitment + residual reconstruction loss
            if mask_weight is not None and mask_denom is not None:
                enc_err = ((z - z_hat.detach()) ** 2 * mask_weight).sum() / mask_denom
                dec_err = ((z_hat - z.detach()) ** 2 * mask_weight).sum() / mask_denom
            else:
                enc_err = F.mse_loss(z, z_hat.detach())
                dec_err = F.mse_loss(z_hat, z.detach())
            commit_loss_acc = commit_loss_acc + self.commit_beta * enc_err + dec_err

            # Track implicit FSQ grid utilization without extra buffers
            with torch.no_grad():
                flat_k = self.digits_to_flat(digits_k).reshape(-1)
                n_unique = int(torch.unique(flat_k).numel())
                active_ratios.append(n_unique / float(self.codebook_size))
                denom_batch = float(min(flat_k.numel(), self.codebook_size))
                batch_utils.append(n_unique / max(denom_batch, 1.0))

        commit_loss = commit_loss_acc / float(self.num_levels)
        assert z_hat_active is not None

        active_code_ratio = torch.tensor(
            sum(active_ratios) / float(len(active_ratios)),
            device=z.device,
            dtype=torch.float32,
        )
        batch_utilization = torch.tensor(
            sum(batch_utils) / float(len(batch_utils)),
            device=z.device,
            dtype=torch.float32,
        )

        return {
            "z_hat": z_hat_active,
            "z_hat_full": z_hat,
            "z_hat_levels": z_hat_levels,
            "codes": codes,
            "commit_loss": commit_loss,
            "active_code_ratio": active_code_ratio,
            "batch_utilization": batch_utilization,
            "per_level_batch_utilization": batch_utils,
            "per_level_active_ratio": active_ratios,
            "cutoff_level": cutoff,
        }
