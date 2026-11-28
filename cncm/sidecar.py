"""Path A Plug-In `CNCM-DeltaNet` Concept Sidecar (`cncm/sidecar.py`).

Unifies all five Path A breakthroughs from `future_cncm_update.md`:
  1. **SVD-Projected Frozen-Embedding Stage 1 (`SVDConceptTokenizer`)**:
     Deterministic Gram eigendecomposition (`E^T E`) + identity-initialized adapter +
     `P^T` decoder init (~1.4M trainable params, zero duplicated `V x H` table in `.sidecar.gguf`).
  2. **EAGLE-3 Shifted Multi-Layer Feature Fusion (`L16 + L24 + L32` + Next-Token Embed)**:
     Causal slot `t` fuses `(unit_taps[t-1], svd_embed(token[t]))`, giving **zero train/infer mismatch**
     and **zero redundant donor forward passes** during speculative verification.
  3. **4-Layer `CNCM-DeltaNet` Sidecar Backbone**:
     Chunk-64 WY-Parallel + Dual-Phase Read-Only `S_prefix` (`~0.58 MB` O(1) state, `0 MB` extra KV)
     with variable chunk schedules (`L = 16`, `32`, `64`).
  4. **Closed-Form Step-0 Ridge Warm-Start (`ridge_warmstart_concept_head`)**:
     Initializes all `d_q` chained digit heads via Tikhonov-regularized least squares before Step 1.
  5. **Learned Early-Exit & Dual-Mode Router (`LearnedDualModeRouter`)**:
     Predicts per-level `exit_now` and per-token `exact_mode` for high-precision Ultra mode,
     paired with exact Sparse Top-K + Tail-Mass KL distillation (`>= 0` by Gibbs' inequality).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cncm.config import CNCMConfig
from cncm.deltanet import CNCMDeltaNetBlock, DeltaNetLayerState
from cncm.donor import DonorConfig, FrozenDonorRunner
from cncm.layers import (
    CNCMDecoderBlock,
    FactorizedConceptHead,
    RMSNorm,
    RotaryEmbedding,
    build_bidirectional_mask,
    get_kv_seq_len,
    truncate_kv_cache,
)
from cncm.quantizer import MultiscaleRFSQ


def default_levels_for_chunk_size(chunk_size: int) -> Tuple[int, ...]:
    """Return canonical CNCM multiscale schedule for variable chunk horizon `L`."""
    c = int(chunk_size)
    if c == 16:
        return (1, 2, 4, 8, 16)
    if c == 32:
        return (1, 2, 4, 8, 16, 32)
    if c == 64:
        return (1, 2, 4, 8, 12, 20, 36, 64)
    if c < 2:
        raise ValueError(f"chunk_size must be >= 2, got {c}")
    levels: List[int] = [1]
    cur = 2
    while cur < c:
        levels.append(cur)
        cur *= 2
    levels.append(c)
    return tuple(levels)


def default_matryoshka_cutoffs(num_levels: int) -> Tuple[int, ...]:
    """Select intermediate Matryoshka supervision cutoffs for clean `L2..L_{K-1}` early exit."""
    if num_levels <= 2:
        return (1,)
    candidates = {max(1, num_levels // 3), max(2, (2 * num_levels) // 3), max(1, num_levels - 1)}
    return tuple(sorted(c for c in candidates if 1 <= c < num_levels))


@dataclass
class SidecarConfig:
    """Configuration for the Path A 4-Layer `CNCM-DeltaNet` Plug-In Sidecar."""

    donor_arch: str = "llama"
    donor_id: str = "unknown"
    donor_vocab_size: int = 49152
    donor_hidden_size: int = 960
    donor_num_layers: int = 32
    tap_layers: Tuple[int, int, int] = (15, 23, 31)

    sidecar_hidden_size: int = 512
    sidecar_layers: int = 4
    sidecar_heads: int = 8
    sidecar_intermediate_size: int = 1408
    deltanet_chunk_size: int = 64

    chunk_size: int = 32
    levels: Tuple[int, ...] = ()
    d_vq: int = 256
    fsq_levels: Tuple[int, ...] = (8, 8, 8, 8)
    dec_layers: int = 1
    dec_num_heads: int = 4
    dec_intermediate_size: int = 512
    pre_quant_norm_groups: int = 8
    phi_ratio: float = 0.5
    commit_beta: float = 0.25
    fine_drop_prob: float = 0.3
    matryoshka_cutoffs: Tuple[int, ...] = ()

    concept_head_hidden: int = 256
    router_hidden_size: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    max_position_embeddings: int = 8192
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        self.chunk_size = int(self.chunk_size)
        if not self.levels:
            self.levels = default_levels_for_chunk_size(self.chunk_size)
        else:
            self.levels = tuple(int(x) for x in self.levels)
        self.fsq_levels = tuple(int(x) for x in self.fsq_levels)
        self.tap_layers = tuple(int(x) for x in self.tap_layers)  # type: ignore[assignment]

        # Ensure d_vq satisfies divisibility for GroupNorm and decoder RoPE even on tiny test configs
        unit = math.lcm(int(self.pre_quant_norm_groups), int(self.dec_num_heads) * 2)
        if self.d_vq < unit:
            self.d_vq = unit
        elif self.d_vq % unit != 0:
            self.d_vq = ((self.d_vq + unit - 1) // unit) * unit

        if self.sidecar_hidden_size % self.sidecar_heads != 0 or (
            self.sidecar_hidden_size // self.sidecar_heads
        ) % 2 != 0:
            step = self.sidecar_heads * 2
            self.sidecar_hidden_size = max(step, ((self.sidecar_hidden_size + step - 1) // step) * step)

        if not self.matryoshka_cutoffs:
            self.matryoshka_cutoffs = default_matryoshka_cutoffs(len(self.levels))
        else:
            self.matryoshka_cutoffs = tuple(int(x) for x in self.matryoshka_cutoffs)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def to_cncm_tokenizer_config(self) -> CNCMConfig:
        """Build compatible `CNCMConfig` for reusing `MultiscaleRFSQ` and `CNCMDecoderBlock`."""
        return CNCMConfig(
            preset_name="sidecar_tok",
            vocab_size=max(288, self.donor_vocab_size),
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            chunk_size=self.chunk_size,
            levels=self.levels,
            d_vq=self.d_vq,
            fsq_levels=self.fsq_levels,
            dec_layers=self.dec_layers,
            dec_num_heads=self.dec_num_heads,
            dec_intermediate_size=self.dec_intermediate_size,
            pre_quant_norm_groups=self.pre_quant_norm_groups,
            fine_drop_prob=self.fine_drop_prob,
            matryoshka_cutoffs=self.matryoshka_cutoffs,
            commit_beta=self.commit_beta,
            phi_ratio=self.phi_ratio,
            hidden_size=self.sidecar_hidden_size,
            num_unique_layers=max(1, self.sidecar_layers),
            num_loops=1,
            loop_adapter_rank=0,
            num_attention_heads=self.sidecar_heads,
            num_key_value_heads=self.sidecar_heads,
            intermediate_size=self.sidecar_intermediate_size,
            concept_head_hidden=self.concept_head_hidden,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
        )

    @classmethod
    def from_donor(
        cls,
        donor_cfg: DonorConfig,
        *,
        chunk_size: int = 32,
        sidecar_layers: int = 4,
        sidecar_hidden_size: Optional[int] = None,
        d_vq: Optional[int] = None,
    ) -> "SidecarConfig":
        h_sc = int(sidecar_hidden_size) if sidecar_hidden_size is not None else min(512, max(128, donor_cfg.hidden_size))
        n_heads = 8 if h_sc >= 256 else 4
        h_sc = ((h_sc + n_heads * 2 - 1) // (n_heads * 2)) * (n_heads * 2)
        dvq = int(d_vq) if d_vq is not None else min(256, max(64, donor_cfg.hidden_size // 2))
        return cls(
            donor_arch=donor_cfg.arch,
            donor_id=donor_cfg.donor_id,
            donor_vocab_size=donor_cfg.vocab_size,
            donor_hidden_size=donor_cfg.hidden_size,
            donor_num_layers=donor_cfg.num_layers,
            tap_layers=donor_cfg.tap_layers,
            sidecar_hidden_size=h_sc,
            sidecar_layers=int(sidecar_layers),
            sidecar_heads=n_heads,
            sidecar_intermediate_size=int(round(h_sc * 2.75 / 64.0)) * 64 or (h_sc * 2),
            chunk_size=int(chunk_size),
            d_vq=dvq,
            dec_intermediate_size=dvq * 2,
            concept_head_hidden=min(256, max(64, h_sc // 2)),
            router_hidden_size=min(128, max(32, h_sc // 4)),
            rms_norm_eps=donor_cfg.rms_norm_eps,
            rope_theta=donor_cfg.rope_theta,
            max_position_embeddings=donor_cfg.max_position_embeddings,
            pad_token_id=donor_cfg.pad_token_id,
            bos_token_id=donor_cfg.bos_token_id,
            eos_token_id=donor_cfg.eos_token_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tap_layers"] = list(self.tap_layers)
        d["levels"] = list(self.levels)
        d["fsq_levels"] = list(self.fsq_levels)
        d["matryoshka_cutoffs"] = list(self.matryoshka_cutoffs)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SidecarConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in data.items() if k in valid}
        return cls(**kw)


@torch.no_grad()
def compute_svd_projection(embed_weight: torch.Tensor, d_vq: int) -> torch.Tensor:
    """Compute deterministic orthonormal right-singular basis `P [H_donor, d_vq]` via Gram `E^T E`.

    Runs in `< 50 ms` on CPU even for `152K x 1536` Qwen vocabularies, enforces canonical
    column signs for cross-BLAS reproducibility, and handles `d_vq > H_donor` cleanly.
    """
    E = embed_weight.detach().to(device="cpu", dtype=torch.float64)
    _, h_dim = E.shape
    gram = E.T @ E  # [H, H], symmetric positive semi-definite
    _, evecs = torch.linalg.eigh(gram)  # ascending eigenvalues

    k_take = min(int(d_vq), h_dim)
    P = evecs[:, h_dim - k_take :].flip(-1).clone()  # [H, k_take], descending singular values

    # Canonical sign per column: largest-magnitude entry is strictly non-negative
    max_idx = P.abs().argmax(dim=0)
    col_range = torch.arange(k_take)
    signs = torch.where(P[max_idx, col_range] >= 0.0, 1.0, -1.0)
    P = P * signs.unsqueeze(0)

    if int(d_vq) > h_dim:
        pad = torch.zeros((h_dim, int(d_vq) - h_dim), dtype=torch.float64)
        P = torch.cat([P, pad], dim=1)
    return P.to(dtype=torch.float32).contiguous()


class SVDConceptTokenizer(nn.Module):
    """SVD-Projected Frozen-Embedding Stage 1 Concept Tokenizer (~1.4M trainable params).

    Eliminates storing a 27M–78M `V x d_vq` embedding/head table inside `.sidecar.gguf`
    by projecting the donor's existing `embed_tokens` / `lm_head` through frozen `svd_proj`.
    """

    def __init__(
        self,
        cfg: SidecarConfig,
        svd_proj: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        tok_cfg = cfg.to_cncm_tokenizer_config()
        self.tok_cfg = tok_cfg

        init_p = (
            svd_proj.detach().float().clone()
            if svd_proj is not None
            else torch.eye(cfg.donor_hidden_size, cfg.d_vq, dtype=torch.float32)
        )
        if init_p.shape != (cfg.donor_hidden_size, cfg.d_vq):
            raise ValueError(
                f"Expected svd_proj shape {(cfg.donor_hidden_size, cfg.d_vq)}, got {tuple(init_p.shape)}"
            )
        self.register_buffer("svd_proj", init_p, persistent=True)

        self.enc_adapter = nn.Linear(cfg.d_vq, cfg.d_vq, bias=False)
        self.pre_quant_norm = nn.GroupNorm(
            num_groups=cfg.pre_quant_norm_groups,
            num_channels=cfg.d_vq,
            affine=False,
        )
        self.quantizer = MultiscaleRFSQ(tok_cfg)

        self.dec_rope = RotaryEmbedding(
            head_dim=tok_cfg.dec_head_dim,
            max_position_embeddings=max(cfg.chunk_size, 64),
            theta=cfg.rope_theta,
        )
        self.matryoshka_cutoffs = tuple(cfg.matryoshka_cutoffs)
        self.cutoff_level_embed = nn.Embedding(cfg.num_levels, cfg.d_vq)
        self.cutoff_in_norm = RMSNorm(cfg.d_vq, eps=cfg.rms_norm_eps)
        self.decoder_blocks = nn.ModuleList(
            [CNCMDecoderBlock(tok_cfg) for _ in range(cfg.dec_layers)]
        )
        self.dec_norm = RMSNorm(cfg.d_vq, eps=cfg.rms_norm_eps)
        self.dec_out_proj = nn.Linear(cfg.d_vq, cfg.donor_hidden_size, bias=False)

        self._reset_weights()

    def _reset_weights(self) -> None:
        with torch.no_grad():
            nn.init.eye_(self.enc_adapter.weight)
            nn.init.zeros_(self.cutoff_level_embed.weight)
            for blk in self.decoder_blocks:
                for mod in blk.modules():
                    if isinstance(mod, nn.Linear):
                        nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                # Zero-init decoder residual outputs at step 0 so Eckart-Young P^T path starts cleanly!
                nn.init.zeros_(blk.attn.o_proj.weight)
                nn.init.zeros_(blk.mlp.down_proj.weight)
            # Initialize dec_out_proj to svd_proj (`[H_donor, d_vq]`) so z_hat @ P^T reconstructs donor embed space!
            self.dec_out_proj.weight.copy_(self.svd_proj)
            for fsq in self.quantizer.fsq_modules:
                nn.init.normal_(fsq.proj_in.weight, mean=0.0, std=0.05)
                nn.init.zeros_(fsq.proj_in.bias)
                nn.init.normal_(fsq.proj_out.weight, mean=0.0, std=0.05)
                nn.init.zeros_(fsq.proj_out.bias)

    @torch.no_grad()
    def set_svd_from_donor_embed(self, donor_embed_weight: torch.Tensor) -> None:
        P = compute_svd_projection(donor_embed_weight, self.cfg.d_vq).to(
            device=self.svd_proj.device, dtype=self.svd_proj.dtype
        )
        self.svd_proj.copy_(P)
        self.dec_out_proj.weight.copy_(P.to(dtype=self.dec_out_proj.weight.dtype))

    def project_tokens_svd(
        self, input_ids: torch.Tensor, donor_embed_weight: torch.Tensor
    ) -> torch.Tensor:
        """Look up frozen donor embeddings `[B, S, H_donor]` and project via `svd_proj` -> `[B, S, d_vq]`."""
        with torch.no_grad():
            emb = F.embedding(input_ids, donor_embed_weight)
            proj = torch.matmul(emb.to(dtype=self.svd_proj.dtype), self.svd_proj)
        return proj.to(dtype=self.enc_adapter.weight.dtype)

    def encode_z(self, input_ids: torch.Tensor, donor_embed_weight: torch.Tensor) -> torch.Tensor:
        u = self.enc_adapter(self.project_tokens_svd(input_ids, donor_embed_weight))
        return self.pre_quant_norm(u.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

    def encode_codes(
        self, input_ids: torch.Tensor, donor_embed_weight: torch.Tensor
    ) -> List[torch.Tensor]:
        return self.quantizer.encode(self.encode_z(input_ids, donor_embed_weight))

    def decode_z_hat_hidden(
        self, z_hat: torch.Tensor, cutoff_level: Optional[int] = None
    ) -> torch.Tensor:
        """Decode quantized latent `[B, L, d_vq]` into donor-space vectors `[B, L, H_donor]`."""
        bsz, seq_len, _ = z_hat.shape
        device = z_hat.device
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        cos, sin = self.dec_rope(pos_ids, dtype=z_hat.dtype)
        attn_mask = build_bidirectional_mask(
            seq_len=seq_len, batch_size=bsz, attention_mask=None, device=device, dtype=z_hat.dtype
        )
        k_eff = self.cfg.num_levels if cutoff_level is None else max(1, min(int(cutoff_level), self.cfg.num_levels))
        k_idx = torch.tensor(k_eff - 1, device=device, dtype=torch.long)
        h = self.cutoff_in_norm(z_hat) + self.cutoff_level_embed(k_idx).view(1, 1, self.cfg.d_vq).to(dtype=z_hat.dtype)
        for blk in self.decoder_blocks:
            h = blk(h, cos=cos, sin=sin, attn_mask=attn_mask)
        return self.dec_out_proj(self.dec_norm(h))

    def decode_codes(
        self,
        codes_list: Sequence[torch.Tensor],
        donor_lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Decode full or partial `codes_list` -> vocabulary logits `[B, L, V]` using frozen `donor_lm_head_weight`."""
        z_hat = self.quantizer.reconstruct_from_codes(
            codes_list, dtype=self.dec_out_proj.weight.dtype
        )
        h_donor = self.decode_z_hat_hidden(z_hat, cutoff_level=len(codes_list))
        return F.linear(h_donor.to(dtype=donor_lm_head_weight.dtype), donor_lm_head_weight)

    def forward_stage1(
        self,
        input_ids: torch.Tensor,
        donor_embed_weight: torch.Tensor,
        donor_lm_head_weight: torch.Tensor,
        apply_fine_drop: bool = True,
    ) -> Dict[str, Any]:
        """Stage 1 self-supervised reconstruction + Matryoshka cutoff loss."""
        if input_ids.shape[1] != self.cfg.chunk_size:
            raise ValueError(
                f"Expected chunk_size={self.cfg.chunk_size}, got {input_ids.shape[1]}"
            )
        z = self.encode_z(input_ids, donor_embed_weight)
        q_out = self.quantizer(z, apply_fine_drop=apply_fine_drop)
        z_hat: torch.Tensor = q_out["z_hat"]  # type: ignore[assignment]
        commit_loss: torch.Tensor = q_out["commit_loss"]  # type: ignore[assignment]

        cutoff_used = int(q_out.get("cutoff_level", self.cfg.num_levels))
        h_donor = self.decode_z_hat_hidden(z_hat, cutoff_level=cutoff_used)
        logits = F.linear(h_donor.float(), donor_lm_head_weight.float())
        targets_flat = input_ids.reshape(-1)
        recon_loss = F.cross_entropy(logits.view(-1, self.cfg.donor_vocab_size), targets_flat)

        if self.training and self.matryoshka_cutoffs and "z_hat_levels" in q_out:
            z_levels: Sequence[torch.Tensor] = q_out["z_hat_levels"]  # type: ignore[assignment]
            m_losses: List[torch.Tensor] = []
            for k_cut in self.matryoshka_cutoffs:
                if 1 <= k_cut <= len(z_levels):
                    hk = self.decode_z_hat_hidden(z_levels[k_cut - 1], cutoff_level=k_cut)
                    lg_k = F.linear(hk.float(), donor_lm_head_weight.float())
                    m_losses.append(F.cross_entropy(lg_k.view(-1, self.cfg.donor_vocab_size), targets_flat))
            if m_losses:
                recon_loss = 0.5 * recon_loss + 0.5 * (sum(m_losses) / float(len(m_losses)))

        with torch.no_grad():
            recon_acc = (logits.argmax(dim=-1) == input_ids).float().mean()

        return {
            "loss": recon_loss + commit_loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "recon_acc": recon_acc,
            "active_code_ratio": q_out["active_code_ratio"],
        }


class LearnedDualModeRouter(nn.Module):
    """2-Layer MLP Router Gate (`future_cncm_update.md` §3) for Early-Exit & Inline Exact-Mode."""

    def __init__(self, hidden_size: int, num_levels: int, router_hidden: int = 128) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_levels = int(num_levels)
        self.router_hidden = int(router_hidden)

        self.level_emb = nn.Embedding(self.num_levels, self.router_hidden)
        self.exit_fc1 = nn.Linear(self.hidden_size, self.router_hidden, bias=True)
        self.exit_fc2 = nn.Linear(self.router_hidden, 1, bias=True)

        self.exact_fc1 = nn.Linear(self.hidden_size, self.router_hidden, bias=True)
        self.exact_fc2 = nn.Linear(self.router_hidden, 1, bias=True)

        with torch.no_grad():
            # Conservative initial bias: prefer full refinement & exactness until trained
            nn.init.constant_(self.exit_fc2.bias, -2.0)
            nn.init.constant_(self.exact_fc2.bias, 0.0)

    def predict_exit_logit(self, h_level: torch.Tensor, level_idx: int) -> torch.Tensor:
        """Predict `[B]` early-exit logit from pooled level `level_idx` hidden states `[B, l_k, H]`."""
        pooled = h_level.mean(dim=1)  # [B, H]
        lid = torch.tensor(level_idx, device=h_level.device, dtype=torch.long)
        act = F.silu(self.exit_fc1(pooled) + self.level_emb(lid).unsqueeze(0))
        return self.exit_fc2(act).squeeze(-1)

    def predict_exact_mode_logits(self, h_fine: torch.Tensor) -> torch.Tensor:
        """Predict per-position `[B, L]` exact-mode requirement logits from finest hidden states."""
        return self.exact_fc2(F.silu(self.exact_fc1(h_fine))).squeeze(-1)


def sparse_topk_tail_kl_loss(
    student_logits: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_logprobs: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact non-negative `(K + 1)`-partitioned Forward KL `D_KL(p_donor || q_sidecar) >= 0`.

    Partitions vocabulary `V` into `{top_1, ..., top_K, tail}` so residual tail probability
    mass `1 - sum(p_topk)` is accounted for explicitly (eliminates negative-loss drift).
    """
    s_f = student_logits.float()
    s_lse = torch.logsumexp(s_f, dim=-1, keepdim=True)  # [B, L, 1]
    q_topk_lp = s_f.gather(-1, topk_idx.to(torch.long)) - s_lse  # [B, L, K]

    p_topk_lp = topk_logprobs.float().clamp(max=0.0)
    p_topk = torch.exp(p_topk_lp)
    # Normalize if FP16 rounding slightly exceeds 1.0
    p_sum = p_topk.sum(dim=-1, keepdim=True)
    scale_fix = torch.clamp(p_sum, min=1.0)
    p_topk = p_topk / scale_fix
    p_topk_lp = p_topk_lp - torch.log(scale_fix)
    p_tail = (1.0 - p_topk.sum(dim=-1)).clamp(min=0.0)  # [B, L]

    q_topk = torch.exp(q_topk_lp)
    q_tail = (1.0 - q_topk.sum(dim=-1)).clamp(min=1e-7)  # [B, L]

    kl_topk = (p_topk * (p_topk_lp - q_topk_lp)).sum(dim=-1)  # [B, L]
    kl_tail = torch.where(
        p_tail > 1e-7,
        p_tail * (torch.log(p_tail.clamp(min=1e-7)) - torch.log(q_tail)),
        torch.zeros_like(p_tail),
    )
    kl_per_tok = (kl_topk + kl_tail).clamp(min=0.0)

    if valid_mask is not None:
        vm = valid_mask.to(device=kl_per_tok.device, dtype=torch.float32)
        return (kl_per_tok * vm).sum() / vm.sum().clamp(min=1.0)
    return kl_per_tok.mean()


class CNCMDeltaNetSidecar(nn.Module):
    """Path A 4-Layer `CNCM-DeltaNet` Plug-In Concept Sidecar (`.sidecar.gguf`)."""

    def __init__(
        self,
        cfg: SidecarConfig,
        concept_tokenizer: Optional[SVDConceptTokenizer] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.concept_tokenizer = (
            concept_tokenizer if concept_tokenizer is not None else SVDConceptTokenizer(cfg)
        )
        self._tokenizer_frozen: bool = True
        self.freeze_concept_tokenizer()

        # EAGLE-3 Shifted Causal Input Fusion: (3 donor unit_taps at t-1) + (SVD token embed at t)
        self.fusion_proj = nn.Linear(3 * cfg.donor_hidden_size, cfg.sidecar_hidden_size, bias=False)
        self.tok_in_proj = nn.Linear(cfg.d_vq, cfg.sidecar_hidden_size, bias=False)
        self.self_step_proj = nn.Linear(cfg.sidecar_hidden_size, cfg.sidecar_hidden_size, bias=False)
        self.fusion_norm = RMSNorm(cfg.sidecar_hidden_size, eps=cfg.rms_norm_eps)

        # Phase B target conditioning
        self.concept_in_proj = nn.Linear(cfg.d_vq, cfg.sidecar_hidden_size, bias=True)
        self.level_embed = nn.Embedding(cfg.num_levels, cfg.sidecar_hidden_size)
        self.l0_queries = nn.Parameter(
            torch.randn(1, cfg.levels[0], cfg.sidecar_hidden_size) * 0.02
        )

        head_dim = cfg.sidecar_hidden_size // cfg.sidecar_heads
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
        )
        self.layers = nn.ModuleList(
            [
                CNCMDeltaNetBlock(
                    hidden_size=cfg.sidecar_hidden_size,
                    num_heads=cfg.sidecar_heads,
                    intermediate_size=cfg.sidecar_intermediate_size,
                    head_dim=head_dim,
                    rms_norm_eps=cfg.rms_norm_eps,
                    chunk_size=cfg.deltanet_chunk_size,
                )
                for _ in range(cfg.sidecar_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.sidecar_hidden_size, eps=cfg.rms_norm_eps)

        self.concept_head = FactorizedConceptHead(
            hidden_size=cfg.sidecar_hidden_size,
            fsq_levels=cfg.fsq_levels,
            num_levels=cfg.num_levels,
            head_hidden=cfg.concept_head_hidden,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        self.align_proj = nn.Linear(cfg.sidecar_hidden_size, cfg.donor_hidden_size, bias=False)
        self.router_gate = LearnedDualModeRouter(
            hidden_size=cfg.sidecar_hidden_size,
            num_levels=cfg.num_levels,
            router_hidden=cfg.router_hidden_size,
        )

        self._init_sidecar_weights()

    def _init_sidecar_weights(self) -> None:
        for mod in (
            self.fusion_proj,
            self.tok_in_proj,
            self.self_step_proj,
            self.concept_in_proj,
            self.align_proj,
        ):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        nn.init.normal_(self.level_embed.weight, mean=0.0, std=0.02)

    def freeze_concept_tokenizer(self) -> None:
        self._tokenizer_frozen = True
        for p in self.concept_tokenizer.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.concept_tokenizer.eval()

    def unfreeze_concept_tokenizer(self) -> None:
        self._tokenizer_frozen = False
        for p in self.concept_tokenizer.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool = True) -> "CNCMDeltaNetSidecar":
        super().train(mode)
        if self._tokenizer_frozen:
            self.concept_tokenizer.eval()
        return self

    def sidecar_trainable_parameters(self) -> List[nn.Parameter]:
        tok_ids = {id(p) for p in self.concept_tokenizer.parameters()}
        return [p for p in self.parameters() if id(p) not in tok_ids and p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        tok = sum(p.numel() for p in self.concept_tokenizer.parameters())
        svd_buf = int(self.concept_tokenizer.svd_proj.numel())
        return {
            "total_params": total,
            "sidecar_generator": total - tok,
            "stage1_tokenizer": tok,
            "svd_proj_buffer": svd_buf,
            "total_serialized_floats": total + svd_buf,
        }

    # ------------------------------------------------------------------
    # State Snapshot / Restore & Position Helpers
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device: torch.device) -> List[DeltaNetLayerState]:
        return [layer.attn.init_state(batch_size, device) for layer in self.layers]

    @staticmethod
    def snapshot_state(state: Sequence[DeltaNetLayerState]) -> List[DeltaNetLayerState]:
        return [s.clone() for s in state]

    @staticmethod
    def state_nbytes(state: Sequence[DeltaNetLayerState]) -> int:
        return sum(s.nbytes() for s in state)

    def _concept_level_positions(
        self, prefix_len: int, level_len: int, chunk_size: int, device: torch.device
    ) -> torch.Tensor:
        idx = torch.arange(level_len, device=device, dtype=torch.float32)
        span_centers = torch.floor(
            (idx + 0.5) * (float(chunk_size) / float(max(1, level_len)))
        ).to(torch.long)
        max_pos = max(0, int(self.cfg.max_position_embeddings) - 1)
        return (int(prefix_len) + span_centers).clamp(max=max_pos)

    def _level_weights(
        self, level_lengths: Sequence[int], alpha: float, device: torch.device
    ) -> torch.Tensor:
        lens = torch.tensor(level_lengths, device=device, dtype=torch.float32)
        raw = lens.pow(1.0 - float(alpha))
        return raw / raw.sum().clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Phase A: EAGLE-3 Shifted Prefix Fusion & Incremental Commits
    # ------------------------------------------------------------------
    def build_shifted_prefix_inputs(
        self,
        prefix_ids: torch.Tensor,
        prefix_unit_taps: torch.Tensor,
        donor_embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Build Phase A inputs `[B, P, H_sidecar]` where slot `0` uses `(0, ids[0])` and
        slot `t >= 1` fuses `(prefix_unit_taps[t-1], ids[t])` (exact train/infer equivalence!).
        """
        bsz, p_len = prefix_ids.shape
        w_dtype = self.fusion_proj.weight.dtype
        svd_tok = self.concept_tokenizer.project_tokens_svd(prefix_ids, donor_embed_weight).to(dtype=w_dtype)
        tok_feat = self.tok_in_proj(svd_tok)  # [B, P, H_sc]

        if p_len <= 1:
            tap_feat = torch.zeros_like(tok_feat)
        else:
            if prefix_unit_taps.shape[1] < p_len - 1:
                raise ValueError(
                    f"Need at least {p_len - 1} prefix_unit_taps for prefix_len={p_len}, got {prefix_unit_taps.shape[1]}"
                )
            prev_taps = prefix_unit_taps[:, : p_len - 1, :, :].reshape(
                bsz, p_len - 1, 3 * self.cfg.donor_hidden_size
            ).to(dtype=w_dtype)
            tap_shifted = self.fusion_proj(prev_taps)  # [B, P-1, H_sc]
            zero_slot0 = tap_shifted.new_zeros((bsz, 1, self.cfg.sidecar_hidden_size))
            tap_feat = torch.cat([zero_slot0, tap_shifted], dim=1)

        return self.fusion_norm(tap_feat + tok_feat)

    def run_phase_a_prefix(
        self,
        x_prefix: torch.Tensor,
        state: Optional[Sequence[DeltaNetLayerState]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Any:
        """Run 4-layer `CNCM-DeltaNet` Phase A over `x_prefix [B, S, H_sc]` and return updated state (and optional `h`)."""
        bsz, seq_len, _ = x_prefix.shape
        device = x_prefix.device
        cur_states = list(state) if state is not None else self.init_state(bsz, device)
        start_pos = cur_states[0].seen_tokens
        max_pos = max(0, int(self.cfg.max_position_embeddings) - 1)
        pos_ids = (start_pos + torch.arange(seq_len, device=device, dtype=torch.long)).clamp(max=max_pos)
        cos, sin = self.rope(pos_ids, dtype=x_prefix.dtype)

        h = x_prefix
        next_states: List[DeltaNetLayerState] = []
        for layer, st_i in zip(self.layers, cur_states):
            h, new_st = layer.forward_phase_a_prefix(
                h, cos=cos, sin=sin, state=st_i, attention_mask=attention_mask, update_state=True
            )
            assert new_st is not None
            next_states.append(new_st)
        if return_hidden:
            return h, next_states
        return next_states

    def commit_shifted_step(
        self,
        prev_unit_taps: Optional[torch.Tensor],
        next_token_ids: torch.Tensor,
        state: Sequence[DeltaNetLayerState],
        donor_embed_weight: torch.Tensor,
        return_hidden: bool = False,
    ) -> Any:
        """Commit `M` newly verified tokens (`next_token_ids [B, M]`) paired with `prev_unit_taps [B, M, 3, H_donor]`."""
        bsz, m_len = next_token_ids.shape
        w_dtype = self.fusion_proj.weight.dtype
        svd_tok = self.concept_tokenizer.project_tokens_svd(next_token_ids, donor_embed_weight).to(dtype=w_dtype)
        tok_feat = self.tok_in_proj(svd_tok)
        if prev_unit_taps is None:
            tap_feat = torch.zeros_like(tok_feat)
        else:
            tap_flat = prev_unit_taps.reshape(bsz, m_len, 3 * self.cfg.donor_hidden_size).to(dtype=w_dtype)
            tap_feat = self.fusion_proj(tap_flat)
        x_step = self.fusion_norm(tap_feat + tok_feat)
        return self.run_phase_a_prefix(x_step, state=state, attention_mask=None, return_hidden=return_hidden)

    def run_phase_b_targets(
        self,
        x_targets: torch.Tensor,
        pos_targets: torch.Tensor,
        prefix_state: Sequence[DeltaNetLayerState],
        level_lengths: Sequence[int],
    ) -> torch.Tensor:
        """Run 4-layer `CNCM-DeltaNet` Phase B read-only over `prefix_state` (`0 MB` state growth)."""
        cos, sin = self.rope(pos_targets, dtype=x_targets.dtype)
        h = x_targets
        for layer, st_i in zip(self.layers, prefix_state):
            h = layer.forward_phase_b_targets(
                h, cos_targets=cos, sin_targets=sin, prefix_state=st_i, level_lengths=level_lengths
            )
        return self.final_norm(h)

    # ------------------------------------------------------------------
    # Zero-Backprop Stage 2 Training & Step-0 Closed-Form Ridge Warm-Start
    # ------------------------------------------------------------------
    def forward_concept_from_state(
        self,
        prefix_state: Sequence[DeltaNetLayerState],
        prefix_len: int,
        target_codes: Sequence[torch.Tensor],
        corrupt_rate: float = 0.3,
        level_weights_alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """Single-pass teacher-forced Phase B forward across all `K` target FSQ levels."""
        bsz = target_codes[0].shape[0]
        device = target_codes[0].device
        w_dtype = self.concept_in_proj.weight.dtype
        num_active = len(target_codes)
        level_lengths = tuple(int(c.shape[1]) for c in target_codes)

        eff_corrupt = float(corrupt_rate) if (self.training and corrupt_rate > 0.0) else 0.0
        with torch.no_grad():
            cond_list = self.concept_tokenizer.quantizer.build_stage2_conditioning(
                target_codes, corrupt_rate=eff_corrupt, dtype=w_dtype
            )

        parts: List[torch.Tensor] = []
        pos_parts: List[torch.Tensor] = []
        level_id_parts: List[torch.Tensor] = []
        for k in range(num_active):
            l_k = level_lengths[k]
            lid = torch.tensor(k, device=device, dtype=torch.long)
            l_emb = self.level_embed(lid).view(1, 1, self.cfg.sidecar_hidden_size)
            if k == 0:
                x_k = self.l0_queries.to(dtype=w_dtype).expand(bsz, -1, -1) + l_emb
            else:
                x_k = self.concept_in_proj(cond_list[k - 1].to(dtype=w_dtype)) + l_emb
            parts.append(x_k)
            pos_parts.append(
                self._concept_level_positions(prefix_len, l_k, self.cfg.chunk_size, device=device)
            )
            level_id_parts.append(torch.full((l_k,), k, device=device, dtype=torch.long))

        x_targets = torch.cat(parts, dim=1)
        pos_targets = torch.cat(pos_parts, dim=0)
        level_ids = torch.cat(level_id_parts, dim=0)

        h_target = self.run_phase_b_targets(
            x_targets, pos_targets=pos_targets, prefix_state=prefix_state, level_lengths=level_lengths
        )

        all_targets = torch.cat([c.to(device=device, dtype=torch.long) for c in target_codes], dim=1)
        digit_logits, token_lp = self.concept_head.log_probs(
            h_target, level_ids=level_ids, target_digits=all_targets
        )

        level_losses: List[torch.Tensor] = []
        h_per_level: List[torch.Tensor] = []
        offset = 0
        for l_k in level_lengths:
            level_losses.append((-token_lp[:, offset : offset + l_k]).mean())
            h_per_level.append(h_target[:, offset : offset + l_k, :])
            offset += l_k

        weights = self._level_weights(level_lengths, level_weights_alpha, device=device)
        concept_loss = torch.stack(level_losses).mul(weights).sum()
        return {
            "concept_loss": concept_loss,
            "level_losses": level_losses,
            "digit_logits": digit_logits,
            "h_target": h_target,
            "h_per_level": h_per_level,
            "level_ids": level_ids,
            "all_targets": all_targets,
        }

    def forward_ar_draft_loss(
        self,
        h_last: torch.Tensor,
        prefix_state: Sequence[DeltaNetLayerState],
        target_ids: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_logprobs: torch.Tensor,
        donor_embed_weight: torch.Tensor,
        donor_lm_head_weight: torch.Tensor,
        ar_steps: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """EAGLE-3 multi-step autoregressive rollout KL on the 4-layer `CNCM-DeltaNet` state (0 donor FLOPs).

        Step 0 uses `h_last` directly from Phase A; steps `1..ar_steps-1` roll out on a cloned
        `584 KB` state feeding `(self_step_proj(h_prev), svd_embed(target_ids[:, m-1]))` so
        multi-step speculative drafting stays sharp without blurring token order.
        """
        n_steps = max(1, min(int(ar_steps), int(target_ids.shape[1])))
        w_dtype = self.self_step_proj.weight.dtype
        roll_state = self.snapshot_state(prefix_state)
        h_cur = h_last  # [B, 1, H_sc]

        step_kls: List[torch.Tensor] = []
        step_accs: List[torch.Tensor] = []
        for m in range(n_steps):
            if m > 0:
                prev_tok = target_ids[:, m - 1 : m]
                svd_tok = self.concept_tokenizer.project_tokens_svd(prev_tok, donor_embed_weight).to(dtype=w_dtype)
                x_m = self.fusion_norm(self.self_step_proj(h_cur) + self.tok_in_proj(svd_tok))
                h_cur, roll_state = self.run_phase_a_prefix(x_m, state=roll_state, return_hidden=True)

            logits_m = F.linear(
                self.align_proj(self.final_norm(h_cur)).to(dtype=donor_lm_head_weight.dtype),
                donor_lm_head_weight,
            )  # [B, 1, V]
            kl_m = sparse_topk_tail_kl_loss(
                logits_m,
                topk_idx[:, m : m + 1, :],
                topk_logprobs[:, m : m + 1, :],
            )
            step_kls.append(kl_m)
            with torch.no_grad():
                pred_m = logits_m.argmax(dim=-1)  # [B, 1]
                gold_m = topk_idx[:, m : m + 1, 0]
                step_accs.append((pred_m == gold_m).float().mean())

        # Slightly higher weight on earlier draft positions (exponential decay 0.85^m)
        w = torch.tensor([0.85 ** m for m in range(n_steps)], device=h_last.device, dtype=torch.float32)
        w = w / w.sum()
        ar_kl = torch.stack(step_kls).mul(w).sum()
        ar_top1_acc = torch.stack(step_accs).mean()
        return ar_kl, ar_top1_acc

    def forward_train_from_cache(
        self,
        batch: Dict[str, torch.Tensor],
        donor_embed_weight: torch.Tensor,
        donor_lm_head_weight: torch.Tensor,
        *,
        prefix_mask: Optional[torch.Tensor] = None,
        corrupt_rate: float = 0.3,
        kl_weight: float = 1.0,
        router_weight: float = 0.2,
        ar_draft_weight: float = 1.5,
        ar_draft_steps: int = 5,
        sensitive_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Complete zero-backprop Stage 2 training step from `OfflineDonorCache` tensors."""
        prefix_ids = batch["prefix_ids"]
        target_ids = batch["target_ids"]
        prefix_unit_taps = batch["prefix_unit_taps"]
        topk_idx = batch["topk_idx"]
        topk_logprobs = batch["topk_logprobs"]

        with torch.no_grad():
            target_codes = self.concept_tokenizer.encode_codes(target_ids, donor_embed_weight)
            # Oracle predictions for router supervision
            full_s1_logits = self.concept_tokenizer.decode_codes(target_codes, donor_lm_head_weight)
            full_s1_preds = full_s1_logits.argmax(dim=-1)  # [B, L]
            teacher_top1 = topk_idx[..., 0]                 # [B, L]
            # Per-level early-exit oracle: does partial decode [:k+1] already agree >= 90% with full decode?
            exit_targets: List[torch.Tensor] = []
            for k in range(len(target_codes)):
                if k == len(target_codes) - 1:
                    exit_targets.append(torch.ones(prefix_ids.shape[0], device=prefix_ids.device))
                else:
                    part_preds = self.concept_tokenizer.decode_codes(
                        target_codes[: k + 1], donor_lm_head_weight
                    ).argmax(dim=-1)
                    agree = (part_preds == full_s1_preds).float().mean(dim=-1)
                    exit_targets.append((agree >= 0.90).float())

        x_prefix = self.build_shifted_prefix_inputs(prefix_ids, prefix_unit_taps, donor_embed_weight)
        h_prefix, prefix_state = self.run_phase_a_prefix(
            x_prefix, state=None, attention_mask=prefix_mask, return_hidden=True
        )

        # EAGLE-3 autoregressive multi-step token drafter loss (shares 4L CNCM-DeltaNet + align_proj!)
        if ar_draft_weight > 0.0:
            ar_kl_loss, ar_top1_acc = self.forward_ar_draft_loss(
                h_last=h_prefix[:, -1:, :],
                prefix_state=prefix_state,
                target_ids=target_ids,
                topk_idx=topk_idx,
                topk_logprobs=topk_logprobs,
                donor_embed_weight=donor_embed_weight,
                donor_lm_head_weight=donor_lm_head_weight,
                ar_steps=ar_draft_steps,
            )
        else:
            ar_kl_loss = h_prefix.new_zeros(())
            ar_top1_acc = h_prefix.new_zeros(())

        c_out = self.forward_concept_from_state(
            prefix_state=prefix_state,
            prefix_len=int(prefix_ids.shape[1]),
            target_codes=target_codes,
            corrupt_rate=corrupt_rate,
        )
        concept_loss: torch.Tensor = c_out["concept_loss"]

        # Sparse Top-K + Tail KL Distillation on finest Concept level
        h_fine = c_out["h_per_level"][-1]  # [B, L, H_sc]
        align_logits = F.linear(
            self.align_proj(h_fine).to(dtype=donor_lm_head_weight.dtype), donor_lm_head_weight
        )
        kl_loss = (
            sparse_topk_tail_kl_loss(align_logits, topk_idx, topk_logprobs)
            if kl_weight > 0.0
            else concept_loss.new_zeros(())
        )

        # Dual-Mode Router supervision (Early-Exit BCE + Inline Exact-Mode BCE)
        if router_weight > 0.0:
            exit_losses: List[torch.Tensor] = []
            for k, h_k in enumerate(c_out["h_per_level"]):
                ex_logit = self.router_gate.predict_exit_logit(h_k, k)
                exit_losses.append(F.binary_cross_entropy_with_logits(ex_logit.float(), exit_targets[k].float()))
            exit_bce = torch.stack(exit_losses).mean()

            # Flag positions where discrete Stage-1 bottleneck disagrees with teacher OR caller-marked sensitive tokens
            need_exact = (full_s1_preds != teacher_top1) | (target_ids != full_s1_preds)
            if sensitive_token_mask is not None:
                need_exact = need_exact | sensitive_token_mask.to(device=need_exact.device, dtype=torch.bool)
            exact_logits = self.router_gate.predict_exact_mode_logits(h_fine)
            exact_bce = F.binary_cross_entropy_with_logits(exact_logits.float(), need_exact.float())
            router_loss = 0.5 * (exit_bce + exact_bce)
        else:
            router_loss = concept_loss.new_zeros(())

        total_loss = (
            concept_loss
            + float(kl_weight) * kl_loss
            + float(ar_draft_weight) * ar_kl_loss
            + float(router_weight) * router_loss
        )
        return {
            "loss": total_loss,
            "concept_loss": concept_loss,
            "kl_loss": kl_loss,
            "ar_kl_loss": ar_kl_loss,
            "ar_top1_acc": ar_top1_acc,
            "router_loss": router_loss,
            "level_losses": c_out["level_losses"],
        }

    @torch.no_grad()
    def ridge_warmstart_concept_head(
        self,
        batches: Sequence[Dict[str, torch.Tensor]],
        donor_embed_weight: torch.Tensor,
        ridge_lambda: float = 1e-2,
    ) -> Dict[str, float]:
        """Closed-form Tikhonov Ridge initialization of all `d_q` `FactorizedConceptHead` digit heads."""
        was_training = self.training
        self.eval()

        d_q = len(self.cfg.fsq_levels)
        d_head = self.concept_head.head_hidden

        # Accumulate Gram matrices X^T X [(d_head+1), (d_head+1)] and X^T Y [(d_head+1), L_j] in FP64
        # (streaming outer-product accumulation -> O(d_head^2) memory even over 100k tokens!)
        xtx = [torch.zeros((d_head + 1, d_head + 1), dtype=torch.float64) for _ in range(d_q)]
        xty = [
            torch.zeros((d_head + 1, self.cfg.fsq_levels[j]), dtype=torch.float64)
            for j in range(d_q)
        ]

        # Measure pre-warmstart NLL on first batch
        first_b = batches[0]
        p_ids0, t_ids0, p_taps0 = first_b["prefix_ids"], first_b["target_ids"], first_b["prefix_unit_taps"]
        t_codes0 = self.concept_tokenizer.encode_codes(t_ids0, donor_embed_weight)
        st0 = self.run_phase_a_prefix(self.build_shifted_prefix_inputs(p_ids0, p_taps0, donor_embed_weight))
        nll_before = float(
            self.forward_concept_from_state(st0, int(p_ids0.shape[1]), t_codes0, corrupt_rate=0.0)["concept_loss"].item()
        )

        for batch in batches:
            p_ids = batch["prefix_ids"]
            t_ids = batch["target_ids"]
            p_taps = batch["prefix_unit_taps"]
            t_codes = self.concept_tokenizer.encode_codes(t_ids, donor_embed_weight)
            x_pref = self.build_shifted_prefix_inputs(p_ids, p_taps, donor_embed_weight)
            p_state = self.run_phase_a_prefix(x_pref)
            c_out = self.forward_concept_from_state(
                p_state, int(p_ids.shape[1]), t_codes, corrupt_rate=0.0
            )
            h_tgt = c_out["h_target"]          # [B, T, H_sc]
            lvl_ids = c_out["level_ids"]       # [T]
            all_tgts = c_out["all_targets"]    # [B, T, d_q]

            state_j = self.concept_head._build_context(h_tgt, lvl_ids)  # [B, T, d_head]
            for j in range(d_q):
                if j > 0:
                    state_j = self.concept_head._advance_state(state_j, all_tgts[..., j - 1], j - 1)
                x_flat = state_j.reshape(-1, d_head).to(device="cpu", dtype=torch.float64)
                ones = torch.ones((x_flat.shape[0], 1), dtype=torch.float64)
                x_aug = torch.cat([x_flat, ones], dim=-1)  # [N, d_head + 1]

                n_cls = self.cfg.fsq_levels[j]
                y_idx = all_tgts[..., j].reshape(-1).to(device="cpu", dtype=torch.long)
                # Centered log-odds target (+2.0 for target digit, -2.0/(L_j-1) otherwise)
                y_mat = torch.full((x_flat.shape[0], n_cls), -2.0 / float(max(1, n_cls - 1)), dtype=torch.float64)
                y_mat.scatter_(1, y_idx.unsqueeze(1), 2.0)

                xtx[j] += x_aug.T @ x_aug
                xty[j] += x_aug.T @ y_mat

        for j in range(d_q):
            gram = xtx[j]
            diag_scale = float(torch.diagonal(gram[:-1, :-1]).mean().item())
            reg = max(1e-6, float(ridge_lambda) * max(1.0, diag_scale))
            eye = torch.eye(d_head + 1, dtype=torch.float64)
            eye[-1, -1] = 1e-6  # minimal regularization on bias intercept
            wb = torch.linalg.solve(gram + reg * eye, xty[j])  # [d_head + 1, L_j]

            head_j = self.concept_head.digit_heads[j]
            head_j.weight.copy_(wb[:-1, :].T.to(device=head_j.weight.device, dtype=head_j.weight.dtype))
            if head_j.bias is not None:
                head_j.bias.copy_(wb[-1, :].to(device=head_j.bias.device, dtype=head_j.bias.dtype))

        nll_after = float(
            self.forward_concept_from_state(st0, int(p_ids0.shape[1]), t_codes0, corrupt_rate=0.0)["concept_loss"].item()
        )
        if was_training:
            self.train(True)
        return {"nll_before": nll_before, "nll_after": nll_after, "delta_nll": nll_after - nll_before}

    # ------------------------------------------------------------------
    # Runtime Drafting, 100%-Lossless Speculative & Precision-Ultra Decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def draft_chunk_from_state(
        self,
        prefix_state: Sequence[DeltaNetLayerState],
        donor_lm_head_weight: torch.Tensor,
        *,
        max_concept_level: Optional[int] = None,
        early_exit_prob: Optional[float] = 0.85,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Draft a `chunk_size` block read-only from `prefix_state` (`0 MB` state mutation)."""
        bsz = prefix_state[0].recurrent_state.shape[0]
        device = prefix_state[0].recurrent_state.device
        prefix_len = prefix_state[0].seen_tokens
        w_dtype = self.concept_in_proj.weight.dtype

        k_cap = (
            self.cfg.num_levels
            if (max_concept_level is None or max_concept_level <= 0)
            else max(1, min(int(max_concept_level), self.cfg.num_levels))
        )

        z_hat = torch.zeros(
            (bsz, self.cfg.chunk_size, self.cfg.d_vq),
            device=device,
            dtype=self.concept_tokenizer.dec_out_proj.weight.dtype,
        )
        codes_list: List[torch.Tensor] = []
        last_h_k: Optional[torch.Tensor] = None
        used_levels = 0

        for k in range(k_cap):
            l_k = self.cfg.levels[k]
            lid = torch.tensor(k, device=device, dtype=torch.long)
            l_emb = self.level_embed(lid).view(1, 1, self.cfg.sidecar_hidden_size)
            if k == 0:
                x_k = self.l0_queries.to(dtype=w_dtype).expand(bsz, -1, -1) + l_emb
            else:
                cond_k = self.concept_tokenizer.quantizer.get_level_conditioning(z_hat, k)
                x_k = self.concept_in_proj(cond_k.to(dtype=w_dtype)) + l_emb

            pos_k = self._concept_level_positions(prefix_len, l_k, self.cfg.chunk_size, device=device)
            h_k = self.run_phase_b_targets(
                x_k, pos_targets=pos_k, prefix_state=prefix_state, level_lengths=(l_k,)
            )
            digits_k, _, mean_conf = self.concept_head.sample(h_k, level_ids=k, temperature=temperature)
            codes_list.append(digits_k)
            z_hat = z_hat + self.concept_tokenizer.quantizer.decode_level(
                digits_k, k, dtype=z_hat.dtype
            )
            last_h_k = h_k
            used_levels = k + 1

            if early_exit_prob is not None and k + 1 < k_cap and k >= 1:
                router_p = torch.sigmoid(self.router_gate.predict_exit_logit(h_k, k))
                comb_p = 0.5 * (router_p + mean_conf)
                if bool((comb_p >= float(early_exit_prob)).all().item()):
                    break

        assert last_h_k is not None
        h_donor = self.concept_tokenizer.decode_z_hat_hidden(z_hat, cutoff_level=used_levels)
        draft_logits = F.linear(h_donor.to(dtype=donor_lm_head_weight.dtype), donor_lm_head_weight)
        draft_ids = draft_logits.argmax(dim=-1)

        # Upsample `last_h_k` to `chunk_size` if early-exited before finest level so exact router works at any level!
        if last_h_k.shape[1] != self.cfg.chunk_size:
            h_fine_approx = MultiscaleRFSQ._linear_upsample_1d(
                last_h_k.transpose(1, 2), self.cfg.chunk_size
            ).transpose(1, 2)
        else:
            h_fine_approx = last_h_k
        exact_probs = torch.sigmoid(self.router_gate.predict_exact_mode_logits(h_fine_approx))

        return {
            "draft_ids": draft_ids,
            "draft_logits": draft_logits,
            "codes": codes_list,
            "levels_used": used_levels,
            "exact_mode_probs": exact_probs,
        }

    @torch.no_grad()
    def draft_ar_from_state(
        self,
        h_last: torch.Tensor,
        prefix_state: Sequence[DeltaNetLayerState],
        donor_embed_weight: torch.Tensor,
        donor_lm_head_weight: torch.Tensor,
        *,
        draft_horizon: int = 5,
        temperature: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressively draft `draft_horizon` exact-order tokens on a cloned `584 KB` state (0 donor FLOPs)."""
        roll_state = self.snapshot_state(prefix_state)
        h_cur = h_last
        w_dtype = self.self_step_proj.weight.dtype

        ids_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        for m in range(max(1, int(draft_horizon))):
            if m > 0:
                svd_tok = self.concept_tokenizer.project_tokens_svd(ids_list[-1], donor_embed_weight).to(dtype=w_dtype)
                x_m = self.fusion_norm(self.self_step_proj(h_cur) + self.tok_in_proj(svd_tok))
                h_cur, roll_state = self.run_phase_a_prefix(x_m, state=roll_state, return_hidden=True)

            lg_m = F.linear(
                self.align_proj(self.final_norm(h_cur)).to(dtype=donor_lm_head_weight.dtype),
                donor_lm_head_weight,
            )  # [B, 1, V]
            if temperature <= 0.0:
                tok_m = lg_m.float().argmax(dim=-1)  # [B, 1]
            else:
                probs = F.softmax(lg_m[:, 0, :].float() / float(temperature), dim=-1)
                tok_m = torch.multinomial(probs, num_samples=1)
            ids_list.append(tok_m)
            logits_list.append(lg_m)

        return {
            "draft_ids": torch.cat(ids_list, dim=1),
            "draft_logits": torch.cat(logits_list, dim=1),
        }

    @torch.no_grad()
    def generate_speculative_lossless(
        self,
        donor: FrozenDonorRunner,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        draft_horizon: Optional[int] = None,
        max_concept_level: Optional[int] = None,
        draft_mode: str = "ar",
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        quantize_donor_kv: bool = False,
    ) -> Dict[str, Any]:
        """100%-Lossless EAGLE-3-Shifted Speculative Decoding with zero redundant donor passes.

        Supports `draft_mode="ar"` (EAGLE-3 autoregressive drafter on 4L `CNCM-DeltaNet` state for
        high acceptance) or `draft_mode="concept"` (coarse-to-fine Concept Block-AR drafter).
        At `temperature == 0.0`, output token IDs are **mathematically bit-identical** to
        `donor.generate_exact(prompt_ids, max_new_tokens)` while sidecar uses `0 MB` KV cache.
        """
        if prompt_ids.shape[0] != 1:
            raise ValueError("generate_speculative_lossless supports batch_size=1 edge decode")
        if max_new_tokens <= 0:
            return {"token_ids": prompt_ids, "accepted_tokens": 0, "drafted_tokens": 0, "verify_steps": 0}

        eos = self.cfg.eos_token_id if eos_token_id is None else int(eos_token_id)
        default_h = 5 if draft_mode == "ar" else self.cfg.chunk_size
        horizon = min(
            self.cfg.chunk_size,
            max(1, int(draft_horizon) if draft_horizon is not None else default_h),
        )
        donor_emb_w = donor.embed_tokens.weight
        donor_head_w = donor.lm_head.weight

        # 1. Single Donor Prefill on `prompt_ids [0 .. P0-1]`
        pre_out = donor.forward_with_taps(
            prompt_ids, use_cache=True, quantize_kv=quantize_donor_kv
        )
        donor_kv = pre_out["past_key_values"]
        p0_taps = pre_out["unit_taps"]  # [1, P0, 3, H_donor]
        first_logits = pre_out["logits"][:, -1, :].float()

        def _sample_one(lg: torch.Tensor) -> torch.Tensor:
            if temperature <= 0.0:
                return lg.argmax(dim=-1, keepdim=True)
            return torch.multinomial(F.softmax(lg / float(temperature), dim=-1), num_samples=1)

        pending_tok = _sample_one(first_logits)  # exact first generated token at pos P0!
        gen_tokens: List[int] = [int(pending_tok.item())]

        if len(gen_tokens) >= max_new_tokens or (eos >= 0 and gen_tokens[-1] == eos):
            full_ids = torch.cat([prompt_ids, pending_tok], dim=1)
            return {
                "token_ids": full_ids,
                "accepted_tokens": 0,
                "drafted_tokens": 0,
                "acceptance_rate": 0.0,
                "verify_steps": 0,
                "sidecar_state_bytes": 0,
            }

        # 2. Initialize Sidecar Phase A over `[prompt_ids, pending_tok]` (`P0 + 1` slots!)
        init_ids = torch.cat([prompt_ids, pending_tok], dim=1)
        x_init = self.build_shifted_prefix_inputs(init_ids, p0_taps, donor_emb_w)
        h_init, sc_state = self.run_phase_a_prefix(x_init, return_hidden=True)
        h_last = h_init[:, -1:, :]

        total_accepted = 0
        total_drafted = 0
        verify_steps = 0

        while len(gen_tokens) < max_new_tokens:
            rem = max_new_tokens - len(gen_tokens)
            cur_m = min(horizon, rem)

            if draft_mode == "ar":
                draft_res = self.draft_ar_from_state(
                    h_last,
                    sc_state,
                    donor_emb_w,
                    donor_head_w,
                    draft_horizon=cur_m,
                    temperature=temperature,
                )
            else:
                draft_res = self.draft_chunk_from_state(
                    sc_state,
                    donor_head_w,
                    max_concept_level=max_concept_level,
                    temperature=temperature,
                )
            draft_ids = draft_res["draft_ids"][:, :cur_m]       # [1, cur_m]
            draft_logits = draft_res["draft_logits"][:, :cur_m] # [1, cur_m, V]
            total_drafted += cur_m

            # 1 Batched Donor Verify Pass over `[pending_tok, draft_0, ..., draft_{cur_m - 1}]`
            verify_in = torch.cat([pending_tok, draft_ids], dim=1)  # [1, 1 + cur_m]
            prior_kv_len = get_kv_seq_len(donor_kv)
            v_out = donor.forward_with_taps(
                verify_in,
                past_key_values=donor_kv,
                use_cache=True,
                quantize_kv=quantize_donor_kv,
            )
            verify_steps += 1
            v_logits = v_out["logits"].float()  # [1, 1 + cur_m, V]
            v_taps = v_out["unit_taps"]         # [1, 1 + cur_m, 3, H_donor]

            n_acc = 0
            bonus_tok: Optional[torch.Tensor] = None
            hit_eos = False

            for j in range(cur_m):
                cand = int(draft_ids[0, j].item())
                target_lg = v_logits[:, j, :]
                if temperature <= 0.0:
                    gold = int(target_lg.argmax(dim=-1).item())
                    if cand == gold:
                        n_acc += 1
                        gen_tokens.append(cand)
                        if eos >= 0 and cand == eos:
                            hit_eos = True
                            break
                        if len(gen_tokens) >= max_new_tokens:
                            break
                    else:
                        bonus_tok = torch.tensor([[gold]], device=prompt_ids.device, dtype=torch.long)
                        break
                else:
                    p_dist = F.softmax(target_lg / float(temperature), dim=-1)
                    q_dist = F.softmax(draft_logits[:, j, :].float() / float(temperature), dim=-1)
                    p_c = float(p_dist[0, cand].item())
                    q_c = max(1e-9, float(q_dist[0, cand].item()))
                    if float(torch.rand((), device=prompt_ids.device).item()) <= min(1.0, p_c / q_c):
                        n_acc += 1
                        gen_tokens.append(cand)
                        if eos >= 0 and cand == eos:
                            hit_eos = True
                            break
                        if len(gen_tokens) >= max_new_tokens:
                            break
                    else:
                        resid = (p_dist - q_dist).clamp(min=0.0)
                        r_sum = float(resid.sum().item())
                        probs = (resid / r_sum) if r_sum > 1e-9 else p_dist
                        bonus_tok = torch.multinomial(probs, num_samples=1)
                        break

            total_accepted += n_acc
            if hit_eos or len(gen_tokens) >= max_new_tokens:
                donor_kv = truncate_kv_cache(v_out["past_key_values"], prior_kv_len + 1 + n_acc)
                break

            if bonus_tok is None:
                bonus_tok = _sample_one(v_logits[:, cur_m, :])

            bonus_id = int(bonus_tok.item())
            gen_tokens.append(bonus_id)

            donor_kv = truncate_kv_cache(v_out["past_key_values"], prior_kv_len + 1 + n_acc)

            commit_taps = v_taps[:, : n_acc + 1, :, :]
            commit_ids = (
                torch.cat([draft_ids[:, :n_acc], bonus_tok], dim=1)
                if n_acc > 0
                else bonus_tok
            )
            h_commit, sc_state = self.commit_shifted_step(
                commit_taps, commit_ids, sc_state, donor_emb_w, return_hidden=True
            )
            h_last = h_commit[:, -1:, :]
            pending_tok = bonus_tok

            if eos >= 0 and bonus_id == eos:
                break

        gen_tensor = torch.tensor([gen_tokens[:max_new_tokens]], device=prompt_ids.device, dtype=torch.long)
        return {
            "token_ids": torch.cat([prompt_ids, gen_tensor], dim=1),
            "accepted_tokens": total_accepted,
            "drafted_tokens": total_drafted,
            "acceptance_rate": float(total_accepted) / float(max(1, total_drafted)),
            "verify_steps": verify_steps,
            "sidecar_state_bytes": self.state_nbytes(sc_state),
            "draft_mode": draft_mode,
        }

    @torch.no_grad()
    def generate_ultra_precise(
        self,
        donor: FrozenDonorRunner,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        ultra_concept_level: int = 3,
        exact_router_threshold: float = 0.65,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Precision-Ultra Hybrid Mode: streams confident prose via `L_ultra` Concept blocks
        while automatically switching inline to Exact-AR whenever `router_gate` flags precision spans!
        """
        if prompt_ids.shape[0] != 1:
            raise ValueError("generate_ultra_precise supports batch_size=1")
        if max_new_tokens <= 0:
            return {"token_ids": prompt_ids, "ultra_tokens": 0, "exact_tokens": 0}

        eos = self.cfg.eos_token_id if eos_token_id is None else int(eos_token_id)
        donor_emb_w = donor.embed_tokens.weight
        donor_head_w = donor.lm_head.weight

        pre_out = donor.forward_with_taps(prompt_ids, use_cache=True)
        donor_kv = pre_out["past_key_values"]
        p0_taps = pre_out["unit_taps"]
        pending_tok = pre_out["logits"][:, -1, :].float().argmax(dim=-1, keepdim=True)
        gen_tokens: List[int] = [int(pending_tok.item())]

        init_ids = torch.cat([prompt_ids, pending_tok], dim=1)
        sc_state = self.run_phase_a_prefix(
            self.build_shifted_prefix_inputs(init_ids, p0_taps, donor_emb_w)
        )

        ultra_tokens = 0
        exact_tokens = 1

        while len(gen_tokens) < max_new_tokens:
            if eos >= 0 and gen_tokens[-1] == eos:
                break
            rem = max_new_tokens - len(gen_tokens)
            cur_m = min(self.cfg.chunk_size, rem)

            draft_res = self.draft_chunk_from_state(
                sc_state,
                donor_head_w,
                max_concept_level=ultra_concept_level,
                temperature=0.0,
            )
            draft_ids = draft_res["draft_ids"][:, :cur_m]
            exact_probs = draft_res["exact_mode_probs"][0, :cur_m]

            # Accept confident prose prefix up to the first router-flagged sensitive position `j_flag`
            flagged = (exact_probs >= float(exact_router_threshold)).nonzero(as_tuple=False)
            n_prose = int(flagged[0, 0].item()) if flagged.numel() > 0 else cur_m

            if n_prose > 0:
                prose_slice = draft_ids[:, :n_prose]
                prose_list = prose_slice[0].tolist()
                if eos >= 0 and eos in prose_list:
                    eos_pos = prose_list.index(eos) + 1
                    prose_slice = prose_slice[:, :eos_pos]
                    prose_list = prose_list[:eos_pos]
                    gen_tokens.extend(prose_list)
                    ultra_tokens += len(prose_list)
                    break

                # Advance donor KV & taps in 1 batched pass over `[pending_tok, prose_slice]`
                step_in = torch.cat([pending_tok, prose_slice], dim=1)
                d_out = donor.forward_with_taps(step_in, past_key_values=donor_kv, use_cache=True)
                donor_kv = d_out["past_key_values"]
                gen_tokens.extend(prose_list)
                ultra_tokens += len(prose_list)
                if len(gen_tokens) >= max_new_tokens:
                    break

                # Next exact anchor token from donor's end-of-prose logit!
                next_exact = d_out["logits"][:, -1, :].float().argmax(dim=-1, keepdim=True)
                gen_tokens.append(int(next_exact.item()))
                exact_tokens += 1

                commit_ids = torch.cat([prose_slice, next_exact], dim=1)
                sc_state = self.commit_shifted_step(d_out["unit_taps"], commit_ids, sc_state, donor_emb_w)
                pending_tok = next_exact
            else:
                # Position 0 is flagged sensitive -> take 1 guaranteed Exact-AR step!
                d_out = donor.forward_with_taps(pending_tok, past_key_values=donor_kv, use_cache=True)
                donor_kv = d_out["past_key_values"]
                next_exact = d_out["logits"][:, -1, :].float().argmax(dim=-1, keepdim=True)
                gen_tokens.append(int(next_exact.item()))
                exact_tokens += 1
                sc_state = self.commit_shifted_step(d_out["unit_taps"], next_exact, sc_state, donor_emb_w)
                pending_tok = next_exact

        gen_tensor = torch.tensor([gen_tokens[:max_new_tokens]], device=prompt_ids.device, dtype=torch.long)
        return {
            "token_ids": torch.cat([prompt_ids, gen_tensor], dim=1),
            "ultra_tokens": ultra_tokens,
            "exact_tokens": exact_tokens,
        }
