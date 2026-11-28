"""Stage 1 CNCM Concept Tokenizer: Minimal Encoder + MultiscaleRFSQ + Bidirectional Decoder.

Architecture highlights vs. vanilla NCM VQ-VAE:
  - Zero-transformer embedding encoder + affine-free GroupNorm (prevents unbounded latent scale)
  - MultiscaleRFSQ: zero codebook tables, zero EMA buffers, ~100% grid utilization
  - Lightweight 1-block bidirectional Transformer decoder (Pre-RMSNorm + QK-RMSNorm + RoPE + SwiGLU)
  - Supports partial-depth decoding (`decode_codes(codes[:k])`) for Stage 2 Dynamic Early Exit
  - Phase B Copy-Aware Stage 1 Decoder (`CopyAwareKeywordCrossAttention` behind `cfg.use_copy_decoder`):
    cross-attends decoder hidden states over planned/oracle keyword token IDs with a zero-init gate
    and a pointer logit boost, trained with automatic oracle keywords and 30% keyword dropout.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cncm.config import CNCMConfig
from cncm.layers import (
    CNCMDecoderBlock,
    RMSNorm,
    RotaryEmbedding,
    build_bidirectional_mask,
    build_causal_mask,
)
from cncm.quantizer import MultiscaleRFSQ


class CopyAwareKeywordCrossAttention(nn.Module):
    """Phase B Copy-Aware Cross-Attention + Pointer Logit Boost over keyword token IDs."""

    def __init__(self, cfg: CNCMConfig) -> None:
        super().__init__()
        self.d_vq = int(cfg.d_vq)
        req_heads = max(1, int(getattr(cfg, "copy_decoder_heads", 2)))
        if self.d_vq % req_heads == 0:
            self.num_heads = req_heads
        else:
            self.num_heads = math.gcd(self.d_vq, req_heads) or 1
        self.head_dim = self.d_vq // self.num_heads

        self.q_norm = RMSNorm(self.d_vq, eps=cfg.rms_norm_eps)
        self.kv_norm = RMSNorm(self.d_vq, eps=cfg.rms_norm_eps)
        self.q_proj = nn.Linear(self.d_vq, self.d_vq, bias=False)
        self.k_proj = nn.Linear(self.d_vq, self.d_vq, bias=False)
        self.v_proj = nn.Linear(self.d_vq, self.d_vq, bias=False)
        self.o_proj = nn.Linear(self.d_vq, self.d_vq, bias=False)

        # Zero-initialized residual gate so init behavior matches non-copy decoder
        self.copy_gate = nn.Parameter(torch.zeros(1))
        # Softplus-gated pointer logit boost scattered onto kw_token_ids
        self.pointer_logit_scale = nn.Parameter(torch.tensor(1.0))
        # Per-position learned copy gate g_l = sigmoid(W_g h_l) so pointer logits fire
        # selectively at identifier positions rather than leaking onto whitespace/punctuation.
        self.pos_gate_proj = nn.Linear(self.d_vq, 1, bias=True)

    def forward(
        self,
        h: torch.Tensor,
        wte: nn.Embedding,
        kw_token_ids: torch.Tensor,
        kw_mask: Optional[torch.Tensor] = None,
        vocab_size: Optional[int] = None,
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross-attend decoder state h [B, L, d_vq] over kw_token_ids [B, K_kw]."""
        bsz, seq_len, _ = h.shape
        device = h.device

        kw_ids = kw_token_ids.to(device=device, dtype=torch.long)
        if kw_ids.ndim == 1:
            kw_ids = kw_ids.unsqueeze(0).expand(bsz, -1)
        if kw_ids.numel() == 0 or kw_ids.shape[1] == 0:
            return h, None

        k_len = int(kw_ids.shape[1])
        if kw_mask is not None:
            valid_kw = kw_mask.to(device=device, dtype=torch.bool)
            if valid_kw.ndim == 1:
                valid_kw = valid_kw.unsqueeze(0).expand(bsz, -1)
        else:
            valid_kw = (kw_ids != int(pad_token_id)) & (kw_ids >= 0)

        if not bool(valid_kw.any().item()):
            return h, None

        row_has_kw = valid_kw.any(dim=-1)  # [B]
        kw_ids_safe = kw_ids.clamp(0, wte.num_embeddings - 1)
        kw_emb = self.kv_norm(wte(kw_ids_safe).to(dtype=h.dtype))

        q = (
            self.q_proj(self.q_norm(h))
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(kw_emb)
            .view(bsz, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(kw_emb)
            .view(bsz, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale_factor = 1.0 / math.sqrt(float(self.head_dim))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor  # [B, H, L, K_kw]

        # NaN-safe masking for rows with zero active keywords
        safe_valid = valid_kw.clone()
        safe_valid[~row_has_kw, 0] = True
        neg_val = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(~safe_valid[:, None, None, :], neg_val)
        attn_probs = F.softmax(scores.float(), dim=-1).to(h.dtype)
        attn_probs = attn_probs * valid_kw[:, None, None, :].to(h.dtype)

        ctx = (
            torch.matmul(attn_probs, v)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.d_vq)
        )
        row_gate = row_has_kw[:, None, None].to(h.dtype)
        h_out = h + torch.tanh(self.copy_gate) * self.o_proj(ctx) * row_gate

        # Scatter per-position gated pointer logit boost onto keyword token IDs
        v_size = int(vocab_size) if vocab_size is not None else int(wte.num_embeddings)
        pointer_logits = h.new_zeros((bsz, seq_len, v_size))
        ptr_weights = attn_probs.mean(dim=1)  # [B, L, K_kw]
        pos_gate = torch.sigmoid(self.pos_gate_proj(h))  # [B, L, 1] in (0, 1)
        ptr_scale = F.softplus(self.pointer_logit_scale).to(h.dtype)
        src = ptr_weights * pos_gate * ptr_scale * valid_kw[:, None, :].to(h.dtype)
        idx = kw_ids_safe.clamp(0, v_size - 1).unsqueeze(1).expand(bsz, seq_len, k_len)
        pointer_logits.scatter_add_(dim=-1, index=idx, src=src)

        return h_out, pointer_logits


class CNCMConceptTokenizer(nn.Module):
    """Stage 1 Concept Tokenizer converting length-L token chunks to/from K-level FSQ codes."""

    def __init__(self, cfg: CNCMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Minimal embedding encoder + optional lightweight semantic encoder blocks
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_vq)
        self.enc_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        enc_layers = max(0, int(getattr(cfg, "enc_layers", 0)))
        self.encoder_blocks: Optional[nn.ModuleList] = (
            nn.ModuleList([CNCMDecoderBlock(cfg) for _ in range(enc_layers)])
            if enc_layers > 0
            else None
        )
        # CRITICAL: affine=False prevents unbounded latent norm inflation
        self.pre_quant_norm = nn.GroupNorm(
            num_groups=cfg.pre_quant_norm_groups,
            num_channels=cfg.d_vq,
            affine=False,
        )

        # Zero-codebook-table Multiscale Residual Finite Scalar Quantizer
        self.quantizer = MultiscaleRFSQ(cfg)

        # Lightweight bidirectional Transformer decoder
        self.dec_rope = RotaryEmbedding(
            head_dim=cfg.dec_head_dim,
            max_position_embeddings=max(cfg.chunk_size, 64),
            theta=cfg.rope_theta,
        )
        # Optional Matryoshka cutoff-level conditioning (created only when matryoshka_cutoffs is non-empty)
        self.matryoshka_cutoffs: Tuple[int, ...] = tuple(
            sorted({c for c in getattr(cfg, "matryoshka_cutoffs", ()) if 1 <= int(c) <= cfg.num_levels})
        )
        self.cutoff_level_embed: Optional[nn.Embedding] = (
            nn.Embedding(cfg.num_levels, cfg.d_vq) if self.matryoshka_cutoffs else None
        )
        self.cutoff_in_norm: Optional[RMSNorm] = (
            RMSNorm(cfg.d_vq, eps=cfg.rms_norm_eps) if self.matryoshka_cutoffs else None
        )
        self.decoder_blocks = nn.ModuleList(
            [CNCMDecoderBlock(cfg) for _ in range(cfg.dec_layers)]
        )
        # Optional Phase B Copy-Aware Keyword Cross-Attention
        self.copy_cross_attn: Optional[CopyAwareKeywordCrossAttention] = (
            CopyAwareKeywordCrossAttention(cfg)
            if getattr(cfg, "use_copy_decoder", False)
            else None
        )
        self.dec_norm = RMSNorm(cfg.d_vq, eps=cfg.rms_norm_eps)
        self.output_head = nn.Linear(cfg.d_vq, cfg.vocab_size, bias=False)

        self._init_weights()
        if cfg.tie_tokenizer_embeddings:
            self.output_head.weight = self.wte.weight

    def _init_weights(self) -> None:
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        for block in self.decoder_blocks:
            for mod in block.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        if self.copy_cross_attn is not None:
            for mod in (
                self.copy_cross_attn.q_proj,
                self.copy_cross_attn.k_proj,
                self.copy_cross_attn.v_proj,
                self.copy_cross_attn.o_proj,
            ):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.copy_cross_attn.copy_gate)
            nn.init.normal_(self.copy_cross_attn.pos_gate_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.copy_cross_attn.pos_gate_proj.bias)
        if self.encoder_blocks is not None:
            for block in self.encoder_blocks:
                for mod in block.modules():
                    if isinstance(mod, nn.Linear):
                        nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                        if mod.bias is not None:
                            nn.init.zeros_(mod.bias)
        if self.cutoff_level_embed is not None:
            nn.init.zeros_(self.cutoff_level_embed.weight)
        # Preserve careful avg-pool/nearest/zero init inside self.quantizer convs;
        # only initialize FSQ linear projections cleanly.
        for fsq in self.quantizer.fsq_modules:
            nn.init.normal_(fsq.proj_in.weight, mean=0.0, std=0.05)
            nn.init.zeros_(fsq.proj_in.bias)
            nn.init.normal_(fsq.proj_out.weight, mean=0.0, std=0.05)
            nn.init.zeros_(fsq.proj_out.bias)

    # ------------------------------------------------------------------
    # Bijective flat <-> digit convenience delegates
    # ------------------------------------------------------------------
    def digits_to_flat(self, digits: torch.Tensor) -> torch.Tensor:
        """Convert FSQ digit tensor [..., d_q] to flat code IDs [...]."""
        return self.quantizer.digits_to_flat(digits)

    def flat_to_digits(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat code IDs [...] to FSQ digit tensor [..., d_q]."""
        return self.quantizer.flat_to_digits(flat)

    # ------------------------------------------------------------------
    # Phase B Oracle Keyword Helper for Self-Supervised Stage 1 Training
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_oracle_kw_token_ids(
        self,
        input_ids: torch.Tensor,
        max_kw: int = 8,
        apply_dropout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract up to `max_kw` salient token IDs per chunk for Phase B Copy Decoder."""
        bsz, _ = input_ids.shape
        device = input_ids.device
        fim_base = self.cfg.vocab_size - 11
        kw_ids = torch.full(
            (bsz, max_kw), self.cfg.pad_token_id, device=device, dtype=torch.long
        )
        kw_mask = torch.zeros((bsz, max_kw), device=device, dtype=torch.bool)

        for b in range(bsz):
            row = input_ids[b].tolist()
            # Prefer multi-byte subwords (272..fim_base-1), then printable ASCII bytes (49..271)
            subwords: List[int] = []
            printable_bytes: List[int] = []
            seen: set[int] = set()
            for tid in row:
                t = int(tid)
                if t in seen or t == self.cfg.pad_token_id:
                    continue
                if 272 <= t < fim_base:
                    seen.add(t)
                    subwords.append(t)
                elif (16 + 33) <= t < 272:
                    seen.add(t)
                    printable_bytes.append(t)
            selected = (subwords + printable_bytes)[:max_kw]
            for j, t in enumerate(selected):
                kw_ids[b, j] = t
                kw_mask[b, j] = True

        drop_p = float(getattr(self.cfg, "copy_kw_drop_prob", 0.30))
        if apply_dropout and self.training and drop_p > 0.0:
            keep_row = torch.rand(bsz, device=device) >= drop_p
            kw_mask = kw_mask & keep_row.unsqueeze(-1)

        return kw_ids, kw_mask

    # ------------------------------------------------------------------
    # Encode / Decode APIs
    # ------------------------------------------------------------------
    def encode_z(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed [B, L] token IDs (+ optional strictly causal semantic encoder) and apply Pre-Quant GroupNorm."""
        z = self.enc_dropout(self.wte(input_ids))
        if self.encoder_blocks is not None:
            bsz, seq_len, _ = z.shape
            pos_ids = torch.arange(seq_len, device=z.device, dtype=torch.long)
            cos, sin = self.dec_rope(pos_ids, dtype=z.dtype)
            enc_mask = build_causal_mask(
                q_len=seq_len,
                kv_len=seq_len,
                past_len=0,
                batch_size=bsz,
                key_padding_mask=None,
                device=z.device,
                dtype=z.dtype,
            )
            for block in self.encoder_blocks:
                z = block(z, cos=cos, sin=sin, attn_mask=enc_mask)
        # GroupNorm expects contiguous [B, C, L] (strict on TPU/XLA backends)
        z_norm = self.pre_quant_norm(z.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return z_norm

    def encode_codes(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Encode token chunk [B, L] into K multiscale FSQ digit tensors [B, l_k, d_q]."""
        del attention_mask  # encoding is spatial per chunk; mask is applied in loss
        z = self.encode_z(input_ids)
        return self.quantizer.encode(z)

    def decode_z_hat(
        self,
        z_hat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kw_token_ids: Optional[torch.Tensor] = None,
        kw_mask: Optional[torch.Tensor] = None,
        cutoff_level: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode continuous quantized latent [B, L, d_vq] -> token logits [B, L, V]."""
        bsz, seq_len, _ = z_hat.shape
        device = z_hat.device
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        cos, sin = self.dec_rope(pos_ids, dtype=z_hat.dtype)
        attn_mask = build_bidirectional_mask(
            seq_len=seq_len,
            batch_size=bsz,
            attention_mask=attention_mask,
            device=device,
            dtype=z_hat.dtype,
        )

        h = z_hat
        if self.cutoff_level_embed is not None and self.cutoff_in_norm is not None:
            k_eff = self.cfg.num_levels if cutoff_level is None else max(1, min(int(cutoff_level), self.cfg.num_levels))
            k_idx = torch.tensor(k_eff - 1, device=device, dtype=torch.long)
            h = self.cutoff_in_norm(h) + self.cutoff_level_embed(k_idx).view(1, 1, self.cfg.d_vq).to(dtype=h.dtype)
        for block in self.decoder_blocks:
            h = block(h, cos=cos, sin=sin, attn_mask=attn_mask)

        pointer_logits: Optional[torch.Tensor] = None
        if self.copy_cross_attn is not None and kw_token_ids is not None:
            h, pointer_logits = self.copy_cross_attn(
                h,
                wte=self.wte,
                kw_token_ids=kw_token_ids,
                kw_mask=kw_mask,
                vocab_size=self.cfg.vocab_size,
                pad_token_id=self.cfg.pad_token_id,
            )

        h = self.dec_norm(h)
        logits = self.output_head(h)
        if pointer_logits is not None:
            logits = logits + pointer_logits
        return logits

    def decode_codes(
        self,
        codes_list: Sequence[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kw_token_ids: Optional[torch.Tensor] = None,
        kw_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode full or partial (early-exit) FSQ codes list -> token logits [B, L, V]."""
        z_hat = self.quantizer.reconstruct_from_codes(
            codes_list, dtype=self.wte.weight.dtype
        )
        return self.decode_z_hat(
            z_hat,
            attention_mask=attention_mask,
            kw_token_ids=kw_token_ids,
            kw_mask=kw_mask,
            cutoff_level=len(codes_list),
        )

    # ------------------------------------------------------------------
    # Stage 1 Training Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        apply_fine_drop: bool = True,
        kw_token_ids: Optional[torch.Tensor] = None,
        kw_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Run full Stage 1 forward pass with nested dropout and reconstruction loss.

        When `cfg.use_copy_decoder=True` and `kw_token_ids is None`, automatically
        extracts oracle keyword token IDs from `input_ids` with 30% keyword dropout.

        Returns dict with keys:
          loss, recon_loss, commit_loss, logits, codes, z_hat,
          active_code_ratio, batch_utilization, recon_acc
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Expected 2D input_ids [B, L], got shape {tuple(input_ids.shape)}")
        if input_ids.shape[1] != self.cfg.chunk_size:
            raise ValueError(
                f"input_ids length ({input_ids.shape[1]}) must match chunk_size ({self.cfg.chunk_size})"
            )

        z = self.encode_z(input_ids)
        q_out = self.quantizer(
            z, attention_mask=attention_mask, apply_fine_drop=apply_fine_drop
        )
        z_hat: torch.Tensor = q_out["z_hat"]  # type: ignore[assignment]
        commit_loss: torch.Tensor = q_out["commit_loss"]  # type: ignore[assignment]

        eff_kw_ids = kw_token_ids
        eff_kw_mask = kw_mask
        if self.copy_cross_attn is not None:
            if eff_kw_ids is None:
                eff_kw_ids, eff_kw_mask = self.extract_oracle_kw_token_ids(
                    input_ids, max_kw=8, apply_dropout=self.training
                )
            elif eff_kw_mask is None and self.training:
                drop_p = float(getattr(self.cfg, "copy_kw_drop_prob", 0.30))
                eff_kw_mask = (eff_kw_ids != self.cfg.pad_token_id) & (eff_kw_ids >= 0)
                if drop_p > 0.0:
                    keep = torch.rand(input_ids.shape[0], device=input_ids.device) >= drop_p
                    eff_kw_mask = eff_kw_mask & keep.unsqueeze(-1)

        logits = self.decode_z_hat(
            z_hat,
            attention_mask=attention_mask,
            kw_token_ids=eff_kw_ids,
            kw_mask=eff_kw_mask,
            cutoff_level=int(q_out.get("cutoff_level", self.cfg.num_levels)),
        )

        def _ce_loss(lg: torch.Tensor) -> torch.Tensor:
            fl = lg.reshape(-1, self.cfg.vocab_size).float()
            ft = input_ids.reshape(-1)
            if attention_mask is not None:
                valid = attention_mask.reshape(-1).to(dtype=torch.bool)
                per_tok = F.cross_entropy(fl, ft, reduction="none")
                denom = valid.float().sum().clamp(min=1.0)
                return (per_tok * valid.float()).sum() / denom
            return F.cross_entropy(fl, ft)

        recon_loss = _ce_loss(logits)

        # Matryoshka Multi-Cutoff Loss: supervise multiple coarse/fine cutoffs every training step
        # so intermediate levels (e.g. k=4, 6, 7) achieve high early-exit fidelity.
        if self.training and self.matryoshka_cutoffs and "z_hat_levels" in q_out:
            z_levels: Sequence[torch.Tensor] = q_out["z_hat_levels"]  # type: ignore[assignment]
            m_losses: List[torch.Tensor] = []
            for k_cut in self.matryoshka_cutoffs:
                if 1 <= k_cut <= len(z_levels):
                    lg_k = self.decode_z_hat(
                        z_levels[k_cut - 1],
                        attention_mask=attention_mask,
                        kw_token_ids=eff_kw_ids,
                        kw_mask=eff_kw_mask,
                        cutoff_level=k_cut,
                    )
                    m_losses.append(_ce_loss(lg_k))
            if m_losses:
                recon_loss = 0.5 * recon_loss + 0.5 * (sum(m_losses) / float(len(m_losses)))

        with torch.no_grad():
            flat_targets = input_ids.reshape(-1)
            if attention_mask is not None:
                valid = attention_mask.reshape(-1).to(dtype=torch.bool)
                preds = logits.argmax(dim=-1).reshape(-1)
                denom = valid.float().sum().clamp(min=1.0)
                recon_acc = ((preds == flat_targets) & valid).float().sum() / denom
            else:
                preds = logits.argmax(dim=-1)
                recon_acc = (preds == input_ids).float().mean()

        loss = recon_loss + commit_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "logits": logits,
            "codes": q_out["codes"],
            "z_hat": z_hat,
            "active_code_ratio": q_out["active_code_ratio"],
            "batch_utilization": q_out["batch_utilization"],
            "recon_acc": recon_acc.detach(),
        }
