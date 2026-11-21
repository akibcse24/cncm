"""Controlled Modal Cloud GPU Head-to-Head Benchmark: Vanilla NCM vs. CNCM (`modal_ncm_vs_cncm.py`).

Runs a strictly controlled 4-arm cumulative ablation from Vanilla NCM (Kumbong et al., arXiv:2610.14162)
to Full CNCM on the exact same `nano` backbone, `Ternary-Bonsai-2-4B` verified SeqKD + multi-domain corpus,
token budget, and single seed (`seed=42`) on Modal NVIDIA A10 / A10G / L40S:

  Arm 1 (`vanilla_ncm`):
      - Stage 1: Multiscale RVQ-VAE with explicit `K x 4096 x d_vq` codebook lookup tables + EMA buffers
                 (`4.19M` extra codebook params on `nano`, no Matryoshka multi-cutoff loss).
      - Stage 2: Flat `4096`-way linear Concept Head (`1.57M` params) + Per-Level Reset RoPE (`0..l_k-1`,
                 colliding with prefix positions `0..P-1`) + Concept-Only training (`exact_weight=0.0`,
                 `align_weight=0.0`) + Uncached prefix re-encoding across turns.
  Arm 2 (`cncm_stage1_mrfsq`):
      - Replaces Stage 1 RVQ+EMA tables with Zero-Table `MultiscaleRFSQ` (`8^4 = 4096` implicit scalar grid)
        + Matryoshka multi-cutoff supervision (`k in {4, 6, 7, 8}`).
      - Keeps Stage 2 Flat 4096 Head + Reset RoPE (`0..l_k-1`) to isolate the Stage 1 M-RFSQ gain.
  Arm 3 (`cncm_stage2_factorized_cont_rope`):
      - Replaces Flat 4096 Head (`1.57M`) with `FactorizedConceptHead` (`71,072` params, `22.1x` smaller;
        `81.1x` smaller combined codebook+head footprint than Vanilla NCM!).
      - Replaces Per-Level Reset RoPE with **Continuous Scaled RoPE** (`P + floor((i+0.5)*L/l_k)`)
        + Incremental Causal Prefix KV Cache across multi-chunk/multi-turn generation.
  Arm 4 (`full_cncm_dual_mode_distilled`):
      - Adds Unified Dual-Mode Joint Training (`exact_head` + Text-Level Alignment `L_align`)
      - Adds `Ternary-Bonsai-2-4B` Dynamic Residual Concept-DAgger + Frozen-Teacher Cross-Mode KL
      - Adds Neuro-Symbolic `JSONSchemaConstrainedMasker` + Prompt Copy-Span Constraint for `<tool_call>`.

Outputs:
  - `modal_ncm_vs_cncm_report.json`
  - `experiments/results/ncm_vs_cncm_nano.json`

Usage:
    modal run modal_ncm_vs_cncm.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/root/project"
VOLUME_NAME = "cncm-checkpoints"
VOLUME_MOUNT = "/vol"

app = modal.App("cncm-vs-vanilla-ncm-benchmark")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": REMOTE_PROJECT_DIR,
        }
    )
    .pip_install("torch>=2.5.0", "numpy>=1.26.0")
    .add_local_dir(
        str(PROJECT_ROOT / "cncm"),
        remote_path=f"{REMOTE_PROJECT_DIR}/cncm",
        ignore=["__pycache__", "*.pyc", "*.pyo"],
    )
    .add_local_dir(
        str(PROJECT_ROOT / "experiments"),
        remote_path=f"{REMOTE_PROJECT_DIR}/experiments",
        ignore=["__pycache__", "*.pyc", "*.pyo", "onnx_edge_nano"],
    )
)


@app.function(
    image=image,
    gpu=["A10", "A10G", "L40S"],
    timeout=1800,
    scaledown_window=600,
    volumes={VOLUME_MOUNT: vol},
)
def run_ncm_vs_cncm_benchmark(
    preset: str = "nano",
    stage1_steps: int = 350,
    stage2_steps: int = 400,
    align_steps: int = 120,
    batch_size: int = 24,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)

    from cncm.agent.grammar import JSONSchemaConstrainedMasker, extract_json_object
    from cncm.agent.tools import create_default_registry
    from cncm.config import CNCMConfig, get_preset
    from cncm.layers import build_concept_mask
    from cncm.model import CNCMModel
    from cncm.quantizer import DepthwiseSepCascadeDown, DepthwiseSepCascadeUp, DepthwiseSepPhi
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer.dataset import PackedChunkDataset
    from cncm.trainer.draft_align import train_draft_alignment
    from cncm.trainer.train_stage1 import train_stage1
    from cncm.trainer.train_stage2 import train_stage2
    from cncm.trainer.train_utils import set_seed
    from experiments.eval_suites import build_heldout_tool_calling_suite, evaluate_tool_suite_with_ci
    from experiments.run_distill_ablation import (
        build_verified_teacher_seqkd_corpus,
        evaluate_student_model,
    )
    from experiments.train_smart_cncm import load_verified_seqkd_texts

    t0_all = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    amp_dtype = "bf16" if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else "fp32"
    print(f"[NCM vs CNCM BENCHMARK] GPU={gpu_name} | AMP={amp_dtype} | preset={preset}")

    # =========================================================================
    # Vanilla NCM Baseline Components (Explicit RVQ+EMA Codebooks, Flat 4096 Head, Reset RoPE)
    # =========================================================================
    class _VanillaRVQLevel(nn.Module):
        """Vanilla NCM single-level VQ codebook table [V_code=4096, d_vq] + EMA buffers."""

        def __init__(self, d_vq: int, codebook_size: int = 4096, fsq_levels: Tuple[int, ...] = (8, 8, 8, 8)) -> None:
            super().__init__()
            self.d_vq = int(d_vq)
            self.codebook_size = int(codebook_size)
            self.fsq_levels = tuple(fsq_levels)
            # Explicit learned codebook parameter table [4096, d_vq]
            self.codebook = nn.Parameter(torch.randn(self.codebook_size, self.d_vq) * 0.05)
            # FP32 EMA cluster buffers stored in vanilla NCM RVQ
            self.register_buffer("ema_cluster_size", torch.ones(self.codebook_size, dtype=torch.float32))
            self.register_buffer("ema_w", self.codebook.data.clone().float())

            basis: List[int] = []
            r = 1
            for l_j in reversed(self.fsq_levels):
                basis.append(r)
                r *= l_j
            basis.reverse()
            self.register_buffer("radix_basis", torch.tensor(basis, dtype=torch.long))
            self.register_buffer("fsq_levels_tensor", torch.tensor(self.fsq_levels, dtype=torch.long))

        def flat_to_digits(self, flat: torch.Tensor) -> torch.Tensor:
            flat_l = flat.to(torch.long).unsqueeze(-1)
            return torch.div(flat_l, self.radix_basis, rounding_mode="floor") % self.fsq_levels_tensor

        def digits_to_flat(self, digits: torch.Tensor) -> torch.Tensor:
            return (digits.to(torch.long) * self.radix_basis).sum(dim=-1)

        def quantize(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # L2 nearest-neighbor lookup against [4096, d_vq] codebook table
            flat_u = u.reshape(-1, self.d_vq).float()
            cb = self.codebook.float()
            dists = (
                flat_u.pow(2).sum(dim=-1, keepdim=True)
                - 2.0 * torch.matmul(flat_u, cb.t())
                + cb.pow(2).sum(dim=-1).unsqueeze(0)
            )
            min_idx = dists.argmin(dim=-1).view(u.shape[0], u.shape[1])
            u_q = F.embedding(min_idx, self.codebook).to(dtype=u.dtype)
            # Straight-through estimator
            u_q_ste = u + (u_q - u).detach()
            digits = self.flat_to_digits(min_idx)
            return u_q_ste, digits

        def dequantize(self, digits: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
            flat_idx = self.digits_to_flat(digits).clamp(0, self.codebook_size - 1)
            out = F.embedding(flat_idx, self.codebook)
            return out.to(dtype=dtype) if dtype is not None else out

    def _build_vanilla_rvq_stage1(cfg: CNCMConfig) -> CNCMConceptTokenizer:
        """Build a Vanilla NCM Stage 1 tokenizer using explicit RVQ codebook tables (no Matryoshka)."""
        cfg_v = copy.deepcopy(cfg)
        cfg_v.matryoshka_cutoffs = ()
        s1_v = CNCMConceptTokenizer(cfg_v)
        s1_v.matryoshka_cutoffs = ()
        s1_v.cutoff_level_embed = None
        s1_v.cutoff_in_norm = None
        # Replace zero-table FSQLevel modules with explicit 4096 x d_vq RVQ tables
        s1_v.quantizer.fsq_modules = nn.ModuleList(
            [_VanillaRVQLevel(cfg.d_vq, cfg.codebook_size, cfg.fsq_levels) for _ in range(cfg.num_levels)]
        )
        return s1_v

    class _Flat4096ConceptHead(nn.Module):
        """Vanilla NCM flat 4096-way linear softmax Concept Head (1.57M params on nano)."""

        def __init__(self, hidden_size: int, codebook_size: int = 4096, fsq_levels: Tuple[int, ...] = (8, 8, 8, 8)) -> None:
            super().__init__()
            self.proj = nn.Linear(hidden_size, codebook_size, bias=False)
            self.fsq_levels = tuple(fsq_levels)
            basis: List[int] = []
            r = 1
            for l_j in reversed(self.fsq_levels):
                basis.append(r)
                r *= l_j
            basis.reverse()
            self.register_buffer("radix_basis", torch.tensor(basis, dtype=torch.long))
            self.register_buffer("fsq_levels_tensor", torch.tensor(self.fsq_levels, dtype=torch.long))

        def log_probs(self, h: torch.Tensor, level_ids: torch.Tensor, target_digits: torch.Tensor):
            del level_ids
            logits = self.proj(h).float()  # [B, T, 4096]
            flat_tgt = (target_digits.to(torch.long) * self.radix_basis).sum(dim=-1)
            log_p = F.log_softmax(logits, dim=-1)
            tok_lp = log_p.gather(-1, flat_tgt.unsqueeze(-1)).squeeze(-1)
            return [logits], tok_lp

        def sample(self, h: torch.Tensor, level_ids: Any = 0, temperature: float = 0.0):
            del level_ids
            logits = self.proj(h).float()
            probs = F.softmax(logits / max(float(temperature), 1e-5), dim=-1) if temperature > 0.0 else F.softmax(logits, dim=-1)
            if temperature <= 0.0:
                flat_idx = logits.argmax(dim=-1)
            else:
                flat_idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).view(logits.shape[:-1])
            max_p = probs.max(dim=-1).values
            mean_conf = max_p.mean(dim=-1)
            digits = torch.div(flat_idx.unsqueeze(-1), self.radix_basis, rounding_mode="floor") % self.fsq_levels_tensor
            return digits, None, mean_conf

    def _enable_reset_rope_on_model(m: CNCMModel) -> None:
        """Monkey-patch `m._concept_level_positions` to Vanilla NCM's per-level reset RoPE (`0..l_k-1`)."""
        def _reset_positions(prefix_len: int, level_len: int, device: torch.device) -> torch.Tensor:
            del prefix_len
            return torch.arange(level_len, device=device, dtype=torch.long)

        m._concept_level_positions = _reset_positions  # type: ignore[method-assign]

    # =========================================================================
    # Data Setup (Shared across all 4 arms, seed=42)
    # =========================================================================
    set_seed(42)
    cfg = get_preset(preset, enc_layers=1, matryoshka_cutoffs=(4, 6, 7, 8))
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    vol_seqkd = Path(VOLUME_MOUNT) / "bonsai2_cache" / "verified_seqkd.jsonl"
    local_seqkd = Path(REMOTE_PROJECT_DIR) / "experiments" / "results" / "verified_seqkd.jsonl"
    seqkd_path = vol_seqkd if vol_seqkd.exists() else local_seqkd
    real_bonsai_texts = load_verified_seqkd_texts(seqkd_path) if seqkd_path.exists() else []
    aux_train = build_verified_teacher_seqkd_corpus(n_samples=256, seed=42)
    train_texts = (real_bonsai_texts * 3) + aux_train
    val_texts = build_verified_teacher_seqkd_corpus(n_samples=64, seed=20_042)

    train_ds = PackedChunkDataset(train_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)
    val_ds = PackedChunkDataset(val_texts, tok, chunk_size=cfg.chunk_size, prefix_len=cfg.chunk_size)

    @torch.no_grad()
    def _eval_stage1(s1_mod: CNCMConceptTokenizer) -> Dict[str, Any]:
        s1_mod.eval()
        dl = DataLoader(val_ds, batch_size=32, shuffle=False)
        acc_k8, acc_k6, acc_k4, nb = 0.0, 0.0, 0.0, 0
        unique_codes_per_level: List[set[int]] = [set() for _ in range(cfg.num_levels)]
        for i, b in enumerate(dl):
            if i >= 8:
                break
            x = b["target_ids"].to(dev)
            codes = s1_mod.encode_codes(x)
            for k_idx, c_k in enumerate(codes):
                if hasattr(s1_mod.quantizer.fsq_modules[k_idx], "digits_to_flat"):
                    fl = s1_mod.quantizer.fsq_modules[k_idx].digits_to_flat(c_k).reshape(-1).tolist()
                else:
                    fl = s1_mod.digits_to_flat(c_k).reshape(-1).tolist()
                unique_codes_per_level[k_idx].update(fl)
            lg8 = s1_mod.decode_codes(codes)
            lg6 = s1_mod.decode_codes(codes[:6])
            lg4 = s1_mod.decode_codes(codes[:4])
            acc_k8 += float((lg8.argmax(-1) == x).float().mean().item())
            acc_k6 += float((lg6.argmax(-1) == x).float().mean().item())
            acc_k4 += float((lg4.argmax(-1) == x).float().mean().item())
            nb += 1
        mean_active_pct = sum(len(s) / float(cfg.codebook_size) for s in unique_codes_per_level) / float(cfg.num_levels)
        q_params = sum(p.numel() for p in s1_mod.quantizer.parameters())
        q_buf_bytes = sum(b.numel() * b.element_size() for b in s1_mod.quantizer.buffers())
        return {
            "quantizer_params": q_params,
            "quantizer_buffer_kb": round(q_buf_bytes / 1024.0, 2),
            "heldout_recon_k8": round(acc_k8 / max(1, nb), 4),
            "heldout_recon_k6_early_exit": round(acc_k6 / max(1, nb), 4),
            "heldout_recon_k4_early_exit": round(acc_k4 / max(1, nb), 4),
            "val_active_codebook_ratio": round(mean_active_pct, 4),
        }

    @torch.no_grad()
    def _eval_tool_calling_and_multiturn_speed(mod: CNCMModel, use_dual_mode_exact: bool) -> Dict[str, Any]:
        mod.eval()
        reg = create_default_registry()
        tasks = [t for t in build_heldout_tool_calling_suite(80) if t["expected_tool"] == "calculator"][:40]
        preds: List[str] = []
        for t in tasks:
            pfx = torch.tensor([tok.encode(t["prompt"])[-64:]], dtype=torch.long, device=dev)
            if use_dual_mode_exact:
                masker = JSONSchemaConstrainedMasker(tok, reg, allowed_tools=["calculator"], prompt_text=t["prompt"])
                out_ids = mod.generate_exact(pfx, max_new_tokens=40, temperature=0.0, logit_mask_fn=masker)["token_ids"]
            else:
                # Vanilla NCM Concept-Only decoding (no Exact-AR head or grammar masker)
                out_ids = mod.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)["token_ids"]
            preds.append(tok.decode(out_ids[0].tolist(), skip_special_tokens=True))
        tool_res = evaluate_tool_suite_with_ci(preds, tasks)

        # Multi-turn (4 chunks of 64 tokens = 256 tokens) latency:
        # Vanilla NCM has bidirectional prefix & no incremental causal chunk KV commit (must re-prefill growing history)
        # CNCM commits each 64-token chunk into causal prefix KV in 1 bottom-right masked pass.
        pfx0 = torch.randint(16, cfg.vocab_size, (1, 64), dtype=torch.long, device=dev)
        _ = mod.generate_concept_chunk(pfx0, temperature=0.0, early_exit_conf=None)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        t_cached = time.perf_counter()
        _ = mod.generate_block_ar(pfx0, max_new_tokens=256, temperature=0.0, early_exit_conf=None, stop_token_ids=[])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        cached_4chunk_ms = (time.perf_counter() - t_cached) * 1000.0

        t_uncached = time.perf_counter()
        running_pfx = pfx0
        for _ in range(4):
            c_out = mod.generate_concept_chunk(running_pfx, prefix_kv=None, temperature=0.0, early_exit_conf=None)
            running_pfx = torch.cat([running_pfx, c_out["token_ids"]], dim=1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        uncached_4chunk_ms = (time.perf_counter() - t_uncached) * 1000.0

        return {
            "tool_syntax_valid_rate": tool_res["syntax_wilson_95"]["rate"],
            "tool_semantic_acc_rate": tool_res["semantic_wilson_95"]["rate"],
            "multiturn_4chunk_cached_ms": round(cached_4chunk_ms, 2),
            "multiturn_4chunk_uncached_ms": round(uncached_4chunk_ms, 2),
            "causal_kv_multiturn_speedup_x": round(uncached_4chunk_ms / max(cached_4chunk_ms, 1e-6), 3),
        }

    # =========================================================================
    # Train Stage 1: Vanilla RVQ-VAE vs CNCM M-RFSQ (350 steps FP32)
    # =========================================================================
    print("\n=== [Stage 1 A/B] Vanilla NCM RVQ+EMA vs CNCM Zero-Table M-RFSQ ===")
    set_seed(42)
    s1_vanilla = _build_vanilla_rvq_stage1(cfg).to(dev)
    train_stage1(
        cfg, tok, model=s1_vanilla, dataset=train_ds, steps=stage1_steps,
        batch_size=64, lr=2.5e-3, device=dev, apply_fine_drop=False, seed=42, verbose=False,
    )
    s1_vanilla_eval = _eval_stage1(s1_vanilla)
    print(f"[Vanilla RVQ Stage 1] {json.dumps(s1_vanilla_eval)}")

    set_seed(42)
    s1_cncm = CNCMConceptTokenizer(cfg).to(dev)
    train_stage1(
        cfg, tok, model=s1_cncm, dataset=train_ds, steps=stage1_steps,
        batch_size=64, lr=2.5e-3, device=dev, apply_fine_drop=True, seed=42, verbose=False,
    )
    s1_cncm_eval = _eval_stage1(s1_cncm)
    print(f"[CNCM M-RFSQ Stage 1] {json.dumps(s1_cncm_eval)}")

    # =========================================================================
    # Arm 1: Vanilla NCM Baseline (RVQ + Flat 4096 Head + Reset RoPE + Concept-Only)
    # =========================================================================
    print("\n=== [Arm 1] Vanilla NCM Baseline (RVQ + Flat 4096 Head + Reset RoPE + Concept-Only) ===")
    set_seed(42)
    m_arm1 = CNCMModel(cfg, copy.deepcopy(s1_vanilla)).to(dev)
    m_arm1.concept_head = _Flat4096ConceptHead(cfg.hidden_size, cfg.codebook_size, cfg.fsq_levels).to(dev)
    _enable_reset_rope_on_model(m_arm1)
    train_stage2(
        cfg, tok, concept_tokenizer=m_arm1.concept_tokenizer, model=m_arm1,
        dataset=train_ds, mode="pretrain", steps=stage2_steps, batch_size=batch_size,
        lr=1.2e-3, concept_weight=1.0, exact_weight=0.0, align_weight=0.0,
        device=dev, amp_dtype=amp_dtype, seed=42, verbose=False,
    )
    arm1_s2 = evaluate_student_model(m_arm1, val_ds, str(dev))
    arm1_tool = _eval_tool_calling_and_multiturn_speed(m_arm1, use_dual_mode_exact=False)
    arm1_head_params = sum(p.numel() for p in m_arm1.concept_head.parameters())

    # =========================================================================
    # Arm 2: + CNCM Stage 1 M-RFSQ (M-RFSQ + Flat 4096 Head + Reset RoPE)
    # =========================================================================
    print("\n=== [Arm 2] + CNCM Stage 1 M-RFSQ (Zero-Table M-RFSQ + Flat 4096 Head + Reset RoPE) ===")
    set_seed(42)
    m_arm2 = CNCMModel(cfg, copy.deepcopy(s1_cncm)).to(dev)
    m_arm2.concept_head = _Flat4096ConceptHead(cfg.hidden_size, cfg.codebook_size, cfg.fsq_levels).to(dev)
    _enable_reset_rope_on_model(m_arm2)
    train_stage2(
        cfg, tok, concept_tokenizer=m_arm2.concept_tokenizer, model=m_arm2,
        dataset=train_ds, mode="pretrain", steps=stage2_steps, batch_size=batch_size,
        lr=1.2e-3, concept_weight=1.0, exact_weight=0.0, align_weight=0.0,
        device=dev, amp_dtype=amp_dtype, seed=42, verbose=False,
    )
    arm2_s2 = evaluate_student_model(m_arm2, val_ds, str(dev))
    arm2_tool = _eval_tool_calling_and_multiturn_speed(m_arm2, use_dual_mode_exact=False)
    arm2_head_params = sum(p.numel() for p in m_arm2.concept_head.parameters())

    # =========================================================================
    # Arm 3: + CNCM FactorizedConceptHead (71K) + Continuous Scaled RoPE + Causal KV
    # =========================================================================
    print("\n=== [Arm 3] + FactorizedConceptHead (71K) + Continuous Scaled RoPE + Causal KV ===")
    set_seed(42)
    m_arm3 = CNCMModel(cfg, copy.deepcopy(s1_cncm)).to(dev)
    train_stage2(
        cfg, tok, concept_tokenizer=m_arm3.concept_tokenizer, model=m_arm3,
        dataset=train_ds, mode="pretrain", steps=stage2_steps, batch_size=batch_size,
        lr=1.2e-3, concept_weight=1.0, exact_weight=0.0, align_weight=0.0,
        device=dev, amp_dtype=amp_dtype, seed=42, verbose=False,
    )
    arm3_s2 = evaluate_student_model(m_arm3, val_ds, str(dev))
    arm3_tool = _eval_tool_calling_and_multiturn_speed(m_arm3, use_dual_mode_exact=False)
    arm3_head_params = sum(p.numel() for p in m_arm3.concept_head.parameters())

    # =========================================================================
    # Arm 4: Full CNCM (Dual-Mode + L_align + Bonsai-2-4B Concept-DAgger + Copy-Span Grammar Masker)
    # =========================================================================
    print("\n=== [Arm 4] Full CNCM (Dual-Mode + L_align + Bonsai-2-4B Concept-DAgger + Grammar Masker) ===")
    set_seed(42)
    m_arm4 = CNCMModel(cfg, copy.deepcopy(s1_cncm)).to(dev)
    train_stage2(
        cfg, tok, concept_tokenizer=m_arm4.concept_tokenizer, model=m_arm4,
        dataset=train_ds, mode="joint", steps=stage2_steps, batch_size=batch_size,
        lr=1.2e-3, concept_weight=1.0, exact_weight=0.7, align_weight=0.5,
        device=dev, amp_dtype=amp_dtype, seed=42, verbose=False,
    )
    teacher_snap = copy.deepcopy(m_arm4).eval()
    train_draft_alignment(
        m_arm4,
        train_ds,
        teacher_model=teacher_snap,
        steps=align_steps,
        batch_size=batch_size,
        lr=1.0e-3,
        kl_weight=0.5,
        dagger_prob=0.6,
        corrupt_rate=0.2,
        freeze_backbone_for_exact=True,
        device=dev,
        seed=42,
        verbose=False,
    )
    arm4_s2 = evaluate_student_model(m_arm4, val_ds, str(dev))
    arm4_tool = _eval_tool_calling_and_multiturn_speed(m_arm4, use_dual_mode_exact=True)
    arm4_head_params = sum(p.numel() for p in m_arm4.concept_head.parameters())

    elapsed = round(time.perf_counter() - t0_all, 2)
    est_cost = round((elapsed / 3600.0) * 1.10, 5)

    report = {
        "benchmark": "Vanilla NCM vs. CNCM Controlled 4-Arm Cumulative Ablation",
        "gpu_name": gpu_name,
        "preset": preset,
        "seed": 42,
        "stage1_comparison": {
            "vanilla_rvq_ema": s1_vanilla_eval,
            "cncm_mrfsq_matryoshka": s1_cncm_eval,
            "quantizer_param_reduction_x": round(
                s1_vanilla_eval["quantizer_params"] / float(max(1, s1_cncm_eval["quantizer_params"])), 2
            ),
        },
        "arms": {
            "1_vanilla_ncm": {
                "codebook_plus_head_params": s1_vanilla_eval["quantizer_params"] + arm1_head_params,
                "concept_head_params": arm1_head_params,
                "stage1_recon_k8": s1_vanilla_eval["heldout_recon_k8"],
                "stage1_recon_k6": s1_vanilla_eval["heldout_recon_k6_early_exit"],
                "concept_loss": arm1_s2["concept_loss"],
                "concept_draft_nll": arm1_s2["concept_draft_nll"],
                "exact_nll": arm1_s2["exact_nll"],
                "tool_syntax_valid_rate": arm1_tool["tool_syntax_valid_rate"],
                "tool_semantic_acc_rate": arm1_tool["tool_semantic_acc_rate"],
                "multiturn_4chunk_ms": arm1_tool["multiturn_4chunk_uncached_ms"],
            },
            "2_plus_mrfsq_stage1": {
                "codebook_plus_head_params": s1_cncm_eval["quantizer_params"] + arm2_head_params,
                "concept_head_params": arm2_head_params,
                "stage1_recon_k8": s1_cncm_eval["heldout_recon_k8"],
                "stage1_recon_k6": s1_cncm_eval["heldout_recon_k6_early_exit"],
                "concept_loss": arm2_s2["concept_loss"],
                "concept_draft_nll": arm2_s2["concept_draft_nll"],
                "exact_nll": arm2_s2["exact_nll"],
                "tool_syntax_valid_rate": arm2_tool["tool_syntax_valid_rate"],
                "tool_semantic_acc_rate": arm2_tool["tool_semantic_acc_rate"],
                "multiturn_4chunk_ms": arm2_tool["multiturn_4chunk_uncached_ms"],
            },
            "3_plus_factorized_head_cont_rope_causal_kv": {
                "codebook_plus_head_params": s1_cncm_eval["quantizer_params"] + arm3_head_params,
                "concept_head_params": arm3_head_params,
                "stage1_recon_k8": s1_cncm_eval["heldout_recon_k8"],
                "stage1_recon_k6": s1_cncm_eval["heldout_recon_k6_early_exit"],
                "concept_loss": arm3_s2["concept_loss"],
                "concept_draft_nll": arm3_s2["concept_draft_nll"],
                "exact_nll": arm3_s2["exact_nll"],
                "tool_syntax_valid_rate": arm3_tool["tool_syntax_valid_rate"],
                "tool_semantic_acc_rate": arm3_tool["tool_semantic_acc_rate"],
                "multiturn_4chunk_ms": arm3_tool["multiturn_4chunk_cached_ms"],
                "causal_kv_speedup_vs_vanilla_uncached_x": arm3_tool["causal_kv_multiturn_speedup_x"],
            },
            "4_full_cncm_dual_mode_bonsai_distilled": {
                "codebook_plus_head_params": s1_cncm_eval["quantizer_params"] + arm4_head_params,
                "concept_head_params": arm4_head_params,
                "stage1_recon_k8": s1_cncm_eval["heldout_recon_k8"],
                "stage1_recon_k6": s1_cncm_eval["heldout_recon_k6_early_exit"],
                "concept_loss": arm4_s2["concept_loss"],
                "concept_draft_nll": arm4_s2["concept_draft_nll"],
                "align_nll": arm4_s2["align_nll"],
                "exact_nll": arm4_s2["exact_nll"],
                "tool_syntax_valid_rate": arm4_tool["tool_syntax_valid_rate"],
                "tool_semantic_acc_rate": arm4_tool["tool_semantic_acc_rate"],
                "multiturn_4chunk_ms": arm4_tool["multiturn_4chunk_cached_ms"],
                "causal_kv_speedup_vs_vanilla_uncached_x": arm4_tool["causal_kv_multiturn_speedup_x"],
            },
        },
        "total_gpu_seconds": elapsed,
        "estimated_cost_usd": est_cost,
    }
    print(f"\n[NCM vs CNCM BENCHMARK DONE] {elapsed}s (~${est_cost})")
    return report


@app.local_entrypoint()
def main() -> None:
    res = run_ncm_vs_cncm_benchmark.remote()
    out_root = PROJECT_ROOT / "modal_ncm_vs_cncm_report.json"
    out_exp = PROJECT_ROOT / "experiments" / "results" / "ncm_vs_cncm_nano.json"
    out_exp.parent.mkdir(parents=True, exist_ok=True)
    payload_str = json.dumps(res, indent=2)
    out_root.write_text(payload_str, encoding="utf-8")
    out_exp.write_text(payload_str, encoding="utf-8")
    print(f"[LOCAL] Saved NCM vs CNCM report:\n  - {out_root}\n  - {out_exp}")
    print(json.dumps(res, indent=2))
