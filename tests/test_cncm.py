"""Comprehensive end-to-end verification suite for CNCM (Compact Causal-Hybrid Next-Concept Model).

Covers all 6 verification pillars:
  1. Preset & Parameter Verification (`tiny_test`, `nano`, `mini`, `base`; `nano` INT4 < 15 MB)
  2. M-RFSQ Stage 1 (zero NaNs, 100% finite STE grads, nested dropout, utilization, bijection, decode)
  3. 2D RoPE (`S == num_attention_heads`) & 4D Mask Invariants (bottom-right causal chunk commit)
  4. Dual-Mode Generation & Lossless Self-Speculative Decoding (FP & INT8 KV, early exit, masks)
  5. End-to-End Training Smoke Tests (Stage 1, Stage 2 Joint/SFT/QAT/Self-Distill, GRPO/CISPO, ckpts)
  6. Agentic AI (`JSONSchemaConstrainedMasker` + `CNCMAgent`), Edge INT4/INT8 Quantization, ONNX & CLI Demo
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cncm import (
    CNCMConceptTokenizer,
    CNCMConfig,
    CNCMModel,
    CNCMTextTokenizer,
    MultiscaleRFSQ,
    get_preset,
    list_presets,
)
from cncm.agent import (
    CNCMAgent,
    JSONSchemaConstrainedMasker,
    ToolRegistry,
    create_default_registry,
    extract_json_object,
    safe_eval_math,
)
from cncm.export import (
    Int4PackedLinear,
    Int8Embedding,
    Int8Linear,
    export_cncm_onnx,
    measure_model_size,
    profile_edge_targets,
    quantize_model_int4,
    quantize_model_int8,
)
from cncm.layers import QuantizedKV, kv_cache_nbytes
from cncm.trainer import (
    BlockARSFTDataset,
    GRPOPromptDataset,
    PackedChunkDataset,
    WarmupCosineScheduler,
    collate_block_ar,
    collate_grpo_prompts,
    compute_verifiable_reward,
    load_checkpoint,
    make_synthetic_corpus,
    make_synthetic_grpo_prompts,
    make_synthetic_sft,
    save_checkpoint,
    set_seed,
    train_grpo,
    train_stage1,
    train_stage2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# 1. Preset, Config, Text Tokenizer & Memory Verification
# ============================================================================


def test_presets_instantiate_and_param_targets() -> None:
    """Verify all 4 presets instantiate cleanly and meet parameter/memory targets."""
    expected_presets = ("tiny_test", "nano", "mini", "base")
    assert set(expected_presets).issubset(set(list_presets()))

    param_ranges = {
        "tiny_test": (200_000, 1_000_000),
        "nano": (10_000_000, 20_000_000),
        "mini": (30_000_000, 50_000_000),
        "base": (75_000_000, 115_000_000),
    }

    for name in expected_presets:
        cfg = get_preset(name)
        assert cfg.preset_name == name
        assert cfg.levels[-1] == cfg.chunk_size
        assert cfg.effective_layers == cfg.num_unique_layers * cfg.num_loops

        # Config round-trip serialization
        cfg_rt = CNCMConfig.from_dict(cfg.to_dict())
        assert cfg_rt == cfg

        s1 = CNCMConceptTokenizer(cfg)
        model = CNCMModel(cfg, s1)
        counts = model.count_parameters()
        lo, hi = param_ranges[name]
        assert lo <= counts["total"] <= hi, f"{name}: total params {counts['total']:,} outside [{lo:,}, {hi:,}]"
        # Stage 1 must be frozen by default on CNCMModel construction
        assert counts["trainable"] == counts["generator"]

        # Verify nano INT4 < 15 MB and mini INT4 < 30 MB on REAL packed quantization!
        if name == "nano":
            q4 = quantize_model_int4(model, group_size=32, quantize_embeddings=True)
            sz4 = measure_model_size(q4)
            assert sz4["total_mb"] < 15.0, f"nano packed INT4 exceeded 15 MB: {sz4['total_mb']:.2f} MB"
            prof = profile_edge_targets(cfg, model=model, context_len=1024)
            assert prof["disk_ram_mb"]["int4"] < 15.0
            assert len(prof["targets"]) == 4
            del q4
        elif name == "mini":
            q4 = quantize_model_int4(model, group_size=32, quantize_embeddings=True)
            sz4 = measure_model_size(q4)
            assert sz4["total_mb"] < 30.0, f"mini packed INT4 exceeded 30 MB: {sz4['total_mb']:.2f} MB"
            del q4

        del model, s1
        gc.collect()


def test_text_tokenizer_roundtrip_specials_and_templates(tmp_path: Path) -> None:
    """Verify CNCMTextTokenizer matches CNCMConfig IDs, round-trips UTF-8, chat templates & HF JSON."""
    cfg = get_preset("tiny_test")
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    assert len(tok) == cfg.vocab_size

    # Special token IDs synced with CNCMConfig
    for attr in (
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
        "system_start_id",
        "user_start_id",
        "assistant_start_id",
        "turn_end_id",
        "think_start_id",
        "think_end_id",
        "tool_call_start_id",
        "tool_call_end_id",
        "tool_resp_start_id",
        "tool_resp_end_id",
        "tools_start_id",
        "tools_end_id",
    ):
        assert getattr(tok, attr) == getattr(cfg, attr), f"Special token mismatch on {attr}"

    # Lossless UTF-8 + special-tag round-trip (including multi-byte Unicode)
    sample = '<think>Check 12 * 7 = 84 & café 🚀</think><tool_call>{"name":"calculator","arguments":{"expression":"12*7"}}</tool_call>'
    ids = tok.encode(sample, add_special_tokens=False)
    assert tok.think_start_id in ids and tok.think_end_id in ids
    assert tok.tool_call_start_id in ids and tok.tool_call_end_id in ids
    assert tok.decode(ids, skip_special_tokens=False) == sample

    # Chat template with tools
    reg = create_default_registry()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Compute 2 + 2"}],
        tools=reg.list_specs(),
        add_generation_prompt=True,
    )
    assert "<|tools|>" in prompt and "<|user|>" in prompt and prompt.endswith("<|assistant|>")

    # Optional HF tokenizer.json loader
    hf_json = tmp_path / "tokenizer.json"
    hf_json.write_text(
        json.dumps({"model": {"vocab": {"Ġconcept": 0, "Ġmodel": 1, "edge": 2}}}),
        encoding="utf-8",
    )
    tok_hf = CNCMTextTokenizer.from_hf_json(hf_json, vocab_size=512)
    assert tok_hf.decode(tok_hf.encode("concept model edge <think>ok</think>")) == "concept model edge <think>ok</think>"


# ============================================================================
# 2. M-RFSQ Stage 1 Concept Tokenizer Invariants
# ============================================================================


def test_stage1_mrfsq_invariants_ste_grads_nested_dropout_and_bijection() -> None:
    """Verify M-RFSQ: zero EMA/codebook tables, affine-free GN, finite STE grads, nested dropout, bijection."""
    set_seed(123)
    cfg = get_preset("tiny_test")
    s1 = CNCMConceptTokenizer(cfg)

    # 1. Structural guarantees: affine-free GroupNorm, zero EMA buffers, zero codebook tables
    assert isinstance(s1.pre_quant_norm, nn.GroupNorm)
    assert s1.pre_quant_norm.affine is False
    assert s1.pre_quant_norm.weight is None and s1.pre_quant_norm.bias is None
    for bname, _ in s1.quantizer.named_buffers():
        assert "ema" not in bname.lower(), f"Unexpected EMA buffer in MultiscaleRFSQ: {bname}"
    for pname, param in s1.quantizer.named_parameters():
        assert cfg.codebook_size not in param.shape, f"Unexpected codebook table parameter: {pname} {param.shape}"

    # 2. Exact bijective round-trip on uniform AND mixed-radix FSQ grids
    for fsq_levels in ((4, 4, 4, 4), (3, 5, 7, 4), (8, 8, 8, 8)):
        cfg_q = get_preset("tiny_test", fsq_levels=fsq_levels)
        q = MultiscaleRFSQ(cfg_q)
        all_ids = torch.arange(cfg_q.codebook_size, dtype=torch.long)
        digits = q.flat_to_digits(all_ids)
        assert digits.shape == (cfg_q.codebook_size, len(fsq_levels))
        for j, lvl in enumerate(fsq_levels):
            assert int(digits[:, j].min().item()) == 0
            assert int(digits[:, j].max().item()) == lvl - 1
        rt_flat = q.digits_to_flat(digits)
        assert torch.equal(rt_flat, all_ids)
        # Also verify CNCMConceptTokenizer helper delegates match
        if fsq_levels == cfg.fsq_levels:
            assert torch.equal(s1.digits_to_flat(s1.flat_to_digits(all_ids)), all_ids)

    # 3. Forward + backward STE gradients (100% finite, zero NaNs, utilization > 0)
    s1.train()
    x = torch.randint(16, cfg.vocab_size, (4, cfg.chunk_size), dtype=torch.long)
    out = s1(x, apply_fine_drop=False)

    for key in ("loss", "recon_loss", "commit_loss", "logits", "codes", "z_hat", "active_code_ratio"):
        assert key in out, f"Missing key {key!r} from Stage 1 forward output"

    assert torch.isfinite(out["loss"]).item()
    assert torch.isfinite(out["recon_loss"]).item()
    assert torch.isfinite(out["commit_loss"]).item()
    assert torch.isfinite(out["logits"]).all().item()
    assert torch.isfinite(out["z_hat"]).all().item()
    assert float(out["active_code_ratio"].item()) > 0.0
    assert float(out["batch_utilization"].item()) > 0.0

    out["loss"].backward()
    nonzero_grad_params = 0
    for pname, p in s1.named_parameters():
        assert p.grad is not None, f"Parameter {pname} received None gradient"
        assert torch.isfinite(p.grad).all().item(), f"Non-finite gradient in {pname}"
        if p.grad.abs().sum().item() > 0:
            nonzero_grad_params += 1
    assert nonzero_grad_params > 0
    # Specifically verify STE flows gradients through FSQ projections back to wte
    assert s1.wte.weight.grad.abs().sum().item() > 0.0
    assert s1.quantizer.fsq_modules[0].proj_in.weight.grad.abs().sum().item() > 0.0
    assert s1.quantizer.fsq_modules[-1].proj_out.weight.grad.abs().sum().item() > 0.0

    # 4. Nested dropout (`fine_drop`) works in training mode and is disabled in eval mode
    cfg_fd = get_preset("tiny_test", fine_drop_prob=1.0, fine_drop_min_keep=1)
    q_fd = MultiscaleRFSQ(cfg_fd)
    q_fd.train()
    z_in = torch.randn(2, cfg_fd.chunk_size, cfg_fd.d_vq)
    cutoffs = set()
    saw_truncated_diff = False
    for _ in range(25):
        q_out = q_fd(z_in, apply_fine_drop=True)
        c_lvl = int(q_out["cutoff_level"])
        cutoffs.add(c_lvl)
        if c_lvl < cfg_fd.num_levels:
            if not torch.allclose(q_out["z_hat"], q_out["z_hat_full"]):
                saw_truncated_diff = True
    assert any(c < cfg_fd.num_levels for c in cutoffs), f"Nested dropout never truncated: {cutoffs}"
    assert saw_truncated_diff, "Truncated z_hat unexpectedly equaled full z_hat"

    q_fd.eval()
    eval_out = q_fd(z_in, apply_fine_drop=True)
    assert eval_out["cutoff_level"] == cfg_fd.num_levels
    assert torch.allclose(eval_out["z_hat"], eval_out["z_hat_full"])

    # 5. encode_codes / decode_codes shape, dtype & partial-depth decoding
    s1.eval()
    with torch.no_grad():
        codes = s1.encode_codes(x)
        assert isinstance(codes, list) and len(codes) == cfg.num_levels
        for k, c_k in enumerate(codes):
            assert c_k.shape == (4, cfg.levels[k], cfg.d_q)
            assert c_k.dtype == torch.long
            assert int(c_k.min().item()) >= 0
            for j, lvl in enumerate(cfg.fsq_levels):
                assert int(c_k[..., j].max().item()) < lvl

        dec_full = s1.decode_codes(codes)
        assert dec_full.shape == (4, cfg.chunk_size, cfg.vocab_size)
        assert dec_full.dtype == torch.float32
        assert torch.isfinite(dec_full).all().item()

        # Partial depth (early exit at level 2)
        dec_partial = s1.decode_codes(codes[:2])
        assert dec_partial.shape == (4, cfg.chunk_size, cfg.vocab_size)
        assert torch.isfinite(dec_partial).all().item()


# ============================================================================
# 3. 2D RoPE & 4D Mask Invariants
# ============================================================================


@pytest.mark.parametrize("num_heads", [4, 8])
def test_2d_rope_when_seq_len_equals_num_heads(num_heads: int) -> None:
    """Explicitly test 2D position_ids [B, S] where S == num_attention_heads (no silent axis-swap bug)."""
    set_seed(7)
    cfg = get_preset(
        "tiny_test",
        hidden_size=num_heads * 16,
        num_attention_heads=num_heads,
        num_key_value_heads=2,
    )
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).eval()

    bsz = 3
    seq_len = num_heads  # CRITICAL: S == num_attention_heads!
    input_ids = torch.randint(16, cfg.vocab_size, (bsz, seq_len), dtype=torch.long)

    # Distinct per-batch 2D position_ids (non-identical offsets so axis bugs immediately surface)
    pos_2d = torch.tensor(
        [
            [0 + i for i in range(seq_len)],
            [17 + i * 2 for i in range(seq_len)],
            [43 + i for i in range(seq_len)],
        ],
        dtype=torch.long,
    )
    assert pos_2d.shape == (bsz, seq_len)

    with torch.no_grad():
        batched_logits = model.forward_exact(input_ids, position_ids=pos_2d)["logits"]
        assert torch.isfinite(batched_logits).all().item()

        # Compare each batch element against standalone 1D position_ids [S]
        for b in range(bsz):
            single_logits = model.forward_exact(
                input_ids[b : b + 1], position_ids=pos_2d[b]
            )["logits"]
            max_diff = (batched_logits[b : b + 1] - single_logits).abs().max().item()
            assert max_diff < 1e-5, f"2D RoPE mismatch for H={num_heads}, S={seq_len}, b={b}: {max_diff}"

        # Also verify explicit position_ids overrides start_pos (past_len)
        prefill = model.forward_exact(input_ids[:, :2], use_cache=True)
        override_pos = torch.arange(100, 100 + seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        default_out = model.forward_exact(
            input_ids, past_key_values=prefill["past_key_values"], use_cache=False
        )["logits"]
        override_out = model.forward_exact(
            input_ids,
            past_key_values=prefill["past_key_values"],
            position_ids=override_pos,
            use_cache=False,
        )["logits"]
        assert (default_out - override_out).abs().max().item() > 1e-3, (
            "Explicit position_ids did not override past_len/start_pos!"
        )


@pytest.mark.parametrize("prefix_len,chunk_len", [(7, 16), (20, 4), (10, 10), (5, 4)])
def test_bottom_right_causal_chunk_commit_equivalence(prefix_len: int, chunk_len: int) -> None:
    """Verify 1-shot forward_exact([prefix, chunk]) matches 2-step cached [prefix] -> [chunk] to <1e-4."""
    set_seed(11)
    cfg = get_preset("tiny_test")
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).eval()

    bsz = 2
    prefix = torch.randint(16, cfg.vocab_size, (bsz, prefix_len), dtype=torch.long)
    chunk = torch.randint(16, cfg.vocab_size, (bsz, chunk_len), dtype=torch.long)
    full = torch.cat([prefix, chunk], dim=1)

    with torch.no_grad():
        # Uncached 1-shot
        one_shot = model.forward_exact(full, use_cache=False)["logits"]
        one_shot_chunk = one_shot[:, prefix_len:, :]

        # 2-step cached commit
        step1 = model.forward_exact(prefix, use_cache=True)
        step2 = model.forward_exact(
            chunk, past_key_values=step1["past_key_values"], use_cache=True
        )
        max_diff = (one_shot_chunk - step2["logits"]).abs().max().item()
        assert max_diff < 1e-4, (
            f"Bottom-right causal commit mismatch for P={prefix_len}, Q={chunk_len}: max_diff={max_diff}"
        )

        # Also test with left-padded attention_mask + all-pad NaN safety row
        prefix_mask = torch.ones((bsz, prefix_len), dtype=torch.long)
        prefix_mask[0, :2] = 0
        chunk_mask = torch.ones((bsz, chunk_len), dtype=torch.long)
        full_mask = torch.cat([prefix_mask, chunk_mask], dim=1)

        one_shot_masked = model.forward_exact(full, attention_mask=full_mask)["logits"][:, prefix_len:, :]
        s1_m = model.forward_exact(prefix, attention_mask=prefix_mask, use_cache=True)
        s2_m = model.forward_exact(
            chunk, attention_mask=full_mask, past_key_values=s1_m["past_key_values"], use_cache=True
        )["logits"]
        max_diff_masked = (one_shot_masked - s2_m).abs().max().item()
        assert max_diff_masked < 1e-4, f"Masked chunk commit mismatch: {max_diff_masked}"

        # All-zero padding row must NEVER produce NaNs
        all_zero_mask = torch.zeros((1, prefix_len), dtype=torch.long)
        nan_check = model.forward_exact(prefix[:1], attention_mask=all_zero_mask)["logits"]
        assert torch.isfinite(nan_check).all().item()


def test_cached_generation_with_delta_prefix_mask() -> None:
    """Regression test: generate_exact & generate_concept_chunk with past_kv (len P) + prefix_mask (len Q)
    must run cleanly and match 1-shot full [p1, p2] with full mask!
    """
    set_seed(19)
    cfg = get_preset("tiny_test")
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).eval()

    p1 = torch.randint(16, cfg.vocab_size, (2, 6), dtype=torch.long)
    p2 = torch.randint(16, cfg.vocab_size, (2, 4), dtype=torch.long)
    mask2 = torch.ones((2, 4), dtype=torch.long)
    mask2[0, 0] = 0
    full_p = torch.cat([p1, p2], dim=1)
    full_mask = torch.cat([torch.ones((2, 6), dtype=torch.long), mask2], dim=1)

    with torch.no_grad():
        pre = model.forward_exact(p1, use_cache=True)
        ex_cached = model.generate_exact(
            p2,
            max_new_tokens=4,
            temperature=0.0,
            past_key_values=pre["past_key_values"],
            prefix_mask=mask2,
        )
        ex_full = model.generate_exact(
            full_p,
            max_new_tokens=4,
            temperature=0.0,
            prefix_mask=full_mask,
        )
        assert ex_cached["token_ids"].shape == (2, 4)
        assert torch.equal(ex_cached["token_ids"], ex_full["token_ids"])

        cc_cached = model.generate_concept_chunk(
            p2,
            prefix_kv=pre["past_key_values"],
            prefix_len=6,
            temperature=0.0,
            early_exit_conf=None,
            prefix_mask=mask2,
        )
        cc_full = model.generate_concept_chunk(
            full_p,
            temperature=0.0,
            early_exit_conf=None,
            prefix_mask=full_mask,
        )
        assert cc_cached["token_ids"].shape == (2, cfg.chunk_size)
        assert cc_cached["prefix_len"] == 10
        assert torch.equal(cc_cached["token_ids"], cc_full["token_ids"])


# ============================================================================
# 4. Dual-Mode Generation & Lossless Self-Speculative Decoding
# ============================================================================


def test_dual_mode_generation_early_exit_int8_kv_and_self_speculative() -> None:
    """Verify generate_concept_chunk (early exit), generate_block_ar (FP & INT8 KV),
    generate_exact (logit_mask_fn), and lossless greedy equivalence of generate_self_speculative.
    """
    set_seed(23)
    cfg = get_preset("tiny_test")
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).eval()

    prefix_b1 = torch.randint(16, cfg.vocab_size, (1, 10), dtype=torch.long)
    prefix_b2 = torch.randint(16, cfg.vocab_size, (2, 10), dtype=torch.long)

    with torch.no_grad():
        # 4a. generate_concept_chunk: full depth vs early exit
        full_chunk = model.generate_concept_chunk(
            prefix_b2, temperature=0.0, early_exit_conf=None, return_codes=True
        )
        assert full_chunk["token_ids"].shape == (2, cfg.chunk_size)
        assert full_chunk["exit_level"] == cfg.num_levels
        assert full_chunk["codes"] is not None and len(full_chunk["codes"]) == cfg.num_levels
        assert full_chunk["prefix_len"] == 10

        # Low early_exit_conf threshold must trigger early exit (< num_levels)
        ee_chunk = model.generate_concept_chunk(
            prefix_b2, temperature=0.0, early_exit_conf=0.05, return_codes=True
        )
        assert ee_chunk["token_ids"].shape == (2, cfg.chunk_size)
        assert 2 <= ee_chunk["exit_level"] < cfg.num_levels
        assert ee_chunk["codes"] is not None and len(ee_chunk["codes"]) == ee_chunk["exit_level"]

        # 4b. generate_block_ar with FP32 KV vs INT8 QuantizedKV
        max_new = cfg.chunk_size * 2 + 5  # forces >= 3 chunks
        bar_fp = model.generate_block_ar(
            prefix_b1,
            max_new_tokens=max_new,
            temperature=0.0,
            early_exit_conf=None,
            quantize_kv=False,
            stop_token_ids=[],
        )
        bar_q = model.generate_block_ar(
            prefix_b1,
            max_new_tokens=max_new,
            temperature=0.0,
            early_exit_conf=None,
            quantize_kv=True,
            stop_token_ids=[],
        )
        assert bar_fp["token_ids"].shape == (1, max_new)
        assert bar_q["token_ids"].shape == (1, max_new)
        assert bar_fp["chunks"] >= 3 and bar_q["chunks"] >= 3
        assert isinstance(bar_q["prefix_kv"][0], QuantizedKV)
        fp_kv_bytes = kv_cache_nbytes(bar_fp["prefix_kv"])
        q_kv_bytes = kv_cache_nbytes(bar_q["prefix_kv"])
        assert fp_kv_bytes / max(1, q_kv_bytes) > 2.5, (
            f"INT8 KV cache did not shrink meaningfully: FP={fp_kv_bytes} vs INT8={q_kv_bytes}"
        )

        # 4c. generate_exact with custom logit_mask_fn
        allowed_set = {42, 43, 44}

        def only_allowed(_gen_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            masked = torch.full_like(logits, float("-inf"))
            for tid in allowed_set:
                masked[..., tid] = logits[..., tid]
            return masked

        ex_masked = model.generate_exact(
            prefix_b2,
            max_new_tokens=12,
            temperature=0.0,
            logit_mask_fn=only_allowed,
        )
        assert ex_masked["token_ids"].shape == (2, 12)
        assert set(ex_masked["token_ids"].reshape(-1).tolist()).issubset(allowed_set)

        # 4d. generate_self_speculative: LOSSLESS GREEDY EQUIVALENCE to generate_exact!
        for pfx in (prefix_b1, prefix_b2):
            exact_ref = model.generate_exact(
                pfx, max_new_tokens=20, temperature=0.0, quantize_kv=False
            )["token_ids"]
            spec_res = model.generate_self_speculative(
                pfx, max_new_tokens=20, temperature=0.0, quantize_kv=False
            )
            assert torch.equal(spec_res["token_ids"], exact_ref), (
                f"Self-speculative greedy output diverged from generate_exact for B={pfx.shape[0]}!"
            )
            assert 0.0 <= spec_res["acceptance_rate"] <= 1.0
            assert spec_res["concept_steps"] > 0 and spec_res["verify_passes"] > 0

        # Stochastic self-speculative + INT8 KV smoke check
        spec_stoch = model.generate_self_speculative(
            prefix_b1, max_new_tokens=12, temperature=0.8, quantize_kv=True
        )
        assert spec_stoch["token_ids"].shape == (1, 12)


# ============================================================================
# 5. End-to-End Training & Checkpoint Smoke Tests
# ============================================================================


def test_training_stage1_stage2_grpo_and_checkpoint_dtypes(tmp_path: Path) -> None:
    """Run real Stage 1, Stage 2 (joint/sft/QAT/self-distill), GRPO/CISPO, and checkpoint dtype round-trip."""
    cfg = get_preset("tiny_test")
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    # 5a. Datasets & WarmupCosineScheduler invariants
    corpus = make_synthetic_corpus(num_samples=48, seed=42)
    s1_ds = PackedChunkDataset(corpus, tok, chunk_size=cfg.chunk_size)
    assert s1_ds.pad_ratio == 0.0
    assert s1_ds[0]["input_ids"].shape == (cfg.chunk_size,)

    sft_raw = make_synthetic_sft(num_samples=24, seed=42)
    s2_ds = BlockARSFTDataset(sft_raw, tok, chunk_size=cfg.chunk_size, max_prefix_len=96, fixed_prefix_len=False)
    collated = collate_block_ar([s2_ds[0], s2_ds[1], s2_ds[2]], pad_token_id=cfg.pad_token_id)
    assert collated["prefix_ids"].shape == collated["prefix_mask"].shape
    assert collated["target_ids"].shape == (3, cfg.chunk_size)
    # Prefix labels must be masked with -100
    p_len = collated["prefix_ids"].shape[1]
    assert (collated["exact_labels"][:, :p_len] == -100).all().item()

    dummy_param = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.AdamW([dummy_param], lr=1e-3)
    sched = WarmupCosineScheduler(opt, total_steps=20, warmup_steps=4, min_lr_ratio=0.1)
    lrs = [sched.get_last_lr()[0]]
    for _ in range(19):
        lrs.append(sched.step()[0])
    assert lrs[0] > 0.0 and lrs[3] == pytest.approx(1e-3, rel=1e-5)
    assert lrs[-1] < lrs[3] and lrs[-1] >= 1e-4 - 1e-9

    # 5b. Stage 1 training smoke test
    s1_ckpt = tmp_path / "stage1.pt"
    s1_res = train_stage1(
        cfg=cfg,
        tokenizer=tok,
        dataset=s1_ds,
        steps=10,
        batch_size=8,
        lr=4e-3,
        device="cpu",
        seed=42,
        save_path=s1_ckpt,
        verbose=False,
    )
    s1_model: CNCMConceptTokenizer = s1_res["model"]
    s1_m = s1_res["metrics"]
    assert s1_m["all_grads_finite"] is True
    assert s1_m["final_loss"] < s1_m["initial_loss"], (
        f"Stage 1 loss did not decrease: {s1_m['initial_loss']:.4f} -> {s1_m['final_loss']:.4f}"
    )
    assert s1_m["final_active_code_ratio"] > 0.0

    # Checkpoint round-trip must preserve EXACT FP32 & INT64 dtypes (no .half() corruption!)
    s1_reloaded = CNCMConceptTokenizer(cfg)
    payload = load_checkpoint(s1_ckpt, model=s1_reloaded, map_location="cpu")
    assert payload["step"] == 10
    saw_fp32 = False
    saw_int64 = False
    for name, orig_t in s1_model.state_dict().items():
        loaded_t = s1_reloaded.state_dict()[name]
        assert loaded_t.dtype == orig_t.dtype, f"Dtype changed on {name}: {orig_t.dtype} -> {loaded_t.dtype}"
        assert loaded_t.dtype != torch.float16, f"Unexpected FP16/.half() corruption on {name}"
        assert torch.equal(loaded_t, orig_t), f"Tensor values differed after checkpoint reload on {name}"
        if loaded_t.dtype == torch.float32:
            saw_fp32 = True
        if loaded_t.dtype == torch.int64:
            saw_int64 = True
    assert saw_fp32 and saw_int64, "Expected both FP32 parameters and INT64 FSQ buffers in Stage 1 ckpt"

    # Snapshot Stage 1 weights to prove Stage 2 & GRPO NEVER mutate them
    s1_snapshot = {k: v.detach().clone() for k, v in s1_model.state_dict().items()}

    # 5c. Stage 2 joint training smoke test
    s2_ckpt = tmp_path / "stage2.pt"
    s2_res = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1_model,
        dataset=s2_ds,
        mode="joint",
        steps=10,
        corrupt_rate=0.5,
        batch_size=4,
        lr=3e-3,
        device="cpu",
        seed=42,
        save_path=s2_ckpt,
        verbose=False,
    )
    model: CNCMModel = s2_res["model"]
    s2_m = s2_res["metrics"]
    assert s2_m["all_grads_finite"] is True
    assert s2_m["concept_tokenizer_frozen"] is True
    assert s2_m["final_loss"] < s2_m["initial_loss"], (
        f"Stage 2 joint loss did not decrease: {s2_m['initial_loss']:.4f} -> {s2_m['final_loss']:.4f}"
    )
    for k, v in model.concept_tokenizer.state_dict().items():
        assert torch.equal(v, s1_snapshot[k]), f"Stage 1 parameter/buffer {k} mutated during Stage 2!"

    # Stage 2 checkpoint round-trip preserves exact FP32/INT64 dtypes
    model_reloaded = CNCMModel(cfg, CNCMConceptTokenizer(cfg))
    load_checkpoint(s2_ckpt, model=model_reloaded, map_location="cpu")
    for k, t_orig in model.state_dict().items():
        t_load = model_reloaded.state_dict()[k]
        assert t_load.dtype == t_orig.dtype and torch.equal(t_load, t_orig)

    # Stage 2 SFT + self_distill + qat_finetune smoke check
    s2_qat = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1_model,
        dataset=s2_ds,
        mode="sft",
        steps=2,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        self_distill=True,
        qat_finetune=True,
        qat_bits=4,
        seed=42,
        verbose=False,
    )
    assert s2_qat["metrics"]["all_grads_finite"] is True

    # 5d. Exact 1-pass compute_code_logprobs + GRPO & CISPO smoke test
    model.eval()
    pfx = torch.randint(16, cfg.vocab_size, (3, 8), dtype=torch.long)
    pfx_mask = torch.ones_like(pfx)
    with torch.no_grad():
        gen_c = model.generate_concept_chunk(pfx, temperature=0.8, early_exit_conf=None, return_codes=True)
        lp_mean = model.compute_code_logprobs(pfx, gen_c["codes"], prefix_mask=pfx_mask, reduction="mean")
        lp_sum = model.compute_code_logprobs(pfx, gen_c["codes"], prefix_mask=pfx_mask, reduction="sum")
        fc = model.forward_concept(pfx, gen_c["codes"], prefix_mask=pfx_mask, corrupt_rate=0.0)
        assert lp_mean.shape == (3,) and lp_sum.shape == (3,)
        assert torch.isfinite(lp_mean).all().item() and torch.isfinite(lp_sum).all().item()
        assert torch.allclose(lp_mean, fc["log_probs"].mean(dim=-1), atol=1e-5)
        assert torch.allclose(lp_sum, fc["log_probs"].sum(dim=-1), atol=1e-4)

    # Verifiable reward sanity
    good_completion = (
        '<think>Compute 6 * 7 = 42.</think>'
        '<tool_call>{"name":"calculator","arguments":{"expression":"6 * 7"}}</tool_call> 42'
    )
    bad_completion = "random gibberish"
    spec = {
        "expected_answer": "42",
        "expected_tool": "calculator",
        "expected_args": {"expression": "6 * 7"},
        "require_think": True,
        "require_tool_call": True,
    }
    assert compute_verifiable_reward(good_completion, spec) > compute_verifiable_reward(bad_completion, spec) + 1.0

    # Real GRPO run (2 steps)
    prompts_raw = make_synthetic_grpo_prompts(num_samples=8, seed=42)
    gds = GRPOPromptDataset(prompts_raw, tok)
    _ = collate_grpo_prompts([gds[0], gds[1]], pad_token_id=cfg.pad_token_id)

    grpo_res = train_grpo(
        cfg=cfg,
        tokenizer=tok,
        model=model,
        prompts=gds,
        steps=2,
        num_generations=4,
        batch_size=2,
        lr=5e-4,
        device="cpu",
        seed=42,
        verbose=False,
    )
    gm = grpo_res["metrics"]
    assert gm["all_logprobs_finite"] is True
    assert gm["all_grads_finite"] is True
    assert gm["policy_updated"] is True and gm["policy_param_delta_l2"] > 0.0
    assert gm["concept_tokenizer_frozen"] is True
    for k, v in model.concept_tokenizer.state_dict().items():
        assert torch.equal(v, s1_snapshot[k]), f"Stage 1 mutated during GRPO on {k}!"

    # Also verify CISPO mode runs cleanly
    cispo_res = train_grpo(
        cfg=cfg,
        tokenizer=tok,
        model=model,
        prompts=gds,
        steps=1,
        num_generations=4,
        batch_size=1,
        lr=5e-4,
        use_cispo=True,
        device="cpu",
        seed=99,
        verbose=False,
    )
    assert cispo_res["metrics"]["algorithm"] == "cispo"
    assert cispo_res["metrics"]["all_logprobs_finite"] is True
    assert cispo_res["metrics"]["all_grads_finite"] is True


# ============================================================================
# 6. Agentic AI, Edge Quantization (INT4 / INT8), ONNX Wrappers & CLI Demo
# ============================================================================


def test_tools_grammar_and_untrained_agent_end_to_end() -> None:
    """Verify safe tools, JSONSchemaConstrainedMasker, and CNCMAgent on an UNTRAINED tiny_test model."""
    set_seed(77)
    # 6a. Built-in tools + security checks
    assert safe_eval_math("12 * 7 + 5") == 89
    assert safe_eval_math("sqrt(144) + pow(2, 3)") == 20
    with pytest.raises(ValueError):
        safe_eval_math("__import__('os').system('echo pwn')")
    with pytest.raises(ZeroDivisionError):
        safe_eval_math("10 / 0")

    reg = create_default_registry()
    assert set(reg.list_names()) == {"calculator", "kv_memory", "text_stats", "clock_utc"}
    assert reg.execute("calculator", {"expression": "6 * 7"}).output["value"] == 42
    assert reg.execute("kv_memory", {"action": "set", "key": "k1", "value": "v1"}).ok is True
    assert reg.execute("kv_memory", {"action": "get", "key": "k1"}).output["value"] == "v1"
    assert reg.execute("text_stats", {"text": "hello cncm world"}).output["words"] == 3
    assert reg.execute("clock_utc", {"format": "iso"}).ok is True
    assert reg.execute("nonexistent", {}).ok is False

    # 6b. Untrained tiny_test CNCMAgent + JSONSchemaConstrainedMasker -> valid JSON & actual calculator execution!
    cfg = get_preset("tiny_test")
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).eval()

    agent = CNCMAgent(
        model=model,
        tokenizer=tok,
        registry=create_default_registry(),
        max_turns=2,
        use_grammar_constraints=True,
        quantize_kv=True,
        early_exit_conf=0.92,
    )
    trace = agent.run("Calculate 12 * 7 + 5 using the calculator.", force_tool="calculator")
    assert len(trace.tool_calls) == 1
    assert len(trace.tool_results) == 1
    tc = trace.tool_calls[0]
    tr = trace.tool_results[0]
    assert tc["name"] == "calculator"
    assert isinstance(tc["arguments"], dict) and "expression" in tc["arguments"]
    assert tr.ok is True, f"Calculator execution failed on untrained model: {tr}"
    assert isinstance(tr.output["value"], (int, float))
    assert trace.concept_passes > 0 and trace.exact_passes > 0
    assert any(sw["from"] == "concept" and sw["to"] == "exact" for sw in trace.mode_switches)

    # All 4 built-in tools succeed on untrained model under JSONSchemaConstrainedMasker
    for tname in ("calculator", "kv_memory", "text_stats", "clock_utc"):
        tr_tool = agent.run(f"Run {tname}", force_tool=tname)
        assert len(tr_tool.tool_results) == 1
        assert tr_tool.tool_calls[0]["name"] == tname
        assert tr_tool.tool_results[0].ok is True, f"Tool {tname} failed: {tr_tool.tool_results[0]}"

    # 6c. Regression test: JSONSchemaConstrainedMasker with prefix_len when prefix has an OLD <tool_call>
    old_prefix = tok.encode(
        '<tool_call>{"name":"clock_utc","arguments":{"format":"iso"}}</tool_call>'
        '<tool_response>{"ok":true}</tool_response>'
    )
    masker = JSONSchemaConstrainedMasker(
        tok, reg, allowed_tools=["calculator"], prefix_len=len(old_prefix)
    )
    out_ex = model.generate_exact(
        torch.tensor([old_prefix], dtype=torch.long),
        max_new_tokens=64,
        temperature=0.0,
        stop_token_ids=[tok.tool_call_end_id],
        logit_mask_fn=masker,
    )
    decoded = tok.decode(out_ex["token_ids"][0].tolist(), skip_special_tokens=True)
    parsed = extract_json_object(decoded)
    assert parsed is not None and parsed.get("name") == "calculator", (
        f"Masker failed when prior <tool_call> existed in prefix: decoded={decoded!r}"
    )


def test_edge_quantization_int4_int8_and_onnx_wrappers(tmp_path: Path) -> None:
    """Verify real >4x Linear byte reduction for INT4, bounded quant error, and ONNX wrappers."""
    set_seed(31)
    cfg = get_preset("tiny_test")
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).eval()

    sz_fp = measure_model_size(model)
    m_int8 = quantize_model_int8(model, quantize_embeddings=True).eval()
    m_int4 = quantize_model_int4(model, group_size=32, quantize_embeddings=True).eval()
    sz_int8 = measure_model_size(m_int8)
    sz_int4 = measure_model_size(m_int4)

    # Real packed tensor storage verification
    assert sz_int4["num_int4_layers"] > 0
    assert sz_int8["num_int8_layers"] > 0
    assert sz_int4["num_int8_embeddings"] > 0
    for mod in m_int4.modules():
        if isinstance(mod, Int4PackedLinear):
            assert mod.qweight.dtype == torch.uint8
            assert mod.scales.dtype == torch.float16
            assert mod.zeros.dtype == torch.float16
        elif isinstance(mod, Int8Linear):
            assert mod.qweight.dtype == torch.int8
        elif isinstance(mod, Int8Embedding):
            assert mod.qweight.dtype == torch.int8

    # >4x Linear byte reduction on INT4 vs FP32, and >3.2x on INT8 vs FP32
    int4_lin_ratio = sz_fp["linear_bytes"] / sz_int4["linear_bytes"]
    int8_lin_ratio = sz_fp["linear_bytes"] / sz_int8["linear_bytes"]
    assert int4_lin_ratio > 4.0, f"INT4 Linear compression ratio too small: {int4_lin_ratio:.2f}x"
    assert int8_lin_ratio > 3.2, f"INT8 Linear compression ratio too small: {int8_lin_ratio:.2f}x"
    assert sz_int4["total_bytes"] < sz_int8["total_bytes"] < sz_fp["total_bytes"]

    # FSQ projections and RMSNorms stay in FP32
    for fsq in m_int4.concept_tokenizer.quantizer.fsq_modules:
        assert isinstance(fsq.proj_in, nn.Linear) and fsq.proj_in.weight.dtype == torch.float32
        assert isinstance(fsq.proj_out, nn.Linear) and fsq.proj_out.weight.dtype == torch.float32

    # Forward passes run cleanly on quantized models with bounded error vs FP32
    x = torch.randint(16, cfg.vocab_size, (2, 12), dtype=torch.long)
    tgt = torch.randint(16, cfg.vocab_size, (2, cfg.chunk_size), dtype=torch.long)
    with torch.no_grad():
        fp_logits = model.forward_exact(x)["logits"]
        q8_logits = m_int8.forward_exact(x)["logits"]
        q4_logits = m_int4.forward_exact(x)["logits"]

        assert torch.isfinite(q8_logits).all().item()
        assert torch.isfinite(q4_logits).all().item()

        mae_int8 = (fp_logits - q8_logits).abs().mean().item()
        mae_int4 = (fp_logits - q4_logits).abs().mean().item()
        assert mae_int8 < 0.05, f"INT8 logit MAE unexpectedly large: {mae_int8}"
        assert mae_int4 < 0.25, f"INT4 logit MAE unexpectedly large: {mae_int4}"
        assert mae_int8 < mae_int4

        # Concept Mode & Dual-Mode Agent on packed INT4 model
        codes = m_int4.concept_tokenizer.encode_codes(tgt)
        c_out = m_int4.forward_concept(x, codes)
        assert torch.isfinite(c_out["loss"]).item()
        g_out = m_int4.generate_concept_chunk(x, temperature=0.0, quantize_kv=True)
        assert g_out["token_ids"].shape == (2, cfg.chunk_size)

    agent_q4 = CNCMAgent(m_int4, tok, create_default_registry(), max_turns=2, use_grammar_constraints=True)
    tr_q4 = agent_q4.run("Compute 5 + 9", force_tool="calculator")
    assert len(tr_q4.tool_results) == 1 and tr_q4.tool_results[0].ok is True

    # ONNX export wrapper verification (graceful even when onnx is absent)
    onnx_report = export_cncm_onnx(model, tmp_path / "onnx_out", opset=17, batch_size=2, prefix_len=8)
    assert onnx_report["wrappers_verified"] is True
    assert Path(onnx_report["manifest"]).exists()
    assert set(onnx_report["wrapper_shapes"].keys()) == {
        "stage1_decoder",
        "stage2_exact_step",
        "stage2_concept_step",
    }


def test_demo_cli_end_to_end() -> None:
    """Run `python demo_cncm.py` end-to-end and assert exit code 0."""
    demo_path = REPO_ROOT / "demo_cncm.py"
    assert demo_path.exists()
    proc = subprocess.run(
        [sys.executable, str(demo_path), "--stage1-steps", "4", "--stage2-steps", "4", "--grpo-steps", "2"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"demo_cncm.py failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "CNCM demo completed successfully." in proc.stdout


# ============================================================================
# 7. PG-FIM (Plan-Guided Fill-in-the-Middle) Verification Suite (Tests 16–19)
# ============================================================================


def test_pg_fim_special_tokens_symbol_table_and_sandbox() -> None:
    """Test 16: PG-FIM 11 special tokens, format_fim_prompt, CodeSymbolTable, and run_code_sandbox."""
    import random
    from cncm.code import (
        CodeSymbolTable,
        check_syntax_and_keywords,
        extract_ast_or_line_spans,
        extract_oracle_keywords,
        extract_symbols,
        run_code_sandbox,
    )

    cfg = get_preset("tiny_test")
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    # 16a. 11 PG-FIM special tokens at [vocab_size - 11 .. vocab_size - 1] synced between config & tokenizer
    fim_attrs = (
        ("fim_prefix_id", "<fim_prefix>"),
        ("fim_suffix_id", "<fim_suffix>"),
        ("fim_middle_id", "<fim_middle>"),
        ("fim_end_id", "<fim_end>"),
        ("kw_start_id", "<kw>"),
        ("kw_end_id", "</kw>"),
        ("kw_empty_id", "<kw/>"),
        ("mlen_1_id", "<mlen_1>"),
        ("mlen_2_id", "<mlen_2>"),
        ("mlen_3_id", "<mlen_3>"),
        ("mlen_4plus_id", "<mlen_4+>"),
    )
    fim_base = cfg.vocab_size - 11
    for offset, (attr, tag_str) in enumerate(fim_attrs):
        expected_id = fim_base + offset
        assert getattr(cfg, attr) == expected_id
        assert getattr(tok, attr) == expected_id
        assert tok.encode(tag_str) == [expected_id]
        assert tok.decode([expected_id]) == tag_str

    assert cfg.mlen_token_ids == tok.mlen_token_ids == (
        cfg.mlen_1_id,
        cfg.mlen_2_id,
        cfg.mlen_3_id,
        cfg.mlen_4plus_id,
    )
    for k, expected_tid in (
        (1, cfg.mlen_1_id),
        (2, cfg.mlen_2_id),
        (3, cfg.mlen_3_id),
        (4, cfg.mlen_4plus_id),
        (7, cfg.mlen_4plus_id),
    ):
        assert cfg.mlen_id_for_chunks(k) == tok.mlen_id_for_chunks(k) == expected_tid
        assert cfg.chunks_for_mlen_id(expected_tid) == tok.chunks_for_mlen_id(expected_tid) == min(k, 4)

    # format_fim_prompt with context budgets (retains tokens nearest the cursor)
    p_code = "def accumulate(items, start=0):\n    total = start\n"
    s_code = "\n    return total\n# trailing comment far from cursor"
    psm_str, psm_ids = tok.format_fim_prompt(p_code, s_code, prefix_budget=6, suffix_budget=5)
    assert psm_ids[0] == tok.fim_prefix_id
    assert tok.fim_suffix_id in psm_ids
    assert psm_ids[-1] == tok.fim_middle_id
    assert len(psm_ids) <= 1 + 6 + 1 + 5 + 1
    assert psm_str.startswith("<fim_prefix>") and psm_str.endswith("<fim_middle>")

    # 16b. CodeSymbolTable: extract_symbols, extract_ast_or_line_spans, extract_oracle_keywords, check_syntax_and_keywords
    sample_code = (
        "def compute_weighted_sum(values, weights):\n"
        "    acc = 0\n"
        "    for v, w in zip(values, weights):\n"
        "        acc += v * w\n"
        "    return acc\n"
    )
    table = CodeSymbolTable()
    sym = table.extract_symbols(
        prefix="def compute_weighted_sum(values, weights):\n    acc = 0\n",
        suffix="\n    return acc\n",
    )
    assert {"compute_weighted_sum", "values", "weights", "acc"}.issubset(sym.defined_identifiers)
    assert "acc" in sym.used_identifiers
    assert sym.is_known("values") and sym.is_known("zip") and sym.is_known("return")
    assert not sym.is_known("completely_unknown_xyz")
    assert extract_symbols("x = 1", "y = x").all_identifiers == {"x", "y"}

    rng = random.Random(42)
    for mode in ("mix", "ast", "function_body", "line", "single_line", "multi_line", "char"):
        p_span, m_span, s_span = extract_ast_or_line_spans(sample_code, mode=mode, rng=rng)
        assert p_span + m_span + s_span == sample_code, f"Span split corrupted code in mode={mode}"
        assert len(m_span) > 0

    kws_code = extract_oracle_keywords(
        middle="    for v, w in zip(values, weights):\n        acc += v * w",
        prefix="def compute_weighted_sum(values, weights):\n    acc = 0\n",
        suffix="\n    return acc\n",
        max_kw=3,
        is_code=True,
    )
    assert 1 <= len(kws_code) <= 3
    assert any(k in ("values", "weights", "acc", "v", "w", "for") for k in kws_code)

    kws_prose = extract_oracle_keywords(
        middle="Multiscale residual quantization eliminates 151 megabytes of codebook tables.",
        max_kw=3,
        is_code=False,
    )
    assert 1 <= len(kws_prose) <= 3

    chk_ok = check_syntax_and_keywords(
        prefix="def add_pair(a, b):\n",
        middle="    res = a + b\n",
        suffix="    return res\n",
        keywords=["res", "a"],
    )
    assert chk_ok["parses"] is True
    assert chk_ok["keywords_present"] is True
    assert chk_ok["missing_keywords"] == []
    assert chk_ok["undefined_identifiers"] == []
    assert chk_ok["non_trivial"] is True

    chk_triv = check_syntax_and_keywords(
        prefix="def add_pair(a, b):\n",
        middle="    pass\n",
        suffix="    return a\n",
        keywords=["b"],
    )
    assert chk_triv["parses"] is True
    assert chk_triv["non_trivial"] is False
    assert chk_triv["keywords_present"] is False
    assert chk_triv["missing_keywords"] == ["b"]

    chk_undef = check_syntax_and_keywords(
        prefix="def add_pair(a, b):\n",
        middle="    return GhostVar + a\n",
        suffix="",
        keywords=["a"],
    )
    assert chk_undef["parses"] is True
    assert "GhostVar" in chk_undef["undefined_identifiers"]

    # 16c. Safe execution sandbox (run_code_sandbox)
    sb_pass = run_code_sandbox(
        "def square_sum(a, b):\n    return a * a + b * b\n",
        test_code="assert square_sum(3, 4) == 25",
    )
    assert sb_pass["parses"] is True
    assert sb_pass["compiles"] is True
    assert sb_pass["tests_passed"] is True
    assert sb_pass["non_trivial"] is True

    sb_trivial = run_code_sandbox(
        "def f(x):\n    pass\n",
        middle="    pass\n",
    )
    assert sb_trivial["parses"] is True
    assert sb_trivial["non_trivial"] is False

    # Dangerous imports / calls must be blocked at AST security stage
    for dangerous_code in (
        "import os\nos.system('echo hi')",
        "import subprocess\nsubprocess.run(['cmd'])",
        "open('hacked.txt', 'w').write('x')",
        "eval('1 + 1')",
    ):
        sb_sec = run_code_sandbox(dangerous_code)
        assert sb_sec["compiles"] is False and sb_sec["tests_passed"] is False
        assert sb_sec["error"] is not None and "SandboxSecurityError" in sb_sec["error"]

    # Infinite loop must be aborted rapidly by tick/timeout guard
    sb_inf = run_code_sandbox("x = 0\nwhile True:\n    x += 1\n", timeout_sec=0.25, max_ticks=5_000)
    assert sb_inf["parses"] is True and sb_inf["compiles"] is True
    assert sb_inf["tests_passed"] is False
    assert sb_inf["error"] is not None and "SandboxTimeoutError" in sb_inf["error"]


def test_keyword_plan_masker_and_parse_plan_tokens() -> None:
    """Test 17: KeywordPlanMasker hard format state machine, soft symbol bias, and parse_plan_tokens."""
    from cncm.agent import KeywordPlanMasker, parse_plan_tokens

    cfg = get_preset("tiny_test")
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    prefix_code = "def compute_total(price_list, tax_rate):\n    subtotal = sum(price_list)\n"
    suffix_code = "\n    return final_total\n"
    masker = KeywordPlanMasker(
        tokenizer=tok,
        prefix_code=prefix_code,
        suffix_code=suffix_code,
        kw_max=3,
        max_plan_tokens=14,
        symbol_bonus=2.5,
        unknown_id_penalty=1.5,
    )

    zero_logits = torch.zeros((1, cfg.vocab_size), dtype=torch.float32)

    # 17a. Step 0 (right after <fim_middle>): ONLY mlen_token_ids allowed
    m_step0 = masker([tok.fim_middle_id], zero_logits)
    finite_step0 = set(torch.where(torch.isfinite(m_step0[0]))[0].tolist())
    assert finite_step0 == set(tok.mlen_token_ids)

    # 17b. Step 1 (after <mlen_2>): ONLY <kw> or <kw/> allowed
    m_step1 = masker([tok.fim_middle_id, tok.mlen_2_id], zero_logits)
    finite_step1 = set(torch.where(torch.isfinite(m_step1[0]))[0].tolist())
    assert finite_step1 == {tok.kw_start_id, tok.kw_empty_id}

    # If <kw/> was emitted, plan is complete -> only </kw> allowed
    m_done = masker([tok.fim_middle_id, tok.mlen_1_id, tok.kw_empty_id], zero_logits)
    finite_done = set(torch.where(torch.isfinite(m_done[0]))[0].tolist())
    assert finite_done == {tok.kw_end_id}

    # 17c. Inside <kw> ... </kw>: soft symbol bonus (+2.5) vs unknown identifier penalty (-1.5), NEVER -inf!
    m_in_kw = masker([tok.fim_middle_id, tok.mlen_1_id, tok.kw_start_id], zero_logits)
    s_tid = tok.encode("s")[0]  # prefix of known symbol 'subtotal' -> +2.5
    assert torch.isfinite(m_in_kw[0, s_tid]).item()
    assert float(m_in_kw[0, s_tid].item()) == pytest.approx(2.5, abs=1e-5)

    # After '<kw>q', appending 'q' forms 'qq' (not a prefix of any symbol or Python builtin) -> -1.5
    q_tid = tok.encode("q")[0]
    m_in_unknown = masker([tok.fim_middle_id, tok.mlen_1_id, tok.kw_start_id, q_tid], zero_logits)
    assert torch.isfinite(m_in_unknown[0, q_tid]).item(), "Unknown identifier token must NOT be hard-masked to -inf!"
    assert float(m_in_unknown[0, q_tid].item()) == pytest.approx(-1.5, abs=1e-5)

    # 17d. Enforces <= 3 '|'-separated keywords and allows </kw> at valid keyword boundary
    kw3_seq = [tok.fim_middle_id, tok.mlen_3_id, tok.kw_start_id] + tok.encode("subtotal | tax_rate | final_total")
    m_after_3 = masker(kw3_seq, zero_logits)
    assert torch.isfinite(m_after_3[0, tok.kw_end_id]).item()
    # Adding a 4th '|' separator must be hard-blocked (-inf)
    pipe_tid = tok.encode("|")[0]
    assert not torch.isfinite(m_after_3[0, pipe_tid]).item()

    # 17e. parse_plan_tokens on full keyword plan and empty <kw/> plan
    full_plan_ids = kw3_seq + [tok.kw_end_id]
    parsed_full = parse_plan_tokens(tok, full_plan_ids)
    assert parsed_full["mlen_id"] == tok.mlen_3_id
    assert parsed_full["num_chunks"] == 3
    assert parsed_full["keywords"] == ["subtotal", "tax_rate", "final_total"]
    assert parsed_full["is_empty_plan"] is False

    empty_plan_ids = [tok.mlen_1_id, tok.kw_empty_id]
    parsed_empty = parse_plan_tokens(tok, empty_plan_ids)
    assert parsed_empty["mlen_id"] == tok.mlen_1_id
    assert parsed_empty["num_chunks"] == 1
    assert parsed_empty["keywords"] == []
    assert parsed_empty["is_empty_plan"] is True


def test_pg_fim_phase_a_and_phase_b_copy_decoder_and_generate_fim() -> None:
    """Test 18: Phase B CopyAwareKeywordCrossAttention + Phase A/B CNCMModel.generate_fim end-to-end."""
    from cncm import CopyAwareKeywordCrossAttention

    set_seed(53)
    # 18a. Phase B Copy-Aware Stage 1 Decoder (`use_copy_decoder=True`)
    cfg_copy = get_preset("tiny_test", use_copy_decoder=True, copy_decoder_heads=2, copy_kw_drop_prob=0.30)
    tok = CNCMTextTokenizer(vocab_size=cfg_copy.vocab_size)
    s1_copy = CNCMConceptTokenizer(cfg_copy)
    assert isinstance(s1_copy.copy_cross_attn, CopyAwareKeywordCrossAttention)
    # copy_gate must be zero-initialized
    assert float(s1_copy.copy_cross_attn.copy_gate.item()) == 0.0

    x = torch.randint(16, cfg_copy.vocab_size - 11, (4, cfg_copy.chunk_size), dtype=torch.long)

    # Automatic oracle keyword extraction + dropout
    s1_copy.train()
    kw_ids, kw_mask = s1_copy.extract_oracle_kw_token_ids(x, max_kw=6, apply_dropout=False)
    assert kw_ids.shape == (4, 6) and kw_mask.shape == (4, 6)
    assert kw_mask.any().item()

    # Forward + backward with automatic oracle keywords & STE + copy cross-attn finite grads
    out_copy = s1_copy(x, apply_fine_drop=True)
    assert torch.isfinite(out_copy["loss"]).item()
    out_copy["loss"].backward()
    for pname, p in s1_copy.copy_cross_attn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all().item(), f"Bad grad on copy_cross_attn.{pname}"
    assert s1_copy.copy_cross_attn.pointer_logit_scale.grad.abs().sum().item() > 0.0

    # Pointer logit boost verification: passing explicit kw_token_ids boosts logits on those exact token IDs
    s1_copy.eval()
    target_kw_tid = 42
    explicit_kw = torch.tensor([[target_kw_tid, 55]], dtype=torch.long)
    with torch.no_grad():
        codes = s1_copy.encode_codes(x[:1])
        logits_no_kw = s1_copy.decode_codes(codes, kw_token_ids=None)
        logits_with_kw = s1_copy.decode_codes(codes, kw_token_ids=explicit_kw)
        boost_on_target = (logits_with_kw[..., target_kw_tid] - logits_no_kw[..., target_kw_tid]).min().item()
        assert boost_on_target > 0.1, f"Expected positive pointer logit boost on kw token, got {boost_on_target}"

    # 18b. CNCMModel.generate_fim end-to-end (Phase A & Phase B, use_plan=True/False, repair, INT8 KV, INT4 weights)
    model = CNCMModel(cfg_copy, s1_copy).eval()
    prefix_str = "def accumulate_positive(nums):\n    total = 0\n    for n in nums:\n        if n > 0:\n"
    suffix_str = "\n    return total\n"

    for use_plan in (True, False):
        for q_kv in (False, True):
            # Model-only mode (default: allow_template_fallback=False) must report honest provenance
            fim_model_only = model.generate_fim(
                prefix=prefix_str,
                suffix=suffix_str,
                tokenizer=tok,
                max_chunks=2,
                temperature=0.0,
                early_exit_conf=0.92,
                quantize_kv=q_kv,
                use_plan=use_plan,
                enable_repair=True,
                allow_template_fallback=False,
            )
            assert fim_model_only["model_generated"] is True
            assert fim_model_only["repair_source"] in ("none", "exact_ar_model", "exact_ar_model_unrepaired")
            assert fim_model_only["full_text"] == prefix_str + fim_model_only["middle_text"] + suffix_str
            assert fim_model_only["sequential_passes"] == (
                fim_model_only["plan_steps"] + fim_model_only["concept_steps"] + fim_model_only["repair_passes"]
            )
            assert fim_model_only["speedup_vs_ar"] > 0.0
            if use_plan:
                assert fim_model_only["plan_steps"] >= 2
                assert 1 <= fim_model_only["planned_chunks"] <= 2

            # Explicit opt-in template fallback upper bound
            fim_res = model.generate_fim(
                prefix=prefix_str,
                suffix=suffix_str,
                tokenizer=tok,
                max_chunks=2,
                temperature=0.0,
                early_exit_conf=0.92,
                quantize_kv=q_kv,
                use_plan=use_plan,
                enable_repair=True,
                allow_template_fallback=True,
            )
            assert isinstance(fim_res["middle_text"], str) and len(fim_res["middle_text"]) > 0
            assert fim_res["full_text"] == prefix_str + fim_res["middle_text"] + suffix_str
            assert fim_res["parses"] is True, f"generate_fim output did not parse: {fim_res['full_text']!r}"
            assert fim_res["keywords_present"] is True

    # Tensor inputs + packed INT4 quantized model compatibility
    p_tensor = torch.tensor([tok.encode(prefix_str)], dtype=torch.long)
    s_tensor = torch.tensor([tok.encode(suffix_str)], dtype=torch.long)
    q4_model = quantize_model_int4(model, group_size=32, quantize_embeddings=True).eval()
    fim_q4 = q4_model.generate_fim(
        prefix=p_tensor,
        suffix=s_tensor,
        tokenizer=tok,
        use_plan=True,
        enable_repair=True,
        allow_template_fallback=True,
        quantize_kv=True,
    )
    assert fim_q4["parses"] is True and fim_q4["keywords_present"] is True


def test_fim_plan_dataset_stage2_joint_and_compiler_in_loop_grpo() -> None:
    """Test 19: FIMPlanDataset (50% FIM, 15%/15% kw corruption, plan+middle exact_labels),
    Stage 2 Joint training on FIMPlanDataset, compute_fim_reward gating, and train_pg_fim_grpo.
    """
    from cncm.trainer import (
        FIMPlanDataset,
        compute_fim_reward,
        make_pg_fim_stage1_corpus,
        make_synthetic_code_corpus,
        make_synthetic_fim_tasks,
        train_pg_fim_grpo,
    )

    set_seed(89)
    cfg = get_preset("tiny_test", use_copy_decoder=True)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)

    # 19a. Synthetic code & PG-FIM Stage 1 / Stage 3 corpora
    code_corpus = make_synthetic_code_corpus(num_samples=24, seed=42)
    assert len(code_corpus) == 24
    s1_fim_corpus = make_pg_fim_stage1_corpus(num_samples=32, seed=42)
    joined_s1 = "\n".join(s1_fim_corpus)
    for sentinel in ("<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<fim_end>", "<kw>", "</kw>", "<kw/>", "<mlen_1>"):
        assert sentinel in joined_s1, f"Missing sentinel {sentinel} in make_pg_fim_stage1_corpus"

    fim_tasks = make_synthetic_fim_tasks(num_samples=36, seed=42, curriculum_stage="all")
    assert len(fim_tasks) == 36
    stages_present = {t["curriculum_stage"] for t in fim_tasks}
    assert {"single_line", "multi_line", "function_body"}.issubset(stages_present)

    # 19b. FIMPlanDataset: ~50% FIM mix, 15% empty / 15% corrupted / 70% clean plans, joint exact_labels
    fim_ds = FIMPlanDataset(
        samples=code_corpus + fim_tasks,
        tokenizer=tok,
        chunk_size=cfg.chunk_size,
        max_prefix_len=96,
        fim_rate=0.5,
        kw_max=3,
        kw_empty_prob=0.15,
        kw_corrupt_prob=0.15,
        seed=42,
        fixed_prefix_len=False,
    )
    assert len(fim_ds) >= 20
    fim_count = sum(int(ex["is_fim"]) for ex in fim_ds.examples)
    non_fim_count = len(fim_ds.examples) - fim_count
    assert fim_count > 0 and non_fim_count > 0, f"Expected 50/50 FIM/Block-AR mix, got fim={fim_count}, ar={non_fim_count}"

    plan_kinds = {ex["plan_kind"] for ex in fim_ds.examples if ex["is_fim"] == 1}
    assert "clean" in plan_kinds and ("empty" in plan_kinds or "corrupted" in plan_kinds)

    # Verify joint exact_labels supervision on BOTH plan tokens (in prefix tail) AND middle target tokens
    saw_plan_supervision = False
    for idx in range(len(fim_ds)):
        item = fim_ds[idx]
        if int(item["is_fim"].item()) == 1 and int(item["plan_len"].item()) > 0:
            p_len = item["prefix_ids"].numel()
            pl_len = int(item["plan_len"].item())
            # Plan tokens at the tail of prefix_ids must have exact_labels != -100
            plan_lbls = item["exact_labels"][p_len - pl_len : p_len]
            assert (plan_lbls != -100).all().item()
            assert torch.equal(plan_lbls, item["prefix_ids"][p_len - pl_len : p_len])
            # Earlier PSM context tokens before the plan must be masked with -100
            if p_len > pl_len:
                assert (item["exact_labels"][: p_len - pl_len] == -100).all().item()
            # Valid target chunk tokens must also be supervised (!= -100)
            assert (item["exact_labels"][p_len:] != -100).any().item()
            saw_plan_supervision = True
            break
    assert saw_plan_supervision, "No FIM example with supervised plan tokens found in FIMPlanDataset"

    # Verify collate_block_ar preserves plan + target exact_labels supervision
    fim_indices = [i for i in range(len(fim_ds)) if int(fim_ds[i]["plan_len"].item()) > 0][:3]
    collated_fim = collate_block_ar([fim_ds[i] for i in fim_indices], pad_token_id=cfg.pad_token_id)
    max_p = collated_fim["prefix_ids"].shape[1]
    assert (collated_fim["exact_labels"][:, :max_p] != -100).any().item(), (
        "collate_block_ar lost plan-token exact_labels supervision!"
    )

    # 19c. Stage 2 Joint training smoke test on FIMPlanDataset
    s1_model = CNCMConceptTokenizer(cfg)
    s1_snap = {k: v.detach().clone() for k, v in s1_model.state_dict().items()}
    s2_fim = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1_model,
        dataset=fim_ds,
        mode="joint",
        steps=8,
        batch_size=4,
        lr=3e-3,
        device="cpu",
        seed=42,
        verbose=False,
    )
    model: CNCMModel = s2_fim["model"]
    assert s2_fim["metrics"]["all_grads_finite"] is True
    assert s2_fim["metrics"]["concept_tokenizer_frozen"] is True
    assert s2_fim["metrics"]["final_loss"] < s2_fim["metrics"]["initial_loss"]
    for k, v in model.concept_tokenizer.state_dict().items():
        assert torch.equal(v, s1_snap[k]), f"Stage 1 mutated during FIM Stage 2 training on {k}!"

    # 19d. Anti-reward-hacking gated compute_fim_reward verification
    task0 = fim_tasks[0]
    # Trivial 'pass' or empty middle must receive 0 parse/compile/test reward!
    r_pass = compute_fim_reward("    pass\n", task0, planned_keywords=task0["oracle_keywords"])
    assert r_pass["non_trivial"] is False
    assert r_pass["parses"] is False and r_pass["compiles"] is False and r_pass["tests_passed"] is False
    assert r_pass["reward"] <= 0.0

    # Valid non-trivial parse+compile that fails unit tests gets +0.20 + 0.30 = 0.50
    lead_ws = task0["middle"][: len(task0["middle"]) - len(task0["middle"].lstrip(" \t"))]
    r_wrong = compute_fim_reward(f"{lead_ws}_dummy_x = 999 + 1", task0, planned_keywords=[])
    assert r_wrong["non_trivial"] is True and r_wrong["parses"] is True and r_wrong["compiles"] is True
    assert r_wrong["tests_passed"] is False
    assert r_wrong["reward"] == pytest.approx(0.50, abs=1e-5)

    # Exact oracle middle passes parse (+0.20) + compile (+0.30) + unit tests (+0.50) + kw_faithful (+0.05) = 1.05
    r_oracle = compute_fim_reward(
        task0["middle"],
        task0,
        planned_keywords=task0["oracle_keywords"],
        planned_chunks=1,
        actual_chunks=1,
    )
    assert r_oracle["parses"] is True and r_oracle["compiles"] is True and r_oracle["tests_passed"] is True
    assert r_oracle["kw_faithful"] is True and r_oracle["length_match"] is True
    assert r_oracle["reward"] == pytest.approx(1.05, abs=1e-5)

    # Wrong length bucket incurs -0.05 penalty
    r_len_mismatch = compute_fim_reward(
        task0["middle"],
        task0,
        planned_keywords=task0["oracle_keywords"],
        planned_chunks=2,
        actual_chunks=1,
    )
    assert r_len_mismatch["length_match"] is False
    assert r_len_mismatch["reward"] == pytest.approx(1.00, abs=1e-5)

    # 19e. compute_pg_fim_logprobs + Stage 3 Compiler-in-the-Loop train_pg_fim_grpo
    from cncm.trainer import train_pg_fim_grpo

    pfx = torch.randint(16, cfg.vocab_size - 11, (2, 10), dtype=torch.long)
    pl_ids = torch.tensor(
        [[cfg.mlen_1_id, cfg.kw_empty_id], [cfg.mlen_1_id, cfg.kw_empty_id]], dtype=torch.long
    )
    with torch.no_grad():
        c_gen = model.generate_concept_chunk(torch.cat([pfx, pl_ids], dim=1), return_codes=True)
        lp_dict = model.compute_pg_fim_logprobs(pfx, pl_ids, c_gen["codes"])
        for k_lp in ("total_logprobs", "plan_logprobs", "code_logprobs"):
            assert lp_dict[k_lp].shape == (2,)
            assert torch.isfinite(lp_dict[k_lp]).all().item()

    fim_grpo_res = train_pg_fim_grpo(
        cfg=cfg,
        tokenizer=tok,
        model=model,
        tasks=fim_tasks[:6],
        steps=2,
        num_generations=3,
        batch_size=1,
        lr=5e-4,
        device="cpu",
        seed=42,
        verbose=False,
    )
    fgm = fim_grpo_res["metrics"]
    assert fgm["fim_mode"] is True
    assert fgm["all_logprobs_finite"] is True
    assert fgm["all_grads_finite"] is True
    assert fgm["policy_updated"] is True and fgm["policy_param_delta_l2"] > 0.0
    assert fgm["concept_tokenizer_frozen"] is True
    assert len(fgm["curriculum_stages_visited"]) >= 1
    for k, v in model.concept_tokenizer.state_dict().items():
        assert torch.equal(v, s1_snap[k]), f"Stage 1 mutated during PG-FIM GRPO on {k}!"


# ============================================================================
# 8. Pre-Training Fixes & 50M-Token Pilot Harness Verification (Tests 20-23)
# ============================================================================


def test_stage1_semantic_encoder_is_strictly_causal() -> None:
    """Test 20: Stage 1 `encode_z` with `enc_layers=1` must be strictly causal:
    perturbing token at index `t` must leave `z[:, :t]` bit-for-bit identical.
    """
    set_seed(301)
    cfg = get_preset("tiny_test", enc_layers=1)
    s1 = CNCMConceptTokenizer(cfg).eval()

    x_orig = torch.randint(16, cfg.vocab_size - 11, (2, cfg.chunk_size), dtype=torch.long)
    t_perturb = cfg.chunk_size // 2
    x_mod = x_orig.clone()
    x_mod[:, t_perturb:] = (x_mod[:, t_perturb:] + 7) % (cfg.vocab_size - 27) + 16
    assert not torch.equal(x_orig[:, t_perturb:], x_mod[:, t_perturb:])

    # Check raw causal encoder output (before GroupNorm across the full L sequence)
    # AND confirm no future token leaks into earlier positions in encoder_blocks.
    with torch.no_grad():
        emb_orig = s1.wte(x_orig)
        emb_mod = s1.wte(x_mod)
        pos_ids = torch.arange(cfg.chunk_size, dtype=torch.long)
        cos, sin = s1.dec_rope(pos_ids, dtype=emb_orig.dtype)
        from cncm.layers import build_causal_mask

        cmask = build_causal_mask(
            q_len=cfg.chunk_size,
            kv_len=cfg.chunk_size,
            past_len=0,
            batch_size=2,
            key_padding_mask=None,
            device=emb_orig.device,
            dtype=emb_orig.dtype,
        )
        h_orig = emb_orig
        h_mod = emb_mod
        assert s1.encoder_blocks is not None and len(s1.encoder_blocks) == 1
        for blk in s1.encoder_blocks:
            h_orig = blk(h_orig, cos=cos, sin=sin, attn_mask=cmask)
            h_mod = blk(h_mod, cos=cos, sin=sin, attn_mask=cmask)

        # Causal attention invariance: positions 0..t_perturb-1 MUST be bit-for-bit identical
        assert torch.equal(h_orig[:, :t_perturb, :], h_mod[:, :t_perturb, :]), (
            "Stage 1 semantic encoder leaked future tokens into earlier positions!"
        )
        # While positions >= t_perturb MUST differ
        assert not torch.allclose(h_orig[:, t_perturb:, :], h_mod[:, t_perturb:, :])

        # Verify full encode_z runs cleanly with finite output
        z_out = s1.encode_z(x_orig)
        assert z_out.shape == (2, cfg.chunk_size, cfg.d_vq)
        assert torch.isfinite(z_out).all().item()


def test_forward_joint_align_loss_and_supervise_prefix_exact_gradients() -> None:
    """Test 21: `forward_joint(..., align_weight=0.3)` returns finite `align_loss` and
    backpropagates non-zero gradients into both `concept_in_proj` and `exact_head`.
    """
    set_seed(302)
    cfg = get_preset("tiny_test", enc_layers=1)
    s1 = CNCMConceptTokenizer(cfg)
    model = CNCMModel(cfg, s1)
    model.train()

    pfx = torch.randint(16, cfg.vocab_size - 11, (2, 12), dtype=torch.long)
    tgt = torch.randint(16, cfg.vocab_size - 11, (2, cfg.chunk_size), dtype=torch.long)
    pfx_mask = torch.ones_like(pfx)

    # 1. Verify h_target is returned by forward_concept
    with torch.no_grad():
        codes = s1.encode_codes(tgt)
    fc_out = model.forward_concept(pfx, codes, prefix_mask=pfx_mask, corrupt_rate=0.0)
    assert "h_target" in fc_out
    assert fc_out["h_target"].shape == (2, cfg.total_concept_tokens, cfg.hidden_size)

    # 2. Verify pure align_loss (concept_weight=0, exact_weight=0, align_weight=0.3)
    #    backpropagates non-zero gradients into BOTH `concept_in_proj` and `exact_head`!
    model.zero_grad(set_to_none=True)
    out_align_only = model.forward_joint(
        pfx,
        tgt,
        prefix_mask=pfx_mask,
        concept_weight=0.0,
        exact_weight=0.0,
        align_weight=0.3,
        corrupt_rate=0.5,
    )
    assert "align_loss" in out_align_only
    assert torch.isfinite(out_align_only["align_loss"]).item()
    assert float(out_align_only["align_loss"].item()) > 0.0
    out_align_only["loss"].backward()

    assert model.concept_in_proj.weight.grad is not None
    assert float(model.concept_in_proj.weight.grad.abs().sum().item()) > 0.0
    assert model.exact_head.weight.grad is not None
    assert float(model.exact_head.weight.grad.abs().sum().item()) > 0.0

    # 3. Verify full joint loss + supervise_prefix_exact=True
    model.zero_grad(set_to_none=True)
    out_full = model.forward_joint(
        pfx,
        tgt,
        prefix_mask=pfx_mask,
        concept_weight=1.0,
        exact_weight=0.3,
        align_weight=0.3,
        supervise_prefix_exact=True,
    )
    for key in ("loss", "concept_loss", "exact_loss", "align_loss"):
        assert torch.isfinite(out_full[key]).item()
    out_full["loss"].backward()
    assert float(model.concept_in_proj.weight.grad.abs().sum().item()) > 0.0
    assert float(model.exact_head.weight.grad.abs().sum().item()) > 0.0


def test_pilot_eval_diagnostics_and_stage2_cli_exact_ar_align(tmp_path: Path) -> None:
    """Test 22: Verify `pilot_eval.py` diagnostics on `tiny_test` and `train_stage2` CLI
    acceptance of `--mode exact_ar --align-weight 0.3`.
    """
    from cncm.trainer import (
        make_synthetic_code_corpus,
        measure_code_entropy,
        measure_concept_vs_exact_gap,
        measure_per_domain_stage1_recon,
        measure_tokenizer_fertility,
    )
    from cncm.trainer.train_stage2 import build_argparser

    set_seed(303)
    cfg = get_preset("tiny_test", enc_layers=1, matryoshka_cutoffs=(3, 4, 5))
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    s1 = CNCMConceptTokenizer(cfg).eval()
    model = CNCMModel(cfg, s1).eval()

    domain_texts = {
        "english": make_synthetic_corpus(num_samples=16, seed=42),
        "python": make_synthetic_code_corpus(num_samples=16, seed=42),
        "json": [
            json.dumps({"name": "calculator", "arguments": {"expression": f"{i} * 3 + 1"}, "ok": True})
            for i in range(16)
        ],
    }

    # 1. Fertility
    fert = measure_tokenizer_fertility(tok, domain_texts)
    assert set(fert["domains"].keys()) == {"english", "python", "json"}
    for dom_stats in fert["domains"].values():
        assert dom_stats["chars_per_token"] > 0.0

    # 2. Per-domain Stage 1 recon
    s1_rec = measure_per_domain_stage1_recon(
        s1, tok, domain_texts, device="cpu", max_chunks_per_domain=8, batch_size=4
    )
    assert s1_rec["k_full"] == cfg.num_levels
    assert s1_rec["k_prev"] == cfg.num_levels - 1
    assert set(s1_rec["domains"].keys()) == {"english", "python", "json"}

    # 3. Code entropy & concept-vs-exact gap
    val_ds = PackedChunkDataset(
        domain_texts["english"] + domain_texts["python"],
        tok,
        chunk_size=cfg.chunk_size,
        prefix_len=16,
    )
    ent = measure_code_entropy(model, val_ds, device="cpu", max_batches=2, batch_size=4)
    assert ent["batches_evaluated"] >= 1
    assert ent["mean_conditional_nats_per_code"] > 0.0
    assert len(ent["per_level_conditional_nats"]) == cfg.num_levels
    assert len(ent["per_level_marginal_nats"]) == cfg.num_levels

    gap = measure_concept_vs_exact_gap(model, val_ds, device="cpu", max_batches=2, batch_size=4)
    assert gap["batches_evaluated"] >= 1
    assert gap["exact_ar_nll"] > 0.0 and gap["align_nll"] > 0.0 and gap["concept_draft_nll_by_exact"] > 0.0

    # 4. train_stage2 CLI parser & train_stage2 align_loss_history check
    parser = build_argparser()
    parsed = parser.parse_args(
        ["--preset", "tiny_test", "--mode", "exact_ar", "--align-weight", "0.3", "--steps", "2"]
    )
    assert parsed.mode == "exact_ar"
    assert parsed.align_weight == pytest.approx(0.3)

    s2_ex = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1,
        dataset=val_ds,
        mode="exact_ar",
        steps=2,
        batch_size=2,
        device="cpu",
        verbose=False,
    )
    assert s2_ex["metrics"]["mode"] == "exact_ar"
    assert s2_ex["metrics"]["all_grads_finite"] is True

    s2_j = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1,
        dataset=val_ds,
        mode="joint",
        steps=2,
        batch_size=2,
        align_weight=0.3,
        supervise_prefix_exact=True,
        device="cpu",
        verbose=False,
    )
    assert len(s2_j["metrics"]["align_loss_history"]) == 2
    assert all(v > 0.0 for v in s2_j["metrics"]["align_loss_history"])

    # 5. Subprocess CLI smoke check on `--mode exact_ar --align-weight 0.3`
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cncm.trainer.train_stage2",
            "--preset",
            "tiny_test",
            "--mode",
            "exact_ar",
            "--align-weight",
            "0.3",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--num-samples",
            "16",
            "--device",
            "cpu",
            "--quiet",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"train_stage2 CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_bpe_loader_merges_and_kaggle_pilot_smoke(tmp_path: Path) -> None:
    """Test 23: Verify BPE merge loading (`from_hf_json` + individual digits) and `kaggle_pilot_50m.py --smoke`."""
    # Construct a minimal HF ByteLevel BPE JSON with Digits pre-tokenizer & merges
    hf_bpe = {
        "model": {
            "type": "BPE",
            "vocab": {
                "<|pad|>": 0,
                "def": 100,
                "Ġreturn": 101,
                "Ġtotal": 102,
            },
            "merges": ["d e", "de f", "Ġ r", "Ġr e", "Ġre t", "Ġret u", "Ġretu r", "Ġretur n"],
        },
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {"type": "ByteLevel", "add_prefix_space": False},
            ],
        },
        "decoder": {"type": "ByteLevel"},
    }
    bpe_file = tmp_path / "mini_bpe.json"
    bpe_file.write_text(json.dumps(hf_bpe), encoding="utf-8")

    tok = CNCMTextTokenizer.from_hf_json(bpe_file, vocab_size=512)
    assert tok.vocab_size == 512
    # Verify single digits are never merged when Digits(individual_digits=True)
    digit_ids = tok.encode("2026")
    assert [tok.decode([tid]) for tid in digit_ids] == ["2", "0", "2", "6"]
    # Verify round-trip with core & PG-FIM special tokens preserved
    sample = "<fim_prefix>def f(): return 123<fim_suffix><fim_middle><mlen_1><kw/>"
    assert tok.decode(tok.encode(sample), skip_special_tokens=False) == sample

    # Verify nano config defaults updated (concept_head_hidden=160, loop_adapter_rank=32)
    nano_cfg = get_preset("nano")
    assert nano_cfg.concept_head_hidden == 160
    assert nano_cfg.loop_adapter_rank == 32

    # Smoke-run kaggle_pilot_50m.py end-to-end on tiny_test (offline synthetic)
    pilot_out = tmp_path / "pilot_smoke"
    report_file = pilot_out / "pilot_50m_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "kaggle_pilot_50m.py",
            "--preset",
            "tiny_test",
            "--smoke",
            "--no-hf-download",
            "--stage1-steps",
            "4",
            "--stage2-steps",
            "4",
            "--grad-accum",
            "1",
            "--device",
            "cpu",
            "--out-dir",
            str(pilot_out),
            "--report-path",
            str(report_file),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"kaggle_pilot_50m.py --smoke failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert report_file.exists()
    rep = json.loads(report_file.read_text(encoding="utf-8"))
    assert rep["overall_status"] in ("PASS", "FAIL")
    assert len(rep["gate_summary"]) == 5


def test_memmap_checkpoint_manager_resume_and_wall_clock_guard(tmp_path: Path) -> None:
    """Test 24: Verify Kaggle-optimized training upgrades:
    1. `MemmapTokenDataset` round-trips exact token windows with 0% padding + deterministic offsets & DDP sharding.
    2. `CheckpointManager` prunes old checkpoints to `keep_last_k=2`, tracks `best.pt`, and resuming
       `train_stage2` from `latest.pt` restores step/optimizer/scheduler/tokens_seen/RNG and continues
       loss descent cleanly without NaNs (including with `grad_checkpointing=True`).
    3. `max_wall_minutes=0.0` triggers a graceful checkpoint save & clean exit for both Stage 1 & Stage 2.
    """
    import math
    import numpy as np
    from cncm.trainer import (
        CheckpointManager,
        MemmapTokenDataset,
        load_checkpoint,
        prepare_memmap_corpus,
    )

    set_seed(404)
    cfg = get_preset("tiny_test", enc_layers=1)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    corpus = make_synthetic_corpus(num_samples=48, seed=404)

    # 1. MemmapTokenDataset exact round-trip & 0% padding
    bin_path = prepare_memmap_corpus(
        corpus,
        tok,
        tmp_path / "corpus_uint16.bin",
        eos_token_id=cfg.eos_token_id,
        bos_token_id=cfg.bos_token_id,
        dtype=np.uint16,
    )
    assert bin_path.exists() and bin_path.stat().st_size > 0

    pc_s1 = PackedChunkDataset(
        corpus, tok, chunk_size=cfg.chunk_size, prefix_len=0, eos_token_id=cfg.eos_token_id, bos_token_id=cfg.bos_token_id
    )
    mm_s1 = MemmapTokenDataset(bin_path, chunk_size=cfg.chunk_size, prefix_len=0, dtype=np.uint16)
    assert mm_s1.pad_ratio == 0.0
    assert mm_s1.num_tokens == pc_s1.num_tokens
    assert len(mm_s1) == len(pc_s1)
    for idx in (0, 1, len(mm_s1) // 2, len(mm_s1) - 1):
        assert torch.equal(mm_s1[idx]["input_ids"], pc_s1[idx]["input_ids"])
        assert torch.equal(mm_s1[idx]["attention_mask"], pc_s1[idx]["attention_mask"])

    prefix_len = 16
    pc_s2 = PackedChunkDataset(
        corpus,
        tok,
        chunk_size=cfg.chunk_size,
        prefix_len=prefix_len,
        eos_token_id=cfg.eos_token_id,
        bos_token_id=cfg.bos_token_id,
    )
    mm_s2 = MemmapTokenDataset(bin_path, chunk_size=cfg.chunk_size, prefix_len=prefix_len, dtype=np.uint16)
    assert mm_s2.pad_ratio == 0.0
    assert len(mm_s2) == len(pc_s2)
    for idx in (0, 2, len(mm_s2) - 1):
        for key in ("prefix_ids", "target_ids", "exact_input_ids", "exact_labels"):
            assert torch.equal(mm_s2[idx][key], pc_s2[idx][key]), f"Mismatch on key={key} idx={idx}"

    # Verify deterministic window offset slicing & DDP rank-strided sharding
    mm_off = mm_s2.with_offset(start_offset=3)
    assert len(mm_off) == len(mm_s2) - 3
    assert torch.equal(mm_off[0]["input_ids"], mm_s2[3]["input_ids"])

    mm_r0 = mm_s2.with_offset(start_offset=0, rank=0, world_size=2)
    mm_r1 = mm_s2.with_offset(start_offset=0, rank=1, world_size=2)
    assert len(mm_r0) + len(mm_r1) == len(mm_s2)
    assert torch.equal(mm_r0[0]["input_ids"], mm_s2[0]["input_ids"])
    assert torch.equal(mm_r1[0]["input_ids"], mm_s2[1]["input_ids"])
    assert torch.equal(mm_r0[1]["input_ids"], mm_s2[2]["input_ids"])

    # 2. CheckpointManager pruning (keep_last_k=2), best.pt tracking, grad_checkpointing=True, and seamless resume
    s1_model = CNCMConceptTokenizer(cfg).eval()
    val_ds = mm_s2.with_offset(start_offset=max(0, len(mm_s2) - 8))
    save_dir = tmp_path / "stage2_rolling"

    run1 = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1_model,
        dataset=mm_s2,
        mode="joint",
        steps=6,
        batch_size=2,
        lr=1.5e-3,
        device="cpu",
        grad_checkpointing=True,
        save_dir=save_dir,
        save_every=2,
        keep_last_k=2,
        val_dataset=val_ds,
        eval_every=2,
        eval_max_batches=2,
        seed=404,
        verbose=False,
    )
    m1 = run1["metrics"]
    assert m1["completed_steps"] == 6
    assert m1["all_grads_finite"] is True
    assert m1["grad_checkpointing"] is True
    assert m1["tokens_seen"] > 0
    assert m1["data_cursor"] > 0
    assert (save_dir / "latest.pt").exists()
    assert (save_dir / "best.pt").exists()
    assert (save_dir / "metrics.json").exists()

    mgr_check = CheckpointManager(save_dir, keep_last_k=2)
    numbered1 = [step_num for step_num, _ in mgr_check.list_numbered_checkpoints()]
    assert len(numbered1) == 2, f"Expected keep_last_k=2 checkpoints, found {numbered1}"
    assert numbered1 == [4, 6]
    assert mgr_check.resolve_resume_path() == save_dir / "latest.pt"

    ckpt1 = load_checkpoint(save_dir / "latest.pt", map_location="cpu")
    assert ckpt1["step"] == 6
    assert ckpt1["tokens_seen"] == m1["tokens_seen"]
    assert ckpt1["data_cursor"] == m1["data_cursor"]
    assert ckpt1["best_val_loss"] is not None
    assert ckpt1["rng_state"] is not None and ckpt1["rng_state"]["torch_cpu"] is not None
    assert ckpt1["optimizer_state_dict"] is not None
    assert ckpt1["scheduler_state_dict"]["current_step"] == 6

    # Resume into a brand-new model from latest.pt and train to step 12
    run2 = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1_model,
        dataset=mm_s2,
        mode="joint",
        steps=12,
        batch_size=2,
        lr=1.5e-3,
        device="cpu",
        grad_checkpointing=True,
        resume_from=save_dir / "latest.pt",
        save_dir=save_dir,
        save_every=2,
        keep_last_k=2,
        val_dataset=val_ds,
        eval_every=2,
        eval_max_batches=2,
        seed=999,  # Overridden by restored RNG state from checkpoint
        verbose=False,
    )
    m2 = run2["metrics"]
    assert m2["resumed_from_step"] == 6
    assert m2["completed_steps"] == 12
    assert m2["all_grads_finite"] is True
    assert m2["tokens_seen"] > m1["tokens_seen"]
    assert len(m2["loss_history"]) == 12
    assert all(math.isfinite(x) for x in m2["loss_history"])
    # Verify continued loss descent across the resume boundary
    assert sum(m2["loss_history"][-3:]) / 3.0 < m2["loss_history"][0]

    numbered2 = [step_num for step_num, _ in mgr_check.list_numbered_checkpoints()]
    assert len(numbered2) == 2, f"Expected keep_last_k=2 after resume, found {numbered2}"
    assert numbered2 == [10, 12]

    # 3. max_wall_minutes=0.0 triggers graceful checkpoint save & clean exit
    wall_dir_s2 = tmp_path / "wall_guard_s2"
    wall_res_s2 = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1_model,
        dataset=mm_s2,
        mode="joint",
        steps=20,
        batch_size=2,
        device="cpu",
        save_dir=wall_dir_s2,
        max_wall_minutes=0.0,
        verbose=False,
    )
    assert wall_res_s2["metrics"]["stopped_early_wall_clock"] is True
    assert wall_res_s2["metrics"]["completed_steps"] == 0
    assert (wall_dir_s2 / "latest.pt").exists()
    assert (wall_dir_s2 / "metrics.json").exists()

    wall_dir_s1 = tmp_path / "wall_guard_s1"
    wall_res_s1 = train_stage1(
        cfg=cfg,
        tokenizer=tok,
        dataset=mm_s1,
        steps=20,
        batch_size=2,
        device="cpu",
        save_dir=wall_dir_s1,
        max_wall_minutes=0.0,
        verbose=False,
    )
    assert wall_res_s1["metrics"]["stopped_early_wall_clock"] is True
    assert wall_res_s1["metrics"]["completed_steps"] == 0
    assert (wall_dir_s1 / "latest.pt").exists()
    assert (wall_dir_s1 / "metrics.json").exists()

    # 4. HF Hub resolve_resume_path fallback + non-fatal async upload thread (offline mock)
    import types

    hf_calls: dict[str, list[Any]] = {"create_repo": [], "upload_file": [], "download": []}
    hf_cache_ckpt = save_dir / "latest.pt"

    class _FakeHfApi:
        def __init__(self, token: Optional[str] = None) -> None:
            self.token = token

        def create_repo(self, repo_id: str, private: bool = True, exist_ok: bool = True) -> None:
            hf_calls["create_repo"].append((repo_id, private, exist_ok))

        def upload_file(self, path_or_fileobj: str, path_in_repo: str, repo_id: str) -> None:
            hf_calls["upload_file"].append((Path(path_or_fileobj).name, path_in_repo, repo_id))

    def _fake_hf_download(repo_id: str, filename: str, local_dir: str, token: Optional[str] = None) -> str:
        hf_calls["download"].append((repo_id, filename))
        dst = Path(local_dir) / Path(filename).name
        dst.write_bytes(hf_cache_ckpt.read_bytes())
        return str(dst)

    fake_hf_mod = types.ModuleType("huggingface_hub")
    fake_hf_mod.HfApi = _FakeHfApi  # type: ignore[attr-defined]
    fake_hf_mod.hf_hub_download = _fake_hf_download  # type: ignore[attr-defined]
    prev_hf_mod = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake_hf_mod
    try:
        hf_dir = tmp_path / "hf_sync_test"
        hf_mgr = CheckpointManager(
            hf_dir,
            keep_last_k=2,
            hf_repo_id="test-user/cncm-private-test",
            hf_upload_every=2,
            is_main_process=True,
        )
        # Empty local dir -> falls back to hf_hub_download("latest.pt")
        resolved = hf_mgr.resolve_resume_path()
        assert resolved is not None and resolved.exists()
        assert hf_calls["download"] == [("test-user/cncm-private-test", "latest.pt")]

        # Saving triggers background thread upload (private=True, exist_ok=True)
        hf_mgr.save(
            model=run2["model"],
            step=2,
            config=cfg,
            metrics={"step": 2},
            val_loss=1.23,
            save_numbered=False,
            force_hf_upload=True,
        )
        hf_mgr.wait_for_uploads(timeout=10.0)
        assert hf_calls["create_repo"] == [("test-user/cncm-private-test", True, True)]
        uploaded_remotes = {item[1] for item in hf_calls["upload_file"]}
        assert {"latest.pt", "best.pt", "metrics.json"}.issubset(uploaded_remotes)
    finally:
        if prev_hf_mod is not None:
            sys.modules["huggingface_hub"] = prev_hf_mod
        else:
            sys.modules.pop("huggingface_hub", None)

    # Release memmap handles cleanly on Windows
    for ds in (mm_s1, mm_s2, mm_off, mm_r0, mm_r1, val_ds):
        ds.close()


def test_moe_routing_load_balance_grad_ckpt_and_base_moe_preset(tmp_path: Path) -> None:
    """Test 25: Verify DeepSeek-style Fine-Grained MoE (`base_moe` preset + per-loop routers,
    load-balancing auxiliary loss, grad checkpointing, Stage 2 training, and generation).
    """
    set_seed(505)

    # 1. Verify base_moe preset geometry & parameter accounting (~221M total, ~94M active)
    base_moe_cfg = get_preset("base_moe", enc_layers=1, matryoshka_cutoffs=(4, 6, 7, 8))
    assert base_moe_cfg.num_experts == 8
    assert base_moe_cfg.num_experts_per_tok == 2
    assert base_moe_cfg.effective_layers == 24
    assert base_moe_cfg.moe_expert_intermediate_size == 768
    assert base_moe_cfg.moe_shared_intermediate_size == 768

    # 2. Functional & gradient verification on tiny MoE model (with grad_checkpointing=True)
    cfg = get_preset(
        "tiny_test",
        enc_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_expert_intermediate_size=64,
        moe_shared_intermediate_size=64,
        moe_aux_loss_weight=0.01,
    )
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    s1 = CNCMConceptTokenizer(cfg).eval()
    model = CNCMModel(cfg, s1)
    pstats = model.count_parameters()
    assert pstats["total"] > pstats["active_total"] > 0
    assert model.step_moe_routers is not None and len(model.step_moe_routers) == cfg.effective_layers

    model.train()
    model.set_grad_checkpointing(True)
    pfx = torch.randint(16, cfg.vocab_size - 11, (2, 12), dtype=torch.long)
    tgt = torch.randint(16, cfg.vocab_size - 11, (2, cfg.chunk_size), dtype=torch.long)
    pfx_mask = torch.ones_like(pfx)

    # forward_exact + aux loss + backward
    model.zero_grad(set_to_none=True)
    full_ids = torch.cat([pfx, tgt], dim=1)
    ex_out = model.forward_exact(full_ids, labels=full_ids)
    assert "moe_aux_loss" in ex_out and float(ex_out["moe_aux_loss"].item()) > 0.0
    assert torch.isfinite(ex_out["loss"]).item()
    ex_out["loss"].backward()

    # Verify per-loop routers & shared expert received finite non-zero gradients
    for rtr in model.step_moe_routers:
        assert rtr.weight.grad is not None
        assert torch.isfinite(rtr.weight.grad).all().item()
        assert float(rtr.weight.grad.abs().sum().item()) > 0.0
    assert model.blocks[0].mlp.shared_expert.down_proj.weight.grad is not None
    assert float(model.blocks[0].mlp.shared_expert.down_proj.weight.grad.abs().sum().item()) > 0.0

    # forward_joint + Stage 2 training loop (both exact_ar and joint modes)
    corpus = make_synthetic_corpus(num_samples=24, seed=505)
    ds = PackedChunkDataset(corpus, tok, chunk_size=cfg.chunk_size, prefix_len=16)
    res = train_stage2(
        cfg=cfg,
        tokenizer=tok,
        concept_tokenizer=s1,
        dataset=ds,
        mode="joint",
        steps=4,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        grad_checkpointing=True,
        model=model,
        save_path=tmp_path / "tiny_moe.pt",
        verbose=False,
    )
    assert res["metrics"]["all_grads_finite"] is True
    assert (tmp_path / "tiny_moe.pt").exists()

    # Exact-AR and Concept-Mode generation under MoE
    model.eval()
    with torch.no_grad():
        g_ex = model.generate_exact(pfx[:1], max_new_tokens=6, temperature=0.0)
        g_cc = model.generate_concept_chunk(pfx[:1], temperature=0.0)
    assert g_ex["token_ids"].shape[1] == 6
    assert g_cc["token_ids"].shape == (1, cfg.chunk_size)




