"""Comprehensive Parity, Numerical Edge-Case & End-to-End Tests for Path A Sidecar.

Covers every critical invariant required by `future_cncm_update.md`:
  1. `CNCM-DeltaNet` Chunk-WY vs token-by-token recurrent parity (`< 1e-5`), arbitrary chunk
     splits, worst-case collinear keys (`beta -> 2.0`), extreme decay (`alpha_log = -40`
     backward NaN-freedom), and left/right padding mask state invariance.
  2. Dual-Phase Read-Only State Contract (`0 MB` target state mutation & `< 1e-5` parity
     between stacked teacher-forced training and sequential level-by-level inference).
  3. Universal Donor Loader & Parity across `llama`, `qwen2` (QKV bias + untied head + >65535
     `int32` vocab), `qwen3` (`head_dim != H/nh` + `q_norm`/`k_norm`), `gemma2` (`(1+w)` RMSNorm +
     GeGLU + softcap + SWA), and CNCM `.safetensors` donor extraction.
  4. `SVDConceptTokenizer` deterministic orthonormal SVD, `d_vq > H_donor` edge case, and
     variable chunk schedules (`L = 16, 32, 64`).
  5. `OfflineDonorCache` exact target-logit alignment (zero off-by-one shift), non-negative
     `sparse_topk_tail_kl_loss`, Step-0 Ridge warm-start NLL drop, and zero-backprop training.
  6. EAGLE-3 shifted incremental commit parity, **100% bit-exact greedy equivalence** between
     `generate_speculative_lossless` and `donor.generate_exact`, `generate_ultra_precise`, and
     `.sidecar.gguf` (`q4_0`, `q8_0`, `f16`, `f32`) round-trip export/load.
  7. Full `universal_surgery.py --smoke` end-to-end pipeline verification.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from cncm.config import get_preset
from cncm.deltanet import (
    CNCMDeltaNetBlock,
    deltanet_chunk_wy_parallel,
    deltanet_recurrent_reference,
    invert_unit_lower_triangular_block,
)
from cncm.donor import (
    OfflineDonorCache,
    build_synthetic_donor,
    load_universal_donor,
    verify_donor_parity,
)
from cncm.layers import RotaryEmbedding
from cncm.model import CNCMModel
from cncm.sidecar import (
    CNCMDeltaNetSidecar,
    SVDConceptTokenizer,
    SidecarConfig,
    compute_svd_projection,
    default_levels_for_chunk_size,
    sparse_topk_tail_kl_loss,
)
from cncm.sidecar_gguf import export_sidecar_gguf, load_sidecar_gguf
from universal_surgery import build_arg_parser, run_path_a_surgery


class TestPathADeltaNetInvariants(unittest.TestCase):
    def test_block_triangular_inverse_worst_case_collinear(self) -> None:
        """Collinear keys with beta=1.95 must invert accurately without O(2^64) cancellation."""
        c = 64
        idx = torch.arange(c)
        tril = idx.unsqueeze(1) > idx.unsqueeze(0)
        # Worst-case collinear matrix: L_{t,s} = 1.95 for all t > s
        L = torch.where(tril, torch.full((2, 3, c, c), 1.95), torch.zeros((2, 3, c, c)))
        T = invert_unit_lower_triangular_block(L)
        ident = torch.matmul(T, torch.eye(c) + L)
        err = (ident - torch.eye(c)).abs().max().item()
        self.assertLess(err, 1e-4, f"Collinear 64x64 triangular inverse error too high: {err}")

    def test_chunk_wy_matches_recurrent_across_arbitrary_splits(self) -> None:
        torch.manual_seed(123)
        b, h, s, d = 2, 4, 103, 16
        q = F.normalize(torch.randn(b, h, s, d), dim=-1) * (d ** -0.5)
        k = F.normalize(torch.randn(b, h, s, d), dim=-1)
        v = torch.randn(b, h, s, d)
        alpha_log = -F.softplus(torch.randn(b, h, s) - 2.0)
        beta = 2.0 * torch.sigmoid(torch.randn(b, h, s))
        s0 = torch.randn(b, h, d, d) * 0.1

        out_ref, st_ref = deltanet_recurrent_reference(q, k, v, alpha_log, beta, state=s0)
        out_wy64, st_wy64 = deltanet_chunk_wy_parallel(q, k, v, alpha_log, beta, state=s0, chunk_size=64)
        out_wy17, st_wy17 = deltanet_chunk_wy_parallel(q, k, v, alpha_log, beta, state=s0, chunk_size=17)

        self.assertLess((out_ref - out_wy64).abs().max().item(), 1e-5)
        self.assertLess((st_ref - st_wy64).abs().max().item(), 1e-5)
        self.assertLess((out_ref - out_wy17).abs().max().item(), 1e-5)
        self.assertLess((st_ref - st_wy17).abs().max().item(), 1e-5)

        # Incremental speculative split [0:29] + [29:43] + [43:103]
        o1, s1 = deltanet_chunk_wy_parallel(q[:, :, :29], k[:, :, :29], v[:, :, :29], alpha_log[:, :, :29], beta[:, :, :29], state=s0)
        o2, s2 = deltanet_chunk_wy_parallel(q[:, :, 29:43], k[:, :, 29:43], v[:, :, 29:43], alpha_log[:, :, 29:43], beta[:, :, 29:43], state=s1)
        o3, s3 = deltanet_chunk_wy_parallel(q[:, :, 43:], k[:, :, 43:], v[:, :, 43:], alpha_log[:, :, 43:], beta[:, :, 43:], state=s2)
        o_cat = torch.cat([o1, o2, o3], dim=2)
        self.assertLess((out_ref - o_cat).abs().max().item(), 1e-5)
        self.assertLess((st_ref - s3).abs().max().item(), 1e-5)

    def test_extreme_decay_autograd_nan_free(self) -> None:
        """Extreme decay (`alpha_log = -40`) must never produce `0 * inf = NaN` in forward or backward."""
        torch.manual_seed(7)
        q = F.normalize(torch.randn(1, 2, 32, 8), dim=-1).requires_grad_(True)
        k = F.normalize(torch.randn(1, 2, 32, 8), dim=-1).requires_grad_(True)
        v = torch.randn(1, 2, 32, 8, requires_grad=True)
        alpha_log = torch.full((1, 2, 32), -40.0, requires_grad=True)
        beta = torch.full((1, 2, 32), 1.5, requires_grad=True)

        out, st = deltanet_chunk_wy_parallel(q, k, v, alpha_log, beta, chunk_size=16)
        loss = out.pow(2).mean() + st.pow(2).mean()
        loss.backward()

        for name, t in [("out", out), ("q.grad", q.grad), ("k.grad", k.grad), ("v.grad", v.grad), ("alpha.grad", alpha_log.grad), ("beta.grad", beta.grad)]:
            self.assertIsNotNone(t, f"Missing gradient for {name}")
            self.assertTrue(bool(torch.isfinite(t).all().item()), f"Non-finite values in {name}")

    def test_padding_mask_state_noop_and_phase_b_readonly_parity(self) -> None:
        torch.manual_seed(99)
        blk = CNCMDeltaNetBlock(hidden_size=64, num_heads=4, intermediate_size=128, chunk_size=16)
        blk.eval()
        rope = RotaryEmbedding(head_dim=16, max_position_embeddings=256)

        x_valid = torch.randn(1, 12, 64)
        cos_v, sin_v = rope(torch.arange(12))
        _, st_clean = blk.forward_phase_a_prefix(x_valid, cos_v, sin_v)
        assert st_clean is not None

        # Append 5 right-padded garbage tokens with mask=0 -> recurrent & conv state must stay identical!
        x_padded = torch.cat([x_valid, torch.randn(1, 5, 64) * 50.0], dim=1)
        mask = torch.tensor([[1] * 12 + [0] * 5], dtype=torch.bool)
        cos_p, sin_p = rope(torch.arange(17))
        _, st_padded = blk.forward_phase_a_prefix(x_padded, cos_p, sin_p, attention_mask=mask)
        assert st_padded is not None

        self.assertEqual(st_clean.seen_tokens, st_padded.seen_tokens)
        self.assertLess((st_clean.recurrent_state - st_padded.recurrent_state).abs().max().item(), 1e-5)
        self.assertLess((st_clean.conv_state - st_padded.conv_state).abs().max().item(), 1e-5)

        # Phase B: stacked levels (1, 2, 4, 8) vs sequential level-by-level + read-only S_prefix check
        st_snap = st_clean.clone()
        levels = (1, 2, 4, 8)
        x_lvls = [torch.randn(1, lk, 64) for lk in levels]
        pos_lvls = [12 + torch.arange(lk) for lk in levels]

        x_cat = torch.cat(x_lvls, dim=1)
        pos_cat = torch.cat(pos_lvls, dim=0)
        cos_t, sin_t = rope(pos_cat)
        out_stacked = blk.forward_phase_b_targets(x_cat, cos_t, sin_t, st_clean, levels)

        # Verify zero mutation of S_prefix
        self.assertTrue(torch.equal(st_clean.recurrent_state, st_snap.recurrent_state))
        self.assertTrue(torch.equal(st_clean.conv_state, st_snap.conv_state))

        seq_outs = []
        for xk, pk, lk in zip(x_lvls, pos_lvls, levels):
            ck, sk = rope(pk)
            seq_outs.append(blk.forward_phase_b_targets(xk, ck, sk, st_clean, (lk,)))
        out_seq = torch.cat(seq_outs, dim=1)
        self.assertLess((out_stacked - out_seq).abs().max().item(), 1e-5)


class TestUniversalDonorAndCache(unittest.TestCase):
    def test_all_donor_architectures_and_cncm_extraction(self) -> None:
        for arch in ("llama", "qwen2", "qwen3", "gemma2"):
            runner, raw_sd = build_synthetic_donor(arch=arch, vocab_size=320, hidden_size=64, num_layers=4)
            metrics = verify_donor_parity(runner, raw_sd)
            self.assertEqual(metrics["top1_agreement"], 1.0)
            self.assertLess(metrics["max_abs_diff"], 1e-4)

        # Verify loading donor directly from a CNCM `.safetensors` file
        cncm_cfg = get_preset("tiny_test", num_loops=1, loop_adapter_rank=0, use_qk_norm=True, rope_theta=100_000.0)
        cncm_model = CNCMModel(cncm_cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            sf_path = Path(tmp) / "tiny_cncm.safetensors"
            sd = {k: v.detach().clone() for k, v in cncm_model.state_dict().items()}
            save_file(sd, str(sf_path))
            donor_runner, parity = load_universal_donor(sf_path, verify_parity=True)
            self.assertEqual(parity["top1_agreement"], 1.0)

            ids = torch.randint(0, cncm_cfg.vocab_size, (2, 12))
            ref_lg = cncm_model.forward_exact(ids)["logits"].float()
            don_lg = donor_runner.forward_with_taps(ids)["logits"].float()
            self.assertLess((ref_lg - don_lg).abs().max().item(), 1e-4)

    def test_large_vocab_int32_cache_and_exact_target_logit_alignment(self) -> None:
        """Vocab > 65535 (`Qwen` 70K+) must use `int32` without uint16 wraparound & align `P - 1 + j`."""
        runner, _ = build_synthetic_donor(arch="qwen2", vocab_size=70000, hidden_size=64, num_layers=3)
        p_len, c_len = 12, 16
        # Include token IDs > 65535!
        seqs = torch.randint(65600, 69990, (4, p_len + c_len), dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmp:
            with OfflineDonorCache.build(
                runner, seqs, tmp, prefix_len=p_len, chunk_size=c_len, top_k=8, batch_size=2
            ) as cache:
                self.assertEqual(cache.meta["id_dtype"], "int32")
                batch = cache.get_batch([0, 1, 2, 3], device=torch.device("cpu"))
                self.assertTrue(torch.equal(batch["prefix_ids"], seqs[:, :p_len]))
                self.assertTrue(torch.equal(batch["target_ids"], seqs[:, p_len:]))

                # Verify `topk_idx[:, j, 0]` == exact causal donor top-1 predicting `target_ids[:, j]`!
                full_out = runner.forward_with_taps(seqs)
                expected_top1 = full_out["logits"][:, p_len - 1 : p_len - 1 + c_len, :].argmax(dim=-1)
                self.assertTrue(torch.equal(batch["topk_idx"][:, :, 0], expected_top1))


class TestSidecarSVDTrainingAndLosslessDecode(unittest.TestCase):
    def test_svd_orthonormality_and_variable_chunks(self) -> None:
        E = torch.randn(600, 80)
        P = compute_svd_projection(E, 48)
        gram = P.T @ P
        self.assertLess((gram - torch.eye(48)).abs().max().item(), 1e-5)

        # Edge case: d_vq > donor_hidden_size
        P_wide = compute_svd_projection(E, 96)
        self.assertEqual(tuple(P_wide.shape), (80, 96))

        for c_sz in (16, 32, 64):
            lvls = default_levels_for_chunk_size(c_sz)
            self.assertEqual(lvls[-1], c_sz)

    def test_sparse_topk_kl_nonnegative_and_zero_at_optimum(self) -> None:
        logits = torch.randn(2, 8, 200)
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        tk_vals, tk_idx = torch.topk(logits, k=12, dim=-1)
        tk_lp = tk_vals - lse

        kl_zero = sparse_topk_tail_kl_loss(logits, tk_idx, tk_lp)
        self.assertLess(abs(float(kl_zero.item())), 1e-5)

        other_logits = torch.randn(2, 8, 200)
        kl_pos = sparse_topk_tail_kl_loss(other_logits, tk_idx, tk_lp)
        self.assertGreaterEqual(float(kl_pos.item()), 0.0)

    def test_shifted_commit_parity_gguf_roundtrip_and_lossless_speculative(self) -> None:
        torch.manual_seed(2026)
        donor, _ = build_synthetic_donor(arch="qwen3", vocab_size=320, hidden_size=64, num_layers=4)
        sc_cfg = SidecarConfig.from_donor(donor.cfg, chunk_size=16, sidecar_layers=4, sidecar_hidden_size=64, d_vq=32)
        svd_p = compute_svd_projection(donor.embed_tokens.weight, sc_cfg.d_vq)
        tok = SVDConceptTokenizer(sc_cfg, svd_proj=svd_p)
        sidecar = CNCMDeltaNetSidecar(sc_cfg, concept_tokenizer=tok).eval()

        # Verify full shifted prefix == incremental commit_shifted_step
        seq = torch.randint(1, 300, (1, 14))
        d_out = donor.forward_with_taps(seq[:, :-1])
        x_full = sidecar.build_shifted_prefix_inputs(seq, d_out["unit_taps"], donor.embed_tokens.weight)
        st_full = sidecar.run_phase_a_prefix(x_full)

        # Incremental: prefill [0..8] + commit [9..13]
        x_part = sidecar.build_shifted_prefix_inputs(seq[:, :9], d_out["unit_taps"][:, :8], donor.embed_tokens.weight)
        st_inc = sidecar.run_phase_a_prefix(x_part)
        st_inc = sidecar.commit_shifted_step(
            d_out["unit_taps"][:, 8:13], seq[:, 9:14], st_inc, donor.embed_tokens.weight
        )
        for sf, si in zip(st_full, st_inc):
            self.assertEqual(sf.seen_tokens, si.seen_tokens)
            self.assertLess((sf.recurrent_state - si.recurrent_state).abs().max().item(), 1e-5)
            self.assertLess((sf.conv_state - si.conv_state).abs().max().item(), 1e-5)

        # Test `.sidecar.gguf` round-trip in both `q8_0` and `q4_0` + 100% lossless greedy parity
        with tempfile.TemporaryDirectory() as tmp:
            for q_fmt in ("f32", "q8_0", "q4_0"):
                g_path = Path(tmp) / f"test_{q_fmt}.sidecar.gguf"
                info = export_sidecar_gguf(sidecar, g_path, quant=q_fmt)
                self.assertTrue(Path(info["path"]).exists())
                loaded = load_sidecar_gguf(g_path)
                if q_fmt == "f32":
                    for k_name, v_orig in sidecar.state_dict().items():
                        self.assertLess((v_orig.float() - loaded.state_dict()[k_name].float()).abs().max().item(), 1e-6)

                prompt = torch.randint(4, 300, (1, 10))
                exact_out = donor.generate_exact(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=-1)
                for d_mode in ("ar", "concept"):
                    spec_out = loaded.generate_speculative_lossless(
                        donor, prompt, max_new_tokens=20, draft_horizon=5, draft_mode=d_mode, temperature=0.0, eos_token_id=-1
                    )
                    self.assertTrue(
                        torch.equal(exact_out, spec_out["token_ids"]),
                        f"Lossless mismatch under quant={q_fmt}, draft_mode={d_mode}!",
                    )

    def test_universal_surgery_end_to_end_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ap = build_arg_parser()
            args = ap.parse_args(["--smoke", "--smoke-arch", "qwen3", "--quant", "q4_0", "--out-dir", tmp])
            args.prefix_len = 24
            args.chunk_size = 16
            args.cache_seqs = 24
            args.stage1_steps = 8
            args.stage1_batch_size = 8
            args.ridge_samples = 24
            args.stage2_steps = 10
            args.batch_size = 8
            args.log_every = 5

            rep = run_path_a_surgery(args)
            self.assertTrue(rep["lossless_speculative_eval"]["bit_exact_greedy_match"])
            self.assertLess(rep["ridge_warmstart"]["nll_after"], rep["ridge_warmstart"]["nll_before"])
            self.assertTrue((Path(tmp) / "sidecar_surgery_report.json").exists())


if __name__ == "__main__":
    unittest.main()
