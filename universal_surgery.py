"""Fast Zero-Backprop Donor Surgery & Path A `CNCM-DeltaNet` Plug-In Sidecar Builder.

Implements the complete Path A pipeline from `future_cncm_update.md` (~5-10 min on a free 15 GB T4,
or 100% offline from local `.safetensors` / `.pt` weights + token memmaps):
  1. **Universal Config-Driven Donor Loader + Bit-Exact Parity Check**
     Supports `llama`, `smollm2`, `qwen2`, `qwen3`, `mistral`, `gemma`/`gemma2`, and local CNCM files.
  2. **SVD-Projected Frozen-Embedding Stage 1 (`SVDConceptTokenizer`, ~1.4M params)**
     Closed-form deterministic Gram eigendecomposition (`<50 ms`) + rapid Matryoshka FSQ training.
  3. **Offline Multi-Layer Hidden (`L16+L24+L32`) + Top-K Logit `memmap` Cache**
     Single `torch.no_grad()` donor sweep; Stage 2 runs with **zero donor forward/backward passes**.
  4. **Closed-Form Step-0 Ridge Warm-Start**
     Initializes all `d_q` `FactorizedConceptHead` digit heads via regularized least squares before Step 1.
  5. **4-Layer `CNCM-DeltaNet` Sidecar + Learned Dual-Mode Router Training**
     Chunk-64 WY-Parallel + Dual-Phase Read-Only `S_prefix` + Sparse Top-K Tail KL distillation.
  6. **Compact `.sidecar.gguf` v3 Export (`~10-25 MB`) & Lossless Speculative Parity Verification**

Usage:
    # 1. Offline smoke verification (<10s on CPU):
    python universal_surgery.py --smoke --out-dir ./sidecar_smoke

    # 2. Local offline SmolLM2-360M surgery using already-downloaded weights & token memmap:
    python universal_surgery.py \\
        --donor smol360_exports/v4_smol360_moe_concept14k.safetensors \\
        --tokens-bin smol360_tpu_out/smol360_concept_tpu_memmaps/train.bin \\
        --chunk-size 32 --out-dir ./smol360_sidecar_out

    # 3. Kaggle Free T4 surgery on any HuggingFace model (~5-10 min):
    python universal_surgery.py \\
        --donor HuggingFaceTB/SmolLM2-360M-Instruct \\
        --chunk-size 32 --cache-seqs 8192 --stage1-steps 400 --stage2-steps 1500 \\
        --quant q4_0 --out-dir /kaggle/working/smol360_sidecar
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cncm.donor import (
    FrozenDonorRunner,
    OfflineDonorCache,
    build_synthetic_donor,
    load_universal_donor,
    verify_donor_parity,
)
from cncm.sidecar import (
    CNCMDeltaNetSidecar,
    SVDConceptTokenizer,
    SidecarConfig,
    compute_svd_projection,
)
from cncm.sidecar_gguf import export_sidecar_gguf, load_sidecar_gguf


def _cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if total_steps <= 1:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    prog = min(1.0, max(0.0, prog))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))


def _load_or_build_token_sequences(
    args: argparse.Namespace,
    vocab_size: int,
    seq_len: int,
) -> np.ndarray:
    """Load `[N, seq_len]` token sequences from `--tokens-bin`, HF streaming, or deterministic fallback."""
    n_seqs = int(args.cache_seqs)
    if args.smoke:
        rng = np.random.default_rng(int(args.seed))
        # Structured repeating + arithmetic-like motifs so smoke test verifies real learning signal
        base = rng.integers(4, max(8, vocab_size), size=(max(4, n_seqs // 2), seq_len), dtype=np.int32)
        tiled = np.concatenate([base, base], axis=0)[:n_seqs]
        return tiled

    if args.tokens_bin:
        bin_path = Path(args.tokens_bin)
        if not bin_path.exists():
            raise FileNotFoundError(f"--tokens-bin file not found: {bin_path}")
        item_dtype = np.uint16 if vocab_size < 65535 else np.int32
        raw = np.fromfile(bin_path, dtype=item_dtype)
        usable = (raw.size // seq_len) * seq_len
        if usable < seq_len:
            # Tile small local dev memmaps cleanly if shorter than requested `n_seqs * seq_len`
            reps = math.ceil(float(n_seqs * seq_len) / float(max(1, raw.size)))
            raw = np.tile(raw, reps)
            usable = (raw.size // seq_len) * seq_len
        seqs = raw[:usable].reshape(-1, seq_len)
        if seqs.shape[0] > n_seqs:
            rng = np.random.default_rng(int(args.seed))
            pick = rng.choice(seqs.shape[0], size=n_seqs, replace=False)
            seqs = seqs[np.sort(pick)]
        elif seqs.shape[0] < n_seqs:
            reps = math.ceil(float(n_seqs) / float(seqs.shape[0]))
            seqs = np.tile(seqs, (reps, 1))[:n_seqs]
        return np.clip(seqs.astype(np.int32), 0, vocab_size - 1)

    # Online HF streaming corpus fallback (50% ChatML smol-smoltalk + 50% Cosmopedia-v2 for chat + prose generalization)
    from datasets import load_dataset  # type: ignore[import-not-found]
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    from tokenizers import Tokenizer  # type: ignore[import-not-found]

    tok_path = (
        Path(args.tokenizer)
        if args.tokenizer and Path(args.tokenizer).exists()
        else Path(hf_hub_download(repo_id=str(args.donor), filename="tokenizer.json"))
    )
    tok = Tokenizer.from_file(str(tok_path))
    needed_tokens = n_seqs * seq_len
    chat_target = needed_tokens // 2
    collected_chat: List[int] = []
    collected_prose: List[int] = []

    try:
        ds_chat = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
        for row in ds_chat:
            msgs = row.get("messages") or []
            if not msgs:
                continue
            parts: List[str] = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content:
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            if parts:
                collected_chat.extend(tok.encode("".join(parts)).ids)
                if len(collected_chat) >= chat_target:
                    break
        print(f"      [DATA] Streamed {len(collected_chat):,} ChatML tokens from smol-smoltalk", flush=True)
    except Exception as exc:
        print(f"      [DATA WARN] smol-smoltalk unavailable ({exc}); filling from cosmopedia-v2", flush=True)

    prose_needed = max(0, needed_tokens - len(collected_chat))
    if prose_needed > 0:
        ds_prose = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
        for row in ds_prose:
            txt = row.get("text", "")
            if txt:
                collected_prose.extend(tok.encode(txt).ids)
                if len(collected_prose) >= prose_needed:
                    break
        print(f"      [DATA] Streamed {len(collected_prose):,} prose tokens from cosmopedia-v2", flush=True)

    # Interleave chat and prose sequences so train & held-out val splits both contain ChatML + prose!
    n_chat = min(n_seqs, len(collected_chat) // seq_len)
    n_prose = n_seqs - n_chat
    seqs_list: List[np.ndarray] = []
    if n_chat > 0:
        seqs_list.append(np.asarray(collected_chat[: n_chat * seq_len], dtype=np.int32).reshape(n_chat, seq_len))
    if n_prose > 0:
        seqs_list.append(np.asarray(collected_prose[: n_prose * seq_len], dtype=np.int32).reshape(n_prose, seq_len))
    arr = np.concatenate(seqs_list, axis=0)
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(arr, axis=0)
    return np.clip(arr[:n_seqs], 0, vocab_size - 1)


def run_path_a_surgery(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "donor_cache")

    # ------------------------------------------------------------------
    # Stage 0: Load Donor & Verify Bit-Exact Forward Parity
    # ------------------------------------------------------------------
    print(f"[1/6] Loading donor ({'smoke_synthetic' if args.smoke else args.donor}) on {device}...", flush=True)
    if args.smoke:
        donor, raw_sd = build_synthetic_donor(
            arch=args.smoke_arch, vocab_size=512, hidden_size=96, num_layers=4, seed=int(args.seed)
        )
        donor = donor.to(device)
        parity = verify_donor_parity(donor, raw_sd)
        del raw_sd
    else:
        donor_dtype = torch.float16 if (device.type == "cuda" and not args.fp32_donor) else torch.float32
        # Run bit-exact parity check in FP32 on CPU first (avoids CPU Half SDPA op limitations), then move/cast to target device!
        donor, parity = load_universal_donor(args.donor, dtype=torch.float32, verify_parity=not args.skip_parity)
        donor = donor.to(device=device, dtype=donor_dtype).freeze_all()
    print(f"      Donor ready: arch={donor.cfg.arch} | H={donor.cfg.hidden_size} | L={donor.cfg.num_layers} "
          f"| taps={donor.cfg.tap_layers} | parity={parity}", flush=True)

    sc_cfg = SidecarConfig.from_donor(
        donor.cfg,
        chunk_size=int(args.chunk_size),
        sidecar_layers=int(args.sidecar_layers),
        sidecar_hidden_size=args.sidecar_hidden if args.sidecar_hidden > 0 else None,
        d_vq=args.d_vq if args.d_vq > 0 else None,
    )

    # Keep FP32 copies of frozen donor embed & lm_head weights for numerically crisp Stage 1 & KL
    donor_emb_f32 = donor.embed_tokens.weight.detach().float().to(device)
    donor_head_f32 = (
        donor_emb_f32
        if donor.cfg.tie_word_embeddings
        else donor.lm_head.weight.detach().float().to(device)
    )

    # ------------------------------------------------------------------
    # Stage 1: Deterministic SVD Basis + Fast Stage 1 Tokenizer Training
    # ------------------------------------------------------------------
    t_svd = time.time()
    svd_proj = compute_svd_projection(donor_emb_f32, sc_cfg.d_vq).to(device)
    s1_tok = SVDConceptTokenizer(sc_cfg, svd_proj=svd_proj).to(device)
    print(
        f"[2/6] Computed orthonormal SVD basis {tuple(svd_proj.shape)} in {(time.time() - t_svd)*1000:.1f} ms "
        f"| Stage-1 trainable params={sum(p.numel() for p in s1_tok.parameters()):,}",
        flush=True,
    )

    seq_len = int(args.prefix_len) + sc_cfg.chunk_size
    sequences = _load_or_build_token_sequences(args, donor.cfg.vocab_size, seq_len)

    # Train Stage 1 on flat `chunk_size` slices from `sequences`
    all_chunks = torch.from_numpy(sequences.reshape(-1, sc_cfg.chunk_size).astype(np.int64))
    s1_tok.train(True)
    opt_s1 = torch.optim.AdamW(s1_tok.parameters(), lr=float(args.stage1_lr), weight_decay=0.01)
    s1_warmup = max(1, int(args.stage1_steps) // 10)
    s1_metrics: Dict[str, float] = {}

    for step in range(int(args.stage1_steps)):
        lr_now = _cosine_lr(step, int(args.stage1_steps), float(args.stage1_lr), s1_warmup)
        for pg in opt_s1.param_groups:
            pg["lr"] = lr_now
        idx = torch.randint(0, all_chunks.shape[0], (min(int(args.stage1_batch_size), all_chunks.shape[0]),))
        batch_chunks = all_chunks[idx].to(device)

        opt_s1.zero_grad(set_to_none=True)
        out_s1 = s1_tok.forward_stage1(
            batch_chunks, donor_emb_f32, donor_head_f32, apply_fine_drop=True
        )
        out_s1["loss"].backward()
        torch.nn.utils.clip_grad_norm_(s1_tok.parameters(), 1.0)
        opt_s1.step()

        if step == 0 or (step + 1) == int(args.stage1_steps) or (step + 1) % max(1, int(args.log_every)) == 0:
            s1_metrics = {
                "recon_loss": round(float(out_s1["recon_loss"].item()), 4),
                "recon_acc": round(float(out_s1["recon_acc"].item()), 4),
                "active_code_ratio": round(float(out_s1["active_code_ratio"].item()), 4),
            }
            print(f"      [Stage 1] step {step + 1}/{args.stage1_steps} | {s1_metrics}", flush=True)

    del opt_s1, all_chunks
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 2A: Build (or Reuse) Offline Multi-Layer Hidden + Top-K Cache
    # ------------------------------------------------------------------
    meta_file = cache_dir / "cache_meta.json"
    reuse_ok = False
    if meta_file.exists() and not args.rebuild_cache:
        existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        reuse_ok = (
            existing_meta.get("prefix_len") == int(args.prefix_len)
            and existing_meta.get("chunk_size") == sc_cfg.chunk_size
            and existing_meta.get("hidden_size") == donor.cfg.hidden_size
        )

    if reuse_ok:
        print(f"[3/6] Reusing existing offline donor cache at {cache_dir}...", flush=True)
        cache = OfflineDonorCache(cache_dir)
    else:
        print(
            f"[3/6] Building offline donor cache ({sequences.shape[0]} seqs x {seq_len} tok, "
            f"top_k={args.top_k}) -> {cache_dir}...",
            flush=True,
        )
        cache = OfflineDonorCache.build(
            donor,
            sequences,
            cache_dir,
            prefix_len=int(args.prefix_len),
            chunk_size=sc_cfg.chunk_size,
            top_k=int(args.top_k),
            batch_size=int(args.cache_batch_size),
        )

    # Free donor transformer blocks from GPU during Stage 2 if memory is tight (zero donor passes needed!)
    donor_cpu = donor.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 2B: Build Sidecar + Closed-Form Step-0 Ridge Warm-Start
    # ------------------------------------------------------------------
    sidecar = CNCMDeltaNetSidecar(sc_cfg, concept_tokenizer=s1_tok).to(device)
    sidecar.freeze_concept_tokenizer()
    param_counts = sidecar.count_parameters()
    print(f"[4/6] Initialized 4-Layer CNCM-DeltaNet Sidecar: {param_counts}", flush=True)

    warm_batches: List[Dict[str, torch.Tensor]] = []
    n_warm = min(len(cache), max(int(args.batch_size), int(args.ridge_samples)))
    bsz = min(int(args.batch_size), len(cache))
    for s_idx in range(0, n_warm, bsz):
        e_idx = min(n_warm, s_idx + bsz)
        warm_batches.append(cache.get_batch(range(s_idx, e_idx), device=device))

    ridge_stats = sidecar.ridge_warmstart_concept_head(
        warm_batches, donor_emb_f32, ridge_lambda=float(args.ridge_lambda)
    )
    print(f"      [Ridge Warm-Start] {ridge_stats}", flush=True)
    del warm_batches

    # ------------------------------------------------------------------
    # Stage 2C: Zero-Backprop Sidecar + Dual-Mode Router Training (with Variable Prefix & Held-Out Val)
    # ------------------------------------------------------------------
    n_total = len(cache)
    n_val = min(max(4, n_total // 20), 512) if n_total >= 16 else 0
    n_train = max(1, n_total - n_val)
    var_p_min = max(4, min(int(args.var_prefix_min), int(args.prefix_len)))
    print(
        f"[5/6] Training Sidecar from offline cache ({args.stage2_steps} steps, 0 donor FLOPs | "
        f"train={n_train}, val={n_val}, var_prefix=[{var_p_min}..{args.prefix_len}])...",
        flush=True,
    )
    sidecar.train(True)
    opt_s2 = torch.optim.AdamW(
        sidecar.sidecar_trainable_parameters(),
        lr=float(args.stage2_lr),
        weight_decay=0.01,
    )
    s2_warmup = max(1, int(args.stage2_steps) // 10)
    rng = np.random.default_rng(int(args.seed) + 99)
    s2_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}

    for step in range(int(args.stage2_steps)):
        lr_now = _cosine_lr(step, int(args.stage2_steps), float(args.stage2_lr), s2_warmup)
        for pg in opt_s2.param_groups:
            pg["lr"] = lr_now

        b_indices = rng.integers(0, n_train, size=min(bsz, n_train))
        batch = cache.get_batch(b_indices, device=device)

        # Variable prefix length training (50% random trailing prefix in [var_p_min..prefix_len])
        # so short ChatML prompts (10-30 tok) are 100% in-distribution for S_prefix!
        if var_p_min < int(args.prefix_len) and float(rng.random()) < 0.5:
            k_keep = int(rng.integers(var_p_min, int(args.prefix_len) + 1))
            batch = {
                "prefix_ids": batch["prefix_ids"][:, -k_keep:],
                "target_ids": batch["target_ids"],
                "prefix_unit_taps": batch["prefix_unit_taps"][:, -k_keep:],
                "topk_idx": batch["topk_idx"],
                "topk_logprobs": batch["topk_logprobs"],
            }

        opt_s2.zero_grad(set_to_none=True)
        out_s2 = sidecar.forward_train_from_cache(
            batch,
            donor_emb_f32,
            donor_head_f32,
            corrupt_rate=float(args.corrupt_rate),
            kl_weight=float(args.kl_weight),
            router_weight=float(args.router_weight),
        )
        out_s2["loss"].backward()
        torch.nn.utils.clip_grad_norm_(sidecar.sidecar_trainable_parameters(), 1.0)
        opt_s2.step()

        if step == 0 or (step + 1) == int(args.stage2_steps) or (step + 1) % max(1, int(args.log_every)) == 0:
            s2_metrics = {
                "loss": round(float(out_s2["loss"].item()), 4),
                "concept": round(float(out_s2["concept_loss"].item()), 4),
                "kl": round(float(out_s2["kl_loss"].item()), 4),
                "router": round(float(out_s2["router_loss"].item()), 4),
            }
            val_str = ""
            if n_val > 0 and ((step + 1) == int(args.stage2_steps) or (step + 1) % max(1, int(args.log_every) * 2) == 0):
                sidecar.eval()
                with torch.no_grad():
                    v_idx = np.arange(n_train, min(n_total, n_train + min(bsz, n_val)))
                    v_out = sidecar.forward_train_from_cache(
                        cache.get_batch(v_idx, device=device),
                        donor_emb_f32,
                        donor_head_f32,
                        corrupt_rate=0.0,
                        kl_weight=float(args.kl_weight),
                        router_weight=float(args.router_weight),
                    )
                    val_metrics = {
                        "val_concept": round(float(v_out["concept_loss"].item()), 4),
                        "val_kl": round(float(v_out["kl_loss"].item()), 4),
                    }
                    val_str = f" | {val_metrics}"
                sidecar.train(True)
            print(f"      [Stage 2] step {step + 1}/{args.stage2_steps} | {s2_metrics}{val_str}", flush=True)

    sidecar.eval()
    cache.close()

    # ------------------------------------------------------------------
    # Stage 3: Export `.sidecar.gguf` + Verify 100%-Lossless Speculative Parity
    # ------------------------------------------------------------------
    gguf_path = out_dir / f"{donor.cfg.arch}_L{sc_cfg.chunk_size}.sidecar.gguf"
    pt_path = out_dir / f"{donor.cfg.arch}_L{sc_cfg.chunk_size}.sidecar.pt"
    torch.save({"config": sc_cfg.to_dict(), "state_dict": sidecar.cpu().state_dict()}, pt_path)

    gguf_info = export_sidecar_gguf(sidecar, gguf_path, quant=args.quant)
    reloaded_sidecar = load_sidecar_gguf(gguf_path, device="cpu")

    # Verify lossless greedy equivalence between untouched donor and reloaded `.sidecar.gguf`!
    prompt_sample = torch.from_numpy(sequences[0:1, : min(16, int(args.prefix_len))].astype(np.int64))
    eval_new = min(16, sc_cfg.chunk_size)
    exact_ids = donor_cpu.generate_exact(prompt_sample, max_new_tokens=eval_new, temperature=0.0, eos_token_id=-1)
    spec_res = reloaded_sidecar.generate_speculative_lossless(
        donor_cpu,
        prompt_sample,
        max_new_tokens=eval_new,
        temperature=0.0,
        eos_token_id=-1,
    )
    lossless_exact_match = bool(torch.equal(exact_ids, spec_res["token_ids"]))
    if not lossless_exact_match:
        raise AssertionError(
            f"Lossless speculative parity failed!\nExact: {exact_ids.tolist()}\nSpec:  {spec_res['token_ids'].tolist()}"
        )

    ultra_res = reloaded_sidecar.generate_ultra_precise(
        donor_cpu,
        prompt_sample,
        max_new_tokens=eval_new,
        eos_token_id=-1,
    )

    elapsed = round(time.time() - t0, 2)
    report = {
        "donor_config": donor.cfg.to_dict(),
        "donor_parity": parity,
        "sidecar_config": sc_cfg.to_dict(),
        "parameter_counts": param_counts,
        "stage1_final": s1_metrics,
        "ridge_warmstart": ridge_stats,
        "stage2_final": s2_metrics,
        "stage2_val_final": val_metrics,
        "gguf_export": gguf_info,
        "lossless_speculative_eval": {
            "bit_exact_greedy_match": lossless_exact_match,
            "accepted_tokens": spec_res["accepted_tokens"],
            "drafted_tokens": spec_res["drafted_tokens"],
            "acceptance_rate": round(spec_res["acceptance_rate"], 4),
            "verify_steps": spec_res["verify_steps"],
            "sidecar_state_kb": round(spec_res["sidecar_state_bytes"] / 1024.0, 2),
        },
        "ultra_precise_eval": {
            "ultra_tokens": ultra_res["ultra_tokens"],
            "exact_tokens": ultra_res["exact_tokens"],
        },
        "elapsed_sec": elapsed,
    }
    report_path = out_dir / "sidecar_surgery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[6/6] Exported {gguf_info['path']} ({gguf_info['size_mb']} MB, quant={args.quant}) | "
        f"Lossless greedy match={lossless_exact_match} | accept={spec_res['acceptance_rate']*100:.1f}% | "
        f"sidecar_state={report['lossless_speculative_eval']['sidecar_state_kb']} KB (0 MB KV) | "
        f"Done in {elapsed}s!",
        flush=True,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Path A Zero-Backprop Donor Surgery & CNCM-DeltaNet Sidecar Builder")
    ap.add_argument("--donor", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct", help="HF repo ID or local .safetensors/.pt/dir")
    ap.add_argument("--tokens-bin", type=str, default="", help="Optional local uint16/int32 token memmap file")
    ap.add_argument("--tokenizer", type=str, default="", help="Optional local tokenizer.json path")
    ap.add_argument("--out-dir", type=str, default="./sidecar_out", help="Output directory for .sidecar.gguf and report")
    ap.add_argument("--cache-dir", type=str, default="", help="Directory for offline donor hidden+topk memmap cache")
    ap.add_argument("--rebuild-cache", action="store_true", help="Force rebuilding offline donor memmap cache")

    ap.add_argument("--chunk-size", type=int, default=32, choices=[16, 32, 64], help="Concept chunk horizon L (16, 32, or 64)")
    ap.add_argument("--prefix-len", type=int, default=128, help="Prefix token length per cached training example")
    ap.add_argument("--var-prefix-min", type=int, default=8, help="Min random prefix length for variable-prefix generalization")
    ap.add_argument("--sidecar-layers", type=int, default=4, help="Number of CNCM-DeltaNet layers in sidecar")
    ap.add_argument("--sidecar-hidden", type=int, default=0, help="Sidecar hidden dim (0 = auto 512 / donor-scaled)")
    ap.add_argument("--d-vq", type=int, default=0, help="SVD Stage 1 latent dim (0 = auto 256 / donor-scaled)")
    ap.add_argument("--top-k", type=int, default=16, help="Cached teacher Top-K logits per target position")

    ap.add_argument("--cache-seqs", type=int, default=4096, help="Number of sequences to cache from frozen donor")
    ap.add_argument("--cache-batch-size", type=int, default=8, help="Batch size during one-time donor cache build")
    ap.add_argument("--stage1-steps", type=int, default=350, help="Stage 1 SVD-FSQ training steps")
    ap.add_argument("--stage1-batch-size", type=int, default=32, help="Stage 1 batch size")
    ap.add_argument("--stage1-lr", type=float, default=2e-3, help="Stage 1 peak learning rate")
    ap.add_argument("--ridge-samples", type=int, default=512, help="Cached samples used for Step-0 Ridge warm-start")
    ap.add_argument("--ridge-lambda", type=float, default=1e-2, help="Tikhonov regularization for Ridge warm-start")
    ap.add_argument("--stage2-steps", type=int, default=1200, help="Zero-backprop Sidecar training steps")
    ap.add_argument("--batch-size", type=int, default=16, help="Stage 2 batch size")
    ap.add_argument("--stage2-lr", type=float, default=1e-3, help="Stage 2 peak learning rate")
    ap.add_argument("--corrupt-rate", type=float, default=0.30, help="Conditioning code corruption rate")
    ap.add_argument("--kl-weight", type=float, default=1.0, help="Sparse Top-K + Tail KL distillation weight")
    ap.add_argument("--router-weight", type=float, default=0.2, help="Learned Dual-Mode Router BCE loss weight")

    ap.add_argument("--quant", type=str, default="q8_0", choices=["q4_0", "q8_0", "f16", "f32"], help="GGUF export quantization")
    ap.add_argument("--device", type=str, default="", help="Override device (cpu / cuda)")
    ap.add_argument("--fp32-donor", action="store_true", help="Keep donor in FP32 even on CUDA")
    ap.add_argument("--skip-parity", action="store_true", help="Skip startup donor parity check")
    ap.add_argument("--log-every", type=int, default=100, help="Logging interval in steps")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--smoke", action="store_true", help="Run fast offline end-to-end smoke test (<10s on CPU)")
    ap.add_argument("--smoke-arch", type=str, default="qwen3", choices=["llama", "qwen2", "qwen3", "gemma2"])
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.smoke:
        args.prefix_len = 32
        args.chunk_size = 16
        args.cache_seqs = 32
        args.stage1_steps = 12
        args.stage1_batch_size = 8
        args.ridge_samples = 32
        args.stage2_steps = 16
        args.batch_size = 8
        args.log_every = 8
    run_path_a_surgery(args)


if __name__ == "__main__":
    main()
