"""Controlled Modal Cloud GPU A/B Benchmark: `ctx0an` Base-MoE vs. Previous Dense Base.

Runs a strictly controlled, apples-to-apples benchmark on Modal NVIDIA A10/L40S/A10G GPU
comparing:
  - Baseline A (Previous Method): CNCM `base` Dense SwiGLU (93.90M total / 93.90M active)
  - Candidate B (New Method):     CNCM `base_moe` DeepSeek-Style Fine-Grained MoE
                                  (221.45M total / 94.05M active, 1 Shared + 8 Routed Top-2,
                                   24 per-loop routers, Switch/DeepSeek load balancing)

Both models share:
  - Identical curated 4-domain corpus (Textbooks + Python + Math CoT + Tool JSON)
  - Identical document-level train/val split (zero leakage)
  - Identical corpus-trained Byte-Level BPE tokenizer (V=6400, individual digits)
  - Identical uint16 MemmapTokenDataset
  - Identical trained Stage 1 strictly causal M-RFSQ Concept Tokenizer (enc_layers=1, Matryoshka)
  - Identical step budgets, batch size, prefix length, LR schedule, and active FFN width (2304)

Outputs:
  - `modal_ctx0an_moe_benchmark.json` (raw structured GPU metrics)

Usage:
    modal run modal_moe_benchmark.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/root/project"

app = modal.App("ctx0an-moe-vs-dense-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "PYTHONPATH": REMOTE_PROJECT_DIR,
        }
    )
    .pip_install("torch>=2.5.0", "numpy>=1.26.0", "datasets>=2.19.0", "tokenizers>=0.19.0")
    .add_local_dir(
        str(PROJECT_ROOT / "cncm"),
        remote_path=f"{REMOTE_PROJECT_DIR}/cncm",
        ignore=["__pycache__", "*.pyc", "*.pyo"],
    )
    .add_local_file(
        str(PROJECT_ROOT / "train_ctx0an_base.py"),
        remote_path=f"{REMOTE_PROJECT_DIR}/train_ctx0an_base.py",
    )
)


@app.function(
    image=image,
    gpu=["A10", "A10G", "L40S"],
    timeout=900,
)
def run_moe_vs_dense_benchmark() -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, REMOTE_PROJECT_DIR)
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.text_tokenizer import CNCMTextTokenizer
    from cncm.tokenizer_model import CNCMConceptTokenizer
    from cncm.trainer import (
        MemmapTokenDataset,
        PackedChunkDataset,
        measure_code_entropy,
        measure_concept_vs_exact_gap,
        measure_per_domain_stage1_recon,
        measure_tokenizer_fertility,
        prepare_memmap_corpus,
        set_seed,
        train_stage1,
        train_stage2,
    )
    from train_ctx0an_base import _load_ctx0an_curated_corpora

    t0_all = time.perf_counter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available()
        else 0.0
    )
    amp_dtype = "bf16" if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else "fp16"
    print(f"[MODAL BENCHMARK] GPU={gpu_name} ({vram_gb:.1f} GB) | AMP={amp_dtype}")

    set_seed(42)
    work_dir = Path("/tmp/ctx0an_bench")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load real curated 4-domain corpus (identical split for both Dense & MoE)
    t_data0 = time.perf_counter()
    train_domains, val_domains, _, corpus_sources = _load_ctx0an_curated_corpora(
        target_en=2400,
        target_py=2000,
        target_math=1000,
        target_json=1000,
        val_fraction=0.10,
        allow_hf_download=True,
        seed=42,
    )
    all_train = (
        train_domains["english"]
        + train_domains["python"]
        + train_domains["math"]
        + train_domains["json"]
    )
    rng = random.Random(42)
    rng.shuffle(all_train)
    all_val = (
        val_domains["english"]
        + val_domains["python"]
        + val_domains["math"]
        + val_domains["json"]
    )
    data_load_sec = time.perf_counter() - t_data0
    print(
        f"[DATA] Loaded {len(all_train):,} train docs / {len(all_val):,} val docs in {data_load_sec:.1f}s | "
        f"sources={corpus_sources}"
    )

    # 2. Train shared Byte-Level BPE tokenizer (V=6400, individual digits)
    base_cfg_s1 = get_preset("base", enc_layers=1, matryoshka_cutoffs=(4, 6, 7, 8))
    bpe_path = work_dir / "ctx0an_bench_bpe.json"
    tokenizer = CNCMTextTokenizer.train_bpe(
        all_train,
        vocab_size=base_cfg_s1.vocab_size,
        save_path=bpe_path,
        min_frequency=2,
    )
    val_domains_3 = {
        "english": val_domains["english"][:200],
        "python": val_domains["python"][:200],
        "json": val_domains["json"][:200],
    }
    gate4_fertility = measure_tokenizer_fertility(tokenizer, val_domains_3)
    print(f"[GATE #4 BPE] passed={gate4_fertility['gate_all_passed']} | {gate4_fertility['domains']}")

    # 3. Build shared uint16 MemmapTokenDatasets
    train_bin = prepare_memmap_corpus(
        all_train,
        tokenizer,
        work_dir / "train.bin",
        eos_token_id=base_cfg_s1.eos_token_id,
        bos_token_id=base_cfg_s1.bos_token_id,
        dtype=np.uint16,
    )
    val_bin = prepare_memmap_corpus(
        all_val,
        tokenizer,
        work_dir / "val.bin",
        eos_token_id=base_cfg_s1.eos_token_id,
        bos_token_id=base_cfg_s1.bos_token_id,
        dtype=np.uint16,
    )
    s1_train_ds = MemmapTokenDataset(train_bin, chunk_size=base_cfg_s1.chunk_size, prefix_len=0)
    prefix_len = 256
    s2_train_ds = MemmapTokenDataset(train_bin, chunk_size=base_cfg_s1.chunk_size, prefix_len=prefix_len)
    s2_val_ds = MemmapTokenDataset(val_bin, chunk_size=base_cfg_s1.chunk_size, prefix_len=prefix_len)
    print(
        f"[MEMMAP] train_tokens={s2_train_ds.num_tokens:,} ({len(s2_train_ds):,} windows) | "
        f"val_tokens={s2_val_ds.num_tokens:,} ({len(s2_val_ds):,} windows)"
    )

    # Per-domain validation datasets for granular domain-level NLL evaluation
    domain_val_datasets: Dict[str, PackedChunkDataset] = {}
    for dname in ("english", "python", "math", "json"):
        domain_val_datasets[dname] = PackedChunkDataset(
            val_domains[dname][:160],
            tokenizer,
            chunk_size=base_cfg_s1.chunk_size,
            prefix_len=prefix_len,
            eos_token_id=base_cfg_s1.eos_token_id,
            bos_token_id=base_cfg_s1.bos_token_id,
        )

    # 4. Train shared Stage 1 Causal M-RFSQ Tokenizer (FP32-safe) ONCE
    print("\n=== [Shared Stage 1] Training Causal M-RFSQ Tokenizer (600 steps, FP32) ===")
    t_s1 = time.perf_counter()
    s1_res = train_stage1(
        cfg=base_cfg_s1,
        tokenizer=tokenizer,
        dataset=s1_train_ds,
        steps=600,
        batch_size=64,
        lr=2.5e-3,
        device=dev,
        apply_fine_drop=True,
        amp_dtype=None,
        seed=42,
        log_every=150,
        verbose=True,
    )
    s1_elapsed = time.perf_counter() - t_s1
    s1_state = {k: v.detach().cpu().clone() for k, v in s1_res["model"].state_dict().items()}
    gates23_recon = measure_per_domain_stage1_recon(
        s1_model=s1_res["model"].eval(),
        tokenizer=tokenizer,
        domain_texts_dict=val_domains_3,
        device=dev,
        max_chunks_per_domain=48,
    )
    print(
        f"[STAGE 1 DONE] {s1_elapsed:.1f}s | k8_pass={gates23_recon['gate_k_full_all_passed']} | "
        f"k7_pass={gates23_recon['gate_k_prev_all_passed']}"
    )

    @torch.no_grad()
    def eval_domain_exact_nll(model: CNCMModel, ds: Any, max_batches: int = 12, bs: int = 8) -> float:
        model.eval()
        total_nll = 0.0
        total_tok = 0
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=False)
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            x = batch["exact_input_ids"].to(dev)
            mask = batch["exact_attention_mask"].to(dev)
            labels = x.clone().masked_fill(~mask.bool(), -100)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=(torch.bfloat16 if amp_dtype == "bf16" else torch.float16),
                enabled=(dev.type == "cuda"),
            ):
                out = model.forward_exact(x, labels=None, attention_mask=mask)
                logits = out["logits"][:, :-1, :].float()
                shift_lbl = labels[:, 1:]
                valid = shift_lbl != -100
                if valid.any():
                    nll = F.cross_entropy(
                        logits[valid],
                        shift_lbl[valid],
                        reduction="sum",
                    )
                    total_nll += float(nll.item())
                    total_tok += int(valid.sum().item())
        return total_nll / float(max(1, total_tok))

    @torch.no_grad()
    def inspect_moe_routing(model: CNCMModel) -> Optional[Dict[str, Any]]:
        if model.cfg.num_experts <= 0 or model.step_moe_routers is None:
            return None
        model.eval()
        num_e = model.cfg.num_experts
        top_k = model.cfg.num_experts_per_tok
        num_u = model.cfg.num_unique_layers
        num_loops = model.cfg.num_loops

        # Capture router assignments per effective layer across domains
        domain_counts: Dict[str, torch.Tensor] = {}
        loop0_top1_all: List[torch.Tensor] = []
        loop1_top1_all: List[torch.Tensor] = []

        for dname in ("english", "python", "math", "json"):
            ds = domain_val_datasets[dname]
            xs = torch.stack([ds[i]["exact_input_ids"] for i in range(min(8, len(ds)))], dim=0).to(dev)
            pos_ids = torch.arange(xs.shape[1], device=dev, dtype=torch.long)
            cos, sin = model.rope(pos_ids, dtype=model.exact_embed.weight.dtype)
            from cncm.layers import build_causal_mask

            mask = build_causal_mask(
                q_len=xs.shape[1],
                kv_len=xs.shape[1],
                past_len=0,
                batch_size=xs.shape[0],
                device=dev,
                dtype=model.exact_embed.weight.dtype,
            )
            h = model.exact_embed(xs)
            counts = torch.zeros(num_e, dtype=torch.float64, device=dev)
            l0_t1: List[torch.Tensor] = []
            l1_t1: List[torch.Tensor] = []

            eff_idx = 0
            for loop_i in range(num_loops):
                for layer_idx in range(num_u):
                    blk = model.blocks[layer_idx]
                    rtr = model.step_moe_routers[eff_idx]
                    adp = model.loop_adapters[eff_idx] if model.loop_adapters is not None else None
                    # Compute attention output first, then inspect router input h_ffn
                    h_attn, _ = blk.attn(
                        model.step_attn_norms[eff_idx](h),
                        cos=cos,
                        sin=sin,
                        attn_mask=mask,
                    )
                    h_mid = h + h_attn
                    h_ffn_in = model.step_ffn_norms[eff_idx](h_mid)
                    r_logits = F.linear(h_ffn_in.reshape(-1, h.shape[-1]).float(), rtr.weight.float())
                    r_probs = F.softmax(r_logits, dim=-1)
                    _, tk_idx = torch.topk(r_probs, k=top_k, dim=-1)
                    one_hot = F.one_hot(tk_idx, num_classes=num_e).to(torch.float64).sum(dim=(0, 1))
                    counts += one_hot
                    if loop_i == 0:
                        l0_t1.append(tk_idx[:, 0].detach().cpu())
                    elif loop_i == 1:
                        l1_t1.append(tk_idx[:, 0].detach().cpu())
                    mlp_out, _ = blk.mlp(h_ffn_in, router=rtr)
                    h = h_mid + mlp_out
                    if adp is not None:
                        h = adp(h)
                    eff_idx += 1

            domain_counts[dname] = (counts / counts.sum().clamp(min=1.0)).detach().cpu()
            if l0_t1 and l1_t1:
                loop0_top1_all.append(torch.cat(l0_t1))
                loop1_top1_all.append(torch.cat(l1_t1))

        overall = torch.stack(list(domain_counts.values()), dim=0).mean(dim=0)
        ent = float(-(overall * (overall.clamp(min=1e-12)).log()).sum().item())
        norm_ent = ent / math.log(float(num_e))

        loop_switch_rate = 0.0
        if loop0_top1_all and loop1_top1_all:
            a0 = torch.cat(loop0_top1_all)
            a1 = torch.cat(loop1_top1_all)
            loop_switch_rate = float((a0 != a1).float().mean().item())

        # L1 distance between Python and English expert distributions (measures domain specialization)
        py_en_tv = 0.5 * float((domain_counts["python"] - domain_counts["english"]).abs().sum().item())
        py_json_tv = 0.5 * float((domain_counts["python"] - domain_counts["json"]).abs().sum().item())

        return {
            "num_experts": num_e,
            "top_k": top_k,
            "overall_expert_shares": [round(float(x), 4) for x in overall.tolist()],
            "min_expert_share": round(float(overall.min().item()), 4),
            "max_expert_share": round(float(overall.max().item()), 4),
            "std_expert_share": round(float(overall.std().item()), 4),
            "normalized_load_entropy": round(norm_ent, 4),
            "active_experts_gt_2pct": int((overall > 0.02).sum().item()),
            "loop0_to_loop1_expert_switch_rate": round(loop_switch_rate, 4),
            "domain_tv_python_vs_english": round(py_en_tv, 4),
            "domain_tv_python_vs_json": round(py_json_tv, 4),
            "per_domain_shares": {
                k: [round(float(x), 4) for x in v.tolist()] for k, v in domain_counts.items()
            },
        }

    @torch.no_grad()
    def benchmark_inference(model: CNCMModel) -> Dict[str, Any]:
        model.eval()
        prompt = "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([ids], dtype=torch.long, device=dev)

        # Warmup
        _ = model.generate_exact(x, max_new_tokens=16, temperature=0.0)
        _ = model.generate_concept_chunk(x, temperature=0.0)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Measure 64-token Exact-AR
        t0 = time.perf_counter()
        ex_out = model.generate_exact(x, max_new_tokens=64, temperature=0.0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        exact_ms = (time.perf_counter() - t0) * 1000.0

        # Measure 64-token Concept Mode (8 parallel passes)
        t1 = time.perf_counter()
        cc_out = model.generate_concept_chunk(x, temperature=0.0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        concept_ms = (time.perf_counter() - t1) * 1000.0

        # Sample generations
        sample_py_exact = prompt + tokenizer.decode(ex_out["token_ids"][0].detach().cpu().tolist(), skip_special_tokens=False)
        sample_py_concept = prompt + tokenizer.decode(cc_out["token_ids"][0].detach().cpu().tolist(), skip_special_tokens=False)

        return {
            "exact_ar_64tok_ms": round(exact_ms, 2),
            "exact_ar_tok_per_sec": round(64.0 / (exact_ms / 1000.0), 1),
            "concept_64tok_ms": round(concept_ms, 2),
            "concept_tok_per_sec": round(64.0 / (concept_ms / 1000.0), 1),
            "concept_vs_exact_speedup": round(exact_ms / max(concept_ms, 1e-3), 2),
            "sample_python_exact": sample_py_exact[:320],
            "sample_python_concept": sample_py_concept[:320],
        }

    def run_variant(label: str, preset_name: str, p1_steps: int, p2_steps: int) -> Dict[str, Any]:
        print(f"\n================ [VARIANT: {label} ({preset_name})] ================")
        set_seed(100)
        cfg = get_preset(preset_name, enc_layers=1, matryoshka_cutoffs=(4, 6, 7, 8))
        s1_copy = CNCMConceptTokenizer(cfg)
        s1_copy.load_state_dict(s1_state, strict=True)
        s1_copy = s1_copy.to(dev).eval()
        model = CNCMModel(cfg, s1_copy).to(dev)
        pstats = model.count_parameters()
        print(
            f"[{label}] Total={pstats['total']/1e6:.2f}M | Active={pstats['active_total']/1e6:.2f}M | "
            f"Experts={cfg.num_experts} (Top-{cfg.num_experts_per_tok})"
        )

        # Phase 1: Pure Exact-AR Foundation
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_p1 = time.perf_counter()
        p1_res = train_stage2(
            cfg=cfg,
            tokenizer=tokenizer,
            concept_tokenizer=s1_copy,
            dataset=s2_train_ds,
            mode="exact_ar",
            steps=p1_steps,
            batch_size=12,
            grad_accum_steps=2,
            lr=5.0e-4,
            device=dev,
            amp_dtype=amp_dtype,
            grad_checkpointing=False,
            num_workers=0,
            model=model,
            seed=100,
            log_every=max(1, p1_steps // 4),
            verbose=True,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize()
        p1_sec = time.perf_counter() - t_p1
        p1_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0
        p1_tok_s = p1_res["metrics"]["tokens_seen"] / max(p1_sec, 1e-6)

        p1_val_domains = {
            d: round(eval_domain_exact_nll(model, domain_val_datasets[d]), 4)
            for d in ("english", "python", "math", "json")
        }
        p1_val_mean = round(sum(p1_val_domains.values()) / len(p1_val_domains), 4)
        print(f"[{label} Phase 1 Val Exact NLL] mean={p1_val_mean:.4f} | {p1_val_domains} | {p1_tok_s:,.0f} tok/s | peak={p1_peak_gb:.2f} GB")

        # Phase 2: Joint Dual-Mode + L_align
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_p2 = time.perf_counter()
        p2_res = train_stage2(
            cfg=cfg,
            tokenizer=tokenizer,
            concept_tokenizer=s1_copy,
            dataset=s2_train_ds,
            mode="joint",
            steps=p2_steps,
            corrupt_rate=0.5,
            concept_weight=1.0,
            exact_weight=0.35,
            align_weight=0.35,
            supervise_prefix_exact=True,
            batch_size=12,
            grad_accum_steps=2,
            lr=3.5e-4,
            device=dev,
            amp_dtype=amp_dtype,
            grad_checkpointing=False,
            num_workers=0,
            model=model,
            seed=101,
            log_every=max(1, p2_steps // 4),
            verbose=True,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize()
        p2_sec = time.perf_counter() - t_p2
        p2_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0
        p2_tok_s = p2_res["metrics"]["tokens_seen"] / max(p2_sec, 1e-6)

        # Held-out post-Phase-2 domain Exact-AR NLL + Gates #1 & #5 + MoE Routing + Inference
        p2_val_domains = {
            d: round(eval_domain_exact_nll(model, domain_val_datasets[d]), 4)
            for d in ("english", "python", "math", "json")
        }
        p2_val_mean = round(sum(p2_val_domains.values()) / len(p2_val_domains), 4)
        p2_val_ppl = round(math.exp(min(20.0, p2_val_mean)), 2)

        gate1_entropy = measure_code_entropy(
            model=model.eval(),
            val_dataset=s2_val_ds,
            device=dev,
            max_batches=12,
            batch_size=8,
            gate_threshold_nats=5.5,
        )
        gate5_gap = measure_concept_vs_exact_gap(
            model=model.eval(),
            val_dataset=s2_val_ds,
            device=dev,
            max_batches=12,
            batch_size=8,
            gate_gap_threshold_nats=0.40,
        )
        moe_diag = inspect_moe_routing(model)
        infer_bench = benchmark_inference(model)

        del model, s1_copy
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "label": label,
            "preset": preset_name,
            "params": pstats,
            "fp32_size_mb": round(pstats["total"] * 4 / (1024 * 1024), 2),
            "int4_est_size_mb": round(pstats["total"] * 0.52 / (1024 * 1024), 2),
            "phase1_exact_ar": {
                "steps": p1_steps,
                "tokens_seen": int(p1_res["metrics"]["tokens_seen"]),
                "elapsed_sec": round(p1_sec, 2),
                "tok_per_sec": round(p1_tok_s, 1),
                "peak_vram_gb": round(p1_peak_gb, 3),
                "train_initial_loss": round(float(p1_res["metrics"]["initial_loss"]), 4),
                "train_final_loss": round(float(p1_res["metrics"]["final_loss"]), 4),
                "all_grads_finite": bool(p1_res["metrics"]["all_grads_finite"]),
                "val_exact_nll_mean": p1_val_mean,
                "val_exact_ppl_mean": round(math.exp(min(20.0, p1_val_mean)), 2),
                "val_exact_nll_per_domain": p1_val_domains,
            },
            "phase2_joint_align": {
                "steps": p2_steps,
                "tokens_seen": int(p2_res["metrics"]["tokens_seen"]),
                "elapsed_sec": round(p2_sec, 2),
                "tok_per_sec": round(p2_tok_s, 1),
                "peak_vram_gb": round(p2_peak_gb, 3),
                "train_initial_loss": round(float(p2_res["metrics"]["initial_loss"]), 4),
                "train_final_loss": round(float(p2_res["metrics"]["final_loss"]), 4),
                "train_final_concept_loss": round(float(p2_res["metrics"]["concept_loss_history"][-1]), 4),
                "train_final_exact_loss": round(float(p2_res["metrics"]["exact_loss_history"][-1]), 4),
                "train_final_align_loss": round(float(p2_res["metrics"]["align_loss_history"][-1]), 4),
                "all_grads_finite": bool(p2_res["metrics"]["all_grads_finite"]),
                "val_exact_nll_mean": p2_val_mean,
                "val_exact_ppl_mean": p2_val_ppl,
                "val_exact_nll_per_domain": p2_val_domains,
            },
            "gate1_code_entropy": gate1_entropy,
            "gate5_concept_vs_exact_gap": gate5_gap,
            "moe_routing_diagnostics": moe_diag,
            "inference_benchmark": infer_bench,
        }

    # Run Controlled A/B: 200 Phase-1 steps + 200 Phase-2 steps each (~3.1M Stage 2 tokens/variant)
    dense_res = run_variant("Previous Dense Base (SwiGLU)", "base", p1_steps=200, p2_steps=200)
    moe_res = run_variant("New ctx0an Base-MoE (1 Shared + 8 Routed Top-2)", "base_moe", p1_steps=200, p2_steps=200)

    # Compute Head-to-Head Deltas (negative delta = MoE better NLL/entropy/gap)
    d_p1_nll = round(
        moe_res["phase1_exact_ar"]["val_exact_nll_mean"] - dense_res["phase1_exact_ar"]["val_exact_nll_mean"], 4
    )
    d_p2_nll = round(
        moe_res["phase2_joint_align"]["val_exact_nll_mean"] - dense_res["phase2_joint_align"]["val_exact_nll_mean"], 4
    )
    d_py_nll = round(
        moe_res["phase2_joint_align"]["val_exact_nll_per_domain"]["python"]
        - dense_res["phase2_joint_align"]["val_exact_nll_per_domain"]["python"],
        4,
    )
    d_code_ent = round(
        moe_res["gate1_code_entropy"]["mean_conditional_nats_per_code"]
        - dense_res["gate1_code_entropy"]["mean_conditional_nats_per_code"],
        4,
    )
    d_align_nll = round(
        moe_res["gate5_concept_vs_exact_gap"]["align_nll"]
        - dense_res["gate5_concept_vs_exact_gap"]["align_nll"],
        4,
    )
    d_concept_draft = round(
        moe_res["gate5_concept_vs_exact_gap"]["concept_draft_nll_by_exact"]
        - dense_res["gate5_concept_vs_exact_gap"]["concept_draft_nll_by_exact"],
        4,
    )

    wins = {
        "phase1_val_exact_nll_better": d_p1_nll < 0.0,
        "phase2_val_exact_nll_better": d_p2_nll < 0.0,
        "python_val_nll_better": d_py_nll < 0.0,
        "concept_code_entropy_better": d_code_ent < 0.0,
        "align_nll_better": d_align_nll < 0.0,
        "zero_expert_collapse": bool(
            moe_res["moe_routing_diagnostics"]
            and moe_res["moe_routing_diagnostics"]["active_experts_gt_2pct"] == 8
        ),
    }

    report = {
        "benchmark_title": "ctx0an Base-MoE vs. Previous Dense Base — Controlled Modal Cloud GPU A/B Benchmark",
        "hardware": {
            "gpu_name": gpu_name,
            "vram_gb": round(vram_gb, 2),
            "amp_dtype": amp_dtype,
        },
        "shared_setup": {
            "corpus_sources": corpus_sources,
            "train_docs": len(all_train),
            "val_docs": len(all_val),
            "memmap_train_tokens": s2_train_ds.num_tokens,
            "memmap_val_tokens": s2_val_ds.num_tokens,
            "prefix_len": prefix_len,
            "chunk_size": base_cfg_s1.chunk_size,
            "vocab_size": base_cfg_s1.vocab_size,
            "gate4_bpe_fertility": gate4_fertility,
            "stage1_elapsed_sec": round(s1_elapsed, 2),
            "stage1_final_loss": round(float(s1_res["metrics"]["final_loss"]), 4),
            "stage1_final_recon_acc": round(float(s1_res["metrics"]["final_recon_acc"]), 4),
            "gates23_stage1_recon": gates23_recon,
        },
        "dense_base": dense_res,
        "ctx0an_base_moe": moe_res,
        "comparison_deltas_moe_minus_dense": {
            "phase1_val_exact_nll_delta": d_p1_nll,
            "phase2_val_exact_nll_delta": d_p2_nll,
            "python_val_exact_nll_delta": d_py_nll,
            "conditional_code_entropy_delta": d_code_ent,
            "align_nll_delta": d_align_nll,
            "concept_draft_nll_delta": d_concept_draft,
            "phase1_throughput_ratio_moe_over_dense": round(
                moe_res["phase1_exact_ar"]["tok_per_sec"] / max(1.0, dense_res["phase1_exact_ar"]["tok_per_sec"]), 3
            ),
            "phase2_throughput_ratio_moe_over_dense": round(
                moe_res["phase2_joint_align"]["tok_per_sec"] / max(1.0, dense_res["phase2_joint_align"]["tok_per_sec"]), 3
            ),
        },
        "verdict_checks": wins,
        "moe_wins_count": sum(1 for v in wins.values() if v),
        "total_checks": len(wins),
        "total_elapsed_sec": round(time.perf_counter() - t0_all, 2),
    }

    s1_train_ds.close()
    s2_train_ds.close()
    s2_val_ds.close()
    return report


@app.local_entrypoint()
def main() -> None:
    report = run_moe_vs_dense_benchmark.remote()
    out_path = PROJECT_ROOT / "modal_ctx0an_moe_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[SAVED] Raw Modal GPU benchmark written to: {out_path}")
    print("VERDICT CHECKS:", json.dumps(report["verdict_checks"], indent=2))
    print("DELTAS (MoE - Dense):", json.dumps(report["comparison_deltas_moe_minus_dense"], indent=2))
