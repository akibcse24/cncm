"""GPU Fair-Baseline Benchmark: torch.compile + CUDA Graphs vs CNCM Concept Mode (Review Item 1.2a).

Designed for NVIDIA A10 / A10G / L40S GPUs (Tier B — run when GPU budget is approved).
Also supports `--dry-run-cpu` to validate script syntax and eager path on CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def benchmark_gpu_compile(preset: str = "nano", gen_tokens: int = 64, dry_run_cpu: bool = False) -> Dict[str, Any]:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from cncm.config import get_preset
    from cncm.model import CNCMModel
    from cncm.tokenizer_model import CNCMConceptTokenizer

    use_cuda = torch.cuda.is_available() and not dry_run_cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cfg = get_preset(preset, enc_layers=1)
    model = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).to(device).eval()

    pfx = torch.randint(16, cfg.vocab_size, (1, 64), device=device)

    def _sync():
        if use_cuda:
            torch.cuda.synchronize()

    @torch.no_grad()
    def _time_fn(fn, warmup: int = 3, iters: int = 8) -> float:
        for _ in range(warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync()
        return ((time.perf_counter() - t0) / iters) * 1000.0

    eager_ar_ms = _time_fn(lambda: model.generate_exact(pfx, max_new_tokens=gen_tokens, temperature=0.0), warmup=1, iters=3)
    eager_concept_ms = _time_fn(lambda: model.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None), warmup=1, iters=3)

    compiled_ar_ms = None
    compiled_note = "CPU dry-run: torch.compile max-autotune CUDA graphs skipped"
    if use_cuda:
        try:
            compiled_step = torch.compile(model.forward_exact, mode="max-autotune", fullgraph=False)

            @torch.no_grad()
            def _run_compiled_ar():
                out = compiled_step(pfx, use_cache=True)
                kv = out["past_key_values"]
                tok = out["logits"][:, -1:, :].argmax(dim=-1)
                for _ in range(gen_tokens - 1):
                    out = compiled_step(tok, past_key_values=kv, use_cache=True)
                    kv = out["past_key_values"]
                    tok = out["logits"][:, -1:, :].argmax(dim=-1)
                return tok

            compiled_ar_ms = _time_fn(_run_compiled_ar, warmup=3, iters=8)
            compiled_note = "CUDA torch.compile(mode='max-autotune')"
        except Exception as exc:
            compiled_note = f"compile_error: {type(exc).__name__}: {exc}"

    return {
        "preset": preset,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if use_cuda else "CPU",
        "eager_exact_ar_64tok_ms": round(eager_ar_ms, 2),
        "eager_concept_64tok_ms": round(eager_concept_ms, 2),
        "eager_speedup": round(eager_ar_ms / max(eager_concept_ms, 1e-6), 3),
        "compiled_exact_ar_64tok_ms": round(compiled_ar_ms, 2) if compiled_ar_ms is not None else None,
        "speedup_vs_compiled_ar": round(compiled_ar_ms / max(eager_concept_ms, 1e-6), 3)
        if compiled_ar_ms is not None
        else None,
        "note": compiled_note,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="nano")
    p.add_argument("--dry-run-cpu", action="store_true")
    args = p.parse_args()
    print(json.dumps(benchmark_gpu_compile(args.preset, dry_run_cpu=args.dry_run_cpu), indent=2))
