"""Physical Edge Hardware Benchmark & Fused INT4/INT8 ONNX Pipeline (`edge_bench.py`).

Addresses Review Items 2.1 & 2.2 (and 1.2a laptop fair ORT baseline):
  1. Exports KV-cached Exact-AR step (`exact_kv_step.onnx`), Concept level step
     (`concept_level_kv.onnx`), and Stage 1 Decoder (`stage1_decoder.onnx`) for `nano`.
  2. Quantizes exported graphs into:
       - INT8 Dynamic (fused MLAS INT8 kernels)
       - INT4 MatMulNBits (`block_size=32`, fused MLAS INT4 / DirectML INT4 kernels)
  3. Benchmarks on REAL physical edge hardware:
       - Laptop CPU (`CPUExecutionProvider`)
       - Laptop Intel Iris Xe iGPU (`DmlExecutionProvider` via `.venv_dml` worker)
       - Pure-PyTorch FP32 vs Pure-PyTorch `Int4PackedLinear` (baseline comparison)
  4. Reports TTFT (ms), decode tok/s, 64-tok latency (ms), ONNX file size (MB),
     peak RSS (MB), Windows thermal zone temperature (C), `% Processor Performance`
     (throttling), AC/battery state, and sustained thermal soak stability.

Usage:
    python -m cncm.export.edge_bench --preset nano --soak-sec 30
    python -m cncm.export.edge_bench --dml-worker --onnx-dir experiments/onnx_edge_nano
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Hardware & Thermal Telemetry (Windows non-admin perf counters + psutil)
# ---------------------------------------------------------------------------
def read_system_telemetry() -> Dict[str, Any]:
    """Read memory, battery/AC, Windows thermal zone (C), and CPU perf % without admin."""
    info: Dict[str, Any] = {}
    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process(os.getpid())
        mi = proc.memory_info()
        info["rss_mb"] = round(mi.rss / (1024 * 1024), 2)
        if hasattr(mi, "peak_wset"):
            info["peak_rss_mb"] = round(mi.peak_wset / (1024 * 1024), 2)
        if hasattr(mi, "private"):
            info["private_mb"] = round(mi.private / (1024 * 1024), 2)
        freq = psutil.cpu_freq()
        if freq is not None:
            info["cpu_freq_mhz"] = round(float(freq.current), 1)
        batt = psutil.sensors_battery()
        if batt is not None:
            info["power_plugged"] = bool(batt.power_plugged)
            info["battery_percent"] = round(float(batt.percent), 1)
    except Exception:
        pass

    if "peak_rss_mb" not in info and platform.system() != "Windows":
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB; macOS reports bytes
            div = (1024 * 1024) if platform.system() == "Darwin" else 1024
            info["peak_rss_mb"] = round(ru / float(div), 2)
        except Exception:
            pass

    if platform.system() == "Windows":
        try:
            cmd = (
                "$tz = (Get-Counter '\\Thermal Zone Information(*)\\Temperature' -ErrorAction SilentlyContinue)"
                ".CounterSamples | Select-Object -First 1 -ExpandProperty CookedValue; "
                "$pp = (Get-Counter '\\Processor Information(_Total)\\% Processor Performance' -ErrorAction SilentlyContinue)"
                ".CounterSamples | Select-Object -First 1 -ExpandProperty CookedValue; "
                "Write-Output \"$tz,$pp\""
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                stderr=subprocess.DEVNULL,
                timeout=6,
            ).decode("utf-8", errors="ignore").strip()
            parts = out.split(",")
            if len(parts) >= 1 and parts[0].strip():
                kelvin = float(parts[0].strip())
                info["thermal_zone_c"] = round(kelvin - 273.15, 1)
            if len(parts) >= 2 and parts[1].strip():
                info["cpu_perf_pct"] = round(float(parts[1].strip()), 1)
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# KV-Cached ONNX Wrappers (Fair Deployed Exact-AR & Concept Mode)
# ---------------------------------------------------------------------------
def _build_and_export_kv_onnx(preset: str, out_dir: Path, opset: int = 18) -> Dict[str, Any]:
    """Export KV-cached Exact-AR, KV-cached Concept level step, and Stage 1 Decoder."""
    import torch
    import torch.nn as nn

    from cncm.config import get_preset
    from cncm.export.export_onnx import Stage1DecoderONNXWrapper
    from cncm.layers import build_cached_concept_step_mask, build_causal_mask
    from cncm.model import CNCMModel
    from cncm.tokenizer_model import CNCMConceptTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_preset(preset, enc_layers=1)
    s1 = CNCMConceptTokenizer(cfg).eval()
    model = CNCMModel(cfg, s1).eval()
    n_eff = cfg.num_unique_layers * cfg.num_loops
    h_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    class ExactKVPrefillAndDecodeWrapper(nn.Module):
        """Takes input_ids [B, Q], position_ids [B, Q], and 2*N_eff past KV tensors."""

        def __init__(self, m: CNCMModel) -> None:
            super().__init__()
            self.m = m
            self.n_eff = m.cfg.num_unique_layers * m.cfg.num_loops

        def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, *past_flat: torch.Tensor):
            bsz, q_len = input_ids.shape
            past_len = past_flat[0].shape[2]
            kv_len = past_len + q_len
            past_kvs = [(past_flat[2 * i], past_flat[2 * i + 1]) for i in range(self.n_eff)]
            x = self.m.exact_embed(input_ids)
            attn_mask = build_causal_mask(
                q_len=q_len,
                kv_len=kv_len,
                past_len=past_len,
                batch_size=bsz,
                device=input_ids.device,
                dtype=x.dtype,
            )
            h, present_kvs = self.m._run_backbone(
                x,
                attn_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=past_kvs,
                use_cache=True,
                quantize_kv=False,
                commit_to_cache=True,
            )
            logits = self.m.exact_head(h)
            flat_present: List[torch.Tensor] = []
            for k_t, v_t in present_kvs:
                flat_present.append(k_t)
                flat_present.append(v_t)
            return (logits, *flat_present)

    class ConceptLevelKVWrapper(nn.Module):
        """Runs one coarse-to-fine concept level step against read-only prefix KV."""

        def __init__(self, m: CNCMModel) -> None:
            super().__init__()
            self.m = m
            self.n_eff = m.cfg.num_unique_layers * m.cfg.num_loops

        def forward(self, x_level: torch.Tensor, position_ids: torch.Tensor, *past_flat: torch.Tensor):
            bsz, l_k, _ = x_level.shape
            past_len = past_flat[0].shape[2]
            past_kvs = [(past_flat[2 * i], past_flat[2 * i + 1]) for i in range(self.n_eff)]
            step_mask = build_cached_concept_step_mask(
                prefix_len=past_len,
                level_len=l_k,
                batch_size=bsz,
                device=x_level.device,
                dtype=x_level.dtype,
            )
            h_k, _ = self.m._run_backbone(
                x_level,
                attn_mask=step_mask,
                position_ids=position_ids,
                past_key_values=past_kvs,
                use_cache=False,
                quantize_kv=False,
                commit_to_cache=False,
            )
            return h_k

    def _export_legacy(mod: nn.Module, args: Tuple[Any, ...], fpath: Path, in_names: List[str], out_names: List[str], dyn: Dict[str, Any]) -> None:
        torch.onnx.export(
            mod,
            args,
            str(fpath),
            input_names=in_names,
            output_names=out_names,
            dynamic_axes=dyn,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    past_in_names = []
    past_out_names = []
    past_dyn: Dict[str, Dict[int, str]] = {}
    sample_past: List[torch.Tensor] = []
    for i in range(n_eff):
        kn, vn = f"past_k_{i}", f"past_v_{i}"
        pkn, pvn = f"present_k_{i}", f"present_v_{i}"
        past_in_names.extend([kn, vn])
        past_out_names.extend([pkn, pvn])
        past_dyn[kn] = {0: "batch", 2: "past_len"}
        past_dyn[vn] = {0: "batch", 2: "past_len"}
        past_dyn[pkn] = {0: "batch", 2: "total_len"}
        past_dyn[pvn] = {0: "batch", 2: "total_len"}
        sample_past.append(torch.zeros((1, h_kv, 4, head_dim), dtype=torch.float32))
        sample_past.append(torch.zeros((1, h_kv, 4, head_dim), dtype=torch.float32))

    # 1. Exact-AR KV step
    exact_wrap = ExactKVPrefillAndDecodeWrapper(model).eval()
    exact_path = out_dir / "exact_kv_step_fp32.onnx"
    s_ids = torch.ones((1, 2), dtype=torch.long)
    s_pos = torch.tensor([[4, 5]], dtype=torch.long)
    _export_legacy(
        exact_wrap,
        (s_ids, s_pos, *sample_past),
        exact_path,
        ["input_ids", "position_ids", *past_in_names],
        ["logits", *past_out_names],
        {
            "input_ids": {0: "batch", 1: "q_len"},
            "position_ids": {0: "batch", 1: "q_len"},
            "logits": {0: "batch", 1: "q_len"},
            **past_dyn,
        },
    )

    # 2. Concept Level KV step
    concept_wrap = ConceptLevelKVWrapper(model).eval()
    concept_path = out_dir / "concept_level_kv_fp32.onnx"
    s_xk = torch.zeros((1, 8, cfg.hidden_size), dtype=torch.float32)
    s_cpos = torch.arange(4, 12, dtype=torch.long).unsqueeze(0)
    c_dyn = {
        "x_level": {0: "batch", 1: "level_len"},
        "position_ids": {0: "batch", 1: "level_len"},
        "h_level": {0: "batch", 1: "level_len"},
    }
    for i in range(n_eff):
        c_dyn[f"past_k_{i}"] = {0: "batch", 2: "past_len"}
        c_dyn[f"past_v_{i}"] = {0: "batch", 2: "past_len"}
    _export_legacy(
        concept_wrap,
        (s_xk, s_cpos, *sample_past),
        concept_path,
        ["x_level", "position_ids", *past_in_names],
        ["h_level"],
        c_dyn,
    )

    # 3. Stage 1 Decoder
    s1_wrap = Stage1DecoderONNXWrapper(model.concept_tokenizer).eval()
    s1_path = out_dir / "stage1_decoder_fp32.onnx"
    s_flat = torch.zeros((1, sum(cfg.levels)), dtype=torch.long)
    _export_legacy(
        s1_wrap,
        (s_flat,),
        s1_path,
        ["flat_codes"],
        ["logits"],
        {"flat_codes": {0: "batch"}, "logits": {0: "batch"}},
    )

    meta = {
        "preset": preset,
        "n_eff_layers": n_eff,
        "num_kv_heads": h_kv,
        "head_dim": head_dim,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "chunk_size": cfg.chunk_size,
        "levels": list(cfg.levels),
        "sum_levels": sum(cfg.levels),
        "params": model.count_parameters(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _quantize_onnx_graphs(out_dir: Path) -> Dict[str, Any]:
    """Produce INT8 Dynamic, INT8 QDQ (static), and INT4 MatMulNBits quantized ONNX graphs."""
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    n_eff = int(meta["n_eff_layers"])
    h_kv = int(meta["num_kv_heads"])
    head_dim = int(meta["head_dim"])
    hidden = int(meta["hidden_size"])
    sum_levels = int(meta["sum_levels"])
    vocab_size = int(meta["vocab_size"])

    class _SyntheticCalibReader(CalibrationDataReader):
        def __init__(self, stem_name: str, n_samples: int = 8) -> None:
            rng = np.random.default_rng(123)
            self._items: List[Dict[str, np.ndarray]] = []
            for _ in range(n_samples):
                past = {
                    f"past_{kv}_{i}": np.zeros((1, h_kv, 4, head_dim), dtype=np.float32)
                    for i in range(n_eff)
                    for kv in ("k", "v")
                }
                if stem_name == "exact_kv_step":
                    self._items.append(
                        {
                            "input_ids": rng.integers(16, vocab_size, size=(1, 2), dtype=np.int64),
                            "position_ids": np.array([[4, 5]], dtype=np.int64),
                            **past,
                        }
                    )
                elif stem_name == "concept_level_kv":
                    self._items.append(
                        {
                            "x_level": rng.standard_normal((1, 8, hidden)).astype(np.float32),
                            "position_ids": np.arange(4, 12, dtype=np.int64).reshape(1, 8),
                            **past,
                        }
                    )
                else:
                    self._items.append(
                        {"flat_codes": np.zeros((1, sum_levels), dtype=np.int64)}
                    )
            self._iter = iter(self._items)

        def get_next(self) -> Optional[Dict[str, np.ndarray]]:
            return next(self._iter, None)

        def rewind(self) -> None:
            self._iter = iter(self._items)

    stems = ["exact_kv_step", "concept_level_kv", "stage1_decoder"]
    sizes_mb: Dict[str, Dict[str, float]] = {"fp32": {}, "int8": {}, "int8_qdq": {}, "int4_nbits": {}}
    quant_notes: Dict[str, str] = {}

    for stem in stems:
        fp32_p = out_dir / f"{stem}_fp32.onnx"
        int8_p = out_dir / f"{stem}_int8.onnx"
        int8_qdq_p = out_dir / f"{stem}_int8_qdq.onnx"
        int4_p = out_dir / f"{stem}_int4_nbits.onnx"

        sizes_mb["fp32"][stem] = round(fp32_p.stat().st_size / (1024 * 1024), 3)

        # INT8 Dynamic (fused MLAS INT8)
        try:
            quantize_dynamic(
                model_input=str(fp32_p),
                model_output=str(int8_p),
                weight_type=QuantType.QUInt8,
            )
            sizes_mb["int8"][stem] = round(int8_p.stat().st_size / (1024 * 1024), 3)
        except Exception as exc:
            quant_notes[f"{stem}_int8"] = f"FAILED: {type(exc).__name__}: {exc}"

        # INT8 QDQ Static (Item 2.2: QuantFormat.QDQ for CPU/DirectML/NNAPI)
        try:
            quantize_static(
                model_input=str(fp32_p),
                model_output=str(int8_qdq_p),
                calibration_data_reader=_SyntheticCalibReader(stem),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
            )
            sizes_mb["int8_qdq"][stem] = round(int8_qdq_p.stat().st_size / (1024 * 1024), 3)
        except Exception as exc:
            quant_notes[f"{stem}_int8_qdq"] = f"FAILED: {type(exc).__name__}: {exc}"

        # INT4 MatMulNBits (block_size=32, fused MLAS/DirectML INT4 kernel!)
        try:
            model_proto = onnx.load(str(fp32_p))
            quant = MatMulNBitsQuantizer(
                model=model_proto,
                bits=4,
                block_size=32,
                is_symmetric=True,
            )
            quant.process()
            quant.model.save_model_to_file(str(int4_p))
            sizes_mb["int4_nbits"][stem] = round(int4_p.stat().st_size / (1024 * 1024), 3)
        except Exception as exc:
            quant_notes[f"{stem}_int4"] = f"FAILED: {type(exc).__name__}: {exc}"

    return {"file_sizes_mb": sizes_mb, "quant_notes": quant_notes}


# ---------------------------------------------------------------------------
# ONNX Runtime Benchmark Runner (Works for both CPUExecutionProvider & DmlExecutionProvider)
# ---------------------------------------------------------------------------
def run_ort_benchmark(
    onnx_dir: Path,
    provider: str,
    precisions: Sequence[str] = ("fp32", "int8", "int8_qdq", "int4_nbits"),
    prefix_len: int = 64,
    gen_tokens: int = 64,
    warmup: int = 1,
    iters: int = 3,
    soak_sec: float = 0.0,
) -> Dict[str, Any]:
    """Run real KV-cached Exact-AR and 8-level Concept Mode benchmarks on an ORT provider."""
    import onnxruntime as ort

    meta = json.loads((onnx_dir / "meta.json").read_text(encoding="utf-8"))
    n_eff = int(meta["n_eff_layers"])
    h_kv = int(meta["num_kv_heads"])
    head_dim = int(meta["head_dim"])
    hidden = int(meta["hidden_size"])
    levels = [int(x) for x in meta["levels"]]
    sum_levels = int(meta["sum_levels"])

    avail = ort.get_available_providers()
    if provider not in avail:
        return {"provider": provider, "available": False, "available_providers": avail}

    def _make_sess(path: Path) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "CPUExecutionProvider":
            so.intra_op_num_threads = min(8, os.cpu_count() or 4)
        return ort.InferenceSession(str(path), sess_options=so, providers=[provider])

    results: Dict[str, Any] = {
        "provider": provider,
        "ort_version": ort.__version__,
        "prefix_len": prefix_len,
        "gen_tokens": gen_tokens,
        "precisions": {},
    }

    rng = np.random.default_rng(42)
    prefix_ids = rng.integers(16, int(meta["vocab_size"]), size=(1, prefix_len), dtype=np.int64)
    prefix_pos = np.arange(prefix_len, dtype=np.int64).reshape(1, prefix_len)
    empty_past = {
        f"past_{kv}_{i}": np.zeros((1, h_kv, 0, head_dim), dtype=np.float32)
        for i in range(n_eff)
        for kv in ("k", "v")
    }
    level_inputs = [
        (
            rng.standard_normal((1, l_k, hidden)).astype(np.float32),
            (prefix_len + np.floor((np.arange(l_k) + 0.5) * (gen_tokens / l_k)).astype(np.int64)).reshape(1, l_k),
        )
        for l_k in levels
    ]
    sample_flat_codes = np.zeros((1, sum_levels), dtype=np.int64)

    for prec in precisions:
        ep = onnx_dir / f"exact_kv_step_{prec}.onnx"
        cp = onnx_dir / f"concept_level_kv_{prec}.onnx"
        sp = onnx_dir / f"stage1_decoder_{prec}.onnx"
        if not (ep.exists() and cp.exists() and sp.exists()):
            results["precisions"][prec] = {"skipped": "missing onnx file"}
            continue

        try:
            sess_exact = _make_sess(ep)
            sess_concept = _make_sess(cp)
            sess_s1 = _make_sess(sp)
        except Exception as exc:
            results["precisions"][prec] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        def _run_exact_ar_once() -> Tuple[float, float]:
            """Returns (ttft_ms, total_ms) for prefix prefill + gen_tokens KV-cached steps."""
            t0 = time.perf_counter()
            feed = {"input_ids": prefix_ids, "position_ids": prefix_pos, **empty_past}
            outs = sess_exact.run(None, feed)
            logits = outs[0]
            next_tok = np.argmax(logits[:, -1:, :], axis=-1).astype(np.int64)
            ttft_ms = (time.perf_counter() - t0) * 1000.0
            kv_feed = {}
            for i in range(n_eff):
                kv_feed[f"past_k_{i}"] = outs[1 + 2 * i]
                kv_feed[f"past_v_{i}"] = outs[1 + 2 * i + 1]
            for step in range(gen_tokens - 1):
                pos = np.array([[prefix_len + step]], dtype=np.int64)
                outs = sess_exact.run(None, {"input_ids": next_tok, "position_ids": pos, **kv_feed})
                next_tok = np.argmax(outs[0][:, -1:, :], axis=-1).astype(np.int64)
                for i in range(n_eff):
                    kv_feed[f"past_k_{i}"] = outs[1 + 2 * i]
                    kv_feed[f"past_v_{i}"] = outs[1 + 2 * i + 1]
            total_ms = (time.perf_counter() - t0) * 1000.0
            return ttft_ms, total_ms

        def _run_concept_once() -> Tuple[float, float]:
            """Returns (prefill_ms, total_chunk_ms) for prefix prefill + K=8 levels + Stage 1 decode."""
            t0 = time.perf_counter()
            feed = {"input_ids": prefix_ids, "position_ids": prefix_pos, **empty_past}
            outs = sess_exact.run(None, feed)
            prefill_ms = (time.perf_counter() - t0) * 1000.0
            kv_feed = {}
            for i in range(n_eff):
                kv_feed[f"past_k_{i}"] = outs[1 + 2 * i]
                kv_feed[f"past_v_{i}"] = outs[1 + 2 * i + 1]
            for x_k, pos_k in level_inputs:
                _ = sess_concept.run(None, {"x_level": x_k, "position_ids": pos_k, **kv_feed})
            _ = sess_s1.run(None, {"flat_codes": sample_flat_codes})
            total_ms = (time.perf_counter() - t0) * 1000.0
            return prefill_ms, total_ms

        try:
            for _ in range(warmup):
                _run_exact_ar_once()
                _run_concept_once()

            ar_ttfts, ar_totals = [], []
            for _ in range(iters):
                ttft, tot = _run_exact_ar_once()
                ar_ttfts.append(ttft)
                ar_totals.append(tot)

            c_prefills, c_totals = [], []
            for _ in range(iters):
                pf, tot = _run_concept_once()
                c_prefills.append(pf)
                c_totals.append(tot)

            telem = read_system_telemetry()
            ar_mean_ms = float(np.mean(ar_totals))
            ar_ttft_ms = float(np.mean(ar_ttfts))
            c_mean_ms = float(np.mean(c_totals))
            c_prefill_ms = float(np.mean(c_prefills))

            prec_entry: Dict[str, Any] = {
                "exact_ar_ttft_ms": round(ar_ttft_ms, 2),
                "exact_ar_64tok_ms": round(ar_mean_ms, 2),
                "exact_ar_tok_per_sec": round(gen_tokens / (ar_mean_ms / 1000.0), 2),
                "concept_prefill_ms": round(c_prefill_ms, 2),
                "concept_ttft_64tok_chunk_ms": round(c_mean_ms, 2),
                "concept_64tok_ms": round(c_mean_ms, 2),
                "concept_tok_per_sec": round(gen_tokens / (c_mean_ms / 1000.0), 2),
                "concept_speedup_vs_exact_ar": round(ar_mean_ms / max(c_mean_ms, 1e-6), 3),
                "telemetry": telem,
            }

            # Optional sustained thermal soak on the fastest quantized precision
            if soak_sec > 0.0 and prec == "int4_nbits":
                soak_start = time.perf_counter()
                t_start_telem = read_system_telemetry()
                chunk_times_ms: List[float] = []
                while (time.perf_counter() - soak_start) < soak_sec:
                    _, cms = _run_concept_once()
                    chunk_times_ms.append(cms)
                t_end_telem = read_system_telemetry()
                n_chunks = len(chunk_times_ms)
                first_q = float(np.mean(chunk_times_ms[: max(1, n_chunks // 4)]))
                last_q = float(np.mean(chunk_times_ms[max(0, n_chunks - max(1, n_chunks // 4)) :]))
                prec_entry["thermal_soak"] = {
                    "duration_sec": round(time.perf_counter() - soak_start, 2),
                    "chunks_completed": n_chunks,
                    "first_quarter_tok_per_sec": round(gen_tokens / (first_q / 1000.0), 2),
                    "last_quarter_tok_per_sec": round(gen_tokens / (last_q / 1000.0), 2),
                    "throttle_ratio_last_over_first": round(first_q / max(last_q, 1e-6), 3),
                    "start_telemetry": t_start_telem,
                    "end_telemetry": t_end_telem,
                }

            results["precisions"][prec] = prec_entry
        except Exception as exc:
            results["precisions"][prec] = {"error": f"{type(exc).__name__}: {exc}"}

        del sess_exact, sess_concept, sess_s1
        gc.collect()

    return results


def run_pytorch_cpu_reference(preset: str, prefix_len: int = 64, gen_tokens: int = 64) -> Dict[str, Any]:
    """Compare pure-PyTorch FP32 vs pure-PyTorch Int4PackedLinear on laptop CPU."""
    import torch

    from cncm.config import get_preset
    from cncm.export import measure_model_size, quantize_model_int4
    from cncm.model import CNCMModel
    from cncm.tokenizer_model import CNCMConceptTokenizer

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    cfg = get_preset(preset, enc_layers=1)
    model_fp32 = CNCMModel(cfg, CNCMConceptTokenizer(cfg)).eval()
    model_int4 = quantize_model_int4(model_fp32, group_size=32, quantize_embeddings=True).eval()

    pfx = torch.randint(16, cfg.vocab_size, (1, prefix_len), dtype=torch.long)

    @torch.no_grad()
    def _bench(m: CNCMModel) -> Dict[str, float]:
        _ = m.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)
        t0 = time.perf_counter()
        for _ in range(2):
            _ = m.generate_concept_chunk(pfx, temperature=0.0, early_exit_conf=None)
        c_ms = ((time.perf_counter() - t0) / 2.0) * 1000.0

        t0 = time.perf_counter()
        _ = m.generate_exact(pfx, max_new_tokens=gen_tokens, temperature=0.0)
        ar_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "concept_64tok_ms": round(c_ms, 2),
            "concept_tok_per_sec": round(gen_tokens / (c_ms / 1000.0), 2),
            "exact_ar_64tok_ms": round(ar_ms, 2),
            "exact_ar_tok_per_sec": round(gen_tokens / (ar_ms / 1000.0), 2),
            "speedup": round(ar_ms / max(c_ms, 1e-6), 3),
        }

    fp32_r = _bench(model_fp32)
    int4_r = _bench(model_int4)
    return {
        "pytorch_fp32": fp32_r,
        "pytorch_fp32_mb": round(measure_model_size(model_fp32)["total_mb"], 2),
        "pytorch_int4_dequant": int4_r,
        "pytorch_int4_mb": round(measure_model_size(model_int4)["total_mb"], 2),
        "pytorch_int4_vs_fp32_speed_ratio": round(
            fp32_r["concept_64tok_ms"] / max(int4_r["concept_64tok_ms"], 1e-6), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical Edge ONNX Benchmark (CPU + DirectML iGPU)")
    parser.add_argument("--preset", type=str, default="nano")
    parser.add_argument("--onnx-dir", type=str, default="experiments/onnx_edge_nano")
    parser.add_argument("--soak-sec", type=float, default=20.0)
    parser.add_argument("--full-soak", action="store_true", help="Run full 5-minute (300s) thermal soak per Item 2.1")
    parser.add_argument("--dml-worker", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    if args.full_soak:
        args.soak_sec = 300.0
    onnx_dir = Path(args.onnx_dir)
    if args.dml_worker:
        res = run_ort_benchmark(
            onnx_dir=onnx_dir,
            provider="DmlExecutionProvider",
            precisions=("fp32", "int4_nbits"),
            iters=3,
            soak_sec=min(args.soak_sec, 15.0),
        )
        print(json.dumps(res))
        return

    print(f"[EDGE BENCH] Exporting KV-cached ONNX graphs for preset={args.preset} -> {onnx_dir}")
    if not args.skip_export:
        meta = _build_and_export_kv_onnx(args.preset, onnx_dir, opset=18)
        quant_info = _quantize_onnx_graphs(onnx_dir)
    else:
        meta = json.loads((onnx_dir / "meta.json").read_text(encoding="utf-8"))
        quant_info = {"skipped_export": True}

    print("[EDGE BENCH] Running Pure-PyTorch CPU reference (FP32 vs Int4PackedLinear)...")
    pt_ref = run_pytorch_cpu_reference(args.preset)
    print(f"  PyTorch FP32 concept={pt_ref['pytorch_fp32']['concept_64tok_ms']}ms | "
          f"PyTorch INT4(dequant)={pt_ref['pytorch_int4_dequant']['concept_64tok_ms']}ms "
          f"(ratio={pt_ref['pytorch_int4_vs_fp32_speed_ratio']}x)")

    print("[EDGE BENCH] Running ORT CPUExecutionProvider (FP32 / INT8 / INT4 MatMulNBits + thermal soak)...")
    cpu_ort = run_ort_benchmark(
        onnx_dir=onnx_dir,
        provider="CPUExecutionProvider",
        precisions=("fp32", "int8", "int8_qdq", "int4_nbits"),
        iters=3,
        soak_sec=args.soak_sec,
    )

    dml_ort: Dict[str, Any] = {"skipped": ".venv_dml not found"}
    dml_py = REPO_ROOT / ".venv_dml" / "Scripts" / "python.exe"
    if dml_py.exists():
        print("[EDGE BENCH] Running ORT DmlExecutionProvider on Intel Iris Xe iGPU via .venv_dml...")
        try:
            out = subprocess.check_output(
                [
                    str(dml_py),
                    str(Path(__file__).resolve()),
                    "--dml-worker",
                    "--onnx-dir",
                    str(onnx_dir),
                    "--soak-sec",
                    str(min(args.soak_sec, 15.0)),
                ],
                cwd=str(REPO_ROOT),
                stderr=subprocess.PIPE,
                timeout=300,
            ).decode("utf-8", errors="ignore")
            # Parse last JSON line
            dml_ort = json.loads(out.strip().splitlines()[-1])
        except Exception as exc:
            dml_ort = {"error": f"{type(exc).__name__}: {exc}"}

    from experiments.harness import save_result

    report = {
        "preset": args.preset,
        "meta": meta,
        "onnx_quantization": quant_info,
        "pytorch_cpu_reference": pt_ref,
        "ort_cpu": cpu_ort,
        "ort_directml_igpu": dml_ort,
    }
    out_path = save_result(f"edge_bench_{args.preset}", report)
    print(f"[EDGE BENCH] Saved -> {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
