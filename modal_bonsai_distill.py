"""Modal Cloud GPU Benchmark: Real `llama-server` (`Ternary-Bonsai-2-4B`) -> `distill_seqkd.py` -> `run_distill_ablation.py`.

Follows the exact requested two-stage pipeline on Modal Cloud GPU (with persistent Volume caching and
`scaledown_window=900` so the Modal container does not auto-shutdown prematurely):
  Stage A (Real Teacher Server + SeqKD JSONL Generation):
      1. Downloads PrismML's pre-built CUDA `llama-server` (`prism-b10828-a153697`) and
         `prism-ml/Ternary-Bonsai-2-4B-gguf` (`Ternary-Bonsai-2-4B-PQ2_0.gguf`, 1.15 GB) into
         the persistent Modal Volume `/vol/bonsai2_cache` (instant reuse across runs).
      2. Launches `llama-server -m Ternary-Bonsai-2-4B-PQ2_0.gguf -ngl 99 -fa on -c 8192 --port 8080`.
      3. Runs `python -m experiments.distill_seqkd --api-base http://127.0.0.1:8080/v1 --model Ternary-Bonsai-2-4B`
         to generate and sandbox-verify a real teacher JSONL (`verified_seqkd.jsonl` + `.meta.json`).
  Stage B (Single-Seed 4-Arm Hierarchical Distillation Ablation):
      4. Stops `llama-server` to free GPU VRAM and runs `run_hierarchical_distill_ablation`
         (`experiments/run_distill_ablation.py`) loading that exact `--seqkd-jsonl` file on single seed `42`!

Outputs saved locally:
  - `experiments/results/verified_seqkd.jsonl` (Real `Ternary-Bonsai-2-4B` verified completions)
  - `experiments/results/verified_seqkd.meta.json` (SHA-256 provenance & acceptance rate)
  - `modal_bonsai_distill_report.json` (Full 4-Arm + Pure-AR control metrics)
  - `experiments/results/distill_hierarchical_nano.json`

Usage:
    modal run modal_bonsai_distill.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = "/root/project"
VOLUME_NAME = "cncm-checkpoints"
VOLUME_MOUNT = "/vol"

app = modal.App("cncm-bonsai-2-4b-distill-benchmark")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

RELEASE_TAG = "prism-b10828-a153697"
LLAMA_BIN_URL_12_4 = (
    f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{RELEASE_TAG}/"
    f"llama-{RELEASE_TAG}-bin-linux-cuda-12.4-x64.tar.gz"
)
LLAMA_BIN_URL_CPU = (
    f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{RELEASE_TAG}/"
    f"llama-{RELEASE_TAG}-bin-ubuntu-x64.tar.gz"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "libgomp1")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": REMOTE_PROJECT_DIR,
        }
    )
    .pip_install("torch>=2.5.0", "numpy>=1.26.0", "huggingface_hub>=0.24.0", "hf_transfer>=0.1.8")
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


def _ensure_bonsai_llama_server_and_gguf(cache_dir: Path) -> Tuple[Path, Path]:
    """Download PrismML llama-server binary and Ternary-Bonsai-2-4B-PQ2_0.gguf into persistent Volume."""
    from huggingface_hub import hf_hub_download

    bin_dir = cache_dir / f"bin_{RELEASE_TAG}"
    llama_server = bin_dir / "llama-server"
    if not llama_server.exists():
        bin_dir.mkdir(parents=True, exist_ok=True)
        tar_path = cache_dir / "llama_bin.tar.gz"
        # Try CUDA 12.4 build first; fallback to Ubuntu x64 build if needed
        for url in (LLAMA_BIN_URL_12_4, LLAMA_BIN_URL_CPU):
            print(f"[BONSAI SETUP] Downloading PrismML llama-server from {url} ...")
            rc = subprocess.call(["curl", "-L", "--fail", "-sS", url, "-o", str(tar_path)])
            if rc == 0 and tar_path.exists() and tar_path.stat().st_size > 100_000:
                subprocess.check_call(["tar", "-xzf", str(tar_path), "-C", str(bin_dir), "--strip-components=1"])
                tar_path.unlink(missing_ok=True)
                for candidate in bin_dir.rglob("llama-server"):
                    candidate.chmod(0o755)
                    if candidate != llama_server:
                        llama_server = candidate
                    break
                if llama_server.exists():
                    break

    gguf_dir = cache_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / "Ternary-Bonsai-2-4B-PQ2_0.gguf"
    if not gguf_path.exists() or gguf_path.stat().st_size < 100_000_000:
        print("[BONSAI SETUP] Downloading prism-ml/Ternary-Bonsai-2-4B-gguf (Ternary-Bonsai-2-4B-PQ2_0.gguf) ...")
        downloaded = hf_hub_download(
            repo_id="prism-ml/Ternary-Bonsai-2-4B-gguf",
            filename="Ternary-Bonsai-2-4B-PQ2_0.gguf",
            local_dir=str(gguf_dir),
        )
        gguf_path = Path(downloaded)

    return llama_server, gguf_path


def _wait_for_llama_server(port: int = 8080, timeout_sec: float = 90.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


@app.function(
    image=image,
    gpu=["A10", "A10G", "L40S"],
    timeout=3600,
    scaledown_window=900,
    volumes={VOLUME_MOUNT: vol},
)
def run_bonsai_distill_on_modal(
    preset: str = "nano",
    n_seqkd_samples: int = 48,
    stage1_steps: int = 150,
    stage2_steps: int = 200,
    align_steps: int = 60,
    batch_size: int = 24,
) -> Dict[str, Any]:
    import torch

    os.chdir(REMOTE_PROJECT_DIR)
    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)

    from experiments.run_distill_ablation import run_hierarchical_distill_ablation

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"[MODAL BONSAI-2-4B PIPELINE] GPU={gpu_name} ({vram_gb} GB) | preset={preset} | "
        f"n_seqkd={n_seqkd_samples} | S1={stage1_steps} S2={stage2_steps} Align={align_steps}"
    )

    cache_dir = Path(VOLUME_MOUNT) / "bonsai2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seqkd_out = Path(REMOTE_PROJECT_DIR) / "experiments" / "results" / "verified_seqkd.jsonl"
    seqkd_meta = seqkd_out.with_suffix(".meta.json")

    # -------------------------------------------------------------------------
    # STAGE A: Start real llama-server (Ternary-Bonsai-2-4B) & run distill_seqkd.py
    # -------------------------------------------------------------------------
    llama_server_bin, gguf_path = _ensure_bonsai_llama_server_and_gguf(cache_dir)
    vol.commit()

    env = os.environ.copy()
    ld_dir = str(llama_server_bin.parent)
    env["LD_LIBRARY_PATH"] = f"{ld_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    server_cmd = [
        str(llama_server_bin),
        "-m",
        str(gguf_path),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-c",
        "4096",
        "--temp",
        "0.7",
        "--top-p",
        "0.8",
        "--top-k",
        "20",
        "--min-p",
        "0",
        "--presence-penalty",
        "1.5",
        "--repeat-penalty",
        "1.0",
        "--port",
        "8080",
    ]
    print(f"[LLAMA-SERVER] Launching: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        ready = _wait_for_llama_server(port=8080, timeout_sec=90.0)
        if not ready:
            # If CUDA 12.4 binary had a driver mismatch on the container, fall back to CPU llama-server binary
            print("[LLAMA-SERVER] CUDA binary health check timed out; checking logs & trying CPU binary fallback...")
            server_proc.terminate()
            try:
                out_log, _ = server_proc.communicate(timeout=5)
                print("[LLAMA-SERVER LOG]:", out_log[-1500:])
            except Exception:
                pass
            cpu_bin_dir = cache_dir / f"bin_cpu_{RELEASE_TAG}"
            cpu_bin_dir.mkdir(parents=True, exist_ok=True)
            tar_cpu = cache_dir / "llama_cpu.tar.gz"
            subprocess.check_call(["curl", "-L", "--fail", "-sS", LLAMA_BIN_URL_CPU, "-o", str(tar_cpu)])
            subprocess.check_call(["tar", "-xzf", str(tar_cpu), "-C", str(cpu_bin_dir), "--strip-components=1"])
            tar_cpu.unlink(missing_ok=True)
            cpu_server = next(cpu_bin_dir.rglob("llama-server"))
            cpu_server.chmod(0o755)
            env["LD_LIBRARY_PATH"] = f"{cpu_server.parent}:{env.get('LD_LIBRARY_PATH', '')}"
            server_cmd[0] = str(cpu_server)
            server_proc = subprocess.Popen(server_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            ready = _wait_for_llama_server(port=8080, timeout_sec=90.0)
            if not ready:
                out_log, _ = server_proc.communicate(timeout=5)
                raise RuntimeError(f"llama-server failed to start:\n{out_log}")

        print("[DISTILL_SEQKD] Running `python -m experiments.distill_seqkd --api-base http://127.0.0.1:8080/v1` ...")
        seqkd_cmd = [
            sys.executable,
            "-m",
            "experiments.distill_seqkd",
            "--api-base",
            "http://127.0.0.1:8080/v1",
            "--model",
            "Ternary-Bonsai-2-4B",
            "--teacher-name",
            "prism-ml/Ternary-Bonsai-2-4B-gguf",
            "--teacher-license",
            "Apache-2.0",
            "--n-samples",
            str(n_seqkd_samples),
            "--max-think-chars",
            "240",
            "--output",
            str(seqkd_out),
        ]
        subprocess.check_call(seqkd_cmd, cwd=REMOTE_PROJECT_DIR)
    finally:
        print("[LLAMA-SERVER] Stopping teacher server to release GPU VRAM for CNCM student training...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except Exception:
            server_proc.kill()

    seqkd_lines = seqkd_out.read_text(encoding="utf-8").strip().splitlines() if seqkd_out.exists() else []
    seqkd_meta_dict = json.loads(seqkd_meta.read_text(encoding="utf-8")) if seqkd_meta.exists() else {}
    print(f"[DISTILL_SEQKD DONE] Verified records={len(seqkd_lines)} | meta={seqkd_meta_dict}")

    # Persist generated JSONL into Modal Volume as well
    vol_seqkd = cache_dir / "verified_seqkd.jsonl"
    vol_seqkd.write_text(seqkd_out.read_text(encoding="utf-8"), encoding="utf-8")
    vol.commit()

    # -------------------------------------------------------------------------
    # STAGE B: Run Single-Seed 4-Arm Hierarchical Distillation loading seqkd_out
    # -------------------------------------------------------------------------
    print(f"[RUN_DISTILL_ABLATION] Loading {seqkd_out} in run_hierarchical_distill_ablation ...")
    res = run_hierarchical_distill_ablation(
        preset=preset,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        align_steps=align_steps,
        batch_size=batch_size,
        seeds=(42,),
        device="cuda" if torch.cuda.is_available() else "cpu",
        teacher_name="prism-ml/Ternary-Bonsai-2-4B-gguf",
        seqkd_jsonl=seqkd_out,
    )

    elapsed = round(time.perf_counter() - t0, 2)
    est_cost_usd = round((elapsed / 3600.0) * 1.10, 5)
    res["seqkd_generation_meta"] = seqkd_meta_dict
    res["seqkd_jsonl_records"] = [json.loads(line) for line in seqkd_lines]
    res["modal_hardware"] = {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "total_gpu_seconds": elapsed,
        "estimated_cost_usd": est_cost_usd,
        "teacher_model": "prism-ml/Ternary-Bonsai-2-4B-gguf (Ternary-Bonsai-2-4B-PQ2_0.gguf via llama-server)",
        "seed_policy": "single_seed_42",
        "scaledown_window_sec": 900,
    }
    print(f"[MODAL BONSAI-2-4B PIPELINE DONE] {elapsed}s (~${est_cost_usd})")
    return res


@app.local_entrypoint()
def main() -> None:
    res = run_bonsai_distill_on_modal.remote()

    # Save verified_seqkd.jsonl & verified_seqkd.meta.json locally
    seqkd_local = PROJECT_ROOT / "experiments" / "results" / "verified_seqkd.jsonl"
    seqkd_local.parent.mkdir(parents=True, exist_ok=True)
    records = res.pop("seqkd_jsonl_records", [])
    with seqkd_local.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta_local = seqkd_local.with_suffix(".meta.json")
    meta_local.write_text(json.dumps(res.get("seqkd_generation_meta", {}), indent=2), encoding="utf-8")

    out_root = PROJECT_ROOT / "modal_bonsai_distill_report.json"
    out_exp = PROJECT_ROOT / "experiments" / "results" / "distill_hierarchical_nano.json"
    payload_str = json.dumps(res, indent=2, default=str)
    out_root.write_text(payload_str, encoding="utf-8")
    out_exp.write_text(payload_str, encoding="utf-8")
    print(
        f"[LOCAL] Saved real llama-server SeqKD & Modal distillation report:\n"
        f"  - {seqkd_local} ({len(records)} verified teacher records)\n"
        f"  - {meta_local}\n"
        f"  - {out_root}\n"
        f"  - {out_exp}"
    )
    if res.get("per_seed"):
        print("METRICS (Seed 42):", json.dumps(res["per_seed"][0]["metrics"], indent=2))
