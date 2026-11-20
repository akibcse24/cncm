"""Private Kaggle GPU T4 x2 DDP Launcher for `ctx0an` (CNCM base_moe: 221.45M total / 94.05M active).

Zero hardcoded secrets: reads `HF_TOKEN` only from environment or Kaggle Secrets if attached,
and persists all checkpoints, BPE tokenizer, uint16 memmaps, and JSON reports cleanly to
`/kaggle/working/ctx0an_out/` under a 325-minute wall-clock guard (fits Kaggle's 6h quota).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _find_code_source() -> Path:
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for match in input_root.rglob("train_ctx0an_base.py"):
            return match.parent
        for zf in input_root.rglob("ctx0an_src.zip"):
            return zf
    raise FileNotFoundError(
        f"Could not locate `train_ctx0an_base.py` or `ctx0an_src.zip` under {input_root}."
    )


def _stage_code(src_path: Path, dst_dir: Path) -> Path:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_path.is_file() and src_path.suffix == ".zip":
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(dst_dir)
    else:
        for item in src_path.iterdir():
            target = dst_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            elif item.suffix == ".zip" and item.name == "ctx0an_src.zip":
                with zipfile.ZipFile(item, "r") as zf:
                    zf.extractall(dst_dir)
            else:
                shutil.copy2(item, target)
    return dst_dir


def _ensure_dependencies() -> None:
    missing: list[str] = []
    for pkg in ("datasets", "tokenizers", "huggingface_hub"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SETUP] Installing missing packages: {missing}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            check=True,
        )


def _seed_prior_session_outputs(out_dir: Path) -> None:
    """If a previous Kaggle session's output is mounted under `/kaggle/input`, copy checkpoints/memmaps."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return
    for prior_meta in input_root.rglob("ctx0an_corpus_meta.json"):
        prior_out = prior_meta.parent.parent
        if prior_out == out_dir:
            continue
        print(f"[RESUME SEED] Found prior session artifacts at {prior_out}", flush=True)
        for sub in ("memmap_data", "checkpoints"):
            src_sub = prior_out / sub
            dst_sub = out_dir / sub
            if src_sub.exists() and not dst_sub.exists():
                shutil.copytree(src_sub, dst_sub)
        break


def main() -> None:
    print("================ [ctx0an Kaggle T4 x2 Launcher] ================", flush=True)
    subprocess.run(["nvidia-smi"], check=False)
    _ensure_dependencies()

    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[HW] PyTorch={torch.__version__} | CUDA={torch.cuda.is_available()} | GPUs={n_gpus}", flush=True)
    for idx in range(n_gpus):
        props = torch.cuda.get_device_properties(idx)
        print(
            f"  GPU {idx}: {props.name} | VRAM={props.total_memory / 1e9:.2f} GB | SM={props.major}.{props.minor}",
            flush=True,
        )

    src_found = _find_code_source()
    work_src = _stage_code(src_found, Path("/tmp/ctx0an_src"))
    print(f"[CODE] Staged clean source from {src_found} -> {work_src}", flush=True)

    out_dir = Path("/kaggle/working/ctx0an_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_prior_session_outputs(out_dir)

    # Read optional HF_TOKEN from Kaggle Secrets (never hardcoded!)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if not env.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

            tok = UserSecretsClient().get_secret("HF_TOKEN")
            if tok:
                env["HF_TOKEN"] = str(tok).strip()
                print("[HF] Loaded HF_TOKEN from Kaggle Secrets (private sync enabled).", flush=True)
        except Exception:
            print("[HF] No Kaggle Secret HF_TOKEN attached; checkpoints will persist to private Kaggle Output.", flush=True)

    train_script = str(work_src / "train_ctx0an_base.py")
    train_args = [
        "--model-name", "ctx0an",
        "--preset", "base",
        "--enc-layers", "1",
        "--prefix-len", "512",
        "--en-docs", "75000",
        "--py-docs", "70000",
        "--math-docs", "35000",
        "--json-docs", "20000",
        "--stage1-steps", "2500",
        "--stage1-batch-size", "64",
        "--stage2-batch-size", "6",
        "--grad-accum", "10",
        "--phase1-steps", "1500",
        "--phase1-lr", "5e-4",
        "--phase2-steps", "2500",
        "--phase2-lr", "3.5e-4",
        "--phase3-sft-steps", "400",
        "--phase3-lr", "1.2e-4",
        "--amp-dtype", "auto",
        "--grad-checkpointing",
        "--save-every", "250",
        "--eval-every", "250",
        "--keep-last-k", "1",
        "--max-wall-minutes", "325",
        "--hf-repo-id", "ctx0akib/ctx0an",
        "--hf-upload-every", "500",
        "--out-dir", str(out_dir),
    ]

    # Step 1: Single-process data prep (streams HF corpora, trains Rust BPE, builds uint16 memmaps)
    # so Rank 1 NEVER sits idle at an NCCL barrier during data download/tokenization.
    prep_cmd = [sys.executable, train_script, *train_args, "--prepare-only"]
    print(f"[STEP 1/2 PREPARE] Running: {' '.join(prep_cmd)}", flush=True)
    prep_proc = subprocess.run(prep_cmd, cwd=str(work_src), env=env)
    if prep_proc.returncode != 0:
        raise SystemExit(prep_proc.returncode)

    # Step 2: Launch multi-GPU DDP training (all ranks open cached uint16 memmaps in <0.1s!)
    if n_gpus >= 2:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={n_gpus}",
            train_script,
            *train_args,
        ]
    else:
        cmd = [sys.executable, train_script, *train_args]

    print(f"[STEP 2/2 TRAIN] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(work_src), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print("[DONE] ctx0an training session exited cleanly.", flush=True)


if __name__ == "__main__":
    main()
