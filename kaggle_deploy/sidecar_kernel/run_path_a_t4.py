"""Kaggle Free 15 GB T4 Launcher for Path A `CNCM-DeltaNet` Plug-In Sidecar (~5-10 min).

Replaces 35-hour TPU v5e-8 full-trunk training and 1.25 GB downloads with:
  - Zero-backprop Offline `L16+L24+L32` Hidden + Top-16 Logit Cache
  - SVD-Projected Frozen-Embedding Stage 1 (~1.78M params)
  - Closed-Form Step-0 Ridge Warm-Start
  - 4-Layer `CNCM-DeltaNet` Sidecar + Learned Dual-Mode Router
  - Exports ONLY a ~13.6 MB `.sidecar.gguf` (`q4_0`) + JSON report in `/kaggle/working`
    (deletes temporary memmap cache before exit so `kaggle kernels output` is <15 MB!).
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
        for match in input_root.rglob("universal_surgery.py"):
            return match.parent
        for zf in input_root.rglob("ctx0an_src.zip"):
            return zf
    local_root = Path(__file__).resolve().parents[2]
    if (local_root / "universal_surgery.py").exists():
        return local_root
    raise FileNotFoundError(f"Could not locate `universal_surgery.py` or `ctx0an_src.zip` under {input_root}.")


def _stage_code(src_path: Path, dst_dir: Path) -> Path:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_path.is_file() and src_path.suffix == ".zip":
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(dst_dir)
    else:
        for item in src_path.iterdir():
            if item.name in (".git", ".venv", ".venv_dml", "__pycache__", ".pytest_cache"):
                continue
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
    for pkg in ("safetensors", "datasets", "tokenizers", "huggingface_hub"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[T4 SETUP] Installing missing packages: {missing}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def main() -> None:
    print("================ [Path A CNCM-DeltaNet Sidecar — Free T4 Launcher] ================", flush=True)
    _ensure_dependencies()
    work_src = _stage_code(_find_code_source(), Path("/tmp/ctx0an_sidecar_src"))

    # Keep temporary memmap cache in `/tmp` so `/kaggle/working` ONLY contains the ~14 MB `.sidecar.gguf`!
    out_dir = Path("/kaggle/working/sidecar_out")
    tmp_cache = Path("/tmp/donor_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    donor_repo = os.environ.get("DONOR_REPO", "HuggingFaceTB/SmolLM2-360M-Instruct")
    chunk_size = os.environ.get("CHUNK_SIZE", "32")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    if not env.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

            tok = UserSecretsClient().get_secret("HF_TOKEN")
            if tok:
                env["HF_TOKEN"] = str(tok).strip()
        except Exception:
            pass

    cmd = [
        sys.executable,
        str(work_src / "universal_surgery.py"),
        "--donor", donor_repo,
        "--chunk-size", chunk_size,
        "--prefix-len", "128",
        "--cache-seqs", "4096",
        "--cache-batch-size", "32",
        "--stage1-steps", "500",
        "--stage1-batch-size", "64",
        "--ridge-samples", "1024",
        "--stage2-steps", "2000",
        "--batch-size", "32",
        "--quant", "q4_0",
        "--cache-dir", str(tmp_cache),
        "--out-dir", str(out_dir),
    ]
    print(f"[LAUNCH] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(work_src), env=env, check=True)

    # Remove intermediate PyTorch `.pt` from `/kaggle/working` if `KEEP_PT != 1` so total download is < 15 MB!
    if os.environ.get("KEEP_PT", "0") != "1":
        for pt_file in out_dir.glob("*.sidecar.pt"):
            pt_file.unlink(missing_ok=True)

    print("[DONE] Output files ready in /kaggle/working/sidecar_out:", flush=True)
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}: {f.stat().st_size / (1024*1024):.2f} MB", flush=True)


if __name__ == "__main__":
    main()
