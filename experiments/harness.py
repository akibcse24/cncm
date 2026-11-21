"""Shared experimental harness and JSON ledger for CNCM remediation experiments.

Every experiment run through this harness:
  1. Uses deterministic seeds (`DEFAULT_SEEDS = (42, 1337, 2025)` — >=3 seeds).
  2. Records exact step counts, token budgets, and a canonical SHA-256 `config_hash`.
  3. Captures hardware/software environment metadata (CPU, OS, Python, PyTorch, ORT, git SHA).
  4. Aggregates scalar metrics across seeds as mean ± 95% CI (Student-t + bootstrap).
  5. Writes atomically to `experiments/results/<name>.json`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from experiments.stats import summarize_seeded_metric

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results"
DEFAULT_SEEDS: Tuple[int, ...] = (42,)
LEDGER_SCHEMA_VERSION = "1.0.0"


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and (if installed) PyTorch RNG seeds deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Compute a stable SHA-256 hex digest of an experiment configuration dict."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", errors="ignore").strip()
        return out or None
    except Exception:
        return None


def collect_hardware_metadata() -> Dict[str, Any]:
    """Collect reproducible host hardware & runtime metadata."""
    meta: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "git_commit": _get_git_commit(),
    }
    try:
        import torch  # type: ignore[import-not-found]

        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        meta["torch_version"] = None
        meta["cuda_available"] = False

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        meta["onnxruntime_version"] = ort.__version__
        meta["ort_providers"] = ort.get_available_providers()
    except ImportError:
        meta["onnxruntime_version"] = None

    return meta


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, default=str), encoding="utf-8")
    tmp.replace(path)


def save_result(
    name: str,
    payload: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    seeds: Optional[Sequence[int]] = None,
    steps: Optional[int] = None,
    total_tokens: Optional[int] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Write an experiment result entry to `experiments/results/<name>.json`."""
    out_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name[:-5] if name.endswith(".json") else name
    out_path = out_dir / f"{safe_name}.json"

    cfg_dict = dict(config) if config is not None else {}
    record: Dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "experiment_name": safe_name,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config_hash": compute_config_hash(cfg_dict),
        "config": cfg_dict,
        "seeds": [int(s) for s in seeds] if seeds is not None else [],
        "steps": int(steps) if steps is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
        "hardware": collect_hardware_metadata(),
        "payload": dict(payload),
    }
    _atomic_write_json(out_path, record)
    return out_path


def load_result(
    name_or_path: Union[str, Path],
    results_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load an experiment ledger JSON file by name or path."""
    p = Path(name_or_path)
    if not p.exists():
        base = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        fname = str(name_or_path)
        if not fname.endswith(".json"):
            fname += ".json"
        p = base / fname
    return json.loads(p.read_text(encoding="utf-8"))


def _flatten_numeric_scalars(d: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    """Extract numeric leaf metrics (excluding booleans) for automatic CI aggregation."""
    flat: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and np.isfinite(float(v)):
            flat[key] = float(v)
        elif isinstance(v, Mapping):
            flat.update(_flatten_numeric_scalars(v, prefix=key))
    return flat


def run_seeded_experiment(
    name: str,
    run_fn: Callable[[int], Dict[str, Any]],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    config: Optional[Mapping[str, Any]] = None,
    steps: Optional[int] = None,
    total_tokens: Optional[int] = None,
    save: bool = True,
    results_dir: Optional[Union[str, Path]] = None,
    require_min_seeds: int = 1,
) -> Dict[str, Any]:
    """Execute `run_fn(seed)` over `seeds`, aggregate all numeric metrics with 95% CIs, and save."""
    if not seeds:
        raise ValueError("seeds must be non-empty")

    per_seed: List[Dict[str, Any]] = []
    metric_buckets: Dict[str, List[float]] = {}
    t_total_start = time.perf_counter()

    for seed in seeds:
        set_global_seed(int(seed))
        t0 = time.perf_counter()
        out = run_fn(int(seed))
        elapsed = round(time.perf_counter() - t0, 3)
        entry = {"seed": int(seed), "elapsed_sec": elapsed, "metrics": out}
        per_seed.append(entry)
        for mk, mv in _flatten_numeric_scalars(out).items():
            metric_buckets.setdefault(mk, []).append(mv)

    aggregated: Dict[str, Any] = {}
    for mk, vals in sorted(metric_buckets.items()):
        if len(vals) == len(seeds):
            aggregated[mk] = summarize_seeded_metric(
                vals,
                confidence=0.95,
                require_min_seeds=require_min_seeds,
            )

    payload: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "meets_min_seed_policy": len(seeds) >= require_min_seeds,
        "total_elapsed_sec": round(time.perf_counter() - t_total_start, 3),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }

    saved_path: Optional[str] = None
    if save:
        p = save_result(
            name=name,
            payload=payload,
            config=config,
            seeds=seeds,
            steps=steps,
            total_tokens=total_tokens,
            results_dir=results_dir,
        )
        saved_path = str(p)

    return {
        "experiment_name": name,
        "config_hash": compute_config_hash(config or {}),
        "saved_path": saved_path,
        **payload,
    }
