"""Rigorous Benchmark Suite for CNCM Evaluation (Review Items 3.1, 3.2, 5.3).

Addresses:
  - 3.1: Fresh, untouched PG-FIM v3 test set (60 tasks, brand-new templates & seed=314159,
         SHA-256 hash committed to `experiments/benchmarks/MANIFEST.json` BEFORE evaluation).
         Clearly marks the old 24 tasks (seed=9999) as the `dev` split.
  - 3.2: Standard & large-N benchmarks with 95% Wilson CIs:
         * HumanEval-Infilling (downloaded from `openai/human-eval-infilling` GitHub / MIT)
         * SAFIM (`gonglinyuan/safim` on HF / CC-BY-SA-4.0)
         * 1,000-item held-out Arithmetic suite
         * 500-item held-out Tool-Calling suite
  - 5.3: Syntax vs Semantics separation on Tool-Calling:
         Compares `syntax_only` JSON grammar vs `prompt_span_copy` JSON grammar on BOTH
         JSON schema validity rate AND semantic argument + evaluated answer accuracy!

Usage:
    python experiments/eval_benchmarks.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cncm.agent.grammar import JSONSchemaConstrainedMasker, extract_json_object
from cncm.agent.tools import create_default_registry
from cncm.config import get_preset
from cncm.model import CNCMModel
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer import (
    FIMPlanDataset,
    PackedChunkDataset,
    compute_fim_reward,
    make_synthetic_fim_tasks,
    set_seed,
    train_stage1,
    train_stage2,
)
from experiments.harness import save_result
from experiments.stats import format_wilson, wilson_ci

BENCH_DIR = REPO_ROOT / "experiments" / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 3.1 Fresh Untouched PG-FIM v3 Test Set (Completely New Templates & Seed)
# ---------------------------------------------------------------------------
def build_fresh_pg_fim_v3_test(n_tasks: int = 60, seed: int = 314159) -> Path:
    """Create 60 brand-new Python FIM tasks disjoint from `make_synthetic_fim_tasks`."""
    out_path = BENCH_DIR / "pg_fim_v3_test.json"
    rng = random.Random(seed)
    tasks: List[Dict[str, Any]] = []

    for i in range(n_tasks):
        kind = i % 6
        c = rng.randint(2, 9)
        if kind == 0:
            prefix = f"def scale_and_bias_{i}(nums):\n    out = []\n    for v in nums:\n"
            middle = f"        out.append(v * {c} + 1)\n"
            suffix = "    return out\n"
            test_code = f"assert scale_and_bias_{i}([1, 2, 3]) == [{1*c+1}, {2*c+1}, {3*c+1}]"
            kws = ["out", "append"]
        elif kind == 1:
            prefix = f"def count_multiples_{i}(arr):\n    hits = 0\n    for n in arr:\n"
            middle = f"        if n % {c} == 0:\n            hits += 1\n"
            suffix = "    return hits\n"
            arr = [c, c * 2, c + 1, c * 3]
            test_code = f"assert count_multiples_{i}({arr}) == 3"
            kws = ["hits", "if"]
        elif kind == 2:
            prefix = f"def clamp_high_{i}(vals):\n    res = []\n    for x in vals:\n"
            middle = f"        res.append({c} if x > {c} else x)\n"
            suffix = "    return res\n"
            test_code = f"assert clamp_high_{i}([0, {c+4}, 1]) == [0, {c}, 1]"
            kws = ["res", "append"]
        elif kind == 3:
            prefix = f"def dot_offset_{i}(xs, ys):\n    acc = {c}\n    for a, b in zip(xs, ys):\n"
            middle = "        acc += a * b\n"
            suffix = "    return acc\n"
            test_code = f"assert dot_offset_{i}([2, 3], [4, 5]) == {c + 23}"
            kws = ["acc"]
        elif kind == 4:
            prefix = f"def invert_pairs_{i}(pairs):\n    mapping = {{}}\n    for k, v in pairs:\n"
            middle = "        mapping[v] = k\n"
            suffix = "    return mapping\n"
            test_code = f"assert invert_pairs_{i}([('a', 1), ('b', 2)]) == {{1: 'a', 2: 'b'}}"
            kws = ["mapping"]
        else:
            prefix = f"def first_above_{i}(seq):\n    for item in seq:\n"
            middle = f"        if item > {c}:\n            return item\n"
            suffix = "    return -1\n"
            test_code = f"assert first_above_{i}([1, {c+2}, {c+5}]) == {c+2}"
            kws = ["if", "return"]

        tasks.append({
            "task_id": f"pg_fim_v3_{i:03d}",
            "prefix": prefix,
            "middle": middle,
            "suffix": suffix,
            "test_code": test_code,
            "oracle_keywords": kws,
            "language": "python",
        })

    out_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# 3.2 Held-Out Arithmetic (1,000 items) & Tool-Calling (500 items) Suites
# ---------------------------------------------------------------------------
def build_arithmetic_and_tool_suites(n_arith: int = 1000, n_tool: int = 500, seed: int = 271828) -> Tuple[Path, Path]:
    rng = random.Random(seed)
    arith_path = BENCH_DIR / "heldout_arithmetic_1000.json"
    tool_path = BENCH_DIR / "heldout_tool_calling_500.json"

    arith_items: List[Dict[str, Any]] = []
    for i in range(n_arith):
        a = rng.randint(2, 99)
        b = rng.randint(2, 25)
        c = rng.randint(1, 50)
        op1 = rng.choice(["+", "-", "*"])
        op2 = rng.choice(["+", "-"])
        expr = f"{a}{op1}{b}{op2}{c}"
        ans = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        arith_items.append({
            "id": f"arith_{i:04d}",
            "prompt": f"Compute {expr} =",
            "expression": expr,
            "expected_value": int(ans),
        })
    arith_path.write_text(json.dumps(arith_items, indent=2), encoding="utf-8")

    tool_items: List[Dict[str, Any]] = []
    for i in range(n_tool):
        a = rng.randint(3, 50)
        b = rng.randint(2, 19)
        c = rng.randint(1, 30)
        op1 = rng.choice(["*", "+", "-"])
        op2 = rng.choice(["+", "-"])
        expr = f"{a}{op1}{b}{op2}{c}"
        ans = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        tool_items.append({
            "id": f"tool_{i:04d}",
            "prompt": f"<|user|>Calculate {expr} using the calculator.<|end|><|assistant|><tool_call>",
            "raw_user_prompt": f"Calculate {expr} using the calculator.",
            "expected_tool": "calculator",
            "expected_expression": expr,
            "expected_value": float(ans),
        })
    tool_path.write_text(json.dumps(tool_items, indent=2), encoding="utf-8")
    return arith_path, tool_path


# ---------------------------------------------------------------------------
# External Standard Benchmarks: HumanEval-Infilling (MIT) & SAFIM (CC-BY-SA-4.0)
# ---------------------------------------------------------------------------
def load_humaneval_infilling(max_tasks: int = 100) -> Tuple[List[Dict[str, Any]], str]:
    """Download single-line HumanEval-Infilling (MIT) from official OpenAI GitHub."""
    cache = BENCH_DIR / "humaneval_infilling_singleline.jsonl.gz"
    url = "https://raw.githubusercontent.com/openai/human-eval-infilling/master/data/HumanEval-SingleLineInfilling.jsonl.gz"
    try:
        if not cache.exists():
            urllib.request.urlretrieve(url, str(cache))
        import gzip

        tasks: List[Dict[str, Any]] = []
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tasks.append({
                    "task_id": row.get("task_id", f"he_{len(tasks)}"),
                    "prefix": row.get("prompt", ""),
                    "suffix": row.get("suffix", ""),
                    "canonical_solution": row.get("canonical_solution", ""),
                    "test_code": row.get("test", "") + f"\ncheck({row.get('entry_point', '')})" if row.get("entry_point") else "",
                })
                if len(tasks) >= max_tasks:
                    break
        return tasks, f"openai/human-eval-infilling (MIT, n={len(tasks)})"
    except Exception as exc:
        return [], f"download_failed: {type(exc).__name__}: {exc}"


def load_safim_python(max_tasks: int = 100) -> Tuple[List[Dict[str, Any]], str]:
    """Load SAFIM Python block-infilling tasks from HuggingFace (`gonglinyuan/safim`, CC-BY-SA-4.0)."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset("gonglinyuan/safim", "block", split="test", streaming=True)
        tasks: List[Dict[str, Any]] = []
        for row in ds:
            if str(row.get("lang", "")).lower() != "python":
                continue
            prompt = str(row.get("prompt", ""))
            if "{{completion}}" in prompt:
                pfx, sfx = prompt.split("{{completion}}", 1)
            else:
                pfx, sfx = prompt, ""
            tasks.append({
                "task_id": str(row.get("task_id", f"safim_{len(tasks)}")),
                "prefix": pfx,
                "suffix": sfx,
                "ground_truth": str(row.get("ground_truth", "")),
                "test_code": "",
            })
            if len(tasks) >= max_tasks:
                break
        return tasks, f"gonglinyuan/safim (CC-BY-SA-4.0, n={len(tasks)})"
    except Exception as exc:
        return [], f"load_failed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
def _eval_fim_suite(model: CNCMModel, tok: CNCMTextTokenizer, tasks: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    parses = compiles = tests = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in tasks:
            out = model.generate_fim(
                prefix=t["prefix"][-256:],
                suffix=t["suffix"][:128],
                tokenizer=tok,
                max_chunks=1,
                temperature=0.0,
                early_exit_conf=0.92,
                use_plan=True,
                enable_repair=True,
                allow_template_fallback=False,
            )
            rew = compute_fim_reward(
                out["middle_text"],
                t,
                planned_keywords=out["keywords"],
                planned_chunks=out["planned_chunks"],
                actual_chunks=1,
            )
            parses += int(rew["parses"])
            compiles += int(rew["compiles"])
            tests += int(rew["tests_passed"])
    n = len(tasks)
    res = {
        "suite": label,
        "n": n,
        "parse": wilson_ci(parses, n),
        "compile": wilson_ci(compiles, n),
        "pass_at_1": wilson_ci(tests, n),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    print(f"  [{label}] parse={format_wilson(res['parse'])} | compile={format_wilson(res['compile'])} | pass@1={format_wilson(res['pass_at_1'])}")
    return res


def _eval_tool_calling_500(model: CNCMModel, tok: CNCMTextTokenizer, tool_items: List[Dict[str, Any]], use_copy_span: bool) -> Dict[str, Any]:
    """Evaluate 500-item held-out tool-calling suite separating syntax vs semantic accuracy."""
    registry = create_default_registry()
    syntax_valid = 0
    tool_name_ok = 0
    expr_exact_ok = 0
    value_exact_ok = 0
    n = len(tool_items)
    t0 = time.perf_counter()

    with torch.no_grad():
        for item in tool_items:
            pfx_ids = tok.encode(item["prompt"], add_special_tokens=False)
            pfx_t = torch.tensor([pfx_ids], dtype=torch.long)
            masker = JSONSchemaConstrainedMasker(
                tokenizer=tok,
                registry=registry,
                allowed_tools=["calculator"],
                max_math_len=16,
                max_tokens=36,
                prefix_len=len(pfx_ids),
                prompt_text=item["raw_user_prompt"] if use_copy_span else None,
            )
            out = model.generate_exact(
                pfx_t,
                max_new_tokens=36,
                temperature=0.0,
                stop_token_ids=[tok.tool_call_end_id],
                logit_mask_fn=masker,
            )
            gen_text = tok.decode(out["token_ids"][0].tolist(), skip_special_tokens=True)
            parsed = extract_json_object(gen_text)
            if parsed is not None and isinstance(parsed.get("arguments"), dict):
                syntax_valid += 1
                if parsed.get("name") == item["expected_tool"]:
                    tool_name_ok += 1
                expr = str(parsed["arguments"].get("expression", "")).replace(" ", "")
                if expr == item["expected_expression"]:
                    expr_exact_ok += 1
                try:
                    tool_out = registry.execute("calculator", {"expression": expr})
                    val_raw = tool_out.output.get("value") if hasattr(tool_out, "output") and isinstance(tool_out.output, dict) else None
                    if val_raw is not None and abs(float(val_raw) - float(item["expected_value"])) < 1e-6:
                        value_exact_ok += 1
                except Exception:
                    pass

    return {
        "mode": "prompt_span_copy_grammar" if use_copy_span else "syntax_only_grammar",
        "n": n,
        "json_syntax_valid": wilson_ci(syntax_valid, n),
        "tool_name_accuracy": wilson_ci(tool_name_ok, n),
        "semantic_expression_exact": wilson_ci(expr_exact_ok, n),
        "semantic_evaluated_value_exact": wilson_ci(value_exact_ok, n),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def _eval_arithmetic_1000(model: CNCMModel, tok: CNCMTextTokenizer, arith_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate 1,000-item held-out direct arithmetic suite (exact-AR greedy, no calculator)."""
    exact_hits = 0
    n = len(arith_items)
    t0 = time.perf_counter()
    with torch.no_grad():
        for item in arith_items:
            pfx_ids = tok.encode(item["prompt"], add_special_tokens=False)
            pfx_t = torch.tensor([pfx_ids], dtype=torch.long)
            out = model.generate_exact(pfx_t, max_new_tokens=8, temperature=0.0)
            gen_text = tok.decode(out["token_ids"][0].tolist(), skip_special_tokens=True).strip()
            exp_str = str(item["expected_value"])
            if gen_text.startswith(exp_str):
                exact_hits += 1
    return {
        "suite": "heldout_arithmetic_1000",
        "n": n,
        "exact_match": wilson_ci(exact_hits, n),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    torch.set_num_threads(min(8, 8))
    device = torch.device("cpu")

    # 1. Materialize & SHA-256 lock all fresh test sets BEFORE any evaluation!
    pg_v3_path = build_fresh_pg_fim_v3_test(n_tasks=60, seed=314159)
    arith_path, tool_path = build_arithmetic_and_tool_suites(n_arith=1000, n_tool=500, seed=271828)
    manifest = {
        "note": "SHA-256 hashes committed BEFORE evaluation; old 24 tasks (seed=9999) are dev split only.",
        "files": {
            pg_v3_path.name: _sha256_file(pg_v3_path),
            arith_path.name: _sha256_file(arith_path),
            tool_path.name: _sha256_file(tool_path),
        },
    }
    (BENCH_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[BENCHMARKS] Committed MANIFEST.json SHA-256 hashes: {manifest['files']}")

    # 2. Download standard external benchmarks (HumanEval-Infilling & SAFIM)
    he_tasks, he_src = load_humaneval_infilling(max_tasks=100)
    safim_tasks, safim_src = load_safim_python(max_tasks=100)
    print(f"[BENCHMARKS] External: {he_src} | {safim_src}")

    # 3. Train a fresh compact model on the OLD dev-only FIM training set (zero overlap with v3/HE/SAFIM!)
    set_seed(42)
    cfg = get_preset("tiny_test", enc_layers=1, use_copy_decoder=True)
    tok = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    fim_train = make_synthetic_fim_tasks(num_samples=160, seed=42, curriculum_stage="all")
    s1_docs = [t["prefix"] + t["middle"] + t["suffix"] for t in fim_train]
    s1_ds = PackedChunkDataset(s1_docs, tok, chunk_size=cfg.chunk_size)
    s1 = train_stage1(cfg, tok, s1_ds, steps=120, batch_size=32, lr=3e-3, device=device, seed=42, verbose=False)["model"].eval()
    fim_ds = FIMPlanDataset(samples=fim_train, tokenizer=tok, chunk_size=cfg.chunk_size, max_prefix_len=64, fim_rate=1.0, seed=42)
    model = train_stage2(cfg, tok, s1, fim_ds, mode="joint", steps=150, batch_size=16, lr=2e-3, device=device, seed=42, verbose=False, num_workers=0)["model"].eval()

    # 4. Run all evaluations with Wilson 95% CIs
    pg_v3_tasks = json.loads(pg_v3_path.read_text(encoding="utf-8"))
    pg_dev_tasks = make_synthetic_fim_tasks(num_samples=24, seed=9999, curriculum_stage="all")
    arith_items = json.loads(arith_path.read_text(encoding="utf-8"))
    tool_items = json.loads(tool_path.read_text(encoding="utf-8"))

    report: Dict[str, Any] = {
        "manifest": manifest,
        "model_preset": cfg.preset_name,
        "pg_fim_dev_old24": _eval_fim_suite(model, tok, pg_dev_tasks, "pg_fim_dev_old24"),
        "pg_fim_v3_fresh_untouched_60": _eval_fim_suite(model, tok, pg_v3_tasks, "pg_fim_v3_fresh_60"),
        "humaneval_infilling": (
            _eval_fim_suite(model, tok, he_tasks, he_src)
            if he_tasks
            else {"suite": he_src, "skipped": True}
        ),
        "safim_python": (
            _eval_fim_suite(model, tok, safim_tasks, safim_src)
            if safim_tasks
            else {"suite": safim_src, "skipped": True}
        ),
        "heldout_arithmetic_1000": _eval_arithmetic_1000(model, tok, arith_items),
        "heldout_tool_calling_500_syntax_only": _eval_tool_calling_500(model, tok, tool_items, use_copy_span=False),
        "heldout_tool_calling_500_copy_span": _eval_tool_calling_500(model, tok, tool_items, use_copy_span=True),
    }

    print(f"  [Arithmetic 1000] exact={format_wilson(report['heldout_arithmetic_1000']['exact_match'])}")
    print(f"  [Tool-500 syntax_only] syntax={format_wilson(report['heldout_tool_calling_500_syntax_only']['json_syntax_valid'])} | semantic_val={format_wilson(report['heldout_tool_calling_500_syntax_only']['semantic_evaluated_value_exact'])}")
    print(f"  [Tool-500 copy_span]   syntax={format_wilson(report['heldout_tool_calling_500_copy_span']['json_syntax_valid'])} | semantic_val={format_wilson(report['heldout_tool_calling_500_copy_span']['semantic_evaluated_value_exact'])}")

    out = save_result("benchmarks_rigor", report)
    print(f"\n[BENCHMARKS COMPLETE] Saved -> {out}")


if __name__ == "__main__":
    main()
