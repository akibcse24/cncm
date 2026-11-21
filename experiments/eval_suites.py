"""Evaluation Suites & Contamination-Free Held-Out Benchmarks (`experiments/eval_suites.py`).

Implements Items 3.1, 3.2, and 5.3 from `remediation_plan.md`:
  1. **PG-FIM Dev vs Held-Out Test Split (Item 3.1)**:
       - Marks the legacy 24 tasks from `cncm.trainer.dataset._build_fim_task_bank` as `split="dev"`.
       - Builds a strictly disjoint held-out `split="test"` bank (36 tasks across `single_line`,
         `multi_line`, `function_body`) with brand-new function names, algorithms, and generator
         seed (`20261120`), completely unseen during development.
       - Pre-commits and verifies the canonical SHA-256 hash in
         `experiments/manifests/eval_suites_sha256.json` BEFORE any evaluation.
  2. **Held-Out Arithmetic Suite (Item 3.2)**:
       - >= 1,000 deterministic problems (seed `20261121`) covering multi-op integer arithmetic,
         operator precedence, and parenthesized expressions with exact ground-truth values.
  3. **Held-Out Tool-Calling Suite (Items 3.2 & 5.3)**:
       - >= 500 deterministic prompts (seed `20261122`) across `calculator`, `kv_memory`,
         `text_stats`, and `clock_utc`, reporting **syntax rate** and **semantic accuracy**
         separately with Wilson 95% confidence intervals.
  4. **External Code FIM Benchmarks (Item 3.2)**:
       - Standardized loaders/evaluators for **HumanEval-Infilling** (`single-line`, `multi-line`,
         `random-span`) and **SAFIM** (`block`, `control`, `api`) with Wilson 95% CIs.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from cncm.agent.grammar import evaluate_tool_call_semantic_accuracy
from cncm.agent.tools import safe_eval_math
from cncm.code.sandbox import run_code_sandbox
from experiments.stats import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "experiments" / "manifests" / "eval_suites_sha256.json"

PG_FIM_TEST_SEED = 20261120
ARITHMETIC_TEST_SEED = 20261121
TOOL_CALL_TEST_SEED = 20261122


def canonical_sha256(items: Sequence[Dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 digest over a list of benchmark task dicts."""
    payload = json.dumps(list(items), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Item 3.1: PG-FIM Held-Out Test Set (Unseen Templates, Distinct Seed)
# ---------------------------------------------------------------------------
def build_pg_fim_heldout_test_bank(seed: int = PG_FIM_TEST_SEED) -> List[Dict[str, Any]]:
    """Build 36 held-out Python FIM tasks with templates disjoint from the legacy dev bank."""
    rng = random.Random(seed)
    tasks: List[Dict[str, Any]] = []

    # 1. single_line (12 tasks, 4 unseen families)
    for idx in range(12):
        kind = idx % 4
        if kind == 0:
            lo = rng.randint(0, 5)
            hi = lo + rng.randint(8, 25)
            prefix = f"def bound_value(x, lo={lo}, hi={hi}):\n    "
            middle = "return max(lo, min(hi, x))"
            suffix = "\n"
            test_code = (
                f"assert bound_value({lo - 3}) == {lo}\n"
                f"assert bound_value({hi + 4}) == {hi}\n"
                f"assert bound_value({(lo + hi) // 2}) == {(lo + hi) // 2}\n"
            )
            kws = ["max", "min", "lo"]
            fname = "bound_value"
        elif kind == 1:
            base = rng.randint(2, 5)
            prefix = (
                f"def count_multiples(seq, k={base}):\n"
                "    hits = 0\n"
                "    for item in seq:\n"
                "        if item % k == 0:\n"
                "            "
            )
            middle = "hits += 1"
            suffix = "\n    return hits\n"
            sample = [base, base + 1, base * 2, base * 3 + 1, base * 4]
            test_code = (
                f"assert count_multiples({sample}) == 3\n"
                "assert count_multiples([]) == 0\n"
            )
            kws = ["hits", "1"]
            fname = "count_multiples"
        elif kind == 2:
            prefix = (
                "def harmonic_pair(a, b):\n"
                "    if a + b == 0:\n"
                "        return 0.0\n"
                "    "
            )
            middle = "return (2.0 * a * b) / (a + b)"
            suffix = "\n"
            test_code = (
                "assert abs(harmonic_pair(2.0, 6.0) - 3.0) < 1e-6\n"
                "assert harmonic_pair(0.0, 0.0) == 0.0\n"
            )
            kws = ["return", "a", "b"]
            fname = "harmonic_pair"
        else:
            prefix = (
                "def swap_ends(items):\n"
                "    out = list(items)\n"
                "    if len(out) >= 2:\n"
                "        "
            )
            middle = "out[0], out[-1] = out[-1], out[0]"
            suffix = "\n    return out\n"
            test_code = (
                "assert swap_ends([1, 2, 3, 4]) == [4, 2, 3, 1]\n"
                "assert swap_ends([7]) == [7]\n"
            )
            kws = ["out", "0", "-1"]
            fname = "swap_ends"

        tasks.append(
            {
                "task_id": f"pg_fim_test_sl_{idx:02d}",
                "split": "test",
                "curriculum_stage": "single_line",
                "func_name": fname,
                "prefix": prefix,
                "middle": middle,
                "suffix": suffix,
                "full_code": prefix + middle + suffix,
                "test_code": test_code,
                "oracle_keywords": kws,
                "expected_chunks": 1,
                "language": "python",
            }
        )

    # 2. multi_line (12 tasks, 4 unseen families)
    for idx in range(12):
        kind = idx % 4
        if kind == 0:
            prefix = (
                "def running_prefix_sums(nums):\n"
                "    out = []\n"
                "    acc = 0\n"
            )
            middle = (
                "    for v in nums:\n"
                "        acc += v\n"
                "        out.append(acc)"
            )
            suffix = "\n    return out\n"
            test_code = (
                "assert running_prefix_sums([2, 3, -1, 4]) == [2, 5, 4, 8]\n"
                "assert running_prefix_sums([]) == []\n"
            )
            kws = ["acc", "out", "append"]
            fname = "running_prefix_sums"
        elif kind == 1:
            prefix = (
                "def invert_unique_map(mapping):\n"
                "    rev = {}\n"
            )
            middle = (
                "    for k, v in mapping.items():\n"
                "        rev[v] = k"
            )
            suffix = "\n    return rev\n"
            test_code = (
                "assert invert_unique_map({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\n"
                "assert invert_unique_map({}) == {}\n"
            )
            kws = ["rev", "items", "for"]
            fname = "invert_unique_map"
        elif kind == 2:
            step_k = rng.randint(2, 4)
            prefix = (
                f"def chunk_list(seq, size={step_k}):\n"
                "    chunks = []\n"
            )
            middle = (
                "    for i in range(0, len(seq), size):\n"
                "        chunks.append(seq[i : i + size])"
            )
            suffix = "\n    return chunks\n"
            test_code = (
                "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
                "assert chunk_list([], 3) == []\n"
            )
            kws = ["chunks", "range", "size"]
            fname = "chunk_list"
        else:
            prefix = (
                "def char_histogram(text):\n"
                "    counts = {}\n"
            )
            middle = (
                "    for ch in text:\n"
                "        counts[ch] = counts.get(ch, 0) + 1"
            )
            suffix = "\n    return counts\n"
            test_code = (
                "assert char_histogram('ababa') == {'a': 3, 'b': 2}\n"
                "assert char_histogram('') == {}\n"
            )
            kws = ["counts", "get", "for"]
            fname = "char_histogram"

        tasks.append(
            {
                "task_id": f"pg_fim_test_ml_{idx:02d}",
                "split": "test",
                "curriculum_stage": "multi_line",
                "func_name": fname,
                "prefix": prefix,
                "middle": middle,
                "suffix": suffix,
                "full_code": prefix + middle + suffix,
                "test_code": test_code,
                "oracle_keywords": kws,
                "expected_chunks": 2,
                "language": "python",
            }
        )

    # 3. function_body (12 tasks, 4 unseen families)
    for idx in range(12):
        kind = idx % 4
        if kind == 0:
            prefix = "def dedup_preserve_order(items):\n"
            middle = (
                "    seen = set()\n"
                "    out = []\n"
                "    for x in items:\n"
                "        if x not in seen:\n"
                "            seen.add(x)\n"
                "            out.append(x)\n"
                "    return out"
            )
            suffix = "\n"
            test_code = (
                "assert dedup_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
                "assert dedup_preserve_order([]) == []\n"
            )
            kws = ["seen", "out", "return"]
            fname = "dedup_preserve_order"
        elif kind == 1:
            prefix = "def second_largest(nums):\n"
            middle = (
                "    uniq = sorted(set(nums))\n"
                "    if len(uniq) < 2:\n"
                "        return None\n"
                "    return uniq[-2]"
            )
            suffix = "\n"
            test_code = (
                "assert second_largest([4, 9, 2, 9, 5]) == 5\n"
                "assert second_largest([7, 7]) is None\n"
            )
            kws = ["uniq", "sorted", "return"]
            fname = "second_largest"
        elif kind == 2:
            prefix = "def is_balanced_parens(s):\n"
            middle = (
                "    depth = 0\n"
                "    for ch in s:\n"
                "        if ch == '(':\n"
                "            depth += 1\n"
                "        elif ch == ')':\n"
                "            depth -= 1\n"
                "            if depth < 0:\n"
                "                return False\n"
                "    return depth == 0"
            )
            suffix = "\n"
            test_code = (
                "assert is_balanced_parens('(()())') is True\n"
                "assert is_balanced_parens('())(') is False\n"
            )
            kws = ["depth", "False", "return"]
            fname = "is_balanced_parens"
        else:
            prefix = "def merge_sorted_pair(a, b):\n"
            middle = (
                "    i, j = 0, 0\n"
                "    out = []\n"
                "    while i < len(a) and j < len(b):\n"
                "        if a[i] <= b[j]:\n"
                "            out.append(a[i])\n"
                "            i += 1\n"
                "        else:\n"
                "            out.append(b[j])\n"
                "            j += 1\n"
                "    out.extend(a[i:])\n"
                "    out.extend(b[j:])\n"
                "    return out"
            )
            suffix = "\n"
            test_code = (
                "assert merge_sorted_pair([1, 4, 6], [2, 3, 5, 7]) == [1, 2, 3, 4, 5, 6, 7]\n"
                "assert merge_sorted_pair([], [1, 2]) == [1, 2]\n"
            )
            kws = ["while", "out", "return"]
            fname = "merge_sorted_pair"

        tasks.append(
            {
                "task_id": f"pg_fim_test_fb_{idx:02d}",
                "split": "test",
                "curriculum_stage": "function_body",
                "func_name": fname,
                "prefix": prefix,
                "middle": middle,
                "suffix": suffix,
                "full_code": prefix + middle + suffix,
                "test_code": test_code,
                "oracle_keywords": kws,
                "expected_chunks": 3,
                "language": "python",
            }
        )

    return tasks


# ---------------------------------------------------------------------------
# Item 3.2: Held-Out Arithmetic Suite (>= 1,000 items)
# ---------------------------------------------------------------------------
def build_heldout_arithmetic_suite(
    n_items: int = 1000,
    seed: int = ARITHMETIC_TEST_SEED,
) -> List[Dict[str, Any]]:
    """Generate deterministic held-out arithmetic benchmark (>=1,000 items)."""
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    for i in range(n_items):
        tier = i % 4
        if tier == 0:
            a, b = rng.randint(10, 999), rng.randint(10, 999)
            op = rng.choice(["+", "-"])
            expr = f"{a} {op} {b}"
            difficulty = "easy_add_sub"
        elif tier == 1:
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            expr = f"{a} * {b}"
            difficulty = "medium_mul"
        elif tier == 2:
            a, b, c = rng.randint(2, 50), rng.randint(2, 30), rng.randint(1, 100)
            op2 = rng.choice(["+", "-"])
            expr = f"{a} * {b} {op2} {c}"
            difficulty = "medium_precedence"
        else:
            a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 15)
            op1 = rng.choice(["+", "-"])
            expr = f"({a} {op1} {b}) * {c}"
            difficulty = "hard_parens"

        val = safe_eval_math(expr)
        items.append(
            {
                "task_id": f"arith_{i:04d}",
                "split": "test",
                "difficulty": difficulty,
                "expression": expr,
                "prompt": f"Compute the exact value of {expr}.",
                "expected_value": int(val) if float(val).is_integer() else round(float(val), 6),
            }
        )
    return items


# ---------------------------------------------------------------------------
# Items 3.2 & 5.3: Held-Out Tool-Calling Suite (>= 500 items)
# ---------------------------------------------------------------------------
def build_heldout_tool_calling_suite(
    n_items: int = 500,
    seed: int = TOOL_CALL_TEST_SEED,
) -> List[Dict[str, Any]]:
    """Generate deterministic held-out tool-calling suite (>=500 items)."""
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    words_bank = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "gamma", "helios", "ion", "kappa"]

    for i in range(n_items):
        kind = i % 4
        if kind == 0:
            a, b, c = rng.randint(3, 85), rng.randint(2, 25), rng.randint(1, 60)
            expr = f"{a}*{b}+{c}" if (i % 8 == 0) else f"({a}+{b})*{c}"
            val = safe_eval_math(expr)
            items.append(
                {
                    "task_id": f"tool_{i:04d}",
                    "split": "test",
                    "prompt": f"Use the calculator tool to evaluate {expr}.",
                    "expected_tool": "calculator",
                    "expected_args": {"expression": expr},
                    "expected_value": int(val) if float(val).is_integer() else round(float(val), 6),
                }
            )
        elif kind == 1:
            k = f"key_{rng.randint(100, 999)}"
            v = rng.choice(words_bank)
            items.append(
                {
                    "task_id": f"tool_{i:04d}",
                    "split": "test",
                    "prompt": f"Store value '{v}' under key '{k}' in kv_memory.",
                    "expected_tool": "kv_memory",
                    "expected_args": {"action": "set", "key": k, "value": v},
                    "expected_value": None,
                }
            )
        elif kind == 2:
            sample_text = " ".join(rng.choices(words_bank, k=rng.randint(3, 7)))
            items.append(
                {
                    "task_id": f"tool_{i:04d}",
                    "split": "test",
                    "prompt": f"Count words and characters using text_stats for: {sample_text}",
                    "expected_tool": "text_stats",
                    "expected_args": {"text": sample_text},
                    "expected_value": None,
                }
            )
        else:
            items.append(
                {
                    "task_id": f"tool_{i:04d}",
                    "split": "test",
                    "prompt": "Check the current UTC timestamp using clock_utc.",
                    "expected_tool": "clock_utc",
                    "expected_args": {},
                    "expected_value": None,
                }
            )
    return items


# ---------------------------------------------------------------------------
# Manifest Commitment & Verification (Item 3.1)
# ---------------------------------------------------------------------------
def build_and_write_manifest(manifest_path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    """Compute and commit SHA-256 hashes for all held-out evaluation suites."""
    pg_fim_test = build_pg_fim_heldout_test_bank()
    arith_test = build_heldout_arithmetic_suite(1000)
    tool_test = build_heldout_tool_calling_suite(500)

    manifest = {
        "pg_fim_heldout_test": {
            "seed": PG_FIM_TEST_SEED,
            "count": len(pg_fim_test),
            "sha256": canonical_sha256(pg_fim_test),
        },
        "arithmetic_heldout_test": {
            "seed": ARITHMETIC_TEST_SEED,
            "count": len(arith_test),
            "sha256": canonical_sha256(arith_test),
        },
        "tool_calling_heldout_test": {
            "seed": TOOL_CALL_TEST_SEED,
            "count": len(tool_test),
            "sha256": canonical_sha256(tool_test),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_manifest(manifest_path: Path = MANIFEST_PATH) -> Dict[str, bool]:
    """Verify live generated held-out suites match the pre-committed SHA-256 manifest."""
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_pg = canonical_sha256(build_pg_fim_heldout_test_bank())
    actual_ar = canonical_sha256(build_heldout_arithmetic_suite(1000))
    actual_tl = canonical_sha256(build_heldout_tool_calling_suite(500))
    return {
        "pg_fim_heldout_test": actual_pg == expected["pg_fim_heldout_test"]["sha256"],
        "arithmetic_heldout_test": actual_ar == expected["arithmetic_heldout_test"]["sha256"],
        "tool_calling_heldout_test": actual_tl == expected["tool_calling_heldout_test"]["sha256"],
    }


# ---------------------------------------------------------------------------
# Generic Evaluators with Wilson 95% CIs (Items 3.1, 3.2, 5.3)
# ---------------------------------------------------------------------------
def evaluate_fim_suite(
    tasks: Sequence[Dict[str, Any]],
    predict_fn: Callable[[str, str], str],
    *,
    timeout_sec: float = 1.5,
) -> Dict[str, Any]:
    """Evaluate a FIM predictor `(prefix, suffix) -> middle` in the safe AST sandbox."""
    n = len(tasks)
    parses = 0
    compiles = 0
    passed = 0
    for t in tasks:
        pred_mid = predict_fn(t["prefix"], t["suffix"])
        code = t["prefix"] + pred_mid + t["suffix"]
        sb = run_code_sandbox(code, test_code=t.get("test_code", ""), timeout_sec=timeout_sec)
        if sb.get("parses"):
            parses += 1
        if sb.get("compiles"):
            compiles += 1
        if sb.get("tests_passed"):
            passed += 1
    return {
        "n_tasks": n,
        "parse_wilson_95": wilson_interval(parses, n),
        "compile_wilson_95": wilson_interval(compiles, n),
        "pass_at_1_wilson_95": wilson_interval(passed, n),
    }


def evaluate_tool_suite_with_ci(
    predictions: Sequence[str],
    tasks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score tool-calling predictions with separate syntax and semantic Wilson 95% CIs."""
    raw = evaluate_tool_call_semantic_accuracy(predictions, tasks)
    n = raw["total"]
    return {
        "total": n,
        "syntax_wilson_95": wilson_interval(raw["syntax_valid"], n),
        "tool_match_wilson_95": wilson_interval(raw["tool_match"], n),
        "semantic_wilson_95": wilson_interval(raw["semantic_correct"], n),
    }


def load_humaneval_infilling_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load local HumanEval-Infilling JSONL (single-line / multi-line / random-span)."""
    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            tasks.append(
                {
                    "task_id": obj.get("task_id", f"hei_{len(tasks)}"),
                    "prefix": obj["prompt"],
                    "suffix": obj["suffix"],
                    "canonical_solution": obj.get("canonical_solution", ""),
                    "test_code": obj.get("test", ""),
                    "entry_point": obj.get("entry_point", ""),
                }
            )
    return tasks


def load_safim_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load local SAFIM JSONL (block / control / api completion)."""
    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("eval_prompt") or obj.get("prompt") or ""
            if "{{completion}}" in prompt:
                prefix, suffix = prompt.split("{{completion}}", 1)
            else:
                prefix, suffix = prompt, ""
            tasks.append(
                {
                    "task_id": obj.get("task_id", f"safim_{len(tasks)}"),
                    "prefix": prefix,
                    "suffix": suffix,
                    "ground_truth": obj.get("ground_truth", ""),
                    "lang": obj.get("lang", "python"),
                }
            )
    return tasks


if __name__ == "__main__":
    m = build_and_write_manifest()
    print(json.dumps(m, indent=2))
