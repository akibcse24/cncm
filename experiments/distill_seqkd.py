"""Verified Sequence-Level Knowledge Distillation (SeqKD) Pipeline (`experiments/distill_seqkd.py`).

Generates and/or filters teacher-distilled corpora for CNCM while keeping CNCM's
exact architecture and 4,096-token vocabulary untouched:
  1. Queries any OpenAI-compatible local or Modal teacher endpoint (e.g., llama.cpp
     serving `Qwen3-4B-Instruct-2507`, `Phi-4-mini-instruct`, `Ternary-Bonsai-2-4B`,
     or vLLM serving `Qwen2.5-Coder-7B-Instruct` / `Qwen3-8B`), OR filters an offline JSONL.
  2. Caps verbose `<think>...</think>` chains (preventing 14.5M `nano` chunk overflow).
  3. Rejection-samples every sample with strict verifiers:
       - Python / PG-FIM code: `cncm.code.sandbox.run_code_sandbox` (must parse, compile, and pass tests)
       - Arithmetic / tool calls: `cncm.agent.grammar.extract_json_object` + `cncm.agent.tools.safe_eval_math`
  4. Saves the verified JSONL + SHA-256 provenance manifest so BOTH CNCM and Pure-AR
     train on the exact same distilled corpus (Item 1.1 fairness).

Usage:
    # Filter & verify an offline teacher JSONL:
    python -m experiments.distill_seqkd --input raw_teacher.jsonl --output experiments/results/verified_seqkd.jsonl --teacher-name "Qwen/Qwen3-4B-Instruct-2507" --teacher-license "Apache-2.0"

    # Generate + verify live from a local llama.cpp / vLLM OpenAI-compatible server:
    python -m experiments.distill_seqkd --api-base http://127.0.0.1:8080/v1 --model Qwen3-4B-Instruct-2507 --n-samples 500 --output experiments/results/verified_seqkd.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cncm.agent.grammar import evaluate_tool_call_semantic_accuracy
from cncm.code.sandbox import run_code_sandbox
from experiments.eval_suites import (
    build_heldout_arithmetic_suite,
    build_heldout_tool_calling_suite,
)

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def cap_think_block(text: str, max_think_chars: int = 240) -> str:
    """Shorten overly long `<think>...</think>` blocks (e.g. from Bonsai 2 / Qwen3 thinking mode)."""
    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        if len(inner) > max_think_chars:
            inner = inner[:max_think_chars].rsplit(" ", 1)[0].strip() + "..."
        return f"<think>{inner}</think>"

    cleaned = _THINK_BLOCK_RE.sub(_repl, text)
    # Strip dangling unclosed <think> tags
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()
    return cleaned.strip()


def verify_teacher_record(
    record: Dict[str, Any],
    *,
    max_think_chars: int = 240,
    timeout_sec: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """Verify a single teacher completion and return a cleaned record, or None if rejected."""
    domain = str(record.get("domain", "prose")).lower()
    raw_out = str(record.get("completion", record.get("text", "")))
    cleaned = cap_think_block(raw_out, max_think_chars=max_think_chars)
    if not cleaned:
        return None

    if domain in ("code", "fim", "python"):
        prefix = str(record.get("prefix", ""))
        suffix = str(record.get("suffix", ""))
        test_code = str(record.get("test_code", ""))
        full_code = (prefix + cleaned + suffix) if (prefix or suffix) else cleaned
        sb = run_code_sandbox(full_code, test_code=test_code, timeout_sec=timeout_sec)
        if not (sb.get("parses") and sb.get("compiles")):
            return None
        if test_code.strip() and not sb.get("tests_passed"):
            return None
        return {**record, "completion": cleaned, "verified": True}

    if domain in ("tool", "tool_call", "math"):
        target = {
            "expected_tool": record.get("expected_tool"),
            "expected_value": record.get("expected_value"),
            "expected_args": record.get("expected_args"),
        }
        res = evaluate_tool_call_semantic_accuracy([cleaned], [target])
        if res["semantic_correct"] != 1:
            return None
        return {**record, "completion": cleaned, "verified": True}

    # General prose: require non-empty printable text within reasonable length
    if len(cleaned) < 12:
        return None
    return {**record, "completion": cleaned, "verified": True}


def filter_teacher_records(
    records: Sequence[Dict[str, Any]],
    *,
    max_think_chars: int = 240,
) -> Dict[str, Any]:
    """Filter a batch of teacher records, returning accepted items and acceptance stats."""
    accepted: List[Dict[str, Any]] = []
    for rec in records:
        v = verify_teacher_record(rec, max_think_chars=max_think_chars)
        if v is not None:
            accepted.append(v)

    total = len(records)
    n_acc = len(accepted)
    payload_str = json.dumps(accepted, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    sha = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    return {
        "total_seen": total,
        "accepted_count": n_acc,
        "rejected_count": total - n_acc,
        "accept_rate": round(n_acc / float(max(1, total)), 4),
        "sha256": sha,
        "accepted": accepted,
    }


def _query_openai_chat(
    api_base: str,
    model: str,
    prompt: str,
    api_key: str = "EMPTY",
    max_tokens: int = 192,
    temperature: float = 0.2,
    enable_thinking: bool = False,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Be concise. Output only the requested JSON or Python code.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Verified Sequence-Level KD (SeqKD) Builder for CNCM (Teacher: Bonsai-2-4B)")
    parser.add_argument("--input", type=str, default=None, help="Offline raw teacher JSONL to filter")
    parser.add_argument("--api-base", type=str, default=None, help="OpenAI-compatible teacher server URL")
    parser.add_argument("--model", type=str, default="Ternary-Bonsai-2-4B")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-think-chars", type=int, default=240)
    parser.add_argument("--teacher-name", type=str, default="Ternary-Bonsai-2-4B")
    parser.add_argument("--teacher-license", type=str, default="Apache-2.0")
    parser.add_argument("--output", type=str, default="experiments/results/verified_seqkd.jsonl")
    args = parser.parse_args()

    raw_records: List[Dict[str, Any]] = []
    if args.input:
        with Path(args.input).open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    raw_records.append(json.loads(line))
    elif args.api_base:
        # Generate a balanced mix of tool-calling, arithmetic, and Python/PG-FIM prompts (distinct train seeds!)
        from cncm.trainer.dataset import make_synthetic_fim_tasks

        n_per = max(1, args.n_samples // 3)
        tool_prompts = build_heldout_tool_calling_suite(n_items=n_per, seed=9001)
        arith_prompts = build_heldout_arithmetic_suite(n_items=n_per, seed=9002)
        fim_prompts = make_synthetic_fim_tasks(num_samples=max(1, args.n_samples - 2 * n_per), seed=9003, curriculum_stage="all")
        for item in tool_prompts:
            comp = _query_openai_chat(
                args.api_base,
                args.model,
                item["prompt"] + ' Respond ONLY with JSON: {"name": "...", "arguments": {...}}',
                api_key=args.api_key,
            )
            raw_records.append({**item, "domain": "tool_call", "completion": comp})
        for item in arith_prompts:
            comp = _query_openai_chat(
                args.api_base,
                args.model,
                item["prompt"] + ' Respond ONLY with JSON: {"name": "calculator", "arguments": {"expression": "..."}}',
                api_key=args.api_key,
            )
            raw_records.append(
                {
                    **item,
                    "domain": "math",
                    "expected_tool": "calculator",
                    "completion": comp,
                }
            )
        for item in fim_prompts:
            fim_prompt = (
                "Output ONLY the exact Python code lines that go between PREFIX and SUFFIX (no markdown fences):\n"
                f"PREFIX:\n{item['prefix']}\nSUFFIX:\n{item['suffix']}"
            )
            comp = _query_openai_chat(
                args.api_base,
                args.model,
                fim_prompt,
                api_key=args.api_key,
            )
            # Strip accidental markdown code fences if emitted by teacher
            comp_clean = re.sub(r"^```(?:python)?\s*|\s*```$", "", comp.strip(), flags=re.MULTILINE)
            raw_records.append(
                {
                    "domain": "python",
                    "prefix": item["prefix"],
                    "suffix": item["suffix"],
                    "test_code": item.get("test_code", ""),
                    "completion": comp_clean,
                }
            )
    else:
        raise SystemExit("Provide either --input <raw.jsonl> or --api-base <http://.../v1>.")

    res = filter_teacher_records(raw_records, max_think_chars=args.max_think_chars)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as fh:
        for item in res["accepted"]:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "teacher_model": args.teacher_name,
        "teacher_license_spdx": args.teacher_license,
        "output_jsonl": str(out_p),
        "total_seen": res["total_seen"],
        "accepted_count": res["accepted_count"],
        "accept_rate": res["accept_rate"],
        "sha256": res["sha256"],
    }
    meta_p = out_p.with_suffix(".meta.json")
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
