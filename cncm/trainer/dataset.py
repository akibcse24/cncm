"""Datasets and deterministic synthetic corpora for CNCM training (Stage 1, Stage 2, PG-FIM, GRPO).

Fixes key MiniMind & NCM data pipeline pitfalls and implements PG-FIM (Plan-Guided FIM):
- PackedChunkDataset: concatenates tokenized texts with EOS and slices contiguous padding-free
  fixed `chunk_size` (e.g. L=64) windows (0% padding waste vs MiniMind's 340-token right padding).
- BlockARSFTDataset: unpacks multi-turn chat/tool conversations into causal
  `(prefix_ids, prefix_mask, target_chunk_ids)` chunk pairs supporting arbitrary response length,
  `<think>` traces, `<tool_call>` JSON spans, and aligned Exact-AR labels.
- FIMPlanDataset: implements PG-FIM Stage 2 joint training (50% PSM `<fim_prefix> P <fim_suffix> S
  <fim_middle> <mlen_k> <kw> kw1 | kw2 </kw> M <fim_end>` + 50% plain Block-AR), with 40/30/30
  AST/line/char span selection, offline keyword extraction (0..3 keywords), 15% `<kw/>` + 15%
  keyword corruption, and joint Exact-AR supervision over plan + target boundary tokens.
- GRPOPromptDataset: pairs prompts with verifiable reward specifications (math, tag structure,
  JSON tool-call accuracy, and compiler-in-the-loop FIM tasks).
- make_synthetic_corpus / make_synthetic_sft / make_synthetic_grpo_prompts /
  make_synthetic_code_corpus / make_pg_fim_stage1_corpus / make_synthetic_fim_tasks:
  deterministic, zero-download data generators for immediate training and verification.
"""

from __future__ import annotations

import ast
import json
import keyword
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers for tokenizer interoperability
# ---------------------------------------------------------------------------

def _get_token_id(tokenizer: Any, attr: str, default: int) -> int:
    val = getattr(tokenizer, attr, None)
    return int(val) if val is not None else default


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.flatten().tolist()]
    return [int(x) for x in ids]


def _format_messages(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    tools: Optional[Sequence[dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
        try:
            rendered = tokenizer.apply_chat_template(
                list(messages),
                tools=list(tools) if tools else None,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(rendered, str):
                return rendered
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                list(messages),
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(rendered, str):
                return rendered

    parts: list[str] = []
    if tools:
        parts.append(f"system: Available tools: {json.dumps(list(tools), ensure_ascii=False)}")
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        content = str(msg.get("content", ""))
        if role == "tool" and "<tool_response>" not in content:
            parts.append(f"tool: <tool_response>{content}</tool_response>")
        else:
            parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("assistant: ")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic Zero-Download Synthetic Corpora (Prose, Agentic SFT, GRPO)
# ---------------------------------------------------------------------------

_SYNTHETIC_TOPICS: tuple[tuple[str, str, str], ...] = (
    (
        "multiscale residual finite scalar quantization",
        "implicit integer grids avoid codebook collapse and eliminate EMA memory overhead",
        "each residual level refines coarse concepts into exact token sequences",
    ),
    (
        "causal prefix block autoregressive decoding",
        "prefixes maintain an incremental causal KV cache across conversation turns",
        "target concept levels attend in block diagonal form with zero target cache growth",
    ),
    (
        "relaxed recursive layer looping",
        "shared transformer blocks execute multiple passes with lightweight low rank adapters",
        "edge models achieve deep reasoning capacity inside tight smartphone RAM budgets",
    ),
    (
        "factorized joint concept prediction",
        "chained micro heads model correlated quantization digits in compact matrix operations",
        "parallel decoding produces sixty four tokens in only eight backbone passes",
    ),
    (
        "dual mode exact action agentic execution",
        "fast concept chunks draft thoughts and prose while exact mode emits strict JSON",
        "grammar masks guarantee schema valid tool calls even at tiny parameter counts",
    ),
    (
        "self speculative concept verification",
        "drafted concept chunks are verified in a single parallel exact forward pass",
        "lossless acceptance matches greedy autoregressive output with higher throughput",
    ),
    (
        "group relative policy optimization in code space",
        "parallel concept rollouts yield exact one pass trajectory log probabilities",
        "verifiable math and tool rewards align reasoning traces without a separate critic",
    ),
    (
        "group wise four bit weight packing",
        "two four bit integers pack into each byte with per group floating point scales",
        "compact models fit comfortably inside fifteen megabytes on low end mobile devices",
    ),
)

_BUILTIN_TOOLS_META: list[dict[str, Any]] = [
    {
        "name": "calculator",
        "description": "Evaluate a safe arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
    {
        "name": "kv_memory",
        "description": "Read or write a key-value entry in local working memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set"]},
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["action", "key"],
        },
    },
    {
        "name": "text_stats",
        "description": "Count characters and words in a text snippet.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "clock_utc",
        "description": "Return the current UTC timestamp.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def make_synthetic_corpus(num_samples: int = 256, seed: int = 42) -> list[str]:
    """Generate a deterministic, zero-download training corpus for Stage 1 & Stage 2 pretrain."""
    rng = random.Random(seed)
    samples: list[str] = []

    for i in range(max(1, int(num_samples))):
        kind = i % 4
        topic, fact_a, fact_b = _SYNTHETIC_TOPICS[i % len(_SYNTHETIC_TOPICS)]
        if kind == 0:
            n1 = rng.randint(4, 64)
            n2 = rng.randint(2, 16)
            text = (
                f"Section {i + 1}: In {topic}, {fact_a}. "
                f"When processing chunks of {n1} tokens across {n2} steps, {fact_b}. "
                f"Consistent padding-free packing improves throughput and keeps gradients stable."
            )
        elif kind == 1:
            a = rng.randint(3, 45)
            b = rng.randint(2, 35)
            op = rng.choice(["+", "-", "*"])
            ans = a + b if op == "+" else (a - b if op == "-" else a * b)
            text = (
                f"Question: Compute {a} {op} {b}. "
                f"<think>Let us evaluate {a} {op} {b} step by step. "
                f"The exact arithmetic value is {ans}.</think> "
                f"The answer is {ans}."
            )
        elif kind == 2:
            a = rng.randint(10, 99)
            b = rng.randint(2, 19)
            expr = f"{a} * {b} + {a - b}"
            ans = a * b + (a - b)
            payload = json.dumps({"name": "calculator", "arguments": {"expression": expr}})
            resp = json.dumps({"result": ans})
            text = (
                f"User: Calculate {expr}. "
                f"Assistant: <think>I should invoke the calculator tool for exact arithmetic.</think>"
                f"<tool_call>{payload}</tool_call>"
                f"<tool_response>{resp}</tool_response>"
                f"The calculated result for {expr} is {ans}."
            )
        else:
            key = rng.choice(["project", "preset", "target_ram", "chunk_len", "device"])
            val = rng.choice(["cncm", "nano", "15MB", "64", "android_cpu"])
            payload = json.dumps(
                {"name": "kv_memory", "arguments": {"action": "set", "key": key, "value": val}}
            )
            resp = json.dumps({"status": "ok", "key": key, "value": val})
            text = (
                f"Note {i}: Remember {key} equals {val}. "
                f"<think>Store {key}={val} in working memory so later turns stay consistent.</think>"
                f"<tool_call>{payload}</tool_call>"
                f"<tool_response>{resp}</tool_response>"
                f"Saved {key} as {val}. In addition, {fact_a}."
            )
        samples.append(text)

    return samples


def make_synthetic_sft(num_samples: int = 128, seed: int = 42) -> list[dict[str, Any]]:
    """Generate deterministic multi-turn chat & tool-calling SFT conversations."""
    rng = random.Random(seed)
    dataset: list[dict[str, Any]] = []

    system_prompt = (
        "You are CNCM, a compact dual-mode agentic assistant. "
        "Reason inside <think>...</think> tags and invoke tools with exact JSON inside "
        "<tool_call>...</tool_call> tags when needed."
    )

    for i in range(max(1, int(num_samples))):
        kind = i % 5
        if kind == 0:
            a = rng.randint(12, 95)
            b = rng.randint(3, 25)
            c = rng.randint(1, 50)
            expr = f"({a} * {b}) + {c}"
            result = (a * b) + c
            call_json = json.dumps({"name": "calculator", "arguments": {"expression": expr}})
            resp_json = json.dumps({"result": result})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Compute the exact value of {expr}."},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>To avoid arithmetic error on {expr}, I will call the calculator tool.</think>"
                        f"<tool_call>{call_json}</tool_call>"
                    ),
                },
                {"role": "tool", "content": resp_json},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>The calculator returned {result}. I can report the final answer clearly.</think>"
                        f"The exact value of {expr} is {result}."
                    ),
                },
            ]
            category = "tool_calculator"

        elif kind == 1:
            key = rng.choice(["budget_mb", "model_name", "owner", "target_hw", "chunk_size"])
            val = rng.choice(["15MB", "cncm-nano", "edge-team", "android", "64"])
            set_call = json.dumps(
                {"name": "kv_memory", "arguments": {"action": "set", "key": key, "value": val}}
            )
            set_resp = json.dumps({"status": "saved", "key": key, "value": val})
            get_call = json.dumps(
                {"name": "kv_memory", "arguments": {"action": "get", "key": key}}
            )
            get_resp = json.dumps({"key": key, "value": val})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Save {key} = {val} in memory."},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>I should store the key '{key}' with value '{val}' using kv_memory.</think>"
                        f"<tool_call>{set_call}</tool_call>"
                    ),
                },
                {"role": "tool", "content": set_resp},
                {
                    "role": "assistant",
                    "content": f"<think>Stored successfully.</think>I have saved {key} = {val}.",
                },
                {"role": "user", "content": f"What value is stored under {key}?"},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>Let me retrieve '{key}' from kv_memory to verify.</think>"
                        f"<tool_call>{get_call}</tool_call>"
                    ),
                },
                {"role": "tool", "content": get_resp},
                {
                    "role": "assistant",
                    "content": f"<think>The lookup confirmed {val}.</think>The stored value for {key} is {val}.",
                },
            ]
            category = "tool_memory"

        elif kind == 2:
            topic, fact_a, _ = _SYNTHETIC_TOPICS[i % len(_SYNTHETIC_TOPICS)]
            snippet = f"{topic}: {fact_a}"
            words = len(snippet.split())
            chars = len(snippet)
            call_json = json.dumps({"name": "text_stats", "arguments": {"text": snippet}})
            resp_json = json.dumps({"words": words, "chars": chars})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Count the words and characters in: '{snippet}'"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>Use the text_stats tool for exact word and character counts.</think>"
                        f"<tool_call>{call_json}</tool_call>"
                    ),
                },
                {"role": "tool", "content": resp_json},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>text_stats reports {words} words and {chars} characters.</think>"
                        f"The text has {words} words and {chars} characters."
                    ),
                },
            ]
            category = "tool_text_stats"

        elif kind == 3:
            topic, fact_a, fact_b = _SYNTHETIC_TOPICS[i % len(_SYNTHETIC_TOPICS)]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Explain why {topic} helps compact edge language models."},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>First recall the core mechanism of {topic}: {fact_a}. "
                        f"Next connect this mechanism to edge latency and memory: {fact_b}. "
                        f"Present a concise two-point explanation.</think>"
                        f"{topic.capitalize()} improves edge deployment in two ways: "
                        f"first, {fact_a}; second, {fact_b}. Together these reduce memory traffic "
                        f"and accelerate multi-chunk generation on mobile CPUs and laptop iGPUs."
                    ),
                },
            ]
            category = "reasoning"

        else:
            x = rng.randint(5, 30)
            y = rng.randint(4, 20)
            z = rng.randint(2, 9)
            s1 = x + y
            s2 = s1 * z
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"What is {x} + {y}?"},
                {
                    "role": "assistant",
                    "content": f"<think>Add {x} and {y} to get {s1}.</think>{x} + {y} = {s1}.",
                },
                {"role": "user", "content": f"Now multiply that result by {z}."},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>The previous sum is {s1}. Multiplying {s1} by {z} gives {s2}.</think>"
                        f"{s1} * {z} = {s2}."
                    ),
                },
            ]
            category = "multi_turn"

        dataset.append({"messages": messages, "tools": _BUILTIN_TOOLS_META, "category": category})

    return dataset


def make_synthetic_grpo_prompts(num_samples: int = 64, seed: int = 42) -> list[dict[str, Any]]:
    """Generate deterministic prompts + verifiable reward specs for GRPO training."""
    rng = random.Random(seed)
    prompts: list[dict[str, Any]] = []
    system_prompt = (
        "You are CNCM. Think inside <think>...</think> and call tools using JSON inside "
        "<tool_call>...</tool_call> when required."
    )

    for i in range(max(1, int(num_samples))):
        kind = i % 3
        if kind == 0:
            a = rng.randint(5, 50)
            b = rng.randint(2, 25)
            ans = a + b
            user_msg = f"Compute {a} + {b}. Show reasoning in <think> tags and give the final number."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(
                {
                    "prompt": f"user: {user_msg}\nassistant: ",
                    "messages": messages,
                    "tools": _BUILTIN_TOOLS_META,
                    "task_type": "math",
                    "expected_answer": str(ans),
                    "expected_tool": None,
                    "expected_args": None,
                    "require_think": True,
                    "require_tool_call": False,
                }
            )
        elif kind == 1:
            a = rng.randint(6, 40)
            b = rng.randint(3, 18)
            expr = f"{a} * {b}"
            ans = a * b
            user_msg = f"Use the calculator tool to compute {expr}."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(
                {
                    "prompt": f"user: {user_msg}\nassistant: ",
                    "messages": messages,
                    "tools": _BUILTIN_TOOLS_META,
                    "task_type": "tool_call",
                    "expected_answer": str(ans),
                    "expected_tool": "calculator",
                    "expected_args": {"expression": expr},
                    "require_think": True,
                    "require_tool_call": True,
                }
            )
        else:
            key = rng.choice(["mode", "preset", "status", "target"])
            val = rng.choice(["concept", "nano", "ready", "edge"])
            user_msg = f"Store {key}={val} using kv_memory."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(
                {
                    "prompt": f"user: {user_msg}\nassistant: ",
                    "messages": messages,
                    "tools": _BUILTIN_TOOLS_META,
                    "task_type": "tool_call",
                    "expected_answer": val,
                    "expected_tool": "kv_memory",
                    "expected_args": {"action": "set", "key": key, "value": val},
                    "require_think": True,
                    "require_tool_call": True,
                }
            )

    return prompts


# ---------------------------------------------------------------------------
# PG-FIM Synthetic Code Corpus, Stage 1 Mix Corpus & Curriculum FIM Tasks
# ---------------------------------------------------------------------------

def _build_fim_task_bank(seed: int = 42) -> list[dict[str, Any]]:
    """Build deterministic bank of Python FIM tasks across single_line, multi_line, function_body."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    # 1. single_line tasks
    for idx in range(12):
        mult = rng.randint(2, 6)
        bias = rng.randint(1, 15)
        kind = idx % 4
        if kind == 0:
            prefix = f"def scale_and_shift(val, factor={mult}, offset={bias}):\n    "
            middle = "res = val * factor + offset"
            suffix = "\n    return res\n"
            test_code = (
                f"assert scale_and_shift(3) == {3 * mult + bias}\n"
                f"assert scale_and_shift(0) == {bias}\n"
            )
            kws = ["res", "val", "factor"]
            fname = "scale_and_shift"
        elif kind == 1:
            prefix = (
                "def sum_positive(nums):\n"
                "    total = 0\n"
                "    for val in nums:\n"
                "        if val > 0:\n"
                "            "
            )
            middle = "total += val"
            suffix = "\n    return total\n"
            test_code = (
                "assert sum_positive([1, -2, 4, 0, 5]) == 10\n"
                "assert sum_positive([-3, -1]) == 0\n"
            )
            kws = ["total", "val"]
            fname = "sum_positive"
        elif kind == 2:
            prefix = (
                "def midpoint_index(left, right):\n"
                "    if right < left:\n"
                "        return -1\n"
                "    "
            )
            middle = "mid = left + (right - left) // 2"
            suffix = "\n    return mid\n"
            test_code = (
                "assert midpoint_index(2, 10) == 6\n"
                "assert midpoint_index(5, 2) == -1\n"
            )
            kws = ["mid", "left", "right"]
            fname = "midpoint_index"
        else:
            prefix = (
                "def dot_pair(vec_a, vec_b):\n"
                "    acc = 0\n"
                "    for x, y in zip(vec_a, vec_b):\n"
                "        "
            )
            middle = "acc += x * y"
            suffix = "\n    return acc\n"
            test_code = (
                "assert dot_pair([1, 2, 3], [4, 5, 6]) == 32\n"
                "assert dot_pair([2, 0], [5, 9]) == 10\n"
            )
            kws = ["acc", "x", "y"]
            fname = "dot_pair"

        tasks.append(
            {
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

    # 2. multi_line tasks
    for idx in range(12):
        kind = idx % 4
        if kind == 0:
            prefix = "def clamp_number(val, low, high):\n"
            middle = (
                "    if val < low:\n"
                "        return low\n"
                "    if val > high:\n"
                "        return high"
            )
            suffix = "\n    return val\n"
            test_code = (
                "assert clamp_number(5, 1, 10) == 5\n"
                "assert clamp_number(-2, 0, 8) == 0\n"
                "assert clamp_number(15, 0, 8) == 8\n"
            )
            kws = ["val", "low", "high"]
            fname = "clamp_number"
        elif kind == 1:
            prefix = (
                "def binary_search(arr, target):\n"
                "    low = 0\n"
                "    high = len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
            )
            middle = (
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1"
            )
            suffix = "\n    return -1\n"
            test_code = (
                "assert binary_search([1, 3, 5, 7, 9], 7) == 3\n"
                "assert binary_search([1, 3, 5, 7, 9], 4) == -1\n"
            )
            kws = ["arr", "mid", "target"]
            fname = "binary_search"
        elif kind == 2:
            prefix = (
                "def count_frequencies(items):\n"
                "    counts = {}\n"
            )
            middle = (
                "    for item in items:\n"
                "        if item in counts:\n"
                "            counts[item] += 1\n"
                "        else:\n"
                "            counts[item] = 1"
            )
            suffix = "\n    return counts\n"
            test_code = (
                "assert count_frequencies(['a', 'b', 'a']) == {'a': 2, 'b': 1}\n"
                "assert count_frequencies([]) == {}\n"
            )
            kws = ["for", "item", "counts"]
            fname = "count_frequencies"
        else:
            prefix = (
                "def filter_even_squares(nums):\n"
                "    out = []\n"
            )
            middle = (
                "    for n in nums:\n"
                "        if n % 2 == 0:\n"
                "            out.append(n * n)"
            )
            suffix = "\n    return out\n"
            test_code = (
                "assert filter_even_squares([1, 2, 3, 4]) == [4, 16]\n"
                "assert filter_even_squares([1, 3, 5]) == []\n"
            )
            kws = ["for", "out", "nums"]
            fname = "filter_even_squares"

        tasks.append(
            {
                "curriculum_stage": "multi_line",
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

    # 3. function_body tasks
    for idx in range(12):
        kind = idx % 4
        if kind == 0:
            prefix = "def fibonacci_iter(n):\n"
            middle = (
                "    if n <= 0:\n"
                "        return 0\n"
                "    prev, curr = 0, 1\n"
                "    for _ in range(1, n):\n"
                "        prev, curr = curr, prev + curr"
            )
            suffix = "\n    return curr\n"
            test_code = (
                "assert fibonacci_iter(0) == 0\n"
                "assert fibonacci_iter(1) == 1\n"
                "assert fibonacci_iter(6) == 8\n"
            )
            kws = ["prev", "curr", "range"]
            fname = "fibonacci_iter"
        elif kind == 1:
            prefix = "def matrix_trace(grid):\n"
            middle = (
                "    total = 0\n"
                "    limit = len(grid)\n"
                "    for idx in range(limit):\n"
                "        total += grid[idx][idx]"
            )
            suffix = "\n    return total\n"
            test_code = (
                "assert matrix_trace([[2, 1], [3, 5]]) == 7\n"
                "assert matrix_trace([[4]]) == 4\n"
            )
            kws = ["total", "idx", "grid"]
            fname = "matrix_trace"
        elif kind == 2:
            prefix = "def max_window_sum(arr, k):\n"
            middle = (
                "    if k <= 0 or k > len(arr):\n"
                "        return 0\n"
                "    curr = sum(arr[:k])\n"
                "    best = curr\n"
                "    for idx in range(k, len(arr)):\n"
                "        curr += arr[idx] - arr[idx - k]\n"
                "        if curr > best:\n"
                "            best = curr"
            )
            suffix = "\n    return best\n"
            test_code = (
                "assert max_window_sum([1, 4, 2, 10, 2], 2) == 12\n"
                "assert max_window_sum([3, 1], 5) == 0\n"
            )
            kws = ["curr", "best", "idx"]
            fname = "max_window_sum"
        else:
            prefix = "def merge_sorted_lists(left, right):\n"
            middle = (
                "    res = []\n"
                "    i, j = 0, 0\n"
                "    while i < len(left) and j < len(right):\n"
                "        if left[i] <= right[j]:\n"
                "            res.append(left[i])\n"
                "            i += 1\n"
                "        else:\n"
                "            res.append(right[j])\n"
                "            j += 1\n"
                "    res.extend(left[i:])\n"
                "    res.extend(right[j:])"
            )
            suffix = "\n    return res\n"
            test_code = (
                "assert merge_sorted_lists([1, 4, 6], [2, 3, 7]) == [1, 2, 3, 4, 6, 7]\n"
                "assert merge_sorted_lists([], [1, 2]) == [1, 2]\n"
            )
            kws = ["res", "left", "right"]
            fname = "merge_sorted_lists"

        tasks.append(
            {
                "curriculum_stage": "function_body",
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

    return tasks


def make_synthetic_fim_tasks(
    num_samples: int = 64,
    seed: int = 42,
    curriculum_stage: str = "all",
) -> list[dict[str, Any]]:
    """Generate deterministic Python FIM tasks with unit tests and 3 curriculum tiers.

    Args:
        num_samples: Number of tasks to return.
        seed: Deterministic RNG seed.
        curriculum_stage: `"single_line"`, `"multi_line"`, `"function_body"`, or `"all"`.
    """
    stage_norm = str(curriculum_stage).strip().lower()
    valid_stages = ("single_line", "multi_line", "function_body", "all")
    if stage_norm not in valid_stages:
        raise ValueError(f"curriculum_stage must be one of {valid_stages}, got {curriculum_stage!r}")

    bank = _build_fim_task_bank(seed=seed)
    if stage_norm != "all":
        bank = [t for t in bank if t["curriculum_stage"] == stage_norm]

    out: list[dict[str, Any]] = []
    for i in range(max(1, int(num_samples))):
        base = dict(bank[i % len(bank)])
        base["task_id"] = f"fim_{base['curriculum_stage']}_{i}"
        base["oracle_keywords"] = list(base["oracle_keywords"])
        out.append(base)
    return out


def make_synthetic_code_corpus(num_samples: int = 256, seed: int = 42) -> list[str]:
    """Generate deterministic, self-contained valid Python source files for code pretraining."""
    rng = random.Random(seed)
    tasks = _build_fim_task_bank(seed=seed)
    samples: list[str] = []

    for i in range(max(1, int(num_samples))):
        t1 = tasks[i % len(tasks)]
        t2 = tasks[(i + 5) % len(tasks)]
        if i % 3 == 0:
            code = f"{t1['full_code']}\n{t1['test_code']}".strip()
        elif i % 3 == 1:
            code = f"{t1['full_code']}\n\n{t2['full_code']}".strip()
        else:
            step = rng.randint(1, 9)
            limit = rng.randint(10, 50)
            code = (
                f"def accumulate_step_{i % 16}(items, step={step}, limit={limit}):\n"
                f"    acc = 0\n"
                f"    for idx, val in enumerate(items):\n"
                f"        if val <= limit:\n"
                f"            acc += val + idx * step\n"
                f"    return acc\n"
            )
        samples.append(code)
    return samples


def make_pg_fim_stage1_corpus(num_samples: int = 256, seed: int = 42) -> list[str]:
    """Generate a 60/40 code + prose Stage 1 corpus enriched with all 11 PG-FIM sentinel tokens.

    Ensures Stage 1 M-RFSQ (and Phase B Copy-Aware Decoder) achieves high reconstruction
    accuracy on Python identifiers, syntax tokens, and all PG-FIM sentinels.
    """
    n_total = max(1, int(num_samples))
    n_code = max(1, int(round(n_total * 0.60)))
    n_prose = max(1, n_total - n_code)

    code_raw = make_synthetic_code_corpus(num_samples=n_code, seed=seed)
    fim_tasks = make_synthetic_fim_tasks(num_samples=n_code, seed=seed, curriculum_stage="all")
    prose_raw = make_synthetic_corpus(num_samples=n_prose, seed=seed + 1)

    mlen_tags = ("<mlen_1>", "<mlen_2>", "<mlen_3>", "<mlen_4+>")
    combined: list[str] = []

    for i in range(n_code):
        if i % 2 == 0:
            # Explicit PG-FIM sequence with sentinels, <mlen_*>, and <kw>...</kw> or <kw/>
            task = fim_tasks[(i // 2) % len(fim_tasks)]
            mlen_tag = mlen_tags[(i // 2) % len(mlen_tags)]
            if i % 6 == 0:
                kw_block = "<kw/>"
            else:
                kws = " | ".join(task["oracle_keywords"][:3])
                kw_block = f"<kw> {kws} </kw>"
            fim_doc = (
                f"<fim_prefix>{task['prefix']}"
                f"<fim_suffix>{task['suffix']}"
                f"<fim_middle>{mlen_tag}{kw_block}{task['middle']}<fim_end>"
            )
            combined.append(fim_doc)
        else:
            combined.append(code_raw[i])

    for j in range(n_total - len(combined)):
        combined.append(prose_raw[j % len(prose_raw)])

    return combined[:n_total]


# ---------------------------------------------------------------------------
# Built-in AST / Line / Char Span Splitter & Keyword Extractor for PG-FIM
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUM_LIT_RE = re.compile(r"\b\d+\b")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "with", "this", "from", "have", "not", "are",
        "was", "were", "will", "would", "can", "could", "should", "into", "about",
        "section", "when", "across", "first", "second", "self", "true", "false", "none",
    }
)
_DISTRACTOR_IDENTIFIERS: tuple[str, ...] = (
    "idx", "total", "acc", "res", "val", "curr", "prev", "left", "right",
    "mid", "low", "high", "items", "nums", "arr", "out", "target", "limit",
    "count", "node", "key", "value", "step", "offset", "factor",
)


def _fallback_extract_spans(
    text: str,
    mode: str = "mix",
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, str]:
    """Split `text` into `(prefix, middle, suffix)` using 40% AST, 30% line, 30% random char."""
    local_rng = rng if rng is not None else random.Random(42)
    if not text or len(text) < 12:
        mid_pt = max(1, len(text) // 2)
        return text[:mid_pt], text[mid_pt:], ""

    chosen_mode = mode
    if mode == "mix":
        u = local_rng.random()
        if u < 0.40:
            chosen_mode = "ast"
        elif u < 0.70:
            chosen_mode = "line"
        else:
            chosen_mode = "random"

    lines = text.splitlines(keepends=True)

    # 1. Try AST-node span (function body, if/for/while block, or statement)
    if chosen_mode == "ast" and len(lines) >= 2:
        try:
            tree = ast.parse(text)
            candidate_ranges: list[tuple[int, int]] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.For, ast.While)):
                    body = getattr(node, "body", None)
                    if body:
                        start_line = int( getattr(body[0], "lineno", 0) )
                        end_line = int( getattr(body[-1], "end_lineno", getattr(body[-1], "lineno", 0)) )
                        if 1 < start_line <= end_line <= len(lines):
                            candidate_ranges.append((start_line - 1, end_line))
                elif isinstance(node, (ast.Return, ast.Assign, ast.AugAssign)):
                    s_line = int(getattr(node, "lineno", 0))
                    e_line = int(getattr(node, "end_lineno", s_line))
                    if 1 < s_line <= e_line <= len(lines):
                        candidate_ranges.append((s_line - 1, e_line))
            if candidate_ranges:
                s_idx, e_idx = local_rng.choice(candidate_ranges)
                prefix = "".join(lines[:s_idx])
                middle = "".join(lines[s_idx:e_idx]).rstrip("\n")
                suffix = "\n" + "".join(lines[e_idx:]) if e_idx < len(lines) else "\n"
                if middle.strip():
                    return prefix, middle, suffix
        except Exception:
            pass
        chosen_mode = "line"

    # 2. Whole-line span
    if chosen_mode == "line" and len(lines) >= 3:
        max_span = max(1, min(4, len(lines) - 2))
        span_len = local_rng.randint(1, max_span)
        start_idx = local_rng.randint(1, max(1, len(lines) - span_len - 1))
        end_idx = min(len(lines) - 1, start_idx + span_len)
        prefix = "".join(lines[:start_idx])
        middle = "".join(lines[start_idx:end_idx]).rstrip("\n")
        suffix = "\n" + "".join(lines[end_idx:])
        if middle.strip():
            return prefix, middle, suffix

    # 3. Random character span (snapped to token boundaries where possible)
    n = len(text)
    s_pos = local_rng.randint(max(1, n // 5), max(2, (n * 3) // 5))
    e_pos = local_rng.randint(min(n - 1, s_pos + 4), max(min(n - 1, s_pos + 4), (n * 4) // 5))
    return text[:s_pos], text[s_pos:e_pos], text[e_pos:]


def _fallback_extract_keywords(
    middle: str,
    prefix: str = "",
    suffix: str = "",
    max_kw: int = 3,
    is_code: bool = True,
) -> list[str]:
    """Extract up to `max_kw` priority keywords (identifiers, control keywords, literals)."""
    del prefix, suffix
    if max_kw <= 0 or not middle.strip():
        return []

    priority_ctrl = {"return", "for", "while", "if", "elif", "else", "yield", "raise", "assert"}
    candidates: list[str] = []
    seen: set[str] = set()

    def _push(tok: str) -> None:
        t = tok.strip()
        if not t or t in seen or len(t) > 24:
            return
        seen.add(t)
        candidates.append(t)

    if is_code:
        # First pass: non-keyword identifiers defined or used in middle
        for match in _IDENT_RE.finditer(middle):
            tok = match.group(0)
            if not keyword.iskeyword(tok) and tok.lower() not in _STOPWORDS:
                _push(tok)
        # Second pass: control-flow keywords in middle
        for match in _IDENT_RE.finditer(middle):
            tok = match.group(0)
            if tok in priority_ctrl:
                _push(tok)
        # Third pass: numeric literals
        for match in _NUM_LIT_RE.finditer(middle):
            _push(match.group(0))
    else:
        for match in _IDENT_RE.finditer(middle):
            tok = match.group(0)
            if len(tok) >= 4 and tok.lower() not in _STOPWORDS:
                _push(tok)
        for match in _NUM_LIT_RE.finditer(middle):
            _push(match.group(0))

    return candidates[: int(max_kw)]


def _extract_spans_interop(
    text: str,
    mode: str = "mix",
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, str]:
    try:
        from cncm.code.symbols import extract_ast_or_line_spans  # type: ignore[import-not-found]

        p, m, s = extract_ast_or_line_spans(text, mode=mode, rng=rng)
        if m:
            return p, m, s
    except Exception:
        pass
    return _fallback_extract_spans(text, mode=mode, rng=rng)


def _extract_keywords_interop(
    middle: str,
    prefix: str = "",
    suffix: str = "",
    max_kw: int = 3,
    is_code: bool = True,
) -> list[str]:
    try:
        from cncm.code.symbols import extract_oracle_keywords  # type: ignore[import-not-found]

        kws = extract_oracle_keywords(
            middle, prefix=prefix, suffix=suffix, max_kw=max_kw, is_code=is_code
        )
        if isinstance(kws, list):
            return [str(k) for k in kws[:max_kw] if str(k).strip()]
    except Exception:
        pass
    return _fallback_extract_keywords(
        middle, prefix=prefix, suffix=suffix, max_kw=max_kw, is_code=is_code
    )


# ---------------------------------------------------------------------------
# PackedChunkDataset (Zero-Padding Stage 1 & Stage 2 Pretrain)
# ---------------------------------------------------------------------------

def _load_texts_source(source: Union[Sequence[str], str, Path]) -> list[str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return [str(source)]
        texts: list[str] = []
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        val = obj.get("text") or obj.get("content") or ""
                        if val:
                            texts.append(str(val))
                    elif isinstance(obj, str):
                        texts.append(obj)
        else:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        texts.append(line)
        return texts
    return [str(t) for t in source if t]


class PackedChunkDataset(Dataset):
    """Padding-free packed chunk dataset for CNCM Stage 1 and Stage 2 pretraining."""

    def __init__(
        self,
        texts: Union[Sequence[str], str, Path],
        tokenizer: Any,
        chunk_size: int = 64,
        prefix_len: int = 0,
        stride: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        add_bos: bool = False,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")

        self.chunk_size = int(chunk_size)
        self.prefix_len = int(prefix_len)
        self.window_size = self.prefix_len + self.chunk_size
        self.stride = int(stride) if stride is not None else self.chunk_size
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

        self.eos_token_id = (
            int(eos_token_id) if eos_token_id is not None else _get_token_id(tokenizer, "eos_token_id", 2)
        )
        self.bos_token_id = (
            int(bos_token_id) if bos_token_id is not None else _get_token_id(tokenizer, "bos_token_id", 1)
        )

        docs = _load_texts_source(texts)
        if not docs:
            raise ValueError("PackedChunkDataset received an empty corpus.")

        stream: list[int] = []
        for doc in docs:
            ids = _encode_text(tokenizer, doc)
            if not ids:
                continue
            if add_bos:
                stream.append(self.bos_token_id)
            stream.extend(ids)
            stream.append(self.eos_token_id)

        if not stream:
            stream = [self.bos_token_id, self.eos_token_id]

        if len(stream) < self.window_size:
            repeats = (self.window_size + len(stream) - 1) // len(stream)
            stream = (stream * repeats)[: self.window_size]

        self.tokens = torch.tensor(stream, dtype=torch.long)
        self._offsets = list(range(0, len(self.tokens) - self.window_size + 1, self.stride))
        if not self._offsets:
            self._offsets = [0]

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.numel())

    @property
    def pad_ratio(self) -> float:
        return 0.0

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self._offsets[idx]
        window = self.tokens[start : start + self.window_size]
        if self.prefix_len == 0:
            return {
                "input_ids": window.clone(),
                "attention_mask": torch.ones(self.chunk_size, dtype=torch.long),
            }

        prefix_ids = window[: self.prefix_len].clone()
        target_ids = window[self.prefix_len :].clone()
        prefix_mask = torch.ones(self.prefix_len, dtype=torch.long)
        target_mask = torch.ones(self.chunk_size, dtype=torch.long)
        exact_labels = torch.full((self.window_size,), -100, dtype=torch.long)
        exact_labels[self.prefix_len :] = target_ids
        return {
            "prefix_ids": prefix_ids,
            "prefix_mask": prefix_mask,
            "target_ids": target_ids,
            "target_chunk_ids": target_ids,
            "target_mask": target_mask,
            "input_ids": window.clone(),
            "attention_mask": torch.ones(self.window_size, dtype=torch.long),
            "exact_input_ids": window.clone(),
            "exact_attention_mask": torch.ones(self.window_size, dtype=torch.long),
            "exact_labels": exact_labels,
            "has_tool_call": torch.tensor(0, dtype=torch.long),
            "plan_len": torch.tensor(0, dtype=torch.long),
            "is_fim": torch.tensor(0, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# MemmapTokenDataset (Memory-Mapped Pre-Tokenized Flat Binary Corpus)
# ---------------------------------------------------------------------------

def prepare_memmap_corpus(
    texts: Union[Iterable[str], Sequence[str], str, Path],
    tokenizer: Any,
    out_bin_path: Union[str, Path],
    *,
    eos_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    add_bos: bool = False,
    dtype: Any = np.uint16,
    flush_tokens: int = 1_000_000,
) -> Path:
    """Pre-tokenize `texts` directly to a flat binary token stream (`np.uint16` by default).

    Writes token IDs incrementally to disk so a 300M-token corpus (~600 MB on disk) can be
    built and later memory-mapped via `MemmapTokenDataset` with <10 MB RAM and <0.1s load time.

    Args:
        texts: Iterable/sequence of strings, or path to a `.txt`/`.jsonl` file.
        tokenizer: Tokenizer implementing `.encode(text, add_special_tokens=False)`.
        out_bin_path: Destination `.bin` path (written atomically via temp file + `os.replace`).
        eos_token_id: EOS token ID appended after each document.
        bos_token_id: Optional BOS token ID prepended when `add_bos=True`.
        add_bos: Whether to prepend `bos_token_id` before each document.
        dtype: NumPy integer dtype (default `np.uint16`, supports `vocab_size <= 65535`).
        flush_tokens: Number of tokens buffered in RAM before flushing to disk.

    Returns:
        Path to the written `.bin` file.
    """
    np_dtype = np.dtype(dtype)
    if not np.issubdtype(np_dtype, np.integer):
        raise ValueError(f"dtype must be a NumPy integer type, got {dtype!r}")
    dtype_info = np.iinfo(np_dtype)
    min_val = int(dtype_info.min)
    max_val = int(dtype_info.max)

    eos_id = int(eos_token_id) if eos_token_id is not None else _get_token_id(tokenizer, "eos_token_id", 2)
    bos_id = int(bos_token_id) if bos_token_id is not None else _get_token_id(tokenizer, "bos_token_id", 1)

    if isinstance(texts, (str, Path)):
        doc_iter: Iterable[str] = _load_texts_source(texts)
    else:
        doc_iter = texts

    out_path = Path(out_bin_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    buf: list[int] = []
    total_written = 0
    flush_thresh = max(1024, int(flush_tokens))

    def _flush(fh: Any) -> None:
        nonlocal total_written
        if not buf:
            return
        lo = min(buf)
        hi = max(buf)
        if lo < min_val or hi > max_val:
            raise ValueError(
                f"Token IDs out of range for {np_dtype}: observed [{lo}, {hi}], "
                f"allowed [{min_val}, {max_val}]."
            )
        arr = np.asarray(buf, dtype=np_dtype)
        arr.tofile(fh)
        total_written += len(buf)
        buf.clear()

    try:
        has_batch_encode = hasattr(tokenizer, "batch_encode") and callable(tokenizer.batch_encode)
        doc_batch: list[str] = []
        with tmp_path.open("wb") as fh:
            for doc in doc_iter:
                if not doc:
                    continue
                doc_batch.append(str(doc))
                if len(doc_batch) >= 2048:
                    encoded_list = (
                        tokenizer.batch_encode(doc_batch, add_special_tokens=False)
                        if has_batch_encode
                        else [_encode_text(tokenizer, d) for d in doc_batch]
                    )
                    doc_batch.clear()
                    for ids in encoded_list:
                        if not ids:
                            continue
                        if add_bos:
                            buf.append(bos_id)
                        buf.extend(ids)
                        buf.append(eos_id)
                        if len(buf) >= flush_thresh:
                            _flush(fh)
            if doc_batch:
                encoded_list = (
                    tokenizer.batch_encode(doc_batch, add_special_tokens=False)
                    if has_batch_encode
                    else [_encode_text(tokenizer, d) for d in doc_batch]
                )
                doc_batch.clear()
                for ids in encoded_list:
                    if not ids:
                        continue
                    if add_bos:
                        buf.append(bos_id)
                    buf.extend(ids)
                    buf.append(eos_id)
                    if len(buf) >= flush_thresh:
                        _flush(fh)
            _flush(fh)

        if total_written == 0:
            raise ValueError("prepare_memmap_corpus produced 0 tokens from the provided corpus.")

        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return out_path


class MemmapTokenDataset(Dataset):
    """Memory-mapped pre-tokenized flat binary dataset (`np.memmap(..., mode="r")`).

    Supports:
    - 0% padding waste (identical dict schema to `PackedChunkDataset`).
    - <10 MB RAM footprint even on 300M+ token corpora (~600 MB `.bin` in `uint16`).
    - Deterministic window offset slicing (`start_offset`) so resuming mid-run skips
      already-consumed windows cleanly in O(1) time.
    - Rank-strided sharding (`rank`, `world_size`) for zero-copy DDP across multi-GPU runs.
    """

    def __init__(
        self,
        bin_path: Union[str, Path],
        chunk_size: int = 64,
        prefix_len: int = 256,
        dtype: Any = np.uint16,
        *,
        stride: Optional[int] = None,
        start_offset: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Memmap token binary not found: {self.bin_path}")

        self.dtype = np.dtype(dtype)
        self.chunk_size = int(chunk_size)
        self.prefix_len = int(prefix_len)
        self.window_size = self.prefix_len + self.chunk_size
        self.stride = int(stride) if stride is not None else self.chunk_size
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

        file_bytes = int(self.bin_path.stat().st_size)
        if file_bytes == 0 or (file_bytes % self.dtype.itemsize != 0):
            raise ValueError(
                f"Invalid memmap file size ({file_bytes} bytes) for dtype {self.dtype} "
                f"(itemsize={self.dtype.itemsize}) at {self.bin_path}."
            )
        self._num_tokens = file_bytes // self.dtype.itemsize

        if self._num_tokens >= self.window_size:
            self._total_windows = 1 + (self._num_tokens - self.window_size) // self.stride
        else:
            self._total_windows = 1

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_offset = max(0, int(start_offset))

        eff_start = self.start_offset % self._total_windows if self.start_offset >= self._total_windows else self.start_offset
        rem = self._total_windows - eff_start
        if rem > self.rank:
            self._effective_start = eff_start
            self._len = 1 + (rem - 1 - self.rank) // self.world_size
        else:
            self._effective_start = 0
            self._len = max(1, (self._total_windows - self.rank + self.world_size - 1) // self.world_size)

        self._mm: Optional[np.memmap] = None

    def _ensure_memmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype=self.dtype, mode="r", shape=(self._num_tokens,))
        return self._mm

    def close(self) -> None:
        """Release underlying memmap handle (helpful for Windows file cleanup)."""
        if self._mm is not None:
            mmap_obj = getattr(self._mm, "_mmap", None)
            self._mm = None
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Re-open read-only memmap cleanly inside DataLoader worker processes
        state["_mm"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._mm = None

    def with_offset(
        self,
        start_offset: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> "MemmapTokenDataset":
        """Return a lightweight zero-copy view starting from `start_offset` (and optional DDP shard)."""
        return MemmapTokenDataset(
            bin_path=self.bin_path,
            chunk_size=self.chunk_size,
            prefix_len=self.prefix_len,
            dtype=self.dtype,
            stride=self.stride,
            start_offset=int(start_offset),
            rank=self.rank if rank is None else int(rank),
            world_size=self.world_size if world_size is None else int(world_size),
        )

    @property
    def tokens(self) -> np.memmap:
        return self._ensure_memmap()

    @property
    def num_tokens(self) -> int:
        return int(self._num_tokens)

    @property
    def total_windows(self) -> int:
        return int(self._total_windows)

    @property
    def pad_ratio(self) -> float:
        return 0.0

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of range for MemmapTokenDataset of length {self._len}")

        win_idx = (self._effective_start + idx * self.world_size + self.rank) % self._total_windows
        mm = self._ensure_memmap()
        if self._num_tokens >= self.window_size:
            tok_start = win_idx * self.stride
            arr = np.array(mm[tok_start : tok_start + self.window_size], dtype=np.int64, copy=True)
        else:
            indices = np.arange(self.window_size, dtype=np.int64) % self._num_tokens
            arr = np.array(mm[indices], dtype=np.int64, copy=True)

        window = torch.from_numpy(arr)
        if self.prefix_len == 0:
            return {
                "input_ids": window,
                "attention_mask": torch.ones(self.chunk_size, dtype=torch.long),
            }

        prefix_ids = window[: self.prefix_len].clone()
        target_ids = window[self.prefix_len :].clone()
        prefix_mask = torch.ones(self.prefix_len, dtype=torch.long)
        target_mask = torch.ones(self.chunk_size, dtype=torch.long)
        exact_labels = torch.full((self.window_size,), -100, dtype=torch.long)
        exact_labels[self.prefix_len :] = target_ids
        return {
            "prefix_ids": prefix_ids,
            "prefix_mask": prefix_mask,
            "target_ids": target_ids,
            "target_chunk_ids": target_ids,
            "target_mask": target_mask,
            "input_ids": window.clone(),
            "attention_mask": torch.ones(self.window_size, dtype=torch.long),
            "exact_input_ids": window.clone(),
            "exact_attention_mask": torch.ones(self.window_size, dtype=torch.long),
            "exact_labels": exact_labels,
            "has_tool_call": torch.tensor(0, dtype=torch.long),
            "plan_len": torch.tensor(0, dtype=torch.long),
            "is_fim": torch.tensor(0, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# BlockARSFTDataset (Causal Multi-Turn / Multi-Chunk SFT + Exact-AR Labels)
# ---------------------------------------------------------------------------

def _load_sft_source(source: Union[Sequence[dict[str, Any]], str, Path]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        items: list[dict[str, Any]] = []
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        else:
            with path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, list):
                    items = loaded
                elif isinstance(loaded, dict):
                    items = [loaded]
        return items
    return list(source)


class BlockARSFTDataset(Dataset):
    """Unpacks multi-turn chat/tool conversations into causal Block-AR chunk training pairs."""

    def __init__(
        self,
        samples: Union[Sequence[dict[str, Any]], str, Path],
        tokenizer: Any,
        chunk_size: int = 64,
        max_prefix_len: int = 256,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        fixed_prefix_len: bool = True,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if max_prefix_len < 1:
            raise ValueError(f"max_prefix_len must be >= 1, got {max_prefix_len}")

        self.chunk_size = int(chunk_size)
        self.max_prefix_len = int(max_prefix_len)
        self.fixed_prefix_len = bool(fixed_prefix_len)
        self.pad_token_id = (
            int(pad_token_id) if pad_token_id is not None else _get_token_id(tokenizer, "pad_token_id", 0)
        )
        self.bos_token_id = (
            int(bos_token_id) if bos_token_id is not None else _get_token_id(tokenizer, "bos_token_id", 1)
        )
        self.eos_token_id = (
            int(eos_token_id) if eos_token_id is not None else _get_token_id(tokenizer, "eos_token_id", 2)
        )
        self.tool_call_start_id = getattr(tokenizer, "tool_call_start_id", None)

        raw_samples = _load_sft_source(samples)
        self.examples: list[dict[str, Any]] = []

        for sample in raw_samples:
            messages = sample.get("messages")
            if not messages:
                prompt_text = str(sample.get("prompt", ""))
                response_text = str(sample.get("response", ""))
                if not response_text:
                    continue
                messages = [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": response_text},
                ]
            tools = sample.get("tools")

            for idx, msg in enumerate(messages):
                if str(msg.get("role", "")).strip().lower() != "assistant":
                    continue
                content = str(msg.get("content", ""))
                if not content:
                    continue

                prefix_text = _format_messages(
                    tokenizer,
                    messages[:idx],
                    tools=tools,
                    add_generation_prompt=True,
                )
                base_prefix_ids = _encode_text(tokenizer, prefix_text)
                if not base_prefix_ids:
                    base_prefix_ids = [self.bos_token_id]

                resp_ids = _encode_text(tokenizer, content)
                if not resp_ids or resp_ids[-1] != self.eos_token_id:
                    resp_ids = resp_ids + [self.eos_token_id]

                num_chunks = (len(resp_ids) + self.chunk_size - 1) // self.chunk_size
                for c_idx in range(num_chunks):
                    c_start = c_idx * self.chunk_size
                    c_end = c_start + self.chunk_size
                    chunk_tokens = resp_ids[c_start:c_end]
                    valid_target_len = len(chunk_tokens)

                    prefix_seq = base_prefix_ids + resp_ids[:c_start]
                    if len(prefix_seq) > self.max_prefix_len:
                        prefix_seq = prefix_seq[-self.max_prefix_len :]
                    if not prefix_seq:
                        prefix_seq = [self.bos_token_id]

                    if valid_target_len < self.chunk_size:
                        padded_target = chunk_tokens + [self.pad_token_id] * (
                            self.chunk_size - valid_target_len
                        )
                        target_mask = [1] * valid_target_len + [0] * (
                            self.chunk_size - valid_target_len
                        )
                    else:
                        padded_target = chunk_tokens
                        target_mask = [1] * self.chunk_size

                    has_tool = (
                        ("<tool_call>" in content)
                        or (
                            self.tool_call_start_id is not None
                            and int(self.tool_call_start_id) in chunk_tokens
                        )
                    )

                    self.examples.append(
                        {
                            "prefix_ids": prefix_seq,
                            "target_ids": padded_target,
                            "target_mask": target_mask,
                            "valid_target_len": valid_target_len,
                            "has_tool_call": int(has_tool),
                            "chunk_index": c_idx,
                            "plan_len": 0,
                            "is_fim": 0,
                        }
                    )

        if not self.examples:
            raise ValueError("BlockARSFTDataset produced 0 training pairs from the input samples.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        raw_prefix = ex["prefix_ids"]
        valid_p = len(raw_prefix)

        if self.fixed_prefix_len:
            pad_len = self.max_prefix_len - valid_p
            prefix_ids = torch.tensor(
                [self.pad_token_id] * pad_len + raw_prefix, dtype=torch.long
            )
            prefix_mask = torch.tensor([0] * pad_len + [1] * valid_p, dtype=torch.long)
        else:
            prefix_ids = torch.tensor(raw_prefix, dtype=torch.long)
            prefix_mask = torch.ones(valid_p, dtype=torch.long)

        target_ids = torch.tensor(ex["target_ids"], dtype=torch.long)
        target_mask = torch.tensor(ex["target_mask"], dtype=torch.long)

        exact_input_ids = torch.cat([prefix_ids, target_ids], dim=0)
        exact_attention_mask = torch.cat([prefix_mask, target_mask], dim=0)
        exact_labels = torch.full_like(exact_input_ids, -100)
        p_len = prefix_ids.numel()
        v_len = ex["valid_target_len"]
        plan_len = int(ex.get("plan_len", 0))
        if plan_len > 0:
            eff_plan = min(plan_len, valid_p)
            exact_labels[p_len - eff_plan : p_len] = prefix_ids[p_len - eff_plan : p_len]
        exact_labels[p_len : p_len + v_len] = target_ids[:v_len]

        return {
            "prefix_ids": prefix_ids,
            "prefix_mask": prefix_mask,
            "target_ids": target_ids,
            "target_chunk_ids": target_ids,
            "target_mask": target_mask,
            "exact_input_ids": exact_input_ids,
            "exact_attention_mask": exact_attention_mask,
            "exact_labels": exact_labels,
            "has_tool_call": torch.tensor(ex["has_tool_call"], dtype=torch.long),
            "chunk_index": torch.tensor(ex["chunk_index"], dtype=torch.long),
            "plan_len": torch.tensor(plan_len, dtype=torch.long),
            "is_fim": torch.tensor(int(ex.get("is_fim", 0)), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# FIMPlanDataset (Section 6 PG-FIM Stage 2 Joint Dual-Mode Dataset)
# ---------------------------------------------------------------------------

class FIMPlanDataset(Dataset):
    """Plan-Guided Fill-in-the-Middle (PG-FIM) dataset for CNCM Stage 2 Joint Training.

    Implements Section 6 of the PG-FIM specification:
      - `fim_rate` (default 0.5): 50% of samples use PSM format
        `<fim_prefix> P <fim_suffix> S <fim_middle> <mlen_k> <kw> kw1 | kw2 </kw> [middle] <fim_end>`,
        while the remaining 50% stay plain left-to-right Block-AR.
      - Span selection: 40% AST-node spans, 30% whole lines, 30% random character spans.
      - Offline keyword extraction: up to `kw_max=3` identifiers, control keywords, and literals.
      - Keyword robustness corruption: 15% replaced by `<kw/>`, 15% one keyword swapped for a
        random identifier, 70% clean oracle keywords.
      - Supervises BOTH the Exact-AR plan tokens (`<mlen_k> <kw> ... </kw>` or `<mlen_k> <kw/>`)
        AND the middle chunk / `<fim_end>` boundary tokens in `exact_labels`.
    """

    def __init__(
        self,
        samples: Optional[Union[Sequence[Union[str, dict[str, Any]]], str, Path]] = None,
        tokenizer: Any = None,
        chunk_size: int = 64,
        max_prefix_len: int = 256,
        fim_rate: float = 0.5,
        span_mode: str = "mix",
        kw_max: int = 3,
        kw_empty_prob: float = 0.15,
        kw_corrupt_prob: float = 0.15,
        fim_prefix_budget: int = 512,
        fim_suffix_budget: int = 256,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        seed: int = 42,
        fixed_prefix_len: bool = True,
    ) -> None:
        super().__init__()
        if tokenizer is None:
            raise ValueError("FIMPlanDataset requires a `tokenizer` instance.")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if max_prefix_len < 16:
            raise ValueError(f"max_prefix_len must be >= 16, got {max_prefix_len}")
        if not (0.0 <= fim_rate <= 1.0):
            raise ValueError(f"fim_rate must be in [0, 1], got {fim_rate}")

        self.chunk_size = int(chunk_size)
        self.max_prefix_len = int(max_prefix_len)
        self.fim_rate = float(fim_rate)
        self.span_mode = str(span_mode)
        self.kw_max = int(kw_max)
        self.kw_empty_prob = float(kw_empty_prob)
        self.kw_corrupt_prob = float(kw_corrupt_prob)
        self.fim_prefix_budget = int(fim_prefix_budget)
        self.fim_suffix_budget = int(fim_suffix_budget)
        self.fixed_prefix_len = bool(fixed_prefix_len)

        self.pad_token_id = (
            int(pad_token_id) if pad_token_id is not None else _get_token_id(tokenizer, "pad_token_id", 0)
        )
        self.bos_token_id = (
            int(bos_token_id) if bos_token_id is not None else _get_token_id(tokenizer, "bos_token_id", 1)
        )
        self.eos_token_id = (
            int(eos_token_id) if eos_token_id is not None else _get_token_id(tokenizer, "eos_token_id", 2)
        )
        self.fim_prefix_id = _get_token_id(tokenizer, "fim_prefix_id", tokenizer.vocab_size - 11)
        self.fim_suffix_id = _get_token_id(tokenizer, "fim_suffix_id", tokenizer.vocab_size - 10)
        self.fim_middle_id = _get_token_id(tokenizer, "fim_middle_id", tokenizer.vocab_size - 9)
        self.fim_end_id = _get_token_id(tokenizer, "fim_end_id", tokenizer.vocab_size - 8)
        self.kw_start_id = _get_token_id(tokenizer, "kw_start_id", tokenizer.vocab_size - 7)
        self.kw_end_id = _get_token_id(tokenizer, "kw_end_id", tokenizer.vocab_size - 6)
        self.kw_empty_id = _get_token_id(tokenizer, "kw_empty_id", tokenizer.vocab_size - 5)

        rng = random.Random(seed)

        if samples is None:
            raw_list: list[Union[str, dict[str, Any]]] = []
            code_docs = make_synthetic_code_corpus(num_samples=96, seed=seed)
            fim_tasks = make_synthetic_fim_tasks(num_samples=64, seed=seed, curriculum_stage="all")
            prose_docs = make_synthetic_corpus(num_samples=48, seed=seed + 7)
            raw_list.extend(fim_tasks)
            raw_list.extend(code_docs)
            raw_list.extend(prose_docs)
        elif isinstance(samples, (str, Path)):
            raw_list = list(_load_texts_source(samples))
        else:
            raw_list = list(samples)

        if not raw_list:
            raise ValueError("FIMPlanDataset received 0 input samples.")

        self.examples: list[dict[str, Any]] = []

        for item in raw_list:
            use_fim = rng.random() < self.fim_rate

            if isinstance(item, dict):
                if "prefix" in item and "middle" in item and "suffix" in item:
                    prefix_str = str(item["prefix"])
                    middle_str = str(item["middle"])
                    suffix_str = str(item["suffix"])
                    full_text = prefix_str + middle_str + suffix_str
                    explicit_kws = item.get("oracle_keywords")
                    is_code = str(item.get("language", "python")).lower() == "python"
                else:
                    full_text = str(item.get("text") or item.get("full_code") or item.get("content") or "")
                    if not full_text and "messages" in item:
                        full_text = _format_messages(tokenizer, item["messages"], add_generation_prompt=False)
                    prefix_str, middle_str, suffix_str = _extract_spans_interop(
                        full_text, mode=self.span_mode, rng=rng
                    )
                    explicit_kws = None
                    is_code = "def " in full_text or "return" in full_text
            else:
                full_text = str(item)
                is_code = "def " in full_text or "return" in full_text
                prefix_str, middle_str, suffix_str = _extract_spans_interop(
                    full_text, mode=self.span_mode, rng=rng
                )
                explicit_kws = None

            if not full_text.strip():
                continue

            if use_fim and middle_str.strip():
                # 1. Extract oracle keywords (0..kw_max)
                if explicit_kws is not None:
                    oracle_kws = [str(k) for k in explicit_kws[: self.kw_max] if str(k).strip()]
                else:
                    oracle_kws = _extract_keywords_interop(
                        middle_str,
                        prefix=prefix_str,
                        suffix=suffix_str,
                        max_kw=self.kw_max,
                        is_code=is_code,
                    )

                # 2. Apply keyword robustness corruption (15% <kw/>, 15% swap one keyword, 70% clean)
                u_kw = rng.random()
                plan_kind = "clean"
                if self.kw_max <= 0 or not oracle_kws or u_kw < self.kw_empty_prob:
                    active_kws: list[str] = []
                    plan_kind = "empty"
                elif u_kw < (self.kw_empty_prob + self.kw_corrupt_prob):
                    active_kws = list(oracle_kws[: self.kw_max])
                    swap_idx = rng.randrange(len(active_kws))
                    distractors = [d for d in _DISTRACTOR_IDENTIFIERS if d not in active_kws]
                    active_kws[swap_idx] = rng.choice(distractors) if distractors else "tmp_val"
                    plan_kind = "corrupted"
                else:
                    active_kws = list(oracle_kws[: self.kw_max])

                # 3. Encode middle + <fim_end> and determine <mlen_k>
                mid_ids = _encode_text(tokenizer, middle_str) + [self.fim_end_id]
                total_chunks = max(1, (len(mid_ids) + self.chunk_size - 1) // self.chunk_size)
                if hasattr(tokenizer, "mlen_id_for_chunks"):
                    mlen_id = int(tokenizer.mlen_id_for_chunks(total_chunks))
                else:
                    mlen_id = int(tokenizer.vocab_size - 4 + min(total_chunks, 4) - 1)

                if not active_kws:
                    plan_ids = [mlen_id, self.kw_empty_id]
                else:
                    kw_body = " | ".join(active_kws)
                    kw_body_ids = _encode_text(tokenizer, kw_body)
                    plan_ids = [mlen_id, self.kw_start_id] + kw_body_ids + [self.kw_end_id]

                # 4. Encode PSM prompt respecting edge context budgets and max_prefix_len
                avail_for_psm = max(8, self.max_prefix_len - len(plan_ids))
                eff_p_budget = min(self.fim_prefix_budget, max(4, (avail_for_psm * 2) // 3 - 2))
                eff_s_budget = min(self.fim_suffix_budget, max(2, avail_for_psm - eff_p_budget - 3))
                if hasattr(tokenizer, "format_fim_prompt"):
                    _, psm_ids = tokenizer.format_fim_prompt(
                        prefix_str,
                        suffix_str,
                        prefix_budget=eff_p_budget,
                        suffix_budget=eff_s_budget,
                    )
                else:
                    p_ids = _encode_text(tokenizer, prefix_str)[-eff_p_budget:]
                    s_ids = _encode_text(tokenizer, suffix_str)[:eff_s_budget]
                    psm_ids = (
                        [self.fim_prefix_id]
                        + p_ids
                        + [self.fim_suffix_id]
                        + s_ids
                        + [self.fim_middle_id]
                    )

                for c_idx in range(total_chunks):
                    c_start = c_idx * self.chunk_size
                    c_end = c_start + self.chunk_size
                    chunk_tokens = mid_ids[c_start:c_end]
                    valid_target_len = len(chunk_tokens)

                    prefix_seq = psm_ids + plan_ids + mid_ids[:c_start]
                    if len(prefix_seq) > self.max_prefix_len:
                        prefix_seq = prefix_seq[-self.max_prefix_len :]

                    if valid_target_len < self.chunk_size:
                        padded_target = chunk_tokens + [self.pad_token_id] * (
                            self.chunk_size - valid_target_len
                        )
                        target_mask = [1] * valid_target_len + [0] * (
                            self.chunk_size - valid_target_len
                        )
                    else:
                        padded_target = chunk_tokens
                        target_mask = [1] * self.chunk_size

                    # Supervise plan tokens on chunk 0 where plan_ids sit at the tail of prefix_seq
                    sup_plan_len = len(plan_ids) if c_idx == 0 else 0

                    self.examples.append(
                        {
                            "prefix_ids": prefix_seq,
                            "target_ids": padded_target,
                            "target_mask": target_mask,
                            "valid_target_len": valid_target_len,
                            "plan_len": sup_plan_len,
                            "plan_ids": list(plan_ids),
                            "keywords": list(active_kws),
                            "plan_kind": plan_kind,
                            "is_fim": 1,
                            "has_tool_call": 0,
                            "chunk_index": c_idx,
                        }
                    )
            else:
                # Plain left-to-right Block-AR sample (50% non-FIM regime)
                doc_ids = _encode_text(tokenizer, full_text) + [self.eos_token_id]
                if len(doc_ids) <= 4:
                    doc_ids = [self.bos_token_id] + doc_ids
                split_pt = max(1, min(len(doc_ids) - 1, len(doc_ids) // 2))
                base_prefix = doc_ids[:split_pt][-self.max_prefix_len :]
                target_stream = doc_ids[split_pt:]
                num_chunks = max(1, (len(target_stream) + self.chunk_size - 1) // self.chunk_size)

                for c_idx in range(num_chunks):
                    c_start = c_idx * self.chunk_size
                    c_end = c_start + self.chunk_size
                    chunk_tokens = target_stream[c_start:c_end]
                    valid_target_len = len(chunk_tokens)
                    prefix_seq = (base_prefix + target_stream[:c_start])[-self.max_prefix_len :]
                    if not prefix_seq:
                        prefix_seq = [self.bos_token_id]

                    if valid_target_len < self.chunk_size:
                        padded_target = chunk_tokens + [self.pad_token_id] * (
                            self.chunk_size - valid_target_len
                        )
                        target_mask = [1] * valid_target_len + [0] * (
                            self.chunk_size - valid_target_len
                        )
                    else:
                        padded_target = chunk_tokens
                        target_mask = [1] * self.chunk_size

                    self.examples.append(
                        {
                            "prefix_ids": prefix_seq,
                            "target_ids": padded_target,
                            "target_mask": target_mask,
                            "valid_target_len": valid_target_len,
                            "plan_len": 0,
                            "plan_ids": [],
                            "keywords": [],
                            "plan_kind": "block_ar",
                            "is_fim": 0,
                            "has_tool_call": 0,
                            "chunk_index": c_idx,
                        }
                    )

        if not self.examples:
            raise ValueError("FIMPlanDataset produced 0 training examples.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        raw_prefix = ex["prefix_ids"]
        valid_p = len(raw_prefix)

        if self.fixed_prefix_len:
            pad_len = self.max_prefix_len - valid_p
            prefix_ids = torch.tensor(
                [self.pad_token_id] * pad_len + raw_prefix, dtype=torch.long
            )
            prefix_mask = torch.tensor([0] * pad_len + [1] * valid_p, dtype=torch.long)
        else:
            prefix_ids = torch.tensor(raw_prefix, dtype=torch.long)
            prefix_mask = torch.ones(valid_p, dtype=torch.long)

        target_ids = torch.tensor(ex["target_ids"], dtype=torch.long)
        target_mask = torch.tensor(ex["target_mask"], dtype=torch.long)

        exact_input_ids = torch.cat([prefix_ids, target_ids], dim=0)
        exact_attention_mask = torch.cat([prefix_mask, target_mask], dim=0)
        exact_labels = torch.full_like(exact_input_ids, -100)

        p_len = prefix_ids.numel()
        v_len = ex["valid_target_len"]
        plan_len = int(ex.get("plan_len", 0))
        if plan_len > 0:
            eff_plan = min(plan_len, valid_p)
            exact_labels[p_len - eff_plan : p_len] = prefix_ids[p_len - eff_plan : p_len]
        exact_labels[p_len : p_len + v_len] = target_ids[:v_len]

        return {
            "prefix_ids": prefix_ids,
            "prefix_mask": prefix_mask,
            "target_ids": target_ids,
            "target_chunk_ids": target_ids,
            "target_mask": target_mask,
            "exact_input_ids": exact_input_ids,
            "exact_attention_mask": exact_attention_mask,
            "exact_labels": exact_labels,
            "plan_len": torch.tensor(plan_len, dtype=torch.long),
            "is_fim": torch.tensor(int(ex["is_fim"]), dtype=torch.long),
            "has_tool_call": torch.tensor(int(ex["has_tool_call"]), dtype=torch.long),
            "chunk_index": torch.tensor(int(ex["chunk_index"]), dtype=torch.long),
        }


def collate_block_ar(
    batch: Sequence[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate Block-AR and FIMPlanDataset samples with tight left-padding to batch max prefix.

    Preserves exact adjacency between the last valid prefix token (e.g. `</kw>` or `<kw/>`)
    and `target_ids[0]`, and preserves `exact_labels` supervision on both PG-FIM plan tokens
    and valid target chunk tokens.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    valid_prefixes: list[torch.Tensor] = []
    for item in batch:
        p_ids = item["prefix_ids"]
        p_mask = item["prefix_mask"]
        valid_len = int(p_mask.sum().item())
        if valid_len <= 0:
            valid_prefixes.append(p_ids[-1:])
        else:
            valid_prefixes.append(p_ids[p_mask.bool()])

    max_p = max(1, max(int(p.numel()) for p in valid_prefixes))
    bsz = len(batch)
    chunk_size = int(batch[0]["target_ids"].numel())

    prefix_ids = torch.full((bsz, max_p), int(pad_token_id), dtype=torch.long)
    prefix_mask = torch.zeros((bsz, max_p), dtype=torch.long)
    for i, vp in enumerate(valid_prefixes):
        n = int(vp.numel())
        prefix_ids[i, max_p - n :] = vp
        prefix_mask[i, max_p - n :] = 1

    target_ids = torch.stack([item["target_ids"] for item in batch], dim=0)
    target_mask = torch.stack(
        [
            item.get("target_mask", torch.ones(chunk_size, dtype=torch.long))
            for item in batch
        ],
        dim=0,
    )

    exact_input_ids = torch.cat([prefix_ids, target_ids], dim=1)
    exact_attention_mask = torch.cat([prefix_mask, target_mask], dim=1)
    exact_labels = torch.full_like(exact_input_ids, -100)
    for i, item in enumerate(batch):
        vp = valid_prefixes[i]
        n = int(vp.numel())
        plan_len = int(item["plan_len"].item()) if "plan_len" in item else 0
        if plan_len > 0:
            eff_plan = min(plan_len, n)
            exact_labels[i, max_p - eff_plan : max_p] = vp[-eff_plan:]
        t_mask = target_mask[i].bool()
        exact_labels[i, max_p:][t_mask] = target_ids[i][t_mask]

    has_tool_call = torch.stack(
        [item.get("has_tool_call", torch.tensor(0, dtype=torch.long)) for item in batch],
        dim=0,
    )
    plan_lens = torch.stack(
        [item.get("plan_len", torch.tensor(0, dtype=torch.long)) for item in batch],
        dim=0,
    )
    is_fim = torch.stack(
        [item.get("is_fim", torch.tensor(0, dtype=torch.long)) for item in batch],
        dim=0,
    )

    return {
        "prefix_ids": prefix_ids,
        "prefix_mask": prefix_mask,
        "target_ids": target_ids,
        "target_chunk_ids": target_ids,
        "target_mask": target_mask,
        "exact_input_ids": exact_input_ids,
        "exact_attention_mask": exact_attention_mask,
        "exact_labels": exact_labels,
        "has_tool_call": has_tool_call,
        "plan_len": plan_lens,
        "is_fim": is_fim,
    }


# ---------------------------------------------------------------------------
# GRPOPromptDataset
# ---------------------------------------------------------------------------

class GRPOPromptDataset(Dataset):
    """Dataset of prompts paired with verifiable reward specs for CNCM code-space GRPO."""

    def __init__(
        self,
        prompts: Sequence[Union[str, dict[str, Any]]],
        tokenizer: Any,
        max_prompt_len: int = 192,
        bos_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not prompts:
            raise ValueError("GRPOPromptDataset received an empty prompt list.")
        self.max_prompt_len = int(max_prompt_len)
        self.bos_token_id = (
            int(bos_token_id) if bos_token_id is not None else _get_token_id(tokenizer, "bos_token_id", 1)
        )
        self.items: list[dict[str, Any]] = []

        for raw in prompts:
            if isinstance(raw, str):
                prompt_text = raw
                spec: dict[str, Any] = {
                    "task_type": "format_reasoning",
                    "expected_answer": "",
                    "expected_tool": None,
                    "expected_args": None,
                    "require_think": True,
                    "require_tool_call": False,
                }
                ids = _encode_text(tokenizer, prompt_text)
            elif "prefix" in raw and "suffix" in raw:
                # PG-FIM task dict
                prefix_str = str(raw.get("prefix", ""))
                suffix_str = str(raw.get("suffix", ""))
                if hasattr(tokenizer, "format_fim_prompt"):
                    prompt_text, ids = tokenizer.format_fim_prompt(
                        prefix_str,
                        suffix_str,
                        prefix_budget=max(16, (self.max_prompt_len * 2) // 3),
                        suffix_budget=max(8, self.max_prompt_len // 3),
                    )
                else:
                    prompt_text = f"<fim_prefix>{prefix_str}<fim_suffix>{suffix_str}<fim_middle>"
                    ids = _encode_text(tokenizer, prompt_text)
                spec = {
                    "task_type": "fim",
                    "curriculum_stage": str(raw.get("curriculum_stage", "single_line")),
                    "prefix": prefix_str,
                    "middle": str(raw.get("middle", "")),
                    "suffix": suffix_str,
                    "test_code": str(raw.get("test_code", "")),
                    "oracle_keywords": list(raw.get("oracle_keywords", [])),
                    "expected_chunks": int(raw.get("expected_chunks", 1)),
                    "expected_answer": str(raw.get("middle", "")),
                    "expected_tool": None,
                    "expected_args": None,
                    "require_think": False,
                    "require_tool_call": False,
                }
            else:
                if "messages" in raw and raw["messages"]:
                    prompt_text = _format_messages(
                        tokenizer,
                        raw["messages"],
                        tools=raw.get("tools"),
                        add_generation_prompt=True,
                    )
                else:
                    prompt_text = str(raw.get("prompt", ""))
                spec = {
                    "task_type": str(raw.get("task_type", "format_reasoning")),
                    "expected_answer": str(raw.get("expected_answer", "")),
                    "expected_tool": raw.get("expected_tool"),
                    "expected_args": raw.get("expected_args"),
                    "require_think": bool(raw.get("require_think", True)),
                    "require_tool_call": bool(raw.get("require_tool_call", False)),
                }
                ids = _encode_text(tokenizer, prompt_text)

            if not ids:
                ids = [self.bos_token_id]
            if len(ids) > self.max_prompt_len:
                ids = ids[-self.max_prompt_len :]

            self.items.append(
                {
                    "prompt": prompt_text,
                    "prompt_ids": torch.tensor(ids, dtype=torch.long),
                    "prompt_mask": torch.ones(len(ids), dtype=torch.long),
                    **spec,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        return {
            **item,
            "prompt_ids": item["prompt_ids"].clone(),
            "prompt_mask": item["prompt_mask"].clone(),
        }


def collate_grpo_prompts(
    batch: Sequence[dict[str, Any]],
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Collate variable-length GRPO prompts with left-padding and preserved reward specs."""
    if not batch:
        raise ValueError("Cannot collate an empty GRPO batch.")
    max_p = max(1, max(int(item["prompt_ids"].numel()) for item in batch))
    bsz = len(batch)
    prompt_ids = torch.full((bsz, max_p), int(pad_token_id), dtype=torch.long)
    prompt_mask = torch.zeros((bsz, max_p), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["prompt_ids"]
        n = int(ids.numel())
        prompt_ids[i, max_p - n :] = ids
        prompt_mask[i, max_p - n :] = 1
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "prompts": [item["prompt"] for item in batch],
        "specs": [
            {
                k: v
                for k, v in item.items()
                if k not in ("prompt", "prompt_ids", "prompt_mask")
            }
            for item in batch
        ],
    }
