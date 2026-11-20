"""Code-Space & Compiler-in-the-Loop PG-FIM GRPO / CISPO Reinforcement Learning trainer for CNCM.

Supports two RL modes on the unified CNCM backbone:
1. **Standard Agentic / Reasoning Code-Space GRPO (`train_grpo`)**:
   - Samples `G` parallel concept-chunk trajectories per prompt via
     `model.generate_concept_chunk(..., return_codes=True)`.
   - Computes exact 1-pass code-space trajectory log-probabilities via
     `model.compute_code_logprobs`.
   - Scores completions with `compute_verifiable_reward` (`<think>`, valid JSON `<tool_call>`,
     task answer match).
2. **Compiler-in-the-Loop PG-FIM GRPO (`train_pg_fim_grpo` / `train_grpo(..., fim_mode=True)`)**:
   - Schedules tasks across a 3-stage curriculum: `"single_line"` -> `"multi_line"` -> `"function_body"`.
   - Samples `G` complete `(Exact-AR Plan + Concept-Mode Middle)` rollouts per FIM prompt using
     `KeywordPlanMasker` for `<mlen_k> (<kw/> | <kw> kw1 | kw2 </kw>)` and `generate_concept_chunk`
     (with optional Phase B `kw_token_ids` copy conditioning).
   - Evaluates assembled `P + M + S` inside `run_code_sandbox` with anti-reward-hacking gated rewards
     (`compute_fim_reward`: non-triviality gate -> parse +0.2 -> compile +0.3 -> unit tests +0.5,
     keyword faithfulness +0.05, wrong length bucket -0.05).
   - Computes joint Exact-AR plan + 1-pass Concept-Mode code log-probabilities via
     `model.compute_pg_fim_logprobs` and updates the generator with clipped GRPO/CISPO + Schulman k3 KL.

CLI usage:
    python -m cncm.trainer.train_grpo --preset tiny_test --steps 4 --num-generations 4
    python -m cncm.trainer.train_grpo --preset tiny_test --fim --steps 4 --num-generations 4
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from cncm.config import CNCMConfig, get_preset
from cncm.model import CNCMModel
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer
from cncm.trainer.dataset import (
    GRPOPromptDataset,
    make_synthetic_fim_tasks,
    make_synthetic_grpo_prompts,
)
from cncm.trainer.train_utils import (
    WarmupCosineScheduler,
    amp_context,
    clip_and_check_grad_norm,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
)


_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", flags=re.DOTALL)
_SPECIAL_TAG_STRIP_RE = re.compile(
    r"<fim_prefix>|<fim_suffix>|<fim_middle>|<fim_end>|<kw>|</kw>|<kw/>|"
    r"<mlen_1>|<mlen_2>|<mlen_3>|<mlen_4\+>|<\|pad\|>|<\|bos\|>|<\|eos\|>|<\|unk\|>"
)


def compute_verifiable_reward(text: str, spec: dict[str, Any]) -> float:
    """Score a decoded CNCM completion against verifiable structure, tool, and answer specs."""
    reward = 0.0
    stripped = text.strip()

    # 1. Reasoning tag structure (<think>...</think>)
    think_match = _THINK_RE.search(text)
    if think_match is not None and think_match.group(1).strip():
        reward += 0.25
    elif spec.get("require_think", True) and "<think>" in text:
        reward += 0.10

    # 2. Valid JSON <tool_call>...</tool_call> structure & accuracy
    tool_match = _TOOL_CALL_RE.search(text)
    expected_tool = spec.get("expected_tool")
    expected_args = spec.get("expected_args")
    if tool_match is not None:
        raw_json = tool_match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                reward += 0.25
                if expected_tool is not None and parsed.get("name") == expected_tool:
                    reward += 0.25
                if isinstance(expected_args, dict) and expected_args and isinstance(parsed.get("arguments"), dict):
                    args_dict = parsed["arguments"]
                    hits = sum(
                        1
                        for k, v in expected_args.items()
                        if str(args_dict.get(k, "")).strip() == str(v).strip()
                    )
                    reward += 0.25 * (float(hits) / float(len(expected_args)))
            else:
                reward -= 0.05
        except Exception:
            reward -= 0.10
    elif "<tool_call>" in text:
        reward -= 0.05

    # 3. Verifiable task answer check
    expected_answer = str(spec.get("expected_answer", "")).strip()
    if expected_answer:
        visible = _THINK_RE.sub(" ", text)
        exact_pat = rf"(?<!\w){re.escape(expected_answer)}(?!\w)"
        if re.search(exact_pat, visible):
            reward += 0.50
        elif expected_answer in text:
            reward += 0.25

    # 4. Small bounded non-degeneracy tie-breaker (max +0.08)
    if stripped:
        tokens = stripped.split()
        unique_ratio = float(len(set(tokens))) / float(max(1, len(tokens)))
        printable_ratio = sum(1 for ch in stripped if ch.isprintable()) / float(len(stripped))
        reward += 0.04 * unique_ratio * printable_ratio
        if expected_answer:
            exp_chars = set(expected_answer)
            overlap = len(exp_chars.intersection(set(stripped))) / float(max(1, len(exp_chars)))
            reward += 0.04 * overlap
        else:
            reward += 0.04 * min(1.0, len(stripped) / 64.0)

    return float(reward)


def compute_fim_reward(
    middle_text: str,
    spec: dict[str, Any],
    planned_keywords: Optional[Sequence[str]] = None,
    planned_chunks: int = 1,
    actual_chunks: int = 1,
    min_tokens: int = 2,
) -> dict[str, Any]:
    """Compute gated compiler-in-the-loop PG-FIM reward (PG-FIM Plan Section 6).

    Gated reward table:
      - `P + M + S` parses: `+0.20` (GATED: requires `non_trivial=True` — not `pass`/empty, >= `min_tokens`)
      - Compiles / type-checks: `+0.30` (GATED: requires `parses=True`)
      - Unit tests pass in sandbox: `+0.50` (GATED: requires `compiles=True`)
      - Keyword faithfulness (all planned keywords appear in `M`): `+0.05`
      - Wrong length bucket (`planned_chunks != actual_chunks`): `-0.05`

    Returns:
      `{"reward": float, "parses": bool, "compiles": bool, "tests_passed": bool,
        "non_trivial": bool, "kw_faithful": bool, "length_match": bool}`
    """
    from cncm.code.sandbox import run_code_sandbox

    clean_mid = _SPECIAL_TAG_STRIP_RE.sub("", middle_text)
    prefix = str(spec.get("prefix", ""))
    suffix = str(spec.get("suffix", ""))
    test_code = str(spec.get("test_code", ""))
    timeout_sec = float(spec.get("timeout_sec", 2.0))

    full_code = f"{prefix}{clean_mid}{suffix}"
    sb = run_code_sandbox(
        full_code,
        test_code=test_code,
        timeout_sec=timeout_sec,
        min_tokens=min_tokens,
        middle=clean_mid,
    )

    non_trivial = bool(sb.get("non_trivial", False))
    raw_parses = bool(sb.get("parses", False))
    raw_compiles = bool(sb.get("compiles", False))
    raw_tests = bool(sb.get("tests_passed", False))

    # Strict hierarchical gating prevents trivial `pass` / empty middles from hacking parse/compile rewards
    gated_parses = non_trivial and raw_parses
    gated_compiles = gated_parses and raw_compiles
    gated_tests = gated_compiles and raw_tests

    reward = 0.0
    if gated_parses:
        reward += 0.20
    if gated_compiles:
        reward += 0.30
    if gated_tests:
        reward += 0.50

    # Keyword faithfulness check
    kws_to_check = (
        [str(k).strip() for k in planned_keywords if str(k).strip()]
        if planned_keywords is not None
        else [str(k).strip() for k in spec.get("oracle_keywords", []) if str(k).strip()]
    )
    if kws_to_check:
        kw_faithful = all(k in clean_mid for k in kws_to_check)
        if kw_faithful:
            reward += 0.05
    else:
        kw_faithful = False

    length_match = int(planned_chunks) == int(actual_chunks)
    if not length_match:
        reward -= 0.05

    return {
        "reward": float(reward),
        "parses": gated_parses,
        "compiles": gated_compiles,
        "tests_passed": gated_tests,
        "non_trivial": non_trivial,
        "kw_faithful": kw_faithful,
        "length_match": length_match,
        "error": sb.get("error"),
    }


def _enforce_frozen_concept_tokenizer(model: CNCMModel) -> None:
    if hasattr(model, "freeze_concept_tokenizer") and callable(model.freeze_concept_tokenizer):
        model.freeze_concept_tokenizer()
    if hasattr(model, "concept_tokenizer") and model.concept_tokenizer is not None:
        for p in model.concept_tokenizer.parameters():
            p.requires_grad_(False)
        model.concept_tokenizer.eval()


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def _compute_pg_fim_logprobs_compat(
    model: CNCMModel,
    prefix_ids: torch.Tensor,
    plan_ids: torch.Tensor,
    sampled_codes: Sequence[torch.Tensor],
    prefix_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute joint Exact-AR plan log-probs + 1-pass Concept code log-probs."""
    if hasattr(model, "compute_pg_fim_logprobs") and callable(model.compute_pg_fim_logprobs):
        out = model.compute_pg_fim_logprobs(
            prefix_ids=prefix_ids,
            plan_ids=plan_ids,
            sampled_codes=sampled_codes,
            prefix_mask=prefix_mask,
        )
        if isinstance(out, dict) and "total_logprobs" in out:
            return out
        if isinstance(out, torch.Tensor):
            return {"total_logprobs": out, "plan_logprobs": out, "code_logprobs": out}

    # Self-contained fallback: 1-pass Exact-AR on [prefix_ids, plan_ids] + 1-pass Concept Mode
    bsz, p_len = prefix_ids.shape
    plan_len = int(plan_ids.shape[1])
    full_prefix = torch.cat([prefix_ids, plan_ids], dim=1) if plan_len > 0 else prefix_ids
    if prefix_mask is not None:
        plan_valid = torch.ones((bsz, plan_len), device=prefix_ids.device, dtype=prefix_mask.dtype)
        full_mask: Optional[torch.Tensor] = (
            torch.cat([prefix_mask, plan_valid], dim=1) if plan_len > 0 else prefix_mask
        )
    else:
        full_mask = None

    if plan_len > 0:
        exact_out = model.forward_exact(full_prefix, attention_mask=full_mask, use_cache=False)
        logits = exact_out["logits"]
        # Positions [p_len - 1 .. p_len + plan_len - 2] predict plan_ids [0 .. plan_len - 1]
        plan_logits = logits[:, p_len - 1 : p_len + plan_len - 1, :].float()
        log_probs_vocab = F.log_softmax(plan_logits, dim=-1)
        token_plan_lp = log_probs_vocab.gather(-1, plan_ids.unsqueeze(-1)).squeeze(-1)
        plan_lp = token_plan_lp.mean(dim=-1)
    else:
        plan_lp = torch.zeros(bsz, device=prefix_ids.device, dtype=torch.float32)

    code_lp = model.compute_code_logprobs(
        full_prefix,
        sampled_codes,
        prefix_mask=full_mask,
    ).float()
    total_lp = 0.5 * (plan_lp + code_lp) if plan_len > 0 else code_lp
    return {
        "total_logprobs": total_lp,
        "plan_logprobs": plan_lp,
        "code_logprobs": code_lp,
    }


def _trim_plan_sequence(
    plan_seq: list[int],
    kw_end_id: int,
    kw_empty_id: int,
    pad_token_id: int,
) -> list[int]:
    """Trim generated plan tokens right after the first `</kw>` or `<kw/>` stop token."""
    trimmed: list[int] = []
    for tid in plan_seq:
        if tid == pad_token_id and trimmed:
            continue
        trimmed.append(int(tid))
        if tid in (kw_end_id, kw_empty_id):
            break
    return trimmed


def _cut_middle_at_fim_end(
    token_ids: list[int],
    fim_end_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> list[int]:
    """Slice middle chunk tokens up to `<fim_end>` or `<|eos|>`."""
    out: list[int] = []
    for tid in token_ids:
        if tid in (fim_end_id, eos_token_id):
            break
        if tid == pad_token_id:
            continue
        out.append(int(tid))
    return out


def _build_curriculum_buckets(
    tasks: Sequence[dict[str, Any]],
    curriculum: Sequence[str],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Group FIM tasks into curriculum stage buckets (`single_line` -> `multi_line` -> `function_body`)."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    stage_order: list[str] = []
    for stage in curriculum:
        st = str(stage).strip().lower()
        if st and st not in buckets:
            buckets[st] = []
            stage_order.append(st)

    fallback_list = list(tasks)
    for t in tasks:
        st = str(t.get("curriculum_stage", "")).strip().lower()
        if st in buckets:
            buckets[st].append(t)

    active_stages = [st for st in stage_order if buckets.get(st)]
    if not active_stages:
        active_stages = ["all"]
        buckets["all"] = fallback_list
    return active_stages, buckets


def train_pg_fim_grpo(
    cfg: CNCMConfig,
    tokenizer: CNCMTextTokenizer,
    model: CNCMModel,
    tasks: Optional[Sequence[dict[str, Any]]] = None,
    steps: int = 10,
    num_generations: int = 4,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: Union[str, torch.device] = "auto",
    *,
    curriculum: Sequence[str] = ("single_line", "multi_line", "function_body"),
    max_plan_tokens: int = 12,
    weight_decay: float = 0.01,
    beta_kl: float = 0.04,
    clip_eps: float = 0.2,
    use_cispo: bool = False,
    cispo_eps: Optional[float] = None,
    ppo_epochs: int = 1,
    temperature: float = 0.9,
    coarse_temp: float = 0.75,
    fine_temp: float = 0.15,
    skip_zero_variance_groups: bool = True,
    early_exit_conf: float = 1.0,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.1,
    amp_dtype: Union[str, torch.dtype, None] = None,
    seed: int = 42,
    log_every: int = 1,
    save_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Compiler-in-the-Loop PG-FIM GRPO / CISPO training on `CNCMModel` (Section 6 Stage 3).

    Samples `G = num_generations` plan + middle rollouts per prompt, evaluates `P + M + S`
    in `run_code_sandbox` with anti-reward-hacking gated rewards, computes joint Exact-AR
    plan + 1-pass Concept-Mode code log-probabilities via `compute_pg_fim_logprobs`, and
    updates the Stage 2 generator while keeping Stage 1 strictly frozen.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if num_generations < 2:
        raise ValueError(f"num_generations must be >= 2 for group normalization, got {num_generations}")

    from cncm.agent.grammar import KeywordPlanMasker, parse_plan_tokens

    set_seed(seed)
    dev = select_device(device)

    if tasks is None:
        raw_tasks = make_synthetic_fim_tasks(num_samples=max(24, steps * batch_size * 2), seed=seed)
    else:
        raw_tasks = list(tasks)
    if not raw_tasks:
        raise ValueError("train_pg_fim_grpo received 0 FIM tasks.")
    active_stages, stage_buckets = _build_curriculum_buckets(raw_tasks, curriculum=curriculum)
    stage_cursors: dict[str, int] = {st: 0 for st in active_stages}

    model = model.to(dev)
    _enforce_frozen_concept_tokenizer(model)

    ref_model = copy.deepcopy(model).to(dev)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    stage1_param_ids = {id(p) for p in model.concept_tokenizer.parameters()}
    trainable_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in stage1_param_ids
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in CNCMModel for PG-FIM GRPO.")

    initial_param_snapshot = [p.detach().clone() for p in trainable_params]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    eff_bs = max(1, min(int(batch_size), len(raw_tasks)))
    loss_history: list[float] = []
    pg_loss_history: list[float] = []
    kl_history: list[float] = []
    mean_reward_history: list[float] = []
    mean_logprob_history: list[float] = []
    parse_rate_history: list[float] = []
    compile_rate_history: list[float] = []
    test_pass_rate_history: list[float] = []
    non_trivial_rate_history: list[float] = []
    kw_faithful_rate_history: list[float] = []
    grad_norm_history: list[float] = []
    zero_var_groups = 0
    total_groups = 0
    stages_visited: list[str] = []
    all_logprobs_finite = True
    all_grads_finite = True

    cursor = 0
    for step_idx in range(1, steps + 1):
        rollout_groups: list[dict[str, Any]] = []
        step_rewards: list[float] = []
        step_logprobs: list[float] = []
        step_parses: list[float] = []
        step_compiles: list[float] = []
        step_tests: list[float] = []
        step_nontrivial: list[float] = []
        step_kw_faithful: list[float] = []

        stage_idx = min(
            len(active_stages) - 1,
            ((step_idx - 1) * len(active_stages)) // max(1, steps),
        )
        cur_stage_key = active_stages[stage_idx]
        cur_bucket = stage_buckets[cur_stage_key]

        model.eval()
        with torch.no_grad():
            for _ in range(eff_bs):
                b_cursor = stage_cursors[cur_stage_key]
                task = cur_bucket[b_cursor % len(cur_bucket)]
                stage_cursors[cur_stage_key] = b_cursor + 1
                stage_name = str(task.get("curriculum_stage", cur_stage_key))
                if stage_name not in stages_visited:
                    stages_visited.append(stage_name)

                prefix_str = str(task.get("prefix", ""))
                suffix_str = str(task.get("suffix", ""))
                _, psm_ids = tokenizer.format_fim_prompt(
                    prefix_str,
                    suffix_str,
                    prefix_budget=min(cfg.fim_prefix_budget, max(16, cfg.max_position_embeddings // 3)),
                    suffix_budget=min(cfg.fim_suffix_budget, max(8, cfg.max_position_embeddings // 6)),
                )
                prefix_1 = torch.tensor([psm_ids], dtype=torch.long, device=dev)
                prefix_mask_1 = torch.ones_like(prefix_1, dtype=torch.long)

                masker = KeywordPlanMasker(
                    tokenizer=tokenizer,
                    prefix_code=prefix_str,
                    suffix_code=suffix_str,
                    language=str(task.get("language", "python")),
                    kw_max=cfg.kw_max,
                    max_plan_tokens=max_plan_tokens,
                    prefix_len=prefix_1.shape[1],
                )

                group_rollouts: list[dict[str, Any]] = []
                raw_rewards: list[float] = []
                tie_breakers: list[float] = []

                for g_idx in range(num_generations):
                    # Vary sampling temperature slightly across G rollouts to encourage exploration
                    temp_g = max(0.2, float(temperature) * (0.85 + 0.10 * float(g_idx)))
                    plan_out = model.generate_exact(
                        prefix_1,
                        max_new_tokens=max_plan_tokens,
                        temperature=temp_g,
                        stop_token_ids=[cfg.kw_end_id, cfg.kw_empty_id],
                        logit_mask_fn=masker,
                    )
                    raw_plan_list = plan_out["token_ids"][0].tolist()
                    clean_plan_list = _trim_plan_sequence(
                        raw_plan_list,
                        kw_end_id=cfg.kw_end_id,
                        kw_empty_id=cfg.kw_empty_id,
                        pad_token_id=cfg.pad_token_id,
                    )
                    if not clean_plan_list:
                        clean_plan_list = [cfg.mlen_1_id, cfg.kw_empty_id]

                    parsed_plan = parse_plan_tokens(tokenizer, clean_plan_list)
                    planned_kws = parsed_plan["keywords"]
                    planned_chunks = int(parsed_plan["num_chunks"])

                    plan_t = torch.tensor([clean_plan_list], dtype=torch.long, device=dev)
                    full_cond_1 = torch.cat([prefix_1, plan_t], dim=1)
                    full_mask_1 = torch.ones_like(full_cond_1, dtype=torch.long)

                    # Extract keyword token IDs for Phase B Copy-Aware Decoder if active
                    kw_tok_ids: Optional[torch.Tensor] = None
                    if cfg.use_copy_decoder and planned_kws:
                        kw_flat = tokenizer.encode(" ".join(planned_kws), add_special_tokens=False)
                        if kw_flat:
                            kw_tok_ids = torch.tensor([kw_flat[:16]], dtype=torch.long, device=dev)

                    # Structured Concept Exploration (#7B): high-T on coarse levels L0..L3
                    # for algorithmic diversity, near-greedy on fine levels L4..L7 for syntax.
                    n_lvls = cfg.num_levels
                    coarse_cut = max(1, n_lvls // 2)
                    lvl_temps = [
                        (float(coarse_temp) if k < coarse_cut else float(fine_temp))
                        if temp_g > 0.0
                        else 0.0
                        for k in range(n_lvls)
                    ]
                    gen_kwargs: dict[str, Any] = {
                        "temperature": temp_g,
                        "level_temperatures": lvl_temps,
                        "early_exit_conf": early_exit_conf,
                        "return_codes": True,
                        "prefix_mask": full_mask_1,
                    }
                    if kw_tok_ids is not None:
                        gen_kwargs["kw_token_ids"] = kw_tok_ids

                    try:
                        chunk_out = model.generate_concept_chunk(full_cond_1, **gen_kwargs)
                    except TypeError:
                        gen_kwargs.pop("kw_token_ids", None)
                        gen_kwargs.pop("level_temperatures", None)
                        chunk_out = model.generate_concept_chunk(full_cond_1, **gen_kwargs)

                    mid_tok_list = _cut_middle_at_fim_end(
                        chunk_out["token_ids"][0].tolist(),
                        fim_end_id=cfg.fim_end_id,
                        eos_token_id=cfg.eos_token_id,
                        pad_token_id=cfg.pad_token_id,
                    )
                    middle_text = _decode_tokens(tokenizer, mid_tok_list)
                    sampled_codes = [c.detach() for c in chunk_out["codes"]]

                    fim_eval = compute_fim_reward(
                        middle_text,
                        task,
                        planned_keywords=planned_kws if planned_kws else task.get("oracle_keywords"),
                        planned_chunks=planned_chunks,
                        actual_chunks=1,
                    )

                    lp_out = _compute_pg_fim_logprobs_compat(
                        model,
                        prefix_ids=prefix_1,
                        plan_ids=plan_t,
                        sampled_codes=sampled_codes,
                        prefix_mask=prefix_mask_1,
                    )
                    ref_out = _compute_pg_fim_logprobs_compat(
                        ref_model,
                        prefix_ids=prefix_1,
                        plan_ids=plan_t,
                        sampled_codes=sampled_codes,
                        prefix_mask=prefix_mask_1,
                    )
                    old_lp = lp_out["total_logprobs"].detach().float()
                    ref_lp = ref_out["total_logprobs"].detach().float()

                    if not (torch.isfinite(old_lp).all() and torch.isfinite(ref_lp).all()):
                        all_logprobs_finite = False

                    # Micro tie-breaker (<= 0.02, strictly used only when all G gated rewards tie)
                    oracle_str = " ".join(task.get("oracle_keywords", []))
                    uniq_bonus = len(set(mid_tok_list)) / float(max(1, len(mid_tok_list)))
                    char_hit = (
                        len(set(oracle_str).intersection(set(middle_text)))
                        / float(max(1, len(set(oracle_str))))
                        if oracle_str
                        else 0.0
                    )
                    tie_breakers.append(0.01 * uniq_bonus + 0.01 * char_hit)
                    raw_rewards.append(float(fim_eval["reward"]))

                    step_rewards.append(float(fim_eval["reward"]))
                    step_logprobs.append(float(old_lp.mean().item()))
                    step_parses.append(1.0 if fim_eval["parses"] else 0.0)
                    step_compiles.append(1.0 if fim_eval["compiles"] else 0.0)
                    step_tests.append(1.0 if fim_eval["tests_passed"] else 0.0)
                    step_nontrivial.append(1.0 if fim_eval["non_trivial"] else 0.0)
                    step_kw_faithful.append(1.0 if fim_eval["kw_faithful"] else 0.0)

                    group_rollouts.append(
                        {
                            "prefix_ids": prefix_1,
                            "prefix_mask": prefix_mask_1,
                            "plan_ids": plan_t,
                            "sampled_codes": sampled_codes,
                            "old_logprobs": old_lp,
                            "ref_logprobs": ref_lp,
                        }
                    )

                rewards_t = torch.tensor(raw_rewards, dtype=torch.float32, device=dev)
                total_groups += 1
                if float(rewards_t.std(unbiased=False).item()) <= 1e-8:
                    zero_var_groups += 1
                    if skip_zero_variance_groups:
                        # Drop zero-variance groups completely instead of injecting tie-breaker noise
                        continue
                    tb_t = torch.tensor(tie_breakers, dtype=torch.float32, device=dev)
                    eff_r = rewards_t + tb_t
                else:
                    eff_r = rewards_t

                r_mean = eff_r.mean()
                r_std = eff_r.std(unbiased=False)
                if float(r_std.item()) > 1e-8:
                    advantages = (eff_r - r_mean) / (r_std + 1e-6)
                else:
                    advantages = torch.zeros_like(eff_r)

                rollout_groups.append({"rollouts": group_rollouts, "advantages": advantages})

        # 2. Policy optimization phase (skip optimizer step if all groups had zero variance)
        step_loss = 0.0
        step_pg = 0.0
        step_kl = 0.0
        step_gnorm = 0.0
        total_rollouts = sum(len(g["rollouts"]) for g in rollout_groups)

        if total_rollouts > 0:
            for _ in range(max(1, int(ppo_epochs))):
                model.train()
                _enforce_frozen_concept_tokenizer(model)
                optimizer.zero_grad(set_to_none=True)

                epoch_loss_val = 0.0
                epoch_pg_val = 0.0
                epoch_kl_val = 0.0

                for grp in rollout_groups:
                    adv_vec = grp["advantages"]
                    for g_idx, r_item in enumerate(grp["rollouts"]):
                        with amp_context(dev, dtype=amp_dtype, stage=2):
                            cur_out = _compute_pg_fim_logprobs_compat(
                                model,
                                prefix_ids=r_item["prefix_ids"],
                                plan_ids=r_item["plan_ids"],
                                sampled_codes=r_item["sampled_codes"],
                                prefix_mask=r_item["prefix_mask"],
                            )
                            cur_lp = cur_out["total_logprobs"].float()
                            if not torch.isfinite(cur_lp).all():
                                all_logprobs_finite = False

                            old_lp = r_item["old_logprobs"]
                            ref_lp = r_item["ref_logprobs"]
                            adv = adv_vec[g_idx : g_idx + 1]

                            log_ratio = (cur_lp - old_lp).clamp(-20.0, 20.0)
                            ratio = torch.exp(log_ratio)

                            if use_cispo or cispo_eps is not None:
                                eps = float(cispo_eps if cispo_eps is not None else clip_eps)
                                clipped_w = torch.clamp(ratio.detach(), 1.0 - eps, 1.0 + eps)
                                pg_loss = -(clipped_w * adv * cur_lp).mean()
                            else:
                                surr1 = ratio * adv
                                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                                pg_loss = -torch.min(surr1, surr2).mean()

                            kl_log_ratio = (ref_lp - cur_lp).clamp(-20.0, 20.0)
                            kl_loss = (torch.exp(kl_log_ratio) - kl_log_ratio - 1.0).mean()

                            loss = pg_loss + float(beta_kl) * kl_loss
                            scaled_loss = loss / float(total_rollouts)

                        scaled_loss.backward()

                        epoch_loss_val += float(loss.detach().item()) / float(total_rollouts)
                        epoch_pg_val += float(pg_loss.detach().item()) / float(total_rollouts)
                        epoch_kl_val += float(kl_loss.detach().item()) / float(total_rollouts)

                gnorm, grads_ok = clip_and_check_grad_norm(trainable_params, max_norm=grad_clip)
                if not grads_ok:
                    all_grads_finite = False
                else:
                    optimizer.step()

                step_loss = epoch_loss_val
                step_pg = epoch_pg_val
                step_kl = epoch_kl_val
                step_gnorm = gnorm

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        n_s = float(max(1, len(step_rewards)))
        mean_r = sum(step_rewards) / n_s
        mean_lp = sum(step_logprobs) / n_s
        p_rate = sum(step_parses) / n_s
        c_rate = sum(step_compiles) / n_s
        t_rate = sum(step_tests) / n_s
        nt_rate = sum(step_nontrivial) / n_s
        kw_rate = sum(step_kw_faithful) / n_s

        loss_history.append(step_loss)
        pg_loss_history.append(step_pg)
        kl_history.append(step_kl)
        mean_reward_history.append(mean_r)
        mean_logprob_history.append(mean_lp)
        parse_rate_history.append(p_rate)
        compile_rate_history.append(c_rate)
        test_pass_rate_history.append(t_rate)
        non_trivial_rate_history.append(nt_rate)
        kw_faithful_rate_history.append(kw_rate)
        grad_norm_history.append(step_gnorm)

        if verbose and (step_idx == 1 or step_idx % max(1, log_every) == 0 or step_idx == steps):
            algo = "PG-FIM-CISPO" if (use_cispo or cispo_eps is not None) else "PG-FIM-GRPO"
            print(
                f"[{algo}] step {step_idx:3d}/{steps} | "
                f"loss={step_loss:.4f} (kl={step_kl:.4f}) | "
                f"reward={mean_r:.4f} | parse={p_rate * 100:4.1f}% | "
                f"test={t_rate * 100:4.1f}% | gnorm={step_gnorm:.3f}"
            )

    _enforce_frozen_concept_tokenizer(model)
    with torch.no_grad():
        param_delta_l2 = sum(
            float((p.detach() - p0).float().pow(2).sum().item())
            for p, p0 in zip(trainable_params, initial_param_snapshot)
        ) ** 0.5

    metrics: dict[str, Any] = {
        "stage": "pg_fim_grpo",
        "algorithm": "cispo" if (use_cispo or cispo_eps is not None) else "grpo",
        "fim_mode": True,
        "steps": steps,
        "num_generations": num_generations,
        "zero_variance_group_count": zero_var_groups,
        "total_group_count": total_groups,
        "zero_variance_group_fraction": float(zero_var_groups) / float(max(1, total_groups)),
        "curriculum_stages_visited": stages_visited,
        "loss_history": loss_history,
        "pg_loss_history": pg_loss_history,
        "kl_history": kl_history,
        "mean_reward_history": mean_reward_history,
        "mean_logprob_history": mean_logprob_history,
        "parse_rate_history": parse_rate_history,
        "compile_rate_history": compile_rate_history,
        "test_pass_rate_history": test_pass_rate_history,
        "non_trivial_rate_history": non_trivial_rate_history,
        "kw_faithful_rate_history": kw_faithful_rate_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "final_mean_reward": mean_reward_history[-1],
        "final_kl": kl_history[-1],
        "final_parse_rate": parse_rate_history[-1],
        "final_compile_rate": compile_rate_history[-1],
        "final_test_pass_rate": test_pass_rate_history[-1],
        "final_non_trivial_rate": non_trivial_rate_history[-1],
        "final_kw_faithful_rate": kw_faithful_rate_history[-1],
        "all_logprobs_finite": all_logprobs_finite,
        "all_grads_finite": all_grads_finite,
        "policy_param_delta_l2": param_delta_l2,
        "policy_updated": param_delta_l2 > 0.0,
        "concept_tokenizer_frozen": all(
            not p.requires_grad for p in model.concept_tokenizer.parameters()
        ),
        "params": count_parameters(model),
        "device": str(dev),
    }

    if save_path is not None:
        save_checkpoint(
            save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=steps,
            config=cfg,
            extra={"stage": "pg_fim_grpo", "metrics": metrics},
        )

    return {"model": model, "metrics": metrics}


def train_grpo(
    cfg: CNCMConfig,
    tokenizer: CNCMTextTokenizer,
    model: CNCMModel,
    prompts: Optional[Union[GRPOPromptDataset, Sequence[Union[str, dict[str, Any]]]]] = None,
    steps: int = 10,
    num_generations: int = 4,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: Union[str, torch.device] = "auto",
    *,
    fim_mode: bool = False,
    curriculum: Sequence[str] = ("single_line", "multi_line", "function_body"),
    weight_decay: float = 0.01,
    beta_kl: float = 0.04,
    clip_eps: float = 0.2,
    use_cispo: bool = False,
    cispo_eps: Optional[float] = None,
    ppo_epochs: int = 1,
    temperature: float = 0.9,
    coarse_temp: float = 0.75,
    fine_temp: float = 0.15,
    skip_zero_variance_groups: bool = True,
    early_exit_conf: float = 1.0,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.1,
    reward_fn: Optional[Callable[[str, dict[str, Any]], float]] = None,
    amp_dtype: Union[str, torch.dtype, None] = None,
    seed: int = 42,
    log_every: int = 1,
    save_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run code-space GRPO / CISPO reinforcement learning on `CNCMModel`.

    If `fim_mode=True` (or if `prompts` contains PG-FIM task dicts with `prefix` and `suffix`),
    routes automatically to `train_pg_fim_grpo`.
    """
    is_fim_tasks = (
        isinstance(prompts, Sequence)
        and len(prompts) > 0
        and isinstance(prompts[0], dict)
        and "prefix" in prompts[0]
        and "suffix" in prompts[0]
    )
    if fim_mode or is_fim_tasks:
        fim_tasks = list(prompts) if is_fim_tasks else None  # type: ignore[arg-type]
        return train_pg_fim_grpo(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            tasks=fim_tasks,
            steps=steps,
            num_generations=num_generations,
            batch_size=batch_size,
            lr=lr,
            device=device,
            curriculum=curriculum,
            weight_decay=weight_decay,
            beta_kl=beta_kl,
            clip_eps=clip_eps,
            use_cispo=use_cispo,
            cispo_eps=cispo_eps,
            ppo_epochs=ppo_epochs,
            temperature=temperature,
            early_exit_conf=early_exit_conf,
            grad_clip=grad_clip,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
            amp_dtype=amp_dtype,
            seed=seed,
            log_every=log_every,
            save_path=save_path,
            verbose=verbose,
        )

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if num_generations < 2:
        raise ValueError(f"num_generations must be >= 2 for group normalization, got {num_generations}")

    set_seed(seed)
    dev = select_device(device)
    scorer = reward_fn or compute_verifiable_reward

    if prompts is None:
        prompt_ds = GRPOPromptDataset(
            make_synthetic_grpo_prompts(num_samples=32, seed=seed),
            tokenizer,
            bos_token_id=cfg.bos_token_id,
        )
    elif isinstance(prompts, GRPOPromptDataset):
        prompt_ds = prompts
    else:
        prompt_ds = GRPOPromptDataset(prompts, tokenizer, bos_token_id=cfg.bos_token_id)

    model = model.to(dev)
    _enforce_frozen_concept_tokenizer(model)

    ref_model = copy.deepcopy(model).to(dev)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    stage1_param_ids = {id(p) for p in model.concept_tokenizer.parameters()}
    trainable_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in stage1_param_ids
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in CNCMModel for GRPO.")

    initial_param_snapshot = [p.detach().clone() for p in trainable_params]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    eff_bs = max(1, min(int(batch_size), len(prompt_ds)))
    loss_history: list[float] = []
    pg_loss_history: list[float] = []
    kl_history: list[float] = []
    mean_reward_history: list[float] = []
    mean_logprob_history: list[float] = []
    grad_norm_history: list[float] = []
    all_logprobs_finite = True
    all_grads_finite = True

    cursor = 0
    for step_idx in range(1, steps + 1):
        rollout_items: list[dict[str, Any]] = []
        step_rewards: list[float] = []
        step_logprobs: list[float] = []

        model.eval()
        with torch.no_grad():
            for _ in range(eff_bs):
                item = prompt_ds[cursor % len(prompt_ds)]
                cursor += 1

                prefix_1 = item["prompt_ids"].unsqueeze(0).to(dev)  # [1, P]
                prefix_g = prefix_1.expand(num_generations, -1).contiguous()  # [G, P]
                prefix_mask_g = torch.ones_like(prefix_g, dtype=torch.long)

                gen_out = model.generate_concept_chunk(
                    prefix_g,
                    temperature=temperature,
                    early_exit_conf=early_exit_conf,
                    return_codes=True,
                )
                token_ids = gen_out["token_ids"]
                sampled_codes = gen_out["codes"]
                if sampled_codes is None:
                    raise RuntimeError(
                        "model.generate_concept_chunk(..., return_codes=True) returned codes=None."
                    )

                spec = {
                    "task_type": item.get("task_type", "format_reasoning"),
                    "expected_answer": item.get("expected_answer", ""),
                    "expected_tool": item.get("expected_tool"),
                    "expected_args": item.get("expected_args"),
                    "require_think": item.get("require_think", True),
                    "require_tool_call": item.get("require_tool_call", False),
                }
                rewards_list = [
                    float(scorer(_decode_tokens(tokenizer, token_ids[g].tolist()), spec))
                    for g in range(num_generations)
                ]
                rewards_t = torch.tensor(rewards_list, dtype=torch.float32, device=dev)
                r_mean = rewards_t.mean()
                r_std = rewards_t.std(unbiased=False)
                if float(r_std.item()) > 1e-8:
                    advantages = (rewards_t - r_mean) / (r_std + 1e-6)
                else:
                    advantages = torch.zeros_like(rewards_t)

                old_lp = model.compute_code_logprobs(
                    prefix_g,
                    sampled_codes,
                    prefix_mask=prefix_mask_g,
                ).detach().float()
                ref_lp = ref_model.compute_code_logprobs(
                    prefix_g,
                    sampled_codes,
                    prefix_mask=prefix_mask_g,
                ).detach().float()

                if not (torch.isfinite(old_lp).all() and torch.isfinite(ref_lp).all()):
                    all_logprobs_finite = False

                step_rewards.extend(rewards_list)
                step_logprobs.append(float(old_lp.mean().item()))
                rollout_items.append(
                    {
                        "prefix_ids": prefix_g,
                        "prefix_mask": prefix_mask_g,
                        "sampled_codes": [c.detach() for c in sampled_codes],
                        "advantages": advantages,
                        "old_logprobs": old_lp,
                        "ref_logprobs": ref_lp,
                    }
                )

        step_loss = 0.0
        step_pg = 0.0
        step_kl = 0.0
        step_gnorm = 0.0

        for _ in range(max(1, int(ppo_epochs))):
            model.train()
            _enforce_frozen_concept_tokenizer(model)
            optimizer.zero_grad(set_to_none=True)

            epoch_loss_val = 0.0
            epoch_pg_val = 0.0
            epoch_kl_val = 0.0

            for r_item in rollout_items:
                with amp_context(dev, dtype=amp_dtype, stage=2):
                    cur_lp = model.compute_code_logprobs(
                        r_item["prefix_ids"],
                        r_item["sampled_codes"],
                        prefix_mask=r_item["prefix_mask"],
                    ).float()

                    if not torch.isfinite(cur_lp).all():
                        all_logprobs_finite = False

                    old_lp = r_item["old_logprobs"]
                    ref_lp = r_item["ref_logprobs"]
                    adv = r_item["advantages"]

                    log_ratio = (cur_lp - old_lp).clamp(-20.0, 20.0)
                    ratio = torch.exp(log_ratio)

                    if use_cispo or cispo_eps is not None:
                        eps = float(cispo_eps if cispo_eps is not None else clip_eps)
                        clipped_w = torch.clamp(ratio.detach(), 1.0 - eps, 1.0 + eps)
                        pg_loss = -(clipped_w * adv * cur_lp).mean()
                    else:
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                        pg_loss = -torch.min(surr1, surr2).mean()

                    kl_log_ratio = (ref_lp - cur_lp).clamp(-20.0, 20.0)
                    kl_loss = (torch.exp(kl_log_ratio) - kl_log_ratio - 1.0).mean()

                    loss = pg_loss + float(beta_kl) * kl_loss
                    scaled_loss = loss / float(len(rollout_items))

                scaled_loss.backward()

                epoch_loss_val += float(loss.detach().item()) / float(len(rollout_items))
                epoch_pg_val += float(pg_loss.detach().item()) / float(len(rollout_items))
                epoch_kl_val += float(kl_loss.detach().item()) / float(len(rollout_items))

            gnorm, grads_ok = clip_and_check_grad_norm(trainable_params, max_norm=grad_clip)
            if not grads_ok:
                all_grads_finite = False
            else:
                optimizer.step()

            step_loss = epoch_loss_val
            step_pg = epoch_pg_val
            step_kl = epoch_kl_val
            step_gnorm = gnorm

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        mean_r = sum(step_rewards) / float(max(1, len(step_rewards)))
        mean_lp = sum(step_logprobs) / float(max(1, len(step_logprobs)))

        loss_history.append(step_loss)
        pg_loss_history.append(step_pg)
        kl_history.append(step_kl)
        mean_reward_history.append(mean_r)
        mean_logprob_history.append(mean_lp)
        grad_norm_history.append(step_gnorm)

        if verbose and (step_idx == 1 or step_idx % max(1, log_every) == 0 or step_idx == steps):
            algo = "CISPO" if (use_cispo or cispo_eps is not None) else "GRPO"
            print(
                f"[{algo}] step {step_idx:3d}/{steps} | "
                f"loss={step_loss:.4f} (pg={step_pg:.4f}, kl={step_kl:.4f}) | "
                f"reward={mean_r:.4f} | logp={mean_lp:.2f} | "
                f"gnorm={step_gnorm:.3f} | lr={current_lr:.2e}"
            )

    _enforce_frozen_concept_tokenizer(model)
    with torch.no_grad():
        param_delta_l2 = sum(
            float((p.detach() - p0).float().pow(2).sum().item())
            for p, p0 in zip(trainable_params, initial_param_snapshot)
        ) ** 0.5

    metrics: dict[str, Any] = {
        "stage": "grpo",
        "algorithm": "cispo" if (use_cispo or cispo_eps is not None) else "grpo",
        "fim_mode": False,
        "steps": steps,
        "num_generations": num_generations,
        "loss_history": loss_history,
        "pg_loss_history": pg_loss_history,
        "kl_history": kl_history,
        "mean_reward_history": mean_reward_history,
        "mean_logprob_history": mean_logprob_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "final_mean_reward": mean_reward_history[-1],
        "final_kl": kl_history[-1],
        "all_logprobs_finite": all_logprobs_finite,
        "all_grads_finite": all_grads_finite,
        "policy_param_delta_l2": param_delta_l2,
        "policy_updated": param_delta_l2 > 0.0,
        "concept_tokenizer_frozen": all(
            not p.requires_grad for p in model.concept_tokenizer.parameters()
        ),
        "params": count_parameters(model),
        "device": str(dev),
    }

    if save_path is not None:
        save_checkpoint(
            save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=steps,
            config=cfg,
            extra={"stage": "grpo", "metrics": metrics},
        )

    return {"model": model, "metrics": metrics}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNCM with Code-Space or PG-FIM GRPO / CISPO.")
    parser.add_argument("--preset", type=str, default="tiny_test", choices=["tiny_test", "nano", "mini", "base"])
    parser.add_argument("--fim", action="store_true", help="Run Compiler-in-the-Loop PG-FIM GRPO.")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=0.04)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--cispo", action="store_true", help="Use CISPO clipped importance sampling.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--stage2-ckpt", type=str, default=None, help="Optional Stage 2 CNCMModel checkpoint.")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = get_preset(args.preset)
    tokenizer = CNCMTextTokenizer(vocab_size=cfg.vocab_size)
    concept_tokenizer = CNCMConceptTokenizer(cfg)
    model = CNCMModel(cfg, concept_tokenizer)
    if args.stage2_ckpt:
        load_checkpoint(args.stage2_ckpt, model=model, map_location="cpu")

    if args.fim:
        tasks = make_synthetic_fim_tasks(num_samples=args.num_prompts, seed=args.seed)
        result = train_pg_fim_grpo(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            tasks=tasks,
            steps=args.steps,
            num_generations=args.num_generations,
            batch_size=args.batch_size,
            lr=args.lr,
            beta_kl=args.beta_kl,
            clip_eps=args.clip_eps,
            use_cispo=args.cispo,
            temperature=args.temperature,
            device=args.device,
            seed=args.seed,
            save_path=args.save_path,
            verbose=not args.quiet,
        )
    else:
        prompts = make_synthetic_grpo_prompts(num_samples=args.num_prompts, seed=args.seed)
        result = train_grpo(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            steps=args.steps,
            num_generations=args.num_generations,
            batch_size=args.batch_size,
            lr=args.lr,
            beta_kl=args.beta_kl,
            clip_eps=args.clip_eps,
            use_cispo=args.cispo,
            temperature=args.temperature,
            device=args.device,
            seed=args.seed,
            save_path=args.save_path,
            verbose=not args.quiet,
        )
    if not args.quiet:
        m = result["metrics"]
        print(
            f"[GRPO Done] reward={m['final_mean_reward']:.4f} | "
            f"kl={m['final_kl']:.4f} | param_delta={m['policy_param_delta_l2']:.5f}"
        )


if __name__ == "__main__":
    main()
