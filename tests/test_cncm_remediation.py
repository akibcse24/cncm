"""Comprehensive unit tests for CNCM remediation tooling, eval suites, grammar, and sandbox.

Pure-Python / NumPy / SciPy tests run anywhere (even without PyTorch installed).
PyTorch-dependent checks are cleanly skipped when `torch` is unavailable.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from cncm.agent.grammar import (
    evaluate_tool_call_semantic_accuracy,
    extract_json_object,
    extract_prompt_math_spans,
)
from cncm.agent.tools import create_default_registry, safe_eval_math
from cncm.code.sandbox import run_code_sandbox
from cncm.config import get_preset, list_presets
from experiments.eval_suites import (
    MANIFEST_PATH,
    build_heldout_arithmetic_suite,
    build_heldout_tool_calling_suite,
    build_pg_fim_heldout_test_bank,
    evaluate_fim_suite,
    evaluate_tool_suite_with_ci,
    verify_manifest,
)
from experiments.harness import compute_config_hash, load_result, run_seeded_experiment, save_result
from experiments.modal_tier_b import check_concept_quality_gate
from experiments.stats import (
    bootstrap_ci,
    expected_speculative_speedup,
    paired_nonparametric_test,
    paired_t_test,
    t_interval,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestStatsUtilities(unittest.TestCase):
    def test_wilson_interval_bounds(self) -> None:
        w0 = wilson_interval(0, 100)
        self.assertEqual(w0["rate"], 0.0)
        self.assertEqual(w0["low"], 0.0)
        self.assertGreater(w0["high"], 0.0)
        self.assertLess(w0["high"], 0.05)

        w50 = wilson_interval(50, 100)
        self.assertAlmostEqual(w50["rate"], 0.5, places=6)
        self.assertAlmostEqual(w50["low"], 0.4038, places=3)
        self.assertAlmostEqual(w50["high"], 0.5962, places=3)

        w100 = wilson_interval(24, 24)
        self.assertEqual(w100["rate"], 1.0)
        self.assertEqual(w100["high"], 1.0)
        self.assertLess(w100["low"], 1.0)

    def test_t_and_bootstrap_ci(self) -> None:
        vals = [1.0, 2.0, 3.0]
        ti = t_interval(vals)
        self.assertAlmostEqual(ti["mean"], 2.0, places=6)
        self.assertAlmostEqual(ti["std"], 1.0, places=6)
        self.assertLess(ti["low"], 2.0)
        self.assertGreater(ti["high"], 2.0)

        bi = bootstrap_ci(vals, n_resamples=2000, seed=42)
        self.assertAlmostEqual(bi["point"], 2.0, places=6)
        self.assertLessEqual(bi["low"], bi["point"])
        self.assertGreaterEqual(bi["high"], bi["point"])

    def test_paired_tests(self) -> None:
        a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        b = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
        pt = paired_t_test(a, b)
        self.assertAlmostEqual(pt["mean_diff"], 2.0, places=6)
        self.assertTrue(pt["significant_05"])

        np_res = paired_nonparametric_test(a, b)
        self.assertEqual(np_res["n_pos"], 6)
        self.assertTrue(np_res["significant_05"])

    def test_expected_speculative_speedup(self) -> None:
        # alpha=0 => expected 1 token per cycle; alpha=1 => gamma+1 tokens per cycle
        s0 = expected_speculative_speedup(alpha=0.0, gamma=8, draft_cycle_cost_ratio=0.5, verify_cycle_cost_ratio=1.0)
        self.assertAlmostEqual(s0["expected_tokens_per_cycle"], 1.0, places=4)
        self.assertAlmostEqual(s0["expected_speedup_vs_ar"], 1.0 / 1.5, places=4)

        s1 = expected_speculative_speedup(alpha=1.0, gamma=8, draft_cycle_cost_ratio=0.5, verify_cycle_cost_ratio=1.0)
        self.assertAlmostEqual(s1["expected_tokens_per_cycle"], 9.0, places=4)
        self.assertAlmostEqual(s1["expected_speedup_vs_ar"], 6.0, places=4)


class TestHarnessAndLedger(unittest.TestCase):
    def test_config_hash_determinism(self) -> None:
        h1 = compute_config_hash({"a": 1, "b": [2, 3]})
        h2 = compute_config_hash({"b": [2, 3], "a": 1})
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_seeded_experiment_and_ledger_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = run_seeded_experiment(
                "unit_dummy_exp",
                lambda seed: {"loss": 2.0 + (seed % 5) * 0.05, "acc": 0.8},
                seeds=(42, 1337, 2025),
                config={"preset": "tiny_test", "steps": 10},
                steps=10,
                total_tokens=1024,
                results_dir=tmp,
            )
            self.assertTrue(res["meets_min_seed_policy"])
            self.assertIn("loss", res["aggregated"])
            loaded = load_result("unit_dummy_exp", results_dir=tmp)
            self.assertEqual(loaded["steps"], 10)
            self.assertEqual(loaded["total_tokens"], 1024)
            self.assertEqual(loaded["seeds"], [42, 1337, 2025])


class TestEvalSuitesAndManifest(unittest.TestCase):
    def test_manifest_sha256_integrity(self) -> None:
        self.assertTrue(MANIFEST_PATH.exists(), "SHA-256 manifest must be pre-committed")
        checks = verify_manifest(MANIFEST_PATH)
        self.assertTrue(all(checks.values()), f"Manifest mismatch: {checks}")

    def test_pg_fim_heldout_oracle_passes_sandbox(self) -> None:
        tasks = build_pg_fim_heldout_test_bank()
        self.assertEqual(len(tasks), 36)
        # Verify all 36 held-out oracle solutions parse, compile, and pass unit tests
        by_prefix_suffix = {(t["prefix"], t["suffix"]): t["middle"] for t in tasks}
        res = evaluate_fim_suite(tasks, lambda p, s: by_prefix_suffix[(p, s)])
        self.assertEqual(res["pass_at_1_wilson_95"]["successes"], 36)
        self.assertEqual(res["pass_at_1_wilson_95"]["rate"], 1.0)

    def test_heldout_arithmetic_and_tool_suites(self) -> None:
        arith = build_heldout_arithmetic_suite(1000)
        self.assertEqual(len(arith), 1000)
        for sample in arith[:25]:
            self.assertAlmostEqual(safe_eval_math(sample["expression"]), sample["expected_value"], places=5)

        tools = build_heldout_tool_calling_suite(500)
        self.assertEqual(len(tools), 500)
        # Oracle predictions should achieve 100% syntax and 100% semantic accuracy
        oracle_preds = [
            json.dumps({"name": t["expected_tool"], "arguments": t["expected_args"]})
            for t in tools
        ]
        scored = evaluate_tool_suite_with_ci(oracle_preds, tools)
        self.assertEqual(scored["syntax_wilson_95"]["rate"], 1.0)
        self.assertEqual(scored["semantic_wilson_95"]["rate"], 1.0)

        # Valid JSON with wrong math expression => 100% syntax, low semantic accuracy (Item 5.3)
        bad_math_preds = [
            json.dumps({"name": "calculator", "arguments": {"expression": "1+1"}})
            for t in tools[:40]
            if t["expected_tool"] == "calculator"
        ]
        calc_tasks = [t for t in tools[:40] if t["expected_tool"] == "calculator"]
        scored_bad = evaluate_tool_suite_with_ci(bad_math_preds, calc_tasks)
        self.assertEqual(scored_bad["syntax_wilson_95"]["rate"], 1.0)
        self.assertEqual(scored_bad["semantic_wilson_95"]["rate"], 0.0)


class TestGrammarAndConfig(unittest.TestCase):
    def test_extract_prompt_math_spans_extended(self) -> None:
        self.assertIn("17*6+9", extract_prompt_math_spans("Compute 17 * 6 + 9 please"))
        self.assertIn("(12+8)*3", extract_prompt_math_spans("Evaluate (12 + 8) * 3 now"))
        self.assertIn("3.5*2", extract_prompt_math_spans("What is 3.5 * 2?"))

    def test_presets_and_dense220(self) -> None:
        presets = list_presets()
        for name in ("tiny_test", "nano", "mini", "base", "base_moe", "base_dense220"):
            self.assertIn(name, presets)
            cfg = get_preset(name)
            self.assertEqual(cfg.preset_name, name)
        d220 = get_preset("base_dense220")
        self.assertEqual(d220.intermediate_size, 6912)
        self.assertEqual(d220.num_experts, 0)

    def test_quality_gate_item_4_3(self) -> None:
        fail = check_concept_quality_gate(concept_draft_nll=8.30, exact_nll=7.9, unigram_nll=8.0)
        self.assertFalse(fail["quality_gate_passed"])
        ok = check_concept_quality_gate(concept_draft_nll=5.2, exact_nll=3.1, unigram_nll=8.0)
        self.assertTrue(ok["quality_gate_passed"])

    def test_all_python_files_parse_cleanly(self) -> None:
        py_files = list((REPO_ROOT / "cncm").rglob("*.py")) + list((REPO_ROOT / "experiments").rglob("*.py"))
        self.assertGreater(len(py_files), 25)
        for p in py_files:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))



class TestDistillSeqKDAndSWA(unittest.TestCase):
    def test_cap_think_block_and_verification(self) -> None:
        from experiments.distill_seqkd import cap_think_block, filter_teacher_records

        long_think = "<think>" + ("reasoning " * 80) + "</think>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"12+8\"}}"
        capped = cap_think_block(long_think, max_think_chars=60)
        self.assertLess(len(capped), len(long_think))
        self.assertIn("</think>", capped)

        records = [
            {
                "domain": "code",
                "prefix": "def add_one(x):\n    ",
                "completion": "return x + 1",
                "suffix": "\n",
                "test_code": "assert add_one(4) == 5",
            },
            {
                "domain": "code",
                "prefix": "def add_one(x):\n    ",
                "completion": "return x - 99",
                "suffix": "\n",
                "test_code": "assert add_one(4) == 5",
            },
            {
                "domain": "tool_call",
                "completion": "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"(12+8)*3\"}}",
                "expected_tool": "calculator",
                "expected_value": 60,
            },
            {
                "domain": "tool_call",
                "completion": "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"1+1\"}}",
                "expected_tool": "calculator",
                "expected_value": 60,
            },
        ]
        res = filter_teacher_records(records)
        self.assertEqual(res["total_seen"], 4)
        self.assertEqual(res["accepted_count"], 2)
        self.assertEqual(res["rejected_count"], 2)

    def test_sliding_window_config_fields(self) -> None:
        cfg = get_preset("base", sliding_window=2048, attention_sink_tokens=4, global_attn_every_n_layers=4)
        self.assertEqual(cfg.sliding_window, 2048)
        self.assertEqual(cfg.attention_sink_tokens, 4)
        self.assertEqual(cfg.global_attn_every_n_layers, 4)



class TestSandboxEndToEnd(unittest.TestCase):
    def test_text_tokenizer_lossless_roundtrip(self) -> None:
        from cncm.text_tokenizer import CNCMTextTokenizer

        tok = CNCMTextTokenizer(vocab_size=4096)
        samples = [
            "Hello CNCM edge model! 17 * 6 + 9 = 111",
            "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
            "বাংলা + English + JSON: {\"name\": \"calculator\", \"arguments\": {\"expression\": \"(12+8)*3\"}}",
        ]
        for s in samples:
            ids = tok.encode(s)
            self.assertGreater(len(ids), 0)
            self.assertEqual(tok.decode(ids), s)

    def test_code_symbols_and_sandbox_security(self) -> None:
        from cncm.code.symbols import extract_oracle_keywords, extract_symbols

        code = "def clamp_val(x, lo, hi):\n    res = max(lo, min(hi, x))\n    return res\n"
        tbl = extract_symbols(code)
        self.assertIn("clamp_val", tbl.defined_identifiers)
        kws = extract_oracle_keywords("res = max(lo, min(hi, x))", prefix="def clamp_val(x, lo, hi):\n    ")
        self.assertGreater(len(kws), 0)

        # Verify sandbox blocks unsafe imports (e.g., os.system)
        unsafe = run_code_sandbox("import os\nos.system('echo hacked')", timeout_sec=1.0)
        self.assertFalse(unsafe.get("tests_passed", False))
        self.assertFalse(unsafe.get("compiles", True) and unsafe.get("ran", False))

    def test_tool_registry_execution(self) -> None:
        reg = create_default_registry()
        r_calc = reg.execute("calculator", {"expression": "(12 + 8) * 3"})
        self.assertTrue(r_calc.ok)
        self.assertIn("60", str(r_calc.output))

        r_kv = reg.execute("kv_memory", {"action": "set", "key": "mode", "value": "concept"})
        self.assertTrue(r_kv.ok)
        r_get = reg.execute("kv_memory", {"action": "get", "key": "mode"})
        self.assertTrue(r_get.ok)
        self.assertIn("concept", str(r_get.output))

    def test_cli_distill_seqkd_and_tier_b_cost_gate(self) -> None:
        import subprocess, sys

        # 1. Verify Tier B refuses to run without --confirm-paid-gpu
        proc = subprocess.run(
            [sys.executable, "-m", "experiments.modal_tier_b", "--exp", "1.2a"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--confirm-paid-gpu", proc.stderr + proc.stdout)

        # 2. End-to-end SeqKD CLI test on a temporary raw teacher JSONL
        with tempfile.TemporaryDirectory() as tmp:
            raw_p = Path(tmp) / "raw.jsonl"
            out_p = Path(tmp) / "verified.jsonl"
            raw_rows = [
                {"domain": "code", "prefix": "def sq(x):\n    ", "completion": "<think>simple</think>return x * x", "suffix": "\n", "test_code": "assert sq(5) == 25"},
                {"domain": "tool_call", "completion": "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"7*8\"}}", "expected_tool": "calculator", "expected_value": 56},
                {"domain": "tool_call", "completion": "{\"name\": \"calculator\", \"arguments\": {\"expression\": \"1+1\"}}", "expected_tool": "calculator", "expected_value": 56},
            ]
            raw_p.write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n", encoding="utf-8")
            p2 = subprocess.run(
                [sys.executable, "-m", "experiments.distill_seqkd", "--input", str(raw_p), "--output", str(out_p)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(p2.returncode, 0, msg=p2.stderr)
            meta = json.loads(out_p.with_suffix(".meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["accepted_count"], 1)  # code had <think> inside code body (fails parse), valid tool passes, wrong tool rejected

    def test_edge_bench_telemetry_runs(self) -> None:
        from cncm.export.edge_bench import read_system_telemetry

        telem = read_system_telemetry()
        self.assertIsInstance(telem, dict)
        self.assertIn("peak_rss_mb", telem)



class TestEdgeKVStack(unittest.TestCase):
    def test_cla_2to1_halves_kv_layers_and_memory(self) -> None:
        from cncm.config import compute_kv_memory_footprint

        cfg_std = get_preset("base")
        cfg_cla = get_preset("base", kv_sharing_group_size=2)
        m_std = compute_kv_memory_footprint(cfg_std, seq_len=4096, kv_bytes=1.05)
        m_cla = compute_kv_memory_footprint(cfg_cla, seq_len=4096, kv_bytes=1.05)
        self.assertEqual(m_std["unique_kv_layers"], 24)
        self.assertEqual(m_cla["unique_kv_layers"], 12)
        self.assertAlmostEqual(m_cla["reduction_factor"], 2.0, places=2)

    def test_concept_compressed_prefix_plan_64k(self) -> None:
        from cncm.config import plan_concept_compressed_prefix

        # Short prefix -> untouched (ratio 1.0)
        short = plan_concept_compressed_prefix(total_prefix_len=512, recent_window=1024, chunk_size=64, coarse_level_len=4)
        self.assertEqual(short["total_kv_slots"], 512)
        self.assertEqual(short["compression_ratio"], 1.0)

        # 64K prefix with recent_window=2048, coarse l_2=4 (16x compression on evicted chunks)
        long_p = plan_concept_compressed_prefix(total_prefix_len=65536, recent_window=2048, chunk_size=64, coarse_level_len=4, sink_tokens=4)
        self.assertGreater(long_p["compression_ratio"], 10.0)
        self.assertLess(long_p["total_kv_slots"], 6200)

    def test_combined_cla_swa_concept_kv_at_64k_on_base(self) -> None:
        from cncm.config import compute_kv_memory_footprint

        # Baseline: full INT8 KV on base @ 64K (~605 MB)
        cfg_full = get_preset("base")
        m_full = compute_kv_memory_footprint(cfg_full, seq_len=65536, kv_bytes=1.05)
        self.assertGreater(m_full["kv_mb"], 550.0)

        # Combined Edge Stack: CLA 2:1 + SWA 2048 + 4 sinks + Concept-Compressed old turns (l_2=4, 16x) + INT8 KV
        cfg_edge = get_preset(
            "base",
            kv_sharing_group_size=2,
            sliding_window=2048,
            attention_sink_tokens=4,
            concept_kv_compress=True,
            concept_kv_level=2,
        )
        m_edge = compute_kv_memory_footprint(cfg_edge, seq_len=65536, kv_bytes=1.05)
        # Drops from ~605 MB down to < 30 MB (>20x reduction!)
        self.assertLess(m_edge["kv_mb"], 30.0)
        self.assertGreater(m_edge["reduction_factor"], 20.0)
        print(f"\n[Edge KV 64K Base] Full INT8={m_full['kv_mb']:.1f} MB -> CLA+SWA+Concept INT8={m_edge['kv_mb']:.1f} MB ({m_edge['reduction_factor']:.1f}x smaller)")

    def test_invalid_cla_or_concept_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_preset("nano", kv_sharing_group_size=0)
        with self.assertRaises(ValueError):
            get_preset("nano", concept_kv_level=99)



class TestSmartPipelineAndSingleSeed(unittest.TestCase):
    def test_single_seed_default_everywhere(self) -> None:
        from experiments.harness import DEFAULT_SEEDS

        self.assertEqual(DEFAULT_SEEDS, (42,))
        with tempfile.TemporaryDirectory() as tmp:
            res = run_seeded_experiment(
                "single_seed_check",
                lambda seed: {"loss": 1.25},
                results_dir=tmp,
            )
            self.assertEqual(res["n_seeds"], 1)
            self.assertTrue(res["meets_min_seed_policy"])

    def test_smart_pipeline_config_and_seqkd_loader(self) -> None:
        from experiments.train_smart_cncm import (
            DEFAULT_LAMBDA_E,
            build_smart_config,
            load_verified_seqkd_texts,
        )

        self.assertAlmostEqual(DEFAULT_LAMBDA_E, 0.7, places=4)
        cfg = build_smart_config("mini", kv_sharing_group_size=2, sliding_window=2048, concept_kv_compress=True)
        self.assertEqual(cfg.preset_name, "mini")
        self.assertEqual(cfg.kv_sharing_group_size, 2)
        self.assertEqual(cfg.sliding_window, 2048)
        self.assertTrue(cfg.concept_kv_compress)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "seqkd.jsonl"
            p.write_text(
                json.dumps({"prompt": "Compute 2+2", "completion": "4", "verified": True}) + "\n"
                + json.dumps({"prompt": "Bad", "completion": "99", "verified": False}) + "\n",
                encoding="utf-8",
            )
            texts = load_verified_seqkd_texts(p)
            self.assertEqual(len(texts), 1)
            self.assertIn("Compute 2+2", texts[0])


if __name__ == "__main__":
    unittest.main()
