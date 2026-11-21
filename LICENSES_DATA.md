# Dataset, Benchmark, and Reference Code License Audit (`LICENSES_DATA.md`)

This document records the verified licenses, usage conditions, and release risks for every external dataset, benchmark, and reference implementation referenced in the `ctx0an` / CNCM repository.

> **Repository License Status:** This repository does not yet contain a top-level `LICENSE` file. Add an explicit license (e.g., `MIT` or `Apache-2.0`) before formal release, and verify compatibility with any bundled or derived artifacts below.

---

## 1. Training Datasets Referenced in Repository

| Dataset | Referenced In | License (SPDX / Name) | Conditions & Release Notes | Source |
|---|---|---|---|---|
| `roneneldan/TinyStories` | `modal_audit.py:109`, `modal_run.py:126`, `modal_v2_verify.py:66`, `kaggle_pilot_50m.py:141`, `train_ctx0an_base.py:289` | `CDLA-Sharing-1.0` | Community Data License Agreement – Sharing 1.0. Requires sharing modifications to the dataset under the same terms; model weights trained on it are generally permitted (consult counsel for commercial redistribution of derived data). | https://huggingface.co/datasets/roneneldan/TinyStories |
| `wikitext` (`wikitext-2-raw-v1` / `Salesforce/wikitext`) | `modal_audit.py:110`, `modal_run.py:125` | `CC-BY-SA-3.0` / `GFDL-1.3` | Derived from Wikipedia. Share-Alike applies to derivative datasets; attribution required. | https://huggingface.co/datasets/Salesforce/wikitext |
| `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`, `fineweb-edu-dedup`, `python-edu`) | `train_ctx0an_base.py:14-15,274,287` | `ODC-By-1.0` | Open Data Commons Attribution v1.0. Subject to Common Crawl ToU (`fineweb-edu-dedup`), original publisher rights, and Software Heritage / The Stack v2 terms for `python-edu`. | https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus |
| `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | `kaggle_pilot_50m.py:142`, `train_ctx0an_base.py:288` | `ODC-By-1.0` | Open Data Commons Attribution v1.0; subject to Common Crawl Terms of Use. | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| `HuggingFaceTB/finemath` (`finemath-4plus`) | `train_ctx0an_base.py:332` | `ODC-By-1.0` | Open Data Commons Attribution v1.0; subject to Common Crawl Terms of Use. | https://huggingface.co/datasets/HuggingFaceTB/finemath |
| `flytech/python-codes-25k` | `train_ctx0an_base.py:15` | **Gemma Terms of Use** (non-OSI) | **Flagged Risk:** Listed under Gemma License on HF. Do **not** redistribute or mix into an unrestricted Apache-2.0/MIT training recipe without reviewing Gemma ToU restrictions; replace with `ODC-By` / `MIT` / `Apache-2.0` Python corpora for clean release. | https://huggingface.co/datasets/flytech/python-codes-25k |
| `meta-math/MetaMathQA` | `train_ctx0an_base.py:16` | `MIT` | MIT License on HF. Note: bootstrapped from GSM8K/MATH using GPT-3.5-Turbo; review OpenAI Terms of Use if releasing commercial weights trained on synthetic GPT outputs. | https://huggingface.co/datasets/meta-math/MetaMathQA |
| `openai/gsm8k` | `train_ctx0an_base.py:16,346` | `MIT` | Permissive MIT License. | https://huggingface.co/datasets/openai/gsm8k |
| Built-in synthetic corpora (`make_synthetic_*`) | `cncm/trainer/dataset.py` | Repo-authored | Generated programmatically in Python within this repository; no external data bundled. | Local (`cncm/trainer/dataset.py`) |

---

## 2. Evaluation Benchmarks

| Benchmark | Used In | License (SPDX) | Conditions & Notes | Source |
|---|---|---|---|---|
| **HumanEval-Infilling** (`openai/human-eval-infilling`) | `experiments/eval_benchmarks.py` | `MIT` | Permissive MIT License. Standard FIM code-infilling benchmark (single-line, multi-line, random-span). | https://github.com/openai/human-eval-infilling |
| **SAFIM** (`gonglinyuan/safim`) | `experiments/eval_benchmarks.py` | `CC-BY-SA-4.0` | Attribution + Share-Alike 4.0. Used strictly for evaluation (never bundled into training data). | https://huggingface.co/datasets/gonglinyuan/safim |
| **Fresh Held-Out PG-FIM v3 Test Set** | `experiments/eval_benchmarks.py` | Repo-authored | Freshly generated with disjoint templates & seed, SHA-256 committed prior to evaluation. | Local (`experiments/benchmarks/pg_fim_v3_test.json`) |
| **Held-Out Arithmetic & Tool-Calling Suites** | `experiments/eval_benchmarks.py` | Repo-authored | Programmatically generated held-out suites (1,000 arithmetic + 500 tool-calling tasks). | Local (`experiments/benchmarks/`) |

---

## 3. Reference Architectures & Prior Work Code

| Project / Paper | Relation to CNCM | License (SPDX) | Source |
|---|---|---|---|
| **Next-Concept Modeling (NCM)** — Stanford Hazy Research (*Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*, arXiv:2610.14162) | Architectural inspiration for coarse-to-fine multi-level concept modeling. CNCM is a clean-room PyTorch implementation replacing RVQ+EMA with M-RFSQ, adding causal-prefix dual-mode Exact-AR, continuous scaled RoPE, and PG-FIM. | `MIT` | https://github.com/HazyResearch/ncm |
| **Large Concept Models (LCM)** — Meta AI (arXiv:2412.08821) | Conceptual reference for concept-space modeling (not code-derived). | `MIT` | https://github.com/facebookresearch/large_concept_model |

---

## 4. Pre-Release Compliance Checklist

- [ ] Add a root `LICENSE` file to `ctx0an/` (recommended: `MIT` or `Apache-2.0`).
- [ ] Replace `flytech/python-codes-25k` (Gemma Terms of Use) in `train_ctx0an_base.py` before any formal public training release.
- [ ] Keep `wikitext` (`CC-BY-SA-3.0`) and `SAFIM` (`CC-BY-SA-4.0`) out of any redistributed dataset packages; use SAFIM for evaluation only.
- [ ] Retain `THIRD_PARTY_NOTICES.md` with attribution for `ODC-By-1.0`, `CDLA-Sharing-1.0`, `MIT`, and `CC-BY-SA` resources.
