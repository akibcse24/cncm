"""CNCM Remediation & Evaluation Experiments Package.

Shared tooling:
  - `experiments.stats`: Wilson score intervals, Student-t & bootstrap CIs, paired tests.
  - `experiments.harness`: Deterministic seeded experiment runner & JSON results ledger.
  - `experiments.run_controls`: Tier A controlled experiments (1.1, 1.2b, 1.3, 1.5, 5.1-5.4).
  - `experiments.eval_suites`: Held-out PG-FIM test set (SHA-256 locked), HumanEval-Infilling,
    SAFIM, held-out arithmetic (>=1,000), and held-out tool-calling (>=500).
  - `experiments.modal_tier_b`: Cost-gated Tier B GPU runners (1.2a, 1.4, 4.1-4.3).
"""
