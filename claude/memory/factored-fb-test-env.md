---
name: factored-fb-test-env
description: How to run tests in the Factored-FB repo and which test failures are pre-existing noise
metadata: 
  node_type: memory
  type: project
  originSessionId: 0824770a-d1ae-4311-a1a2-32d27e3ec12b
---

In the Factored-FB repo, `uv run` is broken (pyproject `build-backend = "setuptools.backends.legacy:build"` fails). Run tests with `.venv/bin/python -m pytest ...` instead (pytest is installed in `.venv`).

Six test modules fail/error at collection due to missing third-party deps (e.g. `pandas`), UNRELATED to core work and NOT regressions: `tests/test_analysis.py`, `tests/test_phase_probe.py`, `tests/test_representation_profile.py` (FAIL) and `tests/test_paper_fragment.py`, `tests/test_plot_sweep.py`, `tests/test_wandb_pull.py` (collection ERROR). Use `--continue-on-collection-errors` to see the full run; treat only non-listed failures as real.

**Why:** avoids re-diagnosing a broken `uv` and mistaking known-missing-dep failures for regressions.
**How to apply:** scope "all green" claims to the relevant test modules; verify the full suite with `--continue-on-collection-errors` and confirm any failure is outside the six modules above.
