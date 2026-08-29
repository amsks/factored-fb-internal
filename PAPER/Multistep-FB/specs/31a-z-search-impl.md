# Spec 31 (z-search interface) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the zero-order (encoder-seeded CEM) reward→z interface for
Density-FB per `specs/31-z-search-interface.md`: a pure-numpy search core, a
jax evaluator on the existing agent API, and the search-first G29-3 battery
harness — eval-only, no training changes.

**Architecture:** `scripts/rescuing_fb/z_search.py` holds a numpy search core
(sphere sampling, min-of-softmax-heads weights, CEM over cached candidate
weight tables) plus a thin jax weights-evaluator factory (lazy jax import).
`scripts/rescuing_fb/eval_spec30_g293.py`'s oracle stage is refactored to call
the core (numerics parity by shared code, not by test). A new
`scripts/rescuing_fb/eval_spec31_zsearch.py` runs gate → families → sweep →
utility → corr on the `ra2_cs_centdual` checkpoint.

**Tech Stack:** numpy (core + tests, torch venv), jax (harness only, GPU venv),
ogbench envs, existing agent API: `agent.sample_actions_with_z(obs, z, seed)`,
`agent.network.select("density_f")(concat([obs, a, z], -1))` → `[2, B, d]`,
`agent.network.select("density_b")(u_sc)` → `[M, d]`,
`agent.density_z_from_reward_samples(u_sc, r_rows)` → ẑ,
`agent.network.select('density_zpool')()` → (pool_raw [256, obs], filled).

## Global Constraints

- EVAL-ONLY: no edits to `train.py`, `agents/`, or any training path.
- z-norm invariant: every candidate z has ‖z‖ = √d (sphere, NOT the soft-FB
  ball); re-project after every CEM refit.
- Actor/eval weight convention: per-head softmax over the pool then
  elementwise MIN over the 2 heads (sub-normalized; spec 30 Item 4). Affine
  assertions use the renormalized variant explicitly.
- Tests run with `.venv/bin/python -m pytest` (torch venv — z_search.py must
  import WITHOUT jax at module level; jax only inside factory functions).
- Harness runs: `/dev/shm/.venv-jax/bin/python` with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MUJOCO_GL=egl`, one GPU via
  `CUDA_VISIBLE_DEVICES` (box is shared), ALWAYS inside tmux.
- Commits: conventional style `feat(spec31): …`, NO Claude co-author line.
- `PAPER/ analysis/ results/ datasets*/` stay untracked; commit only
  code/tests.
- Checkpoint under test: `~/exp_spec30_p1/factored-fb/GCIQL-CubeSingle-*-RA2-CentDual-d50/ra2_cs_centdual`,
  step 500000. Gate reference .816 ± .06 (rollout tolerance, binomial noise).

## Deviation from spec 31 §4 (declared)

Spec test 5 asks for "refactored J == the inlined g293 J on a pinned random
ckpt/batch". The torch test venv has no jax, so a ckpt-level parity test
cannot run under `.venv/bin/python -m pytest`. Replacement with the same
force: (a) `test_min_head_weights_matches_manual_softmax` pins the core
against an independent inline computation, and (b) Task 4 makes the g293
oracle CALL the shared core, so spec-31/g293 parity holds by shared code
rather than by test. Residual numerics note: the oracle's softmax/min moves
from jax float32 to numpy float64 — the oracle stage has never produced a
recorded number, so nothing shifts.
- g293 house numbers: GAMMA=0.99, SUCC_R=0.4 (obs units), temp from
  `acfg["measure_density_temp"]`, hit-spec block `19:22` = cube xyz.

## File Structure

- Create `scripts/rescuing_fb/z_search.py` — search core + jax evaluator
  factory. No `main()`; library only.
- Create `tests/test_z_search.py` — pure numpy tests (no jax, no ckpt).
- Modify `scripts/rescuing_fb/eval_spec30_g293.py` — (a) lift `fam_defs` to
  module level, (b) oracle stage calls `z_search` core. File is currently
  UNTRACKED; it gets committed here as part of the refactor task.
- Create `scripts/rescuing_fb/eval_spec31_zsearch.py` — battery harness,
  stages `{gate, families, sweep, utility, corr, all}`, `--quick`.

---

### Task 1: z_search core — sphere ops + min-head weights

**Files:**
- Create: `scripts/rescuing_fb/z_search.py`
- Test: `tests/test_z_search.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces:
  - `sphere_project(z: np.ndarray) -> np.ndarray` — scale last axis to
    ‖·‖=√d, d inferred from `z.shape[-1]`.
  - `sphere_samples(rng: np.random.Generator, n: int, d: int) -> np.ndarray`
    — `[n, d]`, each row norm √d.
  - `min_head_weights(f_emb: np.ndarray, b: np.ndarray, temp: float,
    renorm: bool = False) -> np.ndarray` — `f_emb [H, S, d]`, `b [M, d]` →
    `[S, M]`: per-head stabilized softmax over M, elementwise min over H;
    if `renorm`, rows divided by their sum (back on the simplex).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_z_search.py
"""Spec 31 z-search core tests — pure numpy, torch venv, no jax/ckpt."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

from z_search import min_head_weights, sphere_project, sphere_samples  # noqa: E402


def test_sphere_samples_norm_sqrt_d():
    rng = np.random.default_rng(0)
    z = sphere_samples(rng, 100, 50)
    assert z.shape == (100, 50)
    np.testing.assert_allclose(np.linalg.norm(z, axis=-1),
                               np.sqrt(50.0), rtol=1e-6)


def test_sphere_project_idempotent_and_norm():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((7, 16)) * 3.0
    p = sphere_project(z)
    np.testing.assert_allclose(np.linalg.norm(p, axis=-1),
                               np.sqrt(16.0), rtol=1e-6)
    np.testing.assert_allclose(sphere_project(p), p, rtol=1e-6)


def test_min_head_weights_matches_manual_softmax():
    rng = np.random.default_rng(2)
    f = rng.standard_normal((2, 5, 8))
    b = rng.standard_normal((30, 8))
    temp = 0.7
    w = min_head_weights(f, b, temp)
    # manual twin (the g293 oracle inline computation)
    logits = np.einsum("hbd,md->hbm", f, b) / temp
    e = np.exp(logits - logits.max(-1, keepdims=True))
    sm = e / e.sum(-1, keepdims=True)
    np.testing.assert_allclose(w, sm.min(0), rtol=1e-10)
    # min of simplex rows is SUB-normalized when heads disagree
    assert (w.sum(-1) <= 1.0 + 1e-9).all()
    # renorm puts rows back on the simplex
    wr = min_head_weights(f, b, temp, renorm=True)
    np.testing.assert_allclose(wr.sum(-1), 1.0, rtol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_z_search.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'z_search'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/rescuing_fb/z_search.py
"""Spec 31 — zero-order z-search core for Density-FB.

Pure numpy at module level (importable from the torch test venv); jax is
imported lazily inside make_weights_fn only. The min-of-softmax-heads
weight computation here is THE shared implementation: the g293 oracle and
the spec-31 harness both call it, so their numerics agree by construction.
"""
from __future__ import annotations

import numpy as np


def sphere_project(z: np.ndarray) -> np.ndarray:
    """Scale rows to the standing ||z|| = sqrt(d) invariant (d = last axis)."""
    z = np.asarray(z, np.float64)
    d = z.shape[-1]
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return np.sqrt(d) * z / np.maximum(n, 1e-12)


def sphere_samples(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    return sphere_project(rng.standard_normal((n, d)))


def min_head_weights(f_emb: np.ndarray, b: np.ndarray, temp: float,
                     renorm: bool = False) -> np.ndarray:
    """f_emb [H, S, d], b [M, d] -> [S, M]: per-head softmax over M, then
    elementwise MIN over heads (the actor/eval convention — sub-normalized
    whenever heads disagree, spec 30 Item 4). renorm=True re-normalizes rows
    (the convention the affine-invariance assertion is exact under)."""
    logits = np.einsum("hbd,md->hbm", np.asarray(f_emb, np.float64),
                       np.asarray(b, np.float64)) / temp
    logits -= logits.max(-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(-1, keepdims=True)
    w = w.min(0)
    if renorm:
        w = w / w.sum(-1, keepdims=True)
    return w
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_z_search.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/z_search.py tests/test_z_search.py
git commit -m "feat(spec31): z-search core — sphere ops + min-head pool weights"
```

---

### Task 2: CEM search over cached candidate weight tables

**Files:**
- Modify: `scripts/rescuing_fb/z_search.py`
- Test: `tests/test_z_search.py`

**Interfaces:**
- Consumes: `sphere_project`, `sphere_samples` (Task 1).
- Produces:
  - `cem_search(score_fn, weights_of, d, rng, seed_z=None, n_init=512,
    n_round=256, rounds=3, elite_frac=0.1) -> tuple[np.ndarray, dict]`
    - `weights_of(Z [K, d]) -> W [K, M]` — per-candidate mean pool weights
      (M arbitrary; in synthetic tests W may just be Z).
    - `score_fn(W [K, M]) -> J [K]`.
    - Returns `(z_star [d], diag)`;
      `diag = {"J_seed": float | None, "J_star": float,
      "cos_seed_star": float | None, "round_max": [float]*(rounds+1),
      "round_median": [float]*(rounds+1), "Z0": [K0,d], "W0": [K0,M],
      "J0": [K0], "n_evaluated": int}` — Z0/W0/J0 are the ROUND-0 table
      (seed first when given); z_star is argmax over ALL rounds.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_z_search.py`:

```python
from z_search import cem_search  # noqa: E402


def _planted(d=16, seed=3):
    rng = np.random.default_rng(seed)
    z_true = sphere_samples(rng, 1, d)[0]
    weights_of = lambda Z: np.asarray(Z, np.float64)      # W == Z
    score_fn = lambda W: W @ z_true                       # max on sphere at z_true
    return z_true, weights_of, score_fn, rng


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_cem_recovers_planted_argmax():
    z_true, weights_of, score_fn, rng = _planted()
    z_star, diag = cem_search(score_fn, weights_of, 16, rng,
                              n_init=256, n_round=128, rounds=3)
    assert _cos(z_star, z_true) > 0.9
    assert diag["round_max"][-1] >= diag["round_max"][0]
    assert diag["J_star"] == max(diag["round_max"])


def test_seed_z_is_round0_candidate_zero():
    z_true, weights_of, score_fn, rng = _planted(seed=4)
    seed_z = sphere_samples(rng, 1, 16)[0]
    z_star, diag = cem_search(score_fn, weights_of, 16, rng, seed_z=seed_z,
                              n_init=64, n_round=32, rounds=1)
    np.testing.assert_allclose(diag["Z0"][0], seed_z, rtol=1e-12)
    assert diag["J_seed"] == diag["J0"][0]
    assert diag["cos_seed_star"] is not None
    # z* can never score below the seed (argmax over all evaluated)
    assert diag["J_star"] >= diag["J_seed"]


def test_all_rounds_stay_on_sphere():
    d = 16
    z_true, weights_of, _, rng = _planted(seed=5)
    seen = []
    def spy_weights(Z):
        seen.append(np.asarray(Z))
        return np.asarray(Z, np.float64)
    score_fn = lambda W: W @ z_true
    cem_search(score_fn, spy_weights, d, rng, n_init=64, n_round=32, rounds=2)
    assert len(seen) == 3                                 # round 0 + 2 refits
    for Z in seen:
        np.testing.assert_allclose(np.linalg.norm(Z, axis=-1),
                                   np.sqrt(float(d)), rtol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_z_search.py -v`
Expected: 3 new FAIL with `ImportError: cannot import name 'cem_search'`
(the Task-1 tests still pass).

- [ ] **Step 3: Write the implementation**

Append to `scripts/rescuing_fb/z_search.py` (`weights_of` is called exactly
`rounds + 1` times — the spy test asserts 3 calls for rounds=2; the refit is
skipped on the last iteration so every sampled candidate gets evaluated):

```python
def cem_search(score_fn, weights_of, d: int, rng: np.random.Generator,
               seed_z: np.ndarray | None = None, n_init: int = 512,
               n_round: int = 256, rounds: int = 3,
               elite_frac: float = 0.1):
    """Encoder-seeded CEM on the sqrt(d)-sphere (spec 31 §1).

    Round 0 = [seed_z?] + iid sphere samples; each refinement round refits a
    diagonal Gaussian on the elites of ALL candidates seen so far, resamples,
    and re-projects to the sphere. z_star = argmax J over every candidate
    evaluated. The full round-0 (Z, W, J) table is returned for the affine
    cells and corr(J, return) — W does not depend on the reward, so J under
    any reward transform is re-derivable from W0 without re-evaluating."""
    Z = sphere_samples(rng, n_init if seed_z is None else n_init - 1, d)
    if seed_z is not None:
        Z = np.concatenate([sphere_project(seed_z)[None], Z], 0)
    Z_all, W_all, J_all = [], [], []
    round_max, round_median = [], []
    for r in range(rounds + 1):
        W = np.asarray(weights_of(Z))
        J = np.asarray(score_fn(W), np.float64)
        Z_all.append(Z); W_all.append(W); J_all.append(J)
        round_max.append(float(J.max()))
        round_median.append(float(np.median(J)))
        if r == rounds:
            break
        Zc = np.concatenate(Z_all, 0)
        Jc = np.concatenate(J_all, 0)
        n_elite = max(2, int(np.ceil(elite_frac * len(Jc))))
        elites = Zc[np.argsort(Jc)[-n_elite:]]
        mu, sd = elites.mean(0), elites.std(0) + 1e-6
        Z = sphere_project(mu[None] + sd[None] * rng.standard_normal((n_round, d)))
    Zc = np.concatenate(Z_all, 0)
    Jc = np.concatenate(J_all, 0)
    k = int(np.argmax(Jc))
    z_star = Zc[k]
    j_seed = float(J_all[0][0]) if seed_z is not None else None
    cos_ss = (float(z_star @ Z_all[0][0] /
                    (np.linalg.norm(z_star) * np.linalg.norm(Z_all[0][0]) + 1e-12))
              if seed_z is not None else None)
    diag = {"J_seed": j_seed, "J_star": float(Jc[k]), "cos_seed_star": cos_ss,
            "round_max": round_max, "round_median": round_median,
            "Z0": Z_all[0], "W0": W_all[0], "J0": J_all[0],
            "n_evaluated": int(len(Jc))}
    return z_star, diag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_z_search.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/z_search.py tests/test_z_search.py
git commit -m "feat(spec31): encoder-seeded CEM on the sqrt(d)-sphere with round-0 table"
```

---

### Task 3: Affine-invariance check + jax weights-evaluator factory

**Files:**
- Modify: `scripts/rescuing_fb/z_search.py`
- Test: `tests/test_z_search.py`

**Interfaces:**
- Consumes: `min_head_weights`, `cem_search` diag `W0`/`J0` (Tasks 1–2).
- Produces:
  - `affine_argmax_check(W0: np.ndarray, r: np.ndarray,
    transforms: list[tuple[float, float]]) -> dict` — for each `(alpha,
    beta)` recompute `J = W @ (alpha * r + beta)` on the FIXED candidate
    table; returns
    `{"argmax_base": int, "renorm": {"(a,b)": {"argmax": int, "match":
    bool}}, "min_convention": {"(a,b)": {"argmax": int, "match": bool}},
    "renorm_all_match": bool, "min_all_match": bool}` where the `renorm`
    block uses row-renormalized W (invariance EXACT for alpha > 0 — spec 31
    §3) and `min_convention` uses W as-is (near-exact; reported, not
    asserted).
  - `make_weights_fn(agent, b_sub, temp, init_obs, seed) -> weights_of` —
    jax-backed; `weights_of(Z [K, d]) -> W [K, M]` where
    `W[k] = mean over init states s0 of min_head_weights(F(s0, pi(s0, z_k),
    z_k), b_sub, temp)`. jax imported INSIDE the function.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_z_search.py`:

```python
from z_search import affine_argmax_check  # noqa: E402


def test_affine_argmax_exact_under_renorm():
    rng = np.random.default_rng(6)
    K, M = 40, 25
    # sub-normalized rows with VARYING row sums (the min-convention regime)
    W = rng.random((K, M))
    W /= W.sum(-1, keepdims=True)
    W *= rng.uniform(0.5, 1.0, size=(K, 1))
    r = rng.standard_normal(M)
    out = affine_argmax_check(W, r, [(2.0, 0.0), (1.0, 3.0), (0.5, -7.0)])
    # renormalized: argmax_z (W@(a r + b)) == argmax_z (W@r) EXACTLY, a > 0
    assert out["renorm_all_match"] is True
    for cell in out["renorm"].values():
        assert cell["match"] is True
    # min convention: reported (bool present), not asserted true
    assert set(out["min_convention"]) == set(out["renorm"])
    assert isinstance(out["min_all_match"], bool)


def test_affine_argmax_base_is_plain_reward_argmax():
    rng = np.random.default_rng(7)
    W = rng.random((10, 6))
    r = rng.standard_normal(6)
    out = affine_argmax_check(W, r, [(1.0, 1.0)])
    Wn = W / W.sum(-1, keepdims=True)
    assert out["argmax_base"] == int(np.argmax(Wn @ r))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_z_search.py -v`
Expected: 2 new FAIL with `ImportError: cannot import name
'affine_argmax_check'`.

- [ ] **Step 3: Write the implementation**

Append to `scripts/rescuing_fb/z_search.py`:

```python
def affine_argmax_check(W0: np.ndarray, r: np.ndarray, transforms):
    """Spec 31 §3 affine cell. W does not depend on r, so J under alpha*r+beta
    is a re-read of the cached table. Renormalized rows: argmax invariance is
    EXACT for alpha > 0 (beta adds a constant). Min-convention rows are
    sub-normalized (beta couples to ||w||_1): reported, never asserted."""
    W0 = np.asarray(W0, np.float64)
    r = np.asarray(r, np.float64)
    Wn = W0 / W0.sum(-1, keepdims=True)
    out = {"argmax_base": int(np.argmax(Wn @ r)), "renorm": {},
           "min_convention": {}}
    base_min = int(np.argmax(W0 @ r))
    for a, b in transforms:
        rt = a * r + b
        kr = int(np.argmax(Wn @ rt))
        km = int(np.argmax(W0 @ rt))
        out["renorm"][f"({a},{b})"] = {"argmax": kr,
                                       "match": kr == out["argmax_base"]}
        out["min_convention"][f"({a},{b})"] = {"argmax": km,
                                               "match": km == base_min}
    out["renorm_all_match"] = all(c["match"] for c in out["renorm"].values())
    out["min_all_match"] = all(c["match"] for c in out["min_convention"].values())
    return out


def make_weights_fn(agent, b_sub: np.ndarray, temp: float,
                    init_obs: np.ndarray, seed):
    """weights_of(Z [K,d]) -> W [K,M] for cem_search, through the agent's own
    actor + F towers (the g293 oracle computation, one candidate at a time).
    jax imported lazily so this module stays importable from the torch venv.
    `seed` is a jax PRNGKey; it is FIXED across candidates so J differences
    come from z, not action noise."""
    import jax.numpy as jnp
    b64 = np.asarray(b_sub, np.float64)
    S = len(init_obs)
    obs_j = jnp.asarray(np.asarray(init_obs, np.float32))

    def weights_of(Z):
        W = np.zeros((len(Z), len(b64)))
        for k, z in enumerate(np.asarray(Z, np.float32)):
            zb = jnp.asarray(np.tile(z[None], (S, 1)))
            a = agent.sample_actions_with_z(obs_j, zb, seed=seed)
            f = agent.network.select("density_f")(
                jnp.concatenate([obs_j, a, zb], -1))
            W[k] = min_head_weights(np.asarray(f), b64, temp).mean(0)
        return W

    return weights_of
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_z_search.py -v`
Expected: 8 passed (jax never imported — `make_weights_fn` body untouched).

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/z_search.py tests/test_z_search.py
git commit -m "feat(spec31): affine-invariance table check + jax weights-evaluator factory"
```

---

### Task 4: Refactor the g293 harness onto the shared core

**Files:**
- Modify: `scripts/rescuing_fb/eval_spec30_g293.py` (currently UNTRACKED —
  this task commits it; flag in the commit body that it predates spec 31)

**Interfaces:**
- Consumes: `min_head_weights`, `make_weights_fn` (z_search).
- Produces (for Task 5's imports):
  - `fam_defs(g1, g2, gt, ga, med_da)` at MODULE level in
    `eval_spec30_g293.py` — same dict-of-families as the current closure
    (keys `dense_g1, hard_g1, region_g1, maxmix, summix, tma05, tma10,
    negavoid`; each `dict(task=…, r=…, succ=…, hit=…)`).
  - Already-module-level helpers stay put and are reused by Task 5:
    `kernel_raw, anytime_ball, min_dist, disc_return, centered_ridge_fit,
    recon_metrics, cos, GAMMA, SUCC_R, AVOID_R, M_SWEEP, GATE_TOL,
    G1_TASK, G2_TASK, GT_TASK, GA_TASK`.

- [ ] **Step 1: Lift `fam_defs` to module level**

In `eval_spec30_g293.py`, move the `def fam_defs():` closure (currently
inside `main()`, after the `g1, g2, gt, ga` / `med_da` lines) to module
level with the signature `def fam_defs(g1, g2, gt, ga, med_da):` — body
unchanged except the closure variables become parameters. In `main()`,
replace the definition with:

```python
    fams = fam_defs(g1, g2, gt, ga, med_da)
```

- [ ] **Step 2: Replace the oracle stage's inline J with the shared core**

In the oracle stage (`if args.stage in ("oracle", "all") and not
args.quick:`), replace the inner `def J(zbatch):` block:

```python
        from z_search import make_weights_fn

        # (inside the per-family loop, after init_obs is built)
            weights_of = make_weights_fn(agent, b_sub, temp, init_obs,
                                         seed=jax.random.PRNGKey(11))

            def J(zbatch):
                return weights_of(np.asarray(zbatch)) @ r_sub
```

Everything downstream (`j_cf`, `j_all`, `k_best`, rollout, json) is
unchanged. Numerics note for the commit body: softmax/min moves from jax
float32 to numpy float64 — the oracle has never produced a number, so no
recorded artifact shifts.

- [ ] **Step 3: Smoke the refactor (imports + flag parsing only)**

Run:
```bash
/dev/shm/.venv-jax/bin/python - <<'EOF'
import sys
sys.path.insert(0, "scripts/rescuing_fb")
import eval_spec30_g293 as m
fams = m.fam_defs(*(__import__("numpy").zeros(3) for _ in range(4)), 1.0)
assert set(fams) == {"dense_g1", "hard_g1", "region_g1", "maxmix", "summix",
                     "tma05", "tma10", "negavoid"}
print("fam_defs OK:", sorted(fams))
EOF
```
Expected: `fam_defs OK: [...]` with the 8 family names. (If
`/dev/shm/.venv-jax` is missing — it is volatile — recreate it per the
`jax-venv-location` memory note before this step; numpy-only, so
`.venv/bin/python` also works for THIS smoke.)

- [ ] **Step 4: Run the full test suite to confirm nothing else broke**

Run: `.venv/bin/python -m pytest tests/test_z_search.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/eval_spec30_g293.py
git commit -m "refactor(spec31): g293 — fam_defs to module level, oracle J via z_search core

eval_spec30_g293.py predates spec 31 (written for spec 30 P3, previously
uncommitted); committed here so spec 31 can import its family battery.
Oracle softmax/min now goes through z_search.min_head_weights (numpy
float64) — the oracle stage has never produced a recorded number, so no
artifact shifts."
```

---

### Task 5: Spec-31 harness — loader, gate, families stages

**Files:**
- Create: `scripts/rescuing_fb/eval_spec31_zsearch.py`

**Interfaces:**
- Consumes: `z_search.{cem_search, make_weights_fn, affine_argmax_check,
  sphere_project}`; `eval_spec30_g293.{fam_defs, kernel_raw, anytime_ball,
  min_dist, disc_return, cos, GAMMA, SUCC_R, AVOID_R, GATE_TOL, G1_TASK,
  G2_TASK, GT_TASK, GA_TASK}`; g293's loader recipe (load_agent_dims,
  build_pool, batched_rollout_record, eval_tasks_for).
- Produces: `results/spec31_zsearch/{gate.json, fam_<name>.json,
  families_summary.json}`; in-process `summary` dict consumed by Task 6's
  stages within the same file.

- [ ] **Step 1: Write the harness skeleton — CLI, loader, pool, rollout**

Create `scripts/rescuing_fb/eval_spec31_zsearch.py`. The loader block is the
g293 recipe verbatim (same sys.path dance, same audited builders):

```python
#!/usr/bin/env python
"""scripts/rescuing_fb/eval_spec31_zsearch.py -- Spec 31: zero-order z-search
as the task-inference interface (search-first G29-3), EVAL-ONLY on the
centered-dual checkpoint.

Stages: gate (native re-roll ONLY -- the harness-pool encoder cell of g293 is
retired as a gate and the search path never encodes), families (per family:
encoder zhat vs encoder-seeded CEM z*, rollouts + affine cells), sweep
(pool-size robustness OF THE SEARCH), utility (KL distribution-matching
showcase), corr (J vs rollout return over round-0 candidates).

Run (tmux, one GPU, box shared):
  env CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec31_zsearch.py \\
    --run-dir ~/exp_spec30_p1/factored-fb/GCIQL-CubeSingle-*-RA2-CentDual-d50/ra2_cs_centdual \\
    --checkpoint-step 500000 --stage all --out-dir results/spec31_zsearch
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

from eval_spec30_g293 import (AVOID_R, G1_TASK, G2_TASK, GA_TASK, GATE_TOL,  # noqa: E402
                              GT_TASK, SUCC_R, anytime_ball, cos, disc_return,
                              fam_defs, kernel_raw, min_dist)
from z_search import (affine_argmax_check, cem_search, make_weights_fn,  # noqa: E402
                      sphere_project)

GATE_REF = 0.816
P_SUB = 2048                    # J integration pool (spec 31 §2)
SWEEP_M = (256, 1024, 8192)     # search-path pool robustness (spec 31 §2)
N_INIT_STATES = 16
CEM = dict(n_init=512, n_round=256, rounds=3, elite_frac=0.1)
AFFINE_TRANSFORMS = [(2.0, 0.0), (0.1, 0.0), (1.0, 1.0), (1.0, -5.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=16384)
    ap.add_argument("--stage", default="all",
                    choices=["gate", "families", "sweep", "utility", "corr", "all"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 4 eps, 2 families, CEM 64/32x1")
    ap.add_argument("--out-dir", default="results/spec31_zsearch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t00 = time.time()
    run_dir = Path(sorted(glob.glob(str(Path(args.run_dir).expanduser())))[0])
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "flags.json").read_text())
    env_name = cfg["env_name"]
    acfg = cfg["agent"]
    assert acfg.get("measure_density_z_encode") == "dual_centered", \
        "spec 31 runs on the centered-dual arm"
    n_eps = 4 if args.quick else args.episodes
    cem_kw = dict(n_init=64, n_round=32, rounds=1) if args.quick else dict(CEM)

    from eval_z_action_ranking import hit_spec_for, load_dataset  # noqa: E402
    from eval_density_zeroshot import build_pool  # noqa: E402
    from eval_zeroshot_rewards import batched_rollout_record, eval_tasks_for  # noqa: E402
    from t3_offpeak_multienv import load_agent_dims  # noqa: E402

    spec = hit_spec_for(env_name, acfg.get("measure_hit_spec"))
    blocks = spec.blocks
    (b0, b1, _e) = blocks[0]
    xyz = slice(b0, b1)

    ds = load_dataset(env_name)
    obs_all = np.asarray(ds["observations"], np.float32)
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step,
                            obs_all.shape[-1], np.asarray(ds["actions"]).shape[-1])
    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    d = int(acfg["measure_density_dim"])
    temp = float(acfg.get("measure_density_temp", 1.0))
    rng = np.random.default_rng(args.seed)

    # big pool (audited builder, g293 rng order)
    pool_idx = rng.choice(len(obs_all), size=min(args.pool_size, len(obs_all)),
                          replace=False)
    pool_obs = obs_all[pool_idx]
    u_raw, _b_online = build_pool(agent, blocks, pool_obs, spec)
    u_sc = np.asarray(GCIQLFlowAgent._density_queries(jnp.asarray(pool_obs), blocks))
    P = len(u_raw)

    # ckpt's OWN frozen eval pool (the ONLY encoding pool -- G29-3 diagnosis)
    pool_raw_frozen, _filled = (np.asarray(x) for x in
                                agent.network.select('density_zpool')())
    assert float(np.asarray(_filled)) == 1.0, "frozen eval pool never filled"
    u_sc_frozen = np.asarray(GCIQLFlowAgent._scale_block_cols(
        jnp.asarray(pool_raw_frozen), blocks))

    def encode_frozen(r_vals):
        z = np.asarray(agent.density_z_from_reward_samples(
            jnp.asarray(u_sc_frozen),
            jnp.asarray(np.atleast_2d(r_vals), jnp.float32)))
        return z[0]

    import ogbench  # noqa: E402
    eval_tasks = eval_tasks_for(env_name)

    def make_envs(task_id, n):
        return [ogbench.make_env_and_datasets(eval_tasks[task_id], env_only=True)
                for _ in range(n)]

    def task_goal_obs(task_id):
        e = ogbench.make_env_and_datasets(eval_tasks[task_id], env_only=True)
        _o, info = e.reset(seed=0)
        g = np.asarray(info["goal"], np.float32)
        e.close()
        return g

    goals = {i: task_goal_obs(i) for i in range(1, 6)}
    _e0 = ogbench.make_env_and_datasets(eval_tasks[1], env_only=True)
    H = _e0.spec.max_episode_steps if _e0.spec and _e0.spec.max_episode_steps else 200
    _e0.close()

    def rollout(task_id, cond, mode, n=None):
        envs = make_envs(task_id, n or n_eps)
        traj, obs_traj, lengths, succ_env = batched_rollout_record(
            agent, envs, cond, H=H, mode=mode)
        for e in envs:
            e.close()
        cube = obs_traj[:, :, xyz].astype(np.float64)
        for i in range(len(cube)):
            L = max(int(lengths[i]), 1)
            cube[i, L:] = cube[i, L - 1]
        return cube, lengths, succ_env

    # J evaluation state set: env-reset states (reset only, no stepping)
    def reset_states(task_id, n=N_INIT_STATES):
        e = ogbench.make_env_and_datasets(eval_tasks[task_id], env_only=True)
        out = []
        for k in range(n):
            o, _ = e.reset(seed=1000 + k)
            out.append(o)
        e.close()
        return np.asarray(out, np.float32)

    # J integration pool: P_SUB subsample of the dataset marginal
    sub = rng.choice(P, P_SUB, replace=False)
    b_sub = np.asarray(agent.network.select("density_b")(jnp.asarray(u_sc[sub])))
    print(f"[s31] {run_dir.name} step={args.checkpoint_step} "
          f"P={P} d={d} temp={temp} ({time.time()-t00:.0f}s)", flush=True)
    # ... stages appended below ...
```

- [ ] **Step 2: Add the gate stage (native re-roll ONLY)**

Append inside `main()`:

```python
    # =================================================================== gate
    if args.stage in ("gate", "all"):
        nat, harness = {}, {}
        for t in range(1, 6):
            cube, lengths, succ_env = rollout(t, goals[t], mode="goal")
            nat[t] = {"succ_env": float(succ_env.mean())}
            kg_f = np.asarray(GCIQLFlowAgent._delta_phi(
                jnp.asarray(goals[t][None]), jnp.asarray(pool_raw_frozen), blocks))[0]
            z_f = encode_frozen(kg_f)
            cube, lengths, succ_env = rollout(t, z_f, mode="z")
            harness[t] = {"succ_env": float(succ_env.mean())}
        oa = float(np.mean([nat[t]["succ_env"] for t in nat]))
        ob = float(np.mean([harness[t]["succ_env"] for t in harness]))
        gate = {"native": nat, "overall_native": oa,
                "frozen_pool_z_api": harness, "overall_z_api": ob,
                "reference": GATE_REF, "tol": GATE_TOL,
                "pass": bool(abs(oa - GATE_REF) <= GATE_TOL
                             and abs(ob - GATE_REF) <= GATE_TOL)}
        (out_dir / "gate.json").write_text(json.dumps(gate, indent=1))
        print(f"[s31] GATE native={oa:.3f} z_api={ob:.3f} ref={GATE_REF} "
              f"pass={gate['pass']} ({time.time()-t00:.0f}s)", flush=True)
        if not gate["pass"] and not args.quick:
            print("[s31] GATE FAILED -- stopping, no new cells read", flush=True)
            return
    if args.stage == "gate":
        return
```

- [ ] **Step 3: Add the families stage (encoder vs search, affine cells)**

Append inside `main()`:

```python
    # =============================================================== families
    g1 = goals[G1_TASK][xyz]; g2 = goals[G2_TASK][xyz]
    gt = goals[GT_TASK][xyz]; ga = goals[GA_TASK][xyz]
    med_da = float(np.median(np.linalg.norm(u_raw - ga[None], axis=-1)))
    fams = fam_defs(g1, g2, gt, ga, med_da)
    if args.quick:
        fams = {k: fams[k] for k in ("dense_g1", "tma10")}

    summary = {}
    if args.stage in ("families", "all"):
        for name, fd in fams.items():
            r_sub = fd["r"](u_raw[sub])
            init_obs = reset_states(fd["task"])
            weights_of = make_weights_fn(agent, b_sub, temp, init_obs,
                                         seed=jax.random.PRNGKey(11))
            z_enc = encode_frozen(fd["r"](pool_raw_frozen))
            z_star, diag = cem_search(lambda W: W @ r_sub, weights_of, d,
                                      np.random.default_rng(args.seed + 100),
                                      seed_z=z_enc, **cem_kw)
            aff = affine_argmax_check(diag["W0"], r_sub, AFFINE_TRANSFORMS)
            # spec 31 §3: renorm invariance is a THEOREM-level property --
            # a violation means a harness bug, hard-abort (house gate rule);
            # the min-convention result is reported, never asserted.
            assert aff["renorm_all_match"], \
                f"{name}: renormalized affine argmax invariance violated -- " \
                f"harness bug, not a finding: {aff['renorm']}"

            cube, lengths, _ = rollout(fd["task"], z_enc, mode="z")
            roll_enc = fd["succ"](cube, lengths)
            roll_enc["disc_return"] = disc_return(cube, lengths, fd["r"])
            cube, lengths, _ = rollout(fd["task"], z_star, mode="z")
            roll_search = fd["succ"](cube, lengths)
            roll_search["disc_return"] = disc_return(cube, lengths, fd["r"])

            rec = {"rollout_encoder": roll_enc, "rollout_search": roll_search,
                   "J_seed": diag["J_seed"], "J_star": diag["J_star"],
                   "cos_seed_star": diag["cos_seed_star"],
                   "round_max": diag["round_max"],
                   "round_median": diag["round_median"],
                   "affine": aff,
                   "n_evaluated": diag["n_evaluated"],
                   "z_enc": z_enc.tolist(), "z_star": z_star.tolist(),
                   "episodes": n_eps, "task_reset": fd["task"]}
            (out_dir / f"fam_{name}.json").write_text(
                json.dumps(rec, indent=1, default=str))
            summary[name] = {**rec,
                             "_W0": diag["W0"], "_J0": diag["J0"],
                             "_Z0": diag["Z0"], "_r_sub": r_sub,
                             "_init_obs": init_obs}
            print(f"[s31] {name}: enc_succ={roll_enc['succ']:.2f} "
                  f"search_succ={roll_search['succ']:.2f} "
                  f"Jseed={diag['J_seed']:.4f} J*={diag['J_star']:.4f} "
                  f"cos={diag['cos_seed_star']:.3f} "
                  f"aff_renorm={aff['renorm_all_match']} "
                  f"aff_min={aff['min_all_match']} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        pub = {n: {k: v for k, v in summary[n].items() if not k.startswith("_")}
               for n in summary}
        (out_dir / "families_summary.json").write_text(
            json.dumps(pub, indent=1, default=str))
```

- [ ] **Step 4: Smoke the file compiles and the CLI parses**

Run:
```bash
/dev/shm/.venv-jax/bin/python -c "
import ast, pathlib
ast.parse(pathlib.Path('scripts/rescuing_fb/eval_spec31_zsearch.py').read_text())
print('parse OK')"
/dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec31_zsearch.py --help
```
Expected: `parse OK`, then the argparse help listing all six stages.

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/eval_spec31_zsearch.py
git commit -m "feat(spec31): zsearch harness -- loader, native gate, families stage (encoder vs CEM)"
```

---

### Task 6: Harness — sweep, utility, corr stages

**Files:**
- Modify: `scripts/rescuing_fb/eval_spec31_zsearch.py`

**Interfaces:**
- Consumes: `summary` dict (with `_W0/_J0/_Z0/_r_sub/_init_obs` per family),
  `b_sub`/`sub`/`u_raw`/`u_sc`, `cem_search`, `make_weights_fn`, `rollout`,
  `disc_return`, `sphere_project` — all in-file from Task 5.
- Produces: `results/spec31_zsearch/{sweep.json, utility.json, corr.json,
  summary.json}`.

- [ ] **Step 1: Add the sweep stage (search-path pool robustness)**

Append inside `main()` after the families stage. Same candidate RNG per M so
z* differences are attributable to the pool, not sampling:

```python
    # ================================================================== sweep
    if args.stage in ("sweep", "all") and not args.quick:
        sweep_fams = ("dense_g1", "tma10")
        sweep = {}
        for name in sweep_fams:
            fd = fams[name]
            init_obs = (summary[name]["_init_obs"] if name in summary
                        else reset_states(fd["task"]))
            z_enc = encode_frozen(fd["r"](pool_raw_frozen))
            cells = {}
            zstars = {}
            for m in SWEEP_M:
                idx_m = rng.choice(P, m, replace=False)
                b_m = np.asarray(agent.network.select("density_b")(
                    jnp.asarray(u_sc[idx_m])))
                r_m = fd["r"](u_raw[idx_m])
                weights_of = make_weights_fn(agent, b_m, temp, init_obs,
                                             seed=jax.random.PRNGKey(11))
                z_star, diag = cem_search(lambda W: W @ r_m, weights_of, d,
                                          np.random.default_rng(args.seed + 200),
                                          seed_z=z_enc, **cem_kw)
                zstars[m] = z_star
                cube, lengths, _ = rollout(fd["task"], z_star, mode="z")
                rr = fd["succ"](cube, lengths)
                cells[str(m)] = {"J_star": diag["J_star"], "succ": rr["succ"],
                                 "cos_vs_enc": cos(z_star, z_enc)}
            stab = {f"cos_{SWEEP_M[i]}_{SWEEP_M[i+1]}":
                    cos(zstars[SWEEP_M[i]], zstars[SWEEP_M[i + 1]])
                    for i in range(len(SWEEP_M) - 1)}
            sweep[name] = {"cells": cells, "stability": stab}
            print(f"[s31] sweep {name}: " + " ".join(
                f"M{m}:succ={cells[str(m)]['succ']:.2f}" for m in SWEEP_M)
                + f" stab={stab} ({time.time()-t00:.0f}s)", flush=True)

        # init-state ablation (spec 31 §2): dataset episode-start states
        # instead of env-reset states, one family, expected no-op
        fd = fams["dense_g1"]
        term = np.asarray(ds["terminals"]).astype(bool)
        starts = np.concatenate([[0], np.where(term[:-1])[0] + 1])
        ds_init = obs_all[rng.choice(starts, N_INIT_STATES, replace=False)]
        z_enc = encode_frozen(fd["r"](pool_raw_frozen))
        r_sub_d = fd["r"](u_raw[sub])
        weights_of = make_weights_fn(agent, b_sub, temp, ds_init,
                                     seed=jax.random.PRNGKey(11))
        z_star_ds, diag_ds = cem_search(lambda W: W @ r_sub_d, weights_of, d,
                                        np.random.default_rng(args.seed + 100),
                                        seed_z=z_enc, **cem_kw)
        z_star_env = (np.asarray(summary["dense_g1"]["z_star"])
                      if "dense_g1" in summary else None)
        cube, lengths, _ = rollout(fd["task"], z_star_ds, mode="z")
        sweep["init_state_ablation"] = {
            "family": "dense_g1", "J_star_dataset_s0": diag_ds["J_star"],
            "succ_dataset_s0": fd["succ"](cube, lengths)["succ"],
            "cos_vs_env_reset_zstar": (cos(z_star_ds, z_star_env)
                                       if z_star_env is not None else None)}
        print(f"[s31] init ablation: succ={sweep['init_state_ablation']['succ_dataset_s0']:.2f} "
              f"cos={sweep['init_state_ablation']['cos_vs_env_reset_zstar']}", flush=True)
        (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=1))
```

(The candidate RNG `args.seed + 100` deliberately matches the families
stage's, so the env-reset vs dataset-s₀ comparison holds the round-0
candidate set fixed and isolates the init-state choice.)

- [ ] **Step 2: Add the utility stage (KL distribution matching — the cell
  the encoder cannot express)**

Target: uniform occupancy over the `region_g1` box (half-width 0.6 around
g1), binned 4×4×4, with an outside catch-all bin. Comparison arm = the
encoder z for `dense_g1` (goal-seeking contrast; the encoder has no utility
input type). Append:

```python
    # ================================================================ utility
    if args.stage in ("utility", "all") and not args.quick:
        BINS = 4
        half_w = 0.6
        lo = g1 - half_w
        edges = [np.linspace(lo[i], lo[i] + 2 * half_w, BINS + 1) for i in range(3)]

        def bin_index(u):
            """u [N,3] -> flat bin id in [0, BINS^3] (last id = outside)."""
            ids = np.zeros(len(u), np.int64)
            outside = np.zeros(len(u), bool)
            mul = 1
            for i in range(3):
                k = np.digitize(u[:, i], edges[i]) - 1
                outside |= (k < 0) | (k >= BINS)
                ids += np.clip(k, 0, BINS - 1) * mul
                mul *= BINS
            ids[outside] = BINS ** 3
            return ids

        pool_bins = bin_index(u_raw[sub])
        n_bins = BINS ** 3 + 1
        target = np.full(n_bins, 1e-3)
        target[:BINS ** 3] = 1.0
        target /= target.sum()

        def f_kl(W):
            """W [K, M] -> J [K] = -KL(Pi(w) || target), rows renormalized."""
            Wn = W / W.sum(-1, keepdims=True)
            proj = np.zeros((len(W), n_bins))
            np.add.at(proj.T, pool_bins, Wn.T)          # sum w over bins
            proj = np.clip(proj, 1e-9, None)
            proj /= proj.sum(-1, keepdims=True)
            return -np.sum(proj * np.log(proj / target[None]), -1)

        init_obs = reset_states(G1_TASK)
        weights_of = make_weights_fn(agent, b_sub, temp, init_obs,
                                     seed=jax.random.PRNGKey(11))
        z_star, diag = cem_search(f_kl, weights_of, d,
                                  np.random.default_rng(args.seed + 300),
                                  seed_z=None, **cem_kw)
        z_goal = encode_frozen(fams["dense_g1"]["r"](pool_raw_frozen))

        def occupancy_report(z, tag):
            cube, lengths, _ = rollout(G1_TASK, z, mode="z")
            ids = bin_index(cube.reshape(-1, 3))
            occ = np.bincount(ids, minlength=n_bins).astype(float)
            occ /= occ.sum()
            occ_c = np.clip(occ, 1e-9, None)
            return {"kl_vs_target": float(np.sum(occ_c * np.log(occ_c / target))),
                    "frac_region_bins_visited": float(
                        (occ[:BINS ** 3] > 0).mean()),
                    "frac_time_in_region": float(occ[:BINS ** 3].sum())}

        util = {"J_star": diag["J_star"],
                "J_median_round0": float(np.median(diag["J0"])),
                "round_max": diag["round_max"],
                "search": occupancy_report(z_star, "search"),
                "goal_contrast": occupancy_report(z_goal, "goal"),
                "z_star": z_star.tolist(), "bins": BINS, "half_w": half_w}
        (out_dir / "utility.json").write_text(json.dumps(util, indent=1))
        print(f"[s31] utility: J*={diag['J_star']:.4f} "
              f"search_cov={util['search']['frac_region_bins_visited']:.2f} "
              f"goal_cov={util['goal_contrast']['frac_region_bins_visited']:.2f} "
              f"({time.time()-t00:.0f}s)", flush=True)
```

- [ ] **Step 3: Add the corr stage (model-exploitation guard) + summary**

Rollout a J-quantile-spanning subset of round-0 candidates and correlate J
with the realized discounted return. Append:

```python
    # =================================================================== corr
    if args.stage in ("corr", "all") and not args.quick:
        from scipy.stats import spearmanr
        corr_fams = [n for n in ("dense_g1", "tma10") if n in summary]
        corr = {}
        for name in corr_fams:
            fd = fams[name]
            J0 = summary[name]["_J0"]; Z0 = summary[name]["_Z0"]
            qs = np.unique(np.quantile(np.arange(len(J0)), np.linspace(0, 1, 12))
                           .astype(int))
            order = np.argsort(J0)
            pick = order[qs]
            rets, js = [], []
            for k in pick:
                cube, lengths, _ = rollout(fd["task"], Z0[k], mode="z", n=5)
                rets.append(disc_return(cube, lengths, fd["r"]))
                js.append(float(J0[k]))
            corr[name] = {"spearman": float(spearmanr(js, rets).statistic),
                          "J": js, "returns": rets, "eps_per_candidate": 5}
            print(f"[s31] corr {name}: spearman={corr[name]['spearman']:.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        (out_dir / "corr.json").write_text(json.dumps(corr, indent=1))

    # top-level summary (whatever stages ran)
    top = {"stages_run": args.stage, "quick": bool(args.quick),
           "cem": cem_kw, "p_sub": P_SUB, "n_init_states": N_INIT_STATES,
           "ckpt": str(run_dir), "step": args.checkpoint_step}
    for f in ("gate", "families_summary", "sweep", "utility", "corr"):
        p = out_dir / f"{f}.json"
        if p.exists():
            top[f] = json.loads(p.read_text())
    (out_dir / "summary.json").write_text(json.dumps(top, indent=1, default=str))
    print(f"[s31] DONE ({time.time()-t00:.0f}s)", flush=True)
```

Note: `corr` needs the families stage's in-memory tables, so `--stage corr`
alone must rerun families first — add at the top of the corr block:

```python
        assert summary, "corr stage needs families in the same invocation " \
                        "(use --stage all or families,corr)"
```

- [ ] **Step 4: Syntax + CLI smoke**

Run:
```bash
/dev/shm/.venv-jax/bin/python -c "
import ast, pathlib
ast.parse(pathlib.Path('scripts/rescuing_fb/eval_spec31_zsearch.py').read_text())
print('parse OK')"
```
Expected: `parse OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/rescuing_fb/eval_spec31_zsearch.py
git commit -m "feat(spec31): zsearch harness -- sweep, KL-utility showcase, J-vs-return corr"
```

---

### Task 7: Quick smoke on the real checkpoint, then the full battery

**Files:**
- No new files; runs produce `results/spec31_zsearch/` (untracked).

**Interfaces:**
- Consumes: the complete harness (Tasks 5–6), ckpt
  `~/exp_spec30_p1/factored-fb/GCIQL-CubeSingle-*-RA2-CentDual-d50/ra2_cs_centdual`.
- Produces: smoke + full-battery artifacts; a `results_map.py` group entry
  and refreshed CSVs after collection.

- [ ] **Step 1: Verify the GPU venv exists (it is volatile)**

Run: `ls /dev/shm/.venv-jax/bin/python && nvidia-smi --query-gpu=index,memory.used --format=csv`
Expected: the interpreter exists; pick the least-loaded GPU index for the
next steps. If `/dev/shm/.venv-jax` is gone, recreate it per the
`jax-venv-location` memory note before proceeding.

- [ ] **Step 2: Run the --quick smoke in tmux**

```bash
tmux new-session -d -s s31smoke
tmux send-keys -t s31smoke "cd ~/git/Factored-FB && env CUDA_VISIBLE_DEVICES=<least-loaded> \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec31_zsearch.py \
  --run-dir '~/exp_spec30_p1/factored-fb/GCIQL-CubeSingle-*-RA2-CentDual-d50/ra2_cs_centdual' \
  --checkpoint-step 500000 --stage all --quick \
  --out-dir results/spec31_zsearch_smoke 2>&1 | tee /tmp/s31smoke.log" Enter
```
Poll (do NOT trust watchers — g293 lesson): `tail -5 /tmp/s31smoke.log`
until `[s31] DONE`. Expected: gate numbers within noise of .816 is NOT
required at 4 eps (quick skips the gate abort); families print two lines
with finite `J*`, `cos`, and both `aff_*` booleans; no tracebacks.

- [ ] **Step 3: Fix anything the smoke surfaces, re-run, commit fixes**

Any code change goes through the unit tests first
(`.venv/bin/python -m pytest tests/test_z_search.py -v`), then re-smoke.
Commit as `fix(spec31): <what the smoke caught>`.

- [ ] **Step 4: Launch the full battery in tmux**

```bash
tmux new-session -d -s s31full
tmux send-keys -t s31full "cd ~/git/Factored-FB && env CUDA_VISIBLE_DEVICES=<least-loaded> \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec31_zsearch.py \
  --run-dir '~/exp_spec30_p1/factored-fb/GCIQL-CubeSingle-*-RA2-CentDual-d50/ra2_cs_centdual' \
  --checkpoint-step 500000 --stage all \
  --out-dir results/spec31_zsearch 2>&1 | tee /tmp/s31full.log" Enter
```
Poll `results/spec31_zsearch/` directly. The gate stage aborts the run if
the native re-roll leaves .816 ± .06 — that is the designed stop, report it
rather than working around it.

- [ ] **Step 5: After collection — results_map group + CSV refresh**

Add a `spec31_zsearch` group entry in `scripts/results_map.py` (follow the
existing group-entry pattern in that file), then:

```bash
.venv/bin/python scripts/compile_results_csvs.py --refresh
git add scripts/results_map.py
git commit -m "chore(spec31): results_map group for the z-search battery"
```

Read out `families_summary.json` / `sweep.json` / `utility.json` /
`corr.json` against spec 31 §5's decision rules and report which branch the
evidence lands in. (Interpretation and the DENSITY-FB-EXPLAINED.md section
are follow-on writing work, not part of this plan.)
