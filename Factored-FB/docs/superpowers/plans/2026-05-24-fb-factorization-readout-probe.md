# FB Factorization-Ablation + Readout-Ceiling Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic probe that decides *why* FB fails at cube transport — (B1) data coverage, (B2) FB's use of the data, or (B3) the bilinear `FᵀB` factorization — by measuring, on frozen checkpoints, whether terminal-placement information is (a) present in the data, (b) present in the learned representation, and (c) extractable by the bilinear form vs a non-separable head.

**Architecture:** Pure numpy/torch helpers in `evals/` (testable, no jax) compute three things on cached per-state embeddings: a *Monte-Carlo discounted return-to-go* target, *supervised readout ceilings* (linear vs MLP, over B vs left-encoder vs raw-obs), and a *bilinear-vs-joint head* comparator. A thin CLI driver (`scripts/factorization_probe.py`) loads an FB checkpoint via the existing `evals.analysis` plumbing, extracts embeddings over sampled training states, runs the helpers per phase × outcome, and writes parquet + a `story.md` with an automatic B1/B2/B3 verdict. A second tier (config/loss ablations) is specced at coarser granularity and is GPU-gated on the Tier-1 result.

**Tech Stack:** Python, PyTorch (`.venv`), numpy, pandas/parquet, pytest. Reuses `evals/analysis.py` (`build_env_and_agent`, `load_cfg`, `load_checkpoint`), `evals/representation_profile.py` (`q_values`, `v_values`), `evals/training_value.py` (`region_labels`, `cube_to_goal_dist`, `_auc`), `data/ogbench.py` (`load_ogbench_dataset`). All Tier-1 work runs under `.venv/bin/python`.

---

## Scientific framing (read first — this is what the probe must answer)

Same offline buffer trains both FB and GCIQL. GCIQL reaches 0.68 transport-inclusive success on state where FB reaches 0.50, so **raw data coverage (B1) is largely pre-excluded** — but the probe still tests it directly as the absolute ceiling. The live question is **B2 (FB doesn't turn existing terminal transitions into supervision: sampling ρ, goal-inference z=B(g)) vs B3 (the rank-50 bilinear `FᵀB` cannot represent the placement predicate even with ideal supervision)**.

The probe resolves this with a decision tree over four measurements (all per phase reach/grasp/transport, focus = transport):

| Measurement | If it FAILS | If it SUCCEEDS |
|---|---|---|
| MLP readout from **raw obs** → placement | **B1** (label not even in the state) | data is sufficient → continue |
| MLP readout from frozen **B(s)** → placement | **B3** (B threw the info away) | info is in B → continue |
| **Linear** readout from B → placement (given MLP works) | **B3-form** (info in B but nonlinear; bilinear value, which is linear in B, can't use it) | info is linearly in B → B2 (downstream: sampling/z/actor) |
| **Joint** MLP head over (F,B) ≫ **bilinear** form at predicting placement | **B3-form** confirmed (non-separable structure) | bilinear suffices → B2 |

"placement" = `‖cube−goal‖ < ε`. The probe reports all four with held-out generalization so a ceiling is a true ceiling, not overfit.

---

## File Structure

**Isolation constraint (user requirement): modify NO existing files in Tier 1.** All Tier-1 code is new files; existing modules are imported read-only. `mc_return_to_go` lives inside the new `evals/factorization_probe.py` (not appended to `evals/training_value.py`).

- `evals/factorization_probe.py` *(new — the ONLY new library file)* — all pure helpers: `mc_return_to_go`, `r2_score`, `fit_eval_readout` (torch linear/MLP, regression→R² / classification→AUC, held-out), `compare_bilinear_vs_joint` (full-bilinear logistic on outer(F,B) vs MLP on concat), `readout_ceiling_table`, `classify_bucket`. numpy in / numpy metrics out; torch internal. No jax, no sklearn. Reuses `_auc` from `evals.training_value` via read-only import.
- `scripts/factorization_probe.py` *(new)* — CLI: load FB checkpoint, extract `B(s)`, `left_enc(s)`, `F(s,a,z)`, `Q_z`, per-state phase/outcome/MC-return; run helpers; dump `*.parquet` + `story.md` + verdict. Imports `evals.analysis`, `evals.training_value` (`region_labels`, `cube_to_goal_dist`), `evals.representation_profile`, `data.ogbench` read-only.
- `scripts/factorization_probe_aggregate.py` *(new)* — pool per-seed outputs, emit aggregate `story.md` with the B1/B2/B3 verdict and a comparison table.
- `tests/evals/test_factorization_probe.py` *(new)* — unit tests for ALL pure helpers (incl. `mc_return_to_go`) on synthetic data with known separability.
- `analysis/factorization_probe/` *(output, untracked)* — parquet + story.md per run.
- `docs/superpowers/plans/2026-05-24-fb-factorization-readout-probe.md` — this plan.

Tier-2 (Task 8) DOES touch `agents/fb/agent.py` and `train.py` — but every change is behind a default-off config flag (`joint_successor`, `terminal_reweight_k`) so the existing code path is byte-for-byte unchanged when the flags are unset. Tier 2 is GPU-gated and run only if Tier 1 points there.

**Execution amendment (Tasks 1 & 2):** Task 1 *creates* `evals/factorization_probe.py` with the module header (docstring + `import numpy as np`, `import torch`, `import torch.nn as nn`, `import pandas as pd`, `from evals.training_value import _auc`) and `mc_return_to_go`; its test file becomes `tests/evals/test_factorization_probe.py` (not `test_mc_return.py`). Task 2 onward *append* to that same file. The Task-1 test imports `from evals.factorization_probe import mc_return_to_go`, and the Task-1 "verify it fails" expectation is `ModuleNotFoundError: No module named 'evals.factorization_probe'`.

---

## Task 1: MC discounted return-to-go target

**Files:**
- Modify: `evals/training_value.py` (add function after `flow_step_labels`)
- Test: `tests/evals/test_mc_return.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/evals/test_mc_return.py
import numpy as np
from evals.training_value import mc_return_to_go


def test_reaches_goal_gives_gamma_powers():
    # cube walks straight onto the goal at the last step
    goal = np.array([1.0, 0.0, 0.0])
    cube = np.array([[0.0, 0, 0], [0.4, 0, 0], [0.7, 0, 0], [1.0, 0, 0]])
    out = mc_return_to_go(cube, goal, gamma=0.9, thresh=0.05)
    # first in-thresh step is t=3; return-to-go(t) = gamma**(3-t)
    np.testing.assert_allclose(out, [0.9**3, 0.9**2, 0.9**1, 0.9**0], rtol=1e-6)


def test_never_reaches_is_zero():
    goal = np.array([5.0, 0.0, 0.0])
    cube = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    out = mc_return_to_go(cube, goal, gamma=0.9, thresh=0.05)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_already_at_goal_is_one():
    goal = np.array([0.0, 0.0, 0.0])
    cube = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    out = mc_return_to_go(cube, goal, gamma=0.9, thresh=0.05)
    assert out[0] == 1.0  # gamma**0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/evals/test_mc_return.py -v`
Expected: FAIL with `ImportError: cannot import name 'mc_return_to_go'`

- [ ] **Step 3: Write minimal implementation**

```python
# evals/training_value.py  (append near flow_step_labels)
def mc_return_to_go(cube_traj: np.ndarray, goal_xyz: np.ndarray,
                    gamma: float, thresh: float) -> np.ndarray:
    """Per-step discounted return-to-go under a sparse goal-reach reward for
    ONE episode: out[t] = gamma**(k*-t) where k* is the first step >= t with
    ||cube_k - goal|| < thresh, else 0.0 if the goal is never reached from t.
    `cube_traj` = [T,3] cube xyz; `goal_xyz` = [3]. This is the ground-truth
    value of the sparse-reward goal-reaching MDP along the data trajectory."""
    cube_traj = np.asarray(cube_traj, np.float64).reshape(-1, 3)
    goal_xyz = np.asarray(goal_xyz, np.float64).reshape(3)
    d = np.linalg.norm(cube_traj - goal_xyz, axis=1)        # [T]
    within = d < thresh                                      # [T] bool
    T = len(d)
    out = np.zeros(T, dtype=np.float64)
    # next_reach[t] = smallest k >= t with within[k], else -1
    next_reach = -np.ones(T, dtype=np.int64)
    nxt = -1
    for t in range(T - 1, -1, -1):
        if within[t]:
            nxt = t
        next_reach[t] = nxt
    reached = next_reach >= 0
    out[reached] = gamma ** (next_reach[reached] - np.arange(T)[reached])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/evals/test_mc_return.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/training_value.py tests/evals/test_mc_return.py
git commit -m "feat(evals): MC discounted return-to-go target for value calibration"
```

---

## Task 2: Supervised readout ceiling (linear vs MLP, regression + classification)

**Files:**
- Create: `evals/factorization_probe.py`
- Test: `tests/evals/test_factorization_probe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/evals/test_factorization_probe.py
import numpy as np
from evals.factorization_probe import fit_eval_readout, r2_score


def _split(X, y, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); cut = int(len(X) * frac)
    tr, te = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[te], y[te]


def test_linear_recovers_linear_regression_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 6))
    y = X @ np.array([1.0, -2, 0.5, 0, 3, -1]) + 0.01 * rng.normal(size=800)
    Xtr, ytr, Xte, yte = _split(X, y)
    out = fit_eval_readout(Xtr, ytr, Xte, yte, kind="linear", task="regression")
    assert out["score"] > 0.95          # R^2

def test_linear_fails_nonlinear_but_mlp_succeeds_classification():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(np.float64)   # XOR-ish, not linearly separable
    Xtr, ytr, Xte, yte = _split(X, y)
    lin = fit_eval_readout(Xtr, ytr, Xte, yte, kind="linear", task="classification")
    mlp = fit_eval_readout(Xtr, ytr, Xte, yte, kind="mlp", task="classification")
    assert lin["score"] < 0.65          # AUC ~ chance
    assert mlp["score"] > 0.90          # MLP cracks it

def test_r2_score_matches_definition():
    y = np.array([1.0, 2, 3, 4]); pred = np.array([1.1, 1.9, 3.2, 3.8])
    assert abs(r2_score(y, pred) - (1 - np.sum((y-pred)**2)/np.sum((y-y.mean())**2))) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.factorization_probe'`

- [ ] **Step 3: Write minimal implementation**

```python
# evals/factorization_probe.py
"""Pure probe helpers (numpy in / numpy metrics out; torch used internally).
No jax, no sklearn — runs under .venv. Decides whether terminal-placement
information is present in the data / representation and whether the bilinear
FᵀB form can extract it."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from evals.training_value import _auc


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _torch_fit(Xtr, ytr, Xte, head, *, task, epochs=300, lr=1e-2, wd=1e-4, seed=0):
    """Train `head` (nn.Module: in_dim->1) on (Xtr,ytr); return test predictions."""
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    with torch.no_grad():
        out = head(Xte_t).reshape(-1).numpy()
    return out


def _make_head(in_dim: int, kind: str) -> nn.Module:
    if kind == "linear":
        return nn.Linear(in_dim, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    raise ValueError(f"unknown kind {kind!r}")


def fit_eval_readout(Xtr, ytr, Xte, yte, *, kind: str, task: str, seed=0) -> dict:
    """Fit a `kind` ('linear'|'mlp') readout to predict y from X; eval held-out.
    task='regression' -> score is R^2; task='classification' -> score is AUC."""
    Xtr, ytr = np.asarray(Xtr, np.float64), np.asarray(ytr, np.float64)
    Xte, yte = np.asarray(Xte, np.float64), np.asarray(yte, np.float64)
    Xtr_s, Xte_s = _standardize(Xtr, Xte)
    head = _make_head(Xtr_s.shape[1], kind)
    pred = _torch_fit(Xtr_s, ytr, Xte_s, head, task=task, seed=seed)
    if task == "classification":
        score = _auc(pred[yte > 0.5], pred[yte <= 0.5])
    else:
        score = r2_score(yte, pred)
    return {"kind": kind, "task": task, "score": float(score), "n_test": int(len(yte))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/factorization_probe.py tests/evals/test_factorization_probe.py
git commit -m "feat(evals): supervised readout-ceiling helper (linear/MLP, R2/AUC, held-out)"
```

---

## Task 3: Bilinear-vs-joint head comparator

**Files:**
- Modify: `evals/factorization_probe.py`
- Test: `tests/evals/test_factorization_probe.py` (add cases)

- [ ] **Step 1: Write the failing test**

```python
# tests/evals/test_factorization_probe.py  (append)
from evals.factorization_probe import compare_bilinear_vs_joint


def test_bilinear_target_both_succeed():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(2000, 5)); B = rng.normal(size=(2000, 5))
    W = rng.normal(size=(5, 5))
    score = np.einsum("ni,ij,nj->n", F, W, B)          # bilinear in (F,B)
    y = (score > np.median(score)).astype(np.float64)
    out = compare_bilinear_vs_joint(F, B, y, seed=0)
    assert out["auc_bilinear"] > 0.85
    assert out["auc_joint"] > 0.85
    assert out["gap"] < 0.10                            # joint not much better

def test_nonbilinear_target_joint_wins():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(3000, 5)); B = rng.normal(size=(3000, 5))
    y = (np.linalg.norm(F - B, axis=1) > np.median(np.linalg.norm(F - B, axis=1))
         ).astype(np.float64)                            # ||F-B||: NOT bilinear
    out = compare_bilinear_vs_joint(F, B, y, seed=0)
    assert out["auc_joint"] - out["auc_bilinear"] > 0.10  # joint clearly wins
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -k bilinear -v`
Expected: FAIL with `ImportError: cannot import name 'compare_bilinear_vs_joint'`

- [ ] **Step 3: Write minimal implementation**

```python
# evals/factorization_probe.py  (append)
def _split_idx(n, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); cut = int(n * frac)
    return idx[:cut], idx[cut:]


def compare_bilinear_vs_joint(F, B, y, *, seed=0) -> dict:
    """Does the placement label y separate under the BILINEAR family
    (score = FᵀW B, i.e. linear in the outer product F⊗B) vs a JOINT MLP over
    concat([F,B])? Both fit to predict y; report held-out AUC and the gap.
    gap = auc_joint - auc_bilinear. A large positive gap => placement is NOT a
    bilinear function of (F,B) => the FᵀB form is the bottleneck (B3-form)."""
    F = np.asarray(F, np.float64); B = np.asarray(B, np.float64)
    y = np.asarray(y, np.float64)
    tr, te = _split_idx(len(F), seed=seed)
    # full bilinear family == linear model on the outer-product features
    outer = (F[:, :, None] * B[:, None, :]).reshape(len(F), -1)   # [N, dF*dB]
    bil = fit_eval_readout(outer[tr], y[tr], outer[te], y[te],
                           kind="linear", task="classification", seed=seed)
    joint_in = np.concatenate([F, B], axis=1)
    joint = fit_eval_readout(joint_in[tr], y[tr], joint_in[te], y[te],
                             kind="mlp", task="classification", seed=seed)
    return {"auc_bilinear": bil["score"], "auc_joint": joint["score"],
            "gap": float(joint["score"] - bil["score"]), "n_test": int(len(te))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/factorization_probe.py tests/evals/test_factorization_probe.py
git commit -m "feat(evals): bilinear-vs-joint head comparator for FB factorization test"
```

---

## Task 4: Per-phase × outcome probe aggregator (pure)

**Files:**
- Modify: `evals/factorization_probe.py`
- Test: `tests/evals/test_factorization_probe.py` (add case)

- [ ] **Step 1: Write the failing test**

```python
# tests/evals/test_factorization_probe.py  (append)
import pandas as pd
from evals.factorization_probe import readout_ceiling_table


def test_readout_ceiling_table_shape_and_keys():
    rng = np.random.default_rng(4)
    n = 1200
    df = pd.DataFrame({
        "region": rng.choice(["reach", "grasp", "transport"], n),
        "placement": rng.integers(0, 2, n).astype(float),
        "mc_return": rng.random(n),
    })
    feats = {"B": rng.normal(size=(n, 8)), "raw": rng.normal(size=(n, 12))}
    tab = readout_ceiling_table(df, feats, targets=("placement", "mc_return"), seed=0)
    # one row per (feature, target, region, kind)
    assert set(tab.columns) >= {"feature", "target", "region", "kind", "score", "n_test"}
    assert set(tab["region"]) <= {"reach", "grasp", "transport", "all"}
    assert set(tab["kind"]) == {"linear", "mlp"}
    assert {"B", "raw"} <= set(tab["feature"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -k ceiling_table -v`
Expected: FAIL with `ImportError: cannot import name 'readout_ceiling_table'`

- [ ] **Step 3: Write minimal implementation**

```python
# evals/factorization_probe.py  (append)
import pandas as pd


def readout_ceiling_table(df, feats: dict, *, targets=("placement", "mc_return"),
                          min_per_group=80, seed=0) -> "pd.DataFrame":
    """For each feature matrix in `feats` (e.g. {'B':..,'left_enc':..,'raw':..}),
    each target column in `df`, each region (+'all'), and kind in {linear,mlp},
    fit a held-out readout and record the score. `df` rows align with feats rows.
    target 'placement' (0/1) -> AUC; anything else -> R^2 regression."""
    rng = np.random.default_rng(seed)
    rows = []
    regions = ["all"] + sorted(df["region"].unique().tolist())
    for fname, X in feats.items():
        X = np.asarray(X, np.float64)
        for region in regions:
            mask = np.ones(len(df), bool) if region == "all" else (df["region"] == region).to_numpy()
            if mask.sum() < min_per_group:
                continue
            idx = np.where(mask)[0]
            perm = rng.permutation(len(idx)); cut = len(idx) // 2
            tr, te = idx[perm[:cut]], idx[perm[cut:]]
            for tgt in targets:
                task = "classification" if tgt == "placement" else "regression"
                y = df[tgt].to_numpy(np.float64)
                if task == "classification" and (y[te].sum() == 0 or y[te].sum() == len(te)):
                    continue  # need both classes in the test split
                for kind in ("linear", "mlp"):
                    out = fit_eval_readout(X[tr], y[tr], X[te], y[te],
                                           kind=kind, task=task, seed=seed)
                    rows.append({"feature": fname, "target": tgt, "region": region,
                                 "kind": kind, "task": task, "score": out["score"],
                                 "n_test": out["n_test"]})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/factorization_probe.py tests/evals/test_factorization_probe.py
git commit -m "feat(evals): per-region readout-ceiling table aggregator"
```

---

## Task 5: Verdict logic (B1/B2/B3 decision tree)

**Files:**
- Modify: `evals/factorization_probe.py`
- Test: `tests/evals/test_factorization_probe.py` (add cases)

- [ ] **Step 1: Write the failing test**

```python
# tests/evals/test_factorization_probe.py  (append)
from evals.factorization_probe import classify_bucket


def _ceiling(raw_mlp, b_mlp, b_lin, region="transport", target="placement"):
    import pandas as pd
    return pd.DataFrame([
        {"feature": "raw", "target": target, "region": region, "kind": "mlp", "score": raw_mlp},
        {"feature": "B",   "target": target, "region": region, "kind": "mlp", "score": b_mlp},
        {"feature": "B",   "target": target, "region": region, "kind": "linear", "score": b_lin},
    ])


def test_bucket_b1_when_raw_cannot_predict():
    v = classify_bucket(_ceiling(0.55, 0.55, 0.52), bilinear_gap=0.0)
    assert v["bucket"] == "B1"

def test_bucket_b3_repr_when_b_loses_to_raw():
    v = classify_bucket(_ceiling(0.95, 0.60, 0.55), bilinear_gap=0.02)
    assert v["bucket"] == "B3-representation"

def test_bucket_b3_form_when_b_nonlinear_or_gap_large():
    v = classify_bucket(_ceiling(0.95, 0.93, 0.60), bilinear_gap=0.20)
    assert v["bucket"] == "B3-form"

def test_bucket_b2_when_b_linear_and_bilinear_ok():
    v = classify_bucket(_ceiling(0.95, 0.93, 0.90), bilinear_gap=0.03)
    assert v["bucket"] == "B2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -k bucket -v`
Expected: FAIL with `ImportError: cannot import name 'classify_bucket'`

- [ ] **Step 3: Write minimal implementation**

```python
# evals/factorization_probe.py  (append)
def classify_bucket(ceiling_df, *, bilinear_gap: float, region="transport",
                    target="placement", lo=0.65, hi=0.80, gap_thr=0.10) -> dict:
    """Map the transport-phase readout ceilings + bilinear gap to a bucket.
    lo: AUC at/below which a readout 'fails'; hi: AUC at/above which it 'succeeds'.
    Decision tree (see plan's scientific framing table)."""
    def s(feature, kind):
        m = ceiling_df[(ceiling_df.feature == feature) & (ceiling_df.kind == kind)
                       & (ceiling_df.region == region) & (ceiling_df.target == target)]
        return float(m["score"].iloc[0]) if len(m) else float("nan")
    raw_mlp, b_mlp, b_lin = s("raw", "mlp"), s("B", "mlp"), s("B", "linear")
    if raw_mlp < lo:
        return {"bucket": "B1", "why": f"raw-obs MLP AUC {raw_mlp:.2f}<{lo}: placement not recoverable from state (data/label problem)"}
    if b_mlp < lo or (raw_mlp - b_mlp) > 0.20:
        return {"bucket": "B3-representation", "why": f"B-MLP AUC {b_mlp:.2f} ≪ raw {raw_mlp:.2f}: the representation discarded placement info"}
    if b_lin < lo or bilinear_gap > gap_thr:
        return {"bucket": "B3-form", "why": f"B-linear AUC {b_lin:.2f}<{lo} or bilinear gap {bilinear_gap:.2f}>{gap_thr}: info in B is non-separable; FᵀB (linear in B) cannot use it"}
    return {"bucket": "B2", "why": f"placement is linearly+bilinearly recoverable (B-lin {b_lin:.2f}, gap {bilinear_gap:.2f}); failure is downstream (ρ-sampling / z=B(g) / actor)"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/evals/test_factorization_probe.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add evals/factorization_probe.py tests/evals/test_factorization_probe.py
git commit -m "feat(evals): B1/B2/B3 bucket classifier from readout ceilings"
```

---

## Task 6: CLI driver — extract embeddings + run probe on a checkpoint

**Files:**
- Create: `scripts/factorization_probe.py`
- (manual run; no unit test — exercised by the smoke run in Step 4)

- [ ] **Step 1: Write the driver**

```python
# scripts/factorization_probe.py
"""Factorization-ablation + readout-ceiling probe on a frozen FB checkpoint.
Run under .venv (torch). Example:

  .venv/bin/python scripts/factorization_probe.py \
    --config configs/domain/cube_single.yaml \
    --checkpoint results/Factored-FB-cube-run/<run>__s3/checkpoints/final.pt \
    --data-path datasets --out analysis/factorization_probe/s3 \
    --n-states 6000 --eps 0.05 --gamma 0.99
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
from evals.training_value import region_labels, cube_to_goal_dist, mc_return_to_go
from evals.factorization_probe import (readout_ceiling_table, compare_bilinear_vs_joint,
                                       classify_bucket)
from evals.representation_profile import q_values   # FB Q_z = (F·z)
from data.ogbench import load_ogbench_dataset


def _episode_files(data_path, domain):
    import glob, os
    return sorted(glob.glob(os.path.join(data_path, domain, "episode_*.npz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6000)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, device=args.device)
    env, agent, evaluator = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.model.eval()

    # 1. Load episodes, sample states, build per-state frame with phase/MC-return.
    files = _episode_files(args.data_path, cfg.domain)
    rng = np.random.default_rng(args.seed)
    obs_l, cube_l, grip_l, lift_l, mc_l, region_pos = [], [], [], [], [], []
    per_ep = max(1, args.n_states // max(1, len(files)))
    tasks = list(range(1, 6))
    # one representative goal per task, from the env task info
    goals = {t: np.asarray(evaluator._goal_xyz_for(t), np.float64) for t in tasks}  # see NOTE
    for f in files:
        z = np.load(f)
        obs = z["observation"]; cube = z["physics"][:, 14:16]  # xy; see coverage appendix
        cube_xyz = np.concatenate([cube, z["physics"][:, 16:17]], axis=1)  # add z-height col
        grip = z["physics"][:, GRIP_COL]; lift = z["physics"][:, LIFT_COL]  # NOTE below
        # MC return-to-go toward task1 goal as the canonical transport target
        mc = mc_return_to_go(cube_xyz, goals[1], gamma=args.gamma, thresh=args.eps)
        sel = rng.choice(len(obs), size=min(per_ep, len(obs)), replace=False)
        obs_l.append(obs[sel]); cube_l.append(cube_xyz[sel])
        grip_l.append(grip[sel]); lift_l.append(lift[sel]); mc_l.append(mc[sel])
    obs = np.concatenate(obs_l); cube = np.concatenate(cube_l)
    grip = np.concatenate(grip_l); lift = np.concatenate(lift_l)
    mc = np.concatenate(mc_l)
    region = region_labels(grip, lift, cfg.thr)
    d = cube_to_goal_dist(cube, goals[1])
    placement = (d < args.eps).astype(np.float64)
    frame = pd.DataFrame({"region": region, "placement": placement, "mc_return": mc, "d": d})

    # 2. Extract frozen embeddings: B(s), left_enc(s), F(s,a,z), Q_z, and raw obs.
    obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device)
    z_task, _ = evaluator._infer_z(1)        # z for task1
    z_t = torch.as_tensor(z_task, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        B = agent.model.backward_map(obs_t).cpu().numpy()
        left = agent.model._left_encoder(agent.model._fw_encoder(
            agent.model._normalize(obs_t))).cpu().numpy()
        # F at the policy action (same convention as value extraction)
        act = agent.model.act(obs_t, z_t.expand(len(obs_t), -1), mean=True)
        Fs = agent.model._forward_map(
            agent.model._left_encoder(agent.model._fw_encoder(agent.model._normalize(obs_t))),
            z_t.expand(len(obs_t), -1), act)          # [num_parallel, N, z_dim]
        F = Fs.mean(0).cpu().numpy()
    feats = {"raw": obs.astype(np.float64), "B": B, "left_enc": left}

    # 3. Readout ceilings, bilinear-vs-joint, verdict.
    ceiling = readout_ceiling_table(frame, feats, targets=("placement", "mc_return"), seed=args.seed)
    tr_mask = (frame["region"] == "transport").to_numpy()
    bvj = compare_bilinear_vs_joint(F[tr_mask], B[tr_mask],
                                    frame["placement"].to_numpy()[tr_mask], seed=args.seed)
    verdict = classify_bucket(ceiling, bilinear_gap=bvj["gap"])

    # 4. Dump.
    ceiling.to_parquet(out / "readout_ceiling.parquet")
    pd.DataFrame([bvj]).to_parquet(out / "bilinear_vs_joint.parquet")
    frame.to_parquet(out / "probe_frame.parquet")
    (out / "verdict.json").write_text(json.dumps({**verdict, **bvj}, indent=2))
    _write_story(out, ceiling, bvj, verdict)
    print(f"[factorization_probe] verdict={verdict['bucket']} :: {verdict['why']}")


def _write_story(out, ceiling, bvj, verdict):
    tr = ceiling[ceiling.region == "transport"]
    lines = ["# Factorization probe — transport phase", "",
             f"**Verdict: {verdict['bucket']}** — {verdict['why']}", "",
             f"Bilinear vs joint (placement): AUC_bilinear={bvj['auc_bilinear']:.3f}, "
             f"AUC_joint={bvj['auc_joint']:.3f}, gap={bvj['gap']:.3f}", "",
             "## Transport-phase readout ceilings", "",
             "| feature | target | kind | score |", "| :-- | :-- | :-- | :-- |"]
    for _, r in tr.sort_values(["target", "feature", "kind"]).iterrows():
        lines.append(f"| {r.feature} | {r.target} | {r.kind} | {r.score:.3f} |")
    (out / "story.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
```

> **NOTE for the implementer (resolve before running, ~15 min):** three repo-specific bindings are marked above and must be filled from existing code, not guessed:
> 1. `GRIP_COL` / `LIFT_COL` and the cube xyz columns — `scripts/training_value_profile.py` already extracts grip/lift/cube from `physics`; copy its exact column indices (the coverage appendix of the paper documents `physics[:,14:16]` for cube xy). Reuse that module's `_sample_states`/`region` extraction verbatim rather than re-deriving.
> 2. `evaluator._infer_z(task)` is the FB z-inference already used in `scripts/training_value_profile.py:158`; confirm the exact attribute name there and reuse it.
> 3. `evaluator._goal_xyz_for(t)` is a placeholder name — `scripts/representation_profile.py:106` has `goal_for(task)` and the paper reads goals via `cur_task_info["goal_xyzs"]`. Wire to the existing helper. If only task1's goal is needed for the canonical transport target, hardcode the single call.
>
> Per your `fb-value-nondeterminism` memory: average B/F/Q over **3 z draws** (loop `_infer_z` with different seeds, mean the embeddings) before the readouts, or the transport signal will be noisy.

- [ ] **Step 2: Resolve the three NOTE bindings**

Open `scripts/training_value_profile.py` and `scripts/representation_profile.py`; copy the exact grip/lift/cube column extraction, `_infer_z`, and `goal_for` calls into the driver. Delete the `GRIP_COL`/`LIFT_COL`/`_goal_xyz_for` placeholders.

- [ ] **Step 3: Smoke-run on one existing checkpoint**

Run:
```bash
.venv/bin/python scripts/factorization_probe.py \
  --config configs/domain/cube_single.yaml \
  --checkpoint "$(ls -d results/Factored-FB-cube-run/*__s3/checkpoints/final.pt | head -1)" \
  --data-path datasets --out analysis/factorization_probe/s3 \
  --n-states 4000 --eps 0.05 --gamma 0.99 --device cuda
```
Expected: prints `[factorization_probe] verdict=B...`; writes `analysis/factorization_probe/s3/{readout_ceiling,bilinear_vs_joint,probe_frame}.parquet`, `verdict.json`, `story.md`.

- [ ] **Step 4: Sanity-check the output**

Run: `cat analysis/factorization_probe/s3/story.md`
Expected: a transport-phase table where `raw` MLP AUC for `placement` is high (≳0.9, cube pose is in the state) — if it isn't, the driver bindings are wrong (the label/feature alignment broke), fix before trusting any verdict.

- [ ] **Step 5: Commit**

```bash
git add scripts/factorization_probe.py
git commit -m "feat(scripts): FB factorization + readout-ceiling probe driver"
```

---

## Task 7: Cross-seed aggregation + verdict

**Files:**
- Create: `scripts/factorization_probe_aggregate.py`
- (manual run)

- [ ] **Step 1: Write the aggregator**

```python
# scripts/factorization_probe_aggregate.py
"""Pool per-seed factorization-probe outputs into a single verdict + table.
  .venv/bin/python scripts/factorization_probe_aggregate.py \
    --root analysis/factorization_probe --out analysis/factorization_probe/aggregate
"""
import argparse, json
from pathlib import Path
import pandas as pd
from evals.factorization_probe import classify_bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ceil = [pd.read_parquet(p).assign(seed=p.parent.name)
            for p in root.glob("*/readout_ceiling.parquet")]
    bvj = [pd.read_parquet(p).assign(seed=p.parent.name)
           for p in root.glob("*/bilinear_vs_joint.parquet")]
    if not ceil:
        raise SystemExit("no per-seed readout_ceiling.parquet under root")
    ceil = pd.concat(ceil, ignore_index=True); bvj = pd.concat(bvj, ignore_index=True)

    mean_ceil = (ceil.groupby(["feature", "target", "region", "kind"])["score"]
                 .mean().reset_index())
    mean_gap = float(bvj["gap"].mean())
    verdict = classify_bucket(mean_ceil, bilinear_gap=mean_gap)

    mean_ceil.to_parquet(out / "readout_ceiling_mean.parquet")
    (out / "verdict.json").write_text(json.dumps(
        {**verdict, "gap_mean": mean_gap, "n_seeds": int(bvj["seed"].nunique())}, indent=2))
    tr = mean_ceil[mean_ceil.region == "transport"].sort_values(["target", "feature", "kind"])
    lines = ["# Factorization probe — cross-seed verdict", "",
             f"**{verdict['bucket']}** — {verdict['why']}",
             f"(mean bilinear gap {mean_gap:.3f}, n_seeds {bvj['seed'].nunique()})", "",
             "| feature | target | kind | mean score |", "| :-- | :-- | :-- | :-- |"]
    for _, r in tr.iterrows():
        lines.append(f"| {r.feature} | {r.target} | {r.kind} | {r.score:.3f} |")
    (out / "story.md").write_text("\n".join(lines) + "\n")
    print(f"[aggregate] {verdict['bucket']} (gap {mean_gap:.3f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on per-seed outputs**

First generate ≥2 seeds (re-run Task 6 with `--checkpoint ...__s4 --out analysis/factorization_probe/s4`), then:
```bash
.venv/bin/python scripts/factorization_probe_aggregate.py \
  --root analysis/factorization_probe --out analysis/factorization_probe/aggregate
cat analysis/factorization_probe/aggregate/story.md
```
Expected: a single bucket verdict with mean transport-phase ceilings.

- [ ] **Step 3: Commit**

```bash
git add scripts/factorization_probe_aggregate.py
git commit -m "feat(scripts): cross-seed aggregation + verdict for factorization probe"
```

---

## Task 8 (Tier 2 — GPU, gated on Tier-1 verdict): retrain ablations

Run **only the branch the Tier-1 verdict points to.** These change training, so they are coarser-grained; follow the repo's existing TDD habits for any new pure helper, and re-use `train.py`.

- [ ] **8a — Rank sweep (tests B3-form via capacity).** Retrain state FB at `z_dim ∈ {50, 256, 512}` (and matching `L_dim`), 2 tasks × 2 seeds, then re-run the Task-6 probe on each. Command per cell:
  `.venv/bin/python train.py domain=cube_single z_dim=256 L_dim=256 ortho_coef=1000 lr_b=1e-4 num_train_steps=1000000 seed=1 device=cuda use_wandb=true`
  Decision: transport placement-AUC and success rise with rank ⇒ partly capacity; flat (as pixel-at-rank-50 predicts) ⇒ not raw rank.

- [ ] **8b — Joint successor head (the definitive B3-form ablation).** Add an optional non-separable successor head to `agents/fb/agent.py:update_fb`: replace `Ms = matmul(Fs, B.T)` scoring with `Ms = joint_head(F_i, B_j)` (a small MLP over concatenated `[F_i, B_j]` broadcast over the batch×batch grid) behind a config flag `joint_successor: true`. Keep `Q_z=(F·z)` for the actor/zero-shot path unchanged. Retrain; re-probe. Decision: joint head resolves transport while bilinear cannot at any rank ⇒ **the FᵀB factorization is the bottleneck** (headline B3 result).

- [ ] **8c — ρ-reweight (tests B2).** In `update_fb`, weight the off-diagonal/diagonal FB loss terms by an importance weight `w(s⁺)` that oversamples near-terminal `goal` rows (e.g. `w ∝ 1[‖cube−any-dataset-goal‖<ε] · k + 1`), behind flag `terminal_reweight_k`. No new data. Retrain; re-probe. Decision: transport resolution jumps ⇒ **B2 (sampling), fixable without architecture change**.

- [ ] **8d — Privileged-z (no retrain; tests B2 goal-inference).** Add `--privileged-z` to `scripts/factorization_probe.py`: build `z` from the true goal state's `B(goal_state)` (a single sharp goal) instead of `evaluator._infer_z`, recompute `Q_z` and success on held-out rollouts. Decision: success/AUC jumps ⇒ the `z=B(g)` inference path is the bottleneck, not F or the value field.

---

## Self-Review

**1. Spec coverage.** The scientific question (data B1 / usage B2 / factorization B3) maps to tasks: B1 → raw-obs readout (Tasks 2,4,5,6); B3-representation → B-MLP readout (Tasks 2,4,5); B3-form → B-linear gap + bilinear-vs-joint (Tasks 3,5) and the joint-successor retrain (8b); B2 → linear-B-recoverable + ρ-reweight (8c) + privileged-z (8d). MC-return calibration target (Task 1) feeds the regression ceilings. Cross-seed robustness (Task 7). The user's original "where is the failure (reach/grasp/transport)" is covered by the per-region tables (Task 4) — the probe is region-conditioned throughout.

**2. Placeholder scan.** Code steps contain complete implementations and tests. The only deliberately deferred items are the three repo-binding NOTEs in Task 6 (grip/lift/cube columns, `_infer_z`, `goal_for`) — these are explicitly flagged with the exact source files/line to copy from (`training_value_profile.py`, `representation_profile.py`) rather than invented, because guessing column indices would be wrong; resolving them is Task 6 Step 2 with a sanity check (Step 4) that fails loudly if mis-wired.

**3. Type consistency.** `fit_eval_readout(..., kind, task) -> {"score",...}` is used identically in Tasks 2/3/4. `compare_bilinear_vs_joint -> {"auc_bilinear","auc_joint","gap"}` consumed by `classify_bucket(..., bilinear_gap=...)` (Task 5) and the driver (Task 6) and aggregator (Task 7) — names match. `readout_ceiling_table` columns `{feature,target,region,kind,score,n_test}` match what `classify_bucket` reads (`feature,kind,region,target,score`) and what the aggregator groups on. `mc_return_to_go(cube_traj, goal_xyz, gamma, thresh)` signature matches its call in the driver. `_auc(pos, neg)` reused from `evals/training_value.py` (verified present).

---

## Execution Handoff

Tier 1 (Tasks 1–7) is the answer to "data vs FB decomposition" and runs entirely on existing checkpoints under `.venv`. Tier 2 (Task 8) is GPU-gated on the Tier-1 verdict — run only the pointed-to branch.
