# Training-Value Outcome-Split (T1 Mirror) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the committed `training_value` pipeline so it labels each training state as success-bound / fail-bound for each task (cube reaches within 4 cm of the goal within the next 200 steps) and reports whether FB/GCIQL value *discriminates* those outcomes per phase — the T1 question on training data.

**Architecture:** Add pure, unit-tested helpers to `evals/training_value.py`; extend the FB pass of `scripts/training_value_profile.py` to track each sampled state's `(episode_id, step_id)`, compute the per-(state,task) outcome label, write it into `fb_values.parquet` + `training_states.npz`; the GCIQL pass reads the stored label; the aggregate adds the outcome-split Spearman table, an AUC/ΔV discrimination table, by-outcome value-vs-distance curves, and an outcome-composition sanity table. Single-dir and multi-seed (`p*/`) pooling both supported.

**Tech Stack:** Python 3 (`.venv` torch / `.venv-jax-cpu` jax), numpy, pandas, matplotlib, MuJoCo/ogbench.

**Spec:** `docs/superpowers/specs/2026-05-22-training-value-outcome-split-design.md`

---

## Conventions & ground rules

- FB pass + aggregate + tests run under `.venv/bin/python`; GCIQL under `.venv-jax-cpu/bin/python` ([[project-venvs]]).
- Commit only `evals/`, `scripts/`, `tests/`, `docs/`. `analysis/`, `RESULTS/`, `datasets/` stay untracked ([[project-untracked-artifacts]]).
- No Claude attribution in commit messages ([[feedback-commit-style]]).
- Outcome label: `success_bound` iff `min ‖cube[t:t+H] − g‖ < THRESH` with `H=200`, `THRESH=0.04`; per (state, task); empty tail → `fail_bound`.
- Phases unchanged: reach / grasp / transport.

## File structure

- `evals/training_value.py` (modify) — add `horizon_reach_label`, `outcome_spearman_table`, `value_discrimination` (+ private `_rankdata`, `_auc`).
- `scripts/training_value_profile.py` (modify) — `_sample_states` returns `(ep_id, step_id, cube_by_ep)`; add `_outcome_labels`; FB writes `outcome` col + npz key; GCIQL reads it; CLI `--horizon`/`--thresh`.
- `scripts/training_value_aggregate.py` (modify) — outcome Spearman table, discrimination table, by-outcome curves, composition; guarded so legacy dirs (no `outcome`) still work.
- `tests/test_training_value.py` (modify) — unit tests for the three new helpers.

---

## Task 1: Pure helpers (`evals/training_value.py`)

**Files:**
- Modify: `evals/training_value.py`
- Test: `tests/test_training_value.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_value.py` (and extend the import line at the top):

```python
from evals.training_value import (region_labels, cube_to_goal_dist,
                                   phase_spearman_table, horizon_reach_label,
                                   outcome_spearman_table, value_discrimination)
```

```python
def test_horizon_reach_label_window_and_thresh():
    d = np.array([0.10, 0.08, 0.03, 0.20])
    assert horizon_reach_label(d, 0.04, 2) is False   # min(first 2)=0.08
    assert horizon_reach_label(d, 0.04, 3) is True    # 0.03 < 0.04
    assert horizon_reach_label(np.array([0.05, 0.06]), 0.04, 5) is False
    assert horizon_reach_label(np.array([]), 0.04, 5) is False  # empty tail


def test_outcome_spearman_table_monotone():
    rows = []
    for oc in ("success_bound", "fail_bound"):
        for i in range(8):
            d = 0.3 - 0.03 * i
            rows.append({"task": "task1", "region": "transport",
                         "outcome": oc, "d": d, "V": 1.0 - d})
    tab = outcome_spearman_table(pd.DataFrame(rows), value_col="V")
    tab = tab.set_index(["region", "outcome"])["rho_mean"]
    assert tab.loc[("transport", "success_bound")] > 0.99
    assert tab.loc[("transport", "fail_bound")] > 0.99


def test_value_discrimination_separable_and_random():
    rows = []
    for t in ("task1", "task2"):
        for v in (1.0, 2.0, 3.0, 4.0):
            rows.append({"task": t, "region": "transport",
                         "outcome": "success_bound", "d": 0.1, "V": v + 10})
            rows.append({"task": t, "region": "transport",
                         "outcome": "fail_bound", "d": 0.1, "V": v})
    disc = value_discrimination(pd.DataFrame(rows), "V").set_index("region")
    assert disc.loc["transport", "auc"] > 0.99
    assert disc.loc["transport", "mean_dV"] > 0

    rng = np.random.default_rng(0)
    rrows = []
    for oc in ("success_bound", "fail_bound"):
        for v in rng.normal(size=2000):
            rrows.append({"task": "task1", "region": "reach",
                          "outcome": oc, "d": 0.1, "V": float(v)})
    auc = value_discrimination(pd.DataFrame(rrows), "V").set_index(
        "region").loc["reach", "auc"]
    assert abs(auc - 0.5) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_training_value.py -v`
Expected: the three new tests FAIL with `ImportError: cannot import name 'horizon_reach_label'`.

- [ ] **Step 3: Implement the helpers**

Append to `evals/training_value.py`:

```python
def horizon_reach_label(d_future: np.ndarray, thresh: float,
                        horizon: int) -> bool:
    """True iff the cube comes within `thresh` of the goal within the first
    `horizon` future steps. `d_future` = cube->goal distances from step t
    onward (within one episode). Empty tail -> False."""
    seg = np.asarray(d_future, np.float64).reshape(-1)[:horizon]
    return bool(seg.size and np.min(seg) < thresh)


def outcome_spearman_table(df, value_col: str = "V", group_cols=("task",)):
    """Per (region, outcome) Spearman rho(value, -d), mean/std over the
    `group_cols` combinations. `df` has columns region, outcome, d,
    <value_col>, + group_cols."""
    import pandas as pd

    group_cols = list(group_cols)
    rows = []
    for (region, outcome), g in df.groupby(["region", "outcome"]):
        vals = []
        for _, gg in g.groupby(group_cols):
            vals.append(_spearman(gg[value_col].to_numpy(),
                                  -gg["d"].to_numpy()))
        vals = np.array(vals, dtype=float)
        rows.append({"region": region, "outcome": outcome,
                     "rho_mean": float(np.nanmean(vals))
                     if np.isfinite(vals).any() else float("nan"),
                     "rho_std": float(np.nanstd(vals))
                     if np.isfinite(vals).any() else float("nan"),
                     "n_groups": int(np.isfinite(vals).sum())})
    return pd.DataFrame(rows)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), tie-aware; vectorized O(n log n)."""
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    is_new = np.empty(len(a), bool)
    is_new[0] = True
    is_new[1:] = sa[1:] != sa[:-1]
    grp = np.cumsum(is_new) - 1
    pos = np.arange(1, len(a) + 1, dtype=np.float64)
    mean_r = np.bincount(grp, weights=pos) / np.bincount(grp)
    ranks = np.empty(len(a), np.float64)
    ranks[order] = mean_r[grp]
    return ranks


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(value(pos) > value(neg)) via the rank-sum statistic."""
    pos = np.asarray(pos, np.float64)
    neg = np.asarray(neg, np.float64)
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[:n_p].sum()
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def value_discrimination(df, value_col: str = "V", group_cols=("task",)):
    """Per region: AUC of `value_col` separating success_bound (positive) from
    fail_bound, plus mean dV after per-group z-scoring. `df` has columns
    region, outcome, <value_col>, + group_cols."""
    import pandas as pd

    group_cols = list(group_cols)
    z = df.copy()
    z["_vz"] = z.groupby(group_cols)[value_col].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    rows = []
    for region, g in z.groupby("region"):
        pos = g[g["outcome"] == "success_bound"]
        neg = g[g["outcome"] == "fail_bound"]
        rows.append({"region": region,
                     "auc": _auc(pos[value_col].to_numpy(),
                                 neg[value_col].to_numpy()),
                     "mean_dV": float(pos["_vz"].mean() - neg["_vz"].mean())
                     if len(pos) and len(neg) else float("nan"),
                     "n_success": int(len(pos)), "n_fail": int(len(neg))})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_training_value.py -v`
Expected: PASS (6 passed — 3 existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add evals/training_value.py tests/test_training_value.py
git commit -m "feat(training_value): outcome helpers — horizon-reach label, outcome spearman, AUC/dV discrimination"
```

---

## Task 2: FB pass — outcome label + storage (`scripts/training_value_profile.py`)

**Files:**
- Modify: `scripts/training_value_profile.py`

- [ ] **Step 1: Extend the import and `_sample_states`**

Change the helper import line (top of file) to add `horizon_reach_label`:

```python
from evals.training_value import (region_labels, cube_to_goal_dist,
                                   horizon_reach_label)
```

Replace `_sample_states` with a version that also returns per-state
`(ep_id, step_id)` and the per-episode cube trajectories:

```python
def _sample_states(files, n_states, seed=0):
    """Aligned (obs, physics, action) sampled across episodes, plus each
    sampled state's (ep_id, step_id) and the per-episode cube trajectories
    (for the outcome-label tail lookups)."""
    rng = np.random.default_rng(seed)
    obs_l, phys_l, act_l, ep_l, st_l, cube_by_ep = [], [], [], [], [], []
    for ei, f in enumerate(files):
        z = np.load(f)
        o = np.asarray(z["observation"], np.float32)
        ph = np.asarray(z["physics"], np.float32)
        obs_l.append(o)
        phys_l.append(ph)
        act_l.append(np.asarray(z["action"], np.float32))
        ep_l.append(np.full(len(o), ei, dtype=np.int64))
        st_l.append(np.arange(len(o), dtype=np.int64))
        cube_by_ep.append(ph[:, CUBE_SLICE].astype(np.float64))
    obs = np.concatenate(obs_l)
    phys = np.concatenate(phys_l)
    act = np.concatenate(act_l)
    ep_id = np.concatenate(ep_l)
    step_id = np.concatenate(st_l)
    if len(obs) > n_states:
        idx = rng.choice(len(obs), n_states, replace=False)
        obs, phys, act = obs[idx], phys[idx], act[idx]
        ep_id, step_id = ep_id[idx], step_id[idx]
    return obs, phys, act, ep_id, step_id, cube_by_ep
```

- [ ] **Step 2: Add the `_outcome_labels` helper**

Insert after `_phase_composition` (above `_fb`):

```python
def _outcome_labels(cube_by_ep, ep_id, step_id, goal_xyz, horizon, thresh):
    """[n_states] bool: success_bound per state for one goal — does the cube
    reach within `thresh` of `goal_xyz` within `horizon` future steps of its
    own episode."""
    g = np.asarray(goal_xyz, np.float64).reshape(3)
    out = np.zeros(len(ep_id), dtype=bool)
    for i in range(len(ep_id)):
        tail = cube_by_ep[ep_id[i]][step_id[i]: step_id[i] + horizon]
        d_future = np.linalg.norm(tail - g, axis=1)
        out[i] = horizon_reach_label(d_future, thresh, horizon)
    return out
```

- [ ] **Step 3: Wire outcomes into `_fb`**

In `_fb`, change the `_sample_states` unpack to the 6-tuple:

```python
    obs, phys, act, ep_id, step_id, cube_by_ep = _sample_states(
        files, args.n_states, seed=cfg.seed)
```

Inside the `for ti, task in enumerate(tasks, start=1):` loop, after
`d = cube_to_goal_dist(cube, g)`, compute the label and keep the column:

```python
        oc = _outcome_labels(cube_by_ep, ep_id, step_id, g,
                             args.horizon, args.thresh)
```

and add `"outcome"` to the per-row dict:

```python
            rows.append({"task": f"task{ti}", "region": str(region[i]),
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]),
                         "V_policy": float(v_pol[i]),
                         "V_data": float(v_dat[i]),
                         "outcome": "success_bound" if oc[i]
                         else "fail_bound"})
```

Accumulate the per-task columns into a matrix. Before the task loop add
`outcome_cols = []`; inside the loop (right after computing `oc`) add
`outcome_cols.append(oc)`. Then change the `np.savez` to store the matrix:

```python
    outcome_mat = np.stack(outcome_cols, axis=1)  # [n_states, n_tasks] bool
    np.savez(out / "training_states.npz", obs=obs, cube=cube,
             region=region.astype("U12"),
             goals=np.stack([goals[f"task{i}"] for i in range(1, 6)]),
             outcome=outcome_mat)
```

- [ ] **Step 4: Add `--horizon` / `--thresh` CLI flags**

In `main`, in the `# fb` argument group, add:

```python
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.04)
```

- [ ] **Step 5: Smoke the FB pass (reduced)**

```bash
D=$(ls -d RESULTS/Factored-FB-cube-run/*__s3 | head -1)
MUJOCO_GL=glfw .venv/bin/python scripts/training_value_profile.py --method fb \
  --config "$D/.hydra/config.yaml" --checkpoint "$D/checkpoints/final.pt" \
  --data-path datasets --mujoco-gl glfw --n-states 800 --out /tmp/tv_oc_smoke
.venv/bin/python - <<'PY'
import numpy as np, pandas as pd
df = pd.read_parquet("/tmp/tv_oc_smoke/fb_values.parquet")
print("cols:", list(df.columns))
print(df["outcome"].value_counts())
st = np.load("/tmp/tv_oc_smoke/training_states.npz")
print("npz keys:", list(st.keys()), "outcome shape:", st["outcome"].shape,
      st["outcome"].dtype)
assert "outcome" in df.columns and set(df["outcome"].unique()) <= {
    "success_bound", "fail_bound"}
assert st["outcome"].shape == (800, 5)
print("OK")
PY
```
Expected: `[training_value:fb] 800 states x 5 tasks`; `outcome` column present with both classes (or at least valid labels); npz has `outcome` of shape `(800, 5)`; `OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/training_value_profile.py
git commit -m "feat(training_value): FB pass — per-(state,task) bounded-reach outcome label + npz storage"
```

---

## Task 3: GCIQL pass — attach stored outcome (`scripts/training_value_profile.py`)

**Files:**
- Modify: `scripts/training_value_profile.py`

- [ ] **Step 1: Read the outcome matrix in `_gciql`**

In `_gciql`, after the existing `region = [str(r) for r in st["region"]]` line, add a guarded read:

```python
    if "outcome" not in st:
        raise SystemExit("training_states.npz has no 'outcome' — re-run the "
                         "FB pass (Task 2) before the GCIQL pass.")
    outcome = np.asarray(st["outcome"], dtype=bool)  # [n_states, n_tasks]
```

- [ ] **Step 2: Attach the outcome column to GCIQL rows**

Inside the `for ti in range(1, 6):` loop, replace the inner row append with:

```python
        oc = outcome[:, ti - 1]
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": region[i],
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]), "V": float(V[i]),
                         "outcome": "success_bound" if oc[i]
                         else "fail_bound"})
```

- [ ] **Step 3: Smoke the GCIQL pass (uses Task 2 smoke npz)**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/training_value_profile.py').read()); print('parse OK')"
D=$(ls -d RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd003_* | head -1)
MUJOCO_GL=glfw .venv-jax-cpu/bin/python scripts/training_value_profile.py --method gciql \
  --run-dir "$D" --step 1000000 --out /tmp/tv_oc_smoke --mujoco-gl glfw
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet("/tmp/tv_oc_smoke/gciql_values.parquet")
print("cols:", list(df.columns)); print(df["outcome"].value_counts())
assert "outcome" in df.columns
print("OK")
PY
```
Expected: `parse OK`; `[training_value:gciql] 800 states x 5 tasks`; `gciql_values.parquet` has an `outcome` column; `OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/training_value_profile.py
git commit -m "feat(training_value): GCIQL pass — attach stored per-(state,task) outcome label"
```

---

## Task 4: Aggregate — outcome split outputs (`scripts/training_value_aggregate.py`)

**Files:**
- Modify: `scripts/training_value_aggregate.py`

- [ ] **Step 1: Extend imports and add label constants**

Change the helper import to:

```python
from evals.training_value import (phase_spearman_table, outcome_spearman_table,
                                   value_discrimination)
```

After `REGIONS = ["reach", "grasp", "transport"]` add:

```python
OUTCOMES = ["success_bound", "fail_bound"]
HORIZON_DEFAULT = 200    # label provenance (see profile --horizon)
THRESH_DEFAULT = 0.04    # label provenance (see profile --thresh)
```

- [ ] **Step 2: Add the outcome-split block in `aggregate`**

In `aggregate`, immediately before the final
`print(f"[training_value_aggregate] -> {out}")`, insert:

```python
    # 5-8. Outcome-split outputs (skip cleanly on legacy dirs w/o 'outcome').
    if not all("outcome" in df.columns for df, _, _ in methods):
        print("[training_value_aggregate] no 'outcome' column — skipping "
              "outcome-split outputs (re-run profile passes).")
    else:
        # 5. Spearman rho(V,-d) by phase x outcome.
        rows = []
        for df, vc, label in methods:
            tab = outcome_spearman_table(df, value_col=vc,
                                         group_cols=group_cols)
            for _, r in tab.iterrows():
                rows.append({"method": label, "region": r["region"],
                             "outcome": r["outcome"],
                             "rho_mean": r["rho_mean"],
                             "rho_std": r["rho_std"]})
        spo = pd.DataFrame(rows)
        spo.to_parquet(out / "spearman_outcome_table.parquet")
        mdo = ["# Training-state value gradient by phase x outcome: "
               "Spearman rho(V, -d)",
               f"_n_seeds={n_seeds}; success-bound = cube within "
               f"{THRESH_DEFAULT} m of goal within {HORIZON_DEFAULT} steps. "
               "Pathology replicates if rho is NOT higher for success_bound._",
               "",
               "| method | outcome | " + " | ".join(REGIONS) + " |",
               "| :--- | :--- | " + " | ".join([":---"] * len(REGIONS)) + " |"]
        for label in [m[2] for m in methods]:
            for oc in OUTCOMES:
                cells = [label, oc]
                for rg in REGIONS:
                    s = spo[(spo.method == label) & (spo.region == rg)
                            & (spo.outcome == oc)]
                    cells.append(f"{s['rho_mean'].iloc[0]:.3f}"
                                 if len(s) else "—")
                mdo.append("| " + " | ".join(cells) + " |")
        (out / "spearman_outcome_table.md").write_text("\n".join(mdo) + "\n")

        # 6. Outcome-discrimination: AUC + mean dV.
        rows = []
        for df, vc, label in methods:
            disc = value_discrimination(df, value_col=vc,
                                        group_cols=group_cols)
            for _, r in disc.iterrows():
                rows.append({"method": label, "region": r["region"],
                             "auc": r["auc"], "mean_dV": r["mean_dV"],
                             "n_success": r["n_success"],
                             "n_fail": r["n_fail"]})
        disc_df = pd.DataFrame(rows)
        disc_df.to_parquet(out / "value_discrimination.parquet")
        mdd = ["# Value outcome-discrimination on training states",
               "_AUC of value separating success-bound vs fail-bound "
               "(0.5 = none); (dV) = z-scored mean value gap_", "",
               "| method | "
               + " | ".join(f"{rg} AUC (dV)" for rg in REGIONS) + " |",
               "| :--- | " + " | ".join([":---"] * len(REGIONS)) + " |"]
        for label in [m[2] for m in methods]:
            cells = [label]
            for rg in REGIONS:
                s = disc_df[(disc_df.method == label) & (disc_df.region == rg)]
                if len(s) and np.isfinite(s["auc"].iloc[0]):
                    cells.append(f"{s['auc'].iloc[0]:.2f} "
                                 f"({s['mean_dV'].iloc[0]:+.2f})")
                else:
                    cells.append("—")
            mdd.append("| " + " | ".join(cells) + " |")
        (out / "value_discrimination.md").write_text("\n".join(mdd) + "\n")

        # 7. value-vs-distance by outcome (rows=method, cols=phase).
        n_m = len(methods)
        fig, axes = plt.subplots(n_m, len(REGIONS),
                                 figsize=(5 * len(REGIONS), 4 * n_m),
                                 squeeze=False)
        for mi, (df, vc, label) in enumerate(methods):
            for ax, rg in zip(axes[mi], REGIONS):
                for oc, color in (("success_bound", "tab:green"),
                                  ("fail_bound", "tab:red")):
                    sub = df[(df.region == rg) & (df.outcome == oc)]
                    if len(sub) < 20:
                        continue
                    v = sub[vc].to_numpy(float)
                    vz = (v - v.mean()) / (v.std() + 1e-9)
                    edges = np.unique(np.quantile(sub["d"],
                                                  np.linspace(0, 1, 8)))
                    if len(edges) < 3:
                        continue
                    ctr = 0.5 * (edges[:-1] + edges[1:])
                    idx = np.clip(np.digitize(sub["d"], edges) - 1,
                                  0, len(ctr) - 1)
                    mv = pd.DataFrame({"b": idx, "vz": vz}).groupby(
                        "b")["vz"].mean()
                    ax.plot(ctr[mv.index], mv.values, marker="o",
                            color=color, label=oc)
                ax.set_title(f"{label} — {rg}")
                ax.set_xlabel("cube -> goal distance")
                ax.set_ylabel("z-scored value")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        fig.suptitle("Training states: value vs distance, by phase and outcome")
        fig.tight_layout()
        fig.savefig(out / "value_vs_dist_by_outcome.png", dpi=130)
        plt.close(fig)

        # 8. Outcome composition (success-bound fraction per phase) — sanity.
        comp_df = methods[0][0].assign(
            _succ=(methods[0][0]["outcome"] == "success_bound").astype(float))
        by_region = (comp_df.groupby("region")["_succ"].agg(["mean", "size"])
                     .reindex(REGIONS))
        by_region.reset_index().rename(
            columns={"mean": "frac_success_bound", "size": "n"}
        ).to_parquet(out / "outcome_composition.parquet")
        mdc = ["# Success-bound fraction on training states (per phase)",
               "_fraction of (state x task) rows that are success_bound_", "",
               "| region | frac success_bound | n |", "| :--- | :--- | :--- |"]
        for rg in REGIONS:
            if rg in by_region.index and np.isfinite(by_region.loc[rg, "mean"]):
                mdc.append(f"| {rg} | {by_region.loc[rg, 'mean']:.3f} | "
                           f"{int(by_region.loc[rg, 'size'])} |")
            else:
                mdc.append(f"| {rg} | — | 0 |")
        (out / "outcome_composition.md").write_text("\n".join(mdc) + "\n")
        print("\n".join(mdo))
        print("\n".join(mdd))
```

- [ ] **Step 3: Smoke the aggregate on the Task 2/3 smoke outputs**

```bash
.venv/bin/python scripts/training_value_aggregate.py --root /tmp/tv_oc_smoke --out /tmp/tv_oc_smoke/aggregate 2>&1 | tail -25
ls /tmp/tv_oc_smoke/aggregate
```
Expected: prints the per-phase×outcome Spearman table and the discrimination table; `aggregate/` contains `spearman_outcome_table.{md,parquet}`, `value_discrimination.{md,parquet}`, `value_vs_dist_by_outcome.png`, `outcome_composition.{md,parquet}` (plus the pre-existing `spearman_table.*`, `value_vs_dist.png`, `phase_composition.png`, `scene_*.png`).

- [ ] **Step 4: Commit**

```bash
git add scripts/training_value_aggregate.py
git commit -m "feat(training_value): aggregate — outcome-split spearman, AUC/dV discrimination, by-outcome curves, composition"
```

---

## Task 5: Real multi-seed run + verification

**Files:** none modified (run + verify).

- [ ] **Step 1: Regenerate the 6 seed pairs (FB then GCIQL per pair)**

The old `analysis/training_value_multiseed/p*/` lack the `outcome` column, so
recreate the dir (it is untracked scratch; the per-phase outputs regenerate
identically alongside the new ones):

```bash
ROOT=analysis/training_value_multiseed
rm -rf "$ROOT" && mkdir -p "$ROOT"
i=0
for s in 3 4 5 6 7 8; do
  P="$ROOT/p$i"
  FB=$(ls -d RESULTS/Factored-FB-cube-run/*__s${s} | head -1)
  GC=$(ls -d RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd00${s}_* | head -1)
  MUJOCO_GL=glfw .venv/bin/python scripts/training_value_profile.py --method fb \
    --config "$FB/.hydra/config.yaml" --checkpoint "$FB/checkpoints/final.pt" \
    --data-path datasets --mujoco-gl glfw --n-states 20000 --out "$P"
  MUJOCO_GL=glfw .venv-jax-cpu/bin/python scripts/training_value_profile.py --method gciql \
    --run-dir "$GC" --step 1000000 --out "$P" --mujoco-gl glfw
  i=$((i+1))
done
```
Expected: six `p0..p5` dirs, each with `fb_values.parquet`, `gciql_values.parquet`, `phase_composition.parquet`, `training_states.npz`; each FB print `20000 states x 5 tasks`.

- [ ] **Step 2: Aggregate + read the outcome tables**

```bash
.venv/bin/python scripts/training_value_aggregate.py --root analysis/training_value_multiseed
cat analysis/training_value_multiseed/aggregate/spearman_outcome_table.md
cat analysis/training_value_multiseed/aggregate/value_discrimination.md
cat analysis/training_value_multiseed/aggregate/outcome_composition.md
```
Expected: per-phase×outcome Spearman table (FB-policy / FB-data / GCIQL × success/fail), the AUC/ΔV discrimination table, and the success-bound composition. Interpretation: if AUC ≈ 0.5 and success_bound ρ is **not** above fail_bound ρ, the outcome-discrimination pathology replicates on training data.

- [ ] **Step 3: Full test suite + commit boundary check**

Run: `.venv/bin/python -m pytest tests/test_training_value.py -q`
Expected: 6 passed.
Run: `git status --short`
Expected: only `analysis/` (untracked) changes remain; no tracked-file diffs left uncommitted.

- [ ] **Step 4: Commit the plan doc**

```bash
git add docs/superpowers/plans/2026-05-22-training-value-outcome-split.md
git commit -m "docs(training-value): implementation plan — outcome-split (T1 mirror)"
```

---

## Self-review notes

- **Spec coverage:** outcome label = bounded-reach H=200/4cm (Task 2 `_outcome_labels` + `horizon_reach_label`); per (state, task) (per-task loop + `[n_states, n_tasks]` matrix); GCIQL reads stored label (Task 3, no recompute); ρ(V,−d) by phase×outcome (Task 4 `spearman_outcome_table`); AUC + mean ΔV discrimination (Task 4 `value_discrimination`); by-outcome curves + outcome composition (Task 4); single-dir & multi-seed pooling (reuses `_load_pooled`/`group_cols`); re-run 6 seeds (Task 5). All covered.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type/name consistency:** `horizon_reach_label(d_future, thresh, horizon)`, `outcome_spearman_table(df, value_col, group_cols)`, `value_discrimination(df, value_col, group_cols)` used identically in helpers, profile, and aggregate; `outcome` parquet column is the string `"success_bound"`/`"fail_bound"` everywhere; npz `outcome` key is the bool `[n_states, n_tasks]` matrix written in Task 2 and read in Task 3; `_sample_states` 6-tuple return matches the `_fb` unpack.
- **Open risk (flagged in spec):** degenerate success_bound class on play data — `outcome_composition` + `n_success`/`n_fail` in the discrimination table surface it; ρ/AUC are nan-safe for empty classes.
```