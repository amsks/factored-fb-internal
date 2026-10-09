# T1 Spearman Metric + Data-Driven Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uninterpretable T1 OLS-slope metric with a scale-free Spearman rho, and replace the hardcoded `story.md` Readout with per-probe verdicts computed from the aggregate data.

**Architecture:** Two code changes plus a re-run. (1) `evals/representation_profile.py`: `_slope`→`_spearman`, `probe_value_landscape` renames `slope_V_negd`→`rho_V_negd` and gains a degeneracy guard. (2) `scripts/representation_profile_aggregate.py`: rename the T1 column everywhere and replace the hardcoded Readout with pure verdict functions + a synthesis line driven by named threshold constants. Then re-run the 7-seed sweep (the `value_landscape.parquet` schema changed) + aggregate + paper fragment.

**Tech Stack:** Python, numpy, pandas (Spearman = Pearson on `pandas.Series.rank()` — no scipy), pytest.

Spec: `docs/superpowers/specs/2026-05-18-t1-spearman-data-driven-readout-design.md`

---

## Verified codebase facts

- `evals/representation_profile.py:103-107` — `_slope(y, x)` OLS helper.
- `evals/representation_profile.py:110-137` — `probe_value_landscape`; emits
  `slope_V_negd`, guard `if m.sum() >= 2`.
- `scripts/representation_profile_aggregate.py:60` — `t1` groupby uses
  `"slope_V_negd"`.
- `scripts/representation_profile_aggregate.py:117-118` — `succ`/`fail`
  series use `"slope_V_negd"`.
- `scripts/representation_profile_aggregate.py:119-150` — `lines` list;
  `132-134` T1 section; `140-148` hardcoded Readout string.
- `scripts/representation_profile_aggregate.py:45-46` — `_ms(series)` →
  `f"{series.mean():.3g} ± {series.std(ddof=0):.2g}"` (pandas mean/std skip
  nan).
- `tests/test_representation_profile.py:123-133` — `_fake_ep`.
- `tests/test_representation_profile.py:136-148` —
  `test_value_landscape_slope_success_vs_fail`.
- `tests/test_representation_profile.py:251-257` — aggregate-test
  `value_landscape.parquet` fixture (writes `slope_V_negd`).
- `_StubModel`/`_StubAgent` make `V = obs[:,0] + 0.5` (agent action 0,
  ensemble mean of `base` and `base+1`). So `_fake_ep(o,d,f)` gives
  `V = f(d) + 0.5` per step.

---

## Task 1: T1 Spearman metric

**Files:**
- Modify: `evals/representation_profile.py:103-137`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Rewrite the T1 unit tests (failing)**

In `tests/test_representation_profile.py`, replace the whole function
`test_value_landscape_slope_success_vs_fail` (currently lines 136-148):

```python
def test_value_landscape_slope_success_vs_fail():
    m, ag = _StubModel(), _StubAgent()
    z = torch.tensor([1.0, 0.0, 0.0])
    # success: V rises as d -> 0 (slope wrt -d > 0); fail: V flat
    eps = [
        _fake_ep("success", [0.30, 0.20, 0.10, 0.02], lambda x: 1.0 - x),
        _fake_ep("transport_fail", [0.30, 0.20, 0.10, 0.05], lambda x: 0.5),
    ]
    per_ep, per_step = rp.probe_value_landscape(m, ag, eps, z)
    s = per_ep.set_index("outcome")["slope_V_negd"]
    assert s.loc["success"] > 0.5
    assert abs(s.loc["transport_fail"]) < 1e-6
    assert set(per_step.columns) >= {"outcome", "d", "V"}
```

with these two functions:

```python
def test_value_landscape_rho_success_vs_fail():
    m, ag = _StubModel(), _StubAgent()
    z = torch.tensor([1.0, 0.0, 0.0])
    # >=5 transport steps so the guard passes; success V strictly rises
    # as d -> 0 (monotone in -d => rho ~ +1); fail V is flat => nan.
    eps = [
        _fake_ep("success",
                 [0.30, 0.25, 0.20, 0.15, 0.10, 0.02], lambda x: 1.0 - x),
        _fake_ep("transport_fail",
                 [0.30, 0.25, 0.20, 0.15, 0.10, 0.05], lambda x: 0.5),
    ]
    per_ep, per_step = rp.probe_value_landscape(m, ag, eps, z)
    s = per_ep.set_index("outcome")["rho_V_negd"]
    assert s.loc["success"] > 0.99            # strictly monotone -> ~1
    assert np.isnan(s.loc["transport_fail"])  # flat V -> degenerate -> nan
    assert set(per_step.columns) >= {"outcome", "d", "V"}


def test_value_landscape_rho_guard_short_episode():
    m, ag = _StubModel(), _StubAgent()
    z = torch.tensor([1.0, 0.0, 0.0])
    # only 4 transport steps (< 5) -> guarded -> nan even though monotone
    eps = [_fake_ep("success", [0.30, 0.20, 0.10, 0.02], lambda x: 1.0 - x)]
    per_ep, _ = rp.probe_value_landscape(m, ag, eps, z)
    s = per_ep.set_index("outcome")["rho_V_negd"]
    assert np.isnan(s.loc["success"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k "value_landscape_rho" -v`
Expected: FAIL — `KeyError: 'rho_V_negd'` (column still named `slope_V_negd`).

- [ ] **Step 3: Replace `_slope` with `_spearman`**

In `evals/representation_profile.py`, replace the whole `_slope` function
(currently lines 103-107):

```python
def _slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x; nan if <2 distinct x."""
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])
```

with:

```python
def _spearman(y: np.ndarray, x: np.ndarray) -> float:
    """Spearman rho (Pearson on average ranks) of y vs x; nan if either
    side is constant or fewer than 2 points. Outlier-robust and
    scale-free, unlike an OLS slope on the unnormalised value V."""
    import pandas as pd

    y = np.asarray(y, np.float64)
    x = np.asarray(x, np.float64)
    if len(x) < 2 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])
```

- [ ] **Step 4: Update `probe_value_landscape` (guard + column rename)**

In `evals/representation_profile.py`, in `probe_value_landscape`, replace
this block (currently lines 127-136):

```python
        if m.sum() >= 2:
            slope = _slope(V[m], -d[m])
            v_secure = float(V[m][0])
            v_end = float(V[-1])
        else:
            slope, v_secure, v_end = float("nan"), float("nan"), float("nan")
        per_ep_rows.append({"episode": i, "outcome": ep["outcome"],
                            "slope_V_negd": slope, "V_at_secure": v_secure,
                            "V_at_end": v_end,
                            "n_transport_steps": int(m.sum())})
```

with:

```python
        n_t = int(m.sum())
        if (n_t >= 5 and np.ptp(d[m]) > 1e-9 and np.ptp(V[m]) > 1e-9):
            rho = _spearman(V[m], -d[m])
            v_secure = float(V[m][0])
            v_end = float(V[-1])
        else:
            rho, v_secure, v_end = float("nan"), float("nan"), float("nan")
        per_ep_rows.append({"episode": i, "outcome": ep["outcome"],
                            "rho_V_negd": rho, "V_at_secure": v_secure,
                            "V_at_end": v_end,
                            "n_transport_steps": n_t})
```

Also update the docstring line (currently line 111-112) from:

```python
    """Per-episode slope of V w.r.t. (-d) over the transport region,
    plus a long-form (outcome, d, V) frame for binned plots."""
```

to:

```python
    """Per-episode Spearman rho of V vs (-d) over the transport region
    (nan unless >=5 steps with non-degenerate d and V), plus a long-form
    (outcome, d, V) frame for binned plots."""
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -q`
Expected: PASS (13 tests; the two new T1 tests green, no regressions).

- [ ] **Step 6: Commit**

```bash
git add evals/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): T1 Spearman rho(V,-d) with degeneracy guard"
```

---

## Task 2: Data-driven Readout in the aggregate

**Files:**
- Modify: `scripts/representation_profile_aggregate.py`
- Test: `tests/test_representation_profile.py`

- [ ] **Step 1: Add failing unit tests for the verdict helpers**

Append to `tests/test_representation_profile.py`:

```python
def test_aggregate_verdict_helpers():
    import importlib
    agg = importlib.import_module("scripts.representation_profile_aggregate")
    # T1: success monotone, fail flat -> SUPPORTS
    assert agg._verdict_t1(0.40, 0.02)[0] == "SUPPORTS"
    # T1: fail rises more than success -> CONTRADICTS
    assert agg._verdict_t1(0.05, 0.40)[0] == "CONTRADICTS"
    # T1: nan in -> INSUFFICIENT DATA
    assert agg._verdict_t1(float("nan"), 0.1)[0] == "INSUFFICIENT DATA"
    # T2: sparse + z mis-pointed -> SUPPORTS
    assert agg._verdict_t2(0.02, 0.20)[0] == "SUPPORTS"
    assert agg._verdict_t2(0.30, 0.80)[0] == "WEAK"
    # T3: weak resolution -> SUPPORTS; strong -> RESOLVES
    assert agg._verdict_t3(0.30, 0.65)[0] == "SUPPORTS"
    assert agg._verdict_t3(0.80, 0.95)[0] == "RESOLVES"
    # T4: fail far more off-support -> SUPPORTS; similar -> NEUTRAL
    assert agg._verdict_t4(3.0, 1.0)[0] == "SUPPORTS"
    assert agg._verdict_t4(2.2, 2.1)[0] == "NEUTRAL"
    assert agg._verdict_t4(float("nan"), 2.0)[0] == "INSUFFICIENT DATA"
    # synthesis names supporting probes
    syn = agg._synthesis({"T1": ("SUPPORTS", ""), "T2": ("SUPPORTS", ""),
                           "T3": ("WEAK", ""), "T4": ("NEUTRAL", "")})
    assert "T1" in syn and "T2" in syn and syn.startswith("Synthesis:")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k aggregate_verdict_helpers -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_verdict_t1'`.

- [ ] **Step 3: Add threshold constants + verdict helpers**

In `scripts/representation_profile_aggregate.py`, immediately after the
`_SEED_RE = re.compile(r"s(\d+)_final")` line (currently line 127), add:

```python
# --- Readout verdict thresholds (heuristic; tune here) -------------------
T1_RHO_MIN = 0.15
T1_RHO_GAP = 0.10
T2_SPARSE_FRAC = 0.05
T2_TOPK_AT_GOAL = 0.5
T3_R2_FAIL = 0.5
T3_ACC_FAIL = 0.75
T3_R2_RESOLVE = 0.7
T3_ACC_RESOLVE = 0.9
T4_OFFSUPPORT_RATIO = 1.25


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except TypeError:
        return True


def _verdict_t1(rho_succ, rho_fail):
    if _isnan(rho_succ) or _isnan(rho_fail):
        return "INSUFFICIENT DATA", "no valid transport-region episodes"
    if rho_succ >= T1_RHO_MIN and rho_succ - rho_fail >= T1_RHO_GAP:
        return "SUPPORTS", "success value rises toward goal; fail flatter"
    if rho_fail - rho_succ >= T1_RHO_GAP:
        return "CONTRADICTS", "fail value rises more than success"
    return "WEAK", "no clear success/fail gradient separation"


def _verdict_t2(frac, topk_at_goal):
    if _isnan(frac) or _isnan(topk_at_goal):
        return "INSUFFICIENT DATA", "no relabel/z-decoding stats"
    if frac < T2_SPARSE_FRAC and topk_at_goal < T2_TOPK_AT_GOAL:
        return "SUPPORTS", "data goal-sparse; z's top states off-goal"
    return "WEAK", "data not strongly goal-sparse or z points at goal"


def _verdict_t3(r2, acc):
    if _isnan(r2) or _isnan(acc):
        return "INSUFFICIENT DATA", "no B-resolution stats"
    if r2 < T3_R2_FAIL and acc < T3_ACC_FAIL:
        return "SUPPORTS", "B barely resolves placement distance"
    if r2 >= T3_R2_RESOLVE and acc >= T3_ACC_RESOLVE:
        return "RESOLVES", "B cleanly resolves placement (contradicts)"
    return "WEAK", "B partially resolves placement"


def _verdict_t4(fail_nn, succ_nn):
    if _isnan(fail_nn) or _isnan(succ_nn):
        return "INSUFFICIENT DATA", "missing transport-region coverage"
    if fail_nn >= T4_OFFSUPPORT_RATIO * succ_nn:
        return "SUPPORTS", "transport-fails sit further off data support"
    return "NEUTRAL", "fails not meaningfully more off-support"


def _synthesis(verdicts) -> str:
    supports = [k for k, (v, _) in verdicts.items() if v == "SUPPORTS"]
    contra = [k for k, (v, _) in verdicts.items()
              if v in ("CONTRADICTS", "RESOLVES")]
    if supports:
        s = (", ".join(supports)
             + " support the representation-failure hypothesis")
    else:
        s = "no probe supports the representation-failure hypothesis"
    if contra:
        s += "; " + ", ".join(contra) + " argue against it"
    return "Synthesis: " + s + "."
```

(`np` is already imported at the top of the file.)

- [ ] **Step 4: Run the verdict-helper test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -k aggregate_verdict_helpers -v`
Expected: PASS.

- [ ] **Step 5: Rename the T1 column + wire the data-driven Readout**

In `scripts/representation_profile_aggregate.py`:

(a) Line 60 — change:

```python
    t1 = (vl.groupby(["task", "outcome"])["slope_V_negd"]
```

to:

```python
    t1 = (vl.groupby(["task", "outcome"])["rho_V_negd"]
```

(b) Replace the `succ`/`fail`/`lines`/`story.md` block (currently lines
117-150, from `succ = vl[...]` through `(out / "story.md").write_text(...)`)
with:

```python
    succ = vl[vl.outcome == "success"]["rho_V_negd"]
    fail = vl[vl.outcome == "transport_fail"]["rho_V_negd"]
    rho_s, rho_f = float(succ.mean()), float(fail.mean())
    frac = float(zd["relabel_pos_frac"].mean())
    tkg = float(zd["topk_pct_at_goal"].mean())
    r2 = float(br["r2"].mean())
    acc = float(br["placed_vs_near_acc"].mean())
    cvt = cv[cv.region == "transport"]
    fail_nn = float(cvt[cvt.outcome == "transport_fail"]["nn_dist"].mean())
    succ_nn = float(cvt[cvt.outcome == "success"]["nn_dist"].mean())

    verdicts = {
        "T1": _verdict_t1(rho_s, rho_f),
        "T2": _verdict_t2(frac, tkg),
        "T3": _verdict_t3(r2, acc),
        "T4": _verdict_t4(fail_nn, succ_nn),
    }
    lines = [
        "# FB representation-failure — cross-seed story", "",
        f"Seeds: {', '.join('s'+str(s) for s in sorted(vl['seed'].unique()))}",
        "",
        "## Sparsity (T2)",
        f"- offline goal-state fraction: {_ms(zd['relabel_pos_frac'])}",
        f"- top-k B·z mean cube→goal dist: {_ms(zd['topk_mean_d'])}",
        f"- top-k % actually at goal: {_ms(zd['topk_pct_at_goal'])}",
        "",
        "## B resolution (T3)",
        f"- ridge R²(B→d): {_ms(br['r2'])}",
        f"- placed-vs-near-miss acc: {_ms(br['placed_vs_near_acc'])}",
        "",
        "## Value gradient (T1)",
        f"- Spearman ρ(V vs −d) success: {_ms(succ)}",
        f"- Spearman ρ(V vs −d) transport-fail: {_ms(fail)}",
        "",
        "## Coverage (T4)",
        f"- transport-region nn_dist (transport_fail): "
        f"{_ms(cv[(cv.region=='transport') & (cv.outcome=='transport_fail')]['nn_dist'])}",
        "",
        "## Readout",
        "_Heuristic verdicts (thresholds in "
        "representation_profile_aggregate.py):_",
        f"- T1 value gradient: rho_success={rho_s:.3g}, "
        f"rho_fail={rho_f:.3g} -> {verdicts['T1'][0]} ({verdicts['T1'][1]})",
        f"- T2 sparsity: relabel_pos_frac={frac:.3g}, "
        f"topk_pct_at_goal={tkg:.3g} -> {verdicts['T2'][0]} "
        f"({verdicts['T2'][1]})",
        f"- T3 B-resolution: r2={r2:.3g}, placed_vs_near_acc={acc:.3g} "
        f"-> {verdicts['T3'][0]} ({verdicts['T3'][1]})",
        f"- T4 coverage: fail={fail_nn:.3g}, success={succ_nn:.3g} "
        f"(transport) -> {verdicts['T4'][0]} ({verdicts['T4'][1]})",
        _synthesis(verdicts),
    ]
    (out / "story.md").write_text("\n".join(lines))
```

- [ ] **Step 6: Update the aggregate-integration test fixture + assertions**

In `tests/test_representation_profile.py`, in
`test_aggregate_builds_tables_and_story`, replace the
`value_landscape.parquet` fixture (currently lines 251-257):

```python
        pd.DataFrame([{"task": "t1", "outcome": "success",
                       "slope_V_negd": 1.0, "V_at_secure": 0.1,
                       "V_at_end": 0.9, "n_transport_steps": 5},
                      {"task": "t1", "outcome": "transport_fail",
                       "slope_V_negd": 0.0, "V_at_secure": 0.5,
                       "V_at_end": 0.5, "n_transport_steps": 5}]
                     ).to_parquet(d / "value_landscape.parquet")
```

with:

```python
        pd.DataFrame([{"task": "t1", "outcome": "success",
                       "rho_V_negd": 0.9, "V_at_secure": 0.1,
                       "V_at_end": 0.9, "n_transport_steps": 5},
                      {"task": "t1", "outcome": "transport_fail",
                       "rho_V_negd": 0.0, "V_at_secure": 0.5,
                       "V_at_end": 0.5, "n_transport_steps": 5}]
                     ).to_parquet(d / "value_landscape.parquet")
```

Then replace the tail of that test (currently lines 275-277):

```python
    txt = (out / "story.md").read_text()
    assert "## Readout" in txt
    assert "0.02" in txt          # relabel_pos_frac mean filled in
```

with:

```python
    txt = (out / "story.md").read_text()
    assert "## Readout" in txt
    assert "0.02" in txt          # relabel_pos_frac mean filled in
    assert "T1 value gradient" in txt and "Synthesis:" in txt
    # rho_succ=0.9, rho_fail=0.0 -> SUPPORTS; frac 0.02 & topk 0.1 -> SUPPORTS
    assert "SUPPORTS" in txt
    assert "Reach ~100%" not in txt   # hardcoded template is gone
```

- [ ] **Step 7: Run the full repr suite to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -q`
Expected: PASS (15 tests).

- [ ] **Step 8: Commit**

```bash
git add scripts/representation_profile_aggregate.py tests/test_representation_profile.py
git commit -m "feat(repr_profile): data-driven per-probe Readout verdicts"
```

---

## Task 3: Full verification + re-run acceptance

**Files:** none (operational).

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (was 45 + 13 repr; now repr has +4 net new/renamed →
the suite passes with no failures; exact count not asserted, zero failures
is the gate).

- [ ] **Step 2: Re-run the 7-seed per-seed sweep**

(The `value_landscape.parquet` schema changed: `slope_V_negd`→`rho_V_negd`.)

```bash
: > /tmp/rp_sweep2.log
for s in 3 4 5 6 7 8 10; do
  RUN=$(ls -d results/Factored-FB-cube-run/*__fb_flowbc__s${s})
  .venv/bin/python scripts/representation_profile.py \
    --config "$RUN/.hydra/config.yaml" \
    --checkpoint "$RUN/checkpoints/final.pt" \
    --out analysis/representation_profile/s${s}_final \
    --data-path datasets --device cpu --mujoco-gl glfw --n-episodes 10 \
    >> /tmp/rp_sweep2.log 2>&1 \
    && echo "seed s${s} OK" >> /tmp/rp_sweep2.log \
    || echo "seed s${s} FAILED" >> /tmp/rp_sweep2.log
done
```

Expected: `grep 'seed s.* OK' /tmp/rp_sweep2.log` shows 7 lines; each
`analysis/representation_profile/s*_final/value_landscape.parquet` now has
a `rho_V_negd` column (verify:
`.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('analysis/representation_profile/s3_final/value_landscape.parquet').columns.tolist())"`).

- [ ] **Step 3: Re-aggregate + rebuild paper fragment**

```bash
.venv/bin/python scripts/representation_profile_aggregate.py
.venv/bin/python scripts/build_paper_fragment.py --paper-subdir fb-cube-failure
```

Expected: `analysis/representation_profile/aggregate/story.md` "## Readout"
contains the four `-> <VERDICT>` lines and a `Synthesis:` line computed from
the data (no `Reach ~100%` template prose); `T1_value_gradient.parquet`
present; `PAPER/fb-cube-failure/` regenerated.

- [ ] **Step 4: Report the real cross-seed verdicts**

Read `analysis/representation_profile/aggregate/story.md` and report the
per-probe verdicts + synthesis to the user (no commit of analysis artifacts
unless the user asks).

---

## Self-review

- **Spec coverage:** T1 Spearman + guard (Task 1); column rename everywhere
  (Task 1 Step 4, Task 2 Step 5a/5b); data-driven verdicts + constants +
  nan-safety + synthesis (Task 2 Step 3/5); test updates incl. guard test
  and verdict tests (Task 1 Step 1, Task 2 Steps 1/6); re-run because schema
  changed (Task 3). All spec sections mapped.
- **Placeholder scan:** none — every code step shows complete code and exact
  paths/commands.
- **Type consistency:** column `rho_V_negd` used identically in
  `probe_value_landscape`, the groupby (5a), the `succ`/`fail` series (5b),
  and both test fixtures. Helper names `_spearman`, `_verdict_t1.._t4`,
  `_synthesis`, `_isnan` consistent between definition and test call sites.
  `_verdict_*` return `(verdict, reason)` tuples consistently; `_synthesis`
  consumes `{k: (verdict, reason)}` consistently.
