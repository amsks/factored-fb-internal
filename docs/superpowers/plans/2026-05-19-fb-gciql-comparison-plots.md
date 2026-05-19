# FB-vs-GCIQL Comparison Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four data-driven FB-vs-GCIQL comparison figures (value-vs-distance, ρ box, outcome funnel, coverage) to the paper.

**Architecture:** GCIQL profiler gains a per-step `value_steps.parquet` (FB-schema-aligned); the comparison aggregate loads FB+GCIQL per-seed parquets and renders four `cmp_*.png`; the paper generator globs every `cmp_*.png` into the user's "FB vs GCIQL" subsection. Then re-run the GCIQL sweep (~6 min) and regenerate.

**Tech Stack:** Python, numpy, pandas, matplotlib (Agg); JAX/flax + vendored OGBench for the GCIQL re-run only; pytest.

Spec: `docs/superpowers/specs/2026-05-19-fb-gciql-comparison-plots-design.md`

---

## Verified codebase facts

- `scripts/gciql_profile.py`: `episodes_to_frames` (def L39) ends L85-87
  `cov = probe_coverage(...)` / `cov.insert(0,"task",...)` /
  `return (pd.DataFrame(vl_rows), cov, pd.DataFrame(fun_rows))`. Loop
  builds `vl_rows`/`fun_rows`/`cov_eps`; has per-ep `d`, `V`, `tm`
  (transport_mask bool), `outcome`. `main()` L213-216:
  `vl, cov, fun = episodes_to_frames(...)` then three `to_parquet`.
- FB per-seed `analysis/representation_profile/s*_final/value_steps.parquet`
  schema: `task, episode, outcome, d, V, transport` (7 seeds, exist).
  FB per-seed `value_landscape.parquet` has per-episode
  `rho_V_negd, outcome, task`. FB per-seed `coverage.parquet`:
  `task, outcome, region, nn_dist` (col order may differ).
- GCIQL per-seed exist (10 seeds):
  `analysis/gciql_profile/s*_final/{value_landscape,coverage,phase_funnel}.parquet`.
- `scripts/gciql_profile_aggregate.py`: `_load(root,name)` globs
  `Path(root)/s*_final/<name>.parquet`, tags `seed`. `aggregate(root,
  out, fb_aggregate=None)` (L49); calls `_comparison(out/"comparison",
  vl, cvt, fb_aggregate, rho_s, rho_f, fail_nn, succ_nn, verdicts)`
  (L116). `_comparison` (L121) reads only FB aggregate
  `T1_value_gradient.parquet`, writes `fb_vs_gciql.parquet`,
  `cmp_value_rho.png`, `comparison.md`. `main()` L168 has `--root`,
  `--out`, `--fb-aggregate`.
- `evals/paper_fragment.py`: `gcmp_block` built L179-202; embeds only
  `cmp_value_rho.png` via `_fig("figures","cmp_value_rho.png",...)`.
  Helpers `_fig(figdir_rel,name,present,caption)`, `_tex(s)`,
  `df_to_booktabs(df,caption,label)`, `_story_to_latex(md)` exist.
- GCIQL CPU venv: `.venv-jax-cpu/bin/python`. Sweep ≈ 5.6 min.

---

## Task 1: GCIQL per-step `value_steps.parquet`

**Files:**
- Modify: `scripts/gciql_profile.py`
- Test: `tests/test_gciql_profile.py`

- [ ] **Step 1: Update tests to the 4-tuple + value_steps schema**

In `tests/test_gciql_profile.py`, in
`test_episodes_to_frames_t1_t4_funnel`, change the call line:

```python
    vl, cov, fun = gp.episodes_to_frames(eps, Thresholds(),
                                         ref_obs=np.zeros((10, 4)))
```

to:

```python
    vl, cov, fun, vs = gp.episodes_to_frames(eps, Thresholds(),
                                             ref_obs=np.zeros((10, 4)))
    assert set(vs.columns) == {"task", "episode", "outcome", "d", "V",
                               "transport"}
    assert len(vs) == 6                       # one row per step
    assert vs["transport"].dtype == bool
```

In `test_episodes_to_frames_attributes_transport_fail_with_real_eff`,
change:

```python
    vl, _, fun = gp.episodes_to_frames([ep], Thresholds(),
                                       ref_obs=np.zeros((10, 4)))
```

to:

```python
    vl, _, fun, _ = gp.episodes_to_frames([ep], Thresholds(),
                                          ref_obs=np.zeros((10, 4)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -k episodes_to_frames -q`
Expected: FAIL — `ValueError: not enough values to unpack (expected 4, got 3)`.

- [ ] **Step 3: Emit `value_steps` from `episodes_to_frames`**

In `scripts/gciql_profile.py`, replace this block (currently L81-87):

```python
        cov_eps.append({"obs": ep["obs"], "outcome": outcome,
                        "grip": ep["grip"], "cube": ep["cube"],
                        "table_z": ep["table_z"],
                        "transport_mask": tm})
    cov = probe_coverage(cov_eps, ref_obs, thr)
    cov.insert(0, "task", episodes[0]["task"] if episodes else "")
    return (pd.DataFrame(vl_rows), cov, pd.DataFrame(fun_rows))
```

with:

```python
        cov_eps.append({"obs": ep["obs"], "outcome": outcome,
                        "grip": ep["grip"], "cube": ep["cube"],
                        "table_z": ep["table_z"],
                        "transport_mask": tm})
        for dd, vv, mm in zip(d, V, tm):
            vs_rows.append({"task": ep["task"], "episode": ep["episode"],
                            "outcome": outcome, "d": float(dd),
                            "V": float(vv), "transport": bool(mm)})
    cov = probe_coverage(cov_eps, ref_obs, thr)
    cov.insert(0, "task", episodes[0]["task"] if episodes else "")
    vs = pd.DataFrame(vs_rows, columns=["task", "episode", "outcome",
                                        "d", "V", "transport"])
    return (pd.DataFrame(vl_rows), cov, pd.DataFrame(fun_rows), vs)
```

Also add `vs_rows` to the row-accumulator init. Find this line near the
top of `episodes_to_frames` (just before the `for ep in episodes:`
loop):

```python
    vl_rows, fun_rows, cov_eps = [], [], []
```

and change it to:

```python
    vl_rows, fun_rows, cov_eps, vs_rows = [], [], [], []
```

- [ ] **Step 4: Write `value_steps.parquet` in `main()`**

In `scripts/gciql_profile.py` `main()`, replace (currently L213-216):

```python
    vl, cov, fun = episodes_to_frames(episodes, Thresholds(), ref)
    vl.to_parquet(out / "value_landscape.parquet")
    cov.to_parquet(out / "coverage.parquet")
    fun.to_parquet(out / "phase_funnel.parquet")
```

with:

```python
    vl, cov, fun, vs = episodes_to_frames(episodes, Thresholds(), ref)
    vl.to_parquet(out / "value_landscape.parquet")
    cov.to_parquet(out / "coverage.parquet")
    fun.to_parquet(out / "phase_funnel.parquet")
    vs.to_parquet(out / "value_steps.parquet")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/gciql_profile.py tests/test_gciql_profile.py
git commit -m "feat(gciql_profile): emit per-step value_steps.parquet (FB schema)"
```

---

## Task 2: Four comparison plots in the aggregate

**Files:**
- Modify: `scripts/gciql_profile_aggregate.py`
- Test: `tests/test_gciql_profile.py`

- [ ] **Step 1: Add the failing comparison test**

Append to `tests/test_gciql_profile.py`:

```python
def _seed_steps(d, root, s, outcome, n=8):
    import numpy as np
    base = root / f"s{s}_final"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"task": "task1", "episode": 0, "outcome": outcome,
                   "rho_V_negd": 0.5 if outcome == "success" else 0.0,
                   "V_at_secure": 0.1, "V_at_end": 0.9,
                   "n_transport_steps": n}]
                 ).to_parquet(base / "value_landscape.parquet")
    pd.DataFrame({"task": ["task1"] * n, "episode": [0] * n,
                  "outcome": [outcome] * n,
                  "d": np.linspace(0.3, 0.05, n),
                  "V": np.linspace(0.0, 1.0, n),
                  "transport": [True] * n}
                 ).to_parquet(base / "value_steps.parquet")
    pd.DataFrame([{"task": "task1", "outcome": outcome,
                   "region": "transport", "nn_dist": 2.0}]
                 ).to_parquet(base / "coverage.parquet")
    pd.DataFrame([{"task": "task1", "episode": 0,
                   "furthest_phase": outcome, "outcome": outcome}]
                 ).to_parquet(base / "phase_funnel.parquet")


def test_comparison_emits_four_plots(tmp_path):
    import importlib
    agg = importlib.import_module("scripts.gciql_profile_aggregate")
    groot = tmp_path / "gciql"
    froot = tmp_path / "repr"
    for s in (1, 2):
        _seed_steps(__import__("pathlib").Path, groot, s, "success")
        _seed_steps(__import__("pathlib").Path, groot, s,
                    "transport_fail")
    for s in (1, 2):
        _seed_steps(__import__("pathlib").Path, froot, s, "success")
        _seed_steps(__import__("pathlib").Path, froot, s,
                    "transport_fail")
    fbagg = tmp_path / "fb_aggregate"
    fbagg.mkdir()
    pd.DataFrame([{"task": "task1", "outcome": "success",
                   "rho_V_negd": 0.13},
                  {"task": "task1", "outcome": "transport_fail",
                   "rho_V_negd": 0.23}]).to_parquet(
        fbagg / "T1_value_gradient.parquet")
    out = tmp_path / "gagg"
    agg.aggregate(groot, out, fb_aggregate=fbagg, fb_seed_root=froot)
    cmp = out / "comparison"
    for fn in ("cmp_value_vs_dist.png", "cmp_rho_box.png",
               "cmp_funnel.png", "cmp_coverage.png",
               "cmp_value_rho.png", "fb_vs_gciql.parquet",
               "comparison.md"):
        assert (cmp / fn).exists(), fn
```

Note: the helper appends two outcomes into the same seed dir; the second
call must not clobber the first. Fix `_seed_steps` to append rather than
overwrite by making the second outcome use `episode=1`. Replace the
`_seed_steps` body's frames to key on a per-call episode: change every
`"episode": 0` / `[0]*n` to use `ep` where
`ep = 0 if outcome == "success" else 1`, and concatenate with any
existing parquet. Concretely use this `_seed_steps` instead:

```python
def _seed_steps(_P, root, s, outcome, n=8):
    import numpy as np
    base = root / f"s{s}_final"
    base.mkdir(parents=True, exist_ok=True)
    ep = 0 if outcome == "success" else 1
    rho = 0.5 if outcome == "success" else 0.0

    def _app(name, df):
        p = base / name
        if p.exists():
            df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
        df.to_parquet(p)

    _app("value_landscape.parquet", pd.DataFrame([
        {"task": "task1", "episode": ep, "outcome": outcome,
         "rho_V_negd": rho, "V_at_secure": 0.1, "V_at_end": 0.9,
         "n_transport_steps": n}]))
    _app("value_steps.parquet", pd.DataFrame(
        {"task": ["task1"] * n, "episode": [ep] * n,
         "outcome": [outcome] * n, "d": np.linspace(0.3, 0.05, n),
         "V": np.linspace(0.0, 1.0, n), "transport": [True] * n}))
    _app("coverage.parquet", pd.DataFrame([
        {"task": "task1", "outcome": outcome, "region": "transport",
         "nn_dist": 2.0}]))
    _app("phase_funnel.parquet", pd.DataFrame([
        {"task": "task1", "episode": ep, "furthest_phase": outcome,
         "outcome": outcome}]))
```

(Use the corrected `_seed_steps`; the calls in
`test_comparison_emits_four_plots` already pass a placeholder first arg
which the body ignores — keep the calls as written.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -k comparison_emits_four -q`
Expected: FAIL — `TypeError: aggregate() got an unexpected keyword argument 'fb_seed_root'`.

- [ ] **Step 3: Thread `fb_seed_root` through `aggregate` + `main`**

In `scripts/gciql_profile_aggregate.py`, change the signature
(currently L49):

```python
def aggregate(root, out, fb_aggregate=None) -> None:
```

to:

```python
def aggregate(root, out, fb_aggregate=None, fb_seed_root=None) -> None:
```

Change the `_comparison(...)` call (currently L116) from:

```python
    if fb_aggregate is not None:
        _comparison(out / "comparison", vl, cvt, fb_aggregate,
                    rho_s, rho_f, fail_nn, succ_nn, verdicts)
```

to:

```python
    if fb_aggregate is not None:
        _comparison(out / "comparison", vl, cvt, fb_aggregate,
                    rho_s, rho_f, fail_nn, succ_nn, verdicts,
                    gciql_root=root, fb_seed_root=fb_seed_root)
```

In `main()`, after the `--fb-aggregate` argument add:

```python
    ap.add_argument("--fb-seed-root",
                    default=str(REPO_ROOT / "analysis"
                               / "representation_profile"))
```

and change the final `aggregate(...)` call from:

```python
    aggregate(root, Path(args.out) if args.out else root / "aggregate",
              fb_aggregate=fb if fb.exists() else None)
```

to:

```python
    fbsr = Path(args.fb_seed_root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate",
              fb_aggregate=fb if fb.exists() else None,
              fb_seed_root=fbsr if fbsr.exists() else None)
```

- [ ] **Step 4: Add the four plots to `_comparison`**

In `scripts/gciql_profile_aggregate.py`, change the `_comparison`
signature (currently L121-122):

```python
def _comparison(cdir, vl, cvt, fb_aggregate, g_rs, g_rf, g_fnn, g_snn,
                g_verd) -> None:
```

to:

```python
def _comparison(cdir, vl, cvt, fb_aggregate, g_rs, g_rf, g_fnn, g_snn,
                g_verd, gciql_root=None, fb_seed_root=None) -> None:
```

Then, immediately before the final `(cdir / "comparison.md").write_text(`
call in `_comparison`, insert this block:

```python
    def _seed_concat(rt, name):
        if rt is None:
            return None
        fr = []
        for p in sorted(Path(rt).glob(f"s*_final/{name}.parquet")):
            fr.append(pd.read_parquet(p))
        return pd.concat(fr, ignore_index=True) if fr else None

    g_vs = _seed_concat(gciql_root, "value_steps")
    f_vs = _seed_concat(fb_seed_root, "value_steps")
    g_vl = _seed_concat(gciql_root, "value_landscape")
    f_vl = _seed_concat(fb_seed_root, "value_landscape")
    g_cv = _seed_concat(gciql_root, "coverage")
    f_cv = _seed_concat(fb_seed_root, "coverage")

    # 1. value vs cube->goal distance (FB & GCIQL x success/fail)
    if g_vs is not None and f_vs is not None:
        pool = np.concatenate([g_vs["d"].to_numpy(),
                               f_vs["d"].to_numpy()])
        edges = np.quantile(pool, np.linspace(0, 1, 10))
        edges = np.unique(edges)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        fig, ax = plt.subplots(figsize=(6, 4))
        for nm, df in (("FB", f_vs), ("GCIQL", g_vs)):
            for oc, ls in (("success", "-"),
                           ("transport_fail", "--")):
                sub = df[df.outcome == oc]
                if sub.empty:
                    continue
                idx = np.clip(np.digitize(sub["d"], edges) - 1, 0,
                              len(ctr) - 1)
                mv = (pd.DataFrame({"b": idx, "V": sub["V"].to_numpy()})
                      .groupby("b")["V"].mean())
                ax.plot(ctr[mv.index], mv.values, ls, marker="o",
                        label=f"{nm} {oc}")
        ax.set_xlabel("cube -> goal distance")
        ax.set_ylabel("V(s)")
        ax.set_title("Value vs goal distance: FB vs GCIQL")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(cdir / "cmp_value_vs_dist.png", dpi=120)
        plt.close(fig)
    else:
        print("  [warn] cmp_value_vs_dist skipped (missing value_steps)")

    # 2. per-episode rho distribution (box) for the four groups
    if g_vl is not None and f_vl is not None:
        groups, labels = [], []
        for nm, df in (("FB", f_vl), ("GCIQL", g_vl)):
            for oc in ("success", "transport_fail"):
                v = df[df.outcome == oc]["rho_V_negd"].dropna()
                groups.append(v.to_numpy() if len(v) else np.array([0.0]))
                labels.append(f"{nm}\n{oc}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_ylabel("Spearman rho(V, -d)")
        ax.set_title("Per-episode value gradient: FB vs GCIQL")
        ax.tick_params(axis="x", labelsize=7)
        fig.tight_layout()
        fig.savefig(cdir / "cmp_rho_box.png", dpi=120)
        plt.close(fig)

    # 3. outcome funnel (normalized fraction) FB vs GCIQL
    if g_vl is not None and f_vl is not None:
        order = ["success", "transport_fail", "other"]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(order))
        for i, (nm, df) in enumerate((("FB", f_vl), ("GCIQL", g_vl))):
            fr = (df["outcome"].value_counts(normalize=True)
                  .reindex(order).fillna(0.0))
            ax.bar(x + (i - 0.5) * 0.4, fr.values, 0.4, label=nm)
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylabel("episode fraction")
        ax.set_title("Outcome funnel: FB vs GCIQL")
        ax.legend()
        fig.tight_layout()
        fig.savefig(cdir / "cmp_funnel.png", dpi=120)
        plt.close(fig)

    # 4. coverage nn_dist by region FB vs GCIQL
    if g_cv is not None and f_cv is not None:
        regs = sorted(set(g_cv["region"]) | set(f_cv["region"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(regs))
        for i, (nm, df) in enumerate((("FB", f_cv), ("GCIQL", g_cv))):
            m = df.groupby("region")["nn_dist"].mean()
            ax.bar(x + (i - 0.5) * 0.4,
                   [float(m.get(r, 0.0)) for r in regs], 0.4, label=nm)
        ax.set_xticks(x); ax.set_xticklabels(regs)
        ax.set_ylabel("nn_dist"); ax.legend()
        ax.set_title("Off-support coverage: FB vs GCIQL")
        fig.tight_layout()
        fig.savefig(cdir / "cmp_coverage.png", dpi=120)
        plt.close(fig)

```

(`np`, `pd`, `plt`, `Path` are already imported at module top.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -q`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/gciql_profile_aggregate.py tests/test_gciql_profile.py
git commit -m "feat(gciql_aggregate): FB-vs-GCIQL value/rho/funnel/coverage plots"
```

---

## Task 3: Paper generator embeds every `cmp_*.png`

**Files:**
- Modify: `evals/paper_fragment.py`
- Test: `tests/test_paper_fragment.py`

- [ ] **Step 1: Add failing assertion for multiple comparison figures**

In `tests/test_paper_fragment.py` `test_build_fragment`, the GCIQL block
currently creates one `cmp_value_rho.png`. After the existing
`(gcmp / "cmp_value_rho.png").write_bytes(b"\x89PNG")` line (the test
writes the comparison dir as `gcmp`), add:

```python
    for extra in ("cmp_value_vs_dist.png", "cmp_rho_box.png",
                  "cmp_funnel.png", "cmp_coverage.png"):
        (gcmp / extra).write_bytes(b"\x89PNG")
```

and after the existing `assert "cmp_value_rho.png" in t3` line add:

```python
    for extra in ("cmp_value_vs_dist.png", "cmp_rho_box.png",
                  "cmp_funnel.png", "cmp_coverage.png"):
        assert extra in t3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_paper_fragment.py::test_build_fragment -q`
Expected: FAIL — `assert 'cmp_value_vs_dist.png' in t3` (only
`cmp_value_rho.png` is embedded).

- [ ] **Step 3: Glob every `cmp_*.png` into the subsection**

In `evals/paper_fragment.py`, replace this block (currently L181-202):

```python
        gciql_cmp = Path(gciql_cmp)
        cmp_png = gciql_cmp / "cmp_value_rho.png"
        if cmp_png.exists():
            shutil.copyfile(cmp_png, figdir / "cmp_value_rho.png")
        cmp_tbl = ""
        pq = gciql_cmp / "fb_vs_gciql.parquet"
        if pq.exists():
            import pandas as pd
            cmp_tbl = df_to_booktabs(
                pd.read_parquet(pq), caption=_tex("FB vs GCIQL"),
                label="tab:fb-vs-gciql")
        cmp_md = ""
        cmp_md_p = gciql_cmp / "comparison.md"
        if cmp_md_p.exists():
            cmp_md = _story_to_latex(cmp_md_p.read_text())
        gcmp_block = "\n".join([
            r"\subsection{FB vs GCIQL: Shared-Axis Comparison}",
            _fig("figures", "cmp_value_rho.png",
                 (figdir / "cmp_value_rho.png").exists(),
                 _tex("FB vs GCIQL: value gradient")),
            cmp_tbl, cmp_md])
```

with:

```python
        gciql_cmp = Path(gciql_cmp)
        cmp_figs = []
        for png in sorted(gciql_cmp.glob("cmp_*.png")):
            shutil.copyfile(png, figdir / png.name)
            cap = _tex(png.stem.replace("cmp_", "").replace("_", " "))
            cmp_figs.append(_fig("figures", png.name, True,
                                 "FB vs GCIQL: " + cap))
        cmp_tbl = ""
        pq = gciql_cmp / "fb_vs_gciql.parquet"
        if pq.exists():
            import pandas as pd
            cmp_tbl = df_to_booktabs(
                pd.read_parquet(pq), caption=_tex("FB vs GCIQL"),
                label="tab:fb-vs-gciql")
        cmp_md = ""
        cmp_md_p = gciql_cmp / "comparison.md"
        if cmp_md_p.exists():
            cmp_md = _story_to_latex(cmp_md_p.read_text())
        gcmp_block = "\n".join([
            r"\subsection{FB vs GCIQL: Shared-Axis Comparison}",
            *cmp_figs, cmp_tbl, cmp_md])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_paper_fragment.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add evals/paper_fragment.py tests/test_paper_fragment.py
git commit -m "feat(paper): embed all cmp_*.png in FB-vs-GCIQL subsection"
```

---

## Task 4: Full verify + GCIQL re-run + regenerate + report

**Files:** none (operational).

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, zero failures.

- [ ] **Step 2: Re-run the 10-seed GCIQL sweep (writes value_steps)**

```bash
rm -rf analysis/gciql_profile
i=0
for RUN in RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd0*_*; do
  i=$((i+1))
  .venv-jax-cpu/bin/python scripts/gciql_profile.py \
    --run-dir "$RUN" --step 1000000 \
    --out analysis/gciql_profile/s${i}_final \
    --tasks 1,2,3,4,5 --n-episodes 10 \
    --dataset-path datasets/cube-single-play-v0
done
```

Expected (~6 min): each `analysis/gciql_profile/s*_final/` has
`value_landscape.parquet, coverage.parquet, phase_funnel.parquet,
value_steps.parquet, metrics.json`. Verify:
`ls analysis/gciql_profile/s1_final/value_steps.parquet`.

- [ ] **Step 3: Aggregate + comparison + paper regen**

```bash
.venv/bin/python scripts/gciql_profile_aggregate.py
.venv/bin/python scripts/build_paper_fragment.py --paper-subdir fb-cube-failure
```

Expected: `analysis/gciql_profile/aggregate/comparison/` contains
`cmp_value_vs_dist.png, cmp_rho_box.png, cmp_funnel.png,
cmp_coverage.png, cmp_value_rho.png, fb_vs_gciql.parquet,
comparison.md`; `PAPER/fb-cube-failure/fb_cube_failure.tex` embeds all
five `cmp_*.png` under "FB vs GCIQL: Shared-Axis Comparison" and keeps
the user's 4-subsection layout.

- [ ] **Step 4: Report**

`grep -c includegraphics PAPER/fb-cube-failure/fb_cube_failure.tex`;
read `analysis/gciql_profile/aggregate/comparison/comparison.md` and
report the figures + synthesis to the user. Do not commit
`analysis/`/`PAPER/` artifacts unless asked.

---

## Self-review

- **Spec coverage:** Component A → Task 1 (4-tuple + value_steps,
  FB-schema). Component B → Task 2 (fb_seed_root threaded; the four
  plots: value-vs-dist quantile-binned, rho box of four groups, outcome
  funnel normalized, coverage by region; missing-input guards). Component
  C → Task 3 (glob `cmp_*.png`). Execution/report → Task 4 (re-run,
  aggregate, regen, report). All spec sections mapped.
- **Placeholder scan:** none — every code step is complete; the
  `_seed_steps` helper is given in corrected final form.
- **Type consistency:** `episodes_to_frames` returns 4-tuple `(vl, cov,
  fun, vs)` in Task 1 and is unpacked as 4 in `main()` (Task 1 Step 4)
  and tests (Task 1 Step 1). `value_steps` schema `{task, episode,
  outcome, d, V, transport}` identical in producer (Task 1 Step 3),
  FB reference (verified facts), and comparison loader (Task 2 Step 4
  `_seed_concat`). `aggregate(..., fb_seed_root=)` /
  `_comparison(..., gciql_root=, fb_seed_root=)` keyword names
  consistent across Task 2 Steps 3-4 and the test (Step 1). Paper glob
  `cmp_*.png` matches the filenames written in Task 2 Step 4 and
  asserted in Task 3.
