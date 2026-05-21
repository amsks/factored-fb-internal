# Design: Extend the cube-failure paper with pixel-based FB results

Date: 2026-05-20
Status: approved (brainstorming)
Branch: `feat/cube-analysis-pipeline`

## Goal

Extend the diagnostic paper *"Diagnosing Forward--Backward Representations on
Cube Manipulation"* (`PAPER/fb-cube-failure/fb_cube_failure.tex`) with a new
section reporting **FB+FlowBC performance on the same `cube-single` tasks from
pixel observations**. This first pass covers headline performance only (learning
curves + final IQM success table) plus qualitative rollouts (filmstrips). GCIQL
pixel runs are not yet available; the section and pipeline reserve a slot so
GCIQL-pixel drops in later with one flag. Deeper diagnostics (funnel, coverage,
value-gradient, B-resolution) for pixels are explicitly out of scope here.

## Context

- FB pixel runs already exist on disk and in wandb:
  - Local: `RESULTS/fb-pixel-results/<ts>__cube-single-play-v0__fb_flowbc__s{0..4}/`
    — each holds `checkpoints/` (final.pt + step_*.pt) and
    `eval_videos/step_*/cube-single-play-singletask-task{1..5}-v0.gif`. No
    metrics files on disk.
  - wandb: project `amsks/factored-fb`, run names
    `visual-cube-single__fb_flowbc__s{0..4}__full1M_amsks`,
    `config.obs_type = pixels`, `config.domain = cube-single-play-v0`,
    full 1M steps, DrQ encoder, `frame_stack: 3`, random-shift augmentation.
- The runs log the **same canonical metric keys** as the state runs
  (`eval/reward/eval/success`, per-task
  `eval/reward/cube-single-play-singletask-task{N}-v0/success`), via the shared
  `train.py` + `evals/ogbench.py` path. So the existing pull/curve code applies.
- The **only** disambiguator from the state runs is `config.obs_type`
  (`pixels` vs `state`): group is null and tags are empty for the pixel runs,
  and the domain string is identical to the state runs.
- wandb is reachable here: `.venv` has `wandb` + `rliable`; `~/.netrc` has
  `api.wandb.ai` credentials.

### Existing pipeline reused

- `scripts/wandb_pull.py` — pulls canonical-schema run histories to a parquet
  cache + `_meta.json`. Filters by group/tags/hours/domains.
- `scripts/fb_gciql_curves.py` — loads FB (wandb cache) + GCIQL (local
  `eval.csv`), produces per-task + aggregate IQM curves (95% stratified
  bootstrap CI) and a final IQM success table (md + tex) via `rliable`, with the
  arch8 RNG-kwarg monkeypatch. Reusable helpers: `load_fb`, `load_gciql`,
  `interp_tensor`, `final_scores`, `iqm_curve`, `iqm_table_estimates`,
  `_plot_curve`, `COMMON_GRID`, `TASKS`, `TASK_LABELS`, `COLORS`,
  `_patch_rliable_for_arch8`.
- Paper figures physically live in `PAPER/fb-cube-failure/figures/` but are
  referenced in the `.tex` as `Cube-Failure/figures/<name>.png` (build-specific
  convention). New figures follow this same convention.

## Chosen approach: reuse-by-refactor (Approach A)

Make the curve/IQM rendering method-generic and add thin pixel drivers, rather
than duplicating logic (Approach B) or building one unified parametrized script
(Approach C, over-engineered / risks state-paper reproduction).

### Components

#### 1. `scripts/wandb_pull.py` — add `--obs-type` filter (edit)

- Add `obs_type: Optional[str]` to `filter_runs(...)`; when set, keep only runs
  whose `r.config.get("obs_type") == obs_type`. Empty/None means no filtering on
  it (preserves current behavior).
- Thread a new `--obs-type` CLI arg through `pull(...)` → `filter_runs(...)`.
- Pull command:
  ```
  .venv/bin/python scripts/wandb_pull.py --project amsks/factored-fb \
      --obs-type pixels --domains cube-single-play-v0 --hours 120 \
      --out analysis/wandb_data_fbpixel
  ```
- Expect exactly the 5 FB pixel seeds to land (verify count + seeds in
  `_meta.json`). Runs lacking the canonical key are skipped with a warning.

#### 2. `scripts/fb_gciql_curves.py` — method-generic `render()` (refactor)

- Change `render(fb, gc, out_dir)` to
  `render(methods: Dict[str, Dict[int, pd.DataFrame]], out_dir: Path, *,
  title_prefix: str = "FB vs GCIQL", n_seeds_note: str | None = None)`.
- Iterate over `methods.items()` everywhere the body currently hardcodes the FB
  and GCIQL tensors and the `for method in ("FB","GCIQL")` table loops.
- `main()` keeps identical behavior: `render({"FB": fb, "GCIQL": gc}, out)`.
  CLI unchanged.
- `COLORS` already maps both "FB" and "GCIQL"; unknown methods fall back to the
  default cycle.

#### 3. `scripts/fb_pixel_curves.py` — pixel driver (new)

- Imports the reusable helpers from `fb_gciql_curves`.
- Loads FB pixel via `load_fb(cache_dir)`; builds `methods = {"FB": fb_pixel}`.
- If `--gciql <root>` is given, adds `methods["GCIQL"] = load_gciql(root)`
  (the reserved slot; no GCIQL pixel data exists yet).
- Calls the shared `render(methods, out_dir, title_prefix="FB (pixels)",
  n_seeds_note="n=5 seeds")`.
- Defaults: `--fb analysis/wandb_data_fbpixel`, `--out analysis/fb_pixel`,
  `--gciql` unset.
- Outputs: `analysis/fb_pixel/{curves_pertask.png, curve_aggregate.png,
  iqm_table.tex, iqm_table.md}`.

#### 4. `scripts/pixel_filmstrips.py` — qualitative filmstrips (new)

- For each task in `task1..task5`: locate the eval GIF for a representative seed
  (default s0) at the final checkpoint
  (`eval_videos/step_1000000/cube-single-play-singletask-task{N}-v0.gif`),
  sample `--frames` (default 6) evenly-spaced frames with `imageio`, composite a
  horizontal filmstrip with `PIL`, and save
  `PAPER/fb-cube-failure/figures/pixel_filmstrip_task{N}.png`.
- CLI: `--results-root RESULTS/fb-pixel-results`, `--seed 0`,
  `--step step_1000000`, `--frames 6`,
  `--out PAPER/fb-cube-failure/figures`.
- Seed run is resolved by matching the `__s{seed}` suffix in the run dir name.
- Caption is neutral (representative pixel rollout; outcome not asserted).

#### 5. `PAPER/fb-cube-failure/fb_cube_failure.tex` — new section (edit)

Insert `\section{Does the failure persist under pixel observations?}` (label
`sec:pixels`) immediately before `\section{Discussion and Next Steps}`:

- One paragraph: same `cube-single` tasks and same FB+FlowBC method, but from
  64x64 RGB observations (DrQ encoder, frame-stack 3, random-shift aug); tests
  whether the diagnosis is specific to low-dimensional state input.
- Figure: `Cube-Failure/figures/pixel_curves_pertask.png` (per-task + aggregate
  IQM curves, FB only, n=5 seeds, 95% CI).
- Table: final IQM success per task + aggregate. FB row populated from
  `analysis/fb_pixel/iqm_table.tex` (tabular pasted inline, matching the paper's
  existing inline-table convention). GCIQL row present but commented out with
  `% GCIQL-pixel: pending`.
- Five filmstrip figures (`pixel_filmstrip_task{1..5}.png`) as subfigures with a
  neutral caption.
- A short note that GCIQL-pixel is forthcoming and will complete the comparison
  and enable the deeper diagnostic chain for the visual setting.
- A `\takeaway{...}` box.
- New figures written to `PAPER/fb-cube-failure/figures/`; referenced via the
  `Cube-Failure/figures/...` convention. The pixel curve PNG is copied from
  `analysis/fb_pixel/curves_pertask.png` to
  `PAPER/fb-cube-failure/figures/pixel_curves_pertask.png`.

## Data flow

```
wandb (amsks/factored-fb, obs_type=pixels, domain=cube-single-play-v0)
  -> wandb_pull.py --obs-type pixels --out analysis/wandb_data_fbpixel/
       -> *.parquet + _meta.json
            -> fb_pixel_curves.py -> analysis/fb_pixel/{curves_pertask.png,
               curve_aggregate.png, iqm_table.{tex,md}}
                 -> copy curve PNG -> PAPER/.../figures/pixel_curves_pertask.png
                 -> paste tabular -> fb_cube_failure.tex

RESULTS/fb-pixel-results/<run __s0>/eval_videos/step_1000000/
  cube-single-play-singletask-task{N}-v0.gif
  -> pixel_filmstrips.py -> PAPER/.../figures/pixel_filmstrip_task{N}.png
       -> embedded in fb_cube_failure.tex
```

## Testing

- `tests/test_fb_gciql_curves.py` — update for the generic `render(methods, ...)`
  signature; keep coverage of the FB+GCIQL path.
- `tests/test_fb_pixel_curves.py` (new) — synthetic 5-seed parquet cache +
  `_meta.json`; assert the 4 outputs exist and `iqm_table.md` has an FB row with
  task1..task5 + overall columns.
- `tests/test_wandb_pull_obs_type.py` (new) — pure `filter_runs` test with fake
  run objects: `--obs-type pixels` keeps pixel-config runs, drops state-config
  runs; empty filter keeps both.
- `tests/test_pixel_filmstrips.py` (new) — write a synthetic multi-frame GIF,
  run the extractor, assert a non-empty PNG of the expected width
  (`frames * frame_width`).
- All scripts/tests run with `.venv/bin/python` (rliable arch8 shim requires
  the venv).

## Risks & mitigations

- **wandb pull completeness**: verify exactly 5 FB pixel seeds land in the cache;
  if fewer, surface the warning and stop before editing the paper.
- **Figure path convention**: follow the existing `Cube-Failure/figures/...`
  reference convention rather than "fixing" the mismatch, so new figures resolve
  identically to existing ones in the user's build.
- **n=5 vs n=10 seeds**: pixel IQM CIs are wider; note `n=5` in the table/figure
  captions.
- **GIF content**: filmstrip samples whatever the eval GIF contains (one
  representative episode); caption stays outcome-neutral.

## Out of scope

- GCIQL pixel runs (not available) — slot reserved only.
- Deeper pixel diagnostics: phase/outcome funnel, nearest-neighbor coverage,
  value-gradient (Spearman / value-vs-distance), backward-map resolution. These
  require running the profiling pipeline through the DrQ encoder on pixel
  checkpoints and are deferred to a later pass.

## Reserved GCIQL-pixel slot

When GCIQL pixel `eval.csv` data exists:
`.venv/bin/python scripts/fb_pixel_curves.py --gciql <gciql_pixel_root>` adds the
GCIQL method to curves + table; uncomment the GCIQL table row in the paper. No
throwaway code is introduced now.
