# Cube Analysis Pipeline — Design Spec

**Date:** 2026-05-17
**Status:** Approved (design); pending implementation plan
**Scope:** One spec, two components. Implement the wandb replot pipeline first,
then the checkpoint re-run / behavior pipeline.

## 1. Motivation

The cube runs (`RESULTS/Factored-FB-cube-run/`, 10 seeds of a single HP cell:
`domain=cube_single`, `ortho_coef=1000`, `lr_b=1e-4`) saved checkpoints and
eval GIFs but **no scalar metrics locally** — `train.log` holds only MuJoCo
startup noise. All success/reward metrics live in wandb
(`entity=amsks`, project `factored-fb`). `analysis/wandb_data/*.parquet`,
`_meta.json`, and `analysis/plots/sweep-*.png` exist as orphaned artifacts:
**no committed script reproduces them**. `scripts/sweep_results.py` only emits
a summary table, not learning curves.

Two needs:

1. **Replot wandb metrics** — a reproducible, configurable pull + plot pipeline
   covering the whole sweep (cube-single + antmaze-medium, all HP cells).
2. **Re-run checkpoints** — load a checkpoint and probe cube agent behavior
   beyond the saved success/reward + first-episode GIF.

## 2. Approach

Approach A — **thin script layer reusing existing code**. Small CLIs under
`scripts/`, with reusable logic factored into one importable, unit-tested
helper module. No new framework/package. Matches existing repo conventions
(`scripts/`, Hydra config, `sweep_results.py` style). Training and eval code
paths are **not modified**; all additions are additive.

## 3. File Layout

```
scripts/wandb_pull.py         # CLI: wandb API -> analysis/wandb_data/<id>.parquet + _meta.json
scripts/plot_sweep.py         # CLI: parquet cache -> analysis/plots/*.png (+ final_summary.md)
scripts/analyze_checkpoint.py # CLI: one .pt + its config -> trajectories / probes / videos
evals/analysis.py             # NEW shared helpers (unit-tested):
                              #   build_agent_from_cfg, load_checkpoint,
                              #   rollout_with_trajectory, z_probe
```

Reused, unmodified: `train.make_agent`, `envs.ogbench.create_ogbench_env`,
`evals.ogbench.OGBenchEvaluator` (for `_infer_z`), `envs.rollout`.
Cache/plot dirs stay at existing `analysis/wandb_data/` and `analysis/plots/`.

## 4. Component 1 — wandb replot pipeline (built first)

### 4.1 `scripts/wandb_pull.py`

- **Args:** `--entity --project --groups --tags --hours --domains
  --out analysis/wandb_data`.
- Uses `wandb.Api()` (same pattern as `scripts/sweep_results.py`). Filters
  runs by group / tag / created-at window / domain.
- Pulls `r.history(...)` for a configurable metric key list; writes one
  parquet per run to `analysis/wandb_data/<run_id>.parquet`; refreshes
  `analysis/wandb_data/_meta.json` (list of `{id, name, state, group,
  config, summary}` — matching the existing cache schema exactly).
- **Per-run `try/except`:** a single failed `r.history` or empty history
  warns and is skipped; the pull never aborts wholesale.

### 4.2 `scripts/plot_sweep.py`

- **Args:** `--in analysis/wandb_data --out analysis/plots --domains`.
- Loads parquet + `_meta.json`, groups runs by `(domain, ortho_coef, lr_b)`
  HP cell, aligns on `_step`, computes mean±std across seeds.
- Per domain, produces:
  - `sweep-<domain>__eval.png` — `eval/success` & `eval/reward` vs step,
    one mean±std band per HP cell.
  - `sweep-<domain>__eval_pertask.png` — the 5 cube singletask success
    curves (`task1..task5`).
  - `sweep-<domain>__train.png` — `train/loss/{fb,orth,actor,bc_flow}`
    vs step, mean±std per HP cell.
  - `sweep-<domain>__final.png` + `analysis/plots/final_summary.md` —
    final (or best) eval success per HP cell, mean±std (n).

### 4.3 Schema drift (key design point)

Cached runs span code versions with different metric names (e.g. old
`eval/cube-single-play-singletask-task1-v0/success` vs current
`eval/reward/eval/success`). Metric selection uses **substring/regex
patterns with an ordered fallback list**. Unmatched keys warn and are
omitted from that run's curves — old and new runs co-plot where they
overlap, nothing crashes.

### 4.4 Testability

The group → align → mean/std aggregation is a **pure function** separate
from matplotlib, unit-testable without wandb or a display.

## 5. Component 2 — checkpoint re-run / behavior pipeline

### 5.1 `scripts/analyze_checkpoint.py`

Operates on **one checkpoint explicitly** (caller scripts any looping).

- **Args:** `--config <run>/.hydra/config.yaml`
  `--checkpoint <run>/checkpoints/step_N.pt`
  `--out analysis/checkpoint_runs/<name>` `--device cpu`
  `--tasks ...` `--n-episodes` `--what traj,probe,video` `--seed`
  `--mujoco-gl`.
- **Load:** `cfg = OmegaConf.load(config)`; force `cfg.device=<--device>`
  (the `.hydra` configs say `device=cuda`; locally on macOS this must be
  `cpu`). Build `env = create_ogbench_env(cfg.domain, ...)`,
  `agent = train.make_agent(cfg, obs_space, action_dim)`,
  then `load_checkpoint(agent, ckpt, map_location=device)`.

### 5.2 Capabilities (selectable via `--what`)

1. **Full trajectory logs** — `rollout_with_trajectory()` wraps the existing
   `envs.rollout` loop, collecting per step:
   `observation, action, reward, terminated, truncated, success(info), z`.
   Output per task: `trajectories/<task>.npz` storing padded arrays shaped
   `[n_episodes, T_max, ...]` plus a `lengths[n_episodes]` array to recover
   each episode's true length; plus `trajectory_summary.parquet` (per-episode
   return, length, success).
2. **z / task-conditioning probes** — reuse `OGBenchEvaluator._infer_z(task)`:
   - **Cross-task z matrix:** roll out in env *i* conditioned on task *j*'s z
     for all 5×5 → `probes/cross_task.parquet` + heatmap PNG.
   - **z interpolation:** for a chosen task pair, sweep
     `z = (1-α)·z_A + α·z_B`, α∈[0,1] → `probes/z_interp.parquet` + line plot.
3. **Configurable video capture** — render selected tasks/episodes to GIF
   (reuse the evaluator's imageio path) into `videos/<task>_ep<k>.gif`.

### 5.3 Output tree

```
analysis/checkpoint_runs/<name>/
  trajectories/   <task>.npz, trajectory_summary.parquet
  probes/         cross_task.parquet + .png, z_interp.parquet + .png
  videos/         <task>_ep<k>.gif
  metrics.json    config echo + aggregate success/reward (provenance)
```

## 6. Error Handling

- **wandb auth/network:** ambient `wandb` login; per-run `try/except` in the
  pull loop; empty history → skip + warn.
- **Schema drift:** pattern-based metric resolution with fallbacks; unmatched
  keys warn and are omitted (see 4.3).
- **Missing HP cell / domain:** plot script emits remaining plots and notes
  the gap in `final_summary.md` instead of erroring.
- **Device:** explicit `cuda`→`cpu` override via `--device`;
  `torch.load(map_location=...)` always set so GPU-trained checkpoints load
  on the Mac.
- **Checkpoint/agent mismatch:** `load_checkpoint` catches `state_dict` key
  errors and raises a clear message naming expected `cfg.agent` vs keys found.
- **Render backend:** MuJoCo rendering on macOS is unreliable (`MUJOCO_GL`
  egl is Linux-only). `--mujoco-gl` selects backend (try `glfw`, fallback
  `osmesa`). Video failure is caught and isolated — trajectory and probe
  outputs still complete; video skipped with a warning.
- **Determinism:** seed set from `cfg.seed` (or `--seed`) before env/rollout
  so probe matrices are reproducible.

## 7. Testing (TDD)

Pure logic tested without wandb / MuJoCo / display:

- **`plot_sweep` aggregation** — synthetic parquet + meta fixtures; assert
  HP-cell grouping, step alignment, mean/std; assert all 4 outputs produced
  (matplotlib `Agg` backend).
- **`wandb_pull`** — `wandb.Api` mocked; assert filter logic
  (group/tag/hours/domain) and that parquet + `_meta.json` are written with
  the expected schema.
- **`evals/analysis.py` helpers** — `build_agent_from_cfg` + `load_checkpoint`
  round-trip on a tiny agent + temp checkpoint; `rollout_with_trajectory`
  returns correctly shaped/typed arrays on a minimal env + stub agent;
  `z_probe` returns a 5×5 matrix of correct shape.
- **Slow/integration** (real MuJoCo eval, real wandb) — marked/skipped by
  default, runnable on demand.

Fixture models: existing `tests/` and `scripts/smoke.py`.

## 8. Non-Goals / YAGNI

- No higher-fidelity many-episode re-eval (wandb metrics suffice; explicitly
  deselected).
- No `analysis/` Python package / framework (Approach B rejected).
- No notebooks (Approach C rejected — pipeline must be reproducible).
- No batch/whole-tree checkpoint driver — caller scripts looping over the
  single-checkpoint CLI.
- No changes to training or eval code paths.

## 9. Build Order

1. `scripts/wandb_pull.py` + tests
2. `scripts/plot_sweep.py` + tests
3. `evals/analysis.py` helpers + tests
4. `scripts/analyze_checkpoint.py` (trajectory → probes → video) + tests
