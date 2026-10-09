# Design: Value-function analysis on the training dataset (state FB vs GCIQL)

Date: 2026-05-22
Status: approved (brainstorming)
Branch: `feat/cube-analysis-pipeline`

## Goal

Test whether FB's and GCIQL's learned value functions reproduce the
terminal-resolution pathology **on the offline training states themselves**, not
only on closed-loop rollouts. The rollout analysis showed FB's value is weakly
aligned with object-to-goal progress during transport. This asks: does the same
weak value↔distance relationship appear on the *training* distribution? If yes,
the pathology is intrinsic to the learned value; if the value ranks training
states well but fails on rollouts, it points to distribution shift / compounding
error instead.

Scope: **state observations only** (both methods state-based).

## What we compute

For each of the five `cube-single` tasks (then aggregated), over a sample of
offline buffer transitions classified into manipulation phases:

- **FB policy value** $V(s)=Q_z(s,\pi(s))$ — matches the rollout pathology metric.
- **FB data-action value** $Q_z(s,a_{\text{data}})$ — value of the actual buffer transition.
- **GCIQL value** $V(s,g)$ — GCIQL's value network.

against cube-to-goal distance $d=\lVert \text{cube}-g\rVert$, **separately per phase**
(reach / grasp / transport), plus the **phase composition of the dataset**.

## Context / reuse

- Offline data: `datasets/cube-single-play-v0/buffer/episode_*.npz`
  (state `observation` 28-d, `physics` 21-d qpos; cube xyz = `physics[:,14:17]`).
- State checkpoints: FB = `RESULTS/Factored-FB-cube-run/*__s{seed}/checkpoints/final.pt`
  (ortho=1000 sweep; seeds 3–8,10 complete); GCIQL =
  `RESULTS/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd*/` (params_1000000).
- FB value: `evals/representation_profile.py:q_values` ($Q_z(s,a)$) and
  `v_values` ($Q_z(s,\pi(s))$); z per task via `OGBenchEvaluator._infer_z`.
- GCIQL value: jax `agent.network.select("value")(obs,goal)` (state checkpoint,
  `.venv-jax-cpu`), goal observation g per task from `env.reset(task_id)` `info["goal"]`.
- Phase taxonomy + thresholds: `evals/_profile_core._region` + `evals.phase_probe.Thresholds`
  (transport = lifted+gripped; grasp = gripper closed; reach = else).
- Two venvs: FB torch (`.venv`), GCIQL jax (`.venv-jax-cpu`) — so the shared
  state sample is written once and both methods score it (identical states).

## Chosen approach (Approach A)

A new profile+aggregate pair, two-venv like `outcome_clips`, reusing the shared
math. Rejected: extending the rollout-centric `representation_profile`/`gciql_profile`
(bloat); a single monolithic script (awkward across venvs).

### Components

#### 1. `scripts/training_value_profile.py --method fb` (`.venv`)

The "primary" pass; builds the shared substrate and FB values.
- Load FB state cfg + checkpoint; load the offline buffer.
- Sample N transitions (default 20000): `obs`, `action`, `physics`.
- **Phase classification (env-based, exact):** for the sample, obtain per-transition
  gripper-opening and cube height by loading each transition's `physics` into the
  env (`set qpos`, `mj_forward`) and reading the env's proprio/privileged signals
  — the same quantities the rollout reads from `info` — then apply `_region` /
  `Thresholds`. Cube xyz comes directly from `physics[:,14:17]`; `table_z` from a
  task's `init_xyzs`. (The plan may identify the gripper obs index once for a
  vectorized variant; correctness target is the env-based labels.)
- **Per task:** infer z; compute $Q_z(\text{obs},\pi(\text{obs}))$ and
  $Q_z(\text{obs},a_{\text{data}})$; distance $d=\lVert\text{cube}-\text{goal\_xyz}\rVert$.
- Write:
  - `analysis/training_value/training_states.parquet` — shared substrate:
    one row per sampled transition with `obs` (as needed by GCIQL), `region`,
    `cube_x/y/z`, and `d_task1..d_task5`.
  - `analysis/training_value/fb_values.parquet` — long form
    (`task, region, d, V_policy, V_data`).
  - `analysis/training_value/phase_composition.parquet` — fraction of
    reach/grasp/transport over a trajectory-level pass of the buffer.

#### 2. `scripts/training_value_profile.py --method gciql` (`.venv-jax-cpu`)

- Read `training_states.parquet` (the exact sampled `obs` + `region` + `d_task*`).
- Load GCIQL state checkpoint; per task get goal obs g; compute $V(\text{obs},g)$.
- Write `analysis/training_value/gciql_values.parquet` (`task, region, d, V`).

#### 3. `scripts/training_value_aggregate.py` (`.venv`)

Reads the three parquets → `analysis/training_value/aggregate/`:
- **Per-phase Spearman** $\rho(V,-d)$ table (rows: phase × method
  {FB-policy, FB-data, GCIQL}; per-task + aggregate) — md + tex.
- **Value-vs-distance curves** faceted by phase, one line per method
  (z-scored within method for shape comparison).
- **Phase-composition bar** (reach/grasp/transport fractions in the dataset).
- **Value-over-cube-xy scenes** per phase × method: binned mean value over cube
  xy (reusing the eval scene-binning approach), the spatial "value function
  plots" on training states.

### Data flow

```
datasets/cube-single-play-v0
  -> training_value_profile.py --method fb (.venv)
       -> training_states.parquet (shared) + fb_values.parquet + phase_composition.parquet
  -> training_value_profile.py --method gciql (.venv-jax-cpu)   [reads training_states.parquet]
       -> gciql_values.parquet
  -> training_value_aggregate.py (.venv)
       -> aggregate/{spearman_table.*, value_vs_dist.png, phase_composition.png, scene_<phase>_<method>.png}
```

## Testing

- Unit (`.venv`, synthetic arrays, no MuJoCo/torch/jax):
  - per-transition region classification matches `_region` on known grip/lift inputs;
  - per-task distance computation;
  - Spearman aggregation + the long-form table builder.
- Smoke: each backend on a few hundred sampled states against a real state
  checkpoint (FB and GCIQL), asserting the parquets have the expected columns
  and non-empty per-phase rows; aggregate produces the table + one scene.

## Risks & mitigations

- **Gripper signal on the buffer:** classification needs gripper-opening, which
  isn't a pre-identified buffer index. Mitigation: derive it via the env
  (`set qpos` + `mj_forward`) on the shared sample (robust); optionally identify
  the obs index once for speed.
- **Sparse transport states in play data:** play data may have few
  lifted+gripped states; report per-phase counts and flag low-N phases (as the
  eval scenes do).
- **Two-venv state sharing:** GCIQL must score the *same* states as FB; enforced
  by GCIQL reading `training_states.parquet` rather than re-sampling.
- **Throughput:** scoring 20k states through the FB model / GCIQL net is light;
  the env-based phase pass is the main cost (one `mj_forward` per sampled state).

## Out of scope

- Pixel observations (state only this round).
- Re-deriving the rollout (eval) value analysis (already done).
- Backward-map / z-decoding probes (this is a value-vs-distance study).

## Commit boundary

Commit `scripts/` + `tests/` + this spec/plan. `analysis/`, `RESULTS/`,
`datasets/` stay untracked ([[project-untracked-artifacts]]). FB runs under
`.venv`, GCIQL under `.venv-jax-cpu` ([[project-venvs]]).
