# Cube Phase-Failure Probe — Design Spec

**Date:** 2026-05-18
**Status:** Approved (design); pending implementation plan
**Builds on:** the checkpoint re-run pipeline
(`docs/superpowers/specs/2026-05-17-cube-analysis-pipeline-design.md`,
`evals/analysis.py`, `scripts/analyze_checkpoint.py`).

## 1. Hypothesis

A cube episode decomposes into **reach** (move hand to cube) → **grasp/
pick-up** (close gripper, lift cube) → **transport** (carry held cube to
goal). The claim under test: **FB reliably reaches the cube and, once the
cube is held, reliably transports it to the goal; the dominant failure is
grasping/pick-up.**

The probe must produce evidence that either supports or refutes this with
both observational and causal measurements.

## 2. Methods

- **M1 — observational phase decomposition.** Run normal rollouts, classify
  per-episode how far the agent got (reach / secure / success), and show
  where rollouts die. Correlational; characterizes failure modes and
  calibrates thresholds.
- **M2 — counterfactual start states.** Start the policy from controlled
  states (baseline / skip-reach / pre-grasped) and compare conditional
  success. Causal; isolates the grasp bottleneck.

Both are produced in one run. M1's classifier is applied to every M2
scenario's rollouts as well.

## 3. Phase Signals & Definitions

Per-step signals, for the single cube (`target_block = 0`). Source
preference: use the `info` dict returned by `env.step` if it carries the
`proprio/`+`privileged/` keys; otherwise call
`env.unwrapped.compute_ob_info()`:

| Signal | Source |
|---|---|
| `eff` (hand xyz) | `proprio/effector_pos` |
| `grip` (0 open … 1 closed) | `proprio/gripper_opening` |
| `cube` (xyz) | `privileged/block_0_pos` |
| `goal` (xyz) | `privileged/target_block_pos` |
| `table_z` | cube z captured at the episode's first step |
| `success` | env's own `_success` (cube within 0.04 m of goal) |

Per-episode ordered milestones:

1. **reached** — `min_t ‖eff − cube‖ < ε_reach`
2. **secured** (pick-up) — ∃ a window of ≥ `k` consecutive steps with
   `cube_z − table_z > δ_lift` **and** `grip > τ_grip` (closed) **and**
   `‖eff_xy − cube_xy‖ < ε_reach` (cube rising with the hand)
3. **success** — env success

`furthest_phase ∈ {none, reached, secured, success}` (monotone: success ⇒
secured ⇒ reached for classification ordering — see §3.1). `fail_phase` =
the first milestone not achieved (`none` if `success`).

Diagnostics retained per episode: `min_eff_cube_dist`,
`max_cube_lift` (= `max_t (cube_z − table_z)`), `final_cube_goal_dist`,
`length`.

Default thresholds (all CLI-overridable):
`ε_reach = 0.06 m`, `δ_lift = 0.03 m`, `k = 5` steps, `τ_grip = 0.5`.

### 3.1 Monotone classification

`reached`, `secured`, `success` are computed independently from the
signals, then forced monotone for the phase ladder: if `success` is True,
`secured` and `reached` are treated as True; if `secured` is True,
`reached` is treated as True. The raw independent booleans are also kept
(columns `reached_raw`, `secured_raw`) for diagnostics. `furthest_phase`
and `fail_phase` use the monotone values.

The `τ_grip` gripper-orientation assumption (≈1 closed) is documented and
overridable; if a run shows it inverted, flip via `--tau-grip`.

## 4. M1 Observational Pipeline

1. Reuse `evals.analysis.load_cfg` / `build_env_and_agent` /
   `load_checkpoint`; build `OGBenchEvaluator` and use `_infer_z(task)`
   for z-conditioning (identical to evaluation/`analyze_checkpoint`).
2. `rollout_with_phase_signals(env, agent, z, n_episodes, thresholds)` —
   same control loop as `evals.analysis.rollout_with_trajectory`, but each
   step also extracts `eff/cube/goal/grip`. `table_z` is the cube z at the
   episode's first step. Returns per-episode signal arrays + `success`.
3. `classify_phases(signals, thresholds)` (pure) → per-episode dict with
   the milestones, `furthest_phase`, `fail_phase`, and diagnostics.
4. Aggregate across the 5 tasks → per-episode rows + per-task summary
   (reached/secured/success rates, n). The **phase histogram** (episodes
   by furthest phase, per task) is the M1 deliverable.

## 5. M2 Counterfactual Start States (no weld)

Each scenario runs `n_episodes`. Per episode: `env.reset(seed)` (valid
task: cube init, task target mocap, arm home), then mutate MuJoCo state
via `env.unwrapped._data` + `mujoco.mj_forward`, then the policy (task z)
takes over. **The task goal/target mocap is never modified**, so success
is comparable across scenarios.

- **S0 baseline** — no mutation. Full task.
- **S1 skip-reach** — `cube_xy ← effector_xy`
  (`_data.site_xpos[_pinch_site_id]`), `cube_z ← table_z`, gripper open,
  written via `_data.joint('object_joint_0').qpos[:3]`. Reach trivially
  satisfied; policy must grasp + lift + transport to the real goal.
  (Cube-to-hand instead of arm IK — equivalent, robust.)
- **S2 pre-grasped (physical)** — `cube ← effector_pos`,
  `_data.ctrl[_gripper_actuator_ids] = 255` (close), step a fixed
  `n_settle = 5` env-internal MuJoCo steps (gripper-close ctrl held, no
  policy action) so the fingers physically clamp the cube, then policy
  takes over. `n_settle` is a module constant (not a CLI knob; YAGNI).
  No weld/constraint: if the policy opens the gripper the cube drops
  (recorded as `secured`/`success` False — data, not error).

Handles used (all confirmed present in OGBench
`manipspace_env.py`/`cube_env.py`): `_pinch_site_id`,
`_gripper_actuator_ids`, `object_joint_0`, `_data`, `mujoco.mj_forward`.
A `scenario` argument selects the mutation; default scenario set is
`S0,S1,S2`.

**Hypothesis readout** (auto-written to `summary.md`): the design predicts
`success(S0) ≈ success(S1)` (still must grasp) and `success(S2) ≫
success(S0)` (transport easy once held), with baseline
`reached-rate ≫ secured-rate`. The readout states the observed numbers and
whether they match the prediction; it does not editorialize beyond that.

## 6. Architecture & Files (additive only)

```
evals/phase_probe.py     # PHASE_ORDER; Thresholds (defaults);
                         # extract_signals(info, env);
                         # rollout_with_phase_signals(env, agent, z, n, thr);
                         # classify_phases(signals, thr)  [pure];
                         # apply_scenario(env, scenario)  [S0/S1/S2 surgery];
                         # run_phase_probe(agent, infer_z, make_env, tasks,
                         #   scenarios, n_episodes, thr) -> (per_ep_df,
                         #   summary_df, hist_data);
                         # plot_phase_histogram, plot_scenario_success
scripts/phase_probe.py   # CLI gluing the above
tests/test_phase_probe.py
```

Reuses, unmodified: `evals.analysis` (load_cfg / build_env_and_agent /
load_checkpoint), `evals.ogbench.OGBenchEvaluator._infer_z`,
`data.ogbench.load_ogbench_dataset`, `envs.ogbench.create_ogbench_env`.
No training/eval code is modified.

### 6.1 CLI

`scripts/phase_probe.py` args:
`--config` (required, run `.hydra/config.yaml`),
`--checkpoint` (required, `.pt`),
`--out` (required),
`--device` (default `cpu`),
`--tasks` (comma-separated; default = all 5),
`--n-episodes` (int, default 20),
`--scenarios` (default `S0,S1,S2`),
`--seed` (int, optional override),
`--mujoco-gl` (sets `MUJOCO_GL` before env import; Linux: `egl`),
`--eps-reach` (0.06), `--delta-lift` (0.03), `--k-steps` (5),
`--tau-grip` (0.5).

Heavy imports (torch/ogbench/evaluator/numpy) inside `main()` after
`parse_args` so `--help` is fast (same pattern as `analyze_checkpoint`).
The caller loops seeds/checkpoints; one invocation = one checkpoint.

### 6.2 Outputs → `analysis/phase_probe/<name>/`

- `per_episode.parquet` — columns: task, scenario, episode, reached,
  secured, success, reached_raw, secured_raw, fail_phase,
  min_eff_cube_dist, max_cube_lift, final_cube_goal_dist, length
- `summary.parquet` + `summary.md` — per task×scenario: reached-rate,
  secured-rate, success-rate, n; plus the hypothesis readout
- `phase_histogram.png` — M1, S0: episodes by furthest phase per task
- `scenario_success.png` — M2: grouped S0/S1/S2 success bars per task
- `metrics.json` — config, checkpoint, thresholds, scenario list (provenance)

## 7. Error Handling

- **Non-manipulation env** (no `env.unwrapped._data` or no
  `_pinch_site_id`): raise a clear error — this probe is cube-only.
- **Gripper orientation**: `τ_grip` assumption documented; `--tau-grip`
  override; M1 `secured_raw` retained so an inverted gripper is diagnosable.
- **S2 cube drop**: recorded as `secured`/`success` False; never an error.
- **Device / rendering**: `--device`, `torch.load(map_location=)`,
  `--mujoco-gl` set before env import.
- **Per-task isolation**: each task in `try/finally env.close()`; a
  per-task failure warns and continues; remaining outputs still written.
- **Empty/short episodes**: classification handles a zero-action episode
  (no IndexError); such an episode is `furthest_phase = none`.

## 8. Testing (TDD)

Pure logic unit-tested without MuJoCo:

- `classify_phases` — synthetic signal arrays produce the correct
  `reached/secured/success/fail_phase/furthest_phase`, including:
  reach-but-no-secure, secure-but-no-success, full-success, never-reach,
  threshold edges (`ε_reach`, `δ_lift`, `k`, `τ_grip`), monotone forcing,
  and a zero-length episode.
- Summary aggregation — per task×scenario rates and n correct on a
  synthetic per-episode frame; `summary.md` contains the hypothesis readout.
- `apply_scenario` — a fake env recording intended `_data` mutations:
  S0 = no-op, S1 = cube moved to effector xy at table_z + gripper open,
  S2 = cube moved to effector + gripper-close ctrl + settle steps.
  Verifies dispatch and the mutation contract (not MuJoCo physics).
- Plot functions — produce the two PNGs with `Agg` backend on synthetic
  data (assert files written).

Real MuJoCo S1/S2 physical correctness is validated by a documented Linux
acceptance run (`--mujoco-gl egl`), the same deferral used for the
Phase B real-env gate. `scripts/phase_probe.py` is thin glue; its logic
lives in the unit-tested `evals/phase_probe.py`.

## 9. Non-Goals / YAGNI

- No welded/attached pre-grasp variant (explicitly deselected).
- No arm inverse-kinematics for S1 (cube-to-hand is equivalent + robust).
- No multi-cube / non-cube domains (single-cube only).
- No batch seed/checkpoint driver (caller loops; one checkpoint per run).
- No new training/eval code; no modification of existing pipeline files.
- No automatic acceptance of/again the hypothesis — the probe reports
  numbers and a stated prediction match; interpretation is the user's.

## 10. Build Order

1. `evals/phase_probe.py`: `Thresholds` + `classify_phases` (+ tests)
2. `evals/phase_probe.py`: `extract_signals` +
   `rollout_with_phase_signals` (+ tests with a stub manip-like env)
3. `evals/phase_probe.py`: `apply_scenario` (+ fake-env mutation tests)
4. `evals/phase_probe.py`: `run_phase_probe` + aggregation + plots (+ tests)
5. `scripts/phase_probe.py`: CLI glue (+ `--help` check) + documented
   Linux acceptance run
