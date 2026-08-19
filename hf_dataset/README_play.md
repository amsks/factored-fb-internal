---
license: mit
task_categories:
  - reinforcement-learning
tags:
  - offline-rl
  - goal-conditioned-rl
  - play-data
  - hindsight-relabeling
  - robotics
  - manipulation
  - metaworld
pretty_name: FactoredFB Testbed — Metaworld Play-Resample
size_categories:
  - 1M<n<10M
---

# FactoredFB Testbed — Metaworld Play-Resample

Offline **play** datasets for Metaworld `push` and `pick-place`, collected so that their
structure matches OGBench's `cube-single-play-v0` rather than the usual one-target-per-episode
demonstration format. Built for offline goal-conditioned RL where hindsight relabeling is the
main mechanism (GCIQL, bilinear/factored critics, CRL, FB).

Each trajectory is 500 wrapper steps (1000 simulator steps) of task-agnostic play: a scripted
Metaworld oracle chases a target that is **re-drawn from the env's own goal space every 44
steps**, on the timer alone and never on success — matching OGBench, whose plan oracle runs to
completion whether or not the block arrived. Coverage comes from chaining relocations, not from
action noise, so the goal marginal spreads over the workspace instead of collapsing onto one
point per episode.

## Regimes

| regime | mixture | intent |
|---|---|---|
| `play-resample` | `eps_max=0, noise_max=0.02` | goal-diverse, near-oracle actions |
| `play-resample-hard` | `eps_max=0, noise_max=0.1` | same chaining, more action noise |

Neither uses epsilon-random actions. OGBench sets `p_random_action=0` for all play data, and
here a uniform action knocks the puck out of the reachable workspace, which the stateless
Metaworld oracle can never recover — an absorbing failure. See the v3 notes below.

Tasks: `mw-push`, `mw-pick-place`. 2000 trajectories × 500 steps = **1,000,000 transitions**
each, obs 39-d, action 4-d, `seed=0` — matched to cube-single-play's 1,001,000.

## Format

One stacked `.npz` per (regime, task) at `data/<regime>/metaworld/<task>.npz`, holding the
8 unified fields as `[N, T, ·]`:

```
observation  next_observation   -> [N, 500, 39]
action                          -> [N, 500, 4]
achieved_goal                   -> [N, 500, 3]   object xyz
reward  terminal  timeout  success -> [N, 500]
```

`manifest.json` indexes every file with recomputed statistics. Loader:
`env/unified.py::load_packed` in the Factored-FB repo, which flattens to `N·T` transitions and
synthesizes trajectory-boundary terminals.

## Structure, versus OGBench cube-single-play

Expressed in units of each benchmark's own eval horizon (cube 200 steps, Metaworld 100 wrapper
steps), the two datasets agree on the properties that govern relabeling — see
`scripts/compare_play_resample_to_ogbench.py` in the repo, which recomputes this table from
the raw arrays.

## Known differences you must account for

1. **The goal is present in the Metaworld observation** at dims `36:39` (`_target_pos`); the
   success rule `‖next_obs[4:7] − next_obs[36:39]‖ < radius` reproduces the stored `success`
   flag exactly (radius 0.05 push / 0.07 pick-place). OGBench cube observations contain **no**
   goal at all. Consequence: an unconditional behaviour-cloning actor solves these tasks by
   reading its own input, so a critic comparison run on this data without masking those dims
   measures nothing.

   **Mask the observation, not just the goal.** In the Factored-FB repo that is
   `obs_mask_dims=[36,37,38]`, which blanks the dims in `observations`/`next_observations` and
   at eval. `goal_mask_dims` blanks only the emitted goal vector and leaves the observation
   intact, so it does **not** close the shortcut.
2. **The Metaworld goal box is much smaller relative to its success radius.** At a 1-2x
   eval-horizon lookahead, P(a relabeled goal is already satisfied) is 0.249 (push) / 0.155
   (pick-place) against cube's 0.013. Temporal distance therefore carries much less
   information here, and this data is a weaker critic discriminator than cube even with
   masking on.
3. **The oracle is closed-loop reactive with no plan and no completion signal**, whereas
   OGBench's is plan-based with an explicit `done`. Segment ends here are a timer rather than
   task completion, and suboptimality comes from action noise rather than randomised plan
   waypoints.
4. `terminal` is all-`False` and `timeout` fires only on the final step — collection is
   fixed-horizon, so trajectory boundaries carry no early-termination information.

## Measured against cube-single-play

Recomputed from both sets of raw arrays by
`scripts/compare_play_resample_to_ogbench.py`, in units of each benchmark's own eval horizon:

| | cube-single-play | mw-push | mw-pick-place |
|---|---|---|---|
| transitions | 1,001,000 | 1,000,000 | 1,000,000 |
| traj length / eval horizon | 5.00 | 5.00 | 5.00 |
| relocations per trajectory | 11.52 | 12.00 | 12.00 |
| segment length / eval horizon | 0.435 | 0.417 | 0.417 |
| object-at-rest fraction | 0.532 | 0.572 | 0.524 |
| achieved-goal entropy / uniform | 1.000 | 1.000 | 1.000 |
| stale goal-channel transitions | 0 | 0 | 0 |
| object escapes workspace | n/a | 0 | 0 |

## Version history

**v3 (2026-08-05, published).** Audited line-by-line against OGBench's own
`generate_manipspace.py` and the Metaworld env internals. Relative to the unpublished v2:

- **Epsilon-random actions removed.** A uniform draw on the gripper dims knocks the puck out of
  the reachable workspace; the stateless Metaworld oracle never recovers it and the collector
  does not reset mid-trajectory, so the failure is **absorbing**. At `eps_max=0.1`, 36% of
  mw-push steps had the puck gone and 4 of 8 trajectories scored zero successes for the rest of
  their length. OGBench sets `p_random_action=0` for all play data.
- **Reject-and-retry on unhealthy trajectories**, as OGBench does for its scene tasks. Even at
  `eps_max=0`, 10.2% of mw-push trajectories escaped at 1M scale, 162 of 203 permanently dead
  (success 0.082 after the escape vs 0.809 overall). Escaped trajectories are now discarded and
  re-rolled; `escape_frac` is 0 and `rejected_trajectories` is recorded in each meta.
- **Timer-only re-draw.** v1 re-drew the instant success fired (10–13 step segments); v2 added a
  post-success dwell on the mistaken reading that cube play "holds" at the goal. It does not —
  cube's ~50%-at-rest is the block sitting still while the arm approaches and then retreats to a
  random idle pose, an artefact of a pick-and-place plan. Under a pure timer the at-rest
  structure emerges instead of being legislated.
- **No more zero-velocity frames.** v2 refreshed the observation with `_get_obs()`, which
  rebuilds Metaworld's `[curr 18 | prev 18 | goal 3]` with both halves equal — a physically
  impossible frame the eval env never produces, at 2.35% (push) / 2.70% (pick-place) of rows,
  landing exactly on the goal-channel jumps. v3 splices only the goal dims.
- **Goal marker and clipping.** The rendered goal site now follows the target (`mj_forward`
  recomputes site positions from `model.site_pos`, so writing `data.site(...).xpos` was
  silently clobbered), and the stored goal channel is clipped exactly as `step()` would clip it.
- **`goal_switch` recorded.** OGBench records no segment marker, which is why its segment
  structure is unrecoverable. This is additive.

**v1/v2** are superseded and not published. v1's defects: goal-channel staleness on 7–10% of
transitions, 100% of re-drawn push targets below the table (`goal_space` z is `U[0.01, 0.02]`
while the env's reset always uses 0.02), `achieved_goal` all zeros, and 18–27% degenerate
1-step segments.

## Reproduce

```bash
MUJOCO_GL=egl python data_collection/collect_mw_play_resample.py \
    --task mw-push --episodes 2000 --seed 0 --out <root>/play-resample
```

## Citation

Collected for the Factored-FB testbed. Metaworld: Yu et al., CoRL 2019. Scripted policies are
Metaworld's own `Sawyer*V2Policy`.
