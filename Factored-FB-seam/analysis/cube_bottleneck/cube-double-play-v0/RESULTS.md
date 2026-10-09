# Cube bottleneck-mechanism probe -- results (cube-double-play-v0)

Spec: `docs/specs/2026-09-04-cube-bottleneck-mechanism-probe.md`. Analysis-only;
no training. All arms at 500k, final checkpoint. Per-seed throughout.

## Gates

### G1 -- H0 reproduces `table_privcoord` within 0.05

| env                 | critic           |   seed |   table |   probe_H0 |   delta | passes   |
|:--------------------|:-----------------|-------:|--------:|-----------:|--------:|:---------|
| cube-double-play-v0 | density_fb       |      0 |    0.6  |      0.592 |  -0.008 | True     |
| cube-double-play-v0 | density_fb       |      1 |    0.64 |      0.616 |  -0.024 | True     |
| cube-double-play-v0 | density_fb       |      2 |    0.68 |      0.66  |  -0.02  | True     |
| cube-double-play-v0 | density_fb_state |      0 |    0.06 |      0.088 |   0.028 | True     |
| cube-double-play-v0 | density_fb_state |      1 |    0.05 |      0.04  |  -0.01  | True     |
| cube-double-play-v0 | density_fb_state |      2 |    0.05 |      0.076 |   0.026 | True     |
| cube-double-play-v0 | fb               |      0 |    0.02 |      0.012 |  -0.008 | True     |
| cube-double-play-v0 | fb_ref           |      0 |    0    |      0.012 |   0.012 | True     |

### G2 -- restore fidelity (held bank)

- kept **64/64** rows (dropped 0.0%; spec ceiling 25%)
- settle: `N_SETTLE = 2` env steps, gripper commanded closed
- bounds: cube drift < 0.04 m, hand-over contact > 0.5

## Read this env as REPORTED, not read

Two independent reasons, both settled before the numbers were looked at:

1. **Both deciding arms are at the floor.** `fb_ref` scores 0.012 and
   `density_fb_state` 0.04-0.09 at H0, so a held-start delta has almost no
   room to move in either direction and separates nothing.
2. **A held start does not remove this task's difficulty.** cube-double
   requires placing BOTH cubes; the bank hands the policy block 0 already
   secured, which leaves the second cube, and the transfer, untouched. The
   method that actually works here (`density_fb`, 0.59-0.66) gains nothing
   from it either (delta +0.02 / -0.07 / +0.02) -- which is the control
   showing the manipulation is not informative rather than FB being immune.

**Phase-label scope.** `_target_block` is 0 for all 5 cube-double tasks
(verified), so the block-0 qpos/obs slices the probe uses are correct. But the
held / transport / drop diagnostics therefore describe **block 0 only**, while
`success` requires both cubes. That is the same convention
`[torch] evals/phase_probe.py` uses, and it means the failure decomposition
below is not directly comparable to cube-single's.

## Recorded facts (spec deliverable 1)

- **Gripper convention.** `ctrl[gripper] = 255 * target_gripper_opening` over a
  driver joint that rises as the Robotiq CLOSES, so **255 (action channel +1) is
  the closing command**. Measured on 16 held rows: action +1 and raw ctrl 255 both
  hold the cube (contact ~0.30 after the probe step); action -1 drops it outright
  (drift 0.16 m). Raw `ctrl = 0` -- what
  `[torch] scripts/value/counterfactual_value_probe.py:165` uses to "establish the
  grasp" -- commands the gripper OPEN: it passes the purely geometric `held` test
  while contact collapses to 0.01, a latent drop. G2 therefore checks contact at
  hand-over as well as geometry.
- **IK integrator.** `manipspace_env.set_control` recomputes effector pose, yaw and
  gripper opening from `_data` on every call and consumes `_target_effector_pose`
  inside that same call, so there is **no cross-step hidden IK state**; spec step 3
  is a no-op here. (`[torch] scripts/atlas/prefix_replay.py`'s docstring does not
  hold for this build's control path.) Confirmed empirically: every observation
  channel except contact reconstructs to < 0.01 from (qpos, qvel) alone.
- **Contact needs rebuilding.** `obs[18]` is `||cfrc_ext(right_pad)||/50`, which
  depends on `data.ctrl` -- hidden state no dataset records. `mj_forward` alone
  leaves it at 0.00; adding `mj_rnePostConstraint` reaches 0.16; two settle steps
  with the gripper commanded closed reach **0.964 against a recorded 0.963**.
  `N_SETTLE = 2` (not the spec's 5) is the measured optimum: drift grows with every
  further step and G2 starts failing at 20.
- **Restore is in-distribution.** 98.7% of restored observation channels fall inside
  the dataset's own held-state 1-99 percentile band -- above the recorded rows'
  98.1%. The only degraded channel is joint velocity (the arm decelerates during
  the settle), itself in-band.

## Test 1 -- held-start evaluation

Success, mean over the env's 5 tasks, per seed:

| env                 | critic           |   seed |    H0 |    H1 |   H2 |   delta_held |
|:--------------------|:-----------------|-------:|------:|------:|-----:|-------------:|
| cube-double-play-v0 | density_fb       |      0 | 0.592 | 0.612 | 0.58 |        0.02  |
| cube-double-play-v0 | density_fb       |      1 | 0.616 | 0.544 | 0.54 |       -0.072 |
| cube-double-play-v0 | density_fb       |      2 | 0.66  | 0.68  | 0.6  |        0.02  |
| cube-double-play-v0 | density_fb_state |      0 | 0.088 | 0.052 | 0.1  |       -0.036 |
| cube-double-play-v0 | density_fb_state |      1 | 0.04  | 0.024 | 0.06 |       -0.016 |
| cube-double-play-v0 | density_fb_state |      2 | 0.076 | 0.02  | 0.02 |       -0.056 |
| cube-double-play-v0 | fb               |      0 | 0.012 | 0.012 | 0.04 |        0     |
| cube-double-play-v0 | fb_ref           |      0 | 0.012 | 0.004 | 0.02 |       -0.008 |

### H1 diagnostics

| env                 | critic           |   seed |   hold_retention_50 |   time_to_first_drop |   drop_censored_frac |   maintain_fail_frac |   transport_fail_frac |   dropped_frac |   near_goal_frac |   far_goal_frac |   n_fail |   n |
|:--------------------|:-----------------|-------:|--------------------:|---------------------:|---------------------:|---------------------:|----------------------:|---------------:|-----------------:|----------------:|---------:|----:|
| cube-double-play-v0 | density_fb       |      0 |               0.706 |               20.516 |                0     |                0.67  |                 0.33  |          0.67  |            0     |           0.33  |       97 | 250 |
| cube-double-play-v0 | density_fb       |      1 |               0.735 |               23.968 |                0     |                0.649 |                 0.351 |          0.649 |            0     |           0.351 |      114 | 250 |
| cube-double-play-v0 | density_fb       |      2 |               0.72  |               20.309 |                0.004 |                0.738 |                 0.262 |          0.738 |            0     |           0.262 |       80 | 250 |
| cube-double-play-v0 | density_fb_state |      0 |               0.71  |               17.848 |                0     |                0.738 |                 0.262 |          0.738 |            0.004 |           0.257 |      237 | 250 |
| cube-double-play-v0 | density_fb_state |      1 |               0.638 |               18.904 |                0     |                0.787 |                 0.213 |          0.787 |            0     |           0.213 |      244 | 250 |
| cube-double-play-v0 | density_fb_state |      2 |               0.624 |               17.048 |                0     |                0.804 |                 0.196 |          0.804 |            0.008 |           0.188 |      245 | 250 |
| cube-double-play-v0 | fb               |      0 |               0.721 |               20.032 |                0     |                0.773 |                 0.227 |          0.773 |            0.004 |           0.223 |      247 | 250 |
| cube-double-play-v0 | fb_ref           |      0 |               0.701 |               19.796 |                0     |                0.823 |                 0.177 |          0.823 |            0     |           0.177 |      249 | 250 |

<details><summary>Per-task detail with bootstrap CIs</summary>

| env                 | critic           |   seed | start   |   task |   success |   count |   ci_lo |   ci_hi |
|:--------------------|:-----------------|-------:|:--------|-------:|----------:|--------:|--------:|--------:|
| cube-double-play-v0 | density_fb       |      0 | H0      |      1 |      0.94 |      50 |    0.86 |    1    |
| cube-double-play-v0 | density_fb       |      0 | H0      |      2 |      0.72 |      50 |    0.6  |    0.84 |
| cube-double-play-v0 | density_fb       |      0 | H0      |      3 |      0.54 |      50 |    0.4  |    0.68 |
| cube-double-play-v0 | density_fb       |      0 | H0      |      4 |      0.24 |      50 |    0.12 |    0.36 |
| cube-double-play-v0 | density_fb       |      0 | H0      |      5 |      0.52 |      50 |    0.38 |    0.66 |
| cube-double-play-v0 | density_fb       |      0 | H1      |      1 |      0.7  |      50 |    0.58 |    0.82 |
| cube-double-play-v0 | density_fb       |      0 | H1      |      2 |      0.66 |      50 |    0.52 |    0.78 |
| cube-double-play-v0 | density_fb       |      0 | H1      |      3 |      0.54 |      50 |    0.4  |    0.68 |
| cube-double-play-v0 | density_fb       |      0 | H1      |      4 |      0.74 |      50 |    0.62 |    0.86 |
| cube-double-play-v0 | density_fb       |      0 | H1      |      5 |      0.42 |      50 |    0.3  |    0.56 |
| cube-double-play-v0 | density_fb       |      0 | H2      |      1 |      0.9  |      10 |    0.7  |    1    |
| cube-double-play-v0 | density_fb       |      0 | H2      |      2 |      0.4  |      10 |    0.1  |    0.7  |
| cube-double-play-v0 | density_fb       |      0 | H2      |      3 |      0.4  |      10 |    0.1  |    0.7  |
| cube-double-play-v0 | density_fb       |      0 | H2      |      4 |      0.6  |      10 |    0.3  |    0.9  |
| cube-double-play-v0 | density_fb       |      0 | H2      |      5 |      0.6  |      10 |    0.3  |    0.9  |
| cube-double-play-v0 | density_fb       |      1 | H0      |      1 |      1    |      50 |    1    |    1    |
| cube-double-play-v0 | density_fb       |      1 | H0      |      2 |      0.58 |      50 |    0.44 |    0.72 |
| cube-double-play-v0 | density_fb       |      1 | H0      |      3 |      0.72 |      50 |    0.6  |    0.84 |
| cube-double-play-v0 | density_fb       |      1 | H0      |      4 |      0.28 |      50 |    0.16 |    0.42 |
| cube-double-play-v0 | density_fb       |      1 | H0      |      5 |      0.5  |      50 |    0.36 |    0.64 |
| cube-double-play-v0 | density_fb       |      1 | H1      |      1 |      0.68 |      50 |    0.54 |    0.82 |
| cube-double-play-v0 | density_fb       |      1 | H1      |      2 |      0.6  |      50 |    0.46 |    0.74 |
| cube-double-play-v0 | density_fb       |      1 | H1      |      3 |      0.56 |      50 |    0.42 |    0.7  |
| cube-double-play-v0 | density_fb       |      1 | H1      |      4 |      0.66 |      50 |    0.52 |    0.78 |
| cube-double-play-v0 | density_fb       |      1 | H1      |      5 |      0.22 |      50 |    0.1  |    0.34 |
| cube-double-play-v0 | density_fb       |      1 | H2      |      1 |      0.6  |      10 |    0.3  |    0.9  |
| cube-double-play-v0 | density_fb       |      1 | H2      |      2 |      0.6  |      10 |    0.3  |    0.9  |
| cube-double-play-v0 | density_fb       |      1 | H2      |      3 |      0.5  |      10 |    0.2  |    0.8  |
| cube-double-play-v0 | density_fb       |      1 | H2      |      4 |      0.4  |      10 |    0.1  |    0.7  |
| cube-double-play-v0 | density_fb       |      1 | H2      |      5 |      0.6  |      10 |    0.3  |    0.9  |
| cube-double-play-v0 | density_fb       |      2 | H0      |      1 |      1    |      50 |    1    |    1    |
| cube-double-play-v0 | density_fb       |      2 | H0      |      2 |      0.88 |      50 |    0.78 |    0.96 |
| cube-double-play-v0 | density_fb       |      2 | H0      |      3 |      0.6  |      50 |    0.46 |    0.74 |
| cube-double-play-v0 | density_fb       |      2 | H0      |      4 |      0.3  |      50 |    0.18 |    0.44 |
| cube-double-play-v0 | density_fb       |      2 | H0      |      5 |      0.52 |      50 |    0.38 |    0.66 |
| cube-double-play-v0 | density_fb       |      2 | H1      |      1 |      0.76 |      50 |    0.64 |    0.86 |
| cube-double-play-v0 | density_fb       |      2 | H1      |      2 |      0.8  |      50 |    0.68 |    0.9  |
| cube-double-play-v0 | density_fb       |      2 | H1      |      3 |      0.74 |      50 |    0.62 |    0.86 |
| cube-double-play-v0 | density_fb       |      2 | H1      |      4 |      0.64 |      50 |    0.5  |    0.76 |
| cube-double-play-v0 | density_fb       |      2 | H1      |      5 |      0.46 |      50 |    0.32 |    0.58 |
| cube-double-play-v0 | density_fb       |      2 | H2      |      1 |      0.6  |      10 |    0.3  |    0.9  |
| cube-double-play-v0 | density_fb       |      2 | H2      |      2 |      0.8  |      10 |    0.5  |    1    |
| cube-double-play-v0 | density_fb       |      2 | H2      |      3 |      0.7  |      10 |    0.4  |    1    |
| cube-double-play-v0 | density_fb       |      2 | H2      |      4 |      0.4  |      10 |    0.1  |    0.7  |
| cube-double-play-v0 | density_fb       |      2 | H2      |      5 |      0.5  |      10 |    0.2  |    0.8  |
| cube-double-play-v0 | density_fb_state |      0 | H0      |      1 |      0.34 |      50 |    0.22 |    0.48 |
| cube-double-play-v0 | density_fb_state |      0 | H0      |      2 |      0.08 |      50 |    0.02 |    0.16 |
| cube-double-play-v0 | density_fb_state |      0 | H0      |      3 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      0 | H0      |      4 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      0 | H0      |      5 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      0 | H1      |      1 |      0.12 |      50 |    0.04 |    0.22 |
| cube-double-play-v0 | density_fb_state |      0 | H1      |      2 |      0.04 |      50 |    0    |    0.1  |
| cube-double-play-v0 | density_fb_state |      0 | H1      |      3 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      0 | H1      |      4 |      0.06 |      50 |    0    |    0.12 |
| cube-double-play-v0 | density_fb_state |      0 | H1      |      5 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      0 | H2      |      1 |      0.3  |      10 |    0    |    0.6  |
| cube-double-play-v0 | density_fb_state |      0 | H2      |      2 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      0 | H2      |      3 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      0 | H2      |      4 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      0 | H2      |      5 |      0.2  |      10 |    0    |    0.5  |
| cube-double-play-v0 | density_fb_state |      1 | H0      |      1 |      0.16 |      50 |    0.08 |    0.28 |
| cube-double-play-v0 | density_fb_state |      1 | H0      |      2 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      1 | H0      |      3 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      1 | H0      |      4 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      1 | H0      |      5 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      1 | H1      |      1 |      0.04 |      50 |    0    |    0.1  |
| cube-double-play-v0 | density_fb_state |      1 | H1      |      2 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      1 | H1      |      3 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      1 | H1      |      4 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      1 | H1      |      5 |      0.04 |      50 |    0    |    0.1  |
| cube-double-play-v0 | density_fb_state |      1 | H2      |      1 |      0.2  |      10 |    0    |    0.5  |
| cube-double-play-v0 | density_fb_state |      1 | H2      |      2 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      1 | H2      |      3 |      0.1  |      10 |    0    |    0.3  |
| cube-double-play-v0 | density_fb_state |      1 | H2      |      4 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      1 | H2      |      5 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H0      |      1 |      0.34 |      50 |    0.22 |    0.48 |
| cube-double-play-v0 | density_fb_state |      2 | H0      |      2 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H0      |      3 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      2 | H0      |      4 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H0      |      5 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      2 | H1      |      1 |      0.04 |      50 |    0    |    0.1  |
| cube-double-play-v0 | density_fb_state |      2 | H1      |      2 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | density_fb_state |      2 | H1      |      3 |      0.04 |      50 |    0    |    0.1  |
| cube-double-play-v0 | density_fb_state |      2 | H1      |      4 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H1      |      5 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H2      |      1 |      0.1  |      10 |    0    |    0.3  |
| cube-double-play-v0 | density_fb_state |      2 | H2      |      2 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H2      |      3 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H2      |      4 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | density_fb_state |      2 | H2      |      5 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H0      |      1 |      0.06 |      50 |    0    |    0.14 |
| cube-double-play-v0 | fb               |      0 | H0      |      2 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H0      |      3 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H0      |      4 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H0      |      5 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H1      |      1 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H1      |      2 |      0.04 |      50 |    0    |    0.1  |
| cube-double-play-v0 | fb               |      0 | H1      |      3 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H1      |      4 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | fb               |      0 | H1      |      5 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H2      |      1 |      0.2  |      10 |    0    |    0.5  |
| cube-double-play-v0 | fb               |      0 | H2      |      2 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H2      |      3 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H2      |      4 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb               |      0 | H2      |      5 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H0      |      1 |      0.06 |      50 |    0    |    0.14 |
| cube-double-play-v0 | fb_ref           |      0 | H0      |      2 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H0      |      3 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H0      |      4 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H0      |      5 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H1      |      1 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H1      |      2 |      0.02 |      50 |    0    |    0.06 |
| cube-double-play-v0 | fb_ref           |      0 | H1      |      3 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H1      |      4 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H1      |      5 |      0    |      50 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H2      |      1 |      0.1  |      10 |    0    |    0.3  |
| cube-double-play-v0 | fb_ref           |      0 | H2      |      2 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H2      |      3 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H2      |      4 |      0    |      10 |    0    |    0    |
| cube-double-play-v0 | fb_ref           |      0 | H2      |      5 |      0    |      10 |    0    |    0    |

</details>

## Pre-registered readings

Every reading in the spec is called below, including those that do not fire and
the conjunct that blocked each.

### does not fire -- FLOOR (density_fb_state loses >0.15 H0->H1 on >=2 seeds)

- detail: seeds losing >0.15: []
- blocked by: fewer than 2 seeds lost >0.15 -- readings stand

### does not fire -- R-HELD1 (bottleneck is load-bearing for FB)

- detail: delta>=0.30 on seeds []; of those within 0.10 of Density-FB: []
- blocked by: only 0 seed(s) reach delta>=0.30

### does not fire -- R-HELD2 (deficit lives AFTER the bottleneck)

- detail: delta<=0.10 on seeds [0]; retention<0.6 on []; maintain+dropped>=50% on [0]; Density-FB stable on [0]
- blocked by: only 1 seed(s) have delta<=0.10

### does not fire -- R-HELD3 (mixed: FB gains but stays far below)

- detail: seeds in (0.10, 0.30) with a >0.10 gap: []
- blocked by: only 0 seed(s) qualify

