# Cube bottleneck-mechanism probe -- results (cube-single-play-v0)

Spec: `docs/specs/2026-09-04-cube-bottleneck-mechanism-probe.md`. Analysis-only;
no training. All arms at 500k, final checkpoint. Per-seed throughout.

## Gates

### G1 -- H0 reproduces `table_privcoord` within 0.05

| env                 | critic           |   seed |   table |   probe_H0 |   delta | passes   |
|:--------------------|:-----------------|-------:|--------:|-----------:|--------:|:---------|
| cube-single-play-v0 | density_fb       |      0 |    0.95 |      0.928 |  -0.022 | True     |
| cube-single-play-v0 | density_fb       |      1 |    0.94 |      0.964 |   0.024 | True     |
| cube-single-play-v0 | density_fb       |      2 |    0.94 |      0.948 |   0.008 | True     |
| cube-single-play-v0 | density_fb_state |      0 |    0.92 |      0.908 |  -0.012 | True     |
| cube-single-play-v0 | density_fb_state |      1 |    0.89 |      0.884 |  -0.006 | True     |
| cube-single-play-v0 | density_fb_state |      2 |    0.92 |      0.9   |  -0.02  | True     |
| cube-single-play-v0 | fb               |      0 |    0.11 |      0.096 |  -0.014 | True     |
| cube-single-play-v0 | fb_ref           |      0 |    0.23 |      0.244 |   0.014 | True     |
| cube-single-play-v0 | fb_ref           |      1 |    0.15 |      0.148 |  -0.002 | True     |
| cube-single-play-v0 | fb_ref           |      2 |    0.23 |      0.224 |  -0.006 | True     |

### G2 -- restore fidelity (held bank)

- kept **64/64** rows (dropped 0.0%; spec ceiling 25%)
- settle: `N_SETTLE = 2` env steps, gripper commanded closed
- bounds: cube drift < 0.04 m, hand-over contact > 0.5

### G4 -- P2 states are decision points

- P2 states with `p_secure` std > 0.20: **44/64** (need 40) -> **PASS**
- median std: P2 0.274, P1 0.041 (null), P3 0.216 (null)

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
| cube-single-play-v0 | density_fb       |      0 | 0.928 | 1     | 0.84 |        0.072 |
| cube-single-play-v0 | density_fb       |      1 | 0.964 | 1     | 0.92 |        0.036 |
| cube-single-play-v0 | density_fb       |      2 | 0.948 | 1     | 0.74 |        0.052 |
| cube-single-play-v0 | density_fb_state |      0 | 0.908 | 0.98  | 0.96 |        0.072 |
| cube-single-play-v0 | density_fb_state |      1 | 0.884 | 0.996 | 0.86 |        0.112 |
| cube-single-play-v0 | density_fb_state |      2 | 0.9   | 0.992 | 0.7  |        0.092 |
| cube-single-play-v0 | fb               |      0 | 0.096 | 0.16  | 0.18 |        0.064 |
| cube-single-play-v0 | fb_ref           |      0 | 0.244 | 0.264 | 0.16 |        0.02  |
| cube-single-play-v0 | fb_ref           |      1 | 0.148 | 0.248 | 0.26 |        0.1   |
| cube-single-play-v0 | fb_ref           |      2 | 0.224 | 0.256 | 0.22 |        0.032 |

### H1 diagnostics

| env                 | critic           |   seed |   hold_retention_50 |   time_to_first_drop |   drop_censored_frac |   maintain_fail_frac |   transport_fail_frac |   dropped_frac |   near_goal_frac |   far_goal_frac |   n_fail |   n |
|:--------------------|:-----------------|-------:|--------------------:|---------------------:|---------------------:|---------------------:|----------------------:|---------------:|-----------------:|----------------:|---------:|----:|
| cube-single-play-v0 | density_fb       |      0 |               0.875 |               16.777 |                0.248 |                0     |                 0     |          0     |                0 |           0     |        0 | 250 |
| cube-single-play-v0 | density_fb       |      1 |               0.891 |               17.967 |                0.28  |                0     |                 0     |          0     |                0 |           0     |        0 | 250 |
| cube-single-play-v0 | density_fb       |      2 |               0.894 |               17.642 |                0.34  |                0     |                 0     |          0     |                0 |           0     |        0 | 250 |
| cube-single-play-v0 | density_fb_state |      0 |               0.808 |               18.035 |                0.076 |                0.6   |                 0.4   |          0.6   |                0 |           0.4   |        5 | 250 |
| cube-single-play-v0 | density_fb_state |      1 |               0.825 |               17.827 |                0.144 |                1     |                 0     |          1     |                0 |           0     |        1 | 250 |
| cube-single-play-v0 | density_fb_state |      2 |               0.86  |               20.607 |                0.176 |                1     |                 0     |          1     |                0 |           0     |        2 | 250 |
| cube-single-play-v0 | fb               |      0 |               0.742 |               18.703 |                0.004 |                0.671 |                 0.329 |          0.671 |                0 |           0.329 |      210 | 250 |
| cube-single-play-v0 | fb_ref           |      0 |               0.751 |               18.546 |                0.004 |                0.495 |                 0.505 |          0.495 |                0 |           0.505 |      184 | 250 |
| cube-single-play-v0 | fb_ref           |      1 |               0.695 |               14.763 |                0.02  |                0.559 |                 0.441 |          0.559 |                0 |           0.441 |      188 | 250 |
| cube-single-play-v0 | fb_ref           |      2 |               0.699 |               13.138 |                0.012 |                0.548 |                 0.452 |          0.548 |                0 |           0.452 |      186 | 250 |

<details><summary>Per-task detail with bootstrap CIs</summary>

| env                 | critic           |   seed | start   |   task |   success |   count |   ci_lo |   ci_hi |
|:--------------------|:-----------------|-------:|:--------|-------:|----------:|--------:|--------:|--------:|
| cube-single-play-v0 | density_fb       |      0 | H0      |      1 |      0.94 |      50 |    0.86 |    1    |
| cube-single-play-v0 | density_fb       |      0 | H0      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      0 | H0      |      3 |      0.96 |      50 |    0.9  |    1    |
| cube-single-play-v0 | density_fb       |      0 | H0      |      4 |      0.82 |      50 |    0.7  |    0.92 |
| cube-single-play-v0 | density_fb       |      0 | H0      |      5 |      0.92 |      50 |    0.84 |    0.98 |
| cube-single-play-v0 | density_fb       |      0 | H1      |      1 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      0 | H1      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      0 | H1      |      3 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      0 | H1      |      4 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      0 | H1      |      5 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      0 | H2      |      1 |      0.9  |      10 |    0.7  |    1    |
| cube-single-play-v0 | density_fb       |      0 | H2      |      2 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      0 | H2      |      3 |      0.9  |      10 |    0.7  |    1    |
| cube-single-play-v0 | density_fb       |      0 | H2      |      4 |      0.8  |      10 |    0.5  |    1    |
| cube-single-play-v0 | density_fb       |      0 | H2      |      5 |      0.6  |      10 |    0.3  |    0.9  |
| cube-single-play-v0 | density_fb       |      1 | H0      |      1 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H0      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H0      |      3 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H0      |      4 |      0.9  |      50 |    0.82 |    0.98 |
| cube-single-play-v0 | density_fb       |      1 | H0      |      5 |      0.92 |      50 |    0.84 |    0.98 |
| cube-single-play-v0 | density_fb       |      1 | H1      |      1 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H1      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H1      |      3 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H1      |      4 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H1      |      5 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H2      |      1 |      0.9  |      10 |    0.7  |    1    |
| cube-single-play-v0 | density_fb       |      1 | H2      |      2 |      0.8  |      10 |    0.5  |    1    |
| cube-single-play-v0 | density_fb       |      1 | H2      |      3 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H2      |      4 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      1 | H2      |      5 |      0.9  |      10 |    0.7  |    1    |
| cube-single-play-v0 | density_fb       |      2 | H0      |      1 |      0.94 |      50 |    0.86 |    1    |
| cube-single-play-v0 | density_fb       |      2 | H0      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H0      |      3 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H0      |      4 |      0.92 |      50 |    0.84 |    0.98 |
| cube-single-play-v0 | density_fb       |      2 | H0      |      5 |      0.88 |      50 |    0.78 |    0.96 |
| cube-single-play-v0 | density_fb       |      2 | H1      |      1 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H1      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H1      |      3 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H1      |      4 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H1      |      5 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H2      |      1 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb       |      2 | H2      |      2 |      0.8  |      10 |    0.5  |    1    |
| cube-single-play-v0 | density_fb       |      2 | H2      |      3 |      0.5  |      10 |    0.2  |    0.8  |
| cube-single-play-v0 | density_fb       |      2 | H2      |      4 |      0.7  |      10 |    0.4  |    1    |
| cube-single-play-v0 | density_fb       |      2 | H2      |      5 |      0.7  |      10 |    0.4  |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H0      |      1 |      0.96 |      50 |    0.9  |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H0      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H0      |      3 |      0.96 |      50 |    0.9  |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H0      |      4 |      0.76 |      50 |    0.64 |    0.88 |
| cube-single-play-v0 | density_fb_state |      0 | H0      |      5 |      0.86 |      50 |    0.76 |    0.96 |
| cube-single-play-v0 | density_fb_state |      0 | H1      |      1 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H1      |      2 |      0.96 |      50 |    0.9  |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H1      |      3 |      0.98 |      50 |    0.94 |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H1      |      4 |      0.98 |      50 |    0.94 |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H1      |      5 |      0.98 |      50 |    0.94 |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H2      |      1 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H2      |      2 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H2      |      3 |      0.8  |      10 |    0.5  |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H2      |      4 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      0 | H2      |      5 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H0      |      1 |      0.84 |      50 |    0.72 |    0.94 |
| cube-single-play-v0 | density_fb_state |      1 | H0      |      2 |      0.94 |      50 |    0.86 |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H0      |      3 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H0      |      4 |      0.82 |      50 |    0.7  |    0.92 |
| cube-single-play-v0 | density_fb_state |      1 | H0      |      5 |      0.82 |      50 |    0.7  |    0.92 |
| cube-single-play-v0 | density_fb_state |      1 | H1      |      1 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H1      |      2 |      0.98 |      50 |    0.94 |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H1      |      3 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H1      |      4 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H1      |      5 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H2      |      1 |      0.9  |      10 |    0.7  |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H2      |      2 |      0.9  |      10 |    0.7  |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H2      |      3 |      0.9  |      10 |    0.7  |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H2      |      4 |      0.8  |      10 |    0.5  |    1    |
| cube-single-play-v0 | density_fb_state |      1 | H2      |      5 |      0.8  |      10 |    0.5  |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H0      |      1 |      0.9  |      50 |    0.8  |    0.98 |
| cube-single-play-v0 | density_fb_state |      2 | H0      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H0      |      3 |      0.98 |      50 |    0.94 |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H0      |      4 |      0.82 |      50 |    0.7  |    0.92 |
| cube-single-play-v0 | density_fb_state |      2 | H0      |      5 |      0.8  |      50 |    0.68 |    0.9  |
| cube-single-play-v0 | density_fb_state |      2 | H1      |      1 |      0.98 |      50 |    0.94 |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H1      |      2 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H1      |      3 |      0.98 |      50 |    0.94 |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H1      |      4 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H1      |      5 |      1    |      50 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H2      |      1 |      1    |      10 |    1    |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H2      |      2 |      0.6  |      10 |    0.3  |    0.9  |
| cube-single-play-v0 | density_fb_state |      2 | H2      |      3 |      0.7  |      10 |    0.4  |    1    |
| cube-single-play-v0 | density_fb_state |      2 | H2      |      4 |      0.6  |      10 |    0.3  |    0.9  |
| cube-single-play-v0 | density_fb_state |      2 | H2      |      5 |      0.6  |      10 |    0.3  |    0.9  |
| cube-single-play-v0 | fb               |      0 | H0      |      1 |      0.06 |      50 |    0    |    0.14 |
| cube-single-play-v0 | fb               |      0 | H0      |      2 |      0.08 |      50 |    0.02 |    0.16 |
| cube-single-play-v0 | fb               |      0 | H0      |      3 |      0.14 |      50 |    0.06 |    0.24 |
| cube-single-play-v0 | fb               |      0 | H0      |      4 |      0.14 |      50 |    0.06 |    0.26 |
| cube-single-play-v0 | fb               |      0 | H0      |      5 |      0.06 |      50 |    0    |    0.14 |
| cube-single-play-v0 | fb               |      0 | H1      |      1 |      0.1  |      50 |    0.02 |    0.2  |
| cube-single-play-v0 | fb               |      0 | H1      |      2 |      0.18 |      50 |    0.08 |    0.3  |
| cube-single-play-v0 | fb               |      0 | H1      |      3 |      0.2  |      50 |    0.1  |    0.32 |
| cube-single-play-v0 | fb               |      0 | H1      |      4 |      0.22 |      50 |    0.1  |    0.34 |
| cube-single-play-v0 | fb               |      0 | H1      |      5 |      0.1  |      50 |    0.02 |    0.2  |
| cube-single-play-v0 | fb               |      0 | H2      |      1 |      0.3  |      10 |    0    |    0.6  |
| cube-single-play-v0 | fb               |      0 | H2      |      2 |      0.3  |      10 |    0.1  |    0.6  |
| cube-single-play-v0 | fb               |      0 | H2      |      3 |      0.3  |      10 |    0    |    0.6  |
| cube-single-play-v0 | fb               |      0 | H2      |      4 |      0    |      10 |    0    |    0    |
| cube-single-play-v0 | fb               |      0 | H2      |      5 |      0    |      10 |    0    |    0    |
| cube-single-play-v0 | fb_ref           |      0 | H0      |      1 |      0.2  |      50 |    0.1  |    0.32 |
| cube-single-play-v0 | fb_ref           |      0 | H0      |      2 |      0.18 |      50 |    0.08 |    0.3  |
| cube-single-play-v0 | fb_ref           |      0 | H0      |      3 |      0.38 |      50 |    0.24 |    0.52 |
| cube-single-play-v0 | fb_ref           |      0 | H0      |      4 |      0.24 |      50 |    0.12 |    0.36 |
| cube-single-play-v0 | fb_ref           |      0 | H0      |      5 |      0.22 |      50 |    0.12 |    0.34 |
| cube-single-play-v0 | fb_ref           |      0 | H1      |      1 |      0.22 |      50 |    0.12 |    0.34 |
| cube-single-play-v0 | fb_ref           |      0 | H1      |      2 |      0.18 |      50 |    0.08 |    0.28 |
| cube-single-play-v0 | fb_ref           |      0 | H1      |      3 |      0.38 |      50 |    0.26 |    0.52 |
| cube-single-play-v0 | fb_ref           |      0 | H1      |      4 |      0.32 |      50 |    0.2  |    0.46 |
| cube-single-play-v0 | fb_ref           |      0 | H1      |      5 |      0.22 |      50 |    0.12 |    0.34 |
| cube-single-play-v0 | fb_ref           |      0 | H2      |      1 |      0    |      10 |    0    |    0    |
| cube-single-play-v0 | fb_ref           |      0 | H2      |      2 |      0.1  |      10 |    0    |    0.3  |
| cube-single-play-v0 | fb_ref           |      0 | H2      |      3 |      0.2  |      10 |    0    |    0.5  |
| cube-single-play-v0 | fb_ref           |      0 | H2      |      4 |      0.3  |      10 |    0    |    0.6  |
| cube-single-play-v0 | fb_ref           |      0 | H2      |      5 |      0.2  |      10 |    0    |    0.5  |
| cube-single-play-v0 | fb_ref           |      1 | H0      |      1 |      0.06 |      50 |    0    |    0.12 |
| cube-single-play-v0 | fb_ref           |      1 | H0      |      2 |      0.18 |      50 |    0.08 |    0.3  |
| cube-single-play-v0 | fb_ref           |      1 | H0      |      3 |      0.26 |      50 |    0.14 |    0.38 |
| cube-single-play-v0 | fb_ref           |      1 | H0      |      4 |      0.14 |      50 |    0.06 |    0.24 |
| cube-single-play-v0 | fb_ref           |      1 | H0      |      5 |      0.1  |      50 |    0.02 |    0.18 |
| cube-single-play-v0 | fb_ref           |      1 | H1      |      1 |      0.16 |      50 |    0.06 |    0.28 |
| cube-single-play-v0 | fb_ref           |      1 | H1      |      2 |      0.2  |      50 |    0.1  |    0.32 |
| cube-single-play-v0 | fb_ref           |      1 | H1      |      3 |      0.52 |      50 |    0.38 |    0.66 |
| cube-single-play-v0 | fb_ref           |      1 | H1      |      4 |      0.14 |      50 |    0.06 |    0.24 |
| cube-single-play-v0 | fb_ref           |      1 | H1      |      5 |      0.22 |      50 |    0.12 |    0.34 |
| cube-single-play-v0 | fb_ref           |      1 | H2      |      1 |      0    |      10 |    0    |    0    |
| cube-single-play-v0 | fb_ref           |      1 | H2      |      2 |      0.4  |      10 |    0.1  |    0.7  |
| cube-single-play-v0 | fb_ref           |      1 | H2      |      3 |      0.4  |      10 |    0.1  |    0.7  |
| cube-single-play-v0 | fb_ref           |      1 | H2      |      4 |      0.3  |      10 |    0    |    0.6  |
| cube-single-play-v0 | fb_ref           |      1 | H2      |      5 |      0.2  |      10 |    0    |    0.5  |
| cube-single-play-v0 | fb_ref           |      2 | H0      |      1 |      0.04 |      50 |    0    |    0.1  |
| cube-single-play-v0 | fb_ref           |      2 | H0      |      2 |      0.4  |      50 |    0.26 |    0.54 |
| cube-single-play-v0 | fb_ref           |      2 | H0      |      3 |      0.26 |      50 |    0.14 |    0.38 |
| cube-single-play-v0 | fb_ref           |      2 | H0      |      4 |      0.18 |      50 |    0.08 |    0.3  |
| cube-single-play-v0 | fb_ref           |      2 | H0      |      5 |      0.24 |      50 |    0.12 |    0.36 |
| cube-single-play-v0 | fb_ref           |      2 | H1      |      1 |      0.16 |      50 |    0.06 |    0.26 |
| cube-single-play-v0 | fb_ref           |      2 | H1      |      2 |      0.44 |      50 |    0.3  |    0.58 |
| cube-single-play-v0 | fb_ref           |      2 | H1      |      3 |      0.22 |      50 |    0.12 |    0.34 |
| cube-single-play-v0 | fb_ref           |      2 | H1      |      4 |      0.2  |      50 |    0.1  |    0.32 |
| cube-single-play-v0 | fb_ref           |      2 | H1      |      5 |      0.26 |      50 |    0.14 |    0.38 |
| cube-single-play-v0 | fb_ref           |      2 | H2      |      1 |      0    |      10 |    0    |    0    |
| cube-single-play-v0 | fb_ref           |      2 | H2      |      2 |      0.5  |      10 |    0.2  |    0.8  |
| cube-single-play-v0 | fb_ref           |      2 | H2      |      3 |      0.2  |      10 |    0    |    0.5  |
| cube-single-play-v0 | fb_ref           |      2 | H2      |      4 |      0.3  |      10 |    0    |    0.6  |
| cube-single-play-v0 | fb_ref           |      2 | H2      |      5 |      0.1  |      10 |    0    |    0.3  |

</details>

## Test 2 -- phase-conditioned measure probe

| env                 | critic           |   seed | phase   |   auc_heldmass |   auc_q |   spearman_heldmass |   coverage |   row_mass |   n_cells |   row_mass_ratio |
|:--------------------|:-----------------|-------:|:--------|---------------:|--------:|--------------------:|-----------:|-----------:|----------:|-----------------:|
| cube-single-play-v0 | density_fb       |      0 | P1      |        nan     | nan     |               0.135 |      0     |      0.001 |      1920 |            1     |
| cube-single-play-v0 | density_fb       |      0 | P2      |          0.778 |   0.545 |               0.33  |      0.797 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb       |      0 | P3      |          0.562 |   0.429 |               0.027 |      0.766 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb       |      1 | P1      |        nan     | nan     |               0.131 |      0     |      0.001 |      1920 |            1     |
| cube-single-play-v0 | density_fb       |      1 | P2      |          0.818 |   0.85  |               0.363 |      0.797 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb       |      1 | P3      |          0.5   |   0.5   |              -0.068 |      0.766 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb       |      2 | P1      |        nan     | nan     |               0.027 |      0     |      0.001 |      1920 |            1     |
| cube-single-play-v0 | density_fb       |      2 | P2      |          0.727 |   0.704 |               0.246 |      0.797 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb       |      2 | P3      |          0.545 |   0.472 |               0.024 |      0.766 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb_state |      0 | P1      |        nan     | nan     |               0.129 |      0     |      0.001 |      1920 |            1     |
| cube-single-play-v0 | density_fb_state |      0 | P2      |          0.438 |   0.55  |              -0.047 |      0.797 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb_state |      0 | P3      |          0.519 |   0.5   |              -0.027 |      0.766 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb_state |      1 | P1      |        nan     | nan     |              -0.021 |      0     |      0.001 |      1920 |            1     |
| cube-single-play-v0 | density_fb_state |      1 | P2      |          0.545 |   0.5   |               0.036 |      0.797 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb_state |      1 | P3      |          0.543 |   0.469 |               0.002 |      0.766 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb_state |      2 | P1      |        nan     | nan     |              -0.018 |      0     |      0.001 |      1920 |            1     |
| cube-single-play-v0 | density_fb_state |      2 | P2      |          0.636 |   0.5   |               0.084 |      0.797 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | density_fb_state |      2 | P3      |          0.5   |   0.457 |              -0.036 |      0.766 |      0.001 |      3840 |            1     |
| cube-single-play-v0 | fb               |      0 | P1      |        nan     | nan     |              -0.175 |      0     |     87.399 |      1920 |            0.942 |
| cube-single-play-v0 | fb               |      0 | P2      |          0.909 |   0.909 |               0.48  |      0.797 |     94.071 |      3840 |            0.942 |
| cube-single-play-v0 | fb               |      0 | P3      |          0.656 |   0.694 |               0.204 |      0.766 |     88.637 |      3840 |            0.942 |
| cube-single-play-v0 | fb_ref           |      0 | P1      |        nan     | nan     |              -0.085 |      0     |     95.275 |      1920 |            0.975 |
| cube-single-play-v0 | fb_ref           |      0 | P2      |          0.55  |   0.95  |               0.008 |      0.797 |    103.279 |      3840 |            0.975 |
| cube-single-play-v0 | fb_ref           |      0 | P3      |          0.469 |   0.657 |              -0.056 |      0.766 |    100.727 |      3840 |            0.975 |
| cube-single-play-v0 | fb_ref           |      1 | P1      |        nan     | nan     |              -0.073 |      0     |     97.291 |      1920 |            0.953 |
| cube-single-play-v0 | fb_ref           |      1 | P2      |          0.778 |   0.95  |               0.275 |      0.797 |    105.115 |      3840 |            0.953 |
| cube-single-play-v0 | fb_ref           |      1 | P3      |          0.5   |   0.667 |               0.007 |      0.766 |    100.208 |      3840 |            0.953 |
| cube-single-play-v0 | fb_ref           |      2 | P1      |        nan     | nan     |              -0.036 |      0     |     96.472 |      1920 |            0.977 |
| cube-single-play-v0 | fb_ref           |      2 | P2      |          0.75  |   0.909 |               0.221 |      0.797 |    104.791 |      3840 |            0.977 |
| cube-single-play-v0 | fb_ref           |      2 | P3      |          0.5   |   0.75  |              -0.096 |      0.766 |    102.358 |      3840 |            0.977 |

## Pre-registered readings

Every reading in the spec is called below, including those that do not fire and
the conjunct that blocked each.

### does not fire -- FLOOR (density_fb_state loses >0.15 H0->H1 on >=2 seeds)

- detail: seeds losing >0.15: []
- blocked by: fewer than 2 seeds lost >0.15 -- readings stand

### does not fire -- R-HELD1 (bottleneck is load-bearing for FB)

- detail: delta>=0.30 on seeds []; of those within 0.10 of Density-FB: []
- blocked by: only 0 seed(s) reach delta>=0.30

### FIRED -- R-HELD2 (deficit lives AFTER the bottleneck)

- detail: delta<=0.10 on seeds [0, 1, 2]; retention<0.6 on []; maintain+dropped>=50% on [0, 1, 2]; Density-FB stable on [0, 1, 2]

### does not fire -- R-HELD3 (mixed: FB gains but stays far below)

- detail: seeds in (0.10, 0.30) with a >0.10 gap: []
- blocked by: only 0 seed(s) qualify

### FIRED -- G4 (P2 states are decision points)

- detail: {"n_states": 64, "n_std_above": 44, "required": 40, "median_std": 0.274019718170166, "passes": true, "p1_median_std": 0.04056645929813385, "p3_median_std": 0.21636182069778442}

### FIRED -- P1 null control

- detail: degenerate (no securing action at P1, AUC undefined -> null holds by construction): {'fb_ref': [0, 1, 2], 'density_fb_state': [0, 1, 2]}; out-of-band where defined: {'fb_ref': [], 'density_fb_state': []}

### does not fire -- R-MEAS1 (misallocation at the bottleneck, shown)

- detail: FB auc<0.60 on [0]; Density-FB auc>=0.75 on []; FB auc_Q<0.60 on []
- blocked by: FB<0.60 on 1 seed(s), Density-FB>=0.75 on 0, FB auc_Q<0.60 on 0 (need >=2 each)

### does not fire -- R-MEAS2 (measure sees it, the z-readout loses it)

- detail: FB auc_heldmass>=0.75 on [1, 2]; FB auc_Q<0.60 on []
- blocked by: FB auc>=0.75 on 2 seed(s), auc_Q<0.60 on 0

### does not fire -- R-MEAS3 (both estimators resolve it)

- detail: FB>=0.75 on [1, 2]; Density-FB>=0.75 on []
- blocked by: at least one arm is below 0.75 on >=2 seeds

### FIRED -- R-MEAS1-3 all silent -- OBSERVED PATTERN (not enumerated by the spec)

- detail: P2 auc_heldmass: FB [(0, 0.55), (1, 0.7777777777777778), (2, 0.75)], Density-FB [(0, 0.4375), (1, 0.5454545454545454), (2, 0.6363636363636364)]; P2 auc_Q: FB [(0, 0.95), (1, 0.95), (2, 0.9090909090909091)], Density-FB [(0, 0.55), (1, 0.5), (2, 0.5)]. The spec's three readings all presuppose FB is the WEAKER discriminator at the bottleneck; here the ordering is reversed.

### does not fire -- R-MEAS4 (unconstrained-scale route is real)

- detail: FB row_mass_ratio(P3/P2) > 3 on seeds []
- blocked by: ratio>3 on 0 seed(s) (need >=2); NOT exclusive with R-MEAS1-3

