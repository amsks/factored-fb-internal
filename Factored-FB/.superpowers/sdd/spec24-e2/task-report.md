# Spec 24 E2 — task report

STATUS: DONE. Eval-only, no training/agent-code changes. New
`scripts/rescuing_fb/eval_density_z_rewards.py` (generalizes
`eval_density_zeroshot.py`, reusing its readout math verbatim: `density_weights`,
`compute_density_primary/secondary`, `build_pool`, `check_density_consistency`),
new `tests/test_eval_density_z_rewards.py` (18/18 green, `.venv/bin/python -m
pytest`). Ran all 3 z24 checkpoints x 2 arms (pool 16384, 25 episodes,
horizon 1000, seed 0, families dirac/dense/region) in tmux `spec24e2`; all 6
runs rc=0, ~2-5 min each. Output: `results/spec24_eval/{e2a,e2b}_z24_pmm_z_s{0,1,2}.json`.

## Family means (success_final), per seed

| seed | arm | dirac | dense | region |
|---|---|---|---|---|
| s0 | e2a | 0.000 | 0.008 | 0.010 |
| s1 | e2a | 0.008 | 0.000 | 0.030 |
| s2 | e2a | 0.008 | 0.008 | 0.030 |
| s0 | e2b | 0.000 | 0.000 | 0.010 |
| s1 | e2b | 0.008 | 0.008 | 0.010 |
| s2 | e2b | 0.000 | 0.000 | 0.010 |

Arm means (3-seed avg over each family, NOT averaged across arms): E2a ≈
{dirac 0.005, dense 0.005, region 0.023}; E2b ≈ {dirac 0.003, dense 0.003,
region 0.010}.

## G3 verdict

Per-seed E2a vs E2b differences are all ≤0.02 (within the ±0.05 band) ⇒
**E2a ≈ E2b holds** — the analytic vs. search index assignment does not
matter. But BOTH arms sit near zero (0.00–0.03), far BELOW the E1 floor
(0.200), not merely "≈" it as one of the pre-registered readings anticipated
— this is a stronger negative: reward-to-index assignment does not work
under z at all, regardless of how the index is built. Anchors for reference:
g-indexed density 0.315, GCIQL 0.242, BC-only 0.235, E1(z, direct-goal) 0.200.
No below-BC claim is made (no CI computed here; the task's BC/E1-CI caveat
covers E1 only, not E2).

## Addendum 2026-08-13: g-indexed control arm

Coordinator flagged that the .315/.242/.235 anchors are goal-conditioned
NATIVE ENV SUCCESS numbers — comparing E2's `region_success_from_obs` reads
against them is a protocol mismatch. Added `--arm control` to
`eval_density_z_rewards.py`: g-indexed retrieval decode
(`decode_goal_retrieval_density`, imported unchanged from
`eval_density_zeroshot.py`) on the SAME families/pool(16384)/episodes(25)/
horizon(1000)/metric, requiring `measure_density_z` absent/False
(`check_control_consistency`, the inverse of `check_density_z_consistency` —
neither arm accepts the other's checkpoint type, verified live, not just
unit-tested). Ran on `p0_pmm_dens_s0` (the only g-indexed pmm checkpoint),
seed 0, in tmux `spec24control`, rc=0. 5 new tests (23/23 total green).

### Three-way table (family means, success_final)

| arm | dirac | dense | region |
|---|---|---|---|
| E2a (3-seed avg) | 0.005 | 0.005 | 0.023 |
| E2b (3-seed avg) | 0.003 | 0.003 | 0.010 |
| control (s0)     | 0.000 | 0.000 | 0.000 |

### Revised G3 verdict

The control ALSO reads at exactly zero on all three families — even lower
than E2a/E2b's already-near-zero reads. Given only 25 episodes/cell, E2a/E2b
(≤0.03) are not distinguishable from the control's zero floor. **This means
E2 is uninformative about z: the near-zero reads are not evidence that z
broke the reward readout — the retrieval-decode + fixed-rollout readout
protocol reads near zero on point-mass-maze regardless of whether the index
is g or z.** The correct conclusion is about this readout on this
environment (family/pool/rollout design, or point-mass-maze's own
task difficulty under a decoded/fixed goal), not about the z-indexing change
spec 24 set out to test. This supersedes the "far below E1" reading in the
body above — that was interpretable only when the goal-conditioned E1 was
the sole reference point.

## Addendum 2026-08-13 (2nd): cube-double, where the metric discriminates

The pmm control read exactly zero, so pmm's metric couldn't answer G3.
Extended `eval_density_z_rewards.py` to cube-double (SUPPORTED_ENVS relaxed
by coordinator; I found and fixed the resulting bug: env/dataset loading was
still hardcoded to load point-mass-maze's dataset regardless of the
checkpoint's own `env_name`, so a cube-double checkpoint got built with
pmm-shaped example tensors, obs_dim=4 act_dim=2 instead of cube-double's
37/5 — the `ZeroDivisionError` in `create()`'s Dense-layer fan computation).
Fixed by dispatching env/dataset loading, per-task goal centers, episode
resets, and env pooling on `env_name` (`is_pmm`); readout algebra, z
construction, and both consistency checks needed no changes (already
domain-agnostic). Renamed `batched_z_rollout_pmm`/`batched_goal_rollout_pmm`
to `batched_z_rollout`/`batched_goal_rollout` (now take a `reset_fn`).
Ran identical protocol (pool 16384, 25 episodes, seed per z-checkpoint,
horizon = env default 500) on `z24_cd_z_s{0,1,2}` (e2a+e2b) and
`d18x_cd_dens_s0` (control), all in tmux `spec24cd`, rc=0 for all 7 jobs.
23/23 tests still green. Output: `results/spec24_eval/cd_{e2a,e2b,control}_s*.json`.

### Cube-double three-way table (family means, success_final / success_any)

| arm | dirac | dense | region |
|---|---|---|---|
| E2a (3-seed avg) | 0.019 / 0.019 | 0.024 / 0.032 | 0.000 / 0.007 |
| E2b (3-seed avg) | 0.016 / 0.021 | 0.029 / 0.029 | 0.000 / 0.013 |
| control (s0)     | 0.464 / 0.504 | 0.552 / 0.584 | 0.300 / 0.790 |

### G3 verdict (against the cube-double control — the decisive read)

The control reaches .30-.79 across families — the metric clearly
discriminates here, unlike pmm. Both z arms sit at 0.00-0.03 on every
family/metric, and E2a ≈ E2b throughout (differences ≤0.02) — the
analytic-vs-search index assignment doesn't matter, matching the pmm
pattern. **This is the clean answer the control arm was built to give: the
reward readout works fine under z's own estimator when g-indexed (control
.30-.79), and z-indexing destroys it almost completely (z arms ~0-.03).**
This corroborates the 2026-08-05 stack-divergence finding (z-conditioning
fails on multi-object manipulation) with a NEW instrument (reward
readout, not just goal-reaching), and rules out the "readout is broken
regardless of index" explanation that pmm's null left open.

## Concerns
- E2's family success (`region_success_from_obs` at each family's own
  center) is a stricter/different metric than E1's own goal-success, so
  "far below E1" is a magnitude read, not a strictly apples-to-apples one.
- E2b candidates drawn from a 4096-sample subset (not the full 16384 pool),
  documented as a deliberate memory-bounded deviation in the script/JSON meta.
- `q_r_primary`/`q_r_secondary` at t=0 are logged per cell in the JSONs but
  not summarized here — worth a follow-up read if diagnosing sign/scale.
