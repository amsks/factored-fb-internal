# Reward families on the frozen 1M cube checkpoints (evaluation only)

**Question.** Do the zero-shot models handle state rewards that are *not* goal-reaching?
Same frozen 1M checkpoints as the paper's OGBench table, same reward-inference route
(10k random training transitions, reward on the next state, +1.0 shift, `agent.infer_eval_cond`,
goal frames dropped), new rewards.

**Answer.** Yes for Density-FB on every family; the region and conjunction families separate
the methods sharply, the lift family does not. Density-FB is the only method that solves the
two-cube families (R6 move-one-keep-the-other 0.72 vs FB −0.20, HILP 0.02, RLDP −0.24; R7
either-cube 0.67 vs 0.04 / 0.18 / 0.06). On cube-single the density retrofits lift their hosts
(R1: RLDP 0.23 → Density-RLDP 0.42; HILP 0.48 → Density-HILP 0.65; FB 0.10 → Density-FB 0.61).
The preference flip (R3) is clean for every method that moves at all: time in A goes to ~0
and time in B to 0.3–0.6 when the weights swap. R4 (lift, no target) is the one family
where every method is close (0.48–0.58) and even BC scores 0.43, because 35 % of dataset
states are already lifted.

## Table of means (3 seeds, mean ± std; 20 episodes per seed, R5: 5 targets × 4 episodes)

| family / metric | FB | Density-FB | HILP | Density-HILP | RLDP | Density-RLDP | Random | BC |
|---|---|---|---|---|---|---|---|---|
| R1 region, mean r/step | 0.10 ± 0.06 | **0.61 ± 0.08** | 0.48 ± 0.04 | **0.65 ± 0.02** | 0.23 ± 0.04 | 0.42 ± 0.11 | 0.00 | 0.03 ± 0.03 |
| R2 either of two | 0.08 ± 0.07 | **0.53 ± 0.07** | 0.36 ± 0.08 | 0.39 ± 0.05 | 0.23 ± 0.03 | 0.36 ± 0.08 | 0.00 | 0.06 ± 0.01 |
| R3 r_a: time in A / B | .04 / .01 | .61 / .00 | .59 / .00 | .62 / .00 | .22 / .00 | .36 / .00 | .00 / .00 | .05 / .05 |
| R3 r_b: time in A / B | .03 / .04 | .00 / .60 | .00 / .29 | .00 / .49 | .03 / .17 | .00 / .36 | .00 / .00 | .02 / .03 |
| R4 lift | 0.48 ± 0.02 | 0.58 ± 0.03 | 0.55 ± 0.01 | 0.53 ± 0.02 | 0.54 ± 0.01 | 0.58 ± 0.01 | 0.00 | 0.43 ± 0.02 |
| R5 dense (−dist, m), dense z | −0.231 ± .003 | **−0.146 ± .014** | −0.174 ± .012 | −0.166 ± .002 | −0.208 ± .000 | −0.156 ± .004 | −0.271 | −0.251 ± .004 |
| R5 success %, dense z | 15 ± 3 | **79 ± 8** | 40 ± 6 | 75 ± 2 | 26 ± 8 | 57 ± 3 | 0 | 9 ± 1 |
| R5 dense (−dist, m), sparse z | −0.253 ± .003 | **−0.119 ± .013** | −0.168 ± .019 | −0.134 ± .004 | −0.241 ± .005 | −0.144 ± .006 | −0.271 | −0.253 ± .008 |
| R5 success %, sparse z | 12 ± 1 | **85 ± 4** | 61 ± 4 | 82 ± 4 | 20 ± 1 | 56 ± 13 | 0 | 11 ± 5 |
| R6 move one, keep other (cube-double) | −0.20 ± 0.02 | **0.72 ± 0.02** | 0.02 ± 0.05 | — | −0.24 ± 0.01 | — | −0.01 | −0.24 ± 0.02 |
| R7 either cube in A (cube-double) | 0.04 ± 0.03 | **0.67 ± 0.05** | 0.18 ± 0.12 | — | 0.06 ± 0.03 | — | 0.00 | 0.02 ± 0.02 |

Discounted returns (γ = 0.99) are in the CSV (`metric = discounted_return`); they rank the
methods identically (R1: Density-FB 40.1, Density-HILP 41.7, HILP 31.8, Density-RLDP 28.1,
RLDP 15.7, FB 6.2, BC 2.2, random 0.0; R6: Density-FB 45.6, HILP 2.7, random −0.04, FB −10.4,
RLDP −11.7, BC −10.3).

### What separates the methods

- **R1 / R2 / R3 (regions).** Every method reaches A at least once in most episodes
  (any-step in A: Density-HILP 100 %, HILP 97 %, Density-RLDP 97 %, RLDP 90 %, Density-FB 83 %,
  FB 45 %, BC 23 %); what differs is whether the policy *stays* (time in A: 0.61–0.65 for the
  Density-FB / Density-HILP / HILP group, 0.23 for RLDP, 0.10 for FB). Under R2 the policies
  pick one region and go there (time in B > time in A for every method, e.g. Density-FB
  0.39 vs 0.14, HILP 0.26 vs 0.10; the reason for the B preference was not investigated);
  nobody splits time between A and B.
- **R3 preference flip.** Time in A goes from 0.6 to 0.0 and time in B from 0.0 to 0.3–0.6
  for all four methods above the floor when r_a is swapped for r_b. The flip is complete;
  the less-preferred region (weight 0.3) is never visited. FB's index barely moves the cube
  either way (0.04 / 0.01 vs 0.03 / 0.04).
- **R4 lift (no target).** Every method has the cube above 3 cm for 53–63 % of steps (mean clipped
  height 5–6 cm); the BC floor is already 0.43. This family does not separate the methods
  (spread 0.48–0.58) and mostly confirms all six indices point at "lifted" states.
- **R5 dense vs sparse.** Under the dense −distance reward every method gets closer to the
  target than under random (−0.27) and BC (−0.25); Density-FB / Density-HILP / Density-RLDP are
  the only ones under −0.17 m. The dense-reward index is *worse* than the sparse index for the
  same target on every method (success 79 vs 85 % Density-FB, 40 vs 61 % HILP, 75 vs 82 %
  Density-HILP): the linear readout of a dense reward spreads z over the whole table.
- **R6 / R7 (two cubes).** Only Density-FB does the conjunction: cube 1 in A 78 % of steps
  while cube 2 stays in its start disc 87 % of steps. FB and RLDP *disturb* cube 2 (in B0
  only 57 % / 49 %, which is why they score below the do-nothing random policy at −0.01);
  HILP leaves cube 2 alone (97 %) but never moves cube 1 (3 %). R7: Density-FB puts a cube in
  A 67 % of steps (cube 1 56 %, cube 2 50 %), HILP 18 %, FB and RLDP ≤ 6 %.

## Exact definitions and constants

All rewards are functions of the simulator state only. Cube positions are read from qpos
(`object_joint_i` = 7 free-joint entries starting at column 14 + 7 i; cube i position = the
first 3). Table-plane coordinates are (x, y); a resting cube's centre is at z = 0.02
(`TABLE_Z`). Regions are open discs of radius **0.06 m** in (x, y). Every disc lies inside the
dataset's cube-position support (x ∈ [0.30, 0.55], y ∈ [−0.30, 0.30]; each disc holds
6.6–7.5 % of dataset rows) and ≥ 0.06 m from every eval reset position; none coincides with
an eval-task goal.

| env | region | centre (x, y) m |
|---|---|---|
| cube-single | A | (0.48, 0.23) |
| cube-single | B | (0.48, −0.23) |
| cube-double | A | (0.50, 0.23) |
| cube-double | B0 | (0.50, −0.10) — cube 2's start under the task-2 reset with the cube order fixed |

| family | reward (unshifted) | reset |
|---|---|---|
| R1 | r = 1[cube ∈ A] | env default (random task, ±1 cm jitter) |
| R2 | r = max(1[A], 1[B]) | env default |
| R3 | r_a = 1[A] + 0.3·1[B]; r_b = 0.3·1[A] + 1[B] | env default |
| R4 | r = clip((z_cube − 0.02) / 0.10, 0, 1) | env default |
| R5 | r = −‖cube − g‖ (3-D), g = task t's `goal_xyzs`; sparse: r = 1[‖cube − g‖ ≤ 0.04] | task_id = t, t = 1..5, 4 episodes each |
| R6 | r = 1[cube1 ∈ A] − 0.5·1[cube2 ∉ B0] | task_id = 2, `_permute_blocks = False` (cube 1 starts at (0.35, −0.10), cube 2 at (0.50, −0.10)) |
| R7 | r = max(1[cube1 ∈ A], 1[cube2 ∈ A]) | env default (random task, random cube order) |

Labels for inference: reward on the NEXT state of each of the 10,000 rows + 1.0 (the table
protocol's `eval_relabel_shift`), through each method's own `infer_cond`: FB / RLDP the
reward-weighted mean of B; HILP a least-squares solve on φ; Density-FB, Density-HILP and
Density-RLDP their training index (the regression solve). The 10k rows are the same for every
method (NumPy re-seeded to 20260914 before the draw). All 330 inferences were valid
(cond changed, finite, non-zero norm; `route_b_valid = 1` on every row of the CSV); the cosine
to the neutral 1_d direction is |cos| ≤ 0.38 everywhere, so no index fell back to the
constant-reward direction.

Rollouts: 20 episodes in lockstep (20 env copies, one batched actor call per step),
episode length 200 (cube-single) / 500 (cube-double), **`_terminate_at_goal = False`** so the
episode is never cut short by the drawn task's own success test, actor rolled at the eval
temperature with its own noise, rewards scored on the state after each step (matching the
next-state labels). Reset seeds are the same for every method (env i reseeded to
20260914 + 1000·family + i). Metrics: mean per-step reward, undiscounted and discounted
(γ = 0.99) return, fraction of steps in each region, any-step success (R5).

Floors: **Random** = uniform actions in [−1, 1]^5 (one run of 20 episodes per family).
**BC** = the flow vector field `actor_vf` of the fb_tdj checkpoints, integrated from Gaussian
noise for 10 steps (`FlowBCActor._rollout`). It is trained by the flow-matching BC loss only;
the value term of the FlowBC objective touches the one-step head `actor`, never `actor_vf`
(impls/actors/flowbc.py::loss), so it is the dataset's behaviour-cloning policy with the value
term off, read from the existing checkpoints (one per seed).

Checkpoints (all local, `params_1000000.pkl`, params-only restore):

| method | critic | run dirs |
|---|---|---|
| FB | fb_tdj | exp/cube-single-play-v0_fb_tdj_flowbc_seed_{0,1,2}; exp/tdjtab/cube-double-play-v0_fb_tdj_flowbc_seed_{0,1,2} |
| Density-FB | density_fb_z | exp/tdjtab/{cube-single,cube-double}-play-v0_density_fb_z_flowbc_seed_{0,1,2} |
| HILP | hilp_tdjc | exp/{cube-single,cube-double}-play-v0_hilp_tdjc_flowbc_seed_{0,1,2} |
| Density-HILP | hilp_density_reg | exp/hlevers/cube-single-play-v0_hilp_density_reg_flowbc_seed_{0,1,2} (all three 1M checkpoints were present; cube-single only) |
| RLDP | rldp, **ortho 1** | exp/ortho1/{cube-single,cube-double}-play-v0_rldp_flowbc_seed_{0,1,2} |
| Density-RLDP | rldp_density_reg, ortho 1 | exp/levers/cube-single-play-v0_rldp_density_reg_flowbc_seed_{0,1,2} (cube-single only; no cube-double run exists) |

## Caveats

1. The retrofits (Density-HILP, Density-RLDP) have no cube-double run, so R6 / R7 have four
   methods. Their cube-single cells use the regression-index variant at ortho 1 as requested.
2. R6's reset is fixed (task 2, cube order fixed) because "cube 2 outside its start region"
   needs a start region that is a fixed disc; in the dataset the cubes start uniformly on the
   table, so there is no "typical" dataset start position, and the eval reset is used instead.
   R5's reset is per target (task_id = t) for the same reason. Every other family uses the env's
   own reset (random task, random cube permutation), which is not the table protocol's
   one-task-per-row reset but is the same for every method.
3. Episodes are not cut at the drawn task's success, unlike the table protocol; this only
   lengthens episodes and is applied uniformly.
4. The +1.0 shift is applied to every label including the dense −distance one (label = 1 −
   dist); FB's and RLDP's reward-weighted-mean indices depend on this constant, the solves do
   not. The shift is the table protocol's and was not tuned.
5. Region and lift constants were chosen once from the dataset support before any rollout and
   not tuned. Radius 0.06 is larger than the env's 0.04 success tolerance.
6. Random floor: 1 run (no std). The BC floor uses fb_tdj's actor_vf; other checkpoints'
   actor_vf would be different draws of the same BC objective.
7. R4 is dominated by the data: 35 % of cube-single dataset states have the cube above 5 cm,
   so BC already scores 0.43; the family confirms the indices point at lifted states but does
   not separate methods.
8. Evaluation only: nothing was trained and no checkpoint was written.

## Reproduction

```
# inside tmux `rewardfam`, from the repo root; ~25 min on 8 workers over GPUs 0/1
bash scripts/reward_families/launch_all.sh           # 38 jobs -> analysis/reward_families/<env>/<method>_seed<k>.json
/dev/shm/.venv-jax/bin/python scripts/reward_families/aggregate.py
#   -> analysis/reward_families/reward_families.csv (+ copy in PAPER/ICLR/figures/reward_families/)
#   -> PAPER/ICLR/figures/reward_families/fig_reward_families.{pdf,png}
#   -> PAPER/ICLR/baselines/reward_families.tex (standalone: tectonic reward_families_standalone.tex)
#   -> analysis/reward_families/per_seed_tables.md
# single job / smoke:
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline \
  /dev/shm/.venv-jax/bin/python scripts/reward_families/eval_families.py \
  --env cube-single-play-v0 --method density_fb_z --seed 0 --episodes 2 --families R1_region --smoke
```

Smoke (density_fb_z seed 0, R1, 2 episodes): z valid (norm 7.07, cos-to-neutral −0.06),
mean reward 0.67, finite; log logs/rewardfam/smoke_dfbz_s0.log. Campaign logs:
logs/rewardfam/<env>_<method>_seed<k>.log, exit codes in logs/rewardfam/launch_all.status
(38/38 exit 0).

## Per-seed values

(generated by aggregate.py; also in analysis/reward_families/per_seed_tables.md)


**R1 region -- `R1_region` / `mean_reward`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.183 | 0.050 | 0.070 | 0.101 | 0.058 |
| Density-FB | 0.515 | 0.612 | 0.714 | 0.614 | 0.081 |
| HILP | 0.510 | 0.499 | 0.419 | 0.476 | 0.041 |
| Density-HILP | 0.631 | 0.630 | 0.679 | 0.647 | 0.023 |
| RLDP | 0.281 | 0.227 | 0.182 | 0.230 | 0.040 |
| Density-RLDP | 0.275 | 0.423 | 0.547 | 0.415 | 0.111 |
| Random | 0.000 |  |  | 0.000 | nan |
| BC | 0.071 | 0.011 | 0.019 | 0.034 | 0.027 |

**R2 either of two -- `R2_either` / `mean_reward`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.176 | 0.039 | 0.034 | 0.083 | 0.066 |
| Density-FB | 0.500 | 0.472 | 0.628 | 0.534 | 0.068 |
| HILP | 0.256 | 0.395 | 0.440 | 0.363 | 0.078 |
| Density-HILP | 0.356 | 0.354 | 0.459 | 0.390 | 0.049 |
| RLDP | 0.197 | 0.258 | 0.231 | 0.229 | 0.025 |
| Density-RLDP | 0.261 | 0.357 | 0.455 | 0.358 | 0.079 |
| Random | 0.000 |  |  | 0.000 | nan |
| BC | 0.066 | 0.059 | 0.041 | 0.055 | 0.011 |

**R3 pref.\ $r_a$: time in A / B -- `R3_pref_a` / `in_A_mean`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.067 | 0.016 | 0.034 | 0.039 | 0.021 |
| Density-FB | 0.544 | 0.578 | 0.716 | 0.613 | 0.075 |
| HILP | 0.585 | 0.605 | 0.589 | 0.593 | 0.009 |
| Density-HILP | 0.569 | 0.725 | 0.564 | 0.619 | 0.075 |
| RLDP | 0.230 | 0.255 | 0.176 | 0.221 | 0.033 |
| Density-RLDP | 0.310 | 0.324 | 0.461 | 0.365 | 0.068 |
| Random | 0.000 |  |  | 0.000 | nan |
| BC | 0.056 | 0.048 | 0.035 | 0.046 | 0.009 |

**R3 pref.\ $r_a$: time in A / B -- `R3_pref_a` / `in_B_mean`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.007 | 0.005 | 0.004 | 0.005 | 0.001 |
| Density-FB | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| HILP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Density-HILP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| RLDP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Density-RLDP | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| Random | 0.000 |  |  | 0.000 | nan |
| BC | 0.111 | 0.017 | 0.018 | 0.048 | 0.044 |

**R3 pref.\ $r_b$: time in A / B -- `R3_pref_b` / `in_A_mean`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.067 | 0.018 | 0.020 | 0.035 | 0.023 |
| Density-FB | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| HILP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Density-HILP | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| RLDP | 0.000 | 0.025 | 0.056 | 0.027 | 0.023 |
| Density-RLDP | 0.002 | 0.000 | 0.000 | 0.001 | 0.001 |
| Random | 0.000 |  |  | 0.000 | nan |
| BC | 0.008 | 0.009 | 0.040 | 0.019 | 0.015 |

**R3 pref.\ $r_b$: time in A / B -- `R3_pref_b` / `in_B_mean`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.017 | 0.052 | 0.050 | 0.040 | 0.016 |
| Density-FB | 0.587 | 0.582 | 0.617 | 0.595 | 0.016 |
| HILP | 0.068 | 0.327 | 0.467 | 0.287 | 0.165 |
| Density-HILP | 0.430 | 0.552 | 0.484 | 0.489 | 0.050 |
| RLDP | 0.175 | 0.176 | 0.159 | 0.170 | 0.008 |
| Density-RLDP | 0.319 | 0.304 | 0.453 | 0.359 | 0.067 |
| Random | 0.000 |  |  | 0.000 | nan |
| BC | 0.064 | 0.000 | 0.012 | 0.025 | 0.028 |

**R4 lift -- `R4_lift` / `mean_reward`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.462 | 0.482 | 0.506 | 0.483 | 0.018 |
| Density-FB | 0.540 | 0.616 | 0.593 | 0.583 | 0.032 |
| HILP | 0.540 | 0.544 | 0.554 | 0.546 | 0.006 |
| Density-HILP | 0.504 | 0.535 | 0.545 | 0.528 | 0.018 |
| RLDP | 0.525 | 0.556 | 0.539 | 0.540 | 0.013 |
| Density-RLDP | 0.583 | 0.580 | 0.569 | 0.577 | 0.006 |
| Random | 0.001 |  |  | 0.001 | nan |
| BC | 0.409 | 0.449 | 0.430 | 0.429 | 0.017 |

**R5 dense ($-$dist, m), dense $z$ -- `R5_dense` / `mean_reward`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | -0.2282 | -0.2358 | -0.2291 | -0.2310 | 0.0034 |
| Density-FB | -0.1610 | -0.1498 | -0.1268 | -0.1459 | 0.0142 |
| HILP | -0.1673 | -0.1911 | -0.1651 | -0.1745 | 0.0117 |
| Density-HILP | -0.1647 | -0.1638 | -0.1690 | -0.1658 | 0.0022 |
| RLDP | -0.2074 | -0.2074 | -0.2082 | -0.2076 | 0.0004 |
| Density-RLDP | -0.1515 | -0.1606 | -0.1556 | -0.1559 | 0.0037 |
| Random | -0.2708 |  |  | -0.2708 | nan |
| BC | -0.2531 | -0.2452 | -0.2545 | -0.2509 | 0.0041 |

**R5 success (\%), dense $z$ -- `R5_dense` / `hit_any`** (cube-single, values x100)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 17.0 | 16.0 | 11.0 | 14.7 | 2.6 |
| Density-FB | 74.0 | 72.0 | 90.0 | 78.7 | 8.1 |
| HILP | 41.0 | 33.0 | 47.0 | 40.3 | 5.7 |
| Density-HILP | 72.0 | 75.0 | 77.0 | 74.7 | 2.1 |
| RLDP | 29.0 | 34.0 | 15.0 | 26.0 | 8.0 |
| Density-RLDP | 61.0 | 54.0 | 55.0 | 56.7 | 3.1 |
| Random | 0.0 |  |  | 0.0 | nan |
| BC | 8.0 | 10.0 | 8.0 | 8.7 | 0.9 |

**R5 dense ($-$dist, m), sparse $z$ -- `R5_sparse` / `dense_mean`** (cube-single, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | -0.2566 | -0.2493 | -0.2543 | -0.2534 | 0.0030 |
| Density-FB | -0.1350 | -0.1182 | -0.1024 | -0.1185 | 0.0133 |
| HILP | -0.1419 | -0.1837 | -0.1785 | -0.1680 | 0.0186 |
| Density-HILP | -0.1347 | -0.1291 | -0.1390 | -0.1343 | 0.0040 |
| RLDP | -0.2471 | -0.2346 | -0.2422 | -0.2413 | 0.0051 |
| Density-RLDP | -0.1435 | -0.1515 | -0.1364 | -0.1438 | 0.0062 |
| Random | -0.2714 |  |  | -0.2714 | nan |
| BC | -0.2504 | -0.2450 | -0.2642 | -0.2532 | 0.0081 |

**R5 success (\%), sparse $z$ -- `R5_sparse` / `hit_any`** (cube-single, values x100)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 12.0 | 13.0 | 11.0 | 12.0 | 0.8 |
| Density-FB | 80.0 | 88.0 | 88.0 | 85.3 | 3.8 |
| HILP | 62.0 | 55.0 | 65.0 | 60.7 | 4.2 |
| Density-HILP | 77.0 | 84.0 | 86.0 | 82.3 | 3.9 |
| RLDP | 21.0 | 21.0 | 18.0 | 20.0 | 1.4 |
| Density-RLDP | 50.0 | 44.0 | 75.0 | 56.3 | 13.4 |
| Random | 0.0 |  |  | 0.0 | nan |
| BC | 9.0 | 17.0 | 6.0 | 10.7 | 4.6 |

**R6 move one, keep other -- `R6_move_keep` / `mean_reward`** (cube-double, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | -0.182 | -0.183 | -0.228 | -0.198 | 0.022 |
| Density-FB | 0.698 | 0.737 | 0.714 | 0.716 | 0.016 |
| HILP | -0.014 | 0.087 | -0.016 | 0.019 | 0.048 |
| RLDP | -0.228 | -0.240 | -0.240 | -0.236 | 0.006 |
| Random | -0.006 |  |  | -0.006 | nan |
| BC | -0.248 | -0.262 | -0.219 | -0.243 | 0.018 |

**R7 either cube in A -- `R7_either_cube` / `mean_reward`** (cube-double, values x1)

| method | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| FB | 0.058 | 0.000 | 0.064 | 0.041 | 0.029 |
| Density-FB | 0.704 | 0.608 | 0.709 | 0.673 | 0.046 |
| HILP | 0.047 | 0.347 | 0.149 | 0.181 | 0.125 |
| RLDP | 0.080 | 0.022 | 0.075 | 0.059 | 0.027 |
| Random | 0.000 |  |  | 0.000 | nan |
| BC | 0.004 | 0.038 | 0.007 | 0.016 | 0.015 |

**Reward-inference witnesses (route_b_valid, ||z||, cos to the neutral 1_d, label mean / positive fraction), seed 0**

| env | family | method | valid | norm | cos1 | label mean | pos frac |
|---|---|---|---|---|---|---|---|
| cube-double | R6_move_keep | Density-FB | 1 | 7.07 | 0.09 | 0.604 | 0.070 |
| cube-double | R7_either_cube | Density-FB | 1 | 7.07 | 0.13 | 1.127 | 0.127 |
| cube-double | R6_move_keep | FB | 1 | 7.07 | 0.11 | 0.604 | 0.070 |
| cube-double | R7_either_cube | FB | 1 | 7.07 | 0.12 | 1.127 | 0.127 |
| cube-double | R6_move_keep | HILP | 1 | 7.07 | 0.04 | 0.604 | 0.070 |
| cube-double | R7_either_cube | HILP | 1 | 7.07 | 0.09 | 1.127 | 0.127 |
| cube-double | R6_move_keep | RLDP | 1 | 7.07 | 0.09 | 0.604 | 0.070 |
| cube-double | R7_either_cube | RLDP | 1 | 7.07 | 0.09 | 1.127 | 0.127 |
| cube-single | R1_region | Density-FB | 1 | 7.07 | -0.06 | 1.067 | 0.067 |
| cube-single | R2_either | Density-FB | 1 | 7.07 | 0.11 | 1.140 | 0.140 |
| cube-single | R3_pref_a | Density-FB | 1 | 7.07 | 0.00 | 1.089 | 0.140 |
| cube-single | R3_pref_b | Density-FB | 1 | 7.07 | 0.19 | 1.094 | 0.140 |
| cube-single | R4_lift | Density-FB | 1 | 7.07 | -0.06 | 1.317 | 0.445 |
| cube-single | R5_dense_t1 | Density-FB | 1 | 7.07 | -0.09 | 0.792 | 0.000 |
| cube-single | R5_sparse_t1 | Density-FB | 1 | 7.07 | -0.03 | 1.022 | 0.022 |
| cube-single | R1_region | FB | 1 | 7.07 | 0.17 | 1.067 | 0.067 |
| cube-single | R2_either | FB | 1 | 7.07 | 0.20 | 1.140 | 0.140 |
| cube-single | R3_pref_a | FB | 1 | 7.07 | 0.18 | 1.089 | 0.140 |
| cube-single | R3_pref_b | FB | 1 | 7.07 | 0.20 | 1.094 | 0.140 |
| cube-single | R4_lift | FB | 1 | 7.07 | 0.15 | 1.317 | 0.445 |
| cube-single | R5_dense_t1 | FB | 1 | 7.07 | 0.17 | 0.792 | 0.000 |
| cube-single | R5_sparse_t1 | FB | 1 | 7.07 | 0.16 | 1.022 | 0.022 |
| cube-single | R1_region | Density-HILP | 1 | 7.07 | -0.09 | 1.067 | 0.067 |
| cube-single | R2_either | Density-HILP | 1 | 7.07 | -0.24 | 1.140 | 0.140 |
| cube-single | R3_pref_a | Density-HILP | 1 | 7.07 | -0.16 | 1.089 | 0.140 |
| cube-single | R3_pref_b | Density-HILP | 1 | 7.07 | -0.26 | 1.094 | 0.140 |
| cube-single | R4_lift | Density-HILP | 1 | 7.07 | 0.05 | 1.317 | 0.445 |
| cube-single | R5_dense_t1 | Density-HILP | 1 | 7.07 | 0.08 | 0.792 | 0.000 |
| cube-single | R5_sparse_t1 | Density-HILP | 1 | 7.07 | 0.17 | 1.022 | 0.022 |
| cube-single | R1_region | HILP | 1 | 7.07 | -0.31 | 1.067 | 0.067 |
| cube-single | R2_either | HILP | 1 | 7.07 | -0.18 | 1.140 | 0.140 |
| cube-single | R3_pref_a | HILP | 1 | 7.07 | -0.28 | 1.089 | 0.140 |
| cube-single | R3_pref_b | HILP | 1 | 7.07 | -0.11 | 1.094 | 0.140 |
| cube-single | R4_lift | HILP | 1 | 7.07 | -0.02 | 1.317 | 0.445 |
| cube-single | R5_dense_t1 | HILP | 1 | 7.07 | -0.00 | 0.792 | 0.000 |
| cube-single | R5_sparse_t1 | HILP | 1 | 7.07 | -0.24 | 1.022 | 0.022 |
| cube-single | R1_region | Density-RLDP | 1 | 7.07 | 0.20 | 1.067 | 0.067 |
| cube-single | R2_either | Density-RLDP | 1 | 7.07 | 0.21 | 1.140 | 0.140 |
| cube-single | R3_pref_a | Density-RLDP | 1 | 7.07 | 0.23 | 1.089 | 0.140 |
| cube-single | R3_pref_b | Density-RLDP | 1 | 7.07 | 0.12 | 1.094 | 0.140 |
| cube-single | R4_lift | Density-RLDP | 1 | 7.07 | 0.05 | 1.317 | 0.445 |
| cube-single | R5_dense_t1 | Density-RLDP | 1 | 7.07 | -0.22 | 0.792 | 0.000 |
| cube-single | R5_sparse_t1 | Density-RLDP | 1 | 7.07 | -0.25 | 1.022 | 0.022 |
| cube-single | R1_region | RLDP | 1 | 7.07 | 0.07 | 1.067 | 0.067 |
| cube-single | R2_either | RLDP | 1 | 7.07 | 0.11 | 1.140 | 0.140 |
| cube-single | R3_pref_a | RLDP | 1 | 7.07 | 0.09 | 1.089 | 0.140 |
| cube-single | R3_pref_b | RLDP | 1 | 7.07 | 0.05 | 1.094 | 0.140 |
| cube-single | R4_lift | RLDP | 1 | 7.07 | 0.05 | 1.317 | 0.445 |
| cube-single | R5_dense_t1 | RLDP | 1 | 7.07 | -0.16 | 0.792 | 0.000 |
| cube-single | R5_sparse_t1 | RLDP | 1 | 7.07 | -0.10 | 1.022 | 0.022 |
