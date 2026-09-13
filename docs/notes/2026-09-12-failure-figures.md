# Failure figures: where FB fails on manipulation, and why (2026-09-12)

Figures for the Density-FB ICLR paper, produced from real OGBench rollouts of the tdjtab
1M checkpoints. Code: `scripts/probes/cube_bottleneck/failure_figs.py` (rollouts, banks,
predictions, action probe), `failure_plots.py` (figures + CSVs),
`failfig_pipeline.sh` (driver). Loader for the reward-inference route:
`scripts/probes/cube_bottleneck/agents.py::load_arm_eval / relabel_cond_agents`.
Intermediate data: `analysis/failure_figs/<env>/{rollouts,banks,predict,actions}/`.
Figures + the CSVs behind every panel: `PAPER/ICLR/figures/failure/`.

Methods: `fb_tdj` (our FB reimplementation, squared successor regression) and
`density_fb_z` (the deployed Density-FB, reward-sample task index). Envs:
cube-single-play-v0 and cube-double-play-v0. Checkpoints: tdjtab campaign, 1M steps,
seeds 0/1/2 (cube-single fb_tdj under `exp/`, everything else under `exp/tdjtab/`; the six
density_fb_z 1M checkpoints were pulled from `amsks/factored-fb-checkpoints`).

## Protocol

**Conditioning (both methods, every stage).** Exactly the eval_relabel route of
`impls/main.py`: for each of the env's 5 tasks, 10,000 random training transitions are
relabelled by their next state's simulator qpos against the task goal (reward shift 1.0,
threshold 0.04 m, `OGBenchAdapter.relabel_infer_batch`), the critic's `infer_cond` turns
those reward samples into a task vector z, and the agent is wrapped in `InferredCondAgent`
so the env goal is never seen (goals dropped, `witness()` checked: cond changed, non-zero
norm). One difference from the training-time eval: the 10k rows are drawn with a fixed
NumPy seed per task, so both methods and all seeds infer their z from the SAME rows
(the runs re-drew them at every eval point). For lockstep rollouts the flat z is tiled to
the batch through the `goals={"z": ...}` channel, which both critics' `map_eval_cond`
pass through untouched when no goal key is present.

**Figure A (where it fails).** 20 episodes per task, 5 tasks, 3 seeds, per (env, method);
policy rolled on its inferred z, episodes stepped in lockstep, deterministic rng per
(task, seed). Every episode gets exactly one outcome by the first unmet phase (RLBrew
phase thresholds, `phases.py`; success is the env's flag = every cube within 4 cm of its
target):

- approach failure: the hand never came within 6 cm of the cube;
- acquisition failure: reached, but the cube was never lifted > 3 cm with the gripper
  closed and the hand over it for 5 consecutive steps;
- maintain failure: acquired, then dropped -- final cube height < 3 cm and final cube
  more than 5 cm from its goal;
- transport failure: still held at the end away from the goal, or placed within 5 cm of
  the goal but outside the env's 4 cm tolerance;
- success.

cube-double: the ladder is evaluated per cube that has to move (goal > 2 cm from its
start; task 1 moves one cube, the others two); cubes that end within 4 cm of their goal
are done, and the episode outcome is the furthest phase reached on an undelivered cube,
i.e. the phase the policy failed at on the cube it was working on. Cube goals are read
from the mocap targets (cube-double permutes init/goal per reset).

**Banks (figure B input).** From the pooled 1M rollouts of BOTH methods (3 seeds each,
600 episodes per env), three banks of 64 recorded simulator states per env, drawn with one
state per episode where possible (seed 20260912):

- approach: before any acquisition in the episode, hand > 6 cm from the (nearest moving,
  undelivered) cube, cube on the table (lift < 1 cm), no contact;
- in-hand: cube lifted > 3 cm, gripper closed, contact > 0.5, hand over the cube, for 5
  consecutive steps, and before any carrying step of the episode;
- carrying: the same hold, held throughout the last 10 steps, and the cube-to-goal
  distance fell by > 2 cm over those 10 steps, cube not yet at the goal.

Each state is restored with the verified restore (`restore.py`: reset to the episode's
task, put back the episode's cube goal targets, `set_state(qpos, qvel)`, `mj_forward` +
`mj_rnePostConstraint`, then 2 settle steps commanding the gripper closed (+1) for held
states and open (-1) for approach states) and passes a fidelity gate: held banks need
contact > 0.5 at hand-over and, after one zero-action step, cube still lifted with the
gripper closed and cube drift < 4 cm; the approach bank needs cube drift < 4 cm, cube
still on the table, hand drift < 4 cm and gripper opening within 0.2 of the recorded
value. Candidates failing the gate are skipped (gate pass rates in `bank_stats.json`).

**Figure B (why it fails there): state-level prediction.** Pool U = 1024 dataset
next-states (seed 0), shared by both critics. For every bank state and every arm:
(i) the critic's predicted distribution over the pool under the task's inferred z, at the
RESTORED observation, averaged over 8 draws of the arm's own policy action: Density-FB
w = mean over heads of softmax_j F(s,a,z).B(u_j)/tau (the distribution its actor readout
integrates); FB M = min over heads of F(s,a,z).B(u_j), clipped at zero and normalized
(negative-mass fraction recorded); projected onto the cube's xy and binned on a 20x20 grid
over x in [0.25, 0.60], y in [-0.35, 0.35]; (ii) the rollout truth: K = 10 continuations
of THAT arm's own policy from the restored state, cube xy over the first 100 steps
weighted by 0.99^t (normalized), same grid; finished episodes hold their final state.
Error = total variation distance. Also recorded: mass within 5 cm of the goal for both,
and each distribution's TV distance to the pool prior. Per (env, bank, method) the TV is
averaged over seeds within a state, then mean +- s.e.m. over the 64 states.

**Figure B, action-conditioned probe (ranking panel).** At every bank state both methods
score the same 12 candidate actions: the FB policy's action and the Density-FB policy's
action (same seed, same rng), a secure action (arm held, gripper +1 = close), a release
action (gripper -1 = open), and 4 Gaussian perturbations (sigma 0.3, clipped to [-1, 1])
of each policy action. For each candidate and method: (i) the critic's own actor value
q(s,a,z) -- FB: F(s,a,z).z through `fb.q_fn`; Density-FB: sum_j w_j(s,a,z) r_z(u_j)
through `density_fb.q_fn` with its `latent_centered` reward rows built on the fixed pool,
exactly what its actor ascends; (ii) truth: execute the candidate for one step, then K = 10
own-policy continuations for 100 steps, giving the discounted cube occupancy on the grid
and the discounted goal-reaching return sum_t 0.99^t 1[cube within 4 cm of its goal].
Per (state, method): occupancy TV averaged over the 12 candidates; action-gap error
= |(q[best] - q[worst])/range(q) - (R[best] - R[worst])/range(R)| with best/worst by true
return; ranking accuracy = Spearman(q, R) over the 12 candidates; decision regret
= (R[best] - R[argmax q]) / range(R). States where all 12 true returns coincide (range 0,
typically FB at approach states whose continuations never reach the goal) give NaN for
the last three and are dropped from the means; counts are in the summary CSV.

**Figure C (prediction quality tracks success).** The carrying-bank TV error (state-level
prediction as above) for every available checkpoint: fb_tdj at 250k/500k/750k/1M x 3
seeds x 2 envs; density_fb_z at 1M x 3 seeds x 2 envs plus the one earlier density_fb_z
checkpoint that exists (cube-single seed 0 at 500k). Success at that checkpoint = the
run's own `eval_relabel.csv` success, mean over tasks 1-5 at that step (the runs evaluated
at multiples of 100k only, so the 250k/750k checkpoints take the mean of the two
neighbouring eval points; two cube-single FB runs logged their 1M eval twice, so the mean
is taken within a task before across tasks). One point per
(checkpoint, seed); single least-squares line through all points; Spearman reported
below, not on the figure.

### What was left out and why

- density_fb_z at 250k/750k (and 500k for five of the six runs) does not exist anywhere
  reachable: the HF repo holds only `params_1000000.pkl` for the tdjtab density_fb_z runs
  (plus cube-single seed 0 at 500k), the local `exp/cube-single-play-v0_density_fb_z_*`
  checkpoints are a different configuration (no reward-relabel eval), and `partb-tier1`
  is the old-reward family. Figure C therefore has 24 FB points and 7 Density-FB points.
- The palette validator of the dataviz skill needs node, which this box lacks; the
  colours are seaborn's colorblind palette (Okabe-Ito derived) with the stack ordered so
  no two adjacent segments are its weakest CVD pair, and white gaps between segments.

## Reproduction

```
# all from the seam worktree; GPU 2/3, JAX venv
export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline
PY=/dev/shm/.venv-jax/bin/python; S=scripts/probes/cube_bottleneck/failure_figs.py
# checkpoints: fb_tdj local (exp/, exp/tdjtab); density_fb_z 1M from HF into exp/tdjtab/<run>/
#   (see logs/failfig/hf_download.log for the exact file list)
for ENV in cube-single-play-v0 cube-double-play-v0; do
  $PY $S rollouts --env $ENV --episodes 20                      # figure A + trajectories
  $PY $S banks    --env $ENV                                    # 3 x 64 gated states
  $PY $S predict  --env $ENV --critics fb_tdj       --epoch 1000000
  $PY $S predict  --env $ENV --critics density_fb_z --epoch 1000000
  for E in 250000 500000 750000; do $PY $S predict --env $ENV --critics fb_tdj --epoch $E --banks carrying; done
  $PY $S predict  --env cube-single-play-v0 --critics density_fb_z --epoch 500000 --banks carrying
  for SEED in 0 1 2; do
    $PY $S actionprobe --env $ENV --seeds $SEED --banks approach inhand --tag _app
    $PY $S actionprobe --env $ENV --seeds $SEED --banks carrying        --tag _car
  done
done
$PY scripts/probes/cube_bottleneck/failure_plots.py            # figures + CSVs
# smoke (one seed, 2 episodes/task, 8-state banks, 4 states/bank, 2 states for the action probe):
$PY $S rollouts --env cube-single-play-v0 --seeds 0 --episodes 2 --tag _smoke
$PY $S banks --env cube-single-play-v0 --tag _smoke --n-bank 8
$PY $S predict --env cube-single-play-v0 --seeds 0 --tag _smoke --bank-file analysis/failure_figs/cube-single-play-v0/banks/banks_smoke.npz --max-states 4
$PY $S actionprobe --env cube-single-play-v0 --seeds 0 --tag _smoke --max-states 2
```
The full pipeline for one env is `bash scripts/probes/cube_bottleneck/failfig_pipeline.sh <env> <gpu>`
(rollouts, banks, then the prediction jobs in parallel); the action probes were launched
as 12 separate processes (env x seed x bank split). Logs: `logs/failfig/`.

Smoke results (cube-single seed 0): restore gate 24/24 states passed; every successful
episode satisfied the approach milestone and the 4 cm delivery check; TV errors in
(0, 1) for both methods; Density-FB already separated from FB at in-hand states
(0.41 vs 0.89).

## Numbers behind the panels

All from the CSVs in `PAPER/ICLR/figures/failure/` (generated by
`scripts/probes/cube_bottleneck/failure_note_tables.py`).

### Figure A: outcome fractions (mean over 3 seeds, 100 episodes per seed)

| env | method | approach | acquisition | maintain | transport | success |
|---|---|---|---|---|---|---|
| cube-double | Density-FB | 0.020 | 0.200 | 0.243 | 0.223 | 0.313 |
| cube-double | FB | 0.007 | 0.137 | 0.433 | 0.383 | 0.040 |
| cube-single | Density-FB | 0.003 | 0.187 | 0.007 | 0.003 | 0.800 |
| cube-single | FB | 0.000 | 0.107 | 0.203 | 0.223 | 0.467 |

Per seed:

| env | method | seed | approach | acquisition | maintain | transport | success | n_episodes |
|---|---|---|---|---|---|---|---|---|
| cube-double | Density-FB | 0 | 0.000 | 0.200 | 0.230 | 0.210 | 0.360 | 100 |
| cube-double | Density-FB | 1 | 0.020 | 0.270 | 0.230 | 0.210 | 0.270 | 100 |
| cube-double | Density-FB | 2 | 0.040 | 0.130 | 0.270 | 0.250 | 0.310 | 100 |
| cube-double | FB | 0 | 0.000 | 0.090 | 0.490 | 0.370 | 0.050 | 100 |
| cube-double | FB | 1 | 0.010 | 0.220 | 0.380 | 0.380 | 0.010 | 100 |
| cube-double | FB | 2 | 0.010 | 0.100 | 0.430 | 0.400 | 0.060 | 100 |
| cube-single | Density-FB | 0 | 0.000 | 0.260 | 0.010 | 0.010 | 0.720 | 100 |
| cube-single | Density-FB | 1 | 0.000 | 0.190 | 0.000 | 0.000 | 0.810 | 100 |
| cube-single | Density-FB | 2 | 0.010 | 0.110 | 0.010 | 0.000 | 0.870 | 100 |
| cube-single | FB | 0 | 0.000 | 0.140 | 0.270 | 0.330 | 0.260 | 100 |
| cube-single | FB | 1 | 0.000 | 0.090 | 0.200 | 0.160 | 0.550 | 100 |
| cube-single | FB | 2 | 0.000 | 0.090 | 0.140 | 0.180 | 0.590 | 100 |

### Figure B (top row): state-level prediction error, mean +- s.e.m. over 64 states

| env | state type | method | TV error | goal mass (pred) | goal mass (truth) | TV pred vs prior | TV truth vs prior |
|---|---|---|---|---|---|---|---|
| cube-double | approach | Density-FB | 0.830 +- 0.018 | 0.371 | 0.146 | 0.911 | 0.928 |
| cube-double | approach | FB | 0.931 +- 0.004 | 0.052 | 0.011 | 0.403 | 0.913 |
| cube-double | carrying | Density-FB | 0.802 +- 0.017 | 0.556 | 0.532 | 0.930 | 0.926 |
| cube-double | carrying | FB | 0.844 +- 0.005 | 0.054 | 0.182 | 0.407 | 0.815 |
| cube-double | in-hand | Density-FB | 0.804 +- 0.015 | 0.424 | 0.320 | 0.923 | 0.901 |
| cube-double | in-hand | FB | 0.832 +- 0.004 | 0.051 | 0.048 | 0.409 | 0.794 |
| cube-single | approach | Density-FB | 0.578 +- 0.015 | 0.658 | 0.426 | 0.858 | 0.888 |
| cube-single | approach | FB | 0.847 +- 0.006 | 0.120 | 0.209 | 0.464 | 0.869 |
| cube-single | carrying | Density-FB | 0.564 +- 0.022 | 0.905 | 0.943 | 0.929 | 0.964 |
| cube-single | carrying | FB | 0.893 +- 0.009 | 0.127 | 0.552 | 0.469 | 0.929 |
| cube-single | in-hand | Density-FB | 0.547 +- 0.013 | 0.736 | 0.752 | 0.856 | 0.881 |
| cube-single | in-hand | FB | 0.830 +- 0.005 | 0.128 | 0.263 | 0.475 | 0.853 |

### Figure B (bottom row) and the action-conditioned probe: mean +- s.e.m. over states (n = states with a non-degenerate true-return range)

| env | state type | method | occupancy TV (action-conditioned) | action-gap error | ranking accuracy (Spearman) | decision regret |
|---|---|---|---|---|---|---|
| cube-double | approach | Density-FB | 0.835 +- 0.017 (n=64) | 0.955 +- 0.047 (n=39) | 0.034 +- 0.048 (n=39) | 0.626 +- 0.047 (n=39) |
| cube-double | approach | FB | 0.937 +- 0.003 (n=64) | 1.093 +- 0.030 (n=56) | -0.069 +- 0.033 (n=56) | 0.956 +- 0.017 (n=56) |
| cube-double | carrying | Density-FB | 0.820 +- 0.013 (n=64) | 0.965 +- 0.051 (n=48) | 0.133 +- 0.042 (n=48) | 0.482 +- 0.042 (n=48) |
| cube-double | carrying | FB | 0.854 +- 0.004 (n=64) | 0.999 +- 0.031 (n=64) | 0.021 +- 0.028 (n=64) | 0.812 +- 0.029 (n=64) |
| cube-double | in-hand | Density-FB | 0.817 +- 0.013 (n=64) | 0.977 +- 0.040 (n=48) | 0.042 +- 0.040 (n=48) | 0.607 +- 0.044 (n=48) |
| cube-double | in-hand | FB | 0.838 +- 0.003 (n=64) | 0.990 +- 0.025 (n=64) | 0.034 +- 0.025 (n=64) | 0.786 +- 0.029 (n=64) |
| cube-single | approach | Density-FB | 0.596 +- 0.014 (n=64) | 1.057 +- 0.038 (n=64) | -0.005 +- 0.028 (n=64) | 0.546 +- 0.031 (n=64) |
| cube-single | approach | FB | 0.847 +- 0.005 (n=64) | 0.992 +- 0.032 (n=63) | 0.046 +- 0.030 (n=63) | 0.644 +- 0.029 (n=63) |
| cube-single | carrying | Density-FB | 0.591 +- 0.017 (n=64) | 0.661 +- 0.028 (n=64) | 0.478 +- 0.021 (n=64) | 0.235 +- 0.031 (n=64) |
| cube-single | carrying | FB | 0.895 +- 0.007 (n=64) | 1.051 +- 0.062 (n=64) | 0.059 +- 0.038 (n=64) | 0.518 +- 0.033 (n=64) |
| cube-single | in-hand | Density-FB | 0.572 +- 0.010 (n=64) | 0.854 +- 0.029 (n=64) | 0.197 +- 0.031 (n=64) | 0.326 +- 0.027 (n=64) |
| cube-single | in-hand | FB | 0.833 +- 0.005 (n=64) | 0.891 +- 0.037 (n=64) | 0.129 +- 0.034 (n=64) | 0.537 +- 0.028 (n=64) |

### Figure C: carrying-state TV error vs success per checkpoint

| env | method | seed | step | TV error (carrying) | success (eval_relabel) |
|---|---|---|---|---|---|
| cube-double | Density-FB | 0 | 1000000 | 0.817 | 0.460 |
| cube-double | Density-FB | 1 | 1000000 | 0.794 | 0.240 |
| cube-double | Density-FB | 2 | 1000000 | 0.795 | 0.360 |
| cube-double | FB | 0 | 250000 | 0.802 | 0.000 |
| cube-double | FB | 0 | 500000 | 0.820 | 0.000 |
| cube-double | FB | 0 | 750000 | 0.835 | 0.010 |
| cube-double | FB | 0 | 1000000 | 0.846 | 0.020 |
| cube-double | FB | 1 | 250000 | 0.844 | 0.000 |
| cube-double | FB | 1 | 500000 | 0.830 | 0.000 |
| cube-double | FB | 1 | 750000 | 0.846 | 0.050 |
| cube-double | FB | 1 | 1000000 | 0.824 | 0.000 |
| cube-double | FB | 2 | 250000 | 0.849 | 0.030 |
| cube-double | FB | 2 | 500000 | 0.845 | 0.020 |
| cube-double | FB | 2 | 750000 | 0.837 | 0.010 |
| cube-double | FB | 2 | 1000000 | 0.862 | 0.000 |
| cube-single | Density-FB | 0 | 500000 | 0.643 | 0.800 |
| cube-single | Density-FB | 0 | 1000000 | 0.554 | 0.680 |
| cube-single | Density-FB | 1 | 1000000 | 0.551 | 0.920 |
| cube-single | Density-FB | 2 | 1000000 | 0.587 | 0.860 |
| cube-single | FB | 0 | 250000 | 0.896 | 0.430 |
| cube-single | FB | 0 | 500000 | 0.897 | 0.440 |
| cube-single | FB | 0 | 750000 | 0.905 | 0.560 |
| cube-single | FB | 0 | 1000000 | 0.908 | 0.370 |
| cube-single | FB | 1 | 250000 | 0.896 | 0.130 |
| cube-single | FB | 1 | 500000 | 0.909 | 0.180 |
| cube-single | FB | 1 | 750000 | 0.878 | 0.380 |
| cube-single | FB | 1 | 1000000 | 0.888 | 0.360 |
| cube-single | FB | 2 | 250000 | 0.869 | 0.320 |
| cube-single | FB | 2 | 500000 | 0.887 | 0.200 |
| cube-single | FB | 2 | 750000 | 0.891 | 0.260 |
| cube-single | FB | 2 | 1000000 | 0.883 | 0.450 |

Fit and correlations: slope = -1.687, intercept = 1.663, spearman = -0.010, pearson = -0.610, n_points = 31, spearman_cube-single = -0.562, n_cube-single = 16, spearman_cube-double = -0.164, n_cube-double = 15, spearman_fb_only = 0.828


## Reading

**Where FB fails (figure A).** Both methods clear the approach phase almost always and lose a
similar share of episodes at acquisition (FB 0.11 / 0.14, Density-FB 0.19 / 0.20 on
cube-single / cube-double). Everything else separates after the grasp: on cube-single FB
loses 0.20 of its episodes by dropping the cube and 0.22 by holding it away from the goal
(Density-FB 0.01 / 0.00); on cube-double FB loses 0.43 + 0.38 there against Density-FB's
0.24 + 0.22. The failure is in the maintain and transport phases, i.e. at carrying states,
which is where the paper places it; at the grasp itself the two are level, consistent with
the 2026-09-04 bottleneck probe.

**Why it fails there (figure B, top).** FB's predicted future-cube-position distribution
stays close to the dataset's occupancy at every state type: its TV distance to the pool
prior is 0.46-0.48 (cube-single) and 0.40-0.41 (cube-double), against 0.86-0.93 for
Density-FB, while the true own-policy occupancy is 0.79-0.96 away from that prior for both
methods. So FB's error against its own rollouts is 0.83-0.93 everywhere, and its predicted
mass within 5 cm of the goal is flat at 0.12-0.13 (cube-single) / 0.05 (cube-double)
whatever the state, even at carrying states where its own policy actually puts 0.55 / 0.18
of the discounted future there. Density-FB's prediction moves with the state: goal mass
0.66 -> 0.74 -> 0.91 across approach / in-hand / carrying on cube-single against a truth of
0.43 / 0.75 / 0.94, and 0.37 / 0.42 / 0.56 against 0.15 / 0.32 / 0.53 on cube-double. On
cube-double the TV gap between the methods is small (0.03-0.10) because Density-FB's
prediction is also far from its own rollouts there in absolute terms; the goal-mass and
prior-distance columns carry the mechanism more clearly than TV does on that env.
Negative FB mass clipped before normalizing: 0.36-0.46 of |M| on average (column
`fb_neg_frac` in `pred_*.csv`).

**Action ranking (figure B, bottom; action probe).** With 12 shared candidates per state,
Density-FB ranks actions by their true discounted goal-reaching return far better than FB
at carrying states on cube-single (Spearman 0.48 vs 0.06; decision regret 0.24 vs 0.52),
moderately better at in-hand states (0.20 vs 0.13; regret 0.33 vs 0.54), and neither
critic ranks anything at approach states (both ~0; regret 0.55 vs 0.64). On cube-double the
ranking accuracies are low for both (Density-FB 0.03 / 0.04 / 0.13, FB -0.07 / 0.03 /
0.02) but the regret gap persists at every state type (0.63 / 0.61 / 0.48 vs 0.96 / 0.79 /
0.81). The action-gap error is close to 1 for both methods almost everywhere (the critic's
own scale does not separate the true best and worst candidates), except Density-FB at
cube-single carrying (0.66) and in-hand (0.85). The action-conditioned occupancy TV
reproduces the state-level column (Density-FB 0.57-0.60 / 0.82-0.84, FB 0.83-0.90 /
0.84-0.94). FB's own-policy candidates at approach states often have zero true return for
all 12 candidates on cube-double (only 56/64 states usable) and Density-FB's at 39/64 of
those states; those NaN rows are excluded, so the approach-state ranking numbers on
cube-double are over the states where at least one candidate reaches the goal.

**Prediction quality tracks success (figure C).** Across the 31 (checkpoint, seed) points
the least-squares line has slope -1.69 (success per unit TV) and Pearson -0.61, but the
pooled Spearman is -0.01: the correlation is carried by the separation between the two
methods (Density-FB points at TV 0.55-0.64 with success 0.68-0.92 on cube-single, 0.79-0.82
with 0.24-0.46 on cube-double; FB at 0.87-0.91 / 0.80-0.86 with 0.13-0.56 / 0.00-0.05),
not by a monotone relation inside a method or across envs. Within cube-single the Spearman
is -0.56 (16 points); within cube-double -0.16 (15 points); across FB's own checkpoints
alone it is +0.83, because cube-double FB has LOWER TV than cube-single FB while succeeding
less, and FB's TV does not fall over training on either env (0.87-0.91 from 250k to 1M on
cube-single). The honest statement for the paper is that the method with the calibrated
successor prediction is the one that succeeds, and that FB's prediction error at carrying
states never improves with training; not that success is a monotone function of TV
across every checkpoint.

## Caveats

- Rollout success here (20 episodes/task, fixed relabel rows) differs from the runs' own
  eval_relabel numbers at 1M (cube-single FB 0.47 mean here vs 0.38 logged; Density-FB
  0.80 vs 0.82; cube-double FB 0.04 vs 0.01, Density-FB 0.31 vs 0.35). The 10k relabel
  rows and their seed change the inferred z, and 20 vs 10 episodes per task change the
  estimate; the ranking is the same.
- Banks are drawn from the pooled rollouts of both methods (composition per bank in
  `bank_stats.json`, roughly 28-37 FB states of 64), and each method's truth is its OWN
  policy's continuation from that state, so a state one method never visits is still
  scored under the other's policy. This is the within-arm calibration comparison the
  2026-09-04 audit asked for; it is not a claim about a shared continuation policy.
- The truth is truncated at 100 steps (0.99^100 = 0.37 of the discounted mass is beyond
  the horizon) and estimated from K = 10 stochastic continuations, so it is itself noisy;
  the action-probe Spearman over 12 candidates is bounded by that noise, which is one
  reason both critics score near zero at approach states, where a single candidate action
  barely changes what the policy does over the next 100 steps.
- cube-double successful episodes occasionally satisfy the env's success flag without the
  "secured" milestone (a cube slid rather than lifted); the outcome ladder is only used for
  failed episodes, so this does not affect figure A.
- Figure C has 7 Density-FB points (6 at 1M, 1 at 500k) against 24 FB points because no
  earlier density_fb_z tdjtab checkpoints exist (see "left out").
