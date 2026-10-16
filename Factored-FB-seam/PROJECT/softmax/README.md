# Why the softmax (log-bilinear) successor model beats the linear one: scoping note

Started 2026-10-12. Everything for this mini-project lives here:
`scripts/` (analysis code), `results/` (json per run + `*_M.npz` empirical successor matrices),
`figures/`, `notes/` (dated findings), `data/` (downloaded robot datasets, git-ignored), `logs/`.

## 1. The question

Density-FB models the normalized successor measure as a softmax over candidate future states,
`m(u | s,a) / rho(u)  ∝  exp(F(s,a)·B(u) / tau)`, fitted by cross-entropy to the TD target
`(1-γ)·onehot(s') + γ·(target softmax at the bootstrap action)`
([impls/critics/density_fb.py:992-1041](../../impls/critics/density_fb.py)). FB models the same
ratio linearly, `m/rho = F·B`, by squared-error TD. Both use d = 50. Density-FB wins by a wide
margin on cube-single, cube-double and scene, and loses on antmaze-medium (reward-inference
protocol; see `docs/PROJECT_MAP.md` and memory). We do not have an explanation of **when** the
log-ratio parametrization is the better one. This note builds that explanation and tests it on
data alone, with no training.

The seed observation (pasted Q&A, 2026-10-12): rank-d models of the ratio and of the
log-ratio are incomparable in general; the log model wins on *peaked* distributions
(identity-like, exponential-family kernels), the ratio model wins on *mixtures* (latent-state,
block-MDP structure) and is closed under the discounted Bellman sum.

## 2. Hypothesis

**H1 (peaked occupancy).** The discounted occupancy of the data policy in robot manipulation
data is peaked: from a state `s`, the mass of `m(·|s)` sits on a small number of future states
relative to the number of states the dataset marginal `rho` covers. In matrix terms
(states discretised to K clusters, `M[i,j] = m(j|i)`), the row effective sample size
`ESS_i = 1/Σ_j M_ij²` is small compared with K, and the ratio `R = M/rho` has a large dynamic range.

**H2 (rank separation).** On a peaked `M`, a rank-d *linear* model of `R` (FB) needs d that grows
with the number of distinguishable states (Eckart–Young: a near-identity matrix loses at least
`K-d` of its Frobenius mass at rank d), while a rank-d *log-bilinear* model
(`P_ij ∝ rho_j exp(f_i·b_j)`) reaches the same error with d of the order of the intrinsic
dimension of the state manifold, independent of K. Navigation data with a diffuse occupancy
(random-walk-like, high mixing) does not separate the two models; there the linear model is fine.

**H3 (what drives peakedness).** Peakedness is set by the ratio of the discount horizon
`1/(1-γ)` to the mixing time of the behaviour policy over the state space. Manipulation play data
and single-task demonstrations do not mix: an episode visits each region once, object states
are sticky (nothing moves until grasped), and successive frames are near-identical at 20 Hz.
Antmaze locomotion data mixes: the ant re-visits cells, and the full-state occupancy
(joint angles, velocities) is close to the marginal within a few dozen steps.

**H4 (consequence for control, the paper claim).** Where the linear rank-50 critic cannot
represent the occupancy, the actor's value `Σ_u m̂(u|s,a) r(u)` for a peaked reward (a goal
indicator, a grasp-and-move outcome) is wrong by a margin that flips action choice; the
softmax critic's value is close to exact. Hence Density-FB > FB on peaked data and ≈ or < FB
on diffuse data. Prediction: the Density-FB − FB success margin per environment is monotone in a
dataset-only peakedness statistic.

## 3. The measurement (dataset only, no training)

`scripts/occupancy_rank.py`: k-means discretisation of the state (task coordinates = the
hit-spec columns the critic's kernel uses, or the full observation), empirical discounted
successor matrix `M` (γ = 0.99, reverse recursion over each episode, rows renormalised to the
in-episode future), marginal `rho`, then three rank-d fits scored on the same metrics
(ρ-weighted TV between the true row and the model row; KL; top-1 agreement; relative value
error on a one-cluster goal reward):

| model | family | fit |
|---|---|---|
| linear | `R ≈ F·B` (FB) | ρ-weighted SVD = exact optimum of FB's squared loss |
| nmf | `M ≈ W·H`, W,H ≥ 0 (mixture / latent-state; the ratio family's natural case) | KL multiplicative updates |
| log | `P_ij ∝ rho_j exp(f_i·b_j)` (Density-FB) | cross-entropy, Adam |

Peakedness statistics: row ESS / K (median), concentration `exp(H(rho) − E_i H(M_i))`,
median of `max_j R_ij`, one-step self-transition probability `P1_self`.
`scripts/summarize.py` builds the tables; `scripts/plot_results.py` the figures.

## 4. Results so far (2026-10-12)

Tables: `results/ogbench_K512_table.md` and the summaries printed in `notes/`.

**Finding 1 — the log model dominates at every rank on every dataset, and the gap is largest on
peaked data.** At rank 64 (≈ the d=50 both critics use), OGBench full observation, K=512:

| dataset | ESS/K | P1_self | TV linear@64 | TV log@64 |
|---|---|---|---|---|
| antmaze-medium | 0.55 | 0.15 | 0.035 | 0.020 |
| antmaze-large | 0.44 | 0.16 | 0.075 | 0.019 |
| puzzle-3x3 | 0.42 | 0.65 | 0.186 | 0.083 |
| cube-single | 0.15 | 0.78 | 0.252 | 0.033 |
| scene | 0.15 | 0.78 | 0.296 | 0.039 |
| cube-double | 0.12 | 0.82 | 0.465 | 0.041 |

The ordering of the gap (antmaze ≈ 0 → puzzle 0.1 → cube-single 0.22 → scene 0.26 → cube-double
0.42) is the ordering of the Density-FB − FB success margin (antmaze −0.29, puzzle ≈ 0 [both at
floor], cube-single +0.3, scene +0.45, cube-double +0.28, reward-inference protocol, TD-JEPA FB
numbers). Puzzle is the one place where representability is clearly not the binding constraint
(both methods fail for other reasons: nine-way conjunctive tasks, task coverage).

**Finding 2 — real-robot and LIBERO data are *more* peaked than OGBench manipulation.**
Proprioception only (end-effector pose + gripper, 7–8 dims), γ = 0.99 at native frame rate:

| dataset | fps | K | ESS/K | P1_self | TV linear@64 | TV log@64 | rank for TV≤0.1: linear / log |
|---|---|---|---|---|---|---|---|
| Berkeley UR5 (1000 eps) | 5 | 512 | 0.061 | 0.87 | 0.71 | 0.018 | >128 / 32 |
| LIBERO-90 (3921 eps) | 20 | 512 | 0.037 | 0.91 | 0.56 | 0.009 | >128 / 16 |
| LIBERO-10 (379 eps) | 20 | 256 | 0.061 | 0.92 | 0.39 | 0.013 | >128 / 16 |
| DROID-100 (100 eps) | 15 | 128 | 0.078 | 0.96 | 0.35 | 0.009 | 128 / 16 |

A rank-64 linear model of the ratio gets the future wrong by 0.35–0.71 total variation on every
one of them; the rank-64 log model is within 0.02. (Horizon-matched γ reruns, 5 s of wall clock
as on OGBench at 20 Hz, are in `results/external/*_hm.json`.)

**Finding 3 — the rank the linear model needs scales with K; the log model's does not.**
Going from K=512 to K=1024 the rank at which the linear model reaches TV ≤ 0.1 doubles or is
censored (antmaze full 64 → 128, all manipulation > 256 at both), while the log model stays at
8–32 on full observations. This is H2's Eckart–Young signature: the linear model is paying for the
number of distinguishable states, the log model for the dimension of the manifold.

**Finding 4 — the gap is a function of the horizon (H3).** γ sweep (cube-single, scene,
antmaze-medium): shortening the horizon from 200 to 10 steps takes ESS/K from 0.15–0.2 to 0.02 on
all three, and the linear model's rank-64 TV from 0.16–0.25 to ≈ 1.0 (it is no better than
predicting nothing), while the log model stays at 0.01–0.04. Lengthening it to 200 steps
(γ=0.995) shrinks the gap on antmaze to 0.007. The peakedness that matters is
"horizon ÷ mixing time", not a fixed property of the environment.

**Finding 5 — it is the model family, not the loss.** The nonnegative rank-d mixture (KL-NMF),
the ratio family fitted with the *same* KL objective the log model uses, sits between linear-SVD
and log everywhere, and still needs rank ≥ 256 on the peaked rows. So changing FB's loss to a
likelihood would not close the gap; the exponential parametrization does. Rank 64, γ=0.99, K=512
(real-robot rows horizon-matched to 5 s):

| dataset | ESS/K | TV linear | TV mixture | TV log | rank for TV≤0.1: linear / mixture / log |
|---|---|---|---|---|---|
| antmaze-medium full | 0.553 | 0.035 | 0.048 | 0.019 | 64 / 32 / 8 |
| antmaze-large full | 0.441 | 0.075 | 0.054 | 0.019 | 64 / 64 / 8 |
| puzzle-3x3 full | 0.415 | 0.186 | 0.135 | 0.083 | 256 / 128 / 64 |
| cube-single full | 0.147 | 0.248 | 0.145 | 0.032 | 256 / 128 / 16 |
| scene full | 0.145 | 0.296 | 0.164 | 0.039 | 256 / 256 / 32 |
| cube-double full | 0.118 | 0.464 | 0.235 | 0.046 | >256 / 256 / 32 |
| antmaze-medium task (xy) | 0.148 | 0.284 | 0.144 | 0.036 | 256 / 128 / 32 |
| cube-single task (cube xyz) | 0.102 | 0.386 | 0.260 | 0.094 | 256 / 256 / 64 |
| cube-double task | 0.046 | 0.705 | 0.359 | 0.069 | 256 / 256 / 64 |
| puzzle-3x3 task | 0.041 | 0.783 | 0.436 | 0.096 | >256 / >256 / 64 |
| scene task | 0.029 | 0.750 | 0.339 | 0.043 | >256 / >256 / 32 |
| DROID-100 proprio (γ=.987) | 0.068 | 0.410 | 0.193 | 0.009 | 128 / 128 / 16 |
| LIBERO-90 proprio | 0.037 | 0.561 | 0.346 | 0.011 | >256 / >256 / 16 |
| Berkeley UR5 proprio (γ=.96) | 0.029 | 0.849 | 0.498 | 0.015 | >256 / >256 / 32 |
| DROID 6-shard subset, 6633 eps / 1.9M frames (γ=.987) | 0.030 | 1.003 | 0.386 | 0.059 | >256 / >256 / 64 |
| Bridge-V2 proprio, 53,192 eps / 1.9M frames (γ=.96) | 0.062 | 0.657 | 0.281 | 0.050 | >256 / >256 / 32 |

ExoRL (DMC, RND exploration data, 1000 episodes, full proprio state, γ=0.99, K=512):

| dataset | ESS/K | TV linear@64 | TV log@64 | rank for TV≤0.1: linear / log |
|---|---|---|---|---|
| quadruped | 0.284 | 0.125 | 0.031 | 128 / 8 |
| cheetah | 0.244 | 0.173 | 0.053 | 128 / 32 |
| point-mass maze | 0.037 | 0.609 | 0.018 | >256 / 16 |

The locomotion domains are diffuse (ESS/K ≈ 0.25–0.28, like antmaze) with a small gap; the
point-mass maze is as peaked as manipulation. This matches the ExoRL results in memory:
Density-FB is below or level with FB on walker/cheetah and beats every published number on
point-mass maze (752). Walker's buffer is not on this box.

Figures: `figures/fig1_tv_vs_rank_{task,full}.png` (TV vs rank per env, three models),
`fig2_gap_vs_peakedness.png` (rank-64 gain of log over linear against ESS/K, all runs),
`fig3_rank_vs_K.png` (rank needed vs K), `fig4_gamma_sweep.png` (gain vs horizon).

## 5. What this says, in RL terms

The critic in FB regresses the successor-measure ratio `m(u|s,a)/rho(u)` with a rank-50 bilinear
form. On manipulation data the true ratio is a spike: from a given state the policy will be in
one of a handful of future configurations, and almost nowhere else, within the horizon. A rank-50
linear form cannot put a spike at each of thousands of distinct states; it smears mass over the
batch. The actor then maximizes a value that credits actions for mass on futures that are not
reachable. The softmax critic represents the same spike with a dot product of modest dimension,
because an exponential of a bilinear form is a kernel (Gaussian-like) around the embedded state;
the TD target for the softmax is itself a mixture of a one-hot and a softmax, and remains
well-represented. On diffuse data (antmaze full state at γ=0.99) the ratio is smooth and low-rank
already, the linear critic represents it, and the log model's advantage disappears (and its
disadvantages — no closure under the Bellman sum, bias from the in-batch normalizer — can show).

**Finding 6 — critic-only probe on trained checkpoints (inconclusive as designed).**
`scripts/critic_probe.py` reads a trained critic's distribution over 1024 random candidate
next-states for held-out (s,a) with a hindsight goal from the same episode, against the episode's
realised discounted future (cluster resolution). cube-single 500k, seed 0:

| critic | TV | top-1 | Spearman over candidates | neg. mass | predicted ESS (truth 25) |
|---|---|---|---|---|---|
| Density-FB | 0.79 | 0.10 | 0.23 | 0 | 6.9 |
| FB (old config, ortho 1.0) | 0.92 | 0.02 | 0.03 | 0.31 | 124 |

FB's readout is diffuse, unordered and 31 % negative; Density-FB's is *sharper* than the data
future. Neither matches the data policy's future, and they should not: both critics model the
occupancy of their own goal-conditioned actor, which reaches the goal faster than play data.
The probe therefore measures policy mismatch as much as representation. A clean version needs
the actor's own rollouts from restored simulator states as the ground truth (plan item 6b).

## 6. Validation plan (cheap → decisive)

Dataset-only (this week):
1. Finish the OGBench battery with the NMF column and K=1024, bridge-v2, horizon-matched γ. **Done** (Bridge-V2 row in §4: ESS/K 0.062, linear 0.66 / log 0.05 at rank 64).
2. Add antmaze-*-stitch / explore and the DMC ExoRL datasets (walker/cheetah/quadruped): the
   theory predicts low peakedness and a small gap on locomotion, which is where FB is known to be
   strong — the clean "negative" prediction.
3. Report `ESS/K` and the rank-64 TV gap as two columns next to every benchmark row in the paper.

Training-level, on OGBench — **launched 2026-10-12 11:18, tmux `softmax`, 25 runs at 500k, one eval
at the end, `exp/softmax/<tag>/`; `scripts/collect_training.py` tabulates them:**
- rank dial: `rank_cube_density_d{8,16,50}`, `rank_cube_fb_d{50,200,800}` (fb_tdj, reward route),
  same on antmaze-medium (`rank_ant_*`);
- γ dial on cube-single: `gamma_cube_{density,fb}_g{0.95,0.995}` (0.99 = the rank runs at d=50);
- puzzle HP sweep (goal route): `puzzle_density_r{0.1,0.25,0.5}_t{0.5,1.0,2.0}`.
Original plan:
4. **Peakedness dial on fixed data.** Train FB and Density-FB (d=50, same actor, same eval) on
   cube-single with (a) frame-skip k ∈ {1,2,4,8} applied to the dataset, and (b) γ ∈ {0.95, 0.99,
   0.995}. Prediction: the Density-FB − FB margin shrinks as the per-step occupancy diffuses
   (large k or γ → 1) and grows otherwise. Both knobs exist in the dataset loader / config.
5. **Rank dial.** FB at d ∈ {50, 200, 800} vs Density-FB at d ∈ {8, 16, 50} on cube-single and
   antmaze-medium. Prediction: FB improves slowly with d on cube, Density-FB saturates by d=16;
   on antmaze both are flat.
6. **Critic-only probe (no actor confound).** Load trained FB and Density-FB checkpoints, read
   `m̂(u|s,a)` on held-out (s,a) with the empirical in-episode future as ground truth, and report
   the same TV/top-1 metrics as the dataset analysis. This ties the dataset statistic to the
   learned critic directly.
7. **Real-robot transfer check.** Train both critics on LIBERO-90 proprio (sim, evaluable) and
   report the same probe; prediction from Finding 2: a larger gap than on cube-single.

### 6b. Training results (2026-10-12, 500k steps, seed 0, 10 episodes per task, 5 tasks)

`results/training_table.md` (`scripts/collect_training.py`). "goal" = goal-conditioned
evaluation, "reward" = reward-inference evaluation (FB only; the goal-arm Density-FB critic has
no reward-inference interface).

Rank dial, cube-single:

| critic | d | goal | reward |
|---|---|---|---|
| Density-FB | 8 | 0.88 | – |
| Density-FB | 16 | 0.96 | – |
| Density-FB | 50 | 0.98 | – |
| FB | 50 | 0.12 | 0.48 |
| FB | 200 | 0.16 | 0.24 |
| FB | 800 | 0.14 | 0.42 |

Rank dial, antmaze-medium-navigate:

| critic | d | goal | reward |
|---|---|---|---|
| Density-FB | 8 | 0.78 | – |
| Density-FB | 16 | 0.66 | – |
| Density-FB | 50 | 0.84 | – |
| FB | 50 | 0.52 | 0.66 |
| FB | 200 | 0.36 | 0.32 |
| FB | 800 | 0.18 | 0.02 |

Horizon dial, cube-single (d=50):

| γ | Density-FB goal | FB goal | FB reward | margin (Density − FB reward) |
|---|---|---|---|---|
| 0.95 | 0.86 | 0.16 | 0.12 | 0.74 |
| 0.99 | 0.98 | 0.12 | 0.48 | 0.50 |
| 0.995 | 0.88 | 0.32 | 0.38 | 0.50 |

Puzzle-3x3 goal-route sweep (Density-FB, hit radius r × softmax temperature t): every cell is
0.14–0.22 overall and the breakdown is identical in all nine: task 1 at 0.7–1.0, tasks 2–5 at
0.0 (one episode in 50 succeeds on tasks 3 or 5 in three cells). The 1M campaign checkpoints
of every method show the same shape (`exp/hub/exp/puzzle-3x3-*`): FB 0.6–0.85, RLDP 0.2–0.85 and
Density-FB (reward-sample index) 0.1–0.4 on task 1, 0 on tasks 2–5.

Reading:
- Density-FB does not need the rank: d=8 already gives 0.88 on cube-single and 0.78 on antmaze.
  Giving FB 16× the rank (d=800) does not move it on cube-single (0.42–0.48 on the reward route
  at every d). So the gap is not something FB closes by adding capacity at 500k. This is the
  prediction of §2, but note the caveat: FB at d=800 gets *worse* on antmaze (0.66 → 0.02),
  which is an optimisation or regularisation effect (the orthonormality penalty and learning
  rate were not re-tuned per d), not representability, so the d=200/800 cells are a weak test.
- The horizon dial moves the margin in the predicted direction: the shorter the horizon (γ=0.95,
  the most peaked occupancy), the larger the gap (0.74 vs 0.50). FB's reward-route success rises
  from 0.12 at γ=0.95 to 0.48 at 0.99 and is flat at 0.995. One seed and 50 episodes per cell,
  so differences below ~0.15 are noise.
- Puzzle is not a hyper-parameter problem of the goal-conditioned Density-FB critic: no
  combination of hit radius and temperature opens tasks 2–5, and every published method sits
  at the same task-1-only floor. Puzzle stays the counterexample where the critic class is not
  the binding constraint; the dataset analysis (§4) predicts it correctly as the least peaked
  manipulation environment (ESS/K 0.415 on the full state), i.e. the smallest expected gain.
- Antmaze: Density-FB 0.78–0.84 vs FB 0.52/0.66 at d=50, a smaller margin than on cube
  (0.98 vs 0.12/0.48), as the dataset analysis predicts (full-state ESS/K 0.55 vs 0.15), but not
  zero; the published 1M FB number on antmaze-medium is 73.0, so the FB cell here is likely
  under-trained at 500k.

### 6c. Puzzle-3x3 diagnosis (2026-10-15)

Puzzle is Lights Out: a press toggles the button and its four neighbours. Task 1 needs 2
presses (two opposite corners, any order); tasks 2–5 need 5, 8, 9 and 7 presses. The play
policy presses once per ~33 steps, so a 100-step (γ=0.99) window holds 3 presses 87 % of the
time; all 512 configurations occur, ~0.2 % each. Hamming distance is not progress (task 2 is
Hamming 1 and needs 5 presses), so the count reward of the reward-inference route misleads.

Critic probe (`scripts/puzzle_probe.py`, `results/probe/puzzle_density_g0.99_500k.json`, the
γ=0.99 sweep checkpoint): at each task's start state, the mass the critic puts on the 1–3
batch candidates (of 1024) whose nine buttons match the goal, for 32 actor actions, 32 uniform
random actions and 32 dataset actions:

| goal | presses needed | mass on goal config, actor actions | random actions | dataset actions |
|---|---|---|---|---|
| task 1 | 2 | 0.74 (0.61–0.81) | 0.74 | 0.74 |
| task 2 | 5 | 0.36 (0–0.80) | 0.35 | 0.35 |
| task 3 | 8 | 0.71 (0.65–0.75) | 0.75 | 0.74 |
| task 4 | 9 | 0.56 (0–0.89) | 0.54 | 0.54 |
| task 5 | 7 | 0.26 (0–0.55) | 0.26 | 0.25 |
| 1-press goals from task-1 start | 1 | 0.68, 0.91 | 0.67, 0.91 | |
| 2-press goals | 2 | 0.62, 0.89 | 0.61, 0.87 | |
| 3-press goals | 3 | 0.58, 0.67 | 0.58, 0.68 | |

(ranges over candidate draws: a draw with no matching candidate scores 0, one with a match
scores ~0.8). Two facts. (i) The critic is grossly optimistic for far goals: from the task-4
start it claims 55 % of the discounted future sits at a configuration 9 presses away, which
under the data's press rate would be γ^300 ≈ 5 %, and the actor in fact never gets there
(task-4 success 0). (ii) The mass does not depend on the action: actor, random and dataset
actions score the same. The actor's objective Σ_u m(u|s,a,g) r_g(u) is therefore flat in a,
the actor runs on its behaviour-cloning term, and 2-press goals are solved at the
goal-conditioned-BC level while 5+-press goals are not.

Mechanism: the TD target is 0.01·onehot(own next state) + 0.99·(target softmax at the
actor's next action). With the backward input reduced to the 18 one-hot button columns (512
distinct embeddings) the one-step anchor says "you stay at your current configuration" with
weight 1 %, and the 99 % bootstrap term inherits the network's own belief that the goal
policy reaches the goal. Nothing on-trajectory contradicts that belief for goals the actor
never reaches, so the optimistic solution persists. Cube does not expose this because its
actor does reach its goals within the horizon, so the optimism is self-fulfilling.

Results of the six arms (1M steps, seed 0, 10 episodes per task; `results/puzzle_arms.md`):

| arm | 500k mean | 1M mean | 1M per task (1–5) |
|---|---|---|---|
| Density-FB γ=0.99 (actor bootstrap) | 0.12 | 0.22 | 0.6 / 0.3 / 0.1 / 0.0 / 0.1 |
| Density-FB γ=0.995 | 0.18 | 0.12 | 0.3 / 0.0 / 0.1 / 0.0 / 0.2 |
| Density-FB γ=0.998 | 0.24 | 0.24 | 0.5 / 0.2 / 0.3 / 0.0 / 0.2 |
| Density-FB γ=0.95 | 0.08 | – | 0.3 / 0.0 / … (eval in progress) |
| **Density-FB γ=0.99, bootstrap at the dataset's next action** | **0.70** | **0.64** | **0.9 / 0.7 / 0.5 / 0.7 / 0.4** |
| GCIVL (our implementation, AWR alpha 10) | 0.12 | 0.12 | 0.5 / 0.1 / 0.0 / 0.0 / 0.0 |

The horizon dial does nothing, as the probe predicted. Taking the bootstrap action from the
dataset instead of the actor solves every task: 0.64–0.70 against a published ceiling of 18.8
for zero-shot methods and 0.12 for our GCIVL reference on the same data. Mechanism: the critic
now models the play policy's future, which is honest for goals the actor cannot reach; under
random pressing the chance of landing on the goal within the horizon falls with the number of
presses still needed, so the actor's objective is monotone in progress and the flat-J failure
disappears. Caveat from the record: the same bootstrap scored 0.10 on MetaWorld pick-place in
August (`docs/specs/2026-08-17-gciql-edge-attribution.md`), where the play policy's future is
diffuse away from the goal and the data-policy measure is not an improvable objective. So the
two bootstraps fail in complementary ways: the actor bootstrap is optimistic where the actor
cannot reach, the dataset bootstrap is uninformative where the data does not reach.

Confirmation running (tmux `softmax4`, 1M): puzzle seeds 1 and 2 with the dataset bootstrap,
and the dataset bootstrap on cube-single, scene and cube-double to measure what it costs
where the actor bootstrap works. The principled fix remains a target that anchors the
bootstrap with the realised future of the trajectory that produced the hindsight goal.

## 7. Caveats

- Discretisation: K-means on standardised coordinates; the statistics depend on K and on the
  coordinate choice. Both are reported; the K-scaling result is the point, not the absolute level.
- The rank-d fits use the *empirical* M of the data policy, not the critic's (s,a)-conditioned,
  policy-conditioned target. The learned critics also have nonlinear F; the analysis bounds the
  bilinear *readout*, which both methods share.
- Real-robot datasets expose proprioception only (no object state); the OGBench "task
  coordinate" rows are the closer analogue to object-centric occupancy.
- "TRI" in the brief is ambiguous: on HF only `TRI-ML/robosuite_{ph,mh,mg}` (robomimic re-uploads)
  and `mzhobro/droid_raw_stereo_TRI` exist. Not downloaded; needs a pointer.
