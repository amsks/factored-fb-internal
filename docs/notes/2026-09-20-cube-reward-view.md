# Reward view: what task reward does each critic see? (cube-single, 2026-09-20)

The cube-single analogue of the tabular lemma: a linear successor critic (FB) reads a reward
through its backward features, as if the reward were replaced by its projection onto the
feature span; at a bottleneck the projected reward is the same on both futures, so the
critic sees no action gap. Trained tdjtab 1M checkpoints, `fb_tdj` and `density_fb_z`,
seeds 0/1/2, the cube bottleneck probe infrastructure (banks, restores, a+ / a- pairs, z).

Figures: `PAPER/ICLR/figures/gap_preservation/fig_reward_view_hist.{pdf,png}` (part 1, every
task) and `fig_reward_view_gaps.{pdf,png}` (part 2, bars + scatter); compact paper versions
from the same saved data: `fig_reward_view_hist_main.{pdf,png}` (the "all tasks" panels
only, 3.0 x 2.6 in) and `fig_reward_view_gaps_main.{pdf,png}` (the bar row only, shared y,
4.2 x 2.3 in). Tables in the same directory:
`reward_view_hist_pooled.csv`, `reward_view_hist_per_seed.csv`, `reward_view_gaps_by_phase.csv`,
`reward_view_gaps_by_phase_per_seed.csv`, `reward_view_gaps_by_phase_consistent{,_per_seed}.csv`,
`reward_view_gaps_per_state.csv`, `reward_view_gaps_checks.json`. Script:
`scripts/probes/cube_bottleneck/reward_view.py` (stages `hist`, `hist_aggregate`, `roll`,
`score`, `tables`; `--smoke` = 2000 dataset states + 3 carrying states x 2 actions x 2
rollouts x H = 20, 77 s on one GPU). Tests: `tests/probes/test_cube_bottleneck_reward_view.py`
(8 tests: AUC against the pairwise definition with ties, d' and the standardisation, the
discounted mean, the critic-gap scale, the three-number row, the calibration slope, the
per-phase table and its sign fractions). Raw data:
`analysis/failure_figs/cube-single-play-v0/reward_view/hist_seed{0,1,2}.npz` (reward views on
the dataset states) and `futures_full_seed{s}_{bank}.npz` (full observations of every future
step, 71 MB in total).

## The two reward views (the critics' own functions, nothing re-implemented)

- FB: r_hat(u) = B(u)^T z_r. B is `FB._B` (impls/critics/fb.py:116, the choke point every
  B call routes through, so `backward_input` is honoured); z_r is the reward-inferred index
  `FB.infer_cond` returns (fb.py:213: z = r^T B over the relabelled rows, projected to norm
  sqrt(d)); the dot with z is how `FB.q_fn` reads F (fb.py:184). Called in
  `reward_view.reward_view`.
- Density-FB: r_z(u) = (B_xi(u) - mu_hat)^T z / sqrt(d), computed by
  `DensityFB.z_actor_reward_rows` (impls/critics/density_fb.py:1074) under
  `measure_density_z_actor_reward = 'latent_centered'` (the shipped setting, asserted). The
  cond it reads is built exactly as `sample_cond` builds it (density_fb.py:1222):
  `_density_zenc_u` = `_B(_zenc(net), u_sc) - zstats.mu`, i.e. the SLOW index tower and
  the stored feature mean; u_sc = `_b_query(net, u, u)`. This is the per-candidate reward the
  actor's `q_fn` ascends (density_fb.py:1133).
- z: the same reward-inferred index every cube probe uses (`agents.relabel_cond_agents`:
  10,000 relabelled training transitions per task, shift 1.0, fixed NumPy seed per task).
- True reward: `env/ogbench.py cube_relabel_reward` (cube within 0.04 m of the task goal,
  shift 1.0 -> 0/1) on the dataset's stored qpos; on rollouts the same rule on the cube joint
  position, as `failure_figs.truth_actions` does.

## Part 1: reward view on dataset states (no rollouts)

N = 20,000 dataset next-states (NumPy seed 20260920, the same rows for every critic, seed and
task), labelled per task (five tasks; positives 375 / 448 / 470 / 446 / 461, all >= 200).
Per (critic, seed, task): AUC of r_hat as a classifier of the true reward and
d' = (mean r_hat | r=1 - mean r_hat | r=0) / std(r_hat | r=0). "All tasks" pools every
(state, task) pair with its own task's z, i.e. one label "cube at this task's goal". Pooled
over seeds: r_hat standardised against the reward-0 states of its (critic, seed, task), then
concatenated (AUC and d' are unchanged by that); per-seed values in the per-seed csv.

| critic | task | AUC pooled | AUC per seed min-max | d' pooled |
|---|---|---|---|---|
| FB | 1 | 0.761 | 0.740-0.792 | 1.01 |
| FB | 2 | 0.892 | 0.878-0.917 | 1.75 |
| FB | 3 | 0.851 | 0.823-0.875 | 1.57 |
| FB | 4 | 0.874 | 0.842-0.912 | 1.62 |
| FB | 5 | 0.881 | 0.859-0.906 | 1.77 |
| FB | all | 0.855 | 0.834-0.884 | 1.56 |
| Density-FB | 1 | 0.998 | 0.998-0.998 | 8.32 |
| Density-FB | 2 | 0.998 | 0.998-0.998 | 9.61 |
| Density-FB | 3 | 0.998 | 0.998-0.999 | 8.26 |
| Density-FB | 4 | 0.999 | 0.999-0.999 | 8.83 |
| Density-FB | 5 | 0.998 | 0.998-0.999 | 9.14 |
| Density-FB | all | 0.998 | 0.998-0.998 | 8.85 |

Reading. FB's reward view overlaps: the reward-1 states sit 1.6 reward-0 standard deviations
above the reward-0 mean, and a reward-1 state outranks a reward-0 state 86 % of the time
(76 % on task 1). Density-FB's reward view separates them almost perfectly (8.9 standard
deviations, AUC 0.998, every task, every seed). The prediction "FB overlaps, Density-FB
separates" holds; "heavily" is too strong for FB (AUC 0.86 is a real but blurred signal, not
noise).

## Part 2: the three-number test (re-roll with full observations)

Protocol: the separation probe's re-roll (`separation_preservation.stage_roll`) with the full
observation stored at every step: the three 64-state banks per seed, the same a+ / a- per
state and model as the decision-by-phase figure (`gap_preservation_per_state.csv`; states
with a tie skipped: FB 23, Density-FB 4 of 576), the same rng keys, K = 10 futures of H = 100
steps under the model's own frozen policy after one forced action, both actions. Stored:
observation (float16; 0.1 mm on the cube position), cube position from the simulator joint
(float32), true hit, per step; 1125 (state, model) pairs.

Per (critic, seed, bank, state), gamma^t weights normalised over the horizon (gamma = 0.99,
sum_{t<100} gamma^t = 63.4):
- (i) TRUE gap = discounted mean true reward over the futures of a+ minus a-;
- (ii) FEATURE-LIMITED gap = the same futures scored with r_hat;
- (iii) CRITIC gap = Q(s,a+) - Q(s,a-), the `delta_hat` of the decision-by-phase probe
  (`failure_figs.q_values`: FB `q_fn` = min over heads of F^T z; Density-FB `q_fn` = the
  measure-weighted r_z over its 1024-state pool, min over heads). FB's Q regresses
  r + gamma Q, an unnormalised discounted sum, so it is divided by sum_{t<100} gamma^t;
  Density-FB's Q is an expectation of r_z under a normalised measure (softmax over the
  pool), so it is left as is.
- Units: r_hat is in each critic's own units, so (ii) and (iii) are converted to true-reward
  units by ONE least-squares slope per (critic, seed), the 0/1 reward regressed on r_hat over
  the 20,000 x 5 dataset rows of part 1 (the intercept cancels in a gap). Slopes: FB 0.0059 /
  0.0051 / 0.0054, Density-FB 0.0242 / 0.0235 / 0.0248 (seeds 0/1/2). Sign fractions and
  correlations do not depend on this; the raw values are in the per-state csv
  (`feat_gap_raw`, `critic_gap_q`, `critic_gap_norm`).

Pooled over seeds (mean +- s.e. over states; sign = fraction of states with the gap > 0;
corr = Pearson of (ii) with (iii) across states):

| critic | phase | n | (i) true | (ii) feature-limited | (iii) critic | (ii)/(i) | (iii)/(i) | sign (ii) | sign (iii) | corr (ii),(iii) |
|---|---|---|---|---|---|---|---|---|---|---|
| FB | approach | 182 | 0.024 +- 0.005 | 0.006 +- 0.001 | -0.000 +- 0.001 | 0.24 | -0.02 | 0.66 | 0.49 | -0.10 |
| FB | in hand | 185 | 0.133 +- 0.015 | 0.013 +- 0.001 | 0.004 +- 0.001 | 0.10 | 0.03 | 0.78 | 0.65 | 0.59 |
| FB | carrying | 186 | 0.455 +- 0.025 | 0.005 +- 0.002 | 0.002 +- 0.001 | 0.01 | 0.004 | 0.57 | 0.50 | 0.69 |
| Density-FB | approach | 188 | 0.134 +- 0.010 | 0.054 +- 0.004 | -0.005 +- 0.003 | 0.40 | -0.04 | 0.94 | 0.47 | -0.07 |
| Density-FB | in hand | 192 | 0.413 +- 0.011 | 0.142 +- 0.005 | 0.016 +- 0.004 | 0.34 | 0.04 | 0.97 | 0.64 | 0.14 |
| Density-FB | carrying | 192 | 0.358 +- 0.015 | 0.061 +- 0.011 | 0.043 +- 0.005 | 0.17 | 0.12 | 0.72 | 0.77 | 0.05 |

Per seed (0/1/2). FB carrying: (ii)/(i) -0.002 / 0.016 / 0.021, (iii)/(i) -0.001 / 0.006 /
0.008, corr 0.60 / 0.73 / 0.77; FB in hand: (ii)/(i) 0.10 / 0.10 / 0.10, (iii)/(i) 0.02 /
0.03 / 0.05, corr 0.61 / 0.47 / 0.71. Density-FB carrying: (ii)/(i) 0.19 / 0.16 / 0.15,
(iii)/(i) 0.17 / 0.15 / 0.05, sign (iii) 0.86 / 0.77 / 0.69; Density-FB in hand: (ii)/(i)
0.34 / 0.32 / 0.37, (iii)/(i) 0.08 / 0.03 / 0.004. Spearman correlations are within 0.05 of
the Pearson ones everywhere. Restricting to the states whose re-rolled true gap is positive
(957 of 1125, see caveat 1) changes no ratio by more than 0.1 and no sign fraction by more
than 0.08 (`_consistent` tables).

Reading (honest).
- FB, where the cube is held: the true futures of a+ and a- differ a lot (0.13 in hand, 0.46
  carrying, on a 0-1 scale), but the same futures scored with FB's own reward view differ by
  10 % of that in hand and 1 % at carrying, and the critic's gap is smaller still (3 % and
  0.4 %). The critic gap tracks the feature-limited gap across states (r = 0.59 in hand, 0.69
  carrying, per seed 0.47-0.77) and both have the right sign at about the rate a coin does at
  carrying (57 % and 50 %). This is the lemma's picture: what FB's Q can see of the action
  gap is bounded by what its reward view sees on the futures, and on cube-single that is
  almost nothing once the cube is carried.
- Density-FB: its reward view keeps 34-40 % of the true gap at approach and in hand with the
  right sign at 94-97 % of states, and 17 % at carrying (72 %). Its critic gap is 12 % of the
  true gap at carrying (0.043 vs 0.358, right sign 77 %, 10x FB's 0.004 ratio) but only 4 %
  in hand and zero at approach, and it does not track the feature-limited gap across states
  (r 0.05-0.14). So "(iii) ~ (i) for Density-FB" is NOT what the numbers say: Density-FB's
  critic underestimates the action gap as well; it is larger than FB's and right-signed
  where the decision matters (carrying), and, unlike FB, the limit is not the reward it reads
  (part 1, and (ii) here).
- Approach: neither critic has a usable gap (both (iii) at zero, signs at chance), consistent
  with the decision-by-phase figure.

## Caveats

1. The a+ / a- labels come from the action probe's rollouts (`failure_figs.stage_actionprobe`,
   rng key 99 + 13 i + seed, 120 envs stepped in lockstep). The flow actor draws its noise
   from the rng key at every temperature (impls/actors/flowbc.py:124; the earlier notes'
   "temperature 0" does not make the rollouts deterministic), so this re-roll (key 777 +
   13 i + seed, 20 envs, the separation probe's convention) is another Monte-Carlo draw of
   the same policies: the re-rolled true gap x 63.4 correlates 0.93 with the probe's
   `delta_mc` (median |difference| 3.5 return units) and is negative at 10 % of states, zero
   at 5 %. The re-roll does reproduce the separation probe's futures: the cube path from the
   stored observations matches `separation/futures_seed{s}_{bank}.npz` to 0.1 mm at 96.5 % of
   the (state, model) pairs (float16 storage), the rest diverge chaotically (max 6.5 cm).
2. Density-FB's value is not a linear readout of its reward view: Q integrates r_z under the
   critic's measure over a fixed pool, not under the policy's own futures, so (ii) for
   Density-FB is a diagnostic of its reward view, not the lemma's prediction; only for FB is
   (ii) ~ (iii) a prediction (and it holds in correlation and in size).
3. The unit conversion of (ii) and (iii) is one global slope per checkpoint from dataset
   states; it cannot repair per-state scale errors. Ratios in the table depend on it, sign
   fractions and correlations do not.
4. FB's (iii) is the min over the two heads (the actor's readout), divided by the H = 100
   truncated discount sum; the infinite-horizon normaliser 1/(1 - gamma) = 100 would scale
   FB's (iii) by 0.63.
5. Density-FB's (iii) uses the action probe's 1024-state pool; part 1 and (ii) use dataset
   states directly (no pool).

## Runtime and reproduce

- Part 1: `hist --seeds 0 1 2`: 4.6 min on one GPU (checkpoint loading dominates).
- Re-roll: 9 CPU jobs (seed x bank), 8.5-11 min each, 21 min wall including checkpoint
  loading; 71 MB of futures.
- Score: 15 min on one GPU (a jit per state; `tables` re-derives tables and figure in seconds).

```
export CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=offline
PY=/dev/shm/.venv-jax/bin/python; S=scripts/probes/cube_bottleneck/reward_view.py
$PY $S --smoke                                   # 77 s end to end
$PY $S hist --seeds 0 1 2                        # part 1
for s in 0 1 2; do for b in approach inhand carrying; do
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" $PY $S roll --seeds $s --banks $b > logs/reward_view/roll_seed${s}_${b}.log 2>&1 &
done; done; wait
$PY $S score                                     # part 2 (+ tables, figure)
$PY $S tables                                    # tables + figure only
$PY -m pytest tests/probes/test_cube_bottleneck_reward_view.py -q
```
