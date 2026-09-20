# Separation preservation: does the critic separate the futures of the best and the worst action? (cube-single, 2026-09-19)

Figure (two panels, one file): `PAPER/ICLR/figures/gap_preservation/fig_separation_by_phase.{pdf,png}`
-- top: separation preserved R; bottom: alignment of the predicted with the true difference of futures.
Tables: `separation_by_phase_cell{3,5,8}cm.csv` (per phase, pooled over seeds),
`separation_by_phase_per_seed.csv` (5 cm), `separation_per_state.csv` (one row per state,
model and grid, now with `align` / `align_top`), `separation_fb_clipping.json`,
`alignment_by_phase_cell{3,5,8}cm.csv`, `alignment_by_phase_per_seed.csv`,
`alignment_permutation_null.csv`. Script:
`scripts/probes/cube_bottleneck/separation_preservation.py` (`roll` per seed, `aggregate`;
`--smoke` = 3 carrying states x 2 actions x 2 rollouts, H = 20: 38 s on CPU). Test:
`tests/probes/test_cube_bottleneck_separation.py` (9 tests: histogram, TV, discounting,
per-state row, per-phase table, cosine alignment, alignment table and null). Raw futures and predictions:
`analysis/failure_figs/cube-single-play-v0/separation/futures_seed{0,1,2}_{bank}.{npz,csv}`.

## Protocol

- Models and states: `fb_tdj` and `density_fb_z`, tdjtab 1M, seeds 0/1/2; the three 64-state
  banks per seed (approach, in-hand, carrying) of the failure figures, restored with the
  verified restore; the SAME a+ / a- per state and model as the decision-by-phase figure
  (true best / worst of the 12 candidates by Monte-Carlo return, read from
  `gap_preservation_per_state.csv`; the candidate set is regenerated with the action probe's
  rng keys). States whose 12 candidates all return 0 have no pair and are skipped (FB 23,
  Density-FB 4 of 576). Task vector z: the same reward-inferred z (10k relabelled rows per
  task, fixed seed).
- True futures M_MC(s, a): restore s, force a once, continue with the model's own frozen
  policy (temperature 0) for H = 100 steps, K = 10 rollouts; cube position from obs[19:22]
  (metres) at every step, weighted gamma^t with gamma = 0.99 (the run's discount),
  normalised over the horizon and the rollouts.
- Critic futures M_hat(s, a): one pool of N = 4096 dataset next-states (seed 20260919, the
  same pool for both models, both actions and every state). Density-FB: the softmax over the
  pool of F(s,a,z).B(x_j)/tau, averaged over heads -- the distribution its actor readout
  ascends. FB: min over heads of F(s,a,z).B(x_j), clipped at 0 and normalised over the pool.
  FB clipping: 50.2 % of pool entries are negative and clipped (mean over 553 (state, action
  pair) rows); the clipped mass is 33.7 % of the total absolute mass.
- Discretisation: cube positions of both the rollouts and the pool states binned on a 3-D
  grid of cubic cells over the workspace x in [0.25, 0.60], y in [-0.35, 0.35], z in
  [-0.05, 0.40] (points outside go to the border cell); main cell 5 cm (success radius
  4 cm), sensitivity 3 cm and 8 cm. TV distance between cell histograms.
  D_true(s) = TV(M_MC(s,a+), M_MC(s,a-)); D_pred(s) = TV(M_hat(s,a+), M_hat(s,a-));
  R(s) = D_pred / D_true; fidelity(s,a) = TV(M_hat(s,a), M_MC(s,a)).
- Exclusion: states with D_true < 0.05. At 5 cm none is excluded (every kept state's two
  futures differ); at 8 cm 25 states are (17 Density-FB approach, 6 Density-FB carrying,
  2 FB).
- Compute: 9 CPU jobs (seed x bank) in parallel, 8.5-10.6 min each, 11 min wall in total;
  1125 (state, model) pairs at K = 10 (no reduction needed).

## Per-phase table (5 cm cells, pooled over seeds; R mean +- s.e. over states)

| model | phase | n | D_true | D_pred | R mean +- s.e. | R median | fidelity |
|---|---|---|---|---|---|---|---|
| FB | approach | 182 | 0.301 | 0.018 | 0.069 +- 0.005 | 0.048 | 0.838 |
| FB | in hand | 185 | 0.539 | 0.027 | 0.051 +- 0.003 | 0.037 | 0.822 |
| FB | carrying | 186 | 0.737 | 0.036 | 0.054 +- 0.003 | 0.041 | 0.880 |
| Density-FB | approach | 188 | 0.356 | 0.045 | 0.170 +- 0.012 | 0.121 | 0.669 |
| Density-FB | in hand | 192 | 0.696 | 0.100 | 0.150 +- 0.006 | 0.129 | 0.704 |
| Density-FB | carrying | 192 | 0.715 | 0.106 | 0.187 +- 0.016 | 0.130 | 0.700 |

Per seed (0/1/2), R mean at carrying: FB 0.043 / 0.067 / 0.050; Density-FB 0.189 / 0.207 /
0.166. In-hand: FB 0.039 / 0.059 / 0.053; Density-FB 0.164 / 0.145 / 0.141.

Grid sensitivity (R mean, approach / in hand / carrying):
3 cm -- FB 0.063 / 0.051 / 0.056, Density-FB 0.178 / 0.145 / 0.131;
5 cm -- FB 0.069 / 0.051 / 0.054, Density-FB 0.170 / 0.150 / 0.187;
8 cm -- FB 0.078 / 0.050 / 0.046, Density-FB 0.244 / 0.146 / 0.155.
The ordering FB < Density-FB by a factor 2.5-4 holds at every phase and every grid.

## Reading (honest)

The real futures after a+ and a- differ a lot once the cube is held (D_true 0.54-0.74, on a
0-1 scale), and they already differ at approach (0.30-0.36). Neither critic reproduces that
difference: FB's two predicted futures are nearly the same (D_pred 0.02-0.04, R about 0.05
at every phase, i.e. 5 % of the true separation), Density-FB's differ by 0.05-0.11 (R
0.15-0.19, 3-4x FB). "Density-FB keeps most of the difference" is NOT what the numbers say;
it keeps about a fifth, FB about a twentieth. Two reasons the absolute R is small for both:
the critic's distribution is spread over the whole 4096-row pool while ten rollouts of a
deterministic policy concentrate on a handful of cells (fidelity TV 0.67-0.88 for both
models, i.e. predictions are far broader than the empirical future), and the a+ / a-
pairs differ by one forced step, so both readouts, which condition on (s, a) through one
forward tower, move little. FB's clipping (half the pool negative) removes a third of its
mass before normalisation, which further flattens its prediction.

## Alignment: does the critic's residual difference point the right way?

From the same saved per-state histograms (no re-roll). Over the cells of the grid,
d_hat(s) = M_hat(s,a+) - M_hat(s,a-) and d_true(s) = M_MC(s,a+) - M_MC(s,a-);
Align(s) = <d_hat, d_true> / (||d_hat|| ||d_true||). A state is skipped when either vector is
zero: none at 3 and 5 cm, 4 at 8 cm (3 Density-FB carrying, 1 FB carrying). Robustness: the
same cosine on the 20 cells of largest |d_true| ("top"). Null: 200 shuffles of which state's
d_true is paired with which state's d_hat within each (model, bank), pooled over seeds;
mean +- sd of the mean alignment over shuffles.

5 cm cells, pooled over seeds (mean +- s.e. over states; frac > 0; null mean +- sd):

| model | phase | n | Align mean +- s.e. | median | frac > 0 | top-20 mean +- s.e. | null mean +- sd |
|---|---|---|---|---|---|---|---|
| FB | approach | 182 | -0.020 +- 0.011 | -0.003 | 0.48 | -0.035 +- 0.026 | 0.000 +- 0.008 |
| FB | in hand | 185 | 0.066 +- 0.009 | 0.058 | 0.71 | 0.168 +- 0.024 | 0.003 +- 0.009 |
| FB | carrying | 186 | 0.026 +- 0.008 | 0.028 | 0.60 | 0.070 +- 0.027 | -0.000 +- 0.007 |
| Density-FB | approach | 188 | -0.022 +- 0.029 | 0.042 | 0.53 | -0.018 +- 0.034 | 0.005 +- 0.013 |
| Density-FB | in hand | 192 | 0.125 +- 0.018 | 0.124 | 0.72 | 0.168 +- 0.023 | 0.009 +- 0.009 |
| Density-FB | carrying | 192 | 0.190 +- 0.025 | 0.178 | 0.72 | 0.233 +- 0.032 | 0.012 +- 0.012 |

Per seed (0/1/2), Align mean: carrying FB -0.010 / 0.044 / 0.040, Density-FB 0.194 / 0.198 /
0.179; in hand FB 0.059 / 0.063 / 0.075, Density-FB 0.124 / 0.155 / 0.096; approach both
models within +-0.05 of zero every seed.
Grid sensitivity (Align mean, approach / in hand / carrying): 3 cm FB -0.005 / 0.026 / 0.015,
Density-FB 0.004 / 0.055 / 0.105; 8 cm FB -0.005 / 0.082 / 0.037, Density-FB -0.018 / 0.164 /
0.219. The top-20-cell version is larger than the full cosine everywhere (the tail cells add
noise to both vectors) and keeps the same ordering.

Reading. At approach neither critic's residual is aligned with the true difference (both at
the null, about half the states positive). Once the cube is held the residual points the
right way: at in-hand both do (FB 0.07, Density-FB 0.13, 71-72 % of states positive); at
carrying FB drops back to 0.03 (60 % positive, one seed below zero) while Density-FB rises
to 0.19 (72 % positive, all seeds 0.18-0.20), fifteen null standard deviations above the
shuffled pairing. So the direction of the residual, not only its size, separates the two
critics where the decision matters most.

Caption line (bottom panel): Both critics underestimate how different the two futures are;
FB nearly erases the difference, Density-FB keeps a larger residual, and that residual
points in the right direction (alignment), which is what the actor needs.

## Caption (plain RL terms)

After the best and the worst candidate action, the real cube futures under each model's own
policy differ (D_true, printed under the axis: about 0.3 at approach, 0.5-0.7 once the cube
is held). The critic's predicted futures for the same two actions differ far less: FB's are
almost identical at every phase (about 5 % of the true separation), while Density-FB keeps
15-19 % of it, three to four times more, with the largest margin at carrying. Both fall
well short of full preservation (dashed line).
