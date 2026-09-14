# 2026-09-14 -- Toy ablation of Density-FB's three changes to FB (nine-state bottleneck MDP)

Code: `analysis/toy_ablation/run.py` (+ `diag_fb_rank3.py`), data `closed.csv`,
`trained.csv`, `diag_fb_rank3.csv`, `exact.json`, `run_log.txt` in the same folder.
Figure: `PAPER/ICLR/figures/toy_ablation/fig_toy_ablation.{pdf,png}` (two panels:
closed-form gap vs rank, TD-trained gap vs rank), per-panel files
`fig_toy_ablation_{closed,trained,mass,E_calibrated,legend}.{pdf,png}`.
Nothing committed. The corridor / FourRooms version of this ablation was not built
(the coordinator moved the primary environment to the didactic MDP before it was started).

## The MDP and the metric

The nine-state process of `scripts/didactic/graph_bottleneck_td.py` at leak 0 (the one
behind `fig_didactic_mdp` / `fig_didactic_values` and Section 4.1): states 0-3 and 5-8
are two groups joined only through state 4, no edge back; two actions per state; goal =
state 8; the decision read is at state 7, where action 0 enters the goal and action 1
returns to the group. Discount 0.8, fixed uniform continuation policy (each action with
probability 1/2), exact transitions. Candidate states carry the uniform reference weight
1/9 (the didactic convention; the stationary distribution of the uniform policy puts all
its mass on states 5-8, so it is not used). Reward 1 on arrival at state 8.

Exact facts (`exact.json`): rows of (1-gamma)M sum to 1 (2e-16); the exact values at
state 7 are 2.20 (action 0) and 0.80 (action 1), gap 1.40; the two leading right
singular directions of the exact matrix hold 0.880 of its squared mass and 0.9987 of the
squared norm of the state-7 row difference lies outside them (the paper's `failure.tex`
says 88% / 96%; the 88% reproduces, the 96% does not -- with this projection the number
is 99.9%).

Metric: fraction of the exact action gap preserved, (Q(7, a0) - Q(7, a1)) / 1.40, where
each arm's Q is its own value estimate for the goal reward. "argmax correct" = the fit
still prefers action 0. "retained squared mass" = 1 - ||P_hat - (1-gamma)M||_F^2 /
||(1-gamma)M||_F^2 with P_hat the arm's implied normalized measure (for the linear arms,
(1-gamma) F B^T / 9 resp. F B^T; for the softmax arms, the softmax itself).

## Arms

Same MDP, same rank d, same policy; d = 2, ..., 9.

- **A, FB.** Linear F B^T fits the ratio M/rho = 9M with squared error; value read as
  F z with z = E_rho[B r] = (1/9) B^T r.
- **B, FB with the normalized target.** F B^T fits (1-gamma)M with squared error; same
  readout, multiplied by the known constant 9/(1-gamma) so it estimates Q.
- **C, softmax + squared error.** p = softmax(F B^T) over all 9 states fitted to
  (1-gamma)M with 0.5 sum_j (p_j - t_j)^2; value = sum_j p_j r_j / (1-gamma).
- **D, Density-FB.** Same model, cross-entropy, same integration readout.
- **E, D's model with FB's readout.** D's F and B, value read as F z, z = (1/9) B^T r.
  Its scale is arbitrary (logits are defined up to a per-row shift and a scale), so the
  raw fraction is reported as specified and, as the most favourable reading, a
  scale-calibrated version (one scalar chosen by least squares against the exact Q over
  all 18 state-action rows).
- **F, D with a soft immediate target.** The one-hot immediate term is replaced by a
  kernel over graph distance, exp(-dist^2 / (2 * 0.5^2)) row-normalized (self weight at
  the goal 0.787), so the fixed point is (1-gamma) M K. Read against the exact
  (unsmoothed) gap, this target caps the fraction at 0.766 even at full rank.

Closed form: A and B are truncated SVDs (exact optimum; B is the SVD of A's matrix
scaled, so the two lines coincide by construction). C, D, F: Adam on the population
objective against the exact target, lr 1e-2, 10k steps, 5 seeds, single init each.
Trained: the didactic TD harness `scripts/didactic/bottleneck_family_td.py` (exact
population expectations, bootstrapped target network with EMA tau 0.005, Adam, 50k
steps, learning rates {0.001, 0.01}, orthogonality penalty at coefficient 1 on the two
linear arms as shipped, unrestricted B), methods fb / normsq / softsq / density plus a
new density_soft, 5 seeds (0-4). E is read from the trained density parameters. Trained
numbers are the mean over 5 seeds x 2 learning rates (10 runs), as in the paper's
Section 4.1 numbers; per-rate means are in `run_log.txt`; 95% t-intervals over the 10
runs.

## Results: fraction of the exact gap at state 7

Closed form (C-F: mean of 5 seeds; seed spread < 0.01 for C, D, F at every rank):

| arm | d=2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| A FB | -0.006 | 0.975 | 1.069 | 1.039 | 1.038 | 1.002 | 1.000 | 1.000 |
| B FB, normalized target | -0.006 | 0.975 | 1.069 | 1.039 | 1.038 | 1.002 | 1.000 | 1.000 |
| C softmax + squared | 1.070 | 1.034 | 0.999 | 1.000 | 1.001 | 1.000 | 1.000 | 1.000 |
| D Density-FB | 1.042 | 1.004 | 1.000 | 1.001 | 1.001 | 1.000 | 0.999 | 1.000 |
| E D, linear readout (raw) | 0.387 | 0.390 | 0.018 | -0.039 | 0.071 | 0.048 | 0.072 | 0.079 |
| E, scale-calibrated | 0.012 | 0.036 | 0.024 | -0.021 | 0.075 | 0.052 | 0.084 | 0.094 |
| F soft target | 0.765 | 0.780 | 0.766 | 0.766 | 0.765 | 0.765 | 0.766 | 0.766 |

TD-trained (mean of 10 runs, 95% t-interval):

| arm | d=2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| A | 0.01 [0.01, 0.01] | 0.05 [-0.02, 0.12] | 0.48 [0.26, 0.71] | 0.95 [0.87, 1.02] | 0.99 [0.95, 1.03] | 0.98 [0.97, 1.00] | 0.99 [0.98, 1.00] | 1.00 |
| B | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.52 [0.23, 0.80] | 1.07 [1.07, 1.07] | 1.07 [1.07, 1.07] | 1.00 | 1.00 | 1.00 |
| C | 0.95 [0.79, 1.12] | 1.00 [0.97, 1.03] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| D | 0.53 [0.50, 0.56] | 0.99 [0.99, 0.99] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| E (raw) | 1.75 [1.50, 2.00] | -0.43 [-0.58, -0.28] | -0.42 [-0.92, 0.09] | 0.01 [-0.21, 0.22] | 0.03 [-0.16, 0.23] | 0.07 [0.02, 0.12] | 0.07 [0.06, 0.09] | 0.07 [0.06, 0.09] |
| E, calibrated | 1.008 | -0.170 | -0.096 | 0.025 | 0.065 | 0.105 | 0.114 | 0.115 |
| F | 0.66 | 0.77 | 0.76 | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 |

Per learning rate (5 seeds each): A at d=3 is 0.013 (lr 0.001) / 0.088 (lr 0.01), at
d=4 0.533 / 0.435; D at d=2 is 0.553 / 0.507; C at d=2 is 0.913 / 0.995; everything
else agrees across the two rates to within 0.02.

Argmax stays on action 0: every closed-form and trained fit of A, B, C, D, F except the
closed-form A/B at d=2 (gap -0.006, wrong action). E: closed form 4/5 seeds at d=2-4 and
6-7, 1/5 at d=5, 5/5 at d=8-9; trained 10/10 at d=2, 0/10 at d=3, 5/10 at d=4, 3/10 at
d=5, 6/10 at d=6, 9/10 at d=7, 10/10 at d=8-9.

Retained squared mass (closed / trained):

| arm | d=2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| A = B | .880 / .844 | .918 / .864 (.875) | .951 / .911 (.914) | .966 / .940 (.946) | .978 / .954 (.959) | .987 / .973 | .994 / .987 (.984) | 1.000 |
| C | .973 / .940 | .996 / .994 | 1.000 / 1.000 | 1 | 1 | 1 | 1 | 1 |
| D = E | .970 / .927 | .994 / .983 | .999 / 1.000 | 1 | 1 | 1 | 1 | 1 |
| F (vs the exact target) | .948 / .941 | .973 / .973 | .983 / .984 | .984 | .984 | .984 | .984 | .984 |

(B's trained values in parentheses where they differ from A's.) Every fit keeps at
least 84% of the squared mass at rank 2 and 86% at rank 3; the gap is lost or kept
independently of that.

Summary table at rank 3 (the paper's rank): arm | closed-form gap | trained gap | retained mass (closed / trained)

| A | 0.975 | 0.05 | .918 / .864 |
| B | 0.975 | 0.01 | .918 / .875 |
| C | 1.034 | 1.00 | .996 / .994 |
| D | 1.004 | 0.99 | .994 / .983 |
| E | 0.390 raw, 0.036 calibrated | -0.43 raw, -0.17 calibrated | .994 / .983 |
| F | 0.780 | 0.77 | .973 / .973 |

## Diagnostic: why trained FB loses the gap at ranks 3-4 when its SVD optimum keeps it

`diag_fb_rank3.py`: FB's loss, ranks 2-4, 10 runs each, 2x2 over {bootstrapped TD
target, exact supervised target} x {orthogonality coefficient 1, 0}:

| d | bootstrap, ortho 1 | bootstrap, ortho 0 | exact target, ortho 1 | exact target, ortho 0 | SVD |
|---|---|---|---|---|---|
| 2 | 0.013 | 0.013 | -0.006 | -0.006 | -0.006 |
| 3 | 0.051 [-0.02, 0.12] | 0.131 [0.01, 0.25] | 0.975 | 0.975 | 0.975 |
| 4 | 0.484 [0.26, 0.71] | 0.529 [0.26, 0.80] | 1.069 | 1.069 | 1.069 |

With the exact target Adam reaches the SVD optimum to three decimals with or without the
penalty. The loss at ranks 3-4 comes from the bootstrapped target: the semi-gradient TD
fixed point of the rank-limited linear model is not the projection of the exact measure.

## Reading

- **A = B.** Normalizing the target changes nothing: identical by construction in closed
  form and indistinguishable under TD (B overshoots to 1.07 at d = 5-6 with a tight
  interval, A sits at 0.95-0.99; both are at 0.01 for d <= 3 and about 0.5 at d = 4).
- **C = D in closed form.** Both keep the whole gap at every rank, including rank 2,
  where the squared-loss linear fit flips the action. On this MDP the output class
  (softmax over states, unit row mass) is what preserves the decision; the loss on top of
  it does not matter at the population optimum. This agrees with the mechanism note of
  2026-09-11 ("not a CE-only result").
- **C and D separate slightly under training, in C's favour at rank 2:** softmax +
  squared error 0.95 [0.79, 1.12] vs cross-entropy 0.53 [0.50, 0.56]; equal from rank 3.
  The TD fixed point hurts D at rank 2 the way it hurts A at ranks 3-4. Cross-entropy is
  therefore not what buys the rank-2 result here; the paper's "55% at rank 2" is the
  cross-entropy arm's TD number, and the squared-error softmax arm does better on it.
- **E fails.** FB's linear readout applied to the softmax model's F and B returns a
  number unrelated to the value: raw fractions 0.39 -> 0.08 in closed form, 1.75 / -0.43 /
  -0.42 at ranks 2-4 under TD, and even after the best global rescaling at most 0.12 at
  full rank with the wrong action in 0-7 of 10 runs at ranks 3-6. Once the model is a
  softmax, the value has to be read by integrating the predicted distribution against
  the reward; the third change is forced by the first two.
- **F is a constant bias, not a rank effect.** The soft immediate target reads 0.766 of
  the gap at every rank from 3 up, exactly the value its own smoothed target gives at
  full rank; at rank 2 it is 0.66 trained.
- **Retained mass does not predict the gap.** At rank 3 every arm keeps >= 86% of the
  squared mass; A/B keep 5% of the gap under TD and C/D keep 99-100%.

## Caveats

1. **The paper's rank-3 FB failure is a TD-training result, not a population-optimum
   result.** FB's exact rank-3 squared-loss optimum keeps 97.5% of the gap on this MDP
   (rank 4: 107%); only at rank 2 does the SVD lose it (-0.6%, wrong action). Section 4.1's
   "1%-2% at ranks 2 and 3, rank 6 to recover 99%" is reproduced by the TD harness
   (0.013 / 0.051 / 0.484 / 0.949 / 0.990 at d = 2..6), and the diagnostic above shows
   it is the bootstrapped target, not the orthogonality penalty, that causes it. The
   closed-form claim in `failure.tex` (leading directions vs the row difference) only
   licenses the rank-2 statement. Any sentence that attributes the rank-3 failure to the
   squared-loss optimum should be reworded to the TD fixed point.
2. The "96% of the difference outside the two leading directions" in `failure.tex` was
   not reproduced (99.9% with a right-singular-vector projection of the state-7 row
   difference of the normalized matrix); whoever computed 96% used a different
   projection.
3. Closed-form C, D, F are Adam fits, not certified global optima; five single-init
   seeds agree to < 0.01 at every rank, and D at full rank reaches 0.9989.
4. E's raw scale is arbitrary; the calibrated column uses the exact Q (an oracle) and is
   the most favourable reading. The seed band for E is omitted in the closed-form panel
   for that reason.
5. Trained means pool two learning rates (the harness's convention and the paper's);
   the per-rate numbers are in `run_log.txt` and differ by more than 0.05 only for A at
   d = 3-4, C at d = 2 and D at d = 2.
6. Nine states, one policy, no actor, no reward encoder, no minibatch candidate pool:
   the same scope limits as the didactic section.
7. Off-scale point in the trained panel: E at rank 2 (1.75) is labelled at the top
   edge.

## Reproduce

```bash
cd /home/adityamohan/git/Factored-FB-seam
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/toy_ablation/run.py            # ~2 min, CPU
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/toy_ablation/run.py --figures-only
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/toy_ablation/diag_fb_rank3.py   # ~30 s
```
(run in tmux session `toyabl`; jax 0.10.0, optax 0.2.8, numpy 2.4.6, float64.)
