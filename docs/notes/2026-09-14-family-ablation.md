# 2026-09-14 -- The toy ablation on programmatically generated bottleneck MDPs

Code: `analysis/toy_ablation/family.py` (the single-process version is `run.py`;
this script reuses its arm colours, interval helper and axis style). Data
`family_closed.csv`, `family_trained.csv`, `family_mdps.json`, `family_log.txt`
in the same folder. Figures in `PAPER/ICLR/figures/bottleneck_family/`:
`fig_family_ablation.{pdf,png}` (closed form and TD-trained, aggregated over the
ten processes), `fig_family_ablation_{closed,trained}`, `fig_family_mass`,
`fig_family_per_mdp` (one panel per process, two rows), and
`fig_family_graph_{1..10}` (the sampled graphs). Nothing committed.

Purpose: the ablation of Section 4.1 ran on one hand-drawn nine-state process,
which invites "you drew the graph until FB broke". This runs the same six arms on
processes drawn at random, with a different action probability at every state, and
aggregates.

**SHIPPED FAMILY (user decision, 2026-09-14 evening): the five single-cut
processes.** `PROCESSES` at the top of the script lists them. The generator also
builds two-cut layouts (`series`, `parallel`); those were measured and are
reported in the "two-cut exploration" section at the end of this note, but they
are NOT in the paper, because at nine states a second cut forces the group sizes,
so layout and group size cannot be varied independently.

What Section 4.1 now shows, all on the five generated processes:
1. `fig_family_values` -- action values at the decision state at rank 3, each
   process mapped so its exact pair is (1, 0): exact 1.00 / 0.00, FB 0.15 / 0.03,
   Density-FB 0.96 / 0.05, 95% intervals over processes and runs.
2. `fig_family_ablation_trained` -- the six-arm rank sweep, TD-trained only,
   lines with shaded intervals.
3. `fig_rare_bottleneck_lemma` (new, single panel from
   `analysis/rare_bottleneck/run.py`) -- the lemma: as the bottleneck rows' share
   of the squared mass falls (dotted, right axis) FB's reconstructed gap falls
   with it while the exact gap and both softmax arms stay at 1.
The population-optimum panel, the retained-mass panel, the per-process panels,
the rank table and the other two rare-bottleneck panels moved to the appendix
(`PAPER/ICLR/content/appendix/bottleneck-family.tex`).

## The shipped family: five single-cut processes (the numbers the paper quotes)

Values at the decision state at rank 3, each process mapped so its exact pair is
(1, 0), mean over 5 processes x 5 seeds x 2 rates: exact 1.00 / 0.00,
FB **0.15 / 0.03**, Density-FB **0.96 / 0.05**.

Fraction of the exact action gap, TD-trained:

| arm | d=2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| A FB | 0.05 | 0.12 | 0.38 | 0.70 | 0.86 | 0.97 |
| B FB, normalized target | 0.05 | 0.05 | 0.28 | 0.75 | 0.94 | 0.99 |
| C softmax + squared | 0.82 | 1.03 | 1.01 | 1.00 | 1.00 | 1.00 |
| D Density-FB | 0.59 | 0.91 | 0.98 | 1.00 | 1.00 | 1.00 |
| F soft target | 0.47 | 0.65 | 0.69 | 0.71 | 0.71 | 0.71 |

Closed form (population optimum): A = B 0.088 / 0.216 / 0.491 / 0.925 at ranks
2 / 3 / 4 / 5; C 0.923 / 1.033; D 0.849 / 0.991; F 0.558 -> 0.708.

Smallest rank reaching 90% of the gap, per process: FB 5, 8, 6, 7, 7 (median 7);
normalized target 5, 6, 5, 7, 6 (6); softmax + squared 2, 2, 3, 3, 2 (2);
Density-FB 3, 2, 4, 5, 3 (3). Argmax on the goal action at ranks 2 and 3: FB 0.20
and 0.28 of runs, Density-FB 0.96 and 1.00. Retained squared mass at rank 3 is at
least 0.86 for every arm.

## The generator

`sample_mdp(seed, variant)`, nine states in every process so that the rank axis
means the same thing in all of them. Common to all variants: two actions per
state, each leading deterministically to one state; inside a group, action 0
follows a random ordering of that group and action 1 goes to a random other state
of the group; the group holding the goal is closed, its ordering ends at the
goal, and the state before it is the decision state (action 0 into the goal,
action 1 elsewhere in the group); p(action 0) is drawn per state from
U(0.25, 0.75), so the policy is NOT uniform and differs across processes; a draw
is kept only if the exact action gap at the decision state is >= 0.5.

- **single** (seeds 0-4): `[A][cut][B]`, group sizes 3-5 / 3-5. The cut's action 0
  enters B, action 1 falls back into A. The only edge into B is the cut's action 0.
- **series** (seeds 5-7): `[G0][cut1][G1][cut2][G2]`, two cuts one after the other.
  At nine states the sizes are forced to 2/2/3. Both cuts lie on the only path:
  the only edge into G1 or cut2 from outside is cut1's action 0, and the only edge
  into G2 is cut2's action 0.
- **parallel** (seeds 8-9): `[A][cut1][cut2][B]`, group sizes 2-4 / 3-5. Both cuts
  are reachable from A and both enter B, so the passage is redundant -- removing
  either still leaves a way in.

These structural facts are asserted in `check()` for every sampled process.
Everything else matches `run.py`: discount 0.8, reward 1 at the goal, uniform
reference weight 1/9, exact population expectations, ranks 2-9, the same six arms
A-F, closed-form fits (SVD for A and B; Adam 10k steps lr 1e-2, 5 seeds for
C, D, F) and TD-trained fits (bootstrapped target, EMA tau 0.005, 50k steps,
5 seeds x 2 learning rates, float64). Total wall time 932 s on CPU.

## Two-cut exploration: the ten processes measured on 2026-09-14

| # | variant | groups | cuts | decision | goal | exact values | gap |
|---|---|---|---|---|---|---|---|
| 1 | single | 3/5 | 3 | 6 | 7 | 1.72 / 0.50 | 1.223 |
| 2 | single | 5/3 | 5 | 7 | 6 | 3.14 / 2.33 | 0.811 |
| 3 | single | 4/4 | 4 | 8 | 6 | 1.64 / 0.79 | 0.855 |
| 4 | single | 3/5 | 3 | 4 | 6 | 1.95 / 0.88 | 1.072 |
| 5 | single | 5/3 | 5 | 6 | 7 | 1.94 / 1.17 | 0.771 |
| 6 | series | 2/2/3 | 2, 5 | 8 | 6 | 3.19 / 2.12 | 1.071 |
| 7 | series | 2/2/3 | 2, 5 | 6 | 7 | 2.65 / 1.82 | 0.828 |
| 8 | series | 2/2/3 | 2, 5 | 7 | 8 | 2.28 / 1.56 | 0.712 |
| 9 | parallel | 3/4 | 3, 4 | 8 | 5 | 1.94 / 1.28 | 0.661 |
| 10 | parallel | 4/3 | 4, 5 | 6 | 7 | 2.21 / 1.49 | 0.714 |

All ten passed on the first draw; p(action 0) spans roughly 0.25-0.75 in each.

## Fraction of the exact action gap, mean over all ten (five single + five two-cut)

Closed form (population optimum):

| arm | d=2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| A FB | 0.108 | 0.182 | 0.563 | 0.922 | 0.964 | 0.976 | 0.996 | 1.000 |
| B FB, normalized target | 0.108 | 0.182 | 0.563 | 0.922 | 0.964 | 0.976 | 0.996 | 1.000 |
| C softmax + squared | 0.911 | 1.012 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| D Density-FB | 0.821 | 0.986 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E D, linear readout | 2.870 | 0.173 | 0.152 | 0.040 | 0.092 | 0.066 | 0.072 | 0.068 |
| F soft target | 0.415 | 0.607 | 0.663 | 0.675 | 0.676 | 0.676 | 0.676 | 0.677 |

TD-trained (mean, 95% t-interval over 10 processes x 5 seeds x 2 rates = 100 runs):

| arm | d=2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| A | 0.05 [0.02, 0.08] | 0.21 [0.14, 0.28] | 0.55 [0.47, 0.64] | 0.77 [0.71, 0.84] | 0.92 [0.88, 0.96] | 0.98 [0.96, 0.99] |
| B | 0.05 [0.02, 0.08] | 0.16 [0.10, 0.22] | 0.48 [0.40, 0.57] | 0.80 [0.75, 0.85] | 0.95 [0.93, 0.96] | 0.97 [0.97, 0.98] |
| C | 0.90 [0.85, 0.95] | 1.01 [1.00, 1.03] | 1.00 | 1.00 | 1.00 | 1.00 |
| D | 0.51 [0.42, 0.59] | 0.92 [0.90, 0.95] | 0.99 [0.98, 1.00] | 1.00 | 1.00 | 1.00 |
| E | 0.54 [-0.15, 1.23] | 0.70 [0.19, 1.21] | 0.31 [-0.02, 0.64] | 0.18 | 0.14 | 0.11 |
| F | 0.39 [0.33, 0.44] | 0.59 [0.55, 0.63] | 0.66 [0.64, 0.68] | 0.68 | 0.68 | 0.68 |

Split by layout (TD-trained, ranks 2 / 3 / 4):

| arm | single (5) | series (3) | parallel (2) |
|---|---|---|---|
| A FB | 0.05 / 0.12 / 0.38 | 0.11 / 0.12 / 0.64 | -0.01 / 0.58 / 0.86 |
| B | 0.05 / 0.05 / 0.28 | 0.11 / 0.12 / 0.59 | -0.01 / 0.49 / 0.85 |
| C | 0.82 / 1.03 / 1.01 | 1.02 / 1.00 / 1.00 | 0.93 / 0.99 / 1.00 |
| D | 0.59 / 0.91 / 0.98 | 0.54 / 0.99 / 1.00 | 0.25 / 0.86 / 1.00 |
| F | 0.47 / 0.65 / 0.69 | 0.30 / 0.68 / 0.68 | 0.31 / 0.33 / 0.56 |

Smallest rank reaching 90% of the gap (TD-trained):

| arm | s1 | s2 | s3 | s4 | s5 | ser6 | ser7 | ser8 | par9 | par10 | median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A FB | 5 | 8 | 6 | 7 | 7 | 4 | 4 | 6 | 6 | 4 | **6** |
| B | 5 | 6 | 5 | 7 | 6 | 4 | 7 | 6 | 6 | 4 | 6 |
| C softmax + squared | 2 | 2 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | **2** |
| D Density-FB | 3 | 2 | 4 | 5 | 3 | 3 | 3 | 3 | 4 | 3 | **3** |
| E | - | 2 | - | 3 | - | 2 | 2 | 3 | 2 | 2 | 2 |
| F | never | | | | | | | | | | - |

Fraction of TD fits ranking the goal action first: A 0.40 / 0.62 / 0.80 at ranks
2 / 3 / 4, B 0.40 / 0.54 / 0.70, D 0.89 / 1.00 / 1.00, C 1.00 everywhere.

Retained squared mass, mean over processes: A/B 0.88 || 0.82 (closed || trained)
at rank 2, 0.95 || 0.89 at rank 3; C/D 0.98 || 0.95 at rank 2. Every arm keeps at
least 82% of the squared mass at every rank while FB keeps 5-21% of the gap.

## Reading

- **All four single-process conclusions reproduce on the larger family.**
  Normalization is inert (A = B in closed form, within noise under TD); both
  softmax arms are far above FB at every rank and agree from rank 4, so the
  parameterization and not the loss is the mechanism; the linear readout fails;
  the soft target is a flat ceiling (0.68). Median rank to recover the decision:
  FB 6, Density-FB 3, softmax + squared error 2.
- **FB's population optimum also loses the gap** (0.11 / 0.18 / 0.56 at ranks
  2 / 3 / 4), as on the five-process family and unlike the hand-drawn process,
  where the optimum kept 97.5% at rank 3 and only TD lost it.
- **No clean effect of the NUMBER of bottlenecks, and the sign is opposite to the
  prediction.** FB does *better* with two cuts than with one: median required
  rank 4 / 4 / 6 on the series processes and 6 / 4 on the parallel ones, against
  5 / 8 / 6 / 7 / 7 on the single-cut ones. The prediction had been that two cuts
  in series make FB worse (two rare states to keep) and parallel cuts make it
  better (redundant passage).
- **That comparison is confounded and should not be reported as a result.** At
  nine states a second cut forces smaller groups (series is 2/2/3 against 3-5/3-5
  for single), so layout and group size move together, and the group holding the
  goal is what sets how rare the decision structure is. Testing the count of
  bottlenecks needs group sizes held fixed while the count varies, which needs
  more states: 1/2/3 cuts in series at n = 13, five processes each, one rank axis.
  Until that runs, the family's role is diversity of layout, not a claim about
  bottleneck count.
- Density-FB at rank 2 is weakest on the parallel processes (0.25) and FB is
  weakest at rank 3-4 on single and series; per-process spread is large enough
  that only the rank-4-and-up statements are safe per process.

## Caveats

1. Nine states, one fixed policy, no actor, no reward encoder, no sampling --
   the same scope limits as the single-process ablation.
2. Fixed n = 9 so rank is comparable; this forces the series layout to 2/2/3.
3. Closed-form C, D, F are Adam fits from one init per seed, not certified
   global optima.
4. E's raw scale is arbitrary; the calibrated column is in the CSV, not plotted.
5. Ten processes here (5 single, 3 series, 2 parallel). The list at the top of
   the script is `PROCESSES`; adding seeds is a one-line change (~90 s per
   process on CPU).

## Reproduce

```bash
cd /home/adityamohan/git/Factored-FB-seam
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/toy_ablation/family.py             # ~16 min, CPU
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/toy_ablation/family.py --figures-only
```
(tmux session `famabl`; jax 0.10.0, optax 0.2.8, numpy 2.4.6, float64.)

## Also changed

`analysis/toy_ablation/run.py`: the TD-trained panel is now drawn in the same
line-and-band style as the closed-form panel (hollow markers keep marking
TD-trained), instead of offset markers with error bars; points outside the axis
are dropped from the line and labelled at the edge. Figures regenerated.
