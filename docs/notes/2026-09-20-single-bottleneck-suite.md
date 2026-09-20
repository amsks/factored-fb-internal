# 2026-09-20 -- The bottleneck suite rebuilt on the single-bottleneck process family

Code `analysis/single_bottleneck/run.py`; data `analysis/single_bottleneck/sb_{closed,trained,
rare_closed,rare_trained,lemma,reward,construct,mitigation}.csv`, `sb_processes.json`,
`sb_summary.json`, `sb_log.txt` (the full tables), `stdout.txt`. Figures in
`PAPER/ICLR/figures/single_bottleneck/`: `fig_sb_graph_{P6,P11,R1..R5}`, `fig_sb_ablation`
(+ `_closed`, `_trained`), `fig_sb_gap_two_arm`, `fig_sb_per_mdp`, `fig_sb_values_d{2,3}`,
`fig_sb_rare_lemma_d{2,3}`, `fig_sb_lemma_check` (+ `_a`, `_b`, `_c`), `fig_sb_readout_swap`,
`fig_sb_margin`, `fig_gap_direction` (from `analysis/single_bottleneck/gap_direction.py`, data
`gap_direction.json`). Nothing committed (`analysis/` is gitignored).

Every fit, readout, metric, trainer and figure style is the existing code, called on the new
processes: the six arms A-F and the TD harness from `analysis/toy_ablation/family.py`, the
row-reweighting sweep from `analysis/rare_bottleneck/run.py`, the projection lemma from
`analysis/lemma_check/run.py`. Those three scripts were only given optional parameters (rank
list, decision rows, weight list, output folder, graph layout); their defaults and outputs are
unchanged (`--figures-only` reruns of both, and `tests/probes/test_lemma_check.py`, still pass).
The old suite's notes are `docs/notes/2026-09-14-family-ablation.md`,
`2026-09-14-rare-bottleneck.md`, `2026-09-20-lemma-check.md`; this note mirrors them.

## The process family

`analysis/lemma_check/design_single_bottleneck.py::build(entry, cluster, tail, merge, mutual)`:
an entry chain of `entry` states (action 0 advances, action 1 steps back, the first state loops
on itself), ONE bottleneck state b whose action 0 enters cluster A and action 1 enters cluster B
(each a cycle of `cluster` states), and either a mutual leak (tail 0: every cluster state's
second action jumps into the other cluster's first state) or a shared closed tail of `tail`
states every cluster state's second action exits into. Deterministic edges, two actions per
state, discount 0.8, uniform reference weight 1/n.

The decision is at b. The reward is the indicator of cluster A: there is no goal state. The
quantity measured everywhere is the fraction of the exact action gap at b that a fit reads,
(Q_hat(b, a0) - Q_hat(b, a1)) / (Q(b, a0) - Q(b, a1)), with Q = M r and M the exact successor
measure (2n rows, one per state-action pair). The closed-form FB fit at rank d reads exactly
dv^T P_U r / n with dv the difference of the two bottleneck rows of X = nM and P_U the
projection onto span(B) (asserted against arm A's readout to 1e-9 at every rank).

| process | spec | n | policy | exact values at b (a0 / a1) | gap | FB closed form d=2 / 3 / 4 |
|---|---|---|---|---|---|---|
| P6 | entry 1, cluster 2, mutual leak | 6 | uniform | 3.000 / 2.000 | 1.000 | 0.000 / 1.044 / 1.044 |
| P11 | entry 2, cluster 2, tail 4 | 11 | uniform | 1.667 / 0.000 | 1.667 | -0.000 / -0.000 / 1.043 |
| R1 | entry 1, cluster 2, mutual leak | 6 | U(.25,.75), p(a0) in [.42,.75] | 2.487 / 1.338 | 1.149 | 0.386 / 1.018 / 1.006 |
| R2 | entry 3, cluster 3, mutual leak | 10 | p(a0) in [.27,.72] | 2.850 / 1.935 | 0.915 | 0.023 / 1.053 / 1.129 |
| R3 | entry 3, cluster 2, tail 4 | 12 | p(a0) in [.26,.75] | 1.398 / 0.000 | 1.398 | -0.007 / -0.108 / 0.804 |
| R4 | entry 3, cluster 3, tail 2 | 12 | p(a0) in [.28,.71] | 1.698 / 0.000 | 1.698 | -0.003 / 0.423 / 1.094 |
| R5 | entry 2, cluster 3, tail 2 | 11 | p(a0) in [.33,.72] | 1.571 / 0.000 | 1.571 | 0.015 / 1.280 / 1.122 |

P6 is the lemma figure's process (`build(entry=1, cluster=2, tail=0, mutual=True)`), P11 the
rank-sweep process (`build(entry=2, cluster=2, tail=4, merge=True)`), both with the uniform
policy. R1-R5 are drawn from the template entry in {1, 2, 3}, cluster in {2, 3}, tail in {0, 2,
4} (tail 0 = mutual leak, else shared tail) with the probability of action 0 drawn per state from
U(0.25, 0.75), one generator per draw (`default_rng(2000 + k)`); a draw is kept iff the exact gap
is > 0.5 and FB's closed-form rank-2 fit reads < 0.5 of it, so the failure is present by
construction. Only one candidate was rejected in the whole sampling (before R5: entry 1,
cluster 3, tail 2, gap 1.35, FB rank-2 fraction 0.72), i.e. the template produces the failure
almost always. R1 has P6's layout with a random policy. Exact facts asserted for every process:
rows of (1-gamma)M sum to 1, b is the only entry into A u B u tail, the tail is closed / the
leaks go where the builder says, P6's values are exactly 3 and 2, P11's are exactly 5/3 and 0.

Structure of the exact values. With a shared tail the two clusters never meet, so Q(b, a1) = 0
exactly and the gap is Q(b, a0) alone (P11, R3, R4, R5); with the mutual leak both actions
eventually visit A and the gap is the difference of two positive numbers (P6, R1, R2). In every
process the gap sits entirely in the two bottleneck rows: no other row of M distinguishes the
two futures at the first step.

## 1. Rank sweep and the six-arm ablation (P11 + R1-R5)

Setup as in `family.py`: ranks 2..n per process; closed form = truncated SVD of nM (A), of
(1-gamma)M (B), Adam on the population objective for the softmax arms (10k steps, lr 1e-2, 5
seeds); TD-trained = bootstrapped target network, EMA 0.005, Adam, 50k steps, 5 seeds x 2
learning rates {0.001, 0.01} = 10 runs per cell, orthogonality penalty at 1 on FB as shipped.
Aggregates are over the ranks all six processes share (2..6, R1 has 6 states); per-process
panels go up to each process's own n.

Fraction of the exact gap, mean over the six processes (closed form; TD-trained mean and 95%
t-interval over 60 runs), ranks 2..6:

| arm | closed d=2 | 3 | 4 | 5 | 6 | trained d=2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|---|---|
| A FB | 0.069 | 0.611 | 1.033 | 1.039 | 1.030 | 0.11 [0.05, 0.18] | 0.38 [0.29, 0.48] | 0.63 [0.53, 0.74] | 0.92 [0.88, 0.96] | 0.95 [0.91, 0.98] |
| B FB, normalized target | 0.069 | 0.611 | 1.033 | 1.039 | 1.030 | 0.11 | 0.40 | 0.74 | 0.96 | 0.98 |
| C softmax + squared | 0.887 | 0.965 | 1.003 | 0.998 | 0.999 | 0.78 [0.71, 0.85] | 0.99 | 1.00 | 1.00 | 1.00 |
| D Density-FB | 0.880 | 0.963 | 0.988 | 0.995 | 1.000 | 0.70 [0.62, 0.78] | 0.94 [0.91, 0.97] | 1.00 | 1.00 | 1.00 |
| E D's model, linear readout | 26.5 | 2.24 | 3.71 | 2.96 | 1.70 | 1.16 | 2.74 | 1.79 | 1.80 | 1.29 |
| F D, soft immediate target | 0.625 | 0.718 | 0.747 | 0.759 | 0.760 | 0.65 | 0.74 | 0.76 | 0.76 | 0.76 |

Smallest rank at which the mean fraction reaches 0.9, per process (TD-trained / closed form):

| arm | P11 | R1 | R2 | R3 | R4 | R5 | median |
|---|---|---|---|---|---|---|---|
| A FB | 5 / 4 | 6 / 3 | 8 / 3 | 5 / 5 | 4 / 4 | 3 / 3 | **5.0** / 3.5 |
| B | 5 / 4 | 5 / 3 | 8 / 3 | 5 / 5 | 4 / 4 | 3 / 3 | 5.0 / 3.5 |
| C | 3 / 2 | 3 / 2 | 3 / 3 | 3 / 3 | 2 / 2 | 2 / 2 | **3.0** / 2.0 |
| D Density-FB | 3 / 2 | 4 / 2 | 4 / 4 | 3 / 2 | 2 / 3 | 3 / 3 | **3.0** / 2.5 |
| E | 2 / 2 | - / - | 3 / 2 | 2 / 2 | 2 / 2 | 3 / 2 | 2.0 / 2.0 (raw scale arbitrary; sign only) |
| F | - | - | - | - | - | - | never (plateaus at 0.58-0.85) |

Per-process TD-trained FB (arm A) up to each process's n: P11 0.00 / 0.00 / 0.41 / 0.96 / ... /
1.00; R1 0.66 / 0.70 / 0.72 / 0.88 / 1.00; R2 0.04 / 0.62 / 0.62 / 0.64 / 0.65 / 0.87 / 0.98 /
1.01 / 1.00; R3 0.01 / 0.01 / -0.02 / 1.02 / ...; R4 0.00 / 0.04 / 0.98 / 0.92 / ...; R5 -0.02 /
0.95 / 1.08 / .... Density-FB (D): P11 0.68 / 1.00 / ...; R1 0.62 / 0.88 / 1.00; R2 0.23 / 0.87
/ 1.00; R3 0.86 / 1.00; R4 0.92 / 0.99; R5 0.89 / 0.92 / 1.00 (all rows in `sb_log.txt`).
The argmax stays on cluster A in 53% / 85% / 82% / 100% / 100% of the trained FB fits at ranks
2..6, in 100% of the C and D fits at every rank.

Values at b, TD-trained, exact pair mapped to (1, 0) (`fig_sb_values_d{2,3}`): rank 2 FB
0.48 / 0.36, Density-FB 0.88 / 0.19; rank 3 FB 0.63 / 0.24, Density-FB 1.00 / 0.06 (the old
family's rank-3 figure read FB 0.15 / 0.03, Density-FB 0.96 / 0.05).

Two things differ from the nine-state family. First, the closed-form FB curve here is a step:
0.07 at rank 2, 0.61 at rank 3, 1.03 from rank 4 on, because on every process the bottleneck
row difference sits almost entirely in ONE singular direction of X (P11: 94.5% of |dv|^2 on
v_4, which carries 3.3% of the matrix's squared mass, `fig_gap_direction`; P6: 96% on v_3), so
the gap appears in full the moment that direction enters the span and not at all before.
Second, the TD-trained FB lands consistently below its own closed-form optimum (0.38 vs 0.61 at
rank 3, 0.63 vs 1.03 at rank 4; the same runs all read 1.00 at full rank), and every run of a
cell converges to the same value (P11 rank 2: ten runs at 0.000; R1: ten at 0.662; R2: ten at
0.035): the TD fixed point of the FB loss with the shipped orthonormality penalty is not the
least-squares projection, and at ranks 3-4 it is a worse one. The softmax arms C and D are at
0.94-0.99 from rank 3 on in both regimes; arm E confirms as before that D's features with a
linear readout are not a value (26.5 at rank 2, 1.7-3.7 elsewhere); arm F's soft immediate target
caps the fraction at 0.76 here (0.96-1.00 on the old family), because the smoothing kernel of
width 0.5 graph edges mixes cluster A with the tail / cluster B states it is one edge from.

## 2. Rarity sweep on P6 (row reweighting)

`rare_bottleneck/run.py`'s construction (a) with the two bottleneck rows (state 1, rows 2 and 3)
carrying total weight w and the other 10 rows sharing 1-w; w in {1/2, 1/4, 1/6 (= uniform, 1/n
with n = 6), 1/8, 1/16, 1/32, 1/64, 1/128, 0}. Arms A (weighted SVD; A+ = the gap-keeping fit
and the w -> 0 limit formula, both asserted), C and D (Adam 50k + per-row polish 10k, 5 seeds)
in closed form at every w; TD-trained A, C, D (50k steps, weighted per-row TD loss, 10 runs) at
w >= 1/128; ranks 2 and 3.

Closed form (A exact; C, D mean of 5 seeds), with the bottleneck rows' share of the target's
squared mass, FB's relative cost of keeping the gap (A+ minus A, over A), and the w -> 0 limit
formula dv^T P_d^0 r / dv^T r; then TD-trained, mean of 10 runs:

| w | rank 2: A | C | D | share of mass | A cost (rel) | A limit | trained A | C | D | rank 3: A | C | D | trained A | C | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1/2 | 1.029 | 1.015 | 0.976 | 0.520 | 0.005 | 0 | 0.714 | 0.713 | 0.635 | 1.029 | 0.998 | 0.983 | 0.714 | 0.893 | 0.914 |
| 1/4 | 0.000 | 1.008 | 0.951 | 0.265 | 0.068 | | 0.714 | 0.831 | 0.399 | 1.042 | 0.994 | 0.973 | 0.714 | 0.992 | 0.958 |
| 1/6 (uniform) | 0.000 | 1.006 | 0.938 | 0.178 | 0.151 | | 0.714 | 0.765 | 0.329 | 1.044 | 0.992 | 0.967 | 0.714 | 0.987 | 0.972 |
| 1/8 | -0.000 | 1.006 | 0.932 | 0.134 | 0.191 | | 0.714 | 0.705 | 0.611 | 1.044 | 0.994 | 0.964 | 0.714 | 0.995 | 0.979 |
| 1/16 | 0.000 | 1.002 | 0.918 | 0.067 | 0.249 | | 0.714 | 0.866 | 0.534 | 1.041 | 0.993 | 0.960 | 0.714 | 0.993 | 0.986 |
| 1/32 | 0.000 | 1.002 | 0.914 | 0.034 | 0.277 | | 0.714 | 0.926 | 0.561 | 1.039 | 0.992 | 0.958 | 0.714 | 0.982 | 0.992 |
| 1/64 | 0.000 | 1.001 | 0.913 | 0.017 | 0.291 | | 0.714 | 0.936 | 0.495 | 1.037 | 0.991 | 0.957 | 0.714 | 0.983 | 0.992 |
| 1/128 | 0.000 | 1.001 | 0.911 | 0.008 | 0.298 | | 0.714 | 0.941 | 0.640 | 1.037 | 0.991 | 0.956 | 0.714 | 0.916 | 0.994 |
| 0 (limit) | 0.000 | 0.426 | 0.381 | 0 | 0.304 | 0.000 | | | | 1.036 | 0.961 | 0.881 | | | |

Rank 2 is the lemma's case on P6: the bottleneck row difference is orthogonal to the ordinary
rows' top-2 span (limit exactly 0.000; the direction it lives in, v_3 of X, carries 3% of the
mass), so FB's closed-form optimum keeps the gap only when the two bottleneck rows carry half
the weight (1.03 at w = 1/2) and reads exactly 0 from w = 1/4 down, including the uniform
weighting: the crossover is between w = 1/2 and 1/4, one step above uniform. Its cost of
keeping the gap rises from 0.5% of the objective at w = 1/2 to 30% in the limit. The two
softmax arms keep it at every finite w (C 1.00-1.02, D 0.91-0.98, sliding slowly as w falls)
and lose more than half of it only in the w = 0 limit, where B is chosen by the ordinary rows
alone (C 0.43, D 0.38; on the old family the rank-2 limit was 0.85 / 0.76, the rank-3 limit
1.00). At rank 3 the row difference is inside the ordinary top-3 span (limit 1.036), so FB keeps
1.03-1.04 at every w and D 0.96-0.98 (0.88 in the limit).

TD-trained: FB's fixed point on P6 does not respond to w at all: every one of the 80 runs per
rank (all w, both learning rates, five seeds) converges to the same rank-2 span -- the two
cluster-entry states 2 and 4 separated, all other states collapsed onto one direction (B rows
after orthonormalization: (-.98, .04) for state 2, (.00, -.98) for state 4, (-.11, -.10) for
states 1, 3, 5) -- whose projection reads exactly 5/7 = 0.714 of the gap, and to the same value
at rank 3. Its least-squares residual is 383 against 175 for the SVD optimum at rank 2. So on
P6 the reweighting moves the closed-form optimum (1.03 -> 0) and leaves the TD fixed point
where it is (0.714). Trained Density-FB at rank 2 is unstable (means 0.33-0.64, single runs
from 0.00 to 0.91 at every w, both learning rates alike) and at rank 3 is at 0.91-0.99 for every
w; trained C is at 0.71-0.94 (rank 2) and 0.89-1.00 (rank 3). The rank-2 figure
`fig_sb_rare_lemma_d2` therefore shows the lemma's two halves for the closed-form fits (dotted:
the bottleneck share of the squared mass 0.52 -> 0.008 -> 0; FB's gap 1.03 -> 0), with the
trained FB flat at 0.71.

## 3. Lemma check (P11 + R1-R5)

`lemma_check/run.py`'s three tests on the same trained FB runs as the rank sweep (one training
pass feeds both): (a) the projection prediction from the trained B alone, dv^T P_U r / n,
against the gap FB reads with its trained F, at every rank and run; (c) the residual direction
dv - P_U dv appended to the span, and the objective cost against the optimal rank-(d+1) fit; (b)
the reward sweep at rank 2 (unit reward on each state + 50 random rewards per run).

(a) 560 trained runs (six processes, ranks 2..n, 10 runs each): correlation between the
projection prediction and the read gap 0.997, sign agreement 0.989, mean |difference| 0.013,
worst 0.22 (a rank-5 run). Per rank (mean lemma | mean read | worst |diff|): d=2 .115|.115|
.0004, 3 .380|.385|.041, 4 .633|.632|.104, 5 .926|.918|.219, 6 .948|.945|.073, 7 .983|.965|
.161, 8 1.018|.996|.184, 9 1.008|.989|.137, 10 1.015|1.005|.065, 11 1.001|.995|.072, 12 1.000|
1.000|.0002. As before the residual is the TD fixed point's distance from the row-wise
least-squares optimum given B, largest where the gap is half kept. Closed-form fits sit on the
diagonal to 1e-9 (asserted at every rank), and the per-rank correlations of 0.19-0.8 at d >= 8
are the correlation of two constants (both ~1.00 with spread < 0.02).

(b) Reward sweep at rank 2, 3620 (reward, run) pairs: correlation 1.0000, worst |lemma - read|
0.0006 -- at rank 2 the trained F is at its least-squares optimum given B on every process. The
kept fraction per unit reward shows the two regimes of the family: on P11, R3, R4, R5 (shared
tail) the rank-2 span is orthogonal to the bottleneck row difference, so EVERY reward with a
gap loses it (fractions -0.04 to +0.08 for all unit rewards on the cluster states), while on R1
(mutual leak, 6 states) the same features keep 0.93 of the gap for the reward at state 2 and
-0.06 for state 3, 1.01 for state 4 and 0.10 for state 5; on R5 the tail rewards at states 9 and
10 (true gaps -0.09 / -0.07) keep 0.93 / 0.25. With the residual direction appended the gap is
exact for every reward (max error 1.3e-15).

(c) Closed-form FB, gap fraction at rank d | with the residual appended | objective above the
optimal rank-(d+1) fit: d=2 0.069 | 1.000 | +7.2%, 3 0.611 | 1.000 | +8.1%, 4 1.033 | 1.000 |
+13.7%, 5 1.039 | +10.5%, 6 1.030 | +11.3%, 7 1.039 | +5.8%, 8 1.024 | +2.6%, 9 1.009 | +2.9%,
10 1.000 | +3.2%, 11-12 exact | 0. Spending one direction on the discarded residual costs 7-8%
of the objective at the ranks where FB has lost the gap.

## 4. Mitigation measurements (closed form, P11 + R1-R5, ranks 2..5)

All feature sets are first orthonormalized (QR of B): the fit F B^T and the softmax family
{softmax(B f)} depend on B only through its span, so this is the canonical representative and
makes the margin comparable across fits. Feature sets: FB's = the top-d right singular
directions of nM (one per process and rank); Density-FB's = the B of arm D's closed-form Adam fit
(the same five seeds as the ablation).

(i) Readout swap. Given a feature set, linear rows are the per-row least squares of nM (F = X B,
fit = X P_B, the lemma's projection); softmax rows are the per-row maximum likelihood of the
target rows (1-gamma)M in the family softmax(B f) (each row's cross-entropy is convex in f;
L-BFGS with the analytic gradient B^T(p - t)). The four (features x readout) cells: FB features
with linear rows IS arm A; Density-FB features with softmax rows is arm D after the per-row
polish (the polish never raises the cross-entropy, asserted).

(ii) Separability margin. For each feature set, max over unit u of min_{s in A} b_s.u -
max_{s not in A} b_s.u on the rows of the orthonormalized B: positive iff cluster A is linearly
separable from all other states in feature space, and then the width of the separating slab.
Computed by 20k random unit directions followed by SLSQP refinement on the sphere from the best
eight.

(iii) Moment check. At a per-row optimum given B the gradient vanishes: B^T(row_fit - row_true)
= 0 with row_true = the row of nM (least squares) or of (1-gamma)M (maximum likelihood).
Reported as the max |.| over all rows and fits; asserted < 1e-9 / 1e-6.

(i) The 2 x 2, mean over the six processes (x 5 seeds for Density-FB's features), fraction
of the exact gap [95% t-interval]:

| rank | FB features, linear rows (= arm A) | FB features, softmax rows | Density-FB features, linear rows | Density-FB features, softmax rows (= arm D) |
|---|---|---|---|---|
| 2 | 0.069 [-0.09, 0.23] | 0.225 [-0.08, 0.53] | 0.300 [0.21, 0.39] | **0.893** [0.86, 0.93] (joint fit 0.880) |
| 3 | 0.611 [-0.01, 1.23] | 0.646 [0.15, 1.14] | 0.197 [0.14, 0.26] | **0.965** [0.95, 0.98] (0.963) |
| 4 | 1.033 [0.91, 1.16] | 1.002 [0.99, 1.01] | 0.562 [0.44, 0.69] | 0.989 [0.98, 1.00] (0.988) |
| 5 | 1.039 [0.97, 1.10] | 1.001 [0.99, 1.01] | 0.815 [0.67, 0.96] | 0.997 [0.99, 1.00] (0.995) |

Per process at rank 2 (linear | softmax rows): FB features P11 .00|.00, R1 .39|.78, R2 .02|.31,
R3 -.01|.07, R4 .00|.17, R5 .01|.02; Density-FB features P11 .38|.99, R1 .00|.93, R2 .11|.75,
R3 .41|.97, R4 .54|.87, R5 .37|.86. Neither half is enough on its own: swapping the readout on
FB's rank-2 features moves 0.07 to 0.22 (R1, where the SVD span already half-separates A, is
the only process it helps), and reading Density-FB's features linearly gives 0.30 / 0.20 / 0.56 /
0.82 at ranks 2-5 -- its features are not a good linear model of nM at any of these ranks. The
gap is kept when the softmax rows are fitted on the features that were trained for them. On
FB's features the softmax rows match the linear rows at ranks >= 4 (1.00 vs 1.03), i.e. once the
span contains the row difference the readout does not matter.

(ii) Separability margin (144 feature sets: 24 SVD + 120 Density-FB, ranks 2..5). Correlation
with the retained gap over all fits: linear rows Pearson 0.881, Spearman 0.909; softmax rows
Pearson 0.374, Spearman 0.693. The scatter (`fig_sb_margin`) is a step for the softmax rows:
every fit with margin > 0.02 keeps >= 0.85 of the gap, and the fits that lose it are the ones
where cluster A is not separable. FB's SVD features separate A in 1/6 processes at rank 2
(margins -0.09 to +0.001), 5/6 at rank 3, 6/6 at ranks 4-5; softmax rows on them read 0.900 when
A is separable and 0.173 when it is not. Density-FB's features separate A in 23/30 fits at rank
2 (the failures are borderline, margin >= -0.017) and 30/30 at ranks 3-5, with mean margins
0.11 / 0.09 / 0.31 / 0.47; softmax rows on them read 0.964 when separable, 0.913 on the seven
borderline fits. For the linear readout the margin is a proxy for whether the row difference
is in the span, hence the higher Pearson; a large margin with a linear readout on Density-FB's
features still gives 0.3-0.7 in many fits (the green hollow points), since separability of A is
necessary but not sufficient for a linear fit of nM to reproduce the bottleneck rows.

(iii) max |B^T (row_fit - row_true)| over all 144 x 2 fits and all rows: least squares 2.4e-14,
maximum likelihood 5.4e-08 (asserted < 1e-9 and < 1e-6). Both readouts are at their per-row
optimum given the features, so the 2 x 2 compares optimal readouts and not optimizer luck.

## 5. Where the gap lives in the spectrum (`fig_gap_direction`, `gap_direction.py`)

Closed form on the exact X = nM of P6 and P11. Share of the matrix's squared mass carried by the
singular directions v_1.. and share of |dv|^2, dv = X[2b] - X[2b+1]:

| | v_1 | v_2 | v_3 | v_4 | v_5 |
|---|---|---|---|---|---|
| P6 mass | 0.906 | 0.038 | 0.030 | 0.010 | 0.009 |
| P6 gap | 0 | 0 | **0.961** | 0 | 0.039 |
| P11 mass | 0.752 | 0.112 | 0.041 | 0.033 | 0.019 |
| P11 gap | 0 | 0 | 0 | **0.945** | 0 |

The bottleneck row difference is carried by one direction that holds 3% of the mass (P6: the
third, P11: the fourth; the rest sits in v_10 for P11, v_5 for P6). The cluster-A reward seen
through the top-d span, P_d r, on the A states / the B states: P11 d=2 0.021, 0.010 / 0.021,
0.010; d=3 0.274, 0.181 / 0.274, 0.181; d=4 0.843, 0.584 / -0.294, -0.223. P6 d=2 0.602, 0.273 /
0.602, 0.273; d=3 1.181, 0.652 / 0.023, -0.106; d=4 1.151, 0.725 / -0.007, -0.032. At ranks
below the gap direction the projected reward is IDENTICAL on the two clusters -- the features
cannot tell "reward on A" from "reward on B" -- and it separates exactly when that direction
enters. TD-trained FB features (mean of 10 runs of the projection onto span(B)): P11 rank 2
0.077, 0.037 / 0.077, 0.037 (gap fraction 0.000); rank 4 0.768, 0.244 / 0.244, 0.083 (0.420);
P6 rank 2 1.061, 0.120 / 0.061, 0.120 (0.714); rank 4 1.023, 0.536 / 0.021, 0.103 (0.840). On P11
the learned rank-2 features show the same A = B symmetry as the SVD span, and the learned rank-4
features separate the clusters only partly (0.42 of the gap; the closed-form rank-4 span reads
1.04).

## Reading

The suite reproduces the old family's structure on the new processes, with sharper numbers.
FB's closed-form fit loses the action gap at the ranks below the one singular direction that
carries the bottleneck row difference (94-96% of |dv|^2 in a direction with 3% of the mass), and
at those ranks the reward on cluster A is literally indistinguishable from the reward on cluster
B through the features. TD-trained FB needs a median rank 5 to read 90% of the gap where the
softmax arms need 3; the readout swap says this is the pair (features trained for a softmax
readout, and that readout), not either half; the margin says the softmax readout keeps the gap
exactly when cluster A is linearly separable in feature space, which FB's spectral features do
not provide at rank 2 and Density-FB's do at every rank >= 2 with one exception per process in
five seeds. The projection lemma holds for trained features (corr 0.997) and for every reward
(corr 1.000 at rank 2), and the reward sweep on the shared-tail processes shows the strongest
form of it: the rank-2 span is orthogonal to the row difference, so all rewards lose the gap.

Two results are not what the old suite said and need to be stated as such. (1) The rarity
lever moves FB's closed-form optimum on P6 exactly as the lemma predicts (crossover between w =
1/2 and 1/4, limit 0), but the TD fixed point of the shipped FB harness on P6 is at 0.714 for
every w and both ranks, so the reweighting experiment does not transfer to the trained arm on
this process; on the old nine-state process trained FB did follow w (0.98 -> 0.05 -> -0.01). (2)
Density-FB at rank 2 is fragile: its closed-form optimum keeps 0.88-0.98 but its w -> 0 limit
keeps 0.38 and its TD fixed point on P6 is unstable across seeds (0.00-0.91); at rank 3 and
above both are at 0.9-1.0. The clean single-bottleneck layout makes rank 2 a harder case for the
softmax arms than the old family's rank 2 was.

## Caveats

1. Aggregates over ranks 2..6 only, because R1 has 6 states; the per-process panels and all
   tables in `sb_log.txt` go up to each n. The random template allows entry 1 / cluster 2 /
   tail 0 (= P6's layout), which R1 drew; the other four draws are 10-12 states.
2. The keep rule for the random draws (exact gap > 0.5, FB rank-2 fraction < 0.5) rejected one
   candidate out of six; the template produces FB's rank-2 failure on its own. The rule
   guarantees the failure at rank 2, not at rank 3 (R5: 1.28 at rank 3 in closed form).
3. TD-trained FB on P6 (and on P11, R1, R2 at rank 2) converges to the same fixed point in
   every run, so the trained intervals there are zero-width and say nothing about seeds. The
   trained-vs-closed-form gap for FB at ranks 3-4 (0.38 vs 0.61, 0.63 vs 1.03) is a property of
   the TD fixed point with the orthonormality penalty at 1, as shipped; it was not varied.
4. Closed-form C and D are Adam fits (10k steps here, as in `family.py`; the rarity sweep uses
   50k + 10k per-row polish as in `rare_bottleneck/run.py`); the per-row maximum-likelihood
   polish in the mitigation block never raised the cross-entropy (asserted) and moved the
   rank-2 fraction from 0.880 to 0.893.
5. The separability margin is computed on the orthonormalized feature basis; its magnitude
   depends on that choice of representative (its sign does not). Random search + SLSQP gives a
   lower bound on the max; for the non-separable fits (margin <= 0) the value is a local
   optimum on the sphere.
6. Arm F's plateau at 0.76 is a property of the smoothing kernel (width 0.5 edges) on graphs
   where cluster A is one edge from the tail and from cluster B; it is not tuned.
7. Same scope as the old suite: one fixed policy per process, no actor, no reward encoder, no
   minibatch pool, exact population expectations, float64, CPU.

## Reproduce

```bash
cd /home/adityamohan/git/Factored-FB-seam
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/run.py                  # full suite, 878 s CPU on the shared box
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/run.py --figures-only   # figures + tables from the CSVs
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/run.py --smoke          # 30 s check
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/single_bottleneck/gap_direction.py         # fig_gap_direction, ~20 s
# the existing scripts still run unchanged:
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/toy_ablation/family.py --figures-only
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/rare_bottleneck/run.py --figures-only
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/probes/test_lemma_check.py -q
```
(jax 0.10.0, optax 0.2.8, float64, CPU only.)
