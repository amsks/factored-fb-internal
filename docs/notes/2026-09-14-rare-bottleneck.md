# 2026-09-14 -- Rare bottleneck: is the state-7 decision locally consequential but globally rare?

Code: `analysis/rare_bottleneck/run.py`; data `closed.csv` (720 rows), `trained.csv`
(420 rows), `exact.json`, `run_log.txt`; the first pass at the shorter fit budget is kept
as `closed_10k.csv` / `run_log_10k.txt` (see caveat 1). Figure
`PAPER/ICLR/figures/rare_bottleneck/fig_rare_bottleneck.{pdf,png}` (three panels, 5.5 in)
and `fig_rare_bottleneck_ranks.{pdf,png}` (both constructions at every rank). Nothing
committed. Built on `analysis/toy_ablation/run.py` (same MDP, same arms A / C / D, same
metric, same TD harness); read `docs/notes/2026-09-14-toy-ablation.md` first.

## Question

The paper wants a lemma of this shape: with training weight w on the two state-7 rows
(the decision: action 0 enters the goal, action 1 returns to the group) and 1-w on all
ordinary rows, (i) the bottleneck rows' share of the weighted squared objective goes to 0
as w -> 0; (ii) the exact return gap at state 7 does not depend on w; (iii) the rank-d
minimizer of the weighted squared objective has predicted gap -> d^T P_d^0 r where P_d^0 is
the projection the ordinary rows choose on their own, so FB's preserved fraction goes to 0
at fixed rank when the row difference lies outside that span; (iv) the softmax /
Density-FB fit at the same rank keeps the gap. (i)-(iii) are proved below; (iv) is tested.

## Setup

Nine-state didactic MDP (`scripts/didactic/graph_bottleneck_td.py`, leak 0), discount
0.8, uniform continuation policy, exact successor matrix M (18 rows = state-action pairs,
9 columns), reference weight 1/n on candidate states, reward 1 at state 8. Exact values
at state 7: 2.2 (action 0) and 0.8 (action 1), gap 1.4. Metric: fraction of that gap the
fit reads, (Q_hat(7,a0) - Q_hat(7,a1)) / 1.4.

Two ways of making the decision rare.

- (a) Row reweighting. The two state-7 rows carry total weight w (w/2 each), the other
  16 rows share 1-w uniformly; w in {1/2, 1/4, 1/8, 1/9, 1/16, 1/32, 1/64, 1/128, 0}.
  w = 1/9 is exactly the uniform weighting of the toy ablation. w = 0 is the limit: B is
  chosen with the bottleneck rows at zero weight, then their two F rows are fitted given
  that B.
- (b) Structural rarity. N extra ordinary states, as N/4 copies of the group-A pattern
  (states 0-3), each copy exiting into the cut state 4 exactly as group A does; uniform
  row weights, so the two state-7 rows carry w = 1/n = 1/(9+N); N in {0, 4, 8, 16, 32,
  64}, n = 9 ... 73. The original nine states cannot reach the copies, so their rows of
  M are unchanged and the exact gap is 1.4 for every N (asserted to 1e-9; the first 18
  rows of M are identical to the N = 0 matrix and are zero on the new columns).

Arms, all at rank d in {2, 3, 4, 6}:

- A, FB: F B^T fits the ratio n M with the weighted squared error
  sum_i w_i (1/n) sum_j (F_i.B_j - n M_ij)^2 (FB's population loss; the 1/n is the
  reference-weight average over candidate states). Exact minimizer = weighted truncated
  SVD. Value read as F z, z = (1/n) B^T r, i.e. Q_hat = M_hat r.
- A+, FB forced to keep the gap: same objective, minimized over rank-d fits whose B-span
  contains the state-7 row difference (one direction on it, the other d-1 optimal given
  that). Its gap fraction is 1 by construction (checked); its objective minus A's is the
  price of keeping the gap. This is an upper bound on the cheapest gap-preserving fit
  (keeping the gap only needs the part of the row difference outside the span to be
  orthogonal to r).
- C, softmax + squared error: p_i = softmax(F_i B^T) fitted to (1-gamma) M with
  sum_i w_i 0.5 sum_j (p_ij - t_ij)^2; value = p r / (1-gamma).
- D, Density-FB: same model, weighted cross-entropy sum_i w_i sum_j -t_ij log p_ij; same
  integration readout.

C and D: Adam (lr 1e-2) on the population objective, 50k steps, 5 seeds (single normal
init each), then with B frozen 10k more steps of every row of F on its own row loss. Rows
decouple given B, so that second step changes nothing about the optimum; it removes the
small-gradient problem of rows whose weight is w/2 -> 0 and is what the w = 0 limit needs.
The objective reported for D is the weighted KL (cross-entropy minus the target rows'
entropy, 0 at a perfect fit). Retained squared mass = 1 - ||P_hat - (1-gamma) M||^2 /
||(1-gamma) M||^2, unweighted (a weighted version is in the CSV).

Trained: the didactic TD harness (`scripts/didactic/bottleneck_family_td.py`:
bootstrapped target network, EMA 0.005, Adam, 50k steps, learning rates {0.001, 0.01},
orthogonality penalty at 1 on FB as shipped) with the per-row TD loss weighted by w_i
instead of averaged (uniform w_i = 1/(2n) reproduces the harness exactly). Arms A, C, D
at d = 3, 5 seeds x 2 rates = 10 runs, both constructions; w = 0 is skipped (TD needs
every row to receive gradient).

## Proof sketch of (i)-(iii); what P_d^0 is

Notation: rows i of X = n M; bottleneck rows i0 = (7, a0), i1 = (7, a1); ordinary rows O
(2n-2 of them); row difference dv = X[i0] - X[i1] in R^n; r = e_goal. The exact gap is
(M[i0] - M[i1]) r = dv.r / n. Row weights w_i = w/2 on the bottleneck rows, (1-w)/(2n-2)
on each ordinary row. FB's objective at rank d:
J_w(F, B) = sum_i w_i (1/n) ||F_i B^T - X_i||^2.

(ii) The exact gap is a property of M; the weights enter only the objective. The code
checks 1.4 for every w and every N.

Projection identity. Take B with orthonormal columns (any B can be re-parametrized so
without changing the fit F B^T). The objective is then a sum over rows of independent
least-squares problems in F_i, so each row is fitted by its own projection: F_i = X_i B and
the fitted row is X_i P_B with P_B = B B^T the orthogonal projection onto span(B). Two
consequences. First, once span(B) is fixed the row weights have no say in how a row is
fitted: a row with weight 1e-9 and a row with weight 1/2 are both fitted as X_i P_B.
Second, the objective collapses to a function of the subspace alone,
J_w(B) = (1/n) tr((I - P_B) G_w), G_w = sum_i w_i X_i^T X_i = w G_bn + (1-w) G_O,
with G_bn the bottleneck rows' Gram (mean of the two) and G_O the ordinary rows' Gram
(uniform mean over O). The minimizer over d-dimensional subspaces is the top-d eigenspace
of G_w (Ky Fan), i.e. the top-d right singular subspace of diag(sqrt w) X -- the weighted
truncated SVD used for arm A.

(i) At any (F, B) the bottleneck rows' share of J_w is w a / (w a + (1-w) b), with a the
mean bottleneck residual and b the mean ordinary residual. At the minimizer a <= (1/n)
mean ||X_bn||^2 (a projection cannot increase a norm) and b >= (1/n) sum_{k>d}
lambda_k(G_O) > 0 for d < rank(X_O) (the ordinary rows' own rank-d residual is a lower
bound under any subspace), so the share is at most w a_max / ((1-w) b_min) -> 0 linearly
in w. The fit-independent version, the bottleneck rows' share of the weighted squared
mass of the target, also -> 0 linearly (tables: 0.57 at w = 1/2, 0.14 uniform, 0.01 at
1/128).

(iii) Define P_d^0 as the orthogonal projection onto the top-d eigenspace of G_O, the
ordinary rows' Gram at uniform weights -- equivalently onto the d leading right singular
directions of the ordinary rows' matrix X_O. It is the subspace the ordinary rows would
choose at rank d on their own. Since G_w -> G_O as w -> 0 and the d-th and (d+1)-th
eigenvalues of G_O are separated (relative gaps 0.44 / 0.023 / 0.015 / 0.011 at d = 2 /
3 / 4 / 6), the top-d eigenspace of G_w converges to that of G_O (Davis-Kahan: subspace
distance <= ||G_w - G_O|| / gap = w ||G_bn - G_O|| / gap), so P_{B_w} -> P_d^0. By the
projection identity the predicted gap of the weighted-SVD fit is
gap_w = (X[i0] - X[i1]) P_{B_w} r / n = dv^T P_{B_w} r / n -> dv^T P_d^0 r / n,
and the preserved fraction converges to dv^T P_d^0 r / dv^T r. The projection identity
enters twice: the bottleneck rows cannot buy themselves a direction through F (their fit
is X_i P_B whatever their weight), and the choice of B is an eigenproblem of a Gram that
sees the bottleneck rows only through the O(w) term w G_bn.

The limit is 1 when dv lies in span(P_d^0) (more generally when the part of dv outside
the span is orthogonal to r); otherwise it is whatever number the projection gives, and
it is not 0 in general. On this MDP: d = 2: -0.021, d = 3: -0.002, d = 4: 0.922, d = 6:
1.052 (share of ||dv||^2 inside the span: 0.002 / 0.012 / 0.705 / 0.938). So at ranks 2
and 3 the limit is zero to three decimals -- the ordinary rows' leading directions are
almost orthogonal to the state-7 row difference -- while at rank 4 the ordinary rows
already span 70% of dv and the limit is 0.92. The lemma's "goes to 0" is right at d <= 3
here; the general statement is "goes to dv^T P_d^0 r / dv^T r". The weighted SVD at w = 0
agrees with the formula to 1e-8 (asserted), and at w = 1/9 it reproduces the toy
ablation's SVD numbers (-0.006 / 0.975 / 1.069 / 1.038).

Cost of keeping the gap (A+). The cheapest rank-d fit with dv in span(B) puts one direction
on dv/||dv|| and d-1 optimal directions on the ordinary problem projected off dv. Its
objective minus the unconstrained optimum, c_w >= 0, decomposes into an ordinary part
(>= 0) and a bottleneck part (<= 0, the benefit). As w -> 0 the benefit is O(w) and c_w ->
c_0 = the ordinary rows' extra residual from giving up one direction, a fixed positive
number whenever dv is outside the top-d span of G_O. That is the mechanism: the optimum
drops the gap once the O(w) benefit falls below the fixed price. So c_w does not go to 0
as w -> 0 (the task's phrasing); it goes to 0 as w grows, when the unconstrained optimum
keeps the gap by itself. What goes to 0 with w is the objective's incentive to keep the
gap (the bottleneck rows' share and their absolute weighted residual).

(iv) is not a projection statement. For the softmax model the same decoupling holds --
given B each row is fitted on its own by maximum likelihood in the exponential family
{softmax(B f)} -- so w again enters only through B. The question is whether a B chosen by
the ordinary rows leaves that family able to put the right mass on the goal for the two
state-7 rows. The ordinary rows already need B_goal separated from the other columns
(the (8, a1) row is a self-loop peaked on the goal; the group-B rows differ mainly in
their goal mass), and once B_goal is separated a single F_i direction raises any row's
goal mass without a new direction in B. That is the argument; the numerics decide.

## Results, construction (a): row reweighting

Gap fraction at d = 3 (C, D: mean of 5 seeds, 95% t-interval; A exact):

| w | A FB | C softmax+sq | D Density-FB | A: bottleneck share of objective | share of target mass | A: cost of keeping gap (abs; relative) | D: bottleneck share of objective | A limit |
|---|---|---|---|---|---|---|---|---|
| 1/2 | 1.032 | 1.000 | 1.001 | 0.046 | 0.567 | 0.026; 0.010 | 0.021 | -0.002 |
| 1/4 | 1.057 | 1.032 [1.009, 1.055] | 0.999 | 0.058 | 0.304 | 0.067; 0.018 | 0.039 | |
| 1/8 | 1.021 | 1.030 | 1.003 | 0.056 | 0.158 | 0.110; 0.026 | 0.141 | |
| 1/9 (uniform) | 0.975 | 1.030 | 1.004 | 0.061 | 0.141 | 0.123; 0.029 | 0.125 | |
| 1/16 | 0.185 | 1.031 | 1.004 | 0.125 | 0.080 | 0.309; 0.074 | 0.073 | |
| 1/32 | 0.031 | 1.032 | 1.003 | 0.076 | 0.041 | 0.562; 0.140 | 0.037 | |
| 1/64 | 0.010 | 1.033 | 1.004 | 0.040 | 0.020 | 0.696; 0.178 | 0.019 | |
| 1/128 | 0.004 | 1.033 | 1.004 | 0.021 | 0.010 | 0.765; 0.198 | 0.010 | |
| 0 (limit) | -0.002 | 1.023 [1.005, 1.041] | 1.004 [1.003, 1.005] | 0 | 0 | 0.833; 0.219 | 0 | -0.002 |

(C and D intervals narrower than 0.005 are omitted. Cost decomposition for A at d = 3:
ordinary part 0.042 / 0.093 / 0.175 / 0.218 / 0.718 / 0.802 / 0.819 / 0.827 / 0.833,
bottleneck part -0.016 / -0.026 / -0.064 / -0.096 / -0.410 / -0.240 / -0.123 / -0.062 / 0
down the same rows.) Retained squared mass at d = 3: A 0.912-0.918 at every w, A+
0.911-0.916, D 0.992-0.994: keeping or losing the gap moves the retained mass by less than
0.005.

FB's optimum crosses over between w = 1/9 and 1/16: at 1/16 it keeps 18.5%, at 1/32 3%,
then converges to the limit -0.002. The two softmax arms hold 1.00-1.03 at every w down to
and including the limit, and D's predicted goal mass on the two state-7 rows stays at
0.433 / 0.152 (exact 0.440 / 0.160) from w = 1/8 to w = 0.

Other ranks (gap fraction; full rows in `run_log.txt`):

| w | A d=2 | C d=2 | D d=2 | A d=4 | C d=4 | D d=4 | A d=6 | C d=6 | D d=6 |
|---|---|---|---|---|---|---|---|---|---|
| 1/2 | 0.048 | 1.006 | 1.015 | 1.029 | 1.000 | 1.000 | 1.001 | 1.000 | 1.000 |
| 1/4 | 0.012 | 1.073 | 1.024 | 1.057 | 1.000 | 0.999 | 1.015 | 1.000 | 1.000 |
| 1/8 | -0.005 | 1.079 | 1.040 | 1.069 | 1.000 | 1.000 | 1.035 | 1.000 | 1.000 |
| 1/9 | -0.006 | 1.071 | 1.042 | 1.069 | 1.000 | 1.000 | 1.038 | 1.000 | 1.000 |
| 1/16 | -0.013 | 1.014 | 0.869 | 1.060 | 1.000 | 1.000 | 1.045 | 1.000 | 1.000 |
| 1/32 | -0.017 | 0.938 | 0.810 | 1.029 | 1.000 | 1.000 | 1.049 | 1.000 | 1.000 |
| 1/64 | -0.019 | 0.901 | 0.783 | 0.992 | 1.000 | 1.000 | 1.050 | 1.000 | 1.000 |
| 1/128 | -0.020 | 0.887 | 0.794 | 0.962 | 1.000 | 1.000 | 1.051 | 1.001 | 1.000 |
| 0 | -0.021 | 0.849 [0.766, 0.931] | 0.761 [0.757, 0.765] | 0.922 | 0.999 | 0.997 | 1.051 | 1.000 | 1.000 |

(d = 2 column at the 50k budget for w = 1/128 and 0; the shorter budget gave 0.767 and
0.751 for D.) At d = 2 FB loses the gap at every w (its uniform-weight optimum already
does), and D declines from 1.04 (uniform) to 0.76 in the limit: goal mass 0.400 / 0.187
instead of 0.440 / 0.160, so the sign and three quarters of the gap survive, but not all of
it. At d = 4 FB's limit is 0.92 because the ordinary rows already span most of the row
difference, and at d = 6 it overshoots to 1.05; C and D are at 1.00.

Bottleneck share of FB's objective at the optimum (d = 3): 0.046 (w = 1/2), 0.061
(uniform), 0.125 (1/16, the crossover: the fit has just dropped the gap and the
bottleneck residual jumps), then 0.076 / 0.040 / 0.021 / 0 -- linear in w below the
crossover, as (i) says. The fit-independent target-mass share is 0.567 -> 0.010 -> 0.

TD-trained, d = 3, mean of 10 runs [95% t-interval] (per-rate means in `run_log.txt`):

| w | A | C | D |
|---|---|---|---|
| 1/2 | 0.981 [0.981, 0.981] | 1.006 [0.958, 1.053] | 0.951 [0.930, 0.972] |
| 1/4 | 0.906 [0.906, 0.906] | 0.997 [0.987, 1.007] | 0.978 [0.960, 0.997] |
| 1/8 | 0.120 [-0.014, 0.253] | 1.009 [0.994, 1.023] | 0.998 [0.993, 1.003] |
| 1/9 | 0.051 [-0.020, 0.121] | 1.002 [0.974, 1.029] | 0.989 [0.989, 0.990] |
| 1/16 | 0.003 [-0.002, 0.007] | 1.011 [1.000, 1.023] | 0.972 [0.969, 0.974] |
| 1/32 | -0.008 [-0.012, -0.003] | 1.016 [1.006, 1.025] | 0.973 [0.966, 0.979] |
| 1/64 | -0.007 [-0.011, -0.003] | 0.971 [0.907, 1.035] | 0.942 [0.939, 0.945] |
| 1/128 | -0.007 [-0.011, -0.003] | 0.970 [0.906, 1.033] | 0.885 [0.879, 0.891] |

Trained FB keeps the gap only when the bottleneck rows carry a quarter or more of the
weight (0.98 at 1/2, 0.91 at 1/4), is at 0.12 at 1/8 and at the uniform 0.05 of the toy
ablation, and reads the wrong action in 10/10 runs from w = 1/32 down. Trained
Density-FB stays at 0.95-1.00 down to w = 1/32 and declines mildly to 0.94 (1/64) and
0.885 (1/128), all runs with the right action; C stays at 0.97-1.02.

## Results, construction (b): structural rarity

Gap fraction (C, D mean of 5 seeds; A exact), with the bottleneck's share of the target's
squared mass and of FB's objective at the optimum, and FB's cost of keeping the gap:

| N | n | w = 1/n | A d=2 | A d=3 | A d=4 | A d=6 | A limit d=3 | share of target | A share of obj (d=3) | A cost d=3 (abs; rel) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9 | 0.111 | -0.006 | 0.975 | 1.069 | 1.038 | -0.002 | 0.141 | 0.061 | 0.12; 0.03 |
| 4 | 13 | 0.077 | -0.006 | -0.005 | 0.570 | 1.066 | -0.020 | 0.109 | 0.123 | 6.88; 0.80 |
| 8 | 17 | 0.059 | -0.005 | -0.005 | -0.003 | 1.064 | -0.021 | 0.089 | 0.057 | 7.11; 0.37 |
| 16 | 25 | 0.040 | -0.004 | -0.004 | -0.004 | -0.001 | -0.021 | 0.065 | 0.028 | 7.11; 0.17 |
| 32 | 41 | 0.024 | -0.003 | -0.003 | -0.003 | -0.003 | -0.020 | 0.042 | 0.014 | 7.11; 0.08 |
| 64 | 73 | 0.014 | -0.001 | -0.001 | -0.001 | -0.001 | -0.019 | 0.025 | 0.007 | 7.11; 0.04 |

| N | C d=2 | D d=2 | C d=3 | D d=3 | C d=4 | D d=4 | C d=6 | D d=6 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.064 | 1.033 | 1.030 | 1.004 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1.107 | 0.979 [0.927, 1.030] | 1.034 | 0.998 | 1.000 | 1.000 | 1.000 | 1.001 |
| 8 | 1.119 | 0.932 [0.887, 0.978] | 1.050 | 1.011 | 1.000 | 1.000 | 1.000 | 1.000 |
| 16 | 1.069 [0.913, 1.225] | 0.735 [0.584, 0.887] | 1.078 | 1.008 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32 | 0.861 [0.816, 0.906] | 0.287 [0.194, 0.379] | 1.090 [1.067, 1.114] | 1.004 [0.980, 1.027] | 1.000 | 1.001 | 1.000 | 1.000 |
| 64 | 0.868 [0.801, 0.934] | 0.020 [0.005, 0.035] | 1.086 [1.056, 1.116] | 0.983 [0.942, 1.024] | 1.000 | 1.000 | 1.000 | 1.000 |

Adding one copy of group A (N = 4, 13 states) is enough to take FB's rank-3 optimum from
0.975 to -0.005: with more ordinary rows the top-3 subspace of the ordinary Gram no longer
happens to lean toward the state-7 row difference (limit -0.020 for every N >= 4). Rank 4
survives one copy (0.57) and fails at two; rank 6 fails at N = 16. FB needs its rank to
grow with the number of ordinary states; every rank in the sweep is at -0.00x by N = 16.
The relative cost of keeping the gap for FB is 0.80 at N = 4 and falls to 0.04 at N = 64
only because the total objective grows with n (the absolute cost is a constant 7.11 from
N = 8 on), while the bottleneck share of the objective falls 0.12 -> 0.007.

Density-FB keeps the gap at d = 3 for every N (0.98-1.01; at N = 64, 73 states, the five
seeds read 0.99 / 1.01 / 1.00 / 0.99 / 0.93) and at d = 4 and 6 exactly. At d = 2 it
holds to N = 8 (0.93), is at 0.74 at N = 16 and collapses at N = 32 and 64 (0.29, 0.02).
Softmax + squared error keeps the gap at every rank and every N (0.86-0.87 at d = 2 for
N >= 32, 1.05-1.12 at d = 3 -- an overshoot that grows with N -- and 1.000 at d >= 4).

TD-trained, d = 3 (10 runs):

| N | A | C | D |
|---|---|---|---|
| 0 | 0.051 [-0.020, 0.121] | 1.002 [0.974, 1.029] | 0.989 [0.989, 0.990] |
| 4 | 0.013 | 1.009 [0.995, 1.024] | 0.974 [0.942, 1.006] |
| 8 | 0.012 | 1.018 [0.970, 1.067] | 0.735 [0.665, 0.805] |
| 16 | 0.011 | 1.025 [0.982, 1.068] | 0.659 [0.620, 0.697] |
| 32 | 0.012 | 1.012 [0.998, 1.025] | 0.508 [0.305, 0.711] |
| 64 | 0.013 | 0.980 [0.895, 1.066] | 0.311 [0.056, 0.565] |

Trained FB is at 0.01 for every N (as at N = 0). Trained Density-FB declines steadily with
N although its optimum does not: 0.74 at N = 8, 0.66 at 16, 0.51 at 32, 0.31 at 64, where
the two learning rates split (lr 0.001: 0.06 with four of five seeds at 0.00; lr 0.01:
0.56 with seeds at 0.68 / 0.81 / 0.58 / 0.76 / -0.01). Trained C stays at 0.98-1.03.
So under structural rarity the TD fixed point of the cross-entropy critic at rank 3 loses
most of the gap that the population optimum keeps -- the same kind of gap between optimum
and TD training the toy ablation found for FB at ranks 3-4.

## Does (iv) hold?

Under row reweighting, yes at rank 3 and above: Density-FB's population optimum keeps
100% of the gap at every w including the w = 0 limit (B chosen with the state-7 rows at
zero weight), while FB's goes to zero. At rank 2 it keeps 76% in the limit, versus FB's
-2%. The mechanism is the one sketched above: the softmax family with the ordinary rows'
B can still put 0.43 on the goal for row (7, a0) because the ordinary rows already
separate B_goal; no extra direction is needed. Under structural rarity, yes at rank 3 and
above up to 73 states, and no at rank 2 beyond 16 extra states (0.29, 0.02): the softmax
fit also needs some rank to keep B_goal separated as the number of columns grows, just
much less than FB does (rank 2 fails only at N >= 32; FB's rank 6 fails at N = 16).
Softmax + squared error is at least as robust as cross-entropy in every cell, so, as in
the toy ablation, the output class rather than the loss carries the result at the
optimum. The TD-trained cross-entropy critic is less robust than its optimum under
structural rarity (0.31 at N = 64) and mildly under reweighting (0.885 at w = 1/128).

## Caveats

1. The N = 64, d = 3 Density-FB cell at the first fit budget (10k Adam steps + 5k polish)
   read 0.128 (five seeds 0.06-0.18), i.e. a lost gap. Longer training (50k + 10k) reaches
   a lower objective (weighted KL 0.026-0.028 vs 0.035-0.060) and keeps the gap
   (0.93-1.01); lr 0.001 at 50k sits at a higher objective (0.032) with the gap lost
   (0.09-0.22). The whole closed-form sweep was therefore re-run at 50k + 10k, and all
   tables above are from that run. The d = 2 cells also moved with the longer budget (D at
   N = 32: 0.035 -> 0.287; at w = 0: 0.751 -> 0.761), always toward a lower objective and
   a larger gap. Closed-form C and D are Adam fits, not certified optima; the trend with
   budget says the remaining under-fit, if any, works against the softmax arms, not for
   them.
2. FB's cost of keeping the gap does not go to 0 as w -> 0; it converges to a positive
   constant (0.83 at d = 3, relative 0.22) and goes to 0 as w grows. The task's phrasing
   had the direction reversed; what vanishes with w is the bottleneck rows' share (and
   absolute contribution), i.e. the incentive. A+ is an upper bound on the cheapest
   gap-preserving fit.
3. The w -> 0 limit of FB's fraction is dv^T P_d^0 r / dv^T r, which is 0 at d <= 3 on
   this MDP and 0.92 / 1.05 at d = 4 / 6. The lemma should state the limit, not "0".
4. The structural construction repeats group A's pattern; the new rows are as "ordinary"
   as the old ones but the copies are structurally identical, which is one particular way
   of adding ordinary states. FB's loss of the gap at N = 4 is a statement about the
   leading directions of this particular ordinary Gram, not a rate.
5. Trained numbers pool two learning rates (the harness convention); the split matters
   only for A at w = 1/8 and 1/9 and for D at N = 32 and 64 (per-rate means in
   `run_log.txt`). TD-trained arms were run at d = 3 only.
6. Same scope limits as the toy ablation: nine (to 73) states, one policy, no actor, no
   reward encoder, no minibatch candidate pool, exact expectations.
7. In the figure the w = 0 limit is drawn at the left end of the log axis (tick "0");
   the dotted vertical line marks the uniform weighting w = 1/9. Hollow markers are the
   TD-trained means with 95% t-intervals over 10 runs.

## Reproduce

```bash
cd /home/adityamohan/git/Factored-FB-seam
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/rare_bottleneck/run.py                 # ~13 min CPU
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/rare_bottleneck/run.py --closed-only   # closed sweep only, reuse trained.csv
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/rare_bottleneck/run.py --figures-only
```
(tmux session `rarebn`; jax 0.10.0, optax 0.2.8, float64; trained.csv is from the first
run, closed.csv from the 50k re-run.)
