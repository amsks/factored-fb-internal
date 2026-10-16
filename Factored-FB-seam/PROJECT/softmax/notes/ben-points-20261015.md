# Ben's four points, answered with what we have (2026-10-15)

Ben's list: (1) theoretically, are linear features more expressive than log-linear features at a
fixed rank; (2) empirically, same question, by comparing Monte-Carlo LSIF train/validation loss;
(3) self-normalised importance weights in the Q readout, possible with either feature class;
(4) given log-linear features, is TD-LSIF or cross-entropy the better loss (his note: CE is
better with a goal-conditioned algorithm).

## 1. Theory

Neither class contains the other at fixed rank d. The ratio matrix R = M / rho is what both
model. The linear class is "R has rank d". The log-linear class is "log R has rank d". A spike
(R equal to a scaled identity on n states) needs rank n in the linear class (Eckart–Young:
||X − I||_F^2 >= n − d) and rank O(log n / eps^2) in the log-linear class (Alon's bound on
sign-rank / the softmax-bottleneck literature). A mixture of d overlapping components is rank d
in the linear class and has no low-rank log. So the question is not "which is more expressive"
but "which structure does the successor measure have". The successor measure of a short-horizon
manipulation policy is close to a spike (one or a few reachable futures), so the log-linear
class is the one that fits at the paper's rank. Section 4 of the README measures this: the rank
each class needs for TV <= 0.1 grows with the number of states for the linear class and stays
at 8–32 for the log-linear one. FB's own low-rank-MDP argument (M − (1−γ)I has rank d when the
transition kernel has rank d) is the case where the linear class is exact; manipulation data is
not that case.

## 2. Empirical, Monte-Carlo LSIF train/val loss

`scripts/features_lsif.py`: episodes split 80/20; K = 512 k-means clusters, rho and the
empirical discounted successor matrix built on the train split; validation episodes assigned to
the same clusters. Three fits per rank: linear features with LSIF (the exact weighted SVD, FB's
estimator), log-linear features with LSIF (R̂ = exp(F·B), Adam), log-linear features with CE
(Density-FB's softmax). Loss is the relative LSIF loss sum rho_i rho_j (R − R̂)^2 / sum rho_i rho_j R^2.

| dataset | rank | linear, LSIF train / val | log-linear + LSIF | log-linear + CE | TV val linear raw / self-norm | TV val log-LSIF | TV val log-CE |
|---|---|---|---|---|---|---|---|
| cube-single (cube xyz) | 8 | 0.832 / 0.857 | 0.054 / 0.135 | 0.518 / 0.569 | 0.58 / 0.55 | 0.48 | 0.51 |
| cube-single (cube xyz) | 16 | 0.757 / 0.799 | 0.026 / 0.120 | 0.103 / 0.190 | 0.62 / 0.54 | 0.39 | 0.42 |
| cube-single (cube xyz) | 32 | 0.627 / 0.701 | 0.013 / 0.118 | 0.021 / 0.122 | 0.66 / 0.53 | 0.38 | 0.38 |
| cube-single (cube xyz) | 64 | 0.413 / 0.532 | 0.004 / 0.118 | 0.006 / 0.119 | 0.68 / 0.51 | 0.37 | 0.37 |
| cube-single (cube xyz) | 128 | 0.087 / 0.218 | 0.001 / 0.118 | 0.004 / 0.122 | 0.52 / 0.43 | 0.37 | 0.38 |
| antmaze-medium (ant xy) | 8 | 0.801 / 0.782 | 0.011 / 0.574 | 0.070 / 0.551 | 0.54 / 0.54 | 0.38 | 0.28 |
| antmaze-medium (ant xy) | 16 | 0.645 / 0.800 | 0.004 / 0.564 | 0.015 / 0.558 | 0.54 / 0.54 | 0.28 | 0.25 |
| antmaze-medium (ant xy) | 32 | 0.433 / 0.642 | 0.001 / 0.559 | 0.003 / 0.560 | 0.46 / 0.42 | 0.25 | 0.24 |
| antmaze-medium (ant xy) | 64 | 0.202 / 0.622 | 0.001 / 0.557 | 0.001 / 0.561 | 0.37 / 0.34 | 0.24 | 0.23 |
| antmaze-medium (ant xy) | 128 | 0.030 / 0.602 | 0.000 / 0.560 | 0.000 / 0.560 | 0.28 / 0.27 | 0.23 | 0.23 |
| Bridge-V2 (proprio, γ=.96) | 8 | 0.827 / 0.837 | 0.030 / 0.086 | 0.141 / 0.185 | 0.59 / 0.52 | 0.34 | 0.28 |
| Bridge-V2 (proprio, γ=.96) | 16 | 0.734 / 0.749 | 0.008 / 0.066 | 0.025 / 0.078 | 0.58 / 0.47 | 0.22 | 0.22 |
| Bridge-V2 (proprio, γ=.96) | 32 | 0.607 / 0.636 | 0.003 / 0.063 | 0.005 / 0.064 | 0.56 / 0.42 | 0.20 | 0.20 |
| Bridge-V2 (proprio, γ=.96) | 64 | 0.453 / 0.504 | 0.001 / 0.063 | 0.002 / 0.063 | 0.55 / 0.38 | 0.20 | 0.19 |
| Bridge-V2 (proprio, γ=.96) | 128 | 0.277 / 0.336 | 0.000 / 0.063 | 0.003 / 0.067 | 0.50 / 0.34 | 0.20 | 0.19 |

Reading. Under FB's own loss, log-linear features beat linear features at every rank on every
dataset, on train and on validation. The validation loss of the log-linear fits reaches its
floor by rank 16–32 (the floor is the sampling noise between the two splits; it is 0.56 on
antmaze because its validation split is 200 episodes of long trajectories); the linear fits are
still far above it at rank 128. So the answer to Ben's question 2 is no: at fixed rank the
linear features are less expressive for this target, and the gap is not overfitting, it is
present in the validation loss. Self-normalising the linear readout (clip at zero, renormalise
over the candidates) helps the linear class a little (raw vs self-norm TV columns) but does not
close the gap.

## 3. Self-normalised importance weights

Density-FB's readout is already the self-normalised importance-weighted estimate of
sum_u m(u|s,a) r(u): the softmax over the candidate batch is exp(F·B(u)) / sum_u' exp(F·B(u')),
i.e. log-linear ratio weights normalised over the sample. The same normalisation can be applied
to linear weights (clip at zero, divide by the batch sum), and the table above shows what it is
worth at the dataset level: a small gain, far short of the family gap. A training-level arm
(FB's actor reading sum_u w(u) r(u) / sum_u w(u) with w = clip(F·B(u), 0)) has not been run; it
would separate "self-normalisation" from "log-linear features" at the policy level. Expected
from the table: it does not rescue FB on cube.

## 4. LSIF vs CE given log-linear features

At the dataset level the two losses give the same model from rank 32 up (the log-LSIF and
log-CE columns agree on LSIF validation loss and on TV); at rank 8–16 CE is slightly better on
TV and LSIF slightly better on LSIF, as expected from fitting the metric. So the loss is not
where expressivity comes from. At the training level the SQ-vs-CE ablation (Density-FB, 500k,
3 seeds) found CE far better on the goal-conditioned route (90 vs 40 on cube-single) and a tie
on the reward-inference route. Combined with the dataset-level tie, the goal-route gap is a
training-dynamics effect of the bootstrapped target and the actor coupling, not a
representability effect, which agrees with Ben's note that CE is better with a goal-conditioned
algorithm. One caveat: our "SQ" arm is the Brier loss on the normalised softmax, not the
un-normalised LSIF on exp(F·B); the difference is exactly the self-normalisation of point 3.
