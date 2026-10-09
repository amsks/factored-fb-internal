# Spherical FB follow-up: source audit and bounded TD checks

Date: 2026-09-11. This file reports one-head fixed-policy tabular TD, not the
production joint actor/critic algorithm and not a global TD optimization result.
The separate theory audit owns the constrained exact-target global certificate.
No paper files were changed by this search.

## Relevant production settings and objective scaling

`impls/configs/critic/fb_tdj.yaml:12–16,31–34` sets two forward heads,
backward output normalization on, and orthogonality coefficient 1000. This is
the FB configuration named by the current TD-JEPA-protocol campaign manifest.
It was verified in the four available resolved configurations:

- `exp/tdjtab/cube-single-play-v0_fb_tdj_flowbc_seed_2/config.json`
- `exp/tdjtab/cube-double-play-v0_fb_tdj_flowbc_seed_{0,1,2}/config.json`

Their backward norm is on; discount is .99; both target rates are .005;
forward and backward learning rates are 1e-4. These are source/protocol facts,
not independently audited benchmark results.

The base `fb.yaml` instead has coefficient 1. The older paper privileged-input
table explicitly reports coefficient 1, and the older main table attributes
some FB cells to an external implementation. There is therefore no single
coefficient covering every historical paper table.

Current density configurations use temperature 1, hard categorical targets,
and target rate .005. They have no orthogonality term. Backward normalization
is off by default (`density_fb.py:1319`), including when the older resolved
configuration omits that key. The paper appendix explicitly states no backward
output normalization or orthogonality penalty.

In `fb.py:145–151`, the squared and immediate terms **sum over forward heads**,
whereas the backward penalty appears once. Let E be the weighted squared error
of normalized successor probabilities, and C=E_rho[B B^T]. The common
exact-target objective above its constant penalty minimum is

    H / (2*(1-gamma)^2) * E + lambda/2 * ||C-I||_F^2,

for H identical heads. At gamma .8 this is 25 E + 500 ||C-I||_F^2 for two heads
and lambda=1000. A one-head calculation with lambda=500 is exactly half of that
objective, and thus has the same relative weighting. This does not assert full
training parity with two independently initialized, task-conditioned heads.
The orthogonality term has a factor **one half**; it is a soft penalty, not the
hard constraint C=I. Only B is normalized, to sqrt(d); F is unrestricted.

## Four-state path with one-head coefficient 500

Environment: deterministic reflecting left/right path 0–1–2–3. Fixed
continuation policy: left/right probability .5 each. Uniform state-action
sampling and uniform reference, which is both the actual next-state marginal
and stationary distribution. Reward: one on arrival at state 1. Start: state 0.
Dimension 2. Thus every backward row has norm sqrt(2).

Three seeds (0,1,2), two rates (.001,.01), and discounts (.5,.8,.9) were run for
100k expected TD updates with target rate .005, float64, and Adam defaults.
All settings apart from the backward norm/orthogonality coefficient follow the
previous tabular runner. The complete results are in
`spherical_td_search_path4_d2_l500/`. They include exact executed wrapper and
runner snapshots matching the hashes in `search_config.json`.

At **each** discount there are two optimal-control runs (seed 0, both rates)
and four wall failures (seeds 1/2, both rates). At gamma .8:

| Seed | Rate | Right-minus-left start value | Actual greedy return / optimum | Exact-target squared E | One-head exact-target objective above constant |
|---:|---:|---:|---:|---:|---:|
| 0 | .001 | +.49999999995 | 1 | .22837370242 | 2.85467128028 |
| 0 | .01 | +.50000000004 | 1 | .22837370242 | 2.85467128028 |
| 1 | .001 | −.23964893108 | 0 | .08330031867 | 1.04125906007 |
| 1 | .01 | −.23965075168 | 0 | .08329994246 | 1.04125435567 |
| 2 | .001 | −.23961798062 | 0 | .08330495036 | 1.04131728309 |
| 2 | .01 | −.23965343828 | 0 | .08329957724 | 1.04124977987 |

The true fixed-policy start gap is +.64705882353. Optimal greedy return is 25/9;
the failed policies remain at state 0 and earn zero. Numerical linear solves
return a tiny negative roundoff (~1e-16) in some zero-return entries.

All saved checkpoints from 10k to 100k preserve these two versus four signs.
The two recovered runs have negligible final gradients and target lag. Three
failed runs have lagged-target gradient norm at most 6.33e-5; seed 2, rate .001
has norm .0236, so do not claim exact numerical convergence. Its normalized
online/target prediction RMSE is nevertheless only 2.31e-6, and its control
sign is stable. The failed solutions' covariance error is approximately
1.42e-4, close to but distinct from exact orthogonality.

The common exact-target objective in the last column uses the same true target
for every saved model; it is not the TD loss evaluated at different learned
target networks. The wrong TD solutions fit that target better than the two
observed correct TD solutions. This comparison does not certify any learned
solution globally optimal. `spherical_td_search_path4_d2_l500_report.json`
contains all discounts, all final metrics, and the definitions. Multiply the
last column by two for the equivalent two-identical-head coefficient-1000
objective.

The previous density runs for this same path and these discounts have no
backward norm or orthogonality penalty, matching the density defaults. They
are in `analysis/deterministic_didactic/path4_td/` and `path4_td_g08/`; all six
at each discount chose the correct action rankings. They were not rerun simply
because the FB orthogonality coefficient changed.

## Rejected empirical prism candidate

A second, bounded batch used six states A0,A1,A2,B0,B1,B2, actions advancing
around the current triangle and swapping groups at fixed phase, continuation
probabilities (.6,.4), reward at A0, and start A2. The reference remains
uniform since both action maps are permutations. Dimension 2, spherical FB
coefficient 500, density norm off; discounts .4/.7/.9 and the same six
seed/rate combinations, 100k expected updates.

This batch does **not** provide a robust TD separator. FB has mixed control
returns. Density also has mixed returns and appreciable final gradients;
some solutions are still moving. At gamma .4, density achieves optimal return
in 5/6 runs; at .7 in 2/6; at .9 in 3/6. Preserve these negative results.
All predictions, factors, metadata, and executed source snapshots are saved in
`spherical_td_search_prism6_d2_l500/`. No further prism or broad graph sweeps
were launched after the path's constrained theorem became the priority.
