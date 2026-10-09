# Deterministic didactic searches and population TD checks

Created 2026-09-11. All artifacts here are local research outputs. These are
fixed-policy, full-expectation, tabular-factor calculations without an actor.
They do not reproduce the production task-conditioned implementation.

## Main current candidate: four-state reflecting path

States 0,1,2,3. Left/right actions move one cell; attempting to leave the path
stays at its endpoint. Environment dynamics are deterministic. The continuation
policy chooses left/right with probability one half at every state. Uniform
state-action sampling has uniform next-state marginal, which is also the
stationary reference. The reward is one on arrival at state 1 and zero elsewhere.
The diagnostic starts at state 0. Feature dimension is two.

At discount 0.8 the unique rank-two weighted squared-loss fit has normalized
right-minus-left value gap -0.0476858924576, whereas the true gap is
+0.129411764706. In ordinary discounted-return units these are -0.238429462288
and +0.647058823529. The FB fit has positive entries. It chooses the wall action
at state 0 and earns zero, while taking right and subsequently returning to state
1 every two steps earns 1/(1-0.8^2) = 25/9.

The categorical exact-successor projection has numerically obtained average
KL 0.00327225377558 and raw action gap +0.4123488668. Its rank-two factors are
saved in `path_exact_fit.json`. The target probabilities have rank four and the
centered log target has rank three, so both model families are misspecified.
A global categorical certificate using the duplicate-row structure is developed
separately by the theory audit; the simple per-state-action-row Bernoulli bound
in this directory is smaller and by itself does not certify gamma 0.8.

**Constraint scope:** the exact squared optimum has backward second moment I
but unequal row norms. The main path TD check therefore uses unconstrained
backward outputs and orthogonality penalty weight 1. It does not establish a
uniform failure under the production spherical backward parametrization.

`td_check.py` implements full-expectation one-head joint F/B TD with Adam and
Polyak targets (tau 0.005), uniform candidates, exact continuation-action
averaging, and float64. FB predicts the unnormalized successor density ratio;
its loss is the usual squared off-diagonal term plus the immediate successor
term and second-moment penalty. Density predicts normalized successor
probabilities and minimizes cross-entropy against the normalized immediate plus
bootstrapped target. No actor is learned. Source hashes and package versions
are stored with every run.

- `path4_td/`: gamma 0.5,0.9,0.99; three seeds, rates 0.001/0.01, 100k steps.
- `path4_td_g08/`: gamma 0.8, the same six runs per method, 100k steps.
- In these 24 runs per method, every FB run chose the wrong start action and
  every density run had the correct fixed-policy action ranking at all states.
- At gamma 0.8, final raw FB gap is -0.2384255 to -0.2383586; density gap is
  +0.4114646 to +0.4119055. FB prediction RMSE from the exact SVD fit is below
  4.75e-5. Density TD is distinct from the exact categorical projection: its
  exact-target KL is about 0.00338710 versus 0.00327225 for the direct fit.
- `path4_td_g099_long/`: six FB runs restarted with the same seeds/rates for
  300k steps because the 100k predictions had not fully converged. All still
  fail. Final SVD RMSE is 2.3e-6 to 9.2e-5; Adam leaves some small oscillation.
- `path4_td_g08_sphere/`: important control with row normalization restored.
  Seed 0 at both rates chooses the correct start action (gap about +0.5);
  seeds 1 and 2 at both rates fail (gap about -0.235292). Do not omit this
  recovery control or generalize the unconstrained-B result to spherical FB.

Example command (the wrapper exposes cached local JAX/Optax installations):

```
JAX_PLATFORMS=cpu ../Factored-FB/.venv/bin/python /tmp/carry-drop-python.py analysis/deterministic_didactic/td_check.py --out /tmp/path4-recheck --gamma .8
```

## Clean four-cycle baseline and its limitation

`verify_cycle4.py` and `verify_cycle4.json` verify a deterministic cycle with stay
and advance actions, deterministic advance continuation, uniform stationary
reference, and dimension three. At gamma 0.5, state 1 and goal 0 have normalized
stay/advance values (1/15,2/15), while the unique squared projection predicts
(0.15,0.05). The features for that optimum and a centered FB comparator both
satisfy row norm sqrt(3) and second moment I. The centered comparator preserves
all action contrasts for every reward. Density dimension three represents the
entire target exactly, so this baseline does not meet the stronger requirement
that both families misspecify the successor distribution.

`cycle4_td/` additionally checks joint TD with spherical FB features. Some runs
recover a centered, control-sufficient representation rather than the SVD
projection. The exact-projection failure must therefore not be stated as a
general failure of TD learning on the cycle. Density fits the exact target.

## Other searches, including failures

- `search.py`, `svd_all_cases.json`: all 240 initial chain, fork, and cycle
  checks with 5–9 states, eight discounts, and dimensions two/three. The smaller
  `svd_search.json` records cases with strict wrong goal rankings. Singular
  boundary and spherical feasibility matter; some dimension-two cycle fits
  split a degenerate Fourier pair and are not unique.
- `fitted_candidates.json`: first 16 direct categorical fits selected by the
  largest potential wrong-decision bound. Chain/fork cases certify easily at
  small discount but have nonstationary uniform reference and unequal feature
  norms. They are less satisfactory than the path construction.
- `cycle_fits.json`, `cycle_refine.json`: all attempted larger cycle density
  fits. Five-cycle dimension-three fits typically fail the global certificate
  and sometimes make a wrong action ranking. Do not present them as a guarantee.
- A seven-cycle at dimension five and discount 0.1 has a direct-fit certificate:
  compare initial successors 0 and 3 for goal 2. True normalized gap is
  +0.0089991009; squared-fit gap -0.3467748243; feasible density KL
  0.00006057698 versus two 1/14-weight rows' wrong bound 0.00044643552.
  Both probability rank seven and centered-log rank six exceed dimension five.
  FB features can be spherical/orthogonal; fitted density features were not
  constrained to be spherical. The squared projection can be negative, and
  no TD claim was checked for this candidate.
