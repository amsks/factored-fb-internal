# Fixed-policy TD validation on a deterministic four-state path

Date: 2026-09-11. These are full-expectation, tabular-factor TD experiments,
separate from the exact-successor approximation theorem. At discount 0.8 and
dimension two, all six FB runs with unrestricted backward norms choose the wall
action at the start; all six categorical runs choose the action toward the goal.
The FB predictions closely approach the squared-loss projection. Restoring
spherical backward normalization rescues two of six start decisions, so the
unrestricted result must not be generalized to that parametrization.

The reusable runner is
[deterministic_td.py](../../scripts/didactic/deterministic_td.py). It is a
formatting-only copy of the executed source, with explanatory comments added;
Python AST equality was checked against the archived source. No new training
was performed while preparing this report.

## Protocol and scope

States are 0, 1, 2, 3 in a line. Left/right move one cell; at a boundary, the
outward action stays in place. Every environment transition is deterministic.
The **fixed continuation policy** chooses left and right with probability one
half, independently of the reward. All eight state-action rows have weight
1/8; all four candidate states have weight 1/4. The candidate reference equals
both the actual next-state marginal and the stationary distribution. Reward is
one on arrival at state 1 and zero elsewhere; the diagnostic starts at state 0.

Both critics use learned tables (F\in\mathbb R^{8\times2}) and
(B\in\mathbb R^{4\times2}), one head, float64, and exact expectations at every
update. They start from independent normal entries with standard deviation 0.1,
using seeds 0, 1, 2 and Adam rates 0.001 and 0.01. The same rate updates both
tables. All six runs per method receive 100,000 updates; predictions are saved
every 10,000 updates. Adam uses the Optax defaults: first/second-moment decay
0.9/0.999, epsilon (10^{-8}), no weight decay. Both target tables start at the
online parameters and update **after** each online update as
(\bar\theta\leftarrow0.995\bar\theta+0.005\theta).

| Evidence | Backward outputs | What it establishes |
|---|---|---|
| Exact-successor approximation theorem | Unrestricted norms; FB second moment can equal (I) | A global approximation objective can discard a representable goal-action distinction. |
| Main TD experiment | FB: unrestricted norms, second-moment penalty coefficient 1. Density: unrestricted norms, no such penalty. | Joint TD learning reproduces the wrong/correct start decisions in this fixed-policy toy. |
| Spherical FB control | Each backward row has norm (\sqrt2); penalty coefficient 1 remains | Two of six start decisions recover; there is no uniform failure claim for spherical FB. |

This is neither a goal-conditioned actor evaluation nor a reward-inference
evaluation. No actor, task-conditioned policy family, reward encoder, behavior
cloning, finite dataset, neural encoder, or ensemble pessimism is trained.
Action scores integrate the known binary reward directly. The two model
families have the same feature dimension but different probability classes;
this comparison does not isolate cross-entropy from model geometry.

## Objective and source alignment

Write (M(s,a,u)=\sum_{t\ge0}\gamma^t
\Pr(s_{t+1}=u\mid s,a,\pi)). FB predicts its density ratio
(m=M/\rho=4M). With stopped target tables and
(\bar m_\pi(s',u)=\tfrac12\sum_{a'}\bar m(s',a',u)), the runner minimizes

\[
\tfrac12\mathbb E_{s,a,u}
[m(s,a,u)-\gamma\bar m_\pi(s',u)]^2
-\mathbb E_{s,a}m(s,a,s')
+\tfrac12\mathbb E_{u,v}[B(u)^\top B(v)]^2
-\mathbb E_u\|B(u)\|^2.
\]

This is the one-head population counterpart of
[FBCritic.stage_loss](../../impls/critics/fb.py#L121), with no auxiliary Q loss
or pessimism. Production's diagonal linear term uses the online-minus-target
difference; the omitted part here depends only on stopped target parameters and
does not change the update gradient. Production excludes equal *minibatch
indices* from its independent-pair estimator. The population calculation
includes independently drawn candidates with equal *state labels*. Averaging
the stochastic continuation action before squaring also changes only a
target-dependent variance term.

Density predicts (q=\operatorname{softmax}(FB^\top)), a normalized successor
distribution. Its target is
((1-\gamma)e_{s'}+\gamma\tfrac12\sum_{a'}\bar q(s',a',\cdot)), stopped before
cross-entropy differentiation. This matches the expectation of the hard
categorical target in
[DensityFBCritic.stage_loss](../../impls/critics/density_fb.py#L950), at
temperature 1. It has no orthogonality penalty or backward normalization,
matching the corresponding defaults in
[density_fb.yaml](../../impls/configs/critic/density_fb.yaml).

The target update order agrees with
[the shared update loop](../../impls/agents/__init__.py#L307) and
[soft_update](../../impls/utils/flax_utils.py#L125). The principal FB departure
is its backward norm: [PhiMap](../../impls/utils/networks.py#L56) normally
projects each row to norm (\sqrt d), and
[fb.yaml](../../impls/configs/critic/fb.yaml) enables that option. The main toy
turns it off; the recovery control restores precisely that normalization.
Penalty coefficient 1 matches `fb.yaml`; it does not match the coefficient
1000 in [fb_ref.yaml](../../impls/configs/critic/fb_ref.yaml).

For reporting, FB reward integration is
(\sum_u\rho(u)m(s,a,u)r(u)=F(s,a)^\top\mathbb E_\rho[Br]), and density
integration is (\sum_u q(s,a,u)r(u)/(1-\gamma)). These put both scores in
ordinary discounted-return units. Production FB uses
[a task-vector readout](../../impls/critics/fb.py#L171); production density
uses [expected candidate rewards](../../impls/critics/density_fb.py#L1092)
without the common (1/(1-\gamma)) scale. The toy tests the known reward
integral, not either production reward-inference procedure. Its fixed uniform
bootstrap policy also differs from their learned, task-conditioned actions.

## Results and convergence checks

The true right-minus-left value gap at state 0 is **+0.647058824**. The unique
rank-two squared projection gives **−0.238429462**. The table reports every
final run; there is no best-seed or best-rate selection. RMSE is measured on
normalized successor entries, ((1-\gamma)M).

| Seed | Adam rate | FB gap | Density gap | FB RMSE from squared projection |
|---:|---:|---:|---:|---:|
| 0 | 0.001 | −0.238425434 | +0.411464569 | (1.95\times10^{-5}) |
| 0 | 0.01 | −0.238420520 | +0.411905482 | (9.34\times10^{-6}) |
| 1 | 0.001 | −0.238423390 | +0.411469739 | (1.27\times10^{-5}) |
| 1 | 0.01 | −0.238358586 | +0.411469895 | (4.74\times10^{-5}) |
| 2 | 0.001 | −0.238425484 | +0.411464744 | (1.88\times10^{-5}) |
| 2 | 0.01 | −0.238377378 | +0.411724319 | (2.67\times10^{-5}) |

At every saved checkpoint from 10,000 through 100,000 updates, all six FB
runs have a negative start gap and all six density runs have a positive start
gap: 60/60 decisions of each sign, across ten repeated checkpoints of six runs.
The ranges over those checkpoints are [−0.239023, −0.237398] for FB and
[+0.409869, +0.413872] for density. These repeated observations are not 60
independent seeds. Density also ranks both actions correctly at every state at
the final checkpoint under the fixed continuation policy.

FB's final backward second-moment error is at most (3.70\times10^{-4}),
and its learned subspace contains the path's first nonconstant eigenvector
within residual (3.80\times10^{-5}). Every final FB predicted entry exceeds
0.05344. Its squared projection has a zero self-target update gradient to
numerical precision. The finite-rate Adam runs still oscillate slightly:
the final self-target gradient norms are (1.51\times10^{-4}) to
(8.00\times10^{-4}), and normalized online/target prediction RMSE is below
(4.74\times10^{-5}). The saved `gradient_norm` field uses the online parameters
as both online and stopped target; it is not the gradient at the lagged target.
Recomputing the latter from saved parameters gives FB norms
(5.26\times10^{-4}) to (2.48\times10^{-3}). The evidence supports stable
wrong decisions near the analytic projection, not exact numerical convergence.

Density TD is also distinct from direct exact-successor fitting. Its final
average KL to the true successor is 0.00338695–0.00338711, whereas the separately
optimized rank-two categorical fit attains 0.00327225378 and start gap
+0.412348867. Both choose right. The direct-fit certificate does not certify
the TD optimization dynamics. Density's final lagged-target gradient norms are
(8.40\times10^{-10}) to (1.41\times10^{-4}), with normalized online/target
prediction RMSE below (9.90\times10^{-5}).

The unrestricted FB greedy policy stays forever at state 0 and earns zero.
The density greedy policy enters state 1 and subsequently alternates between
states 0 and 1, earning (1/(1-0.8^2)=25/9), the optimal return from state 0.
These are exact consequences of the readout's greedy actions, not rollouts
from a learned actor.

With spherical FB backward normalization, seed 0 at both rates instead has
start gap approximately **+0.500000**; seeds 1 and 2 at both rates have gap
approximately **−0.235292**. All six still have one incorrect fixed-policy
action ranking. In the recovered runs it is at state 1: they prefer right,
then alternate between states 1 and 2. This still achieves the optimal return
25/9 from state 0. In the other four runs the incorrect ranking is at state 0,
where they stay forever. Thus the spherical control has two actual control
recoveries and four failures; the ranking error at the goal does not negate
those recoveries.

### What the spherical control does and does not settle

The unrestricted SVD proof does not apply after enforcing equal backward norms.
However, the recovered TD solutions do not establish that the constrained
global optimum succeeds. Evaluating the saved final predictions against the
same **exact** successor target gives squared objective
0.22837370242 for the two recovered runs and 0.08256815439–0.08256822043 for
the four failed runs. The latter fit the full successor matrix better. Including
the actual FB regression scaling and the orthogonality penalty preserves this
ordering: the objectives above the common penalty minimum are 2.8546712803
versus 1.0352625791–1.0352634409. These are common exact-target evaluations,
not comparisons between losses with different learned TD targets. Neither
observed solution is certified globally optimal. The recomputed quantities
are in `analysis/deterministic_didactic/path4_sphere_exact_target_losses.json`.

Spherical normalization also leaves enough capacity for this goal. Let
\(p_i=N_{i,1}=(22,33,18,12)_i/85\), set
\(B_0=B_1=(1,1)\), \(B_2=B_3=(1,-1)\), and
\(F_i=(1,4p_i-1)\). Then
\[
\widehat N_{i,:}=F_iB^\mathsf T/4
=(p_i,p_i,\tfrac12-p_i,\tfrac12-p_i).
\]
Every entry is positive, rows sum to one, every backward norm is \(\sqrt2\),
and \(\mathbb E_\rho[BB^\mathsf T]=I\). The goal column is exact. This is a
feasible control-sufficient witness, not a proof that the spherical objective's
global optimum selects or rejects it. Establishing that optimum remains open.

## Reproduction and artifacts

With NumPy, JAX and Optax available:

```bash
JAX_PLATFORMS=cpu python scripts/didactic/deterministic_td.py --problem path4 --gamma .8 --out analysis/deterministic_path_td/main
JAX_PLATFORMS=cpu python scripts/didactic/deterministic_td.py --problem path4 --gamma .8 --methods fb --sphere --out analysis/deterministic_path_td/sphere
```

Defaults specify all three seeds, both rates, 100,000 updates and target rate
0.005. The runner refuses an output directory containing `config.json`.
The executed environment used NumPy 2.4.4, JAX 0.10.2 and Optax 0.2.8 on CPU.
The archived executed source SHA256 is
`92063666d6d12bb3e5a85398cfe5a4ad78e8338c62bd37ff22fd9d80d78d446f`.

Local artifacts, ignored by Git, are under
[analysis/deterministic_didactic](../../analysis/deterministic_didactic/README.md):

- `path4_td_g08/`: configurations, all checkpoints, final online/target tables,
  exact successors, and squared projections for both methods.
- `path4_td_g08_sphere/`: the six spherical FB recovery runs.
- `path4_td_report.json`: final source hashes, checkpoint ranges, and metrics
  recomputed from saved online and target parameters.
- `path_exact_fit.json`: the separate direct categorical fits.

The note and runner preserve the protocol and empirical findings independently
of those local arrays. Other discounts and rejected graph candidates are indexed
in the artifact README; they are not pooled into this six-run result.
