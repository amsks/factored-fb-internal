# Didactic critic diagnostics

The deterministic four-state path check is implemented in
[`deterministic_td.py`](deterministic_td.py), with the measured results and
production-source audit in
[the September 11 TD note](../../docs/notes/2026-09-11-deterministic-path-td.md).
It uses a fixed uniform continuation policy, exact population updates, dimension
two, and exact reward integration. The main FB comparison disables backward-row
normalization; its spherical control includes recovered start decisions. The
exact-successor theorem and empirical TD validation are distinct claims.

## Carry-or-drop critic diagnostic

Run from the repository root with Python, NumPy, JAX, Optax and Matplotlib:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s scripts/didactic -v
JAX_PLATFORMS=cpu python scripts/didactic/carry_drop.py --out analysis/carry_drop/expected
JAX_PLATFORMS=cpu python scripts/didactic/carry_drop.py --out analysis/carry_drop/sampled --samples 16
MPLCONFIGDIR=/tmp/carry-drop-mpl python scripts/didactic/plot_carry_drop.py --source analysis/carry_drop/expected
MPLCONFIGDIR=/tmp/carry-drop-mpl python scripts/didactic/plot_carry_drop.py --source analysis/carry_drop/sampled
```

The runner refuses to overwrite a run directory containing a configuration.
Use a new `--out` path for repetitions. Plotting reads saved results and can be
repeated independently. The plotting title assumes the declared 100k-update,
three-seed sweep; do not use it unchanged for a different protocol.

## Question and design

Does a persistent, uncontrollable resting state cause a full-capacity FB critic
to lose the action differences at a chain of transport decisions? This is an
experimental candidate, not an MDP on which FB is mathematically unable to solve
the task. Exact representability and zero semi-gradients are tested explicitly.

There are L held states, goal G and rest R. Advance succeeds with probability .9
and drops with probability .1; deliberate drop always reaches R. G resets to H0
on the next step. R returns to H0 with probability 1/k and otherwise stays.
Both actions have the same effect at G and R. The process continues indefinitely.
Reward is +1 on arrival at G and zero otherwise. Discount is .98.

The sweep crosses L={1,3,6}, k={1,10,50}, seeds={0,1,2}, and Adam learning
rates={.001,.01}. Each run has 100,000 updates with target EMA coefficient .005.
There are N=L+2 latent dimensions for N states, with learned tabular F(s,a) and
B(u). No rank restriction excludes the exact answer. Each state-action has equal
weight, and all N distinct candidate states are enumerated with uniform reference
rho=1/N. The continuation policy takes advance and drop with probability 1/2.

The expected sweep integrates next-state transitions exactly **at every TD
update**. It still bootstraps from a learned target network; it is not supervised
regression to exact successor targets. The sampled sweep draws 16 independent
next states from **each** state-action pair at every update, with identical draws
across methods and learning rates for a given seed and environment. Continuation
actions are averaged exactly in both sweeps. These are fresh generative samples,
not a finite offline dataset. Candidate frequencies and state visitation cannot
starve a particular state in this diagnostic.

## Objectives and reference

Let M(s,a,u)=sum_{t>=0} gamma^t Pr(s_{t+1}=u | s,a,pi). Its row mass is
1/(1-gamma). The exact reference solves M=P+gamma P M_pi by linear algebra,
separately for every environment. Changing k changes the true MDP and reference.

- **FB:** F B^T predicts the density ratio M/rho=N M. B rows are normalized to
  length sqrt(N), as in the production backward-network normalization. Expected
  squared TD minus the immediate linear term is used, with orthogonality
  coefficient 1. The independent uniform-candidate expectation of the
  orthogonality loss is `.5 E[(B(u)^T B(v))^2] - E[||B(u)||^2]`.
- **FB without orthogonality:** the same model with that coefficient zero.
- **Density FB:** unconstrained F/B, softmax(F B^T), temperature 1. The
  cross-entropy target is `(1-gamma)P + gamma P p_target_pi`.
- **Normalized squared:** FB's factorization and orthogonality, predicting
  `(1-gamma)M` directly with squared TD. This is an amplitude control, not a
  softmax distribution: entries and total mass are unconstrained.
- **Direct squared table:** a free table predicting N M, with the same squared
  semi-gradient as FB and no factorization or orthogonality.

For squared methods, integrating out the next-state draw before computing the
square removes a target-only variance term. Because the target is stopped, this
does not change the expected semi-gradient. We include all candidate-state pairs:
identical state labels drawn independently are legitimate pairs, unlike excluding
the diagonal observation indices in the production minibatch estimator.

Exact FB parameters can use B=sqrt(N) I and F=sqrt(N) M. Exact density parameters
can use B=I and F=log((1-gamma)M); all entries of M are positive here. The tests
check all five parameterizations and their training gradients at the reference.

These are **single-policy, single-head critic diagnostics**. Production FB uses
task-conditioned F, two heads and learned actions; production Density FB also
has task/reward encoding and a finite minibatch candidate pool. Those mechanisms,
neural feature sharing, dataset coverage, and actor optimization are intentionally
outside this experiment. Density readout directly integrates the known goal
reward; reward inference and the default centered latent reward are not tested.
All calculations use float64. F and B start with independent normal(.0,.1)
entries (FB then normalizes B); the direct table also starts normal(.0,.1).
These initializations do not give identical successor predictions across methods.

## Metrics and artifacts

`summary.csv` contains every final run, with no best-seed or best-rate selection.
Per-method CSVs contain learning curves every 1,000 updates. Compressed NPZ files
contain all predicted rows, true successors, kernels, and final online parameters.
`config.json` records versions, options and the training-source SHA256.
`capacity.json` records exact representation error and successor matrix rank.

The main metric is the predicted advance-minus-drop value gap at H0 divided by
its exact gap under the fixed continuation policy. We also record every held-state
gap via minimum/mean ratios, count nonpositive gaps, measure column-wise errors
and total mass, and evaluate one greedy policy improvement by an exact solve.
That greedy policy's return is compared with the always-advance optimum for this
MDP. A zero gap counts as an unresolved decision even though deterministic argmax
breaks the tie toward advance. We do not train an actor or claim an actor rollout.

Figures are written to `PAPER/ICLR/figures/didactic/carry_drop/`. This directory
and the result directories are ignored by Git in this project; source and the
durable findings note remain separate from those local artifacts.

On the execution machine, the older worktree's Python 3.11 environment had NumPy
and Matplotlib. JAX 0.10.2 and Optax 0.2.8 were loaded from the existing local uv
cache using `/tmp/carry-drop-python.py`; no packages were installed. The saved
configuration provides the exact numerical package versions.
