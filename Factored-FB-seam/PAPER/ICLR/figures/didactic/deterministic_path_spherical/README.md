# Deterministic path: spherical FB certificate

Use [mainmdp.png](mainmdp.png), [mainmdp.pdf](mainmdp.pdf), or
[mainmdp.svg](mainmdp.svg), with [caption.tex](caption.tex). The image shows
only the MDP. No action-value bars from the earlier unrestricted construction
are reused. That earlier figure package is preserved separately.

The proof is
[the spherical-counterexample note](../../../../../docs/notes/2026-09-11-deterministic-path-spherical-counterexample.md).
The [exact verifier](../../../../../scripts/didactic/verify_deterministic_path_spherical.py)
passes all assertions; its saved output is [verification_data.json](verification_data.json).
The independent rational-matrix audit is
[recorded here](../../../../../analysis/spherical_symmetric_certificate_audit.json).
[certificate_summary.json](certificate_summary.json) records the conservative
bound and the separate feasible density candidate used by this caption.

## MDP and interpretation

There are four states in a line, with deterministic Left and Right actions.
An outward action at an endpoint stays in place. Start is state 0; reward is
one whenever the next state is state 1, and zero otherwise. Goal is continuing:
its actions lead to state 0 or state 2. The continuation policy used to define
successors chooses either action with probability one half. This stochastic
policy does not make the environment dynamics stochastic.

The discount is .8. Each of the eight state-action rows has weight 1/8. Their
next-state marginal, the stationary distribution, and the candidate reference
are all uniform over the four states. The target, conditional on immediate
successor i, is row i of

```
N = 1/85 * [[43,22,12, 8],
            [22,33,18,12],
            [12,18,33,22],
            [ 8,12,22,43]].
```

Action values are obtained by integrating the known goal reward. Normalized
values are the discounted reward integral multiplied by `1-gamma=.2`.
The exact normalized Right-minus-Left gap at Start is `11/85`, approximately
+.129411765. Greedy actions optimize the predicted reward integral.

## What is certified

Both models have feature dimension two. FB backward outputs have fixed norm
sqrt(2); its second moment is encouraged by a **soft** orthogonality penalty.
The coefficient is 1000 with two forward heads, matching the modern campaign's
[FB configuration](../../../../../impls/configs/critic/fb_tdj.yaml).
Density uses rank-two bilinear logits, normalized by softmax, with unrestricted
backward norms and no orthogonality penalty, matching its implementation.

Every global regularized squared exact-target FB fit has normalized start gap
at most **−.02875**, so greedy execution remains at Start and earns zero.
This is a conservative certified inequality, not a displayed numerical optimizer
or a reuse of the unrestricted SVD solution. The proof uses a feasible squared
error budget .081 and a rational positive-definite matrix certificate. It does
not assert that the backward covariance is exactly the identity at the optimum.

Dimension two still suffices for this reward. If `p_i=N[i,1]`, take backward
rows `(1,1),(1,1),(1,-1),(1,-1)` and forward row `(1,4*p_i-1)`. The predicted
normalized probabilities are `(p_i,p_i,.5-p_i,.5-p_i)`. They are positive,
sum to one, have the exact goal column, and satisfy both the spherical norm
and identity backward second moment.

A feasible rank-two categorical model has mean KL .003304199258, below the
wrong-decision lower bounds .004806898167 and .020199509967 at states 0 and 2.
Consequently every global categorical optimum enters Goal from those states;
either Goal action returns through one of them. The resulting start return is
optimal, `25/9`. The explicit candidate's normalized start gap is about
+.0839262; this is a **feasible candidate's** gap, not a claimed optimum value.
Both full target families are misspecified: N has rank four and its centered
log-probability matrix has rank three. The comparison changes model geometry
as well as the fitting loss; it is not a cross-entropy-only isolation.

These are fixed-policy exact-successor approximation statements. They do not
assert global optimality or convergence for joint semi-gradient TD, a learned
actor, or production reward inference. Separate spherical TD checks are
[indexed here](../../../../../analysis/spherical_td_search_report.md): with
one-head coefficient 500, giving the same relative exact-target objective
weight as two identical heads and coefficient 1000, two of six runs recover
optimal control and four fail. Those are distinct training outcomes, not a
contradiction of the global exact-target theorem.

## Generate the graph

From the repository root, with NumPy and Matplotlib:

```bash
python PAPER/ICLR/figures/didactic/deterministic_path_spherical/draw.py
```

The self-contained generator copies the graph geometry of
`../deterministic_path/draw.py` and exports only the MDP under the new name
`mainmdp`. It does not run training, fit parameters, import the earlier value
panel, or modify paper source files. PNG has a white background; PDF keeps
vectors and embeds fonts; SVG keeps text. This folder is under the ignored
`PAPER/` tree, so files being present does not imply they are tracked in Git.
