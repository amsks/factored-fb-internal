# Two finite-capacity objective counterexamples

The primary previews are [natural_reference_counterexample.png](natural_reference_counterexample.png)
and [loss_only_counterexample.png](loss_only_counterexample.png). Each has a
matching PDF and editable SVG. These are mathematical population constructions,
not trained-agent or manipulation benchmark results.

## Reading the figures

Each MDP has two equally likely initial configurations, two actions, and the
same four terminal outcomes. The terminal nodes are repeated between panels
for readability. Action **A** means `a = +1`; **B** means `a = -1`.
The four columns are O1, O2, **Goal**, and O4. Goal is the **third outcome**
(zero-based column `2` in the scripts); it pays reward 1 on arrival, while the
other three outcomes pay 0. The discount is zero and the episode ends after
one action. Each action arrow enters a row of probabilities for sampling one
of the terminal outcomes above the columns. These percentages are the true
transition probabilities, not critic predictions.

Source–action pairs are sampled uniformly. With zero discount, the critic's
target is the immediate next-outcome distribution; the sampled categorical
target is the one-hot observed outcome. FB fits its rank-two successor
density-ratio prediction with the population squared TD objective. A categorical
critic minimizes cross-entropy. There is no bootstrap. A greedy decision
maximizes the predicted reward integrated against the outcome distribution;
no actor or reward encoder is trained in this illustration.

In the value panel, numbers are probabilities, and
`Δ = value(A) − value(B)`. “Optimal FB values” are the unique global squared-loss
prediction computed from the truncated SVD, subject to the stated assumptions.
“Explicit density model” is an analytically specified **feasible** model;
its numerical values are **not claimed to be the global cross-entropy optimum**.
The certificate proves the actions of every global cross-entropy optimum without
computing its probabilities. “Actual success” evaluates the selected action
under the **true MDP**, then averages the two equally likely starts. It is not
the predicted value averaged across starts.

## Natural reference and spherical backward features

This figure uses the construction in
[the objective proof](../../../../../docs/notes/2026-09-10-objective-counterexample.md).
The reference is the natural uniform next-outcome marginal. Both compared
factorizations have two-dimensional features with each backward feature norm
equal to `sqrt(2)` and backward second moment equal to the identity.

The true goal gap is `−0.04` in each context, so B is optimal. The unique FB
loss optimum predicts `+0.01`, choosing A instead. The displayed feasible
categorical model predicts gaps `−0.0499` and `−0.0301`, choosing B.
Actual success is 23% for the FB decisions and 27% for the optimal decisions
and for every global categorical cross-entropy optimum.

There is also a feasible FB representation satisfying the same backward
constraints that preserves the true action-value difference for **every reward**.
It is included in `verification_data.json` as `all_reward_control_comparator`:
its squared loss is 0.0801, compared with the optimum's 0.0101. Thus the example
does not merely assert insufficient dimension for this control task.

The two feature families define **different probability classes**. The explicit
categorical model has rank-two logits and rank-four probabilities, whereas FB
has rank at most two in probability space. Squared loss over this same softmax
class also chooses both actions correctly: the feasible squared error
0.00039204 is below the universal wrong-order lower bound 0.0008. This is not
a separation attributable to cross-entropy alone.

## A loss-only separation in a shared probability class

This figure uses the `strict` construction in
[the loss-only proof](../../../../../docs/notes/2026-09-10-loss-only-counterexample.md).
Squared loss and cross-entropy are compared within the same stochastic
rank-two probability class. Backward outputs are unrestricted. The query
reference is independently chosen to be uniform, while the actual outcome
marginal is `(0.49, 0.49, 0.01, 0.01)`. This uniform reference makes the FB
population objective proportional to ordinary squared probability error.

The true and displayed feasible-model gaps are `+0.018`. The unique squared
optimum predicts `−3.52682571 × 10⁻⁵`: a strict reversal, not a tie. True optimal
decisions and every global cross-entropy optimum yield 1.9% actual success;
the FB decisions yield 0.1%. These are the exact probabilities of this small
constructed MDP. They are not broad trained-method success rates.

The explicit feasible density model has only two distinct probability rows.
It belongs to the shared stochastic rank-two probability class and also admits
rank-two logits. The cross-entropy certificate therefore applies to both
classes separately. It preserves action-value differences for every reward.
The spherical production constraint is not covered. Replacing the chosen
uniform reference with the natural marginal restores correct squared-fit
ordering, as the source verification output records.

## Why the certificate covers every optimum

Let `pA` and `pB` be the true goal probabilities at one context and
`m = (pA + pB) / 2`. Coarsening each predicted distribution into Goal versus
non-Goal cannot increase forward KL. Among reversed or tied predictions, the
smallest two-row Bernoulli KL is attained at a tie, with both predicted goal
probabilities equal to `m`. Since each row has weight 1/4, any wrong or tied
decision at that context has mean KL at least

`[kl(pA || m) + kl(pB || m)] / 4`.

The minimum of the two context bounds applies to any incorrect decision:

| Construction | Feasible mean KL | Wrong/tied bound, configuration 1 | Wrong/tied bound, configuration 2 |
| --- | ---: | ---: | ---: |
| Natural reference | 0.0002064994354 | 0.0004742086049 | 0.0006318215591 |
| Loss-only | 0.0011691237635 | 0.0029330986417 | 0.0022145518367 |

A global cross-entropy optimum cannot have greater KL than a feasible model.
The strict inequalities therefore rule out every wrong or tied decision in
either context. Displayed KL values in the figures are rounded; the checks use
full numerical precision. Full proof arguments and scope are in the linked notes.

## Reproduction

From the repository root:

```bash
/home/adityamohan/git/Factored-FB/.venv/bin/python \
  PAPER/ICLR/figures/didactic/objective_counterexamples/draw_counterexamples.py
```

Any Python environment with NumPy, SciPy, and Matplotlib can run the generator.
It invokes both existing source verification scripts with the same interpreter,
loads their JSON output, computes the plotted decisions and actual success
under the true transitions, checks the strict reversals and KL certificates,
then writes all six image exports and `verification_data.json`. No parameter
search or density fit is added by the generator. Both PNGs were inspected for
clipping, overlap, legibility, and numerical agreement with the source output.

The figure directory is under the repository's ignored `PAPER/` tree; these
are local artifacts until explicitly included in version control.
