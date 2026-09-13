# Audit of the corridor k40 figures

2026-09-10. Scope: `PAPER/ICLR/figures/corridor/k40/`, its generator and saved
data, and its suitability for the current ICLR paper. The paper and original
figures were not edited.

## Verdict

The numerical maps and reported gaps reproduce. The example demonstrates that a
particular rank-8 squared-error fit to exact successor targets loses much of the
action-value difference at the exit, while a rank-8 softmax model fit by cross-entropy
preserves it. It is useful as an **approximation example with those qualifications**.

It does not currently establish the paper's specific explanation that large
successor targets for persistent states dominate FB's training, and it does not
demonstrate a learned FB policy failing to exit. The reference-distribution
weighting contributes substantially to the effect, the displayed FB fit still
selects the correct action, and the separately saved TD results do not reproduce
the displayed ranking of the methods.

## What was checked

- Independently constructed the 39-state grid, all four transition matrices, the
  uniform-policy transition matrix, its stationary distribution, and the exact
  successor matrix by solving the discounted linear system. These agree with
  the generator to numerical precision for k = 1, 5, 10, 20, 40.
- Confirmed successor row sums equal 50 at discount 0.98. A full-rank FB fit
  reconstructs the k40 successor matrix with maximum absolute error 4.1e-14.
- Independently recomputed the weighted SVD and its reward readout.
- Re-ran all five k40 rank-8 density fits with the generator's seed construction,
  3,000 Adam steps, learning rate 0.01, and one initialization per seed. Their
  normalized gaps agree with `corridor14.csv` within 1.45e-6.
- Recomputed the deterministic 13-step approach and both future-state maps, their
  errors, and the signed contrast. Checked the printed door/room fractions.
- Re-rendered the individual panels, contrast maps, colorbars, and gap curve into
  `/tmp/corridor-audit/rendered/k40/`; inspected the original stitched map and
  gap curve. Raster outputs are not byte-identical in this environment. Individual
  map mean absolute channel differences are below 0.48 on a 0–255 scale; the
  underlying numbers match. This is numerical reproduction, not a claim of
  byte-identical graphical output.
- Read the separate TD generator and summarized its existing five-seed CSV.
  Did not rerun its 100,000-step TD training.

Scratch reproducibility files from this audit: `/tmp/corridor-audit/check.py`,
`report.json`, `arrays.npz`, and `check.log`. The script imports the existing
generator, uses locally cached JAX/Optax packages, writes only under `/tmp`,
and separately constructs the transition model for comparison.

## Reproduced measurements

At k40, rank 8, the exact value gap between right and the mean of up/down/left
at the exit is 3.83159. FB's gap is 0.143892 times that value. The five density
ratios are 0.890688, 0.879673, 0.903184, 0.900233, 0.903616, averaging about
0.8955. All six fitted models prefer right; the gap has not changed sign.

The maps show a deterministic walk from the first corridor cell, one selected
action at the exit, then uniform-random actions forever. The displayed failing
action is left. Fractions below are normalized by the displayed row's total,
matching the figure labels:

| Estimator | Exit: door / room | Fail: door / room | Contrast L1 / true contrast L1 |
|---|---|---|---|
| Exact | 0.06756 / 0.71322 | 0.12043 / 0.65158 | 1.00000 |
| FB | 0.04745 / 0.73225 | 0.05086 / 0.71928 | 0.23308 |
| Density, seed 0 | 0.07212 / 0.70869 | 0.11830 / 0.65364 | 1.01713 |

The exact deterministic prefix is composed with the learned exit-state rows;
the models do not independently predict the whole approach from the starting dot.
The prefix cancels in the difference between the two maps.

The six displayed raw rows are all nonnegative, so clipping negatives in the
plotter does not manufacture this visual difference. FB's row sums are 1.00056
and 1.00176, rather than exactly one; the discrepancy is small but makes
"half-L1 error" more precise than probability-distribution TV for those raw rows.

Sources: [generator](../../analysis/corridor_estimator/run.py),
[saved sweep and map statistics](../../analysis/corridor_estimator/corridor14.csv),
[figure caption](../../PAPER/ICLR/figures/corridor/k40/subfigures.tex).

## The fitted objective and why weighting matters

Let M contain exact successor occupancies, and let rho be the stationary
next-state distribution. The generator factorizes the density ratio M/rho. It
performs truncated SVD on `M / sqrt(rho)` and converts the result back to
occupancies. Consequently it minimizes

```
mean_over_uniform_state_action_rows sum_u (M_hat(s,a,u) - M(s,a,u))^2 / rho(u).
```

This is a valid supervised, weighted low-rank approximation problem. It is not
the full FB TD objective. The input rows are uniform over all state-action pairs,
whereas the TD script samples those rows from a dataset. The SVD also fits the
exact long-horizon answer directly instead of using a target network and a
one-step Bellman update. Calling it "FB's population optimum" without stating
this surrogate problem overstates what was established.

As k rises, the chance of moving left out of the exit decreases. This makes
the preceding corridor much less common under the stationary random policy:
its stationary mass falls from 1/3 at k1 to 0.01235 at k40. At k40, those
13 preceding corridor columns account for **97.06% of the squared norm of
the matrix sent to SVD**. The exit column accounts for 0.19%, and the room
columns for 2.74%. This is target energy, not a direct measurement of training
gradients or proof of which individual singular component causes the loss.

Controls recomputed at the same rank, dynamics, and reward:

| k | Original weighted SVD | Uniform reference distribution | Stationary input-row weights |
|---|---|---|---|
| 1 | 0.93277 | 0.93277 | 0.93277 |
| 5 | 0.64345 | 0.74186 | 0.69796 |
| 10 | 0.49773 | 0.84802 | 0.54875 |
| 20 | 0.38403 | 0.82691 | 0.41339 |
| 40 | 0.14389 | 0.81359 | 0.31279 |

Entries are fitted/true action-value gap. The middle control replaces rho with
a uniform reference in both the supervised ratio fit and its readout; in
occupancy space this is ordinary unweighted squared error. It is a diagnostic,
not a claim that this is the original FB training distribution. The right
control retains the original reference and additionally weights input rows by
rho(s)/4, matching stationary behavior sampling. Neither control is a TD run.

The first control recovers most of the lost gap without changing the dynamics,
rank, or using a categorical model. Therefore this sweep does not isolate
increased persistence as the cause of the original curve. Reference weighting
and capacity allocation require explicit discussion.

The paper's categorical model also changes the model class: a rank-8 logit
matrix followed by softmax can have an occupancy matrix of rank greater than
8. Matching feature dimension or parameter count does not hold the set of
representable successor matrices fixed. The example tests the combined model
and fitting objective; it does not isolate cross-entropy alone.

## What the policy and reward actually are

The exit action always succeeds. Under the uniform continuation policy it is
selected with probability 1/4 at each visit to the exit. Increasing k from 1
to 40 changes expected consecutive time at that cell from 2 to 3.902 steps,
not from 1 to 40. Up and down already hit walls; only left's transition
actually changes with k. After the one failing action in the lower row, the
agent can retry and enter the room. The room is neither permanently inaccessible
after failure nor an absorbing terminal state.

The plotted value gap uses reward **one on every room cell**, not the starred
corner. Both rows eventually spend substantial discounted time in the room.
The star is only a marker of a far room cell, as the caption says, but it
visually suggests a goal-reaching experiment that was not performed.

Thus the plots illustrate lost sensitivity to a one-step action's consequences
under a fixed continuation policy. They do not show trajectories of learned
policies, zero-shot task inference, or inability to enter the room.

## Separate TD evidence

The later [TD script](../../analysis/corridor_estimator/trained_k40_maps.py)
uses the same k40 grid and rank, a 100,000-transition random-policy dataset,
100,000 updates, and five seeds. Its [saved CSV](../../analysis/corridor_estimator/trained_k40.csv)
reports mean normalized gap 0.8973 for FB and 0.6414 for density. FB's values
range from 0.7492 to 1.0000, and density's from 0.6102 to 0.6662. Both prefer
the exit in every seed.

These numbers are not a controlled contradiction of the supervised-fit result:
they optimize different objectives, sample input rows differently, and introduce
dataset approximation. In particular, resets every 500 transitions mean the
dataset is not stationary. Freshly regenerated next-state histograms for those
five datasets differ from the stationary rho by L1 distances 0.220–0.276.
The saved TD script nevertheless uses stationary rho at readout for both
methods. The empirical-reference diagnostic scripts address this issue, but
their presence alone is not proof of a validated result.

However, the saved TD evidence is enough to reject presenting the k40 SVD map
as a demonstrated failure of TD-trained FB. It must be discussed or resolved
before claiming that the training algorithm reproduces the toy mechanism.

## Paper and presentation changes needed before use

1. Label the columns "exact successor", "rank-8 squared-error fit", and
   "rank-8 categorical fit". State that both are fit to exact successor
   targets under a fixed random continuation policy.
2. Replace "closed-form fits" and "CE optimum" with "fits to exact targets"
   and "optimized categorical fit". Only FB's SVD optimum is available in
   closed form; 3,000 Adam steps do not certify a global density optimum.
3. Describe the finding as **underestimation of the exit action's advantage**.
   The current title "does the critic prefer exiting?" has the answer yes for
   both methods. A more accurate title is "How much of the exit advantage is
   preserved?" The map contrast and right-hand curve are different metrics:
   L1 contrast of left versus right from the start, versus reward-integrated
   right-minus-mean-other-actions at the exit.
4. Shade or label the whole room as rewarding; remove the goal-like star or
   explicitly label it as having no special reward. Explain the failing left
   action and the random continuation. Consider showing maps directly at the
   exit so the exact, hand-composed approach does not distract.
5. Keep the common color scale and add a clear explanation of the signed
   contrast. Prefer a cellwise rendering for this tabular example. The current
   maps use spatial smoothing and a power color scale; that is display-only,
   not a mass-preserving transformation of the tabular probabilities.
6. Include the weighting control and acknowledge the TD results. Do not use
   this example alone to justify the specific "large persistent-state targets
   dominate FB training" claim in [the preliminaries](../../PAPER/ICLR/content/prelims.tex).

Suggested claim:

> With limited representation size, fitting successor targets can preserve the
> broad future-state distribution while substantially underestimating the value
> difference between two actions. In this corridor example, a categorical fit
> preserves that difference more accurately than the specified weighted
> squared-error fit.

The figure currently is **not included** by the paper's active LaTeX inputs.
Its generated block requires `subcaption`, which the current main preamble does
not load. These are integration tasks, separate from numerical correctness.
