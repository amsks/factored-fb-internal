# FourRooms tabular estimator demo: FB (squared loss) vs Density (softmax / cross-entropy)

Everything here is a small, closed-form tabular experiment (no learned policy, no neural
net beyond the two rank-`d` fits themselves). The goal is a controlled demonstration of a
specific claim: **FB's rank-`d` fit is a rho(s')-weighted *squared-loss* fit to the ratio
M(s,a,s')/rho(s'), and a squared loss under-penalizes small probabilities, so it blurs
action distinctions at a "sticky" bottleneck. A normalized rank-`d` fit trained with
cross-entropy (call it "Density") does not have this failure mode, because cross-entropy
penalizes getting a small probability wrong on a log scale.**

Run with:

```bash
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/fourrooms_estimator/run.py
```

Takes about 100 seconds on CPU (JAX). Produces `gap.csv`, `occupancy.csv`, and the paper
figure under `PAPER/ICLR/figures/fig_fourrooms_estimator.{pdf,png}`.

## Environment

Standard 11x11 FourRooms grid. The state space is the 104 free cells of an 11x11 grid
with an interior "+" wall (row 5 and column 5 are walls) punched with four single-cell
doorways, one per arm, connecting the four rooms:

- `(2,5)`: top-left <-> top-right (this is the bottleneck **D** used throughout)
- `(8,5)`: bottom-left <-> bottom-right
- `(5,2)`: top-left <-> bottom-left
- `(5,8)`: top-right <-> bottom-right

4 deterministic actions (up/down/left/right); bumping into a wall or the grid edge means
"stay". The behavior policy is uniform random over the 4 actions everywhere.

**Sticky-door rule (parameter k).** At D, the action that leaves D into the next room
(`a_cross` = "right", into the top-right room) always succeeds, unmodified by k. Each of
the other three actions (`a_fail` = up/down/left) keeps the agent at D with probability
`1 - 1/k` and otherwise follows the normal (k=1) dynamics. At k=1 this is a no-op (no
stickiness). Note that at D, "up" and "down" are *already* wall-bumps under normal
dynamics (D sits on the vertical wall arm, so only left/right are physically open) --
stickiness only changes anything for the "left" (back-into-the-room) action; up/down are
"stay" regardless of k. This is intentional: stickiness only has to act on the one action
that would otherwise carry real signal.

Reward `G` (used for the reward-conditioned metrics and reused as the eps-greedy target
for the directed-policy experiment): the deep corner cell `(4,10)` of the top-right room.
Start state `S0 = (2,2)`, three cells before the door along the same corridor row.

## Exact quantities (all closed form, numpy)

- `P0[a]`: deterministic k=1 per-action transition matrices.
- `P_k[a]`: per-action transition matrices with the sticky rule applied only at row D.
- `P_pi(k) = mean_a P_k[a]` (uniform random policy).
- `rho`: stationary distribution of `P_pi(k)`, by power iteration (converges in 1 step at
  k=1, ~4100 steps for k>1; L1 delta < 1e-14 at convergence in every run).
- Successor measure `M(s,a,s') = [P_k[a] (I - gamma P_pi(k))^{-1}]_{s,s'}`, built as one
  `(|S||A|, |S|)` matrix. **Sanity-checked every run**: every row of M sums to
  `1/(1-gamma)` to machine precision (max error ~1e-13).

## FB fit (population optimum, closed form)

FB regresses `R(s,a,s') = M(s,a,s')/rho(s')` with a squared loss weighted by `rho(s')`.
The rank-`d` population optimum of a rho-weighted squared loss is the truncated SVD of
`R diag(sqrt(rho))`: scale columns by `sqrt(rho)`, take the top-`d` SVD, unscale. This
gives `F (S*A, d)` and `B (S, d)` with `F(s,a) . B(s') = R_d(s,a,s')` exactly (not
`M_FB` -- the extra `rho(s')` factor is applied at readout time, matching how FB's
`z = sum rho(s') r(s') B(s')`, `Q_FB(s,a) = F(s,a).z` readout works in practice).

## Density fit (rank-`d` softmax, cross-entropy, Adam via optax)

Target rows `m(s,a,.) = (1-gamma) M(s,a,.)` (a genuine probability distribution, since
`M` rows sum to `1/(1-gamma)`). Model `p_theta(s'|s,a) = softmax_s'(F(s,a).B(s'))`, fit
by minimizing mean row cross-entropy with Adam, lr=1e-2, 3000 steps, 3 random restarts
(keep the lowest final loss). Readout `Q_D(s,a) = (1/(1-gamma)) sum_s' p_theta(s'|s,a) r(s')`.

## Sanity checks (run automatically, printed by `run.py`)

1. Every M row (random-policy and directed-policy) sums to `1/(1-gamma)`: max error
   ~1e-13, every (k, gamma) cell, both experiments.
2. At `d = |S| = 104` (gamma=0.98, k=20, room reward): FB's normalized action gap at D
   = **1.000000** (exact, as expected -- rank-104 SVD of a 416x104 matrix is exact
   reconstruction). Density's = **0.999940** (well within the 0.05 tolerance; optimization,
   not closed form, so not bit-exact).
3. At k=1 (no stickiness), the true raw action gap at D is **15.83** (room reward,
   gamma=0.98) -- clearly nonzero, i.e. crossing and failing are still distinguishable
   actions before any stickiness is introduced.

## Main sweep -> `gap.csv` (240 rows)

k in {1,5,10,20,40} x d in {4,8,16,32} x gamma in {0.9,0.98} x reward in {room,
target_cell} x method in {true,fb,density}. Columns: `Q_cross`, `Q_fail_mean`, `raw_gap`
(= `Q_cross - Q_fail_mean`), `norm_gap` (= `raw_gap / raw_gap_true`, so true's own
norm_gap is always 1 by construction), `argmax_action`, `argmax_is_cross`,
`mass_next_room_a_cross` (probability mass of the `a_cross` row landing in the next room).
`Q_true(s,a) = sum_s' M(s,a,s') r(s')` -- the standard Q^pi(s,a) for reward r under the
random policy; FB's and Density's readouts are built to approximate the same quantity, so
all three are directly comparable.

**Headline (room reward, gamma=0.98): normalized action gap at D, k x d.**

FB:

| k\d |   4   |   8   |  16   |  32   |
|-----|-------|-------|-------|-------|
|  1  | 0.910 | 0.912 | 0.945 | 0.978 |
|  5  | 0.644 | 0.694 | 0.730 | 0.932 |
| 10  | 0.455 | 0.504 | 0.564 | 0.907 |
| 20  | 0.272 | 0.316 | 0.407 | 0.875 |
| 40  | 0.132 | 0.171 | 0.290 | 0.855 |

Density:

| k\d |   4   |   8   |  16   |  32   |
|-----|-------|-------|-------|-------|
|  1  | 0.917 | 0.950 | 0.976 | 0.993 |
|  5  | 0.707 | 0.865 | 0.939 | 0.982 |
| 10  | 0.570 | 0.832 | 0.923 | 0.976 |
| 20  | 0.458 | 0.801 | 0.911 | 0.971 |
| 40  | 0.349 | 0.799 | 0.900 | 0.971 |

Every cell here has `argmax_is_cross = True` for both methods at gamma=0.98. At
**gamma=0.9**, FB's argmax actually *flips away* from `a_cross` at `d=4, k in {10,20,40}`
(raw normalized gap goes slightly negative: -0.007, -0.057, -0.032) -- a genuine sign
flip, not just a shrinking gap. Density never flips under the room reward at either
gamma. See `gap.csv` for the full table (both gammas, both reward types). The sparser
`target_cell` reward is considerably noisier for both methods (rank-d SVD/softmax fits
optimized under an aggregate loss, evaluated on a single-cell reward) and produces
several argmax flips for both FB and Density -- see the notes doc for the honest read.

## Directed-policy occupancy experiment ("Figure-1 analogue") -> `occupancy.csv` (27 rows)

A second, independent demonstration, mirroring the "predicted occupancy collapses to the
dataset marginal instead of the goal" figure. Defines an eps-greedy (eps=0.1) shortest-path
policy `pi_g` toward `G=(4,10)`, computed by BFS on the grid graph (giving a fixed greedy
action per state, independent of k). Computes the exact successor measure `M_g` under
*continuation* policy `pi_g` (only the continuation changes; the initial action still goes
through the k-sticky per-action matrices; the dataset marginal `rho` is still the
random-policy stationary distribution -- the data is what it is, only the evaluated
policy changes). Fits FB and Density to `M_g` exactly as before (same `rho` for FB's
weighting).

Row inspected: `(S0, a*)` where `a*` is `pi_g`'s greedy action at `S0=(2,2)` (= "right").
For `k in {1,20,40}`, `d in {4,8,16}`: `tv_to_true` (total-variation distance of the
fit's row to the true row), `tv_to_rho` (to the dataset marginal), `mass_next_room`.

**Honest reading (see the notes doc for the full discussion): the naive "FB's row is
literally closer to rho than to the truth" does NOT hold here in absolute total variation**
(FB's `tv_to_true` is 0.01-0.05; `tv_to_rho` for every method, including the true row
itself, is ~0.8-0.87 -- rho is diffuse and *any* concentrated successor row is far from it
in TV, so this metric can't discriminate "collapse toward rho" from "still mostly right").
What *does* hold: **FB's `tv_to_true` is consistently 2-4x Density's**, across every
(k, d) cell tested, and the FB row is visibly more spread out over the room in the figure.
`mass_next_room` is nearly identical across all three methods here (~0.93) -- this
particular start state is far enough from the goal that it doesn't discriminate.

## Seeds and confidence intervals (gap.csv)

FB's rank-`d` fit is the closed-form rho-weighted SVD -- deterministic, one row per cell
(`seed = -1`, same as `true`). Density is fit independently for 5 seeds per (gamma, k, d)
cell (`seed in {0,...,4}`), each a single init with no restart-keep-best, so the 5 rows
are a genuine sample for a confidence interval, not a best-of-N estimate. The gap-vs-k
panels plot the Density mean with a 95% interval across those 5 seeds (`rliable`'s
stratified bootstrap if `rliable` is importable, otherwise a t-interval -- `rliable` is
not installed in this repo's jax venv, so every figure here used the t-interval fallback,
labeled in-figure). The door-map figures (below) fix Density to seed 0.

## Paper figures

- `fig_fourrooms_estimator.{pdf,png}` -- full appendix figure (unchanged by the restyle).
- `fig_fourrooms_main.{pdf,png}` -- compact v1 (unchanged by the restyle).
- `fig_fourrooms_main_v2.{pdf,png}` -- compact v2: one shared truncated-Blues ramp for
  every occupancy map (method colors live only in the gap-panel lines), PowerNorm(gamma=0.6)
  fixed per row to the true row's range, TV-to-true printed under each fit, gap panel with
  the Density mean + 95% CI band. Caption text: `caption_main_v2.tex`.
- `fourrooms/maps_cross.{pdf,png}`, `fourrooms/maps_fail.{pdf,png}` -- the same door-row
  heatmaps as v2, split out one row per file (for LaTeX subfigures).
- `fourrooms/gap_vs_k.{pdf,png}` -- the same gap-vs-k(d=8) panel as v2, standalone.
- `fourrooms/gap_vs_k_all_d.{pdf,png}` -- gap-vs-k at every d in {4,8,16,32} (appendix).
- `subfigures.tex` -- LaTeX snippet placing maps_cross / maps_fail / gap_vs_k as three
  `subcaptionbox` blocks (requires the `subcaption` package).

## Files

- `run.py` -- the whole experiment (environment, exact math, both fits, sweep, occupancy
  experiment, sanity checks, all figures). Self-contained; the only repo-external deps are
  numpy / jax / optax / matplotlib / scipy (scipy only for the t-interval fallback).
- `gap.csv` -- main k x d x gamma x reward x method x seed sweep at the door bottleneck D.
- `occupancy.csv` -- directed-policy occupancy experiment (k x d, gamma=0.98 only).
- `caption_main.tex`, `caption_main_v2.tex` -- LaTeX caption text for the two compact figures.
- `subfigures.tex` -- LaTeX snippet for the three split subfigures.
- `../../PAPER/ICLR/figures/fig_fourrooms_estimator.{pdf,png}` -- the appendix figure.
- `../../PAPER/ICLR/figures/fig_fourrooms_main{,_v2}.{pdf,png}`,
  `../../PAPER/ICLR/figures/fourrooms/*.{pdf,png}` -- the compact and split paper figures.
- `../../docs/notes/2026-09-09-fourrooms-estimator-demo.md` -- write-up and verdict.
