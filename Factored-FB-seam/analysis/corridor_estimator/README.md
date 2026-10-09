# Corridor-with-exit estimator demo: FB (squared loss) vs Density-FB (softmax / cross-entropy)

A second, simpler grid for the FourRooms demonstration in `analysis/fourrooms_estimator/`
(read that README first; everything here reuses its machinery unchanged). The claim
being demonstrated is the same: FB's rank-`d` fit is a rho(s')-weighted *squared-loss*
fit to M(s,a,s')/rho(s'), which blurs the difference between the two actions at a
sticky bottleneck; a normalized rank-`d` fit trained with cross-entropy (Density-FB)
keeps them apart.

Run:

```bash
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/corridor_estimator/run.py
# figures + caption only (reads gap.csv / gap_trained.csv, ~20 s):
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/corridor_estimator/run.py --figures-only
```

Full run: ~9 min on CPU (JAX); the closed-form sweep is 10 s, the rest is the 50
TD-training runs.

## Environment

7 x 15 grid with a 1-cell wall border. A corridor of 8 free cells, 1 cell wide (row 3,
cols 1..8), leads into a 5 x 5 room (rows 1..5, cols 9..13); the corridor joins the room
at the middle of the room's left wall. |S| = 33 free cells. 4 deterministic actions
(up/down/left/right), bumping into a wall = stay. Behavior policy uniform random.

**The exit D = (3, 8)**, the last corridor cell before the room. `a_cross` = right (into
the room) always succeeds. **Sticky rule (k)**: each other action at D (up/down/left)
keeps the agent at D with probability 1 - 1/k, else normal dynamics. At D, up and down
are wall bumps already; only "left" (back into the corridor) carries real signal, exactly
as in FourRooms. gamma = 0.98. Reward for the margin: 1 on every room cell. Star cell for
the maps: G = (1, 13), the far corner of the room.

## What is computed (identical code to FourRooms)

- Exact successor measure `M(s,a,s') = [P_k(a) (I - gamma P_pi)^{-1}]`, stationary rho by
  power iteration; every row of M sums to 1/(1-gamma) (max error ~4e-14, every k).
- FB fit: rank-d truncated SVD of `(M/rho) diag(sqrt rho)` (closed form, deterministic).
- Density fit: rank-d softmax model, cross-entropy against (1-gamma)M, Adam lr 1e-2,
  3000 steps, **5 seeds, single init each** (no restart-keep-best), so the 5 values are a
  genuine sample for the CI.
- TD-trained FB and TD-trained Density-FB at d = 8 (`gap_trained.csv`): same TD losses as
  FourRooms (FB: squared TD on F.B with Polyak targets; Density: cross-entropy against
  (1-gamma) one-hot + gamma softmax target), 20k Adam steps, lr 3e-4, tau 0.005, batch
  256, dataset = 100k transitions of the random policy (restart every 500 steps),
  5 seeds per k.

Metrics per row: `norm_gap` = (Q(right) - mean Q(up,down,left)) / true margin at D;
`door_frac_cross` / `door_frac_fail` = fraction of the exit row's / the "left" row's mass
on D itself (FB rows are clipped at 0 before this fraction, as in the figure);
`tv_cross` / `tv_fail` = total-variation distance to the true row (FB unclipped);
`mass_room_cross` = mass of the exit row inside the room.

## Sanity checks (printed by `run.py`, all pass)

1. Row sums of M = 1/(1-gamma) to 4e-14 at every k.
2. At d = |S| = 33 (k = 20): FB normalized margin = 1.000000; Density = 0.999965.
3. At k = 1 the true raw margin is 12.29 (room reward), i.e. the two actions differ
   before any stickiness.

## Results: normalized margin at D, gamma = 0.98 (room reward)

**d = 8 (the figure's panel):**

| k  | FB closed | Density closed (5-seed mean, 95% CI) | FB TD-trained (5 seeds) | Density TD-trained (5 seeds) |
|----|-----------|--------------------------------------|-------------------------|------------------------------|
| 1  | 0.956 | 0.967 [0.967, 0.968] | 0.328 [-0.013, 0.669] | 0.870 [0.826, 0.914] |
| 5  | 0.811 | 0.922 [0.914, 0.930] | 1.000 [0.510, 1.490]  | 1.260 [1.175, 1.344] |
| 10 | 0.758 | 0.914 [0.904, 0.924] | 0.630 [0.112, 1.148]  | 1.296 [1.237, 1.354] |
| 20 | 0.423 | 0.908 [0.902, 0.914] | 0.560 [-0.317, 1.438] | 1.272 [1.187, 1.357] |
| 40 | 0.305 | 0.901 [0.896, 0.905] | 0.760 [0.487, 1.033]  | 1.050 [0.883, 1.217] |

**FB closed form, k x d:**

| k\d |   4   |   8   |  16   |  32   |
|-----|-------|-------|-------|-------|
|  1  | 0.867 | 0.956 | 0.981 | 1.000 |
|  5  | 0.629 | 0.811 | 0.956 | 0.999 |
| 10  | 0.444 | 0.758 | 0.946 | 0.999 |
| 20  | 0.228 | 0.423 | 0.939 | 0.999 |
| 40  | 0.100 | 0.305 | 0.935 | 0.999 |

**Density closed form (5-seed mean), k x d:**

| k\d |   4   |   8   |  16   |  32   |
|-----|-------|-------|-------|-------|
|  1  | 0.935 | 0.967 | 0.985 | 1.000 |
|  5  | 0.866 | 0.922 | 0.969 | 1.000 |
| 10  | 0.831 | 0.914 | 0.963 | 1.000 |
| 20  | 0.837 | 0.908 | 0.961 | 1.000 |
| 40  | 0.824 | 0.901 | 0.961 | 1.000 |

d = 32 is one short of full rank (|S| = 33), so both fits are essentially exact there;
the interesting ranks are 4-16. FB's argmax at D stays on `right` in every closed-form
cell; the TD-trained FB argmax flips away from `right` in 4 of 25 seeds (k = 1: 2 seeds,
k = 10: 1, k = 20: 1); TD-trained Density-FB never flips.

**Door-cell mass fractions in the six maps (k = 20, d = 8, seed 0):** true exit 0.067 /
fail 0.134; FB 0.063 / 0.082; Density-FB 0.073 / 0.133. TV to the true row: FB 0.042
(exit) / 0.103 (fail); Density-FB 0.019 / 0.011. So FB loses ~40% of the "stuck at the
exit" mass on the failing row, while Density-FB reproduces it.

## Honest reading

- The closed-form story is as clean as in FourRooms: FB's margin at d = 8 falls from
  0.96 to 0.31 as k goes 1 -> 40 while Density-FB stays at 0.90-0.97. FB's k = 40 value is
  a little higher than in FourRooms (0.31 vs 0.17) because the corridor grid is smaller
  (33 vs 104 states) and rank 8 covers more of it; at d = 4 it reaches 0.10.
- TD-trained FB is very seed-noisy here (CI reaches zero at k = 1, 10, 20) -- its
  point estimates do not track the closed-form curve the way they did in FourRooms.
- TD-trained Density-FB *overshoots* the true margin (1.26-1.30 at k = 5-20) rather than
  sitting just under it as in FourRooms (0.78-0.85). It always keeps the right ordering
  and its TV to the true rows (0.11-0.25) is 2-3x smaller than TD-FB's (0.23-0.43), but
  the overshoot is a real feature of this run and is stated in the caption.

## Figure and colors

`PAPER/ICLR/figures/corridor/fig_toy_corridor.{pdf,svg,png}` (SVG has editable text,
`svg.fonttype = none`), plus the subfigures `maps_exit`, `maps_fail`, `margin_vs_k`
(`.pdf,.png`). Same design as `fourrooms/fig_toy_main`: maps smoothed with a small
wall-aware Gaussian (sigma 0.3 cells, normalized by the free-cell mask so the 1-wide
corridor is not dimmed by its walls), color scale shared within each column across the
two rows, black dot at D, gold star at G, "door: 0.xx" under each map. Series colors are
inherited unchanged from `fig_toy_main` (FB blue `#2171b5`, Density-FB orange-red
`#e34a33`, true grey dashed); node is not installed on this box so the dataviz palette
validator was not re-run, but this blue/orange-red pair is the same CVD-safe pair already
used throughout the paper. CI: rliable is not installed in the jax venv, so every
interval is the scipy t-interval (95%, df = 4), labeled in the caption.

## Files

- `run.py` -- the whole experiment (environment, exact math, both fits, TD training,
  sanity checks, figures, caption). `--figures-only` re-renders from the CSVs.
- `gap.csv` (140 rows) -- closed-form k x d x method x seed sweep at D.
- `gap_trained.csv` (50 rows) -- TD-trained FB / Density-FB, k x seed, d = 8.
- `caption.tex` -- LaTeX caption for the figure.
- `../../docs/notes/2026-09-10-corridor-estimator-demo.md` -- write-up and verdict.

## Variant: agent starts before the exit (`fig_toy_corridor_approach`)

`PAPER/ICLR/figures/corridor/fig_toy_corridor_approach.{pdf,svg,png}`. S0 = (3, 6), two
corridor cells before D (black dot; D is a small hollow marker). Three map rows from S0
taking the action toward the door (right), under "walk right to D, take action a at D
once, uniform random afterwards": row 1 "approach" a = exit w.p. 0.5 / fail (left)
otherwise; row 2 "then exit" a = right; row 3 "then fail" a = left (sticky, k = 20). The
walk S0 -> (3,7) -> D is deterministic and is entered exactly ((1-g)[e_(3,7) + g e_D]);
only the door rows g^2 M(D, a, .) come from the estimators -- the same k = 20, d = 8,
seed-0 fits as the main figure. The 2-step truncated alternative for row 1 would contain
no estimator content at all (it is the exact prefix), so the 0.5-mixture policy was used.
Color scale shared within each column across the three rows; right panel unchanged.

| row | true door | FB door | Density door | TV FB | TV Density |
|-----|-----------|---------|--------------|-------|------------|
| approach  | 0.116 | 0.089 | 0.118 | 0.068 | 0.009 |
| then exit | 0.084 | 0.080 | 0.089 | 0.041 | 0.018 |
| then fail | 0.148 | 0.099 | 0.147 | 0.099 | 0.010 |

Reading: Density-FB matches all three rows; FB's "then exit" and "then fail" rows are
nearly the same blob (door 0.08 vs 0.10, true 0.08 vs 0.15). Row 1 is NOT identical for
FB (TV 0.068, door 0.09 vs 0.12): its deficit at the exit cell is already present in the
0.5 mixture, because FB under-weights the door in both of the rows the mixture averages.

## Variants: 14-cell corridor, start / mid (`--corridor14`)

```bash
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/corridor_estimator/run.py --corridor14   # ~16 s
```

Same room, corridor lengthened to 14 cells (`CORRIDOR_LEN` env var; grid 7 x 21, |S| = 39,
D = (3, 14), star (1, 19)). Recomputes exact M, FB closed form, Density (5 seeds) for
every k x d on this layout into `corridor14.csv` (gap.csv columns plus `variant`, `s0`,
`door_frac`, `room_frac`, `tv` for the 12 appended map rows); TD training is skipped
(closed-form lines only on the right panel). Sanity checks pass (row sums 3.6e-14; d = |S| = 39
margins 1.000000 / 1.000089; k = 1 raw margin 12.49). Margin at d = 8 on this layout:
FB 0.933 / 0.643 / 0.498 / 0.384 / 0.144 and Density-FB 0.963 / 0.923 / 0.906 / 0.885 / 0.895
for k = 1 / 5 / 10 / 20 / 40.

`fig_toy_corridor_start` (S0 = first corridor cell, 13 steps before D) and
`fig_toy_corridor_mid` (S0 = middle cell (3, 7), 7 steps before D): two rows "then exit" /
"then fail", composed as in the approach variant (deterministic walk entered exactly,
door rows from the k = 20, d = 8, seed-0 fits). "door / room" = mass fraction on D / in
the room; TV to the true row:

| variant | row | true door/room | FB door/room (TV) | Density door/room (TV) |
|---------|-----|----------------|-------------------|------------------------|
| start | then exit | 0.067 / 0.710 | 0.062 / 0.713 (0.037) | 0.072 / 0.705 (0.015) |
| start | then fail | 0.119 / 0.641 | 0.075 / 0.681 (0.076) | 0.116 / 0.643 (0.010) |
| mid   | then exit | 0.076 / 0.801 | 0.070 / 0.803 (0.042) | 0.081 / 0.796 (0.017) |
| mid   | then fail | 0.134 / 0.723 | 0.085 / 0.768 (0.086) | 0.131 / 0.726 (0.011) |

The longer the walk, the more of the row is the (exact) corridor prefix, so all fits look
closer to the truth from the start cell than from the middle; the exit/fail contrast in
door mass is still true 0.07 vs 0.12-0.13, FB 0.06-0.07 vs 0.08, Density-FB matches.

## Contrast and error figures (14-cell corridor, start cell; produced by `--corridor14`)

- `fig_toy_corridor_start_contrast.{pdf,svg,png}`: rows then-exit / then-fail with a
  PowerNorm(gamma=0.6) shared per column, plus fail-minus-exit on one symmetric diverging
  scale (`RdBu_r`, red = more mass after failing, blue = more after exiting) fixed to the
  true row's max |value|; |Δ|/|Δ_true| printed under FB (0.84) and Density-FB (1.00).
- `fig_toy_corridor_start_error.{pdf,svg,png}`: fitted minus true per cell, exit/fail x
  FB/Density-FB, one shared symmetric scale, TV under each (FB 0.037 / 0.076, Density-FB
  0.015 / 0.010).

## k = 40 single-panel set (`PAPER/ICLR/figures/corridor/k40/`, produced by `--corridor14`)

14-cell corridor from the start cell at k = 40, d = 8, gamma = 0.98, Density-FB seed 0.
Files: `exit_{true,fb,density}`, `fail_{true,fb,density}` (one shared PowerNorm-0.6
occupancy scale across all six, `colorbar_occ`), `contrast_{true,fb,density}` (fail minus
exit, one symmetric scale fixed to the true contrast, `colorbar_contrast`), `margin_vs_k`
(closed-form, k = 40 marked), all `.pdf/.png` (+ `.svg` for maps), and `subfigures.tex`
(3 x 3 + margin block with `\subcaptionbox` labels). door / room: true .068/.713 (exit),
.120/.652 (fail); FB .047/.732, .051/.719; Density-FB .072/.709, .118/.654. Contrast
|Δ|/|Δ_true|: FB 0.23, Density-FB 1.02; contrast TV to true: FB .052, Density-FB .021.
Rows appended to `corridor14.csv` (`variant` = `k40_start_*`).
