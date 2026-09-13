# 2026-09-10 -- Corridor-with-exit estimator demo (FourRooms demo, second grid)

Code: `analysis/corridor_estimator/run.py` (+ `gap.csv`, `gap_trained.csv`, `caption.tex`,
`README.md`). Figure: `PAPER/ICLR/figures/corridor/fig_toy_corridor.{pdf,svg,png}` and the
subfigures `maps_exit`, `maps_fail`, `margin_vs_k`. Companion to
`docs/notes/2026-09-09-fourrooms-estimator-demo.md`; same machinery, new grid.

## Setup

An 8-cell corridor (1 wide) leads into a 5x5 room through the middle of the room's left
wall; 33 free cells, 1-cell wall border. The exit D is the last corridor cell. "right" at D
enters the room and always succeeds; up/down/left at D keep the agent at D with
probability 1 - 1/k. Uniform-random behavior policy, gamma 0.98, reward 1 on every room
cell. Everything else -- exact successor measure, FB's closed-form rho-weighted SVD fit,
the Density rank-d softmax cross-entropy fit (5 seeds), the TD-trained tabular versions
of both (5 seeds, 20k steps, 100k-transition random-policy dataset), t-interval CIs --
is the FourRooms code verbatim.

Sanity: M rows sum to 1/(1-gamma) (4e-14); at d = |S| = 33 FB recovers the true margin to
1e-6 and Density to 4e-5; at k = 1 the true raw margin is 12.3.

## Result: normalized margin at D, d = 8

| k  | FB closed | Density closed | FB TD-trained | Density TD-trained |
|----|-----------|----------------|---------------|--------------------|
| 1  | 0.96 | 0.97 | 0.33 [-0.01, 0.67] | 0.87 [0.83, 0.91] |
| 5  | 0.81 | 0.92 | 1.00 [0.51, 1.49]  | 1.26 [1.18, 1.34] |
| 10 | 0.76 | 0.91 | 0.63 [0.11, 1.15]  | 1.30 [1.24, 1.35] |
| 20 | 0.42 | 0.91 | 0.56 [-0.32, 1.44] | 1.27 [1.19, 1.36] |
| 40 | 0.31 | 0.90 | 0.76 [0.49, 1.03]  | 1.05 [0.88, 1.22] |

Closed-form Density CIs are tight (width < 0.02). Door-cell mass in the six k=20 maps:
true 0.07 (exit) / 0.13 (fail); FB 0.06 / 0.08; Density-FB 0.07 / 0.13. TV to the true
row at k=20, d=8: FB 0.04 / 0.10, Density-FB 0.02 / 0.01.

## Verdict

The closed-form story is at least as clear as in FourRooms, and the picture is simpler
to read: the true "fail to exit" future has a dark spot at the exit cell, FB's map
smears that spot into the room (door mass 0.08 vs the true 0.13), Density-FB keeps it
(0.13). FB's margin at d = 8 falls 0.96 -> 0.31 across k while Density-FB's stays at
0.90-0.97; the collapse is monotone in k and worsens at lower rank (d = 4: 0.87 -> 0.10).
FB's endpoint is somewhat higher than FourRooms' 0.17 because the state space is a third
the size, so rank 8 is relatively richer; d = 32 is near full rank and both fits are exact
there (it is in the CSV only).

Two things are weaker than in FourRooms, both on the TD-trained side, and both are stated
in the caption:

1. TD-trained FB is seed-noisy: its 95% interval reaches zero at k = 1, 10 and 20, and its
   argmax at D leaves "right" in 4 of 25 seeds. Its point estimates do not follow the
   closed-form curve.
2. TD-trained Density-FB overshoots the true margin (1.26-1.30 at k = 5-20; 1.05 at
   k = 40) instead of sitting just under it as in FourRooms (0.78-0.85). It keeps the right
   ordering in 25/25 seeds and its TV to the true rows (0.11-0.25) is 2-3x TD-FB's
   (0.23-0.43), but the value it regresses is too large, not just distinct. Hyperparameters
   were not retuned for the smaller grid (same 20k steps, lr 3e-4, batch 256 as FourRooms).

Net: use the closed-form lines as the demonstration; treat the TD diamonds as a check that
training the same two losses by TD keeps the ordering, not as a quantitative match.

## Reproduce

```bash
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/corridor_estimator/run.py            # ~9 min
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/corridor_estimator/run.py --figures-only
```

## Variant: start before the exit (`fig_toy_corridor_approach`, added same day)

S0 = (3, 6), two cells before D. Rows: "approach" (walk to D, then exit w.p. 0.5 / fail
otherwise, random after), "then exit", "then fail" (k = 20); the deterministic walk is
entered exactly, only the door rows come from the main figure's k = 20, d = 8, seed-0
fits. The 2-step truncated construction was not used because it contains no estimator
content. TV to true (approach / exit / fail): FB 0.068 / 0.041 / 0.099, Density-FB
0.009 / 0.018 / 0.010; door mass: true 0.12 / 0.08 / 0.15, FB 0.09 / 0.08 / 0.10,
Density-FB 0.12 / 0.09 / 0.15. Rows 2-3 read as intended (FB's two futures nearly the
same; Density-FB and true differ sharply). Row 1 is *not* identical across columns: FB
is visibly lighter at the exit (TV 0.07) because its door deficit is present in both rows
the mixture averages. Stated in the second caption in `caption.tex`.

## Variants: 14-cell corridor, start and mid (`--corridor14`, added same day)

Corridor lengthened to 14 cells (|S| = 39), exact M / FB / Density (5 seeds) recomputed
for every k x d into `analysis/corridor_estimator/corridor14.csv`; TD training skipped.
Margin at d = 8 on this layout: FB 0.93 -> 0.64 -> 0.50 -> 0.38 -> 0.14, Density-FB
0.96 -> 0.92 -> 0.91 -> 0.89 -> 0.90 (k = 1..40) -- FB's collapse is slightly stronger than on
the 8-cell corridor (0.31 at k = 40). Figures `fig_toy_corridor_start` (S0 = first cell,
13 steps before D) and `fig_toy_corridor_mid` (S0 = 7 steps before D), rows "then exit" /
"then fail", same composition as the approach variant. Door / room mass and TV to true:
start-exit true .07/.71, FB .06/.71 (TV .037), Density .07/.70 (.015); start-fail true
.12/.64, FB .08/.68 (.076), Density .12/.64 (.010); mid-exit true .08/.80, FB .07/.80
(.042), Density .08/.80 (.017); mid-fail true .13/.72, FB .08/.77 (.086), Density .13/.73
(.011). Reading unchanged: FB's exit and fail rows are nearly the same blob (door .06-.07
vs .08) while true and Density-FB separate them (.07-.08 vs .12-.13); the longer walk only
dilutes every row with the exact corridor prefix. Captions in `caption.tex`.

## Contrast and error figures, 14-cell corridor from the start cell (added same day)

`fig_toy_corridor_start_contrast`: rows then-exit / then-fail (PowerNorm 0.6 per column) and
fail-minus-exit on one symmetric diverging scale (RdBu reversed: red = more mass after
failing) fixed to the true row's max |value| (0.052, at the exit cell). L1 of the fitted
contrast relative to the true contrast: **FB 0.84, Density-FB 1.00**. Note the L1 ratio
understates FB's problem: FB keeps 84% of the contrast's mass but puts it in the wrong
place (a diffuse smear over the room instead of a spot at the exit), which is why its
margin at D still collapses. `fig_toy_corridor_start_error`: fitted minus true per cell,
2x2 (exit/fail x FB/Density-FB), shared scale; TV FB 0.037 / 0.076, Density-FB 0.015 /
0.010. FB's error = deficit at the exit + surplus across the room. Captions in `caption.tex`.

## k = 40 single-panel set, 14-cell corridor from the start cell (added same day)

`PAPER/ICLR/figures/corridor/k40/` (produced by `--corridor14`): exit_{true,fb,density},
fail_{true,fb,density} (one shared PowerNorm-0.6 occupancy scale across all six maps, so
`colorbar_occ` is exact; hue = estimator), contrast_{true,fb,density} (fail minus exit,
one symmetric scale fixed to the true contrast's max, `colorbar_contrast`), `margin_vs_k`
(closed-form, k = 40 marked), and `subfigures.tex` (3x3 + margin, subcaptionbox labels).
d = 8, gamma = 0.98, Density-FB seed 0. door / room: true exit .068/.713, fail .120/.652;
FB exit .047/.732, fail .051/.719; Density-FB exit .072/.709, fail .118/.654. TV to true:
FB .088 / .112, Density-FB .015 / .008. Contrast: |Δ|/|Δ_true| FB **0.23**, Density-FB
1.02; TV of the contrast map to the true contrast FB .052, Density-FB .021. At k = 40 FB's
two futures are essentially the same map (door .047 vs .051) -- the strongest version of
the story in this folder; at k = 20 the same ratio was 0.84. Rows appended to
`corridor14.csv` (variant `k40_start_*`, column `contrast_ratio`).
