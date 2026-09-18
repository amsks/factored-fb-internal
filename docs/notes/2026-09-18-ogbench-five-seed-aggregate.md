# OGBench main-table cells at five seeds (2026-09-18)

## The estimator behind the 81.3 / 30.4 / 52.7 / 4.7 cells

The cells in `PAPER/ICLR/content/experiments.tex` (tab:manipulation, OGBench block) and in
`docs/notes/2026-09-09-tdjtab-results-aggregate.md` are the Route-B (reward-inference) score
of `scripts/make_results_table.py:score_run` applied to each run's `eval_relabel.csv`: rows are
grouped by `step`, the `success` column is averaged over the env's five task rows at each
step, and the run's score is the mean of that task-mean over the last three eval steps
(800k, 900k, 1M). A cell is the mean over seeds of that per-run score. The 09-09 note quotes
the seed spread as the sample standard deviation (ddof=1); the paper's `\pmse{}` values for
the FB, HILP and Density-FB cells are the population standard deviation (ddof=0; 4.6 -> 3.8
on cube-single) and for the RLDP row the sample standard deviation -- none of them is a
standard error, although the caption says "standard error". The 09-09 note's `runs.csv`
is no longer on the hub; every cell below was recomputed from the csvs directly and the
three-seed values reproduce to the decimal.

Sources per cell: density_fb_z seeds 0-2 = hub `exp/<run>/eval_relabel.csv` (byte-identical
to the local `exp/tdjtab/` copies for cube), seeds 3-4 = local `exp/tdjtab/`; density_fb_z_state
cube seeds 0-2 = hub, everything else = local `exp/dfb_state/`; fb_tdj and hilp_tdjc = hub
`exp/<run>/` (cube-single fb_tdj is the clean re-run, not the contaminated two-arm dir);
rldp = local `exp/ortho1/` (ortho_coef=1, the arm the paper's RLDP row uses; the hub's
`exp/<env>_rldp_*` runs are the ortho 0.01 arm: cube-double 2.2, scene 4.7, puzzle 2.4).
Hub csvs were pulled into the session scratchpad (`hub_csv/`), not into `exp/`.

## Recomputed table (Route B, last-3, success %)

| env | arm | n | mean ± sd | mean ± s.e. | per-seed 0..4 | 3-seed (0-2) mean ± sd | delta |
|---|---|---|---|---|---|---|---|
| cube-single | density_fb_z | 5 | 82.9 ± 4.0 | 82.9 ± 1.8 | 76.0, 84.0, 84.0, 84.7, 86.0 | 81.3 ± 4.6 | +1.6 |
| cube-double | density_fb_z | 5 | 31.7 ± 3.1 | 31.7 ± 1.4 | 34.0, 29.3, 28.0, 32.0, 35.3 | 30.4 ± 3.2 | +1.3 |
| scene | density_fb_z | 5 | 54.0 ± 3.5 | 54.0 ± 1.6 | 48.7, 52.7, 56.7, 54.7, 57.3 | 52.7 ± 4.0 | +1.3 |
| puzzle-3x3 | density_fb_z | 5 | 5.9 ± 2.2 | 5.9 ± 1.0 | 3.3, 5.3, 5.3, 9.3, 6.0 | 4.7 ± 1.2 | +1.2 |
| cube-single | density_fb_z_state | 5 | 91.7 ± 2.2 | 91.7 ± 1.0 | 91.3, 92.0, 89.3, 95.3, 90.7 | 90.9 ± 1.4 | +0.8 |
| cube-double | density_fb_z_state | 5 | 16.3 ± 2.1 | 16.3 ± 0.9 | 16.7, 14.0, 16.7, 14.7, 19.3 | 15.8 ± 1.5 | +0.5 |
| scene | density_fb_z_state | 5 | 46.3 ± 5.6 | 46.3 ± 2.5 | 54.0, 46.7, 48.0, 38.7, 44.0 | 49.6 ± 3.9 | -3.3 |
| puzzle-3x3 | density_fb_z_state | 5 | 5.5 ± 3.6 | 5.5 ± 1.6 | 6.7, 3.3, 3.3, 2.7, 11.3 | 4.4 ± 1.9 | +1.0 |
| cube-single | fb_tdj | 3 | 45.6 ± 7.2 | 45.6 ± 4.2 | 50.7, 37.3, 48.7 | 45.6 ± 7.2 | -- |
| cube-double | fb_tdj | 3 | 2.2 ± 3.3 | 2.2 ± 1.9 | 0.7, 6.0, 0.0 | 2.2 ± 3.3 | -- |
| scene | fb_tdj | 3 | 10.2 ± 5.0 | 10.2 ± 2.9 | 10.0, 15.3, 5.3 | 10.2 ± 5.0 | -- |
| puzzle-3x3 | fb_tdj | 3 | 2.7 ± 1.2 | 2.7 ± 0.7 | 2.0, 4.0, 2.0 | 2.7 ± 1.2 | -- |
| cube-single | hilp_tdjc | 3 | 71.1 ± 6.9 | 71.1 ± 4.0 | 63.3, 76.7, 73.3 | 71.1 ± 6.9 | -- |
| cube-double | hilp_tdjc | 3 | 11.1 ± 4.0 | 11.1 ± 2.3 | 7.3, 10.7, 15.3 | 11.1 ± 4.0 | -- |
| scene | hilp_tdjc | 3 | 39.3 ± 7.0 | 39.3 ± 4.1 | 38.7, 46.7, 32.7 | 39.3 ± 7.0 | -- |
| puzzle-3x3 | hilp_tdjc | 3 | 1.3 ± 0.7 | 1.3 ± 0.4 | 2.0, 1.3, 0.7 | 1.3 ± 0.7 | -- |
| cube-single | rldp (ortho 1) | 3 | 27.6 ± 3.7 | 27.6 ± 2.1 | 24.0, 31.3, 27.3 | 27.6 ± 3.7 | -- |
| cube-double | rldp (ortho 1) | 3 | 2.7 ± 1.3 | 2.7 ± 0.8 | 1.3, 2.7, 4.0 | 2.7 ± 1.3 | -- |
| scene | rldp (ortho 1) | 3 | 10.2 ± 3.2 | 10.2 ± 1.8 | 6.7, 12.7, 11.3 | 10.2 ± 3.2 | -- |
| puzzle-3x3 | rldp (ortho 1) | 3 | 10.4 ± 1.7 | 10.4 ± 1.0 | 8.7, 12.0, 10.7 | 10.4 ± 1.7 | -- |

sd = sample standard deviation over seeds (ddof=1); s.e. = sd / sqrt(n).

Reading: no main-table Density-FB cell moved by more than 2 points going from three seeds to
five (largest +1.6 on cube-single). The one cell that moved more, density_fb_z_state on scene
(49.6 -> 46.3, seeds 3 and 4 came in at 38.7 and 44.0), is not in the main table; the
full-state arm appears only for cube in `PAPER/ICLR/baselines/td_jepa_density.tex`. Ordering
against the baselines is unchanged on every env.

## Baseline cells still at three seeds

All twelve baseline cells of the OGBench block: fb_tdj, hilp_tdjc and rldp (ortho 1) on
cube-single, cube-double, scene and puzzle-3x3. No seed 3 or 4 exists for any of them locally
or on the hub (checked `exp/`, `exp/tdjtab/`, `exp/ortho1/`, hub `exp/` and `status/`). The
24 commands that close the gap are in `handoff/baseline_seeds34/commands.txt`, with the
covering message in `handoff/baseline_seeds34/mailbox_message.md` (not posted). The RLDP
commands carry `--override ortho_coef=1 --save_dir exp/ortho1` because that is the arm the
table reports; the manifest's own rldp lines are the ortho 0.01 arm.
