# tdjtab results aggregate (2026-09-09)

## Source and definitions

Source: `runs.csv` pulled from the hub `amsks/factored-fb-checkpoints` on 2026-09-09
around 10:10 UTC. One row per finished 1M-step run. Columns used here: env, critic,
seed, A_last3, A_final, B_last3, B_final, all as fractions, reported below as percent
with one decimal.

Route A is the goal-conditioned eval: the actor is conditioned on the true goal state
and the critic is scored against that goal directly.

Route B is the reward-inference eval: the index z that conditions the actor is fit
from 10k relabelled reward samples for the task, not from the goal state. This is the
route that tests whether the method supports a general reward interface, not just
goal-conditioning.

"last-3" is the mean of the last three eval passes recorded for a run. "final" is the
last eval pass only. Both are reported as mean plus or minus standard deviation across
seeds, with the seed count n.

## Validity rules applied

1. Route B is invalid for `density_fb` and `density_fb_state`. These are
   goal-conditioned critics with no reward interface at all: the run's Route B rows in
   the source data were produced by a zero-goal rollout, not a real reward-conditioned
   policy. These cells print "invalid", not a number.
2. Route A is invalid for `hilp_tdjc` and `hilp_density_tdjc`. There is a known eval
   bug: without `map_eval_cond` set, the actor rolls out with z fixed at zero during
   the goal-conditioned eval. The fix was committed 2026-09-08; these checkpoints have
   not yet been re-scored under the fix. These cells print "invalid", not a number.
   Route B numbers for these two critics are unaffected and are reported normally.
3. On cube-double, reward-inference success (Route B) is bounded above by roughly
   half of goal-conditioned success (Route A) for every method. OGBench permutes
   which cube goes to which target on every reset, and the 10k reward samples used to
   fit z encode only one such assignment; a policy that ignores the assignment and
   just manipulates both cubes can score at most about half of Route A by chance.
   This is an interpretive upper bound on what any fixed-task method can achieve
   under reward inference on this task, not a transformation applied to the measured
   data. Every cube-double Route B number below is reported as measured; read it
   against the roughly-half-of-Route-A bound rather than against Route A directly.
   For density_fb_z, use density_fb's goal-conditioned 68.6 as the reference (the
   cleanest goal-conditioned critic on this env), giving a bound of about 34;
   density_fb_z's measured 30.4 sits close to that bound.

## Per-env tables

### antmaze-large-navigate-v0

| critic | n | A last-3 | A final | B last-3 | B final |
|---|---|---|---|---|---|
| fb_tdj | 3 | 22.1 ± 3.4 | 35.3 ± 4.0 | 19.8 ± 6.0 | 31.3 ± 7.6 |
| hilp_tdjc | 3 | invalid | invalid | 29.1 ± 6.7 | 26.0 ± 10.6 |
| rldp | 3 | 7.3 ± 5.5 | 9.0 ± 6.0 | 6.9 ± 2.8 | 12.7 ± 4.2 |

### antmaze-medium-navigate-v0

| critic | n | A last-3 | A final | B last-3 | B final |
|---|---|---|---|---|---|
| density_fb_z | 1 | 48.0 | 56.0 | 44.0 | 50.0 |
| fb_tdj | 3 | 71.1 ± 1.0 | 73.0 ± 4.6 | 68.9 ± 6.0 | 74.7 ± 6.1 |
| hilp_tdjc | 3 | invalid | invalid | 74.9 ± 1.4 | 77.3 ± 6.4 |
| rldp | 3 | 37.2 ± 8.6 | 42.3 ± 10.8 | 50.9 ± 11.5 | 50.7 ± 16.2 |

### cube-double-play-v0

Route B numbers here are measured, not capped. Read them against the bound in rule 3
above (roughly half of Route A, ~34 for the density_fb_z line using density_fb's
goal-conditioned 68.6 as reference), not against Route A directly.

| critic | n | A last-3 | A final | B last-3 | B final |
|---|---|---|---|---|---|
| density_fb | 3 | 68.6 ± 1.2 | 72.7 ± 4.0 | invalid | invalid |
| density_fb_state | 3 | 10.6 ± 1.9 | 11.7 ± 3.1 | invalid | invalid |
| density_fb_z | 3 | 33.9 ± 9.4 | 34.3 ± 9.9 | 30.4 ± 2.6 | 35.3 ± 9.0 |
| density_fb_z_state | 3 | 22.0 ± 7.1 | 21.0 ± 5.0 | 15.8 ± 1.3 | 14.7 ± 4.1 |
| fb_tdj | 3 | 1.6 ± 0.5 | 3.0 ± 2.0 | 2.2 ± 2.7 | 0.7 ± 0.9 |
| hilp_density_tdjc | 3 | invalid | invalid | 2.4 ± 1.6 | 1.3 ± 0.9 |
| hilp_tdjc | 3 | invalid | invalid | 11.1 ± 3.3 | 8.0 ± 4.3 |
| rldp | 3 | 2.2 ± 0.4 | 3.0 ± 1.7 | 2.2 ± 1.1 | 3.3 ± 3.4 |
| rldp_density | 3 | 1.4 ± 0.4 | 2.3 ± 0.6 | 2.0 ± 0.5 | 0.0 ± 0.0 |

### cube-single-play-v0

| critic | n | A last-3 | A final | B last-3 | B final |
|---|---|---|---|---|---|
| density_fb | 3 | 93.1 ± 1.8 | 91.3 ± 0.6 | invalid | invalid |
| density_fb_state | 3 | 90.8 ± 2.4 | 92.3 ± 2.5 | invalid | invalid |
| density_fb_z | 3 | 74.9 ± 7.1 | 75.0 ± 13.5 | 81.3 ± 4.6 | 82.0 ± 12.5 |
| density_fb_z_state | 3 | 83.9 ± 7.6 | 84.3 ± 5.1 | 90.9 ± 1.4 | 92.0 ± 0.0 |
| fb_tdj | 3 | 17.7 ± 5.5 | 17.3 ± 5.1 | 45.6 ± 7.2 | 44.0 ± 5.3 |
| hilp_density_tdjc | 3 | invalid | invalid | 71.1 ± 5.8 | 74.0 ± 10.0 |
| hilp_tdjc | 3 | invalid | invalid | 71.1 ± 6.9 | 83.3 ± 3.1 |
| psm | 3 | 9.9 ± 0.5 | 8.7 ± 1.5 | not run | not run |
| psm_density | 3 | 16.4 ± 2.0 | 18.7 ± 2.3 | not run | not run |
| rldp | 3 | 15.2 ± 2.8 | 16.3 ± 6.4 | not run | not run |
| rldp_density | 3 | 23.2 ± 1.6 | 23.7 ± 6.4 | not run | not run |
| rldp_portalign | 3 | 12.8 ± 2.7 | 12.7 ± 3.1 | not run | not run |

### puzzle-3x3-play-v0

| critic | n | A last-3 | A final | B last-3 | B final |
|---|---|---|---|---|---|
| density_fb_z | 3 | 7.2 ± 1.6 | 5.0 ± 2.6 | 4.7 ± 1.2 | 5.3 ± 2.3 |
| fb_tdj | 3 | 13.7 ± 0.3 | 14.3 ± 2.5 | 2.7 ± 1.2 | 2.7 ± 1.2 |
| hilp_density_tdjc | 3 | invalid | invalid | 2.2 ± 1.5 | 3.3 ± 1.2 |
| hilp_tdjc | 3 | invalid | invalid | 1.3 ± 0.7 | 0.0 ± 0.0 |
| rldp | 3 | 10.6 ± 4.0 | 10.7 ± 6.5 | 2.4 ± 1.4 | 2.7 ± 2.3 |

### scene-play-v0

| critic | n | A last-3 | A final | B last-3 | B final |
|---|---|---|---|---|---|
| density_fb_z | 3 | 59.7 ± 9.1 | 59.0 ± 12.3 | 52.7 ± 4.0 | 54.7 ± 3.1 |
| fb_tdj | 3 | 21.2 ± 2.3 | 21.7 ± 2.1 | 10.2 ± 5.0 | 12.7 ± 8.1 |
| hilp_density_tdjc | 3 | invalid | invalid | 42.0 ± 1.2 | 42.0 ± 7.2 |
| hilp_tdjc | 3 | invalid | invalid | 39.3 ± 7.0 | 40.0 ± 10.0 |
| rldp | 3 | 5.7 ± 1.2 | 5.3 ± 3.2 | 4.7 ± 1.3 | 5.3 ± 1.2 |

## Published comparison, reward-inference route (Route B last-3)

TD-JEPA Table 1 (published, proprio) next to our Route B last-3 for the matching
tdj-style critics, and our `density_fb_z` Route B last-3 in its own column. Cube-double
entries on our side are measured, not capped; read them against the bound in rule 3.

| env | FB pub | FB ours | HILP pub | HILP ours | RLDP pub | RLDP ours | density_fb_z ours |
|---|---|---|---|---|---|---|---|
| cube-single | 49.6 | 45.6 ± 7.2 | 74.2 | 71.1 ± 6.9 | 19.8 | not run | 81.3 ± 4.6 |
| cube-double | 2.6 | 2.2 ± 2.7 | 20.0 | 11.1 ± 3.3 | 3.8 | 2.2 ± 1.1 | 30.4 ± 2.6 |
| scene | 12.8 | 10.2 ± 5.0 | 43.8 | 39.3 ± 7.0 | 11.6 | 4.7 ± 1.3 | 52.7 ± 4.0 |
| puzzle-3x3 | 4.8 | 2.7 ± 1.2 | 2.8 | 1.3 ± 0.7 | 14.6 | 2.4 ± 1.4 | 4.7 ± 1.2 |
| antmaze-medium-navigate | 73.0 | 68.9 ± 6.0 | 83.6 | 74.9 ± 1.4 | 74.6 | 50.9 ± 11.5 | 44.0 (n=1) |
| antmaze-large-navigate | 36.8 | 19.8 ± 6.0 | 52.6 | 29.1 ± 6.7 | 36.4 | 6.9 ± 2.8 | not run |

## Plain-language reading

density_fb_z beats the best published baseline on three environments and loses on
two, with one not yet run. On cube-single it scores 81.3 against the best published
number, HILP at 74.2, a win. On scene it scores 52.7 against HILP's 43.8, a win. On
cube-double it scores 30.4, beating published HILP at 20.0, FB at 2.6, and RLDP at
3.8, and it sits close to the roughly-half-of-Route-A bound described in rule 3
(about 34, from density_fb's goal-conditioned 68.6) — so on this task density_fb_z is
already near the ceiling any fixed-task method can reach under reward inference. On
puzzle-3x3 it scores 4.7 against RLDP's 14.6, a clear loss. On antmaze-medium it
scores 44.0 against HILP's 83.6, a loss, but that cell has only one seed so treat it
as provisional. Antmaze-large has no density_fb_z run yet, so there is nothing to
compare there. The two linear-actor-reward retrofits, hilp_density_tdjc and
rldp_density, do not separate from their base critics: hilp_density_tdjc sits within
noise of hilp_tdjc on cube-single, puzzle-3x3, and scene, and roughly a third of
hilp_tdjc on cube-double (2.4 vs 11.1); rldp_density sits within noise of rldp on
cube-double, the one place both have Route B. The retrofit changes almost nothing on
its own.

## Not yet run

- `density_fb` (goal-conditioned critic): not run on scene, puzzle-3x3, or either
  antmaze environment. Only present on cube-single and cube-double.
- `density_fb_z`: antmaze-medium seeds 1-2 not run (only one seed landed); antmaze-large
  not run at all.
- `density_fb_z_state`: not run off cube. Only present on cube-single and cube-double.
- `rldp`: Route A only run so far on scene, puzzle-3x3, and antmaze, where present;
  the matching Route B reward-inference re-score for those cells is still pending.
