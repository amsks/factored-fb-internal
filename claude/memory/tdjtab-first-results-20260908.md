---
name: tdjtab-first-results-20260908
description: 2026-09-08 first tdjtab cells done — cube-double FB 1.6% / RLDP 2.2% (last-3 estimator) vs td_jepa published 2.6 / 3.8; vista's s2 moved to evalbox on SU exhaustion
metadata:
  type: project
---

**First 14 of 104 cells complete** (8 prior + 6 handed back by evalbox 2026-09-08 09:30Z).

**cube-double, scored the way the table scores** — mean over the LAST 3 eval points
(window 800k/900k/1M, `make_results_table.py:33`), NOT the final step:

| critic | ours | td_jepa published |
|---|---|---|
| FB (`fb_tdj`) | **1.6%** (.010/.017/.020) | 2.6 |
| RLDP | **2.2%** (.020/.027/.020) | 3.8 |

Same ordering, both slightly low. These are the project's **first uncontaminated `fb_tdj` cube cells**
(the old dirs mixed lr_b 1e-4/1e-5). Wall: fb_tdj 8.5 h/1M, rldp 4.2 h for the 500k→1M half.
⚠ `status/*.json` on the hub carry **final-step** numbers (that is the tier-1 schema); the table's
numbers are last-3. Do not mix them — noted in `scripts/tdjtab_handback.py`'s docstring.

## RESULTS TABLE as of 2026-09-08 16:40Z (28/104 done) — last-3 estimator, % success, A=goal / B=reward-inference

    cube-single     fb_tdj 17.7/45.6   hilp_tdjc 12.3/71.1   density_fb 93.1/11.1   density_fb_z 74.9/81.3
                    rldp 15.2/-        hilp_dens 12.3/71.1   rldp_dens 23.2/-
    cube-double     fb_tdj  1.6/ 2.2   hilp_tdjc  2.0/11.1   density_fb 68.6/ 8.9   density_fb_z 33.9/30.4
                    rldp 2.2/2.2       hilp_dens 1.7/2.4     rldp_dens 2.0/0.00
    scene-play      fb_tdj 20.3/ 7.7   hilp_tdjc  4.8/39.3
    puzzle-3x3      fb_tdj 13.7/ 2.7   hilp_tdjc  1.6/ 1.3
    antmaze-medium  fb_tdj 71.1/68.9   hilp_tdjc 11.1/74.9
    antmaze-large   fb_tdj 35.3/31.3   hilp_tdjc  4.2/32.7

**Two patterns, both new and both load-bearing for the paper:**
1. ⚠ **RETRACTED 2026-09-08 19:23** — "HILP's eval-route gap is systematic (5-8x)" is NOT a finding.
   Route A is GOAL-BLIND for hilp/hilp_density/rldp_density (no `map_eval_cond` hook ⇒ actor rolls on
   z=0). Every Route-A number for those three critics in the table above is invalid. See
   [[route-a-goal-blind-bug-20260908]]. The fb_tdj / rldp / density_fb / density_fb_z rows are UNAFFECTED.
2. **Route B tracks reward STRUCTURE, not difficulty.** Works where the reward is a distance to a point
   (antmaze both sizes); collapses where it is a CONJUNCTION (puzzle-3x3's 9 buttons: fb_tdj 13.7→2.7;
   cube-double's 2 cubes: density_fb 68.6→8.9). antmaze-large is HARDER than puzzle by Route A yet
   Route B works there and not on puzzle. Same mechanism as [[z-index-cube-double-conjunction-20260907]].
3. **Density-FB is the mirror of HILP**: best in the table on Route A (93.1 cube-single) and among the
   worst on Route B (11.1) — the central claim in one row. `density_fb_z` is the only method whose
   routes agree (74.9/81.3), at a cost in Route A.

⚠ `cube_double/rldp_density` route_b = **0.00 on all 3 seeds with goals_dropped at the full 5000** —
a REAL zero (goal channel dropped, policy never solves), not a plumbing failure. Caption it that way.

**New tooling on evalbox (uncommitted, and evalbox has NO git creds to push):**
- `scripts/tdjtab_status.py` — campaign dashboard, joins manifest + `assignments/` + `status/` + hub
  `exp/` + local logs. `--all`, `--site X`, `--no-hub`.
- `scripts/tdjtab_handback.py` — writes `status/<run-dir>.json` + uploads the 4 contract files;
  REFUSES on no `params_1000000.pkl`, task-count mismatch between routes, or
  `inferred_cond_goals_dropped == 0`.

**vista is DONE and was SU-starved.** `cube_single/fb_tdj` s0 (2.96 h) and s1 (2.78 h) at 1M,
`lr_b = 1e-4` verified from the written config.json ⇒ contaminated arm genuinely replaced. **s2 was
refused at submit** (9 SU, 8 encumbered by another user, 1 usable, needs 4) — the only cell in the
campaign with no path to completion — so **evalbox took it** (GPU 5, 09:35Z, fresh start). Partition
re-verified `OK: 96 run ids, no site overlap`. A second Claude session on the vista box had submitted
the two real runs and its own stray `smoke_cs_gh`; it cancelled the stray and ran exactly the assigned
ids, so no duplication resulted.

**luis:** Case B executed, 12 `density_fb` cells resubmitted as array 7936970 with the Route-B
overrides; gate now asserts them on all 84. Corrected my cost estimate — the 12 were still PENDING, so
**0 GPU-h discarded, not ~6**. luis's 84 at 24 concurrent (~69 h) is the campaign's critical path;
evalbox has idle GPUs and has asked luis for still-PENDING array indices to take over under the same
reassign-then-cancel protocol that worked for s2.
See [[tdjtab-launched-20260908]], [[eval-is-85pct-of-noncube-run-20260908]].
