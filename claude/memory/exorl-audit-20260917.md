---
name: exorl-audit-20260917
description: 2026-09-17 four-subagent ExoRL audit — pipeline clean, FB port faithful (default fb.yaml collapses, reference cfg = published), Density-FB gap = flat hit kernel on full-obs block + linear proxy actor reward + maze actor collapse; 2M reference recipe no better than 1M
metadata:
  type: project
---

Audit note: `docs/notes/2026-09-17-exorl-audit.md`, four full reports under
`docs/notes/2026-09-17-exorl-audit/` (seam worktree, UNCOMMITTED as of 2026-09-17).

Facts (2026-09-17):
- ExoRL pipeline (data 5000 RND eps, relabel, task inference, eval) matches official code. Not the bottleneck.
- FB loss/towers match td_jepa term for term. Reference-config walker 1M = 920/804/400/985 (>= published).
  Default `fb.yaml` collapses on walker (357/49/16/853 @500k): left encoder 512x4->50-d sphere, batch 256,
  tau .005. Every ExoRL FB baseline row must use `scripts/launch_walker_fbref.sh`'s config; only walker exists.
- Density-FB 2M/5-seed domain means: walker 634, cheetah 232, quadruped 569, maze 302 vs published FB
  walker 654-811, cheetah 513-673, quadruped 569-606, maze 427 (RLDP paper).
- Causes: (1) hit radius calibrated by random-pair hit rate on the full obs block is nearly flat in 17-78 dims
  (bump spreads over 277/1024 walker, 879/1024 quadruped vs 60 cube) -> encoder/actor training rows carry no
  task signal; (2) actor's linear proxy reward explains 73-80% walker, 14-26% maze reward; cheetah rewards
  ~45% linear; exact-reward kernel readout never run on ExoRL; (3) maze seed 3 = collapsed TD3 actor
  (constant action, BC weight 0); (4) 2M tau .001/[1024] recipe scored BELOW the 1M tau .005/[512]x3 runs
  on walker; walker flat from 250k. Inference is stable (cos>=.993 across 50k draws) — not noise.
- Walker 4th task: ours/TD-JEPA = spin; Touati/HILP/RLDP = flip. TD-JEPA maze (6x margin, 8 tasks) not comparable.

**Why:** the user asked why ExoRL numbers are bad; these are the verified mechanisms.
**How to apply:** don't re-audit the pipeline; fix = kernel readout port + hit-kernel concentration
target + sphere-fraction/BC ablation + FB-reference runs on cheetah/quadruped/maze. See
[[handoff-20260916-density-fb-5seed-campaign]], [[kernel-actor-port-and-runs-20260908]].

**2026-09-17 FIX QUEUED (user chose radius-only):** `scripts/calibrate_hit_spec.py --target_support 85.6`
(cube-single's measured bump support) → walker 0:24:9.5603, cheetah 0:17:4.6159, quadruped 0:78:6.2467,
maze unchanged. Phase 3 (20 jobs, 1M, save `exp/exorl_dfb_sup86/`) appended to `logs/exorl_dfb/manifest.tsv`
via `gen_phase3.sh`; `dfbsched` runs it after OGBench Phase 2. Score with `scripts/exorl_paper_eval.py`
(50k samples) → `exp/paper_eval_exorl_sup86_1M.jsonl`. All of this UNCOMMITTED.

**2026-09-17 16:40Z PHASE 3 RESULT = NEGATIVE.** Support-calibrated radius (1M, 5 seeds, paper protocol):
walker 553 / cheetah 212 / quadruped **140** / maze 273 (all 294) vs old-radius 2M 634 / 232 / 569 / 302 (434).
Quadruped collapsed on all tasks+seeds; not a budget effect. Sharper goal bump on the full-obs block is NOT the
lever (radius also scales B's input 3.4x on quadruped). Next candidate = the actor's linear-proxy reward
(audit finding 2: exact-reward kernel readout never run on ExoRL). Rows: `exp/paper_eval_exorl_sup86_1M.jsonl`.
