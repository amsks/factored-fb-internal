---
name: handoff-20260916-density-fb-5seed-campaign
description: Overnight scheduler running Density-FB 5-seed on ExoRL (phase 1) then OGBench top-up (phase 2)
metadata: 
  node_type: memory
  type: project
  originSessionId: 29587e3c-8b70-46c2-91f2-1fdc51f5a0b0
  modified: 2026-09-16T01:30:23.474Z
---

Overnight 2026-09-16: one scheduler (tmux `dfbsched`, `logs/exorl_dfb/scheduler.py` +
`manifest.tsv`, 2 jobs/GPU on 8 L4s, phase-gated, auto_resume, re-reads manifest per phase)
trains ONLY Density-FB — RLDP/HILP/FB numbers are taken from the TD-JEPA paper.

- Phase 1 (running): `density_fb_z`, 2M steps, 5 seeds, 4 ExoRL domains (walker/cheetah/
  quadruped/point-mass-maze) = 20 runs, save `exp/exorl_dfb/`. Reward-inference eval
  (`--infer_cond_episodes 50000 --eval_on_inferred_cond`) is MANDATORY — the old
  `exp/walker_dfbref` returns were z=0 goal-blind, NOT comparable. Quadruped hit-spec
  calibrated tonight = `0:78:21.087` (random-pair hit rate matched to cube-single 0.0133).
  Config = post-FB-audit walker reference (value [1024], tau 0.001, B [256,256] motivo norm,
  lr 1e-4, motivo TD3 actor, alpha 0, q_normalize off).
- Phase 2 (auto after phase 1): OGBench top-up to 5 seeds (seeds 3,4), flowbc, 1M, Route-B
  relabel eval, 4 manip envs. Two arms: `density_fb_z`->exp/tdjtab (WITH object coords,
  backward_input=hit_spec) and `density_fb_z_state`->exp/dfb_state (WITHOUT = full state).
  16 runs. ⚠ verify with/without mapping + that cube density_fb_z_state seeds 0-2 (in
  `exp/hub/exp/`) aggregate with new 3,4 (in exp/dfb_state).

Decisions locked with user: match TD-JEPA (RLDP/FB/DFB = 3M single-phase, live full ortho,
HILP 1M) was the plan for the FULL 4-method grid, but user then narrowed to Density-FB-only
at 2M/5seed. See docs/notes/2026-09-16-density-fb-5seed-exorl-ogbench.md. Related:
[[kernel-actor-negative-500k-20260909]] [[method-is-the-density-objective]]
[[primary-paper-dir-iclr]].
