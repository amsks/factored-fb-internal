---
name: fb-to-gciql-bridge-knobs
description: "IN PROGRESS (2026-06-29): implementing 5 FB knobs to morph FB-IQL(learned B) -> bilinear-GCIQL; I3/I4 DONE, I1/I2/I5 + run ladder + paper update TODO."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29d24434-2cf7-47b2-89cc-f7b4407c6f44
---

**Goal:** make FB reach bilinear-GCIQL+Flow (and stock-GCIQL+Flow) via Hydra knobs, then
A/B at 3 seeds x 500k vs priors re-extracted @500k on seeds {0,1,2}. Base for the bridge
= **FB-IQL critic with LEARNED B** (`fixed_b=none`, `critic_mode=iql`) — NOT z=g; bilinear
GCIQL's psi(g) IS the learned B (IQL-trains it end-to-end). See [[handoff-iwr-program]].

**The 5 implementation items (in `agents/fb/agent.py`, config `configs/agent/fb.yaml`, plumb `train.py`):**
- **I1 `critic_goal_mix=[traj,cur,rand]`** — value-goal sampler PLUMBED into the IQL path
  (currently the IQL path ignores train_goal_ratio/decouple — `diql≡iqllb`). GCIQL = [0.5,0.2,0.3].
  cur = current state as goal (="off-trajectory"). **TODO (medium — the real work).**
- **I2 `actor_goal_mix=[traj,cur,rand]`** (default [1,0,0]) — actor-goal sampler/decouple in IQL. **TODO.**
- **I3 `decondition_F` (bool)** — feed zeros to F's task slot => Q=<phi(s,a),z> task-agnostic.
  **DONE.** `_fb_z` gates on `onestep or decondition_F`; centralized `_fb_Q_heads()` used by critic+actor.
- **I4 `d_norm` (bool)** — scale bilinear value by 1/sqrt(z_dim) (`_value_scale` property). **DONE.**
- **I5 GCIQL dataset goal-matching reward** — replace FB-IQL's cube-xyz `iql_thresh` success
  (`agent.py:~449` `||next_phys[14:17]-fut_phys[14:17]||<thresh`) with GCIQL's relabeled-goal
  index match (reward {0,-1}, mask=1-reached). **TODO (medium).** USER DECISION: use dataset goal-match.

**DONE so far (TDD, 2026-06-29):** I3+I4 — `tests/test_decondition_dnorm.py` (6 tests, green);
16 FB-family regression tests green; config knobs + `train.py` plumbing added.

**Left UNIMPLEMENTED (known residual gap to faithful bilinear-GCIQL — user scoped out for now):**
bilinear V head (V stays MLP([s,z])), min(q1,q2) vs FB's mean-of-2-heads, actor-cond z vs raw g.
So the "all-on" run approximates bilinear-GCIQL up to these 3.

**LAUNCHED 2026-06-29 (tmux `bridgeknobs`, `scripts/launch_bridge_knobs.sh` DRY_RUN=0 STEPS=500000):**
15 FB runs (5 variants x3 seeds @500k), round-robin GPUs 0-7, logs `/dev/shm/bridge_knobs_logs/<tag>.log`.
⚠️ BUG FIXED at launch: script was missing `train.py` (`$PY $cmd` -> `$PY train.py $cmd`); first launch
failed instantly ("can't open file 'agent=fb_flowbc'"), fixed + relaunched. NB OpenGL "Failed to load
libOpenGL.so.0" INFO lines in logs are harmless (eval-env). First train metric appears after slow step-0
flow-actor eval. Done check: `grep -l train/q_loss /dev/shm/bridge_knobs_logs/*.log`; success via wandb
amsks/factored-fb groups `fb_bridge_<tag>`.

**Run ladder (cube 3 seeds x 500k; learned B, critic_mode=iql, GCIQL reward I5, ortho 0,
flow actor, actor_goal_mix=[1,0,0], lr 3e-4):**
- B0 base (decondition_F F, d_norm off, critic_mix=[1,0,0]) — re-anchored FB-IQL
- B1 +decondition_F · B2 +d_norm · B3 +critic_mix=[0.5,0.2,0.3] · B4 all-on (~bilinear-GCIQL)
= 5 variants x3 = **15 runs**. B1-3 give marginal effects vs B0; B4 = combined.
Compare vs priors @500k {0,1,2}: bilinear-GCIQL 92, FB-IQL-critic 24, zg_iql 44, plain FB, stock+flow 73.

**RESULTS @500k DONE (2026-06-29, the 15 FB runs finished, flow actor, 3 seeds):**
Goal-matching ladder (all FlowBC): CRL 30 / plain FB 34 → **FB-IQL goal_match `r0` 59** →
+decondition_F `r1` 58 / +d_norm `r2` 65 / +both `r3` 57 → stock GCIQL+flow 71 / bilinear-GCIQL 90.
**KEY FINDINGS:** (1) the OBJECTIVE (measure→IQL goal-match) is the lever — doubles FB (30→59),
actor+form fixed. (2) Structural knobs (decondition φ, √d) are FLAT (57-65, overlapping n=3).
(3) ⚠️ **The paper's FB-IQL=24 "value-blind" spine is WRONG** — that used lr1e-4+ortho1000 (collapsed
2/3 seeds); at lr3e-4+ortho0+goal-match it's STABLE ~59. The 24 was an lr/ortho artifact. (4) Real
gap to GCIQL (59→71→90) NOT closed by our knobs (candidates: bilinear V, min-Q, learned ψ).
Caveats: n=3 @500k; plain FB still climbing @500k (its @1M=48); lr vs ortho not isolated.
a0_cubexyz (cube_xyz reward) ≈ r0 (62 vs 59) ⇒ the reward criterion swap doesn't matter; lr/ortho do.

**PAPER UPDATED (2026-06-29):** added the 500k goal-match ladder subsection; added two-tier framing
(Part 1 = 500k/3-seed dev = the ladder; Part 2 = 1M/10-seed final = the grid); **spine reframed in 3
places** (ladder "Reading it", §"importing doesn't help" red Revision note, conclusion) — FB-IQL ~59
not value-blind-24; date/subtitle updated. **Physical 2-part re-bucket DONE:** top-level sections =
Two-reporting-tiers / Part 1 Development (500k,3seed = ladder, moved to front) / Part 2 Final (1M,10seed
= all @1M eliminations + grid) / Cross-domain / Conclusion. Grid's internal "Part A/B" renamed "Grid A/B"
to avoid collision. PDF rebuilds clean (140KB). STILL TODO: confirm ladder at 1M-10seed; design-audit section.

**FINAL TASK (user-requested):** update `PAPER/fb-gciql-bridge/fb-gciql-bridge.tex` with a
"what changes from one algorithm to another" section — the per-knob design-decision audit
(FB -> GCIQL+Flow and -> bilinear-GCIQL+Flow), grounded in the audit done this session
(critic struct / objective / sampling / actor / optim axes). Conventions: [[commit-no-claude-coauthor]],
[[factored-fb-test-env]] (`.venv/bin/python -m pytest`), [[always-run-in-tmux]].
