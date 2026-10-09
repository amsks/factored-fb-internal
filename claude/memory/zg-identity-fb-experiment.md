---
name: zg-identity-fb-experiment
description: "RUNNING (launched 2026-06-25, tmux zg_cube): FB with z=g directly (fixed_b=identity, B=I, norm_z off) + 100% on-trajectory HER (train_goal_ratio=1.0), cube 1Mx3, two critic modes. Tests whether the learned backward map B is FB's bottleneck."
metadata: 
  node_type: memory
  type: project
  originSessionId: 592e4344-752f-4754-87f1-5a90dc556133
---

**Hypothesis (user's):** FB's gap to GCIQL on cube is the **learned backward map B**
compressing every goal through a shared bottleneck. Test: strip B entirely — use the
goal as the task vector DIRECTLY (z = g), give FB GCIQL's exact goal distribution
(100% on-trajectory HER), and see if it recovers. If z=g still underperforms, the
bottleneck is the bilinear F·z value form, not B.

**NEW agent feature added this session — `fixed_b=identity`** (z = g literally):
- `agents/fb/fixed_backward.py`: `FixedIdentityBackward(obs_dim)` — B(obs)=obs, param-free, z_dim=obs_dim.
- `agents/fb/model.py`: identity branch sets z_dim=obs_dim AND **norm_z=False** (so z=g, not √d·g/‖g‖; project_z becomes a no-op).
- `agents/fb/agent.py`: sync `self.z_dim=self.model.z_dim` after model build (for the IQL value MLP); add identity to `_use_eval_context`; ortho loss now gated `fixed_b=='none'` only (param-free B skips it); guard `backward_optimizer`/backward soft-update against None in the IQL path (was unguarded → "Tensor list must have at least one tensor"); `eval_context` identity branch → z = normalized goal obs.
- `evals/ogbench.py`: source `goal_obs` (=g*) for identity too (was goal_cond-only).
- Tests: `tests/test_fixed_identity_backward.py` (new) + existing fixed_b/iql/traj suites all pass (25). End-to-end cube smoke OK for both critic modes.
- **ALL UNCOMMITTED** (working tree). Conventions: code commits OK, no Claude co-author [[commit-no-claude-coauthor]].

**RUNNING — tmux `zg_cube`, cube-single, 1M × 3 seeds, `scripts/launch_zg_cube.sh`,
logs `/dev/shm/zg_cube_logs/<tag>.log`.** Base matches bridge rows exactly
(`agent=fb_flowbc domain=cube_single ortho_coef=1000 lr_b=1e-4 bc_coeff=3.0`; ortho/lr_b
inert under identity) + `fixed_b=identity train_goal_mode=traj train_goal_ratio=1.0`:
- **zg_iql** (`zgiql_s*`, GPUs 0/1/2, `critic_mode=iql`) → wandb `fb_cube_zg_iql`.
  Compare vs bridge **FB-IQL-critic ~23.6 @1M** (same but learned B + ratio 0.5).
- **zg_her** (`zgher_s*`, GPUs 5/6/7, `critic_mode=measure`) → wandb `fb_cube_zg_her`.
  Compare vs bridge **zHER ~32 @1M** (same but learned B + ratio 0.5).

**train_goal_ratio semantics (confirmed):** 1.0 = ALL z's are the goal (on-trajectory in
traj mode); 0.0 = all random. User had it backwards. So these runs are 100% on-traj goals.

**TWO caveats for interpretation:**
1. The comparison vs zHER/IQL-critic rows changes TWO things at once (z=g AND ratio 0.5→1.0).
   A clean z=g-only ablation would also need learned-B @ ratio 1.0 as control (not run).
2. **zg_her (measure mode) is conceptually confounded:** FB's measure implies reward
   r_z(s)=z·B(s); with identity B that's g·s (LINEAR), maximized by moving in g's
   direction, NOT by reaching g. So a low zg_her number is partly expected and does NOT
   cleanly indict B. **zg_iql is the fair test** (its IQL target is real goal-reaching
   success via cube physics). FB-IQL is cube-only (success uses cube xyz slice 14:17).

**RESULTS @1M (DONE, EXIT=0, 3 seeds):**
- **zg_iql** (z=g + IQL critic): 0.42/0.40/0.50 → **~0.44**. vs FB-IQL-critic learned-B **~0.24**.
  → z=g ~DOUBLES it and edges plain FB (~0.43). Strong support that learned B is a bottleneck.
  Critic healthy (Q≈-15, V≈-14.6, iql_r_pos≈0.21, q_loss→0.22).
- **zg_her** (z=g + measure): 0.28/0.18/0.10 → **~0.19**. vs zHER learned-B **~0.32** → WORSE.
  Confound CONFIRMED: measure+identity B ⇒ linear reward g·s, q EXPLODED (q≈1460, fb_loss≈95).
  Uninformative about B. zg_iql is the real signal.
- Caveat: both also moved train_goal_ratio 0.5→1.0 (2-var change vs anchors) → phase-2 controls added.

**Scene sanity (g3/g4 @500k, from [[sanity-antmaze-scene-campaign]]):** GCIQL-bilinear-d50+flow
~0.29 (0.29/0.32/0.26) ≫ FB-flowbc ~0.05 (0.06/0.02/0.06). Same direction as cube; opposite antmaze.

**NEW FEATURE — `decouple_actor_z` (GCIQL-style value/actor goal split for FB):** measure-mode
only. Critic keeps the broad `sample_mixed_z` (random dirs + B(buffer goals) + off-diag
negatives); the ACTOR instead trains on `z_actor = B(on-traj future)` (100% reachable, like
GCIQL `actor_p_trajgoal=1.0`), detached. Code: `agents/fb/agent.py` update() computes z_actor
from `batch["future"]`; new ctor arg `decouple_actor_z` (default False=byte-identical);
`train.py` plumbs it + triggers `future_gamma`; `configs/agent/fb.yaml` knob. Tests
`tests/test_decouple_actor_z.py` (actor z≠critic z when on; ==when off) pass. Rationale:
GCIQL separates "what the value learns about" (incl. unreachable negatives) from "what the
actor practices" (only reachable goals); stock FB FUSES them into one shared z → actor wastes
capacity chasing non-reachable task directions. RUNNING: `fb_cube_decouple` (cube 1M×3, GPUs
0-2, fixed_b=none learned B, measure). Compare vs plain FB ~43.

**PHASE-2 controls (DONE 2026-06-29, tmux `phase2`, success/overall @1M from phase2_logs):**
- **iql_lb** (IQL + LEARNED B + ratio1, single-var control for zg_iql 0.44-vs-0.24):
  0.36/0.22/0.04 → mean **~0.21**. This is THE clean control: identity-B(z=g)+IQL@r1 = 0.44
  vs learned-B+IQL@r1 = 0.21 — the ratio is now held fixed, so **z=g (stripping learned B)
  is what doubles it → learned B IS the bottleneck, confirmed cleanly.**
- **her_lb** (HER + learned B + r1): 0.10/0.18/0.06 → **~0.11**.
- **her_idnorm** (HER + identity B but NORMALIZED z): 0.22/0.18/0.20 → **~0.20** ≈ zg_her 0.19
  → bounding/normalizing z does NOT fix the measure+identity-B blowup (still confounded, still low).
NEW code: `fixed_b=identity_norm` mode (B=I, norm_z kept True). All tests pass.
**Backed up** to ~/bridge_ckpt_backup. (Also in phase2: `decpl`=decouple measure+perm 0.39 —
see [[decouple-critic-campaign]]; antmaze G1/G2 controls resolved — see [[sanity-antmaze-scene-campaign]].)

Relates: [[handoff-bridge-10seed-1M-campaign]] (the @1M anchors), [[fb-state-run-name-map]],
[[sanity-antmaze-scene-campaign]] (also running). Conventions: [[wandb-entity-amsks]],
[[always-run-in-tmux]], [[factored-fb-test-env]].
