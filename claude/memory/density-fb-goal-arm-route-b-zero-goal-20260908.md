---
name: density-fb-goal-arm-route-b-zero-goal-20260908
description: 2026-09-08 goal-conditioned density_fb has NO reward-inference interface; its Route B cells (cube-single 11.1, cube-double 8.9) are goal=0 rollouts (inferred_cond_norm=0.0 on every hub row); eval-only, ckpts valid; density_fb_z is the real Route B column and WINS on cube (75 / 30 vs HILP 71 / 11, FB 46 / 2)
metadata:
  type: project
---

**Fact.** `impls/critics/density_fb.py::infer_cond` returns a zero goal placeholder when the reward-sample index is off (goal arm). The Route B block in `impls/main.py` (~515-546) never calls `infer_cond_noop_reason`, and `tdjtab_handback.py` only refuses on `goals_dropped`, so the tdjtab manifest's `density_fb` rows (which carry `eval_relabel_*` flags) produced scored rows with `inferred_cond_norm = 0.0` on every task. Cells 93.1/11.1 (cube-single) and 68.6/8.9 (cube-double): left halves (Route A) CLEAN, right halves INVALID. The "Density-FB is the mirror of HILP" and "Route B collapses on conjunctions 68.6→8.9" readings are withdrawn. Same class of bug as [[route-a-goal-blind-bug-20260908]].

**Under the TD-JEPA protocol (Route B, 1M, 3 seeds, last-3)** the z variant `density_fb_z` is the best method on both cube envs: cube-single .75 (HILP ours .71 / pub .74; FB .46 / .50; RLDP .25 / .20; TD-JEPA pub .34), cube-double .30 (HILP .11 / .20; FB .02 / .03; RLDP .02 / .04). The earlier ".43/.25 z collapse" was the pre-`latent_centered` actor reward ([[density-fb-z-port-gap-20260906]]), superseded. No Density-FB number exists off cube yet (scene/puzzle/antmaze queued on luis/evalbox).

**Why:** with only the goal arm's placeholder the eval had nothing to infer from; nobody guarded for it.

**How to apply:** (1) never report Route B for the goal arm without a reward interface; candidate = goal readout from reward samples (argmax-reward or value-ranked top-k positive sample as the goal), scored 2026-09-08 in scratchpad readout probe; (2) a Route B guard (NaN + invalid_reason on no-op) was added 2026-09-08 on this box, UNCOMMITTED; (3) the luis `density_fb` runs are burning ~66 GPU-h of Route B eval on the placeholder unless `$RELABEL` is dropped; (4) cube-double campaign reward is a COUNT of placed cubes (-1/0/1), not a strict conjunction; strict positives are 1-13 of 10k samples ([[z-index-cube-double-conjunction-20260907]]).
