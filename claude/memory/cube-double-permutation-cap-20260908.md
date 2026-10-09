---
name: cube-double-permutation-cap-20260908
description: 2026-09-08 PROTOCOL CAP: under reward inference (TD-JEPA protocol) cube-double success for ANY fixed-task method is capped at goal-conditioned success / 2 because OGBench permutes init+goal xyz together per reset (cube_env.py:593-599) and success is per cube index vs permuted target (:709-720), while the relabel reward (env/ogbench.py:82-107; td_jepa metamotivo/envs/ogbench.py:117) uses the canonical assignment. Goal readout from reward samples recovers cube-single (0.90 vs 0.97 env goal) but not cube-double (0.34 vs 0.73)
metadata:
  type: project
---

**Fact.** Goal-readout experiment (1M goal-conditioned density_fb, seeds 0/1/2, campaign Route B protocol, scratchpad report_readout.md): cube-single argmax-reward goal .81, top-8 by critic value .90, env goal .97 (recorded Route B was the invalid .06). cube-double argmax .34, top-8 .15, env goal .73; with permute_blocks=False argmax rises .20→.46 (seed 0); env goal with cubes swapped scores 0.00. Root cause verified in code: OGBench draws a permutation at reset and applies it to BOTH init_xyzs and goal_xyzs, success checks object_joint_i against its permuted mocap target; the reward samples encode the canonical assignment, so half the episodes ask for a different task than the samples describe. Cap = goal-conditioned success / n! (n cubes). Applies equally to HILP/FB/RLDP/TD-JEPA under their published protocol (same canonical relabel), so published cube-double HILP 20.0 is under this cap too.

**Why:** the user wants to win cube-double under the TD-JEPA protocol; the achievable target is the cap (~.37 for our .73 goal critic), not the goal-conditioned .69–.73. density_fb_z at .30 is already near it.

**How to apply:** report the cap in the paper next to the reward-inference column and show the goal-conditioned number alongside; do not chase >.37 on cube-double under reward inference; cube-triple would be capped at 1/6. For cube-single the goal readout (value-ranked top-k positive sample) is a valid reward-only interface for the goal critic (.90). Related: [[density-fb-goal-arm-route-b-zero-goal-20260908]], [[density-fb-actor-exact-reward-readout-20260908]].
