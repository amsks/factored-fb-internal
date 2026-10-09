---
name: tdmpc2-cube-failure-diagnosis
description: "Why the TD-MPC2 state agent gets 0% on cube-single — accurate-but-uncontrollable world model, not collapse or horizon"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7dcdc9f6-0108-490b-bbe3-720d78ad7ae6
---

TD-MPC2 state run `20260529_002709__tdmpc2_state_10seeds` (cube-single-play-v0, 1M steps, 10 seeds) gets eval/success ≈ 0 (9/10 seeds flat 0.0 / reward −200, the sparse floor; s0 had one fluke task5 episode → 0.02). Diagnosed 2026-05-29 with three probes on s0/step_1000000.pt:

- **Latent probe** (`scripts/agents/tdmpc2/latent_probe.py`, drives `evals/phase_probe.py::tdmpc2_latent_phase_probe`): cube_xyz linearly decodable from latent at R²≈0.999, open-loop MSE flat over 3 steps → world model is NOT collapsed. (`consistency_loss≈0` in training is accurate next-latent prediction, NOT collapse — easy to misread.)
- **Horizon probe** (`scripts/agents/tdmpc2/horizon_probe.py`): success 0.000/−200 at H=3, 8, AND 16 → not a myopic-planner / short-horizon problem.
- **Action-sensitivity** (inline): over 8 imagined steps the action-controllable cube displacement is only ~0.0155 m (cross-action std), vs 0.04 m success threshold and ~0.1–0.3 m needed to reach a goal. Zero-action drift 0.022 m, real 0.060 m.

- **Real-env rollout** (faithful, info["success"]/privileged block pos): 0% is REAL, not an eval/integration bug. Actions are real (norm ~1.6, adim 5), goal correctly wired (z=[0.425,-0.1,0.02], cube ~0.19m away). Agent LEARNED TO REACH (hand→cube min 0.035–0.044m, threshold 0.06) but NEVER grasps/transports: cube moves 0.000m in 2/3 eps despite hand on it; in 3rd it's knocked 0.175m AWAY from goal. Fails at the manipulate phase, not reach.

**Root cause:** model is accurate but uncontrollable. Cube = 3 of 28 obs dims and is mostly static in play data, so MSE/consistency is dominated by easy arm dynamics; the contact-rich gripper→cube coupling is under-learned. The planner steers the arm fine (directly actuated) but has no control authority over the cube → finds "reach" plans, never "grasp-lift-place" → all plans tie at the floor.

**Why it matters:** the two cheapest fixes are ruled out by evidence (horizon, representation). Real levers, in order: (1) up-weight cube/contact channels in the world-model loss in `agents/tdmpc2/world_model.py`; (2) dense distance-to-goal reward (necessary, not sufficient); (3) contact-richer data. **How to apply:** before re-launching any TD-MPC2 cube run, change model controllability — not planner knobs. Compare against FB/CRL on same cube-single ([[crl-flowbc-campaign]]) to separate model issue from task hardness. TD-MPC2 agent wiring: obs+3 goal fold, cube xyz at physics[14:17], success @0.04m.
