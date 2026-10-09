---
name: fb-eval-loop-reset-vs-context
description: "FB eval loop — agent.reset() is per-EPISODE, eval_context/_infer_z is per-TASK; do not clear per-task state in reset()"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7ec5ebfa-57d2-4061-8ee8-dac2a4f99f8e
---

In the FB eval loop, `agent.reset()` (envs/rollout.py:20,38) is called at the start of EACH episode, but the eval context z (and, for goal-conditioned V1, `self._eval_goal`) is set ONCE per task by `Evaluator.eval_context`/`_infer_z` (evals/ogbench.py) BEFORE `rollout(..., ctx=z)` runs all `n_episodes`.

**Why it matters:** Clearing per-task state in `reset()` (as the 2026-06-02 goal-conditioned-fb plan Task 3.2 proposed for `_eval_goal`) would wipe the goal after episode 1 and crash episodes 2..n of every V1 task — the goal-conditioned `NoiseConditionedActor.embed_z` requires the goal, so `goal=None` gives a `mat1/mat2` shape mismatch.

**How to apply:** Per-task eval state must NOT be cleared in `reset()`. There is no cross-task leak to fix: `eval_context` overwrites `_eval_goal` every task, V2 (`fixed_b=cube_xyz`) sets it `None`, and baseline never reads it (`act` only uses it when `goal_cond` is true). So Task 3.2 was intentionally skipped (left `reset()` as the inherited no-op). Relates to [[tdmpc2-cube-failure-diagnosis]] (stateful planners DO override reset to clear per-episode state — the opposite case).
