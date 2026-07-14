"""envs/adapter.py — the single env-family boundary for Factored-FB.

Every environment family (OGBench today; rlmini-envs / Metaworld / ManiSkill in
the alignment work) is consumed by the training loop, rollout, and evaluator
through ONE structural contract: :class:`EnvAdapter`. Agents and ``train.py``
never import an env family directly — they go through ``envs.registry``.

Contract (see docs/alignment_design.md §3a / §7):

  reset(seed) -> (obs_vec: np.ndarray[obs_dim], info: dict)
  step(action) -> (obs_vec, reward: float, terminated: bool, truncated: bool, info: dict)
  render() -> np.ndarray[H, W, 3]

``obs_vec`` is a FLAT policy observation (float). ``info`` SHOULD carry, when the
family can provide them (rlmini/Metaworld do; OGBench provides ``success``):

  info["success"]        -> bool   (per-step task success; consumed by evaluators)
  info["achieved_goal"]  -> np.ndarray[g_dim]   (goal-relevant slice; generalizes
                            OGBench's ``physics`` — used for relabeling / z-inference)
  info["factored_state"] -> dict[str, np.ndarray]   (object-centric view; optional,
                            enables the FB interaction-variable analysis)

Optional attributes (rlmini/Metaworld set these; OGBench leaves them None):

  goal_space:    gym.spaces.Box | None
  factor_schema: dict[str, int]           # {object_name: size}
  achieved_goal(obs=None, factored=None) -> np.ndarray
  set_task(task) -> None                   # variant / singletask selection

OGBench envs already satisfy the reset/step/render/space parts of this protocol,
so the ``ogbench`` family passes them straight through (P0). The rlmini adapter
(P1) is the first concrete implementation that fills in achieved_goal/goal_space/
factored_state from rlmini's native ``get_achieved_goal`` / ``factored_state``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

try:  # gymnasium is a hard dep of the repo; kept import-safe for tooling
    import gymnasium as gym

    _Space = gym.spaces.Space
except Exception:  # pragma: no cover - gymnasium always present at runtime
    _Space = Any  # type: ignore


@runtime_checkable
class EnvAdapter(Protocol):
    """Structural contract every env family presents to Factored-FB.

    This is a typing Protocol (structural / duck-typed): OGBench envs match the
    reset/step/render/space members without subclassing, and concrete adapters
    (e.g. the rlmini adapter) implement the full surface. Nothing in the repo
    needs to subclass it; it documents and (via ``runtime_checkable``) lets us
    assert the surface in tests.
    """

    observation_space: _Space
    action_space: _Space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]: ...

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: ...

    def render(self) -> Optional[np.ndarray]: ...


# Well-known info keys, centralized so producers (adapters) and consumers
# (rollout, evaluators, collectors) never disagree on spelling.
INFO_SUCCESS = "success"
INFO_ACHIEVED_GOAL = "achieved_goal"
INFO_FACTORED_STATE = "factored_state"
