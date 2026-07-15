"""envs/rlmini_adapter.py — EnvAdapter over rlmini-envs (P1).

Wraps a *raw* ``rlmini_envs.make(...)`` environment (dict observation, native
4-tuple ``step``) and presents Factored-FB's :class:`envs.adapter.EnvAdapter`
contract: a FLAT float observation, a gymnasium 5-tuple ``step``, and an ``info``
enriched with ``success`` / ``achieved_goal`` / ``factored_state`` / ``goal``.

We deliberately wrap the raw env (not rlmini's ``GymnasiumWrapper`` +
``FactoredObservationWrapper`` stack) so the adapter keeps full access to the
whole ``factored_state`` every step — including the ``Goal`` factor, which the
flat wrapper would drop. The adapter then:

  * builds the flat policy observation from ``obs_keys`` (default: every physical
    factor EXCEPT the goal factor — FB conditions on z, not on the goal in-obs);
  * extracts ``achieved_goal`` from an explicit per-env spec (rlmini's
    ``Goal.get_achieved_goal`` is unreliable across envs, e.g. Metaworld leaves
    ``target_idx``/``all_names`` unset, so we do not depend on it);
  * carries ``success`` straight from the underlying env's ``info`` (Metaworld /
    ManiSkill provide it).

Requires the ``rlmini-envs`` package (installed in ``.venv-rlmini``; see
docs/alignment_design.md §7). Import is lazy so the OGBench ``.venv`` never needs it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # gymnasium is always present at runtime
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None  # type: ignore

# Non-physical bookkeeping factors that never belong in a policy observation.
_NON_STATE = ("Action", "Reward", "Done")

# Per-env achieved-goal spec: list of (factor_name, start, stop) slices whose
# concatenation is the goal-relevant vector used for relabeling / z-inference.
# Metaworld: hand xyz + manipulated-object xyz covers both reach (hand) and
# pick/push (object) tasks; the evaluator's success still comes from info["success"].
_ACHIEVED_GOAL_SPECS: Dict[str, List[Tuple[str, int, int]]] = {
    "Metaworld": [("Hand", 0, 3), ("Object1", 0, 3)],
}


class RLMiniAdapter:
    """EnvAdapter implementation over one rlmini-envs environment."""

    def __init__(
        self,
        env_name: str,
        variant: str = "default",
        *,
        obs_keys: Optional[Sequence[str]] = None,
        achieved_goal_spec: Optional[Sequence[Tuple[str, int, int]]] = None,
        goal_key: str = "Goal",
        horizon: int = 500,
        seed: Optional[int] = None,
        render: bool = False,
        **make_kwargs: Any,
    ):
        import rlmini_envs

        self._raw = rlmini_envs.make(
            env_name, variant=variant, horizon=horizon, seed=seed or 0, render=render, **make_kwargs
        )
        self.name = getattr(self._raw, "name", env_name)
        self.variant = variant
        self._render_on = render

        all_names = list(getattr(self._raw, "all_names", []))
        object_sizes = dict(getattr(self._raw, "object_sizes", {}))
        object_range = dict(getattr(self._raw, "object_range", {}))
        skip = set(_NON_STATE) | {goal_key}
        self.obs_keys: List[str] = (
            list(obs_keys)
            if obs_keys is not None
            else [n for n in all_names if n not in skip and n in object_sizes]
        )
        self.goal_key = goal_key
        self.factor_schema = {k: int(object_sizes[k]) for k in self.obs_keys}

        # Flat observation Box from concatenated per-factor ranges (float32).
        lows, highs = [], []
        for k in self.obs_keys:
            lo, hi = object_range.get(k, (None, None))
            n = int(object_sizes[k])
            lo = np.full(n, -np.inf) if lo is None else np.asarray(lo, dtype=np.float64).reshape(-1)
            hi = np.full(n, np.inf) if hi is None else np.asarray(hi, dtype=np.float64).reshape(-1)
            lows.append(lo)
            highs.append(hi)
        flat_low = np.concatenate(lows).astype(np.float32)
        flat_high = np.concatenate(highs).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=flat_low, high=flat_high, dtype=np.float32)
        self.action_space = self._raw.action_space
        self.goal_space = getattr(self._raw, "goal_space", None)

        spec = achieved_goal_spec if achieved_goal_spec is not None else _ACHIEVED_GOAL_SPECS.get(self.name)
        if spec is None:  # fall back to the goal factor's own width (achieved == same space as goal)
            gsz = int(object_sizes.get(goal_key, 0))
            spec = [(self.obs_keys[0], 0, gsz)] if gsz and self.obs_keys else []
        self.achieved_goal_spec: List[Tuple[str, int, int]] = list(spec)

    # -- observation / goal extraction -------------------------------------

    def _flat_obs(self, factored: Dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(factored[k], dtype=np.float32).reshape(-1) for k in self.obs_keys]
        )

    def achieved_goal(self, factored: Dict[str, Any]) -> np.ndarray:
        if not self.achieved_goal_spec:
            return np.zeros(0, dtype=np.float32)
        parts = [np.asarray(factored[name], dtype=np.float32).reshape(-1)[a:b]
                 for (name, a, b) in self.achieved_goal_spec]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _enrich(self, factored: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        from envs.adapter import INFO_ACHIEVED_GOAL, INFO_FACTORED_STATE, INFO_SUCCESS

        out = dict(info)
        out[INFO_SUCCESS] = bool(info.get("success", False))
        out[INFO_ACHIEVED_GOAL] = self.achieved_goal(factored)
        out[INFO_FACTORED_STATE] = factored
        if self.goal_key in factored:
            out["goal"] = np.asarray(factored[self.goal_key], dtype=np.float32).reshape(-1)
        return out

    # -- gymnasium-style API -----------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # rlmini envs vary in reset() signature (Metaworld takes `render`, the base
        # takes `seed`/`options`), so seed via .seed() and call a bare reset().
        if seed is not None and hasattr(self._raw, "seed"):
            self._raw.seed(seed)
        result = self._raw.reset()
        obs, info = result if isinstance(result, tuple) and len(result) == 2 else (result, {})
        factored = obs["factored_state"]
        return self._flat_obs(factored), self._enrich(factored, info)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self._raw.step(action)
        factored = obs["factored_state"]
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done and not truncated)
        return self._flat_obs(factored), float(reward), terminated, truncated, self._enrich(factored, info)

    def render(self):
        return self._raw.render()

    def close(self):
        if hasattr(self._raw, "close"):
            return self._raw.close()

    def seed(self, seed):
        if hasattr(self._raw, "seed"):
            return self._raw.seed(seed)


def make_rlmini_env(
    name: str,
    *,
    seed: Optional[int] = None,
    obs_type: str = "state",
    frame_stack: int = 1,
    horizon: int = 500,
    render: bool = False,
    **kw: Any,
) -> Tuple[RLMiniAdapter, Dict[str, Any]]:
    """Registry entry for the ``rlmini`` family. ``name`` is ``Env`` or ``Env/variant``
    (e.g. ``Metaworld/pick-place``). Returns ``(adapter, {})`` to mirror create_ogbench_env."""
    if obs_type != "state":
        raise NotImplementedError("rlmini adapter currently supports obs_type='state' only")
    env_name, _, variant = name.partition("/")
    adapter = RLMiniAdapter(
        env_name, variant or "default", seed=seed, horizon=horizon, render=render, **kw
    )
    return adapter, {}
