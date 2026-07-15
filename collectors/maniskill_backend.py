"""collectors/maniskill_backend.py — ManiSkill env wrapper for collection.

ManiSkill tasks come from the newt repo (``envs.maniskill.MANISKILL_TASKS`` +
``make_env``, ManiSkill3). newt's code imports the top-level package name ``envs``,
which COLLIDES with Factored-FB's own ``envs/`` package — the two cannot coexist in
one process. So this module lives under ``collectors`` (NOT ``envs``) and is used
only by the standalone ManiSkill collector (scripts/collect_maniskill.py), which
runs in a mani_skill-capable venv and never imports Factored-FB's ``envs`` package.
The collected unified-format data is then read by FB training offline
(train.py build_env=false), which needs no ManiSkill env in-process.

Presents the EnvAdapter surface (flat obs, 5-tuple step, info with
success/achieved_goal/goal). achieved_goal / goal are best-effort from
``env.unwrapped`` (obj/cube pose, goal_site); FB training doesn't use them, the
(future) evaluator does.
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

_MAKE_ENV = None


def _load_newt_make_env(newt_tdmpc2: Optional[str] = None):
    """Import newt's envs.maniskill.make_env via the same stub trick newt's own
    sanity script uses (register a lightweight ``envs`` package pointing at newt's
    envs dir so ``import envs.maniskill`` resolves there and does not run newt's
    eager envs/__init__.py). Cached; call only in a process that does NOT use
    Factored-FB's ``envs`` package."""
    global _MAKE_ENV
    if _MAKE_ENV is not None:
        return _MAKE_ENV
    newt_tdmpc2 = newt_tdmpc2 or os.environ.get("NEWT_TDMPC2_ROOT", "/raid/tongle/rl/newt/tdmpc2")
    os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
    if newt_tdmpc2 not in sys.path:
        sys.path.insert(0, newt_tdmpc2)
    if "envs" not in sys.modules:
        pkg = types.ModuleType("envs")
        pkg.__path__ = [os.path.join(newt_tdmpc2, "envs")]
        pkg.__package__ = "envs"
        sys.modules["envs"] = pkg
    import envs.maniskill as ms  # noqa: E402  (needs the stub + sys.path first)

    _MAKE_ENV = ms.make_env
    return _MAKE_ENV


def _pos3(x) -> np.ndarray:
    """Coerce a ManiSkill Pose / tensor / array into a 3-vector (float32)."""
    if hasattr(x, "p"):          # sapien Pose -> position
        x = x.p
    if hasattr(x, "detach"):     # torch tensor (GPU)
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    return a[:3]


class ManiSkillEnv:
    """EnvAdapter over newt's ManiSkill make_env for one ms-<task>."""

    def __init__(self, task: str, *, seed: Optional[int] = None, render_size: int = 64,
                 newt_tdmpc2: Optional[str] = None):
        make_env = _load_newt_make_env(newt_tdmpc2)
        cfg = SimpleNamespace(task=task, obs="state", render_size=render_size)
        self._env = make_env(cfg)
        self.task = task
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.goal_space = None
        self.max_episode_steps = getattr(self._env, "max_episode_steps", None)
        self._seed = seed

    # -- achieved_goal / goal from env.unwrapped (best-effort) --------------

    def _achieved(self) -> np.ndarray:
        u = self._env.unwrapped
        for attr in ("obj", "cube", "peg", "sphere"):
            o = getattr(u, attr, None)
            if o is not None and hasattr(o, "pose"):
                return _pos3(o.pose)
        return np.zeros(0, dtype=np.float32)

    def _goal(self) -> np.ndarray:
        u = self._env.unwrapped
        gs = getattr(u, "goal_site", None)
        if gs is not None and hasattr(gs, "pose"):
            return _pos3(gs.pose)
        gp = getattr(u, "goal_pos", None)
        if gp is not None:
            return _pos3(gp)
        return np.zeros(0, dtype=np.float32)

    def _enrich(self, info: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(info)
        out["success"] = bool(float(info.get("success", 0.0)) > 0)
        out["achieved_goal"] = self._achieved()
        g = self._goal()
        if g.size:
            out["goal"] = g
        return out

    # -- gymnasium-style API -----------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        seed = seed if seed is not None else self._seed
        if seed is not None:
            np.random.seed(seed)
            self._env.action_space.seed(seed)
        obs, info = self._env.reset()
        return np.asarray(obs, dtype=np.float32), self._enrich(info)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), self._enrich(info)

    def render(self):
        return self._env.render()

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()
