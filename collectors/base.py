"""collectors/base.py — roll a policy in an EnvAdapter, write unified-format data.

The collection side of the alignment: drive any env (rlmini/Metaworld/ManiSkill,
via envs.registry.make_env) with a policy and record per-episode trajectories in
the unified format (data/unified.py), which FB training then consumes. A random
policy is provided for pipeline sanity checks; a TD-MPC2 collector (quality play
data) is the intended follow-on (collectors/tdmpc2_collector.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from data.unified import dataset_dir, write_episode, write_meta

# A policy maps a flat observation to an action.
Policy = Callable[[np.ndarray], np.ndarray]


class RandomPolicy:
    """Uniform action_space sampling — for dataflow sanity checks / exploration mixtures."""

    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


def collect_dataset(
    env,
    policy: Policy,
    *,
    name: str,
    data_path: Union[str, Path],
    n_episodes: int,
    max_steps: Optional[int] = None,
    seed: int = 0,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Roll ``policy`` in ``env`` (an EnvAdapter) for ``n_episodes`` and write unified
    per-episode .npz under ``<data_path>/<sanitized name>/buffer/`` + a meta.json.
    Returns the dataset root."""
    root = dataset_dir(data_path, name)
    buffer_dir = root / "buffer"
    buffer_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0]) if env.action_space.shape else 0
    total = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        O, A, NO, AG, R, TE, TO, SU = [], [], [], [], [], [], [], []
        t = 0
        while True:
            action = np.asarray(policy(obs))
            nobs, reward, terminated, truncated, info = env.step(action)
            O.append(np.asarray(obs, dtype=np.float32))
            A.append(action.astype(np.float32) if action.ndim else action.reshape(1).astype(np.float32))
            NO.append(np.asarray(nobs, dtype=np.float32))
            AG.append(np.asarray(info["achieved_goal"], dtype=np.float32))
            R.append(float(reward))
            TE.append(bool(terminated))
            TO.append(bool(truncated))
            SU.append(bool(info.get("success", False)))
            obs = nobs
            t += 1
            if terminated or truncated or (max_steps is not None and t >= max_steps):
                break
        traj = {
            "observation": np.stack(O),
            "action": np.stack(A),
            "next_observation": np.stack(NO),
            "achieved_goal": np.stack(AG),
            "reward": np.asarray(R, dtype=np.float32),
            "terminal": np.asarray(TE, dtype=bool),
            "timeout": np.asarray(TO, dtype=bool),
            "success": np.asarray(SU, dtype=bool),
        }
        write_episode(buffer_dir, ep, traj)
        total += t

    meta = {
        "env_family": "rlmini",
        "env_name": name,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "achieved_goal_dim": int(AG[-1].shape[0]) if AG else 0,
        "n_episodes": n_episodes,
        "n_transitions": total,
        "collector": {"policy": type(policy).__name__, "seed": seed},
    }
    if meta_extra:
        meta.update(meta_extra)
    write_meta(root, meta)
    return root
