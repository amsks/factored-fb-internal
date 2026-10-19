"""MetaWorld goal-observable envs, wrapped as the offline data was collected.

Reset takes one zero-action step to fill the previous-observation half of the obs.
A step is two simulator steps with rewards summed.
"""

import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

ACTION_REPEAT = 2
HORIZON = 100


class MetaWorldWrapper:
    """MetaWorld wrapper with 39-dim state obs and 4-dim actions."""

    def __init__(self, env):
        self.env = env
        self.env.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False

    @staticmethod
    def _info(info, terminated=False, truncated=False):
        out = {
            "terminated": bool(info.get("terminated", terminated)),
            "truncated": bool(info.get("truncated", truncated)),
            "success": float(info.get("success", 0.0)),
        }
        out["score"] = out["success"]
        return out

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _r, _term, _trunc, info = self.env.step(
            np.zeros(self.env.action_space.shape, dtype=np.float32)
        )
        return obs.astype(np.float32), self._info(info)

    def step(self, action):
        reward = 0.0
        for _ in range(ACTION_REPEAT):
            obs, r, terminated, truncated, info = self.env.step(
                np.asarray(action).copy()
            )
            reward += r
            if terminated or truncated:
                break
        return (
            obs.astype(np.float32),
            reward,
            terminated,
            truncated,
            self._info(info, terminated, truncated),
        )

    def render(self, *_args, **_kwargs):
        return np.flipud(np.asarray(self.env.render()))

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        self.env.close()


def make_metaworld_env(task, seed=0):
    """Return (env, horizon) for a `mw-<name>` task slug."""
    env_id = task.split("-", 1)[-1] + "-v2-goal-observable"
    if env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise ValueError(f"unknown Metaworld task {task!r} (no env {env_id!r})")
    raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](
        seed=seed, render_mode="rgb_array"
    )
    return MetaWorldWrapper(raw), HORIZON
