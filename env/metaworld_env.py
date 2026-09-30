"""env/metaworld_env.py — Metaworld's goal-observable envs behind the same surface the
collected offline data was recorded through.

The demonstrations in the unified Metaworld datasets were produced through newt's
`envs.metaworld.MetaWorldWrapper`, so the eval env has to reproduce that wrapper *exactly*
or the policy sees a different observation/timescale than it was trained on. The three
things that matter for parity:

    * `_freeze_rand_vec = False`  -- every reset redraws the task's random vector, so the
      eval distribution matches how the data was collected.
    * reset = env.reset() followed by ONE zero-action step, whose obs is what is returned.
      Metaworld's raw reset obs leaves the `prev_obs` half of the 39-dim vector stale; the
      zero step fills it. The dataset's first transition is that post-zero-step obs.
    * step = action repeat 2 with the rewards summed. A "step" in the dataset is two
      simulator steps, so the horizon of 100 is 200 raw MuJoCo steps.

We re-implement it here rather than importing newt because newt's module pulls in torch +
torchvision at import time (only for its `render`), and this repo is JAX-only.
"""
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

ACTION_REPEAT = 2
HORIZON = 100


class MetaWorldWrapper:
    """newt's MetaWorldWrapper, torch-free. Obs is the raw 39-dim state, actions are 4-dim."""

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
            np.zeros(self.env.action_space.shape, dtype=np.float32))
        return obs.astype(np.float32), self._info(info)

    def step(self, action):
        reward = 0.0
        for _ in range(ACTION_REPEAT):
            obs, r, terminated, truncated, info = self.env.step(np.asarray(action).copy())
            reward += r
            if terminated or truncated:
                break
        return (obs.astype(np.float32), reward, terminated, truncated,
                self._info(info, terminated, truncated))

    def render(self, *_args, **_kwargs):
        # The vertical flip makes the frame right-side up.
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
    """-> (env, horizon) for a `mw-<name>` task slug, e.g. `mw-bin-picking`.

    `render_mode='rgb_array'` unconditionally, as in the collection env — it only enables
    `render()`, it does not render during `step`."""
    env_id = task.split("-", 1)[-1] + "-v2-goal-observable"
    if env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise ValueError(f"unknown Metaworld task {task!r} (no env {env_id!r})")
    raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed, render_mode="rgb_array")
    return MetaWorldWrapper(raw), HORIZON
