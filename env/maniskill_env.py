"""env/maniskill_env.py — ManiSkill3 envs behind the same surface the collected offline data
was recorded through, without newt's `envs.maniskill` backend (which needs newt's `/raid`
checkout + its multitask vectorized wrapper).

The offline data was recorded through newt's wrapper: `gym.make(<id>, obs_mode='state',
control_mode=<per-task>, num_envs=1)` then `CPUGymWrapper`. Reproducing exactly that here is
what makes the eval env's 42/35/48-dim state match the dataset — verified per task in the
parity probe (env obs/act == dataset obs/act on all of pick/push/stack/lift/poke).

CPUGymWrapper(ignore_terminations=True) runs the full fixed horizon and returns single-env
numpy, so the unified eval loop consumes it exactly like the Metaworld/Box2D envs.
"""
import os

import numpy as np

# task slug -> (env id, control mode, horizon). Control modes are newt's, and are load-bearing:
# they set the action dim, which must match the recorded data (probe-verified).
TASKS = {
    "ms-pick-cube":  ("PickCube-v1",       "pd_ee_delta_pos",  25),
    "ms-push-cube":  ("PushCube-v1",       "pd_ee_delta_pose", 25),
    "ms-stack-cube": ("StackCube-v1",      "pd_ee_delta_pose", 25),
    "ms-lift-peg":   ("LiftPegUpright-v1", "pd_ee_delta_pose", 25),
    "ms-poke-cube":  ("PokeCube-v1",       "pd_ee_delta_pose", 25),
}


class ManiSkillWrapper:
    """Thin single-env numpy adapter over CPUGymWrapper — reset/step/render/close and a
    numpy float32 obs, matching what the unified eval loop expects."""

    def __init__(self, env, horizon):
        self.env = env
        self._horizon = horizon

    def reset(self, seed=None, **kw):
        obs, info = self.env.reset(seed=seed, **kw) if seed is not None else self.env.reset(**kw)
        return np.asarray(obs, np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(np.asarray(action))
        return np.asarray(obs, np.float32), float(r), bool(term), bool(trunc), info

    def render(self, *_a, **_k):
        return np.asarray(self.env.render())

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
        try:
            self.env.close()
        except Exception:
            pass


def make_maniskill_env(task, render_size=128):
    """-> (env, horizon). SAPIEN needs its Vulkan ICD pointed at explicitly on Vista (its
    auto-probe misses /etc/vulkan/icd.d/nvidia_icd.aarch64.json)."""
    if task not in TASKS:
        raise ValueError(f"unknown ManiSkill task {task!r} (known: {sorted(TASKS)})")
    _icd = "/etc/vulkan/icd.d/nvidia_icd.aarch64.json"
    if os.path.exists(_icd):
        os.environ.setdefault("VK_ICD_FILENAMES", _icd)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the envs)
    from mani_skill.utils.wrappers import CPUGymWrapper

    env_id, control_mode, horizon = TASKS[task]
    env = gym.make(env_id, obs_mode="state", control_mode=control_mode, num_envs=1,
                   render_mode="rgb_array",
                   sensor_configs=dict(width=render_size, height=render_size),
                   human_render_camera_configs=dict(width=render_size, height=render_size),
                   reconfiguration_freq=None, sim_backend="auto")
    env = CPUGymWrapper(env, ignore_terminations=True)   # full fixed horizon, single-env numpy
    return ManiSkillWrapper(env, horizon), horizon
