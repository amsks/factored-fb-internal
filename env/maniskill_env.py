"""ManiSkill3 state-observation envs, built as the offline data was collected."""

import os

import numpy as np

# task slug -> (env id, control mode, horizon); control mode sets the action dim of the data
TASKS = {
    "ms-pick-cube": ("PickCube-v1", "pd_ee_delta_pos", 25),
    "ms-push-cube": ("PushCube-v1", "pd_ee_delta_pose", 25),
    "ms-stack-cube": ("StackCube-v1", "pd_ee_delta_pose", 25),
    "ms-lift-peg": ("LiftPegUpright-v1", "pd_ee_delta_pose", 25),
    "ms-poke-cube": ("PokeCube-v1", "pd_ee_delta_pose", 25),
}


class ManiSkillWrapper:
    """Single-env numpy wrapper over CPUGymWrapper."""

    def __init__(self, env, horizon):
        self.env = env
        self._horizon = horizon

    def reset(self, seed=None, **kw):
        obs, info = (
            self.env.reset(seed=seed, **kw)
            if seed is not None
            else self.env.reset(**kw)
        )
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


def make_maniskill_env(task, render_size=128, max_episode_steps=None):
    """Return (env, horizon) for an `ms-<name>` task slug.

    :param max_episode_steps: Overrides the env's 50-step time limit; leave None for eval.
    """
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
    limit = (
        {}
        if max_episode_steps is None
        else dict(max_episode_steps=int(max_episode_steps))
    )
    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode=control_mode,
        num_envs=1,
        render_mode="rgb_array",
        sensor_configs=dict(width=render_size, height=render_size),
        human_render_camera_configs=dict(width=render_size, height=render_size),
        reconfiguration_freq=None,
        sim_backend="auto",
        **limit,
    )
    env = CPUGymWrapper(env, ignore_terminations=True)
    return ManiSkillWrapper(env, horizon), horizon
