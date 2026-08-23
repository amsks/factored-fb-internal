"""Environment + dataset construction, ported from third_party/ogbench/impls/utils/env_utils.py.
Uses the compact dataset and wraps it in FBDataset for the FB training batch."""
import collections
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from impls.utils.datasets import FBDataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}
        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, frame_stack=None, add_info=False):
    """Make OGBench environment and FB datasets (compact, unlabeled).

    `add_info` asks OGBench's loader for the raw simulator state columns ('qpos', 'qvel')
    that ship in every npz but are dropped by default. Pixel runs of density_fb NEED
    'qpos': the measure's hit kernel is a column slice of object coordinates, which a
    (64, 64, 3) image has no analogue of (see docs/specs/2026-08-21-density-fb-pixels-drq.md
    section 1). It stays OFF by default so a state run's batch keeps exactly the keys it
    always had -- `Dataset.get_subset` gathers every field, so loading qpos on a state run
    would silently add two unused columns to every batch.
    """
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, compact_dataset=True, add_info=add_info)
    train_dataset = FBDataset.create(**train_dataset)
    val_dataset = FBDataset.create(**val_dataset)
    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
    env.reset()
    return env, train_dataset, val_dataset
