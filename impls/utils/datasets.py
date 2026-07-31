"""Dataset utilities for the impls/ tree.

`Dataset` is ported from third_party/ogbench/impls/utils/datasets.py (the compact-dataset
FrozenDict with idx+1 next-observation inference). `FBDataset` adds the FB training batch:
observations/actions/next_observations plus reward=0 (play data is unlabeled) and
terminals=0 (always-gamma bootstrap, matching the PyTorch reference).
"""
import dataclasses
from functools import partial
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Compact dataset: stores observations (and optionally next_observations). If
    next_observations is absent it is inferred by shifting indices by 1; `valids` masks
    trajectory-boundary transitions."""

    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class FBDataset(Dataset):
    """Dataset yielding the FB training batch. goal=next_obs is handled inside the critic,
    so no goal machinery is needed here."""

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        sub = self.get_subset(idxs)
        B = sub['observations'].shape[0]
        if 'rewards' in sub:
            rewards = np.asarray(sub['rewards'], np.float32).reshape(B, -1)
        else:
            rewards = np.zeros((B, 1), np.float32)  # play data is unlabeled
        return {
            'observations': np.asarray(sub['observations'], np.float32),
            'actions': np.asarray(sub['actions'], np.float32),
            'next_observations': np.asarray(sub['next_observations'], np.float32),
            'rewards': rewards,
            'terminals': np.zeros((B,), np.float32),  # always-gamma
        }


@dataclasses.dataclass
class GCDataset:
    """Goal-conditioned dataset — faithful port of OGBench utils/datasets.py:GCDataset,
    state-only (frame-stack / image augmentation dropped). Produces value_goals, actor_goals,
    rewards, masks. np.random call order matches the vendored source for seeded determinism.
    """
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1
        c = self.config
        assert np.isclose(c['value_p_curgoal'] + c['value_p_trajgoal'] + c['value_p_randomgoal'], 1.0)
        assert np.isclose(c['actor_p_curgoal'] + c['actor_p_trajgoal'] + c['actor_p_randomgoal'], 1.0)

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.get_subset(idxs)
        c = self.config
        value_goal_idxs = self.sample_goals(
            idxs, c['value_p_curgoal'], c['value_p_trajgoal'], c['value_p_randomgoal'], c['value_geom_sample'])
        actor_goal_idxs = self.sample_goals(
            idxs, c['actor_p_curgoal'], c['actor_p_trajgoal'], c['actor_p_randomgoal'], c['actor_geom_sample'])
        obs = self.dataset['observations']
        batch['observations'] = np.asarray(batch['observations'], np.float32)
        batch['next_observations'] = np.asarray(batch['next_observations'], np.float32)
        batch['actions'] = np.asarray(batch['actions'], np.float32)
        batch['value_goals'] = np.asarray(obs[value_goal_idxs], np.float32)
        batch['actor_goals'] = np.asarray(obs[actor_goal_idxs], np.float32)
        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if c['gc_negative'] else 0.0)
        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        batch_size = len(idxs)
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs)
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)
        return goal_idxs


@dataclasses.dataclass
class VIPDataset:
    """VIP sub-trajectory batches (Ma et al., Algorithm 1), state-based.

    Anchor k is a random valid index; o0 ~ U[traj_start, k]; the goal is the trajectory's
    FINAL frame; (o_k, o_{k+1}) is the TD transition. reward = -1 unless k+1 is the final
    frame (VIP's sparse goal reward). actor_goals = value_goals so BC actors condition on
    the same goal.
    """
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.get_subset(idxs)
        traj = np.searchsorted(self.terminal_locs, idxs)
        term, init = self.terminal_locs[traj], self.initial_locs[traj]
        t0 = init + np.floor(np.random.rand(len(idxs)) * (idxs - init + 1)).astype(np.int64)
        obs = self.dataset['observations']
        batch['observations'] = np.asarray(batch['observations'], np.float32)
        batch['next_observations'] = np.asarray(batch['next_observations'], np.float32)
        batch['actions'] = np.asarray(batch['actions'], np.float32)
        batch['initial_observations'] = np.asarray(obs[t0], np.float32)
        batch['value_goals'] = np.asarray(obs[term], np.float32)
        batch['actor_goals'] = batch['value_goals']
        reached = (idxs + 1 >= term).astype(np.float32)
        batch['rewards'] = reached - 1.0
        batch['masks'] = 1.0 - reached
        return batch


def make_dataset(base, config):
    """Wrap a base Dataset per the critic's dataset_class.

    Goal-conditioned critics (gciql and the PR4+ family) need GCDataset's trajectory goal
    sampling; FB carries no goal machinery and uses the base dataset unchanged.
    """
    if config.get('dataset_class') == 'GCDataset':
        return GCDataset(base, config)
    if config.get('dataset_class') == 'VIPDataset':
        return VIPDataset(base, config)
    return base
