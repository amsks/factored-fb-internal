"""Datasets and the batch samplers each critic trains on."""
import dataclasses
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict

from impls.utils.config_utils import upgrade_config


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Compact dataset; next_observations default to the next row."""

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
        result['index'] = np.asarray(idxs).reshape(-1)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class FBDataset(Dataset):
    """Transition batches for FB, with zero reward on unlabeled data."""

    # Observation dims zeroed in both frames; set by make_dataset.
    mask_dims = None

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        sub = self.get_subset(idxs)
        B = sub['observations'].shape[0]
        if 'rewards' in sub:
            rewards = np.asarray(sub['rewards'], np.float32).reshape(B, -1)
        else:
            rewards = np.zeros((B, 1), np.float32)
        obs = np.asarray(sub['observations'], np.float32)
        nobs = np.asarray(sub['next_observations'], np.float32)
        if self.mask_dims:
            obs[:, self.mask_dims] = 0.0
            nobs[:, self.mask_dims] = 0.0
        return {
            'observations': obs,
            'actions': np.asarray(sub['actions'], np.float32),
            'next_observations': nobs,
            'rewards': rewards,
            'terminals': np.zeros((B,), np.float32),
            'index': np.asarray(sub['index']).reshape(-1),
        }


@dataclasses.dataclass
class GCDataset:
    """Goal-conditioned dataset with hindsight value/actor goals and goal-reaching reward."""
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
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
        batch['observations'] = np.asarray(batch['observations'], np.float32)
        batch['next_observations'] = np.asarray(batch['next_observations'], np.float32)
        batch['actions'] = np.asarray(batch['actions'], np.float32)
        batch['value_goals'] = np.asarray(self.dataset['observations'][value_goal_idxs], np.float32)
        batch['actor_goals'] = np.asarray(self.dataset['observations'][actor_goal_idxs], np.float32)
        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if c['gc_negative'] else 0.0)
        self._maybe_keep_dataset_rewards(batch, idxs)
        # Goal masking
        mask_dims = c.get('goal_mask_dims')
        if mask_dims:
            batch['value_goals'][:, list(mask_dims)] = 0.0
            batch['actor_goals'][:, list(mask_dims)] = 0.0
        # Observation masking, so goal information enters only through the conditioning
        obs_mask_dims = c.get('obs_mask_dims')
        if obs_mask_dims:
            batch['observations'][:, list(obs_mask_dims)] = 0.0
            batch['next_observations'][:, list(obs_mask_dims)] = 0.0
        return batch

    def _maybe_keep_dataset_rewards(self, batch, idxs):
        """If keep_dataset_rewards, use the dataset's task reward and always bootstrap."""
        if not self.config.get('keep_dataset_rewards'):
            return
        if 'rewards' not in self.dataset:
            raise KeyError(
                "keep_dataset_rewards=true but the base dataset carries no 'rewards' field "
                "(an unlabeled play dataset cannot supply a task reward).")
        n = len(idxs)
        batch['rewards'] = np.asarray(self.dataset['rewards'][idxs], np.float32).reshape(n)
        batch['masks'] = np.ones(n, np.float32)

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
class WindowedDataset:
    """FB batch plus an h-step window of future actions and states, for RLDP.

    future_actions [B, h-1, act_dim] and future_observations [B, h-1, obs_dim]; column k of
    the states is reached by column k of the actions. Windows stop at the trajectory end.
    """
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        assert self.terminal_locs[-1] == self.size - 1
        dims = sorted(set(self.config.get('obs_mask_dims') or ()) |
                      set(self.config.get('goal_mask_dims') or ()))
        self.mask_dims = list(dims) or None

    @property
    def horizon(self):
        h = self.config.get('horizon')
        if h is None:
            raise KeyError(
                "WindowedDataset requires config key 'horizon' (the number of steps in a "
                "window; RLDP's default is 5)")
        h = int(h)
        assert h >= 1, f"horizon must be >= 1, got {h}"
        return h

    def sample(self, batch_size, idxs=None):
        h = self.horizon
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        idxs = np.asarray(idxs)
        sub = self.dataset.get_subset(idxs)
        B = idxs.shape[0]

        obs = np.asarray(sub['observations'], np.float32)
        nobs = np.asarray(sub['next_observations'], np.float32)
        if 'rewards' in sub:
            rewards = np.asarray(sub['rewards'], np.float32).reshape(B, -1)
        else:
            rewards = np.zeros((B, 1), np.float32)
        batch = {
            'observations': obs,
            'actions': np.asarray(sub['actions'], np.float32),
            'next_observations': nobs,
            'rewards': rewards,
            'terminals': np.zeros((B,), np.float32),
        }

        if h > 1:
            all_obs, all_acts = self.dataset['observations'], self.dataset['actions']
            final_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
            act_win = np.minimum(idxs[:, None] + np.arange(1, h)[None, :],      # [B, h-1]
                                 final_idxs[:, None])
            obs_win = np.minimum(idxs[:, None] + np.arange(2, h + 1)[None, :],  # [B, h-1]
                                 final_idxs[:, None])
            batch['future_observations'] = np.asarray(all_obs[obs_win], np.float32)
            batch['future_actions'] = np.asarray(all_acts[act_win], np.float32)

        if self.mask_dims:
            batch['observations'][:, self.mask_dims] = 0.0
            batch['next_observations'][:, self.mask_dims] = 0.0
            if h > 1:
                batch['future_observations'][:, :, self.mask_dims] = 0.0
        return batch


@dataclasses.dataclass
class HILPFutureDataset:
    """FB batch plus one future state per row, for HILP.

    The future is a geometric offset clipped to the trajectory end, replaced by a random
    state with probability p_randomgoal.
    """
    dataset: Any
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        self.p = 1.0 - float(self.config.get('goal_future', 0.99))
        assert 0.0 < self.p <= 1.0, (
            f"goal_future must be in [0, 1); got {self.config.get('goal_future')}")
        self.p_randomgoal = float(self.config.get('p_randomgoal', 0.0))
        dims = sorted(set(self.config.get('obs_mask_dims') or ()) |
                      set(self.config.get('goal_mask_dims') or ()))
        self.mask_dims = list(dims) or None

    def sample(self, batch_size, idxs=None):
        batch = self.dataset.sample(batch_size, idxs)
        idxs = np.asarray(batch['index']).reshape(-1)
        final_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        offs = np.random.geometric(p=self.p, size=idxs.shape[0])
        fut = np.minimum(idxs + offs, final_idxs)
        goals = np.asarray(self.dataset['observations'][fut], np.float32)
        n = goals.shape[0]
        if self.p_randomgoal > 0.0:
            rnd = np.asarray(
                self.dataset['observations'][np.random.randint(0, self.size, n)], np.float32)
            goals = np.where((np.random.rand(n) < self.p_randomgoal)[:, None], rnd, goals)
        batch['hilp_future_observations'] = goals

        if self.mask_dims:
            for k in ('observations', 'next_observations', 'hilp_future_observations'):
                batch[k] = np.array(batch[k], np.float32, copy=True)
                batch[k][:, self.mask_dims] = 0.0
        return batch


_DATASET_CLASSES = {'GCDataset': GCDataset, 'WindowedDataset': WindowedDataset,
                    'HILPFutureDataset': HILPFutureDataset}


def make_dataset(base, config):
    """Wrap a base dataset in the critic's `dataset_class`, or return it with mask dims set."""
    config = upgrade_config(config)
    cls = _DATASET_CLASSES.get(config.get('dataset_class'))
    if cls is not None:
        return cls(base, config)
    if isinstance(base, FBDataset):
        dims = sorted(set(config.get('obs_mask_dims') or ()) |
                      set(config.get('goal_mask_dims') or ()))
        base.mask_dims = list(dims) or None
    return base
