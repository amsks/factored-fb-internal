"""Dataset utilities.

`Dataset` is ported from OGBench upstream impls/utils/datasets.py (the compact-dataset
FrozenDict with idx+1 next-observation inference). The samplers on top of it produce the
training batch each critic needs:

  FBDataset          -- observations/actions/next_observations, reward 0 for unlabeled play
                        data, terminals 0 (always bootstrap). fb.
  GCDataset          -- adds hindsight value/actor goals and the goal-reaching reward.
                        density_fb.
  WindowedDataset    -- the FB batch plus an h-step window of future actions/states. rldp.
  HILPFutureDataset  -- the FB batch plus one geometrically-sampled future state. hilp.
"""
import dataclasses
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
        # The row's identity in the dataset (HILPFutureDataset draws its future from it).
        result['index'] = np.asarray(idxs).reshape(-1)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class FBDataset(Dataset):
    """Dataset yielding the FB training batch. goal=next_obs is handled inside the critic,
    so no goal machinery is needed here."""

    # Set by make_dataset from obs_mask_dims|goal_mask_dims. FB's goal is next_observations,
    # so blanking both frames is the whole mask protocol for this dataset; eval blanks the
    # same dims (env/unified.py::evaluate).
    mask_dims = None

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        sub = self.get_subset(idxs)
        B = sub['observations'].shape[0]
        if 'rewards' in sub:
            rewards = np.asarray(sub['rewards'], np.float32).reshape(B, -1)
        else:
            rewards = np.zeros((B, 1), np.float32)  # play data is unlabeled
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
            'terminals': np.zeros((B,), np.float32),  # always bootstrap
            'index': np.asarray(sub['index']).reshape(-1),
        }


@dataclasses.dataclass
class GCDataset:
    """Goal-conditioned dataset, ported from OGBench utils/datasets.py:GCDataset.
    Produces value_goals, actor_goals, the goal-reaching reward and masks. The np.random
    call order matches the OGBench source for seeded determinism."""
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
        # Goal masking: zero the listed dims in the emitted goals only (e.g. the target
        # coordinates embedded in goal-observable MetaWorld/ManiSkill observations).
        mask_dims = c.get('goal_mask_dims')
        if mask_dims:
            batch['value_goals'][:, list(mask_dims)] = 0.0
            batch['actor_goals'][:, list(mask_dims)] = 0.0
        # Observation masking: on goal-observable envs the target is inside the agent's own
        # observation, so an unconditioned actor could read "go here" off its input. Blanking
        # the same dims in observations forces all goal information through the conditioning
        # vector. Eval blanks the same dims (env/unified.py::evaluate).
        obs_mask_dims = c.get('obs_mask_dims')
        if obs_mask_dims:
            batch['observations'][:, list(obs_mask_dims)] = 0.0
            batch['next_observations'][:, list(obs_mask_dims)] = 0.0
        return batch

    def _maybe_keep_dataset_rewards(self, batch, idxs):
        """Reward-specified domains (ExoRL): put the dataset's own task reward back into
        `batch['rewards']`, replacing the goal-reaching reward. The critic's reward-weighted
        task inference (`infer_cond(net, obs, rewards)`) reads these rewards, so without this
        it would infer the goal-reaching task instead of the one asked for.

        `masks` goes to 1: dm_control episodes end on a time limit, never on an absorbing
        state, so the TD target always bootstraps. Off by default (no-op, no randomness)."""
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
    """h-step windowed batches for RLDP: the FB batch plus, for every sampled index, the
    rest of the h-step self-predictive rollout's (action, next-state) columns.

    RLDP's self-predictive head rolls a learned predictor forward h steps from B(obs) and
    matches step k against the target backward map at s_{t+k}. After the critic prepends
    the one-step pair (a_t -> s_{t+1}), zipping actions with states gives the rollout
    (a_{t+k-1} -> s_{t+k}); this class emits the rest of it:

      * future_observations -- [B, h-1, obs_dim],    column k = row min(idx + 2 + k, traj_end)
      * future_actions      -- [B, h-1, action_dim], column k = row min(idx + 1 + k, traj_end)

    Column k of future_observations is the state reached by column k of future_actions.
    Windows are clamped to the anchor row's own trajectory end, so a window never reads
    into the next episode. h == 1 emits no future keys: the batch is exactly an FB batch.
    """
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        assert self.terminal_locs[-1] == self.size - 1
        # obs_mask_dims | goal_mask_dims, unioned as for FBDataset: the goal is the next state.
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
            rewards = np.zeros((B, 1), np.float32)  # play data is unlabeled
        batch = {
            'observations': obs,
            'actions': np.asarray(sub['actions'], np.float32),
            'next_observations': nobs,
            'rewards': rewards,
            'terminals': np.zeros((B,), np.float32),  # always bootstrap, as FBDataset
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
    """The FB batch plus one geometrically-sampled hindsight future state, for HILP.

    HILP's phi objective takes `hilp_future_observations` as the goal. Following the HILP
    reference buffer, the future is `step + Geometric(1 - hilp_future)` clipped to the
    anchor's own trajectory end (mean offset 100 steps at 0.99), then replaced by a uniformly
    random dataset state with probability `hilp_p_randomgoal`. The clip can make obs == goal,
    which is the row HILP's indicator reward is written for.
    """
    dataset: Any
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        self.p = 1.0 - float(self.config.get('hilp_future', 0.99))
        assert 0.0 < self.p <= 1.0, (
            f"hilp_future must be in [0, 1); got {self.config.get('hilp_future')}")
        self.p_randomgoal = float(self.config.get('hilp_p_randomgoal', 0.0))
        # obs_mask_dims | goal_mask_dims, unioned as for FBDataset.
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
    """Wrap a base Dataset per the critic's `dataset_class`. With none set (fb), the base
    FBDataset is used directly, with the obs/goal mask dims attached."""
    cls = _DATASET_CLASSES.get(config.get('dataset_class'))
    if cls is not None:
        return cls(base, config)
    if isinstance(base, FBDataset):
        dims = sorted(set(config.get('obs_mask_dims') or ()) |
                      set(config.get('goal_mask_dims') or ()))
        base.mask_dims = list(dims) or None
    return base
