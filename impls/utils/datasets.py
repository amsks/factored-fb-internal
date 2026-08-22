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

    # Set by make_dataset from obs_mask_dims|goal_mask_dims. FB's goal IS next_observations
    # (B() reads it inside stage_loss), so blanking both frames is the whole repaired
    # protocol for this dataset; without it the mask overrides are silently inert and FB
    # trains with the target visible in obs while eval blanks it (2026-08-17 FB-zero bug).
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
        # Optional OOD-goal split (Arm D): restrict every training goal to one side of a
        # hyperplane in observation space (e.g. cube block-x = dim 19, threshold 0).
        # Goals sampled on the held-out side are replaced by uniform draws from the
        # allowed pool, so eval goals on that side are never trained on.
        self.goal_ok = None
        if c.get('goal_filter_dim') is not None:
            col = np.asarray(self.dataset['observations'][:, int(c['goal_filter_dim'])])
            thresh = float(c.get('goal_filter_thresh', 0.0))
            side = c.get('goal_filter_side', 'below')
            self.goal_ok = col < thresh if side == 'below' else col >= thresh
            self.goal_pool = np.nonzero(self.goal_ok)[0]
            assert len(self.goal_pool) > 0, "goal_filter excludes every state"

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.get_subset(idxs)
        c = self.config
        value_goal_idxs = self.sample_goals(
            idxs, c['value_p_curgoal'], c['value_p_trajgoal'], c['value_p_randomgoal'], c['value_geom_sample'])
        actor_goal_idxs = self.sample_goals(
            idxs, c['actor_p_curgoal'], c['actor_p_trajgoal'], c['actor_p_randomgoal'], c['actor_geom_sample'])
        if self.goal_ok is not None:
            value_goal_idxs = self._filter_goals(value_goal_idxs)
            actor_goal_idxs = self._filter_goals(actor_goal_idxs)
        obs = self.dataset['observations']
        # SARSA bootstrap support (density_fb's `measure_density_bootstrap=sarsa`): the action
        # the BEHAVIOUR policy actually took at s'. Emitted unconditionally — it is one gather,
        # and every critic that ignores the key is bit-identical with it present.
        #
        # Trajectory-boundary-safe by clamping to the row's OWN final index rather than to
        # size-1: at a trajectory's last row, idx+1 belongs to the NEXT trajectory, and
        # bootstrapping a target on an action from an unrelated episode is exactly the kind of
        # silent cross-trajectory leak `valids` exists to prevent. Clamping repeats that
        # trajectory's last action instead, which is the standard convention and the only
        # locally-available choice (these datasets are fixed-horizon, so `terminals` never
        # fires as a true absorbing state and the target bootstraps regardless).
        final_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        batch['next_actions'] = np.asarray(
            self.dataset['actions'][np.minimum(idxs + 1, final_idxs)], np.float32)
        batch['observations'] = np.asarray(batch['observations'], np.float32)
        batch['next_observations'] = np.asarray(batch['next_observations'], np.float32)
        batch['actions'] = np.asarray(batch['actions'], np.float32)
        batch['value_goals'] = np.asarray(obs[value_goal_idxs], np.float32)
        batch['actor_goals'] = np.asarray(obs[actor_goal_idxs], np.float32)
        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if c['gc_negative'] else 0.0)
        # Optional goal-context masking (Arm C): zero the listed obs dims in the emitted
        # goals only (e.g. the embedded target coords on goal-observable envs). Applied
        # after the reward/mask computation, which is index-based and unaffected.
        mask_dims = c.get('goal_mask_dims')
        if mask_dims:
            batch['value_goals'][:, list(mask_dims)] = 0.0
            batch['actor_goals'][:, list(mask_dims)] = 0.0
        # Optional OBSERVATION masking. goal_mask_dims above only blanks the emitted goals,
        # which stops the goal channel of g being a nuisance feature for the critic but does
        # NOT close the behaviour-cloning shortcut: on goal-observable envs (Metaworld
        # obs[36:39], ManiSkill ms-push-cube obs[25:28], Box2D b2d-goal-ctrl obs[5:7]) the
        # target is inside the agent's own observation, so an unconditional BC actor reads
        # "go here" off its input and no critic is ever on the causal path. Blanking the same
        # dims in observations/next_observations forces every bit of goal information through
        # the conditioning vector, which is the only way the critic comparison measures a
        # critic. Eval must blank the same dims -- see env/unified.py::evaluate.
        obs_mask_dims = c.get('obs_mask_dims')
        if obs_mask_dims:
            batch['observations'][:, list(obs_mask_dims)] = 0.0
            batch['next_observations'][:, list(obs_mask_dims)] = 0.0
        return batch

    def _filter_goals(self, goal_idxs):
        bad = ~self.goal_ok[goal_idxs]
        if bad.any():
            goal_idxs = goal_idxs.copy()
            goal_idxs[bad] = self.goal_pool[np.random.randint(len(self.goal_pool), size=int(bad.sum()))]
        return goal_idxs

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
class HGCDataset(GCDataset):
    """Hierarchical goal-conditioned dataset — faithful port of OGBench
    utils/datasets.py:HGCDataset, state-only (frame-stack / image augmentation dropped),
    matching GCDataset's conventions above. Reads one extra config key, `subgoal_steps`.

    Adds the three keys HIQL needs and nothing else in the seam produces:
      * low_actor_goals    — the state `subgoal_steps` ahead, clamped to the trajectory's
        own final index so a subgoal never crosses into the next episode;
      * high_actor_goals   — a future-or-random goal for the high-level policy;
      * high_actor_targets — the subgoal the high policy is trained to predict for that
        goal, clamped to the goal itself so the target never overshoots it.

    `value_goals`/`rewards`/`masks` are produced exactly as GCDataset does. `actor_goals`
    is NOT emitted: vendored HGCDataset replaces it with the two high-level keys, and a
    flat critic paired with this dataset would silently read a goal that isn't there.
    np.random call order follows the vendored source so a seed reproduces its batches.
    """

    def sample(self, batch_size, idxs=None):
        c = self.config
        if c.get('subgoal_steps') is None:
            raise KeyError(
                "HGCDataset requires config key 'subgoal_steps' (vendored default 25); "
                "without it the low-level goal horizon is undefined")
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        batch = self.dataset.get_subset(idxs)
        obs = self.dataset['observations']

        value_goal_idxs = self.sample_goals(
            idxs, c['value_p_curgoal'], c['value_p_trajgoal'], c['value_p_randomgoal'],
            c['value_geom_sample'])
        if self.goal_ok is not None:
            value_goal_idxs = self._filter_goals(value_goal_idxs)

        final_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        batch['next_actions'] = np.asarray(
            self.dataset['actions'][np.minimum(idxs + 1, final_idxs)], np.float32)
        batch['observations'] = np.asarray(batch['observations'], np.float32)
        batch['next_observations'] = np.asarray(batch['next_observations'], np.float32)
        batch['actions'] = np.asarray(batch['actions'], np.float32)
        batch['value_goals'] = np.asarray(obs[value_goal_idxs], np.float32)
        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if c['gc_negative'] else 0.0)

        # low-level goals: subgoal_steps ahead, clamped to this trajectory's end
        sg = int(c['subgoal_steps'])
        low_goal_idxs = np.minimum(idxs + sg, final_idxs)
        batch['low_actor_goals'] = np.asarray(obs[low_goal_idxs], np.float32)

        # high-level goals + the subgoal targets the high policy predicts for them
        if c['actor_geom_sample']:
            offsets = np.random.geometric(p=1 - c['discount'], size=len(idxs))   # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_idxs)
        else:
            distances = np.random.rand(len(idxs))                                # in [0, 1)
            high_traj_goal_idxs = np.round(
                np.minimum(idxs + 1, final_idxs) * distances + final_idxs * (1 - distances)
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + sg, high_traj_goal_idxs)

        high_random_goal_idxs = self.dataset.get_random_idxs(len(idxs))
        high_random_target_idxs = np.minimum(idxs + sg, final_idxs)

        pick_random = np.random.rand(len(idxs)) < c['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
        batch['high_actor_goals'] = np.asarray(obs[high_goal_idxs], np.float32)
        batch['high_actor_targets'] = np.asarray(obs[high_target_idxs], np.float32)

        mask_dims = c.get('goal_mask_dims')
        if mask_dims:
            for k in ('value_goals', 'low_actor_goals', 'high_actor_goals', 'high_actor_targets'):
                batch[k][:, list(mask_dims)] = 0.0
        obs_mask_dims = c.get('obs_mask_dims')
        if obs_mask_dims:
            batch['observations'][:, list(obs_mask_dims)] = 0.0
            batch['next_observations'][:, list(obs_mask_dims)] = 0.0
        return batch


@dataclasses.dataclass
class WindowedDataset:
    """h-step WINDOWED batches: the FB training batch plus, for every sampled index, the
    next h-1 (observation, action) pairs.

    Added for RLDP (impls/critics/rldp.py), whose self-predictive head rolls a learned
    predictor forward h steps from B(obs) and matches each step against the target
    backward map at the corresponding future state. The torch reference gets these from
    `DictBuffer.sample(batch_size, horizon=h)` and slices `batch["observation"][:, 1:]` /
    `batch["action"][:, 1:]` (agents/rldp/agent.py::update); this is the seam's equivalent.

    Emits, on top of everything `FBDataset.sample` emits:
      * future_observations -- [B, h-1, obs_dim]
      * future_actions      -- [B, h-1, action_dim]
    with column k (0-based) being dataset row `min(idx + 1 + k, traj_final_idx)`.
    Column 0 is therefore s_{t+1}, the same row `next_observations` carries.

    TRAJECTORY BOUNDARIES. The window is clamped to the anchor row's OWN final index,
    exactly as `HGCDataset.sample` clamps `low_actor_goals`
    (`np.minimum(idxs + subgoal_steps, final_state_idxs)`) and as `GCDataset.sample`
    clamps `next_actions`. Clamping to `size - 1` instead -- the shape-correct, silently
    wrong version -- would let a window that starts near the end of an episode read the
    opening rows of the NEXT episode, and the predictor would be trained to roll a
    dynamics model into an unrelated trajectory with nothing failing loudly. Clamping
    repeats the trajectory's own last row instead, which is the standard convention and
    the only locally-available choice.

    h == 1 is the degenerate no-window case (the reference's `if self.horizon > 1` guard):
    no future keys are emitted at all, and the batch is exactly an FB batch.
    """
    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1
        # obs_mask_dims|goal_mask_dims are unioned exactly as make_dataset does for
        # FBDataset: FB's goal IS next_observations, so both knobs blank the same frames.
        dims = sorted(set(self.config.get('obs_mask_dims') or ()) |
                      set(self.config.get('goal_mask_dims') or ()))
        self.mask_dims = list(dims) or None

    @property
    def horizon(self):
        h = self.config.get('horizon')
        if h is None:
            raise KeyError(
                "WindowedDataset requires config key 'horizon' (the number of steps in a "
                "window; RLDP's default is 5); without it the self-predictive rollout "
                "length is undefined and would silently degrade to a one-step loss")
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
            'terminals': np.zeros((B,), np.float32),  # always-gamma, as FBDataset
        }

        if h > 1:
            all_obs, all_acts = self.dataset['observations'], self.dataset['actions']
            final_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
            offsets = np.arange(1, h)                                   # [h-1]
            win = np.minimum(idxs[:, None] + offsets[None, :],          # [B, h-1]
                             final_idxs[:, None])
            batch['future_observations'] = np.asarray(all_obs[win], np.float32)
            batch['future_actions'] = np.asarray(all_acts[win], np.float32)

        if self.mask_dims:
            batch['observations'][:, self.mask_dims] = 0.0
            batch['next_observations'][:, self.mask_dims] = 0.0
            if h > 1:
                batch['future_observations'][:, :, self.mask_dims] = 0.0
        return batch


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
    if config.get('dataset_class') == 'HGCDataset':
        return HGCDataset(base, config)
    if config.get('dataset_class') == 'VIPDataset':
        return VIPDataset(base, config)
    if config.get('dataset_class') == 'WindowedDataset':
        return WindowedDataset(base, config)
    if isinstance(base, FBDataset):
        dims = sorted(set(config.get('obs_mask_dims') or ()) |
                      set(config.get('goal_mask_dims') or ()))
        base.mask_dims = list(dims) or None
    return base
