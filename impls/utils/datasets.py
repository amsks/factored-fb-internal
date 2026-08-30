"""Dataset utilities for the impls/ tree.

`Dataset` is ported from OGBench upstream impls/utils/datasets.py (the compact-dataset
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

from impls.utils.encoders import random_shifts_batch


def parse_qpos_slices(spec):
    """Parse `measure_qpos_slices` -- "a:b" or "a:b,c:d" -- into a tuple of (a, b) pairs.

    This is the PRIVILEGED MEASURE CHANNEL for pixel runs: which columns of the dataset's
    `qpos` field make up the object-coordinate row `impls/critics/density_fb.py`'s hit kernel
    slices. On `visual-cube-single-play-v0` that is `"14:17"` -- MuJoCo's
    `object_joint_0` free joint starts at qpos index 14, so qpos[14:17] IS
    `info['privileged/block_0_pos']` (verified, and pinned by
    tests/impls/test_density_fb_pixels.py).

    None (the default, and every state config) -> no channel, and `GCDataset` emits nothing
    extra. The state path never reads this key.
    """
    if spec is None:
        return None
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        a, b = int(a), int(b)
        assert b > a, f"measure_qpos_slices block {part!r} is empty or reversed"
        out.append((a, b))
    assert out, f"measure_qpos_slices={spec!r} parsed to no blocks"
    return tuple(out)


def _qpos_rows(qpos, idxs, slices):
    """[N, sum(b-a)] object-coordinate rows gathered at `idxs` -- the measure channel."""
    return np.concatenate([np.asarray(qpos[idxs][:, a:b], np.float32) for (a, b) in slices],
                          axis=-1)


def _batch_array(arr):
    """float32 for state rows (what this seam has always emitted); the RAW dtype for image
    tensors. `DrQEncoder` normalizes uint8 internally (`x/255 - 0.5`), so a float32 copy
    would quadruple the host->device transfer for a bit-identical result -- and would also
    make a 1024x64x64x3 batch 50MB instead of 12MB."""
    a = np.asarray(arr)
    return a if a.ndim > 2 else np.asarray(a, np.float32)


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
        # The row's identity in the dataset, not its position in the batch. PSM's proto
        # behaviour sampler keys its policy assignment on this: without it the same
        # transition draws a different proto policy every resample, so psm_psi has no fixed
        # policy to converge to (reference: batch["index"], enabled via with_index=True for
        # psm only). Cheap, and every other critic ignores it.
        result['index'] = np.asarray(idxs).reshape(-1)
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
            'index': np.asarray(sub['index']).reshape(-1),
        }


@dataclasses.dataclass
class GCDataset:
    """Goal-conditioned dataset — faithful port of OGBench utils/datasets.py:GCDataset.
    Produces value_goals, actor_goals, rewards, masks. np.random call order matches the
    vendored source for seeded determinism.

    Three PIXEL features, each gated on a config key that no state config sets, so the
    state batch is byte-for-byte what it always was:

    * ``frame_stack`` (int|None) — stack the last k frames on the channel axis, clamped to
      the trajectory's own first row so a stack never crosses an episode boundary. Computed
      ON THE FLY, unlike vendored OGBench's `preprocess_frame_stack=True` default: the
      preprocessed form of `visual-cube-single-play-v0` at k=3 is 1,001,000x64x64x9 uint8
      = 36 GB of RAM.
    * ``p_aug`` (float|None) — probability of applying `random_shifts_batch` (replicate-pad
      by `aug_pad`, default 2, then integer-crop) to the image keys. HOST-SIDE NUMPY on
      purpose; see impls/utils/encoders.py::random_shifts_batch.
    * ``measure_qpos_slices`` (str|None) — emit the PRIVILEGED object-coordinate channel
      alongside every image key: `observations_qpos`, `next_observations_qpos`,
      `value_goals_qpos`, `actor_goals_qpos`. density_fb's hit kernel and query support are
      literal column slices, which an image has no analogue of; this is the channel they
      slice. Sourced from the dataset's own `qpos` field, gathered at exactly the same row
      indices as the image it accompanies -- that index-sharing is what makes the two
      channels describe the same state.

    FRAME STACK x QPOS CHANNEL (spec section 9): the image key carries k stacked frames, the
    qpos channel carries ONE row -- the LATEST frame (index `idxs`, the same row the
    unstacked observation would be). The measure therefore always refers to the current
    state, never to a frame in the stack's history.
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

        # --- pixel plumbing. All three are None on every state config, and each `if` below
        # is a plain python branch on a None, so the state batch is unchanged. ---
        self.frame_stack = c.get('frame_stack')
        if self.frame_stack is not None:
            self.frame_stack = int(self.frame_stack)
            assert self.frame_stack >= 1, f"frame_stack must be >= 1, got {self.frame_stack}"
            assert 'next_observations' not in self.dataset, (
                "frame_stack needs the COMPACT layout (observations only): a stack is built "
                "by walking backwards from a row's index, which a stored next_observations "
                "column cannot be reindexed against")
        self.qpos_slices = parse_qpos_slices(c.get('measure_qpos_slices'))
        if self.qpos_slices is not None and 'qpos' not in self.dataset:
            raise KeyError(
                "measure_qpos_slices is set but the dataset has no 'qpos' field. That field "
                "is the ONLY source of the privileged measure channel on pixels; without it "
                "density_fb's hit kernel would fall back to slicing the image tensor and "
                "return plausible-looking numbers computed from pixel values.")
        self.aug_pad = int(c.get('aug_pad', 2) or 2)

    def get_observations(self, idxs):
        """Observations at `idxs`, frame-stacked when `frame_stack` is set."""
        if self.frame_stack is None:
            return self.dataset['observations'][idxs]
        return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """The last `frame_stack` frames at `idxs`, concatenated on the CHANNEL axis.

        Each earlier frame is clamped to the row's own trajectory start (`initial_locs`), so
        the opening frames of an episode repeat rather than reading the tail of the previous
        one. Vendored OGBench does exactly this (`get_stacked_observations`); the only
        deviation is that we never preprocess the whole dataset (see the class docstring)."""
        obs = self.dataset['observations']
        initial = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = [obs[np.maximum(idxs - i, initial)] for i in reversed(range(self.frame_stack))]
        return np.concatenate(rets, axis=-1)

    def augment(self, batch, keys):
        """FB-faithful random shifts on the 4-D keys of `batch`, in place. pad defaults to 2
        (FB's `Augmentator`), not vendored OGBench's 3."""
        random_shifts_batch(batch, keys, pad=self.aug_pad)

    def _add_qpos_channel(self, batch, idxs, next_idxs, value_goal_idxs, actor_goal_idxs):
        """The privileged object-coordinate twin of each image key, gathered at the SAME row
        indices. Emitted under `<key>_qpos` names so nothing that reads the image keys can
        pick one up by accident."""
        qp, sl = self.dataset['qpos'], self.qpos_slices
        batch['observations_qpos'] = _qpos_rows(qp, idxs, sl)
        batch['next_observations_qpos'] = _qpos_rows(qp, next_idxs, sl)
        batch['value_goals_qpos'] = _qpos_rows(qp, value_goal_idxs, sl)
        batch['actor_goals_qpos'] = _qpos_rows(qp, actor_goal_idxs, sl)

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
        # The row `next_observations` refers to, spelled once: Dataset.get_subset infers it
        # as observations[min(idx+1, size-1)], and the frame-stack and qpos channels below
        # MUST index the same row or the three channels describe different states.
        next_idxs = np.minimum(idxs + 1, self.dataset.size - 1)
        if self.frame_stack is not None:
            batch['observations'] = self.get_stacked_observations(idxs)
            batch['next_observations'] = self.get_stacked_observations(next_idxs)
        batch['observations'] = _batch_array(batch['observations'])
        batch['next_observations'] = _batch_array(batch['next_observations'])
        batch['actions'] = np.asarray(batch['actions'], np.float32)
        batch['value_goals'] = _batch_array(self.get_observations(value_goal_idxs))
        batch['actor_goals'] = _batch_array(self.get_observations(actor_goal_idxs))
        if self.qpos_slices is not None:
            self._add_qpos_channel(batch, idxs, next_idxs, value_goal_idxs, actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if c['gc_negative'] else 0.0)
        self._maybe_keep_dataset_rewards(batch, idxs)
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
        # Image augmentation, LAST -- after the goals exist and after any masking, so every
        # image key the agent will actually see is shifted, and none of the index-derived
        # scalars (rewards/masks) can be affected by it. One coin per batch (vendored
        # OGBench's convention), then INDEPENDENT shifts per key inside `augment`. The
        # `_qpos` channels are 2-D and pass through untouched -- the measure must not be
        # perturbed by a nuisance transformation the encoder is meant to be invariant to.
        if c.get('p_aug') is not None and np.random.rand() < float(c['p_aug']):
            self.augment(batch, ['observations', 'next_observations',
                                 'value_goals', 'actor_goals'])
        return batch

    def _maybe_keep_dataset_rewards(self, batch, idxs):
        """REWARD-SPECIFIED domains (ExoRL / dm_control -- see env/exorl_env.py): put the
        dataset's OWN task reward back into `batch['rewards']`, replacing the goal-reaching
        reward synthesised just above.

        Why this needs to exist. The synthesised column is the right reward for a
        goal-reaching task and the WRONG one for a task defined by a reward function, and
        overwriting is silent: `impls/main.py::infer_cond_noop_reason` only flags an
        ALL-ZERO reward vector, and the goal-reaching column is not all zero. Everything the
        successor-measure family claims zero-shot runs through
        `Critic.infer_cond(net, obs, rewards)`, which `impls/main.py` feeds straight from
        `train_ds.sample(...)['rewards']` -- the "WHICH REWARDS" paragraph in
        `impls/critics/density_fb.py::infer_cond` names exactly this failure mode. Without
        this knob, `density_fb` / `td_fb` (the two GCDataset critics in that family) cannot
        be evaluated on a reward-specified domain at all, and would report a number for a
        task nobody asked about.

        `masks` goes to 1.0 with it: dm_control episodes end on a time limit, never on an
        absorbing state, so the value target must always bootstrap. The goal-reaching mask
        (0 at the "goal reached" row) encodes a termination that does not exist here.

        Default OFF. Off, this is a no-op on the batch and draws no randomness, so every
        existing goal-conditioned run is bit-identical.
        """
        if not self.config.get('keep_dataset_rewards'):
            return
        if 'rewards' not in self.dataset:
            raise KeyError(
                "keep_dataset_rewards=true but the base dataset carries no 'rewards' field. "
                "The config is asking for the environment's real task reward and there is "
                "none to pass through (an unlabeled play dataset cannot supply one) -- "
                "failing rather than silently handing back the goal-reaching reward.")
        n = len(idxs)
        batch['rewards'] = np.asarray(self.dataset['rewards'][idxs], np.float32).reshape(n)
        batch['masks'] = np.ones(n, np.float32)

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

    def __post_init__(self):
        super().__post_init__()
        # HGCDataset's `sample` is its own -- it does NOT go through GCDataset.sample -- so
        # the pixel features added there (frame stacking, augmentation, the qpos measure
        # channel) are simply absent here. Refuse rather than silently ignore them: a config
        # that asks for a frame stack and gets unstacked frames is exactly the class of
        # quiet mismatch this tree keeps paying for.
        for key in ('frame_stack', 'p_aug', 'measure_qpos_slices'):
            if self.config.get(key) is not None:
                raise NotImplementedError(
                    f"HGCDataset does not implement {key!r} (it overrides GCDataset.sample "
                    f"entirely). Only GCDataset carries the pixel path.")

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
        self._maybe_keep_dataset_rewards(batch, idxs)

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
    rest of the h-step self-predictive rollout's (action, target-state) columns.

    Added for RLDP (impls/critics/rldp.py), whose self-predictive head rolls a learned
    predictor forward h steps from B(obs) and matches step k against the target backward
    map at s_{t+k}, k = 1..h. The true reference sampler is td_jepa's
    `metamotivo/buffers/parallel.py::_sample` (H > 1 branch): `traj_observation` holds
    s_{t+2}..s_{t+H} while `traj_action` holds a_{t+1}..a_{t+H-1} -- the observation
    column leads the action column by ONE STEP, so that after the agent prepends the
    one-step pair (a_t -> s_{t+1}), zipping actions with targets gives the aligned
    rollout (a_{t+k-1} -> s_{t+k}). This class emits the same layout:

      * future_observations -- [B, h-1, obs_dim],    column k = row `min(idx + 2 + k, traj_final_idx)`  (s_{t+2+k})
      * future_actions      -- [B, h-1, action_dim], column k = row `min(idx + 1 + k, traj_final_idx)`  (a_{t+1+k})

    on top of everything `FBDataset.sample` emits. Column k of future_observations is the
    next state REACHED BY column k of future_actions.

    HISTORY (2026-08-25 off-by-one fix). This class originally sliced BOTH windows at
    rows t+1..t+h-1 (copying the torch port agents/rldp/agent.py, which slices
    `batch["observation"][:, 1:]` where td_jepa reads the NEXT-observation sequence), so
    future_observations column 0 duplicated `next_observations` (s_{t+1}): rollout step 2
    was matched against s_{t+1} again -- training the predictor that a_{t+1} does not
    move the embedding -- and the rollout never reached s_{t+h}. The off-by-one was the
    port's, not td_jepa's. The observation window now extends one row further (to
    s_{t+h}) and drops the duplicated s_{t+1} column, reproducing the reference exactly.

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
        # Default False keeps the shipped clamping behavior bit-identical; the official
        # parity arm turns it on (see the branch in sample()).
        self.no_clamp_windows = bool(self.config.get('rldp_no_clamp_windows', False))
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
            # The observation window LEADS the action window by one row (see the class
            # docstring): column k pairs action row t+1+k with the state it reaches,
            # row t+2+k. Both windows clamp to the anchor's own trajectory end.
            if self.no_clamp_windows:
                # OFFICIAL RLDP draws segment starts from [0, L-H] (agent.py::sample_segments
                # step 2), so a window NEVER reaches a trajectory end and never clamps. The
                # clamping below instead REPEATS the terminal state, which feeds the SP head
                # "nothing moved" targets and biases the predictor toward the identity --
                # exactly the degenerate solution the phase-1 predictiveness gate exists to
                # detect. Here the anchors are already drawn, so rather than resample we
                # SHIFT any window that would overrun back so it ends at the trajectory end,
                # which preserves the batch size and the contiguity of the segment.
                span = h                       # the obs window reaches idx + h
                start = np.minimum(idxs, final_idxs - span)
                # floor at the anchor's own trajectory START, using the same
                # searchsorted convention GCDataset uses at line ~231.
                start = np.maximum(start, self.initial_locs[
                    np.searchsorted(self.initial_locs, idxs, side='right') - 1])
                act_win = start[:, None] + np.arange(1, h)[None, :]
                obs_win = start[:, None] + np.arange(2, h + 1)[None, :]
                act_win = np.minimum(act_win, final_idxs[:, None])
                obs_win = np.minimum(obs_win, final_idxs[:, None])
            else:
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
    """The FB batch plus ONE geometrically-sampled hindsight future observation.

    Added for HILP (impls/critics/hilp.py), whose phi objective takes `future_obs` as the
    goal and asks whether the current state IS that goal:
    `rewards = 1[||obs - goals|| < 1e-6] - 1`.

    WHY A NEW CLASS RATHER THAN AN EXISTING ONE. Neither existing option is HILP's
    sampler. `FBDataset` emits no goal at all (FB's goal IS next_obs, resolved inside the
    critic). `WindowedDataset` emits FIXED offsets 1..h, which is RLDP's rollout window,
    not a hindsight draw. HILP's buffer samples a GEOMETRIC offset
    (`in_memory_replay_buffer.py:231-233`):

        future_idx = step_idx + np.random.geometric(p=(1 - future))
        future_idx = np.clip(future_idx, 0, eps_lengths)

    with `future = 0.99` (train_offline.py:69), i.e. p = 0.01 and a mean offset of 100
    steps. Using a fixed offset instead would change WHICH states become goals and so what
    the Hilbert metric is fitted to -- it would be a different representation, silently.

    The clip is to the anchor row's OWN trajectory end, the same discipline
    WindowedDataset and HGCDataset use: a future drawn past the episode boundary would
    otherwise be a state from the NEXT episode, which is not a reachable future at all.
    numpy's `geometric` returns >= 1, so the future is always strictly ahead of the anchor;
    obs == goal still occurs, via the clip at the trajectory end, and that is exactly the
    row HILP's indicator reward is written for.
    """
    dataset: Any
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(np.asarray(self.dataset['terminals']) > 0)
        self.p = 1.0 - float(self.config.get('hilp_future', 0.99))
        assert 0.0 < self.p <= 1.0, (
            f"hilp_future must be in [0, 1); got {self.config.get('hilp_future')}")
        # The two goal REPLACEMENT probabilities (in_memory_replay_buffer.py:247-249).
        # Both default to 0 in train_offline.py's Config, but every published HILP-zsrl
        # command in the repo README passes p_randomgoal=0.375 -- so 0 is the source
        # default and 0.375 is the setting the numbers were produced at.
        self.p_randomgoal = float(self.config.get('hilp_p_randomgoal', 0.0))
        self.p_currgoal = float(self.config.get('hilp_p_currgoal', 0.0))

    def sample(self, batch_size, idxs=None):
        batch = self.dataset.sample(batch_size, idxs)
        idxs = np.asarray(batch['index']).reshape(-1)
        final_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        offs = np.random.geometric(p=self.p, size=idxs.shape[0])
        fut = np.minimum(idxs + offs, final_idxs)
        # `hilp_future_observations`, NOT `future_observations`: WindowedDataset already
        # owns that name for rldp's [B, h-1, obs] rollout window, and this is a single
        # [B, obs] draw. Same-name-different-shape is how a batch key silently becomes the
        # wrong tensor for one of its two consumers.
        goals = np.asarray(self.dataset['observations'][fut], np.float32)

        # Upstream applies TWO replacements, in this order, AFTER the geometric draw:
        #   future_obs = where(rand < p_currgoal/(1-p_randomgoal), curr_obs,   future_obs)
        #   future_obs = where(rand < p_randomgoal,                random_obs, future_obs)
        # (in_memory_replay_buffer.py:247-249). The odd first threshold is upstream's own:
        # dividing by (1-p_randomgoal) makes p_currgoal the MARGINAL current-goal rate
        # after the second replacement overwrites part of the first. Order and threshold
        # are reproduced rather than simplified.
        n = goals.shape[0]
        if self.p_currgoal > 0.0:
            thr = self.p_currgoal / max(1.0 - self.p_randomgoal, 1e-12)
            cur = np.asarray(batch['observations'], np.float32)
            goals = np.where((np.random.rand(n) < thr)[:, None], cur, goals)
        if self.p_randomgoal > 0.0:
            # A uniformly random row IS upstream's "random episode, random step within it"
            # on ExoRL, where every episode is exactly 1000 steps.
            rnd = np.asarray(
                self.dataset['observations'][np.random.randint(0, self.size, n)], np.float32)
            goals = np.where((np.random.rand(n) < self.p_randomgoal)[:, None], rnd, goals)

        batch['hilp_future_observations'] = goals
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
    if config.get('dataset_class') == 'WindowedDataset':
        return WindowedDataset(base, config)
    if config.get('dataset_class') == 'HILPFutureDataset':
        return HILPFutureDataset(base, config)
    if isinstance(base, FBDataset):
        dims = sorted(set(config.get('obs_mask_dims') or ()) |
                      set(config.get('goal_mask_dims') or ()))
        base.mask_dims = list(dims) or None
    return base
