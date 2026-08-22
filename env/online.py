"""env/online.py — the ONLINE (env-interactive) half of the training loop.

impls/main.py is offline by default and stays that way: `--online` (default False) is the
only switch, and when it is off nothing in this file is imported, constructed or executed
(impls/main.py imports it lazily, inside the `if FLAGS.online:` block).

What online mode adds
---------------------
    ReplayBuffer   transitions collected from the env, in the SAME flat layout the offline
                   loaders produce (env/unified.py::load_unified): observations / actions /
                   next_observations / rewards / terminals, with `terminals` marking each
                   episode's LAST row (a trajectory BOUNDARY, not an absorbing state).
    OnlineTrainer  drives the env between gradient steps and hands the training loop batches.

The batch-key problem, and why the buffer does not build batches itself
-----------------------------------------------------------------------
A critic's `stage_loss` reads batch keys directly, so a missing key surfaces as an opaque
KeyError several frames deep. Rather than re-derive the key set (which differs per critic:
FBDataset emits 5 keys, GCDataset 9, HGCDataset a different 9), the buffer exposes its
storage as an `FBDataset` and runs it through the SAME `make_dataset(base, cfg)` the offline
path uses. The online batch is therefore produced by exactly the dataset class the offline
run would have used, with the same keys, the same dtypes, the same obs/goal masking and the
same np.random draw order. `_check_keys` still validates the result up front against the key
set the offline dataset actually emitted, and names any missing key.

GOAL KEYS FOR GOAL-CONDITIONED CRITICS -- the deliberate choice, and its limits
------------------------------------------------------------------------------
`value_goals` / `actor_goals` are NOT synthesised, and are NEVER zeros. Because the buffer
is wrapped by the real `GCDataset`, they are HINDSIGHT goals relabelled from the online
trajectories themselves -- future states of the sampled transition's own episode, under the
config's value_p_curgoal / p_trajgoal / p_randomgoal split. That is the same relabelling
offline uses; only the trajectories are new. This is what makes online mode work for the
goal-conditioned critics at all, and it is why the buffer stores whole episodes with
boundary `terminals` instead of loose transitions.

Two real limitations follow, and neither is papered over:

  1. Only COMPLETE episodes are visible. GCDataset needs trajectory boundaries and asserts
     its last row is terminal, so the in-progress episode is held back until it ends. With a
     long horizon and a small `--online_start_steps` the first sample would otherwise lag the
     first env step by up to one episode; `prefill()` blocks until at least one episode has
     finished for exactly this reason.
  2. The COLLECTION policy is conditioned on the EVAL goal (`eval_goal_mode`: one fixed goal
     frame, or a per-episode synthesised one), not on a diverse goal distribution. Online
     exploration is therefore task-directed, and the consequence compounds: the buffer's
     state coverage is whatever pi(.|s, g_eval) induces, and the hindsight goals GCDataset
     then relabels are drawn from THAT support, so both halves of the online batch --
     the states and the goals -- inherit the eval task's narrowness. Nothing in the loss
     notices; it simply trains a successor measure on a sliver of the state space.

     `--online_explore_goal_prob` (default 0.0 = the behaviour described above, exactly)
     is the one small dial provided against this: with probability p, an episode is
     collected against a goal drawn uniformly from the states already in the replay buffer
     instead of the eval goal. It widens the support the rollouts visit; it is NOT a
     principled exploration schedule (no novelty, no coverage objective, no curriculum) and
     is not claimed to be one. See `OnlineTrainer._episode_goal`.

Reproducibility
---------------
Two RNG streams are kept off the global NumPy stream that GCDataset draws its batch indices
from, so turning online mode on does not reshuffle anything a seed is supposed to control:
exploration noise and warmup actions come from a private `np.random.Generator`, and the
policy's own randomness from a private JAX key. Env resets go through
`env/unified.py::seeded_reset`, which saves and restores the global state itself.
Reset seeds are shifted by `--online_reset_seed_offset` (default 1e6) so that ONLINE
episodes never land on the initial states `evaluate()` scores (it uses seeds 0..N).
"""
import json
import os

import numpy as np

from impls.utils.datasets import FBDataset, make_dataset

# Bumped whenever the on-disk layout of online_state_<step>.npz changes incompatibly.
STATE_VERSION = 1


def state_path(run_dir, step):
    """Where a run's online state (replay buffer + collector position + RNG) is written.
    Sits beside params_<step>.pkl so a checkpoint and its buffer travel together."""
    return os.path.join(run_dir, f"online_state_{int(step)}.npz")


# The keys the offline datasets emit, for reference and for tests. The actual requirement is
# per-critic and is taken from the offline dataset at runtime (see OnlineTrainer.create):
# FBDataset emits five of these, GCDataset adds next_actions/masks plus value_goals and
# actor_goals.
BATCH_KEYS = ("observations", "actions", "next_observations", "next_actions",
              "rewards", "masks", "terminals")

_STORED = ("observations", "actions", "next_observations", "rewards", "terminals")


def _check_keys(batch, required, where):
    """Raise a message that NAMES the missing key. Without this the failure mode is a bare
    KeyError inside a critic's stage_loss, several jit frames from anything readable."""
    missing = [k for k in required if k not in batch]
    if missing:
        raise KeyError(
            f"{where}: online batch is missing {missing} -- the offline dataset for this "
            f"critic emits {sorted(required)}, the online batch has {sorted(batch)}. A "
            f"critic's stage_loss reads these keys directly, so this would otherwise fail "
            f"as an opaque KeyError deep inside the loss."
        )


def _flag(x):
    return bool(x.any()) if hasattr(x, "any") else bool(x)


class ReplayBuffer:
    """Fixed-capacity, episode-structured transition store.

    Rows are written contiguously (never a wrapping ring): GCDataset's goal sampling is
    trajectory-relative and asserts the last row is a trajectory end, so the storage has to
    stay a valid flat dataset at all times. When it fills, the OLDEST WHOLE EPISODES are
    dropped and the survivors moved down -- amortised O(1) per transition, at the cost of one
    memmove per compaction.

    `terminals` follows the offline convention (env/unified.py::load_unified): 1.0 on each
    episode's last row as a BOUNDARY marker. It is not an absorbing-state flag, and FBDataset
    zeroes it in the emitted batch anyway (always-gamma bootstrap).
    """

    def __init__(self, capacity, obs_dim, act_dim, config, required_keys=None):
        capacity = int(capacity)
        assert capacity > 0, f"online_buffer_size must be positive, got {capacity}"
        self.capacity = capacity
        self.config = config
        self.required_keys = tuple(required_keys) if required_keys else ()
        self.observations = np.zeros((capacity, int(obs_dim)), np.float32)
        self.actions = np.zeros((capacity, int(act_dim)), np.float32)
        self.next_observations = np.zeros((capacity, int(obs_dim)), np.float32)
        self.rewards = np.zeros((capacity,), np.float32)
        self.terminals = np.zeros((capacity,), np.float32)
        self._n = 0            # rows written, INCLUDING the in-progress episode
        self._n_complete = 0   # rows in finished episodes -- what sampling sees
        self._episodes = 0     # finished episodes currently stored
        self._dirty = True
        self._view = None

    # ---------------- introspection ----------------
    @property
    def size(self):
        """Rows visible to sampling (finished episodes only)."""
        return self._n_complete

    @property
    def n_written(self):
        return self._n

    @property
    def n_episodes(self):
        return self._episodes

    # ---------------- writing ----------------
    def add(self, obs, action, next_obs, reward):
        if self._n >= self.capacity:
            self._compact()
        i = self._n
        self.observations[i] = obs
        self.actions[i] = action
        self.next_observations[i] = next_obs
        self.rewards[i] = reward
        self.terminals[i] = 0.0     # the slot may hold a 1 left by a compacted-away episode
        self._n = i + 1

    def end_episode(self):
        """Close the in-progress episode: mark its last row as the trajectory boundary and
        make it visible to sampling. A no-op when no rows are pending."""
        if self._n <= self._n_complete:
            return False
        self.terminals[self._n - 1] = 1.0
        self._n_complete = self._n
        self._episodes += 1
        self._dirty = True
        return True

    def seed_from(self, dataset, max_transitions=None):
        """Warm-start from the offline dataset (`--online_seed_offline`).

        Takes a PREFIX of whole episodes -- deterministic, and draws no randomness, so
        seeding cannot shift the batch stream a --seed is meant to control. Rows are copied,
        so later compaction never touches the offline arrays.
        """
        fields = {k: np.asarray(dataset[k]) for k in _STORED}
        n = len(fields["observations"])
        ends = np.nonzero(fields["terminals"] > 0)[0]
        if not len(ends):
            raise ValueError(
                "--online_seed_offline: the offline dataset has no terminals, so it cannot "
                "be split into episodes; the replay buffer stores whole episodes only")
        limit = self.capacity if max_transitions is None else min(self.capacity,
                                                                  int(max_transitions))
        limit = min(limit, n)
        take = ends[ends < limit]                       # whole episodes that fit in `limit`
        if not len(take):
            raise ValueError(
                f"--online_seed_offline: the first offline episode is {int(ends[0]) + 1} rows, "
                f"which does not fit in the seeding budget of {limit} rows "
                f"(--online_buffer_size / --online_seed_transitions)")
        cut = int(take[-1]) + 1
        for name in _STORED:
            getattr(self, name)[:cut] = fields[name][:cut]
        self._n = self._n_complete = cut
        self._episodes = int(len(take))
        self._dirty = True
        return cut

    # ---------------- persistence (--restore_path with --online) ----------------
    def state_dict(self):
        """The FINISHED-episode prefix, as plain arrays.

        The in-progress episode is deliberately dropped: it is not sampleable (no boundary
        terminal), the collector restarts on a fresh `env.reset()` anyway, and keeping it
        would mean also persisting the env's own internal state, which no backend here
        exposes. The cost is at most one partial episode of collected data per resume.
        """
        n = self._n_complete
        out = {name: getattr(self, name)[:n].copy() for name in _STORED}
        out["_n_complete"] = np.int64(n)
        out["_episodes"] = np.int64(self._episodes)
        out["_capacity"] = np.int64(self.capacity)
        return out

    def load_state_dict(self, d):
        """Inverse of `state_dict`, with the shape checks that turn a mismatched resume into
        a named error instead of a silently half-filled buffer."""
        obs = np.asarray(d["observations"])
        act = np.asarray(d["actions"])
        n = int(np.asarray(d["_n_complete"]))
        if obs.shape[1:] != self.observations.shape[1:] or act.shape[1:] != self.actions.shape[1:]:
            raise ValueError(
                f"online state has observations{obs.shape[1:]} / actions{act.shape[1:]} but "
                f"this run's buffer is observations{self.observations.shape[1:]} / "
                f"actions{self.actions.shape[1:]} -- the checkpoint is from a different "
                f"env/domain, and restoring it would train on the wrong shapes.")
        if n > self.capacity:
            raise ValueError(
                f"online state holds {n} rows but --online_buffer_size is {self.capacity}. "
                f"Raise it to at least {n} (silently truncating a replay buffer changes the "
                f"data distribution the resume trains on).")
        for name in _STORED:
            getattr(self, name)[:n] = np.asarray(d[name])
        self._n = self._n_complete = n
        self._episodes = int(np.asarray(d["_episodes"]))
        self._dirty = True
        return n

    def _compact(self):
        """Drop the oldest whole episodes until at least half the capacity is free."""
        ends = np.nonzero(self.terminals[:self._n_complete] > 0)[0]
        # Candidate cut points: the start of every stored episode, plus the start of the
        # in-progress one (= drop every finished episode, keep only what is being collected).
        starts = [0] + [int(s) for s in (ends[:-1] + 1)] + [self._n_complete]
        target = self.capacity // 2
        keep_from = next((s for s in starts if s > 0 and self._n - s <= target), None)
        if keep_from is None:
            raise RuntimeError(
                f"online replay buffer cannot be compacted: the in-progress episode is "
                f"{self._n - self._n_complete} rows and the newest complete episodes do not "
                f"fit in half of --online_buffer_size ({self.capacity}). Raise the buffer "
                f"size to at least a few times the episode horizon.")
        remaining = int(np.count_nonzero(self.terminals[keep_from:self._n_complete] > 0))
        span = self._n - keep_from
        for name in _STORED:
            arr = getattr(self, name)
            arr[:span] = arr[keep_from:self._n]
        self._n = span
        self._n_complete = max(0, self._n_complete - keep_from)
        self._episodes = remaining
        self._dirty = True

    # ---------------- reading ----------------
    def _rebuild(self):
        """Expose the finished-episode prefix as the SAME dataset class the offline run uses.

        `freeze=False` because the storage arrays are reused in place; every batch leaves
        through fancy indexing, which copies, so nothing downstream aliases them.
        """
        n = self._n_complete
        fields = {name: getattr(self, name)[:n] for name in _STORED}
        self._view = make_dataset(FBDataset.create(freeze=False, **fields), self.config)
        self._dirty = False

    def dataset(self):
        if self._n_complete == 0:
            raise RuntimeError(
                "online replay buffer holds no finished episode yet -- sampling needs whole "
                "trajectories (GCDataset's goals are trajectory-relative). Raise "
                "--online_start_steps to at least one episode horizon.")
        if self._dirty:
            self._rebuild()
        return self._view

    def sample(self, batch_size):
        batch = self.dataset().sample(batch_size)
        if self.required_keys:
            _check_keys(batch, self.required_keys, "ReplayBuffer.sample")
        return batch


class OnlineTrainer:
    """Steps the env between gradient steps and serves batches from the replay buffer.

    `env_steps_per_update` is the env-steps-to-gradient-steps ratio (UTD = 1 / ratio). It is
    a float carried on an accumulator, so 0.5 means one env step every other gradient step
    (UTD 2) and 4 means four env steps per gradient step.
    """

    def __init__(self, adapter, env, cfg, buffer, *, env_steps_per_update=1.0,
                 action_noise=0.1, temperature=1.0, warmup_steps=0,
                 reset_seed_offset=1_000_000, seed=0, horizon=None, tasks=None,
                 explore_goal_prob=0.0):
        self.adapter = adapter
        self.env = env
        self.cfg = cfg
        self.buffer = buffer
        self.ratio = float(env_steps_per_update)
        assert self.ratio >= 0.0, "online_env_steps_per_update must be >= 0"
        self.noise = float(action_noise)
        self.temperature = float(temperature)
        self.warmup_steps = int(warmup_steps)
        self.reset_seed_offset = int(reset_seed_offset)
        self.explore_goal_prob = float(explore_goal_prob)
        assert 0.0 <= self.explore_goal_prob <= 1.0, \
            f"online_explore_goal_prob must be in [0,1], got {explore_goal_prob}"
        self.obs_mask = list(cfg.get("obs_mask_dims") or ())
        self.goal_mask = list(cfg.get("goal_mask_dims") or ())
        self.horizon = int(horizon if horizon is not None
                           else (getattr(adapter, "_horizon", None) or cfg.get("horizon", 100)))
        self.tasks = list(tasks) if tasks else [None]

        import jax
        self._key = jax.random.PRNGKey(int(seed))
        self._np_rng = np.random.default_rng(int(seed))  # NOT the global stream, see docstring
        self._debt = 0.0
        self._obs = None
        self._ep_goal = None
        self._ep_len = 0
        self._ep_return = 0.0
        self._ep_success = False
        self._episode_index = 0
        self.env_steps = 0
        self._returns, self._successes, self._lengths = [], [], []

    # ------------ env plumbing (reuses env/unified.py's rollout machinery) ------------
    def _is_unified(self):
        return getattr(self.adapter, "name", "") == "unified"

    def _mask_obs(self, o):
        """Same rule as env/unified.py::evaluate's _mask_obs. The policy must see the SAME
        blanked observation online that it sees at eval and in training batches; the buffer
        itself stores the RAW frame, exactly as the offline loaders do, and the dataset
        wrapper applies obs_mask_dims on the way out."""
        if not self.obs_mask:
            return np.asarray(o, np.float32)
        o = np.array(o, np.float32, copy=True)
        o[..., self.obs_mask] = 0.0
        return o

    def _prep_goal(self, g):
        """Same rule as env/unified.py::evaluate's _prep_goal."""
        g = np.array(g, np.float32, copy=True)
        if self.goal_mask:
            g[self.goal_mask] = 0.0
        if self.obs_mask:
            g[self.obs_mask] = 0.0
        return g

    def _needs_goal(self, agent):
        """LITERALLY the rule env/unified.py::evaluate uses to decide whether the actor has
        to be handed an explicit goal frame -- the same function, not a second copy of it
        (the copy is how the two drifted apart before)."""
        from env.base import needs_goal_frame
        return needs_goal_frame(agent)

    def _reset(self):
        task = self.tasks[self._episode_index % len(self.tasks)]
        if self._is_unified():
            from env.unified import seeded_reset
            # Offset so online rollouts never reuse the initial states evaluate() scores.
            return seeded_reset(self.env, self.cfg["task"],
                                self.reset_seed_offset + self._episode_index)
        return self.env.reset(options=dict(task_id=task, render_goal=False))

    def _episode_goal(self, agent, raw_obs, info):
        """The goal this EPISODE is collected against -- i.e. what the exploration policy is
        conditioned on for the whole rollout.

        ⚠ READ THIS BEFORE TRUSTING ONLINE COVERAGE. By default this is the EVAL goal: one
        fixed frame from the offline data, or the per-episode synthesised one. Every online
        episode is therefore a rollout of pi(.|s, g_eval), the buffer holds only the states
        that policy reaches, and GCDataset's hindsight relabelling can only draw goals from
        those same states. Online mode does not widen a critic's goal support by itself; it
        deepens it along one task. That is a real limitation of this file, not an accident of
        configuration (module docstring, limitation 2).

        `--online_explore_goal_prob` (default 0.0) is the deliberately small dial against it:
        per episode, with that probability, collect against a state sampled uniformly from
        the buffer instead. The draw uses the PRIVATE generator, and at p == 0 no draw
        happens at all, so the default path is bit-identical to before the knob existed.
        """
        goal = self._eval_goal(agent, raw_obs, info)
        if goal is None or self.explore_goal_prob <= 0.0:
            return goal
        # Buffer-empty (first episodes) has nothing to sample: fall back to the eval goal
        # rather than inventing a frame.
        if self.buffer.size > 0 and self._np_rng.random() < self.explore_goal_prob:
            i = int(self._np_rng.integers(self.buffer.size))
            return self._prep_goal(self.buffer.next_observations[i])
        return goal

    def _eval_goal(self, agent, raw_obs, info):
        """The eval-protocol goal frame -- the pre-existing `_episode_goal` body, unchanged."""
        if not self._needs_goal(agent):
            return None
        if not self._is_unified():
            goal = info.get("goal") if hasattr(info, "get") else None
            if goal is None:
                raise RuntimeError(
                    "online mode: this critic needs a goal frame but the env reset info "
                    "carries no 'goal'. Online rollouts have no other source for it.")
            return np.asarray(goal, np.float32)
        mode = str(self.cfg.get("eval_goal_mode", "fixed"))
        if mode == "synth":
            from env.unified import synth_goal_frame
            return self._prep_goal(synth_goal_frame(self.cfg["task"], raw_obs))
        goal = getattr(self.adapter, "_goal", None)
        if goal is None:
            raise RuntimeError(
                "online mode with a goal-conditioned critic needs an eval goal frame. "
                "eval_goal_mode='fixed' takes it from the highest-reward offline "
                "next_observation, which is unavailable here; set eval_goal_mode=synth (needs "
                "a goal-in-obs layout in env/unified.py::_GOAL_LAYOUT) or run a critic that "
                "conditions on z instead.")
        return self._prep_goal(goal)

    def _begin_episode(self, agent):
        obs, info = self._reset()
        self._obs = np.asarray(obs, np.float32)
        self._ep_goal = self._episode_goal(agent, obs, info)
        self._ep_len = 0
        self._ep_return = 0.0
        self._ep_success = _flag(info.get("success", 0)) if hasattr(info, "get") else False

    def _action(self, agent):
        """Exploration policy: the agent's own sample_actions plus Gaussian noise, uniform
        random during warmup. Both the noise and the warmup draws come from the private
        generator, so exploration never shifts the batch stream.

        The only exploration pressure here is isotropic action noise around a policy that is
        conditioned on `self._ep_goal` -- see `_episode_goal` for why that goal is normally
        the eval goal, and what that costs the buffer's coverage."""
        act_dim = self.buffer.actions.shape[1]
        if self.env_steps < self.warmup_steps:
            return self._np_rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        import jax
        self._key, sub = jax.random.split(self._key)
        a = np.asarray(agent.sample_actions(observations=self._mask_obs(self._obs),
                                            goals=self._ep_goal, seed=sub,
                                            temperature=self.temperature), np.float32)
        if self.noise > 0.0:
            a = a + self.noise * self._np_rng.standard_normal(a.shape).astype(np.float32)
        return a

    def step_env(self, agent):
        """One env step, stored in the buffer. Closes the episode on terminated / truncated /
        horizon, which is what makes that row a trajectory boundary."""
        if self._obs is None:
            self._begin_episode(agent)
        action = np.clip(self._action(agent), -1.0, 1.0)
        next_obs, reward, term, trunc, info = self.env.step(action)
        next_obs = np.asarray(next_obs, np.float32)
        reward = float(np.asarray(reward).reshape(-1)[0])
        self.buffer.add(self._obs, action, next_obs, reward)
        self.env_steps += 1
        self._ep_len += 1
        self._ep_return += reward
        if hasattr(info, "get"):
            self._ep_success = self._ep_success or _flag(info.get("success", 0))
        self._obs = next_obs
        if _flag(term) or _flag(trunc) or self._ep_len >= self.horizon:
            self.buffer.end_episode()
            self._returns.append(self._ep_return)
            self._successes.append(float(self._ep_success))
            self._lengths.append(float(self._ep_len))
            for buf in (self._returns, self._successes, self._lengths):
                del buf[:-100]                          # keep a trailing window only
            self._episode_index += 1
            self._obs = None
        return info

    # ---------------- training-loop API ----------------
    def prefill(self, agent):
        """Collect --online_start_steps env steps before the first gradient step, and keep
        going until at least one episode has finished (sampling needs whole trajectories)."""
        while self.env_steps < self.warmup_steps or self.buffer.size == 0:
            self.step_env(agent)
        return self.env_steps

    def collect(self, agent):
        """The env steps owed to this gradient step, carried on a float accumulator so a
        fractional ratio (UTD > 1) is honoured over time rather than rounded away."""
        self._debt += self.ratio
        n = int(self._debt)
        self._debt -= n
        for _ in range(n):
            self.step_env(agent)
        return n

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    # ---------------- persistence (--restore_path with --online) ----------------
    #
    # WHY THIS EXISTS. `--restore_path` restores the AGENT (impls/utils/flax_utils.py::
    # restore_agent: params, target params, optimizer state, rng). Under --online that was
    # only half the run's state: the replay buffer came back EMPTY, `prefill` re-ran, and the
    # resumed run trained a fully-trained agent against a few thousand fresh warmup
    # transitions -- a silent distribution change that looks exactly like a working resume in
    # the logs. Everything the collector carries is therefore checkpointed here, and
    # impls/main.py REFUSES the combination outright when this file is missing rather than
    # falling back to that behaviour (see --online_restore_buffer).
    def save_state(self, run_dir, step):
        """Write the replay buffer + collector position + both private RNG streams next to
        params_<step>.pkl. Returns the path written."""
        import jax
        d = self.buffer.state_dict()
        d["_version"] = np.int64(STATE_VERSION)
        d["_env_steps"] = np.int64(self.env_steps)
        d["_episode_index"] = np.int64(self._episode_index)
        d["_debt"] = np.float64(self._debt)
        # `key_data` gives the raw uint32 words for either key flavour (legacy array key or
        # typed key<fry>); `restore_state` puts them back in whichever flavour this trainer
        # is already using, so a resume never silently changes the key's TYPE.
        d["_jax_key"] = np.asarray(jax.random.key_data(self._key))
        # The bit-generator state is a nested dict of python ints; json is the only lossless
        # container an npz will take.
        d["_np_rng_state"] = np.asarray(json.dumps(self._np_rng.bit_generator.state,
                                                   default=int))
        path = state_path(run_dir, step)
        np.savez(path, **d)
        return path

    def restore_state(self, run_dir, step):
        """Inverse of `save_state`. Raises FileNotFoundError (named, with the path) when the
        checkpoint has no online state -- the caller decides what that means, and
        impls/main.py treats it as a hard error."""
        import jax
        path = state_path(run_dir, step)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no online state at {path}. A --restore_path checkpoint carries the replay "
                f"buffer only if it was itself saved by an --online run of this code; "
                f"resuming without it would restart collection from an EMPTY buffer and "
                f"re-run prefill. Pass --online_restore_buffer=false if that is genuinely "
                f"what you want.")
        with np.load(path, allow_pickle=False) as z:
            d = dict(z)
        version = int(np.asarray(d["_version"]))
        if version != STATE_VERSION:
            raise ValueError(
                f"{path} was written by online-state version {version}, this code reads "
                f"{STATE_VERSION}")
        n = self.buffer.load_state_dict(d)
        self.env_steps = int(np.asarray(d["_env_steps"]))
        self._episode_index = int(np.asarray(d["_episode_index"]))
        self._debt = float(np.asarray(d["_debt"]))
        # Match the flavour the trainer was constructed with (see save_state): a raw uint32
        # array key stays raw, a typed key gets re-wrapped. wrap_key_data always returns a
        # TYPED key, so applying it unconditionally would change the object type across a
        # resume even though the bits are the same.
        key = jax.numpy.asarray(np.asarray(d["_jax_key"]))
        self._key = (key if jax.numpy.issubdtype(self._key.dtype, jax.numpy.integer)
                     else jax.random.wrap_key_data(key))
        self._np_rng.bit_generator.state = json.loads(str(np.asarray(d["_np_rng_state"])))
        # The in-progress episode was not persisted (see ReplayBuffer.state_dict), so the
        # next collect() starts a fresh one.
        self._obs = None
        return n

    def stats(self):
        out = {"online/env_steps": float(self.env_steps),
               "online/buffer_size": float(self.buffer.size),
               "online/buffer_episodes": float(self.buffer.n_episodes),
               "online/episodes_rolled": float(self._episode_index)}
        if self._returns:
            out["online/return"] = float(np.mean(self._returns))
            out["online/success"] = float(np.mean(self._successes))
            out["online/ep_len"] = float(np.mean(self._lengths))
        return out

    # ---------------- construction ----------------
    @classmethod
    def create(cls, adapter, env, cfg, offline_ds, example_batch, flags_ns,
               seed_offline=None):
        """Build the buffer + trainer from the resolved config and the --online_* flags.

        `example_batch` is the one-row offline batch impls/main.py already draws; its KEY SET
        is what online batches are validated against, so the requirement is whatever the
        offline dataset for THIS critic actually emits.

        `seed_offline` overrides --online_seed_offline; impls/main.py passes False when it is
        about to `restore_state`, since a restored buffer replaces its contents wholesale and
        seeding first would be work whose result is thrown away.
        """
        obs_dim = int(np.asarray(example_batch["observations"]).shape[-1])
        act_dim = int(np.asarray(example_batch["actions"]).shape[-1])
        buffer = ReplayBuffer(flags_ns.online_buffer_size, obs_dim, act_dim, cfg,
                              required_keys=tuple(example_batch.keys()))
        if seed_offline is None:
            seed_offline = flags_ns.online_seed_offline
        if seed_offline:
            base = getattr(offline_ds, "dataset", offline_ds)   # unwrap GCDataset/HGCDataset
            n = buffer.seed_from(base, flags_ns.online_seed_transitions or None)
            print(f"[online] seeded replay buffer with {n} offline rows "
                  f"({buffer.n_episodes} episodes)", flush=True)
        return cls(adapter, env, cfg, buffer,
                   env_steps_per_update=flags_ns.online_env_steps_per_update,
                   action_noise=flags_ns.online_action_noise,
                   temperature=flags_ns.online_temperature,
                   warmup_steps=flags_ns.online_start_steps,
                   reset_seed_offset=flags_ns.online_reset_seed_offset,
                   seed=flags_ns.seed, tasks=adapter.eval_tasks(env, cfg),
                   explore_goal_prob=getattr(flags_ns, "online_explore_goal_prob", 0.0))
