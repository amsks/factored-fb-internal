"""SACSingleTaskAgent: the vendored, non-goal-conditioned OGBench SAC agent
(`third_party/ogbench/impls/agents/sac.py`), made launchable via
`main.py`/`run_gciql.py` for offline single-task training against OGBench's
native `<env>-play-singletask-taskN-v0` reward-relabeled datasets.

Why this file exists (spec 27 Q4 / spec 29 section 2 -- the offline
supervised topline): the FB literature reports zero-shot success as a
percentage of an offline agent trained with the true task reward on the same
data. This repo had no such baseline wired up. `SACAgent` already IS exactly
that shape -- `critic_loss`/`actor_loss` call `GCActor`/`GCValue` with NO
goal argument at all, trained directly on `batch['rewards']`/`['masks']` --
but two things stop it working out of the box:

1. Its stock `get_config()` is missing keys `main.py`/`GCDataset` require
   (`dataset_class`, `discrete`, `encoder`, `frame_stack`, `p_aug`, the
   goal-sampling probabilities), because upstream only ever launches it
   through its own (unvendored) singletask harness.
2. `evaluate()` (`utils/evaluation.py`) is generic across every agent and
   ALWAYS calls `agent.sample_actions(..., goals=goal)` with the env's real
   goal state. Vendored `SACAgent.sample_actions` passes that `goals`
   argument straight to the actor network, which concatenates it onto the
   observation -- but the network was INITIALIZED (and trained, every
   `update()` call) with `goals=None` (no concatenation, obs-dim-only
   input), since `critic_loss`/`actor_loss` never pass a goal. The result is
   a hard shape mismatch on the first eval (actor input width doubles).

Never edits OGBench: (1) is solved by subclassing only `get_config` (extends
the unmodified `agents.sac.get_config()`); (2) is solved by overriding only
`sample_actions` to drop the `goals` argument before calling the actor --
three lines, otherwise byte-identical to `SACAgent.sample_actions`. Every
other method (`critic_loss`, `actor_loss`, `update`, `create`) is inherited
unchanged. Registered as `agents['sac_singletask']` by `sitecustomize.py`
(same pattern as `gciql_flow`/`crl_flowbc`/etc); does not touch or replace
`agents['sac']`.

`dataset_class='GCDataset'` (not a bespoke plain-`Dataset` path) is
deliberate: `main.py`'s `dataset_class` dict only knows `GCDataset`/
`HGCDataset`, and `Dataset` itself is not drop-in compatible with the
`dataset_class(Dataset.create(**train_dataset), config)` call main.py makes
(`Dataset.__init__` does not accept a `config` positional). Reusing
`GCDataset` means main.py needs no changes; the goal-sampling probabilities
below are set to something that satisfies `GCDataset.__post_init__`'s
sum-to-1 asserts, but this agent's `critic_loss`/`actor_loss` never read
`batch['goals']`/`batch['value_goals']`/`batch['actor_goals']` at all, so
their exact values are inert.

The one substantive training-time addition is `native_reward_singletask=
True`: it opts into the `GCDataset.sample` wrap installed by
`sitecustomize.py`, which overwrites GCDataset's synthetic hindsight
`rewards`/`masks` (computed from whether the sampled index equals a
resampled goal index) with the dataset's OWN `rewards`/`masks` fields --
the ones OGBench's `relabel_dataset` (`ogbench/relabel_utils.py`) already
wrote into the `.npz` for the fixed `_reward_task_id` when `env_name`
contains `singletask`. That is the actual environment reward for that one
task, not a goal-substitution proxy.
"""
import jax
import jax.numpy as jnp
import ml_collections
from agents.sac import SACAgent, get_config as _sac_get_config


class SACSingleTaskAgent(SACAgent):
    """SACAgent with an eval-safe `sample_actions` (see module docstring
    point 2). `update`/`critic_loss`/`actor_loss`/`create` are inherited
    unchanged from `SACAgent`."""

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        del goals  # never concatenated in training (critic_loss/actor_loss
        # call the actor with no goal); passing evaluate()'s real goal
        # through would double the actor's input width vs its init/train
        # shape (obs-dim only) and raise flax.errors.ScopeParamShapeError.
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions


def get_config():
    config = _sac_get_config()
    config.agent_name = 'sac_singletask'
    # -- keys main.py reads unconditionally (KeyError otherwise) --------------
    config.dataset_class = 'GCDataset'
    config.discrete = False
    config.encoder = ml_collections.config_dict.placeholder(str)
    config.frame_stack = ml_collections.config_dict.placeholder(int)
    # -- keys GCDataset.__post_init__/.sample require --------------------------
    config.p_aug = 0.0
    # This agent ignores batch['value_goals']/['actor_goals']; these just
    # need to sum to 1 per GCDataset's asserts. Fixed-random keeps
    # sample_goals() cheap (no geometric-sampling RNG loop) and avoids
    # reading trajectory-boundary structure this agent has no use for.
    config.value_p_curgoal = 0.0
    config.value_p_trajgoal = 0.0
    config.value_p_randomgoal = 1.0
    config.value_geom_sample = False
    config.actor_p_curgoal = 0.0
    config.actor_p_trajgoal = 0.0
    config.actor_p_randomgoal = 1.0
    config.actor_geom_sample = False
    config.gc_negative = True
    # -- the actual behavior change: real per-task reward, not hindsight -1/0 --
    config.native_reward_singletask = True
    return config
