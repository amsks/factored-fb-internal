"""The goal-conditioned eval route must reach every z-conditioned critic.

`Agent.sample_actions` (impls/agents/__init__.py) merges the env's goal frame into cond and
then calls `critic.map_eval_cond` ONLY IF the critic has one. A z-conditioned critic without
the hook rolls its actor on `eval_cond0`'s zero z for the whole episode and the policy is
goal-blind, silently. That is the 2026-09-08 HILP bug: hilp / hilp_density had no hook, so
every goal-conditioned HILP cell (the privcoord table, the 'HILP 0.14' cube number) was a
z = 0 rollout.

The reference goal->z map is official HILP's `get_goal_meta`
(~/git/baselines/HILP/hilp_zsrl/url_benchmark/agent/sf.py:514-532):

    z = sqrt(d) * normalize(feature_net(g) - feature_net(s))

with feature_net = phi1 - running_mean (sf.py:120-122) and s the CURRENT observation,
recomputed every step (train_offline.py:325-327). td_jepa has no goal-conditioned HILP
eval at all (only `reward_inference`, sf/model.py:144), so official HILP is the reference.
"""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.agents import Agent, CRITICS
from impls.critics._measure_common import scaled_hit_cols

OBS_DIM, ACT_DIM, B = 37, 5, 6


def _reads_z(cls):
    return 'cond["z"]' in inspect.getsource(cls.cond_vec)


def test_every_z_conditioned_critic_resolves_map_eval_cond():
    """Every critic whose cond_vec reads cond["z"] must own the eval hook, or the
    goal-conditioned route rolls it on z = 0."""
    missing = [name for name, cls in CRITICS.items()
               if _reads_z(cls) and not hasattr(cls, "map_eval_cond")]
    assert not missing, (
        f"z-conditioned critics with NO map_eval_cond (goal-blind on the goal route): "
        f"{missing}")


def _built(critic, overrides=()):
    cfg = make_config(critic, "flowbc", "cube_single", list(overrides))
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
    act = jnp.asarray(rng.standard_normal((B, ACT_DIM)))
    return Agent.create(critic, "flowbc", 0, obs, act, cfg), obs, act, cfg


def _reference_z(ag, goal, obs):
    """Official HILP get_goal_meta, reconstructed WITHOUT the critic's helpers: the raw
    flax phi1 tower on the critic's phi input view, centered by the stats slot."""
    cr = ag.critic
    p = ag.net["hilp_phi1"].params
    mean = ag.net["hilp_stats"].params["mean"]
    view = (lambda x: x) if cr.backward_input == "state" else \
        (lambda x: scaled_hit_cols(x, cr.blocks))
    phi = lambda x: cr.hilp_phi1.apply({"params": p}, view(x)) - mean
    diff = np.asarray(phi(goal) - phi(obs))
    return np.sqrt(diff.shape[-1]) * diff / np.linalg.norm(diff, axis=-1, keepdims=True)


HILP_ARMS = ["hilp_oc", "hilp_density_oc", "hilp_tdjc", "hilp_density_tdjc"]


@pytest.mark.parametrize("critic", HILP_ARMS)
def test_hilp_goal_maps_to_the_reference_z(critic):
    ag, obs, _, _ = _built(critic)
    cr = ag.critic
    # A stats mean that is NOT zero, so a hook that forgets the centering is caught
    # (it cancels in the difference only if applied to BOTH terms).
    stats = ag.net["hilp_stats"]
    ag = ag.replace(net={**ag.net, "hilp_stats": stats.replace(
        params={**stats.params, "mean": jnp.full((cr.z_dim,), 0.3)})})
    goal, s = obs[0] + 2.0, obs[1]
    cond0 = {"z": jnp.zeros((cr.z_dim,))}
    assert float(jnp.linalg.norm(cr.cond_vec(cond0))) == 0.0

    mapped = cr.map_eval_cond(ag.net, {**cond0, "goal": goal}, observations=s)
    z = np.asarray(mapped["z"])
    assert z.shape == (cr.z_dim,)
    assert np.linalg.norm(z) > 1.0, "mapped z is (near) zero: still goal-blind"
    np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(cr.z_dim), rtol=1e-5)
    np.testing.assert_allclose(z, _reference_z(ag, goal, s), rtol=1e-5, atol=1e-6)

    # A batch of observations against one goal: one z per row, each the reference.
    mb = cr.map_eval_cond(ag.net, {**cond0, "goal": goal}, observations=obs)
    np.testing.assert_allclose(np.asarray(mb["z"]), _reference_z(ag, goal, obs),
                               rtol=1e-5, atol=1e-6)
    # No goal in cond: the hook is a no-op (the reward-sample route keeps its z).
    assert cr.map_eval_cond(ag.net, cond0, observations=s) is cond0


@pytest.mark.parametrize("critic", ["hilp_oc", "hilp_density_oc"])
def test_hilp_actor_is_not_goal_blind_on_the_goal_route(critic):
    """Through Agent.sample_actions -- the path the evaluator drives -- two different goals
    must give two different actions. Before the hook both rolled on z = 0."""
    ag, obs, _, _ = _built(critic)
    s = obs[0]
    g1, g2 = obs[1].at[19:22].add(0.5), obs[2].at[19:22].add(-0.5)
    a1 = ag.sample_actions(s, goals=g1, seed=ag.rng)
    a2 = ag.sample_actions(s, goals=g2, seed=ag.rng)
    a1b = ag.sample_actions(s, goals=g1, seed=ag.rng)
    np.testing.assert_allclose(a1, a1b)   # deterministic given (s, g, seed)
    assert not np.allclose(a1, a2), "same action for two goals: the actor never saw the goal"


def test_hilp_density_hook_is_the_parents_not_a_second_application():
    """hilp_density's whitened-goal hook lives in sample_cond (TRAINING, goal_kernel only);
    at eval it must inherit the parent's map exactly once."""
    from impls.critics.hilp import HILPCritic
    from impls.critics.hilp_density import HILPDensityCritic
    assert HILPDensityCritic.map_eval_cond is HILPCritic.map_eval_cond
    ag, obs, _, _ = _built("hilp_density_oc")
    cr = ag.critic
    cond = {"z": jnp.zeros((cr.z_dim,)), "goal": obs[0] + 1.0}
    once = cr.map_eval_cond(ag.net, cond, observations=obs[1])
    twice = cr.map_eval_cond(ag.net, once, observations=obs[1])
    np.testing.assert_allclose(np.asarray(once["z"]), np.asarray(twice["z"]))
    np.testing.assert_allclose(np.asarray(once["z"]), _reference_z(ag, obs[0] + 1.0, obs[1]),
                               rtol=1e-5, atol=1e-6)
