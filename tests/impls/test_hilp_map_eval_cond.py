"""Goal-conditioned eval reaches every z-conditioned critic through map_eval_cond.

HILP maps a goal to z = sqrt(d) * normalize(phi(g) - phi(s)) with centered phi.
"""

import inspect

import jax.numpy as jnp
import numpy as np

from env import make_config
from impls.agents import CRITICS, Agent

OBS_DIM, ACT_DIM, B = 37, 5, 6


def _reads_z(cls):
    return 'cond["z"]' in inspect.getsource(cls.cond_vec)


def test_every_z_conditioned_critic_resolves_map_eval_cond():
    """Every critic whose cond_vec reads z defines map_eval_cond."""
    missing = [
        name
        for name, cls in CRITICS.items()
        if _reads_z(cls) and not hasattr(cls, "map_eval_cond")
    ]
    assert not missing, (
        f"z-conditioned critics with NO map_eval_cond (goal-blind on the goal route): "
        f"{missing}"
    )


def _built(critic, overrides=()):
    cfg = make_config(critic, "flowbc", "cube_single", list(overrides))
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, OBS_DIM)))
    act = jnp.asarray(rng.standard_normal((B, ACT_DIM)))
    return Agent.create(critic, "flowbc", 0, obs, act, cfg), obs, act, cfg


def _reference_z(ag, goal, obs):
    """Goal-to-z map from the raw phi1 tower, centered by the stats slot."""
    cr = ag.critic
    p = ag.net["hilp_phi1"].params
    mean = ag.net["hilp_stats"].params["mean"]
    phi = lambda x: cr.hilp_phi1.apply({"params": p}, x) - mean
    diff = np.asarray(phi(goal) - phi(obs))
    return np.sqrt(diff.shape[-1]) * diff / np.linalg.norm(diff, axis=-1, keepdims=True)


def test_hilp_goal_maps_to_the_reference_z():
    ag, obs, _, _ = _built("hilp")
    cr = ag.critic
    # Non-zero stats mean, so a missing centering is caught.
    stats = ag.net["hilp_stats"]
    ag = ag.replace(
        net={
            **ag.net,
            "hilp_stats": stats.replace(
                params={**stats.params, "mean": jnp.full((cr.z_dim,), 0.3)}
            ),
        }
    )
    goal, s = obs[0] + 2.0, obs[1]
    cond0 = {"z": jnp.zeros((cr.z_dim,))}
    assert float(jnp.linalg.norm(cr.cond_vec(cond0))) == 0.0

    mapped = cr.map_eval_cond(ag.net, {**cond0, "goal": goal}, observations=s)
    z = np.asarray(mapped["z"])
    assert z.shape == (cr.z_dim,)
    assert np.linalg.norm(z) > 1.0, "mapped z is (near) zero: still goal-blind"
    np.testing.assert_allclose(np.linalg.norm(z), np.sqrt(cr.z_dim), rtol=1e-5)
    np.testing.assert_allclose(z, _reference_z(ag, goal, s), rtol=1e-5, atol=1e-6)

    # Batch of observations, one goal: one z per row.
    mb = cr.map_eval_cond(ag.net, {**cond0, "goal": goal}, observations=obs)
    np.testing.assert_allclose(
        np.asarray(mb["z"]), _reference_z(ag, goal, obs), rtol=1e-5, atol=1e-6
    )
    # No goal in cond: the hook is a no-op.
    assert cr.map_eval_cond(ag.net, cond0, observations=s) is cond0


def test_hilp_actor_is_not_goal_blind_on_the_goal_route():
    """Two goals give two different HILP actions through Agent.sample_actions."""
    ag, obs, _, _ = _built("hilp")
    s = obs[0]
    g1, g2 = obs[1].at[19:22].add(0.5), obs[2].at[19:22].add(-0.5)
    a1 = ag.sample_actions(s, goals=g1, seed=ag.rng)
    a2 = ag.sample_actions(s, goals=g2, seed=ag.rng)
    a1b = ag.sample_actions(s, goals=g1, seed=ag.rng)
    np.testing.assert_allclose(a1, a1b)  # deterministic given (s, g, seed)
    assert not np.allclose(a1, a2), (
        "same action for two goals: the actor never saw the goal"
    )
