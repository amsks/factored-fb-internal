"""Goal-conditioned training variants for the HILP/RLDP families (2026-09-06).

  hilp_density_gk / _gk_oc : HILP-density with goal_kernel (whitened goal index + kernel reward)
  rldp_density_gk_oc       : RLDP-density goal_kernel, object-coordinate B input
  hilp_gc / _oc, rldp_gc / _oc : vanilla HILP (mix_ratio 1.0) / RLDP (train_goal_ratio 1.0)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env import make_config
from impls.agents import Agent

OBS, ACT, B = 28, 5, 16


def _agent(critic, overrides=()):
    cfg = make_config(critic, "flowbc", "cube_single", list(overrides))
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, OBS))); act = jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))
    return Agent.create(critic, "flowbc", 0, obs, act, cfg), cfg


def _batch(seed=1):
    rng = np.random.default_rng(seed)
    return dict(observations=jnp.asarray(rng.standard_normal((B, OBS))),
                actions=jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))),
                next_observations=jnp.asarray(rng.standard_normal((B, OBS))),
                value_goals=jnp.asarray(rng.standard_normal((B, OBS))),
                actor_goals=jnp.asarray(rng.standard_normal((B, OBS))),
                rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                hilp_future_observations=jnp.asarray(rng.standard_normal((B, OBS))),
                # rldp's SP rollout (WindowedDataset): [B, horizon, OBS] future window
                future_observations=jnp.asarray(rng.standard_normal((B, 4, OBS))),
                future_actions=jnp.asarray(np.tanh(rng.standard_normal((B, 4, ACT)))))


@pytest.mark.parametrize("critic", ["hilp_density_gk", "hilp_density_gk_oc", "rldp_density_gk_oc",
                                    "hilp_gc", "hilp_gc_oc", "rldp_gc", "rldp_gc_oc", "density_fb_z_state"])
def test_new_configs_build(critic):
    agent, cfg = _agent(critic)
    assert agent is not None


def test_hilp_density_goal_kernel_default_off_is_unchanged():
    agent, _ = _agent("hilp_density_cube")
    c = agent.critic
    assert c.goal_kernel is False and c.kernel_blocks == ()
    cond = c.sample_cond(agent.net, _batch(), jax.random.PRNGKey(0))
    assert "_goal_obs" not in cond and "_pool_phi" in cond and "z" in cond


def test_hilp_density_goal_kernel_publishes_goal_and_kernel_operands():
    agent, _ = _agent("hilp_density_gk")
    c = agent.critic
    assert c.goal_kernel and len(c.kernel_blocks) == 1
    batch = _batch()
    cond = c.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(cond["_goal_obs"]), np.asarray(batch["actor_goals"]))
    assert cond["_pool_u_raw"].shape == (B, 3)
    assert cond["z"].shape == (B, c.z_dim)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(cond["z"]), axis=-1), np.sqrt(c.z_dim), rtol=1e-4)


def test_hilp_density_goal_kernel_changes_the_actor_reward_only():
    """Same params, same z: goal_kernel swaps the reward vector, nothing else."""
    import dataclasses
    agent, _ = _agent("hilp_density_gk")
    c_on = agent.critic; c_off = dataclasses.replace(c_on, goal_kernel=False)
    batch = _batch(); params = {k: v.params for k, v in agent.net.items()}
    cond = c_on.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    q_on, _ = c_on.q_fn(params, batch["observations"], cond, batch["actions"])
    q_off, _ = c_off.q_fn(params, batch["observations"], cond, batch["actions"])
    assert np.isfinite(np.asarray(q_on)).all() and np.isfinite(np.asarray(q_off)).all()
    assert not np.allclose(np.asarray(q_on), np.asarray(q_off))


def test_goal_kernel_agents_take_an_update_step():
    for critic in ("hilp_density_gk", "rldp_density_gk_oc", "hilp_gc", "rldp_gc"):
        agent, cfg = _agent(critic)
        b = _batch(); b = {**b, "observations": b["observations"]}
        agent, info = agent.update(b)
        assert all(np.isfinite(float(v)) for v in info.values() if np.ndim(v) == 0), critic


def test_vanilla_goal_conditioned_knobs():
    a, _ = _agent("hilp_gc"); assert a.critic.mix_ratio == 1.0
    a, _ = _agent("rldp_gc"); assert a.critic.train_goal_ratio == 1.0
