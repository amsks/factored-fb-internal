import numpy as np
import jax.numpy as jnp
from impls.agents import Agent
from tests.impls.test_pr3_gciql import _cfg


def test_sample_actions_with_goals():
    agent = Agent.create("gciql", "ddpgbc", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg("ddpgbc"))
    obs = jnp.asarray(np.zeros((4, 8), np.float32))
    goals = jnp.asarray(np.ones((4, 8), np.float32))
    a = agent.sample_actions(obs, goals=goals, seed=jnp.asarray(np.array([0, 0], np.uint32)))
    assert a.shape == (4, 5)
    assert jnp.isfinite(a).all()
    assert jnp.all(a <= 1.0) and jnp.all(a >= -1.0)  # clipped
