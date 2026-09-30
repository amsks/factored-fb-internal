import jax
import jax.numpy as jnp
import numpy as np

from impls.agents import Agent
from impls.utils.config_utils import load_config
from impls.utils.flax_utils import restore_agent, save_agent


def _cfg():
    return load_config("fb", "ddpgbc", "cube_single", overrides=["batch_size=8"])


def _batch(B=8, obs=8, act=5, seed=0):
    rng = np.random.default_rng(seed)
    return {"observations": rng.standard_normal((B, obs)).astype(np.float32),
            "actions": np.tanh(rng.standard_normal((B, act))).astype(np.float32),
            "next_observations": rng.standard_normal((B, obs)).astype(np.float32),
            "rewards": np.zeros((B, 1), np.float32),
            "terminals": np.zeros((B,), np.float32)}


def _flat(params):
    return np.asarray(jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(params)]))


def test_restore_recovers_params_bitwise(tmp_path):
    agent = Agent.create("fb", "ddpgbc", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg())
    for _ in range(3):
        agent, _ = agent.update(_batch())
    save_agent(agent, str(tmp_path), 3)

    fresh = Agent.create("fb", "ddpgbc", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg())
    restored = restore_agent(fresh, str(tmp_path), 3)
    for g in agent.net:
        np.testing.assert_array_equal(_flat(agent.net[g].params), _flat(restored.net[g].params))
