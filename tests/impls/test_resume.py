import jax.numpy as jnp
import numpy as np
from impls.agents import Agent
from impls.utils.flax_utils import save_agent, restore_agent


def _cfg():
    return dict(value_hidden_dims=[32, 32], layer_norm=True, discount=0.99, expectile=0.9,
                tau=0.005, lr=3e-4, lr_actor=3e-4, gc_negative=True,
                actor=dict(hidden_dims=[32, 32], alpha=0.3, const_std=True, layer_norm=False))


def _batch(B=8, obs=8, act=5, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((B, obs)).astype(np.float32)
    return {"observations": rng.standard_normal((B, obs)).astype(np.float32),
            "actions": np.tanh(rng.standard_normal((B, act))).astype(np.float32),
            "next_observations": rng.standard_normal((B, obs)).astype(np.float32),
            "value_goals": g, "actor_goals": g,
            "rewards": rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
            "masks": rng.integers(0, 2, size=B).astype(np.float32)}


def test_restore_recovers_params_bitwise(tmp_path):
    agent = Agent.create("gciql", "ddpgbc", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg())
    for _ in range(3):
        agent, _ = agent.update(_batch())
    save_agent(agent, str(tmp_path), 3)

    fresh = Agent.create("gciql", "ddpgbc", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg())
    restored = restore_agent(fresh, str(tmp_path), 3)
    for g in agent.net:
        a = jnp.ravel(jnp.concatenate([jnp.ravel(x) for x in
                                       jax_leaves(agent.net[g].params)]))
        b = jnp.ravel(jnp.concatenate([jnp.ravel(x) for x in
                                       jax_leaves(restored.net[g].params)]))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def jax_leaves(tree):
    import jax
    return jax.tree_util.tree_leaves(tree)
