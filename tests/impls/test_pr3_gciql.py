import numpy as np
import jax.numpy as jnp
import pytest
from impls.agents import Agent
from impls.critics.base import ComboError


def _cfg(actor_kind):
    base = dict(
        value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9, tau=0.005,
        lr=3e-4, lr_actor=3e-4, gc_negative=True,
    )
    if actor_kind in ("ddpgbc", "awr"):
        base["actor"] = dict(hidden_dims=[32, 32], alpha=3.0, const_std=True, layer_norm=False)
    else:  # fql
        base["actor"] = dict(hidden_dims=[32, 32], alpha=10.0, flow_steps=5,
                             normalize_q_loss=False, layer_norm=False)
    return base


def _batch(B=16, obs=8, act=5, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((B, obs)).astype(np.float32)
    return {"observations": rng.standard_normal((B, obs)).astype(np.float32),
            "actions": np.tanh(rng.standard_normal((B, act))).astype(np.float32),
            "next_observations": rng.standard_normal((B, obs)).astype(np.float32),
            "value_goals": g, "actor_goals": g,
            "rewards": rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
            "masks": rng.integers(0, 2, size=B).astype(np.float32)}


def test_gciql_provides_v_awr_ok():
    # AWR requires V; gciql provides it -> constructs (no ComboError).
    agent = Agent.create("gciql", "awr", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg("awr"))
    assert agent is not None


@pytest.mark.parametrize("actor_kind", ["ddpgbc", "awr", "fql"])
def test_gciql_pair_smoke_trains(actor_kind):
    agent = Agent.create("gciql", actor_kind, 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg(actor_kind))
    for _ in range(3):
        agent, info = agent.update(_batch())
    assert jnp.isfinite(jnp.asarray(info["value_loss"])).all()
    assert jnp.isfinite(jnp.asarray(info["critic_loss"])).all()
    assert jnp.isfinite(jnp.asarray(info["actor_loss"])).all()
