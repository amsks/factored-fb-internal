import numpy as np
import jax.numpy as jnp
import pytest
from impls.agents import Agent
from impls.critics.base import ComboError


def _cfg(actor_kind):
    base = dict(
        z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
        f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
        fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
        forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
    )
    if actor_kind in ("ddpgbc", "awr"):
        base["actor"] = dict(hidden_dims=[32, 32], alpha=3.0, const_std=True, layer_norm=False)
    else:  # fql
        base["actor"] = dict(hidden_dims=[32, 32], alpha=10.0, flow_steps=5,
                             normalize_q_loss=False, layer_norm=False)
    return base


def _batch(B=16, obs=8, act=5, seed=0):
    rng = np.random.default_rng(seed)
    return {"observations": rng.standard_normal((B, obs), np.float32),
            "actions": np.tanh(rng.standard_normal((B, act))).astype(np.float32),  # strictly in (-1,1)
            "next_observations": rng.standard_normal((B, obs), np.float32),
            "rewards": np.zeros((B, 1), np.float32), "terminals": np.zeros((B,), np.float32)}


def test_awr_on_fb_builds_with_policy_baseline():
    # fb has no V; AWRActor now falls back to a policy-baseline advantage, so the pair builds.
    agent = Agent.create("fb", "awr", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg("awr"))
    assert agent is not None


@pytest.mark.parametrize("actor_kind", ["ddpgbc", "fql"])
def test_valid_pair_smoke_trains(actor_kind):
    agent = Agent.create("fb", actor_kind, 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg(actor_kind))
    for _ in range(3):
        agent, info = agent.update(_batch())
    assert jnp.isfinite(jnp.asarray(info["fb_loss"])).all()
    assert jnp.isfinite(jnp.asarray(info["actor_loss"])).all()
