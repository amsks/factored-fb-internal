import jax.numpy as jnp
import numpy as np
import pytest

from impls.agent import Agent


def _cfg(actor_kind):
    base = dict(
        z_dim=8,
        L_dim=8,
        num_parallel=2,
        discount=0.99,
        f_target_tau=0.005,
        b_target_tau=0.005,
        ortho_coef=1.0,
        train_goal_ratio=0.5,
        norm_z=True,
        lr_f=1e-4,
        lr_b=1e-4,
        lr_actor=1e-4,
        forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
    )
    if actor_kind == "ddpgbc":
        base["actor"] = dict(
            hidden_dims=[32],
            q_normalize=True,
            q_noise_std=0.2,
            q_noise_clip=0.3,
            boot_noise_std=0.2,
            boot_noise_clip=0.3,
        )
    else:
        base["actor"] = dict(
            hidden_dim=32,
            hidden_layers=2,
            embedding_layers=2,
            bc_coeff=3.0,
            flow_steps=5,
            lr_actor_vf=3e-4,
            flow_vf_hidden_dim=32,
            flow_vf_hidden_layers=2,
        )
    return base


def _batch(B=16, obs=8, act=5, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((B, obs), np.float32),
        "actions": np.tanh(rng.standard_normal((B, act))).astype(
            np.float32
        ),  # strictly in (-1,1)
        "next_observations": rng.standard_normal((B, obs), np.float32),
        "rewards": np.zeros((B, 1), np.float32),
        "terminals": np.zeros((B,), np.float32),
    }


@pytest.mark.parametrize("actor_kind", ["ddpgbc", "flowbc"])
def test_fb_with_each_actor_trains(actor_kind):
    agent = Agent.create(
        "fb", actor_kind, 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _cfg(actor_kind)
    )
    for _ in range(3):
        agent, info = agent.update(_batch())
    assert jnp.isfinite(jnp.asarray(info["fb_loss"])).all()
    assert jnp.isfinite(jnp.asarray(info["actor_loss"])).all()
    a = agent.sample_actions(jnp.asarray(_batch()["observations"][0]))
    assert a.shape == (5,) and bool((jnp.abs(a) <= 1).all())
