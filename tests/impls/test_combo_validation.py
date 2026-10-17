import pytest

from impls.agents import Agent
from impls.critics.base import ComboError


def _min_cfg():
    return dict(
        seed=0,
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
        actor=dict(
            type="flow",
            hidden_dim=32,
            hidden_layers=2,
            embedding_layers=2,
            bc_coeff=3.0,
            flow_steps=10,
            lr_actor_vf=3e-4,
            flow_vf_hidden_dim=32,
            flow_vf_hidden_layers=4,
        ),
    )


def test_flowbc_on_fb_is_valid():
    import jax.numpy as jnp

    ex_obs = jnp.zeros((2, 8))
    ex_act = jnp.zeros((2, 5))
    agent = Agent.create("fb", "flowbc", 0, ex_obs, ex_act, _min_cfg())
    assert set(agent.net) == {
        "forward",
        "backward",
        "left_encoder",
        "actor",
        "actor_vf",
    }


def test_actor_requiring_a_missing_cond_key_raises(monkeypatch):
    from impls import agents as ag

    class _GoalOnly:
        GROUPS = ("actor",)
        REQUIRES_COND = frozenset({"goal"})

        @classmethod
        def build(cls, o, a, c):
            return cls(), {"actor": {}}

    monkeypatch.setitem(ag.ACTORS, "goal_stub", _GoalOnly)
    import jax.numpy as jnp

    with pytest.raises(ComboError, match="requires cond key"):
        ag.Agent.create(
            "fb", "goal_stub", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _min_cfg()
        )
