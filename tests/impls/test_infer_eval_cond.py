"""infer_eval_cond and the no-op inference detector."""

import jax.numpy as jnp
import numpy as np

import impls.main as m
from impls.agents import Agent


def _fb_cfg():
    return dict(
        batch_size=8,
        z_dim=50,
        L_dim=50,
        num_parallel=2,
        discount=0.99,
        f_target_tau=0.005,
        b_target_tau=0.005,
        ortho_coef=1.0,
        train_goal_ratio=0.5,
        fb_pessimism_penalty=0.0,
        norm_z=True,
        lr_f=1e-4,
        lr_b=1e-4,
        lr_actor=1e-4,
        forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=32, hidden_layers=2, norm=True),
        left_encoder=dict(hidden_dim=32, hidden_layers=2, norm=True),
        actor=dict(
            type="flow",
            hidden_dim=32,
            hidden_layers=2,
            embedding_layers=2,
            bc_coeff=3.0,
            flow_steps=3,
            lr_actor_vf=3e-4,
            flow_vf_hidden_dim=32,
            flow_vf_hidden_layers=2,
        ),
    )


def test_infer_eval_cond_changes_z():
    """Non-degenerate rewards move the inferred z."""
    agent = Agent.create(
        "fb", "flowbc", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _fb_cfg()
    )
    before = np.asarray(agent.cond_eval["z"])
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((32, 8)).astype(np.float32)
    rew = rng.standard_normal((32, 1)).astype(np.float32)
    after = np.asarray(agent.infer_eval_cond(obs, rew).cond_eval["z"])
    assert not np.allclose(before, after), "infer_eval_cond did not change cond_eval"


def test_infer_eval_cond_is_noop_on_real_cube_single_data():
    """On cube-single play data, rewards are zero and the inferred z equals eval_cond0."""
    from impls.utils.config_utils import load_config
    from impls.utils.datasets import make_dataset
    from impls.utils.env_utils import make_env_and_datasets

    cfg = load_config("fb", "flowbc", "cube_single")
    cfg["batch_size"] = 8
    _, train_ds, _ = make_env_and_datasets(cfg["env_name"])
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(
        "fb",
        "flowbc",
        0,
        jnp.asarray(ex["observations"]),
        jnp.asarray(ex["actions"]),
        cfg,
    )

    rb = train_ds.sample(32)
    assert bool(jnp.all(jnp.asarray(rb["rewards"]) == 0)), (
        "test premise: cube-single-play-v0 rewards must be all zero"
    )

    before = agent.cond_eval
    after_agent = agent.infer_eval_cond(
        jnp.asarray(rb["observations"]), jnp.asarray(rb["rewards"])
    )
    after = after_agent.cond_eval

    np.testing.assert_array_equal(np.asarray(after["z"]), np.asarray(before["z"]))
    np.testing.assert_array_equal(
        np.asarray(after["z"]), np.zeros_like(np.asarray(after["z"]))
    )

    reason = m.infer_cond_noop_reason(before, after, rb["rewards"])
    assert reason is not None, "the runtime no-op detector must catch this exact case"


def test_infer_cond_noop_reason_detects_all_zero_rewards():
    pre = {"z": jnp.zeros((4,))}
    post = {"z": jnp.zeros((4,))}
    rewards = jnp.zeros((8, 1))
    reason = m.infer_cond_noop_reason(pre, post, rewards)
    assert reason is not None and "zero" in reason


def test_infer_cond_noop_reason_detects_unchanged_cond():
    pre = {"goal": jnp.zeros((4,))}
    post = {"goal": jnp.zeros((4,))}  # fixed placeholder
    rewards = jnp.ones((8, 1))  # non-zero rewards; only the cond is unchanged
    reason = m.infer_cond_noop_reason(pre, post, rewards)
    assert reason is not None


def test_infer_cond_noop_reason_is_none_when_cond_actually_changed():
    pre = {"z": jnp.zeros((4,))}
    post = {"z": jnp.ones((4,))}
    rewards = jnp.ones((8, 1))
    assert m.infer_cond_noop_reason(pre, post, rewards) is None
