"""Seam tests for SimpleSfCritic: construction, protocol surface, forward invariants, a smoke
train, the build-time asserts that replace the reference's silent behaviours, combo validity,
and — the point of this file — the exact-zero-gradient property of the grounded psi leg.

SimpleSf has NO external numerical reference: Chua et al.'s Simple SF is ported only through
tools/wandb_mode_shim/gciql_flow.py, and the parity scout lists no torch or vendored analogue.
So nothing here fabricates a numerical lock. What IS locked:
  * structure + protocol surface (like every seam critic),
  * forward invariants recomputed INDEPENDENTLY in numpy (readout formula, ||b||=1, project_z
    sphere) — an independent recomputation of the documented spec, not a self-snapshot,
  * the defining fact: under ground_psi the IQL TD gives psi EXACTLY ZERO gradient, and the
    reward regression is psi's only gradient path. That is the single most important test.

Fixtures exercise the scale factors the audit calls out: the critic is ensembled (2 heads)
and latent_dim = 4 != 1.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS
from impls.critics.base import ComboError
from impls.critics.simple_sf import SimpleSfCritic, SimpleSfConfigError
from impls.utils.networks import GCSimpleSFValue

B, OBS, ACT, D = 8, 6, 3, 4


def _cfg(**over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, expectile=0.9, tau=0.005,
        value_hidden_dims=[32, 32], layer_norm=True, seed=0,
        value_latent_dim=D, ground_psi=False, reward_coef=0.0, project_z=False,
        double_q=True, actor_cond="goal", ground_fade=False,
    )
    cfg.update(over)
    return cfg


def _actor_block(kind):
    blk = dict(hidden_dims=[32, 32], layer_norm=False, const_std=True, alpha=3.0,
               lr_actor_vf=3e-4, flow_steps=4, num_flow_steps=4, z_dim=D)
    if kind == "fql":
        blk.update(alpha=10.0)
    return blk


def _full_cfg(actor="ddpgbc", **over):
    cfg = _cfg(**over)
    cfg["actor"] = _actor_block(actor)
    return cfg


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((B, OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))))


def _batch():
    rng = np.random.default_rng(1)
    g = jnp.asarray(rng.standard_normal((B, OBS)))
    return dict(
        observations=jnp.asarray(rng.standard_normal((B, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((B, OBS))),
        value_goals=g, actor_goals=g,
        rewards=jnp.asarray(rng.integers(0, 2, size=B) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=B).astype(float)),
    )


def _cond(b):
    return {"value_goal": b["value_goals"], "goal": b["actor_goals"]}


# ---------------------------------------------------------------- registration / protocol
def test_registered():
    assert CRITICS["simple_sf"] is SimpleSfCritic


def test_protocol_surface():
    ex_obs, ex_act = _ex()
    c, params, targets = SimpleSfCritic.build(ex_obs, ex_act, _cfg())
    assert c.PROVIDES_V is True
    assert c.PROVIDES_COND == frozenset({"value_goal", "goal"})
    assert c.GROUPS == ("value", "critic") and c.STAGES == ("iql",)
    assert set(params) == {"value", "critic"} and set(targets) == {"critic"}
    assert c.TARGETS == {"critic": 0.005}
    assert c.cvec_dim == OBS


# ---------------------------------------------------------------- forward invariants (numpy)
@pytest.mark.parametrize("ground_psi,reward_coef", [(False, 0.0), (True, 1.0)])
def test_forward_invariants_recomputed_in_numpy(ground_psi, reward_coef):
    """Independent recomputation of the SF readout: Q=<Psi,z>/sqrt(d) and ||b||=1. stop_gradient
    is identity in the forward pass, so v is the same whether or not psi is grounded."""
    ex_obs, ex_act = _ex()
    c, params, _ = SimpleSfCritic.build(
        ex_obs, ex_act, _cfg(ground_psi=ground_psi, reward_coef=reward_coef))
    b = _batch()
    v, Psi, z, bas = c.critic.apply({"params": params["critic"]},
                                    b["observations"], b["value_goals"], b["actions"], info=True)
    Psi, z, v, bas = map(np.asarray, (Psi, z, v, bas))
    assert Psi.shape == (2, B, D) and z.shape == (B, D) and v.shape == (2, B)
    v_np = (Psi * z[None]).sum(-1) / np.sqrt(D)                       # readout, recomputed
    np.testing.assert_allclose(v, v_np, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(bas, axis=-1), 1.0, atol=2e-6)  # ||b|| = 1


def test_project_z_spheres_the_task_vector():
    ex_obs, ex_act = _ex()
    c, params, _ = SimpleSfCritic.build(ex_obs, ex_act, _cfg(project_z=True))
    z = np.asarray(c.critic.apply({"params": params["critic"]},
                                  _batch()["value_goals"], method=GCSimpleSFValue.psi_of))
    np.testing.assert_allclose(np.linalg.norm(z, axis=-1), np.sqrt(D), rtol=1e-5)


# ---------------------------------------------------------------- THE defining test
@pytest.mark.parametrize("ground_psi,reward_coef", [(False, 0.0), (True, 1.0)])
def test_ground_psi_gives_psi_exactly_zero_gradient(ground_psi, reward_coef):
    """ground_psi is the only stop_gradient-psi in the design. Under it, the IQL TD leg must
    give the psi submodule of BOTH nets EXACTLY zero gradient (bit-zero, from stop_gradient —
    not merely small); without it, psi is trained by the TD. The trunk/head must receive
    gradient in BOTH cases, so a zero psi-grad is proven to come from the stop_gradient and not
    from a trivially-dead loss."""
    ex_obs, ex_act = _ex()
    c, params, targets = SimpleSfCritic.build(
        ex_obs, ex_act, _cfg(ground_psi=ground_psi, reward_coef=reward_coef))
    b = _batch()
    g = jax.grad(lambda p: c._iql_loss(p, targets, b, _cond(b))[0])(params)

    psi_leaves = (jax.tree_util.tree_leaves(g["value"]["psi"])
                  + jax.tree_util.tree_leaves(g["critic"]["psi"]))
    psi_grad_sum = sum(float(np.abs(np.asarray(l)).sum()) for l in psi_leaves)
    if ground_psi:
        for l in psi_leaves:
            np.testing.assert_array_equal(np.asarray(l), np.zeros_like(l))
    else:
        assert psi_grad_sum > 0.0

    # the trunk (both nets) always trains from the TD — guards against a globally-dead grad.
    for net in ("value", "critic"):
        assert any(np.abs(np.asarray(l)).sum() > 0
                   for l in jax.tree_util.tree_leaves(g[net]["trunk"]))


def test_reward_loss_trains_only_psi():
    """The complement of the zero-grad test: the reward regression is what actually trains psi
    under grounding. Its gradient reaches psi (via z=psi(g)) and NOWHERE ELSE — b is detached
    and Psi/the trunk are not in the term at all, so trunk gets exactly zero."""
    ex_obs, ex_act = _ex()
    c, params, _ = SimpleSfCritic.build(ex_obs, ex_act, _cfg(ground_psi=True, reward_coef=1.0))
    b = _batch()
    g = jax.grad(lambda p: c._reward_loss(p, b, _cond(b))[0])(params)
    # psi receives gradient
    psi_leaves = (jax.tree_util.tree_leaves(g["value"]["psi"])
                  + jax.tree_util.tree_leaves(g["critic"]["psi"]))
    assert sum(float(np.abs(np.asarray(l)).sum()) for l in psi_leaves) > 0.0
    # trunk (b's only params) is stop_gradded -> exactly zero
    for net in ("value", "critic"):
        for l in jax.tree_util.tree_leaves(g[net]["trunk"]):
            np.testing.assert_array_equal(np.asarray(l), np.zeros_like(l))


# ---------------------------------------------------------------- objective wiring
def test_rung3_adds_reward_term_rung2_does_not():
    ex_obs, ex_act = _ex()
    b = _batch()
    c2, p2, t2 = SimpleSfCritic.build(ex_obs, ex_act, _cfg())                     # rung 2
    l2, i2 = c2.stage_loss("iql", p2, t2, b, _cond(b), None)
    assert "simplesf_reward_loss" not in i2
    c3, p3, t3 = SimpleSfCritic.build(ex_obs, ex_act, _cfg(ground_psi=True, reward_coef=2.0))
    l3, i3 = c3.stage_loss("iql", p3, t3, b, _cond(b), None)
    assert "simplesf_reward_loss" in i3
    iql3, _ = c3._iql_loss(p3, t3, b, _cond(b))
    np.testing.assert_allclose(float(l3), float(iql3) + 2.0 * float(i3["simplesf_reward_loss"]),
                               rtol=1e-10)


def test_double_q_false_uses_q1_only():
    ex_obs, ex_act = _ex()
    b = _batch()
    c1, p, _ = SimpleSfCritic.build(ex_obs, ex_act, _cfg(double_q=True))
    c2, _, _ = SimpleSfCritic.build(ex_obs, ex_act, _cfg(double_q=False))
    q_min, qs = c1.q_fn(p, b["observations"], _cond(b), b["actions"])
    q_one, _ = c2.q_fn(p, b["observations"], _cond(b), b["actions"])
    np.testing.assert_allclose(np.asarray(q_min), np.minimum(*np.asarray(qs)), atol=1e-12)
    np.testing.assert_allclose(np.asarray(q_one), np.asarray(qs[0]), atol=1e-12)
    assert not np.allclose(np.asarray(q_one), np.asarray(q_min))


# ---------------------------------------------------------------- build-time asserts
def test_ground_fade_is_rejected():
    ex_obs, ex_act = _ex()
    with pytest.raises(SimpleSfConfigError, match="ground_fade"):
        SimpleSfCritic.build(ex_obs, ex_act, _cfg(ground_fade=True))


def test_actor_cond_pinned_to_goal():
    ex_obs, ex_act = _ex()
    with pytest.raises(SimpleSfConfigError, match="pins actor_cond='goal'"):
        SimpleSfCritic.build(ex_obs, ex_act, _cfg(actor_cond="z"))


def test_ground_psi_without_reward_coef_freezes_psi_and_is_rejected():
    ex_obs, ex_act = _ex()
    with pytest.raises(SimpleSfConfigError, match="FREEZES psi"):
        SimpleSfCritic.build(ex_obs, ex_act, _cfg(ground_psi=True, reward_coef=0.0))


def test_reward_coef_without_ground_psi_is_rejected():
    ex_obs, ex_act = _ex()
    with pytest.raises(SimpleSfConfigError, match="inert"):
        SimpleSfCritic.build(ex_obs, ex_act, _cfg(ground_psi=False, reward_coef=1.0))


# ---------------------------------------------------------------- combos + smoke train
@pytest.mark.parametrize("actor", ["awr", "ddpgbc", "fql"])
def test_valid_combos_construct(actor):
    ex_obs, ex_act = _ex()
    agent = Agent.create("simple_sf", actor, 0, ex_obs, ex_act, _full_cfg(actor))
    assert agent is not None


def test_flowbc_combo_builds():
    """FlowBCActor now conditions on SimpleSfCritic.cond_vec (the goal), so it builds a
    goal-conditioned flow-BC actor."""
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg = load_config("simple_sf", "flowbc", "cube_single")
    assert Agent.create("simple_sf", "flowbc", 0, ex_obs, ex_act, cfg) is not None


@pytest.mark.parametrize("over", [dict(), dict(project_z=True),
                                  dict(ground_psi=True, reward_coef=1.0),
                                  dict(ground_psi=True, reward_coef=1.0, project_z=True),
                                  dict(double_q=False)])
def test_three_step_smoke_train(over):
    ex_obs, ex_act = _ex()
    agent = Agent.create("simple_sf", "ddpgbc", 0, ex_obs, ex_act, _full_cfg("ddpgbc", **over))
    b = _batch()
    for _ in range(3):
        agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite under {over}"
