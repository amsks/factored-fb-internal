"""Seam tests for IqlBilinearCritic: construction, protocol surface, a smoke train, the
build-time asserts that replace the reference's silent coercions, and combo validity.

The build-time asserts are the point of this file. Each one corresponds to a place where
tools/wandb_mode_shim/gciql_flow.py silently overrides or silently ignores a knob (audit
S2/S3/S6/S12); here it must raise IqlBilinearConfigError with a message that names the knob.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS
from impls.critics.base import ComboError
from impls.critics.iql_bilinear import IqlBilinearCritic, IqlBilinearConfigError

B, OBS, ACT, D = 8, 6, 3, 4


def _cfg(**over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, expectile=0.9, tau=0.005,
        value_hidden_dims=[32, 32], layer_norm=True, seed=0,
        value_latent_dim=D, psi_base="learned", project_z=False, double_q=True,
        iql_ortho_coef=0.0, actor_cond="goal", actor_z_from_critic=False,
        actor_z_infer=False,
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


# ---------------------------------------------------------------- registration / protocol
def test_registered():
    assert CRITICS["iql_bilinear"] is IqlBilinearCritic


def test_protocol_surface():
    ex_obs, ex_act = _ex()
    c, params, targets = IqlBilinearCritic.build(ex_obs, ex_act, _cfg())
    assert c.PROVIDES_V is True
    assert c.PROVIDES_COND == frozenset({"value_goal", "goal"})
    assert c.GROUPS == ("value", "critic") and c.STAGES == ("iql",)
    assert set(params) == {"value", "critic"} and set(targets) == {"critic"}
    assert c.TARGETS == {"critic": 0.005}


def test_cvec_dim_follows_actor_cond():
    ex_obs, ex_act = _ex()
    goal_c, _, _ = IqlBilinearCritic.build(ex_obs, ex_act, _cfg())
    assert goal_c.cvec_dim == OBS
    z_c, _, _ = IqlBilinearCritic.build(ex_obs, ex_act, _cfg(actor_cond="z"))
    assert z_c.cvec_dim == D
    # actor_z_infer alone also moves the actor into psi-space (reference L1513).
    inf_c, _, _ = IqlBilinearCritic.build(ex_obs, ex_act, _cfg(actor_z_infer=True))
    assert inf_c.cvec_dim == D


@pytest.mark.parametrize("psi_base,d", [("learned", D), ("frozen", D), ("identity", OBS)])
def test_psi_base_variants_build_and_produce_finite_losses(psi_base, d):
    ex_obs, ex_act = _ex()
    c, params, targets = IqlBilinearCritic.build(
        ex_obs, ex_act, _cfg(psi_base=psi_base, value_latent_dim=d))
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    loss, info = c.stage_loss("iql", params, targets, b, cond, boot=None)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(info["value_loss"]) and jnp.isfinite(info["critic_loss"])
    # identity has no psi submodule at all; the other two do.
    assert ("psi" in params["critic"]) == (psi_base != "identity")


def test_frozen_psi_has_exactly_zero_gradient():
    """psi_base='frozen' must be frozen by CONSTRUCTION (stop_gradient), not by an optimizer
    mask. There is no numerical reference for this, so it is asserted structurally."""
    ex_obs, ex_act = _ex()
    c, params, targets = IqlBilinearCritic.build(ex_obs, ex_act, _cfg(psi_base="frozen"))
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    g = jax.grad(lambda p: c.stage_loss("iql", p, targets, b, cond, None)[0])(params)
    for leaf in jax.tree_util.tree_leaves(g["critic"]["psi"]):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))
    # ...while phi very much does receive gradient (guards against a trivially-zero grad).
    assert any(np.abs(np.asarray(l)).sum() > 0 for l in jax.tree_util.tree_leaves(g["critic"]["phi"]))


def test_learned_psi_does_receive_gradient():
    ex_obs, ex_act = _ex()
    c, params, targets = IqlBilinearCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    g = jax.grad(lambda p: c.stage_loss("iql", p, targets, b, cond, None)[0])(params)
    assert any(np.abs(np.asarray(l)).sum() > 0 for l in jax.tree_util.tree_leaves(g["critic"]["psi"]))


def test_project_z_reaches_both_nets():
    """Audit S3: the reference passes project_z only to the critic, so the actor conditions
    on an unsphered value psi while the critic trains a sphered one. Ours is symmetric."""
    ex_obs, ex_act = _ex()
    c, params, _ = IqlBilinearCritic.build(ex_obs, ex_act, _cfg(project_z=True))
    from impls.utils.networks import GCBilinearPsiValue
    assert c.value.project_z is True and c.critic.project_z is True
    goals = _batch()["actor_goals"]
    v_psi = c._psi_value(params["value"], goals)                                     # [B,d]
    c_psi = c.critic.apply({"params": params["critic"]}, goals,
                           method=GCBilinearPsiValue.psi_of)                         # [2,B,d]
    for psi in (v_psi, c_psi):
        np.testing.assert_allclose(np.linalg.norm(np.asarray(psi), axis=-1),
                                   np.sqrt(D), rtol=1e-5)
    # NB: _psi_critic head-MEANS the two sphered heads, so the actor's z is inside the
    # sphere, not on it (norm <= sqrt(d)). Same as the reference; recorded here so it is a
    # known property rather than a surprise.
    assert (np.linalg.norm(np.asarray(c._psi_critic(params["critic"], goals)), axis=-1)
            <= np.sqrt(D) + 1e-6).all()


def test_double_q_false_uses_q1_only():
    ex_obs, ex_act = _ex()
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    c1, p, _ = IqlBilinearCritic.build(ex_obs, ex_act, _cfg(double_q=True))
    c2, _, _ = IqlBilinearCritic.build(ex_obs, ex_act, _cfg(double_q=False))
    q_min, qs = c1.q_fn(p, b["observations"], cond, b["actions"])
    q_one, _ = c2.q_fn(p, b["observations"], cond, b["actions"])
    np.testing.assert_allclose(np.asarray(q_min), np.minimum(*np.asarray(qs)), atol=1e-12)
    np.testing.assert_allclose(np.asarray(q_one), np.asarray(qs[0]), atol=1e-12)
    assert not np.allclose(np.asarray(q_one), np.asarray(q_min))


def test_ortho_term_appears_only_when_coef_positive():
    ex_obs, ex_act = _ex()
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    c0, p, t = IqlBilinearCritic.build(ex_obs, ex_act, _cfg())
    l0, i0 = c0.stage_loss("iql", p, t, b, cond, None)
    assert "ortho_loss" not in i0
    c1, p1, t1 = IqlBilinearCritic.build(ex_obs, ex_act, _cfg(iql_ortho_coef=1.0))
    l1, i1 = c1.stage_loss("iql", p1, t1, b, cond, None)
    assert "ortho_loss" in i1
    np.testing.assert_allclose(float(l1), float(l0) + float(i1["ortho_loss"]), rtol=1e-10)


def test_actor_cond_z_puts_psi_in_cond_and_cond_vec():
    ex_obs, ex_act = _ex()
    cfg = _full_cfg(actor_cond="z")
    agent = Agent.create("iql_bilinear", "ddpgbc", 0, ex_obs, ex_act, cfg)
    b = _batch()
    cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
    assert cond["z"].shape == (B, D)
    np.testing.assert_array_equal(np.asarray(agent.critic.cond_vec(cond)), np.asarray(cond["z"]))
    expected = agent.critic._psi_value({g: agent.net[g].params for g in ("value",)}["value"],
                                       b["actor_goals"])
    np.testing.assert_allclose(np.asarray(cond["z"]), np.asarray(expected), atol=1e-12)


def test_map_eval_cond_supplies_z_at_eval():
    """actor_cond='z' must be evaluable: the eval goal has to be mapped through psi, or the
    actor is fed a raw goal of the wrong width."""
    ex_obs, ex_act = _ex()
    agent = Agent.create("iql_bilinear", "ddpgbc", 0, ex_obs, ex_act, _full_cfg(actor_cond="z"))
    goals = _batch()["actor_goals"]
    mapped = agent.critic.map_eval_cond(agent.net, {"goal": goals})
    assert mapped["z"].shape == (B, D)
    acts = agent.sample_actions(_batch()["observations"], goals=goals, seed=jax.random.PRNGKey(0))
    assert acts.shape == (B, ACT) and np.isfinite(np.asarray(acts)).all()


# ---------------------------------------------------------------- build-time asserts
def test_identity_psi_base_pins_latent_dim():
    ex_obs, ex_act = _ex()
    with pytest.raises(IqlBilinearConfigError, match="pins value_latent_dim"):
        IqlBilinearCritic.build(ex_obs, ex_act, _cfg(psi_base="identity", value_latent_dim=D))
    # the pinned value builds fine
    IqlBilinearCritic.build(ex_obs, ex_act, _cfg(psi_base="identity", value_latent_dim=OBS))


def test_actor_z_from_critic_requires_actor_cond_z():
    ex_obs, ex_act = _ex()
    with pytest.raises(IqlBilinearConfigError, match="silent no-op unless actor_cond='z'"):
        IqlBilinearCritic.build(ex_obs, ex_act, _cfg(actor_z_from_critic=True))
    IqlBilinearCritic.build(ex_obs, ex_act, _cfg(actor_z_from_critic=True, actor_cond="z"))


def test_project_z_rejected_on_identity_base():
    ex_obs, ex_act = _ex()
    with pytest.raises(IqlBilinearConfigError, match="project_z=True is inert"):
        IqlBilinearCritic.build(ex_obs, ex_act,
                                _cfg(psi_base="identity", value_latent_dim=OBS, project_z=True))


@pytest.mark.parametrize("psi_base,d", [("identity", OBS), ("frozen", D)])
def test_ortho_rejected_where_it_has_no_gradient(psi_base, d):
    ex_obs, ex_act = _ex()
    with pytest.raises(IqlBilinearConfigError, match="silent no-op"):
        IqlBilinearCritic.build(ex_obs, ex_act,
                                _cfg(psi_base=psi_base, value_latent_dim=d, iql_ortho_coef=1.0))


@pytest.mark.parametrize("knob,bad", [("psi_base", "random"), ("actor_cond", "latent")])
def test_unknown_enum_values_rejected(knob, bad):
    ex_obs, ex_act = _ex()
    with pytest.raises(IqlBilinearConfigError, match="must be one of"):
        IqlBilinearCritic.build(ex_obs, ex_act, _cfg(**{knob: bad}))


# ---------------------------------------------------------------- combos + smoke train
@pytest.mark.parametrize("actor", ["awr", "ddpgbc", "fql"])
def test_valid_combos_construct(actor):
    ex_obs, ex_act = _ex()
    agent = Agent.create("iql_bilinear", actor, 0, ex_obs, ex_act, _full_cfg(actor))
    assert agent is not None


def test_flowbc_combo_rejected():
    """FlowBCActor hard-requires cond['z'] unconditionally; IqlBilinearCritic only promises
    value_goal/goal (its z exists only under actor_cond='z'), so the pair is invalid."""
    ex_obs, ex_act = _ex()
    with pytest.raises(ComboError, match="z"):
        Agent.create("iql_bilinear", "flowbc", 0, ex_obs, ex_act, _full_cfg("flowbc"))


@pytest.mark.parametrize("over", [dict(), dict(actor_cond="z"), dict(project_z=True),
                                  dict(iql_ortho_coef=1.0),
                                  dict(actor_cond="z", actor_z_from_critic=True),
                                  dict(actor_z_infer=True)])
def test_three_step_smoke_train(over):
    ex_obs, ex_act = _ex()
    agent = Agent.create("iql_bilinear", "ddpgbc", 0, ex_obs, ex_act, _full_cfg("ddpgbc", **over))
    b = _batch()
    for _ in range(3):
        agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite under {over}"
