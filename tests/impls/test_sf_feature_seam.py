"""Seam tests for SfFeatureCritic: construction, protocol surface, the DECISIVE structural
locks (there is no external numerical reference for the SF vector-TD loss — see the module
docstring), the build-time asserts that replace the bridge's silent coercions (S4/S11 + the
sf_feature_td guard block), and combo validity.

The structural locks are the point of this file:
  * the gamma=0 decomposition L_SF == mean((phi - b(s))^2) — a per-sample regression a B×B
    measure matrix could never equal — for both the identity and frozen bases;
  * the exact-zero gradient of a FIXED (identity/frozen) base, and the presence of gradient on
    a LEARNED base (reward-regression / joint-psi);
  * the additive reward-regression split loss == sf_coef*(td + sf_reward_coef*rew);
  * the boot["next_action"] wiring (mutating it moves the loss);
and the S4 correction: a conflicting actor_cond RAISES, never coerces.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS
from impls.critics.base import ComboError
from impls.critics.sf_feature import SfFeatureCritic, SfFeatureConfigError

B, OBS, ACT, D = 8, 6, 3, 5


def _cfg(**over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, tau=0.005,
        value_hidden_dims=[32, 32], layer_norm=True, seed=0,
        value_latent_dim=D, sf_base="frozen", sf_mask="goal", sf_base_index="current",
        sf_joint_psi=False, sf_reward_regress=False, sf_reward_coef=1.0, sf_reward_shift=0.0,
        sf_ortho_coef=0.0, sf_psi_ortho_coef=0.0, sf_coef=1.0, fb_forward_z=False,
        actor_cond="goal",
    )
    cfg.update(over)
    return cfg


def _actor_block(kind):
    blk = dict(hidden_dims=[32, 32], layer_norm=False, const_std=True, alpha=3.0,
               lr_actor_vf=3e-4, flow_steps=4, hidden_dim=32, hidden_layers=2,
               embedding_layers=2, bc_coeff=3.0, flow_vf_hidden_dim=32, flow_vf_hidden_layers=4,
               normalize_q_loss=False, z_dim=D)
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


def _boot(b):
    return {"next_action": b["actions"]}


# ---------------------------------------------------------------- registration / protocol
def test_registered():
    assert CRITICS["sf_feature"] is SfFeatureCritic


def test_protocol_surface():
    ex_obs, ex_act = _ex()
    c, params, targets = SfFeatureCritic.build(ex_obs, ex_act, _cfg())
    assert c.PROVIDES_V is False               # SF-only: no value function
    assert c.PROVIDES_COND == frozenset({"value_goal", "goal"})
    assert c.GROUPS == ("critic",) and c.STAGES == ("sf",)
    assert set(params) == {"critic"} and set(targets) == {"critic"}
    assert c.TARGETS == {"critic": 0.005}
    assert c.cvec_dim == OBS                    # actor conditions on the raw goal (pinned)


# ---------------------------------------------------------------- DECISIVE structural locks
@pytest.mark.parametrize("sf_base,d", [("identity", OBS), ("frozen", D)])
def test_gamma0_is_feature_td_not_measure(sf_base, d):
    """gamma=0 collapses the target to b(s), so L_SF == mean((phi - b(s))^2): a per-sample
    vector regression, impossible for a B×B measure matrix. THE signature of the SF objective."""
    ex_obs, ex_act = _ex()
    c, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base=sf_base, value_latent_dim=d,
                                                          discount=0.0))
    b = _batch()
    loss, info = c.stage_loss("sf", p, t, b, _cond(b), _boot(b))
    _, phi, _ = c._critic(p["critic"], b["observations"], b["value_goals"], b["actions"], None)
    if sf_base == "identity":
        base = b["observations"][None]                                     # b(s)=s
    else:
        _, _, base = c._critic(p["critic"], b["observations"], b["observations"],
                               b["actions"], None)                          # b(s)=psi(s)
    expected = jnp.mean((phi - base) ** 2)
    np.testing.assert_allclose(float(loss), float(expected), atol=1e-9)
    np.testing.assert_allclose(float(info["sf_loss"]), float(expected), atol=1e-9)


def test_frozen_base_has_exactly_zero_gradient():
    """The frozen base is not trained by the SF TD (stop_gradient), so grad(psi)==0 EXACTLY —
    the structural guarantee that stands in for a numerical reference. phi still trains."""
    ex_obs, ex_act = _ex()
    c, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="frozen"))
    b = _batch()
    g = jax.grad(lambda pp: c.stage_loss("sf", pp, t, b, _cond(b), _boot(b))[0])(p)
    for leaf in jax.tree_util.tree_leaves(g["critic"]["psi"]):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))
    assert any(np.abs(np.asarray(l)).sum() > 0
               for l in jax.tree_util.tree_leaves(g["critic"]["phi"]))


def test_identity_base_has_no_psi_params():
    ex_obs, ex_act = _ex()
    c, p, _ = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="identity", value_latent_dim=OBS))
    assert "psi" not in p["critic"] and "phi" in p["critic"]


@pytest.mark.parametrize("over", [
    dict(sf_base="learned", sf_reward_regress=True),                                  # reward path
    dict(sf_base="learned", sf_joint_psi=True, sf_psi_ortho_coef=1.0, sf_mask="none"),  # joint path
])
def test_learned_base_psi_receives_gradient(over):
    """The learned base's ONLY grad paths are reward regression and joint-psi. On either, psi
    must receive gradient (the exact opposite of the frozen base)."""
    ex_obs, ex_act = _ex()
    c, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg(**over))
    b = _batch()
    g = jax.grad(lambda pp: c.stage_loss("sf", pp, t, b, _cond(b), _boot(b))[0])(p)
    assert any(np.abs(np.asarray(l)).sum() > 0
               for l in jax.tree_util.tree_leaves(g["critic"]["psi"]))


def test_reward_regress_additive_decomposition():
    """loss == sf_coef*(td + sf_reward_coef*rew): the reward regression is a clean additive
    term, and info reports both pieces. Turning it on must strictly change the loss."""
    ex_obs, ex_act = _ex()
    b = _batch()
    c0, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="frozen"))
    l0, _ = c0.stage_loss("sf", p, t, b, _cond(b), _boot(b))     # td only, same params
    c1, _, _ = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="learned",
                                                          sf_reward_regress=True, sf_reward_coef=2.0))
    l1, i1 = c1.stage_loss("sf", p, t, b, _cond(b), _boot(b))    # reuse frozen params (same tree)
    # td term is identical (same phi/base params); the difference is exactly coef*rew.
    np.testing.assert_allclose(float(l1), float(i1["sf_loss"]) + 2.0 * float(i1["sf_reward_loss"]),
                               rtol=1e-10)
    np.testing.assert_allclose(float(i1["sf_loss"]), float(l0), rtol=1e-10)


@pytest.mark.parametrize("knob,info_key", [("sf_ortho_coef", "orth"),
                                           ("sf_psi_ortho_coef", "psi_ortho")])
def test_ortho_terms_are_additive_and_gated(knob, info_key):
    ex_obs, ex_act = _ex()
    b = _batch()
    c0, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg())
    l0, i0 = c0.stage_loss("sf", p, t, b, _cond(b), _boot(b))
    assert info_key not in i0
    c1, _, _ = SfFeatureCritic.build(ex_obs, ex_act, _cfg(**{knob: 1.0}))
    l1, i1 = c1.stage_loss("sf", p, t, b, _cond(b), _boot(b))
    assert info_key in i1
    np.testing.assert_allclose(float(l1), float(l0) + float(i1[info_key]), rtol=1e-9)


def test_bootstrap_action_is_wired():
    """The a'~pi bootstrap arrives via boot['next_action']; perturbing it must move the loss
    (with gamma>0), proving the seam bootstrap is actually consumed."""
    ex_obs, ex_act = _ex()
    b = _batch()
    c, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg())
    l_a, _ = c.stage_loss("sf", p, t, b, _cond(b), {"next_action": b["actions"]})
    l_b, _ = c.stage_loss("sf", p, t, b, _cond(b), {"next_action": -b["actions"]})
    assert not np.allclose(float(l_a), float(l_b))


def test_sf_mask_changes_loss():
    ex_obs, ex_act = _ex()
    b = _batch()
    cg, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_mask="goal"))
    cn, _, _ = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_mask="none", value_latent_dim=D))
    lg, _ = cg.stage_loss("sf", p, t, b, _cond(b), _boot(b))
    ln, _ = cn.stage_loss("sf", p, t, b, _cond(b), _boot(b))
    assert not np.allclose(float(lg), float(ln))


def test_fb_forward_z_z_is_stopgrad_into_phi():
    """Under fb_forward_z the z fed to phi is stop-gradded, so on a FROZEN base (whose psi is
    already frozen) psi still has exactly zero gradient — the z path opens no back door."""
    ex_obs, ex_act = _ex()
    c, p, t = SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="frozen", fb_forward_z=True))
    b = _batch()
    g = jax.grad(lambda pp: c.stage_loss("sf", pp, t, b, _cond(b), _boot(b))[0])(p)
    for leaf in jax.tree_util.tree_leaves(g["critic"]["psi"]):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))


# ---------------------------------------------------------------- build-time asserts (S4/S11)
def test_actor_cond_conflict_raises_not_coerces():
    """THE S4 correction: the bridge silently sets actor_cond='goal'; we RAISE on a conflict."""
    ex_obs, ex_act = _ex()
    with pytest.raises(SfFeatureConfigError, match="pins actor conditioning to 'goal'"):
        SfFeatureCritic.build(ex_obs, ex_act, _cfg(actor_cond="z"))
    SfFeatureCritic.build(ex_obs, ex_act, _cfg(actor_cond="goal"))   # the pinned value builds


def test_identity_base_pins_latent_dim():
    ex_obs, ex_act = _ex()
    with pytest.raises(SfFeatureConfigError, match="pins value_latent_dim"):
        SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="identity", value_latent_dim=D))
    SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="identity", value_latent_dim=OBS))


@pytest.mark.parametrize("base", ["identity", "frozen"])
def test_reward_regress_requires_learned_base(base):
    ex_obs, ex_act = _ex()
    d = OBS if base == "identity" else D
    with pytest.raises(SfFeatureConfigError, match="sf_reward_regress requires sf_base='learned'"):
        SfFeatureCritic.build(ex_obs, ex_act,
                              _cfg(sf_base=base, value_latent_dim=d, sf_reward_regress=True))


def test_learned_base_requires_a_grad_path():
    ex_obs, ex_act = _ex()
    with pytest.raises(SfFeatureConfigError, match="sf_base='learned' requires"):
        SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base="learned"))


@pytest.mark.parametrize("over,match", [
    (dict(sf_base="frozen", sf_joint_psi=True, sf_psi_ortho_coef=1.0),
     "sf_joint_psi requires sf_base='learned'"),
    (dict(sf_base="learned", sf_joint_psi=True, sf_reward_regress=True, sf_psi_ortho_coef=1.0),
     "mutually exclusive"),
    (dict(sf_base="learned", sf_joint_psi=True, sf_psi_ortho_coef=0.0),
     "requires sf_psi_ortho_coef>0"),
    (dict(sf_base="learned", sf_joint_psi=True, sf_psi_ortho_coef=1.0, sf_mask="goal"),
     "requires fb_forward_z"),
])
def test_joint_psi_guards(over, match):
    ex_obs, ex_act = _ex()
    with pytest.raises(SfFeatureConfigError, match=match):
        SfFeatureCritic.build(ex_obs, ex_act, _cfg(**over))


def test_base_index_next_requires_no_goal_mask():
    ex_obs, ex_act = _ex()
    with pytest.raises(SfFeatureConfigError, match="sf_base_index='next' requires sf_mask='none'"):
        SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base_index="next", sf_mask="goal"))
    SfFeatureCritic.build(ex_obs, ex_act, _cfg(sf_base_index="next", sf_mask="none"))


@pytest.mark.parametrize("knob,bad", [("sf_base", "random"), ("sf_mask", "all"),
                                      ("sf_base_index", "prev")])
def test_unknown_enum_values_rejected(knob, bad):
    ex_obs, ex_act = _ex()
    with pytest.raises(SfFeatureConfigError, match="must be one of"):
        SfFeatureCritic.build(ex_obs, ex_act, _cfg(**{knob: bad}))


# ---------------------------------------------------------------- combos + smoke train
@pytest.mark.parametrize("actor", ["ddpgbc", "fql"])
def test_valid_combos_construct(actor):
    ex_obs, ex_act = _ex()
    agent = Agent.create("sf_feature", actor, 0, ex_obs, ex_act, _full_cfg(actor))
    assert agent is not None


@pytest.mark.parametrize("actor", ["flowbc", "awr"])
def test_flowbc_and_awr_now_build(actor):
    """Both now compose: flowbc conditions on SfFeature's cond_vec (goal); awr uses a
    policy-baseline advantage since SfFeature is SF-only (PROVIDES_V=False)."""
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg = load_config("sf_feature", actor, "cube_single")
    assert Agent.create("sf_feature", actor, 0, ex_obs, ex_act, cfg) is not None


@pytest.mark.parametrize("over", [
    dict(),                                                                     # frozen baseline
    dict(sf_base="identity", value_latent_dim=OBS),
    dict(sf_base="learned", sf_reward_regress=True, sf_reward_shift=1.0),
    dict(sf_base="learned", sf_joint_psi=True, sf_psi_ortho_coef=1.0, sf_mask="none"),
    dict(fb_forward_z=True),
    dict(sf_ortho_coef=1.0),
    dict(sf_base_index="next", sf_mask="none"),
])
def test_three_step_smoke_train(over):
    ex_obs, ex_act = _ex()
    agent = Agent.create("sf_feature", "ddpgbc", 0, ex_obs, ex_act, _full_cfg("ddpgbc", **over))
    b = _batch()
    for _ in range(3):
        agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite under {over}"
