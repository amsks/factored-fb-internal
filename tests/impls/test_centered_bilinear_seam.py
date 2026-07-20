"""Seam tests for CenteredBilinearCritic: construction per mode, the baseline-centering
decomposition (Q = U + A_bil, the one exact algebraic invariant this first-party critic has),
the centered-measure stop-gradient structure, the S11 pinned-value asserts, the S13
next-actions build-time + runtime checks, actor pairing, and finite losses (incl. the
scale-anchor path) — a smoke train per mode.

There is no external numerical reference for the dueling/centered losses (parity scout:
Axis-3 at best), so this file locks STRUCTURE and INVARIANTS, not numbers.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS, _LR_KEY
from impls.critics.base import ComboError
from impls.critics.centered_bilinear import CenteredBilinearCritic, CenteredBilinearConfigError
from impls.utils.config_utils import REQUIRED_KEYS, load_config

B, OBS, ACT, D = 8, 6, 3, 4


def _cfg(**over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, expectile=0.9, tau=0.005,
        value_hidden_dims=[32, 32], layer_norm=True, seed=0,
        value_latent_dim=D, double_q=True, mode="dueling",
        dueling_b_coef=1.0, centered_measure_coef=1.0, center_b_coef=1.0, center_b_samples=1,
        centered_split_bootstrap=True, centered_scale_anchors=False, use_next_actions=False,
    )
    cfg.update(over)
    return cfg


def _actor_block(kind):
    blk = dict(hidden_dims=[32, 32], layer_norm=False, const_std=True, alpha=3.0,
               lr_actor_vf=3e-4, flow_steps=4, z_dim=D)
    if kind == "fql":
        blk.update(alpha=10.0, normalize_q_loss=False)
    return blk


def _full_cfg(actor="fql", **over):
    cfg = _cfg(**over)
    cfg["actor"] = _actor_block(actor)
    return cfg


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((B, OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))))


def _batch(with_next_actions=False, seed=1):
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.standard_normal((B, OBS)))
    b = dict(
        observations=jnp.asarray(rng.standard_normal((B, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((B, OBS))),
        value_goals=g, actor_goals=g,
        rewards=jnp.asarray(rng.integers(0, 2, size=B) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=B).astype(float)),
    )
    if with_next_actions:
        b["next_actions"] = jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))
    return b


def _cond(b):
    return {"value_goal": b["value_goals"], "goal": b["actor_goals"]}


def _boot(b, seed=2):
    rng = np.random.default_rng(seed)
    return {"next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))}


# ------------------------------------------------------------- registration / protocol
def test_registered_and_lr_key():
    assert CRITICS["centered_bilinear"] is CenteredBilinearCritic
    assert _LR_KEY["center_b"] == "lr"


def test_protocol_surface():
    ex_obs, ex_act = _ex()
    c, params, targets = CenteredBilinearCritic.build(ex_obs, ex_act, _cfg())
    assert c.PROVIDES_V is False               # value net is internal, not actor-facing
    assert c.PROVIDES_COND == frozenset({"value_goal", "goal"})
    assert c.STAGES == ("centered",)
    assert c.cvec_dim == OBS                    # conditions on the raw goal, no z
    assert c.TARGETS == {"critic": 0.005}
    assert not hasattr(c, "v_fn")               # PROVIDES_V=False => no actor-facing V


@pytest.mark.parametrize("mode,groups,tgt", [
    ("dueling", ("value", "critic"), {"value", "critic"}),
    ("centered_aux", ("value", "critic", "center_b"), {"value", "critic", "center_b"}),
    ("centered_only", ("critic", "center_b"), {"critic", "center_b"}),
])
def test_groups_and_params_per_mode(mode, groups, tgt):
    ex_obs, ex_act = _ex()
    over = dict(use_next_actions=True) if mode == "centered_only" else {}
    c, params, targets = CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode=mode, **over))
    assert c.GROUPS == groups
    assert set(params) == tgt
    assert set(targets) == {"critic"}          # only the critic is polyak-tracked
    if mode == "centered_only":
        assert c.value is None


# ------------------------------------------------------------- the centering invariant
def test_dueling_Q_decomposes_into_U_plus_A_bil_exactly():
    """The one exact algebraic property: Q = U + A_bil with A_bil = (phi - b)·psi/sqrt(d).
    This is the baseline-centering the critic is built around."""
    from impls.utils.networks import GCDuelingValue
    ex_obs, ex_act = _ex()
    c, params, _ = CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="dueling"))
    b = _batch()
    out = c.critic.apply({"params": params["critic"]}, b["observations"], b["actor_goals"],
                         b["actions"], full_info=True)
    # Q = U + A_bil
    np.testing.assert_allclose(np.asarray(out["Q"]),
                               np.asarray(out["U"] + out["A_bil"]), atol=1e-12)
    # A_bil is literally the recentred bilinear score (phi - b)·psi/sqrt(d).
    recomputed = ((out["phi"] - out["bs"]) * out["psi"] / np.sqrt(D)).sum(-1)
    np.testing.assert_allclose(np.asarray(out["A_bil"]), np.asarray(recomputed), atol=1e-12)


def test_dueling_actor_q_term_is_A_bil_only_not_full_Q():
    """The actor must maximise A_bil (the part it can move), NOT U+A_bil — otherwise it
    chases U(s,g), a state-goal cardinal it cannot influence through the action."""
    ex_obs, ex_act = _ex()
    c, params, _ = CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="dueling"))
    b = _batch()
    out = c.critic.apply({"params": params["critic"]}, b["observations"], b["actor_goals"],
                         b["actions"], full_info=True)
    q, qs = c.q_fn(params, b["observations"], _cond(b), b["actions"])
    a1, a2 = out["A_bil"]
    np.testing.assert_allclose(np.asarray(q), np.minimum(np.asarray(a1), np.asarray(a2)), atol=1e-12)
    # ...and it is genuinely NOT the full-Q min (U is non-zero, so the two differ).
    full_min = np.minimum(*[np.asarray(x) for x in out["Q"]])
    assert not np.allclose(np.asarray(q), full_min)


def test_centered_q_fn_is_full_bilinear_Q():
    ex_obs, ex_act = _ex()
    c, params, _ = CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="centered_aux"))
    b = _batch()
    q1, q2 = c.critic.apply({"params": params["critic"]}, b["observations"], b["actor_goals"],
                            b["actions"])
    q, _ = c.q_fn(params, b["observations"], _cond(b), b["actions"])
    np.testing.assert_allclose(np.asarray(q), np.minimum(np.asarray(q1), np.asarray(q2)), atol=1e-12)


# ------------------------------------------------------------- gradient / stop-grad structure
def test_dueling_b_regression_trains_b():
    ex_obs, ex_act = _ex()
    c, params, targets = CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="dueling"))
    b = _batch()
    g = jax.grad(lambda p: c.stage_loss("centered", p, targets, b, _cond(b), None)[0])(params)
    # b lives inside the dueling critic module; its regression must send it real gradient.
    b_leaves = jax.tree_util.tree_leaves(g["critic"]["b"])
    assert any(np.abs(np.asarray(l)).sum() > 0 for l in b_leaves)


@pytest.mark.parametrize("mode", ["centered_aux", "centered_only"])
def test_centered_b_trained_only_by_its_regression(mode):
    """center_b_coef=0 => the ΔM/base legs send NO gradient into center_b (b is stop-gradded
    on both TD sides); center_b_coef>0 => the regression trains it. Mirrors the reference
    T10 stop-gradient contract."""
    ex_obs, ex_act = _ex()
    over = dict(centered_split_bootstrap=False) if mode == "centered_only" else {}
    b = _batch()
    boot = _boot(b)

    def grad_center_b(coef):
        c, params, targets = CenteredBilinearCritic.build(
            ex_obs, ex_act, _cfg(mode=mode, center_b_coef=coef, **over))
        g = jax.grad(lambda p: c.stage_loss("centered", p, targets, b, _cond(b), boot)[0])(params)
        return max(float(jnp.max(jnp.abs(l))) for l in jax.tree_util.tree_leaves(g["center_b"]))

    assert grad_center_b(0.0) == 0.0
    assert grad_center_b(1.0) > 0.0


def test_centered_only_value_group_absent():
    """T10-b has no scalar TD: there is no value net at all (so it cannot receive gradient)."""
    ex_obs, ex_act = _ex()
    c, params, _ = CenteredBilinearCritic.build(
        ex_obs, ex_act, _cfg(mode="centered_only", centered_split_bootstrap=False))
    assert "value" not in params and c.value is None


# ------------------------------------------------------------- finite losses (scale control)
@pytest.mark.parametrize("mode", ["dueling", "centered_aux", "centered_only"])
@pytest.mark.parametrize("anchors", [False, True])
def test_stage_loss_finite(mode, anchors):
    ex_obs, ex_act = _ex()
    if mode == "dueling" and anchors:
        pytest.skip("centered_scale_anchors is a centered-mode knob (rejected under dueling)")
    over = {"centered_scale_anchors": anchors}
    if mode == "centered_only":
        over["centered_split_bootstrap"] = False   # pi bootstrap: no next_actions needed
    c, params, targets = CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode=mode, **over))
    b = _batch()
    loss, info = c.stage_loss("centered", params, targets, b, _cond(b), _boot(b))
    assert jnp.isfinite(loss)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite ({mode}, anchors={anchors})"


# ------------------------------------------------------------- S13: next-actions is a config key
def test_s13_centered_only_split_requires_use_next_actions_at_build():
    ex_obs, ex_act = _ex()
    with pytest.raises(CenteredBilinearConfigError, match="use_next_actions"):
        CenteredBilinearCritic.build(ex_obs, ex_act,
                                     _cfg(mode="centered_only", centered_split_bootstrap=True,
                                          use_next_actions=False))
    # granting the key builds fine
    CenteredBilinearCritic.build(ex_obs, ex_act,
                                 _cfg(mode="centered_only", centered_split_bootstrap=True,
                                      use_next_actions=True))


def test_s13_runtime_refuses_batch_without_next_actions():
    """use_next_actions=True but the batch does not actually carry next_actions => loud
    failure, not a silent wrong bootstrap."""
    ex_obs, ex_act = _ex()
    c, params, targets = CenteredBilinearCritic.build(
        ex_obs, ex_act, _cfg(mode="centered_only", centered_split_bootstrap=True,
                             use_next_actions=True))
    b = _batch(with_next_actions=False)
    # FIX 4: an explicit named raise (not a bare `assert`, which python -O would strip),
    # so the runtime invariant holds regardless of optimization level.
    with pytest.raises(CenteredBilinearConfigError, match="next_actions"):
        c.stage_loss("centered", params, targets, b, _cond(b), _boot(b))
    # with next_actions present it runs and is finite.
    bn = _batch(with_next_actions=True)
    loss, _ = c.stage_loss("centered", params, targets, bn, _cond(bn), _boot(bn))
    assert jnp.isfinite(loss)


def test_use_next_actions_inert_elsewhere_raises():
    ex_obs, ex_act = _ex()
    # inert under centered_aux (nothing reads next_actions there)
    with pytest.raises(CenteredBilinearConfigError, match="inert"):
        CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="centered_aux", use_next_actions=True))
    # inert under centered_only WITHOUT split (pi bootstrap reads no next_actions)
    with pytest.raises(CenteredBilinearConfigError, match="inert"):
        CenteredBilinearCritic.build(ex_obs, ex_act,
                                     _cfg(mode="centered_only", centered_split_bootstrap=False,
                                          use_next_actions=True))


# ------------------------------------------------------------- S11 pinned values RAISE
@pytest.mark.parametrize("over,msg", [
    (dict(actor_cond="z"), "actor_cond='goal'"),
    (dict(actor_vf_cond="goal"), "actor_vf_cond='none'"),
    (dict(actor_z_infer=True), "actor_z_infer"),
])
def test_s11_pins_raise(over, msg):
    ex_obs, ex_act = _ex()
    with pytest.raises(CenteredBilinearConfigError, match=msg):
        CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(**over))


def test_center_b_samples_pinned_to_one():
    ex_obs, ex_act = _ex()
    with pytest.raises(CenteredBilinearConfigError, match="center_b_samples must be 1"):
        CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(center_b_samples=8))


def test_unknown_mode_rejected():
    ex_obs, ex_act = _ex()
    with pytest.raises(CenteredBilinearConfigError, match="mode must be one of"):
        CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="nope"))


def test_dueling_rejects_centered_only_knobs():
    ex_obs, ex_act = _ex()
    with pytest.raises(CenteredBilinearConfigError, match="inert under mode='dueling'"):
        CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="dueling", centered_scale_anchors=True))
    with pytest.raises(CenteredBilinearConfigError, match="inert under mode='dueling'"):
        CenteredBilinearCritic.build(ex_obs, ex_act, _cfg(mode="dueling", use_next_actions=True))


# ------------------------------------------------------------- actor pairing + smoke train
def test_pairs_with_fql_and_ddpgbc_rejects_flowbc_and_awr():
    ex_obs, ex_act = _ex()
    for actor in ("fql", "ddpgbc"):
        assert Agent.create("centered_bilinear", actor, 0, ex_obs, ex_act,
                            _full_cfg(actor)) is not None
    with pytest.raises(ComboError, match="z"):            # flowbc needs cond["z"]
        Agent.create("centered_bilinear", "flowbc", 0, ex_obs, ex_act, _full_cfg("flowbc"))
    with pytest.raises(ComboError, match="V function"):   # awr needs a V
        Agent.create("centered_bilinear", "awr", 0, ex_obs, ex_act, _full_cfg("awr"))


@pytest.mark.parametrize("mode", ["dueling", "centered_aux", "centered_only"])
@pytest.mark.parametrize("actor", ["fql", "ddpgbc"])
def test_smoke_train_per_mode(mode, actor):
    ex_obs, ex_act = _ex()
    over = {}
    with_na = False
    if mode == "centered_only":
        over = dict(centered_split_bootstrap=True, use_next_actions=True)
        with_na = True
    agent = Agent.create("centered_bilinear", actor, 0, ex_obs, ex_act, _full_cfg(actor, mode=mode, **over))
    info = None
    for s in range(3):
        agent, info = agent.update(_batch(with_next_actions=with_na, seed=s))
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite ({mode}, {actor})"


# ------------------------------------------------------------- config composition
def test_config_composes_and_required_keys_complete():
    cfg = load_config("centered_bilinear", "fql", "cube_single")
    for key in REQUIRED_KEYS["centered_bilinear"]:
        node, present = cfg, True
        for part in key.split("."):
            assert isinstance(node, dict) and part in node, key
            node = node[part]
        assert node is not None, key
    assert cfg["mode"] == "dueling" and cfg["center_b_samples"] == 1
