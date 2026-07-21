"""Seam tests for FbForwardBilinearCritic: construction, the phi_module enum, the two_tower×
z-forward COMPOSITION that unblocks B2/B4, every objective mode's smoke train, the build-time
asserts that replace the reference's silent coercions, and combo validity.

THE HEADLINE (audit S1): in the bridge reference the forward architecture is a mutually
exclusive if/elif, so fwd_two_tower alongside fb_forward_z is silently swallowed and B2/B4 are
unreachable. Here two_tower is a phi_module CHOICE on an always-z-indexed critic, so
`test_two_tower_composes_with_z_forward` constructing and training is the proof that B2/B4 are
now expressible.

VERIFICATION HONESTY: the z-index has no external reference (see the critic docstring). These
tests lock STRUCTURE — construction, invariants, the loss decomposition, and exact-zero-grad
where a leg is untrained — not numerics. The numeric transcription lock is
test_fb_forward_bilinear_static.py; the SG-ISFB objective is structural-only here (ISFBCritic
owns the verified torch-faithful version).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS, _LR_KEY
from impls.critics.base import ComboError
from impls.critics.fb_forward_bilinear import FbForwardBilinearCritic, FbForwardBilinearConfigError
from impls.utils.config_utils import REQUIRED_KEYS, load_config

B, OBS, ACT, D = 8, 6, 3, 4


def _cfg(**over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, expectile=0.9, tau=0.005,
        value_hidden_dims=[32, 32], layer_norm=True, seed=0,
        value_latent_dim=D, phi_module="flat", objective="iql", project_z=False, double_q=True,
        measure_coef=0.0, measure_legs="critic", measure_scale_norm=False,
        measure_stopgrad_phi=False, measure_ortho_coef=0.0, measure_test_states="rho",
        measure_test_rank="full", iql_ortho_coef=0.0,
        measure_insample="scalar_gate_hard", alpha_n=1.0, alpha_m=1.0, alpha_v=1.0,
        awr_beta=3.0, awr_wmax=100.0, awr_normalize_adv=True, fb_pessimism_penalty=0.0,
        goal_mask=True,
        actor_cond="goal", actor_z_from_critic=False, actor_z_infer=False,
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


def _obj_over(objective, phi_module):
    over = dict(objective=objective, phi_module=phi_module)
    if objective in ("measure", "mix"):
        over["measure_coef"] = 1.0
    return over


# ---------------------------------------------------------------- registration / protocol
def test_registered():
    assert CRITICS["fb_forward_bilinear"] is FbForwardBilinearCritic
    assert _LR_KEY["value"] == "lr" and _LR_KEY["critic"] == "lr"   # reuses the gciql groups


def test_protocol_surface():
    ex_obs, ex_act = _ex()
    c, params, targets = FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg())
    assert c.PROVIDES_V is True
    assert c.PROVIDES_COND == frozenset({"value_goal", "goal"})
    assert c.GROUPS == ("value", "critic") and c.STAGES == ("fb_forward",)
    assert set(params) == {"value", "critic"} and set(targets) == {"critic"}
    assert c.TARGETS == {"critic": 0.005}


def test_cvec_dim_follows_actor_cond():
    ex_obs, ex_act = _ex()
    assert FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg())[0].cvec_dim == OBS
    assert FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(actor_cond="z"))[0].cvec_dim == D
    assert FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(actor_z_infer=True))[0].cvec_dim == D


# ---------------------------------------------------------------- phi_module enum + composition
@pytest.mark.parametrize("phi_module,objective", [
    ("flat", "iql"), ("two_tower", "iql"), ("isfb_twin_leg", "isfb")])
def test_phi_modules_build(phi_module, objective):
    ex_obs, ex_act = _ex()
    c, params, _ = FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(**_obj_over(objective, phi_module)))
    assert c.phi_module == phi_module
    # the twin leg's F_V submodule exists iff isfb_twin_leg.
    assert ("fv" in params["critic"]) == (phi_module == "isfb_twin_leg")


def test_two_tower_submodules_present():
    """two_tower's forward leg is a real trunk(embed_sa, embed_sz) — the z enters through
    embed_sz, which is what makes it compose with the z-index rather than swallow it."""
    ex_obs, ex_act = _ex()
    _, params, _ = FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(phi_module="two_tower"))
    phi = params["critic"]["phi"]
    # ensemblized _TwoTowerPhi -> its named MLP submodules live under the tower scope.
    flat = {"/".join(map(str, k)) for k in
            __import__("flax").traverse_util.flatten_dict(phi).keys()}
    assert any("embed_sa" in k for k in flat)
    assert any("embed_sz" in k for k in flat)
    assert any("trunk" in k for k in flat)


def test_two_tower_composes_with_z_forward():
    """B2/B4 constructibility — the payoff of the S1 fix. two_tower + z-forward + sphere +
    ortho must construct AND train, which it never could in the reference (the elif swallowed
    fwd_two_tower). Here it is one coherent module."""
    ex_obs, ex_act = _ex()
    # B2: two_tower + z-forward under IQL.  B4: + project_z + ortho.
    for over in (dict(phi_module="two_tower"),
                 dict(phi_module="two_tower", project_z=True, iql_ortho_coef=1.0)):
        agent = Agent.create("fb_forward_bilinear", "ddpgbc", 0, ex_obs, ex_act,
                             _full_cfg("ddpgbc", **over))
        b = _batch()
        for _ in range(3):
            agent, info = agent.update(b)
        for k, v in info.items():
            assert np.isfinite(np.asarray(v)).all(), (over, k)


def test_project_z_reaches_both_nets():
    """Audit S3: project_z is symmetric across the value and critic nets."""
    ex_obs, ex_act = _ex()
    c, params, _ = FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(project_z=True))
    assert c.value.project_z is True and c.critic.project_z is True
    from impls.utils.networks import GCFbForwardValue
    goals = _batch()["actor_goals"]
    c_psi = c.critic.apply({"params": params["critic"]}, goals, method=GCFbForwardValue.psi_of)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(c_psi), axis=-1), np.sqrt(D), rtol=1e-5)


# ---------------------------------------------------------------- objective smoke trains
@pytest.mark.parametrize("objective,phi_module,actor", [
    ("iql", "flat", "ddpgbc"), ("iql", "two_tower", "awr"), ("iql", "flat", "fql"),
    ("measure", "flat", "ddpgbc"), ("measure", "two_tower", "ddpgbc"),
    ("mix", "flat", "awr"), ("mix", "flat", "ddpgbc"),
    ("isfb", "isfb_twin_leg", "ddpgbc"), ("isfb", "isfb_twin_leg", "awr")])
def test_smoke_train_per_objective(objective, phi_module, actor):
    ex_obs, ex_act = _ex()
    agent = Agent.create("fb_forward_bilinear", actor, 0, ex_obs, ex_act,
                         _full_cfg(actor, **_obj_over(objective, phi_module)))
    b = _batch()
    for _ in range(3):
        agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), (objective, phi_module, actor, k)


def test_isfb_coord_drops_l_v():
    """isfb.tex's boxed L_Coord-ISFB has no L_V term; coord must report it exactly 0.0."""
    ex_obs, ex_act = _ex()
    agent = Agent.create("fb_forward_bilinear", "ddpgbc", 0, ex_obs, ex_act,
                         _full_cfg("ddpgbc", objective="isfb", phi_module="isfb_twin_leg",
                                   measure_insample="coord"))
    agent, info = agent.update(_batch())
    assert float(info["isfb_L_V"]) == 0.0


def test_measure_objective_leaves_value_net_untrained():
    """Under objective='measure' the value net inits but must not train (exactly like the
    reference's measure_only). Its stage group is dropped, so it receives no gradient."""
    ex_obs, ex_act = _ex()
    c, params, targets = FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(objective="measure", measure_coef=1.0))
    assert c.stage_groups("fb_forward") == ("critic",)
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    boot = {"next_action": b["actions"]}
    g = jax.grad(lambda p: c.stage_loss("fb_forward", {**params, **p}, targets, b, cond, boot)[0])(
        {"critic": params["critic"]})
    # the critic phi/psi DO receive gradient (guards against a trivially-zero grad).
    assert any(np.abs(np.asarray(l)).sum() > 0
               for l in jax.tree_util.tree_leaves(g["critic"]["phi"]))


def test_measure_value_legs_actually_trains():
    """FIX 1 (silent dead objective): objective='measure' + measure_legs='value' puts the
    whole loss on the z-BLIND value net (a real objective — it trains fine under 'mix'), so
    stage_groups must return ('value',). If it returned ('critic',) the value net would get the
    gradient but never be stepped: finite loss, nothing moves, 500k dead steps. This test would
    FAIL before the fix (stage_groups=('critic',) => value params unchanged after an update)."""
    ex_obs, ex_act = _ex()
    over = dict(objective="measure", measure_coef=1.0, measure_legs="value")
    c = FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(**over))[0]
    assert c.stage_groups("fb_forward") == ("value",)
    agent = Agent.create("fb_forward_bilinear", "ddpgbc", 0, ex_obs, ex_act,
                         _full_cfg("ddpgbc", **over))
    before = jax.tree_util.tree_leaves(agent.net["value"].params)
    agent, _ = agent.update(_batch())
    after = jax.tree_util.tree_leaves(agent.net["value"].params)
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after)), \
        "measure+measure_legs='value' must actually step the value net"


def test_fv_leg_materializes_on_plain_3arg_init():
    """FIX 2 (latent init footgun): a plain 3-arg .init(obs,goals,actions) on GCFbForwardValue
    with phi_module='isfb_twin_leg' must create the F_V leg, so a later apply(return_fv=True)
    does not raise ScopeCollectionNotFound. Before the fix fv was lazy on the return_fv path
    only: 'fv' would be absent from params and the apply below would raise."""
    from impls.utils.networks import GCFbForwardValue
    ex_obs, ex_act = _ex()
    net = GCFbForwardValue(hidden_dims=(32, 32), latent_dim=D, layer_norm=True, ensemble=True,
                           phi_module="isfb_twin_leg", project_z=False)
    params = net.init(jax.random.PRNGKey(0), ex_obs, ex_obs, ex_act)["params"]  # 3-arg drop-in
    assert "fv" in params, "the F_V leg must exist after a plain 3-arg init"
    out = net.apply({"params": params}, ex_obs, ex_obs, ex_act, info=True, return_fv=True)
    assert len(out) == 5 and out[3].shape == (2, B, D)   # (v, phi, psi, fv, z); fv is [P,B,d]


def test_isfb_fv_leg_receives_gradient():
    """The SG-ISFB value measure N=<F_V,B> must actually train the F_V leg."""
    ex_obs, ex_act = _ex()
    c, params, targets = FbForwardBilinearCritic.build(
        ex_obs, ex_act, _cfg(objective="isfb", phi_module="isfb_twin_leg"))
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    g = jax.grad(lambda p: c.stage_loss("fb_forward", p, targets, b, cond, None)[0])(params)
    assert any(np.abs(np.asarray(l)).sum() > 0
               for l in jax.tree_util.tree_leaves(g["critic"]["fv"]))


# ---------------------------------------------------------------- build-time asserts
def test_unknown_enums_rejected():
    ex_obs, ex_act = _ex()
    for knob, bad, pat in [("phi_module", "mlp", "phi_module must be one of"),
                           ("objective", "sarsa", "objective must be one of"),
                           ("actor_cond", "latent", "actor_cond must be one of"),
                           ("measure_legs", "both", "measure_legs must be one of"),
                           ("measure_insample", "soft", "measure_insample must be one of")]:
        with pytest.raises(FbForwardBilinearConfigError, match=pat):
            FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(**{knob: bad}))


def test_isfb_objective_requires_twin_leg():
    ex_obs, ex_act = _ex()
    with pytest.raises(FbForwardBilinearConfigError, match="requires phi_module='isfb_twin_leg'"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(objective="isfb", phi_module="flat"))


def test_twin_leg_requires_isfb_objective():
    ex_obs, ex_act = _ex()
    with pytest.raises(FbForwardBilinearConfigError, match="only trains under objective='isfb'"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(objective="iql", phi_module="isfb_twin_leg"))


def test_measure_coef_guards():
    ex_obs, ex_act = _ex()
    with pytest.raises(FbForwardBilinearConfigError, match="ignores measure_coef"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(objective="iql", measure_coef=1.0))
    with pytest.raises(FbForwardBilinearConfigError, match="needs measure_coef>0"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(objective="measure", measure_coef=0.0))


def test_task_rank_requires_goals_test_states():
    ex_obs, ex_act = _ex()
    with pytest.raises(FbForwardBilinearConfigError, match="measure_test_rank='task' requires"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(
            objective="measure", measure_coef=1.0, measure_test_rank="task", measure_test_states="rho"))


def test_measure_test_knobs_only_on_critic_leg_measure():
    ex_obs, ex_act = _ex()
    with pytest.raises(FbForwardBilinearConfigError, match="only apply to the critic-leg measure"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(objective="iql", measure_test_states="goals"))


def test_actor_z_from_critic_requires_actor_cond_z():
    ex_obs, ex_act = _ex()
    with pytest.raises(FbForwardBilinearConfigError, match="silent no-op unless actor_cond='z'"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(actor_z_from_critic=True))
    FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(actor_z_from_critic=True, actor_cond="z"))


def test_iql_ortho_only_on_scalar_path():
    ex_obs, ex_act = _ex()
    with pytest.raises(FbForwardBilinearConfigError, match="iql_ortho_coef>0 only applies"):
        FbForwardBilinearCritic.build(ex_obs, ex_act, _cfg(objective="measure", measure_coef=1.0, iql_ortho_coef=1.0))


# ---------------------------------------------------------------- combos
@pytest.mark.parametrize("actor", ["ddpgbc", "awr", "fql"])
def test_valid_combos_construct(actor):
    ex_obs, ex_act = _ex()
    assert Agent.create("fb_forward_bilinear", actor, 0, ex_obs, ex_act, _full_cfg(actor)) is not None


def test_flowbc_combo_rejected():
    ex_obs, ex_act = _ex()
    with pytest.raises(ComboError, match="z"):
        Agent.create("fb_forward_bilinear", "flowbc", 0, ex_obs, ex_act, _full_cfg("flowbc"))


def test_actor_cond_z_and_eval_mapping():
    ex_obs, ex_act = _ex()
    agent = Agent.create("fb_forward_bilinear", "ddpgbc", 0, ex_obs, ex_act, _full_cfg("ddpgbc", actor_cond="z"))
    b = _batch()
    cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
    assert cond["z"].shape == (B, D)
    np.testing.assert_array_equal(np.asarray(agent.critic.cond_vec(cond)), np.asarray(cond["z"]))
    mapped = agent.critic.map_eval_cond(agent.net, {"goal": b["actor_goals"]})
    assert mapped["z"].shape == (B, D)
    acts = agent.sample_actions(b["observations"], goals=b["actor_goals"], seed=jax.random.PRNGKey(0))
    assert acts.shape == (B, ACT) and np.isfinite(np.asarray(acts)).all()


# ---------------------------------------------------------------- config completeness
def test_config_composes_and_required_keys_complete():
    cfg = load_config("fb_forward_bilinear", "ddpgbc", "cube_single")
    for key in REQUIRED_KEYS["fb_forward_bilinear"]:
        node, present = cfg, True
        for part in key.split("."):
            assert isinstance(node, dict) and part in node, key
            node = node[part]
        assert node is not None, key
    assert cfg["dataset_class"] == "GCDataset"
