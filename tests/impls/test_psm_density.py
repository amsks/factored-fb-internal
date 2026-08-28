"""PSMDensityCritic — PSM's proto representation learner, Density-FB SF branch.

Same parity method as tests/impls/test_rldp_density.py: numpy reconstruction downstream
of the towers (tower parity is pinned by test_psm_seam.py's reference fixture). Pins:

  * PROTO-STAGE PARITY: the "psm" stage of PSMDensityCritic is PSMCritic's, to the bit —
    same loss, same info — at identical params (the delegation, not a re-implementation);
  * the SF density CE against a numpy reconstruction, with headwise targets on the
    simplex and terminal rows collapsed;
  * phi isolation — the SF density stage must have ZERO gradient to phi (online phi,
    detached, both sides: the parent convention, and what keeps the softmax's additive
    gauge away from infer_cond's reward moment);
  * q_fn = sum_j p_j (phi_j^T z) at FROZEN towers, pessimism applied to the per-head
    SCALARS after integration (mean - actor_pessimism_penalty * uncertainty), centered
    second return;
  * sample_cond stashes the sg'd phi pool on top of the parent's cond (z, z_psm,
    psm_next_action all intact); the build-time lattice refusals.
"""
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.critics.psm import PSMCritic, PsmConfigError
from impls.critics.psm_density import PSMDensityCritic
from impls.utils.fb_math import targets_uncertainty

B, OBS, ACT, Z, P, MLS = 6, 12, 4, 5, 2, 4
TAU = 0.7


def _cfg(**over):
    cfg = dict(
        seed=0, batch_size=B, z_dim=Z, max_log_seed=MLS, num_parallel=P, discount=0.99,
        ortho_coef=1.0, pessimism_penalty=0.0, actor_pessimism_penalty=0.5,
        mix_ratio=0.5, norm_z=True, target_tau=0.01, proto_seed=0,
        lr_phi=1e-4, lr_sf=1e-4, lr_actor=1e-4,
        phi=dict(hidden_dim=16, hidden_layers=2, norm=True),
        sf=dict(hidden_dim=32, hidden_layers=1, embedding_layers=2),
        temp=TAU, actor_temp=None, actor_q_center=True,
    )
    cfg.update(over)
    return cfg


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((B, OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))))


def _batch(seed=1, terminals=None):
    rng = np.random.default_rng(seed)
    t = np.zeros((B,)) if terminals is None else np.asarray(terminals, np.float64)
    return dict(
        observations=jnp.asarray(rng.standard_normal((B, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((B, OBS))),
        terminals=jnp.asarray(t),
        index=jnp.arange(B),
    )


def _cond_boot(seed=2):
    rng = np.random.default_rng(seed)
    cond = {"z": jnp.asarray(rng.standard_normal((B, Z))),
            "z_psm": jnp.asarray((rng.integers(0, 2, (B, MLS))).astype(np.float64)),
            "psm_next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))}
    boot = {"next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))}
    return cond, boot


def _build(**over):
    ex_obs, ex_actions = _ex()
    return PSMDensityCritic.build(ex_obs, ex_actions, _cfg(**over))


def _build_plain():
    ex_obs, ex_actions = _ex()
    cfg = {k: v for k, v in _cfg().items() if k not in ("temp", "actor_temp", "actor_q_center")}
    return PSMCritic.build(ex_obs, ex_actions, cfg)


def _softmax_np(x):
    x = np.asarray(x, np.float64)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _np_sf_target_and_ce(critic, params, target, batch, cond, boot):
    goal = batch["next_observations"]
    phi = np.asarray(critic._phi(params["phi"], goal))
    Fs = np.asarray(critic._sf(params["sf_psi"], batch["observations"], cond["z"], batch["actions"]))
    logp = np.log(_softmax_np(np.einsum("pbz,cz->pbc", Fs, phi) / critic.temp))
    tFs = np.asarray(critic._sf(target["sf_psi"], batch["next_observations"], cond["z"],
                                boot["next_action"]))
    w_bar = _softmax_np(np.einsum("pbz,cz->pbc", tFs, phi) / critic.temp)
    d = critic.discount * (1.0 - np.asarray(batch["terminals"]).reshape(B, 1))
    t = (1.0 - d) * np.eye(B) + d * w_bar
    return t, -(t * logp).sum(-1).mean()


def test_proto_stage_is_bit_identical_to_psm():
    dens, params, target = _build()
    plain, p2, t2 = _build_plain()
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), params, p2)
    batch, (cond, boot) = _batch(), _cond_boot()
    l1, i1 = dens.stage_loss("psm", params, target, batch, cond, boot)
    l2, i2 = plain.stage_loss("psm", p2, t2, batch, cond, boot)
    assert float(l1) == float(l2)
    assert set(i1) == set(i2)
    for k in i1:
        assert float(i1[k]) == float(i2[k]), k


def test_sf_density_ce_matches_numpy_reconstruction():
    critic, params, target = _build()
    batch, (cond, boot) = _batch(), _cond_boot()
    loss, info = critic.stage_loss("sf", params, target, batch, cond, boot)
    _, ce = _np_sf_target_and_ce(critic, params, target, batch, cond, boot)
    assert np.allclose(float(loss), ce, atol=1e-10)
    assert np.allclose(float(info["sf_loss"]), ce, atol=1e-10)


def test_sf_target_rows_simplex_terminal_collapse_headwise():
    critic, params, target = _build()
    terminals = np.array([1, 0, 0, 1, 0, 0], np.float64)
    batch = _batch(terminals=terminals)
    cond, boot = _cond_boot()
    t, _ = _np_sf_target_and_ce(critic, params, target, batch, cond, boot)
    assert np.allclose(t.sum(-1), 1.0, atol=1e-12)
    for i in np.flatnonzero(terminals):
        assert np.allclose(t[:, i, :], np.eye(B)[i], atol=1e-12)
    nonterm = np.flatnonzero(terminals == 0)
    assert not np.allclose(t[0, nonterm], t[1, nonterm]), \
        "target rows identical across heads — the headwise target was collapsed"


def test_sf_density_stage_has_zero_phi_gradient():
    critic, params, target = _build()
    batch, (cond, boot) = _batch(), _cond_boot()
    g = jax.grad(lambda p: critic.stage_loss("sf", p, target, batch, cond, boot)[0])(params)
    phinorm = sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(g["phi"]))
    assert phinorm == 0.0, "the SF density CE leaked gradient into phi"
    sfnorm = sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(g["sf_psi"]))
    assert sfnorm > 0.0


def test_q_fn_integration_with_scalar_pessimism_after_the_fact():
    critic, params, _ = _build()
    batch, (cond, _) = _batch(), _cond_boot()
    pool = jnp.asarray(np.random.default_rng(5).standard_normal((B, Z)))
    cond = {**cond, "_pool_phi": pool}
    obs, actions = batch["observations"], batch["actions"]

    q, qs2 = critic.q_fn(params, obs, cond, actions)
    Fs = np.asarray(critic._sf(params["sf_psi"], obs, cond["z"], actions))
    w = _softmax_np(np.einsum("pbz,cz->pbc", Fs, np.asarray(pool)) / critic.temp)
    kg = np.einsum("cz,bz->bc", np.asarray(pool), np.asarray(cond["z"]))
    Qs = (w * kg[None]).sum(-1)                                      # [P, B] scalars
    qmean, qunc = targets_uncertainty(jnp.asarray(Qs), P)
    q_np = np.asarray(qmean) - critic.actor_pessimism_penalty * np.asarray(qunc)
    assert np.allclose(np.asarray(q), q_np, atol=1e-10)
    assert np.allclose(np.asarray(qs2), (q_np - q_np.mean())[None], atol=1e-10)

    gp = jax.grad(lambda p: critic.q_fn(p, obs, cond, actions)[0].sum())(params)
    pnorm = sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(gp))
    assert pnorm == 0.0, "q_fn leaked gradient into critic params"
    ga = jax.grad(lambda a: critic.q_fn(params, obs, cond, a)[0].sum())(actions)
    assert float(jnp.abs(ga).sum()) > 0.0


def test_sample_cond_extends_but_does_not_disturb_the_parent():
    critic, params, _ = _build()
    plain, p2, _ = _build_plain()
    net = {g: SimpleNamespace(params=params[g]) for g in critic.GROUPS}
    batch = _batch()
    c1 = critic.sample_cond(net, batch, jax.random.PRNGKey(0))
    c2 = plain.sample_cond({g: SimpleNamespace(params=p2[g]) for g in plain.GROUPS},
                           batch, jax.random.PRNGKey(0))
    for k in ("z", "z_psm", "psm_next_action"):
        assert np.allclose(np.asarray(c1[k]), np.asarray(c2[k]), atol=0), k
    expect = critic._phi(params["phi"], batch["next_observations"])
    assert np.allclose(np.asarray(c1["_pool_phi"]), np.asarray(expect), atol=0)


def test_build_refuses_pessimism_on_distributions_and_bad_temp():
    ex_obs, ex_actions = _ex()
    with pytest.raises(PsmConfigError, match="pessimism_penalty"):
        PSMDensityCritic.build(ex_obs, ex_actions, _cfg(pessimism_penalty=0.5))
    with pytest.raises(PsmConfigError, match="temp"):
        PSMDensityCritic.build(ex_obs, ex_actions, _cfg(temp=-1.0))


# ── backward_input: "hit_spec" — the object-coordinate phi input mode (psm family) ─────
# The mode lives on the PARENT (impls/critics/psm.py): `_phi_in` inside the `_phi` choke
# point, so the density subclass inherits it everywhere `_phi` is called (both stages'
# phi(goal), the ortho term, sample_cond's mixed z + pool stash, map_eval_cond,
# infer_cond).

OC_SPEC = "3:6:0.5"          # fits inside the OBS=12-wide test observations
OC_BLOCKS = ((3, 6, 0.5),)
OBS_ = OBS


def _scaled_slice_np(x, blocks):
    """Manual twin of _measure_common.scaled_hit_cols (density_fb's _density_queries
    semantics): slice each block, divide by its eps, concatenate."""
    return np.concatenate([np.asarray(x)[..., a:b] / e for (a, b, e) in blocks], -1)


def test_backward_input_state_is_bit_identical():
    """backward_input: "state" (key present) must be BIT-identical to the key being
    absent — same params, same stage losses, same q_fn — for the parent AND the density
    subclass. This is the default, so it pins shipped behavior."""
    batch, (cond, boot) = _batch(), _cond_boot()
    pool = jnp.asarray(np.random.default_rng(5).standard_normal((B, Z)))

    # density subclass
    base, params, target = _build()
    st, p2, t2 = _build(backward_input="state")
    assert st.backward_input == "state" and st.blocks == ()
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), params, p2)
    for stage in ("psm", "sf"):
        l1, _ = base.stage_loss(stage, params, target, batch, cond, boot)
        l2, _ = st.stage_loss(stage, p2, t2, batch, cond, boot)
        np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
    qc = {**cond, "_pool_phi": pool}
    for a, b_ in zip(base.q_fn(params, batch["observations"], qc, batch["actions"]),
                     st.q_fn(p2, batch["observations"], qc, batch["actions"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))

    # parent
    plain, pp, pt = _build_plain()
    ex_obs, ex_actions = _ex()
    pcfg = {k: v for k, v in _cfg().items() if k not in ("temp", "actor_temp", "actor_q_center")}
    plain2, pp2, pt2 = PSMCritic.build(ex_obs, ex_actions, {**pcfg, "backward_input": "state"})
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), pp, pp2)
    for stage in ("psm", "sf"):
        l1, _ = plain.stage_loss(stage, pp, pt, batch, cond, boot)
        l2, _ = plain2.stage_loss(stage, pp2, pt2, batch, cond, boot)
        np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
    q1, _ = plain.q_fn(pp, batch["observations"], cond, batch["actions"])
    q2, _ = plain2.q_fn(pp2, batch["observations"], cond, batch["actions"])
    np.testing.assert_array_equal(np.asarray(q1), np.asarray(q2))


def test_backward_input_hit_spec_phi_gets_the_scaled_slice():
    """In hit_spec mode `_phi` must hand the tower exactly the eps-scaled block columns:
    _phi_in matches a manual slice, and _phi(full_obs) == tower.apply(manual slice)."""
    critic, params, _ = _build(backward_input="hit_spec", measure_hit_spec=OC_SPEC)
    assert critic.backward_input == "hit_spec" and critic.blocks == OC_BLOCKS
    x = jnp.asarray(np.random.default_rng(11).standard_normal((B, OBS_)))
    manual = _scaled_slice_np(x, OC_BLOCKS)
    np.testing.assert_allclose(np.asarray(critic._phi_in(x)), manual, atol=0)
    out = critic._phi(params["phi"], x)
    direct = critic.phi.apply({"params": params["phi"]}, jnp.asarray(manual))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(direct))
    # a single row (map_eval_cond's eval goal) slices the same way
    np.testing.assert_allclose(np.asarray(critic._phi_in(x[0])), manual[0], atol=0)


def test_backward_input_build_infers_slice_dim_and_refuses_bad_specs():
    """PhiMap's input width comes from the example input: d_c (the block width) in
    hit_spec mode, OBS in state mode. Out-of-range specs and unknown modes refuse at
    build (PsmConfigError, the family's build-refusal convention)."""
    critic, params, _ = _build(backward_input="hit_spec", measure_hit_spec=OC_SPEC)
    assert params["phi"]["Dense_0"]["kernel"].shape[0] == 3
    _, sparams, _ = _build()
    assert sparams["phi"]["Dense_0"]["kernel"].shape[0] == OBS_
    ex_obs, ex_actions = _ex()   # OBS = 12 wide; the cube default reaches column 22
    with pytest.raises(PsmConfigError, match="hit_spec"):
        PSMDensityCritic.build(ex_obs, ex_actions,
                               _cfg(backward_input="hit_spec", measure_hit_spec="19:22:0.4"))
    with pytest.raises(PsmConfigError, match="backward_input"):
        PSMDensityCritic.build(ex_obs, ex_actions, _cfg(backward_input="qpos"))


@pytest.mark.parametrize("alias", ["psm_oc", "psm_density_oc"])
def test_oc_alias_every_consumer_path_runs(alias):
    """Agent.create through the alias (real shipped yaml — the SHIPPED psm HPs: ortho 1.0,
    batch 256, discount 0.99) + one update on a synthetic batch with obs_dim > 22, then
    map_eval_cond / infer_cond / q_fn / sample_actions — every consumer of phi must run
    without shape errors in hit_spec mode."""
    from impls.agents import Agent, CRITIC_ALIASES
    from impls.utils.config_utils import load_config

    assert alias in CRITIC_ALIASES
    cfg = load_config(alias, "flowbc", "cube_single")
    assert cfg["backward_input"] == "hit_spec"
    assert cfg["measure_hit_spec"] == "19:22:0.4"
    # the SHIPPED psm HPs, not the repair values
    assert cfg["ortho_coef"] == 1.0 and cfg["batch_size"] == 256 and cfg["discount"] == 0.99

    OBS2, NB, ACT2 = 28, 8, ACT   # wider than the spec's column 22
    cfg["batch_size"] = NB
    rng = np.random.default_rng(0)
    ex_obs = jnp.asarray(rng.standard_normal((NB, OBS2)))
    ex_act = jnp.asarray(np.tanh(rng.standard_normal((NB, ACT2))))
    agent = Agent.create(alias, "flowbc", 0, ex_obs, ex_act, cfg)
    assert agent.critic.backward_input == "hit_spec"
    assert agent.critic.blocks == ((19, 22, 0.4),)
    # phi's first Dense is exactly the 3-wide cube-xyz slice
    assert agent.net["phi"].params["Dense_0"]["kernel"].shape[0] == 3

    batch = dict(
        observations=jnp.asarray(rng.standard_normal((NB, OBS2))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((NB, ACT2)))),
        next_observations=jnp.asarray(rng.standard_normal((NB, OBS2))),
        terminals=jnp.zeros((NB,)),
        index=jnp.arange(NB),
    )
    agent, info = agent.update(batch)      # both stages + sample_cond + q_fn (actor term)
    assert np.isfinite(float(info["psm_loss"])) and np.isfinite(float(info["sf_loss"]))

    # map_eval_cond: the env goal row through phi -> z of the trained width
    cond = agent.critic.map_eval_cond(agent.net, {"goal": jnp.asarray(
        rng.standard_normal((OBS2,)))})
    assert cond["z"].shape == (cfg["z_dim"],)
    # infer_cond: reward-weighted phi moment on full observation rows
    ic = agent.critic.infer_cond(agent.net, batch["next_observations"],
                                 jnp.asarray(rng.standard_normal((NB,))))
    assert ic["z"].shape == (cfg["z_dim"],)
    # sample_cond standalone (also inside update; kept explicit for the stash keys)
    net = {g: SimpleNamespace(params=agent.net[g].params) for g in agent.critic.GROUPS}
    sc = agent.critic.sample_cond(net, batch, jax.random.PRNGKey(1))
    assert sc["z"].shape == (NB, cfg["z_dim"])
    # eval end-path: sample_actions maps the goal and rolls the actor
    a = agent.sample_actions(batch["observations"][0],
                             goals=jnp.asarray(rng.standard_normal((OBS2,))),
                             seed=jax.random.PRNGKey(2))
    assert np.asarray(a).shape[-1] == ACT2
