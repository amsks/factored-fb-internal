"""RLDPDensityCritic — Density-FB's categorical estimator under RLDP's representation.

PARITY METHOD: numpy reconstruction of everything DOWNSTREAM of the towers (softmax rows,
headwise targets, row discount, CE reduction), reading tower outputs through the critic's
own apply methods. Tower forward parity is already pinned by tests/impls/test_rldp_seam.py's
from-scratch numpy nets; re-pinning it here would duplicate that file without adding
coverage. What THIS file pins:

  * the categorical target t = (1 - d_i) one_hot(i) + d_i softmax(target logits), built
    HEADWISE (no mean-minus-uncertainty on distributions), rows on the simplex, terminal
    rows collapsed to their own one-hot;
  * the CE value against the reconstruction;
  * gradient isolation — the density CE must train forward + left_encoder ONLY: the
    backward/predictor gradients of the density stage_loss equal the PARENT's at identical
    params (both are the SP + ortho gradient; the successor term never touches B in
    either class);
  * q_fn = reward integration sum_j p_j (B_j^T z) at FROZEN towers (zero grad to critic
    params, nonzero through actions), mean over heads, centered second return;
  * sample_cond stashes the sg'd pool; the two build-time lattice refusals.
"""
import dataclasses
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.critics.rldp import RLDPCritic
from impls.critics.rldp_density import RLDPDensityCritic, RldpDensityConfigError

B, OBS, ACT, Z, L, H, P = 6, 12, 4, 5, 5, 16, 2
TAU = 0.7


def _cfg(**over):
    cfg = dict(
        seed=0, batch_size=B, z_dim=Z, L_dim=L, num_parallel=P, discount=0.99,
        f_target_tau=0.005, b_target_tau=0.005, ortho_coef=0.01, train_goal_ratio=0.5,
        fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
        horizon=1,     # no window: SP degenerates to the one-step pair (parent contract)
        forward=dict(hidden_dim=H, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=H, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=H, hidden_layers=4, norm=True),
        predictor=dict(hidden_dim=H, hidden_layers=2, embedding_layers=2),
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
    )


def _cond_boot(seed=2):
    rng = np.random.default_rng(seed)
    return ({"z": jnp.asarray(rng.standard_normal((B, Z)))},
            {"next_action": jnp.asarray(np.tanh(rng.standard_normal((B, ACT))))})


def _build(**over):
    ex_obs, ex_actions = _ex()
    return RLDPDensityCritic.build(ex_obs, ex_actions, _cfg(**over))


def _softmax_np(x):
    x = np.asarray(x, np.float64)
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _np_target_and_ce(critic, params, target, batch, cond, boot):
    """The density successor term, reconstructed with numpy softmax/CE from the critic's
    own tower outputs. Returns (t, ce)."""
    goal = batch["next_observations"]
    le = critic._le(params["left_encoder"], batch["observations"])
    Fs = np.asarray(critic._F(params["forward"], le, cond["z"], batch["actions"]))
    Bg = np.asarray(critic._B(params["backward"], goal))
    logits = np.einsum("pbz,cz->pbc", Fs, Bg) / critic.temp
    logp = np.log(_softmax_np(logits))

    nle = critic._le(target["left_encoder"], batch["next_observations"])
    tFs = np.asarray(critic._F(target["forward"], nle, cond["z"], boot["next_action"]))
    tB = np.asarray(critic._B(target["backward"], goal))
    w_bar = _softmax_np(np.einsum("pbz,cz->pbc", tFs, tB) / critic.temp)

    d = critic.discount * (1.0 - np.asarray(batch["terminals"]).reshape(B, 1))
    t = (1.0 - d) * np.eye(B) + d * w_bar
    ce = -(t * logp).sum(-1).mean()
    return t, ce


def test_density_ce_matches_numpy_reconstruction():
    critic, params, target = _build()
    batch, (cond, boot) = _batch(), _cond_boot()
    loss, info = critic.stage_loss("fb", params, target, batch, cond, boot)
    _, ce = _np_target_and_ce(critic, params, target, batch, cond, boot)
    assert np.allclose(float(info["density_ce"]), ce, atol=1e-10)
    # loss = ce + sp + ortho_coef * orth; sp/ortho are the parent's, pinned elsewhere.
    assert np.allclose(float(loss),
                       ce + float(info["sp_loss"]) + critic.ortho_coef * float(info["ortho_loss"]),
                       atol=1e-10)


def test_target_rows_are_simplex_and_terminal_rows_collapse():
    critic, params, target = _build()
    terminals = np.array([1, 0, 0, 1, 0, 0], np.float64)
    batch = _batch(terminals=terminals)
    cond, boot = _cond_boot()
    t, _ = _np_target_and_ce(critic, params, target, batch, cond, boot)
    assert np.allclose(t.sum(-1), 1.0, atol=1e-12), "every target row must stay on the simplex"
    for i in np.flatnonzero(terminals):
        assert np.allclose(t[:, i, :], np.eye(B)[i], atol=1e-12), \
            "a terminal row must put ALL its mass on the observed successor"
    # headwise: with distinct heads the nonterminal target rows must differ across heads
    nonterm = np.flatnonzero(terminals == 0)
    assert not np.allclose(t[0, nonterm], t[1, nonterm]), \
        "target rows identical across heads — the headwise target was collapsed"


def test_density_ce_trains_exactly_what_the_parent_successor_term_trains():
    """Backward + predictor gradients must be IDENTICAL to the parent's at the same
    params: in both classes the successor term never touches B (divergence 1), so those
    gradients are the SP + ortho gradient alone. A mismatch means the density CE leaked
    into the representation learner."""
    ex_obs, ex_actions = _ex()
    dens, params, target = RLDPDensityCritic.build(ex_obs, ex_actions, _cfg())
    plain, p2, t2 = RLDPCritic.build(ex_obs, ex_actions,
                                     {k: v for k, v in _cfg().items()
                                      if k not in ("temp", "actor_temp", "actor_q_center")})
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), params, p2)

    batch, (cond, boot) = _batch(), _cond_boot()
    gd = jax.grad(lambda p: dens.stage_loss("fb", p, target, batch, cond, boot)[0])(params)
    gp = jax.grad(lambda p: plain.stage_loss("fb", p, t2, batch, cond, boot)[0])(p2)
    for grp in ("backward", "predictor"):
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b, atol=1e-10), gd[grp], gp[grp])
    # and the density CE DOES train the forward map (nonzero forward grad).
    fnorm = sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(gd["forward"]))
    assert fnorm > 0.0


def test_q_fn_matches_manual_integration_and_freezes_towers():
    critic, params, _ = _build()
    batch, (cond, _) = _batch(), _cond_boot()
    pool = jnp.asarray(np.random.default_rng(5).standard_normal((B, Z)))
    cond = {**cond, "_pool_B": pool}
    obs, actions = batch["observations"], batch["actions"]

    q, qs2 = critic.q_fn(params, obs, cond, actions)
    le = critic._le(params["left_encoder"], obs)
    Fs = np.asarray(critic._F(params["forward"], le, cond["z"], actions))
    w = _softmax_np(np.einsum("pbz,cz->pbc", Fs, np.asarray(pool)) / critic.temp)
    kg = np.einsum("cz,bz->bc", np.asarray(pool), np.asarray(cond["z"]))
    q_np = (w * kg[None]).sum(-1).mean(0)                    # mean over heads
    assert np.allclose(np.asarray(q), q_np, atol=1e-10)
    # actor_q_center: the SECOND return is the centered vector (FlowBC's normalizer input)
    assert np.allclose(np.asarray(qs2), (q_np - q_np.mean())[None], atol=1e-10)

    # frozen towers: zero grad to every critic param, live grad through actions.
    gp = jax.grad(lambda p: critic.q_fn(p, obs, cond, actions)[0].sum())(params)
    pnorm = sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(gp))
    assert pnorm == 0.0, "q_fn leaked gradient into critic params"
    ga = jax.grad(lambda a: critic.q_fn(params, obs, cond, a)[0].sum())(actions)
    assert float(jnp.abs(ga).sum()) > 0.0


def test_q_fn_uncentered_returns_per_head_values():
    critic, params, _ = _build(actor_q_center=False)
    batch, (cond, _) = _batch(), _cond_boot()
    cond = {**cond, "_pool_B": jnp.asarray(np.random.default_rng(5).standard_normal((B, Z)))}
    _, qs = critic.q_fn(params, batch["observations"], cond, batch["actions"])
    assert qs.shape == (P, B)


def test_sample_cond_stashes_the_pool():
    critic, params, _ = _build()
    net = {g: SimpleNamespace(params=params[g]) for g in critic.GROUPS}
    batch = _batch()
    cond = critic.sample_cond(net, batch, jax.random.PRNGKey(0))
    assert "z" in cond
    expect = critic._B(params["backward"], batch["next_observations"])
    assert np.allclose(np.asarray(cond["_pool_B"]), np.asarray(expect), atol=0)


def test_build_refuses_pessimism_on_distributions_and_bad_temp():
    ex_obs, ex_actions = _ex()
    with pytest.raises(RldpDensityConfigError, match="fb_pessimism_penalty"):
        RLDPDensityCritic.build(ex_obs, ex_actions, _cfg(fb_pessimism_penalty=0.5))
    with pytest.raises(RldpDensityConfigError, match="temp"):
        RLDPDensityCritic.build(ex_obs, ex_actions, _cfg(temp=0.0))


def test_actor_temp_flattens_only_the_readout():
    """actor_temp changes q_fn's softmax, never stage_loss's."""
    ex_obs, ex_actions = _ex()
    c1, params, target = RLDPDensityCritic.build(ex_obs, ex_actions, _cfg())
    c2 = dataclasses.replace(c1, actor_temp=10.0)
    batch, (cond, boot) = _batch(), _cond_boot()
    l1, _ = c1.stage_loss("fb", params, target, batch, cond, boot)
    l2, _ = c2.stage_loss("fb", params, target, batch, cond, boot)
    assert float(l1) == float(l2)
    cond = {**cond, "_pool_B": jnp.asarray(np.random.default_rng(5).standard_normal((B, Z)))}
    q1, _ = c1.q_fn(params, batch["observations"], cond, batch["actions"])
    q2, _ = c2.q_fn(params, batch["observations"], cond, batch["actions"])
    assert not np.allclose(np.asarray(q1), np.asarray(q2))


# ── Exp A: goal_kernel — density_fb's task interface over the learned-B outcome space ──

# A hit spec whose blocks EXIST inside the OBS=12-wide test observations (the cube default
# "19:22:0.4" deliberately does not — that is test_goal_kernel_build_refuses_short_obs).
GK_SPEC = "3:6:0.5"
GK_BLOCKS = ((3, 6, 0.5),)


def _delta_phi_np(goal_obs, u_raw, blocks):
    """Numpy twin of density_fb._delta_phi: eps-scaled internally, softmax over the
    QUERY axis."""
    parts_x = [np.asarray(goal_obs)[:, a:b] / e for (a, b, e) in blocks]
    off, parts_p = 0, []
    for (a, b, e) in blocks:
        parts_p.append(np.asarray(u_raw)[:, off:off + (b - a)] / e)
        off += b - a
    xc, pc = np.concatenate(parts_x, -1), np.concatenate(parts_p, -1)
    logits = -0.5 * np.sum((xc[:, None, :] - pc[None, :, :]) ** 2, -1)
    return _softmax_np(logits)


def test_goal_kernel_false_is_bit_identical():
    """goal_kernel=false (explicit, spec key present) must be BIT-identical to the shipped
    default-off build: same stage_loss, same q_fn outputs on fixed inputs, and sample_cond
    stashes nothing the old path did not."""
    from types import SimpleNamespace as NS
    base, params, target = _build()
    gk_off, p2, t2 = _build(goal_kernel=False, measure_hit_spec=GK_SPEC)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), params, p2)
    assert gk_off.goal_kernel is False and gk_off.blocks == ()

    batch, (cond, boot) = _batch(), _cond_boot()
    l1, _ = base.stage_loss("fb", params, target, batch, cond, boot)
    l2, _ = gk_off.stage_loss("fb", p2, t2, batch, cond, boot)
    np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))

    qcond = {**cond, "_pool_B": jnp.asarray(np.random.default_rng(5).standard_normal((B, Z)))}
    q1, c1 = base.q_fn(params, batch["observations"], qcond, batch["actions"])
    q2, c2 = gk_off.q_fn(p2, batch["observations"], qcond, batch["actions"])
    np.testing.assert_array_equal(np.asarray(q1), np.asarray(q2))
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))

    net = {g: NS(params=p2[g]) for g in gk_off.GROUPS}
    sc = gk_off.sample_cond(net, batch, jax.random.PRNGKey(0))
    assert "_goal_obs" not in sc and "_pool_u_raw" not in sc
    assert set(sc) == {"z", "_pool_B"}


def test_goal_kernel_q_fn_kg_is_the_object_coordinate_kernel():
    """In goal_kernel mode q_fn's reward vector kg must be the query-set-normalized
    object-coordinate kernel (rows on the simplex, matching a manual _delta_phi), the w
    softmax staying over the learned-B pool exactly as before."""
    critic, params, _ = _build(goal_kernel=True, measure_hit_spec=GK_SPEC)
    assert critic.goal_kernel is True and critic.blocks == GK_BLOCKS
    batch, (cond, _) = _batch(), _cond_boot()
    rng = np.random.default_rng(7)
    pool = jnp.asarray(rng.standard_normal((B, Z)))
    goal_obs = jnp.asarray(rng.standard_normal((B, OBS)))
    u_raw = np.asarray(batch["next_observations"])[:, 3:6]
    cond = {**cond, "_pool_B": pool, "_goal_obs": goal_obs,
            "_pool_u_raw": jnp.asarray(u_raw)}
    obs, actions = batch["observations"], batch["actions"]

    q, qs2 = critic.q_fn(params, obs, cond, actions)
    le = critic._le(params["left_encoder"], obs)
    Fs = np.asarray(critic._F(params["forward"], le, cond["z"], actions))
    w = _softmax_np(np.einsum("pbz,cz->pbc", Fs, np.asarray(pool)) / critic.temp)
    kg = _delta_phi_np(goal_obs, u_raw, GK_BLOCKS)
    assert np.allclose(kg.sum(-1), 1.0, atol=1e-12), "kg rows must be on the simplex"
    # the manual kernel matches the imported _delta_phi the critic actually calls
    # (atol 1e-6, not 1e-10: _delta_phi casts its protos to float32 internally, the
    # numpy twin runs in float64 — float32 noise, not a semantic difference).
    from impls.critics.density_fb import _delta_phi
    np.testing.assert_allclose(np.asarray(_delta_phi(goal_obs, jnp.asarray(u_raw),
                                                     GK_BLOCKS)), kg, atol=1e-6)
    q_np = (w * kg[None]).sum(-1).mean(0)                    # mean over heads, as before
    assert np.allclose(np.asarray(q), q_np, atol=1e-6)
    assert np.allclose(np.asarray(qs2), (q_np - q_np.mean())[None], atol=1e-6)

    # frozen towers, live actions — the discipline is unchanged by the kernel readout.
    gp = jax.grad(lambda p: critic.q_fn(p, obs, cond, actions)[0].sum())(params)
    assert sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(gp)) == 0.0
    ga = jax.grad(lambda a: critic.q_fn(params, obs, cond, a)[0].sum())(actions)
    assert float(jnp.abs(ga).sum()) > 0.0


def test_goal_kernel_sample_cond_stashes_goals_and_embeds_them():
    """sample_cond must carry _goal_obs/_pool_u_raw and z = project_z(B(goals)) with NO
    random mix — actor_goals when the batch has them, else permuted next_observations."""
    from types import SimpleNamespace as NS
    from impls.utils.fb_math import project_z
    critic, params, _ = _build(goal_kernel=True, measure_hit_spec=GK_SPEC)
    net = {g: NS(params=params[g]) for g in critic.GROUPS}
    batch = _batch()

    # fallback path: WindowedDataset serves no goals -> permuted next_observations.
    cond = critic.sample_cond(net, batch, jax.random.PRNGKey(3))
    nobs = np.asarray(batch["next_observations"])
    g = np.asarray(cond["_goal_obs"])
    assert (g[:, None, :] == nobs[None, :, :]).all(-1).any(-1).all(), \
        "fallback goals must be rows of next_observations (a permutation)"
    np.testing.assert_array_equal(np.asarray(cond["_pool_u_raw"]), nobs[:, 3:6])
    np.testing.assert_allclose(
        np.asarray(cond["z"]),
        np.asarray(project_z(critic._B(params["backward"], cond["_goal_obs"]),
                             critic.norm_z)), atol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(cond["_pool_B"]),
        np.asarray(critic._B(params["backward"], batch["next_observations"])))

    # actor_goals path: the dataset's goals are used verbatim.
    agoals = jnp.asarray(np.random.default_rng(9).standard_normal((B, OBS)))
    cond2 = critic.sample_cond(net, {**batch, "actor_goals": agoals},
                               jax.random.PRNGKey(3))
    np.testing.assert_array_equal(np.asarray(cond2["_goal_obs"]), np.asarray(agoals))
    np.testing.assert_allclose(
        np.asarray(cond2["z"]),
        np.asarray(project_z(critic._B(params["backward"], agoals), critic.norm_z)),
        atol=1e-12)


def test_goal_kernel_build_refuses_short_obs():
    """goal_kernel=true with a hit spec reaching past the observation width must refuse at
    build — jax would slice short rows into a silently different reward."""
    ex_obs, ex_actions = _ex()   # OBS = 12 wide; the cube default reaches column 22
    with pytest.raises(RldpDensityConfigError, match="goal_kernel"):
        RLDPDensityCritic.build(ex_obs, ex_actions,
                                _cfg(goal_kernel=True, measure_hit_spec="19:22:0.4"))


# ── backward_input: "hit_spec" — the object-coordinate B input mode (rldp family) ──────
# The mode lives on the PARENT (impls/critics/rldp.py): `_b_in` inside the `_B` choke
# point, so the density subclass inherits it everywhere `_B` is called. `blocks` also
# lives on the parent now; the goal_kernel build must not clobber it (regression below).

OC_SPEC = "3:6:0.5"          # fits inside the OBS=12-wide test observations
OC_BLOCKS = ((3, 6, 0.5),)


def _scaled_slice_np(x, blocks):
    """Manual twin of _measure_common.scaled_hit_cols (density_fb's _density_queries
    semantics): slice each block, divide by its eps, concatenate."""
    return np.concatenate([np.asarray(x)[..., a:b] / e for (a, b, e) in blocks], -1)


def test_backward_input_state_is_bit_identical():
    """backward_input: "state" (key present) must be BIT-identical to the key being
    absent — same params, same stage_loss, same q_fn on fixed inputs — for the parent
    AND the density subclass. This is the default, so it pins shipped behavior."""
    batch, (cond, boot) = _batch(), _cond_boot()
    pool = jnp.asarray(np.random.default_rng(5).standard_normal((B, Z)))

    # density subclass
    base, params, target = _build()
    st, p2, t2 = _build(backward_input="state")
    assert st.backward_input == "state" and st.blocks == ()
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), params, p2)
    l1, _ = base.stage_loss("fb", params, target, batch, cond, boot)
    l2, _ = st.stage_loss("fb", p2, t2, batch, cond, boot)
    np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
    qc = {**cond, "_pool_B": pool}
    for a, b_ in zip(base.q_fn(params, batch["observations"], qc, batch["actions"]),
                     st.q_fn(p2, batch["observations"], qc, batch["actions"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))

    # parent
    ex_obs, ex_actions = _ex()
    pcfg = {k: v for k, v in _cfg().items() if k not in ("temp", "actor_temp", "actor_q_center")}
    plain, pp, pt = RLDPCritic.build(ex_obs, ex_actions, pcfg)
    plain2, pp2, pt2 = RLDPCritic.build(ex_obs, ex_actions, {**pcfg, "backward_input": "state"})
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), pp, pp2)
    l1, _ = plain.stage_loss("fb", pp, pt, batch, cond, boot)
    l2, _ = plain2.stage_loss("fb", pp2, pt2, batch, cond, boot)
    np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
    q1, _ = plain.q_fn(pp, batch["observations"], cond, batch["actions"])
    q2, _ = plain2.q_fn(pp2, batch["observations"], cond, batch["actions"])
    np.testing.assert_array_equal(np.asarray(q1), np.asarray(q2))


def test_backward_input_hit_spec_B_gets_the_scaled_slice():
    """In hit_spec mode `_B` must hand the tower exactly the eps-scaled block columns:
    _b_in matches a manual slice, and _B(full_obs) == tower.apply(manual slice)."""
    critic, params, _ = _build(backward_input="hit_spec", measure_hit_spec=OC_SPEC)
    assert critic.backward_input == "hit_spec" and critic.blocks == OC_BLOCKS
    x = jnp.asarray(np.random.default_rng(11).standard_normal((B, OBS)))
    manual = _scaled_slice_np(x, OC_BLOCKS)
    np.testing.assert_allclose(np.asarray(critic._b_in(x)), manual, atol=0)
    out = critic._B(params["backward"], x)
    direct = critic.backward.apply({"params": params["backward"]}, jnp.asarray(manual))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(direct))
    # a single row (map_eval_cond's eval goal) slices the same way
    np.testing.assert_allclose(np.asarray(critic._b_in(x[0])), manual[0], atol=0)


def test_backward_input_hit_spec_survives_the_density_subclass_build():
    """Regression (2026-08-25 near-miss): the subclass build's goal_kernel replace must
    KEEP the parent's parsed blocks when goal_kernel is off, or the hit_spec input map
    is severed."""
    critic, _, _ = _build(backward_input="hit_spec", measure_hit_spec=OC_SPEC)
    assert critic.goal_kernel is False
    assert critic.blocks == OC_BLOCKS


def test_backward_input_build_infers_slice_dim_and_refuses_bad_specs():
    """PhiMap's input width comes from the example input: d_c (the block width) in
    hit_spec mode, OBS in state mode. Out-of-range specs and unknown modes refuse at
    build."""
    critic, params, _ = _build(backward_input="hit_spec", measure_hit_spec=OC_SPEC)
    assert params["backward"]["Dense_0"]["kernel"].shape[0] == 3
    _, sparams, _ = _build()
    assert sparams["backward"]["Dense_0"]["kernel"].shape[0] == OBS
    ex_obs, ex_actions = _ex()   # OBS = 12 wide; the cube default reaches column 22
    with pytest.raises(AssertionError, match="hit_spec"):
        RLDPDensityCritic.build(ex_obs, ex_actions,
                                _cfg(backward_input="hit_spec", measure_hit_spec="19:22:0.4"))
    with pytest.raises(AssertionError, match="backward_input"):
        RLDPDensityCritic.build(ex_obs, ex_actions, _cfg(backward_input="qpos"))


@pytest.mark.parametrize("alias", ["rldp_oc", "rldp_density_oc"])
def test_oc_alias_every_consumer_path_runs(alias):
    """Agent.create through the alias (real shipped yaml) + one update on a synthetic
    windowed batch with obs_dim > 22, then map_eval_cond / infer_cond / q_fn /
    sample_actions — every consumer of B must run without shape errors in hit_spec mode."""
    from types import SimpleNamespace as NS
    from impls.agents import Agent, CRITIC_ALIASES
    from impls.utils.config_utils import load_config

    assert alias in CRITIC_ALIASES
    cfg = load_config(alias, "flowbc", "cube_single")
    assert cfg["backward_input"] == "hit_spec"
    assert cfg["measure_hit_spec"] == "19:22:0.4"

    OBS2, NB = 28, 8            # wider than the spec's column 22
    cfg["batch_size"] = NB
    rng = np.random.default_rng(0)
    ex_obs = jnp.asarray(rng.standard_normal((NB, OBS2)))
    ex_act = jnp.asarray(np.tanh(rng.standard_normal((NB, ACT))))
    agent = Agent.create(alias, "flowbc", 0, ex_obs, ex_act, cfg)
    assert agent.critic.backward_input == "hit_spec"
    assert agent.critic.blocks == ((19, 22, 0.4),)
    # B's first Dense is exactly the 3-wide cube-xyz slice
    assert agent.net["backward"].params["Dense_0"]["kernel"].shape[0] == 3

    h = int(cfg["horizon"])
    batch = dict(
        observations=jnp.asarray(rng.standard_normal((NB, OBS2))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((NB, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((NB, OBS2))),
        future_observations=jnp.asarray(rng.standard_normal((NB, h - 1, OBS2))),
        future_actions=jnp.asarray(np.tanh(rng.standard_normal((NB, h - 1, ACT)))),
        terminals=jnp.zeros((NB,)),
    )
    agent, info = agent.update(batch)      # stage_loss + sample_cond + q_fn (actor term)
    assert np.isfinite(float(info["fb_loss"]))

    # map_eval_cond: the env goal row through B -> z of the trained width
    cond = agent.critic.map_eval_cond(agent.net, {"goal": jnp.asarray(
        rng.standard_normal((OBS2,)))})
    assert cond["z"].shape == (cfg["z_dim"],)
    # infer_cond: reward-weighted B moment on full observation rows
    ic = agent.critic.infer_cond(agent.net, batch["next_observations"],
                                 jnp.asarray(rng.standard_normal((NB,))))
    assert ic["z"].shape == (cfg["z_dim"],)
    # sample_cond standalone (also inside update; kept explicit for the stash keys)
    net = {g: NS(params=agent.net[g].params) for g in agent.critic.GROUPS}
    sc = agent.critic.sample_cond(net, batch, jax.random.PRNGKey(1))
    assert sc["z"].shape == (NB, cfg["z_dim"])
    # eval end-path: sample_actions maps the goal and rolls the actor
    a = agent.sample_actions(batch["observations"][0],
                             goals=jnp.asarray(rng.standard_normal((OBS2,))),
                             seed=jax.random.PRNGKey(2))
    assert np.asarray(a).shape[-1] == ACT
