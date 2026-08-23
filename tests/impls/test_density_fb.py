"""DensityFBCritic — ported assertions from the bridge's
tests/test_gciql_measure_density.py (origin/gciql-fb-bridge @ 542c058), adapted to our
seam (no subprocess harness needed — jax runs natively here). Numpy-reference fallback
per docs/specs/2026-08-10-measure-objectives-on-mw-play.md P4 (no cross-import parity
harness): CE vs a direct-module numpy/jax reconstruction, simplex rows, eps-scaling
(kernel gets RAW cols, B tower gets pre-scaled queries), grad hygiene, target soft-update.

Plus: registry/combo (density_fb x flowbc, density_fb x bcflow build and run one update),
and a bit-identity/regression check that gciql is unaffected by the registry edit.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS
from impls.critics.density_fb import (
    DensityFBCritic, _delta_phi, _density_queries, _raw_block_cols,
)
from impls.critics._measure_common import parse_hit_spec

B, OBS, ACT, D = 12, 40, 5, 8


def _cfg(**over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, tau=0.005, seed=0,
        value_hidden_dims=[32, 32], layer_norm=True,
        measure_hit_spec="19:22:0.4", measure_density_dim=D, measure_density_temp=1.0,
        actor_q_center=True,
    )
    cfg.update(over)
    return cfg


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((B, OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))))


def _batch(seed=1, b=B):
    rng = np.random.default_rng(seed)
    g = jnp.asarray(rng.standard_normal((b, OBS)))
    return dict(
        observations=jnp.asarray(rng.standard_normal((b, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((b, OBS))),
        value_goals=g, actor_goals=g,
        rewards=jnp.asarray(rng.integers(0, 2, size=b) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=b).astype(float)),
    )


def _boot(b=B, seed=2):
    """Two DISTINCT bootstrap actions, as FlowBCActor.sample_bootstrap now emits them:
    `next_action` at the ACTOR goal (the policy-improvement draw) and `next_action_vgoal`
    at the VALUE goal (the only valid bootstrap for a target taken at value_goals — the
    bridge's `a_plus = actor(nobs, batch['value_goals'], noise)`). They differ so a test
    that reads the wrong one fails."""
    rng = np.random.default_rng(seed)
    return {"next_action": jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
            "next_action_vgoal": jnp.asarray(np.tanh(rng.standard_normal((b, ACT))))}


# --------------------------------------------------------------------------
# 1. eps-scaling contract: kernel gets RAW cols, B tower gets pre-scaled queries
# --------------------------------------------------------------------------
def test_query_scaling_matches_delta_phi_internal_convention():
    blocks = parse_hit_spec("19:22:0.4,28:31:0.8")
    rng = np.random.default_rng(3)
    nobs = rng.standard_normal((9, OBS)).astype(np.float32)
    nobs_j = jnp.asarray(nobs)

    def phi_np_xc(obs, blocks):
        parts = [obs[:, a:b] / e for (a, b, e) in blocks]
        return np.concatenate(parts, axis=1)

    u_sc = np.asarray(_density_queries(nobs_j, blocks))
    xc_ref = phi_np_xc(nobs, blocks)
    assert np.allclose(u_sc, xc_ref, atol=1e-6)

    u_raw = np.asarray(_raw_block_cols(nobs_j, blocks))
    raw_ref = np.concatenate([nobs[:, a:b] for (a, b, e) in blocks], axis=1)
    assert np.allclose(u_raw, raw_ref, atol=1e-6)
    assert not np.allclose(u_raw, u_sc, atol=1e-3), \
        "u_raw must be RAW obs units, not eps-scaled (the double-scale trap)"

    k = np.asarray(_delta_phi(nobs_j, u_raw, blocks))
    assert k.shape == (9, 9)
    assert np.allclose(k.sum(-1), 1.0, atol=1e-5)
    # free self-coverage: row i's nearest query is query i itself.
    assert (np.argmax(k, axis=-1) == np.arange(9)).all()


def test_softmax_is_over_the_query_axis():
    """The wrong axis (batch, not query) also runs and also looks plausible — pin the
    convention directly: rows of `w` (and of the kernel `k`) must sum to 1 over the LAST
    (query) axis, not the batch axis."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"],
            "_density_u_raw": _raw_block_cols(b["next_observations"], c.blocks),
            "_density_u_sc": _density_queries(b["next_observations"], c.blocks)}
    loss, info = c.stage_loss("density", params, targets, b, cond, boot=_boot())
    assert jnp.isfinite(loss)
    k = _delta_phi(b["next_observations"], _raw_block_cols(b["next_observations"], c.blocks), c.blocks)
    np.testing.assert_allclose(np.asarray(k.sum(-1)), 1.0, atol=1e-5)
    assert not np.allclose(np.asarray(k.sum(0)), 1.0, atol=1e-2), \
        "k summed over the WRONG (batch) axis should not also land on 1 -- otherwise this " \
        "test cannot discriminate the axis convention on this batch"


# --------------------------------------------------------------------------
# 2. CE matches a numpy/direct-module reference (incl. G2 ESS gate instruments)
# --------------------------------------------------------------------------
def test_ce_and_ess_match_numpy_reference():
    # NOTE: this reference reconstructs the SOFT (kernel) CE, so the soft path is pinned
    # explicitly -- measure_density_hard now DEFAULTS TRUE.
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(
        ex_obs, ex_act, _cfg(measure_density_hard=False))
    b = _batch(seed=11, b=6)
    boot = _boot(b=6, seed=12)
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}

    loss, info = c.stage_loss("density", params, targets, b, cond, boot)

    # direct-module reference, mirroring density_loss's own computation.
    blocks = c.blocks
    nobs, obs, acts, g = b["next_observations"], b["observations"], b["actions"], b["value_goals"]
    u_raw = _raw_block_cols(nobs, blocks)
    u_sc = _density_queries(nobs, blocks)
    b_emb = c.density_b.apply({"params": params["density_b"]}, u_sc)
    f_emb = c.density_f.apply({"params": params["density_f"]}, jnp.concatenate([obs, acts, g], -1))
    logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / c.temp
    logp = np.asarray(jax.nn.log_softmax(logits, -1))
    a_plus = boot["next_action_vgoal"]   # the VALUE-goal bootstrap (bridge density_loss)
    tb = c.density_b.apply({"params": targets["density_b"]}, u_sc)
    tf = c.density_f.apply({"params": targets["density_f"]}, jnp.concatenate([nobs, a_plus, g], -1))
    w_bar = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / c.temp, -1))
    k = np.asarray(_delta_phi(nobs, u_raw, blocks))
    t = (1.0 - c.discount) * k[None] + c.discount * w_bar
    ce_ref = -np.mean(np.sum(t * logp, axis=-1))
    assert np.allclose(ce_ref, float(loss), atol=1e-5)
    assert np.allclose(t.sum(-1), 1.0, atol=1e-5)

    w = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    ess_rows = 1.0 / np.sum(w ** 2, axis=-1)
    assert np.allclose(np.mean(ess_rows), float(info["density_ess"]), atol=1e-5)
    assert np.allclose(np.median(ess_rows), float(info["density_ess_median"]), atol=1e-5)
    assert np.allclose(np.mean(ess_rows[0]), float(info["density_ess_h0"]), atol=1e-5)
    assert np.allclose(np.mean(ess_rows[1]), float(info["density_ess_h1"]), atol=1e-5)
    # the median statistic must be a REAL median, not a relabeled mean.
    assert not np.allclose(np.mean(ess_rows), np.median(ess_rows), atol=1e-9), \
        "test setup: mean and median coincide on this batch, cannot discriminate them"


# --------------------------------------------------------------------------
# 3. target towers: init mirror + soft update, BOTH towers (via a real Agent.update)
# --------------------------------------------------------------------------
def test_target_towers_init_mirror_and_soft_update():
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg = load_config("density_fb", "flowbc", "cube_single")
    cfg["measure_density_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    agent = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg)

    for tower in ("density_f", "density_b"):
        onl0 = jax.tree_util.tree_leaves(agent.net[tower].params)
        tgt0 = jax.tree_util.tree_leaves(agent.target[tower])
        assert len(onl0) == len(tgt0) and len(onl0) > 0
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(onl0, tgt0))

    b1 = _batch(seed=0, b=16)
    a1, info1 = agent.update(b1)
    tau = cfg["tau"]
    for tower in ("density_f", "density_b"):
        onl0 = jax.tree_util.tree_leaves(agent.net[tower].params)
        onl1 = jax.tree_util.tree_leaves(a1.net[tower].params)
        assert any(not np.allclose(np.asarray(o0), np.asarray(o1))
                   for o0, o1 in zip(onl0, onl1)), f"{tower} did not move after one update()"

        b2 = _batch(seed=1, b=16)
        a2, info2 = a1.update(b2)
        # Agent.apply_update EMAs with the POST-gradient-step params of the same update
        # (impls/agents/__init__.py: soft_update(net[g].params, ...) after apply_gradients).
        onl2 = jax.tree_util.tree_leaves(a2.net[tower].params)
        tgt1 = jax.tree_util.tree_leaves(a1.target[tower])
        tgt2 = jax.tree_util.tree_leaves(a2.target[tower])
        for onl_post, old_pre, new in zip(onl2, tgt1, tgt2):
            expected = tau * np.asarray(onl_post) + (1 - tau) * np.asarray(old_pre)
            np.testing.assert_allclose(np.asarray(new), expected, atol=1e-6)


# --------------------------------------------------------------------------
# 4. grad hygiene
# --------------------------------------------------------------------------
def test_stage_loss_reads_the_TARGET_towers_and_not_the_online_ones():
    """The bootstrap w_bar must come from the POLYAK targets. Swapping
    `target["density_f"]/["density_b"]` for `params[...]` in stage_loss survives this whole
    file, test_density_fb_reward_index.py and the 60-pair byte-identity gate (verified by
    mutation), because `build()` initialises each target as an exact COPY of its tower and
    no test evaluates the loss at a target that has drifted away from it.

    test_target_towers_init_mirror_and_soft_update pins the polyak RECURSION; this pins that
    the loss actually READS its output. Perturb the targets away from the online towers and
    require the loss to respond."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b, boot = _batch(), _boot()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}

    base = float(c.stage_loss("density", params, targets, b, cond, boot)[0])
    drifted = jax.tree_util.tree_map(lambda x: x + 0.25, targets)
    moved = float(c.stage_loss("density", params, drifted, b, cond, boot)[0])
    assert not np.isclose(base, moved, rtol=1e-9, atol=1e-9), (
        "the density loss is INVARIANT to its target towers -- it is bootstrapping off the "
        "online towers")

    # control: it also depends on the online towers, so the check above is about WHICH
    # tower the target slot reaches, not about a dead loss
    p_drift = jax.tree_util.tree_map(lambda x: x + 0.25, params)
    assert not np.isclose(base, float(c.stage_loss("density", p_drift, targets, b, cond,
                                                   boot)[0]), rtol=1e-9, atol=1e-9)


def test_stage_loss_grad_touches_only_the_two_towers():
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    boot = _boot()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}

    def loss_fn(p):
        full = {"density_f": p["density_f"], "density_b": p["density_b"]}
        return c.stage_loss("density", full, targets, b, cond, boot)[0]

    grads = jax.grad(loss_fn)(params)
    for group in ("density_f", "density_b"):
        assert any(float(jnp.sum(jnp.abs(l))) > 1e-8 for l in jax.tree_util.tree_leaves(grads[group]))


def test_actor_q_term_reads_frozen_towers():
    """q_fn must not backprop into density_f/density_b: the actor's -Q.mean() term reads
    the towers at stop-gradient params."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    nobs = b["next_observations"]
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"],
            "_density_u_raw": _raw_block_cols(nobs, c.blocks),
            "_density_u_sc": _density_queries(nobs, c.blocks)}

    def q_of(p):
        q, _ = c.q_fn(p, b["observations"], cond, b["actions"])
        return q.mean()

    grads = jax.grad(q_of)(params)
    for group in ("density_f", "density_b"):
        for leaf in jax.tree_util.tree_leaves(grads[group]):
            np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))


def test_actor_q_center_subtracts_batch_mean():
    ex_obs, ex_act = _ex()
    c_on, params, _ = DensityFBCritic.build(ex_obs, ex_act, _cfg(actor_q_center=True))
    c_off, _, _ = DensityFBCritic.build(ex_obs, ex_act, _cfg(actor_q_center=False))
    b = _batch()
    nobs = b["next_observations"]
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"],
            "_density_u_raw": _raw_block_cols(nobs, c_on.blocks),
            "_density_u_sc": _density_queries(nobs, c_on.blocks)}
    q_on, qs_on = c_on.q_fn(params, b["observations"], cond, b["actions"])
    q_off, qs_off = c_off.q_fn(params, b["observations"], cond, b["actions"])
    np.testing.assert_allclose(float(jnp.mean(qs_on[0])), 0.0, atol=1e-5)
    np.testing.assert_allclose(float(jnp.mean(qs_on[1])), 0.0, atol=1e-5)
    assert not np.allclose(np.asarray(qs_off[0]).mean(), 0.0, atol=1e-3)


# --------------------------------------------------------------------------
# 5. registry / combo
# --------------------------------------------------------------------------
def test_registered():
    assert CRITICS["density_fb"] is DensityFBCritic


@pytest.mark.parametrize("actor", ["flowbc", "bcflow"])
def test_combo_builds_and_updates(actor):
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg = load_config("density_fb", actor, "cube_single")
    cfg["measure_density_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    cfg["batch_size"] = 16
    agent = Agent.create("density_fb", actor, 0, ex_obs, ex_act, cfg)
    b = _batch(b=16)
    agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite"


# --------------------------------------------------------------------------
# 6. existing critics unaffected by the registry edit
# --------------------------------------------------------------------------
def test_gciql_registry_and_update_unaffected():
    from impls.critics.gciql import GCIQLCritic
    assert CRITICS["gciql"] is GCIQLCritic
    cfg = dict(value_hidden_dims=[32, 32], layer_norm=True, discount=0.99, expectile=0.9,
              tau=0.005, lr=3e-4, lr_actor=3e-4, gc_negative=True, q_bound=0.0,
              actor=dict(hidden_dims=[32, 32], alpha=0.3, const_std=True, layer_norm=False))
    ex_obs, ex_act = _ex()
    a0 = Agent.create("gciql", "ddpgbc", 0, ex_obs, ex_act, cfg)
    a1 = Agent.create("gciql", "ddpgbc", 0, ex_obs, ex_act, cfg)
    b = _batch()
    o0, i0 = a0.update(b)
    o1, i1 = a1.update(b)
    for k in i0:
        np.testing.assert_allclose(np.asarray(i0[k]), np.asarray(i1[k]), atol=0)


def test_actor_q_center_touches_only_the_normalizer():
    """Bridge semantics (gciql_flow.py ~1168): actor_q_center changes the q-loss
    NORMALIZER (|q - q.mean()|), never the -Q.mean() numerator. Centering the numerator
    zeroes the greedy gradient (batch-mean of a batch-centered quantity) and reduces the
    actor to pure BC — the failure mode of the first anchor generation (P5, 2026-08-11:
    success 0.05 = BC floor instead of 0.929)."""
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    q_outs = {}
    for center in (False, True):
        cfg = load_config("density_fb", "flowbc", "cube_single")
        cfg["measure_density_dim"] = D
        cfg["value_hidden_dims"] = [32, 32]
        cfg["actor_q_center"] = center
        agent = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg)
        b = _batch(seed=0, b=16)
        cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
        params = {g: agent.net[g].params for g in agent.net}
        q, qs = agent.critic.q_fn(params, b["observations"], cond, b["actions"])
        q_outs[center] = np.asarray(q)
        if center:
            # normalizer input is centered ...
            assert abs(float(np.asarray(qs).mean())) < 1e-8
    # ... but the numerator (first output) is IDENTICAL with and without centering.
    np.testing.assert_allclose(q_outs[True], q_outs[False], atol=0)


# --------------------------------------------------------------------------
# 7. measure_density_hard (spec 21): one-hot immediate target
# --------------------------------------------------------------------------
def test_hard_immediate_target_is_one_hot():
    """measure_density_hard=True swaps the immediate CE target k for the identity
    (one-hot at the row's own successor, valid because M == B — the query support is
    this batch's own next states). Mirrors test_ce_and_ess_match_numpy_reference but
    with k = np.eye(B) in the numpy reference."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg(measure_density_hard=True))
    assert c.hard is True
    b = _batch(seed=11, b=6)
    boot = _boot(b=6, seed=12)
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}

    loss, info = c.stage_loss("density", params, targets, b, cond, boot)

    blocks = c.blocks
    nobs, obs, acts, g = b["next_observations"], b["observations"], b["actions"], b["value_goals"]
    u_sc = _density_queries(nobs, blocks)
    b_emb = c.density_b.apply({"params": params["density_b"]}, u_sc)
    f_emb = c.density_f.apply({"params": params["density_f"]}, jnp.concatenate([obs, acts, g], -1))
    logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / c.temp
    logp = np.asarray(jax.nn.log_softmax(logits, -1))
    a_plus = boot["next_action_vgoal"]   # the VALUE-goal bootstrap (bridge density_loss)
    tb = c.density_b.apply({"params": targets["density_b"]}, u_sc)
    tf = c.density_f.apply({"params": targets["density_f"]}, jnp.concatenate([nobs, a_plus, g], -1))
    w_bar = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / c.temp, -1))
    k = np.eye(nobs.shape[0])
    t = (1.0 - c.discount) * k[None] + c.discount * w_bar
    ce_ref = -np.mean(np.sum(t * logp, axis=-1))
    assert np.allclose(ce_ref, float(loss), atol=1e-5)
    assert np.allclose(t.sum(-1), 1.0, atol=1e-5)
    assert np.allclose(float(info["density_k_self"]), 1.0, atol=1e-6), \
        "hard k's diagonal is the identity's diagonal -- k_self must sit at its ceiling"


def test_hard_defaults_true_and_soft_must_be_asked_for():
    """measure_density_hard DEFAULTS TRUE (the shipped variant). Previously this file
    asserted the opposite -- that an absent key equalled hard=False -- which is why the
    code fallback and the yaml disagreed for so long without anything failing.

    The contract now: absent key == hard, and explicit False is a DIFFERENT objective.
    """
    ex_obs, ex_act = _ex()
    c_soft, p_soft, t_soft = DensityFBCritic.build(
        ex_obs, ex_act, _cfg(measure_density_hard=False))
    c_default, p_def, t_def = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    c_hard, p_hard, t_hard = DensityFBCritic.build(
        ex_obs, ex_act, _cfg(measure_density_hard=True))
    assert c_soft.hard is False
    assert c_default.hard is True, "an absent measure_density_hard must mean HARD"
    assert c_hard.hard is True

    b, boot = _batch(), _boot()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    l_soft, _ = c_soft.stage_loss("density", p_soft, t_soft, b, cond, boot)
    l_def, _ = c_default.stage_loss("density", p_def, t_def, b, cond, boot)
    l_hard, _ = c_hard.stage_loss("density", p_hard, t_hard, b, cond, boot)

    np.testing.assert_allclose(float(l_def), float(l_hard), atol=0)   # default IS hard
    assert float(l_soft) != float(l_def), "soft and hard must be different objectives"


def test_stage_loss_bootstraps_at_the_value_goal_action():
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    boot = _boot()
    l0, _ = c.stage_loss("density", params, targets, b, cond, boot)
    # perturbing the ACTOR-goal bootstrap must NOT move the loss ...
    boot_a = {**boot, "next_action": jnp.zeros_like(boot["next_action"])}
    l1, _ = c.stage_loss("density", params, targets, b, cond, boot_a)
    np.testing.assert_allclose(float(l1), float(l0), atol=0)
    # ... perturbing the VALUE-goal bootstrap MUST.
    boot_v = {**boot, "next_action_vgoal": jnp.zeros_like(boot["next_action_vgoal"])}
    l2, _ = c.stage_loss("density", params, targets, b, cond, boot_v)
    assert not np.allclose(float(l2), float(l0), atol=1e-9)


def test_stage_loss_refuses_a_boot_without_the_value_goal_action():
    """No silent fallback to boot['next_action'] — that fallback is the bug."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    with pytest.raises(KeyError, match="next_action_vgoal"):
        c.stage_loss("density", params, targets, b, cond,
                     {"next_action": _boot()["next_action"]})


def test_flowbc_bootstrap_draws_the_value_goal_action():
    """FlowBCActor emits BOTH bootstraps off the SAME noise draw: `next_action` at
    _cvec(cond) (the actor goal, what the policy-improvement term uses) and
    `next_action_vgoal` at cond['value_goal']. Same noise => every critic that reads
    `next_action` is bit-identical to before the fix."""
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg = load_config("density_fb", "flowbc", "cube_single")
    cfg["measure_density_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    agent = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg)
    b = _batch(seed=5)
    b["value_goals"] = jnp.asarray(np.random.default_rng(7).standard_normal((B, OBS)))
    cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
    rng = jax.random.PRNGKey(3)
    boot = agent.actor.sample_bootstrap(agent.net, b, cond, rng)

    noise = jax.random.normal(jax.random.split(rng, 4)[0], (B, ACT))
    p = agent.net["actor"].params
    a_actor = agent.actor._actor(p, b["next_observations"], b["actor_goals"], noise)
    a_value = agent.actor._actor(p, b["next_observations"], b["value_goals"], noise)
    np.testing.assert_allclose(np.asarray(boot["next_action"]), np.asarray(a_actor), atol=0)
    np.testing.assert_allclose(np.asarray(boot["next_action_vgoal"]), np.asarray(a_value), atol=0)
    assert not np.allclose(np.asarray(a_actor), np.asarray(a_value), atol=1e-6), \
        "test setup: the two goals must give different actions for this to discriminate"


# --------------------------------------------------------------------------
# 7. bridge spec 22: the ddpgbc extraction (impls/actors/ddpgbc_density.py)
# --------------------------------------------------------------------------
def _dd_cfg(**over):
    from impls.utils.config_utils import load_config
    cfg = load_config("density_fb", "ddpgbc_density", "cube_single")
    cfg["measure_density_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    cfg["batch_size"] = B
    cfg.update(over)
    return cfg


def test_ddpgbc_density_registered_and_distinct_from_stock():
    from impls.agents import ACTORS
    from impls.actors.ddpgbc import DDPGBCActor
    from impls.actors.ddpgbc_density import DDPGBCDensityActor
    assert ACTORS["ddpgbc_density"] is DDPGBCDensityActor
    assert ACTORS["ddpgbc"] is DDPGBCActor, "stock ddpgbc must keep its own class"
    # The stock branch is what the existing gciql x ddpgbc cells ran; it must not inherit
    # the new normalizer or the value-goal bootstrap.
    assert DDPGBCActor.loss is not DDPGBCDensityActor.loss
    assert DDPGBCActor.sample_bootstrap is not DDPGBCDensityActor.sample_bootstrap


def test_ddpgbc_density_builds_and_updates():
    ex_obs, ex_act = _ex()
    agent = Agent.create("density_fb", "ddpgbc_density", 0, ex_obs, ex_act, _dd_cfg())
    agent, info = agent.update(_batch())
    for k in ("actor_loss", "q_loss", "bc_loss", "q_scale", "action_gap_l2", "density_ce"):
        assert k in info, f"missing {k}"
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite"


def test_ddpgbc_density_bootstraps_at_the_value_goal_with_mode():
    """density_fb's forward tower is F(s,a,g), so the TD target must bootstrap at the VALUE
    goal; and under const_std the draw is dist.mode(), which is rng-independent."""
    ex_obs, ex_act = _ex()
    agent = Agent.create("density_fb", "ddpgbc_density", 0, ex_obs, ex_act, _dd_cfg())
    b = _batch(seed=5)
    b["value_goals"] = jnp.asarray(np.random.default_rng(7).standard_normal((B, OBS)))
    cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, b, cond, jax.random.PRNGKey(3))

    assert "next_action_vgoal" in boot, "density_fb refuses a boot without it"
    assert boot["next_action_vgoal"].shape == (B, ACT)
    p = agent.net["actor"].params
    mode_v = jnp.clip(agent.actor._dist(p, b["next_observations"], b["value_goals"]).mode(), -1, 1)
    mode_a = jnp.clip(agent.actor._dist(p, b["next_observations"], b["actor_goals"]).mode(), -1, 1)
    np.testing.assert_allclose(np.asarray(boot["next_action_vgoal"]), np.asarray(mode_v), atol=0)
    np.testing.assert_allclose(np.asarray(boot["next_action"]), np.asarray(mode_a), atol=0)
    assert not np.allclose(np.asarray(mode_v), np.asarray(mode_a), atol=1e-6), \
        "test setup: the two goals must give different actions for this to discriminate"

    # mode(), not sample(): a different rng must not move either draw.
    boot2 = agent.actor.sample_bootstrap(agent.net, b, cond, jax.random.PRNGKey(999))
    np.testing.assert_allclose(np.asarray(boot2["next_action_vgoal"]),
                               np.asarray(boot["next_action_vgoal"]), atol=0)


def test_ddpgbc_density_normalizer_is_the_centered_one():
    """Bridge branch hardcodes q_scale = |q - q.mean()|.mean(). We read it off the critic's
    SECOND q_fn return, which is the centered leg when actor_q_center is on."""
    ex_obs, ex_act = _ex()
    for center, expect_centered in ((True, True), (False, False)):
        cfg = _dd_cfg(actor_q_center=center)
        agent = Agent.create("density_fb", "ddpgbc_density", 0, ex_obs, ex_act, cfg)
        b = _batch()
        cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
        boot = agent.actor.sample_bootstrap(agent.net, b, cond, jax.random.PRNGKey(1))
        params = {g: agent.net[g].params for g in agent.net}
        _, info = agent.actor.loss(params, agent.target, b, agent.critic, cond, boot)

        dist = agent.actor._dist(params["actor"], b["observations"], agent.critic.cond_vec(cond))
        q, qs = agent.critic.q_fn(params, b["observations"], cond,
                                  jnp.clip(dist.mode(), -1, 1))
        centered = float(jnp.abs(q - q.mean()).mean())
        got = float(info["q_scale"])
        if expect_centered:
            assert abs(got - centered) < 1e-8, f"q_scale {got} is not the centered |q-q.mean()|"
        else:
            assert abs(got - float(jnp.abs(qs).mean())) < 1e-8
            assert abs(got - centered) > 1e-8, "uncentered config must not equal the centered one"


def test_ddpgbc_density_q_term_reaches_the_actor():
    """A pure-BC actor would be the failure mode of centering the NUMERATOR (the failed first
    anchor generation). The q-term must produce a non-zero gradient contribution."""
    ex_obs, ex_act = _ex()
    agent = Agent.create("density_fb", "ddpgbc_density", 0, ex_obs, ex_act, _dd_cfg())
    b = _batch()
    cond = agent.critic.sample_cond(agent.net, b, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, b, cond, jax.random.PRNGKey(1))
    online = {g: agent.net[g].params for g in agent.net}

    import dataclasses

    def q_only(ap, alpha):
        a = dataclasses.replace(agent.actor, alpha=alpha)
        loss, _ = a.loss({**online, "actor": ap}, agent.target, b, agent.critic, cond, boot)
        return loss

    g_full = jax.grad(q_only)(online["actor"], 0.3)
    g_bconly = jax.grad(q_only)(online["actor"], 0.0)   # alpha=0 -> q-term only
    flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
    assert float(jnp.abs(flat(g_bconly)).max()) > 1e-10, "q-term contributes no gradient"
    assert not np.allclose(np.asarray(flat(g_full)), np.asarray(flat(g_bconly)), atol=1e-12)


def test_flowbc_drift_instrumentation_does_not_change_the_update():
    """A4 logs Q(s,a_data)-Q(s,a_actor) and ||a_actor-a_data|| in flowbc.loss info. It is
    reporting only: the resulting parameters must be bit-identical with it stubbed out."""
    from impls.actors.flowbc import FlowBCActor
    ex_obs, ex_act = _ex()
    from impls.utils.config_utils import load_config
    cfg = load_config("density_fb", "flowbc", "cube_single")
    cfg["measure_density_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    b = _batch()

    a_real, _ = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg).update(b)
    orig = FlowBCActor._policy_drift
    try:
        FlowBCActor._policy_drift = staticmethod(lambda *a, **k: {})
        a_stub, _ = Agent.create("density_fb", "flowbc", 0, ex_obs, ex_act, cfg).update(b)
    finally:
        # staticmethod(...) is REQUIRED: `orig` is the underlying function, and assigning
        # it back plainly rebinds it as an INSTANCE method, so every subsequent
        # FlowBCActor.loss in the same process dies with "takes 7 positional arguments but
        # 8 were given". Leaked into test_density_fb_actors.py, which is the first test to
        # use flowbc after this file.
        FlowBCActor._policy_drift = staticmethod(orig)

    for g in a_real.net:
        for x, y in zip(jax.tree_util.tree_leaves(a_real.net[g].params),
                        jax.tree_util.tree_leaves(a_stub.net[g].params)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# --------------------------------------------------------------------------
# 8. edge-attribution spec (docs/specs/2026-08-17-gciql-edge-attribution.md)
#    W1 measure_density_bootstrap=sarsa (M1) / W2 measure_density_actor_temp (M2)
# --------------------------------------------------------------------------
def _cond_for(c, b):
    return {"value_goal": b["value_goals"], "goal": b["actor_goals"],
            "_density_u_raw": _raw_block_cols(b["next_observations"], c.blocks),
            "_density_u_sc": _density_queries(b["next_observations"], c.blocks)}


def _batch_with_next_actions(seed=1, b=B):
    batch = _batch(seed=seed, b=b)
    rng = np.random.default_rng(seed + 500)
    batch["next_actions"] = jnp.asarray(np.tanh(rng.standard_normal((b, ACT))))
    return batch


def test_w1_default_bootstrap_is_bit_identical():
    """The knob defaults to 'actor'. A batch that merely CARRIES next_actions (GCDataset now
    always emits it) must not move a single bit of the default path."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    assert c.bootstrap == "actor" and c.actor_temp is None
    boot = _boot()
    plain, withna = _batch(), _batch_with_next_actions()
    l0, i0 = c.stage_loss("density", params, targets, plain, _cond_for(c, plain), boot)
    l1, i1 = c.stage_loss("density", params, targets, withna, _cond_for(c, withna), boot)
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l1))
    for k in i0:
        np.testing.assert_array_equal(np.asarray(i0[k]), np.asarray(i1[k]), err_msg=k)


def test_w1_sarsa_uses_the_dataset_action_and_never_the_actor():
    """Under sarsa the actor is outside the critic's fixed point entirely — so the target must
    change when the DATASET's next action changes, must NOT change when the actor's does, and
    must still compute when the actor supplied no bootstrap at all."""
    ex_obs, ex_act = _ex()
    cfg = _cfg(measure_density_bootstrap="sarsa")
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, cfg)
    assert c.bootstrap == "sarsa"
    b = _batch_with_next_actions()
    cond = _cond_for(c, b)

    base, _ = c.stage_loss("density", params, targets, b, cond, _boot())
    # A different ACTOR draw must not move it.
    other_actor, _ = c.stage_loss("density", params, targets, b, cond, _boot(seed=77))
    np.testing.assert_array_equal(np.asarray(base), np.asarray(other_actor))
    # A different DATASET next action must.
    b2 = dict(b); b2["next_actions"] = jnp.asarray(
        np.tanh(np.random.default_rng(9).standard_normal((B, ACT))))
    moved, _ = c.stage_loss("density", params, targets, b2, cond, _boot())
    assert not np.allclose(np.asarray(base), np.asarray(moved)), \
        "sarsa target ignored batch['next_actions']"
    # No actor bootstrap at all: still computes (the actor is not in the target).
    ok, _ = c.stage_loss("density", params, targets, b, cond, boot={})
    np.testing.assert_array_equal(np.asarray(base), np.asarray(ok))


def test_w1_sarsa_without_next_actions_fails_loudly():
    """A silent fallback to the actor action would restore M1 and invalidate the arm."""
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg(measure_density_bootstrap="sarsa"))
    b = _batch()
    with pytest.raises(KeyError, match="next_actions"):
        c.stage_loss("density", params, targets, b, _cond_for(c, b), _boot())


def test_w1_rejects_an_unknown_bootstrap_mode():
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg(measure_density_bootstrap="behaviour"))
    b = _batch_with_next_actions()
    with pytest.raises(ValueError, match="must be 'actor' or 'sarsa'"):
        c.stage_loss("density", params, targets, b, _cond_for(c, b), _boot())


def test_w1_gcdataset_next_actions_respect_trajectory_boundaries():
    """idx+1 at a trajectory's LAST row belongs to the next trajectory. Clamping must repeat
    that trajectory's own final action instead of leaking one from an unrelated episode."""
    from impls.utils.datasets import Dataset, GCDataset
    T, n_traj = 5, 3
    n = T * n_traj
    terminals = np.zeros(n, np.float32); terminals[T - 1::T] = 1.0
    # actions[i] = i, so a leak across the boundary is immediately visible as a value jump.
    ds = Dataset.create(observations=np.arange(n, dtype=np.float32).reshape(n, 1),
                        actions=np.arange(n, dtype=np.float32).reshape(n, 1),
                        terminals=terminals)
    cfg = dict(value_p_curgoal=0.0, value_p_trajgoal=1.0, value_p_randomgoal=0.0,
               value_geom_sample=False, actor_p_curgoal=0.0, actor_p_trajgoal=1.0,
               actor_p_randomgoal=0.0, actor_geom_sample=False, gc_negative=True,
               discount=0.99)
    gc = GCDataset(ds, cfg)
    idxs = np.arange(n)
    batch = gc.sample(n, idxs=idxs)
    got = np.asarray(batch["next_actions"]).ravel()
    expect = np.minimum(idxs + 1, (idxs // T) * T + (T - 1)).astype(np.float32)
    np.testing.assert_array_equal(got, expect)
    for t in range(n_traj):                      # the boundary rows specifically
        last = t * T + (T - 1)
        assert got[last] == float(last), f"row {last} leaked action {got[last]}"


def test_w2_actor_temp_moves_only_the_actor_readout():
    """Raising the actor temperature must smooth q_fn's softmax and leave stage_loss — the
    thing the measure is actually trained to be — bit-identical."""
    ex_obs, ex_act = _ex()
    b = _batch_with_next_actions()
    c0, p0, t0 = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    c2, p2, t2 = DensityFBCritic.build(ex_obs, ex_act, _cfg(measure_density_actor_temp=5.0))
    assert c2.actor_temp == 5.0
    cond = _cond_for(c0, b)
    acts = b["actions"]

    q0, _ = c0.q_fn(p0, b["observations"], cond, acts)
    q2, _ = c2.q_fn(p2, b["observations"], cond, acts)
    assert not np.allclose(np.asarray(q0), np.asarray(q2), atol=1e-9), \
        "actor_temp did not reach q_fn"

    l0, i0 = c0.stage_loss("density", p0, t0, b, cond, _boot())
    l2, i2 = c2.stage_loss("density", p2, t2, b, cond, _boot())
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l2))
    for k in i0:
        np.testing.assert_array_equal(np.asarray(i0[k]), np.asarray(i2[k]), err_msg=k)

    # None must be exactly the old behaviour, not merely close.
    cN, pN, _ = DensityFBCritic.build(ex_obs, ex_act, _cfg(measure_density_actor_temp=None))
    qN, _ = cN.q_fn(pN, b["observations"], cond, acts)
    np.testing.assert_array_equal(np.asarray(q0), np.asarray(qN))


def test_w1_logs_the_substitution_size():
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
    b = _batch_with_next_actions()
    boot = _boot()
    _, info = c.stage_loss("density", params, targets, b, _cond_for(c, b), boot)
    assert "density_sarsa_subst" in info
    expect = float(jnp.linalg.norm(boot["next_action_vgoal"] - b["next_actions"], axis=-1).mean())
    assert abs(float(info["density_sarsa_subst"]) - expect) < 1e-9
