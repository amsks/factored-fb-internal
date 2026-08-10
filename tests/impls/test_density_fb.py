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
    rng = np.random.default_rng(seed)
    return {"next_action": jnp.asarray(np.tanh(rng.standard_normal((b, ACT))))}


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
    ex_obs, ex_act = _ex()
    c, params, targets = DensityFBCritic.build(ex_obs, ex_act, _cfg())
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
    a_plus = boot["next_action"]
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
        tgt1 = jax.tree_util.tree_leaves(a1.target[tower])
        tgt2 = jax.tree_util.tree_leaves(a2.target[tower])
        for onl_pre, old_pre, new in zip(onl1, tgt1, tgt2):
            expected = tau * np.asarray(onl_pre) + (1 - tau) * np.asarray(old_pre)
            np.testing.assert_allclose(np.asarray(new), expected, atol=1e-6)


# --------------------------------------------------------------------------
# 4. grad hygiene
# --------------------------------------------------------------------------
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
