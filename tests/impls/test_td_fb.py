"""TDFBCritic — the two-phase ball-visitation + hindsight successor measure. Assertions
written from `_measure_terms`/`measure_loss`'s semantics (docs/successor-measure-
objectives.md §6-10, tools/wandb_mode_shim/gciql_flow.py on origin/gciql-fb-bridge @
542c058) per docs/specs/2026-08-10-measure-objectives-on-mw-play.md P4: absorbing vs
non-absorbing fixed points on a hand-built chain, the hit matrix reproducing a
block+radius success rule on a crafted batch, hindsight columns coming from
`value_goals`, the FrozenPsi build-time assert, registry/combo, and warm_start_agent
(impls/utils/flax_utils.py, spec P3).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS
from impls.critics.td_fb import TDFBCritic, _measure_terms
from impls.critics._measure_common import parse_hit_spec, hit_matrix
from impls.utils.flax_utils import save_agent, warm_start_agent

B, OBS, ACT, D = 10, 40, 5, 8


def _cfg(**over):
    cfg = dict(
        lr=3e-4, lr_actor=3e-4, batch_size=B, discount=0.99, tau=0.005, seed=0,
        value_hidden_dims=[32, 32], layer_norm=True, value_latent_dim=D,
        measure_hit_spec="19:22:0.4", measure_scale_norm=True,
        measure_first_passage=True, measure_fp_absorbing=True,
        measure_stopgrad_psi=False, measure_testpoint_dist="hindsight",
        measure_ortho_coef=0.0, actor_q_center=False, warm_start=False,
    )
    cfg.update(over)
    return cfg


def _phase2_over(**extra):
    over = dict(measure_fp_absorbing=False, measure_stopgrad_psi=True,
               actor_q_center=True, warm_start=True)
    over.update(extra)
    return over


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((B, OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))))


def _batch(seed=1, b=B):
    rng = np.random.default_rng(seed)
    vg = jnp.asarray(rng.standard_normal((b, OBS)))
    ag = jnp.asarray(rng.standard_normal((b, OBS)))
    return dict(
        observations=jnp.asarray(rng.standard_normal((b, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((b, OBS))),
        value_goals=vg, actor_goals=ag,
        rewards=jnp.asarray(rng.integers(0, 2, size=b) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=b).astype(float)),
    )


def _boot(b=B, seed=2):
    rng = np.random.default_rng(seed)
    return {"next_action": jnp.asarray(np.tanh(rng.standard_normal((b, ACT))))}


# --------------------------------------------------------------------------
# 1. hit matrix reproduces a block+radius success rule
# --------------------------------------------------------------------------
def test_hit_matrix_reproduces_block_radius_rule():
    """Block 4:7, radius 0.05 -- the MW success-rule shape (spec P6: obs[4:7] within
    0.05m, in raw meters, no scaling) -- exercised generically on a crafted batch (no MW
    env dependency; P6 itself is out of scope for this port)."""
    blocks = parse_hit_spec("4:7:0.05")
    rng = np.random.default_rng(5)
    row = rng.standard_normal((6, OBS)).astype(np.float32)
    col = row.copy()
    # row 0: exact hit. row 1: just inside (0.04 < 0.05). row 2: just outside (0.06 > 0.05).
    col[0, 4:7] = row[0, 4:7]
    col[1, 4:7] = row[1, 4:7] + np.array([0.02, 0.02, 0.02])   # norm ~0.0346
    col[2, 4:7] = row[2, 4:7] + np.array([0.05, 0.05, 0.05])   # norm ~0.0866
    h = np.asarray(hit_matrix(jnp.asarray(row), jnp.asarray(col), blocks, jnp.float32))
    assert h[0, 0] == 1.0
    assert h[1, 1] == 1.0
    assert h[2, 2] == 0.0
    # masking outside the 4:7 block never touches the hit rule.
    row2 = row.copy(); row2[:, 30:33] += 100.0
    h2 = np.asarray(hit_matrix(jnp.asarray(row2), jnp.asarray(col), blocks, jnp.float32))
    np.testing.assert_array_equal(h, h2)


def test_hit_matrix_is_a_conjunction_over_blocks():
    blocks = parse_hit_spec("0:2:0.1,5:7:0.1")
    row = jnp.zeros((1, 10))
    col_both = jnp.zeros((1, 10))
    col_one = jnp.zeros((1, 10)).at[0, 5].set(1.0)   # fails only the second block
    dtype = jnp.float32
    assert float(hit_matrix(row, col_both, blocks, dtype)[0, 0]) == 1.0
    assert float(hit_matrix(row, col_one, blocks, dtype)[0, 0]) == 0.0


# --------------------------------------------------------------------------
# 2. absorbing vs non-absorbing fixed point (hand-built 2-state chain)
# --------------------------------------------------------------------------
def test_absorbing_vs_nonabsorbing_residual_on_hand_built_chain():
    """2 rows/cols: row(col) 0 is OUTSIDE the goal region, row(col) 1 is INSIDE it. hit =
    [[0,0],[0,1]] (only the (1,1) pair is a hit). With absorbing=True the (1,1) residual
    drops the tM bootstrap entirely (the region terminates the recursion); with
    absorbing=False it keeps it (occupancy, revisits counted) -- the two must disagree
    exactly there and agree everywhere hit=0."""
    gamma = 0.9
    M = jnp.array([[[0.2, 0.3], [0.1, 0.8]]])     # [1,2,2] (single head)
    tM = jnp.array([[[0.25, 0.35], [0.15, 0.9]]])
    hit = jnp.array([[0.0, 0.0], [0.0, 1.0]])

    off_a, diag_a = _measure_terms(M, tM, gamma, scale_norm=False, hit=hit, absorbing=True)
    off_b, diag_b = _measure_terms(M, tM, gamma, scale_norm=False, hit=hit, absorbing=False)

    # hand-compute both diffs directly.
    cont_a = 1.0 - hit          # [[1,1],[1,0]]
    diff_a = M[0] - hit - gamma * cont_a * tM[0]
    diff_b = M[0] - hit - gamma * tM[0]
    # they must differ exactly at the (1,1) entry (the only hit) and agree elsewhere.
    assert not np.isclose(float(diff_a[1, 1]), float(diff_b[1, 1]))
    np.testing.assert_allclose(np.asarray(diff_a)[:1], np.asarray(diff_b)[:1], atol=1e-7)
    np.testing.assert_allclose(np.asarray(diff_a)[1, :1], np.asarray(diff_b)[1, :1], atol=1e-7)

    off = 1.0 - jnp.eye(2)
    expected_off_a = 0.5 * jnp.sum((diff_a * off) ** 2) / jnp.sum(off)
    expected_off_b = 0.5 * jnp.sum((diff_b * off) ** 2) / jnp.sum(off)
    expected_diag_a = jnp.mean(jnp.diagonal(diff_a) ** 2)
    expected_diag_b = jnp.mean(jnp.diagonal(diff_b) ** 2)
    np.testing.assert_allclose(float(off_a[0]), float(expected_off_a), atol=1e-6)
    np.testing.assert_allclose(float(off_b[0]), float(expected_off_b), atol=1e-6)
    np.testing.assert_allclose(float(diag_a[0]), float(expected_diag_a), atol=1e-6)
    np.testing.assert_allclose(float(diag_b[0]), float(expected_diag_b), atol=1e-6)
    assert not np.isclose(float(diag_a[0]), float(diag_b[0]))


def test_hit_none_falls_back_to_plain_fb_td():
    """measure_first_passage=False drops `hit` entirely -- plain FB-style TD residual, NOT
    either phase of this objective (see td_fb.py module docstring)."""
    gamma = 0.9
    M = jnp.array([[[0.2, 0.3], [0.1, 0.8]]])
    tM = jnp.array([[[0.25, 0.35], [0.15, 0.9]]])
    off_none, diag_none = _measure_terms(M, tM, gamma, scale_norm=False, hit=None)
    diff = M[0] - gamma * tM[0]
    off = 1.0 - jnp.eye(2)
    np.testing.assert_allclose(float(off_none[0]), float(0.5 * jnp.sum((diff * off) ** 2) / jnp.sum(off)), atol=1e-6)
    np.testing.assert_allclose(float(diag_none[0]), float(-jnp.mean(jnp.diagonal(diff))), atol=1e-6)


# --------------------------------------------------------------------------
# 3. hindsight columns come from value_goals
# --------------------------------------------------------------------------
def test_hindsight_columns_are_value_goals_not_next_observations():
    ex_obs, ex_act = _ex()
    c_hs, params, targets = TDFBCritic.build(ex_obs, ex_act, _cfg(measure_testpoint_dist="hindsight"))
    c_batch, _, _ = TDFBCritic.build(ex_obs, ex_act, _cfg(measure_testpoint_dist="batch"))
    b = _batch()
    boot = _boot()
    cond1 = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    b2 = dict(b)
    b2["value_goals"] = jnp.asarray(np.random.default_rng(99).standard_normal((B, OBS)))
    cond2 = {"value_goal": b2["value_goals"], "goal": b2["actor_goals"]}

    loss_hs_1, _ = c_hs.stage_loss("measure", params, targets, b, cond1, boot)
    loss_hs_2, _ = c_hs.stage_loss("measure", params, targets, b2, cond2, boot)
    assert not np.isclose(float(loss_hs_1), float(loss_hs_2)), \
        "hindsight columns must move with value_goals"

    loss_b_1, _ = c_batch.stage_loss("measure", params, targets, b, cond1, boot)
    loss_b_2, _ = c_batch.stage_loss("measure", params, targets, b2, cond2, boot)
    # batch columns (next_observations, unchanged between b and b2) must NOT move with value_goals.
    np.testing.assert_allclose(float(loss_b_1), float(loss_b_2), atol=1e-6)


# --------------------------------------------------------------------------
# 4. FrozenPsi trap: measure_stopgrad_psi requires warm_start
# --------------------------------------------------------------------------
def test_stopgrad_psi_without_warm_start_is_a_hard_error():
    ex_obs, ex_act = _ex()
    with pytest.raises(AssertionError, match="FrozenPsi"):
        TDFBCritic.build(ex_obs, ex_act, _cfg(measure_stopgrad_psi=True, warm_start=False))
    # with warm_start declared, it builds fine.
    TDFBCritic.build(ex_obs, ex_act, _cfg(measure_stopgrad_psi=True, warm_start=True))


def test_stopgrad_psi_actually_zeros_the_psi_gradient():
    ex_obs, ex_act = _ex()
    c, params, targets = TDFBCritic.build(
        ex_obs, ex_act, _cfg(**_phase2_over()))
    b = _batch()
    boot = _boot()
    cond = {"value_goal": b["value_goals"], "goal": b["actor_goals"]}
    grads = jax.grad(lambda p: c.stage_loss("measure", p, targets, b, cond, boot)[0])(params)
    for leaf in jax.tree_util.tree_leaves(grads["critic"]["psi"]):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))
    assert any(float(jnp.sum(jnp.abs(l))) > 0 for l in jax.tree_util.tree_leaves(grads["critic"]["phi"]))


# --------------------------------------------------------------------------
# 5. registry / combo
# --------------------------------------------------------------------------
def test_registered():
    assert CRITICS["td_fb"] is TDFBCritic


@pytest.mark.parametrize("actor", ["flowbc", "bcflow"])
def test_phase1_combo_builds_and_updates(actor):
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg = load_config("td_fb", actor, "cube_single")
    cfg["value_latent_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    cfg["batch_size"] = 16
    agent = Agent.create("td_fb", actor, 0, ex_obs, ex_act, cfg)
    b = _batch(b=16)
    agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite"


def test_phase2_flags_combo_builds_and_updates():
    """Phase-2 loss flags (measure_fp_absorbing=False, measure_stopgrad_psi=True,
    actor_q_center=True, warm_start=True) build and run one update -- the param tree is
    identical to phase 1's (same module, only loss flags differ), so this does not need an
    actual checkpoint to build (only warm_start_agent, tested below, needs one)."""
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg = load_config("td_fb", "flowbc", "cube_single")
    cfg["value_latent_dim"] = D
    cfg["value_hidden_dims"] = [32, 32]
    cfg["batch_size"] = 16
    cfg.update(_phase2_over())
    agent = Agent.create("td_fb", "flowbc", 0, ex_obs, ex_act, cfg)
    b = _batch(b=16)
    agent, info = agent.update(b)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite"


# --------------------------------------------------------------------------
# 6. warm_start_agent (impls/utils/flax_utils.py, spec P3)
# --------------------------------------------------------------------------
def test_warm_start_agent_splices_params_with_fresh_optimizer_and_rng(tmp_path):
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg1 = load_config("td_fb", "flowbc", "cube_single")
    cfg1["value_latent_dim"] = D
    cfg1["value_hidden_dims"] = [32, 32]
    cfg1["batch_size"] = 16

    phase1 = Agent.create("td_fb", "flowbc", 0, ex_obs, ex_act, cfg1)
    for i in range(3):
        phase1, _ = phase1.update(_batch(seed=i, b=16))
    save_agent(phase1, str(tmp_path), 3)

    cfg2 = dict(cfg1)
    cfg2.update(_phase2_over())
    fresh = Agent.create("td_fb", "flowbc", 1, ex_obs, ex_act, cfg2)  # different seed on purpose
    warmed = warm_start_agent(fresh, str(tmp_path), 3)

    # critic params + target spliced bit-exactly from the phase-1 checkpoint.
    for a, b_ in zip(jax.tree_util.tree_leaves(warmed.net["critic"].params),
                     jax.tree_util.tree_leaves(phase1.net["critic"].params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))
    for a, b_ in zip(jax.tree_util.tree_leaves(warmed.target["critic"]),
                     jax.tree_util.tree_leaves(phase1.target["critic"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))
    # actor params spliced too (whole-agent warm start, matching the bridge convention).
    for a, b_ in zip(jax.tree_util.tree_leaves(warmed.net["actor"].params),
                     jax.tree_util.tree_leaves(phase1.net["actor"].params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))

    # optimizer state is FRESH (not copied from the 3-step-trained phase-1 run): step count
    # resets, and Adam's moment estimates are back at their init (zeros).
    assert int(warmed.net["critic"].step) == 1
    for leaf in jax.tree_util.tree_leaves(warmed.net["critic"].opt_state):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))

    # rng is fresh (agent created with seed=1), not restored from the checkpoint's own rng.
    assert not np.array_equal(np.asarray(warmed.rng), np.asarray(phase1.rng))

    # Agent pytree structure is untouched: no new field was added.
    assert set(vars(warmed).keys()) == set(vars(fresh).keys())

    # one update() from the warm-started agent runs without error.
    warmed, info = warmed.update(_batch(seed=42, b=16))
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all()


def test_warm_start_leaves_shape_mismatched_groups_at_fresh_init(tmp_path):
    """A group present in the checkpoint whose architecture does not match the fresh
    agent's stays at its fresh init rather than raising (the bridge's own splice
    semantics)."""
    from impls.utils.config_utils import load_config
    ex_obs, ex_act = _ex()
    cfg_small = load_config("td_fb", "flowbc", "cube_single")
    cfg_small["value_latent_dim"] = D
    cfg_small["value_hidden_dims"] = [32, 32]
    cfg_small["batch_size"] = 16
    small = Agent.create("td_fb", "flowbc", 0, ex_obs, ex_act, cfg_small)
    save_agent(small, str(tmp_path), 1)

    cfg_big = dict(cfg_small)
    cfg_big["value_latent_dim"] = D + 4   # architecture mismatch on the critic
    big_fresh = Agent.create("td_fb", "flowbc", 2, ex_obs, ex_act, cfg_big)
    warmed = warm_start_agent(big_fresh, str(tmp_path), 1)
    # critic shape mismatched -> stays at big_fresh's own init, not small's.
    for a, b_ in zip(jax.tree_util.tree_leaves(warmed.net["critic"].params),
                     jax.tree_util.tree_leaves(big_fresh.net["critic"].params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b_))
