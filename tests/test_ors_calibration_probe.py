"""Tests for scripts/rescuing_fb/ors_calibration_probe.py -- spec 19b sec.1
(P1.2 HARD GATE). Contract: PAPER/Multistep-FB/specs/19b-ors-extractor-impl.md
sec.1; conventions verified against the paper + released code: specs/19b-ors-
transcription.md.

No `measure_density_behavior` checkpoint exists yet (spec 19a is landing on a
parallel track in tools/wandb_mode_shim/gciql_flow.py -- see that script's own
module docstring "UNVALIDATED END-TO-END"). Per the task brief, this file
proves the script two ways:

  (A) PURE-NUMPY LAYER (no jax, no agent, no dataset) -- runs under the plain
      `.venv` pytest: ground truth on a synthetic trajectory where the
      truncated discounted sum is HAND-COMPUTABLE, index sampling, the eps-
      scaled distance, the r_W algebra (mean-of-heads softmax weighting),
      the Spearman gate's pass/fail logic, the action-axis near-duplicate
      grouping, and the consistency-refusal logic. Loaded via
      importlib.util.spec_from_file_location, same precedent as
      tests/test_eval_density_zeroshot.py -- the module stays jax-free at
      import time (verified: env_geometry / eval_zeroshot_rewards / t6_gpi_
      eval / evals._profile_core never import jax at module level).

  (B) JAX-DEPENDENT PLUMBING LAYER -- tests.jax_subprocess.run_jax (skips if
      no /dev/shm/.venv-jax), building a STRUCTURALLY VALID but UNTRAINED
      `measure_density_behavior=True` agent in-process via
      `GCIQLFlowAgent.create(...)` (the exact harness
      tests/test_gciql_measure_density.py already uses for the sibling spec
      -- same base config, same OBS_D/D/hit_spec). This proves the plumbing
      (tower selection, the F-tower's no-goal-leg convention, shapes,
      determinism) is correct; an untrained agent's r_W is NOT expected to
      pass the 0.65 Spearman gate against real data and this file does not
      test that it does -- the "spec-18 lesson: use a repeat run as the noise
      floor" determinism test is the point, not a numeric gate pass.

Runner: `.venv/bin/python -m pytest tests/test_ors_calibration_probe.py -x -q`
(NOT uv; part (B) shells out to the jax venv itself via jax_subprocess).
"""
from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.jax_subprocess import REPO, run_jax

SCRIPT = REPO / "scripts" / "rescuing_fb" / "ors_calibration_probe.py"


@pytest.fixture(scope="module")
def ocp():
    spec = importlib.util.spec_from_file_location("ors_calibration_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


# ============================ (a) eps-scaled squared distance ======================

def test_eps_scaled_sq_dist_hand_computed(ocp):
    """1-block spec eps=0.5: d2 = ||(u-g)/0.5||^2, hand-verified per row."""
    blocks = ((0, 2, 0.5),)
    u = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    g = np.array([0.0, 0.0])
    d2 = ocp.eps_scaled_sq_dist(u, g, blocks)
    np.testing.assert_allclose(d2, [4.0, 4.0, 32.0], atol=1e-12)


def test_eps_scaled_sq_dist_multiblock_heterogeneous_eps(ocp):
    """2-block spec, DIFFERENT eps per block (scene-like heterogeneity):
    d2 = sum over blocks of ||(u_blk-g_blk)/eps_blk||^2, computed block-by-
    block by hand."""
    blocks = ((0, 2, 0.4), (2, 3, 2.0))
    u = np.array([[0.4, 0.0, 2.0]])   # block0 diff (0.4,0) -> (1,0)/0.4-scale contrib 1^2=1
    g = np.array([0.0, 0.0, 0.0])     # block1 diff (2.0) -> (2/2)^2 = 1
    d2 = ocp.eps_scaled_sq_dist(u, g, blocks)
    np.testing.assert_allclose(d2, [2.0], atol=1e-12)


def test_eps_scaled_sq_dist_dim_mismatch_refused(ocp):
    blocks = ((0, 2, 0.5),)
    with pytest.raises(AssertionError):
        ocp.eps_scaled_sq_dist(np.zeros((3, 2)), np.zeros(3), blocks)


# ============================ (b) ground truth: hand-computable ====================

def test_ground_truth_truncated_and_renormalized_hand_computed(ocp):
    """Straight-line synthetic trajectory x_t = (t, 0), goal g=(0,0), 1 block
    eps=0.5, gamma=0.5, horizon=2 (< the trajectory's remaining length so
    TRUNCATION actually binds -- the case production calls never hit given
    the sampler's own >=500-remaining contract, but the function must still
    be exactly right here since ground_truth_batch dispatches short rows
    through this same path).

    idx=0 in an episode [0,4) (rows 0,1,2,3): x1=(1,0), x2=(2,0).
      k=1: d2(x1,g) = (1/0.5)^2 = 4        -> gamma^0 * (-4) = -4
      k=2: d2(x2,g) = (2/0.5)^2 = 16       -> gamma^1 * (-16) = -8
      sum = -12; mass = gamma^0 + gamma^1 = 1.5
      val = -12/1.5 = -8.0
    """
    blocks = ((0, 2, 0.5),)
    coords_all = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    g = np.array([0.0, 0.0])
    val = ocp.ground_truth_discounted_distance(coords_all, blocks, g, idx=0, ep_stop=4,
                                               gamma=0.5, horizon=2)
    assert val == pytest.approx(-8.0, abs=1e-10)


def test_ground_truth_truncated_by_episode_end_not_horizon(ocp):
    """Same trajectory, horizon=10 but the episode only has 2 future rows
    (ep_stop=4, idx=1 -> remaining=2) -- K is capped by the EPISODE, not the
    horizon. k=1: x2=(2,0) d2=16 -> -16; k=2: x3=(3,0) d2=(3/0.5)^2=36 ->
    gamma*-36=-18 (gamma=0.5). sum=-34, mass=1.5, val=-34/1.5."""
    blocks = ((0, 2, 0.5),)
    coords_all = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    g = np.array([0.0, 0.0])
    val = ocp.ground_truth_discounted_distance(coords_all, blocks, g, idx=1, ep_stop=4,
                                               gamma=0.5, horizon=10)
    assert val == pytest.approx(-34.0 / 1.5, abs=1e-10)


def test_ground_truth_requires_at_least_one_future_row(ocp):
    blocks = ((0, 2, 0.5),)
    coords_all = np.zeros((4, 2))
    with pytest.raises(AssertionError):
        ocp.ground_truth_discounted_distance(coords_all, blocks, np.zeros(2), idx=3,
                                             ep_stop=4, gamma=0.5, horizon=5)


def test_ground_truth_batch_matches_scalar_reference_full_and_truncated(ocp):
    """The vectorized batch path (used in production, K==horizon for every
    row under the sampler's own contract) must agree EXACTLY with the scalar
    reference on both a full-horizon row and a row where K<horizon (the
    scalar-splice branch) -- random synthetic multi-episode trajectory."""
    rng = np.random.default_rng(3)
    blocks = ((0, 3, 0.7),)
    n_rows = 40
    coords_all = rng.normal(size=(n_rows, 3))
    g = rng.normal(size=3)
    bounds = np.array([[0, 12], [12, 40]])
    # row 3 in episode0 [0,12): plenty of remaining steps for horizon=5 (full-K).
    # row 9 in episode0 [0,12): only 2 remaining steps for horizon=5 (truncated).
    idxs = np.array([3, 9, 15])
    stops = np.array([12, 12, 40])
    gamma = 0.9
    horizon = 5
    batch = ocp.ground_truth_batch(coords_all, blocks, g, idxs, stops, gamma, horizon=horizon)
    for i, (idx, stop) in enumerate(zip(idxs, stops)):
        ref = ocp.ground_truth_discounted_distance(coords_all, blocks, g, int(idx), int(stop),
                                                    gamma, horizon=horizon)
        assert batch[i] == pytest.approx(ref, rel=1e-10), (i, idx, stop)


def test_ground_truth_batch_empty(ocp):
    out = ocp.ground_truth_batch(np.zeros((3, 2)), ((0, 2, 0.5),), np.zeros(2),
                                 np.zeros(0, np.int64), np.zeros(0, np.int64), 0.9)
    assert out.shape == (0,)


# ============================ (c) index sampling ====================================

def test_sample_valid_indices_respects_min_remaining(ocp):
    """3 episodes [0,10) [10,15) [15,120): with min_remaining=5, valid rows in
    ep0 are [0,4] (row 4 has remaining=10-1-4=5, row 5 would have 4<5), ep1 is
    empty (len 5, last row idx 14, 14-1-5=8 < 10 start, i.e. no row has
    remaining>=5 since the episode itself is only 5 rows), ep2 valid rows are
    [15,114]. Sample n_samples larger than the pool -> returns EVERY valid
    row, sorted, none from ep1."""
    bounds = np.array([[0, 10], [10, 15], [15, 120]])
    idxs, stops = ocp.sample_valid_indices(bounds, seed=0, n_samples=10_000, min_remaining=5)
    assert idxs.min() >= 0
    assert not np.any((idxs >= 10) & (idxs < 15)), "ep1 (too short) must contribute nothing"
    assert not np.all(idxs <= 4)  # ep2 rows are also present
    # every returned row must have >=5 same-trajectory steps remaining
    remaining = stops - 1 - idxs
    assert np.all(remaining >= 5)
    assert np.all(np.isin(idxs, np.concatenate([np.arange(0, 5), np.arange(15, 115)])))


def test_sample_valid_indices_deterministic_given_seed(ocp):
    bounds = np.array([[0, 1000], [1000, 3000]])
    a1, s1 = ocp.sample_valid_indices(bounds, seed=42, n_samples=200, min_remaining=500)
    a2, s2 = ocp.sample_valid_indices(bounds, seed=42, n_samples=200, min_remaining=500)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(s1, s2)
    a3, _ = ocp.sample_valid_indices(bounds, seed=43, n_samples=200, min_remaining=500)
    assert not np.array_equal(a1, a3), "a different seed should (almost certainly) differ"


def test_sample_valid_indices_spreads_across_multiple_trajectories(ocp):
    """10 long episodes, sample more than fit in one episode alone -> the
    result must draw from more than one episode (a degenerate "all from
    episode 0" implementation would fail this)."""
    bounds = np.array([[i * 1000, (i + 1) * 1000] for i in range(10)])
    idxs, stops = ocp.sample_valid_indices(bounds, seed=0, n_samples=2000, min_remaining=500)
    episodes_hit = len(np.unique(stops))
    assert episodes_hit >= 2, f"only touched {episodes_hit} episode(s)"


def test_sample_valid_indices_empty_when_no_row_qualifies(ocp):
    bounds = np.array([[0, 3], [3, 8]])
    idxs, stops = ocp.sample_valid_indices(bounds, seed=0, n_samples=10, min_remaining=500)
    assert len(idxs) == 0 and len(stops) == 0


# ============================ (d) r_W algebra =======================================

def test_rw_from_weights_hand_computed(ocp):
    """r_W = scale*(1-gamma)*sum_j w_j*(-d2_j). w=[0.5,0.3,0.2], d2=[1,4,9],
    gamma=0.9, scale=2.0: sum = 0.5*-1+0.3*-4+0.2*-9 = -0.5-1.2-1.8=-3.5;
    r_W = 2.0*0.1*-3.5 = -0.7."""
    w = np.array([0.5, 0.3, 0.2])
    d2 = np.array([1.0, 4.0, 9.0])
    rw = ocp.rw_from_weights(w, d2, gamma=0.9, scale=2.0)
    assert rw == pytest.approx(-0.7, abs=1e-10)


def test_rw_from_weights_scale_and_gamma_are_positive_multipliers(ocp):
    """Doubling `scale` exactly doubles r_W (both a sanity check on the
    formula and the reason --ors-reward-scale cannot change the Spearman
    gate -- see module docstring)."""
    w = np.array([0.5, 0.3, 0.2])
    d2 = np.array([1.0, 4.0, 9.0])
    r1 = ocp.rw_from_weights(w, d2, gamma=0.9, scale=1.0)
    r2 = ocp.rw_from_weights(w, d2, gamma=0.9, scale=2.0)
    assert r2 == pytest.approx(2.0 * r1, abs=1e-10)


def test_ors_weights_mean_of_heads_not_min(ocp):
    """Spec 19b sec.2.3: w = MEAN of the two F-head softmaxes, NOT
    eval_density_zeroshot.density_weights' elementwise MIN -- construct
    2 heads with DIFFERENT logits so mean != min and pin the exact mean."""
    P = 4
    f_emb = np.array([
        [[10.0, 0.0, 0.0, 0.0]],   # head0: near one-hot on j=0
        [[0.0, 10.0, 0.0, 0.0]],   # head1: near one-hot on j=1
    ])   # [H=2, N=1, d=4] treated as raw "logit-producing" embeddings via identity B
    b_pool = np.eye(4)              # so logits = f_emb . I = f_emb itself
    w = ocp.ors_weights(f_emb, b_pool, tau=1.0)
    assert w.shape == (1, 4)
    from eval_density_zeroshot import softmax_np
    w0 = softmax_np(f_emb[0], axis=-1)
    w1 = softmax_np(f_emb[1], axis=-1)
    np.testing.assert_allclose(w[0], (w0[0] + w1[0]) / 2.0, atol=1e-10)
    # sanity: this differs from the density-readout's MIN convention
    w_min = np.minimum(w0[0], w1[0])
    assert not np.allclose(w[0], w_min)


def test_ors_weights_requires_explicit_head_axis(ocp):
    with pytest.raises(AssertionError):
        ocp.ors_weights(np.zeros(5), np.zeros((3, 5)), tau=1.0)


def test_ors_weights_sums_to_one_per_row(ocp):
    rng = np.random.default_rng(7)
    f_emb = rng.normal(size=(2, 6, 5))
    b_pool = rng.normal(size=(9, 5))
    w = ocp.ors_weights(f_emb, b_pool, tau=0.7)
    np.testing.assert_allclose(w.sum(axis=-1), np.ones(6), atol=1e-8)


# ============================ (e) Spearman gate ======================================

def test_spearman_gate_passes_on_strong_monotonic_relation(ocp):
    rng = np.random.default_rng(0)
    gt = rng.normal(size=500)
    rw = gt * 3.0 + rng.normal(scale=0.01, size=500)   # near-perfect monotonic
    rho, passed = ocp.spearman_gate(rw, gt)
    assert rho > 0.99 and passed


def test_spearman_gate_fails_on_no_relation(ocp):
    rng = np.random.default_rng(1)
    gt = rng.normal(size=500)
    rw = rng.normal(size=500)   # independent
    rho, passed = ocp.spearman_gate(rw, gt)
    assert abs(rho) < 0.2 and not passed


def test_spearman_gate_threshold_boundary(ocp):
    rng = np.random.default_rng(2)
    gt = np.arange(200, dtype=float)
    rw = gt.copy()
    rho, passed = ocp.spearman_gate(rw, gt)
    assert rho == pytest.approx(1.0, abs=1e-9) and passed
    rho2, passed2 = ocp.spearman_gate(rw, gt, threshold=1.1)
    assert not passed2   # threshold above any achievable rho


def test_spearman_gate_degenerate_constant_input(ocp):
    rho, passed = ocp.spearman_gate(np.ones(10), np.arange(10, dtype=float))
    assert rho is None and not passed


# ============================ (f) action-axis grouping ==============================

def test_action_axis_finds_planted_near_duplicate_states(ocp):
    """Plant 2 near-duplicate-state rows (within tol) with clearly different
    actions among a cloud of well-separated states -- must find exactly one
    group of size 2 covering those two rows, and report nonzero coverage."""
    blocks = ((0, 2, 1.0),)
    rng = np.random.default_rng(4)
    coords = rng.normal(scale=10.0, size=(30, 2))  # well-separated
    coords[5] = np.array([0.0, 0.0])
    coords[6] = np.array([0.001, 0.0])   # near-duplicate of row 5 (eps-scaled dist ~0.001)
    actions = rng.normal(size=(30, 3))
    actions[5] = np.array([1.0, 1.0, 1.0])
    actions[6] = np.array([-1.0, -1.0, -1.0])   # clearly different action
    groups = ocp.find_action_axis_groups(coords, actions, blocks,
                                         state_tol=0.05, action_gap=0.5)
    assert len(groups) == 1
    assert set(groups[0].tolist()) == {5, 6}


def test_action_axis_reports_empty_honestly_when_no_duplicates(ocp):
    """Spec sec.1: near-duplicate (s,.) pairs are RARE -- a well-separated
    random cloud (the typical continuous-manipulation case) must report ZERO
    groups, not fabricate any, and this is the EXPECTED/correct outcome."""
    blocks = ((0, 3, 1.0),)
    rng = np.random.default_rng(5)
    coords = rng.normal(scale=10.0, size=(200, 3))
    actions = rng.normal(size=(200, 4))
    groups = ocp.find_action_axis_groups(coords, actions, blocks,
                                         state_tol=0.01, action_gap=0.1)
    assert groups == []


def test_action_axis_drops_near_identical_action_duplicates(ocp):
    """Near-duplicate states whose actions are ALSO near-identical carry no
    action-axis ordering signal and must be dropped (action_gap filter)."""
    blocks = ((0, 2, 1.0),)
    coords = np.array([[0.0, 0.0], [0.0001, 0.0], [50.0, 50.0], [50.0001, 50.0]])
    actions = np.array([[1.0, 0.0], [1.0001, 0.0], [2.0, 0.0], [2.00005, 0.0]])
    groups = ocp.find_action_axis_groups(coords, actions, blocks,
                                         state_tol=0.01, action_gap=0.5)
    assert groups == []


# ============================ (g) consistency refusal ===============================

def test_consistency_refuses_missing_behavior_flag(ocp):
    with pytest.raises(ocp.OrsCalibrationMismatchError):
        ocp.check_behavior_density_consistency(
            {"measure_density": True, "measure_density_behavior": False})


def test_consistency_refuses_missing_density_flag(ocp):
    with pytest.raises(ocp.OrsCalibrationMismatchError):
        ocp.check_behavior_density_consistency(
            {"measure_density": False, "measure_density_behavior": True})


def test_consistency_refuses_plain_nondensity_checkpoint(ocp):
    with pytest.raises(ocp.OrsCalibrationMismatchError):
        ocp.check_behavior_density_consistency({})


def test_consistency_force_override_warns_not_raises(ocp):
    warnings = ocp.check_behavior_density_consistency({}, force=True, run_dir="/x")
    assert len(warnings) == 1 and "FORCE OVERRIDE" in warnings[0]


def test_consistency_accepts_correct_checkpoint_silently(ocp):
    warnings = ocp.check_behavior_density_consistency(
        {"measure_density": True, "measure_density_behavior": True})
    assert warnings == []


# ============================ (h) CLI smoke ==========================================

def test_build_parser_defaults(ocp):
    ap = ocp.build_parser()
    args = ap.parse_args(["--run-dir", "/fake"])
    assert args.n_samples == ocp.DEFAULT_N_SAMPLES == 2000
    assert args.n_random_goals == ocp.DEFAULT_N_RANDOM_GOALS == 20
    assert args.min_remaining == ocp.DEFAULT_MIN_REMAINING == 500
    assert args.horizon == ocp.DEFAULT_HORIZON == 500
    assert args.pool_size == ocp.DEFAULT_POOL_SIZE == 4096
    assert args.ors_reward_scale == 1.0
    assert ocp.GATE_SPEARMAN == 0.65


# ============================ (B) jax-dependent plumbing =============================

OBS_D = 40
D = 8

JAX_COMMON = f"""
    import sys, jax, jax.numpy as jnp, numpy as np
    sys.path.insert(0, {str(REPO / "scripts")!r})
    sys.path.insert(0, {str(REPO / "scripts" / "rescuing_fb")!r})
    sys.path.insert(0, {str(REPO)!r})
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec
    from ors_calibration_probe import build_ors_pool, compute_f_emb, ors_weights, \\
        eps_scaled_sq_dist, rw_from_weights

    OBS_D = {OBS_D}
    D = {D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def behavior_cfg(**kw):
        # mirrors tests/test_gciql_measure_density.py's base_cfg, extended for
        # spec 19a's measure_density_behavior (actor_q_center must be False here --
        # the opposite of plain measure_density's own requirement).
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["actor_q_center"] = False
        cfg["measure_density"] = True
        cfg["measure_density_behavior"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg.update(kw)
        return cfg

    def make_agent(seed=0):
        cfg = behavior_cfg()
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg), cfg
"""


def _run(body, **kw):
    code = textwrap.dedent(JAX_COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_agent_creation_registers_density_f_b_no_goal_leg_shapes():
    """A measure_density_behavior=True agent registers density_f/density_b
    (and target twins), and density_f's input dim is [s,a] (OBS_D+5), NOT
    [s,a,g] (OBS_D+5+OBS_D) -- the spec-19a no-goal-leg convention this
    probe's compute_f_emb relies on."""
    _run("""
    agent, cfg = make_agent()
    for mod in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b"):
        assert mod in agent.network.params, mod

    blocks = _parse_hit_spec(cfg["measure_hit_spec"])
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(6, OBS_D)).astype(np.float32)
    acts = np.clip(rng.normal(size=(6, 5)), -1, 1).astype(np.float32)
    pool_obs = rng.normal(size=(10, OBS_D)).astype(np.float32)

    u_raw, b_pool = build_ors_pool(agent, blocks, pool_obs)
    assert u_raw.shape == (10, 3) and b_pool.shape == (10, D)

    f_emb = compute_f_emb(agent, obs, acts)
    assert f_emb.shape == (2, 6, D), f_emb.shape   # 2 heads, no goal leg blew up the dim

    w = ors_weights(f_emb, b_pool, tau=cfg["measure_density_temp"])
    assert w.shape == (6, 10)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-5)
    print("OK")
    """)


def test_f_emb_matches_direct_module_call_numpy_reference():
    """compute_f_emb's output must equal a DIRECT call to
    agent.network.select('density_f') on the SAME concatenated [s,a] input --
    proves the wrapper is not silently mis-assembling the input (e.g.
    accidentally including a goal leg)."""
    _run("""
    agent, cfg = make_agent()
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(4, OBS_D)).astype(np.float32)
    acts = np.clip(rng.normal(size=(4, 5)), -1, 1).astype(np.float32)

    f_emb = compute_f_emb(agent, obs, acts)
    ref = np.asarray(agent.network.select("density_f")(
        jnp.concatenate([jnp.asarray(obs), jnp.asarray(acts)], -1)), np.float64)
    assert np.allclose(f_emb, ref, atol=1e-10)
    print("OK")
    """)


def test_determinism_repeat_run_is_bit_identical():
    """Spec-18-lesson determinism check (module docstring 'repeat run as the
    noise floor'): build the SAME agent + pool + f_emb + ors_weights +
    r_W TWICE with identical seeds/inputs -- must be EXACTLY (bitwise)
    identical, not merely close."""
    _run("""
    agent1, cfg1 = make_agent(seed=0)
    agent2, cfg2 = make_agent(seed=0)
    blocks = _parse_hit_spec(cfg1["measure_hit_spec"])

    rng = np.random.default_rng(2)
    obs = rng.normal(size=(5, OBS_D)).astype(np.float32)
    acts = np.clip(rng.normal(size=(5, 5)), -1, 1).astype(np.float32)
    pool_obs = rng.normal(size=(12, OBS_D)).astype(np.float32)
    goal = rng.normal(size=3).astype(np.float64)

    def full_pipeline(agent, cfg):
        u_raw, b_pool = build_ors_pool(agent, blocks, pool_obs)
        f_emb = compute_f_emb(agent, obs, acts)
        w = ors_weights(f_emb, b_pool, tau=cfg["measure_density_temp"])
        d2 = eps_scaled_sq_dist(u_raw, goal, blocks)
        rw = rw_from_weights(w, d2, gamma=cfg["discount"], scale=1.0)
        return u_raw, b_pool, f_emb, w, rw

    out1 = full_pipeline(agent1, cfg1)
    out2 = full_pipeline(agent2, cfg2)
    for a, b in zip(out1, out2):
        assert np.array_equal(a, b), "repeat run diverged -- expected bit-identical"

    # a genuinely different seed must NOT reproduce the same params (sanity that
    # this test isn't vacuously true because the agent ignores its own params)
    agent3, cfg3 = make_agent(seed=99)
    out3 = full_pipeline(agent3, cfg3)
    assert not np.array_equal(out1[-1], out3[-1]), \\
        "a different seed produced identical r_W -- params likely not varying"
    print("OK")
    """)


def test_default_off_bit_identical_param_tree():
    """measure_density_behavior=False (the default): no density_f/b modules
    registered at all, mirroring test_gciql_measure_density.py's own
    default-off discipline -- this probe script must never be blamed for a
    training-side regression, so its own harness re-confirms the off-path
    here too."""
    _run("""
    cfg_off = dict(get_config())
    cfg_off["encoder"] = None; cfg_off["frame_stack"] = None
    agent_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_off)
    for mod in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b"):
        assert mod not in agent_off.network.params, mod
    print("OK")
    """)
