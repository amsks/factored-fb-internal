"""Tests for scripts/rescuing_fb/eval_smb.py + eval_smb_measure.py + the
rollout-scoring / calibration pieces of smb_common.py (spec 11 items 5.2-5.3).
Structure + math only -- no env, no checkpoint, no jax: reward families on
synthetic cube tracks, achieved-discounted-reward with padding, the Wing-B
metrics (affine fit, relative error, additivity defect, min-head readout
math), z-inference route parity with the pinned renewal_head interface, and
the two CLIs' --help contracts (subprocess)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "rescuing_fb"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


@pytest.fixture(scope="module")
def smc():
    return _load("smb_common")


# ============================ reward families ====================================

def test_reward_occhoice_weights_and_disjoint(smc):
    gS = np.array([0.0, 0.0, 0.0])
    gT = np.array([1.0, 1.0, 0.8])
    cube = np.array([gS, gT, [5.0, 5.0, 5.0], gS + [0.39, 0, 0]])
    r = smc.reward_occhoice(cube, gS, gT, w_s=1.0, w_t=4.0)
    np.testing.assert_allclose(r, [1.0, 4.0, 0.0, 1.0])
    # weights scale linearly
    np.testing.assert_allclose(smc.reward_occhoice(cube, gS, gT, 2.0, 6.0),
                               [2.0, 6.0, 0.0, 2.0])


def test_reward_union(smc):
    cs = [np.zeros(3), np.array([2.0, 0, 0]), np.array([0, 2.0, 0])]
    cube = np.array([[0, 0, 0], [2.0, 0, 0], [0, 2.0, 0], [1.0, 1.0, 0]])
    np.testing.assert_allclose(smc.reward_union(cube, cs), [1, 1, 1, 0])


def test_reward_corridor_signs_and_lambda(smc):
    gA = np.zeros(3)
    cB = np.array([2.0, 0, 0])
    cube = np.array([gA, cB, [5.0, 5, 5]])
    np.testing.assert_allclose(smc.reward_corridor(cube, gA, cB, lam=1), [1, -1, 0])
    np.testing.assert_allclose(smc.reward_corridor(cube, gA, cB, lam=5), [1, -5, 0])
    np.testing.assert_allclose(smc.reward_corridor(cube, gA, cB, lam=0), [1, 0, 0])
    # avoid ball uses AVOID_R = 0.5, goal ball uses eps = 0.4
    edge = np.array([[0.45, 0, 0]])                    # outside goal, not near cB
    np.testing.assert_allclose(smc.reward_corridor(edge + [2.0, 0, 0] - [2.0, 0, 0],
                                                   gA, cB, 1), [0.0])
    near_b = cB + [0.49, 0, 0]
    np.testing.assert_allclose(smc.reward_corridor(near_b[None], gA, cB, 1), [-1.0])


def test_query_member_dispatch(smc):
    cube = np.array([[0, 0, 0], [0.9, 0, 0], [2.0, 0, 0]])
    ball = {"type": "ball", "center": [0, 0, 0], "eps": 0.4}
    box = {"type": "box", "center": [1.0, 0, 0], "half": [0.2, 0.5, 0.5]}
    uni = {"type": "union2", "centers": [[0, 0, 0], [2.0, 0, 0]], "eps": 0.4}
    np.testing.assert_array_equal(smc.query_member(ball, cube), [True, False, False])
    np.testing.assert_array_equal(smc.query_member(box, cube), [False, True, False])
    np.testing.assert_array_equal(smc.query_member(uni, cube), [True, False, True])
    with pytest.raises(ValueError):
        smc.query_member({"type": "cone"}, cube)


# ============================ trajectory scoring =================================

def test_per_episode_discounted_padding_and_lengths(smc):
    g = smc.GAMMA
    ctr = np.zeros(3)
    far = np.array([9.0, 9, 9])
    traj = np.zeros((2, 5, 3))
    traj[0] = [ctr, far, ctr, far, ctr]     # last step is PADDING (length 4)
    traj[1] = [far, far, far, far, far]
    lengths = [4, 0]                         # ep1: zero real steps -> 0
    rfn = lambda c: smc.in_ball(c, ctr).astype(float)  # noqa: E731
    out = smc.per_episode_discounted(traj, lengths, rfn)
    np.testing.assert_allclose(out, [1 + g ** 2, 0.0])
    # dwell == discounted indicator time
    np.testing.assert_allclose(
        smc.discounted_time_in(traj, lengths, lambda c: smc.in_ball(c, ctr)), out)


def test_frac_time_and_final_in_any_ball(smc):
    ctr = np.zeros(3)
    far = np.array([9.0, 9, 9])
    traj = np.zeros((2, 4, 3))
    traj[0] = [ctr, far, ctr, ctr]          # ends at ctr
    traj[1] = [ctr, far, far, far]          # ends far (but visited)
    lengths = [4, 4]
    np.testing.assert_allclose(
        smc.frac_time_in_ball(traj, lengths, ctr, 0.4), [0.75, 0.25])
    np.testing.assert_allclose(
        smc.final_in_any_ball(traj, lengths, [ctr]), [1.0, 0.0])
    # respects lengths: episode truncated before the final ctr visit
    np.testing.assert_allclose(smc.final_in_any_ball(traj, [2, 1], [ctr]), [0.0, 1.0])


def test_chosen_region(smc):
    gS = np.zeros(3)
    gT = np.array([2.0, 0, 0])
    far = np.array([9.0, 9, 9])
    traj = np.zeros((3, 4, 3))
    traj[0] = [gS, gS, gS, gT]              # mostly stable
    traj[1] = [far, gT, gT, gT]             # transient
    traj[2] = [far, far, far, far]          # none
    labels, dS, dT = smc.chosen_region(traj, [4, 4, 4], gS, gT)
    assert labels == ["stable", "transient", "none"]
    assert dS[0] > dT[0] and dT[1] > dS[1] and dS[2] == dT[2] == 0.0


def test_cube_m_to_ou_roundtrip(smc):
    m = np.array([[0.425, 0.0, 0.0], [0.5, -0.1, 0.08]])
    ou = smc.cube_m_to_ou(m)
    np.testing.assert_allclose(ou[0], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(smc.ou_to_m(ou), m, atol=1e-12)


# ============================ Wing-B metrics =====================================

def test_affine_fit_and_median_rel_err(smc):
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 10, 100)
    gt = 3.0 * pred + 2.0
    a, b = smc.affine_fit(pred, gt)
    assert a == pytest.approx(3.0) and b == pytest.approx(2.0)
    err, n = smc.median_rel_err(a * pred + b, gt)
    assert n == 100 and err == pytest.approx(0.0, abs=1e-9)
    err_raw, _ = smc.median_rel_err(pred, gt)
    assert err_raw > 0.3                       # raw is badly off (the B1 read)
    # gt at/below floor is excluded
    err0, n0 = smc.median_rel_err(np.ones(3), np.zeros(3))
    assert err0 is None and n0 == 0
    # degenerate fit falls back to identity
    assert smc.affine_fit(np.ones(5), np.arange(5.0)) == (1.0, 0.0)


def test_additivity_defect(smc):
    assert smc.additivity_defect(5.0, 2.0, 3.0, gt_union=10.0) == 0.0
    assert smc.additivity_defect(4.0, 2.0, 3.0, gt_union=10.0) == pytest.approx(0.1)
    # floor guards the empty-gt division
    assert smc.additivity_defect(1.0, 0.0, 0.0, gt_union=0.0) == pytest.approx(1000.0)


def test_regression_slope_r2(smc):
    x = np.arange(10.0)
    slope, b, r2 = smc.regression_slope_r2(x, 2 * x + 1)
    assert slope == pytest.approx(2.0) and b == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)
    s_nan, _, _ = smc.regression_slope_r2(np.ones(5), np.arange(5.0))
    assert np.isnan(s_nan)


def test_pair_min_head_dot(smc):
    d = 4
    # 2 heads: head 0 gives large values, head 1 small -> min must pick head 1
    phi = np.stack([np.ones((3, d)) * 2.0, np.ones((3, d))])          # [2,3,4]
    psi = np.stack([np.ones((5, d)), np.ones((5, d)) * 0.5])          # [2,5,4]
    out = smc.pair_min_head_dot(phi, psi, d)
    assert out.shape == (3, 5)
    np.testing.assert_allclose(out, 0.5 * d / np.sqrt(d))             # head-1 dot
    # unensembled [N,d] x [G,d] broadcast
    out2 = smc.pair_min_head_dot(np.ones((2, d)), np.ones((3, d)) * 2, d)
    np.testing.assert_allclose(out2, 2 * d / np.sqrt(d))
    # head-wise pairing (NOT min over cross terms): head0 big*small = head1 small*big
    phi_x = np.stack([np.ones((1, d)) * 4.0, np.ones((1, d))])
    psi_x = np.stack([np.ones((1, d)), np.ones((1, d)) * 4.0])
    np.testing.assert_allclose(smc.pair_min_head_dot(phi_x, psi_x, d),
                               4.0 * d / np.sqrt(d))


# ============================ z-inference routes =================================

def test_infer_z_route_moment_matches_pinned_interface(smc):
    from renewal_head import infer_z
    rng = np.random.default_rng(0)
    Psi = rng.normal(size=(50, 8))
    r = rng.uniform(0, 1, 50)
    z, diag = smc.infer_z_route(Psi, r, "moment", match_norm=3.0)
    np.testing.assert_allclose(z, infer_z(Psi, r, None, 3.0).astype(np.float32))
    assert diag["route"] == "moment" and diag["n_pos"] == 50


def test_infer_z_route_retrieval_on_manifold(smc):
    Psi = np.eye(4)
    r = np.array([0.0, 0.0, 1.0, 0.0])
    z, diag = smc.infer_z_route(Psi, r, "retrieval", match_norm=2.0, retrieval_k=1)
    np.testing.assert_allclose(z, [0, 0, 2.0, 0], atol=1e-6)   # = matchnorm psi_2
    assert diag["retrieval_n"] == 1


def test_infer_z_route_degenerate_and_ridge_norm(smc):
    z, diag = smc.infer_z_route(np.ones((5, 3)), np.zeros(5), "moment", 1.0)
    assert z is None and diag["z_raw_norm"] == 0.0
    rng = np.random.default_rng(1)
    z2, _ = smc.infer_z_route(rng.normal(size=(500, 6)), rng.uniform(0, 1, 500),
                              "ridge", match_norm=3.5)
    assert np.linalg.norm(z2) == pytest.approx(3.5, rel=1e-4)


def test_ball_candidate_indices(smc):
    cube = np.array([[0, 0, 0], [0.1, 0, 0], [0.3, 0, 0], [5.0, 0, 0]], float)
    idxs, ok = smc.ball_candidate_indices(cube, np.zeros(3), eps=0.4, k=2)
    assert ok and idxs.tolist() == [0, 1]          # nearest-first, capped at k
    idxs2, ok2 = smc.ball_candidate_indices(cube, np.array([10.0, 0, 0]), eps=0.4)
    assert not ok2 and idxs2.tolist() == [3]       # fallback: single nearest


# ============================ CLI contracts ======================================

def test_eval_smb_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPTS / "eval_smb.py"), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--checkpoint-step", "--tasks-json", "--tiers",
                 "--episodes", "--horizon", "--z-inference", "--gpi",
                 "--oracle-z", "--out"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_eval_smb_parser_defaults():
    esm = _load("eval_smb")
    ns = esm.build_parser().parse_args(
        ["--run-dir", "/tmp/x", "--tasks-json", "/tmp/t.json"])
    assert ns.episodes == 25 and ns.horizon == 200
    assert ns.z_inference == "moment" and ns.gpi is False and ns.oracle_z is False
    assert set(ns.tiers.split(",")) == set(esm.ALL_TIERS)
    assert "t0" not in esm.ALL_TIERS               # native comes from t6_gpi_eval


def test_eval_smb_measure_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPTS / "eval_smb_measure.py"),
                           "--help"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--checkpoint-step", "--tasks-json",
                 "--conditionings", "--episodes", "--horizon", "--out"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_eval_smb_measure_b3_tasks():
    esm = _load("eval_smb_measure")
    art = {"t3_occhoice": {"pairs": [
        {"name": "pair0", "stable": {"center": [0, 0, 0]},
         "transient": {"center": [1, 1, 0.8]}, "w_S": 1.0, "w_T": 4.0}]},
        "t4_union3": {"tasks": [
            {"name": "union0", "centers": [[0, 0, 0], [2, 0, 0], [0, 2, 0]]}]}}
    goal_ou = {1: np.zeros(3), 2: np.array([2.0, 0, 0])}
    tasks = esm.build_b3_tasks(art, goal_ou)
    names = [n for n, _ in tasks]
    assert names == ["t1_dirac_t1", "t1_dirac_t2", "t3_pair0", "t4_union0"]
    cube = np.array([[0, 0, 0], [1, 1, 0.8], [9, 9, 9]])
    fns = dict(tasks)
    np.testing.assert_allclose(fns["t3_pair0"](cube), [1.0, 4.0, 0.0])
    np.testing.assert_allclose(fns["t4_union0"](cube), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(fns["t1_dirac_t1"](cube), [1.0, 0.0, 0.0])
