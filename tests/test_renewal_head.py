"""Spec 07 item 2 unit tests: renewal_head.py math (MC renewal targets, entry
detection, FP_mu censoring, infer_z, bundle roundtrip). Pure numpy -- no jax,
no checkpoint, no dataset: renewal_head's module top level is numpy-only (jax
stays inside its fit/eval functions), so this runs in the repo .venv."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

import renewal_head as rh  # noqa: E402


def _traj(positions_1d):
    """Lift a 1-D coordinate list to [T,3] cube positions (y=z=0)."""
    p = np.zeros((len(positions_1d), 3))
    p[:, 0] = positions_1d
    return p


# ---------- (a) MC renewal target matches the closed form on a cyclic chain ----------

def test_renewal_target_matches_closed_form_cyclic():
    """Deterministic cyclic trajectory: in the ball exactly at t=0,5,10 (period 5)
    within one 15-step episode. Entries at 0,5,10 (isolated visits => 3 entries);
    per-entry discounted revisit mass hand-computed:
        e=0 : g^0 + g^5 + g^10
        e=5 : g^0 + g^5
        e=10: g^0
    target = mean of the three."""
    g = 0.9
    pos = [10.0] * 15
    for t in (0, 5, 10):
        pos[t] = 0.0
    cube = _traj(pos)
    tgts, nent = rh.renewal_targets(cube, [(0, 15)], _traj([0.0]), gamma=g, eps=0.1)
    assert nent[0] == 3
    expected = np.mean([1 + g**5 + g**10, 1 + g**5, 1.0])
    assert np.isclose(tgts[0], expected, atol=1e-12), (tgts[0], expected)


def test_renewal_target_respects_episode_boundaries():
    """A visit at the last row of episode 1 and the first row of episode 2:
    consecutive ROWS but different episodes => TWO entries, each with mass 1
    (the discounted sum truncates at the episode end; no cross-episode credit)."""
    g = 0.9
    pos = [10.0] * 12
    pos[5] = 0.0   # last row of episode (0,6)
    pos[6] = 0.0   # first row of episode (6,12)
    cube = _traj(pos)
    tgts, nent = rh.renewal_targets(cube, [(0, 6), (6, 12)], _traj([0.0]),
                                    gamma=g, eps=0.1)
    assert nent[0] == 2
    assert np.isclose(tgts[0], 1.0, atol=1e-12)


def test_renewal_target_no_entry_is_nan():
    tgts, nent = rh.renewal_targets(_traj([10.0] * 8), [(0, 8)], _traj([0.0]),
                                    gamma=0.9, eps=0.1)
    assert nent[0] == 0 and np.isnan(tgts[0])


# ---------- (b) entry detection: consecutive in-ball steps = ONE entry ----------

def test_consecutive_inball_steps_are_one_entry():
    """In the ball for t=2,3,4 of an 8-step episode: ONE entry (a first passage),
    not three, with mass 1 + g + g^2."""
    g = 0.9
    pos = [10.0] * 8
    for t in (2, 3, 4):
        pos[t] = 0.0
    cube = _traj(pos)
    tgts, nent = rh.renewal_targets(cube, [(0, 8)], _traj([0.0]), gamma=g, eps=0.1)
    assert nent[0] == 1, "consecutive in-ball steps must collapse to one entry"
    assert np.isclose(tgts[0], 1 + g + g**2, atol=1e-12)
    # helper agrees
    inball = np.array([0, 0, 1, 1, 1, 0, 0, 0], bool)
    assert list(rh._ball_entries(inball)) == [2]


def test_entry_at_episode_start_counts():
    inball = np.array([1, 1, 0, 1], bool)
    assert list(rh._ball_entries(inball)) == [0, 3]


# ---------- (c) infer_z: renorm exact, renewal direction, None => plain ----------

def test_infer_z_renormalization_exact():
    psi = np.array([[1.0, 0.0], [0.0, 1.0]])
    z = rh.infer_z(psi, rewards=[1.0, 1.0], R_vals=None, match_norm=2.0)
    assert np.isclose(np.linalg.norm(z), 2.0, atol=1e-12)
    # plain moment direction: mean of e1,e2 => [1,1]/sqrt(2) * 2
    assert np.allclose(z, np.array([np.sqrt(2.0), np.sqrt(2.0)]), atol=1e-12)


def test_infer_z_renewal_weighting_tilts_and_none_reduces_to_plain():
    psi = np.array([[1.0, 0.0], [0.0, 1.0]])
    r = np.array([1.0, 1.0])
    plain = rh.infer_z(psi, r, None, match_norm=1.0)
    ones = rh.infer_z(psi, r, np.array([1.0, 1.0]), match_norm=1.0)
    assert np.allclose(plain, ones, atol=1e-12), "R_vals=1 must equal the plain moment"
    assert np.isclose(plain[0], plain[1], atol=1e-12)
    # R larger on goal 1 => z tilts toward psi(g1)=e1
    weighted = rh.infer_z(psi, r, np.array([4.0, 1.0]), match_norm=1.0)
    assert weighted[0] > weighted[1], "renewal weighting must tilt z toward high-R goals"
    assert np.isclose(np.linalg.norm(weighted), 1.0, atol=1e-12)
    # exact direction: mean_j r_j R_j psi_j = [2, 0.5] normalized
    exp = np.array([2.0, 0.5]) / np.linalg.norm([2.0, 0.5])
    assert np.allclose(weighted, exp, atol=1e-12)


def test_infer_z_zero_moment_degenerate():
    psi = np.array([[1.0, 0.0], [-1.0, 0.0]])
    z = rh.infer_z(psi, [1.0, 1.0], None, match_norm=3.0)
    assert np.allclose(z, 0.0), "all-zero moment stays zero (no renorm blowup)"


# ---------- (d) FP_mu targets: censoring + first-hit discount ----------

def test_fp_target_never_hits_is_censored_zero():
    cube = _traj([10.0] * 10)
    tg, cens = rh.fp_targets(cube, [(0, 10)], q_rows=[0], goals_xyz=_traj([0.0]),
                             gamma=0.99, eps=0.4)
    assert tg[0] == 0.0 and bool(cens[0]), "no hit => target 0, counted as censored"


def test_fp_target_first_hit_discount():
    g = 0.99
    pos = [10.0] * 10
    pos[3] = 0.0   # from t=0: first hit at k=3 => T=3, target g^(T-1)=g^2
    pos[7] = 0.0   # a later revisit must NOT change the FIRST-hit target
    cube = _traj(pos)
    tg, cens = rh.fp_targets(cube, [(0, 10)], q_rows=[0], goals_xyz=_traj([0.0]),
                             gamma=g, eps=0.4)
    assert not cens[0]
    assert np.isclose(tg[0], g**2, atol=1e-12)


def test_fp_target_inball_at_query_does_not_count():
    """T >= 1 convention: being in the ball AT the query step is not a hit."""
    pos = [0.0] + [10.0] * 5
    cube = _traj(pos)
    tg, cens = rh.fp_targets(cube, [(0, 6)], q_rows=[0], goals_xyz=_traj([0.0]),
                             gamma=0.99, eps=0.4)
    assert tg[0] == 0.0 and bool(cens[0])


def test_fp_target_stays_within_episode():
    """Hit exists only in the NEXT episode => censored for a query in episode 1."""
    pos = [10.0] * 5 + [0.0] + [10.0] * 4
    cube = _traj(pos)
    tg, cens = rh.fp_targets(cube, [(0, 5), (5, 10)], q_rows=[0],
                             goals_xyz=_traj([0.0]), gamma=0.99, eps=0.4)
    assert tg[0] == 0.0 and bool(cens[0])


# ---------- bundle roundtrip: save -> load_renewal_head (numpy path) ----------

def _random_bundle(in_dim, seed=0):
    rng = np.random.default_rng(seed)
    dims = (in_dim, 256, 256, 1)
    b = {}
    for li in range(3):
        b[f"W{li}"] = rng.normal(size=(dims[li], dims[li + 1])) / np.sqrt(dims[li])
        b[f"b{li}"] = np.zeros(dims[li + 1])
    b.update(x_mean=np.zeros(in_dim), x_std=np.ones(in_dim),
             y_mean=np.float64(0.5), y_std=np.float64(2.0))
    return b


def test_bundle_roundtrip_and_cube_head_callable(tmp_path):
    bundle = _random_bundle(3)
    meta = {"kind": "renewal", "input": "cube", "epsilon": 0.4, "gamma": 0.99,
            "run_dir": "", "checkpoint_step": 0}
    path = tmp_path / "head.npz"
    rh.save_head_bundle(path, bundle, meta)
    R = rh.load_renewal_head(path)
    obs = np.random.default_rng(1).normal(size=(7, 29)).astype(np.float32)
    out = R(obs)
    assert out.shape == (7,)
    assert (out >= 0.0).all(), "R predictions must be clamped at 0"
    # matches the raw forward on the cube slice, clamped
    ref = np.maximum(rh.predict_from_bundle(bundle, obs[:, rh.CUBE_OBS]), 0.0)
    assert np.allclose(out, ref, atol=1e-10)
    # already-sliced [N,3] input is accepted and identical
    assert np.allclose(R(obs[:, rh.CUBE_OBS]), out, atol=1e-10)
    # fp_mu loader refuses a renewal bundle and vice versa
    with pytest.raises(AssertionError):
        rh.load_fp_mu_head(path)


def test_mlp_forward_shape_and_gelu_sanity():
    bundle = _random_bundle(5, seed=2)
    params = {k: bundle[k] for k in ("W0", "b0", "W1", "b1", "W2", "b2")}
    x = np.random.default_rng(0).normal(size=(11, 5))
    y = rh.mlp_forward(params, x)
    assert y.shape == (11,)
    assert np.isfinite(y).all()
    # tanh-GELU spot values: gelu(0)=0, gelu(large)~large, gelu(-large)~0
    assert rh._gelu(np.array(0.0)) == 0.0
    assert np.isclose(rh._gelu(np.array(10.0)), 10.0, atol=1e-6)
    assert np.isclose(rh._gelu(np.array(-10.0)), 0.0, atol=1e-6)


def test_meta_json_preserved(tmp_path):
    bundle = _random_bundle(3, seed=3)
    meta = {"kind": "renewal", "input": "cube", "epsilon": 0.25, "gamma": 0.97,
            "run_dir": "/nonexistent", "checkpoint_step": 123}
    path = tmp_path / "m.npz"
    rh.save_head_bundle(path, bundle, meta)
    data = np.load(path)
    assert json.loads(str(data["meta_json"])) == meta
