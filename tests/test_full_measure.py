"""Unit tests for the eval-layer full successor measure (spec 10 L1+L2).

Ground truth here is analytic: a synthetic dataset with known density and a fake
ball-value oracle M-tilde(x) = f(x) * N-true(x), so the density layer must
recover f, set masses must integrate f over regions, and the corrected moment
must equal the analytic reweighted average.
"""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rescuing_fb"))

from full_measure import FullMeasure, select_z_by_measure


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    # two clusters with a 4:1 density ratio + uniform background
    a = rng.normal([0, 0, 0], 0.15, (8000, 3))
    b = rng.normal([2, 0, 0], 0.15, (2000, 3))
    u = rng.uniform(-1, 3, (2000, 3)) * [1, 0.3, 0.3]
    cube = np.concatenate([a, b, u])
    fm = FullMeasure(cube, eps=0.2, seed=0)
    # smooth "true density" of the successor measure w.r.t. rho
    f = lambda x: 1.0 + np.asarray(x)[..., 0]          # linear in x-coordinate
    # the oracle ball value = f * data ball mass (same kernel!)
    oracle = lambda idx: f(cube[idx]) * fm.nhat(cube[idx])
    return fm, cube, f, oracle


def test_nhat_matches_bruteforce(world):
    fm, cube, _, _ = world
    q = cube[::1500]
    brute = np.array([(np.linalg.norm(cube - c, axis=1) <= 0.2).mean() for c in q])
    assert np.allclose(fm.nhat(q), np.maximum(brute, fm.floor), atol=1e-12)


def test_density_recovers_f(world):
    fm, cube, f, oracle = world
    idx = np.arange(0, len(cube), 700)
    dens = fm.density(oracle(idx), cube[idx])
    assert np.allclose(dens, f(cube[idx]), rtol=1e-9)


def test_set_mass_matches_integral_and_is_additive(world):
    fm, cube, f, oracle = world
    ball1 = dict(kind="ball", center=[0.0, 0.0, 0.0], eps=0.3)
    ball2 = dict(kind="ball", center=[2.0, 0.0, 0.0], eps=0.3)
    m1 = fm.set_mass(oracle, fm.member_indices(ball1))
    m2 = fm.set_mass(oracle, fm.member_indices(ball2))
    mu = fm.set_mass(oracle, fm.member_indices(
        dict(kind="union", parts=[ball1, ball2])))
    # integral check: M-hat(X) ~ (1/n) sum_{x_i in X} f(x_i)  (exact when all
    # members evaluated; here subsampled -> loose tolerance)
    i1 = fm.member_indices(ball1)
    truth1 = f(cube[i1]).sum() / fm.n
    assert m1["mass"] == pytest.approx(truth1, rel=0.15)
    # additivity over the disjoint union
    assert mu["mass"] == pytest.approx(m1["mass"] + m2["mass"], rel=0.15)


def test_box_and_union_members(world):
    fm, cube, _, _ = world
    box = dict(kind="box", center=[0, 0, 0], half=[0.3, 0.3, 0.3])
    idx = fm.member_indices(box)
    assert np.all(np.abs(cube[idx]) <= 0.3 + 1e-12)
    assert len(idx) > 50


def test_q_r_matches_reweighted_mean(world):
    fm, cube, f, oracle = world
    rng = np.random.default_rng(1)
    sample = rng.integers(0, len(cube), 500)
    r = rng.normal(0, 1, 500)
    r[rng.random(500) < 0.5] = 0.0                     # sparse + signed
    got = fm.q_r(oracle, sample, r)
    truth = (r * f(cube[sample])).mean()
    assert got["q"] == pytest.approx(truth, rel=1e-9, abs=1e-12)
    assert got["n_live"] == int((r != 0).sum())


def test_corrected_moment_z(world):
    fm, cube, _, _ = world
    rng = np.random.default_rng(2)
    m = 400
    idx = rng.integers(0, len(cube), m)
    Psi = rng.normal(0, 1, (m, 8))
    r = rng.random(m)
    out = fm.corrected_moment_z(Psi, r, cube[idx], match_norm=2.0)
    zt = ((r / fm.nhat(cube[idx]))[:, None] * Psi).mean(0)
    zt = zt * 2.0 / np.linalg.norm(zt)
    assert np.allclose(out["z"], zt, atol=1e-5)
    assert 1.0 <= out["ess"] <= m


def test_nhat_floor_bounds_inverse(world):
    fm, cube, _, _ = world
    far = np.array([[50.0, 50.0, 50.0]])               # zero support
    assert fm.nhat(far)[0] == fm.floor
    assert fm.floor > 0


def test_select_z_by_measure():
    cands = [("a", np.array([1.0])), ("b", np.array([2.0])), ("c", np.array([0.5]))]
    out = select_z_by_measure(cands, lambda z: -abs(float(z[0]) - 1.9))
    assert out["chosen"] == "b" and set(out["scores"]) == {"a", "b", "c"}
