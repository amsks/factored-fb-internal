"""Spec 14 §3.2 — FullMeasure over conjunctive multi-block regions.

The exactness claim under test: a Chebyshev KDTree on per-block-scaled
coordinates returns a strict SUPERSET of the conjunctive region (per-block
L2 <= 1 implies every coordinate <= 1 implies L-inf <= 1), so filtering those
candidates exactly reproduces a brute-force scan.
"""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rescuing_fb"))

from full_measure import FullMeasure
from env_geometry import HitSpec


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    coords = rng.uniform(-2, 2, (4000, 5))
    spec = HitSpec(((0, 3, 0.4), (3, 5, 0.7)))   # block widths 3 and 2 = dim 5
    return FullMeasure(coords, spec=spec, seed=0), coords, spec


def test_nhat_matches_bruteforce_conjunction(world):
    fm, coords, spec = world
    q = coords[::400]
    brute = np.array([spec.contains(coords, c).mean() for c in q])
    assert np.allclose(fm.nhat(q), np.maximum(brute, fm.floor), atol=1e-12)


def test_member_indices_ball_matches_bruteforce(world):
    fm, coords, spec = world
    for mult in (1.0, 2.0, 4.0):
        c = coords[7]
        got = fm.member_indices({"kind": "ball", "center": c, "mult": mult})
        want = np.flatnonzero(spec.contains(coords, c, mult=mult))
        assert np.array_equal(got, want), f"mult={mult}"


def test_member_indices_box_matches_bruteforce(world):
    fm, coords, spec = world
    c = coords[11]
    got = fm.member_indices({"kind": "box", "center": c, "mult": 2.0})
    want = np.flatnonzero(spec.box_contains(coords, c, mult=2.0))
    assert np.array_equal(got, want)


def test_conjunction_is_not_a_joint_ball(world):
    """A point far in block 0 but near in block 1 must be excluded even when
    its joint L2 distance is small — the failure mode a naive port would have."""
    fm, coords, spec = world
    c = np.zeros(5)
    idx = fm.member_indices({"kind": "ball", "center": c, "mult": 1.0})
    assert np.all(np.linalg.norm(coords[idx][:, :3] - c[:3], axis=1) <= 0.4 + 1e-12)
    assert np.all(np.linalg.norm(coords[idx][:, 3:] - c[3:], axis=1) <= 0.7 + 1e-12)


def test_single_block_spec_equals_legacy_scalar_eps():
    """Backwards compatibility: the archived cube-single protocol is a
    one-block spec and must give bit-identical membership and N-hat."""
    rng = np.random.default_rng(1)
    cube = rng.normal(0, 0.5, (3000, 3))
    legacy = FullMeasure(cube, eps=0.4, seed=0)
    modern = FullMeasure(cube, spec=HitSpec.single(0, 3, 0.4), seed=0)
    q = cube[::250]
    assert np.allclose(legacy.nhat(q), modern.nhat(q), atol=0, rtol=0)
    for c in q[:5]:
        a = legacy.member_indices({"kind": "ball", "center": c, "eps": 0.4})
        b = modern.member_indices({"kind": "ball", "center": c, "mult": 1.0})
        assert np.array_equal(a, b)


def test_legacy_half_key_still_supported():
    rng = np.random.default_rng(2)
    cube = rng.normal(0, 0.5, (2000, 3))
    fm = FullMeasure(cube, eps=0.4, seed=0)
    c = cube[3]
    got = fm.member_indices({"kind": "box", "center": c, "half": np.full(3, 0.8)})
    want = np.flatnonzero(np.all(np.abs(cube - c) <= 0.8, axis=1))
    assert np.array_equal(got, want)


def test_union_still_works_over_conjunctive_parts(world):
    fm, coords, spec = world
    p1 = {"kind": "ball", "center": coords[1], "mult": 1.0}
    p2 = {"kind": "ball", "center": coords[2], "mult": 1.0}
    u = fm.member_indices({"kind": "union", "parts": [p1, p2]})
    want = np.union1d(fm.member_indices(p1), fm.member_indices(p2))
    assert np.array_equal(u, want)
