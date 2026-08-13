"""Tests for spec 26 R0b: scripts/rescuing_fb/bfm_act.py (waypoint logic) +
scripts/rescuing_fb/eval_tracking.py (split/metrics). Pure numpy, no
jax/env/checkpoint at module level (same discipline as
tests/test_eval_density_zeroshot.py; both scripts keep jax lazy).

The contract's three tests (specs/26-bfm-roadmap.md sec.2):
  1. waypoint-progress MONOTONICITY on a synthetic path,
  2. shuffled-CONTROL wiring (permutation seeded + non-identity, applied to
     the conditioning row ONLY -- metrics never see it),
  3. DETERMINISM repeat (split + tracker + metrics pipelines bit-identical
     across two independent runs).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)     # numpy-only at module level (jax is lazy)
    return mod


@pytest.fixture(scope="module")
def ba():
    return _load("bfm_act_test", "scripts/rescuing_fb/bfm_act.py")


@pytest.fixture(scope="module")
def et():
    return _load("eval_tracking_test", "scripts/rescuing_fb/eval_tracking.py")


@pytest.fixture(scope="module")
def spec3():
    from scripts.rescuing_fb.env_geometry import HitSpec
    return HitSpec.parse("0:3:0.4")     # cube-single-shaped: one 3-dim block


def _line_demo(T=101, obs_dim=6):
    """Synthetic demo: object block (cols 0:3) walks a straight line in x."""
    obs = np.zeros((T, obs_dim), np.float32)
    obs[:, 0] = np.linspace(0.0, 2.0, T)
    return obs


# ============ 1. waypoint-progress monotonicity ====================================

def test_progress_monotone_on_synthetic_path(ba, spec3):
    """Progress never decreases, even when the current state is closest to an
    EARLIER demo row than the cursor (the argmin is restricted to >= prev)."""
    demo = _line_demo()
    tr = ba.WaypointTracker(demo, spec3, delta=25)
    # advance to the middle of the path
    cur = np.zeros(6, np.float32); cur[0] = 1.0
    tr.goal(cur)
    p_mid = tr.progress
    assert p_mid == 50
    # object retreats to the very start -- progress must NOT move backwards
    tr.goal(np.zeros(6, np.float32))
    assert tr.progress == p_mid
    # random walk: progress is monotone non-decreasing throughout
    rng = np.random.default_rng(0)
    prev = tr.progress
    for _ in range(200):
        cur = np.zeros(6, np.float32)
        cur[0] = rng.uniform(-0.5, 2.5)
        tr.goal(cur)
        assert tr.progress >= prev
        prev = tr.progress
    # and the raw update_progress function agrees on the boundary case
    ds = spec3.scaled(spec3.coords(demo))
    assert ba.update_progress(ds, ds[10], prev=40) == 40      # clamped, not 10

    # the waypoint is delta ahead of progress, clamped at the final row
    assert ba.waypoint_index(50, 25, 101) == 75
    assert ba.waypoint_index(90, 25, 101) == 100
    # goal returned is the demo row at the waypoint (no perm)
    tr2 = ba.WaypointTracker(demo, spec3, delta=25)
    g, info = tr2.goal(np.zeros(6, np.float32))
    assert info["progress"] == 0 and info["waypoint"] == 25
    assert info["conditioned_row"] == 25
    np.testing.assert_array_equal(g, demo[25])


# ============ 2. shuffled-control wiring ===========================================

def test_shuffled_control_wiring(ba, et, spec3):
    """The control's permutation is seeded, deterministic, non-identity, and
    touches ONLY the conditioning row: progress/waypoint report the TRUE
    index, and tracking_metrics takes no permutation at all (metrics are
    always against the true demo)."""
    T = 101
    p1 = ba.conditioning_permutation(T, seed=7)
    p2 = ba.conditioning_permutation(T, seed=7)
    np.testing.assert_array_equal(p1, p2)                     # deterministic
    assert not np.array_equal(p1, np.arange(T))               # non-identity
    assert np.array_equal(np.sort(p1), np.arange(T))          # a permutation
    assert not np.array_equal(p1, ba.conditioning_permutation(T, seed=8))

    demo = _line_demo(T)
    tr = ba.WaypointTracker(demo, spec3, delta=25, perm=p1)
    g, info = tr.goal(np.zeros(6, np.float32))
    assert info["waypoint"] == 25                             # TRUE index reported
    assert info["conditioned_row"] == int(p1[25])             # conditioning remapped
    np.testing.assert_array_equal(g, demo[int(p1[25])])
    # progress still runs against the TRUE demo path
    cur = np.zeros(6, np.float32); cur[0] = 1.0
    tr.goal(cur)
    assert tr.progress == 50

    # metrics NEVER see the permutation: tracking_metrics has no perm argument
    # and scores a perfect true-demo rollout at hit_rate=1 regardless of any
    # shuffling that happened at conditioning time.
    import inspect
    assert "perm" not in inspect.signature(et.tracking_metrics).parameters
    roll = spec3.coords(demo[1:])                             # perfect tracking
    m = et.tracking_metrics(roll, spec3.coords(demo), spec3, delta=25)
    assert m["hit_rate"] == 1.0 and m["completion"] is True
    assert m["path_error"] == 0.0
    assert m["waypoints"] == [25, 50, 75, 100]

    # a rollout that sits still at the start misses far waypoints
    roll_still = np.repeat(spec3.coords(demo[:1]), T - 1, axis=0)
    m2 = et.tracking_metrics(roll_still, spec3.coords(demo), spec3, delta=25)
    assert m2["hit_rate"] < 1.0 and m2["completion"] is False
    assert m2["path_error"] > 0.0


# ============ 3. determinism repeat ================================================

def test_determinism_repeat(ba, et, spec3):
    """Two independent runs of the pure pipeline (split -> tracker stream ->
    metrics) are bit-identical; the split is a valid episode-level holdout."""
    h1 = et.held_out_episode_indices(1000, 50, seed=0)
    h2 = et.held_out_episode_indices(1000, 50, seed=0)
    np.testing.assert_array_equal(h1, h2)
    assert len(h1) == 50 and len(np.unique(h1)) == 50         # no repeats
    assert np.all(np.diff(h1) > 0)                            # sorted
    assert not np.array_equal(h1, et.held_out_episode_indices(1000, 50, seed=1))

    def pipeline():
        demo = _line_demo()
        tr = ba.WaypointTracker(demo, spec3, delta=25,
                                perm=ba.conditioning_permutation(len(demo), 3))
        rng = np.random.default_rng(123)
        goals, infos = [], []
        cur = demo[0].copy()
        for _ in range(60):
            cur = cur + rng.normal(0, 0.05, size=cur.shape).astype(np.float32)
            g, info = tr.goal(cur)
            goals.append(g.copy()); infos.append(dict(info))
        roll = np.stack([g[:3] for g in goals] + [demo[t, :3] for t in range(61, 101)])
        m = et.tracking_metrics(spec3.coords(np.pad(roll, ((0, 0), (0, 3)))),
                                spec3.coords(demo), spec3, delta=25)
        return goals, infos, m

    g1, i1, m1 = pipeline()
    g2, i2, m2 = pipeline()
    assert i1 == i2 and m1 == m2
    np.testing.assert_array_equal(np.stack(g1), np.stack(g2))


# ============ supporting pins (interface-level, still jax-free) ====================

def test_classify_prompt(ba):
    assert ba.classify_prompt(np.zeros(28)) == "goal"
    assert ba.classify_prompt(np.zeros((10, 28))) == "demo"
    assert ba.classify_prompt(lambda c: np.zeros(len(c))) == "reward"
    with pytest.raises(TypeError):
        ba.classify_prompt(np.zeros((2, 3, 4)))


def test_waypoint_schedule(ba):
    np.testing.assert_array_equal(ba.waypoint_schedule(201, 25),
                                  [25, 50, 75, 100, 125, 150, 175, 200])
    np.testing.assert_array_equal(ba.waypoint_schedule(101, 100), [100])
    with pytest.raises(AssertionError):
        ba.waypoint_schedule(1, 25)
