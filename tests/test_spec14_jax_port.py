"""Tests for the spec 14 §7 multi-env port of eval_zeroshot_rewards.py.

Pure numpy — no env, no checkpoint, no jax (the harness stays numpy-only at
module level). Covers the new geometry-aware helpers:
  * eval_tasks_for: singletask id construction for any play env;
  * reward_conj_dirac: conjunctive membership == brute force, and reduces to
    the archived single-ball reward_dirac on cube-single geometry;
  * reward_conj_dense: reduces to the archived reward_dense on cube-single;
  * draw_region_centers: hindsight rule + min_pos degeneracy guard;
  * region_success_from_obs: final/any conjunctive membership over obs trajs.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_zeroshot_rewards.py"


@pytest.fixture(scope="module")
def ezr():
    spec = importlib.util.spec_from_file_location("eval_zeroshot_rewards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


@pytest.fixture(scope="module")
def geom():
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))
    import env_geometry
    return env_geometry


def test_eval_tasks_for(ezr):
    t = ezr.eval_tasks_for("cube-double-play-v0")
    assert t[2] == "cube-double-play-singletask-task2-v0"
    assert ezr.eval_tasks_for("scene-play-v0")[5] == "scene-play-singletask-task5-v0"
    assert ezr.eval_tasks_for(ezr.CUBE_SINGLE) == ezr.EVAL_TASKS
    with pytest.raises(AssertionError):
        ezr.eval_tasks_for("cube-double-play")   # not a -v0 id


def test_reward_conj_dirac_matches_bruteforce(ezr, geom):
    spec = geom.HitSpec.parse("0:3:0.4,3:6:0.5")
    rng = np.random.default_rng(0)
    coords = rng.uniform(-1, 1, size=(500, 6))
    center = rng.uniform(-0.5, 0.5, size=6)
    r = ezr.reward_conj_dirac(coords, center, spec)
    d1 = np.linalg.norm(coords[:, :3] - center[:3], axis=1)
    d2 = np.linalg.norm(coords[:, 3:] - center[3:], axis=1)
    expect = ((d1 <= 0.4) & (d2 <= 0.5)).astype(float)
    np.testing.assert_allclose(r, expect)
    assert set(np.unique(r)) <= {0.0, 1.0}
    # conjunction: one block inside, the other outside -> 0
    c2 = np.concatenate([coords[0, :3], coords[0, 3:] + 10.0])
    assert ezr.reward_conj_dirac(coords[:1], c2, spec)[0] == 0.0


def test_reward_conj_dirac_reduces_to_archived_single_ball(ezr, geom):
    spec = geom.GEOMETRY["cube-single-play-v0"].hit_spec        # 19:22:0.4
    rng = np.random.default_rng(1)
    obs = rng.uniform(-2, 2, size=(300, 29))
    goal_obs = rng.uniform(-2, 2, size=29)
    r_new = ezr.reward_conj_dirac(spec.coords(obs), spec.coords(goal_obs[None])[0], spec)
    r_old = ezr.reward_dirac(ezr.cube_obs_units(obs), goal_obs[19:22])
    np.testing.assert_allclose(r_new, r_old)


def test_reward_conj_dense_reduces_to_archived_on_cube_single(ezr, geom):
    spec = geom.GEOMETRY["cube-single-play-v0"].hit_spec
    rng = np.random.default_rng(2)
    obs = rng.uniform(-2, 2, size=(200, 29))
    goal_obs = rng.uniform(-2, 2, size=29)
    r_new = ezr.reward_conj_dense(spec.coords(obs), spec.coords(goal_obs[None])[0], spec)
    r_old = ezr.reward_dense(ezr.cube_obs_units(obs), goal_obs[19:22])   # sigma 0.8
    np.testing.assert_allclose(r_new, r_old, rtol=1e-12)


def test_draw_region_centers_hindsight_and_min_pos(ezr, geom):
    """Centers must be visited-state coords with >= min_pos relabel positives;
    a region nobody's relabel sample covers must be rejected, not returned."""
    spec = geom.HitSpec.parse("0:2:0.4")
    # dataset: 2 episodes of 50 rows; episode 0 clusters at the origin (dense),
    # episode 1 marches off to a region the relabel sample never visits.
    rng0 = np.random.default_rng(0)
    ep0 = rng0.normal(0.0, 0.05, size=(50, 2))
    ep1 = np.linspace(50.0, 60.0, 50)[:, None] * np.ones((1, 2))
    coords_all = np.concatenate([ep0, ep1])
    terminals = np.zeros(100)
    terminals[49] = terminals[99] = 1.0
    bounds = geom.episode_bounds(terminals)
    relabel = rng0.normal(0.0, 0.05, size=(400, 2))     # covers ep0 only
    rng = np.random.default_rng(3)
    centers, npos, rejected = ezr.draw_region_centers(
        coords_all, bounds, spec, relabel, rng, k=4, min_pos=5, max_tries=500)
    assert len(centers) == 4
    assert (npos >= 5).all()
    # every accepted center is near the origin cluster (ep1 regions rejected)
    assert (np.linalg.norm(centers, axis=1) < 1.0).all()
    assert rejected > 0
    # each accepted center coincides with a dataset row (visited state)
    for c in centers:
        assert np.min(np.linalg.norm(coords_all - c, axis=1)) < 1e-12


def test_region_success_from_obs(ezr, geom):
    spec = geom.HitSpec.parse("0:2:0.5,2:4:0.5")
    center = np.zeros(4)
    inside = np.zeros(4)
    outside = np.full(4, 5.0)
    half = np.array([0.1, 0.1, 5.0, 5.0])              # block 1 in, block 2 out
    # ep0: enters then leaves (any=1, final=0); ep1: ends inside (any=1, final=1);
    # ep2: only the HALF-conjunction ever holds (any=0, final=0); ep3: length 0.
    obs_traj = np.zeros((4, 3, 4))
    obs_traj[0] = [outside, inside, outside]
    obs_traj[1] = [outside, outside, inside]
    obs_traj[2] = [half, half, half]
    lengths = np.array([3, 3, 3, 0])
    fin, anys = ezr.region_success_from_obs(obs_traj, lengths, center, spec)
    np.testing.assert_allclose(fin, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(anys, [1.0, 1.0, 0.0, 0.0])
    # padded steps beyond `lengths` must be ignored
    obs_traj2 = obs_traj.copy()
    lengths2 = np.array([2, 2, 3, 0])
    obs_traj2[0, 2] = inside                            # padded row, not real
    fin2, any2 = ezr.region_success_from_obs(obs_traj2, lengths2, center, spec)
    np.testing.assert_allclose(fin2, [1.0, 0.0, 0.0, 0.0])
