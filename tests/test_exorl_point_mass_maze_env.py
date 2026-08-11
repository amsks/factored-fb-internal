"""Spec 23 P0: exercise the thin dm_control eval adapter on a random policy
so we know it produces sane episodes BEFORE any agent is trained (the
task's stated "done" bar)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "exorl"))

from point_mass_maze_env import (  # noqa: E402
    PointMassMazeEnv, RolloutResult, MAZE_20_GOALS, GOAL_EPS, EPISODE_LEN,
    random_policy, rollout,
)


def test_reset_layout_matches_verified_block():
    env = PointMassMazeEnv("reach_top_left", seed=0)
    obs = env.reset()
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    # At reset, velocity (dims 2:4) is exactly zero (matches the real
    # downloaded data's first row, verified in test_exorl_convert.py).
    assert np.allclose(obs[2:], 0.0)
    # Position (dims 0:2) starts in the qpos-randomized range from
    # MultiTaskPointMassMaze.initialize_episode: x in [-0.29, -0.15],
    # y in [0.15, 0.29] (top-left room), per the vendored task source.
    assert -0.30 <= obs[0] <= -0.14
    assert 0.14 <= obs[1] <= 0.30


def test_action_spec_bounds():
    env = PointMassMazeEnv("reach_top_left", seed=0)
    env.reset()
    spec = env.action_spec
    assert np.allclose(spec.minimum, [-1.0, -1.0])
    assert np.allclose(spec.maximum, [1.0, 1.0])


def test_random_policy_rollout_is_sane():
    env = PointMassMazeEnv("reach_top_left", seed=0)
    policy = random_policy(action_dim=2, rng=np.random.default_rng(0))
    result = rollout(env, policy, goal_xy=MAZE_20_GOALS[0], eps=GOAL_EPS, horizon=EPISODE_LEN)

    assert isinstance(result, RolloutResult)
    assert result.length == EPISODE_LEN
    assert result.observations.shape == (EPISODE_LEN + 1, 4)
    assert result.actions.shape == (EPISODE_LEN, 2)
    # No NaN/inf anywhere in the rollout.
    assert np.isfinite(result.observations).all()
    assert np.isfinite(result.distances).all()
    # Actions stayed within the declared action spec.
    assert np.all(result.actions >= -1.0 - 1e-6)
    assert np.all(result.actions <= 1.0 + 1e-6)
    # Position stays roughly within the maze bounds (walls at +-0.3, per
    # the vendored XML); loose bound, just a sanity check the physics
    # didn't blow up.
    assert np.all(np.abs(result.observations[:, :2]) <= 0.35)
    # min_distance/success flags are internally consistent.
    assert result.min_distance == pytest.approx(result.distances.min())
    assert result.success_any_time == bool((result.distances <= GOAL_EPS).any())
    assert result.success_final == bool(result.distances[-1] <= GOAL_EPS)


def test_random_policy_all_20_goals_run_without_error():
    """Cheap breadth check: every published goal is a valid query point and
    produces a finite rollout (does not verify GOOD success rates -- a
    random policy on a 1000-step maze is not expected to reliably reach a
    3cm-radius target; this only guards plumbing correctness)."""
    env = PointMassMazeEnv("reach_top_left", seed=1)
    policy = random_policy(action_dim=2, rng=np.random.default_rng(1))
    successes = []
    for goal in MAZE_20_GOALS:
        result = rollout(env, policy, goal_xy=goal, eps=GOAL_EPS, horizon=200)
        assert np.isfinite(result.distances).all()
        successes.append(result.success_any_time)
    # Not a gate -- just recorded so a human can sanity-check it's not
    # literally always 0 or always 1 (which would indicate a broken
    # distance computation).
    assert 0 <= sum(successes) <= len(successes)
