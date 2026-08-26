"""eval/humenv_goal.py, the numpy port of HumEnv's goal metric — torch-free checks.

The parity-against-torch half lives in scripts/humenv_metric_parity.py, which runs in
.venv-humenv. Everything here needs only numpy and runs in the normal suite.
"""
import numpy as np

from eval.humenv_goal import (BOUND, MARGIN, POSE_DIM, aggregate, distance_smpl,
                              episode_goal_stats)

OBS = 358


def test_pose_block_matches_the_env_module():
    """The domain\'s measure_hit_spec and this metric must resolve the SAME columns; the
    hum-humanoid hit spec is 0:214 precisely because the benchmark slices next_obs[..., :214].
    """
    from env.humenv_env import POSE_BLOCK
    assert POSE_BLOCK == (0, POSE_DIM)


def test_distance_ignores_everything_past_the_pose_block():
    rng = np.random.default_rng(0)
    goal = rng.normal(size=(OBS,)).astype(np.float32)
    obs = goal.copy()[None].repeat(5, 0)
    obs[:, POSE_DIM:] += 100.0          # garbage outside the pose block
    np.testing.assert_allclose(distance_smpl(obs, goal), np.zeros(5), atol=1e-5)


def test_success_is_latching_not_terminal():
    """One in-bounds step anywhere scores the episode, exactly like ExoRL point_mass_maze.
    A metric read as \'ended inside the radius\' would score these opposite ways."""
    goal = np.zeros((OBS,), np.float32)
    obs = np.zeros((4, OBS), np.float32)
    obs[:, 0] = [0.0, 10.0, 10.0, 10.0]      # inside only at the FIRST step
    assert episode_goal_stats(obs, goal)["success"] == 1.0


def test_proximity_ramp_endpoints():
    goal = np.zeros((OBS,), np.float32)
    def at(d):
        o = np.zeros((1, OBS), np.float32); o[0, 0] = d
        return episode_goal_stats(o, goal)["proximity"]
    assert at(0.0) == 1.0                       # inside the bound -> full credit
    assert at(BOUND) == 1.0                     # ON the bound is still inside
    assert at(BOUND + MARGIN) == 0.0            # top of the ramp -> no credit
    assert at(BOUND + MARGIN + 5.0) == 0.0      # beyond -> no credit, not negative
    np.testing.assert_allclose(at(BOUND + MARGIN / 2), 0.5, atol=1e-6)


def test_aggregate_reports_spread_and_count():
    eps = [{"success": 1.0, "proximity": 0.8, "distance": 1.0},
           {"success": 0.0, "proximity": 0.4, "distance": 3.0}]
    agg = aggregate(eps)
    assert agg["n_episodes"] == 2
    assert agg["success"] == 0.5
    assert agg["distance_sd"] > 0
