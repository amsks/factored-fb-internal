"""Synthesised eval goal frames and goal distances for MetaWorld and ManiSkill."""
import numpy as np
import pytest

from env.unified import goal_distance, goal_layout, synth_goal_frame


# Synth goal frames

def test_mw_synth_frame_moves_hand_and_object_onto_the_target():
    """End effector and object, and their previous-frame copies, move onto the target."""
    rng = np.random.RandomState(0)
    obs = rng.randn(39).astype(np.float32)

    expect = obs.copy()
    tgt = expect[36:39].copy()
    for a, b in ((0, 3), (4, 7), (18, 21), (22, 25)):
        expect[a:b] = tgt

    got = synth_goal_frame("mw-push", obs)
    np.testing.assert_array_equal(got, expect)
    # Target dims stay the episode's own.
    np.testing.assert_array_equal(got[36:39], obs[36:39])
    assert got is not obs and not np.shares_memory(got, obs)


def test_mw_synth_frame_leaves_untouched_dims_alone():
    obs = np.arange(39, dtype=np.float32)
    got = synth_goal_frame("mw-pick-place", obs)
    for d in (3, 7, 11, 17, 21, 25, 30, 35):
        assert got[d] == obs[d], f"dim {d} should not be rewritten"


def test_ms_push_cube_synth_frame():
    """PushCube (35) = qpos9 | qvel9 | tcp_pose7 | goal_pos3 | obj_pose7."""
    obs = np.random.RandomState(1).randn(35).astype(np.float32)
    got = synth_goal_frame("ms-push-cube", obs)
    tgt = obs[25:28]
    np.testing.assert_array_equal(got[18:21], tgt)   # tcp position -> target
    np.testing.assert_array_equal(got[28:31], tgt)   # object position -> target
    np.testing.assert_array_equal(got[25:28], tgt)   # goal slot unchanged
    np.testing.assert_array_equal(got[:18], obs[:18])
    np.testing.assert_array_equal(got[31:], obs[31:])  # object quaternion untouched


def test_ms_pick_cube_synth_frame_is_self_consistent():
    """PickCube synth frame keeps the base tcp-to-obj offset, zeroes obj_to_goal and sets is_grasped."""
    obs = np.random.RandomState(2).randn(42).astype(np.float32)
    got = synth_goal_frame("ms-pick-cube", obs)
    tgt = obs[26:29]
    base_offset = obs[19:22] - obs[29:32]   # base frame's tcp-to-obj offset
    if np.linalg.norm(base_offset) > 0.1:   # degenerate: median grasp fallback
        base_offset = np.array([0.0, 0.0, 0.018], np.float32)
    np.testing.assert_allclose(got[19:22], tgt + base_offset, atol=1e-6)   # tcp position
    np.testing.assert_array_equal(got[29:32], tgt)          # object position
    np.testing.assert_array_equal(got[39:42], np.zeros(3, np.float32))   # obj_to_goal
    assert got[18] == 1.0                                   # is_grasped
    # tcp_to_obj = obj - tcp
    np.testing.assert_allclose(got[36:39], got[29:32] - got[19:22], atol=1e-6)
    np.testing.assert_allclose(got[39:42], got[26:29] - got[29:32], atol=1e-6)


def test_ms_pick_cube_synth_frame_falls_back_on_degenerate_base_offset():
    """A non-grasp base offset falls back to the dataset-median grasp above the object."""
    obs = np.random.RandomState(4).randn(42).astype(np.float32)
    obs[19:22] = obs[29:32] + np.array([5.0, -3.0, 2.0], np.float32)   # norm >> 0.1
    got = synth_goal_frame("ms-pick-cube", obs)
    tgt = obs[26:29]
    np.testing.assert_allclose(got[19:22], tgt + np.array([0.0, 0.0, 0.018], np.float32),
                                atol=1e-6)
    np.testing.assert_array_equal(got[29:32], tgt)


def test_synth_frame_rejects_tasks_with_no_goal_in_obs():
    assert goal_layout("ms-lift-peg") is None
    with pytest.raises(ValueError, match="no goal-in-obs layout"):
        synth_goal_frame("ms-lift-peg", np.zeros(12, np.float32))


# Goal distance

def test_goal_distance_uses_each_env_success_metric():
    obs = np.zeros(35, np.float32)
    obs[25:28] = [0.0, 0.0, 0.02]      # goal
    obs[28:31] = [0.3, 0.4, 0.90]      # object, far above the goal in z
    # PushCube scores planar XY distance to its goal region, so the z gap must not count.
    assert goal_distance("ms-push-cube", obs) == pytest.approx(0.5, abs=1e-6)

    obs = np.zeros(42, np.float32)
    obs[26:29] = [0.0, 0.0, 0.0]
    obs[29:32] = [0.0, 0.3, 0.4]
    assert goal_distance("ms-pick-cube", obs) == pytest.approx(0.5, abs=1e-6)  # 3-D

    obs = np.zeros(39, np.float32)
    obs[36:39] = [1.0, 0.0, 0.0]
    obs[4:7] = [1.0, 0.0, 0.3]
    assert goal_distance("mw-push", obs) == pytest.approx(0.3, abs=1e-6)


def test_goal_distance_is_none_without_a_layout():
    assert goal_distance("ms-lift-peg", np.zeros(12, np.float32)) is None
    assert goal_distance("dmc-walker", np.zeros(24, np.float32)) is None


def test_synth_frame_is_at_zero_goal_distance():
    """A synthesised goal frame has zero goal distance."""
    for task, dim in (("mw-push", 39), ("ms-push-cube", 35), ("ms-pick-cube", 42)):
        obs = np.random.RandomState(3).randn(dim).astype(np.float32)
        assert goal_distance(task, synth_goal_frame(task, obs)) == pytest.approx(0.0, abs=1e-6)
