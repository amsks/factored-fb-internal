"""E0 shared infrastructure for the deconfound-size-coverage spec
(docs/specs/2026-08-10-deconfound-size-coverage.md):

  E0.1  synth per-episode eval goal frames  (env/unified.py::synth_goal_frame)
  E0.2  per-episode goal distances          (env/unified.py::goal_distance)
  E0.3  cube dataset size ladder            (env/ogbench.py::truncate_whole_trajectories)

These are pure-array unit tests: no env, no dataset download, no JAX compilation.
"""
import numpy as np
import pytest

from env.ogbench import truncate_whole_trajectories
from env.unified import goal_distance, goal_layout, synth_goal_frame


# ── E0.1: synth goal frames ──────────────────────────────────────────────────────────────

def test_mw_synth_frame_matches_the_bon_probe_construction():
    """The Metaworld slot list is the one eval/bon_probe.py verified (and where synth eval
    lifted every MW selection cell by +0.12..+0.43). It must not drift."""
    rng = np.random.RandomState(0)
    obs = rng.randn(39).astype(np.float32)

    expect = obs.copy()
    tgt = expect[36:39].copy()
    for a, b in ((0, 3), (4, 7), (18, 21), (22, 25)):
        expect[a:b] = tgt

    got = synth_goal_frame("mw-push", obs)
    np.testing.assert_array_equal(got, expect)
    # The target dims themselves stay the episode's own: that is what makes (s, g) a
    # context-consistent pair rather than a frame from some other episode.
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
    """PickCube (42) carries two DERIVED channels, tcp_to_obj_pos (36:39) and obj_to_goal_pos
    (39:42). Real success frames never have tcp == obj (grasp offset mean/median 0.021/0.018,
    the same scale as the ~0.02-0.025 success radius), so the synth frame keeps the base
    frame's own tcp-to-obj offset rather than collapsing tcp onto the target; only obj_to_goal
    (target == object) is algebraically zero. is_grasped (18) is 1.0 on 100% of success frames
    in the collected data, because ManiSkill's success rule requires a grasp."""
    obs = np.random.RandomState(2).randn(42).astype(np.float32)
    got = synth_goal_frame("ms-pick-cube", obs)
    tgt = obs[26:29]
    base_offset = obs[19:22] - obs[29:32]   # the base frame's own (pre-synth) tcp-to-obj offset
    if np.linalg.norm(base_offset) > 0.1:   # degenerate -> dataset-median grasp fallback
        base_offset = np.array([0.0, 0.0, 0.018], np.float32)
    np.testing.assert_allclose(got[19:22], tgt + base_offset, atol=1e-6)   # tcp position
    np.testing.assert_array_equal(got[29:32], tgt)          # object position
    np.testing.assert_array_equal(got[39:42], np.zeros(3, np.float32))   # obj_to_goal
    assert got[18] == 1.0                                   # is_grasped
    # ManiSkill's own algebra reproduced by the frame: tcp_to_obj = obj - tcp, no longer 0.
    np.testing.assert_allclose(got[36:39], got[29:32] - got[19:22], atol=1e-6)
    np.testing.assert_allclose(got[39:42], got[26:29] - got[29:32], atol=1e-6)


def test_ms_pick_cube_synth_frame_falls_back_on_degenerate_base_offset():
    """A base frame whose tcp is far from its object (not a grasp pose) must not propagate
    that nonsense offset into the synth goal frame; it falls back to the dataset-median grasp,
    straight above the object."""
    obs = np.random.RandomState(4).randn(42).astype(np.float32)
    obs[19:22] = obs[29:32] + np.array([5.0, -3.0, 2.0], np.float32)   # degenerate: norm >> 0.1
    got = synth_goal_frame("ms-pick-cube", obs)
    tgt = obs[26:29]
    np.testing.assert_allclose(got[19:22], tgt + np.array([0.0, 0.0, 0.018], np.float32),
                                atol=1e-6)
    np.testing.assert_array_equal(got[29:32], tgt)


def test_synth_frame_rejects_families_with_no_goal_in_obs():
    assert goal_layout("b2d-goal-ctrl-maze") is None
    with pytest.raises(ValueError, match="no goal-in-obs layout"):
        synth_goal_frame("b2d-goal-ctrl-maze", np.zeros(12, np.float32))


# ── E0.2: per-episode goal distance ──────────────────────────────────────────────────────

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
    assert goal_distance("b2d-goal-ctrl-maze", np.zeros(12, np.float32)) is None


def test_synth_frame_is_at_zero_goal_distance():
    """A synthesised goal frame is, by construction, a state that satisfies its own goal —
    the property that makes it a legitimate goal to condition on."""
    for task, dim in (("mw-push", 39), ("ms-push-cube", 35), ("ms-pick-cube", 42)):
        obs = np.random.RandomState(3).randn(dim).astype(np.float32)
        assert goal_distance(task, synth_goal_frame(task, obs)) == pytest.approx(0.0, abs=1e-6)


# ── E0.3: cube size ladder ───────────────────────────────────────────────────────────────

def _compact(n_traj=40, traj_rows=11, obs_dim=3):
    """A dataset in OGBench's COMPACT layout: T+1 rows per trajectory, valids == 0 on each
    trajectory's last row, terminals == 1 on its last TWO rows (ogbench/utils.py)."""
    n = n_traj * traj_rows
    valids = np.ones(n, np.float32)
    valids[traj_rows - 1::traj_rows] = 0.0
    terminals = np.zeros(n, np.float32)
    terminals[traj_rows - 1::traj_rows] = 1.0
    terminals[traj_rows - 2::traj_rows] = 1.0
    return dict(
        observations=np.arange(n * obs_dim, dtype=np.float32).reshape(n, obs_dim),
        actions=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        terminals=terminals, valids=valids,
    )


def _rows(ds):
    return np.asarray(ds["observations"])


def test_truncation_at_full_size_is_a_bit_for_bit_no_op():
    ds = _compact()
    n = len(ds["observations"])
    for request in (n, n + 1, 10 * n, None):
        out = truncate_whole_trajectories(ds, request)
        for k in ds:
            np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(ds[k]),
                                          err_msg=f"{k} changed at request={request}")


def test_truncation_keeps_whole_trajectories_and_meets_the_request():
    ds = _compact(n_traj=40, traj_rows=11)
    out = truncate_whole_trajectories(ds, 100)
    valids = np.asarray(out["valids"])
    assert len(valids) >= 100
    assert len(valids) % 11 == 0                       # only whole trajectories
    ends = np.nonzero(valids == 0)[0]
    np.testing.assert_array_equal(ends, np.arange(10, len(valids), 11))
    # The smallest whole-trajectory count that reaches 100 rows.
    assert len(valids) == 110
    # GCDataset asserts the final row is a boundary.
    assert np.asarray(out["terminals"])[-1] > 0


def test_truncation_preserves_row_contents_and_trajectory_order():
    """Kept rows are the original rows, unmodified, still in dataset order — the subset is a
    sub-sequence of the source, not a reshuffle."""
    ds = _compact()
    out = truncate_whole_trajectories(ds, 150)
    src, got = _rows(ds), _rows(out)
    first_col = got[:, 0]
    assert np.all(np.diff(first_col) > 0), "rows are not in increasing source order"
    keep = np.searchsorted(src[:, 0], first_col)
    np.testing.assert_array_equal(got, src[keep])
    np.testing.assert_array_equal(np.asarray(out["actions"]),
                                  np.asarray(ds["actions"])[keep])


def test_truncation_is_deterministic_and_nested_across_sizes():
    """Every run at a given size must see identical data (or the critic contrast is confounded
    with a data draw), and the ladder's rungs must nest (or a size effect is confounded with a
    change of which trajectories are present)."""
    ds = _compact(n_traj=100, traj_rows=11)
    a = _rows(truncate_whole_trajectories(ds, 200))
    b = _rows(truncate_whole_trajectories(ds, 200))
    np.testing.assert_array_equal(a, b)

    small = set(_rows(truncate_whole_trajectories(ds, 200))[:, 0].tolist())
    for bigger in (400, 700):
        large = set(_rows(truncate_whole_trajectories(ds, bigger))[:, 0].tolist())
        assert small <= large, f"{bigger}-row subset does not contain the 200-row subset"
        small = large


def test_truncation_falls_back_to_terminals_without_valids():
    """The non-compact layout has no `valids`; `terminals` marks the same boundary rows."""
    ds = _compact()
    del ds["valids"]
    ds["terminals"] = np.zeros(len(ds["observations"]), np.float32)
    ds["terminals"][10::11] = 1.0
    out = truncate_whole_trajectories(ds, 100)
    assert len(out["observations"]) == 110
    np.testing.assert_array_equal(np.nonzero(np.asarray(out["terminals"]) > 0)[0],
                                  np.arange(10, 110, 11))


def test_truncation_rejects_a_dataset_that_does_not_end_on_a_boundary():
    ds = _compact()
    ds["valids"][-1] = 1.0
    with pytest.raises(AssertionError, match="trajectory boundary"):
        truncate_whole_trajectories(ds, 100)
