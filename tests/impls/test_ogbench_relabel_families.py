"""Reward-inference (relabel) eval for the NON-cube OGBench families: antmaze, scene, puzzle.

Transcribed from td_jepa's per-env reward functions
(~/git/baselines/td_jepa/metamotivo/envs/ogbench.py:42-56, get_relabel_fn :113-127). One
thing that file does NOT show and that matters for scene/puzzle: td_jepa's "physics" row is
the npz `qpos` CONCATENATED with `button_states` when the npz has it
(scripts/data_processing/ogbench/extract_all.py:56-58), which is why scene's reward reads
button states at `physics[-2:]` and puzzle's at `physics[-9:]`. `env/ogbench.py` rebuilds
that row the same way.

Pinned here:
  1. Each family's reward against hand-computed synthetic physics rows (positives/negatives,
     thresholds, shapes), and the shift semantics (satisfied - k + 1; NOT a 0/1 indicator for
     scene/puzzle).
  2. `OGBenchAdapter.relabel_infer_batch` dispatches on the task_info's goal key, reads the
     NEXT row's physics, and stays byte-identical for cube.
  3. Integration on the real datasets (skipped when ~/.ogbench/data lacks them): batch
     produced, rewards in the expected integer range, positive counts per task reported.

Run standalone on CPU:

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
        tests/impls/test_ogbench_relabel_families.py -q -p no:cacheprovider
"""
import os
import types

import numpy as np
import pytest

from env.ogbench import (
    OGBenchAdapter,
    antmaze_relabel_reward,
    cube_relabel_reward,
    puzzle_relabel_reward,
    relabel_reward_for_task,
    scene_relabel_reward,
)
from impls.utils.datasets import FBDataset

OBS_DIM, ACT_DIM = 6, 2
DATA_DIR = os.path.expanduser("~/.ogbench/data")


def _dataset_present(name):
    return os.path.exists(os.path.join(DATA_DIR, f"{name}.npz"))


# ======================================================================================
# 1. Per-family rewards, hand computed.
# ======================================================================================
def test_antmaze_reward_hand_computed():
    goal_xy = np.array([4.0, 8.0])
    qpos = np.zeros((4, 15))
    qpos[0, :2] = [4.0, 8.0]         # exact -> success
    qpos[1, :2] = [4.5, 8.0]         # exactly 0.5 away -> success (<=, td_jepa ant_reward_fn)
    qpos[2, :2] = [4.51, 8.0]        # just outside -> failure
    qpos[3, :2] = [0.0, 0.0]         # far -> failure
    qpos[:, 2:] = 7.0                # the other 13 joints must be ignored
    r = antmaze_relabel_reward(qpos, goal_xy)
    assert r.shape == (4,)
    np.testing.assert_allclose(r, [0.0, 0.0, -1.0, -1.0])
    np.testing.assert_allclose(r + 1.0, [1.0, 1.0, 0.0, 0.0])   # shift 1 -> 0/1 indicator


def test_puzzle_reward_hand_computed():
    goal = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1])
    qpos = np.zeros((3, 23))
    bs = np.zeros((3, 9), np.int64)
    bs[0] = goal                                  # all 9 match
    bs[1] = goal; bs[1, 0] = 0                    # one wrong
    bs[2] = 1 - goal                              # all wrong
    physics = np.concatenate([qpos, bs], axis=-1)   # td_jepa's physics row
    r = puzzle_relabel_reward(physics, goal)
    assert r.shape == (3,)
    np.testing.assert_allclose(r, [0.0, -1.0, -9.0])
    # Shifted reward is (#matched buttons - 8): 1 only when ALL match, 0 for one miss, and
    # NEGATIVE below that -- not a 0/1 indicator.
    np.testing.assert_allclose(r + 1.0, [1.0, 0.0, -8.0])


def test_puzzle_reward_reads_the_last_k_columns_only():
    goal = np.array([1, 1, 0])
    physics = np.array([[9.0, 9.0, 1.0, 1.0, 0.0],
                        [9.0, 9.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(puzzle_relabel_reward(physics, goal), [0.0, -3.0])


def _scene_goal(cube=(0.35, 0.05, 0.02), buttons=(1, 1), drawer=-0.16, window=0.2):
    # Same key ORDER as ogbench's scene task_infos[...]['goal'] (scene_env.py:72-77); td_jepa
    # flattens `.values()` in that order.
    return dict(block_xyzs=np.array([list(cube)]), button_states=np.array(buttons),
                drawer_pos=float(drawer), window_pos=float(window))


def _scene_physics(cube, buttons, drawer, window, n_qpos=25):
    row = np.zeros(n_qpos + 2)
    row[14:17] = cube
    row[23] = drawer
    row[24] = window
    row[25:27] = buttons
    return row


def test_scene_reward_hand_computed():
    goal = _scene_goal()
    rows = np.stack([
        _scene_physics([0.35, 0.05, 0.02], [1, 1], -0.16, 0.2),    # all 5 satisfied
        _scene_physics([0.35, 0.05, 0.02], [1, 1], -0.16, 0.0),    # window off (0.2 away)
        _scene_physics([0.35, 0.05, 0.02], [0, 1], -0.16, 0.2),    # button 1 off
        _scene_physics([0.35, 0.05, 0.02], [1, 0], -0.16, 0.2),    # button 2 off
        _scene_physics([0.35, 0.05, 0.02], [1, 1], 0.0, 0.2),      # drawer off (0.16 away)
        _scene_physics([0.35, 0.05, 0.10], [1, 1], -0.16, 0.2),    # cube 0.08 away
        _scene_physics([0.60, 0.30, 0.10], [0, 0], 0.0, 0.0),      # nothing satisfied
    ])
    r = scene_relabel_reward(rows, goal)
    assert r.shape == (7,)
    np.testing.assert_allclose(r, [0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -5.0])
    np.testing.assert_allclose(r + 1.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0])


def test_scene_thresholds_are_inclusive():
    """td_jepa's scene_reward_fn uses <= for cube, drawer and window (cube_reward_fn uses <)."""
    goal = _scene_goal(cube=(0.0, 0.0, 0.0), drawer=0.0, window=0.0)
    rows = np.stack([
        _scene_physics([0.04, 0.0, 0.0], [1, 1], 0.0, 0.0),     # cube exactly at 0.04 -> ok
        _scene_physics([0.0, 0.0, 0.0], [1, 1], 0.04, 0.0),     # drawer exactly at 0.04 -> ok
        _scene_physics([0.0, 0.0, 0.0], [1, 1], 0.0, -0.04),    # window exactly at 0.04 -> ok
        _scene_physics([0.0, 0.0, 0.0], [1, 1], 0.0401, 0.0),   # drawer just outside
    ])
    np.testing.assert_allclose(scene_relabel_reward(rows, goal), [0.0, 0.0, 0.0, -1.0])


def test_scene_reward_accepts_the_flattened_7_vector_too():
    goal = _scene_goal()
    flat = np.concatenate([np.array([v]).ravel() for v in goal.values()])   # td_jepa's form
    assert flat.shape == (7,)
    rows = np.stack([_scene_physics([0.35, 0.05, 0.02], [1, 1], -0.16, 0.2),
                     _scene_physics([0.35, 0.05, 0.02], [0, 0], -0.16, 0.2)])
    np.testing.assert_allclose(scene_relabel_reward(rows, flat), scene_relabel_reward(rows, goal))
    np.testing.assert_allclose(scene_relabel_reward(rows, flat), [0.0, -2.0])


def test_scene_reward_rejects_a_goal_dict_in_the_wrong_order():
    bad = dict(window_pos=0.2, drawer_pos=-0.16, button_states=np.array([1, 1]),
               block_xyzs=np.array([[0.35, 0.05, 0.02]]))
    with pytest.raises(AssertionError, match="goal"):
        scene_relabel_reward(np.zeros((1, 27)), bad)


# ======================================================================================
# 2. Dispatch on the task_info's goal key.
# ======================================================================================
def test_dispatch_picks_the_family_from_the_task_info():
    ant = relabel_reward_for_task({"goal_xy": np.array([1.0, 1.0])},
                                  np.array([[1.0, 1.0, 0.0], [5.0, 5.0, 0.0]]))
    np.testing.assert_allclose(ant, [0.0, -1.0])

    puz = relabel_reward_for_task({"goal_button_states": np.array([1, 0])},
                                  np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(puz, [0.0, -1.0])

    scn = relabel_reward_for_task({"goal": _scene_goal()},
                                  np.stack([_scene_physics([0.35, 0.05, 0.02], [1, 1], -0.16, 0.2)]))
    np.testing.assert_allclose(scn, [0.0])

    q = np.zeros((1, 21)); q[0, 14:17] = [0.1, 0.2, 0.3]
    cube = relabel_reward_for_task({"goal_xyzs": np.array([[0.1, 0.2, 0.3]])}, q)
    np.testing.assert_allclose(cube, [0.0])
    np.testing.assert_allclose(cube, cube_relabel_reward(q, np.array([[0.1, 0.2, 0.3]])))


def test_dispatch_rejects_an_unknown_task_info():
    with pytest.raises(AssertionError, match="goal_xyzs"):
        relabel_reward_for_task({"goal_foo": np.zeros(2)}, np.zeros((1, 5)))


# ======================================================================================
# 3. relabel_infer_batch on synthetic datasets per family: NEXT-row physics, shift, shapes.
# ======================================================================================
def _fake_env(task_infos):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(task_infos=task_infos))


def _ds(n, qpos, button_states=None, seed=0):
    rng = np.random.default_rng(seed)
    valids = np.ones(n, np.float32)
    valids[-1] = 0.0
    fields = dict(observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
                  actions=np.zeros((n, ACT_DIM), np.float32), qpos=qpos.astype(np.float32),
                  valids=valids, terminals=1.0 - valids, rewards=np.zeros(n, np.float32))
    if button_states is not None:
        fields["button_states"] = button_states.astype(np.int64)
    ds = FBDataset.create(**fields)
    # Deterministic draw idx = 0..n-2 -> next rows 1..n-1.
    ds.get_random_idxs = lambda k: np.arange(min(k, ds.size - 1))
    return ds


def test_antmaze_batch_reads_next_row_xy():
    n = 6
    qpos = np.zeros((n, 15))
    qpos[2, :2] = [3.0, 4.0]                        # reachable via idx=1
    qpos[4, :2] = [3.3, 4.0]                        # 0.3 away, reachable via idx=3
    ds = _ds(n, qpos)
    env = _fake_env([{"goal_xy": np.array([3.0, 4.0])}])
    obs, r = OGBenchAdapter().relabel_infer_batch(ds, env, 1, 5, shift_reward=1.0)
    assert obs.shape == (5, OBS_DIM) and r.shape == (5,) and r.dtype == np.float32
    np.testing.assert_allclose(r, [0.0, 1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(obs, np.asarray(ds["observations"])[1:6])


def test_puzzle_batch_uses_button_states_not_qpos():
    n = 5
    goal = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1])
    qpos = np.zeros((n, 23))
    qpos[:, -9:] = goal                             # decoy: qpos's last 9 cols spell the goal
    bs = np.zeros((n, 9), np.int64)
    bs[1] = goal                                    # reachable via idx=0 -> reward 0 (+1)
    bs[2] = goal; bs[2, 4] = 1                      # one wrong -> -1 (+1 = 0)
    ds = _ds(n, qpos, bs)
    env = _fake_env([{"goal_button_states": goal}])
    _, r = OGBenchAdapter().relabel_infer_batch(ds, env, 1, 4, shift_reward=1.0)
    # rows 3,4: button_states all zero vs a goal with four 0s -> 4 matched -> -5 + 1 = -4.
    np.testing.assert_allclose(r, [1.0, 0.0, -4.0, -4.0])


def test_puzzle_batch_requires_button_states():
    ds = _ds(4, np.zeros((4, 23)))                  # no button_states column
    env = _fake_env([{"goal_button_states": np.zeros(9, np.int64)}])
    with pytest.raises(AssertionError, match="button_states"):
        OGBenchAdapter().relabel_infer_batch(ds, env, 1, 3)


def test_scene_batch_reads_next_row_physics():
    n = 5
    qpos = np.zeros((n, 25))
    bs = np.zeros((n, 2), np.int64)
    qpos[1, 14:17], qpos[1, 23], qpos[1, 24], bs[1] = [0.35, 0.05, 0.02], -0.16, 0.2, [1, 1]
    qpos[2, 14:17], qpos[2, 23], qpos[2, 24], bs[2] = [0.35, 0.05, 0.02], -0.16, 0.2, [0, 1]
    ds = _ds(n, qpos, bs)
    env = _fake_env([{"goal": _scene_goal()}])
    _, r = OGBenchAdapter().relabel_infer_batch(ds, env, 1, 4, shift_reward=1.0)
    # row 3/4: cube at origin (off), buttons 0/0 (both off), drawer 0 (off), window 0 (off)
    np.testing.assert_allclose(r, [1.0, 0.0, -4.0, -4.0])


def test_scene_batch_requires_button_states():
    ds = _ds(4, np.zeros((4, 25)))
    env = _fake_env([{"goal": _scene_goal()}])
    with pytest.raises(AssertionError, match="button_states"):
        OGBenchAdapter().relabel_infer_batch(ds, env, 1, 3)


def test_cube_batch_is_unchanged_by_the_dispatch():
    """Cube must stay byte-identical: same rows, same rewards as cube_relabel_reward on the
    next row's qpos, and a stray button_states column must NOT be consulted."""
    n = 6
    qpos = np.random.default_rng(1).uniform(-0.5, 0.5, size=(n, 21))
    goal = np.array([[0.1, 0.2, 0.3]])
    qpos[3, 14:17] = goal[0]
    ds = _ds(n, qpos)
    env = _fake_env([{"goal_xyzs": goal}])
    _, r = OGBenchAdapter().relabel_infer_batch(ds, env, 1, 5, shift_reward=1.0)
    want = cube_relabel_reward(np.asarray(ds["qpos"])[1:6], goal) + 1.0
    np.testing.assert_allclose(r, want)
    np.testing.assert_allclose(r, [0.0, 0.0, 1.0, 0.0, 0.0])


def test_shift_is_applied_verbatim_for_every_family():
    qpos = np.zeros((3, 15)); qpos[1, :2] = [1.0, 1.0]
    ds = _ds(3, qpos)
    env = _fake_env([{"goal_xy": np.array([1.0, 1.0])}])
    _, r0 = OGBenchAdapter().relabel_infer_batch(ds, env, 1, 2, shift_reward=0.0)
    _, r5 = OGBenchAdapter().relabel_infer_batch(ds, env, 1, 2, shift_reward=0.5)
    np.testing.assert_allclose(r0, [0.0, -1.0])
    np.testing.assert_allclose(r5, [0.5, -0.5])


# ======================================================================================
# 4. Integration on the real datasets. Skipped when ~/.ogbench/data lacks them.
# ======================================================================================
# (env_name, number of conditions k -> unshifted reward in [-k, 0])
REAL_ENVS = [("antmaze-medium-navigate-v0", 1), ("scene-play-v0", 5),
             ("puzzle-3x3-play-v0", 9), ("cube-single-play-v0", 1)]


def _positive_counts(adapter, train_ds, env, num_samples, shift):
    """Per-task count of relabelled rows at the shifted-maximum (every condition satisfied)."""
    out = {}
    for tid in range(1, len(env.unwrapped.task_infos) + 1):
        _, r = adapter.relabel_infer_batch(train_ds, env, tid, num_samples, shift_reward=shift)
        out[tid] = (r, int((r >= shift - 1e-6).sum()))
    return out


@pytest.mark.ogbench_dataset
@pytest.mark.parametrize("env_name,k", REAL_ENVS)
def test_relabel_on_the_real_dataset(env_name, k):
    if not _dataset_present(env_name):
        pytest.skip(f"{env_name} not in {DATA_DIR}")
    np.random.seed(0)
    adapter = OGBenchAdapter()
    env, train_ds, _ = adapter.make_env_and_datasets(
        {"env_name": env_name, "eval_relabel_size": 2000})
    base = adapter._column_backed(train_ds)
    assert "qpos" in base
    if env_name.startswith(("scene", "puzzle")):
        assert "button_states" in base, "loader must carry button_states for scene/puzzle"

    n = 2000
    res = _positive_counts(adapter, train_ds, env, n, shift=1.0)
    if not any(c > 0 for _, c in res.values()):
        n = 10_000
        res = _positive_counts(adapter, train_ds, env, n, shift=1.0)
    lines = []
    for tid, (r, c) in res.items():
        assert r.shape == (n,) and r.dtype == np.float32
        # unshifted in [-k, 0] integers -> shifted in [1-k, 1]
        assert np.all(np.abs(r - np.round(r)) < 1e-6)
        assert r.min() >= 1.0 - k - 1e-6 and r.max() <= 1.0 + 1e-6
        lines.append(f"task{tid}: {c}/{n} positives ({100.0 * c / n:.2f}%)")
    print(f"\n[relabel positives] {env_name}: " + "; ".join(lines))
    assert any(c > 0 for _, c in res.values()), (
        f"{env_name}: no positive relabel in {n} samples for ANY task -- " + "; ".join(lines))
