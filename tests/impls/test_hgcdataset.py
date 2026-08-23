"""HGCDataset — hierarchical goal sampling for HIQL (workstream 2).

Faithful port of vendored third_party/ogbench/impls/utils/datasets.py:HGCDataset,
state-only, matching the seam's GCDataset conventions. Nothing vendored is imported.

The three keys HIQL needs and nothing else in the seam produces:
  low_actor_goals   -- the state subgoal_steps ahead, clamped to the trajectory end
  high_actor_goals  -- a future-or-random goal for the high-level policy
  high_actor_targets-- the subgoal the high policy is trained to predict for that goal
"""
import numpy as np
import pytest

from impls.utils.datasets import Dataset, make_dataset

T, N_TRAJ, OBS, ACT = 20, 5, 6, 3
SIZE = T * N_TRAJ


def _base():
    rng = np.random.default_rng(0)
    terminals = np.zeros(SIZE, np.float32)
    terminals[np.arange(1, N_TRAJ + 1) * T - 1] = 1.0
    return Dataset.create(
        observations=rng.standard_normal((SIZE, OBS)).astype(np.float32),
        next_observations=rng.standard_normal((SIZE, OBS)).astype(np.float32),
        actions=rng.standard_normal((SIZE, ACT)).astype(np.float32),
        terminals=terminals,
    )


def _cfg(**over):
    c = dict(dataset_class="HGCDataset", discount=0.99, subgoal_steps=5, gc_negative=True,
             value_p_curgoal=0.2, value_p_trajgoal=0.5, value_p_randomgoal=0.3,
             value_geom_sample=True, actor_p_curgoal=0.0, actor_p_trajgoal=1.0,
             actor_p_randomgoal=0.0, actor_geom_sample=False)
    c.update(over)
    return c


def test_make_dataset_dispatches_hgc():
    from impls.utils.datasets import HGCDataset
    ds = make_dataset(_base(), _cfg())
    assert isinstance(ds, HGCDataset)


def test_emits_the_three_hierarchical_keys():
    ds = make_dataset(_base(), _cfg())
    b = ds.sample(32)
    for k in ("low_actor_goals", "high_actor_goals", "high_actor_targets"):
        assert k in b, f"missing {k}"
        assert b[k].shape == (32, OBS)
    # GCDataset's own contract still holds
    for k in ("observations", "next_observations", "actions", "value_goals", "rewards", "masks"):
        assert k in b


def test_low_actor_goal_is_subgoal_steps_ahead_clamped_to_the_trajectory_end():
    """vendored: low_goal_idxs = min(idxs + subgoal_steps, final_state_idxs). A subgoal must
    never cross into the next trajectory."""
    base = _base()
    ds = make_dataset(base, _cfg(subgoal_steps=5))
    obs = np.asarray(base["observations"])
    idxs = np.array([0, 3, T - 1, T + 2, SIZE - 1])
    b = ds.sample(len(idxs), idxs=idxs)
    final = (idxs // T + 1) * T - 1
    expected = obs[np.minimum(idxs + 5, final)]
    assert np.allclose(b["low_actor_goals"], expected)


def test_high_actor_target_never_overshoots_its_goal():
    """vendored: high_traj_target_idxs = min(idxs + subgoal_steps, high_traj_goal_idxs).
    The subgoal the high policy predicts must lie between the state and its goal."""
    base = _base()
    ds = make_dataset(base, _cfg())
    obs = np.asarray(base["observations"])
    b = ds.sample(256)
    # every emitted target must be an actual dataset row
    for row in b["high_actor_targets"]:
        assert np.isclose(obs, row).all(-1).any()


def _idx_of(obs, rows):
    """The dataset row index each emitted observation was gathered from. The fixture's rows
    are iid gaussian, so the match is unique."""
    out = []
    for r in rows:
        m = np.flatnonzero((obs == r).all(-1))
        assert m.size == 1, "ambiguous row match -- the fixture is not unique"
        out.append(int(m[0]))
    return np.asarray(out)


def test_high_actor_target_is_exactly_min_of_the_subgoal_step_and_the_goal():
    """`test_high_actor_target_never_overshoots_its_goal` above asserts only that the emitted
    target is SOME dataset row, which is true of any index gather -- clamping to the
    trajectory end instead of to the goal passes it (verified by mutation). Pin the vendored
    formula itself:

        high_traj_target_idxs = min(idxs + subgoal_steps, high_traj_goal_idxs)

    With actor_p_randomgoal=0 every goal is a trajectory goal, so the relation is exact."""
    base = _base()
    sg = 5
    ds = make_dataset(base, _cfg(subgoal_steps=sg, actor_p_randomgoal=0.0))
    obs = np.asarray(base["observations"])
    np.random.seed(0)
    idxs = np.arange(SIZE)
    b = ds.sample(len(idxs), idxs=idxs)
    g = _idx_of(obs, np.asarray(b["high_actor_goals"]))
    t = _idx_of(obs, np.asarray(b["high_actor_targets"]))

    # discriminating guard: unless some goal lands INSIDE the subgoal horizon, clamping to
    # the goal and clamping to the trajectory end agree everywhere and the pin is vacuous
    assert (g < idxs + sg).any(), "no goal falls inside the subgoal horizon -- vacuous"
    np.testing.assert_array_equal(t, np.minimum(idxs + sg, g))


def test_subgoal_steps_is_required():
    with pytest.raises(KeyError):
        make_dataset(_base(), _cfg(subgoal_steps=None)).sample(4)
