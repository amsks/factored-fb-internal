"""WindowedDataset: h-step windows for the RLDP latent-prediction loss.

Column k of future_actions is row min(idx + 1 + k, traj_end) and column k of
future_observations is row min(idx + 2 + k, traj_end), so windows never cross episodes.
"""

import numpy as np
import pytest

from impls.utils.datasets import Dataset, FBDataset, WindowedDataset, make_dataset

T, N_TRAJ, OBS, ACT = 10, 4, 6, 3
SIZE = T * N_TRAJ


def _base(cls=FBDataset):
    """Four length-10 trajectories with the row index in column 0."""
    rng = np.random.default_rng(0)
    terminals = np.zeros(SIZE, np.float32)
    terminals[np.arange(1, N_TRAJ + 1) * T - 1] = 1.0
    obs = rng.standard_normal((SIZE, OBS)).astype(np.float32)
    obs[:, 0] = np.arange(SIZE)  # row tag
    acts = rng.standard_normal((SIZE, ACT)).astype(np.float32)
    acts[:, 0] = np.arange(SIZE)  # row tag
    return cls.create(observations=obs, actions=acts, terminals=terminals)


def _cfg(**over):
    c = dict(dataset_class="WindowedDataset", horizon=4)
    c.update(over)
    return c


def _final_idxs(idxs):
    return (np.asarray(idxs) // T + 1) * T - 1


def test_make_dataset_dispatches_windowed():
    ds = make_dataset(_base(), _cfg())
    assert isinstance(ds, WindowedDataset)


def test_horizon_is_required():
    """A missing horizon raises."""
    with pytest.raises(KeyError, match="horizon"):
        make_dataset(_base(), _cfg(horizon=None)).sample(4)


@pytest.mark.parametrize("h", [2, 3, 5])
def test_shapes(h):
    ds = make_dataset(_base(), _cfg(horizon=h))
    b = ds.sample(7)
    assert b["future_observations"].shape == (7, h - 1, OBS)
    assert b["future_actions"].shape == (7, h - 1, ACT)
    assert b["future_observations"].dtype == np.float32
    assert b["future_actions"].dtype == np.float32


def test_still_emits_the_fb_training_batch():
    """The batch keeps every FBDataset key, shape and dtype."""
    ds = make_dataset(_base(), _cfg())
    b = ds.sample(9)
    assert b["observations"].shape == (9, OBS)
    assert b["next_observations"].shape == (9, OBS)
    assert b["actions"].shape == (9, ACT)
    assert b["rewards"].shape == (9, 1)
    assert b["terminals"].shape == (9,)
    assert np.all(b["terminals"] == 0.0)  # as FBDataset


def test_horizon_one_emits_no_future_keys():
    """h=1 gives a plain FB batch with no future keys."""
    ds = make_dataset(_base(), _cfg(horizon=1))
    b = ds.sample(5)
    assert "future_observations" not in b and "future_actions" not in b


def test_window_contents_are_the_actual_next_h_rows():
    base = _base()
    obs = np.asarray(base["observations"])
    acts = np.asarray(base["actions"])
    ds = make_dataset(base, _cfg(horizon=4))
    idxs = np.array([0, 1, 5, T + 2, 2 * T + 3])  # away from trajectory ends
    b = ds.sample(len(idxs), idxs=idxs)
    for k in range(3):  # h-1 future columns
        np.testing.assert_allclose(b["future_observations"][:, k], obs[idxs + 2 + k])
        np.testing.assert_allclose(b["future_actions"][:, k], acts[idxs + 1 + k])


def test_future_column_zero_starts_one_step_past_the_transition():
    """Column 0 pairs action a_{t+1} with observation s_{t+2}."""
    base = _base()
    obs = np.asarray(base["observations"])
    acts = np.asarray(base["actions"])
    ds = make_dataset(base, _cfg(horizon=3))
    idxs = np.array([0, 2, T + 1])
    b = ds.sample(len(idxs), idxs=idxs)
    np.testing.assert_allclose(b["future_observations"][:, 0], obs[idxs + 2])
    np.testing.assert_allclose(b["future_actions"][:, 0], acts[idxs + 1])
    assert not np.allclose(b["future_observations"][:, 0], b["next_observations"]), (
        "column 0 duplicates next_observations (off-by-one)"
    )


def test_window_near_a_trajectory_end_is_clamped_not_bled():
    """A window near a trajectory end clamps to its last row."""
    base = _base()
    obs = np.asarray(base["observations"])
    ds = make_dataset(base, _cfg(horizon=4))
    idx = np.array([T - 3])
    b = ds.sample(1, idxs=idx)
    got_o = b["future_observations"][0][:, 0]  # row tag
    got_a = b["future_actions"][0][:, 0]
    np.testing.assert_allclose(got_o, [T - 1, T - 1, T - 1])
    np.testing.assert_allclose(got_a, [T - 2, T - 1, T - 1])
    assert not np.any(got_o >= T) and not np.any(got_a >= T), (
        "window bled into the next episode"
    )
    np.testing.assert_allclose(b["future_observations"][0], obs[[T - 1, T - 1, T - 1]])


def test_clamping_to_the_trajectory_end_over_every_start_row():
    """For every start row, windows equal the clamped row formulas."""
    base = _base()
    obs = np.asarray(base["observations"])
    acts = np.asarray(base["actions"])
    h = 6
    ds = make_dataset(base, _cfg(horizon=h))
    idxs = np.arange(SIZE)
    b = ds.sample(SIZE, idxs=idxs)
    final = _final_idxs(idxs)
    for k in range(h - 1):
        np.testing.assert_allclose(
            b["future_observations"][:, k], obs[np.minimum(idxs + 2 + k, final)]
        )
        np.testing.assert_allclose(
            b["future_actions"][:, k], acts[np.minimum(idxs + 1 + k, final)]
        )


def test_every_window_row_stays_inside_its_own_trajectory():
    """Every window row stays in the anchor's trajectory and never goes back in time."""
    ds = make_dataset(_base(), _cfg(horizon=5))
    idxs = np.arange(SIZE)
    b = ds.sample(SIZE, idxs=idxs)
    for key in ("future_observations", "future_actions"):
        rows = b[key][:, :, 0]  # row tags, [SIZE, h-1]
        assert np.all(rows // T == (idxs // T)[:, None]), (
            "a window crossed into another episode"
        )
        assert np.all(rows >= idxs[:, None]), "a window went backwards in time"
        assert np.all(np.diff(rows, axis=1) >= 0), "a window is not monotone in time"


def test_observation_rows_lead_action_rows_by_one_step():
    """Observation row k is the state reached by action row k."""
    ds = make_dataset(_base(), _cfg(horizon=6))
    idxs = np.arange(SIZE)
    b = ds.sample(SIZE, idxs=idxs)
    final = _final_idxs(idxs)
    np.testing.assert_allclose(
        b["future_observations"][:, :, 0],
        np.minimum(b["future_actions"][:, :, 0] + 1, final[:, None]),
    )


def test_random_sampling_also_respects_boundaries():
    base = _base()
    ds = make_dataset(base, _cfg(horizon=5))
    np.random.seed(0)
    b = ds.sample(512)
    anchors = b["observations"][:, 0].astype(int)
    rows = b["future_observations"][:, :, 0].astype(int)
    assert np.all(rows // T == (anchors // T)[:, None])


def test_obs_mask_dims_blank_the_window_too():
    """obs_mask_dims also blanks the window observations."""
    ds = make_dataset(_base(), _cfg(horizon=4, obs_mask_dims=[2, 3]))
    b = ds.sample(8)
    for k in ("observations", "next_observations"):
        assert np.all(b[k][:, [2, 3]] == 0.0)
    assert np.all(b["future_observations"][:, :, [2, 3]] == 0.0)
    assert not np.all(b["future_observations"][:, :, 4] == 0.0)


def test_accepts_a_plain_dataset_base():
    """WindowedDataset accepts a plain Dataset base."""
    ds = make_dataset(_base(cls=Dataset), _cfg())
    b = ds.sample(4)
    assert b["future_observations"].shape == (4, 3, OBS)
