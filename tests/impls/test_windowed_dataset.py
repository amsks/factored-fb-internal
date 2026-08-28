"""WindowedDataset — h-step windowed batches for the RLDP self-predictive head.

The whole point of this class is TRAJECTORY BOUNDARIES. A window that silently spans
two episodes trains the predictor to roll a dynamics model into an unrelated
trajectory, and nothing fails loudly — the shapes are right and the loss is finite.
So the boundary behaviour is pinned here first, in the same way
tests/impls/test_hgcdataset.py pins HGCDataset's `low_actor_goals` clamp.

Contract (td_jepa's parallel.py layout; clamping per HGCDataset's convention):
  * future_observations / future_actions have shape [B, h-1, ...];
  * future_actions      column k (0-based) is dataset row `min(idx + 1 + k, traj_final_idx)` (a_{t+1+k});
  * future_observations column k (0-based) is dataset row `min(idx + 2 + k, traj_final_idx)` (s_{t+2+k})
    -- the observation column LEADS the action column by one step, so column k is the
    aligned rollout pair (a_{t+1+k} -> the state it reaches, s_{t+2+k}); prepending the
    batch's own (action, next_observations) gives RLDP's h-step rollout, step k -> s_{t+k}
    (the 2026-08-25 off-by-one fix: both windows used to sit at rows t+1..t+h-1, so the
    SP target sequence duplicated s_{t+1} and never reached s_{t+h});
  * a window starting near the end of a trajectory REPEATS that trajectory's last row
    rather than reading the first rows of the next episode.
"""
import numpy as np
import pytest

from impls.utils.datasets import Dataset, FBDataset, WindowedDataset, make_dataset

T, N_TRAJ, OBS, ACT = 10, 4, 6, 3
SIZE = T * N_TRAJ


def _base(cls=FBDataset):
    """Four length-10 trajectories. Observations are row-index-tagged (column 0 == row)
    so a window's contents can be checked by value, not just by shape."""
    rng = np.random.default_rng(0)
    terminals = np.zeros(SIZE, np.float32)
    terminals[np.arange(1, N_TRAJ + 1) * T - 1] = 1.0
    obs = rng.standard_normal((SIZE, OBS)).astype(np.float32)
    obs[:, 0] = np.arange(SIZE)                      # row tag
    acts = rng.standard_normal((SIZE, ACT)).astype(np.float32)
    acts[:, 0] = np.arange(SIZE)                     # row tag
    return cls.create(observations=obs, actions=acts, terminals=terminals)


def _cfg(**over):
    c = dict(dataset_class="WindowedDataset", horizon=4)
    c.update(over)
    return c


def _final_idxs(idxs):
    return (np.asarray(idxs) // T + 1) * T - 1


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
def test_make_dataset_dispatches_windowed():
    ds = make_dataset(_base(), _cfg())
    assert isinstance(ds, WindowedDataset)


def test_horizon_is_required():
    """A missing horizon must be a named error, not a silent h=1 that quietly turns the
    self-predictive loss into a one-step loss (exactly HGCDataset's `subgoal_steps` rule)."""
    with pytest.raises(KeyError, match="horizon"):
        make_dataset(_base(), _cfg(horizon=None)).sample(4)


# --------------------------------------------------------------------------
# shapes + the FB batch contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("h", [2, 3, 5])
def test_shapes(h):
    ds = make_dataset(_base(), _cfg(horizon=h))
    b = ds.sample(7)
    assert b["future_observations"].shape == (7, h - 1, OBS)
    assert b["future_actions"].shape == (7, h - 1, ACT)
    assert b["future_observations"].dtype == np.float32
    assert b["future_actions"].dtype == np.float32


def test_still_emits_the_fb_training_batch():
    """RLDP is FB plus a head: everything FBDataset emits must still be here, same
    shapes and same dtypes, or every actor family breaks on this dataset."""
    ds = make_dataset(_base(), _cfg())
    b = ds.sample(9)
    assert b["observations"].shape == (9, OBS)
    assert b["next_observations"].shape == (9, OBS)
    assert b["actions"].shape == (9, ACT)
    assert b["rewards"].shape == (9, 1)
    assert b["terminals"].shape == (9,)
    assert np.all(b["terminals"] == 0.0)             # always-gamma, as FBDataset


def test_horizon_one_emits_no_future_keys():
    """h=1 is the degenerate no-window case (the torch reference's `horizon > 1` guard):
    the batch must be exactly an FB batch, with no zero-width future arrays for the
    critic to have to special-case."""
    ds = make_dataset(_base(), _cfg(horizon=1))
    b = ds.sample(5)
    assert "future_observations" not in b and "future_actions" not in b


# --------------------------------------------------------------------------
# the window is the actual next-h dataset rows
# --------------------------------------------------------------------------
def test_window_contents_are_the_actual_next_h_rows():
    base = _base()
    obs = np.asarray(base["observations"])
    acts = np.asarray(base["actions"])
    ds = make_dataset(base, _cfg(horizon=4))
    idxs = np.array([0, 1, 5, T + 2, 2 * T + 3])     # all far from any trajectory end
    b = ds.sample(len(idxs), idxs=idxs)
    for k in range(3):                               # h-1 == 3 future columns
        np.testing.assert_allclose(b["future_observations"][:, k], obs[idxs + 2 + k])
        np.testing.assert_allclose(b["future_actions"][:, k], acts[idxs + 1 + k])


def test_future_column_zero_starts_one_step_past_the_transition():
    """Column 0 of the observation window is s_{t+2} -- the row AFTER the one
    `next_observations` carries -- paired with action column 0 = a_{t+1}, the action
    taken AT s_{t+1}. The pre-fix window duplicated `next_observations` in column 0
    (2026-08-25 off-by-one), which taught the predictor that a_{t+1} does not move the
    embedding; this is the regression pin."""
    base = _base()
    obs = np.asarray(base["observations"])
    acts = np.asarray(base["actions"])
    ds = make_dataset(base, _cfg(horizon=3))
    idxs = np.array([0, 2, T + 1])
    b = ds.sample(len(idxs), idxs=idxs)
    np.testing.assert_allclose(b["future_observations"][:, 0], obs[idxs + 2])
    np.testing.assert_allclose(b["future_actions"][:, 0], acts[idxs + 1])
    assert not np.allclose(b["future_observations"][:, 0], b["next_observations"]), \
        "column 0 duplicates next_observations -- the pre-fix off-by-one is back"


# --------------------------------------------------------------------------
# THE point: boundary clamping
# --------------------------------------------------------------------------
def test_window_near_a_trajectory_end_is_clamped_not_bled():
    """idx = T-3 with h=4: the observation window wants rows T-1, T, T+1 and the action
    window rows T-2, T-1, T -- rows T and T+1 are in the NEXT episode. Both must clamp
    to T-1 (this trajectory's own last row)."""
    base = _base()
    obs = np.asarray(base["observations"])
    ds = make_dataset(base, _cfg(horizon=4))
    idx = np.array([T - 3])
    b = ds.sample(1, idxs=idx)
    got_o = b["future_observations"][0][:, 0]        # the row tag
    got_a = b["future_actions"][0][:, 0]
    np.testing.assert_allclose(got_o, [T - 1, T - 1, T - 1])
    np.testing.assert_allclose(got_a, [T - 2, T - 1, T - 1])
    assert not np.any(got_o >= T) and not np.any(got_a >= T), "window bled into the next episode"
    np.testing.assert_allclose(b["future_observations"][0], obs[[T - 1, T - 1, T - 1]])


def test_clamping_matches_the_hgc_convention_over_every_start_row():
    """Exhaustive: for EVERY dataset row and a horizon long enough to overrun, the action
    window must equal `min(idx + 1 + k, final_state_idx)` and the observation window
    `min(idx + 2 + k, final_state_idx)` -- HGCDataset's clamp, applied to all h-1
    columns. A version that clamped to `size - 1` instead of the trajectory end would
    pass every interior row and fail here."""
    base = _base()
    obs = np.asarray(base["observations"])
    acts = np.asarray(base["actions"])
    h = 6
    ds = make_dataset(base, _cfg(horizon=h))
    idxs = np.arange(SIZE)
    b = ds.sample(SIZE, idxs=idxs)
    final = _final_idxs(idxs)
    for k in range(h - 1):
        np.testing.assert_allclose(b["future_observations"][:, k], obs[np.minimum(idxs + 2 + k, final)])
        np.testing.assert_allclose(b["future_actions"][:, k], acts[np.minimum(idxs + 1 + k, final)])


def test_every_window_row_stays_inside_its_own_trajectory():
    """Stated as the property rather than the formula: every row of every window must
    carry the same trajectory id as the anchor, and never move backwards in time."""
    ds = make_dataset(_base(), _cfg(horizon=5))
    idxs = np.arange(SIZE)
    b = ds.sample(SIZE, idxs=idxs)
    for key in ("future_observations", "future_actions"):
        rows = b[key][:, :, 0]                       # row tags, [SIZE, h-1]
        assert np.all(rows // T == (idxs // T)[:, None]), "a window crossed into another episode"
        assert np.all(rows >= idxs[:, None]), "a window went backwards in time"
        assert np.all(np.diff(rows, axis=1) >= 0), "a window is not monotone in time"


def test_observation_rows_lead_action_rows_by_one_step():
    """The alignment invariant of the fixed window: column k's observation is the state
    REACHED by column k's action, i.e. obs row = min(action row + 1, traj_final_idx),
    everywhere -- interior AND clamped. Equal rows (the pre-fix layout) would silently
    pair each action with the state it was taken FROM, the classic off-by-one in
    windowed samplers."""
    ds = make_dataset(_base(), _cfg(horizon=6))
    idxs = np.arange(SIZE)
    b = ds.sample(SIZE, idxs=idxs)
    final = _final_idxs(idxs)
    np.testing.assert_allclose(
        b["future_observations"][:, :, 0],
        np.minimum(b["future_actions"][:, :, 0] + 1, final[:, None]))


# --------------------------------------------------------------------------
# random sampling + masking
# --------------------------------------------------------------------------
def test_random_sampling_also_respects_boundaries():
    base = _base()
    ds = make_dataset(base, _cfg(horizon=5))
    np.random.seed(0)
    b = ds.sample(512)
    anchors = b["observations"][:, 0].astype(int)
    rows = b["future_observations"][:, :, 0].astype(int)
    assert np.all(rows // T == (anchors // T)[:, None])


def test_obs_mask_dims_blank_the_window_too():
    """obs_mask_dims exists to force goal information through the conditioning vector
    (datasets.py:GCDataset). A dim blanked in observations but left visible inside the
    window is the same leak by another door."""
    ds = make_dataset(_base(), _cfg(horizon=4, obs_mask_dims=[2, 3]))
    b = ds.sample(8)
    for k in ("observations", "next_observations"):
        assert np.all(b[k][:, [2, 3]] == 0.0)
    assert np.all(b["future_observations"][:, :, [2, 3]] == 0.0)
    assert not np.all(b["future_observations"][:, :, 4] == 0.0)


def test_accepts_a_plain_dataset_base():
    """make_dataset is handed whatever env_utils built; WindowedDataset must not depend
    on its base being an FBDataset."""
    ds = make_dataset(_base(cls=Dataset), _cfg())
    b = ds.sample(4)
    assert b["future_observations"].shape == (4, 3, OBS)
