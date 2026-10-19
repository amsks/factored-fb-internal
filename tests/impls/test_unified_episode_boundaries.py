"""Episode boundaries, goal masks and reward relabelling on the unified MetaWorld/ManiSkill layout."""

import numpy as np
import pytest

from env.unified import UnifiedAdapter, load_packed
from impls.utils.datasets import (
    FBDataset,
    HILPFutureDataset,
    WindowedDataset,
    make_dataset,
)

N_EP, T, OBS, ACT = 6, 25, 8, 3


@pytest.fixture
def packed(tmp_path):
    """Pack with episode id in obs[..., 0] and step in obs[..., 1]; success from step 10."""
    ep = np.repeat(np.arange(N_EP, dtype=np.float32)[:, None], T, axis=1)
    t = np.repeat(np.arange(T, dtype=np.float32)[None, :], N_EP, axis=0)
    obs = np.zeros((N_EP, T, OBS), np.float32)
    obs[..., 0], obs[..., 1] = ep, t
    obs[..., 2:] = np.random.RandomState(0).randn(N_EP, T, OBS - 2)
    nobs = np.roll(obs, -1, axis=1)
    nobs[:, -1] = obs[:, 0]  # last row's next_obs = a reset frame
    act = (
        np.random.RandomState(1).uniform(-2, 2, size=(N_EP, T, ACT)).astype(np.float32)
    )
    success = (t >= 10).astype(bool)
    reward = t / T
    p = tmp_path / "pack.npz"
    np.savez(
        p,
        observation=obs,
        action=act,
        next_observation=nobs,
        reward=reward,
        terminal=np.zeros((N_EP, T), bool),
        timeout=(t == T - 1),
        success=success,
        achieved_goal=np.zeros((N_EP, T, 3), np.float32),
    )
    return str(p)


def _base(packed):
    d = load_packed(packed)
    return FBDataset.create(**d)


def test_load_packed_marks_every_episode_end_and_keeps_success(packed):
    d = load_packed(packed)
    assert d["observations"].shape == (N_EP * T, OBS)
    (ends,) = np.nonzero(d["terminals"] > 0)
    assert ends.tolist() == [T * (i + 1) - 1 for i in range(N_EP)]
    assert "success" in d and d["success"].dtype == np.float32
    assert (
        d["success"].reshape(N_EP, T)[:, 10:].all()
        and not d["success"].reshape(N_EP, T)[:, :10].any()
    )
    assert np.abs(d["actions"]).max() <= 1.0  # clipped


def test_hilp_future_never_leaves_the_anchor_episode(packed):
    np.random.seed(0)
    ds = make_dataset(
        _base(packed),
        dict(dataset_class="HILPFutureDataset", goal_future=0.99, p_randomgoal=0.0),
    )
    assert isinstance(ds, HILPFutureDataset)
    # p = 0.01: most geometric offsets exceed T and must clamp.
    for _ in range(20):
        b = ds.sample(256)
        fut = b["hilp_future_observations"]
        assert np.array_equal(fut[:, 0], b["observations"][:, 0]), (
            "future from another episode"
        )
        assert (fut[:, 1] >= b["observations"][:, 1]).all(), "future behind the anchor"
    # Clamp to the episode's last row, not the next reset frame.
    b = ds.sample(4096)
    assert (b["hilp_future_observations"][:, 1] == T - 1).mean() > 0.5


@pytest.mark.parametrize("h", [2, 5])
def test_rldp_window_never_leaves_the_anchor_episode(packed, h):
    np.random.seed(0)
    ds = make_dataset(_base(packed), dict(dataset_class="WindowedDataset", horizon=h))
    assert isinstance(ds, WindowedDataset)
    for _ in range(20):
        b = ds.sample(256)
        fo = b["future_observations"]  # [B, h-1, obs]
        assert fo.shape[1] == h - 1
        anchor_ep = b["observations"][:, 0][:, None]
        assert np.array_equal(fo[..., 0], np.broadcast_to(anchor_ep, fo[..., 0].shape))
        assert (fo[..., 1] >= b["observations"][:, 1][:, None]).all()
        assert (fo[..., 1] <= T - 1).all()


def test_hilp_future_dataset_applies_the_goal_mask_union(packed):
    np.random.seed(0)
    ds = make_dataset(
        _base(packed),
        dict(
            dataset_class="HILPFutureDataset",
            goal_future=0.99,
            p_randomgoal=0.375,
            obs_mask_dims=[5, 6],
            goal_mask_dims=[6, 7],
        ),
    )
    b = ds.sample(128)
    for k in ("observations", "next_observations", "hilp_future_observations"):
        assert (b[k][:, [5, 6, 7]] == 0.0).all(), f"{k} not blanked"
        assert (b[k][:, 2:5] != 0.0).any(), f"{k} over-blanked"
    # same union as WindowedDataset
    w = make_dataset(
        _base(packed),
        dict(
            dataset_class="WindowedDataset",
            horizon=3,
            obs_mask_dims=[5, 6],
            goal_mask_dims=[6, 7],
        ),
    )
    bw = w.sample(64)
    assert (bw["future_observations"][:, :, [5, 6, 7]] == 0.0).all()


class _Cfg(dict):
    pass


def test_unified_relabel_batch_is_a_masked_zero_one_task_signal(packed):
    np.random.seed(0)
    ad = UnifiedAdapter()
    ad._cfg = dict(
        obs_mask_dims=[5, 6], goal_mask_dims=[6, 7]
    )  # as set by make_env_and_datasets
    base = _base(packed)
    wrapped = make_dataset(
        base, dict(dataset_class="HILPFutureDataset", goal_future=0.99)
    )
    obs, r = ad.relabel_infer_batch(
        wrapped, env=None, task_id=1, num_samples=500, shift_reward=1.0
    )
    assert obs.shape == (500, OBS) and r.shape == (500,)
    assert set(np.unique(r).tolist()) <= {0.0, 1.0}
    # r == 1 on rows whose stored success flag fires (anchor step >= 10).
    assert np.array_equal(r == 1.0, (obs[:, 1] >= 11) | (obs[:, 1] == 0))
    assert (obs[:, [5, 6, 7]] == 0.0).all() and (obs[:, 2:5] != 0.0).any()
    assert 0.4 < r.mean() < 0.8  # 15/25 steps succeed
    # Dense mode: stored reward + shift
    ad._cfg["eval_relabel_reward"] = "dense"
    _, rd = ad.relabel_infer_batch(base, None, 1, 100, shift_reward=1.0)
    assert rd.min() >= 1.0 and rd.max() <= 2.0
    # Success mode raises on a pack without the success flag
    ad._cfg["eval_relabel_reward"] = "success"
    d = load_packed(packed)
    d.pop("success")
    with pytest.raises(ValueError, match="success"):
        ad.relabel_infer_batch(FBDataset.create(**d), None, 1, 10)
