"""Episode boundaries and masks on the UNIFIED (MetaWorld / ManiSkill packed npz) layout.

The unified adapter flattens [N, T, ...] packs and marks each episode's LAST row in
`terminals` (env/unified.py::load_packed). The two baseline samplers that read rows AHEAD
of the anchor -- HILP's geometric hindsight future (HILPFutureDataset) and RLDP's h-step
self-prediction window (WindowedDataset) -- must clamp to that boundary, or a future drawn
past it is the next episode's reset frame. Also pinned here: the goal-mask union those
samplers apply, and the unified reward-relabel batch (Route B) that
scripts/launch_mwms_baselines.sh switches on for every critic.

Pure numpy + a tiny synthetic pack; no env, no jax compilation.
"""
import numpy as np
import pytest

from env.unified import UnifiedAdapter, load_packed
from impls.utils.datasets import (FBDataset, HILPFutureDataset, WindowedDataset,
                                  make_dataset)

N_EP, T, OBS, ACT = 6, 25, 8, 3


@pytest.fixture
def packed(tmp_path):
    """A pack in the HF layout: obs[..., 0] = episode id, obs[..., 1] = step index, so a row's
    episode is readable off the observation itself; success fires from step 10 on."""
    ep = np.repeat(np.arange(N_EP, dtype=np.float32)[:, None], T, axis=1)
    t = np.repeat(np.arange(T, dtype=np.float32)[None, :], N_EP, axis=0)
    obs = np.zeros((N_EP, T, OBS), np.float32)
    obs[..., 0], obs[..., 1] = ep, t
    obs[..., 2:] = np.random.RandomState(0).randn(N_EP, T, OBS - 2)
    nobs = np.roll(obs, -1, axis=1)
    nobs[:, -1] = obs[:, 0]                       # last row's next_obs = a reset frame
    act = np.random.RandomState(1).uniform(-2, 2, size=(N_EP, T, ACT)).astype(np.float32)
    success = (t >= 10).astype(bool)
    reward = t / T
    p = tmp_path / "pack.npz"
    np.savez(p, observation=obs, action=act, next_observation=nobs, reward=reward,
             terminal=np.zeros((N_EP, T), bool), timeout=(t == T - 1), success=success,
             achieved_goal=np.zeros((N_EP, T, 3), np.float32))
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
    assert d["success"].reshape(N_EP, T)[:, 10:].all() and not d["success"].reshape(N_EP, T)[:, :10].any()
    assert np.abs(d["actions"]).max() <= 1.0     # clipped


def test_hilp_future_never_leaves_the_anchor_episode(packed):
    np.random.seed(0)
    ds = make_dataset(_base(packed), dict(dataset_class="HILPFutureDataset", hilp_future=0.99,
                                           hilp_p_randomgoal=0.0))
    assert isinstance(ds, HILPFutureDataset)
    # p = 0.01 => the geometric offset is > T for most draws: every one must clamp.
    for _ in range(20):
        b = ds.sample(256)
        fut = b["hilp_future_observations"]
        assert np.array_equal(fut[:, 0], b["observations"][:, 0]), "future from another episode"
        assert (fut[:, 1] >= b["observations"][:, 1]).all(), "future behind the anchor"
    # and the clamp lands on the episode's own last row, not the next reset frame
    b = ds.sample(4096)
    assert (b["hilp_future_observations"][:, 1] == T - 1).mean() > 0.5


@pytest.mark.parametrize("h", [2, 5])
def test_rldp_window_never_leaves_the_anchor_episode(packed, h):
    np.random.seed(0)
    ds = make_dataset(_base(packed), dict(dataset_class="WindowedDataset", horizon=h))
    assert isinstance(ds, WindowedDataset)
    for _ in range(20):
        b = ds.sample(256)
        fo = b["future_observations"]                       # [B, h-1, obs]
        assert fo.shape[1] == h - 1
        anchor_ep = b["observations"][:, 0][:, None]
        assert np.array_equal(fo[..., 0], np.broadcast_to(anchor_ep, fo[..., 0].shape))
        assert (fo[..., 1] >= b["observations"][:, 1][:, None]).all()
        assert (fo[..., 1] <= T - 1).all()


def test_hilp_future_dataset_applies_the_goal_mask_union(packed):
    np.random.seed(0)
    ds = make_dataset(_base(packed), dict(dataset_class="HILPFutureDataset", hilp_future=0.99,
                                           hilp_p_randomgoal=0.375,
                                           obs_mask_dims=[5, 6], goal_mask_dims=[6, 7]))
    b = ds.sample(128)
    for k in ("observations", "next_observations", "hilp_future_observations"):
        assert (b[k][:, [5, 6, 7]] == 0.0).all(), f"{k} not blanked"
        assert (b[k][:, 2:5] != 0.0).any(), f"{k} over-blanked"
    # the same union WindowedDataset already applied
    w = make_dataset(_base(packed), dict(dataset_class="WindowedDataset", horizon=3,
                                          obs_mask_dims=[5, 6], goal_mask_dims=[6, 7]))
    bw = w.sample(64)
    assert (bw["future_observations"][:, :, [5, 6, 7]] == 0.0).all()


class _Cfg(dict):
    pass


def test_unified_relabel_batch_is_a_masked_zero_one_task_signal(packed):
    np.random.seed(0)
    ad = UnifiedAdapter()
    ad._cfg = dict(obs_mask_dims=[5, 6], goal_mask_dims=[6, 7])   # what make_env_and_datasets stashes
    base = _base(packed)
    wrapped = make_dataset(base, dict(dataset_class="HILPFutureDataset", hilp_future=0.99))
    obs, r = ad.relabel_infer_batch(wrapped, env=None, task_id=1, num_samples=500,
                                    shift_reward=1.0)
    assert obs.shape == (500, OBS) and r.shape == (500,)
    assert set(np.unique(r).tolist()) <= {0.0, 1.0}
    # r == 1 exactly on rows whose stored success flag fires (anchor step >= 10). The returned
    # frame is the NEXT observation: step t+1, or the reset frame (step 0) after the last row.
    assert np.array_equal(r == 1.0, (obs[:, 1] >= 11) | (obs[:, 1] == 0))
    assert (obs[:, [5, 6, 7]] == 0.0).all() and (obs[:, 2:5] != 0.0).any()
    assert 0.4 < r.mean() < 0.8                       # 15/25 steps succeed
    # dense mode: the stored reward + shift
    ad._cfg["eval_relabel_reward"] = "dense"
    _, rd = ad.relabel_infer_batch(base, None, 1, 100, shift_reward=1.0)
    assert rd.min() >= 1.0 and rd.max() <= 2.0
    # success mode refuses a pack without the flag instead of relabelling with garbage
    ad._cfg["eval_relabel_reward"] = "success"
    d = load_packed(packed); d.pop("success")
    with pytest.raises(ValueError, match="success"):
        ad.relabel_infer_batch(FBDataset.create(**d), None, 1, 10)
