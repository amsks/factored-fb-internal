"""Tests for trajectory-future goal sampling (evals/traj_goals.py)."""
import numpy as np

from evals.traj_goals import episode_last_index, sample_future_idx


def _timestep(ep_lens):
    return np.concatenate([np.arange(L) for L in ep_lens])


def test_episode_last_index():
    ts = _timestep([3, 4, 2])          # flat: 0,1,2 | 0,1,2,3 | 0,1
    last = episode_last_index(ts)
    assert last.tolist() == [2, 2, 2, 6, 6, 6, 6, 8, 8]


def test_future_never_crosses_episode_boundary():
    ts = _timestep([5, 5, 5])
    last = episode_last_index(ts)
    rng = np.random.default_rng(0)
    idx = np.arange(15)
    for _ in range(50):
        j = sample_future_idx(idx, last, gamma=0.98, rng=rng)
        assert (j >= idx + 0).all()
        assert (j <= last[idx]).all()           # same episode
        assert (j > idx).all() | (idx == last[idx]).any()  # strictly future unless at end


def test_geometric_mass_decays():
    ts = _timestep([1000])
    last = episode_last_index(ts)
    rng = np.random.default_rng(1)
    j = sample_future_idx(np.zeros(20000, dtype=int), last, gamma=0.9, rng=rng)
    k = j  # idx=0 so k=j
    # mean of Geom(0.1) is 10; truncation at 999 negligible here
    assert 8.5 < k.mean() < 11.5
    assert (k >= 1).all()


def test_truncation_piles_on_episode_end():
    ts = _timestep([3])
    last = episode_last_index(ts)
    rng = np.random.default_rng(2)
    j = sample_future_idx(np.zeros(5000, dtype=int), last, gamma=0.99, rng=rng)
    assert (j <= 2).all()
    assert (j == 2).mean() > 0.9   # gamma=0.99 => almost all mass truncates to the end


def _tiny_buffer(ep_lens, obs_dim=4):
    """DictBuffer with deterministic next/observation = flat index broadcast."""
    import torch
    from buffers.transition import DictBuffer
    n = sum(ep_lens)
    data = {
        "observation": np.zeros((n, obs_dim), np.float32),
        "action": np.zeros((n, 2), np.float32),
        "next": {
            "observation": np.tile(np.arange(n, dtype=np.float32)[:, None], (1, obs_dim)),
            "terminated": np.zeros((n, 1), bool),
        },
        "timestep": _timestep(ep_lens).astype(np.int32),
    }
    buf = DictBuffer(capacity=n, device="cpu")
    buf.extend(data)
    return buf


def test_buffer_future_gamma_attaches_aligned_future():
    import torch
    buf = _tiny_buffer([10, 10])
    torch.manual_seed(0)
    b = buf.sample(64, future_gamma=0.9)
    assert "future" in b and "observation" in b["future"]
    cur = b["next"]["observation"][:, 0]          # encodes the flat index
    fut = b["future"]["observation"][:, 0]
    assert (fut >= cur).all()                      # future, same or later
    assert (fut <= 19).all()
    # same-episode: indices <10 stay <10
    first_ep = cur < 10
    assert (fut[first_ep] < 10).all()


def test_buffer_default_has_no_future_key():
    import torch
    buf = _tiny_buffer([10])
    torch.manual_seed(0)
    b = buf.sample(8)
    assert "future" not in b
