# tests/test_data.py
# These tests use synthetic .npz files — no real OGBench data needed.
import numpy as np
import tempfile
from pathlib import Path


def _make_fake_episode(obs_dim=10, action_dim=4, physics_dim=5, n=50):
    return {
        "observation": np.random.randn(n, obs_dim).astype(np.float32),
        "action": np.random.randn(n, action_dim).astype(np.float32),
        "physics": np.random.randn(n, physics_dim).astype(np.float32),
        "discount": np.ones(n, dtype=np.float32),
    }


def test_load_transitions():
    from data.ogbench import load_transitions
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_fake_episode()
        fname = Path(tmpdir) / "ep0.npz"
        np.savez(fname, **ep)

        storage = load_transitions([fname])
        assert storage["observation"].shape == (49, 10)   # N-1 transitions
        assert storage["action"].shape == (49, 4)
        assert "next" in storage
        assert storage["next"]["observation"].shape == (49, 10)
        assert storage["next"]["terminated"].shape == (49,)


def test_load_ogbench_data():
    from data.ogbench import load_ogbench_data
    with tempfile.TemporaryDirectory() as tmpdir:
        buf_dir = Path(tmpdir) / "antmaze-medium-navigate-v0" / "buffer"
        buf_dir.mkdir(parents=True)
        for i in range(3):
            np.savez(buf_dir / f"ep{i}.npz", **_make_fake_episode())

        replay_buffer = load_ogbench_data(
            domain="antmaze-medium-navigate-v0",
            data_path=tmpdir,
            load_n_episodes=2,
            device="cpu",
        )
        assert "train" in replay_buffer
        assert len(replay_buffer["train"]) == 2 * 49  # 2 episodes × 49 transitions
