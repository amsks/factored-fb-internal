"""Spec 23 P0 round-trip test (mandatory per spec section 5): the
ExoRL -> GCDataset conversion must reproduce the source arrays exactly and
preserve trajectory-boundary structure. Two tiers:

1. A synthetic tiny buffer (deterministic, no network/download dependency)
   -- exercises the general conversion logic and the off-by-one guard.
2. The REAL downloaded RND episode 0, guarded by a path check so this file
   still collects (skipped, not erroring) if the dataset isn't present on
   the machine running pytest.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "exorl"))

from convert_exorl import convert_domain, _load_episode_transitions  # noqa: E402
from tests.jax_subprocess import run_jax  # noqa: E402

REAL_BUFFER_DIR = Path.home() / "exorl_data" / "exorl" / "datasets" / "point_mass_maze" / "rnd" / "buffer"


def _make_synthetic_episode(fn: Path, n: int, obs_dim: int, action_dim: int, rng: np.random.Generator):
    """Mimic exorl's replay_buffer.save_episode schema exactly: arrays of
    length n+1, action[0] a dummy zero row."""
    obs = rng.normal(size=(n + 1, obs_dim)).astype(np.float32)
    act = rng.normal(size=(n + 1, action_dim)).astype(np.float32)
    act[0] = 0.0
    reward = rng.normal(size=(n + 1, 1)).astype(np.float32)
    discount = np.ones((n + 1, 1), dtype=np.float32)
    physics = obs.astype(np.float64)
    np.savez_compressed(fn, observation=obs, action=act, reward=reward, discount=discount, physics=physics)
    return obs, act


def _make_synthetic_buffer(tmp_path: Path, episode_lens, obs_dim=4, action_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    buf_dir = tmp_path / "buffer"
    buf_dir.mkdir()
    raw = {}
    for i, n in enumerate(episode_lens):
        fn = buf_dir / f"episode_{i:06d}_{n}.npz"
        obs, act = _make_synthetic_episode(fn, n, obs_dim, action_dim, rng)
        raw[i] = (obs, act)
    return buf_dir, raw


def test_synthetic_round_trip_exact(tmp_path):
    episode_lens = [5, 5, 5, 5]  # 4 tiny episodes
    buf_dir, raw = _make_synthetic_buffer(tmp_path, episode_lens)

    out_prefix = tmp_path / "converted" / "point_mass_maze_rnd_toy"
    stats = convert_domain(buf_dir, out_prefix, val_episodes=1)

    assert stats.n_episodes_total == 4
    assert stats.n_episodes_train == 3
    assert stats.n_episodes_val == 1
    assert stats.n_transitions_train == 15  # 3 * 5
    assert stats.n_transitions_val == 5

    train = np.load(out_prefix.with_suffix(".npz"))
    val = np.load(Path(str(out_prefix) + "-val.npz"))

    # Exact reproduction of the source arrays, transition by transition,
    # for the train split (episodes 0, 1, 2).
    for ep_idx in range(3):
        obs, act = raw[ep_idx]
        n = episode_lens[ep_idx]
        row0 = ep_idx * n
        got_obs = train["observations"][row0 : row0 + n]
        got_next_obs = train["next_observations"][row0 : row0 + n]
        got_act = train["actions"][row0 : row0 + n]
        assert np.array_equal(got_obs, obs[:-1])
        assert np.array_equal(got_next_obs, obs[1:])
        assert np.array_equal(got_act, act[1:])  # dummy action[0] dropped

    # Val split reproduces episode 3 exactly.
    obs3, act3 = raw[3]
    assert np.array_equal(val["observations"], obs3[:-1])
    assert np.array_equal(val["next_observations"], obs3[1:])
    assert np.array_equal(val["actions"], act3[1:])


def test_terminal_structure_matches_episode_boundaries(tmp_path):
    episode_lens = [7, 3, 10]
    buf_dir, _raw = _make_synthetic_buffer(tmp_path, episode_lens)
    out_prefix = tmp_path / "converted" / "toy2"
    stats = convert_domain(buf_dir, out_prefix, val_episodes=0)

    data = np.load(out_prefix.with_suffix(".npz"))
    terminals = data["terminals"]
    assert terminals.shape[0] == sum(episode_lens)

    expected_terminal_rows = np.cumsum(episode_lens) - 1
    (got_terminal_rows,) = np.nonzero(terminals > 0)
    assert np.array_equal(got_terminal_rows, expected_terminal_rows)

    # No off-by-one: the vendored (jax-only) GCDataset must accept this
    # without its `terminal_locs[-1] == size - 1` assertion failing, and
    # must recover exactly these boundaries. GCDataset lives under
    # third_party/ogbench/impls (jax-only, third_party/ogbench/impls/utils
    # also collides with the repo's own top-level utils.py on sys.path) --
    # run this check in the isolated jax venv via the repo's existing
    # jax-subprocess test helper.
    npz_path = out_prefix.with_suffix(".npz")
    proc = run_jax(
        f"""
        import numpy as np
        from utils.datasets import Dataset, GCDataset
        data = np.load({str(npz_path)!r})
        ds = Dataset.create(**{{k: data[k] for k in data.files}})
        config = dict(
            value_p_curgoal=0.2, value_p_trajgoal=0.5, value_p_randomgoal=0.3, value_geom_sample=True,
            actor_p_curgoal=0.2, actor_p_trajgoal=0.5, actor_p_randomgoal=0.3, actor_geom_sample=True,
            gc_negative=True, p_aug=None, frame_stack=None, discount=0.99,
        )
        gcds = GCDataset(ds, config)
        expected = np.array({expected_terminal_rows.tolist()!r})
        assert np.array_equal(gcds.terminal_locs, expected), (gcds.terminal_locs, expected)
        expected_initial = np.concatenate([[0], expected[:-1] + 1])
        assert np.array_equal(gcds.initial_locs, expected_initial), (gcds.initial_locs, expected_initial)
        print("GCDATASET_BOUNDARY_CHECK_OK")
        """
    )
    assert "GCDATASET_BOUNDARY_CHECK_OK" in proc.stdout, (proc.stdout, proc.stderr)


def test_no_cross_episode_leakage_at_boundary(tmp_path):
    """The specific off-by-one defect class flagged for this task: a
    terminal row's next_observations must be the TRUE final state of its
    OWN episode, never the first observation of the following episode."""
    episode_lens = [5, 5]
    buf_dir, raw = _make_synthetic_buffer(tmp_path, episode_lens, seed=42)
    out_prefix = tmp_path / "converted" / "toy3"
    convert_domain(buf_dir, out_prefix, val_episodes=0)
    data = np.load(out_prefix.with_suffix(".npz"))

    ep0_obs, _ = raw[0]
    ep1_obs, _ = raw[1]
    last_row_of_ep0 = 4  # 0-indexed, episode 0 has 5 transitions (rows 0..4)
    next_obs_at_boundary = data["next_observations"][last_row_of_ep0]
    # Must equal episode 0's TRUE final state (obs[-1], i.e. obs index 5)...
    assert np.array_equal(next_obs_at_boundary, ep0_obs[-1])
    # ...and must NOT equal episode 1's first observation.
    assert not np.array_equal(next_obs_at_boundary, ep1_obs[0])


@pytest.mark.skipif(not REAL_BUFFER_DIR.exists(), reason="real ExoRL point_mass_maze/rnd buffer not downloaded")
def test_real_episode_round_trip():
    fn = sorted(REAL_BUFFER_DIR.glob("*.npz"))[0]
    obs, act, next_obs = _load_episode_transitions(fn)
    with np.load(fn) as f:
        raw_obs = f["observation"]
        raw_act = f["action"]
    n = raw_obs.shape[0] - 1
    assert obs.shape == (n, 4)
    assert act.shape == (n, 2)
    assert np.array_equal(obs, raw_obs[:-1])
    assert np.array_equal(next_obs, raw_obs[1:])
    assert np.array_equal(act, raw_act[1:])
    # dims 0:2 are position, per the layout verification in
    # third_party/exorl_dmc_tasks/PROVENANCE.md and task-report.md: the
    # first row's velocity must be exactly zero (episode reset, before any
    # action has been applied) -- a real-data sanity check on the layout,
    # not just a shape check.
    assert np.allclose(obs[0, 2:], 0.0)
