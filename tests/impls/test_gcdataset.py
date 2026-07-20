import numpy as np
from impls.utils.datasets import Dataset, GCDataset

_CFG = dict(discount=0.99, gc_negative=True,
            value_p_curgoal=0.2, value_p_trajgoal=0.5, value_p_randomgoal=0.3, value_geom_sample=True,
            actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0, actor_geom_sample=False)


def _toy(size=20, obs_dim=4, act_dim=3):
    # two trajectories of length 10; terminals at 9 and 19.
    terminals = np.zeros(size, np.float32); terminals[9] = 1.0; terminals[19] = 1.0
    return Dataset.create(
        observations=np.arange(size * obs_dim, dtype=np.float32).reshape(size, obs_dim),
        actions=np.zeros((size, act_dim), np.float32),
        terminals=terminals,
    )


def test_reward_mask_identities_and_shapes():
    ds = GCDataset(dataset=_toy(), config=_CFG)
    np.random.seed(0)
    idxs = np.array([0, 3, 9, 12])
    b = ds.sample(len(idxs), idxs=idxs)
    assert b["value_goals"].shape == (4, 4) and b["actor_goals"].shape == (4, 4)
    assert b["rewards"].shape == (4,) and b["masks"].shape == (4,)
    # gc_negative: reward = success - 1; mask = 1 - success; success = (idx == value_goal_idx)
    successes = b["masks"] == 0.0
    assert np.allclose(b["rewards"], successes.astype(np.float32) - 1.0)
    assert np.allclose(b["masks"], 1.0 - successes.astype(np.float32))


def test_seeded_determinism():
    ds = GCDataset(dataset=_toy(), config=_CFG)
    idxs = np.array([1, 4, 7, 11, 15])
    np.random.seed(42); b1 = ds.sample(len(idxs), idxs=idxs)
    np.random.seed(42); b2 = ds.sample(len(idxs), idxs=idxs)
    assert np.array_equal(b1["value_goals"], b2["value_goals"])
    assert np.array_equal(b1["actor_goals"], b2["actor_goals"])


def test_trajgoal_stays_in_trajectory():
    # actor goals are pure trajgoal (p_trajgoal=1): must be within the same trajectory, >= idx.
    ds = GCDataset(dataset=_toy(), config=_CFG)
    np.random.seed(1)
    idxs = np.array([0, 2, 5])  # all in traj 0 (indices 0..9)
    gi = ds.sample_goals(idxs, 0.0, 1.0, 0.0, False)
    assert np.all(gi >= idxs) and np.all(gi <= 9)
