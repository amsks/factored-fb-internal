import numpy as np
from impls.utils.datasets import FBDataset, GCDataset, make_dataset


def _base(n=200):
    obs = np.random.RandomState(0).randn(n, 8).astype(np.float32)
    act = np.random.RandomState(1).randn(n, 5).astype(np.float32)
    term = np.zeros((n,), np.float32)
    term[49::50] = 1.0          # 4 trajectories of length 50
    return FBDataset.create(observations=obs, actions=act, terminals=term)


def test_gcdataset_config_yields_goal_batches():
    cfg = dict(dataset_class="GCDataset", discount=0.99, gc_negative=True,
               value_p_curgoal=0.2, value_p_trajgoal=0.5, value_p_randomgoal=0.3,
               value_geom_sample=True, actor_p_curgoal=0.0, actor_p_trajgoal=1.0,
               actor_p_randomgoal=0.0, actor_geom_sample=False)
    ds = make_dataset(_base(), cfg)
    assert isinstance(ds, GCDataset)
    b = ds.sample(16)
    for k in ("observations", "actions", "next_observations", "value_goals",
              "actor_goals", "rewards", "masks"):
        assert k in b, f"missing {k}"
    assert b["value_goals"].shape == (16, 8)


def test_fb_config_passes_dataset_through():
    base = _base()
    ds = make_dataset(base, dict(z_dim=50))     # no dataset_class key
    assert ds is base
    b = ds.sample(16)
    assert "value_goals" not in b
