import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas.prefix_replay import PrefixState, branch, make_env, replay_to


def test_prefix_replay_is_bit_exact():
    env = make_env(task_id=1)
    rng = np.random.RandomState(4)
    prefix = np.stack([rng.uniform(-1, 1, 5).astype(np.float32) for _ in range(25)])
    cont = [rng.uniform(-1, 1, 5).astype(np.float32) for _ in range(30)]
    ps = PrefixState(seed=0, task_id=1, prefix=prefix)
    first = rng.uniform(-1, 1, 5).astype(np.float32)
    o1 = branch(env, ps, first, cont)
    o2 = branch(env, ps, first, cont)
    div = max(float(np.abs(a - b).max()) for a, b in zip(o1, o2))
    assert div == 0.0, f"prefix-replay not bit-exact: {div}"


def test_replay_to_returns_current_obs():
    env = make_env(task_id=1)
    rng = np.random.RandomState(1)
    prefix = np.stack([rng.uniform(-1, 1, 5).astype(np.float32) for _ in range(10)])
    ps = PrefixState(seed=0, task_id=1, prefix=prefix)
    ob = replay_to(env, ps)
    assert ob.shape == (28,)
