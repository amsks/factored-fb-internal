import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.atlas.gates import dataset_state_restore_drift
from scripts.atlas.prefix_replay import make_env


def test_residual_report_shape():
    env = make_env(task_id=1)
    T = 60
    ds = dict(qpos=np.random.randn(T, 21).astype(np.float32),
              qvel=np.random.randn(T, 20).astype(np.float32),
              actions=np.random.uniform(-1, 1, (T, 5)).astype(np.float32),
              observations=np.random.randn(T, 28).astype(np.float32))
    r = dataset_state_restore_drift(env, ds, idx=10, horizon=20)
    assert set(r) >= {'cube_xyz_drift_max', 'full_state_drift_max', 'threshold_m', 'passes'}
    assert r['threshold_m'] == 0.04
    assert isinstance(r['passes'], bool)
    # random synthetic qpos/qvel are not physical -> restore cannot match recorded rows
    assert r['cube_xyz_drift_max'] > 0.0
