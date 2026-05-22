import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import (region_labels, cube_to_goal_dist,
                                   phase_spearman_table)
from evals.phase_probe import Thresholds


def test_region_labels_reach_grasp_transport():
    thr = Thresholds()
    grip = np.array([0.0, 0.9, 0.9])          # open, closed, closed
    lift = np.array([0.0, 0.0, 0.20])         # on table, on table, lifted
    out = region_labels(grip, lift, thr)
    assert list(out) == ["reach", "grasp", "transport"]


def test_cube_to_goal_dist():
    cube = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    d = cube_to_goal_dist(cube, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(d, [0.0, 5.0])


def test_phase_spearman_table_monotone():
    rows = []
    for ph in ("reach", "transport"):
        for i in range(8):
            d = 0.3 - 0.03 * i
            rows.append({"task": "task1", "region": ph, "d": d,
                         "V": 1.0 - d})
    df = pd.DataFrame(rows)
    tab = phase_spearman_table(df, value_col="V")
    r = tab.set_index("region")["rho_mean"]
    assert r.loc["transport"] > 0.99 and r.loc["reach"] > 0.99
