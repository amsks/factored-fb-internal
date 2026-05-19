import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import gciql_profile_aggregate as ga


def test_phase_action_fields_basic():
    # one episode: cube moves +x by 0.1 each step in 'transport'
    df = pd.DataFrame({
        "episode": [0, 0, 0],
        "t": [0, 1, 2],
        "cube_x": [0.0, 0.1, 0.2],
        "cube_y": [0.0, 0.0, 0.0],
        "Vz": [-1.0, 0.0, 1.0],
    })
    grid = (np.linspace(-0.1, 0.3, 9), np.linspace(-0.2, 0.2, 9))
    vz, uv, counts = ga._phase_action_fields(
        df, grid, n_arrow=4, n_min=1)
    assert vz.shape == (8, 8)            # len(edges)-1 cells per axis
    assert np.nanmax(vz) > np.nanmin(vz)
    # net displacement is +x, ~0 y
    us, vs = uv
    assert np.nanmean(us[np.isfinite(us)]) > 0
    assert abs(np.nanmean(vs[np.isfinite(vs)])) < 1e-6
    assert counts.sum() >= 1


def test_phase_action_fields_empty():
    df = pd.DataFrame(columns=["episode", "t", "cube_x", "cube_y", "Vz"])
    grid = (np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    vz, uv, counts = ga._phase_action_fields(
        df, grid, n_arrow=4, n_min=1)
    assert np.isnan(vz).all()
    assert counts.sum() == 0
