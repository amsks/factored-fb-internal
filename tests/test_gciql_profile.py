import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import gciql_profile as gp
from evals.phase_probe import Thresholds


def test_parse_flags_reads_env_and_seed(tmp_path):
    (tmp_path / "flags.json").write_text(json.dumps(
        {"env_name": "cube-single-play-v0", "seed": 10,
         "agent": "agents/gciql.py"}))
    info = gp.parse_flags(tmp_path)
    assert info["env_name"] == "cube-single-play-v0"
    assert info["seed"] == 10


def _ep(outcome, d, v, grip, cube_z, table_z=0.02):
    T = len(d)
    return {
        "obs": np.zeros((T, 4), np.float32),
        "d": np.asarray(d, np.float64),
        "V": np.asarray(v, np.float64),
        "grip": np.asarray(grip, np.float64),
        "cube": np.column_stack([np.zeros(T), np.zeros(T),
                                 np.asarray(cube_z)]),
        "table_z": table_z, "success": outcome == "success",
        "task": "task1", "episode": 0,
    }


def test_episodes_to_frames_t1_t4_funnel():
    # 6 transport steps, V strictly rises as d -> 0  => rho ~ +1
    d = [0.30, 0.25, 0.20, 0.15, 0.10, 0.02]
    v = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grip = [1.0] * 6
    cube_z = [0.20] * 6                    # lifted (table_z 0.02)
    eps = [_ep("success", d, v, grip, cube_z)]
    vl, cov, fun = gp.episodes_to_frames(eps, Thresholds(),
                                         ref_obs=np.zeros((10, 4)))
    assert set(vl.columns) >= {"task", "episode", "outcome",
                               "rho_V_negd", "n_transport_steps"}
    assert vl.iloc[0]["rho_V_negd"] > 0.99
    assert vl.iloc[0]["outcome"] == "success"
    assert set(cov.columns) >= {"task", "outcome", "region", "nn_dist"}
    assert set(fun.columns) >= {"task", "episode", "furthest_phase",
                                "outcome"}
