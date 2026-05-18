import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import phase_probe as pp


def _sig(eff, cube, grip, success, goal=(1.0, 1.0, 0.0), table_z=0.02):
    return {
        "eff": np.asarray(eff, dtype=np.float32),
        "cube": np.asarray(cube, dtype=np.float32),
        "grip": np.asarray(grip, dtype=np.float32),
        "goal": np.asarray(goal, dtype=np.float32),
        "success": bool(success),
        "table_z": float(table_z),
    }


def test_full_success_episode():
    T = 8
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.ones(T)            # closed
    cube[:, 2] = 0.10            # lifted well above table_z=0.02
    out = pp.classify_phases(_sig(eff, cube, grip, success=True), pp.Thresholds())
    assert out["reached"] and out["secured"] and out["success"]
    assert out["furthest_phase"] == "success"
    assert out["fail_phase"] == "none"
    assert out["length"] == T


def test_reach_but_no_secure():
    T = 6
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.zeros(T)           # never closes
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["reached"] and not out["secured"] and not out["success"]
    assert out["furthest_phase"] == "reached"
    assert out["fail_phase"] == "grasp"


def test_never_reach():
    T = 5
    eff = np.zeros((T, 3)); cube = np.full((T, 3), 9.0)
    grip = np.zeros(T)
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert not out["reached"]
    assert out["furthest_phase"] == "none"
    assert out["fail_phase"] == "reach"


def test_secure_but_no_success():
    T = 7
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.ones(T); cube[:, 2] = 0.10
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] and not out["success"]
    assert out["furthest_phase"] == "secured"
    assert out["fail_phase"] == "transport"


def test_monotone_forcing_success_implies_lower():
    T = 4
    eff = np.full((T, 3), 9.0); cube = np.zeros((T, 3))   # raw reach False
    grip = np.zeros(T)                                    # raw secure False
    out = pp.classify_phases(_sig(eff, cube, grip, success=True), pp.Thresholds())
    assert out["reached_raw"] is False and out["secured_raw"] is False
    assert out["reached"] and out["secured"] and out["success"]
    assert out["furthest_phase"] == "success"


def test_k_steps_window_required():
    T = 10
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3)); grip = np.ones(T)
    cube[:, 2] = 0.02            # at table; not lifted
    cube[3:5, 2] = 0.10          # lifted only 2 steps (< k=5)
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] is False


def test_zero_length_episode():
    z = np.zeros((0, 3))
    out = pp.classify_phases(_sig(z, z, np.zeros(0), success=False),
                             pp.Thresholds())
    assert out["length"] == 0
    assert out["furthest_phase"] == "none"
    assert out["fail_phase"] == "reach"
    assert np.isnan(out["min_eff_cube_dist"])
