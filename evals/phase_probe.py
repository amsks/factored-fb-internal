"""evals/phase_probe.py — test where FB fails in a cube episode.

M1: classify normal rollouts into reach/secure/success phases.
M2: counterfactual start states (S0 baseline / S1 skip-reach /
S2 pre-grasped) to isolate the grasp bottleneck.

See docs/superpowers/specs/2026-05-18-cube-phase-failure-probe-design.md.
Pure logic here is unit-tested; MuJoCo physics is validated by a
documented Linux acceptance run. No training/eval code is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PHASE_ORDER = ["none", "reached", "secured", "success"]
N_SETTLE = 5  # MuJoCo steps to clamp the cube in S2 (module constant; YAGNI)


@dataclass
class Thresholds:
    eps_reach: float = 0.06   # m, hand-to-cube distance for "reached"
    delta_lift: float = 0.03  # m, cube height above table for "lifted"
    k_steps: int = 5          # consecutive steps the lift must hold
    tau_grip: float = 0.5     # gripper_opening > tau => "closed"


def _max_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def classify_phases(signals: Dict[str, Any], thr: Thresholds) -> Dict[str, Any]:
    """Classify ONE episode's per-step signals into phase milestones.

    signals: eff[T,3], cube[T,3], grip[T], goal[3], success(bool),
             table_z(float).
    Returns the milestones, monotone furthest_phase/fail_phase, and
    diagnostics. Handles T == 0.
    """
    eff = np.asarray(signals["eff"], dtype=np.float64).reshape(-1, 3)
    cube = np.asarray(signals["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(signals["grip"], dtype=np.float64).reshape(-1)
    goal = np.asarray(signals["goal"], dtype=np.float64).reshape(3)
    table_z = float(signals["table_z"])
    success = bool(signals["success"])
    T = eff.shape[0]

    if T == 0:
        return {
            "reached": success, "secured": success, "success": success,
            "reached_raw": False, "secured_raw": False,
            "furthest_phase": "success" if success else "none",
            "fail_phase": "none" if success else "reach",
            "min_eff_cube_dist": float("nan"),
            "max_cube_lift": float("nan"),
            "final_cube_goal_dist": float("nan"),
            "length": 0,
        }

    eff_cube_dist = np.linalg.norm(eff - cube, axis=1)
    eff_cube_xy = np.linalg.norm(eff[:, :2] - cube[:, :2], axis=1)
    lift = cube[:, 2] - table_z

    reached_raw = bool(eff_cube_dist.min() < thr.eps_reach)
    secured_mask = (
        (lift > thr.delta_lift)
        & (grip > thr.tau_grip)
        & (eff_cube_xy < thr.eps_reach)
    )
    secured_raw = bool(_max_true_run(secured_mask) >= thr.k_steps)

    # Monotone ladder: success => secured => reached.
    secured = secured_raw or success
    reached = reached_raw or secured

    if success:
        furthest, fail = "success", "none"
    elif secured:
        furthest, fail = "secured", "transport"
    elif reached:
        furthest, fail = "reached", "grasp"
    else:
        furthest, fail = "none", "reach"

    return {
        "reached": reached, "secured": secured, "success": success,
        "reached_raw": reached_raw, "secured_raw": secured_raw,
        "furthest_phase": furthest, "fail_phase": fail,
        "min_eff_cube_dist": float(eff_cube_dist.min()),
        "max_cube_lift": float(lift.max()),
        "final_cube_goal_dist": float(np.linalg.norm(cube[-1] - goal)),
        "length": int(T),
    }
