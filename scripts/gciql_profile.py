"""scripts/gciql_profile.py — per-checkpoint GCIQL profiler.

Runs ONLY in the isolated JAX venv (it imports jax/flax/vendored
OGBench inside main()/_jax_*). The pure seams below (flags parsing,
episode-records -> frames) are JAX-free and unit-tested.

T1 (Spearman rho(V(s,g), -d)), T4 (off-support coverage) and the phase
funnel are computed with evals._profile_core / evals.phase_probe so the
FB and GCIQL numbers are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals._profile_core import _spearman, probe_coverage, transport_mask
from evals.phase_probe import Thresholds, classify_phases


def parse_flags(run_dir) -> Dict[str, Any]:
    """Read OGBench flags.json from a GCIQL seed dir."""
    data = json.loads((Path(run_dir) / "flags.json").read_text())
    return {"env_name": data.get("env_name", "cube-single-play-v0"),
            "seed": int(data.get("seed", 0)),
            "agent": data.get("agent", "agents/gciql.py")}


def episodes_to_frames(episodes: List[Dict[str, Any]], thr: Thresholds,
                       ref_obs: np.ndarray):
    """Episode records -> (value_landscape, coverage, funnel) frames.

    Each record: obs[T,O], d[T], V[T], grip[T], cube[T,3], table_z,
    success(bool), task(str), episode(int). Pure; no jax.
    """
    vl_rows, fun_rows, cov_eps = [], [], []
    for ep in episodes:
        d = np.asarray(ep["d"], np.float64)
        V = np.asarray(ep["V"], np.float64)
        tm = transport_mask(ep, thr)
        sig = {
            "eff": np.zeros((len(d), 3)),
            "cube": np.asarray(ep["cube"], np.float64).reshape(-1, 3),
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "goal": np.asarray(ep["cube"], np.float64).reshape(-1, 3)[-1],
            "success": bool(ep["success"]),
            "table_z": float(ep["table_z"]),
            "length": len(d),
        }
        cls = classify_phases(sig, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            outcome = "transport_fail"
        else:
            outcome = "other"
        n_t = int(tm.sum())
        if n_t >= 5 and np.ptp(d[tm]) > 1e-9 and np.ptp(V[tm]) > 1e-9:
            rho = _spearman(V[tm], -d[tm])
            v_secure, v_end = float(V[tm][0]), float(V[-1])
        else:
            rho, v_secure, v_end = float("nan"), float("nan"), float("nan")
        vl_rows.append({"task": ep["task"], "episode": ep["episode"],
                        "outcome": outcome, "rho_V_negd": rho,
                        "V_at_secure": v_secure, "V_at_end": v_end,
                        "n_transport_steps": n_t})
        fun_rows.append({"task": ep["task"], "episode": ep["episode"],
                         "furthest_phase": cls["fail_phase"]
                         if not cls["success"] else "success",
                         "outcome": outcome})
        cov_eps.append({"obs": ep["obs"], "outcome": outcome,
                        "grip": ep["grip"], "cube": ep["cube"],
                        "table_z": ep["table_z"],
                        "transport_mask": tm})
    cov = probe_coverage(cov_eps, ref_obs, thr)
    cov.insert(0, "task", episodes[0]["task"] if episodes else "")
    return (pd.DataFrame(vl_rows), cov, pd.DataFrame(fun_rows))
