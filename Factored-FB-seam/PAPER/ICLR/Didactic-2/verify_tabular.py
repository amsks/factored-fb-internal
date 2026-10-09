#!/usr/bin/env python3
"""Tabular verification wrapper for the exact deterministic-path certificate."""
from pathlib import Path
import json, runpy

ROOT = Path(__file__).resolve().parents[3]
out = runpy.run_path(str(ROOT / "scripts/didactic/verify_deterministic_path_spherical.py"))["verify"]()
summary = {
    "objective": out["objective"],
    "candidate_square_objective": out["candidate_error_and_objective"],
    "candidate_start_gap": out["candidate_goal_gap"],
    "certified_global_start_gap_upper": out["every_global_minimizer_normalized_goal_gap_upper"],
    "normalized_control_witness_square_error": out["spherical_control_witness_error"],
    "normalized_control_witness_probabilities": out["spherical_control_witness_probabilities"],
    "capacity_and_normalization_control": {
        "dimension": 2,
        "row_norm": "sqrt(2)",
        "second_moment": "I",
        "correct_control_witness_exists": True,
    },
    "interpretation": "Every global spherical FB square-loss optimum has a negative start gap, although the same normalized rank-2 class contains a correct-control witness.",
}
# Existing full-expectation tabular TD check on the same path and gamma.  The
# spherical FB arm and density arm were run with the same six seed/lr settings;
# this is empirical TD evidence, separate from the exact-target certificate.
td_root = ROOT / "analysis/deterministic_didactic"
def compact(path):
    rows = json.loads(path.read_text())
    return [{k: r[k] for k in ("seed", "lr", "start_gap", "true_start_gap", "wrong_strict_states", "gradient_norm")} for r in rows]
summary["tabular_td_same_path"] = {
    "fb_spherical": compact(td_root / "path4_td_g08_sphere/fb_latest.json"),
    "density": compact(td_root / "path4_td_g08/density_latest.json"),
    "protocol": "full-expectation tabular TD; path4; gamma=.8; six seed/lr settings; 100k updates",
    "scope": "Empirical TD check, not the exact global-optimum theorem.",
}
(Path(__file__).resolve().parent / "tabular_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
