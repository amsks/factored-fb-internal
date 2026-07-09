#!/usr/bin/env python3
"""Audit IWS representation / sampling parity between Factored-FB and UM runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    "algorithm_fb": "iws",
    "algorithm_um": "interaction_weighted_sampling",
    "hidden_dims": [512, 512],
    "repr_dim": 256,
    "critic_score_type": "cosine",
    "critic_type": "cpc",
    "context_repetition_factor": 8,
    "iws_full_episode_future": True,
    "contact_distance": 0.12,
    "contact_sigma": 30.0,
    "near_contact_distance": 0.24,
    "random_goals": 0.5,
    "logsumexp_coeff": 0.01,
    "batch_size": 64,
    "updates_per_iter": 16,
    "future_discount": 0.99,
}


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _um_algo(cfg: dict) -> dict:
    algo = cfg.get("algorithm", {})
    return algo if isinstance(algo, dict) else {}


def _check_fb_seed(run_dir: Path) -> list[str]:
    issues: list[str] = []
    cfg = _load_cfg(run_dir / "config.json")
    if cfg.get("algorithm") != EXPECTED["algorithm_fb"]:
        issues.append(f"algorithm={cfg.get('algorithm')}")
    if cfg.get("hidden_dims") != EXPECTED["hidden_dims"]:
        issues.append(f"hidden_dims={cfg.get('hidden_dims')}")
    if cfg.get("repr_dim") != EXPECTED["repr_dim"]:
        issues.append(f"repr_dim={cfg.get('repr_dim')}")
    if cfg.get("critic_score_type") != EXPECTED["critic_score_type"]:
        issues.append(f"critic_score_type={cfg.get('critic_score_type')}")
    if cfg.get("context_repetition_factor") != EXPECTED["context_repetition_factor"]:
        issues.append(f"context_repetition_factor={cfg.get('context_repetition_factor')}")
    if cfg.get("iws_full_episode_future") != EXPECTED["iws_full_episode_future"]:
        issues.append(f"iws_full_episode_future={cfg.get('iws_full_episode_future')}")
    for key in ("contact_distance", "contact_sigma", "near_contact_distance"):
        if cfg.get(key) != EXPECTED[key]:
            issues.append(f"{key}={cfg.get(key)}")
    if cfg.get("use_iws") is not True:
        issues.append(f"use_iws={cfg.get('use_iws')}")
    return issues


def _check_um_seed(run_dir: Path) -> list[str]:
    issues: list[str] = []
    cfg = _load_cfg(run_dir / "config.json")
    algo = _um_algo(cfg)
    if algo.get("name") != EXPECTED["algorithm_um"]:
        issues.append(f"algorithm.name={algo.get('name')}")
    if algo.get("hidden_dims") != EXPECTED["hidden_dims"]:
        issues.append(f"hidden_dims={algo.get('hidden_dims')}")
    if algo.get("repr_dim") != EXPECTED["repr_dim"]:
        issues.append(f"repr_dim={algo.get('repr_dim')}")
    if algo.get("critic_score_type") != EXPECTED["critic_score_type"]:
        issues.append(f"critic_score_type={algo.get('critic_score_type')}")
    if algo.get("critic_type") != EXPECTED["critic_type"]:
        issues.append(f"critic_type={algo.get('critic_type')}")
    if algo.get("iws_context_repetition_factor") != EXPECTED["context_repetition_factor"]:
        issues.append(f"iws_context_repetition_factor={algo.get('iws_context_repetition_factor')}")
    if algo.get("iws_full_episode_future") is not True:
        issues.append(f"iws_full_episode_future={algo.get('iws_full_episode_future')}")
    pairs = {
        "iws_contact_distance": "contact_distance",
        "iws_contact_sigma": "contact_sigma",
        "iws_near_contact_distance": "near_contact_distance",
    }
    for um_key, exp_key in pairs.items():
        if algo.get(um_key) != EXPECTED[exp_key]:
            issues.append(f"{um_key}={algo.get(um_key)}")
    return issues


def _iws_events(run_dir: Path) -> float | None:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "replay/iws_events" in row:
            vals.append(float(row["replay/iws_events"]))
    return vals[-1] if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fb-root", type=Path, default=Path("runs/air_hockey"))
    ap.add_argument("--um-root", type=Path, default=Path("../Unsupervised_Manipulation/runs/air_hockey"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 42, 43])
    args = ap.parse_args()

    print("=== IWS representation / config parity ===")
    print(json.dumps(EXPECTED, indent=2))
    print()

    for seed in args.seeds:
        fb_dir = args.fb_root / f"iws_seed{seed}"
        um_dir = args.um_root / f"iws_seed{seed}"
        fb_issues = _check_fb_seed(fb_dir)
        um_issues = _check_um_seed(um_dir)
        print(f"seed {seed}:")
        print(f"  FB config: {'OK' if not fb_issues else 'MISMATCH: ' + ', '.join(fb_issues)}")
        print(f"  UM config: {'OK' if not um_issues else 'MISMATCH: ' + ', '.join(um_issues)}")
        fb_iws = _iws_events(fb_dir)
        if fb_iws is not None:
            flag = "OK" if fb_iws > 100 else "LOW (IWS events may not be caching)"
            print(f"  FB replay/iws_events: {fb_iws:.0f} [{flag}]")

    compare_script = Path(__file__).with_name("compare_air_hockey_multiseed.py")
    print("\n=== Timestep-synchronized eval comparison ===")
    subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--campaign",
            "iws",
            "--fb-root",
            str(args.fb_root),
            "--um-root",
            str(args.um_root),
            *("--seeds", *[str(s) for s in args.seeds]),
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
