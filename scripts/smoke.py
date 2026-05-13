"""
scripts/smoke.py — Synthetic-data pipeline smoke (no env / no dataset).

Builds FBAgent and FBFlowBCAgent from the Hydra configs and runs a few update
steps on random Box observations / actions.

Usage:
    python scripts/smoke.py                          # all OGBench domains
    python scripts/smoke.py --benchmark hecrl        # HECRL domains
    python scripts/smoke.py --benchmark all          # both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from gymnasium.spaces import Box
from hydra import compose, initialize_config_dir

from train import make_agent


# Per-benchmark smoke sets: (domain_config_name, expected_agent_type, obs_dim, action_dim)
SMOKE_SETS = {
    "ogbench": [
        ("antmaze_medium", "FBAgent",       15, 4),
        ("antmaze_large",  "FBAgent",       15, 4),
        ("cube_single",    "FBFlowBCAgent", 15, 4),
        ("scene",          "FBFlowBCAgent", 15, 4),
        ("puzzle_3x3",     "FBFlowBCAgent", 15, 4),
    ],
    "hecrl": [
        ("ppp_cube",       "FBFlowBCAgent", 21, 5),     # 6+5*3 obs (num_cubes=3), 5-D action
    ],
}


def smoke(domain: str, expected_type: str, obs_dim: int, action_dim: int,
          *, batch_size: int = 32, n_steps: int = 5) -> None:
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"domain={domain}", "device=cpu", f"batch_size={batch_size}"],
        )

    obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=action_dim)
    assert type(agent).__name__ == expected_type, (type(agent).__name__, expected_type)

    last = None
    for step in range(n_steps):
        batch = {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, action_dim).clamp(-1, 1),
            "next": {
                "observation": torch.randn(batch_size, obs_dim),
                "terminated": torch.zeros(batch_size, 1),
            },
        }
        last = agent.update(batch, step)

    assert all(np.isfinite(v) for v in last.values()), last
    print(
        f"OK {cfg.benchmark:8s} {domain:18s} {expected_type:14s} "
        f"loss/fb={last['loss/fb']:.2f} "
        f"loss/actor={last['loss/actor']:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="ogbench",
                    choices=["ogbench", "hecrl", "all"],
                    help="Which benchmark's domain configs to smoke (default: ogbench).")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    benches = list(SMOKE_SETS) if args.benchmark == "all" else [args.benchmark]
    for bench in benches:
        for entry in SMOKE_SETS[bench]:
            smoke(*entry)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
