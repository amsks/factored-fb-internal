"""scripts/collect_maniskill.py — collect a ManiSkill task into the unified format.

Runs in a mani_skill-capable venv (e.g. /raid/tongle/rl/newt/.venv) and uses newt's
make_env via collectors.maniskill_backend. It must NOT import Factored-FB's ``envs``
package (newt's backend registers its own top-level ``envs``), so it wires only
collectors.* + data.unified. FB then trains offline on the output (train.py
build_env=false).

    /raid/tongle/rl/newt/.venv/bin/python scripts/collect_maniskill.py \
        --task ms-pick-cube --n-episodes 50 --out /raid/tongle/rl/fb-data/testbed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root first so scripts/data/ doesn't shadow the repo-root data/ package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collectors.base import RandomPolicy, collect_dataset  # noqa: E402
from collectors.maniskill_backend import ManiSkillEnv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="ms-<name>, e.g. ms-pick-cube")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--out", default="/raid/tongle/rl/fb-data/testbed")
    p.add_argument("--render-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    env = ManiSkillEnv(a.task, seed=a.seed, render_size=a.render_size)
    root = collect_dataset(
        env,
        RandomPolicy(env.action_space),
        name=f"ManiSkill/{a.task}",
        data_path=a.out,
        n_episodes=a.n_episodes,
        max_steps=env.max_episode_steps,
        seed=a.seed,
        meta_extra={"env_family": "maniskill", "task": a.task},
    )
    print(f"[collect] {a.n_episodes} episodes of ManiSkill/{a.task} -> {root}")


if __name__ == "__main__":
    main()
