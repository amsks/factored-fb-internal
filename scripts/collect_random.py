"""scripts/collect_random.py — collect a random-policy dataset for a testbed task.

Pipeline sanity / bootstrap collector: rolls a uniform-random policy in a testbed
env (via envs.registry) and writes unified-format episodes (data/unified.py) that
FB training then consumes. Run in .venv-rlmini (rlmini/Metaworld/ManiSkill family).

    .venv-rlmini/bin/python scripts/collect_random.py \
        --task rlmini:Box2D/goal-ctrl-target \
        --n-episodes 100 --max-steps 100 \
        --out /raid/tongle/rl/fb-data/testbed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root must precede scripts/ on sys.path, else scripts/data/ shadows the
# repo-root data/ package (data.unified).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collectors.base import RandomPolicy, collect_dataset  # noqa: E402
from envs.registry import make_env, parse_domain  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="e.g. rlmini:Box2D/goal-ctrl-target")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--out", default="/raid/tongle/rl/fb-data/testbed")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    family, name = parse_domain(a.task)
    env, _ = make_env(a.task, seed=a.seed, horizon=a.max_steps)
    root = collect_dataset(
        env,
        RandomPolicy(env.action_space),
        name=name,
        data_path=a.out,
        n_episodes=a.n_episodes,
        max_steps=a.max_steps,
        seed=a.seed,
    )
    print(f"[collect] {a.n_episodes} episodes of {a.task} -> {root}")


if __name__ == "__main__":
    main()
