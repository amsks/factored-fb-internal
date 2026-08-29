"""Random-policy return floors for the ExoRL domains (spec sec.6 item 3).

Every published number in this campaign is a RAW episode return, and raw returns have no
natural zero -- dm_control tasks differ in how much reward an untrained flailing body
collects by accident. Without the floor, "cheetah run = 16.6" and "walker stand = 226.0"
are not comparable, and neither can be read as "near zero" without checking. This measures
what a uniform-random policy scores, so every table cell can be read against its own floor.

Actions are drawn uniformly from [-1, 1]^dim, which is the action range the ExoRL wrapper
clips to. Same episode length (1000) and the same seeding discipline as evaluation, so the
floor is measured under the eval harness rather than an approximation of it.

    python scripts/exorl_random_floors.py [--episodes 10] [--out floors.json]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from env.exorl_env import make_exorl_env, TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--slugs", nargs="*", default=["dmc-walker", "dmc-cheetah"])
ap.add_argument("--out", default="")
a = ap.parse_args()

rows = {}
for slug in a.slugs:
    for task in TASKS[slug][2]:
        env, horizon = make_exorl_env(slug, reward_task=task, seed=0)
        adim = env.env.action_spec().shape[0]
        rng = np.random.default_rng(0)
        rets = []
        for ep in range(a.episodes):
            env.reset(seed=ep)
            tot = 0.0
            for _ in range(horizon):
                _, r, _, trunc, _ = env.step(rng.uniform(-1.0, 1.0, adim))
                tot += r
                if trunc:
                    break
            rets.append(tot)
        env.close()
        m, s = float(np.mean(rets)), float(np.std(rets))
        rows[f"{slug}/{task}"] = {"mean": m, "std": s, "episodes": a.episodes}
        print(f"{slug+'/'+task:<28} random floor = {m:8.2f} +- {s:6.2f}", flush=True)

if a.out:
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {a.out}")
