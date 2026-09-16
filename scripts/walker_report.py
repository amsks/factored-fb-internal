"""One table for the ExoRL walker block: every critic scored under the td_jepa protocol.

Reads the paper_eval jsonl files (one row per critic x seed x task, 10 episodes each) and
prints per-task means, the 4-task mean per seed, and mean +- population sd over seeds.
"""
import json, glob, collections, sys
import numpy as np

FILES = sys.argv[1:] or ["exp/paper_eval_walker_20260914.jsonl",
                         "exp/paper_eval_walker_feat.jsonl",
                         "exp/paper_eval_walker_dfb.jsonl",
                         "exp/paper_eval_tdj_rescore_20260914.jsonl"]
TASKS = ["stand", "walk", "run", "spin"]

rows = collections.defaultdict(dict)        # (critic, seed) -> task -> return
for f in FILES:
    try:
        lines = open(f).read().splitlines()
    except FileNotFoundError:
        print(f"(missing {f})"); continue
    for line in lines:
        r = json.loads(line)
        if r.get("domain") != "dmc-walker":
            continue
        rows[(r["critic"], r["seed"])][r.get("exorl_task")] = float(r["episode_return"])

crits = sorted({c for c, _ in rows})
print(f"{'critic':26s}" + "".join(f"{t:>8s}" for t in TASKS) + f"{'mean':>9s}  seeds")
for c in crits:
    seeds = sorted(s for cc, s in rows if cc == c)
    per_task = {t: [rows[(c, s)].get(t) for s in seeds] for t in TASKS}
    means = [np.mean([v for v in per_task[t] if v is not None]) if any(
        v is not None for v in per_task[t]) else np.nan for t in TASKS]
    per_seed = [np.mean([rows[(c, s)][t] for t in TASKS if t in rows[(c, s)]]) for s in seeds]
    line = f"{c:26s}" + "".join(f"{m:8.0f}" for m in means)
    line += f"{np.mean(per_seed):7.0f}±{np.std(per_seed):3.0f}  n={len(seeds)}"
    print(line)
