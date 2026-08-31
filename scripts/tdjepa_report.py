"""Aggregate td_jepa's own reward.csv files into the same table shape as exorl_paper_eval.

td_jepa writes one row per eval EPISODE (their launcher asks for 10) per task per eval
point, with `timestep` naming the training step. So a cell is: average the episodes
within a seed first, then take mean/std across seeds. Averaging all rows together
instead would weight a seed by how many episodes it happened to log and would report
the episode spread as if it were the seed spread -- the same row-vs-seed confusion that
exorl_paper_eval.report() had.

    python scripts/tdjepa_report.py ~/tdj_runs 'rldp_walker_s*'
"""
import csv, glob, os, statistics, sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tdj_runs")
pattern = sys.argv[2] if len(sys.argv) > 2 else "*_walker_s*"

runs = sorted(glob.glob(os.path.join(root, pattern)))
if not runs:
    sys.exit(f"no run dirs matching {pattern} under {root}")

per_seed = defaultdict(dict)          # task -> {run: mean over its episodes}
steps = {}
for run in runs:
    path = os.path.join(run, "reward.csv")
    if not os.path.isfile(path):
        print(f"  (no reward.csv yet in {os.path.basename(run)})")
        continue
    rows = list(csv.DictReader(open(path)))
    if not rows:
        continue
    last = max(int(r["timestep"]) for r in rows)
    steps[run] = last
    by_task = defaultdict(list)
    for r in rows:
        if int(r["timestep"]) == last:
            by_task[r["task"]].append(float(r["reward"]))
    for task, eps in by_task.items():
        per_seed[task][run] = statistics.mean(eps)   # episodes -> one number per seed

if steps:
    lo, hi = min(steps.values()), max(steps.values())
    print(f"latest eval step: {lo}" + (f" - {hi}  (SEEDS NOT AT THE SAME STEP)" if lo != hi else ""))
print(f"{'task':<16} {'mean':>9} {'std':>8}  n   (episodes averaged within seed first)")
tot = []
for task in sorted(per_seed):
    vals = list(per_seed[task].values())
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{task:<16} {statistics.mean(vals):9.1f} {sd:8.1f}  {len(vals)}")
    tot.append(statistics.mean(vals))
if tot:
    print(f"{'AGGREGATE':<16} {statistics.mean(tot):9.1f}")
