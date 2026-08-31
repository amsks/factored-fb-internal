"""Aggregate td_jepa's own reward.csv files into the same table shape as exorl_paper_eval.

td_jepa writes one row per eval EPISODE (their launcher asks for 10) per task per eval
point, with `timestep` naming the training step. So a cell is: average the episodes
within a seed first, then take mean/std across seeds. Averaging all rows together
instead would weight a seed by how many episodes it happened to log and would report
the episode spread as if it were the seed spread -- the same row-vs-seed confusion that
exorl_paper_eval.report() had.

    python scripts/tdjepa_report.py ~/tdj_runs 'rldp_walker_s*'
"""
import csv, glob, os, re, statistics, sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/tdj_runs")
pattern = sys.argv[2] if len(sys.argv) > 2 else "*_walker_s*"

runs = sorted(glob.glob(os.path.join(root, pattern)))
if not runs:
    sys.exit(f"no run dirs matching {pattern} under {root}")

per_seed = defaultdict(dict)          # task -> {SEED: mean over its episodes}
steps = {}
seed_owner = {}                       # seed -> (run, step), to catch two dirs per seed


def _seed_of(run):
    """The seed a run dir is for. Keyed on this rather than on the dir path because n is
    reported as a seed count: globbing '*_walker_s*' would count a re-run directory
    (..._s0_re, or a relaunch under a new path) as an extra seed, inflating n and
    shrinking the std toward zero. Same defect as counting jsonl rows as seeds -- n is
    the length of whatever got collected, and nothing checks the elements are seeds."""
    m = re.search(r"_s(?:eed)?[_-]?(\d+)(?:\D|$)", os.path.basename(run))
    if not m:
        sys.exit(f"cannot parse a seed from {run!r}; refusing to report an n that would "
                 f"count directories instead of seeds")
    return int(m.group(1))
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
    seed = _seed_of(run)
    prev = seed_owner.get(seed)
    if prev is not None:
        # Two dirs for one seed: keep the further-trained one and SAY so, rather than
        # letting both through as two seeds or silently picking whichever globbed first.
        keep_new = last > prev[1]
        print(f"NOTE: seed {seed} has two run dirs; keeping "
              f"{os.path.basename(run if keep_new else prev[0])} @ "
              f"{max(last, prev[1])} and dropping "
              f"{os.path.basename(prev[0] if keep_new else run)} @ {min(last, prev[1])}")
        if not keep_new:
            continue
    seed_owner[seed] = (run, last)
    for task, eps in by_task.items():
        per_seed[task][seed] = statistics.mean(eps)  # episodes -> one number per seed

if steps:
    lo, hi = min(steps.values()), max(steps.values())
    print(f"latest eval step: {lo}" + (f" - {hi}  (SEEDS NOT AT THE SAME STEP)" if lo != hi else ""))
print(f"{'task':<16} {'mean':>9} {'std':>8}  n   (episodes averaged within seed first)")
tot = []
for task in sorted(per_seed):
    vals = [per_seed[task][k] for k in sorted(per_seed[task])]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{task:<16} {statistics.mean(vals):9.1f} {sd:8.1f}  {len(vals)}")
    tot.append(statistics.mean(vals))
if tot:
    print(f"{'AGGREGATE':<16} {statistics.mean(tot):9.1f}")
