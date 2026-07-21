"""campaign/wandb_agg_daemon.py — aggregate the per-seed inline-eval jsonls into one
'overall result' wandb run per group.

Each training run appends {seed,step,success_rate,family} to
    {agg_dir}/{algo}__{task}__s{seed}.jsonl
This daemon groups the seed files by {algo}__{task}, and once all n_seeds have reported a
given step, logs a run named "{algo}__{task}__agg" (id "agg-...") with
    eval/success_mean, eval/success_std, eval/n_seeds
at that step. Because the metric keys differ from the per-seed runs' eval/success_rate,
you can plot the mean/variance curve on its own -> the aggregate the seed names hid.

    <newt>/.venv/bin/python campaign/wandb_agg_daemon.py \
        --agg_dir /raid/tongle/rl/fb-data/campaign/dev_jax_eval \
        --project factored-fb-dev --n_seeds 3 --interval 60
"""
import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def groups_in(agg_dir):
    gs = set()
    for f in glob.glob(os.path.join(agg_dir, "*__s*.jsonl")):
        gs.add(re.sub(r"__s\d+\.jsonl$", "", os.path.basename(f)))
    return sorted(gs)


def read_group(agg_dir, group):
    """-> ({step: [success,...]}, {step: [at_goal_frac,...]}, family)."""
    per_step = defaultdict(list)
    per_dwell = defaultdict(list)
    fam = None
    for f in glob.glob(os.path.join(agg_dir, f"{group}__s*.jsonl")):
        try:
            lines = Path(f).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            per_step[int(d["step"])].append(float(d["success_rate"]))
            per_dwell[int(d["step"])].append(float(d.get("at_goal_frac", float("nan"))))
            fam = d.get("family", fam)
    return per_step, per_dwell, fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_dir", required=True)
    ap.add_argument("--project", default="factored-fb-dev")
    ap.add_argument("--n_seeds", type=int, default=3, help="log a step once this many seeds report it")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb

    Path(args.agg_dir).mkdir(parents=True, exist_ok=True)
    state_fp = Path(args.agg_dir) / ".agg_state.json"
    logged = defaultdict(set)   # group -> {steps already logged}
    if state_fp.exists():       # resume cleanly across restarts (no out-of-order re-logging)
        for g, steps in json.loads(state_fp.read_text()).items():
            logged[g] = set(steps)
        print(f"[agg] resumed state for {len(logged)} groups", flush=True)

    def save_state():
        state_fp.write_text(json.dumps({g: sorted(s) for g, s in logged.items()}))

    print(f"[agg] watching {args.agg_dir} (n_seeds={args.n_seeds}, every {args.interval}s)", flush=True)
    while True:
        for group in groups_in(args.agg_dir):
            per_step, per_dwell, fam = read_group(args.agg_dir, group)
            ready = sorted(s for s, v in per_step.items()
                           if len(v) >= args.n_seeds and s not in logged[group])
            if not ready:
                continue
            run_id = "agg-" + re.sub(r"[^A-Za-z0-9_-]", "-", group)
            run = wandb.init(project=args.project, id=run_id, name=f"{group}__agg",
                             group=group, tags=[fam] if fam else None,
                             resume="allow", reinit=True,
                             config=dict(kind="aggregate", group=group, family=fam,
                                         n_seeds=args.n_seeds))
            for s in ready:
                v = np.asarray(per_step[s], dtype=np.float64)
                d = np.asarray(per_dwell[s], dtype=np.float64)
                wandb.log({"eval/success_mean": float(v.mean()),
                           "eval/success_std": float(v.std()),
                           "eval/at_goal_mean": float(np.nanmean(d)),
                           "eval/at_goal_std": float(np.nanstd(d)),
                           "eval/n_seeds": int(len(v))}, step=s)
                logged[group].add(s)
            print(f"[agg] {group}: logged steps {ready} "
                  f"(succ@{ready[-1]}={np.mean(per_step[ready[-1]]):.3f} "
                  f"dwell={np.nanmean(per_dwell[ready[-1]]):.3f})", flush=True)
            run.finish()
            save_state()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
