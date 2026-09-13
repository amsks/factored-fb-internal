#!/usr/bin/env python
"""Hand a finished tdjtab run back to the hub: status/<run-dir>.json + exp/<run-dir>/ files.

The handback contract from the campaign (mailbox 2026-09-08T01-00-48Z): when a run reaches 1M,
write status/<run-dir>.json and upload params_1000000.pkl, config.json, eval.csv, eval_relabel.csv.
Schema matches what the cluster already wrote for the tier-1 runs.

    python scripts/tdjtab_handback.py --run exp/tdjtab/cube-double-play-v0_rldp_flowbc_seed_0
    python scripts/tdjtab_handback.py --all-finished --exp-root exp/tdjtab   # every 1M run there
    python scripts/tdjtab_handback.py --run <dir> --dry-run                  # print, upload nothing

route_a / route_b are the mean success over the env's tasks AT THE FINAL STEP, matching the existing
status/ files. NOTE this is not what the paper's table computes: make_results_table.py:33 scores a run
as the mean over its LAST 3 eval points. The status file is a handback receipt, not the table.
"""
import argparse, csv, json, os, sys
from pathlib import Path

REPO = "amsks/factored-fb-checkpoints"
WANT = ("params_1000000.pkl", "config.json", "eval.csv", "eval_relabel.csv")


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _by_step(rows, col="success"):
    per = {}
    for r in rows:
        if not r.get("step") or r.get(col) in (None, ""):
            continue
        per.setdefault(int(float(r["step"])), []).append(float(r[col]))
    return per


def _final(rows, col="success"):
    """(mean over tasks at the last COMPLETE eval pass, n_tasks, step) or (None, 0, None).

    A pass is complete when it has as many task rows as the widest pass in the file. An
    in-flight run's newest rows are often a half-written pass -- the later tasks have not
    been logged yet -- and averaging it scores the run on a subset of tasks and inflates it
    (luis, 2026-09-08: read scene-play as +0.48 instead of +0.41 for exactly this reason).
    A finished 1M run is unaffected, since its final eval completes before the checkpoint is
    written, but nothing here should depend on that."""
    per = _by_step(rows, col)
    if not per:
        return None, 0, None
    width = max(len(v) for v in per.values())
    complete = [s for s, v in per.items() if len(v) == width]
    if not complete:
        return None, 0, None
    last = max(complete)
    vals = per[last]
    return sum(vals) / len(vals), len(vals), last


def summarize(run: Path):
    ev, rel = _rows(run / "eval.csv"), _rows(run / "eval_relabel.csv")
    a, na, last_a = _final(ev)
    b, nb, last_b = _final(rel)
    ck = run / "params_1000000.pkl"
    wall = None
    cfg = run / "config.json"
    if ck.exists() and cfg.exists():
        wall = round((ck.stat().st_mtime - cfg.stat().st_mtime) / 3600.0, 2)
    d = {"final_step": last_a, "route_a": None if a is None else round(a, 4),
         "route_b": None if b is None else round(b, 4),
         "wall_hours": wall, "slurm_job": None,
         "note": f"evalbox (8x L4, tmux), tdjtab campaign; route_a/route_b = mean over "
                 f"{na or '?'} tasks at step {last_a}"}
    if rel:
        gd, _, _ = _final(rel, "inferred_cond_goals_dropped")
        d["goals_dropped_min"] = None if gd is None else round(gd, 1)
    else:
        d["route_b_absent"] = ("no eval_relabel.csv -- this run carried no eval_relabel_size, so "
                               "Route B never ran (main.py:517)")
    problems = []
    if not ck.exists():
        problems.append("no params_1000000.pkl -- run has not reached 1M")
    if na and nb and na != nb:
        problems.append(f"task count differs between routes: A={na} B={nb}")
    per_a = _by_step(ev)
    if per_a and last_a is not None and max(per_a) != last_a:
        problems.append(f"newest eval pass at step {max(per_a)} is INCOMPLETE "
                        f"({len(per_a[max(per_a)])} of {len(per_a[last_a])} tasks); "
                        f"scored the last complete pass at {last_a} instead")
    if rel and d.get("goals_dropped_min") in (0, 0.0):
        problems.append("inferred_cond_goals_dropped == 0 -- Route B rows are NOT trustworthy")
    return d, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="a run dir (repeatable)")
    ap.add_argument("--all-finished", action="store_true")
    ap.add_argument("--exp-root", default="exp/tdjtab")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="upload even with problems")
    a = ap.parse_args()

    runs = [Path(r) for r in a.run]
    if a.all_finished:
        runs += [p.parent for p in Path(a.exp_root).glob("*/params_1000000.pkl")]
    runs = sorted({r.resolve() for r in runs})
    if not runs:
        print("no runs selected", file=sys.stderr)
        return 1

    api = None
    if not a.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()

    rc = 0
    for run in runs:
        d, problems = summarize(run)
        print(f"\n=== {run.name}")
        print(json.dumps(d, indent=2))
        for p in problems:
            print(f"  !! {p}")
        if problems and not a.force:
            print("  -> NOT uploaded (pass --force to override)")
            rc = 1
            continue
        if a.dry_run:
            print("  -> dry run, nothing uploaded")
            continue
        for name in WANT:
            f = run / name
            if not f.exists():
                continue
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"exp/{run.name}/{name}",
                            repo_id=REPO, repo_type="model",
                            commit_message=f"handback: {run.name}/{name}")
            print(f"  uploaded exp/{run.name}/{name}")
        import io
        api.upload_file(path_or_fileobj=io.BytesIO((json.dumps(d, indent=2) + "\n").encode()),
                        path_in_repo=f"status/{run.name}.json", repo_id=REPO, repo_type="model",
                        commit_message=f"status: {run.name}")
        print(f"  wrote status/{run.name}.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
