"""Collect the final evaluation of every run in exp/softmax/* into a markdown table.

Each run directory holds one <env>_<critic>_<actor>_seed_<k>/ folder with eval.csv
(goal-conditioned route: one row per task, `success` column, `step`) and, for the
FB runs, eval_relabel.csv (reward-inference route, same layout). Success is the
mean over the task rows at the last evaluation step.
Usage: python scripts/collect_training.py [--md results/training_table.md] [--json results/training_table.json]"""
import glob, os, csv, argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--md"); ap.add_argument("--json"); a = ap.parse_args()

def last_eval(path):
    if not os.path.exists(path): return None, None
    rows = list(csv.DictReader(open(path)))
    rows = [r for r in rows if r.get("success") not in ("", None)]
    if not rows: return None, None
    step = rows[-1]["step"]
    sel = [float(r["success"]) for r in rows if r["step"] == step]
    per_task = {}
    for r in rows:
        if r["step"] == step: per_task.setdefault(r.get("task_id", "?"), []).append(float(r["success"]))
    return step, dict(mean=sum(sel) / len(sel), n=len(sel), per_task={k: sum(v)/len(v) for k, v in per_task.items()})

rows = []
for d in sorted(glob.glob("../../exp/softmax/*/")):
    tag = os.path.basename(d.rstrip("/"))
    subs = [s for s in glob.glob(os.path.join(d, "*/")) if os.path.exists(os.path.join(s, "config.json"))]
    r = dict(tag=tag, step="-", goal_success="-", reward_success="-", status="running", per_task_goal=None, per_task_reward=None)
    log = os.path.join(d, "stdout.log")
    if os.path.exists(log) and "Traceback" in open(log, errors="ignore").read(): r["status"] = "CRASHED"
    if subs:
        s = subs[0]
        tr = os.path.join(s, "train.csv")
        if os.path.exists(tr):
            try: r["step"] = sum(1 for _ in open(tr)) - 1
            except Exception: pass
        gs, g = last_eval(os.path.join(s, "eval.csv"))
        rs, z = last_eval(os.path.join(s, "eval_relabel.csv"))
        if g:
            r["step"] = gs; r["goal_success"] = f"{g['mean']:.3f}"; r["per_task_goal"] = g["per_task"]
            if r["status"] != "CRASHED": r["status"] = "done" if int(float(gs)) >= 500000 else "partial"
        if z:
            r["reward_success"] = f"{z['mean']:.3f}"; r["per_task_reward"] = z["per_task"]
    rows.append(r)
hdr = ["tag", "step", "goal_success", "reward_success", "status"]
out = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows)
print(out)
if a.md: open(a.md, "w").write(out + "\n")
if a.json: json.dump(rows, open(a.json, "w"), indent=1)
