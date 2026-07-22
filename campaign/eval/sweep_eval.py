"""campaign/eval/sweep_eval.py — sweep the per-checkpoint evaluator over a family's
tasks x training-seeds and aggregate into the reporting format:

  * per task:   success rate, averaged over training seeds  (mean +/- std)
  * per family: mean success over tasks (each task = mean over its seeds)

Each (task, seed) eval runs `eval_fb.py` as a subprocess in newt's venv (isolated),
using FIXED per-episode seeds so every checkpoint sees identical initial states.

    python campaign/eval/sweep_eval.py --family Box2D \
        --native_dir /raid/tongle/rl/fb-data/campaign/native \
        --agent fb_flowbc --seeds 0,1,2 --ckpt final.pt \
        --n_episodes 100 --n_video 4 --out /raid/tongle/rl/fb-data/campaign/eval
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/raid/tongle/rl/Factored-FB/campaign")
from registry import DATASETS, COLLECTED  # noqa: E402

NEWT_PY = "/raid/tongle/rl/newt/.venv/bin/python"
EVAL_FB = "/raid/tongle/rl/Factored-FB/campaign/eval/eval_fb.py"


def family_tasks(family):
    return [d for d in DATASETS if d["family"] == family]


def run_one(run_dir, task, data_dir, ckpt, n_episodes, seed_base, n_video, out_dir):
    out_dir = Path(out_dir)
    cmd = [NEWT_PY, EVAL_FB, "--run_dir", run_dir, "--ckpt", ckpt, "--task", task,
           "--data_dir", data_dir, "--n_episodes", str(n_episodes),
           "--seed_base", str(seed_base), "--n_video", str(n_video), "--out", str(out_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    mfp = out_dir / "metrics.json"
    if mfp.exists():
        return json.loads(mfp.read_text())
    return {"success_rate": None, "error": r.stderr[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help="Box2D | Metaworld | ManiSkill | OGBench")
    ap.add_argument("--native_dir", default="/raid/tongle/rl/fb-data/campaign/native")
    ap.add_argument("--agent", default="fb_flowbc", help="fb | fb_flowbc")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--n_video", type=int, default=4)
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/campaign/eval")
    ap.add_argument("--tasks", default=None, help="comma list of slugs to restrict to")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = family_tasks(args.family)
    if args.tasks:
        keep = set(args.tasks.split(","))
        tasks = [t for t in tasks if t["slug"] in keep]

    per_task = {}
    for ds in tasks:
        slug, task = ds["slug"], ds["domain"].split("/")[-1] if "/" in ds["domain"] else ds["slug"]
        data_dir = f"{COLLECTED}/{ds['family']}__{slug}"
        seed_success = []
        for sd in seeds:
            run_dir = f"{args.native_dir}/{args.agent}__{slug}__s{sd}"
            if not Path(run_dir, "checkpoints", args.ckpt).exists():
                print(f"[sweep] MISSING {run_dir}/checkpoints/{args.ckpt}")
                continue
            out_dir = f"{args.out}/{args.agent}/{slug}/s{sd}"
            m = run_one(run_dir, slug, data_dir, args.ckpt, args.n_episodes, 0,
                        args.n_video if sd == seeds[0] else 0, out_dir)  # video for first seed only
            sr = m.get("success_rate")
            print(f"[sweep] {args.agent} {slug} s{sd}: success={sr}")
            if sr is not None:
                seed_success.append(sr)
        if seed_success:
            per_task[slug] = dict(mean=float(np.mean(seed_success)),
                                  std=float(np.std(seed_success)),
                                  seeds=seed_success)

    task_means = [v["mean"] for v in per_task.values()]
    summary = {
        "family": args.family, "agent": args.agent, "ckpt": args.ckpt,
        "n_episodes": args.n_episodes, "seeds": seeds,
        "per_task": per_task,
        "family_mean_success": float(np.mean(task_means)) if task_means else None,
        "family_std_over_tasks": float(np.std(task_means)) if task_means else None,
    }
    outp = Path(args.out) / f"summary_{args.family}_{args.agent}.json"
    outp.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.family} / {args.agent} ({args.ckpt}) — {args.n_episodes} eps, seeds {seeds} ===")
    for slug, v in per_task.items():
        print(f"  {slug:28s} {v['mean']:.3f} +/- {v['std']:.3f}")
    if task_means:
        print(f"  {'FAMILY MEAN (over tasks)':28s} {summary['family_mean_success']:.3f} "
              f"+/- {summary['family_std_over_tasks']:.3f}")
    print(f"[sweep] wrote {outp}")


if __name__ == "__main__":
    main()
