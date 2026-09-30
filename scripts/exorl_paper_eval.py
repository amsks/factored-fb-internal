"""Final ExoRL evaluation of Density-FB for the paper's ExoRL table.

For every (domain, task, seed) this runs scripts/reeval_checkpoint.py once: draw 50,000
transitions from the offline dataset, relabel their rewards for the task by re-simulating
MuJoCo (env/exorl_env.py), infer the task vector z from those reward samples, then roll the
frozen policy on z for 10 episodes with no goal given. Each cell appends one row to --out.
The agent is rebuilt from the config.json the run recorded, so the evaluation uses exactly
the networks and hit spec it was trained with; only the task is set here.

Tasks per domain: TD-JEPA's task set for walker, cheetah and quadruped, and the four maze
corners for point-mass. The table cell is the mean over seeds of each seed's task-mean
return (scripts/score_tables.py).

Run directories are <root>/<domain>/density_fb_z/<domain>_density_fb_z_ddpgbc_seed_<s>
(the layout scripts/launch_exorl.sh writes).

    python -m scripts.exorl_paper_eval --root exp/exorl --domains dmc-walker \
        --out exp/paper_eval_exorl_walker.jsonl
    python -m scripts.exorl_paper_eval --out exp/paper_eval_exorl_walker.jsonl --report-only
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import exorl_env  # noqa: E402

TASKS = {
    "dmc-walker": ("stand", "walk", "run", "spin"),
    "dmc-cheetah": ("walk", "walk_backward", "run", "run_backward"),
    "dmc-quadruped": ("stand", "walk", "run", "jump"),
    "dmc-point-mass-maze": ("reach_top_left", "reach_top_right",
                            "reach_bottom_left", "reach_bottom_right"),
}
CRITIC, ACTOR = "density_fb_z", "ddpgbc"


def _check_tasks():
    """The task names must be the env layer's, or a typo would score a different task."""
    for dom, tasks in TASKS.items():
        missing = [t for t in tasks if t not in set(exorl_env.reward_tasks(dom))]
        assert not missing, f"{dom}: {missing} not in env/exorl_env.py TASKS"


def run_cell(args, domain, task, seed):
    run_dir = os.path.join(args.root, domain, CRITIC, f"{domain}_{CRITIC}_{ACTOR}_seed_{seed}")
    cmd = [args.python, "-m", "scripts.reeval_checkpoint",
           "--critic", CRITIC, "--actor", ACTOR, "--domain", domain,
           "--seed", str(seed), "--restore_path", run_dir,
           "--restore_epoch", str(args.steps),
           "--eval_episodes", str(args.episodes),
           "--infer_cond_episodes", str(args.inference_samples),
           "--eval_on_inferred_cond",
           "--tag", f"{CRITIC}|{domain}|{task}|s{seed}",
           "--out_jsonl", args.out,
           "--override", f"exorl_task={task}"]
    if args.dry_run:
        print(("MISSING-RUNDIR " if not os.path.isdir(run_dir) else "") + " ".join(cmd))
        return
    if not os.path.isdir(run_dir):
        print(f"[missing] {run_dir}", flush=True)
        return
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Reward relabelling re-simulates MuJoCo. If this environment's mujoco is too old for
    # dm_control, point FB_MUJOCO_SITE_PACKAGES at a newer one (env/exorl_env.py).
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] {domain}/{task}/s{seed} rc={r.returncode}\n{r.stderr[-2000:]}", flush=True)
    else:
        print(f"[ok] {domain}/{task}/s{seed}", flush=True)


def report(out_path):
    """Per (domain, task): mean and sd of the episode return over seeds. A cell re-scored
    later in the append-only file keeps its last row, so a seed is never counted twice."""
    cells = {}
    for line in open(out_path):
        if not line.strip():
            continue
        row = json.loads(line)
        _, dom, task, seed = str(row["tag"]).split("|")
        cells.setdefault((dom, task), {})[seed] = float(row["episode_return"])
    print(f"{'domain':<22} {'task':<20} {'mean':>9} {'sd':>8}  n")
    for (dom, task), by_seed in sorted(cells.items()):
        vals = list(by_seed.values())
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{dom:<22} {task:<20} {statistics.mean(vals):9.1f} {sd:8.1f}  {len(vals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exp/exorl")
    ap.add_argument("--domains", nargs="*", default=sorted(TASKS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=1_000_000, help="checkpoint step to restore")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--inference-samples", type=int, default=50_000,
                    help="reward-labelled transitions used to infer z")
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--jobs", type=int, default=2, help="cells evaluated concurrently")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    _check_tasks()
    if args.report_only:
        report(args.out)
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    todo = [(d, t, s) for d in args.domains for t in TASKS[d] for s in args.seeds]
    print(f"{len(todo)} cells ({args.jobs} at a time)", flush=True)
    # Each cell is its own subprocess; the threads only wait on them.
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda t: run_cell(args, *t), todo))
    if not args.dry_run:
        print()
        report(args.out)


if __name__ == "__main__":
    main()
