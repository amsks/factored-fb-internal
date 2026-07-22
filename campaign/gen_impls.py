"""campaign/gen_impls.py — emit a job list that drives the FACTORED tree (impls.main,
critic x actor) instead of our old monolithic jax_agents.

    <fb>/.venv/bin/python campaign/gen_impls.py --critics biql --actors ddpgbc,fql
        -> campaign/jobs_impls.txt

Each line is one (critic, actor, domain, seed). Eval/checkpoint cadence and the per-seed
eval jsonl match the previous campaigns, so campaign/wandb_agg_daemon.py aggregates the
seeds into {critic}_{actor}__{domain}__agg with no changes.
"""
import argparse
import glob
import re
import shutil
from pathlib import Path

FB = "/raid/tongle/rl/Factored-FB"
NEWT_PY = "/raid/tongle/rl/newt/.venv/bin/python"      # unified venv: jax 0.4.38 + all envs
CAMPAIGN_OUT = "/raid/tongle/rl/fb-data/campaign"
TEACHER_EVAL = f"{CAMPAIGN_OUT}/teacher_eval/v1"       # pretrained-teacher baseline evals
RUN_TAG = "impls_v1"


def seed_bc_baseline(agg_dir, policy, demos, domains):
    """Copy the pretrained teacher's OWN eval (from TEACHER_EVAL) into the grid's agg_dir as a
    `bc-{policy}{demos}__{task}` group, so /result shows the BC baseline the grid distills from
    right alongside the critic x actor rows. Skips tasks the teacher wasn't trained on (cube)."""
    n = 0
    for dom in domains:
        for src in glob.glob(f"{TEACHER_EVAL}/{policy}{demos}__{dom}__s*.jsonl"):
            seed = re.search(r"__s(\d+)\.jsonl$", src).group(1)
            dst = f"{agg_dir}/bc-{policy}{demos}__{dom}__s{seed}.jsonl"
            shutil.copy(src, dst)
            n += 1
    print(f"[gen_impls]   seeded {n} BC-baseline jsonls (bc-{policy}{demos}) into the agg_dir")

# The 10 dev tasks. cube-single goes through the `ogbench` adapter (mean success over its
# task_infos); the other 9 through the `unified` adapter (success + Box2D-only at-goal ticks).
DOMAINS = ["cube-single",
           "b2d-center-ctrl", "b2d-center-ctrl-target", "b2d-center-wall",
           "b2d-goal-ctrl-maze", "b2d-goal-ctrl-target",
           "mw-push", "mw-pick-place", "ms-pick-cube", "ms-push-cube"]
SEEDS = [0, 1, 2]
TRAIN_STEPS = 500_000
SAVE_INTERVAL = 100_000
EVAL_INTERVAL = 20_000
EVAL_EPISODES = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critics", default="biql")
    ap.add_argument("--actors", default="ddpgbc,fql")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--out", default=f"{FB}/campaign/jobs_impls.txt")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--teacher", default=None, help="detached-BC teacher: dp | l2bc | None")
    ap.add_argument("--teacher_demos", type=int, default=128, help="teacher checkpoint (128 or 64)")
    args = ap.parse_args()

    critics = [c for c in args.critics.split(",") if c]
    actors = [a for a in args.actors.split(",") if a]
    domains = [d for d in args.domains.split(",") if d]
    agg_dir = f"{CAMPAIGN_OUT}/impls_eval/{args.tag}"
    Path(agg_dir).mkdir(parents=True, exist_ok=True)
    tflag = (f" --teacher {args.teacher} --teacher_demos {args.teacher_demos}"
             if args.teacher else "")

    lines = []
    # seed innermost so a group's 3 seeds are contiguous -> they co-run and the __agg run
    # (which needs all seeds at a step) fills in early.
    for critic in critics:
        for actor in actors:
            for dom in domains:
                for s in SEEDS:
                    save_dir = f"{CAMPAIGN_OUT}/{args.tag}/{critic}_{actor}__{dom}__s{s}"
                    lines.append(
                        f"MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false "
                        f"PYTHONPATH={FB} {NEWT_PY} -m impls.main "
                        f"--critic {critic} --actor {actor} --domain {dom} --seed {s} "
                        f"--save_dir {save_dir} --train_steps {TRAIN_STEPS} "
                        f"--save_interval {SAVE_INTERVAL} --eval_interval {EVAL_INTERVAL} "
                        f"--eval_episodes {EVAL_EPISODES} "
                        f"--agg_dir {agg_dir} "
                        f"--agg_group {critic}_{actor}__{dom}{tflag}"
                    )
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[gen_impls] wrote {len(lines)} jobs -> {args.out}")
    print(f"[gen_impls]   {len(critics)} critic(s) x {len(actors)} actor(s) x "
          f"{len(domains)} domains x {len(SEEDS)} seeds")
    print(f"[gen_impls]   agg_dir: {agg_dir}")
    # Phase D: when a grid distills from a teacher, drop that teacher's OWN eval into the
    # agg_dir so the BC baseline appears as a row in /result next to the critic x actor grid.
    if args.teacher:
        seed_bc_baseline(agg_dir, args.teacher, args.teacher_demos, domains)


if __name__ == "__main__":
    main()
