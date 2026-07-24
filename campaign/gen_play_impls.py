"""campaign/gen_play_impls.py — the PLAY-DATA experiment: does biql (and the other
goal-conditioned critics) work as well on TD-MPC2 play data as it does on OGBench cube-single?

Two things are compared, both with a co-trained BC head (--teacher off) so the ONLY thing that
changes is the data regime:
  * play grid   : {iql_bilinear, gciql, fb} x flowbc on the play data (data_collection/collect_play.py)
  * expert ctrl : iql_bilinear x flowbc on the original expert demos (registry default dir)

The agg_group encodes the regime (`iql_bilinear_flowbc_play`, `..._expert`), so /result puts the
two regimes side by side per task. cube-single is included as the play-data reference point (its
OGBench play data is what motivated the whole thing).

    <fb>/.venv/bin/python campaign/gen_play_impls.py   -> campaign/jobs_play.txt
"""
import argparse
import os
import sys
from pathlib import Path

FB = "/raid/tongle/rl/Factored-FB"
NEWT_PY = "/raid/tongle/rl/newt/.venv/bin/python"
CAMPAIGN_OUT = "/raid/tongle/rl/fb-data/campaign"
PLAY_ROOT = "/raid/tongle/rl/fb-data/collected_play"
sys.path.insert(0, FB)
from env import registry  # noqa: E402

TASKS = ["b2d-center-ctrl", "b2d-center-ctrl-target", "b2d-center-wall",
         "b2d-goal-ctrl-maze", "b2d-goal-ctrl-target",
         "ms-pick-cube", "ms-push-cube", "mw-push", "mw-pick-place"]
SEEDS = [0, 1, 2]
TRAIN_STEPS = 500_000
SAVE_INTERVAL = 100_000
EVAL_INTERVAL = 50_000
EVAL_EPISODES = 50
VIDEO_INTERVAL = 250_000
VIDEO_EPISODES = 2


def play_dir(task):
    fam = registry._prefix(task)[2]                 # Box2D / Metaworld / ManiSkill
    return os.path.join(PLAY_ROOT, f"{fam}__{task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="play_v1")
    ap.add_argument("--out", default=f"{FB}/campaign/jobs_play.txt")
    ap.add_argument("--play_critics", default="iql_bilinear,gciql,fb")
    ap.add_argument("--expert_critics", default="iql_bilinear")   # biql expert control
    args = ap.parse_args()

    agg_dir = f"{CAMPAIGN_OUT}/impls_eval/{args.tag}"
    Path(agg_dir).mkdir(parents=True, exist_ok=True)
    play_critics = [c for c in args.play_critics.split(",") if c]
    expert_critics = [c for c in args.expert_critics.split(",") if c]

    def emit(lines, critic, task, seed, regime):
        algo = f"{critic}_flowbc_{regime}"
        save_dir = f"{CAMPAIGN_OUT}/{args.tag}/{algo}__{task}__s{seed}"
        ov = "" if regime == "expert" else f"--override data_dir={play_dir(task)} "
        video = (f"--eval_video_episodes {VIDEO_EPISODES} --video_interval {VIDEO_INTERVAL} "
                 if seed == 0 else "")
        lines.append(
            f"MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH={FB} {NEWT_PY} "
            f"-m impls.main --critic {critic} --actor flowbc --domain {task} --seed {seed} "
            f"{ov}--save_dir {save_dir} --train_steps {TRAIN_STEPS} "
            f"--save_interval {SAVE_INTERVAL} --eval_interval {EVAL_INTERVAL} "
            f"--eval_episodes {EVAL_EPISODES} " + video +
            f"--agg_dir {agg_dir} --agg_group {algo}__{task}")

    lines = []
    for critic in play_critics:                     # play grid (seed-contiguous per group)
        for task in TASKS:
            for s in SEEDS:
                emit(lines, critic, task, s, "play")
    for critic in expert_critics:                   # expert control
        for task in TASKS:
            for s in SEEDS:
                emit(lines, critic, task, s, "expert")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[gen_play] wrote {len(lines)} jobs -> {args.out}")
    print(f"[gen_play]   play: {play_critics} x flowbc x {len(TASKS)} tasks x {len(SEEDS)} seeds")
    print(f"[gen_play]   expert control: {expert_critics} x flowbc")
    print(f"[gen_play]   agg_dir: {agg_dir}")

    # GRID.json for /result
    import json
    json.dump(dict(campaign=args.tag, hypothesis="does biql work on TD-MPC2 play data like on cube-single",
                   play_critics=play_critics, expert_critics=expert_critics, actor="flowbc",
                   teacher="off (co-trained BC)", tasks=TASKS, seeds=SEEDS,
                   train_steps=TRAIN_STEPS, eval_interval=EVAL_INTERVAL,
                   regimes=["play", "expert"], group_format="{critic}_flowbc_{play|expert}__{task}",
                   note="compare {critic}_flowbc_play vs _expert per task; biql is the focus"),
              open(f"{agg_dir}/GRID.json", "w"), indent=1)


if __name__ == "__main__":
    main()
