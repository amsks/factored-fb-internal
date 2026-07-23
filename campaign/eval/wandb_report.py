"""campaign/eval/wandb_report.py — push ONE campaign run to wandb for iteration:
training curves (parsed from the queue's job log) + eval success + the 2x2 video grid.

One wandb run per (algo, task, seed), grouped by family, so algorithms line up
side-by-side in the wandb UI. Keeps only the useful signals (train curves / eval / video).

    python campaign/eval/wandb_report.py \
        --run_dir /raid/.../campaign/native/fb__b2d-center-ctrl__s0 \
        --agent fb --task b2d-center-ctrl --seed 0 --family Box2D \
        --eval_dir /raid/.../campaign/eval/fb/b2d-center-ctrl/s0 \
        --jobs campaign/jobs_native.txt --logs /raid/.../campaign/logs_native \
        --project factored-fb-campaign [--entity <team>]
"""
import argparse
import json
import re
from pathlib import Path

import wandb

# keep only these training signals (the useful ones); drop the rest of the noisy log.
KEEP = re.compile(r"train/(fb_loss|orth_loss|actor_loss|bc_flow_loss|bc_error|q|fb_diag|fb_offdiag|"
                  r"critic|actor)$|"
                  r"(actor_loss|critic_loss|bc_loss|q_mean|binary_accuracy|mse)")


def find_job_log(run_dir, jobs_file, logs_dir):
    if not (jobs_file and logs_dir and Path(jobs_file).exists()):
        return None
    lines = Path(jobs_file).read_text().splitlines()
    for i, line in enumerate(lines):
        if run_dir in line:
            p = Path(logs_dir) / f"job{i:04d}.log"
            return p if p.exists() else None
    return None


def parse_curves(log_path):
    """Parse '[step N] k=v k=v ...' (native) and '[jax_train] step N: {..}' (jax) lines."""
    rows = []
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = re.search(r"\[step\s+(\d+)\]\s+(.*)", line)          # native train.py
        if m:
            step = int(m.group(1))
            kv = {k: float(v) for k, v in re.findall(r"([\w/#]+)=([-\d.eE]+)", m.group(2))
                  if KEEP.search(k)}
            if kv:
                rows.append((step, kv))
            continue
        m = re.search(r"\[jax_train\] step (\d+): (\{.*\})", line)  # jax launcher
        if m:
            step = int(m.group(1))
            d = json.loads(m.group(2).replace("'", '"'))
            rows.append((step, {k: float(v) for k, v in d.items() if KEEP.search(k)}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--eval_dir", default=None, help="dir with metrics.json + eval_grid.mp4")
    ap.add_argument("--jobs", default=None)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--project", default="factored-fb-campaign")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--mode", default="online")
    args = ap.parse_args()

    run = wandb.init(project=args.project, entity=args.entity, mode=args.mode,
                     group=args.family, job_type=args.agent,
                     name=f"{args.agent}__{args.task}__s{args.seed}",
                     config=dict(agent=args.agent, task=args.task, seed=args.seed, family=args.family),
                     reinit=True)

    # 1) training curves
    log = find_job_log(args.run_dir, args.jobs, args.logs)
    n_curve = 0
    if log:
        for step, kv in parse_curves(log):
            wandb.log(kv, step=step)
            n_curve += 1

    # 2) eval success + video
    if args.eval_dir and (Path(args.eval_dir) / "metrics.json").exists():
        m = json.loads((Path(args.eval_dir) / "metrics.json").read_text())
        wandb.summary["eval/success_rate"] = m["success_rate"]
        wandb.summary["eval/n_episodes"] = m["n_episodes"]
        vid = m.get("video") or (str(Path(args.eval_dir) / "eval_grid.mp4")
                                 if (Path(args.eval_dir) / "eval_grid.mp4").exists() else None)
        if vid and Path(vid).exists():
            wandb.log({"eval/video": wandb.Video(vid, format="mp4")})
        print(f"[wandb_report] eval success={m['success_rate']:.3f}")

    run.finish()
    print(f"[wandb_report] {args.agent}__{args.task}__s{args.seed}: "
          f"{n_curve} curve points -> {run.url if hasattr(run,'url') else 'wandb'}")


if __name__ == "__main__":
    main()
