"""campaign/run_queue.py — GPU job queue with per-GPU packing.

Reads a job file (one shell command per line, WITHOUT any CUDA_VISIBLE_DEVICES —
the queue assigns it), spreads jobs over gpus x pack concurrent slots, and pulls the
next job onto a slot as soon as it frees. Writes a live status.json. Detached-friendly.

    setsid nohup <fb>/.venv/bin/python campaign/run_queue.py \
        --jobs campaign/jobs_native.txt --gpus 0,1,2,3,4,5,6,7 --pack 4 \
        --logdir /raid/tongle/rl/fb-data/campaign/logs_native \
        > /raid/tongle/rl/fb-data/campaign/logs_native/queue.log 2>&1 & disown
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--pack", type=int, default=4)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--cwd", default="/raid/tongle/rl/Factored-FB")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    slots = [g for g in gpus for _ in range(args.pack)]      # one entry per concurrent slot
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    jobs = [l.strip() for l in Path(args.jobs).read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    status_fp = logdir / "status.json"
    t0 = time.time()

    free = list(slots)
    running = []          # list of dicts: proc, idx, gpu, log
    ji = 0
    done, failed = [], []

    def write_status(finished=False):
        status_fp.write_text(json.dumps({
            "total": len(jobs), "launched": ji, "running": len(running),
            "done": len(done), "failed": len(failed), "failed_idx": failed,
            "elapsed_min": round((time.time() - t0) / 60, 1),
            "finished": finished,
        }, indent=2))

    while ji < len(jobs) or running:
        while free and ji < len(jobs):
            gpu = free.pop()
            log = open(logdir / f"job{ji:04d}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_GL="egl")
            log.write(f"# job {ji} on GPU {gpu}\n# {jobs[ji]}\n")
            log.flush()
            p = subprocess.Popen(jobs[ji], shell=True, cwd=args.cwd, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running.append(dict(proc=p, idx=ji, gpu=gpu, log=log))
            ji += 1
        write_status()
        time.sleep(5)
        for r in list(running):
            rc = r["proc"].poll()
            if rc is not None:
                r["log"].close()
                (done if rc == 0 else failed).append(r["idx"])
                free.append(r["gpu"])
                running.remove(r)
    write_status(finished=True)
    print(f"[run_queue] ALL DONE: {len(done)} ok, {len(failed)} failed "
          f"in {round((time.time()-t0)/60,1)} min; failed idx: {failed}")


if __name__ == "__main__":
    main()
