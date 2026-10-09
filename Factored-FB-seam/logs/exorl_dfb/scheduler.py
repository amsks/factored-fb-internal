#!/usr/bin/env python3
"""Phase-gated GPU job scheduler for the Density-FB campaign.

Reads a manifest (TSV: phase<TAB>name<TAB>script<TAB>done_marker<TAB>logfile) and runs the
jobs with a fixed number of concurrent slots per GPU. Jobs whose done_marker already exists
are skipped (idempotent / resume-aware, since each job script passes --auto_resume). A later
phase does not start until every job of the earlier phase has produced its done_marker.

Robust to being killed: re-run it and it picks up where the checkpoints left off.

    python3 logs/exorl_dfb/scheduler.py --manifest logs/exorl_dfb/manifest.tsv \
        --gpus 0,1,2,3,4,5,6,7 --slots-per-gpu 2 --max-retries 2
"""
import argparse, os, subprocess, sys, time, datetime

def log(msg, statusfile=None):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if statusfile:
        with open(statusfile, "a") as f:
            f.write(line + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x != ""]
    statusfile = os.path.join(os.path.dirname(args.manifest), "scheduler_status.txt")

    def read_manifest():
        # Re-read on every phase so rows appended while an earlier phase runs (e.g. the
        # OGBench phase 2) get picked up. Keyed by name; retry counts reset per (re)read,
        # which is fine because auto_resume continues from the last checkpoint.
        out = []
        with open(args.manifest) as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln.strip():
                    continue
                phase, name, script, done, logf = ln.split("\t")
                out.append(dict(phase=int(phase), name=name, script=script,
                                done=done, log=logf, retries=0))
        return out

    log(f"scheduler start: gpus {gpus}, {args.slots_per_gpu}/gpu = "
        f"{len(gpus)*args.slots_per_gpu} slots", statusfile)

    running = {}   # name -> dict(proc, gpu, job)
    gpu_load = {g: 0 for g in gpus}
    failed = set()  # names that exhausted retries; never re-picked

    def done_exists(j):
        return os.path.exists(j["done"])

    def settled(j):
        # done, or given up on — either way it no longer blocks its phase
        return done_exists(j) or j["name"] in failed

    def launch(j):
        # pick least-loaded gpu with a free slot
        g = min((g for g in gpus if gpu_load[g] < args.slots_per_gpu),
                key=lambda g: gpu_load[g], default=None)
        if g is None:
            return False
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        lf = open(j["log"], "a")
        lf.write(f"\n==== scheduler launch {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
                 f"gpu={g} try={j['retries']} ====\n")
        lf.flush()
        p = subprocess.Popen(["bash", j["script"]], stdout=lf, stderr=subprocess.STDOUT,
                             env=env, cwd=os.path.expanduser("~/git/Factored-FB-seam"))
        running[j["name"]] = dict(proc=p, gpu=g, job=j, lf=lf)
        gpu_load[g] += 1
        log(f"LAUNCH {j['name']} on gpu{g} (try {j['retries']})  [{len(running)} running]",
            statusfile)
        return True

    while True:
        jobs = read_manifest()
        # smallest phase that still has an unsettled job; the phase gate is implicit —
        # a later phase is only reached once every earlier phase is fully settled
        # (done or permanently failed).
        pend_phases = sorted({j["phase"] for j in jobs if not settled(j)})
        if not pend_phases:
            break
        phase = pend_phases[0]
        pjobs = [j for j in jobs if j["phase"] == phase]
        log(f"=== PHASE {phase}: {len(pjobs)} jobs ===", statusfile)
        pending = [j for j in pjobs if not settled(j)]
        for j in pjobs:
            if done_exists(j):
                log(f"SKIP(done) {j['name']}", statusfile)

        while pending or running:
            # fill free slots from this phase's pending queue
            while pending and any(gpu_load[g] < args.slots_per_gpu for g in gpus):
                j = pending.pop(0)
                if done_exists(j):
                    log(f"SKIP(done) {j['name']}", statusfile)
                    continue
                if not launch(j):
                    pending.insert(0, j)
                    break
            # poll
            time.sleep(args.poll)
            for name in list(running.keys()):
                r = running[name]
                rc = r["proc"].poll()
                if rc is None:
                    continue
                gpu_load[r["gpu"]] -= 1
                r["lf"].close()
                del running[name]
                j = r["job"]
                if done_exists(j):
                    log(f"DONE {name} (rc={rc})", statusfile)
                elif j["retries"] < args.max_retries:
                    j["retries"] += 1
                    log(f"RETRY {name} (rc={rc}, no done_marker) -> try {j['retries']}",
                        statusfile)
                    pending.append(j)
                else:
                    failed.add(name)
                    log(f"FAILED {name} (rc={rc}, gave up after {j['retries']} retries)",
                        statusfile)
        log(f"=== PHASE {phase} COMPLETE ===", statusfile)

    log("ALL PHASES COMPLETE", statusfile)

if __name__ == "__main__":
    main()
