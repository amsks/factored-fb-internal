#!/usr/bin/env python3
"""Compare the FB-matched PSM lr_phi re-check (NEW basis: z_dim50/disc0.99/tau0.005)
against the original sweep (OLD basis: z_dim128/disc0.98/tau0.01) AT THE SAME STEP,
for both lr_phi in {1e-4, 1e-5}. ortho_coef=1 in both.

For each lr_phi we take the NEW runs' current progress (min max-step across its 3
seeds, so all three have data) as the target step, then read OLD at that same step.
Reports eval success (latest common eval) + key critic metrics (q, sf_M_diag,
sf_offdiag, orth_offdiag) at the target step, averaged over the 3 seeds.

Usage: python scripts/lrcheck_progress.py [NEW_GRID_DIR] [OLD_GRID_DIR]
Defaults: NEW = newest logs/grid_* containing 'fbmatch_lrcheck'; OLD = grid_20260608_080705.
"""
import glob
import os
import re
import sys

STEP_RE = re.compile(r"^\[step\s+(\d+)\]")
METRICS = ["q", "sf_M_diag", "sf_offdiag", "orth_loss_offdiag"]
EVAL_KEY = "eval/reward/eval/success"


def _val(line, key):
    m = re.search(rf"train/{re.escape(key)}=([-0-9.eE]+)", line)
    return float(m.group(1)) if m else None


def parse(path):
    """-> (train: list[(step, {metric:val})], evals: list[(step, success)])."""
    train, evals = [], []
    with open(path) as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            d = {k: _val(line, k) for k in METRICS}
            if any(v is not None for v in d.values()):
                train.append((step, d))
            em = re.search(rf"{re.escape(EVAL_KEY)}=([-0-9.eE]+)", line)
            if em:
                evals.append((step, float(em.group(1))))
    return train, evals


def at_step(series, target, key=None):
    """Last entry with step <= target. key=None -> eval (val is the float)."""
    best = None
    for step, payload in series:
        if step <= target:
            best = (step, payload)
    if best is None:
        return None
    step, payload = best
    return (step, payload if key is None else payload.get(key))


def seed_logs(grid, lr):
    return sorted(glob.glob(os.path.join(grid, f"*ortho_coef1__lr_phi{lr}__s*.log")))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else "  -  "


def main():
    grids = glob.glob("logs/grid_*")
    new_default = max(
        (g for g in grids if seed_logs(g, "1e-4") and "fbmatch_lrcheck" in
         (open(seed_logs(g, "1e-4")[0]).readline() if seed_logs(g, "1e-4") else "")),
        default=None, key=os.path.getmtime,
    )
    # robust fallback: newest grid whose seed logs are named *fbmatch_lrcheck*
    if new_default is None:
        cand = [g for g in grids if glob.glob(os.path.join(g, "*fbmatch_lrcheck*"))]
        new_default = max(cand, key=os.path.getmtime) if cand else None
    NEW = sys.argv[1] if len(sys.argv) > 1 else new_default
    OLD = sys.argv[2] if len(sys.argv) > 2 else "logs/grid_20260608_080705"
    print(f"NEW (FB-matched z50/0.99/0.005): {NEW}")
    print(f"OLD (orig      z128/0.98/0.01):  {OLD}\n")

    for lr in ["1e-4", "1e-5"]:
        new_logs, old_logs = seed_logs(NEW, lr), seed_logs(OLD, lr)
        if not new_logs:
            print(f"lr_phi={lr}: no NEW logs yet\n")
            continue
        new_parsed = [parse(p) for p in new_logs]
        old_parsed = [parse(p) for p in old_logs]
        # target step = min over NEW seeds of their max train step
        new_maxsteps = [t[-1][0] for t, _ in new_parsed if t]
        if not new_maxsteps:
            print(f"lr_phi={lr}: NEW not logging yet\n")
            continue
        target = min(new_maxsteps)

        print(f"=== lr_phi={lr}  (target step = {target}, n_new={len(new_logs)} n_old={len(old_logs)}) ===")
        header = f"{'metric':<16}{'NEW':>12}{'OLD':>12}{'Δ(new-old)':>14}"
        print(header)
        for key in METRICS:
            nv = mean([(at_step(t, target, key) or (None, None))[1] for t, _ in new_parsed])
            ov = mean([(at_step(t, target, key) or (None, None))[1] for t, _ in old_parsed])
            d = (nv - ov) if (nv is not None and ov is not None) else None
            print(f"{key:<16}{fmt(nv):>12}{fmt(ov):>12}{fmt(d):>14}")
        # eval success at latest common NEW eval step
        new_eval_max = [e[-1][0] for _, e in new_parsed if e]
        if new_eval_max:
            etarget = min(new_eval_max)
            ns = mean([(at_step(e, etarget) or (None, None))[1] for _, e in new_parsed])
            os_ = mean([(at_step(e, etarget) or (None, None))[1] for _, e in old_parsed])
            d = (ns - os_) if (ns is not None and os_ is not None) else None
            print(f"{'eval_success':<16}{fmt(ns):>12}{fmt(os_):>12}{fmt(d):>14}   (@eval step {etarget})")
        else:
            print(f"{'eval_success':<16}{'(no eval yet)':>12}")
        print()


if __name__ == "__main__":
    main()
