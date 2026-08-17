#!/usr/bin/env python3
"""scripts/rescore_radius.py — E5 of docs/specs/2026-08-10-deconfound-size-coverage.md:
re-score every MW/MS cell at a goal radius that makes the env as DISCRIMINATING as cube.

C5: Metaworld's goal box is small relative to its success radius, so a relabeled goal is
often already satisfied a full eval horizon later -- P(satisfied at 1-2 x H) is 0.255 (push) /
0.155 (pick-place) against cube's 0.013. Temporal distance therefore carries less signal and
the env is a weaker critic discriminator even with size, masking and eval conditioning all
repaired. That is a property of the env, not of the data, and the fix is to score at a
TIGHTER radius r' at which the same probability matches cube's.

Two inputs, both already produced by the rest of the spec:
  * the dataset npz, for the displacement distribution that defines r' (E0/E2/E3 collections);
  * <run_dir>/eval_episodes.jsonl, the per-episode goal distances evaluate() logs (E0.2) --
    which is why this is an analysis pass and not a re-run.

    python scripts/rescore_radius.py \
        --data $SCRATCH/fb-data/play-resample/data/metaworld/mw-push.npz --task mw-push \
        --runs '$SCRATCH/factored-fb-E3/mw-push/*/mw-push_*_seed_*' \
        --json-out results/deconfound/e5-mw-push.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np

# (success radius, components the env's success rule measures, eval horizon in env steps).
TASKS = {"mw-push": (0.05, 3, 100), "mw-pick-place": (0.07, 3, 100),
         "ms-push-cube": (0.10, 2, 25), "ms-pick-cube": (0.025, 3, 25)}
OBJ = {"mw-push": slice(4, 7), "mw-pick-place": slice(4, 7),
       "ms-push-cube": slice(28, 31), "ms-pick-cube": slice(29, 32)}
# The target this matches to: cube-single-play's P(||block(t+dt) - block(t)|| < 0.04) over
# dt in [1, 2] x its 200-step eval horizon, measured by
# scripts/compare_play_resample_to_ogbench.py.
CUBE_P = 0.013


def displacements(npz, task, n_anchor=60, n_traj=300, seed=0):
    """||object(t+dt) - object(t)|| for dt in [1, 2] x H, in the env's own metric."""
    radius, k, H = TASKS[task]
    z = np.load(npz)
    pos = z["next_observation"][..., OBJ[task]][..., :k]
    N, T = pos.shape[:2]
    lo, hi = max(1, H), min(2 * H, T - 1)
    if lo >= hi:
        raise SystemExit(f"{task}: trajectories are {T} steps, too short for a {lo}-{hi} "
                         f"lookahead; r' is not defined on this data")
    rng = np.random.RandomState(seed)
    out = []
    for i in rng.choice(N, size=min(N, n_traj), replace=False):
        t = rng.randint(0, T - hi, size=n_anchor)
        off = rng.randint(lo, hi, size=n_anchor)
        out.append(np.linalg.norm(pos[i, t + off] - pos[i, t], axis=-1))
    return np.concatenate(out)


def matched_radius(disp, target=CUBE_P):
    """The r' at which P(displacement < r') == target. This is just the empirical quantile of
    the displacement distribution, which is exactly what "as discriminating as cube" means."""
    return float(np.quantile(disp, target))


def load_runs(patterns):
    """-> {run_name: [record]}, one record per (step, scope) from eval_episodes.jsonl."""
    out = {}
    for pat in patterns:
        for d in sorted(glob.glob(os.path.expandvars(os.path.expanduser(pat)))):
            f = os.path.join(d, "eval_episodes.jsonl")
            if not os.path.exists(f):
                continue
            recs = [json.loads(l) for l in open(f) if l.strip()]
            if recs:
                out.setdefault(os.path.basename(d.rstrip("/")), []).extend(recs)
    return out


def final_point(recs):
    """The last eval step present. Duplicate passes over the same step APPEND rather than
    overwrite (the eval.csv corruption mode), so the latest record per step wins."""
    step = max(r["step"] for r in recs)
    last = [r for r in recs if r["step"] == step][-1]
    return step, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the dataset npz r' is derived from")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--runs", nargs="+", required=True, help="glob(s) over run directories")
    ap.add_argument("--target-p", type=float, default=CUBE_P)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    radius, k, H = TASKS[a.task]
    disp = displacements(a.data, a.task)
    p_native = float((disp < radius).mean())
    rp = matched_radius(disp, a.target_p)
    print(f"\n{a.task}: native radius {radius:.4f} -> P(goal satisfied at 1-2 x H) = "
          f"{p_native:.3f}   (cube {a.target_p:.3f})")
    print(f"{a.task}: matched radius r' = {rp:.4f} "
          f"({rp / radius:.2f} x native), P = {float((disp < rp).mean()):.3f}")
    if rp >= radius:
        print("  NOTE r' >= the native radius: this env is already at least as discriminating "
              "as cube, so E5 cannot be hiding an effect here.")

    runs = load_runs(a.runs)
    if not runs:
        raise SystemExit(f"no eval_episodes.jsonl under {a.runs} -- E0.2 logging must have "
                         f"been on for these runs (it needs a goal-in-obs layout)")

    rows = []
    print(f"\n{'run':<44}{'step':>9}{'native':>10}{'@radius':>10}{'@r-prime':>11}{'n':>5}")
    print("-" * 90)
    for name in sorted(runs):
        step, rec = final_point(runs[name])
        dmin = np.asarray(rec["goal_dist_min"], float)
        row = dict(run=name, step=int(step), n_episodes=len(dmin),
                   # The env's own flag, i.e. the number eval.csv reports. It can differ from
                   # `at_radius` because a success rule may AND extra predicates onto the
                   # distance test (PickCube requires a static robot; PushCube requires the
                   # cube still on the table).
                   success_native=float(np.mean(rec["success"])),
                   success_at_radius=float(np.mean(dmin < radius)),
                   success_at_rprime=float(np.mean(dmin < rp)))
        rows.append(row)
        print(f"{name:<44}{step:>9}{row['success_native']:>10.3f}"
              f"{row['success_at_radius']:>10.3f}{row['success_at_rprime']:>11.3f}"
              f"{row['n_episodes']:>5}")

    print("\nPre-registered reading: if the biGCIQL-GCIQL separation appears only in the "
          "@r-prime column, C5 (env discriminability) was hiding a real effect and the paper "
          "should report matched-radius numbers. If the cells stay flat at r' too, parity on "
          "this env is genuine.")
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(dict(task=a.task, native_radius=radius, p_native=p_native,
                           target_p=a.target_p, r_prime=rp, eval_horizon=H, rows=rows),
                      f, indent=2)
        print(f"[e5] wrote {a.json_out}")


if __name__ == "__main__":
    main()
