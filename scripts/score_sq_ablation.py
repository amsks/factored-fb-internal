"""Score the Density-FB loss ablation (scripts/launch_sq_ablation.sh): squared error vs
cross-entropy on the same softmax critic, for the goal-conditioned model (eval.csv) and the
reward-inference model (eval_relabel.csv), on cube-single, cube-double and scene.

Per run: task-mean success averaged over the evaluations at --steps. Cell: mean +- SE
(sample sd / sqrt(n)) over seeds. Squared-error runs are read from exp/sq_ablation; the
cross-entropy runs are the table runs, read from --ce-exp (the exp/ tree of HF
amsks/factored-fb-checkpoints@iclr2027, e.g. downloaded with
`--include 'exp/*density_fb*/eval*.csv' 'exp/tdjtab/*density_fb*/eval*.csv'`).

    python scripts/score_sq_ablation.py --ce-exp hf/exp --steps 300000 400000 500000 --seeds 0 1 2
    python scripts/score_sq_ablation.py --ce-exp hf/exp --steps 800000 900000 1000000 --seeds 0 1 2 3 4
"""
import argparse
import collections
import csv
import math
import os
import statistics as st

ENVS = ["cube-single", "cube-double", "scene"]
MODELS = [("density_fb", "eval", "goal-conditioned"),
          ("density_fb_z", "eval_relabel", "reward inference")]


def run_score(path, steps):
    """Mean over `steps` of the task-mean success (%), or None if any step is missing."""
    if not os.path.exists(path):
        return None
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        try:
            by[int(float(r["step"]))].append(float(r["success"]))
        except ValueError:
            pass
    vals = [st.mean(by[s]) for s in steps if s in by and not any(math.isnan(x) for x in by[s])]
    return 100 * st.mean(vals) if len(vals) == len(steps) else None


def cell(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    se = st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return st.mean(vals), se, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq-exp", default="exp/sq_ablation")
    ap.add_argument("--ce-exp", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[300000, 400000, 500000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--csv", help="also write per-seed scores here")
    a = ap.parse_args()
    rows = []
    for critic, log, label in MODELS:
        print(f"== {label} ({critic}, {log}.csv)")
        for env in ENVS:
            run = lambda root, s: f"{root}/{env}-play-v0_{critic}_flowbc_seed_{s}/{log}.csv"
            sq = [run_score(run(a.sq_exp, s), a.steps) for s in a.seeds]
            ce = []
            for s in a.seeds:   # table runs: seeds 0-2 at the top of exp/, some under tdjtab/
                v = run_score(run(a.ce_exp, s), a.steps)
                ce.append(v if v is not None else run_score(run(a.ce_exp + "/tdjtab", s), a.steps))
            (ms, ss, ns), (mc, sc, nc) = cell(sq), cell(ce)
            fmt = lambda m, s, n: "missing" if m is None else f"{m:5.1f} +- {s:4.1f} (n={n})"
            print(f"  {env:12s} squared {fmt(ms, ss, ns):22s} cross-entropy {fmt(mc, sc, nc)}")
            for s, vs, vc in zip(a.seeds, sq, ce):
                rows.append({"model": label, "critic": critic, "env": env, "seed": s,
                             "squared": vs, "cross_entropy": vc})
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
