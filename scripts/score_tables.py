"""Score the OGBench and ExoRL cells of the paper's benchmark table from evaluation logs.

OGBench (success rate, %). Per run: the task-mean success at each evaluation step, averaged
over the last three evaluations (800k, 900k, 1M). Per cell: mean over seeds 0-4, error =
sample standard deviation / sqrt(n). Which log a method is scored from:

    FB, HILP, RLDP      eval_relabel.csv   task vector inferred from 10k reward samples
    Density-FB          eval.csv           goal-conditioned (the table's Density-FB column)
    Density-FB (z)      eval_relabel.csv   task vector inferred from reward samples (in-text)

ExoRL (DMC returns). Per seed: episode return averaged over the domain's tasks, read from the
jsonl written by scripts/exorl_paper_eval.py (params_1000000.pkl, 10 episodes, 50k reward
samples). Per cell: mean over seeds 0-4, error = population standard deviation / sqrt(n).

GCIQL/CRL/QRL (OGBench) and FB/HILP/RLDP (ExoRL) are published numbers and are only read
from the tex, never recomputed.

    python scripts/score_tables.py --exp hf/exp                       # print every cell
    python scripts/score_tables.py --exp hf/exp --tex experiments.tex # also diff against the paper
    python scripts/score_tables.py --exp hf/exp --tex experiments.tex --write
                                   # rewrite the table body: cells, benchmark means, delta column

`--exp` is the `exp/` tree of the HF repo amsks/factored-fb-checkpoints (branch iclr2027):
    huggingface-cli download amsks/factored-fb-checkpoints --revision iclr2027 \
        --include 'exp/**/eval*.csv' 'exp/paper_eval_*.jsonl' --local-dir hf
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import statistics
import sys

OGB_ENVS = {"cube-single": "cube-single-play-v0", "cube-double": "cube-double-play-v0",
            "scene": "scene-play-v0"}
SEEDS = range(5)

# method -> (critic name in the run directory, log file, subdirectories of --exp searched in
# order). FB and HILP seeds 0-2 sit at the top of exp/, seeds 3-4 under tdjtab/. RLDP is the
# ortho_coef=1 arm, which lives only under ortho1/ (exp/<env>_rldp_* is a different arm).
OGB_METHODS = {
    "FB": ("fb_tdj", "eval_relabel", ("", "tdjtab")),
    "HILP": ("hilp_tdjc", "eval_relabel", ("", "tdjtab")),
    "RLDP": ("rldp", "eval_relabel", ("ortho1",)),
    "Density-FB": ("density_fb", "eval", ("", "tdjtab")),
    "Density-FB (z)": ("density_fb_z", "eval_relabel", ("", "tdjtab")),
}

# ExoRL table row -> (jsonl under --exp, domain). Each row is the recipe reported in the paper
# (see the ExoRL hyperparameter appendix).
EXORL_CELLS = {
    "walker": ("paper_eval_exorl_sqt.jsonl", "dmc-walker"),
    "quadruped": ("paper_eval_exorl_kernels_quadruped_vu.jsonl", "dmc-quadruped"),
    "point-mass": ("paper_eval_exorl_1mrecipe.jsonl", "dmc-point-mass-maze"),
    "cheetah": ("paper_eval_exorl_cheetah3_velblock.jsonl", "dmc-cheetah"),
}

# Table layout. tab:benchmarks has columns FB, HILP, RLDP, GCIQL, CRL, QRL, Density-FB;
# tab:exorl has FB, HILP, RLDP, Density-FB. In both, FB/HILP/RLDP are the first three cells
# and Density-FB the last, followed by the delta column. Each table is a list of blocks.
TABLES = [["OGBench", "MetaWorld", "ManiSkill"], ["ExoRL"]]
COLUMN = {"FB": 0, "HILP": 1, "RLDP": 2, "Density-FB": -1}
BLOCKS = {"OGBench": ["cube-single", "cube-double", "scene"],
          "MetaWorld": ["push", "bin-picking", "box-close"],
          "ManiSkill": ["push-cube", "pick-cube", "stack-cube", "poke-cube", "lift-peg"],
          "ExoRL": ["walker", "quadruped", "point-mass", "cheetah"]}
BLOCK_TITLES = {"OGBench": r"\textit{OGBench (success rate, \%)}",
                "MetaWorld": r"\textit{MetaWorld (success rate, \%)}",
                "ManiSkill": r"\textit{ManiSkill (success rate, \%)}",
                "ExoRL": r"\textit{ExoRL (DMC returns)}"}


# ----------------------------------------------------------------------------- scoring

def run_score(path):
    """Mean over the last three evaluation steps of the task-mean success, in %."""
    by_step = collections.defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_step[int(float(row["step"]))].append(float(row["success"]))
    steps = sorted(by_step)[-3:]
    return 100 * statistics.mean(statistics.mean(by_step[s]) for s in steps), steps[-1]


def find_log(exp, env, critic, log, subdirs, seed):
    run = f"{OGB_ENVS[env]}_{critic}_flowbc_seed_{seed}"
    for sub in subdirs:
        p = os.path.join(exp, sub, run, f"{log}.csv")
        if os.path.exists(p):
            return p
    return None


def cell(values, pop=False):
    n = len(values)
    if n == 0:
        return None
    sd = statistics.pstdev(values) if pop else (statistics.stdev(values) if n > 1 else 0.0)
    return statistics.mean(values), sd / math.sqrt(n), n


def score_ogbench(exp):
    out = {}
    for method, (critic, log, subdirs) in OGB_METHODS.items():
        for env in OGB_ENVS:
            per_seed = []
            for s in SEEDS:
                p = find_log(exp, env, critic, log, subdirs, s)
                if p is not None:
                    per_seed.append(run_score(p))
            out[method, env] = (cell([v for v, _ in per_seed]), per_seed)
    return out


def score_exorl(exp):
    out = {}
    for row, (jsonl, domain) in EXORL_CELLS.items():
        per_seed = collections.defaultdict(dict)
        with open(os.path.join(exp, jsonl)) as f:
            for line in f:
                r = json.loads(line)
                if r["domain"] == domain:
                    per_seed[r["seed"]][r["exorl_task"]] = r["episode_return"]
        means = [statistics.mean(t.values()) for _, t in sorted(per_seed.items())]
        out["Density-FB", row] = (cell(means, pop=True), [(m, None) for m in means])
    return out


# ----------------------------------------------------------------------------- tex table

CELL_RE = re.compile(r"(?:\\textbf\{)?([\d.]+)\}?\\pmse\{([\d.]+)\}")
ROW_RE = re.compile(r"\s*\\texttt\{([\w-]+)\}\s*&(.*)\\\\")


def parse_rows(tex):
    """{task: [(mean, se) | None, ...]} for every task row, delta column dropped."""
    rows = {}
    for line in tex.splitlines():
        m = ROW_RE.match(line)
        if not m or m.group(1) not in sum(BLOCKS.values(), []):
            continue
        vals = []
        for c in [c.strip() for c in m.group(2).split("&")][:-1]:
            mm = CELL_RE.search(c)
            vals.append(None if c == "--" or mm is None else (float(mm.group(1)), float(mm.group(2))))
        rows[m.group(1)] = vals
    return rows


def delta(v):
    """Density-FB's relative improvement (%) over the best baseline in the row."""
    best = max(x[0] for x in v[:-1] if x)
    return 100 * (v[-1][0] - best) / best


def block_mean(rows, tasks):
    """Per column: mean over the block's tasks, SE = sqrt(sum se^2) / n_tasks."""
    agg = []
    for j in range(len(rows[tasks[0]])):
        if rows[tasks[0]][j] is None:
            agg.append(None)
            continue
        mu = sum(rows[t][j][0] for t in tasks) / len(tasks)
        se = math.sqrt(sum(rows[t][j][1] ** 2 for t in tasks)) / len(tasks)
        agg.append((round(mu, 1), round(se, 1)))
    return agg


def fmt_row(label, v, indent):
    """One tex row; bold marks the best mean and every mean within its SE."""
    red, gray, green, cap = (210, 20, 20), (100, 100, 100), (0, 150, 40), 25.0
    best, best_se = max((x for x in v if x), key=lambda x: x[0])

    def fmt(x):
        if x is None:
            return "--"
        m = "%.1f" % x[0]
        return (r"\textbf{%s}" % m if x[0] >= best - best_se - 1e-9 else m) + r"\pmse{%.1f}" % x[1]

    d = delta(v)
    t = max(-1.0, min(1.0, d / cap))
    lo, hi = (gray, green) if t >= 0 else (gray, red)
    rgb = ",".join(str(round(lo[i] + (hi[i] - lo[i]) * abs(t))) for i in range(3))
    dcell = r"\textcolor[RGB]{%s}{(%s%.1f\%%)}" % (rgb, "$+$" if d >= 0 else "$-$", abs(d))
    return indent + "%-38s & " % label + " & ".join(fmt(x) for x in v) + " & " + dcell + r" \\"


def write_tables(tex_path, rows):
    """Rewrite each table's body, from its first block title down to its \\bottomrule."""
    lines = open(tex_path).read().split("\n")
    for blocks in TABLES:
        s = next(i for i, l in enumerate(lines) if l.strip().startswith(BLOCK_TITLES[blocks[0]]))
        e = next(i for i, l in enumerate(lines) if i > s and l.strip() == r"\bottomrule")
        indent = lines[s][:len(lines[s]) - len(lines[s].lstrip())]
        body = []
        for block in blocks:
            tasks = BLOCKS[block]
            body += [indent + r"\midrule", fmt_row(BLOCK_TITLES[block], block_mean(rows, tasks), indent),
                     indent + r"\midrule"] + [fmt_row(r"\texttt{%s}" % t, rows[t], indent) for t in tasks]
        lines[s:e] = body[1:]
    open(tex_path, "w").write("\n".join(lines))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="exp/ tree holding the run directories and jsonl files")
    ap.add_argument("--tex", help="experiments.tex holding the benchmark table")
    ap.add_argument("--write", action="store_true", help="rewrite the table body in --tex")
    args = ap.parse_args()

    scores = {**score_ogbench(args.exp), **score_exorl(args.exp)}
    for (method, task), (c, per_seed) in scores.items():
        head = f"{method:15s} {task:12s}"
        if c is None:
            print(f"{head} no runs found")
            continue
        seeds = [round(v, 1) for v, _ in per_seed]
        print(f"{head} {c[0]:6.1f} +- {c[1]:4.1f}  n={c[2]}  per-seed {seeds}")

    if not args.tex:
        return 0
    rows = parse_rows(open(args.tex).read())
    mismatches = 0
    for (method, task), (c, _) in scores.items():
        if method == "Density-FB (z)" or c is None:
            continue                                  # in-text numbers, not table cells
        paper = rows[task][COLUMN[method]]
        ours = (round(c[0], 1), round(c[1], 1))
        ok = paper is not None and ours == paper
        mismatches += not ok
        print(f"{'ok  ' if ok else 'DIFF'} {method:10s} {task:12s} computed {ours}  paper {paper}")
        rows[task][COLUMN[method]] = ours
    if args.write:
        write_tables(args.tex, rows)
        print(f"rewrote the table bodies in {args.tex}")
    return 1 if mismatches and not args.write else 0


if __name__ == "__main__":
    sys.exit(main())
