"""scripts/make_results_table.py — the running results table: benchmark × data_type × critic × actor.

Each training run writes <root>/<env>_<critic>_<actor>_seed_<n>/eval.csv. A run's score is the
mean success over its last 3 eval points (less optimistic than max-over-curve); a cell is the
mean ± sd of that over seeds. `data_type` is not in the run name, so it comes from which ROOT
the run lives under — one save-dir per regime, passed as --root <data_type>=<dir> (repeatable).

    python scripts/make_results_table.py \
        --root expert=$SCRATCH/factored-fb-exp \
        --root play-noise=$SCRATCH/factored-fb-exp-play-noise \
        --root play-resample=$SCRATCH/factored-fb-exp-play-resample \
        --out results/table.tex --csv results/table.csv

Emits LaTeX (booktabs) + a flat CSV. bcflow is the pure-BC control and is shown as its own
"BC" column, since it answers the question the rest of the table is really asking: does any
critic×actor beat cloning the data?
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

BENCH = {"mw": "Metaworld", "ms": "ManiSkill", "b2d": "Box2D", "p2d": "Box2D"}
# Column order. bcflow is pulled out as the BC baseline; the rest are critic × {flowbc, ddpgbc}.
CRITICS = ["fb", "gciql", "iql_bilinear", "crl"]
ACTORS = ["flowbc", "ddpgbc"]
CRITIC_TeX = {"fb": "FB", "gciql": "GCIQL", "iql_bilinear": "biGCIQL", "crl": "CRL"}
ACTOR_TeX = {"flowbc": "Flow", "ddpgbc": "DDPG"}


def score_run(eval_csv):
    """Mean success over the last 3 eval points, or None if unreadable/empty.

    OGBench eval.csv files carry one row per TASK per step (impls/main.py logs each task_id
    separately; eval_aggregate is off by default), so rows are first grouped by step and
    averaged across tasks; a csv with one row per step is unchanged by that."""
    try:
        rows = list(csv.DictReader(open(eval_csv)))
    except OSError:
        return None
    pts = defaultdict(list)
    for r in rows:
        try:
            pts[int(float(r["step"]))].append(float(r["success"]))
        except (KeyError, ValueError):
            pass
    if not pts:
        return None
    return st.mean(st.mean(pts[s]) for s in sorted(pts)[-3:])


# Critic/actor names can themselves contain underscores (iql_bilinear, sf_feature, bcflow),
# so a positional split is wrong. Match the known names off the end instead. Longest first so
# e.g. fb_forward_bilinear wins over fb.
# Names are the --critic values a run dir is written under: the CRITICS keys and the
# config-only CRITIC_ALIASES keys of impls/agents/__init__.py (kept in sync by hand; importing
# that module would pull in jax and every critic class just to build a table).
KNOWN_CRITICS = sorted(
    ["fb", "gciql", "iql_bilinear", "crl", "isfb", "simple_sf", "sf_feature",
     "centered_bilinear", "fb_forward_bilinear",
     # impls/agents/__init__.py CRITICS
     "gcivl", "hiql", "qrl", "density_fb", "psm", "rldp", "psm_density", "rldp_density",
     "hilp_density", "hilp",
     # impls/agents/__init__.py CRITIC_ALIASES
     "density_fb_pixels", "density_fb_z", "rldp_density_gk", "rldp_density_gk_oc",
     "density_fb_z_state", "hilp_density_gk", "hilp_density_gk_oc", "hilp_gc", "hilp_gc_oc",
     "rldp_gc", "rldp_gc_oc", "psm_repair", "psm_density_repair", "rldp_oc", "rldp_density_oc",
     "psm_oc", "psm_density_oc", "hilp_oc", "hilp_density_oc", "hilp_cube", "fb_oc",
     "hilp_tdjc", "hilp_density_tdjc", "density_fb_state", "hilp_density_cube", "fb_ref",
     "fb_tdj", "rldp_portalign", "rldp_official", "rldp_orthoa", "rldp_orthob",
     "rldp_density_official", "rldp_tdj", "hilp_tdj", "rldp_density_tdj", "hilp_density_tdj"],
    key=len, reverse=True)
KNOWN_ACTORS = sorted(["flowbc", "ddpgbc", "bcflow", "awr", "fql"], key=len, reverse=True)


def parse_run(name):
    """<env>_<critic>_<actor>_seed_<n> -> (env, critic, actor, seed), matching critic/actor
    against the known registries so underscore-bearing names (iql_bilinear) parse correctly."""
    if "_seed_" not in name:
        return None
    head, seed = name.rsplit("_seed_", 1)
    actor = next((a for a in KNOWN_ACTORS if head.endswith("_" + a)), None)
    if actor is None:
        return None
    rest = head[: -(len(actor) + 1)]
    critic = next((c for c in KNOWN_CRITICS if rest.endswith("_" + c)), None)
    if critic is None:
        return None
    env = rest[: -(len(critic) + 1)]
    try:
        return env, critic, actor, int(seed)
    except ValueError:
        return None


def collect(roots):
    """roots: dict data_type -> dir. -> cells[(bench, env, dtype, critic, actor)] = [scores]."""
    cells = defaultdict(list)
    for dtype, root in roots.items():
        for ec in glob.glob(os.path.join(root, "*", "eval.csv")):
            p = parse_run(os.path.basename(os.path.dirname(ec)))
            if not p:
                continue
            env, critic, actor, _seed = p
            s = score_run(ec)
            if s is None:
                continue
            bench = BENCH.get(env.split("-", 1)[0], "?")
            cells[(bench, env, dtype, critic, actor)].append(s)
    return cells


def fmt(vals):
    if not vals:
        return "--"
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f}\\,\\pm\\,{sd:.2f}" if sd else f"{m:.2f}"


def emit_latex(cells, roots):
    dtypes = list(roots)
    # rows: one per (bench, env, data_type) that has data. columns: BC | critic×actor.
    keys = sorted({(b, e, d) for (b, e, d, _c, _a) in cells})
    ncol = 1 + len(CRITICS) * len(ACTORS)
    L = []
    L.append("% Auto-generated by scripts/make_results_table.py — do not edit by hand.")
    L.append("\\begin{table}[t]\\centering\\small")
    L.append("\\caption{Success rate (mean over last 3 evals, $\\pm$ sd over seeds). "
             "\\textbf{BC} is the pure behaviour-cloning control (flow actor, critic value "
             "term off). A critic earns its keep only where it beats BC.}")
    colspec = "lll " + "c" * ncol
    L.append("\\begin{tabular}{" + colspec + "}")
    L.append("\\toprule")
    head2 = ["Benchmark", "Env", "Data", "BC"]
    for c in CRITICS:
        for a in ACTORS:
            head2.append(f"{CRITIC_TeX[c]}$\\times${ACTOR_TeX[a]}")
    L.append(" & ".join(head2) + " \\\\")
    L.append("\\midrule")
    last_bench = None
    for (bench, env, dtype) in keys:
        bc = cells.get((bench, env, dtype, "fb", "bcflow")) or \
            next((cells[(bench, env, dtype, c, "bcflow")] for c in CRITICS
                  if cells.get((bench, env, dtype, c, "bcflow"))), [])
        row = [bench if bench != last_bench else "",
               env.replace("_", "\\_"), dtype, f"${fmt(bc)}$"]
        last_bench = bench
        for c in CRITICS:
            for a in ACTORS:
                row.append(f"${fmt(cells.get((bench, env, dtype, c, a), []))}$")
        L.append(" & ".join(row) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L)


def emit_csv(cells, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "env", "data_type", "critic", "actor", "n_seeds", "mean", "sd"])
        for k in sorted(cells):
            v = cells[k]
            w.writerow([*k, len(v), f"{st.mean(v):.4f}",
                        f"{st.stdev(v):.4f}" if len(v) > 1 else "0"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[], metavar="DTYPE=DIR",
                    help="repeatable; e.g. --root expert=/path --root play-noise=/path")
    ap.add_argument("--out", default="results/table.tex")
    ap.add_argument("--csv", default="results/table.csv")
    args = ap.parse_args()
    roots = {}
    for spec in args.root:
        dt, _, d = spec.partition("=")
        roots[dt] = d
    if not roots:
        ap.error("pass at least one --root DTYPE=DIR")
    cells = collect(roots)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(emit_latex(cells, roots) + "\n")
    emit_csv(cells, args.csv)
    n = sum(len(v) for v in cells.values())
    print(f"[table] {len(cells)} cells, {n} runs -> {args.out} + {args.csv}")


if __name__ == "__main__":
    main()
