"""Aggregate analysis/transfer_table/routes.jsonl into the sec:reward-transfer table.

One protocol for both columns: the same 1M checkpoint, 10 episodes per task, all of the
environment's 5 tasks, scored by scripts/reeval_checkpoint.py. Route A conditions on the
env's goal frame (each critic's own map_eval_cond); route B infers the task index from
10,000 reward-relabelled transitions and rolls the policy on it.
"""
import json, collections, sys
import numpy as np

ROWS = [("cube_single", "cube-single"), ("cube_double", "cube-double"),
        ("scene_play", "scene"), ("puzzle_3x3", "puzzle-3x3")]
COLS = [("fb_tdj", "FB"), ("hilp_tdjc", "HILP"), ("rldp", "RLDP"), ("density_fb_z", "Density-FB")]

ENV2DOM = {"cube-single-play-v0": "cube_single", "cube-double-play-v0": "cube_double",
           "scene-play-v0": "scene_play", "puzzle-3x3-play-v0": "puzzle_3x3"}


def load(a="analysis/transfer_table/routeA.jsonl", b="analysis/transfer_table/routeB.jsonl"):
    """Route A rows come from scripts/reeval_checkpoint (tagged); route B rows from
    scripts/relabel_rescore (per-task reward relabelling, keyed by env_name)."""
    per = collections.defaultdict(list)          # (route, dom, critic, seed) -> task successes
    for line in open(a):
        r = json.loads(line)
        if not r.get("tag", "").startswith("routeA"):
            continue
        per[("A", r["domain"], r["critic"], r["seed"])].append(float(r["success"]))
    for line in open(b):
        r = json.loads(line)
        per[("B", ENV2DOM[r["env_name"]], r["critic"], r["seed"])].append(float(r["success"]))
    return per

def cell(per, route, dom, critic):
    seeds = [np.mean(v) for (rt, d, c, s), v in per.items()
             if rt == route and d == dom and c == critic]
    if not seeds:
        return None
    return 100 * np.mean(seeds), 100 * np.std(seeds), len(seeds)

if __name__ == "__main__":
    per = load()
    for route, name in (("A", "goal conditioning"), ("B", "reward inference")):
        print(f"\n=== route {route}: {name} ===")
        print(f"{'task':14s}" + "".join(f"{n:>22s}" for _, n in COLS))
        for dom, label in ROWS:
            line = f"{label:14s}"
            for critic, _ in COLS:
                c = cell(per, route, dom, critic)
                line += f"{'--':>22s}" if c is None else f"{c[0]:16.1f}±{c[1]:.1f}({c[2]})"
            print(line)
    print("\nlatex rows (reward inference / goal conditioning per method):")
    for dom, label in ROWS:
        cells = []
        for critic, _ in COLS:
            b, a = cell(per, "B", dom, critic), cell(per, "A", dom, critic)
            cells.append(("---" if b is None else f"{b[0]:.1f}\\pmse{{{b[1]:.1f}}}") + " & " +
                         ("---" if a is None else f"{a[0]:.1f}\\pmse{{{a[1]:.1f}}}"))
        print(f"\\texttt{{{label}}} & " + " & ".join(cells) + r" \\")
