"""Emit the didactic section's table and its numbers, straight from the runs."""
import argparse, collections, json
from pathlib import Path
import numpy as np

ARMS = [("fb", "main2", "FB"), ("fb", "sphere2", "FB, spherical $B$"),
        ("normsq", "main2", "Normalized squared"), ("density", "main2", "Density-FB")]


def load(root, sub):
    g = collections.defaultdict(list)
    for r in json.load(open(root / sub / "summary.json")):
        g[(r["method"], r["d"], r["R"])].append(r)
    return g


def mean(g, key, *k):
    return float(np.mean([r[key] for r in g[k]]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/didactic_family"))
    ap.add_argument("--out", type=Path, default=Path("PAPER/ICLR/didactic_new"))
    ap.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4, 6])
    ap.add_argument("--rooms", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    a = ap.parse_args()
    G = {s: load(a.root, s) for s in ("main2", "sphere2")}

    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"& \multicolumn{5}{c}{states $n$} \\ \cmidrule(lr){2-6}",
             "critic & " + " & ".join(f"${R+3}$" for R in a.rooms) + r" \\",
             r"\midrule"]
    for method, sub, lab in ARMS:
        cells = []
        for R in a.rooms:
            hit = next((d for d in a.ranks
                        if mean(G[sub], "gap_ratio_near", method, d, R) >= 0.9), None)
            cells.append(f"${hit}$" if hit else r"$>6$")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (a.out / "table_rank.tex").write_text("\n".join(lines) + "\n")

    nums = {}
    for method, sub, lab in ARMS:
        nums[lab] = {
            "gap_ratio_rank2": [round(mean(G[sub], "gap_ratio_near", method, 2, R), 3)
                                for R in a.rooms],
            "rel_err_goal_rank2": [round(mean(G[sub], "rel_err_goal", method, 2, R), 2)
                                   for R in a.rooms],
            "l2_residual_rank2": [round(mean(G[sub], "l2_residual", method, 2, R), 4)
                                  for R in a.rooms],
            "inverted_rank2": [sum(r["inverted_near"] for r in G[sub][(method, 2, R)])
                               for R in a.rooms],
            "runs_per_cell": len(G[sub][(method, 2, a.rooms[0])]),
        }
    (a.out / "numbers.json").write_text(json.dumps(nums, indent=2))
    print(json.dumps(nums, indent=2))
    print(f"wrote {a.out/'table_rank.tex'} and {a.out/'numbers.json'}")
