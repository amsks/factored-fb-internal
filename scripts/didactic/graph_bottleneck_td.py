"""Nine states connected in a graph, with one state every path has to pass.

Each state has two actions and each action leads to another state, so the graph
is a network rather than a chain. States 0-3 are joined to each other, states
5-8 are joined to each other, and state 4 is the only state through which the
first group reaches the second. The goal is state 8; the decision the diagnostic
reads is at state 7, where one action reaches the goal and the other does not.

`leak` is how often a transition ignores the drawn edges and lands on a uniformly
random state in the other group. At leak 0 every path passes through state 4. As
leak grows the two groups mix directly and state 4 stops being special, so it is
the knob that turns the bottleneck off.
"""

import argparse, hashlib, json, time
from itertools import product
from pathlib import Path

import numpy as np
import jax, optax

import bottleneck_family_td as bf

EDGES = np.array([
    [1, 2],   # 0
    [3, 0],   # 1
    [3, 1],   # 2
    [4, 2],   # 3  the only state that reaches 4
    [5, 3],   # 4  the only state that reaches the second group
    [6, 7],   # 5
    [7, 5],   # 6
    [8, 6],   # 7  the decision: reach the goal, or stay in the group
    [5, 8],   # 8  the goal
])
GROUP_A = [0, 1, 2, 3]
GROUP_B = [5, 6, 7, 8]
CUT, GOAL, DECISION = 4, 8, 7


def build_graph(leak=0.0):
    """(n, 2, n) transition kernel. Rows are distributions over next states."""
    n = len(EDGES)
    p = np.eye(n)[EDGES]
    if leak > 0:
        # Only the two groups leak. State 4 keeps its edges, so the knob removes
        # its special position rather than rewiring it.
        other = np.zeros((n, n))
        mix = np.zeros(n)
        for s in range(n):
            tgt = GROUP_B if s in GROUP_A else (GROUP_A if s in GROUP_B else [])
            if tgt:
                other[s, tgt] = 1.0 / len(tgt)
                mix[s] = leak
        w = mix[:, None, None]
        p = (1 - w) * p + w * other[:, None, :]
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("analysis/didactic_graph/main"))
    ap.add_argument("--leaks", nargs="+", type=float, default=[0, .02, .05, .1, .2, .4])
    ap.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4, 6])
    ap.add_argument("--methods", nargs="+", default=["fb", "density"])
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.001, 0.01])
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--sphere", action="store_true",
                    help="normalize backward rows to norm sqrt(d); fb only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "config.json").exists():
        raise FileExistsError("output has a run")
    (a.out / "config.json").write_text(json.dumps(
        {**vars(a), "out": str(a.out), "numpy": np.__version__, "jax": jax.__version__,
         "optax": optax.__version__, "edges": EDGES.tolist(),
         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))

    rows, t0 = [], time.monotonic()
    for leak, d, m in product(a.leaks, a.ranks, a.methods):
        p = build_graph(leak)
        g = bf.run_kernel(p, CUT, GOAL, DECISION, d, m, a.gamma,
                          a.seeds, a.rates, a.steps, a.tau, a.sphere)
        g = [dict(r, leak=leak) for r in g]
        rows += g
        print(json.dumps({"leak": leak, "d": d, "method": m,
                          "gap_ratio": round(float(np.mean(
                              [r["gap_ratio_near"] for r in g])), 4),
                          "inverted": sum(r["inverted_near"] for r in g),
                          "elapsed": round(time.monotonic() - t0, 1)}), flush=True)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {a.out/'summary.json'}  ({len(rows)} runs)")
