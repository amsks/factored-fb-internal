"""Printed comparison of action-gap recovery by linear and softmax fits on single-bottleneck layouts.

Run: JAX_PLATFORMS=cpu python analysis/lemma_check/design_single_bottleneck.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation"))
sys.path.insert(0, str(REPO / "scripts" / "didactic"))
import bottleneck_family_td as bf     # noqa: E402
import family as fam                  # noqa: E402

GAMMA = 0.8


def build(entry=4, cluster=3, tail=0, merge=False, mutual=False):
    """Return edges (n, 2), bottleneck index, and the A, B and tail states."""
    E = list(range(entry)); b = entry
    A = list(range(b + 1, b + 1 + cluster)); B = list(range(A[-1] + 1, A[-1] + 1 + cluster))
    T = list(range(B[-1] + 1, B[-1] + 1 + tail)) if merge else []
    n = (T[-1] + 1) if T else (B[-1] + 1)
    edges = np.zeros((n, 2), int)
    # Entry chain to b; second action steps back
    for i, s in enumerate(E):
        edges[s, 0] = E[i + 1] if i + 1 < len(E) else b
        edges[s, 1] = E[i - 1] if i > 0 else E[0]
    edges[b, 0], edges[b, 1] = A[0], B[0]
    for C, other in ((A, B), (B, A)):
        for i, s in enumerate(C):
            edges[s, 0] = C[(i + 1) % len(C)]
            if mutual:
                edges[s, 1] = other[0]
            else:
                edges[s, 1] = (T[0] if (merge and T) else C[(i + 2) % len(C)])
    for i, s in enumerate(T):
        edges[s, 0] = T[(i + 1) % len(T)]
        edges[s, 1] = T[(i - 1) % len(T)]
    return edges, b, A, B, T


def exact_ratio(edges):
    n = len(edges)
    kernel = np.eye(n)[edges].reshape(-1, n)
    t = bf.exact(kernel, np.full((n, 2), 0.5), GAMMA)      # (1-gamma) M
    return t / (1 - GAMMA) * n, t                            # density ratio n M, and (1-gamma) M


def linear_fraction(X, b, r, d):
    _, B = fam.svd_fit(X, d)
    P = B @ np.linalg.pinv(B)
    dv = X[2 * b] - X[2 * b + 1]
    true = dv @ r; fit = dv @ P @ r
    return fit / true if abs(true) > 1e-9 else np.nan, true / len(r)


def softmax_fraction(t, b, r, d, seeds=(0, 1, 2)):
    n = t.shape[1]
    vals = []
    for s in seeds:
        F, Bm, P, _ = fam.softmax_fit(t, n, d, s, "ce", steps=6000, lr=1e-2)
        dv_true = t[2 * b] - t[2 * b + 1]; dv_fit = P[2 * b] - P[2 * b + 1]
        vals.append((dv_fit @ r) / (dv_true @ r))
    return float(np.mean(vals))


def report(name, edges, b, A, B, T):
    X, t = exact_ratio(edges); n = len(edges)
    rewards = {"goal(A0)": np.eye(n)[A[0]], "cluster A": np.isin(np.arange(n), A).astype(float)}
    print(f"\n== {name}: n={n} entry={b} clusters {A} | {B} tail {T}")
    for rn, r in rewards.items():
        tg = (X[2 * b] - X[2 * b + 1]) @ r / n
        lin = " ".join(f"d{d}:{linear_fraction(X, b, r, d)[0]:+.2f}" for d in range(2, min(7, n)))
        print(f"  reward {rn:10s} true gap {tg:+.3f} | linear kept: {lin}")
    # Mean fraction kept over unit rewards with a nonzero gap
    units = [np.eye(n)[k] for k in range(n) if abs((X[2 * b] - X[2 * b + 1]) @ np.eye(n)[k] / n) > 0.05]
    for d in (2, 3):
        lin = np.mean([linear_fraction(X, b, r, d)[0] for r in units])
        sm = np.mean([softmax_fraction(t, b, r, d, seeds=(0,)) for r in units])
        print(f"  d={d}: mean fraction kept over {len(units)} unit rewards: linear {lin:+.2f}  softmax(CE) {sm:+.2f}")
    # Share of the gap row inside the top-d span of the other rows
    dv = X[2 * b] - X[2 * b + 1]
    Xo = np.delete(X, [2 * b, 2 * b + 1], axis=0)
    _, _, Vt = np.linalg.svd(Xo, full_matrices=False)
    shares = [float(np.sum((Vt[:d] @ dv) ** 2) / (dv @ dv)) for d in range(1, min(7, n))]
    print("  share of |dv|^2 inside ordinary rows' top-d span, d=1..:", " ".join(f"{s:.2f}" for s in shares))


if __name__ == "__main__":
    for name, kw in [("disjoint 4|3|3", dict(entry=4, cluster=3)),
                     ("disjoint 8|3|3", dict(entry=8, cluster=3)),
                     ("merge 4|3|3 tail 3", dict(entry=4, cluster=3, tail=3, merge=True)),
                     ("merge 4|3|3 tail 6", dict(entry=4, cluster=3, tail=6, merge=True)),
                     ("merge 4|2|2 tail 6", dict(entry=4, cluster=2, tail=6, merge=True)),
                     ("merge 6|2|2 tail 8", dict(entry=6, cluster=2, tail=8, merge=True))]:
        edges, b, A, B, T = build(**kw)
        report(name, edges, b, A, B, T)


def build_passthrough(entry=2, tail=4):
    """Pass-through bottleneck: entry chain to b, b to A or B, both into a shared tail ring."""
    E = list(range(entry)); b = entry; A = [b + 1]; B = [b + 2]
    T = list(range(b + 3, b + 3 + tail)); n = T[-1] + 1
    edges = np.zeros((n, 2), int)
    for i, s in enumerate(E):
        edges[s] = [E[i + 1] if i + 1 < len(E) else b, E[i - 1] if i > 0 else E[0]]
    edges[b] = [A[0], B[0]]
    for s in (A[0], B[0]):
        edges[s] = [T[0], T[1]]
    for i, s in enumerate(T):
        edges[s] = [T[(i + 1) % tail], T[(i + 2) % tail]]
    return edges, b, A, B, T
