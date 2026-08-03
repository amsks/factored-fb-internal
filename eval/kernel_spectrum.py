"""Ground-truth spectral analysis of the discounted-reachability kernel (no learning).

Grounds the surviving mechanism account (docs/2026-08-03-bilinear-mechanism-hypothesis.md)
in an intrinsic property of the environment: if the bilinear critic works at rank d because
the family of optimal goal-conditioned value functions {V*(., g)} spans a ~d-dimensional
function space, then the eps-rank of the TRUE kernel K = gamma^{d*} -- estimated from the
dataset alone, with no critic -- must match the measured success-saturation dimension
(d-sweep: saturation between d=15 and d=20 on cube-single).

Estimate of d*: shortest paths on a graph over subsampled dataset states with
  * temporal edges between successive subsampled states of the same trajectory
    (weight = timestep gap: the dataset demonstrates this path), and
  * kNN shortcut edges between observation-space near neighbours across trajectories
    (weight = 1: states this close are dynamically interchangeable at the resolution of
    one step; standard graph-quasimetric practice).
Then K = gamma^{D}, SVD, and report the spectrum: dims to reach 90/95/99% energy,
effective rank exp(H(sigma^2)), and the same for the (1-gamma)-scaled value matrix
V* = (K-1)/(1-gamma).

    python -m eval.kernel_spectrum --data ~/.ogbench/data/cube-single-play-v0-val.npz \
        --out /tmp/kernel_spectrum.json
"""
import argparse
import json
import os

import numpy as np


def build_graph_distances(obs, terminals, n_anchors=2500, stride=5, knn=10,
                          radius_q=0.9, seed=0):
    """Graph nodes = every `stride`-th dataset state (dense enough that the strict
    shortcut radius keeps the graph connected). Temporal edges between consecutive nodes
    of one trajectory carry their true timestep gap; kNN shortcut edges connect nodes
    within the radius_q quantile of the dataset's `stride`-step displacement and carry
    weight `stride` — the granularity of the node set, so shortcuts cannot beat a
    demonstrated path at the same resolution. Distances are computed from `n_anchors`
    random source nodes and returned as the anchors x anchors submatrix."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    (term_locs,) = np.nonzero(terminals > 0)
    starts = np.concatenate([[0], term_locs[:-1] + 1])
    traj_id = np.zeros(len(obs), np.int64)
    for k, (a, b) in enumerate(zip(starts, term_locs)):
        traj_id[a:b + 1] = k

    idx = np.arange(0, len(obs), stride)
    sub, sub_traj = obs[idx], traj_id[idx]
    n = len(sub)

    rows, cols, w = [], [], []
    same = sub_traj[:-1] == sub_traj[1:]
    for a in np.nonzero(same)[0]:                      # temporal edges (demonstrated paths)
        rows.append(a); cols.append(a + 1); w.append(float(idx[a + 1] - idx[a]))
    # shortcut radius from the stride-step displacement distribution (within-trajectory)
    ok = traj_id[:-stride] == traj_id[stride:]
    hop = np.linalg.norm(obs[stride:] - obs[:-stride], axis=1)[ok]
    radius = float(np.quantile(hop, radius_q))
    tree = cKDTree(sub)
    dists, nbrs = tree.query(sub, k=knn + 1)
    for i in range(n):
        for dist, j in zip(dists[i, 1:], nbrs[i, 1:]):
            if dist <= radius:
                rows.append(i); cols.append(int(j)); w.append(float(stride))

    anchors = np.sort(rng.choice(n, size=min(n_anchors, n), replace=False))
    g = coo_matrix((np.asarray(w), (np.asarray(rows), np.asarray(cols))), shape=(n, n))
    D = dijkstra(g.tocsr(), directed=False, indices=anchors)[:, anchors]
    return D, radius


def spectrum_report(M):
    s = np.linalg.svd(M, compute_uv=False)
    e = s ** 2
    c = np.cumsum(e) / e.sum()
    p = e / e.sum()
    p = p[p > 0]
    return {
        "dims_90": int(np.searchsorted(c, 0.90) + 1),
        "dims_95": int(np.searchsorted(c, 0.95) + 1),
        "dims_99": int(np.searchsorted(c, 0.99) + 1),
        "eff_rank": float(np.exp(-(p * np.log(p)).sum())),
        "top_sv_frac": [float(x) for x in (e[:30] / e.sum())],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n_anchors", type=int, default=2500)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--radius_q", type=float, default=0.9,
                    help="shortcut radius = this quantile of one-step obs displacement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with np.load(os.path.expanduser(args.data)) as f:
        obs = np.asarray(f["observations"], np.float32)
        terminals = np.asarray(f["terminals"])

    D, radius = build_graph_distances(obs, terminals, args.n_anchors, args.stride,
                                      args.knn, radius_q=args.radius_q, seed=args.seed)
    finite = np.isfinite(D)
    Dc = np.where(finite, D, np.nanmax(D[finite]))     # unreachable -> max observed distance
    K = args.gamma ** Dc
    V = (K - 1.0) / (1.0 - args.gamma)

    report = {
        "gamma": args.gamma, "n_nodes": int(D.shape[0]), "knn_radius": float(radius),
        "frac_reachable": float(finite.mean()),
        "d_star_median": float(np.median(Dc)), "d_star_p90": float(np.quantile(Dc, 0.9)),
        "kernel_K": spectrum_report(K),
        "value_V": spectrum_report(V),
        # centered value matrix: removes the rank-1 offset a critic's bias absorbs
        "value_V_centered": spectrum_report(V - V.mean()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "top_sv_frac"}, indent=2,
                     default=str))


if __name__ == "__main__":
    main()
