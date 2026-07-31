"""Build the fixed probe set for the metric probes (spec §4) from an OGBench val npz.

Deterministic (seeded); every model is probed on the identical arrays. Produces:
  states      (n_states, D)  — subsample for the pairwise distance matrix (P3-P7)
  pair_s/pair_g (n_pairs, D), pair_dt (n_pairs,) — same-trajectory pairs with time gap (P1)

Usage:
  python -m eval.probe_set --data ~/.ogbench/data/cube-single-play-v0-val.npz \
      --out exp/metric-probe/probe_set.npz
"""
import argparse
import os

import numpy as np


def build(data_path, n_states=1024, n_pairs=50000, max_dt=300, seed=0):
    with np.load(os.path.expanduser(data_path)) as f:
        obs = np.asarray(f["observations"], np.float32)
        terminals = np.asarray(f["terminals"])
    (term_locs,) = np.nonzero(terminals > 0)
    init_locs = np.concatenate([[0], term_locs[:-1] + 1])
    rng = np.random.default_rng(seed)

    states = obs[rng.choice(len(obs), size=n_states, replace=False)]

    # same-trajectory (s_t, s_{t+dt}) pairs
    traj = rng.integers(len(term_locs), size=n_pairs)
    lo, hi = init_locs[traj], term_locs[traj]
    t = lo + rng.integers(0, np.maximum(hi - lo, 1))
    dt = 1 + rng.integers(0, np.minimum(np.maximum(hi - t, 1), max_dt))
    t2 = np.minimum(t + dt, hi)
    return dict(states=states, pair_s=obs[t], pair_g=obs[t2],
                pair_dt=(t2 - t).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_states", type=int, default=1024)
    ap.add_argument("--n_pairs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **build(args.data, args.n_states, args.n_pairs, seed=args.seed))
    print(f"probe set -> {args.out}")


if __name__ == "__main__":
    main()
