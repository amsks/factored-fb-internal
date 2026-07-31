"""Aggregate metric-probe outputs: cross-method RSA (P6) + scalar table.

RSA = Spearman correlation between the upper triangles of two runs' distance matrices on
the shared probe states. Feed it the .npz files written by eval/metric_probe.py:

  python -m eval.probe_report --out exp/metric-probe/report \
      exp/metric-probe/probes/*.npz
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np

from eval.metric_probe import spearman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(itertools.chain.from_iterable(glob.glob(p) for p in args.npz))

    tris = {}
    for path in files:
        run = os.path.splitext(os.path.basename(path))[0]
        with np.load(path) as f:
            for key in f.files:
                if key.startswith("mat_"):
                    M = np.asarray(f[key], np.float64)
                    iu = np.triu_indices(M.shape[0], k=1)
                    tris[f"{run}:{key[4:]}"] = M[iu]

    names = sorted(tris)
    rsa = {a: {b: (1.0 if a == b else spearman(tris[a], tris[b])) for b in names} for a in names}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + "_rsa.json", "w") as f:
        json.dump(rsa, f, indent=2)
    # compact console table
    w = max(len(n) for n in names) + 1
    print(" " * w + "  ".join(f"{i:>5d}" for i in range(len(names))))
    for i, a in enumerate(names):
        print(f"{a:<{w}}" + "  ".join(f"{rsa[a][b]:+.2f}" for b in names) + f"   [{i}] {a}")
    print(f"\nRSA matrix -> {args.out}_rsa.json  ({len(names)} constructions)")


if __name__ == "__main__":
    main()
