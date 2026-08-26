"""Calibrate a `measure_hit_spec` radius by the procedure recorded in env/domains/dmc-walker.yaml:
pick eps so the fraction of RANDOM observation pairs inside the block's L2 ball matches
cube-single's own hit rate under its published spec (19:22:0.4 -> 0.0133).

    $SCRATCH/.venv-jax-fb/bin/python scripts/calibrate_hit_spec.py \
        --data_dir $SCRATCH/fb-data/humenv-S1 --block 0:214 --episodes 200

The radius is then just a quantile of the pairwise-distance distribution, which is why this
is a measurement and not a tuning knob.
"""
import argparse, glob, os
import numpy as np

TARGET = 0.0133   # cube-single 19:22:0.4, the reference hit rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--block", default="0:214", help="a:b over the RAW observation")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=400_000)
    ap.add_argument("--probe", type=float, nargs="*", default=[2.0, 4.0],
                    help="also report the hit rate at these radii")
    args = ap.parse_args()
    a, b = (int(x) for x in args.block.split(":"))

    eps_files = sorted(glob.glob(os.path.join(args.data_dir, "buffer", "episode_*.npz")))[: args.episodes]
    assert eps_files, f"no episodes under {args.data_dir}/buffer"
    obs = np.concatenate([np.load(f)["observation"][:, a:b] for f in eps_files]).astype(np.float64)

    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(obs), args.pairs), rng.integers(0, len(obs), args.pairs)
    keep = i != j
    d = np.linalg.norm(obs[i[keep]] - obs[j[keep]], axis=-1)

    eps = float(np.quantile(d, TARGET))
    print(f"block {args.block}  (d_c={b - a})   states={len(obs):,}  pairs={len(d):,}")
    print(f"pairwise L2: median {np.median(d):.3f}   mean {d.mean():.3f}   min {d.min():.3f}")
    print(f"\nCALIBRATED eps = {eps:.3f}   -> hit rate {(d <= eps).mean():.4f}  (target {TARGET})")
    print(f"  hit-spec string: \"{a}:{b}:{eps:.3f}\"")
    for p in args.probe:
        print(f"  probe eps={p:<6g} -> hit rate {(d <= p).mean():.4f}")


if __name__ == "__main__":
    main()
