"""Calibrate the goal kernel radius eps for an observation block.

Default: eps so the hit rate of random state pairs matches cube-single's (0.0133).
With --target_support: eps so the softmax goal kernel over 1024 candidate next states has a
given effective support exp(entropy).

    python scripts/calibrate_hit_spec.py --ogbench_npz ~/.ogbench/data/cube-single-play-v0.npz \
        --block 19:22 --report_eps 0.4
    python scripts/calibrate_hit_spec.py --data_dir ~/exorl_data/exorl/datasets/walker/rnd \
        --block 0:24 --target_support 60
"""
import argparse, glob, os
import numpy as np

TARGET = 0.0133   # cube-single hit rate at radius 0.4


def effective_support(block, eps, n_cand=1024, n_goals=64, seed=0):
    """Mean exp(entropy) of softmax_j(-0.5 ||u_j - g||^2 / eps^2) over random goals g."""
    rng = np.random.default_rng(seed)
    block = np.asarray(block, np.float64)
    u = block[rng.choice(len(block), n_cand, replace=False)]   # distinct candidates
    g = block[rng.integers(0, len(block), n_goals)]
    d2 = ((g[:, None, :] - u[None, :, :]) ** 2).sum(-1)          # [G, M]
    logits = -0.5 * d2 / (float(eps) ** 2)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    return float(np.exp(ent).mean())


def calibrate_support(block, target, n_cand=1024, n_goals=64, seed=0, iters=60):
    """Bisect log eps so effective_support matches target."""
    block = np.asarray(block, np.float64)
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(block), 20000), rng.integers(0, len(block), 20000)
    d = np.linalg.norm(block[i] - block[j], axis=-1)
    lo, hi = np.log(max(d[d > 0].min(), 1e-9) * 1e-2), np.log(d.max() * 1e2)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if effective_support(block, np.exp(mid), n_cand, n_goals, seed) < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def load_block(args):
    a, b = (int(x) for x in args.block.split(":"))
    if args.ogbench_npz:
        obs = np.load(args.ogbench_npz)["observations"][:, a:b]
        return obs.astype(np.float64), a, b
    eps_files = sorted(glob.glob(os.path.join(args.data_dir, "buffer", "episode_*.npz")))[: args.episodes]
    assert eps_files, f"no episodes under {args.data_dir}/buffer"
    obs = np.concatenate([np.load(f)["observation"][:, a:b] for f in eps_files]).astype(np.float64)
    return obs, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", help="ExoRL domain dir containing buffer/episode_*.npz")
    ap.add_argument("--ogbench_npz", help="an OGBench dataset npz (uses its 'observations')")
    ap.add_argument("--block", default="0:214", help="a:b over the RAW observation")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=400_000)
    ap.add_argument("--probe", type=float, nargs="*", default=[2.0, 4.0],
                    help="also report the hit rate at these radii")
    ap.add_argument("--target_support", type=float, default=None,
                    help="SUPPORT mode: eps so the softmax bump's effective support over 1024 "
                         "candidates equals this (cube-single 19:22:0.4 -> see --report_eps)")
    ap.add_argument("--report_eps", type=float, nargs="*", default=[],
                    help="report the effective support at these radii (no calibration)")
    ap.add_argument("--n_goals", type=int, default=256)
    args = ap.parse_args()
    assert bool(args.data_dir) != bool(args.ogbench_npz), "give exactly one of --data_dir / --ogbench_npz"
    obs, a, b = load_block(args)

    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(obs), args.pairs), rng.integers(0, len(obs), args.pairs)
    keep = i != j
    d = np.linalg.norm(obs[i[keep]] - obs[j[keep]], axis=-1)
    print(f"block {args.block}  (d_c={b - a})   states={len(obs):,}  pairs={len(d):,}")
    print(f"pairwise L2: median {np.median(d):.3f}   mean {d.mean():.3f}   min {d.min():.3f}")

    for e in args.report_eps:
        s = effective_support(obs, e, n_goals=args.n_goals)
        print(f"  eps={e:<8g} -> support {s:7.1f} / 1024   hit rate {(d <= e).mean():.4f}")

    if args.target_support is not None:
        eps = calibrate_support(obs, args.target_support, n_goals=args.n_goals)
        s = effective_support(obs, eps, n_goals=args.n_goals)
        print(f"\nCALIBRATED (support) eps = {eps:.4f}   -> support {s:.1f} / 1024 "
              f"(target {args.target_support})   hit rate {(d <= eps).mean():.5f}   median/eps {np.median(d)/eps:.2f}")
        print(f"  hit-spec string: \"{a}:{b}:{eps:.4f}\"")
        return

    if not args.report_eps:
        eps = float(np.quantile(d, TARGET))
        print(f"\nCALIBRATED (hit rate) eps = {eps:.3f}   -> hit rate {(d <= eps).mean():.4f}  (target {TARGET})")
        print(f"  hit-spec string: \"{a}:{b}:{eps:.3f}\"")
        print(f"  support at that eps: {effective_support(obs, eps, n_goals=args.n_goals):.1f} / 1024")
        for p in args.probe:
            print(f"  probe eps={p:<6g} -> hit rate {(d <= p).mean():.4f}")


if __name__ == "__main__":
    main()
