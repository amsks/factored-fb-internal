"""Effective rank of the learned measure representation, per cube checkpoint. Eval-only.

THE QUESTION. Every cube arm is nominally rank 50 (`z_dim`/`L_dim`/`measure_density_dim`),
and td_jepa's OGBench configs are 50 too, so rank is not a divergence. But the NOMINAL rank
only bounds what the factorization can express; what it actually uses is the spectrum of the
learned backward embedding over real states. Spec 30 already measured Density-FB's centered
covariance at an effective rank of 6-21 of 50. If the arms that collapse on cube-double are
likewise using a fraction of their 50, then buying more rank cannot be the fix and a rank
sweep would be spending GPU-days to confirm a null.

WHAT IT MEASURES. Embed N dataset next-states through the checkpoint's OWN backward tower
(never a reimplementation), center, and report three standard spectral summaries:

  participation ratio  (sum l)^2 / sum l^2   -- effective number of comparably-sized modes
  stable rank          ||M||_F^2 / ||M||_2^2 -- rank robust to a few dominant directions
  n99                  #eigenvalues to reach 99% of the variance

All three are <= the nominal rank; the gap between them and 50 is the headroom question.
Each is scale-free, so they are comparable across arms whose embeddings have different norms.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

FINAL = 500_000


def _embed(critic, net, pool):
    """The arm's backward/feature embedding of a batch of states, via its own tower."""
    import jax.numpy as jnp

    u = jnp.asarray(pool)
    name = type(critic).__name__
    if hasattr(critic, "_feature_net"):                       # HILP / Density-HILP
        # `_feature_net` takes the phi1 PARAMS, not the TrainState container
        # (hilp.py:414 is the reference call site).
        return np.asarray(critic._feature_net(net["hilp_phi1"].params,
                                              net["hilp_stats"].params, u))
    if "density_b" in net:                                    # Density-FB
        u_sc = critic._b_query(net, u, u)
        return np.asarray(critic._B(net["density_b"].params, u_sc))
    if "backward" in net:                                     # FB / RLDP / PSM
        return np.asarray(critic._B(net["backward"].params, u))
    raise NotImplementedError(f"no known backward tower on {name}: {sorted(net)}")


def spectrum(E: np.ndarray) -> dict:
    E = np.asarray(E, np.float64)
    E = E - E.mean(0, keepdims=True)                          # centered: the gauge spec 30 fixed
    s = np.linalg.svd(E, compute_uv=False)
    lam = s ** 2
    tot = lam.sum()
    if tot <= 0:
        return dict(nominal=E.shape[1], participation=0.0, stable=0.0, n99=0)
    cum = np.cumsum(lam) / tot
    return dict(nominal=int(E.shape[1]),
                participation=float(tot ** 2 / (lam ** 2).sum()),
                stable=float(tot / lam.max()),
                n99=int(np.searchsorted(cum, 0.99) + 1),
                top1_frac=float(lam.max() / tot))


def main() -> int:
    import jax.numpy as jnp

    from env import get_adapter
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset
    from impls.utils.flax_utils import restore_agent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cube-double-play-v0")
    ap.add_argument("--critics", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epoch", type=int, default=FINAL)
    ap.add_argument("--pool", type=int, default=8192)
    ap.add_argument("--exp-root", default="exp")
    ap.add_argument("--out", default="analysis/measure_rank.json")
    args = ap.parse_args()

    rows = []
    for critic in args.critics:
        for seed in args.seeds:
            d = Path(args.exp_root) / f"{args.env}_{critic}_flowbc_seed_{seed}"
            if not (d / f"params_{args.epoch}.pkl").exists():
                continue
            meta = json.loads((d / "config.json").read_text())
            cfg = meta["config"]
            np.random.seed(meta["seed"])
            adapter = get_adapter(cfg)
            env, train_ds, _v = adapter.make_env_and_datasets(cfg)
            ex = make_dataset(train_ds, cfg).sample(1)
            agent = Agent.create(meta["critic"], meta["actor"], meta["seed"],
                                 jnp.asarray(ex["observations"]),
                                 jnp.asarray(ex["actions"]), cfg)
            agent = restore_agent(agent, str(d), args.epoch)
            rng = np.random.default_rng(0)
            idxs = np.sort(rng.choice(train_ds.size if hasattr(train_ds, "size")
                                      else len(train_ds.valid_idxs), args.pool, replace=False))
            batch = train_ds.sample(len(idxs), idxs=idxs)
            pool = np.asarray(batch["next_observations"], np.float32)
            st = spectrum(_embed(agent.critic, agent.net, pool))
            st.update(critic=critic, seed=seed, env=args.env)
            rows.append(st)
            print(f"{critic:<20} s{seed}  nominal={st['nominal']:<4} "
                  f"participation={st['participation']:6.2f}  stable={st['stable']:6.2f}  "
                  f"n99={st['n99']:<4} top1={st['top1_frac']:.3f}", flush=True)
            if hasattr(env, "close"):
                env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
