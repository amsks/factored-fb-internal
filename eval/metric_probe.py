"""Metric probes for trained critics (companion to docs/specs/2026-07-31-bilinear-gciql-metric-probe.md).

Loads a checkpoint WITHOUT creating an env, computes per-construction distance matrices on
the fixed probe set, and runs the probe battery:
  P1 temporal-distance rank correlation (+ partialed on raw-obs distance)
  P3 triangle-inequality violation rate/magnitude (global + trust region)
  P4 symmetry index, self-distance
  P5 cosine-proxy (is the bilinear logit secretly cosine?)
  P7 effective rank / spectrum of embeddings
Writes <out>.json (scalars) and <out>.npz (float16 matrices + embeddings, for cross-method
RSA via eval/probe_report.py).

Usage (mirror the training run's critic/actor/domain/overrides/seed so shapes match):
  python -m eval.metric_probe --critic iql_bilinear --actor flowbc --domain cube-single \
      --override value_latent_dim=50 --seed 0 \
      --restore_path 'exp/metric-probe/cube-single-play-v0_iql_bilinear_flowbc_seed_0' \
      --restore_epoch 500000 --probe_set exp/metric-probe/probe_set.npz \
      --out exp/metric-probe/probes/iql_bilinear_d50_s0_500k
Pass --init_only to probe the untrained network (the D9 null control).
"""
import argparse
import json
import os

import numpy as np


# ---------------- numpy stats helpers ----------------

def _ranks(x):
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def spearman(a, b):
    ra, rb = _ranks(np.asarray(a, np.float64)), _ranks(np.asarray(b, np.float64))
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def partial_spearman(x, y, z):
    """rank corr of x,y controlling for z."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / den) if den > 0 else float("nan")


def eff_rank(E):
    s = np.linalg.svd(np.asarray(E, np.float64) - E.mean(0), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------- probes on a distance matrix ----------------

def matrix_probes(D, rng, n_triplets=1_000_000):
    N = D.shape[0]
    off = ~np.eye(N, dtype=bool)
    i, j, k = (rng.integers(N, size=n_triplets) for _ in range(3))
    ok = (i != j) & (j != k) & (i != k)
    i, j, k = i[ok], j[ok], k[ok]
    v = D[i, k] - D[i, j] - D[j, k]
    tau = np.quantile(D[off], 0.5)
    trust = (D[i, k] <= tau) & (D[i, j] <= tau) & (D[j, k] <= tau)
    pos = v > 0
    out = {
        "tri_violation_rate": float(pos.mean()),
        "tri_violation_mag": float(v[pos].mean()) if pos.any() else 0.0,
        "tri_violation_rate_trust": float(pos[trust].mean()) if trust.any() else float("nan"),
        "tri_violation_mag_trust": float(v[trust][v[trust] > 0].mean())
            if (trust.any() and (v[trust] > 0).any()) else 0.0,
        "sym_index": float(np.abs(D - D.T)[off].mean() / np.maximum((D + D.T)[off].mean(), 1e-9)),
        "self_dist_mean": float(np.diag(D).mean()),
        "d_offdiag_mean": float(D[off].mean()),
    }
    return out


# ---------------- per-critic constructions ----------------

def steps_from_value(V, gamma):
    return np.log(np.clip(1.0 + (1.0 - gamma) * V, 1e-6, 1.0)) / np.log(gamma)


def cos_matrix_dist(E):
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return 1.0 - En @ En.T


def l2_matrix(E):
    sq = (E ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * E @ E.T, 0.0))


def cos_proxy_stats(phi, psi):
    dots = (phi * psi).sum(1)
    cos = dots / np.maximum(np.linalg.norm(phi, axis=1) * np.linalg.norm(psi, axis=1), 1e-9)
    npn, ppn = np.linalg.norm(phi, axis=1), np.linalg.norm(psi, axis=1)
    return {"dot_cos_spearman": spearman(dots, cos),
            "dot_cos_pearson": float(np.corrcoef(dots, cos)[0, 1]),
            "phi_norm_cv": float(npn.std() / npn.mean()),
            "psi_norm_cv": float(ppn.std() / ppn.mean())}


def build_constructions(agent, critic_name, cfg, S, pair_s, pair_g):
    """Return (matrices: name->(N,N) distance, pairs: name->(M,) distance, embeds, extras)."""
    import jax.numpy as jnp
    gamma = float(cfg["discount"])
    p = {g: agent.net[g].params for g in agent.net}
    mats, pairs, embeds, extras = {}, {}, {}, {}

    if critic_name in ("iql_bilinear", "crl"):
        mod = agent.critic.value
        _, phi, psi = map(np.asarray, mod.apply({"params": p["value"]}, jnp.asarray(S),
                                                jnp.asarray(S), info=True))
        d = phi.shape[-1]
        F = phi @ psi.T / np.sqrt(d)                       # F[i,j] = f(s_i, g_j)
        mats["selfsub"] = np.diag(F)[None, :] - F          # d(s,g) = f(g,g) - f(s,g)
        mats["psi_cos"] = cos_matrix_dist(psi)
        mats["psi_l2"] = l2_matrix(psi)
        if critic_name == "iql_bilinear":
            mats["steps"] = steps_from_value(F, gamma)
        embeds["phi"], embeds["psi"] = phi, psi
        extras["cos_proxy"] = cos_proxy_stats(phi, psi)
        extras["self_value_mean"] = float(np.diag(F).mean())
        # pairwise: f(s,g) and f(g,g)
        _, phs, _ = map(np.asarray, mod.apply({"params": p["value"]}, jnp.asarray(pair_s),
                                              jnp.asarray(pair_s), info=True))
        _, phg, psg = map(np.asarray, mod.apply({"params": p["value"]}, jnp.asarray(pair_g),
                                                jnp.asarray(pair_g), info=True))
        f_sg = (phs * psg).sum(1) / np.sqrt(d)
        f_gg = (phg * psg).sum(1) / np.sqrt(d)
        pairs["selfsub"] = f_gg - f_sg
        pairs["psi_cos"] = 1.0 - (lambda a, b: (a * b).sum(1) / np.maximum(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9))(
                np.asarray(mod.apply({"params": p["value"]}, jnp.asarray(pair_s),
                                     jnp.asarray(pair_s), info=True)[2]), psg)
        if critic_name == "iql_bilinear":
            pairs["steps"] = steps_from_value(f_sg, gamma)

    elif critic_name == "gciql":
        mod, vp = agent.critic.value, p["value"]
        import jax
        vfn = jax.jit(lambda o, g: mod.apply({"params": vp}, o, g))
        N = S.shape[0]
        V = np.empty((N, N), np.float64)
        Sj = jnp.asarray(S)
        for j in range(N):
            V[:, j] = np.asarray(vfn(Sj, jnp.broadcast_to(Sj[j], S.shape)))
        mats["steps"] = steps_from_value(V, gamma)
        pairs["steps"] = steps_from_value(
            np.asarray(vfn(jnp.asarray(pair_s), jnp.asarray(pair_g))), gamma)
        extras["self_value_mean"] = float(np.diag(V).mean())

    elif critic_name == "qrl":
        from impls.utils.quasimetrics import iqe
        mod, vp = agent.critic.value, p["value"]
        U = np.asarray(mod.apply({"params": vp}, jnp.asarray(S), method="embed"))
        k = int(cfg["qrl_components"])
        alpha = float(1 / (1 + np.exp(-np.asarray(vp["alpha_raw"]))))
        N, D = U.shape[0], np.empty((U.shape[0], U.shape[0]), np.float64)
        Uj = jnp.asarray(U)
        chunk = 16
        for i in range(0, N, chunk):
            D[i:i + chunk] = np.asarray(
                iqe(Uj[i:i + chunk, None, :], Uj[None, :, :], k, alpha))
        mats["native"] = D
        us = np.asarray(mod.apply({"params": vp}, jnp.asarray(pair_s), method="embed"))
        ug = np.asarray(mod.apply({"params": vp}, jnp.asarray(pair_g), method="embed"))
        pairs["native"] = np.asarray(iqe(jnp.asarray(us), jnp.asarray(ug), k, alpha))
        embeds["encoder"] = U

    elif critic_name == "vip":
        mod, vp = agent.critic.value, p["value"]
        E = np.asarray(mod.apply({"params": vp}, jnp.asarray(S)))
        mats["native"] = l2_matrix(E)
        es = np.asarray(mod.apply({"params": vp}, jnp.asarray(pair_s)))
        eg = np.asarray(mod.apply({"params": vp}, jnp.asarray(pair_g)))
        pairs["native"] = np.linalg.norm(es - eg, axis=1)
        embeds["phi"] = E
    else:
        raise ValueError(f"no probe constructions for critic {critic_name!r}")

    return mats, pairs, embeds, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", required=True)
    ap.add_argument("--actor", default="bcflow")
    ap.add_argument("--domain", default="cube-single")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restore_path", default="")
    ap.add_argument("--restore_epoch", type=int, default=0)
    ap.add_argument("--init_only", action="store_true", help="probe the untrained net (D9 null)")
    ap.add_argument("--probe_set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_triplets", type=int, default=1_000_000)
    args = ap.parse_args()

    import jax.numpy as jnp  # noqa: F401  (imported before agent code touches jax)
    from env import make_config
    from impls.agents import Agent
    from impls.utils.flax_utils import restore_agent

    ps = np.load(args.probe_set)
    S, pair_s, pair_g, pair_dt = (ps[k] for k in ("states", "pair_s", "pair_g", "pair_dt"))

    cfg = make_config(args.critic, args.actor, args.domain, args.override)
    ex_obs = jnp.asarray(S[:1])
    ex_act = jnp.zeros((1, int(cfg.get("action_dim", 0))), jnp.float32) \
        if cfg.get("action_dim") else None
    if ex_act is None:
        # action dim from any run's dataset is not available env-free; cube-single is 5-dim.
        ex_act = jnp.zeros((1, int(cfg.get("probe_action_dim", 5))), jnp.float32)
    agent = Agent.create(args.critic, args.actor, args.seed, ex_obs, ex_act, cfg)
    if not args.init_only:
        assert args.restore_path, "pass --restore_path or --init_only"
        agent = restore_agent(agent, args.restore_path, args.restore_epoch)

    mats, pairs, embeds, extras = build_constructions(agent, args.critic, cfg, S, pair_s, pair_g)

    # raw-observation baseline (D10) — always included
    mats["rawobs_l2"] = l2_matrix(S.astype(np.float64))
    raw_pair = np.linalg.norm(pair_s.astype(np.float64) - pair_g.astype(np.float64), axis=1)
    pairs["rawobs_l2"] = raw_pair

    rng = np.random.default_rng(0)
    report = {"critic": args.critic, "epoch": 0 if args.init_only else args.restore_epoch,
              "seed": args.seed, "init_only": args.init_only, "constructions": {}, **extras}
    for name, D in mats.items():
        row = matrix_probes(np.asarray(D, np.float64), rng, args.n_triplets)
        if name in pairs:
            row["p1_dt_spearman"] = spearman(pairs[name], pair_dt)
            row["p1_dt_spearman_partial_rawobs"] = partial_spearman(pairs[name], pair_dt, raw_pair)
        report["constructions"][name] = row
    for ename, E in embeds.items():
        report[f"eff_rank_{ename}"] = eff_rank(E)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    np.savez_compressed(args.out + ".npz",
                        **{f"mat_{k}": v.astype(np.float16) for k, v in mats.items()},
                        **{f"emb_{k}": v.astype(np.float16) for k, v in embeds.items()})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
