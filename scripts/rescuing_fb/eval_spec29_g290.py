#!/usr/bin/env python
"""scripts/rescuing_fb/eval_spec29_g290.py -- spec 29 G29-0: the R-A pre-flight gate.

EVAL ONLY. No training, no training code path touched, nothing written outside
results/spec29_g290/. Thin wrapper: the pool/tower/kernel machinery is imported
from the committed spec-28 evals, the ridge solve from `spec28_encodings`, and
the collapse instrument (`z_geometry`) from `eval_z_action_ranking` -- only the
per-goal batching, the second spread statistic and the table are new here.

THE QUESTION (spec 29 G29-0). Spec 28 P0 established that the SHIPPED index
`z(g) = sqrt(d)*normalize(B_bar(g))` is COLLAPSED: mean pairwise cosine .998-1.000
over 4096 dataset goals against a sqrt(d)-sphere reference of .002, because
Sigma_B is rank-1 dominated. The dual coefficient `(Sigma_B + lambda I)^-1 m`
repaired reward READOUT and ASSIGNMENT. R-A wants to CONDITION on that dual
coefficient -- which is only worth a multi-day training arm if the dual indices
are themselves spread out. So, on existing checkpoints, encode the SAME goal set
both ways and put the two cosines next to the sphere reference.

  zhat(g) = sqrt(d) * normalize( (Sigma_hat_B + lambda I)^-1 m_hat(g) )
  m_hat(g) = (1/P) sum_j B_bar(u_j) * r_j,   r_j = kg_j = _delta_phi(goals, u_raw)

`kg` is the QUERY-SET SOFTMAX-NORMALIZED kernel (standing spec-28 rule): the
same object the g-indexed path integrates, so the g comparison means something.
lambda sweeps the three P0c-winning scales {1e-3,1e-2,1e-1} * tr(Sigma_hat)/d.

TWO SPREAD STATISTICS, NEVER ONE. Mean pairwise cosine near a collapsed cone is
exactly the statistic that misled this program before, and it saturates. So every
index set is also reported as a PARTICIPATION RATIO -- (sum lam)^2 / sum lam^2 of
the set's own second-moment (and covariance) spectrum, which is 1 for a perfectly
collapsed set and d for an isotropic one, with a matched random-sphere reference
computed through the identical code path.

TOWER. Primary = the tower `_density_z_encode` ITSELF encodes with
(`agent._density_z_enc_module()`: `target_density_b`, or the slow-EMA
`density_b_zenc` under spec-25 C3) -- the geometry R-A's index would live in.
The ONLINE `density_b` pool is reported too so the verdict cannot turn on a
tower choice.

READOUT-ONLY, like all of spec 28 P0: no F call, no actor call, no rollout,
nothing conditioned on the dual index (these checkpoints never trained on one).

Run (CPU jax, ~1-2 min/checkpoint):
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    taskset -c 24-39 /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/eval_spec29_g290.py \\
    --run-dir ~/exp_spec24/factored-fb/GCIQL-CubeDouble-*-DensityZ-d50/z24_cd_z_s0 \\
    --checkpoint-step 500000 --out results/spec29_g290/z24_cd_s0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

RIDGE_MULTS = (1e-3, 1e-2, 1e-1)


# ------------------------------------------------------------- pure logic --

def dual_index_batch(b_pool: np.ndarray, r_rows: np.ndarray, lam: float,
                     d: int | None = None) -> np.ndarray:
    """zhat(g) = sqrt(d) * normalize( (Sigma_B + lambda I)^-1 m(g) ) for a BATCH
    of reward rows. `r_rows` [N,P], `b_pool` [P,d] -> [N,d].

    Sigma_B = B^T B / P and m(g) = (1/P) sum_j r_gj B_j are exactly
    `spec28_encodings.ridge_index`'s objects (asserted equal, per-row, by a unit
    test) -- this function exists only to solve N right-hand sides against one
    factorization instead of N. The sqrt(d) re-normalization is spec 29 sec.1's
    contract ("||zhat||=sqrt(d) is re-imposed after the ridge solve"), and uses
    `gciql_flow._sphere`'s exact convention x*sqrt(d)/(||x||+1e-6).
    """
    b = np.asarray(b_pool, np.float64)
    r = np.atleast_2d(np.asarray(r_rows, np.float64))
    P, dd = b.shape
    assert r.shape[1] == P, f"dual_index_batch: r_rows {r.shape} vs pool {b.shape}"
    d = int(dd if d is None else d)
    sigma = b.T @ b / P
    m = r @ b / P                                                   # [N,d]
    c = np.linalg.solve(sigma + float(lam) * np.eye(dd), m.T).T     # [N,d]
    n = np.linalg.norm(c, axis=1, keepdims=True)
    return c * np.sqrt(float(d)) / (n + 1e-6)


def participation_ratio(evals: np.ndarray) -> float:
    """(sum lam)^2 / sum lam^2 of a non-negative spectrum: d for a flat one,
    -> 1 for a rank-1 one. Same formula `eval_spec28_p0b` already logs as
    `eff_rank` for Sigma_B -- reused here on the INDEX SET's spectrum."""
    ev = np.clip(np.asarray(evals, np.float64), 0.0, None)
    s2 = float((ev ** 2).sum())
    return float(ev.sum() ** 2 / max(s2, 1e-300))


def set_spread(z: np.ndarray) -> dict:
    """The scale-free companion to mean pairwise cosine, for a set of indices
    `z` [N,d]. Reports the participation ratio of BOTH the second moment
    (z^T z / N -- collapse to a single ray drives this to 1 even if the rays'
    lengths vary) and the covariance (centered -- how many directions the set
    actually varies in), plus the raw spectra's top eigenvalue share.

    A random sqrt(d)-sphere set scores ~d on both; a set on one ray scores ~1
    on the second moment. Reported alongside cosine, never instead of it."""
    z = np.asarray(z, np.float64)
    n, d = z.shape
    ev2 = np.linalg.eigvalsh(z.T @ z / n)
    mu = z.mean(0)
    evc = np.linalg.eigvalsh((z - mu).T @ (z - mu) / n)
    ev2 = np.clip(ev2, 0.0, None)
    evc = np.clip(evc, 0.0, None)
    return {
        "d": int(d), "n": int(n),
        "pr_second_moment": participation_ratio(ev2),
        "pr_covariance": participation_ratio(evc),
        "top_eig_share_second_moment": float(ev2[-1] / max(ev2.sum(), 1e-300)),
        "top_eig_share_covariance": float(evc[-1] / max(evc.sum(), 1e-300)),
        "norm_median": float(np.median(np.linalg.norm(z, axis=1))),
    }


def sphere_reference(n: int, d: int, rng) -> np.ndarray:
    """n draws on the sqrt(d) sphere -- the reference BOTH spread statistics are
    read against (the same object `z_geometry`'s `sphere_ref_cos_mean` uses)."""
    x = rng.normal(size=(int(n), int(d)))
    return x * np.sqrt(float(d)) / np.linalg.norm(x, axis=1, keepdims=True)


# ------------------------------------------------------------------ runner --

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--pool_size", type=int, default=16384)
    ap.add_argument("--n-goals", type=int, default=4096)
    ap.add_argument("--goal-chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    t00 = time.time()

    for p in (str(REPO), str(REPO / "third_party" / "ogbench" / "impls"),
              str(REPO / "tools" / "wandb_mode_shim"), str(REPO / "scripts"),
              str(REPO / "scripts" / "rescuing_fb")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from eval_density_zeroshot import build_pool
    from eval_density_z_rewards import analytic_zr
    from eval_z_action_ranking import z_geometry
    from eval_zeroshot_rewards import eval_tasks_for
    from env_geometry import geometry_for
    from spec28_encodings import ridge_index, ridge_lambdas

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])
    assert bool(cfg.get("measure_density", False)), \
        f"{run_dir} is not a measure_density checkpoint"
    env_name = flags["env_name"]
    d = int(cfg["measure_density_dim"])
    spec = geometry_for(env_name, str(cfg.get("measure_hit_spec") or "")).hit_spec
    blocks = spec.blocks

    import ogbench

    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks_for(env_name)[1])
    obs_all = np.asarray(ds1["observations"], np.float32)
    obs_dim, act_dim = obs_all.shape[-1], np.asarray(ds1["actions"]).shape[-1]

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)

    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    zenc_module = agent._density_z_enc_module()
    print(f"[g290] restored {run_dir.name} step={args.checkpoint_step} env={env_name} "
          f"z={bool(cfg.get('measure_density_z', False))} d={d} "
          f"zenc_tower={zenc_module} P={args.pool_size} N={args.n_goals} "
          f"({time.time() - t00:.0f}s)", flush=True)

    # ---- pool + towers (audited builders) ----
    rng = np.random.default_rng(args.seed)
    n_ds = len(obs_all)
    pool_idx = rng.choice(n_ds, size=min(args.pool_size, n_ds),
                          replace=(args.pool_size > n_ds))
    pool_obs = obs_all[pool_idx]
    u_raw, b_online = build_pool(agent, blocks, pool_obs, spec)
    u_sc = GCIQLFlowAgent._density_queries(jnp.asarray(pool_obs, jnp.float32), blocks)
    b_zenc = np.asarray(agent.network.select(zenc_module)(u_sc), np.float64)
    towers = {"zenc": b_zenc, "online": b_online}
    P = len(pool_obs)

    geom = {}
    for tname, b in towers.items():
        sig = b.T @ b / len(b)
        ev = np.clip(np.linalg.eigvalsh(sig), 0.0, None)
        geom[tname] = {"trace_over_d": float(np.trace(sig) / d),
                       "cond": float(ev[-1] / max(ev[0], 1e-12)),
                       "eff_rank": participation_ratio(ev)}
    print(f"[g290] Sigma_B[{zenc_module}] eff_rank="
          f"{geom['zenc']['eff_rank']:.4f}/{d} cond={geom['zenc']['cond']:.3g} "
          f"({time.time() - t00:.0f}s)", flush=True)

    # ---- the goal set: 4096 dataset goals, encoded BOTH ways ----
    g_idx = rng.choice(n_ds, size=min(args.n_goals, n_ds), replace=False)
    g_obs = obs_all[g_idx]
    N = len(g_obs)

    z_point = np.asarray(agent._density_z_encode(jnp.asarray(g_obs, jnp.float32)),
                         np.float64)                                          # [N,d]

    lam_by_tower = {t: ridge_lambdas(b, RIDGE_MULTS) for t, b in towers.items()}
    duals = {t: {m: np.zeros((N, d)) for m in lam_by_tower[t]} for t in towers}
    moments = {t: np.zeros((N, d)) for t in towers}
    for lo in range(0, N, args.goal_chunk):
        hi = min(lo + args.goal_chunk, N)
        kg = np.asarray(GCIQLFlowAgent._delta_phi(
            jnp.asarray(g_obs[lo:hi], jnp.float32), u_raw, blocks), np.float64)  # [n,P]
        for tname, b in towers.items():
            for m, lam in lam_by_tower[tname].items():
                duals[tname][m][lo:hi] = dual_index_batch(b, kg, lam, d)
            moments[tname][lo:hi] = np.stack(
                [analytic_zr(kg[i], b, d) for i in range(hi - lo)])
        if (hi // args.goal_chunk) % 4 == 0:
            print(f"[g290] goals {hi}/{N} ({time.time() - t00:.0f}s)", flush=True)

    # INSTRUMENT CONTROL: the batched solve must equal the committed per-row one.
    kg0 = np.asarray(GCIQLFlowAgent._delta_phi(
        jnp.asarray(g_obs[:4], jnp.float32), u_raw, blocks), np.float64)
    ctl = []
    for tname, b in towers.items():
        for m, lam in lam_by_tower[tname].items():
            ref = np.stack([ridge_index(b, kg0[i], lam) for i in range(4)])
            ref = ref * np.sqrt(float(d)) / (
                np.linalg.norm(ref, axis=1, keepdims=True) + 1e-6)
            err = float(np.max(np.abs(ref - duals[tname][m][:4])))
            assert err < 1e-8, f"batched dual != ridge_index ({tname},{m}): {err}"
            ctl.append(err)
    print(f"[g290] control: batched dual == spec28 ridge_index (max err "
          f"{max(ctl):.2e})", flush=True)

    # ---- spread: cosine (P0a instrument) + participation ratio, vs sphere ----
    rng_geo = np.random.default_rng(args.seed + 31)
    ref_z = sphere_reference(N, d, np.random.default_rng(args.seed + 32))
    sets = {"sphere_encoding_z(g)": z_point,
            "sphere_reference": ref_z}
    for tname in towers:
        sets[f"moment_primal[{tname}]"] = moments[tname]
        for m in lam_by_tower[tname]:
            sets[f"dual_ridge_{m:g}[{tname}]"] = duals[tname][m]

    rows = {}
    for name, z in sets.items():
        g = z_geometry(z, np.random.default_rng(args.seed + 31))
        rows[name] = {"cos_mean": g["cos_mean"], "cos_median": g["cos_median"],
                      "dist_median": g["dist_median"],
                      "sphere_ref_cos_mean": g["sphere_ref_cos_mean"],
                      "spread": set_spread(z)}
    del rng_geo

    out = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "d": d, "pool_size": int(P), "n_goals": int(N),
        "seed": args.seed, "tag": args.tag or run_dir.name,
        "is_density_z": bool(cfg.get("measure_density_z", False)),
        "zenc_tower": zenc_module,
        "measure_density_z_enc_tau": float(cfg.get("measure_density_z_enc_tau", 0.0)),
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
        "b_geometry": geom,
        "ridge_lambda": {t: {str(m): v for m, v in lam.items()}
                         for t, lam in lam_by_tower.items()},
        "reward_row": "r_j = kg_j = GCIQLFlowAgent._delta_phi(goals, u_raw, blocks) "
                      "-- the QUERY-SET SOFTMAX-NORMALIZED kernel (standing spec-28 "
                      "rule); m_hat(g) = mean_j r_j B_bar(u_j)",
        "probe": "spec 29 G29-0 pre-flight: is the DUAL index de-collapsed? "
                 "zhat = sqrt(d)*normalize((Sigma+lam I)^-1 m). READOUT ONLY -- "
                 "no F call, no actor, no rollout, nothing conditioned on zhat.",
        "spec": "PAPER/Multistep-FB/specs/29-ra-dual-index.md G29-0",
        "gate": "PASS = mean pairwise cosine well below .99 (sphere ref ~.002), "
                "corroborated by a participation ratio lifted off 1.",
    }, "index_sets": rows}

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec29_g290" / f"g290_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=lambda o: float(o)))

    print(f"\n[g290] {run_dir.name}  (sphere ref cos "
          f"{rows['sphere_reference']['cos_mean']:.3f}, PR2 "
          f"{rows['sphere_reference']['spread']['pr_second_moment']:.1f}/{d})")
    print(f"  {'index set':34s} {'cos_mean':>9s} {'PR(2nd mom)':>12s} {'PR(cov)':>9s}")
    for name, r in rows.items():
        print(f"  {name:34s} {r['cos_mean']:9.4f} "
              f"{r['spread']['pr_second_moment']:12.3f} "
              f"{r['spread']['pr_covariance']:9.3f}")
    print(f"[g290] DONE ({time.time() - t00:.0f}s) -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
