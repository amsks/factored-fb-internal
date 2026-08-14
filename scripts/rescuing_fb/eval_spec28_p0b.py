#!/usr/bin/env python
"""scripts/rescuing_fb/eval_spec28_p0b.py -- spec 28 P0b: three-encoding reward
reconstruction (PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md).

EVAL ONLY. No training, no training code path is touched, nothing is written
outside results/spec28_p0b/.

QUESTION. Density-FB's softmax normalization destroys the linear identity
`Q_r = F^T E[r B]`, and the CE loss pins only the product `F^T B` -- so B's
Euclidean geometry is gauge-arbitrary. `z(g) = sqrt(d)*normalize(B_bar(g))`
(what the interface uses) and `m = E[r B]` (what spec 24's analytic assignment
uses) are both PRIMAL feature vectors; the actor's reward readout
`r_z(u) = B~(u)^T z / sqrt(d)` needs the DUAL coefficient
`c = (Sigma_B + lambda I)^{-1} m`. They coincide only under a whitened B --
which this project never trains for (the orthonormality loss is a convicted
toxin here and stays refused). So: build all three indices for the same goal
and ask which one's reconstruction `rhat(u) = B(u)^T z` actually looks like the
goal kernel `kg` over the pool.

PROTOCOL (reuses the audited machinery; nothing is reimplemented).
- Pool: `eval_density_zeroshot.build_pool` at P=16384, drawn with the SAME rng
  stream ORDER as `eval_density_z_rewards.main()` (relabel draw of
  --relabel-size FIRST, then the pool draw) so the cells line up one-for-one
  with the archived e2a/e2b cells on the same seed.
- Cells: the 5 eval-task goals (their env-reset `info['goal']` observations)
  plus the --regions hindsight region centers drawn by the audited
  `draw_region_centers_with_obs` under default_rng(seed+2) -- the same 14 cells
  the archived e2b runs used.
- Target `kg` = `GCIQLFlowAgent._delta_phi(goal_obs, u_raw, blocks)`: the EXACT
  kernel the g-indexed path uses, QUERY-SET SOFTMAX-NORMALIZED over the M pool
  queries (spec 28 sec.0's first audit correction -- not raw kappa).
- Encodings scored: `point` (the agent's own `_density_z_encode`), `moment`
  (spec 24's `analytic_zr`, imported and called, i.e. sqrt(d)*normalize(E[kg B]))
  and `moment_raw` (its un-normalized variant), `ridge_<mult>` for
  lambda = mult*tr(Sigma_B)/d, mult in {1e-3,1e-2,1e-1}, and `random` (a
  sqrt(d)-sphere draw, the empirical chance floor).
- Scores: hit/non-hit AUC, Spearman vs kg, top-k precision vs kg, and
  hit-vs-hard-negative margin in units of the reconstruction's own std.

TOWER. The primary tower is the TARGET one (`target_density_b`): it is what
`_density_z_encode` encodes with AND what `_density_actor_q_z` reads the reward
off (`tb24`), so it is the geometry the interface actually lives in. The ONLINE
`density_b` pool (what `build_pool` returns, used for the F^T B logits) is
scored too, under "online", so the verdict cannot turn on a tower choice.

READOUT-ONLY. A ridge coefficient is NOT norm-preserving, and ||z||=sqrt(d)
holds wherever z enters F or the actor -- so on existing checkpoints the ridge
index may only be READ OUT, never conditioned on (spec 28 sec.0, third bullet).
Nothing here conditions anything on it: no F call, no actor call, no rollout.

Run (CPU jax; a few minutes per checkpoint, no rollouts):
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec28_p0b.py \\
    --run-dir ~/exp_spec24/factored-fb/GCIQL-CubeDouble-*-DensityZ-d50/z24_cd_z_s0 \\
    --checkpoint-step 500000 --pool_size 16384 --seed 0 \\
    --out results/spec28_p0b/z24_cd_z_s0.json
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

from eval_density_zeroshot import build_pool  # noqa: E402
from eval_density_z_rewards import analytic_zr, target_b_pool  # noqa: E402
from eval_partition_zeroshot import draw_region_centers_with_obs  # noqa: E402
from eval_zeroshot_rewards import eval_tasks_for, reward_conj_dirac  # noqa: E402
from env_geometry import episode_bounds, geometry_for  # noqa: E402
from spec28_encodings import (  # noqa: E402
    moment_index,
    ridge_index,
    ridge_lambdas,
    score_encoding,
    score_rhat,
)

RIDGE_MULTS = (1e-3, 1e-2, 1e-1)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--regions", type=int, default=4)
    ap.add_argument("--region-min-pos", type=int, default=5)
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--pool_size", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=3,
                    help="sqrt(d)-sphere draws averaged into the `random` chance floor")
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])
    assert bool(cfg.get("measure_density", False)), \
        f"{run_dir} is not a measure_density checkpoint (P0b scores B's geometry)"
    env_name = flags["env_name"]
    d = int(cfg["measure_density_dim"])
    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    blocks = spec.blocks
    eval_tasks = eval_tasks_for(env_name)

    t00 = time.time()
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    _obs_dim = int(np.asarray(ds1["observations"]).shape[-1])
    _act_dim = int(np.asarray(ds1["actions"]).shape[-1])

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, _obs_dim, _act_dim)

    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent, _sphere

    # ---- rng stream: IDENTICAL ORDER to eval_density_z_rewards.main() ----
    rng_np = np.random.default_rng(args.seed)
    obs_all = np.asarray(ds1["observations"], np.float32)
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(ds1["terminals"]))
    idx = rng_np.choice(len(coords_all), size=min(args.relabel_size, len(coords_all)),
                        replace=False)
    relabel_coords = coords_all[idx]
    pool_idx = rng_np.choice(len(obs_all), size=min(args.pool_size, len(obs_all)),
                             replace=(args.pool_size > len(obs_all)))
    pool_obs = obs_all[pool_idx]
    pool_coords = coords_all[pool_idx]
    u_raw, b_online = build_pool(agent, blocks, pool_obs, spec)          # ONLINE density_b
    b_target = target_b_pool(agent, blocks, pool_obs)                    # TARGET tower
    towers = {"target": b_target, "online": b_online}
    print(f"[p0b] {run_dir.name}: pool P={len(pool_obs)} d={d} "
          f"({time.time()-t00:.0f}s)", flush=True)

    # ---- cells: the 5 task goals + the audited hindsight region centers ----
    cells = []
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        g_obs = np.asarray(info["goal"], np.float32)
        cells.append(("task", f"t{i}", g_obs, spec.coords(g_obs[None])[0]))
        env.close()
    rng_reg = np.random.default_rng(args.seed + 2)
    centers, cobs, _npos, _rej = draw_region_centers_with_obs(
        obs_all, coords_all, bounds, spec, relabel_coords, rng_reg,
        args.regions, min_pos=args.region_min_pos)
    for k_i in range(len(centers)):
        cells.append(("region", f"c{k_i}", np.asarray(cobs[k_i], np.float32),
                      np.asarray(centers[k_i], np.float64)))

    # ---- per-tower B geometry diagnostics (how far from whitened is B?) ----
    geom = {}
    for tname, b in towers.items():
        sigma = b.T @ b / len(b)
        ev = np.linalg.eigvalsh(sigma)
        ev = np.clip(ev, 0.0, None)
        geom[tname] = {
            "trace_over_d": float(np.trace(sigma) / d),
            "eig_max": float(ev[-1]), "eig_min": float(ev[0]),
            "cond": float(ev[-1] / max(ev[0], 1e-12)),
            # participation ratio: d for a white Sigma, ->1 for a rank-1 one.
            "eff_rank": float(ev.sum() ** 2 / max(float((ev ** 2).sum()), 1e-30)),
            "b_norm_mean": float(np.linalg.norm(b, axis=1).mean()),
        }
    print(f"[p0b] Sigma_B target: cond={geom['target']['cond']:.3g} "
          f"eff_rank={geom['target']['eff_rank']:.2f}/{d}", flush=True)

    rng_z = np.random.default_rng(args.seed + 101)
    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "d": d, "pool_size": int(len(pool_obs)),
        "relabel_size": int(len(relabel_coords)), "seed": args.seed,
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
        "measure_density_z": bool(cfg.get("measure_density_z", False)),
        "measure_density_z_enc_tau": float(cfg.get("measure_density_z_enc_tau", 0.0)),
        "measure_density_z_tilt": float(cfg.get("measure_density_z_tilt", 0.0)),
        "ridge_mults": list(RIDGE_MULTS),
        "b_geometry": geom,
        "probe": "spec 28 P0b: rhat(u)=B(u)^T z vs the exact query-set-normalized "
                 "kernel kg=_delta_phi(goal, U_raw). point/moment are PRIMAL, "
                 "ridge is the DUAL coefficient (Sigma_B+lambda I)^-1 E[kg B]. "
                 "READOUT ONLY -- no F call, no actor, no rollout, no conditioning "
                 "on the ridge index (not norm-preserving; OOD for these ckpts).",
        "spec": "PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md P0b",
    }, "cells": {}}

    lam_by_tower = {t: ridge_lambdas(b, RIDGE_MULTS) for t, b in towers.items()}
    results["meta"]["ridge_lambda"] = {
        t: {str(m): v for m, v in lam.items()} for t, lam in lam_by_tower.items()}

    for kind, name, g_obs, center in cells:
        kg = np.asarray(GCIQLFlowAgent._delta_phi(
            jnp.asarray(g_obs, jnp.float32)[None], u_raw, blocks), np.float64)[0]  # [P]
        hit = reward_conj_dirac(pool_coords, center, spec)
        dist = np.linalg.norm(spec.scaled(pool_coords)
                              - spec.scaled(np.atleast_2d(center))[0][None], axis=-1)
        z_point = np.asarray(agent._density_z_encode(
            jnp.asarray(g_obs, jnp.float32)[None]), np.float64)[0]                 # [d]
        cell = {"kind": kind, "center_coord": np.asarray(center).tolist(),
                "n_hit": int(hit.sum()), "kg_max": float(kg.max()),
                "dist_min_scaled": float(dist.min()), "towers": {}}
        for tname, b in towers.items():
            enc = {"point": z_point,
                   "moment": analytic_zr(kg, b, d),
                   "moment_raw": moment_index(b, kg)}
            for m, lam in lam_by_tower[tname].items():
                enc[f"ridge_{m:g}"] = ridge_index(b, kg, lam)
            scored = {k: score_encoding(b, z, kg, hit, dist) for k, z in enc.items()}
            # INSTRUMENT CONTROL, both ends, through the identical scoring path:
            # the target kernel itself must score 1.0 and a sqrt(d)-sphere draw
            # must score chance. Neither is an encoding; both gate the cell.
            scored["oracle_kg"] = score_rhat(kg, kg, hit, dist)
            rnd = [score_encoding(b, np.asarray(_sphere(
                jnp.asarray(rng_z.normal(size=(1, d)), jnp.float32), float(d))[0],
                np.float64), kg, hit, dist) for _ in range(args.n_random)]
            scored["random"] = {
                "auc": float(np.mean([r["auc"] for r in rnd])) if rnd[0]["auc"] is not None else None,
                "n_hit": rnd[0]["n_hit"],
                "spearman_kg": float(np.mean([r["spearman_kg"] for r in rnd])),
                "topk_prec": {k: float(np.mean([r["topk_prec"][k] for r in rnd]))
                              for k in rnd[0]["topk_prec"]},
                "margin_std": float(np.mean([r["margin_std"] for r in rnd]))
                if rnd[0]["margin_std"] is not None else None,
                "z_norm": float(np.mean([r["z_norm"] for r in rnd])),
                "n_draws": len(rnd)}
            # cos(primal moment, dual ridge): 1.0 would mean the two indices are
            # the SAME direction and the whole primal/dual distinction is moot.
            mm = enc["moment_raw"]
            scored["_cos_moment_ridge"] = {
                f"{m:g}": float(mm @ enc[f'ridge_{m:g}']
                                / (np.linalg.norm(mm)
                                   * np.linalg.norm(enc[f'ridge_{m:g}']) + 1e-30))
                for m in lam_by_tower[tname]}
            cell["towers"][tname] = scored
        results["cells"][name] = cell
        tgt = cell["towers"]["target"]
        print(f"[p0b] {name:4s} n_hit={cell['n_hit']:5d} | AUC point={tgt['point']['auc']} "
              f"moment={tgt['moment']['auc']} "
              f"ridge1e-2={tgt['ridge_0.01']['auc']} rand={tgt['random']['auc']} | "
              f"rho point={tgt['point']['spearman_kg']:.3f} "
              f"moment={tgt['moment']['spearman_kg']:.3f} "
              f"ridge1e-2={tgt['ridge_0.01']['spearman_kg']:.3f}", flush=True)

    # ---- aggregate per (tower, encoding) over cells ----
    agg = {}
    enc_names = [k for k in results["cells"][cells[0][1]]["towers"]["target"]
                 if not k.startswith("_")]
    for tname in towers:
        agg[tname] = {}
        for e in enc_names:
            aucs = [c["towers"][tname][e]["auc"] for c in results["cells"].values()
                    if c["towers"][tname][e]["auc"] is not None]
            rhos = [c["towers"][tname][e]["spearman_kg"] for c in results["cells"].values()]
            mgs = [c["towers"][tname][e]["margin_std"] for c in results["cells"].values()
                   if c["towers"][tname][e]["margin_std"] is not None]
            p64 = [c["towers"][tname][e]["topk_prec"]["64"]
                   for c in results["cells"].values()]
            agg[tname][e] = {
                "auc_mean": float(np.mean(aucs)) if aucs else None,
                "auc_n": len(aucs),
                "spearman_mean": float(np.mean(rhos)),
                "topk64_mean": float(np.mean(p64)),
                "margin_std_mean": float(np.mean(mgs)) if mgs else None}
    results["meta"]["aggregate"] = agg

    # ---- INSTRUMENT GATE (spec 28 P0: a control before any z cell is read) ----
    # Both ends of the scale, scored through the identical path as the encodings:
    # the target itself must saturate, a random index must read chance. A probe
    # that fails either cannot license a verdict about the encodings between them.
    orc, rnd = agg["target"]["oracle_kg"], agg["target"]["random"]
    gate = {
        "oracle_auc": orc["auc_mean"], "oracle_spearman": orc["spearman_mean"],
        "oracle_topk64": orc["topk64_mean"],
        "random_auc": rnd["auc_mean"], "random_spearman": rnd["spearman_mean"],
        "random_topk64": rnd["topk64_mean"],
        "pass": bool(orc["auc_mean"] is not None and orc["auc_mean"] > 0.999
                     and orc["spearman_mean"] > 0.999 and orc["topk64_mean"] > 0.999
                     and rnd["auc_mean"] is not None and abs(rnd["auc_mean"] - 0.5) < 0.1
                     and abs(rnd["spearman_mean"]) < 0.15 and rnd["topk64_mean"] < 0.05),
        "rule": "oracle (rhat := kg) must score AUC/rho/p@64 > .999; random "
                "sqrt(d)-sphere index must read AUC within .1 of chance, |rho| < .15, "
                "p@64 < .05",
    }
    results["meta"]["instrument_gate"] = gate
    print(f"[p0b] INSTRUMENT GATE: {'PASS' if gate['pass'] else 'FAIL'} "
          f"oracle AUC={gate['oracle_auc']} rho={gate['oracle_spearman']:.4f} "
          f"p@64={gate['oracle_topk64']:.3f} | random AUC={gate['random_auc']:.3f} "
          f"rho={gate['random_spearman']:.3f} p@64={gate['random_topk64']:.3f}",
          flush=True)

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec28_p0b" / f"{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[p0b] DONE ({time.time()-t00:.0f}s) -> {outp}")
    for e in enc_names:
        a = agg["target"][e]
        print(f"[p0b]   {e:14s} AUC={a['auc_mean']} rho={a['spearman_mean']:.3f} "
              f"p@64={a['topk64_mean']:.3f} margin={a['margin_std_mean']}", flush=True)
    return 0 if results["meta"]["instrument_gate"]["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
