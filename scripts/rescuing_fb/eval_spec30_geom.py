#!/usr/bin/env python
"""scripts/rescuing_fb/eval_spec30_geom.py -- centered-vs-uncentered B geometry.

EVAL ONLY. Nothing written outside results/spec30_geom/.

QUESTION (external review sec.4.4/4.5, 2026-08-16). The softmax density is
exactly invariant to B(x) -> B(x) + b (F^T b is constant across the query axis),
so the MEAN of B is pure gauge. Spec 28's rank-1 claim used the UNCENTERED
second moment Sigma_B = E[B B^T]; a large mean mu_B makes Sigma_B look rank-1
(Sigma_B = Cov(B) + mu mu^T) even if the centered geometry Cov(B) is rich --
and the z(g)=normalize(B(g)) collapse might be mostly the mean, fixable by
centering alone. Also: the uncentered ridge encoder is NOT invariant to reward
shifts r -> r + beta (m_{r+beta} = m_r + beta*mu_B), while the centered encoder
is exactly invariant.

MEASURED HERE, per checkpoint, target tower primary / online scored too:
  A. ||mu_B|| vs mean ||B(u) - mu_B|| (is the mean dominant?).
  B. eff_rank + cond of Sigma_B (uncentered; CONTROL, must reproduce the
     archived ~1.0002) AND of C_B = Cov(B) (centered; the headline number).
  C. Index-set geometry over 4096 dataset goals (the P0a z_geometry convention,
     pairwise cosine + participation ratio) for four indices:
       point            normalize(Bbar(g))                     [control .998+]
       point_centered   normalize(Bbar(g) - mu_B)
       ridge            (Sigma_B + lam I)^-1 m_g               [control: de-collapsed]
       ridge_centered   (C_B + lam_c I)^-1 m_g^c
  D. P0b-style reward reconstruction for the two centered encodings alongside
     the archived point/moment/ridge, same cells, same scoring path; plus a
     DISJOINT SPLIT (fit mu/Sigma/C/m on even pool half, score on odd half) --
     the review's in-sample-ridge concern.
  E. Reward-shift probe: cos( enc(kg), enc(kg + 1) ) for moment / ridge /
     centered ridge. Centered must be 1.0 up to numerics.

CONTROL FIRST: the uncentered eff_rank and the point pairwise cosine must
reproduce the archived spec-28 numbers before any centered cell is read.

Run (CPU jax):
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec30_geom.py \\
    --run-dir <dir> --checkpoint-step 500000 --out results/spec30_geom/<name>.json
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
)

LAM_MULT = 1e-2  # the P0c-winning scale; lambda = LAM_MULT * tr(.)/d per matrix


# ------------------------------------------------------------------ geometry --

def spectrum_stats(mat_pool: np.ndarray, centered: bool) -> dict:
    """eff_rank (participation ratio of eigenvalues) + cond of Sigma or Cov."""
    b = np.asarray(mat_pool, np.float64)
    if centered:
        b = b - b.mean(axis=0, keepdims=True)
    s = b.T @ b / len(b)
    ev = np.clip(np.linalg.eigvalsh(s), 0.0, None)
    return {
        "eff_rank": float(ev.sum() ** 2 / max(float((ev ** 2).sum()), 1e-30)),
        "cond": float(ev[-1] / max(ev[0], 1e-12)),
        "eig_max": float(ev[-1]), "eig_min": float(ev[0]),
        "trace_over_d": float(ev.sum() / b.shape[1]),
    }


def index_set_geometry(z: np.ndarray, rng: np.random.Generator,
                       n_pairs: int = 20000) -> dict:
    """The P0a z_geometry convention (pairwise cosine of normalized rows,
    sphere reference) + participation ratio of the normalized index set."""
    z = np.asarray(z, np.float64)
    n, d = z.shape
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    cos = np.sum(zn[i] * zn[j], axis=1)
    ref = rng.normal(size=(min(n, 2048), d))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    ri = rng.integers(0, len(ref), 4000)
    rj = rng.integers(0, len(ref), 4000)
    rcos = np.sum(ref[ri] * ref[rj], axis=1)
    ev = np.clip(np.linalg.eigvalsh(zn.T @ zn / n), 0.0, None)
    return {"n_goals": int(n), "d": int(d),
            "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
            "pr_normalized": float(ev.sum() ** 2 / max(float((ev ** 2).sum()), 1e-30)),
            "sphere_ref_cos_mean": float(rcos.mean())}


def ridge_batch(b_pool: np.ndarray, kg_rows: np.ndarray, centered: bool) -> np.ndarray:
    """Per-goal ridge indices, one linear solve for all goals.
    Uncentered: (Sigma+lam I)^-1 E[kg B]. Centered: (Cov+lam I)^-1
    E[(B-mu)(kg-mean kg)] -- exactly shift-invariant in the reward.
    b_pool [P,d], kg_rows [N,P] -> [N,d]."""
    b = np.asarray(b_pool, np.float64)
    kg = np.asarray(kg_rows, np.float64)
    P, d = b.shape
    if centered:
        mu = b.mean(axis=0, keepdims=True)
        bc = b - mu
        m = (kg - kg.mean(axis=1, keepdims=True)) @ bc / P          # [N,d]
        s = bc.T @ bc / P
    else:
        m = kg @ b / P
        s = b.T @ b / P
    lam = LAM_MULT * float(np.trace(s)) / d
    return np.linalg.solve(s + lam * np.eye(d), m.T).T


# ---------------------------------------------------------------------- main --

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
    ap.add_argument("--n-geom-goals", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
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
    assert bool(cfg.get("measure_density", False)), f"{run_dir}: not measure_density"
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
    from gciql_flow import GCIQLFlowAgent

    # ---- pool: SAME rng stream order as p0b so numbers line up ----
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
    u_raw, b_online = build_pool(agent, blocks, pool_obs, spec)
    b_target = target_b_pool(agent, blocks, pool_obs)
    towers = {"target": np.asarray(b_target, np.float64),
              "online": np.asarray(b_online, np.float64)}
    print(f"[s30] {run_dir.name}: pool P={len(pool_obs)} d={d} "
          f"({time.time()-t00:.0f}s)", flush=True)

    res = {"meta": {"run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
                    "env": env_name, "d": d, "pool_size": int(len(pool_obs)),
                    "seed": args.seed, "lam_mult": LAM_MULT,
                    "probe": "spec30: centered-vs-uncentered B geometry "
                             "(review 4.4 mean-gauge test + 4.5 shift invariance). "
                             "EVAL ONLY, readout only."}}

    # ---- A + B: mean dominance and spectra, both towers ----
    geom = {}
    for tname, b in towers.items():
        mu = b.mean(axis=0)
        geom[tname] = {
            "mu_norm": float(np.linalg.norm(mu)),
            "centered_norm_mean": float(np.linalg.norm(b - mu[None], axis=1).mean()),
            "b_norm_mean": float(np.linalg.norm(b, axis=1).mean()),
            "uncentered": spectrum_stats(b, centered=False),
            "centered": spectrum_stats(b, centered=True),
        }
        geom[tname]["mu_dominance"] = (
            geom[tname]["mu_norm"] / max(geom[tname]["centered_norm_mean"], 1e-12))
    res["geometry"] = geom
    g_t = geom["target"]
    print(f"[s30] target: ||mu||={g_t['mu_norm']:.3f} "
          f"E||B-mu||={g_t['centered_norm_mean']:.3f} "
          f"eff_rank uncentered={g_t['uncentered']['eff_rank']:.4f} "
          f"CENTERED={g_t['centered']['eff_rank']:.2f}/{d}", flush=True)

    # ---- CONTROL GATE half 1: uncentered eff_rank must reproduce spec 28 ----
    ctrl_rank = g_t["uncentered"]["eff_rank"]

    # ---- C: index-set geometry over 4096 dataset goals (target tower) ----
    rng = np.random.default_rng(args.seed + 7)
    gsel = rng.choice(len(obs_all), size=min(args.n_geom_goals, len(obs_all)),
                      replace=False)
    goal_obs = obs_all[gsel]
    bg = np.asarray(target_b_pool(agent, blocks, goal_obs), np.float64)   # Bbar(g) [N,d]
    mu_t = towers["target"].mean(axis=0)

    # kernel rows for every geometry goal, chunked
    kg_rows = np.empty((len(goal_obs), len(pool_obs)), np.float64)
    for lo in range(0, len(goal_obs), 256):
        hi = min(lo + 256, len(goal_obs))
        kg_rows[lo:hi] = np.asarray(GCIQLFlowAgent._delta_phi(
            jnp.asarray(goal_obs[lo:hi], jnp.float32), u_raw, blocks), np.float64)
    print(f"[s30] kernel rows done ({time.time()-t00:.0f}s)", flush=True)

    idx_sets = {
        "point": bg,
        "point_centered": bg - mu_t[None],
        "ridge": ridge_batch(towers["target"], kg_rows, centered=False),
        "ridge_centered": ridge_batch(towers["target"], kg_rows, centered=True),
    }
    res["index_geometry"] = {k: index_set_geometry(v, np.random.default_rng(args.seed + 11))
                             for k, v in idx_sets.items()}
    for k, v in res["index_geometry"].items():
        print(f"[s30]   {k:16s} cos_mean={v['cos_mean']:.4f} "
              f"PR={v['pr_normalized']:.2f}/{d}", flush=True)

    # ---- CONTROL GATE half 2: point cosine must reproduce spec 28 P0a ----
    ctrl_cos = res["index_geometry"]["point"]["cos_mean"]
    gate = {"uncentered_eff_rank": ctrl_rank, "point_cos_mean": ctrl_cos,
            "pass": bool(ctrl_rank < 1.05 and ctrl_cos > 0.99),
            "rule": "uncentered eff_rank must reproduce ~1.0002 (<1.05) and the "
                    "point index pairwise cos must reproduce .998+ (>0.99) before "
                    "any centered number is read"}
    res["control_gate"] = gate
    print(f"[s30] CONTROL GATE: {'PASS' if gate['pass'] else 'FAIL'} "
          f"(eff_rank={ctrl_rank:.4f}, point cos={ctrl_cos:.4f})", flush=True)
    if not gate["pass"]:
        outp = Path(args.out) if args.out else \
            (REPO / "results" / "spec30_geom" / f"{run_dir.name}.json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(res, indent=2, default=float))
        print(f"[s30] STOPPED at control gate -> {outp}")
        return 3

    # ---- D: reconstruction cells (p0b protocol) + disjoint split ----
    cells = []
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        g_obs2 = np.asarray(info["goal"], np.float32)
        cells.append(("task", f"t{i}", g_obs2, spec.coords(g_obs2[None])[0]))
        env.close()
    rng_reg = np.random.default_rng(args.seed + 2)
    centers, cobs, _np_, _rj = draw_region_centers_with_obs(
        obs_all, coords_all, bounds, spec, relabel_coords, rng_reg,
        args.regions, min_pos=args.region_min_pos)
    for k_i in range(len(centers)):
        cells.append(("region", f"c{k_i}", np.asarray(cobs[k_i], np.float32),
                      np.asarray(centers[k_i], np.float64)))

    b = towers["target"]
    even, odd = np.arange(0, len(b), 2), np.arange(1, len(b), 2)
    lam_unc = ridge_lambdas(b, (LAM_MULT,))[LAM_MULT]
    res["cells"] = {}
    for kind, name, g_obs2, center in cells:
        kg = np.asarray(GCIQLFlowAgent._delta_phi(
            jnp.asarray(g_obs2, jnp.float32)[None], u_raw, blocks), np.float64)[0]
        hit = reward_conj_dirac(pool_coords, center, spec)
        dist = np.linalg.norm(spec.scaled(pool_coords)
                              - spec.scaled(np.atleast_2d(center))[0][None], axis=-1)
        zp = np.asarray(target_b_pool(agent, blocks, g_obs2[None]), np.float64)[0]
        enc = {
            "point": zp,
            "point_centered": zp - mu_t,
            "moment": analytic_zr(kg, b, d),
            "ridge": ridge_index(b, kg, lam_unc),
            "ridge_centered": ridge_batch(b, kg[None], centered=True)[0],
        }
        cell = {"kind": kind, "n_hit": int(hit.sum()),
                "insample": {k: score_encoding(b, z, kg, hit, dist)
                             for k, z in enc.items()}}
        # disjoint split: fit on even half, score on odd half
        enc_fit = {
            "ridge_split": ridge_batch(b[even], kg[even][None], centered=False)[0],
            "ridge_centered_split": ridge_batch(b[even], kg[even][None], centered=True)[0],
        }
        cell["outsample"] = {k: score_encoding(b[odd], z, kg[odd], hit[odd], dist[odd])
                             for k, z in enc_fit.items()}
        # in-sample scored on the SAME odd half for a like-for-like delta
        cell["outsample"]["ridge_full_on_odd"] = score_encoding(
            b[odd], enc["ridge"], kg[odd], hit[odd], dist[odd])
        res["cells"][name] = cell
        ins = cell["insample"]
        print(f"[s30] {name:4s} AUC point={ins['point']['auc']} "
              f"pointC={ins['point_centered']['auc']} ridge={ins['ridge']['auc']} "
              f"ridgeC={ins['ridge_centered']['auc']} | out ridge="
              f"{cell['outsample']['ridge_split']['auc']} "
              f"ridgeC={cell['outsample']['ridge_centered_split']['auc']}", flush=True)

    # aggregate
    agg = {}
    for e in ["point", "point_centered", "moment", "ridge", "ridge_centered"]:
        aucs = [c["insample"][e]["auc"] for c in res["cells"].values()
                if c["insample"][e]["auc"] is not None]
        rhos = [c["insample"][e]["spearman_kg"] for c in res["cells"].values()]
        agg[e] = {"auc_mean": float(np.mean(aucs)) if aucs else None,
                  "spearman_mean": float(np.mean(rhos))}
    for e in ["ridge_split", "ridge_centered_split", "ridge_full_on_odd"]:
        aucs = [c["outsample"][e]["auc"] for c in res["cells"].values()
                if c["outsample"][e]["auc"] is not None]
        rhos = [c["outsample"][e]["spearman_kg"] for c in res["cells"].values()]
        agg[e] = {"auc_mean": float(np.mean(aucs)) if aucs else None,
                  "spearman_mean": float(np.mean(rhos))}
    res["aggregate"] = agg

    # ---- E: reward-shift probe on the task cells ----
    shift = {}
    for kind, name, g_obs2, _c in cells:
        if kind != "task":
            continue
        kg = np.asarray(GCIQLFlowAgent._delta_phi(
            jnp.asarray(g_obs2, jnp.float32)[None], u_raw, blocks), np.float64)[0]
        kg1 = kg + 1.0

        def _cos(a_, b_):
            return float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-30))
        shift[name] = {
            "moment": _cos(moment_index(b, kg), moment_index(b, kg1)),
            "ridge": _cos(ridge_index(b, kg, lam_unc), ridge_index(b, kg1, lam_unc)),
            "ridge_centered": _cos(ridge_batch(b, kg[None], centered=True)[0],
                                   ridge_batch(b, kg1[None], centered=True)[0]),
        }
    res["shift_probe"] = shift
    sm = {k: float(np.mean([v[k] for v in shift.values()]))
          for k in ["moment", "ridge", "ridge_centered"]}
    res["shift_probe"]["mean"] = sm
    print(f"[s30] shift cos(enc(r),enc(r+1)): moment={sm['moment']:.4f} "
          f"ridge={sm['ridge']:.4f} ridge_centered={sm['ridge_centered']:.6f}",
          flush=True)

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec30_geom" / f"{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, default=float))
    print(f"[s30] DONE ({time.time()-t00:.0f}s) -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
