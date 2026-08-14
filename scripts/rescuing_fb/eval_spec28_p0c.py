#!/usr/bin/env python
"""scripts/rescuing_fb/eval_spec28_p0c.py -- spec 28 P0c: the e2b assignment
protocol re-scored with the RIDGE (dual) index, READOUT ONLY.

EVAL ONLY. No training, no training code path touched, output confined to
results/spec28_p0c/.

WHAT IS BEING ASKED. Spec 27's sharpest result: e2b's argmax over candidate
indices lands at a coordinate percentile of .35 (z25 actor) / .47 (z25 full)
against a .50 null -- reward -> index assignment at chance, even though exact
matches exist in the candidate set. Spec 28's mechanism says why: the assignment
compares a reward MOMENT (primal) against candidate indices, when the readout it
must agree with needs a dual COEFFICIENT `c = (Sigma_B+lambda I)^{-1} m`. So
re-run the same protocol, on the same candidate set, with the same metric, and
change ONLY the index the argmax is taken over.

METRIC -- the coordinate percentile, NOT policy success. These checkpoints'
actors sit at the BC floor, so every rollout-terminated metric reads ~0 for
every arm regardless of readout quality; that trap has cost this program three
rounds (spec 27 sec.6.5), so this script performs NO ROLLOUTS AT ALL. The metric
is `spec28_encodings.coord_percentile` (fraction of candidates strictly closer
to the true reward center than the selected one; 0 = best, .50 = chance),
aggregated as the mean of the per-family means -- the definition pinned by
reproducing the archived .35/.47 exactly from the archived e2b JSONs.

ARMS (one selection rule each, over an IDENTICAL candidate set = the leading
--e2b_candidate_size pool observations, exactly as the archived e2b run):
  readout   the archived rule: `decode_goal_retrieval_density` scoring each
            candidate index by this readout's own PRIMARY term
            (w - kappa(x0)).r(U_P), retrieval_k=1. On a z checkpoint the
            candidates are z_j = `_density_z_encode(u_j)` riding the cand_obs
            slot through `_ZIndexView` (imported from the audited spec-24
            harness); on a g checkpoint they are the goal observations u_j
            themselves -- i.e. the same arm gives the G-INDEXED CONTROL
            reference percentile on the same candidate set.
  moment    argmax_j <z_j, m_r>, m_r = E_pool[r(u) B~(u)] -- the PRIMAL index
            spec 24's analytic assignment uses (`analytic_zr` up to a positive
            scalar, which cannot move an argmax).
  ridge_M   argmax_j <z_j, c_r>, c_r = (Sigma_B + M*tr(Sigma)/d * I)^{-1} m_r --
            the DUAL correction. THIS IS READOUT ONLY: c_r is never fed to F or
            to the actor (it is not norm-preserving, and ||z||=sqrt(d) holds
            wherever z enters either; conditioning on it is out of distribution
            for these checkpoints and would need retraining -- spec 28 sec.0).
  *_braw    the same two rules with the RAW embedding B~(u_j) in place of the
            sphere-projected index z_j: <B~(u_j), c_r> IS the reward
            reconstruction rhat(u_j), so its argmax asks "where does the
            reconstructed reward peak" rather than "which index matches".

CONTROL GATE (mandatory, spec 28 P0c / the three-times-burned rule). With
--gate-against <archived e2b json> the `readout` arm's per-cell `top1_idx` must
equal the archived run's EXACTLY, cell for cell -- a bit-exact reproduction of
the archived assignment, not a tolerance. No ridge number is read unless it
passes; the verdict lands in the output JSON and a failure exits non-zero.

Run (CPU jax; ~5 min per checkpoint, no rollouts):
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec28_p0c.py \\
    --run-dir ~/exp_spec25_stage1/factored-fb/*Full-d50/z25_cs_full \\
    --checkpoint-step 500000 --pool_size 16384 --seed 0 \\
    --gate-against results/spec27_q2/e2b_z25_cs_full.json \\
    --out results/spec28_p0c/z25_cs_full.json
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

from eval_density_zeroshot import build_pool, decode_goal_retrieval_density  # noqa: E402
from eval_density_z_rewards import _ZIndexView, target_b_pool, z_candidates  # noqa: E402
from eval_partition_zeroshot import draw_region_centers_with_obs  # noqa: E402
from eval_zeroshot_rewards import (  # noqa: E402
    eval_tasks_for,
    reward_conj_dense,
    reward_conj_dirac,
)
from env_geometry import episode_bounds, geometry_for  # noqa: E402
from spec28_encodings import (  # noqa: E402
    coord_percentile,
    family_mean_aggregate,
    moment_index,
    ridge_index,
    ridge_lambdas,
)

RIDGE_MULTS = (1e-3, 1e-2, 1e-1)


def gate_top1_exact(mine: dict, archived: dict):
    """The P0c control gate: every cell present in BOTH must have the SAME
    `top1_idx`. Bit-exact by construction (an integer index), no tolerance.
    `mine`/`archived` map "fam/cell" -> int. Returns (ok, rows)."""
    rows, ok = [], True
    for key in sorted(set(mine) & set(archived)):
        good = int(mine[key]) == int(archived[key])
        ok = ok and good
        rows.append({"cell": key, "mine": int(mine[key]),
                     "archived": int(archived[key]), "ok": good})
    assert rows, "gate_top1_exact: no comparable cells"
    return ok, rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--families", default="dirac,dense,region")
    ap.add_argument("--regions", type=int, default=4)
    ap.add_argument("--region-min-pos", type=int, default=5)
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--pool_size", type=int, default=16384)
    ap.add_argument("--e2b_candidate_size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-readout", action="store_true",
                    help="skip the (expensive, actor-touching) readout arm; the "
                         "moment/ridge arms are pure linear algebra")
    ap.add_argument("--gate-against", default="",
                    help="archived e2b JSON whose per-cell top1_idx the readout "
                         "arm must reproduce EXACTLY")
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])
    assert bool(cfg.get("measure_density", False)), f"{run_dir} is not measure_density"
    is_z = bool(cfg.get("measure_density_z", False))
    env_name = flags["env_name"]
    d = int(cfg["measure_density_dim"])
    temp = float(cfg.get("measure_density_temp", 1.0))
    double_q = bool(cfg.get("double_q", True))
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
    u_raw, b_pool = build_pool(agent, blocks, pool_obs, spec)
    b_target = target_b_pool(agent, blocks, pool_obs)

    n_cand = min(args.e2b_candidate_size, len(pool_obs))
    cand_obs_raw = pool_obs[:n_cand]
    cand_coords = coords_all[pool_idx[:n_cand]]
    cand_coords_sc = spec.scaled(cand_coords)
    z_cand = z_candidates(agent, cand_obs_raw).astype(np.float64)       # [N,d] sphere
    b_cand = b_target[:n_cand]                                          # [N,d] raw B~
    zview = _ZIndexView(agent)
    lams = ridge_lambdas(b_target, RIDGE_MULTS)
    print(f"[p0c] {run_dir.name}: P={len(pool_obs)} N_cand={n_cand} d={d} "
          f"is_z={is_z} lambdas={ {f'{k:g}': round(v, 6) for k, v in lams.items()} } "
          f"({time.time()-t00:.0f}s)", flush=True)

    # ---- cells: the archived e2b cell set (5 tasks x dirac/dense + regions) ----
    goal_coord, x0_by_task = {}, {}
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        o0, info = env.reset(seed=0)
        goal_coord[i] = spec.coords(np.asarray(info["goal"], np.float32)[None])[0]
        x0_by_task[i] = np.asarray(o0, np.float32)
        env.close()
    rng_reg = np.random.default_rng(args.seed + 2)
    centers, _cobs, npos_c, rejected = draw_region_centers_with_obs(
        obs_all, coords_all, bounds, spec, relabel_coords, rng_reg,
        args.regions, min_pos=args.region_min_pos)

    cells = []   # (family, name, center_coord, r_pool, x0_obs)
    if "dirac" in families:
        for i in task_ids:
            cells.append(("dirac", f"t{i}", goal_coord[i],
                          reward_conj_dirac(pool_coords, goal_coord[i], spec),
                          x0_by_task[i]))
    if "dense" in families:
        for i in task_ids:
            cells.append(("dense", f"t{i}", goal_coord[i],
                          reward_conj_dense(pool_coords, goal_coord[i], spec),
                          x0_by_task[i]))
    if "region" in families:
        v0 = task_ids[0]
        for k_i, c in enumerate(centers):
            cells.append(("region", f"c{k_i}", np.asarray(c, np.float64),
                          reward_conj_dirac(pool_coords, c, spec), x0_by_task[v0]))

    arm_names = ["moment", "moment_braw"] \
        + [f"ridge_{m:g}" for m in RIDGE_MULTS] \
        + [f"ridge_{m:g}_braw" for m in RIDGE_MULTS]
    if not args.skip_readout:
        arm_names = ["readout"] + arm_names

    per_cell = {a: {} for a in arm_names}          # arm -> family -> cell -> pct
    for a in arm_names:
        per_cell[a] = {}
    detail, top1_mine = {}, {}
    for fam, name, center, r_pool, x0_obs in cells:
        sel = {}
        if not args.skip_readout:
            _chosen_obs, info = decode_goal_retrieval_density(
                zview if is_z else agent, x0_obs, u_raw, b_pool, r_pool, blocks,
                temp, double_q, cand_coords,
                z_cand.astype(np.float32) if is_z else cand_obs_raw,
                retrieval_k=1, rng_seed=args.seed)
            sel["readout"] = int(info["top1_idx"])
            top1_mine[f"{fam}/{name}"] = int(info["top1_idx"])
        m_r = moment_index(b_target, r_pool)
        sel["moment"] = int(np.argmax(z_cand @ m_r))
        sel["moment_braw"] = int(np.argmax(b_cand @ m_r))
        for mult, lam in lams.items():
            c_r = ridge_index(b_target, r_pool, lam)
            sel[f"ridge_{mult:g}"] = int(np.argmax(z_cand @ c_r))
            sel[f"ridge_{mult:g}_braw"] = int(np.argmax(b_cand @ c_r))
        ctr_sc = spec.scaled(np.atleast_2d(center))[0]
        rec = {"family": fam, "center_coord": np.asarray(center).tolist(),
               "n_hit_pool": int(np.sum(r_pool >= 1.0)) if fam != "dense" else None,
               "sel": {}, "pct": {}}
        for a, j in sel.items():
            p = coord_percentile(cand_coords_sc, ctr_sc, j)
            per_cell[a].setdefault(fam, {})[name] = p
            rec["sel"][a] = j
            rec["pct"][a] = p
        detail[f"{fam}/{name}"] = rec
        print(f"[p0c] {fam:6s} {name:4s} pct: "
              + " ".join(f"{a}={rec['pct'][a]:.3f}" for a in arm_names)
              + f"  ({time.time()-t00:.0f}s)", flush=True)

    agg = {a: family_mean_aggregate(per_cell[a]) for a in arm_names}
    agg_flat = {a: float(np.mean([v for fam in per_cell[a].values()
                                  for v in fam.values()])) for a in arm_names}

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "d": d, "is_z_checkpoint": is_z, "seed": args.seed,
        "pool_size": int(len(pool_obs)), "e2b_candidate_size": n_cand,
        "relabel_size": int(len(relabel_coords)), "temp": temp,
        "double_q": double_q, "regions": int(len(centers)),
        "regions_rejected": int(rejected),
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
        "ridge_lambda": {f"{k:g}": v for k, v in lams.items()},
        "metric": "coordinate percentile = fraction of the 4096 candidates "
                  "strictly closer (eps-scaled coords) to the true reward center "
                  "than the selected candidate. 0 = best, 0.50 = chance, LOWER IS "
                  "BETTER. Aggregate = mean of per-family means (the archived "
                  "spec-27 convention, pinned by reproducing .35/.47).",
        "readout_only": "the ridge index is used ONLY to rank candidates; it is "
                        "never fed to F or the actor (not norm-preserving; "
                        "||z||=sqrt(d) is required wherever z conditions either). "
                        "No rollouts anywhere in this script.",
        "spec": "PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md P0c",
    }, "aggregate_family_mean": agg, "aggregate_flat_mean": agg_flat,
        "per_family": {a: {f: float(np.mean(list(v.values())))
                           for f, v in per_cell[a].items()} for a in arm_names},
        "cells": detail}

    if args.gate_against and not args.skip_readout:
        arch = json.loads(Path(args.gate_against).read_text())
        arch_top1 = {}
        for fam, F in arch["families"].items():
            for k, v in F.items():
                if isinstance(v, dict) and "decode" in v:
                    arch_top1[f"{fam}/{k}"] = int(v["decode"]["top1_idx"])
        ok, rows = gate_top1_exact(top1_mine, arch_top1)
        results["meta"]["control_gate"] = {
            "against": str(args.gate_against), "kind": "bit-exact top1_idx",
            "pass": bool(ok), "n_cells": len(rows), "rows": rows}
        print(f"[p0c] CONTROL GATE vs {args.gate_against}: "
              f"{'PASS' if ok else 'FAIL'} ({sum(r['ok'] for r in rows)}/{len(rows)} "
              f"cells bit-exact)", flush=True)
        for r in rows:
            if not r["ok"]:
                print(f"[p0c]   MISMATCH {r['cell']}: mine={r['mine']} "
                      f"archived={r['archived']}", flush=True)

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec28_p0c" / f"{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[p0c] DONE ({time.time()-t00:.0f}s) -> {outp}")
    for a in arm_names:
        print(f"[p0c]   {a:16s} pct(family-mean)={agg[a]:.3f} "
              f"flat={agg_flat[a]:.3f}", flush=True)
    if args.gate_against and not args.skip_readout \
            and not results["meta"]["control_gate"]["pass"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
