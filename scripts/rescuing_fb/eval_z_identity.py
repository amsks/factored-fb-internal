#!/usr/bin/env python
"""scripts/rescuing_fb/eval_z_identity.py -- spec 24 follow-up (coordinator,
2026-08-13): an ACTOR-INDEPENDENT test of whether z(g) =
`agent._density_z_encode(g)` identifies g among a pool of dataset goal
configurations, on cube-double.

Context: G4 (eval_measure_g4_spearman.py) showed the z-indexed measure is
INTERNALLY CALIBRATED (spearman_tie .759-.807, matching/beating the g-indexed
control's .739) -- but G4's MC target rolls the agent's OWN policy, so it is
a SELF-CONSISTENCY test (the measure accurately predicts the occupancy of
the policy it trained alongside), not proof a GOAL-DIRECTED policy is
extractable from the measure. The live suspect for why extraction (E1 .017,
E2 ~0) still fails despite a calibrated measure: the actor maximizes
r_z = B(u)^T z. If z(g) is a LOSSY encoding of g (many distinct
configurations map to similar z), maximizing r_z does not correspond to
reaching g even though the measure and the policy still agree with each
other about where the policy goes -- exactly the observed picture
(calibrated, mobile, not goal-directed).

This script tests z(g)'s discriminability DIRECTLY: no actor, no rollout, no
readout dot-product. Take N query goals from the SAME 16,384-pool convention
`eval_density_z_rewards.py`/`eval_measure_g4_spearman.py` use; encode with
the agent's OWN `_density_z_encode` (target_density_b + sphere-normalize,
called live -- the exact transform E1/E2's z(g) used, NOT reimplemented);
nearest-neighbor search the REST of the pool in z-space (cosine similarity,
the natural metric on a sqrt(d)-sphere); check whether the top-1 / top-8
nearest OTHER pool points' COORDINATES fall inside the query's own hit-spec
eps-ball (`env_geometry.HitSpec.contains`, mult=1.0 -- the SAME
conjunctive-tolerance convention success/region membership use everywhere
else in this repo) -- "close enough to count as the same goal", not an
exact-row match (continuous coordinates have no exact duplicates in
general; the eps-ball IS what every other part of this project treats as
goal-equivalent). "Nearest WRONG goal" = the first ranked OTHER pool point
whose coordinate falls OUTSIDE that eps-ball; its cosine to z(g) is
reported -- a high value means the encoding places a genuinely different
goal close in z-space even when top-1/top-8 happen to succeed.

CONTROL (same run, `--run-dir` pointed at a g-indexed checkpoint):
`_density_z_encode` is architecture-generic -- it only reads
`measure_density_dim`/`measure_hit_spec` from config and calls
`target_density_b`, BOTH of which exist on any `measure_density=True`
checkpoint regardless of whether F/the actor were ever trained to CONDITION
on that vector (only `measure_density_z=True` checkpoints did). Running this
script unmodified on `d18x_cd_dens_s0` is therefore a genuine
apples-to-apples "for scale" baseline: does the backward tower's
sphere-normalized embedding support goal identification AT ALL,
independent of whether z-conditioning was ever trained into F, or is poor
retrieval specific to the z-indexed training run.

Run (CPU-only; GPU training jobs are live on the box):
  taskset -c 24-39 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \\
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=4 \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_z_identity.py \\
    --run-dir '/home/adityamohan/exp_spec24/factored-fb/GCIQL-CubeDouble-Flow-Measure-DensityZ-d50/z24_cd_z_s0' \\
    --checkpoint-step 500000 --pool-size 16384 --n-queries 500 --seed 0
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

# Reused, not reimplemented: the base measure_density consistency refusal (this
# script deliberately does NOT require/refuse measure_density_z either way --
# it is meant to run symmetrically on z-indexed AND g-indexed checkpoints, see
# module docstring "CONTROL") and the shared task-family goal helper.
from eval_density_zeroshot import DensityMismatchError, check_density_consistency  # noqa: E402
from eval_zeroshot_rewards import eval_tasks_for  # noqa: E402
from env_geometry import HitSpec  # noqa: E402

CUBE_DOUBLE_ENV_NAME = "cube-double-play-v0"


# ======================= retrieval algebra (pure numpy, unit-tested) ===============

def retrieval_accuracy(z_pool: np.ndarray, coords_pool: np.ndarray, query_idx,
                       spec: HitSpec, k_list=(1, 8)) -> dict:
    """Self-excluded cosine-NN retrieval in z-space, scored against
    coordinate eps-ball membership (spec.contains, mult=1.0).

    z_pool: [P, d] -- need not be pre-normalized (cosine computed explicitly,
      not assumed from a sqrt(d)-sphere convention, so this is robust to any
      encoder). coords_pool: [P, d_c] hit-spec coordinates of the SAME pool
      rows. query_idx: int array of indices INTO the pool -- queries ARE pool
      members and are excluded from their own candidate search (searching a
      deterministic embedding against itself would trivially "retrieve"
      itself at cosine 1.0, which tests nothing).

    Returns {"topK": hit_rate for K in k_list, "nearest_wrong_cosine_mean":
    float, "n_queries": int} -- pure numpy, no jax, no env."""
    z_pool = np.asarray(z_pool, np.float64)
    coords_pool = np.asarray(coords_pool, np.float64)
    query_idx = np.asarray(query_idx, int)
    assert z_pool.ndim == 2 and coords_pool.ndim == 2 and len(z_pool) == len(coords_pool), \
        f"z_pool {z_pool.shape} / coords_pool {coords_pool.shape} shape mismatch"
    assert len(query_idx) > 0, "no queries"
    norms = np.linalg.norm(z_pool, axis=-1, keepdims=True)
    z_unit = z_pool / (norms + 1e-12)
    max_k = max(k_list)
    hits = {k: [] for k in k_list}
    nearest_wrong_cos = []
    for qi in query_idx:
        q_unit = z_unit[qi]
        cos = z_unit @ q_unit                          # [P]
        cos = cos.copy()
        cos[qi] = -np.inf                               # exclude self
        order = np.argsort(-cos)                        # descending similarity
        true_coord = coords_pool[qi]
        is_match_all = spec.contains(coords_pool[order], true_coord, mult=1.0)
        for k in k_list:
            hits[k].append(bool(is_match_all[:k].any()))
        wrong_positions = np.flatnonzero(~is_match_all)
        if len(wrong_positions):
            nearest_wrong_cos.append(float(cos[order[int(wrong_positions[0])]]))
    out = {f"top{k}": float(np.mean(hits[k])) for k in k_list}
    out["nearest_wrong_cosine_mean"] = (float(np.mean(nearest_wrong_cos))
                                        if nearest_wrong_cos else None)
    out["n_queries"] = int(len(query_idx))
    out["n_queries_with_a_wrong_candidate"] = len(nearest_wrong_cos)
    return out


# ======================= live-agent integration (jax-dependent) ====================

def encode_pool_z(agent, pool_obs: np.ndarray) -> np.ndarray:
    """z(x) for every pool observation via the agent's OWN `_density_z_encode`
    (E1/E2's exact transform, target_density_b + sphere, called live -- see
    module docstring). One batched call. Returns [P, d] float64."""
    import jax.numpy as jnp

    z = agent._density_z_encode(jnp.asarray(pool_obs, jnp.float32))
    return np.asarray(z, np.float64)


def run_one_checkpoint(args) -> dict:
    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])

    ckpt_is_density = bool(cfg.get("measure_density", False))
    for _w in check_density_consistency(ckpt_is_density, True, force=args.force,
                                        run_dir=str(run_dir)):
        print(f"[zid] {_w}", flush=True)

    env_name = flags["env_name"]
    assert env_name == CUBE_DOUBLE_ENV_NAME, \
        f"this script is scoped to {CUBE_DOUBLE_ENV_NAME!r} (spec 24 coordinator " \
        f"follow-up); got {env_name!r}"
    is_density_z = bool(cfg.get("measure_density_z", False))
    d = int(cfg["measure_density_dim"])
    spec = HitSpec.parse(cfg["measure_hit_spec"])

    t00 = time.time()
    eval_tasks = eval_tasks_for(env_name)
    _e0, train, _val = ogbench.make_env_and_datasets(eval_tasks[1])
    obs_dim = int(np.asarray(train["observations"]).shape[-1])
    act_dim = int(np.asarray(train["actions"]).shape[-1])

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)

    print(f"[zid] restored {run_dir.name} step={args.checkpoint_step} "
          f"env={env_name} d={d} hit_spec={cfg.get('measure_hit_spec')!r} "
          f"measure_density_z={is_density_z} pool_size={args.pool_size} "
          f"n_queries={args.n_queries}", flush=True)

    rng = np.random.default_rng(args.seed)
    obs_all = np.asarray(train["observations"], np.float32)
    n_rows = len(obs_all)
    pool_idx = rng.choice(n_rows, size=min(args.pool_size, n_rows), replace=False)
    pool_obs = obs_all[pool_idx]
    coords_pool = spec.coords(pool_obs)

    z_pool = encode_pool_z(agent, pool_obs)
    print(f"[zid] pool encoded: P={len(pool_obs)} ({time.time()-t00:.0f}s)", flush=True)

    query_idx = rng.choice(len(pool_obs), size=min(args.n_queries, len(pool_obs)),
                           replace=False)
    result = retrieval_accuracy(z_pool, coords_pool, query_idx, spec, k_list=(1, 8))
    print(f"[zid] {run_dir.name}: top1={result['top1']:.3f} top8={result['top8']:.3f} "
          f"nearest_wrong_cosine_mean={result['nearest_wrong_cosine_mean']} "
          f"n={result['n_queries']} ({time.time()-t00:.0f}s)", flush=True)

    return {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "measure_density_z": is_density_z, "d": d,
        "hit_spec": cfg.get("measure_hit_spec"), "seed": args.seed,
        "pool_size": len(pool_obs), "n_queries": result["n_queries"],
        "protocol": {
            "encode": "agent._density_z_encode(pool_obs) -- target_density_b + "
                     "sphere-normalize, called live, the EXACT transform E1/E2's "
                     "z(g) used (not reimplemented); architecture-generic, so this "
                     "runs UNCHANGED on g-indexed checkpoints as a for-scale control",
            "retrieval": "cosine-NN over the SAME pool (self excluded); top-K = "
                        "any of the K nearest OTHER pool points' coordinate falls "
                        "inside the query's own hit-spec eps-ball "
                        "(env_geometry.HitSpec.contains, mult=1.0); "
                        "nearest_wrong_cosine_mean = cosine to the first ranked "
                        "candidate OUTSIDE that eps-ball, averaged over queries",
            "spec": "coordinator follow-up 2026-08-13, spec 24 E1/E2/G4 chain",
        },
        "result": result,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--pool-size", type=int, default=16384)
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="override the measure_density consistency refusal")
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    res = run_one_checkpoint(args)
    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec24_eval" / f"zid_{Path(args.run_dir).name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, default=lambda o: float(o)))
    print(f"\n[zid] DONE -> {outp}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
