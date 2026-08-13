#!/usr/bin/env python
"""scripts/rescuing_fb/eval_density_z_rewards.py -- spec 24 sec.3 E2:
reward-to-index assignment for `measure_density_z` checkpoints, VERSION 2
(2026-08-13 rewrite; PAPER/Multistep-FB/specs/24-z-indexed-density.md sec.3-4).

WHY A REWRITE (root cause of v1, whose artifacts are kept in
results/spec24_eval/ without a _v2 suffix): v1 re-derived its own protocol
instead of wrapping the audited g-indexed harness, and its g-indexed CONTROL
therefore disagreed with the archived reference numbers
(results/xenv_eval/dzs_cd_dens_s0_RET_k1.json) far beyond tolerance -- a
broken instrument, so none of its z numbers are licensed. Specifically v1
(a) replaced the audited success metric (env-native `info['success']` for the
dirac/dense families) with `region_success_from_obs` everywhere, (b) decoded
the control's goal from a 4096-candidate draw at retrieval_k=8 instead of the
audited 50000-candidate retrieval_k=1 protocol, (c) consumed the numpy rng in
a different order (pool before relabel), so its region centers were not the
archived ones, and (d) on point-mass-maze the final-state-in-a-.03-ball
metric reads 0.000 for EVERY arm structurally.

STRUCTURE OF V2 (the fix is structural, not a patch): this script is a THIN
WRAPPER over scripts/rescuing_fb/eval_density_zeroshot.py -- the audited
spec-18 harness that produced the archived control numbers. It imports and
calls that module's `build_pool`, `density_query`, `compute_density_primary`
/ `compute_density_secondary`, `decode_goal_retrieval_density`, and
`batched_goal_rollout_density` end-to-end, mirrors its main()'s rng stream
ORDER exactly (relabel draw of --relabel-size FIRST, then the pool draw), its
reward-family builders (`reward_conj_dirac`/`reward_conj_dense` on raw
hit-spec coords), its region-center draw (`draw_region_centers_with_obs`
under default_rng(seed+2) on the relabel sample), and its success accounting
(env-native success for dirac/dense; `region_success_from_obs` final/any for
region). The ONLY thing the three arms change is the INDEX-SELECTION step:

  control  g-indexed retrieval decode, `decode_goal_retrieval_density` over
           the --relabel-size dataset sample at --retrieval-k (default 1,
           the archived RET_k1 gate protocol) -- byte-identical to the
           audited harness's own run_cell minus the pool_sweep diagnostics
           (which never touch the decode or the rollout). Requires a
           measure_density_z=False checkpoint (`check_control_consistency`).
           With --gate-against <archived.json> the family means are compared
           against the archived reference within --gate-tol (default 0.02)
           and the PASS/FAIL verdict lands in the output JSON -- the spec-24
           HARD ACCEPTANCE GATE: no z number is meaningful unless this arm
           reproduces the archived control.
  e2a      analytic assignment (spec 24 sec.3):
           z_r = sqrt(d) * normalize( E_pool[ r(x) * B_bar(x_tilde) ] ).
           B_bar(x_tilde) comes from the agent's OWN code path
           (`_density_queries` eps-scaling -> `target_density_b` target
           tower, i.e. exactly `_density_z_encode`'s pipeline before its
           `_sphere` projection), and the sqrt(d)-sphere projection IS
           gciql_flow's `_sphere` called live -- nothing about scaling or
           normalization is reimplemented here. A pure-numpy reference
           (`analytic_zr`, unit-tested) cross-checks the live value at
           runtime and hard-fails on disagreement.
  e2b      search assignment (spec 24 sec.3): candidate indices
           {z(u_j)} = `agent._density_z_encode(u_j)` for the first
           --e2b_candidate_size observations OF THE READOUT POOL ITSELF (a
           uniform random subset, since the pool is a uniform draw -- no
           extra rng stream), scored from the initial state by the STANDARD
           pool readout via `decode_goal_retrieval_density` itself: the
           z-candidates ride in that function's `cand_obs` slot, so the
           candidate actions (`sample_actions` -> here `sample_actions_with_z`
           through `_ZIndexView`), the F/B scoring, and the
           (w - kappa(x0)).r(U_P) ranking are the SAME audited code v1 was
           supposed to use. retrieval_k is forced to 1 for this arm (spec
           says argmax; coordinate-averaging over z candidates' block coords
           at k>1 would be a different, unspecified decode).

Both z arms then roll out `batched_goal_rollout_density` UNCHANGED, with the
chosen z riding in the `g_hat_obs` slot through `_ZIndexView`, a 5-line
adapter whose only job is to route the goals slot of `sample_actions` into
the agent's own `sample_actions_with_z` (gciql_flow.py:2843 -- the entry
point for a PRECOMPUTED z; plain `sample_actions` would try to
`_density_z_encode` the z as if it were a goal observation,
gciql_flow.py:2823). The t=0 readout inside the rollout is also correct
under the view with zero changes: for a measure_density_z checkpoint,
`density_f`'s input IS [obs, action, z] (width d, gciql_flow.py:1020), which
is exactly what `density_query` concatenates when the goals slot carries z.

CONSISTENCY REFUSAL (unchanged from v1; unit-tested): the e2a/e2b arms
refuse a checkpoint without measure_density_z=True; the control arm refuses
one WITH it; both inherit the base measure_density=True refusal. --force
overrides loudly, never silently.

Run (CPU only -- big pools OOM the 23G GPUs; long runs via nohup):
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_density_z_rewards.py \\
    --run-dir /home/adityamohan/exp_spec24/factored-fb/GCIQL-CubeDouble-Flow-Measure-DensityZ-d50/z24_cd_z_s0 \\
    --checkpoint-step 500000 --arm e2a --pool_size 16384 --episodes 25 --seed 0 \\
    --out results/spec24_eval/cd_e2a_s0_v2.json
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

# The audited harness this script wraps. Reused, never reimplemented.
from eval_density_zeroshot import (  # noqa: E402
    DensityMismatchError,
    batched_goal_rollout_density,
    build_pool,
    check_density_consistency,
    decode_goal_retrieval_density,
)
from eval_partition_zeroshot import draw_region_centers_with_obs  # noqa: E402
from eval_zeroshot_rewards import (  # noqa: E402
    eval_tasks_for,
    region_success_from_obs,
    reward_conj_dense,
    reward_conj_dirac,
)
from env_geometry import episode_bounds, geometry_for  # noqa: E402

FAMILIES = ("dirac", "dense", "region")
ARMS = ("control", "e2a", "e2b")


# ======================= consistency refusal (unit-tested, jax-free) ===============

class DensityZMismatchError(DensityMismatchError):
    """The checkpoint's recorded `measure_density_z` config disagrees with the
    requested arm (spec 24: the e2a/e2b readouts need the z-wide F/actor
    architecture; the control arm needs the g-indexed one). Subclasses
    DensityMismatchError so callers catching the base class still catch this."""


def check_density_z_consistency(ckpt_is_density: bool, ckpt_is_density_z: bool,
                                force: bool = False, run_dir: str = "<run>") -> list:
    """E2 arms: base measure_density refusal (reused) PLUS a refusal if
    `measure_density_z` is not set. Returns warning strings; raises
    DensityZMismatchError on an un-forced mismatch."""
    warnings = list(check_density_consistency(ckpt_is_density, True, force=force,
                                              run_dir=run_dir))
    if not ckpt_is_density_z:
        msg = (f"checkpoint {run_dir} was NOT trained with measure_density_z=True "
               "(spec 24) -- the e2a/e2b arms assume the z-conditioned density_f/actor "
               "architecture (F/actor conditioning width = measure_density_dim, not "
               "obs width); a g-indexed checkpoint here would crash on a shape "
               "mismatch or silently score the wrong architecture.")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are LIKELY GARBAGE): " + msg)
        else:
            raise DensityZMismatchError(msg + " Pass --force to override (not recommended).")
    return warnings


def check_control_consistency(ckpt_is_density: bool, ckpt_is_density_z: bool,
                              force: bool = False, run_dir: str = "<run>") -> list:
    """Control arm: the inverse z check -- refuses a checkpoint that WAS
    trained with measure_density_z=True (running it through the g-indexed
    decode would not be a control at all)."""
    warnings = list(check_density_consistency(ckpt_is_density, True, force=force,
                                              run_dir=run_dir))
    if ckpt_is_density_z:
        msg = (f"checkpoint {run_dir} WAS trained with measure_density_z=True -- the "
               "control arm exists to reproduce the archived g-indexed reference "
               "numbers and requires measure_density_z absent/False.")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are LIKELY GARBAGE): " + msg)
        else:
            raise DensityZMismatchError(msg + " Pass --force to override (not recommended).")
    return warnings


# ======================= pure logic (unit-tested, jax-free) ========================

def analytic_zr(r_pool: np.ndarray, b_pool_target: np.ndarray, d: int) -> np.ndarray:
    """Numpy REFERENCE for E2a's z_r = sqrt(d)*normalize(E_pool[r(x) B_bar(x)]),
    with `_sphere`'s exact convention x*sqrt(d)/(||x||+1e-6). The PRODUCTION
    value in main() comes from the agent's own helpers + gciql_flow._sphere
    (never this function); this reference exists so a unit test pins the
    formula and a runtime assert (`main`) pins live==reference. Inputs:
    r_pool [P], b_pool_target [P,d] (target-tower embeddings). Returns [d]."""
    r_pool = np.asarray(r_pool, np.float64)
    b = np.asarray(b_pool_target, np.float64)
    assert r_pool.ndim == 1 and b.ndim == 2 and len(r_pool) == len(b), \
        f"r_pool {r_pool.shape} / b_pool_target {b.shape} shape mismatch"
    num = np.mean(r_pool[:, None] * b, axis=0)                       # [d]
    return num * np.sqrt(float(d)) / (float(np.linalg.norm(num)) + 1e-6)


def extract_family_means(families: dict) -> dict:
    """The gate's comparable per-family scalars from a results['families']
    dict (this script's AND the archived harness's -- same shape by
    construction): dirac/dense 'mean' (env-native success), region
    'mean_final' and 'mean_any'."""
    out = {}
    if "dirac" in families:
        out["dirac"] = families["dirac"].get("mean")
    if "dense" in families:
        out["dense"] = families["dense"].get("mean")
    if "region" in families:
        out["region_final"] = families["region"].get("mean_final")
        out["region_any"] = families["region"].get("mean_any")
    return out


def gate_control_match(mine: dict, archived: dict, tol: float = 0.02):
    """The spec-24 HARD ACCEPTANCE GATE (jax-free, unit-tested): compare this
    run's family means against the archived reference per family; every
    family present in BOTH must match within `tol`. Returns (ok: bool,
    rows: list of dicts with family/mine/archived/diff/ok)."""
    m, a = extract_family_means(mine), extract_family_means(archived)
    rows, ok = [], True
    for fam in sorted(set(m) & set(a)):
        mv, av = m[fam], a[fam]
        if mv is None or av is None:
            continue
        diff = abs(float(mv) - float(av))
        good = diff <= tol
        ok = ok and good
        rows.append({"family": fam, "mine": float(mv), "archived": float(av),
                     "diff": diff, "ok": good})
    assert rows, "gate_control_match: no comparable families between the two results"
    return ok, rows


# ======================= the z-index adapter =======================================

class _ZIndexView:
    """Routes the audited machinery's `goals` slot into the agent's own
    precomputed-z entry point. `decode_goal_retrieval_density` and
    `batched_goal_rollout_density` only ever touch (a) `sample_actions(
    observations, goals, seed, temperature)` and (b) `network.select(...)`;
    for a measure_density_z checkpoint the actor and density_f both condition
    on z where the g-indexed arch conditions on goals, so carrying z in the
    goals slot is exactly the trained interface -- EXCEPT that plain
    `sample_actions` would re-encode the slot through `_density_z_encode`
    (gciql_flow.py:2823, the E1 goal-obs path). This view swaps in
    `sample_actions_with_z` (gciql_flow.py:2843) and changes nothing else."""

    def __init__(self, agent):
        self._agent = agent

    @property
    def network(self):
        return self._agent.network

    def sample_actions(self, observations, goals=None, seed=None, temperature=0.0):
        return self._agent.sample_actions_with_z(
            observations=observations, z=goals, seed=seed, temperature=temperature)


# ======================= live z construction (agent's own code paths) ==============

def target_b_pool(agent, blocks, pool_obs: np.ndarray) -> np.ndarray:
    """B_bar(x_tilde) [P,d] for the pool via the agent's OWN eps-scaling
    (`_density_queries`) and TARGET tower (`target_density_b`) -- the exact
    pipeline `_density_z_encode` uses before its `_sphere` projection."""
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    u_sc = GCIQLFlowAgent._density_queries(jnp.asarray(pool_obs, jnp.float32), blocks)
    return np.asarray(agent.network.select("target_density_b")(u_sc), np.float64)


def analytic_zr_live(agent, blocks, pool_obs: np.ndarray, r_pool: np.ndarray,
                     d: int) -> tuple:
    """E2a's z_r through the agent's own code paths end-to-end: target-tower
    pool embedding (target_b_pool) -> reward-weighted pool mean (the
    definition) -> gciql_flow._sphere (the agent's own sqrt(d)-normalization,
    the same call `_density_z_encode` makes). Cross-checked against the numpy
    reference `analytic_zr`; hard-fails on disagreement. Returns
    (z_r [d] float32-safe np.ndarray, info dict)."""
    import jax.numpy as jnp
    from gciql_flow import _sphere

    b_t = target_b_pool(agent, blocks, pool_obs)                     # [P, d]
    num = jnp.mean(jnp.asarray(r_pool, jnp.float64)[:, None] * jnp.asarray(b_t), axis=0)
    z_live = np.asarray(_sphere(num[None], float(d))[0], np.float64)
    z_ref = analytic_zr(r_pool, b_t, d)
    err = float(np.max(np.abs(z_live - z_ref)))
    assert err < 1e-5, \
        f"analytic z_r live-vs-reference disagreement {err} -- normalization drift"
    return z_live, {"arm": "e2a", "z_norm": float(np.linalg.norm(z_live)),
                    "ref_maxdiff": err,
                    "r_pool_mean": float(np.mean(r_pool)),
                    "r_pool_pos": int(np.sum(np.asarray(r_pool) > 0))}


def z_candidates(agent, cand_obs: np.ndarray) -> np.ndarray:
    """{z(u_j)} via the agent's OWN `_density_z_encode` (deterministic, no
    rng: sqrt(d)*target_B(eps-scaled coords)/||.||). cand_obs: [N, obs_dim]
    FULL observations (the helper slices hit-spec blocks itself). [N,d]."""
    import jax.numpy as jnp

    return np.asarray(agent._density_z_encode(jnp.asarray(cand_obs, jnp.float32)),
                      np.float32)


# ======================= CLI ======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json; first match)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--arm", required=True, choices=ARMS,
                    help="control (g-indexed retrieval decode, g-checkpoints only) | "
                         "e2a (analytic z_r) | e2b (argmax search over pool-sample "
                         "z candidates) -- z-checkpoints only")
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--regions", type=int, default=4)
    ap.add_argument("--region-min-pos", type=int, default=5)
    ap.add_argument("--retrieval-k", type=int, default=1,
                    help="control arm's retrieval decode k (archived RET_k1 gate "
                         "protocol = 1); e2b always uses 1 (spec: argmax)")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=0,
                    help="0 = the env's own max_episode_steps")
    ap.add_argument("--relabel-size", type=int, default=50000,
                    help="dataset sample for region centers + the control arm's "
                         "retrieval candidates (audited harness default)")
    ap.add_argument("--pool_size", type=int, default=16384,
                    help="P: readout pool size (spec 24 protocol: 16384)")
    ap.add_argument("--e2b_candidate_size", type=int, default=4096,
                    help="e2b: leading pool observations whose z(u_j) become search "
                         "candidates ([N,P] scoring memory bound)")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-against", default="",
                    help="control arm: archived reference JSON; family means are "
                         "compared within --gate-tol and the verdict recorded")
    ap.add_argument("--gate-tol", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="",
                    help="output JSON (default results/spec24_eval/"
                         "<arm>_<run>_v2.json)")
    return ap


# ======================= main =====================================================

def main() -> int:
    args = build_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    assert not unknown, f"unknown families {unknown}; choose from {FAMILIES}"
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    assert all(1 <= t <= 5 for t in task_ids), f"task ids must be in 1..5, got {task_ids}"

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])

    ckpt_is_density = bool(cfg.get("measure_density", False))
    ckpt_is_density_z = bool(cfg.get("measure_density_z", False))
    _check = check_control_consistency if args.arm == "control" \
        else check_density_z_consistency
    for _w in _check(ckpt_is_density, ckpt_is_density_z, force=args.force,
                     run_dir=str(run_dir)):
        print(f"[density-z2] {_w}", flush=True)

    env_name = flags["env_name"]
    gamma = float(cfg["discount"])
    double_q = bool(cfg.get("double_q", True))
    temp = float(cfg.get("measure_density_temp", 1.0))
    d = int(cfg["measure_density_dim"])

    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    blocks = spec.blocks
    eval_tasks = eval_tasks_for(env_name)

    t00 = time.time()
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    _obs_dim = int(np.asarray(ds1["observations"]).shape[-1])
    _act_dim = int(np.asarray(ds1["actions"]).shape[-1])
    H = args.horizon if args.horizon > 0 else int(_e0.spec.max_episode_steps)

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, _obs_dim, _act_dim)

    print(f"[density-z2] restored {run_dir.name} step={args.checkpoint_step} "
          f"arm={args.arm} env={env_name} d={d} gamma={gamma} "
          f"hit_spec={cfg.get('measure_hit_spec')!r} double_q={double_q} temp={temp} "
          f"pool_size={args.pool_size} retrieval_k={args.retrieval_k} H={H}", flush=True)

    # ---- rng stream: IDENTICAL ORDER to eval_density_zeroshot.main() ----
    # (relabel draw first, then the pool draw) so the control arm's S / pool /
    # region centers are the archived run's own.
    rng_np = np.random.default_rng(args.seed)
    obs_all = np.asarray(ds1["observations"], np.float32)
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(ds1["terminals"]))
    idx = rng_np.choice(len(coords_all), size=min(args.relabel_size, len(coords_all)),
                        replace=False)
    relabel_coords = coords_all[idx]
    S_obs, S_coords = obs_all[idx], relabel_coords

    n_ds = len(obs_all)
    pool_idx = rng_np.choice(n_ds, size=min(args.pool_size, n_ds),
                             replace=(args.pool_size > n_ds))
    pool_obs = obs_all[pool_idx]
    u_raw, b_pool = build_pool(agent, blocks, pool_obs, spec)
    print(f"[density-z2] pool drawn: P={len(pool_obs)} ({time.time()-t00:.0f}s)",
          flush=True)

    # z machinery (z arms only; touches nothing above).
    zview = _ZIndexView(agent)
    if args.arm == "e2b":
        n_cand = min(args.e2b_candidate_size, len(pool_obs))
        cand_obs_z = z_candidates(agent, pool_obs[:n_cand])          # [N, d]
        cand_coords_z = coords_all[pool_idx[:n_cand]]
        print(f"[density-z2] e2b candidates: N={n_cand} (leading pool subset, "
              f"z via agent._density_z_encode)", flush=True)

    goal_obs, goal_coord = {}, {}
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        goal_coord[i] = spec.coords(goal_obs[i][None])[0]
        env.close()

    pools = {}

    def pool_envs(i):
        if i not in pools:
            pools[i] = [ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
                        for _ in range(args.episodes)]
        return pools[i]

    def close_all():
        for i in list(pools):
            for e in pools.pop(i):
                if hasattr(e, "close"):
                    e.close()

    def run_cell(r_pool_full, rfn, envs):
        """The audited run_cell with ONLY the index-selection step swapped.
        r_pool_full: family reward at the pool coords [P]. Returns the same
        tuple shape the audited harness's run_cell returns (minus sweep)."""
        r_pool = r_pool_full[: args.pool_size]
        x0_obs, _ = envs[0].reset(seed=0)
        if args.arm == "control":
            g_hat, decode_info = decode_goal_retrieval_density(
                agent, x0_obs, u_raw, b_pool, r_pool, blocks, temp, double_q,
                S_coords, S_obs, retrieval_k=args.retrieval_k, rng_seed=args.seed)
            roll_agent, index = agent, g_hat
            decode_info["arm"] = "control"
        elif args.arm == "e2a":
            z_r, decode_info = analytic_zr_live(agent, blocks, pool_obs,
                                                r_pool, d)
            roll_agent, index = zview, np.asarray(z_r, np.float32)
        else:  # e2b -- audited decode, z candidates in the cand_obs slot, k=1 argmax
            z_star, decode_info = decode_goal_retrieval_density(
                zview, x0_obs, u_raw, b_pool, r_pool, blocks, temp, double_q,
                cand_coords_z, cand_obs_z, retrieval_k=1, rng_seed=args.seed)
            decode_info["arm"] = "e2b"
            decode_info["n_candidates"] = int(len(cand_obs_z))
            roll_agent, index = zview, np.asarray(z_star, np.float32)
        traj, ot, ln, senv, t0 = batched_goal_rollout_density(
            roll_agent, u_raw, b_pool, r_pool, blocks, gamma, temp, double_q,
            spec, rfn, envs, index, ridge=args.ridge, H=H)
        return traj, ot, ln, senv, t0, decode_info

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "arm": args.arm, "env": env_name, "d": d, "gamma": gamma,
        "double_q": double_q, "temp": temp,
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
        "pool_size": args.pool_size,
        "e2b_candidate_size": args.e2b_candidate_size if args.arm == "e2b" else None,
        "retrieval_k": args.retrieval_k, "relabel_size": int(len(relabel_coords)),
        "ridge": args.ridge, "seed": args.seed, "episodes": args.episodes,
        "horizon": H, "tasks": task_ids, "families": families,
        "harness": "THIN WRAPPER over eval_density_zeroshot.py (audited spec-18 "
                   "harness): build_pool/density_query/compute_density_primary/"
                   "decode_goal_retrieval_density/batched_goal_rollout_density "
                   "imported and called end-to-end; identical rng stream order "
                   "(relabel then pool); env-native success for dirac/dense, "
                   "region_success_from_obs for region. ONLY the index-selection "
                   "step differs per --arm. v2 2026-08-13; v1 artifacts (broken "
                   "instrument, see module docstring) kept without _v2.",
        "index_selection": {
            "control": "decode_goal_retrieval_density over the relabel sample, "
                       "retrieval_k as given (archived RET_k1 protocol = 1)",
            "e2a": "z_r = _sphere( E_pool[r(x) target_density_b(_density_queries(x))] ) "
                   "-- agent's own scaling/tower/normalization end-to-end, numpy "
                   "reference cross-checked at runtime",
            "e2b": "argmax_j PRIMARY-score over z(u_j)=_density_z_encode(u_j), "
                   "u_j = leading e2b_candidate_size pool observations, scored by "
                   "decode_goal_retrieval_density itself (z in the cand_obs slot), "
                   "retrieval_k=1",
            "rollout": "batched_goal_rollout_density unchanged; z arms condition "
                       "the actor via sample_actions_with_z through _ZIndexView "
                       "(gciql_flow.py:2843)",
        }[args.arm] if args.arm in ("control", "e2a", "e2b") else None,
        "goal_coords": {f"t{i}": goal_coord[i].tolist() for i in task_ids},
        "spec": "PAPER/Multistep-FB/specs/24-z-indexed-density.md sec.3-4",
    }, "families": {}}

    def build_rec(senv, t0, decode_info, extra=None):
        rec = {"decode": decode_info,
               "success": float(np.mean(senv)),
               "qr_primary_t0_mean": float(np.mean(t0["qr_primary"])),
               "qr_secondary_t0_mean": float(np.mean(t0["qr_secondary"]))}
        if extra:
            rec.update(extra)
        return rec

    # ================= dirac / dense (audited family builders + env success) =======
    for fam_name, builder in (("dirac", reward_conj_dirac), ("dense", reward_conj_dense)):
        if fam_name not in families:
            continue
        fam = {}
        for i in task_ids:
            r_pool_full = builder(coords_all[pool_idx], goal_coord[i], spec)

            def rfn(coords, c=goal_coord[i], f=builder):
                return f(coords, c, spec)

            _tj, _ot, _ln, senv, t0, decode_info = run_cell(r_pool_full, rfn,
                                                            pool_envs(i))
            rec = build_rec(senv, t0, decode_info)
            fam[f"t{i}"] = rec
            print(f"[density-z2][{fam_name}] t{i}: succ={rec['success']} "
                  f"qr0={rec['qr_primary_t0_mean']:.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success"] for v in fam.values() if isinstance(v, dict)]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"][fam_name] = fam

    # ================= region (audited hindsight centers + membership success) =====
    if "region" in families:
        rng_reg = np.random.default_rng(args.seed + 2)
        centers, _cobs, npos_c, rejected = draw_region_centers_with_obs(
            obs_all, coords_all, bounds, spec, relabel_coords, rng_reg,
            args.regions, min_pos=args.region_min_pos)
        v0 = task_ids[0]
        fam = {"n_centers": int(len(centers)), "centers_rejected": int(rejected),
               "center_min_pos": int(args.region_min_pos)}
        print(f"[density-z2][region] {len(centers)}/{args.regions} hindsight "
              f"centers ({rejected} rejected)", flush=True)
        for k_i in range(len(centers)):
            c = centers[k_i]
            r_pool_full = reward_conj_dirac(coords_all[pool_idx], c, spec)

            def rfn(coords, c=c):
                return reward_conj_dirac(coords, c, spec)

            _tj, ot, ln, _se, t0, decode_info = run_cell(r_pool_full, rfn,
                                                         pool_envs(v0))
            fin, anys = region_success_from_obs(ot, ln, c, spec)
            rec = build_rec(np.asarray(fin), t0, decode_info,
                            extra={"center_coord": np.asarray(c).tolist(),
                                   "center_n_pos": int(npos_c[k_i]),
                                   "success_final": float(fin.mean()),
                                   "success_any": float(anys.mean())})
            fam[f"c{k_i}"] = rec
            print(f"[density-z2][region] c{k_i}: fin={rec['success_final']} "
                  f"any={rec['success_any']} ({time.time()-t00:.0f}s)", flush=True)
        cells = [v for k, v in fam.items() if k.startswith("c") and isinstance(v, dict)]
        fam["mean_final"] = float(np.mean([v["success_final"] for v in cells])) \
            if cells else None
        fam["mean_any"] = float(np.mean([v["success_any"] for v in cells])) \
            if cells else None
        results["families"]["region"] = fam

    results["meta"]["family_means"] = extract_family_means(results["families"])

    # ---- HARD ACCEPTANCE GATE (control arm) ----
    if args.arm == "control" and args.gate_against:
        archived = json.loads(Path(args.gate_against).read_text())
        ok, rows = gate_control_match(results["families"], archived["families"],
                                      tol=args.gate_tol)
        results["meta"]["control_gate"] = {
            "against": str(args.gate_against), "tol": args.gate_tol,
            "pass": bool(ok), "rows": rows}
        print(f"[density-z2] CONTROL GATE vs {args.gate_against}: "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        for r in rows:
            print(f"[density-z2]   {r['family']}: mine={r['mine']:.3f} "
                  f"archived={r['archived']:.3f} diff={r['diff']:.3f} "
                  f"{'ok' if r['ok'] else 'MISMATCH'}", flush=True)

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec24_eval" / f"{args.arm}_{run_dir.name}_v2.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[density-z2] DONE ({time.time()-t00:.0f}s) -> {outp}\n"
          f"[density-z2] family means: {results['meta']['family_means']}", flush=True)
    close_all()
    if args.arm == "control" and args.gate_against \
            and not results["meta"]["control_gate"]["pass"]:
        return 3   # loud non-zero exit: the instrument did not validate
    return 0


if __name__ == "__main__":
    sys.exit(main())
