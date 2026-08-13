#!/usr/bin/env python
"""scripts/rescuing_fb/probe_b_reward_decode.py -- spec 25 P0 (coordinator,
2026-08-13): "B-decodability probe" -- BEFORE any training, measure how well
the three standard reward families are linearly reconstructible from a
`measure_density` checkpoint's B(u~tilde_pool) features by closed-form ridge.

Context: `PAPER/Multistep-FB/specs/25-z-coupling-port.md` sec.1 (P0). Spec 24's
z-indexed density measure trains healthily (CE/ESS match the g-indexed
control) but its actor never extracts a goal-directed policy (E1~.017,
E2~.03 vs g-anchors .549/.488/.177). Spec 25's mechanism C2 hypothesizes the
z-indexed B's span does not support linear readout of reward the way FB's
orthonormalized B does; C2's training-time port is a per-batch ridge
auxiliary loss (spec sec.0 C2: `v* = (B_U^T B_U + lambda I)^-1 B_U^T r_c(U)`,
`L_dec = coef * ||B_U v* - r_c(U)||^2 / n`). This script is the EVAL-ONLY twin
of that formula, run on EXISTING checkpoints (no training, no rollouts, no
actor) to test the diagnosis before spending any training compute on it:

Pre-registered readings (spec sec.1):
  - g-B decodes, z-B does not  => C2 is binding; the rdecode arm should matter.
  - both decode                => B-geometry is fine; C1/C3 carry the failure.
  - neither decodes (raw does) => B under CE never supports linear readout;
    C2 becomes the headline mechanism.

REUSE, NOT REIMPLEMENT (thin-wrapper convention; v1-E2 lesson: a control is
mandatory, see below): `build_pool` (pool draw -> raw block coords U_raw +
B(u~tilde_pool) via the checkpoint's OWN `density_b` tower, live) and
`ridge_fit_wr` (the closed-form ridge solve, `(B^T B + ridge I)^-1 B^T r`,
IDENTICAL algebra to spec 25's C2 formula with U -> the eval pool) both come
from `eval_density_zeroshot.py`, unchanged. The three reward families --
sparse (`reward_conj_dirac`, exact conjunctive hit-spec indicator at a task
goal), dense (`reward_conj_dense`, a smooth kernel of scaled distance to a
task goal) and region (`reward_conj_dirac` again, but centered at a
DIFFERENT, hindsight-drawn coordinate per `draw_region_centers_with_obs` --
broader spatial coverage than the 5 fixed task goals) -- are the SAME family
builders `eval_density_zeroshot.py` itself uses for its own dirac/dense/
region families; imported from `eval_zeroshot_rewards.py` /
`eval_partition_zeroshot.py`, not reimplemented. `check_density_consistency`
/ `DensityMismatchError` (measure_density=True refusal), `geometry_for` /
`HitSpec` / `episode_bounds` (the single hit-spec geometry every eval script
in this repo shares), and the `load_agent_dims` restore + sys.path dance are
all copied verbatim from the pattern `eval_z_identity.py` / (this script's
closest sibling: also actor-free, also eval-only, also runs symmetrically on
z-indexed AND g-indexed checkpoints) and `eval_density_zeroshot.py` use.

MANDATORY CONTROL, TWO ARMS (v1-E2 lesson: a prior probe in this program
produced a dramatic null that turned out to be a bug, caught only because a
control was run alongside it; another reported retrieval accuracy with no
chance/ceiling context; a first version of THIS script shipped a control
that was itself under-powered by construction -- see "ENRICHED CONTROL"
below, kept as history, not silently dropped):

  (1) `raw_coord_control` -- ridge (+bias) directly against the RAW hit-spec
      coordinates `u_raw` ([P, d_c], `build_pool`'s own output). A LINEAR
      floor: a Gaussian bump (dense) or a hard indicator (sparse/region) is
      not a linear function of a handful of raw coordinates in general, so
      this arm is expected to score near zero on an ISOTROPIC pool (no
      direction is preferred) and only rises when the pool happens to be
      asymmetric relative to the reward's center -- see
      `test_raw_coord_control_high_for_asymmetric_domain_low_for_symmetric`.
      It is NOT, by itself, evidence that a family is undecodable from
      coordinates in general -- only that it is not LINEARLY decodable.

  (2) `raw_coord_rbf_control` (ENRICHED CONTROL) -- ridge (+bias) against
      `rff_features_multiscale`: a fixed, reward-independent Random Fourier
      Feature bank (Rahimi & Recht 2007) approximating a MIXTURE of RBF/
      Gaussian kernels over the SCALED hit-spec coordinates (`spec.scaled
      (u_raw)`, i.e. each block divided by its own eps -- the same unit-
      radius convention `reward_conj_dense`'s `sigma_scaled` already lives
      in, so the `--rbf-bandwidths` list is meaningful across blocks with
      different raw eps), sized to match `B`'s dimension (`--rbf-dim 0` =
      auto = the checkpoint's `measure_density_dim`) so the B-vs-control
      comparison is fair on FEATURE COUNT, not just feature identity.
      MULTIPLE bandwidths, not one: an empirical sweep (bandwidths 0.25 to
      8.0, real cube-double pool) found dense R^2 peaking near bandwidth~4
      while a SINGLE bandwidth=1.0 bank (matched to the sharp-indicator
      scale) under-shot even the plain linear-coordinate floor on dense --
      one scale cannot serve both a sharp ball indicator (sparse/region,
      scaled hit radius 1.0) and a broad bump (dense, `sigma_scaled=2.0`) at
      once, so the default (`--rbf-bandwidths 0.5,1,2,4`) mixes several,
      dividing the feature budget across them (see `rff_features_multiscale`
      docstring for the sweep numbers). This is the arm that is actually
      "known to be high" for the dense family (mixture R^2 ~0.52 on a real
      checkpoint pool, comfortably above the linear floor's ~0.25): a kernel
      bank can represent a smooth bump the same way `B` can (both are
      order-`d` nonlinear feature maps of the coordinate), whereas six raw
      linear coordinates structurally cannot, no matter how well the
      geometry is represented anywhere. If arm (2) ALSO reads near zero on a
      family -- and the SAME sweep found sparse/region near-zero (~0.01-0.04)
      at EVERY tested bandwidth, not just the default mixture, so this is not
      a bandwidth-mismatch artifact -- the informative conclusion is "not
      linearly decodable from ANY generic coordinate feature map at this
      pool size", not "B lacks structure": arm (2) is what lets those two be
      told apart; arm (1) alone cannot.

R^2 CONVENTION: `ridge_fit_wr` (mirroring spec 25's own C2 formula) fits
without a bias/intercept term. A diagnostic R^2 without a bias column is
comparing every fit against a rigid "predict zero everywhere" baseline
rather than a "predict the mean" baseline, and can read spuriously negative
for both arms even when the underlying representation is fine (the two
reward families here are never zero-mean: sparse/region are mostly-zero
indicators with a small positive mean, dense is bounded (0, 1] with a
strictly positive mean). To keep R^2 meaningful and comparable to the
literature's usual "chance = 0 (predict the mean), ceiling = 1 (exact
reconstruction)" reading, this script appends a constant bias column of ones
to BOTH the B features and the raw-coordinate control features before
calling the SAME reused `ridge_fit_wr` -- applied identically to both arms,
so the B-vs-raw-coordinate comparison stays apples-to-apples; this is a
documented, deliberate deviation from spec sec.0's bare training-time
formula (which has no bias term, and does not need one -- it is a per-batch
auxiliary GRADIENT signal shaping `B`'s span over training, not a diagnostic
R^2 read against a fixed checkpoint) and is noted again on every result's
`meta.r2_convention` field. `R^2 = 1 - SS_res/SS_tot`, `SS_tot` against the
reward vector's own mean: chance = 0, ceiling = 1, by construction of R^2 --
no separate chance-baseline run is needed (unlike the retrieval-accuracy
metric in `eval_z_identity.py`, which has no such closed-form floor).

Run (CPU-only; GPU training jobs are live on the box):
  taskset -c 24-39 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \\
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=4 \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/probe_b_reward_decode.py \\
    --run-dir '/home/adityamohan/exp_spec24/factored-fb/GCIQL-CubeDouble-Flow-Measure-DensityZ-d50/z24_cd_z_s0' \\
    --checkpoint-step 500000 --pool-size 16384 --ridge 1e-4 --seed 0
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

# Reused, not reimplemented: the pool builder + closed-form ridge solve (spec
# 25 sec.0 C2's exact formula), the density-checkpoint consistency refusal,
# and the reward-family / region-center / geometry helpers this program's
# other density scripts already share.
from eval_density_zeroshot import (  # noqa: E402
    DensityMismatchError,
    build_pool,
    check_density_consistency,
    ridge_fit_wr,
)
from eval_partition_zeroshot import draw_region_centers_with_obs  # noqa: E402
from eval_zeroshot_rewards import (  # noqa: E402
    eval_tasks_for,
    print_table,
    reward_conj_dense,
    reward_conj_dirac,
)
from env_geometry import episode_bounds, geometry_for  # noqa: E402

FAMILIES = ("sparse", "dense", "region")


# ======================= ridge-R^2 algebra (pure numpy, unit-tested) ===============

def ridge_r2(features: np.ndarray, r: np.ndarray, ridge: float,
            add_bias: bool = True) -> dict:
    """Ridge-reconstruction R^2 of `r` from `features` (spec 25 sec.1 P0):
    fit `v* = ridge_fit_wr(X, r, ridge)` (the SAME reused closed-form solve
    spec sec.0 C2 trains with), predict `r_hat = X @ v*`, report
    `R^2 = 1 - SS_res/SS_tot` against `r`'s own mean (chance=0, ceiling=1 by
    construction -- see module docstring "R^2 CONVENTION").

    `add_bias`: append a constant column of ones to `features` before the
    solve (default True -- see module docstring; both the B arm and the
    raw-coordinate control always get this applied identically).

    features: [P, k]; r: [P]. Returns {"r2": float | None (None iff `r` is
    exactly constant -- SS_tot == 0, R^2 undefined, NOT reported as 0/1),
    "ss_res": float, "ss_tot": float, "n": int, "n_features": int,
    "v": [k(+1)] list, "add_bias": bool}. Pure numpy, no jax."""
    features = np.asarray(features, np.float64)
    r = np.asarray(r, np.float64).reshape(-1)
    assert features.ndim == 2 and len(features) == len(r), \
        f"features {features.shape} / r {r.shape} shape mismatch"
    assert len(r) > 0, "ridge_r2 needs at least one row"
    x = (np.concatenate([features, np.ones((len(features), 1))], axis=-1)
         if add_bias else features)
    v = ridge_fit_wr(x, r, ridge)
    r_hat = x @ v
    ss_res = float(np.sum((r - r_hat) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = None if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {"r2": r2, "ss_res": ss_res, "ss_tot": ss_tot, "n": int(len(r)),
           "n_features": int(x.shape[1]), "v": v.tolist(), "add_bias": bool(add_bias)}


def rff_features(x_scaled: np.ndarray, dim: int, bandwidth: float, seed: int) -> np.ndarray:
    """Random Fourier Features (Rahimi & Recht 2007) approximating an RBF/
    Gaussian kernel `exp(-||x-y||^2 / (2*bandwidth^2))` over `x_scaled`:
        phi(x) = sqrt(2/dim) * cos(x @ W + b),  W ~ N(0, 1/bandwidth^2),
                                                 b ~ Uniform(0, 2*pi)
    A FIXED, reward-independent feature map -- `W`/`b` depend only on
    `seed`/`dim`/`bandwidth`, never on any reward vector, so this is an
    honest control (it cannot have been fit to leak information about the
    specific family/cell it will be scored against). `x_scaled` is expected
    in the SAME unit-radius convention `spec.scaled` produces (each block
    divided by its own eps) so one `bandwidth` is meaningful across blocks
    with different raw eps -- see module docstring "ENRICHED CONTROL".
    x_scaled: [P, d_c]. Returns [P, dim], pure numpy."""
    x_scaled = np.asarray(x_scaled, np.float64)
    assert x_scaled.ndim == 2, f"x_scaled must be [P, d_c], got {x_scaled.shape}"
    assert dim > 0, f"rff dim must be positive, got {dim}"
    assert bandwidth > 0, f"rff bandwidth must be positive, got {bandwidth}"
    rng = np.random.default_rng(seed)
    d_c = x_scaled.shape[-1]
    w = rng.normal(scale=1.0 / bandwidth, size=(d_c, dim))
    b = rng.uniform(0.0, 2.0 * np.pi, size=dim)
    return np.sqrt(2.0 / dim) * np.cos(x_scaled @ w + b)


def rff_features_multiscale(x_scaled: np.ndarray, dim: int, bandwidths, seed: int) -> np.ndarray:
    """A SINGLE bandwidth cannot be well-matched to both a sharp ball
    indicator (sparse/region, scaled hit radius exactly 1.0) and a broad
    Gaussian bump (dense, `sigma_scaled=2.0`) at once -- an empirical sweep
    over bandwidths 0.25..8.0 on a real cube-double pool showed dense R^2
    peaking near bandwidth~4 (0.79) while a bandwidth=1.0 control (matched
    to the sharp-indicator scale) UNDER-shot even the plain linear-
    coordinate floor on dense (0.05 vs 0.25) -- exactly the failure mode a
    single global bandwidth predicts. This concatenates `rff_features` banks
    at SEVERAL bandwidths (dividing `dim` as evenly as possible across them,
    remainder to the first few bands, so the TOTAL feature count still
    matches `dim` -- fair vs B on feature count regardless of how many
    bandwidths are mixed in), so the enriched control is not hostage to one
    length-scale guess. The SAME sweep found sparse/region near-zero (~0.01-
    0.04) across EVERY tested bandwidth AND this mixture -- i.e. that null is
    not a bandwidth-mismatch artifact (see script module docstring 'ENRICHED
    CONTROL' / spec 25 P0 reading (b)). Deterministic given (x_scaled, dim,
    bandwidths, seed): band k uses `rff_features(..., seed=seed*1009 + k)`
    (a fixed, arbitrary large odd stride so distinct bandwidths never
    collide with plausible small seeds). x_scaled: [P, d_c]. Returns
    [P, dim] (exactly, via the remainder split)."""
    bandwidths = list(bandwidths)
    assert bandwidths, "need at least one bandwidth"
    n = len(bandwidths)
    per, rem = divmod(int(dim), n)
    dims = [per + (1 if k < rem else 0) for k in range(n)]
    assert all(d_k > 0 for d_k in dims), \
        f"dim={dim} too small to split across {n} bandwidths"
    blocks = [rff_features(x_scaled, d_k, bw, seed=int(seed) * 1009 + k)
             for k, (bw, d_k) in enumerate(zip(bandwidths, dims))]
    out = np.concatenate(blocks, axis=-1)
    assert out.shape[-1] == dim
    return out


def decode_cell(b_pool: np.ndarray, u_raw: np.ndarray, u_rbf: np.ndarray,
                r_pool: np.ndarray, ridge: float) -> dict:
    """One reward vector, THREE arms (B-decode + the two-arm mandatory
    control -- linear-raw-coordinate FLOOR and RBF/RFF-enriched control,
    see module docstring "MANDATORY CONTROL, TWO ARMS"), same ridge -- the
    atomic unit every family/cell below reports. Also carries reward-vector
    diagnostics (mean/var/n_positive) so a degenerate reward (e.g. a sparse
    indicator that hit nothing in the pool) is visible in the output rather
    than silently producing a meaningless R^2. `b_pool`/`u_raw`/`u_rbf` must
    share the SAME pool ordering/size as `r_pool` -- shape mismatches raise
    (the exact bug class a mandatory control exists to catch)."""
    r_pool = np.asarray(r_pool, np.float64).reshape(-1)
    return {
        "b_decode": ridge_r2(b_pool, r_pool, ridge),
        "raw_coord_control": ridge_r2(u_raw, r_pool, ridge),
        "raw_coord_rbf_control": ridge_r2(u_rbf, r_pool, ridge),
        "reward_mean": float(r_pool.mean()),
        "reward_var": float(r_pool.var()),
        "n_positive": int((r_pool > 0.5).sum()),
        "n_pool": int(len(r_pool)),
    }


# ======================= live-agent integration (jax-dependent; lazy) ==============

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
    warnings = check_density_consistency(ckpt_is_density, True, force=args.force,
                                         run_dir=str(run_dir))
    for _w in warnings:
        print(f"[b-decode] {_w}", flush=True)

    env_name = args.env or flags["env_name"]
    is_density_z = bool(cfg.get("measure_density_z", False))
    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    blocks = spec.blocks
    d_c = spec.dim
    d = int(cfg["measure_density_dim"])
    eval_tasks = eval_tasks_for(env_name)

    t00 = time.time()
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    obs_dim = int(np.asarray(ds1["observations"]).shape[-1])
    act_dim = int(np.asarray(ds1["actions"]).shape[-1])

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)

    print(f"[b-decode] restored {run_dir.name} step={args.checkpoint_step} "
          f"env={env_name} d={d} d_c={d_c} measure_density_z={is_density_z} "
          f"pool_size={args.pool_size} ridge={args.ridge}", flush=True)

    rng_np = np.random.default_rng(args.seed)
    obs_all = np.asarray(ds1["observations"], np.float32)
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(ds1["terminals"]))
    n_ds = len(obs_all)

    pool_idx = rng_np.choice(n_ds, size=min(args.pool_size, n_ds),
                             replace=(args.pool_size > n_ds))
    pool_obs = obs_all[pool_idx]
    u_raw, b_pool = build_pool(agent, blocks, pool_obs, spec)
    rbf_dim = args.rbf_dim if args.rbf_dim > 0 else b_pool.shape[-1]
    rbf_bandwidths = [float(b) for b in args.rbf_bandwidths.split(",") if b.strip()]
    u_rbf = rff_features_multiscale(spec.scaled(u_raw), rbf_dim, rbf_bandwidths, args.seed)
    print(f"[b-decode] pool built: P={len(pool_obs)} d={b_pool.shape[-1]} "
          f"d_c={u_raw.shape[-1]} rbf_dim={rbf_dim} rbf_bandwidths={rbf_bandwidths} "
          f"({time.time()-t00:.0f}s)", flush=True)

    relabel_idx = rng_np.choice(n_ds, size=min(args.relabel_size, n_ds), replace=False)
    relabel_coords = coords_all[relabel_idx]

    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    assert all(1 <= t <= 5 for t in task_ids), f"task ids must be in 1..5, got {task_ids}"
    goal_coord = {}
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_coord[i] = spec.coords(np.asarray(info["goal"], np.float32)[None])[0]
        env.close()

    rng_reg = np.random.default_rng(args.seed + 2)
    centers, _centers_obs, npos_c, rejected = draw_region_centers_with_obs(
        obs_all, coords_all, bounds, spec, relabel_coords, rng_reg, args.regions,
        min_pos=args.region_min_pos)
    print(f"[b-decode] region centers: {len(centers)}/{args.regions} drawn "
          f"({rejected} rejected at min_pos={args.region_min_pos})", flush=True)

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "measure_density_z": is_density_z, "d": d, "d_c": d_c,
        "hit_spec": cfg.get("measure_hit_spec"), "pool_size": int(len(pool_obs)),
        "ridge": args.ridge, "seed": args.seed, "tasks": task_ids,
        "regions_requested": args.regions, "regions_drawn": int(len(centers)),
        "regions_rejected": int(rejected),
        "rbf_dim": int(rbf_dim), "rbf_bandwidths": rbf_bandwidths,
        "r2_convention": "R^2 = 1 - SS_res/SS_tot against the reward vector's own "
                        "mean (chance=0, ceiling=1 by construction); B, the linear "
                        "raw-coordinate control, and the RBF/RFF-enriched control "
                        "all append a constant bias column before the SAME reused "
                        "closed-form ridge solve (ridge_fit_wr, identical to spec 25 "
                        "sec.0 C2's formula) -- see module docstring 'R^2 CONVENTION' "
                        "and 'MANDATORY CONTROL, TWO ARMS'",
        "protocol": "spec 25 sec.1 P0 -- ridge-reconstruction R^2 of sparse/dense/"
                   "region reward families from B(u~tilde_pool) vs. a mandatory "
                   "two-arm control (linear raw hit-spec coordinates + an "
                   "RBF/RFF-enriched coordinate feature bank sized to match B's "
                   "dimension), on EXISTING checkpoints, eval-only (no training, "
                   "no rollouts, no actor)",
    }, "families": {}}
    table = []

    def _fmt(x):
        return "None" if x is None else f"{x:.4f}"

    def record(fam_name, cell_name, r_pool):
        cell = decode_cell(b_pool, u_raw, u_rbf, r_pool, args.ridge)
        results["families"].setdefault(fam_name, {})[cell_name] = cell
        b_r2 = cell["b_decode"]["r2"]
        raw_r2 = cell["raw_coord_control"]["r2"]
        rbf_r2 = cell["raw_coord_rbf_control"]["r2"]
        table.append((fam_name, cell_name, "r2_B", _fmt(b_r2)))
        table.append((fam_name, cell_name, "r2_raw_control", _fmt(raw_r2)))
        table.append((fam_name, cell_name, "r2_raw_rbf_control", _fmt(rbf_r2)))
        print(f"[b-decode][{fam_name}] {cell_name}: r2_B={_fmt(b_r2)} "
              f"r2_raw_control={_fmt(raw_r2)} r2_raw_rbf_control={_fmt(rbf_r2)} "
              f"n_pos={cell['n_positive']} ({time.time()-t00:.0f}s)", flush=True)

    if "sparse" in args.families.split(","):
        for i in task_ids:
            r_pool = reward_conj_dirac(u_raw, goal_coord[i], spec)
            record("sparse", f"t{i}", r_pool)

    if "dense" in args.families.split(","):
        for i in task_ids:
            r_pool = reward_conj_dense(u_raw, goal_coord[i], spec)
            record("dense", f"t{i}", r_pool)

    if "region" in args.families.split(","):
        for k_i, c in enumerate(centers):
            r_pool = reward_conj_dirac(u_raw, c, spec)
            record("region", f"c{k_i}", r_pool)
        results["families"].setdefault("region", {})["_centers"] = {
            f"c{k_i}": {"coord": np.asarray(c).tolist(), "n_pos_relabel": int(npos_c[k_i])}
            for k_i, c in enumerate(centers)}

    for fam_name, cells in results["families"].items():
        cell_items = [(k, v) for k, v in cells.items()
                     if not k.startswith("_") and isinstance(v, dict) and "b_decode" in v]

        def _vals(arm):
            return [v[arm]["r2"] for _k, v in cell_items if v[arm]["r2"] is not None]
        cells["mean_r2_B"] = float(np.mean(_vals("b_decode"))) if _vals("b_decode") else None
        cells["mean_r2_raw_control"] = (float(np.mean(_vals("raw_coord_control")))
                                        if _vals("raw_coord_control") else None)
        cells["mean_r2_raw_rbf_control"] = (float(np.mean(_vals("raw_coord_rbf_control")))
                                            if _vals("raw_coord_rbf_control") else None)
        table.append((fam_name, "MEAN", "r2_B", _fmt(cells["mean_r2_B"])))
        table.append((fam_name, "MEAN", "r2_raw_control", _fmt(cells["mean_r2_raw_control"])))
        table.append((fam_name, "MEAN", "r2_raw_rbf_control",
                     _fmt(cells["mean_r2_raw_rbf_control"])))

    print_table(table)
    print(f"[b-decode] DONE {run_dir.name} ({time.time()-t00:.0f}s)", flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json; first match "
                         "used; must be a measure_density=True checkpoint)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--env", default="", help="override env (default: flags.json env_name)")
    ap.add_argument("--tasks", default="1,2,3,4,5",
                    help="comma-sep task indices 1..5 (sparse/dense families' goals)")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help=f"comma-sep subset of {FAMILIES}")
    ap.add_argument("--regions", type=int, default=4,
                    help="region family: number of hindsight centers")
    ap.add_argument("--region-min-pos", type=int, default=5,
                    help="region family: min positive relabel samples per center")
    ap.add_argument("--relabel-size", type=int, default=50000,
                    help="dataset sample size for the region family's hindsight draw")
    ap.add_argument("--pool-size", type=int, default=16384,
                    help="P: the standard pool size (spec 25 sec.1 P0)")
    ap.add_argument("--ridge", type=float, default=1e-4,
                    help="ridge lambda for the B-decode AND BOTH raw-coordinate-"
                         "control closed-form solves (spec 25 sec.0 C2's dose)")
    ap.add_argument("--rbf-dim", type=int, default=0,
                    help="feature count for the RBF/RFF-enriched control (0 = "
                         "auto = match B's own dimension, so the comparison is "
                         "fair on feature count -- see module docstring "
                         "'ENRICHED CONTROL'), split across --rbf-bandwidths")
    ap.add_argument("--rbf-bandwidths", default="0.5,1,2,4",
                    help="comma-sep RFF kernel bandwidths over SCALED (unit-"
                         "radius) hit-spec coordinates, mixed into one bank "
                         "(rff_features_multiscale) -- spans the sparse/region "
                         "hit radius (1.0) and the dense family's sigma_scaled=2.0; "
                         "a single bandwidth under-serves one family or the other, "
                         "see module docstring 'ENRICHED CONTROL'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="override the measure_density consistency refusal")
    ap.add_argument("--out", default="",
                    help="output JSON (default results/spec25_p0/probe_<run>.json)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    assert not unknown, f"unknown families {unknown}; choose from {FAMILIES}"

    res = run_one_checkpoint(args)
    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec25_p0" / f"probe_{Path(args.run_dir).name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, default=lambda o: float(o)))
    print(f"\n[b-decode] DONE -> {outp}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
