#!/usr/bin/env python
"""scripts/rescuing_fb/eval_spec30_g293.py -- G29-3: the general-reward-interface
evaluation (spec 30 Rev 1 P3), EVAL-ONLY on the centered-dual checkpoint.

THE QUESTION. ra2_cs_centdual's native goal eval (.816) already goes through the
reward-sample API over a frozen M=256 kernel pool -- but only for GOAL-KERNEL
rewards. G29-3 asks whether the same interface (reward samples -> zhat -> frozen
policy) controls for reward families the actor never trained on: mixtures,
target-minus-avoid, negatives, plus affine copies (which the centered encoder is
provably invariant to) and the constant-reward fallback.

CONTROL GATE FIRST (three-times-burned rule): reproduce the arm's own native
number through (a) the arm's own sample_actions path and (b) THIS harness's
explicit density_z_from_reward_samples(M=256) + sample_actions_with_z path,
before any new family is read. Rollout gates are tolerance-based (binomial noise
at 50 eps/task), unlike the bit-exact readout gates: PASS = overall within
+/-0.06 of .816 on both paths.

SOFT-VARIANT CAVEAT (spec 30 Rev 1): this checkpoint is measure_density_hard=
False, so every reward is evaluated under kernel-smoothed semantics K_{rho,eps}r.
Pre-registered distortion expectations: sharp exclusion boundaries (tma) and the
max at the two-goal boundary (maxmix) are softened; sums are linear and safe.

Machinery reused, not rewritten: restore = t3_offpeak_multienv.load_agent_dims;
pool = eval_density_zeroshot.build_pool (audited spec-18 builder, same rng
order); rollouts = eval_zeroshot_rewards.batched_rollout_record (proven lockstep
protocol, records obs_traj); envs = ogbench singletask envs via
eval_tasks_for(env_name).

Success geometry: hit-spec '19:22:0.4' -- obs dims 19:22 = cube xyz in obs units
(10x meters), and the 0.4-obs-unit kernel eps equals the native OGBench success
radius 0.04 m exactly. All family successes are computed in obs space from
obs_traj[:, :, 19:22]; native cells also report the env's own success flag as a
cross-check.

Run (GPU ok -- rollouts are lockstep; box shared, use one GPU):
  env CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec30_g293.py \\
    --run-dir ~/exp_spec30_p1/factored-fb/GCIQL-CubeSingle-*-RA2-CentDual-d50/ra2_cs_centdual \\
    --checkpoint-step 500000 --stage all --out-dir results/spec30_g293
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

GAMMA = 0.99
SUCC_R = 0.4          # obs units == 0.04 m, the native OGBench cube criterion
AVOID_R = 0.8         # pre-registered avoid threshold = 2x success radius
RIDGE_MULT = 0.01     # matches measure_density_z_ridge_lambda_mult of the arm
# Deviation from spec 30 P3's {256,1024,16384}: the encoder fit pool is the
# FIT HALF of the disjoint fit/eval split (8192 of P=16384), so the top sweep
# point is the fit-half maximum. The spec's 16384 assumed the full pool;
# sampling with replacement or growing P are both ruled out (P=32768 had the
# G3 artifact, 65536 OOMs). Recorded in summary.json.
M_SWEEP = (256, 1024, 8192)
GATE_TOL = 0.06

# pre-registered goal assignment from the 5-task eval set
G1_TASK, G2_TASK, GT_TASK, GA_TASK = 2, 4, 1, 3


# ----------------------------------------------------------------- helpers --

def kernel_raw(u_raw: np.ndarray, g_xyz: np.ndarray, eps: float = 0.4) -> np.ndarray:
    """UNNORMALIZED Gaussian kernel exp(-0.5||(u-g)/eps||^2), [P]. Composites use
    this (pool-normalized rows have per-goal scales that make max/sum
    scale-inconsistent); for single-goal rewards it gives the IDENTICAL index as
    the _delta_phi row by the encoder's exact r->alpha*r invariance."""
    d2 = ((u_raw - g_xyz[None]) / eps) ** 2
    return np.exp(-0.5 * d2.sum(-1))


def anytime_ball(cube_traj: np.ndarray, g_xyz: np.ndarray, r: float = SUCC_R) -> np.ndarray:
    """[n,H,3] obs-space cube traj -> [n] 0/1 anytime within r of g_xyz."""
    d = np.linalg.norm(cube_traj - g_xyz[None, None], axis=-1)
    return (d.min(-1) <= r).astype(float)


def min_dist(cube_traj: np.ndarray, g_xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(cube_traj - g_xyz[None, None], axis=-1).min(-1)


def disc_return(cube_traj: np.ndarray, lengths: np.ndarray, r_fn) -> float:
    """Mean over episodes of sum_t gamma^t r(cube_t) (analytic reward on the
    cube coordinate -- the proxy return under the requested reward)."""
    n, H, _ = cube_traj.shape
    disc = GAMMA ** np.arange(H)
    vals = []
    for i in range(n):
        L = max(int(lengths[i]), 1)
        vals.append(float((r_fn(cube_traj[i, :L]) * disc[:L]).sum()))
    return float(np.mean(vals))


def centered_ridge_fit(b_fit: np.ndarray, r_fit: np.ndarray, lam_mult: float = RIDGE_MULT):
    """Numpy twin of the centered encoder's regression (fit on the M samples the
    encoder sees). Returns (c, mu, rbar) for r_hat(u) = (B(u)-mu)^T c + rbar."""
    mu = b_fit.mean(0)
    rbar = float(r_fit.mean())
    bc = b_fit - mu[None]
    C = bc.T @ bc / len(b_fit)
    C = 0.5 * (C + C.T)
    lam = lam_mult * np.trace(C) / C.shape[0]
    m_c = bc.T @ (r_fit - rbar) / len(b_fit)
    c = np.linalg.solve(C + lam * np.eye(C.shape[0]), m_c)
    return c, mu, rbar


def recon_metrics(r_true: np.ndarray, r_hat: np.ndarray, hit: np.ndarray | None):
    from scipy.stats import spearmanr
    out = {"mse": float(np.mean((r_true - r_hat) ** 2)),
           "spearman": float(spearmanr(r_true, r_hat).statistic)}
    if hit is not None and 0 < hit.sum() < len(hit):
        # Mann-Whitney AUC (rank-based, ties averaged) -- numpy, no sklearn
        from scipy.stats import rankdata
        ranks = rankdata(r_hat)
        n1 = int(hit.sum())
        n0 = len(hit) - n1
        out["auc"] = float((ranks[hit.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    else:
        out["auc"] = None
    return out


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fam_defs(g1, g2, gt, ga, med_da):
    F = {}
    F["dense_g1"] = dict(task=G1_TASK, r=lambda u: np.exp(-np.linalg.norm(u - g1[None], axis=-1) / 0.8),
                         succ=lambda c, L: {"succ": float(anytime_ball(c, g1).mean())},
                         hit=lambda u: (np.linalg.norm(u - g1[None], axis=-1) <= SUCC_R))
    F["hard_g1"] = dict(task=G1_TASK, r=lambda u: kernel_raw(u, g1, eps=0.2),
                        succ=lambda c, L: {"succ": float(anytime_ball(c, g1).mean())},
                        hit=lambda u: (np.linalg.norm(u - g1[None], axis=-1) <= SUCC_R))
    F["region_g1"] = dict(task=G1_TASK,
                          r=lambda u: (np.abs(u - g1[None]) <= 0.6).all(-1).astype(float),
                          succ=lambda c, L: {"succ": float(
                              ((np.abs(c - g1[None, None]) <= 0.6).all(-1).any(-1)).mean())},
                          hit=lambda u: (np.abs(u - g1[None]) <= 0.6).all(-1))
    F["maxmix"] = dict(task=G1_TASK,
                       r=lambda u: np.maximum(kernel_raw(u, g1), kernel_raw(u, g2)),
                       succ=lambda c, L: {"succ": float(np.maximum(
                           anytime_ball(c, g1), anytime_ball(c, g2)).mean()),
                           "succ_g1": float(anytime_ball(c, g1).mean()),
                           "succ_g2": float(anytime_ball(c, g2).mean())},
                       hit=lambda u: np.minimum(np.linalg.norm(u - g1[None], axis=-1),
                                                np.linalg.norm(u - g2[None], axis=-1)) <= SUCC_R)
    F["summix"] = dict(task=G1_TASK,
                       r=lambda u: kernel_raw(u, g1) + 0.5 * kernel_raw(u, g2),
                       succ=lambda c, L: {"succ": float(anytime_ball(c, g1).mean()),
                                          "succ_g2": float(anytime_ball(c, g2).mean())},
                       hit=lambda u: (np.linalg.norm(u - g1[None], axis=-1) <= SUCC_R))
    for lam, tag in ((0.5, "tma05"), (1.0, "tma10")):
        F[tag] = dict(task=GT_TASK,
                      r=(lambda u, l=lam: kernel_raw(u, gt) - l * kernel_raw(u, ga)),
                      succ=lambda c, L: {"succ": float((anytime_ball(c, gt) *
                                                        (min_dist(c, ga) > AVOID_R)).mean()),
                                         "succ_reach": float(anytime_ball(c, gt).mean()),
                                         "frac_avoid_ok": float((min_dist(c, ga) > AVOID_R).mean()),
                                         "min_dist_avoid_mean": float(min_dist(c, ga).mean())},
                      hit=lambda u: (np.linalg.norm(u - gt[None], axis=-1) <= SUCC_R))
    F["negavoid"] = dict(task=GA_TASK, r=lambda u: -kernel_raw(u, ga),
                         succ=lambda c, L: {"succ": float((np.linalg.norm(
                             c - ga[None, None], axis=-1).mean(-1) > med_da).mean()),
                             "timeavg_dist": float(np.linalg.norm(
                                 c - ga[None, None], axis=-1).mean()),
                             "dataset_median_dist": med_da},
                         hit=None)
    return F


# --------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=16384)
    ap.add_argument("--stage", default="all", choices=["gate", "families", "oracle", "all"])
    ap.add_argument("--quick", action="store_true", help="smoke: 4 eps, M=256 only")
    ap.add_argument("--out-dir", default="results/spec30_g293")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t00 = time.time()
    run_dir = Path(sorted(glob.glob(str(Path(args.run_dir).expanduser())))[0])
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "flags.json").read_text())
    env_name = cfg["env_name"]
    acfg = cfg["agent"]
    assert acfg.get("measure_density_z_encode") == "dual_centered", "G29-3 needs the centered arm"
    assert not acfg.get("measure_density_hard", False), "expected the soft flagship variant"
    n_eps = 4 if args.quick else args.episodes
    m_sweep = (256,) if args.quick else M_SWEEP

    from eval_z_action_ranking import hit_spec_for, load_dataset  # noqa: E402
    from eval_density_zeroshot import build_pool  # noqa: E402
    from eval_zeroshot_rewards import batched_rollout_record, eval_tasks_for  # noqa: E402
    from t3_offpeak_multienv import load_agent_dims  # noqa: E402

    spec = hit_spec_for(env_name, acfg.get("measure_hit_spec"))
    blocks = spec.blocks
    (b0, b1, _e) = blocks[0]
    assert len(blocks) == 1, "cube-single: one block expected"
    xyz = slice(b0, b1)

    ds = load_dataset(env_name)
    obs_all = np.asarray(ds["observations"], np.float32)
    # utils-package collision fix (eval_z_action_ranking.py:496-503, verbatim)
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step,
                            obs_all.shape[-1], np.asarray(ds["actions"]).shape[-1])
    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    # ---- pool (audited builder, rng order = spec-18 harness) ----
    rng = np.random.default_rng(args.seed)
    pool_idx = rng.choice(len(obs_all), size=min(args.pool_size, len(obs_all)), replace=False)
    pool_obs = obs_all[pool_idx]
    u_raw, _b_online = build_pool(agent, blocks, pool_obs, spec)          # [P,3]
    u_sc = np.asarray(GCIQLFlowAgent._density_queries(jnp.asarray(pool_obs), blocks))
    enc_mod = agent._density_z_enc_module()
    b_enc = np.asarray(agent.network.select(enc_mod)(jnp.asarray(u_sc)), np.float64)
    P = len(u_raw)
    half = P // 2                                                          # fit | eval split
    print(f"[g293] {run_dir.name} step={args.checkpoint_step} P={P} enc_mod={enc_mod} "
          f"({time.time()-t00:.0f}s)", flush=True)

    import ogbench  # noqa: E402
    eval_tasks = eval_tasks_for(env_name)

    def make_envs(task_id, n):
        return [ogbench.make_env_and_datasets(eval_tasks[task_id], env_only=True)
                for _ in range(n)]

    def task_goal_obs(task_id):
        e = ogbench.make_env_and_datasets(eval_tasks[task_id], env_only=True)
        _o, info = e.reset(seed=0)
        g = np.asarray(info["goal"], np.float32)
        e.close()
        return g

    goals = {i: task_goal_obs(i) for i in range(1, 6)}
    H = 0
    _e0 = ogbench.make_env_and_datasets(eval_tasks[1], env_only=True)
    H = _e0.spec.max_episode_steps if _e0.spec and _e0.spec.max_episode_steps else 200
    _e0.close()
    print(f"[g293] H={H} goals cube xyz: "
          + " ".join(f"t{i}={goals[i][xyz].round(2)}" for i in range(1, 6)), flush=True)

    def encode(r_rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """[M] or [B,M] rewards on FRESH pool subset idx -> zhat [d] via the AGENT's
        own reward-sample API (serialized moments, identical solve to training)."""
        z = np.asarray(agent.density_z_from_reward_samples(
            jnp.asarray(u_sc[idx]), jnp.asarray(np.atleast_2d(r_rows), jnp.float32)))
        return z[0]

    # ---- the checkpoint's OWN frozen eval pool (ZEvalPoolState) ----
    # The native .816 was produced through THIS pool; the sample encoder's base
    # measure is kernel x pool density, so encodes are pool-composition
    # dependent. Primary encoding for every cell below = rewards evaluated AT
    # these 256 states; the fresh big-pool M-sweep is the robustness axis.
    pool_raw_frozen, _filled = (np.asarray(x) for x in
                                agent.network.select('density_zpool')())
    assert float(np.asarray(_filled)) == 1.0, "frozen eval pool never filled"
    u_sc_frozen = np.asarray(GCIQLFlowAgent._scale_block_cols(
        jnp.asarray(pool_raw_frozen), blocks))
    print(f"[g293] frozen pool M={len(pool_raw_frozen)} loaded from params", flush=True)

    def encode_frozen(r_vals: np.ndarray) -> np.ndarray:
        """[256] rewards at the frozen pool's states -> zhat [d] -- the
        pool-consistent, deployment-realistic encoding (256 reward labels)."""
        z = np.asarray(agent.density_z_from_reward_samples(
            jnp.asarray(u_sc_frozen), jnp.asarray(np.atleast_2d(r_vals), jnp.float32)))
        return z[0]

    def rollout(task_id, cond, mode, n=None):
        envs = make_envs(task_id, n or n_eps)
        traj, obs_traj, lengths, succ_env = batched_rollout_record(
            agent, envs, cond, H=H, mode=mode)
        for e in envs:
            e.close()
        cube = obs_traj[:, :, xyz].astype(np.float64)                       # obs units
        # freeze cube coords past episode end at their final value
        for i in range(len(cube)):
            L = max(int(lengths[i]), 1)
            cube[i, L:] = cube[i, L - 1]
        return cube, lengths, succ_env

    m_idx = {m: rng.choice(half, size=m, replace=False) for m in m_sweep}   # fit half only
    eval_idx = np.arange(half, P)

    # =================================================================== gate
    gate = {}
    if args.stage in ("gate", "all"):
        nat_a, nat_b, psens = {}, {}, {}
        for t in range(1, 6):
            cube, lengths, succ_env = rollout(t, goals[t], mode="goal")
            nat_a[t] = {"succ_env": float(succ_env.mean()),
                        "succ_ball": float(anytime_ball(cube, goals[t][xyz]).mean())}
            # harness path THROUGH THE FROZEN POOL: same kg object as native
            # sample_actions (_density_z_eval_samples), computed explicitly
            kg_f = np.asarray(GCIQLFlowAgent._delta_phi(
                jnp.asarray(goals[t][None]), jnp.asarray(pool_raw_frozen), blocks))[0]
            z_f = encode_frozen(kg_f)
            # fresh-pool twin (the pool-sensitivity finding)
            kg_fresh = kernel_raw(u_raw[m_idx[256]], goals[t][xyz])
            z_fresh = encode(kg_fresh, m_idx[256])
            psens[t] = {"cos_frozen_fresh": cos(z_f, z_fresh)}
            cube, lengths, succ_env = rollout(t, z_f, mode="z")
            nat_b[t] = {"succ_env": float(succ_env.mean()),
                        "succ_ball": float(anytime_ball(cube, goals[t][xyz]).mean()),
                        "z_norm": float(np.linalg.norm(z_f))}
        oa = float(np.mean([nat_a[t]["succ_env"] for t in nat_a]))
        ob = float(np.mean([nat_b[t]["succ_env"] for t in nat_b]))
        gate = {"native_sample_actions": nat_a, "overall_native": oa,
                "harness_z_api_frozen_pool": nat_b, "overall_harness": ob,
                "reference": 0.816, "tol": GATE_TOL,
                "pass": bool(abs(oa - 0.816) <= GATE_TOL and abs(ob - 0.816) <= GATE_TOL)}
        (out_dir / "gate.json").write_text(json.dumps(gate, indent=1))
        (out_dir / "pool_sensitivity.json").write_text(json.dumps(psens, indent=1))
        print(f"[g293] GATE native={oa:.3f} harness={ob:.3f} ref=.816 "
              f"pass={gate['pass']} pool_sens={[round(psens[t]['cos_frozen_fresh'],3) for t in psens]} "
              f"({time.time()-t00:.0f}s)", flush=True)
        if not gate["pass"] and not args.quick:
            print("[g293] GATE FAILED -- stopping, no new cells read", flush=True)
            return

    if args.stage == "gate":
        return

    # =============================================================== families
    g1, g2, gt, ga = goals[G1_TASK][xyz], goals[G2_TASK][xyz], goals[GT_TASK][xyz], goals[GA_TASK][xyz]
    med_da = float(np.median(np.linalg.norm(u_raw - ga[None], axis=-1)))

    fams = fam_defs(g1, g2, gt, ga, med_da)
    summary = {}
    if args.stage in ("families", "all"):
        # per-family, per-M cells
        z_frozen_by_fam = {}
        for name, fd in fams.items():
            cells = {}
            zs = {}
            # PRIMARY: reward evaluated at the frozen pool's 256 states
            z_pf = encode_frozen(fd["r"](pool_raw_frozen))
            z_frozen_by_fam[name] = z_pf
            for m in m_sweep:
                idx = m_idx[m]
                r_fit = fd["r"](u_raw[idx])
                z = encode(r_fit, idx)
                zs[m] = z
                c, mu, rbar = centered_ridge_fit(b_enc[idx], r_fit)
                r_hat = (b_enc[eval_idx] - mu[None]) @ c + rbar
                r_true = fd["r"](u_raw[eval_idx])
                hit = fd["hit"](u_raw[eval_idx]) if fd["hit"] is not None else None
                cells[m] = {"recon": recon_metrics(r_true, r_hat, hit),
                            "cos_vs_frozen": cos(z, z_pf)}
            stab = {f"cos_{m_sweep[i]}_{m_sweep[i+1]}": cos(zs[m_sweep[i]], zs[m_sweep[i + 1]])
                    for i in range(len(m_sweep) - 1)
                    if m_sweep[i] in zs and m_sweep[i + 1] in zs}
            cube, lengths, _se = rollout(fd["task"], z_pf, mode="z")
            roll_frozen = fd["succ"](cube, lengths)
            roll_frozen["disc_return"] = disc_return(cube, lengths, fd["r"])
            roll_frozen["episodes"] = n_eps
            rec = {"rollout_frozen": roll_frozen,
                   "cells": {str(m): cells[m] for m in cells}, "stability": stab,
                   "z_m8192": zs.get(8192, zs[max(zs)]).tolist(),
                   "task_reset": fd["task"]}
            (out_dir / f"fam_{name}.json").write_text(json.dumps(rec, indent=1, default=str))
            summary[name] = rec
            summary[name]["_z_m_top"] = zs[max(zs)]
            print(f"[g293] {name}: frozen:succ={roll_frozen['succ']:.2f} "
                  + " ".join(f"cosM{m}={cells[m]['cos_vs_frozen']:.3f}" for m in cells)
                  + f" stab={stab} ({time.time()-t00:.0f}s)", flush=True)

        # robustness rollouts: big-pool M-top index for the families where the
        # frozen-vs-fresh encodings disagree most
        m_top = max(m_sweep)
        disagree = sorted(summary,
                          key=lambda n: summary[n]["cells"][str(m_top)]["cos_vs_frozen"])[:3]
        for name in disagree:
            fd = fams[name]
            cube, lengths, _se = rollout(fd["task"], summary[name]["_z_m_top"], mode="z")
            rr = fd["succ"](cube, lengths)
            rr["disc_return"] = disc_return(cube, lengths, fd["r"])
            summary[name]["rollout_m_top"] = rr
            rec = json.loads((out_dir / f"fam_{name}.json").read_text())
            rec["rollout_m_top"] = rr
            (out_dir / f"fam_{name}.json").write_text(json.dumps(rec, indent=1, default=str))
            print(f"[g293] {name} m_top rollout: succ={rr['succ']:.2f} "
                  f"(frozen was {summary[name]['rollout_frozen']['succ']:.2f})", flush=True)
        for name in summary:
            summary[name].pop("_z_m_top", None)

        # ---- affine invariance (native_g1 kernel + maxmix base, M=256) ----
        aff = {}
        probe_s = obs_all[rng.choice(len(obs_all), 32, replace=False)]
        seed0 = jax.random.PRNGKey(7)

        def act_at(z):
            return np.asarray(agent.sample_actions_with_z(
                jnp.asarray(probe_s), jnp.asarray(np.tile(z[None], (32, 1)), jnp.float32),
                seed=seed0))

        for base_name, base_r in (("native_g1", lambda u: kernel_raw(u, g1)),
                                  ("maxmix", fams["maxmix"]["r"])):
            r0 = base_r(pool_raw_frozen)
            z0 = encode_frozen(r0)
            a0 = act_at(z0)
            rows = {}
            for tag, rt in (("plus1", r0 + 1.0), ("minus5", r0 - 5.0),
                            ("times2", 2.0 * r0), ("times0.1", 0.1 * r0)):
                zt = encode_frozen(rt)
                rows[tag] = {"cos": cos(z0, zt),
                             "max_abs_dz": float(np.abs(z0 - zt).max()),
                             "max_abs_da": float(np.abs(a0 - act_at(zt)).max())}
            aff[base_name] = rows
        # end-to-end confirmation rollout: native_g1 r+1 (5 eps)
        z_p1 = encode_frozen(kernel_raw(pool_raw_frozen, g1) + 1.0)
        z_b = encode_frozen(kernel_raw(pool_raw_frozen, g1))
        cube_b, L_b, _ = rollout(G1_TASK, z_b, mode="z", n=5)
        cube_p, L_p, _ = rollout(G1_TASK, z_p1, mode="z", n=5)
        aff["endtoend_plus1"] = {
            "succ_base": float(anytime_ball(cube_b, g1).mean()),
            "succ_plus1": float(anytime_ball(cube_p, g1).mean()),
            "traj_max_abs_diff": float(np.abs(cube_b - cube_p).max())}
        (out_dir / "affine.json").write_text(json.dumps(aff, indent=1))
        print(f"[g293] affine: {json.dumps(aff)[:400]}", flush=True)

        # ---- constant-reward fallback ----
        z_c = encode_frozen(np.ones(len(pool_raw_frozen), np.float32))
        z_neutral = np.ones_like(z_c)  # spec 30a: z_neutral = 1_d (||.|| = sqrt d)
        const = {"z_is_neutral": bool(np.allclose(z_c, z_neutral, atol=1e-5)),
                 "z_norm": float(np.linalg.norm(z_c)),
                 "cos_vs_ones": cos(z_c, z_neutral)}
        (out_dir / "const.json").write_text(json.dumps(const, indent=1))
        print(f"[g293] const: {const}", flush=True)

        # ---- policy sensitivity across families (M=max sweep) ----
        z_by_fam = dict(z_frozen_by_fam)
        acts = {n: act_at(z) for n, z in z_by_fam.items()}
        names = list(z_by_fam)
        sens = {f"{a}|{b}": float(np.linalg.norm(acts[a] - acts[b], axis=-1).mean())
                for i, a in enumerate(names) for b in names[i + 1:]}
        (out_dir / "sensitivity.json").write_text(json.dumps(
            {"pairs": sens, "encoding": "frozen_pool_256"}, indent=1))

        (out_dir / "summary.json").write_text(json.dumps(
            {"deviation": "M sweep {256,1024,8192}: 8192 is the fit-half maximum "
                          "under the disjoint fit/eval split; spec 30 P3's 16384 "
                          "assumed the full pool (no replacement, no pool growth)",
             "gate": gate, "families": {k: {m: {kk: vv for kk, vv in c.items()}
                                            for m, c in v["cells"].items()}
                                        for k, v in summary.items()},
             "affine": aff, "const": const, "sensitivity_mean": float(np.mean(list(sens.values())))},
            indent=1, default=str))

    # ================================================================= oracle
    if args.stage in ("oracle", "all") and not args.quick:
        from z_search import make_weights_fn

        d = int(acfg["measure_density_dim"])
        n_cand, n_states, p_sub = 256, 16, 2048
        zc = rng.standard_normal((n_cand, d))
        zc = np.sqrt(d) * zc / np.linalg.norm(zc, axis=-1, keepdims=True)
        sub = rng.choice(P, p_sub, replace=False)
        b_sub = np.asarray(agent.network.select("density_b")(jnp.asarray(u_sc[sub])))
        temp = float(acfg.get("measure_density_temp", 1.0))
        # worst two families by M=256 rollout succ (fallback: tma10, negavoid)
        worst = sorted(((summary[n]["rollout_frozen"].get("succ", 1.0), n)
                        for n in summary), key=lambda x: x[0])[:2] if summary else \
            [(None, "tma10"), (None, "negavoid")]
        oracle = {}
        for _s, name in worst:
            fd = fams[name]
            r_sub = fd["r"](u_raw[sub])
            init_obs = []
            e = ogbench.make_env_and_datasets(eval_tasks[fd["task"]], env_only=True)
            for k in range(n_states):
                o, _ = e.reset(seed=1000 + k)
                init_obs.append(o)
            e.close()
            init_obs = np.asarray(init_obs, np.float32)

            weights_of = make_weights_fn(agent, b_sub, temp, init_obs,
                                         seed=jax.random.PRNGKey(11))

            def J(zbatch):
                return weights_of(np.asarray(zbatch)) @ r_sub

            z_cf = encode_frozen(fd["r"](pool_raw_frozen))
            j_cf = float(J(z_cf[None])[0])
            j_all = J(zc)
            k_best = int(np.argmax(j_all))
            cube, lengths, _se = rollout(fd["task"], zc[k_best], mode="z")
            oracle[name] = {"J_closed_form": j_cf, "J_search_best": float(j_all[k_best]),
                            "J_search_median": float(np.median(j_all)),
                            "cos_zcf_zbest": cos(z_cf, zc[k_best]),
                            "rollout_search": fd["succ"](cube, lengths),
                            "n_candidates": n_cand}
            print(f"[g293] oracle {name}: Jcf={j_cf:.4f} Jbest={oracle[name]['J_search_best']:.4f} "
                  f"succ_search={oracle[name]['rollout_search']}", flush=True)
        (out_dir / "oracle.json").write_text(json.dumps(oracle, indent=1))

    print(f"[g293] DONE ({time.time()-t00:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
