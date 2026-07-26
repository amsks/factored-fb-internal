#!/usr/bin/env python
"""scripts/rescuing_fb/eval_smb_measure.py -- Spec 11 item 5.3: the SMB Wing B
(measure-calibration) runner -- the wing no published FB eval has.

Protocol (spec 11 section 2): for each conditioning c in {5 native task goals},
roll the checkpoint's OWN policy (batched_rollout_record, 25 eps x 200 steps),
record the cube trajectory, and compare the model's PREDICTED discounted
occupancies against the empirical discounted visitation of those same rollouts:
    gt(X) = mean_eps sum_t gamma^t 1[cube(s_{t+1}) in X].
One rollout scores the whole frozen query battery simultaneously, so this is
independent of actor quality, decode quality and reward-inference quality: a
Wing-B failure is the OBJECT.

Predictor adapters (each method's own critic, select('critic') path):
  diagonal critics (fb_forward_z=False):  M_hat(s0,a0,x) = critic(s0, x, a0)
      -- for every OFF-diagonal query this is the WRONG-POLICY surrogate (the
      column x is evaluated under conditioning x, not c); diagonal-form
      checkpoints cannot express M(s,a,z_c,x), and that mismatch IS the B2
      measurement (documented in the output).
  ZX critics (fb_forward_z=True):  M_hat = F(s0,a0,z=psi(g_c))^T B(x)/sqrt(d).
  box/union queries (no single center): prediction = mean of the readout over
      the <= 8 frozen dataset states inside the region (`rep_rows` from the
      artifact) -- the KERNEL-AVERAGE surrogate (kernel mass != density; the
      spec-10 modification is what would make this calibrated).

(s0, a0): the first env reset states of each episode (seed-deterministic, the
exact states the rollout replays) and the policy's own deterministic action
there; predictions are recorded per episode and averaged.

Outputs:
  B1  diagonal calibration: own-ball predictions vs gt across 5 goals x
      episodes -- Spearman, median relative error raw + after one GLOBAL
      affine fit.
  B2  full-battery calibration per query family (balls / boxes / unions):
      Spearman + relative error (raw / affine), and the union additivity
      defect |M(X1 u X2) - M(X1) - M(X2)| / gt(X1 u X2).
  B3  reward-integration accuracy: for each Wing-A T1/T3/T4 task and each
      conditioning, Q_hat = mean_j r_j M_hat(s0,a0,z_c,g_j) over the m=2000
      relabel sample vs the MC discounted reward of that conditioning's own
      rollouts -- rank correlation + regression slope/R^2 (the number that
      licenses or kills the "integrate r dM" claim per method).

Run (one free GPU):
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<free> MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_smb_measure.py \\
    --run-dir '<run dir>' --tasks-json analysis/rescuing-fb/smb_cube_single_v1.json
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
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from smb_common import (  # noqa: E402  (numpy-only at module level)
    CUBE_OBS, EPS_OBS, EVAL_TASKS, GAMMA, SPEC, _policy_call, affine_fit,
    additivity_defect, batched_rollout_record, capture_reset_obs, cube_m_to_ou,
    discounted_time_in, in_ball, make_measure_readout, median_rel_err,
    per_episode_discounted, query_member, regression_slope_r2, reward_dirac,
    reward_occhoice, reward_union, spearman, to_native,
)
from t6_gpi_eval import load_agent_flexible, make_psi  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--tasks-json", required=True)
    ap.add_argument("--conditionings", default="native",
                    help="only 'native' (the 5 task goals) is implemented")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--relabel-size", type=int, default=2000,
                    help="B3 reward-sample budget m (spec 11: 2000)")
    ap.add_argument("--dataset-path", default="~/.ogbench/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    return ap


def build_b3_tasks(tasks_art: dict, goal_ou: dict) -> list:
    """The Wing-A T1/T3/T4 tasks B3 integrates over: (name, reward_fn) with
    reward_fn mapping cube positions (obs units) -> rewards."""
    tasks = []
    for i in sorted(goal_ou):
        tasks.append((f"t1_dirac_t{i}",
                      lambda c, g=np.asarray(goal_ou[i]): reward_dirac(c, g)))
    for pair in tasks_art["t3_occhoice"]["pairs"]:
        gS = np.asarray(pair["stable"]["center"], np.float64)
        gT = np.asarray(pair["transient"]["center"], np.float64)
        tasks.append((f"t3_{pair['name']}",
                      lambda c, gS=gS, gT=gT, wS=float(pair["w_S"]),
                      wT=float(pair["w_T"]): reward_occhoice(c, gS, gT, wS, wT)))
    for task in tasks_art["t4_union3"]["tasks"]:
        cs = [np.asarray(c, np.float64) for c in task["centers"]]
        tasks.append((f"t4_{task['name']}",
                      lambda c, cs=cs: reward_union(c, cs)))
    return tasks


def main() -> int:
    args = build_parser().parse_args()
    assert args.conditionings == "native", "only --conditionings native exists"
    tasks_art = json.loads(Path(args.tasks_json).read_text())
    battery = tasks_art["wingb_battery"]

    import jax
    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()

    t00 = time.time()
    agent, d, cfg = load_agent_flexible(run_dir, args.checkpoint_step,
                                        arch_explicit=False, d=0,
                                        actor_cond="z", actor_vf_cond="none")
    fbz = bool(cfg.get("fb_forward_z", False))
    measure_only = bool(cfg.get("measure_only", False))
    actor_mode = "z" if cfg.get("actor_cond") == "z" else "goal"
    H = args.horizon
    print(f"[smbB] restored {run_dir.name} step={args.checkpoint_step} d={d} "
          f"fbz={fbz} measure_only={measure_only} actor_mode={actor_mode}", flush=True)

    # ---- dataset: battery rep rows index the SAME loading convention as
    # smb_tasks.py (renewal_head.load_dataset row order) ----
    from renewal_head import load_dataset
    n_tr = int(tasks_art["dataset"]["n_transitions"])
    obs_ds, _nobs, _acts, _term = load_dataset(args.dataset_path, n_tr)
    cube_ds = obs_ds[:, CUBE_OBS].astype(np.float64)
    env0 = ogbench.make_env_and_datasets(EVAL_TASKS[1], env_only=True)
    act_dim = int(env0.action_space.shape[0])
    env0.close()

    psi = make_psi(agent, src="critic" if (measure_only or fbz) else "value",
                   act_dim=act_dim, fbz=fbz)
    readout = make_measure_readout(agent, d, fbz, act_dim)

    goal_obs, goal_ou = {}, {}
    for i in range(1, 6):
        env = ogbench.make_env_and_datasets(EVAL_TASKS[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        gx = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        goal_ou[i] = cube_m_to_ou(gx)
        env.close()

    rng = np.random.default_rng(args.seed)
    idx_rel = rng.choice(len(obs_ds), size=min(args.relabel_size, len(obs_ds)),
                         replace=False)
    S_rel = obs_ds[idx_rel]
    cube_rel = cube_ds[idx_rel]
    b3_tasks = build_b3_tasks(tasks_art, goal_ou)
    b3_rewards = {name: np.asarray(rfn(cube_rel), np.float64)
                  for name, rfn in b3_tasks}

    # ---- assemble the goal-obs matrix ONE readout call scores per cond ----
    # layout: [balls 64 | union components (flattened rep rows) | box rep rows
    #          | relabel m | own-goal 1]
    balls = battery["balls"]
    boxes = battery["boxes"]
    unions = battery["unions"]
    ball_rows = [int(b["row"]) for b in balls]
    union_rep, union_slices = [], []      # per union: (slice comp1, slice comp2)
    for u in unions:
        s1 = slice(len(union_rep), len(union_rep) + len(u["rep_rows"][0]))
        union_rep += [int(r) for r in u["rep_rows"][0]]
        s2 = slice(len(union_rep), len(union_rep) + len(u["rep_rows"][1]))
        union_rep += [int(r) for r in u["rep_rows"][1]]
        union_slices.append((s1, s2))
    box_rep, box_slices = [], []
    for b in boxes:
        s = slice(len(box_rep), len(box_rep) + len(b["rep_rows"]))
        box_rep += [int(r) for r in b["rep_rows"]]
        box_slices.append(s)
    n_b, n_u, n_x = len(ball_rows), len(union_rep), len(box_rep)
    goals_fixed = np.concatenate([
        obs_ds[ball_rows], obs_ds[union_rep], obs_ds[box_rep], S_rel], axis=0)
    o_ball = slice(0, n_b)
    o_union = n_b
    o_box = n_b + n_u
    o_rel = slice(n_b + n_u + n_x, n_b + n_u + n_x + len(S_rel))
    o_own = n_b + n_u + n_x + len(S_rel)

    queries = ([{"type": "ball", "center": b["center"], "eps": b["eps"]} for b in balls]
               + [{"type": "union2", "centers": u["centers"], "eps": u["eps"]}
                  for u in unions]
               + [{"type": "box", "center": b["center"], "half": b["half"]}
                  for b in boxes])
    q_family = (["ball"] * len(balls) + ["union"] * len(unions) + ["box"] * len(boxes))

    pools = {}

    def pool(i):
        # terminate_at_goal=False: OGBench ends the episode on task success by
        # default, which TRUNCATES the ground-truth discounted occupancy at first
        # arrival for native conditionings -- exactly the dwell mass Wing B exists
        # to measure. Wing A keeps the stock env (success metrics use the standard
        # protocol); the calibration wing must not.
        if i not in pools:
            pools[i] = [ogbench.make_env_and_datasets(EVAL_TASKS[i], env_only=True,
                                                      terminate_at_goal=False)
                        for _ in range(args.episodes)]
        return pools[i]

    per_cond = {}
    for i in range(1, 6):
        z_cond = np.asarray(psi(goal_obs[i]), np.float64).astype(np.float32)
        cond = goal_obs[i] if actor_mode == "goal" else z_cond
        envs = pool(i)
        s0 = capture_reset_obs(envs)
        cond_b = np.tile(np.asarray(cond, np.float32)[None], (len(envs), 1))
        a0 = np.clip(np.asarray(_policy_call(
            agent, actor_mode, s0, cond_b, jax.random.PRNGKey(0))), -1, 1)
        traj, _obs_traj, ln = batched_rollout_record(agent, envs, cond, H,
                                                     mode=actor_mode)
        tou = cube_m_to_ou(traj)
        # ---- ground truths (per episode) ----
        gt_q = np.stack([discounted_time_in(tou, ln, lambda c, q=q: query_member(q, c))
                         for q in queries], axis=1)              # [eps, nq]
        own_q = {"type": "ball", "center": list(map(float, goal_ou[i])), "eps": EPS_OBS}
        gt_own = discounted_time_in(tou, ln, lambda c: query_member(own_q, c))
        gt_union_comp = np.stack(
            [np.stack([discounted_time_in(
                tou, ln, lambda c, ctr=u["centers"][k]: in_ball(c, ctr, u["eps"]))
                for k in (0, 1)], axis=1) for u in unions], axis=1)  # [eps, nu, 2]
        # ---- predictions (per episode) ----
        goals_all = np.concatenate([goals_fixed, goal_obs[i][None]], axis=0)
        M = readout(s0, a0, goals_all, z_cond=z_cond if fbz else None)  # [eps, G]
        pred_ball = M[:, o_ball]                                    # [eps, 64]
        pred_union, pred_union_comp = [], []
        for (s1, s2) in union_slices:
            m1 = M[:, o_union + np.arange(s1.start, s1.stop)].mean(1)
            m2 = M[:, o_union + np.arange(s2.start, s2.stop)].mean(1)
            pred_union_comp.append(np.stack([m1, m2], axis=1))
            # kernel-average surrogate over BOTH components' rep states:
            both = M[:, o_union + np.arange(s1.start, s2.stop)].mean(1)
            pred_union.append(both)
        pred_union = np.stack(pred_union, axis=1)                   # [eps, nu]
        pred_union_comp = np.stack(pred_union_comp, axis=1)         # [eps, nu, 2]
        pred_box = np.stack([M[:, o_box + np.arange(s.start, s.stop)].mean(1)
                             for s in box_slices], axis=1)          # [eps, nx]
        pred_q = np.concatenate([pred_ball, pred_union, pred_box], axis=1)
        pred_own = M[:, o_own]
        M_rel = M[:, o_rel]                                         # [eps, m]
        # ---- B3 pieces ----
        b3 = {}
        for name, rfn in b3_tasks:
            r = b3_rewards[name]
            q_hat = float((M_rel * r[None, :]).mean(1).mean())
            mc = float(per_episode_discounted(tou, ln, rfn).mean())
            b3[name] = {"Q_hat": q_hat, "G_mc": mc}
        per_cond[i] = {
            "success_gt_own_mean": float(gt_own.mean()),
            "gt_q": gt_q, "gt_own": gt_own, "gt_union_comp": gt_union_comp,
            "pred_q": pred_q, "pred_own": pred_own,
            "pred_union": pred_union, "pred_union_comp": pred_union_comp,
            "b3": b3, "lengths": ln.tolist(),
        }
        print(f"[smbB] cond t{i}: gt_own={gt_own.mean():.3f} "
              f"pred_own={pred_own.mean():.3f} gt_q>0 for "
              f"{int((gt_q.mean(0) > 0).sum())}/{gt_q.shape[1]} queries "
              f"({time.time()-t00:.0f}s)", flush=True)

    # ======================= B1: diagonal calibration ==========================
    b1_pred = np.concatenate([per_cond[i]["pred_own"] for i in range(1, 6)])
    b1_gt = np.concatenate([per_cond[i]["gt_own"] for i in range(1, 6)])
    a1, c1 = affine_fit(b1_pred, b1_gt)
    raw_err, raw_n = median_rel_err(b1_pred, b1_gt)
    aff_err, _ = median_rel_err(a1 * b1_pred + c1, b1_gt)
    B1 = {"n_points": int(len(b1_pred)),
          "spearman": spearman(b1_pred, b1_gt),
          "median_rel_err_raw": raw_err, "n_gt_pos": raw_n,
          "affine": {"a": a1, "b": c1}, "median_rel_err_affine": aff_err,
          "gt_own_mean_per_cond": {f"t{i}": float(per_cond[i]["gt_own"].mean())
                                   for i in range(1, 6)},
          "pred_own_mean_per_cond": {f"t{i}": float(per_cond[i]["pred_own"].mean())
                                     for i in range(1, 6)}}

    # ======================= B2: full-battery calibration ======================
    fam_names = ("ball", "union", "box")
    fam_mask = {f: np.array([x == f for x in q_family]) for f in fam_names}
    all_pred = np.concatenate([per_cond[i]["pred_q"].mean(0) for i in range(1, 6)])
    all_gt = np.concatenate([per_cond[i]["gt_q"].mean(0) for i in range(1, 6)])
    a2, c2 = affine_fit(all_pred, all_gt)
    B2 = {"affine_global": {"a": a2, "b": c2},
          "per_conditioning": {}, "per_family": {}}
    for f in fam_names:
        preds = np.concatenate([per_cond[i]["pred_q"].mean(0)[fam_mask[f]]
                                for i in range(1, 6)])
        gts = np.concatenate([per_cond[i]["gt_q"].mean(0)[fam_mask[f]]
                              for i in range(1, 6)])
        raw_err, n_pos = median_rel_err(preds, gts)
        aff_err, _ = median_rel_err(a2 * preds + c2, gts)
        B2["per_family"][f] = {
            "n": int(len(preds)), "n_gt_pos": n_pos,
            "spearman": spearman(preds, gts),
            "median_rel_err_raw": raw_err, "median_rel_err_affine": aff_err}
    for i in range(1, 6):
        p = per_cond[i]["pred_q"].mean(0)
        g = per_cond[i]["gt_q"].mean(0)
        B2["per_conditioning"][f"t{i}"] = {
            "spearman_all": spearman(p, g),
            "n_gt_pos": int((g > 1e-3).sum())}
    defects, defects_abs = [], []
    for i in range(1, 6):
        pu = per_cond[i]["pred_union"].mean(0)            # [nu]
        pc = per_cond[i]["pred_union_comp"].mean(0)       # [nu, 2]
        gu_idx = fam_mask["union"]
        gu = per_cond[i]["gt_q"].mean(0)[gu_idx]
        for k in range(len(pu)):
            defects_abs.append(float(abs(pu[k] - (pc[k, 0] + pc[k, 1]))))
            if gu[k] > 1e-3:                              # normalized defect needs gt mass
                defects.append(additivity_defect(pu[k], pc[k, 0], pc[k, 1], gu[k]))
    B2["union_additivity_defect"] = {
        "n_total": len(defects_abs), "n_gt_pos": len(defects),
        "median": float(np.median(defects)) if defects else None,
        "mean": float(np.mean(defects)) if defects else None,
        "median_abs_unnormalized": float(np.median(defects_abs))}

    # ======================= B3: reward integration ============================
    names = [n for n, _ in b3_tasks]
    q_hats, g_mcs, cells = [], [], {}
    for name in names:
        for i in range(1, 6):
            rec = per_cond[i]["b3"][name]
            q_hats.append(rec["Q_hat"])
            g_mcs.append(rec["G_mc"])
            cells[f"{name}@t{i}"] = rec
    slope, intercept, r2 = regression_slope_r2(q_hats, g_mcs)
    B3 = {"n_pairs": len(q_hats), "spearman": spearman(q_hats, g_mcs),
          "slope": slope, "intercept": intercept, "r2": r2, "cells": cells}

    results = {
        "meta": {
            "spec": SPEC, "wing": "B",
            "tasks_json": str(Path(args.tasks_json).resolve()),
            "tasks_version": tasks_art["version"],
            "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
            "d": d, "fb_forward_z": fbz, "measure_only": measure_only,
            "actor_mode": actor_mode, "episodes": args.episodes, "horizon": H,
            "relabel_size": int(len(S_rel)), "gamma": GAMMA, "seed": args.seed,
            "predictor": ("ZX: F(s0,a0,z=psi(g_cond))^T B(x)/sqrt(d)" if fbz else
                          "diagonal: critic(s0, x, a0) -- the WRONG-POLICY "
                          "surrogate for off-diagonal queries (spec 11 B2: "
                          "diagonal-form checkpoints cannot express M(s,a,z_c,x); "
                          "this mismatch is the measurement)"),
            "box_union_surrogate": "kernel-average: mean readout over the frozen "
                                   "rep_rows dataset states inside the region "
                                   "(kernel mass != density)",
            "gt_def": "mean_eps sum_t gamma^t 1[cube(s_{t+1}) in X] over the "
                      "conditioning's own rollouts",
        },
        "B1": B1, "B2": B2, "B3": B3,
        "per_conditioning_b3": {f"t{i}": per_cond[i]["b3"] for i in range(1, 6)},
    }

    outp = Path(args.out) if args.out else \
        REPO / "analysis" / "rescuing-fb" / f"smb_wingB_{run_dir.name}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(to_native(results), indent=2))
    print(f"[smbB] B1 spearman={B1['spearman']:.3f} raw_err={B1['median_rel_err_raw']} "
          f"| B2 ball spearman={B2['per_family']['ball']['spearman']:.3f} "
          f"| B3 spearman={B3['spearman']:.3f} slope={B3['slope']:.3f}", flush=True)
    print(f"[smbB] DONE ({time.time()-t00:.0f}s) -> {outp}", flush=True)
    for i in list(pools):
        for e in pools.pop(i):
            if hasattr(e, "close"):
                e.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
