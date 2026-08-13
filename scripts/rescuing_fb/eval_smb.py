#!/usr/bin/env python
"""scripts/rescuing_fb/eval_smb.py -- Spec 11 item 5.2: the SMB Wing A (control)
runner. Wraps the proven eval_zeroshot_rewards.py machinery (checkpoint restore,
z inference routes, batched lockstep rollouts, GPI) and scores every tier by the
spec-11 primary metric: ACHIEVED DISCOUNTED REWARD (gamma=0.99) computed from
the recorded post-step cube trajectory against the task's reward function.
Success flags stay secondary where defined. Tasks T3/T4/T5 come from the frozen
artifact built by smb_tasks.py (never regenerated here).

Tiers (--tiers): T0 native comes from t6_gpi_eval / the run's own eval -- NOT
run here.
  t1  dirac + dense per native task: success + achieved reward (+ T7 dwell).
  t2  two-goal max-mixtures (10 pairs): success-at-either + achieved reward;
      --gpi adds GPI over {z_r, psi(g_i), psi(g_j)} (t6_gpi_eval protocol).
  t3  occupancy-weighted choice (the R-discriminator): per pair, BOTH dedicated
      single-region arms are rolled (their max = the oracle), then the main arm
      (inferred z; --gpi = GPI over the two region candidates with the method's
      own critic readout at z_r -- select('critic') path, so GC-IQL ranks by Q
      and measure checkpoints by their M readout). Reports achieved
      reward/oracle + chose-correct-region rate + dwell columns.
  t4  union-of-3-balls: end-in-any-ball success + achieved reward (+ dwell).
  t5  corridor reach-avoid, lambda in {0 (control), 1, 5}: success-at-A,
      time-in-B, achieved SIGNED reward, deflection = timeB(lam)/timeB(0).
  t6  feature-linear rewards: achieved reward vs the random-z control AND the
      best-candidate control (GPI over the 5 native conditionings, critic
      readout at z_tilde) -- the second control separates "learned z~" from
      "any reasonable policy scores".

Cross-cutting A-z axis (--oracle-z): every tier swaps the INFERRED conditioning
for the oracle one -- psi(g*)/goal obs for goal families, the occupancy-correct
region candidate for t3, the best-of-candidates for t4, psi(g_A) for t5, and
z_tilde itself for t6 (ZX only reaches it honestly; goal-mode actors get the
decoded proxy). Decomposes control failures into inference vs object/actor.

Conditioning spaces: actor deployment follows eval_zeroshot_rewards (actor_cond
'z' consumes z_r; 'goal' decodes z_r to the argmax-cos goal-proxy dataset obs).
psi source: CRITIC psi whenever measure_only or fb_forward_z (under measure_only
the value net is never trained -- renewal_head's reasoning; NOTE this widens
eval_zeroshot_rewards' rule, which kept value-psi for measure-only goal-actors,
so that z_r and the critic-phi GPI readout live in the SAME space), value psi
otherwise (the GC-IQL anchor).

Run (one free GPU; rollouts are the cost):
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<free> MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_smb.py \\
    --run-dir '/home/adityamohan/exp_spec07/factored-fb/GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50/vis_s0' \\
    --tasks-json analysis/rescuing-fb/smb_cube_single_v1.json \\
    --tiers t1,t2,t3,t4,t5,t6 --episodes 25 --horizon 200 --gpi
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from smb_common import (  # noqa: E402  (numpy-only at module level)
    AVOID_R, EPS_OBS, EVAL_TASKS, GAMMA, SPEC, _policy_call,
    ball_candidate_indices, batched_gpi_rollout_record, batched_rollout_record,
    capture_reset_obs, chosen_region, cube_m_to_ou, cube_obs_units,
    discounted_time_in, final_in_any_ball, frac_time_in_ball, in_ball,
    infer_z_route, make_measure_readout, per_episode_discounted,
    reward_corridor, reward_dense, reward_dirac, reward_linear, reward_mixture,
    reward_occhoice, reward_union, sample_sphere_z, to_native,
    z_to_goal_proxy_idx,
)
from full_measure import FullMeasure, select_z_by_measure  # noqa: E402
from t6_gpi_eval import (  # noqa: E402
    batched_gpi_rollout as _unused_gpi,  # noqa: F401  (record variant used instead)
    cosine, decode_cube, load_agent_flexible, m_goal, m_pair, make_psi,
    make_q_at_z, rescale,
)

ALL_TIERS = ("t1", "t2", "t3", "t4", "t5", "t6")


def measure_select(cands, qr_fn):
    """Spec 10 §2 z-search: rank candidate conditionings by the full measure's
    own ∫r dM estimate at (s0, a0~pi(s0,z)) and act with the argmax, via
    full_measure.select_z_by_measure. cands: [(name, cond)]; qr_fn(cond) ->
    full_measure q_r dict. Returns the select_z_by_measure dict + per-candidate
    q_r details (n_live etc.) for the output JSON."""
    details = {}

    def score_fn(cond):
        d = qr_fn(cond)
        details[id(cond)] = d
        return d["q"]

    out = select_z_by_measure(cands, score_fn)
    out["details"] = {name: details[id(c)] for name, c in cands}
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--tasks-json", required=True,
                    help="frozen task artifact from smb_tasks.py")
    ap.add_argument("--tiers", default=",".join(ALL_TIERS),
                    help=f"comma-sep subset of {ALL_TIERS} (t0 native: use t6_gpi_eval)")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--z-inference",
                    choices=["moment", "ridge", "retrieval", "cmoment",
                             "cmoment_trained"],
                    default="moment",
                    help="cmoment: the density-corrected moment (spec 10 §1) "
                         "z ∝ mean_j [r_j / N-hat(g_j)] psi(g_j) via "
                         "full_measure.corrected_moment_z (ess reported). "
                         "cmoment_trained (phase 2b): same, with N-hat from the "
                         "checkpoint's TRAINED nhat_head instead of the KDTree "
                         "(needs measure_nhat_head=True checkpoints)")
    ap.add_argument("--retrieval-k", type=int, default=8)
    ap.add_argument("--ridge-lambda", type=float, default=1e-2)
    ap.add_argument("--density-cap-q", type=float, default=0.0,
                    help="winsorize the density's upper tail at this quantile "
                         "inside q_r comparisons (0 = off); tames the 1/N-hat "
                         "importance tail at thin-support regions (T3 fix)")
    ap.add_argument("--z-select", choices=["off", "measure"], default="off",
                    help="measure: rank the candidate set (region/native "
                         "candidates + the inferred z) by the full measure's "
                         "∫r dM at (s0, a0~pi(s0,z)) and roll the argmax "
                         "(spec 10 §2 z-search; tiers t3/t4/t5/t6)")
    ap.add_argument("--gpi", action="store_true",
                    help="GPI over candidate conditionings (t2/t3/t4 main arms)")
    ap.add_argument("--oracle-z", action="store_true",
                    help="A-z axis: oracle conditioning instead of inferred z")
    ap.add_argument("--relabel-size", type=int, default=50000,
                    help="reward-labeled sample for z inference (existing-harness "
                         "default; spec 11 states m=2000 -- pass 2000 for that arm)")
    ap.add_argument("--candidate-pool", type=int, default=50000,
                    help="dataset subsample for region candidates + goal-proxy decode")
    ap.add_argument("--n-linear", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in ALL_TIERS]
    assert not unknown, f"unknown tiers {unknown}; choose from {ALL_TIERS}"
    tasks_art = json.loads(Path(args.tasks_json).read_text())
    assert tasks_art.get("version"), f"{args.tasks_json} is not an SMB artifact"

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), f"{run_dir} has no flags.json"

    t00 = time.time()
    agent, d, cfg = load_agent_flexible(run_dir, args.checkpoint_step,
                                        arch_explicit=False, d=0,
                                        actor_cond="z", actor_vf_cond="none")
    fbz = bool(cfg.get("fb_forward_z", False))
    measure_only = bool(cfg.get("measure_only", False))
    use_critic_psi = measure_only or fbz          # see module docstring
    actor_mode = "z" if cfg.get("actor_cond") == "z" else "goal"
    H = args.horizon
    print(f"[smb] restored {run_dir.name} step={args.checkpoint_step} d={d} "
          f"actor_cond={cfg.get('actor_cond')} (mode={actor_mode}) fbz={fbz} "
          f"psi_src={'critic' if use_critic_psi else 'value'} gpi={args.gpi} "
          f"oracle_z={args.oracle_z} route={args.z_inference}", flush=True)

    # ---- dataset samples: S_rel (z inference) + S_cand (candidates/decode) ----
    _e0, ds1, _ = ogbench.make_env_and_datasets(EVAL_TASKS[1])
    n_ds = len(ds1["observations"])
    act_dim = int(np.asarray(ds1["actions"]).shape[-1])
    psi = make_psi(agent, src="critic" if use_critic_psi else "value",
                   act_dim=act_dim, fbz=fbz)
    q_at_z = make_q_at_z(agent, d, fbz=fbz)
    rng = np.random.default_rng(args.seed)
    idx_rel = rng.choice(n_ds, size=min(args.relabel_size, n_ds), replace=False)
    obs_all = np.asarray(ds1["observations"], np.float32)
    S_rel = obs_all[idx_rel]
    cube_rel = cube_obs_units(S_rel)
    Psi_rel = psi(S_rel).astype(np.float64)
    match_data = float(np.linalg.norm(Psi_rel, axis=-1).mean())
    if args.candidate_pool == args.relabel_size:
        S_cand, cube_cand, Psi_cand = S_rel, cube_rel, Psi_rel
    else:
        idx_cand = np.random.default_rng(args.seed + 100).choice(
            n_ds, size=min(args.candidate_pool, n_ds), replace=False)
        S_cand = obs_all[idx_cand]
        cube_cand = cube_obs_units(S_cand)
        Psi_cand = psi(S_cand).astype(np.float64)

    # ---- eval-layer full measure (spec 10 L1+L2): cmoment + measure-ranked z ----
    need_fm = args.z_inference in ("cmoment", "cmoment_trained") \
        or (args.z_select == "measure")
    fm, nobs_all, readout_m = None, None, None
    if need_fm:
        cube_all = cube_obs_units(obs_all)
        nobs_all = np.asarray(ds1["next_observations"], np.float32)
        nhat_fn = None
        if args.z_inference == "cmoment_trained":
            # phase 2b: the checkpoint's trained N-hat head replaces the KDTree
            # denominator (floor still applied inside FullMeasure).
            from smb_common import make_trained_nhat_fn
            nhat_fn = make_trained_nhat_fn(agent, psi, cube_all, nobs_all)
        fm = FullMeasure(cube_all, eps=EPS_OBS, nhat_fn=nhat_fn)
        readout_m = make_measure_readout(agent, d, fbz, act_dim)
        print(f"[smb] FullMeasure over {fm.n} rows, eps={fm.eps}, "
              f"floor={fm.floor:.2e}, "
              f"nhat={'trained_head' if nhat_fn is not None else 'kdtree'} "
              f"(z_inference={args.z_inference}, z_select={args.z_select})",
              flush=True)

    goal_obs, goal_xyz = {}, {}
    for i in range(1, 6):
        env = ogbench.make_env_and_datasets(EVAL_TASKS[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        goal_xyz[i] = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        env.close()
    psi_g = {i: psi(goal_obs[i]).astype(np.float64) for i in goal_obs}
    goal_ou = {i: cube_m_to_ou(goal_xyz[i]) for i in goal_obs}

    pools = {}

    def pool(i):
        if i not in pools:
            pools[i] = [ogbench.make_env_and_datasets(EVAL_TASKS[i], env_only=True)
                        for _ in range(args.episodes)]
        return pools[i]

    def close_all():
        for i in list(pools):
            for e in pools.pop(i):
                if hasattr(e, "close"):
                    e.close()

    def infer(rewards):
        if args.z_inference in ("cmoment", "cmoment_trained"):
            # spec 10 §1: density-corrected moment, weights 1/N-hat at the
            # reward samples' cube positions (N-hat = KDTree kernel count, or
            # the trained head under cmoment_trained); ess flags degeneracy.
            r = np.asarray(rewards, np.float64)
            out = fm.corrected_moment_z(Psi_rel, r, cube_rel, match_data)
            diag = {"route": args.z_inference, "n_pos": int((r > 0).sum()),
                    "n_neg": int((r < 0).sum()), "ess": out["ess"],
                    "z_raw_norm": out["raw_norm"], "match_data_norm": match_data}
            return out["z"], diag
        return infer_z_route(Psi_rel, rewards, args.z_inference, match_data,
                             retrieval_k=args.retrieval_k,
                             ridge_lambda=args.ridge_lambda)

    def measure_qr_fn(r_samples, variant):
        """qr_fn factory for measure_select: the full measure's ∫r dM at the
        variant's reset states s0 with a0 ~ pi(s0, candidate) -- ball values
        from the checkpoint's own critic readout at dataset next-obs rows."""
        import jax
        s0 = capture_reset_obs(pool(variant))
        r_samples = np.asarray(r_samples, np.float64)

        def qr(cond):
            cond_b = np.tile(np.asarray(cond, np.float32)[None], (len(s0), 1))
            a0 = np.clip(np.asarray(_policy_call(
                agent, actor_mode, s0, cond_b, jax.random.PRNGKey(0))), -1, 1)
            zc = np.asarray(cond, np.float32) if fbz else None

            def bvf(idx):
                idx = np.asarray(idx, np.int64)
                return readout_m(s0, a0, nobs_all[idx], z_cond=zc).mean(0)

            return fm.q_r(bvf, idx_rel, r_samples,
                          cap_q=args.density_cap_q or None)

        return qr

    def deploy(z):
        """z_r -> actor conditioning (eval_zeroshot_rewards protocol)."""
        if actor_mode == "z":
            return np.asarray(z, np.float32), None
        j = z_to_goal_proxy_idx(Psi_cand, z)
        return S_cand[j], {"proxy_idx": int(j),
                           "proxy_cube_ou": cube_cand[j].tolist()}

    def region_candidate(center, eps=EPS_OBS):
        """Ball region -> candidate conditioning: z-mode = matchnorm mean psi
        over the k nearest in-ball dataset states (retrieval-style, stays on
        the psi(g) manifold); goal-mode = the nearest in-ball dataset obs."""
        idxs, ok = ball_candidate_indices(cube_cand, center, eps, k=args.retrieval_k)
        info = {"n_in_ball": int(len(idxs)) if ok else 0, "fallback_nearest": not ok,
                "nearest_cube_ou": cube_cand[idxs[0]].tolist()}
        if actor_mode == "z":
            zc = Psi_cand[idxs].mean(0)
            zc = (zc * match_data / (np.linalg.norm(zc) + 1e-8)).astype(np.float32)
            return zc, info
        return S_cand[idxs[0]], info

    def native_cond(i):
        """Oracle conditioning for the native goal family."""
        return (psi_g[i].astype(np.float32) if actor_mode == "z" else goal_obs[i])

    def roll(cond, variant):
        traj, obs_traj, ln = batched_rollout_record(agent, pool(variant), cond, H,
                                                    mode=actor_mode)
        return cube_m_to_ou(traj), traj, obs_traj, ln

    def roll_gpi(z_r, cand_conds, variant):
        traj, obs_traj, ln, sw, cc = batched_gpi_rollout_record(
            agent, q_at_z, pool(variant), z_r, cand_conds, H, mode=actor_mode)
        return cube_m_to_ou(traj), traj, obs_traj, ln, sw, cc

    results = {"meta": {
        "spec": SPEC, "tasks_json": str(Path(args.tasks_json).resolve()),
        "tasks_version": tasks_art["version"],
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step, "d": d,
        "actor_cond": cfg.get("actor_cond"), "actor_mode": actor_mode,
        "fb_forward_z": fbz, "measure_only": measure_only,
        "psi_src": "critic" if use_critic_psi else "value",
        "tiers": tiers, "episodes": args.episodes, "horizon": H,
        "z_inference": args.z_inference, "z_select": args.z_select,
        "gpi": bool(args.gpi),
        "oracle_z": bool(args.oracle_z),
        "relabel_size": int(len(S_rel)), "candidate_pool": int(len(S_cand)),
        "match_data_norm": match_data, "gamma": GAMMA, "eps_obs": EPS_OBS,
        "seed": args.seed,
        "achieved_reward_def": "sum_t gamma^t r(cube(s_{t+1})) over real steps, "
                               "obs units, from the recorded rollout track",
    }, "tiers": {}}

    def _mean(x):
        return float(np.mean(x)) if len(x) else None

    # ========================= t1: dirac + dense ==================================
    if "t1" in tiers:
        tier = {}
        for famname, rfn_of in (
                ("dirac", lambda i: (lambda c: reward_dirac(c, goal_ou[i]))),
                ("dense", lambda i: (lambda c: reward_dense(c, goal_ou[i])))):
            fam = {}
            for i in range(1, 6):
                rfn = rfn_of(i)
                rec = {}
                if args.oracle_z:
                    cond = native_cond(i)
                    rec["conditioning"] = "oracle"
                else:
                    z, diag = infer(np.asarray(rfn(cube_rel)))
                    rec["diag"] = diag
                    if z is None:
                        fam[f"t{i}"] = rec
                        continue
                    rec["cos_zr_psig"] = cosine(z, psi_g[i])
                    cond, proxy = deploy(z)
                    if proxy is not None:
                        rec["goal_proxy"] = proxy
                tou, traj, _ot, ln = roll(cond, i)
                rec["success"] = float(m_goal(traj, ln, goal_xyz[i]).mean())
                rec["G_achieved"] = _mean(per_episode_discounted(tou, ln, rfn))
                rec["dwell_goal_ball"] = _mean(discounted_time_in(
                    tou, ln, lambda c: in_ball(c, goal_ou[i])))
                fam[f"t{i}"] = rec
                print(f"[smb][t1/{famname}] t{i}: succ={rec['success']:.2f} "
                      f"G={rec['G_achieved']:.2f} ({time.time()-t00:.0f}s)", flush=True)
            succ = [v.get("success") for v in fam.values() if v.get("success") is not None]
            gs = [v.get("G_achieved") for v in fam.values() if v.get("G_achieved") is not None]
            fam["mean_success"] = _mean(succ)
            fam["mean_G"] = _mean(gs)
            tier[famname] = fam
        results["tiers"]["t1"] = tier

    # ========================= t2: mixture (+- GPI) ===============================
    if "t2" in tiers:
        fam = {}
        for i, j in combinations(range(1, 6), 2):
            rfn = lambda c, gi=goal_ou[i], gj=goal_ou[j]: reward_mixture(c, [gi, gj])
            rec = {"gpi": bool(args.gpi)}
            z = None
            if not args.oracle_z:
                z, diag = infer(np.asarray(rfn(cube_rel)))
                rec["diag"] = diag
                if z is None:
                    fam[f"t{i}-t{j}"] = rec
                    continue
            succs, Gs, per_var = [], [], {}
            for v in (i, j):
                if args.oracle_z:
                    # oracle = best single-goal candidate: both dedicated arms.
                    best = None
                    for w in (i, j):
                        tou, traj, _ot, ln = roll(native_cond(w), v)
                        s, _f = m_pair(traj, ln, goal_xyz[i], goal_xyz[j])
                        G = per_episode_discounted(tou, ln, rfn)
                        cand = {"success": float(s.mean()), "G": _mean(G)}
                        if best is None or cand["G"] > best["G"]:
                            best, s_best, G_best = cand, s, G
                    per_var[f"t{v}"] = best
                    succs.append(s_best)
                    Gs.append(G_best)
                elif args.gpi:
                    cond, _proxy = deploy(z)
                    cand = ([z, psi_g[i].astype(np.float32), psi_g[j].astype(np.float32)]
                            if actor_mode == "z" else [cond, goal_obs[i], goal_obs[j]])
                    tou, traj, _ot, ln, sw, cc = roll_gpi(z, cand, v)
                    s, _f = m_pair(traj, ln, goal_xyz[i], goal_xyz[j])
                    G = per_episode_discounted(tou, ln, rfn)
                    per_var[f"t{v}"] = {"success": float(s.mean()), "G": _mean(G),
                                        "switch_rate": float(sw.mean()),
                                        "choice_frac": (cc / max(cc.sum(), 1)).tolist()}
                    succs.append(s)
                    Gs.append(G)
                else:
                    cond, _proxy = deploy(z)
                    tou, traj, _ot, ln = roll(cond, v)
                    s, _f = m_pair(traj, ln, goal_xyz[i], goal_xyz[j])
                    G = per_episode_discounted(tou, ln, rfn)
                    per_var[f"t{v}"] = {"success": float(s.mean()), "G": _mean(G)}
                    succs.append(s)
                    Gs.append(G)
            rec["per_variant"] = per_var
            rec["success"] = float(np.concatenate(succs).mean())
            rec["G_achieved"] = float(np.concatenate(Gs).mean())
            fam[f"t{i}-t{j}"] = rec
            print(f"[smb][t2] t{i}-t{j}: succ={rec['success']:.2f} "
                  f"G={rec['G_achieved']:.2f} ({time.time()-t00:.0f}s)", flush=True)
        vals = [v.get("success") for v in fam.values() if v.get("success") is not None]
        gs = [v.get("G_achieved") for v in fam.values() if v.get("G_achieved") is not None]
        fam["mean_success"] = _mean(vals)
        fam["mean_G"] = _mean(gs)
        results["tiers"]["t2"] = fam

    # ========================= t3: occupancy-weighted choice ======================
    if "t3" in tiers:
        fam = {"note": "oracle = max over the two dedicated single-region arms; "
                       "chose-correct = larger unweighted discounted dwell in the "
                       "occupancy-correct region"}
        pairs_out = {}
        for p_i, pair in enumerate(tasks_art["t3_occhoice"]["pairs"]):
            gS = np.asarray(pair["stable"]["center"], np.float64)
            gT = np.asarray(pair["transient"]["center"], np.float64)
            wS, wT = float(pair["w_S"]), float(pair["w_T"])
            correct = pair["correct_region"]
            rfn = lambda c, gS=gS, gT=gT, wS=wS, wT=wT: reward_occhoice(c, gS, gT, wS, wT)
            variant = p_i % 5 + 1
            condS, infoS = region_candidate(gS)
            condT, infoT = region_candidate(gT)
            rec = {"variant": variant, "w_S": wS, "w_T": wT,
                   "correct_region": correct, "validation": pair["validation"],
                   "cand_stable": infoS, "cand_transient": infoT, "arms": {}}
            # dedicated single-region arms (both are part of the tier)
            ded = {}
            for name, cond in (("stable", condS), ("transient", condT)):
                tou, _traj, _ot, ln = roll(cond, variant)
                G = per_episode_discounted(tou, ln, rfn)
                lab, dS, dT = chosen_region(tou, ln, gS, gT)
                ded[name] = {"G": _mean(G), "dwell_S": _mean(dS), "dwell_T": _mean(dT)}
                rec["arms"][f"dedicated_{name}"] = ded[name]
            oracle = max(ded["stable"]["G"], ded["transient"]["G"])
            rec["oracle_G"] = oracle
            rec["oracle_arm"] = ("stable" if ded["stable"]["G"] >= ded["transient"]["G"]
                                 else "transient")
            # main arm
            main = {}
            if args.oracle_z:
                cond = condS if correct == "stable" else condT
                tou, _traj, _ot, ln = roll(cond, variant)
                main["conditioning"] = f"oracle_candidate_{correct}"
            else:
                z, diag = infer(np.asarray(rfn(cube_rel)))
                main["diag"] = diag
                if z is None:
                    rec["arms"]["main"] = main
                    pairs_out[pair["name"]] = rec
                    continue
                if args.z_select == "measure":
                    cond_zr, proxy = deploy(z)
                    if proxy is not None:
                        main["goal_proxy"] = proxy
                    sel = measure_select(
                        [("stable", condS), ("transient", condT), ("z_r", cond_zr)],
                        measure_qr_fn(rfn(cube_rel), variant))
                    main["measure_select"] = {"chosen": sel["chosen"],
                                              "scores": sel["scores"],
                                              "details": sel["details"]}
                    tou, _traj, _ot, ln = roll(sel["z"], variant)
                    main["conditioning"] = f"measure_select:{sel['chosen']}"
                elif args.gpi:
                    tou, _traj, _ot, ln, sw, cc = roll_gpi(z, [condS, condT], variant)
                    main["switch_rate"] = float(sw.mean())
                    main["choice_frac"] = (cc / max(cc.sum(), 1)).tolist()
                    main["conditioning"] = "gpi_over_region_candidates@z_r"
                else:
                    cond, proxy = deploy(z)
                    if proxy is not None:
                        main["goal_proxy"] = proxy
                    tou, _traj, _ot, ln = roll(cond, variant)
                    main["conditioning"] = "inferred_z"
            G = per_episode_discounted(tou, ln, rfn)
            lab, dS, dT = chosen_region(tou, ln, gS, gT)
            main.update({
                "G": _mean(G),
                "G_over_oracle": (float(np.mean(G) / oracle) if oracle and oracle > 0
                                  else None),
                "chose_correct_rate": float(np.mean([x == correct for x in lab])),
                "chosen_frac": {k: float(np.mean([x == k for x in lab]))
                                for k in ("stable", "transient", "none")},
                "dwell_S": _mean(dS), "dwell_T": _mean(dT),
            })
            rec["arms"]["main"] = main
            pairs_out[pair["name"]] = rec
            print(f"[smb][t3] {pair['name']}: G={main['G']:.3f} oracle={oracle:.3f} "
                  f"(ded S={ded['stable']['G']:.3f} T={ded['transient']['G']:.3f}) "
                  f"chose_correct={main['chose_correct_rate']:.2f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        fam["pairs"] = pairs_out
        mains = [p["arms"].get("main", {}) for p in pairs_out.values()]
        fam["mean_G"] = _mean([m["G"] for m in mains if m.get("G") is not None])
        fam["mean_G_over_oracle"] = _mean([m["G_over_oracle"] for m in mains
                                           if m.get("G_over_oracle") is not None])
        fam["mean_chose_correct"] = _mean([m["chose_correct_rate"] for m in mains
                                           if m.get("chose_correct_rate") is not None])
        results["tiers"]["t3"] = fam

    # ========================= t4: union3 =========================================
    if "t4" in tiers:
        fam = {}
        for k, task in enumerate(tasks_art["t4_union3"]["tasks"]):
            centers = [np.asarray(c, np.float64) for c in task["centers"]]
            rfn = lambda c, cs=centers: reward_union(c, cs)
            variant = k % 5 + 1
            rec = {"variant": variant}
            cands, cinfos = zip(*[region_candidate(c) for c in centers])
            rec["cands"] = list(cinfos)
            if args.oracle_z:
                best = None
                for ci, cond in enumerate(cands):
                    tou, _traj, _ot, ln = roll(cond, variant)
                    G = per_episode_discounted(tou, ln, rfn)
                    arm = {"G": _mean(G),
                           "success_end": _mean(final_in_any_ball(tou, ln, centers)),
                           "dwell_union": _mean(discounted_time_in(
                               tou, ln, lambda c: reward_union(c, centers) > 0))}
                    rec[f"dedicated_ball{ci}"] = arm
                    if best is None or arm["G"] > best["G"]:
                        best = arm
                rec["main"] = dict(best, conditioning="oracle_best_candidate")
            else:
                z, diag = infer(np.asarray(rfn(cube_rel)))
                rec["diag"] = diag
                if z is None:
                    fam[task["name"]] = rec
                    continue
                main = {}
                if args.z_select == "measure":
                    cond_zr, proxy = deploy(z)
                    if proxy is not None:
                        main["goal_proxy"] = proxy
                    mcands = [(f"ball{ci}", cands[ci]) for ci in range(len(cands))]
                    mcands.append(("z_r", cond_zr))
                    sel = measure_select(mcands,
                                         measure_qr_fn(rfn(cube_rel), variant))
                    main["measure_select"] = {"chosen": sel["chosen"],
                                              "scores": sel["scores"],
                                              "details": sel["details"]}
                    tou, _traj, _ot, ln = roll(sel["z"], variant)
                    main["conditioning"] = f"measure_select:{sel['chosen']}"
                elif args.gpi:
                    tou, _traj, _ot, ln, sw, cc = roll_gpi(z, list(cands), variant)
                    main["switch_rate"] = float(sw.mean())
                    main["choice_frac"] = (cc / max(cc.sum(), 1)).tolist()
                    main["conditioning"] = "gpi_over_ball_candidates@z_r"
                else:
                    cond, proxy = deploy(z)
                    if proxy is not None:
                        main["goal_proxy"] = proxy
                    tou, _traj, _ot, ln = roll(cond, variant)
                    main["conditioning"] = "inferred_z"
                main["G"] = _mean(per_episode_discounted(tou, ln, rfn))
                main["success_end"] = _mean(final_in_any_ball(tou, ln, centers))
                main["success_any"] = _mean([
                    float(reward_union(tou[e, :int(ln[e])], centers).any())
                    if int(ln[e]) else 0.0 for e in range(len(tou))])
                main["dwell_union"] = _mean(discounted_time_in(
                    tou, ln, lambda c: reward_union(c, centers) > 0))
                rec["main"] = main
            fam[task["name"]] = rec
            m = rec.get("main", {})
            print(f"[smb][t4] {task['name']}: G={m.get('G')} "
                  f"succ_end={m.get('success_end')} ({time.time()-t00:.0f}s)", flush=True)
        gs = [v["main"]["G"] for v in fam.values()
              if v.get("main", {}).get("G") is not None]
        se = [v["main"]["success_end"] for v in fam.values()
              if v.get("main", {}).get("success_end") is not None]
        fam["mean_G"] = _mean(gs)
        fam["mean_success_end"] = _mean(se)
        results["tiers"]["t4"] = fam

    # ========================= t5: corridor reach-avoid ===========================
    if "t5" in tiers:
        fam = {"note": "lambda=0 = the no-penalty control pricing the detour; "
                       "deflection = time_in_B(lam)/time_in_B(0)"}
        tasks_out = {}
        for task in tasks_art["t5_corridor"]["tasks"]:
            i = int(task["task"])
            gA = goal_ou[i]
            cB = np.asarray(task["avoid_center"], np.float64)
            rB = float(task["avoid_radius"])
            lams = [0] + [float(x) for x in task["lambdas"]]
            rec = {"avoid_center": cB.tolist(), "avoid_radius": rB, "arms": {}}
            t_in_b0 = None
            oracle_roll = None
            for lam in lams:
                rfn = lambda c, gA=gA, cB=cB, lam=lam, rB=rB: \
                    reward_corridor(c, gA, cB, lam, avoid_r=rB)
                arm = {"lambda": lam}
                if args.oracle_z:
                    # oracle conditioning ignores the penalty -> one rollout,
                    # re-scored per lambda.
                    if oracle_roll is None:
                        oracle_roll = roll(native_cond(i), i)
                    tou, traj, _ot, ln = oracle_roll
                    arm["conditioning"] = "oracle_psi_gA"
                else:
                    z, diag = infer(np.asarray(rfn(cube_rel)))
                    arm["diag"] = diag
                    if z is None:
                        rec["arms"][f"lam{lam:g}"] = arm
                        continue
                    cond, proxy = deploy(z)
                    if proxy is not None:
                        arm["goal_proxy"] = proxy
                    if args.z_select == "measure":
                        sel = measure_select(
                            [("goal_A", native_cond(i)), ("z_r", cond)],
                            measure_qr_fn(rfn(cube_rel), i))
                        arm["measure_select"] = {"chosen": sel["chosen"],
                                                 "scores": sel["scores"],
                                                 "details": sel["details"]}
                        cond = sel["z"]
                    tou, traj, _ot, ln = roll(cond, i)
                arm["success_at_A"] = float(m_goal(traj, ln, goal_xyz[i]).mean())
                arm["time_in_B"] = _mean(frac_time_in_ball(tou, ln, cB, rB))
                arm["G_signed"] = _mean(per_episode_discounted(tou, ln, rfn))
                arm["dwell_A"] = _mean(discounted_time_in(
                    tou, ln, lambda c: in_ball(c, gA)))
                if lam == 0:
                    t_in_b0 = arm["time_in_B"]
                elif t_in_b0 is not None and t_in_b0 > 0:
                    arm["deflection"] = float(arm["time_in_B"] / t_in_b0)
                rec["arms"][f"lam{lam:g}"] = arm
                print(f"[smb][t5] t{i} lam={lam:g}: A={arm['success_at_A']:.2f} "
                      f"tB={arm['time_in_B']:.4f} G={arm['G_signed']:.3f} "
                      f"({time.time()-t00:.0f}s)", flush=True)
            rec["corridor_binds"] = (t_in_b0 is not None and t_in_b0 >= 0.05)
            tasks_out[f"t{i}"] = rec
        fam["tasks"] = tasks_out
        for lam in (0, 1, 5):
            key = f"lam{lam:g}"
            arms = [t["arms"].get(key, {}) for t in tasks_out.values()]
            fam[f"mean_G_{key}"] = _mean([a["G_signed"] for a in arms
                                          if a.get("G_signed") is not None])
            fam[f"mean_timeB_{key}"] = _mean([a["time_in_B"] for a in arms
                                              if a.get("time_in_B") is not None])
        results["tiers"]["t5"] = fam

    # ========================= t6: feature-linear =================================
    if "t6" in tiers:
        fam = {"note": "controls: random-z (floor) + best-candidate (GPI over the "
                       "5 native conditionings scored at z_tilde) -- the second "
                       "separates 'learned z~' from 'any reasonable policy scores'"}
        rng_lin = np.random.default_rng(args.seed + 1)
        v0 = 1
        draws = []
        for k in range(args.n_linear):
            z_tilde = sample_sphere_z(rng_lin, d, match_data)
            rec = {"draw": k}

            def G_from_obs(obs_traj, ln):
                Gs = []
                for e in range(len(ln)):
                    L = int(ln[e])
                    if L == 0:
                        Gs.append(0.0)
                        continue
                    psi_traj = psi(obs_traj[e, :L]).astype(np.float64)
                    r_steps = reward_linear(psi_traj, z_tilde)
                    Gs.append(float((r_steps * GAMMA ** np.arange(L)).sum()))
                return float(np.mean(Gs))

            # main arm
            if args.oracle_z:
                cond, proxy = deploy(z_tilde.astype(np.float32))
                if proxy is not None:
                    rec["goal_proxy_main"] = proxy
                _tou, _traj, obs_traj, ln = roll(cond, v0)
                rec["conditioning"] = "oracle_z_tilde"
            else:
                r = reward_linear(Psi_rel, z_tilde)
                z, diag = infer(r)
                rec["diag"] = diag
                rec["cos_zr_ztilde"] = None if z is None else cosine(z, z_tilde)
                if z is None:
                    draws.append(rec)
                    continue
                cond, proxy = deploy(z)
                if proxy is not None:
                    rec["goal_proxy_main"] = proxy
                if args.z_select == "measure":
                    mcands = [(f"psi_g{ii}", native_cond(ii)) for ii in range(1, 6)]
                    mcands.append(("z_r", cond))
                    sel = measure_select(mcands, measure_qr_fn(r, v0))
                    rec["measure_select"] = {"chosen": sel["chosen"],
                                             "scores": sel["scores"],
                                             "details": sel["details"]}
                    cond = sel["z"]
                    rec["conditioning"] = f"measure_select:{sel['chosen']}"
                else:
                    rec["conditioning"] = "inferred_z"
                _tou, _traj, obs_traj, ln = roll(cond, v0)
            rec["G_main"] = G_from_obs(obs_traj, ln)
            # control 1: random z
            z_ctrl = sample_sphere_z(rng_lin, d, match_data).astype(np.float32)
            cond_c, _p = deploy(z_ctrl)
            _tou, _traj, obs_traj_c, ln_c = roll(cond_c, v0)
            rec["G_control_random_z"] = G_from_obs(obs_traj_c, ln_c)
            # control 2: best-candidate GPI over the 5 native conditionings,
            # critic readout at z_tilde (each method's own critic).
            cand = ([psi_g[i].astype(np.float32) for i in range(1, 6)]
                    if actor_mode == "z" else [goal_obs[i] for i in range(1, 6)])
            _tou, _traj, obs_traj_b, ln_b, sw, cc = roll_gpi(
                z_tilde.astype(np.float32), cand, v0)
            rec["G_control_best_candidate"] = G_from_obs(obs_traj_b, ln_b)
            rec["best_candidate_choice_frac"] = (cc / max(cc.sum(), 1)).tolist()
            draws.append(rec)
            print(f"[smb][t6] draw{k}: G={rec['G_main']:.3f} "
                  f"ctrl_rand={rec['G_control_random_z']:.3f} "
                  f"ctrl_best={rec['G_control_best_candidate']:.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        fam["draws"] = draws
        for key in ("G_main", "G_control_random_z", "G_control_best_candidate"):
            vals = [r[key] for r in draws if r.get(key) is not None]
            fam[f"mean_{key}"] = _mean(vals)
        results["tiers"]["t6"] = fam

    # ========================= out ================================================
    tag = ("oracle" if args.oracle_z else args.z_inference) + \
        ("_gpi" if args.gpi else "") + \
        ("_msel" if args.z_select == "measure" else "") + \
        (f"_cap{args.density_cap_q:g}" if args.density_cap_q else "")
    outp = Path(args.out) if args.out else \
        REPO / "analysis" / "rescuing-fb" / f"smb_wingA_{run_dir.name}_{tag}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(to_native(results), indent=2))
    print(f"\n[smb] DONE ({time.time()-t00:.0f}s) -> {outp}", flush=True)
    close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
