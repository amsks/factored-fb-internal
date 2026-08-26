#!/usr/bin/env python
"""scripts/rescuing_fb/eval_spec31_zsearch.py -- Spec 31: zero-order z-search
as the task-inference interface (search-first G29-3), EVAL-ONLY on the
centered-dual checkpoint.

Stages: gate (native re-roll ONLY -- the harness-pool encoder cell of g293 is
retired as a gate and the search path never encodes), families (per family:
encoder zhat vs encoder-seeded CEM z*, rollouts + affine cells), sweep
(pool-size robustness OF THE SEARCH), utility (KL distribution-matching
showcase), corr (J vs rollout return over round-0 candidates).

Run (tmux, one GPU, box shared):
  env CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_spec31_zsearch.py \\
    --run-dir ~/exp_spec30_p1/factored-fb/GCIQL-CubeSingle-*-RA2-CentDual-d50/ra2_cs_centdual \\
    --checkpoint-step 500000 --stage all --out-dir results/spec31_zsearch
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

from eval_spec30_g293 import (AVOID_R, G1_TASK, G2_TASK, GA_TASK, GATE_TOL,  # noqa: E402
                              GT_TASK, SUCC_R, anytime_ball, cos, disc_return,
                              fam_defs, kernel_raw, min_dist)
from z_search import (affine_argmax_check, cem_search, make_weights_fn,  # noqa: E402
                      sphere_project)

GATE_REF = 0.816
P_SUB = 2048                    # J integration pool (spec 31 §2)
SWEEP_M = (256, 1024, 8192)     # search-path pool robustness (spec 31 §2)
N_INIT_STATES = 16
CEM = dict(n_init=512, n_round=256, rounds=3, elite_frac=0.1)
AFFINE_TRANSFORMS = [(2.0, 0.0), (0.1, 0.0), (1.0, 1.0), (1.0, -5.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=16384)
    ap.add_argument("--stage", default="all",
                    choices=["gate", "families", "sweep", "utility", "corr", "all"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 4 eps, 2 families, CEM 64/32x1")
    ap.add_argument("--out-dir", default="results/spec31_zsearch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t00 = time.time()
    run_dir = Path(sorted(glob.glob(str(Path(args.run_dir).expanduser())))[0])
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "flags.json").read_text())
    env_name = cfg["env_name"]
    acfg = cfg["agent"]
    assert acfg.get("measure_density_z_encode") == "dual_centered", \
        "spec 31 runs on the centered-dual arm"
    n_eps = 4 if args.quick else args.episodes
    cem_kw = dict(n_init=64, n_round=32, rounds=1) if args.quick else dict(CEM)

    from eval_z_action_ranking import hit_spec_for, load_dataset  # noqa: E402
    from eval_density_zeroshot import build_pool  # noqa: E402
    from eval_zeroshot_rewards import batched_rollout_record, eval_tasks_for  # noqa: E402
    from t3_offpeak_multienv import load_agent_dims  # noqa: E402

    spec = hit_spec_for(env_name, acfg.get("measure_hit_spec"))
    blocks = spec.blocks
    (b0, b1, _e) = blocks[0]
    xyz = slice(b0, b1)

    ds = load_dataset(env_name)
    obs_all = np.asarray(ds["observations"], np.float32)
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

    d = int(acfg["measure_density_dim"])
    temp = float(acfg.get("measure_density_temp", 1.0))
    rng = np.random.default_rng(args.seed)

    # big pool (audited builder, g293 rng order)
    pool_idx = rng.choice(len(obs_all), size=min(args.pool_size, len(obs_all)),
                          replace=False)
    pool_obs = obs_all[pool_idx]
    u_raw, _b_online = build_pool(agent, blocks, pool_obs, spec)
    u_sc = np.asarray(GCIQLFlowAgent._density_queries(jnp.asarray(pool_obs), blocks))
    P = len(u_raw)

    # ckpt's OWN frozen eval pool (the ONLY encoding pool -- G29-3 diagnosis)
    pool_raw_frozen, _filled = (np.asarray(x) for x in
                                agent.network.select('density_zpool')())
    assert float(np.asarray(_filled)) == 1.0, "frozen eval pool never filled"
    u_sc_frozen = np.asarray(GCIQLFlowAgent._scale_block_cols(
        jnp.asarray(pool_raw_frozen), blocks))

    def encode_frozen(r_vals):
        z = np.asarray(agent.density_z_from_reward_samples(
            jnp.asarray(u_sc_frozen),
            jnp.asarray(np.atleast_2d(r_vals), jnp.float32)))
        return z[0]

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
    _e0 = ogbench.make_env_and_datasets(eval_tasks[1], env_only=True)
    H = _e0.spec.max_episode_steps if _e0.spec and _e0.spec.max_episode_steps else 200
    _e0.close()

    def rollout(task_id, cond, mode, n=None):
        envs = make_envs(task_id, n or n_eps)
        traj, obs_traj, lengths, succ_env = batched_rollout_record(
            agent, envs, cond, H=H, mode=mode)
        for e in envs:
            e.close()
        cube = obs_traj[:, :, xyz].astype(np.float64)
        for i in range(len(cube)):
            L = max(int(lengths[i]), 1)
            cube[i, L:] = cube[i, L - 1]
        return cube, lengths, succ_env

    # J evaluation state set: env-reset states (reset only, no stepping)
    def reset_states(task_id, n=N_INIT_STATES):
        e = ogbench.make_env_and_datasets(eval_tasks[task_id], env_only=True)
        out = []
        for k in range(n):
            o, _ = e.reset(seed=1000 + k)
            out.append(o)
        e.close()
        return np.asarray(out, np.float32)

    # J integration pool: P_SUB subsample of the dataset marginal
    sub = rng.choice(P, P_SUB, replace=False)
    b_sub = np.asarray(agent.network.select("density_b")(jnp.asarray(u_sc[sub])))
    print(f"[s31] {run_dir.name} step={args.checkpoint_step} "
          f"P={P} d={d} temp={temp} ({time.time()-t00:.0f}s)", flush=True)

    # =================================================================== gate
    if args.stage in ("gate", "all"):
        nat, harness = {}, {}
        for t in range(1, 6):
            cube, lengths, succ_env = rollout(t, goals[t], mode="goal")
            nat[t] = {"succ_env": float(succ_env.mean())}
            kg_f = np.asarray(GCIQLFlowAgent._delta_phi(
                jnp.asarray(goals[t][None]), jnp.asarray(pool_raw_frozen), blocks))[0]
            z_f = encode_frozen(kg_f)
            cube, lengths, succ_env = rollout(t, z_f, mode="z")
            harness[t] = {"succ_env": float(succ_env.mean())}
        oa = float(np.mean([nat[t]["succ_env"] for t in nat]))
        ob = float(np.mean([harness[t]["succ_env"] for t in harness]))
        gate = {"native": nat, "overall_native": oa,
                "frozen_pool_z_api": harness, "overall_z_api": ob,
                "reference": GATE_REF, "tol": GATE_TOL,
                "pass": bool(abs(oa - GATE_REF) <= GATE_TOL
                             and abs(ob - GATE_REF) <= GATE_TOL)}
        (out_dir / "gate.json").write_text(json.dumps(gate, indent=1))
        print(f"[s31] GATE native={oa:.3f} z_api={ob:.3f} ref={GATE_REF} "
              f"pass={gate['pass']} ({time.time()-t00:.0f}s)", flush=True)
        if not gate["pass"] and not args.quick:
            print("[s31] GATE FAILED -- stopping, no new cells read", flush=True)
            return
    if args.stage == "gate":
        return

    # =============================================================== families
    g1 = goals[G1_TASK][xyz]; g2 = goals[G2_TASK][xyz]
    gt = goals[GT_TASK][xyz]; ga = goals[GA_TASK][xyz]
    med_da = float(np.median(np.linalg.norm(u_raw - ga[None], axis=-1)))
    fams = fam_defs(g1, g2, gt, ga, med_da)
    if args.quick:
        fams = {k: fams[k] for k in ("dense_g1", "tma10")}

    summary = {}
    if args.stage in ("families", "all"):
        for name, fd in fams.items():
            r_sub = fd["r"](u_raw[sub])
            init_obs = reset_states(fd["task"])
            weights_of = make_weights_fn(agent, b_sub, temp, init_obs,
                                         seed=jax.random.PRNGKey(11))
            z_enc = encode_frozen(fd["r"](pool_raw_frozen))
            z_star, diag = cem_search(lambda W: W @ r_sub, weights_of, d,
                                      np.random.default_rng(args.seed + 100),
                                      seed_z=z_enc, **cem_kw)
            aff = affine_argmax_check(diag["W0"], r_sub, AFFINE_TRANSFORMS)
            # spec 31 §3: renorm invariance is a THEOREM-level property --
            # a violation means a harness bug, hard-abort (house gate rule);
            # the min-convention result is reported, never asserted.
            assert aff["renorm_all_match"], \
                f"{name}: renormalized affine argmax invariance violated -- " \
                f"harness bug, not a finding: {aff['renorm']}"

            cube, lengths, _ = rollout(fd["task"], z_enc, mode="z")
            roll_enc = fd["succ"](cube, lengths)
            roll_enc["disc_return"] = disc_return(cube, lengths, fd["r"])
            cube, lengths, _ = rollout(fd["task"], z_star, mode="z")
            roll_search = fd["succ"](cube, lengths)
            roll_search["disc_return"] = disc_return(cube, lengths, fd["r"])

            rec = {"rollout_encoder": roll_enc, "rollout_search": roll_search,
                   "J_seed": diag["J_seed"], "J_star": diag["J_star"],
                   "cos_seed_star": diag["cos_seed_star"],
                   "round_max": diag["round_max"],
                   "round_median": diag["round_median"],
                   "affine": aff,
                   "n_evaluated": diag["n_evaluated"],
                   "z_enc": z_enc.tolist(), "z_star": z_star.tolist(),
                   "episodes": n_eps, "task_reset": fd["task"]}
            (out_dir / f"fam_{name}.json").write_text(
                json.dumps(rec, indent=1, default=str))
            summary[name] = {**rec,
                             "_W0": diag["W0"], "_J0": diag["J0"],
                             "_Z0": diag["Z0"], "_r_sub": r_sub,
                             "_init_obs": init_obs}
            print(f"[s31] {name}: enc_succ={roll_enc['succ']:.2f} "
                  f"search_succ={roll_search['succ']:.2f} "
                  f"Jseed={diag['J_seed']:.4f} J*={diag['J_star']:.4f} "
                  f"cos={diag['cos_seed_star']:.3f} "
                  f"aff_renorm={aff['renorm_all_match']} "
                  f"aff_min={aff['min_all_match']} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        pub = {n: {k: v for k, v in summary[n].items() if not k.startswith("_")}
               for n in summary}
        (out_dir / "families_summary.json").write_text(
            json.dumps(pub, indent=1, default=str))

    # ================================================================== sweep
    if args.stage in ("sweep", "all") and not args.quick:
        sweep_fams = ("dense_g1", "tma10")
        sweep = {}
        for name in sweep_fams:
            fd = fams[name]
            init_obs = (summary[name]["_init_obs"] if name in summary
                        else reset_states(fd["task"]))
            z_enc = encode_frozen(fd["r"](pool_raw_frozen))
            cells = {}
            zstars = {}
            for m in SWEEP_M:
                idx_m = rng.choice(P, m, replace=False)
                b_m = np.asarray(agent.network.select("density_b")(
                    jnp.asarray(u_sc[idx_m])))
                r_m = fd["r"](u_raw[idx_m])
                weights_of = make_weights_fn(agent, b_m, temp, init_obs,
                                             seed=jax.random.PRNGKey(11))
                z_star, diag = cem_search(lambda W: W @ r_m, weights_of, d,
                                          np.random.default_rng(args.seed + 200),
                                          seed_z=z_enc, **cem_kw)
                zstars[m] = z_star
                cube, lengths, _ = rollout(fd["task"], z_star, mode="z")
                rr = fd["succ"](cube, lengths)
                cells[str(m)] = {"J_star": diag["J_star"], "succ": rr["succ"],
                                 "cos_vs_enc": cos(z_star, z_enc)}
            stab = {f"cos_{SWEEP_M[i]}_{SWEEP_M[i+1]}":
                    cos(zstars[SWEEP_M[i]], zstars[SWEEP_M[i + 1]])
                    for i in range(len(SWEEP_M) - 1)}
            sweep[name] = {"cells": cells, "stability": stab}
            print(f"[s31] sweep {name}: " + " ".join(
                f"M{m}:succ={cells[str(m)]['succ']:.2f}" for m in SWEEP_M)
                + f" stab={stab} ({time.time()-t00:.0f}s)", flush=True)

        # init-state ablation (spec 31 §2): dataset episode-start states
        # instead of env-reset states, one family, expected no-op
        fd = fams["dense_g1"]
        term = np.asarray(ds["terminals"]).astype(bool)
        starts = np.concatenate([[0], np.where(term[:-1])[0] + 1])
        ds_init = obs_all[rng.choice(starts, N_INIT_STATES, replace=False)]
        z_enc = encode_frozen(fd["r"](pool_raw_frozen))
        r_sub_d = fd["r"](u_raw[sub])
        weights_of = make_weights_fn(agent, b_sub, temp, ds_init,
                                     seed=jax.random.PRNGKey(11))
        z_star_ds, diag_ds = cem_search(lambda W: W @ r_sub_d, weights_of, d,
                                        np.random.default_rng(args.seed + 100),
                                        seed_z=z_enc, **cem_kw)
        z_star_env = (np.asarray(summary["dense_g1"]["z_star"])
                      if "dense_g1" in summary else None)
        cube, lengths, _ = rollout(fd["task"], z_star_ds, mode="z")
        sweep["init_state_ablation"] = {
            "family": "dense_g1", "J_star_dataset_s0": diag_ds["J_star"],
            "succ_dataset_s0": fd["succ"](cube, lengths)["succ"],
            "cos_vs_env_reset_zstar": (cos(z_star_ds, z_star_env)
                                       if z_star_env is not None else None)}
        print(f"[s31] init ablation: succ={sweep['init_state_ablation']['succ_dataset_s0']:.2f} "
              f"cos={sweep['init_state_ablation']['cos_vs_env_reset_zstar']}", flush=True)
        (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=1))

    # ================================================================ utility
    if args.stage in ("utility", "all") and not args.quick:
        BINS = 4
        half_w = 0.6
        lo = g1 - half_w
        edges = [np.linspace(lo[i], lo[i] + 2 * half_w, BINS + 1) for i in range(3)]

        def bin_index(u):
            """u [N,3] -> flat bin id in [0, BINS^3] (last id = outside)."""
            ids = np.zeros(len(u), np.int64)
            outside = np.zeros(len(u), bool)
            mul = 1
            for i in range(3):
                k = np.digitize(u[:, i], edges[i]) - 1
                outside |= (k < 0) | (k >= BINS)
                ids += np.clip(k, 0, BINS - 1) * mul
                mul *= BINS
            ids[outside] = BINS ** 3
            return ids

        pool_bins = bin_index(u_raw[sub])
        n_bins = BINS ** 3 + 1
        target = np.full(n_bins, 1e-3)
        target[:BINS ** 3] = 1.0
        target /= target.sum()

        def f_kl(W):
            """W [K, M] -> J [K] = -KL(Pi(w) || target), rows renormalized."""
            Wn = W / W.sum(-1, keepdims=True)
            proj = np.zeros((len(W), n_bins))
            np.add.at(proj.T, pool_bins, Wn.T)          # sum w over bins
            proj = np.clip(proj, 1e-9, None)
            proj /= proj.sum(-1, keepdims=True)
            return -np.sum(proj * np.log(proj / target[None]), -1)

        init_obs = reset_states(G1_TASK)
        weights_of = make_weights_fn(agent, b_sub, temp, init_obs,
                                     seed=jax.random.PRNGKey(11))
        z_star, diag = cem_search(f_kl, weights_of, d,
                                  np.random.default_rng(args.seed + 300),
                                  seed_z=None, **cem_kw)
        z_goal = encode_frozen(fams["dense_g1"]["r"](pool_raw_frozen))

        def occupancy_report(z, tag):
            cube, lengths, _ = rollout(G1_TASK, z, mode="z")
            ids = bin_index(cube.reshape(-1, 3))
            occ = np.bincount(ids, minlength=n_bins).astype(float)
            occ /= occ.sum()
            occ_c = np.clip(occ, 1e-9, None)
            return {"kl_vs_target": float(np.sum(occ_c * np.log(occ_c / target))),
                    "frac_region_bins_visited": float(
                        (occ[:BINS ** 3] > 0).mean()),
                    "frac_time_in_region": float(occ[:BINS ** 3].sum())}

        util = {"J_star": diag["J_star"],
                "J_median_round0": float(np.median(diag["J0"])),
                "round_max": diag["round_max"],
                "search": occupancy_report(z_star, "search"),
                "goal_contrast": occupancy_report(z_goal, "goal"),
                "z_star": z_star.tolist(), "bins": BINS, "half_w": half_w}
        (out_dir / "utility.json").write_text(json.dumps(util, indent=1))
        print(f"[s31] utility: J*={diag['J_star']:.4f} "
              f"search_cov={util['search']['frac_region_bins_visited']:.2f} "
              f"goal_cov={util['goal_contrast']['frac_region_bins_visited']:.2f} "
              f"({time.time()-t00:.0f}s)", flush=True)

    # =================================================================== corr
    if args.stage in ("corr", "all") and not args.quick:
        from scipy.stats import spearmanr
        assert summary, "corr stage needs families in the same invocation " \
                        "(use --stage all or families,corr)"
        corr_fams = [n for n in ("dense_g1", "tma10") if n in summary]
        corr = {}
        for name in corr_fams:
            fd = fams[name]
            J0 = summary[name]["_J0"]; Z0 = summary[name]["_Z0"]
            qs = np.unique(np.quantile(np.arange(len(J0)), np.linspace(0, 1, 12))
                           .astype(int))
            order = np.argsort(J0)
            pick = order[qs]
            rets, js = [], []
            for k in pick:
                cube, lengths, _ = rollout(fd["task"], Z0[k], mode="z", n=5)
                rets.append(disc_return(cube, lengths, fd["r"]))
                js.append(float(J0[k]))
            corr[name] = {"spearman": float(spearmanr(js, rets).statistic),
                          "J": js, "returns": rets, "eps_per_candidate": 5}
            print(f"[s31] corr {name}: spearman={corr[name]['spearman']:.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        (out_dir / "corr.json").write_text(json.dumps(corr, indent=1))

    # top-level summary (whatever stages ran)
    top = {"stages_run": args.stage, "quick": bool(args.quick),
           "cem": cem_kw, "p_sub": P_SUB, "n_init_states": N_INIT_STATES,
           "ckpt": str(run_dir), "step": args.checkpoint_step}
    for f in ("gate", "families_summary", "sweep", "utility", "corr"):
        p = out_dir / f"{f}.json"
        if p.exists():
            top[f] = json.loads(p.read_text())
    (out_dir / "summary.json").write_text(json.dumps(top, indent=1, default=str))
    print(f"[s31] DONE ({time.time()-t00:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
