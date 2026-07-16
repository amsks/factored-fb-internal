#!/usr/bin/env python
"""scripts/rescuing_fb/t6_generality_probe.py — T6: zero-shot generality of the
reward->z->policy API (pre-registered: PAPER/rescuing-fb/05-t6-generality-spec.md).

The G4 z-cond agent trained z only at psi(g) for single dataset goals. Does moment
inference z = E_rho[r.psi(s')] produce sensible policies for rewards OUTSIDE the
single-goal family? Families (spec section 2): (a) two-goal mixtures (10 pairs),
(b) lift-anywhere, (c) region-goals (3 non-goal boxes), (d) uniform-reward negative
control, (e) single-goal positive anchors (harness gate: must reproduce E1a 0.955
under match_g, else STOP).

All rewards are computed from the decoded cube xyz of ONE fixed 50k dataset
subsample (rng(0)); z_raw = sum_i r_i psi(s'_i); norms: match_data (primary,
goal-free), sqrtd (secondary), match_g (anchors only, E1a-faithful gate).
Rollouts are batched: 25 envs per (reward, task-variant, norm) stepped in
lockstep, one jitted sample_actions_with_z call per step; metrics computed from
recorded cube trajectories (real, pre-termination steps only).

Run (CPU-only, capped to cores 0-47 per user directive — box is shared):
  taskset -c 0-47 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
    JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/t6_generality_probe.py --stage gate   # then --stage full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))

EVAL_TASKS = {i: f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)}
CUBE_QPOS = slice(14, 17)          # privileged cube xyz, meters (E1a convention)
CUBE_OBS = slice(19, 22)           # obs cube xyz = (m - CEN) * 10
CEN = np.array([0.425, 0.0, 0.0])
THRESH = 0.04                      # goal-reach radius, meters
Z_LIFT = 0.131                     # table p95 (0.031) + 0.10 m  (spec section 0)
LIFT_HOLD = 10                     # consecutive steps above Z_LIFT
BOX_HALF = 0.03                    # 6 cm box
REGIONS = {                        # spec section 2c (pre-registered selection)
    "R1": np.array([0.3515, 0.1720, 0.02]),
    "R2": np.array([0.4251, -0.1708, 0.02]),
    "R3": np.array([0.4980, 0.0848, 0.02]),
}
DEFAULT_CKPT = Path("~/bridge_ckpt_backup/gciql_outputs/factored-fb/gciql_g4_zcond/"
                    "sd000_20260616_193331").expanduser()


def decode_cube(obs: np.ndarray) -> np.ndarray:
    """Dataset/env obs -> cube xyz in meters (spec section 0; 1 obs unit = 10 cm)."""
    return np.asarray(obs)[..., CUBE_OBS] / 10.0 + CEN


def make_psi(agent):
    import jax.numpy as jnp

    def psi(obs, chunk=8192):
        obs = np.asarray(obs, np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None]
        out = []
        for i in range(0, len(obs), chunk):
            g = jnp.asarray(obs[i:i + chunk])
            _v, _phi, ps = agent.network.select("value")(g, g, info=True)
            out.append(np.asarray(ps))
        out = np.concatenate(out, 0)
        return out[0] if single else out

    return psi


def rescale(z, tgt):
    return (z * tgt / (np.linalg.norm(z) + 1e-8)).astype(np.float32)


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def batched_rollout(agent, envs, z, H=200):
    """Roll the z-actor in lockstep across len(envs) envs from reset seeds 0..n-1.
    Returns (cube_traj [n,H,3] meters, lengths [n]) — steps beyond an env's
    termination are padded with its last cube position; metrics must use lengths."""
    import jax
    n = len(envs)
    zb = np.tile(np.asarray(z, np.float32)[None], (n, 1))
    obs = np.zeros((n, envs[0].observation_space.shape[0]), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    lengths = np.zeros(n, int)
    rng = jax.random.PRNGKey(0)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        acts = np.asarray(agent.sample_actions_with_z(
            observations=obs, z=zb, seed=sub, temperature=0.0))
        acts = np.clip(acts, -1, 1)
        for i, e in enumerate(envs):
            if not active[i]:
                traj[i, t] = traj[i, t - 1]
                continue
            o, _r, term, trunc, _ = e.step(acts[i])
            obs[i] = o
            traj[i, t] = e.unwrapped.data.qpos[CUBE_QPOS]
            lengths[i] = t + 1
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    return traj, lengths


# ---------------- metrics (real steps only) ----------------

def m_goal(traj, lengths, goal_xyz, thresh=THRESH):
    """Per-episode: any real step with cube within thresh of goal."""
    return np.array([(np.linalg.norm(traj[i, :lengths[i]] - goal_xyz, axis=-1)
                      < thresh).any() for i in range(len(traj))], float)


def m_pair(traj, lengths, g1, g2):
    """(success either, first-hit id per episode: 0=none/1/2)."""
    succ, first = [], []
    for i in range(len(traj)):
        d1 = np.linalg.norm(traj[i, :lengths[i]] - g1, axis=-1) < THRESH
        d2 = np.linalg.norm(traj[i, :lengths[i]] - g2, axis=-1) < THRESH
        succ.append(float(d1.any() or d2.any()))
        t1 = int(np.argmax(d1)) if d1.any() else 10 ** 9
        t2 = int(np.argmax(d2)) if d2.any() else 10 ** 9
        first.append(0 if not (d1.any() or d2.any()) else (1 if t1 <= t2 else 2))
    return np.array(succ), np.array(first)


def m_lift(traj, lengths):
    """Per-episode: >= LIFT_HOLD consecutive real steps with cube_z > Z_LIFT."""
    out = []
    for i in range(len(traj)):
        above = traj[i, :lengths[i], 2] > Z_LIFT
        run = best = 0
        for a in above:
            run = run + 1 if a else 0
            best = max(best, run)
        out.append(float(best >= LIFT_HOLD))
    return np.array(out)


def m_region(traj, lengths, center):
    """(final-step success [primary], any-step success [secondary])."""
    fin, anys = [], []
    for i in range(len(traj)):
        seg = traj[i, :lengths[i]]
        inside = (np.abs(seg - center) <= BOX_HALF).all(-1) | \
                 (np.linalg.norm(seg - center, axis=-1) < THRESH)
        fin.append(float(inside[-1]))
        anys.append(float(inside.any()))
    return np.array(fin), np.array(anys)


def choice_entropy(firsts):
    hits = firsts[firsts > 0]
    if len(hits) == 0:
        return 0.0
    p = float((hits == 1).mean())
    if p in (0.0, 1.0):
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", default=str(DEFAULT_CKPT))
    ap.add_argument("--checkpoint-step", type=int, default=1000000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--stage", choices=["gate", "full"], default="full",
                    help="gate = anchors/match_g only (must hit >=0.90 before full)")
    ap.add_argument("--out", default=str(REPO / "analysis/rescuing-fb/t6_generality_g4.json"))
    args = ap.parse_args()

    import ogbench
    from phase_probe_gciql import load_agent

    t00 = time.time()
    agent = load_agent(Path(args.checkpoint_dir), args.checkpoint_step, seed=0,
                       value_arch="bilinear", value_latent_dim=args.d, actor_cond="z")
    psi = make_psi(agent)
    sqrtd = float(np.sqrt(args.d))
    print(f"[t6] agent restored d={args.d} step={args.checkpoint_step}", flush=True)

    # ---- shared relabel sample (task1 dataset; obs shared across variants) ----
    _e0, ds1, _ = ogbench.make_env_and_datasets(EVAL_TASKS[1])
    n_ds = len(ds1["observations"])
    idx = np.random.default_rng(0).choice(n_ds, size=min(args.relabel_size, n_ds),
                                          replace=False)
    S = np.asarray(ds1["observations"], np.float32)[idx]
    cube_ds = decode_cube(S)                                   # [m,3] meters
    Psi = psi(S).astype(np.float64)                            # [m,d]
    match_data = float(np.linalg.norm(Psi, axis=-1).mean())
    ds_rewards = {1: (np.asarray(ds1["rewards"], np.float64) + 1.0)[idx]}
    for i in range(2, 6):
        _e, dsi, _ = ogbench.make_env_and_datasets(EVAL_TASKS[i])
        ds_rewards[i] = (np.asarray(dsi["rewards"], np.float64) + 1.0)[idx]
        del dsi

    # ---- per-task goal obs / xyz + env pools ----
    goal_obs, goal_xyz = {}, {}
    for i, task in EVAL_TASKS.items():
        env = ogbench.make_env_and_datasets(task, env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        goal_xyz[i] = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        env.close()
    psi_g = {i: psi(goal_obs[i]) for i in EVAL_TASKS}

    def diag(z_raw, n_pos):
        cos_g = {f"t{i}": cosine(z_raw, psi_g[i]) for i in EVAL_TASKS}
        near = max(cos_g, key=cos_g.get)
        return {"n_pos": int(n_pos), "z_raw_norm": float(np.linalg.norm(z_raw)),
                "cos_to_psi_g": cos_g, "nearest_psi_g": near,
                "cos_nearest": cos_g[near],
                "cos_to_mean_psi": cosine(z_raw, Psi.mean(0))}

    def infer(r):
        r = np.asarray(r, np.float64)
        return (r[:, None] * Psi).sum(0), int(r.sum())

    # synthetic single-goal indicators + agreement with dataset relabel
    syn_r = {i: (np.linalg.norm(cube_ds - goal_xyz[i], axis=-1) < THRESH).astype(float)
             for i in EVAL_TASKS}
    agreement = {f"t{i}": float((syn_r[i] == ds_rewards[i]).mean()) for i in EVAL_TASKS}
    print(f"[t6] relabel sample m={len(S)}  match_data_norm={match_data:.2f} "
          f"sqrtd={sqrtd:.2f}  ds-vs-synthetic reward agreement={agreement}", flush=True)

    # ---- env pools, built lazily per variant ----
    pools = {}

    def pool(i):
        if i not in pools:
            t0 = time.time()
            pools[i] = [ogbench.make_env_and_datasets(EVAL_TASKS[i], env_only=True)
                        for _ in range(args.episodes)]
            print(f"[t6] pool task{i}: {args.episodes} envs in {time.time()-t0:.0f}s",
                  flush=True)
        return pools[i]

    def close_pool(i):
        for e in pools.pop(i, []):
            if hasattr(e, "close"):
                e.close()

    H = args.horizon
    results = {"meta": {
        "checkpoint_dir": str(args.checkpoint_dir), "checkpoint_step": args.checkpoint_step,
        "d": args.d, "episodes": args.episodes, "horizon": H,
        "relabel_size": len(S), "match_data_norm": match_data, "sqrt_d": sqrtd,
        "z_lift_m": Z_LIFT, "lift_hold_steps": LIFT_HOLD, "box_half_m": BOX_HALF,
        "regions_m": {k: v.tolist() for k, v in REGIONS.items()},
        "goal_xyzs_m": {f"t{i}": goal_xyz[i].tolist() for i in EVAL_TASKS},
        "reward_agreement_ds_vs_synthetic": agreement,
        "stage": args.stage,
        "spec": "PAPER/rescuing-fb/05-t6-generality-spec.md",
    }}

    # ================= (e) anchors + GATE =================
    anchors = {}
    for i in EVAL_TASKS:
        z_raw, n_pos = infer(ds_rewards[i])
        rec = {"dataset_r": diag(z_raw, n_pos), "success": {}}
        traj, ln = batched_rollout(agent, pool(i), rescale(z_raw, np.linalg.norm(psi_g[i])), H)
        rec["success"]["match_g"] = float(m_goal(traj, ln, goal_xyz[i]).mean())
        if args.stage == "full":
            zs_raw, ns = infer(syn_r[i])
            rec["synthetic_r"] = diag(zs_raw, ns)
            for norm, tgt in (("match_data", match_data), ("sqrtd", sqrtd)):
                traj, ln = batched_rollout(agent, pool(i), rescale(zs_raw, tgt), H)
                rec["success"][norm] = float(m_goal(traj, ln, goal_xyz[i]).mean())
        anchors[f"t{i}"] = rec
        print(f"[t6][e] t{i}: {rec['success']}  n_pos={n_pos} "
              f"cos_g={rec['dataset_r']['cos_to_psi_g'][f't{i}']:.3f} "
              f"({time.time()-t00:.0f}s)", flush=True)
    results["anchors"] = anchors
    gate_mean = float(np.mean([anchors[f"t{i}"]["success"]["match_g"] for i in EVAL_TASKS]))
    results["gate"] = {"mean_match_g": gate_mean, "threshold": 0.90,
                       "passed": bool(gate_mean >= 0.90)}
    print(f"[t6] GATE mean match_g = {gate_mean:.3f} -> "
          f"{'PASS' if gate_mean >= 0.90 else 'FAIL (STOP)'}", flush=True)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    if args.stage == "gate" or not results["gate"]["passed"]:
        for i in list(pools):
            close_pool(i)
        return 0 if results["gate"]["passed"] else 1

    norms = (("match_data", match_data), ("sqrtd", sqrtd))

    # ================= (a) two-goal mixtures =================
    pairs = {}
    for i, j in combinations(range(1, 6), 2):
        r = np.maximum(syn_r[i], syn_r[j])
        z_raw, n_pos = infer(r)
        rec = {"diag": diag(z_raw, n_pos), "per_norm": {}}
        for norm, tgt in norms:
            z = rescale(z_raw, tgt)
            succs, firsts = [], []
            per_var = {}
            for v in (i, j):
                traj, ln = batched_rollout(agent, pool(v), z, H)
                s, f = m_pair(traj, ln, goal_xyz[i], goal_xyz[j])
                per_var[f"t{v}"] = {"success": float(s.mean()),
                                    "hit_g1": int((f == 1).sum()),
                                    "hit_g2": int((f == 2).sum())}
                succs.append(s)
                firsts.append(f)
            firsts = np.concatenate(firsts)
            rec["per_norm"][norm] = {
                "success_mean": float(np.concatenate(succs).mean()),
                "per_variant": per_var,
                "choice_entropy_bits": choice_entropy(firsts),
                "frac_first_g1": float((firsts == 1).sum() / max((firsts > 0).sum(), 1)),
            }
        pairs[f"t{i}-t{j}"] = rec
        print(f"[t6][a] t{i}-t{j}: "
              + " ".join(f"{n}={rec['per_norm'][n]['success_mean']:.3f}"
                         f"(H={rec['per_norm'][n]['choice_entropy_bits']:.2f})"
                         for n, _ in norms)
              + f"  n_pos={n_pos} ({time.time()-t00:.0f}s)", flush=True)
    results["pairs"] = pairs

    # ================= (b) lift-anywhere =================
    r = (cube_ds[:, 2] > Z_LIFT).astype(float)
    z_raw, n_pos = infer(r)
    lift = {"diag": diag(z_raw, n_pos), "per_norm": {}}
    for norm, tgt in norms:
        z = rescale(z_raw, tgt)
        per_var = {}
        for v in EVAL_TASKS:
            traj, ln = batched_rollout(agent, pool(v), z, H)
            per_var[f"t{v}"] = float(m_lift(traj, ln).mean())
        lift["per_norm"][norm] = {"success_mean": float(np.mean(list(per_var.values()))),
                                  "per_variant": per_var}
        print(f"[t6][b] lift {norm}: {lift['per_norm'][norm]['success_mean']:.3f} "
              f"{per_var} n_pos={n_pos} ({time.time()-t00:.0f}s)", flush=True)
    results["lift"] = lift

    # ================= (c) region goals =================
    regions = {}
    for name, c in REGIONS.items():
        r = ((np.abs(cube_ds - c) <= BOX_HALF).all(-1)
             | (np.linalg.norm(cube_ds - c, axis=-1) < THRESH)).astype(float)
        z_raw, n_pos = infer(r)
        rec = {"center_m": c.tolist(), "diag": diag(z_raw, n_pos), "per_norm": {}}
        for norm, tgt in norms:
            z = rescale(z_raw, tgt)
            per_var_f, per_var_a = {}, {}
            for v in EVAL_TASKS:
                traj, ln = batched_rollout(agent, pool(v), z, H)
                fin, anys = m_region(traj, ln, c)
                per_var_f[f"t{v}"] = float(fin.mean())
                per_var_a[f"t{v}"] = float(anys.mean())
            rec["per_norm"][norm] = {
                "success_final_mean": float(np.mean(list(per_var_f.values()))),
                "success_any_mean": float(np.mean(list(per_var_a.values()))),
                "per_variant_final": per_var_f, "per_variant_any": per_var_a}
        regions[name] = rec
        print(f"[t6][c] {name}: "
              + " ".join(f"{n}: fin={rec['per_norm'][n]['success_final_mean']:.3f}"
                         f"/any={rec['per_norm'][n]['success_any_mean']:.3f}"
                         for n, _ in norms)
              + f"  n_pos={n_pos} ({time.time()-t00:.0f}s)", flush=True)
    results["regions"] = regions

    # ================= (d) negative control =================
    z_raw, n_pos = infer(np.ones(len(S)))
    neg = {"diag": diag(z_raw, n_pos), "per_norm": {}}
    for norm, tgt in norms:
        z = rescale(z_raw, tgt)
        scored = {}
        for v in EVAL_TASKS:
            traj, ln = batched_rollout(agent, pool(v), z, H)
            scored[f"t{v}"] = {
                "goal_success": {f"t{g}": float(m_goal(traj, ln, goal_xyz[g]).mean())
                                 for g in EVAL_TASKS},
                "lift": float(m_lift(traj, ln).mean()),
                "regions_final": {n2: float(m_region(traj, ln, c)[0].mean())
                                  for n2, c in REGIONS.items()},
            }
        own_goal = float(np.mean([scored[f"t{v}"]["goal_success"][f"t{v}"]
                                  for v in EVAL_TASKS]))
        all_goal = float(np.mean([scored[f"t{v}"]["goal_success"][f"t{g}"]
                                  for v in EVAL_TASKS for g in EVAL_TASKS]))
        neg["per_norm"][norm] = {"scored": scored, "mean_own_goal": own_goal,
                                 "mean_all_goals": all_goal,
                                 "mean_lift": float(np.mean(
                                     [scored[f"t{v}"]["lift"] for v in EVAL_TASKS])),
                                 "mean_regions_final": float(np.mean(
                                     [scored[f"t{v}"]["regions_final"][n2]
                                      for v in EVAL_TASKS for n2 in REGIONS]))}
        print(f"[t6][d] uniform {norm}: own_goal={own_goal:.3f} all_goals={all_goal:.3f} "
              f"lift={neg['per_norm'][norm]['mean_lift']:.3f} "
              f"regions={neg['per_norm'][norm]['mean_regions_final']:.3f} "
              f"({time.time()-t00:.0f}s)", flush=True)
    results["uniform_control"] = neg

    # ================= summary + pre-registered read =================
    a_mean = float(np.mean([p["per_norm"]["match_data"]["success_mean"]
                            for p in pairs.values()]))
    c_mean = float(np.mean([r_["per_norm"]["match_data"]["success_final_mean"]
                            for r_ in regions.values()]))
    b_mean = lift["per_norm"]["match_data"]["success_mean"]
    e_match_data = float(np.mean([anchors[f"t{i}"]["success"].get("match_data", np.nan)
                                  for i in EVAL_TASKS]))
    if a_mean >= 0.7 and c_mean >= 0.5:
        verdict = "CONFIRMED"
    elif a_mean >= 0.7:
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"
    results["summary"] = {
        "gate_anchor_match_g": gate_mean,
        "anchor_match_data": e_match_data,
        "a_pairs_match_data": a_mean,
        "b_lift_match_data": b_mean,
        "c_regions_final_match_data": c_mean,
        "d_uniform_own_goal_match_data": neg["per_norm"]["match_data"]["mean_own_goal"],
        "verdict_preregistered": verdict,
    }
    outp.write_text(json.dumps(results, indent=2))
    print(f"[t6] DONE ({time.time()-t00:.0f}s): gate={gate_mean:.3f} "
          f"a={a_mean:.3f} b={b_mean:.3f} c={c_mean:.3f} -> {verdict}  -> {outp}", flush=True)
    for i in list(pools):
        close_pool(i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
