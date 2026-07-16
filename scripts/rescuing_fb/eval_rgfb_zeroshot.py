#!/usr/bin/env python
"""scripts/rescuing_fb/eval_rgfb_zeroshot.py — RG-FB zero-shot evaluation (spec §2.3).

Generalizes scripts/eval_zeroshot_inference.py (E1a) to ANY gciql_flow run dir:
the agent architecture knobs (value_arch, value_latent_dim, actor_cond,
actor_vf_cond, seed, ...) are read from the run dir's flags.json, so the same
script evaluates v0, the d512 fallback, and any later variant.

Protocol (PAPER/rescuing-fb/02-rgfb-spec.md §1.3): given reward samples
{(s'_i, r_i)} from the offline relabeled dataset (r in {0,1}),

    z_raw = sum_i r_i . psi(s'_i)              (dataset moment E_rho[r.psi])
    z     = z_raw rescaled to a target norm

GOAL-FREE norm hygiene: the actor consumed RAW psi(g) at training time and
||psi|| is task-dependent (no project_z), so the inferred z must be rescaled to
the typical ||psi|| scale WITHOUT peeking at the goal: target = mean ||psi(s')||
over the relabel sample ('match_data', the goal-free version of E1a's
'match_g'). Both norms are reported, as in E1a:
    match_data : ||z|| = mean_i ||psi(s'_i)||   (headline, goal-free)
    sqrtd      : ||z|| = sqrt(d)                (the FB-spec choice)

The deployed policy NEVER sees the goal observation, only the inferred z
(sample_actions_with_z). The exact-z arm (z = psi(g), sample_actions) is rolled
too, clearly labeled exact_z_success -- it is the DIAGNOSTIC ANCHOR, not the
headline. cos(z, psi(g)) and ||psi(g)|| are reported as diagnostics only.

Run (CPU is fine for rollouts; one process, all seeds):
  JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/eval_rgfb_zeroshot.py \\
    --run-dirs '/home/adityamohan/exp_rgfb/factored-fb/RGFB-Cube-Single-State-d50-v0/*' \\
    --checkpoint-step 500000 --episodes 50 --relabel-size 50000
JSON out (default): analysis/rescuing-fb/rgfb_zeroshot_<group>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))

EVAL_TASKS = [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)]
CUBE_QPOS = slice(14, 17)   # privileged cube xyz (matches E1a / eval_zcond)
THRESH = 0.04


def load_run(run_dir: Path, step: int):
    """Restore a GCIQLFlowAgent with the architecture recorded in the run's own
    flags.json (no CLI arch knobs to get wrong). Returns (agent, flags)."""
    import jax.numpy as jnp
    import ogbench
    from gciql_flow import GCIQLFlowAgent, get_config
    from utils.flax_utils import restore_agent

    flags = json.loads((run_dir / "flags.json").read_text())
    agent_cfg = flags["agent"]
    cfg = dict(get_config())
    cfg.update(agent_cfg)                      # value_arch/latent_dim/actor_cond/actor_vf_cond/...
    cfg["encoder"] = agent_cfg.get("encoder")  # state runs: None
    cfg["frame_stack"] = agent_cfg.get("frame_stack")

    tmp_env = ogbench.make_env_and_datasets(EVAL_TASKS[0], env_only=True)
    ex_obs_np, _ = tmp_env.reset()
    ex_obs = jnp.asarray(ex_obs_np, dtype=jnp.float32)[None]
    ex_act = jnp.zeros((1, *tmp_env.action_space.shape), dtype=jnp.float32)
    if hasattr(tmp_env, "close"):
        tmp_env.close()

    agent = GCIQLFlowAgent.create(int(flags["seed"]), ex_obs, ex_act, cfg)
    agent = restore_agent(agent, str(run_dir), step)
    return agent, flags


def make_psi(agent):
    """B := the VALUE net's psi (the same map actor_cond='z' consumes)."""
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


def rollout(agent, env, cond, goal_xyz, episodes, mode, max_steps=200, seed0=0):
    """mode='exact' => cond is the RAW goal obs (agent maps through value.psi
    internally); mode='inferred' => cond is a precomputed d-dim z. Success =
    cube within THRESH of the goal xyz at any step."""
    import jax
    cond = np.asarray(cond, np.float32)
    rng = jax.random.PRNGKey(seed0)
    succ = []
    for ep in range(episodes):
        obs, _info = env.reset(seed=seed0 + ep)
        ep_hit = False
        for _t in range(max_steps):
            rng, sub = jax.random.split(rng)
            o = np.asarray(obs, np.float32)
            if mode == "exact":
                a = np.asarray(agent.sample_actions(observations=o, goals=cond,
                                                    seed=sub, temperature=0.0))
            else:
                a = np.asarray(agent.sample_actions_with_z(observations=o, z=cond,
                                                           seed=sub, temperature=0.0))
            obs, _r, term, trunc, _info = env.step(np.clip(a, -1, 1))
            cube = env.unwrapped.data.qpos[CUBE_QPOS].copy()
            if np.linalg.norm(cube - goal_xyz) < THRESH:
                ep_hit = True
            if term or trunc:
                break
        succ.append(float(ep_hit))
    return float(np.mean(succ))


def rescale(z: np.ndarray, target_norm: float) -> np.ndarray:
    return z * target_norm / (np.linalg.norm(z) + 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def eval_run(run_dir: Path, args) -> dict:
    import ogbench

    agent, flags = load_run(run_dir, args.checkpoint_step)
    d = int(flags["agent"]["value_latent_dim"])
    sqrtd = float(np.sqrt(d))
    psi = make_psi(agent)
    seed = int(flags["seed"])
    print(f"[rgfb] {run_dir.name}: restored step={args.checkpoint_step} seed={seed} "
          f"d={d} actor_cond={flags['agent'].get('actor_cond')} "
          f"actor_vf_cond={flags['agent'].get('actor_vf_cond', 'none')}")

    tasks = EVAL_TASKS if not args.tasks else \
        [f"cube-single-play-singletask-task{i}-v0" for i in args.tasks.split(",")]
    rng = np.random.default_rng(seed)
    per_task = {}
    for task in tasks:
        env, train_ds, _val = ogbench.make_env_and_datasets(task)
        S_all = np.asarray(train_ds["observations"], np.float32)
        r_all = np.asarray(train_ds["rewards"], np.float64) + 1.0     # indicator 0/1
        m = min(args.relabel_size, len(S_all))
        idx = rng.choice(len(S_all), size=m, replace=False)
        S, r = S_all[idx], r_all[idx]

        _obs, info = env.reset(seed=0)
        goal_obs = np.asarray(info["goal"], np.float32)
        goal_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        psi_g = psi(goal_obs)
        norm_g = float(np.linalg.norm(psi_g))

        Psi = psi(S)                                   # [m, d]
        mean_psi_norm = float(np.linalg.norm(Psi, axis=-1).mean())  # goal-free target
        rec = {"n_samples": int(m), "r_pos": float(r.mean()),
               "psi_g_norm": norm_g, "mean_psi_data_norm": mean_psi_norm,
               "sqrt_d": sqrtd}

        # Diagnostic anchor (labeled): exact z = psi(g) through the actor's own map.
        rec["exact_z_success"] = rollout(agent, env, goal_obs, goal_xyz,
                                         args.episodes, "exact")

        if r.sum() == 0:
            print(f"  {task}: r_pos=0 in subsample -> inference undefined; "
                  f"exact={rec['exact_z_success']:.3f}")
            rec["zeroshot"] = None
            per_task[task] = rec
            if hasattr(env, "close"):
                env.close()
            continue

        z_raw = (r[:, None] * Psi).sum(0)              # moment E_rho[r.psi] (unnorm.)
        entry = {"cos_to_psi_g": cosine(z_raw, psi_g),
                 "raw_norm": float(np.linalg.norm(z_raw))}
        for norm_mode, tgt in (("match_data", mean_psi_norm), ("sqrtd", sqrtd)):
            entry[f"success_{norm_mode}"] = rollout(
                agent, env, rescale(z_raw, tgt).astype(np.float32), goal_xyz,
                args.episodes, "inferred")
        rec["zeroshot"] = entry
        per_task[task] = rec
        print(f"  {task}: exact={rec['exact_z_success']:.3f}  "
              f"zeroshot match_data={entry['success_match_data']:.3f} "
              f"sqrtd={entry['success_sqrtd']:.3f}  cos={entry['cos_to_psi_g']:.3f}  "
              f"||z||tgt={mean_psi_norm:.2f} (||psi_g||={norm_g:.2f}, sqrt_d={sqrtd:.2f})")
        if hasattr(env, "close"):
            env.close()

    def mean_over(getter):
        vals = [getter(v) for v in per_task.values()]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "run_dir": str(run_dir), "seed": seed, "d": d,
        "run_group": flags.get("run_group"),
        "actor_vf_cond": flags["agent"].get("actor_vf_cond", "none"),
        "per_task": per_task,
        "mean_exact_z_success": mean_over(lambda v: v.get("exact_z_success")),
        "mean_zeroshot_match_data": mean_over(
            lambda v: (v.get("zeroshot") or {}).get("success_match_data")),
        "mean_zeroshot_sqrtd": mean_over(
            lambda v: (v.get("zeroshot") or {}).get("success_sqrtd")),
        "mean_cos": mean_over(lambda v: (v.get("zeroshot") or {}).get("cos_to_psi_g")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", required=True, nargs="+",
                    help="run dir(s) or glob(s); each must hold flags.json + params_<step>.pkl")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--tasks", default="", help="comma-sep task indices 1..5 (default all)")
    ap.add_argument("--out", default="",
                    help="output JSON (default analysis/rescuing-fb/rgfb_zeroshot_<group>.json)")
    args = ap.parse_args()

    run_dirs = sorted({Path(p).resolve() for pat in args.run_dirs
                       for p in glob.glob(str(Path(pat).expanduser()))})
    run_dirs = [p for p in run_dirs if (p / "flags.json").exists()]
    assert run_dirs, f"no run dirs with flags.json matched {args.run_dirs}"

    runs = [eval_run(rd, args) for rd in run_dirs]

    def agg(key):
        vals = [r[key] for r in runs if r[key] is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    zs_m, zs_s = agg("mean_zeroshot_match_data")
    ex_m, ex_s = agg("mean_exact_z_success")
    group = runs[0].get("run_group") or "unknown"
    summ = {
        "group": group,
        "checkpoint_step": args.checkpoint_step,
        "episodes": args.episodes, "relabel_size": args.relabel_size,
        "n_seeds": len(runs),
        "zeroshot_success": zs_m, "zeroshot_success_std": zs_s,
        "exact_z_success": ex_m, "exact_z_success_std": ex_s,
        "zeroshot_minus_exact_gap": (zs_m - ex_m) if (zs_m is not None and ex_m is not None) else None,
        "zeroshot_success_sqrtd": agg("mean_zeroshot_sqrtd")[0],
        "mean_cos": agg("mean_cos")[0],
        "runs": runs,
    }
    out = Path(args.out) if args.out else \
        REPO / "analysis" / "rescuing-fb" / f"rgfb_zeroshot_{group}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=2))
    print(f"[rgfb] {group} @ {args.checkpoint_step} ({len(runs)} seeds): "
          f"zeroshot={zs_m} (exact={ex_m}, gap={summ['zeroshot_minus_exact_gap']}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
