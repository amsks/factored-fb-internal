#!/usr/bin/env python
"""scripts/rescuing_fb/rgfb_rerank_eval.py -- Spec 06 item 3: selection-based
extraction on RG-FB (no training).

Deployment-mode RG-FB via reranking: at every env step, draw N action proposals
from the trained BC flow, score each with the FROZEN min-head critic at the
INFERRED task vector z, and execute argmax_i min_heads Q(s, a_i, z). z is inferred
once per task from the offline relabel (moment z = E_rho[r.psi(s')], match_data
norm -- the same goal-free rule as eval_rgfb_zeroshot.py). Nothing is retrained.

Proposal sources (spec: "goal-agnostic vf OR the z-conditioned vf -- report both").
From a single RG-FB v0 checkpoint (actor_vf_cond='z') we report both proposal
distributions that the checkpoint actually supports:
  flow_z   : roll the z-conditioned BC vector field actor_vf at the inferred z.
  oneshot_z: the one-shot distilled GCNoiseActor at the inferred z (sample_actions_with_z).
A goal-agnostic-vf checkpoint (actor_vf_cond='none') is handled too: proposals
roll actor_vf on obs only (flow_none). The conditioning is auto-detected from the
run's flags.json.

N-sweep {8,16,32,64}: the off-support-peak check on the z-critic. If success
DECLINES through N=64 the critic is over-optimistic off the support peak -> cap N
and report; if it holds/climbs, selection is safe. Floor = execute proposal 0
(no rerank, N=1) = the raw BC-flow policy.

Pre-registered read (spec 06 item 3): rerank success >= 0.90 => deployment-mode
RG-FB matches the supervised anchor ("zero-shot 0.90 via selection").

Run (CPU is fine; one process, all seeds):
  JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/rgfb_rerank_eval.py \\
    --run-dirs '/home/adityamohan/exp_rgfb/factored-fb/RGFB-Cube-Single-State-d50-v0/*' \\
    --checkpoint-step 500000 --episodes 25 --n-sweep 8,16,32,64
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
CUBE_QPOS = slice(14, 17)
THRESH = 0.04


def load_run(run_dir: Path, step: int):
    import jax.numpy as jnp
    import ogbench
    from gciql_flow import GCIQLFlowAgent, get_config
    from utils.flax_utils import restore_agent

    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(get_config()); cfg.update(flags["agent"])
    cfg["encoder"] = flags["agent"].get("encoder"); cfg["frame_stack"] = flags["agent"].get("frame_stack")
    env = ogbench.make_env_and_datasets(EVAL_TASKS[0], env_only=True)
    ex_obs = jnp.asarray(env.reset()[0], jnp.float32)[None]
    ex_act = jnp.zeros((1, *env.action_space.shape), jnp.float32)
    if hasattr(env, "close"):
        env.close()
    agent = GCIQLFlowAgent.create(int(flags["seed"]), ex_obs, ex_act, cfg)
    agent = restore_agent(agent, str(run_dir), step)
    return agent, flags, cfg


def make_psi(agent):
    import jax.numpy as jnp

    def psi(obs, chunk=8192):
        obs = np.asarray(obs, np.float32); single = obs.ndim == 1
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


def make_flow_and_q(agent, d, vf_cond):
    """Returns (flow_proposals, oneshot_proposals, q_at_z) jitted helpers.
    flow_proposals(obs[E,obs], z[d], noise[E,N,act]) -> [E,N,act]: roll actor_vf.
    q_at_z(obs[E*N,obs], acts[E*N,act], z[d]) -> [E*N]: min-head Q=phi.z/sqrt(d)."""
    import jax
    import jax.numpy as jnp
    fs = agent.config["flow_steps"]; act_dim = agent.config["action_dim"]
    sqd = jnp.sqrt(float(d))

    def _vf_input(obs, z):
        if vf_cond == "z":
            zt = jnp.broadcast_to(z[None, :], (obs.shape[0], z.shape[-1]))
            return jnp.concatenate([obs, zt], axis=-1)
        return obs  # 'none' (goal-agnostic vf)

    @jax.jit
    def flow_proposals(obs, z, noise):
        # obs[E,obs], z[d], noise[E,N,act] -> [E,N,act]
        E, N = noise.shape[0], noise.shape[1]
        obs_rep = jnp.repeat(obs, N, axis=0)                       # [E*N,obs]
        vf_obs = _vf_input(obs_rep, z)
        a = noise.reshape(E * N, act_dim)
        for i in range(fs):
            t = jnp.ones((E * N, 1)) * (i / fs)
            a = a + agent.network.select("actor_vf")(vf_obs, a, t) / fs
        return jnp.clip(a, -1.0, 1.0).reshape(E, N, act_dim)

    @jax.jit
    def oneshot_proposals(obs, z, noise):
        # one-shot GCNoiseActor at z; noise[E,N,act] -> [E,N,act]
        E, N = noise.shape[0], noise.shape[1]
        obs_rep = jnp.repeat(obs, N, axis=0)
        zt = jnp.broadcast_to(z[None, :], (E * N, z.shape[-1]))
        a = agent.network.select("actor")(obs_rep, zt, noise.reshape(E * N, act_dim))
        return jnp.clip(a, -1.0, 1.0).reshape(E, N, act_dim)

    @jax.jit
    def q_at_z(obs, acts, z):
        _, phi, _ = agent.network.select("critic")(obs, obs, acts, info=True)  # [2,M,d]
        q = (phi * z[None, None, :]).sum(-1) / sqd                             # [2,M]
        return q.min(axis=0)                                                   # [M]

    return flow_proposals, oneshot_proposals, q_at_z


def rescale(z, tgt):
    return (z * tgt / (np.linalg.norm(z) + 1e-8)).astype(np.float32)


def rerank_rollout(agent, propose, q_at_z, envs, z, N, H=200):
    """Lockstep rerank rollout. propose(obs,z,noise)->[E,N,act]; execute
    argmax_i min-head Q(s,a_i,z). N=1 => floor (execute proposal 0). Returns
    per-episode success (cube within THRESH of the task goal at any step)."""
    import jax
    import jax.numpy as jnp
    n = len(envs); act_dim = agent.config["action_dim"]
    obs = np.zeros((n, envs[0].observation_space.shape[0]), np.float32)
    goal_xyz = np.asarray(envs[0].unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool); hit = np.zeros(n, bool)
    rng = jax.random.PRNGKey(0)
    z_j = jnp.asarray(z)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        noise = jax.random.normal(sub, (n, N, act_dim))
        cand = np.asarray(propose(jnp.asarray(obs), z_j, noise))               # [n,N,act]
        if N == 1:
            acts = cand[:, 0, :]
        else:
            cand_flat = jnp.asarray(cand.reshape(n * N, act_dim))
            obs_rep = jnp.asarray(np.repeat(obs, N, axis=0))
            q = np.asarray(q_at_z(obs_rep, cand_flat, z_j)).reshape(n, N)
            acts = cand[np.arange(n), q.argmax(1)]
        acts = np.clip(acts, -1, 1)
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, _r, term, trunc, info = e.step(acts[i])
            obs[i] = o
            cube = e.unwrapped.data.qpos[CUBE_QPOS]
            if np.linalg.norm(cube - goal_xyz) < THRESH:
                hit[i] = True
            if term or trunc:
                active[i] = False
        if not active.any():
            break
    return hit.astype(float)


def eval_run(run_dir: Path, args) -> dict:
    import ogbench
    agent, flags, cfg = load_run(run_dir, args.checkpoint_step)
    d = int(cfg["value_latent_dim"]); seed = int(flags["seed"])
    vf_cond = cfg.get("actor_vf_cond", "none")
    psi = make_psi(agent)
    flow_prop, oneshot_prop, q_at_z = make_flow_and_q(agent, d, vf_cond)
    # proposal sources this checkpoint supports.
    sources = {}
    sources[f"flow_{vf_cond}"] = flow_prop
    if cfg.get("actor_cond") == "z":
        sources["oneshot_z"] = oneshot_prop
    Ns = [int(x) for x in args.n_sweep.split(",")]
    print(f"[rerank] {run_dir.name}: d={d} seed={seed} vf_cond={vf_cond} "
          f"sources={list(sources)} N={Ns}", flush=True)

    per_task = {t: {} for t in EVAL_TASKS}
    for task in EVAL_TASKS:
        env, ds, _ = ogbench.make_env_and_datasets(task)
        S = np.asarray(ds["observations"], np.float32)
        r = np.asarray(ds["rewards"], np.float64) + 1.0
        m = min(args.relabel_size, len(S))
        idx = np.random.default_rng(seed).choice(len(S), size=m, replace=False)
        Ssub, rsub = S[idx], r[idx]
        Psi = psi(Ssub)
        mean_psi = float(np.linalg.norm(Psi, axis=-1).mean())
        z = rescale((rsub[:, None] * Psi).sum(0), mean_psi)
        pool = [ogbench.make_env_and_datasets(task, env_only=True) for _ in range(args.episodes)]
        for src_name, propose in sources.items():
            for N in ([1] + Ns):
                key = f"{src_name}_N{N}" if N > 1 else f"{src_name}_floor"
                s = rerank_rollout(agent, propose, q_at_z, pool, z, N, args.horizon)
                per_task[task][key] = float(s.mean())
        for e in pool:
            if hasattr(e, "close"):
                e.close()
        if hasattr(env, "close"):
            env.close()
        print(f"  {task}: " + " ".join(f"{k}={v:.2f}" for k, v in per_task[task].items()), flush=True)

    # aggregate per (source, N)
    keys = sorted({k for t in per_task.values() for k in t})
    agg = {k: float(np.mean([per_task[t][k] for t in EVAL_TASKS if k in per_task[t]])) for k in keys}
    return {"run_dir": str(run_dir), "seed": seed, "d": d, "vf_cond": vf_cond,
            "run_group": flags.get("run_group"), "per_task": per_task, "agg": agg}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", required=True, nargs="+")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--n-sweep", default="8,16,32,64")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dirs = sorted({Path(p).resolve() for pat in args.run_dirs
                       for p in glob.glob(str(Path(pat).expanduser()))})
    run_dirs = [p for p in run_dirs if (p / "flags.json").exists()]
    assert run_dirs, f"no run dirs with flags.json matched {args.run_dirs}"

    runs = [eval_run(rd, args) for rd in run_dirs]
    all_keys = sorted({k for r in runs for k in r["agg"]})
    summary_agg = {k: (float(np.mean([r["agg"][k] for r in runs if k in r["agg"]])),
                       float(np.std([r["agg"][k] for r in runs if k in r["agg"]])))
                   for k in all_keys}
    # pre-registered read on the best rerank cell (any source, N in sweep, not floor).
    rerank_cells = {k: v for k, v in summary_agg.items() if "_N" in k}
    best = max(rerank_cells.items(), key=lambda kv: kv[1][0]) if rerank_cells else (None, (None, None))
    read = ("MATCHES_ANCHOR (>=0.90 via selection)" if best[1][0] is not None and best[1][0] >= 0.90
            else "BELOW_ANCHOR")
    group = runs[0].get("run_group") or "unknown"
    summ = {"group": group, "checkpoint_step": args.checkpoint_step, "n_seeds": len(runs),
            "episodes": args.episodes, "agg_mean_std": {k: v for k, v in summary_agg.items()},
            "best_rerank_cell": best[0], "best_rerank_success": best[1][0],
            "verdict_preregistered": read, "runs": runs}
    out = Path(args.out) if args.out else \
        REPO / "analysis" / "rescuing-fb" / f"rgfb_rerank_{group}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=2))
    print(f"[rerank] {group}: best {best[0]}={best[1][0]} -> {read}  -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
