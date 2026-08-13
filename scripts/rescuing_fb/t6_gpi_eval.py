#!/usr/bin/env python
"""scripts/rescuing_fb/t6_gpi_eval.py -- Spec 06 item 1: GPI over goal candidates.

Fork of t6_generality_probe.py. T6 showed z=E_rho[r.psi] gives a POOR actor for
two-goal max-mixtures (the actor under-covers the composite task). This asks:
is that an ACTOR-coverage failure that GPI (generalized policy improvement) can
repair, or a critic-side failure?

Per composite task r = max_k r_{g_k} we build a candidate set of task vectors
    Z = { z_r,  psi(g_1), ..., psi(g_K) }
where z_r = E_rho[r.psi(s')] is the inferred composite z (match_data norm, the
goal-free deployment rule) and psi(g_k) is the EXACT single-goal embedding (its
own norm -- the exact-z anchor policy, which reaches its own goal ~0.9). Per env
step we sample a_j = pi(s, z_j) for every candidate (one batched policy call over
E*(K+1) rows) and execute
    a_{j*},  j* = argmax_j  min_heads Q(s, a_j, z_r),   Q = phi(s,a).z_r/sqrt(d)
i.e. GPI: whichever candidate policy the COMPOSITE-task critic ranks highest at
this state. The critic readout is always at z_r (the composite task); only the
proposal policies differ.

Pre-registered reads (spec 06 item 1, binding):
  two-goal max-mixture success >= 0.60 => GPI fixes actor coverage (predict ~0.8+)
  0.30-0.60 = partial (report per-pair split)
  <= 0.30   => actor coverage is NOT the mixture failure -> re-open E1a ladder.
Also logged: argmax-switch rate per episode (sanity: must be state-dependent, not
constant) and the per-candidate selection distribution.
No-regression gate: the T6 single-goal anchors re-run through the SAME GPI path
(candidates {z_r_single, psi(g_i)}) must reproduce ~0.9, else the GPI path itself
is broken and the mixture read is void.

Run (CPU-only, cores <=48 per user directive -- box is shared):
  taskset -c 0-47 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
    JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/t6_gpi_eval.py \\
    --run-dir '/home/adityamohan/exp_rgfb/factored-fb/RGFB-Cube-Single-State-d512-v0/rgfb_*_s0' \\
    --checkpoint-step 500000
  # G4 d512 (explicit arch, no flags.json read):
  ... --checkpoint-dir ~/bridge_ckpt_backup/.../gciql_g4_zcond/sd000_* \\
      --checkpoint-step 1000000 --arch-explicit --d 512 --actor-cond z
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
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))

EVAL_TASKS = {i: f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)}
CUBE_QPOS = slice(14, 17)          # privileged cube xyz, meters (E1a convention)
CUBE_OBS = slice(19, 22)           # obs cube xyz = (m - CEN) * 10
CEN = np.array([0.425, 0.0, 0.0])
THRESH = 0.04                      # goal-reach radius, meters


def decode_cube(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs)[..., CUBE_OBS] / 10.0 + CEN


def load_agent_flexible(run_dir: Path, step: int, arch_explicit: bool,
                        d: int, actor_cond: str, actor_vf_cond: str):
    """Restore a GCIQLFlowAgent. Default: read the arch from run_dir/flags.json
    (the RG-FB path -- captures actor_vf_cond='z', which widens the vf param
    tree, so an explicit-arch restore would mis-align). arch_explicit=True: build
    from CLI knobs (the G4 zcond ckpt, which predates flags.json capture of
    actor_vf_cond and is plain actor_vf_cond='none')."""
    import jax.numpy as jnp
    import ogbench
    from gciql_flow import GCIQLFlowAgent, get_config
    from utils.flax_utils import restore_agent

    tmp_env = ogbench.make_env_and_datasets(EVAL_TASKS[1], env_only=True)
    ex_obs_np, _ = tmp_env.reset()
    ex_obs = jnp.asarray(ex_obs_np, dtype=jnp.float32)[None]
    ex_act = jnp.zeros((1, *tmp_env.action_space.shape), dtype=jnp.float32)
    if hasattr(tmp_env, "close"):
        tmp_env.close()

    cfg = dict(get_config())
    flags_path = run_dir / "flags.json"
    if not arch_explicit and flags_path.exists():
        flags = json.loads(flags_path.read_text())
        cfg.update(flags["agent"])
        cfg["encoder"] = flags["agent"].get("encoder")
        cfg["frame_stack"] = flags["agent"].get("frame_stack")
        seed = int(flags["seed"])
    else:
        cfg["value_arch"] = "bilinear"
        cfg["value_latent_dim"] = d
        cfg["actor_cond"] = actor_cond
        cfg["actor_vf_cond"] = actor_vf_cond
        cfg["encoder"] = None
        cfg["frame_stack"] = None
        seed = 0
    d_out = int(cfg["value_latent_dim"])
    agent = GCIQLFlowAgent.create(seed, ex_obs, ex_act, cfg)
    agent = restore_agent(agent, str(run_dir), step)
    return agent, d_out, cfg


def make_psi(agent, src="value", act_dim=None, fbz=False):
    import jax.numpy as jnp

    def psi(obs, chunk=8192):
        obs = np.asarray(obs, np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None]
        out = []
        for i in range(0, len(obs), chunk):
            g = jnp.asarray(obs[i:i + chunk])
            if src == "critic":
                # measure-only cells train ONLY the critic towers, and their z-actors
                # read the CRITIC psi (actor_z_from_critic) -- the value-net psi is
                # untrained there. phi needs placeholder actions; fb_forward_z critics
                # additionally take a z_goals argument.
                a0 = jnp.zeros((g.shape[0], int(act_dim)), jnp.float32)
                args = (g, g, a0) + ((g,) if fbz else ())
                _v, _phi, ps = agent.network.select("critic")(*args, info=True)
                ps = ps.mean(axis=0) if ps.ndim == 3 else ps
            else:
                _v, _phi, ps = agent.network.select("value")(g, g, info=True)
            out.append(np.asarray(ps))
        out = np.concatenate(out, 0)
        return out[0] if single else out

    return psi


def make_q_at_z(agent, d, fbz=False):
    """Jitted GPI critic readout: Q(s,a,z_r) = phi(s,a).z_r/sqrt(d), min over the
    2 critic heads. Plain bilinear: phi depends on (obs, action) only -- goals is a
    placeholder. fbz (fb_forward_z critics): phi = F(s,a,z_r), the task-conditioned
    forward map evaluated AT z_r via the z_vec path -- the true FB readout
    F(s,a,z_r).z_r/sqrt(d)."""
    import jax
    import jax.numpy as jnp
    sqd = jnp.sqrt(float(d))

    @jax.jit
    def q_at_z(obs, acts, z_r):
        if fbz:
            zb = jnp.broadcast_to(z_r[None, :], (obs.shape[0], z_r.shape[0]))
            _, phi, _ = agent.network.select("critic")(obs, obs, acts, obs,
                                                       info=True, z_vec=zb)     # [2,N,d]
        else:
            _, phi, _ = agent.network.select("critic")(obs, obs, acts, info=True)  # [2,N,d]
        q = (phi * z_r[None, None, :]).sum(-1) / sqd                            # [2,N]
        return q.min(axis=0)                                                    # [N]

    return q_at_z


def rescale(z, tgt):
    return (z * tgt / (np.linalg.norm(z) + 1e-8)).astype(np.float32)


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def batched_gpi_rollout(agent, q_at_z, envs, z_r, cand_zs, H=200):
    """GPI lockstep rollout across len(envs) envs. cand_zs: [C,d] candidate task
    vectors the proposal policies condition on; z_r: [d] the critic-readout task
    vector (the composite task). Returns (cube_traj [n,H,3] m, lengths [n],
    switch_rate [n] per-episode argmax-switch fraction, choice_counts [C] total
    executions per candidate over real steps)."""
    import jax
    n = len(envs)
    C = len(cand_zs)
    Zc = np.asarray(cand_zs, np.float32)                     # [C,d]
    z_r = np.asarray(z_r, np.float32)
    obs = np.zeros((n, envs[0].observation_space.shape[0]), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    lengths = np.zeros(n, int)
    prev_j = -np.ones(n, int)
    switches = np.zeros(n, float)
    steps_taken = np.zeros(n, float)
    choice_counts = np.zeros(C, float)
    rng = jax.random.PRNGKey(0)
    import jax.numpy as jnp
    z_r_j = jnp.asarray(z_r)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        # one batched policy call over E*C (env-major): row (i,c) = i*C + c.
        obs_big = np.repeat(obs, C, axis=0)                  # [n*C, obs]
        z_big = np.tile(Zc, (n, 1))                          # [n*C, d]
        acts_big = np.asarray(agent.sample_actions_with_z(
            observations=jnp.asarray(obs_big), z=jnp.asarray(z_big),
            seed=sub, temperature=0.0))                      # [n*C,5]
        acts_big = np.clip(acts_big, -1, 1)
        qmin = np.asarray(q_at_z(jnp.asarray(obs_big), jnp.asarray(acts_big), z_r_j))  # [n*C]
        qmat = qmin.reshape(n, C)
        jstar = qmat.argmax(1)                               # [n]
        acts_exec = acts_big.reshape(n, C, 5)[np.arange(n), jstar]
        for i, e in enumerate(envs):
            if not active[i]:
                traj[i, t] = traj[i, t - 1]
                continue
            choice_counts[jstar[i]] += 1
            if prev_j[i] >= 0 and jstar[i] != prev_j[i]:
                switches[i] += 1
            prev_j[i] = jstar[i]
            steps_taken[i] += 1
            o, _r, term, trunc, _ = e.step(acts_exec[i])
            obs[i] = o
            traj[i, t] = e.unwrapped.data.qpos[CUBE_QPOS]
            lengths[i] = t + 1
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    switch_rate = switches / np.maximum(steps_taken - 1, 1)   # per-episode fraction of switches
    return traj, lengths, switch_rate, choice_counts


def m_goal(traj, lengths, goal_xyz, thresh=THRESH):
    return np.array([(np.linalg.norm(traj[i, :lengths[i]] - goal_xyz, axis=-1)
                      < thresh).any() for i in range(len(traj))], float)


def m_pair(traj, lengths, g1, g2):
    succ, first = [], []
    for i in range(len(traj)):
        d1 = np.linalg.norm(traj[i, :lengths[i]] - g1, axis=-1) < THRESH
        d2 = np.linalg.norm(traj[i, :lengths[i]] - g2, axis=-1) < THRESH
        succ.append(float(d1.any() or d2.any()))
        t1 = int(np.argmax(d1)) if d1.any() else 10 ** 9
        t2 = int(np.argmax(d2)) if d2.any() else 10 ** 9
        first.append(0 if not (d1.any() or d2.any()) else (1 if t1 <= t2 else 2))
    return np.array(succ), np.array(first)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (must hold params_<step>.pkl; flags.json read unless --arch-explicit)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--arch-explicit", action="store_true",
                    help="build arch from CLI knobs instead of flags.json (the G4 zcond ckpt)")
    ap.add_argument("--d", type=int, default=512, help="value_latent_dim (arch-explicit only)")
    ap.add_argument("--actor-cond", default="z")
    ap.add_argument("--actor-vf-cond", default="none")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--stage", choices=["gate", "full"], default="full",
                    help="gate = single-goal anchors through the GPI path only")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()

    t00 = time.time()
    agent, d, cfg = load_agent_flexible(run_dir, args.checkpoint_step, args.arch_explicit,
                                        args.d, args.actor_cond, args.actor_vf_cond)
    psi = make_psi(agent)
    q_at_z = make_q_at_z(agent, d)
    sqrtd = float(np.sqrt(d))
    print(f"[gpi] restored {run_dir.name} step={args.checkpoint_step} d={d} "
          f"actor_cond={cfg.get('actor_cond')} actor_vf_cond={cfg.get('actor_vf_cond','none')}",
          flush=True)

    # ---- shared relabel sample (task1 dataset; obs shared across variants) ----
    _e0, ds1, _ = ogbench.make_env_and_datasets(EVAL_TASKS[1])
    n_ds = len(ds1["observations"])
    idx = np.random.default_rng(0).choice(n_ds, size=min(args.relabel_size, n_ds), replace=False)
    S = np.asarray(ds1["observations"], np.float32)[idx]
    cube_ds = decode_cube(S)
    Psi = psi(S).astype(np.float64)
    match_data = float(np.linalg.norm(Psi, axis=-1).mean())

    goal_obs, goal_xyz = {}, {}
    for i, task in EVAL_TASKS.items():
        env = ogbench.make_env_and_datasets(task, env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        goal_xyz[i] = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        env.close()
    psi_g = {i: psi(goal_obs[i]) for i in EVAL_TASKS}

    syn_r = {i: (np.linalg.norm(cube_ds - goal_xyz[i], axis=-1) < THRESH).astype(float)
             for i in EVAL_TASKS}

    def infer(r):
        r = np.asarray(r, np.float64)
        return (r[:, None] * Psi).sum(0), int(r.sum())

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

    H = args.horizon
    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step, "d": d,
        "episodes": args.episodes, "horizon": H, "relabel_size": len(S),
        "match_data_norm": match_data, "sqrt_d": sqrtd,
        "arch_explicit": args.arch_explicit,
        "goal_xyzs_m": {f"t{i}": goal_xyz[i].tolist() for i in EVAL_TASKS},
        "spec": "PAPER/rescuing-fb/specs/06-phenomenon-implementations.md item 1",
    }}

    # ================= no-regression GATE: single-goal anchors via GPI =================
    # candidates {z_r_single (match_data), psi(g_i)}; critic readout at z_r_single.
    anchors = {}
    for i in EVAL_TASKS:
        zr_raw, n_pos = infer(syn_r[i])
        z_r = rescale(zr_raw, match_data)
        cand = [z_r, psi_g[i].astype(np.float32)]
        traj, ln, sw, cc = batched_gpi_rollout(agent, q_at_z, pool(i), z_r, cand, H)
        succ = float(m_goal(traj, ln, goal_xyz[i]).mean())
        anchors[f"t{i}"] = {"success": succ, "n_pos": n_pos,
                            "switch_rate": float(sw.mean()),
                            "choice_frac": (cc / max(cc.sum(), 1)).tolist(),
                            "cand_labels": ["z_r", f"psi_g{i}"]}
        print(f"[gpi][gate] t{i}: succ={succ:.3f} switch={sw.mean():.3f} "
              f"choose={anchors[f't{i}']['choice_frac']} n_pos={n_pos} ({time.time()-t00:.0f}s)",
              flush=True)
    results["anchors"] = anchors
    gate_mean = float(np.mean([anchors[f"t{i}"]["success"] for i in EVAL_TASKS]))
    results["gate"] = {"mean_single_goal_gpi": gate_mean, "threshold": 0.90,
                       "passed": bool(gate_mean >= 0.85)}
    print(f"[gpi] GATE single-goal GPI mean = {gate_mean:.3f} -> "
          f"{'PASS' if gate_mean >= 0.85 else 'WEAK (interpret with care)'}", flush=True)

    outp = Path(args.out) if args.out else \
        REPO / "analysis" / "rescuing-fb" / f"t6_gpi_{run_dir.name}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    if args.stage == "gate":
        close_all()
        return 0

    # ================= two-goal max-mixtures via GPI =================
    # candidates {z_r (composite, match_data), psi(g_i), psi(g_j)}; readout at z_r.
    pairs = {}
    for i, j in combinations(range(1, 6), 2):
        r = np.maximum(syn_r[i], syn_r[j])
        zr_raw, n_pos = infer(r)
        z_r = rescale(zr_raw, match_data)
        cand = [z_r, psi_g[i].astype(np.float32), psi_g[j].astype(np.float32)]
        succs, firsts, sws = [], [], []
        cc_tot = np.zeros(3)
        per_var = {}
        for v in (i, j):
            traj, ln, sw, cc = batched_gpi_rollout(agent, q_at_z, pool(v), z_r, cand, H)
            s, f = m_pair(traj, ln, goal_xyz[i], goal_xyz[j])
            per_var[f"t{v}"] = {"success": float(s.mean()),
                                "hit_g1": int((f == 1).sum()), "hit_g2": int((f == 2).sum()),
                                "switch_rate": float(sw.mean())}
            succs.append(s); firsts.append(f); sws.append(sw); cc_tot += cc
        firsts = np.concatenate(firsts)
        pairs[f"t{i}-t{j}"] = {
            "n_pos": n_pos, "cos_zr_gi": cosine(zr_raw, psi_g[i]),
            "cos_zr_gj": cosine(zr_raw, psi_g[j]),
            "success_mean": float(np.concatenate(succs).mean()),
            "per_variant": per_var,
            "switch_rate": float(np.concatenate(sws).mean()),
            "choice_frac": (cc_tot / max(cc_tot.sum(), 1)).tolist(),
            "cand_labels": ["z_r", f"psi_g{i}", f"psi_g{j}"],
            "frac_first_g1": float((firsts == 1).sum() / max((firsts > 0).sum(), 1)),
        }
        print(f"[gpi][pair] t{i}-t{j}: succ={pairs[f't{i}-t{j}']['success_mean']:.3f} "
              f"switch={pairs[f't{i}-t{j}']['switch_rate']:.3f} "
              f"choose={[round(x,2) for x in pairs[f't{i}-t{j}']['choice_frac']]} "
              f"n_pos={n_pos} ({time.time()-t00:.0f}s)", flush=True)
    results["pairs"] = pairs

    a_mean = float(np.mean([p["success_mean"] for p in pairs.values()]))
    if a_mean >= 0.60:
        verdict = "GPI_FIXES_COVERAGE"
    elif a_mean >= 0.30:
        verdict = "PARTIAL"
    else:
        verdict = "COVERAGE_NOT_THE_FAILURE"
    results["summary"] = {
        "gate_single_goal_gpi": gate_mean,
        "two_goal_mixture_gpi_success": a_mean,
        "mean_switch_rate": float(np.mean([p["switch_rate"] for p in pairs.values()])),
        "verdict_preregistered": verdict,
    }
    outp.write_text(json.dumps(results, indent=2))
    print(f"[gpi] DONE ({time.time()-t00:.0f}s): gate={gate_mean:.3f} "
          f"two_goal_gpi={a_mean:.3f} -> {verdict}  -> {outp}", flush=True)
    close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
