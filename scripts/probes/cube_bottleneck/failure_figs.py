"""Failure-figure data: rollouts with per-episode failure phase, restored state banks, and predicted vs rollout cube futures.

Stages: rollouts, banks, predict, actionprobe, plot. All rngs are seeded from constants in this file.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import banks as BK
from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.phases import PhaseThresholds, _max_true_run

OUT_ROOT = Path("analysis/failure_figs")
METHODS = ["fb", "density_fb_z"]
METHOD_LABEL = {"fb": "FB", "density_fb_z": "Density-FB"}
# Critic name each method's runs were saved under. It names the run directories and
# fills the "critic" column of every probe file, so on-disk data keeps these names.
RUN_NAME = {"fb": "fb_tdj", "density_fb_z": "density_fb_z"}
METHOD_OF = {**{m: m for m in METHODS}, **{run: m for m, run in RUN_NAME.items()}}
OUTCOMES = ["approach", "acquisition", "maintain", "transport", "success"]
RANK = {o: i for i, o in enumerate(OUTCOMES)}

TABLE_Z = BK.TABLE_Z
XYZ_CENTER = np.array([0.425, 0.0, 0.0])
EPS_NEAR = 0.05  # transport: final cube within 5 cm of its goal
MOVING_MIN = 0.02  # cube must move if its goal is farther than this
CARRY_WINDOW = 10  # carrying: cube-to-goal distance fell over the last 10 steps
CARRY_MIN_DROP = 0.02
GOAL_RADIUS = 0.05  # radius for predicted mass near the goal
N_BANK = 64
N_CANDIDATES = 160  # candidate draws per bank; first 64 passing the gate are kept
POOL_SIZE = 1024
POOL_SEED = 0
GRID_N = 20
GRID_X = (0.25, 0.60)  # workspace bounds, padded
GRID_Y = (-0.35, 0.35)
K_CONT = 10
HORIZON = 100
N_PRED_ACTIONS = 8  # policy draws the prediction is averaged over
STATES_PER_BATCH = 8  # states per lockstep batch of K_CONT envs each
ROLLOUT_SEED = 0
BANK_SEED = 20260912


# Geometry helpers


def cube_xy_from_obs(obs: np.ndarray, c: int) -> np.ndarray:
    """Absolute xy of cube `c` from the observation (9 columns per cube starting at 19)."""
    lo = 19 + 9 * c
    return obs[..., lo : lo + 2] / BK.XYZ_SCALER + XYZ_CENTER[:2]


def n_cubes(env) -> int:
    return int(env.unwrapped._num_cubes)


def cube_goals(env) -> np.ndarray:
    """Return [C, 3] cube goal positions from the env's mocap targets."""
    u = env.unwrapped
    return np.asarray(u._data.mocap_pos[u._cube_target_mocap_ids], np.float64).copy()


def set_cube_goals(env, goals: np.ndarray) -> None:
    u = env.unwrapped
    for i in range(n_cubes(env)):
        u._data.mocap_pos[u._cube_target_mocap_ids[i]] = goals[i]


def signals_now(env) -> Dict[str, Any]:
    u = env.unwrapped
    info = u.compute_ob_info()
    C = n_cubes(env)
    return dict(
        eff=np.asarray(info["proprio/effector_pos"], np.float64),
        cube=np.stack(
            [
                np.asarray(info[f"privileged/block_{i}_pos"], np.float64)
                for i in range(C)
            ]
        ),
        grip=float(np.asarray(info["proprio/gripper_opening"]).reshape(-1)[0]),
        contact=float(np.asarray(info["proprio/gripper_contact"]).reshape(-1)[0]),
        qpos=np.asarray(info["qpos"], np.float64).copy(),
        qvel=np.asarray(info["qvel"], np.float64).copy(),
    )


# Phase ladder


def cube_ladder(eff, cube_c, grip, goal_c, thr: PhaseThresholds) -> Dict[str, Any]:
    """Reach/secure milestones of one cube over one episode, with the thresholds of phases.classify_phases."""
    d = np.linalg.norm(eff - cube_c, axis=1)
    dxy = np.linalg.norm(eff[:, :2] - cube_c[:, :2], axis=1)
    lift = cube_c[:, 2] - TABLE_Z
    reached = bool(d.min() < thr.eps_reach)
    sec_mask = (lift > thr.delta_lift) & (grip > thr.tau_grip) & (dxy < thr.eps_reach)
    secured = bool(_max_true_run(sec_mask) >= thr.k_steps)
    final_goal_dist = float(np.linalg.norm(cube_c[-1] - goal_c))
    return dict(
        reached=reached,
        secured=secured,
        final_lift=float(lift[-1]),
        final_goal_dist=final_goal_dist,
        min_dist=float(d.min()),
        max_lift=float(lift.max()),
        delivered=bool(final_goal_dist <= 0.04),
    )


def cube_outcome(l: Dict[str, Any], thr: PhaseThresholds) -> str:
    """First unmet phase for an undelivered cube."""
    if l["final_goal_dist"] <= EPS_NEAR:
        return "transport"  # placed near the goal but outside 4 cm
    if l["final_lift"] >= thr.delta_lift and l["secured"]:
        return "transport"  # still held at the end, away from the goal
    if l["secured"]:
        return "maintain"  # acquired, then dropped
    if l["reached"]:
        return "acquisition"
    return "approach"


def episode_outcome(ep: Dict[str, Any], thr: PhaseThresholds) -> Dict[str, Any]:
    """One outcome per episode: success, or the furthest phase reached on an undelivered cube."""
    ladders = []
    for c in np.flatnonzero(ep["moving"]):
        l = cube_ladder(ep["eff"], ep["cube"][:, c], ep["grip"], ep["goal"][c], thr)
        l["cube"] = int(c)
        ladders.append(l)
    if ep["success"]:
        return dict(outcome="success", ladders=ladders)
    undelivered = [l for l in ladders if not l["delivered"]]
    if not undelivered:
        # All moving cubes near their goals but no env success flag
        return dict(outcome="transport", ladders=ladders)
    outs = [cube_outcome(l, thr) for l in undelivered]
    best = max(outs, key=lambda o: RANK[o])
    return dict(outcome=best, ladders=ladders)


# Stage 1: rollouts


def make_env_pool(env_name: str, n: int) -> List[Any]:
    return [R.make_env(env_name, seed=1000 + i) for i in range(n)]


def rollout_task(
    cond_agent, envs, task_id: int, n_episodes: int, rng_key
) -> List[Dict[str, Any]]:
    """Roll `n_episodes` episodes of one task in lockstep with the inferred-z agent; return trajectories."""
    from impls.utils.evaluation import supply_rng

    actor_fn = supply_rng(cond_agent.sample_actions, rng=rng_key)
    eps = []
    obs = []
    for m in range(n_episodes):
        env = envs[m]
        ob, _info = env.reset(options=dict(task_id=task_id, render_goal=False))
        s = signals_now(env)
        goals = cube_goals(env)
        init = s["cube"].copy()
        moving = np.linalg.norm(goals - init, axis=1) > MOVING_MIN
        eps.append(
            dict(
                task=task_id,
                episode=m,
                goal=goals,
                init_cube=init,
                moving=moving,
                obs=[np.asarray(ob, np.float32)],
                eff=[s["eff"]],
                cube=[s["cube"]],
                grip=[s["grip"]],
                contact=[s["contact"]],
                qpos=[s["qpos"]],
                qvel=[s["qvel"]],
                actions=[],
                success=False,
                alive=True,
                length=0,
            )
        )
        obs.append(np.asarray(ob, np.float32))
    obs = np.stack(obs)
    while any(e["alive"] for e in eps):
        acts = np.clip(
            np.asarray(actor_fn(observations=obs, goals=None, temperature=0.0)), -1, 1
        )
        for m, e in enumerate(eps):
            if not e["alive"]:
                continue
            a = acts[m].astype(np.float32)
            ob, _r, term, trunc, info = envs[m].step(a)
            obs[m] = np.asarray(ob, np.float32)
            s = signals_now(envs[m])
            e["obs"].append(obs[m].copy())
            e["actions"].append(a)
            for k in ("eff", "cube", "grip", "contact", "qpos", "qvel"):
                e[k].append(s[k])
            e["success"] = e["success"] or bool(info.get("success", False))
            e["length"] += 1
            if term or trunc:
                e["alive"] = False
    for e in eps:
        for k in ("obs", "eff", "cube", "grip", "contact", "qpos", "qvel", "actions"):
            e[k] = np.asarray(e[k])
        del e["alive"]
    return eps


def stage_rollouts(
    env_name: str, run: Dict, n_episodes: int, out_dir: Path, tag: str = ""
) -> Path:
    import pandas as pd

    variant = A.load_variant_eval(run["run_dir"], run["epoch"])
    R.ensure_cube_env(variant.env)
    tasks = A.eval_tasks(variant)
    conds = A.relabel_cond_agents(variant, tasks)
    thr = PhaseThresholds()
    envs = make_env_pool(env_name, n_episodes)
    label = f"{variant.critic}_seed{variant.seed}_{variant.epoch}{tag}"
    print(f"[rollouts] {label}: {len(tasks)} tasks x {n_episodes} episodes", flush=True)
    rows, store = [], {}
    t0 = time.time()
    for tid in tasks:
        import jax

        key = jax.random.PRNGKey(ROLLOUT_SEED * 1000 + int(tid) * 10 + variant.seed)
        eps = rollout_task(conds[tid], envs, tid, n_episodes, key)
        for e in eps:
            o = episode_outcome(e, thr)
            lad = o["ladders"]
            rows.append(
                dict(
                    env=env_name,
                    critic=variant.critic,
                    method=METHOD_LABEL[METHOD_OF[variant.critic]],
                    seed=variant.seed,
                    epoch=variant.epoch,
                    task=tid,
                    episode=e["episode"],
                    outcome=o["outcome"],
                    success=int(e["success"]),
                    length=int(e["length"]),
                    n_moving=int(e["moving"].sum()),
                    reached_all=int(all(l["reached"] for l in lad)) if lad else -1,
                    secured_all=int(all(l["secured"] for l in lad)) if lad else -1,
                    delivered_all=int(all(l["delivered"] for l in lad)) if lad else -1,
                    **{
                        f"relabel_{k}": v
                        for k, v in conds[tid].relabel_stats.items()
                        if k != "task"
                    },
                )
            )
            p = f"t{tid}_e{e['episode']}"
            for k in (
                "obs",
                "eff",
                "cube",
                "grip",
                "contact",
                "qpos",
                "qvel",
                "actions",
                "goal",
                "init_cube",
                "moving",
            ):
                store[f"{p}_{k}"] = np.asarray(e[k])
            store[f"{p}_success"] = np.asarray(int(e["success"]))
        sr = np.mean([r["success"] for r in rows if r["task"] == tid])
        print(
            f"  [{label}] task {tid}: success {sr:.2f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"traj_{label}.npz", **store)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"episodes_{label}.csv", index=False)
    frac = df.outcome.value_counts(normalize=True).reindex(OUTCOMES).fillna(0)
    print(
        f"[rollouts] {label}: " + " ".join(f"{o}={frac[o]:.2f}" for o in OUTCOMES),
        flush=True,
    )
    return out_dir / f"traj_{label}.npz"


# Stage 2: banks


def _persist(mask: np.ndarray, k: int) -> np.ndarray:
    """True at t when mask holds at t..t+k-1."""
    out = mask.copy()
    for j in range(1, k):
        sh = np.zeros_like(mask)
        sh[:-j] = mask[j:]
        out &= sh
    return out


def episode_candidates(
    e: Dict[str, np.ndarray], thr: PhaseThresholds
) -> Dict[str, List[tuple]]:
    """(t, cube) candidates of one episode for the three banks."""
    eff, cube, grip, contact = e["eff"], e["cube"], e["grip"], e["contact"]
    goal, moving = e["goal"], e["moving"].astype(bool)
    T1, C = cube.shape[0], cube.shape[1]
    lift = cube[:, :, 2] - TABLE_Z  # [T1, C]
    dist = np.linalg.norm(eff[:, None, :] - cube, axis=-1)  # [T1, C]
    dxy = np.linalg.norm(eff[:, None, :2] - cube[:, :, :2], axis=-1)
    dgoal = np.linalg.norm(cube[:, :, :2] - goal[None, :, :2], axis=-1)
    held = (
        (lift > thr.delta_lift)
        & (grip[:, None] > thr.tau_grip)
        & (contact[:, None] > BK.CONTACT_THRESH)
        & (dxy < thr.eps_reach)
    )
    delivered = np.linalg.norm(cube - goal[None], axis=-1) <= 0.04
    out = dict(approach=[], inhand=[], carrying=[])
    first_acq_any = T1
    per_cube = {}
    for c in range(C):
        if not moving[c]:
            continue
        hp = _persist(held[:, c], thr.k_steps)
        acq = np.flatnonzero(hp)
        first_acq = int(acq[0]) if len(acq) else T1
        first_acq_any = min(first_acq_any, first_acq)
        # Carrying: held over the last CARRY_WINDOW steps and goal distance fell
        carry = np.zeros(T1, bool)
        for t in range(CARRY_WINDOW, T1):
            if (
                hp[t]
                and held[t - CARRY_WINDOW : t + 1, c].all()
                and dgoal[t, c] < dgoal[t - CARRY_WINDOW, c] - CARRY_MIN_DROP
                and not delivered[t, c]
            ):
                carry[t] = True
        fc = np.flatnonzero(carry)
        first_carry = int(fc[0]) if len(fc) else T1
        inhand = hp & (np.arange(T1) < first_carry)
        per_cube[c] = dict(first_acq=first_acq, carry=carry, inhand=inhand)
        out["carrying"] += [(int(t), c) for t in np.flatnonzero(carry)]
        out["inhand"] += [(int(t), c) for t in np.flatnonzero(inhand)]
    # Approach: before any acquisition, nearest moving undelivered cube
    for t in range(1, min(first_acq_any, T1 - 1)):
        cands = [c for c in range(C) if moving[c] and not delivered[t, c]]
        if not cands or contact[t] > BK.CONTACT_THRESH:
            continue
        c = min(cands, key=lambda c: dist[t, c])
        if dist[t, c] > thr.eps_reach and lift[t, c] < 0.01:
            out["approach"].append((int(t), c))
    return out


def restore_full(env, rec: Dict[str, Any]) -> np.ndarray:
    """Restore a recorded state and its cube goals; settle the gripper closed if held, else open."""
    env.reset(options=dict(task_id=int(rec["task"]), render_goal=False))
    set_cube_goals(env, np.asarray(rec["goal"]))
    R.set_state(env, rec["qpos"], rec["qvel"])
    R.settle(env, a_grip=+1.0 if rec["grip"] > 0.5 else -1.0)
    return R.observation(env)


def fidelity_gate(
    env, rec: Dict[str, Any], bank: str, thr: PhaseThresholds
) -> Dict[str, Any]:
    """Check a restored state: restore.g2_check for held banks; small cube/hand/gripper drift for approach."""
    c = int(rec["cube"])
    s0 = signals_now(env)
    contact = s0["contact"]
    env.step(np.zeros(env.action_space.shape, np.float32))
    R.forward(env)
    s = signals_now(env)
    drift = float(np.linalg.norm(s["cube"][c] - rec["cube_pos"][c]))
    eff_drift = float(np.linalg.norm(s["eff"] - rec["eff"]))
    lift = float(s["cube"][c, 2] - TABLE_Z)
    if bank in ("inhand", "carrying"):
        held = lift > thr.delta_lift and s["grip"] > thr.tau_grip
        ok = held and contact > R.G2_CONTACT_MIN and drift < R.G2_DRIFT_M
    else:
        ok = (
            drift < R.G2_DRIFT_M
            and lift < thr.delta_lift
            and eff_drift < R.G2_DRIFT_M
            and abs(s["grip"] - rec["grip"]) < 0.2
        )
    return dict(
        passes=bool(ok),
        drift_m=drift,
        eff_drift_m=eff_drift,
        lift=lift,
        grip=float(s["grip"]),
        contact=float(contact),
    )


def stage_banks(
    env_name: str,
    traj_files: Sequence[Path],
    out_dir: Path,
    n_bank: int = N_BANK,
    tag: str = "",
) -> Dict[str, Any]:
    import pandas as pd

    thr = PhaseThresholds()
    cands = dict(approach=[], inhand=[], carrying=[])
    n_eps = 0
    for f in traj_files:
        d = np.load(f)
        name = f.stem[len("traj_") :]
        critic = name.split("_seed")[0]
        seed = int(name.split("_seed")[1].split("_")[0])
        keys = sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_obs")})
        for p in keys:
            e = {
                k: d[f"{p}_{k}"]
                for k in (
                    "obs",
                    "eff",
                    "cube",
                    "grip",
                    "contact",
                    "qpos",
                    "qvel",
                    "goal",
                    "init_cube",
                    "moving",
                )
            }
            tid, epi = int(p[1:].split("_e")[0]), int(p.split("_e")[1])
            n_eps += 1
            cc = episode_candidates(e, thr)
            for bank, lst in cc.items():
                for t, c in lst:
                    cands[bank].append(
                        dict(
                            file=str(f),
                            critic=critic,
                            seed=seed,
                            task=tid,
                            episode=epi,
                            step=t,
                            cube=c,
                        )
                    )
    print(
        f"[banks] {env_name}: {n_eps} episodes; candidates "
        + " ".join(f"{b}={len(v)}" for b, v in cands.items()),
        flush=True,
    )
    rng = np.random.default_rng(BANK_SEED)
    env = R.make_env(env_name, seed=7)
    R.ensure_cube_env(env)
    banks, gates, stats = {}, [], {}
    cache = {}
    for bank in ("approach", "inhand", "carrying"):
        cl = cands[bank]
        if not cl:
            stats[bank] = dict(candidates=0, banked=0)
            banks[bank] = []
            continue
        # One state per episode where possible
        groups: Dict[tuple, list] = {}
        for x in cl:
            groups.setdefault((x["file"], x["task"], x["episode"]), []).append(x)
        gkeys = list(groups)
        order = rng.permutation(len(gkeys))
        draws = []
        rounds = 0
        while len(draws) < N_CANDIDATES and rounds < 8:
            for gi in order:
                g = groups[gkeys[gi]]
                draws.append(g[rng.integers(len(g))])
                if len(draws) >= N_CANDIDATES:
                    break
            rounds += 1
        kept, n_tried = [], 0
        for x in draws:
            if x["file"] not in cache:
                cache[x["file"]] = np.load(x["file"])
            d = cache[x["file"]]
            p = f"t{x['task']}_e{x['episode']}"
            t, c = x["step"], x["cube"]
            rec = dict(
                task=x["task"],
                goal=d[f"{p}_goal"],
                qpos=d[f"{p}_qpos"][t],
                qvel=d[f"{p}_qvel"][t],
                obs=d[f"{p}_obs"][t],
                grip=float(d[f"{p}_grip"][t]),
                contact=float(d[f"{p}_contact"][t]),
                eff=d[f"{p}_eff"][t],
                cube_pos=d[f"{p}_cube"][t],
                cube=c,
                critic=x["critic"],
                seed=x["seed"],
                episode=x["episode"],
                step=t,
                moving=d[f"{p}_moving"],
            )
            ob = restore_full(env, rec)
            g = fidelity_gate(env, rec, bank, thr)
            n_tried += 1
            gates.append(
                dict(
                    env=env_name,
                    bank=bank,
                    **{
                        k: rec[k]
                        for k in ("critic", "seed", "task", "episode", "step", "cube")
                    },
                    **g,
                )
            )
            if g["passes"]:
                rec["obs_restored"] = ob
                kept.append(rec)
            if len(kept) >= n_bank:
                break
        banks[bank] = kept
        stats[bank] = dict(
            candidates=len(cl),
            episodes=len(gkeys),
            tried=n_tried,
            banked=len(kept),
            gate_pass=float(len(kept) / max(1, n_tried)),
            from_fb=int(sum(r["critic"] == RUN_NAME["fb"] for r in kept)),
            from_density=int(
                sum(r["critic"] == RUN_NAME["density_fb_z"] for r in kept)
            ),
        )
        print(f"[banks] {env_name} {bank}: {stats[bank]}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = {}
    for bank, kept in banks.items():
        if not kept:
            continue
        for k in (
            "qpos",
            "qvel",
            "obs",
            "obs_restored",
            "goal",
            "eff",
            "cube_pos",
            "moving",
        ):
            store[f"{bank}_{k}"] = np.stack([np.asarray(r[k]) for r in kept])
        for k in ("task", "cube", "seed", "episode", "step"):
            store[f"{bank}_{k}"] = np.asarray([r[k] for r in kept])
        store[f"{bank}_grip"] = np.asarray([r["grip"] for r in kept])
        store[f"{bank}_contact"] = np.asarray([r["contact"] for r in kept])
        store[f"{bank}_critic"] = np.asarray([r["critic"] for r in kept])
    np.savez(out_dir / f"banks{tag}.npz", **store)
    pd.DataFrame(gates).to_csv(out_dir / f"bank_gate{tag}.csv", index=False)
    (out_dir / f"bank_stats{tag}.json").write_text(json.dumps(stats, indent=2))
    return stats


# Stage 3: predictions vs rollout truth


def build_pool(env_name: str, out_dir: Path) -> Dict[str, np.ndarray]:
    """Sample 1024 dataset next-states, fixed by POOL_SEED."""
    f = out_dir / "pool.npz"
    if f.exists():
        return dict(np.load(f))
    raw = BK.load_raw(env_name)
    rng = np.random.default_rng(POOL_SEED)
    n = len(raw["observations"])
    # a next-state is row i+1 of the same episode
    ok = np.flatnonzero(~raw["terminals"][:-1])
    sel = np.sort(rng.choice(ok, size=POOL_SIZE, replace=False)) + 1
    pool = dict(row=sel.astype(np.int64), obs=raw["observations"][sel])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(f, **pool)
    return pool


def hist_xy(xy: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """[GRID_N, GRID_N] normalised histogram on the workspace grid; outside points go to the border."""
    x = np.clip(xy[..., 0].ravel(), GRID_X[0], GRID_X[1] - 1e-9)
    y = np.clip(xy[..., 1].ravel(), GRID_Y[0], GRID_Y[1] - 1e-9)
    h, _, _ = np.histogram2d(
        x,
        y,
        bins=GRID_N,
        range=[GRID_X, GRID_Y],
        weights=None if w is None else w.ravel(),
    )
    s = h.sum()
    return h / s if s > 0 else np.full_like(h, 1.0 / h.size)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def predict_density(variant, obs_rep, acts, z, pool_obs) -> np.ndarray:
    """Density-FB weights w = mean_h softmax_j F(s,a,z).B(u_j)/tau over the pool."""
    import jax
    import jax.numpy as jnp

    critic, net = variant.agent.critic, variant.agent.net
    f_p, b_p = net["density_f"].params, net["density_b"].params
    pool = jnp.asarray(pool_obs)
    b_emb = critic._B(b_p, critic._b_query(pool))  # [M, d]
    g = jnp.broadcast_to(jnp.asarray(z), (len(acts), jnp.asarray(z).shape[-1]))
    f_emb = critic._F(f_p, jnp.asarray(obs_rep), jnp.asarray(acts), g)
    w = jax.nn.softmax(jnp.einsum("hbd,md->hbm", f_emb, b_emb) / critic.temperature, -1)
    return np.asarray(jnp.mean(w, axis=0))  # [B, M]


def predict_fb(variant, obs_rep, acts, z, pool_obs):
    """FB measure min_h F(s,a,z).B(u), clipped at zero and normalised; returns ([B, M], negative fraction)."""
    from scripts.probes.cube_bottleneck.measure_probe import fb_readout

    M, _q = fb_readout(
        variant.agent.critic, variant.agent.net, obs_rep, acts, z, pool_obs
    )
    neg = float(
        (np.minimum(M, 0).sum(-1) / np.maximum(np.abs(M).sum(-1), 1e-12)).mean() * -1
    )
    pos = np.maximum(M, 0.0)
    den = pos.sum(-1, keepdims=True)
    w = np.where(den > 0, pos / np.maximum(den, 1e-12), 1.0 / M.shape[-1])
    return w, neg


def truth_batch(cond_agent, envs, recs: List[Dict[str, Any]], discount: float, rng_key):
    """Roll K_CONT policy continuations from each state; return per-state cube xy, success flag, and discount weights."""
    from impls.utils.evaluation import supply_rng

    actor_fn = supply_rng(cond_agent.sample_actions, rng=rng_key)
    n = len(recs) * K_CONT
    assert len(envs) >= n
    obs, cidx = [], []
    for i, rec in enumerate(recs):
        for k in range(K_CONT):
            e = envs[i * K_CONT + k]
            obs.append(restore_full(e, rec))
            cidx.append(int(rec["cube"]))
    obs = np.stack(obs).astype(np.float32)
    alive = np.ones(n, bool)
    succ = np.zeros(n, bool)
    xy = np.zeros((HORIZON, n, 2), np.float32)
    for t in range(HORIZON):
        acts = np.clip(
            np.asarray(actor_fn(observations=obs, goals=None, temperature=0.0)), -1, 1
        )
        for j in range(n):
            if alive[j]:
                o, _r, term, trunc, info = envs[j].step(acts[j].astype(np.float32))
                obs[j] = np.asarray(o, np.float32)
                succ[j] = succ[j] or bool(info.get("success", False))
                if term or trunc:
                    alive[j] = False
            xy[t, j] = cube_xy_from_obs(obs[j], cidx[j])
    w = discount ** np.arange(HORIZON, dtype=np.float64)
    w = w / w.sum()
    out = []
    for i in range(len(recs)):
        sl = slice(i * K_CONT, (i + 1) * K_CONT)
        out.append((xy[:, sl], float(succ[sl].mean())))
    return out, w


def stage_predict(
    env_name: str,
    run: Dict,
    bank_file: Path,
    bank_names: Sequence[str],
    out_dir: Path,
    tag: str = "",
    max_states: int = 0,
) -> Path:
    import jax
    import pandas as pd

    variant = A.load_variant_eval(run["run_dir"], run["epoch"])
    R.ensure_cube_env(variant.env)
    tasks = A.eval_tasks(variant)
    conds = A.relabel_cond_agents(variant, tasks)
    disc = float(variant.config.get("discount", 0.99))
    is_density = variant.critic.startswith("density_fb")
    label = f"{variant.critic}_seed{variant.seed}_{variant.epoch}{tag}"
    pool = build_pool(env_name, out_dir)
    pool_obs = pool["obs"]
    bk = np.load(bank_file)
    envs = make_env_pool(env_name, STATES_PER_BATCH * K_CONT)
    prior = {c: hist_xy(cube_xy_from_obs(pool_obs, c)) for c in range(n_cubes(envs[0]))}
    rows, maps = [], {}
    t0 = time.time()
    for bank in bank_names:
        if f"{bank}_qpos" not in bk:
            print(f"[predict] {label}: bank {bank} empty, skipped", flush=True)
            continue
        n = (
            len(bk[f"{bank}_qpos"])
            if not max_states
            else min(max_states, len(bk[f"{bank}_qpos"]))
        )
        recs = []
        for i in range(n):
            recs.append(
                dict(
                    idx=i,
                    task=int(bk[f"{bank}_task"][i]),
                    goal=bk[f"{bank}_goal"][i],
                    qpos=bk[f"{bank}_qpos"][i],
                    qvel=bk[f"{bank}_qvel"][i],
                    grip=float(bk[f"{bank}_grip"][i]),
                    cube=int(bk[f"{bank}_cube"][i]),
                    source_critic=str(bk[f"{bank}_critic"][i]),
                    source_seed=int(bk[f"{bank}_seed"][i]),
                )
            )
        # Group by task: one inferred z per batch
        by_task: Dict[int, list] = {}
        for r in recs:
            by_task.setdefault(r["task"], []).append(r)
        for tid, rl in by_task.items():
            ag = conds[tid]
            z = np.asarray(ag.cond_eval["z"])
            for s in range(0, len(rl), STATES_PER_BATCH):
                chunk = rl[s : s + STATES_PER_BATCH]
                key = jax.random.PRNGKey(
                    ROLLOUT_SEED * 7919 + tid * 101 + s + variant.seed * 13
                )
                truths, w = truth_batch(ag, envs, chunk, disc, key)
                for rec, (xy_t, succ) in zip(chunk, truths):
                    c = rec["cube"]
                    goal_xy = np.asarray(rec["goal"])[c, :2]
                    # Prediction at the restored observation, averaged over policy draws
                    ob = restore_full(envs[0], rec)
                    obs_rep = np.repeat(ob[None], N_PRED_ACTIONS, axis=0)
                    acts = np.asarray(
                        ag.sample_actions(
                            obs_rep,
                            goals=None,
                            seed=jax.random.PRNGKey(rec["idx"] + 31 * variant.seed),
                        )
                    )
                    if is_density:
                        wp = predict_density(variant, obs_rep, acts, z, pool_obs)
                        neg = 0.0
                    else:
                        wp, neg = predict_fb(variant, obs_rep, acts, z, pool_obs)
                    wp = wp.mean(0)  # [M]
                    pool_xy = cube_xy_from_obs(pool_obs, c)
                    p_pred = hist_xy(pool_xy, wp)
                    wt = np.broadcast_to(w[:, None] / K_CONT, xy_t.shape[:2])
                    p_true = hist_xy(xy_t, wt)
                    goal_mass_pred = float(
                        wp[
                            np.linalg.norm(pool_xy - goal_xy, axis=1) < GOAL_RADIUS
                        ].sum()
                    )
                    goal_mass_true = float(
                        wt[np.linalg.norm(xy_t - goal_xy, axis=-1) < GOAL_RADIUS].sum()
                    )
                    goal_mass_prior = float(
                        (np.linalg.norm(pool_xy - goal_xy, axis=1) < GOAL_RADIUS).mean()
                    )
                    ent = float(-(wp * np.log(wp + 1e-12)).sum())
                    rows.append(
                        dict(
                            env=env_name,
                            critic=variant.critic,
                            method=METHOD_LABEL[METHOD_OF[variant.critic]],
                            seed=variant.seed,
                            epoch=variant.epoch,
                            bank=bank,
                            state=rec["idx"],
                            task=tid,
                            cube=c,
                            source_critic=rec["source_critic"],
                            source_seed=rec["source_seed"],
                            tv=tv(p_pred, p_true),
                            tv_pred_prior=tv(p_pred, prior[c]),
                            tv_truth_prior=tv(p_true, prior[c]),
                            goal_mass_pred=goal_mass_pred,
                            goal_mass_truth=goal_mass_true,
                            goal_mass_prior=goal_mass_prior,
                            pred_entropy=ent,
                            fb_neg_frac=neg,
                            truth_success=succ,
                            truth_goal_dist_final=float(
                                np.linalg.norm(xy_t[-1] - goal_xy, axis=-1).mean()
                            ),
                        )
                    )
                    maps[f"{bank}_{rec['idx']}_pred"] = p_pred.astype(np.float32)
                    maps[f"{bank}_{rec['idx']}_true"] = p_true.astype(np.float32)
                print(
                    f"  [{label}] {bank} task {tid}: {min(s + STATES_PER_BATCH, len(rl))}/{len(rl)} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        sub = [r for r in rows if r["bank"] == bank]
        print(
            f"[predict] {label} {bank}: TV {np.mean([r['tv'] for r in sub]):.3f} "
            f"goal-mass pred {np.mean([r['goal_mass_pred'] for r in sub]):.3f} "
            f"truth {np.mean([r['goal_mass_truth'] for r in sub]):.3f}",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"pred_{label}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    np.savez_compressed(out_dir / f"maps_{label}.npz", **maps)
    return f


# Stage 4: action probe. Both critics score the same 12 candidate actions per state against
# the Monte-Carlo goal-reaching return of each method's own policy.

N_PERTURB = 4
PERTURB_SIGMA = 0.3
SUCCESS_RADIUS = 0.04


def candidate_actions(a_fb, a_d, adim, rng):
    secure = np.zeros(adim, np.float32)
    secure[-1] = +1.0
    release = np.zeros(adim, np.float32)
    release[-1] = -1.0
    acts = [a_fb, a_d, secure, release]
    for base in (a_fb, a_d):
        for _ in range(N_PERTURB):
            acts.append(
                np.clip(base + rng.normal(0, PERTURB_SIGMA, size=adim), -1, 1).astype(
                    np.float32
                )
            )
    return np.stack(acts)


CANDIDATE_NAMES = (
    ["fb_policy", "density_policy", "secure", "release"]
    + [f"fb_perturb{i}" for i in range(N_PERTURB)]
    + [f"density_perturb{i}" for i in range(N_PERTURB)]
)


def q_values(variant, obs_rep, acts, z, pool_obs) -> np.ndarray:
    """Critic value q(s, a, z) used by the actor, for a batch of actions at one state."""
    import jax
    import jax.numpy as jnp

    critic, net = variant.agent.critic, variant.agent.net
    params = {k: v.params for k, v in net.items()}
    zb = jnp.broadcast_to(jnp.asarray(z), (len(acts), jnp.asarray(z).shape[-1]))
    if variant.critic.startswith("density_fb"):
        from impls.critics.density_fb import _raw_block_cols

        pool = jnp.asarray(pool_obs)
        u_raw = _raw_block_cols(pool, critic.blocks)
        u_sc = critic._b_query(pool)
        st = critic._zstats(net)
        cond = {
            "z": zb,
            "_density_u_raw": u_raw,
            "_density_u_sc": u_sc,
            "_density_zenc_u": jax.lax.stop_gradient(
                critic._B(critic._zenc(net), u_sc) - st["mu"][None]
            ),
        }
    else:
        cond = {"z": zb}
    q, _qs = critic.q_fn(params, jnp.asarray(obs_rep), cond, jnp.asarray(acts))
    return np.asarray(q)


def truth_actions(cond_agent, envs, rec, acts, cidx, goal_xyz, discount, rng_key):
    """Apply each candidate once, then K_CONT policy continuations; return cube xy and discounted return."""
    from impls.utils.evaluation import supply_rng

    actor_fn = supply_rng(cond_agent.sample_actions, rng=rng_key)
    n_acts = len(acts)
    n = n_acts * K_CONT
    assert len(envs) >= n
    obs = np.zeros((n, envs[0].observation_space.shape[0]), np.float32)
    alive = np.ones(n, bool)
    hit = np.zeros((HORIZON, n), np.float32)
    xy = np.zeros((HORIZON, n, 2), np.float32)
    for i in range(n_acts):
        for k in range(K_CONT):
            j = i * K_CONT + k
            e = envs[j]
            restore_full(e, rec)
            o, _r, term, trunc, _info = e.step(np.asarray(acts[i], np.float32))
            obs[j] = np.asarray(o, np.float32)
            if term or trunc:
                alive[j] = False

    def _record(t):
        for j in range(n):
            u = envs[j].unwrapped
            pos = np.asarray(u._data.joint(f"object_joint_{cidx}").qpos[:3])
            xy[t, j] = pos[:2]
            hit[t, j] = float(np.linalg.norm(pos - goal_xyz) <= SUCCESS_RADIUS)

    _record(0)
    for t in range(1, HORIZON):
        a = np.clip(
            np.asarray(actor_fn(observations=obs, goals=None, temperature=0.0)), -1, 1
        )
        for j in range(n):
            if alive[j]:
                o, _r, term, trunc, _info = envs[j].step(a[j].astype(np.float32))
                obs[j] = np.asarray(o, np.float32)
                if term or trunc:
                    alive[j] = False
        _record(t)
    w = discount ** np.arange(HORIZON, dtype=np.float64)
    ret = (w[:, None] * hit).sum(0)  # [n]
    xy = xy.reshape(HORIZON, n_acts, K_CONT, 2).transpose(1, 0, 2, 3)
    return xy, ret.reshape(n_acts, K_CONT).mean(-1), w / w.sum()


def _spearman(x, y):
    import pandas as pd

    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def stage_actionprobe(
    env_name: str,
    seed: int,
    bank_file: Path,
    bank_names,
    out_dir: Path,
    epoch: int = 1000000,
    tag: str = "",
    max_states: int = 0,
) -> Path:
    import jax
    import pandas as pd

    run_names = [RUN_NAME[m] for m in METHODS]
    runs = {
        METHOD_OF[r["critic"]]: r
        for r in A.discover_roots(env_name, run_names, [seed], epoch)
    }
    assert set(runs) == set(METHODS), f"seed {seed}: found {sorted(runs)}"
    variants = {m: A.load_variant_eval(runs[m]["run_dir"], epoch) for m in METHODS}
    tasks = A.eval_tasks(variants["fb"])
    conds = {m: A.relabel_cond_agents(variants[m], tasks) for m in METHODS}
    disc = float(variants["fb"].config.get("discount", 0.99))
    pool_obs = build_pool(env_name, out_dir)["obs"]
    bk = np.load(bank_file)
    n_acts = 4 + 2 * N_PERTURB
    envs = make_env_pool(env_name, n_acts * K_CONT)
    adim = int(envs[0].action_space.shape[0])
    label = f"seed{seed}_{epoch}{tag}"
    rows, long_rows = [], []
    t0 = time.time()
    for bank in bank_names:
        if f"{bank}_qpos" not in bk:
            continue
        n = (
            len(bk[f"{bank}_qpos"])
            if not max_states
            else min(max_states, len(bk[f"{bank}_qpos"]))
        )
        for i in range(n):
            rec = dict(
                idx=i,
                task=int(bk[f"{bank}_task"][i]),
                goal=bk[f"{bank}_goal"][i],
                qpos=bk[f"{bank}_qpos"][i],
                qvel=bk[f"{bank}_qvel"][i],
                grip=float(bk[f"{bank}_grip"][i]),
                cube=int(bk[f"{bank}_cube"][i]),
            )
            c, tid = rec["cube"], rec["task"]
            goal_xyz = np.asarray(rec["goal"])[c]
            ob = restore_full(envs[0], rec)
            key = jax.random.PRNGKey(4242 + i * 7 + seed)
            a_fb = np.asarray(conds["fb"][tid].sample_actions(ob, goals=None, seed=key))
            a_d = np.asarray(
                conds["density_fb_z"][tid].sample_actions(ob, goals=None, seed=key)
            )
            rng = np.random.default_rng(
                [BANK_SEED, seed, i, {"approach": 0, "inhand": 1, "carrying": 2}[bank]]
            )
            acts = candidate_actions(
                a_fb.astype(np.float32), a_d.astype(np.float32), adim, rng
            )
            obs_rep = np.repeat(ob[None], n_acts, axis=0)
            pool_xy = cube_xy_from_obs(pool_obs, c)
            for m in METHODS:
                variant, ag = variants[m], conds[m][tid]
                z = np.asarray(ag.cond_eval["z"])
                q = q_values(variant, obs_rep, acts, z, pool_obs)
                if m.startswith("density_fb"):
                    wp = predict_density(variant, obs_rep, acts, z, pool_obs)
                else:
                    wp, _neg = predict_fb(variant, obs_rep, acts, z, pool_obs)
                xy_t, ret, w = truth_actions(
                    ag,
                    envs,
                    rec,
                    acts,
                    c,
                    goal_xyz,
                    disc,
                    jax.random.PRNGKey(99 + i * 13 + seed),
                )
                tvs = []
                for ai in range(n_acts):
                    p_pred = hist_xy(pool_xy, wp[ai])
                    wt = np.broadcast_to(w[:, None] / K_CONT, xy_t[ai].shape[:2])
                    p_true = hist_xy(xy_t[ai], wt)
                    tvs.append(tv(p_pred, p_true))
                    long_rows.append(
                        dict(
                            env=env_name,
                            seed=seed,
                            epoch=epoch,
                            bank=bank,
                            state=i,
                            task=tid,
                            critic=RUN_NAME[m],
                            method=METHOD_LABEL[m],
                            candidate=CANDIDATE_NAMES[ai],
                            q_pred=float(q[ai]),
                            return_true=float(ret[ai]),
                            tv=tvs[-1],
                        )
                    )
                tvs = np.asarray(tvs)
                ib, iw = int(np.argmax(ret)), int(np.argmin(ret))
                rng_q, rng_r = float(np.ptp(q)), float(np.ptp(ret))
                gap_err = (
                    abs((q[ib] - q[iw]) / rng_q - (ret[ib] - ret[iw]) / rng_r)
                    if rng_q > 1e-9 and rng_r > 1e-9
                    else float("nan")
                )
                regret = (
                    ((ret[ib] - ret[int(np.argmax(q))]) / rng_r)
                    if rng_r > 1e-9
                    else float("nan")
                )
                rows.append(
                    dict(
                        env=env_name,
                        seed=seed,
                        epoch=epoch,
                        bank=bank,
                        state=i,
                        task=tid,
                        cube=c,
                        critic=RUN_NAME[m],
                        method=METHOD_LABEL[m],
                        tv_action=float(tvs.mean()),
                        gap_error=gap_err,
                        rank_spearman=_spearman(q, ret),
                        regret=regret,
                        return_range=rng_r,
                        q_range=rng_q,
                        return_best=float(ret[ib]),
                        return_chosen=float(ret[int(np.argmax(q))]),
                        return_own_policy=float(ret[0 if m == "fb" else 1]),
                        return_secure=float(ret[2]),
                        return_release=float(ret[3]),
                        best_candidate=CANDIDATE_NAMES[ib],
                        chosen_candidate=CANDIDATE_NAMES[int(np.argmax(q))],
                    )
                )
            if (i + 1) % 4 == 0:
                sub = [r for r in rows if r["bank"] == bank]
                msg = " ".join(
                    f"{m}: rank {np.nanmean([r['rank_spearman'] for r in sub if r['critic'] == RUN_NAME[m]]):.2f} "
                    f"regret {np.nanmean([r['regret'] for r in sub if r['critic'] == RUN_NAME[m]]):.2f}"
                    for m in METHODS
                )
                print(
                    f"  [{label}] {bank} {i + 1}/{n} ({time.time() - t0:.0f}s) {msg}",
                    flush=True,
                )
            if (i + 1) % 16 == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(out_dir / f"actions_{label}.csv", index=False)
                pd.DataFrame(long_rows).to_csv(
                    out_dir / f"actions_long_{label}.csv", index=False
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"actions_{label}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    pd.DataFrame(long_rows).to_csv(out_dir / f"actions_long_{label}.csv", index=False)
    return f


# CLI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["rollouts", "banks", "predict", "actionprobe"])
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument(
        "--critics",
        nargs="+",
        default=METHODS,
        help="methods to probe; run-directory names (fb_tdj) are also accepted",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epoch", type=int, default=1000000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--banks", nargs="+", default=["approach", "inhand", "carrying"])
    ap.add_argument("--bank-file", default=None)
    ap.add_argument("--n-bank", type=int, default=N_BANK)
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument(
        "--exp-roots",
        nargs="+",
        default=list(A.DEFAULT_ROOTS),
        help="exp roots searched in order; the density_fb_z runs under exp/tdjtab share "
        "their name with older local runs under exp/, so pass exp/tdjtab "
        "alone to pin them",
    )
    args = ap.parse_args()
    args.critics = [RUN_NAME.get(c, c) for c in args.critics]  # run-directory names

    out = Path(args.out_root) / args.env
    if args.stage == "rollouts":
        runs = A.discover_roots(args.env, args.critics, args.seeds, args.epoch)
        if not runs:
            raise SystemExit("no runs found")
        for r in runs:
            stage_rollouts(args.env, r, args.episodes, out / "rollouts", tag=args.tag)
    elif args.stage == "banks":
        files = sorted((out / "rollouts").glob(f"traj_*_{args.epoch}{args.tag}.npz"))
        files = [
            f
            for f in files
            if any(f.name.startswith(f"traj_{c}_seed") for c in args.critics)
        ]
        if not files:
            raise SystemExit("no trajectory files")
        stage_banks(args.env, files, out / "banks", n_bank=args.n_bank, tag=args.tag)
    elif args.stage == "predict":
        runs = A.discover_roots(
            args.env, args.critics, args.seeds, args.epoch, roots=args.exp_roots
        )
        if not runs:
            raise SystemExit("no runs found")
        bank_file = (
            Path(args.bank_file) if args.bank_file else out / "banks" / "banks.npz"
        )
        for r in runs:
            stage_predict(
                args.env,
                r,
                bank_file,
                args.banks,
                out / "predict",
                tag=args.tag,
                max_states=args.max_states,
            )
    elif args.stage == "actionprobe":
        bank_file = (
            Path(args.bank_file) if args.bank_file else out / "banks" / "banks.npz"
        )
        for seed in args.seeds:
            stage_actionprobe(
                args.env,
                seed,
                bank_file,
                args.banks,
                out / "actions",
                epoch=args.epoch,
                tag=args.tag,
                max_states=args.max_states,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
