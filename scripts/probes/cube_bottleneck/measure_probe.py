"""Measure probe at P2 states: does predicted successor mass on held futures track p_secure(s, a)?

Pass A rolls out the BC actor to label p_secure per (state, action); Pass B reads both critics
on the same (state, action) pairs against a shared pool.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import banks as BK
from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.phases import PhaseThresholds

N_ACTIONS = 12
N_CONT = 8
HORIZON = 25
POOL_SIZE = 1024
POOL_SEED = 0
ACTION_SEED = 0
G4_STD_MIN = 0.20
G4_MIN_STATES = 40


# Shared pool


def build_pool(env_name: str, out_dir: Path) -> Dict[str, np.ndarray]:
    """Build and cache 1024 dataset states with phase labels, fixed by `POOL_SEED`."""
    f = out_dir / "pool.npz"
    if f.exists():
        return dict(np.load(f))
    raw = BK.load_raw(env_name)
    rng = np.random.default_rng(POOL_SEED)
    sel = np.sort(rng.choice(len(raw["observations"]), size=POOL_SIZE, replace=False))
    obs = raw["observations"][sel]
    held = (obs[:, BK.CONTACT_IDX] > BK.CONTACT_THRESH) & (
        raw["qpos"][sel][:, BK.CUBE_QPOS_Z] > BK.LIFT_THRESH
    )
    pool = dict(row=sel.astype(np.int64), obs=obs, held=held.astype(np.int8))
    np.savez(f, **pool)
    return pool


# Action set


def action_set(
    actor_fn,
    obs: np.ndarray,
    goal,
    dataset_action: np.ndarray,
    rng: np.random.Generator,
    action_dim: int,
) -> np.ndarray:
    """12 actions: dataset action, 6 policy draws, dataset action with gripper at each extreme, 3 uniform."""
    acts = [np.asarray(dataset_action, np.float32)]
    # Policy draws in one batched call
    draws = np.asarray(
        actor_fn(
            observations=np.repeat(np.asarray(obs)[None], 6, axis=0),
            goals=np.repeat(np.asarray(goal)[None], 6, axis=0),
            temperature=1.0,
        )
    )
    for d in draws:
        acts.append(np.clip(d, -1, 1).astype(np.float32))
    closed = np.asarray(dataset_action, np.float32).copy()
    closed[-1] = +1.0
    opened = np.asarray(dataset_action, np.float32).copy()
    opened[-1] = -1.0
    acts += [closed, opened]
    for _ in range(3):
        acts.append(rng.uniform(-1, 1, size=action_dim).astype(np.float32))
    out = np.stack(acts)
    assert out.shape == (N_ACTIONS, action_dim)
    return out


# Pass A: ground truth


def make_env_pool(env_name: str, n: int) -> List[Any]:
    """Make `n` envs to step in lockstep with one batched actor call."""
    return [R.make_env(env_name, seed=i) for i in range(n)]


def p_secure_for_state(actor_fn, envs, qpos, qvel, acts, goals, thr) -> np.ndarray:
    """Return p_secure per action: fraction of N_CONT continuations ending held.

    Rollouts that terminate early keep their final state.
    """
    n_acts = len(acts)
    assert len(envs) >= n_acts * N_CONT, "env pool too small for the action set"
    obs, goal_b, alive, slots = [], [], [], []
    for ai in range(n_acts):
        for k in range(N_CONT):
            m = ai * N_CONT + k
            task_id, goal = goals[k % len(goals)]
            e = envs[m]
            e.reset(options=dict(task_id=task_id, render_goal=False))
            R.set_state(e, qpos, qvel)
            R.settle(e)
            ob, _r, term, trunc, _i = e.step(
                np.clip(np.asarray(acts[ai], np.float32), -1, 1)
            )
            obs.append(np.asarray(ob, np.float32))
            goal_b.append(goal)
            alive.append(not (term or trunc))
            slots.append((ai, k, m))
    obs = np.stack(obs)
    goal_b = np.stack(goal_b)
    alive = np.asarray(alive)

    for _ in range(HORIZON - 1):
        if not alive.any():
            break
        a = np.clip(
            np.asarray(actor_fn(observations=obs, goals=goal_b, temperature=1.0)), -1, 1
        )
        for j, (_ai, _k, m) in enumerate(slots):
            if not alive[j]:
                continue
            ob, _r, term, trunc, _i = envs[m].step(a[j].astype(np.float32))
            obs[j] = np.asarray(ob, np.float32)
            if term or trunc:
                alive[j] = False

    out = np.zeros(n_acts, float)
    for j, (ai, _k, m) in enumerate(slots):
        e = envs[m]
        R.forward(e)
        u = e.unwrapped
        cube_z = float(u._data.qpos[R.CUBE_QPOS][2])
        table_z = float(
            u.cur_task_info["init_xyzs"][int(getattr(u, "_target_block", 0) or 0)][2]
        )
        grip = float(R.observation(e)[17] / R.GRIPPER_SCALER)
        out[ai] += int((cube_z - table_z) > thr.delta_lift and grip > thr.tau_grip)
    return out / N_CONT


def ground_truth(
    variant,
    banks: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    thr: PhaseThresholds,
    tag: str = "",
    state_slice: Optional[slice] = None,
) -> Dict[str, Any]:
    """Pass A over P2 states and the P1/P3 controls, for one slice of states.

    The action rng is seeded per state, so results do not depend on slicing.
    """
    import jax

    from impls.utils.evaluation import supply_rng

    part = out_dir / "gt_parts"
    part.mkdir(parents=True, exist_ok=True)
    if state_slice is None:
        f = out_dir / f"ground_truth{tag}.npz"
        if f.exists():
            return dict(np.load(f, allow_pickle=True))
    else:
        f = part / f"gt{tag}_{state_slice.start}_{state_slice.stop}.npz"
        if f.exists():
            return dict(np.load(f, allow_pickle=True))

    env, agent = variant.env, variant.agent
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(ACTION_SEED))
    pool_envs = make_env_pool(variant.env_name, N_ACTIONS * N_CONT)
    print(f"[gt] env pool: {len(pool_envs)} envs stepped in lockstep", flush=True)
    tasks = A.eval_tasks(variant)
    goals = []
    for t in tasks:
        _o, info = env.reset(options=dict(task_id=t, render_goal=False))
        goals.append((t, np.asarray(info["goal"], np.float32)))
    action_dim = int(env.action_space.shape[0])

    store = {}
    for phase, bank in banks.items():
        n_full = len(bank["qpos"])
        idxs = range(*state_slice.indices(n_full)) if state_slice else range(n_full)
        idxs = list(idxs)
        n = len(idxs)
        acts_all = np.zeros((n, N_ACTIONS, action_dim), np.float32)
        psec_all = np.zeros((n, N_ACTIONS), np.float32)
        for slot, i in enumerate(idxs):
            # Per-state rng; crc32 because str hash() is salted per process
            rng = np.random.default_rng([ACTION_SEED, zlib.crc32(phase.encode()), i])
            # Actions drawn at the restored observation
            env.reset(options=dict(task_id=goals[0][0], render_goal=False))
            R.set_state(env, bank["qpos"][i], bank["qvel"][i])
            R.settle(env)
            ob = R.observation(env)
            acts = action_set(
                actor_fn, ob, goals[0][1], bank["action"][i], rng, action_dim
            )
            acts_all[slot] = acts
            psec_all[slot] = p_secure_for_state(
                actor_fn, pool_envs, bank["qpos"][i], bank["qvel"][i], acts, goals, thr
            )
            if (slot + 1) % 4 == 0:
                print(f"    [{phase}] {slot + 1}/{n} states", flush=True)
        store[f"{phase}_actions"] = acts_all
        store[f"{phase}_p_secure"] = psec_all
        store[f"{phase}_index"] = np.asarray(idxs, np.int64)
        sd = psec_all.std(axis=1)
        print(
            f"  [{phase}] p_secure std across actions: median {np.median(sd):.3f}, "
            f"states with std>{G4_STD_MIN}: {int((sd > G4_STD_MIN).sum())}/{n}",
            flush=True,
        )
    store["_gt_arm"] = np.asarray(variant.label)
    np.savez(f, **store)
    return store


def merge_ground_truth(
    out_dir: Path, phases, sizes: Dict[str, int], tag: str = ""
) -> Dict[str, Any]:
    """Merge sliced Pass A parts into ground_truth.npz; raise if any state is missing."""
    parts = sorted((out_dir / "gt_parts").glob(f"gt{tag}_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no Pass A parts under {out_dir / 'gt_parts'}")
    store: Dict[str, Any] = {}
    for phase in phases:
        n = sizes[phase]
        acts = np.full((n, N_ACTIONS, 5), np.nan, np.float32)
        psec = np.full((n, N_ACTIONS), np.nan, np.float32)
        seen = np.zeros(n, bool)
        for f in parts:
            d = np.load(f, allow_pickle=True)
            if f"{phase}_index" not in d:
                continue
            idx = d[f"{phase}_index"]
            if acts.shape[-1] != d[f"{phase}_actions"].shape[-1]:
                acts = np.full(
                    (n, N_ACTIONS, d[f"{phase}_actions"].shape[-1]), np.nan, np.float32
                )
            acts[idx] = d[f"{phase}_actions"]
            psec[idx] = d[f"{phase}_p_secure"]
            seen[idx] = True
        if not seen.all():
            raise ValueError(
                f"Pass A incomplete for {phase}: missing states "
                f"{np.flatnonzero(~seen).tolist()}"
            )
        store[f"{phase}_actions"], store[f"{phase}_p_secure"] = acts, psec
    np.savez(out_dir / f"ground_truth{tag}.npz", **store)
    return store


# Pass B: critic readouts


def fb_readout(critic, net, obs, acts, z, pool_obs):
    """FB readout M(s,a,z;u) = F(s,a,z).B(u), min over heads."""
    import jax.numpy as jnp

    le = critic._le(net["left_encoder"].params, jnp.asarray(obs))
    # Broadcast z to the batch
    zb = jnp.broadcast_to(jnp.asarray(z), (len(acts), jnp.asarray(z).shape[-1]))
    Fs = critic._F(net["forward"].params, le, zb, jnp.asarray(acts))  # [P,B,d]
    Bm = critic._B(net["backward"].params, jnp.asarray(pool_obs))  # [C,d]
    M = jnp.einsum("pbz,cz->pbc", Fs, Bm)
    M = jnp.min(M, axis=0)  # [B,C]
    Q = jnp.min((Fs * zb).sum(-1), axis=0)  # [B]
    return np.asarray(M), np.asarray(Q)


def density_readout(critic, net, obs, acts, goal, pool_obs):
    """Density-FB readout w = softmax_j F(s,a,g).B(u_j)/tau, averaged over heads.

    Also returns q = min over heads of sum_j w_j kernel(goal, u_j), as in q_fn.
    """
    import jax
    import jax.numpy as jnp

    from impls.critics.density_fb import _delta_phi, _raw_block_cols

    f_p, b_p = net["density_f"].params, net["density_b"].params
    pool = jnp.asarray(pool_obs)
    u_sc = critic._b_query(pool)
    b_emb = critic._B(b_p, u_sc)  # [M,d]
    g = jnp.broadcast_to(jnp.asarray(goal), (len(acts), len(goal)))
    f_emb = critic._F(f_p, jnp.asarray(obs), jnp.asarray(acts), g)
    w_heads = jax.nn.softmax(
        jnp.einsum("hbd,md->hbm", f_emb, b_emb) / critic.temperature, -1
    )
    w = jnp.mean(w_heads, axis=0)  # [B,M]
    # Actor readout: goal reward kernel in raw coordinates
    u_raw = _raw_block_cols(pool, critic.blocks)
    kg = _delta_phi(jnp.asarray(goal)[None], u_raw, critic.blocks)  # [1,M]
    q = jnp.min((w_heads * kg[None]).sum(-1), axis=0)  # [B]
    return np.asarray(w), np.asarray(q)


# Metrics


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Within-state AUC by the rank formula; NaN when only one class is present."""
    pos, neg = labels.sum(), (~labels.astype(bool)).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    import pandas as pd

    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# Pass B driver


def readouts(variant, banks, gt, pool, out_dir: Path) -> List[Dict[str, Any]]:
    """Every (phase, state, action, goal) readout for one variant."""
    import jax.numpy as jnp

    critic, net, env = variant.agent.critic, variant.agent.net, variant.env
    is_density = variant.critic.startswith("density_fb")
    tasks = A.eval_tasks(variant)
    pool_obs = pool["obs"]
    held_mask = pool["held"].astype(bool)
    assert len(pool_obs) == POOL_SIZE, (
        f"pool must be {POOL_SIZE} rows, got {len(pool_obs)}"
    )

    goals = []
    for t in tasks:
        _o, info = env.reset(options=dict(task_id=t, render_goal=False))
        goals.append((t, np.asarray(info["goal"], np.float32)))

    rows = []
    for phase, bank in banks.items():
        acts_all = gt[f"{phase}_actions"]
        psec_all = gt[f"{phase}_p_secure"]
        for i in range(len(bank["qpos"])):
            # Read at the restored observation
            env.reset(options=dict(task_id=goals[0][0], render_goal=False))
            R.set_state(env, bank["qpos"][i], bank["qvel"][i])
            R.settle(env)
            ob = R.observation(env)
            acts = acts_all[i]
            obs_rep = np.repeat(ob[None], len(acts), axis=0)
            for task_id, goal in goals:
                if is_density:
                    w, q = density_readout(critic, net, obs_rep, acts, goal, pool_obs)
                    held_mass = w[:, held_mask].sum(-1)
                    row_mass = w.mean(-1)  # always 1/M
                else:
                    z = np.asarray(
                        critic.map_eval_cond(net, {"goal": jnp.asarray(goal)})["z"]
                    )
                    M, q = fb_readout(critic, net, obs_rep, acts, z, pool_obs)
                    pos = np.maximum(M, 0.0)
                    denom = pos.sum(-1)
                    held_mass = np.where(
                        denom > 0,
                        pos[:, held_mask].sum(-1) / np.maximum(denom, 1e-12),
                        np.nan,
                    )
                    row_mass = M.mean(-1)
                for ai in range(len(acts)):
                    rows.append(
                        dict(
                            env=variant.env_name,
                            critic=variant.critic,
                            seed=variant.seed,
                            phase=phase,
                            state=i,
                            action=ai,
                            task=task_id,
                            p_secure=float(psec_all[i, ai]),
                            held_mass=float(held_mass[ai]),
                            row_mass=float(row_mass[ai]),
                            q=float(q[ai]),
                            bank_row=int(bank["row"][i]),
                        )
                    )
        print(
            f"  [{variant.label}] {phase}: {len(bank['qpos'])} states x {N_ACTIONS} actions "
            f"x {len(goals)} goals",
            flush=True,
        )
    return rows


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=["fb_ref", "density_fb_state"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--exp-root", default="exp")
    ap.add_argument("--out-root", default="analysis/cube_bottleneck")
    ap.add_argument(
        "--gt-only",
        action="store_true",
        help="compute the shared p_secure ground truth and stop",
    )
    ap.add_argument(
        "--state-start",
        type=int,
        default=None,
        help="Pass A worker: first state index (inclusive)",
    )
    ap.add_argument(
        "--state-end",
        type=int,
        default=None,
        help="Pass A worker: last state index (exclusive)",
    )
    ap.add_argument(
        "--merge-gt",
        action="store_true",
        help="assemble the sliced Pass A parts and stop",
    )
    ap.add_argument(
        "--max-states",
        type=int,
        default=0,
        help="truncate every bank to this many states (0 = all); smoke runs only",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.env
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = PhaseThresholds()
    pool = build_pool(args.env, out_dir)
    print(
        f"[pool] {len(pool['obs'])} rows, held fraction {pool['held'].mean():.3f}",
        flush=True,
    )

    pb = np.load(out_dir / "phase_banks.npz")
    banks = {
        p: {k: pb[f"{p}_{k}"] for k in ("row", "obs", "action", "qpos", "qvel")}
        for p in ("P2", "P1", "P3")
    }
    tag = ""
    if args.max_states:
        banks = {
            p: {k: v[: args.max_states] for k, v in b.items()} for p, b in banks.items()
        }
        tag = f"_smoke{args.max_states}"
        print(
            f"[smoke] banks truncated to {args.max_states} states per phase", flush=True
        )

    if args.merge_gt:
        merge_ground_truth(
            out_dir,
            ("P2", "P1", "P3"),
            {p: len(b["qpos"]) for p, b in banks.items()},
            tag=tag,
        )
        print(f"[gt] merged Pass A parts -> {out_dir / f'ground_truth{tag}.npz'}")
        return 0

    runs = A.discover(args.exp_root, args.env, args.critics, args.seeds, args.epoch)
    if not runs:
        raise SystemExit("no runs discovered")

    # Pass A: shared ground truth from the first agent's actor
    gt_variant = A.load_variant(runs[0]["run_dir"], args.epoch)
    R.ensure_cube_env(gt_variant.env)
    print(
        f"[gt] p_secure continuations under {gt_variant.label}'s flow actor", flush=True
    )
    sl = (
        slice(args.state_start, args.state_end)
        if args.state_start is not None or args.state_end is not None
        else None
    )
    gt = ground_truth(gt_variant, banks, out_dir, thr, tag=tag, state_slice=sl)
    if sl is not None:
        print(
            f"[gt] slice {sl.start}:{sl.stop} written; run --merge-gt when all are done"
        )
        return 0

    # Check that P2 states are decision points
    sd = gt["P2_p_secure"].std(axis=1)
    n_ok = int((sd > G4_STD_MIN).sum())
    g4 = dict(
        n_states=int(len(sd)),
        n_std_above=n_ok,
        required=G4_MIN_STATES,
        median_std=float(np.median(sd)),
        passes=bool(n_ok >= G4_MIN_STATES),
        p1_median_std=float(np.median(gt["P1_p_secure"].std(axis=1))),
        p3_median_std=float(np.median(gt["P3_p_secure"].std(axis=1))),
    )
    (out_dir / f"g4{tag}.json").write_text(json.dumps(g4, indent=2))
    print(
        f"[G4] P2 states with p_secure std > {G4_STD_MIN}: {n_ok}/{len(sd)} "
        f"(need {G4_MIN_STATES}) -> {'PASS' if g4['passes'] else 'FAIL'}",
        flush=True,
    )
    if args.gt_only:
        return 0

    part_dir = out_dir / "measure_probe"
    part_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        variant = (
            gt_variant
            if r["run_dir"] == runs[0]["run_dir"]
            else A.load_variant(r["run_dir"], args.epoch)
        )
        rows = readouts(variant, banks, gt, pool, out_dir)
        out = part_dir / f"{r['critic']}_seed{r['seed']}{tag}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[test2] wrote {len(rows)} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
