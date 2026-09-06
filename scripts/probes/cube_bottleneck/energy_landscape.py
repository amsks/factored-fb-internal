"""Energy landscapes over goal space — FB's degeneracy in the weakly-controlled factor.

Illustrative figure for PAPER/ICLR. No rollouts: every number is a critic readout at a
restored dataset state, so the whole pass is minutes on CPU jax.

The question each panel asks is the one the task asks: *hold the state fixed, move the
goal.* Sweeping the goal's CUBE position walks the weakly-controlled factor (effective
controllability .52 on cube-single); sweeping the goal's EFFECTOR position walks the
strongly-controlled one (~1). A critic that has resolved a factor puts a basin around the
target it is asked for; a critic that has not returns the same value everywhere.

Energy E = -Q, so "downhill = the goal I can deliver" and a basin is a minimum.

The sweep is a SINGLE-FACTOR COUNTERFACTUAL and is deliberately symmetric between the two
factors: the goal row is the state's own observation with two columns overwritten on a
grid, every other goal coordinate held. So the effector sweep moves the goal's effector
columns without re-solving the goal's joint angles -- the goal is a query vector handed to
B, not a state we claim to visit, and the same objection would apply to the cube sweep
(which does stay on the table manifold). What makes the comparison fair is that both
factors get the identical operation.

Held state bank, pool and restore semantics are the cube-bottleneck probe's
(`banks.py`, `measure_probe.build_pool`, `restore.py` -- ctrl=0 is OPEN, obs[18] needs 2
settle steps).
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
from typing import Any, Dict

import numpy as np

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import restore as R

# Observation columns (cube_env.compute_observation, state branch; see restore.py):
#   12:15 effector_pos, 17 gripper, 18 contact, 19:22 cube pos -- all in the env's
#   centered/scaled frame ((p - [0.425, 0, 0]) * 10), which is the frame the goal lives in.
EFF_OBS = slice(12, 15)
CUBE_OBS = slice(19, 22)

SWEEPS = {"cube": CUBE_OBS, "effector": EFF_OBS}
ANALYSIS = Path("analysis/cube_bottleneck")

# ── state-side sweeps (mode "state") ────────────────────────────────────────────────────
# The goal is HELD FIXED at a real task goal and the STATE is moved instead, so the panel
# reads as "how far am I from done?" and the goal supplies the reference the goal-side
# sweep lacks: a critic that has resolved a factor must put its energy minimum ON the goal.
#
#   assembly  the held cube AND the effector translate together, the grasp offset kept --
#             the physically meaningful counterfactual at a held state ("the arm is holding
#             the cube, at position p"), and the transport axis the audit implicates.
#   cube      the cube alone moves (the grasp is broken; kept as the isolated factor).
#   effector  the arm alone moves, cube where it is -- the strongly-controlled factor, the
#             positive control that says the critic is not flat everywhere.
STATE_SWEEPS = ("assembly", "cube", "effector")


def _fb_q(critic, net, obs_rep, acts_rep, goals):
    """Plain FB: Q(s, a, z(g)) = min over the two heads of <F(s,a,z), z>, the training
    target. z = project_z(B(goal)) is map_eval_cond's own hook, so this is exactly the
    quantity the FB actor ascends at eval."""
    import jax.numpy as jnp

    z = critic.map_eval_cond(net, {"goal": jnp.asarray(goals)})["z"]          # [G, d]
    le = critic._le(net["left_encoder"].params, jnp.asarray(obs_rep))
    Fs = critic._F(net["forward"].params, le, z, jnp.asarray(acts_rep))       # [P, G, d]
    return np.asarray(jnp.min((Fs * z).sum(-1), axis=0))


def _density_q(critic, net, obs_rep, acts_rep, goals, pool_obs):
    """Density-FB: q = sum_j w_j(s,a,g) * kernel(g, u_j), min over heads -- q_fn's own
    readout, reproduced on the frozen pool (measure_probe.density_readout)."""
    import jax
    import jax.numpy as jnp

    from impls.critics.density_fb import _delta_phi, _raw_block_cols

    f_p, b_p = net["density_f"].params, net["density_b"].params
    pool = jnp.asarray(pool_obs)
    b_emb = critic._B(b_p, critic._b_query(net, pool, pool))                  # [M, d]
    f_emb = critic._F(f_p, critic._sobs(net, jnp.asarray(obs_rep)),
                      jnp.asarray(acts_rep), jnp.asarray(goals))              # [H, G, d]
    w = jax.nn.softmax(jnp.einsum("hbd,md->hbm", f_emb, b_emb) / critic.temp, -1)
    kg = _delta_phi(jnp.asarray(goals), _raw_block_cols(pool, critic.blocks), critic.blocks)
    return np.asarray(jnp.min((w * kg[None]).sum(-1), axis=0))


def _q_generic(critic, net, obs, acts, goals):
    """Q through the SHIPPED critic interface -- `map_eval_cond` then `q_fn` -- which every
    critic in impls/critics implements identically (base.py:30) and which is the seam the
    actors themselves ascend. So one readout serves FB, RLDP, HILP, PSM and their density
    variants without a bespoke reimplementation per method.

    The hand-rolled `_fb_q` / `_density_q` above stay because they were written against the
    cube-bottleneck probe's own readouts and the published probe numbers came through them;
    `--generic-q` switches every arm onto this path instead, which is what a multi-method
    panel needs (one quantity, one code path, no per-method drift). `test_q_readout_parity`
    in tests/probes pins the two against each other for the two arms that have both.
    """
    import jax.numpy as jnp

    # `net` is a dict of per-module TrainStates; `q_fn` wants the matching dict of param
    # pytrees (it indexes params["forward"], params["left_encoder"], ...), which is what the
    # actors are handed inside their loss. Not `net.params` -- there is no such attribute.
    params = {k: v.params for k, v in net.items()}
    cond = critic.map_eval_cond(net, {"goal": jnp.asarray(goals)})
    q, _ = critic.q_fn(params, jnp.asarray(obs), cond, jnp.asarray(acts),
                       stop_encoder_grad=True)
    return np.asarray(q)


def grid_axes(pool_obs: np.ndarray, n: int, lo_pct: float = 2.0) -> Dict[str, Any]:
    """Sweep extent from the DATA, not from a guessed table rectangle: the 2nd-98th
    percentile of each factor's own xy over the frozen pool."""
    ax = {}
    for name, sl in SWEEPS.items():
        cols = pool_obs[:, sl][:, :2]
        lo = np.percentile(cols, lo_pct, axis=0)
        hi = np.percentile(cols, 100 - lo_pct, axis=0)
        ax[name] = (np.linspace(lo[0], hi[0], n).astype(np.float32),
                    np.linspace(lo[1], hi[1], n).astype(np.float32))
    return ax


def run_arm(run_dir: Path, bank: Dict[str, np.ndarray], pool_obs: np.ndarray,
            n_grid: int, n_states: int, out: Path) -> None:
    arm = A.load_arm(run_dir)
    is_density = arm.critic.startswith("density_fb")
    critic, net, env = arm.agent.critic, arm.agent.net, arm.env
    task0 = A.eval_tasks(arm)[0]

    axes = grid_axes(pool_obs, n_grid)
    n_states = min(n_states, len(bank["qpos"]))
    store: Dict[str, Any] = {"grid": np.asarray(n_grid)}
    for name, (xs, ys) in axes.items():
        store[f"{name}_x"], store[f"{name}_y"] = xs, ys
        store[f"q_{name}"] = np.zeros((n_states, n_grid, n_grid), np.float32)
    store["state_cube_xy"] = np.zeros((n_states, 2), np.float32)
    store["state_eff_xy"] = np.zeros((n_states, 2), np.float32)
    store["state_row"] = bank["row"][:n_states]

    for i in range(n_states):
        env.reset(options=dict(task_id=task0, render_goal=False))
        R.set_state(env, bank["qpos"][i], bank["qvel"][i])
        R.settle(env)
        ob = R.observation(env)
        store["state_cube_xy"][i] = ob[CUBE_OBS][:2]
        store["state_eff_xy"][i] = ob[EFF_OBS][:2]

        n = n_grid * n_grid
        obs_rep = np.repeat(ob[None], n, axis=0)
        acts_rep = np.repeat(np.asarray(bank["action"][i], np.float32)[None], n, axis=0)
        for name, sl in SWEEPS.items():
            xs, ys = axes[name]
            # meshgrid indexing="ij" so the stored array is [x, y]; the plotter transposes
            # once, at the imshow call, rather than every consumer guessing.
            XX, YY = np.meshgrid(xs, ys, indexing="ij")
            goals = obs_rep.copy()
            goals[:, sl.start] = XX.ravel()
            goals[:, sl.start + 1] = YY.ravel()
            q = (_density_q(critic, net, obs_rep, acts_rep, goals, pool_obs) if is_density
                 else _fb_q(critic, net, obs_rep, acts_rep, goals))
            store[f"q_{name}"][i] = q.reshape(n_grid, n_grid)
        if (i + 1) % 8 == 0:
            print(f"  [{arm.label}] {i + 1}/{n_states} states", flush=True)

    # The headline statistic, over every state rather than the one that gets drawn:
    # how much of the critic's own dynamic range the weak factor commands.
    rc = np.ptp(store["q_cube"].reshape(n_states, -1), axis=1)
    re = np.ptp(store["q_effector"].reshape(n_states, -1), axis=1)
    ratio = rc / np.maximum(re, 1e-12)
    store["range_cube"], store["range_effector"], store["range_ratio"] = rc, re, ratio
    print(f"[{arm.label}] cube/effector goal-range ratio over {n_states} states: "
          f"median {np.median(ratio):.3f}  IQR [{np.percentile(ratio, 25):.3f}, "
          f"{np.percentile(ratio, 75):.3f}]", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, critic=np.asarray(arm.critic), seed=np.asarray(arm.seed), **store)
    print(f"[{arm.label}] wrote {out}", flush=True)


def run_arm_state(run_dir: Path, bank: Dict[str, np.ndarray], pool_obs: np.ndarray,
                  n_grid: int, n_states: int, out: Path) -> None:
    """Energy over the STATE, goal fixed at each real task goal (see STATE_SWEEPS)."""
    arm = A.load_arm(run_dir)
    is_density = arm.critic.startswith("density_fb")
    critic, net, env = arm.agent.critic, arm.agent.net, arm.env

    tasks = A.eval_tasks(arm)
    goals = []
    for t in tasks:
        _o, info = env.reset(options=dict(task_id=t, render_goal=False))
        goals.append(np.asarray(info["goal"], np.float32))
    goals = np.stack(goals)                                              # [T, obs_dim]

    axes = grid_axes(pool_obs, n_grid)
    xs_c, ys_c = axes["cube"]
    xs_e, ys_e = axes["effector"]
    n_states, T, n = min(n_states, len(bank["qpos"])), len(tasks), n_grid * n_grid
    store: Dict[str, Any] = {
        "grid": np.asarray(n_grid), "task_id": np.asarray(tasks),
        "cube_x": xs_c, "cube_y": ys_c, "effector_x": xs_e, "effector_y": ys_e,
        "goal_cube_xy": goals[:, CUBE_OBS][:, :2], "goal_eff_xy": goals[:, EFF_OBS][:, :2],
        "state_cube_xy": np.zeros((n_states, 2), np.float32),
        "state_eff_xy": np.zeros((n_states, 2), np.float32),
        "state_row": bank["row"][:n_states],
    }
    for sw in STATE_SWEEPS:
        store[f"q_{sw}"] = np.zeros((T, n_states, n_grid, n_grid), np.float32)

    for i in range(n_states):
        env.reset(options=dict(task_id=tasks[0], render_goal=False))
        R.set_state(env, bank["qpos"][i], bank["qvel"][i])
        R.settle(env)
        ob = R.observation(env)
        cube0, eff0 = ob[CUBE_OBS][:2].copy(), ob[EFF_OBS][:2].copy()
        store["state_cube_xy"][i], store["state_eff_xy"][i] = cube0, eff0

        obs_base = np.repeat(ob[None], n, axis=0)
        acts_rep = np.repeat(np.asarray(bank["action"][i], np.float32)[None], n, axis=0)
        variants = {}
        XX, YY = np.meshgrid(xs_c, ys_c, indexing="ij")
        gx, gy = XX.ravel(), YY.ravel()
        for sw in STATE_SWEEPS:
            o = obs_base.copy()
            if sw in ("assembly", "cube"):
                o[:, CUBE_OBS.start], o[:, CUBE_OBS.start + 1] = gx, gy
                if sw == "assembly":
                    # keep the grasp: the effector rides with the cube it is holding
                    o[:, EFF_OBS.start] = gx + (eff0[0] - cube0[0])
                    o[:, EFF_OBS.start + 1] = gy + (eff0[1] - cube0[1])
            else:
                EX, EY = np.meshgrid(xs_e, ys_e, indexing="ij")
                o[:, EFF_OBS.start], o[:, EFF_OBS.start + 1] = EX.ravel(), EY.ravel()
            variants[sw] = o

        for ti in range(T):
            g_rep = np.repeat(goals[ti][None], n, axis=0)
            for sw, o in variants.items():
                q = (_density_q(critic, net, o, acts_rep, g_rep, pool_obs) if is_density
                     else _fb_q(critic, net, o, acts_rep, g_rep))
                store[f"q_{sw}"][ti, i] = q.reshape(n_grid, n_grid)
        if (i + 1) % 4 == 0:
            print(f"  [{arm.label}] {i + 1}/{n_states} states x {T} tasks", flush=True)

    # Does the energy point at the goal? A critic that has resolved the factor must fall
    # as the factor approaches the goal, i.e. corr(E, distance-to-goal) > 0 with E = -Q.
    for sw in STATE_SWEEPS:
        xs, ys = (xs_e, ys_e) if sw == "effector" else (xs_c, ys_c)
        gxy = store["goal_eff_xy"] if sw == "effector" else store["goal_cube_xy"]
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        cors = np.full((T, n_states), np.nan, np.float32)
        for ti in range(T):
            d = np.hypot(XX - gxy[ti, 0], YY - gxy[ti, 1]).ravel()
            for i in range(n_states):
                e = -store[f"q_{sw}"][ti, i].ravel()
                if np.ptp(e) > 1e-12:
                    cors[ti, i] = np.corrcoef(e, d)[0, 1]
        store[f"corr_{sw}"] = cors
        print(f"[{arm.label}] {sw:9s}: corr(energy, dist-to-goal) median "
              f"{np.nanmedian(cors):+.3f}  frac>0 {np.nanmean(cors > 0):.2f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, critic=np.asarray(arm.critic), seed=np.asarray(arm.seed), **store)
    print(f"[{arm.label}] wrote {out}", flush=True)


def run_arm_pool(run_dir: Path, pool_obs: np.ndarray, n_rows: int, chunk: int,
                 out: Path, seed: int = 20260904, policy_action: bool = False,
                 generic_q: bool = False) -> None:
    """Energy over REAL dataset states, goal fixed at each real task goal.

    The counterfactual sweeps above move one factor of the observation and hold the rest,
    which is clean on paper and off the data manifold in practice: at a held state the cube
    cannot be somewhere the gripper is not, and a network evaluated there returns something
    arbitrary rather than something wrong. Measured: both critics produced incoherent
    landscapes that way (`--mode state`), which says nothing about either.

    So this mode fabricates nothing. It evaluates Q(s, a_dataset, g) at `n_rows` real
    dataset transitions and lets the PLOT do the sweeping -- bin the rows by the state's own
    cube xy for the weakly-controlled factor, by its own effector xy for the strongly
    controlled one. Every point is a state the data actually visited; the map over each
    factor is the marginal of the value over it.
    """
    from scripts.probes.cube_bottleneck import banks as BK

    arm = A.load_arm(run_dir)
    is_density = arm.critic.startswith("density_fb")
    critic, net, env = arm.agent.critic, arm.agent.net, arm.env

    tasks = A.eval_tasks(arm)
    goals = []
    for t in tasks:
        _o, info = env.reset(options=dict(task_id=t, render_goal=False))
        goals.append(np.asarray(info["goal"], np.float32))
    goals = np.stack(goals)

    actor_fn = None
    if policy_action:
        # Q(s, a_dataset, g) reads the value at an action chosen for some OTHER purpose, so
        # much of its variation across states is action noise rather than value structure --
        # visible as mottling in the workspace maps. Evaluating at the arm's OWN action for
        # THIS goal gives V(s, g) = Q(s, pi(s,g), g), the landscape the policy actually
        # induces. The action is then goal-conditioned, so (unlike the dataset action) it is
        # part of what the goal changes -- which is the point, not a confound.
        import jax

        from impls.utils.evaluation import supply_rng
        actor_fn = supply_rng(arm.agent.sample_actions, rng=jax.random.PRNGKey(0))

    raw = BK.load_raw(arm.env_name)
    rng = np.random.default_rng(seed)          # fixed: every arm sees the SAME rows
    n_rows = min(n_rows, len(raw["observations"]))
    sel = np.sort(rng.choice(len(raw["observations"]), size=n_rows, replace=False))
    obs = np.asarray(raw["observations"][sel], np.float32)
    acts = np.asarray(raw["actions"][sel], np.float32)

    q = np.zeros((len(tasks), n_rows), np.float32)
    for ti in range(len(tasks)):
        for a in range(0, n_rows, chunk):
            o, ac = obs[a:a + chunk], acts[a:a + chunk]
            g = np.repeat(goals[ti][None], len(o), axis=0)
            if actor_fn is not None:
                ac = np.clip(np.asarray(actor_fn(observations=o, goals=g,
                                                 temperature=0.0)), -1, 1).astype(np.float32)
            if generic_q:
                q[ti, a:a + chunk] = _q_generic(critic, net, o, ac, g)
            else:
                q[ti, a:a + chunk] = (_density_q(critic, net, o, ac, g, pool_obs)
                                      if is_density else _fb_q(critic, net, o, ac, g))
        print(f"  [{arm.label}] task {tasks[ti]}: {n_rows} rows", flush=True)

    store = dict(q=q, row=sel.astype(np.int64), task_id=np.asarray(tasks),
                 policy_action=np.asarray(bool(policy_action)),
                 generic_q=np.asarray(bool(generic_q)),
                 state_cube_xy=obs[:, CUBE_OBS][:, :2], state_eff_xy=obs[:, EFF_OBS][:, :2],
                 goal_cube_xy=goals[:, CUBE_OBS][:, :2],
                 goal_eff_xy=goals[:, EFF_OBS][:, :2],
                 held=((obs[:, BK.CONTACT_IDX] > BK.CONTACT_THRESH)
                       & (np.asarray(raw["qpos"][sel])[:, BK.CUBE_QPOS_Z] > BK.LIFT_THRESH)
                       ).astype(np.int8))

    # Same question as before, now asked of on-manifold states: does the energy rise as the
    # factor gets further from the goal? Reported per factor, over the real rows.
    for name, key, gkey in (("cube", "state_cube_xy", "goal_cube_xy"),
                            ("effector", "state_eff_xy", "goal_eff_xy")):
        cors = []
        for ti in range(len(tasks)):
            d = np.linalg.norm(store[key] - store[gkey][ti], axis=1)
            cors.append(np.corrcoef(-q[ti], d)[0, 1])
        store[f"corr_{name}"] = np.asarray(cors, np.float32)
        print(f"[{arm.label}] {name:9s}: corr(energy, dist-to-goal) per task "
              f"{np.round(cors, 3)}  median {np.median(cors):+.3f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, critic=np.asarray(arm.critic), seed=np.asarray(arm.seed), **store)
    print(f"[{arm.label}] wrote {out}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cube-single-play-v0")
    p.add_argument("--critics", nargs="+", default=["fb_ref", "density_fb_state"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epoch", type=int, default=500000)
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--exp-root", default="exp")
    p.add_argument("--mode", choices=["goal", "state", "pool"], default="pool",
                   help="goal: move the goal, state fixed. state: move the state, goal "
                        "fixed (both counterfactual, both off-manifold). pool: real "
                        "dataset states, goal fixed -- the figure's mode.")
    p.add_argument("--rows", type=int, default=40000, help="pool mode: dataset rows")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--generic-q", action="store_true",
                   help="read Q through the shipped map_eval_cond + q_fn seam for EVERY "
                        "arm, so methods beyond FB/Density-FB are one code path.")
    p.add_argument("--policy-action", action="store_true",
                   help="pool mode: evaluate at the arm's own action for the goal, i.e. "
                        "V(s,g), instead of at the dataset's action.")
    args = p.parse_args()

    root = ANALYSIS / args.env
    bank = dict(np.load(root / "held_bank.npz", allow_pickle=True))
    pool_obs = dict(np.load(root / "pool.npz"))["obs"]
    out_dir = root / "energy"

    runs = A.discover(args.exp_root, args.env, args.critics, args.seeds, args.epoch)
    assert runs, f"no checkpoints for {args.critics} x {args.seeds} in {args.exp_root}"
    tag = {"goal": "", "state": "state_", "pool": "pool_"}[args.mode]
    if args.mode == "pool" and args.policy_action:
        tag = "poolpi_"
    if args.mode == "pool" and args.generic_q:
        tag = tag[:-1] + "g_"
    for r in runs:
        out = out_dir / f"energy_{tag}{r['critic']}_s{r['seed']}.npz"
        if out.exists():
            print(f"[skip] {out} exists")
            continue
        if args.mode == "pool":
            run_arm_pool(r["run_dir"], pool_obs, args.rows, args.chunk, out,
                         policy_action=args.policy_action, generic_q=args.generic_q)
        elif args.mode == "state":
            run_arm_state(r["run_dir"], bank, pool_obs, args.grid, args.states, out)
        else:
            run_arm(r["run_dir"], bank, pool_obs, args.grid, args.states, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
