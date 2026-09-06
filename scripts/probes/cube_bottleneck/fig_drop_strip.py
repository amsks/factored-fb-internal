"""Companion to fig_carry_strip_v2: the same strip on a rollout of plain FB's OWN policy
from a held start, choosing an episode in which FB drops the cube. Both critics are read
on the same rollout states, so the maps stay a paired comparison; the frames themselves
are FB's behaviour. Output dict matches fig_carry_strip_v2.render's contract."""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np

from scripts.probes.cube_bottleneck import fig_carry_strip_v2 as V2
from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import banks as BK
from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.fig_outcome_map import CM, task_goals
from scripts.probes.cube_bottleneck.fig_carry_strip import b_embed, fit_probe, predicted_cube_mass
from scripts.probes.cube_bottleneck.phases import PhaseThresholds

ROOT = V2.ROOT
CACHE = ROOT / "drop_strip_cache.npz"   # overridden by --cache-tag


def rollout(arm, actor_fn, qpos, qvel, task_id, max_steps, thr):
    env = arm.env
    _o, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal = np.asarray(info["goal"], np.float32)
    R.set_state(env, qpos, qvel); R.settle(env)
    ob = R.observation(env)
    obs, acts, qp, qv, held, succ = [], [], [], [], [], False
    u = env.unwrapped
    for t in range(max_steps):
        a = np.clip(np.asarray(actor_fn(observations=ob[None], goals=goal[None], temperature=0))[0], -1, 1).astype(np.float32)
        obs.append(ob.copy()); acts.append(a); qp.append(np.asarray(u._data.qpos).copy()); qv.append(np.asarray(u._data.qvel).copy())
        cube_z = float(u._data.qpos[R.CUBE_QPOS][2]); table_z = float(u.cur_task_info["init_xyzs"][int(getattr(u, "_target_block", 0) or 0)][2])
        grip = float(ob[17] / R.GRIPPER_SCALER)
        held.append((cube_z - table_z) > thr.delta_lift and grip > thr.tau_grip)
        ob_, _r, term, trunc, info = env.step(a)
        ob = np.asarray(ob_, np.float32)
        succ = succ or bool(info.get("success", False))
        if term or trunc:
            break
    return dict(obs=np.stack(obs), acts=np.stack(acts), qpos=np.stack(qp), qvel=np.stack(qv),
                held=np.asarray(held), success=succ, goal=goal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=V2.ORDER)
    ap.add_argument("--policy", default="fb_ref", help="whose policy produces the rollout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--goal-draws", type=int, default=8)
    ap.add_argument("--probe-rows", type=int, default=20000)
    ap.add_argument("--bw", type=float, default=2.5)
    ap.add_argument("--bank-row", type=int, default=None, help="force this held-bank row (no search)")
    ap.add_argument("--cache-tag", default="", help="suffix for the cache/figure names")
    ap.add_argument("--min-drop-step", type=int, default=8)
    ap.add_argument("--min-carry-cm", type=float, default=8.0)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global CACHE
    CACHE = ROOT / f"drop_strip_cache{args.cache_tag}.npz"
    args.out = args.out or str(ROOT / "figures" / f"fig_drop_strip{args.cache_tag}")

    if CACHE.exists() and not args.recompute:
        d = dict(np.load(CACHE, allow_pickle=True)); print("loaded", CACHE)
        V2.render(d, args); return

    import jax
    from impls.utils.evaluation import supply_rng
    thr = PhaseThresholds()
    arms = {a: A.load_arm(Path("exp") / f"{args.env}_{a}_flowbc_seed_{args.seed}") for a in args.critics}
    pol = arms[args.policy]; R.ensure_cube_env(pol.env)
    actor_fn = supply_rng(pol.agent.sample_actions, rng=jax.random.PRNGKey(0))
    hb = np.load(ROOT / "held_bank.npz")
    # candidates ordered by initial cube-goal distance (far first), so the drop follows a real carry
    g_tmp = task_goals(pol, args.task, args.goal_draws, 20260904)
    gxy = V2.world_to_plot(np.mean([g[V2.CUBE_OBS][:2] for g in g_tmp], axis=0) * CM)
    start_xy = V2.world_to_plot(np.asarray(hb["obs"])[:, V2.CUBE_OBS][:, :2] * CM)
    order = np.argsort(-np.linalg.norm(start_xy - gxy, axis=1))
    chosen = None
    if args.bank_row is not None:
        i = args.bank_row
        ro = rollout(pol, actor_fn, hb["qpos"][i], hb["qvel"][i], args.task, args.max_steps, thr)
        h = ro["held"]; first_drop = int(np.flatnonzero(~h)[0]) if (~h).any() else -1
        print(f"  forced bank row {i}: steps {len(h)} success {ro['success']} first_drop {first_drop}")
        chosen = (int(i), ro, first_drop)
        order = []
    for i in order:
        ro = rollout(pol, actor_fn, hb["qpos"][i], hb["qvel"][i], args.task, args.max_steps, thr)
        h = ro["held"]; first_drop = int(np.flatnonzero(~h)[0]) if (~h).any() else -1
        cxy = V2.world_to_plot(ro["obs"][:, V2.CUBE_OBS][:, :2] * CM)
        carried = float(np.linalg.norm(cxy[max(first_drop, 0)] - cxy[0])) if first_drop > 0 else 0.0
        print(f"  bank row {i}: start-goal {np.linalg.norm(start_xy[i]-gxy):.1f} cm, steps {len(h)}, "
              f"success {ro['success']}, first_drop {first_drop}, carried {carried:.1f} cm")
        if (not ro["success"]) and first_drop >= args.min_drop_step and carried >= args.min_carry_cm \
                and (~h[first_drop:]).mean() > 0.8:
            chosen = (int(i), ro, first_drop); break
    assert chosen is not None, "no dropped episode found"
    i, ro, fd = chosen
    T = len(ro["held"]); end = (T - 1) if (ro["success"] or fd < 0) else min(T - 1, fd + 12)
    rows = np.unique(np.linspace(0, end, args.frames).astype(int))
    obs_f, act_f = ro["obs"][rows], ro["acts"][rows]
    true_xy = V2.world_to_plot(obs_f[:, V2.CUBE_OBS][:, :2] * CM)
    lifted = ro["held"][rows]
    path_xy = V2.world_to_plot(ro["obs"][: end + 1][:, V2.CUBE_OBS][:, :2] * CM)
    pool_obs = dict(np.load(ROOT / "pool.npz"))["obs"]
    pool_xy = V2.world_to_plot(pool_obs[:, V2.CUBE_OBS][:, :2] * CM)

    import ogbench
    env = ogbench.make_env_and_datasets(args.env, env_only=True, render_mode="rgb_array")
    env.reset(seed=0, options=dict(task_id=args.task, render_goal=False))
    imgs = []
    for r in rows:
        R.set_state(env, ro["qpos"][r], ro["qvel"][r]); R.forward(env); imgs.append(np.asarray(env.render()))
    imgs = np.stack(imgs)

    raw = BK.load_raw(args.env)
    probe_idx = np.random.default_rng(0).choice(len(raw["observations"]), size=args.probe_rows, replace=False)
    obs_p = np.asarray(raw["observations"][probe_idx], np.float32)
    tgt_cube = V2.world_to_plot(obs_p[:, V2.CUBE_OBS][:, :2] * CM)

    d = dict(rows=rows, true_xy=true_xy, lifted=lifted, path_xy=path_xy, pool_xy=pool_xy, imgs=imgs,
             bank_row=i, first_drop=fd, policy=args.policy, success=bool(ro["success"]))
    goal_xy = None
    for a, arm in arms.items():
        goals = task_goals(arm, args.task, args.goal_draws, 20260904)
        if goal_xy is None:
            goal_xy = V2.world_to_plot(np.mean([g[V2.CUBE_OBS][:2] for g in goals], axis=0) * CM)
        W, r2 = fit_probe(arm, obs_p, tgt_cube)
        E = b_embed(arm, obs_f)
        d[f"dec_{a}"] = np.concatenate([E, np.ones((len(E), 1), np.float32)], 1) @ W
        d[f"r2_{a}"] = r2
        d[f"mass_{a}"] = np.stack([predicted_cube_mass(arm, obs_f[k], act_f[k], goals, pool_obs)[0] for k in range(len(rows))])
        print(f"  {a:20s} R2 {r2:+.3f} decode err (cm) {np.round(np.linalg.norm(d[f'dec_{a}'] - true_xy, axis=1), 1).tolist()}")
    d["goal_xy"] = goal_xy
    np.savez(CACHE, **d); print("cached", CACHE, "| bank row", i, "first_drop", fd, "frames", rows.tolist())
    V2.render(d, args)


if __name__ == "__main__":
    main()
