"""Video version of the lead figure (fig_lead_combined): FB and Density-FB each run their
own policy from the SAME held start (held-bank row 21, task 4), side by side. Under each
rollout, the critic's predicted future cube positions at the current state and action,
computed at every step (fig_drop_strip computes them on six frames only).

Top of each column: the stage of the current step from the gripper contact signal
(Carrying, Set down = on the table but still gripped, Released, At goal).
Map: weighted KDE over the reference pool of cube positions, weights = the critic's
predicted successor distribution averaged over the task's goal draws, as in the figure.

Rollouts are run on CPU (JAX_PLATFORMS=cpu): on GPU the flow actor's float
non-determinism changes the rollout from run to run.

Usage:
  JAX_PLATFORMS=cpu MUJOCO_GL=egl python scripts/probes/cube_bottleneck/video_headline.py \
      --out website/static/video/lead_compare
"""
import argparse
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import fig_carry_strip_v2 as V2
from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.fig_carry_strip import predicted_cube_mass
from scripts.probes.cube_bottleneck.fig_outcome_map import CM, task_goals
from scripts.probes.cube_bottleneck.phases import PhaseThresholds

ROOT = V2.ROOT
STAGE_COLOURS = {"Carrying": "#D55E00", "Set down": "#DE8F05", "Released": "#BB2525", "At goal": "#029E73"}
HEAD_COLOURS = {"fb_ref": "#1f77b4", "density_fb_state": "#e8641b"}
SUCCESS_RADIUS_M = 0.04


def rollout_fixed_goal(arm, actor_fn, qpos, qvel, task_id, goal, max_steps, thr):
    """fig_drop_strip.rollout with the goal row passed in. The env's own goal row redraws
    the arm pose on every reset from an unseeded RNG, which makes the rollout differ from
    run to run; a dataset goal from task_goals is reproducible (see its docstring)."""
    env = arm.env
    env.reset(options=dict(task_id=task_id, render_goal=False))
    R.set_state(env, qpos, qvel); R.settle(env)
    ob = R.observation(env)
    goal = np.asarray(goal, np.float32)
    obs, acts, qp, qv, held, succ = [], [], [], [], [], False
    u = env.unwrapped
    for t in range(max_steps):
        a = np.clip(np.asarray(actor_fn(observations=ob[None], goals=goal[None], temperature=0))[0], -1, 1).astype(np.float32)
        obs.append(ob.copy()); acts.append(a); qp.append(np.asarray(u._data.qpos).copy()); qv.append(np.asarray(u._data.qvel).copy())
        cube_z = float(u._data.qpos[R.CUBE_QPOS][2])
        table_z = float(u.cur_task_info["init_xyzs"][int(getattr(u, "_target_block", 0) or 0)][2])
        grip = float(ob[17] / R.GRIPPER_SCALER)
        held.append((cube_z - table_z) > thr.delta_lift and grip > thr.tau_grip)
        ob_, _r, term, trunc, info = env.step(a)
        ob = np.asarray(ob_, np.float32)
        succ = succ or bool(info.get("success", False))
        if term or trunc:
            break
    # also keep the state after the last step (where the env reports success)
    obs.append(ob.copy()); acts.append(acts[-1]); qp.append(np.asarray(u._data.qpos).copy()); qv.append(np.asarray(u._data.qvel).copy())
    held.append(held[-1])
    return dict(obs=np.stack(obs), acts=np.stack(acts), qpos=np.stack(qp), qvel=np.stack(qv),
                held=np.asarray(held), success=succ)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def stages(lift, contact, hand_cube, to_goal, thr: PhaseThresholds):
    """Stage from the gripper's contact signal, not from cube height alone: a cube lowered
    onto the table while still gripped is "Set down", not a drop."""
    out = []
    for t in range(len(lift)):
        gripped = contact[t] > 0.5 and hand_cube[t] < thr.eps_reach
        if to_goal[t] < SUCCESS_RADIUS_M:   # the env's own test: 3-D distance to the goal
            out.append("At goal")
        elif not gripped:
            out.append("Released")
        elif lift[t] > thr.delta_lift:
            out.append("Carrying")
        else:
            out.append("Set down")
    return out

def occupancy_image(grid, cmap, ext, path_xy, now_xy, goal_xy, width_px, title):
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, width_px * aspect / dpi + 0.42), dpi=dpi)
    ax = fig.add_axes([0.01, 0.01, 0.98, (width_px * aspect / dpi) / (width_px * aspect / dpi + 0.42) - 0.02])
    ax.imshow((grid / max(grid.max(), 1e-12)).T, origin="lower", extent=ext, aspect="auto",
              cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="#8a8a86", lw=1.4, zorder=4)
    ax.plot(goal_xy[0], goal_xy[1], marker="*", ms=20, color="#ffd23f", mec="#1a1a1a", mew=1.2, ls="none", zorder=7)
    ax.plot(now_xy[0], now_xy[1], marker="o", ms=11, mfc="#0b0b0b", mec="white", mew=1.6, ls="none", zorder=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cfcfcb")
    fig.text(0.5, 1 - 0.2 / (width_px * aspect / dpi + 0.42), title, ha="center", va="center",
             fontsize=16, color="#333333", family="DejaVu Sans")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, facecolor="white"); plt.close(fig)
    return np.asarray(Image.open(buf).convert("RGB"))


def column(name, colour, stage, t, sim, occ, final_text=None):
    w = sim.shape[1]
    head, band = 56, 52
    occ = np.asarray(Image.fromarray(occ).resize((w, int(occ.shape[0] * w / occ.shape[1]))))
    H = head + band + sim.shape[0] + occ.shape[0]
    canvas = Image.new("RGB", (w, H), "white")
    canvas.paste(Image.fromarray(sim), (0, head + band))
    canvas.paste(Image.fromarray(occ), (0, head + band + sim.shape[0]))
    dr = ImageDraw.Draw(canvas)
    dr.text((w // 2, head // 2), name, fill=colour, font=font(32, True), anchor="mm")
    dr.rectangle([0, head, w, head + band], fill=STAGE_COLOURS[stage])
    dr.text((w // 2, head + band // 2), stage, fill="white", font=font(28, True), anchor="mm")
    y_img_bot = head + band + sim.shape[0]
    dr.text((14, y_img_bot - 12), f"t = {t}", fill="white", font=font(22), anchor="ld")
    if final_text:
        dr.rectangle([0, y_img_bot - 100, w, y_img_bot - 56], fill=(0, 0, 0))
        dr.text((w // 2, y_img_bot - 78), final_text, fill="white", font=font(24, True), anchor="mm")
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--bank-row", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--after-drop", type=int, default=10**6, help="steps kept after a failed rollout's first drop")
    ap.add_argument("--goal-draws", type=int, default=8)
    ap.add_argument("--bw", type=float, default=2.5)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="website/static/video/lead_compare")
    args = ap.parse_args()

    import jax
    import imageio.v2 as imageio
    import ogbench
    from impls.utils.evaluation import supply_rng

    thr = PhaseThresholds()
    hb = np.load(ROOT / "held_bank.npz")
    pool_obs = dict(np.load(ROOT / "pool.npz"))["obs"]
    pool_xy = V2.world_to_plot(pool_obs[:, V2.CUBE_OBS][:, :2] * CM)
    renv = ogbench.make_env_and_datasets(args.env, env_only=True, render_mode="rgb_array",
                                         width=args.size, height=args.size)
    runs = {}
    goal_xy = None
    for a in V2.ORDER:
        arm = A.load_arm(Path("exp") / f"{args.env}_{a}_flowbc_seed_{args.seed}")
        R.ensure_cube_env(arm.env)
        actor_fn = supply_rng(arm.agent.sample_actions, rng=jax.random.PRNGKey(0))
        i = args.bank_row
        goals = task_goals(arm, args.task, args.goal_draws, 20260904)
        ro = rollout_fixed_goal(arm, actor_fn, hb["qpos"][i], hb["qvel"][i], args.task, goals[0],
                                args.max_steps, thr)
        if goal_xy is None:
            goal_xy = V2.world_to_plot(np.mean([g[V2.CUBE_OBS][:2] for g in goals], axis=0) * CM)
        h = ro["held"]
        first_drop = int(np.flatnonzero(~h)[0]) if (~h).any() else -1
        T = len(h) if ro["success"] or first_drop < 0 else min(len(h), first_drop + args.after_drop)
        cube_xy = V2.world_to_plot(ro["obs"][:T, V2.CUBE_OBS][:, :2] * CM)
        u = arm.env.unwrapped
        table_z = float(u.cur_task_info["init_xyzs"][0][2])
        goal_w = np.asarray(u.cur_task_info["goal_xyzs"][0], float)
        cube_w = ro["qpos"][:T, R.CUBE_QPOS]
        eff = []
        for q, v in zip(ro["qpos"][:T], ro["qvel"][:T]):
            R.set_state(arm.env, q, v); R.forward(arm.env)
            eff.append(np.asarray(u._data.site_xpos[u._pinch_site_id]).copy())
        sig = dict(lift=cube_w[:, 2] - table_z, contact=ro["obs"][:T, R.CONTACT_IDX],
                   hand_cube=np.linalg.norm(np.asarray(eff) - cube_w, axis=1),
                   to_goal=np.linalg.norm(cube_w - goal_w, axis=1))
        mass = np.stack([predicted_cube_mass(arm, ro["obs"][t], ro["acts"][t], goals, pool_obs)[0]
                         for t in range(T)])
        print(f"{V2.LABEL[a]}: steps {len(h)} kept {T} success {ro['success']} first_drop {first_drop}")
        runs[a] = dict(qpos=ro["qpos"][:T], qvel=ro["qvel"][:T], cube_xy=cube_xy, mass=mass,
                       held=h[:T], success=bool(ro["success"]), sig=sig)
    for a, r in runs.items():
        r["stages"] = stages(**r["sig"], thr=thr)

    allpath = np.concatenate([r["cube_xy"] for r in runs.values()])
    lo = np.minimum(np.percentile(pool_xy, 0.5, axis=0), np.minimum(allpath.min(0), goal_xy)) - 3.0
    hi = np.maximum(np.percentile(pool_xy, 99.5, axis=0), np.maximum(allpath.max(0), goal_xy)) + 3.0
    xs = np.linspace(lo[0], hi[0], 140); ys = np.linspace(lo[1], hi[1], 70)
    ext = [lo[0], hi[0], lo[1], hi[1]]

    renv.reset(seed=0, options=dict(task_id=args.task, render_goal=False))
    crop = slice(int(0.12 * args.size), int(0.97 * args.size))

    def sim_frame(r, t):
        R.set_state(renv, r["qpos"][t], r["qvel"][t]); R.forward(renv)
        return np.asarray(renv.render())[crop]

    cache = {}
    n = max(len(r["qpos"]) for r in runs.values())
    frames = []
    for t in range(n):
        cols = []
        for a in V2.ORDER:
            r = runs[a]; tt = min(t, len(r["qpos"]) - 1)
            if (a, tt) not in cache:
                grid = V2.kde_grid(pool_xy, r["mass"][tt], xs, ys, args.bw)
                occ = occupancy_image(grid, V2.CMAPS[a], ext, r["cube_xy"][: tt + 1], r["cube_xy"][tt], goal_xy,
                                      args.size, "critic: where will the cube be?")
                cache[(a, tt)] = (sim_frame(r, tt), occ)
            sim, occ = cache[(a, tt)]
            final = None
            if t >= len(r["qpos"]) - 1:
                final = "Success" if r["success"] else "Episode over: cube not on goal"
            cols.append(column(V2.LABEL[a], HEAD_COLOURS[a], r["stages"][tt], tt, sim, occ, final))
        gap = np.full((cols[0].shape[0], 16, 3), 255, np.uint8)
        frames.append(np.concatenate([cols[0], gap, cols[1]], axis=1))
    frames += [frames[-1]] * args.fps

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(f"{out}.mp4", frames, fps=args.fps, codec="libx264", quality=8,
                     pixelformat="yuv420p", macro_block_size=16,
                     output_params=["-movflags", "+faststart"])
    Image.fromarray(frames[len(runs["density_fb_state"]["qpos"]) - 1]).save(f"{out}_poster.jpg", quality=88)
    print(f"wrote {out}.mp4 ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


if __name__ == "__main__":
    main()
