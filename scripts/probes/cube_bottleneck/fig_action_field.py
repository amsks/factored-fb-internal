"""Fig: WHY the policies differ -- the critic's preferred action, over the workspace.

fig_truth_map.py shows what each policy did. This shows the thing that caused it. A policy
here is trained by ascending its critic's Q, so the critic's action ranking is the
mechanism; if the ranking points at the goal the policy transports, and if it does not, it
wanders.

At held states (cube in the gripper, so moving the effector moves the cube) the candidate
actions are 16 headings on the unit circle in the effector's xy translation channel --
`action[:3]` is the effector translation delta (manipspace_env.set_control), so a heading is
literally "push the cube this way". The remaining channels (z, yaw, gripper) are held at the
state's own dataset action, so the ONLY thing varying is the direction.

Both critics are asked the identical question at the identical states with the identical
action set, and each answers with its argmax heading. Per bin the unit headings are
averaged, so

    arrow direction = where the critic wants the cube pushed
    arrow length    = how consistently it wants that (1 = every state in the bin agrees,
                      0 = the headings cancel, i.e. no coherent preference)

Apples-to-apples: nothing is projected onto the cube coordinate and no distribution is
compared, so Density-FB's cube-block reward kernel confers no advantage in the measurement.
The question -- "does your value rank actions toward the goal" -- is the one that decides
control for both.
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import banks as BK
from scripts.probes.cube_bottleneck.fig_outcome_map import CM, task_goals
from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ROOT = Path("analysis/cube_bottleneck/cube-single-play-v0")
CUBE_OBS = slice(19, 22)
ORDER = ["fb_ref", "density_fb_state"]


def q_of(arm, obs, acts, goal, pool_obs):
    import jax.numpy as jnp
    from scripts.probes.cube_bottleneck.measure_probe import density_readout, fb_readout
    c, net = arm.agent.critic, arm.agent.net
    if arm.critic.startswith("density_fb"):
        return density_readout(c, net, obs, acts, goal, pool_obs)[1]
    z = np.asarray(c.map_eval_cond(net, {"goal": jnp.asarray(goal)})["z"])
    return fb_readout(c, net, obs, acts, z, pool_obs)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--states", type=int, default=600)
    ap.add_argument("--headings", type=int, default=16)
    ap.add_argument("--goal-draws", type=int, default=2)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--bare", action="store_true")
    args = ap.parse_args()

    pool_obs = dict(np.load(ROOT / "pool.npz"))["obs"]
    raw = BK.load_raw(args.env)
    obs_all = np.asarray(raw["observations"], np.float32)
    act_all = np.asarray(raw["actions"], np.float32)
    held = ((obs_all[:, BK.CONTACT_IDX] > BK.CONTACT_THRESH)
            & (np.asarray(raw["qpos"])[:, BK.CUBE_QPOS_Z] > BK.LIFT_THRESH))
    idx = np.random.default_rng(0).choice(np.flatnonzero(held), size=args.states,
                                          replace=False)
    obs_s, act_s = obs_all[idx], act_all[idx]
    cube_s = obs_s[:, CUBE_OBS][:, :2] * CM

    th = np.linspace(0, 2 * np.pi, args.headings, endpoint=False)
    dirs = np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32)

    lo, hi = np.percentile(pool_obs[:, CUBE_OBS][:, :2] * CM, [2, 98], axis=0)
    xe = np.linspace(lo[0], hi[0], args.bins + 1)
    ye = np.linspace(lo[1], hi[1], args.bins + 1)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2

    fields, goal_xy = {}, None
    for a in args.critics:
        acc = np.zeros((args.bins, args.bins, 2))
        cnt = np.zeros((args.bins, args.bins))
        for s in args.seeds:
            run = Path("exp") / f"{args.env}_{a}_flowbc_seed_{s}"
            if not (run / "params_500000.pkl").exists():
                continue
            arm = A.load_arm(run)
            goals = task_goals(arm, args.task, args.goal_draws, 20260904)
            if goal_xy is None:
                goal_xy = np.mean([g[CUBE_OBS][:2] for g in goals], axis=0) * CM
            for g in goals:
                for i in range(len(obs_s)):
                    acts = np.repeat(act_s[i][None], args.headings, axis=0)
                    acts[:, 0:2] = dirs                       # only the heading varies
                    q = q_of(arm, np.repeat(obs_s[i][None], args.headings, axis=0),
                             acts, g, pool_obs)
                    bx = np.clip(np.searchsorted(xe, cube_s[i, 0]) - 1, 0, args.bins - 1)
                    by = np.clip(np.searchsorted(ye, cube_s[i, 1]) - 1, 0, args.bins - 1)
                    acc[bx, by] += dirs[int(np.argmax(q))]
                    cnt[bx, by] += 1
            print(f"  [{a} s{s}] {len(obs_s)} states x {len(goals)} goals", flush=True)
        fields[a] = acc / np.maximum(cnt[..., None], 1)
        coh = np.linalg.norm(fields[a], axis=-1)[cnt > 0]
        print(f"  {a:20s} mean heading coherence {coh.mean():.3f}")

    cols = [a for a in args.critics if a in fields]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.6 * len(cols) + 0.6, 3.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    for ax, a in zip(axes, cols):
        ax.set_facecolor(SURFACE)
        U, V = fields[a][..., 0], fields[a][..., 1]
        mag = np.hypot(U, V)
        ax.quiver(X, Y, U, V, mag, cmap="Reds", clim=(0, 1), angles="xy",
                  scale_units="xy", scale=0.12, width=0.011, edgecolor="none")
        ax.plot(goal_xy[0], goal_xy[1], marker="*", ms=16, color="#ffd166", mec="#1a1a1a",
                mew=1.0, zorder=5)
        if not args.bare:
            ax.text(0.04, 0.02, f"coherence {mag.mean():.2f}", transform=ax.transAxes,
                    fontsize=8, color=INK, va="bottom", ha="left",
                    bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=1.5))
        ax.set_title(ARM_LABEL[a], fontsize=10.5, color=ARM_COLOR[a], fontweight="bold",
                     pad=6)
        ax.set_xlabel("table x (cm)", fontsize=8, color=INK2)
        ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
        for sp in ax.spines.values():
            sp.set_color("#d6d5d1")
        ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    axes[0].set_ylabel("table y (cm)", fontsize=8, color=INK2)
    if not args.bare:
        fig.suptitle("Which way does the critic want to push the cube?", fontsize=11.5,
                     color=INK, y=1.06)
        fig.text(0.5, 1.005, "Preferred heading at held states.  Arrow length = how "
                 "consistently that heading is preferred.  * = the goal.",
                 fontsize=8.5, color=INK2, ha="center")
    stem = "fig_action_field" + ("_bare" if args.bare else "")
    out = ROOT / "figures" / stem
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
