"""Reach-map figure: each critic's predicted future gripper position relative to the cube, at pre-grasp states.

Maps are averaged over the P1 bank in gripper-minus-cube coordinates and shown next to the
uniform-pool prior.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck.fig_outcome_map import (CM, MASS_CMAP, task_goals)
from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ANALYSIS = Path("analysis/cube_bottleneck")
CUBE_OBS = slice(19, 22)
EFF_OBS = slice(12, 15)
ORDER = ["fb_ref", "density_fb_state"]
NEAR_CM = 5.0          # radius for "gripper at the cube"


def batched_mass(arm, obs_row, act_row, goals, pool_obs):
    """Return [G, M] predicted pool mass for one state under G goals, and FB's negative fraction.

    FB's signed measure is clipped at zero and renormalised per row.
    """
    import jax.numpy as jnp

    from scripts.probes.cube_bottleneck.measure_probe import density_readout, fb_readout

    c, net = arm.agent.critic, arm.agent.net
    G = len(goals)
    obs = np.repeat(np.asarray(obs_row, np.float32)[None], G, axis=0)
    acts = np.repeat(np.asarray(act_row, np.float32)[None], G, axis=0)
    if arm.critic.startswith("density_fb"):
        # density_readout takes one goal at a time
        m = np.stack([density_readout(c, net, obs[:1], acts[:1], g, pool_obs)[0][0]
                      for g in goals])
        neg = 0.0
    else:
        z = np.asarray(c.map_eval_cond(net, {"goal": jnp.asarray(goals)})["z"])   # [G, d]
        M, _ = fb_readout(c, net, obs, acts, z, pool_obs)
        m, neg = M, float((M < 0).mean())
    p = np.maximum(m, 0.0)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-12), neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--phase", default="P1", choices=["P1", "P2"],
                    help="P1 = cube untouched, arm away (the pre-grasp state); P2 = the "
                         "contact decision points.")
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--goal-draws", type=int, default=8)
    ap.add_argument("--goal-seed", type=int, default=20260904)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = ANALYSIS / args.env
    banks = dict(np.load(root / "phase_banks.npz"))
    obs_b = banks[f"{args.phase}_obs"]
    act_b = banks[f"{args.phase}_action"]
    pool_obs = dict(np.load(root / "pool.npz"))["obs"]

    # Gripper position relative to the cube, per pool row
    off = (pool_obs[:, EFF_OBS][:, :2] - pool_obs[:, CUBE_OBS][:, :2]) * CM     # [M, 2] cm
    lim = float(np.percentile(np.abs(off), 98))
    xe = np.linspace(-lim, lim, args.bins + 1)
    ye = np.linspace(-lim, lim, args.bins + 1)
    ext = [-lim, lim, -lim, lim]
    near = np.linalg.norm(off, axis=1) < NEAR_CM

    def to_grid(w):
        g, _, _ = np.histogram2d(off[:, 0], off[:, 1], bins=[xe, ye], weights=w)
        return g

    # Prior panel: uniform weight over the pool
    prior = np.full(len(pool_obs), 1.0 / len(pool_obs))
    panels = {"prior": to_grid(prior)}
    summary = {"prior": (float(prior[near].sum()),
                         float(np.exp(-(prior * np.log(prior)).sum())), 0.0)}

    for a in args.critics:
        acc, nears, neffs, negs = None, [], [], []
        for s in args.seeds:
            run = Path("exp") / f"{args.env}_{a}_flowbc_seed_{s}"
            if not (run / "params_500000.pkl").exists():
                continue
            arm = A.load_arm(run)
            goals = task_goals(arm, args.task, args.goal_draws, args.goal_seed)
            for i in range(len(obs_b)):
                p, neg = batched_mass(arm, obs_b[i], act_b[i], goals, pool_obs)
                pm = p.mean(0)                       # average over goal draws
                acc = pm if acc is None else acc + pm
                nears.append(float(pm[near].sum()))
                nz = pm[pm > 0]
                neffs.append(float(np.exp(-(nz * np.log(nz)).sum())))
                negs.append(neg)
            print(f"  [{a} s{s}] {len(obs_b)} {args.phase} states", flush=True)
        assert acc is not None, f"no checkpoints for {a}"
        acc = acc / max(len(nears), 1)
        panels[a] = to_grid(acc)
        summary[a] = (float(np.mean(nears)), float(np.mean(neffs)), float(np.mean(negs)))

    cols = ["prior"] + [a for a in args.critics if a in panels]
    titles = {"prior": "dataset prior\n(uniform over the pool)"}
    titles.update({a: ARM_LABEL[a] for a in args.critics})
    colours = {"prior": INK2}
    colours.update({a: ARM_COLOR[a] for a in args.critics})

    vmax = float(max(panels[c].max() for c in cols))
    fig, axes = plt.subplots(1, len(cols), figsize=(2.7 * len(cols) + 0.9, 3.1),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    for ax, c in zip(axes, cols):
        ax.set_facecolor(SURFACE)
        im = ax.imshow(panels[c].T, origin="lower", extent=ext, aspect="equal",
                       cmap=MASS_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
        # The cube is the origin, and the NEAR_CM ring is the "gripper at the cube" test.
        ax.plot(0, 0, marker="+", ms=11, color="#0b0b0b", mew=1.8, zorder=5)
        ax.add_patch(plt.Circle((0, 0), NEAR_CM, fill=False, ec="#0b0b0b", lw=1.0,
                                ls=(0, (3, 2)), zorder=4))
        m_near, n_eff, neg = summary[c]
        if not args.bare:
            lbl = f"mass within {NEAR_CM:.0f}cm: {m_near:.2f}\nN_eff {n_eff:.0f}"
            if neg:
                lbl += f"\n{100 * neg:.0f}% neg. clipped"
            ax.text(0.04, 0.02, lbl, transform=ax.transAxes, fontsize=7.5, color=INK,
                    va="bottom", ha="left",
                    bbox=dict(fc=SURFACE, ec="none", alpha=0.82, pad=1.6))
        ax.set_title(titles[c], fontsize=9.5, color=colours[c],
                     fontweight="bold" if c != "prior" else "normal", pad=6)
        ax.set_xlabel("gripper x − cube x  (cm)", fontsize=8, color=INK2)
        for sp in ax.spines.values():
            sp.set_color("#d6d5d1")
        ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    axes[0].set_ylabel("gripper y − cube y  (cm)", fontsize=8, color=INK2)

    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    cb.outline.set_edgecolor("#d6d5d1")
    if not args.bare:
        cb.set_label("predicted probability", fontsize=8, color=INK2)
        fig.suptitle("Before the grasp: does the critic predict the gripper reaching the "
                     "cube?", fontsize=11, color=INK, y=1.06)
        fig.text(0.5, 1.005, f"Cube at the origin (+), dashed ring = {NEAR_CM:.0f} cm.  "
                 f"Averaged over {len(obs_b)} {args.phase} states x {len(args.seeds)} seeds.",
                 fontsize=8.5, color=INK2, ha="center")

    stem = f"fig_reach_map_{args.phase}" + ("_bare" if args.bare else "")
    out = Path(args.out) if args.out else root / "figures" / stem
    out.parent.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png")
    for c in cols:
        m_near, n_eff, neg = summary[c]
        print(f"  {c:18s} mass within {NEAR_CM:.0f}cm {m_near:.4f} | N_eff {n_eff:7.1f} | "
              f"negative {100 * neg:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
