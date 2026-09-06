"""Fig: "where will the cube end up?" -- the predicted successor measure over the table.

This is the figure the value maps could not be. A value over states answers "how good is
this state", which is not what makes a method work (control is a ranking over ACTIONS at a
state, and measured here Density-FB scores AUC .56 on "is this state near the goal" while
solving cube-single at .91). What separates the two families is the predicted OUTCOME
DISTRIBUTION itself:

    FB          M(s,a,.) = F(s,a,z)^T B(.)     signed, unnormalised, rank <= z_dim,
                                               fit by squared regression whose target is
                                               dominated by the "cube unchanged" atom
    Density-FB  w(s,a,.) = softmax(F.B / tau)  a probability distribution by construction,
                                               so mass is competed for and the rare
                                               moved-cube outcomes resolve against each other

Measured over the frozen 1024-row pool at 32 held states x 5 goals: 48.9% of FB's entries
are NEGATIVE, and after clipping and renormalising its effective support is 213 of 1024
rows with 1.8% on the top outcome. Density-FB: 0% negative, support 16.4, top-1 mass 28.6%.
Thirteen times more concentrated, and not a tuning gap -- FB's parameterisation cannot
represent a normalised distribution at all.

Each panel bins the pool's rows by the cube position of the state they represent and sums
the predicted mass per bin, so the picture is "probability the cube ends up here". The
support is 1024 discrete real states, not a continuum; the binning is a summary of them.

NOT SHOWN, and the caption must not imply it: the truth. Which of these predictions is
right needs rollouts under each arm's OWN policy (FIGURE-PLAN Fig 1's third column); this
figure claims concentration, not correctness.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck.figures import ARM_COLOR, ARM_LABEL, INK, INK2, SURFACE

ANALYSIS = Path("analysis/cube_bottleneck")
CUBE_OBS = slice(19, 22)
CM = 10.0
ORDER = ["fb_ref", "density_fb_state"]

# Sequential single hue: probability is a magnitude, so pale -> red, red = the cube most
# likely ends up here. Keeps the "red = the wanted thing" reading of the value figures.
MASS_CMAP = LinearSegmentedColormap.from_list(
    "mass", ["#fbf7f4", "#fbd5c4", "#f5865c", "#d1341c", "#6d1000"])


def predicted_mass(arm, obs_row, act_row, goal, pool_obs):
    """[M] predicted mass over the pool, plus (% negative before clipping, N_eff, top-1).

    FB's measure is signed, so it must be clipped and renormalised before it can be read as
    a distribution at all -- that clip is not a plotting convenience, it is the finding, and
    the fraction discarded is reported per panel. Density-FB's weights are already a
    distribution and are used untouched.
    """
    import jax.numpy as jnp

    from scripts.probes.cube_bottleneck.measure_probe import density_readout, fb_readout

    c, net = arm.agent.critic, arm.agent.net
    ob, ac = np.asarray(obs_row)[None], np.asarray(act_row)[None]
    if arm.critic.startswith("density_fb"):
        w, _ = density_readout(c, net, ob, ac, goal, pool_obs)
        m, neg = w[0], 0.0
    else:
        z = np.asarray(c.map_eval_cond(net, {"goal": jnp.asarray(goal)})["z"])
        M, _ = fb_readout(c, net, ob, ac, z, pool_obs)
        m = M[0]
        neg = float((m < 0).mean())
    p = np.maximum(m, 0.0)
    p = p / max(p.sum(), 1e-12)
    nz = p[p > 0]
    n_eff = float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else 0.0
    return p, neg, n_eff, float(p.max())


def task_goals(arm, task_id: int, n: int, seed: int) -> np.ndarray:
    """`n` goal rows for `task_id`, built from the DATASET rather than from the env.

    The env cannot supply a reproducible goal. `task_id` pins the goal's cube target, but
    the rest of the goal row -- the arm pose -- is redrawn on every reset from an RNG that
    neither `np.random.seed` nor gymnasium's `reset(seed=...)` controls: measured, the same
    eight seeded draws summed to 37.93 in one process and 40.23 in the next. Since both
    critics read the whole goal row, that made the figure non-reproducible and, worse, gave
    each arm a different goal.

    So the cube TARGET is taken from the env (that part is deterministic) and the goal rows
    are then drawn from real dataset states whose cube already sits within the task's own
    4 cm success radius of it. Reproducible from (dataset, seed), on-manifold, and it
    marginalises the arm pose over the poses the data actually visits at that target rather
    than over an arbitrary simulator draw.
    """
    from scripts.probes.cube_bottleneck import banks as BK

    _o, info = arm.env.reset(options=dict(task_id=task_id, render_goal=False))
    target = np.asarray(info["goal"], np.float32)[CUBE_OBS][:2]

    raw = BK.load_raw(arm.env_name)
    obs = np.asarray(raw["observations"], np.float32)
    near = np.flatnonzero(np.linalg.norm(obs[:, CUBE_OBS][:, :2] - target, axis=1) < 0.4)
    assert len(near) >= n, f"only {len(near)} dataset rows within 4cm of task {task_id}"
    sel = np.random.default_rng(seed).choice(near, size=n, replace=False)
    return obs[np.sort(sel)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=ORDER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=4, help="goal held fixed across columns")
    ap.add_argument("--states", nargs="+", type=int, default=[3, 11, 27],
                    help="held-bank rows to show, one per column")
    ap.add_argument("--bins", type=int, default=14)
    ap.add_argument("--goal-seed", type=int, default=20260904,
                    help="numpy seed pinned immediately before each arm's env.reset, so "
                         "every arm is scored against the SAME task goal.")
    ap.add_argument("--goal-draws", type=int, default=16,
                    help="goal rows drawn per task; the map is their average, which "
                         "marginalises the nuisance arm pose in the goal vector.")
    ap.add_argument("--bare", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = ANALYSIS / args.env
    bank = dict(np.load(root / "held_bank.npz"))
    pool_obs = dict(np.load(root / "pool.npz"))["obs"]
    pool_xy = pool_obs[:, CUBE_OBS][:, :2]

    lo, hi = np.percentile(pool_xy, [1, 99], axis=0)
    xe = np.linspace(lo[0], hi[0], args.bins + 1)
    ye = np.linspace(lo[1], hi[1], args.bins + 1)
    ext = [xe[0] * CM, xe[-1] * CM, ye[0] * CM, ye[-1] * CM]

    arms = [a for a in args.critics
            if (Path("exp") / f"{args.env}_{a}_flowbc_seed_{args.seed}").exists()]
    assert arms, f"no runs for {args.critics} seed {args.seed}"

    grids, stats, goal_xy = {}, {}, None
    for a in arms:
        arm = A.load_arm(Path("exp") / f"{args.env}_{a}_flowbc_seed_{args.seed}")
        # The task goal is NOT a deterministic function of task_id: OGBench draws it from
        # the global numpy RNG, and `load_arm` reseeds that per run. So without this the
        # arms are compared on DIFFERENT goals and the figure changes between renders (it
        # did: N_eff moved 171/186/228 -> 166/208/248 across two identical runs). Reseeding
        # immediately before the reset pins one goal for every arm.
        # A task_id pins the goal's CUBE target but not the rest of the goal vector: the
        # arm pose in the goal row is redrawn every reset, and both critics read the whole
        # row. Measured, one redraw moved Density-FB's top outcome from 0.40 to 0.52 and its
        # argmax from pool row 907 to 114 with the cube target unchanged. So a single draw
        # would be an arbitrary nuisance pose dressed up as "the goal": average the
        # prediction over `--goal-draws` of them instead, and seed so it is reproducible.
        goals = task_goals(arm, args.task, args.goal_draws, args.goal_seed)
        # Dataset-drawn goals sit WITHIN the 4cm success radius of the target rather than
        # exactly on it, so the star is their centroid and the spread is bounded by 0.4.
        gxy = np.mean([g[CUBE_OBS][:2] for g in goals], axis=0)
        assert np.max(np.linalg.norm([g[CUBE_OBS][:2] - gxy for g in goals], axis=1)) < 0.8, \
            "goal rows are not clustered on one cube target"
        if goal_xy is not None:
            assert np.allclose(goal_xy, gxy), (
                "arms drew different goals; the comparison would not be like-for-like")
        goal_xy = gxy
        for si in args.states:
            # Average the DISTRIBUTION over draws; average the SCALARS per draw. N_eff of a
            # mixture is not the mean N_eff -- mixing always spreads it -- so quoting the
            # mixture's N_eff would overstate every method's diffuseness.
            ps, negs, neffs, tops = [], [], [], []
            for goal in goals:
                p, neg, n_eff, top1 = predicted_mass(arm, bank["obs"][si],
                                                     bank["action"][si], goal, pool_obs)
                ps.append(p), negs.append(neg), neffs.append(n_eff), tops.append(top1)
            p = np.mean(ps, axis=0)
            g, _, _ = np.histogram2d(pool_xy[:, 0], pool_xy[:, 1], bins=[xe, ye], weights=p)
            grids[(a, si)] = g
            stats[(a, si)] = (float(np.mean(negs)), float(np.mean(neffs)),
                              float(np.mean(tops)))
        print(f"  [{a}] {[f'{stats[(a, s)][1]:.0f}' for s in args.states]} N_eff, "
              f"{100 * stats[(a, args.states[0])][0]:.0f}% negative", flush=True)

    # One colour scale across the whole figure: the panels are probabilities on the same
    # support, so a per-panel scale would erase exactly the concentration being claimed.
    vmax = float(max(g.max() for g in grids.values()))

    fig, axes = plt.subplots(len(arms), len(args.states),
                             figsize=(2.35 * len(args.states) + 1.0, 2.5 * len(arms)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(SURFACE)

    for r, a in enumerate(arms):
        for c, si in enumerate(args.states):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            im = ax.imshow(grids[(a, si)].T, origin="lower", extent=ext, aspect="equal",
                           cmap=MASS_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
            cur = bank["obs"][si][CUBE_OBS][:2] * CM
            ax.plot(cur[0], cur[1], marker="o", ms=8, mfc="none", mec="#0b0b0b", mew=1.6,
                    zorder=5)
            ax.plot(goal_xy[0] * CM, goal_xy[1] * CM, marker="*", ms=15, color="#ffd166",
                    mec="#1a1a1a", mew=1.0, zorder=5)
            neg, n_eff, top1 = stats[(a, si)]
            if not args.bare:
                # top-1 matters on the shared scale: FB's row is near-invisible against
                # Density-FB's peak, and without the number a reader reads "predicts
                # nothing" where the truth is "predicts a thin spread over the whole table".
                lbl = (f"N_eff {n_eff:.0f}   peak {top1:.3f}"
                       + (f"\n{100 * neg:.0f}% neg. clipped" if neg else ""))
                ax.text(0.04, 0.02, lbl, transform=ax.transAxes, fontsize=7, color=INK,
                        va="bottom", ha="left",
                        bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1.4))
            for s in ax.spines.values():
                s.set_color("#d6d5d1")
            ax.tick_params(colors=INK2, labelsize=7.5, length=3)
            if r == 0:
                ax.set_title(f"state {si}", fontsize=9, color=INK, pad=6)
            if c == 0:
                ax.set_ylabel("table y (cm)", fontsize=8, color=INK2)
                ax.text(-0.72, 0.5, ARM_LABEL[a], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10.5, fontweight="bold",
                        color=ARM_COLOR[a])
            if r == len(arms) - 1:
                ax.set_xlabel("table x (cm)", fontsize=8, color=INK2)

    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cb.ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    cb.outline.set_edgecolor("#d6d5d1")
    if not args.bare:
        cb.set_label("predicted probability the cube ends up here", fontsize=8, color=INK2)
        fig.suptitle("Where will the cube end up?", fontsize=11.5, color=INK, y=1.055)
        fig.text(0.5, 1.005, "o = the cube now.  * = the goal.  Held states, one goal, "
                 "predicted mass over the frozen 1024-state pool.",
                 fontsize=8.5, color=INK2, ha="center")

    stem = "fig_outcome_map" + ("_bare" if args.bare else "")
    out = Path(args.out) if args.out else root / "figures" / stem
    out.parent.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{out}.{e}", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}.pdf / .png")
    for a in arms:
        n = [stats[(a, s)][1] for s in args.states]
        t = [stats[(a, s)][2] for s in args.states]
        print(f"  {a:18s} N_eff {np.round(n, 1)}  top-1 {np.round(t, 3)}  "
              f"negative {100 * stats[(a, args.states[0])][0]:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
