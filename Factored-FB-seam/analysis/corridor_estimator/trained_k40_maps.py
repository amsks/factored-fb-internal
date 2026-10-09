"""TD-TRAINED twin of run.py's `run_k40_panels`.

Same grid, same k, same rank, same start cell, same drawing code -- the only change is
that F and B come from TD training on a random-policy dataset instead of from the
closed-form fits (FB: rho-weighted SVD; Density-FB: rank-d softmax CE optimum).

    JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/corridor_estimator/trained_k40_maps.py

Writes PAPER/ICLR/figures/corridor/k40_trained/{exit,fail}_{true,fb,density}.{pdf,png,svg},
contrast_*, the three stitched grids, and trained_k40.csv (per-seed margins and map stats).
"""
import os, sys, time, csv
sys.argv = [sys.argv[0], "--corridor14"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import jax.numpy as jnp
from jax import random
import run as R

K, D_RANK = 40, 8
STEPS = 100_000          # 20k was under-trained; converged by 100k (diag_density_td.py)
RHO_READOUT = True       # weight the density readout by rho, as FB's readout already is
SEEDS = [0, 1, 2, 3, 4]
PANEL_SEED = 0
S0_RC = (R.CORRIDOR_ROW, 1)
OUT = R.FIG_DIR / "k40_trained"


def payload_td(k, d, seed):
    """run.py's door_payload_closed_form, with TD-trained towers."""
    P_k = R.sticky_action_matrices(k)
    P_pi = R.policy_matrix_uniform(P_k)
    rho, _, _ = R.stationary(P_pi)
    M = R.successor_measure(P_k, P_pi, R.GAMMA)

    s_idx, a_idx, sp_idx = R.generate_dataset_td(k, seed)
    s_j, a_j, sp_j = jnp.asarray(s_idx), jnp.asarray(a_idx), jnp.asarray(sp_idx)
    key_fb, key_d = random.split(random.PRNGKey(20_000 * k + seed))

    F_fb, B_fb, _ = R._train_fb_td(key_fb, s_j, a_j, sp_j, d, STEPS, R.TRAIN_LR,
                                   R.TRAIN_TAU, R.TRAIN_BATCH_T, R.TRAIN_BATCH_U, R.GAMMA)
    F_fb, B_fb = np.asarray(F_fb), np.asarray(B_fb)
    F_d, B_d, _ = R._train_density_td(key_d, s_j, a_j, sp_j, d, STEPS, R.TRAIN_LR,
                                      R.TRAIN_TAU, R.TRAIN_BATCH_T, R.TRAIN_BATCH_U, R.GAMMA)
    F_d, B_d = np.asarray(F_d), np.asarray(B_d)

    Q_fb = R.fb_readout(F_fb, B_fb, rho, R.R_ROOM)
    Q_d, p_all = R.density_readout(F_d, B_d, R.R_ROOM, R.GAMMA)
    if RHO_READOUT:
        # Training normalises its softmax over candidates DRAWN FROM THE DATA, so the
        # logits model m(u)/rho(u). FB's readout already multiplies its row by rho
        # (run_trained_sweep); this is the same correction for the categorical fit.
        w = p_all * rho[None, :]
        p_all = w / w.sum(-1, keepdims=True)
        Q_d = (1.0 / (1.0 - R.GAMMA)) * (p_all @ R.R_ROOM)
    Q_true = M @ R.R_ROOM

    gaps = {}
    for tag, Q in [("true", Q_true), ("fb", Q_fb), ("density", Q_d)]:
        g, _, _, amax = R.door_metrics({a: Q[R.ROW_D[a]] for a in R.ACTIONS})
        gaps[tag] = (g, amax == R.A_CROSS)

    pay = dict(
        a_fail_show=R.A_FAIL_SHOW, density_seed_used=seed,
        p_true_cross=(1 - R.GAMMA) * M[R.ROW_D[R.A_CROSS]],
        p_true_fail=(1 - R.GAMMA) * M[R.ROW_D[R.A_FAIL_SHOW]],
        p_fb_cross=(1 - R.GAMMA) * R.fb_row(F_fb, B_fb, R.ROW_D[R.A_CROSS]) * rho,
        p_fb_fail=(1 - R.GAMMA) * R.fb_row(F_fb, B_fb, R.ROW_D[R.A_FAIL_SHOW]) * rho,
        p_density_cross=p_all[R.ROW_D[R.A_CROSS]],
        p_density_fail=p_all[R.ROW_D[R.A_FAIL_SHOW]])
    return pay, gaps


def draw(payload):
    import matplotlib
    matplotlib.rcParams.update(R.RC_PARAMS)
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    OUT.mkdir(parents=True, exist_ok=True)
    rows = R.walk_rows(payload, S0_RC)
    col_key = {"true": "true", "FB": "fb", "Density-FB": "density"}
    clipped = {n: [np.clip(v, 0.0, None) for v in rows[n]] for n in rows}
    vmax = max(float(np.nanmax(R.smooth_upsample(v))) for vals in clipped.values() for v in vals)
    norm = PowerNorm(gamma=R.K40_POWER, vmin=0.0, vmax=vmax)
    diffs = {n: rows[n][1] - rows[n][0] for n in rows}
    vmax_abs = float(np.nanmax(np.abs(R.smooth_upsample(diffs["true"]))))

    stats = {}
    for name, cmap_name in R.COL_SPECS:
        for ridx, which in enumerate(["exit", "fail"]):
            fig = plt.figure(figsize=(2.4, 1.2), constrained_layout=True)
            ax = fig.add_subplot(111)
            vals = clipped[name][ridx]
            R.draw_toy_occupancy_panel(ax, vals, cmap_name, "", norm=norm, start_rc=S0_RC,
                                       show_room_frac=True)
            R._save_to(fig, OUT, f"{which}_{col_key[name]}", svg=True)
            stats[(col_key[name], which)] = (R.door_mass_fraction(vals),
                                             R.room_mass_fraction(vals),
                                             R.tv_half(rows[name][ridx], rows["true"][ridx]))
        fig = plt.figure(figsize=(2.4, 1.2), constrained_layout=True)
        ax = fig.add_subplot(111)
        R.draw_signed_panel(ax, diffs[name], vmax_abs, "", S0_RC)
        R._save_to(fig, OUT, f"contrast_{col_key[name]}", svg=True)
    return stats


def stitch():
    """The same grids we built for the closed-form panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    cols = [("true", "true"), ("fb", "FB"), ("density", "Density-FB")]
    combos = [("maps_exit", [("exit", "then exit")]),
              ("maps_fail", [("fail", "then fail")]),
              ("maps_exit_fail", [("exit", "then exit"), ("fail", "then fail")])]
    for stem, rws in combos:
        fig = plt.figure(figsize=(11.4, 2.15 * len(rws)), dpi=300)
        gs = fig.add_gridspec(len(rws), 3, left=0.055, right=0.995,
                              top=0.895 if len(rws) > 1 else 0.80,
                              bottom=0.075 if len(rws) > 1 else 0.04,
                              wspace=0.045, hspace=0.10)
        for r, (rkey, rlabel) in enumerate(rws):
            for c, (ckey, clabel) in enumerate(cols):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(mpimg.imread(f"{OUT}/{rkey}_{ckey}.png"), interpolation="lanczos")
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(clabel, fontsize=13, color="0.15", pad=6)
                if c == 0:
                    ax.text(-0.045, 0.55, rlabel, transform=ax.transAxes, rotation=90,
                            ha="center", va="center", fontsize=12, color="0.15")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{OUT}/{stem}.{ext}", dpi=300, facecolor="white",
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  wrote {OUT}/{stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    t0 = time.time()
    print(f"TD-trained k={K}, d={D_RANK}, {STEPS} steps, rho_readout={RHO_READOUT}, "
          f"{R.TRAIN_DATASET_N} transitions, seeds {SEEDS}")
    out_rows, panel_payload = [], None
    for sd in SEEDS:
        pay, gaps = payload_td(K, D_RANK, sd)
        if sd == PANEL_SEED:
            panel_payload = pay
        gt = gaps["true"][0]
        for tag in ("fb", "density"):
            g, is_cross = gaps[tag]
            out_rows.append(dict(k=K, d=D_RANK, seed=sd, method=f"{tag}_td",
                                 raw_gap=g, norm_gap=g / gt, argmax_is_cross=is_cross))
        print(f"  seed {sd}: true {gt:6.3f} | FB {gaps['fb'][0]:7.3f} "
              f"({gaps['fb'][0]/gt:+.3f}, prefers exit={gaps['fb'][1]}) | "
              f"Density {gaps['density'][0]:6.3f} ({gaps['density'][0]/gt:+.3f}, "
              f"prefers exit={gaps['density'][1]})   [{time.time()-t0:.0f}s]")

    stats = draw(panel_payload)
    stitch()
    with open(os.path.join(HERE, "trained_k40.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print("\nmap stats (panel seed 0)  door / room / TV-to-true")
    for kk, v in stats.items():
        print(f"  {kk[0]:>8s} {kk[1]:>4s}   {v[0]:.3f}  {v[1]:.3f}  {v[2]:.3f}")
    print(f"\ndone in {time.time()-t0:.0f}s")
