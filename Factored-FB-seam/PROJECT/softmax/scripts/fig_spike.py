"""Draft paper figure: the predicted future-state distribution from one start state, in the
task coordinates (cube xy for cube-single, ant xy for antmaze), for the true data occupancy and
for the best rank-d linear (FB) and rank-d softmax / log-bilinear (Density-FB) fits.
Usage: python scripts/fig_spike.py --rank 50 --out figures/fig_spike.png"""
import argparse, numpy as np, jax, jax.numpy as jnp, optax
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=50); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--out", default="figures/fig_spike.png"); ap.add_argument("--row_pick", default="median")

# (label, saved occupancy matrix, columns of the cluster centre to plot, axis labels)
ENVS_TASK = [("cube-single\n(cube xy)", "results/ogbench/cube-single-play-v0_task_K512_M.npz", (0, 1), "cube x", "cube y"),
             ("antmaze-medium\n(ant xy)", "results/ogbench/antmaze-medium-navigate-v0_task_K512_M.npz", (0, 1), "ant x", "ant y")]
ENVS_FULL = [("cube-single\nfull state, cube xy shown", "results/ogbench/cube-single-play-v0_full_K512_M.npz", (19, 20), "cube x", "cube y"),
             ("antmaze-medium\nfull state, ant xy shown", "results/ogbench/antmaze-medium-navigate-v0_full_K512_M.npz", (0, 1), "ant x", "ant y"),
             ("Bridge-V2 (real robot)\nproprio, gripper xy shown", "results/external/bridge_full_K512_hm_M.npz", (0, 1), "x", "y")]
ap.add_argument("--coords", default="full", choices=["full", "task"])
a = ap.parse_args()
ENVS = ENVS_FULL if a.coords == "full" else ENVS_TASK

def fit_log(M, rho, w, r, steps, seed=0):
    K = M.shape[0]; Mj = jnp.asarray(M); logrho = jnp.log(jnp.maximum(jnp.asarray(rho), 1e-12)); wj = jnp.asarray(w)
    kf, kb = jax.random.split(jax.random.PRNGKey(seed))
    p = dict(F=0.1 * jax.random.normal(kf, (K, r)), B=0.1 * jax.random.normal(kb, (K, r)))
    opt = optax.adam(3e-2); st = opt.init(p)
    def loss_fn(p):
        return -(wj[:, None] * Mj * jax.nn.log_softmax(p["F"] @ p["B"].T + logrho[None, :], -1)).sum()
    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss_fn)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    for _ in range(steps): p, st, _ = step(p, st)
    return np.asarray(jax.nn.softmax(p["F"] @ p["B"].T + logrho[None, :], -1), np.float64)

def fit_lin(M, rho, r):
    sr = np.sqrt(rho); A = (M / np.maximum(sr[:, None], 1e-12)) * sr[None, :]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vt[:r]) * sr[:, None] / np.maximum(sr[None, :], 1e-12)

fig, axes = plt.subplots(len(ENVS), 4, figsize=(13.5, 3.2 * len(ENVS)), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2], wspace=0.08))
C_LIN, C_LOG, C_TRUE = "#1baf7a", "#2a78d6", "#333333"
for ei, (name, path, (ix, iy), xl, yl) in enumerate(ENVS):
    z = np.load(path); M, rho, cen, mu, sd = z["M"], z["rho"], z["centers"], z["mu"], z["sd"]
    M = M / M.sum(1, keepdims=True); w = rho / rho.sum()
    Mlin = fit_lin(M, rho, a.rank); Mlog = fit_log(M, rho, w, a.rank, a.steps)
    X = cen * sd + mu
    ess = 1.0 / (M ** 2).sum(1)
    # start state = the rho-weighted median-ESS row: a typical *visited* state, not a typical cluster
    order = np.argsort(ess); cw = np.cumsum(w[order])
    i = order[np.searchsorted(cw, 0.5)] if a.row_pick == "median" else int(a.row_pick)
    rows = [("data: where the policy goes", M[i]), (f"linear rank-{a.rank} (FB)", Mlin[i]), (f"softmax rank-{a.rank} (Density-FB)", Mlog[i])]
    vmax = M[i].max()
    for ci, (title, row) in enumerate(rows):
        ax = axes[ei, ci]
        o = np.argsort(np.abs(row)); col = np.sign(row) * np.sqrt(np.abs(row) / vmax)   # signed sqrt scale
        sc = ax.scatter(X[o, ix], X[o, iy], c=col[o], cmap="RdBu_r", vmin=-1, vmax=1, s=26, linewidths=0)
        ax.scatter(X[i, ix], X[i, iy], marker="o", facecolors="none", edgecolors="k", s=140, linewidths=1.4)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel(name, fontsize=10)
        tv = 0.5 * np.abs(M[i] - row).sum(); neg = np.clip(-row, 0, None).sum()
        if ci > 0: ax.text(0.02, 0.02, f"TV {tv:.2f}" + (f"\nneg. mass {neg:.2f}" if neg > 1e-3 else ""), transform=ax.transAxes, fontsize=8, va="bottom")
    ax = axes[ei, 3]; o = np.argsort(-M[i])[:60]; xs = np.arange(len(o))
    ax.bar(xs, M[i][o], color=C_TRUE, alpha=0.35, width=1.0, label="data")
    ax.plot(xs, Mlin[i][o], color=C_LIN, lw=1.8, label="linear (FB)")
    ax.plot(xs, Mlog[i][o], color=C_LOG, lw=1.8, label="softmax (Density-FB)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("future states, sorted by true mass", fontsize=9); ax.set_ylabel("mass", fontsize=9)
    ax.set_title(f"top-60 futures of the same start (ESS {ess[i]:.0f} / {M.shape[0]} states)", fontsize=9)
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    if ei == 0: ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle(f"Where will the policy be?  discounted future-state distribution from one start state (○); rank {a.rank}; colour ∝ √mass, blue = negative mass", fontsize=11)
fig.subplots_adjust(hspace=0.3); fig.savefig(a.out, dpi=160, bbox_inches="tight"); print("wrote", a.out)
