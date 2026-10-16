"""Simple version of the spike figure.
Left three panels: the discounted future-state distribution averaged over start states, with
future states ranked by their true probability (so panel shows the typical row shape), for the
data, the best rank-d linear fit (FB's critic class) and the best rank-d softmax fit (Density-FB's).
Right panel: rank needed by each class to reach TV <= 0.1 on every dataset, ordered by peakedness.
Usage: python scripts/fig_spike_simple.py --rank 50 --out figures/fig_spike_simple.png"""
import argparse, json, numpy as np, jax, jax.numpy as jnp, optax
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=50); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--top", type=int, default=25); ap.add_argument("--out", default="figures/fig_spike_simple.png"); a = ap.parse_args()
C_LIN, C_LOG, C_DATA = "#1baf7a", "#2a78d6", "#b0b0b0"
PROFILES = [("cube-single (sim manipulation)", "results/ogbench/cube-single-play-v0_full_K512_M.npz"),
            ("Bridge-V2 (real robot)", "results/external/bridge_full_K512_hm_M.npz"),
            ("antmaze-medium (sim locomotion)", "results/ogbench/antmaze-medium-navigate-v0_full_K512_M.npz")]
RANKS = [("antmaze-medium", "results/ogbench/antmaze-medium-navigate-v0_full_K512.json"), ("antmaze-large", "results/ogbench/antmaze-large-navigate-v0_full_K512.json"),
         ("puzzle-3x3", "results/ogbench/puzzle-3x3-play-v0_full_K512.json"), ("cube-single", "results/ogbench/cube-single-play-v0_full_K512.json"),
         ("scene", "results/ogbench/scene-play-v0_full_K512.json"), ("cube-double", "results/ogbench/cube-double-play-v0_full_K512.json"),
         ("DROID", "results/external/droid100_full_K128_hm.json"), ("Bridge-V2", "results/external/bridge_full_K512_hm.json"),
         ("LIBERO-90", "results/external/libero90_full_K512_hm.json"), ("UR5", "results/external/ur5_full_K512_hm.json")]

def fit_log(M, rho, w, r, steps, seed=0):
    K = M.shape[0]; Mj = jnp.asarray(M); logrho = jnp.log(jnp.maximum(jnp.asarray(rho), 1e-12)); wj = jnp.asarray(w)
    kf, kb = jax.random.split(jax.random.PRNGKey(seed))
    p = dict(F=0.1 * jax.random.normal(kf, (K, r)), B=0.1 * jax.random.normal(kb, (K, r)))
    opt = optax.adam(3e-2); st = opt.init(p)
    def loss_fn(p): return -(wj[:, None] * Mj * jax.nn.log_softmax(p["F"] @ p["B"].T + logrho[None, :], -1)).sum()
    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss_fn)(p); up, st = opt.update(g, st, p); return optax.apply_updates(p, up), st, l
    for _ in range(steps): p, st, _ = step(p, st)
    return np.asarray(jax.nn.softmax(p["F"] @ p["B"].T + logrho[None, :], -1), np.float64)

def fit_lin(M, rho, r):
    sr = np.sqrt(rho); A = (M / np.maximum(sr[:, None], 1e-12)) * sr[None, :]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vt[:r]) * sr[:, None] / np.maximum(sr[None, :], 1e-12)

def sorted_profile(Mhat, M, w, top):
    o = np.argsort(-M, axis=1)[:, :top]
    return (w[:, None] * np.take_along_axis(Mhat, o, 1)).sum(0)

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35], wspace=0.28))
for ax, (name, path) in zip(axes[:3], PROFILES):
    z = np.load(path); M, rho = z["M"], z["rho"]; M = M / M.sum(1, keepdims=True); w = rho / rho.sum()
    Mlin = fit_lin(M, rho, a.rank); Mlog = fit_log(M, rho, w, a.rank, a.steps)
    xs = np.arange(1, a.top + 1)
    pd, pl, pg = (sorted_profile(X, M, w, a.top) for X in (M, Mlin, Mlog))
    ax.bar(xs, pd, color=C_DATA, width=0.85, label="data")
    ax.plot(xs, pg, "-o", color=C_LOG, lw=2.2, ms=5, label="softmax (Density-FB)")
    ax.plot(xs, pl, "-s", color=C_LIN, lw=2.2, ms=5, label="linear (FB)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(name); ax.set_xlabel("future states, ranked by true probability" if name.startswith("Bridge") else "")
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ess = float((w / (M ** 2).sum(1)).sum())
    ax.text(0.97, 0.95, f"typical start reaches\n{ess:.0f} of {M.shape[0]} states", transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#444")
axes[0].set_ylabel("probability of future state"); axes[0].legend(frameon=False, loc="center right", fontsize=11)

ax = axes[3]; names, rl, rg, ess = [], [], [], []
for name, path in RANKS:
    d = json.load(open(path)); K = d["stats"]["K"]
    def need(fam):
        rs = sorted(int(r) for r in d[fam]); hit = [r for r in rs if d[fam][str(r)]["tv"] <= 0.1]
        return (hit[0], False) if hit else (rs[-1] * 2, True)
    names.append(name); rl.append(need("linear")); rg.append(need("log")); ess.append(d["stats"]["ess_over_K"])
o = np.argsort(ess)[::-1]; names = [names[i] for i in o]; rl = [rl[i] for i in o]; rg = [rg[i] for i in o]
x = np.arange(len(names)); bw = 0.38
for j, (vals, c, lab) in enumerate([(rl, C_LIN, "linear (FB)"), (rg, C_LOG, "softmax (Density-FB)")]):
    ax.bar(x + (j - 0.5) * bw, [v for v, _ in vals], bw, color=c, label=lab)
    for xi, (v, cens) in zip(x + (j - 0.5) * bw, vals):
        if cens: ax.text(xi, v * 1.08, ">", ha="center", va="bottom", fontsize=12, color=c, fontweight="bold")
ax.set_yscale("log", base=2); ax.set_yticks([8, 16, 32, 64, 128, 256, 512]); ax.set_yticklabels([8, 16, 32, 64, 128, 256, 512])
ax.axhline(50, color="k", lw=0.8, ls="--"); ax.text(len(names) - 0.5, 50, " d = 50", va="center", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_ylabel("rank needed to fit the future (TV ≤ 0.1)")
ax.set_title("diffuse  →  peaked"); ax.legend(frameon=False, loc="upper left", fontsize=11)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(a.out, dpi=160, bbox_inches="tight"); print("wrote", a.out)
