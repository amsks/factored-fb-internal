"""Figures for PROJECT/softmax. Usage: python scripts/plot_results.py"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = dict(log="#2a78d6", nmf="#eb6834", lin="#1baf7a")   # fixed categorical slots: 1 blue, 2 orange, 3 aqua
NAME = dict(log="log-bilinear softmax (Density-FB)", nmf="nonneg rank-d mixture (ratio family, best case)", lin="linear rank-d (FB, weighted SVD)")
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})

def load(pattern): return {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(pattern))}

# --- Fig 1: TV vs rank, OGBench K=512, task coords and full obs
res = load("results/ogbench/*_K512.json")
envs = ["cube-single-play-v0", "cube-double-play-v0", "scene-play-v0", "puzzle-3x3-play-v0", "antmaze-medium-navigate-v0", "antmaze-large-navigate-v0"]
for rep in ["task", "full"]:
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8), sharey=True)
    for ax, env in zip(axes, envs):
        j = res.get(f"{env}_{rep}_K512");
        if j is None: continue
        ranks = j["ranks"]
        for m, key in [("lin", "linear"), ("nmf", "nmf"), ("log", "log")]:
            if not j.get(key): continue
            ax.plot(ranks, [j[key][str(r)]["tv"] for r in ranks], color=C[m], lw=2, marker="o", ms=4, label=NAME[m])
        ax.axvline(50, color="#52514e", lw=0.8, ls=":"); ax.set_xscale("log", base=2)
        s = j["stats"]; ax.set_title(f"{env.replace('-play-v0','').replace('-navigate-v0','')}\nESS/K={s['ess_over_K']:.2f}  P1self={s['P1_self']:.2f}", fontsize=8)
        ax.set_xlabel("rank d")
    axes[0].set_ylabel("TV( M_i , M̂_i ), ρ-weighted"); axes[0].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"OGBench, K=512 clusters, γ=0.99, {'task coordinates (hit-spec columns)' if rep=='task' else 'full observation'}; dotted = d=50 used by FB and Density-FB", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"figures/fig1_tv_vs_rank_{rep}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# --- Fig 2: gap at rank 64 vs peakedness, all datasets
pts = []
for pat, mk in [("results/ogbench/*_K512.json", "o"), ("results/ogbench/*_K1024.json", "s"), ("results/external/*.json", "^"), ("results/gamma/*.json", "x")]:
    for k, j in load(pat).items():
        s = j["stats"]; r = "64" if "64" in j["linear"] else str(j["ranks"][-1])
        pts.append((s["ess_over_K"], j["linear"][r]["tv"] - j["log"][r]["tv"], mk, k, "external" in pat))
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for mk, lab in [("o", "OGBench K=512"), ("s", "OGBench K=1024"), ("^", "real-robot / LIBERO (proprio)"), ("x", "OGBench γ sweep")]:
    P = [p for p in pts if p[2] == mk]
    ax.scatter([p[0] for p in P], [p[1] for p in P], marker=mk, s=28, color="#2a78d6" if mk != "^" else "#eb6834", label=lab, alpha=0.85)
for p in pts:
    if p[4]: ax.annotate(p[3].split("_")[0], (p[0], p[1]), fontsize=6, xytext=(3, 2), textcoords="offset points", color="#52514e")
ax.set_xscale("log"); ax.set_xlabel("peakedness: median row ESS of the discounted occupancy / K   (small = peaked)")
ax.set_ylabel("TV gain of log model over linear, rank 64")
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig("figures/fig2_gap_vs_peakedness.png", dpi=150); plt.close(fig)

# --- Fig 3: rank needed for TV<=0.1 vs K (K scaling)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
def eps_rank(j, key, eps=0.1):
    for r in j["ranks"]:
        if j[key][str(r)]["tv"] <= eps: return r
    return j["ranks"][-1] * 2  # censored
for i, env in enumerate(envs):
    for rep, ls in [("full", "-"), ("task", "--")]:
        xs, yl, yg = [], [], []
        for K in [512, 1024]:
            j = res.get(f"{env}_{rep}_K512") if K == 512 else load(f"results/ogbench/{env}_{rep}_K1024.json").get(f"{env}_{rep}_K1024")
            if j is None: continue
            xs.append(K); yl.append(eps_rank(j, "linear")); yg.append(eps_rank(j, "log"))
        if len(xs) == 2:
            ax.plot(xs, yl, color=C["lin"], ls=ls, marker="o", ms=3, lw=1.2, alpha=0.8)
            ax.plot(xs, yg, color=C["log"], ls=ls, marker="o", ms=3, lw=1.2, alpha=0.8)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2); ax.set_xlabel("number of state clusters K"); ax.set_ylabel("rank needed for TV ≤ 0.1 (≥512 = censored)")
ax.plot([], [], color=C["lin"], label="linear (FB)"); ax.plot([], [], color=C["log"], label="log-bilinear (Density-FB)"); ax.plot([], [], color="#52514e", ls="-", label="full obs"); ax.plot([], [], color="#52514e", ls="--", label="task coords")
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig("figures/fig3_rank_vs_K.png", dpi=150); plt.close(fig)

# --- Fig 4: gamma sweep
G = load("results/gamma/*.json"); G.update({k: v for k, v in res.items()})
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for ax, rep in zip(axes, ["task", "full"]):
    for env, col in zip(["cube-single-play-v0", "scene-play-v0", "antmaze-medium-navigate-v0"], ["#2a78d6", "#eb6834", "#1baf7a"]):
        gs, gaps, ess = [], [], []
        for g, key in [(0.9, f"{env}_{rep}_g0.9"), (0.95, f"{env}_{rep}_g0.95"), (0.99, f"{env}_{rep}_K512"), (0.995, f"{env}_{rep}_g0.995")]:
            j = G.get(key)
            if j: gs.append(1 / (1 - g)); gaps.append(j["linear"]["64"]["tv"] - j["log"]["64"]["tv"]); ess.append(j["stats"]["ess_over_K"])
        ax.plot(gs, gaps, color=col, marker="o", ms=4, lw=2, label=env.replace("-play-v0", "").replace("-navigate-v0", ""))
    ax.set_xscale("log"); ax.set_xlabel("effective horizon 1/(1-γ) [steps]"); ax.set_title(f"{rep} coordinates", fontsize=9)
axes[0].set_ylabel("TV gain of log over linear, rank 64"); axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("figures/fig4_gamma_sweep.png", dpi=150); plt.close(fig)
print("figures written:", sorted(os.listdir("figures")))
