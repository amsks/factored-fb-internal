"""Reward view: what task reward does each critic SEE? (cube-single, FB vs Density-FB;
docs/notes/2026-09-20-cube-reward-view.md)

The tabular section's lemma: a linear successor critic (FB) reads a reward through its
backward features, i.e. as if the reward were replaced by its projection onto the span of
B; at a bottleneck that projected reward is the same on both futures, so the critic sees no
action gap. This probe is the cube-single analogue with the trained tdjtab 1M checkpoints.

Reward views (each critic's OWN readout of the task, no re-implementation):
  FB          r_hat(u) = B(u)^T z_r            B = impls/critics/fb.py FB._B (the choke point
                                               every B call routes through), z_r the
                                               reward-inferred index (FB.infer_cond)
  Density-FB  r_z(u) = (B_xi(u) - mu_hat)^T z / sqrt(d)   impls/critics/density_fb.py
                                               DensityFB.z_actor_reward_rows under
                                               'latent_centered' (slow index tower, stored
                                               mu_hat) -- the reward its actor ascends.
Both use the same task-inferred z as every other cube probe (agents.relabel_cond_agents).

Stage `hist` (no rollouts): N dataset next-states, labelled by the TRUE relabel reward per
task (cube within 4 cm of the task goal; env/ogbench.py cube_relabel_reward, shift 1.0), the
reward view of each critic on every state; AUC of r_hat as a classifier of the true reward
and d' = (mean r_hat | r=1 - mean r_hat | r=0) / std(r_hat | r=0), per task and pooled.

Stage `roll` (re-roll with FULL observations): the separation probe's protocol (same banks,
same a+ / a- per state and model, same z, same rng keys, K = 10, H = 100), storing every
future observation (float16) plus the cube position from the simulator and the true hit.

Stage `score` (three numbers per (critic, seed, bank, state), gamma^t weights normalised
over the horizon):
  (i)   TRUE gap            = disc. mean true reward after a+  - after a-
  (ii)  FEATURE-LIMITED gap = the same futures scored with r_hat instead of the true reward
  (iii) CRITIC gap          = Q(s,a+) - Q(s,a-), the decision-by-phase probe's own readout
                              (gap_preservation_per_state.csv delta_hat), put on the scale
                              of (i): FB's Q = F^T z is an unnormalised discounted sum, so it
                              is divided by sum_{t<H} gamma^t; Density-FB's Q is an
                              expectation of r_z under a normalised measure, so it is left
                              as is (critic_gap_scale).
  Units: r_hat is in the critic's own reward units (FB's z is norm-sqrt(d) projected, Density-
  FB's r_z is divided by sqrt(d)), so (ii) and (iii) are converted to true-reward units by
  ONE least-squares slope per (critic, seed): the true 0/1 reward regressed on r_hat over the
  histogram stage's dataset states, all tasks pooled (calibration_slope). The intercept
  cancels in a gap; signs and correlations do not depend on it. The raw values are kept
  (feat_gap_raw, critic_gap_q = delta_hat, critic_gap_norm = delta_hat x critic_gap_scale).

    python scripts/probes/cube_bottleneck/reward_view.py --smoke
    python scripts/probes/cube_bottleneck/reward_view.py hist --seeds 0 1 2
    python scripts/probes/cube_bottleneck/reward_view.py roll --seeds 0 --banks carrying
    python scripts/probes/cube_bottleneck/reward_view.py score
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = str(Path(__file__).resolve().parents[3])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.probes.cube_bottleneck import separation_preservation as SP  # noqa: E402

ENV, EPOCH, ROOT, FIG = SP.ENV, SP.EPOCH, SP.ROOT, SP.FIG
METHODS, METHOD_LABEL, BANKS, BANK_LABEL, BANK_IDX = SP.METHODS, SP.METHOD_LABEL, SP.BANKS, SP.BANK_LABEL, SP.BANK_IDX
K_ROLL, HORIZON = SP.K_ROLL, SP.HORIZON
OUT = ROOT / ENV / "reward_view"
N_HIST = 20000
HIST_SEED = 20260920
MIN_POS = 200
SUCCESS_RADIUS = 0.04
CHUNK = 8192
STATE_KEYS = ["critic", "seed", "bank", "state"]
QUANT = ["true_gap", "feat_gap", "critic_gap"]
QUANT_LABEL = {"true_gap": "true", "feat_gap": "feature-limited", "critic_gap": "critic"}


# ── pure helpers (unit-tested) ──────────────────────────────────────────────────────────

def auc(scores, labels) -> float:
    """Area under the ROC curve of `scores` as a classifier of binary `labels` (Mann-Whitney
    with average ranks on ties). NaN when a class is empty."""
    s = np.asarray(scores, float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _u, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts + 1
    ranks = np.empty(len(s))
    ranks[order] = (first + (counts - 1) / 2.0)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def dprime(scores, labels) -> float:
    """(mean r_hat | r=1 - mean r_hat | r=0) / std(r_hat | r=0)."""
    s = np.asarray(scores, float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if y.sum() == 0 or (~y).sum() < 2:
        return float("nan")
    sd = float(s[~y].std(ddof=1))
    return float((s[y].mean() - s[~y].mean()) / sd) if sd > 0 else float("nan")


def separation_stats(scores, labels) -> Dict[str, float]:
    s = np.asarray(scores, float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    return dict(n1=int(y.sum()), n0=int((~y).sum()), auc=auc(s, y), dprime=dprime(s, y),
                mean1=float(s[y].mean()) if y.any() else float("nan"),
                mean0=float(s[~y].mean()) if (~y).any() else float("nan"),
                std0=float(s[~y].std(ddof=1)) if (~y).sum() > 1 else float("nan"),
                std1=float(s[y].std(ddof=1)) if y.sum() > 1 else float("nan"))


def standardise(scores, labels) -> np.ndarray:
    """(r_hat - mean0) / std0 against the reward-0 states: the scale on which seeds (whose
    r_hat units differ) can be pooled into one histogram. AUC and d' are unchanged by it."""
    s = np.asarray(scores, float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    m, sd = s[~y].mean(), s[~y].std(ddof=1)
    return (s - m) / (sd if sd > 0 else 1.0)


def discounted_mean(r: np.ndarray, gamma: float) -> float:
    """r [H, K] per-step reward of K futures -> sum_t w_t mean_k r[t, k], w_t = gamma^t
    normalised over the horizon (separation_preservation.discount_weights)."""
    r = np.asarray(r, float)
    w = SP.discount_weights(gamma, r.shape[0])
    return float((w[:, None] * r).sum(0).mean())


def horizon_sum(gamma: float, H: int) -> float:
    return float((gamma ** np.arange(H, dtype=np.float64)).sum())


def critic_gap_scale(critic: str, gamma: float, H: int) -> float:
    """Factor that puts Q(s,a+) - Q(s,a-) on the scale of a horizon-normalised discounted
    mean. FB: Q = F^T z regresses r + gamma Q, an unnormalised discounted sum, so divide by
    sum_{t<H} gamma^t (the truncation to H is stated in the note). Density-FB: Q =
    sum_j w_j r_z(u_j) with w a distribution (softmax over the pool), already a mean."""
    return 1.0 if critic.startswith("density_fb") else 1.0 / horizon_sum(gamma, H)


def calibration_slope(rhat, labels) -> float:
    """Least-squares slope of the true reward on the reward view, cov(r, r_hat) / var(r_hat):
    the factor that converts r_hat units into reward units (NaN when r_hat is constant)."""
    x = np.asarray(rhat, float).ravel(); y = np.asarray(labels, float).ravel()
    v = float(x.var())
    return float(((x - x.mean()) * (y - y.mean())).mean() / v) if v > 0 else float("nan")


def gap_row(true_r: np.ndarray, rhat: np.ndarray, q_gap: float, gamma: float, critic: str,
            slope: float = 1.0) -> Dict[str, float]:
    """true_r, rhat: [2, H, K] (index 0 = a+, 1 = a-). Returns (i) true gap, (ii) feature-
    limited gap and (iii) critic gap (both in reward units through `slope`), their raw
    versions and the per-action discounted means."""
    H = true_r.shape[1]
    tp, tm = discounted_mean(true_r[0], gamma), discounted_mean(true_r[1], gamma)
    fp, fm = discounted_mean(rhat[0], gamma), discounted_mean(rhat[1], gamma)
    cn = float(q_gap) * critic_gap_scale(critic, gamma, H)
    return dict(true_plus=tp, true_minus=tm, feat_plus=fp, feat_minus=fm, slope=float(slope),
                true_gap=tp - tm, feat_gap_raw=fp - fm, feat_gap=float(slope) * (fp - fm),
                critic_gap_q=float(q_gap), critic_gap_norm=cn, critic_gap=float(slope) * cn)


def _corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(Pearson, Spearman) of two vectors; NaN when either is constant or n < 3."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan")
    pr = float(np.corrcoef(a, b)[0, 1])
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    sp = float(np.corrcoef(ra, rb)[0, 1])
    return pr, sp


def phase_table(rows: pd.DataFrame, by_seed: bool = False) -> pd.DataFrame:
    """Per (critic, bank[, seed]): n, mean +- s.e. of the three gaps, the fraction of states
    where (ii) and (iii) have the right sign (> 0, a+ above a-), and the Pearson / Spearman
    correlation of (ii) with (iii) across states; also (ii)/(i) and (iii)/(i) as ratios of
    means."""
    keys = ["critic", "bank"] + (["seed"] if by_seed else [])
    out = []
    for k, g in rows.groupby(keys, sort=False):
        k = (k,) if not isinstance(k, tuple) else k
        row = dict(zip(keys, k))
        row["method"] = METHOD_LABEL.get(row["critic"], row["critic"])
        row["n"] = len(g)
        for q in QUANT:
            v = g[q].to_numpy(float)
            row[f"{q}_mean"] = float(v.mean())
            row[f"{q}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            row[f"{q}_median"] = float(np.median(v))
        row["feat_sign_ok"] = float((g["feat_gap"] > 0).mean())
        row["critic_sign_ok"] = float((g["critic_gap"] > 0).mean())
        row["feat_over_true"] = row["feat_gap_mean"] / row["true_gap_mean"] if row["true_gap_mean"] else float("nan")
        row["critic_over_true"] = row["critic_gap_mean"] / row["true_gap_mean"] if row["true_gap_mean"] else float("nan")
        row["corr_feat_critic"], row["spearman_feat_critic"] = _corr(g["feat_gap"], g["critic_gap"])
        row["corr_true_critic"], row["spearman_true_critic"] = _corr(g["true_gap"], g["critic_gap"])
        row["corr_true_feat"], row["spearman_true_feat"] = _corr(g["true_gap"], g["feat_gap"])
        out.append(row)
    t = pd.DataFrame(out)
    t["bank"] = pd.Categorical(t["bank"], BANKS, ordered=True)
    return t.sort_values(keys).reset_index(drop=True)


# ── reward views (the critics' own functions) ───────────────────────────────────────────

def reward_view(arm, z: np.ndarray, obs: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """r_hat(u) for every row of `obs` [N, obs_dim] under task index `z` [d].

    FB: `FB._B(params, u) @ z` -- B through the critic's choke point, dotted with the
    reward-inferred z exactly as `FB.q_fn` dots F with z.
    Density-FB: `DensityFB.z_actor_reward_rows(cond)` with cond["_density_zenc_u"] built the
    way `sample_cond` builds it (slow index tower `_zenc`, minus the stored mu_hat) -- the
    function the actor's readout `q_fn` calls, on rows u instead of the batch's candidates."""
    import jax
    import jax.numpy as jnp

    critic, net = arm.agent.critic, arm.agent.net
    z = jnp.asarray(z, jnp.float32)
    out = []
    if arm.critic.startswith("density_fb"):
        assert critic.z_actor_reward == "latent_centered", critic.z_actor_reward
        st = critic._zstats(net)
        zenc = critic._zenc(net)

        @jax.jit
        def f(u):
            u_sc = critic._b_query(net, u, critic._kernel_rows({"next_observations": u}, "next_observations", u))
            cond = {"z": z[None], "_density_zenc_u": critic._B(zenc, u_sc) - st["mu"][None]}
            return critic.z_actor_reward_rows(cond)[0]
    else:
        bp = net["backward"].params

        @jax.jit
        def f(u):
            return critic._B(bp, u) @ z
    for i in range(0, len(obs), chunk):
        out.append(np.asarray(f(jnp.asarray(np.asarray(obs[i:i + chunk], np.float32)))))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def load_seed(seed: int):
    """Arms, per-task conditioned agents (same z as every cube probe) and the task list."""
    from scripts.probes.cube_bottleneck import agents as A

    runs = {r["critic"]: r for r in A.discover_roots(ENV, METHODS, [seed], EPOCH)}
    assert set(runs) == set(METHODS), f"seed {seed}: found {sorted(runs)}"
    arms = {m: A.load_arm_eval(runs[m]["run_dir"], EPOCH) for m in METHODS}
    tasks = A.eval_tasks(arms["fb_tdj"])
    conds = {m: A.relabel_cond_agents(arms[m], tasks) for m in METHODS}
    return arms, conds, tasks


# ── stage hist ──────────────────────────────────────────────────────────────────────────

def draw_states(n: int, seed: int) -> Dict[str, np.ndarray]:
    """n dataset NEXT-states (observation + simulator qpos), fixed by seed."""
    from scripts.probes.cube_bottleneck import banks as BK

    raw = BK.load_raw(ENV)
    d = np.load(BK.dataset_path(ENV))
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(~raw["terminals"][:-1])
    sel = np.sort(rng.choice(ok, size=n, replace=False)) + 1
    return dict(row=sel.astype(np.int64), obs=raw["observations"][sel], qpos=np.asarray(d["qpos"][sel], np.float64))


def true_labels(env, qpos: np.ndarray, tasks: Sequence[int], threshold: float = SUCCESS_RADIUS) -> np.ndarray:
    """[T, N] 0/1: the relabel reward of env/ogbench.py (cube within `threshold` of the task
    goal, shift 1.0) -- the rule that labelled the rows z was inferred from."""
    from env.ogbench import cube_relabel_reward

    infos = env.unwrapped.task_infos
    lab = np.stack([cube_relabel_reward(qpos, infos[int(t) - 1]["goal_xyzs"], threshold) + 1.0 for t in tasks])
    return (lab > 0.5).astype(np.int8)


def stage_hist(seeds: Sequence[int], out_dir: Path, n: int = N_HIST, hist_seed: int = HIST_SEED, tag: str = "") -> None:
    states = draw_states(n, hist_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for seed in seeds:
        arms, conds, tasks = load_seed(seed)
        lab = true_labels(arms["fb_tdj"].env, states["qpos"], tasks)
        store = dict(row=states["row"], labels=lab, tasks=np.asarray(tasks, np.int64))
        for m in METHODS:
            rh = np.stack([reward_view(arms[m], np.asarray(conds[m][t].cond_eval["z"]), states["obs"]) for t in tasks])
            store[f"rhat_{m}"] = rh.astype(np.float32)
            store[f"z_{m}"] = np.stack([np.asarray(conds[m][t].cond_eval["z"]) for t in tasks]).astype(np.float32)
        np.savez_compressed(out_dir / f"hist_seed{seed}{tag}.npz", **store)
        print(f"[hist] seed {seed}: {n} states x {len(tasks)} tasks, positives per task "
              f"{lab.sum(1).tolist()} ({time.time() - t0:.0f}s)", flush=True)


def hist_tables(out_dir: Path, tag: Optional[str] = None, min_pos: int = MIN_POS):
    """Per (critic, seed, task | pooled) separation stats, the pooled-over-seeds stats on the
    standardised scores, and the standardised scores themselves for the figure."""
    fs = sorted(glob.glob(str(out_dir / "hist_seed*.npz")))
    fs = [f for f in fs if (("smoke" not in f) if tag is None else f.endswith(f"{tag}.npz"))]
    assert fs, f"no hist files under {out_dir}"
    rows, pooled_scores = [], {}
    for f in fs:
        d = np.load(f)
        seed = int(Path(f).stem.split("seed")[1].split("_")[0])
        lab, tasks = d["labels"], d["tasks"].tolist()
        for m in METHODS:
            rh = d[f"rhat_{m}"]
            keep = [i for i, t in enumerate(tasks) if lab[i].sum() >= min_pos]
            for i in keep:
                s = separation_stats(rh[i], lab[i])
                rows.append(dict(critic=m, method=METHOD_LABEL[m], seed=seed, task=int(tasks[i]), **s))
                pooled_scores.setdefault((m, int(tasks[i])), []).append((standardise(rh[i], lab[i]), lab[i]))
            # one pooled label: every (state, task) pair, "cube at this task's goal"
            sc = np.concatenate([standardise(rh[i], lab[i]) for i in keep]) if keep else np.zeros(0)
            lb = np.concatenate([lab[i] for i in keep]) if keep else np.zeros(0)
            raw = np.concatenate([rh[i] for i in keep]) if keep else np.zeros(0)
            s = separation_stats(raw, lb)
            rows.append(dict(critic=m, method=METHOD_LABEL[m], seed=seed, task=0, **s))
            pooled_scores.setdefault((m, 0), []).append((sc, lb))
    per_seed = pd.DataFrame(rows)
    pooled = []
    for (m, t), lst in sorted(pooled_scores.items()):
        sc = np.concatenate([x[0] for x in lst]); lb = np.concatenate([x[1] for x in lst])
        ps = per_seed[(per_seed["critic"] == m) & (per_seed["task"] == t)]
        pooled.append(dict(critic=m, method=METHOD_LABEL[m], task=t, n_seeds=len(lst),
                           auc_pooled=auc(sc, lb), dprime_pooled=dprime(sc, lb),
                           auc_mean=float(ps["auc"].mean()), auc_min=float(ps["auc"].min()), auc_max=float(ps["auc"].max()),
                           dprime_mean=float(ps["dprime"].mean()), n1=int(lb.sum()), n0=int(len(lb) - lb.sum())))
    return per_seed, pd.DataFrame(pooled), pooled_scores


def plot_hist(pooled: pd.DataFrame, scores: Dict, fig_dir: Path, name: str = "fig_reward_view_hist") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.probes.cube_bottleneck.failure_plots import METHOD_COLOR

    tasks = sorted({t for (_m, t) in scores})
    tasks = [t for t in tasks if t != 0] + [0]
    fig, axes = plt.subplots(len(METHODS), len(tasks), figsize=(1.75 * len(tasks) + 0.4, 1.6 * len(METHODS) + 0.5),
                             sharex="row", squeeze=False)
    for r, m in enumerate(METHODS):
        for c, t in enumerate(tasks):
            ax = axes[r, c]
            lst = scores.get((m, t))
            if not lst:
                ax.set_axis_off(); continue
            sc = np.concatenate([x[0] for x in lst]); lb = np.concatenate([x[1] for x in lst]).astype(bool)
            lo, hi = np.percentile(sc, 0.2), max(np.percentile(sc, 99.8), np.percentile(sc[lb], 99.5) if lb.any() else 0)
            bins = np.linspace(lo, hi, 50)
            ax.hist(sc[~lb], bins=bins, density=True, color="#c8c8c8", lw=0, label="true reward 0")
            ax.hist(sc[lb], bins=bins, density=True, histtype="stepfilled", facecolor=METHOD_COLOR[m], alpha=0.75,
                    edgecolor="#222222", lw=0.6, label="true reward 1 (cube at goal)")
            p = pooled[(pooled["critic"] == m) & (pooled["task"] == t)].iloc[0]
            ax.text(0.97, 0.93, f"AUC {p['auc_pooled']:.2f}\nd' {p['dprime_pooled']:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5)
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title("all tasks" if t == 0 else f"task {t}", fontsize=8)
            if c == 0:
                ax.set_ylabel(METHOD_LABEL[m], fontsize=8)
    fig.supxlabel("reward view $\\hat r$, in units of its spread over reward-0 states (per critic, seed and task)",
                  fontsize=7, y=0.02)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=2)
    fig.suptitle("Task reward as each critic reads it on dataset states", fontsize=8.5, y=1.04)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {fig_dir / name}.pdf/.png")


# ── compact paper versions (same data, read from the saved tables) ──────────────────────

def plot_hist_main(pooled: pd.DataFrame, scores: Dict, fig_dir: Path, name: str = "fig_reward_view_hist_main") -> None:
    """Only the "all tasks" histograms, FB above Density-FB, one shared legend at the top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.probes.cube_bottleneck.failure_plots import METHOD_COLOR

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(3.0, 2.6), squeeze=False, gridspec_kw=dict(hspace=0.35))
    for r, m in enumerate(METHODS):
        ax = axes[r, 0]
        lst = scores[(m, 0)]
        sc = np.concatenate([x[0] for x in lst]); lb = np.concatenate([x[1] for x in lst]).astype(bool)
        lo, hi = np.percentile(sc, 0.2), max(np.percentile(sc, 99.8), np.percentile(sc[lb], 99.5))
        bins = np.linspace(lo, hi, 50)
        ax.hist(sc[~lb], bins=bins, density=True, color="#c8c8c8", lw=0, label="true reward 0")
        ax.hist(sc[lb], bins=bins, density=True, histtype="stepfilled", facecolor=METHOD_COLOR[m], alpha=0.75,
                edgecolor="#222222", lw=0.6, label="true reward 1")
        p = pooled[(pooled["critic"] == m) & (pooled["task"] == 0)].iloc[0]
        ax.text(0.97, 0.92, f"{METHOD_LABEL[m]}\nAUC {p['auc_pooled']:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=METHOD_COLOR[m] if m != "fb_tdj" else "#444444")
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("reward as the critic reads it\n(units of its spread on reward-0 states)", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.subplots_adjust(top=0.9)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {fig_dir / name}.pdf/.png")


def plot_gaps_main(t: pd.DataFrame, fig_dir: Path, name: str = "fig_reward_view_gaps_main") -> None:
    """Only the bar row: per phase, true / feature-limited / critic gap, FB and Density-FB side
    by side on a shared y axis, legend at the top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scripts.probes.cube_bottleneck.failure_plots import METHOD_COLOR

    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.2, 2.3), sharey=True, squeeze=False, gridspec_kw=dict(wspace=0.12))
    x = np.arange(len(BANKS))
    w = 0.26
    fills = {"true_gap": dict(facecolor="#ffffff", hatch="////"), "feat_gap": dict(alpha=0.45), "critic_gap": dict(alpha=1.0)}
    ymax = float(np.nanmax(t[[f"{q}_mean" for q in QUANT]].to_numpy()) + np.nanmax(t[[f"{q}_se" for q in QUANT]].to_numpy()))
    ymin = float(np.nanmin(t[[f"{q}_mean" for q in QUANT]].to_numpy()))
    for j, m in enumerate(METHODS):
        ax, col = axes[0, j], METHOD_COLOR[m]
        b = t[t["critic"] == m].set_index("bank").reindex(BANKS)
        for qi, q in enumerate(QUANT):
            xo = x + (qi - 1) * w
            kw = dict(fills[q]); kw.setdefault("facecolor", col)
            ax.bar(xo, b[f"{q}_mean"], width=w * 0.9, edgecolor=col, linewidth=0.8, yerr=b[f"{q}_se"],
                   error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor="#333333"), **kw)
            for xi, v, e in zip(xo, b[f"{q}_mean"], b[f"{q}_se"].fillna(0.0)):
                ax.text(xi, max(v + e, 0) + 0.02 * ymax, f"{v:.2f}", ha="center", va="bottom", fontsize=5, color="#333333")
        ax.axhline(0, color="#666666", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([BANK_LABEL[bk] for bk in BANKS], fontsize=7)
        ax.text(0.03, 0.95, METHOD_LABEL[m], transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                color=col if m != "fb_tdj" else "#444444")
        ax.tick_params(axis="y", labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylim(min(0.0, ymin * 1.3), ymax * 1.22)
    axes[0, 0].set_ylabel("action gap ($a^+$ minus $a^-$),\ndiscounted mean reward", fontsize=7)
    fig.legend(handles=[Patch(facecolor="#ffffff", edgecolor="#333333", hatch="////", label="true reward"),
                        Patch(facecolor="#333333", alpha=0.45, label="critic's reward view on the same futures"),
                        Patch(facecolor="#333333", label="critic's own gap $Q(s,a^+)-Q(s,a^-)$")],
               frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0, handlelength=1.4)
    fig.subplots_adjust(top=0.86)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {fig_dir / name}.pdf/.png")


def stage_hist_aggregate(out_dir: Path, fig_dir: Path, tag: Optional[str] = None, min_pos: int = MIN_POS) -> None:
    per_seed, pooled, scores = hist_tables(out_dir, tag, min_pos=min_pos)
    fig_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(fig_dir / "reward_view_hist_per_seed.csv", index=False)
    pooled.to_csv(fig_dir / "reward_view_hist_pooled.csv", index=False)
    print("=== reward view on dataset states: separation of true-reward-1 from reward-0 states (task 0 = pooled) ===")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot_hist(pooled, scores, fig_dir)
    plot_hist_main(pooled, scores, fig_dir)


# ── stage roll (full observations) ──────────────────────────────────────────────────────

def roll_futures_full(cond_agent, envs, rec, acts, K: int, H: int, rng_key, goal_xyz, cidx: int):
    """separation_preservation.roll_futures, keeping the FULL observation at every step
    (float16), the cube position from the simulator joint and the true hit
    1[|cube - goal| <= 4 cm] (failure_figs.truth_actions' rule). Returns obs [n, H, K, D],
    cube [n, H, K, 3], hit [n, H, K]."""
    from impls.utils.evaluation import supply_rng
    from scripts.probes.cube_bottleneck.failure_figs import restore_full

    actor_fn = supply_rng(cond_agent.sample_actions, rng=rng_key)
    n_acts, n = len(acts), len(acts) * K
    D = envs[0].observation_space.shape[0]
    obs = np.zeros((n, D), np.float32)
    alive = np.ones(n, bool)
    full = np.zeros((H, n, D), np.float16)
    cube = np.zeros((H, n, 3), np.float32)
    hit = np.zeros((H, n), np.int8)
    for i in range(n_acts):
        for k in range(K):
            j = i * K + k
            restore_full(envs[j], rec)
            o, _r, term, trunc, _info = envs[j].step(np.asarray(acts[i], np.float32))
            obs[j] = np.asarray(o, np.float32)
            if term or trunc:
                alive[j] = False

    def _record(t):
        full[t] = obs
        for j in range(n):
            pos = np.asarray(envs[j].unwrapped._data.joint(f"object_joint_{cidx}").qpos[:3])
            cube[t, j] = pos
            hit[t, j] = int(np.linalg.norm(pos - goal_xyz) <= SUCCESS_RADIUS)
    _record(0)
    for t in range(1, H):
        a = np.clip(np.asarray(actor_fn(observations=obs, goals=None, temperature=0.0)), -1, 1)
        for j in range(n):
            if alive[j]:
                o, _r, term, trunc, _info = envs[j].step(a[j].astype(np.float32))
                obs[j] = np.asarray(o, np.float32)
                if term or trunc:
                    alive[j] = False
        _record(t)
    r = lambda x: x.reshape((H, n_acts, K) + x.shape[2:]).swapaxes(0, 1)
    return r(full), r(cube), r(hit)


def stage_roll(seed: int, banks: Sequence[str], out_dir: Path, per_state_csv: Path,
               max_states: int = 0, K: int = K_ROLL, H: int = HORIZON, tag: str = "") -> List[Path]:
    """One file per (seed, bank): futures_full_seed{seed}_{bank}{tag}.npz + .csv."""
    import jax
    from scripts.probes.cube_bottleneck import failure_figs as FF
    from scripts.probes.cube_bottleneck import restore as R

    arms, conds, _tasks = load_seed(seed)
    gamma = {m: float(arms[m].config.get("discount", 0.99)) for m in METHODS}
    ps = pd.read_csv(per_state_csv)
    ps = ps[ps["seed"] == seed].set_index(["critic", "bank", "state"])
    bk = np.load(ROOT / ENV / "banks" / "banks.npz")
    envs = FF.make_env_pool(ENV, 2 * K)
    R.ensure_cube_env(envs[0])
    adim = int(envs[0].action_space.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    files, t0 = [], time.time()
    for bank in banks:
        store, meta = {}, []
        n = len(bk[f"{bank}_qpos"]) if not max_states else min(max_states, len(bk[f"{bank}_qpos"]))
        for i in range(n):
            rec = dict(idx=i, task=int(bk[f"{bank}_task"][i]), goal=bk[f"{bank}_goal"][i],
                       qpos=bk[f"{bank}_qpos"][i], qvel=bk[f"{bank}_qvel"][i],
                       grip=float(bk[f"{bank}_grip"][i]), cube=int(bk[f"{bank}_cube"][i]))
            tid, c = rec["task"], rec["cube"]
            goal_xyz = np.asarray(rec["goal"])[c]
            ob = FF.restore_full(envs[0], rec)
            key = jax.random.PRNGKey(4242 + i * 7 + seed)
            a_fb = np.asarray(conds["fb_tdj"][tid].sample_actions(ob, goals=None, seed=key))
            a_d = np.asarray(conds["density_fb_z"][tid].sample_actions(ob, goals=None, seed=key))
            rng = np.random.default_rng([FF.BANK_SEED, seed, i, BANK_IDX[bank]])
            cands = FF.candidate_actions(a_fb.astype(np.float32), a_d.astype(np.float32), adim, rng)
            for m in METHODS:
                row = ps.loc[(m, bank, i)]
                if int(row["tie"]):
                    continue
                ip, im = FF.CANDIDATE_NAMES.index(row["best"]), FF.CANDIDATE_NAMES.index(row["worst"])
                acts = cands[[ip, im]]
                full, cube, hit = roll_futures_full(conds[m][tid], envs, rec, acts, K, H,
                                                    jax.random.PRNGKey(777 + i * 13 + seed), goal_xyz, c)
                p = f"{bank}_{i}_{m}"
                store[f"{p}_obs"], store[f"{p}_cube"], store[f"{p}_hit"] = full, cube, hit
                store[f"{p}_acts"] = acts.astype(np.float32)
                meta.append(dict(critic=m, seed=seed, bank=bank, state=i, task=tid, cube=c, best=row["best"],
                                 worst=row["worst"], gamma=gamma[m], K=K, H=H,
                                 delta_mc=float(row["delta_mc"]), delta_hat=float(row["delta_hat"]),
                                 goal_x=goal_xyz[0], goal_y=goal_xyz[1], goal_z=goal_xyz[2]))
            if (i + 1) % 8 == 0 or i + 1 == n:
                print(f"  [rv seed{seed}] {bank} {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
        f = out_dir / f"futures_full_seed{seed}_{bank}{tag}.npz"
        np.savez_compressed(f, **store)
        pd.DataFrame(meta).to_csv(f.with_suffix(".csv"), index=False)
        files.append(f)
    print(f"[rv] seed {seed}: {len(files)} bank files in {time.time() - t0:.0f}s", flush=True)
    return files


# ── stage score ─────────────────────────────────────────────────────────────────────────

def score_seed(seed: int, out_dir: Path, tag: Optional[str], arms=None, conds=None) -> pd.DataFrame:
    """The three numbers per (critic, bank, state) of one seed, from the stored futures."""
    fs = sorted(glob.glob(str(out_dir / f"futures_full_seed{seed}_*.csv")))
    fs = [f for f in fs if (("smoke" not in f) if tag is None else f.endswith(f"{tag}.csv"))]
    if not fs:
        return pd.DataFrame()
    if arms is None:
        arms, conds, _t = load_seed(seed)
    hist_f = out_dir / f"hist_seed{seed}{tag or ''}.npz"
    slopes = {m: 1.0 for m in METHODS}
    if hist_f.exists():
        hd = np.load(hist_f)
        slopes = {m: calibration_slope(hd[f"rhat_{m}"], np.broadcast_to(hd["labels"], hd[f"rhat_{m}"].shape)) for m in METHODS}
    else:
        print(f"[rv] WARNING no {hist_f}: gaps left in r_hat units (slope 1)", flush=True)
    print(f"[rv] seed {seed} calibration slopes (reward per r_hat unit): {slopes}", flush=True)
    rows = []
    for f in fs:
        meta = pd.read_csv(f)
        d = np.load(f[:-4] + ".npz")
        sep_f = ROOT / ENV / "separation" / f"futures_seed{seed}_{meta['bank'].iloc[0]}.npz"
        sep = np.load(sep_f) if sep_f.exists() and tag is None else None
        for r in meta.itertuples():
            p = f"{r.bank}_{r.state}_{r.critic}"
            obs, hit = d[f"{p}_obs"], d[f"{p}_hit"].astype(float)              # [2,H,K,D], [2,H,K]
            z = np.asarray(conds[r.critic][int(r.task)].cond_eval["z"])
            rh = reward_view(arms[r.critic], z, obs.reshape(-1, obs.shape[-1])).reshape(obs.shape[:3])
            g = gap_row(hit, rh, float(r.delta_hat), float(r.gamma), r.critic, slope=slopes[r.critic])
            # consistency with the earlier probes: the true gap x sum gamma^t is delta_mc of
            # the decision figure; the cube path from obs matches the separation futures
            g["delta_mc_reroll"] = g["true_gap"] * horizon_sum(float(r.gamma), int(r.H))
            g["delta_mc_probe"] = float(r.delta_mc)
            if sep is not None and f"{p}_mc" in sep:
                g["cube_path_maxdiff"] = float(np.abs(SP.cube_xyz_from_obs(obs.astype(np.float32)) - sep[f"{p}_mc"]).max())
            rows.append(dict(critic=r.critic, method=METHOD_LABEL[r.critic], seed=seed, bank=r.bank, state=int(r.state),
                             task=int(r.task), best=r.best, worst=r.worst, gamma=float(r.gamma), K=int(r.K), H=int(r.H),
                             rhat_std=float(rh.std()), **g))
    return pd.DataFrame(rows)


def plot_gaps(t: pd.DataFrame, per: pd.DataFrame, fig_dir: Path, name: str = "fig_reward_view_gaps") -> None:
    """Top: per phase, three bars (true / feature-limited / critic) per method. Bottom: (ii)
    vs (iii) per state, per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scripts.probes.cube_bottleneck.failure_plots import METHOD_COLOR

    fig, axes = plt.subplots(2, len(METHODS), figsize=(3.3 * len(METHODS), 5.0), squeeze=False,
                             gridspec_kw=dict(hspace=0.45, wspace=0.3))
    x = np.arange(len(BANKS))
    w = 0.26
    fills = {"true_gap": dict(facecolor="#ffffff", hatch="////"), "feat_gap": dict(alpha=0.45), "critic_gap": dict(alpha=1.0)}
    ymax = float(np.nanmax(t[[f"{q}_mean" for q in QUANT]].to_numpy()) + np.nanmax(t[[f"{q}_se" for q in QUANT]].to_numpy()))
    for j, m in enumerate(METHODS):
        ax, col = axes[0, j], METHOD_COLOR[m]
        b = t[t["critic"] == m].set_index("bank").reindex(BANKS)
        for qi, q in enumerate(QUANT):
            xo = x + (qi - 1) * w
            kw = dict(fills[q]); kw.setdefault("facecolor", col)
            ax.bar(xo, b[f"{q}_mean"], width=w * 0.9, edgecolor=col, linewidth=0.8, yerr=b[f"{q}_se"],
                   error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#333333"), **kw)
            for xi, v, e in zip(xo, b[f"{q}_mean"], b[f"{q}_se"].fillna(0.0)):
                ax.text(xi, max(v + e, 0) + 0.02 * ymax, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, color="#333333")
        ax.axhline(0, color="#666666", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([BANK_LABEL[bk] for bk in BANKS])
        ax.set_title(METHOD_LABEL[m], fontsize=8.5)
        ymin = float(np.nanmin(t[[f"{q}_mean" for q in QUANT]].to_numpy()))
        ax.set_ylim(min(0.0, ymin * 1.3), ymax * 1.25)
        if j == 0:
            ax.set_ylabel("action gap, discounted mean\n($a^+$ minus $a^-$)")
            fig.legend(handles=[Patch(facecolor="#ffffff", edgecolor="#333333", hatch="////", label="true reward"),
                                Patch(facecolor="#333333", alpha=0.45, label="critic's reward view on the same futures"),
                                Patch(facecolor="#333333", label="critic $Q(s,a^+)-Q(s,a^-)$")],
                       frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
        ax2 = axes[1, j]
        markers = {"approach": "o", "inhand": "s", "carrying": "^"}
        for bk in BANKS:
            g = per[(per["critic"] == m) & (per["bank"] == bk)]
            ax2.scatter(g["feat_gap"], g["critic_gap"], s=9, marker=markers[bk], color=col, alpha=0.55, lw=0,
                        label=BANK_LABEL[bk])
        lim = float(np.nanmax(np.abs(per[per["critic"] == m][["feat_gap", "critic_gap"]].to_numpy()))) * 1.05 + 1e-9
        ax2.plot([-lim, lim], [-lim, lim], color="#666666", lw=0.6, ls="--")
        ax2.axhline(0, color="#bbbbbb", lw=0.5); ax2.axvline(0, color="#bbbbbb", lw=0.5)
        ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
        pr, sp_ = _corr(per[per["critic"] == m]["feat_gap"], per[per["critic"] == m]["critic_gap"])
        ax2.text(0.03, 0.95, f"r = {pr:.2f}\n$\\rho$ = {sp_:.2f}", transform=ax2.transAxes, ha="left", va="top", fontsize=7)
        ax2.set_xlabel("feature-limited gap (ii)")
        if j == 0:
            ax2.set_ylabel("critic gap (iii)")
        ax2.legend(frameon=False, fontsize=6, loc="lower right", markerscale=1.3)
    fig.suptitle("Is the critic's action gap the gap of its own reward view?", fontsize=8.5, y=0.995)
    fig.subplots_adjust(top=0.86)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {fig_dir / name}.pdf/.png")


def stage_score(seeds: Sequence[int], out_dir: Path, fig_dir: Path, tag: Optional[str] = None) -> None:
    per = pd.concat([score_seed(s, out_dir, tag) for s in seeds], ignore_index=True)
    assert len(per), "no scored states"
    fig_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(fig_dir / "reward_view_gaps_per_state.csv", index=False)
    stage_tables(fig_dir, per)


def stage_tables(fig_dir: Path, per: Optional[pd.DataFrame] = None) -> None:
    """Tables and the figure from the per-state csv (no models needed). Also the subset of
    states whose RE-ROLLED true gap is positive (a+ still beats a- under this draw of the
    policy's noise): `_consistent` tables."""
    if per is None:
        per = pd.read_csv(fig_dir / "reward_view_gaps_per_state.csv")
    t, ts = phase_table(per), phase_table(per, by_seed=True)
    cons = per[per["true_gap"] > 0]
    phase_table(cons).to_csv(fig_dir / "reward_view_gaps_by_phase_consistent.csv", index=False)
    phase_table(cons, by_seed=True).to_csv(fig_dir / "reward_view_gaps_by_phase_consistent_per_seed.csv", index=False)
    t.to_csv(fig_dir / "reward_view_gaps_by_phase.csv", index=False)
    ts.to_csv(fig_dir / "reward_view_gaps_by_phase_per_seed.csv", index=False)
    cols = ["method", "bank", "n", "true_gap_mean", "true_gap_se", "feat_gap_mean", "feat_gap_se", "critic_gap_mean",
            "critic_gap_se", "feat_over_true", "critic_over_true", "feat_sign_ok", "critic_sign_ok",
            "corr_feat_critic", "spearman_feat_critic", "corr_true_critic"]
    print("=== three-number test, pooled over seeds (gamma^t weights normalised over H) ===")
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=== per seed ===")
    print(ts[["seed"] + cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"=== states with re-rolled true gap > 0 only ({len(cons)} of {len(per)}) ===")
    print(phase_table(cons)[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    chk = dict(delta_mc_abs_diff_max=float((per["delta_mc_reroll"] - per["delta_mc_probe"]).abs().max()),
               delta_mc_abs_diff_mean=float((per["delta_mc_reroll"] - per["delta_mc_probe"]).abs().mean()),
               n_states=int(len(per)))
    if "cube_path_maxdiff" in per:
        chk["cube_path_maxdiff_max"] = float(per["cube_path_maxdiff"].max())
    (fig_dir / "reward_view_gaps_checks.json").write_text(json.dumps(chk, indent=2))
    print(f"=== re-roll consistency: {chk}")
    plot_gaps(t, per, fig_dir)
    plot_gaps_main(t, fig_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["hist", "hist_aggregate", "roll", "score", "tables"], default="score")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--banks", nargs="+", default=BANKS)
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--n", type=int, default=N_HIST)
    ap.add_argument("--min-pos", type=int, default=MIN_POS)
    ap.add_argument("--K", type=int, default=K_ROLL)
    ap.add_argument("--H", type=int, default=HORIZON)
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-state-csv", default=str(FIG / "gap_preservation_per_state.csv"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--fig", default=str(FIG))
    args = ap.parse_args()
    out, fig = Path(args.out), Path(args.fig)
    if args.smoke:
        stage_hist([args.seeds[0]], out, n=2000, tag="_smoke")
        stage_hist_aggregate(out, fig / "smoke_rv", tag="_smoke", min_pos=5)
        stage_roll(args.seeds[0], ["carrying"], out, Path(args.per_state_csv), max_states=3, K=2, H=20, tag="_smoke")
        stage_score([args.seeds[0]], out, fig / "smoke_rv", tag="_smoke")
        return 0
    if args.stage == "hist":
        stage_hist(args.seeds, out, n=args.n, tag=args.tag)
        stage_hist_aggregate(out, fig, tag=args.tag or None, min_pos=args.min_pos)
    elif args.stage == "hist_aggregate":
        stage_hist_aggregate(out, fig, tag=args.tag or None, min_pos=args.min_pos)
    elif args.stage == "tables":
        stage_tables(fig)
    elif args.stage == "roll":
        for s in args.seeds:
            stage_roll(s, args.banks, out, Path(args.per_state_csv), max_states=args.max_states,
                       K=args.K, H=args.H, tag=args.tag)
    else:
        stage_score(args.seeds, out, fig, tag=args.tag or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
