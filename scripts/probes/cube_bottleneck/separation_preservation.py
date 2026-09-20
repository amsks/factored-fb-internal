"""Separation preservation: does the critic tell apart the cube FUTURES caused by the best
and the worst action? (cube-single, FB vs Density-FB; docs/notes/2026-09-19-separation-
preservation.md)

At every restored state s of the failure-figure banks, with the same a+ / a- (true best /
worst candidate by Monte-Carlo return) the decision-by-phase figure used
(gap_preservation_per_state.csv), and the same reward-inferred task vector z:

  M_MC(s, a, .)   restore s, force a once, continue with the model's OWN frozen policy for
                  H = 100 steps, K = 10 rollouts; cube position (obs[19:22], metres) at every
                  step, weighted gamma^t (normalised) -> the discounted empirical future.
  M_hat(s, a, .)  the critic's distribution over ONE fixed pool of N = 4096 dataset
                  next-states (same pool for both methods, both actions, every state).
                  Density-FB: softmax_j F(s,a,z).B(x_j)/tau averaged over heads (the
                  distribution its actor readout q_fn ascends; failure_figs.predict_density).
                  FB: min-over-heads F(s,a,z).B(x_j) (measure_probe.fb_readout), clipped at
                  0 and normalised over the pool; the clipping rate is recorded.
  cells           cube positions binned on a 3-D grid of CELL m cubes over the workspace
                  (main 5 cm; 3 cm and 8 cm for sensitivity); TV between cell histograms.
  D_true(s) = TV(M_MC(s,a+), M_MC(s,a-))      D_pred(s) = TV(M_hat(s,a+), M_hat(s,a-))
  R(s) = D_pred / D_true                       states with D_true < 0.05 are excluded
  fidelity(s, a) = TV(M_hat(s,a), M_MC(s,a))   critic vs reality, sanity number
  Align(s) = cos(d_hat, d_true)                 d_hat = M_hat(a+) - M_hat(a-) over cells,
                                               d_true likewise; also on the top-20 |d_true|
                                               cells; permutation null = shuffled pairing

Stages: `roll` (per seed: rollouts + predictions, cached as npz), `aggregate` (TVs at the
three grids, per-phase tables, the figure). `--smoke` = 3 carrying states x 2 actions x 2
rollouts, H = 20, then aggregate that.

    python scripts/probes/cube_bottleneck/separation_preservation.py --smoke
    python scripts/probes/cube_bottleneck/separation_preservation.py roll --seeds 0
    python scripts/probes/cube_bottleneck/separation_preservation.py aggregate
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

ENV = "cube-single-play-v0"
EPOCH = 1000000
ROOT = Path("analysis/failure_figs")
FIG = Path("PAPER/ICLR/figures/gap_preservation")
METHODS = ["fb_tdj", "density_fb_z"]
METHOD_LABEL = {"fb_tdj": "FB", "density_fb_z": "Density-FB"}
BANKS = ["approach", "inhand", "carrying"]
BANK_LABEL = {"approach": "approach", "inhand": "in hand", "carrying": "carrying"}
BANK_IDX = {"approach": 0, "inhand": 1, "carrying": 2}
POOL_N = 4096
POOL_SEED = 20260919
K_ROLL = 10
HORIZON = 100
CELLS = (0.03, 0.05, 0.08)
CELL_MAIN = 0.05
D_TRUE_MIN = 0.05
XYZ_SCALER = 10.0
XYZ_CENTER = np.array([0.425, 0.0, 0.0])
# workspace: object sampling bounds [0.30,0.55]x[-0.30,0.30] padded; z from below the table
# top (0.02) to a comfortable carry height
BOUNDS = ((0.25, 0.60), (-0.35, 0.35), (-0.05, 0.40))


# ── pure helpers (unit-tested) ──────────────────────────────────────────────────────────

def cube_xyz_from_obs(obs: np.ndarray) -> np.ndarray:
    """Absolute cube position in metres from the state observation (block 0: obs[19:22])."""
    return np.asarray(obs)[..., 19:22] / XYZ_SCALER + XYZ_CENTER


def grid_edges(cell: float, bounds=BOUNDS) -> List[np.ndarray]:
    return [np.arange(lo, hi + cell - 1e-9, cell) for lo, hi in bounds]


def hist3d(pts: np.ndarray, w: Optional[np.ndarray], cell: float, bounds=BOUNDS) -> np.ndarray:
    """Normalised histogram of 3-D points on `cell`-sized cubes over `bounds`; points outside
    are clipped to the border cell. Returns the flattened cell vector (sums to 1)."""
    edges = grid_edges(cell, bounds)
    p = np.asarray(pts, float).reshape(-1, 3).copy()
    for d in range(3):
        p[:, d] = np.clip(p[:, d], edges[d][0], edges[d][-1] - 1e-9)
    h, _ = np.histogramdd(p, bins=edges, weights=None if w is None else np.asarray(w, float).ravel())
    s = h.sum()
    return (h / s if s > 0 else np.full_like(h, 1.0 / h.size)).ravel()


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def discount_weights(gamma: float, horizon: int) -> np.ndarray:
    w = gamma ** np.arange(horizon, dtype=np.float64)
    return w / w.sum()


TOP_CELLS = 20


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """<a, b> / (||a|| ||b||); NaN when either norm is 0."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def alignment(d_hat: np.ndarray, d_true: np.ndarray, top: int = TOP_CELLS) -> Dict[str, float]:
    """Cosine between the critic's and the true difference-of-futures vectors over cells, and
    the same restricted to the `top` cells of largest |d_true|."""
    idx = np.argsort(-np.abs(d_true), kind="stable")[:top]
    return dict(align=cosine(d_hat, d_true), align_top=cosine(d_hat[idx], d_true[idx]))


def separation_row(mc: np.ndarray, pred: np.ndarray, pool_xyz: np.ndarray, gamma: float,
                   cell: float, return_vectors: bool = False):
    """mc: [2, H, K, 3] cube futures after a+ (0) and a- (1); pred: [2, N] critic weights over
    the pool; pool_xyz: [N, 3]. Returns D_true, D_pred, R, the two fidelities and the cosine
    alignment of d_hat = M_hat(a+) - M_hat(a-) with d_true = M_MC(a+) - M_MC(a-) (full and
    top-|d_true| cells); with `return_vectors` also (d_hat, d_true)."""
    H, K = mc.shape[1], mc.shape[2]
    w = np.broadcast_to(discount_weights(gamma, H)[:, None] / K, (H, K))
    h_mc = [hist3d(mc[i], w, cell) for i in range(2)]
    h_pr = [hist3d(pool_xyz, pred[i], cell) for i in range(2)]
    d_true, d_pred = tv(h_mc[0], h_mc[1]), tv(h_pr[0], h_pr[1])
    dh, dt = h_pr[0] - h_pr[1], h_mc[0] - h_mc[1]
    out = dict(d_true=d_true, d_pred=d_pred,
               R=d_pred / d_true if d_true > 0 else float("nan"),
               fidelity_plus=tv(h_pr[0], h_mc[0]), fidelity_minus=tv(h_pr[1], h_mc[1]),
               **alignment(dh, dt))
    return (out, dh, dt) if return_vectors else out


def align_table(rows: pd.DataFrame, by_seed: bool = False) -> pd.DataFrame:
    """Per (critic, bank[, seed]): alignment mean +- s.e., median, fraction > 0, n (states with
    a defined cosine), n_undefined (a zero difference vector on either side); same for the
    top-cell version."""
    keys = ["critic", "bank"] + (["seed"] if by_seed else [])
    out = []
    for k, g in rows.groupby(keys, sort=False):
        k = (k,) if not isinstance(k, tuple) else k
        row = dict(zip(keys, k))
        row["method"] = METHOD_LABEL.get(row["critic"], row["critic"])
        for col in ("align", "align_top"):
            v = g[col].to_numpy(float); v = v[np.isfinite(v)]
            row[f"{col}_n"] = len(v)
            row[f"{col}_undefined"] = int(len(g) - len(v))
            row[f"{col}_mean"] = float(v.mean()) if len(v) else float("nan")
            row[f"{col}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            row[f"{col}_median"] = float(np.median(v)) if len(v) else float("nan")
            row[f"{col}_pos"] = float((v > 0).mean()) if len(v) else float("nan")
        out.append(row)
    t = pd.DataFrame(out)
    t["bank"] = pd.Categorical(t["bank"], BANKS, ordered=True)
    return t.sort_values(keys).reset_index(drop=True)


def permutation_null(vectors: Dict[Tuple, Tuple[np.ndarray, np.ndarray]], n_perm: int = 200,
                     seed: int = 0) -> pd.DataFrame:
    """Null for the mean alignment: within each (critic, bank), shuffle which state's d_true
    is paired with which state's d_hat (pooled over seeds), recompute the mean cosine, repeat.
    `vectors`: {(critic, seed, bank, state): (d_hat, d_true)}. Returns per (critic, bank) the
    null mean and sd of the mean alignment (full and top cells)."""
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray]]] = {}
    for (c, _s, b, _i), v in vectors.items():
        groups.setdefault((c, b), []).append(v)
    out = []
    for (c, b), lst in groups.items():
        dh = [v[0] for v in lst]; dt = [v[1] for v in lst]
        means, means_top = [], []
        for _ in range(n_perm):
            perm = rng.permutation(len(lst))
            a = [alignment(dh[i], dt[perm[i]]) for i in range(len(lst))]
            m1 = np.array([x["align"] for x in a]); m2 = np.array([x["align_top"] for x in a])
            means.append(np.nanmean(m1)); means_top.append(np.nanmean(m2))
        out.append(dict(critic=c, bank=b, method=METHOD_LABEL.get(c, c), n_states=len(lst), n_perm=n_perm,
                        null_mean=float(np.mean(means)), null_sd=float(np.std(means, ddof=1)),
                        null_top_mean=float(np.mean(means_top)), null_top_sd=float(np.std(means_top, ddof=1))))
    t = pd.DataFrame(out)
    t["bank"] = pd.Categorical(t["bank"], BANKS, ordered=True)
    return t.sort_values(["critic", "bank"]).reset_index(drop=True)


def phase_table(rows: pd.DataFrame, d_true_min: float = D_TRUE_MIN, by_seed: bool = False) -> pd.DataFrame:
    """Per (critic, bank[, seed]): n kept, n excluded (D_true < d_true_min), mean/median R with
    the s.e. of the mean, mean D_true / D_pred / fidelity."""
    keys = ["critic", "bank"] + (["seed"] if by_seed else [])
    out = []
    for k, g in rows.groupby(keys, sort=False):
        k = (k,) if not isinstance(k, tuple) else k
        keep = g[g["d_true"] >= d_true_min]
        r = keep["R"].to_numpy(float)
        row = dict(zip(keys, k))
        row.update(method=METHOD_LABEL.get(row["critic"], row["critic"]), n=len(keep),
                   n_excluded=int(len(g) - len(keep)),
                   R_mean=float(r.mean()) if len(r) else float("nan"),
                   R_se=float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan"),
                   R_median=float(np.median(r)) if len(r) else float("nan"),
                   d_true_mean=float(keep["d_true"].mean()) if len(keep) else float("nan"),
                   d_pred_mean=float(keep["d_pred"].mean()) if len(keep) else float("nan"),
                   fidelity_mean=float(keep[["fidelity_plus", "fidelity_minus"]].to_numpy().mean()) if len(keep) else float("nan"))
        out.append(row)
    t = pd.DataFrame(out)
    t["bank"] = pd.Categorical(t["bank"], BANKS, ordered=True)
    return t.sort_values(keys).reset_index(drop=True)


# ── stage roll ──────────────────────────────────────────────────────────────────────────

def build_pool(out_dir: Path, n: int = POOL_N, seed: int = POOL_SEED) -> np.ndarray:
    """N dataset NEXT-states, fixed by seed; one file, shared by every seed and method."""
    from scripts.probes.cube_bottleneck import banks as BK

    f = out_dir / f"pool{n}.npz"
    if f.exists():
        return np.load(f)["obs"]
    raw = BK.load_raw(ENV)
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(~raw["terminals"][:-1])
    sel = np.sort(rng.choice(ok, size=n, replace=False)) + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(f, row=sel.astype(np.int64), obs=raw["observations"][sel])
    return raw["observations"][sel]


def roll_futures(cond_agent, envs, rec, acts, K: int, H: int, rng_key) -> np.ndarray:
    """Each of the len(acts) actions once, then K own-policy continuations for H steps in
    lockstep. Returns [n_acts, H, K, 3] cube positions (metres) from the observation."""
    from impls.utils.evaluation import supply_rng
    from scripts.probes.cube_bottleneck.failure_figs import restore_full

    actor_fn = supply_rng(cond_agent.sample_actions, rng=rng_key)
    n_acts, n = len(acts), len(acts) * K
    obs = np.zeros((n, envs[0].observation_space.shape[0]), np.float32)
    alive = np.ones(n, bool)
    xyz = np.zeros((H, n, 3), np.float32)
    for i in range(n_acts):
        for k in range(K):
            j = i * K + k
            restore_full(envs[j], rec)
            o, _r, term, trunc, _info = envs[j].step(np.asarray(acts[i], np.float32))
            obs[j] = np.asarray(o, np.float32)
            if term or trunc:
                alive[j] = False
    xyz[0] = cube_xyz_from_obs(obs)
    for t in range(1, H):
        a = np.clip(np.asarray(actor_fn(observations=obs, goals=None, temperature=0.0)), -1, 1)
        for j in range(n):
            if alive[j]:
                o, _r, term, trunc, _info = envs[j].step(a[j].astype(np.float32))
                obs[j] = np.asarray(o, np.float32)
                if term or trunc:
                    alive[j] = False
        xyz[t] = cube_xyz_from_obs(obs)
    return xyz.reshape(H, n_acts, K, 3).transpose(1, 0, 2, 3)


def stage_roll(seed: int, banks: Sequence[str], out_dir: Path, per_state_csv: Path,
               max_states: int = 0, K: int = K_ROLL, H: int = HORIZON, tag: str = "") -> Path:
    import jax
    from scripts.probes.cube_bottleneck import agents as A
    from scripts.probes.cube_bottleneck import failure_figs as FF
    from scripts.probes.cube_bottleneck import restore as R
    from scripts.probes.cube_bottleneck.measure_probe import fb_readout

    runs = {r["critic"]: r for r in A.discover_roots(ENV, METHODS, [seed], EPOCH)}
    assert set(runs) == set(METHODS), f"seed {seed}: found {sorted(runs)}"
    arms = {m: A.load_arm_eval(runs[m]["run_dir"], EPOCH) for m in METHODS}
    tasks = A.eval_tasks(arms["fb_tdj"])
    conds = {m: A.relabel_cond_agents(arms[m], tasks) for m in METHODS}
    gamma = {m: float(arms[m].config.get("discount", 0.99)) for m in METHODS}
    pool_obs = build_pool(out_dir)
    ps = pd.read_csv(per_state_csv)
    ps = ps[ps["seed"] == seed].set_index(["critic", "bank", "state"])
    bk = np.load(ROOT / ENV / "banks" / "banks.npz")
    envs = FF.make_env_pool(ENV, 2 * K)
    R.ensure_cube_env(envs[0])
    adim = int(envs[0].action_space.shape[0])
    store, meta = {}, []
    t0 = time.time()
    for bank in banks:
        n = len(bk[f"{bank}_qpos"]) if not max_states else min(max_states, len(bk[f"{bank}_qpos"]))
        for i in range(n):
            rec = dict(idx=i, task=int(bk[f"{bank}_task"][i]), goal=bk[f"{bank}_goal"][i],
                       qpos=bk[f"{bank}_qpos"][i], qvel=bk[f"{bank}_qvel"][i],
                       grip=float(bk[f"{bank}_grip"][i]), cube=int(bk[f"{bank}_cube"][i]))
            tid = rec["task"]
            # the SAME candidate set as failure_figs.stage_actionprobe (same rng keys)
            ob = FF.restore_full(envs[0], rec)
            key = jax.random.PRNGKey(4242 + i * 7 + seed)
            a_fb = np.asarray(conds["fb_tdj"][tid].sample_actions(ob, goals=None, seed=key))
            a_d = np.asarray(conds["density_fb_z"][tid].sample_actions(ob, goals=None, seed=key))
            rng = np.random.default_rng([FF.BANK_SEED, seed, i, BANK_IDX[bank]])
            cands = FF.candidate_actions(a_fb.astype(np.float32), a_d.astype(np.float32), adim, rng)
            for m in METHODS:
                row = ps.loc[(m, bank, i)]
                if int(row["tie"]):
                    continue                     # no best/worst pair at this state
                ip, im = FF.CANDIDATE_NAMES.index(row["best"]), FF.CANDIDATE_NAMES.index(row["worst"])
                acts = cands[[ip, im]]
                arm, ag = arms[m], conds[m][tid]
                z = np.asarray(ag.cond_eval["z"])
                obs_rep = np.repeat(ob[None], 2, axis=0)
                if m.startswith("density_fb"):
                    pred = FF.predict_density(arm, obs_rep, acts, z, pool_obs)
                    clip_frac = neg_mass = 0.0
                else:
                    M, _q = fb_readout(arm.agent.critic, arm.agent.net, obs_rep, acts, z, pool_obs)
                    clip_frac = float((M < 0).mean())
                    neg_mass = float((-np.minimum(M, 0).sum(-1) / np.maximum(np.abs(M).sum(-1), 1e-12)).mean())
                    pos = np.maximum(M, 0.0)
                    den = pos.sum(-1, keepdims=True)
                    pred = np.where(den > 0, pos / np.maximum(den, 1e-12), 1.0 / M.shape[-1])
                mc = roll_futures(ag, envs, rec, acts, K, H, jax.random.PRNGKey(777 + i * 13 + seed))
                p = f"{bank}_{i}_{m}"
                store[f"{p}_mc"] = mc.astype(np.float32)
                store[f"{p}_pred"] = np.asarray(pred, np.float32)
                store[f"{p}_acts"] = acts.astype(np.float32)
                meta.append(dict(critic=m, seed=seed, bank=bank, state=i, task=tid, best=row["best"],
                                 worst=row["worst"], gamma=gamma[m], K=K, H=H,
                                 fb_clip_frac=clip_frac, fb_neg_mass=neg_mass))
            if (i + 1) % 8 == 0 or i + 1 == n:
                print(f"  [sep seed{seed}] {bank} {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"futures_seed{seed}{tag}.npz", **store)
        pd.DataFrame(meta).to_csv(out_dir / f"futures_seed{seed}{tag}.csv", index=False)
    print(f"[sep] seed {seed}: {len(meta)} (state, model) pairs in {time.time() - t0:.0f}s", flush=True)
    return out_dir / f"futures_seed{seed}{tag}.npz"


# ── stage aggregate ─────────────────────────────────────────────────────────────────────

def load_rows(out_dir: Path, tag: Optional[str], cells: Sequence[float] = CELLS,
              vectors: Optional[Dict] = None, vector_cell: float = CELL_MAIN) -> pd.DataFrame:
    """Per-state rows at every grid; if `vectors` (a dict) is given it is filled with the
    (d_hat, d_true) cell vectors at `vector_cell` for the permutation null."""
    fs = sorted(glob.glob(str(out_dir / "futures_seed*.csv")))
    fs = [f for f in fs if (("smoke" not in f) if tag is None else f.endswith(f"{tag}.csv"))]
    assert fs, f"no futures under {out_dir}"
    pool_xyz = cube_xyz_from_obs(np.load(out_dir / f"pool{POOL_N}.npz")["obs"])
    rows = []
    for f in fs:
        meta = pd.read_csv(f)
        d = np.load(f[:-4] + ".npz")
        for r in meta.itertuples():
            p = f"{r.bank}_{r.state}_{r.critic}"
            for cell in cells:
                s, dh, dt = separation_row(d[f"{p}_mc"], d[f"{p}_pred"], pool_xyz, float(r.gamma), cell,
                                           return_vectors=True)
                if vectors is not None and cell == vector_cell:
                    vectors[(r.critic, int(r.seed), r.bank, int(r.state))] = (dh, dt)
                rows.append(dict(critic=r.critic, seed=int(r.seed), bank=r.bank, state=int(r.state),
                                 task=int(r.task), best=r.best, worst=r.worst, cell=cell,
                                 fb_clip_frac=r.fb_clip_frac, fb_neg_mass=r.fb_neg_mass, **s))
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, ts: pd.DataFrame, out_dir: Path, name: str = "fig_separation_by_phase",
         at: Optional[pd.DataFrame] = None, ats: Optional[pd.DataFrame] = None) -> None:
    """Top: separation preserved R per phase. Bottom (when `at` given): cosine alignment of
    the critic's difference-of-futures with the true one, per phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.probes.cube_bottleneck.failure_plots import METHOD_COLOR, METHOD_MARKER

    x = np.arange(len(BANKS))
    w = 0.34
    if at is None:
        fig, ax = plt.subplots(figsize=(3.3, 2.7))
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.3, 4.5), sharex=True,
                                      gridspec_kw=dict(height_ratios=[1, 1], hspace=0.3))
        for j, m in enumerate(METHODS):
            col, mk = METHOD_COLOR[m], METHOD_MARKER[m]
            b = at[at["critic"] == m].set_index("bank").reindex(BANKS)
            xo = x + (j - 0.5) * w
            sb = ats[ats["critic"] == m]
            for bi, bk in enumerate(BANKS):
                v = sb[sb["bank"] == bk]["align_mean"].to_numpy()
                ax2.scatter(np.full(len(v), xo[bi]), v, marker=mk, s=9, color=col, alpha=0.35, lw=0, zorder=1)
            ax2.errorbar(xo, b["align_mean"], yerr=b["align_se"], fmt=mk, ms=6, color=col, ecolor=col,
                         elinewidth=1.0, capsize=2.5, lw=0, zorder=3)
        ax2.axhline(0.0, color="#666666", lw=0.7, ls="--")
        ax2.set_ylabel("alignment of predicted with\ntrue difference (cosine)")
        ax2.set_title("Does the residual point the right way?", fontsize=8.5)
        lo = min(-0.05, float((at["align_mean"] - at["align_se"]).min()) - 0.05)
        hi = max(0.3, float((at["align_mean"] + at["align_se"]).max()) + 0.05)
        ax2.set_ylim(lo, hi)
    for j, m in enumerate(METHODS):
        col, mk = METHOD_COLOR[m], METHOD_MARKER[m]
        b = t[t["critic"] == m].set_index("bank").reindex(BANKS)
        xo = x + (j - 0.5) * w
        sb = ts[ts["critic"] == m]
        for bi, bk in enumerate(BANKS):
            v = sb[sb["bank"] == bk]["R_mean"].to_numpy()
            ax.scatter(np.full(len(v), xo[bi]), v, marker=mk, s=9, color=col, alpha=0.35, lw=0, zorder=1)
        ax.errorbar(xo, b["R_mean"], yerr=b["R_se"], fmt=mk, ms=6, color=col, ecolor=col,
                    elinewidth=1.0, capsize=2.5, lw=0, label=METHOD_LABEL[m], zorder=3)
        axl = ax if at is None else ax2
        for xi, (_, r) in zip(xo, b.iterrows()):
            # mean true separation under the tick label (bottom axis), one row per model
            axl.annotate(f"{r['d_true_mean']:.2f}", (xi, -0.13 - 0.07 * j), xycoords=("data", "axes fraction"),
                         ha="center", va="top", fontsize=6.5, color=col, annotation_clip=False)
    axl.annotate("$D_{true}$", (-0.55, -0.13), xycoords=("data", "axes fraction"), ha="right", va="top",
                 fontsize=6.5, color="#666666", annotation_clip=False)
    ax.axhline(1.0, color="#666666", lw=0.7, ls="--")
    ax.text(len(BANKS) - 1 + 0.5, 1.02, "fully preserved", fontsize=7, color="#666666", ha="right", va="bottom")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("separation preserved\n$D_{pred}\\,/\\,D_{true}$")
    axl.set_xticks(x)
    axl.set_xticklabels([BANK_LABEL[b] for b in BANKS])
    axl.set_xlim(-0.6, len(BANKS) - 0.4)
    ax.set_title("Does the critic separate the futures of $a^+$ and $a^-$?", fontsize=8.5)
    ax.legend(frameon=False, loc="upper left")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {out_dir / name}.pdf/.png")


def stage_aggregate(out_dir: Path, fig_dir: Path, tag: Optional[str] = None, n_perm: int = 200) -> None:
    vectors: Dict = {}
    rows = load_rows(out_dir, tag, vectors=vectors)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(fig_dir / "separation_per_state.csv", index=False)
    tables = {}
    for cell in CELLS:
        sub = rows[rows["cell"] == cell]
        tables[cell] = (phase_table(sub), phase_table(sub, by_seed=True))
        tables[cell][0].assign(cell=cell).to_csv(fig_dir / f"separation_by_phase_cell{int(cell * 100)}cm.csv", index=False)
    t, ts = tables[CELL_MAIN]
    ts.to_csv(fig_dir / "separation_by_phase_per_seed.csv", index=False)
    fb = rows[(rows["critic"] == "fb_tdj") & (rows["cell"] == CELL_MAIN)]
    clip = dict(fb_clip_frac_mean=float(fb["fb_clip_frac"].mean()), fb_neg_mass_mean=float(fb["fb_neg_mass"].mean()),
                n_pairs=int(len(fb)))
    (fig_dir / "separation_fb_clipping.json").write_text(json.dumps(clip, indent=2))
    cols = ["method", "bank", "n", "n_excluded", "d_true_mean", "d_pred_mean", "R_mean", "R_se", "R_median", "fidelity_mean"]
    for cell in CELLS:
        print(f"=== cell {cell * 100:.0f} cm (R = D_pred / D_true; states with D_true < {D_TRUE_MIN} excluded) ===")
        print(tables[cell][0][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"=== FB clipping: {clip}")
    # alignment: per phase at every grid, per seed at the main grid, permutation null at the main grid
    for cell in CELLS:
        a = align_table(rows[rows["cell"] == cell])
        a.assign(cell=cell).to_csv(fig_dir / f"alignment_by_phase_cell{int(cell * 100)}cm.csv", index=False)
        print(f"=== alignment, cell {cell * 100:.0f} cm (cosine of d_hat with d_true; top = top-{TOP_CELLS} |d_true| cells) ===")
        print(a[["method", "bank", "align_n", "align_undefined", "align_mean", "align_se", "align_median", "align_pos",
                 "align_top_mean", "align_top_se", "align_top_pos"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    at = align_table(rows[rows["cell"] == CELL_MAIN])
    ats = align_table(rows[rows["cell"] == CELL_MAIN], by_seed=True)
    ats.to_csv(fig_dir / "alignment_by_phase_per_seed.csv", index=False)
    null = permutation_null(vectors, n_perm=n_perm)
    null.to_csv(fig_dir / "alignment_permutation_null.csv", index=False)
    print(f"=== permutation null ({n_perm} shuffles of the d_true <-> d_hat pairing within (model, bank)), cell {CELL_MAIN * 100:.0f} cm ===")
    print(null.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    plot(t, ts, fig_dir, at=at, ats=ats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=["roll", "aggregate"], default="aggregate")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--banks", nargs="+", default=BANKS)
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--K", type=int, default=K_ROLL)
    ap.add_argument("--H", type=int, default=HORIZON)
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-state-csv", default=str(FIG / "gap_preservation_per_state.csv"))
    ap.add_argument("--out", default=str(ROOT / ENV / "separation"))
    ap.add_argument("--fig", default=str(FIG))
    args = ap.parse_args()
    out, fig = Path(args.out), Path(args.fig)
    if args.smoke:
        stage_roll(args.seeds[0], ["carrying"], out, Path(args.per_state_csv), max_states=3, K=2, H=20, tag="_smoke")
        stage_aggregate(out, fig / "smoke_sep", tag="_smoke")
        return 0
    if args.stage == "roll":
        for s in args.seeds:
            stage_roll(s, args.banks, out, Path(args.per_state_csv), max_states=args.max_states,
                       K=args.K, H=args.H, tag=args.tag)
    else:
        stage_aggregate(out, fig, tag=args.tag or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
