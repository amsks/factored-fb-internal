"""Reward-class coverage analysis, ExoRL walker -- Figure 1 (raw-norm-gated cosine CDFs)
and Figure 2 (k-hot sweep vs its analytic/simulated null).

docs/notes/2026-08-31-z-coverage-brainstorm.md is the design doc this implements ("first
figures": Fig B+C combined, and Fig F). Plain-language summary of the question: an FB-style
method promises that a policy conditioned on z = infer_cond(reward) is good for that
reward, but only for rewards whose inferred z lands where the actor was actually trained
(the training mix of goal-derived z's and random-sphere z's). This script draws many
synthetic reward functions from a few reasonable "reward classes" (k-hot, bounded random
features, white noise, trajectory-smoothed), infers z for each under the checkpoint's own
REAL inference code, and reports where those z's land relative to that training mix and
relative to a random-guessing baseline (a uniformly random direction on the sphere).

EVAL-ONLY. No training-path changes, no checkpoint writes. The encoder forward pass itself
(RLDPCritic._B / HILPCritic._feature_net) and the exact z-readout formula are both taken
verbatim from impls/critics/{rldp,hilp}.py -- this script never recomputes what a reward
"means" to the checkpoint on its own; it only calls that checkpoint's own code, batched over
many reward draws, and checks the batched version against a live call to the real
`infer_cond` before trusting any of the numbers (see `_fidelity_check`).

Usage:
    /dev/shm/.venv-jax/bin/python -m scripts.z_coverage_walker \\
        --method rldp --ckpt exp/exorl-ortho/dmc-walker/rldp_orthoa/dmc-walker_rldp_orthoa_ddpgbc_seed_0 \\
        --step 5000000 --out-dir exp/z_coverage

    /dev/shm/.venv-jax/bin/python -m scripts.z_coverage_walker \\
        --method hilp --ckpt exp/exorl-tdj/dmc-walker/hilp_tdj/dmc-walker_hilp_tdj_ddpgbc_seed_0 \\
        --step 300000 --out-dir exp/z_coverage
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("FB_MUJOCO_SITE_PACKAGES",
                      os.path.expanduser("~/git/Factored-FB/.venv/lib/python3.11/site-packages"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from env import get_adapter, make_config
from env.exorl_env import load_domain_data
from impls.agents import Agent
from impls.critics.base import infer_cond_rows
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent
from scripts.reeval_checkpoint import _recorded_config_overrides
from impls.utils.z_coverage_nulls import (
    project_sphere, cos_to, theta90, resultant_length,
    uniform_cos_cdf, sample_uniform_sphere,
    khot_null_params, sample_khot_null, k_eff,
    permutation_floor_norms,
)

K_SWEEP = (1, 4, 16, 64, 256, 1024)
DEFAULT_TASKS = ("stand", "walk", "run", "spin")   # spin, not flip -- see brainstorm doc S2
INFERENCE_SAMPLES = {"rldp": 50_000, "hilp": 10_000}


# =========================================================================================
# 1. Loading: checkpoint + offline data (thin wrap of scripts/reeval_checkpoint.py's path,
#    so a training run and this analysis can never disagree about how a checkpoint is built)
# =========================================================================================

def load_agent_and_data(critic, actor, domain, run_dir, step, seed):
    base = _recorded_config_overrides(run_dir)
    cfg = make_config(critic, actor, domain, base)
    adapter = get_adapter(cfg)
    env, train_ds, _val_ds = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(critic, actor, seed, jnp.asarray(ex["observations"]),
                         jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, run_dir, step, params_only=True)
    return agent, train_ds, cfg


def sample_inference_rows(train_ds, n, rng):
    """A fixed-seed sample of `n` row indices into the TRAIN split, and the batch those
    rows produce. `idxs` also indexes directly into the FULL offline array any
    `load_domain_data(...)` call returns, because `UnifiedAdapter` builds the train split
    as a PREFIX slice of the full array (env/unified.py: `tr = {k: v[:cut] ...}`)."""
    idxs = rng.integers(0, train_ds.size, size=n)
    rows = train_ds.sample(n, idxs=idxs)
    return idxs, rows


# =========================================================================================
# 2. Per-method raw encoder output + the exact z-readout formula, reused verbatim from the
#    critic and then batched over many reward draws (one encoder forward pass, not one per
#    reward draw). `_fidelity_check` proves the batched version agrees with a live call to
#    the real `infer_cond` before anything downstream trusts it.
# =========================================================================================

class MethodOps:
    """method == 'rldp': raw = B(s') [N,d] (RLDPCritic._B). z_raw(r) = (r @ raw) / N,
    i.e. RLDPCritic.infer_cond's own two lines (rldp.py:459-462), batched over many `r`.

    method == 'hilp': raw = the running-mean-centered phi(s') [N,d] (HILPCritic._feature_net,
    the same tensor infer_cond's lstsq is solved against). z_raw(r) = lstsq(raw, r) per
    column of r, i.e. HILPCritic.infer_cond's own lstsq (hilp.py:344-354), batched over many
    right-hand sides in one solve."""

    def __init__(self, method, agent):
        self.method = method
        self.agent = agent
        self.critic = agent.critic
        self.norm_z = bool(self.critic.norm_z)

    def raw_matrix(self, obs):
        obs = jnp.asarray(obs)
        if self.method == "rldp":
            out = self.critic._B(self.agent.net["backward"].params, obs)
        else:
            out = self.critic._feature_net(self.agent.net["hilp_phi1"].params,
                                           self.agent.net["hilp_stats"].params, obs)
        return np.asarray(out, dtype=np.float64)

    def z_raw_batch(self, raw, R):
        """raw: [N,d] encoder output. R: [M,N] reward draws (one per row). -> [M,d] z_raw,
        BEFORE the sqrt(d) projection (angular statistics are invariant to that positive
        rescaling, so everything downstream works directly on z_raw; see module docstring
        of impls/utils/z_coverage_nulls.py)."""
        R = np.asarray(R, dtype=np.float64)
        N = raw.shape[0]
        assert R.shape[1] == N
        if self.method == "rldp":
            return (R @ raw) / N
        # HILP: lstsq(phi, r) solved for every column of R at once.
        sol, *_ = np.linalg.lstsq(raw, R.T, rcond=None)
        return sol.T

    def real_infer_cond_z(self, obs, r):
        """One live call to the checkpoint's REAL infer_cond, for the fidelity check."""
        out = self.critic.infer_cond(self.agent.net, jnp.asarray(obs), jnp.asarray(r))
        return np.asarray(out["z"], dtype=np.float64)


def fidelity_check(ops, obs, raw, rng, atol=1e-4):
    """Prove z_raw_batch's fast path agrees with a live infer_cond call (same direction,
    same order of magnitude), so every number this script reports is provably the
    checkpoint's own computation, not a reimplementation of it."""
    r_test = rng.normal(size=raw.shape[0])
    fast = ops.z_raw_batch(raw, r_test[None, :])[0]
    fast_proj = project_sphere(fast[None, :])[0]
    real = ops.real_infer_cond_z(obs, r_test)
    cos = cos_to(fast_proj, real)
    err = np.linalg.norm(fast_proj - real) / (np.linalg.norm(real) + 1e-12)
    ok = cos > 1 - 1e-4 and err < 1e-2
    print(f"[fidelity] batched-readout vs live infer_cond: cos={cos:.6f} rel_err={err:.2e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise RuntimeError("fidelity check failed: batched readout disagrees with the "
                           "checkpoint's real infer_cond -- refusing to report numbers "
                           "from an unverified fast path.")


# =========================================================================================
# 3. Reward-class generators, all defined ON THE INFERENCE SET rows (indices 0..N-1 of the
#    sampled batch) so every class is exactly what infer_cond would see.
# =========================================================================================

def khot_scattered(N, k, M, rng):
    R = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        hot = rng.choice(N, size=k, replace=False)
        R[m, hot] = 1.0
    return R


def khot_clustered(obs_std, k, M, rng):
    """S_k = the k nearest ROWS (in standardized observation space, since walker's raw
    obs mixes orientation/height/velocity scales with very different magnitudes) of a
    random seed row, both drawn from the inference set itself."""
    N = obs_std.shape[0]
    R = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        seed = rng.integers(N)
        d2 = np.sum((obs_std - obs_std[seed]) ** 2, axis=1)
        hot = np.argpartition(d2, k - 1)[:k]
        R[m, hot] = 1.0
    return R


def c2a_rff(obs_std, M, rng, D=128, tau=1.0):
    """r(s) = tanh(w^T RFF(s) / tau), RFF at bandwidth 1 in standardized-observation units
    (one bandwidth for this first pass; the brainstorm doc's fuller sweep is future work)."""
    N, dobs = obs_std.shape
    Omega = rng.normal(size=(D, dobs))
    b = rng.uniform(0, 2 * np.pi, size=D)
    phi = np.sqrt(2.0 / D) * np.cos(obs_std @ Omega.T + b[None, :])   # [N, D]
    R = np.empty((M, N), dtype=np.float64)
    for m in range(M):
        w = rng.normal(size=D) / np.sqrt(D)
        R[m] = np.tanh((phi @ w) / tau)
    return R


def c2c_white_noise(N, M, rng):
    return rng.uniform(-1.0, 1.0, size=(M, N))


def episode_bounds(terminals):
    """-> (initial_locs, terminal_locs) sorted arrays of each trajectory's first/last row,
    read off the REAL terminals column (not an assumed fixed episode length)."""
    terminal_locs = np.nonzero(np.asarray(terminals) > 0)[0]
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    return initial_locs, terminal_locs


def c3a_heat_kernel(full_size, initial_locs, terminal_locs, idxs, k, W, lam, M, rng):
    """r_W(s_t) = sum_{|d|<=W} lam^|d| * r0(s_{t+d}) / (normalizer), r0 a k-hot seed drawn
    from the FULL train array (not just the inference sample), so neighbors are real
    trajectory neighbors. Evaluated at the inference set's own `idxs`. Episode-boundary
    aware: a window never crosses into a neighboring trajectory."""
    N = idxs.shape[0]
    deltas = np.arange(-W, W + 1)
    weights = lam ** np.abs(deltas)
    ep_start = initial_locs[np.searchsorted(initial_locs, idxs, side="right") - 1]
    ep_end = terminal_locs[np.searchsorted(terminal_locs, idxs)]
    R = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        seeds = rng.choice(full_size, size=k, replace=False)
        seed_mask = np.zeros(full_size, dtype=np.float64)
        seed_mask[seeds] = 1.0
        acc = np.zeros(N, dtype=np.float64)
        wsum = np.zeros(N, dtype=np.float64)
        for delta, w in zip(deltas, weights):
            cand = idxs + delta
            valid = (cand >= ep_start) & (cand <= ep_end)
            contrib = np.where(valid, seed_mask[np.clip(cand, 0, full_size - 1)], 0.0)
            acc += w * contrib
            wsum += np.where(valid, w, 0.0)
        R[m] = acc / np.maximum(wsum, 1e-12)
    return R


# =========================================================================================
# 4. Analysis per class: raw norms (pre-projection), theta90 (cos-invariant to projection),
#    cos-to-B-mean, permutation floor on one representative draw, within-reward bootstrap
#    jitter on one representative draw.
# =========================================================================================

def bootstrap_jitter(ops, raw_full, N_full, r_single, rng, n_boot=15, frac=0.5):
    """Re-infer z for ONE fixed reward over `n_boot` resamples of the inference set (with
    replacement, `frac` of N rows each time) -- Monte Carlo jitter from the finite
    inference sample, as opposed to genuine across-reward spread (doc pitfall 4)."""
    m = max(int(frac * N_full), 8)
    zs = []
    for _ in range(n_boot):
        sub = rng.integers(0, N_full, size=m)
        z = ops.z_raw_batch(raw_full[sub], r_single[sub][None, :])[0]
        zs.append(z)
    zs = np.stack(zs)
    return theta90(zs)


def analyze_class(name, R_raw, raw, mu_dir, rng, n_perm=200, center=False):
    R = R_raw - R_raw.mean(axis=1, keepdims=True) if center else R_raw
    Z = ops_global.z_raw_batch(raw, R)                      # [M, d]
    norms = np.linalg.norm(Z, axis=1)
    cos_muB = cos_to(Z, mu_dir)
    t90 = theta90(Z)
    Rbar = resultant_length(Z)
    # permutation floor from ONE representative (first) reward draw of this class
    floor = permutation_floor_norms(R[0], raw, num_shuffles=n_perm,
                                    rng=np.random.RandomState(rng.integers(1 << 30)))
    return dict(name=name, centered=center, Z=Z, raw_norms=norms, cos_to_muB=cos_muB,
               theta90_deg=t90, resultant_length=Rbar, k_eff=float(np.mean([k_eff(r) for r in R])),
               floor_norms=floor)


# =========================================================================================
# 5. Figures
# =========================================================================================

def fig1_cos_and_norms(method, classes, anchors, uniform_null_cos, hindsight_cos, d, out_path):
    raw_classes = [c for c in classes if not c["centered"]]
    n = len(raw_classes)
    anchor_colors = plt.cm.Dark2(np.linspace(0, 1, max(len(anchors), 1)))
    class_colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig = plt.figure(figsize=(4.2 * n, 9.5))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.35], hspace=0.55, wspace=0.35)

    # -- row 1: one small-multiple per class -- pre-projection raw-norm validity gate ----
    for i, (c, col) in enumerate(zip(raw_classes, class_colors)):
        ax = fig.add_subplot(gs[0, i])
        logn = np.log10(c["raw_norms"] + 1e-12)
        ax.hist(logn, bins=15, color=col, alpha=0.75)
        lo, hi = np.percentile(np.log10(c["floor_norms"] + 1e-12), [1, 99])
        ax.axvspan(lo, hi, color="gray", alpha=0.25, label="perm. noise floor (1-99pct)")
        for (name, a), acol in zip(anchors.items(), anchor_colors):
            ax.axvline(np.log10(a["raw_norm"] + 1e-12), color=acol, linestyle=":", linewidth=1.3)
        ax.set_title(c["name"], fontsize=9)
        ax.set_xlabel("log10 ||z_raw||", fontsize=8)
        if i == 0:
            ax.set_ylabel("count")
            ax.legend(fontsize=6, loc="upper left")

    # -- row 2: cos-to-class-mean CDFs + analytic uniform null + prior halves ------------
    ax_cos = fig.add_subplot(gs[1, :])
    grid = np.linspace(-1, 1, 400)
    ax_cos.plot(grid, uniform_cos_cdf(grid, d), "k--", linewidth=2,
               label=f"uniform-sphere null Beta(1/2,{(d-1)/2:.0f})")
    for c, col in zip(raw_classes, class_colors):
        mdir = c["Z"].mean(axis=0)
        mdir = mdir / (np.linalg.norm(mdir) + 1e-12)
        for variant, ls, alpha in ((c, "-", 1.0),
                                   (next(x for x in classes if x["name"] == c["name"]
                                        and x["centered"]), "--", 0.7)):
            cos_vals = cos_to(variant["Z"], mdir)
            xs = np.sort(cos_vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            lbl = c["name"] + (" (raw)" if ls == "-" else " (reward-centered)")
            ax_cos.plot(xs, ys, color=col, linewidth=1.8, linestyle=ls, alpha=alpha, label=lbl)
    xs = np.sort(uniform_null_cos)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax_cos.plot(xs, ys, color="dimgray", linewidth=1.4, alpha=0.9,
               label="uniform prior half (to encoder-output mean direction)")
    xs = np.sort(hindsight_cos)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax_cos.plot(xs, ys, color="saddlebrown", linewidth=1.4, alpha=0.9,
               label="hindsight prior half (to encoder-output mean direction)")
    for (name, a), acol in zip(anchors.items(), anchor_colors):
        ax_cos.axvline(a["cos_to_muB"], color=acol, linestyle=":", linewidth=1.5)
        ax_cos.text(a["cos_to_muB"], 1.01, name, rotation=0, fontsize=8, va="bottom",
                   ha="center", color=acol)
    ax_cos.set_xlabel("cosine similarity")
    ax_cos.set_ylabel("CDF")
    ax_cos.set_title("cos to each class's own mean direction (solid=raw, dashed=reward-"
                     "centered) vs the uniform-sphere null and the training-prior halves\n"
                     "colored ticks = real task-reward anchors' cos to the pooled encoder-"
                     "output mean direction (the gauge check)")
    ax_cos.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax_cos.set_xlim(-1.05, 1.05)
    ax_cos.set_ylim(-0.02, 1.08)

    fig.suptitle(f"{method.upper()} walker -- Figure 1: reward-class angular coverage, "
                f"gated by the raw-norm validity check", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_ksweep(method, sweep_scattered, sweep_clustered, sweep_null, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ks = K_SWEEP

    ax = axes[0]
    ax.plot(ks, [sweep_scattered[k]["theta90"] for k in ks], "o-", label="scattered", color="C0")
    ax.plot(ks, [sweep_clustered[k]["theta90"] for k in ks], "s-", label="clustered", color="C1")
    if sweep_null is not None:
        ax.plot(ks, [sweep_null[k]["theta90"] for k in ks], "k--",
               label=sweep_null["label"] if "label" in sweep_null else "null")
    ax.set_xscale("log"); ax.set_xlabel("k (hot-set size)")
    ax.set_ylabel("theta_90 (deg, cap containing 90% of class z's)")
    ax.set_title(f"{method.upper()} walker -- k-hot angular spread vs k")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ks, [sweep_scattered[k]["cos_to_muB"] for k in ks], "o-", label="scattered", color="C0")
    ax.plot(ks, [sweep_clustered[k]["cos_to_muB"] for k in ks], "s-", label="clustered", color="C1")
    if sweep_null is not None:
        ax.plot(ks, [sweep_null[k]["cos_to_muB"] for k in ks], "k--", label="null")
    ax.set_xscale("log"); ax.set_xlabel("k (hot-set size)")
    ax.set_ylabel("mean cos(z, encoder-output mean direction)")
    ax.set_title("collapse-onto-the-gauge-mean check")
    ax.legend(fontsize=8)

    fig.suptitle(f"{method.upper()} walker -- k-hot sweep vs its own null "
                f"({'analytic CLT' if method == 'rldp' else 'simulated (see results note)'})")
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=150)
    plt.close(fig)


# =========================================================================================
# main
# =========================================================================================

ops_global = None   # set in main(); analyze_class reads it (single-checkpoint script)


def main():
    global ops_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rldp", "hilp"], required=True)
    ap.add_argument("--ckpt", required=True, help="run dir, e.g. exp/.../dmc-walker_rldp_orthoa_ddpgbc_seed_0")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--critic", default=None, help="critic name if different from --method "
                    "(e.g. rldp_orthoa, hilp_tdj); default = --method")
    ap.add_argument("--actor", default="ddpgbc")
    ap.add_argument("--domain", default="dmc-walker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="exp/z_coverage")
    ap.add_argument("--n-infer", type=int, default=None)
    ap.add_argument("--n-draws", type=int, default=30, help="reward draws per class")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=15)
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    critic_name = args.critic or args.method
    N = args.n_infer or INFERENCE_SAMPLES[args.method]
    rng = np.random.default_rng(args.rng_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] critic={critic_name} ckpt={args.ckpt} step={args.step}", flush=True)
    agent, train_ds, cfg = load_agent_and_data(critic_name, args.actor, args.domain,
                                               args.ckpt, args.step, args.seed)
    ops = MethodOps(args.method, agent)
    ops_global = ops

    idxs, rows = sample_inference_rows(train_ds, N, rng)
    obs_col = infer_cond_rows(agent.critic)
    assert obs_col == "next_observations", obs_col
    obs = np.asarray(rows[obs_col], dtype=np.float64)
    print(f"[infer set] rho: N={N} rows, obs_dim={obs.shape[1]}, marginal={obs_col}", flush=True)

    raw = ops.raw_matrix(obs)                     # [N, d] B(s') or centered phi(s')
    d = raw.shape[1]
    fidelity_check(ops, obs, raw, rng)

    mu_dir = raw.mean(axis=0)
    mu_dir_norm = np.linalg.norm(mu_dir)
    mu_dir_unit = mu_dir / (mu_dir_norm + 1e-12)
    print(f"[gauge] ||mean encoder output|| = {mu_dir_norm:.4f} over N={N} rows, d={d}",
         flush=True)

    obs_mean, obs_std = obs.mean(axis=0), obs.std(axis=0) + 1e-6
    obs_std_z = (obs - obs_mean) / obs_std

    # -- task-reward anchors -----------------------------------------------------------
    anchors = {}
    for t in args.tasks:
        rew = np.asarray(load_domain_data({**cfg, "exorl_task": t})["rewards"],
                         dtype=np.float64)[idxs]
        z = ops.z_raw_batch(raw, rew[None, :])[0]
        anchors[t] = dict(raw_norm=float(np.linalg.norm(z)),
                          cos_to_muB=float(cos_to(z, mu_dir_unit)),
                          z=z)
        print(f"[anchor] {t}: raw_norm={anchors[t]['raw_norm']:.4f} "
             f"cos_to_muB={anchors[t]['cos_to_muB']:.4f}", flush=True)

    # -- prior halves (empirical) --------------------------------------------------------
    uniform_half = rng.normal(size=(2000, d))                              # sphere direction only
    if args.method == "rldp":
        hindsight_half = raw.copy()                                        # project_z(B(g)), all rho rows
    else:
        cov = (raw.T @ raw) / raw.shape[0]
        hindsight_half = raw @ np.linalg.pinv(cov)                         # sf.py:336-338, verbatim
    uniform_half_cos = cos_to(uniform_half, mu_dir_unit)
    hindsight_half_cos = cos_to(hindsight_half, mu_dir_unit)

    # =====================================================================================
    # FIGURE 1
    # =====================================================================================
    print("[fig1] building reward classes...", flush=True)
    init_locs, term_locs = episode_bounds(train_ds.dataset["terminals"])
    classes_spec = [
        ("1-hot-clustered", lambda: khot_clustered(obs_std_z, 1, args.n_draws, rng)),
        ("64-hot-scattered", lambda: khot_scattered(N, 64, args.n_draws, rng)),
        ("C2a-rff-tanh", lambda: c2a_rff(obs_std_z, args.n_draws, rng)),
        ("C2c-white-noise", lambda: c2c_white_noise(N, args.n_draws, rng)),
        # Base hot-set for C3a is much larger than the 64 used for "64-hot-scattered":
        # the full offline dataset has millions of rows and the inference set rho is an
        # i.i.d. sample of N << that, so a WINDOW=5 dilation of a k=64 seed set overlaps
        # essentially none of rho (expected ~N*k*11/dataset_size rows -- single digits at
        # N=50k). k=4096 buys a few hundred expected rho rows actually inside the smoothed
        # window; still a small minority of rho, which is itself a finding (see the
        # results note) -- a trajectory-LOCAL reward class is mostly invisible to an i.i.d.
        # inference sample at this N, structurally, independent of the encoder.
        ("C3a-heat-smooth-W5", lambda: c3a_heat_kernel(train_ds.size, init_locs, term_locs,
                                                       idxs, 4096, 5, 0.8, args.n_draws, rng)),
    ]
    classes = []
    for name, gen in classes_spec:
        R = gen()
        classes.append(analyze_class(name, R, raw, mu_dir_unit, rng, n_perm=args.n_perm,
                                     center=False))
        classes.append(analyze_class(name, R, raw, mu_dir_unit, rng, n_perm=args.n_perm,
                                     center=True))

    out1 = os.path.join(args.out_dir, f"{args.method}_walker_fig1_cos_cdf_rawnorm")
    fig1_cos_and_norms(args.method, classes, anchors, uniform_half_cos, hindsight_half_cos,
                      d, out1)
    print(f"[fig1] wrote {out1}.pdf/.png", flush=True)

    # per-reward jitter vs across-reward spread (one representative class + the walk anchor)
    rep = classes_spec[1]  # 64-hot-scattered
    R_rep = khot_scattered(N, 64, 1, rng)
    jitter_theta90 = bootstrap_jitter(ops, raw, N, R_rep[0], rng, n_boot=args.n_boot)
    across_theta90 = [c for c in classes if c["name"] == "64-hot-scattered" and not c["centered"]][0]["theta90_deg"]
    jitter_ratio = jitter_theta90 / max(across_theta90, 1e-6)
    print(f"[jitter] 64-hot-scattered: within-reward theta90={jitter_theta90:.2f} deg, "
         f"across-reward theta90={across_theta90:.2f} deg, ratio={jitter_ratio:.3f}",
         flush=True)

    # =====================================================================================
    # FIGURE 2: k-hot sweep
    # =====================================================================================
    print("[fig2] k-hot sweep...", flush=True)
    sweep_scattered, sweep_clustered = {}, {}
    for k in K_SWEEP:
        Rs = khot_scattered(N, k, args.n_draws, rng)
        Zs = ops.z_raw_batch(raw, Rs)
        sweep_scattered[k] = dict(theta90=theta90(Zs), cos_to_muB=float(np.mean(cos_to(Zs, mu_dir_unit))))

        Rc = khot_clustered(obs_std_z, k, args.n_draws, rng)
        Zc = ops.z_raw_batch(raw, Rc)
        sweep_clustered[k] = dict(theta90=theta90(Zc), cos_to_muB=float(np.mean(cos_to(Zc, mu_dir_unit))))
        print(f"  k={k:5d}  scattered theta90={sweep_scattered[k]['theta90']:6.2f}  "
             f"clustered theta90={sweep_clustered[k]['theta90']:6.2f}", flush=True)

    if args.method == "rldp":
        Sigma_B = np.cov(raw.T, bias=True)
        sweep_null = {"label": "analytic CLT null (sum of k random B-rows)"}
        for k in K_SWEEP:
            samples = sample_khot_null(mu_dir, Sigma_B, k, N, num_samples=args.n_draws,
                                       rng=np.random.RandomState(int(rng.integers(1 << 30))))
            sweep_null[k] = dict(theta90=theta90(samples),
                                 cos_to_muB=float(np.mean(cos_to(samples, mu_dir_unit))))
    else:
        # HILP's lstsq readout is not linear in the reward, so the CLT closed form does
        # not apply; per the design decision, SIMULATE the null instead of deriving it.
        # The null IS the scattered-k-hot curve itself (doc pitfall 6(ii): scattered k-hot
        # is determined by encoder-row statistics alone), so this duplicates
        # sweep_scattered under a different label -- kept separate so the plot's legend is
        # honest about what "null" means for this method.
        sweep_null = {"label": "simulated null (= scattered k-hot itself; see results note)"}
        for k in K_SWEEP:
            sweep_null[k] = dict(sweep_scattered[k])

    out2 = os.path.join(args.out_dir, f"{args.method}_walker_fig2_ksweep")
    fig2_ksweep(args.method, sweep_scattered, sweep_clustered, sweep_null, out2)
    print(f"[fig2] wrote {out2}.pdf/.png", flush=True)

    # =====================================================================================
    # summary json
    # =====================================================================================
    summary = dict(
        method=args.method, critic=critic_name, ckpt=args.ckpt, step=args.step,
        domain=args.domain, z_dim=d, n_infer=N, n_draws=args.n_draws, rng_seed=args.rng_seed,
        mu_dir_norm=float(mu_dir_norm),
        anchors={t: {"raw_norm": a["raw_norm"], "cos_to_muB": a["cos_to_muB"]}
                for t, a in anchors.items()},
        classes=[{"name": c["name"], "centered": c["centered"],
                 "theta90_deg": c["theta90_deg"], "resultant_length": c["resultant_length"],
                 "mean_raw_norm": float(np.mean(c["raw_norms"])),
                 "floor_median": float(np.median(c["floor_norms"])),
                 "floor_p99": float(np.percentile(c["floor_norms"], 99)),
                 "mean_cos_to_muB": float(np.mean(c["cos_to_muB"])),
                 "k_eff": c["k_eff"],
                 "below_floor": bool(np.mean(c["raw_norms"]) <= np.percentile(c["floor_norms"], 99))}
                for c in classes],
        jitter=dict(within_reward_theta90=float(jitter_theta90),
                   across_reward_theta90=float(across_theta90),
                   ratio=float(jitter_ratio)),
        ksweep=dict(scattered={str(k): v for k, v in sweep_scattered.items()},
                   clustered={str(k): v for k, v in sweep_clustered.items()},
                   null_label=sweep_null["label"],
                   null={str(k): sweep_null[k] for k in K_SWEEP}),
        prior_halves=dict(
            uniform_mean_cos_to_muB=float(np.mean(uniform_half_cos)),
            hindsight_mean_cos_to_muB=float(np.mean(hindsight_half_cos)),
            uniform_theta90=theta90(uniform_half),
            hindsight_theta90=theta90(hindsight_half)),
    )
    out_json = os.path.join(args.out_dir, f"{args.method}_walker_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
