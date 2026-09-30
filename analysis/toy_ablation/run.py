"""
Toy ablation of Density-FB's three changes to FB, on the paper's nine-state
bottleneck MDP (scripts/didactic/graph_bottleneck_td.py, leak 0, gamma 0.8).

Arms (same MDP, same exact transitions, same rank d, same fixed uniform
continuation policy, uniform reference weight rho = 1/n on candidate states):

  A  fb            linear F.B fits the ratio M/rho = n M with squared error;
                   value read as F.z, z = E_rho[B r].            (FB as shipped)
  B  fb_norm       linear F.B fits the normalized measure (1-gamma) M with
                   squared error; same readout (scaled by the known constant).
  C  softsq        softmax(F.B) fitted to (1-gamma) M with 0.5 sum_j (p_j - t_j)^2;
                   value read as sum_j p_j r_j / (1-gamma).
  D  density       softmax(F.B) fitted to (1-gamma) M with cross-entropy;
                   same integration readout.                   (Density-FB)
  E  density_lin   the model of D, but the value read as F.z with z = E_rho[B r]
                   (FB's linear shortcut applied to the softmax model).
  F  density_soft  D with the immediate one-hot replaced by a narrow kernel over
                   graph distance (sigma = 0.5 edges), i.e. the target is
                   (1-gamma) M K instead of (1-gamma) M.

Closed form: A and B are truncated SVDs (exact optimum); C, D, F are Adam fits
of the population objective against the exact target (5 seeds, single init
each); E reuses D's fit.  Trained: the didactic TD harness
(scripts/didactic/bottleneck_family_td.py: exact population expectations,
bootstrapped target network, Adam, 50k steps, tau 0.005, learning rates
{0.001, 0.01}), 5 seeds, methods fb / normsq / softsq / density / density_soft;
E is read from the trained density parameters.

Metric: fraction of the exact action gap at state 7 (action 0 -> goal minus
action 1 -> back into the group) preserved by the fit, plus whether the fit
still prefers action 0, plus the retained squared mass of the fit on the
normalized-measure scale, 1 - ||P_hat - (1-gamma)M||_F^2 / ||(1-gamma)M||_F^2.

Run (CPU, ~3 min):
    JAX_PLATFORMS=cpu python analysis/toy_ablation/run.py
    JAX_PLATFORMS=cpu python analysis/toy_ablation/run.py --figures-only
Outputs: analysis/toy_ablation/{closed.csv,trained.csv,exact.json,run_log.txt},
         figures/toy_ablation/fig_toy_ablation.{pdf,png} (+ per-panel files).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import csv
import json
import pathlib
import sys
import time
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp
import optax

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "didactic"))
import bottleneck_family_td as bf      # noqa: E402  (sets jax x64)
import graph_bottleneck_td as gb       # noqa: E402

FIG_DIR = REPO / "figures" / "toy_ablation"
LOG_LINES = []


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# ----------------------------------------------------------------------------
# 1. The MDP and its exact successor measure (verbatim from the didactic scripts).
# ----------------------------------------------------------------------------
GAMMA = 0.8
LEAK = 0.0
P3 = gb.build_graph(LEAK)                 # (n, 2, n)
N = P3.shape[0]
KERNEL = P3.reshape(-1, N)                # (2n, n), row 2s+a
PI = np.full((N, 2), 0.5)                 # fixed uniform continuation policy
T_EXACT = bf.exact(KERNEL, PI, GAMMA)     # (1-gamma) M, rows sum to 1
M_EXACT = T_EXACT / (1.0 - GAMMA)
RHO = np.full(N, 1.0 / N)                 # uniform reference weight (didactic convention)
GOAL, DECISION = gb.GOAL, gb.DECISION
R_GOAL = np.zeros(N); R_GOAL[GOAL] = 1.0
Q_EXACT = M_EXACT @ R_GOAL                # (2n,)
ROW0, ROW1 = 2 * DECISION, 2 * DECISION + 1
GAP_EXACT = Q_EXACT[ROW0] - Q_EXACT[ROW1]  # action 0 (to goal) minus action 1 (back)
RANKS = list(range(2, N + 1))
SEEDS = [0, 1, 2, 3, 4]
SOFT_SIGMA = 0.5                          # kernel width in graph edges for arm F


def graph_distance():
    adj = [set() for _ in range(N)]
    for s, (t0, t1) in enumerate(gb.EDGES):
        for t in (t0, t1):
            if t != s:
                adj[s].add(int(t)); adj[int(t)].add(s)
    dist = np.full((N, N), np.inf)
    for s in range(N):
        dist[s, s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1; q.append(v)
    return dist


DIST = graph_distance()
K_SOFT = np.exp(-DIST ** 2 / (2 * SOFT_SIGMA ** 2))
K_SOFT = K_SOFT / K_SOFT.sum(axis=1, keepdims=True)       # row-stochastic smoothing kernel
T_SOFT = T_EXACT @ K_SOFT                                  # fixed point of the soft-target TD


# ----------------------------------------------------------------------------
# 2. Readouts and metrics shared by closed-form and trained arms.
# ----------------------------------------------------------------------------
def gap_stats(Q):
    gap = float(Q[ROW0] - Q[ROW1])
    return dict(q0=float(Q[ROW0]), q1=float(Q[ROW1]), gap=gap,
                gap_ratio=gap / GAP_EXACT, argmax_correct=bool(gap > 0))


def retained_mass(P_hat, T=T_EXACT):
    return float(1.0 - np.sum((P_hat - T) ** 2) / np.sum(T ** 2))


def linear_readout(F, B, r=R_GOAL):
    """FB's shortcut: z = E_rho[B r], value = F.z (uniform rho)."""
    z = B.T @ (RHO * r)
    return F @ z


def calibrate(v, Q=Q_EXACT):
    """Best scalar a (least squares over all 18 rows) so that a*v + b ~ Q; returns a
    and the Pearson correlation.  Used only for arm E, whose scale is arbitrary."""
    vc = v - v.mean(); qc = Q - Q.mean()
    a = float(vc @ qc / max(vc @ vc, 1e-300))
    corr = float(vc @ qc / max(np.sqrt((vc @ vc) * (qc @ qc)), 1e-300))
    return a, corr


def e_arm_rows(F, B, base):
    """Arm E from a softmax model's (F, B): raw linear readout and scale-calibrated."""
    v = linear_readout(F, B)
    a, corr = calibrate(v)
    raw = gap_stats(v)
    cal = gap_stats(a * v)
    return dict(base, arm="E", method="density_lin",
                q0=raw["q0"], q1=raw["q1"], gap=raw["gap"], gap_ratio=raw["gap_ratio"],
                argmax_correct=raw["argmax_correct"],
                gap_ratio_calibrated=cal["gap_ratio"], calib_scale=a, corr_with_exact=corr)


# ----------------------------------------------------------------------------
# 3. Closed-form fits.
# ----------------------------------------------------------------------------
def svd_fit(X, d):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :d] * s[:d][None, :], Vt[:d].T


def _softmax_fit(target, d, seed, loss_kind, steps, lr):
    """Rank-d softmax model fitted to `target` (rows are distributions) by Adam from a
    single normal(0, 0.1) init.  loss_kind: 'ce' or 'sq' (0.5 * sum_j (p_j - t_j)^2)."""
    rng = np.random.default_rng(seed)
    params = {"f": jnp.array(rng.normal(0, 0.1, (2 * N, d))),
              "b": jnp.array(rng.normal(0, 0.1, (N, d)))}
    tgt = jnp.array(target)

    def loss_fn(p):
        logits = p["f"] @ p["b"].T
        if loss_kind == "ce":
            return -jnp.mean(jnp.sum(tgt * jax.nn.log_softmax(logits, axis=-1), axis=-1))
        return 0.5 * jnp.mean(jnp.sum((jax.nn.softmax(logits, axis=-1) - tgt) ** 2, axis=-1))

    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def train(carry):
        def body(c, _):
            p, st = c
            g = jax.grad(loss_fn)(p)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), None
        return jax.lax.scan(body, carry, None, length=steps)[0]

    params, _ = train((params, state))
    F, B = np.asarray(params["f"]), np.asarray(params["b"])
    logits = F @ B.T
    P_hat = np.exp(logits - logits.max(axis=1, keepdims=True))
    P_hat /= P_hat.sum(axis=1, keepdims=True)
    return F, B, P_hat, float(loss_fn(params))


CLOSED_STEPS = 10000
CLOSED_LR = 1e-2


def run_closed_form():
    rows = []
    t0 = time.time()
    for d in RANKS:
        # --- A: FB, ratio target n M, squared error, linear readout ---
        F, B = svd_fit(N * M_EXACT, d)
        Q = linear_readout(F, B)                       # (1/n) F B^T r = M_hat r
        P_hat = (1 - GAMMA) * (F @ B.T) / N
        rows.append(dict(arm="A", method="fb", d=d, seed=-1, **gap_stats(Q),
                         retained_mass=retained_mass(P_hat),
                         own_loss=float(0.5 * np.mean((F @ B.T - N * M_EXACT) ** 2))))
        # --- B: FB with the normalized target (1-gamma) M, same readout ---
        F, B = svd_fit(T_EXACT, d)
        Q = linear_readout(F, B) * N / (1 - GAMMA)     # known constant puts F.z on Q's scale
        P_hat = F @ B.T
        rows.append(dict(arm="B", method="fb_norm", d=d, seed=-1, **gap_stats(Q),
                         retained_mass=retained_mass(P_hat),
                         own_loss=float(0.5 * np.mean((P_hat - T_EXACT) ** 2))))
        for sd in SEEDS:
            seed = 1000 * sd + 17 * d
            # --- C: softmax + squared error ---
            F, B, P_hat, loss = _softmax_fit(T_EXACT, d, seed, "sq", CLOSED_STEPS, CLOSED_LR)
            Q = P_hat @ R_GOAL / (1 - GAMMA)
            rows.append(dict(arm="C", method="softsq", d=d, seed=sd, **gap_stats(Q),
                             retained_mass=retained_mass(P_hat), own_loss=loss))
            # --- D: softmax + cross-entropy (Density-FB) ---
            F, B, P_hat, loss = _softmax_fit(T_EXACT, d, seed, "ce", CLOSED_STEPS, CLOSED_LR)
            Q = P_hat @ R_GOAL / (1 - GAMMA)
            rows.append(dict(arm="D", method="density", d=d, seed=sd, **gap_stats(Q),
                             retained_mass=retained_mass(P_hat), own_loss=loss))
            # --- E: D's model, FB's linear readout ---
            rows.append(e_arm_rows(F, B, dict(d=d, seed=sd, retained_mass=retained_mass(P_hat),
                                              own_loss=loss)))
            # --- F: D with the soft immediate target ---
            F, B, P_hat, loss = _softmax_fit(T_SOFT, d, seed, "ce", CLOSED_STEPS, CLOSED_LR)
            Q = P_hat @ R_GOAL / (1 - GAMMA)
            rows.append(dict(arm="F", method="density_soft", d=d, seed=sd, **gap_stats(Q),
                             retained_mass=retained_mass(P_hat),
                             retained_mass_vs_soft_target=retained_mass(P_hat, T_SOFT),
                             own_loss=loss))
        log(f"  [closed] d={d} done ({time.time() - t0:.1f}s)")
    return rows


# ----------------------------------------------------------------------------
# 4. Trained arms: the didactic TD harness, extended with the soft-target method and
#    with predictions/parameters returned so E and the retained mass can be computed.
# ----------------------------------------------------------------------------
TRAIN_STEPS = 50000
TRAIN_TAU = 0.005
TRAIN_RATES = [0.001, 0.01]
TRAIN_METHODS = ["fb", "normsq", "softsq", "density", "density_soft"]
ARM_OF = {"fb": "A", "normsq": "B", "softsq": "C", "density": "D", "density_soft": "F"}


def _base_method(method):
    return "density" if method == "density_soft" else method


def loss_any(params, target, kernel, pi, gamma, method, sphere, ksoft):
    if method != "density_soft":
        return bf.loss_fn(params, target, kernel, pi, gamma, method, sphere)
    n = kernel.shape[1]
    tp = jax.lax.stop_gradient(bf.predict(target, "density", sphere)).reshape(n, 2, n)
    v = jnp.sum(pi[:, :, None] * tp, axis=1)
    labels = (1 - gamma) * (kernel @ ksoft) + gamma * (kernel @ v)
    return -jnp.mean(jnp.sum(
        labels * jax.nn.log_softmax(params["f"] @ params["b"].T, axis=-1), axis=-1))


def train_group(d, method, seeds, rates, steps=TRAIN_STEPS, tau=TRAIN_TAU):
    """Mirror of bottleneck_family_td.run_kernel (same init, optimizer, target EMA,
    vmapped seeds x rates) returning predictions and final parameters."""
    kernel, jp, ks = jnp.array(KERNEL), jnp.array(PI), jnp.array(K_SOFT)
    cfg = [{"seed": s, "lr": r} for s in seeds for r in rates]

    def init(c):
        rng = np.random.default_rng(c["seed"])
        return {"f": jnp.array(rng.normal(0, 0.1, (2 * N, d))),
                "b": jnp.array(rng.normal(0, 0.1, (N, d)))}

    params = jax.tree.map(lambda *x: jnp.stack(x), *[init(c) for c in cfg])
    opt = optax.scale_by_adam()
    state = jax.vmap(opt.init)(params)
    lrs = jnp.array([c["lr"] for c in cfg])

    def one(params, target, state, lr):
        g = jax.grad(loss_any)(params, target, kernel, jp, GAMMA, method, False, ks)
        upd, state = opt.update(g, state, params)
        params = jax.tree.map(lambda x, u: x - lr * u, params, upd)
        target = jax.tree.map(lambda t, x: (1 - tau) * t + tau * x, target, params)
        return params, target, state

    @jax.jit
    def train(carry):
        return jax.lax.scan(lambda c, _: (jax.vmap(one)(*c, lrs), None),
                            carry, None, length=steps)[0]

    final = train((params, params, state))[0]
    pred = np.asarray(jax.vmap(lambda x: bf.predict(x, _base_method(method), False))(final))
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError(f"nonfinite prediction: d={d} {method}")
    F_all, B_all = np.asarray(final["f"]), np.asarray(final["b"])
    return cfg, pred, F_all, B_all


def run_trained():
    rows = []
    t0 = time.time()
    for d in RANKS:
        for method in TRAIN_METHODS:
            cfg, pred, F_all, B_all = train_group(d, method, SEEDS, TRAIN_RATES)
            for i, c in enumerate(cfg):
                P_hat = bf.readout(pred[i], _base_method(method), GAMMA, N)  # normalized scale
                Q = P_hat @ R_GOAL / (1 - GAMMA)
                base = dict(d=d, seed=c["seed"], lr=c["lr"])
                extra = {}
                if method == "density_soft":
                    extra["retained_mass_vs_soft_target"] = retained_mass(P_hat, T_SOFT)
                rows.append(dict(arm=ARM_OF[method], method=method, **base, **gap_stats(Q),
                                 retained_mass=retained_mass(P_hat),
                                 l2_residual=float(np.sqrt(np.mean((P_hat - T_EXACT) ** 2))),
                                 max_mass_error=float(np.max(np.abs(P_hat.sum(1) - 1))), **extra))
                if method == "density":
                    rows.append(e_arm_rows(F_all[i], B_all[i],
                                           dict(base, retained_mass=retained_mass(P_hat))))
            g = [r for r in rows if r["d"] == d and r["method"] == method]
            log(f"  [trained] d={d} {method:13s} gap_ratio mean {np.mean([r['gap_ratio'] for r in g]):.3f}"
                f"  retained {np.mean([r['retained_mass'] for r in g]):.3f}"
                f"  argmax ok {sum(r['argmax_correct'] for r in g)}/{len(g)}  ({time.time() - t0:.0f}s)")
    return rows


# ----------------------------------------------------------------------------
# 5. Sanity checks and exact facts.
# ----------------------------------------------------------------------------
def run_sanity():
    log("=== sanity ===")
    err = float(np.max(np.abs(T_EXACT.sum(1) - 1)))
    log(f"rows of (1-gamma)M sum to 1: max err {err:.1e}"); assert err < 1e-10
    log(f"exact values at state {DECISION}: action0 (to goal) {Q_EXACT[ROW0]:.4f}, "
        f"action1 (back) {Q_EXACT[ROW1]:.4f}, gap {GAP_EXACT:.4f}")
    assert abs(Q_EXACT[ROW0] - 2.2) < 1e-9 and abs(Q_EXACT[ROW1] - 0.8) < 1e-9
    U, s, Vt = np.linalg.svd(T_EXACT)
    top2 = float((s[:2] ** 2).sum() / (s ** 2).sum())
    diff = T_EXACT[ROW0] - T_EXACT[ROW1]
    proj = Vt[:2].T @ (Vt[:2] @ diff)
    outside = float(1 - (proj @ proj) / (diff @ diff))
    log(f"top-2 right singular directions hold {top2:.3f} of the squared mass; "
        f"{outside:.4f} of the squared norm of the state-7 row difference lies outside them")
    F, B = svd_fit(N * M_EXACT, N)
    r_full = gap_stats(linear_readout(F, B))["gap_ratio"]
    log(f"A at full rank d={N}: gap ratio {r_full:.6f}"); assert abs(r_full - 1) < 1e-8
    _, _, P_hat, loss = _softmax_fit(T_EXACT, N, 999, "ce", CLOSED_STEPS, CLOSED_LR)
    r_d = gap_stats(P_hat @ R_GOAL / (1 - GAMMA))["gap_ratio"]
    log(f"D at full rank d={N}: gap ratio {r_d:.4f} (CE {loss:.5f})"); assert abs(r_d - 1) < 0.02
    soft_gap = (T_SOFT[ROW0] - T_SOFT[ROW1]) @ R_GOAL / (1 - GAMMA) / GAP_EXACT
    log(f"F's target (1-gamma) M K at full rank reads a gap ratio of {soft_gap:.4f} "
        f"(kernel sigma {SOFT_SIGMA} edges; self weight at the goal {K_SOFT[GOAL, GOAL]:.3f})")
    facts = dict(n=N, gamma=GAMMA, leak=LEAK, decision_state=DECISION, goal=GOAL,
                 q_exact_action0=float(Q_EXACT[ROW0]), q_exact_action1=float(Q_EXACT[ROW1]),
                 gap_exact=float(GAP_EXACT), singular_values=s.tolist(),
                 top2_retained_sq_mass=top2, diff_sq_frac_outside_top2=outside,
                 soft_target_full_rank_gap_ratio=float(soft_gap), soft_sigma=SOFT_SIGMA,
                 kernel_goal_self_weight=float(K_SOFT[GOAL, GOAL]),
                 closed_form_adam=dict(steps=CLOSED_STEPS, lr=CLOSED_LR, seeds=SEEDS),
                 trained=dict(steps=TRAIN_STEPS, tau=TRAIN_TAU, rates=TRAIN_RATES, seeds=SEEDS,
                              ortho_coef_fb_normsq=1.0, sphere=False))
    (HERE / "exact.json").write_text(json.dumps(facts, indent=2))
    log("sanity passed.\n")
    return facts


# ----------------------------------------------------------------------------
# 6. Aggregation and figure.
# ----------------------------------------------------------------------------
def t_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float); n = len(values); m = float(values.mean())
    if n < 2:
        return m, m, m
    from scipy import stats
    tcrit = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1))
    h = tcrit * float(values.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h


ARMS = ["A", "B", "C", "D", "E", "F"]
ARM_LABEL = {
    "A": "A  FB (squared error, linear readout)",
    "B": "B  FB, normalized target",
    "C": "C  softmax + squared error",
    "D": "D  Density-FB (softmax + cross-entropy)",
    "E": "E  Density-FB model, linear readout",
    "F": "F  Density-FB, soft immediate target",
}
# seaborn "colorblind" palette (Wong 2011); blue (slot 1) is kept for the exact reference
# as in the FourRooms / corridor figures, FB keeps orange, Density-FB keeps green.
ARM_COLOR = {"A": "#DE8F05", "B": "#CC78BC", "C": "#D55E00", "D": "#029E73",
             "E": "#56B4E9", "F": "#CA9161"}
ARM_MARKER = {"A": "o", "B": "s", "C": "^", "D": "D", "E": "v", "F": "P"}
EXACT_COLOR = "#0173B2"
INK = "#2a2a2a"


def agg(rows, arm, key="gap_ratio"):
    out = {}
    for d in RANKS:
        vals = [float(r[key]) for r in rows if r["arm"] == arm and int(r["d"]) == d]
        out[d] = t_interval(vals) if vals else (np.nan, np.nan, np.nan)
    return out


def _style_axis(ax, ylabel):
    ax.axhline(1.0, color=EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    ax.set_xticks(RANKS)
    ax.set_xlabel("representation rank", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=8, length=2.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9a9a9a")


def draw_closed_panel(ax, closed_rows, key="gap_ratio", ylabel="fraction of the exact action gap"):
    for arm in ARMS:
        a = agg(closed_rows, arm, key)
        y = [a[d][0] for d in RANKS]; lo = [a[d][1] for d in RANKS]; hi = [a[d][2] for d in RANKS]
        if arm in ("C", "D", "F"):        # E's raw scale is arbitrary, so its seed band is not shown
            ax.fill_between(RANKS, lo, hi, color=ARM_COLOR[arm], alpha=0.18, linewidth=0, zorder=2)
        # B coincides with A by construction: drawn underneath with a larger square so its
        # corners stay visible around A's circles.
        ax.plot(RANKS, y, color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                markersize=6.0 if arm == "B" else 4,
                markeredgecolor="white", markeredgewidth=0.6, linewidth=1.5,
                linestyle="-", zorder=3 + (arm == "A"))
    _style_axis(ax, ylabel)


def draw_trained_panel(ax, trained_rows, key="gap_ratio", ylabel="fraction of the exact action gap"):
    """Same line style as the closed-form panel, hollow markers and a light band for the
    spread over runs; points outside the axis are labelled at the edge."""
    for arm in ARMS:
        a = agg(trained_rows, arm, key)
        y = np.array([a[d][0] for d in RANKS])
        lo = np.array([a[d][1] for d in RANKS]); hi = np.array([a[d][2] for d in RANKS])
        if arm != "E":        # E's raw scale is arbitrary, so its band is not shown
            ax.fill_between(RANKS, lo, hi, color=ARM_COLOR[arm], alpha=0.16, linewidth=0, zorder=2)
        shown = np.where((y > GAP_YLIM[1]) | (y < GAP_YLIM[0]), np.nan, y) \
            if key == "gap_ratio" else y
        ax.plot(RANKS, shown, color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                markersize=6.0 if arm == "B" else 4.2, markerfacecolor="none",
                markeredgecolor=ARM_COLOR[arm], markeredgewidth=1.1, linewidth=1.5,
                linestyle="-", zorder=3 + (arm == "A"))
        if key == "gap_ratio":        # points off the axis are labelled at the edge instead
            for xi, yi in zip(RANKS, y):
                if yi > GAP_YLIM[1] or yi < GAP_YLIM[0]:
                    up = yi > GAP_YLIM[1]
                    ax.annotate(f"{yi:.2f}", (xi, GAP_YLIM[1] - 0.02 if up else GAP_YLIM[0] + 0.02),
                                fontsize=6.5, color=ARM_COLOR[arm], ha="center",
                                va="top" if up else "bottom")
    _style_axis(ax, ylabel)


GAP_YLIM = (-0.5, 1.15)


def legend_handles():
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=ARM_COLOR[a], marker=ARM_MARKER[a], markersize=4.5,
                 linewidth=1.5, label=ARM_LABEL[a]) for a in ARMS]
    hs.append(Line2D([0], [0], color=EXACT_COLOR, linestyle="--", linewidth=1.0, label="exact"))
    hs.append(Line2D([0], [0], color="#555555", marker="o", markersize=4.5, markerfacecolor="none",
                     linestyle="none", label="hollow markers: TD-trained; band: 95% CI"))
    return hs


def save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    log(f"wrote {FIG_DIR / stem}.pdf / .png")


def make_figures(closed_rows, trained_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK,
                         "axes.labelcolor": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})
    # main: gap vs rank, closed form (left) and TD-trained (right)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True)
    draw_closed_panel(axes[0], closed_rows)
    draw_trained_panel(axes[1], trained_rows)
    axes[1].set_ylabel("")
    axes[0].set_ylim(*GAP_YLIM)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, -0.30), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig_toy_ablation"); plt.close(fig)

    # per-panel files
    for stem, fn, rows in (("fig_toy_ablation_closed", draw_closed_panel, closed_rows),
                           ("fig_toy_ablation_trained", draw_trained_panel, trained_rows)):
        fig, ax = plt.subplots(figsize=(2.75, 2.35))
        fn(ax, rows); ax.set_ylim(*GAP_YLIM)
        fig.tight_layout(pad=0.3); save(fig, stem); plt.close(fig)

    # retained squared mass, closed form (lines) and trained (hollow markers), one panel
    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    draw_closed_panel(ax, closed_rows, key="retained_mass", ylabel="retained squared mass")
    draw_trained_panel(ax, trained_rows, key="retained_mass", ylabel="retained squared mass")
    ax.set_ylim(0.8, 1.02)
    fig.tight_layout(pad=0.3); save(fig, "fig_toy_ablation_mass"); plt.close(fig)

    # arm E with its scale calibrated (supplement)
    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    for rows, hollow in ((closed_rows, False), (trained_rows, True)):
        a = agg(rows, "E", "gap_ratio_calibrated")
        y = np.array([a[d][0] for d in RANKS])
        lo = y - np.array([a[d][1] for d in RANKS]); hi = np.array([a[d][2] for d in RANKS]) - y
        ax.errorbar(RANKS, y, yerr=[lo, hi], fmt=ARM_MARKER["E"], markersize=4.2,
                    markerfacecolor="none" if hollow else ARM_COLOR["E"],
                    markeredgecolor=ARM_COLOR["E"], ecolor=ARM_COLOR["E"], elinewidth=0.9,
                    capsize=1.8, linestyle="none" if hollow else "-", color=ARM_COLOR["E"],
                    linewidth=1.5)
    _style_axis(ax, "fraction of the exact gap, E scale-calibrated")
    fig.tight_layout(pad=0.3); save(fig, "fig_toy_ablation_E_calibrated"); plt.close(fig)

    # legend-only file
    fig = plt.figure(figsize=(5.5, 0.7))
    fig.legend(handles=legend_handles(), loc="center", ncol=2, fontsize=7.6, frameon=False,
               handlelength=1.8, columnspacing=1.2)
    save(fig, "fig_toy_ablation_legend"); plt.close(fig)


# ----------------------------------------------------------------------------
# 7. Summary tables (printed; the note copies them).
# ----------------------------------------------------------------------------
def print_tables(closed_rows, trained_rows):
    def line(arm, rows, key, fmt="{:.3f}"):
        a = agg(rows, arm, key)
        return " | ".join(fmt.format(a[d][0]) for d in RANKS)

    log("\n=== gap ratio at state 7 vs rank (closed form; C-F mean of 5 seeds) ===")
    log("arm | " + " | ".join(f"d={d}" for d in RANKS))
    for arm in ARMS:
        log(f"{arm} | {line(arm, closed_rows, 'gap_ratio')}")
    log("E calibrated | " + line("E", closed_rows, "gap_ratio_calibrated"))
    log("\n=== gap ratio at state 7 vs rank (TD-trained; mean of 5 seeds x 2 learning rates) ===")
    for arm in ARMS:
        log(f"{arm} | {line(arm, trained_rows, 'gap_ratio')}")
    log("E calibrated | " + line("E", trained_rows, "gap_ratio_calibrated"))
    for lr in TRAIN_RATES:
        log(f"\n--- TD-trained, lr={lr} only (5 seeds) ---")
        sub = [r for r in trained_rows if float(r["lr"]) == lr]
        for arm in ARMS:
            log(f"{arm} | {line(arm, sub, 'gap_ratio')}")
    log("\n=== retained squared mass vs rank (closed / trained) ===")
    for arm in ARMS:
        log(f"{arm} | {line(arm, closed_rows, 'retained_mass')}  ||  {line(arm, trained_rows, 'retained_mass')}")
    log("\n=== argmax stays on action 0 (fraction of fits) closed / trained ===")
    for arm in ARMS:
        c = [np.mean([float(r['argmax_correct'] in (True, 'True')) for r in closed_rows
                      if r['arm'] == arm and int(r['d']) == d]) for d in RANKS]
        t = [np.mean([float(r['argmax_correct'] in (True, 'True')) for r in trained_rows
                      if r['arm'] == arm and int(r['d']) == d]) for d in RANKS]
        log(f"{arm} | " + " | ".join(f"{x:.2f}" for x in c) + "  ||  " + " | ".join(f"{x:.2f}" for x in t))
    log("\n=== 95% CI of the TD-trained gap ratio (10 runs) ===")
    for arm in ARMS:
        a = agg(trained_rows, arm, "gap_ratio")
        log(f"{arm} | " + " | ".join(f"{a[d][0]:.2f} [{a[d][1]:.2f}, {a[d][2]:.2f}]" for d in RANKS))


def write_csv(rows, name):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    log(f"wrote {HERE / name} ({len(rows)} rows)")


def read_csv(name):
    with open(HERE / name) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    t0 = time.time()
    if "--figures-only" in sys.argv:
        closed_rows, trained_rows = read_csv("closed.csv"), read_csv("trained.csv")
    else:
        run_sanity()
        closed_rows = run_closed_form(); write_csv(closed_rows, "closed.csv")
        trained_rows = run_trained(); write_csv(trained_rows, "trained.csv")
    make_figures(closed_rows, trained_rows)
    print_tables(closed_rows, trained_rows)
    log(f"\ntotal wall time {time.time() - t0:.0f}s")
    if "--figures-only" not in sys.argv:      # keep the full run log from the real run
        (HERE / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n")
