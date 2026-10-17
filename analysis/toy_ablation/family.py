"""Toy ablation over a family of randomly generated nine-state bottleneck processes: gap tables and figures.

Run: JAX_PLATFORMS=cpu python analysis/toy_ablation/family.py [--figures-only]
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import csv
import json
import pathlib
import sys
import time
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "didactic"))
import bottleneck_family_td as bf  # noqa: E402

sys.path.insert(0, str(HERE))
import run as single  # noqa: E402

FIG_DIR = REPO / "figures" / "bottleneck_family"

GAMMA = 0.8
N = 9
RANKS = list(range(2, N + 1))
SEEDS = [0, 1, 2, 3, 4]
RATES = [0.001, 0.01]
# Reported family: five single-cut processes ("series"/"parallel" are also supported)
PROCESSES = [(0, "single"), (1, "single"), (2, "single"), (3, "single"), (4, "single")]
SOFT_SIGMA = 0.5
CLOSED_STEPS, CLOSED_LR = 10000, 1e-2
TRAIN_STEPS, TRAIN_TAU = 50000, 0.005
MIN_GAP = 0.5

LOG_LINES = []


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# Sampling a bottleneck process
def graph_distance(edges, n):
    adj = [set() for _ in range(n)]
    for s, (t0, t1) in enumerate(edges):
        for t in (int(t0), int(t1)):
            if t != s:
                adj[s].add(t)
                adj[t].add(s)
    dist = np.full((n, n), np.inf)
    for s in range(n):
        dist[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


def _wire_group(rng, edges, states, exit_to):
    """Wire a group: action 0 along a random order ending at `exit_to`, action 1 within the group."""
    order = [int(x) for x in rng.permutation(states)]
    for i, s in enumerate(order):
        edges[s, 0] = order[i + 1] if i + 1 < len(order) else exit_to
        alt = [t for t in states if t not in (s, edges[s, 0])] or [
            t for t in states if t != s
        ]
        edges[s, 1] = int(rng.choice(alt))
    return order


def _wire_last_group(rng, edges, states):
    """Wire the goal group; the state before the goal is the decision state."""
    order = [int(x) for x in rng.permutation(states)]
    goal, decision = order[-1], order[-2]
    for i, s in enumerate(order):
        edges[s, 0] = order[i + 1] if i + 1 < len(order) else int(rng.choice(states))
        alt = [t for t in states if t not in (s, edges[s, 0])] or [
            t for t in states if t != s
        ]
        edges[s, 1] = int(rng.choice(alt))
    edges[decision, 0] = goal
    edges[decision, 1] = int(
        rng.choice([t for t in states if t not in (decision, goal)])
    )
    return goal, decision


def _finish(seed, variant, n, gamma, edges, pi, groups, cuts, goal, decision, attempt):
    kernel = np.eye(n)[edges].reshape(-1, n)  # (2n, n), row 2s+a
    t_exact = bf.exact(kernel, pi, gamma)  # (1-gamma) M
    m_exact = t_exact / (1 - gamma)
    r = np.zeros(n)
    r[goal] = 1.0
    q = m_exact @ r
    gap = float(q[2 * decision] - q[2 * decision + 1])
    if gap < MIN_GAP:
        return None
    dist = graph_distance(edges, n)
    k_soft = np.exp(-(dist**2) / (2 * SOFT_SIGMA**2))
    k_soft = k_soft / k_soft.sum(axis=1, keepdims=True)
    return dict(
        seed=seed,
        variant=variant,
        n=n,
        edges=edges,
        pi=pi,
        cuts=cuts,
        groups=groups,
        goal=goal,
        decision=decision,
        kernel=kernel,
        t_exact=t_exact,
        m_exact=m_exact,
        r=r,
        q_exact=q,
        gap=gap,
        k_soft=k_soft,
        t_soft=t_exact @ k_soft,
        rho=np.full(n, 1.0 / n),
        attempts=attempt + 1,
    )


def sample_mdp(seed, variant="single", n=N, gamma=GAMMA):
    """Sample one bottleneck process ("single", "series" or "parallel")."""
    rng = np.random.default_rng(1000 + seed)
    for attempt in range(1000):
        edges = np.zeros((n, 2), int)

        if variant == "single":
            # [group A][cut][group B]
            n_a = int(rng.integers(3, 6))
            if n - 1 - n_a < 3:
                continue
            group_a = list(range(n_a))
            cut = n_a
            group_b = list(range(n_a + 1, n))
            _wire_group(rng, edges, group_a, cut)
            edges[cut, 0] = int(rng.choice(group_b))
            edges[cut, 1] = int(rng.choice(group_a))
            goal, decision = _wire_last_group(rng, edges, group_b)
            groups, cuts = [group_a, group_b], [cut]

        elif variant == "series":
            # [G0][cut 1][G1][cut 2][G2]
            n_0 = int(rng.integers(2, n - 2 - 3 - 2 + 1))
            n_1 = int(rng.integers(2, n - 2 - n_0 - 3 + 1))
            n_2 = n - 2 - n_0 - n_1
            if n_2 < 3:
                continue
            g0 = list(range(n_0))
            c1 = n_0
            g1 = list(range(n_0 + 1, n_0 + 1 + n_1))
            c2 = n_0 + 1 + n_1
            g2 = list(range(c2 + 1, n))
            _wire_group(rng, edges, g0, c1)
            edges[c1, 0] = int(rng.choice(g1))
            edges[c1, 1] = int(rng.choice(g0))
            _wire_group(rng, edges, g1, c2)
            edges[c2, 0] = int(rng.choice(g2))
            edges[c2, 1] = int(rng.choice(g1))
            goal, decision = _wire_last_group(rng, edges, g2)
            groups, cuts = [g0, g1, g2], [c1, c2]

        else:
            # [group A][cut 1][cut 2][group B], cuts in parallel
            n_a = int(rng.integers(2, n - 2 - 3 + 1))
            n_b = n - 2 - n_a
            if n_b < 3:
                continue
            group_a = list(range(n_a))
            c1, c2 = n_a, n_a + 1
            group_b = list(range(n_a + 2, n))
            _wire_group(rng, edges, group_a, c1)
            second = int(rng.choice(group_a))
            edges[second, 1] = c2
            for c in (c1, c2):
                edges[c, 0] = int(rng.choice(group_b))
                edges[c, 1] = int(rng.choice(group_a))
            goal, decision = _wire_last_group(rng, edges, group_b)
            groups, cuts = [group_a, group_b], [c1, c2]

        p0 = rng.uniform(0.25, 0.75, size=n)
        pi = np.stack([p0, 1.0 - p0], axis=1)
        mdp = _finish(
            seed, variant, n, gamma, edges, pi, groups, cuts, goal, decision, attempt
        )
        if mdp is not None:
            return mdp
    raise RuntimeError("no valid sample")


# Readouts and metrics
def gap_stats(mdp, Q):
    row0, row1 = 2 * mdp["decision"], 2 * mdp["decision"] + 1
    gap = float(Q[row0] - Q[row1])
    return dict(
        q0=float(Q[row0]),
        q1=float(Q[row1]),
        gap=gap,
        gap_ratio=gap / mdp["gap"],
        argmax_correct=bool(gap > 0),
    )


def retained_mass(mdp, P_hat, T=None):
    T = mdp["t_exact"] if T is None else T
    return float(1.0 - np.sum((P_hat - T) ** 2) / np.sum(T**2))


def linear_readout(mdp, F, B):
    z = B.T @ (mdp["rho"] * mdp["r"])
    return F @ z


def calibrate(mdp, v):
    vc = v - v.mean()
    qc = mdp["q_exact"] - mdp["q_exact"].mean()
    a = float(vc @ qc / max(vc @ vc, 1e-300))
    corr = float(vc @ qc / max(np.sqrt((vc @ vc) * (qc @ qc)), 1e-300))
    return a, corr


def e_arm_rows(mdp, F, B, base):
    v = linear_readout(mdp, F, B)
    a, corr = calibrate(mdp, v)
    raw, cal = gap_stats(mdp, v), gap_stats(mdp, a * v)
    return dict(
        base,
        arm="E",
        method="density_lin",
        q0=raw["q0"],
        q1=raw["q1"],
        gap=raw["gap"],
        gap_ratio=raw["gap_ratio"],
        argmax_correct=raw["argmax_correct"],
        gap_ratio_calibrated=cal["gap_ratio"],
        calib_scale=a,
        corr_with_exact=corr,
    )


# Closed-form fits
def svd_fit(X, d):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :d] * s[:d][None, :], Vt[:d].T


def softmax_fit(target, n, d, seed, loss_kind, steps=CLOSED_STEPS, lr=CLOSED_LR):
    rng = np.random.default_rng(seed)
    params = {
        "f": jnp.array(rng.normal(0, 0.1, (2 * n, d))),
        "b": jnp.array(rng.normal(0, 0.1, (n, d))),
    }
    tgt = jnp.array(target)

    def loss_fn(p):
        logits = p["f"] @ p["b"].T
        if loss_kind == "ce":
            return -jnp.mean(
                jnp.sum(tgt * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            )
        return 0.5 * jnp.mean(
            jnp.sum((jax.nn.softmax(logits, axis=-1) - tgt) ** 2, axis=-1)
        )

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
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return F, B, P, float(loss_fn(params))


def _ranks_for(mdp, ranks):
    """Ranks for a process: None (RANKS), a list, or a callable of the process."""
    if ranks is None:
        return RANKS
    return list(ranks(mdp)) if callable(ranks) else list(ranks)


def run_closed_form(mdps, ranks=None):
    rows = []
    t0 = time.time()
    for mdp in mdps:
        n, k = mdp["n"], mdp["seed"]
        for d in _ranks_for(mdp, ranks):
            F, B = svd_fit(n * mdp["m_exact"], d)
            Q = linear_readout(mdp, F, B)
            rows.append(
                dict(
                    arm="A",
                    method="fb",
                    mdp=k,
                    d=d,
                    seed=-1,
                    **gap_stats(mdp, Q),
                    retained_mass=retained_mass(mdp, (1 - GAMMA) * (F @ B.T) / n),
                )
            )
            F, B = svd_fit(mdp["t_exact"], d)
            Q = linear_readout(mdp, F, B) * n / (1 - GAMMA)
            rows.append(
                dict(
                    arm="B",
                    method="fb_norm",
                    mdp=k,
                    d=d,
                    seed=-1,
                    **gap_stats(mdp, Q),
                    retained_mass=retained_mass(mdp, F @ B.T),
                )
            )
            for sd in SEEDS:
                sseed = 1000 * sd + 17 * d + 101 * k
                F, B, P, _ = softmax_fit(mdp["t_exact"], n, d, sseed, "sq")
                rows.append(
                    dict(
                        arm="C",
                        method="softsq",
                        mdp=k,
                        d=d,
                        seed=sd,
                        **gap_stats(mdp, P @ mdp["r"] / (1 - GAMMA)),
                        retained_mass=retained_mass(mdp, P),
                    )
                )
                F, B, P, _ = softmax_fit(mdp["t_exact"], n, d, sseed, "ce")
                rows.append(
                    dict(
                        arm="D",
                        method="density",
                        mdp=k,
                        d=d,
                        seed=sd,
                        **gap_stats(mdp, P @ mdp["r"] / (1 - GAMMA)),
                        retained_mass=retained_mass(mdp, P),
                    )
                )
                rows.append(
                    e_arm_rows(
                        mdp,
                        F,
                        B,
                        dict(mdp=k, d=d, seed=sd, retained_mass=retained_mass(mdp, P)),
                    )
                )
                F, B, P, _ = softmax_fit(mdp["t_soft"], n, d, sseed, "ce")
                rows.append(
                    dict(
                        arm="F",
                        method="density_soft",
                        mdp=k,
                        d=d,
                        seed=sd,
                        **gap_stats(mdp, P @ mdp["r"] / (1 - GAMMA)),
                        retained_mass=retained_mass(mdp, P),
                    )
                )
        log(f"  [closed] mdp {k} done ({time.time() - t0:.0f}s)")
    return rows


# TD-trained fits
TRAIN_METHODS = ["fb", "normsq", "softsq", "density", "density_soft"]
ARM_OF = {"fb": "A", "normsq": "B", "softsq": "C", "density": "D", "density_soft": "F"}


def _base_method(method):
    return "density" if method == "density_soft" else method


def loss_any(params, target, kernel, pi, gamma, method, ksoft):
    if method != "density_soft":
        return bf.loss_fn(params, target, kernel, pi, gamma, method, False)
    n = kernel.shape[1]
    tp = jax.lax.stop_gradient(bf.predict(target, "density", False)).reshape(n, 2, n)
    v = jnp.sum(pi[:, :, None] * tp, axis=1)
    labels = (1 - gamma) * (kernel @ ksoft) + gamma * (kernel @ v)
    return -jnp.mean(
        jnp.sum(
            labels * jax.nn.log_softmax(params["f"] @ params["b"].T, axis=-1), axis=-1
        )
    )


def train_group(
    mdp, d, method, seeds=SEEDS, rates=RATES, steps=TRAIN_STEPS, tau=TRAIN_TAU
):
    n = mdp["n"]
    kernel, jp, ks = (
        jnp.array(mdp["kernel"]),
        jnp.array(mdp["pi"]),
        jnp.array(mdp["k_soft"]),
    )
    cfg = [{"seed": s, "lr": r} for s in seeds for r in rates]

    def init(c):
        rng = np.random.default_rng(c["seed"])
        return {
            "f": jnp.array(rng.normal(0, 0.1, (2 * n, d))),
            "b": jnp.array(rng.normal(0, 0.1, (n, d))),
        }

    params = jax.tree.map(lambda *x: jnp.stack(x), *[init(c) for c in cfg])
    opt = optax.scale_by_adam()
    state = jax.vmap(opt.init)(params)
    lrs = jnp.array([c["lr"] for c in cfg])

    def one(params, target, state, lr):
        g = jax.grad(loss_any)(params, target, kernel, jp, GAMMA, method, ks)
        upd, state = opt.update(g, state, params)
        params = jax.tree.map(lambda x, u: x - lr * u, params, upd)
        target = jax.tree.map(lambda t, x: (1 - tau) * t + tau * x, target, params)
        return params, target, state

    @jax.jit
    def train(carry):
        return jax.lax.scan(
            lambda c, _: (jax.vmap(one)(*c, lrs), None), carry, None, length=steps
        )[0]

    final = train((params, params, state))[0]
    pred = np.asarray(
        jax.vmap(lambda x: bf.predict(x, _base_method(method), False))(final)
    )
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError(
            f"nonfinite prediction: mdp={mdp['seed']} d={d} {method}"
        )
    return cfg, pred, np.asarray(final["f"]), np.asarray(final["b"])


def run_trained(mdps, ranks=None):
    rows = []
    t0 = time.time()
    for mdp in mdps:
        k, n = mdp["seed"], mdp["n"]
        for d in _ranks_for(mdp, ranks):
            for method in TRAIN_METHODS:
                cfg, pred, F_all, B_all = train_group(mdp, d, method)
                for i, c in enumerate(cfg):
                    P = bf.readout(pred[i], _base_method(method), GAMMA, n)
                    base = dict(mdp=k, d=d, seed=c["seed"], lr=c["lr"])
                    rows.append(
                        dict(
                            arm=ARM_OF[method],
                            method=method,
                            **base,
                            **gap_stats(mdp, P @ mdp["r"] / (1 - GAMMA)),
                            retained_mass=retained_mass(mdp, P),
                        )
                    )
                    if method == "density":
                        rows.append(
                            e_arm_rows(
                                mdp,
                                F_all[i],
                                B_all[i],
                                dict(base, retained_mass=retained_mass(mdp, P)),
                            )
                        )
            log(f"  [trained] mdp {k} d={d} done ({time.time() - t0:.0f}s)")
    return rows


# Figures
ARMS = single.ARMS
ARM_COLOR, ARM_MARKER, ARM_LABEL = single.ARM_COLOR, single.ARM_MARKER, single.ARM_LABEL
GAP_YLIM = single.GAP_YLIM


def agg(rows, arm, key="gap_ratio", ranks=RANKS):
    out = {}
    for d in ranks:
        vals = [
            float(r[key])
            for r in rows
            if r["arm"] == arm and int(r["d"]) == d and r.get(key, "") != ""
        ]
        out[d] = single.t_interval(vals) if vals else (np.nan, np.nan, np.nan)
    return out


def draw_panel(
    ax,
    rows,
    hollow,
    key="gap_ratio",
    ylabel="fraction of the exact action gap",
    ylim=GAP_YLIM,
    ranks=None,
):
    """One line per arm with a band over runs; hollow markers for TD-trained."""
    RANKS = list(ranks) if ranks is not None else globals()["RANKS"]
    for arm in ARMS:
        a = agg(rows, arm, key, ranks=RANKS)
        y = np.array([a[d][0] for d in RANKS])
        lo = np.array([a[d][1] for d in RANKS])
        hi = np.array([a[d][2] for d in RANKS])
        if arm != "E":  # E's scale is arbitrary
            ax.fill_between(
                RANKS, lo, hi, color=ARM_COLOR[arm], alpha=0.16, linewidth=0, zorder=2
            )
        shown = (
            np.where((y > ylim[1]) | (y < ylim[0]), np.nan, y)
            if ylim is not None
            else y
        )
        ax.plot(
            RANKS,
            shown,
            color=ARM_COLOR[arm],
            marker=ARM_MARKER[arm],
            markersize=6.0 if arm == "B" else 4.2,
            markerfacecolor="none" if hollow else ARM_COLOR[arm],
            markeredgecolor=ARM_COLOR[arm] if hollow else "white",
            markeredgewidth=1.1 if hollow else 0.6,
            linewidth=1.5,
            linestyle="-",
            zorder=3 + (arm == "A"),
        )
        if ylim is not None:  # label off-axis points at the edge
            for xi, yi in zip(RANKS, y):
                if yi > ylim[1] or yi < ylim[0]:
                    up = yi > ylim[1]
                    ax.annotate(
                        f"{yi:.2f}",
                        (xi, ylim[1] - 0.02 if up else ylim[0] + 0.02),
                        fontsize=6.5,
                        color=ARM_COLOR[arm],
                        ha="center",
                        va="top" if up else "bottom",
                    )
    single._style_axis(ax, ylabel)
    if ranks is not None:
        ax.set_xticks(RANKS)
    if ylim is not None:
        ax.set_ylim(*ylim)


VALUES_RANK = 3


def values_rows(trained_rows, mdps, d=VALUES_RANK, arms=("A", "D")):
    """Decision-state action values, rescaled so the exact pair is (1, 0); {arm: {action: (mean, lo, hi)}}."""
    qx = {
        m["seed"]: (
            m["q_exact"][2 * m["decision"]],
            m["q_exact"][2 * m["decision"] + 1],
        )
        for m in mdps
    }
    out = {}
    for arm in arms:
        cols = {0: [], 1: []}
        for r in trained_rows:
            if r["arm"] != arm or int(r["d"]) != d:
                continue
            q0e, q1e = qx[int(r["mdp"])]
            for a, key in ((0, "q0"), (1, "q1")):
                cols[a].append((float(r[key]) - q1e) / (q0e - q1e))
        out[arm] = {a: single.t_interval(v) for a, v in cols.items()}
    out["exact"] = {0: (1.0, 1.0, 1.0), 1: (0.0, 0.0, 0.0)}
    return out


def make_two_arm_figure(trained_rows, arms=("A", "D")):
    """FB vs Density-FB figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "text.color": single.INK,
            "axes.labelcolor": single.INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axhline(1.0, color=single.EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    for arm in arms:
        a = agg(trained_rows, arm)
        y = np.array([a[d][0] for d in RANKS])
        lo = np.array([a[d][1] for d in RANKS])
        hi = np.array([a[d][2] for d in RANKS])
        ax.fill_between(
            RANKS, lo, hi, color=ARM_COLOR[arm], alpha=0.16, linewidth=0, zorder=2
        )
        ax.plot(
            RANKS,
            y,
            color=ARM_COLOR[arm],
            marker=ARM_MARKER[arm],
            markersize=4.2,
            markerfacecolor="none",
            markeredgecolor=ARM_COLOR[arm],
            markeredgewidth=1.1,
            linewidth=1.6,
            zorder=3,
        )
    single._style_axis(ax, "fraction of the exact action gap")
    ax.set_ylim(-0.1, 1.15)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=ARM_COLOR["A"],
                marker=ARM_MARKER["A"],
                markerfacecolor="none",
                markersize=4.2,
                linewidth=1.6,
                label="FB",
            ),
            Line2D(
                [0],
                [0],
                color=ARM_COLOR["D"],
                marker=ARM_MARKER["D"],
                markerfacecolor="none",
                markersize=4.2,
                linewidth=1.6,
                label="\\textsc{Density-FB}" if False else "Density-FB",
            ),
            Line2D(
                [0],
                [0],
                color=single.EXACT_COLOR,
                linestyle="--",
                linewidth=1.0,
                label="exact gap",
            ),
        ],
        loc="lower right",
        fontsize=7.4,
        frameon=False,
        handlelength=1.6,
        labelspacing=0.3,
        borderpad=0.2,
    )
    fig.tight_layout(pad=0.3)
    save(fig, "fig_family_gap_two_arm")
    plt.close(fig)


def make_values_figure(
    trained_rows,
    mdps,
    d=VALUES_RANK,
    stem="fig_family_values",
    fig_dir=None,
    action_labels=("action into the goal", "action away"),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "text.color": single.INK,
            "axes.labelcolor": single.INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    vals = values_rows(trained_rows, mdps, d=d)
    groups = [
        ("exact", "exact", single.EXACT_COLOR),
        ("A", "FB", ARM_COLOR["A"]),
        ("D", "Density-FB", ARM_COLOR["D"]),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    width, gap = 0.34, 0.12
    xt, xl = [], []
    for i, (key, name, color) in enumerate(groups):
        base = i * (2 * width + gap + 0.3)
        for a in (0, 1):
            m, lo, hi = vals[key][a]
            x = base + a * (width + 0.03)
            ax.bar(
                x,
                m,
                width=width,
                color=color,
                alpha=1.0 if a == 0 else 0.42,
                edgecolor=color,
                linewidth=0.8,
                zorder=2,
            )
            if hi > lo:
                ax.errorbar(
                    x,
                    m,
                    yerr=[[m - lo], [hi - m]],
                    ecolor=single.INK,
                    elinewidth=0.9,
                    capsize=2.0,
                    linestyle="none",
                    zorder=3,
                )
            ax.annotate(
                f"{m:.2f}",
                (x, max(m, hi)),
                textcoords="offset points",
                xytext=(0, 2.5),
                ha="center",
                fontsize=6.5,
                color=single.INK,
            )
        xt.append(base + width / 2 + 0.015)
        xl.append(name)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8.5)
    ax.set_ylabel(
        "value at the decision state\n(exact pair mapped to 1 and 0)", fontsize=8
    )
    ax.tick_params(axis="y", labelsize=8, length=2.5)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9a9a9a")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#6a6a6a", label=action_labels[0]),
            Patch(facecolor="#6a6a6a", alpha=0.42, label=action_labels[1]),
        ],
        fontsize=7.2,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.22),
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.3)
    save(fig, stem, fig_dir)
    plt.close(fig)


def save(fig, stem, fig_dir=None):
    fig_dir = FIG_DIR if fig_dir is None else pathlib.Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    log(f"wrote {fig_dir / stem}.pdf / .png")


def make_figures(closed_rows, trained_rows, mdps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "text.color": single.INK,
            "axes.labelcolor": single.INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True)
    draw_panel(axes[0], closed_rows, hollow=False)
    draw_panel(axes[1], trained_rows, hollow=True)
    axes[1].set_ylabel("")
    fig.legend(
        handles=single.legend_handles(),
        loc="lower center",
        ncol=2,
        fontsize=7.6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.30),
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig_family_ablation")
    plt.close(fig)

    # Standalone panels with their own legend
    for stem, rows, hollow in (
        ("fig_family_ablation_closed", closed_rows, False),
        ("fig_family_ablation_trained", trained_rows, True),
    ):
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        draw_panel(ax, rows, hollow=hollow)
        fig.legend(
            handles=single.legend_handles()[:-1],
            loc="lower center",
            ncol=2,
            fontsize=6.4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.20),
            handlelength=1.6,
            columnspacing=1.0,
            labelspacing=0.3,
        )
        fig.tight_layout(pad=0.3)
        save(fig, stem)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    draw_panel(
        ax,
        closed_rows,
        hollow=False,
        key="retained_mass",
        ylabel="retained squared mass",
        ylim=(0.8, 1.02),
    )
    draw_panel(
        ax,
        trained_rows,
        hollow=True,
        key="retained_mass",
        ylabel="retained squared mass",
        ylim=(0.8, 1.02),
    )
    fig.tight_layout(pad=0.3)
    save(fig, "fig_family_mass")
    plt.close(fig)

    # Per-process trained panels
    ncol = 5
    nrow = int(np.ceil(len(mdps) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.2 * ncol, 2.35 * nrow), sharey=True, squeeze=False
    )
    for ax, mdp in zip(axes.ravel(), mdps):
        sub = [r for r in trained_rows if int(r["mdp"]) == mdp["seed"]]
        draw_panel(ax, sub, hollow=True)
        ax.set_title(label(mdp), fontsize=8)
        ax.set_ylabel("")
    for ax in axes.ravel()[len(mdps) :]:
        ax.axis("off")
    for r in range(nrow):
        axes[r, 0].set_ylabel("fraction of the exact action gap", fontsize=8.5)
    fig.legend(
        handles=single.legend_handles(),
        loc="lower center",
        ncol=4,
        fontsize=7.6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12 / nrow),
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=1.2)
    save(fig, "fig_family_per_mdp")
    plt.close(fig)

    make_values_figure(trained_rows, mdps)
    make_two_arm_figure(trained_rows)
    for mdp in mdps:
        draw_graph(mdp)


def graph_layout(mdp):
    """Node positions: groups on small circles, cuts between them."""
    n, cuts, groups = mdp["n"], mdp["cuts"], mdp["groups"]
    pos, centers = {}, {}
    if mdp["variant"] == "series":
        col_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
        blocks = [(groups[0], col_x[0]), (groups[1], col_x[2]), (groups[2], col_x[4])]
        cut_pos = {
            cuts[0]: np.array([col_x[1], 0.0]),
            cuts[1]: np.array([col_x[3], 0.0]),
        }
        radius = 0.42
    elif mdp["variant"] == "parallel":
        blocks = [(groups[0], -1.35), (groups[1], 1.35)]
        cut_pos = {cuts[0]: np.array([0.0, 0.5]), cuts[1]: np.array([0.0, -0.5])}
        radius = 0.62
    else:
        blocks = [(groups[0], -1.35), (groups[1], 1.35)]
        cut_pos = {cuts[0]: np.array([0.0, 0.0])}
        radius = 0.62
    for states, x in blocks:
        for i, st in enumerate(states):
            ang = np.pi / 2 - 2 * np.pi * i / len(states)
            pos[st] = np.array([x + radius * np.cos(ang), radius * np.sin(ang)])
            centers[st] = np.array([x, 0.0])
    for c, p in cut_pos.items():
        pos[c] = p
        centers[c] = p + np.array([0.0, -0.9 if p[1] <= 0 else 0.9])
    out = {
        st: (pos[st] - centers[st]) / max(np.linalg.norm(pos[st] - centers[st]), 1e-9)
        for st in range(n)
    }
    return pos, out


def draw_graph(
    mdp,
    layout=None,
    node_style=None,
    stem=None,
    title=None,
    fig_dir=None,
    figsize=None,
    xlim=None,
):
    """Draw a process graph; layout and node_style default to the family's own."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    n, goal, dec, cuts = mdp["n"], mdp["goal"], mdp["decision"], list(mdp["cuts"])
    pos, out = (layout or graph_layout)(mdp)
    wide = mdp["variant"] == "series"
    node_r = 0.15 if wide else 0.17

    fig, ax = plt.subplots(figsize=figsize or ((4.6, 2.0) if wide else (3.2, 2.0)))
    for s in range(n):
        for a in (0, 1):
            t = int(mdp["edges"][s, a])
            color = single.INK if a == 0 else "#9a9a9a"
            style = "-" if a == 0 else (0, (3, 2))
            if t == s:
                ax.add_patch(
                    Circle(
                        pos[s] + 0.24 * out[s],
                        0.11,
                        fill=False,
                        edgecolor=color,
                        linewidth=0.8,
                        linestyle=style,
                        zorder=1,
                    )
                )
                continue
            ax.add_patch(
                FancyArrowPatch(
                    pos[s],
                    pos[t],
                    connectionstyle="arc3,rad=0.18",
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=0.8,
                    linestyle=style,
                    color=color,
                    shrinkA=8,
                    shrinkB=8,
                    zorder=1,
                    alpha=0.9,
                )
            )
    for s in range(n):
        if node_style is not None:
            face, edge, txt = node_style(mdp, s)
        elif s == goal:
            face, edge, txt = "#029E73", "#029E73", "white"
        elif s == dec:
            face, edge, txt = "white", "#029E73", single.INK
        elif s in cuts:
            face, edge, txt = "#DE8F05", "#DE8F05", "white"
        else:
            face, edge, txt = "white", "#9a9a9a", single.INK
        ax.add_patch(
            Circle(
                pos[s], node_r, facecolor=face, edgecolor=edge, linewidth=1.2, zorder=2
            )
        )
        ax.text(
            *pos[s], str(s), ha="center", va="center", fontsize=6.5, color=txt, zorder=3
        )
        loops = int(mdp["edges"][s, 0]) == s or int(mdp["edges"][s, 1]) == s
        lab = pos[s] + (0.48 if loops else 0.25) * out[s]
        ax.text(
            lab[0],
            lab[1],
            f"{mdp['pi'][s, 0]:.2f}",
            ha="center",
            va="center",
            fontsize=5.5,
            color="#6a6a6a",
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.08",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )
    ax.set_xlim(*(xlim or ((-2.85, 2.85) if wide else (-2.35, 2.35))))
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is None:
        title = (
            f"{label(mdp)}: cut{'s' if len(cuts) > 1 else ''} "
            f"{', '.join(str(c) for c in cuts)}, decision {dec}, goal {goal}, "
            f"exact gap {mdp['gap']:.2f}"
        )
    ax.set_title(title, fontsize=7, color=single.INK)
    fig.tight_layout(pad=0.2)
    save(fig, stem or f"fig_family_graph_{mdp['seed'] + 1}", fig_dir)
    plt.close(fig)


# Tables and IO
def print_tables(closed_rows, trained_rows, mdps):
    def line(rows, arm, key="gap_ratio"):
        a = agg(rows, arm, key)
        return " | ".join(f"{a[d][0]:.3f}" for d in RANKS)

    log("\n=== gap ratio vs rank, mean over all processes (closed form) ===")
    log("arm | " + " | ".join(f"d={d}" for d in RANKS))
    for arm in ARMS:
        log(f"{arm} | {line(closed_rows, arm)}")
    log("\n=== gap ratio vs rank, mean over all processes (TD-trained) ===")
    for arm in ARMS:
        a = agg(trained_rows, arm)
        log(
            f"{arm} | "
            + " | ".join(f"{a[d][0]:.2f} [{a[d][1]:.2f}, {a[d][2]:.2f}]" for d in RANKS)
        )
    for variant in ("single", "series", "parallel"):
        keep = {m["seed"] for m in mdps if m["variant"] == variant}
        sub = [r for r in trained_rows if int(r["mdp"]) in keep]
        if not sub:
            continue
        log(
            f"\n=== gap ratio vs rank, TD-trained, {variant} bottleneck only "
            f"({len(keep)} processes) ==="
        )
        for arm in ARMS:
            log(f"{arm} | {line(sub, arm)}")

    log("\n=== per process, TD-trained gap ratio ===")
    for mdp in mdps:
        sub = [r for r in trained_rows if int(r["mdp"]) == mdp["seed"]]
        log(
            f"-- {label(mdp)} (gap {mdp['gap']:.2f}, "
            f"groups {'/'.join(str(len(g)) for g in mdp['groups'])}, "
            f"cuts {mdp['cuts']}) --"
        )
        for arm in ARMS:
            log(f"   {arm} | {line(sub, arm)}")

    log(
        "\n=== smallest rank reaching 90% of the exact gap (per process, TD-trained) ==="
    )
    log("arm | " + " | ".join(label(m).replace(" ", "") for m in mdps) + " | median")
    for arm in ARMS:
        need = []
        for mdp in mdps:
            sub = [r for r in trained_rows if int(r["mdp"]) == mdp["seed"]]
            a = agg(sub, arm)
            hit = [d for d in RANKS if a[d][0] >= 0.9]
            need.append(min(hit) if hit else np.nan)
        med = np.nanmedian(need) if not all(np.isnan(x) for x in need) else np.nan
        log(
            f"{arm} | "
            + " | ".join("-" if np.isnan(x) else str(int(x)) for x in need)
            + f" | {med if np.isnan(med) else med:.1f}"
        )

    log(
        "\n=== retained squared mass vs rank, mean over all processes (closed || trained) ==="
    )
    for arm in ARMS:
        log(
            f"{arm} | {line(closed_rows, arm, 'retained_mass')}  ||  "
            f"{line(trained_rows, arm, 'retained_mass')}"
        )
    log("\n=== argmax stays on the goal action (fraction of fits, TD-trained) ===")
    for arm in ARMS:
        vals = [
            np.mean(
                [
                    float(str(r["argmax_correct"]) == "True")
                    for r in trained_rows
                    if r["arm"] == arm and int(r["d"]) == d
                ]
            )
            for d in RANKS
        ]
        log(f"{arm} | " + " | ".join(f"{v:.2f}" for v in vals))


def write_csv(rows, name):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {HERE / name} ({len(rows)} rows)")


def read_csv(name):
    with open(HERE / name) as f:
        return list(csv.DictReader(f))


def label(mdp):
    return f"{mdp['variant']} ({mdp['seed'] + 1})"


def check(mdp):
    """Assert the generator's structural guarantees."""
    edges, n = mdp["edges"], mdp["n"]
    assert float(np.max(np.abs(mdp["t_exact"].sum(1) - 1))) < 1e-10
    last = mdp["groups"][-1]
    for s in last:  # goal group is closed
        assert all(int(t) in last for t in edges[s]), "last group leaks back"
    into_last = [
        (s, a)
        for s in range(n)
        for a in (0, 1)
        if s not in last and int(edges[s, a]) in last
    ]
    cuts = mdp["cuts"]
    want = (
        [(cuts[0], 0), (cuts[1], 0)]
        if mdp["variant"] == "parallel"
        else [(cuts[-1], 0)]
    )
    assert sorted(into_last) == sorted(want), (into_last, want)
    if mdp["variant"] == "series":  # middle group has one entry
        mid = mdp["groups"][1] + [cuts[1]]
        into_mid = [
            (s, a)
            for s in range(n)
            for a in (0, 1)
            if s not in mid and int(edges[s, a]) in mid
        ]
        assert into_mid == [(cuts[0], 0)], into_mid
    if mdp["variant"] == "parallel":  # both cuts reachable from A
        for c in cuts:
            assert any(
                int(edges[s, a]) == c for s in mdp["groups"][0] for a in (0, 1)
            ), c


def describe(mdps):
    log("=== sampled processes ===")
    out = []
    for mdp in mdps:
        check(mdp)
        log(
            f"{label(mdp):14s} groups {'/'.join(str(len(g)) for g in mdp['groups'])}, "
            f"cuts {mdp['cuts']}, decision {mdp['decision']}, goal {mdp['goal']}, "
            f"exact values {mdp['q_exact'][2 * mdp['decision']]:.2f} / "
            f"{mdp['q_exact'][2 * mdp['decision'] + 1]:.2f}, gap {mdp['gap']:.3f}, "
            f"p(action 0) in [{mdp['pi'][:, 0].min():.2f}, {mdp['pi'][:, 0].max():.2f}], "
            f"{mdp['attempts']} draw(s)"
        )
        out.append(
            dict(
                seed=int(mdp["seed"]),
                variant=mdp["variant"],
                n=int(mdp["n"]),
                edges=mdp["edges"].tolist(),
                pi=mdp["pi"].tolist(),
                cuts=[int(c) for c in mdp["cuts"]],
                goal=int(mdp["goal"]),
                decision=int(mdp["decision"]),
                groups=[[int(x) for x in g] for g in mdp["groups"]],
                gap=mdp["gap"],
                q_exact=mdp["q_exact"].tolist(),
                attempts=mdp["attempts"],
            )
        )
    (HERE / "family_mdps.json").write_text(
        json.dumps(
            dict(
                gamma=GAMMA,
                ranks=RANKS,
                seeds=SEEDS,
                rates=RATES,
                steps=TRAIN_STEPS,
                tau=TRAIN_TAU,
                closed_steps=CLOSED_STEPS,
                closed_lr=CLOSED_LR,
                soft_sigma=SOFT_SIGMA,
                min_gap=MIN_GAP,
                processes=out,
            ),
            indent=2,
        )
    )
    log("")


if __name__ == "__main__":
    t0 = time.time()
    mdps = [sample_mdp(k, v) for k, v in PROCESSES]
    describe(mdps)
    if "--figures-only" in sys.argv:
        keep = {k for k, _ in PROCESSES}
        closed_rows = [
            r for r in read_csv("family_closed.csv") if int(r["mdp"]) in keep
        ]
        trained_rows = [
            r for r in read_csv("family_trained.csv") if int(r["mdp"]) in keep
        ]
    else:
        closed_rows = run_closed_form(mdps)
        write_csv(closed_rows, "family_closed.csv")
        trained_rows = run_trained(mdps)
        write_csv(trained_rows, "family_trained.csv")
    make_figures(closed_rows, trained_rows, mdps)
    print_tables(closed_rows, trained_rows, mdps)
    log(f"\ntotal wall time {time.time() - t0:.0f}s")
    if "--figures-only" not in sys.argv:
        (HERE / "family_log.txt").write_text("\n".join(LOG_LINES) + "\n")
