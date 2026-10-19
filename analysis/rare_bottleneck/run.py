"""Rare-bottleneck experiment: action-gap recovery of FB, softmax and Density-FB fits as the decision state's rows get rare.

Run: JAX_PLATFORMS=cpu python analysis/rare_bottleneck/run.py [--figures-only | --closed-only | --smoke]
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import csv
import functools
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "didactic"))
import bottleneck_family_td as bf  # noqa: E402
import graph_bottleneck_td as gb  # noqa: E402

FIG_DIR = REPO / "figures" / "rare_bottleneck"
LOG_LINES = []
SMOKE = "--smoke" in sys.argv


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


# MDPs
GAMMA = 0.8
GOAL, DECISION, CUT = gb.GOAL, gb.DECISION, gb.CUT
ROW0, ROW1 = 2 * DECISION, 2 * DECISION + 1  # the two decision-state rows
RANKS = [2, 3, 4, 6]
SEEDS = [0, 1, 2, 3, 4]
W_SWEEP = [1 / 2, 1 / 4, 1 / 8, 1 / 9, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 0.0]
N_SWEEP = [0, 4, 8, 16, 32, 64]
if SMOKE:
    RANKS, SEEDS = [3], [0, 1]
    W_SWEEP, N_SWEEP = [1 / 2, 1 / 9, 1 / 64, 0.0], [0, 8]


def build_mdp(N):
    """Nine-state process plus N/4 copies of group A, each exiting into state 4."""
    assert N % 4 == 0
    edges = gb.EDGES.tolist()
    for k in range(N // 4):
        b = 9 + 4 * k
        edges += [[b + 1, b + 2], [b + 3, b + 0], [b + 3, b + 1], [CUT, b + 2]]
    edges = np.array(edges)
    n = len(edges)
    kernel = np.eye(n)[edges].reshape(-1, n)  # (2n, n), row 2s+a
    pi = np.full((n, 2), 0.5)
    T = bf.exact(kernel, pi, GAMMA)  # (1-gamma) M
    M = T / (1.0 - GAMMA)
    r = np.zeros(n)
    r[GOAL] = 1.0
    Q = M @ r
    return dict(
        N=N,
        n=n,
        edges=edges,
        kernel=kernel,
        pi=pi,
        T=T,
        M=M,
        X=n * M,
        r=r,
        Q=Q,
        gap=float(Q[ROW0] - Q[ROW1]),
    )


def _rows(mdp):
    """The two decision-state rows."""
    return tuple(int(x) for x in mdp.get("rows", (ROW0, ROW1)))


def _goal(mdp):
    return int(mdp.get("goal", GOAL))


def row_weights(mdp, w):
    """Total weight w on the two state-7 rows, 1-w shared by the other rows."""
    m = 2 * mdp["n"]
    wts = np.full(m, (1.0 - w) / (m - 2))
    wts[list(_rows(mdp))] = w / 2.0
    return wts


def ordinary_rows(mdp):
    return np.array([i for i in range(2 * mdp["n"]) if i not in _rows(mdp)])


# Metrics
def gap_stats(mdp, Q):
    ROW0, ROW1 = _rows(mdp)
    gap = float(Q[ROW0] - Q[ROW1])
    return dict(
        q0=float(Q[ROW0]),
        q1=float(Q[ROW1]),
        gap=gap,
        gap_ratio=gap / mdp["gap"],
        argmax_correct=bool(gap > 0),
    )


def retained(mdp, P_hat, wts=None):
    """1 - ||P_hat - T||^2 / ||T||^2, optionally row-weighted."""
    T = mdp["T"]
    if wts is None:
        return float(1.0 - np.sum((P_hat - T) ** 2) / np.sum(T**2))
    return float(
        1.0 - np.sum(wts[:, None] * (P_hat - T) ** 2) / np.sum(wts[:, None] * T**2)
    )


def share(wts, row_loss, rows):
    tot = float(np.sum(wts * row_loss))
    return float(np.sum(wts[rows] * row_loss[rows]) / tot) if tot > 0 else float("nan")


# Variant A: FB, closed form
def fb_objective_rows(mdp, fit):
    """Per-row FB loss: (1/n) sum_j (fit_ij - n M_ij)^2."""
    return np.sum((fit - mdp["X"]) ** 2, axis=1) / mdp["n"]


def fb_fit_from_B(mdp, B):
    """Per-row least squares given the span of B (orthonormal columns): fit = X P_B."""
    F = mdp["X"] @ B
    return F, F @ B.T


def fb_weighted(mdp, wts, d):
    """Exact minimizer of sum_i w_i ||F_i B^T - X_i||^2 at rank d (weighted SVD)."""
    S = np.sqrt(wts)[:, None]
    _, s, Vt = np.linalg.svd(S * mdp["X"], full_matrices=False)
    B = Vt[:d].T
    F, fit = fb_fit_from_B(mdp, B)
    return F, B, fit, s


def fb_constrained(mdp, wts, d):
    """Same objective, span of B forced to contain the decision-row difference."""
    X = mdp["X"]
    ROW0, ROW1 = _rows(mdp)
    dv = X[ROW0] - X[ROW1]
    dh = dv / np.linalg.norm(dv)
    Xp = X - np.outer(X @ dh, dh)
    S = np.sqrt(wts)[:, None]
    _, _, Vt = np.linalg.svd(S * Xp, full_matrices=False)
    Bp = Vt[: d - 1].T
    Bp = Bp - np.outer(dh, dh @ Bp)
    Bp, _ = np.linalg.qr(Bp)
    B = np.concatenate([dh[:, None], Bp], axis=1)
    assert np.allclose(B.T @ B, np.eye(d), atol=1e-8)
    F, fit = fb_fit_from_B(mdp, B)
    return F, B, fit


def fb_limit(mdp, d):
    """Limit w -> 0: project onto the top-d right singular directions of the other rows."""
    O = ordinary_rows(mdp)
    X = mdp["X"]
    G = X[O].T @ X[O]
    lam, V = np.linalg.eigh(G)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    P = V[:, :d] @ V[:, :d].T
    ROW0, ROW1 = _rows(mdp)
    dv = X[ROW0] - X[ROW1]
    frac = float((dv @ P @ mdp["r"]) / (dv @ mdp["r"]))
    in_span = float((dv @ P @ dv) / (dv @ dv))
    gap_rel = float((lam[d - 1] - lam[d]) / lam[0]) if d < len(lam) else float("nan")
    return frac, in_span, gap_rel


def run_variant_a(mdp, wts, d, construction, w):
    X, n, r = mdp["X"], mdp["n"], mdp["r"]
    F, B, fit, s = fb_weighted(mdp, wts, d)
    Q = fit @ r / n  # F z with z = (1/n) B^T r
    rl = fb_objective_rows(mdp, fit)
    P_hat = (1 - GAMMA) * fit / n
    Fc, Bc, fitc = fb_constrained(mdp, wts, d)
    rlc = fb_objective_rows(mdp, fitc)
    Qc = fitc @ r / n
    J, Jc = float(np.sum(wts * rl)), float(np.sum(wts * rlc))
    O = ordinary_rows(mdp)
    bn = np.array(_rows(mdp))
    lim, in_span, spec_gap = fb_limit(mdp, d)
    tgt_rows = np.sum(X**2, axis=1) / n
    rows = [
        dict(
            construction=construction,
            w=w,
            N=mdp["N"],
            n=n,
            d=d,
            arm="A",
            method="fb",
            seed=-1,
            **gap_stats(mdp, Q),
            retained_mass=retained(mdp, P_hat),
            retained_mass_w=retained(mdp, P_hat, wts),
            own_loss=J,
            bn_share_obj=share(wts, rl, bn),
            bn_share_target=share(wts, tgt_rows, bn),
            bn_loss=float(np.sum(wts[bn] * rl[bn])),
            cost_preserve=Jc - J,
            cost_preserve_rel=(Jc - J) / J if J > 0 else float("nan"),
            cost_preserve_ord=float(np.sum(wts[O] * (rlc[O] - rl[O]))),
            cost_preserve_bn=float(np.sum(wts[bn] * (rlc[bn] - rl[bn]))),
            limit_gap_ratio=lim,
            limit_diff_in_span=in_span,
            limit_spectral_gap=spec_gap,
        ),
        dict(
            construction=construction,
            w=w,
            N=mdp["N"],
            n=n,
            d=d,
            arm="A+",
            method="fb_keepgap",
            seed=-1,
            **gap_stats(mdp, Qc),
            retained_mass=retained(mdp, (1 - GAMMA) * fitc / n),
            retained_mass_w=retained(mdp, (1 - GAMMA) * fitc / n, wts),
            own_loss=Jc,
            bn_share_obj=share(wts, rlc, bn),
            bn_loss=float(np.sum(wts[bn] * rlc[bn])),
        ),
    ]
    return rows


# Variants C and D: softmax fits
CLOSED_STEPS = 300 if SMOKE else 50000
POLISH_STEPS = 200 if SMOKE else 10000
CLOSED_LR = 1e-2


def _row_losses(f, b, target, loss_kind):
    logits = f @ b.T
    if loss_kind == "ce":
        return -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    return 0.5 * jnp.sum((jax.nn.softmax(logits, axis=-1) - target) ** 2, axis=-1)


@functools.lru_cache(maxsize=None)
def _trainer(loss_kind, steps, polish_steps, lr):
    def loss_joint(p, target, wts):
        return jnp.sum(wts * _row_losses(p["f"], p["b"], target, loss_kind))

    def loss_polish(f, b, target):
        return jnp.mean(_row_losses(f, b, target, loss_kind))

    def fit_one(params, target, wts):
        opt = optax.adam(lr)

        def body(c, _):
            p, st = c
            g = jax.grad(loss_joint)(p, target, wts)
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), None

        (params, _), _ = jax.lax.scan(
            body, (params, opt.init(params)), None, length=steps
        )
        f, b = params["f"], params["b"]
        opt2 = optax.adam(lr)

        def body2(c, _):
            f, st = c
            g = jax.grad(loss_polish)(f, b, target)
            u, st = opt2.update(g, st, f)
            return (optax.apply_updates(f, u), st), None

        (f, _), _ = jax.lax.scan(body2, (f, opt2.init(f)), None, length=polish_steps)
        return {"f": f, "b": b}

    return jax.jit(jax.vmap(fit_one, in_axes=(0, None, None)))


def softmax_fits(mdp, wts, d, loss_kind, seeds):
    n = mdp["n"]
    inits = []
    for sd in seeds:
        rng = np.random.default_rng(1000 * sd + 17 * d)
        inits.append(
            {
                "f": jnp.array(rng.normal(0, 0.1, (2 * n, d))),
                "b": jnp.array(rng.normal(0, 0.1, (n, d))),
            }
        )
    params = jax.tree.map(lambda *x: jnp.stack(x), *inits)
    out = _trainer(loss_kind, CLOSED_STEPS, POLISH_STEPS, CLOSED_LR)(
        params, jnp.array(mdp["T"]), jnp.array(wts)
    )
    return np.asarray(out["f"]), np.asarray(out["b"])


def run_variants_cd(mdp, wts, d, construction, w):
    rows = []
    T, r, n = mdp["T"], mdp["r"], mdp["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.sum(np.where(T > 0, T * np.log(np.where(T > 0, T, 1.0)), 0.0), axis=1)
    bn = np.array(_rows(mdp))
    ROW0, ROW1 = _rows(mdp)
    GOAL = _goal(mdp)
    for variant, method, kind in (("C", "softsq", "sq"), ("D", "density", "ce")):
        F_all, B_all = softmax_fits(mdp, wts, d, kind, SEEDS)
        for i, sd in enumerate(SEEDS):
            logits = F_all[i] @ B_all[i].T
            P_hat = np.exp(logits - logits.max(axis=1, keepdims=True))
            P_hat /= P_hat.sum(axis=1, keepdims=True)
            Q = P_hat @ r / (1 - GAMMA)
            if kind == "ce":
                rl = (
                    -np.sum(T * np.log(np.maximum(P_hat, 1e-300)), axis=1) - H
                )  # KL(t || p) per row
            else:
                rl = 0.5 * np.sum((P_hat - T) ** 2, axis=1)
            rows.append(
                dict(
                    construction=construction,
                    w=w,
                    N=mdp["N"],
                    n=n,
                    d=d,
                    arm=variant,
                    method=method,
                    seed=sd,
                    **gap_stats(mdp, Q),
                    retained_mass=retained(mdp, P_hat),
                    retained_mass_w=retained(mdp, P_hat, wts),
                    own_loss=float(np.sum(wts * rl)),
                    bn_share_obj=share(wts, rl, bn),
                    bn_loss=float(np.sum(wts[bn] * rl[bn])),
                    bn_goal_mass_0=float(P_hat[ROW0, GOAL]),
                    bn_goal_mass_1=float(P_hat[ROW1, GOAL]),
                    bn_goal_mass_0_true=float(T[ROW0, GOAL]),
                    bn_goal_mass_1_true=float(T[ROW1, GOAL]),
                )
            )
    return rows


# Trained TD harness with weighted rows
TRAIN_STEPS = 500 if SMOKE else 50000
TRAIN_TAU = 0.005
TRAIN_RATES = [0.001, 0.01]
TRAIN_METHODS = ["fb", "softsq", "density"]
VARIANT_OF = {"fb": "A", "softsq": "C", "density": "D"}
TRAIN_RANK = 3


def td_loss(params, target, kernel, pi, gamma, method, wts):
    """TD loss with the mean over rows replaced by a w-weighted sum."""
    n = kernel.shape[1]
    p = bf.predict(params, method, False)
    tp = jax.lax.stop_gradient(bf.predict(target, method, False)).reshape(n, 2, n)
    v = jnp.sum(pi[:, :, None] * tp, axis=1)
    boot = gamma * (kernel @ v)
    labels = (1 - gamma) * kernel + boot
    if method == "density":
        return -jnp.sum(
            wts
            * jnp.sum(
                labels * jax.nn.log_softmax(params["f"] @ params["b"].T, axis=-1),
                axis=-1,
            )
        )
    if method == "softsq":
        return 0.5 * jnp.sum(wts * jnp.mean((p - labels) ** 2, axis=-1))
    b = params["b"]
    ortho = 0.5 * jnp.mean((b @ b.T) ** 2) - jnp.mean(jnp.sum(b * b, axis=-1))
    td = 0.5 * jnp.sum(wts * jnp.mean((p - boot) ** 2, axis=-1)) - jnp.sum(
        wts * jnp.sum(kernel * p, axis=-1)
    )
    return td + ortho


def train_group(mdp, wts, d, method, seeds, rates, steps=TRAIN_STEPS, tau=TRAIN_TAU):
    n = mdp["n"]
    kernel, jp, jw = jnp.array(mdp["kernel"]), jnp.array(mdp["pi"]), jnp.array(wts)
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
        g = jax.grad(td_loss)(params, target, kernel, jp, GAMMA, method, jw)
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
    pred = np.asarray(jax.vmap(lambda x: bf.predict(x, method, False))(final))
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError(f"nonfinite prediction: n={n} d={d} {method}")
    return cfg, pred


def run_trained(mdp, wts, construction, w, d=TRAIN_RANK):
    rows = []
    n, r = mdp["n"], mdp["r"]
    for method in TRAIN_METHODS:
        cfg, pred = train_group(mdp, wts, d, method, SEEDS, TRAIN_RATES)
        for i, c in enumerate(cfg):
            P_hat = bf.readout(pred[i], method, GAMMA, n)
            Q = P_hat @ r / (1 - GAMMA)
            rows.append(
                dict(
                    construction=construction,
                    w=w,
                    N=mdp["N"],
                    n=n,
                    d=d,
                    arm=VARIANT_OF[method],
                    method=method,
                    seed=c["seed"],
                    lr=c["lr"],
                    **gap_stats(mdp, Q),
                    retained_mass=retained(mdp, P_hat),
                    retained_mass_w=retained(mdp, P_hat, wts),
                    max_mass_error=float(np.max(np.abs(P_hat.sum(1) - 1))),
                )
            )
    return rows


# Sweeps
def sweep_closed():
    rows = []
    t0 = time.time()
    base = build_mdp(0)
    for w in W_SWEEP:
        wts = row_weights(base, w)
        for d in RANKS:
            rows += run_variant_a(base, wts, d, "reweight", w)
            rows += run_variants_cd(base, wts, d, "reweight", w)
        g = lambda variant, d: np.mean(
            [
                x["gap_ratio"]
                for x in rows
                if x["construction"] == "reweight"
                and x["w"] == w
                and x["d"] == d
                and x["arm"] == variant
            ]
        )
        log(
            f"  [closed a] w={w:.4f}  d=3: A {g('A', 3 if 3 in RANKS else RANKS[0]):.3f} "
            f"C {g('C', 3 if 3 in RANKS else RANKS[0]):.3f} D {g('D', 3 if 3 in RANKS else RANKS[0]):.3f}"
            f"  ({time.time() - t0:.0f}s)"
        )
    for N in N_SWEEP:
        mdp = build_mdp(N)
        assert abs(mdp["gap"] - base["gap"]) < 1e-9, (N, mdp["gap"])
        assert np.allclose(mdp["M"][:18, :9], base["M"]) and np.allclose(
            mdp["M"][:18, 9:], 0.0
        )
        wts = row_weights(mdp, 1.0 / mdp["n"])
        assert np.allclose(wts, 1.0 / (2 * mdp["n"]))
        for d in RANKS:
            rows += run_variant_a(mdp, wts, d, "structural", 1.0 / mdp["n"])
            rows += run_variants_cd(mdp, wts, d, "structural", 1.0 / mdp["n"])
        dd = 3 if 3 in RANKS else RANKS[0]
        g = lambda variant: np.mean(
            [
                x["gap_ratio"]
                for x in rows
                if x["construction"] == "structural"
                and x["N"] == N
                and x["d"] == dd
                and x["arm"] == variant
            ]
        )
        log(
            f"  [closed b] N={N:3d} n={mdp['n']:3d}  d={dd}: A {g('A'):.3f} C {g('C'):.3f} D {g('D'):.3f}"
            f"  ({time.time() - t0:.0f}s)"
        )
    return rows


def sweep_trained():
    rows = []
    t0 = time.time()
    base = build_mdp(0)
    for w in W_SWEEP:
        if w == 0.0:
            continue  # TD needs every row to receive gradient
        rows += run_trained(base, row_weights(base, w), "reweight", w)
        g = lambda variant: np.mean(
            [
                x["gap_ratio"]
                for x in rows
                if x["construction"] == "reweight"
                and x["w"] == w
                and x["arm"] == variant
            ]
        )
        log(
            f"  [trained a] w={w:.4f}  A {g('A'):.3f} C {g('C'):.3f} D {g('D'):.3f}  ({time.time() - t0:.0f}s)"
        )
    for N in N_SWEEP:
        mdp = build_mdp(N)
        rows += run_trained(
            mdp, row_weights(mdp, 1.0 / mdp["n"]), "structural", 1.0 / mdp["n"]
        )
        g = lambda variant: np.mean(
            [
                x["gap_ratio"]
                for x in rows
                if x["construction"] == "structural"
                and x["N"] == N
                and x["arm"] == variant
            ]
        )
        log(
            f"  [trained b] N={N:3d}  A {g('A'):.3f} C {g('C'):.3f} D {g('D'):.3f}  ({time.time() - t0:.0f}s)"
        )
    return rows


def run_sanity():
    log("=== sanity ===")
    base = build_mdp(0)
    assert abs(base["gap"] - 1.4) < 1e-9
    log(f"exact gap at state 7: {base['gap']:.4f} (2.2 - 0.8)")
    facts = dict(
        gamma=GAMMA,
        decision_state=DECISION,
        goal=GOAL,
        gap_exact=base["gap"],
        w_sweep=W_SWEEP,
        n_sweep=N_SWEEP,
        ranks=RANKS,
        seeds=SEEDS,
        closed_form_adam=dict(
            steps=CLOSED_STEPS, polish_steps=POLISH_STEPS, lr=CLOSED_LR
        ),
        trained=dict(
            steps=TRAIN_STEPS,
            tau=TRAIN_TAU,
            rates=TRAIN_RATES,
            rank=TRAIN_RANK,
            ortho_coef_fb=1.0,
        ),
        structural={},
        limits={},
    )
    for N in N_SWEEP:
        m = build_mdp(N)
        assert abs(m["gap"] - base["gap"]) < 1e-9
        O = ordinary_rows(m)
        wts = row_weights(m, 1.0 / m["n"])
        tgt = np.sum(m["X"] ** 2, axis=1) / m["n"]
        facts["structural"][N] = dict(
            n=m["n"],
            rows=2 * m["n"],
            w=1.0 / m["n"],
            bn_share_target=share(wts, tgt, np.array([ROW0, ROW1])),
            gap=m["gap"],
            rank_X=int(np.linalg.matrix_rank(m["X"])),
            rank_X_ordinary=int(np.linalg.matrix_rank(m["X"][O])),
        )
        log(
            f"N={N:3d}: n={m['n']}, gap {m['gap']:.6f}, bottleneck share of the target's squared mass "
            f"{facts['structural'][N]['bn_share_target']:.4f}, rank(X)={facts['structural'][N]['rank_X']}, "
            f"rank(X_ordinary)={facts['structural'][N]['rank_X_ordinary']}"
        )
    for d in RANKS:
        lim, in_span, sg = fb_limit(base, d)
        facts["limits"][d] = dict(
            limit_gap_ratio=lim, diff_in_span=in_span, spectral_gap_rel=sg
        )
        log(
            f"d={d}: w->0 limit of A's gap fraction d^T P_d^0 r / d^T r = {lim:.4f}; "
            f"fraction of ||d||^2 inside span(P_d^0) = {in_span:.4f}; (lam_d - lam_d+1)/lam_1 = {sg:.4f}"
        )
    # Weighted SVD at w = 0 matches the limit; A+ keeps the gap exactly
    for d in RANKS:
        wts0 = row_weights(base, 0.0)
        a = run_variant_a(base, wts0, d, "check", 0.0)
        assert abs(a[0]["gap_ratio"] - facts["limits"][d]["limit_gap_ratio"]) < 1e-8
        assert abs(a[1]["gap_ratio"] - 1.0) < 1e-8
        wts_u = row_weights(base, 1 / 9)
        a = run_variant_a(base, wts_u, d, "check", 1 / 9)
        assert a[0]["cost_preserve"] >= -1e-10
    log(
        "uniform weights (w = 1/9) reproduce the toy-ablation SVD: "
        + ", ".join(
            f"d={d}: {run_variant_a(base, row_weights(base, 1 / 9), d, 'check', 1 / 9)[0]['gap_ratio']:.3f}"
            for d in RANKS
        )
    )
    (HERE / "exact.json").write_text(json.dumps(facts, indent=2, default=float))
    log("sanity passed.\n")
    return facts


# Aggregation, figure, tables
def t_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    m = float(values.mean())
    if n < 2:
        return m, m, m
    from scipy import stats

    tcrit = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1))
    h = tcrit * float(values.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h


def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            x = r[k]
            if isinstance(v, float):
                ok &= abs(float(x) - v) < 1e-12
            elif isinstance(v, int):
                ok &= int(x) == v
            else:
                ok &= str(x) == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def agg(rows, key="gap_ratio", **kw):
    vals = [float(r[key]) for r in sel(rows, **kw)]
    return t_interval(vals) if vals else (np.nan, np.nan, np.nan)


VARIANT_COLOR = {"A": "#DE8F05", "C": "#D55E00", "D": "#029E73", "A+": "#CC78BC"}
VARIANT_MARKER = {"A": "o", "C": "^", "D": "D", "A+": "s"}
VARIANT_LABEL = {
    "A": "A  FB (squared error)",
    "C": "C  softmax + squared error",
    "D": "D  Density-FB (softmax + cross-entropy)",
}
EXACT_COLOR = "#0173B2"
INK = "#2a2a2a"
W_POS = {0.0: 1 / 1024}  # x-position of the w = 0 point on the log axis


def xpos(w):
    return W_POS.get(float(w), float(w))


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=8, length=2.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9a9a9a")


def _w_axis(ax, w_sweep=None, uniform=1 / 9):
    W_SWEEP = globals()["W_SWEEP"] if w_sweep is None else list(w_sweep)
    ax.set_xscale("log")
    shown = [w for w in (1 / 2, 1 / 16, 1 / 128) if w in W_SWEEP]
    ticks = shown + [W_POS[0.0]]
    labels = [("1/%d" % round(1 / w)) for w in shown] + ["0"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xticks([], minor=True)
    ax.axvline(uniform, color="#cfcfcf", linewidth=0.7, linestyle=":", zorder=1)


def draw_panel_gap(
    ax,
    closed,
    trained,
    d=3,
    variants=("A", "C", "D"),
    construction="reweight",
    w_sweep=None,
    uniform=1 / 9,
):
    """Gap fraction vs row weight at rank d."""
    W_SWEEP = globals()["W_SWEEP"] if w_sweep is None else list(w_sweep)
    ax.axhline(1.0, color=EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    for variant in variants:
        xs, ys, los, his = [], [], [], []
        for w in W_SWEEP:
            m, lo, hi = agg(
                closed, construction=construction, w=float(w), d=d, arm=variant
            )
            xs.append(xpos(w))
            ys.append(m)
            los.append(lo)
            his.append(hi)
        ax.fill_between(
            xs,
            los,
            his,
            color=VARIANT_COLOR[variant],
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=VARIANT_COLOR[variant],
            marker=VARIANT_MARKER[variant],
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.4,
            zorder=3,
        )
        tx, ty, tlo, thi = [], [], [], []
        for w in W_SWEEP:
            m, lo, hi = agg(
                trained, construction=construction, w=float(w), d=d, arm=variant
            )
            if np.isfinite(m):
                tx.append(
                    xpos(w)
                    * (1.06 if variant == "D" else 0.94 if variant == "A" else 1.0)
                )
                ty.append(m)
                tlo.append(m - lo)
                thi.append(hi - m)
        if tx:
            ax.errorbar(
                tx,
                ty,
                yerr=[tlo, thi],
                fmt=VARIANT_MARKER[variant],
                markersize=4.2,
                markerfacecolor="none",
                markeredgecolor=VARIANT_COLOR[variant],
                markeredgewidth=1.0,
                ecolor=VARIANT_COLOR[variant],
                elinewidth=0.8,
                capsize=1.6,
                linestyle="none",
                zorder=4,
            )
    _w_axis(ax, W_SWEEP, uniform)
    _style(ax, "bottleneck weight $w$", "fraction of the exact action gap")


def draw_panel_share(ax, closed, d=3):
    xs = [xpos(w) for w in W_SWEEP]
    a_share = [
        agg(closed, "bn_share_obj", construction="reweight", w=float(w), d=d, arm="A")[
            0
        ]
        for w in W_SWEEP
    ]
    a_tgt = [
        agg(
            closed, "bn_share_target", construction="reweight", w=float(w), d=d, arm="A"
        )[0]
        for w in W_SWEEP
    ]
    d_share = [
        agg(closed, "bn_share_obj", construction="reweight", w=float(w), d=d, arm="D")[
            0
        ]
        for w in W_SWEEP
    ]
    ax.plot(xs, a_tgt, color="#949494", linestyle=":", linewidth=1.1, zorder=2)
    ax.plot(
        xs,
        a_share,
        color=VARIANT_COLOR["A"],
        marker=VARIANT_MARKER["A"],
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.4,
        zorder=3,
    )
    ax.plot(
        xs,
        d_share,
        color=VARIANT_COLOR["D"],
        marker=VARIANT_MARKER["D"],
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.4,
        zorder=3,
    )
    ax.set_ylim(-0.02, 0.62)
    _w_axis(ax)
    _style(ax, "bottleneck weight $w$", "bottleneck rows' share of the objective")
    ax2 = ax.twinx()
    cost = [
        agg(
            closed,
            "cost_preserve_rel",
            construction="reweight",
            w=float(w),
            d=d,
            arm="A",
        )[0]
        for w in W_SWEEP
    ]
    ax2.plot(
        xs,
        cost,
        color=VARIANT_COLOR["A+"],
        marker=VARIANT_MARKER["A+"],
        markersize=3.6,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.4,
        linestyle="-",
        zorder=3,
    )
    ax2.set_ylabel(
        "FB: relative cost of keeping the gap", fontsize=8.5, color=VARIANT_COLOR["A+"]
    )
    ax2.tick_params(labelsize=8, length=2.5, colors=VARIANT_COLOR["A+"])
    ax2.set_ylim(bottom=-0.01)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(VARIANT_COLOR["A+"])
    ax2.spines["left"].set_visible(False)
    return ax2


def draw_panel_structural(ax, closed, trained, d=3, variants=("A", "C", "D")):
    ax.axhline(1.0, color=EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    pos = {N: i for i, N in enumerate(N_SWEEP)}
    for variant in variants:
        ys, los, his = [], [], []
        for N in N_SWEEP:
            m, lo, hi = agg(closed, construction="structural", N=N, d=d, arm=variant)
            ys.append(m)
            los.append(lo)
            his.append(hi)
        xs = [pos[N] for N in N_SWEEP]
        ax.fill_between(
            xs,
            los,
            his,
            color=VARIANT_COLOR[variant],
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=VARIANT_COLOR[variant],
            marker=VARIANT_MARKER[variant],
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.4,
            zorder=3,
        )
        tx, ty, tlo, thi = [], [], [], []
        for N in N_SWEEP:
            m, lo, hi = agg(trained, construction="structural", N=N, d=d, arm=variant)
            if np.isfinite(m):
                tx.append(
                    pos[N]
                    + (0.12 if variant == "D" else -0.12 if variant == "A" else 0.0)
                )
                ty.append(m)
                tlo.append(m - lo)
                thi.append(hi - m)
        if tx:
            ax.errorbar(
                tx,
                ty,
                yerr=[tlo, thi],
                fmt=VARIANT_MARKER[variant],
                markersize=4.2,
                markerfacecolor="none",
                markeredgecolor=VARIANT_COLOR[variant],
                markeredgewidth=1.0,
                ecolor=VARIANT_COLOR[variant],
                elinewidth=0.8,
                capsize=1.6,
                linestyle="none",
                zorder=4,
            )
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([str(N) for N in N_SWEEP])
    _style(ax, "extra ordinary states $N$", "fraction of the exact action gap")


def legend_handles(with_share=True):
    from matplotlib.lines import Line2D

    hs = [
        Line2D(
            [0],
            [0],
            color=VARIANT_COLOR[a],
            marker=VARIANT_MARKER[a],
            markersize=4.2,
            linewidth=1.4,
            label=VARIANT_LABEL[a],
        )
        for a in ("A", "C", "D")
    ]
    hs.append(
        Line2D(
            [0],
            [0],
            color=EXACT_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="exact gap",
        )
    )
    hs.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            markersize=4.2,
            markerfacecolor="none",
            linestyle="none",
            label="hollow: TD-trained, mean and 95% CI (10 runs)",
        )
    )
    if with_share:
        hs.append(
            Line2D(
                [0],
                [0],
                color="#949494",
                linestyle=":",
                linewidth=1.1,
                label="bottleneck share of the target's squared mass",
            )
        )
        hs.append(
            Line2D(
                [0],
                [0],
                color=VARIANT_COLOR["A+"],
                marker=VARIANT_MARKER["A+"],
                markersize=4.0,
                linewidth=1.4,
                label="FB: (loss of gap-keeping fit $-$ optimum) / optimum",
            )
        )
    return hs


def save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    log(f"wrote {FIG_DIR / stem}.pdf / .png")


def make_figures(closed, trained):
    import matplotlib
    from matplotlib.lines import Line2D

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    d = 3 if 3 in RANKS else RANKS[0]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.1))
    draw_panel_gap(axes[0], closed, trained, d=d)
    axes[0].set_ylim(-0.25, 1.15)
    draw_panel_share(axes[1], closed, d=d)
    draw_panel_structural(axes[2], closed, trained, d=d)
    axes[2].set_ylim(-0.25, 1.15)
    axes[2].set_ylabel("")
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=2,
        fontsize=7.2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.33),
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "fig_rare_bottleneck")
    plt.close(fig)

    # Main-text panel: decision rows' share of the objective (right axis) and FB's gap
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    draw_panel_gap(ax, closed, trained, d=d)
    ax.set_ylim(-0.25, 1.15)
    ax2 = ax.twinx()
    share = [
        agg(
            closed, "bn_share_target", construction="reweight", w=float(w), d=d, arm="A"
        )[0]
        for w in W_SWEEP
    ]
    ax2.plot(
        [xpos(w) for w in W_SWEEP],
        share,
        color="#949494",
        linestyle=":",
        linewidth=1.2,
        zorder=2,
    )
    ax2.set_ylabel(
        "bottleneck share of the squared mass", fontsize=7.6, color="#6a6a6a"
    )
    ax2.tick_params(labelsize=7.6, length=2.5, colors="#6a6a6a")
    ax2.set_ylim(bottom=-0.01)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#9a9a9a")
    ax.legend(
        handles=legend_handles(with_share=True)[:4]
        + [
            Line2D(
                [0],
                [0],
                color="#949494",
                linestyle=":",
                linewidth=1.2,
                label="bottleneck share of the squared mass",
            )
        ],
        loc="center left",
        fontsize=6.4,
        frameon=False,
        handlelength=1.6,
        labelspacing=0.35,
        borderpad=0.2,
    )
    fig.tight_layout(pad=0.3)
    save(fig, "fig_rare_bottleneck_lemma")
    plt.close(fig)

    # Supplement: gap fraction vs w (row 1) and vs N (row 2) at every rank
    fig, axes = plt.subplots(2, len(RANKS), figsize=(5.5, 3.6), sharey=True)
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T
    for j, dd in enumerate(RANKS):
        draw_panel_gap(axes[0, j], closed, trained if dd == TRAIN_RANK else [], d=dd)
        draw_panel_structural(
            axes[1, j], closed, trained if dd == TRAIN_RANK else [], d=dd
        )
        axes[0, j].text(
            0.03, 0.05, f"$d={dd}$", transform=axes[0, j].transAxes, fontsize=8
        )
        axes[1, j].text(
            0.03, 0.05, f"$d={dd}$", transform=axes[1, j].transAxes, fontsize=8
        )
        axes[0, j].set_xlabel("bottleneck weight $w$" if j == 1 else "")
        axes[1, j].set_xlabel("extra ordinary states $N$" if j == 1 else "")
        axes[0, j].set_ylabel("fraction of the exact gap" if j == 0 else "")
        axes[1, j].set_ylabel("fraction of the exact gap" if j == 0 else "")
    axes[0, 0].set_ylim(-0.25, 1.2)
    fig.legend(
        handles=legend_handles(with_share=False),
        loc="lower center",
        ncol=3,
        fontsize=7.2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=1.0)
    save(fig, "fig_rare_bottleneck_ranks")
    plt.close(fig)


def fmt_w(w, uniform=1 / 9):
    return (
        "0 (limit)"
        if w == 0
        else (
            "1/%d (uniform)" % round(1 / uniform)
            if abs(w - uniform) < 1e-12
            else "1/%d" % round(1 / w)
        )
    )


def print_tables(closed, trained):
    for d in RANKS:
        log(
            f"\n=== construction (a) row reweighting, d={d}: gap fraction (C, D mean of {len(SEEDS)} seeds) ==="
        )
        log(
            "w | A gap | C gap | D gap | A bn share of obj | A bn share of target | A cost of keeping gap "
            "(abs, rel; ordinary part, bottleneck part) | D bn share of obj | A retained | D retained | A+ retained | A limit"
        )
        for w in W_SWEEP:
            a = sel(closed, construction="reweight", w=float(w), d=d, arm="A")[0]
            c = agg(closed, construction="reweight", w=float(w), d=d, arm="C")
            dd = agg(closed, construction="reweight", w=float(w), d=d, arm="D")
            ds = agg(
                closed,
                "bn_share_obj",
                construction="reweight",
                w=float(w),
                d=d,
                arm="D",
            )[0]
            dr = agg(
                closed,
                "retained_mass",
                construction="reweight",
                w=float(w),
                d=d,
                arm="D",
            )[0]
            ap = sel(closed, construction="reweight", w=float(w), d=d, arm="A+")[0]
            log(
                f"{fmt_w(w)} | {float(a['gap_ratio']):.3f} | {c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}] | "
                f"{dd[0]:.3f} [{dd[1]:.3f},{dd[2]:.3f}] | {float(a['bn_share_obj']):.4f} | "
                f"{float(a['bn_share_target']):.4f} | {float(a['cost_preserve']):.4f}, "
                f"{float(a['cost_preserve_rel']):.3f}; {float(a['cost_preserve_ord']):.4f}, "
                f"{float(a['cost_preserve_bn']):.4f} | {ds:.4f} | {float(a['retained_mass']):.3f} | "
                f"{dr:.3f} | {float(ap['retained_mass']):.3f} | {float(a['limit_gap_ratio']):.3f}"
            )
    for d in RANKS:
        log(f"\n=== construction (b) structural rarity, d={d} ===")
        log(
            "N | n | w=1/n | A gap | C gap | D gap | A bn share of obj | A bn share of target | A cost (abs, rel) | D bn share | A limit (P_d^0 of the ordinary rows)"
        )
        for N in N_SWEEP:
            a = sel(closed, construction="structural", N=N, d=d, arm="A")[0]
            c = agg(closed, construction="structural", N=N, d=d, arm="C")
            dd = agg(closed, construction="structural", N=N, d=d, arm="D")
            ds = agg(
                closed, "bn_share_obj", construction="structural", N=N, d=d, arm="D"
            )[0]
            log(
                f"{N} | {a['n']} | {float(a['w']):.4f} | {float(a['gap_ratio']):.3f} | "
                f"{c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}] | {dd[0]:.3f} [{dd[1]:.3f},{dd[2]:.3f}] | "
                f"{float(a['bn_share_obj']):.4f} | {float(a['bn_share_target']):.4f} | "
                f"{float(a['cost_preserve']):.4f}, {float(a['cost_preserve_rel']):.3f} | {ds:.4f} | "
                f"{float(a['limit_gap_ratio']):.3f}"
            )
    if trained:
        log(
            f"\n=== TD-trained, d={TRAIN_RANK}: gap fraction, mean of 10 runs [95% t-interval]; per-rate means ==="
        )
        for cons, keyname, keys in (
            ("reweight", "w", [w for w in W_SWEEP if w > 0]),
            ("structural", "N", N_SWEEP),
        ):
            log(f"--- {cons} ---")
            for k in keys:
                kw = {keyname: (float(k) if keyname == "w" else int(k))}
                parts = []
                for variant in ("A", "C", "D"):
                    m = agg(trained, construction=cons, d=TRAIN_RANK, arm=variant, **kw)
                    per = [
                        agg(
                            trained,
                            construction=cons,
                            d=TRAIN_RANK,
                            arm=variant,
                            lr=float(lr),
                            **kw,
                        )[0]
                        for lr in TRAIN_RATES
                    ]
                    ok = np.mean(
                        [
                            float(str(r["argmax_correct"]) == "True")
                            for r in sel(
                                trained,
                                construction=cons,
                                d=TRAIN_RANK,
                                arm=variant,
                                **kw,
                            )
                        ]
                    )
                    parts.append(
                        f"{variant} {m[0]:.3f} [{m[1]:.3f},{m[2]:.3f}] (lr {per[0]:.3f}/{per[1]:.3f}; argmax ok {ok:.1f})"
                    )
                label = fmt_w(k) if keyname == "w" else f"N={k}"
                log(f"{label} | " + " | ".join(parts))
    log(
        "\n=== D: predicted vs exact goal mass on the two state-7 rows (construction a, d=3, seed mean) ==="
    )
    for w in W_SWEEP:
        d = 3 if 3 in RANKS else RANKS[0]
        g0 = agg(
            closed, "bn_goal_mass_0", construction="reweight", w=float(w), d=d, arm="D"
        )[0]
        g1 = agg(
            closed, "bn_goal_mass_1", construction="reweight", w=float(w), d=d, arm="D"
        )[0]
        r0 = sel(closed, construction="reweight", w=float(w), d=d, arm="D")[0]
        log(
            f"{fmt_w(w)} | action 0: {g0:.4f} (exact {float(r0['bn_goal_mass_0_true']):.4f}) | "
            f"action 1: {g1:.4f} (exact {float(r0['bn_goal_mass_1_true']):.4f})"
        )


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


if __name__ == "__main__":
    t0 = time.time()
    if "--figures-only" in sys.argv:
        closed, trained = read_csv("closed.csv"), read_csv("trained.csv")
    else:
        run_sanity()
        closed = sweep_closed()
        write_csv(closed, "closed.csv")
        if "--closed-only" in sys.argv:
            trained = read_csv("trained.csv")
        else:
            trained = sweep_trained()
            write_csv(trained, "trained.csv")
    make_figures(closed, trained)
    print_tables(closed, trained)
    log(f"\ntotal wall time {time.time() - t0:.0f}s")
    if "--figures-only" not in sys.argv:
        (HERE / "run_log.txt").write_text("\n".join(LOG_LINES) + "\n")
