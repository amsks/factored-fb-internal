"""Fixed-policy TD on a bottleneck family, sweeping rank and task sparsity.

The family is a left chain of L states, a bottleneck b, and a right room of R
states whose last state is the goal. Crossing at b is the only way into the
room, and the room is closed. The decision the diagnostic reads is at the state
one step from the goal inside the room: advance to the goal, or retreat.

Three critics share the factorization F(s,a) B(u)^T at rank d, one head, exact
population expectations, a fixed uniform continuation policy, and float64.

  fb       predicts the density ratio m = M/rho with FB's squared Bellman
           residual minus the immediate linear term, plus the second-moment
           penalty at coefficient 1. Optional spherical backward rows.
  normsq   predicts the normalized measure (1-gamma)M directly with an ordinary
           squared Bellman residual, same factorization and same penalty. This
           is the amplitude control: it removes FB's density-ratio scale and
           loss form while keeping squared error.
  density  predicts softmax(F B^T) against the categorical target
           (1-gamma)e_{s'} + gamma * target-policy-average, no penalty, matching
           the density_fb defaults.
  softsq   the same softmax output class and the same target as density, scored
           with squared error instead of cross entropy, no penalty.
  expsq    exp(F B^T) with no normalization, squared error against the same
           target, no penalty. Positive and rank-d in log space but not a
           distribution, so it separates normalization from log-space rank.

fb and normsq carry the penalty, density and softsq do not; that follows the
shipped configs and is stated rather than equalized. The four arms form a two by
two over output class and loss: normsq and softsq are squared error on a linear
and a softmax output, density is cross entropy on the softmax output. normsq
against softsq isolates the output class at fixed loss; softsq against density
isolates the loss at fixed output class.

All differences from production -- single critic, no actor, full expectations,
tabular factors, fixed policy, known reward -- are deliberate.
"""

import argparse, hashlib, json, time
from itertools import product
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


def build(L: int, R: int):
    """Transition map, goal index, bottleneck index, near-goal decision index."""
    n = L + 1 + R
    b = L
    nx = np.zeros((n, 2), int)
    for i in range(L):
        nx[i, 0] = i + 1 if i + 1 < L else b     # advance
        nx[i, 1] = max(i - 1, 0)                 # retreat
    nx[b, 0] = L + 1                             # cross into the room
    nx[b, 1] = L - 1                             # slip back
    for j in range(R):
        s = L + 1 + j
        nx[s, 0] = L + 1 + min(j + 1, R - 1)     # advance toward the goal
        nx[s, 1] = L + 1 + max(j - 1, 0)         # retreat inside the room
    return np.eye(n)[nx], n, b, n - 1, L + 1 + R - 2


def exact(kernel2d, pi, gamma):
    """Normalized successor measure: rows are (1-gamma) sum_t gamma^t P(s_{t+1})."""
    n = kernel2d.shape[1]
    T = np.sum(pi[:, :, None] * kernel2d.reshape(n, 2, n), axis=1)
    return kernel2d @ ((1 - gamma) * np.linalg.inv(np.eye(n) - gamma * T))


def backward(params, sphere):
    b = params["b"]
    if sphere:
        b = b * jnp.sqrt(b.shape[1]) / jnp.maximum(
            jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return b


def predict(params, method, sphere):
    logits = params["f"] @ backward(params, sphere).T
    if method in ("density", "softsq"):
        return jax.nn.softmax(logits, axis=-1)
    if method == "expsq":
        return jnp.exp(jnp.clip(logits, -30.0, 30.0))
    return logits


def loss_fn(params, target, kernel, pi, gamma, method, sphere):
    n = kernel.shape[1]
    p = predict(params, method, sphere)
    tp = jax.lax.stop_gradient(predict(target, method, sphere)).reshape(n, 2, n)
    v = jnp.sum(pi[:, :, None] * tp, axis=1)
    boot = gamma * (kernel @ v)
    if method == "density":
        labels = (1 - gamma) * kernel + boot
        return -jnp.mean(jnp.sum(
            labels * jax.nn.log_softmax(params["f"] @ params["b"].T, axis=-1), axis=-1))
    if method in ("softsq", "expsq"):
        return 0.5 * jnp.mean((p - ((1 - gamma) * kernel + boot)) ** 2)
    b = backward(params, sphere)
    ortho = 0.5 * jnp.mean((b @ b.T) ** 2) - jnp.mean(jnp.sum(b * b, axis=-1))
    if method == "normsq":
        return 0.5 * jnp.mean((p - ((1 - gamma) * kernel + boot)) ** 2) + ortho
    # fb: squared residual against the bootstrap, minus the immediate linear term
    td = 0.5 * jnp.mean((p - boot) ** 2) - jnp.mean(jnp.sum(kernel * p, axis=-1))
    return td + ortho


def readout(pred, method, gamma, n):
    """Put every critic on the normalized-measure scale of `exact`."""
    if method == "fb":
        return (1 - gamma) * pred / n
    return pred


def metrics(pred, method, gamma, P, goal, b, near):
    n = P.shape[1]
    Q = readout(pred, method, gamma, n)
    gap = lambda M, s: M[2 * s + 1, goal] - M[2 * s, goal]
    col = np.abs(Q[:, goal] - P[:, goal])
    qg = Q[1::2, goal] - Q[0::2, goal]
    tg = P[1::2, goal] - P[0::2, goal]
    return {
        "q_decision": [float(Q[2 * near, goal]), float(Q[2 * near + 1, goal])],
        "q_decision_true": [float(P[2 * near, goal]), float(P[2 * near + 1, goal])],
        "state_gap_ratios": np.where(np.abs(tg) > 1e-12, qg / np.where(tg == 0, 1, tg),
                                     np.nan).tolist(),
        "gap_near": float(gap(Q, near)),
        "gap_near_true": float(gap(P, near)),
        "gap_ratio_near": float(gap(Q, near) / gap(P, near)),
        "gap_ratio_b": float(gap(Q, b) / gap(P, b)),
        "inverted_near": bool(gap(Q, near) * gap(P, near) < 0),
        "wrong_states": int(np.sum(
            (Q[1::2, goal] - Q[0::2, goal]) * (P[1::2, goal] - P[0::2, goal]) < -1e-10)),
        "goal_mass_near": float(P[2 * near, goal]),
        "abs_err_goal": float(col.max()),
        "rel_err_goal": float((col / np.maximum(P[:, goal], 1e-12)).max()),
        "l2_residual": float(np.sqrt(np.mean((Q - P) ** 2))),
        "min_entry": float(Q.min()),
        "max_mass_error": float(np.max(abs(Q.sum(axis=1) - 1))),
    }


def run_group(L, R, d, method, gamma, seeds, rates, steps, tau, sphere):
    """One (graph, rank, method) group; seeds x rates are vmapped."""
    p, n, b, goal, near = build(L, R)
    rows = run_kernel(p, b, goal, near, d, method, gamma, seeds, rates, steps, tau, sphere)
    return [dict(r, L=L, R=R) for r in rows]


def run_kernel(p, b, goal, near, d, method, gamma, seeds, rates, steps, tau, sphere):
    """Train one (transition kernel, rank, method) group. `p` is (n, 2, n) and
    each row is a distribution over next states, so stochastic graphs are fine."""
    n = p.shape[0]
    kernel_np = p.reshape(-1, n)
    P = exact(kernel_np, np.full((n, 2), 0.5), gamma)
    kernel, jp = jnp.array(kernel_np), jnp.full((n, 2), 0.5)
    cfg = [{"seed": s, "lr": r} for s in seeds for r in rates]

    def init(c):
        rng = np.random.default_rng(c["seed"])
        return {"f": jnp.array(rng.normal(0, 0.1, (2 * n, d))),
                "b": jnp.array(rng.normal(0, 0.1, (n, d)))}

    params = jax.tree.map(lambda *x: jnp.stack(x), *[init(c) for c in cfg])
    opt = optax.scale_by_adam()
    state = jax.vmap(opt.init)(params)
    lrs = jnp.array([c["lr"] for c in cfg])
    sph = sphere and method == "fb"

    def one(params, target, state, lr):
        g = jax.grad(loss_fn)(params, target, kernel, jp, gamma, method, sph)
        upd, state = opt.update(g, state, params)
        params = jax.tree.map(lambda x, u: x - lr * u, params, upd)
        target = jax.tree.map(lambda t, x: (1 - tau) * t + tau * x, target, params)
        return params, target, state

    @jax.jit
    def train(carry):
        return jax.lax.scan(lambda c, _: (jax.vmap(one)(*c, lrs), None),
                            carry, None, length=steps)[0]

    final = train((params, params, state))[0]
    pred = np.asarray(jax.vmap(lambda x: predict(x, method, sph))(final))
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError(f"nonfinite prediction: d={d} {method}")
    return [dict(c, d=d, n=n, method=method, gamma=gamma, sphere=sph,
                 rho_goal=1.0 / n,
                 **metrics(pred[i], method, gamma, P, goal, b, near))
            for i, c in enumerate(cfg)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("analysis/didactic_family/main"))
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--rooms", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4, 6])
    ap.add_argument("--methods", nargs="+", default=["fb", "normsq", "softsq", "expsq", "density"])
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.001, 0.01])
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--sphere", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "config.json").exists():
        raise FileExistsError("output has a run")
    (a.out / "config.json").write_text(json.dumps(
        {**vars(a), "out": str(a.out), "numpy": np.__version__, "jax": jax.__version__,
         "optax": optax.__version__,
         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))

    rows, t0 = [], time.monotonic()
    for R, d, m in product(a.rooms, a.ranks, a.methods):
        g = run_group(a.L, R, d, m, a.gamma, a.seeds, a.rates, a.steps, a.tau, a.sphere)
        rows += g
        print(json.dumps({"R": R, "d": d, "method": m, "n": g[0]["n"],
                          "gap_ratio_near": round(float(np.mean(
                              [r["gap_ratio_near"] for r in g])), 4),
                          "rel_err_goal": round(float(np.mean(
                              [r["rel_err_goal"] for r in g])), 3),
                          "l2": round(float(np.mean([r["l2_residual"] for r in g])), 4),
                          "inverted": sum(r["inverted_near"] for r in g),
                          "elapsed": round(time.monotonic() - t0, 1)}), flush=True)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {a.out/'summary.json'}  ({len(rows)} runs)")
