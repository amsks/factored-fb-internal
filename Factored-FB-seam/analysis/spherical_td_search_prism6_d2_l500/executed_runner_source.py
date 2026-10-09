"""Small full-expectation joint forward/backward TD check.

Path4: deterministic reflecting left/right actions, uniform continuation policy,
uniform state-action and candidate weights. Rank2. FB backward outputs are
unconstrained, with the standard second-moment penalty of weight 1. Density uses
ordinary rank2 bilinear logits and the exact categorical bootstrap target.
All differences from production (single critic, no actor, full expectations,
tabular factors, fixed policy) are deliberate. Exact-successor SVD is only a
comparison, not called the optimum of the joint TD training dynamics.
"""

import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


def problem(name):
    n = 4
    if name == "path4":
        nexts = np.stack(
            (np.maximum(0, np.arange(n) - 1), np.minimum(n - 1, np.arange(n) + 1)), axis=1
        )
        pi = np.full((n, 2), 0.5)
        d = 2
        goal = 1
    else:
        nexts = np.stack((np.arange(n), (np.arange(n) + 1) % n), axis=1)
        pi = np.tile([0.0, 1.0], (n, 1))
        d = 3
        goal = 0
    return np.eye(n)[nexts], pi, d, goal


def backward(params, sphere=False):
    b = params["b"]
    if sphere:
        b = (
            b
            * jnp.sqrt(b.shape[1])
            / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        )
    return b


def predict(params, method, sphere=False):
    logits = params["f"] @ backward(params, sphere).T
    return jax.nn.softmax(logits, axis=-1) if method == "density" else logits


def loss_fn(params, target, kernel, pi, gamma, method, sphere=False):
    n = kernel.shape[1]
    p = predict(params, method, sphere)
    # Average the fixed continuation policy, never the greedy evaluation action.
    tp = jax.lax.stop_gradient(predict(target, method, sphere)).reshape(n, 2, n)
    v = jnp.sum(pi[:, :, None] * tp, axis=1)
    boot = gamma * (kernel @ v)
    if method == "density":
        labels = (1 - gamma) * kernel + boot
        return -jnp.mean(
            jnp.sum(labels * jax.nn.log_softmax(params["f"] @ params["b"].T, axis=-1), axis=-1)
        )
    b = backward(params, sphere)
    # Population counterpart of FB's off-diagonal square and diagonal linear
    # term. Independent candidate draws may have the same state label; do not
    # exclude those pairs by masking the four-state table's diagonal.
    td = 0.5 * jnp.mean((p - boot) ** 2) - jnp.mean(jnp.sum(kernel * p, axis=-1))
    ortho = 0.5 * jnp.mean((b @ b.T) ** 2) - jnp.mean(jnp.sum(b * b, axis=-1))
    return td + ortho


def exact_and_svd(p, pi, gamma, d):
    n = p.shape[0]
    T = np.sum(pi[:, :, None] * p, axis=1)
    D = (1 - gamma) * np.linalg.inv(np.eye(n) - gamma * T)
    P = p.reshape(-1, n) @ D
    U, S, Vt = np.linalg.svd(P / np.sqrt(2), full_matrices=False)
    H = np.sqrt(2) * (U[:, :d] * S[:d]) @ Vt[:d]
    B = np.sqrt(n) * Vt[:d].T
    F = H @ B / (1 - gamma)
    return P, H, S, {"f": jnp.asarray(F), "b": jnp.asarray(B)}


def eval_metrics(params, p, pi, gamma, method, sphere, goal, P, H):
    n = p.shape[0]
    pred = np.asarray(predict(params, method, sphere))
    Q = pred if method == "density" else (1 - gamma) * pred / n
    gap = (Q[:, goal].reshape(n, 2)[:, 1] - Q[:, goal].reshape(n, 2)[:, 0]) / (1 - gamma)
    truegap = (P[:, goal].reshape(n, 2)[:, 1] - P[:, goal].reshape(n, 2)[:, 0]) / (1 - gamma)
    svdgap = (H[:, goal].reshape(n, 2)[:, 1] - H[:, goal].reshape(n, 2)[:, 0]) / (1 - gamma)
    bp = np.asarray(backward(params, sphere))
    b0 = np.ones(n)
    b1 = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    proj = bp @ np.linalg.pinv(bp)
    # A stationary-point diagnostic with the target set equal to the online
    # parameters. This is distinct from the update gradient at lagged targets.
    grad = jax.grad(loss_fn)(
        params, params, jnp.asarray(p.reshape(-1, n)), jnp.asarray(pi), gamma, method, sphere
    )
    return {
        "start_gap": float(gap[0]),
        "true_start_gap": float(truegap[0]),
        "svd_start_gap": float(svdgap[0]),
        "all_state_gaps": gap.tolist(),
        "true_state_gaps": truegap.tolist(),
        "wrong_strict_states": int(np.sum(gap * truegap < -1e-10)),
        "rmse_to_exact": float(np.sqrt(np.mean((Q - P) ** 2))),
        "rmse_to_svd": float(np.sqrt(np.mean((Q - H) ** 2))),
        "backward_second_moment_error": float(np.linalg.norm(bp.T @ bp / n - np.eye(bp.shape[1]))),
        "contains_constant_error": float(np.linalg.norm((np.eye(n) - proj) @ b0)),
        "contains_path_fiedler_error": float(np.linalg.norm((np.eye(n) - proj) @ b1)),
        "gradient_norm": float(
            np.sqrt(sum(np.sum(np.asarray(g) ** 2) for g in jax.tree.leaves(grad)))
        ),
        "min_entry": float(Q.min()),
        "max_mass_error": float(np.max(abs(Q.sum(axis=1) - 1))),
    }


def train(args, method):
    p, pi, d, goal = problem(args.problem)
    n = len(p)
    kernel = jnp.array(p.reshape(-1, n))
    jp = jnp.array(pi)
    cfg = [
        {"seed": seed, "lr": rate, "gamma": gamma}
        for gamma in args.gamma
        for seed in args.seeds
        for rate in args.rates
    ]
    exact = [exact_and_svd(p, pi, c["gamma"], d) for c in cfg]

    def init(c):
        rng = np.random.default_rng(c["seed"])
        return {
            "f": jnp.array(rng.normal(0, 0.1, (2 * n, d))),
            "b": jnp.array(rng.normal(0, 0.1, (n, d))),
        }

    params = jax.tree.map(lambda *x: jnp.stack(x), *[init(c) for c in cfg])
    optimizer = optax.scale_by_adam()
    state = jax.vmap(optimizer.init)(params)
    carry = (params, params, state)
    rates = jnp.array([c["lr"] for c in cfg])
    gammas = jnp.array([c["gamma"] for c in cfg])
    sphere = args.sphere and method == "fb"

    def one(params, target, state, rate, gamma):
        grad = jax.grad(loss_fn)(params, target, kernel, jp, gamma, method, sphere)
        update, state = optimizer.update(grad, state, params)
        params = jax.tree.map(lambda x, u: x - rate * u, params, update)
        target = jax.tree.map(lambda t, x: (1 - args.tau) * t + args.tau * x, target, params)
        return params, target, state

    @jax.jit
    def chunk(carry):
        def body(c, _):
            return jax.vmap(one)(*c, rates, gammas), None

        return jax.lax.scan(body, carry, None, length=args.interval)[0]

    rows = []
    snaps = []
    t0 = time.monotonic()
    for step in range(0, args.steps + 1, args.interval):
        if step:
            carry = chunk(carry)
        pred = np.asarray(jax.vmap(lambda x: predict(x, method, sphere))(carry[0]))
        snaps.append(pred)
        if not np.all(np.isfinite(pred)):
            raise FloatingPointError("nonfiniteprediction")
        last = []
        for i, c in enumerate(cfg):
            par = jax.tree.map(lambda x: x[i], carry[0])
            P, H, S, svdpar = exact[i]
            r = dict(
                c,
                method=method,
                step=step,
                **eval_metrics(par, p, pi, c["gamma"], method, sphere, goal, P, H),
            )
            rows.append(r)
            last.append(r)
        print(
            json.dumps(
                {
                    "method": method,
                    "problem": args.problem,
                    "sphere": sphere,
                    "step": step,
                    "elapsed": round(time.monotonic() - t0, 2),
                    "results": [
                        {
                            "g": r["gamma"],
                            "seed": r["seed"],
                            "lr": r["lr"],
                            "startgap": round(r["start_gap"], 9),
                            "wrong": r["wrong_strict_states"],
                            "grad": r["gradient_norm"],
                        }
                        for r in last
                    ],
                }
            ),
            flush=True,
        )
        (args.out / f"{method}_latest.json").write_text(json.dumps(last, indent=2))
    np.savez_compressed(
        args.out / f"{method}.npz",
        predictions=np.stack(snaps),
        kernels=p,
        policy=pi,
        **{f"final_{k}": np.asarray(v) for k, v in carry[0].items()},
        **{f"target_{k}": np.asarray(v) for k, v in carry[1].items()},
        exact=np.stack([x[0] for x in exact]),
        svd=np.stack([x[1] for x in exact]),
    )
    (args.out / f"{method}_history.json").write_text(json.dumps(rows, indent=2))
    (args.out / f"{method}_configs.json").write_text(json.dumps(cfg, indent=2))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("analysis/deterministic_didactic/path4_td"))
    ap.add_argument("--problem", choices=["path4", "cycle4"], default="path4")
    ap.add_argument("--methods", nargs="+", default=["fb", "density"])
    ap.add_argument("--gamma", nargs="+", type=float, default=[0.5, 0.9, 0.99])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.001, 0.01])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--interval", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--sphere", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "config.json").exists():
        raise FileExistsError("outputhasrun")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "out": str(args.out),
                "numpy": np.__version__,
                "jax": jax.__version__,
                "optax": optax.__version__,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    p, pi, d, goal = problem(args.problem)
    checks = []
    for gamma in args.gamma:
        P, H, S, params = exact_and_svd(p, pi, gamma, d)
        g = jax.grad(loss_fn)(
            params, params, jnp.array(p.reshape(-1, 4)), jnp.array(pi), gamma, "fb", args.sphere
        )
        checks.append(
            {
                "gamma": gamma,
                "svd_singular_values": S.tolist(),
                "svd_frozen_target_grad_norm": float(
                    np.sqrt(sum(np.sum(np.asarray(v) ** 2) for v in jax.tree.leaves(g)))
                ),
                "svd_goal_action_gaps_raw": (
                    (H[:, goal].reshape(4, 2)[:, 1] - H[:, goal].reshape(4, 2)[:, 0]) / (1 - gamma)
                ).tolist(),
            }
        )
    (args.out / "svd_stationarity.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({"svd_checks": checks}), flush=True)
    for method in args.methods:
        train(args, method)
