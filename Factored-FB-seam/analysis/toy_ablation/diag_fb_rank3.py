"""Why does TD-trained FB lose the gap at rank 3 when its closed-form SVD optimum keeps it?

2x2 over {bootstrapped TD target, exact supervised target} x {orthogonality penalty
coefficient 1 (harness default), 0}, FB loss, ranks 2-4, 5 seeds x 2 learning rates,
same init / optimizer / steps as the trained arm.  Writes diag_fb_rank3.csv.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import csv, sys, pathlib
import numpy as np
import jax, jax.numpy as jnp, optax

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run as R                       # noqa: E402
bf = R.bf

KERNEL, PI = jnp.array(R.KERNEL), jnp.array(R.PI)
RATIO_EXACT = jnp.array(R.N * R.M_EXACT)      # FB's target n M


def loss(params, target, ortho_coef, supervised):
    p = params["f"] @ params["b"].T
    if supervised:
        td = 0.5 * jnp.mean((p - RATIO_EXACT) ** 2)
    else:
        n = KERNEL.shape[1]
        tp = jax.lax.stop_gradient(target["f"] @ target["b"].T).reshape(n, 2, n)
        v = jnp.sum(PI[:, :, None] * tp, axis=1)
        boot = R.GAMMA * (KERNEL @ v)
        td = 0.5 * jnp.mean((p - boot) ** 2) - jnp.mean(jnp.sum(KERNEL * p, axis=-1))
    b = params["b"]
    ortho = 0.5 * jnp.mean((b @ b.T) ** 2) - jnp.mean(jnp.sum(b * b, axis=-1))
    return td + ortho_coef * ortho


def group(d, ortho_coef, supervised):
    cfg = [{"seed": s, "lr": r} for s in R.SEEDS for r in R.TRAIN_RATES]

    def init(c):
        rng = np.random.default_rng(c["seed"])
        return {"f": jnp.array(rng.normal(0, 0.1, (2 * R.N, d))),
                "b": jnp.array(rng.normal(0, 0.1, (R.N, d)))}

    params = jax.tree.map(lambda *x: jnp.stack(x), *[init(c) for c in cfg])
    opt = optax.scale_by_adam(); state = jax.vmap(opt.init)(params)
    lrs = jnp.array([c["lr"] for c in cfg])

    def one(params, target, state, lr):
        g = jax.grad(loss)(params, target, ortho_coef, supervised)
        upd, state = opt.update(g, state, params)
        params = jax.tree.map(lambda x, u: x - lr * u, params, upd)
        target = jax.tree.map(lambda t, x: (1 - R.TRAIN_TAU) * t + R.TRAIN_TAU * x, target, params)
        return params, target, state

    @jax.jit
    def train(carry):
        return jax.lax.scan(lambda c, _: (jax.vmap(one)(*c, lrs), None), carry, None,
                            length=R.TRAIN_STEPS)[0]

    final = train((params, params, state))[0]
    F, B = np.asarray(final["f"]), np.asarray(final["b"])
    rows = []
    for i, c in enumerate(cfg):
        Q = R.linear_readout(F[i], B[i])
        P_hat = (1 - R.GAMMA) * (F[i] @ B[i].T) / R.N
        rows.append(dict(d=d, ortho_coef=ortho_coef, target="exact" if supervised else "bootstrap",
                         **c, **R.gap_stats(Q), retained_mass=R.retained_mass(P_hat)))
    return rows


if __name__ == "__main__":
    rows = []
    for d in (2, 3, 4):
        for supervised in (False, True):
            for oc in (1.0, 0.0):
                g = group(d, oc, supervised); rows += g
                m, lo, hi = R.t_interval([r["gap_ratio"] for r in g])
                print(f"d={d} target={'exact' if supervised else 'bootstrap':9s} ortho={oc}: "
                      f"gap ratio {m:.3f} [{lo:.3f}, {hi:.3f}]  retained "
                      f"{np.mean([r['retained_mass'] for r in g]):.3f}", flush=True)
    R.write_csv(rows, "diag_fb_rank3.csv")
