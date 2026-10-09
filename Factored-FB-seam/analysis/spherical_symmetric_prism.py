"""Bounded exact-target search for the six-state directed triangular prism."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

OUT = Path(__file__).with_suffix('.json')
NSTATE = 6
ADVANCE = np.array([1, 2, 0, 4, 5, 3])
SWAP = np.array([3, 4, 5, 0, 1, 2])
IDENTITY = np.eye(NSTATE)
SIGN = np.array([1, 1, 1, -1, -1, -1])
PROJ = (np.ones((6, 6)) + np.outer(SIGN, SIGN)) / 6


def bkl(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def fit(target, seed=0):
    rng = np.random.default_rng(seed)
    centered = np.log(target) - np.log(target).mean(axis=1, keepdims=True)
    u, singular, vh = np.linalg.svd(centered)
    starts = [(u[:, :2] * np.sqrt(singular[:2]), vh[:2].T * np.sqrt(singular[:2]))]
    for _ in range(2):
        starts.append((rng.normal(size=(6, 2)), rng.normal(size=(6, 2))))
    best = None
    const = np.mean(np.sum(target * np.log(target), axis=1))

    def loss(parameters):
        f, b = parameters[:12].reshape(6, 2), parameters[12:].reshape(6, 2)
        logits = f @ b.T
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        residual = (np.exp(lp) - target) / 6
        grad = np.concatenate([(residual @ b).ravel(), (residual.T @ f).ravel()])
        return float(-np.mean(np.sum(target * lp, axis=1)) + const), grad

    for f, b in starts:
        result = minimize(loss, np.concatenate([f.ravel(), b.ravel()]), jac=True,
                          method='L-BFGS-B', options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-9})
        if best is None or result.fun < best.fun:
            best = result
    f, b = best.x[:12].reshape(6, 2), best.x[12:].reshape(6, 2)
    logits = f @ b.T
    q = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return dict(kl=float(best.fun), q=q.tolist(), f=f.tolist(), b=b.tolist(),
                grad=float(np.linalg.norm(best.jac)), success=bool(best.success))


records = []
for swap_probability in (.2, .3, .4, .45):
    for gamma in (.1, .3, .6, .85, .97):
        transition = (1 - swap_probability) * IDENTITY[ADVANCE] + swap_probability * IDENTITY[SWAP]
        target = (1 - gamma) * np.linalg.inv(IDENTITY - gamma * transition)
        fb = target @ PROJ
        singular = np.linalg.svd(target, compute_uv=False)
        if not np.isclose(np.sum((target - fb) ** 2), np.sum(singular[2:] ** 2), atol=1e-12):
            continue
        # Wanted decisions on the optimal path from A2: Advance, then Swap forever.
        pairs = [(3, 1), (0, 4), (0, 5)]
        if any(target[good, 0] <= target[bad, 0] for good, bad in pairs):
            continue
        bounds = []
        for good, bad in pairs:
            p1, p2 = target[[good, bad], 0]
            midpoint = (p1 + p2) / 2
            bounds.append(float((bkl(p1, midpoint) + bkl(p2, midpoint)) / 6))
        density = fit(target)
        record = dict(swap_probability=swap_probability, gamma=gamma,
                      target=target.tolist(), fb=fb.tolist(), singular=singular.tolist(),
                      bounds=bounds, ratio=density['kl'] / min(bounds), density=density,
                      fb_goal_gap=float(fb[3, 0] - fb[1, 0]),
                      exact_goal_gap=float(target[3, 0] - target[1, 0]))
        records.append(record)
        OUT.write_text(json.dumps(records, indent=2) + '\n')
        print({key: record[key] for key in ('swap_probability', 'gamma', 'ratio',
                                           'fb_goal_gap', 'exact_goal_gap')},
              'KL', density['kl'], 'bounds', bounds, flush=True)
