"""Spec 31 — zero-order z-search core for Density-FB.

Pure numpy at module level (importable from the torch test venv); jax is
imported lazily inside make_weights_fn only. The min-of-softmax-heads
weight computation here is THE shared implementation: the g293 oracle and
the spec-31 harness both call it, so their numerics agree by construction.
"""
from __future__ import annotations

import numpy as np


def sphere_project(z: np.ndarray) -> np.ndarray:
    """Scale rows to the standing ||z|| = sqrt(d) invariant (d = last axis)."""
    z = np.asarray(z, np.float64)
    d = z.shape[-1]
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return np.sqrt(d) * z / np.maximum(n, 1e-12)


def sphere_samples(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    return sphere_project(rng.standard_normal((n, d)))


def min_head_weights(f_emb: np.ndarray, b: np.ndarray, temp: float,
                     renorm: bool = False) -> np.ndarray:
    """f_emb [H, S, d], b [M, d] -> [S, M]: per-head softmax over M, then
    elementwise MIN over heads (the actor/eval convention — sub-normalized
    whenever heads disagree, spec 30 Item 4). renorm=True re-normalizes rows
    (the convention the affine-invariance assertion is exact under)."""
    logits = np.einsum("hbd,md->hbm", np.asarray(f_emb, np.float64),
                       np.asarray(b, np.float64)) / temp
    logits -= logits.max(-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(-1, keepdims=True)
    w = w.min(0)
    if renorm:
        w = w / w.sum(-1, keepdims=True)
    return w


def cem_search(score_fn, weights_of, d: int, rng: np.random.Generator,
               seed_z: np.ndarray | None = None, n_init: int = 512,
               n_round: int = 256, rounds: int = 3,
               elite_frac: float = 0.1):
    """Encoder-seeded CEM on the sqrt(d)-sphere (spec 31 §1).

    Round 0 = [seed_z?] + iid sphere samples; each refinement round refits a
    diagonal Gaussian on the elites of ALL candidates seen so far, resamples,
    and re-projects to the sphere. z_star = argmax J over every candidate
    evaluated. The full round-0 (Z, W, J) table is returned for the affine
    cells and corr(J, return) — W does not depend on the reward, so J under
    any reward transform is re-derivable from W0 without re-evaluating."""
    Z = sphere_samples(rng, n_init if seed_z is None else n_init - 1, d)
    if seed_z is not None:
        Z = np.concatenate([sphere_project(seed_z)[None], Z], 0)
    Z_all, W_all, J_all = [], [], []
    round_max, round_median = [], []
    for r in range(rounds + 1):
        W = np.asarray(weights_of(Z))
        J = np.asarray(score_fn(W), np.float64)
        Z_all.append(Z); W_all.append(W); J_all.append(J)
        round_max.append(float(J.max()))
        round_median.append(float(np.median(J)))
        if r == rounds:
            break
        Zc = np.concatenate(Z_all, 0)
        Jc = np.concatenate(J_all, 0)
        n_elite = max(2, int(np.ceil(elite_frac * len(Jc))))
        elites = Zc[np.argsort(Jc)[-n_elite:]]
        mu, sd = elites.mean(0), elites.std(0) + 1e-6
        Z = sphere_project(mu[None] + sd[None] * rng.standard_normal((n_round, d)))
    Zc = np.concatenate(Z_all, 0)
    Jc = np.concatenate(J_all, 0)
    k = int(np.argmax(Jc))
    z_star = Zc[k]
    j_seed = float(J_all[0][0]) if seed_z is not None else None
    cos_ss = (float(z_star @ Z_all[0][0] /
                    (np.linalg.norm(z_star) * np.linalg.norm(Z_all[0][0]) + 1e-12))
              if seed_z is not None else None)
    diag = {"J_seed": j_seed, "J_star": float(Jc[k]), "cos_seed_star": cos_ss,
            "round_max": round_max, "round_median": round_median,
            "Z0": Z_all[0], "W0": W_all[0], "J0": J_all[0],
            "n_evaluated": int(len(Jc))}
    return z_star, diag


def affine_argmax_check(W0: np.ndarray, r: np.ndarray, transforms):
    """Spec 31 §3 affine cell. W does not depend on r, so J under alpha*r+beta
    is a re-read of the cached table. Renormalized rows: argmax invariance is
    EXACT for alpha > 0 (beta adds a constant). Min-convention rows are
    sub-normalized (beta couples to ||w||_1): reported, never asserted."""
    W0 = np.asarray(W0, np.float64)
    r = np.asarray(r, np.float64)
    Wn = W0 / W0.sum(-1, keepdims=True)
    out = {"argmax_base": int(np.argmax(Wn @ r)), "renorm": {},
           "min_convention": {}}
    base_min = int(np.argmax(W0 @ r))
    for a, b in transforms:
        rt = a * r + b
        kr = int(np.argmax(Wn @ rt))
        km = int(np.argmax(W0 @ rt))
        out["renorm"][f"({a},{b})"] = {"argmax": kr,
                                       "match": kr == out["argmax_base"]}
        out["min_convention"][f"({a},{b})"] = {"argmax": km,
                                               "match": km == base_min}
    out["renorm_all_match"] = all(c["match"] for c in out["renorm"].values())
    out["min_all_match"] = all(c["match"] for c in out["min_convention"].values())
    return out


def make_weights_fn(agent, b_sub: np.ndarray, temp: float,
                    init_obs: np.ndarray, seed):
    """weights_of(Z [K,d]) -> W [K,M] for cem_search, through the agent's own
    actor + F towers (the g293 oracle computation, one candidate at a time).
    jax imported lazily so this module stays importable from the torch venv.
    `seed` is a jax PRNGKey; it is FIXED across candidates so J differences
    come from z, not action noise."""
    import jax.numpy as jnp
    b64 = np.asarray(b_sub, np.float64)
    S = len(init_obs)
    obs_j = jnp.asarray(np.asarray(init_obs, np.float32))

    def weights_of(Z):
        W = np.zeros((len(Z), len(b64)))
        for k, z in enumerate(np.asarray(Z, np.float32)):
            zb = jnp.asarray(np.tile(z[None], (S, 1)))
            a = agent.sample_actions_with_z(obs_j, zb, seed=seed)
            f = agent.network.select("density_f")(
                jnp.concatenate([obs_j, a, zb], -1))
            W[k] = min_head_weights(np.asarray(f), b64, temp).mean(0)
        return W

    return weights_of
