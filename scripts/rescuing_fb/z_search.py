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
