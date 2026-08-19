"""Exact tabular ground truth for the Density-FB successor measure (spec 30a Item 4).

Pure numpy, no jax, no env, no checkpoint: a toy MDP small enough that the
successor measure is a matrix inverse, so the zero-shot de-biased readout
(`eval_density_zeroshot.compute_density_primary`) can be checked against an
object computed by linear algebra rather than against another estimator.

CONVENTIONS PINNED HERE (these are what the tests assert, and each one is a
place a sign or an off-by-one could hide):

  * MEASURE START INDEX -- the shipped TD target is
        t = (1-gamma) * k(x')  +  gamma * w_bar(s', a'~pi)
    (`gciql_flow.density_loss`), whose population fixed point is
        w(s,a) = (1-gamma) * P[s,a,:] (I - gamma P^pi)^{-1}
    i.e. `w = nu = (1-gamma) M` with
        M[s,a,x] = sum_{t>=1} gamma^{t-1} Pr(x_t = x | s_0=s, a_0=a).
    The sum starts at the SUCCESSOR x_1. x_0 is never in it (the same statement
    `eval_measure_g4_spearman.py` lines 380-387 makes about the MC ground truth).

  * REWARD ATTACHMENT -- reward is read at the states the measure lands on, so
        M^T r = E[ sum_{t>=0} gamma^t r(x_{t+1}) ],
    which relates to the ordinary current-state Q by  Q_std = r(x_0) + gamma*M^T r.

  * 1-gamma NORMALIZATION -- nu is a probability vector (rows sum to 1), so the
    readout's single 1/(1-gamma) restores total mass 1/(1-gamma).

  * TERMINALS -- `density_loss` applies NO terminal mask to the bootstrap leg, so
    the only self-consistent semantics is ABSORBING (self-loop) terminals, which
    is also the only way P^pi stays row-stochastic and nu stays a distribution.

  * IMMEDIATE KERNEL -- hard: k = one-hot at the row's own successor
    (`jnp.eye`); soft: k = `_delta_phi(nobs, u_raw)`, the eps-kernel softmax over
    the query pool (`eps_kernel` below is its numpy twin; the test suite
    cross-checks the twin against the real `_delta_phi` when jax is importable).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "policy_transition",
    "exact_successor_measure",
    "truncated_successor_measure",
    "normalized_measure",
    "measure_integral",
    "q_current_state",
    "eps_kernel",
    "density_td_fixed_point",
    "band_random_walk",
    "absorbing_mdp",
    "restrict_to_pool",
]


# ----------------------------------------------------------------- exact GT ---

def policy_transition(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """P^pi[s, x] = sum_a pi[s,a] P[s,a,x]  -- [S, S], row-stochastic."""
    return np.einsum("sa,sax->sx", np.asarray(pi, np.float64), np.asarray(P, np.float64))


def exact_successor_measure(P: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    """M[s,a,x] = sum_{t>=1} gamma^{t-1} Pr(x_t = x) = P[s,a,:] (I - gamma P^pi)^{-1}.

    THE ground truth: one matrix inverse, no estimator, no sampling.
    """
    P = np.asarray(P, np.float64)
    n = P.shape[0]
    resolvent = np.linalg.inv(np.eye(n) - gamma * policy_transition(P, pi))
    return P @ resolvent                                            # [S, A, S]


def truncated_successor_measure(P: np.ndarray, pi: np.ndarray, gamma: float,
                                horizon: int) -> np.ndarray:
    """Independent GT for the GT: the explicit series sum_{t=1..H} gamma^{t-1}
    P[s,a,:] (P^pi)^{t-1}. Guards `exact_successor_measure`'s algebra against
    my own off-by-one rather than trusting a single derivation."""
    P = np.asarray(P, np.float64)
    Ppi = policy_transition(P, pi)
    acc = np.zeros_like(P)
    step = P.copy()                                                  # t = 1 term
    for t in range(horizon):
        acc += (gamma ** t) * step
        step = step @ Ppi
    return acc


def normalized_measure(M: np.ndarray, gamma: float) -> np.ndarray:
    """nu = (1-gamma) M -- the object the softmax `w` represents."""
    return (1.0 - gamma) * np.asarray(M, np.float64)


def measure_integral(M: np.ndarray, r: np.ndarray) -> np.ndarray:
    """M^T r  [S, A] -- the quantity the PRIMARY readout is supposed to return."""
    return np.asarray(M, np.float64) @ np.asarray(r, np.float64)


def q_current_state(P: np.ndarray, pi: np.ndarray, gamma: float,
                    r: np.ndarray) -> np.ndarray:
    """The OTHER convention, kept explicit so the tests can show the readout is
    not it: Q_std(s,a) = E[sum_{t>=0} gamma^t r(x_t)] = r(s) + gamma * M^T r."""
    M = exact_successor_measure(P, pi, gamma)
    r = np.asarray(r, np.float64)
    return r[:, None] + gamma * measure_integral(M, r)


# --------------------------------------------------------- immediate kernel ---

def eps_kernel(x_rows: np.ndarray, u_pool: np.ndarray, eps) -> np.ndarray:
    """Numpy twin of `GCIQLFlowAgent._delta_phi`:
        k[i, j] = softmax_j( -0.5 * sum_c ((x_i,c - u_j,c) / eps_c)^2 )
    x_rows [B, d] raw coords, u_pool [P, d] raw pool coords, eps scalar or [d].
    Softmax normalized over the POOL (j), exactly as the shipped kernel is."""
    x = np.atleast_2d(np.asarray(x_rows, np.float64))
    u = np.atleast_2d(np.asarray(u_pool, np.float64))
    eps = np.asarray(eps, np.float64)
    d = (x[:, None, :] - u[None, :, :]) / eps
    logits = -0.5 * np.sum(d ** 2, axis=-1)
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


# ------------------------------------------------- the SHIPPED TD fixed point ---

def density_td_fixed_point(P: np.ndarray, pi: np.ndarray, gamma: float,
                           k_rows: np.ndarray, iters: int = 20000,
                           tol: float = 1e-15) -> np.ndarray:
    """Population version of `gciql_flow.density_loss`'s target, iterated to its
    fixed point:

        w(s,a) <- sum_x P[s,a,x] [ (1-gamma) k_rows[x]  +  gamma * sum_a' pi[x,a'] w(x,a') ]

    `k_rows[x]` is the immediate-term row for SUCCESSOR x: the identity for hard
    mode, the eps-kernel for soft mode. This is what makes the tests a check of
    the *shipped* target -- start index, (1-gamma) placement and all -- and not
    only of my closed form.  Returns w [S, A, P]."""
    P = np.asarray(P, np.float64)
    pi = np.asarray(pi, np.float64)
    k_rows = np.asarray(k_rows, np.float64)
    S, A = P.shape[0], P.shape[1]
    w = np.tile(k_rows.mean(0), (S, A, 1))
    for _ in range(iters):
        w_pi = np.einsum("sa,sap->sp", pi, w)                        # [S, P]
        nxt = np.einsum("sax,xp->sap", P, (1.0 - gamma) * k_rows + gamma * w_pi)
        delta = np.abs(nxt - w).max()
        w = nxt
        if delta < tol:
            break
    return w


# ------------------------------------------------------------- toy MDPs -----

def band_random_walk(n_states: int, lo: int, hi: int, n_actions: int = 2,
                     seed: int = 0) -> np.ndarray:
    """P [S, A, S]: inside the band [lo, hi] a random local walk on {s-1, s, s+1}
    (clipped into the band, action-dependent Dirichlet weights); outside the band
    every state self-loops so P stays row-stochastic. The band + buffer layout
    matters for the SOFT tests: every visited state has a complete, symmetric
    kernel neighbourhood inside the pool, which is what makes the kernel
    smoothing bias constant over the visited support."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        if not (lo <= s <= hi):
            P[s, :, s] = 1.0
            continue
        for a in range(n_actions):
            nbrs = [max(lo, s - 1), s, min(hi, s + 1)]
            wts = rng.dirichlet(np.ones(3) * 1.5)
            for nb, wt in zip(nbrs, wts):
                P[s, a, nb] += wt
    return P


def absorbing_mdp(n_states: int = 6, absorbing: int = 5, seed: int = 0,
                  n_actions: int = 2) -> np.ndarray:
    """P [S, A, S] with `absorbing` a true self-loop sink reachable from every
    transient state -- the ONLY terminal semantics consistent with a TD target
    that carries no terminal mask."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n_states, n_actions, n_states))
    P[absorbing, :, absorbing] = 1.0
    for s in range(n_states):
        if s == absorbing:
            continue
        for a in range(n_actions):
            row = rng.dirichlet(np.ones(n_states) * 0.7)
            P[s, a, :] = row
    return P


def restrict_to_pool(nu_row: np.ndarray, pool_idx) -> np.ndarray:
    """`w` on a sub-pool. A softmax restricted to a subset of its support and
    renormalized by THAT subset's sum is identical to the softmax computed over
    just that subset (the prefix property `eval_density_zeroshot.pool_sweep_
    primary` is built on), so a finite pool sees nu restricted and renormalized."""
    sub = np.asarray(nu_row, np.float64)[list(pool_idx)]
    return sub / sub.sum()
