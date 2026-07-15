"""A^m_s = Q_m(s,a,g) - E_{mu}[Q_m(s,.,g)], then / ||.||_F (one positive global scalar).

Centering removes the per-goal state offset (leaving the action-relevant part); Frobenius
standardization makes cross-critic magnitudes comparable without changing argmax, singular
vectors, action rankings, or modal topology (spec section 2/10).
"""
import numpy as np


def center_and_standardize(Q, Q_mu_center):
    """Q [n_a, n_g]; Q_mu_center [n_mu, n_g] evaluated at the SAME goals.
    Subtract the per-goal mu mean, divide by the Frobenius norm."""
    center = np.asarray(Q_mu_center).mean(axis=0)          # [n_g]
    A = np.asarray(Q, np.float64) - center[None, :]
    nrm = np.linalg.norm(A)
    return A / (nrm + 1e-12)


def build_atlas(critic, banks, action_set='mu', goal_cat='eval'):
    """Evaluate A^m_s for every (phase, state) in the banks.

    action_set: 'mu' (B_mu only -> semantic atlas) or 'all' (B_mu + perturb + global).
    goal_cat:   'eval' | 'hindsight' | 'grid'.
    Returns dict(A=object[P,K] of [n_a,n_g] float64, goals=[n_g,28], action_set, goal_cat).
    """
    goals = banks[{'eval': 'goals_eval', 'hindsight': 'goals_hindsight',
                   'grid': 'goals_grid'}[goal_cat]]
    P, K = banks['state_obs'].shape[:2]
    A = np.empty((P, K), dtype=object)
    for p in range(P):
        for k in range(K):
            obs = banks['state_obs'][p, k][None]              # [1,28]
            if action_set == 'mu':
                acts = banks['A_mu'][p, k][None]              # [1,N_MU,5]
            else:
                acts = np.concatenate([banks['A_mu'][p, k], banks['A_perturb'][p, k],
                                       banks['A_global']], axis=0)[None]
            cen = banks['mu_center_samples'][p, k][None]      # [1,N_MU,5]
            Q = critic.Q(obs, acts, goals)[0]                 # [n_a, n_g]
            Qmu = critic.Q(obs, cen, goals)[0]                # [N_MU, n_g]
            A[p, k] = center_and_standardize(Q, Qmu)
    return dict(A=A, goals=goals, action_set=action_set, goal_cat=goal_cat)
