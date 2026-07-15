"""Oracle-FREE atlas analysis: what structure does each critic's action-goal preference
operator A^m_s(a,g) have, and how does bilinear differ from MLP? Pure critic forward passes
(no simulator) -- answers 'what is bilinear doing' descriptively; the oracle only validates.

Metrics per critic (aggregated over states, on the 64-goal grid so rank is not n_g-capped):
  - singular spectrum + EFFECTIVE RANK (entropy of normalized singular values)
  - top-sv share (mode concentration)
Cross-critic (bilinear vs MLP, matched states):
  - PRINCIPAL ANGLES between the top-k goal-mode subspaces span(V_k)
  - multi-modality: connected components in the top-q% of |A(.,g)| on a shared kNN action graph
"""
import numpy as np

from scripts.atlas import config as C


def effective_rank(A):
    """Entropy-based effective rank: exp(-sum p_i log p_i), p_i = s_i/sum(s). In [1, min(shape)]."""
    s = np.linalg.svd(np.asarray(A, np.float64), compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def top_sv_share(A):
    s = np.linalg.svd(np.asarray(A, np.float64), compute_uv=False)
    return float(s[0] / (s.sum() + 1e-12))


def goal_mode_subspace(A, k):
    """Top-k right singular vectors (goal modes) as columns [n_g, k]."""
    _, _, Vt = np.linalg.svd(np.asarray(A, np.float64), full_matrices=False)
    return Vt[:k].T


def principal_angles(A1, A2, k=3):
    """Principal angles (radians) between the top-k goal-mode subspaces of A1 and A2.
    Small angles => same goal-preference structure; large => genuinely different operators."""
    Q1 = np.linalg.qr(goal_mode_subspace(A1, k))[0]
    Q2 = np.linalg.qr(goal_mode_subspace(A2, k))[0]
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def basin_count(A_col, actions, k=None, q=None):
    """Number of separated high-value action basins for one goal column A(.,g).
    Connected components among the top-q% actions on a kNN action graph."""
    k = k or C.KNN_K
    q = q or C.TOPQ
    a = np.asarray(A_col, np.float64)
    n = len(a)
    thr = np.quantile(a, 1 - q)
    top = np.where(a >= thr)[0]
    if len(top) < 2:
        return len(top)
    acts = np.asarray(actions)
    # kNN adjacency among the top actions (Euclidean in the 5-D action space)
    sub = acts[top]
    D = np.linalg.norm(sub[:, None] - sub[None], axis=-1)
    adj = {i: set() for i in range(len(top))}
    for i in range(len(top)):
        nn = np.argsort(D[i])[1:k + 1]
        for j in nn:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    seen, comps = set(), 0
    for i in range(len(top)):
        if i in seen:
            continue
        comps += 1
        stack = [i]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
    return comps
