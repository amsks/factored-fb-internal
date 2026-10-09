"""The four-state control-bottleneck construction of the failure lemma / recovery proposition.

Base process (states 0,1,2,3, two actions each, discount gamma):
  * state 0: either action moves to state 1 w.p. 1/2 and to each of states 2, 3 w.p. 1/4;
  * state 1 is the control bottleneck: action a_+ (index 0) moves to state 2, a_- (index 1) to 3;
  * states 2 and 3 are absorbing; reward is 1 at state 2 and 0 elsewhere.
The continuation policy takes each action with probability 1/2.  So M(1, a_+, .) is a point mass at
the rewarded state 2 and M(1, a_-, .) a point mass at 3, an action-value difference of 1/(1-gamma)
that a rank-d successor fit must read from the two futures.

For feature dimension d the proof adds d-1 disconnected components, each an absorbing state plus four
other states whose two actions both lead to that absorbing state (ten state-action pairs sharing one
point mass).  Each added component contributes one large singular direction, so a rank-d SVD spends
d-1 of its directions on them and fits the base process at rank one, where the two bottleneck futures
collapse to the same row.  The softmax fit is not forced into this trade-off (see the proposition).

`make_mdp(n_components)` returns the process dict used by analysis/toy_ablation/family.py, so the same
fits, readouts and metrics apply unchanged.
"""
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "didactic"))
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation"))
import bottleneck_family_td as bf     # noqa: E402  (sets jax x64)
import family as fam                  # noqa: E402  (graph_distance, SOFT_SIGMA)

GAMMA = 0.8
DECISION = 1            # the control bottleneck: state 1
REWARD_STATE = 2        # a_+ (action 0) reaches it; a_- (action 1) reaches state 3
COMPONENT_SIZE = 5      # one absorbing state + four others


def component_states(j):
    """States of added component j (>=0): (absorbing, four others)."""
    base = 4 + COMPONENT_SIZE * j
    return base, [base + 1, base + 2, base + 3, base + 4]


def build_kernel(n_components):
    """(2n, n) transition kernel, row 2s+a a distribution over next states."""
    n = 4 + COMPONENT_SIZE * n_components
    K = np.zeros((2 * n, n))
    row = lambda s, a: 2 * s + a
    for a in (0, 1):                                  # state 0
        K[row(0, a), 1] = 0.5
        K[row(0, a), 2] = 0.25
        K[row(0, a), 3] = 0.25
    K[row(1, 0), REWARD_STATE] = 1.0                  # bottleneck: a_+ -> 2
    K[row(1, 1), 3] = 1.0                             #             a_- -> 3
    for s in (2, 3):                                  # absorbing
        for a in (0, 1):
            K[row(s, a), s] = 1.0
    for j in range(n_components):                     # added components
        absorbing, others = component_states(j)
        for a in (0, 1):
            K[row(absorbing, a), absorbing] = 1.0
        for o in others:
            for a in (0, 1):
                K[row(o, a), absorbing] = 1.0
    return K, n


def support_edges(kernel, n):
    """A representative (n, 2) next-state map (the two most-likely successors) so the family's
    graph-distance / soft-kernel helpers apply to the stochastic state 0 as well."""
    edges = np.zeros((n, 2), int)
    K = kernel.reshape(n, 2, n)
    for s in range(n):
        probs = K[s].sum(0)
        order = np.argsort(-probs)
        top = [int(t) for t in order if probs[int(t)] > 0][:2]
        while len(top) < 2:
            top.append(s)
        edges[s] = top
    return edges


def make_mdp(n_components, seed=None):
    """The family.py process dict for the base process plus `n_components` added components.
    `seed` labels the process (defaults to the feature dimension d = n_components + 1)."""
    kernel, n = build_kernel(n_components)
    pi = np.full((n, 2), 0.5)
    t_exact = bf.exact(kernel, pi, GAMMA)             # normalized measure (1-gamma) sum gamma^t P
    m_exact = t_exact / (1 - GAMMA)
    r = np.zeros(n)
    r[REWARD_STATE] = 1.0
    q = m_exact @ r
    gap = float(q[2 * DECISION] - q[2 * DECISION + 1])
    edges = support_edges(kernel, n)
    dist = fam.graph_distance(edges, n)
    k_soft = np.exp(-dist ** 2 / (2 * fam.SOFT_SIGMA ** 2))
    k_soft = k_soft / k_soft.sum(axis=1, keepdims=True)
    A = [REWARD_STATE]                                # "cluster A" is the single rewarded state
    B = [3]
    return dict(seed=(n_components + 1) if seed is None else seed, variant="four_state",
                n_components=n_components, n=n, edges=edges, pi=pi, kernel=kernel,
                t_exact=t_exact, m_exact=m_exact, r=r, q_exact=q, decision=DECISION,
                goal=REWARD_STATE, gap=gap, k_soft=k_soft, t_soft=t_exact @ k_soft,
                rho=np.full(n, 1.0 / n), groups=[[0], A, B], cuts=[DECISION], attempts=1)


def make_connected_mdp(entry=1, cluster=2, tail=6, seed=0):
    """A single connected shared-tail bottleneck MDP (the physically faithful version): an entry chain,
    one bottleneck b whose two actions enter clusters A and B, and a shared closed tail every cluster
    state exits into.  The two futures overlap in the tail and differ only on the brief transient
    clusters, so the action difference is a low-mass direction the tail dominates.  Reward is the
    indicator of cluster A.  Built with analysis/lemma_check/design_single_bottleneck.build, wrapped as
    the same family.py process dict.  Everything is in one figure -- no disconnected states."""
    import sys as _sys
    _sys.path.insert(0, str(REPO / "analysis" / "lemma_check"))
    import design_single_bottleneck as dsb     # noqa: E402

    edges, b, A, B, T = dsb.build(entry=entry, cluster=cluster, tail=tail, merge=True)
    n = len(edges)
    pi = np.full((n, 2), 0.5)
    kernel = np.eye(n)[edges].reshape(-1, n)
    t_exact = bf.exact(kernel, pi, GAMMA)
    m_exact = t_exact / (1 - GAMMA)
    r = np.isin(np.arange(n), A).astype(float)          # reward = indicator of cluster A
    q = m_exact @ r
    gap = float(q[2 * b] - q[2 * b + 1])
    dist = fam.graph_distance(edges, n)
    k_soft = np.exp(-dist ** 2 / (2 * fam.SOFT_SIGMA ** 2))
    k_soft = k_soft / k_soft.sum(axis=1, keepdims=True)
    return dict(seed=seed, variant="connected", n=n, edges=edges, pi=pi, kernel=kernel,
                t_exact=t_exact, m_exact=m_exact, r=r, q_exact=q, decision=b, goal=A[0], gap=gap,
                k_soft=k_soft, t_soft=t_exact @ k_soft, rho=np.full(n, 1.0 / n),
                groups=[list(range(b)), list(A), list(B), list(T)], cuts=[b], attempts=1)


if __name__ == "__main__":
    for k in range(0, 4):
        m = make_mdp(k)
        print(f"components={k}  n={m['n']:>2}  gap={m['gap']:.3f}  "
              f"q(a+,a-)=({m['q_exact'][2]:.2f},{m['q_exact'][3]:.2f})")
    c = make_connected_mdp()
    print(f"connected n={c['n']}  gap={c['gap']:.3f}  A={c['groups'][1]} B={c['groups'][2]} tail={c['groups'][3]}")
