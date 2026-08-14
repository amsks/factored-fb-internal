"""Spec 26 R0a-v2: fork-and-dwell synthetic, redesigned.

Contract: `PAPER/Multistep-FB/specs/26-bfm-roadmap.md`, section
"R0a verdict (2026-08-13) and the R0a-v2 redesign" (delta on 6d5c4cc, whose
v1 module `fork_dwell.py` is left untouched as the record of the falsification).

Why v2 exists
-------------
R0a-v1 failed GATE P1 *by instrument*: it injected duplicated ``s' == s`` rows,
which changes the EMPIRICAL KERNEL (P-hat(stay) inflates), so the exact
successor measure of the corrupted kernel drifts and every converged tabular arm
rides that same curve.  The thesis is an ESTIMATOR claim -- scale/capacity
pathology under function approximation and finite optimization -- so the
intervention has to hold the kernel fixed.  v2 changes four things:

1. **Multiplicity as a sampling weight, not row injection.**  The dataset rows
   are never touched; a per-row weight multiplies the *draw probability* of
   dwell rows by ``w in {1,2,5,10,20}``.

   INTERPRETATION (deviation from the literal words "stationary rows", flagged
   deliberately): the weight is keyed on the *state* being a dwell state (a
   state carrying the geometric self-loop), NOT on the transition satisfying
   ``s' == s``.  Reason: an importance weight applied only to ``s' == s`` rows
   is *algebraically identical to v1 row duplication* in expectation -- it
   reweights transitions *within* an ``(s,a)`` group and therefore reproduces
   exactly the kernel corruption v2 exists to remove.  Weighting every row of a
   dwell state by the same factor is the unique version that leaves
   ``P-hat(s'|s,a)`` pointwise unchanged (test:
   `test_empirical_kernel_byte_identical_across_weights`), so the tabular fixed
   point is invariant and multiplicity acts only through the *projection*
   distribution (state marginal / query pool) and the optimization path.  This
   is the same object R1's `measure_dup_passive` upweights (passive frames).

2. **Rank-constrained arms under a fixed budget.**  Scores are built from a
   feature map through a SHARED TRUNK (one hidden layer over phi(s)), so all
   regions compete for the same capacity, and the bilinear rank ``d`` (or the
   joint MLP's width) is the capacity knob: ``d in {2,4}`` against the ~10
   states / rank ~10 the exact measure needs.  Metrics are logged at
   1k/3k/10k steps, not only at convergence.

3. **Task hardening.**  v1 success was 1.0 in every cell.  v2 adds a DISTRACTOR
   fork arm whose region is *aliased* with the left region in feature space
   (identical coarse code, distinguished only by an ``alias_eps``-scaled
   identity component), and scores the decision by the MARGIN of the left action
   over the best competing action, normalized by the exact margin.

4. **A converged tabular column is kept as the predicted-null control**
   (one-hot features, full rank, 10x budget): it must show NO separation.

MDP (pinned, extending v1)
--------------------------
States: decision ``D``; left cells ``L_1..L_m``; right cells ``R_1..R_m``;
distractor cells ``X_1..X_m``; neutral ``N``.  Actions: left/right/distract/
neutral.  At ``D`` action k routes uniformly into region k's cells (neutral ->
``N``).  Inside any non-decision state: stay with ``p_stay = 1 - 1/T_dwell``,
else return to ``D`` (action-independent).  Horizon ``H=200``, ``gamma=0.99``,
uniform behavior policy.  ``m=3`` -> 11 states, so the exact ``nu`` matrix needs
rank ~10 and ``d in {2,4}`` genuinely under-parameterizes it.

Arms are unchanged from v1 (fb_unnorm, density_ce, density_mse, joint_unnorm,
joint_ce) and still share ONE query-pool + Bellman-target implementation --
`fork_dwell.loss_and_score_grad` is imported, not reimplemented.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.synthetic.fork_dwell import (  # noqa: E402
    Adam, exact_nu, loss_and_score_grad,
)

LEFT, RIGHT, DISTRACT, NEUTRAL = 0, 1, 2, 3
N_ACTIONS = 4

NORMALIZED_ARMS = ("density_ce", "density_mse", "joint_ce")
UNNORMALIZED_ARMS = ("fb_unnorm", "joint_unnorm")
ALL_ARMS = ("fb_unnorm", "density_ce", "density_mse", "joint_unnorm", "joint_ce")

# mode is shared with v1; model kind now also carries the capacity semantics.
ARM_MODE = {
    "fb_unnorm": "unnorm",
    "density_ce": "ce",
    "density_mse": "mse",
    "joint_unnorm": "unnorm",
    "joint_ce": "ce",
}
ARM_MODEL = {
    "fb_unnorm": "bilinear",
    "density_ce": "bilinear",
    "density_mse": "bilinear",
    "joint_unnorm": "mlp",
    "joint_ce": "mlp",
}

WEIGHTS = (1, 2, 5, 10, 20)
T_DWELLS = (5, 20, 44, 100)

# Columns: (feature mode, bilinear rank d, joint-MLP width, optimization budget).
# d2/d4 are the rank-constrained / finite-budget regime; `feat_full` isolates
# "aliased features" from "low rank"; `tab` is the converged tabular control
# that GATE P1-v2 predicts shows NO separation.
COLUMNS = {
    "d2":        dict(features="aliased", d=2,  hidden=2,  steps=10000),
    "d4":        dict(features="aliased", d=4,  hidden=4,  steps=10000),
    "d8":        dict(features="aliased", d=8,  hidden=8,  steps=10000),
    "feat_full": dict(features="aliased", d=16, hidden=64, steps=30000),
    "tab":       dict(features="onehot",  d=16, hidden=64, steps=100000),
}
# The gate reads ONLY the contracted d in {2,4}; d8 is an exploratory rung of the
# capacity ladder (reported, never part of the verdict, so it cannot inflate the
# multiple comparisons behind half A).
CONSTRAINED_COLUMNS = ("d2", "d4")
EXPLORATORY_COLUMNS = ("d8", "feat_full")
CONTROL_COLUMN = "tab"
# The contract's transients (1k/3k/10k) plus 30k; each column also logs its own
# final step.  The control's budget is 100k because at 30k its unnormalized
# joint arm is still improving -- an unconverged control is not the contracted
# control.  Its 30k reading stays logged so both are visible.
CHECKPOINTS = (1000, 3000, 10000, 30000)
CONTRACT_TRANSIENTS = (1000, 3000, 10000)


# ----------------------------------------------------------------------------
# MDP
# ----------------------------------------------------------------------------

class ForkDwellV2MDP:
    """Fork-and-dwell MDP with three fork arms, one of them feature-aliased."""

    def __init__(self, T_dwell: float, m: int = 3, gamma: float = 0.99,
                 H: int = 200, alias_eps: float = 0.1):
        assert T_dwell >= 1.0 and m >= 1
        self.T_dwell = float(T_dwell)
        self.m = int(m)
        self.gamma = float(gamma)
        self.H = int(H)
        self.alias_eps = float(alias_eps)

        self.n_states = 3 * m + 2
        self.D = 0
        self.left_cells = list(range(1, m + 1))
        self.right_cells = list(range(m + 1, 2 * m + 1))
        self.distract_cells = list(range(2 * m + 1, 3 * m + 1))
        self.N = 3 * m + 1
        self.p_stay = 1.0 - 1.0 / self.T_dwell

        P = np.zeros((N_ACTIONS, self.n_states, self.n_states))
        for act, cells in ((LEFT, self.left_cells), (RIGHT, self.right_cells),
                           (DISTRACT, self.distract_cells)):
            P[act, self.D, cells] = 1.0 / m
        P[NEUTRAL, self.D, self.N] = 1.0
        # every non-decision state dwells geometrically, action-independent
        self.dwell_states = np.zeros(self.n_states, dtype=bool)
        for s in self.left_cells + self.right_cells + self.distract_cells + [self.N]:
            self.dwell_states[s] = True
            for a in range(N_ACTIONS):
                P[a, s, s] = self.p_stay
                P[a, s, self.D] = 1.0 - self.p_stay
        assert np.allclose(P.sum(-1), 1.0)
        self.P = P
        self.phi_aliased = self._features(alias_eps)
        self.phi_onehot = np.eye(self.n_states)

    def _features(self, alias_eps: float) -> np.ndarray:
        """phi(s) = [coarse code ; alias_eps * onehot(s)].

        The coarse code is IDENTICAL for L_k and X_k (the aliased pair) and
        distinct for D, N and each R_k.  A model that cannot resolve the small
        identity component confuses the distractor arm with the left arm.
        """
        m = self.m
        coarse_dim = 2 + 2 * m   # D, N, m aliased L/X pairs, m right cells
        phi = np.zeros((self.n_states, coarse_dim + self.n_states))
        phi[self.D, 0] = 1.0
        phi[self.N, 1] = 1.0
        for k in range(m):
            phi[self.left_cells[k], 2 + k] = 1.0
            phi[self.distract_cells[k], 2 + k] = 1.0   # aliased with L_k
            phi[self.right_cells[k], 2 + m + k] = 1.0
        phi[:, coarse_dim:] = alias_eps * np.eye(self.n_states)
        return phi

    def features(self, mode: str) -> np.ndarray:
        return self.phi_aliased if mode == "aliased" else self.phi_onehot

    def uniform_policy(self) -> np.ndarray:
        return np.full((self.n_states, N_ACTIONS), 1.0 / N_ACTIONS)


# ----------------------------------------------------------------------------
# Dataset + the v2 intervention: per-row sampling WEIGHTS (rows untouched)
# ----------------------------------------------------------------------------

def generate_dataset(mdp: ForkDwellV2MDP, n_episodes: int, rng: np.random.Generator):
    """Uniform-policy episodes from D; returns (s, a, sp) int arrays."""
    ns = mdp.n_states
    s_list, a_list, sp_list = [], [], []
    for _ in range(n_episodes):
        s = mdp.D
        for _ in range(mdp.H):
            a = int(rng.integers(N_ACTIONS))
            sp = int(rng.choice(ns, p=mdp.P[a, s]))
            s_list.append(s)
            a_list.append(a)
            sp_list.append(sp)
            s = sp
    return (np.asarray(s_list, dtype=np.int64),
            np.asarray(a_list, dtype=np.int64),
            np.asarray(sp_list, dtype=np.int64))


def row_weights(s: np.ndarray, mdp: ForkDwellV2MDP, weight: float) -> np.ndarray:
    """Per-row sampling weight: `weight` on rows whose STATE is a dwell state.

    Keyed on the state, not on ``s' == s`` -- see the module docstring.  All
    rows of an ``(s,a)`` group share a weight, so the weighted empirical kernel
    equals the unweighted one exactly and the tabular fixed point is invariant.
    """
    assert weight >= 1.0
    return np.where(mdp.dwell_states[s], float(weight), 1.0)


def empirical_kernel(s, a, sp, n_states: int, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted empirical transition kernel P-hat[a, s, s'] (rows renormalized).

    Unvisited (s,a) rows are left as exact zeros so the comparison across
    weights is byte-for-byte.  With state-keyed integer weights the equality is
    exact, not approximate: numerator and denominator are both scaled by the
    same exactly-representable integer, and IEEE division is correctly rounded
    from the identical exact quotient.
    """
    if w is None:
        w = np.ones(len(s))
    K = np.zeros((N_ACTIONS, n_states, n_states))
    np.add.at(K, (a, s, sp), w)
    tot = K.sum(-1, keepdims=True)
    return np.divide(K, tot, out=np.zeros_like(K), where=tot > 0)


class WeightedSampler:
    """O(log N) weighted row sampler (precomputed CDF)."""

    def __init__(self, w: np.ndarray):
        p = np.asarray(w, dtype=float)
        self.cdf = np.cumsum(p / p.sum())
        self.cdf[-1] = 1.0
        self.n = len(p)

    def draw(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.searchsorted(self.cdf, rng.random(size), side="right").clip(0, self.n - 1)


def weighted_rho(sp: np.ndarray, w: np.ndarray, n_states: int) -> np.ndarray:
    """Query-pool marginal = weighted next-state distribution."""
    rho = np.bincount(sp, weights=w, minlength=n_states)
    return rho / rho.sum()


def bellman_batch_weighted(s, a, sp, sampler: WeightedSampler, batch_size: int,
                           rng: np.random.Generator):
    """Identical to v1's `bellman_batch` except rows are drawn from the weighted
    sampler.  Query pool = the batch's own successors, next actions uniform."""
    idx = sampler.draw(batch_size, rng)
    bs, ba, bsp = s[idx], a[idx], sp[idx]
    bap = rng.integers(N_ACTIONS, size=batch_size)
    return bs, ba, bsp, bap, bsp


# ----------------------------------------------------------------------------
# Score models: shared trunk over features, capacity knob = rank / width
# ----------------------------------------------------------------------------

class BilinearTrunkScore:
    """S[a,s,u] = F(s,a) . B(u) with F and B read off ONE shared ReLU trunk.

    Trunk: Z = relu(phi @ W1 + b1)  [n_states, h] -- shared by every state, so
    the regions compete for capacity.  Heads: F = (Z @ WF) -> [ns, A, d];
    B = Z @ WB -> [ns, d].  `d` is the rank knob.
    """

    def __init__(self, phi: np.ndarray, d: int, rng: np.random.Generator,
                 hidden: int = 32):
        self.phi = phi
        self.ns, in_dim = phi.shape
        self.d = int(d)
        self.h = int(hidden)
        self.params = {
            "W1": rng.standard_normal((in_dim, self.h)) * np.sqrt(2.0 / in_dim),
            "b1": np.zeros(self.h),
            "WF": rng.standard_normal((self.h, N_ACTIONS * self.d)) * np.sqrt(1.0 / self.h),
            "WB": rng.standard_normal((self.h, self.d)) * np.sqrt(1.0 / self.h),
        }

    def _trunk(self, params):
        pre = self.phi @ params["W1"] + params["b1"]
        return pre, np.maximum(pre, 0.0)

    def score(self, params: dict) -> np.ndarray:
        _, Z = self._trunk(params)
        F = (Z @ params["WF"]).reshape(self.ns, N_ACTIONS, self.d)
        B = Z @ params["WB"]
        return np.einsum("sad,ud->asu", F, B)

    def backward(self, params: dict, dS: np.ndarray) -> dict:
        pre, Z = self._trunk(params)
        F = (Z @ params["WF"]).reshape(self.ns, N_ACTIONS, self.d)
        B = Z @ params["WB"]
        dF = np.einsum("asu,ud->sad", dS, B).reshape(self.ns, N_ACTIONS * self.d)
        dB = np.einsum("asu,sad->ud", dS, F)
        dZ = dF @ params["WF"].T + dB @ params["WB"].T
        dpre = dZ * (pre > 0.0)
        return {
            "W1": self.phi.T @ dpre,
            "b1": dpre.sum(0),
            "WF": Z.T @ dF,
            "WB": Z.T @ dB,
        }


class JointTrunkScore:
    """S[a,s,u] = MLP([phi(s), onehot(a), phi(u)]); width is the capacity knob."""

    def __init__(self, phi: np.ndarray, hidden: int, rng: np.random.Generator):
        self.ns, fdim = phi.shape
        self.hidden = int(hidden)
        in_dim = 2 * fdim + N_ACTIONS
        aa, ss, uu = np.meshgrid(np.arange(N_ACTIONS), np.arange(self.ns),
                                 np.arange(self.ns), indexing="ij")
        X = np.zeros((N_ACTIONS * self.ns * self.ns, in_dim))
        X[:, :fdim] = phi[ss.ravel()]
        X[np.arange(X.shape[0]), fdim + aa.ravel()] = 1.0
        X[:, fdim + N_ACTIONS:] = phi[uu.ravel()]
        self.X = X
        self.params = {
            "W1": rng.standard_normal((in_dim, self.hidden)) * np.sqrt(2.0 / in_dim),
            "b1": np.zeros(self.hidden),
            "W2": rng.standard_normal((self.hidden, 1)) * 0.01,
            "b2": np.zeros(1),
        }

    def score(self, params: dict) -> np.ndarray:
        h = np.maximum(self.X @ params["W1"] + params["b1"], 0.0)
        out = h @ params["W2"] + params["b2"]
        return out.ravel().reshape(N_ACTIONS, self.ns, self.ns)

    def backward(self, params: dict, dS: np.ndarray) -> dict:
        pre = self.X @ params["W1"] + params["b1"]
        h = np.maximum(pre, 0.0)
        dout = dS.ravel()[:, None]
        dh = dout @ params["W2"].T
        dh[pre <= 0.0] = 0.0
        return {"W1": self.X.T @ dh, "b1": dh.sum(0),
                "W2": h.T @ dout, "b2": dout.sum(0)}


def make_model(arm: str, mdp: ForkDwellV2MDP, column: str, rng: np.random.Generator):
    spec = COLUMNS[column]
    phi = mdp.features(spec["features"])
    if ARM_MODEL[arm] == "bilinear":
        return BilinearTrunkScore(phi, spec["d"], rng)
    return JointTrunkScore(phi, spec["hidden"], rng)


# ----------------------------------------------------------------------------
# Training under a FIXED budget with transient checkpoints
# ----------------------------------------------------------------------------

def train_arm(arm: str, mdp: ForkDwellV2MDP, data, sampler: WeightedSampler,
              column: str, seed: int, batch_size: int = 128, lr: float = 5e-3,
              polyak: float = 0.05, checkpoints=CHECKPOINTS, log_every: int = 250,
              steps: int | None = None):
    """Train one arm; returns ({step: S}, loss_curve).

    The budget is the column's `steps` (`steps=` overrides it for smoke tests
    only); `checkpoints` (plus the final step) are the transient read-outs the
    contract asks for (1k/3k/10k, not convergence only).
    """
    steps = int(COLUMNS[column]["steps"] if steps is None else steps)
    mode = ARM_MODE[arm]
    rng = np.random.default_rng(seed)
    model = make_model(arm, mdp, column, rng)
    params = model.params
    target = {k: v.copy() for k, v in params.items()}
    opt = Adam(params, lr=lr)
    s, a, sp = data
    want = sorted({c for c in checkpoints if c <= steps} | {steps})
    snaps, curve = {}, []
    for t in range(1, steps + 1):
        bs, ba, bsp, bap, pool = bellman_batch_weighted(s, a, sp, sampler,
                                                        batch_size, rng)
        S_online = model.score(params)
        S_target = model.score(target)
        loss, dS = loss_and_score_grad(S_online, S_target, bs, ba, bsp, bap,
                                       pool, mdp.gamma, mode)
        grads = model.backward(params, dS)
        opt.step(params, grads)
        for k in target:
            target[k] = (1 - polyak) * target[k] + polyak * params[k]
        if t % log_every == 0 or t == 1:
            curve.append(float(loss))
        if t in want:
            snaps[t] = model.score(params)
    return snaps, curve


# ----------------------------------------------------------------------------
# Metrics (v1 metrics + the aliasing-hardened decision score)
# ----------------------------------------------------------------------------

def predicted_nu(S: np.ndarray, rho: np.ndarray, gamma: float, normalized: bool):
    """State-level predicted normalized measure (identical convention to v1)."""
    if normalized:
        z = S - S.max(axis=-1, keepdims=True)
        w = rho[None, None, :] * np.exp(z)
        return w / w.sum(axis=-1, keepdims=True)
    return (1.0 - gamma) * S * rho[None, None, :]


def compute_metrics(S: np.ndarray, arm: str, rho: np.ndarray,
                    mdp: ForkDwellV2MDP, nu_true: np.ndarray) -> dict:
    gamma = mdp.gamma
    normalized = arm in NORMALIZED_ARMS
    nu_hat = predicted_nu(S, rho, gamma, normalized)

    r = np.zeros(mdp.n_states)
    r[mdp.left_cells] = 1.0                      # left-favoring test reward
    Q = np.einsum("asu,u->as", nu_hat, r) / (1.0 - gamma)
    Q_true = np.einsum("asu,u->as", nu_true, r) / (1.0 - gamma)

    def margin(q):
        others = [q[a, mdp.D] for a in (RIGHT, DISTRACT, NEUTRAL)]
        return float(q[LEFT, mdp.D] - max(others))

    marg, marg_true = margin(Q), margin(Q_true)
    gap = float(Q[LEFT, mdp.D] - Q[RIGHT, mdp.D])
    gap_true = float(Q_true[LEFT, mdp.D] - Q_true[RIGHT, mdp.D])

    pos = np.clip(nu_hat, 0.0, None)
    pos = pos / np.clip(pos.sum(-1, keepdims=True), 1e-30, None)
    ent = -np.sum(pos * np.log(pos + 1e-30), axis=-1)

    nu_pi = nu_hat.mean(axis=0)
    T_nu = (1 - gamma) * mdp.P + gamma * np.einsum("asx,xu->asu", mdp.P, nu_pi)
    fp_residual = float(np.abs(nu_hat - T_nu).sum(-1).mean())
    mass = nu_hat.sum(-1)

    return {
        "arm": arm,
        "normalized": bool(normalized),
        "critic_scale_mean_abs": float(np.abs(S).mean()),
        "critic_scale_max_abs": float(np.abs(S).max()),
        "total_mass_mean": float(mass.mean()),
        "mass_ratio": float(mass.mean()),
        "entropy_mean": float(ent.mean()),
        "action_gap": gap,
        "action_gap_true": gap_true,
        "action_gap_ratio": float(gap / gap_true) if gap_true != 0 else float("nan"),
        "margin": marg,
        "margin_true": marg_true,
        "margin_ratio": float(marg / marg_true) if marg_true != 0 else float("nan"),
        "q_left_D": float(Q[LEFT, mdp.D]),
        "q_right_D": float(Q[RIGHT, mdp.D]),
        "q_distract_D": float(Q[DISTRACT, mdp.D]),
        "q_neutral_D": float(Q[NEUTRAL, mdp.D]),
        "alias_confusion": float(Q[DISTRACT, mdp.D] / Q[LEFT, mdp.D])
        if Q[LEFT, mdp.D] != 0 else float("nan"),
        # hardened: the left action must STRICTLY beat every rival (an exact tie
        # with the aliased distractor is a failure, not an argmax tie-break win)
        "success": float(marg > 0.0),
        "fixed_point_error": fp_residual,
        "gt_l1": float(np.abs(nu_hat - nu_true).sum(-1).mean()),
    }


# ----------------------------------------------------------------------------
# One cell
# ----------------------------------------------------------------------------

def run_cell(T_dwell: float, weight: float, seed: int, column: str,
             arms=ALL_ARMS, n_episodes: int = 300, batch_size: int = 128,
             lr: float = 5e-3, gamma: float = 0.99, H: int = 200, m: int = 3,
             alias_eps: float = 0.1, checkpoints=CHECKPOINTS,
             steps: int | None = None) -> dict:
    """Train every arm for one (T_dwell, weight, column, seed) cell.

    The dataset depends ONLY on (T_dwell, seed): the weight enters exclusively
    through the sampler and the query-pool marginal.
    """
    mdp = ForkDwellV2MDP(T_dwell, m=m, gamma=gamma, H=H, alias_eps=alias_eps)
    nu_true = exact_nu(mdp.P, gamma)
    data_rng = np.random.default_rng(10_000 + seed)
    s, a, sp = generate_dataset(mdp, n_episodes, data_rng)
    w = row_weights(s, mdp, weight)
    sampler = WeightedSampler(w)
    rho = weighted_rho(sp, w, mdp.n_states)

    out = {"T_dwell": T_dwell, "weight": weight, "seed": seed, "column": column,
           "n_transitions": int(len(s)),
           "dwell_row_frac": float(np.mean(mdp.dwell_states[s])),
           "dwell_sample_frac": float(w[mdp.dwell_states[s]].sum() / w.sum()),
           "rho": rho.tolist(), "arms": {}}
    for arm in arms:
        snaps, curve = train_arm(arm, mdp, (s, a, sp), sampler, column,
                                 seed=seed, batch_size=batch_size, lr=lr,
                                 checkpoints=checkpoints, steps=steps)
        per_step = {}
        for step, S in snaps.items():
            per_step[str(step)] = compute_metrics(S, arm, rho, mdp, nu_true)
        final = max(snaps)
        out["arms"][arm] = {"by_step": per_step, "final_step": final,
                            "loss_curve": curve, **per_step[str(final)]}
    return out


# ----------------------------------------------------------------------------
# Exact weighted tabular fixed point (used by tests to certify the null)
# ----------------------------------------------------------------------------

def exact_empirical_nu(s, a, sp, n_states: int, gamma: float,
                       w: np.ndarray | None = None) -> np.ndarray:
    """Exact successor measure of the WEIGHTED empirical kernel.

    With state-keyed weights this is byte-identical across weight factors --
    the analytic statement behind the predicted-null control column.
    """
    K = empirical_kernel(s, a, sp, n_states, w)
    return exact_nu(K, gamma)
