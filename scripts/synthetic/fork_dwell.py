"""Spec 26 R0a: fork-and-dwell synthetic (contract: specs/26-bfm-roadmap.md section 1).

Standalone numpy implementation -- NO dependency on the jax/torch agent stack.

MDP (pinned): states = {decision D, left region L1..Lm, right region R1..Rm,
neutral N}. At D: action left -> L1, action right -> R1, action neutral -> N.
Inside any region cell (L*, R*, N): geometric dwell -- stay in the SAME cell
with prob p_stay = 1 - 1/T_dwell, exit to D otherwise (all actions identical
inside regions). Episode horizon H=200, gamma=0.99, uniform behavior policy.
Note: with the pinned dynamics only cell 1 of each region is reachable, so the
default is m=1 (the m parameter is kept for future variants).

Dataset duplication knob: replicate stationary transitions (s' == s, which by
construction happen only in region cells) x{1,2,5,10,20}. This is a collection
artifact; the true MDP (and hence exact ground truth) is unchanged.

Ground truth (exact, matrix inversion): nu(s,a,.) = (1-gamma) * P_a (I - gamma
P_pi)^{-1}, the normalized successor measure over NEXT states (nu = (1-gamma)M,
M counted from s_{t+1}); rows sum to 1 exactly.

Five arms sharing ONE query-pool + Bellman-target implementation
(`bellman_batch` -> `loss_and_score_grad`):
  fb_unnorm    -- bilinear F(s,a)^T B(u), FB squared+linear measure loss
                  (unnormalized; the tiny-ortho variant is omitted -- optional
                  in the contract).
  density_ce   -- bilinear logits, softmax over the query pool, target
                  p_tgt = (1-gamma) e_i + gamma p_bar_plus, cross-entropy.
  density_mse  -- same normalized prediction and target, squared-error fit
                  (the CE-vs-normalization deconfound).
  joint_unnorm -- MLP([s,a,u]) scalar score, FB squared+linear loss.
  joint_ce     -- MLP scores, softmax over pool, CE (normalized joint).

Per-cell metrics (pinned): critic scale, total predicted mass vs truth,
entropy, action gap at D (Q(left)-Q(right) under a left-favoring test reward),
greedy-policy success, fixed-point error (Bellman residual under the TRUE
dynamics) plus L1 error to exact ground truth.
"""

from __future__ import annotations

import numpy as np

LEFT, RIGHT, NEUTRAL = 0, 1, 2
N_ACTIONS = 3

NORMALIZED_ARMS = ("density_ce", "density_mse", "joint_ce")
UNNORMALIZED_ARMS = ("fb_unnorm", "joint_unnorm")
ALL_ARMS = ("fb_unnorm", "density_ce", "density_mse", "joint_unnorm", "joint_ce")


# ----------------------------------------------------------------------------
# MDP
# ----------------------------------------------------------------------------

class ForkDwellMDP:
    """Fork-and-dwell MDP with geometric region dwell."""

    def __init__(self, T_dwell: float, m: int = 1, gamma: float = 0.99, H: int = 200):
        assert T_dwell >= 1.0
        self.T_dwell = float(T_dwell)
        self.m = int(m)
        self.gamma = float(gamma)
        self.H = int(H)
        self.n_states = 2 * m + 2
        self.D = 0
        self.left_cells = list(range(1, m + 1))
        self.right_cells = list(range(m + 1, 2 * m + 1))
        self.N = 2 * m + 1
        p_stay = 1.0 - 1.0 / self.T_dwell
        self.p_stay = p_stay

        P = np.zeros((N_ACTIONS, self.n_states, self.n_states))
        # decision state routing
        P[LEFT, self.D, self.left_cells[0]] = 1.0
        P[RIGHT, self.D, self.right_cells[0]] = 1.0
        P[NEUTRAL, self.D, self.N] = 1.0
        # region cells: geometric dwell, action-independent
        for s in self.left_cells + self.right_cells + [self.N]:
            for a in range(N_ACTIONS):
                P[a, s, s] = p_stay
                P[a, s, self.D] = 1.0 - p_stay
        assert np.allclose(P.sum(-1), 1.0)
        self.P = P

    def uniform_policy(self) -> np.ndarray:
        return np.full((self.n_states, N_ACTIONS), 1.0 / N_ACTIONS)


def exact_nu(P: np.ndarray, gamma: float, pi: np.ndarray | None = None) -> np.ndarray:
    """Exact normalized successor measure nu(s,a,u) = (1-gamma)*M(s,a,u).

    M counts occupancy of s_{t+1}, s_{t+2}, ... (FB convention); each
    nu(s,a,.) row sums to 1 exactly. pi defaults to uniform.
    """
    na, ns, _ = P.shape
    if pi is None:
        pi = np.full((ns, na), 1.0 / na)
    P_pi = np.einsum("sa,asu->su", pi, P)
    inv = np.linalg.inv(np.eye(ns) - gamma * P_pi)
    return (1.0 - gamma) * np.einsum("asx,xu->asu", P, inv)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

def generate_dataset(mdp: ForkDwellMDP, n_episodes: int, rng: np.random.Generator):
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


def duplicate_stationary(s: np.ndarray, a: np.ndarray, sp: np.ndarray, dup: int):
    """Replicate stationary transitions (s'==s; only region cells) x dup."""
    assert dup >= 1
    if dup == 1:
        return s, a, sp
    stat = s == sp
    reps = np.where(stat, dup, 1)
    return np.repeat(s, reps), np.repeat(a, reps), np.repeat(sp, reps)


# ----------------------------------------------------------------------------
# Score models (tabular-scale)
# ----------------------------------------------------------------------------

class BilinearScore:
    """FB parameterization: S[a,s,u] = F[s,a] . B[u]."""

    def __init__(self, n_states: int, d: int, rng: np.random.Generator,
                 init_scale: float = 0.1):
        self.n_states = n_states
        self.d = d
        self.params = {
            "F": init_scale * rng.standard_normal((n_states, N_ACTIONS, d)),
            "B": init_scale * rng.standard_normal((n_states, d)),
        }

    @staticmethod
    def score(params: dict) -> np.ndarray:
        return np.einsum("sad,ud->asu", params["F"], params["B"])

    @staticmethod
    def backward(params: dict, dS: np.ndarray) -> dict:
        return {
            "F": np.einsum("asu,ud->sad", dS, params["B"]),
            "B": np.einsum("asu,sad->ud", dS, params["F"]),
        }


class MLPScore:
    """Joint critic: S[a,s,u] = MLP([onehot(s), onehot(a), onehot(u)]).

    Evaluated on the full (a,s,u) grid (tabular-scale), gradients exact.
    """

    def __init__(self, n_states: int, hidden: int, rng: np.random.Generator):
        self.n_states = n_states
        self.hidden = hidden
        in_dim = 2 * n_states + N_ACTIONS
        # all (a,s,u) combos as one-hot rows, ordered to match S[a,s,u].ravel()
        aa, ss, uu = np.meshgrid(np.arange(N_ACTIONS), np.arange(n_states),
                                 np.arange(n_states), indexing="ij")
        X = np.zeros((N_ACTIONS * n_states * n_states, in_dim))
        rows = np.arange(X.shape[0])
        X[rows, ss.ravel()] = 1.0
        X[rows, n_states + aa.ravel()] = 1.0
        X[rows, n_states + N_ACTIONS + uu.ravel()] = 1.0
        self.X = X
        self.params = {
            "W1": rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim),
            "b1": np.zeros(hidden),
            "W2": rng.standard_normal((hidden, 1)) * 0.01,
            "b2": np.zeros(1),
        }

    def score(self, params: dict) -> np.ndarray:
        h = np.maximum(self.X @ params["W1"] + params["b1"], 0.0)
        out = h @ params["W2"] + params["b2"]
        return out.ravel().reshape(N_ACTIONS, self.n_states, self.n_states)

    def backward(self, params: dict, dS: np.ndarray) -> dict:
        pre = self.X @ params["W1"] + params["b1"]
        h = np.maximum(pre, 0.0)
        dout = dS.ravel()[:, None]
        dh = dout @ params["W2"].T
        dh[pre <= 0.0] = 0.0
        return {
            "W1": self.X.T @ dh,
            "b1": dh.sum(0),
            "W2": h.T @ dout,
            "b2": dout.sum(0),
        }


class Adam:
    def __init__(self, params: dict, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ----------------------------------------------------------------------------
# Shared query-pool + Bellman-target implementation (used by ALL five arms)
# ----------------------------------------------------------------------------

def bellman_batch(s, a, sp, batch_size: int, rng: np.random.Generator):
    """Sample one training batch. The query pool is the batch's own successor
    states (u_j = sp_j), so the index-level one-hot e_i is the identity column
    i, and next actions a' are drawn from the uniform policy. Identical for
    every arm."""
    idx = rng.integers(len(s), size=batch_size)
    bs, ba, bsp = s[idx], a[idx], sp[idx]
    bap = rng.integers(N_ACTIONS, size=batch_size)
    pool = bsp  # query pool ~ replay next-state marginal (duplication included)
    return bs, ba, bsp, bap, pool


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_score_grad(S_online: np.ndarray, S_target: np.ndarray,
                        bs, ba, bsp, bap, pool, gamma: float, mode: str):
    """Loss and gradient w.r.t. the full online score tensor S[a,s,u].

    mode: 'unnorm' -> FB squared+linear measure loss on raw scores;
          'ce'     -> softmax over pool, cross-entropy to
                      (1-gamma) e_i + gamma p_bar_plus;
          'mse'    -> same prediction/target, squared error on probabilities.
    Every arm consumes the SAME batch/pool/target-network ingredients.
    """
    B = len(bs)
    scores = S_online[ba[:, None], bs[:, None], pool[None, :]]     # [B, K]
    tgt_scores = S_target[bap[:, None], bsp[:, None], pool[None, :]]
    dS = np.zeros_like(S_online)

    if mode == "unnorm":
        td = scores - gamma * tgt_scores
        loss = 0.5 * np.mean(td ** 2) - np.mean(np.diag(scores))
        g = td / td.size
        g[np.arange(B), np.arange(B)] -= 1.0 / B
    else:
        p = _softmax(scores)
        p_plus = _softmax(tgt_scores)
        p_tgt = gamma * p_plus
        p_tgt[np.arange(B), np.arange(B)] += 1.0 - gamma  # e_i at own successor
        if mode == "ce":
            loss = -np.mean(np.sum(p_tgt * np.log(p + 1e-30), axis=1))
            g = (p - p_tgt) / B
        elif mode == "mse":
            diff = p - p_tgt
            loss = np.mean(np.sum(diff ** 2, axis=1))
            gp = 2.0 * diff / B
            g = p * (gp - np.sum(gp * p, axis=1, keepdims=True))
        else:
            raise ValueError(mode)

    np.add.at(dS,
              (np.broadcast_to(ba[:, None], g.shape).ravel(),
               np.broadcast_to(bs[:, None], g.shape).ravel(),
               np.broadcast_to(pool[None, :], g.shape).ravel()),
              g.ravel())
    return loss, dS


ARM_SPECS = {
    "fb_unnorm": dict(model="bilinear", mode="unnorm"),
    "density_ce": dict(model="bilinear", mode="ce"),
    "density_mse": dict(model="bilinear", mode="mse"),
    "joint_unnorm": dict(model="mlp", mode="unnorm"),
    "joint_ce": dict(model="mlp", mode="ce"),
}


def make_model(arm: str, n_states: int, rng: np.random.Generator,
               d: int = 16, hidden: int = 64):
    kind = ARM_SPECS[arm]["model"]
    if kind == "bilinear":
        return BilinearScore(n_states, d, rng)
    return MLPScore(n_states, hidden, rng)


def train_arm(arm: str, mdp: ForkDwellMDP, data, seed: int,
              steps: int = 10000, batch_size: int = 128, lr: float = 5e-3,
              polyak: float = 0.05, log_every: int = 250):
    """Train one arm on (possibly duplicated) data; returns (S, loss_curve)."""
    mode = ARM_SPECS[arm]["mode"]
    rng = np.random.default_rng(seed)
    model = make_model(arm, mdp.n_states, rng)
    params = model.params
    target = {k: v.copy() for k, v in params.items()}
    opt = Adam(params, lr=lr)
    s, a, sp = data
    curve = []
    for t in range(steps):
        bs, ba, bsp, bap, pool = bellman_batch(s, a, sp, batch_size, rng)
        S_online = model.score(params)
        S_target = model.score(target)
        loss, dS = loss_and_score_grad(S_online, S_target, bs, ba, bsp, bap,
                                       pool, mdp.gamma, mode)
        grads = model.backward(params, dS)
        opt.step(params, grads)
        for k in target:
            target[k] = (1 - polyak) * target[k] + polyak * params[k]
        if t % log_every == 0 or t == steps - 1:
            curve.append(float(loss))
    return model.score(params), curve


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def empirical_rho(sp: np.ndarray, n_states: int) -> np.ndarray:
    """Pool marginal = replay next-state distribution (duplication included)."""
    return np.bincount(sp, minlength=n_states) / len(sp)


def predicted_nu(S: np.ndarray, rho: np.ndarray, gamma: float, normalized: bool):
    """State-level predicted normalized measure nu_hat[a,s,u].

    Normalized arms: population limit of the pool softmax,
      nu_hat(u|s,a) proportional to rho(u) * exp(S[a,s,u])  (sums to 1).
    Unnormalized arms: S is a density w.r.t. rho in M units, so
      nu_hat = (1-gamma) * S * rho  (mass unconstrained, may be negative).
    """
    if normalized:
        z = S - S.max(axis=-1, keepdims=True)
        w = rho[None, None, :] * np.exp(z)
        return w / w.sum(axis=-1, keepdims=True)
    return (1.0 - gamma) * S * rho[None, None, :]


def compute_metrics(S: np.ndarray, arm: str, rho: np.ndarray,
                    mdp: ForkDwellMDP, nu_true: np.ndarray) -> dict:
    gamma = mdp.gamma
    normalized = arm in NORMALIZED_ARMS
    nu_hat = predicted_nu(S, rho, gamma, normalized)
    ns = mdp.n_states

    # left-favoring test reward (pinned): 1 on left-region cells, else 0
    r = np.zeros(ns)
    r[mdp.left_cells] = 1.0

    # Q in M units (1/(1-gamma) horizon)
    Q = np.einsum("asu,u->as", nu_hat, r) / (1.0 - gamma)
    Q_true = np.einsum("asu,u->as", nu_true, r) / (1.0 - gamma)
    gap = float(Q[LEFT, mdp.D] - Q[RIGHT, mdp.D])
    gap_true = float(Q_true[LEFT, mdp.D] - Q_true[RIGHT, mdp.D])

    # entropy of the predicted future distribution (clip for unnorm arms)
    pos = np.clip(nu_hat, 0.0, None)
    pos = pos / np.clip(pos.sum(-1, keepdims=True), 1e-30, None)
    ent = -np.sum(pos * np.log(pos + 1e-30), axis=-1)

    # fixed-point (Bellman) residual under the TRUE dynamics, uniform pi
    nu_pi = nu_hat.mean(axis=0)  # [s, u]
    T_nu = (1 - gamma) * mdp.P + gamma * np.einsum("asx,xu->asu", mdp.P, nu_pi)
    fp_residual = float(np.abs(nu_hat - T_nu).sum(-1).mean())

    mass = nu_hat.sum(-1)  # per (a,s), truth = 1 in nu units
    return {
        "arm": arm,
        "normalized": bool(normalized),
        "critic_scale_mean_abs": float(np.abs(S).mean()),
        "critic_scale_max_abs": float(np.abs(S).max()),
        "total_mass_mean": float(mass.mean()),
        "total_mass_max": float(mass.max()),
        "mass_ratio": float(mass.mean()),  # truth = 1.0 exactly
        "entropy_mean": float(ent.mean()),
        "entropy_D_left": float(ent[LEFT, mdp.D]),
        "action_gap": gap,
        "action_gap_true": gap_true,
        "action_gap_ratio": float(gap / gap_true) if gap_true != 0 else float("nan"),
        "q_left_D": float(Q[LEFT, mdp.D]),
        "q_right_D": float(Q[RIGHT, mdp.D]),
        "q_neutral_D": float(Q[NEUTRAL, mdp.D]),
        "success": float(int(np.argmax(Q[:, mdp.D]) == LEFT)),
        "fixed_point_error": fp_residual,
        "gt_l1": float(np.abs(nu_hat - nu_true).sum(-1).mean()),
    }


def run_cell(T_dwell: float, dup: int, seed: int, arms=ALL_ARMS,
             n_episodes: int = 300, steps: int = 10000, batch_size: int = 128,
             lr: float = 5e-3, gamma: float = 0.99, H: int = 200) -> dict:
    """Train all arms for one (T_dwell, duplication, seed) cell."""
    mdp = ForkDwellMDP(T_dwell, gamma=gamma, H=H)
    nu_true = exact_nu(mdp.P, gamma)
    data_rng = np.random.default_rng(10_000 + seed)
    s, a, sp = generate_dataset(mdp, n_episodes, data_rng)
    s_d, a_d, sp_d = duplicate_stationary(s, a, sp, dup)
    rho = empirical_rho(sp_d, mdp.n_states)
    out = {"T_dwell": T_dwell, "dup": dup, "seed": seed,
           "n_transitions": int(len(s_d)),
           "stationary_frac": float(np.mean(s_d == sp_d)),
           "rho": rho.tolist(), "arms": {}}
    for arm in arms:
        S, curve = train_arm(arm, mdp, (s_d, a_d, sp_d), seed=seed,
                             steps=steps, batch_size=batch_size, lr=lr)
        m = compute_metrics(S, arm, rho, mdp, nu_true)
        m["loss_curve"] = curve
        out["arms"][arm] = m
    return out
