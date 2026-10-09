"""
FourRooms tabular demonstration: FB's squared-loss rank-d successor-measure fit
vs a normalized (softmax / cross-entropy) rank-d fit, at a sticky bottleneck.

Everything here is closed-form / exact except the two rank-d fits:
  - FB fit: population optimum of a rho(s')-weighted squared loss on the ratio
    R(s,a,s') = M(s,a,s') / rho(s'). Closed form: truncated SVD of R diag(sqrt(rho)).
  - Density fit: rank-d softmax model trained by cross-entropy against the true
    conditional distribution (1-gamma) M(s,a,.), via Adam (optax).

Run:
    JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/fourrooms_estimator/run.py

Outputs (all under this directory):
    gap.csv        - main k x d x gamma sweep at the door bottleneck D
    occupancy.csv  - directed-policy occupancy experiment (Figure-1 analogue)
    run_log.txt    - console log + sanity-check summary
and the paper figure under PAPER/ICLR/figures/fig_fourrooms_estimator.{pdf,png}.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import csv
import functools
import pathlib
import sys
import time
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE.parent.parent / "PAPER" / "ICLR" / "figures"
LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(str(msg))


# ----------------------------------------------------------------------------
# 1. Environment: 11x11 FourRooms.
# ----------------------------------------------------------------------------
GRID_N = 11
ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
A = len(ACTIONS)

# Interior "+" wall at row 5 / col 5, punched with four doorways, one per arm.
DOORWAYS = {
    "topleft_topright": (2, 5),   # connects top-left room  <-> top-right room
    "botleft_botright": (8, 5),   # connects bottom-left    <-> bottom-right
    "topleft_botleft": (5, 2),    # connects top-left       <-> bottom-left
    "topright_botright": (5, 8),  # connects top-right      <-> bottom-right
}


def build_grid():
    wall = np.zeros((GRID_N, GRID_N), dtype=bool)
    wall[5, :] = True
    wall[:, 5] = True
    for rc in DOORWAYS.values():
        wall[rc] = False
    free_cells = [(r, c) for r in range(GRID_N) for c in range(GRID_N) if not wall[r, c]]
    cell_to_idx = {rc: i for i, rc in enumerate(free_cells)}
    return wall, free_cells, cell_to_idx


WALL, FREE_CELLS, CELL_TO_IDX = build_grid()
S = len(FREE_CELLS)
assert S == 104, f"expected 104 free cells (11x11 with a 4-doorway '+' wall), got {S}"


def in_room(rc, rows, cols):
    r, c = rc
    return r in rows and c in cols


ROOMS = {
    "top_left": (range(0, 5), range(0, 5)),
    "top_right": (range(0, 5), range(6, 11)),
    "bottom_left": (range(6, 11), range(0, 5)),
    "bottom_right": (range(6, 11), range(6, 11)),
}
NEXT_ROOM = "top_right"  # room across the D doorway


def step_normal(rc, a):
    """k=1 (non-sticky) deterministic dynamics: bump into a wall/edge = stay."""
    r, c = rc
    dr, dc = DELTA[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_N and 0 <= nc < GRID_N and not WALL[nr, nc]:
        return (nr, nc)
    return (r, c)


# Per-action deterministic (k=1) transition matrices, shape (A, S, S).
P0 = np.zeros((A, S, S))
for ai, a in enumerate(ACTIONS):
    for i, rc in enumerate(FREE_CELLS):
        j = CELL_TO_IDX[step_normal(rc, a)]
        P0[ai, i, j] = 1.0

D_RC = DOORWAYS["topleft_topright"]
D = CELL_TO_IDX[D_RC]
A_CROSS = "right"          # action that leaves D into the top-right room
A_FAIL = ["up", "down", "left"]
assert step_normal(D_RC, A_CROSS) != D_RC, "a_cross must actually leave D"

G_RC = (4, 10)              # target cell "deep" in the next room (far corner from D)
G = CELL_TO_IDX[G_RC]
assert in_room(G_RC, *ROOMS[NEXT_ROOM])

S0_RC = (2, 2)               # start state: 3 cells before the door, same corridor row
S0 = CELL_TO_IDX[S0_RC]


def room_mask(room_name):
    rows, cols = ROOMS[room_name]
    m = np.zeros(S, dtype=bool)
    for i, rc in enumerate(FREE_CELLS):
        if in_room(rc, rows, cols):
            m[i] = True
    return m


NEXT_ROOM_MASK = room_mask(NEXT_ROOM)

R_ROOM = NEXT_ROOM_MASK.astype(np.float64)          # reward = 1 on every cell of next room
R_TARGET = np.zeros(S)
R_TARGET[G] = 1.0                                    # reward = 1 only on the deep target cell


# ----------------------------------------------------------------------------
# 1b. Continuous 2D action space (15x15 = 225 discretized actions), for the
#     action-value heatmap experiment ("action_value_door"). Same grid, same
#     bottleneck D, but a in [-1,1]^2 replaces the 4 discrete actions.
# ----------------------------------------------------------------------------
AV_GRID_N = 15
_av_xs = np.linspace(-1, 1, AV_GRID_N)
_av_ys = np.linspace(1, -1, AV_GRID_N)     # row 0 = +1 ("up"), so imshow puts +y at the top
_av_gy, _av_gx = np.meshgrid(_av_ys, _av_xs, indexing="ij")
AV_ACTIONS = np.stack([_av_gx.ravel(), _av_gy.ravel()], axis=-1)   # (225,2), row-major i*15+j
AV_NA = AV_ACTIONS.shape[0]
assert AV_NA == 225


def _av_direction(ax_, ay_):
    """Nearest cardinal direction (by angle, i.e. largest dot with a unit vector) if
    |a| >= 0.5, else None (agent stays)."""
    if (ax_ * ax_ + ay_ * ay_) ** 0.5 < 0.5:
        return None
    dots = {"right": ax_, "left": -ax_, "up": ay_, "down": -ay_}
    return max(dots, key=dots.get)


AV_DIRECTION = [_av_direction(ax_, ay_) for ax_, ay_ in AV_ACTIONS]
AV_IN_CONE = np.array([(ax_ >= 0.5) and (abs(ay_) <= 0.25) for ax_, ay_ in AV_ACTIONS], dtype=bool)
assert all(AV_DIRECTION[m] == "right" for m in range(AV_NA) if AV_IN_CONE[m]), \
    "every crossing-cone action must have 'right' as its nearest cardinal direction"

# Deterministic (non-sticky) next-state lookup for every one of the 225 actions.
AV_DET_NEXT = np.zeros((AV_NA, S), dtype=np.int64)
for _m in range(AV_NA):
    _dirn = AV_DIRECTION[_m]
    for _si, _rc in enumerate(FREE_CELLS):
        AV_DET_NEXT[_m, _si] = _si if _dirn is None else CELL_TO_IDX[step_normal(_rc, _dirn)]


def av_sticky_action_matrices(k):
    """Per-(continuous)-action transition matrices with the sticky rule applied only
    at D, only for actions OUTSIDE the crossing cone (mirrors sticky_action_matrices,
    generalized from 4 actions to 225)."""
    P = np.zeros((AV_NA, S, S))
    for m in range(AV_NA):
        P[m, np.arange(S), AV_DET_NEXT[m]] = 1.0
    stay_prob = 0.0 if k <= 1 else (1.0 - 1.0 / k)
    if stay_prob > 0.0:
        stay_row = np.zeros(S); stay_row[D] = 1.0
        for m in range(AV_NA):
            if AV_IN_CONE[m]:
                continue  # crossing cone: always succeeds, unaffected by k
            normal_row = P[m, D, :].copy()
            P[m, D, :] = stay_prob * stay_row + (1.0 - stay_prob) * normal_row
    return P


def av_stack_sa(P):
    """(AV_NA,S,S) -> (S*AV_NA,S), row index s*AV_NA+m (state-major, action-minor)."""
    return P.transpose(1, 0, 2).reshape(S * AV_NA, S)


def av_successor_measure(P, P_continue, gamma):
    Pstack = av_stack_sa(P)
    Ncont = np.linalg.inv(np.eye(S) - gamma * P_continue)
    return Pstack @ Ncont


# ----------------------------------------------------------------------------
# 2. Sticky-door dynamics and exact successor measure.
# ----------------------------------------------------------------------------
def sticky_action_matrices(k):
    """P(a) per action, with the sticky rule applied only at state D for
    the 'fail' actions (a_cross is unaffected: it always succeeds)."""
    P = P0.copy()
    for a in A_FAIL:
        ai = ACTIONS.index(a)
        normal_row = P0[ai, D, :].copy()
        stay_row = np.zeros(S)
        stay_row[D] = 1.0
        P[ai, D, :] = (1.0 - 1.0 / k) * stay_row + (1.0 / k) * normal_row
    return P


def policy_matrix_uniform(P):
    return P.mean(axis=0)


def stack_sa(P):
    """(A,S,S) -> (S*A,S) with row index s*A+a, matching M's (s,a) ordering."""
    return P.transpose(1, 0, 2).reshape(S * A, S)


def stationary(P_pi, iters=200000, tol=1e-14):
    rho = np.ones(S) / S
    it = 0
    diff = np.inf
    for it in range(1, iters + 1):
        rho_new = rho @ P_pi
        diff = np.abs(rho_new - rho).sum()
        rho = rho_new
        if diff < tol:
            break
    rho = rho / rho.sum()
    return rho, it, diff


def successor_measure(P, P_continue, gamma):
    """M(s,a,s') = [P(a) (I - gamma P_continue)^{-1}]_{s,s'}."""
    Pstack = stack_sa(P)                       # (S*A, S)
    Ncont = np.linalg.inv(np.eye(S) - gamma * P_continue)
    M = Pstack @ Ncont                          # (S*A, S)
    return M


def check_row_sums(M, gamma, tag):
    target = 1.0 / (1.0 - gamma)
    err = np.max(np.abs(M.sum(axis=1) - target))
    log(f"  [sanity] {tag}: max |row_sum(M) - 1/(1-g)| = {err:.3e} (target {target:.4f})")
    assert err < 1e-8, f"row-sum check failed for {tag}: err={err}"
    return err


# ----------------------------------------------------------------------------
# 3. FB fit: rho-weighted rank-d truncated SVD (population optimum, closed form).
# ----------------------------------------------------------------------------
def fb_fit(M, rho, d):
    R = M / rho[None, :]                        # (S*A, S) ratio M/rho
    Ascaled = R * np.sqrt(rho)[None, :]
    U, sv, Vt = np.linalg.svd(Ascaled, full_matrices=False)
    d_eff = min(d, sv.shape[0])
    Ud, Sd, Vtd = U[:, :d_eff], sv[:d_eff], Vt[:d_eff, :]
    F = Ud * Sd[None, :]                         # (S*A, d)
    B = Vtd.T / np.sqrt(rho)[:, None]            # (S, d)
    return F, B


def fb_readout(F, B, rho, r):
    z = B.T @ (rho * r)                          # (d,)
    return F @ z                                  # (S*A,)


def fb_row(F, B, row_idx):
    """FB's fitted M_FB(row,:) = R_d(row,:) * rho, i.e. F[row] . B^T elementwise * rho."""
    return F[row_idx] @ B.T                       # this is R_d(row, :); caller multiplies by rho


# ----------------------------------------------------------------------------
# 4. Density fit: rank-d softmax model, cross-entropy, Adam (optax).
# ----------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _train_density(key, m_target, d, steps, lr):
    SA, Sdim = m_target.shape
    kf, kb = random.split(key)
    F0 = random.normal(kf, (SA, d)) * 0.1
    B0 = random.normal(kb, (Sdim, d)) * 0.1
    params0 = (F0, B0)

    def loss_fn(params):
        F, B = params
        logits = F @ B.T
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(m_target * logp, axis=-1))

    opt = optax.adam(lr)
    opt_state0 = opt.init(params0)

    def body(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, opt_state), losses = jax.lax.scan(body, (params0, opt_state0), None, length=steps)
    final_loss = loss_fn(params)
    return params, final_loss


def density_fit(m_target_np, d, seed, steps=3000, lr=1e-2, restarts=3):
    m_target = jnp.asarray(m_target_np)
    key = random.PRNGKey(seed)
    best_loss, best_params = None, None
    for i in range(restarts):
        key, sub = random.split(key)
        params, floss = _train_density(sub, m_target, d, steps, lr)
        floss = float(floss)
        if best_loss is None or floss < best_loss:
            best_loss, best_params = floss, params
    F, B = best_params
    return np.asarray(F), np.asarray(B), best_loss


def density_readout(F, B, r, gamma):
    logits = F @ B.T
    p = jax.nn.softmax(jnp.asarray(logits), axis=-1)
    Q = (1.0 / (1.0 - gamma)) * (np.asarray(p) @ r)
    return np.asarray(Q), np.asarray(p)


# ----------------------------------------------------------------------------
# 5. BFS shortest-path -> eps-greedy directed policy pi_g toward G.
# ----------------------------------------------------------------------------
def bfs_dist_to_goal(goal_idx):
    dist = np.full(S, np.inf)
    dist[goal_idx] = 0
    q = deque([goal_idx])
    # build undirected adjacency from the k=1 dynamics (a move that actually changes cell)
    adj = [[] for _ in range(S)]
    for ai in range(A):
        for i in range(S):
            j = np.argmax(P0[ai, i])
            if j != i:
                adj[i].append(j)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def greedy_actions(dist):
    """For each state, the action (fixed tie-break order) that most reduces
    distance-to-goal under normal (k=1) dynamics."""
    greedy = np.zeros(S, dtype=int)
    for i, rc in enumerate(FREE_CELLS):
        best_a, best_d = 0, dist[i]
        for ai, a in enumerate(ACTIONS):
            j = CELL_TO_IDX[step_normal(rc, a)]
            if dist[j] < best_d:
                best_d = dist[j]
                best_a = ai
        greedy[i] = best_a
    return greedy


def eps_greedy_policy_matrix(greedy, eps=0.1):
    """pi_g(a|s) as an (S,A) matrix."""
    pi = np.full((S, A), eps / A)
    pi[np.arange(S), greedy] += (1.0 - eps)
    return pi


def continuation_from_policy(P_actions, pi_sa):
    """P_pi[s,s'] = sum_a pi(a|s) P(a)[s,s']."""
    # P_actions: (A,S,S), pi_sa: (S,A)
    return np.einsum("sa,asj->sj", pi_sa, P_actions)


# ----------------------------------------------------------------------------
# 6. Metrics.
# ----------------------------------------------------------------------------
def door_metrics(Q_by_action, tag):
    """Q_by_action: dict action->Q value at D. Returns raw gap, argmax info."""
    q_cross = Q_by_action[A_CROSS]
    q_fail_mean = np.mean([Q_by_action[a] for a in A_FAIL])
    raw_gap = q_cross - q_fail_mean
    argmax_a = max(Q_by_action, key=Q_by_action.get)
    return raw_gap, q_cross, q_fail_mean, argmax_a


def tv_half(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def mass_in_room(p):
    return float(np.sum(p[NEXT_ROOM_MASK]))


def interval_estimate(values, confidence=0.95):
    """Mean + confidence interval across seeds. Uses rliable's stratified bootstrap
    if rliable is importable; otherwise falls back to a t-interval (scipy if
    available, else the hardcoded t_{0.975,df=4} critical value for n=5 seeds).
    Returns (mean, lo, hi, method_str)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    try:
        import rliable.library as rly

        def agg(x):
            return np.array([np.mean(x)])

        scores = {"m": values.reshape(n, 1)}
        point, interval = rly.get_interval_estimates(scores, agg, reps=2000)
        lo, hi = float(interval["m"][0, 0]), float(interval["m"][1, 0])
        return mean, lo, hi, "rliable stratified bootstrap (2000 reps)"
    except ImportError:
        pass
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.5 + confidence / 2, df=n - 1))
        method = f"t-interval ({int(confidence*100)}%, scipy, df={n-1})"
    except ImportError:
        tcrit = 2.776  # t_{0.975, df=4}, hardcoded fallback for n=5, scipy unavailable
        method = f"t-interval ({int(confidence*100)}%, hardcoded t-crit, df={n-1})"
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    margin = tcrit * se
    return mean, mean - margin, mean + margin, method


# ----------------------------------------------------------------------------
# 7. Main sweep -> gap.csv
# ----------------------------------------------------------------------------
DENSITY_SEEDS = [0, 1, 2, 3, 4]  # 5 independent single-init fits per cell, for the CI band


def run_main_sweep():
    ks = [1, 5, 10, 20, 40]
    ds = [4, 8, 16, 32]
    gammas = [0.9, 0.98]
    rewards = {"room": R_ROOM, "target_cell": R_TARGET}

    rows = []
    door_figure_payload = {}
    t0 = time.time()
    for gamma in gammas:
        for k in ks:
            P_k = sticky_action_matrices(k)
            P_pi = policy_matrix_uniform(P_k)
            rho, n_iter, diff = stationary(P_pi)
            log(f"[gamma={gamma} k={k}] stationary dist converged in {n_iter} iters, "
                f"final L1 delta={diff:.2e}, min(rho)={rho.min():.2e}")
            M = successor_measure(P_k, P_pi, gamma)
            check_row_sums(M, gamma, f"gamma={gamma} k={k} (random-policy M)")

            row_D = {a: D * A + ACTIONS.index(a) for a in ACTIONS}

            for d in ds:
                # --- fits computed ONCE per (gamma, k, d), shared across both reward
                # types below (neither fit depends on the reward). FB's rank-d SVD is
                # deterministic (no seed); Density is fit independently for 5 seeds,
                # each a single init with no restart-keep-best, so the 5 values are a
                # genuine sample for a confidence interval, not a best-of-N estimate. ---
                F_fb, B_fb = fb_fit(M, rho, d)
                Rd_cross_fb = fb_row(F_fb, B_fb, row_D[A_CROSS])
                m_target = (1 - gamma) * M
                density_fits = {}
                for sd in DENSITY_SEEDS:
                    F_d, B_d, floss = density_fit(m_target, d, seed=1000 * sd + d * 17 + k * 3,
                                                    restarts=1)
                    density_fits[sd] = (F_d, B_d, floss)

                for rname, r in rewards.items():
                    Q_true_all = M @ r
                    q_true_by_a = {a: Q_true_all[row_D[a]] for a in ACTIONS}
                    raw_gap_true, qc_true, qf_true, amax_true = door_metrics(q_true_by_a, "true")
                    mass_true = (1 - gamma) * M[row_D[A_CROSS]][NEXT_ROOM_MASK].sum()
                    rows.append(dict(
                        gamma=gamma, k=k, d=d, reward_type=rname, method="true", seed=-1,
                        Q_cross=qc_true, Q_fail_mean=qf_true, raw_gap=raw_gap_true, norm_gap=1.0,
                        argmax_action=amax_true, argmax_is_cross=(amax_true == A_CROSS),
                        mass_next_room_a_cross=mass_true,
                    ))

                    Q_fb_all = fb_readout(F_fb, B_fb, rho, r)
                    q_fb_by_a = {a: Q_fb_all[row_D[a]] for a in ACTIONS}
                    raw_gap_fb, qc_fb, qf_fb, amax_fb = door_metrics(q_fb_by_a, "fb")
                    mass_fb = (1 - gamma) * (Rd_cross_fb * rho)[NEXT_ROOM_MASK].sum()
                    norm_gap_fb = raw_gap_fb / raw_gap_true if raw_gap_true != 0 else np.nan
                    rows.append(dict(
                        gamma=gamma, k=k, d=d, reward_type=rname, method="fb", seed=-1,
                        Q_cross=qc_fb, Q_fail_mean=qf_fb, raw_gap=raw_gap_fb, norm_gap=norm_gap_fb,
                        argmax_action=amax_fb, argmax_is_cross=(amax_fb == A_CROSS),
                        mass_next_room_a_cross=mass_fb,
                    ))

                    for sd in DENSITY_SEEDS:
                        F_d, B_d, floss = density_fits[sd]
                        Q_d_all, p_all = density_readout(F_d, B_d, r, gamma)
                        q_d_by_a = {a: Q_d_all[row_D[a]] for a in ACTIONS}
                        raw_gap_d, qc_d, qf_d, amax_d = door_metrics(q_d_by_a, "density")
                        mass_d = p_all[row_D[A_CROSS]][NEXT_ROOM_MASK].sum()
                        norm_gap_d = raw_gap_d / raw_gap_true if raw_gap_true != 0 else np.nan
                        rows.append(dict(
                            gamma=gamma, k=k, d=d, reward_type=rname, method="density", seed=sd,
                            Q_cross=qc_d, Q_fail_mean=qf_d, raw_gap=raw_gap_d, norm_gap=norm_gap_d,
                            argmax_action=amax_d, argmax_is_cross=(amax_d == A_CROSS),
                            mass_next_room_a_cross=mass_d,
                        ))

                        if (gamma == 0.98 and k == 20 and d == 8 and rname == "room" and sd == 0):
                            a_fail_show = "left"
                            row_fail = row_D[a_fail_show]
                            p_true_cross = (1 - gamma) * M[row_D[A_CROSS]]
                            p_true_fail = (1 - gamma) * M[row_fail]
                            p_fb_cross = (1 - gamma) * Rd_cross_fb * rho
                            Rd_fail_fb = fb_row(F_fb, B_fb, row_fail)
                            p_fb_fail = (1 - gamma) * Rd_fail_fb * rho
                            p_density_cross = p_all[row_D[A_CROSS]]
                            p_density_fail = p_all[row_fail]
                            door_figure_payload = dict(
                                a_fail_show=a_fail_show, density_seed_used=sd,
                                p_true_cross=p_true_cross, p_true_fail=p_true_fail,
                                p_fb_cross=p_fb_cross, p_fb_fail=p_fb_fail,
                                p_density_cross=p_density_cross, p_density_fail=p_density_fail,
                            )
            log(f"  ...done gamma={gamma} k={k} ({time.time()-t0:.1f}s elapsed)")

    csv_path = HERE / "gap.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {csv_path} ({len(rows)} rows)")
    return rows, door_figure_payload


# ----------------------------------------------------------------------------
# 8. Directed-policy occupancy experiment (Figure-1 analogue) -> occupancy.csv
# ----------------------------------------------------------------------------
def run_occupancy_experiment():
    gamma = 0.98
    ks = [1, 20, 40]
    ds = [4, 8, 16]
    eps = 0.1

    dist_to_G = bfs_dist_to_goal(G)
    greedy = greedy_actions(dist_to_G)
    pi_g = eps_greedy_policy_matrix(greedy, eps=eps)
    a_star_idx = int(greedy[S0])
    a_star = ACTIONS[a_star_idx]
    log(f"[occupancy] greedy action at S0={S0_RC} toward G={G_RC}: {a_star} "
        f"(BFS dist S0->G = {int(dist_to_G[S0])})")

    rows = []
    seed = 10000
    figure_payload = {}
    for k in ks:
        P_k = sticky_action_matrices(k)
        P_pi_rand = policy_matrix_uniform(P_k)
        rho, _, _ = stationary(P_pi_rand)  # dataset marginal: always the RANDOM policy's stationary dist

        P_pi_g = continuation_from_policy(P_k, pi_g)
        M_g = successor_measure(P_k, P_pi_g, gamma)
        check_row_sums(M_g, gamma, f"gamma={gamma} k={k} (directed-policy M_g)")

        row_idx = S0 * A + a_star_idx
        p_true = (1 - gamma) * M_g[row_idx]
        tv_true_rho = tv_half(p_true, rho)
        mass_true = mass_in_room(p_true)

        for d in ds:
            F_fb, B_fb = fb_fit(M_g, rho, d)
            Rd_row = fb_row(F_fb, B_fb, row_idx)
            p_fb = (1 - gamma) * Rd_row * rho
            fb_rowsum = float(p_fb.sum())
            tv_fb_true = tv_half(p_fb, p_true)
            tv_fb_rho = tv_half(p_fb, rho)
            mass_fb = mass_in_room(p_fb)

            m_target = (1 - gamma) * M_g
            seed += 1
            F_d, B_d, floss = density_fit(m_target, d, seed=seed)
            logits_row = F_d[row_idx] @ B_d.T
            p_density = np.asarray(jax.nn.softmax(jnp.asarray(logits_row)))
            tv_d_true = tv_half(p_density, p_true)
            tv_d_rho = tv_half(p_density, rho)
            mass_d = mass_in_room(p_density)

            rows.append(dict(gamma=gamma, k=k, d=d, method="true",
                              tv_to_true=0.0, tv_to_rho=tv_true_rho, mass_next_room=mass_true))
            rows.append(dict(gamma=gamma, k=k, d=d, method="fb",
                              tv_to_true=tv_fb_true, tv_to_rho=tv_fb_rho, mass_next_room=mass_fb))
            rows.append(dict(gamma=gamma, k=k, d=d, method="density",
                              tv_to_true=tv_d_true, tv_to_rho=tv_d_rho, mass_next_room=mass_d))

            if k == 20 and d == 8:
                figure_payload = dict(
                    p_true=p_true, p_fb=p_fb, p_density=p_density, rho=rho,
                    S0_RC=S0_RC, D_RC=D_RC, G_RC=G_RC, a_star=a_star,
                )
        log(f"  [occupancy] k={k}: tv(true,rho)={tv_true_rho:.3f}  "
            f"(reference point: how far the data marginal itself is from the true directed row)")

    csv_path = HERE / "occupancy.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {csv_path} ({len(rows)} rows)")
    return rows, figure_payload


# ----------------------------------------------------------------------------
# 8b. TD-trained tabular agents (no closed form): fixed uniform-random policy pi,
#     no actor, no z-conditioning. Two separate tables (F, B) per method, target
#     copies (F_bar, B_bar) via Polyak averaging, trained on a sampled dataset.
# ----------------------------------------------------------------------------
TRAIN_KS = [1, 5, 10, 20, 40]
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TRAIN_D = 8
TRAIN_GAMMA = 0.98
TRAIN_STEPS = 20000
TRAIN_LR = 3e-4
TRAIN_TAU = 0.005
TRAIN_BATCH_T = 256
TRAIN_BATCH_U = 256
TRAIN_DATASET_N = 100_000
TRAIN_RESET_EVERY = 500

# Deterministic (k=1) per-action next-state lookup, shared by the dataset sampler.
_DET_NEXT = np.zeros((A, S), dtype=np.int64)
for _ai in range(A):
    for _si, _rc in enumerate(FREE_CELLS):
        _DET_NEXT[_ai, _si] = CELL_TO_IDX[step_normal(_rc, ACTIONS[_ai])]
_A_FAIL_IDX = set(ACTIONS.index(a) for a in A_FAIL)


def generate_dataset_td(k, seed, n=TRAIN_DATASET_N, reset_every=TRAIN_RESET_EVERY):
    """A single long rollout of the uniform-random policy under the k-sticky dynamics,
    restarting from a fresh uniform-random state every `reset_every` steps ('random
    starts'). Returns (s_idx, a_idx, sp_idx) int64 arrays of length n. Pure numpy --
    the dynamics are a sequential Markov chain, not vectorizable across time."""
    rng = np.random.default_rng(seed * 100_003 + k)
    s_idx = np.empty(n, dtype=np.int64)
    a_idx = np.empty(n, dtype=np.int64)
    sp_idx = np.empty(n, dtype=np.int64)
    s = int(rng.integers(0, S))
    stay_prob = 0.0 if k <= 1 else (1.0 - 1.0 / k)
    for t in range(n):
        a = int(rng.integers(0, A))
        nxt = _DET_NEXT[a, s]
        if s == D and a in _A_FAIL_IDX and stay_prob > 0.0 and rng.random() < stay_prob:
            nxt = D
        s_idx[t] = s; a_idx[t] = a; sp_idx[t] = nxt
        s = nxt
        if (t + 1) % reset_every == 0:
            s = int(rng.integers(0, S))
    return s_idx, a_idx, sp_idx


def _td_init(key, d):
    kf, kb = random.split(key)
    F0 = random.normal(kf, (S * A, d)) * 0.1
    B0 = random.normal(kb, (S, d)) * 0.1
    return F0, B0


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def _train_fb_td(key, s_idx, a_idx, sp_idx, d, steps, lr, tau, batch_t, batch_u, gamma):
    n = s_idx.shape[0]
    key, kinit = random.split(key)
    F0, B0 = _td_init(kinit, d)
    Fbar0, Bbar0 = F0, B0
    opt = optax.adam(lr)
    opt_state0 = opt.init((F0, B0))

    def loss_fn(params, target_params, s, a, sp, ap, u):
        F, B = params
        F_bar, B_bar = target_params
        F_sa = F[s * A + a]
        B_u = B[u]
        B_sp = B[sp]
        F_bar_spap = F_bar[sp * A + ap]
        B_bar_u = B_bar[u]
        pred = F_sa @ B_u.T
        boot = jax.lax.stop_gradient(gamma * (F_bar_spap @ B_bar_u.T))
        term1 = jnp.mean((pred - boot) ** 2)
        term2 = jnp.mean(jnp.sum(F_sa * B_sp, axis=-1))
        return term1 - 2.0 * term2

    def body(carry, key_t):
        params, target_params, opt_state = carry
        kb, ku, ka = random.split(key_t, 3)
        idx_t = random.randint(kb, (batch_t,), 0, n)
        idx_u = random.randint(ku, (batch_u,), 0, n)
        s = s_idx[idx_t]; a = a_idx[idx_t]; sp = sp_idx[idx_t]
        u = sp_idx[idx_u]
        ap = random.randint(ka, (batch_t,), 0, A)
        loss, grads = jax.value_and_grad(loss_fn)(params, target_params, s, a, sp, ap, u)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target_params = jax.tree_util.tree_map(
            lambda tp, p: (1 - tau) * tp + tau * p, target_params, params)
        return (params, target_params, opt_state), loss

    keys = random.split(key, steps)
    init_carry = ((F0, B0), (Fbar0, Bbar0), opt_state0)
    (params, target_params, opt_state), losses = jax.lax.scan(body, init_carry, keys)
    F, B = params
    return F, B, losses[-1]


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def _train_density_td(key, s_idx, a_idx, sp_idx, d, steps, lr, tau, batch_t, batch_u, gamma):
    n = s_idx.shape[0]
    key, kinit = random.split(key)
    F0, B0 = _td_init(kinit, d)
    Fbar0, Bbar0 = F0, B0
    opt = optax.adam(lr)
    opt_state0 = opt.init((F0, B0))

    def loss_fn(params, target_params, s, a, sp, ap, u):
        F, B = params
        F_bar, B_bar = target_params
        F_sa = F[s * A + a]
        B_u = B[u]; B_bar_u = B_bar[u]
        B_sp = B[sp]; B_bar_sp = B_bar[sp]
        F_bar_spap = F_bar[sp * A + ap]

        logits_pred = jnp.concatenate(
            [F_sa @ B_u.T, jnp.sum(F_sa * B_sp, axis=-1, keepdims=True)], axis=1)
        logits_bar = jnp.concatenate(
            [F_bar_spap @ B_bar_u.T, jnp.sum(F_bar_spap * B_bar_sp, axis=-1, keepdims=True)], axis=1)
        p_bar = jax.nn.softmax(logits_bar, axis=1)
        onehot_extra = jax.nn.one_hot(jnp.full((batch_t,), batch_u), batch_u + 1)
        target = jax.lax.stop_gradient((1.0 - gamma) * onehot_extra + gamma * p_bar)
        logp = jax.nn.log_softmax(logits_pred, axis=1)
        return -jnp.mean(jnp.sum(target * logp, axis=1))

    def body(carry, key_t):
        params, target_params, opt_state = carry
        kb, ku, ka = random.split(key_t, 3)
        idx_t = random.randint(kb, (batch_t,), 0, n)
        idx_u = random.randint(ku, (batch_u,), 0, n)
        s = s_idx[idx_t]; a = a_idx[idx_t]; sp = sp_idx[idx_t]
        u = sp_idx[idx_u]
        ap = random.randint(ka, (batch_t,), 0, A)
        loss, grads = jax.value_and_grad(loss_fn)(params, target_params, s, a, sp, ap, u)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target_params = jax.tree_util.tree_map(
            lambda tp, p: (1 - tau) * tp + tau * p, target_params, params)
        return (params, target_params, opt_state), loss

    keys = random.split(key, steps)
    init_carry = ((F0, B0), (Fbar0, Bbar0), opt_state0)
    (params, target_params, opt_state), losses = jax.lax.scan(body, init_carry, keys)
    F, B = params
    return F, B, losses[-1]


def run_trained_sweep():
    """FB-TD and Density-TD, TRAIN_KS x TRAIN_SEEDS, room reward, d=TRAIN_D,
    gamma=TRAIN_GAMMA. Writes gap_trained.csv; returns (rows, door_payload_trained)
    where door_payload_trained is captured at k=20, seed=0 (for maps_contrast_trained)."""
    rows = []
    door_payload_trained = {}
    t0 = time.time()
    row_D = {a: D * A + ACTIONS.index(a) for a in ACTIONS}
    for k in TRAIN_KS:
        P_k = sticky_action_matrices(k)
        P_pi = policy_matrix_uniform(P_k)
        rho, _, _ = stationary(P_pi)
        M = successor_measure(P_k, P_pi, TRAIN_GAMMA)
        Q_true_all = M @ R_ROOM
        q_true_by_a = {a: Q_true_all[row_D[a]] for a in ACTIONS}
        raw_gap_true, qc_true, qf_true, amax_true = door_metrics(q_true_by_a, "true")
        p_true_cross = (1 - TRAIN_GAMMA) * M[row_D[A_CROSS]]
        p_true_fail = (1 - TRAIN_GAMMA) * M[row_D["left"]]

        for seed in TRAIN_SEEDS:
            s_idx, a_idx, sp_idx = generate_dataset_td(k, seed)
            s_idx_j = jnp.asarray(s_idx); a_idx_j = jnp.asarray(a_idx); sp_idx_j = jnp.asarray(sp_idx)
            key = random.PRNGKey(20_000 * k + seed)
            key_fb, key_density = random.split(key)

            F_fb, B_fb, loss_fb = _train_fb_td(
                key_fb, s_idx_j, a_idx_j, sp_idx_j, TRAIN_D, TRAIN_STEPS, TRAIN_LR, TRAIN_TAU,
                TRAIN_BATCH_T, TRAIN_BATCH_U, TRAIN_GAMMA)
            F_fb = np.asarray(F_fb); B_fb = np.asarray(B_fb)
            Q_fb_all = fb_readout(F_fb, B_fb, rho, R_ROOM)
            q_fb_by_a = {a: Q_fb_all[row_D[a]] for a in ACTIONS}
            raw_gap_fb, qc_fb, qf_fb, amax_fb = door_metrics(q_fb_by_a, "fb_td")
            norm_gap_fb = raw_gap_fb / raw_gap_true if raw_gap_true != 0 else np.nan
            Rd_cross_fb = fb_row(F_fb, B_fb, row_D[A_CROSS])
            Rd_fail_fb = fb_row(F_fb, B_fb, row_D["left"])
            p_fb_cross = (1 - TRAIN_GAMMA) * Rd_cross_fb * rho
            p_fb_fail = (1 - TRAIN_GAMMA) * Rd_fail_fb * rho
            tv_fb_cross = tv_half(p_fb_cross, p_true_cross)
            tv_fb_fail = tv_half(p_fb_fail, p_true_fail)
            rows.append(dict(k=k, seed=seed, method="fb_td", raw_gap=raw_gap_fb,
                              norm_gap=norm_gap_fb, tv_cross=tv_fb_cross, tv_fail=tv_fb_fail,
                              final_train_loss=float(loss_fb)))

            F_d, B_d, loss_d = _train_density_td(
                key_density, s_idx_j, a_idx_j, sp_idx_j, TRAIN_D, TRAIN_STEPS, TRAIN_LR, TRAIN_TAU,
                TRAIN_BATCH_T, TRAIN_BATCH_U, TRAIN_GAMMA)
            F_d = np.asarray(F_d); B_d = np.asarray(B_d)
            Q_d_all, p_all = density_readout(F_d, B_d, R_ROOM, TRAIN_GAMMA)
            q_d_by_a = {a: Q_d_all[row_D[a]] for a in ACTIONS}
            raw_gap_d, qc_d, qf_d, amax_d = door_metrics(q_d_by_a, "density_td")
            norm_gap_d = raw_gap_d / raw_gap_true if raw_gap_true != 0 else np.nan
            p_density_cross = p_all[row_D[A_CROSS]]
            p_density_fail = p_all[row_D["left"]]
            tv_d_cross = tv_half(p_density_cross, p_true_cross)
            tv_d_fail = tv_half(p_density_fail, p_true_fail)
            rows.append(dict(k=k, seed=seed, method="density_td", raw_gap=raw_gap_d,
                              norm_gap=norm_gap_d, tv_cross=tv_d_cross, tv_fail=tv_d_fail,
                              final_train_loss=float(loss_d)))

            if k == 20 and seed == 0:
                door_payload_trained = dict(
                    a_fail_show="left", density_seed_used=seed,
                    p_true_cross=p_true_cross, p_true_fail=p_true_fail,
                    p_fb_cross=p_fb_cross, p_fb_fail=p_fb_fail,
                    p_density_cross=p_density_cross, p_density_fail=p_density_fail,
                )
        log(f"  [trained] k={k} done, 5 seeds x 2 methods ({time.time()-t0:.1f}s elapsed)")

    csv_path = HERE / "gap_trained.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {csv_path} ({len(rows)} rows)")
    return rows, door_payload_trained


def trained_gap_ci(trained_rows, method, k):
    vals = [float(r["norm_gap"]) for r in trained_rows if r["method"] == method and int(r["k"]) == k]
    return interval_estimate(vals)


# ----------------------------------------------------------------------------
# 8c. Action-value heatmap at the bottleneck (Soft-FB-style Q(D, a) surface over a
#     2D continuous action space), closed-form FB and Density fits only (no TD).
# ----------------------------------------------------------------------------
def run_action_value_door_experiment():
    gamma = 0.98
    ks_main = [1, 5, 10, 20, 40]
    k_for_d_sweep = 20
    ds_for_k_sweep = [8, 16]
    ds_for_d_sweep = [2, 4, 8, 16, 32]
    r = R_ROOM
    n_density_seeds = 5

    needed = set()
    for k in ks_main:
        for d in ds_for_k_sweep:
            needed.add((k, d))
    for d in ds_for_d_sweep:
        needed.add((k_for_d_sweep, d))

    t0 = time.time()
    cache_by_k = {}
    for k in sorted({kk for kk, _ in needed}):
        P_k = av_sticky_action_matrices(k)
        P_pi = policy_matrix_uniform(P_k)
        rho, n_iter, diff = stationary(P_pi)
        M = av_successor_measure(P_k, P_pi, gamma)
        row_sum_err = float(np.max(np.abs(M.sum(axis=1) - 1.0 / (1 - gamma))))
        cache_by_k[k] = (M, rho)
        log(f"[action-value] k={k}: stationary in {n_iter} iters (delta={diff:.1e}), "
            f"M shape={M.shape}, row-sum err={row_sum_err:.2e}")
        assert row_sum_err < 1e-6, f"action-value M row-sum check failed at k={k}"

    row_D_slice = slice(D * AV_NA, D * AV_NA + AV_NA)
    results = {}
    csv_rows = []
    for (k, d) in sorted(needed):
        M, rho = cache_by_k[k]
        Q_true_all = M @ r
        Q_true_D = Q_true_all[row_D_slice]
        gap_true = float(Q_true_D[AV_IN_CONE].mean() - Q_true_D[~AV_IN_CONE].mean())

        F_fb, B_fb = fb_fit(M, rho, d)
        Q_fb_all = fb_readout(F_fb, B_fb, rho, r)
        Q_fb_D = Q_fb_all[row_D_slice]
        gap_fb = float(Q_fb_D[AV_IN_CONE].mean() - Q_fb_D[~AV_IN_CONE].mean())
        norm_gap_fb = gap_fb / gap_true if gap_true != 0 else np.nan
        csv_rows.append(dict(k=k, d=d, method="true", seed=-1, gap=gap_true, norm_gap=1.0))
        csv_rows.append(dict(k=k, d=d, method="fb", seed=-1, gap=gap_fb, norm_gap=norm_gap_fb))

        m_target = (1 - gamma) * M
        Q_density_D_seed0, norm_gap_density_seed0 = None, None
        density_norm_gaps = []
        for seed in range(n_density_seeds):
            F_d, B_d, floss = density_fit(m_target, d, seed=seed, restarts=1)
            Q_d_all, p_all = density_readout(F_d, B_d, r, gamma)
            Q_d_D = Q_d_all[row_D_slice]
            gap_d = float(Q_d_D[AV_IN_CONE].mean() - Q_d_D[~AV_IN_CONE].mean())
            norm_gap_d = gap_d / gap_true if gap_true != 0 else np.nan
            density_norm_gaps.append(norm_gap_d)
            csv_rows.append(dict(k=k, d=d, method="density", seed=seed, gap=gap_d, norm_gap=norm_gap_d))
            if seed == 0:
                Q_density_D_seed0 = Q_d_D.copy()
                norm_gap_density_seed0 = norm_gap_d

        results[(k, d)] = dict(
            Q_true_D=Q_true_D, Q_fb_D=Q_fb_D, Q_density_D=Q_density_D_seed0,
            gap_true=gap_true, norm_gap_fb=norm_gap_fb,
            norm_gap_density_seed0=norm_gap_density_seed0, density_norm_gaps=density_norm_gaps,
        )
        log(f"  [action-value] k={k} d={d}: gap_true={gap_true:.4f} norm_gap_fb={norm_gap_fb:.3f} "
            f"norm_gap_density(seed0)={norm_gap_density_seed0:.3f} "
            f"density_mean={np.mean(density_norm_gaps):.3f} ({time.time()-t0:.1f}s elapsed)")

    csv_path = HERE / "action_value_door.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    log(f"wrote {csv_path} ({len(csv_rows)} rows)")

    npz_payload = {"actions": AV_ACTIONS, "in_cone": AV_IN_CONE}
    for (k, d), res in results.items():
        npz_payload[f"true_k{k}_d{d}"] = res["Q_true_D"]
        npz_payload[f"fb_k{k}_d{d}"] = res["Q_fb_D"]
        npz_payload[f"density_k{k}_d{d}"] = res["Q_density_D"]
    npz_path = HERE / "action_value_door.npz"
    np.savez(npz_path, **npz_payload)
    log(f"wrote {npz_path}")

    return results


def draw_action_heatmap(ax, Q_D_vector, title, vmin=None, vmax=None):
    import matplotlib.patches as mpatches
    grid = Q_D_vector.reshape(AV_GRID_N, AV_GRID_N)
    if vmin is None or vmax is None:
        vmin, vmax = float(grid.min()), float(grid.max())
    im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, extent=[-1, 1, -1, 1],
                    origin="upper", interpolation="nearest")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=5.5, length=2, color=MUTED, labelcolor=MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.6)
    ax.set_title(title, fontsize=7.8, color=INK, pad=3)
    rect = mpatches.Rectangle((0.5, -0.25), 0.5, 0.5, fill=False, edgecolor="white",
                               linewidth=1.3, zorder=5)
    ax.add_patch(rect)
    amax = int(np.argmax(Q_D_vector))
    ax_amax, ay_amax = AV_ACTIONS[amax]
    in_cone_amax = bool(AV_IN_CONE[amax])
    ax.scatter([ax_amax], [ay_amax], s=32, facecolor="white", edgecolor="black",
               linewidth=0.9, zorder=6)
    return im, in_cone_amax


def save_action_value_figure(results, columns, column_kind, out_stem, d_fixed=None, k_fixed=None,
                              column_shared_scale=False):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
        "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    ncols = len(columns)
    fig = plt.figure(figsize=(2.1 * ncols, 6.6), constrained_layout=True)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.10])
    grid_gs = outer[0].subgridspec(3, ncols, hspace=0.35, wspace=0.15)

    row_specs = [("true", "Q_true_D", None), ("FB", "Q_fb_D", "norm_gap_fb"),
                 ("Density-FB", "Q_density_D", "norm_gap_density_seed0")]

    col_scale = {}
    if column_shared_scale:
        for col_val in columns:
            cell_key = (col_val, d_fixed) if column_kind == "k" else (k_fixed, col_val)
            true_vals = results[cell_key]["Q_true_D"]
            col_scale[col_val] = (float(true_vals.min()), float(true_vals.max()))

    argmax_report = {}  # column value -> {"fb": bool, "density": bool}
    axes_by_cell = {}
    ims_by_col = {}
    for ridx, (row_name, key, gap_key) in enumerate(row_specs):
        for cidx, col_val in enumerate(columns):
            cell_key = (col_val, d_fixed) if column_kind == "k" else (k_fixed, col_val)
            res = results[cell_key]
            ax = fig.add_subplot(grid_gs[ridx, cidx])
            axes_by_cell[(ridx, cidx)] = ax
            title = (f"k = {col_val}" if column_kind == "k" else f"d = {col_val}") if ridx == 0 else ""
            vmin_c, vmax_c = col_scale.get(col_val, (None, None))
            im, in_cone_amax = draw_action_heatmap(ax, res[key], title, vmin=vmin_c, vmax=vmax_c)
            ims_by_col[cidx] = im
            if ridx == 0:
                ax.set_xlabel("")
            if gap_key is not None:
                ax.text(0.5, -0.14, f"gap = {res[gap_key]:.2f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=6.4, color=MUTED)
                argmax_report.setdefault(col_val, {})["fb" if ridx == 1 else "density"] = in_cone_amax
            else:
                argmax_report.setdefault(col_val, {})["true"] = in_cone_amax
            if cidx == 0:
                ax.set_ylabel(row_name, fontsize=8.2, color=INK)

    if column_shared_scale:
        for cidx in range(ncols):
            col_axes = [axes_by_cell[(0, cidx)], axes_by_cell[(1, cidx)], axes_by_cell[(2, cidx)]]
            cb = fig.colorbar(ims_by_col[cidx], ax=col_axes, location="bottom",
                               shrink=0.85, aspect=18, pad=0.04)
            cb.ax.tick_params(labelsize=4.6, color=MUTED, labelcolor=MUTED, length=2)
            cb.outline.set_visible(False)

    cap_ax = fig.add_subplot(outer[1])
    cap_ax.axis("off")
    xlabel = "k" if column_kind == "k" else "d"
    fixed_note = f"d={d_fixed}" if column_kind == "k" else f"k={k_fixed}"
    scale_note = ("each COLUMN shares one color scale, fixed to that column's true panel's "
                  "min/max (so a fading cone is visible directly)" if column_shared_scale else
                  "each panel normalized to its OWN min/max (shape, not magnitude, is "
                  "comparable across panels)")
    cap_ax.text(0.0, 1.0,
                f"Q(D, a) over the 15x15 discretized action plane ($a_x$ horizontal, $a_y$ "
                f"vertical), room reward, $\\gamma$=0.98, {fixed_note}; viridis, {scale_note}; "
                "white rectangle = the crossing cone ($a_x{\\geq}0.5$, $|a_y|{\\leq}0.25$); "
                "white dot = argmax action; Density uses seed 0.",
                transform=cap_ax.transAxes, ha="left", va="top", fontsize=6.4, color=MUTED,
                linespacing=1.5)

    DOOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = DOOR_FIGURES_DIR / f"{out_stem}.pdf"
    out_png = DOOR_FIGURES_DIR / f"{out_stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=260)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")
    return argmax_report


# ----------------------------------------------------------------------------
# 9. Sanity checks (§ "Sanity checks" in the task spec).
# ----------------------------------------------------------------------------
def run_sanity_checks():
    log("\n=== Sanity checks ===")
    gamma = 0.98
    k = 20
    P_k = sticky_action_matrices(k)
    P_pi = policy_matrix_uniform(P_k)
    rho, _, _ = stationary(P_pi)
    M = successor_measure(P_k, P_pi, gamma)
    check_row_sums(M, gamma, "d=|S| sanity block")

    row_D_cross = D * A + ACTIONS.index(A_CROSS)
    row_D_fail = [D * A + ACTIONS.index(a) for a in A_FAIL]
    Q_true_all = M @ R_ROOM
    raw_gap_true = Q_true_all[row_D_cross] - np.mean(Q_true_all[row_D_fail])

    d_full = S  # 104
    F_fb, B_fb = fb_fit(M, rho, d_full)
    Q_fb_all = fb_readout(F_fb, B_fb, rho, R_ROOM)
    raw_gap_fb = Q_fb_all[row_D_cross] - np.mean(Q_fb_all[row_D_fail])
    norm_gap_fb = raw_gap_fb / raw_gap_true
    log(f"[d=|S|={d_full}] FB normalized gap = {norm_gap_fb:.6f} (want 1 within 1e-3)")
    assert abs(norm_gap_fb - 1.0) < 1e-3, "FB at d=|S| should reproduce the true gap exactly"

    m_target = (1 - gamma) * M
    F_d, B_d, floss = density_fit(m_target, d_full, seed=999)
    Q_d_all, _ = density_readout(F_d, B_d, R_ROOM, gamma)
    raw_gap_d = Q_d_all[row_D_cross] - np.mean(Q_d_all[row_D_fail])
    norm_gap_d = raw_gap_d / raw_gap_true
    log(f"[d=|S|={d_full}] Density normalized gap = {norm_gap_d:.6f} "
        f"(want 1 within 0.05; final CE loss={floss:.5f})")
    assert abs(norm_gap_d - 1.0) < 0.05, "Density at d=|S| should reproduce the true gap closely"

    # k=1: the two actions should still differ under the TRUE measure (no stickiness yet,
    # but crossing the doorway vs bumping around inside the room are still different).
    P_k1 = sticky_action_matrices(1)
    P_pi1 = policy_matrix_uniform(P_k1)
    rho1, _, _ = stationary(P_pi1)
    M1 = successor_measure(P_k1, P_pi1, gamma)
    Q1_all = M1 @ R_ROOM
    raw_gap_true_k1 = Q1_all[row_D_cross] - np.mean(Q1_all[row_D_fail])
    log(f"[k=1] true raw gap (room reward) = {raw_gap_true_k1:.4f} (want clearly nonzero)")
    assert abs(raw_gap_true_k1) > 1e-2, "actions should still differ at k=1 (no stickiness)"

    log("all sanity checks passed.\n")


# ----------------------------------------------------------------------------
# 10. Figure. Colors: seaborn "colorblind" palette (Wong 2011 CVD-safe hues),
#     one fixed hue per method throughout, never cycled or reassigned.
#     (seaborn itself is not installed in this offline venv; these are its
#     documented "colorblind" hex values, used directly.)
# ----------------------------------------------------------------------------
TRUE_COLOR = "#0173B2"      # colorblind slot 1 (blue)   - true successor measure
FB_COLOR = "#DE8F05"        # colorblind slot 2 (orange) - FB squared-loss rank-d fit
DENSITY_COLOR = "#029E73"   # colorblind slot 3 (green)  - Density softmax rank-d fit
RHO_COLOR = "#949494"       # colorblind slot 8 (gray)   - dataset marginal rho (reference, not a fit)
WALL_COLOR = "#2b2b2b"
INK = "#2a2a2a"
MUTED = "#6b6b6b"


def to_grid(vals):
    g = np.full((GRID_N, GRID_N), np.nan)
    for i, (r, c) in enumerate(FREE_CELLS):
        g[r, c] = vals[i]
    return g


def neutral_cmap():
    """One shared sequential hue for every heatmap magnitude in the figure --
    method identity is carried by the corner swatch + row label, not by hue,
    so that the color scale is genuinely comparable across true/FB/Density.
    (Used by the full appendix figure.)"""
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("seq_neutral", ["#ffffff", "#2a2a4d"])
    cmap.set_bad(WALL_COLOR)
    return cmap


def seq_cmap(hex_color):
    """Per-method sequential hue (white -> that method's categorical color).
    Used by the compact main figure, where identity is carried by hue directly
    (matching the gap-vs-k line colors) instead of a separate swatch."""
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", ["#ffffff", hex_color])
    cmap.set_bad(WALL_COLOR)
    return cmap


def mark_cells(ax, cells, marker_style):
    for label, rc, kw in cells:
        r, c = rc
        ax.scatter([c], [r], zorder=5, **{**marker_style, **kw})
        ax.annotate(label, xy=(c, r), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5, color=INK,
                     fontweight="bold", zorder=6)


def draw_heatmap(ax, vals, vmin, vmax, cmap, title, mark_specs, show_negative_note=False,
                  identity_color=None):
    g = to_grid(vals)
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.8)
    ax.set_title(title, fontsize=9.5, color=INK, pad=(14 if mark_specs else 4))
    mark_cells(ax, mark_specs, dict(s=64, linewidths=1.2, edgecolors="white"))
    if identity_color is not None:
        # method-identity swatch (color carries identity here, never the heatmap hue itself,
        # so the shared magnitude scale stays comparable across methods)
        import matplotlib.patches as mpatches
        ax.add_patch(mpatches.Rectangle((0.03, 0.86), 0.12, 0.10, transform=ax.transAxes,
                                         facecolor=identity_color, edgecolor="white",
                                         linewidth=0.8, zorder=7))
    if show_negative_note:
        vmin_actual = float(np.nanmin(g))
        if vmin_actual < -1e-9:
            ax.text(0.5, 0.02, f"min entry = {vmin_actual:.3f} (not a valid probability)",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=MUTED, edgecolor="none", alpha=0.85))
    return im


def caption_axis(fig, gs_slot, text, fontsize=8.3):
    ax = fig.add_subplot(gs_slot)
    ax.axis("off")
    ax.text(0.0, 1.0, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, color=INK, linespacing=1.5)
    return ax


def make_figure(door_payload, occ_payload, gap_rows):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "text.color": INK,
        "axes.edgecolor": "#cfcfcf",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": False,
        "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.5, 15.0), constrained_layout=True)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.05, 2.55])
    cmap = neutral_cmap()

    # =========================================================================
    # Top group: directed-occupancy panels (Figure-1 analogue), + caption row.
    # =========================================================================
    top_group = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.12])
    top_gs = top_group[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1])
    occ_axes = [fig.add_subplot(top_gs[0, i]) for i in range(4)]

    p_true, p_fb, p_density, rho = (occ_payload["p_true"], occ_payload["p_fb"],
                                     occ_payload["p_density"], occ_payload["rho"])
    S0_RC_, D_RC_, G_RC_, a_star = (occ_payload["S0_RC"], occ_payload["D_RC"],
                                     occ_payload["G_RC"], occ_payload["a_star"])
    marks = [("S0", S0_RC_, dict(marker="o", color="#444444")),
             ("D", D_RC_, dict(marker="D", color="#444444")),
             ("G", G_RC_, dict(marker="*", color="#444444", s=140))]

    vmax_occ = max(np.nanmax(p_true), np.nanmax(p_fb), np.nanmax(p_density))
    vmin_occ = min(0.0, np.nanmin(p_fb))
    draw_heatmap(occ_axes[0], p_true, vmin_occ, vmax_occ, cmap,
                 "true  $M_g(S_0,a^\\ast,\\cdot)$", marks, identity_color=TRUE_COLOR)
    draw_heatmap(occ_axes[1], p_fb, vmin_occ, vmax_occ, cmap,
                 "FB fit", marks, show_negative_note=True, identity_color=FB_COLOR)
    im_occ = draw_heatmap(occ_axes[2], p_density, vmin_occ, vmax_occ, cmap,
                           "Density fit", marks, identity_color=DENSITY_COLOR)
    cb = fig.colorbar(im_occ, ax=occ_axes[0:3], location="bottom", shrink=0.9, aspect=45, pad=0.03)
    cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("probability mass at $s'$  (colored squares = true / FB / Density identity;\n"
                 "true, FB, and Density share this one scale)", fontsize=7.3, color=MUTED)

    im_rho = draw_heatmap(occ_axes[3], rho, 0.0, np.nanmax(rho), cmap,
                           r"data marginal $\rho$", marks, identity_color=RHO_COLOR)
    cb2 = fig.colorbar(im_rho, ax=occ_axes[3], location="bottom", shrink=0.9, aspect=20, pad=0.03)
    cb2.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED)
    cb2.outline.set_visible(False)
    cb2.set_label(r"$\rho$ scale (own, for reference)", fontsize=7.3, color=MUTED)

    caption_axis(
        fig, top_group[1],
        f"Directed occupancy (Figure-1 analogue): from $S_0$={S0_RC_}, action $a^\\ast$={a_star}\n"
        f"(the policy's action toward $G$={G_RC_}), continuing under the goal-directed policy "
        "$\\pi_g$ (k=20, d=8, $\\gamma$=0.98).\n"
        "FB's row is visibly more diffuse across the room than Density's; by total variation, FB's "
        "distance to the true row is 2$-$4$\\times$ Density's (occupancy.csv), though neither collapses all the way to $\\rho$.")

    # =========================================================================
    # Bottom group: door-action heatmaps (left) + gap-vs-k panels (right).
    # =========================================================================
    bottom_gs = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.35])

    # --- door heatmaps ---
    door_group = bottom_gs[0].subgridspec(2, 1, height_ratios=[1.0, 0.08])
    door_gs = door_group[0].subgridspec(3, 2)
    a_fail_show = door_payload["a_fail_show"]
    rows_spec = [
        ("true", TRUE_COLOR, door_payload["p_true_cross"], door_payload["p_true_fail"]),
        ("FB", FB_COLOR, door_payload["p_fb_cross"], door_payload["p_fb_fail"]),
        ("Density", DENSITY_COLOR, door_payload["p_density_cross"], door_payload["p_density_fail"]),
    ]
    cross_vals = [door_payload["p_true_cross"], door_payload["p_fb_cross"], door_payload["p_density_cross"]]
    fail_vals = [door_payload["p_true_fail"], door_payload["p_fb_fail"], door_payload["p_density_fail"]]
    vmax_cross = max(np.nanmax(v) for v in cross_vals)
    vmin_cross = min(0.0, min(np.nanmin(v) for v in cross_vals))
    vmax_fail = max(np.nanmax(v) for v in fail_vals)
    vmin_fail = min(0.0, min(np.nanmin(v) for v in fail_vals))
    door_marks = [("D", D_RC_, dict(marker="D", color="#444444"))]

    cross_axes, fail_axes = [], []
    im_cross_last = im_fail_last = None
    for ridx, (name, color, p_cross, p_fail) in enumerate(rows_spec):
        ax_c = fig.add_subplot(door_gs[ridx, 0])
        ax_f = fig.add_subplot(door_gs[ridx, 1])
        title_c = f"row for $a_{{cross}}$=right" if ridx == 0 else ""
        title_f = f"row for $a_{{fail}}$={a_fail_show}" if ridx == 0 else ""
        im_cross_last = draw_heatmap(ax_c, p_cross, vmin_cross, vmax_cross, cmap,
                                      title_c, door_marks, show_negative_note=(name == "FB"),
                                      identity_color=color)
        im_fail_last = draw_heatmap(ax_f, p_fail, vmin_fail, vmax_fail, cmap,
                                     title_f, door_marks, show_negative_note=(name == "FB"),
                                     identity_color=color)
        ax_c.set_ylabel(name, fontsize=9.5, color=INK)
        cross_axes.append(ax_c)
        fail_axes.append(ax_f)
    cb3 = fig.colorbar(im_cross_last, ax=cross_axes, location="right", shrink=0.85, aspect=30, pad=0.04)
    cb3.ax.tick_params(labelsize=6.6, color=MUTED, labelcolor=MUTED)
    cb3.outline.set_visible(False)
    cb4 = fig.colorbar(im_fail_last, ax=fail_axes, location="right", shrink=0.85, aspect=30, pad=0.04)
    cb4.ax.tick_params(labelsize=6.6, color=MUTED, labelcolor=MUTED)
    cb4.outline.set_visible(False)

    caption_axis(
        fig, door_group[1],
        "Successor row out of the doorway D at k=20, d=8, $\\gamma$=0.98: crossing (left column)\n"
        "vs a failed/blocked action (right column). Each column shares one color scale across\n"
        "the three rows; colored squares mark true / FB / Density identity.")

    # --- gap vs k panels ---
    gap_group = bottom_gs[1].subgridspec(2, 1, height_ratios=[1.0, 0.08])
    gap_gs = gap_group[0].subgridspec(2, 2)
    ds_panels = [4, 8, 16, 32]
    ks = sorted(set(int(r["k"]) for r in gap_rows))
    for pidx, d in enumerate(ds_panels):
        ax = fig.add_subplot(gap_gs[pidx // 2, pidx % 2])
        fb_y = [next(float(r["norm_gap"]) for r in gap_rows
                      if r["method"] == "fb" and int(r["d"]) == d and int(r["k"]) == k
                      and r["reward_type"] == "room" and float(r["gamma"]) == 0.98) for k in ks]
        density_y = [next(float(r["norm_gap"]) for r in gap_rows
                            if r["method"] == "density" and int(r["d"]) == d and int(r["k"]) == k
                            and r["reward_type"] == "room" and float(r["gamma"]) == 0.98) for k in ks]
        ax.axhline(1.0, color=TRUE_COLOR, linestyle="--", linewidth=1.6, zorder=1,
                   label="true (=1 by construction)")
        ax.axhline(0.0, color="#cfcfcf", linestyle="-", linewidth=1.0, zorder=1)
        ax.plot(ks, fb_y, color=FB_COLOR, linewidth=2.0, marker="o", markersize=6.5,
                markeredgecolor="white", markeredgewidth=1.0, label="FB", zorder=3)
        ax.plot(ks, density_y, color=DENSITY_COLOR, linewidth=2.0, marker="o", markersize=6.5,
                markeredgecolor="white", markeredgewidth=1.0, label="Density", zorder=3)
        # mark FB's zero-crossing if the sign flips between two consecutive k's
        for i in range(len(ks) - 1):
            if fb_y[i] * fb_y[i + 1] < 0:
                k0, k1 = ks[i], ks[i + 1]
                y0, y1 = fb_y[i], fb_y[i + 1]
                kx = k0 + (k1 - k0) * (0 - y0) / (y1 - y0)
                ax.scatter([kx], [0], marker="x", color=FB_COLOR, s=70, zorder=4)
                ax.annotate("FB crosses 0", (kx, 0), textcoords="offset points",
                            xytext=(4, 8), fontsize=7, color=FB_COLOR)
        ax.set_title(f"d = {d}", fontsize=9.5, color=INK)
        ax.set_xlabel("k (stickiness)", fontsize=8.3)
        ax.set_ylabel("normalized action gap", fontsize=8.3)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=7.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#cfcfcf")
        if pidx == 0:
            ax.legend(frameon=False, fontsize=7.6, loc="lower left")

    caption_axis(
        fig, gap_group[1],
        "Normalized action gap at D vs stickiness k, room reward, $\\gamma$=0.98\n"
        "(1 = matches the true gap; 0 = the fit cannot tell crossing from failing).")

    out_pdf = FIG_DIR / "fig_fourrooms_estimator.pdf"
    out_png = FIG_DIR / "fig_fourrooms_estimator.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")


# ----------------------------------------------------------------------------
# 11. Compact single-column paper figure (fig_fourrooms_main) + LaTeX caption.
# ----------------------------------------------------------------------------
def make_compact_figure(door_payload, occ_payload, gap_rows):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7.5,
        "text.color": INK,
        "axes.edgecolor": "#cfcfcf",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": False,
        "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    outer = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.0, 1.05, 0.16])

    # ---- Top row: true / FB / Density directed-occupancy heatmaps ----
    top_gs = outer[0].subgridspec(1, 3)
    occ_axes = [fig.add_subplot(top_gs[0, i]) for i in range(3)]
    p_true, p_fb, p_density, rho = (occ_payload["p_true"], occ_payload["p_fb"],
                                     occ_payload["p_density"], occ_payload["rho"])
    S0_RC_, D_RC_, G_RC_, a_star = (occ_payload["S0_RC"], occ_payload["D_RC"],
                                     occ_payload["G_RC"], occ_payload["a_star"])
    marks = [("S0", S0_RC_, dict(marker="o", color="#444444", s=22)),
             ("D", D_RC_, dict(marker="D", color="#444444", s=22)),
             ("G", G_RC_, dict(marker="*", color="#444444", s=55))]
    p_fb_clipped = np.clip(p_fb, 0.0, None)
    vmax_occ = max(np.nanmax(p_true), np.nanmax(p_fb_clipped), np.nanmax(p_density))
    for ax, vals, title, mcolor in [(occ_axes[0], p_true, "true", TRUE_COLOR),
                                     (occ_axes[1], p_fb_clipped, "FB", FB_COLOR),
                                     (occ_axes[2], p_density, "Density", DENSITY_COLOR)]:
        im = draw_heatmap_compact(ax, vals, 0.0, vmax_occ, seq_cmap(mcolor), title, marks)
    # one shared colorbar for the (shared) value scale; drawn in the Density panel's hue,
    # the same scale applies to all three panels above (they share vmin/vmax)
    cb = fig.colorbar(im, ax=occ_axes, location="right", shrink=0.85, aspect=22, pad=0.02)
    cb.ax.tick_params(labelsize=5.6, color=MUTED, labelcolor=MUTED, length=2)
    cb.outline.set_visible(False)
    cb.set_label("occ. prob. (shared scale)", fontsize=6.2, color=MUTED)

    # ---- Bottom row: left = FB/Density x a_cross/a_fail 2x2 mini grid; right = gap-vs-k ----
    bottom_gs = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.15])

    door_gs = bottom_gs[0].subgridspec(2, 2, hspace=0.28, wspace=0.12)
    a_fail_show = door_payload["a_fail_show"]
    door_marks = [("D", D_RC_, dict(marker="D", color="#444444", s=18))]
    fb_cross = np.clip(door_payload["p_fb_cross"], 0.0, None)
    fb_fail = np.clip(door_payload["p_fb_fail"], 0.0, None)
    rows_spec = [
        ("FB", fb_cross, fb_fail, FB_COLOR),
        ("Density", door_payload["p_density_cross"], door_payload["p_density_fail"], DENSITY_COLOR),
    ]
    vmax_cross = max(np.nanmax(fb_cross), np.nanmax(door_payload["p_density_cross"]))
    vmax_fail = max(np.nanmax(fb_fail), np.nanmax(door_payload["p_density_fail"]))
    im_last = None
    for ridx, (name, p_cross, p_fail, mcolor) in enumerate(rows_spec):
        ax_c = fig.add_subplot(door_gs[ridx, 0])
        ax_f = fig.add_subplot(door_gs[ridx, 1])
        title_c = "$a_{cross}$" if ridx == 0 else ""
        title_f = "$a_{fail}$" if ridx == 0 else ""
        draw_heatmap_compact(ax_c, p_cross, 0.0, vmax_cross, seq_cmap(mcolor), title_c, door_marks)
        im_last = draw_heatmap_compact(ax_f, p_fail, 0.0, vmax_fail, seq_cmap(mcolor), title_f, door_marks)
        ax_c.set_ylabel(name, fontsize=7.3, color=INK)

    ax_gap = fig.add_subplot(bottom_gs[1])
    d_show = 8
    ks = sorted(set(int(r["k"]) for r in gap_rows))
    fb_y = [next(float(r["norm_gap"]) for r in gap_rows
                  if r["method"] == "fb" and int(r["d"]) == d_show and int(r["k"]) == k
                  and r["reward_type"] == "room" and float(r["gamma"]) == 0.98) for k in ks]
    density_y = [next(float(r["norm_gap"]) for r in gap_rows
                        if r["method"] == "density" and int(r["d"]) == d_show and int(r["k"]) == k
                        and r["reward_type"] == "room" and float(r["gamma"]) == 0.98) for k in ks]
    ax_gap.axhline(1.0, color=TRUE_COLOR, linestyle="--", linewidth=1.4, zorder=1,
                    label="true")
    ax_gap.axhline(0.0, color="#cfcfcf", linestyle="-", linewidth=0.8, zorder=1)
    ax_gap.plot(ks, fb_y, color=FB_COLOR, linewidth=1.8, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.7, label="FB", zorder=3)
    ax_gap.plot(ks, density_y, color=DENSITY_COLOR, linewidth=1.8, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.7, label="Density", zorder=3)
    ax_gap.set_title(f"gap vs k (d={d_show})", fontsize=7.5, color=INK)
    ax_gap.set_xlabel("k", fontsize=6.8)
    ax_gap.set_ylabel("norm. gap", fontsize=6.8)
    ax_gap.set_xticks(ks)
    ax_gap.tick_params(labelsize=6.0, length=2)
    for spine in ["top", "right"]:
        ax_gap.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_gap.spines[spine].set_color("#cfcfcf")
    ax_gap.legend(frameon=False, fontsize=6.0, loc="lower left", handlelength=1.5)

    cap_ax = fig.add_subplot(outer[2])
    cap_ax.axis("off")
    cap_ax.text(0.0, 1.0,
                "FB negative entries (min $\\approx$ -0.001) clipped to 0 for display; "
                "colors: true = blue, FB = orange, Density = green throughout.",
                transform=cap_ax.transAxes, ha="left", va="top", fontsize=6.0, color=MUTED)

    out_pdf = FIG_DIR / "fig_fourrooms_main.pdf"
    out_png = FIG_DIR / "fig_fourrooms_main.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")

    caption = (
        "This figure shows the true, FB, and Density successor-measure rows for the "
        "directed policy from $S_0$ toward $G$ (top), the FB and Density rows for "
        "crossing versus failing to cross the sticky doorway $D$ (bottom left), and the "
        "normalized action gap at $D$ versus the stickiness parameter $k$ at rank $d=8$ "
        "(bottom right; FB's negative entries are clipped to 0 for display). "
        "A gap of 1 means the fit matches the true value difference between crossing and "
        "failing to cross; a gap of 0 means the fit cannot tell the two actions apart, and "
        "FB's gap falls toward 0 as the doorway gets stickier while Density's stays high."
    )
    cap_path = HERE / "caption_main.tex"
    with open(cap_path, "w") as f:
        f.write(caption + "\n")
    log(f"wrote {cap_path}")


def draw_heatmap_compact(ax, vals, vmin, vmax, cmap, title, mark_specs):
    """Like draw_heatmap but no min-entry note, no identity swatch, smaller marks/text
    (used by the compact single-column paper figure)."""
    g = to_grid(vals)
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.6)
    ax.set_title(title, fontsize=7.5, color=INK, pad=3)
    for label, rc, kw in mark_specs:
        r, c = rc
        ax.scatter([c], [r], zorder=5, linewidths=0.8, edgecolors="white", **kw)
    return im


# ----------------------------------------------------------------------------
# 12. Compact figure v2: full-size door maps (2 rows x 3 cols) + one gap-vs-k panel.
#     The directed-occupancy row is dropped (a near-deterministic policy piles all
#     its mass at G for every estimator, so that row carries little signal); this
#     version spends the space on the door-bottleneck comparison instead.
# ----------------------------------------------------------------------------
def truncated_blues(low=0.05, high=1.0, n=256):
    """matplotlib 'Blues', truncated to [low, high] so the light end isn't pure white
    and the dark end is pushed to a strong dark navy. One shared ramp for every
    occupancy map (true / FB / Density) -- comparison is by darkness alone."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    base = mpl.colormaps["Blues"]
    colors = base(np.linspace(low, high, n))
    cmap = mcolors.ListedColormap(colors, name="blues_trunc")
    cmap.set_bad(WALL_COLOR)
    return cmap


def draw_occ_heatmap(ax, vals, norm, cmap, title, mark_specs):
    """Like draw_heatmap_compact but takes a shared Normalize/PowerNorm object
    instead of vmin/vmax (used by the v2 restyle: one ramp, one norm per row)."""
    g = to_grid(vals)
    im = ax.imshow(g, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.6)
    ax.set_title(title, fontsize=8.0, color=INK, pad=3)
    for label, rc, kw in mark_specs:
        r, c = rc
        ax.scatter([c], [r], zorder=5, linewidths=0.9, edgecolors="white", **kw)
    return im


GAMMA_NORM = 0.6  # PowerNorm gamma shared by every restyled occupancy heatmap (v2 + splits)
DOOR_FIGURES_DIR = FIG_DIR / "fourrooms"


def door_row_values(door_payload, which):
    """which: 'cross' or 'fail'. Returns (p_true_r, p_fb_r_raw, p_fb_r_clipped,
    p_density_r, tv_fb, tv_density)."""
    p_true_r = door_payload[f"p_true_{which}"]
    p_fb_raw = door_payload[f"p_fb_{which}"]
    p_fb_clipped = np.clip(p_fb_raw, 0.0, None)
    p_density_r = door_payload[f"p_density_{which}"]
    tv_fb = tv_half(p_fb_raw, p_true_r)
    tv_density = tv_half(p_density_r, p_true_r)
    return p_true_r, p_fb_raw, p_fb_clipped, p_density_r, tv_fb, tv_density


def render_door_row(fig, row_axes, p_true_r, p_fb_r_clipped, p_density_r, tv_fb, tv_density,
                     cmap, mark_specs, show_col_titles):
    """Draws true/FB/Density-FB into the 3 given axes for one door row (a_cross or
    a_fail), all sharing one PowerNorm fixed to the true row's own range. Returns the
    mappable (for a colorbar) and the norm's vmax."""
    from matplotlib.colors import PowerNorm
    vmax_row = float(np.nanmax(p_true_r))
    norm = PowerNorm(gamma=GAMMA_NORM, vmin=0.0, vmax=vmax_row)
    col_titles = ["true", "FB", "Density-FB"]
    vals_row = [p_true_r, p_fb_r_clipped, p_density_r]
    im = None
    for cidx, (name, vals) in enumerate(zip(col_titles, vals_row)):
        ax = row_axes[cidx]
        title = name if show_col_titles else ""
        im = draw_occ_heatmap(ax, vals, norm, cmap, title, mark_specs)
        if cidx == 1:
            ax.text(0.5, -0.08, f"TV = {tv_fb:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.6, color=MUTED)
        if cidx == 2:
            ax.text(0.5, -0.08, f"TV = {tv_density:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.6, color=MUTED)
    return im, vmax_row


def _fb_argmax_flip_sentence(gap_rows):
    """One caption sentence: FB's argmax stays in the crossing cone at d=8, gamma=0.98
    (action-value experiment) even as the margin collapses, but DOES flip away from
    a_cross in the discrete 4-action sweep at the cells this cites from gap.csv."""
    flips = find_fb_argmax_flips(gap_rows, method="fb")
    room_flips = [f for f in flips if f[3] == "room"]
    if room_flips:
        cells = "; ".join(f"$\\gamma$={g}, d={d}, k={k} (norm.\\ gap {ng:.2f})"
                           for g, d, k, _, ng in room_flips)
        flip_clause = (f"In the discrete 4-action sweep (gap.csv), FB's argmax at $D$ does "
                       f"flip away from $a_{{cross}}$ for the room reward at: {cells}.")
    else:
        flip_clause = ("In the discrete 4-action sweep (gap.csv) FB's argmax never flips "
                       "away from $a_{cross}$ for the room reward at $\\gamma$=0.98.")
    return ("In the action-value experiment (Q(D,a) over the continuous 2D action plane), "
            "FB's argmax stays inside the crossing cone at d=8, $\\gamma$=0.98 for every "
            "k tested even as its margin collapses (the normalized gap falls to 0.41 by "
            f"k=40) -- the fit keeps the ranking right while losing the separation. {flip_clause}")


def find_fb_argmax_flips(gap_rows, method="fb"):
    """(gamma, d, k, reward_type, norm_gap) for every cell where this method's argmax
    at D is NOT a_cross (used to cite the flip cells in captions/notes)."""
    flips = []
    for r in gap_rows:
        if r["method"] == method and r["argmax_is_cross"] == "False":
            flips.append((float(r["gamma"]), int(r["d"]), int(r["k"]), r["reward_type"],
                          float(r["norm_gap"])))
    return sorted(flips)


def collect_gap_ci(gap_rows, d_show, gamma=0.98, reward_type="room"):
    """FB (deterministic, single value) + Density mean/CI across DENSITY_SEEDS, per k,
    at fixed d/gamma/reward_type. Returns (ks, fb_y, density_mean, density_lo,
    density_hi, ci_method)."""
    ks = sorted(set(int(r["k"]) for r in gap_rows))
    fb_y, density_mean, density_lo, density_hi = [], [], [], []
    ci_method = None
    for k in ks:
        fb_val = next(float(r["norm_gap"]) for r in gap_rows
                       if r["method"] == "fb" and int(r["d"]) == d_show and int(r["k"]) == k
                       and r["reward_type"] == reward_type and float(r["gamma"]) == gamma)
        fb_y.append(fb_val)
        density_vals = [float(r["norm_gap"]) for r in gap_rows
                         if r["method"] == "density" and int(r["d"]) == d_show and int(r["k"]) == k
                         and r["reward_type"] == reward_type and float(r["gamma"]) == gamma]
        mean, lo, hi, ci_method = interval_estimate(density_vals)
        density_mean.append(mean); density_lo.append(lo); density_hi.append(hi)
    return ks, fb_y, density_mean, density_lo, density_hi, ci_method


def draw_gap_panel(ax, d_show, gap_rows, show_legend=True, title=None, trained_rows=None):
    """FB as a single (deterministic) line; Density as mean +/- 95% CI band across
    DENSITY_SEEDS. If trained_rows (gap_trained.csv rows) is given AND d_show==TRAIN_D,
    also overlays TD-trained FB/Density as hollow markers with 95% t-intervals across
    TRAIN_SEEDS. Returns the CI method string (for the caption)."""
    ks, fb_y, dmean, dlo, dhi, ci_method = collect_gap_ci(gap_rows, d_show)
    has_trained = trained_rows is not None and d_show == TRAIN_D
    fb_label = "FB (SVD optimum)" if has_trained else "FB (deterministic)"
    density_label = "Density-FB (CE optimum)" if has_trained else "Density (mean, 95% CI)"
    ax.axhline(1.0, color=TRUE_COLOR, linestyle="--", linewidth=1.4, zorder=1, label="true")
    ax.axhline(0.0, color="#cfcfcf", linestyle="-", linewidth=0.8, zorder=1)
    ax.fill_between(ks, dlo, dhi, color=DENSITY_COLOR, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(ks, dmean, color=DENSITY_COLOR, linewidth=1.8, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7, label=density_label, zorder=3)
    ax.plot(ks, fb_y, color=FB_COLOR, linewidth=1.8, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7, label=fb_label, zorder=3)

    if has_trained:
        train_ks = sorted(set(int(r["k"]) for r in trained_rows))
        fb_td_mean, fb_td_lo, fb_td_hi = [], [], []
        density_td_mean, density_td_lo, density_td_hi = [], [], []
        for k in train_ks:
            m, lo, hi, _ = trained_gap_ci(trained_rows, "fb_td", k)
            fb_td_mean.append(m); fb_td_lo.append(m - lo); fb_td_hi.append(hi - m)
            m, lo, hi, _ = trained_gap_ci(trained_rows, "density_td", k)
            density_td_mean.append(m); density_td_lo.append(m - lo); density_td_hi.append(hi - m)
        ax.errorbar(train_ks, fb_td_mean, yerr=[fb_td_lo, fb_td_hi], fmt="D", markersize=5,
                    markerfacecolor="none", markeredgecolor=FB_COLOR, markeredgewidth=1.3,
                    ecolor=FB_COLOR, elinewidth=1.1, capsize=2.5, linestyle="none",
                    label="FB (TD-trained)", zorder=4)
        ax.errorbar(train_ks, density_td_mean, yerr=[density_td_lo, density_td_hi], fmt="D",
                    markersize=5, markerfacecolor="none", markeredgecolor=DENSITY_COLOR,
                    markeredgewidth=1.3, ecolor=DENSITY_COLOR, elinewidth=1.1, capsize=2.5,
                    linestyle="none", label="Density-FB (TD-trained)", zorder=4)

    ax.set_title(title or f"gap vs k (d={d_show})", fontsize=7.5, color=INK)
    ax.set_xlabel("k", fontsize=6.8)
    ax.set_ylabel("norm. gap", fontsize=6.8)
    ax.set_xticks(ks)
    ax.tick_params(labelsize=6.0, length=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cfcfcf")
    if show_legend:
        ax.legend(frameon=False, fontsize=5.2, loc="lower left", handlelength=1.5)
    return ci_method


def make_compact_figure_v2(door_payload, gap_rows, trained_rows=None):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7.5,
        "text.color": INK,
        "axes.edgecolor": "#cfcfcf",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": False,
        "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    cmap = truncated_blues(low=0.05, high=1.0)
    door_marks = [("D", D_RC, dict(marker="D", color="#444444", s=22))]
    seed_used = door_payload.get("density_seed_used", 0)

    rows_spec = [
        ("cross the door", "cross"),
        ("fail to cross", "fail"),
    ]

    fig = plt.figure(figsize=(8.6, 4.5), constrained_layout=True)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.30])
    main_gs = outer[0].subgridspec(1, 2, width_ratios=[1.9, 1.0])
    door_gs = main_gs[0].subgridspec(2, 3, hspace=0.16, wspace=0.10)

    for ridx, (row_title, which) in enumerate(rows_spec):
        p_true_r, _, p_fb_clipped, p_density_r, tv_fb, tv_density = door_row_values(door_payload, which)
        row_axes = [fig.add_subplot(door_gs[ridx, cidx]) for cidx in range(3)]
        im, _ = render_door_row(fig, row_axes, p_true_r, p_fb_clipped, p_density_r, tv_fb, tv_density,
                                 cmap, door_marks, show_col_titles=(ridx == 0))
        row_axes[0].set_ylabel(row_title, fontsize=7.8, color=INK)
        cb = fig.colorbar(im, ax=row_axes, location="right", shrink=0.85, aspect=16, pad=0.015)
        cb.ax.tick_params(labelsize=5.4, color=MUTED, labelcolor=MUTED, length=2)
        cb.outline.set_visible(False)
        cb.set_label("occupancy mass", fontsize=6.0, color=MUTED)

    ax_gap = fig.add_subplot(main_gs[1])
    ci_method = draw_gap_panel(ax_gap, 8, gap_rows, trained_rows=trained_rows)

    cap_ax = fig.add_subplot(outer[1])
    cap_ax.axis("off")
    cap_ax.text(0.0, 1.0,
                f"Heatmaps: one shared truncated-Blues ramp (light=little mass, dark navy="
                f"concentrated), PowerNorm(gamma={GAMMA_NORM}),\n"
                "each row's scale fixed to the true row's own range; FB negative entries "
                "(min $\\approx$ -0.001) clipped to 0 for display;\n"
                f"door maps use Density seed {seed_used}. Gap panel: FB is deterministic "
                "(rank-d SVD, single line); Density is the mean\n"
                f"over 5 seeds with a shaded 95% CI ({ci_method}). "
                "Colors: true = blue, FB = orange, Density = green.",
                transform=cap_ax.transAxes, ha="left", va="top", fontsize=5.8, color=MUTED,
                linespacing=1.5)

    out_pdf = FIG_DIR / "fig_fourrooms_main_v2.pdf"
    out_png = FIG_DIR / "fig_fourrooms_main_v2.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")

    ci_method_tex = ci_method.replace("%", r"\%")
    caption = (
        "This figure shows the true, FB, and Density-FB successor-measure rows out of the "
        "sticky doorway $D$ at k=20, d=8, $\\gamma$=0.98 (door maps use Density seed "
        f"{seed_used}): crossing the door (top row) versus failing to cross (bottom row), "
        "each cell shaded by one shared truncated-Blues colormap (light = little occupancy "
        "mass, dark navy = concentrated mass, identical for true/FB/Density-FB) under a "
        f"PowerNorm(gamma={GAMMA_NORM}) fixed per row to the true row's own range, so "
        "darkness alone carries the comparison; total-variation distance to the true row is "
        "printed under each fit (FB's negative entries, min $\\approx$ -0.001, are clipped to "
        "0 for display and for that TV number), plus the normalized action gap at $D$ versus "
        "the stickiness parameter $k$ at rank $d=8$ (right; true = blue dashed, FB = orange "
        "deterministic line, Density = green mean line with a shaded 95\\% CI band across 5 "
        f"independently seeded fits, {ci_method_tex}). "
        "A gap of 1 means the fit matches the true value difference between crossing and "
        "failing to cross; a gap of 0 means the fit cannot tell the two actions apart, and "
        "FB's gap falls toward 0 as the doorway gets stickier while Density's stays close to "
        "the truth, matching the larger TV distance and paler, more spread-out darkness "
        "pattern already visible for FB at k=20 on the left. "
        "Both fits see exactly the same successor-measure targets and neither uses TD "
        "training or an actor: FB's occupancies are the closed-form rank-$d$ optimum of "
        "its squared loss (the truncated SVD of $m/\\rho$ in the $\\rho$-weighted norm), "
        "while Density-FB's are a rank-$d$ softmax model fit by cross-entropy. "
        + _fb_argmax_flip_sentence(gap_rows)
    )
    cap_path = HERE / "caption_main_v2.tex"
    with open(cap_path, "w") as f:
        f.write(caption + "\n")
    log(f"wrote {cap_path}")


# ----------------------------------------------------------------------------
# 13. Split subfigures: maps_cross, maps_fail, gap_vs_k, gap_vs_k_all_d + subfigures.tex
# ----------------------------------------------------------------------------
def save_maps_figure(door_payload, which, out_stem):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
        "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    cmap = truncated_blues(low=0.05, high=1.0)
    door_marks = [("D", D_RC, dict(marker="D", color="#444444", s=24))]
    seed_used = door_payload.get("density_seed_used", 0)
    p_true_r, _, p_fb_clipped, p_density_r, tv_fb, tv_density = door_row_values(door_payload, which)

    fig = plt.figure(figsize=(6.4, 2.5), constrained_layout=True)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.16])
    row_gs = outer[0].subgridspec(1, 3, wspace=0.10)
    row_axes = [fig.add_subplot(row_gs[0, c]) for c in range(3)]
    im, _ = render_door_row(fig, row_axes, p_true_r, p_fb_clipped, p_density_r, tv_fb, tv_density,
                             cmap, door_marks, show_col_titles=True)
    cb = fig.colorbar(im, ax=row_axes, location="right", shrink=0.9, aspect=14, pad=0.02)
    cb.ax.tick_params(labelsize=5.6, color=MUTED, labelcolor=MUTED, length=2)
    cb.outline.set_visible(False)
    cb.set_label("occupancy mass", fontsize=6.2, color=MUTED)

    cap_ax = fig.add_subplot(outer[1])
    cap_ax.axis("off")
    label = "crossing the door" if which == "cross" else "failing to cross"
    cap_ax.text(0.0, 1.0,
                f"k=20, d=8, $\\gamma$=0.98, {label}; shared truncated-Blues ramp,\n"
                f"PowerNorm(gamma={GAMMA_NORM}) fixed to the true row's range; FB negatives "
                f"clipped to 0; Density uses seed {seed_used}.",
                transform=cap_ax.transAxes, ha="left", va="top", fontsize=6.0, color=MUTED,
                linespacing=1.5)

    DOOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = DOOR_FIGURES_DIR / f"{out_stem}.pdf"
    out_png = DOOR_FIGURES_DIR / f"{out_stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")
    return tv_fb, tv_density


CONTRAST_POWER = 0.5  # symmetric power scaling sign(x)*|x|^p applied to every contrast map


def _sympow(x, p=CONTRAST_POWER):
    return np.sign(x) * np.abs(x) ** p


def _sympow_inv(y, p=CONTRAST_POWER):
    return np.sign(y) * np.abs(y) ** (1.0 / p)


def draw_diverging_heatmap_sympow(ax, vals, vmax_abs, cmap, title, mark_specs, power=CONTRAST_POWER):
    """Diverging heatmap under a symmetric power norm sign(x)|x|^p (p<1 lifts small
    values off of white so a near-zero room-side isn't visually indistinguishable from
    an exactly-zero one); the colorbar this feeds is relabeled back to original units."""
    g = to_grid(vals)
    g_t = _sympow(g, power)
    vmax_t = _sympow(vmax_abs, power)
    im = ax.imshow(g_t, cmap=cmap, vmin=-vmax_t, vmax=vmax_t, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.6)
    ax.set_title(title, fontsize=8.0, color=INK, pad=3)
    for label, rc, kw in mark_specs:
        r, c = rc
        ax.scatter([c], [r], zorder=5, linewidths=0.9, edgecolors="white", **kw)
    return im


def save_contrast_figure(c_true, c_fb, c_density, seed_used, out_stem, k_show=20, extra_note=""):
    """The action contrast at D: (1-gamma)(M(D,a_cross,.) - M(D,a_fail,.)), one shared
    symmetric diverging scale (fixed to the true panel's max |value|, under a symmetric
    power norm) across true/FB/Density-FB. Shared by the closed-form and TD-trained
    contrast figures. Returns (ratio_fb, ratio_density) = |Delta|/|Delta_true|."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
        "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    cmap = mpl.colormaps["RdBu"].copy()
    cmap.set_bad(WALL_COLOR)
    door_marks = [("D", D_RC, dict(marker="D", color="#444444", s=24))]

    vmax_abs = float(np.max(np.abs(c_true)))
    l1_true = float(np.sum(np.abs(c_true)))
    ratio_fb = float(np.sum(np.abs(c_fb))) / l1_true
    ratio_density = float(np.sum(np.abs(c_density))) / l1_true

    fig = plt.figure(figsize=(6.4, 2.5), constrained_layout=True)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.16])
    row_gs = outer[0].subgridspec(1, 3, wspace=0.10)
    row_axes = [fig.add_subplot(row_gs[0, c]) for c in range(3)]
    col_titles = ["true", "FB", "Density-FB"]
    vals_row = [c_true, c_fb, c_density]
    im = None
    for cidx, (name, vals) in enumerate(zip(col_titles, vals_row)):
        im = draw_diverging_heatmap_sympow(row_axes[cidx], vals, vmax_abs, cmap, name, door_marks)
    row_axes[1].text(0.5, -0.08, f"|Δ| / |Δ_true| = {ratio_fb:.2f}",
                      transform=row_axes[1].transAxes, ha="center", va="top",
                      fontsize=6.6, color=MUTED)
    row_axes[2].text(0.5, -0.08, f"|Δ| / |Δ_true| = {ratio_density:.2f}",
                      transform=row_axes[2].transAxes, ha="center", va="top",
                      fontsize=6.6, color=MUTED)
    cb = fig.colorbar(im, ax=row_axes, location="right", shrink=0.9, aspect=14, pad=0.02)
    # relabel ticks from the sign(x)|x|^p display space back to original occupancy units
    ticks_t = np.asarray(cb.get_ticks())
    ticks_orig = _sympow_inv(ticks_t)
    cb.set_ticks(ticks_t)
    cb.set_ticklabels([f"{v:.3f}" for v in ticks_orig])
    cb.ax.tick_params(labelsize=5.6, color=MUTED, labelcolor=MUTED, length=2)
    cb.outline.set_visible(False)
    cb.set_label("occupancy difference (cross - fail)", fontsize=6.2, color=MUTED)

    cap_ax = fig.add_subplot(outer[1])
    cap_ax.axis("off")
    cap_ax.text(0.0, 1.0,
                f"k={k_show}, d=8, $\\gamma$=0.98; (1-$\\gamma$)(M(D,a$_{{cross}}$,.) - "
                "M(D,a$_{fail}$,.)); blue = more mass after crossing, red = more mass\n"
                f"after failing; display uses sign(x)|x|^{CONTRAST_POWER} per panel (one "
                "shared scale, fixed to $\\pm$|true|'s max; colorbar ticks show original "
                f"values); Density uses seed {seed_used}.{extra_note}",
                transform=cap_ax.transAxes, ha="left", va="top", fontsize=6.0, color=MUTED,
                linespacing=1.5)

    DOOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = DOOR_FIGURES_DIR / f"{out_stem}.pdf"
    out_png = DOOR_FIGURES_DIR / f"{out_stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")
    return ratio_fb, ratio_density


def save_maps_contrast_figure(door_payload):
    seed_used = door_payload.get("density_seed_used", 0)
    c_true = door_payload["p_true_cross"] - door_payload["p_true_fail"]
    c_fb = door_payload["p_fb_cross"] - door_payload["p_fb_fail"]  # raw (unclipped); a
    # signed contrast is exactly where FB's negative entries are meaningful, not an error
    c_density = door_payload["p_density_cross"] - door_payload["p_density_fail"]
    return save_contrast_figure(c_true, c_fb, c_density, seed_used, "maps_contrast", k_show=20)


def save_gap_vs_k_figure(gap_rows, trained_rows=None):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
        "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.4, 3.2), constrained_layout=True)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.34])
    main_gs = outer[0].subgridspec(1, 2, width_ratios=[1.0, 0.62])
    ax = fig.add_subplot(main_gs[0])
    ci_method = draw_gap_panel(ax, 8, gap_rows, title="gap vs k (d=8)", trained_rows=trained_rows,
                                show_legend=False)
    leg_ax = fig.add_subplot(main_gs[1])
    leg_ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    leg_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=6.6,
                  handlelength=1.6, borderaxespad=0.0)

    cap_ax = fig.add_subplot(outer[1])
    cap_ax.axis("off")
    trained_note = (" Hollow diamonds: TD-trained (20k Adam steps, mean +/- 95% CI\n"
                     "across 5 seeds each; FB-TD is not a closed form)."
                     if trained_rows is not None else "")
    cap_ax.text(0.0, 1.0,
                f"Room reward, $\\gamma$=0.98. FB: deterministic rank-8 SVD.\n"
                f"Density: mean over 5 independently seeded fits, shaded 95% CI\n"
                f"({ci_method}).{trained_note}",
                transform=cap_ax.transAxes, ha="left", va="top", fontsize=5.6, color=MUTED,
                linespacing=1.5)

    DOOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = DOOR_FIGURES_DIR / "gap_vs_k.pdf"
    out_png = DOOR_FIGURES_DIR / "gap_vs_k.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")
    return ci_method


def save_gap_vs_k_all_d_figure(gap_rows, trained_rows=None):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
        "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.6, 4.6), constrained_layout=True)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.16])
    grid = outer[0].subgridspec(2, 2, hspace=0.30, wspace=0.30)
    ci_method = None
    for pidx, d in enumerate([4, 8, 16, 32]):
        ax = fig.add_subplot(grid[pidx // 2, pidx % 2])
        ci_method = draw_gap_panel(ax, d, gap_rows, show_legend=(pidx == 0), title=f"d = {d}",
                                    trained_rows=trained_rows)

    cap_ax = fig.add_subplot(outer[1])
    cap_ax.axis("off")
    trained_note = (" Hollow diamonds on d=8 (only rank trained): TD-trained mean +/- 95% CI, 5 seeds."
                     if trained_rows is not None else "")
    cap_ax.text(0.0, 1.0,
                f"Room reward, $\\gamma$=0.98, k in {{1,5,10,20,40}}. FB: deterministic "
                f"rank-d SVD.\nDensity: mean over 5 independently seeded fits, shaded 95% CI "
                f"({ci_method}).{trained_note}",
                transform=cap_ax.transAxes, ha="left", va="top", fontsize=6.4, color=MUTED,
                linespacing=1.5)

    out_pdf = DOOR_FIGURES_DIR / "gap_vs_k_all_d.pdf"
    out_png = DOOR_FIGURES_DIR / "gap_vs_k_all_d.png"
    DOOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")
    return ci_method


def write_subfigures_tex(ci_method, ratio_fb=None, ratio_density=None):
    ci_method_tex = ci_method.replace("%", r"\%")
    ratio_line = ""
    if ratio_fb is not None and ratio_density is not None:
        ratio_line = (rf" ($|\Delta|/|\Delta_{{\mathrm{{true}}}}|$: FB {ratio_fb:.2f}, "
                      rf"Density-FB {ratio_density:.2f})")
    tex = r"""% Auto-generated by analysis/fourrooms_estimator/run.py -- do not hand-edit.
% Requires \usepackage{subcaption} (or graphicx + subcaption's \subcaptionbox).
\begin{figure}[t]
  \centering
  \begin{subcaptionbox}{Action contrast at $D$: crossing minus failing.\label{fig:fourrooms-maps-contrast}}
    {\includegraphics[width=0.75\linewidth]{figures/fourrooms/maps_contrast.pdf}}
  \end{subcaptionbox}
  \\[0.5em]
  \begin{subcaptionbox}{Crossing the door ($a_{\mathrm{cross}}$).\label{fig:fourrooms-maps-cross}}
    {\includegraphics[width=0.75\linewidth]{figures/fourrooms/maps_cross.pdf}}
  \end{subcaptionbox}
  \\[0.5em]
  \begin{subcaptionbox}{Failing to cross ($a_{\mathrm{fail}}$).\label{fig:fourrooms-maps-fail}}
    {\includegraphics[width=0.75\linewidth]{figures/fourrooms/maps_fail.pdf}}
  \end{subcaptionbox}
  \\[0.5em]
  \begin{subcaptionbox}{Normalized action gap at $D$ vs.\ stickiness $k$, rank $d=8$.\label{fig:fourrooms-gap-vs-k}}
    {\includegraphics[width=0.5\linewidth]{figures/fourrooms/gap_vs_k.pdf}}
  \end{subcaptionbox}
  \caption{FourRooms sticky-doorway comparison: the action contrast $(1{-}\gamma)(M(D,a_{\mathrm{cross}},\cdot) - M(D,a_{\mathrm{fail}},\cdot))$
    (top; diverging scale, blue = more mass after crossing, red = more mass after failing"""\
    + ratio_line + r"""),
    the true, FB, and Density-FB successor-measure
    rows for crossing and failing to cross the door $D$ at $k{=}20$,
    $d{=}8$, $\gamma{=}0.98$ (one shared truncated-Blues color scale per row, fixed to the
    true row's range; FB's negative entries are clipped to 0 for display), and the
    normalized action gap at $D$ versus $k$ at $d{=}8$ (bottom; FB is the deterministic
    rank-$d$ SVD fit, Density is the mean over 5 independently seeded fits with a shaded
    95\% CI, """ + ci_method_tex + r"""). See also
    \texttt{figures/fourrooms/gap\_vs\_k\_all\_d.pdf} for the same gap panel at every
    rank $d \in \{4,8,16,32\}$ (appendix).}
  \label{fig:fourrooms-subfigures}
\end{figure}
"""
    tex_path = HERE / "subfigures.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    log(f"wrote {tex_path}")


# ----------------------------------------------------------------------------
# 14. fig_toy_main: a compact figure that visually echoes fig_lead_combined.png
#     (Figure 1) -- smooth blurred occupancy blobs, FB in blue, Density-FB in
#     orange-red -- with the ground truth added, plus a rank-correlation panel.
# ----------------------------------------------------------------------------
def smooth_upsample(vals, factor=8, sigma_cells=0.35):
    """11x11 (vals over FREE_CELLS, wall-aware) -> smooth (11*factor)x(11*factor)
    array, NaN at walls. Bilinear upsample then Gaussian blur (sigma in units of
    ORIGINAL cells), matching Figure 1's soft blurred-blob look. sigma_cells is kept
    small (~0.35) so a sharp peak at a single cell (the door) is not smoothed away."""
    from scipy.ndimage import zoom, gaussian_filter
    g = to_grid(vals)
    wall = np.isnan(g)
    g_filled = np.nan_to_num(g, nan=0.0)
    g_hi = zoom(g_filled, factor, order=1)
    g_hi = gaussian_filter(g_hi, sigma=sigma_cells * factor)
    wall_hi = zoom(wall.astype(np.float64), factor, order=0) > 0.5
    return np.where(wall_hi, np.nan, g_hi)


def door_mass_fraction(vals_raw):
    """Fraction of this row's total mass sitting on the door cell itself."""
    total = float(np.sum(vals_raw))
    return float(vals_raw[D] / total) if total > 0 else 0.0


def draw_toy_occupancy_panel(ax, vals_raw, cmap_name, title, vmax=None, show_door_frac=False):
    """One Figure-1-style smooth occupancy panel, no ticks/gridlines, light-grey
    frame, walls in dark grey, black dot at D, gold star at the deep target cell
    G. If vmax is given it is used as-is (for sharing a scale across panels);
    otherwise the panel is normalized to its own max."""
    import matplotlib as mpl
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(WALL_COLOR)
    g_hi = smooth_upsample(vals_raw)
    if vmax is None:
        vmax = float(np.nanmax(g_hi))
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(g_hi, cmap=cmap, vmin=0.0, vmax=vmax, extent=[-0.5, 10.5, 10.5, -0.5],
                    origin="upper", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.8)
    ax.set_title(title, fontsize=8.0, color=INK, pad=3)
    Dr, Dc = D_RC
    Gr, Gc = G_RC
    ax.scatter([Dc], [Dr], s=42, facecolor="black", edgecolor="white", linewidth=1.0, zorder=6)
    ax.scatter([Gc], [Gr], marker="*", s=170, facecolor="#f5c518", edgecolor="black",
               linewidth=0.8, zorder=6)
    if show_door_frac:
        frac = door_mass_fraction(vals_raw)
        ax.text(0.5, -0.05, f"door: {frac:.2f}", transform=ax.transAxes, ha="center",
                va="top", fontsize=6.2, color=MUTED)
    return im


TOY_FB_BLUE = "#2171b5"       # Figure 1's FB hue family (Blues, dark end)
TOY_DENSITY_ORANGE = "#e34a33"  # Figure 1's Density hue family (OrRd, dark end)
TOY_TRUE_GREY = "#4d4d4d"


def save_toy_main_figure(gap_rows, trained_rows):
    """PAPER/ICLR/figures/fourrooms/fig_toy_main.{pdf,png}: LEFT a 2x3 block of
    smooth Figure-1-style occupancy maps (rows = cross/fail, cols = true/FB/
    Density-FB) at k=20,d=8,gamma=0.98,seed0; RIGHT the normalized crossing-vs-
    failing value margin at D versus k (the gap_vs_k curve, recolored to match
    the map hues: FB blue, Density-FB orange-red)."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
        "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "svg.fonttype": "none",
    })
    import matplotlib.pyplot as plt

    # ---- LEFT: recompute the closed-form cross/fail rows at k=20,d=8,gamma=0.98 ----
    gamma, k, d = 0.98, 20, 8
    P_k = sticky_action_matrices(k)
    P_pi = policy_matrix_uniform(P_k)
    rho, _, _ = stationary(P_pi)
    M = successor_measure(P_k, P_pi, gamma)
    row_cross = D * A + ACTIONS.index(A_CROSS)
    row_fail = D * A + ACTIONS.index("left")
    p_true_cross = (1 - gamma) * M[row_cross]
    p_true_fail = (1 - gamma) * M[row_fail]

    F_fb, B_fb = fb_fit(M, rho, d)
    p_fb_cross = np.clip((1 - gamma) * fb_row(F_fb, B_fb, row_cross) * rho, 0.0, None)
    p_fb_fail = np.clip((1 - gamma) * fb_row(F_fb, B_fb, row_fail) * rho, 0.0, None)

    m_target = (1 - gamma) * M
    F_d, B_d, _ = density_fit(m_target, d, seed=0, restarts=1)
    Q_d_all, p_all = density_readout(F_d, B_d, R_ROOM, gamma)
    p_density_cross = p_all[row_cross]
    p_density_fail = p_all[row_fail]

    # ---- figure ----
    fig = plt.figure(figsize=(7.0, 3.6), constrained_layout=True)
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.55, 1.0])

    left_gs = outer[0].subgridspec(2, 3, hspace=0.12, wspace=0.08)
    row_labels = ["cross the door", "fail to cross"]
    col_specs = [("true", "Greys"), ("FB", "Blues"), ("Density-FB", "OrRd")]
    # rows_by_col[cidx] = [vals_cross, vals_fail]
    rows_by_col = [[p_true_cross, p_true_fail], [p_fb_cross, p_fb_fail],
                   [p_density_cross, p_density_fail]]
    show_door_frac = True  # numeric backup: fraction of mass on the door cell itself
    for cidx, ((name, cmap_name), (v_cross, v_fail)) in enumerate(zip(col_specs, rows_by_col)):
        # color scale shared WITHIN this column, across the cross/fail rows
        vmax_col = max(float(np.nanmax(smooth_upsample(v_cross))),
                       float(np.nanmax(smooth_upsample(v_fail))))
        for ridx, vals in enumerate([v_cross, v_fail]):
            ax = fig.add_subplot(left_gs[ridx, cidx])
            title = name if ridx == 0 else ""
            draw_toy_occupancy_panel(ax, vals, cmap_name, title, vmax=vmax_col,
                                      show_door_frac=show_door_frac)
            if cidx == 0:
                ax.set_ylabel(row_labels[ridx], fontsize=7.8, color=INK)

    # ---- RIGHT: normalized crossing-vs-failing margin at D versus k ----
    ax4 = fig.add_subplot(outer[1])
    ks, fb_y, dmean, dlo, dhi, ci_method = collect_gap_ci(gap_rows, TRAIN_D)
    ax4.axhline(1.0, color="#9a9a9a", linestyle="--", linewidth=1.4, zorder=1, label="true")
    ax4.axhline(0.0, color="#cfcfcf", linestyle="-", linewidth=0.8, zorder=1)
    ax4.fill_between(ks, dlo, dhi, color=TOY_DENSITY_ORANGE, alpha=0.20, linewidth=0, zorder=2)
    ax4.plot(ks, dmean, color=TOY_DENSITY_ORANGE, linewidth=1.8, marker="o", markersize=4.5,
             markeredgecolor="white", markeredgewidth=0.7, label="Density-FB (CE optimum)", zorder=3)
    ax4.plot(ks, fb_y, color=TOY_FB_BLUE, linewidth=1.8, marker="o", markersize=4.5,
             markeredgecolor="white", markeredgewidth=0.7, label="FB (SVD optimum)", zorder=3)

    train_ks = sorted(set(int(r["k"]) for r in trained_rows))
    fb_td_mean, fb_td_lo, fb_td_hi = [], [], []
    density_td_mean, density_td_lo, density_td_hi = [], [], []
    for kk in train_ks:
        m, lo, hi, _ = trained_gap_ci(trained_rows, "fb_td", kk)
        fb_td_mean.append(m); fb_td_lo.append(m - lo); fb_td_hi.append(hi - m)
        m, lo, hi, _ = trained_gap_ci(trained_rows, "density_td", kk)
        density_td_mean.append(m); density_td_lo.append(m - lo); density_td_hi.append(hi - m)
    ax4.errorbar(train_ks, fb_td_mean, yerr=[fb_td_lo, fb_td_hi], fmt="D", markersize=5,
                 markerfacecolor="none", markeredgecolor=TOY_FB_BLUE, markeredgewidth=1.3,
                 ecolor=TOY_FB_BLUE, elinewidth=1.1, capsize=2.5, linestyle="none",
                 label="FB (TD-trained)", zorder=4)
    ax4.errorbar(train_ks, density_td_mean, yerr=[density_td_lo, density_td_hi], fmt="D",
                 markersize=5, markerfacecolor="none", markeredgecolor=TOY_DENSITY_ORANGE,
                 markeredgewidth=1.3, ecolor=TOY_DENSITY_ORANGE, elinewidth=1.1, capsize=2.5,
                 linestyle="none", label="Density-FB (TD-trained)", zorder=4)

    ax4.set_title("does the critic prefer crossing?", fontsize=8.2, color=INK)
    ax4.set_xlabel("k (stickiness)", fontsize=7.4)
    ax4.set_ylabel("normalized margin", fontsize=7.4)
    ax4.set_xticks(ks)
    ax4.tick_params(labelsize=6.4, length=2)
    for spine in ["top", "right"]:
        ax4.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax4.spines[spine].set_color("#cfcfcf")
    ax4.legend(frameon=False, fontsize=5.4, loc="lower left", handlelength=1.4)

    DOOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = DOOR_FIGURES_DIR / "fig_toy_main.pdf"
    out_png = DOOR_FIGURES_DIR / "fig_toy_main.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    log(f"wrote {out_pdf}")
    log(f"wrote {out_png}")
    return dict(ci_method=ci_method)


def write_toy_caption_tex(summary=None):
    caption = (
        "At the door $D$, the true futures of crossing and failing to cross differ "
        "sharply: crossing spreads probability mass into the next room while failing "
        "leaves the agent stuck at the doorway. "
        "FB's fit assigns both actions a nearly identical, diffuse future, so its "
        "critic barely prefers the precise crossing action over a failed one, and this "
        "margin collapses further as the failed crossing becomes more persistent. "
        "Density-FB keeps the two futures visually and numerically distinct, and keeps "
        "the margin between them close to the truth at every stickiness level tested. "
        "Each column shares one color scale between its cross and fail rows (fixed to "
        "the max of the two, per estimator), so the difference between the rows is a "
        "genuine change in the estimator's occupancy prediction, not a rescaling artifact."
    )
    cap_path = HERE / "caption_toy_main.tex"
    with open(cap_path, "w") as f:
        f.write(caption + "\n")
    log(f"wrote {cap_path}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    log("=== FourRooms FB-vs-Density estimator demo ===")
    log(f"|S| = {S} free cells, D (bottleneck) = {D_RC}, "
        f"a_cross = {A_CROSS}, a_fail = {A_FAIL}, G (deep target) = {G_RC}, S0 = {S0_RC}")

    run_sanity_checks()
    gap_rows, door_payload = run_main_sweep()
    occ_rows, occ_payload = run_occupancy_experiment()

    if os.environ.get("FOURROOMS_CACHE_FOR_FIGURE_ITERATION"):
        import pickle
        with open(os.environ["FOURROOMS_CACHE_FOR_FIGURE_ITERATION"], "wb") as f:
            pickle.dump((door_payload, occ_payload, gap_rows), f)

    make_figure(door_payload, occ_payload, gap_rows)
    make_compact_figure(door_payload, occ_payload, gap_rows)

    # --- TD-trained tabular agents (fixed pi, no actor, no z) ---
    trained_rows, door_payload_trained = run_trained_sweep()
    for k_show in (1, 20):
        m_fb, lo_fb, hi_fb, _ = trained_gap_ci(trained_rows, "fb_td", k_show)
        m_d, lo_d, hi_d, _ = trained_gap_ci(trained_rows, "density_td", k_show)
        log(f"[trained gap] k={k_show}: FB-TD={m_fb:.3f} [{lo_fb:.3f},{hi_fb:.3f}]  "
            f"Density-TD={m_d:.3f} [{lo_d:.3f},{hi_d:.3f}]")

    make_compact_figure_v2(door_payload, gap_rows, trained_rows=trained_rows)

    save_maps_figure(door_payload, "cross", "maps_cross")
    save_maps_figure(door_payload, "fail", "maps_fail")
    ratio_fb, ratio_density = save_maps_contrast_figure(door_payload)
    log(f"[maps_contrast] |Delta|/|Delta_true|: FB={ratio_fb:.3f} Density(seed0)={ratio_density:.3f}")

    ratio_fb_t, ratio_density_t = save_contrast_figure(
        door_payload_trained["p_true_cross"] - door_payload_trained["p_true_fail"],
        door_payload_trained["p_fb_cross"] - door_payload_trained["p_fb_fail"],
        door_payload_trained["p_density_cross"] - door_payload_trained["p_density_fail"],
        door_payload_trained["density_seed_used"], "maps_contrast_trained", k_show=20,
        extra_note=" TD-trained (20k Adam steps), not closed-form.")
    log(f"[maps_contrast_trained] |Delta|/|Delta_true|: FB-TD={ratio_fb_t:.3f} "
        f"Density-TD(seed0)={ratio_density_t:.3f}")

    save_gap_vs_k_figure(gap_rows, trained_rows=trained_rows)
    ci_method_all_d = save_gap_vs_k_all_d_figure(gap_rows, trained_rows=trained_rows)
    write_subfigures_tex(ci_method_all_d, ratio_fb, ratio_density)

    save_toy_main_figure(gap_rows, trained_rows)
    write_toy_caption_tex()

    # --- Action-value heatmap at the bottleneck (2D continuous action space) ---
    av_results = run_action_value_door_experiment()
    av_argmax_k = save_action_value_figure(av_results, [1, 5, 10, 20, 40], "k",
                                            "action_value_door", d_fixed=8)
    av_argmax_d = save_action_value_figure(av_results, [2, 4, 8, 16, 32], "d",
                                            "action_value_door_d", k_fixed=20)
    av_argmax_shared = save_action_value_figure(av_results, [1, 5, 10, 20, 40], "k",
                                                 "action_value_door_shared", d_fixed=8,
                                                 column_shared_scale=True)
    log(f"[action_value_door] argmax-in-cone by k (d=8): {av_argmax_k}")
    log(f"[action_value_door_d] argmax-in-cone by d (k=20): {av_argmax_d}")
    log(f"[action_value_door_shared] argmax-in-cone by k (d=8): {av_argmax_shared}")
    for k_show in [1, 5, 10, 20, 40]:
        res = av_results[(k_show, 8)]
        log(f"[action_value_door] k={k_show} d=8: norm_gap_fb={res['norm_gap_fb']:.3f} "
            f"norm_gap_density(seed0)={res['norm_gap_density_seed0']:.3f} "
            f"density_mean={np.mean(res['density_norm_gaps']):.3f}")

    log(f"\nTotal wall time: {time.time()-t_start:.1f}s")
    print("\n--- sanity/log summary (paste into README/notes) ---")
    print("\n".join(LOG_LINES))
