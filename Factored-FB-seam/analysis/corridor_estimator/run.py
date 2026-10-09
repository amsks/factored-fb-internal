"""
Corridor-with-exit tabular demonstration: FB's squared-loss rank-d successor-measure
fit vs a normalized (softmax / cross-entropy) rank-d fit, at a sticky corridor exit.

A straight corridor of 8 free cells (1 cell wide) leads into a 5x5 room, joining the
room at the middle of its left wall. The bottleneck D is the LAST corridor cell before
the room (the exit). Everything is the FourRooms script's machinery
(analysis/fourrooms_estimator/run.py) on the new grid: exact successor measure, FB's
closed-form rho-weighted truncated-SVD fit, the Density rank-d softmax cross-entropy
fit (5 seeds), and TD-trained tabular versions of both (5 seeds, 20k steps, 100k-step
random-policy dataset).

Run:
    JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python analysis/corridor_estimator/run.py

Outputs (all under this directory unless noted):
    gap.csv          - closed-form sweep, k x d at the exit D (gamma = 0.98, room reward)
    gap_trained.csv  - TD-trained FB / Density-FB, k x seed, d = 8
    caption.tex      - LaTeX caption for the paper figure
    PAPER/ICLR/figures/corridor/fig_toy_corridor.{pdf,svg,png}  (+ maps_exit, maps_fail,
    margin_vs_k subfigures)
"""

import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--corridor14" in sys.argv:          # longer-corridor variants (see run_corridor14)
    os.environ["CORRIDOR_LEN"] = "14"

import csv
import functools
import pathlib
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE.parent.parent / "PAPER" / "ICLR" / "figures" / "corridor"
LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(str(msg))


# ----------------------------------------------------------------------------
# 1. Environment: 8-cell corridor + 5x5 room, 1-cell wall border.
#    Grid is 7 rows x 15 cols. Corridor = row 3, cols 1..8; room = rows 1..5,
#    cols 9..13; everything else is wall.
# ----------------------------------------------------------------------------
CORRIDOR_LEN = int(os.environ.get("CORRIDOR_LEN", "8"))   # 8 (main) or 14 (--corridor14)
ROOM_N = 5
GRID_R, GRID_C = 7, CORRIDOR_LEN + ROOM_N + 2
CORRIDOR_ROW = 3
ROOM_ROWS = range(1, 1 + ROOM_N)          # 1..5
ROOM_COLS = range(1 + CORRIDOR_LEN, 1 + CORRIDOR_LEN + ROOM_N)   # 9..13
ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
A = len(ACTIONS)


def build_grid():
    wall = np.ones((GRID_R, GRID_C), dtype=bool)
    for c in range(1, 1 + CORRIDOR_LEN):
        wall[CORRIDOR_ROW, c] = False
    for r in ROOM_ROWS:
        for c in ROOM_COLS:
            wall[r, c] = False
    free_cells = [(r, c) for r in range(GRID_R) for c in range(GRID_C) if not wall[r, c]]
    cell_to_idx = {rc: i for i, rc in enumerate(free_cells)}
    return wall, free_cells, cell_to_idx


WALL, FREE_CELLS, CELL_TO_IDX = build_grid()
S = len(FREE_CELLS)
assert S == CORRIDOR_LEN + ROOM_N * ROOM_N, f"expected {CORRIDOR_LEN + ROOM_N * ROOM_N} free cells, got {S}"


def step_normal(rc, a):
    """k=1 (non-sticky) deterministic dynamics: bump into a wall = stay."""
    r, c = rc
    dr, dc = DELTA[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_R and 0 <= nc < GRID_C and not WALL[nr, nc]:
        return (nr, nc)
    return (r, c)


P0 = np.zeros((A, S, S))
for ai, a in enumerate(ACTIONS):
    for i, rc in enumerate(FREE_CELLS):
        j = CELL_TO_IDX[step_normal(rc, a)]
        P0[ai, i, j] = 1.0

D_RC = (CORRIDOR_ROW, CORRIDOR_LEN)      # (3, 8): last corridor cell = the exit
D = CELL_TO_IDX[D_RC]
A_CROSS = "right"                        # the action that moves into the room
A_FAIL = ["up", "down", "left"]
A_FAIL_SHOW = "left"                     # the failing action drawn in the maps
assert step_normal(D_RC, A_CROSS) == (CORRIDOR_ROW, CORRIDOR_LEN + 1), "right at D must enter the room"

G_RC = (1, GRID_C - 2)                   # (1, 13): far corner of the room (gold star)
G = CELL_TO_IDX[G_RC]

ROOM_MASK = np.array([(r in ROOM_ROWS and c in ROOM_COLS) for (r, c) in FREE_CELLS])
assert ROOM_MASK.sum() == ROOM_N * ROOM_N and ROOM_MASK[G]
R_ROOM = ROOM_MASK.astype(np.float64)    # reward = 1 on every room cell


# ----------------------------------------------------------------------------
# 2. Sticky-exit dynamics and exact successor measure (verbatim from FourRooms).
# ----------------------------------------------------------------------------
def sticky_action_matrices(k):
    """P(a) per action, sticky rule only at D for the failing actions; a_cross always
    succeeds and is unaffected by k."""
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
    """(A,S,S) -> (S*A,S) with row index s*A+a."""
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
    Pstack = stack_sa(P)
    Ncont = np.linalg.inv(np.eye(S) - gamma * P_continue)
    return Pstack @ Ncont


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
    R = M / rho[None, :]
    Ascaled = R * np.sqrt(rho)[None, :]
    U, sv, Vt = np.linalg.svd(Ascaled, full_matrices=False)
    d_eff = min(d, sv.shape[0])
    Ud, Sd, Vtd = U[:, :d_eff], sv[:d_eff], Vt[:d_eff, :]
    F = Ud * Sd[None, :]
    B = Vtd.T / np.sqrt(rho)[:, None]
    return F, B


def fb_readout(F, B, rho, r):
    z = B.T @ (rho * r)
    return F @ z


def fb_row(F, B, row_idx):
    """R_d(row, :) = F[row] . B^T; caller multiplies by rho to get M_FB(row, :)."""
    return F[row_idx] @ B.T


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
# 5. Metrics.
# ----------------------------------------------------------------------------
def door_metrics(Q_by_action):
    q_cross = Q_by_action[A_CROSS]
    q_fail_mean = np.mean([Q_by_action[a] for a in A_FAIL])
    raw_gap = q_cross - q_fail_mean
    argmax_a = max(Q_by_action, key=Q_by_action.get)
    return raw_gap, q_cross, q_fail_mean, argmax_a


def tv_half(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def door_mass_fraction(vals_raw):
    """Fraction of this row's total mass sitting on the exit cell D itself."""
    total = float(np.sum(vals_raw))
    return float(vals_raw[D] / total) if total > 0 else 0.0


def interval_estimate(values, confidence=0.95):
    """Mean + CI across seeds (rliable stratified bootstrap if importable, else a
    t-interval). Returns (mean, lo, hi, method_str)."""
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
        tcrit = 2.776
        method = f"t-interval ({int(confidence*100)}%, hardcoded t-crit, df={n-1})"
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    margin = tcrit * se
    return mean, mean - margin, mean + margin, method


# ----------------------------------------------------------------------------
# 6. Closed-form sweep -> gap.csv
# ----------------------------------------------------------------------------
DENSITY_SEEDS = [0, 1, 2, 3, 4]
KS = [1, 5, 10, 20, 40]
DS = [4, 8, 16, 32]
GAMMA = 0.98
ROW_D = {a: D * A + ACTIONS.index(a) for a in ACTIONS}


def row_metrics(p_cross, p_fail, p_true_cross, p_true_fail, clip_for_door=False):
    pc = np.clip(p_cross, 0.0, None) if clip_for_door else p_cross
    pf = np.clip(p_fail, 0.0, None) if clip_for_door else p_fail
    return dict(
        door_frac_cross=door_mass_fraction(pc), door_frac_fail=door_mass_fraction(pf),
        tv_cross=tv_half(p_cross, p_true_cross), tv_fail=tv_half(p_fail, p_true_fail),
        mass_room_cross=float(np.sum(pc[ROOM_MASK])),
    )


def run_main_sweep(csv_name="gap.csv"):
    rows = []
    door_payload = {}
    t0 = time.time()
    gamma = GAMMA
    for k in KS:
        P_k = sticky_action_matrices(k)
        P_pi = policy_matrix_uniform(P_k)
        rho, n_iter, diff = stationary(P_pi)
        log(f"[gamma={gamma} k={k}] stationary dist converged in {n_iter} iters, "
            f"final L1 delta={diff:.2e}, min(rho)={rho.min():.2e}, rho(D)={rho[D]:.3f}")
        M = successor_measure(P_k, P_pi, gamma)
        check_row_sums(M, gamma, f"gamma={gamma} k={k} (random-policy M)")

        Q_true_all = M @ R_ROOM
        q_true_by_a = {a: Q_true_all[ROW_D[a]] for a in ACTIONS}
        raw_gap_true, qc_true, qf_true, amax_true = door_metrics(q_true_by_a)
        p_true_cross = (1 - gamma) * M[ROW_D[A_CROSS]]
        p_true_fail = (1 - gamma) * M[ROW_D[A_FAIL_SHOW]]
        m_target = (1 - gamma) * M

        for d in DS:
            rows.append(dict(
                gamma=gamma, k=k, d=d, method="true", seed=-1,
                Q_cross=qc_true, Q_fail_mean=qf_true, raw_gap=raw_gap_true, norm_gap=1.0,
                argmax_action=amax_true, argmax_is_cross=(amax_true == A_CROSS),
                **row_metrics(p_true_cross, p_true_fail, p_true_cross, p_true_fail)))

            F_fb, B_fb = fb_fit(M, rho, d)
            Q_fb_all = fb_readout(F_fb, B_fb, rho, R_ROOM)
            q_fb_by_a = {a: Q_fb_all[ROW_D[a]] for a in ACTIONS}
            raw_gap_fb, qc_fb, qf_fb, amax_fb = door_metrics(q_fb_by_a)
            p_fb_cross = (1 - gamma) * fb_row(F_fb, B_fb, ROW_D[A_CROSS]) * rho
            p_fb_fail = (1 - gamma) * fb_row(F_fb, B_fb, ROW_D[A_FAIL_SHOW]) * rho
            rows.append(dict(
                gamma=gamma, k=k, d=d, method="fb", seed=-1,
                Q_cross=qc_fb, Q_fail_mean=qf_fb, raw_gap=raw_gap_fb,
                norm_gap=raw_gap_fb / raw_gap_true,
                argmax_action=amax_fb, argmax_is_cross=(amax_fb == A_CROSS),
                **row_metrics(p_fb_cross, p_fb_fail, p_true_cross, p_true_fail, clip_for_door=True)))

            for sd in DENSITY_SEEDS:
                # single init, no restart-keep-best: the 5 values are a genuine sample
                F_d, B_d, floss = density_fit(m_target, d, seed=1000 * sd + d * 17 + k * 3, restarts=1)
                Q_d_all, p_all = density_readout(F_d, B_d, R_ROOM, gamma)
                q_d_by_a = {a: Q_d_all[ROW_D[a]] for a in ACTIONS}
                raw_gap_d, qc_d, qf_d, amax_d = door_metrics(q_d_by_a)
                p_d_cross = p_all[ROW_D[A_CROSS]]
                p_d_fail = p_all[ROW_D[A_FAIL_SHOW]]
                rows.append(dict(
                    gamma=gamma, k=k, d=d, method="density", seed=sd,
                    Q_cross=qc_d, Q_fail_mean=qf_d, raw_gap=raw_gap_d,
                    norm_gap=raw_gap_d / raw_gap_true,
                    argmax_action=amax_d, argmax_is_cross=(amax_d == A_CROSS),
                    **row_metrics(p_d_cross, p_d_fail, p_true_cross, p_true_fail)))
                if k == 20 and d == 8 and sd == 0:
                    door_payload = dict(
                        a_fail_show=A_FAIL_SHOW, density_seed_used=sd,
                        p_true_cross=p_true_cross, p_true_fail=p_true_fail,
                        p_fb_cross=p_fb_cross, p_fb_fail=p_fb_fail,
                        p_density_cross=p_d_cross, p_density_fail=p_d_fail)
        log(f"  ...done k={k} ({time.time()-t0:.1f}s elapsed)")

    csv_path = HERE / csv_name
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {csv_path} ({len(rows)} rows)")
    return rows, door_payload


# ----------------------------------------------------------------------------
# 7. TD-trained tabular agents (fixed uniform-random policy, no actor, no z);
#    losses verbatim from the FourRooms script.
# ----------------------------------------------------------------------------
TRAIN_KS = KS
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TRAIN_D = 8
TRAIN_GAMMA = GAMMA
TRAIN_STEPS = 20000
TRAIN_LR = 3e-4
TRAIN_TAU = 0.005
TRAIN_BATCH_T = 256
TRAIN_BATCH_U = 256
TRAIN_DATASET_N = 100_000
TRAIN_RESET_EVERY = 500

_DET_NEXT = np.zeros((A, S), dtype=np.int64)
for _ai in range(A):
    for _si, _rc in enumerate(FREE_CELLS):
        _DET_NEXT[_ai, _si] = CELL_TO_IDX[step_normal(_rc, ACTIONS[_ai])]
_A_FAIL_IDX = set(ACTIONS.index(a) for a in A_FAIL)


def generate_dataset_td(k, seed, n=TRAIN_DATASET_N, reset_every=TRAIN_RESET_EVERY):
    """One long uniform-random rollout under the k-sticky dynamics, restarting from a
    fresh uniform-random state every `reset_every` steps."""
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
    init_carry = ((F0, B0), (F0, B0), opt_state0)
    (params, target_params, opt_state), losses = jax.lax.scan(body, init_carry, keys)
    F, B = params
    return F, B, losses[-1]


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def _train_density_td(key, s_idx, a_idx, sp_idx, d, steps, lr, tau, batch_t, batch_u, gamma):
    n = s_idx.shape[0]
    key, kinit = random.split(key)
    F0, B0 = _td_init(kinit, d)
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
    init_carry = ((F0, B0), (F0, B0), opt_state0)
    (params, target_params, opt_state), losses = jax.lax.scan(body, init_carry, keys)
    F, B = params
    return F, B, losses[-1]


def run_trained_sweep():
    rows = []
    t0 = time.time()
    for k in TRAIN_KS:
        P_k = sticky_action_matrices(k)
        P_pi = policy_matrix_uniform(P_k)
        rho, _, _ = stationary(P_pi)
        M = successor_measure(P_k, P_pi, TRAIN_GAMMA)
        Q_true_all = M @ R_ROOM
        q_true_by_a = {a: Q_true_all[ROW_D[a]] for a in ACTIONS}
        raw_gap_true, _, _, _ = door_metrics(q_true_by_a)
        p_true_cross = (1 - TRAIN_GAMMA) * M[ROW_D[A_CROSS]]
        p_true_fail = (1 - TRAIN_GAMMA) * M[ROW_D[A_FAIL_SHOW]]

        for seed in TRAIN_SEEDS:
            s_idx, a_idx, sp_idx = generate_dataset_td(k, seed)
            s_j, a_j, sp_j = jnp.asarray(s_idx), jnp.asarray(a_idx), jnp.asarray(sp_idx)
            key = random.PRNGKey(20_000 * k + seed)
            key_fb, key_density = random.split(key)

            F_fb, B_fb, loss_fb = _train_fb_td(
                key_fb, s_j, a_j, sp_j, TRAIN_D, TRAIN_STEPS, TRAIN_LR, TRAIN_TAU,
                TRAIN_BATCH_T, TRAIN_BATCH_U, TRAIN_GAMMA)
            F_fb = np.asarray(F_fb); B_fb = np.asarray(B_fb)
            Q_fb_all = fb_readout(F_fb, B_fb, rho, R_ROOM)
            q_fb_by_a = {a: Q_fb_all[ROW_D[a]] for a in ACTIONS}
            raw_gap_fb, _, _, amax_fb = door_metrics(q_fb_by_a)
            p_fb_cross = (1 - TRAIN_GAMMA) * fb_row(F_fb, B_fb, ROW_D[A_CROSS]) * rho
            p_fb_fail = (1 - TRAIN_GAMMA) * fb_row(F_fb, B_fb, ROW_D[A_FAIL_SHOW]) * rho
            rows.append(dict(k=k, seed=seed, method="fb_td", raw_gap=raw_gap_fb,
                             norm_gap=raw_gap_fb / raw_gap_true, argmax_is_cross=(amax_fb == A_CROSS),
                             **row_metrics(p_fb_cross, p_fb_fail, p_true_cross, p_true_fail, clip_for_door=True),
                             final_train_loss=float(loss_fb)))

            F_d, B_d, loss_d = _train_density_td(
                key_density, s_j, a_j, sp_j, TRAIN_D, TRAIN_STEPS, TRAIN_LR, TRAIN_TAU,
                TRAIN_BATCH_T, TRAIN_BATCH_U, TRAIN_GAMMA)
            F_d = np.asarray(F_d); B_d = np.asarray(B_d)
            Q_d_all, p_all = density_readout(F_d, B_d, R_ROOM, TRAIN_GAMMA)
            q_d_by_a = {a: Q_d_all[ROW_D[a]] for a in ACTIONS}
            raw_gap_d, _, _, amax_d = door_metrics(q_d_by_a)
            rows.append(dict(k=k, seed=seed, method="density_td", raw_gap=raw_gap_d,
                             norm_gap=raw_gap_d / raw_gap_true, argmax_is_cross=(amax_d == A_CROSS),
                             **row_metrics(p_all[ROW_D[A_CROSS]], p_all[ROW_D[A_FAIL_SHOW]],
                                           p_true_cross, p_true_fail),
                             final_train_loss=float(loss_d)))
        log(f"  [trained] k={k} done, 5 seeds x 2 methods ({time.time()-t0:.1f}s elapsed)")

    csv_path = HERE / "gap_trained.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {csv_path} ({len(rows)} rows)")
    return rows


def trained_gap_ci(trained_rows, method, k):
    vals = [float(r["norm_gap"]) for r in trained_rows if r["method"] == method and int(r["k"]) == k]
    return interval_estimate(vals)


# ----------------------------------------------------------------------------
# 8. Sanity checks.
# ----------------------------------------------------------------------------
def run_sanity_checks():
    log("\n=== Sanity checks ===")
    gamma, k = GAMMA, 20
    P_k = sticky_action_matrices(k)
    P_pi = policy_matrix_uniform(P_k)
    rho, _, _ = stationary(P_pi)
    M = successor_measure(P_k, P_pi, gamma)
    check_row_sums(M, gamma, "d=|S| sanity block")

    row_cross = ROW_D[A_CROSS]
    rows_fail = [ROW_D[a] for a in A_FAIL]
    Q_true_all = M @ R_ROOM
    raw_gap_true = Q_true_all[row_cross] - np.mean(Q_true_all[rows_fail])

    d_full = S
    F_fb, B_fb = fb_fit(M, rho, d_full)
    Q_fb_all = fb_readout(F_fb, B_fb, rho, R_ROOM)
    norm_gap_fb = (Q_fb_all[row_cross] - np.mean(Q_fb_all[rows_fail])) / raw_gap_true
    log(f"[d=|S|={d_full}] FB normalized margin = {norm_gap_fb:.6f} (want 1 within 1e-3)")
    assert abs(norm_gap_fb - 1.0) < 1e-3

    F_d, B_d, floss = density_fit((1 - gamma) * M, d_full, seed=999)
    Q_d_all, _ = density_readout(F_d, B_d, R_ROOM, gamma)
    norm_gap_d = (Q_d_all[row_cross] - np.mean(Q_d_all[rows_fail])) / raw_gap_true
    log(f"[d=|S|={d_full}] Density normalized margin = {norm_gap_d:.6f} "
        f"(want 1 within 0.05; final CE loss={floss:.5f})")
    assert abs(norm_gap_d - 1.0) < 0.05

    P_k1 = sticky_action_matrices(1)
    M1 = successor_measure(P_k1, policy_matrix_uniform(P_k1), gamma)
    Q1_all = M1 @ R_ROOM
    raw_gap_true_k1 = Q1_all[row_cross] - np.mean(Q1_all[rows_fail])
    log(f"[k=1] true raw margin (room reward) = {raw_gap_true_k1:.4f} (want clearly nonzero)")
    assert abs(raw_gap_true_k1) > 1e-2
    log("all sanity checks passed.\n")


# ----------------------------------------------------------------------------
# 9. Figure (fig_toy_main style): smooth occupancy maps + margin-vs-k panel.
# ----------------------------------------------------------------------------
WALL_COLOR = "#2b2b2b"
INK = "#2a2a2a"
MUTED = "#6b6b6b"
TOY_FB_BLUE = "#2171b5"
TOY_DENSITY_ORANGE = "#e34a33"
TOY_TRUE_GREY = "#4d4d4d"
RC_PARAMS = {
    "font.family": "sans-serif", "font.size": 7.5, "text.color": INK,
    "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False, "svg.fonttype": "none",
}


def to_grid(vals):
    g = np.full((GRID_R, GRID_C), np.nan)
    for i, (r, c) in enumerate(FREE_CELLS):
        g[r, c] = vals[i]
    return g


def smooth_upsample(vals, factor=8, sigma_cells=0.30):
    """Wall-aware smooth upsample: bilinear zoom then a normalized Gaussian blur
    (blur(g*free)/blur(free)) so the 1-cell-wide corridor is not dimmed by its wall
    neighbours; sigma small so the exit cell keeps its mass. NaN at walls."""
    from scipy.ndimage import zoom, gaussian_filter
    g = to_grid(vals)
    free = ~np.isnan(g)
    g_filled = np.nan_to_num(g, nan=0.0)
    g_hi = zoom(g_filled, factor, order=1)
    free_hi = zoom(free.astype(np.float64), factor, order=0)
    g_hi = g_hi * free_hi
    num = gaussian_filter(g_hi, sigma=sigma_cells * factor)
    den = gaussian_filter(free_hi, sigma=sigma_cells * factor)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return np.where(free_hi > 0.5, out, np.nan)


def room_mass_fraction(vals_raw):
    total = float(np.sum(vals_raw))
    return float(np.sum(vals_raw[ROOM_MASK]) / total) if total > 0 else 0.0


def draw_toy_occupancy_panel(ax, vals_raw, cmap_name, title, vmax=None, show_door_frac=True,
                             start_rc=None, show_room_frac=False, norm=None):
    import matplotlib as mpl
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(WALL_COLOR)
    g_hi = smooth_upsample(vals_raw)
    if vmax is None:
        vmax = float(np.nanmax(g_hi))
    vmax = vmax if vmax > 0 else 1.0
    extent = [-0.5, GRID_C - 0.5, GRID_R - 0.5, -0.5]
    if norm is not None:
        im = ax.imshow(g_hi, cmap=cmap, norm=norm, extent=extent, origin="upper", interpolation="nearest")
    else:
        im = ax.imshow(g_hi, cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, origin="upper",
                       interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.8)
    ax.set_title(title, fontsize=8.0, color=INK, pad=3)
    Dr, Dc = D_RC
    Gr, Gc = G_RC
    if start_rc is None:
        ax.scatter([Dc], [Dr], s=30, facecolor="black", edgecolor="white", linewidth=0.9, zorder=6)
    else:
        # agent starts before the exit: black dot at S0, small hollow marker at D
        ax.scatter([start_rc[1]], [start_rc[0]], s=30, facecolor="black", edgecolor="white",
                   linewidth=0.9, zorder=6)
        ax.scatter([Dc], [Dr], s=22, facecolor="none", edgecolor="black", linewidth=0.9, zorder=6)
    ax.scatter([Gc], [Gr], marker="*", s=120, facecolor="#f5c518", edgecolor="black",
               linewidth=0.7, zorder=6)
    if show_door_frac:
        frac = door_mass_fraction(vals_raw)
        txt = f"door: {frac:.2f}"
        if show_room_frac:
            txt = f"door {frac:.2f}   room {room_mass_fraction(vals_raw):.2f}"
        ax.text(0.5, -0.06, txt, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.2, color=MUTED)
    return im


ROW_LABELS = ["exit the corridor", "fail to exit"]
COL_SPECS = [("true", "Greys"), ("FB", "Blues"), ("Density-FB", "OrRd")]


def _map_columns(door_payload):
    """rows_by_col[cidx] = [vals_exit, vals_fail], FB clipped at 0 for display."""
    return [[door_payload["p_true_cross"], door_payload["p_true_fail"]],
            [np.clip(door_payload["p_fb_cross"], 0.0, None), np.clip(door_payload["p_fb_fail"], 0.0, None)],
            [door_payload["p_density_cross"], door_payload["p_density_fail"]]]


def draw_maps_block(fig, gs, door_payload, rows=(0, 1), row_labels=True):
    """2x3 (or 1x3) block of smooth occupancy maps; color scale shared within each
    column across the exit/fail rows. Returns the door-mass fractions drawn."""
    fracs = {}
    for cidx, ((name, cmap_name), col_vals) in enumerate(zip(COL_SPECS, _map_columns(door_payload))):
        vmax_col = max(float(np.nanmax(smooth_upsample(col_vals[0]))),
                       float(np.nanmax(smooth_upsample(col_vals[1]))))
        for pos, ridx in enumerate(rows):
            vals = col_vals[ridx]
            ax = fig.add_subplot(gs[pos, cidx])
            title = name if pos == 0 else ""
            draw_toy_occupancy_panel(ax, vals, cmap_name, title, vmax=vmax_col)
            if cidx == 0 and row_labels:
                ax.set_ylabel(ROW_LABELS[ridx], fontsize=7.8, color=INK)
            fracs[(name, ["exit", "fail"][ridx])] = door_mass_fraction(vals)
    return fracs


def collect_gap_ci(gap_rows, d_show):
    ks = sorted(set(int(r["k"]) for r in gap_rows))
    fb_y, dmean, dlo, dhi = [], [], [], []
    ci_method = None
    for k in ks:
        fb_y.append(next(float(r["norm_gap"]) for r in gap_rows
                         if r["method"] == "fb" and int(r["d"]) == d_show and int(r["k"]) == k))
        vals = [float(r["norm_gap"]) for r in gap_rows
                if r["method"] == "density" and int(r["d"]) == d_show and int(r["k"]) == k]
        m, lo, hi, ci_method = interval_estimate(vals)
        dmean.append(m); dlo.append(lo); dhi.append(hi)
    return ks, fb_y, dmean, dlo, dhi, ci_method


def draw_margin_panel(ax, gap_rows, trained_rows, legend_loc="lower right"):
    ks, fb_y, dmean, dlo, dhi, ci_method = collect_gap_ci(gap_rows, TRAIN_D)
    ax.axhline(1.0, color="#9a9a9a", linestyle="--", linewidth=1.4, zorder=1, label="true")
    ax.axhline(0.0, color="#cfcfcf", linestyle="-", linewidth=0.8, zorder=1)
    ax.fill_between(ks, dlo, dhi, color=TOY_DENSITY_ORANGE, alpha=0.20, linewidth=0, zorder=2)
    ax.plot(ks, dmean, color=TOY_DENSITY_ORANGE, linewidth=1.8, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7, label="Density-FB (CE optimum)", zorder=3)
    ax.plot(ks, fb_y, color=TOY_FB_BLUE, linewidth=1.8, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7, label="FB (SVD optimum)", zorder=3)

    train_ks = sorted(set(int(r["k"]) for r in trained_rows)) if trained_rows else []
    series = {}
    for method, color, label in ([("fb_td", TOY_FB_BLUE, "FB (TD-trained)"),
                                  ("density_td", TOY_DENSITY_ORANGE, "Density-FB (TD-trained)")]
                                 if trained_rows else []):
        mean, lo, hi = [], [], []
        for kk in train_ks:
            m, l, h, _ = trained_gap_ci(trained_rows, method, kk)
            mean.append(m); lo.append(m - l); hi.append(h - m)
        ax.errorbar(train_ks, mean, yerr=[lo, hi], fmt="D", markersize=5,
                    markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.3,
                    ecolor=color, elinewidth=1.1, capsize=2.5, linestyle="none", label=label, zorder=4)
        series[method] = mean

    ax.set_title("does the critic prefer exiting?", fontsize=8.2, color=INK)
    ax.set_xlabel("k (stickiness)", fontsize=7.4)
    ax.set_ylabel("normalized margin", fontsize=7.4)
    ax.set_xticks(ks)
    ax.tick_params(labelsize=6.4, length=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cfcfcf")
    if legend_loc is not None:
        ax.legend(frameon=False, fontsize=5.4, loc=legend_loc, handlelength=1.4)
    return ci_method


def _save(fig, stem, svg=False):
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    if svg:
        fig.savefig(FIG_DIR / f"{stem}.svg")
    plt.close(fig)
    log(f"wrote {FIG_DIR / stem}.{{pdf,png{',svg' if svg else ''}}}")


def save_toy_corridor_figure(door_payload, gap_rows, trained_rows):
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 2.55), constrained_layout=True)
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.75, 1.0])
    # maps are wide and short: two map rows + the margin panel's legend underneath
    # (fills what would otherwise be dead space next to the taller right panel)
    left_gs = outer[0].subgridspec(3, 3, hspace=0.10, wspace=0.06, height_ratios=[1.0, 1.0, 0.42])
    fracs = draw_maps_block(fig, left_gs, door_payload)
    ax = fig.add_subplot(outer[1])
    ci_method = draw_margin_panel(ax, gap_rows, trained_rows, legend_loc=None)
    leg_ax = fig.add_subplot(left_gs[2, :])
    leg_ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    leg_ax.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=5.8,
                  handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)
    _save(fig, "fig_toy_corridor", svg=True)
    return fracs, ci_method


S0_RC = (CORRIDOR_ROW, CORRIDOR_LEN - 2)   # (3, 6): corridor cell 2 steps before the exit
S0 = CELL_TO_IDX[S0_RC]
PRE_D_RC = (CORRIDOR_ROW, CORRIDOR_LEN - 1)  # (3, 7): the cell between S0 and D
APPROACH_ROW_LABELS = ["approach", "then exit", "then fail"]


def walk_rows(door_payload, s0_rc):
    """Occupancy from corridor cell s0_rc taking 'right', under 'walk right to D, take
    action a at D once, uniform random afterwards'. The n-step walk is deterministic and
    entered exactly; only the door row M(D, a, .) is the estimator's. Normalized:
    (1-g) sum_{t<n} g^t e_{cell t} + g^n p(D, a, .) (sums to 1).
    Returns {col: [row_exit, row_fail]} with FB rows raw (unclipped)."""
    g = GAMMA
    n = D_RC[1] - s0_rc[1]
    assert s0_rc[0] == CORRIDOR_ROW and n >= 1
    prefix = np.zeros(S)
    for t in range(n):
        prefix[CELL_TO_IDX[(CORRIDOR_ROW, s0_rc[1] + 1 + t)]] = (1 - g) * g ** t
    gn = g ** n

    def compose(p_exit, p_fail):
        return [prefix + gn * p_exit, prefix + gn * p_fail]

    return {
        "true": compose(door_payload["p_true_cross"], door_payload["p_true_fail"]),
        "FB": compose(door_payload["p_fb_cross"], door_payload["p_fb_fail"]),
        "Density-FB": compose(door_payload["p_density_cross"], door_payload["p_density_fail"]),
    }


WALK_ROW_LABELS = ["then exit", "then fail"]


def draw_walk_block(fig, gs, rows, s0_rc):
    """2x3 block: rows then-exit / then-fail, cols true / FB / Density-FB, shared color
    scale per column; 'door x  room y' under each map. Returns {(col,row): (door, room, tv)}."""
    stats = {}
    for cidx, (name, cmap_name) in enumerate(COL_SPECS):
        col_vals = [np.clip(v, 0.0, None) for v in rows[name]]
        vmax_col = max(float(np.nanmax(smooth_upsample(v))) for v in col_vals)
        for ridx, vals in enumerate(col_vals):
            ax = fig.add_subplot(gs[ridx, cidx])
            draw_toy_occupancy_panel(ax, vals, cmap_name, name if ridx == 0 else "",
                                     vmax=vmax_col, start_rc=s0_rc, show_room_frac=True)
            if cidx == 0:
                ax.set_ylabel(WALK_ROW_LABELS[ridx], fontsize=7.8, color=INK)
            tv = tv_half(rows[name][ridx], rows["true"][ridx])
            stats[(name, WALK_ROW_LABELS[ridx])] = (door_mass_fraction(vals), room_mass_fraction(vals), tv)
    return stats


def save_walk_figure(door_payload, gap_rows, trained_rows, s0_rc, stem):
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt

    rows = walk_rows(door_payload, s0_rc)
    fig = plt.figure(figsize=(7.2, 2.55), constrained_layout=True)
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.75, 1.0])
    left_gs = outer[0].subgridspec(3, 3, hspace=0.10, wspace=0.06, height_ratios=[1.0, 1.0, 0.42])
    stats = draw_walk_block(fig, left_gs, rows, s0_rc)
    ax = fig.add_subplot(outer[1])
    draw_margin_panel(ax, gap_rows, trained_rows, legend_loc=None)
    leg_ax = fig.add_subplot(left_gs[2, :])
    leg_ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    leg_ax.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=5.8,
                  handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)
    _save(fig, stem, svg=True)
    for (name, row), (door, room, tv) in stats.items():
        log(f"[{stem}] {name:11s} {row:9s}: door {door:.3f}  room {room:.3f}  tv {tv:.3f}")
    return stats


DIVERGING_CMAP = "RdBu_r"   # red = positive (more mass after failing), blue = negative, white midpoint


def draw_signed_panel(ax, vals, vmax_abs, title, start_rc, note=None):
    """Diverging map of a signed per-cell quantity, symmetric scale +-vmax_abs, walls dark
    grey, black dot at the start, hollow marker at D, gold star at G."""
    import matplotlib as mpl
    cmap = mpl.colormaps[DIVERGING_CMAP].copy()
    cmap.set_bad(WALL_COLOR)
    g_hi = smooth_upsample(vals)
    im = ax.imshow(g_hi, cmap=cmap, vmin=-vmax_abs, vmax=vmax_abs,
                   extent=[-0.5, GRID_C - 0.5, GRID_R - 0.5, -0.5], origin="upper",
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cfcfcf")
        spine.set_linewidth(0.8)
    ax.set_title(title, fontsize=8.0, color=INK, pad=3)
    ax.scatter([start_rc[1]], [start_rc[0]], s=30, facecolor="black", edgecolor="white",
               linewidth=0.9, zorder=6)
    ax.scatter([D_RC[1]], [D_RC[0]], s=22, facecolor="none", edgecolor="black", linewidth=0.9, zorder=6)
    ax.scatter([G_RC[1]], [G_RC[0]], marker="*", s=120, facecolor="#f5c518", edgecolor="black",
               linewidth=0.7, zorder=6)
    if note:
        ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=MUTED)
    return im


def _add_legend_row(fig, gs_slot, ax_margin):
    leg_ax = fig.add_subplot(gs_slot)
    leg_ax.axis("off")
    handles, labels = ax_margin.get_legend_handles_labels()
    leg_ax.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=5.8,
                  handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)


def save_start_contrast_figure(door_payload, gap_rows, s0_rc, stem="fig_toy_corridor_start_contrast",
                               power=0.6):
    """3 rows x 3 cols: then-exit / then-fail (sequential, PowerNorm(power) shared per
    column) and fail-minus-exit (diverging, one symmetric scale fixed to the true row's
    max |value|). Returns {col: |Delta|/|Delta_true|} for FB and Density-FB."""
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    rows = walk_rows(door_payload, s0_rc)
    diffs = {name: rows[name][1] - rows[name][0] for name in rows}   # fail - exit, FB raw
    l1_true = float(np.sum(np.abs(diffs["true"])))
    ratios = {name: float(np.sum(np.abs(diffs[name]))) / l1_true for name in ("FB", "Density-FB")}
    vmax_abs = float(np.nanmax(np.abs(smooth_upsample(diffs["true"]))))

    fig = plt.figure(figsize=(7.2, 3.3), constrained_layout=True)
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.75, 1.0])
    left_gs = outer[0].subgridspec(4, 3, hspace=0.10, wspace=0.06, height_ratios=[1, 1, 1, 0.40])
    row3_axes = []
    im_div = None
    for cidx, (name, cmap_name) in enumerate(COL_SPECS):
        col_vals = [np.clip(v, 0.0, None) for v in rows[name]]
        vmax_col = max(float(np.nanmax(smooth_upsample(v))) for v in col_vals)
        norm = PowerNorm(gamma=power, vmin=0.0, vmax=vmax_col)
        for ridx, vals in enumerate(col_vals):
            ax = fig.add_subplot(left_gs[ridx, cidx])
            draw_toy_occupancy_panel(ax, vals, cmap_name, name if ridx == 0 else "", norm=norm,
                                     start_rc=s0_rc, show_room_frac=True)
            if cidx == 0:
                ax.set_ylabel(WALK_ROW_LABELS[ridx], fontsize=7.8, color=INK)
        ax = fig.add_subplot(left_gs[2, cidx])
        note = None if name == "true" else f"|Δ| / |Δ_true| = {ratios[name]:.2f}"
        im_div = draw_signed_panel(ax, diffs[name], vmax_abs, "", s0_rc, note=note)
        if cidx == 0:
            ax.set_ylabel("fail minus exit", fontsize=7.8, color=INK)
        row3_axes.append(ax)
    cb = fig.colorbar(im_div, ax=row3_axes, location="right", shrink=0.9, aspect=12, pad=0.01)
    cb.ax.tick_params(labelsize=5.0, color=MUTED, labelcolor=MUTED, length=2)
    cb.outline.set_visible(False)
    ax_m = fig.add_subplot(outer[1])
    draw_margin_panel(ax_m, gap_rows, None, legend_loc=None)
    _add_legend_row(fig, left_gs[3, :], ax_m)
    _save(fig, stem, svg=True)
    log(f"[{stem}] |Delta|/|Delta_true|: FB={ratios['FB']:.3f} Density-FB={ratios['Density-FB']:.3f} "
        f"(true max|Delta| per cell = {float(np.max(np.abs(diffs['true']))):.4f})")
    return ratios


def save_start_error_figure(door_payload, gap_rows, s0_rc, stem="fig_toy_corridor_start_error"):
    """2 rows (exit, fail) x 2 cols (FB, Density-FB): fitted minus true per cell, one
    shared symmetric diverging scale, TV printed under each. Returns {(col,row): tv}."""
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt

    rows = walk_rows(door_payload, s0_rc)
    errs = {name: [rows[name][r] - rows["true"][r] for r in range(2)] for name in ("FB", "Density-FB")}
    vmax_abs = max(float(np.nanmax(np.abs(smooth_upsample(e)))) for v in errs.values() for e in v)
    tvs = {}
    fig = plt.figure(figsize=(6.4, 2.55), constrained_layout=True)
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.25, 1.0])
    left_gs = outer[0].subgridspec(3, 2, hspace=0.10, wspace=0.06, height_ratios=[1, 1, 0.30])
    axes = []
    im = None
    for cidx, name in enumerate(("FB", "Density-FB")):
        for ridx in range(2):
            ax = fig.add_subplot(left_gs[ridx, cidx])
            tv = tv_half(rows[name][ridx], rows["true"][ridx])
            tvs[(name, WALK_ROW_LABELS[ridx])] = tv
            im = draw_signed_panel(ax, errs[name][ridx], vmax_abs, name if ridx == 0 else "", s0_rc,
                                   note=f"TV = {tv:.2f}")
            if cidx == 0:
                ax.set_ylabel(WALK_ROW_LABELS[ridx], fontsize=7.8, color=INK)
            axes.append(ax)
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.55, aspect=28, pad=0.02, fraction=0.06)
    cb.ax.tick_params(labelsize=5.0, color=MUTED, labelcolor=MUTED, length=2)
    cb.outline.set_visible(False)
    cb.set_label("fitted − true", fontsize=6.0, color=MUTED)
    ax_m = fig.add_subplot(outer[1])
    draw_margin_panel(ax_m, gap_rows, None, legend_loc=None)
    _add_legend_row(fig, left_gs[2, :], ax_m)
    _save(fig, stem, svg=True)
    log(f"[{stem}] TV: " + ", ".join(f"{n}/{r}={v:.3f}" for (n, r), v in tvs.items()))
    return tvs


K40_DIR_NAME = "k40"
K40_POWER = 0.6


def _save_to(fig, out_dir, stem, svg=False):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    if svg:
        fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)
    log(f"wrote {out_dir / stem}.{{pdf,png{',svg' if svg else ''}}}")


def run_k40_panels(gap_rows, s0_rc, k=40, d=8, seed=0):
    """Individual single-panel files under PAPER/ICLR/figures/corridor/k40/ for the 14-cell
    corridor from s0_rc at the strongest stickiness: six occupancy maps (one shared
    PowerNorm(0.6) scale across all six, fixed to their common max, so one colorbar file
    is exact), three fail-minus-exit contrast maps (one symmetric scale fixed to the true
    contrast's max |value|), the closed-form margin-vs-k panel with k marked, two colorbar
    files, and subfigures.tex. Returns the rows to append to corridor14.csv."""
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import PowerNorm

    out_dir = FIG_DIR / K40_DIR_NAME
    payload = door_payload_closed_form(k=k, d=d, seed=seed)
    rows = walk_rows(payload, s0_rc)
    col_key = {"true": "true", "FB": "fb", "Density-FB": "density"}
    clipped = {name: [np.clip(v, 0.0, None) for v in rows[name]] for name in rows}
    vmax_occ = max(float(np.nanmax(smooth_upsample(v))) for vals in clipped.values() for v in vals)
    norm_occ = PowerNorm(gamma=K40_POWER, vmin=0.0, vmax=vmax_occ)
    diffs = {name: rows[name][1] - rows[name][0] for name in rows}
    vmax_abs = float(np.nanmax(np.abs(smooth_upsample(diffs["true"]))))

    csv_rows = []
    stats = {}
    for name, cmap_name in COL_SPECS:
        for ridx, which in enumerate(["exit", "fail"]):
            fig = plt.figure(figsize=(2.4, 1.2), constrained_layout=True)
            ax = fig.add_subplot(111)
            vals = clipped[name][ridx]
            draw_toy_occupancy_panel(ax, vals, cmap_name, "", norm=norm_occ, start_rc=s0_rc,
                                     show_room_frac=True)
            _save_to(fig, out_dir, f"{which}_{col_key[name]}", svg=True)
            door, room = door_mass_fraction(vals), room_mass_fraction(vals)
            tv = tv_half(rows[name][ridx], rows["true"][ridx])
            stats[(name, which)] = (door, room, tv)
            csv_rows.append(dict(gamma=GAMMA, k=k, d=d, method=col_key[name],
                                 seed=(seed if name == "Density-FB" else -1),
                                 variant=f"k{k}_start_{which}", s0=str(s0_rc),
                                 door_frac=door, room_frac=room, tv=tv))
        fig = plt.figure(figsize=(2.4, 1.2), constrained_layout=True)
        ax = fig.add_subplot(111)
        draw_signed_panel(ax, diffs[name], vmax_abs, "", s0_rc)
        _save_to(fig, out_dir, f"contrast_{col_key[name]}", svg=True)
        tv_c = tv_half(diffs[name], diffs["true"])
        ratio = float(np.sum(np.abs(diffs[name]))) / float(np.sum(np.abs(diffs["true"])))
        stats[(name, "contrast")] = (ratio, tv_c)
        csv_rows.append(dict(gamma=GAMMA, k=k, d=d, method=col_key[name],
                             seed=(seed if name == "Density-FB" else -1),
                             variant=f"k{k}_start_contrast", s0=str(s0_rc), tv=tv_c,
                             contrast_ratio=ratio))

    # margin-vs-k, closed-form only, k marked
    fig = plt.figure(figsize=(3.0, 2.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    draw_margin_panel(ax, gap_rows, None, legend_loc="lower left")
    ax.axvline(k, color=MUTED, linestyle=":", linewidth=1.0, zorder=1)
    _save_to(fig, out_dir, "margin_vs_k")

    # colorbar files
    fig = plt.figure(figsize=(2.6, 1.05), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    for i, (name, cmap_name) in enumerate(COL_SPECS):
        cax = fig.add_subplot(gs[i, 0])
        cmap = mpl.colormaps[cmap_name]
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm_occ, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.set_ticks([t for t in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10) if t <= vmax_occ])
        cb.ax.tick_params(labelsize=5.0, color=MUTED, labelcolor=MUTED, length=2)
        if i < 2:
            cb.ax.set_xticklabels([])
        cb.ax.text(-0.02, 0.5, name, transform=cb.ax.transAxes, ha="right", va="center",
                   fontsize=6.0, color=INK)
    fig.suptitle(f"occupancy mass, one shared scale (PowerNorm {K40_POWER})", fontsize=6.0, color=MUTED)
    _save_to(fig, out_dir, "colorbar_occ")

    fig = plt.figure(figsize=(2.6, 0.55), constrained_layout=True)
    cax = fig.add_subplot(111)
    cmap = mpl.colormaps[DIVERGING_CMAP]
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(-vmax_abs, vmax_abs),
                                   orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.0, color=MUTED, labelcolor=MUTED, length=2)
    cb.set_label("fail minus exit (red = more mass after failing)", fontsize=6.0, color=MUTED)
    _save_to(fig, out_dir, "colorbar_contrast")

    # LaTeX block
    fb_r, fb_tv = stats[("FB", "contrast")]
    de_r, de_tv = stats[("Density-FB", "contrast")]
    tex = r"""%% Auto-generated by analysis/corridor_estimator/run.py --corridor14 -- do not hand-edit.
%% Requires \usepackage{subcaption} and \usepackage{graphicx}. Paths are relative to PAPER/ICLR/.
\begin{figure}[t]
  \centering
  \begin{minipage}[c]{0.66\linewidth}
    \centering
    \subcaptionbox{true, exit\label{fig:k40-exit-true}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/exit_true.pdf}}\hfill
    \subcaptionbox{FB, exit\label{fig:k40-exit-fb}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/exit_fb.pdf}}\hfill
    \subcaptionbox{Density-FB, exit\label{fig:k40-exit-density}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/exit_density.pdf}}\\[0.4em]
    \subcaptionbox{true, fail\label{fig:k40-fail-true}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/fail_true.pdf}}\hfill
    \subcaptionbox{FB, fail\label{fig:k40-fail-fb}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/fail_fb.pdf}}\hfill
    \subcaptionbox{Density-FB, fail\label{fig:k40-fail-density}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/fail_density.pdf}}\\[0.4em]
    \subcaptionbox{true, fail $-$ exit\label{fig:k40-contrast-true}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/contrast_true.pdf}}\hfill
    \subcaptionbox{FB, fail $-$ exit\label{fig:k40-contrast-fb}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/contrast_fb.pdf}}\hfill
    \subcaptionbox{Density-FB, fail $-$ exit\label{fig:k40-contrast-density}}[0.32\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/contrast_density.pdf}}\\[0.3em]
    \includegraphics[width=0.42\linewidth]{figures/corridor/k40/colorbar_occ.pdf}\hfill
    \includegraphics[width=0.42\linewidth]{figures/corridor/k40/colorbar_contrast.pdf}
  \end{minipage}\hfill
  \begin{minipage}[c]{0.32\linewidth}
    \centering
    \subcaptionbox{margin at $D$ vs.\ $k$\label{fig:k40-margin}}[\linewidth]{\includegraphics[width=\linewidth]{figures/corridor/k40/margin_vs_k.pdf}}
  \end{minipage}
  \caption{Corridor with exit, 14 corridor cells, agent starting at the first corridor cell (black dot; hollow marker = exit $D$, star = far room cell), at stickiness $k{=}%(k)d$, rank $d{=}%(d)d$, $\gamma{=}0.98$, closed-form fits (FB: rank-$d$ SVD optimum of its $\rho$-weighted squared loss; Density-FB: rank-$d$ softmax cross-entropy optimum, seed %(seed)d).
    Rows: occupancy after walking to $D$ and exiting (a--c), after walking to $D$ and taking a failing action (d--f), and the signed difference fail minus exit (g--i; red = more mass after failing).
    Occupancy maps share one PowerNorm(%(power)s) scale (left colorbar; hue = estimator); contrast maps share one symmetric scale fixed to the true contrast's largest entry (right colorbar).
    The number line under each map is the mass fraction on $D$ and inside the room.
    $|\Delta|/|\Delta_{\mathrm{true}}|$: FB %(fb_r).2f, Density-FB %(de_r).2f; TV of the contrast map to the true contrast: FB %(fb_tv).2f, Density-FB %(de_tv).2f.
    Right: normalized value margin at $D$ (exit minus mean failing action, divided by the true margin) versus $k$; dotted line = the $k$ shown on the left.}
  \label{fig:corridor-k40}
\end{figure}
""" % dict(k=k, d=d, seed=seed, power=K40_POWER, fb_r=fb_r, de_r=de_r, fb_tv=fb_tv, de_tv=de_tv)
    with open(out_dir / "subfigures.tex", "w") as f:
        f.write(tex)
    log(f"wrote {out_dir / 'subfigures.tex'}")

    for (name, which), v in stats.items():
        if which == "contrast":
            log(f"[k{k}] {name:11s} contrast: |D|/|D_true| {v[0]:.3f}  tv {v[1]:.3f}")
        else:
            log(f"[k{k}] {name:11s} {which:4s}: door {v[0]:.3f}  room {v[1]:.3f}  tv {v[2]:.3f}")
    return csv_rows


def run_corridor14():
    """--corridor14: 14-cell corridor. Sanity checks, closed-form k x d sweep (-> corridor14.csv,
    gap.csv-style, with the two variants' map numbers appended as extra rows), and the
    start / mid figures. TD training skipped (closed-form lines only on the right)."""
    assert CORRIDOR_LEN == 14
    run_sanity_checks()
    gap_rows, door_payload = run_main_sweep(csv_name="corridor14.csv")
    s0_start = (CORRIDOR_ROW, 1)                     # first corridor cell, 13 steps before D
    s0_mid = (CORRIDOR_ROW, CORRIDOR_LEN - 7)        # 7 steps before D
    extra = []
    for stem, s0 in [("fig_toy_corridor_start", s0_start), ("fig_toy_corridor_mid", s0_mid)]:
        stats = save_walk_figure(door_payload, gap_rows, None, s0, stem)
        variant = "start" if s0 == s0_start else "mid"
        for (name, row), (door, room, tv) in stats.items():
            method = {"true": "true", "FB": "fb", "Density-FB": "density"}[name]
            extra.append(dict(gamma=GAMMA, k=20, d=8, method=method, seed=(0 if method == "density" else -1),
                              variant=f"{variant}_{row.replace('then ', '')}", s0=str(s0),
                              door_frac=door, room_frac=room, tv=tv))
    save_start_contrast_figure(door_payload, gap_rows, s0_start)
    save_start_error_figure(door_payload, gap_rows, s0_start)
    extra += run_k40_panels(gap_rows, s0_start)
    csv_path = HERE / "corridor14.csv"
    fieldnames = list(gap_rows[0].keys()) + ["variant", "s0", "door_frac", "room_frac", "tv", "contrast_ratio"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows([dict(r, variant="sweep") for r in gap_rows])
        w.writerows(extra)
    log(f"wrote {csv_path} ({len(gap_rows)} sweep rows + {len(extra)} variant rows)")
    ks, fb_y, dmean, dlo, dhi, _ = collect_gap_ci(gap_rows, TRAIN_D)
    for i, k in enumerate(ks):
        log(f"[corridor14 margin d=8] k={k:2d}: FB {fb_y[i]:.3f} | Density {dmean[i]:.3f} [{dlo[i]:.3f},{dhi[i]:.3f}]")


def approach_rows(door_payload, p_mix=0.5):
    """Occupancy from S0 taking the action toward the door (right), under the policy
    'walk right along the corridor, take action a at D once, uniform random afterwards'.
    The walk S0 -> (3,7) -> D is deterministic, so its two terms are entered exactly;
    only the door row M(D, a, .) is the estimator's (the same k=20, d=8, seed-0 fits as
    the main figure). Normalized: (1-g)[e_pre + g e_D] + g^2 p(D, a, .), which sums to 1.
      row 0 'approach' : a = exit w.p. p_mix, fail (left) otherwise
      row 1 'then exit': a = right
      row 2 'then fail': a = left (sticky, k=20)
    Returns {col_name: [row0, row1, row2]} with FB rows raw (unclipped)."""
    g = GAMMA
    prefix = np.zeros(S)
    prefix[CELL_TO_IDX[PRE_D_RC]] = (1 - g)
    prefix[D] = (1 - g) * g

    def compose(p_exit, p_fail):
        r_exit = prefix + g * g * p_exit
        r_fail = prefix + g * g * p_fail
        r_mix = prefix + g * g * (p_mix * p_exit + (1 - p_mix) * p_fail)
        return [r_mix, r_exit, r_fail]

    return {
        "true": compose(door_payload["p_true_cross"], door_payload["p_true_fail"]),
        "FB": compose(door_payload["p_fb_cross"], door_payload["p_fb_fail"]),
        "Density-FB": compose(door_payload["p_density_cross"], door_payload["p_density_fail"]),
    }


def approach_tv(rows):
    """TV to the true row, per (column, row); FB on its raw (unclipped) rows."""
    out = {}
    for name in ("FB", "Density-FB"):
        for ridx, label in enumerate(APPROACH_ROW_LABELS):
            out[(name, label)] = tv_half(rows[name][ridx], rows["true"][ridx])
    return out


def draw_approach_block(fig, gs, rows):
    fracs = {}
    for cidx, (name, cmap_name) in enumerate(COL_SPECS):
        col_vals = [np.clip(v, 0.0, None) for v in rows[name]]
        vmax_col = max(float(np.nanmax(smooth_upsample(v))) for v in col_vals)
        for ridx, vals in enumerate(col_vals):
            ax = fig.add_subplot(gs[ridx, cidx])
            draw_toy_occupancy_panel(ax, vals, cmap_name, name if ridx == 0 else "",
                                     vmax=vmax_col, start_rc=S0_RC)
            if cidx == 0:
                ax.set_ylabel(APPROACH_ROW_LABELS[ridx], fontsize=7.8, color=INK)
            fracs[(name, APPROACH_ROW_LABELS[ridx])] = door_mass_fraction(vals)
    return fracs


def save_toy_corridor_approach_figure(door_payload, gap_rows, trained_rows):
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt

    rows = approach_rows(door_payload)
    tvs = approach_tv(rows)
    fig = plt.figure(figsize=(7.2, 3.3), constrained_layout=True)
    outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.75, 1.0])
    left_gs = outer[0].subgridspec(4, 3, hspace=0.10, wspace=0.06, height_ratios=[1, 1, 1, 0.40])
    fracs = draw_approach_block(fig, left_gs, rows)
    ax = fig.add_subplot(outer[1])
    draw_margin_panel(ax, gap_rows, trained_rows, legend_loc=None)
    leg_ax = fig.add_subplot(left_gs[3, :])
    leg_ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    leg_ax.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=5.8,
                  handlelength=1.6, columnspacing=1.2, borderaxespad=0.0)
    _save(fig, "fig_toy_corridor_approach", svg=True)
    log("[approach] door-mass fractions: " + ", ".join(f"{n}/{r}={v:.3f}" for (n, r), v in fracs.items()))
    log("[approach] TV to true: " + ", ".join(f"{n}/{r}={v:.3f}" for (n, r), v in tvs.items()))
    return fracs, tvs


def save_maps_subfigure(door_payload, which):
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt
    ridx = 0 if which == "exit" else 1
    fig = plt.figure(figsize=(6.0, 1.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.06)
    draw_maps_block(fig, gs, door_payload, rows=(ridx,), row_labels=True)
    _save(fig, f"maps_{which}")


def save_margin_subfigure(gap_rows, trained_rows):
    import matplotlib
    matplotlib.rcParams.update(RC_PARAMS)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3.2, 2.9), constrained_layout=True)
    ax = fig.add_subplot(111)
    draw_margin_panel(ax, gap_rows, trained_rows)
    _save(fig, "margin_vs_k")


def write_caption_tex(ci_method):
    ci_tex = ci_method.replace("%", r"\%")
    caption = (
        "At the corridor exit $D$, the true futures of exiting and failing to exit differ "
        "sharply: exiting spreads probability mass across the room while failing leaves the "
        "agent stuck at the exit cell. "
        "FB's fit assigns both actions a nearly identical, diffuse future, so its critic barely "
        "prefers the precise exit action over a failed one, and this margin collapses as the "
        "failed exit becomes more persistent (larger $k$). "
        "Density-FB keeps the two futures distinct and keeps the margin close to the truth at "
        "every $k$ in closed form (lines: FB is the deterministic rank-8 SVD optimum of its "
        "squared loss, Density-FB the cross-entropy optimum, mean over 5 seeds with a shaded "
        f"95\\% CI, {ci_tex}), and the TD-trained tabular versions of the same two losses "
        "(hollow diamonds: 20k steps on 100k random-policy transitions, mean and 95\\% CI over 5 "
        "seeds) preserve the ordering, though TD-trained FB is seed-noisy (its interval reaches "
        "zero at $k{=}1$, 10, 20) and TD-trained Density-FB overshoots the true margin; maps use "
        "$k{=}20$, $d{=}8$, seed 0, one color scale per column shared between its two rows."
    )
    caption_approach = (
        "% fig_toy_corridor_approach: the agent starts two cells before the exit\n"
        "Starting from $S_0$ (black dot), two corridor cells before the exit $D$ (hollow "
        "marker), the row labelled `approach' is the discounted occupancy under a policy that "
        "walks right to $D$, exits with probability one half and fails otherwise, then behaves "
        "uniformly at random; Density-FB reproduces it almost exactly (TV 0.01), while FB already "
        "puts too little mass on the exit cell (TV 0.07). "
        "Committing to the exit action at $D$ (`then exit') or to a failing action (`then fail', "
        "sticky, $k{=}20$) changes the true future sharply -- spread across the room versus "
        "stuck at the exit -- and Density-FB reproduces both, but FB's two rows stay nearly the "
        "same diffuse blob, so the effect of the action taken at the door is lost. "
        "The walk $S_0 \\to D$ is deterministic and entered exactly; only the door rows "
        "$M(D,a,\\cdot)$ come from the fits ($d{=}8$, seed 0, the same fits as the main figure), "
        "each column shares one color scale across its three rows, and the right panel is "
        "unchanged (normalized margin at $D$ versus $k$)."
    )
    walk_common = (
        "walks right to the exit $D$ (hollow marker) and then either exits (`then exit') or "
        "takes a failing action at $D$ (`then fail', sticky, $k{=}20$); the walk is deterministic "
        "and entered exactly, only the door rows $M(D,a,\\cdot)$ come from the fits ($d{=}8$, seed 0) "
        "on this 14-cell layout, each column shares one color scale across its two rows, and the "
        "numbers under each map are the mass fraction on $D$ and inside the room. "
        "The true futures differ (more mass stuck at the exit, less in the room, after failing); "
        "Density-FB reproduces both rows (TV 0.01--0.02), while FB's two rows are nearly the same "
        "diffuse blob (door 0.06 vs 0.08, TV 0.04 / 0.08), so the value of the action taken at "
        "the exit is lost. Right: the normalized margin at $D$ versus $k$ for this layout, "
        "closed-form fits only (FB deterministic rank-8 SVD; Density-FB mean over 5 seeds with a "
        "95\\% t-interval band)."
    )
    caption_start = (
        "% fig_toy_corridor_start: 14-cell corridor, agent starts at the first corridor cell\n"
        "On a 14-cell corridor, the agent starts at the first corridor cell (black dot, 13 steps "
        "before the exit), " + walk_common
    )
    caption_mid = (
        "% fig_toy_corridor_mid: 14-cell corridor, agent starts at the middle corridor cell\n"
        "On a 14-cell corridor, the agent starts at the middle corridor cell (black dot, 7 steps "
        "before the exit), " + walk_common
    )
    caption_contrast = (
        "% fig_toy_corridor_start_contrast: 14-cell corridor, start cell; rows exit / fail / (fail - exit)\n"
        "Same layout and start cell as the previous figure: the first two rows are the occupancy "
        "after walking to the exit and then exiting or failing (one PowerNorm($\\gamma{=}0.6$) scale "
        "per column, so the corridor walk is visible), and the third row is their signed difference, "
        "fail minus exit, on one symmetric diverging scale fixed to the true row's largest entry "
        "(red = more mass after failing, blue = more after exiting). "
        "The true contrast is a red spot at the exit cell and a blue room; Density-FB reproduces it "
        "($|\\Delta|/|\\Delta_{\\mathrm{true}}|$ = 1.00), while FB's contrast is fainter and "
        "spread into the room ($|\\Delta|/|\\Delta_{\\mathrm{true}}|$ = 0.84): its two futures "
        "differ by a diffuse smear rather than by mass at the exit, which is what the value margin "
        "at $D$ reads out. Right: closed-form normalized margin at $D$ versus $k$ for this layout."
    )
    caption_error = (
        "% fig_toy_corridor_start_error: 14-cell corridor, start cell; fitted minus true per cell\n"
        "Fitted minus true occupancy per cell for the two futures (rows) and the two estimators "
        "(columns) on the 14-cell corridor from the start cell, one shared symmetric diverging "
        "scale (red = the fit puts too much mass there, blue = too little), with the "
        "total-variation distance to the true map under each panel. "
        "FB's error is a blue deficit at the exit cell and a red surplus across the room, largest "
        "for the failing action (TV 0.08 vs 0.04 for exiting); Density-FB's error is an order of "
        "magnitude smaller (TV 0.01--0.02) and has no structure at the exit. "
        "Right: closed-form normalized margin at $D$ versus $k$ for this layout."
    )
    cap_path = HERE / "caption.tex"
    with open(cap_path, "w") as f:
        f.write("\n\n".join([caption, caption_approach, caption_start, caption_mid,
                              caption_contrast, caption_error]) + "\n")
    log(f"wrote {cap_path}")


def print_summary(gap_rows, trained_rows, fracs):
    log("\n=== normalized margin at D, d=8, gamma=0.98 (k x {FB closed, Density closed, "
        "FB trained, Density trained}) ===")
    ks, fb_y, dmean, dlo, dhi, _ = collect_gap_ci(gap_rows, TRAIN_D)
    for i, k in enumerate(ks):
        m_fb, lo_fb, hi_fb, _ = trained_gap_ci(trained_rows, "fb_td", k)
        m_d, lo_d, hi_d, _ = trained_gap_ci(trained_rows, "density_td", k)
        log(f"k={k:2d}: FB {fb_y[i]:.3f} | Density {dmean[i]:.3f} [{dlo[i]:.3f},{dhi[i]:.3f}] | "
            f"FB-TD {m_fb:.3f} [{lo_fb:.3f},{hi_fb:.3f}] | Density-TD {m_d:.3f} [{lo_d:.3f},{hi_d:.3f}]")
    log("\n=== FB closed-form normalized margin, k x d ===")
    for k in ks:
        vals = [next(float(r["norm_gap"]) for r in gap_rows
                     if r["method"] == "fb" and int(r["d"]) == d and int(r["k"]) == k) for d in DS]
        log(f"k={k:2d}: " + "  ".join(f"d={d}:{v:.3f}" for d, v in zip(DS, vals)))
    log("=== Density closed-form normalized margin (5-seed mean), k x d ===")
    for k in ks:
        vals = [np.mean([float(r["norm_gap"]) for r in gap_rows
                         if r["method"] == "density" and int(r["d"]) == d and int(r["k"]) == k]) for d in DS]
        log(f"k={k:2d}: " + "  ".join(f"d={d}:{v:.3f}" for d, v in zip(DS, vals)))
    log("\n=== door-cell mass fractions in the six maps (k=20, d=8, seed 0) ===")
    for (name, which), v in fracs.items():
        log(f"{name:11s} {which:4s}: {v:.3f}")
    log("=== TV to truth at k=20, d=8 ===")
    for r in gap_rows:
        if int(r["k"]) == 20 and int(r["d"]) == 8 and r["method"] in ("fb", "density") and int(r["seed"]) in (-1, 0):
            log(f"{r['method']:8s}: tv_exit={float(r['tv_cross']):.3f} tv_fail={float(r['tv_fail']):.3f}")


def door_payload_closed_form(k=20, d=8, seed=0):
    """The k=20, d=8, seed-0 map payload, recomputed with the sweep's exact density seed
    (used by --figures-only so figure iterations do not repeat the TD training)."""
    P_k = sticky_action_matrices(k)
    P_pi = policy_matrix_uniform(P_k)
    rho, _, _ = stationary(P_pi)
    M = successor_measure(P_k, P_pi, GAMMA)
    F_fb, B_fb = fb_fit(M, rho, d)
    F_d, B_d, _ = density_fit((1 - GAMMA) * M, d, seed=1000 * seed + d * 17 + k * 3, restarts=1)
    _, p_all = density_readout(F_d, B_d, R_ROOM, GAMMA)
    return dict(
        a_fail_show=A_FAIL_SHOW, density_seed_used=seed,
        p_true_cross=(1 - GAMMA) * M[ROW_D[A_CROSS]], p_true_fail=(1 - GAMMA) * M[ROW_D[A_FAIL_SHOW]],
        p_fb_cross=(1 - GAMMA) * fb_row(F_fb, B_fb, ROW_D[A_CROSS]) * rho,
        p_fb_fail=(1 - GAMMA) * fb_row(F_fb, B_fb, ROW_D[A_FAIL_SHOW]) * rho,
        p_density_cross=p_all[ROW_D[A_CROSS]], p_density_fail=p_all[ROW_D[A_FAIL_SHOW]])


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    t_start = time.time()
    log("=== Corridor-with-exit FB-vs-Density estimator demo ===")
    log(f"grid {GRID_R}x{GRID_C}, |S| = {S} free cells ({CORRIDOR_LEN} corridor + {ROOM_N}x{ROOM_N} room), "
        f"D (exit) = {D_RC}, a_cross = {A_CROSS}, a_fail = {A_FAIL}, G (star) = {G_RC}")
    if "--corridor14" in sys.argv:
        run_corridor14()
        log(f"\nTotal wall time: {time.time()-t_start:.1f}s")
        sys.exit(0)
    if "--figures-only" in sys.argv:
        gap_rows = _read_csv(HERE / "gap.csv")
        trained_rows = _read_csv(HERE / "gap_trained.csv")
        door_payload = door_payload_closed_form()
    else:
        run_sanity_checks()
        gap_rows, door_payload = run_main_sweep()
        trained_rows = run_trained_sweep()
    fracs, ci_method = save_toy_corridor_figure(door_payload, gap_rows, trained_rows)
    save_toy_corridor_approach_figure(door_payload, gap_rows, trained_rows)
    save_maps_subfigure(door_payload, "exit")
    save_maps_subfigure(door_payload, "fail")
    save_margin_subfigure(gap_rows, trained_rows)
    write_caption_tex(ci_method)
    print_summary(gap_rows, trained_rows, fracs)
    log(f"\nTotal wall time: {time.time()-t_start:.1f}s")
