#!/usr/bin/env python
"""scripts/rescuing_fb/smb_common.py -- shared machinery for the SMB eval suite
(PAPER/rescuing-fb/specs/11-smb-eval-suite.md, section 5).

WHY a separate module: eval_zeroshot_rewards.py is the proven zero-shot harness
and must keep its behavior/CLI untouched (spec 11 *wraps* it). Everything the
three SMB entry points (smb_tasks.py / eval_smb.py / eval_smb_measure.py) share
lives here:

  - the new reward families (occhoice T3 / union3 T4 / corridor T5);
  - trajectory scoring in the achieved-discounted-reward convention (gamma=0.99
    over the recorded post-step cube track -- spec 11 Wing A primary metric);
  - the MC dataset region statistics behind the T3 stable/transient selection
    (renewal_head.renewal_targets protocol: entries are first passages,
    consecutive in-ball steps are one sojourn, episode boundaries respected);
  - Wing-B query membership + calibration metrics (affine fit, relative error,
    union additivity defect);
  - the jax-side measure readout (each method's OWN critic: diagonal critics
    give the wrong-policy surrogate for off-diagonal queries, ZX critics give
    F(s0,a0,z_cond).B(x)/sqrt(d)) and a recording GPI rollout.

Module level is numpy-only (jax imports live inside functions) so the pure math
is unit-testable in the repo .venv.

Units: cube xyz = obs[19:22] in OBS units (1 unit = 10 cm) everywhere in this
module unless a name says otherwise; rollout trajectories recorded in METERS by
batched_rollout_record are converted with cube_m_to_ou. eps = 0.4 obs units =
4 cm (the FP hit ball); gamma = 0.99.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "third_party" / "ogbench" / "impls",
           REPO / "tools" / "wandb_mode_shim", REPO / "scripts",
           REPO / "scripts" / "rescuing_fb"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Proven machinery, imported not rewritten (all numpy-only at module level).
from eval_zeroshot_rewards import (  # noqa: E402,F401
    CEN, CUBE_OBS, CUBE_QPOS, EPS_OBS, EVAL_TASKS, GAMMA, SIGMA_OBS, THRESH,
    _policy_call, batched_rollout_record, cube_obs_units, discounted_reward,
    gpi_pick_actions, gpi_select, moment_z, reward_dense, reward_dirac,
    reward_linear, reward_mixture, ridge_z, sample_sphere_z,
    z_to_goal_proxy_idx,
)
from measure_readout_head import spearman, split_episodes  # noqa: E402,F401
from renewal_head import infer_z  # noqa: E402

TASK_JSON_VERSION = "smb_cube_single_v1"
ELEV_DZ = 0.8      # transient regions float 8 cm above the table (obs units)
AVOID_R = 0.5      # T5 avoid-ball radius, obs units (5 cm)
SPEC = "PAPER/rescuing-fb/specs/11-smb-eval-suite.md"


# ======================= units =====================================================

def cube_m_to_ou(cube_m: np.ndarray) -> np.ndarray:
    """Cube xyz meters (rollout qpos recordings) -> OBS units (obs[19:22])."""
    return (np.asarray(cube_m, np.float64) - CEN) * 10.0


def ou_to_m(cube_ou: np.ndarray) -> np.ndarray:
    return np.asarray(cube_ou, np.float64) / 10.0 + CEN


# ======================= region membership (pure numpy) ============================

def in_ball(cube_ou, center, eps: float = EPS_OBS) -> np.ndarray:
    d = np.linalg.norm(np.asarray(cube_ou, np.float64)
                       - np.asarray(center, np.float64), axis=-1)
    return d <= eps


def in_box(cube_ou, center, half) -> np.ndarray:
    c = np.asarray(cube_ou, np.float64)
    return (np.abs(c - np.asarray(center, np.float64))
            <= np.asarray(half, np.float64)).all(-1)


def query_member(query: dict, cube_ou) -> np.ndarray:
    """Frozen-battery query dict -> boolean membership over cube positions.
    Types: ball {center, eps}; box {center, half}; union2 {centers, eps}."""
    t = query["type"]
    if t == "ball":
        return in_ball(cube_ou, query["center"], query.get("eps", EPS_OBS))
    if t == "box":
        return in_box(cube_ou, query["center"], query["half"])
    if t == "union2":
        eps = query.get("eps", EPS_OBS)
        return np.any([in_ball(cube_ou, c, eps) for c in query["centers"]], axis=0)
    raise ValueError(f"unknown query type {t!r}")


# ======================= SMB reward families (spec 11 section 1) ===================

def reward_occhoice(cube_ou, g_stable, g_transient, w_s: float, w_t: float,
                    eps: float = EPS_OBS) -> np.ndarray:
    """T3: r(x) = w_S 1[x in B(g_S)] + w_T 1[x in B(g_T)]. Additive; the balls
    are disjoint by construction (centers differ in z by ELEV_DZ = 2*eps)."""
    return (float(w_s) * in_ball(cube_ou, g_stable, eps).astype(np.float64)
            + float(w_t) * in_ball(cube_ou, g_transient, eps).astype(np.float64))


def reward_union(cube_ou, centers, eps: float = EPS_OBS) -> np.ndarray:
    """T4 union3 (also Wing-B union2): r = 1[x in any of the eps-balls]."""
    return np.any([in_ball(cube_ou, c, eps) for c in centers], axis=0).astype(np.float64)


def reward_corridor(cube_ou, g_a, c_avoid, lam: float, eps: float = EPS_OBS,
                    avoid_r: float = AVOID_R) -> np.ndarray:
    """T5 corridor reach-avoid: r = 1[x in B_eps(g_A)] - lam * 1[x in B_avoid]."""
    return (in_ball(cube_ou, g_a, eps).astype(np.float64)
            - float(lam) * in_ball(cube_ou, c_avoid, avoid_r).astype(np.float64))


# ======================= trajectory scoring ========================================

def per_episode_discounted(cube_traj_ou, lengths, reward_fn,
                           gamma: float = GAMMA) -> np.ndarray:
    """Achieved discounted reward per episode: sum_t gamma^t r(c(s_{t+1})) over
    the REAL (pre-termination) steps of the recorded post-step cube track.
    reward_fn maps [L,3] cube positions (obs units) -> [L] rewards."""
    out = np.zeros(len(cube_traj_ou))
    for i in range(len(cube_traj_ou)):
        L = int(lengths[i])
        if L == 0:
            continue
        r = np.asarray(reward_fn(np.asarray(cube_traj_ou[i, :L], np.float64)), np.float64)
        out[i] = float((r * gamma ** np.arange(L)).sum())
    return out


def discounted_time_in(cube_traj_ou, lengths, member_fn,
                       gamma: float = GAMMA) -> np.ndarray:
    """T7 dwell column: per-episode discounted time-in-region."""
    return per_episode_discounted(
        cube_traj_ou, lengths, lambda c: member_fn(c).astype(np.float64), gamma)


def frac_time_in_ball(cube_traj_ou, lengths, center, radius: float) -> np.ndarray:
    """Undiscounted per-episode fraction of real steps inside the ball (T5's
    time-in-B metric, matching the old reachavoid read)."""
    out = np.zeros(len(cube_traj_ou))
    for i in range(len(cube_traj_ou)):
        L = int(lengths[i])
        out[i] = float(in_ball(cube_traj_ou[i, :L], center, radius).mean()) if L else 0.0
    return out


def final_in_any_ball(cube_traj_ou, lengths, centers, eps: float = EPS_OBS) -> np.ndarray:
    """T4 success flag: episode ENDS inside any of the balls."""
    out = np.zeros(len(cube_traj_ou))
    centers = np.asarray(centers, np.float64)
    for i in range(len(cube_traj_ou)):
        L = int(lengths[i])
        if L:
            out[i] = float(in_ball(centers, cube_traj_ou[i, L - 1], eps).any())
    return out


def chosen_region(cube_traj_ou, lengths, g_s, g_t, eps: float = EPS_OBS,
                  gamma: float = GAMMA):
    """Per-episode region the rollout actually committed to: the ball with the
    larger (unweighted) discounted dwell; 'none' if neither is ever entered.
    Returns (labels list, dwell_S [n], dwell_T [n])."""
    d_s = discounted_time_in(cube_traj_ou, lengths, lambda c: in_ball(c, g_s, eps), gamma)
    d_t = discounted_time_in(cube_traj_ou, lengths, lambda c: in_ball(c, g_t, eps), gamma)
    labels = []
    for a, b in zip(d_s, d_t):
        if a <= 0 and b <= 0:
            labels.append("none")
        else:
            labels.append("stable" if a >= b else "transient")
    return labels, d_s, d_t


# ======================= MC dataset region statistics (T3 selection) ==============

def mc_region_stats(cube, spans, centers, gamma: float = GAMMA,
                    eps: float = EPS_OBS, g_chunk: int = 32,
                    t_chunk: int = 100_000) -> dict:
    """Per candidate eps-ball, from dataset trajectories (episode boundaries =
    `spans`, respected exactly as renewal_head.renewal_targets does):

      occ_mean   MC post-arrival discounted occupancy: mean over ENTRIES e of
                 sum_{t >= e, t < ep_end} gamma^(t-e) 1[cube_t in ball]
                 (an entry is the first step of an in-ball sojourn; nan if the
                 ball is never entered)
      n_entries  total entries across episodes
      first_med  median (over episodes that ever hit) first-arrival step index
                 from the episode START -- the reach proxy is gamma^first_med
      hit_frac   fraction of episodes with >= 1 in-ball step
      support    number of dataset rows inside the ball
    """
    cube = np.asarray(cube, np.float64)
    centers = np.asarray(centers, np.float64)
    n = len(centers)
    occ = np.full(n, np.nan)
    nent = np.zeros(n, np.int64)
    first_med = np.full(n, np.nan)
    hit_frac = np.zeros(n)
    support = np.zeros(n, np.int64)
    n_ep = max(len(spans), 1)
    for i0 in range(0, n, g_chunk):
        gg = centers[i0:i0 + g_chunk]
        inball = np.empty((len(gg), len(cube)), bool)
        for t0 in range(0, len(cube), t_chunk):
            c = cube[t0:t0 + t_chunk]
            d2 = ((gg[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            inball[:, t0:t0 + t_chunk] = d2 <= eps * eps
        support[i0:i0 + len(gg)] = inball.sum(1)
        for k in range(len(gg)):
            vals, firsts, hits = [], [], 0
            for a, b in spans:
                idx = np.nonzero(inball[k, a:b])[0]
                if len(idx) == 0:
                    continue
                hits += 1
                firsts.append(int(idx[0]))
                first = np.ones(len(idx), bool)
                first[1:] = (idx[1:] - idx[:-1]) > 1
                for e in idx[first]:
                    later = idx[idx >= e]
                    vals.append(float(np.sum(gamma ** (later - e))))
            if vals:
                occ[i0 + k] = float(np.mean(vals))
                nent[i0 + k] = len(vals)
                first_med[i0 + k] = float(np.median(firsts))
            hit_frac[i0 + k] = hits / n_ep
    return {"occ_mean": occ, "n_entries": nent, "first_med": first_med,
            "hit_frac": hit_frac, "support": support}


# ======================= calibration metrics (Wing B) ==============================

def affine_fit(pred, gt):
    """Global least-squares (a, b) with gt ~ a*pred + b. Degenerate -> (1, 0)."""
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    if len(pred) < 2 or np.std(pred) == 0:
        return 1.0, 0.0
    A = np.vstack([pred, np.ones_like(pred)]).T
    a, b = np.linalg.lstsq(A, gt, rcond=None)[0]
    return float(a), float(b)


def median_rel_err(pred, gt, floor: float = 1e-3):
    """Median |pred - gt| / gt over points with gt > floor (relative error is
    meaningless at zero occupancy). Returns (median or None, n_used)."""
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    mask = gt > floor
    if not mask.any():
        return None, 0
    return float(np.median(np.abs(pred[mask] - gt[mask]) / gt[mask])), int(mask.sum())


def additivity_defect(m_union: float, m_1: float, m_2: float, gt_union: float,
                      floor: float = 1e-3) -> float:
    """Spec 11 B2: |M(X1 u X2) - M(X1) - M(X2)| / max(gt(X1 u X2), floor)."""
    return float(abs(m_union - (m_1 + m_2)) / max(float(gt_union), floor))


def regression_slope_r2(x, y):
    """(slope, intercept, r2) of y ~ slope*x + b (B3's calibrated-value read)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 3 or np.std(x) == 0:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(res[0]) if len(res) else float(((y - (slope * x + b)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(b), float(r2)


# ======================= readout math + z inference ================================

def pair_min_head_dot(phi, psi, d: int) -> np.ndarray:
    """min over ensemble heads of phi_h @ psi_h^T / sqrt(d): the critic's own
    value convention (v [2,B] -> min over heads), as an [N,G] outer product.
    phi [2,N,d] or [N,d]; psi [2,G,d] or [G,d] (singleton sides broadcast)."""
    phi = np.asarray(phi, np.float64)
    psi = np.asarray(psi, np.float64)
    if phi.ndim == 2:
        phi = phi[None]
    if psi.ndim == 2:
        psi = psi[None]
    if phi.shape[0] != psi.shape[0]:
        H = max(phi.shape[0], psi.shape[0])
        phi = np.broadcast_to(phi, (H,) + phi.shape[1:])
        psi = np.broadcast_to(psi, (H,) + psi.shape[1:])
    vals = np.einsum("hnd,hgd->hng", phi, psi) / np.sqrt(float(d))
    return vals.min(axis=0)


def infer_z_route(Psi, rewards, route: str, match_norm: float, R_vals=None,
                  retrieval_k: int = 8, ridge_lambda: float = 1e-2):
    """eval_zeroshot_rewards.infer_cell's logic factored out of its closure so
    eval_smb can reuse it verbatim (moment = renewal_head.infer_z pinned math;
    ridge = spec 08 E3 GLS; retrieval = top-k on-manifold mean).
    Returns (z float32 or None, diag dict); z is None when degenerate."""
    Psi = np.asarray(Psi, np.float64)
    r = np.asarray(rewards, np.float64)
    z_raw = moment_z(Psi, r, R_vals)
    raw_norm = float(np.linalg.norm(z_raw))
    diag = {"route": route, "n_pos": int((r > 0).sum()), "n_neg": int((r < 0).sum()),
            "z_raw_norm": raw_norm, "match_data_norm": float(match_norm)}
    if raw_norm < 1e-12:
        return None, diag
    if route == "retrieval":
        w = r if R_vals is None else r * np.asarray(R_vals, np.float64)
        idx = np.argsort(w)[-int(retrieval_k):]
        idx = idx[w[idx] > 0]
        if len(idx) == 0:
            return None, diag
        zr = Psi[idx].mean(0)
        z = zr * match_norm / (np.linalg.norm(zr) + 1e-8)
        diag["retrieval_n"] = int(len(idx))
    elif route == "ridge":
        z = ridge_z(Psi, r, R_vals, match_norm, ridge=ridge_lambda)
        diag["ridge_lambda"] = float(ridge_lambda)
    else:
        assert route == "moment", f"unknown z-inference route {route!r}"
        z = np.asarray(infer_z(Psi, r, R_vals, match_norm), np.float64)
    diag["z_norm"] = float(np.linalg.norm(z))
    return z.astype(np.float32), diag


def ball_candidate_indices(cube_ou, center, eps: float = EPS_OBS, k: int = 8):
    """Dataset-sample indices representing an eps-ball as a policy candidate:
    the up-to-k in-ball samples nearest the center. Falls back to the single
    nearest sample when the ball has no support in the sample (ok=False so the
    caller can record the fallback). Deterministic (stable sort)."""
    d = np.linalg.norm(np.asarray(cube_ou, np.float64)
                       - np.asarray(center, np.float64), axis=-1)
    inside = np.nonzero(d <= eps)[0]
    if len(inside) == 0:
        return np.array([int(np.argmin(d))]), False
    order = inside[np.argsort(d[inside], kind="stable")]
    return order[:int(k)], True


def to_native(obj):
    """Recursive numpy -> native-python conversion for deterministic json."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ======================= jax-side plumbing =========================================

def make_measure_readout(agent, d: int, fbz: bool, act_dim: int):
    """Batched M-readout in each method's OWN critic space (spec 11 Wing B):

      diagonal critics (fb_forward_z=False):
          M_hat(s0, a0, g) = min-head phi(s0,a0).psi(g)/sqrt(d) == critic(s0,g,a0)
          -- for off-diagonal queries this is the WRONG-POLICY surrogate
          (the query column is evaluated under the query's own conditioning);
          that mismatch is the B2 measurement, documented in the output.
      ZX critics (fb_forward_z=True):
          M_hat = min-head F(s0,a0,z_cond).B(g)/sqrt(d)  (z via the z_vec path).

    Returns readout(s_obs [N,obs], acts [N,act], goal_obs [G,obs],
    z_cond None|[d]) -> [N,G] numpy."""
    import jax.numpy as jnp

    def readout(s_obs, acts, goal_obs, z_cond=None, chunk=4096):
        s = jnp.asarray(np.asarray(s_obs, np.float32))
        a = jnp.asarray(np.asarray(acts, np.float32))
        if fbz:
            assert z_cond is not None, "ZX readout needs the conditioning z"
            zb = jnp.broadcast_to(
                jnp.asarray(np.asarray(z_cond, np.float32))[None], (s.shape[0], int(d)))
            _v, phi, _ps = agent.network.select("critic")(s, s, a, s, info=True, z_vec=zb)
        else:
            _v, phi, _ps = agent.network.select("critic")(s, s, a, info=True)
        phi = np.asarray(phi)                                   # [2,N,d] (or [N,d])
        goal_obs = np.asarray(goal_obs, np.float32)
        psis = []
        for i in range(0, len(goal_obs), chunk):
            g = jnp.asarray(goal_obs[i:i + chunk])
            a0 = jnp.zeros((g.shape[0], int(act_dim)), jnp.float32)
            if fbz:
                _v, _f, ps = agent.network.select("critic")(g, g, a0, g, info=True)
            else:
                _v, _f, ps = agent.network.select("critic")(g, g, a0, info=True)
            psis.append(np.asarray(ps))
        psi = np.concatenate(psis, axis=-2)                     # [2,G,d] (or [G,d])
        return pair_min_head_dot(phi, psi, d)

    return readout


def batched_gpi_rollout_record(agent, q_at_z, envs, z_r, cand_conds, H=200, mode="z"):
    """eval_zeroshot_rewards.batched_gpi_rollout with the full post-step obs
    trajectory ALSO recorded (the T6 best-candidate control scores the linear
    reward off psi(s') along the rollout). Same proven protocol otherwise:
    env-major batched policy/critic calls, argmax through gpi_select /
    gpi_pick_actions, critic readout ALWAYS at z_r.
    Returns (cube_traj [n,H,3] METERS, obs_traj [n,H,obs], lengths [n],
    switch_rate [n], choice_counts [C])."""
    import jax
    import jax.numpy as jnp
    n = len(envs)
    C = len(cand_conds)
    Zc = np.asarray(cand_conds, np.float32)
    z_r = np.asarray(z_r, np.float32)
    obs_dim = envs[0].observation_space.shape[0]
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    prev_j = -np.ones(n, int)
    switches = np.zeros(n, float)
    steps_taken = np.zeros(n, float)
    choice_counts = np.zeros(C, float)
    rng = jax.random.PRNGKey(0)
    z_r_j = jnp.asarray(z_r)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        obs_big = np.repeat(obs, C, axis=0)
        z_big = np.tile(Zc, (n, 1))
        acts_big = _policy_call(agent, mode, jnp.asarray(obs_big),
                                jnp.asarray(z_big), sub)
        acts_big = np.clip(acts_big, -1, 1)
        qmin = np.asarray(q_at_z(jnp.asarray(obs_big), jnp.asarray(acts_big), z_r_j))
        jstar = gpi_select(qmin, n, C)
        acts_exec = gpi_pick_actions(acts_big, jstar, n, C)
        for i, e in enumerate(envs):
            if not active[i]:
                traj[i, t] = traj[i, t - 1]
                continue
            choice_counts[jstar[i]] += 1
            if prev_j[i] >= 0 and jstar[i] != prev_j[i]:
                switches[i] += 1
            prev_j[i] = jstar[i]
            steps_taken[i] += 1
            o, _r, term, trunc, _ = e.step(acts_exec[i])
            obs[i] = o
            traj[i, t] = e.unwrapped.data.qpos[CUBE_QPOS]
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    switch_rate = switches / np.maximum(steps_taken - 1, 1)
    return traj, obs_traj, lengths, switch_rate, choice_counts


def capture_reset_obs(envs) -> np.ndarray:
    """The exact s0 batch batched_rollout_record will see: it resets each env
    with reset(seed=i) internally, and OGBench resets are seed-deterministic,
    so resetting here first records the same initial states the subsequent
    rollout replays. Returns [n, obs_dim] float32."""
    obs = np.zeros((len(envs), envs[0].observation_space.shape[0]), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    return obs
