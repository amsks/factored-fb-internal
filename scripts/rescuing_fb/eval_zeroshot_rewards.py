#!/usr/bin/env python
"""scripts/rescuing_fb/eval_zeroshot_rewards.py — Spec 07 item 3: the zero-shot
reward-capability eval harness (PAPER/rescuing-fb/specs/07-fp-capability-and-renewal-measure.md).

One harness, run against (i) the frozen FP-NoOrtho 0.967 checkpoints
(inference-only) and (ii) the item-1 TPActor checkpoints. Task families (ALL in
the [0,1] reward convention of the FP corner — NOT gc_negative):

  dirac      single-goal indicator rewards (the 5 OGBench cube-single tasks);
             must reproduce the run's native eval (~0.967 for the FP cells).
  mixture    two-goal max-mixtures r = max_k 1[goal_k]; --gpi adds GPI over the
             candidate set {z_r, psi(g1), psi(g2)} (t6_gpi_eval protocol).
  region     r = 1[cube in region] (T6 region protocol: 6cm box OR 4cm ball).
  dense      r(g) = exp(-||cube(g) - c*|| / sigma), sigma = 0.8 obs units
             (= 2*eps = 8cm); success = the OGBench criterion at c*.
  linear     r(s') = sg[psi(s')].z_tilde / sqrt(d), z_tilde ~ sphere x
             match-data norm; reports achieved discounted reward under rollout
             (gamma=0.99) vs a random-z control — no success-rate read exists.
  reachavoid r = 1[cube in A] - 1[cube in B] (A=R1, B=R2, pre-registered);
             reports success-at-A and fraction-of-time-in-B. Exploratory.

Protocol: match-data norm rule throughout (eval_rgfb_zeroshot.py conventions);
z inference via renewal_head.infer_z (spec 07 item 2c; --z-weighting renewal
multiplies each reward sample by the renewal factor R(g_j)); rollouts are
batched lockstep envs (t6_generality_probe.py); the GPI path is forked from
t6_gpi_eval.py (spec 06 item 1). Every cell reports z-inference diagnostics:
cos(z_r, psi(g*)) where applicable and ||z_r||_raw vs the match-data norm.

Actor conditioning (read from flags.json):
  actor_cond='z'    -> the actor consumes z_r directly (sample_actions_with_z).
  actor_cond='goal' -> the frozen FP-NoOrtho 0.967 cells: the flow actor's
    conditioning leg is the RAW goal obs (the critic-internal z=sg[psi(g)] is
    NOT an actor input), so z_r is DECODED to a goal-proxy observation
    g_hat = argmax_j psi(s'_j).z_r over the relabel sample (the critic's own
    readout of which dataset state best represents the task) and the actor
    rolls at g_hat. GPI candidates are then goal observations; the critic
    readout stays at z_r (phi(s,a) is goal-independent for value_arch=bilinear).
    The g_hat decode is reported per cell (proxy cube xyz + dist to c* where
    defined) so the extra inference step is auditable.

Run (CPU rollouts; box is shared — cap cores):
  taskset -c 0-47 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_zeroshot_rewards.py \\
    --run-dir '/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/*s0*' \\
    --tasks 1,2,3,4,5 --episodes 25 --horizon 200 --gpi \\
    --z-weighting renewal --renewal-head analysis/rescuing-fb/renewal_head_s0.npz

Spec 14 §7 (2026-08-03): MULTI-ENV port (cube-double, scene). The env comes from
the checkpoint's flags.json (env_name, --env overrides) and the measure
coordinate from env_geometry.geometry_for(env, agent.measure_hit_spec) — the
geometry travels with the weights (spec 14 §3.4). Cube-single keeps every
archived code path bit-for-bit (regression gate: the dirac/region families must
reproduce analysis/rescuing-fb/zeroshot_infonly_s2_RET_k1.json). Off cube-single:
  * families restricted to dirac / region / dense;
  * goal coordinate = HitSpec.coords of the task GOAL OBSERVATION — this fixes
    the two documented silent wrong-answer bugs (spec 14 §2):
    goal_xyzs.reshape(-1)[:3] truncated cube-double to cube 0, and scene has no
    goal_xyzs key at all (that line now only runs on cube-single, where it is
    exact);
  * dirac/dense success = the env's own info['success'] (native OGBench
    criterion; these envs terminate at goal, so any-time == final);
  * region rewards + success are CONJUNCTIVE (HitSpec.contains at the trained
    radii) with VISITED-STATE hindsight centers (env_geometry probe policy —
    off-diagonal conjunctive hit rate is 0.046% on cube-double, spec 14 §2.1),
    rolled out on the first selected task variant;
  * checkpoint restore via t3_offpeak_multienv.load_agent_dims (domain-correct
    example dims; load_agent_flexible hardcoded a cube-single example env);
  * --horizon 0 = the env's own max_episode_steps (cube-double 500, scene 750 —
    a 200-step cap would depress success vs the native eval.csv).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

# Proven machinery, imported not rewritten (both are numpy-only at module level):
#   t6_gpi_eval (spec 06 item 1): checkpoint restore, psi/critic readouts, metrics.
#   t6_generality_probe (T6): region protocol constants + region metric.
from t6_gpi_eval import (  # noqa: E402
    cosine,
    decode_cube,
    m_goal,
    m_pair,
    make_psi,
    make_q_at_z,
    rescale,
)
from t6_generality_probe import BOX_HALF, REGIONS, m_region  # noqa: E402

# Spec 14 §3.1: the single definition of the measure coordinate + region rule.
from env_geometry import episode_bounds, geometry_for, hindsight_rows  # noqa: E402

CUBE_SINGLE = "cube-single-play-v0"
EVAL_TASKS = {i: f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)}
CUBE_QPOS = slice(14, 17)   # privileged cube xyz, meters (E1a convention)
CUBE_OBS = slice(19, 22)    # obs cube xyz = (m - CEN) * 10  (1 obs unit = 10 cm)
CEN = np.array([0.425, 0.0, 0.0])
THRESH = 0.04               # OGBench goal-reach radius, meters (= EPS_OBS in obs units)
EPS_OBS = 0.4               # FP hit radius, obs units (gciql_flow.py measure_loss, ~L973)
SIGMA_OBS = 0.8             # dense-family scale = 2*eps (spec 07 item 3 family 4)
GAMMA = 0.99
REACH_A, REACH_B = "R1", "R2"   # reach-avoid regions (pre-registered, exploratory)
ALL_FAMILIES = ("dirac", "mixture", "region", "dense", "linear", "reachavoid")


# ======================= reward-sample construction (pure numpy) ==================
# Reward samples are built over dataset goal states; each function maps decoded
# cube positions -> per-sample reward in the [0,1] convention (reachavoid: [-1,1]).

def cube_obs_units(obs: np.ndarray) -> np.ndarray:
    """Dataset/env obs -> cube xyz in OBS UNITS (obs[19:22]; 1 unit = 10 cm)."""
    return np.asarray(obs, np.float64)[..., CUBE_OBS]


def reward_dirac(cube_obs: np.ndarray, goal_obs_xyz: np.ndarray,
                 eps: float = EPS_OBS) -> np.ndarray:
    """1[||cube - g*|| <= eps] in obs units — the FP hit convention
    (gciql_flow.py first-passage hit matrix: <= 0.4 obs units = 4 cm)."""
    d = np.linalg.norm(np.asarray(cube_obs, np.float64)
                       - np.asarray(goal_obs_xyz, np.float64), axis=-1)
    return (d <= eps).astype(np.float64)


def reward_mixture(cube_obs: np.ndarray, goal_obs_xyzs, eps: float = EPS_OBS) -> np.ndarray:
    """max_k 1[goal_k] — the two-goal (K-goal) max-mixture reward."""
    rs = [reward_dirac(cube_obs, g, eps) for g in goal_obs_xyzs]
    return np.max(np.stack(rs, 0), axis=0)


def reward_region(cube_m: np.ndarray, center_m: np.ndarray,
                  box_half: float = BOX_HALF, thresh: float = THRESH) -> np.ndarray:
    """T6 region protocol (t6_generality_probe (c)): inside the 6cm box OR the
    4cm ball around the region center. Inputs in METERS."""
    cube_m = np.asarray(cube_m, np.float64)
    center_m = np.asarray(center_m, np.float64)
    in_box = (np.abs(cube_m - center_m) <= box_half).all(-1)
    in_ball = np.linalg.norm(cube_m - center_m, axis=-1) < thresh
    return (in_box | in_ball).astype(np.float64)


def reward_dense(cube_obs: np.ndarray, cstar_obs: np.ndarray,
                 sigma: float = SIGMA_OBS) -> np.ndarray:
    """exp(-||cube - c*|| / sigma), obs units — in (0, 1], =1 at c*."""
    d = np.linalg.norm(np.asarray(cube_obs, np.float64)
                       - np.asarray(cstar_obs, np.float64), axis=-1)
    return np.exp(-d / sigma)


def reward_linear(Psi: np.ndarray, z_tilde: np.ndarray) -> np.ndarray:
    """r(s') = psi(s') . z_tilde / sqrt(d) — the critic's own readout form."""
    Psi = np.asarray(Psi, np.float64)
    return Psi @ np.asarray(z_tilde, np.float64) / np.sqrt(Psi.shape[-1])


def reward_reachavoid(cube_m: np.ndarray, center_a_m: np.ndarray,
                      center_b_m: np.ndarray) -> np.ndarray:
    """1[cube in A] - 1[cube in B]  (in {-1, 0, +1}; overlap cancels to 0)."""
    return reward_region(cube_m, center_a_m) - reward_region(cube_m, center_b_m)


def sample_sphere_z(rng: np.random.Generator, d: int, norm: float) -> np.ndarray:
    """z_tilde ~ uniform on the sphere, scaled to `norm` (match-data rule)."""
    v = rng.normal(size=d)
    return v * norm / (np.linalg.norm(v) + 1e-12)


def z_to_goal_proxy_idx(Psi: np.ndarray, z_r: np.ndarray) -> int:
    """Decode z_r to a goal-proxy DATASET STATE for goal-conditioned actors
    (actor_cond='goal'): j* = argmax_j cos(psi(s'_j), z_r). COSINE, not the raw
    dot product — ||psi|| is task-scale-dependent (no project_z), and the raw
    dot argmax is dominated by large-||psi|| states (verified on the FP-NoOrtho
    s0 ckpt: dot top-10 contains 0 goal-ball states, cos ranks the goal-ball
    state first). Pure numpy (unit-testable)."""
    Psi = np.asarray(Psi, np.float64)
    z = np.asarray(z_r, np.float64)
    scores = Psi @ z / (np.linalg.norm(Psi, axis=-1) * np.linalg.norm(z) + 1e-12)
    return int(np.argmax(scores))


# ======================= multi-env geometry (spec 14 §7; pure numpy) ==============

def eval_tasks_for(env_name: str) -> dict:
    """{task_id: singletask env id} for any OGBench '<base>-v0' play env."""
    assert env_name.endswith("-v0"), f"unexpected env name {env_name!r}"
    base = env_name[: -len("-v0")]
    return {i: f"{base}-singletask-task{i}-v0" for i in range(1, 6)}


def reward_conj_dirac(coords: np.ndarray, center: np.ndarray, spec,
                      mult: float = 1.0) -> np.ndarray:
    """1[every hit-spec block within mult*eps_block of center] — the trained
    conjunctive indicator (spec 14 §3.2) over the FULL region coordinate.
    Reduces to reward_dirac exactly on a single-block spec."""
    return spec.contains(coords, center, mult=mult).astype(np.float64)


def reward_conj_dense(coords: np.ndarray, center: np.ndarray, spec,
                      sigma_scaled: float = 2.0) -> np.ndarray:
    """exp(-d/sigma) in UNIT-RADIUS coordinates (each column / its block eps).
    On cube-single (one block, eps 0.4) this equals the archived reward_dense
    exactly: d_scaled = d_ou/0.4 and sigma_scaled=2 <=> SIGMA_OBS=0.8=2*eps."""
    d = np.linalg.norm(spec.scaled(np.atleast_2d(coords))
                       - spec.scaled(np.atleast_2d(center)), axis=-1)
    return np.exp(-d / sigma_scaled)


def draw_region_centers(coords_all: np.ndarray, bounds: np.ndarray, spec,
                        relabel_coords: np.ndarray, rng, k: int,
                        min_pos: int = 5, max_tries: int = 400):
    """k region centers = coordinates of VISITED dataset states drawn by the
    env_geometry hindsight rule (a row strictly later in a random start row's
    own episode), rejecting centers with < min_pos positive relabel samples at
    the trained radii: off-diagonal conjunctive hit rates are ~0.046% on
    cube-double (spec 14 §2.1), so unchecked centers can leave the z-inference
    moment degenerate. Returns (centers [m<=k, D], n_pos [m], n_rejected)."""
    centers, n_pos, rejected = [], [], 0
    n_rows = len(coords_all)
    for _ in range(int(max_tries)):
        if len(centers) == k:
            break
        start = int(rng.integers(0, n_rows))
        rows = hindsight_rows(bounds, start, rng, 1)
        if len(rows) == 0:                      # start at its episode's end
            rejected += 1
            continue
        c = coords_all[int(rows[0])]
        npos = int(spec.contains(relabel_coords, c).sum())
        if npos < min_pos:
            rejected += 1
            continue
        centers.append(c)
        n_pos.append(npos)
    return (np.asarray(centers, np.float64), np.asarray(n_pos, int), rejected)


def region_success_from_obs(obs_traj: np.ndarray, lengths: np.ndarray,
                            center: np.ndarray, spec):
    """(final, any) conjunctive region membership per episode, computed on the
    full geometry coordinate of the recorded obs trajectory."""
    fin, anys = [], []
    for i in range(len(obs_traj)):
        L = int(lengths[i])
        if L == 0:
            fin.append(0.0)
            anys.append(0.0)
            continue
        hits = spec.contains(spec.coords(obs_traj[i, :L]), center)
        fin.append(float(hits[L - 1]))
        anys.append(float(hits.any()))
    return np.array(fin), np.array(anys)


# ======================= z inference (spec 07 item 2c interface) ==================

def moment_z(Psi: np.ndarray, rewards: np.ndarray, R_vals=None) -> np.ndarray:
    """Raw (un-normalized) renewal-weighted moment: mean_j r_j R(g_j) psi(g_j).
    Used only for diagnostics (||z_raw||, cosines); the deployed z comes from
    infer_z (renewal_head.py pinned interface), which renormalizes."""
    Psi = np.asarray(Psi, np.float64)
    w = np.asarray(rewards, np.float64)
    if R_vals is not None:
        w = w * np.asarray(R_vals, np.float64)
    return (w[:, None] * Psi).mean(0)


def ridge_z(Psi, rewards, R_vals, match_norm, ridge=1e-3):
    """GLS / whitened moment (spec 08 E3): z = (Sigma + lam I)^{-1} b, with feature
    covariance Sigma = E_n[psi psi^T], moment b = E_n[r (R) psi], and lam = ridge *
    tr(Sigma)/d (feature-scale-relative so `ridge` is dimensionless). Reduces to the
    moment DIRECTION when Sigma is isotropic; corrects the moment's anisotropy bias under
    a regularizer-free (no-ortho) psi -- the suspected cause of the dense-reward z_r
    misaim. Mirrors evals/z_inference.infer_l1; renormalized to match_norm for deployment
    (the raw GLS magnitude is uninformative under anisotropic features)."""
    Psi = np.asarray(Psi, np.float64)
    b = moment_z(Psi, rewards, R_vals)
    n, d = Psi.shape
    Sigma = Psi.T @ Psi / n
    lam = float(ridge) * (np.trace(Sigma) / d)
    z = np.linalg.solve(Sigma + lam * np.eye(d), b)
    return (z * match_norm / (np.linalg.norm(z) + 1e-8)).astype(np.float64)


def _fallback_infer_z(psi, rewards, R_vals, match_norm):
    """Local copy of the PINNED renewal_head.infer_z contract (spec 07 item 2c):
    z = mean_j rewards[j] * (R_vals[j] if R_vals is not None else 1) * psi[j],
    renormalized to L2 norm match_norm. Used only until renewal_head.py lands;
    must stay bit-compatible with it."""
    z = moment_z(psi, rewards, R_vals)
    return (z * match_norm / (np.linalg.norm(z) + 1e-8)).astype(np.float64)


def get_infer_z():
    """renewal_head.infer_z when the module exists (spec 07 item 2c), else the
    pinned-interface fallback above (identical math)."""
    try:
        from renewal_head import infer_z  # scripts/rescuing_fb on sys.path
        return infer_z
    except ImportError:
        return _fallback_infer_z


def load_renewal_R(path: str, goal_obs: np.ndarray) -> np.ndarray:
    """R(g_j) over the relabel goal states via renewal_head.load_renewal_head
    (pinned interface: the callable takes raw goal obs [N, obs_dim] and resolves
    psi-vs-cube input internally from the bundle metadata)."""
    from renewal_head import load_renewal_head
    R = load_renewal_head(path)
    return np.asarray(R(np.asarray(goal_obs, np.float32)), np.float64)


# ======================= GPI selection (factored for tests) =======================

def gpi_select(q_flat, n: int, C: int) -> np.ndarray:
    """Per-env GPI argmax. q_flat: [n*C] critic values at the composite z_r,
    ENV-MAJOR layout (row i*C + c = env i, candidate c — the t6_gpi_eval
    batched-call layout). Returns jstar [n]."""
    qmat = np.asarray(q_flat, np.float64).reshape(n, C)
    return qmat.argmax(1)


def gpi_pick_actions(acts_big, jstar, n: int, C: int) -> np.ndarray:
    """acts_big [n*C, act_dim] env-major -> executed actions [n, act_dim]."""
    A = np.asarray(acts_big).reshape(n, C, -1)
    return A[np.arange(n), np.asarray(jstar)]


# ======================= spec 17: delta-measure eval-side conversion ==============
# Background: PAPER/Multistep-FB/specs/17-delta-measure.md section 1 / 17a section 2.
# A `measure_delta=True` checkpoint's critic learns J(s,a,g), not M(s,a,g):
#     J(s,a,g) = (1-gamma)*M(s,a,g) - h_g(x0),   h_g(x0) = 1[obj(observations) in
#     region(g)] computed EXACTLY as GCIQLFlowAgent._hit_matrix computes it from
#     measure_hit_spec. Inverting: Mhat = (h_g(x0) + Jhat) / (1-gamma). The identity
#     is pinned in the training-code tests by
#     tests/test_gciql_measure_delta.py::test_identity_tabular (reward counted at the
#     NEXT state) -- this section mirrors that convention exactly on the eval side.
#
# The harness's ONLY readout of the critic's raw bilinear measure value is
# `q_at_z` (t6_gpi_eval.make_q_at_z: Q(s,a,z)=phi(s,a).z/sqrt(d)), consumed
# exclusively by batched_gpi_rollout (the mixture family's --gpi path), where
# z is always a COMPOSITE task vector z_r = sum_j w_j psi(g_j) (moment_z /
# retrieval / ridge_z: a weighted combination over the relabel sample, THEN
# renormalized to match_data_norm). Because q_at_z is LINEAR in z (it's a
# bilinear dot product), the raw critic readout at z_r is automatically exact:
#     Jhat(s,a,z_r) = phi(s,a).z_r/sqrt(d) = sum_j w_j Jhat(s,a,g_j)
#                   = sum_j w_j [(1-gamma)M(s,a,g_j) - h_{g_j}(x0)]
#                   = (1-gamma) Mhat(s,a,z_r) - sum_j w_j h_{g_j}(x0)
# where Mhat(s,a,z_r) := sum_j w_j M(s,a,g_j) is the natural linear extension
# of M to a composite z. So the ONLY approximation risk is in the h-term: it
# must be the SAME weighted sum sum_j w_j h_{g_j}(x0) that built z_r --
# `hit_indicator(x0, S, blocks) @ w` -- NOT a nearest-neighbor proxy (a single
# proxy state's h is a biased estimate of the weighted sum and, critically,
# the bias varies WITH the candidate z being ranked in GPI, exactly corrupting
# the comparison --delta exists to make correct). `w` is threaded through
# from the same z-construction call that built z_r (compute_z_weights below),
# never re-derived from z_r after the fact.
#
# `--delta` off is a pure passthrough (make_measure_readout returns
# q_at_z_raw UNCHANGED -- no new call, no cast -- so downstream GPI selection
# is bit-identical to today's behavior).

RIDGE_EXACT_MAX_N = 2000  # cap on the n x n push-through solve below (O(n^3));
                          # above this, ridge falls back to the proxy decode.


def hit_indicator(row_obs: np.ndarray, col_obs: np.ndarray, blocks) -> np.ndarray:
    """[R,C] region-hit indicator, bit-identical to the training-side hit matrix:
    reuses `GCIQLFlowAgent._hit_matrix` (gciql_flow.py) directly rather than
    reimplementing the block/eps geometry. `blocks` must come from
    `gciql_flow._parse_hit_spec(measure_hit_spec)`, read from the CHECKPOINT's own
    recorded config -- never a CLI default (a mismatch here silently evaluates the
    wrong region, spec 17 requirement 2). row_obs/col_obs: full observation arrays
    (blocks index into them); numpy in, numpy out -- jax used only internally."""
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent
    row = jnp.asarray(np.atleast_2d(np.asarray(row_obs, np.float32)))
    col = jnp.asarray(np.atleast_2d(np.asarray(col_obs, np.float32)))
    h = GCIQLFlowAgent._hit_matrix(row, col, blocks, jnp.float32)
    return np.asarray(h, np.float64)


def delta_to_measure(h: np.ndarray, J: np.ndarray, gamma: float) -> np.ndarray:
    """Spec 17 conversion identity: Mhat = (h_g(x0) + Jhat) / (1 - gamma). Pure
    numpy (no jax) -- unit-testable in the plain, non-jax venv. `gamma` MUST come
    from the checkpoint's own recorded config (cfg['discount']), never a hardcoded
    constant (spec 17 requirement 3)."""
    return (np.asarray(h, np.float64) + np.asarray(J, np.float64)) / (1.0 - float(gamma))


class DeltaMismatchError(RuntimeError):
    """--delta and the checkpoint's recorded `measure_delta` config disagree.
    Spec 17 section 1: "quote nothing raw" -- refuse rather than silently
    evaluate the wrong convention."""


def check_delta_consistency(ckpt_is_delta: bool, requested_delta: bool,
                            force: bool = False, run_dir: str = "<run>") -> list:
    """Refuse a --delta / checkpoint-config mismatch in EITHER direction unless
    `force` is set (a loud warning, never a silent override). Returns a list of
    warning strings to print (empty if there is nothing to warn about); raises
    DeltaMismatchError on an un-forced mismatch. Pure logic, no jax/ogbench --
    directly unit-testable."""
    warnings = []
    if ckpt_is_delta and not requested_delta:
        msg = (f"checkpoint {run_dir} was trained with measure_delta=True (recorded "
               f"config) but --delta was NOT passed -- quoting raw J as M is exactly "
               f"the failure spec 17 warns about (PAPER/Multistep-FB/specs/17-delta-measure.md section 1).")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are WRONG raw-J-as-M): " + msg)
        else:
            raise DeltaMismatchError(msg + " Pass --delta, or --force-delta-mismatch "
                                     "to override (not recommended).")
    if requested_delta and not ckpt_is_delta:
        msg = (f"--delta was passed but checkpoint {run_dir}'s recorded config has "
               f"measure_delta={ckpt_is_delta!r} (not a delta checkpoint) -- applying "
               f"the (h+J)/(1-gamma) conversion to an already-raw M readout would "
               f"corrupt every converted number.")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are WRONG converted-raw-M): "
                            + msg)
        else:
            raise DeltaMismatchError(msg + " Drop --delta, or --force-delta-mismatch "
                                     "to override (not recommended).")
    return warnings


# --- exact per-mode weight reconstruction (the fix: threading `w` through) --------
# Each z-inference mode builds z_deployed = w_pre @ Psi, then renormalizes to
# match_data_norm: z_deployed = (w_pre * scale) @ Psi = w_full @ Psi. These
# functions reproduce EXACTLY the weight vectors those modes use (mirroring
# moment_z / the retrieval branch of infer_cell / ridge_z's math), so
# `hit_indicator(x0, S, blocks) @ w_full` is the weighted h-term the delta
# identity needs -- not a re-derivation from z_r after the fact.

def moment_weights(rewards, R_vals, n: int) -> np.ndarray:
    """Pre-renormalization weights for the 'moment' mode (moment_z / infer_z /
    renewal_head.infer_z -- all pinned bit-compatible with moment_z + renorm,
    per get_infer_z's own contract): w_j = rewards_j * (R_vals_j or 1) / n."""
    w = np.asarray(rewards, np.float64)
    if R_vals is not None:
        w = w * np.asarray(R_vals, np.float64)
    return w / float(n)


def retrieval_weights(rewards, R_vals, k: int, n: int) -> np.ndarray:
    """Pre-renormalization weights for the top-k 'retrieval' mode: 1/len(idx)
    on the top-k positive-scoring indices (mirrors infer_cell's retrieval
    branch's idx selection exactly), 0 elsewhere."""
    wsc = np.asarray(rewards, np.float64)
    if R_vals is not None:
        wsc = wsc * np.asarray(R_vals, np.float64)
    idx = np.argsort(wsc)[-int(k):]
    idx = idx[wsc[idx] > 0]
    out = np.zeros(n, np.float64)
    if len(idx) > 0:
        out[idx] = 1.0 / len(idx)
    return out


def ridge_weights_exact(Psi, rewards, R_vals, ridge: float) -> np.ndarray:
    """Exact pre-renormalization weights for the 'ridge' (GLS/whitened-moment)
    mode via the push-through identity. ridge_z computes
    z_pre = (Sigma + lam I)^-1 b, Sigma = Psi^T Psi/n, b = moment_z(...),
    lam = ridge*tr(Sigma)/d -- a d x d solve that does NOT directly expose a
    per-sample weight vector. Push-through: (A^T A + mu I)^-1 A^T =
    A^T (A A^T + mu I)^-1 for A=Psi, mu=n*lam, gives z_pre = Psi^T @ alpha,
    alpha = (Psi Psi^T + mu I)^-1 y, y = rewards*(R_vals or 1) -- an n x n
    solve (caller must gate this on n via RIDGE_EXACT_MAX_N; it is NOT cheap
    at the harness's default relabel_size)."""
    Psi = np.asarray(Psi, np.float64)
    n, d = Psi.shape
    y = np.asarray(rewards, np.float64)
    if R_vals is not None:
        y = y * np.asarray(R_vals, np.float64)
    Sigma = Psi.T @ Psi / n
    lam = float(ridge) * (np.trace(Sigma) / d)
    mu = n * lam
    G = Psi @ Psi.T + mu * np.eye(n)
    return np.linalg.solve(G, y)


def renorm_scale(w_pre, Psi, match_norm: float) -> float:
    """The scalar factor infer_cell applies to go from the pre-renorm z
    (w_pre @ Psi) to the deployed z (||z|| == match_norm)."""
    z_pre = np.asarray(w_pre, np.float64) @ np.asarray(Psi, np.float64)
    return float(match_norm) / (np.linalg.norm(z_pre) + 1e-8)


def compute_z_weights(mode: str, rewards, R_vals, Psi, match_data: float,
                      k: int = 8, ridge_lambda: float = 1e-2):
    """Full weight vector w (length n = len(Psi)) such that z_deployed =
    w @ Psi EXACTLY, for whichever z-inference mode built z_deployed (spec 17
    correctness note from code review: the delta h-term MUST use the SAME
    weights that built z_r, not a nearest-neighbor proxy, or the linearity
    identity breaks and the bias varies with the candidate z being compared --
    precisely where GPI ranking lives). Returns (w, h_construction):
    h_construction is 'exact_weighted' except for 'ridge' when n exceeds
    RIDGE_EXACT_MAX_N, where w is None and h_construction is 'proxy_decode'
    (the caller must then fall back to the nearest-dataset-state decode)."""
    Psi = np.asarray(Psi, np.float64)
    n = len(Psi)
    if mode == "retrieval":
        w_pre = retrieval_weights(rewards, R_vals, k, n)
    elif mode == "ridge":
        if n > RIDGE_EXACT_MAX_N:
            return None, "proxy_decode"
        w_pre = ridge_weights_exact(Psi, rewards, R_vals, ridge_lambda)
    else:
        w_pre = moment_weights(rewards, R_vals, n)
    scale = renorm_scale(w_pre, Psi, match_data)
    return w_pre * scale, "exact_weighted"


def make_measure_readout(q_at_z_raw, blocks, gamma: float, delta: bool,
                         S=None, w=None, psi_for_decode=None):
    """Wrap a raw critic readout (t6_gpi_eval.make_q_at_z) with the spec 17
    delta->measure conversion.

    delta=False: `q_at_z_raw` is returned UNCHANGED (identity passthrough --
    no wrapping closure, no extra numpy cast) so the off-path is bit-identical
    to calling q_at_z_raw directly.

    delta=True, `w` given (the general case -- moment, retrieval, and ridge
    when compute_z_weights could solve exactly): EXACT conversion,
    Mhat = (sum_j w_j h_{g_j}(x0) + Jhat) / (1-gamma), via
    `hit_indicator(x0, S, blocks) @ w` -- `w` MUST be the same weight vector
    that built the z_r this readout will be called with (compute_z_weights),
    over the SAME relabel table `S` (dataset states, rows aligned to `w`).

    delta=True, `w=None` (ridge-only fallback, n too large for the exact
    push-through solve): APPROXIMATE conversion via the nearest-dataset-state
    proxy decode (`z_to_goal_proxy_idx`, the same decode used for
    actor_cond='goal' deployment) -- callers must record which branch ran."""
    if not delta:
        return q_at_z_raw
    assert S is not None and blocks, \
        "make_measure_readout(delta=True) needs S (relabel states) + blocks"

    if w is not None:
        w = np.asarray(w, np.float64)

        def readout(obs, acts, z_r):
            raw = q_at_z_raw(obs, acts, z_r)
            h = hit_indicator(np.asarray(obs), S, blocks) @ w
            return delta_to_measure(h, np.asarray(raw), gamma)

        return readout

    assert psi_for_decode is not None, \
        "make_measure_readout(delta=True, w=None) needs psi_for_decode for the proxy fallback"

    def readout(obs, acts, z_r):
        raw = q_at_z_raw(obs, acts, z_r)
        j = z_to_goal_proxy_idx(psi_for_decode, np.asarray(z_r, np.float64))
        h = hit_indicator(np.asarray(obs), np.asarray(S)[j:j + 1], blocks)[:, 0]
        return delta_to_measure(h, np.asarray(raw), gamma)

    return readout


# ======================= rollouts =================================================

def _policy_call(agent, mode, observations, cond, seed):
    """One batched deterministic policy call. mode='z': cond is a precomputed
    task vector batch [N,d] (sample_actions_with_z); mode='goal': cond is a
    goal-obs batch [N,obs_dim] (sample_actions — actor_cond='goal' runs)."""
    if mode == "z":
        return np.asarray(agent.sample_actions_with_z(
            observations=observations, z=cond, seed=seed, temperature=0.0))
    return np.asarray(agent.sample_actions(
        observations=observations, goals=cond, seed=seed, temperature=0.0))


def batched_rollout_record(agent, envs, cond, H=200, mode="z"):
    """Fork of t6_generality_probe.batched_rollout (proven lockstep protocol),
    additionally recording the full post-step obs trajectory (the linear family
    reads r(s') = psi(s').z_tilde off it) and supporting both actor modes.
    cond: task vector [d] (mode='z') or goal obs [obs_dim] (mode='goal').
    Returns (cube_traj [n,H,3] meters — cube-single semantics only, obs_traj
    [n,H,obs_dim], lengths [n], succ_env [n] = the env's own info['success']
    any-time flag — the native OGBench criterion, used off cube-single)."""
    import jax
    n = len(envs)
    obs_dim = envs[0].observation_space.shape[0]
    zb = np.tile(np.asarray(cond, np.float32)[None], (n, 1))
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    succ_env = np.zeros(n)
    rng = jax.random.PRNGKey(0)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        acts = _policy_call(agent, mode, obs, zb, sub)
        acts = np.clip(acts, -1, 1)
        for i, e in enumerate(envs):
            if not active[i]:
                traj[i, t] = traj[i, t - 1]
                continue
            o, _r, term, trunc, inf = e.step(acts[i])
            obs[i] = o
            traj[i, t] = e.unwrapped.data.qpos[CUBE_QPOS]
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if inf.get("success"):
                succ_env[i] = 1.0
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    return traj, obs_traj, lengths, succ_env


def batched_gpi_rollout(agent, q_at_z, envs, z_r, cand_conds, H=200, mode="z"):
    """Copied from t6_gpi_eval.batched_gpi_rollout (spec 06 item 1, proven) with
    (a) the argmax factored through gpi_select/gpi_pick_actions so the selection
    logic is unit-testable with a stubbed critic, and (b) candidate CONDITIONING
    generalized: cand_conds are task vectors [C,d] (mode='z') or goal obs
    [C,obs_dim] (mode='goal'). The critic readout is ALWAYS at z_r (the
    composite task) — only the proposal policies differ, exactly as in t6."""
    import jax
    import jax.numpy as jnp
    n = len(envs)
    C = len(cand_conds)
    Zc = np.asarray(cand_conds, np.float32)                  # [C, d|obs]
    z_r = np.asarray(z_r, np.float32)
    obs = np.zeros((n, envs[0].observation_space.shape[0]), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    lengths = np.zeros(n, int)
    prev_j = -np.ones(n, int)
    switches = np.zeros(n, float)
    steps_taken = np.zeros(n, float)
    choice_counts = np.zeros(C, float)
    rng = jax.random.PRNGKey(0)
    z_r_j = jnp.asarray(z_r)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        # one batched policy call over E*C (env-major): row (i,c) = i*C + c.
        obs_big = np.repeat(obs, C, axis=0)                  # [n*C, obs]
        z_big = np.tile(Zc, (n, 1))                          # [n*C, d|obs]
        acts_big = _policy_call(agent, mode, jnp.asarray(obs_big),
                                jnp.asarray(z_big), sub)     # [n*C, act]
        acts_big = np.clip(acts_big, -1, 1)
        qmin = np.asarray(q_at_z(jnp.asarray(obs_big), jnp.asarray(acts_big), z_r_j))
        jstar = gpi_select(qmin, n, C)                       # [n]
        acts_exec = gpi_pick_actions(acts_big, jstar, n, C)  # [n, act]
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
            lengths[i] = t + 1
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    switch_rate = switches / np.maximum(steps_taken - 1, 1)
    return traj, lengths, switch_rate, choice_counts


def frac_time_in_region(traj, lengths, center_m):
    """Per-episode fraction of REAL steps with the cube inside the region
    (reach-avoid B metric)."""
    out = []
    for i in range(len(traj)):
        seg = traj[i, :lengths[i]]
        out.append(float(reward_region(seg, center_m).mean()) if lengths[i] > 0 else 0.0)
    return np.array(out)


def discounted_reward(rewards_per_step: np.ndarray, gamma: float = GAMMA) -> float:
    """sum_t gamma^t r_t over one episode's real steps."""
    r = np.asarray(rewards_per_step, np.float64)
    return float((r * gamma ** np.arange(len(r))).sum())


# ======================= CLI ======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json; first match used)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--env", default="",
                    help="override env (default: flags.json env_name; spec 14 §7)")
    ap.add_argument("--regions", type=int, default=4,
                    help="multi-env region family: number of hindsight centers")
    ap.add_argument("--region-min-pos", type=int, default=5,
                    help="multi-env region family: min positive relabel samples "
                         "per center (degeneracy guard, spec 14 §2.1)")
    ap.add_argument("--tasks", default="1,2,3,4,5",
                    help="comma-sep task indices 1..5")
    ap.add_argument("--families", default=",".join(ALL_FAMILIES),
                    help=f"comma-sep subset of {ALL_FAMILIES}")
    ap.add_argument("--gpi", action="store_true",
                    help="mixture family: GPI over candidates {z_r, psi(g1), psi(g2)}")
    ap.add_argument("--z-inference", choices=["moment", "retrieval", "ridge"], default="moment",
                    help="ridge: GLS/whitened moment (spec 08 E3) z=(Sigma+lam I)^-1 b, "
                         "lam=ridge_lambda*tr(Sigma)/d; corrects anisotropy bias. "
                         "retrieval: z = matchnorm(mean psi over the top-k reward samples "
                         "by r*R score) — stays on the psi(g) manifold, collapsing to "
                         "exact-z for Dirac rewards (the cm200 inference-geometry fix)")
    ap.add_argument("--retrieval-k", type=int, default=8)
    ap.add_argument("--ridge-lambda", type=float, default=1e-2,
                    help="--z-inference ridge: dimensionless ridge; lam = it * tr(Sigma)/d")
    ap.add_argument("--z-weighting", choices=["plain", "renewal"], default="plain",
                    help="renewal: weight reward samples by R(g_j) (spec 07 item 2c)")
    ap.add_argument("--renewal-head", default="",
                    help="renewal-head .npz bundle (required for --z-weighting renewal)")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=200,
                    help="0 = the env's own max_episode_steps (use off cube-single)")
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--n-linear", type=int, default=3,
                    help="number of random z_tilde draws for the linear family")
    ap.add_argument("--seed", type=int, default=0,
                    help="rng for the relabel subsample + z_tilde draws")
    ap.add_argument("--out", default="",
                    help="output JSON (default analysis/rescuing-fb/zeroshot_rewards_<run>.json)")
    ap.add_argument("--delta", action="store_true",
                    help="spec 17a: the checkpoint was trained with measure_delta=True "
                         "(the critic's fixed point is J, not M). Converts every "
                         "critic-measure readout via Mhat=(h_g(x0)+Jhat)/(1-gamma) "
                         "before it is used. Auto-checked against the checkpoint's own "
                         "recorded config (flags.json agent.measure_delta); a mismatch "
                         "in EITHER direction errors out unless --force-delta-mismatch "
                         "is also passed (PAPER/Multistep-FB/specs/17-delta-measure.md section 1: "
                         "'quote nothing raw').")
    ap.add_argument("--force-delta-mismatch", action="store_true",
                    help="override a --delta / checkpoint measure_delta mismatch. Prints "
                         "a loud warning and proceeds with whatever convention --delta "
                         "requested (which will be WRONG relative to the checkpoint). "
                         "Not recommended; exists as an escape hatch only.")
    return ap


def print_table(rows):
    """rows: list of (family, cell, metric, value_str) — aligned text table."""
    if not rows:
        return
    hdr = ("family", "cell", "metric", "value")
    widths = [max(len(hdr[c]), max(len(str(r[c])) for r in rows)) for c in range(4)]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)), flush=True)


# ======================= main =====================================================

def main() -> int:
    args = build_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in ALL_FAMILIES]
    assert not unknown, f"unknown families {unknown}; choose from {ALL_FAMILIES}"
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    assert all(1 <= t <= 5 for t in task_ids), f"task ids must be in 1..5, got {task_ids}"
    if args.z_weighting == "renewal":
        assert args.renewal_head, "--z-weighting renewal requires --renewal-head PATH"

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])
    # spec 17 requirement 4: auto-detect + refuse a --delta / checkpoint mismatch
    # in EITHER direction before doing any other work (fail fast).
    ckpt_is_delta = bool(cfg.get("measure_delta", False))
    for _w in check_delta_consistency(ckpt_is_delta, args.delta,
                                      force=args.force_delta_mismatch,
                                      run_dir=str(run_dir)):
        print(f"[zsr] {_w}", flush=True)
    env_name = args.env or flags["env_name"]
    multi = env_name != CUBE_SINGLE
    eval_tasks = eval_tasks_for(env_name)
    # spec 14 §3.4: the checkpoint's own hit spec wins over the registry default.
    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    if multi:
        bad = [f for f in families if f not in ("dirac", "region", "dense")]
        assert not bad, (f"families {bad} are cube-single-only; off cube-single "
                         f"this harness supports dirac/region/dense (spec 14 §7)")

    t00 = time.time()
    # ---- shared relabel sample (task1 dataset, rng(seed); t6 protocol) ----
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    n_ds = len(ds1["observations"])
    _act_dim = int(np.asarray(ds1["actions"]).shape[-1])
    _obs_dim = int(np.asarray(ds1["observations"]).shape[-1])
    H = args.horizon if args.horizon > 0 else int(_e0.spec.max_episode_steps)

    # checkpoint restore: t3_offpeak_multienv.load_agent_dims (domain-correct
    # example dims; flags.json arch read via scripts.atlas.jax_load).
    from t3_offpeak_multienv import load_agent_dims
    # t3's module-level sys.path inserts put REPO (which holds the torch stack's
    # utils.py) ahead of ogbench impls, shadowing the impls `utils` PACKAGE that
    # restore_agent lives in; put impls back in front before the lazy import.
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, _obs_dim, _act_dim)
    d = int(cfg["value_latent_dim"])
    fbz = bool(cfg.get("fb_forward_z", False))
    use_critic_psi = bool(cfg.get("measure_only", False)) and (
        fbz or cfg.get("actor_z_from_critic", False) or cfg.get("actor_cond") == "z")
    infer_z = get_infer_z()
    # actor deployment interface (see module docstring): 'z' actors consume z_r
    # directly; 'goal' actors (the frozen FP-NoOrtho cells) need the g_hat decode.
    actor_mode = "z" if cfg.get("actor_cond") == "z" else "goal"
    print(f"[zsr] restored {run_dir.name} step={args.checkpoint_step} d={d} "
          f"env={env_name} blocks={spec.blocks} H={H} "
          f"actor_cond={cfg.get('actor_cond')} (mode={actor_mode}) "
          f"z_weighting={args.z_weighting} gpi={args.gpi}", flush=True)

    psi = make_psi(agent, src="critic" if use_critic_psi else "value",
                   act_dim=_act_dim, fbz=fbz)
    q_at_z_raw = make_q_at_z(agent, d, fbz=fbz)
    if use_critic_psi:
        print(f"[zsr] psi source = CRITIC (measure_only z-actor; fbz={fbz})", flush=True)
    idx = np.random.default_rng(args.seed).choice(
        n_ds, size=min(args.relabel_size, n_ds), replace=False)
    S = np.asarray(ds1["observations"], np.float32)[idx]
    cube_m = decode_cube(S)                      # meters (cube-single semantics)
    cube_ou = cube_obs_units(S)                  # obs units (cube-single semantics)
    coords_ds = spec.coords(S)                   # FULL region coordinate (spec 14)
    Psi = psi(S).astype(np.float64)              # [m, d]
    match_data = float(np.linalg.norm(Psi, axis=-1).mean())

    # spec 17 requirement 1/2/3: the harness's ONE critic-measure readout site
    # is q_at_z_raw, consumed only by batched_gpi_rollout (the mixture --gpi
    # path). gamma + the hit-spec both come from the CHECKPOINT's own recorded
    # config, never a CLI default or the module's GAMMA constant (that
    # constant is for the unrelated `linear` family's discounted-reward
    # readout). The actual wrapping (make_measure_readout) happens PER MIXTURE
    # CELL below, once the exact per-cell weight vector `w` is known (code
    # review correctness fix: the delta h-term must use the SAME weights that
    # built z_r -- see the "spec 17" section docstring above) -- --delta off
    # never builds a wrapper at all (q_at_z_raw used directly, requirement 6).
    gamma = float(cfg.get("discount", GAMMA))
    delta_blocks = None
    if args.delta:
        from gciql_flow import _parse_hit_spec
        delta_blocks = _parse_hit_spec(str(cfg.get("measure_hit_spec") or "19:22:0.4"))
        if args.z_inference != "ridge":
            _ridge_n_note = " (exact_weighted h -- moment/retrieval are always exact)"
        elif len(Psi) <= RIDGE_EXACT_MAX_N:
            _ridge_n_note = (f" (n={len(Psi)} <= RIDGE_EXACT_MAX_N={RIDGE_EXACT_MAX_N}: "
                             f"exact_weighted h)")
        else:
            _ridge_n_note = (f" (n={len(Psi)} > RIDGE_EXACT_MAX_N={RIDGE_EXACT_MAX_N}: "
                             f"PROXY FALLBACK, approximate h)")
        print(f"[zsr] --delta: gamma={gamma} hit_spec={cfg.get('measure_hit_spec')} "
              f"z_inference={args.z_inference}{_ridge_n_note}", flush=True)

    R_vals = None
    if args.z_weighting == "renewal":
        R_vals = load_renewal_R(args.renewal_head, S)
        print(f"[zsr] renewal R over {len(S)} goals: mean={R_vals.mean():.4f} "
              f"var/mean^2={R_vals.var() / max(R_vals.mean() ** 2, 1e-12):.4f}", flush=True)

    # ---- per-task goal obs / coords + psi(g) ----
    # Off cube-single the goal coordinate is HitSpec.coords of the goal OBS —
    # goal_xyzs[:3] truncates cube-double to cube 0 and scene has no goal_xyzs
    # key at all (spec 14 §2); the archived cube-single line is kept where it
    # is exact.
    goal_obs, goal_xyz, goal_coord = {}, {}, {}
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        goal_coord[i] = spec.coords(goal_obs[i][None])[0]
        if not multi:
            goal_xyz[i] = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        env.close()
    psi_g = {i: psi(goal_obs[i]).astype(np.float64) for i in task_ids}
    goal_ou = ({i: (goal_xyz[i] - CEN) * 10.0 for i in task_ids}
               if not multi else {})             # obs units (cube-single only)

    # ---- lazy env pools (t6 pattern) ----
    pools = {}

    def pool(i):
        if i not in pools:
            pools[i] = [ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
                        for _ in range(args.episodes)]
        return pools[i]

    def close_all():
        for i in list(pools):
            for e in pools.pop(i):
                if hasattr(e, "close"):
                    e.close()

    def infer_cell(rewards):
        """Reward samples -> (z_r float32, z_raw, diagnostics dict). z is None
        if the weighted moment is degenerate (no informative samples)."""
        z_raw = moment_z(Psi, rewards, R_vals)
        raw_norm = float(np.linalg.norm(z_raw))
        diag = {"n_pos": int((np.asarray(rewards) > 0).sum()),
                "n_neg": int((np.asarray(rewards) < 0).sum()),
                "z_raw_norm": raw_norm, "match_data_norm": match_data}
        if raw_norm < 1e-12:
            return None, z_raw, diag
        if args.z_inference == "retrieval":
            # Top-k retrieval: the moment misaims under anisotropic no-ortho psi
            # (cos~0.5 to psi(g*)); averaging only the k best-scoring samples'
            # psi's keeps z on the psi(g) manifold the actor was trained on.
            w = np.asarray(rewards, np.float64)
            if R_vals is not None:
                w = w * np.asarray(R_vals, np.float64)
            idx = np.argsort(w)[-args.retrieval_k:]
            idx = idx[w[idx] > 0]
            if len(idx) == 0:
                return None, z_raw, diag
            zr = Psi[idx].mean(0)
            z = zr * match_data / (np.linalg.norm(zr) + 1e-8)
            diag["retrieval_n"] = int(len(idx))
        elif args.z_inference == "ridge":
            # GLS / whitened moment (spec 08 E3): corrects the moment's anisotropy bias
            # under no-ortho psi -- the dense-reward misaim forensic.
            z = ridge_z(Psi, np.asarray(rewards, np.float64), R_vals,
                        match_data, ridge=args.ridge_lambda)
            diag["ridge_lambda"] = float(args.ridge_lambda)
        else:
            z = np.asarray(infer_z(Psi, np.asarray(rewards, np.float64), R_vals,
                                   match_data), np.float64)
        diag["z_norm"] = float(np.linalg.norm(z))
        return z.astype(np.float32), z_raw, diag

    def deploy(z):
        """z_r -> actor conditioning. 'z' mode: z itself. 'goal' mode: the
        argmax goal-proxy dataset obs (z_to_goal_proxy_idx) + decode info.
        Off cube-single the decode report carries the FULL geometry coordinate
        (spec 14 §7), not cube xyz."""
        if actor_mode == "z":
            return z, None
        j = z_to_goal_proxy_idx(Psi, z)
        info = {"proxy_idx": j}
        if multi:
            info["proxy_coord"] = coords_ds[j].tolist()
        else:
            info["proxy_cube_m"] = cube_m[j].tolist()
        return S[j], info

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step, "d": d,
        "env": env_name,
        "hit_spec": ",".join(f"{a_}:{b_}:{e_}" for a_, b_, e_ in spec.blocks),
        "probe_centers": geo.probe_centers,
        "actor_cond": cfg.get("actor_cond"), "actor_mode": actor_mode,
        "tasks": task_ids, "families": families, "gpi": bool(args.gpi),
        "z_inference": args.z_inference, "retrieval_k": args.retrieval_k,
        "z_weighting": args.z_weighting, "renewal_head": args.renewal_head or None,
        "episodes": args.episodes, "horizon": H, "relabel_size": int(len(S)),
        "match_data_norm": match_data, "gamma": GAMMA,
        "eps_obs": EPS_OBS, "sigma_obs": SIGMA_OBS, "thresh_m": THRESH,
        "reward_convention": "[0,1] indicator (FP corner), reachavoid in [-1,1]",
        "goal_xyzs_m": ({f"t{i}": goal_xyz[i].tolist() for i in task_ids}
                        if not multi else None),
        "goal_coords_ou": {f"t{i}": goal_coord[i].tolist() for i in task_ids},
        "regions_m": ({k: v.tolist() for k, v in REGIONS.items()}
                      if not multi else None),
        "reachavoid": {"A": REACH_A, "B": REACH_B},
        "seed": args.seed,
        "spec": "PAPER/rescuing-fb/specs/07-fp-capability-and-renewal-measure.md item 3"
                " + specs/14-multienv-measure-eval.md §7",
    }, "families": {}}
    # spec 17 requirement 5: "quote conversion/identity conventions next to every
    # number" -- record the delta convention whenever delta is in play in ANY way
    # (flag passed, checkpoint recorded measure_delta=True, or --force used), so a
    # dangerous force-override state is self-documented too. Omitted entirely when
    # delta never enters the picture, so the fully-off path's JSON is unchanged
    # from today (requirement 6: bit-identical structure AND values off-path).
    if args.delta or ckpt_is_delta or args.force_delta_mismatch:
        # h_construction here is the a-priori (per-run, mode/n-determined) answer;
        # each mixture cell also records its own 'delta_h_construction' (identical
        # across cells for a given run, since it only depends on z_inference mode
        # and n=len(Psi), not on the per-cell rewards) for full auditability.
        _h_construction = (
            "proxy_decode" if (args.delta and args.z_inference == "ridge"
                              and len(Psi) > RIDGE_EXACT_MAX_N)
            else "exact_weighted" if args.delta else None)
        results["meta"]["measure_delta"] = {
            "applied": bool(args.delta),
            "ckpt_measure_delta": ckpt_is_delta,
            "forced_mismatch": bool(args.force_delta_mismatch),
            "gamma": gamma if args.delta else None,
            "hit_spec": (",".join(f"{a_}:{b_}:{e_}" for a_, b_, e_ in delta_blocks)
                        if args.delta else None),
            "h_construction": _h_construction,
            "ridge_exact_max_n": RIDGE_EXACT_MAX_N if args.delta else None,
            "convention": (
                ("Mhat = (sum_j w_j h_{g_j}(x0) + Jhat) / (1 - gamma), w = the SAME "
                 "weight vector that built z_r (compute_z_weights, exact for moment/"
                 "retrieval/ridge-when-n-is-small); h_{g_j} via GCIQLFlowAgent._hit_matrix "
                 "over the relabel states S"
                 if _h_construction == "exact_weighted" else
                 "Mhat = (h_g(x0) + Jhat) / (1 - gamma), h_g via the z_r NEAREST-"
                 f"DATASET-STATE PROXY decode (ridge fallback: n={len(Psi)} exceeds "
                 f"RIDGE_EXACT_MAX_N={RIDGE_EXACT_MAX_N}) -- APPROXIMATE, not exact"
                 ) + " (PAPER/Multistep-FB/specs/17-delta-measure.md section 1); applies "
                "only to the GPI critic readout (q_at_z, mixture --gpi family) -- "
                "the harness's sole raw critic-measure readout site"
                if args.delta else
                "no conversion applied (measure_delta not requested for this run)"),
        }
    table = []

    # ================= 1. dirac =================
    if "dirac" in families:
        fam = {}
        for i in task_ids:
            # cube-single: the archived single-ball path, bit-for-bit (gate).
            # multi-env: the trained conjunctive indicator over the full region
            # coordinate of the task GOAL OBSERVATION (spec 14 §7 fixes).
            r = (reward_conj_dirac(coords_ds, goal_coord[i], spec) if multi
                 else reward_dirac(cube_ou, goal_ou[i]))
            z, z_raw, diag = infer_cell(r)
            rec = {"diag": diag}
            if z is None:
                rec["success"] = None
            else:
                rec["cos_zr_psig"] = cosine(z_raw, psi_g[i])
                cond, proxy = deploy(z)
                if proxy is not None:
                    if multi:
                        proxy["dist_to_goal_scaled"] = float(np.linalg.norm(
                            spec.scaled(np.asarray(proxy["proxy_coord"])[None])
                            - spec.scaled(goal_coord[i][None])))
                    else:
                        proxy["dist_to_goal_m"] = float(np.linalg.norm(
                            np.asarray(proxy["proxy_cube_m"]) - goal_xyz[i]))
                    rec["goal_proxy"] = proxy
                traj, _ot, ln, senv = batched_rollout_record(agent, pool(i), cond, H,
                                                             mode=actor_mode)
                rec["success_env"] = float(senv.mean())   # native OGBench criterion
                rec["success"] = (rec["success_env"] if multi
                                  else float(m_goal(traj, ln, goal_xyz[i]).mean()))
            fam[f"t{i}"] = rec
            table.append(("dirac", f"t{i}", "success", f"{rec['success']}"))
            print(f"[zsr][dirac] t{i}: succ={rec['success']} "
                  f"env={rec.get('success_env')} "
                  f"cos={rec.get('cos_zr_psig', float('nan')):.3f} "
                  f"||z_raw||={diag['z_raw_norm']:.3g} ({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success"] for v in fam.values() if v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["dirac"] = fam
        table.append(("dirac", "MEAN", "success", f"{fam['mean']}"))

    # ================= 2. mixture (± GPI) =================
    if "mixture" in families:
        fam = {}
        for i, j in combinations(task_ids, 2):
            r = reward_mixture(cube_ou, [goal_ou[i], goal_ou[j]])
            z, z_raw, diag = infer_cell(r)
            rec = {"diag": diag, "gpi": bool(args.gpi)}
            if z is None:
                rec["success"] = None
            else:
                rec["cos_zr_gi"] = cosine(z_raw, psi_g[i])
                rec["cos_zr_gj"] = cosine(z_raw, psi_g[j])
                cond, proxy = deploy(z)
                if proxy is not None:
                    rec["goal_proxy"] = proxy
                # spec 17 (code-review correctness fix): z_r = sum_j w_j psi(g_j)
                # over THIS cell's own reward samples `r`, so the delta h-term
                # must use the SAME w -- compute_z_weights mirrors infer_cell's
                # own math exactly (moment/retrieval/ridge). Built per cell
                # (not once globally) because `r`, hence `w`, differs per cell;
                # only needed when GPI will actually call the critic readout.
                q_at_z_cell = q_at_z_raw
                if args.delta and args.gpi:
                    w_full, h_construction = compute_z_weights(
                        args.z_inference, r, R_vals, Psi, match_data,
                        k=args.retrieval_k, ridge_lambda=args.ridge_lambda)
                    rec["delta_h_construction"] = h_construction
                    q_at_z_cell = make_measure_readout(
                        q_at_z_raw, delta_blocks, gamma, True,
                        S=S, w=w_full, psi_for_decode=Psi)
                succs, per_var = [], {}
                for v in (i, j):
                    if args.gpi:
                        # candidates: the inferred-task policy + the two exact
                        # single-goal policies (t6_gpi_eval item-1 protocol),
                        # in the actor's native conditioning space.
                        if actor_mode == "z":
                            cand = [z, psi_g[i].astype(np.float32),
                                    psi_g[j].astype(np.float32)]
                        else:
                            cand = [cond, goal_obs[i], goal_obs[j]]
                        traj, ln, sw, cc = batched_gpi_rollout(
                            agent, q_at_z_cell, pool(v), z, cand, H, mode=actor_mode)
                        s, f = m_pair(traj, ln, goal_xyz[i], goal_xyz[j])
                        per_var[f"t{v}"] = {
                            "success": float(s.mean()),
                            "switch_rate": float(sw.mean()),
                            "choice_frac": (cc / max(cc.sum(), 1)).tolist()}
                    else:
                        traj, _ot, ln, _se = batched_rollout_record(agent, pool(v), cond, H,
                                                                    mode=actor_mode)
                        s, f = m_pair(traj, ln, goal_xyz[i], goal_xyz[j])
                        per_var[f"t{v}"] = {"success": float(s.mean())}
                    per_var[f"t{v}"]["hit_g1"] = int((f == 1).sum())
                    per_var[f"t{v}"]["hit_g2"] = int((f == 2).sum())
                    succs.append(s)
                rec["per_variant"] = per_var
                rec["success"] = float(np.concatenate(succs).mean())
            fam[f"t{i}-t{j}"] = rec
            table.append(("mixture", f"t{i}-t{j}",
                          "success" + ("+gpi" if args.gpi else ""), f"{rec['success']}"))
            print(f"[zsr][mixture{'+gpi' if args.gpi else ''}] t{i}-t{j}: "
                  f"succ={rec['success']} ({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success"] for v in fam.values() if v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["mixture"] = fam
        table.append(("mixture", "MEAN",
                      "success" + ("+gpi" if args.gpi else ""), f"{fam['mean']}"))

    # ================= 3. region =================
    # cube-single: the archived T6 protocol (REGIONS, box-or-ball, all task
    # variants) — unchanged for the regression gate. Multi-env: conjunctive
    # regions (HitSpec.contains at the trained radii) at VISITED-STATE hindsight
    # centers (env_geometry policy, spec 14 §2.1/§3.3), rolled out on the first
    # selected task variant (region rewards ignore the env goal; variants share
    # the initial-state distribution).
    if "region" in families and multi:
        bounds = episode_bounds(np.asarray(ds1["terminals"]))
        coords_all = spec.coords(np.asarray(ds1["observations"], np.float32))
        rng_reg = np.random.default_rng(args.seed + 2)
        centers, npos_c, rejected = draw_region_centers(
            coords_all, bounds, spec, coords_ds, rng_reg, args.regions,
            min_pos=args.region_min_pos)
        v0 = task_ids[0]
        fam = {"probe_policy": geo.probe_centers, "region_variant": v0,
               "n_centers": int(len(centers)), "centers_rejected": int(rejected),
               "center_min_pos": int(args.region_min_pos)}
        print(f"[zsr][region] {len(centers)}/{args.regions} hindsight centers "
              f"({rejected} rejected at min_pos={args.region_min_pos})", flush=True)
        for k_i, c in enumerate(centers):
            r = reward_conj_dirac(coords_ds, c, spec)
            z, _z_raw, diag = infer_cell(r)
            rec = {"center_coord": np.asarray(c).tolist(),
                   "center_n_pos": int(npos_c[k_i]), "diag": diag}
            if z is None:
                rec["success_final"] = rec["success_any"] = None
            else:
                cond, proxy = deploy(z)
                if proxy is not None:
                    proxy["dist_to_center_scaled"] = float(np.linalg.norm(
                        spec.scaled(np.asarray(proxy["proxy_coord"])[None])
                        - spec.scaled(np.asarray(c)[None])))
                    rec["goal_proxy"] = proxy
                _tj, ot, ln, _se = batched_rollout_record(agent, pool(v0), cond, H,
                                                          mode=actor_mode)
                fin, anys = region_success_from_obs(ot, ln, c, spec)
                rec["success_final"] = float(fin.mean())
                rec["success_any"] = float(anys.mean())
            fam[f"c{k_i}"] = rec
            table.append(("region", f"c{k_i}", "success_final", f"{rec['success_final']}"))
            print(f"[zsr][region] c{k_i}: fin={rec['success_final']} "
                  f"any={rec['success_any']} n_pos={rec['center_n_pos']} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        cells = [v for k, v in fam.items() if k.startswith("c") and isinstance(v, dict)]
        vals = [v["success_final"] for v in cells if v.get("success_final") is not None]
        anyv = [v["success_any"] for v in cells if v.get("success_any") is not None]
        fam["mean_final"] = float(np.mean(vals)) if vals else None
        fam["mean_any"] = float(np.mean(anyv)) if anyv else None
        fam["degenerate_frac"] = (float(np.mean(
            [v.get("success_final") is None for v in cells])) if cells else None)
        results["families"]["region"] = fam
        table.append(("region", "MEAN", "success_final", f"{fam['mean_final']}"))
    elif "region" in families:
        fam = {}
        for name, c in REGIONS.items():
            r = reward_region(cube_m, c)
            z, _z_raw, diag = infer_cell(r)
            rec = {"center_m": np.asarray(c).tolist(), "diag": diag}
            if z is None:
                rec["success_final"] = rec["success_any"] = None
            else:
                cond, proxy = deploy(z)
                if proxy is not None:
                    proxy["dist_to_center_m"] = float(np.linalg.norm(
                        np.asarray(proxy["proxy_cube_m"]) - np.asarray(c)))
                    rec["goal_proxy"] = proxy
                per_f, per_a = {}, {}
                for v in task_ids:
                    traj, _ot, ln, _se = batched_rollout_record(agent, pool(v), cond, H,
                                                                mode=actor_mode)
                    fin, anys = m_region(traj, ln, c)
                    per_f[f"t{v}"] = float(fin.mean())
                    per_a[f"t{v}"] = float(anys.mean())
                rec["per_variant_final"] = per_f
                rec["per_variant_any"] = per_a
                rec["success_final"] = float(np.mean(list(per_f.values())))
                rec["success_any"] = float(np.mean(list(per_a.values())))
            fam[name] = rec
            table.append(("region", name, "success_final", f"{rec['success_final']}"))
            print(f"[zsr][region] {name}: fin={rec['success_final']} "
                  f"any={rec['success_any']} ({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success_final"] for v in fam.values() if v["success_final"] is not None]
        fam["mean_final"] = float(np.mean(vals)) if vals else None
        results["families"]["region"] = fam
        table.append(("region", "MEAN", "success_final", f"{fam['mean_final']}"))

    # ================= 4. dense =================
    if "dense" in families:
        fam = {}
        for i in task_ids:
            # multi-env: exp(-d/2) in unit-radius coordinates == the archived
            # exp(-d/0.8) on cube-single's one 0.4-block (reward_conj_dense).
            r = (reward_conj_dense(coords_ds, goal_coord[i], spec) if multi
                 else reward_dense(cube_ou, goal_ou[i]))
            z, z_raw, diag = infer_cell(r)
            rec = {"diag": diag, "sigma_obs": SIGMA_OBS,
                   "reward_stats": {"mean": float(r.mean()), "max": float(r.max())}}
            if z is None:
                rec["success"] = None
            else:
                rec["cos_zr_psig"] = cosine(z_raw, psi_g[i])
                cond, proxy = deploy(z)
                if proxy is not None:
                    if multi:
                        proxy["dist_to_goal_scaled"] = float(np.linalg.norm(
                            spec.scaled(np.asarray(proxy["proxy_coord"])[None])
                            - spec.scaled(goal_coord[i][None])))
                    else:
                        proxy["dist_to_goal_m"] = float(np.linalg.norm(
                            np.asarray(proxy["proxy_cube_m"]) - goal_xyz[i]))
                    rec["goal_proxy"] = proxy
                traj, _ot, ln, senv = batched_rollout_record(agent, pool(i), cond, H,
                                                             mode=actor_mode)
                rec["success_env"] = float(senv.mean())
                rec["success"] = (rec["success_env"] if multi
                                  else float(m_goal(traj, ln, goal_xyz[i]).mean()))
            fam[f"t{i}"] = rec
            table.append(("dense", f"t{i}", "success", f"{rec['success']}"))
            print(f"[zsr][dense] t{i}: succ={rec['success']} "
                  f"cos={rec.get('cos_zr_psig', float('nan')):.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success"] for k, v in fam.items()
                if k.startswith("t") and v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        # headline read: dense vs the SAME-c* dirac cell (spec item 3 family 4)
        dir_fam = results["families"].get("dirac", {})
        deltas = [fam[f"t{i}"]["success"] - dir_fam[f"t{i}"]["success"]
                  for i in task_ids
                  if fam.get(f"t{i}", {}).get("success") is not None
                  and dir_fam.get(f"t{i}", {}).get("success") is not None]
        fam["mean_delta_vs_dirac"] = float(np.mean(deltas)) if deltas else None
        results["families"]["dense"] = fam
        table.append(("dense", "MEAN", "success", f"{fam['mean']}"))
        table.append(("dense", "MEAN", "delta_vs_dirac", f"{fam['mean_delta_vs_dirac']}"))

    # ================= 5. linear (no success read exists) =================
    if "linear" in families:
        fam = {"note": "achieved discounted reward under rollout vs a random-z "
                       "control policy (proxy floor; the BC-flow floor needs a "
                       "separate BC rollout)"}
        rng_lin = np.random.default_rng(args.seed + 1)
        v0 = task_ids[0]           # rollouts on the first selected task variant
        cells = []
        for k in range(args.n_linear):
            z_tilde = sample_sphere_z(rng_lin, d, match_data)
            r = reward_linear(Psi, z_tilde)
            z, z_raw, diag = infer_cell(r)
            rec = {"diag": diag, "cos_zr_ztilde":
                   cosine(z_raw, z_tilde) if diag["z_raw_norm"] > 0 else None,
                   "dataset_reward_stats": {"mean": float(r.mean()),
                                            "std": float(r.std())}}
            if z is not None:
                for label, z_use in (("z_r", z),
                                     ("control_random_z",
                                      sample_sphere_z(rng_lin, d, match_data)
                                      .astype(np.float32))):
                    cond, proxy = deploy(z_use)
                    if proxy is not None:
                        rec[f"goal_proxy_{label}"] = proxy
                    traj, obs_traj, ln, _se = batched_rollout_record(
                        agent, pool(v0), cond, H, mode=actor_mode)
                    Gs = []
                    for e_i in range(len(ln)):
                        if ln[e_i] == 0:
                            Gs.append(0.0)
                            continue
                        psi_traj = psi(obs_traj[e_i, :ln[e_i]]).astype(np.float64)
                        r_steps = reward_linear(psi_traj, z_tilde)
                        Gs.append(discounted_reward(r_steps))
                    rec[f"discounted_reward_{label}"] = float(np.mean(Gs))
            cells.append(rec)
            print(f"[zsr][linear] draw{k}: "
                  f"G(z_r)={rec.get('discounted_reward_z_r')} "
                  f"G(ctrl)={rec.get('discounted_reward_control_random_z')} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        fam["draws"] = cells
        g_zr = [c.get("discounted_reward_z_r") for c in cells
                if c.get("discounted_reward_z_r") is not None]
        g_ct = [c.get("discounted_reward_control_random_z") for c in cells
                if c.get("discounted_reward_control_random_z") is not None]
        fam["mean_discounted_reward_z_r"] = float(np.mean(g_zr)) if g_zr else None
        fam["mean_discounted_reward_control"] = float(np.mean(g_ct)) if g_ct else None
        results["families"]["linear"] = fam
        table.append(("linear", "MEAN", "G(z_r)", f"{fam['mean_discounted_reward_z_r']}"))
        table.append(("linear", "MEAN", "G(control)",
                      f"{fam['mean_discounted_reward_control']}"))

    # ================= 6. reachavoid (exploratory) =================
    if "reachavoid" in families:
        cA, cB = REGIONS[REACH_A], REGIONS[REACH_B]
        r = reward_reachavoid(cube_m, cA, cB)
        z, _z_raw, diag = infer_cell(r)
        fam = {"A": REACH_A, "B": REACH_B, "diag": diag}
        if z is None:
            fam["success_at_A"] = fam["frac_time_in_B"] = None
        else:
            cond, proxy = deploy(z)
            if proxy is not None:
                proxy["dist_to_A_m"] = float(np.linalg.norm(
                    np.asarray(proxy["proxy_cube_m"]) - np.asarray(cA)))
                fam["goal_proxy"] = proxy
            per_a, per_frac = {}, {}
            for v in task_ids:
                traj, _ot, ln, _se = batched_rollout_record(agent, pool(v), cond, H,
                                                            mode=actor_mode)
                finA, anyA = m_region(traj, ln, cA)
                per_a[f"t{v}"] = {"final": float(finA.mean()), "any": float(anyA.mean())}
                per_frac[f"t{v}"] = float(frac_time_in_region(traj, ln, cB).mean())
            fam["per_variant_A"] = per_a
            fam["per_variant_frac_B"] = per_frac
            fam["success_at_A"] = float(np.mean([x["final"] for x in per_a.values()]))
            fam["success_at_A_any"] = float(np.mean([x["any"] for x in per_a.values()]))
            fam["frac_time_in_B"] = float(np.mean(list(per_frac.values())))
        results["families"]["reachavoid"] = fam
        table.append(("reachavoid", f"{REACH_A}-{REACH_B}", "success_at_A",
                      f"{fam['success_at_A']}"))
        table.append(("reachavoid", f"{REACH_A}-{REACH_B}", "frac_time_in_B",
                      f"{fam['frac_time_in_B']}"))
        print(f"[zsr][reachavoid] A={fam['success_at_A']} inB={fam['frac_time_in_B']} "
              f"({time.time()-t00:.0f}s)", flush=True)

    # ================= summary + spec 07 success bars =================
    famr = results["families"]
    summary = {
        "dirac_mean": famr.get("dirac", {}).get("mean"),
        "mixture_mean": famr.get("mixture", {}).get("mean"),
        "mixture_gpi": bool(args.gpi),
        "region_mean_final": famr.get("region", {}).get("mean_final"),
        "dense_mean": famr.get("dense", {}).get("mean"),
        "dense_mean_delta_vs_dirac": famr.get("dense", {}).get("mean_delta_vs_dirac"),
        "linear_G_zr": famr.get("linear", {}).get("mean_discounted_reward_z_r"),
        "linear_G_control": famr.get("linear", {}).get("mean_discounted_reward_control"),
        "reachavoid_success_at_A": famr.get("reachavoid", {}).get("success_at_A"),
        "reachavoid_frac_time_in_B": famr.get("reachavoid", {}).get("frac_time_in_B"),
    }
    bars = {}
    if summary["dirac_mean"] is not None:
        bars["dirac_ge_0.90"] = bool(summary["dirac_mean"] >= 0.90)
    if summary["mixture_mean"] is not None and args.gpi:
        bars["mixture_gpi_ge_0.75"] = bool(summary["mixture_mean"] >= 0.75)
    if summary["region_mean_final"] is not None:
        bars["regions_ge_0.672"] = bool(summary["region_mean_final"] >= 0.672)
    if summary["dense_mean_delta_vs_dirac"] is not None:
        bars["dense_within_5pts_of_dirac"] = bool(
            summary["dense_mean_delta_vs_dirac"] >= -0.05)
    summary["bars"] = bars
    results["summary"] = summary

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec14_jax" / f"zsr_{run_dir.name}.json" if multi
         else REPO / "analysis" / "rescuing-fb" / f"zeroshot_rewards_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\n[zsr] DONE ({time.time()-t00:.0f}s) -> {outp}\n", flush=True)
    print_table(table)
    print(f"\n[zsr] bars: {bars}", flush=True)
    close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
