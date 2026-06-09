"""evals/task_distribution.py — FB vs PSM training task-distribution analysis.

Characterizes the goal-relabelled task distribution each method trains its
actor / successor net on: a pool of next-states s' drawn from the offline cube
buffer, embedded into the per-method training-z cloud
  FB : z = project_z(backward_map(s'))      (z_dim=50)
  PSM: z = project_z(phi(s'))               (z_dim=128)

Pure logic (label_interaction, estimate_table_z, participation_ratio,
separability_auc, coverage_score) is unit-tested on synthetic data; the
checkpoint-embedding path (embed_pool) and plotting/IO live in the driver
(scripts/task_distribution_probe.py). See PAPER/specs/2026-06-07-fb-vs-psm-
task-distribution-analysis.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# State layout (cube-single, 28-d obs; verified from data + grasp_feature_probe.py).
# The obs is NORMALISED, so effector/cube proximity must be computed WITHIN the obs
# frame (mixing obs-effector with physics-cube gives meaningless distances).
#   obs 0:12 arm joints (pos+vel)   12:15 effector xyz   18 contact flag (0/1)
#   obs 19:22 cube xyz (normalised)
# Lift uses the physics cube z (real metres) for an interpretable threshold:
#   physics[14:17] cube xyz (world metres, cf. FB _coverage_weights).
EFF_SLICE = slice(12, 15)      # effector xyz, obs frame (normalised)
CUBE_OBS = slice(19, 22)       # cube xyz, obs frame (normalised) — same frame as EFF
CONTACT_IDX = 18               # gripper contact flag (0/1)
CUBE_PHYSICS = slice(14, 17)   # cube xyz in physics vector (world metres) — for lift

INTERACTION_BUCKETS = ("no_contact", "reaching", "grasped", "lifted_or_transport")


def estimate_table_z(cube_z: np.ndarray, q: float = 1.0) -> float:
    """Robust table-surface height = low percentile of cube z (cube at rest).

    q is the percentile (default 1.0 => 1st percentile), so transient lifts do
    not bias the estimate downward-resistant baseline.
    """
    return float(np.percentile(np.asarray(cube_z, dtype=float), q))


def label_interaction(
    eff_xyz: np.ndarray,
    cube_xyz: np.ndarray,
    lift: np.ndarray,
    contact: np.ndarray,
    *,
    grasp_radius: float = 0.12,
    reach_radius: float = 0.5,
    delta_lift: float = 0.03,
    tau_contact: float = 0.5,
) -> np.ndarray:
    """Per-state interaction bucket (index into INTERACTION_BUCKETS).

    Frame-agnostic: `eff_xyz`/`cube_xyz` must be in the SAME frame (their norm
    is the hand-cube distance); `lift` (cube height above table) and `contact`
    (gripper contact flag) are passed precomputed by the caller.

    Monotone ladder (highest wins):
      lifted_or_transport : lift > delta_lift
      grasped             : hand at cube (dist < grasp_radius) AND in contact
      reaching            : hand approaching (grasp_radius <= dist < reach_radius)
      no_contact          : otherwise
    """
    eff = np.asarray(eff_xyz, dtype=float)
    cube = np.asarray(cube_xyz, dtype=float)
    lift = np.asarray(lift, dtype=float).reshape(-1)
    contact = np.asarray(contact, dtype=float).reshape(-1)
    dist = np.linalg.norm(eff - cube, axis=1)

    out = np.zeros(len(cube), dtype=np.int64)  # default no_contact (0)
    out[dist < reach_radius] = 1
    out[(dist < grasp_radius) & (contact > tau_contact)] = 2
    out[lift > delta_lift] = 3
    return out


def bucket_histogram(labels: np.ndarray) -> Dict[str, float]:
    """Fraction of the pool in each interaction bucket."""
    labels = np.asarray(labels)
    n = max(len(labels), 1)
    return {name: float((labels == i).sum()) / n for i, name in enumerate(INTERACTION_BUCKETS)}


def participation_ratio(Z: np.ndarray) -> float:
    """Effective dimensionality of the z cloud: (Σλ)² / Σλ², λ = cov eigenvalues.

    1 => all variance in one direction; D => isotropic over D dims.
    """
    Z = np.asarray(Z, dtype=float)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / max(len(Z) - 1, 1)
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0, None)
    s = lam.sum()
    if s <= 0:
        return 0.0
    return float(s * s / (lam @ lam))


def separability_auc(Z: np.ndarray, pos: np.ndarray) -> float:
    """AUC of a linear (Fisher LDA) classifier separating pos vs ~pos in z.

    Closed-form LDA direction + Mann-Whitney rank-statistic AUC (no sklearn).
    Returns 0.5 when a class is empty (undefined => chance). Symmetric: the
    returned value is max(auc, 1-auc) so direction sign doesn't matter.
    """
    Z = np.asarray(Z, dtype=float)
    pos = np.asarray(pos, dtype=bool)
    if pos.all() or (~pos).all():
        return 0.5
    mu1, mu0 = Z[pos].mean(0), Z[~pos].mean(0)
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / max(len(Z) - 1, 1)
    cov += 1e-6 * np.eye(cov.shape[0])
    w = np.linalg.solve(cov, mu1 - mu0)
    s = Z @ w
    s1, s0 = s[pos], s[~pos]
    # AUC = P(s1 > s0) via rank statistic
    order = np.argsort(np.concatenate([s0, s1]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r1 = ranks[len(s0):].sum()
    auc = (r1 - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))
    return float(max(auc, 1.0 - auc))


def coverage_score(train_Z: np.ndarray, query_z: np.ndarray) -> Dict[str, np.ndarray]:
    """Score each query z against the training z cloud.

    Returns per-query:
      cos_nn       : max cosine similarity to any training z (in-distribution => ~1)
      mahalanobis  : Mahalanobis distance to the training cloud (large => OOD)
    """
    train_Z = np.asarray(train_Z, dtype=float)
    query_z = np.atleast_2d(np.asarray(query_z, dtype=float))

    tn = train_Z / (np.linalg.norm(train_Z, axis=1, keepdims=True) + 1e-8)
    qn = query_z / (np.linalg.norm(query_z, axis=1, keepdims=True) + 1e-8)
    cos_nn = (qn @ tn.T).max(axis=1)

    mu = train_Z.mean(0)
    Zc = train_Z - mu
    cov = (Zc.T @ Zc) / max(len(train_Z) - 1, 1)
    cov += 1e-6 * np.eye(cov.shape[0])
    prec = np.linalg.inv(cov)
    d = query_z - mu
    maha = np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d).clip(0))
    return {"cos_nn": cos_nn, "mahalanobis": maha}
