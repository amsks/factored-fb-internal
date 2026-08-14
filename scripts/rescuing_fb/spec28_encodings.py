"""scripts/rescuing_fb/spec28_encodings.py -- spec 28 P0b/P0c PURE LOGIC.

Jax-free, unit-tested numpy core shared by `eval_spec28_p0b.py` (three-encoding
reward reconstruction) and `eval_spec28_p0c.py` (e2b assignment with a ridge
index). Kept in its own module for the same reason spec 24's harness split its
`analytic_zr`/`gate_control_match` out: everything here is testable without a
checkpoint, a GPU, or mujoco.

THE MECHANISM UNDER TEST (spec 28 sec.0). Density-FB's softmax normalization
destroys the linear identity `Q_r = F^T E[r B]` that made `z_r = E[r B]` a valid
reward->index inverse in standard FB, and the CE loss pins only the PRODUCT
`F^T B`, leaving B's Euclidean geometry gauge-arbitrary. `z(g) =
sqrt(d)*normalize(B_bar(g))` and `E[r B]` are both PRIMAL feature vectors; the
actor readout `r_z(u) = B(u)^T z / sqrt(d)` needs a DUAL coefficient
`c = (Sigma_B + lambda I)^{-1} m`. Primal and dual coincide only under the
whitened geometry FB's orthonormality loss enforces -- which this project
dropped deliberately and keeps dropped. So the repair candidate is the ridge
coefficient, and these functions score it against the two primal encodings.

SCALE INVARIANCE (deliberate, and a self-check): AUC, Spearman and top-k
precision are invariant to a positive rescaling of `z`, and `margin_std` divides
by the reconstruction's own std, so the normalized and un-normalized moment
MUST score identically on every metric here. That equality is asserted by a unit
test -- it is the cheapest possible check that "normalization" is not what is
being measured; only the ridge SOLVE can move these numbers.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------- encodings --


def moment_index(b_pool: np.ndarray, r_pool: np.ndarray) -> np.ndarray:
    """m = E_pool[r(u) B(u)] -- the UN-normalized moment (spec 28 P0b's
    "moment, un-normalized variant"). `analytic_zr` (spec 24, reused as-is by
    the P0b/P0c runners) is exactly `sqrt(d)*normalize(this)`.
    b_pool [P,d], r_pool [P] -> [d]."""
    b = np.asarray(b_pool, np.float64)
    r = np.asarray(r_pool, np.float64)
    assert b.ndim == 2 and r.ndim == 1 and len(b) == len(r), \
        f"moment_index: b_pool {b.shape} / r_pool {r.shape} mismatch"
    return np.mean(r[:, None] * b, axis=0)


def ridge_lambdas(b_pool: np.ndarray, mults=(1e-3, 1e-2, 1e-1)) -> dict:
    """{mult: lambda} with lambda = mult * tr(Sigma_B)/d (spec 28 P0b's sweep),
    Sigma_B = B^T B / P. tr(Sigma)/d is the mean eigenvalue, so the multiplier
    is dimensionless and comparable across checkpoints/towers."""
    b = np.asarray(b_pool, np.float64)
    d = b.shape[1]
    scale = float(np.trace(b.T @ b) / len(b) / d)
    return {float(m): float(m) * scale for m in mults}


def ridge_index(b_pool: np.ndarray, r_pool: np.ndarray, lam: float) -> np.ndarray:
    """c = (Sigma_B + lambda I)^{-1} m, Sigma_B = B^T B / P, m = E[r B]. The DUAL
    coefficient. This is the SAME object C2's training-time
    `_density_rdecode_loss` solves (`v* = (B^T B + lambda I)^{-1} B^T r_c`, up to
    the shared 1/P that cancels between Sigma and m for a correspondingly scaled
    lambda) -- spec 28 sec.0's point is that the loss computed it and the
    INTERFACE never used it. b_pool [P,d], r_pool [P] -> [d]."""
    b = np.asarray(b_pool, np.float64)
    r = np.asarray(r_pool, np.float64)
    P, d = b.shape
    sigma = b.T @ b / P
    m = moment_index(b, r)
    return np.linalg.solve(sigma + float(lam) * np.eye(d), m)


def reconstruct(b_pool: np.ndarray, z: np.ndarray) -> np.ndarray:
    """rhat(u) = B(u)^T z over the pool [P]. The readout the actor's reward term
    `B~(u)^T z / sqrt(d)` is (a positive multiple of)."""
    return np.asarray(b_pool, np.float64) @ np.asarray(z, np.float64)


# ------------------------------------------------------------------- scoring --


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared -- the tie handling Spearman/AUC
    both need."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average-rank tie correction (Pearson on ranks)."""
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if den == 0.0:
        return 0.0
    return float(ra @ rb / den)


def auc_hit(score: np.ndarray, hit: np.ndarray):
    """Mann-Whitney AUC of `score` separating hit (1) from non-hit (0), with
    average-rank tie handling (a tied score contributes 0.5). Returns
    (auc or None if either class is empty, n_hit)."""
    score = np.asarray(score, np.float64)
    hit = np.asarray(hit).astype(bool)
    n1 = int(hit.sum())
    n0 = int(len(hit) - n1)
    if n1 == 0 or n0 == 0:
        return None, n1
    r = _rank(score)
    return float((r[hit].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), n1


def topk_precision(score: np.ndarray, target: np.ndarray, k: int) -> float:
    """|top-k(score) ∩ top-k(target)| / k -- how much of the exact kernel's own
    top-k the reconstruction recovers."""
    k = int(min(k, len(score)))
    a = set(np.argsort(-np.asarray(score, np.float64), kind="mergesort")[:k].tolist())
    b = set(np.argsort(-np.asarray(target, np.float64), kind="mergesort")[:k].tolist())
    return float(len(a & b) / k)


def hard_negatives(dist: np.ndarray, hit: np.ndarray, n_hard: int) -> np.ndarray:
    """Boolean mask of the `n_hard` NON-hit pool points nearest the reward
    center (nearest in the eps-scaled coordinate the hit rule itself uses) --
    the negatives a reward reconstruction has to push down and a mere "points
    roughly over there" signal will not."""
    dist = np.asarray(dist, np.float64)
    nz = np.flatnonzero(~np.asarray(hit).astype(bool))
    take = nz[np.argsort(dist[nz], kind="mergesort")[:int(n_hard)]]
    mask = np.zeros(len(dist), bool)
    mask[take] = True
    return mask


def margin_std(score: np.ndarray, hit: np.ndarray, hard: np.ndarray):
    """(mean score over hits - mean score over hard negatives) / std(score) --
    scale-free so it is comparable across encodings whose z has a different
    norm. None when either set is empty."""
    score = np.asarray(score, np.float64)
    hit = np.asarray(hit).astype(bool)
    hard = np.asarray(hard).astype(bool)
    if hit.sum() == 0 or hard.sum() == 0:
        return None
    sd = float(score.std())
    if sd == 0.0:
        return 0.0
    return float((score[hit].mean() - score[hard].mean()) / sd)


def score_rhat(rhat: np.ndarray, kg: np.ndarray, hit: np.ndarray,
               dist: np.ndarray, topks=(16, 64, 256), n_hard: int = 256) -> dict:
    """All four spec-28 P0b scores for an arbitrary pool-shaped reconstruction.
    Split out from `score_encoding` so the INSTRUMENT CONTROL can be run through
    the identical scoring path: feeding the probe the target `kg` itself must
    return AUC/Spearman/top-k = 1.0, and a random index must return ~0.5/0/0.
    A probe that cannot score its own target perfectly is broken, and no z cell
    is read until both ends read correctly (the three-times-burned rule)."""
    rhat = np.asarray(rhat, np.float64)
    auc, n_hit = auc_hit(rhat, hit)
    hard = hard_negatives(dist, hit, max(n_hard, n_hit))
    return {
        "auc": auc,
        "n_hit": int(n_hit),
        "spearman_kg": spearman(rhat, kg),
        "topk_prec": {str(k): topk_precision(rhat, kg, k) for k in topks},
        "margin_std": margin_std(rhat, hit, hard),
    }


def score_encoding(b_pool: np.ndarray, z: np.ndarray, kg: np.ndarray,
                   hit: np.ndarray, dist: np.ndarray,
                   topks=(16, 64, 256), n_hard: int = 256) -> dict:
    """`score_rhat` of the readout `rhat(u) = B(u)^T z` for one candidate index."""
    out = score_rhat(reconstruct(b_pool, z), kg, hit, dist, topks, n_hard)
    out["z_norm"] = float(np.linalg.norm(z))
    return out


# ------------------------------------------------- coordinate percentile (P0c) --


def coord_percentile(cand_coords_scaled: np.ndarray, center_scaled: np.ndarray,
                     selected: int) -> float:
    """THE spec-27 metric (pinned by reproducing its archived .35/.47 from the
    archived e2b JSONs): the fraction of candidates STRICTLY CLOSER to the true
    reward center than the selected candidate. 0.0 = the argmax picked the
    nearest candidate in the set; 0.50 = chance; LOWER IS BETTER. Distances are
    taken in the eps-scaled coordinate (`HitSpec.scaled`), the same units the
    hit rule uses.

    Why this and not policy success: the checkpoints' actors sit at the BC
    floor, so every rollout-terminated metric reads ~0 for every arm and cannot
    isolate the readout -- the trap that has now cost this program three
    rounds (spec 27 sec.6.5)."""
    c = np.asarray(cand_coords_scaled, np.float64)
    ctr = np.asarray(center_scaled, np.float64).reshape(-1)
    d = np.linalg.norm(c - ctr[None], axis=-1)
    return float(np.mean(d < d[int(selected)]))


def family_mean_aggregate(per_cell: dict) -> float:
    """The archived spec-27 aggregate: the UNWEIGHTED MEAN OF THE PER-FAMILY
    MEANS (not the mean over cells). `per_cell` maps family -> {cell: value}.
    Verified against the archived e2b JSONs: this reproduces 0.3469 -> "0.35"
    (z25_cs_actor) and 0.4734 -> "0.47" (z25_cs_full); the flat mean over all
    14 cells gives 0.343/0.469 and would NOT reproduce the actor figure."""
    fam_means = []
    for fam in sorted(per_cell):
        vals = [v for v in per_cell[fam].values() if v is not None]
        if vals:
            fam_means.append(float(np.mean(vals)))
    assert fam_means, "family_mean_aggregate: no cells"
    return float(np.mean(fam_means))
