"""The eval-layer FULL successor measure over a trained (z,x)-separated checkpoint
(spec 10 L1+L2; gated by the SMB failures measured 2026-07-26).

The trained ZX critic gives the kernel-smoothed mass
    M-tilde(s,a,z,g) ~= M^{pi_z}(s,a, B_eps(g))            (ball mass at center g).
Two constructions promote it to a bona fide measure WITHOUT retraining:

  density   m-hat(s,a,z,x)  = M-tilde(s,a,z,x) / N-hat(x)          (dM/d rho)
  set mass  M-hat(s,a,z,X)  = (1/n) sum_{i: x_i in X} m-hat(x_i)   (any X, additive
                                                                    BY CONSTRUCTION)
  integral  Q-hat_r(s,a,z)  = (1/m) sum_j r_j m-hat(g_j)           (∫ r dM, g_j ~ rho)

where N-hat(x) = rho(B_eps(x)) is the dataset mass of the SAME kernel ball -- a pure
kernel count, no training. The eps-smoothing biases cancel to first order in the
ratio (numerator and denominator share the kernel, including clipped-ball boundary
effects at the table).

Inference-side (spec 10 §1-2):
  corrected moment  z_r  propto  (1/m) sum_j [r_j / N-hat(g_j)] psi(g_j)
  measure-ranked selection: score candidate z's by Q-hat_r(s0, a0~pi(s0,z), z) and
  act with the argmax (the Soft-FB-style z-search; GPI over the continuum's samples).

Numerics: N-hat is floored at `floor_counts` dataset points (half a count by
default) so 1/N-hat cannot explode where support is thin; the effective sample
size of the corrected moment is reported so degenerate inferences are visible.

All functions are pure numpy over caller-supplied readout closures, so the module
is unit-testable without jax or checkpoints (tests/test_full_measure.py).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

EPS_OBS = 0.4          # the measure's own kernel radius (4 cm), spec-wide


class FullMeasure:
    """Kernel-denominator layer over a dataset of cube positions (obs units).

    Parameters
    ----------
    cube_ds : [n,3] dataset cube positions (obs units) -- defines rho and N-hat.
    eps : kernel radius; MUST equal the radius the checkpoint was trained with.
    floor_counts : minimum ball occupancy (in dataset points) used in 1/N-hat.
    max_members : per-set-query cap on evaluated member samples (subsampled
        uniformly; the estimate stays unbiased, variance reported via n_used).
    """

    def __init__(self, cube_ds, eps: float = EPS_OBS, floor_counts: float = 0.5,
                 max_members: int = 64, seed: int = 0):
        self.cube = np.asarray(cube_ds, np.float64)
        self.n = len(self.cube)
        self.eps = float(eps)
        self.tree = cKDTree(self.cube)
        self.floor = float(floor_counts) / self.n
        self.rng = np.random.default_rng(seed)
        self.max_members = int(max_members)

    # ------------------------------------------------------------- N-hat ----
    def nhat(self, centers) -> np.ndarray:
        """rho(B_eps(center)) as a fraction of the dataset, floored."""
        centers = np.atleast_2d(np.asarray(centers, np.float64))
        counts = self.tree.query_ball_point(centers, self.eps,
                                            return_length=True)
        return np.maximum(np.asarray(counts, np.float64) / self.n, self.floor)

    # ----------------------------------------------------------- density ----
    def density(self, ball_vals, centers) -> np.ndarray:
        """m-hat = M-tilde / N-hat at the given centers (obs-unit positions)."""
        return np.asarray(ball_vals, np.float64) / self.nhat(centers)

    # -------------------------------------------------------- set queries ----
    def member_indices(self, query: dict) -> np.ndarray:
        """Dataset row indices inside a query region.
        query: {'kind':'ball','center':[3],'eps':r} | {'kind':'box','center':[3],
        'half':[3]} | {'kind':'union','parts':[subqueries]}."""
        kind = query["kind"]
        if kind == "ball":
            idx = self.tree.query_ball_point(
                np.asarray(query["center"], np.float64),
                float(query.get("eps", self.eps)))
            return np.asarray(sorted(idx), np.int64)
        if kind == "box":
            c = np.asarray(query["center"], np.float64)
            h = np.asarray(query["half"], np.float64)
            m = np.all(np.abs(self.cube - c) <= h, axis=1)
            return np.nonzero(m)[0]
        if kind == "union":
            parts = [self.member_indices(p) for p in query["parts"]]
            return np.unique(np.concatenate(parts)) if parts else \
                np.zeros(0, np.int64)
        raise ValueError(f"unknown query kind {kind!r}")

    def set_mass(self, ball_value_fn, member_idx) -> dict:
        """M-hat(X) = (1/n) sum_{i in X} m-hat(x_i), subsampled to max_members.

        ball_value_fn(idx [K]) -> M-tilde values at the dataset rows idx (the
        caller closes over (s0, a0, z) and supplies full observations).
        Additive over disjoint X by construction (a sum over dataset rows)."""
        member_idx = np.asarray(member_idx, np.int64)
        n_x = len(member_idx)
        if n_x == 0:
            return dict(mass=0.0, n_members=0, n_used=0)
        use = member_idx if n_x <= self.max_members else \
            self.rng.choice(member_idx, self.max_members, replace=False)
        dens = self.density(ball_value_fn(use), self.cube[use])
        return dict(mass=float(n_x / self.n * dens.mean()),
                    n_members=int(n_x), n_used=int(len(use)),
                    density_mean=float(dens.mean()))

    # ----------------------------------------------------------- ∫ r dM ----
    def q_r(self, ball_value_fn, sample_idx, rewards,
            cap_q: float | None = None) -> dict:
        """Q-hat_r = (1/m) sum_j r_j m-hat(g_j) over reward samples g_j ~ rho.
        Skips zero-reward samples (they contribute exactly 0) so sparse rewards
        cost only their support.

        cap_q: winsorize the density's UPPER tail at this quantile of the live
        densities before integrating. Motivation (gate iteration 2026-07-26):
        thin-support regions (~20x fewer dataset points) let 1/N-hat amplify any
        critic over-smoothing, so cross-region comparisons (T3 candidate
        scoring) overweight exactly the regions the data can least support --
        the classic importance-weight tail. Capping is applied across the
        call's live samples, i.e. precisely where regions compete."""
        sample_idx = np.asarray(sample_idx, np.int64)
        r = np.asarray(rewards, np.float64)
        live = np.nonzero(r != 0.0)[0]
        if len(live) == 0:
            return dict(q=0.0, n_live=0)
        dens = self.density(ball_value_fn(sample_idx[live]),
                            self.cube[sample_idx[live]])
        n_capped = 0
        if cap_q is not None and len(dens) > 1:
            hi = float(np.quantile(dens, cap_q))
            n_capped = int((dens > hi).sum())
            dens = np.minimum(dens, hi)
        return dict(q=float((r[live] * dens).sum() / len(r)),
                    n_live=int(len(live)), n_capped=n_capped)

    # ------------------------------------------------- corrected moment z ----
    def corrected_moment_z(self, Psi, rewards, centers, match_norm: float) -> dict:
        """z_r propto (1/m) sum_j [r_j / N-hat(g_j)] psi(g_j), rescaled to the
        data norm (spec 10 §1: the 1/N-hat weight converts the mass readout's
        moment into the density-correct one; same-kernel biases cancel).
        Reports the importance-weight effective sample size."""
        Psi = np.asarray(Psi, np.float64)
        r = np.asarray(rewards, np.float64)
        w = np.abs(r) / self.nhat(centers)
        ess = float(w.sum() ** 2 / (np.square(w).sum() + 1e-12))
        z = (r / self.nhat(centers))[:, None] * Psi
        z = z.mean(0)
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            return dict(z=None, ess=ess, raw_norm=nz)
        return dict(z=(z * match_norm / nz).astype(np.float32),
                    ess=ess, raw_norm=nz)


def select_z_by_measure(candidates, score_fn) -> dict:
    """Measure-ranked z selection (spec 10 §2): score each candidate z by the
    measure's own value estimate for the task reward and return the argmax.
    candidates: list of (name, z); score_fn(z) -> float (Q-hat_r at s0 with
    a0 ~ pi(s0, z) -- the caller owns the closure)."""
    scored = [(name, z, float(score_fn(z))) for name, z in candidates]
    name, z, s = max(scored, key=lambda t: t[2])
    return dict(z=z, chosen=name, score=s,
                scores={n: sc for n, _, sc in scored})
