"""The playpen MDP: a minimal tabular environment exhibiting the structures the
two-phase successor-measure curriculum builds in (spec 11 §6; note §3-§4).

Layout: a 7x7 grid (49 states).
  - semantic coordinate = COLUMN  (analog of cube xyz: the goal-region quotient)
  - interaction mode    = ROW      (top row is a CONVEYOR: the moving/carried mode)
Dynamics (5 actions: stay/up/down/left/right):
  - bowl cells (2x2 patch, rows 2-3) are STICKY: a move attempt leaves w.p. 0.1
    (cube-at-rest analog)  -> large recurrence.
  - conveyor row (top): drifts +2 columns per step regardless of action ("carried"/
    moving mode -- fast transport, but position cannot be held); `down` steps off.
    A first arrival into column c on the conveyor is ejected from the column next
    step -> transient arrival mode. The column quotient cannot see the difference.
  - elsewhere: intended move w.p. 0.95, slip-stay 0.05; walls bump.
Behavior/"play" policy: goal-directed mixture over reach-a-column policies + noise
(play data is purposeful, not a random walk). gamma = 0.99 for the exact demos
(cube-like: reaching values compress toward 1, recurrence spans a wide range);
gamma = 0.95 for the SGD estimator face-off (contraction speed).

Demos:
  1. occupancy = reaching x recurrence: EXACT for singleton goals (renewal
     identity); under play data the recurrence factor dominates M's variance while
     the reaching term dominates the ACTION ranking; the column quotient breaks
     strict arrival-independence of R (E5 in miniature: conveyor vs ground entry).
  2. T3 occupancy-weighted choice in miniature: arrival ranking picks the conveyor
     region, occupancy ranking picks the bowl; exact policy evaluation confirms the
     bowl. Plus the diagonal-slice defect: slice-sum exceeds any single policy's
     achievable value.
  3. estimator face-off at rank d=8 (Adam SGD on offline play transitions, fixed
     policy): FB density-TD (no-ortho / ortho) vs bounded column-region regression
     (absorbing-first curriculum and direct visitation). Reports scale traces,
     relative error, per-state goal-ranking. NOTE the actor loop is deliberately
     absent here -- this isolates the estimator/target-conditioning story only.

Usage:  .venv/bin/python scripts/rescuing_fb/tabular_playpen.py [--steps 30000]
Writes analysis/rescuing-fb/tabular_playpen.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# ----------------------------------------------------------------- MDP ----
W, H = 7, 7
N_GRID = W * H
PIT = N_GRID                     # falling off the belt end lands here (costly)
N_S = N_GRID + 1
N_A = 5                          # stay, up, down, left, right
G_EXACT = 0.99                   # exact demos (cube-like horizon)
G_TD = 0.95                      # SGD face-off
BOWL = {(2, 2), (2, 3), (3, 2), (3, 3)}
BOWL_STICK = 0.9
SLIP = 0.05
CONV = H - 1                     # conveyor row


def sid(row, col):
    return row * W + col


def column_of(s):
    return -1 if s == PIT else s % W


def build_P():
    P = np.zeros((N_S, N_A, N_S))
    # pit: hard to escape (E[escape] ~ 33 steps) -> re-entry after falling is
    # expensive relative to the horizon, like re-fetching a dropped cube.
    P[PIT, :, PIT] = 0.97
    P[PIT, :, sid(0, W - 1)] = 0.03
    for s in range(N_GRID):
        r, c = s // W, s % W
        if r == CONV:
            # conveyor: drift +2 cols regardless of action; `down` (a=2) steps off
            # to the row below at the drifted column; falls off at the right end.
            for a in range(N_A):
                nc = c + 2
                if a == 2:
                    P[s, a, sid(CONV - 1, min(nc, W - 1))] = 1.0
                elif nc >= W:
                    P[s, a, PIT] = 1.0                     # rides off the end
                else:
                    P[s, a, sid(CONV, nc)] = 1.0
            continue
        for a in range(N_A):
            if a == 0:
                tgt = s
            elif a == 1:
                tgt = sid(r + 1, c) if r + 1 < H else s
            elif a == 2:
                tgt = s if r == 0 else sid(r - 1, c)
            elif a == 3:
                tgt = s if c == 0 else sid(r, c - 1)
            else:
                tgt = s if c == W - 1 else sid(r, c + 1)
            if (r, c) in BOWL and a != 0:
                P[s, a, tgt] += 1.0 - BOWL_STICK
                P[s, a, s] += BOWL_STICK
            elif a == 0:
                P[s, a, tgt] += 1.0
            else:
                P[s, a, tgt] += 1.0 - SLIP
                P[s, a, s] += SLIP
    assert np.allclose(P.sum(-1), 1.0)
    return P


def policy_matrix(P, pi):
    return np.einsum("sa,sax->sx", pi, P)


def occupancy(P, pi, ind, gamma):
    """Exact M^pi(s,a,X)=E[sum_t gamma^t 1[s_{t+1} in X]] and from-state N^pi(x,X)."""
    Ppi = policy_matrix(P, pi)
    N = np.linalg.solve(np.eye(N_S) - gamma * Ppi, Ppi @ ind)
    M = np.einsum("sax,x->sa", P, ind + gamma * N)
    return M, N


def reach_Q(P, ind, gamma, pi=None, iters=4000, tol=1e-12):
    """First-passage value Q(s,a)=E[gamma^{T-1}] to X (absorbing).
    pi=None -> greedy control; else policy evaluation."""
    Q = np.zeros((N_S, N_A))
    for _ in range(iters):
        V = Q.max(1) if pi is None else (pi * Q).sum(1)
        Qn = np.einsum("sax,x->sa", P, ind + gamma * (1.0 - ind) * V)
        if np.abs(Qn - Q).max() < tol:
            return Qn
        Q = Qn
    return Q


def greedy(Q):
    pi = np.zeros((N_S, N_A))
    pi[np.arange(N_S), Q.argmax(1)] = 1.0
    return pi


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def value_of_policy(P, pi, r_state, gamma):
    Ppi = policy_matrix(P, pi)
    return np.linalg.solve(np.eye(N_S) - gamma * Ppi, Ppi @ r_state)


def col_ind(c):
    return np.array([1.0 if column_of(s) == c else 0.0 for s in range(N_S)])


def play_policy(P, gamma):
    """Goal-directed 'play': mixture of greedy reach-a-column policies + noise."""
    mix = np.mean([greedy(reach_Q(P, col_ind(c), gamma)) for c in range(W)], axis=0)
    return 0.75 * mix + 0.25 * np.full((N_S, N_A), 1.0 / N_A)


# ------------------------------------------------------------- demo 1 ----
def demo1(P, mu, gamma):
    out = {}
    errs, Rs, Ms, Qs = [], [], [], []
    for g in range(N_GRID):
        ind = np.zeros(N_S); ind[g] = 1.0
        M, N = occupancy(P, mu, ind, gamma)
        Q = reach_Q(P, ind, gamma, pi=mu)
        R = 1.0 + gamma * N[g]
        errs.append(np.abs(M - Q * R).max())
        Rs.append(R); Ms.append(M); Qs.append(Q)
    Rs = np.array(Rs)
    Mflat = np.stack(Ms).reshape(N_GRID, -1)
    Qflat = np.stack(Qs).reshape(N_GRID, -1)
    out["singleton_factorization_max_abs_err"] = float(np.max(errs))
    out["R_mean"], out["R_min"], out["R_max"] = (
        float(Rs.mean()), float(Rs.min()), float(Rs.max()))
    out["R_bowl_mean"] = float(np.mean([Rs[sid(r, c)] for r, c in BOWL]))
    out["R_conveyor_mean"] = float(np.mean([Rs[sid(CONV, c)] for c in range(W)]))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, Mflat.size, 20000)
    g_idx, sa_idx = np.unravel_index(idx, Mflat.shape)
    m = Mflat[g_idx, sa_idx]
    out["spearman_M_vs_Rgoal"] = spearman(m, Rs[g_idx])
    out["spearman_M_vs_Qreach"] = spearman(m, Qflat[g_idx, sa_idx])
    # the note's statistic: predict occupancy ACROSS GOALS (goal-only R(g)
    # regressor vs the reaching term), averaged over query (s,a)
    sa_sample = rng.integers(0, Mflat.shape[1], 300)
    out["crossgoal_spearman_M_vs_Rgoal"] = float(np.mean(
        [spearman(Mflat[:, k], Rs) for k in sa_sample]))
    out["crossgoal_spearman_M_vs_Qreach"] = float(np.mean(
        [spearman(Mflat[:, k], Qflat[:, k]) for k in sa_sample]))

    # column-region quotient: arrival-mode dependence + argmax agreement.
    # NOTE the arrival CV is structurally SMALL here: any column is re-enterable in
    # a few steps, so at gamma=0.99 a transient arrival costs little -- unlike the
    # cube, where a dropped object must be re-fetched (E5's refutation needs
    # expensive re-entry). The playpen demonstrates the factorization's usefulness,
    # not its cube-scale limits.
    agree, cvs = [], []
    Mcols, Qcols, Rcols = [], [], []
    for c in range(W):
        ind = col_ind(c)
        Qg = reach_Q(P, ind, gamma)
        pig = greedy(Qg)
        M, N = occupancy(P, pig, ind, gamma)
        Qe = reach_Q(P, ind, gamma, pi=pig)
        Rarr = 1.0 + gamma * N[ind > 0]              # post-arrival occ by entry state
        cvs.append(float(Rarr.std() / Rarr.mean()))
        far = ind < 0.5
        agree.append(float((M[far].argmax(1) == Qe[far].argmax(1)).mean()))
        # region-level stats under the PLAY policy (the note's setting)
        Mp, Np = occupancy(P, mu, ind, gamma)
        Qp = reach_Q(P, ind, gamma, pi=mu)
        Mcols.append(Mp.reshape(-1)); Qcols.append(Qp.reshape(-1))
        Rcols.append(float(np.mean(1.0 + gamma * Np[ind > 0])))
    out["column_region_argmax_agreement_mean"] = float(np.mean(agree))
    out["column_region_R_arrival_cv_mean"] = float(np.mean(cvs))
    # the note's magnitude statistic at REGION level: across the 7 column regions,
    # does the goal-only recurrence R(region) or the reaching term predict occupancy?
    Mc = np.stack(Mcols); Qc = np.stack(Qcols); Rc = np.array(Rcols)
    rng2 = np.random.default_rng(1)
    sa_s = rng2.integers(0, Mc.shape[1], 300)
    out["region_crossgoal_spearman_M_vs_R"] = float(np.mean(
        [spearman(Mc[:, k], Rc) for k in sa_s]))
    out["region_crossgoal_spearman_M_vs_Q"] = float(np.mean(
        [spearman(Mc[:, k], Qc[:, k]) for k in sa_s]))
    out["region_Q_range"] = [float(Qc.min()), float(Qc.max())]
    out["region_R_range"] = [float(Rc.min()), float(Rc.max())]
    return out


# ------------------------------------------------------------- demo 2 ----
def demo2(P, gamma):
    out = {}
    indA = np.zeros(N_S)
    for r, c in BOWL:
        indA[sid(r, c)] = 1.0                        # stable bowl region
    indB = np.zeros(N_S); indB[sid(CONV, 4)] = 1.0   # transient conveyor cell
    s0 = sid(0, 0)
    res = {}
    for name, ind in [("A_bowl", indA), ("B_conveyor", indB)]:
        Qg = reach_Q(P, ind, gamma)
        pig = greedy(Qg)
        M, _ = occupancy(P, pig, ind, gamma)
        Qe = reach_Q(P, ind, gamma, pi=pig)
        res[name] = dict(pi=pig, reach=float(Qe[s0].max()), occ=float(M[s0].max()))
    qa, qb = res["A_bowl"]["reach"], res["B_conveyor"]["reach"]
    ma, mb = res["A_bowl"]["occ"], res["B_conveyor"]["occ"]
    out["reach_A"], out["reach_B"], out["occ_A"], out["occ_B"] = qa, qb, ma, mb
    lo, hi = qa / qb, ma / mb
    wB = float(np.sqrt(lo * hi)); wA = 1.0
    out["wB_between_ratios"] = dict(wB=wB, reach_ratio=lo, occ_ratio=hi)
    out["arrival_ranking_picks"] = "B" if wB * qb > wA * qa else "A"
    out["occupancy_ranking_picks"] = "B" if wB * mb > wA * ma else "A"
    r_state = wA * indA + wB * indB
    vA = float(value_of_policy(P, res["A_bowl"]["pi"], r_state, gamma)[s0])
    vB = float(value_of_policy(P, res["B_conveyor"]["pi"], r_state, gamma)[s0])
    out["true_return_piA"], out["true_return_piB"] = vA, vB
    out["correct_region"] = "A" if vA > vB else "B"
    out["slice_sum"] = wA * ma + wB * mb
    out["best_single_policy_value"] = max(vA, vB)
    return out


# ------------------------------------------------------------- demo 3 ----
def sample_dataset(P, mu, n, rng):
    s = rng.integers(0, N_S, n)
    cdf_a = mu.cumsum(-1)
    a = (rng.random(n)[:, None] < cdf_a[s]).argmax(1)
    cdf = P.cumsum(-1)
    s2 = (rng.random(n)[:, None] < cdf[s, a]).argmax(1)
    return s, a, s2


class Fact:
    """Rank-d bilinear M(s,a,x) = F[s,a]·B[x], Adam + EMA target."""

    def __init__(self, d, rng, lr=1e-2, tau=0.01):
        self.F = rng.normal(0, 0.1, (N_S * N_A, d))
        self.B = rng.normal(0, 0.1, (N_S, d))
        self.Ft, self.Bt = self.F.copy(), self.B.copy()
        self.lr, self.tau, self.d = lr, tau, d
        self._m, self._v, self._t = {}, {}, 0

    def m(self, sa, x, target=False):
        F, B = (self.Ft, self.Bt) if target else (self.F, self.B)
        return (F[sa] * B[x]).sum(-1) / np.sqrt(self.d)

    def step(self, grads):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for arr, g in grads:
            key = id(arr)
            m = self._m.setdefault(key, np.zeros_like(arr))
            v = self._v.setdefault(key, np.zeros_like(arr))
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            arr -= self.lr * (m / (1 - b1**self._t)) / (
                np.sqrt(v / (1 - b2**self._t)) + eps)
        self.Ft += self.tau * (self.F - self.Ft)
        self.Bt += self.tau * (self.B - self.Bt)


def _scatter(shape, idx, vals):
    out = np.zeros(shape)
    np.add.at(out, idx, vals)
    return out


def _sample_a2(mu, s2, rng):
    cdf = mu.cumsum(-1)
    return (rng.random(len(s2))[:, None] < cdf[s2]).argmax(1)


def train_density(mu, data, d, steps, rng, gamma, ortho=0.0, batch=256):
    """FB density-TD: quadratic term at x~rho + linear observed-successor term."""
    s, a, s2 = data
    n = len(s)
    f = Fact(d, rng)
    trace = []
    for t in range(steps):
        i = rng.integers(0, n, batch)
        j = rng.integers(0, n, batch)
        sa = s[i] * N_A + a[i]
        sa2 = s2[i] * N_A + _sample_a2(mu, s2[i], rng)
        x = s2[j]
        diff = f.m(sa, x) - gamma * f.m(sa2, x, target=True)
        gF = _scatter(f.F.shape, sa, (diff / np.sqrt(d))[:, None] * f.B[x]) / batch
        gB = _scatter(f.B.shape, x, (diff / np.sqrt(d))[:, None] * f.F[sa]) / batch
        gF += _scatter(f.F.shape, sa, -f.B[s2[i]] / np.sqrt(d)) / batch
        gB += _scatter(f.B.shape, s2[i], -f.F[sa] / np.sqrt(d)) / batch
        if ortho > 0.0:
            Bb = f.B[x]
            C = Bb @ Bb.T
            off = C - np.diag(np.diag(C))
            gO = (2.0 * off @ Bb) / (batch**2) - 2.0 * Bb / batch
            gB += ortho * _scatter(f.B.shape, x, gO)
        f.step([(f.F, gF), (f.B, gB)])
        if t % 200 == 0:
            trace.append(float(np.abs(f.m(sa, x)).mean()))
    return f, trace


def train_region(mu, data, d, steps, rng, gamma, absorbing_first=True, batch=256):
    """Bounded COLUMN-region regression: phase 1 absorbing, phase 2 visitation."""
    s, a, s2 = data
    n = len(s)
    f = Fact(d, rng)
    cols = np.array([column_of(x) for x in range(N_S)])
    trace = []
    frozen = False
    plan = ([(True, steps // 2), (False, steps - steps // 2)]
            if absorbing_first else [(False, steps)])
    for absorbing, ph in plan:
        if not absorbing and absorbing_first:
            frozen = True                              # freeze psi in phase 2
        for t in range(ph):
            i = rng.integers(0, n, batch)
            j = rng.integers(0, n, batch)
            sa = s[i] * N_A + a[i]
            sa2 = s2[i] * N_A + _sample_a2(mu, s2[i], rng)
            x = s2[j]
            hit = ((cols[s2[i]] == cols[x]) & (cols[x] >= 0)).astype(float)
            y = hit + gamma * ((1.0 - hit) if absorbing else 1.0) * f.m(
                sa2, x, target=True)
            diff = f.m(sa, x) - y
            gF = _scatter(f.F.shape, sa,
                          (diff / np.sqrt(d))[:, None] * f.B[x]) / batch
            grads = [(f.F, gF)]
            if not frozen:
                gB = _scatter(f.B.shape, x,
                              (diff / np.sqrt(d))[:, None] * f.F[sa]) / batch
                grads.append((f.B, gB))
            f.step(grads)
            if t % 200 == 0:
                trace.append(float(np.abs(f.m(sa, x)).mean()))
    return f, trace


def demo3(P, mu, steps, gamma):
    out = {}
    rng = np.random.default_rng(0)
    data = sample_dataset(P, mu, 20000, rng)
    cols = np.array([column_of(x) for x in range(N_S)])
    Mcol = np.zeros((N_S, N_A, W))
    for c in range(W):
        Mcol[:, :, c], _ = occupancy(P, mu, (cols == c).astype(float), gamma)
    rho = np.bincount(data[2], minlength=N_S) / len(data[2])
    Mpt = np.zeros((N_S, N_A, N_S))
    for g in range(N_S):  # includes the pit as a point query (it has rho mass)
        ind = np.zeros(N_S); ind[g] = 1.0
        Mpt[:, :, g], _ = occupancy(P, mu, ind, gamma)
    m_true = Mpt / np.maximum(rho, 1e-8)[None, None, :]
    live = rho > 0
    out["density_target_dynamic_range"] = float(
        np.quantile(m_true[:, :, live], 0.999) /
        max(np.quantile(m_true[:, :, live], 0.5), 1e-9))
    out["region_target_max"] = float(Mcol.max())

    d = 8
    sa_grid = np.arange(N_S * N_A)
    true = Mcol.reshape(-1, W)
    results = {}
    for name, (fn, kw) in {
        "density_noortho": (train_density, dict(ortho=0.0)),
        "density_ortho1": (train_density, dict(ortho=1.0)),
        "region_curriculum": (train_region, dict(absorbing_first=True)),
        "region_direct_visit": (train_region, dict(absorbing_first=False)),
    }.items():
        f, trace = fn(mu, data, d, steps, np.random.default_rng(1), gamma, **kw)
        if name.startswith("density"):
            pred_pt = (f.F[sa_grid] @ f.B.T) / np.sqrt(d)
            pred = np.stack([(pred_pt * rho[None, :])[:, cols == c].sum(1)
                             for c in range(W)], axis=1)
        else:
            centers = [int(np.argmax(cols == c)) for c in range(W)]
            pred = (f.F[sa_grid] @ f.B[centers].T) / np.sqrt(d)
        rel = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
        rank = float(np.mean([spearman(pred[k], true[k])
                              for k in range(0, len(true), 5)]))
        results[name] = dict(rel_err=rel, mean_rank_corr=rank,
                             scale_final=trace[-1], scale_max=float(np.max(trace)),
                             scale_trace=trace[::5])
    out["fits"] = results
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--out", default="analysis/rescuing-fb/tabular_playpen.json")
    args = ap.parse_args()
    P = build_P()
    mu_e = play_policy(P, G_EXACT)
    report = {
        "mdp": dict(states=N_S, actions=N_A, gamma_exact=G_EXACT, gamma_td=G_TD,
                    bowl_stick=BOWL_STICK, conveyor_row=CONV),
        "demo1_factorization": demo1(P, mu_e, G_EXACT),
        "demo2_occupancy_choice": demo2(P, G_EXACT),
        "demo3_estimators": demo3(P, play_policy(P, G_TD), args.steps, G_TD),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "demo3_estimators"},
                     indent=2))
    d3 = report["demo3_estimators"]
    print("demo3 (rank-8 fits):")
    for k, v in d3["fits"].items():
        print(f"  {k:22s} rel_err={v['rel_err']:.3f} rank={v['mean_rank_corr']:.3f} "
              f"scale_final={v['scale_final']:.2f} scale_max={v['scale_max']:.2f}")
    print("density target dynamic range:", round(d3["density_target_dynamic_range"], 1),
          "| region target max:", round(d3["region_target_max"], 2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
