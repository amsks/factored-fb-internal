"""The paper's tabular bottleneck suite, rebuilt on the single-bottleneck process family.

The process (analysis/lemma_check/design_single_bottleneck.py, `build`): an entry chain, ONE
bottleneck state b whose two actions enter cluster A and cluster B, and either a mutual leak
between the clusters (tail 0) or a shared closed tail every cluster state can exit into.  The
decision is at b, the reward is the INDICATOR OF CLUSTER A (no goal state), and the quantity
measured everywhere is the fraction of the exact action gap at b that a fit reads,
(Q_hat(b, a0) - Q_hat(b, a1)) / (Q(b, a0) - Q(b, a1)).  Discount 0.8, uniform reference 1/n,
exact population expectations, float64.

Processes
  P6   build(entry=1, cluster=2, tail=0, mutual=True): 6 states, uniform policy (the lemma figure)
  P11  build(entry=2, cluster=2, tail=4, merge=True): 11 states, uniform policy (the rank sweep)
  R1-5 random draws from the template entry in {1,2,3}, cluster in {2,3}, tail in {0,2,4} (tail 0
       => mutual leak), action probabilities U(0.25, 0.75) per state; kept iff FB's closed-form
       rank-2 fit reads < 0.5 of the exact gap and the exact gap is > 0.5.

Measurements (every harness call is the existing code: analysis/toy_ablation/family.py for the six
arms A-F, closed-form and TD-trained; analysis/rare_bottleneck/run.py for the row-reweighting
sweep; analysis/lemma_check/run.py for the projection lemma)
  1. rank sweep + six-arm ablation on P11 + R1-R5, ranks 2..n, closed-form and TD-trained
     (5 seeds x 2 learning rates, 50k steps, bootstrapped target, EMA 0.005);
  2. rarity sweep on P6 at ranks 2 and 3: the two bottleneck rows carry weight w in
     {1/2, 1/4, 1/6 (= uniform), 1/8, 1/16, ..., 1/128, 0}; arms A, C, D closed form, TD-trained
     for w >= 1/128;
  3. lemma check on P11 + R1-R5: projection prediction from the trained B vs FB's readout at every
     rank and run; residual direction appended; reward sweep at rank 2;
  4. mitigation measurements, closed form on P11 + R1-R5 at ranks 2..5: readout swap (FB's SVD
     features with per-row maximum-likelihood softmax rows; Density-FB's features with per-row
     least-squares linear rows), the linear-separability margin of cluster A in feature space,
     and the moment identity B^T (row_fit - row_true) = 0 of both per-row optima.

Run (CPU, ~40 min):
    JAX_PLATFORMS=cpu python analysis/single_bottleneck/run.py
    ... --figures-only      (rebuild figures + tables from the CSVs)
    ... --smoke             (tiny budget, for a quick check)
Outputs: analysis/single_bottleneck/sb_*.csv, sb_processes.json, sb_summary.json, sb_log.txt,
         figures/single_bottleneck/fig_sb_*.{pdf,png}.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import csv
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "didactic"))
sys.path.insert(0, str(REPO / "analysis" / "toy_ablation"))
sys.path.insert(0, str(REPO / "analysis" / "lemma_check"))
import bottleneck_family_td as bf          # noqa: E402  (sets jax x64)
import family as fam                       # noqa: E402  (arms, fits, TD harness, figure style)
import design_single_bottleneck as dsb     # noqa: E402  (the process builder)


def _load(name, path):
    """Both companion scripts are called run.py; load them under distinct module names."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = _load("lemma_check_run", REPO / "analysis" / "lemma_check" / "run.py")
rb = _load("rare_bottleneck_run", REPO / "analysis" / "rare_bottleneck" / "run.py")
single = fam.single

FIG_DIR = REPO / "figures" / "single_bottleneck"
GAMMA = 0.8
SMOKE = "--smoke" in sys.argv
SEEDS = [0] if SMOKE else list(fam.SEEDS)
RATES = [0.01] if SMOKE else list(fam.RATES)
TRAIN_STEPS = 300 if SMOKE else fam.TRAIN_STEPS
N_RANDOM = 1 if SMOKE else 5
TEMPLATE = dict(entry=[2], cluster=[2], tail=[4])   # 2026-09-20: the eleven-state template only (paper decision)
MIN_GAP, MAX_LIN2 = 0.5, 0.5                  # keep a draw iff exact gap > 0.5 and FB rank-2 fraction < 0.5
W_UNIFORM = 1.0 / 6.0                          # P6 has 6 states: uniform row weight = 1/n on the bottleneck pair
W_SWEEP = [1 / 2, 1 / 6, 1 / 64, 0.0] if SMOKE else [1 / 2, 1 / 4, 1 / 6, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 0.0]
RARITY_RANKS = [2, 3]
MIT_RANKS = [2, 3] if SMOKE else [2, 3, 4, 5]
REWARD_RANK = 2
N_RANDOM_REWARDS = 5 if SMOKE else 50
ID_P11, ID_P6 = 0, 99
LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def ranks_of(mdp):
    return [2, 3] if SMOKE else list(range(2, mdp["n"] + 1))


# ----------------------------------------------------------------------------
# 1. Processes.
# ----------------------------------------------------------------------------
def finish(pid, name, spec, edges, pi, b, A, B, T, attempts=1):
    """The family.py process dict (so its fits, readouts and metrics apply unchanged), with the
    reward = indicator of cluster A, the decision at b, and the rare_bottleneck keys added."""
    n = len(edges)
    E = list(range(b))
    kernel = np.eye(n)[edges].reshape(-1, n)                 # (2n, n), row 2s+a
    t_exact = bf.exact(kernel, pi, GAMMA)                    # (1-gamma) M
    m_exact = t_exact / (1 - GAMMA)
    r = np.isin(np.arange(n), A).astype(float)
    q = m_exact @ r
    gap = float(q[2 * b] - q[2 * b + 1])
    dist = fam.graph_distance(edges, n)
    k_soft = np.exp(-dist ** 2 / (2 * fam.SOFT_SIGMA ** 2))
    k_soft = k_soft / k_soft.sum(axis=1, keepdims=True)
    groups = [E, list(A), list(B)] + ([list(T)] if T else [])
    return dict(seed=pid, variant=name, n=n, edges=np.asarray(edges), pi=np.asarray(pi), cuts=[b],
                groups=groups, goal=int(A[0]), decision=b, kernel=kernel, t_exact=t_exact,
                m_exact=m_exact, r=r, q_exact=q, gap=gap, k_soft=k_soft, t_soft=t_exact @ k_soft,
                rho=np.full(n, 1.0 / n), attempts=attempts, spec=spec, entry=E, clusterA=list(A),
                clusterB=list(B), tail=list(T),
                # keys the rare_bottleneck functions read
                N=0, T=t_exact, M=m_exact, X=n * m_exact, Q=q, rows=(2 * b, 2 * b + 1))


def fixed_processes():
    out = []
    for pid, name, kw in ((ID_P6, "P6", dict(entry=1, cluster=2, tail=0, mutual=True)),
                          (ID_P11, "P11", dict(entry=2, cluster=2, tail=4, merge=True))):
        edges, b, A, B, T = dsb.build(**kw)
        pi = np.full((len(edges), 2), 0.5)
        out.append(finish(pid, name, kw, edges, pi, b, A, B, T))
    return out


def linear_fraction(mdp, d):
    """FB's closed-form rank-d fit: fraction of the exact gap it reads (projection lemma)."""
    _, B = fam.svd_fit(mdp["X"], d)
    return lc.lemma_gap(mdp, B, mdp["r"]) / mdp["gap"]


def draw_random(k):
    rng = np.random.default_rng(2000 + k)
    rejected = []
    for attempt in range(1000):
        spec = dict(entry=int(rng.choice(TEMPLATE["entry"])), cluster=int(rng.choice(TEMPLATE["cluster"])),
                    tail=int(rng.choice(TEMPLATE["tail"])))
        spec.update(merge=spec["tail"] > 0, mutual=spec["tail"] == 0)
        edges, b, A, B, T = dsb.build(**spec)
        n = len(edges)
        p0 = rng.uniform(0.25, 0.75, size=n)
        pi = np.stack([p0, 1.0 - p0], axis=1)
        mdp = finish(k, f"R{k}", spec, edges, pi, b, A, B, T, attempt + 1)
        lin2 = linear_fraction(mdp, 2)
        if mdp["gap"] > MIN_GAP and lin2 < MAX_LIN2:
            mdp["rejected"] = rejected
            mdp["lin2_at_draw"] = float(lin2)
            return mdp
        rejected.append(dict(spec={k_: v for k_, v in spec.items() if k_ in ("entry", "cluster", "tail")},
                             gap=round(mdp["gap"], 3), lin2=round(float(lin2), 3)))
    raise RuntimeError("no valid draw")


def check(mdp):
    """Structure and exact quantities the builder promises."""
    n, b, E, A, B, T = mdp["n"], mdp["decision"], mdp["entry"], mdp["clusterA"], mdp["clusterB"], mdp["tail"]
    edges = mdp["edges"]
    assert float(np.max(np.abs(mdp["t_exact"].sum(1) - 1))) < 1e-10          # rows of (1-gamma)M sum to 1
    assert np.all(mdp["t_exact"] >= -1e-12)
    assert abs(mdp["gap"] - (mdp["q_exact"][2 * b] - mdp["q_exact"][2 * b + 1])) < 1e-12
    assert int(edges[b, 0]) == A[0] and int(edges[b, 1]) == B[0]
    inside = set(A) | set(B) | set(T)
    into = [(s, a) for s in range(n) for a in (0, 1) if s not in inside and int(edges[s, a]) in inside]
    assert sorted(into) == [(b, 0), (b, 1)], into                             # b is the only way in
    assert all(int(t) in set(E) | {b} for s in E for t in edges[s])           # the entry chain only reaches b
    for s in list(A) + list(B) + list(T):
        assert all(int(t) in inside for t in edges[s]), "cluster/tail leaks back"
    if mdp["spec"].get("mutual"):
        assert all(int(edges[s, 1]) == B[0] for s in A) and all(int(edges[s, 1]) == A[0] for s in B)
    else:
        assert all(int(edges[s, 1]) == T[0] for s in list(A) + list(B))
        assert all(int(t) in T for s in T for t in edges[s])                # tail closed
    if mdp["variant"] == "P6":
        assert abs(mdp["gap"] - 1.0) < 1e-9 and abs(mdp["q_exact"][2] - 3.0) < 1e-9 and abs(mdp["q_exact"][3] - 2.0) < 1e-9
    if mdp["variant"] == "P11":
        assert abs(mdp["gap"] - 5.0 / 3.0) < 1e-9 and abs(mdp["q_exact"][5]) < 1e-12


def describe(mdps):
    log("=== processes ===")
    out = []
    for m in mdps:
        check(m)
        sp = m["spec"]
        log(f"{m['variant']:4s} n={m['n']:2d} entry {m['entry']} b={m['decision']} A {m['clusterA']} B {m['clusterB']} "
            f"tail {m['tail']} ({'mutual leak' if sp.get('mutual') else 'shared tail'}); exact values at b "
            f"{m['q_exact'][2 * m['decision']]:.3f} / {m['q_exact'][2 * m['decision'] + 1]:.3f}, gap {m['gap']:.3f}; "
            f"p(a0) in [{m['pi'][:, 0].min():.2f}, {m['pi'][:, 0].max():.2f}]; FB closed-form fraction at d=2..4: "
            + " ".join(f"{linear_fraction(m, d):+.3f}" for d in (2, 3, 4)) + f"; {m['attempts']} draw(s)")
        out.append(dict(id=int(m["seed"]), name=m["variant"], n=int(m["n"]),
                        spec={k: (bool(v) if isinstance(v, bool) else int(v)) for k, v in sp.items()},
                        edges=m["edges"].tolist(), pi=m["pi"].tolist(), entry=m["entry"], bottleneck=int(m["decision"]),
                        clusterA=m["clusterA"], clusterB=m["clusterB"], tail=m["tail"], gap=m["gap"],
                        q_exact=m["q_exact"].tolist(), attempts=int(m["attempts"]),
                        rejected=m.get("rejected", []), lin2_at_draw=m.get("lin2_at_draw")))
    (HERE / "sb_processes.json").write_text(json.dumps(
        dict(gamma=GAMMA, seeds=SEEDS, rates=RATES, steps=TRAIN_STEPS, tau=fam.TRAIN_TAU,
             closed_steps=fam.CLOSED_STEPS, closed_lr=fam.CLOSED_LR, soft_sigma=fam.SOFT_SIGMA,
             template=TEMPLATE, min_gap=MIN_GAP, max_lin2=MAX_LIN2, w_sweep=W_SWEEP, w_uniform=W_UNIFORM,
             rarity_ranks=RARITY_RANKS, mitigation_ranks=MIT_RANKS, reward_rank=REWARD_RANK,
             processes=out), indent=2))
    log("")


# ----------------------------------------------------------------------------
# 2. Rank sweep + lemma check (one training pass; FB's trained B feeds the lemma rows).
# ----------------------------------------------------------------------------
def run_closed(mdps):
    """The six arms in closed form (family.py) plus the closed-form lemma / constructive rows."""
    closed = fam.run_closed_form(mdps, ranks=ranks_of)
    lemma_rows, construct_rows = [], []
    for mdp in mdps:
        X, r, n = mdp["X"], mdp["r"], mdp["n"]
        for d in ranks_of(mdp):
            Fs, Bs = fam.svd_fit(X, d)
            g_svd, g_lem = lc.trained_gap(mdp, Fs, Bs, r), lc.lemma_gap(mdp, Bs, r)
            assert abs(g_svd - g_lem) < 1e-9, (mdp["variant"], d, g_svd, g_lem)
            assert abs(g_lem + lc.discarded_gap(mdp, Bs, r) - mdp["gap"]) < 1e-9
            a_row = [x for x in closed if x["arm"] == "A" and x["mdp"] == mdp["seed"] and x["d"] == d][0]
            assert abs(a_row["gap_ratio"] - g_lem / mdp["gap"]) < 1e-9          # arm A == the lemma's projection
            lemma_rows.append(dict(mdp=mdp["seed"], d=d, kind="closed", seed=-1, lr=0.0,
                                   lemma=g_lem / mdp["gap"], read=g_svd / mdp["gap"],
                                   discarded=lc.discarded_gap(mdp, Bs, r) / mdp["gap"]))
            Ba = lc.augmented_features(mdp, Bs)
            g_aug = lc.lemma_gap(mdp, Ba, r)
            assert abs(g_aug - mdp["gap"]) < 1e-9, (mdp["variant"], d, g_aug)
            Fs1, Bs1 = fam.svd_fit(X, min(d + 1, n))
            construct_rows.append(dict(mdp=mdp["seed"], d=d, gap_rank_d=g_lem / mdp["gap"], gap_aug=g_aug / mdp["gap"],
                                       obj_rank_d=lc.objective(mdp, Bs), obj_aug=lc.objective(mdp, Ba),
                                       obj_rank_d1=lc.objective(mdp, Bs1), aug_cols=Ba.shape[1]))
    return closed, lemma_rows, construct_rows


def run_trained(mdps):
    """family.run_trained's loop, with FB's trained (F, B) also fed to the lemma check."""
    abl, lemma_rows, reward_rows = [], [], []
    t0 = time.time()
    for mdp in mdps:
        k, n, r = mdp["seed"], mdp["n"], mdp["r"]
        for d in ranks_of(mdp):
            for method in fam.TRAIN_METHODS:
                cfg, pred, F_all, B_all = fam.train_group(mdp, d, method, SEEDS, RATES, steps=TRAIN_STEPS)
                for i, c in enumerate(cfg):
                    P = bf.readout(pred[i], fam._base_method(method), GAMMA, n)
                    base = dict(mdp=k, d=d, seed=c["seed"], lr=c["lr"])
                    abl.append(dict(arm=fam.ARM_OF[method], method=method, **base,
                                    **fam.gap_stats(mdp, P @ r / (1 - GAMMA)),
                                    retained_mass=fam.retained_mass(mdp, P)))
                    if method == "density":
                        abl.append(fam.e_arm_rows(mdp, F_all[i], B_all[i], dict(base, retained_mass=fam.retained_mass(mdp, P))))
                    if method == "fb":
                        F, B = F_all[i], B_all[i]
                        g_read, g_lem = lc.trained_gap(mdp, F, B, r), lc.lemma_gap(mdp, B, r)
                        assert abs(g_read - abl[-1]["gap"]) < 1e-9                  # same readout as arm A
                        lemma_rows.append(dict(mdp=k, d=d, kind="trained", seed=c["seed"], lr=c["lr"],
                                               lemma=g_lem / mdp["gap"], read=g_read / mdp["gap"],
                                               discarded=lc.discarded_gap(mdp, B, r) / mdp["gap"]))
                        Ba = lc.augmented_features(mdp, B)
                        assert abs(lc.lemma_gap(mdp, Ba, r) - mdp["gap"]) < 1e-9
                        if d == REWARD_RANK:
                            rng = np.random.default_rng(1000 * k + 10 * c["seed"] + int(c["lr"] * 1000))
                            rewards = [("unit", j, np.eye(n)[j]) for j in range(n)]
                            rewards += [("random", j, rng.uniform(0, 1, n)) for j in range(N_RANDOM_REWARDS)]
                            for kind, j, rr in rewards:
                                reward_rows.append(dict(mdp=k, seed=c["seed"], lr=c["lr"], kind=kind, k=j,
                                                        true=lc.true_gap(mdp, rr), lemma=lc.lemma_gap(mdp, B, rr),
                                                        read=lc.trained_gap(mdp, F, B, rr), aug=lc.lemma_gap(mdp, Ba, rr)))
            means = {arm: np.mean([x["gap_ratio"] for x in abl if x["mdp"] == k and x["d"] == d and x["arm"] == arm])
                     for arm in ("A", "C", "D")}
            log(f"  [trained] {mdp['variant']} d={d}: A {means['A']:.3f} C {means['C']:.3f} D {means['D']:.3f} "
                f"({time.time() - t0:.0f}s)")
    return abl, lemma_rows, reward_rows


# ----------------------------------------------------------------------------
# 3. Rarity sweep on P6 (rare_bottleneck functions with the process's own rows).
# ----------------------------------------------------------------------------
def run_rarity(p6):
    closed, trained = [], []
    t0 = time.time()
    for w in W_SWEEP:
        wts = rb.row_weights(p6, w)
        assert abs(wts.sum() - 1) < 1e-12
        if abs(w - W_UNIFORM) < 1e-12:
            assert np.allclose(wts, 1.0 / (2 * p6["n"]))
        for d in RARITY_RANKS:
            a = rb.run_arm_A(p6, wts, d, "reweight", w)
            assert abs(a[1]["gap_ratio"] - 1.0) < 1e-8                       # A+ keeps the gap by construction
            if w == 0.0:
                assert abs(a[0]["gap_ratio"] - a[0]["limit_gap_ratio"]) < 1e-8   # weighted SVD at w=0 == the limit formula
            closed += a
            closed += rb.run_arms_CD(p6, wts, d, "reweight", w)
        g = lambda arm, d: np.mean([x["gap_ratio"] for x in closed if x["w"] == w and x["d"] == d and x["arm"] == arm])
        log(f"  [rarity closed] w={w:.4f}: " + "  ".join(f"d={d}: A {g('A', d):.3f} C {g('C', d):.3f} D {g('D', d):.3f}"
                                                        for d in RARITY_RANKS) + f"  ({time.time() - t0:.0f}s)")
    for w in W_SWEEP:
        if w == 0.0:
            continue                                                        # TD needs every row to receive gradient
        wts = rb.row_weights(p6, w)
        for d in RARITY_RANKS:
            trained += rb.run_trained(p6, wts, "reweight", w, d=d)
        g = lambda arm, d: np.mean([x["gap_ratio"] for x in trained if x["w"] == w and x["d"] == d and x["arm"] == arm])
        log(f"  [rarity trained] w={w:.4f}: " + "  ".join(f"d={d}: A {g('A', d):.3f} C {g('C', d):.3f} D {g('D', d):.3f}"
                                                         for d in RARITY_RANKS) + f"  ({time.time() - t0:.0f}s)")
    return closed, trained


# ----------------------------------------------------------------------------
# 4. Mitigation measurements (closed form).
# ----------------------------------------------------------------------------
def orthonormal(B):
    """A canonical basis of span(B): the fit F B^T only depends on the span, so all feature-space
    quantities below are computed on this representative."""
    Q, _ = np.linalg.qr(B)
    return Q


def ls_rows(mdp, B):
    """Per-row least squares of X = nM given orthonormal B: F = X B, fit = X P_B."""
    F = mdp["X"] @ B
    return F, F @ B.T


def ml_rows(B, T):
    """Per-row maximum likelihood of the target rows T in the softmax family {softmax(B f)}: each
    row's cross-entropy is convex in f, solved by L-BFGS with the analytic gradient B^T (p - t)."""
    from scipy.optimize import minimize
    n, d = B.shape
    F = np.zeros((T.shape[0], d))
    P = np.zeros_like(T)

    def loss_grad(f, t):
        z = B @ f
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        return -float(t @ np.log(np.maximum(p, 1e-300))), B.T @ (p - t)

    for i in range(T.shape[0]):
        res = minimize(loss_grad, np.zeros(d), args=(T[i],), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=5000, gtol=1e-12, ftol=1e-15))
        F[i] = res.x
        z = B @ res.x
        p = np.exp(z - z.max())
        P[i] = p / p.sum()
    return F, P


def separability_margin(B, A_idx, n_dirs=20000, seed=0):
    """max over unit u of  min_{a in A} b_a.u - max_{s not in A} b_s.u  on the rows of B.
    Random search on the sphere followed by SLSQP refinement (||u|| = 1) from the best directions;
    positive iff cluster A is linearly separable from the other states in feature space."""
    from scipy.optimize import minimize
    n, d = B.shape
    inA = np.zeros(n, bool)
    inA[list(A_idx)] = True
    BA, BR = B[inA], B[~inA]

    def value(u):
        u = u / max(np.linalg.norm(u), 1e-300)
        return float((BA @ u).min() - (BR @ u).max())

    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_dirs, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    vals = (BA @ U.T).min(axis=0) - (BR @ U.T).max(axis=0)
    order = np.argsort(vals)[::-1]
    best_val, best_u = float(vals[order[0]]), U[order[0]]
    for j in order[:8]:
        u0 = U[j]
        x0 = np.concatenate([u0, [(BA @ u0).min(), (BR @ u0).max()]])
        cons = [dict(type="ineq", fun=lambda x: BA @ x[:d] - x[d]),
                dict(type="ineq", fun=lambda x: x[d + 1] - BR @ x[:d]),
                dict(type="eq", fun=lambda x: x[:d] @ x[:d] - 1.0)]
        res = minimize(lambda x: -(x[d] - x[d + 1]), x0, constraints=cons, method="SLSQP",
                       options=dict(maxiter=500, ftol=1e-12))
        v = value(res.x[:d])
        if v > best_val:
            best_val, best_u = v, res.x[:d] / np.linalg.norm(res.x[:d])
    return best_val, best_u


def run_mitigation(mdps):
    rows = []
    t0 = time.time()
    for mdp in mdps:
        k, n, r, X, T = mdp["seed"], mdp["n"], mdp["r"], mdp["X"], mdp["t_exact"]
        for d in MIT_RANKS:
            feats = [("fb", -1, fam.svd_fit(X, d)[1], None)]
            for sd in SEEDS:
                sseed = 1000 * sd + 17 * d + 101 * k                     # the same fits as arm D's closed form
                F, B, P, _ = fam.softmax_fit(T, n, d, sseed, "ce")
                feats.append(("density", sd, B, P))
            for fname, sd, B_raw, P_joint in feats:
                B = orthonormal(B_raw)
                F_lin, fit = ls_rows(mdp, B)
                gap_lin = fam.gap_stats(mdp, fit @ r / n)
                mom_lin = float(np.max(np.abs(B.T @ (fit - X).T)))
                F_sm, P = ml_rows(B, T)
                gap_sm = fam.gap_stats(mdp, P @ r / (1 - GAMMA))
                mom_sm = float(np.max(np.abs(B.T @ (P - T).T)))
                mg, _ = separability_margin(B, mdp["clusterA"])
                row = dict(mdp=k, d=d, features=fname, seed=sd, margin=mg,
                           gap_linear=gap_lin["gap_ratio"], gap_softmax=gap_sm["gap_ratio"],
                           argmax_linear=gap_lin["argmax_correct"], argmax_softmax=gap_sm["argmax_correct"],
                           moment_linear=mom_lin, moment_softmax=mom_sm,
                           retained_linear=fam.retained_mass(mdp, (1 - GAMMA) * fit / n),
                           retained_softmax=fam.retained_mass(mdp, P),
                           ce_softmax=float(-np.mean(np.sum(T * np.log(np.maximum(P, 1e-300)), axis=1))))
                if P_joint is not None:                                     # arm D's joint Adam fit, before the per-row polish
                    row["gap_softmax_joint"] = fam.gap_stats(mdp, P_joint @ r / (1 - GAMMA))["gap_ratio"]
                    row["ce_softmax_joint"] = float(-np.mean(np.sum(T * np.log(np.maximum(P_joint, 1e-300)), axis=1)))
                    assert row["ce_softmax"] <= row["ce_softmax_joint"] + 1e-9   # the per-row optimum is at least as good
                rows.append(row)
        log(f"  [mitigation] {mdp['variant']} done ({time.time() - t0:.0f}s)")
    return rows


# ----------------------------------------------------------------------------
# 5. Figures.
# ----------------------------------------------------------------------------
COL_B = "#0173B2"          # cluster B: blue; bottleneck orange and cluster A green as in family.py
ARM_COLOR, ARM_MARKER = single.ARM_COLOR, single.ARM_MARKER
INK = single.INK


def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK,
                         "axes.labelcolor": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})
    return plt


def save(fig, stem):
    fam.save(fig, stem, FIG_DIR)


def sb_layout(mdp):
    """Entry chain on the left, the bottleneck in the middle, cluster A above and cluster B below
    on the right, the shared tail (if any) at the far right."""
    E, A, B, T, b = mdp["entry"], mdp["clusterA"], mdp["clusterB"], mdp["tail"], mdp["decision"]
    pos, out = {}, {}
    xs = np.linspace(-2.3, -1.1, len(E)) if len(E) > 1 else [-1.7]
    for s, x in zip(E, xs):
        pos[s], out[s] = np.array([x, 0.0]), np.array([0.0, -1.0])
    pos[b], out[b] = np.array([-0.35, 0.0]), np.array([0.0, -1.0])
    for C, cy in ((A, 0.78), (B, -0.78)):
        c = np.array([0.75, cy])
        rad = 0.34 if len(C) > 2 else 0.3
        for i, s in enumerate(C):
            ang = np.pi + np.pi * i / max(len(C) - 1, 1) if len(C) == 2 else np.pi / 2 - 2 * np.pi * i / len(C)
            pos[s] = c + rad * np.array([np.cos(ang), np.sin(ang)])
            out[s] = (pos[s] - c) / max(np.linalg.norm(pos[s] - c), 1e-9)
            if len(C) == 2:
                out[s] = np.array([0.0, np.sign(cy)])
    c = np.array([2.05, 0.0])
    for i, s in enumerate(T):
        ang = np.pi / 2 - 2 * np.pi * i / max(len(T), 1)
        pos[s] = c + 0.42 * np.array([np.cos(ang), np.sin(ang)])
        out[s] = (pos[s] - c) / max(np.linalg.norm(pos[s] - c), 1e-9)
    return pos, out


def sb_node_style(mdp, s):
    if s == mdp["decision"]:
        return "#DE8F05", "#DE8F05", "white"
    if s in mdp["clusterA"]:
        return "#029E73", "#029E73", "white"
    if s in mdp["clusterB"]:
        return COL_B, COL_B, "white"
    return "white", "#9a9a9a", INK


def draw_graphs(mdps):
    for m in mdps:
        title = (f"{m['variant']}: entry {m['entry']}, bottleneck {m['decision']}, A {m['clusterA']}, "
                 f"B {m['clusterB']}, tail {m['tail'] or '-'}, exact gap {m['gap']:.2f}")
        fam.draw_graph(m, layout=sb_layout, node_style=sb_node_style, stem=f"fig_sb_graph_{m['variant']}",
                       title=title, fig_dir=FIG_DIR, figsize=(3.8, 2.0), xlim=(-2.7, 2.7))


def common_ranks(mdps):
    return list(range(2, min(m["n"] for m in mdps) + 1)) if not SMOKE else [2, 3]


def make_ablation_figures(closed, trained, mdps):
    plt = _rc()
    from matplotlib.lines import Line2D
    ranks = common_ranks(mdps)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True)
    fam.draw_panel(axes[0], closed, hollow=False, ranks=ranks)
    fam.draw_panel(axes[1], trained, hollow=True, ranks=ranks)
    axes[0].set_ylabel("$\\widehat{G}_r\,/\,G_r$", fontsize=9)
    axes[1].set_ylabel("")
    fig.legend(handles=single.legend_handles(), loc="lower center", ncol=2, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, -0.30), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig_sb_ablation")
    plt.close(fig)
    for stem, rows, hollow in (("fig_sb_ablation_closed", closed, False), ("fig_sb_ablation_trained", trained, True)):
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        fam.draw_panel(ax, [r for r in rows if r["arm"] != "E"], hollow=hollow, ranks=ranks)   # E has no meaningful scale
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlim(ranks[0] - 0.25, ranks[-1] + 0.25); ax.set_xticks(ranks)
        ax.set_ylabel("recovery", fontsize=9)
        short = {"A": "FB", "B": "FB, normalized target", "C": "softmax + squared error", "D": "Density-FB",
                 "F": "Density-FB, soft target"}
        hs = [Line2D([0], [0], color=single.ARM_COLOR[k], marker=single.ARM_MARKER[k], markersize=4, linewidth=1.4,
                     markerfacecolor="none" if hollow else single.ARM_COLOR[k], label=v) for k, v in short.items()]
        hs.append(Line2D([0], [0], color=single.EXACT_COLOR, linestyle="--", linewidth=1.0, label="exact"))
        ax.legend(handles=hs, loc="lower right", fontsize=6.4, frameon=False, handlelength=1.6, labelspacing=0.3)
        fig.tight_layout(pad=0.3)
        save(fig, stem)
        plt.close(fig)
    # FB vs Density-FB only, TD-trained
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axhline(1.0, color=single.EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    for arm in ("A", "D"):
        a = fam.agg(trained, arm, ranks=ranks)
        y, lo, hi = (np.array([a[d][i] for d in ranks]) for i in range(3))
        ax.fill_between(ranks, lo, hi, color=ARM_COLOR[arm], alpha=0.16, linewidth=0, zorder=2)
        ax.plot(ranks, y, color=ARM_COLOR[arm], marker=ARM_MARKER[arm], markersize=4.2, markerfacecolor="none",
                markeredgecolor=ARM_COLOR[arm], markeredgewidth=1.1, linewidth=1.6, zorder=3)
    single._style_axis(ax, "$\\widehat{G}_r\,/\,G_r$")
    ax.set_xticks(ranks); ax.set_xlim(ranks[0] - 0.25, ranks[-1] + 0.25)
    ax.set_ylim(-0.1, 1.15)
    ax.legend(handles=[Line2D([0], [0], color=ARM_COLOR["A"], marker="o", markerfacecolor="none", markersize=4.2, linewidth=1.6, label="FB"),
                       Line2D([0], [0], color=ARM_COLOR["D"], marker="D", markerfacecolor="none", markersize=4.2, linewidth=1.6, label="Density-FB"),
                       Line2D([0], [0], color=single.EXACT_COLOR, linestyle="--", linewidth=1.0, label="exact gap")],
              loc="lower right", fontsize=7.4, frameon=False, handlelength=1.6, labelspacing=0.3, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_sb_gap_two_arm")
    plt.close(fig)
    # per-process panels, three to a row, each up to its own n
    ncol = 3
    nrow = int(np.ceil(len(mdps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.35 * nrow), sharey=True, squeeze=False)
    for ax, m in zip(axes.ravel(), mdps):
        sub = [x for x in trained if int(x["mdp"]) == m["seed"]]
        fam.draw_panel(ax, sub, hollow=True, ranks=ranks_of(m))
        ax.set_title(f"{m['variant']} (n={m['n']}, gap {m['gap']:.2f})", fontsize=8)
        ax.set_ylabel("")
    for ax in axes.ravel()[len(mdps):]:
        ax.axis("off")
    for i in range(nrow):
        axes[i, 0].set_ylabel("$\\widehat{G}_r\,/\,G_r$", fontsize=8.5)
    fig.legend(handles=single.legend_handles(), loc="lower center", ncol=4, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, -0.12 / nrow), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=1.2)
    save(fig, "fig_sb_per_mdp")
    plt.close(fig)
    # values at the bottleneck at ranks 2 and 3
    for d in (2, 3):
        fam.make_values_figure(trained, mdps, d=d, stem=f"fig_sb_values_d{d}", fig_dir=FIG_DIR,
                               action_labels=("action into cluster A", "action into cluster B"))


def make_rarity_figure(closed, trained):
    plt = _rc()
    from matplotlib.lines import Line2D
    for d in RARITY_RANKS:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        rb.draw_panel_gap(ax, closed, trained, d=d, w_sweep=W_SWEEP, uniform=W_UNIFORM)
        ax.set_ylim(-0.25, 1.15)
        ax2 = ax.twinx()
        share = [rb.agg(closed, "bn_share_target", construction="reweight", w=float(w), d=d, arm="A")[0] for w in W_SWEEP]
        ax2.plot([rb.xpos(w) for w in W_SWEEP], share, color="#949494", linestyle=":", linewidth=1.2, zorder=2)
        ax2.set_ylabel("bottleneck share of the squared mass", fontsize=7.6, color="#6a6a6a")
        ax2.tick_params(labelsize=7.6, length=2.5, colors="#6a6a6a")
        ax2.set_ylim(bottom=-0.01)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_color("#9a9a9a")
        ax.legend(handles=rb.legend_handles(with_share=True)[:4] +
                  [Line2D([0], [0], color="#949494", linestyle=":", linewidth=1.2, label="bottleneck share of the squared mass")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6.0, frameon=False,
                  handlelength=1.6, labelspacing=0.35, columnspacing=1.0)
        ax.text(0.03, 0.95, f"P6, rank {d}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
        fig.tight_layout(pad=0.3)
        save(fig, f"fig_sb_rare_lemma_d{d}")
        plt.close(fig)


def make_lemma_figure(lemma_rows, reward_rows, construct_rows, mdps):
    plt = _rc()
    from matplotlib.lines import Line2D
    ranks = sorted({int(x["d"]) for x in lemma_rows})
    cmap = plt.get_cmap("viridis")
    rank_color = {d: cmap((d - ranks[0]) / max(ranks[-1] - ranks[0], 1)) for d in ranks}
    names = {m["seed"]: m["variant"] for m in mdps}

    def panel_a(ax):
        tr = [x for x in lemma_rows if x["kind"] == "trained"]
        for d in ranks:
            ax.scatter([x["lemma"] for x in tr if int(x["d"]) == d], [x["read"] for x in tr if int(x["d"]) == d],
                       s=11, color=rank_color[d], alpha=0.75, linewidths=0, zorder=3)
        cl = [x for x in lemma_rows if x["kind"] == "closed"]
        ax.scatter([x["lemma"] for x in cl], [x["read"] for x in cl], s=26, facecolors="none", edgecolors=INK,
                   linewidths=0.7, zorder=4)
        lo, hi = -0.35, 1.25
        ax.plot([lo, hi], [lo, hi], color="#9a9a9a", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("$\\widehat{G}_r\,/\,G_r$ allowed by the features", fontsize=8)
        ax.set_ylabel("$\\widehat{G}_r\,/\,G_r$ read by the trained critic", fontsize=8)
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=ranks[0], vmax=ranks[-1]))
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, ticks=[ranks[0], (ranks[0] + ranks[-1]) // 2, ranks[-1]])
        cb.set_label("representation rank", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2)
        cb.outline.set_visible(False)
        ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor="none", markeredgecolor=INK, label="closed-form fit"),
                           Line2D([0], [0], marker="o", linestyle="", markersize=4, color=rank_color[ranks[2]], label="TD-trained fit")],
                  fontsize=6.4, frameon=False, loc="upper left", handlelength=1.0, labelspacing=0.3)

    def panel_b(ax):
        units = [x for x in reward_rows if x["kind"] == "unit"]
        rands = [x for x in reward_rows if x["kind"] == "random"]
        ax.scatter([x["lemma"] for x in rands], [x["read"] for x in rands], s=6, color="#b0b0b0", alpha=0.5, linewidths=0, zorder=2, label="random rewards")
        pcol = plt.get_cmap("tab10")
        for i, m in enumerate(sorted({int(x["mdp"]) for x in units})):
            u = [x for x in units if int(x["mdp"]) == m]
            ax.scatter([x["lemma"] for x in u], [x["read"] for x in u], s=13, color=pcol(i), alpha=0.8, linewidths=0, zorder=3,
                       label=f"unit reward, {names[m]}")
        allv = [x["lemma"] for x in reward_rows] + [x["read"] for x in reward_rows]
        lo, hi = min(allv) - 0.1, max(allv) + 0.1
        ax.plot([lo, hi], [lo, hi], color="#9a9a9a", linewidth=0.8, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("gap the features allow")
        ax.set_ylabel("gap the trained critic reads")
        ax.legend(fontsize=6.0, frameon=False, loc="upper left", handlelength=1.0, labelspacing=0.3)

    def panel_c(ax):
        ds = sorted({int(x["d"]) for x in construct_rows})
        def mean_over(key, d):
            return float(np.mean([x[key] for x in construct_rows if int(x["d"]) == d]))
        def rel_cost(d):
            return float(np.mean([x["obj_aug"] / x["obj_rank_d1"] - 1.0 if x["obj_rank_d1"] > 1e-12 else 0.0
                                  for x in construct_rows if int(x["d"]) == d]))
        ax.plot(ds, [mean_over("gap_rank_d", d) for d in ds], marker="o", markersize=4, color=ARM_COLOR["A"], label="FB, optimal rank $d$")
        ax.plot(ds, [mean_over("gap_aug", d) for d in ds], marker="s", markersize=4, color=ARM_COLOR["D"], label="rank $d$ + residual direction  ($d{+}1$)")
        ax.axhline(1.0, color="#9a9a9a", linewidth=0.7, linestyle="--")
        ax.set_ylim(-0.1, 1.15)
        ax.set_xlabel("representation rank")
        ax.set_ylabel("$\\widehat{G}_r\,/\,G_r$")
        ax2 = ax.twinx()
        ax2.plot(ds, [rel_cost(d) for d in ds], color="#6a6a6a", linestyle=":", linewidth=1.2, marker="^", markersize=3.5,
                 label="objective above the optimal rank $d{+}1$ fit")
        ax2.set_ylabel("relative objective cost", color="#6a6a6a", fontsize=7.6)
        ax2.tick_params(labelsize=7.6, colors="#6a6a6a")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color("#9a9a9a")
        ax2.set_ylim(bottom=-0.02)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, handlelength=1.6)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    panel_a(axes[0]); panel_b(axes[1]); panel_c(axes[2])
    fig.tight_layout(pad=0.4, w_pad=1.6)
    save(fig, "fig_sb_lemma_check"); plt.close(fig)
    for name, fn in (("a", panel_a), ("b", panel_b), ("c", panel_c)):
        fig, ax = plt.subplots(figsize=(3.4, 2.7) if name == "a" else (3.3, 2.8)); fn(ax); fig.tight_layout(pad=0.3)
        save(fig, f"fig_sb_lemma_check_{name}"); plt.close(fig)


def make_mitigation_figures(mit):
    plt = _rc()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    combos = [("fb", "gap_linear", "FB features, linear rows (= FB)", ARM_COLOR["A"], ""),
              ("fb", "gap_softmax", "FB features, softmax rows", ARM_COLOR["A"], "///"),
              ("density", "gap_linear", "Density-FB features, linear rows", ARM_COLOR["D"], ""),
              ("density", "gap_softmax", "Density-FB features, softmax rows (= Density-FB)", ARM_COLOR["D"], "///")]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    width = 0.19
    for j, (feat, key, lab, col, hatch) in enumerate(combos):
        for i, d in enumerate(MIT_RANKS):
            vals = [float(x[key]) for x in mit if x["features"] == feat and int(x["d"]) == d]
            m, lo, hi = single.t_interval(vals)
            x = i + (j - 1.5) * width
            ax.bar(x, m, width=width, color=col, alpha=0.9 if not hatch else 0.55, hatch=hatch, edgecolor=col, linewidth=0.6, zorder=2)
            if hi > lo:
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], ecolor=INK, elinewidth=0.7, capsize=1.5, linestyle="none", zorder=3)
    ax.axhline(1.0, color=single.EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    ax.set_xticks(range(len(MIT_RANKS))); ax.set_xticklabels([str(d) for d in MIT_RANKS])
    ax.set_xlabel("representation rank", fontsize=8.5)
    ax.set_ylabel("$\\widehat{G}_r\,/\,G_r$", fontsize=8.5)
    ax.tick_params(labelsize=8, length=2.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(-0.1, 1.85)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.9 if not h else 0.55, hatch=h, edgecolor=c, label=l) for _, _, l, c, h in combos],
              fontsize=5.8, frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_sb_readout_swap"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for feat, col in (("fb", ARM_COLOR["A"]), ("density", ARM_COLOR["D"])):
        sub = [x for x in mit if x["features"] == feat]
        ax.scatter([float(x["margin"]) for x in sub], [float(x["gap_linear"]) for x in sub], s=16, marker="o",
                   facecolors="none", edgecolors=col, linewidths=0.8, zorder=3)
        ax.scatter([float(x["margin"]) for x in sub], [float(x["gap_softmax"]) for x in sub], s=16, marker="D",
                   color=col, alpha=0.8, linewidths=0, zorder=3)
    ax.axvline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    ax.axhline(1.0, color=single.EXACT_COLOR, linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=1)
    ax.set_xlabel("separability margin of cluster A in feature space", fontsize=8.5)
    ax.set_ylabel("$\\widehat{G}_r\,/\,G_r$", fontsize=8.5)
    ax.tick_params(labelsize=8, length=2.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=ARM_COLOR["A"], label="FB features, linear rows"),
                       Line2D([0], [0], marker="D", linestyle="", color=ARM_COLOR["A"], label="FB features, softmax rows"),
                       Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=ARM_COLOR["D"], label="Density-FB features, linear rows"),
                       Line2D([0], [0], marker="D", linestyle="", color=ARM_COLOR["D"], label="Density-FB features, softmax rows")],
              fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_sb_margin"); plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Tables and summary.
# ----------------------------------------------------------------------------
def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def summarize(mdps, closed, trained, rare_closed, rare_trained, lemma_rows, reward_rows, construct_rows, mit):
    S = {}
    sweep = [m for m in mdps if m["seed"] != ID_P6]
    p6 = [m for m in mdps if m["seed"] == ID_P6][0]
    ranks = common_ranks(sweep)
    names = {m["seed"]: m["variant"] for m in mdps}

    # --- rank sweep
    log("\n=== 1. rank sweep: fraction of the exact gap vs rank, mean over P11 + R1-R5 (ranks all six share) ===")
    log("closed form:  arm | " + " | ".join(f"d={d}" for d in ranks))
    S["rank_sweep"] = {}
    for arm in single.ARMS:
        a = fam.agg(closed, arm, ranks=ranks)
        S["rank_sweep"][f"closed_{arm}"] = {d: a[d][0] for d in ranks}
        log(f"  {arm} | " + " | ".join(f"{a[d][0]:.3f}" for d in ranks))
    log("TD-trained (mean [95% t-interval over processes x runs]):")
    for arm in single.ARMS:
        a = fam.agg(trained, arm, ranks=ranks)
        S["rank_sweep"][f"trained_{arm}"] = {d: a[d][0] for d in ranks}
        log(f"  {arm} | " + " | ".join(f"{a[d][0]:.2f} [{a[d][1]:.2f},{a[d][2]:.2f}]" for d in ranks))
    log("\nper process, TD-trained (arm | ranks 2..n):")
    for m in sweep:
        sub = [x for x in trained if int(x["mdp"]) == m["seed"]]
        log(f"-- {m['variant']} (n={m['n']}, gap {m['gap']:.2f}) --")
        for arm in single.ARMS:
            a = fam.agg(sub, arm, ranks=ranks_of(m))
            log(f"   {arm} | " + " | ".join(f"{a[d][0]:.3f}" for d in ranks_of(m)))
    for kind, rows in (("TD-trained", trained), ("closed form", closed)):
        log(f"\nsmallest rank reaching 90% of the exact gap ({kind}; mean over runs at that rank):")
        log("  arm | " + " | ".join(names[m["seed"]] for m in sweep) + " | median")
        S[f"rank90_{kind.split()[0].lower()}"] = {}
        for arm in single.ARMS:
            need = []
            for m in sweep:
                sub = [x for x in rows if int(x["mdp"]) == m["seed"]]
                a = fam.agg(sub, arm, ranks=ranks_of(m))
                hit = [d for d in ranks_of(m) if a[d][0] >= 0.9]
                need.append(min(hit) if hit else np.nan)
            med = float(np.nanmedian(need)) if not all(np.isnan(x) for x in need) else float("nan")
            S[f"rank90_{kind.split()[0].lower()}"][arm] = dict(per_process={names[m["seed"]]: (None if np.isnan(x) else int(x)) for m, x in zip(sweep, need)}, median=med)
            log(f"  {arm} | " + " | ".join("-" if np.isnan(x) else str(int(x)) for x in need) + f" | {med:.1f}")
    log("\nargmax stays on cluster A (fraction of TD-trained fits):")
    for arm in single.ARMS:
        vals = [np.mean([float(str(x["argmax_correct"]) == "True") for x in trained if x["arm"] == arm and int(x["d"]) == d]) for d in ranks]
        log(f"  {arm} | " + " | ".join(f"{v:.2f}" for v in vals))
    vals = {}
    for d in (2, 3):
        v = fam.values_rows(trained, sweep, d=d)
        vals[d] = {arm: [v[arm][0][0], v[arm][1][0]] for arm in ("A", "D")}
        log(f"values at b, TD-trained rank {d}, exact pair mapped to (1, 0): FB {v['A'][0][0]:.2f} / {v['A'][1][0]:.2f}, "
            f"Density-FB {v['D'][0][0]:.2f} / {v['D'][1][0]:.2f}")
    S["values"] = vals

    # --- rarity
    log(f"\n=== 2. rarity sweep on P6 (gap {p6['gap']:.2f}): fraction of the exact gap vs bottleneck weight w ===")
    S["rarity"] = {}
    for d in RARITY_RANKS:
        log(f"rank {d}: w | A FB | C softmax+sq | D Density-FB | bottleneck share of target mass | A share of objective | "
            "A cost of keeping the gap (abs; rel) | A limit || TD-trained A | C | D")
        S["rarity"][d] = {}
        for w in W_SWEEP:
            a = rb.sel(rare_closed, construction="reweight", w=float(w), d=d, arm="A")[0]
            c = rb.agg(rare_closed, construction="reweight", w=float(w), d=d, arm="C")
            dd = rb.agg(rare_closed, construction="reweight", w=float(w), d=d, arm="D")
            tr = {arm: rb.agg(rare_trained, construction="reweight", w=float(w), d=d, arm=arm) for arm in ("A", "C", "D")} if w > 0 else None
            S["rarity"][d][rb.fmt_w(w, W_UNIFORM)] = dict(A=float(a["gap_ratio"]), C=c[0], D=dd[0],
                                                          share_target=float(a["bn_share_target"]), share_obj=float(a["bn_share_obj"]),
                                                          cost_rel=float(a["cost_preserve_rel"]), limit=float(a["limit_gap_ratio"]),
                                                          trained={k_: v[0] for k_, v in tr.items()} if tr else None)
            line = (f"  {rb.fmt_w(w, W_UNIFORM):14s} | {float(a['gap_ratio']):.3f} | {c[0]:.3f} | {dd[0]:.3f} | "
                    f"{float(a['bn_share_target']):.4f} | {float(a['bn_share_obj']):.4f} | {float(a['cost_preserve']):.4f}; "
                    f"{float(a['cost_preserve_rel']):.3f} | {float(a['limit_gap_ratio']):.3f}")
            if tr:
                line += " || " + " | ".join(f"{tr[k_][0]:.3f} [{tr[k_][1]:.2f},{tr[k_][2]:.2f}]" for k_ in ("A", "C", "D"))
            log(line)
        # crossover: the largest w at which FB's closed-form fraction is below 0.5
        ws = [w for w in W_SWEEP if w > 0]
        fr = {w: float(rb.sel(rare_closed, construction="reweight", w=float(w), d=d, arm="A")[0]["gap_ratio"]) for w in ws}
        below = [w for w in ws if fr[w] < 0.5]
        S["rarity"][d]["fb_crossover_w"] = max(below) if below else None
        log(f"  FB closed form at rank {d} reads < 0.5 of the gap for every w <= {rb.fmt_w(max(below), W_UNIFORM) if below else 'never'}; "
            f"w -> 0 limit {fr[ws[-1]]:.3f} -> {float(rb.sel(rare_closed, construction='reweight', w=0.0, d=d, arm='A')[0]['limit_gap_ratio']):.3f}")

    # --- lemma check
    log("\n=== 3. lemma check on P11 + R1-R5 ===")
    tr = [x for x in lemma_rows if x["kind"] == "trained"]
    a, b = np.array([x["lemma"] for x in tr]), np.array([x["read"] for x in tr])
    S["lemma"] = dict(n=int(len(a)), corr=_corr(a, b), sign_agree=float(np.mean(np.sign(a) == np.sign(b))),
                      max_abs_diff=float(np.abs(a - b).max()), mean_abs_diff=float(np.abs(a - b).mean()))
    log(f"(a) trained features, {len(a)} runs: corr(lemma, read) {S['lemma']['corr']:.4f}, sign agreement "
        f"{S['lemma']['sign_agree']:.3f}, mean |diff| {S['lemma']['mean_abs_diff']:.4f}, max |diff| {S['lemma']['max_abs_diff']:.4f}")
    all_ranks = sorted({int(x["d"]) for x in tr})
    log("    per rank: d: mean lemma | mean read | max |diff| | corr")
    S["lemma"]["per_rank"] = {}
    for d in all_ranks:
        aa = np.array([x["lemma"] for x in tr if int(x["d"]) == d]); bb = np.array([x["read"] for x in tr if int(x["d"]) == d])
        S["lemma"]["per_rank"][d] = dict(lemma=float(aa.mean()), read=float(bb.mean()), max_abs_diff=float(np.abs(aa - bb).max()), corr=_corr(aa, bb), n=int(len(aa)))
        log(f"    d={d:2d}: {aa.mean():6.3f} | {bb.mean():6.3f} | {np.abs(aa - bb).max():.4f} | {_corr(aa, bb):.4f}  (n={len(aa)})")
    allr, allq = np.array([x["lemma"] for x in reward_rows]), np.array([x["read"] for x in reward_rows])
    S["reward_sweep"] = dict(n=int(len(allr)), corr=_corr(allr, allq), max_abs_diff=float(np.abs(allr - allq).max()),
                             aug_max_err=float(np.max([abs(x["aug"] - x["true"]) for x in reward_rows])))
    log(f"(b) reward sweep at rank {REWARD_RANK}: {len(allr)} (reward, run) pairs, corr {S['reward_sweep']['corr']:.4f}, "
        f"max |lemma - read| {S['reward_sweep']['max_abs_diff']:.4f}; residual appended: max |gap - true| {S['reward_sweep']['aug_max_err']:.1e}")
    units = [x for x in reward_rows if x["kind"] == "unit"]
    for m in sweep:
        line = []
        for j in range(m["n"]):
            u = [x for x in units if int(x["mdp"]) == m["seed"] and int(x["k"]) == j]
            tg = u[0]["true"]
            lem, rd = np.mean([x["lemma"] for x in u]), np.mean([x["read"] for x in u])
            line.append(f"s{j}({tg:+.2f}) {lem / tg:+.2f}|{rd / tg:+.2f}" if abs(tg) > 0.05 else f"s{j}({tg:+.2f}) --")
        log(f"    {m['variant']}: " + "  ".join(line))
    log("(c) closed-form FB: d: gap fraction at rank d | with the residual appended | objective cost vs optimal d+1")
    S["construct"] = {}
    for d in sorted({int(x["d"]) for x in construct_rows}):
        c = [x for x in construct_rows if int(x["d"]) == d]
        g, ga = np.mean([x["gap_rank_d"] for x in c]), np.mean([x["gap_aug"] for x in c])
        cost = np.mean([x["obj_aug"] / x["obj_rank_d1"] - 1 if x["obj_rank_d1"] > 1e-12 else 0.0 for x in c])
        S["construct"][d] = dict(gap_rank_d=float(g), gap_aug=float(ga), rel_cost=float(cost))
        log(f"    d={d:2d}: {g:6.3f} | {ga:6.3f} | {cost:+.3f}")

    # --- mitigation
    log("\n=== 4. mitigation measurements, closed form on P11 + R1-R5, ranks 2..5 ===")
    log("(i) readout swap: fraction of the exact gap (mean over processes [x seeds]); rows = features, columns = readout")
    S["swap"] = {}
    for d in MIT_RANKS:
        S["swap"][d] = {}
        for feat in ("fb", "density"):
            sub = [x for x in mit if x["features"] == feat and int(x["d"]) == d]
            gl, gs = single.t_interval([float(x["gap_linear"]) for x in sub]), single.t_interval([float(x["gap_softmax"]) for x in sub])
            S["swap"][d][feat] = dict(linear=gl[0], softmax=gs[0], n=len(sub))
            extra = ""
            if feat == "density":
                gj = np.mean([float(x["gap_softmax_joint"]) for x in sub])
                S["swap"][d][feat]["softmax_joint"] = float(gj)
                extra = f"  (joint Adam fit before the per-row polish: {gj:.3f})"
            log(f"    d={d} {feat:8s} features: linear rows {gl[0]:.3f} [{gl[1]:.2f},{gl[2]:.2f}] | softmax rows {gs[0]:.3f} [{gs[1]:.2f},{gs[2]:.2f}]{extra}")
        for feat in ("fb", "density"):
            sub = [x for x in mit if x["features"] == feat and int(x["d"]) == d]
            log(f"      {feat} per process, linear|softmax: " + "  ".join(
                f"{names[m['seed']]} {np.mean([float(x['gap_linear']) for x in sub if int(x['mdp']) == m['seed']]):+.2f}|"
                f"{np.mean([float(x['gap_softmax']) for x in sub if int(x['mdp']) == m['seed']]):+.2f}" for m in sweep))
    log("(ii) separability margin of cluster A vs the retained gap, all fits:")
    S["margin"] = {}
    for key, lab in (("gap_softmax", "softmax rows"), ("gap_linear", "linear rows")):
        mg, gp = [float(x["margin"]) for x in mit], [float(x[key]) for x in mit]
        S["margin"][key] = dict(pearson=_corr(mg, gp), spearman=_spearman(mg, gp), n=len(mg))
        log(f"    {lab}: Pearson {S['margin'][key]['pearson']:.3f}, Spearman {S['margin'][key]['spearman']:.3f} (n={len(mg)})")
    for feat in ("fb", "density"):
        sub = [x for x in mit if x["features"] == feat]
        sep = [x for x in sub if float(x["margin"]) > 0]
        S["margin"][f"{feat}_separable_frac"] = len(sep) / len(sub)
        S["margin"][f"{feat}_gap_softmax_when_separable"] = float(np.mean([float(x["gap_softmax"]) for x in sep])) if sep else float("nan")
        S["margin"][f"{feat}_gap_softmax_when_not"] = float(np.mean([float(x["gap_softmax"]) for x in sub if float(x["margin"]) <= 0])) if len(sep) < len(sub) else float("nan")
        log(f"    {feat} features: A separable in {len(sep)}/{len(sub)} fits; softmax-row gap when separable "
            f"{S['margin'][f'{feat}_gap_softmax_when_separable']:.3f}, when not {S['margin'][f'{feat}_gap_softmax_when_not']:.3f}")
        for d in MIT_RANKS:
            s2 = [x for x in sub if int(x["d"]) == d]
            log(f"      d={d}: margin mean {np.mean([float(x['margin']) for x in s2]):+.3f} "
                f"[{min(float(x['margin']) for x in s2):+.3f}, {max(float(x['margin']) for x in s2):+.3f}], "
                f"separable {sum(float(x['margin']) > 0 for x in s2)}/{len(s2)}")
    ml, ms = max(float(x["moment_linear"]) for x in mit), max(float(x["moment_softmax"]) for x in mit)
    S["moment"] = dict(max_abs_linear=ml, max_abs_softmax=ms)
    log(f"(iii) moment check B^T (row_fit - row_true): max |.| over all rows and fits: least squares {ml:.1e}, maximum likelihood {ms:.1e}")
    assert ml < 1e-9 and ms < 1e-6, (ml, ms)
    return S


# ----------------------------------------------------------------------------
# 7. IO.
# ----------------------------------------------------------------------------
def write_csv(rows, name):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {HERE / name} ({len(rows)} rows)")


def read_csv(name):
    def conv(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    with open(HERE / name) as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


def rebuild_processes(stored):
    mdps = []
    for p in stored["processes"]:
        edges, b, A, B, T = dsb.build(**{k: v for k, v in p["spec"].items()})
        m = finish(p["id"], p["name"], p["spec"], edges, np.array(p["pi"]), b, A, B, T, p["attempts"])
        assert m["edges"].tolist() == p["edges"] and abs(m["gap"] - p["gap"]) < 1e-9
        m["rejected"], m["lin2_at_draw"] = p.get("rejected", []), p.get("lin2_at_draw")
        mdps.append(m)
    return mdps


if __name__ == "__main__":
    t0 = time.time()
    figures_only = "--figures-only" in sys.argv
    if figures_only:
        mdps = rebuild_processes(json.load(open(HERE / "sb_processes.json")))
        describe(mdps)
        closed, trained = read_csv("sb_closed.csv"), read_csv("sb_trained.csv")
        rare_closed, rare_trained = read_csv("sb_rare_closed.csv"), read_csv("sb_rare_trained.csv")
        lemma_rows, reward_rows, construct_rows = read_csv("sb_lemma.csv"), read_csv("sb_reward.csv"), read_csv("sb_construct.csv")
        mit = read_csv("sb_mitigation.csv")
    else:
        fixed = fixed_processes()
        p6 = [m for m in fixed if m["variant"] == "P6"][0]
        p11 = [m for m in fixed if m["variant"] == "P11"][0]
        randoms = [draw_random(k) for k in range(1, N_RANDOM + 1)]
        mdps = [p11] + randoms + [p6]
        describe(mdps)
        for m in randoms:
            if m["rejected"]:
                log(f"{m['variant']}: {len(m['rejected'])} draw(s) rejected before it: " + "; ".join(
                    f"{r['spec']} gap {r['gap']} lin2 {r['lin2']}" for r in m["rejected"]))
        sweep = [p11] + randoms
        log("--- closed form (six arms + lemma) ---")
        closed, lemma_closed, construct_rows = run_closed(sweep)
        write_csv(closed, "sb_closed.csv")
        write_csv(construct_rows, "sb_construct.csv")
        log("--- mitigation ---")
        mit = run_mitigation(sweep)
        write_csv(mit, "sb_mitigation.csv")
        log("--- rarity sweep on P6 ---")
        rare_closed, rare_trained = run_rarity(p6)
        write_csv(rare_closed, "sb_rare_closed.csv")
        write_csv(rare_trained, "sb_rare_trained.csv")
        log("--- TD-trained (six arms + lemma) ---")
        trained, lemma_trained, reward_rows = run_trained(sweep)
        write_csv(trained, "sb_trained.csv")
        lemma_rows = lemma_closed + lemma_trained
        write_csv(lemma_rows, "sb_lemma.csv")
        write_csv(reward_rows, "sb_reward.csv")
    sweep = [m for m in mdps if m["seed"] != ID_P6]
    p6 = [m for m in mdps if m["seed"] == ID_P6][0]
    draw_graphs(mdps)
    make_ablation_figures(closed, trained, sweep)
    make_rarity_figure(rare_closed, rare_trained)
    make_lemma_figure(lemma_rows, reward_rows, construct_rows, sweep)
    make_mitigation_figures(mit)
    S = summarize(mdps, closed, trained, rare_closed, rare_trained, lemma_rows, reward_rows, construct_rows, mit)
    (HERE / "sb_summary.json").write_text(json.dumps(S, indent=2, default=float))
    log(f"\ntotal wall time {time.time() - t0:.0f}s")
    if not figures_only:
        (HERE / "sb_log.txt").write_text("\n".join(LOG) + "\n")
