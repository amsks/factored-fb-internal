"""Dataset-only test of the "peaked occupancy" hypothesis.

For an offline dataset of trajectories we
  1. discretise states into K clusters (k-means on chosen coordinates),
  2. estimate the empirical discounted successor matrix
         M[i, j] = E[ sum_k (1-g) g^k 1{c_{t+1+k} = j} | c_t = i ]   (row-stochastic),
     the data marginal rho[j], and the ratio R = M / rho (what FB regresses with F.B,
     and what Density-FB models as softmax_j(F_i . B_j) over candidates j ~ rho),
  3. compare, at every rank d,
       linear  : argmin_{rank d} sum_ij rho_i rho_j (R_ij - F_i.B_j)^2   (= weighted SVD; FB's loss)
       log     : P_ij = rho_j exp(F_i.B_j) / Z_i fitted by CE  sum_i rho_i KL(M_i || P_i)  (Density-FB)
     on common metrics: TV(M_i, Mhat_i), KL, top-1 agreement, and the actor-relevant
     value error for peaked rewards.
  4. peakedness statistics of M that the theory says drive the gap.

Usage: python scripts/occupancy_rank.py --npz PATH --coords 19:22 --K 512 --gamma 0.99 --out results/x.json
       --coords full  -> all observation columns, standardised.
"""
import argparse, json, os, time
import numpy as np
import jax, jax.numpy as jnp, optax

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--coords", default="full", help="'full' or comma list of a:b column slices")
ap.add_argument("--K", type=int, default=512)
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--ranks", default="2,4,8,16,32,64,128")
ap.add_argument("--max_steps", type=int, default=2_000_000)
ap.add_argument("--fit_steps", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--nmf", action="store_true", help="also fit rank-d nonneg (KL-NMF) mixture = best case for the ratio family")
ap.add_argument("--save_matrix", action="store_true")
a = ap.parse_args()
t0 = time.time()
rng = np.random.default_rng(a.seed)

# ------------------------------------------------------------------ data
d = np.load(a.npz)
obs = d["observations"].astype(np.float32); term = d["terminals"].astype(bool)
if len(obs) > a.max_steps:
    # keep whole episodes from the front
    cut = np.flatnonzero(term)[np.searchsorted(np.flatnonzero(term), a.max_steps) - 1] + 1
    obs, term = obs[:cut], term[:cut]
if a.coords == "full":
    X = obs
else:
    cols = np.concatenate([np.arange(int(s.split(":")[0]), int(s.split(":")[1])) for s in a.coords.split(",")])
    X = obs[:, cols]
mu, sd = X.mean(0), X.std(0) + 1e-6
X = (X - mu) / sd
N, D = X.shape
ep_end = np.flatnonzero(term); ep_start = np.concatenate([[0], ep_end[:-1] + 1])
ep_len = ep_end - ep_start + 1
E = len(ep_end)
print(f"[{a.tag}] N={N} D={D} episodes={E} mean_len={ep_len.mean():.0f} max_len={ep_len.max()}", flush=True)

# ------------------------------------------------------------------ k-means (jax)
K = a.K
Xj = jnp.asarray(X)
sub = Xj[rng.choice(N, min(N, 300_000), replace=False)]
# k-means++ seeding on the subsample (greedy, vectorised D2 sampling)
cent = [sub[rng.integers(len(sub))]]
d2 = jnp.sum((sub - cent[0]) ** 2, -1)
for k in range(1, K):
    p = np.asarray(d2 / d2.sum())
    c = sub[rng.choice(len(sub), p=p)]
    cent.append(c); d2 = jnp.minimum(d2, jnp.sum((sub - c) ** 2, -1))
C = jnp.stack(cent)

@jax.jit
def assign(x, C):
    return jnp.argmin(jnp.sum(x ** 2, -1)[:, None] - 2 * x @ C.T + jnp.sum(C ** 2, -1)[None], -1)

def assign_all(x, C, bs=200_000):
    return np.concatenate([np.asarray(assign(x[i:i + bs], C)) for i in range(0, len(x), bs)])

for it in range(30):
    lab = assign_all(sub, C)
    cnt = np.bincount(lab, minlength=K)
    newC = jax.ops.segment_sum(sub, jnp.asarray(lab), K) / jnp.maximum(cnt, 1)[:, None]
    dead = cnt == 0
    if dead.any():  # re-seed dead clusters at random points
        newC = newC.at[jnp.asarray(np.flatnonzero(dead))].set(sub[rng.choice(len(sub), dead.sum())])
    C = newC
lab = assign_all(Xj, C).astype(np.int64)             # [N]
rho = np.bincount(lab, minlength=K).astype(np.float64); rho /= rho.sum()
print(f"[{a.tag}] kmeans done {time.time()-t0:.0f}s; occupied clusters={(rho>0).sum()}", flush=True)

# ------------------------------------------------------------------ discounted successor matrix
# pad episodes to [E, Tmax]; reverse recursion S_t = (1-g) onehot(c_{t+1}) + g S_{t+1}
g = a.gamma
Tmax = int(ep_len.max())
pad = np.full((E, Tmax), -1, np.int64)
for e in range(E):
    pad[e, :ep_len[e]] = lab[ep_start[e]:ep_end[e] + 1]
padj = jnp.asarray(pad)

def body(S, t):
    nxt = padj[:, t + 1]                                  # [E]
    valid = nxt >= 0
    S = jnp.where(valid[:, None], (1 - g) * jax.nn.one_hot(nxt, K) + g * S, 0.0)
    cur = padj[:, t]
    contrib = jax.ops.segment_sum(S, jnp.where(cur >= 0, cur, K), K + 1)[:K]   # rows at c_t
    return S, contrib

@jax.jit
def successor():
    S0 = jnp.zeros((E, K))
    _, contribs = jax.lax.scan(body, S0, jnp.arange(Tmax - 2, -1, -1))
    return contribs.sum(0)

Mraw = np.asarray(successor(), np.float64)               # [K, K] unnormalised
row_mass = Mraw.sum(1, keepdims=True)
M = Mraw / np.maximum(row_mass, 1e-12)                   # condition on in-episode future
rows_ok = row_mass[:, 0] > 0
# one-step transition matrix
c0 = pad[:, :-1].ravel(); c1 = pad[:, 1:].ravel(); ok = (c0 >= 0) & (c1 >= 0)
P1 = np.zeros((K, K)); np.add.at(P1, (c0[ok], c1[ok]), 1.0); P1 /= np.maximum(P1.sum(1, keepdims=True), 1e-12)
print(f"[{a.tag}] successor matrix done {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ peakedness statistics
def H(p):
    p = np.clip(p, 1e-300, None); return -(p * np.log(p)).sum(-1)
Hrho = H(rho); HM = H(M); HP1 = H(P1)
ess = 1.0 / np.maximum((M ** 2).sum(1), 1e-300)
R = M / np.maximum(rho[None, :], 1e-12)
maxR = R.max(1)
w = rho  # row weights
stats = dict(
    K=K, gamma=g, N=int(N), episodes=int(E), D=int(D),
    H_rho=float(Hrho), H_M_mean=float((w * HM).sum()), H_P1_mean=float((w * HP1).sum()),
    concentration=float(np.exp(Hrho - (w * HM).sum())),           # exp(H(rho)) / exp(E H(M_i))
    ess_mean=float((w * ess).sum()), ess_median=float(np.median(ess[rows_ok])),
    ess_over_K=float((w * ess).sum() / K),
    maxR_median=float(np.median(maxR[rows_ok])), maxR_p90=float(np.quantile(maxR[rows_ok], .9)),
    frac_rows_maxR_gt10=float((w * (maxR > 10)).sum()), frac_rows_maxR_gt100=float((w * (maxR > 100)).sum()),
    diag_mass=float((w * np.diag(M)).sum()),
    P1_self=float((w * np.diag(P1)).sum()), P1_maxprob=float((w * P1.max(1)).sum()),
)

# ------------------------------------------------------------------ linear (FB) model: weighted SVD
sr = np.sqrt(rho); A = (M / np.maximum(sr[:, None], 1e-12)) * sr[None, :]   # diag(rho^-1/2) M diag(rho^1/2)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
tot = (S ** 2).sum()
ranks = [int(r) for r in a.ranks.split(",") if int(r) <= K]

def metrics(Mhat, name):
    tv = 0.5 * np.abs(M - Mhat).sum(1)
    top1 = (Mhat.argmax(1) == M.argmax(1)).astype(float)
    Mc = np.clip(Mhat, 1e-12, None); Mc /= Mc.sum(1, keepdims=True)
    kl = (M * (np.log(np.clip(M, 1e-300, None)) - np.log(Mc))).sum(1)
    # actor-relevant: reward = indicator of a single target cluster j* = argmax_j M_ij  (peaked goal
    # reward); value error relative to the true value, averaged over rows.
    jstar = M.argmax(1); vt = M[np.arange(K), jstar]; vh = Mhat[np.arange(K), jstar]
    rel = np.abs(vt - vh) / np.maximum(vt, 1e-12)
    # ranking of future clusters: Spearman between rows (what a goal-ranking readout needs)
    return dict(tv=float((w * tv).sum()), kl=float((w * kl).sum()), top1=float((w * top1).sum()),
                goal_value_relerr=float((w * rel).sum()))

lin = {}
for r in ranks:
    Ar = (U[:, :r] * S[:r]) @ Vt[:r]
    Mhat = Ar * sr[:, None] / np.maximum(sr[None, :], 1e-12)
    lin[r] = metrics(Mhat, "lin"); lin[r]["sq_err_frac"] = float(1 - (S[:r] ** 2).sum() / tot)
    lin[r]["neg_mass"] = float((w * np.clip(-Mhat, 0, None).sum(1)).sum())
print(f"[{a.tag}] linear SVD done {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ log (Density-FB) model: CE fit
Mj = jnp.asarray(M); rhoj = jnp.asarray(rho); logrho = jnp.log(jnp.maximum(rhoj, 1e-12))
wj = jnp.asarray(w)

def fit_log(r, steps, seed):
    key = jax.random.PRNGKey(seed)
    kf, kb = jax.random.split(key)
    params = dict(F=0.1 * jax.random.normal(kf, (K, r)), B=0.1 * jax.random.normal(kb, (K, r)))
    opt = optax.adam(3e-2); st = opt.init(params)
    def loss_fn(p):
        logits = p["F"] @ p["B"].T + logrho[None, :]       # candidates j ~ rho  => softmax over j of (F.B + log rho)
        logP = jax.nn.log_softmax(logits, -1)
        return -(wj[:, None] * Mj * logP).sum()
    @jax.jit
    def step(p, st):
        l, gr = jax.value_and_grad(loss_fn)(p); up, st = opt.update(gr, st, p); return optax.apply_updates(p, up), st, l
    for i in range(steps):
        params, st, l = step(params, st)
    logits = params["F"] @ params["B"].T + logrho[None, :]
    return np.asarray(jax.nn.softmax(logits, -1), np.float64), float(l)

log = {}
for r in ranks:
    Phat, l = fit_log(r, a.fit_steps, a.seed)
    log[r] = metrics(Phat, "log")
    # the same model scored on FB's own weighted squared ratio loss (for a common axis)
    Rhat = Phat / np.maximum(rho[None, :], 1e-12)
    log[r]["sq_err_frac"] = float(((rho[:, None] * rho[None, :]) * (R - Rhat) ** 2).sum() / tot)
    print(f"[{a.tag}] log rank {r}: tv {log[r]['tv']:.3f} kl {log[r]['kl']:.3f} | lin tv {lin[r]['tv']:.3f} kl {lin[r]['kl']:.3f}  ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ NMF control: nonneg rank-d MIXTURE (ratio family, KL-optimal)
nmf = {}
if a.nmf:
    def fit_nmf(r, iters=600, seed=0):
        k = jax.random.PRNGKey(seed); k1, k2 = jax.random.split(k)
        W = jax.random.uniform(k1, (K, r), minval=0.1, maxval=1.0); Hm = jax.random.uniform(k2, (r, K), minval=0.1, maxval=1.0)
        Mw = wj[:, None] * Mj                      # row-weighted KL objective sum_i rho_i KL(M_i || (WH)_i)
        @jax.jit
        def upd(W, Hm):
            WH = jnp.maximum(W @ Hm, 1e-12)
            Hm = Hm * (W.T @ (Mw / WH)) / jnp.maximum(W.T @ (wj[:, None] * jnp.ones_like(Mj)), 1e-12)
            WH = jnp.maximum(W @ Hm, 1e-12)
            W = W * ((Mw / WH) @ Hm.T) / jnp.maximum(wj[:, None] * jnp.ones((K, 1)) @ Hm.sum(1, keepdims=True).T, 1e-12)
            return W, Hm
        for _ in range(iters): W, Hm = upd(W, Hm)
        P = np.asarray(W @ Hm, np.float64); P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
        return P
    for r in ranks:
        Phat = fit_nmf(r)
        nmf[r] = metrics(Phat, "nmf")
        print(f"[{a.tag}] nmf rank {r}: tv {nmf[r]['tv']:.3f} kl {nmf[r]['kl']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

out = dict(tag=a.tag, npz=a.npz, coords=a.coords, stats=stats, singular_values=S[:256].tolist(),
           linear={str(r): v for r, v in lin.items()}, log={str(r): v for r, v in log.items()}, nmf={str(r): v for r, v in nmf.items()},
           ranks=ranks, seconds=time.time() - t0)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
if a.save_matrix: np.savez_compressed(a.out.replace(".json", "_M.npz"), M=M.astype(np.float32), rho=rho, P1=P1.astype(np.float32), centers=np.asarray(C), mu=mu, sd=sd)
print(f"[{a.tag}] stats: " + ", ".join(f"{k}={v:.3g}" for k, v in stats.items() if isinstance(v, float)))
