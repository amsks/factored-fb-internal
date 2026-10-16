"""Ben's question 2: at a fixed rank, are linear features (FB: R ~ F.B) or log-linear features
(Density-FB: R ~ exp(F.B)) more expressive for the discounted future-state ratio
R_ij = M_ij / rho_j, measured by Monte-Carlo LSIF train/validation loss?

Episodes are split 80/20. K-means clusters, rho and the empirical discounted successor matrix M
are built on the TRAIN split; validation episodes are assigned to the same clusters and give
M_val. Fits on M_train at each rank:
  lin_lsif : argmin sum_ij rho_i rho_j (R_ij - F_i.B_j)^2        exact (weighted SVD)  = FB
  log_lsif : same loss with R_hat = exp(F_i.B_j)                 Adam                  = log-linear + LSIF
  log_ce   : P_ij = rho_j exp(F_i.B_j)/Z_i, cross-entropy        Adam                  = Density-FB
Reported on train and val: relative LSIF loss (sum rho rho (R - Rhat)^2 / sum rho rho R^2), and the
TV of the predicted row after the readout each method would use (lin raw; lin clipped and
self-normalised over the candidates; log_lsif self-normalised; log_ce softmax).
Usage: python scripts/features_lsif.py --npz PATH --coords 19:22 --K 512 --gamma 0.99 --out results/lsif/x.json
"""
import argparse, json, os, time
import numpy as np, jax, jax.numpy as jnp, optax
ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True); ap.add_argument("--coords", default="full"); ap.add_argument("--K", type=int, default=512)
ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--ranks", default="8,16,32,64,128")
ap.add_argument("--max_steps", type=int, default=2_000_000); ap.add_argument("--fit_steps", type=int, default=4000)
ap.add_argument("--val_frac", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(a.seed)
d = np.load(a.npz); obs = d["observations"].astype(np.float32); term = d["terminals"].astype(bool)
if len(obs) > a.max_steps:
    cut = np.flatnonzero(term)[np.searchsorted(np.flatnonzero(term), a.max_steps) - 1] + 1; obs, term = obs[:cut], term[:cut]
if a.coords == "full": X = obs
else:
    cols = np.concatenate([np.arange(int(s.split(":")[0]), int(s.split(":")[1])) for s in a.coords.split(",")]); X = obs[:, cols]
ep_end = np.flatnonzero(term); ep_start = np.concatenate([[0], ep_end[:-1] + 1]); ep_len = ep_end - ep_start + 1; E = len(ep_end)
is_val = rng.random(E) < a.val_frac
step_val = np.repeat(is_val, ep_len)
mu, sd = X[~step_val].mean(0), X[~step_val].std(0) + 1e-6; X = (X - mu) / sd
K = a.K; Xj = jnp.asarray(X); tr_idx = np.flatnonzero(~step_val)
sub = Xj[rng.choice(tr_idx, min(len(tr_idx), 300_000), replace=False)]
cent = [sub[rng.integers(len(sub))]]; d2 = jnp.sum((sub - cent[0]) ** 2, -1)
for k in range(1, K):
    p = np.asarray(d2 / d2.sum()); c = sub[rng.choice(len(sub), p=p)]; cent.append(c); d2 = jnp.minimum(d2, jnp.sum((sub - c) ** 2, -1))
C = jnp.stack(cent)
@jax.jit
def assign(x, C): return jnp.argmin(jnp.sum(x ** 2, -1)[:, None] - 2 * x @ C.T + jnp.sum(C ** 2, -1)[None], -1)
def assign_all(x, C, bs=200_000): return np.concatenate([np.asarray(assign(x[i:i + bs], C)) for i in range(0, len(x), bs)])
for it in range(30):
    lab = assign_all(sub, C); cnt = np.bincount(lab, minlength=K)
    newC = jax.ops.segment_sum(sub, jnp.asarray(lab), K) / jnp.maximum(cnt, 1)[:, None]; dead = cnt == 0
    if dead.any(): newC = newC.at[jnp.asarray(np.flatnonzero(dead))].set(sub[rng.choice(len(sub), dead.sum())])
    C = newC
lab = assign_all(Xj, C).astype(np.int64)
print(f"[{a.tag}] kmeans {time.time()-t0:.0f}s", flush=True)
g = a.gamma
def successor(eps):
    Tmax = int(ep_len[eps].max()); pad = np.full((len(eps), Tmax), -1, np.int64)
    for r, e in enumerate(eps): pad[r, :ep_len[e]] = lab[ep_start[e]:ep_end[e] + 1]
    padj = jnp.asarray(pad); n = len(eps)
    def body(S, t):
        nxt = padj[:, t + 1]; valid = nxt >= 0
        S = jnp.where(valid[:, None], (1 - g) * jax.nn.one_hot(nxt, K) + g * S, 0.0)
        cur = padj[:, t]; return S, jax.ops.segment_sum(S, jnp.where(cur >= 0, cur, K), K + 1)[:K]
    _, contribs = jax.lax.scan(body, jnp.zeros((n, K)), jnp.arange(Tmax - 2, -1, -1))
    Mraw = np.asarray(contribs.sum(0), np.float64); rm = Mraw.sum(1, keepdims=True)
    rho = np.bincount(lab[np.concatenate([np.arange(ep_start[e], ep_end[e] + 1) for e in eps])], minlength=K).astype(np.float64); rho /= rho.sum()
    return Mraw / np.maximum(rm, 1e-12), rho, rm[:, 0] > 0
Mtr, rho_tr, ok_tr = successor(np.flatnonzero(~is_val)); Mva, rho_va, ok_va = successor(np.flatnonzero(is_val))
print(f"[{a.tag}] successor matrices {time.time()-t0:.0f}s; val rows with data {ok_va.sum()}/{K}", flush=True)
rho = rho_tr                                                     # the candidate distribution both losses are defined on
R_tr = Mtr / np.maximum(rho[None], 1e-12); R_va = Mva / np.maximum(rho[None], 1e-12)
w_tr = rho_tr * ok_tr; w_tr /= w_tr.sum(); w_va = rho_va * ok_va; w_va /= w_va.sum()
W_tr = w_tr[:, None] * rho[None]; W_va = w_va[:, None] * rho[None]
def lsif_rel(Rhat, R, W): return float((W * (R - Rhat) ** 2).sum() / (W * R ** 2).sum())
def tv(Mhat, M, wr): return float((wr * 0.5 * np.abs(M - Mhat).sum(1)).sum())
def selfnorm(Rhat): P = np.clip(Rhat, 0, None) * rho[None]; return P / np.maximum(P.sum(1, keepdims=True), 1e-12)
sr = np.sqrt(w_tr); sc = np.sqrt(rho)
A = (sr[:, None] * R_tr) * sc[None]; U, S, Vt = np.linalg.svd(A, full_matrices=False)
Rj, Wj, rhoj, Mj, wj = jnp.asarray(R_tr), jnp.asarray(W_tr), jnp.asarray(rho), jnp.asarray(Mtr), jnp.asarray(w_tr)
logrho = jnp.log(jnp.maximum(rhoj, 1e-12))
def fit(r, kind, steps):
    kf, kb = jax.random.split(jax.random.PRNGKey(a.seed)); p = dict(F=0.1 * jax.random.normal(kf, (K, r)), B=0.1 * jax.random.normal(kb, (K, r)))
    if kind == "log_lsif":  # init at log of the clipped linear solution scale: start near R = 1
        p["c"] = jnp.zeros(())
    opt = optax.adam(1e-2 if kind == "log_lsif" else 3e-2); st = opt.init(p)
    def loss_fn(p):
        L = p["F"] @ p["B"].T
        if kind == "log_lsif": return (Wj * (Rj - jnp.exp(L + p["c"])) ** 2).sum() / (Wj * Rj ** 2).sum()
        return -(wj[:, None] * Mj * jax.nn.log_softmax(L + logrho[None], -1)).sum()
    @jax.jit
    def step(p, st):
        l, gr = jax.value_and_grad(loss_fn)(p); up, st = opt.update(gr, st, p); return optax.apply_updates(p, up), st, l
    for _ in range(steps): p, st, l = step(p, st)
    L = p["F"] @ p["B"].T
    if kind == "log_lsif": return np.asarray(jnp.exp(L + p["c"]), np.float64)
    return np.asarray(jax.nn.softmax(L + logrho[None], -1), np.float64) / np.maximum(rho[None], 1e-12)   # as a ratio
res = dict(tag=a.tag, npz=a.npz, coords=a.coords, K=K, gamma=g, val_frac=a.val_frac, n_train_eps=int((~is_val).sum()), n_val_eps=int(is_val.sum()),
           ess_over_K_train=float((w_tr / (Mtr ** 2).sum(1).clip(1e-300)).sum() / K), fits={})
for r in [int(x) for x in a.ranks.split(",")]:
    Rl = ((U[:, :r] * S[:r]) @ Vt[:r]) / np.maximum(sr[:, None], 1e-12) / np.maximum(sc[None], 1e-12)
    Rg = fit(r, "log_lsif", a.fit_steps); Rc = fit(r, "log_ce", a.fit_steps)
    out = {}
    for name, Rh in (("lin_lsif", Rl), ("log_lsif", Rg), ("log_ce", Rc)):
        out[name] = dict(lsif_train=lsif_rel(Rh, R_tr, W_tr), lsif_val=lsif_rel(Rh, R_va, W_va),
                         tv_val_selfnorm=tv(selfnorm(Rh), Mva, w_va), tv_train_selfnorm=tv(selfnorm(Rh), Mtr, w_tr))
        if name == "lin_lsif":
            Praw = Rh * rho[None]; out[name]["tv_val_raw"] = tv(Praw, Mva, w_va); out[name]["neg_mass_val"] = float((w_va * np.clip(-Praw, 0, None).sum(1)).sum())
    res["fits"][r] = out
    print(f"[{a.tag}] rank {r}: " + " | ".join(f"{n} lsif tr/va {o['lsif_train']:.3f}/{o['lsif_val']:.3f} TVva {o['tv_val_selfnorm']:.3f}" for n, o in out.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
