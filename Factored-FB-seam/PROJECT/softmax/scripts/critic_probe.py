"""Critic-only probe: how well does a TRAINED critic's readout over a candidate set match the
realized discounted future of held-out episodes?

For held-out rows (s_t, a_t) with a hindsight goal g = s_{t+1+k}, k ~ Geom(1-γ) from the same
episode, the critic predicts a distribution over M random candidate next-states U:
  Density-FB : softmax_m( F(s,a,g)·B(u_m) / τ )                 (mean of the two heads)
  FB         : F(s,a,z)·B(u_m), z = project_z(B(g)), clipped at 0 and normalised over m
Ground truth per row: the episode's own realised discounted future, at cluster resolution
(k-means centres saved by occupancy_rank.py --save_matrix):
  truth_m ∝ S_t[c(u_m)] / rho[c(u_m)],   S_t = Σ_k (1-γ) γ^k onehot(c(s_{t+1+k}))
Metrics: TV, top-1 agreement, Spearman, and the relative error of the actor-relevant value
Σ_m w_m r(u_m) for the goal-kernel reward r = _delta_phi(g, U).
Usage: python critic_probe.py --run_dir DIR --epoch 500000 --val VAL.npz --matrix M.npz --out out.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True); ap.add_argument("--epoch", type=int, required=True)
ap.add_argument("--val", required=True); ap.add_argument("--matrix", required=True)
ap.add_argument("--coords", default="19:22"); ap.add_argument("--n_rows", type=int, default=4096)
ap.add_argument("--n_cand", type=int, default=1024); ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
a = ap.parse_args()
import jax, jax.numpy as jnp
from scipy.stats import spearmanr
from env import get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent
from impls.critics.density_fb import _delta_phi, _raw_block_cols


rng = np.random.default_rng(a.seed)
meta = json.load(open(os.path.join(a.run_dir, "config.json"))); cfg = dict(meta["config"])
adapter = get_adapter(cfg); env, train_ds, _ = adapter.make_env_and_datasets(cfg)
ex = make_dataset(train_ds, cfg).sample(1)
agent = Agent.create(meta["critic"], meta["actor"], meta["seed"], jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
agent = restore_agent(agent, a.run_dir, a.epoch)
critic = agent.critic; net = {g: agent.net[g].params for g in agent.net}
kind = "density" if meta["critic"].startswith("density_fb") else "fb"
print("loaded", meta["critic"], "groups", list(net), flush=True)

# ---- held-out rows
d = np.load(a.val); obs = d["observations"].astype(np.float32); acts = d["actions"].astype(np.float32); term = d["terminals"].astype(bool)
ep_end = np.flatnonzero(term); ep_start = np.concatenate([[0], ep_end[:-1] + 1])
mz = np.load(a.matrix); centers, mu, sd, rho = mz["centers"], mz["mu"], mz["sd"], mz["rho"]
cols = np.concatenate([np.arange(int(s.split(":")[0]), int(s.split(":")[1])) for s in a.coords.split(",")])
X = (obs[:, cols] - mu) / sd
lab = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), -1) if len(X) < 200000 else np.concatenate([np.argmin(((X[i:i+50000, None, :] - centers[None]) ** 2).sum(-1), -1) for i in range(0, len(X), 50000)])
K = len(rho); g_ = a.gamma
# realised discounted future per step, per episode (reverse recursion)
S = np.zeros((len(obs), K), np.float32)
for s0, e0 in zip(ep_start, ep_end):
    acc = np.zeros(K, np.float32)
    for t in range(e0 - 1, s0 - 1, -1):
        acc = g_ * acc; acc[lab[t + 1]] += (1 - g_); S[t] = acc
# sample rows and hindsight goals
ep_of = np.repeat(np.arange(len(ep_end)), ep_end - ep_start + 1)
cand_pool = np.flatnonzero(~term)  # next_observations exist
rows = []
while len(rows) < a.n_rows:
    i = int(rng.choice(cand_pool)); e = ep_of[i]
    if ep_end[e] - i < 2: continue
    k = min(int(rng.geometric(1 - g_)) - 1, ep_end[e] - i - 1)
    rows.append((i, i + 1 + k))
rows = np.array(rows); ri, gi = rows[:, 0], rows[:, 1]
U_idx = rng.choice(cand_pool, a.n_cand, replace=False); U = obs[U_idx + 1]            # candidate next states
truth = S[ri][:, lab[U_idx + 1]] / np.maximum(rho[lab[U_idx + 1]], 1e-12)[None]      # [N, M]
keep = truth.sum(1) > 0; print(f"rows with future mass on candidates: {keep.mean():.3f}", flush=True)
truth = truth[keep] / truth[keep].sum(1, keepdims=True); ri, gi = ri[keep], gi[keep]
s, act, goal = jnp.asarray(obs[ri]), jnp.asarray(acts[ri]), jnp.asarray(obs[gi]); Uj = jnp.asarray(U)
blocks = critic.blocks if getattr(critic, "blocks", None) else None

# ---- critic predictions
pred, negmass = [], []
bs = 512
for i0 in range(0, len(ri), bs):
    sb, ab, gb = s[i0:i0+bs], act[i0:i0+bs], goal[i0:i0+bs]
    if kind == "density":
        f = critic._F(net["density_f"], critic._sobs(net, sb), ab, gb)                       # [2,B,d]
        b = critic._B(net["density_b"], critic._b_query(net, Uj, Uj))                        # [M,d]
        w = jax.nn.softmax(jnp.einsum("hbd,md->hbm", f, b) / critic.temp, -1).mean(0)       # [B,M]
        pred.append(np.asarray(w)); negmass.append(np.zeros(len(sb)))
    else:
        from impls.critics.fb import project_z
        z = project_z(critic._B(net["backward"], gb), critic.norm_z)                          # [B,z]
        le = critic._le(net["left_encoder"], sb)
        F = critic._F(net["forward"], le, z, ab)                                              # [P,B,z]
        Bm = critic._B(net["backward"], Uj)                                                   # [M,z]
        ratio = jnp.einsum("pbz,mz->pbm", F, Bm).mean(0)                                      # [B,M]
        r = np.asarray(ratio); negmass.append(np.clip(-r, 0, None).sum(1) / np.abs(r).sum(1))
        r = np.clip(r, 0, None); pred.append(r / np.maximum(r.sum(1, keepdims=True), 1e-12))
pred = np.concatenate(pred); negmass = np.concatenate(negmass)

# ---- metrics
tv = 0.5 * np.abs(pred - truth).sum(1)
top1 = (pred.argmax(1) == truth.argmax(1)).astype(float)
sp = np.array([spearmanr(pred[i], truth[i]).correlation for i in range(len(pred))])
if blocks is not None:
    kg = np.asarray(_delta_phi(goal, _raw_block_cols(Uj, blocks), blocks))                     # [N,M] goal-kernel reward
else:
    from impls.critics._measure_common import parse_hit_spec
    bl = parse_hit_spec(cfg.get("measure_hit_spec") or "19:22:0.4")
    kg = np.asarray(_delta_phi(goal, _raw_block_cols(Uj, bl), bl))
vt, vp = (truth * kg).sum(1), (pred * kg).sum(1)
vrel = np.abs(vt - vp) / np.maximum(vt, 1e-12)
# value RANKING across rows (what the actor compares): Spearman of predicted vs true goal value over rows
vrank = spearmanr(vp, vt).correlation
out = dict(run_dir=a.run_dir, epoch=a.epoch, critic=meta["critic"], kind=kind, n_rows=int(len(pred)), n_cand=a.n_cand,
           tv_mean=float(tv.mean()), tv_median=float(np.median(tv)), top1=float(top1.mean()),
           spearman_mean=float(np.nanmean(sp)), goal_value_relerr_median=float(np.median(vrel)),
           goal_value_rank_corr=float(vrank), neg_mass_mean=float(negmass.mean()),
           truth_ess_mean=float((1 / (truth ** 2).sum(1)).mean()), pred_ess_mean=float((1 / (pred ** 2).sum(1)).mean()))
print(json.dumps(out, indent=1)); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
