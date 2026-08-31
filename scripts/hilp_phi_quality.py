"""Is HILP's Hilbert representation actually a temporal metric? Per seed.

HILP's whole premise is that ||phi(s) - phi(g)|| approximates the temporal distance from s
to g, so a z fitted by least squares on phi spans the reward functions that matter. If phi
loses that property the policy has nothing to steer by, and the failure shows up as a
seed that trains "fine" by its losses and cannot do the task.

Reported per checkpoint:
  eff.rank     participation ratio of phi's covariance eigenvalues, out of z_dim. A
               collapsed representation cannot express distinct rewards.
  mean|cos|    pairwise cosine between phi rows; 0 = spread, 1 = collapsed.
  d(k)         mean ||phi(s_t) - phi(s_{t+k})|| at temporal gap k, over sampled windows.
               For a temporal metric this must INCREASE with k.
  spearman     rank correlation between k and d(k) across the sampled gaps. ~1.0 is a
               working Hilbert metric; ~0 means phi has no temporal structure left.

    python scripts/hilp_phi_quality.py <run_dir> <epoch> [critic] [domain]
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax.numpy as jnp, numpy as np
from env import make_config, get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset

run_dir, epoch = sys.argv[1], int(sys.argv[2])
critic = sys.argv[3] if len(sys.argv) > 3 else "hilp_tdj"
domain = sys.argv[4] if len(sys.argv) > 4 else "dmc-walker"
cfg = make_config(critic, "ddpgbc", domain, ["discount=0.98"] + sys.argv[5:])
env, ds, _ = get_adapter(cfg).make_env_and_datasets(cfg)
ds = make_dataset(ds, cfg)
b0 = ds.sample(4)
ag = Agent.create(critic, "ddpgbc", 0, jnp.asarray(b0["observations"]),
                  jnp.asarray(b0["actions"]), cfg)
with open(os.path.join(run_dir, f"params_{epoch}.pkl"), "rb") as f:
    net = pickle.load(f)["agent"]["net"]
cr = ag.critic
phi_of = lambda x: np.asarray(cr._feature_net(net["hilp_phi1"]["params"],
                                              net["hilp_stats"]["params"], jnp.asarray(x)))

b = ds.sample(2048)
P = phi_of(b["observations"])
Pn = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
C = Pn @ Pn.T
off = ~np.eye(len(C), dtype=bool)
ev = np.clip(np.linalg.eigvalsh(np.cov(P.T)), 0, None)
pr = (ev.sum() ** 2) / max((ev ** 2).sum(), 1e-30)

# Temporal structure: the raw ExoRL episodes are contiguous, so walk the flat arrays and
# only keep pairs that stay inside one episode.
obs = np.asarray(ds.dataset["observations"])
term = np.asarray(ds.dataset.get("terminals", np.zeros(len(obs))))
ends = set(np.flatnonzero(term).tolist())
gaps = [1, 2, 5, 10, 25, 50, 100]
rng = np.random.RandomState(0)
starts = rng.randint(0, len(obs) - max(gaps) - 1, size=1500)
starts = np.array([i for i in starts if not any((i + j) in ends for j in range(max(gaps)))])
ds_by_gap = []
for k in gaps:
    a, c = phi_of(obs[starts]), phi_of(obs[starts + k])
    ds_by_gap.append(float(np.linalg.norm(a - c, axis=-1).mean()))
r = np.argsort(np.argsort(ds_by_gap))
sp = float(np.corrcoef(np.arange(len(gaps)), r)[0, 1])

print(f"{os.path.basename(run_dir)[-6:]} @{epoch}  eff.rank {pr:6.1f}/{P.shape[1]}  "
      f"mean|cos| {np.abs(C[off]).mean():.4f}  ||phi|| {np.linalg.norm(P,axis=-1).mean():7.2f}  "
      f"spearman(k,d) {sp:+.3f}")
print("      d(k): " + "  ".join(f"k={k}:{d:7.3f}" for k, d in zip(gaps, ds_by_gap)))
