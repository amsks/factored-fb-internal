"""Is the phase-1 backward map actually PREDICTIVE, or has it just collapsed?

A falling SP loss is not sufficient evidence. SP also falls if B degenerates toward a
constant, because the predictor then trivially matches a target that never moves. Phase 1
optimizes nothing but SP (once `rldp_ortho_inert: true`), so nothing structurally prevents
that collapse -- it has to be checked, not assumed.

Three numbers, all computed with the critic's own code on a real batch:

  SP        the trained rollout's error, ||P^k(B(s_t)) - B_tgt(s_{t+k})||^2
  NO-MOTION the same error for a predictor that outputs B(s_t) unchanged at every step.
            This is what a collapsed / non-predictive representation scores.
  ratio     SP / NO-MOTION. Well below 1 means the predictor learned dynamics the static
            baseline does not capture. At or above 1 means it did not, and a low SP is
            only telling us the targets stopped moving.

Plus two collapse diagnostics on B itself over the batch:
  per-dim std (mean over dims) -- ~0 means constant output
  participation ratio of the embedding covariance eigenvalues -- effective dimensionality
  out of z_dim. A PR near 1 is rank-1 collapse; near z_dim is well-spread.

Usage (checkpoint must be a phase-1 one):
  python scripts/rldp_phase1_predictive.py <run_dir> <epoch>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from env import make_config, get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent

run_dir, epoch = sys.argv[1], int(sys.argv[2])
cfg = make_config("rldp_official","ddpgbc","dmc-walker",
                  ["discount=0.98","exorl_max_episodes=300",
                   "actor.alpha=0.0","actor.q_normalize=false"])
adapter = get_adapter(cfg)
env, ds, _ = adapter.make_env_and_datasets(cfg)
ds = make_dataset(ds, cfg)
b0 = ds.sample(4)
agent = Agent.create("rldp_official","ddpgbc",0,
                     jnp.asarray(b0["observations"]), jnp.asarray(b0["actions"]), cfg)
agent = restore_agent(agent, run_dir, epoch)
critic = agent.critic
params = {g: agent.net[g].params for g in agent.net}
target = agent.target
batch = ds.sample(int(cfg["batch_size"]))

actions, targets = critic._sp_pairs(batch)
P = critic.num_parallel
ys = [jax.lax.stop_gradient(critic._B(target["backward"], s)) for s in targets]
B0 = critic._B(params["backward"], batch["observations"])
official = bool(getattr(critic, "sp_official_scale", False))

def reduce_(se):
    return jnp.mean(se) if official else jnp.mean(jnp.sum(se, axis=-1))

# trained rollout
curr = jnp.broadcast_to(B0[None], (P,) + B0.shape); sp = 0.0
for a_t, y_t in zip(actions, ys):
    curr = critic._P(params["predictor"], curr, a_t)
    sp = sp + reduce_((curr - y_t[None])**2)
if not official: sp = sp / max(len(actions),1)

# no-motion baseline: the SAME reduction, predictor replaced by identity
stat = jnp.broadcast_to(B0[None], (P,) + B0.shape); nm = 0.0
for y_t in ys:
    nm = nm + reduce_((stat - y_t[None])**2)
if not official: nm = nm / max(len(ys),1)

Bn = np.asarray(B0)
cov = np.cov(Bn.T); ev = np.linalg.eigvalsh(cov); ev = np.clip(ev, 0, None)
pr = (ev.sum()**2) / max((ev**2).sum(), 1e-30)
print(f"checkpoint {run_dir} @ {epoch}   (sp_official_scale={official}, horizon={critic.horizon})")
print(f"  SP (trained rollout) = {float(sp):.6f}")
print(f"  NO-MOTION baseline   = {float(nm):.6f}")
print(f"  ratio SP/no-motion   = {float(sp)/max(float(nm),1e-12):.4f}   (<1 = predictive)")
print(f"  B per-dim std (mean) = {Bn.std(axis=0).mean():.6f}   (~0 = collapsed)")
print(f"  B participation ratio= {pr:.2f} of {Bn.shape[1]}   (~1 = rank-1 collapse)")
