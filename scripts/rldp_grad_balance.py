"""Measure RLDP phase-1's SP-vs-ortho GRADIENT balance on the real config and data.

Written for the 2026-08-28 official-parity campaign postmortem. The campaign failed its
replication gate (walker at 2-24% of RLDP Table 1) and the loss magnitudes were misleading
about why: phase 1 logged sp ~ 1.66 against ortho terms ~ 510, which looks like ortho
merely being large. Losses are not gradients. This measures the thing that actually drives
learning, and the answer was unambiguous:

    ||grad SP||    =      0.588
    ||grad ortho|| = 35806.6        ratio 6.1e4

i.e. phase 1 optimized whitening essentially to the exclusion of the self-predictive term,
so B never learned dynamics and phase 2 regressed F onto a dynamics-blind frozen basis.

Run it before trusting any phase-1 configuration:
    FB_MUJOCO_SITE_PACKAGES=... python scripts/rldp_grad_balance.py

Original docstring follows.

Measure the phase-1 gradient balance: SP term vs ortho term, on the real
checkpoint and real data. Answers 'is phase 1 dominated by whitening?' with a
number instead of an argument about loss magnitudes (which are not gradients)."""
import os, sys, pickle
sys.path.insert(0, "/home/adityamohan/git/Factored-FB-seam")
import jax, jax.numpy as jnp, numpy as np
from env import make_config, get_adapter

cfg = make_config(__import__('os').environ.get('GB_CRITIC','rldp_official'),"ddpgbc","dmc-walker",
                  ["discount=0.98","exorl_max_episodes=300"])
adapter = get_adapter(cfg)
env, train_ds, _ = adapter.make_env_and_datasets(cfg)
from impls.utils.datasets import make_dataset
train_ds = make_dataset(train_ds, cfg)   # main.py:281 -- applies dataset_class
from impls.agents import Agent
b0 = train_ds.sample(4)
agent = Agent.create("rldp_official", "ddpgbc", 0,
                     jnp.asarray(b0["observations"]), jnp.asarray(b0["actions"]), cfg)
batch = train_ds.sample(int(cfg["batch_size"]))
critic = agent.critic
params = {g: agent.net[g].params for g in agent.net}
target = agent.target

def sp_only(bp):
    p = {**params, "backward": bp}
    sp, _ = critic._sp_loss(p, target, batch)
    return sp
def ortho_only(bp):
    p = {**params, "backward": bp}
    _, B_pred = critic._sp_loss(p, target, batch)
    n = B_pred.shape[0]; off = 1.0 - jnp.eye(n)
    o, _, _ = critic._ortho(B_pred, off, off.sum())
    return cfg["ortho_coef"] * o

bp = params["backward"]
gs = jax.grad(sp_only)(bp); go = jax.grad(ortho_only)(bp)
nrm = lambda t: float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(t))))
ns, no = nrm(gs), nrm(go)
print(f"phase-1 gradient norms wrt BACKWARD params (batch={cfg['batch_size']}, ortho_coef={cfg['ortho_coef']}):")
print(f"  ||grad SP||    = {ns:.6g}")
print(f"  ||grad ortho|| = {no:.6g}")
print(f"  ratio ortho/SP = {no/max(ns,1e-12):.4g}")
