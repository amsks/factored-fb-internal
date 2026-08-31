"""Cosine similarity between state representations -- the RLDP paper's Figure 1 metric.

The paper's central claim is that latent-dynamics prediction ALONE drifts toward a
degenerate solution: "while the solutions do not collapse, there is an increase in feature
similarity over the course of learning, which we refer to as a mild form of collapse. As
the space of reward functions is spanned by state features, such an increase in feature
similarity directly reduces [expressiveness]." Its fix is to regularize by minimizing
cosine similarity between any two states.

So this is the number that says whether the regularizer is doing its job, and it is
measurable long before any return is (phase 1 trains no actor, so every eval before
rldp_phase1_steps scores an untrained policy).

    mean |cos|   over random pairs of B(s). 0 = maximally spread, 1 = collapsed.
    eff. rank    participation ratio of B's covariance eigenvalues, out of z_dim.

    python scripts/rldp_feature_similarity.py <run_dir> <epoch> [critic] [domain]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from env import make_config, get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent

run_dir, epoch = sys.argv[1], int(sys.argv[2])
critic = sys.argv[3] if len(sys.argv) > 3 else "rldp_official"
domain = sys.argv[4] if len(sys.argv) > 4 else "dmc-walker"
cfg = make_config(critic, "ddpgbc", domain,
                  ["discount=0.98", "exorl_max_episodes=200", "actor.alpha=0.0",
                   "actor.q_normalize=false", "actor.motivo_archi=true",
                   "actor.hidden_dims=[1024]"] + sys.argv[5:])
env, ds, _ = get_adapter(cfg).make_env_and_datasets(cfg)
ds = make_dataset(ds, cfg)
b0 = ds.sample(4)
ag = Agent.create(critic, "ddpgbc", 0, jnp.asarray(b0["observations"]),
                  jnp.asarray(b0["actions"]), cfg)
ag = restore_agent(ag, run_dir, epoch)
b = ds.sample(1024)
B = np.asarray(ag.critic._B(ag.net["backward"].params, jnp.asarray(b["next_observations"])))
Bn = B / np.maximum(np.linalg.norm(B, axis=-1, keepdims=True), 1e-12)
C = Bn @ Bn.T
off = ~np.eye(len(C), dtype=bool)
ev = np.clip(np.linalg.eigvalsh(np.cov(B.T)), 0, None)
pr = (ev.sum() ** 2) / max((ev ** 2).sum(), 1e-30)
print(f"{critic} @ {epoch}   mean|cos| {np.abs(C[off]).mean():.4f}   mean cos {C[off].mean():+.4f}   "
      f"eff.rank {pr:.1f}/{B.shape[1]}   ||B|| {np.linalg.norm(B,axis=-1).mean():.2f}")
