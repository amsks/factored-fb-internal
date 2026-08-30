"""Has the unanchored TD3 actor drifted off the behavior distribution?

Official RLDP extracts with `actor_loss = -Q.mean()` and NO behavior-cloning term
(agent.py::update_td3_actor). Nothing pulls the policy toward the offline data, and the
training logs show `bc_log_prob` falling monotonically to ~-1.7e8 through phase 2. That is
expected rather than a bug -- but an offline agent whose policy leaves the data support is
a standard failure mode, and it fits run 2's pattern: representation measurably good
(phi 32x better than a no-motion baseline), returns poor, worse on the sparser domain.

Three numbers on a real checkpoint, against the dataset the agent was trained on:

  saturation   fraction of action DIMENSIONS at the clip boundary (|a| > 0.99). A TD3
               actor maximizing an unbounded bilinear Q tends to push to the corners;
               near 1.0 means the policy is effectively bang-bang and its Q ordering
               is being read entirely off the boundary of the action cube.
  |a| mean     policy vs dataset. The RND behavior policy is exploratory, not saturated,
               so a large gap is drift, not merely a different optimum.
  action MSE   policy(s) vs the dataset action at the SAME state s.

None of these is a verdict on its own -- a good policy CAN be more saturated than a random
one. Read them together, and against the dataset row, which is the distribution the
successor measure was estimated on.

    python scripts/rldp_actor_drift.py <run_dir> <epoch> [domain]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from env import make_config, get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent

run_dir, epoch = sys.argv[1], int(sys.argv[2])
domain = sys.argv[3] if len(sys.argv) > 3 else "dmc-walker"
disc = "0.98"
cfg = make_config("rldp_official", "ddpgbc", domain,
                  [f"discount={disc}", "exorl_max_episodes=300",
                   "actor.alpha=0.0", "actor.q_normalize=false",
                   "actor.q_noise_std=0.2", "actor.boot_noise_std=0.2"])
adapter = get_adapter(cfg)
env, ds, _ = adapter.make_env_and_datasets(cfg)
ds = make_dataset(ds, cfg)
b0 = ds.sample(4)
agent = Agent.create("rldp_official", "ddpgbc", 0,
                     jnp.asarray(b0["observations"]), jnp.asarray(b0["actions"]), cfg)
agent = restore_agent(agent, run_dir, epoch)

batch = ds.sample(2048)
obs = jnp.asarray(batch["observations"])
data_a = np.asarray(batch["actions"])
# The eval-time action, through the SAME path adapter.evaluate uses: infer z from
# reward-labelled rows, install it as the eval cond, then Agent.sample_actions.
cond = agent.critic.infer_cond(agent.net, jnp.asarray(batch["next_observations"]),
                               jnp.asarray(batch["rewards"]).reshape(-1))
# infer_cond returns ONE z vector; the actor concatenates it per row, so broadcast it
# to the batch exactly as evaluate() does when it rolls a single inferred index.
cond = {k: jnp.broadcast_to(v, (obs.shape[0],) + v.shape) if jnp.asarray(v).ndim == 1 else v
        for k, v in cond.items()}
agent = agent.replace(cond_eval={**agent.cond_eval, **cond})
acts = agent.sample_actions(obs, seed=jax.random.PRNGKey(0))
pol_a = np.asarray(acts)

sat = lambda a: float((np.abs(a) > 0.99).mean())
print(f"{run_dir}  @ {epoch}   domain={domain}")
print(f"  saturation  policy {sat(pol_a):.3f}   dataset {sat(data_a):.3f}")
print(f"  |a| mean    policy {np.abs(pol_a).mean():.3f}   dataset {np.abs(data_a).mean():.3f}")
print(f"  action MSE  policy vs dataset action at the same state: {((pol_a-data_a)**2).mean():.4f}")
