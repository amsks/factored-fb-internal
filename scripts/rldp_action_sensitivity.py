"""Does the critic distinguish actions at all?

The symptom that started this: M(s,a,s+) came out essentially identical for the dataset
action, a uniformly random action, and the NEGATED action -- 589.1 / 591.9 / 586.0. A
successor measure that cannot tell "step forward" from "step backward" cannot support
control, and no amount of actor training fixes it.

The hypothesis is that this is downstream of representation collapse: when B spans a
~25-dimensional positive cone, F(s,a,z)^T B(s+) has no room left to separate one action's
successor distribution from another's. This script is the test. Run it on a collapsed
checkpoint and a de-collapsed one at the SAME step and compare.

  act_spread   mean over states of the std of Q across the three action variants
  state_spread std of Q across states, at the dataset action
  ratio        act_spread / state_spread. Near 0 means action-blind: the critic's output
               is a function of the state and nothing else. The values at the top of this
               investigation were 0.049 at init and 0.061 after 5M steps of training.

Loads the checkpoint pickle directly rather than through restore_agent, so a run trained
before an optimizer-chain change can still be scored. Extra argv are config overrides,
e.g. rldp_official_nets=false for the pre-port runs.

    python scripts/rldp_action_sensitivity.py <run_dir> <epoch> <critic> <domain> [ov...]
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from env import make_config, get_adapter
from impls.agents import Agent
from impls.utils.datasets import make_dataset

run_dir, epoch, critic, domain = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = make_config(critic, "ddpgbc", domain,
                  ["discount=0.98", "exorl_max_episodes=200", "actor.alpha=0.0",
                   "actor.q_normalize=false", "actor.motivo_archi=true",
                   "actor.hidden_dims=[1024]"] + sys.argv[5:])
env, ds, _ = get_adapter(cfg).make_env_and_datasets(cfg)
ds = make_dataset(ds, cfg)
b0 = ds.sample(4)
ag = Agent.create(critic, "ddpgbc", 0, jnp.asarray(b0["observations"]),
                  jnp.asarray(b0["actions"]), cfg)
with open(os.path.join(run_dir, f"params_{epoch}.pkl"), "rb") as f:
    net = pickle.load(f)["agent"]["net"]
cr = ag.critic

b = ds.sample(1024)
obs, act = jnp.asarray(b["observations"]), jnp.asarray(b["actions"])
nobs = jnp.asarray(b["next_observations"])
key = jax.random.PRNGKey(0)
# One z, held fixed: the question is whether F responds to `a`, not to `z`.
z = jax.random.normal(key, (cr.z_dim,))
z = z / jnp.linalg.norm(z) * np.sqrt(cr.z_dim)
z = jnp.broadcast_to(z, (obs.shape[0], cr.z_dim))

le = cr._le(net["left_encoder"]["params"], obs)
B = cr._B(net["backward"]["params"], nobs)

def q_of(a):
    F = cr._F(net["forward"]["params"], le, z, a)
    F = F[0] if F.ndim == 3 else F          # min over heads is not wanted here
    return np.asarray(jnp.einsum("bz,bz->b", F, B))

variants = {"dataset": act,
            "random": jax.random.uniform(jax.random.fold_in(key, 1), act.shape,
                                         minval=-1.0, maxval=1.0),
            "negated": -act}
qs = {k: q_of(v) for k, v in variants.items()}
# The ratio below is NOT self-interpreting: in any control task Q varies far more across
# states than across actions at a fixed state, because the state value dominates the
# advantage. A small ratio is therefore normal and cannot by itself establish
# action-blindness. The sign tests are scale-free and are the part that carries meaning --
# a critic that cannot rank the dataset action above a random or negated one is unusable
# for control no matter how large its spread is.
win_rand = float((qs["dataset"] > qs["random"]).mean())
win_neg = float((qs["dataset"] > qs["negated"]).mean())
stack = np.stack(list(qs.values()))          # [3, B]
act_spread = stack.std(axis=0).mean()
state_spread = qs["dataset"].std()
print(f"{critic} @ {epoch}  " + "  ".join(f"Q_{k} {v.mean():9.2f}" for k, v in qs.items()))
print(f"{'':>{len(critic)}}  {'':>{len(str(epoch))}}  act_spread {act_spread:.4f}   "
      f"state_spread {state_spread:.4f}   ratio {act_spread/max(state_spread,1e-12):.4f}   "
      f"P[Q_data>Q_rand] {win_rand:.3f}   P[Q_data>Q_neg] {win_neg:.3f}")
