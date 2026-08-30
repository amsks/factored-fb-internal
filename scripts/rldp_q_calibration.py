"""Is the critic's Q on the same SCALE as the return it is supposed to predict?

Everything else in this campaign has been checked in isolation -- the representation is
predictive, the loss is stable, the phi gradient balance is right -- and the returns are
still ~10% of published. This asks the question those checks cannot: does Q(s,a,z_r) mean
anything numerically?

For a reward r, the FB/SF value is Q = <F(s,a,z_r), z_r>, and it is SUPPOSED to be the
discounted return sum_t gamma^t r(s_{t+1}). With walker/walk rewards averaging 0.189 and
gamma 0.98, that return cannot exceed about 0.189/(1-0.98) ~ 9.5, and the observed
q_mean in training is ~1300.

Three numbers, all on a real checkpoint:
  Q_model     mean <F(s,a,z_r), z_r> over dataset (s,a)
  R_mc        the MONTE-CARLO discounted return the policy actually achieves, rolled in
              the env from reset, with the same inferred z
  ratio       Q_model / R_mc. ~1 is calibrated. A large ratio means the actor is
              maximizing a surface with no numerical relationship to return -- which
              cannot be fixed by training longer.

    python scripts/rldp_q_calibration.py <run_dir> <epoch> [domain] [task]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from env import make_config, get_adapter
from env.exorl_env import make_exorl_env
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent

run_dir, epoch = sys.argv[1], int(sys.argv[2])
domain = sys.argv[3] if len(sys.argv) > 3 else "dmc-walker"
task = sys.argv[4] if len(sys.argv) > 4 else "walk"
cfg = make_config("rldp_official", "ddpgbc", domain,
                  ["discount=0.98", "exorl_max_episodes=400", f"exorl_task={task}",
                   "actor.alpha=0.0", "actor.q_normalize=false",
                   # the official actor architecture -- a checkpoint trained with it cannot
                   # be restored into the seam's default MLP actor (different param tree)
                   "actor.motivo_archi=true", "actor.hidden_dims=[1024]",
                   "actor.q_noise_std=0.2", "actor.boot_noise_std=0.2"])
adapter = get_adapter(cfg)
env_, ds, _ = adapter.make_env_and_datasets(cfg)
ds = make_dataset(ds, cfg)
b0 = ds.sample(4)
agent = Agent.create("rldp_official", "ddpgbc", 0,
                     jnp.asarray(b0["observations"]), jnp.asarray(b0["actions"]), cfg)
agent = restore_agent(agent, run_dir, epoch)

# z inferred exactly as evaluation does
rb = ds.sample(50000)
cond = agent.critic.infer_cond(agent.net, jnp.asarray(rb["next_observations"]),
                               jnp.asarray(rb["rewards"]).reshape(-1))
z1 = cond["z"]
gamma = float(cfg["discount"])

batch = ds.sample(2048)
obs, act = jnp.asarray(batch["observations"]), jnp.asarray(batch["actions"])
zb = jnp.broadcast_to(z1, (obs.shape[0],) + z1.shape)
params = {g: agent.net[g].params for g in agent.net}
qs, _ = agent.critic.q_fn(params, obs, {"z": zb}, act)
q_model = float(jnp.mean(qs))

# Monte-Carlo: roll the policy and discount the SAME reward the env scores
agent = agent.replace(cond_eval={**agent.cond_eval, "z": jnp.asarray(z1)[None]})
env, horizon = make_exorl_env(domain, reward_task=task, seed=0)
rets, disc_rets = [], []
for ep in range(5):
    o, _ = env.reset(seed=ep)
    tot = dtot = 0.0
    for t in range(horizon):
        # goals=None on purpose: passing `goals` routes the vector through
        # map_eval_cond, which applies B(goal) and would hand the 512-dim z to a tower
        # expecting a 24-dim observation. The inferred z belongs in cond_eval, which is
        # what evaluate() does.
        a = np.asarray(agent.sample_actions(jnp.asarray(o)[None],
                                            seed=jax.random.PRNGKey(ep))[0])
        o, r, _, trunc, _ = env.step(a)
        tot += r; dtot += (gamma ** t) * r
        if trunc: break
    rets.append(tot); disc_rets.append(dtot)
env.close()
r_mc, r_und = float(np.mean(disc_rets)), float(np.mean(rets))
mean_r = float(np.mean(np.asarray(rb["rewards"])))
print(f"{domain}/{task} @ {epoch}")
print(f"  dataset mean reward        {mean_r:.4f}   =>  max discounted return ~ {mean_r/(1-gamma):.2f}")
print(f"  Q_model  <F,z_r>           {q_model:.2f}")
print(f"  R_mc     discounted return {r_mc:.2f}   (undiscounted {r_und:.1f})")
print(f"  ratio Q_model / R_mc       {q_model/max(r_mc,1e-9):.1f}x     (~1 = calibrated)")
