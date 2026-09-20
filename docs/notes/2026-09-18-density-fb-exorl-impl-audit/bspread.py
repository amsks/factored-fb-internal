"""CPU diagnostic: B-feature spread and where the eval z sits in B's spectrum, for one
density_fb_z checkpoint and one fb checkpoint on the same 20k walker rows."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.expanduser("~/git/Factored-FB-seam"))
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import restore_agent
from impls.utils.fb_math import project_z

def base_overrides(run_dir):
    rec = json.load(open(os.path.join(run_dir, "config.json")))["config"]
    skip = {"seed","save_dir","restore_path","restore_epoch","train_steps","eval_episodes",
            "eval_interval","save_interval","log_interval","exorl_task","wandb","run_group","online"}
    out = []
    def walk(prefix, node):
        for k, v in node.items():
            if not prefix and k in skip: continue
            key = f"{prefix}{k}"
            if isinstance(v, dict): walk(f"{key}.", v)
            elif isinstance(v, str): out.append(f'{key}="{v}"')
            else: out.append(f"{key}={json.dumps(v)}")
    walk("", rec); return out

def load(critic, run_dir, task, epoch, seed=0):
    cfg = make_config(critic, "ddpgbc", "dmc-walker", base_overrides(run_dir) + [f"exorl_task={task}"])
    np.random.seed(seed)
    adapter = get_adapter(cfg)
    env, ds, _ = adapter.make_env_and_datasets(cfg)
    ds = make_dataset(ds, cfg)
    ex = ds.sample(1)
    agent = Agent.create(critic, "ddpgbc", seed, jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    init_agent = agent
    agent = restore_agent(agent, run_dir, epoch, params_only=True)
    return agent, init_agent, ds

def spread(Bm):
    mu = Bm.mean(0); C = np.cov(Bm.T, bias=True); ev = np.linalg.eigvalsh(C)[::-1]
    n = len(Bm); G = Bm @ Bm.T; off = G[~np.eye(n, dtype=bool)] / 50.0
    return dict(mu2=float(mu @ mu), trC=float(ev.sum()), effrank=float(np.exp(-(ev/ev.sum()*np.log(np.maximum(ev/ev.sum(),1e-30))).sum())),
                mean_pair_cos=float(off.mean()), rms_pair_cos=float(np.sqrt((off**2).mean()))), C

N = 20000
task = sys.argv[1] if len(sys.argv) > 1 else "walk"
# ---- Density-FB z (walker_arch seed 0, the 681 recipe) ----
dfb_dir = "exp/walker_arch/dmc-walker_density_fb_z_ddpgbc_seed_0"
for epoch in (250000, 1000000):
    agent, init_agent, ds = load("density_fb_z", dfb_dir, task, epoch)
    np.random.seed(123); rb = ds.sample(N)
    obs = jnp.asarray(rb["next_observations"]); r = np.asarray(rb["rewards"]).reshape(-1)
    crit, net = agent.critic, agent.net
    u_sc = crit._b_query(net, obs, obs)
    Bz = np.asarray(crit._B(crit._zenc(net), u_sc)); Bon = np.asarray(crit._B(net["density_b"].params, u_sc))
    Bi = np.asarray(init_agent.critic._B(init_agent.net["density_b"].params, u_sc))
    s_z, C = spread(Bz); s_on, _ = spread(Bon); s_i, _ = spread(Bi)
    print(f"DFB epoch {epoch} task {task}: zenc tower {s_z}\n   online tower {s_on}\n   INIT tower {s_i}")
    z = np.asarray(agent.infer_eval_cond(obs, jnp.asarray(r)).cond_eval["z"])
    st = crit._zstats(net); Cs = np.asarray(st["cov"]); mu = np.asarray(st["mu"])
    ev, V = np.linalg.eigh(0.5*(Cs+Cs.T)); ev = ev[::-1]; V = V[:, ::-1]
    proj = (V.T @ z)**2 / 50.0   # fraction of z's energy per eigen-direction (descending variance)
    rng = np.random.default_rng(0); zs = rng.normal(size=(4000,50)); zs *= np.sqrt(50)/np.linalg.norm(zs,axis=1,keepdims=True)
    q_sphere = np.einsum('bi,ij,bj->b', zs, Cs, zs); q_eval = z @ Cs @ z
    proxy = (Bz - mu) @ z / np.sqrt(50)
    print(f"   eval z: z^T C z = {q_eval:.3f} vs sphere-z median {np.median(q_sphere):.3f} [p5 {np.percentile(q_sphere,5):.3f}, p95 {np.percentile(q_sphere,95):.3f}]; "
          f"energy of z in top-5 / bottom-25 eigdirs = {proj[:5].sum():.2f} / {proj[25:].sum():.2f}; "
          f"proxy reward std {proxy.std():.3f}, corr(proxy, r) {np.corrcoef(proxy, r)[0,1]:.3f}; r mean {r.mean():.3f} std {r.std():.3f}")
    # linear fit of r on centered B (R2), and on RAW obs for reference
    X = np.concatenate([Bz - mu, np.ones((N,1))],1); w = np.linalg.lstsq(X, r, rcond=None)[0]
    r2 = 1-((X@w-r)**2).sum()/((r-r.mean())**2).sum()
    print(f"   R2(r ~ linear in zenc B) = {r2:.3f}")
# ---- FB reference (walker_fbref seed 0) ----
fb_dir = "exp/walker_fbref/dmc-walker_fb_ddpgbc_seed_0"
agent, init_agent, ds = load("fb", fb_dir, task, 1000000)
np.random.seed(123); rb = ds.sample(N)
obs = jnp.asarray(rb["next_observations"]); r = np.asarray(rb["rewards"]).reshape(-1)
crit, net = agent.critic, agent.net
Bm = np.asarray(crit._B(net["backward"].params, obs)); s_fb, Cfb = spread(Bm)
Bi = np.asarray(init_agent.critic._B(init_agent.net["backward"].params, obs)); s_fbi, _ = spread(Bi)
print(f"FB ref 1M: {s_fb}\n   INIT: {s_fbi}")
z = np.asarray(project_z((jnp.asarray(r).reshape(1,-1) @ jnp.asarray(Bm)).reshape(-1), True))
rng = np.random.default_rng(0); zs = rng.normal(size=(4000,50)); zs *= np.sqrt(50)/np.linalg.norm(zs,axis=1,keepdims=True)
q_sphere = np.einsum('bi,ij,bj->b', zs, Cfb, zs); q_eval = z @ Cfb @ z
proxy = Bm @ z / np.sqrt(50)
X = np.concatenate([Bm, np.ones((N,1))],1); w = np.linalg.lstsq(X, r, rcond=None)[0]
r2 = 1-((X@w-r)**2).sum()/((r-r.mean())**2).sum()
print(f"   FB eval z: z^T C z = {q_eval:.3f} vs sphere median {np.median(q_sphere):.3f} [p5 {np.percentile(q_sphere,5):.3f}, p95 {np.percentile(q_sphere,95):.3f}]; "
      f"proxy (B.z/sqrt d) std {proxy.std():.3f}, corr(proxy,r) {np.corrcoef(proxy,r)[0,1]:.3f}; R2(r ~ linear in B) = {r2:.3f}")
print("DONE")
