import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.expanduser("~/git/Factored-FB-seam"))
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import restore_agent
domain, root, hit, disc = sys.argv[1:5]
for seed in [int(s) for s in sys.argv[5].split(",")]:
    run_dir = os.path.join(root, domain, "density_fb_z", f"{domain}_density_fb_z_ddpgbc_seed_{seed}")
    rec = json.load(open(os.path.join(run_dir, "config.json")))["config"]
    skip = {"seed","save_dir","restore_path","restore_epoch","train_steps","eval_episodes","eval_interval","save_interval","log_interval","exorl_task","wandb","run_group","online"}
    base=[]
    def walk(p,n):
        for k,v in n.items():
            if not p and k in skip: continue
            key=f"{p}{k}"
            if isinstance(v,dict): walk(f"{key}.",v)
            elif isinstance(v,str): base.append(f'{key}="{v}"')
            else: base.append(f"{key}={json.dumps(v)}")
    walk("",rec)
    cfg = make_config("density_fb_z","ddpgbc",domain, base+[f"discount={disc}", f'measure_hit_spec="{hit}"',"actor.alpha=0.0","actor.q_normalize=false","actor.q_noise_std=0.2","actor.boot_noise_std=0.2"])
    np.random.seed(seed)
    adapter=get_adapter(cfg); env,ds,_=adapter.make_env_and_datasets(cfg); ds=make_dataset(ds,cfg)
    ex=ds.sample(1)
    agent=Agent.create("density_fb_z","ddpgbc",seed,jnp.asarray(ex["observations"]),jnp.asarray(ex["actions"]),cfg)
    agent=restore_agent(agent,run_dir,2000000,params_only=True)
    obs=jnp.asarray(ds.sample(512)["observations"])
    rng=np.random.default_rng(seed)
    out=[]
    for k in range(4):
        z=rng.normal(size=50); z=z/np.linalg.norm(z)*np.sqrt(50)
        a=np.asarray(agent.sample_actions(obs, goals={"z": jnp.asarray(np.tile(z[None], (512, 1)))}, seed=jax.random.PRNGKey(0)))
        out.append(dict(mean_abs=float(np.abs(a).mean()), frac_sat=float((np.abs(a)>0.99).mean()), std_over_states=float(a.std(0).mean()), mean_action=[round(float(x),2) for x in a.mean(0)]))
    # sensitivity to z: two z's on same states
    print("ACT", json.dumps(dict(domain=domain, seed=seed, per_z=out)), flush=True)
