"""Diagnostic (read-only on the repo): for a trained density_fb_z checkpoint, draw the task
vector z from K independent 50k reward-sample draws, report (a) pairwise cosines of the z's,
(b) how well a linear function of the centered index-tower features fits the task reward
(R^2 and corr of the actor's proxy reward with the true reward), (c) episode returns per z.
Usage: python diag_z.py <domain> <seed> <restore_root> <tasks comma> <K> <episodes> <hit> <discount>
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.expanduser("~/git/Factored-FB-seam"))
from impls.agents import Agent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import resolve_run_dir, restore_agent
from impls.critics.base import infer_cond_rows
from impls.utils.eval_cond import InferredCondAgent

domain, seed, root, tasks, K, E, hit, disc = sys.argv[1:9]
seed, K, E = int(seed), int(K), int(E)
run_dir = os.path.join(root, domain, "density_fb_z", f"{domain}_density_fb_z_ddpgbc_seed_{seed}")
rec = json.load(open(os.path.join(run_dir, "config.json")))["config"]
skip = {"seed", "save_dir", "restore_path", "restore_epoch", "train_steps", "eval_episodes",
        "eval_interval", "save_interval", "log_interval", "exorl_task", "wandb", "run_group", "online"}
base = []
def walk(prefix, node):
    for k, v in node.items():
        if not prefix and k in skip: continue
        key = f"{prefix}{k}"
        if isinstance(v, dict): walk(f"{key}.", v)
        elif isinstance(v, str): base.append(f'{key}="{v}"')
        else: base.append(f"{key}={json.dumps(v)}")
walk("", rec)
out = []
for task in tasks.split(","):
    cfg = make_config("density_fb_z", "ddpgbc", domain, base + [f"exorl_task={task}", f"discount={disc}",
                      f'measure_hit_spec="{hit}"', "actor.alpha=0.0", "actor.q_normalize=false",
                      "actor.q_noise_std=0.2", "actor.boot_noise_std=0.2"])
    np.random.seed(seed)
    adapter = get_adapter(cfg)
    env, train_ds, _ = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create("density_fb_z", "ddpgbc", seed, jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, run_dir, 2000000, params_only=True)
    crit, net = agent.critic, agent.net
    rows = infer_cond_rows(crit)
    zs, fits = [], []
    ev_agents = []
    for k in range(K):
        np.random.seed(1000 * seed + 17 * k + 3)
        rb = train_ds.sample(50000)
        obs = jnp.asarray(rb[rows]); r = jnp.asarray(rb["rewards"])
        ea = agent.infer_eval_cond(obs, r)
        z = np.asarray(ea.cond_eval["z"]); zs.append(z); ev_agents.append(ea)
        # linear-proxy fit quality on THESE samples, in the encoder's own basis
        u_sc = crit._b_query(net, obs, obs)
        b = np.asarray(crit._B(crit._zenc(net), u_sc)) - np.asarray(crit._zstats(net)["mu"])[None]
        rr = np.asarray(r).reshape(-1)
        X = np.concatenate([b, np.ones((len(b), 1))], 1)
        w = np.linalg.lstsq(X, rr, rcond=None)[0]
        r2 = 1 - ((X @ w - rr) ** 2).sum() / ((rr - rr.mean()) ** 2).sum()
        proxy = b @ z / np.sqrt(len(z))
        corr = np.corrcoef(proxy, rr)[0, 1]
        # also: fit quality with the fast target tower (what F is trained against) for reference
        fits.append((float(r2), float(corr), float(rr.mean()), float((rr > 0).mean())))
    Z = np.stack(zs); Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    cos = Zn @ Zn.T
    offd = cos[~np.eye(K, dtype=bool)]
    rets = []
    for k in range(min(K, 3)):
        ea = InferredCondAgent(ev_agents[k])
        row, _ = adapter.evaluate(ea, env, None, cfg, E, 0)
        rets.append(float(row["episode_return"]))
    rec_out = dict(domain=domain, seed=seed, task=task, cos_min=float(offd.min()), cos_mean=float(offd.mean()),
                   cos_neutral=[float(z.sum() / (np.linalg.norm(z) * np.sqrt(len(z)))) for z in zs],
                   r2_linear_in_B=[f[0] for f in fits], corr_proxy_reward=[f[1] for f in fits],
                   reward_mean=fits[0][2], reward_frac_pos=fits[0][3], returns_per_z=rets)
    print("DIAG", json.dumps(rec_out), flush=True)
    out.append(rec_out)
print("DONE", flush=True)
