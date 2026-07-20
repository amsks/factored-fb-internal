"""campaign/jax_train_unified.py — train an OGBench GCRL agent (gciql / crl / crl_flowbc)
on a unified Factor-FB dataset, TRAIN-ONLY (no env, no eval), saving flax params.

Runs under .venv-jax with the OGBench impls + wandb_mode_shim on PYTHONPATH:

    PYTHONPATH=<impls>:<wandb_mode_shim> \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
    .venv-jax/bin/python campaign/jax_train_unified.py \
        --data_dir /raid/tongle/rl/fb-data/collected/Box2D__b2d-goal-ctrl \
        --agent_file agents/gciql.py --save_dir <out> \
        --train_steps 500000 --save_interval 100000 --seed 0 --set alpha=1.0

Reuses OGBench's GCDataset (which synthesizes goal-conditioned rewards/masks from the
`terminals` trajectory boundaries), so the dataset's own reward is unused — exactly the
GCRL baseline recipe, just fed our offline data instead of make_env_and_datasets().
"""
import argparse
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

IMPLS = "/raid/tongle/rl/Factored-FB/third_party/ogbench/impls"
SHIM = "/raid/tongle/rl/Factored-FB/tools/wandb_mode_shim"
for p in (IMPLS, SHIM):
    if p not in sys.path:
        sys.path.insert(0, p)


def _ep_idx(p):
    return int(re.findall(r"\d+", Path(p).stem)[-1])


def load_unified(data_dir):
    """Concatenate unified per-episode npz -> flat OGBench dataset dict with terminals
    marking each episode's last step (trajectory boundaries for GCDataset goal sampling)."""
    eps = sorted(glob.glob(f"{data_dir}/buffer/episode_*.npz"), key=_ep_idx)
    assert eps, f"no episodes in {data_dir}"
    obs, act, nobs, term = [], [], [], []
    for ep in eps:
        z = np.load(ep)
        T = len(z["action"])
        t = np.zeros(T, dtype=np.float32)
        t[-1] = 1.0
        obs.append(z["observation"].astype(np.float32))
        act.append(np.clip(z["action"], -1.0, 1.0).astype(np.float32))
        nobs.append(z["next_observation"].astype(np.float32))
        term.append(t)
    return dict(
        observations=np.concatenate(obs),
        actions=np.concatenate(act),
        next_observations=np.concatenate(nobs),
        terminals=np.concatenate(term),
    )


def load_agent_config(agent_file):
    """Import an agent file (agents/gciql.py, .../crl_flowbc.py) -> its get_config()."""
    path = agent_file if os.path.isabs(agent_file) else os.path.join(IMPLS, agent_file)
    if not os.path.exists(path):  # shim lives outside impls
        path = os.path.join("/raid/tongle/rl/Factored-FB", agent_file)
    spec = importlib.util.spec_from_file_location("agent_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--agent_file", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--train_steps", type=int, default=500_000)
    ap.add_argument("--save_interval", type=int, default=100_000)
    ap.add_argument("--log_interval", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=val")
    args = ap.parse_args()

    import agents
    from utils.datasets import Dataset, GCDataset
    from utils.flax_utils import save_agent

    cfg = load_agent_config(args.agent_file)
    with cfg.unlocked():
        for kv in args.set:
            k, v = kv.split("=", 1)
            cfg[k] = type(cfg[k])(v) if k in cfg and cfg[k] is not None else float(v)
    agent_name = cfg["agent_name"]
    assert agent_name in agents.agents, f"{agent_name} not registered ({list(agents.agents)})"

    data = load_unified(args.data_dir)
    print(f"[jax_train] {args.data_dir}: {len(data['observations'])} transitions, "
          f"obs={data['observations'].shape[1]} act={data['actions'].shape[1]} "
          f"agent={agent_name}", flush=True)

    ds = GCDataset(Dataset.create(**data), cfg)
    ex = ds.sample(1)
    agent = agents.agents[agent_name].create(args.seed, ex["observations"], ex["actions"], cfg)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    import tqdm
    for i in tqdm.tqdm(range(1, args.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        agent, info = agent.update(ds.sample(cfg["batch_size"]))
        if i % args.log_interval == 0:
            msg = {k: round(float(v), 4) for k, v in list(info.items())[:5]}
            print(f"[jax_train] step {i}: {msg}", flush=True)
        if i % args.save_interval == 0:
            save_agent(agent, args.save_dir, i)
    save_agent(agent, args.save_dir, args.train_steps)
    print(f"[jax_train] DONE {agent_name} {args.data_dir} -> {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
