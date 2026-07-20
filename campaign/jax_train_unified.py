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


def build_eval(eval_task, data_dir):
    """Build the eval env + offline (next_obs, reward) subsample for FB-z / GCRL-goal.
    Returns None for unsupported tasks (e.g. cube-single -> evaluated separately)."""
    if not eval_task or not eval_task.startswith(("b2d-", "p2d-", "mw-", "ms-")):
        return None
    sys.path.insert(0, "/raid/tongle/rl/Factored-FB/campaign/eval")
    from eval_common import make_eval_env, load_offline
    env, horizon = make_eval_env(eval_task, render=True)
    nobs, rew = load_offline(data_dir)
    if len(nobs) > 10000:
        idx = np.random.default_rng(0).choice(len(nobs), 10000, replace=False)
        nobs, rew = nobs[idx], rew[idx]
    return dict(env=env, horizon=horizon, nobs=nobs, rew=rew)


def _flag(x):
    return bool(x.any()) if hasattr(x, "any") else bool(x)


def run_eval(agent, agent_name, ev, task, n_episodes, n_video):
    """Fixed-seed rollout -> (success_rate, per-episode frame lists for the first n_video)."""
    import jax
    from eval_common import seeded_reset
    if agent_name in ("fb", "fb_flowbc"):
        cond = {"z": np.asarray(agent.infer_z(ev["nobs"], ev["rew"]))}
    else:
        cond = {"goals": ev["nobs"][int(np.argmax(ev["rew"]))].astype(np.float32)}
    env, horizon = ev["env"], ev["horizon"]
    successes, frames = [], []
    for i in range(n_episodes):
        obs, info = seeded_reset(env, task, i)
        key = jax.random.PRNGKey(i)
        ep_s = _flag(info.get("success", 0))
        rec = i < n_video
        ff = [env.render()] if rec else None
        for _ in range(horizon):
            key, sub = jax.random.split(key)
            a = np.asarray(agent.sample_actions(observations=np.asarray(obs, np.float32),
                                                seed=sub, temperature=0, **cond))
            obs, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
            ep_s = ep_s or _flag(info.get("success", 0))
            if rec:
                ff.append(env.render())
            if _flag(term) or _flag(trunc):
                break
        successes.append(float(ep_s))
        if rec:
            frames.append(ff)
    return float(np.mean(successes)), frames


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
    ap.add_argument("--wandb", type=int, default=0)
    ap.add_argument("--wandb_project", default="factored-fb-campaign")
    ap.add_argument("--wandb_group", default=None, help="set to <algo>__<task> so the 3 seeds aggregate")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_family", default=None)
    # inline eval during training (fixed-seed episodes -> success + video grid -> wandb)
    ap.add_argument("--eval_task", default=None, help="task slug for the eval env (b2d-/mw-/ms-); None=no eval")
    ap.add_argument("--eval_interval", type=int, default=100_000)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--eval_video", type=int, default=4)
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

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         name=args.wandb_name or f"{agent_name}__s{args.seed}", reinit=True,
                         tags=[args.wandb_family] if args.wandb_family else None,
                         config=dict(agent=agent_name, seed=args.seed, task=args.eval_task,
                                     family=args.wandb_family, data_dir=args.data_dir))

    ev = build_eval(args.eval_task, args.data_dir)      # eval env (None if unsupported)
    print(f"[jax_train] inline eval: {'on '+args.eval_task if ev else 'off'}", flush=True)

    def do_eval(step):
        if ev is None:
            return
        sr, frames = run_eval(agent, agent_name, ev, args.eval_task, args.eval_episodes, args.eval_video)
        print(f"[jax_train] step {step}: eval/success_rate={sr:.3f}", flush=True)
        if run is not None:
            log = {"eval/success_rate": sr}
            if args.eval_video > 0 and frames:
                from eval_common import save_video_grid
                vp = save_video_grid(frames, os.path.join(args.save_dir, f"eval_step{step}.mp4"))
                if vp:
                    log["eval/video"] = wandb.Video(vp, format="mp4")
            wandb.log(log, step=step)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    import tqdm
    for i in tqdm.tqdm(range(1, args.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        agent, info = agent.update(ds.sample(cfg["batch_size"]))
        if i % args.log_interval == 0:
            fi = {k: float(v) for k, v in info.items()}
            print(f"[jax_train] step {i}: {dict(list(fi.items())[:5])}", flush=True)
            if run is not None:
                wandb.log({f"train/{k}": v for k, v in fi.items()}, step=i)
        if i % args.save_interval == 0:
            save_agent(agent, args.save_dir, i)
        if args.eval_interval > 0 and i % args.eval_interval == 0:
            do_eval(i)
    save_agent(agent, args.save_dir, args.train_steps)
    do_eval(args.train_steps)
    if run is not None:
        run.finish()
    print(f"[jax_train] DONE {agent_name} {args.data_dir} -> {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
