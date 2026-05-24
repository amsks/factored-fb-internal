"""Extract GCIQL/GCIVL value-network PENULTIMATE features phi(s,g) for the
representation readout probe (the positive control for the FB factorization
probe). Run in .venv-jax-cpu (JAX). Dumps one npz per task with
{phi, cube, grip, lift, table_z, goal_xyz}; the torch consumer
scripts/gciql_feature_readout.py computes region/d and the R^2 readout ceiling.

phi = the last hidden activation of GCValue.value_net (MLP [512,512,512,1]),
i.e. what the final value Dense reads from — the analog of FB's backward map B.

Example:
  .venv-jax-cpu/bin/python -m scripts.gciql_feature_extract \
    --run-dir results/gciql_20260518_201030/factored-fb/factored-fb-gciql/sd001_20260518_201036 \
    --step 1000000 --data-path datasets --out analysis/gciql_feature/state_sd001 \
    --n-states 6000 --tasks 1,2,3,4,5
Use --inspect once to print the intermediates tree and confirm the penultimate.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_states(data_path, domain, n, seed, obs_key):
    files = sorted(glob.glob(str(Path(data_path) / domain / "buffer" / "episode_*.npz")))
    if not files:
        raise SystemExit(f"no episodes under {data_path}/{domain}/buffer")
    rng = np.random.default_rng(seed)
    obs_l, phys_l = [], []
    for f in files:
        z = np.load(f)
        obs_l.append(np.asarray(z[obs_key]))
        phys_l.append(np.asarray(z["physics"], np.float32))
    obs = np.concatenate(obs_l); phys = np.concatenate(phys_l)
    if len(obs) > n:
        idx = rng.choice(len(obs), n, replace=False)
        obs, phys = obs[idx], phys[idx]
    return obs, phys


def _collect_arrays(d, path, out):
    if isinstance(d, dict):
        for k, v in d.items():
            _collect_arrays(v, path + [str(k)], out)
    elif isinstance(d, (tuple, list)):
        for v in d:
            _collect_arrays(v, path, out)
    else:
        out.append(("/".join(path), np.asarray(d)))


def _find_penultimate(intermediates, width):
    arrs = []
    _collect_arrays(intermediates, [], arrs)
    cands = [(p, a) for p, a in arrs
             if a.ndim >= 2 and a.shape[-1] == width and "value" in p.lower()]
    if not cands:
        return None, None

    def dense_idx(p):
        m = re.findall(r"Dense_(\d+)", p)
        return int(m[-1]) if m else -1
    cands.sort(key=lambda pa: dense_idx(pa[0]))
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6000)
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--inspect", action="store_true", help="print intermediates tree and exit")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ogb = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    if ogb in sys.path:
        sys.path.remove(ogb)
    sys.path.insert(0, ogb)
    import ogbench  # noqa: F401
    import jax.numpy as jnp
    from utils.flax_utils import restore_agent

    flags = json.loads((Path(args.run_dir) / "flags.json").read_text())
    saved = flags.get("agent")
    name = (saved.get("agent_name") if isinstance(saved, dict) else None) or "gciql"
    if name == "gcivl":
        from agents.gcivl import GCIVLAgent as Cls, get_config
    else:
        from agents.gciql import GCIQLAgent as Cls, get_config
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v
    width = int(config["value_hidden_dims"][-1])
    fs = config.get("frame_stack")
    env_name = flags["env_name"]
    is_pixel = env_name.startswith("visual")
    obs_key = "pixels" if is_pixel else "observation"
    domain = env_name.replace("visual-", "")

    obs, phys = _sample_states(args.data_path, domain, args.n_states,
                               int(flags.get("seed", 0)), obs_key)
    cube = phys[:, 14:17].astype(np.float64)
    grip = phys[:, 6].astype(np.float64)
    lift = phys[:, 16].astype(np.float64)

    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    for t in task_ids:
        env = ogbench.make_env_and_datasets(env_name, env_only=True)
        if fs:
            from utils.env_utils import FrameStackWrapper
            env = FrameStackWrapper(env, fs)
        ex_obs, info = env.reset(options=dict(task_id=t))
        ex_act = env.action_space.sample()
        agent = Cls.create(int(flags["seed"]),
                           np.asarray(ex_obs, np.float32)[None],
                           np.asarray(ex_act, np.float32)[None], config)
        agent = restore_agent(agent, str(args.run_dir), args.step)
        goal = np.asarray(info["goal"])
        u = env.unwrapped
        tb = int(getattr(u, "_target_block", 0) or 0)
        gxyz = np.asarray(u.cur_task_info["goal_xyzs"][tb], np.float64)
        table_z = float(u.cur_task_info["init_xyzs"][tb][2])
        env.close()

        N = len(obs); bs = 256 if is_pixel else 4096
        phis = []
        used_path = None
        for i in range(0, N, bs):
            ob = jnp.asarray(obs[i:i + bs], jnp.float32)
            g = jnp.broadcast_to(jnp.asarray(goal, jnp.float32)[None],
                                 (ob.shape[0],) + goal.shape)
            _, state = agent.network.select("value")(
                ob, g, capture_intermediates=True, mutable=["intermediates"])
            inter = state["intermediates"]
            if args.inspect and i == 0:
                arrs = []
                _collect_arrays(inter, [], arrs)
                for p, a in arrs:
                    print(f"  {p}  {a.shape}")
                return
            p, a = _find_penultimate(inter, width)
            if a is None:
                raise SystemExit("penultimate not found; rerun with --inspect")
            used_path = p
            phis.append(np.asarray(a))
        phi = np.concatenate(phis, 0)
        np.savez(out / f"task{t}.npz", phi=phi, cube=cube, grip=grip, lift=lift,
                 table_z=table_z, goal_xyz=gxyz, agent=name, is_pixel=is_pixel)
        print(f"[extract:{name}] task{t}: phi {phi.shape} via {used_path}")


if __name__ == "__main__":
    main()
