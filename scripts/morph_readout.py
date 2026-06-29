#!/usr/bin/env python
"""scripts/morph_readout.py — morph the frozen-basis readout control into GCIQL one
ingredient at a time, to find which single change flips grasp-ranking 0.5 -> 0.89.

Three binary axes over a goal-conditioned IQL value V(input(s), cond(g)):
  --input phi|raw     : frozen PSM phi(s) features   vs  raw normalized obs
  --cond  z|goal      : FB inferred z=E[r*phi] (eval) vs  per-goal embedding (GCIQL)
  --goal-mode uniform|traj : independent buffer goals vs  trajectory futures

Scored (no rollouts) on the stored counterfactual grasps
(analysis/value/repsep/cf_records_psm_psm_o1000_ms*.parquet): rank-AUC of V over
success-bound vs fail-bound grasps, conditioned on each task's goal. The frozen PSM
checkpoint supplies phi, the normalizer, and (for cond=z) the relabel inference.
CPU-only. Out: analysis/morph_readout/<input>_<cond>_<goalmode>/sN.json.
"""
from __future__ import annotations

import argparse, copy, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.ogbench import load_ogbench_dataset
from envs.ogbench import ALL_TASKS, get_relabel_fn, create_ogbench_env
from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
from evals.z_inference import value_auc
from buffers.transition import episode_last_index, sample_future_idx

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis" / "morph_readout"
DOMAIN = "cube-single-play-v0"
DATA = "/dev/shm/factored-fb/datasets"
PSM_RUN = "/dev/shm/factored-fb_outputs/psm_state_orthohi__ortho_coef1000__lr_phi1e-5__s{s}"
CUBE_PHYS = slice(14, 17)
GOAL_THRESH = 0.04
P_TRAJ = 0.625


def expectile_loss(diff, tau):
    return (torch.where(diff > 0, tau, 1 - tau) * diff.pow(2)).mean()


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, *xs):
        return self.net(torch.cat(xs, -1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--input", choices=["phi", "raw"], required=True)
    ap.add_argument("--cond", choices=["z", "goal"], required=True)
    ap.add_argument("--goal-mode", choices=["uniform", "traj"], required=True)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--expectile", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--tau-target", type=float, default=0.005)
    ap.add_argument("--ckpt", default="step_1000000.pt")
    ap.add_argument("--eval-every", type=int, default=25_000)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(6)

    rundir = Path(PSM_RUN.format(s=args.seed))
    cfg = load_cfg(str(rundir / ".hydra/config.yaml"), device="cpu")
    _, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, str(rundir / "checkpoints" / args.ckpt)); m = agent.model; m.train(False)

    def inp(obs):                                   # state representation
        with torch.no_grad():
            n = m._normalize(obs)
            return m.phi(m._bw_encoder(n)) if args.input == "phi" else n
    def goalrep(obs):                               # goal representation (cond=goal)
        return inp(obs)
    in_dim = inp(torch.zeros(2, 28)).shape[-1]
    cond_dim = in_dim

    Vnet = MLP(in_dim + cond_dim); Qnet = MLP(in_dim + 5 + cond_dim)
    Qt = copy.deepcopy(Qnet)
    oV = torch.optim.Adam(Vnet.parameters(), args.lr); oQ = torch.optim.Adam(Qnet.parameters(), args.lr)

    buf = load_ogbench_dataset(domain=DOMAIN, data_path=DATA, obs_type="state", load_n_episodes=1000)
    store = buf._storage; ep_last = episode_last_index(store["timestep"]); nB = len(store["timestep"])
    rng = np.random.default_rng(args.seed)

    # ---- eval scaffold: stored grasps + per-task conditioning ----
    rec = pd.read_parquet(REPO / f"analysis/value/repsep/cf_records_psm_psm_o1000_ms{args.seed}.parquet")
    ocols = [c for c in rec.columns if c.startswith("obs_")]
    grasp = torch.as_tensor(rec[ocols].values, dtype=torch.float32)
    task_cond = {}
    rb = buf.sample(10_000)
    for ti, task in enumerate(ALL_TASKS[DOMAIN], 1):
        if args.cond == "z":
            r = get_relabel_fn(DOMAIN, task)(rb["next"]["physics"].cpu().numpy(),
                                             rb["action"].cpu().numpy()) + 1.0
            z = m.reward_inference(next_obs=rb["next"]["observation"].float(),
                                   reward=torch.as_tensor(r, dtype=torch.float32))
            task_cond[ti] = z.detach().reshape(-1)
        else:  # cond=goal: the task's actual goal observation
            env, _ = create_ogbench_env(task, seed=args.seed, obs_type="state")
            _, info = env.reset(); env.close()
            g = torch.as_tensor(info["goal"], dtype=torch.float32).reshape(1, -1)
            task_cond[ti] = goalrep(g).detach().reshape(-1)

    def cf_auc():
        aucs = []
        for ti in sorted(task_cond):
            sel = torch.as_tensor((rec["task"] == ti).values)
            o = grasp[sel]; c = task_cond[ti].unsqueeze(0).expand(o.shape[0], -1)
            with torch.no_grad():
                v = Vnet(inp(o), c)
            y = torch.as_tensor(rec.loc[(rec["task"] == ti).values, "success"].values.astype(np.float32))
            aucs.append(value_auc(v, y > 0))
        return float(np.nanmean(aucs)), [round(a, 3) for a in aucs]

    hist = []
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, nB, args.batch)
        s = torch.as_tensor(store["observation"][idx], dtype=torch.float32)
        a = torch.as_tensor(store["action"][idx], dtype=torch.float32)
        sn = torch.as_tensor(store["next/observation"][idx], dtype=torch.float32)
        phys_n = torch.as_tensor(store["next/physics"][idx], dtype=torch.float32)
        # goals
        if args.goal_mode == "traj":
            jt = sample_future_idx(idx, ep_last, args.gamma, rng)
            ju = rng.integers(0, nB, args.batch)
            j = np.where(rng.random(args.batch) < P_TRAJ, jt, ju)
        else:
            j = rng.integers(0, nB, args.batch)
        g_obs = torch.as_tensor(store["next/observation"][j], dtype=torch.float32)
        g_phys = torch.as_tensor(store["next/physics"][j], dtype=torch.float32)
        c = goalrep(g_obs)                                    # conditioning (per-goal, both modes)
        r = (phys_n[:, CUBE_PHYS] - g_phys[:, CUBE_PHYS]).norm(dim=-1).lt(GOAL_THRESH).float()
        mask = 1 - r
        Is, Isn = inp(s), inp(sn)
        with torch.no_grad():
            tq = Qt(Is, a, c)
        v = Vnet(Is, c); lV = expectile_loss(tq - v, args.expectile)
        oV.zero_grad(); lV.backward(); oV.step()
        with torch.no_grad():
            y = r + args.gamma * mask * Vnet(Isn, c)
        q = Qnet(Is, a, c); lQ = (q - y).pow(2).mean()
        oQ.zero_grad(); lQ.backward(); oQ.step()
        with torch.no_grad():
            for p, pt in zip(Qnet.parameters(), Qt.parameters()):
                pt.mul_(1 - args.tau_target).add_(args.tau_target * p)
        if step % args.eval_every == 0 or step == args.steps:
            auc, per = cf_auc()
            hist.append({"step": step, "cf_auc": auc, "per_task": per, "pos": float(r.mean())})
            print(f"[s{args.seed} {args.input}/{args.cond}/{args.goal_mode}] step {step}: "
                  f"cf_auc={auc:.3f} per={per} pos={float(r.mean()):.3f}", flush=True)

    outdir = OUT / f"{args.input}_{args.cond}_{args.goal_mode}"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"s{args.seed}.json").write_text(json.dumps(
        {"cfg": vars(args), "history": hist, "final_cf_auc": hist[-1]["cf_auc"]}, indent=2))
    print(f"[s{args.seed}] DONE {args.input}/{args.cond}/{args.goal_mode} final={hist[-1]['cf_auc']:.3f}")


if __name__ == "__main__":
    main()
