#!/usr/bin/env python
"""scripts/mlp_readout_control.py — the matched-readout ablation (bottleneck B2).

Claim under test: PSM's basis contains the controllability signal but the bilinear
readout psi(s,z)·z cannot read it (cf-AUC 0.49 vs GCIQL-MLP 0.89). Control: train a
small MLP value Q(norm_obs(s), a, z) with the SAME conditioning z = project(phi(g))
from the FROZEN PSM-o1000 checkpoint, on the SAME data distribution PSM trains on
(uniform buffer goals + a g=s' "diagonal" fraction), via IQL-style expectile TD.
Then score the stored counterfactual grasps (analysis/value/repsep/cf_records_*) —
no rollouts needed. If MLP-on-frozen-basis recovers high cf-AUC, the bilinear
readout form is causally the bottleneck; if it stays at chance, the deficit is
upstream (data/objective), not the readout form.

CPU-only. Run:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python scripts/mlp_readout_control.py --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.ogbench import load_ogbench_dataset
from envs.ogbench import ALL_TASKS, get_relabel_fn
from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
from evals.z_inference import value_auc

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis" / "readout_control"
DOMAIN = "cube-single-play-v0"
DATA = "/dev/shm/factored-fb/datasets"
PSM_RUN = "/dev/shm/factored-fb_outputs/psm_state_orthohi__ortho_coef1000__lr_phi1e-5__s{s}"
CUBE_PHYS = slice(14, 17)
GOAL_THRESH = 0.04          # metres, cube-to-goal-cube success radius (probe convention)
P_DIAG = 0.2                # fraction of goals g = s' (FB diagonal-term analogue)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff.pow(2)).mean()


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, *xs):
        return self.net(torch.cat(xs, dim=-1)).squeeze(-1)


def frozen_embed(model, obs):
    with torch.no_grad():
        return model.phi(model._bw_encoder(model._normalize(obs)))


def frozen_norm(model, obs):
    with torch.no_grad():
        return model._normalize(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--expectile", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--tau-target", type=float, default=0.005)
    ap.add_argument("--ckpt", default="step_1000000.pt")
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--goal-mode", default="uniform", choices=["uniform", "traj"],
                    help="uniform = FB-style batch-perm goals (+P_DIAG g=s'); "
                         "traj = GCIQL-style discounted trajectory futures")
    ap.add_argument("--p-traj", type=float, default=0.625,
                    help="traj mode: fraction of trajectory-future goals (rest uniform)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(6)

    rundir = Path(PSM_RUN.format(s=args.seed))
    cfg = load_cfg(str(rundir / ".hydra/config.yaml"), device="cpu")
    _, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, str(rundir / "checkpoints" / args.ckpt))
    model = agent.model
    model.train(False)

    buf = load_ogbench_dataset(domain=DOMAIN, data_path=DATA, obs_type="state",
                               load_n_episodes=1000)
    obs_dim = buf.sample(2)["observation"].shape[-1]
    z_dim = int(cfg.z_dim)
    act_dim = buf.sample(2)["action"].shape[-1]

    qnet = MLP(obs_dim + act_dim + z_dim)
    vnet = MLP(obs_dim + z_dim)
    q_targ = copy.deepcopy(qnet)
    opt_q = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(vnet.parameters(), lr=args.lr)

    # ---- counterfactual eval setup: stored grasps + per-task inferred z (L0) ----
    rec = pd.read_parquet(
        REPO / f"analysis/value/repsep/cf_records_psm_psm_o1000_ms{args.seed}.parquet")
    obs_cols = [c for c in rec.columns if c.startswith("obs_")]
    grasp_obs = torch.as_tensor(rec[obs_cols].values, dtype=torch.float32)
    task_zs = {}
    rb = buf.sample(10_000)
    for ti, task in enumerate(ALL_TASKS[DOMAIN], start=1):
        relabel_fn = get_relabel_fn(DOMAIN, task)
        r = relabel_fn(rb["next"]["physics"].cpu().numpy(), rb["action"].cpu().numpy()) + 1.0
        z = model.reward_inference(next_obs=rb["next"]["observation"].float(),
                                   reward=torch.as_tensor(r, dtype=torch.float32))
        task_zs[ti] = z.detach().reshape(-1)

    def cf_auc():
        aucs = []
        for ti in sorted(task_zs):
            m = rec["task"] == ti
            o = grasp_obs[torch.as_tensor(m.values)]
            zb = task_zs[ti].unsqueeze(0).expand(o.shape[0], -1)
            with torch.no_grad():
                a = model.act(o, zb, mean=True)
                q = qnet(frozen_norm(model, o), a.float(), zb)
            y = torch.as_tensor(rec.loc[m, "success"].values.astype(np.float32))
            aucs.append(value_auc(q, y > 0))
        return float(np.nanmean(aucs)), [round(a, 3) for a in aucs]

    # ---- training loop ----
    # goal modes: "uniform" = FB-style batch-perm goals + P_DIAG of g=s';
    # "traj" = GCIQL-style: p_traj discounted trajectory-futures, rest uniform.
    from evals.traj_goals import episode_last_index, sample_future_idx
    rng = np.random.default_rng(args.seed)
    store = buf._storage
    ep_last = episode_last_index(store["timestep"])
    n_buf = len(store["timestep"])

    history = []
    for step in range(1, args.steps + 1):
        if args.goal_mode == "traj":
            idx = rng.integers(0, n_buf, args.batch)
            s = torch.as_tensor(store["observation"][idx], dtype=torch.float32)
            a = torch.as_tensor(store["action"][idx], dtype=torch.float32)
            sn = torch.as_tensor(store["next/observation"][idx], dtype=torch.float32)
            phys_n = torch.as_tensor(store["next/physics"][idx], dtype=torch.float32)
            jt = sample_future_idx(idx, ep_last, args.gamma, rng)         # traj futures
            ju = rng.integers(0, n_buf, args.batch)                       # uniform goals
            use_traj = rng.random(args.batch) < args.p_traj
            j = np.where(use_traj, jt, ju)
            g_obs = torch.as_tensor(store["next/observation"][j], dtype=torch.float32)
            g_phys = torch.as_tensor(store["next/physics"][j], dtype=torch.float32)
        else:
            b = buf.sample(args.batch)
            s = b["observation"].float(); a = b["action"].float()
            sn = b["next"]["observation"].float()
            phys_n = b["next"]["physics"].float()
            # goals: permute next-states within batch (FB's train_goal), then overwrite
            # a P_DIAG fraction with g = s' (FB diagonal analogue, guarantees positives)
            perm = torch.randperm(args.batch)
            g_obs = sn[perm].clone(); g_phys = phys_n[perm].clone()
            diag = torch.rand(args.batch) < P_DIAG
            g_obs[diag] = sn[diag]; g_phys[diag] = phys_n[diag]
        z = model.project_z(frozen_embed(model, g_obs))
        dist = (phys_n[:, CUBE_PHYS] - g_phys[:, CUBE_PHYS]).norm(dim=-1)
        r = (dist < GOAL_THRESH).float()
        mask = 1.0 - r                                   # absorb at goal
        ns, na, nz = frozen_norm(model, s), a, z
        nsn = frozen_norm(model, sn)

        with torch.no_grad():
            target_q = q_targ(ns, na, nz)
        v = vnet(ns, nz)
        loss_v = expectile_loss(target_q - v, args.expectile)
        opt_v.zero_grad(); loss_v.backward(); opt_v.step()

        with torch.no_grad():
            v_next = vnet(nsn, nz)
            y = r + args.gamma * mask * v_next
        q = qnet(ns, na, nz)
        loss_q = (q - y).pow(2).mean()
        opt_q.zero_grad(); loss_q.backward(); opt_q.step()

        with torch.no_grad():
            for p, pt in zip(qnet.parameters(), q_targ.parameters()):
                pt.mul_(1 - args.tau_target).add_(args.tau_target * p)

        if step % args.eval_every == 0 or step == args.steps:
            auc, per_task = cf_auc()
            history.append({"step": step, "cf_auc": auc, "per_task": per_task,
                            "loss_q": float(loss_q), "loss_v": float(loss_v),
                            "r_pos_frac": float(r.mean())})
            print(f"[s{args.seed}] step {step}: cf_auc={auc:.3f} per_task={per_task} "
                  f"q_loss={float(loss_q):.4f} pos={float(r.mean()):.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    bilinear = float(np.nanmean([value_auc(
        torch.as_tensor(rec.loc[rec.task == ti, "value"].values),
        torch.as_tensor(rec.loc[rec.task == ti, "success"].values.astype(np.float32)) > 0)
        for ti in sorted(task_zs)]))
    out = {"seed": args.seed, "goal_mode": args.goal_mode, "history": history,
           "final_cf_auc": history[-1]["cf_auc"],
           "bilinear_cf_auc_same_grasps": bilinear}
    suffix = "" if args.goal_mode == "uniform" else f"_{args.goal_mode}"
    (OUT / f"s{args.seed}{suffix}.json").write_text(json.dumps(out, indent=2))
    print(f"[s{args.seed}] DONE final cf_auc={history[-1]['cf_auc']:.3f} "
          f"(bilinear on same grasps: {bilinear:.3f})")


if __name__ == "__main__":
    main()
