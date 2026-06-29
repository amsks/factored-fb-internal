#!/usr/bin/env python
"""scripts/z_inference_ladder.py — error-aware task inference on existing checkpoints (B3).

For each (method, seed, task): build the SAME relabel batch the evaluator uses,
compute the task vector z under the inference ladder (evals/z_inference.py):
  L0 = E[rB] (current rule)   L1 = GLS projection   L2 = Bellman-consistent z
then roll out the trained policy conditioned on each z and report success, plus a
no-rollout diagnostic (value rank-AUC for r>0 states under each z).

No retraining — pure inference-time intervention. CPU-only. Uses step_1000000.pt
explicitly (comparable with the paper's 1M numbers).

Run:  CUDA_VISIBLE_DEVICES= PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python \
        scripts/z_inference_ladder.py --method fb_plain --seeds 0,1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.ogbench import load_ogbench_dataset
from envs.ogbench import ALL_TASKS, create_ogbench_env, get_relabel_fn
from envs.rollout import rollout
from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
from evals.z_inference import infer_l0, infer_l1, infer_l2, project_sphere, value_auc

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis" / "z_inference"
DOMAIN = "cube-single-play-v0"
DATA = "/dev/shm/factored-fb/datasets"
SHIFT_REWARD = 1.0  # mirrors OGBenchEvaluator default (relabel -1/0 -> 0/1)

METHODS = {
    # name -> (run-dir pattern with {s}, embed kind)
    "fb_plain": ("/dev/shm/factored-fb/runs/20260607_070225__fb_state_plain__s{s}", "fb"),
    "psm_o1000": ("/dev/shm/factored-fb_outputs/psm_state_orthohi__ortho_coef1000__lr_phi1e-5__s{s}", "psm"),
    "psm_tuned": ("/dev/shm/factored-fb_outputs/psm_state_tuned__ortho_coef1__lr_phi1e-5__s{s}", "psm"),
    # IWR runs: B whitened under rho' but eval relabeling samples rho -> the L1/GLS
    # correction is the targeted test for the IWR success drop (fb-iwr.tex §5.3).
    "fb_state_iwr": ("/dev/shm/factored-fb/runs/fb_state_iwr__s{s}", "fb"),
    "fb_state_iwr_wz": ("/dev/shm/factored-fb/runs/fb_state_iwr_wz__s{s}", "fb"),
}


def embed_goals(model, kind, next_obs):
    with torch.no_grad():
        if kind == "fb":
            return model.backward_map(next_obs)
        return model.phi(model._bw_encoder(model._normalize(next_obs)))


def make_q_pair_fn(model, kind, obs, act, next_obs, z_ref, refresh_every=50):
    """Closure returning (Q(s,a,z), Q(s',a',z)) differentiable in z; next actions
    are produced by the trained actor (semi-gradient) and refreshed periodically."""
    state = {"step": 0, "a_next": None}

    def next_actions(z):
        with torch.no_grad():
            zb = project_sphere(z.detach()).unsqueeze(0).expand(next_obs.shape[0], -1)
            return model.act(next_obs, zb, mean=True)

    def q_of(o, a, zb):
        if kind == "fb":
            F = model.forward_map(o, zb, a)            # [P,N,d]
        else:
            enc = model._fw_encoder(model._normalize(o))
            F = model.sf_psi(enc, zb, a)               # [P,N,d]
        return (F * zb).sum(-1).mean(0)                # mean over parallel heads

    def q_pair(z):
        if state["a_next"] is None or state["step"] % refresh_every == 0:
            state["a_next"] = next_actions(z)
        state["step"] += 1
        zb = z.unsqueeze(0).expand(obs.shape[0], -1)
        return q_of(obs, act, zb), q_of(next_obs, state["a_next"], zb)

    return q_pair


def diagnostic_auc(model, kind, next_obs, r, z):
    """Rank-AUC of V_z(s') = Q(s', pi(s',z), z) for rewarding vs non-rewarding states."""
    with torch.no_grad():
        zb = z.unsqueeze(0).expand(next_obs.shape[0], -1)
        a = model.act(next_obs, zb, mean=True)
        if kind == "fb":
            F = model.forward_map(next_obs, zb, a)
        else:
            enc = model._fw_encoder(model._normalize(next_obs))
            F = model.sf_psi(enc, zb, a)
        V = (F * zb).sum(-1).mean(0)
    return value_auc(V, torch.as_tensor(r > 0))


def run_seed(method, seed, args, buf):
    pattern, kind = METHODS[method]
    rundir = Path(pattern.format(s=seed))
    cfg = load_cfg(str(rundir / ".hydra/config.yaml"), device="cpu")
    _, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, str(rundir / "checkpoints" / args.ckpt))
    model = agent.model
    model.train(False)
    gamma = float(cfg.discount)

    results = {}
    for task in ALL_TASKS[DOMAIN]:
        torch.manual_seed(1234 + seed)
        np.random.seed(1234 + seed)
        batch = buf.sample(args.relabel_size)
        obs = batch["observation"].float()
        act = batch["action"].float()
        next_obs = batch["next"]["observation"].float()
        term = batch["next"]["terminated"].float().reshape(-1)
        relabel_fn = get_relabel_fn(DOMAIN, task)
        r = relabel_fn(batch["next"]["physics"].cpu().numpy(), act.cpu().numpy()) + SHIFT_REWARD
        r_t = torch.as_tensor(r, dtype=torch.float32).reshape(-1)
        disc = gamma * (1.0 - term)

        B = embed_goals(model, kind, next_obs)
        zs = {
            "L0": project_sphere(infer_l0(B, r_t)),
            "L1": project_sphere(infer_l1(B, r_t)),
        }
        if "L2" in args.rules or "L2b" in args.rules:
            sub = torch.randperm(len(obs))[: args.l2_batch]
            rs, ds = r_t[sub], disc[sub]
            if "L2" in args.rules:
                qfn = make_q_pair_fn(model, kind, obs[sub], act[sub], next_obs[sub], zs["L0"])
                z2, info = infer_l2(zs["L0"], qfn, rs, ds, steps=args.l2_steps, lr=args.l2_lr)
                zs["L2"] = z2
                results.setdefault("l2_info", {})[task] = info
            if "L2b" in args.rules:
                # reward-balanced residual: positives weighted to half the total mass,
                # so the sparse goal term isn't drowned by the 98% zero-reward states
                pos = (rs > 0).float()
                n_pos = float(pos.sum().clamp_min(1))
                w = torch.where(pos > 0, len(rs) / (2 * n_pos), torch.tensor(0.5)) \
                    * len(rs) / (torch.where(pos > 0, len(rs) / (2 * n_pos),
                                             torch.tensor(0.5)).sum())
                qfn_b = make_q_pair_fn(model, kind, obs[sub], act[sub], next_obs[sub], zs["L0"])
                z2b, info_b = infer_l2(zs["L0"], qfn_b, rs, ds, steps=args.l2_steps,
                                       lr=args.l2_lr, sample_weight=w)
                zs["L2b"] = z2b
                results.setdefault("l2b_info", {})[task] = info_b

        task_res = {}
        for rule, z in zs.items():
            if rule not in args.rules:
                continue
            auc = diagnostic_auc(model, kind, next_obs, r_t, z)
            env, _ = create_ogbench_env(task, seed=seed, obs_type="state")
            try:
                stats, infos, _ = rollout(env, agent, args.episodes, ctx=z, record=False)
            finally:
                env.close()
            succ = float(np.mean([any(si.get("success", False) for si in ep) for ep in infos]))
            cos_l0 = float(torch.nn.functional.cosine_similarity(
                z.reshape(1, -1), zs["L0"].reshape(1, -1)))
            task_res[rule] = {"success": succ, "value_auc": auc, "cos_to_L0": cos_l0}
            print(f"    {task.split('-')[-2]} {rule}: succ={succ:.2f} auc={auc:.3f} cos_L0={cos_l0:.3f}")
        results[task] = task_res
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--rules", default="L0,L1,L2,L2b")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--relabel-size", type=int, default=10000)
    ap.add_argument("--l2-batch", type=int, default=4096)
    ap.add_argument("--l2-steps", type=int, default=200)
    ap.add_argument("--l2-lr", type=float, default=0.05)
    ap.add_argument("--ckpt", default="step_1000000.pt")
    args = ap.parse_args()
    args.rules = args.rules.split(",")

    buf = load_ogbench_dataset(domain=DOMAIN, data_path=DATA, obs_type="state",
                               load_n_episodes=1000)
    outdir = OUT / args.method
    outdir.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in args.seeds.split(",")]:
        print(f"[{args.method}] seed {s}")
        res = run_seed(args.method, s, args, buf)
        (outdir / f"s{s}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
