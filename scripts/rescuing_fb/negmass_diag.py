"""D1 negative-MASS diagnostic (ledger §9.15, gate G1).

The point-count negativity (neg_frac ~0.45-0.53) understates or overstates the
defect depending on where the negative entries sit. This measures the share of
ABSOLUTE MASS that is negative, its magnitude relative to the positive entries,
and its distribution over data-density quartiles — the numbers that decide
whether the object is "approximately a measure with sign noise" or structurally
signed. 2026-07-29 baseline (p2_rzK50): neg mass share 0.27-0.29, WORSE in high-
density quartiles ⇒ truncated-basis ringing, not fringe noise.

Query convention matches torchfb_full_measure.py: M(s0,a0,g) = F(s0,z,a0)·B(g),
z = project_z(B(g_rand)), a0 = the actor's own action. Gate G1: point neg_frac
<= 0.10 AND neg mass share <= 0.05.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def region_density(spec, ref_coords, centers) -> np.ndarray:
    """rho(region(center)) for each center — the conjunctive generalization of
    the old single-ball KDTree count (spec 14 §3.4)."""
    from full_measure import FullMeasure
    fm = FullMeasure(ref_coords, spec=spec, floor_counts=0.0)
    return fm.nhat(centers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="hydra run dir")
    ap.add_argument("--ckpt", default="checkpoints/final.pt")
    ap.add_argument("--starts", type=int, default=32)
    ap.add_argument("--goals", type=int, default=20000)
    ap.add_argument("--rho-samples", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    from evals.analysis import load_cfg, load_checkpoint
    from envs.ogbench import create_ogbench_env
    from train import make_agent

    cfg = load_cfg(f"{a.run}/.hydra/config.yaml", device=a.device)
    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type)
    agent = make_agent(cfg, env.observation_space, int(env.action_space.shape[0]))
    load_checkpoint(agent, f"{a.run}/{a.ckpt}", map_location=a.device)
    agent.model.eval()
    dev = a.device

    from env_geometry import geometry_for, resolve_dataset
    geo = geometry_for(cfg.domain, getattr(cfg, "hit_spec", "") or "")
    spec = geo.hit_spec

    npz = np.load(resolve_dataset(cfg.data_path, cfg.domain))
    obs_all = npz["observations"]
    gidx = rng.choice(len(obs_all), a.goals, replace=False)
    goals = torch.as_tensor(obs_all[gidx], device=dev)
    cube = spec.coords(obs_all[gidx])

    ref = spec.coords(obs_all[rng.choice(len(obs_all), a.rho_samples, replace=False)])
    nhat = region_density(spec, ref, cube)
    qcut = np.quantile(nhat, [0.25, 0.5, 0.75])
    quart = np.digitize(nhat, qcut)

    with torch.no_grad():
        Bg = agent.model.backward_map(goals)

    srows = rng.choice(len(obs_all), a.starts, replace=False)
    grows = rng.choice(len(obs_all), a.starts, replace=False)
    neg_frac, neg_mass, mag_neg, mag_pos, neg_by_q = [], [], [], [], []
    for s_row, g_row in zip(srows, grows):
        s0 = torch.as_tensor(obs_all[s_row][None], device=dev)
        with torch.no_grad():
            z = agent.model.project_z(agent.model.backward_map(
                torch.as_tensor(obs_all[g_row][None], device=dev)))
            a0 = agent.act(s0, z, eval_mode=True)
            F0 = agent.model.forward_map(s0, z, a0).mean(0).squeeze(0)
            vals = (Bg @ F0).double().cpu().numpy()
        neg = vals < 0
        neg_frac.append(float(neg.mean()))
        neg_mass.append(float(np.abs(vals[neg]).sum() / np.abs(vals).sum()))
        mag_neg.append(float(np.abs(vals[neg]).mean()) if neg.any() else 0.0)
        mag_pos.append(float(np.abs(vals[~neg]).mean()))
        neg_by_q.append([float(neg[quart == k].mean()) for k in range(4)])

    res = {
        "run": a.run,
        "env": cfg.domain,
        "hit_spec": ",".join(f"{a_}:{b_}:{e_}" for a_, b_, e_ in spec.blocks),
        "neg_frac": float(np.mean(neg_frac)),
        "neg_mass_share": float(np.mean(neg_mass)),
        "mag_neg": float(np.mean(mag_neg)),
        "mag_pos": float(np.mean(mag_pos)),
        "neg_frac_by_density_quartile": np.array(neg_by_q).mean(0).round(4).tolist(),
        "gate_G1": bool(np.mean(neg_frac) <= 0.10 and np.mean(neg_mass) <= 0.05),
        "starts": a.starts, "goals": a.goals, "seed": a.seed,
    }
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
