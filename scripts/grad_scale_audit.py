#!/usr/bin/env python
"""scripts/grad_scale_audit.py — does the measure loss actually shape B/phi, or is
ortho_coef=1000 'practically switching FB off'?

For each checkpoint along training, replicate the agent's OWN backward-map loss
construction on real batches and separate the gradient on the backward-map (B / phi)
parameters into its two linear components:

    g_total = g_measure + ortho_coef * g_ortho

Reported per checkpoint (mean over batches): ||g_measure||, ortho_coef*||g_ortho||,
their ratio, and cos(g_measure, g_ortho). If the ratio >> 1 through training, the
representation-shaping is whitening; the measure term is a perturbation.
(NB the F/psi side gets NO ortho gradient at all — it is pure measure/TD — so this
audit concerns B/phi only, i.e. exactly the basis-shaping question.)

CPU-only, no optimizer steps, no mutation (model in eval mode).
Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python scripts/grad_scale_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.ogbench import load_ogbench_dataset
from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis" / "grad_audit"
DATA = "/dev/shm/factored-fb/datasets"
DOMAIN = "cube-single-play-v0"
N_BATCHES = 8

METHODS = {
    "fb_plain": "/dev/shm/factored-fb/runs/20260607_070225__fb_state_plain__s0",
    "psm_o1000": "/dev/shm/factored-fb_outputs/psm_state_orthohi__ortho_coef1000__lr_phi1e-5__s0",
    "psm_tuned": "/dev/shm/factored-fb_outputs/psm_state_tuned__ortho_coef1__lr_phi1e-5__s0",
}


def flat_grad(loss, params, retain=True):
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def fb_terms(agent, batch):
    """Replicates agents/fb/agent.py update()/update_fb() B-loss construction."""
    model = agent.model
    obs = batch["observation"].float()
    action = batch["action"].float()
    next_obs = batch["next"]["observation"].float()
    term = batch["next"]["terminated"].float()
    discount = (agent.discount * (1.0 - term)).reshape(-1, 1)
    from agents.fb.agent import fb_successor_terms, ortho_cov

    with torch.no_grad():
        obs_n = model._obs_normalizer(obs)
        next_n = model._obs_normalizer(next_obs)
    obs_e, next_e, goal = agent.enc(obs_n, next_n)
    z = agent.sample_mixed_z(train_goal=goal.detach()).clone()
    zf = agent._fb_z(z)

    with torch.no_grad():
        next_left = model._target_left_encoder(next_e)
        actor_in = next_left if agent.actor_encode_obs else next_e
        next_action = agent.sample_action_from_norm_obs(actor_in, z)
        tFs = model._target_forward_map(next_left, zf, next_action)
        tB = model._target_backward_map(goal)
        tMs = torch.matmul(tFs, tB.T)
        _, _, tM = agent.get_targets_uncertainty(tMs, agent.fb_pessimism_penalty)

    left = model._left_encoder(obs_e)
    Fs = model._forward_map(left, zf, action)
    B = model._backward_map(goal)
    Ms = torch.matmul(Fs, B.T)
    diff = Ms - discount * tM
    fb_offdiag, fb_diag = fb_successor_terms(diff, agent.off_diag, agent.off_diag_sum, None, False)
    measure = fb_offdiag + fb_diag

    Cov = ortho_cov(B, None)
    orth = 0.5 * (Cov * agent.off_diag).pow(2).sum() / agent.off_diag_sum - Cov.diag().mean()
    return measure, orth, list(model._backward_map.parameters())


def psm_terms(agent, batch):
    """Replicates agents/psm/agent.py _update_psm() phi-loss construction."""
    model = agent.model
    obs = batch["observation"].float()
    action = batch["action"].float()
    next_obs = batch["next"]["observation"].float()
    term = batch["next"]["terminated"].float()
    discount = (agent.discount * (1.0 - term)).reshape(-1, 1)
    bsz = obs.shape[0]

    with torch.no_grad():
        obs_n = model._obs_normalizer(obs)
        next_n = model._obs_normalizer(next_obs)
    obs_enc = agent._enc_obs(obs_n).detach()
    next_enc = agent._enc_obs(next_n).detach()
    goal = agent._enc_goal(next_n)
    z = model.sample_z_psm(bsz, device="cpu")
    hash_ = torch.arange(bsz)

    with torch.no_grad():
        next_action = model.proto_sampler(hash_, z)
        tpsis = model.target_psm_psi(next_enc, z, next_action)
        tphi = model.target_phi(goal)
        tMs = torch.matmul(tpsis, tphi.T)
        tmean, tunc = model.get_targets_uncertainty(tMs)
        tM = tmean - agent.pessimism_penalty * tunc

    psis = model.psm_psi(obs_enc, z, action)
    phi = model.phi(goal)
    Ms = torch.matmul(psis, phi.T)
    diff = Ms - discount * tM
    psm_offdiag = 0.5 * (diff * agent.off_diag).pow(2).sum() / agent.off_diag_sum
    psm_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
    measure = psm_offdiag + psm_diag

    Cov = torch.matmul(phi, phi.T)
    orth = 0.5 * (Cov * agent.off_diag).pow(2).sum() / agent.off_diag_sum - Cov.diag().mean()
    # phi params = the bw branch (phi net [+ bw encoder, Identity for state])
    return measure, orth, list(model.phi.parameters() if hasattr(model.phi, "parameters")
                               else model._phi.parameters())


def audit(method, rundir, buf):
    rundir = Path(rundir)
    cfg = load_cfg(str(rundir / ".hydra/config.yaml"), device="cpu")
    ckpts = sorted(rundir.glob("checkpoints/step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    rows = []
    for ck in ckpts:
        _, agent, _, _ = build_env_and_agent(cfg)
        load_checkpoint(agent, str(ck))
        agent.model.train(False)
        C = float(agent.ortho_coef)
        terms_fn = fb_terms if method.startswith("fb") else psm_terms
        nm, no, cos = [], [], []
        torch.manual_seed(0)
        for _ in range(N_BATCHES):
            batch = buf.sample(int(cfg.batch_size))
            measure, orth, params = terms_fn(agent, batch)
            gm = flat_grad(measure, params, retain=True)
            go = flat_grad(orth, params, retain=False)
            nm.append(float(gm.norm()))
            no.append(C * float(go.norm()))
            cos.append(float(torch.nn.functional.cosine_similarity(
                gm.unsqueeze(0), go.unsqueeze(0))))
        step = int(ck.stem.split("_")[1])
        row = {"step": step, "g_measure": float(np.mean(nm)),
               "C_g_ortho": float(np.mean(no)),
               "ratio_ortho_over_measure": float(np.mean(no) / max(np.mean(nm), 1e-12)),
               "cos(g_m,g_o)": float(np.mean(cos)), "ortho_coef": C}
        rows.append(row)
        print(f"  {method} @{step}: ||g_meas||={row['g_measure']:.3e} "
              f"C||g_orth||={row['C_g_ortho']:.3e} ratio={row['ratio_ortho_over_measure']:.2f} "
              f"cos={row['cos(g_m,g_o)']:.3f}", flush=True)
    return rows


def main():
    buf = load_ogbench_dataset(domain=DOMAIN, data_path=DATA, obs_type="state",
                               load_n_episodes=1000)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for method, rundir in METHODS.items():
        print(f"[audit] {method}")
        all_rows[method] = audit(method, rundir, buf)
    (OUT / "grad_scale_audit.json").write_text(json.dumps(all_rows, indent=2))
    print(f"wrote {OUT}/grad_scale_audit.json")


if __name__ == "__main__":
    main()
