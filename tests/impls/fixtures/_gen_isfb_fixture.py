"""THROWAWAY fixture generator — imports the TORCH reference (agents/fb/isfb/) to bake a
float64 SG-ISFB reference. NOT run in CI; the only committed file permitted to import agents/.

Run with a TORCH interpreter. The repo's own .venv is EMPTY (zero packages) and the jax venv
has no torch, so neither works; use the dedicated torch venv:
  cd /u/amsks/git/Factored-FB && \
    /tmp/factored-fb-venv-torch/bin/python tests/impls/fixtures/_gen_isfb_fixture.py
(the parity test itself runs under /tmp/factored-fb-venv-jax/bin/python, which has no torch)

NOTE: the repo-root `utils.py` shadows package imports, so we sys.path.insert the repo root
explicitly rather than relying on PYTHONPATH (same pattern as _gen_gciql_fixture.py).

Fixture shape (deliberate — see .superpowers/sdd/phase1-isfb-scout.md §8):
  P (num_parallel) = 3  and  z_dim = 8, and every other dimension distinct (B=16, obs=5,
  act=3, L=6). The scout flagged a cluster of scale-factor risks that are individually
  invisible: the 1/sqrt(d) that torch omits and the jax shim applies; L_M head-SUMMED vs
  head-AVERAGED; and the xP factor on L_V and L_N. With P == 1 or z_dim == 1 (or with
  P == z_dim) those factors collapse to 1 or to each other and the disagreement goes
  untested. P=3 also makes get_targets_uncertainty's /(P^2-P) = 1/6 divisor distinguishable
  from a /P or /P^2 mistake, and fb_pessimism_penalty is set NON-ZERO so the head-spread
  reduction is actually exercised.

Every loss term is dumped SEPARATELY (L_V, L_N, L_M, ortho, total) for all three
measure_insample modes, so a mis-scaled term cannot be masked by a compensating one.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import gymnasium

from agents.fb.isfb.agent import ISFBAgent
from nn_models import (BackwardArchiConfig, ForwardArchiConfig, IdentityNNConfig,
                       NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig)
from normalizers import IdentityNormalizerConfig

# --- shape / hyperparameter choices (all distinct; see module docstring) ---
B, OBS, ACT, Z, L, P, H = 16, 5, 3, 8, 6, 3, 32
DISCOUNT = 0.99
EXPECTILE = 0.7
ORTHO_COEF = 1.0
PESSIMISM = 0.5          # non-zero: exercises the head-spread reduction
ALPHA_N, ALPHA_M = 0.7, 1.3
AWR_BETA, AWR_WMAX = 3.0, 100.0
MODES = ("coord", "scalar_gate_hard", "scalar_gate_awr")

# Weight-map state_dict prefixes we bake (the actor is not part of the critic parity).
MAP_PREFIXES = ("_forward_map.", "_value_forward_map.", "_backward_map.", "_left_encoder.",
                "_target_forward_map.", "_target_value_forward_map.",
                "_target_backward_map.", "_target_left_encoder.")


def build_agent(mode):
    torch.manual_seed(0)
    return ISFBAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(OBS,), dtype=np.float32),
        action_dim=ACT, z_dim=Z, L_dim=L, batch_size=B,
        forward_cfg=ForwardArchiConfig(hidden_dim=H, hidden_layers=2, embedding_layers=2,
                                       num_parallel=P),
        backward_cfg=BackwardArchiConfig(hidden_dim=H, hidden_layers=4, norm=True),
        left_encoder_cfg=BackwardArchiConfig(hidden_dim=H, hidden_layers=4, norm=True),
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        discount=DISCOUNT, ortho_coef=ORTHO_COEF, fb_pessimism_penalty=PESSIMISM,
        iql_expectile=EXPECTILE, f_target_tau=0.005, b_target_tau=0.005,
        measure_insample=mode, alpha_n=ALPHA_N, alpha_m=ALPHA_M,
        awr_beta=AWR_BETA, awr_wmax=AWR_WMAX,
        device="cpu",
    )


def make_inputs():
    g = torch.Generator().manual_seed(7)
    obs = torch.randn(B, OBS, generator=g)
    action = torch.rand(B, ACT, generator=g) * 2 - 1
    next_obs = torch.randn(B, OBS, generator=g)
    z = torch.randn(B, Z, generator=g)
    z = np.sqrt(Z) * z / z.norm(dim=-1, keepdim=True)      # project_z(norm_z=True)
    # masks in OGBench semantics (0 => bootstrap cut). The torch update_fb takes the
    # already-multiplied per-row discount [B,1]; our jax critic forms discount*masks[:,None].
    masks = (torch.rand(B, generator=g) < 0.75).to(torch.float64)
    return dict(observations=obs, actions=action, next_observations=next_obs, z=z, masks=masks)


def intermediates(agent, inp):
    """Recompute the pre-loss quantities so a parity failure localizes to a tensor."""
    obs, action, next_obs, z = (inp["observations"], inp["actions"],
                                inp["next_observations"], inp["z"])
    goal = next_obs
    zero_a = torch.zeros_like(action)
    m = agent.model
    with torch.no_grad():
        nle = m._target_left_encoder(next_obs)
        tFv = m._target_value_forward_map(nle, z, zero_a)
        tB = m._target_backward_map(goal)
        tN = torch.matmul(tFv, tB.T)
        _, _, tN_M = agent.get_targets_uncertainty(tN, PESSIMISM)

        le = m._left_encoder(obs)
        Fs = m._forward_map(le, z, action)
        Fv = m._value_forward_map(le, z, zero_a)
        Bm = m._backward_map(goal)
        M = torch.matmul(Fs, Bm.T)
        N = torch.matmul(Fv, Bm.T)
        q = (Fs * z).sum(-1)
        v = (Fv * z).sum(-1)
    return dict(le=le, Fs=Fs, Fv=Fv, B=Bm, M=M, N=N, tN=tN, tN_M=tN_M, q=q, v=v)


def main():
    inp = make_inputs()
    master = build_agent(MODES[0])
    sd = copy.deepcopy(master.model.state_dict())

    out = {}
    for k, t in sd.items():
        if k.startswith(MAP_PREFIXES):
            out["w__" + k] = t.detach().cpu().numpy()
    for k, t in inp.items():
        out["in__" + k] = t.detach().cpu().numpy()

    for k, t in intermediates(master, inp).items():
        out["exp/" + k] = t.detach().cpu().numpy()

    discount = (DISCOUNT * inp["masks"]).view(-1, 1)
    for mode in MODES:
        a = build_agent(mode)
        a.model.load_state_dict(copy.deepcopy(sd))          # identical weights for all modes
        m = a.update_fb(obs=inp["observations"], action=inp["actions"], discount=discount,
                        next_obs=inp["next_observations"], goal=inp["next_observations"],
                        z=inp["z"], q_loss_coef=None, clip_grad_norm=None)
        L_V, L_N, L_M = float(m["L_V"]), float(m["L_N"]), float(m["L_M"])
        orth = float(m["orth_loss"])
        total = ALPHA_N * L_N + ALPHA_M * L_M + ORTHO_COEF * orth
        if mode != "coord":
            total = total + L_V
        for name, val in (("L_V", L_V), ("L_N", L_N), ("L_M", L_M), ("ortho", orth),
                          ("total", total), ("adv_mean", float(m["adv_mean"])),
                          ("M_mean", float(m["M_mean"]))):
            out[f"exp/{mode}/{name}"] = np.asarray(val)
        print(f"{mode:18s} L_V={L_V:.12g} L_N={L_N:.12g} L_M={L_M:.12g} "
              f"ortho={orth:.12g} total={total:.12g}")

    meta = dict(B=B, obs_dim=OBS, action_dim=ACT, z_dim=Z, L_dim=L, num_parallel=P,
                hidden_dim=H, discount=DISCOUNT, expectile=EXPECTILE, ortho_coef=ORTHO_COEF,
                fb_pessimism_penalty=PESSIMISM, alpha_n=ALPHA_N, alpha_m=ALPHA_M,
                awr_beta=AWR_BETA, awr_wmax=AWR_WMAX)
    for k, v in meta.items():
        out["meta/" + k] = np.asarray(v)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isfb_reference.npz")
    np.savez(path, **out)
    print("wrote", path, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
