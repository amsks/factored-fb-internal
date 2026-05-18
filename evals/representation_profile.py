"""evals/representation_profile.py — why does FB fail at transport?

Four probes (value landscape, z-decoding+sparsity, B-resolution,
coverage) over FB checkpoints. Pure logic here is unit-tested with
stubs; MuJoCo rollouts reuse evals.phase_probe. No training/eval code
is modified. See
docs/superpowers/specs/2026-05-18-fb-representation-failure-profile-design.md
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


@torch.no_grad()
def _ensemble_q(F: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """F [P, B, z_dim], z [z_dim] or [B, z_dim] -> Q [B] (mean over heads)."""
    if z.dim() == 1:
        z = z.view(1, 1, -1)
    elif z.dim() == 2:
        z = z.unsqueeze(0)
    q = (F * z).sum(-1)            # [P, B]
    return q.mean(0)              # [B]


@torch.no_grad()
def q_values(model, obs: torch.Tensor, action: torch.Tensor,
             z: torch.Tensor) -> np.ndarray:
    F = model.forward_map(obs, z, action)
    return _ensemble_q(F, z).cpu().numpy()


@torch.no_grad()
def v_values(model, agent, obs: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    action = agent.act(obs, z)
    if isinstance(action, np.ndarray):
        action = torch.as_tensor(action, dtype=torch.float32)
    return q_values(model, obs, action.to(obs.device).float(), z)


def transport_mask(ep: Dict[str, Any], thr) -> np.ndarray:
    """Per-step boolean: cube lifted above table and gripper closed."""
    cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(ep["grip"], dtype=np.float64).reshape(-1)
    lift = cube[:, 2] - float(ep["table_z"])
    return (lift > thr.delta_lift) & (grip > thr.tau_grip)


def rollout_for_profile(env, agent, z, n_episodes: int, thr,
                        seed_offset: int = 0) -> List[Dict[str, Any]]:
    """S0 rollouts with obs/action recorded, each tagged with outcome,
    per-step cube->goal distance, and a transport-region mask."""
    from evals.phase_probe import classify_phases, rollout_with_phase_signals

    eps = rollout_with_phase_signals(
        env, agent, z, n_episodes, thr, scenario="S0", record_obs=True)
    out = []
    for ep in eps:
        cls = classify_phases(ep, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            outcome = "transport_fail"
        else:
            outcome = "other"
        cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
        goal = np.asarray(ep["goal"], dtype=np.float64).reshape(3)
        d = np.linalg.norm(cube - goal, axis=1)
        out.append({
            "obs": ep["obs"], "action": ep["action"],
            "cube": cube, "goal": goal, "d": d,
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "table_z": float(ep["table_z"]),
            "transport_mask": transport_mask(ep, thr),
            "outcome": outcome, "length": ep["length"],
        })
    return out
