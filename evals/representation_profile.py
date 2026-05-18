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
