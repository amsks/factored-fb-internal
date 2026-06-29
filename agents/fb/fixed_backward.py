"""agents/fb/fixed_backward.py — V2 (USFA): parameter-free backward map B(s) = cube xyz.

Replaces the learned FB backward map with the privileged goal variable (the cube
position), so B encodes the goal geometry by construction (R^2(B->d)=1). See
docs/superpowers/specs/2026-06-02-goal-conditioned-fb-design.md.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

# Verified state cube-single layout (agents/tdmpc2/agent.py:18-22): block_0 xyz,
# stored scaled as (pos - [0.425,0,0]) * 10.
CUBE_OBS_SLICE = slice(19, 22)
OBS_XYZ_OFFSET = (0.425, 0.0, 0.0)
OBS_XYZ_SCALE = 10.0


class FixedCubeBackward(nn.Module):
    """B(obs) = obs[cube_slice]  (the scaled cube xyz). No learnable parameters."""

    def __init__(self, cube_slice: slice = CUBE_OBS_SLICE) -> None:
        super().__init__()
        self.cube_slice = cube_slice
        self._output_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    @property
    def output_space(self):
        return self._output_space

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., self.cube_slice]


class FixedIdentityBackward(nn.Module):
    """B(obs) = obs  (the full, already-normalized observation). z = g LITERALLY:
    the task vector IS the goal observation, with NO learned backward map and NO
    sphere projection (the model sets norm_z=False for this map). Strips the FB
    bottleneck entirely so Q = F(s,a,z)·z becomes a BVN-style bilinear value with
    the goal fed in raw. Parameter-free; z_dim = obs_dim."""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self._output_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @property
    def output_space(self):
        return self._output_space

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


def scale_goal_xyz(goal_xyz: torch.Tensor) -> torch.Tensor:
    """Map a real-metre goal cube xyz into the scaled obs space FixedCubeBackward sees."""
    offset = torch.tensor(OBS_XYZ_OFFSET, dtype=goal_xyz.dtype, device=goal_xyz.device)
    return (goal_xyz - offset) * OBS_XYZ_SCALE
