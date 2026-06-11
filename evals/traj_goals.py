"""evals/traj_goals.py — trajectory-future (HER-style) goal sampling over a flat
transition buffer with episode-relative `timestep` (DictBuffer layout).

The discounted-future goal distribution g = s_{t+k}, k ~ Geom(1-gamma), truncated at
the episode end — GCIQL's p_trajgoal sampler, the goal distribution that carries the
controllability signal per the matched-readout control (B2 verdict,
PAPER/specs/2026-06-11-fb-psm-crl-gap-analysis-and-program.md).
"""
from __future__ import annotations

import numpy as np
import torch

# canonical implementations live next to the buffer (used by DictBuffer.sample
# future_gamma); re-exported here for probe scripts/tests.
from buffers.transition import episode_last_index, sample_future_idx  # noqa: F401


def future_goal_batch(storage_next: dict, timestep: np.ndarray, idx: np.ndarray,
                      gamma: float, rng: np.random.Generator):
    """Convenience: trajectory-future (next-state) goals for flat indices `idx`.
    Returns (goal_obs, goal_physics_or_None, j)."""
    ep_last = episode_last_index(timestep)
    j = sample_future_idx(idx, ep_last, gamma, rng)
    obs = storage_next["observation"][j]
    phys = storage_next.get("physics")
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    phys_t = torch.as_tensor(np.asarray(phys[j]), dtype=torch.float32) if phys is not None else None
    return obs, phys_t, j
