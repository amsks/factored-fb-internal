"""Aligned training milestones (eval / checkpoint) — matches UM ``_build_eval_schedule``."""

from __future__ import annotations

import math

import numpy as np


def align_step(step: int, align_to: int) -> int:
    """Round ``step`` up to the next multiple of ``align_to`` (min 1)."""
    align_to = max(1, int(align_to))
    return int(math.ceil(int(step) / align_to) * align_to)


def build_aligned_schedule(
    total_env_steps: int,
    every_n_steps: int,
    align_to: int,
) -> list[int]:
    """Periodic milestones aligned to vectorized env step increments."""
    align_to = max(1, int(align_to))
    if int(every_n_steps) <= 0:
        return []
    periodic = np.arange(
        int(every_n_steps),
        int(total_env_steps) + int(every_n_steps),
        int(every_n_steps),
    )
    aligned = np.ceil(periodic / align_to).astype(np.int64) * align_to
    return sorted(
        {
            int(min(value, int(total_env_steps)))
            for value in aligned
            if value > 0
        }
    )


def advance_schedule_index(schedule: list[int], total_steps: int) -> int:
    """Index of the next schedule entry strictly after ``total_steps``."""
    for index, step in enumerate(schedule):
        if int(total_steps) < int(step):
            return index
    return len(schedule)


def next_pending_schedule_index(schedule: list[int], completed_steps: set[int]) -> int:
    """Index of the first schedule entry not yet completed."""
    for index, step in enumerate(schedule):
        if int(step) not in completed_steps:
            return index
    return len(schedule)


def legacy_nominal_to_aligned_map(
    *,
    total_env_steps: int,
    every_n_steps: int,
    align_to: int,
) -> dict[int, int]:
    """Map pre-alignment eval steps (exact ``k * every_n_steps``) to aligned milestones."""
    if int(every_n_steps) <= 0:
        return {}
    out: dict[int, int] = {}
    nominal = int(every_n_steps)
    while nominal <= int(total_env_steps):
        out[nominal] = align_step(nominal, align_to)
        nominal += int(every_n_steps)
    return out


def normalize_completed_eval_steps(
    completed: set[int],
    *,
    total_env_steps: int,
    every_n_steps: int,
    align_to: int,
) -> set[int]:
    """Treat legacy unaligned eval steps as their aligned milestone."""
    legacy_map = legacy_nominal_to_aligned_map(
        total_env_steps=total_env_steps,
        every_n_steps=every_n_steps,
        align_to=align_to,
    )
    out: set[int] = set()
    for step in completed:
        out.add(legacy_map.get(int(step), int(step)))
    return out


def eval_milestones(
    *,
    total_env_steps: int = 20_000_000,
    every_n_steps: int = 500_000,
    align_to: int = 256,
) -> list[int]:
    """Canonical eval steps for air-hockey multiseed comparisons (256 train envs)."""
    return build_aligned_schedule(total_env_steps, every_n_steps, align_to)
