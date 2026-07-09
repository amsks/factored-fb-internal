"""Interaction phase labels for IWS (UM-aligned).

Phase 0: passive
Phase 1: near_contact
Phase 2: contact
Phase 3: post_contact or near_goal
"""

from __future__ import annotations

import numpy as np


def compute_phase_labels(
    hand_positions: np.ndarray,
    object_positions: np.ndarray,
    transition_hand_positions: np.ndarray,
    transition_object_positions: np.ndarray,
    active_post_contact: np.ndarray,
    *,
    contact_distance: float = 0.09,
    near_contact_distance: float = 0.18,
    near_goal_distance: float = 0.18,
    post_contact_steps: int = 20,
    transition_goal_distance: np.ndarray | None = None,
    dones: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(phase_labels, updated_post_contact_countdown)`` for one env step."""
    hand = np.asarray(hand_positions, dtype=np.float32).reshape(-1, 2)
    obj = np.asarray(object_positions, dtype=np.float32).reshape(-1, 2)
    next_hand = np.asarray(transition_hand_positions, dtype=np.float32).reshape(-1, 2)
    next_obj = np.asarray(transition_object_positions, dtype=np.float32).reshape(-1, 2)
    countdown = np.asarray(active_post_contact, dtype=np.int64).reshape(-1).copy()

    dist = np.linalg.norm(hand - obj, axis=-1)
    next_dist = np.linalg.norm(next_hand - next_obj, axis=-1)
    distance = np.minimum(dist, next_dist)

    near_contact = distance <= float(near_contact_distance)
    contact = distance <= float(contact_distance)
    post_contact = countdown > 0
    if transition_goal_distance is not None:
        near_goal = np.asarray(transition_goal_distance, dtype=np.float32).reshape(-1) <= float(
            near_goal_distance
        )
    else:
        near_goal = np.zeros_like(contact, dtype=bool)

    phase = np.zeros(hand.shape[0], dtype=np.int64)
    phase = np.where(near_contact, 1, phase)
    phase = np.where(post_contact | near_goal, 3, phase)
    phase = np.where(contact, 2, phase)

    countdown = np.clip(countdown - 1, 0, None)
    countdown = np.where(contact, int(post_contact_steps), countdown)
    if dones is not None:
        countdown = np.where(np.asarray(dones, dtype=bool).reshape(-1), 0, countdown)

    return phase, countdown


def hand_object_distance(
    hand_positions: np.ndarray,
    object_positions: np.ndarray,
    transition_hand_positions: np.ndarray,
    transition_object_positions: np.ndarray,
) -> np.ndarray:
    """Minimum paddle-puck distance before/after a step (for IWS spatial weights)."""
    hand = np.asarray(hand_positions, dtype=np.float32).reshape(-1, 2)
    obj = np.asarray(object_positions, dtype=np.float32).reshape(-1, 2)
    next_hand = np.asarray(transition_hand_positions, dtype=np.float32).reshape(-1, 2)
    next_obj = np.asarray(transition_object_positions, dtype=np.float32).reshape(-1, 2)
    dist = np.linalg.norm(hand - obj, axis=-1)
    next_dist = np.linalg.norm(next_hand - next_obj, axis=-1)
    return np.minimum(dist, next_dist).astype(np.float32)
