"""Tests for UM-aligned IWS phase / contact detection."""

from __future__ import annotations

import numpy as np

from agents.crl_iws.phase import compute_phase_labels, hand_object_distance
from agents.crl_iws.replay import IWSReplayBuffer


def test_phase_contact_uses_hand_object_not_state_slices():
    """History-style obs: state[0:2]/[4:6] are not paddle/puck; metrics are."""
    state = np.zeros(30, dtype=np.float32)
    state[0:2] = [0.0, 0.0]
    state[4:6] = [0.0, 0.0]
    state[12:14] = [0.5, 0.1]
    state[27:29] = [0.8, 0.1]

    hand = state[12:14]
    obj = state[27:29]
    legacy_dist = float(np.linalg.norm(state[0:2] - state[4:6]))
    metric_dist = float(np.linalg.norm(hand - obj))

    assert legacy_dist < 0.09
    assert metric_dist > 0.09

    phase, _ = compute_phase_labels(
        hand[None],
        obj[None],
        hand[None],
        obj[None],
        np.zeros(1, dtype=np.int64),
        contact_distance=0.09,
    )
    assert int(phase[0]) == 0

    near_hand = np.array([[0.5, 0.1]], dtype=np.float32)
    near_obj = np.array([[0.52, 0.11]], dtype=np.float32)
    phase_contact, _ = compute_phase_labels(
        near_hand,
        near_obj,
        near_hand,
        near_obj,
        np.zeros(1, dtype=np.int64),
        contact_distance=0.09,
    )
    assert int(phase_contact[0]) == 2


def test_replay_iws_events_from_phase_labels():
    replay = IWSReplayBuffer(
        capacity_transitions=500,
        max_episode_steps=20,
        num_envs=1,
        state_dim=30,
        goal_dim=2,
        action_dim=2,
        contact_distance=0.09,
    )
    obs = np.zeros((1, 30), dtype=np.float32)
    obs[0, 12:14] = [0.0, 0.0]
    obs[0, 27:29] = [0.5, 0.0]
    replay.reset(obs, np.zeros((1, 2), dtype=np.float32))

    hand = obs[:, 12:14]
    obj = obs[:, 27:29]
    for step in range(8):
        contact = step >= 3 and step <= 5
        if contact:
            obj = hand + np.array([[0.02, 0.0]], dtype=np.float32)
        else:
            obj = hand + np.array([[0.5, 0.0]], dtype=np.float32)
        next_obs = obs.copy()
        next_obs[0, 27:29] = obj[0]
        dist = hand_object_distance(hand, obj, hand, obj)
        phase = np.array([2 if contact else 0], dtype=np.int64)
        done = np.array([step == 7])
        replay.append(
            states=obs,
            achieved_goals=np.zeros((1, 2), dtype=np.float32),
            actions=np.zeros((1, 2), dtype=np.float32),
            next_states=next_obs,
            next_achieved_goals=obj,
            dones=done,
            phase_labels=phase,
            hand_object_distances=dist,
        )
        obs = next_obs
        if done[0]:
            break

    stats = replay.stats()
    assert stats["replay/iws_events"] >= 1.0
