"""NumPy episode replay buffer with Interaction-Weighted Sampling (IWS).

This is a faithful NumPy port of Unsupervised_Manipulation's VectorizedReplayBuffer
and its IWS sampling path (_sample_interaction_weighted_indices et al.).

IWS recap:
  1. On episode completion, scan for contact frames via stored ``phase_labels == 2``
     (paddle-puck distance from env metrics, not raw state indices).
  2. At sample time, select an event uniformly (or use "first"), then draw
     `context_repetition_factor` anchor steps weighted around the event center
     via Gaussian temporal decay + spatial decay from the center distance.
  3. Future goals are sampled from [anchor, event_end] with geometric discount
     (or from the full episode if `full_episode_future=True`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveBatch:
    states: np.ndarray       # (B, state_dim)
    next_states: np.ndarray  # (B, state_dim)
    goals: np.ndarray        # (B, goal_dim)
    actions: np.ndarray      # (B, action_dim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contact_event_bounds(
    contact_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sorted contact-step indices into (event_starts, event_ends) arrays."""
    if contact_steps.size <= 1:
        return contact_steps, contact_steps
    gap = np.diff(contact_steps) > 1
    starts = np.concatenate([contact_steps[:1], contact_steps[1:][gap]])
    ends = np.concatenate([contact_steps[:-1][gap], contact_steps[-1:]])
    return starts, ends


def _event_centers_from_bounds(
    event_starts: np.ndarray,
    event_ends: np.ndarray,
    distances: np.ndarray | None,
) -> np.ndarray:
    centers = []
    for s, e in zip(event_starts, event_ends):
        if distances is None:
            centers.append((s + e) // 2)
        else:
            steps = np.arange(int(s), int(e) + 1)
            centers.append(steps[np.argmin(distances[steps])])
    return np.array(centers, dtype=np.int64)


def _segment_bounds(centers: np.ndarray, event_idx: int, length: int) -> tuple[int, int]:
    c = int(centers[event_idx])
    start = 0 if event_idx == 0 else ((int(centers[event_idx - 1]) + c) // 2) + 1
    end = length - 1 if event_idx + 1 >= len(centers) else (c + int(centers[event_idx + 1])) // 2
    return max(0, min(start, length - 1)), max(start, min(end, length - 1))


def _sample_future_index(
    step_indices: np.ndarray,
    future_lengths: np.ndarray,
    future_discount: float,
) -> np.ndarray:
    """Geometric discount future sampling for each row."""
    batch = len(step_indices)
    future = np.empty(batch, dtype=np.int64)
    for i in range(batch):
        s = int(step_indices[i])
        length = int(future_lengths[i])
        horizon = max(1, length - s)
        discount = float(future_discount)
        offsets = np.arange(1, horizon + 1)
        weights = discount ** (offsets.astype(np.float32) - 1.0)
        weights /= weights.sum() + 1e-8
        offset = np.random.choice(offsets, p=weights)
        future[i] = min(s + offset, length)
    return future


def _weighted_segment_steps(
    center: int,
    start: int,
    end: int,
    repetition: int,
    sigma: float,
    distances: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `repetition` steps around `center` with temporal + spatial weights."""
    candidates = np.arange(start, end + 1, dtype=np.int64)
    time_dist = np.abs(candidates - center).astype(np.float32)
    weights = np.exp(-time_dist / max(sigma, 1e-6))
    if distances is not None and center < len(distances):
        cd = float(distances[center])
        local_d = distances[candidates]
        scale = max(2.0 * cd + 0.25, 0.25)
        spatial = np.exp(-np.clip(local_d - cd, 0.0, None) / scale)
        weights = weights * spatial
    weights = weights + 1e-3

    sampled = candidates[np.random.choice(len(candidates), size=max(0, repetition - 1),
                                          replace=True, p=weights / weights.sum())]
    steps = np.concatenate([[center], sampled]).astype(np.int64)
    future_lengths = np.full(repetition, end + 1, dtype=np.int64)
    return steps, future_lengths


# ---------------------------------------------------------------------------
# Main replay buffer
# ---------------------------------------------------------------------------


class IWSReplayBuffer:
    """Episode-based replay buffer with IWS sampling.

    Stores full trajectories and performs future-goal relabeling (HER-style)
    at sample time.  IWS caches contact-event geometry on episode completion
    so sampling is efficient.

    Args:
        capacity_transitions: Approximate total transition capacity.
        max_episode_steps:    Maximum steps per episode.
        num_envs:             Number of parallel environments.
        state_dim:            State vector dimension.
        goal_dim:             Goal vector dimension (often == state_dim).
        action_dim:           Action dimension.
        contact_distance:     Distance threshold for defining contact.
    """

    def __init__(
        self,
        *,
        capacity_transitions: int,
        max_episode_steps: int,
        num_envs: int,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        contact_distance: float = 0.09,
    ) -> None:
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.contact_distance = float(contact_distance)
        self.capacity_episodes = max(1, int(capacity_transitions) // max(1, self.max_episode_steps))

        S, G, A = int(state_dim), int(goal_dim), int(action_dim)
        T = self.max_episode_steps
        E = self.capacity_episodes
        N = self.num_envs

        # Stored episodes
        self.states = np.zeros((E, T + 1, S), dtype=np.float32)
        self.achieved_goals = np.zeros((E, T + 1, G), dtype=np.float32)
        self.actions = np.zeros((E, T, A), dtype=np.float32)
        self.lengths = np.zeros(E, dtype=np.int64)
        self.valid = np.zeros(E, dtype=bool)

        # IWS event cache (populated on episode completion)
        self.iws_event_centers = np.zeros((E, T), dtype=np.int64)
        self.iws_event_starts = np.zeros((E, T), dtype=np.int64)
        self.iws_event_ends = np.zeros((E, T), dtype=np.int64)
        self.iws_event_counts = np.zeros(E, dtype=np.int64)
        self.iws_distances = np.zeros((E, T), dtype=np.float32)
        self.iws_has_distances = np.zeros(E, dtype=bool)
        self.phase_labels = np.zeros((E, T), dtype=np.int64)
        self.hand_object_distances = np.zeros((E, T), dtype=np.float32)

        # Active episode buffers (per env)
        self.active_states = np.zeros((N, T + 1, S), dtype=np.float32)
        self.active_achieved_goals = np.zeros((N, T + 1, G), dtype=np.float32)
        self.active_actions = np.zeros((N, T, A), dtype=np.float32)
        self.active_phase_labels = np.zeros((N, T), dtype=np.int64)
        self.active_hand_object_distances = np.zeros((N, T), dtype=np.float32)
        self.active_lengths = np.zeros(N, dtype=np.int64)

        self.write_index = 0
        self.num_episodes = 0

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self, states: np.ndarray, achieved_goals: np.ndarray | None = None) -> None:
        """Initialize active buffers from reset observations."""
        self.active_states.fill(0.0)
        self.active_achieved_goals.fill(0.0)
        self.active_actions.fill(0.0)
        self.active_phase_labels.fill(0)
        self.active_hand_object_distances.fill(0.0)
        self.active_lengths[:] = 0
        self.active_states[:, 0] = states
        if achieved_goals is not None:
            self.active_achieved_goals[:, 0] = achieved_goals
        else:
            self.active_achieved_goals[:, 0] = states

    def append(
        self,
        *,
        states: np.ndarray,           # (N, S) — observation before action
        achieved_goals: np.ndarray,   # (N, G) — achieved goal at this step
        actions: np.ndarray,          # (N, A)
        next_states: np.ndarray,      # (N, S) — transition observation after action
        next_achieved_goals: np.ndarray,  # (N, G) — transition achieved goal
        dones: np.ndarray,            # (N,) bool
        reset_states: np.ndarray | None = None,  # (N, S) — post-reset obs when done
        reset_achieved_goals: np.ndarray | None = None,  # (N, G)
        phase_labels: np.ndarray | None = None,  # (N,) int phase for this step
        hand_object_distances: np.ndarray | None = None,  # (N,) paddle-puck dist
    ) -> None:
        env_idx = np.arange(self.num_envs)
        step_idx = np.clip(self.active_lengths, 0, self.max_episode_steps - 1)
        next_idx = np.clip(step_idx + 1, 0, self.max_episode_steps)

        self.active_actions[env_idx, step_idx] = actions
        self.active_states[env_idx, next_idx] = next_states
        self.active_achieved_goals[env_idx, next_idx] = next_achieved_goals
        if phase_labels is not None:
            self.active_phase_labels[env_idx, step_idx] = np.asarray(
                phase_labels, dtype=np.int64
            ).reshape(-1)
        if hand_object_distances is not None:
            self.active_hand_object_distances[env_idx, step_idx] = np.asarray(
                hand_object_distances, dtype=np.float32
            ).reshape(-1)
        self.active_lengths = np.minimum(self.active_lengths + 1, self.max_episode_steps)

        done_mask = np.asarray(dones, dtype=bool)
        if done_mask.any():
            self._store_completed(done_mask)
            seed_states = next_states if reset_states is None else reset_states
            seed_goals = next_achieved_goals if reset_achieved_goals is None else reset_achieved_goals
            self._reset_active(done_mask, seed_states, seed_goals)

    def _store_completed(self, done_mask: np.ndarray) -> None:
        env_indices = np.nonzero(done_mask)[0]
        n = int(env_indices.size)
        if n == 0:
            return
        storage = (np.arange(n) + self.write_index) % self.capacity_episodes
        self.states[storage] = self.active_states[env_indices]
        self.achieved_goals[storage] = self.active_achieved_goals[env_indices]
        self.actions[storage] = self.active_actions[env_indices]
        self.phase_labels[storage] = self.active_phase_labels[env_indices]
        self.hand_object_distances[storage] = self.active_hand_object_distances[env_indices]
        self.lengths[storage] = self.active_lengths[env_indices]
        self.valid[storage] = True
        self._update_iws_event_cache(storage)
        self.write_index = int((self.write_index + n) % self.capacity_episodes)
        self.num_episodes = min(self.capacity_episodes, self.num_episodes + n)

    def _reset_active(
        self,
        done_mask: np.ndarray,
        next_states: np.ndarray,
        next_achieved_goals: np.ndarray,
    ) -> None:
        env_indices = np.nonzero(done_mask)[0]
        if env_indices.size == 0:
            return
        self.active_states[env_indices] = 0.0
        self.active_achieved_goals[env_indices] = 0.0
        self.active_actions[env_indices] = 0.0
        self.active_phase_labels[env_indices] = 0
        self.active_hand_object_distances[env_indices] = 0.0
        self.active_lengths[env_indices] = 0
        self.active_states[env_indices, 0] = next_states[env_indices]
        self.active_achieved_goals[env_indices, 0] = next_achieved_goals[env_indices]

    # ------------------------------------------------------------------
    # IWS event cache
    # ------------------------------------------------------------------

    def _update_iws_event_cache(self, storage_indices: np.ndarray) -> None:
        """Compute and cache contact events for newly stored episodes."""
        self.iws_event_counts[storage_indices] = 0
        self.iws_has_distances[storage_indices] = False

        for idx in storage_indices:
            length = int(self.lengths[idx])
            if length <= 0:
                continue

            stored_dist = self.hand_object_distances[idx, :length]
            has_phase = bool(np.any(self.phase_labels[idx, :length]))
            distances: np.ndarray | None = None
            if has_phase or bool(np.any(stored_dist)):
                distances = stored_dist.astype(np.float32)
                self.iws_distances[idx, :length] = distances
                self.iws_has_distances[idx] = True
            elif self.states.shape[-1] >= 6:
                # Legacy fallback for flat state layouts (not air-hockey history obs).
                ep_states = self.states[idx, :length]
                source_xy = ep_states[:, 0:2]
                target_xy = ep_states[:, 4:6]
                distances = np.linalg.norm(source_xy - target_xy, axis=-1).astype(np.float32)
                self.iws_distances[idx, :length] = distances
                self.iws_has_distances[idx] = True

            contact_steps = np.nonzero(self.phase_labels[idx, :length] == 2)[0]
            if contact_steps.size == 0 and distances is not None:
                contact_steps = np.nonzero(distances <= self.contact_distance)[0]
            elif contact_steps.size == 0:
                continue

            event_starts, event_ends = _contact_event_bounds(contact_steps)
            centers = _event_centers_from_bounds(event_starts, event_ends, distances)
            event_count = min(int(centers.size), self.max_episode_steps)
            if event_count == 0:
                continue

            for ei in range(event_count):
                start, end = _segment_bounds(centers, ei, length)
                self.iws_event_centers[idx, ei] = centers[ei]
                self.iws_event_starts[idx, ei] = start
                self.iws_event_ends[idx, ei] = end
            self.iws_event_counts[idx] = event_count

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def can_sample(self, batch_size: int) -> bool:
        valid_lengths = self.lengths[self.valid]
        if valid_lengths.size == 0:
            return False
        return bool(valid_lengths.sum() >= int(batch_size))

    def sample(
        self,
        batch_size: int,
        *,
        future_discount: float,
        context_repetition_factor: int = 4,
        contact_sigma: float = 20.0,
        random_fraction: float = 0.0,
        full_episode_future: bool = False,
        event_selection: str = "uniform",
        use_iws: bool = True,
    ) -> ContrastiveBatch:
        valid_idx = np.nonzero(self.valid & (self.lengths > 0))[0]
        if valid_idx.size == 0:
            raise RuntimeError("Replay buffer is empty.")

        weights = self.lengths[valid_idx].astype(np.float32)

        if use_iws and context_repetition_factor > 1:
            episode_indices, step_index, future_lengths = self._sample_iws_indices(
                valid_idx,
                weights,
                int(batch_size),
                context_repetition_factor=int(context_repetition_factor),
                sigma=float(contact_sigma),
                random_fraction=float(random_fraction),
                full_episode_future=bool(full_episode_future),
                event_selection=str(event_selection),
            )
        else:
            episode_indices = valid_idx[np.random.choice(
                valid_idx.size, size=batch_size, replace=True,
                p=weights / weights.sum()
            )]
            lengths = self.lengths[episode_indices]
            step_index = np.floor(np.random.random(batch_size) * lengths).astype(np.int64)
            future_lengths = lengths

        future_index = _sample_future_index(step_index, future_lengths, future_discount)
        return self._make_batch(episode_indices, step_index, future_index)

    def _sample_iws_indices(
        self,
        valid_idx: np.ndarray,
        weights: np.ndarray,
        batch_size: int,
        *,
        context_repetition_factor: int,
        sigma: float,
        random_fraction: float,
        full_episode_future: bool,
        event_selection: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        repetition = max(2, int(context_repetition_factor))
        base_n = max(1, math.ceil(batch_size / repetition))

        probs = weights / weights.sum()
        base_ep = valid_idx[np.random.choice(valid_idx.size, size=base_n, replace=True, p=probs)]

        step_matrix = np.empty((base_n, repetition), dtype=np.int64)
        fl_matrix = np.empty((base_n, repetition), dtype=np.int64)

        cached_mask = self.iws_event_counts[base_ep] > 0

        # Vectorized path for cached episodes
        if cached_mask.any():
            cached_eps = base_ep[cached_mask]
            cs, cf = self._sample_cached_iws_vectorized(
                cached_eps, repetition=repetition, sigma=sigma,
                random_fraction=random_fraction, full_episode_future=full_episode_future,
                event_selection=event_selection,
            )
            step_matrix[cached_mask] = cs
            fl_matrix[cached_mask] = cf

        # Scalar path for episodes with no cached events
        for pos in np.nonzero(~cached_mask)[0]:
            ep = base_ep[pos]
            steps, fls = self._sample_iws_episode_fallback(
                ep, repetition=repetition, sigma=sigma,
                random_fraction=random_fraction, full_episode_future=full_episode_future,
            )
            step_matrix[pos] = steps
            fl_matrix[pos] = fls

        ep_indices = np.repeat(base_ep, repetition)[:batch_size]
        steps = step_matrix.ravel()[:batch_size]
        fls = fl_matrix.ravel()[:batch_size]
        return ep_indices, steps, fls

    def _sample_cached_iws_vectorized(
        self,
        episode_indices: np.ndarray,
        *,
        repetition: int,
        sigma: float,
        random_fraction: float,
        full_episode_future: bool,
        event_selection: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(episode_indices.size)
        if n == 0:
            return np.empty((0, repetition), dtype=np.int64), np.empty((0, repetition), dtype=np.int64)

        event_counts = np.clip(self.iws_event_counts[episode_indices], 1, None)
        if event_selection == "first":
            event_idx = np.zeros(n, dtype=np.int64)
        else:
            event_idx = np.floor(np.random.random(n) * event_counts).astype(np.int64)

        centers = self.iws_event_centers[episode_indices, event_idx]
        starts = self.iws_event_starts[episode_indices, event_idx]
        ends = self.iws_event_ends[episode_indices, event_idx]
        lengths = np.clip(self.lengths[episode_indices], 1, None)

        # Build step grid for each episode
        step_grid = np.arange(self.max_episode_steps)[None, :]  # (1, T)
        valid = (step_grid >= starts[:, None]) & (step_grid <= ends[:, None])
        time_dist = np.abs(step_grid - centers[:, None]).astype(np.float32)
        w = np.exp(-time_dist / max(sigma, 1e-6))

        has_dist = self.iws_has_distances[episode_indices]
        if has_dist.any():
            dists = self.iws_distances[episode_indices]  # (n, T)
            cd = dists[np.arange(n), centers]             # (n,)
            scale = np.clip(2.0 * cd + 0.25, 0.25, None)
            spatial = np.exp(-np.clip(dists - cd[:, None], 0.0, None) / scale[:, None])
            w = np.where(has_dist[:, None], w * spatial, w)

        sample_w = np.where(valid, w + 1e-3, 0.0)

        steps = np.empty((n, repetition), dtype=np.int64)
        fl = np.empty((n, repetition), dtype=np.int64)
        steps[:, 0] = centers
        fl[:, :] = (ends + 1)[:, None]

        if repetition > 1:
            for i in range(n):
                row_w = sample_w[i]
                total = row_w.sum()
                if total <= 0:
                    steps[i, 1:] = centers[i]
                else:
                    sampled = np.random.choice(self.max_episode_steps, size=repetition - 1,
                                               replace=True, p=row_w / total)
                    steps[i, 1:] = sampled

        # Random fraction mixing
        rc = max(0, min(repetition - 1, int(round(repetition * float(random_fraction)))))
        if rc > 0:
            rand_steps = np.floor(
                np.random.random((n, rc)) * lengths[:, None]
            ).astype(np.int64)
            steps[:, -rc:] = rand_steps
            fl[:, -rc:] = lengths[:, None]

        if full_episode_future:
            fl[:, :] = lengths[:, None]

        return steps, fl

    def _sample_iws_episode_fallback(
        self,
        ep_idx: int,
        *,
        repetition: int,
        sigma: float,
        random_fraction: float,
        full_episode_future: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        length = int(self.lengths[ep_idx])
        if length <= 0:
            return np.zeros(repetition, np.int64), np.ones(repetition, np.int64)

        rand_steps = np.floor(np.random.random(repetition) * length).astype(np.int64)
        return rand_steps, np.full(repetition, length, dtype=np.int64)

    def _make_batch(
        self,
        episode_indices: np.ndarray,
        step_index: np.ndarray,
        future_index: np.ndarray,
    ) -> ContrastiveBatch:
        return ContrastiveBatch(
            states=self.states[episode_indices, step_index],
            next_states=self.states[episode_indices, step_index + 1],
            goals=self.achieved_goals[episode_indices, future_index],
            actions=self.actions[episode_indices, step_index],
        )

    # ------------------------------------------------------------------
    # Stats and export
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        valid_lengths = self.lengths[self.valid]
        n_ep = int(self.valid.sum())
        n_tr = int(valid_lengths.sum()) if valid_lengths.size > 0 else 0
        mean_len = float(valid_lengths.mean()) if valid_lengths.size > 0 else 0.0
        n_iws = int(self.iws_event_counts[self.valid].sum()) if n_ep > 0 else 0
        return {
            "replay/episodes": n_ep,
            "replay/transitions": n_tr,
            "replay/mean_episode_length": mean_len,
            "replay/iws_events": n_iws,
        }

    def export_npz(self, out_dir: str | Path, *, max_episodes: int | None = None) -> list[Path]:
        """Export valid episodes to per-episode .npz files (Factored-FB episode format)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        valid_slots = np.nonzero(self.valid & (self.lengths > 0))[0]
        if max_episodes is not None:
            valid_slots = valid_slots[: int(max_episodes)]

        written: list[Path] = []
        for ep_num, slot in enumerate(valid_slots):
            length = int(self.lengths[slot])
            obs = self.states[slot, : length + 1].astype(np.float32)     # (T+1, S)
            act = self.actions[slot, :length].astype(np.float32)          # (T, A)
            ag = self.achieved_goals[slot, : length + 1].astype(np.float32)

            pad_action = np.zeros_like(act[-1])[None]
            action_arr = np.concatenate([act, pad_action], axis=0)

            reward_arr = np.zeros((length + 1, 1), dtype=np.float32)
            discount_arr = np.ones((length + 1, 1), dtype=np.float32)
            terminated_arr = np.zeros((length + 1, 1), dtype=bool)
            terminated_arr[-1, 0] = True

            ep = {
                "observation": obs,
                "physics": obs.copy(),
                "action": action_arr,
                "reward": reward_arr,
                "discount": discount_arr,
                "terminated": terminated_arr,
                "achieved_goal": ag,
            }
            path = out_dir / f"episode_{ep_num:06d}_{length}.npz"
            np.savez_compressed(path, **ep)
            written.append(path)

        return written

    def state_dict(self) -> dict[str, Any]:
        return {
            "states": self.states,
            "achieved_goals": self.achieved_goals,
            "actions": self.actions,
            "lengths": self.lengths,
            "valid": self.valid,
            "iws_event_centers": self.iws_event_centers,
            "iws_event_starts": self.iws_event_starts,
            "iws_event_ends": self.iws_event_ends,
            "iws_event_counts": self.iws_event_counts,
            "iws_distances": self.iws_distances,
            "iws_has_distances": self.iws_has_distances,
            "phase_labels": self.phase_labels,
            "hand_object_distances": self.hand_object_distances,
            "write_index": self.write_index,
            "num_episodes": self.num_episodes,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        for key in ("states", "achieved_goals", "actions", "lengths", "valid",
                    "iws_event_centers", "iws_event_starts", "iws_event_ends",
                    "iws_event_counts", "iws_distances", "iws_has_distances",
                    "phase_labels", "hand_object_distances"):
            getattr(self, key)[:] = d[key]
        self.write_index = int(d["write_index"])
        self.num_episodes = int(d["num_episodes"])
