"""evals/hecrl.py — Per-episode evaluator for HECRL ManipObj.

Unlike OGBench (5 fixed tasks per domain), HECRL ManipObj resamples the
goal at every env.reset(). We therefore infer z afresh each episode
against env._cur_goal_ob_info["qpos"], using the env's own vectorised
compute_reward_batch.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from envs.hecrl import create_hecrl_env


class HECRLEvaluator:
    def __init__(
        self,
        domain: str,
        agent,
        offline_buffer,
        relabel_size: int = 10_000,
        n_episodes: int = 10,
        shift_reward: float = 1.0,
        obs_type: str = "state",
        seed: int = 0,
        device: str = "cpu",
        **_unused,
    ):
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.relabel_size = relabel_size
        self.n_episodes = n_episodes
        self.shift_reward = shift_reward
        self.device = device

        # One env reused across episodes (instantiation is heavy: ~2-5s).
        self.env, _ = create_hecrl_env(domain, obs_type=obs_type, seed=seed)

    def run(self) -> Dict[str, float]:
        successes, returns = [], []
        for _ in range(self.n_episodes):
            obs, _info = self.env.reset()
            z = self._infer_z()
            ep_return, ep_success = 0.0, False
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.agent.act(
                    obs=torch.tensor(obs, device=self.device, dtype=torch.float32)[None],
                    z=z[None],
                ).cpu().numpy()[0]
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_return += float(reward)
                ep_success = ep_success or bool(info.get("success", False))
            successes.append(float(ep_success))
            returns.append(ep_return)

        return {
            f"{self.domain}/success": float(np.mean(successes)),
            f"{self.domain}/reward": float(np.mean(returns)),
        }

    @torch.no_grad()
    def _infer_z(self) -> Tensor:
        n = min(self.relabel_size, len(self.offline_buffer))
        if n < 8:
            return self.agent.model.sample_z(1).squeeze(0)

        batch = self.offline_buffer.sample(n)
        if "physics" not in batch["next"]:
            raise KeyError(
                "HECRL buffer is missing batch['next']['physics']. "
                "Reload via data.hecrl.load_hecrl_dataset."
            )

        # ManipObjEnv sets _cur_goal_ob_info during initialize_episode (called from reset).
        # ob_info['qpos'] is the goal-state mujoco qpos.
        goal_qpos_np = self.env.unwrapped._cur_goal_ob_info["qpos"]
        buf_qpos = batch["next"]["physics"].to(self.device).float()              # [N, qpos_dim]
        goal_qpos = (
            torch.as_tensor(goal_qpos_np, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .expand_as(buf_qpos)
        )

        # compute_reward_batch returns [N, 1]; assumes reward_mode in {sparse, step}.
        rewards = self.env.unwrapped.compute_reward_batch(buf_qpos, goal_qpos)
        rewards = rewards + self.shift_reward

        return self.agent.model.reward_inference(
            next_obs=batch["next"]["observation"].to(self.device).float(),
            reward=rewards,
        )
