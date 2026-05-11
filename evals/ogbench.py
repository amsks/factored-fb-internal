"""evals/ogbench.py — OGBench evaluator.

For each of the 5 tasks in a domain:
  1. Sample a batch from the offline buffer.
  2. Relabel with `get_relabel_fn(domain, task)(next.physics, action)`.
  3. Shift rewards by `shift_reward` (default 1.0).
  4. Infer z = project(sum r_i * B(next_obs_i)).
  5. Roll out `n_episodes` episodes via envs/rollout.py.
  6. Report mean success and mean return per task.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor

from envs.ogbench import ALL_TASKS, create_ogbench_env, get_relabel_fn
from envs.rollout import rollout


class OGBenchEvaluator:
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
    ):
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.relabel_size = relabel_size
        self.n_episodes = n_episodes
        self.shift_reward = shift_reward
        self.device = device

        tasks = ALL_TASKS.get(domain, [])
        self.task_envs: Dict[str, object] = {}
        self.relabel_fns: Dict[str, object] = {}
        for task in tasks:
            env, _ = create_ogbench_env(task, seed=seed, obs_type=obs_type)
            self.task_envs[task] = env
            self.relabel_fns[task] = get_relabel_fn(domain, task)

    def run(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for task, env in self.task_envs.items():
            z = self._infer_z(task)
            stats, infos = rollout(env, self.agent, self.n_episodes, ctx=z)
            success = float(np.mean([
                any(step.get("success", False) for step in ep_info)
                for ep_info in infos
            ]))
            metrics[f"{task}/success"] = success
            metrics[f"{task}/reward"] = float(np.mean(stats["reward"]))
        return metrics

    def _infer_z(self, task: str) -> Tensor:
        relabel_fn = self.relabel_fns[task]
        n = min(self.relabel_size, len(self.offline_buffer))
        if n < 8:
            return self.agent.model.sample_z(1).squeeze(0)

        batch = self.offline_buffer.sample(n)

        if "physics" not in batch["next"]:
            raise KeyError(
                "Buffer batch is missing batch['next']['physics']. "
                "Re-load the dataset with data.ogbench.load_ogbench_dataset "
                "(updated to place physics under next/)."
            )

        phys_np = batch["next"]["physics"].cpu().numpy()
        act_np = batch["action"].cpu().numpy()

        rewards_np = relabel_fn(phys_np, act_np)               # [N, 1]
        rewards_np = rewards_np + self.shift_reward
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)

        return self.agent.model.reward_inference(
            next_obs=batch["next"]["observation"].to(self.device),
            reward=rewards,
        )
