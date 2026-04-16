"""
evals/ogbench.py — OGBench evaluator.

For each of the 5 tasks in a domain:
  1. Relabel offline buffer with task reward via get_relabel_fn
  2. Solve min_w ||B(obs)·w − r||² for z
  3. Rollout policy via envs/rollout.py
  4. Report success rate + mean return
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
        relabel_size: int = 512,
        n_episodes: int = 10,
        obs_type: str = "state",
        seed: int = 0,
        device: str = "cpu",
    ):
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.relabel_size = relabel_size
        self.n_episodes = n_episodes
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
        # Buffer stores "observation" (not "obs")
        obs_np  = batch["observation"].cpu().numpy()
        act_np  = batch["action"].cpu().numpy()

        rewards_np = relabel_fn(obs_np, act_np)          # [N, 1]
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32).squeeze(-1)

        mask = rewards.abs() > 1e-6
        if mask.sum() < 8:
            return self.agent.model.sample_z(1).squeeze(0)

        return self.agent.infer_z(batch["observation"][mask], rewards[mask])