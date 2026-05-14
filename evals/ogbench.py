"""evals/ogbench.py — OGBench evaluator.

For each of the 5 tasks in a domain:
  1. Sample a batch from the offline buffer.
  2. Relabel with `get_relabel_fn(domain, task)(next.physics, action)`.
  3. Shift rewards by `shift_reward` (default 1.0).
  4. Infer z = project(sum r_i * B(next_obs_i)).
  5. Roll out `n_episodes` episodes via envs/rollout.py.
  6. Report mean/std success and reward per task, plus relabel-inference
     debug metrics (matches td_jepa's OGBenchRewardEvaluation.run output).
  7. Aggregate across tasks: eval/reward, eval/reward#std, eval/success.
"""

from __future__ import annotations

from typing import Dict, Tuple

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
        # Match td_jepa's OGBenchRewardEvaluation: do NOT build envs upfront.
        # Building all 5 eval envs + 5 relabel-fn envs at startup would add
        # ~50s of mujoco init even on runs that never call run(). Build them
        # lazily inside run() / _infer_z() and tear them down after each task.
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.relabel_size = relabel_size
        self.n_episodes = n_episodes
        self.shift_reward = shift_reward
        self.obs_type = obs_type
        self.seed = seed
        self.device = device
        self.tasks = ALL_TASKS.get(domain, [])

    def run(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        all_rewards: list[np.ndarray] = []
        all_successes: list[np.ndarray] = []
        for task in self.tasks:
            env, _ = create_ogbench_env(task, seed=self.seed, obs_type=self.obs_type)
            try:
                z, relabel_metrics = self._infer_z(task)
                stats, infos = rollout(env, self.agent, self.n_episodes, ctx=z)
            finally:
                env.close()

            per_ep_success = np.array(
                [any(step.get("success", False) for step in ep_info) for ep_info in infos],
                dtype=np.float32,
            )
            per_ep_reward = np.array(stats["reward"], dtype=np.float32)

            metrics[f"{task}/success"] = float(np.mean(per_ep_success))
            metrics[f"{task}/reward"] = float(np.mean(per_ep_reward))
            metrics[f"{task}/reward#std"] = float(np.std(per_ep_reward))
            for k, v in relabel_metrics.items():
                metrics[f"{task}/{k}"] = float(v)

            all_rewards.append(per_ep_reward)
            all_successes.append(per_ep_success)

        if all_rewards:
            rewards_all = np.concatenate(all_rewards)
            successes_all = np.concatenate(all_successes)
            metrics["eval/reward"] = float(np.mean(rewards_all))
            metrics["eval/reward#std"] = float(np.std(rewards_all))
            metrics["eval/success"] = float(np.mean(successes_all))

        return metrics

    def _infer_z(self, task: str) -> Tuple[Tensor, Dict[str, float]]:
        relabel_fn = get_relabel_fn(self.domain, task)  # builds + closes a temp env internally
        n = min(self.relabel_size, len(self.offline_buffer))
        if n < 8:
            return self.agent.model.sample_z(1).squeeze(0), {}

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

        non_zero = int(np.count_nonzero(rewards_np.ravel()))
        relabel_metrics = {
            "relabel_reward#mean": float(np.mean(rewards_np)),
            "relabel_reward#nonzero": float(non_zero),
            "relabel_reward#zero": float(rewards_np.size - non_zero),
            "relabel_reward#num_samples": float(rewards_np.size),
        }

        z = self.agent.model.reward_inference(
            next_obs=batch["next"]["observation"].to(self.device),
            reward=rewards,
        )
        return z, relabel_metrics
