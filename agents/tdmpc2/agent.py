"""TDMPC2Agent — BaseAgent wrapper bridging OGBench (offline, reward-free,
goal-conditioned) to TD-MPC2 (reward-driven MPC). All OGBench-specific logic
lives here; agents/tdmpc2/core.py stays algorithmically identical to upstream."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from agents.base import BaseAgent
from agents.tdmpc2.config import build_tdmpc2_cfg
from agents.tdmpc2.core import TDMPC2Core

GOAL_DIM = 3                 # cube target xyz
CUBE_SLICE = slice(14, 17)   # physics[..., 14:17] == cube xyz (see envs/ogbench.cube_reward_fn)
SUCCESS_THRESHOLD = 0.04


class TDMPC2Agent(BaseAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        *,
        device: str = "cpu",
        horizon: int = 3,
        batch_size: int = 256,
        random_goal_ratio: float = 0.5,
        success_threshold: float = SUCCESS_THRESHOLD,
        **cfg_overrides,
    ):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.random_goal_ratio = float(random_goal_ratio)
        self.success_threshold = float(success_threshold)
        obs_dim = int(obs_space.shape[0])
        # core._update allocates zs with cfg.batch_size, so it MUST equal the
        # actual sampled batch size (buffer.sample returns exactly batch_size rows).
        cfg_overrides.setdefault("batch_size", int(batch_size))
        self.cfg = build_tdmpc2_cfg(
            obs_dim=obs_dim, action_dim=action_dim, device=str(device),
            horizon=horizon, **cfg_overrides,
        )
        self.horizon = int(self.cfg.horizon)
        self.core = TDMPC2Core(self.cfg)
        self._t0 = True

    # ---- goal / reward helpers ------------------------------------------- #
    def _cube_xyz(self, physics: Tensor) -> Tensor:
        return physics[..., CUBE_SLICE].to(self.device).float()

    def _sample_goals(self, next_phys_window: Tensor) -> Tensor:
        """next_phys_window: [B, H, phys]. Returns goal cube xyz [B, 3].
        Hindsight: window-final achieved cube xyz; with prob random_goal_ratio,
        swap for another window's final cube xyz (random goal relabeling)."""
        final_cube = self._cube_xyz(next_phys_window[:, -1, :])  # [B,3]
        B = final_cube.shape[0]
        use_random = torch.rand(B, device=self.device) < self.random_goal_ratio
        perm = torch.randperm(B, device=self.device)
        return torch.where(use_random.unsqueeze(-1), final_cube[perm], final_cube)

    def _reward(self, next_phys_window: Tensor, goal: Tensor) -> Tensor:
        """[B,H,phys], goal [B,3] -> reward [B,H,1]: 0 if within threshold else -1."""
        cube = self._cube_xyz(next_phys_window)               # [B,H,3]
        dist = torch.linalg.norm(cube - goal.unsqueeze(1), dim=-1)  # [B,H]
        return torch.where(dist < self.success_threshold, 0.0, -1.0).unsqueeze(-1)

    @staticmethod
    def _fold(obs: Tensor, goal: Tensor) -> Tensor:
        """obs [...,obs_dim], goal [B,3] broadcast over leading dims -> concat."""
        if obs.dim() == goal.dim():           # [B,obs] + [B,3]
            return torch.cat([obs, goal], dim=-1)
        # obs [T,B,obs] + goal [B,3] -> broadcast over T
        g = goal.unsqueeze(0).expand(obs.shape[0], *goal.shape)
        return torch.cat([obs, g], dim=-1)

    # ---- BaseAgent API --------------------------------------------------- #
    def update(self, batch: Dict, step: int) -> Dict[str, float]:
        # batch fields are [B, H, ...] windows from DictBuffer.sample(horizon=H).
        obs = batch["observation"].to(self.device).float()          # [B,H,obs]
        next_obs = batch["next"]["observation"].to(self.device).float()
        action = batch["action"].to(self.device).float()            # [B,H,act]
        next_phys = batch["next"]["physics"].to(self.device).float()  # [B,H,phys]
        term = batch["next"]["terminated"].to(self.device).float()  # [B,H,1] (or [B,H])
        if term.dim() == 2:
            term = term.unsqueeze(-1)

        goal = self._sample_goals(next_phys)                        # [B,3]
        reward = self._reward(next_phys, goal)                      # [B,H,1]

        # Build H+1 obs sequence: obs[:,0..H-1] then final next_obs[:,H-1].
        obs_seq = torch.cat([obs, next_obs[:, -1:, :]], dim=1)      # [B,H+1,obs]
        obs_seq = self._fold(obs_seq.transpose(0, 1), goal)         # -> [H+1,B,obs+3]
        action_t = action.transpose(0, 1)                           # [H,B,act]
        reward_t = reward.transpose(0, 1)                           # [H,B,1]
        term_t = term.transpose(0, 1)                               # [H,B,1]

        return self.core.update_step(obs_seq, action_t, reward_t, term_t, task=None)

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        # z carries the goal cube xyz (shape [3] or [1,3]) from eval_context.
        obs = obs.to(self.device).float()              # [1, obs_dim]
        goal = z.to(self.device).float().reshape(GOAL_DIM)
        folded = torch.cat([obs.reshape(-1), goal], dim=-1)   # [obs_dim+3]
        action = self.core.act(folded, t0=self._t0, eval_mode=eval_mode)  # [act_dim]
        self._t0 = False
        return action.to(self.device).unsqueeze(0)     # [1, act_dim] for rollout

    @torch.no_grad()
    def eval_context(self, *, env, domain, task):
        goal_xyzs = env.unwrapped.cur_task_info["goal_xyzs"]   # [num_cubes, 3]
        goal = torch.as_tensor(goal_xyzs[0], dtype=torch.float32, device=self.device)
        return goal, {}

    def reset(self) -> None:
        self._t0 = True
        with torch.no_grad():
            self.core._prev_mean.zero_()

    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:  # unused for tdmpc2
        raise NotImplementedError("TDMPC2 uses eval_context, not infer_z")

    def state_dict(self) -> dict:
        return {"core": self.core.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.core.load_state_dict(state["core"])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))
