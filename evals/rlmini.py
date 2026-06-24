"""Evaluator for rlmini_envs environments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from envs.rlmini import create_rlmini_env
from envs.rollout import rollout


class RLMiniEvaluator:
    def __init__(
        self,
        domain: str,
        agent,
        offline_buffer,
        n_episodes: int = 10,
        obs_type: str = "state",
        frame_stack: int = 1,
        seed: int = 0,
        save_videos: bool = False,
        video_dir: Optional[Path | str] = None,
        use_wandb: bool = False,
        cfg=None,
        **_,
    ):
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.n_episodes = n_episodes
        self.obs_type = obs_type
        self.frame_stack = frame_stack
        self.seed = seed
        self.save_videos = save_videos
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.use_wandb = use_wandb
        self.cfg = cfg
        if self.save_videos and self.video_dir is None:
            raise ValueError("save_videos=True requires video_dir to be set")

    def _eval_context(self):
        if hasattr(self.agent, "model") and hasattr(self.agent.model, "sample_z"):
            return self.agent.model.sample_z(1).squeeze(0)
        return None

    def run(self, step: int = 0) -> Dict[str, float]:
        env, _ = create_rlmini_env(
            self.domain,
            cfg=self.cfg,
            seed=self.seed,
            obs_type=self.obs_type,
            frame_stack=self.frame_stack,
        )
        try:
            stats, infos, frames = rollout(
                env,
                self.agent,
                self.n_episodes,
                ctx=self._eval_context(),
                record=self.save_videos,
            )
        finally:
            env.close()

        rewards = np.array(stats["reward"], dtype=np.float32)
        successes = np.array(
            [any(bool(step_info.get("success", False)) for step_info in ep_info) for ep_info in infos],
            dtype=np.float32,
        )
        metrics = {
            "eval/reward": float(np.mean(rewards)),
            "eval/reward#std": float(np.std(rewards)),
            "eval/success": float(np.mean(successes)),
        }
        if self.save_videos and frames is not None and len(frames) > 0:
            self._save_gif(step=step, episode_frames=frames[0])
        return metrics

    def _save_gif(self, *, step: int, episode_frames: list) -> None:
        import imageio

        ep_arr = np.stack(episode_frames)
        step_dir = self.video_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        gif_path = step_dir / f"{self.domain}.gif"
        imageio.mimsave(gif_path, ep_arr, format="GIF", fps=30, loop=0)
        print(f"[eval] wrote {gif_path}")

        if self.use_wandb:
            import wandb

            wandb.log(
                {f"eval_video/{self.domain}": wandb.Video(str(gif_path), fps=30, format="gif")},
                step=step,
            )
