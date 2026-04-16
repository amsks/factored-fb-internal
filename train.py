"""
train.py — FB on OGBench, Hydra-configured.

Examples
--------
python train.py domain=antmaze-medium-navigate-v0
python train.py domain=antmaze-medium-navigate-v0 agent=fb_flowbc
python train.py +experiment=smoke_test domain=antmaze-medium-navigate-v0
python train.py --multirun seed=1,2,3 domain=antmaze-medium-navigate-v0
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agents.fb.agent import FBAgent
from agents.fb.flow_bc.agent import FBFlowBCAgent
from buffers.transition import DictBuffer
from data.ogbench import load_ogbench_dataset
from envs.ogbench import create_ogbench_env
from evals.ogbench import OGBenchEvaluator
from nn_models import ForwardArchiConfig, BackwardArchiConfig, SimpleActorArchiConfig, NoiseConditionedActorArchiConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable — falling back to CPU")
        return "cpu"
    return requested


def make_agent(cfg: DictConfig, obs_space, action_dim: int) -> FBAgent:
    forward_cfg = ForwardArchiConfig(
        hidden_dim=cfg.hidden_dim,
        num_parallel=cfg.n_ensemble,
    )
    backward_cfg = BackwardArchiConfig(
        hidden_dim=cfg.hidden_dim,
    )

    shared = dict(
        obs_space=obs_space,
        action_dim=action_dim,
        z_dim=cfg.z_dim,
        forward_cfg=forward_cfg,
        backward_cfg=backward_cfg,
        discount=cfg.discount,
        lr_f=cfg.lr_f,
        lr_b=cfg.lr_b,
        lr_actor=cfg.lr_actor,
        ortho_coef=cfg.ortho_coef,
        pessimism=cfg.pessimism,
        train_goal_ratio=cfg.train_goal_ratio,
        actor_std=cfg.actor_std,
        f_target_tau=cfg.f_target_tau,
        b_target_tau=cfg.b_target_tau,
        device=cfg.device,
    )
    if cfg.agent == "fb_flowbc":
        actor_cfg = NoiseConditionedActorArchiConfig(hidden_dim=cfg.hidden_dim)
        return FBFlowBCAgent(actor_cfg=actor_cfg, bc_coeff=cfg.bc_coeff,
                             n_action_samples=cfg.flow_steps, **shared)

    actor_cfg = SimpleActorArchiConfig(hidden_dim=cfg.hidden_dim)
    return FBAgent(actor_cfg=actor_cfg, **shared)


def log_metrics(metrics: Dict[str, float], step: int, *, use_wandb: bool, prefix: str = "") -> None:
    tagged = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics
    if use_wandb:
        import wandb
        wandb.log(tagged, step=step)
    kv = "  ".join(f"{k}={v:.4f}" for k, v in tagged.items())
    print(f"[step {step:>8d}]  {kv}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = DictConfig(cfg)
    cfg.device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))

    # ── W&B ──────────────────────────────────────────────────────────── #
    if cfg.use_wandb:
        import wandb
        run_name = cfg.wandb_run_name or f"{cfg.domain}__{cfg.agent}__s{cfg.seed}"
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.getcwd(),
        )

    # ── Environment (single env for obs/act space; evals build their own) ─ #
    print(f"[train] env: {cfg.domain}")
    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=cfg.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    print(f"[train] obs_space={obs_space.shape}  action_dim={action_dim}")

    # ── Offline data ──────────────────────────────────────────────────── #
    print(f"[train] Loading dataset: {cfg.domain}")
    buffer = load_ogbench_dataset(
        domain=cfg.domain,
        device=cfg.device,
        n_transitions=cfg.n_transitions if cfg.get("n_transitions") else None,
    )
    print(f"[train] Buffer size: {len(buffer):,}")

    # ── Agent ─────────────────────────────────────────────────────────── #
    agent = make_agent(cfg, obs_space, action_dim)
    print(f"[train] Agent: {type(agent).__name__}")

    # ── Evaluator ─────────────────────────────────────────────────────── #
    evaluator = OGBenchEvaluator(
        domain=cfg.domain,
        agent=agent,
        offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size,
        n_episodes=cfg.eval_n_episodes,
        obs_type=cfg.obs_type,
        seed=cfg.seed,
        device=cfg.device,
    )

    # ── Save dir (Hydra sets cwd automatically) ────────────────────────── #
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────── #
    # Training loop
    # ──────────────────────────────────────────────────────────────────── #
    print(f"[train] {cfg.num_train_steps:,} steps")
    t0 = time.time()
    accum: Dict[str, float] = {}
    accum_n = 0

    for step in range(1, cfg.num_train_steps + 1):
        batch = buffer.sample(cfg.batch_size)
        metrics = agent.update(batch, step)

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        accum_n += 1

        if step % cfg.log_every == 0:
            avg = {k: v / accum_n for k, v in accum.items()}
            avg["misc/steps_per_sec"] = cfg.log_every / (time.time() - t0)
            log_metrics(avg, step, use_wandb=cfg.use_wandb, prefix="train/")
            accum, accum_n = {}, 0
            t0 = time.time()

        if step % cfg.eval_every == 0:
            print(f"[eval] step={step}")
            log_metrics(evaluator.run(), step, use_wandb=cfg.use_wandb, prefix="eval/")

        if step % cfg.save_every == 0:
            ckpt = save_dir / f"step_{step}.pt"
            torch.save(agent.state_dict(), ckpt)
            print(f"[ckpt] {ckpt}")

    log_metrics(evaluator.run(), cfg.num_train_steps, use_wandb=cfg.use_wandb, prefix="eval/")
    torch.save(agent.state_dict(), save_dir / "final.pt")
    print("[train] Done.")

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()