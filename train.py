"""train.py — Faithful FB on state-based OGBench, Hydra-configured.

Examples
--------
    python train.py domain=antmaze_medium
    python train.py domain=cube_single                # FlowBC selected by domain
    python train.py --multirun seed=1,2,3 domain=antmaze_medium
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

# Match td_jepa/train.py: allow TF32 matmuls on CUDA. Affects every F @ B.T,
# B @ B.T, F @ z reduction. Without this, our F-side metrics (M1, Q_actor,
# fb_offdiag) drift ~6-22% from td_jepa over 50k steps.
torch.set_float32_matmul_precision("high")

from agents.fb.agent import FBAgent
from agents.fb.flow_bc.agent import FBFlowBCAgent
from data.ogbench import load_ogbench_dataset
from envs.ogbench import create_ogbench_env
from evals.ogbench import OGBenchEvaluator
from nn_models import (
    BackwardArchiConfig,
    ForwardArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
from normalizers import IdentityNormalizerConfig


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


def make_agent(cfg: DictConfig, obs_space, action_dim: int):
    forward_cfg = ForwardArchiConfig(
        hidden_dim=cfg.forward.hidden_dim,
        hidden_layers=cfg.forward.hidden_layers,
        embedding_layers=cfg.forward.embedding_layers,
        num_parallel=cfg.forward.num_parallel,
    )
    backward_cfg = BackwardArchiConfig(
        hidden_dim=cfg.backward.hidden_dim,
        hidden_layers=cfg.backward.hidden_layers,
        norm=cfg.backward.norm,
    )
    left_encoder_cfg = BackwardArchiConfig(
        hidden_dim=cfg.left_encoder.hidden_dim,
        hidden_layers=cfg.left_encoder.hidden_layers,
        norm=cfg.left_encoder.norm,
    )

    shared = dict(
        obs_space=obs_space,
        action_dim=action_dim,
        batch_size=cfg.batch_size,
        z_dim=cfg.z_dim,
        L_dim=cfg.L_dim,
        actor_encode_obs=cfg.actor_encode_obs,
        forward_cfg=forward_cfg,
        backward_cfg=backward_cfg,
        left_encoder_cfg=left_encoder_cfg,
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        discount=cfg.discount,
        lr_f=cfg.lr_f,
        lr_b=cfg.lr_b,
        lr_actor=cfg.lr_actor,
        weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm,
        ortho_coef=cfg.ortho_coef,
        train_goal_ratio=cfg.train_goal_ratio,
        fb_pessimism_penalty=cfg.fb_pessimism_penalty,
        actor_pessimism_penalty=cfg.actor_pessimism_penalty,
        actor_std=cfg.actor_std,
        stddev_clip=cfg.stddev_clip,
        f_target_tau=cfg.f_target_tau,
        b_target_tau=cfg.b_target_tau,
        bc_coeff=cfg.get("bc_coeff", 0.0),
        q_loss_coef=cfg.get("q_loss_coef", 0.0),
        amp=cfg.get("amp", False),
        device=cfg.device,
    )

    if cfg.agent == "fb_flowbc":
        actor_cfg = NoiseConditionedActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        actor_vf_cfg = SimpleVectorFieldArchiConfig(
            hidden_dim=cfg.actor_vf.hidden_dim,
            hidden_layers=cfg.actor_vf.hidden_layers,
        )
        return FBFlowBCAgent(
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            flow_steps=cfg.flow_steps,
            lr_actor_vf=cfg.lr_actor_vf,
            **shared,
        )

    actor_cfg = SimpleActorArchiConfig(
        hidden_dim=cfg.actor.hidden_dim,
        hidden_layers=cfg.actor.hidden_layers,
        embedding_layers=cfg.actor.embedding_layers,
    )
    return FBAgent(actor_cfg=actor_cfg, **shared)


def log_metrics(metrics: Dict[str, float], step: int, *, use_wandb: bool, prefix: str = "") -> None:
    tagged = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else metrics
    if use_wandb:
        import wandb
        wandb.log(tagged, step=step)
    kv = "  ".join(f"{k}={v:.4f}" for k, v in tagged.items())
    print(f"[step {step:>8d}]  {kv}")


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg.device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    if cfg.use_wandb:
        import wandb
        run_name = cfg.wandb_run_name or f"{cfg.domain}__{cfg.agent}__s{cfg.seed}"
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.getcwd(),
            save_code=cfg.wandb_save_code,
        )

    print(f"[train] env: {cfg.domain}")
    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=cfg.seed)
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    print(f"[train] obs_space={obs_space.shape}  action_dim={action_dim}")

    print(f"[train] loading dataset for {cfg.domain}")
    buffer = load_ogbench_dataset(
        domain=cfg.domain,
        data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes,
        device=cfg.device,
        n_transitions=cfg.n_transitions,
    )
    print(f"[train] buffer size: {len(buffer):,}")

    agent = make_agent(cfg, obs_space, action_dim)
    print(f"[train] agent: {type(agent).__name__}")

    evaluator = OGBenchEvaluator(
        domain=cfg.domain,
        agent=agent,
        offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size,
        n_episodes=cfg.eval_n_episodes,
        shift_reward=cfg.eval_shift_reward,
        obs_type=cfg.obs_type,
        seed=cfg.seed,
        device=cfg.device,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
