"""train.py — Faithful FB on state-based OGBench, Hydra-configured.

Examples
--------
    python train.py domain=antmaze_medium
    python train.py domain=cube_single                # FlowBC selected by domain
    python train.py --multirun seed=1,2,3 domain=antmaze_medium
"""

from __future__ import annotations

import os
import re

# Headless OpenGL backend for MuJoCo (needed when env.render() is called
# without an X display, e.g. during eval-video capture on a remote host).
os.environ.setdefault("MUJOCO_GL", "egl")

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
from agents.psm.agent import PSMAgent
from agents.psm.flow_bc.agent import PSMFlowBCAgent
from agents.rldp.agent import RLDPAgent
from agents.rldp.flow_bc.agent import RLDPFlowBCAgent
from data.ogbench import load_ogbench_dataset
from envs.ogbench import create_ogbench_env
from evals.ogbench import OGBenchEvaluator
from nn_models import (
    AugmentatorArchiConfig,
    BackwardArchiConfig,
    DrQEncoderArchiConfig,
    ForwardArchiConfig,
    IdentityNNConfig,
    MLPEncoderArchiConfig,
    NoiseConditionedActorArchiConfig,
    SimpleActorArchiConfig,
    SimpleVectorFieldArchiConfig,
)
from normalizers import IdentityNormalizerConfig, RGBNormalizerConfig
from resume import assert_no_clobber, resolve_resume, write_train_state


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


def build_pixel_cfgs(cfg):
    """Select obs-normalizer / rgb-encoder / augmentator configs.

    Defaults (no pixel blocks in cfg) reproduce the state path exactly:
    Identity normalizer + Identity encoder + Identity augmentator.
    """
    norm_block = cfg.get("obs_normalizer", None)
    if norm_block is not None and norm_block.get("name") == "RGBNormalizerConfig":
        obs_normalizer_cfg = RGBNormalizerConfig()
    else:
        obs_normalizer_cfg = IdentityNormalizerConfig()

    rgb_block = cfg.get("rgb_encoder", None)
    if rgb_block is not None and rgb_block.get("name") == "drq":
        rgb_encoder_cfg = DrQEncoderArchiConfig(feature_dim=rgb_block.get("feature_dim"))
    elif rgb_block is not None and rgb_block.get("name") == "mlp":
        # State-side param-matched MLP encoder (cube_single_encmatch ablation).
        rgb_encoder_cfg = MLPEncoderArchiConfig(
            feature_dim=rgb_block.get("feature_dim", 256),
            hidden_dim=rgb_block.get("hidden_dim", 1100),
            hidden_layers=rgb_block.get("hidden_layers", 5),
        )
    else:
        rgb_encoder_cfg = IdentityNNConfig()

    aug_block = cfg.get("augmentator", None)
    if aug_block is not None and aug_block.get("name") == "random_shifts":
        augmentator_cfg = AugmentatorArchiConfig(pad=aug_block.get("pad", 4))
    else:
        augmentator_cfg = IdentityNNConfig()

    return obs_normalizer_cfg, rgb_encoder_cfg, augmentator_cfg


def make_agent(cfg: DictConfig, obs_space, action_dim: int):
    if cfg.agent == "tdmpc2":
        from agents.tdmpc2 import TDMPC2Agent
        t = cfg.get("tdmpc2", {})
        return TDMPC2Agent(
            obs_space,
            action_dim,
            device=cfg.device,
            horizon=cfg.get("horizon", 3),
            batch_size=cfg.batch_size,
            random_goal_ratio=t.get("random_goal_ratio", 0.5),
            success_threshold=t.get("success_threshold", 0.04),
            **{k: v for k, v in t.items()
               if k not in ("random_goal_ratio", "success_threshold", "horizon")},
        )

    if cfg.agent in ("psm", "psm_flowbc"):
        _obs_normalizer_cfg, _rgb_encoder_cfg, _augmentator_cfg = build_pixel_cfgs(cfg)
        psm_shared = dict(
            obs_space=obs_space,
            action_dim=action_dim,
            batch_size=cfg.batch_size,
            z_dim=cfg.z_dim,
            max_log_seed=cfg.max_log_seed,
            norm_z=cfg.norm_z,
            phi_input=cfg.phi_input,
            phi_cfg={"hidden_dim": cfg.phi.hidden_dim,
                     "hidden_layers": cfg.phi.hidden_layers,
                     "norm": cfg.phi.norm,
                     "batch_norm": cfg.phi.batch_norm},
            sf_cfg={"hidden_dim": cfg.sf.hidden_dim,
                    "hidden_layers": cfg.sf.hidden_layers,
                    "embedding_layers": cfg.sf.embedding_layers},
            obs_normalizer_cfg=_obs_normalizer_cfg,
            rgb_encoder_cfg=_rgb_encoder_cfg,
            augmentator_cfg=_augmentator_cfg,
            num_parallel=cfg.sf.num_parallel,
            discount=cfg.discount,
            lr_sf=cfg.lr_sf,
            lr_phi=cfg.lr_phi,
            lr_actor=cfg.lr_actor,
            weight_decay=cfg.weight_decay,
            clip_grad_norm=cfg.clip_grad_norm,
            target_tau=cfg.target_tau,
            ortho_coef=cfg.ortho_coef,
            mix_ratio=cfg.mix_ratio,
            pessimism_penalty=cfg.pessimism_penalty,
            actor_pessimism_penalty=cfg.actor_pessimism_penalty,
            actor_std=cfg.stddev,
            stddev_clip=cfg.stddev_clip,
            amp=cfg.get("amp", False),
            device=cfg.device,
        )
        if cfg.agent == "psm_flowbc":
            actor_cfg = NoiseConditionedActorArchiConfig(
                hidden_dim=cfg.actor.hidden_dim,
                hidden_layers=cfg.actor.hidden_layers,
                embedding_layers=cfg.actor.embedding_layers,
            )
            actor_vf_cfg = SimpleVectorFieldArchiConfig(
                hidden_dim=cfg.actor_vf.hidden_dim,
                hidden_layers=cfg.actor_vf.hidden_layers,
            )
            return PSMFlowBCAgent(
                actor_cfg=actor_cfg,
                actor_vf_cfg=actor_vf_cfg,
                flow_steps=cfg.flow_steps,
                lr_actor_vf=cfg.lr_actor_vf,
                bc_coeff=cfg.get("bc_coeff", 0.0),
                **psm_shared,
            )
        # plain psm: TD3 actor built inside PSMModel from the actor_cfg dict
        psm_shared["actor_cfg"] = {"hidden_dim": cfg.actor.hidden_dim,
                                   "hidden_layers": cfg.actor.hidden_layers,
                                   "embedding_layers": cfg.actor.embedding_layers}
        return PSMAgent(**psm_shared)

    if cfg.agent == "dfb":
        # Density-FB torch port (agents/dfb): two-tower categorical successor
        # density + reward-integrating extraction. Early return: dfb.yaml does
        # not define the FB forward/backward blocks `shared` below requires.
        from agents.dfb.agent import DFBAgent
        actor_kind = cfg.get("actor_kind", "flow")
        if actor_kind == "flow":
            dfb_actor_cfg = NoiseConditionedActorArchiConfig(
                hidden_dim=cfg.actor.hidden_dim,
                hidden_layers=cfg.actor.hidden_layers,
                embedding_layers=cfg.actor.embedding_layers,
            )
        else:
            dfb_actor_cfg = None  # stock gaussian Actor defaults inside DFBModel
        return DFBAgent(
            obs_space,
            action_dim,
            measure_hit_spec=cfg.measure_hit_spec,
            measure_density_dim=cfg.measure_density_dim,
            measure_density_temp=cfg.measure_density_temp,
            measure_density_hard=cfg.measure_density_hard,
            hidden_dims=tuple(cfg.density_hidden_dims),
            actor_kind=actor_kind,
            actor_cfg=dfb_actor_cfg,
            actor_vf_cfg=SimpleVectorFieldArchiConfig(
                hidden_dim=cfg.actor_vf.hidden_dim,
                hidden_layers=cfg.actor_vf.hidden_layers,
            ) if actor_kind == "flow" else None,
            discount=cfg.discount,
            lr_f=cfg.lr_f,
            lr_b=cfg.lr_b,
            lr_actor=cfg.lr_actor,
            lr_actor_vf=cfg.get("lr_actor_vf", 3e-4),
            flow_steps=cfg.get("flow_steps", 10),
            bc_coeff=cfg.get("bc_coeff", 3.0),
            alpha=cfg.get("alpha", 0.3),
            const_std=cfg.get("const_std", True),
            double_q=cfg.get("double_q", True),
            actor_q_center=cfg.get("actor_q_center", True),
            target_tau=cfg.get("target_tau", 0.005),
            weight_decay=cfg.weight_decay,
            device=cfg.device,
        )

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

    _obs_normalizer_cfg, _rgb_encoder_cfg, _augmentator_cfg = build_pixel_cfgs(cfg)
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
        obs_normalizer_cfg=_obs_normalizer_cfg,
        rgb_encoder_cfg=_rgb_encoder_cfg,
        augmentator_cfg=_augmentator_cfg,
        discount=cfg.discount,
        lr_f=cfg.lr_f,
        lr_b=cfg.lr_b,
        lr_actor=cfg.lr_actor,
        weight_decay=cfg.weight_decay,
        clip_grad_norm=cfg.clip_grad_norm,
        ortho_coef=cfg.ortho_coef,
        train_goal_ratio=cfg.train_goal_ratio,
        reward_z_ratio=cfg.get("reward_z_ratio", 0.0),
        reward_z_family=cfg.get("reward_z_family", "kernel"),
        reward_z_bw=tuple(cfg.get("reward_z_bw", [0.2, 2.0])),
        reward_z_slice=tuple(cfg.get("reward_z_slice", [19, 22])),
        fb_pessimism_penalty=cfg.fb_pessimism_penalty,
        actor_pessimism_penalty=cfg.actor_pessimism_penalty,
        actor_std=cfg.actor_std,
        stddev_clip=cfg.stddev_clip,
        f_target_tau=cfg.f_target_tau,
        b_target_tau=cfg.b_target_tau,
        bc_coeff=cfg.get("bc_coeff", 0.0),
        q_loss_coef=cfg.get("q_loss_coef", 0.0),
        reweight_alpha=cfg.get("reweight_alpha", 0.0),
        reweight_clip=cfg.get("reweight_clip", 10.0),
        reweight_density_path=cfg.get("reweight_density_path", None),
        weight_diag=cfg.get("weight_diag", False),
        weight_z=cfg.get("weight_z", False),
        reweight_mode=cfg.get("reweight_mode", "coverage"),
        iwr_c=cfg.get("iwr_c", 0.03),
        iwr_sigma=cfg.get("iwr_sigma", 0.05),
        iwr_eps=cfg.get("iwr_eps", 0.1),
        train_goal_mode=cfg.get("train_goal_mode", "perm"),
        decouple_actor_z=cfg.get("decouple_actor_z", False),
        critic_mode=cfg.get("critic_mode", "measure"),
        iql_expectile=cfg.get("iql_expectile", 0.9),
        iql_thresh=cfg.get("iql_thresh", 0.04),
        onestep=cfg.get("onestep", False),
        decondition_F=cfg.get("decondition_F", False),
        d_norm=cfg.get("d_norm", False),
        bilinear_v=cfg.get("bilinear_v", False),
        critic_goal_mix=cfg.get("critic_goal_mix", (1.0, 0.0, 0.0)),
        actor_goal_mix=cfg.get("actor_goal_mix", (1.0, 0.0, 0.0)),
        iql_reward=cfg.get("iql_reward", "cube_xyz"),
        goal_cond=cfg.get("goal_cond", False),
        fixed_b=cfg.get("fixed_b", "none"),
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

    if cfg.agent == "fb_hitdirac_flowbc":
        # Deconfounding control: stock FB's loss (linear diag + ortho ON) with
        # the Dirac support I -> privileged cube-xyz ball. Isolates the reward
        # support from the bounded/no-ortho objective.
        from agents.fb.hitdirac.agent import FBHitDiracFlowBCAgent
        actor_cfg = NoiseConditionedActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        actor_vf_cfg = SimpleVectorFieldArchiConfig(
            hidden_dim=cfg.actor_vf.hidden_dim,
            hidden_layers=cfg.actor_vf.hidden_layers,
        )
        return FBHitDiracFlowBCAgent(
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            flow_steps=cfg.flow_steps,
            lr_actor_vf=cfg.lr_actor_vf,
            hit_eps=cfg.get("hit_eps", 0.4),
            hit_slice=tuple(cfg.get("hit_slice", [19, 22])),
            hit_spec=cfg.get("hit_spec", ""),
            **shared,
        )

    if cfg.agent == "fb_measure_flowbc":
        # Two-phase bounded successor measure in FB's own house (the jax
        # gciql_flow recipe): hit-referenced squared residual, no ortho.
        from agents.fb.measure.agent import FBMeasureFlowBCAgent
        actor_cfg = NoiseConditionedActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        actor_vf_cfg = SimpleVectorFieldArchiConfig(
            hidden_dim=cfg.actor_vf.hidden_dim,
            hidden_layers=cfg.actor_vf.hidden_layers,
        )
        return FBMeasureFlowBCAgent(
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            flow_steps=cfg.flow_steps,
            lr_actor_vf=cfg.lr_actor_vf,
            measure_mode=cfg.get("measure_mode", "first_passage"),
            hit_eps=cfg.get("hit_eps", 0.4),
            hit_slice=tuple(cfg.get("hit_slice", [19, 22])),
            hit_identity=cfg.get("hit_identity", False),
            scale_norm=cfg.get("scale_norm", True),
            freeze_b=cfg.get("freeze_b", False),
            nhat_head=cfg.get("nhat_head", False),
            warm_start_ckpt=cfg.get("warm_start_ckpt", ""),
            nonneg_coef=cfg.get("nonneg_coef", 0.0),
            mass_coef=cfg.get("mass_coef", 0.0),
            hit_kernel=cfg.get("hit_kernel", "ball"),
            hit_spec=cfg.get("hit_spec", ""),
            **shared,
        )

    if cfg.agent == "fb_measure":
        # NO-FLOW arm: the same region-measure critic behind FBAgent's native
        # TD3+BC actor. Same kwargs as fb_measure_flowbc minus the flow actor's
        # (actor_cfg / actor_vf_cfg / flow_steps / lr_actor_vf).
        from agents.fb.measure.agent import FBMeasureAgent
        return FBMeasureAgent(
            measure_mode=cfg.get("measure_mode", "first_passage"),
            hit_eps=cfg.get("hit_eps", 0.4),
            hit_slice=tuple(cfg.get("hit_slice", [19, 22])),
            hit_identity=cfg.get("hit_identity", False),
            scale_norm=cfg.get("scale_norm", True),
            freeze_b=cfg.get("freeze_b", False),
            nhat_head=cfg.get("nhat_head", False),
            warm_start_ckpt=cfg.get("warm_start_ckpt", ""),
            nonneg_coef=cfg.get("nonneg_coef", 0.0),
            mass_coef=cfg.get("mass_coef", 0.0),
            hit_kernel=cfg.get("hit_kernel", "ball"),
            hit_spec=cfg.get("hit_spec", ""),
            **shared,
        )

    if cfg.agent == "isfb_flowbc":
        from agents.fb.isfb.agent import ISFBAgent
        actor_cfg = NoiseConditionedActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        actor_vf_cfg = SimpleVectorFieldArchiConfig(
            hidden_dim=cfg.actor_vf.hidden_dim,
            hidden_layers=cfg.actor_vf.hidden_layers,
        )
        return ISFBAgent(
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            flow_steps=cfg.flow_steps,
            lr_actor_vf=cfg.lr_actor_vf,
            measure_insample=cfg.get("measure_insample", "scalar_gate_hard"),
            alpha_n=cfg.get("alpha_n", 1.0),
            alpha_m=cfg.get("alpha_m", 1.0),
            awr_beta=cfg.get("awr_beta", 3.0),
            awr_wmax=cfg.get("awr_wmax", 100.0),
            **shared,
        )

    if cfg.agent == "rldp":
        from nn_models import VForwardArchiConfig
        predictor_cfg = VForwardArchiConfig(
            hidden_dim=cfg.predictor.hidden_dim,
            hidden_layers=cfg.predictor.hidden_layers,
            embedding_layers=cfg.predictor.embedding_layers,
            num_parallel=cfg.predictor.num_parallel,
        )
        actor_cfg = SimpleActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        return RLDPAgent(
            actor_cfg=actor_cfg,
            predictor_cfg=predictor_cfg,
            horizon=cfg.horizon,
            **shared,
        )

    if cfg.agent == "rldp_flowbc":
        from nn_models import VForwardArchiConfig
        predictor_cfg = VForwardArchiConfig(
            hidden_dim=cfg.predictor.hidden_dim,
            hidden_layers=cfg.predictor.hidden_layers,
            embedding_layers=cfg.predictor.embedding_layers,
            num_parallel=cfg.predictor.num_parallel,
        )
        actor_cfg = NoiseConditionedActorArchiConfig(
            hidden_dim=cfg.actor.hidden_dim,
            hidden_layers=cfg.actor.hidden_layers,
            embedding_layers=cfg.actor.embedding_layers,
        )
        actor_vf_cfg = SimpleVectorFieldArchiConfig(
            hidden_dim=cfg.actor_vf.hidden_dim,
            hidden_layers=cfg.actor_vf.hidden_layers,
        )
        return RLDPFlowBCAgent(
            actor_cfg=actor_cfg,
            actor_vf_cfg=actor_vf_cfg,
            predictor_cfg=predictor_cfg,
            horizon=cfg.horizon,
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

    save_dir = Path(cfg.save_dir)
    plan = resolve_resume(save_dir, cfg.get("resume", False), cfg.get("resume_from"))
    if plan.ckpt_path is None:
        assert_no_clobber(save_dir, cfg.get("resume", False), cfg.get("force", False))
    save_dir.mkdir(parents=True, exist_ok=True)
    if plan.ckpt_path is not None:
        print(f"[resume] from {plan.ckpt_path} at step {plan.start_step}")

    if cfg.use_wandb:
        import wandb
        run_name = cfg.wandb_run_name or f"{cfg.domain}__{cfg.agent}__s{cfg.seed}"
        init_kwargs = dict(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=os.getcwd(),
            save_code=cfg.wandb_save_code,
        )
        if plan.wandb_run_id:
            init_kwargs["id"] = plan.wandb_run_id
            init_kwargs["resume"] = "allow"
        wandb.init(**init_kwargs)
        wandb_run_id = wandb.run.id
    else:
        wandb_run_id = None

    print(f"[train] env: {cfg.domain}")
    env, _ = create_ogbench_env(
        cfg.domain,
        obs_type=cfg.obs_type,
        seed=cfg.seed,
        frame_stack=cfg.get("frame_stack", 1),
    )
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
        obs_type=cfg.obs_type,
        frame_stack=cfg.get("frame_stack", 1),
        # PSM's proto-behavior sampler must key on the GLOBAL buffer row index of
        # each transition (ref baselines/PSM/agent/psm.py:255 next_observation_hash
        # = np.arange(N)). Enable the row-index passthrough so sample() emits
        # batch["index"], which PSM*Agent.update consumes as next_obs_hash.
        # Default-off for FB/all other agents (byte-identical: no "index" key).
        with_index=cfg.agent in ("psm", "psm_flowbc"),
    )
    print(f"[train] buffer size: {len(buffer):,}")

    agent = make_agent(cfg, obs_space, action_dim)
    print(f"[train] agent: {type(agent).__name__}")

    if plan.ckpt_path is not None:
        agent.load_state_dict(torch.load(plan.ckpt_path, map_location=cfg.device))
        set_seed(cfg.seed + plan.start_step)   # avoid replaying identical minibatches
        print(f"[resume] agent state loaded; continuing from step {plan.start_step}")

    video_dir = Path(cfg.eval_videos_dir)
    if cfg.save_eval_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    evaluator = OGBenchEvaluator(
        domain=cfg.domain,
        agent=agent,
        offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size,
        n_episodes=cfg.eval_n_episodes,
        shift_reward=cfg.eval_shift_reward,
        obs_type=cfg.obs_type,
        frame_stack=cfg.get("frame_stack", 1),
        seed=cfg.seed,
        device=cfg.device,
        save_videos=cfg.save_eval_videos,
        video_dir=video_dir,
        use_wandb=cfg.use_wandb,
    )

    print(f"[train] {cfg.num_train_steps:,} steps")
    # Match td_jepa Workspace.train_offline:
    #   - loop range(0, N+1) — runs one extra "warm-up" optimizer step at t=0
    #   - log at t=0 (divisor=1) and every log_every steps (divisor=log_every)
    #   - skip checkpoint at t=0 (matches `t != self._checkpoint_time` guard)
    #   - eval triggers at t=0 if step_zero_should_trigger (configurable)
    start_time = time.time()
    fps_start_time = time.time()
    total_metrics: Dict[str, float] | None = None

    for step in range(plan.start_step, cfg.num_train_steps + 1):
        # Checkpoint (skip t=0 per td_jepa convention)
        if step > 0 and step % cfg.save_every == 0:
            ckpt = save_dir / f"step_{step}.pt"
            torch.save(agent.state_dict(), ckpt)
            write_train_state(save_dir, step, wandb_run_id, ckpt.name)
            print(f"[ckpt] {ckpt}")

        # Eval (td_jepa's EveryNStepsChecker triggers at t=0 and every N)
        if cfg.eval_every > 0 and (step == 0 or step % cfg.eval_every == 0):
            print(f"[eval] step={step}")
            eval_out = evaluator.run(step=step)
            log_metrics(eval_out, step, use_wandb=cfg.use_wandb, prefix="eval/reward/")
            # Unified eval keys shared with the jax/OGBench runs (success/overall,
            # success/taskN) so both codebases land on the same wandb panels; see
            # the mirror in tools/wandb_mode_shim/sitecustomize.py.
            unified: Dict[str, float] = {}
            if "eval/success" in eval_out:
                unified["success/overall"] = eval_out["eval/success"]
            for _k, _v in eval_out.items():
                _m = re.search(r"singletask-task(\d+)-v0/success$", _k)
                if _m:
                    unified[f"success/task{_m.group(1)}"] = _v
            if unified:
                log_metrics(unified, step, use_wandb=cfg.use_wandb)

        # Update
        horizon = getattr(cfg, "horizon", 1)
        # zHER (train_goal_mode=traj): attach each transition's discounted
        # trajectory-future next-state as batch["future"] for goal-derived z's.
        future_gamma = (float(cfg.discount)
                        if (str(getattr(cfg, "train_goal_mode", "perm")) == "traj"
                            or str(getattr(cfg, "critic_mode", "measure")) == "iql"
                            or bool(getattr(cfg, "decouple_actor_z", False))) else None)
        batch = buffer.sample(cfg.batch_size, horizon=horizon, future_gamma=future_gamma)
        metrics = agent.update(batch, step)

        # Accumulate (matches td_jepa: copy on first, add on subsequent)
        if total_metrics is None:
            total_metrics = {k: float(v) for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                total_metrics[k] = total_metrics[k] + float(v)

        # Log at t=0 and every log_every steps (matches td_jepa log_time_checker)
        if step == 0 or step % cfg.log_every == 0:
            divisor = 1 if step == 0 else cfg.log_every
            m_dict = {k: round(v / divisor, 6) for k, v in total_metrics.items()}
            m_dict["duration"] = time.time() - start_time
            m_dict["FPS"] = divisor / (time.time() - fps_start_time)
            log_metrics(m_dict, step, use_wandb=cfg.use_wandb, prefix="train/")
            total_metrics = None
            fps_start_time = time.time()

    torch.save(agent.state_dict(), save_dir / "final.pt")
    print("[train] Done.")

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
