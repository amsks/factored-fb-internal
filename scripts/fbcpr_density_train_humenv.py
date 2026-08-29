# COPY of metamotivo examples/fbcpr_train_humenv.py (main branch, fetched
# 2026-08-27) with two kinds of edits: (1) EXACTLY TWO lines marked DENSITY-SWAP —
# the import and the agent class; (2) checkpoint/resume support, every edit marked
# RESUME-SUPPORT (full-state checkpoints, --resume flag, signal handling).
# Every config value, schedule, buffer, and eval is theirs.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import collections
import dataclasses
import json
import numbers
import random
import signal  # RESUME-SUPPORT
import sys  # RESUME-SUPPORT
import time
from pathlib import Path
from typing import List

import gymnasium
import humenv
import numpy as np
import tyro
from gymnasium.wrappers import TimeAwareObservation
from humenv import make_humenv
from humenv.bench import (
    RewardEvaluation,
    TrackingEvaluation,
)
from humenv.misc.motionlib import canonicalize, load_episode_based_h5
from packaging.version import Version
from tqdm import tqdm

import wandb
from metamotivo.buffers.buffers import DictBuffer, TrajectoryBuffer
from metamotivo.fb_cpr import FBcprAgentConfig
from impls.torch_density.fbcpr_density import DensityFBcprAgent  # DENSITY-SWAP
from metamotivo.wrappers.humenvbench import RewardWrapper, TrackingWrapper

if Version(humenv.__version__) < Version("0.1.2"):
    raise RuntimeError("This script requires humenv>=0.1.2")
if Version(gymnasium.__version__) < Version("1.0"):
    raise RuntimeError("This script requires gymnasium>=1.0")


# RESUME-SUPPORT: cooperative shutdown. Slurm sends SIGTERM before a walltime
# kill, and the launcher requests USR1 10 minutes early (#SBATCH --signal=B:USR1@600,
# forwarded by the batch shell). The handler only sets a flag; the training loop
# checks it once per iteration, writes a full-state checkpoint, and exits with
# code 99 so the launcher knows to resubmit with --resume.
_STOP_SIGNAL = {"num": None}


def _request_stop(signum, frame):  # RESUME-SUPPORT
    _STOP_SIGNAL["num"] = signum


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_expert_trajectories(motions: str | Path, motions_root: str | Path, device: str, sequence_length: int) -> TrajectoryBuffer:
    with open(motions, "r") as txtf:
        h5files = [el.strip().replace(" ", "") for el in txtf.readlines()]
    episodes = []
    for h5 in tqdm(h5files, leave=False):
        h5 = canonicalize(h5, base_path=motions_root)
        _ep = load_episode_based_h5(h5, keys=None)
        for el in _ep:
            el["observation"] = el["observation"].astype(np.float32)
            del el["file_name"]
        episodes.extend(_ep)
    buffer = TrajectoryBuffer(
        capacity=len(episodes),
        seq_length=sequence_length,
        device=device,
    )
    buffer.extend(episodes)
    return buffer


@dataclasses.dataclass
class TrainConfig:
    seed: int = 0
    motions: str = ""
    motions_root: str = ""
    buffer_size: int = 5_000_000
    online_parallel_envs: int = 50
    log_every_updates: int = 100_000
    work_dir: str | None = None
    num_env_steps: int = 30_000_000
    update_agent_every: int | None = None
    num_seed_steps: int | None = None
    num_agent_updates: int | None = None
    checkpoint_every_steps: int = 5_000_000
    resume: bool = False  # RESUME-SUPPORT: restore from work_dir/resume_ckpt if a checkpoint exists, else start fresh
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = "fbcpr_humenv"

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"
    buffer_device: str = "cpu"

    # eval
    evaluate: bool = False
    eval_every_steps: int = 1_000_000
    reward_eval_num_envs: int = 5
    reward_eval_num_eval_episodes: int = 10
    reward_eval_num_inference_samples: int = 50_000
    reward_eval_tasks: List[str] | None = None

    tracking_eval_num_envs: int = 60
    tracking_eval_motions: str | None = None
    tracking_eval_motions_root: str | None = None

    def __post_init__(self):
        if self.reward_eval_tasks is None:
            # this is just a subset of the tasks available in humenv
            self.reward_eval_tasks = [
                "move-ego-0-0",
                "jump-2",
                "move-ego-0-2",
                "move-ego-90-2",
                "move-ego-180-2",
                "rotate-x-5-0.8",
                "rotate-y-5-0.8",
                "rotate-z-5-0.8"
            ]
        if self.update_agent_every is None:
            self.update_agent_every = 10 * self.online_parallel_envs
        if self.num_seed_steps is None:
            self.num_seed_steps = 1000 * self.online_parallel_envs
        if self.num_agent_updates is None:
            self.num_agent_updates = self.online_parallel_envs
        if self.prioritization:
            # NOTE: when using prioritization train and eval motions must match
            self.tracking_eval_motions = self.motions
            self.tracking_eval_motions_root = self.motions_root
            self.evaluate = True


class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBcprAgentConfig) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        if self.cfg.work_dir is None:
            import string

            tmp_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            self.work_dir = Path.cwd() / "tmp_fbcpr" / tmp_name
            self.cfg.work_dir = str(self.work_dir)
        else:
            self.work_dir = self.cfg.work_dir
        print(f"Workdir: {self.work_dir}")
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        set_seed_everywhere(self.cfg.seed)
        self.agent = DensityFBcprAgent(**dataclasses.asdict(agent_cfg))  # DENSITY-SWAP

        if self.cfg.use_wandb:
            exp_name = "fbcpr"
            wandb_name = exp_name
            # fmt: off
            wandb_config = dataclasses.asdict(self.cfg)
            wandb.init(entity=self.cfg.wandb_ename, project=self.cfg.wandb_pname,
                group=self.cfg.wandb_gname, name=wandb_name,  # mode="disabled",
                config=wandb_config)  # type: ignore
            # fmt: on

        with (self.work_dir / "config.json").open("w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

        self.manager = None

    def train(self):
        self.start_time = time.time()
        self.train_online()

    # RESUME-SUPPORT: full training-state checkpoint. Contents:
    #   * self.agent._model.state_dict() — covers every net INCLUDING the target
    #     copies (_target_forward_map/_target_backward_map from fb/model.py
    #     _prepare_for_train, _target_critic from fb_cpr/model.py) and the
    #     BatchNorm1d obs normalizer's running stats, since all are registered
    #     submodules of FBcprModel.
    #   * all five optimizers that setup_training creates: forward/backward/actor
    #     (metamotivo fb/agent.py) + critic/discriminator (fb_cpr/agent.py).
    #   * the agent's z_buffer (rollout-context reservoir) and the online 'train'
    #     DictBuffer. The buffer storage tensors are preallocated at full capacity;
    #     they are saved whole (idx/is_full mark the valid region) to avoid a
    #     multi-GB transient clone of a sliced copy.
    #   * the 'expert_slicer' TrajectoryBuffer is rebuilt from the motions files at
    #     startup, so only its priorities vector (mutated when prioritization is on)
    #     is persisted. The env-side motion_buffer priorities are NOT persisted
    #     (prioritization is off in the production run).
    #   * env-step counter and python/numpy/torch/cuda RNG states.
    # Written atomically (tmp file then os.replace); the latest 2 kept.
    def save_resume_checkpoint(self, t: int, replay_buffer, reason: str) -> None:
        ckpt_dir = self.work_dir / "resume_ckpt"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        train_buf = replay_buffer["train"]
        z_buf = self.agent.z_buffer
        resume_t = t + self.cfg.online_parallel_envs  # the iteration at step t is complete when this runs
        state = {
            "resume_t": resume_t,
            "num_env_steps": self.cfg.num_env_steps,
            "model": self.agent._model.state_dict(),
            "optimizers": {
                "forward": self.agent.forward_optimizer.state_dict(),
                "backward": self.agent.backward_optimizer.state_dict(),
                "actor": self.agent.actor_optimizer.state_dict(),
                "critic": self.agent.critic_optimizer.state_dict(),
                "discriminator": self.agent.discriminator_optimizer.state_dict(),
            },
            "z_buffer": {
                "storage": z_buf._storage.detach().to("cpu"),
                "idx": z_buf._idx,
                "is_full": z_buf._is_full,
            },
            "train_buffer": {
                "storage": train_buf.storage,
                "idx": train_buf._idx,
                "is_full": train_buf._is_full,
                "capacity": train_buf.capacity,
                "non_nested_key": getattr(train_buf, "_non_nested_key", None),
            },
            "expert_priorities": replay_buffer["expert_slicer"].priorities,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        start = time.time()
        tmp_path = ckpt_dir / "tmp_ckpt.pt"
        torch.save(state, tmp_path)
        final_path = ckpt_dir / f"ckpt_{resume_t:010d}.pt"
        os.replace(tmp_path, final_path)  # atomic within the same filesystem
        size_gb = final_path.stat().st_size / 1e9
        print(
            f"[resume-ckpt] wrote {final_path.name} ({size_gb:.2f} GB, buffer len {len(train_buf)}) "
            f"in {time.time() - start:.1f}s (reason={reason})",
            flush=True,
        )
        for old in sorted(ckpt_dir.glob("ckpt_*.pt"))[:-2]:  # keep the latest 2
            old.unlink()

    # RESUME-SUPPORT: restore everything saved above. Returns the env step to
    # continue from (0 when no checkpoint exists -> fresh start). The online envs
    # cannot be restored mid-episode; the caller resets them, which loses at most
    # one partial episode per parallel env, and the rollout context z is resampled
    # from the restored z_buffer on the first loop iteration.
    def load_resume_checkpoint(self, replay_buffer) -> int:
        ckpt_dir = self.work_dir / "resume_ckpt"
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt")) if ckpt_dir.exists() else []
        if not ckpts:
            print("[resume-ckpt] no checkpoint found, starting fresh", flush=True)
            return 0
        path = ckpts[-1]
        state = torch.load(path, map_location=None, weights_only=False)
        self.agent._model.load_state_dict(state["model"])
        self.agent.forward_optimizer.load_state_dict(state["optimizers"]["forward"])
        self.agent.backward_optimizer.load_state_dict(state["optimizers"]["backward"])
        self.agent.actor_optimizer.load_state_dict(state["optimizers"]["actor"])
        self.agent.critic_optimizer.load_state_dict(state["optimizers"]["critic"])
        self.agent.discriminator_optimizer.load_state_dict(state["optimizers"]["discriminator"])
        z_buf = self.agent.z_buffer
        z_buf._storage.copy_(state["z_buffer"]["storage"].to(z_buf._storage.device))
        z_buf._idx = state["z_buffer"]["idx"]
        z_buf._is_full = state["z_buffer"]["is_full"]
        train_buf = replay_buffer["train"]
        tb = state["train_buffer"]
        if tb["storage"] is not None:
            assert tb["capacity"] == train_buf.capacity, (
                f"buffer_size changed: checkpoint {tb['capacity']} vs config {train_buf.capacity}"
            )
            train_buf.storage = tb["storage"]
            train_buf._idx = tb["idx"]
            train_buf._is_full = tb["is_full"]
            if tb["non_nested_key"] is not None:
                train_buf._non_nested_key = tb["non_nested_key"]
        expert_buf = replay_buffer["expert_slicer"]
        if state["expert_priorities"] is not None and expert_buf.priorities is not None:
            expert_buf.priorities.copy_(state["expert_priorities"].to(expert_buf.priorities.device))
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        print(
            f"[resume-ckpt] restored {path.name}: resuming at t={state['resume_t']} "
            f"(train buffer len {len(train_buf)}, z_buffer len {len(z_buf)})",
            flush=True,
        )
        return state["resume_t"]

    def train_online(self) -> None:
        print("Loading expert trajectories")
        expert_buffer = load_expert_trajectories(self.cfg.motions, self.cfg.motions_root, device=self.cfg.buffer_device, sequence_length=self.agent_cfg.model.seq_length)

        print("Creating the training environment")
        train_env, mp_info = make_humenv(
            num_envs=self.cfg.online_parallel_envs,
            # vectorization_mode="sync",
            wrappers=[
                gymnasium.wrappers.FlattenObservation,
                lambda env: TimeAwareObservation(env, flatten=False),
            ],
            render_width=320,
            render_height=320,
            motions=self.cfg.motions,
            motion_base_path=self.cfg.motions_root,
            fall_prob=0.2,
            state_init="MoCapAndFall",
        )

        print("Allocating buffers")
        replay_buffer = {
            "train": DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device),
            "expert_slicer": expert_buffer,
        }

        # RESUME-SUPPORT: restore model/optimizers/buffers/RNG and get the step to
        # continue from. The envs below are freshly reset regardless (they cannot be
        # restored mid-episode); the logging aggregates (total_metrics) restart too.
        # The seeding phase is skipped automatically when start_t >= num_seed_steps
        # because the loop's `t < num_seed_steps` check uses the absolute step.
        start_t = 0
        if self.cfg.resume:
            start_t = self.load_resume_checkpoint(replay_buffer)

        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps)
        progb.update(start_t)  # RESUME-SUPPORT
        td, info = train_env.reset()
        done = np.zeros(self.cfg.online_parallel_envs, dtype=np.bool)
        total_metrics, context = None, None
        start_time = time.time()
        fps_start_time = time.time()
        for t in range(start_t, self.cfg.num_env_steps, self.cfg.online_parallel_envs):  # RESUME-SUPPORT: was range(0, ...)
            if self.cfg.evaluate and t % self.cfg.eval_every_steps == 0:
                eval_metrics = self.eval(t, replay_buffer=replay_buffer)
                if self.cfg.prioritization:
                    # priorities
                    index_in_buffer = {}
                    for i, ep in enumerate(replay_buffer["expert_slicer"].storage):
                        index_in_buffer[ep["motion_id"][0].item()] = i
                    motions_id, priorities, idxs = [], [], []
                    for _, metr in eval_metrics["tracking"].items():
                        motions_id.append(metr["motion_id"])
                        priorities.append(metr["emd"])
                        idxs.append(index_in_buffer[metr["motion_id"]])
                    priorities = (
                        torch.clamp(
                            torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                            min=self.cfg.prioritization_min_val,
                            max=self.cfg.prioritization_max_val,
                        )
                        * self.cfg.prioritization_scale
                    )
                    bins = torch.floor(priorities)
                    for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                        mask = bins == i
                        n = mask.sum().item()
                        if n > 0:
                            priorities[mask] = 1 / n

                    if mp_info is not None:
                        mp_info["motion_buffer"].update_priorities(motions_id=motions_id, priorities=priorities.cpu().numpy())
                    else:
                        train_env.unwrapped.motion_buffer.update_priorities(motions_id=motions_id, priorities=priorities.cpu().numpy())
                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            with torch.no_grad():
                obs = torch.tensor(td["obs"], dtype=torch.float32, device=self.agent.device)
                step_count = torch.tensor(td["time"], device=self.agent.device)
                context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count)
                if t < self.cfg.num_seed_steps:
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    action = self.agent.act(obs=obs, z=context, mean=False).cpu().detach().numpy()
            new_td, reward, terminated, truncated, new_info = train_env.step(action)
            real_next_obs = new_td["obs"].astype(np.float32).copy()
            new_done = np.logical_or(terminated.ravel(), truncated.ravel())

            if Version(gymnasium.__version__) >= Version("1.0"):
                # We add only transitions corresponding to environments that have not reset in the previous step.
                # For environments that have reset in the previous step, the new observation corresponds to the state after reset.
                indexes = ~done
                data = {
                    "observation": obs[indexes],
                    "action": action[indexes],
                    "z": context[indexes],
                    "step_count": step_count[indexes],
                    "qpos": info["qpos"][indexes],
                    "qvel": info["qvel"][indexes],
                    "next": {
                        "observation": real_next_obs[indexes],
                        "terminated": terminated[indexes].reshape(-1, 1),
                        "truncated": truncated[indexes].reshape(-1, 1),
                        "reward": reward[indexes].reshape(-1, 1),
                        "qpos": new_info["qpos"][indexes],
                        "qvel": new_info["qvel"][indexes],
                    },
                }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            replay_buffer["train"].extend(data)

            if len(replay_buffer["train"]) > 0 and t > self.cfg.num_seed_steps and t % self.cfg.update_agent_every == 0:
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, t)
                    if total_metrics is None:
                        num_metrics_updates = 1
                        total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
                    else:
                        num_metrics_updates += 1
                        total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if t % self.cfg.log_every_updates == 0 and total_metrics is not None:
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / num_metrics_updates
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()

            if t % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(str(self.work_dir / "checkpoint"))

            # RESUME-SUPPORT: full-state checkpoint at the checkpoint_every_steps
            # cadence (skipping t=0 and the step just restored from) and immediately
            # when SIGTERM/SIGUSR1 arrived; on a signal, exit 99 so the launcher
            # resubmits with --resume.
            if (t > start_t and t % self.cfg.checkpoint_every_steps == 0) or _STOP_SIGNAL["num"] is not None:
                reason = f"signal {_STOP_SIGNAL['num']}" if _STOP_SIGNAL["num"] is not None else "cadence"
                self.save_resume_checkpoint(t, replay_buffer, reason=reason)
                if _STOP_SIGNAL["num"] is not None:
                    print(f"[resume-ckpt] stopping at t={t} on {reason}; resubmit with --resume", flush=True)
                    if mp_info is not None:
                        mp_info["manager"].shutdown()
                    sys.exit(99)
            progb.update(self.cfg.online_parallel_envs)
            td = new_td
            done = new_done
            info = new_info
        self.agent.save(str(self.work_dir / "checkpoint"))
        if mp_info is not None:
            mp_info["manager"].shutdown()

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        inference_function: str = "reward_wr_inference"

        self.agent._model.to("cpu")
        self.agent._model.train(False)

        # ---------------------------------------------------------------
        # Reward evaluation
        # ---------------------------------------------------------------
        eval_agent = RewardWrapper(
            model=self.agent._model,
            inference_dataset=replay_buffer["train"],
            num_samples_per_inference=self.cfg.reward_eval_num_inference_samples,
            inference_function=inference_function,
            max_workers=1,
            process_executor=False,
        )
        reward_eval = RewardEvaluation(
            tasks=self.cfg.reward_eval_tasks,
            env_kwargs={"state_init": "Fall", "context": "spawn"},
            num_contexts=1,
            num_envs=self.cfg.reward_eval_num_envs,
            num_episodes=self.cfg.reward_eval_num_eval_episodes,
        )
        start_t = time.time()
        reward_metrics = {}
        if not replay_buffer["train"].empty():
            print(f"Reward started at {time.ctime(start_t)}", flush=True)
            reward_metrics = reward_eval.run(agent=eval_agent)
            duration = time.time() - start_t
            print(f"Reward eval time: {duration}")
            if self.cfg.use_wandb:
                m_dict = {}
                avg_return = []
                for task in reward_metrics.keys():
                    m_dict[f"{task}/return"] = np.mean(reward_metrics[task]["reward"])
                    m_dict[f"{task}/return#std"] = np.std(reward_metrics[task]["reward"])
                    avg_return.append(reward_metrics[task]["reward"])
                m_dict["reward/return"] = np.mean(avg_return)
                m_dict["reward/return#std"] = np.std(avg_return)
                m_dict["reward/time"] = duration
                wandb.log(
                    {f"eval/reward/{k}": v for k, v in m_dict.items()},
                    step=t,
                )
        # ---------------------------------------------------------------
        # Tracking evaluation
        # ---------------------------------------------------------------
        eval_agent = TrackingWrapper(model=self.agent._model)
        tracking_eval = TrackingEvaluation(
            motions=self.cfg.tracking_eval_motions,
            motion_base_path=self.cfg.tracking_eval_motions_root,
            env_kwargs={
                "state_init": "Default",
            },
            num_envs=self.cfg.tracking_eval_num_envs,
        )
        start_t = time.time()
        print(f"Tracking started at {time.ctime(start_t)}", flush=True)
        tracking_metrics = tracking_eval.run(agent=eval_agent)
        duration = time.time() - start_t
        print(f"Tracking eval time: {duration}")
        if self.cfg.use_wandb:
            aggregate, m_dict = collections.defaultdict(list), {}
            for _, metr in tracking_metrics.items():
                for k, v in metr.items():
                    if isinstance(v, numbers.Number):
                        aggregate[k].append(v)
            for k, v in aggregate.items():
                m_dict[k] = np.mean(v)
                m_dict[f"{k}#std"] = np.std(v)
            m_dict["time"] = duration

            wandb.log(
                {f"eval/tracking/{k}": v for k, v in m_dict.items()},
                step=t,
            )
        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        self.agent._model.to("cuda")
        self.agent._model.train()

        return {"reward": reward_metrics, "tracking": tracking_metrics}


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    # RESUME-SUPPORT: catch slurm's pre-kill signals (SIGTERM at walltime/scancel,
    # SIGUSR1 forwarded by the launcher 10 minutes before the limit) so the loop
    # can write a resume checkpoint before dying.
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)

    env, _ = make_humenv(
        num_envs=1,
        vectorization_mode="sync",
        wrappers=[gymnasium.wrappers.FlattenObservation],
        render_width=320,
        render_height=320,
    )

    agent_config = FBcprAgentConfig()
    agent_config.model.obs_dim = env.observation_space.shape[0]
    agent_config.model.action_dim = env.action_space.shape[0]
    agent_config.model.norm_obs = True
    agent_config.train.batch_size = 1024
    agent_config.train.use_mix_rollout = 1
    agent_config.train.update_z_every_step = 150
    agent_config.model.actor_std = 0.2
    agent_config.model.seq_length = 8
    # archi
    # the config of the model trained in the paper
    model, hidden_dim, hidden_layers = "simple", 1024, 2
    # uncomment the line below for the config of model deployed in the demo
    # WARNING: you need to use compile=True on a A100 GPU or better, as otherwise training can be very slow
    # model, hidden_dim, hidden_layers = "residual", 2048, 12
    agent_config.model.archi.z_dim = 256
    agent_config.model.archi.b.norm = 1
    agent_config.model.archi.norm_z = 1
    agent_config.model.archi.f.hidden_dim = hidden_dim
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.actor.hidden_dim = hidden_dim
    agent_config.model.archi.critic.hidden_dim = hidden_dim
    agent_config.model.archi.f.hidden_layers = hidden_layers
    agent_config.model.archi.b.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = hidden_layers
    agent_config.model.archi.critic.hidden_layers = hidden_layers
    agent_config.model.archi.f.model = model
    agent_config.model.archi.actor.model = model
    agent_config.model.archi.critic.model = model
    # optim
    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-5
    agent_config.train.lr_actor = 1e-4
    agent_config.train.lr_critic = 1e-4
    agent_config.train.ortho_coef = 100
    agent_config.train.train_goal_ratio = 0.2
    agent_config.train.expert_asm_ratio = 0.6
    agent_config.train.relabel_ratio = 0.8
    agent_config.train.reg_coeff = 0.01
    agent_config.train.q_loss_coef = 0.1  # or 0
    # discriminator cfg
    agent_config.train.grad_penalty_discriminator = 10
    agent_config.train.weight_decay_discriminator = 0
    agent_config.train.lr_discriminator = 1e-5
    agent_config.model.archi.discriminator.hidden_layers = 3
    agent_config.model.archi.discriminator.hidden_dim = 1024
    agent_config.model.device = config.device
    # misc
    agent_config.train.discount = 0.98
    agent_config.compile = config.compile
    agent_config.cudagraphs = config.cudagraphs
    env.close()

    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()
