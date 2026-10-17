"""Rebuild and restore trained agents from run directories, as the eval path does."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class Arm:
    run_dir: Path
    critic: str
    actor: str
    seed: int
    epoch: int
    config: Dict[str, Any]
    agent: Any = None
    env: Any = None
    dataset: Any = field(default=None, repr=False)

    @property
    def env_name(self) -> str:
        return self.config["env_name"]

    @property
    def label(self) -> str:
        return f"{self.critic}_seed{self.seed}"


def read_meta(run_dir: Path) -> Dict[str, Any]:
    with open(Path(run_dir) / "config.json") as f:
        return json.load(f)


def load_arm(run_dir, epoch: int = 500000, *, with_env: bool = True) -> Arm:
    """Rebuild and restore one checkpoint. `with_env=False` skips the env (dataset only)."""
    import jax.numpy as jnp

    from env import get_adapter
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset
    from impls.utils.flax_utils import restore_agent

    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    cfg = meta["config"]
    np.random.seed(meta["seed"])

    adapter = get_adapter(cfg)
    env, train_ds, _val = adapter.make_env_and_datasets(cfg)
    ex = make_dataset(train_ds, cfg).sample(1)
    agent = Agent.create(meta["critic"], meta["actor"], meta["seed"],
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, str(run_dir), epoch)
    return Arm(run_dir=run_dir, critic=meta["critic"], actor=meta["actor"], seed=meta["seed"],
               epoch=epoch, config=cfg, agent=agent, env=env if with_env else None,
               dataset=train_ds)


def eval_tasks(arm: Arm) -> List[int]:
    from env import get_adapter

    tasks = get_adapter(arm.config).eval_tasks(arm.env, arm.config)
    assert tasks and tasks[0] is not None, (
        "the probe needs the per-task_id OGBench eval (eval_aggregate off), which is what "
        "table_privcoord's cells were produced by")
    return list(tasks)


def discover(exp_root, env_name: str, critics, seeds, epoch: int = 500000) -> List[Dict]:
    """Every (critic, seed) run present for `env_name`, with its final-checkpoint path."""
    out = []
    for critic in critics:
        for seed in seeds:
            d = Path(exp_root) / f"{env_name}_{critic}_flowbc_seed_{seed}"
            ckpt = d / f"params_{epoch}.pkl"
            if ckpt.exists():
                out.append(dict(run_dir=d, critic=critic, seed=seed, epoch=epoch))
    return out


# Reward-inference loading: task z inferred from relabelled dataset rewards, as in impls/main.py

DEFAULT_ROOTS = ("exp", "exp/tdjtab")


def discover_roots(env_name: str, critics, seeds, epoch: int,
                   roots=DEFAULT_ROOTS) -> List[Dict]:
    """Like `discover`, over several exp roots; the first root holding the checkpoint wins."""
    out = []
    for critic in critics:
        for seed in seeds:
            for root in roots:
                d = Path(root) / f"{env_name}_{critic}_flowbc_seed_{seed}"
                if (d / f"params_{epoch}.pkl").exists():
                    out.append(dict(run_dir=d, critic=critic, seed=seed, epoch=epoch))
                    break
    return out


def load_arm_eval(run_dir, epoch: int) -> Arm:
    """`load_arm` restoring parameters only (no optimizer state)."""
    import jax.numpy as jnp

    from env import get_adapter
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset
    from impls.utils.flax_utils import restore_agent

    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    cfg = meta["config"]
    np.random.seed(meta["seed"])
    adapter = get_adapter(cfg)
    env, train_ds, _val = adapter.make_env_and_datasets(cfg)
    ex = make_dataset(train_ds, cfg).sample(1)
    agent = Agent.create(meta["critic"], meta["actor"], meta["seed"],
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, str(run_dir), epoch, params_only=True)
    return Arm(run_dir=run_dir, critic=meta["critic"], actor=meta["actor"], seed=meta["seed"],
               epoch=epoch, config=cfg, agent=agent, env=env, dataset=train_ds)


def relabel_cond_agents(arm: Arm, tasks, *, seed: int = 20260912) -> Dict[int, Any]:
    """Return {task_id: InferredCondAgent}, z inferred from 10k relabelled rows per task.

    The global NumPy RNG is reseeded per task from `seed`, so all agents use the same rows.
    """
    import jax.numpy as jnp

    from env import get_adapter
    from impls.utils.eval_cond import InferredCondAgent

    class BatchedInferredCondAgent(InferredCondAgent):
        """InferredCondAgent for batched rollouts: tiles the [d] z to [B, d] and passes it via `goals`."""

        def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
            self.action_calls += 1
            if goals is not None:
                self.goals_dropped += 1
            z = jnp.asarray(self.cond_eval["z"])
            obs = jnp.asarray(observations)
            if obs.ndim == 2 and z.ndim == 1:
                z = jnp.broadcast_to(z[None], (obs.shape[0], z.shape[-1]))
            return self.agent.sample_actions(observations, goals={"z": z}, seed=seed,
                                             temperature=temperature)

    cfg = arm.config
    adapter = get_adapter(cfg)
    n = int(cfg.get("eval_relabel_size") or 10000)
    shift = float(cfg.get("eval_relabel_shift", 1.0))
    out = {}
    for tid in tasks:
        np.random.seed(seed + int(tid))
        obs, rew = adapter.relabel_infer_batch(arm.dataset, arm.env, int(tid), n,
                                               shift_reward=shift)
        pre = arm.agent.cond_eval
        ag = BatchedInferredCondAgent(
            arm.agent.infer_eval_cond(jnp.asarray(obs), jnp.asarray(rew)))
        w = ag.witness()
        key = "z"
        assert key in ag.cond_eval, f"{arm.critic}: reward inference returned no 'z'"
        assert not np.allclose(np.asarray(ag.cond_eval[key]), np.asarray(pre.get(key, 0))), (
            f"{arm.label} task {tid}: reward inference was a no-op")
        assert w["inferred_cond_norm"] > 0, f"{arm.label} task {tid}: zero inferred index"
        ag.relabel_stats = dict(task=int(tid), n=n, shift=shift,
                                reward_mean=float(np.mean(rew)),
                                reward_pos_frac=float(np.mean(rew > 0.5)), **w)
        out[int(tid)] = ag
    return out
