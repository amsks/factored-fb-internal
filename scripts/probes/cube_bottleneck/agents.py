"""Build + restore a trained arm exactly the way the eval path does (spec section 4, "Loader").

One helper, `load_arm`, following `scripts/probe_eval_interface.py:120-145`: read the run's
own `config.json`, rebuild the adapter/env/dataset it trained on, construct `Agent.create`
from a real batch's shapes, then `restore_agent` at the requested epoch. Nothing here
reimplements a critic; the probes drive the shipped interface.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
