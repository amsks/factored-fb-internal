"""Training entrypoint for the factored impls/ tree (PR1: FB critic × FlowBC actor).

PR1 uses a flat merged config via get_config() + absl flags; the Hydra critic=/actor=
sibling-group system (spec §3) lands with the multi-critic PRs.
"""
import os

# Device/EGL pin MUST precede any mujoco/ogbench import (carried from run_gciql.py:97-104).
# CUDA_VISIBLE_DEVICES masks JAX but not MuJoCo's EGL enumeration; pin EGL to the same
# physical GPU so concurrent eval renders don't pile onto GPU 0 (GL 0x8cdd).
os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

from absl import app, flags
import jax.numpy as jnp

from impls.agents import Agent
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate
from impls.utils.flax_utils import save_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "fb", "Critic name.")
flags.DEFINE_string("actor", "flowbc", "Actor name.")
flags.DEFINE_string("env_name", "cube-single-play-v0", "OGBench dataset/env name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("train_steps", 1_000_000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Log every N steps.")
flags.DEFINE_integer("eval_interval", 100_000, "Eval every N steps.")
flags.DEFINE_integer("save_interval", 1_000_000, "Checkpoint every N steps.")
flags.DEFINE_integer("eval_episodes", 20, "Episodes per eval task.")
flags.DEFINE_string("save_dir", "exp/", "Checkpoint directory.")


def get_config():
    """Flat merged config mirroring PSMFlows fb.py:get_config() at cube-single launch values."""
    return dict(
        batch_size=256, z_dim=50, L_dim=50, num_parallel=2,
        discount=0.99, f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0,
        train_goal_ratio=0.5, fb_pessimism_penalty=0.0, norm_z=True,
        lr_f=1.0e-4, lr_b=1.0e-4, lr_actor=1.0e-4,
        forward=dict(hidden_dim=512, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=512, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=512, hidden_layers=4, norm=True),
        actor=dict(type="flow", hidden_dim=512, hidden_layers=2, embedding_layers=2,
                   bc_coeff=3.0, flow_steps=10, lr_actor_vf=3.0e-4,
                   flow_vf_hidden_dim=512, flow_vf_hidden_layers=4),
    )


def main(_):
    cfg = get_config()
    env, train_ds, val_ds = make_env_and_datasets(FLAGS.env_name)
    ex = train_ds.sample(1)
    agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)

    for i in range(1, FLAGS.train_steps + 1):
        batch = train_ds.sample(cfg["batch_size"])
        agent, info = agent.update(batch)
        if i % FLAGS.log_interval == 0:
            print(i, {k: float(jnp.asarray(v).mean()) for k, v in info.items()}, flush=True)
        if i == 1 or i % FLAGS.eval_interval == 0:
            n_tasks = len(env.unwrapped.task_infos)
            for task_id in range(1, n_tasks + 1):
                stats, _, _ = evaluate(agent, env, task_id=task_id, config=cfg,
                                       num_eval_episodes=FLAGS.eval_episodes)
                print("eval", i, "task", task_id, dict(stats), flush=True)
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)


if __name__ == "__main__":
    app.run(main)
