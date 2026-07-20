"""Training entrypoint for the factored impls/ tree (PR1: FB critic × FlowBC actor).

Config is composed from the critic/actor/domain sibling groups under impls/configs/ via
load_config() (spec §3); the old flat get_config() has been replaced.
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
from impls.utils.config_utils import load_config
from impls.utils.datasets import make_dataset
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate
from impls.utils.flax_utils import restore_agent, save_agent
from impls.utils.log_utils import CsvLogger, get_exp_name, setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "fb", "Critic name (impls/configs/critic/<name>.yaml).")
flags.DEFINE_string("actor", "flowbc", "Actor name (impls/configs/actor/<name>.yaml).")
flags.DEFINE_string("domain", "cube_single", "Domain name (impls/configs/domain/<name>.yaml).")
flags.DEFINE_multi_string("override", [], "Config override, dotlist form (e.g. batch_size=64).")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("train_steps", 1_000_000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Log every N steps.")
flags.DEFINE_integer("eval_interval", 100_000, "Eval every N steps.")
flags.DEFINE_integer("save_interval", 1_000_000, "Checkpoint every N steps.")
flags.DEFINE_integer("eval_episodes", 20, "Episodes per eval task.")
flags.DEFINE_string("save_dir", "exp/", "Checkpoint directory.")
flags.DEFINE_string("restore_path", "", "Glob matching exactly one run dir to restore from.")
flags.DEFINE_integer("restore_epoch", 0, "Checkpoint step to restore (params_{epoch}.pkl).")
flags.DEFINE_integer("infer_cond_episodes", 0,
                     "If >0, infer eval conditioning from this many relabelled transitions "
                     "before each eval (FB zero-shot path). 0 = use eval_cond0.")


def main(_):
    cfg = load_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, FLAGS.override)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    exp_name = get_exp_name(FLAGS.seed, FLAGS.critic, FLAGS.actor, cfg["env_name"])
    run = setup_wandb(cfg, group=f"{FLAGS.critic}-{FLAGS.actor}", name=exp_name)
    csv_train = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    csv_eval = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))

    env, train_ds, val_ds = make_env_and_datasets(cfg["env_name"])
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)

    start_step = 1
    if FLAGS.restore_path:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        start_step = FLAGS.restore_epoch + 1

    for i in range(start_step, FLAGS.train_steps + 1):
        batch = train_ds.sample(cfg["batch_size"])
        agent, info = agent.update(batch)
        if i % FLAGS.log_interval == 0:
            row = {k: float(jnp.asarray(v).mean()) for k, v in info.items()}
            csv_train.log(row, step=i)
            if run is not None:
                run.log({f"train/{k}": v for k, v in row.items()}, step=i)
            print(i, row, flush=True)
        if i == 1 or i % FLAGS.eval_interval == 0:
            eval_agent = agent
            if FLAGS.infer_cond_episodes > 0:
                rb = train_ds.sample(FLAGS.infer_cond_episodes)
                eval_agent = agent.infer_eval_cond(
                    jnp.asarray(rb["observations"]), jnp.asarray(rb["rewards"]))
            n_tasks = len(env.unwrapped.task_infos)
            for task_id in range(1, n_tasks + 1):
                stats, _, _ = evaluate(eval_agent, env, task_id=task_id, config=cfg,
                                       num_eval_episodes=FLAGS.eval_episodes)
                row = {k: float(v) for k, v in dict(stats).items()}
                csv_eval.log({**row, "task_id": task_id}, step=i)
                if run is not None:
                    run.log({f"eval/task{task_id}/{k}": v for k, v in row.items()}, step=i)
                print("eval", i, "task", task_id, row, flush=True)
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    csv_train.close()
    csv_eval.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    app.run(main)
