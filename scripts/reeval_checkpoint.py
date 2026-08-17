"""Eval-only re-scoring of an EXISTING checkpoint, no retraining.

Built for the ms-pick-cube synth-goal-frame fix (env/unified.py::synth_goal_frame): the E2
masked runs were trained and originally evaluated with a synth goal frame that forced
tcp == obj == target, off-distribution at the success-radius scale. This script re-runs
`evaluate()` against the SAME checkpoint with the fixed frame, so the number moves only
because the eval-time goal construction changed -- nothing about the agent or data does.

Reuses impls.main's own code path (adapter, Agent.create, restore_agent, adapter.evaluate)
rather than reimplementing it, so this script and a real training run can never disagree
about how a checkpoint is built or scored.

Usage:
  python -m scripts.reeval_checkpoint \
      --critic gciql --actor bcflow --domain ms-pick-cube --seed 0 \
      --restore_path "$SCRATCH/factored-fb-E2/ms-pick-cube/BC/ms-pick-cube_gciql_bcflow_seed_0" \
      --restore_epoch 300000 --eval_episodes 50 \
      --override obs_mask_dims=[26,27,28,39,40,41] --override goal_mask_dims=[26,27,28,39,40,41] \
      --override value_p_randomgoal=0.0 --override value_p_trajgoal=0.8 \
      --override eval_goal_mode=synth
"""
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

from absl import app, flags
import jax.numpy as jnp
import numpy as np

from impls.agents import Agent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import restore_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "fb", "Critic name.")
flags.DEFINE_string("actor", "flowbc", "Actor name.")
flags.DEFINE_string("domain", "cube_single", "Domain name.")
flags.DEFINE_multi_string("override", [], "Config override, dotlist form.")
flags.DEFINE_integer("seed", 0, "Random seed (must match the checkpoint's training seed).")
flags.DEFINE_string("restore_path", "", "Glob matching exactly one run dir to restore from.")
flags.DEFINE_integer("restore_epoch", 300000, "Checkpoint step to restore (params_{epoch}.pkl).")
flags.DEFINE_integer("eval_episodes", 50, "Episodes to evaluate.")
flags.DEFINE_string("tag", "", "Free-text label written into the output row (e.g. run name).")
flags.DEFINE_string("out_jsonl", "", "Optional: append the result row to this jsonl file.")


def main(_):
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, FLAGS.override)
    np.random.seed(FLAGS.seed)

    adapter = get_adapter(cfg)
    env, train_ds, _val_ds = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    for task_id in adapter.eval_tasks(env, cfg):
        row, _frames = adapter.evaluate(agent, env, task_id, cfg, FLAGS.eval_episodes, 0)
        row.pop("_per_episode", None)   # scalar-only report; per-episode payload not needed here
        print(f"REEVAL tag={FLAGS.tag} task={task_id} epoch={FLAGS.restore_epoch} "
              f"episodes={FLAGS.eval_episodes} {row}", flush=True)
        if FLAGS.out_jsonl:
            rec = dict(tag=FLAGS.tag, critic=FLAGS.critic, actor=FLAGS.actor,
                       domain=FLAGS.domain, seed=FLAGS.seed,
                       restore_path=FLAGS.restore_path, restore_epoch=FLAGS.restore_epoch,
                       eval_episodes=FLAGS.eval_episodes, task_id=task_id, **row)
            with open(FLAGS.out_jsonl, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
    # Return NOTHING. absl does `sys.exit(main(argv))`, so a returned list becomes the exit
    # status: Python prints it to stderr and exits 1. Returning the row list made every re-eval
    # look like a failure to the calling slurm script (job 903390 reported "12 failure(s)"
    # while all 12 checkpoints had in fact evaluated and written their records).


if __name__ == "__main__":
    app.run(main)
