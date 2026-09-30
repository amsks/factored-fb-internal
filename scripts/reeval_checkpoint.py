"""Evaluate an existing checkpoint without retraining.

Uses impls.main's own path (adapter, Agent.create, restore_agent, adapter.evaluate), so an
evaluation here and one inside a training run build and score the agent the same way.

The critic is trained without task rewards; rewards are only read when the task vector is
inferred. One checkpoint per (domain, seed) therefore serves every reward task of that
domain, and scoring another task is a re-evaluation: `--override exorl_task=<task>` relabels
the dataset rewards and builds the env for that task together. With --infer_cond_episodes N
and --eval_on_inferred_cond, z is inferred from N relabelled transitions and the policy is
rolled on z with no goal given.

    python -m scripts.reeval_checkpoint --critic density_fb_z --actor ddpgbc \
        --domain dmc-walker --seed 0 --restore_path <run dir> --restore_epoch 1000000 \
        --eval_episodes 10 --infer_cond_episodes 50000 --eval_on_inferred_cond \
        --override exorl_task=walk --out_jsonl walker.jsonl
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
from impls.utils.flax_utils import resolve_run_dir, restore_agent
from impls.critics.base import infer_cond_rows
# NOT from impls.main: that module defines absl flags at import time, and this script defines
# its own --critic/--actor/--domain, so importing it raises DuplicateFlagError before main()
# ever runs. The class lives in a flag-free module for exactly this reason.
from impls.utils.eval_cond import InferredCondAgent

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
flags.DEFINE_bool("keep_per_episode", False,
                  "Keep the per-episode arrays in the jsonl row instead of dropping them. "
                  "The mean hides the SHAPE of the return distribution, and on some tasks "
                  "the shape is the finding: walker/spin is bimodal (nine episodes near 975, "
                  "one near 40), so one catastrophic episode moves the mean ~100 and the std "
                  "reads as 'high variance' when the truth is 'rare total failure'. Off by "
                  "default because it makes rows much larger.")
flags.DEFINE_integer("infer_cond_episodes", 0,
                     "If >0, infer eval conditioning from this many relabelled transitions "
                     "before evaluating (the zero-shot reward-sample route). Same meaning as "
                     "impls/main.py's flag of this name, including that the rows fed to "
                     "infer_cond come from the critic's own INFER_COND_MARGINAL.")
flags.DEFINE_bool("config_from_run", True,
                  "Rebuild the agent from the config.json the run recorded next to its "
                  "checkpoints rather than from the current yaml, so the evaluation builds "
                  "the same networks the checkpoint was written from. --override still "
                  "wins over it.")
flags.DEFINE_bool("eval_on_inferred_cond", False,
                  "Evaluate on the inferred conditioning INSTEAD of the env's task goal. "
                  "Requires --infer_cond_episodes>0. Without it the evaluator hands a goal "
                  "frame to any goal-capable critic and the inferred index is discarded "
                  "before it is used -- see impls/main.py.")


def _recorded_config_overrides(run_dir):
    """Dotlist overrides reproducing the config a run recorded, minus the keys that are
    properties of THIS eval rather than of the trained networks."""
    path = os.path.join(resolve_run_dir(run_dir), "config.json")
    if not os.path.isfile(path):
        print(f"[config_from_run] no config.json at {path}; using yaml defaults")
        return []
    with open(path) as f:
        recorded = json.load(f).get("config", {})
    # Keys the eval sets for itself. Everything else describes the trained agent.
    skip = {"seed", "save_dir", "restore_path", "restore_epoch", "train_steps",
            "eval_episodes", "eval_interval", "save_interval", "log_interval",
            "exorl_task", "wandb", "run_group", "online"}
    out = []

    def walk(prefix, node):
        for k, v in node.items():
            if not prefix and k in skip:
                continue
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(f"{key}.", v)
            elif isinstance(v, str):
                out.append(f'{key}="{v}"')
            else:
                out.append(f"{key}={json.dumps(v)}")

    walk("", recorded)
    return out


def main(_):
    # Recorded config first, CLI --override last, so an explicit override still wins.
    base = (_recorded_config_overrides(FLAGS.restore_path)
            if FLAGS.config_from_run and FLAGS.restore_path else [])
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, base + list(FLAGS.override))
    np.random.seed(FLAGS.seed)

    adapter = get_adapter(cfg)
    env, train_ds, _val_ds = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    # params_only: an eval never steps the optimizer, so a checkpoint written under a
    # different optax chain must still score. See restore_agent's docstring.
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch, params_only=True)

    if FLAGS.eval_on_inferred_cond and FLAGS.infer_cond_episodes <= 0:
        raise ValueError("--eval_on_inferred_cond needs --infer_cond_episodes>0: it "
                         "evaluates on an index nothing has inferred otherwise.")
    # Mirrors impls/main.py's eval block exactly, including WHICH rows feed infer_cond (the
    # critic's own INFER_COND_MARGINAL, not a hard-coded column) and the wrapper that stops
    # a goal frame overwriting the index. Kept as the same two calls in the same order so a
    # re-eval and a training-time eval cannot disagree about what was conditioned on.
    eval_agent = agent
    if FLAGS.infer_cond_episodes > 0:
        rb = train_ds.sample(FLAGS.infer_cond_episodes)
        rows = infer_cond_rows(agent.critic)
        eval_agent = agent.infer_eval_cond(jnp.asarray(rb[rows]),
                                           jnp.asarray(rb["rewards"]))
        if FLAGS.eval_on_inferred_cond:
            eval_agent = InferredCondAgent(eval_agent)
        print(f"[infer-cond] conditioning inferred from {FLAGS.infer_cond_episodes} "
              f"rb[{rows!r}] rows; eval_on_inferred_cond={FLAGS.eval_on_inferred_cond}",
              flush=True)

    for task_id in adapter.eval_tasks(env, cfg):
        row, _frames = adapter.evaluate(eval_agent, env, task_id, cfg, FLAGS.eval_episodes, 0)
        if hasattr(eval_agent, "witness"):
            row.update(eval_agent.witness())
            eval_agent.reset_counts()
        per_episode = row.pop("_per_episode", None)   # kept out of the printed line either way
        print(f"REEVAL tag={FLAGS.tag} task={task_id} epoch={FLAGS.restore_epoch} "
              f"episodes={FLAGS.eval_episodes} {row}", flush=True)
        if FLAGS.out_jsonl:
            rec = dict(tag=FLAGS.tag, critic=FLAGS.critic, actor=FLAGS.actor,
                       domain=FLAGS.domain, seed=FLAGS.seed,
                       restore_path=FLAGS.restore_path, restore_epoch=FLAGS.restore_epoch,
                       eval_episodes=FLAGS.eval_episodes, task_id=task_id,
                       exorl_task=cfg.get("exorl_task"), **row)
            if FLAGS.keep_per_episode and per_episode:
                rec["_per_episode"] = per_episode
            with open(FLAGS.out_jsonl, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
    # Return NOTHING. absl does `sys.exit(main(argv))`, so a returned list becomes the exit
    # status: Python prints it to stderr and exits 1. Returning the row list made every re-eval
    # look like a failure to the calling slurm script (job 903390 reported "12 failure(s)"
    # while all 12 checkpoints had in fact evaluated and written their records).


if __name__ == "__main__":
    app.run(main)
