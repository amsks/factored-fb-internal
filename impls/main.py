"""Training entrypoint for the factored impls/ tree (PR1: FB critic × FlowBC actor).

Config is composed from the critic/actor/domain sibling groups under impls/configs/ via
load_config() (spec §3); the old flat get_config() has been replaced.
"""
import os
import warnings

# Device/EGL pin MUST precede any mujoco/ogbench import (carried from run_gciql.py:97-104).
# CUDA_VISIBLE_DEVICES masks JAX but not MuJoCo's EGL enumeration; pin EGL to the same
# physical GPU so concurrent eval renders don't pile onto GPU 0 (GL 0x8cdd).
os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

from absl import app, flags
import jax.numpy as jnp

from impls.agents import Agent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
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
flags.DEFINE_integer("eval_video_episodes", 0,
                     "Episodes to render into an eval mp4 (0 = no video). Works for every "
                     "adapter/task; both adapters already return frames, this decides how many.")
flags.DEFINE_integer("video_interval", 0,
                     "Render video every N steps (0 = every eval_interval). Videos are far "
                     "more expensive than the scalar eval, so this can be coarser.")
flags.DEFINE_string("save_dir", "exp/", "Checkpoint directory.")
flags.DEFINE_string("restore_path", "", "Glob matching exactly one run dir to restore from.")
flags.DEFINE_integer("restore_epoch", 0, "Checkpoint step to restore (params_{epoch}.pkl).")
flags.DEFINE_string("agg_dir", None, "optional: dir for per-seed eval jsonls (campaign/wandb_agg_daemon.py)")
flags.DEFINE_string("agg_group", None, "optional aggregation key; default <critic>_<actor>__<domain>")
flags.DEFINE_integer("infer_cond_episodes", 0,
                     "If >0, infer eval conditioning from this many relabelled transitions "
                     "before each eval (FB zero-shot path). 0 = use eval_cond0.")


def infer_cond_noop_reason(pre_cond, post_cond, rewards):
    """Detect when infer_eval_cond() had no real effect, and return a string naming why
    (for a caller to warn on), or None if inference looks like it did something.

    Two known no-op cases on the currently-supported configs:
      - cube-single-play-v0 is unlabeled play data, so FBDataset zeroes every reward
        (impls/utils/datasets.py); FB's z = (rewards^T B)/N is then exactly 0, and
        project_z(0) returns exact zeros -- bit-identical to eval_cond0's zero z.
      - GCIQLCritic.infer_cond (impls/critics/gciql.py) ignores obs/rewards entirely and
        always returns a zero placeholder cond["goal"]; evaluate() overrides cond["goal"]
        with the env task goal regardless (impls/utils/evaluation.py), so this cond is
        never even read.
    A user passing --infer_cond_episodes>0 on either path gets ordinary eval_cond0/env-goal
    behavior with no indication z-inference actually ran, unless we say so here.
    """
    rewards = jnp.asarray(rewards)
    if rewards.size and bool(jnp.all(rewards == 0)):
        return "sampled rewards are all zero (no reward signal to infer a conditioning vector from)"
    common = set(pre_cond) & set(post_cond)
    if common and all(
        jnp.array_equal(jnp.asarray(pre_cond[k]), jnp.asarray(post_cond[k])) for k in common
    ):
        return "inferred cond is bit-identical to the pre-inference cond (inference had no effect)"
    return None


_FAMILY = {"b2d": "Box2D", "p2d": "Box2D", "mw": "Metaworld", "ms": "ManiSkill"}


def _append_agg(agg_dir, group, seed, step, row, task):
    """Optional: append this run's eval point as a jsonl the campaign agg daemon reads.
    Pure reporting -- writes nothing unless --agg_dir is set; touches no algorithm state."""
    import json
    os.makedirs(agg_dir, exist_ok=True)
    rec = dict(seed=int(seed), step=int(step),
               success_rate=float(row.get("success", row.get("success_rate", 0.0))),
               at_goal_ticks=float(row.get("at_goal_ticks", float("nan"))),
               family=_FAMILY.get(str(task).split("-")[0]))
    with open(os.path.join(agg_dir, f"{group}__s{seed}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def main(_):
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, FLAGS.override)

    exp_name = get_exp_name(FLAGS.seed, FLAGS.critic, FLAGS.actor, cfg["env_name"])
    # Each run gets its own subdirectory so two seeds or two critic x actor pairs sharing a
    # --save_dir don't overwrite each other's train.csv/eval.csv/params_N.pkl.
    # --restore_path stays a glob over run directories (impls/utils/flax_utils.py:139-148);
    # it should now point at save_dir/exp_name for a prior run.
    run_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    agg_group = FLAGS.agg_group or f"{FLAGS.critic}_{FLAGS.actor}__{FLAGS.domain}"
    run = setup_wandb(cfg, group=f"{FLAGS.critic}-{FLAGS.actor}", name=exp_name)
    csv_train = CsvLogger(os.path.join(run_dir, "train.csv"))
    csv_eval = CsvLogger(os.path.join(run_dir, "eval.csv"))

    try:
        adapter = get_adapter(cfg)
        env, train_ds, val_ds = adapter.make_env_and_datasets(cfg)
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
                    pre_cond = agent.cond_eval
                    eval_agent = agent.infer_eval_cond(
                        jnp.asarray(rb["observations"]), jnp.asarray(rb["rewards"]))
                    reason = infer_cond_noop_reason(pre_cond, eval_agent.cond_eval, rb["rewards"])
                    if reason is not None:
                        warnings.warn(
                            f"--infer_cond_episodes={FLAGS.infer_cond_episodes} requested but "
                            f"z-inference had no effect this eval: {reason}. Evaluation is "
                            f"using eval_cond0 (or the env task goal), not an inferred "
                            f"conditioning vector -- do not interpret these eval numbers as a "
                            f"zero-shot z-inference result."
                        )
                # Video is opt-in and can run on a coarser cadence than the scalar eval.
                vid_every = FLAGS.video_interval or FLAGS.eval_interval
                n_video = FLAGS.eval_video_episodes if (
                    FLAGS.eval_video_episodes > 0 and (i == 1 or i % vid_every == 0)) else 0
                for task_id in adapter.eval_tasks(env, cfg):
                    row, frames = adapter.evaluate(eval_agent, env, task_id, cfg,
                                                   FLAGS.eval_episodes, n_video)
                    scope = "eval" if task_id is None else f"eval/task{task_id}"
                    if n_video and frames:
                        from env.video import log_video, save_video_grid
                        tag = "" if task_id is None else f"_task{task_id}"
                        p = save_video_grid(frames, os.path.join(run_dir, f"eval{tag}_step{i}.mp4"),
                                            max_episodes=FLAGS.eval_video_episodes)
                        log_video(run, p, key=f"{scope}/video", step=i)
                    csv_eval.log({**row, "task_id": -1 if task_id is None else task_id}, step=i)
                    if run is not None:
                        run.log({f"{scope}/{k}": v for k, v in row.items()}, step=i)
                    if FLAGS.agg_dir:
                        _append_agg(FLAGS.agg_dir, agg_group, FLAGS.seed, i, row,
                                    cfg.get("task", cfg.get("env_name")))
                    print("eval", i, "task", task_id, row, flush=True)
            if i % FLAGS.save_interval == 0:
                save_agent(agent, run_dir, i)
    finally:
        csv_train.close()
        csv_eval.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    app.run(main)
