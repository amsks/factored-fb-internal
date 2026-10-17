"""Training entry point: offline training and evaluation.

eval.csv uses the env's task (goal, or z inferred from rewards with --eval_on_inferred_cond).
eval_relabel.csv is reward-inference evaluation: z is inferred from relabeled rewards.
"""

import json
import os
import signal
import warnings

# Pin MuJoCo's EGL device to the visible GPU before any mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import jax.numpy as jnp
import numpy as np
from absl import app, flags

from env import get_adapter, make_config
from impls.agents import Agent
from impls.critics.base import infer_cond_rows
from impls.utils.datasets import make_dataset

# Re-exported
from impls.utils.eval_cond import InferredCondAgent
from impls.utils.flax_utils import restore_agent, save_agent
from impls.utils.log_utils import CsvLogger, get_exp_name, setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "fb", "Critic name (impls/configs/critic/<name>.yaml).")
flags.DEFINE_string("actor", "flowbc", "Actor name (impls/configs/actor/<name>.yaml).")
flags.DEFINE_string(
    "domain", "cube_single", "Domain name (env/domains or impls/configs/domain)."
)
flags.DEFINE_multi_string(
    "override", [], "Config override, dotlist form (e.g. batch_size=64)."
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("train_steps", 1_000_000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Log every N steps.")
flags.DEFINE_integer("eval_interval", 100_000, "Eval every N steps.")
flags.DEFINE_integer("save_interval", 1_000_000, "Checkpoint every N steps.")
flags.DEFINE_integer("eval_episodes", 20, "Episodes per eval task.")
flags.DEFINE_string("save_dir", "exp/", "Checkpoint directory.")
flags.DEFINE_bool(
    "auto_resume",
    False,
    "Resume from the latest params_<N>.pkl in this run's own directory "
    "(save_dir/exp_name), continuing at step N+1. With no checkpoint present "
    "this is a fresh start, so a launcher can pass it unconditionally. "
    "eval.csv/train.csv are appended, so the step axis continues.",
)
flags.DEFINE_integer(
    "infer_cond_episodes",
    0,
    "If >0, infer the eval task vector from this many dataset transitions "
    "and their rewards before each eval. The rows fed to infer_cond are the "
    "critic's declared INFER_COND_MARGINAL (impls/critics/base.py). On its "
    "own this only computes the vector; --eval_on_inferred_cond rolls it.",
)
flags.DEFINE_bool(
    "eval_on_inferred_cond",
    False,
    "Evaluate on the task vector --infer_cond_episodes inferred instead of the "
    "env's task goal. Without it, the evaluator hands goal-capable critics a "
    "goal frame, which overrides the inferred vector.",
)


def infer_cond_noop_reason(pre_cond, post_cond, rewards, agent=None):
    """Reason infer_eval_cond() had no effect, or None.

    Detects all-zero rewards, an unchanged conditioning, and an eval agent that will be
    given the env goal (which overrides the inferred z).
    """
    rewards = jnp.asarray(rewards)
    if rewards.size and bool(jnp.all(rewards == 0)):
        return "sampled rewards are all zero (no reward signal to infer a conditioning vector from)"
    common = set(pre_cond) & set(post_cond)
    if common and all(
        jnp.array_equal(jnp.asarray(pre_cond[k]), jnp.asarray(post_cond[k]))
        for k in common
    ):
        return "inferred cond is bit-identical to the pre-inference cond (inference had no effect)"
    if agent is not None:
        from env.base import needs_goal_frame

        if needs_goal_frame(agent):
            return (
                "the eval route supplies an env goal frame for this critic, and "
                "Agent.sample_actions merges a supplied goal over cond_eval (critics "
                "with a map_eval_cond hook then re-derive the conditioning from it), so "
                "the inferred vector is discarded before any action is drawn -- pass "
                "--eval_on_inferred_cond to evaluate on it instead"
            )
    return None


def route_b_invalid_reason(pre_cond, relabel_agent, rewards):
    """Reason a reward-inference row is invalid (no-op inference or zero z), or None."""
    reason = infer_cond_noop_reason(
        pre_cond, relabel_agent.cond_eval, rewards, agent=relabel_agent
    )
    if reason is None:
        norm = relabel_agent.witness()["inferred_cond_norm"]
        if not norm > 0.0:
            reason = (
                f"inferred_cond_norm={norm} (the conditioning vector that would be "
                f"rolled is zero or undefined)"
            )
    if reason is None:
        return None
    return f"{type(relabel_agent.critic).__name__}: {reason}"


def _latest_checkpoint_epoch(run_dir):
    """Largest N with run_dir/params_<N>.pkl, or None."""
    import glob as _glob
    import re as _re

    epochs = []
    for p in _glob.glob(os.path.join(run_dir, "params_*.pkl")):
        m = _re.fullmatch(r"params_(\d+)\.pkl", os.path.basename(p))
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


# Set on SIGUSR1/SIGTERM; the loop then checkpoints and exits.
_CHECKPOINT_SIGNAL = {"num": None}


def _request_checkpoint(signum, frame):  # noqa: ARG001
    _CHECKPOINT_SIGNAL["num"] = signum


def _drain_per_episode(run_dir, scope, step, row):
    """Move the row's `_per_episode` values to eval_episodes.jsonl; return the scalar row."""
    per = row.pop("_per_episode", None)
    if per is None:
        return row
    try:
        with open(os.path.join(run_dir, "eval_episodes.jsonl"), "a") as f:
            f.write(json.dumps({"step": int(step), "scope": scope, **per}) + "\n")
    except Exception as e:  # noqa: BLE001
        warnings.warn(
            f"per-episode eval log at step {step} failed: {type(e).__name__}: {e}"
        )
    return row


def main(_):
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, FLAGS.override)

    if FLAGS.eval_on_inferred_cond and FLAGS.infer_cond_episodes <= 0:
        raise ValueError(
            "--eval_on_inferred_cond needs --infer_cond_episodes>0: it evaluates on the "
            "conditioning inference produced, and with inference off that is eval_cond0 "
            "(a zero task vector => Q == 0, a task-blind policy), not a zero-shot readout."
        )

    # Datasets sample from the global NumPy RNG
    np.random.seed(FLAGS.seed)

    exp_name = get_exp_name(FLAGS.seed, FLAGS.critic, FLAGS.actor, cfg["env_name"])
    run_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save the resolved config
    try:
        with open(os.path.join(run_dir, "config.json"), "w") as _f:
            json.dump(
                {
                    "seed": FLAGS.seed,
                    "critic": FLAGS.critic,
                    "actor": FLAGS.actor,
                    "domain": FLAGS.domain,
                    "override": list(FLAGS.override or []),
                    "config": cfg,
                },
                _f,
                indent=2,
                default=str,
            )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"could not write config.json: {type(e).__name__}: {e}")

    # Checkpoint on walltime signals
    signal.signal(signal.SIGUSR1, _request_checkpoint)
    signal.signal(signal.SIGTERM, _request_checkpoint)

    run = setup_wandb(cfg, group=f"{FLAGS.critic}-{FLAGS.actor}", name=exp_name)
    csv_train = CsvLogger(os.path.join(run_dir, "train.csv"))
    csv_eval = CsvLogger(os.path.join(run_dir, "eval.csv"))
    csv_eval_relabel = CsvLogger(os.path.join(run_dir, "eval_relabel.csv"))

    try:
        adapter = get_adapter(cfg)
        env, train_ds, val_ds = adapter.make_env_and_datasets(cfg)
        train_ds = make_dataset(train_ds, cfg)
        ex = train_ds.sample(1)
        agent = Agent.create(
            FLAGS.critic,
            FLAGS.actor,
            FLAGS.seed,
            jnp.asarray(ex["observations"]),
            jnp.asarray(ex["actions"]),
            cfg,
        )

        start_step = 1
        if FLAGS.auto_resume:
            latest = _latest_checkpoint_epoch(run_dir)
            if latest is not None:
                print(f"[auto_resume] found {run_dir}/params_{latest}.pkl", flush=True)
                agent = restore_agent(agent, run_dir, latest)
                start_step = latest + 1
                # Re-seed so the resumed run does not replay its batches
                np.random.seed((FLAGS.seed * 1_000_003 + start_step) % (2**32))
            else:
                print(
                    f"[auto_resume] no params_*.pkl under {run_dir}; fresh start",
                    flush=True,
                )

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
                    rows = infer_cond_rows(agent.critic)
                    pre_cond = agent.cond_eval
                    eval_agent = agent.infer_eval_cond(
                        jnp.asarray(rb[rows]), jnp.asarray(rb["rewards"])
                    )
                    if FLAGS.eval_on_inferred_cond:
                        eval_agent = InferredCondAgent(eval_agent)
                    reason = infer_cond_noop_reason(
                        pre_cond, eval_agent.cond_eval, rb["rewards"], agent=eval_agent
                    )
                    if reason is not None:
                        warnings.warn(
                            f"--infer_cond_episodes={FLAGS.infer_cond_episodes} requested but "
                            f"inference had no effect this eval: {reason}. Evaluation is "
                            f"using eval_cond0 (or the env task goal), not an inferred "
                            f"task vector -- these numbers are not a zero-shot result."
                        )
                    if FLAGS.eval_on_inferred_cond:
                        print(
                            f"[infer-cond] step {i}: evaluating on the inferred "
                            f"conditioning from {FLAGS.infer_cond_episodes} rb[{rows!r}] "
                            f"rows; the env goal is NOT supplied",
                            flush=True,
                        )
                for task_id in adapter.eval_tasks(env, cfg):
                    if isinstance(eval_agent, InferredCondAgent):
                        eval_agent.reset_counts()
                    row, _ = adapter.evaluate(
                        eval_agent, env, task_id, cfg, FLAGS.eval_episodes, 0
                    )
                    if isinstance(eval_agent, InferredCondAgent):
                        row = {**row, **eval_agent.witness()}
                    scope = "eval" if task_id is None else f"eval/task{task_id}"
                    row = _drain_per_episode(run_dir, scope, i, row)
                    csv_eval.log(
                        {**row, "task_id": -1 if task_id is None else task_id}, step=i
                    )
                    if run is not None:
                        run.log({f"{scope}/{k}": v for k, v in row.items()}, step=i)
                    print("eval", i, "task", task_id, row, flush=True)

                # Reward-inference evaluation
                if cfg.get("eval_relabel_size"):
                    relabel_fn = getattr(adapter, "relabel_infer_batch", None)
                    if relabel_fn is None:
                        raise NotImplementedError(
                            f"eval_relabel_size={cfg['eval_relabel_size']} is set but adapter "
                            f"{type(adapter).__name__} has no relabel_infer_batch"
                        )
                    _task_infos = getattr(
                        getattr(env, "unwrapped", env), "task_infos", None
                    )
                    n_relabel_tasks = (
                        len(_task_infos)
                        if _task_infos is not None
                        else len(adapter.eval_tasks(env, cfg))
                    )
                    n_relabel_eps = int(cfg.get("eval_relabel_episodes", 10))
                    for tid in range(1, n_relabel_tasks + 1):
                        relabel_obs, relabel_rewards = relabel_fn(
                            train_ds,
                            env,
                            tid,
                            cfg["eval_relabel_size"],
                            shift_reward=cfg.get("eval_relabel_shift", 1.0),
                        )
                        pre_cond = agent.cond_eval
                        relabel_agent = InferredCondAgent(
                            agent.infer_eval_cond(
                                jnp.asarray(relabel_obs), jnp.asarray(relabel_rewards)
                            )
                        )
                        # Invalid rows are logged with success=NaN and not rolled
                        invalid = route_b_invalid_reason(
                            pre_cond, relabel_agent, relabel_rewards
                        )
                        if invalid is not None:
                            warnings.warn(
                                f"eval_relabel task {tid} at step {i}: reward inference had "
                                f"no effect ({invalid}); the row is written with success=NaN "
                                f"and invalid_reason set, and is NOT a zero-shot result."
                            )
                            row = {"success": float("nan"), **relabel_agent.witness()}
                        else:
                            row, _ = adapter.evaluate(
                                relabel_agent,
                                env,
                                tid,
                                cfg,
                                n_relabel_eps,
                                0,
                                any_step_success=True,
                            )
                            row = {**row, **relabel_agent.witness()}
                        row = {
                            **row,
                            "route_b_valid": 0.0 if invalid else 1.0,
                            "invalid_reason": invalid or "",
                        }
                        scope = f"eval_relabel/task{tid}"
                        row = _drain_per_episode(run_dir, scope, i, row)
                        csv_eval_relabel.log({**row, "task_id": tid}, step=i)
                        if run is not None:
                            run.log(
                                {
                                    f"{scope}/{k}": v
                                    for k, v in row.items()
                                    if not isinstance(v, str)
                                },
                                step=i,
                            )
                        print("eval_relabel", i, "task", tid, row, flush=True)
            if i % FLAGS.save_interval == 0:
                save_agent(agent, run_dir, i)
            if _CHECKPOINT_SIGNAL["num"] is not None:
                if i % FLAGS.save_interval != 0:
                    save_agent(agent, run_dir, i)
                print(
                    f"[signal] caught signal {_CHECKPOINT_SIGNAL['num']} -- checkpointed "
                    f"at step {i} and exiting; resubmit with --auto_resume to continue",
                    flush=True,
                )
                break
    finally:
        csv_train.close()
        csv_eval.close()
        csv_eval_relabel.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    app.run(main)
