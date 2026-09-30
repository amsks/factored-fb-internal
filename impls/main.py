"""Training entry point: compose a critic x actor x domain config, train offline, evaluate.

Two evaluation routes run at every eval point:
  * eval.csv -- the env's own task specification. Goal-conditioned for OGBench, MetaWorld
    and ManiSkill; with --infer_cond_episodes and --eval_on_inferred_cond (ExoRL) the policy
    is instead conditioned on the task vector inferred from dataset reward samples.
  * eval_relabel.csv -- reward-inference evaluation (when the config sets
    eval_relabel_size): relabel dataset transitions with each task's reward, infer the
    task vector from those samples, and roll the policy on it with no goal supplied.
"""
import json
import os
import signal
import warnings

# The EGL device pin must precede any mujoco/ogbench import. CUDA_VISIBLE_DEVICES masks JAX
# but not MuJoCo's EGL enumeration, so pin EGL to the same physical GPU.
os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

from absl import app, flags
import jax.numpy as jnp
import numpy as np

from impls.agents import Agent
from impls.critics.base import infer_cond_rows
# Re-exported: callers and tests look for it here. It lives in a flag-free module so a
# second entry point can import it without re-registering this module's absl flags.
from impls.utils.eval_cond import InferredCondAgent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import restore_agent, save_agent
from impls.utils.log_utils import CsvLogger, get_exp_name, setup_wandb

FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "fb", "Critic name (impls/configs/critic/<name>.yaml).")
flags.DEFINE_string("actor", "flowbc", "Actor name (impls/configs/actor/<name>.yaml).")
flags.DEFINE_string("domain", "cube_single", "Domain name (env/domains or impls/configs/domain).")
flags.DEFINE_multi_string("override", [], "Config override, dotlist form (e.g. batch_size=64).")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("train_steps", 1_000_000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Log every N steps.")
flags.DEFINE_integer("eval_interval", 100_000, "Eval every N steps.")
flags.DEFINE_integer("save_interval", 1_000_000, "Checkpoint every N steps.")
flags.DEFINE_integer("eval_episodes", 20, "Episodes per eval task.")
flags.DEFINE_string("save_dir", "exp/", "Checkpoint directory.")
flags.DEFINE_bool("auto_resume", False,
                  "Resume from the latest params_<N>.pkl in this run's own directory "
                  "(save_dir/exp_name), continuing at step N+1. With no checkpoint present "
                  "this is a fresh start, so a launcher can pass it unconditionally. "
                  "eval.csv/train.csv are appended, so the step axis continues.")
flags.DEFINE_integer("infer_cond_episodes", 0,
                     "If >0, infer the eval task vector from this many dataset transitions "
                     "and their rewards before each eval. The rows fed to infer_cond are the "
                     "critic's declared INFER_COND_MARGINAL (impls/critics/base.py). On its "
                     "own this only computes the vector; --eval_on_inferred_cond rolls it.")
flags.DEFINE_bool("eval_on_inferred_cond", False,
                  "Evaluate on the task vector --infer_cond_episodes inferred instead of the "
                  "env's task goal. Without it, the evaluator hands goal-capable critics a "
                  "goal frame, which overrides the inferred vector.")


def infer_cond_noop_reason(pre_cond, post_cond, rewards, agent=None):
    """Why infer_eval_cond() had no real effect, as a string to warn on, or None.

    Cases detected:
      - the reward samples are all zero (e.g. unlabeled OGBench play data), so the reward-
        weighted readout is zero and the task vector equals eval_cond0;
      - the inferred conditioning is bit-identical to the one before inference;
      - `agent` (the object eval will be driven with) is handed an env goal frame, which
        Agent.sample_actions merges over cond_eval, so the inferred vector is discarded.
    """
    rewards = jnp.asarray(rewards)
    if rewards.size and bool(jnp.all(rewards == 0)):
        return "sampled rewards are all zero (no reward signal to infer a conditioning vector from)"
    common = set(pre_cond) & set(post_cond)
    if common and all(
        jnp.array_equal(jnp.asarray(pre_cond[k]), jnp.asarray(post_cond[k])) for k in common
    ):
        return "inferred cond is bit-identical to the pre-inference cond (inference had no effect)"
    if agent is not None:
        from env.base import needs_goal_frame
        if needs_goal_frame(agent):
            return ("the eval route supplies an env goal frame for this critic, and "
                    "Agent.sample_actions merges a supplied goal over cond_eval (critics "
                    "with a map_eval_cond hook then re-derive the conditioning from it), so "
                    "the inferred vector is discarded before any action is drawn -- pass "
                    "--eval_on_inferred_cond to evaluate on it instead")
    return None


def route_b_invalid_reason(pre_cond, relabel_agent, rewards):
    """Why a reward-inference (eval_relabel) row must NOT be scored, or None if valid.

    A critic whose infer_cond is a placeholder (goal-conditioned density_fb with the task
    index off returns the zero goal) would be rolled on goal=0 and look like a zero-shot
    number. So the row is checked with `infer_cond_noop_reason`, and then the rolled vector
    itself: a zero or undefined norm is not a task specification."""
    reason = infer_cond_noop_reason(pre_cond, relabel_agent.cond_eval, rewards,
                                    agent=relabel_agent)
    if reason is None:
        norm = relabel_agent.witness()["inferred_cond_norm"]
        if not norm > 0.0:                       # also catches NaN
            reason = (f"inferred_cond_norm={norm} (the conditioning vector that would be "
                      f"rolled is zero or undefined)")
    if reason is None:
        return None
    return f"{type(relabel_agent.critic).__name__}: {reason}"


def _latest_checkpoint_epoch(run_dir):
    """Largest N for which run_dir/params_<N>.pkl exists, or None. Signal-triggered
    checkpoints land at arbitrary steps, so this scans rather than stepping by
    save_interval."""
    import glob as _glob
    import re as _re
    epochs = []
    for p in _glob.glob(os.path.join(run_dir, "params_*.pkl")):
        m = _re.fullmatch(r"params_(\d+)\.pkl", os.path.basename(p))
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


# Set by the signal handler, checked once per training step. SIGUSR1 is what slurm's
# `--signal=USR1@N` delivers before the walltime kill; SIGTERM is what scancel sends. Either
# way: finish the current step, write a checkpoint, exit so a resubmission with
# --auto_resume continues from it.
_CHECKPOINT_SIGNAL = {"num": None}


def _request_checkpoint(signum, frame):                     # noqa: ARG001 - signal API
    _CHECKPOINT_SIGNAL["num"] = signum


def _drain_per_episode(run_dir, scope, step, row):
    """Pop the `_per_episode` payload out of an eval row, append it to
    <run_dir>/eval_episodes.jsonl, and return the scalar-only row.

    Per-episode values (goal distances, episode returns) cannot ride in the row: CsvLogger
    freezes its header on the first row and wandb logs scalars. The file is append-only and
    keyed by (step, scope)."""
    per = row.pop("_per_episode", None)
    if per is None:
        return row
    try:
        with open(os.path.join(run_dir, "eval_episodes.jsonl"), "a") as f:
            f.write(json.dumps({"step": int(step), "scope": scope, **per}) + "\n")
    except Exception as e:                                  # noqa: BLE001 - reporting only
        warnings.warn(f"per-episode eval log at step {step} failed: {type(e).__name__}: {e}")
    return row


def main(_):
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, FLAGS.override)

    # --eval_on_inferred_cond with no inference would evaluate on eval_cond0 (a zero task
    # vector, Q == 0, a task-blind policy) while logging "inferred".
    if FLAGS.eval_on_inferred_cond and FLAGS.infer_cond_episodes <= 0:
        raise ValueError(
            "--eval_on_inferred_cond needs --infer_cond_episodes>0: it evaluates on the "
            "conditioning inference produced, and with inference off that is eval_cond0 "
            "(a zero task vector => Q == 0, a task-blind policy), not a zero-shot readout.")

    # The datasets draw every batch index, goal index and relabel coin from the global NumPy
    # RNG, so --seed has to reach it or two seeds replay the same batch stream.
    np.random.seed(FLAGS.seed)

    exp_name = get_exp_name(FLAGS.seed, FLAGS.critic, FLAGS.actor, cfg["env_name"])
    run_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    # The resolved config next to the results (the seed is injected after make_config).
    try:
        with open(os.path.join(run_dir, "config.json"), "w") as _f:
            json.dump({"seed": FLAGS.seed, "critic": FLAGS.critic, "actor": FLAGS.actor,
                       "domain": FLAGS.domain, "override": list(FLAGS.override or []),
                       "config": cfg}, _f, indent=2, default=str)
    except Exception as e:                                  # noqa: BLE001
        warnings.warn(f"could not write config.json: {type(e).__name__}: {e}")

    # Walltime insurance: the handler only sets a flag; the loop checkpoints and exits.
    signal.signal(signal.SIGUSR1, _request_checkpoint)
    signal.signal(signal.SIGTERM, _request_checkpoint)

    run = setup_wandb(cfg, group=f"{FLAGS.critic}-{FLAGS.actor}", name=exp_name)
    csv_train = CsvLogger(os.path.join(run_dir, "train.csv"))
    csv_eval = CsvLogger(os.path.join(run_dir, "eval.csv"))
    # A separate file: CsvLogger freezes its header on the first row, so the relabel route's
    # extra inferred_cond_* columns would be dropped if a goal-conditioned row logged first.
    csv_eval_relabel = CsvLogger(os.path.join(run_dir, "eval_relabel.csv"))

    try:
        adapter = get_adapter(cfg)
        env, train_ds, val_ds = adapter.make_env_and_datasets(cfg)
        train_ds = make_dataset(train_ds, cfg)
        ex = train_ds.sample(1)
        agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                             jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)

        start_step = 1
        if FLAGS.auto_resume:
            latest = _latest_checkpoint_epoch(run_dir)
            if latest is not None:
                print(f"[auto_resume] found {run_dir}/params_{latest}.pkl", flush=True)
                agent = restore_agent(agent, run_dir, latest)
                start_step = latest + 1
                # Re-seed the batch stream as a function of (seed, step): the resumed run does
                # not replay the batches it already trained on, and seeds stay distinct.
                np.random.seed((FLAGS.seed * 1_000_003 + start_step) % (2 ** 32))
            else:
                print(f"[auto_resume] no params_*.pkl under {run_dir}; fresh start", flush=True)

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
                    # Which rows feed infer_cond is the critic's declaration: the FB family
                    # is solved against the next-state marginal its backward net is trained on.
                    rows = infer_cond_rows(agent.critic)
                    pre_cond = agent.cond_eval
                    eval_agent = agent.infer_eval_cond(
                        jnp.asarray(rb[rows]), jnp.asarray(rb["rewards"]))
                    if FLAGS.eval_on_inferred_cond:
                        # Eval through a view that never forwards a goal, so nothing can
                        # overwrite the inferred task vector.
                        eval_agent = InferredCondAgent(eval_agent)
                    reason = infer_cond_noop_reason(pre_cond, eval_agent.cond_eval,
                                                    rb["rewards"], agent=eval_agent)
                    if reason is not None:
                        warnings.warn(
                            f"--infer_cond_episodes={FLAGS.infer_cond_episodes} requested but "
                            f"inference had no effect this eval: {reason}. Evaluation is "
                            f"using eval_cond0 (or the env task goal), not an inferred "
                            f"task vector -- these numbers are not a zero-shot result."
                        )
                    if FLAGS.eval_on_inferred_cond:
                        print(f"[infer-cond] step {i}: evaluating on the inferred "
                              f"conditioning from {FLAGS.infer_cond_episodes} rb[{rows!r}] "
                              f"rows; the env goal is NOT supplied", flush=True)
                for task_id in adapter.eval_tasks(env, cfg):
                    if isinstance(eval_agent, InferredCondAgent):
                        eval_agent.reset_counts()      # per-task counters, not cumulative
                    row, _ = adapter.evaluate(eval_agent, env, task_id, cfg,
                                              FLAGS.eval_episodes, 0)
                    if isinstance(eval_agent, InferredCondAgent):
                        # Columns that prove the run rolled an inferred task vector.
                        row = {**row, **eval_agent.witness()}
                    scope = "eval" if task_id is None else f"eval/task{task_id}"
                    row = _drain_per_episode(run_dir, scope, i, row)
                    csv_eval.log({**row, "task_id": -1 if task_id is None else task_id}, step=i)
                    if run is not None:
                        run.log({f"{scope}/{k}": v for k, v in row.items()}, step=i)
                    print("eval", i, "task", task_id, row, flush=True)

                # Reward-inference evaluation: for each task, relabel dataset transitions
                # with that task's reward, infer the task vector, roll it with no goal.
                if cfg.get("eval_relabel_size"):
                    relabel_fn = getattr(adapter, "relabel_infer_batch", None)
                    if relabel_fn is None:
                        raise NotImplementedError(
                            f"eval_relabel_size={cfg['eval_relabel_size']} is set but adapter "
                            f"{type(adapter).__name__} has no relabel_infer_batch")
                    # OGBench: one relabel task per task_info. Single-task adapters expose no
                    # task_infos, so the loop runs once with tid=1.
                    _task_infos = getattr(getattr(env, "unwrapped", env), "task_infos", None)
                    n_relabel_tasks = (len(_task_infos) if _task_infos is not None
                                       else len(adapter.eval_tasks(env, cfg)))
                    n_relabel_eps = int(cfg.get("eval_relabel_episodes", 10))
                    for tid in range(1, n_relabel_tasks + 1):
                        relabel_obs, relabel_rewards = relabel_fn(
                            train_ds, env, tid, cfg["eval_relabel_size"],
                            shift_reward=cfg.get("eval_relabel_shift", 1.0))
                        pre_cond = agent.cond_eval
                        relabel_agent = InferredCondAgent(agent.infer_eval_cond(
                            jnp.asarray(relabel_obs), jnp.asarray(relabel_rewards)))
                        # No-op guard: a placeholder infer_cond would otherwise be rolled on
                        # goal=0 and scored. Such a row is written with success=NaN and an
                        # invalid_reason, and is not rolled. Both columns are on every row so
                        # the frozen CSV header carries them.
                        invalid = route_b_invalid_reason(pre_cond, relabel_agent,
                                                         relabel_rewards)
                        if invalid is not None:
                            warnings.warn(
                                f"eval_relabel task {tid} at step {i}: reward inference had "
                                f"no effect ({invalid}); the row is written with success=NaN "
                                f"and invalid_reason set, and is NOT a zero-shot result.")
                            row = {"success": float("nan"), **relabel_agent.witness()}
                        else:
                            row, _ = adapter.evaluate(
                                relabel_agent, env, tid, cfg, n_relabel_eps, 0,
                                any_step_success=True)
                            row = {**row, **relabel_agent.witness()}
                        row = {**row, "route_b_valid": 0.0 if invalid else 1.0,
                               "invalid_reason": invalid or ""}
                        scope = f"eval_relabel/task{tid}"
                        row = _drain_per_episode(run_dir, scope, i, row)
                        csv_eval_relabel.log({**row, "task_id": tid}, step=i)
                        if run is not None:
                            run.log({f"{scope}/{k}": v for k, v in row.items()
                                     if not isinstance(v, str)}, step=i)
                        print("eval_relabel", i, "task", tid, row, flush=True)
            if i % FLAGS.save_interval == 0:
                save_agent(agent, run_dir, i)
            if _CHECKPOINT_SIGNAL["num"] is not None:
                # Signal-triggered checkpoint at an arbitrary step, then a clean exit.
                if i % FLAGS.save_interval != 0:
                    save_agent(agent, run_dir, i)
                print(f"[signal] caught signal {_CHECKPOINT_SIGNAL['num']} -- checkpointed "
                      f"at step {i} and exiting; resubmit with --auto_resume to continue",
                      flush=True)
                break
    finally:
        csv_train.close()
        csv_eval.close()
        csv_eval_relabel.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    app.run(main)
