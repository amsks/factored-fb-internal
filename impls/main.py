"""Training entrypoint for the factored impls/ tree (PR1: FB critic × FlowBC actor).

Config is composed from the critic/actor/domain sibling groups under impls/configs/ via
load_config() (spec §3); the old flat get_config() has been replaced.
"""
import dataclasses
import os
import signal
import warnings

# Device/EGL pin MUST precede any mujoco/ogbench import (carried from run_gciql.py:97-104).
# CUDA_VISIBLE_DEVICES masks JAX but not MuJoCo's EGL enumeration; pin EGL to the same
# physical GPU so concurrent eval renders don't pile onto GPU 0 (GL 0x8cdd).
os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

from absl import app, flags
import jax.numpy as jnp
import numpy as np

from impls.agents import Agent
from impls.critics.base import infer_cond_rows
# Re-exported: this class used to be defined here, and impls/main.py is where callers
# and tests look for it. It moved to a flag-free module so a second entrypoint can
# import it without re-registering this module's absl flags (see that file's docstring).
from impls.utils.eval_cond import InferredCondAgent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import (resolve_run_dir, restore_agent, save_agent,
                                    warm_start_agent)
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
flags.DEFINE_integer("num_video", 0,
                     "Episodes to render per eval, written to <run_dir>/videos/. These are "
                     "EXTRA episodes -- the metric episodes are never rendered.")
flags.DEFINE_string("save_dir", "exp/", "Checkpoint directory.")
flags.DEFINE_string("restore_path", "", "Glob matching exactly one run dir to restore from.")
flags.DEFINE_integer("restore_epoch", 0, "Checkpoint step to restore (params_{epoch}.pkl).")
flags.DEFINE_bool("auto_resume", False,
                  "Resume from the LATEST params_<N>.pkl already in this run's own "
                  "directory (save_dir/exp_name), continuing training at step N+1. With no "
                  "checkpoint present this is a fresh start, so a launcher can pass it "
                  "unconditionally and resubmission after a walltime kill just works. "
                  "Takes precedence over --restore_path when a checkpoint exists; "
                  "eval.csv/train.csv are appended, not truncated (CsvLogger already does "
                  "this), so the step axis continues.")
flags.DEFINE_string("warm_start_path", "",
                    "Glob matching exactly one run dir to warm-start params from (params + "
                    "target params only, fresh optimizer state and rng -- distinct from "
                    "--restore_path, which is a full-state resume). td_fb's phase-2 recipe "
                    "(measure_stopgrad_psi=true) requires this.")
flags.DEFINE_integer("warm_start_epoch", 0, "Checkpoint step to warm-start from (params_{epoch}.pkl).")
flags.DEFINE_string("agg_dir", None, "optional: dir for per-seed eval jsonls (campaign/wandb_agg_daemon.py)")
flags.DEFINE_string("agg_group", None, "optional aggregation key; default <critic>_<actor>__<domain>")
flags.DEFINE_integer("infer_cond_episodes", 0,
                     "If >0, infer eval conditioning from this many relabelled transitions "
                     "before each eval (FB zero-shot path). 0 = use eval_cond0. The rows fed "
                     "to infer_cond are chosen by the critic's own INFER_COND_MARGINAL "
                     "declaration (impls/critics/base.py), not hard-coded here. On its own "
                     "this flag only PRODUCES a conditioning vector -- see "
                     "--eval_on_inferred_cond for the flag that makes eval actually roll it.")
flags.DEFINE_bool("eval_on_inferred_cond", False,
                  "Evaluate on the conditioning --infer_cond_episodes inferred, INSTEAD of "
                  "the env's task goal. Requires --infer_cond_episodes>0. Without this the "
                  "evaluator hands every goal-capable critic a goal frame, Agent."
                  "sample_actions merges it over cond_eval, and the inferred index is "
                  "discarded before it is ever used -- so the reward-sample interface was "
                  "unreachable through the normal eval path. Off by default: a run that is "
                  "legitimately goal-conditioned must keep behaving exactly as it did.")

# --- ONLINE (env-interactive) mode. OFF by default; see env/online.py. -------------------
# Every knob below is inert unless --online is passed: with the flag off, the training loop
# executes exactly the statements it did before online mode existed (one `if` on a local
# that is None), env/online.py is never even imported, and no extra randomness is drawn.
# That is deliberate -- this repo compares runs against recorded numbers.
flags.DEFINE_bool("online", False,
                  "Train ONLINE: step the env inside the training loop and sample gradient "
                  "batches from a replay buffer instead of the offline dataset. Default "
                  "False = the offline path, unchanged.")
flags.DEFINE_float("online_env_steps_per_update", 1.0,
                   "Env steps per gradient step (UTD = 1/this). Fractional values are "
                   "accumulated, so 0.5 means one env step every other gradient step.")
flags.DEFINE_float("online_action_noise", 0.1,
                   "Std of the Gaussian action noise added to the agent's sampled action "
                   "during online collection (0 = greedy w.r.t. the actor's own sampling).")
flags.DEFINE_float("online_temperature", 1.0,
                   "Temperature passed to agent.sample_actions during online collection "
                   "(eval uses 0).")
flags.DEFINE_integer("online_start_steps", 5000,
                     "Env steps collected with UNIFORM RANDOM actions before the first "
                     "gradient step. Collection continues past this until at least one "
                     "episode has finished (the buffer stores whole trajectories).")
flags.DEFINE_integer("online_buffer_size", 200_000, "Replay-buffer capacity in transitions.")
flags.DEFINE_bool("online_seed_offline", False,
                  "Warm-start the replay buffer from the offline dataset (a prefix of whole "
                  "episodes).")
flags.DEFINE_integer("online_seed_transitions", 0,
                     "Cap on the rows copied by --online_seed_offline. 0 = as many whole "
                     "episodes as the buffer holds.")
flags.DEFINE_integer("online_reset_seed_offset", 1_000_000,
                     "Offset applied to online episode reset seeds so online rollouts never "
                     "reuse the initial states evaluate() scores (it uses seeds 0..N).")
flags.DEFINE_bool("online_restore_buffer", True,
                  "On --restore_path with --online, restore the replay buffer, collector "
                  "position and exploration RNG from the checkpoint's online_state_<epoch>."
                  "npz, and REFUSE the resume if that file is absent. False accepts an "
                  "EMPTY buffer and a re-run prefill instead -- i.e. a fully-trained agent "
                  "collecting a fresh warmup set, which is almost never what a resume means.")
flags.DEFINE_float("online_explore_goal_prob", 0.0,
                   "Per EPISODE, probability of collecting against a goal drawn uniformly "
                   "from the replay buffer's own states instead of the eval goal. 0.0 "
                   "(default) reproduces the existing behaviour bit-for-bit -- no draw is "
                   "made at all. This is a coverage dial against the narrowness documented "
                   "in env/online.py (the collection policy is conditioned on the eval goal, "
                   "so hindsight goals inherit that support); it is not an exploration "
                   "algorithm.")


def infer_cond_noop_reason(pre_cond, post_cond, rewards, agent=None):
    """Detect when infer_eval_cond() had no real effect, and return a string naming why
    (for a caller to warn on), or None if inference looks like it did something.

    Three known no-op cases on the currently-supported configs:
      - cube-single-play-v0 is unlabeled play data, so FBDataset zeroes every reward
        (impls/utils/datasets.py); FB's z = (rewards^T B)/N is then exactly 0, and
        project_z(0) returns exact zeros -- bit-identical to eval_cond0's zero z.
      - GCIQLCritic.infer_cond (impls/critics/gciql.py) ignores obs/rewards entirely and
        always returns a zero placeholder cond["goal"]; evaluate() overrides cond["goal"]
        with the env task goal regardless (impls/utils/evaluation.py), so this cond is
        never even read.
      - THE ONE THAT MOTIVATED --eval_on_inferred_cond: inference genuinely changed
        cond_eval, but the eval route is about to hand the actor a goal frame, which
        Agent.sample_actions merges over cond_eval and map_eval_cond re-encodes. The
        inferred vector is computed, logged as "changed", and then discarded before a
        single action is drawn from it. Pass `agent` (the object eval will actually be
        driven with) to get this case detected.
    A user passing --infer_cond_episodes>0 on any of these paths gets ordinary
    eval_cond0/env-goal behavior with no indication z-inference actually ran, unless we
    say so here.
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
    """Why a Route B (eval_relabel) row must NOT be scored, or None if inference is live.

    Route B rolls whatever `infer_eval_cond` returned. For a critic whose `infer_cond` is a
    placeholder (goal-conditioned density_fb with measure_density_z off returns the zero
    goal; gciql likewise) that means the policy is rolled on goal=0 and the resulting success
    rate looks exactly like a zero-shot number in eval_relabel.csv. So the row is checked
    twice: `infer_cond_noop_reason` (bit-identical cond, all-zero rewards, goal override) and
    then the witness itself -- a rolled conditioning vector of zero or undefined norm is
    not a task specification whatever the detector said. The reason names the critic so a
    hub row is self-explaining."""
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


_FAMILY = {"b2d": "Box2D", "p2d": "Box2D", "mw": "Metaworld", "ms": "ManiSkill",
           "dmc": "ExoRL"}


def _latest_checkpoint_epoch(run_dir):
    """Largest N for which run_dir/params_<N>.pkl exists, or None. Signal-triggered
    checkpoints land at arbitrary (non-save_interval) steps, so this scans rather than
    stepping by save_interval."""
    import glob as _glob
    import re as _re
    epochs = []
    for p in _glob.glob(os.path.join(run_dir, "params_*.pkl")):
        m = _re.fullmatch(r"params_(\d+)\.pkl", os.path.basename(p))
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


# Set by the signal handler, checked once per training step (the jax step is synchronous,
# so a between-steps check is prompt enough). SIGUSR1 is what `#SBATCH --signal=USR1@N`
# delivers N seconds before the walltime kill; SIGTERM is what scancel sends first. Either
# way: finish the current step, write a checkpoint, exit 0 so the resubmission (with
# --auto_resume) continues from it.
_CHECKPOINT_SIGNAL = {"num": None}


def _request_checkpoint(signum, frame):                     # noqa: ARG001 - signal API
    _CHECKPOINT_SIGNAL["num"] = signum


def _append_agg(agg_dir, group, seed, step, row, task):
    """Optional: append this run's eval point as a jsonl the campaign agg daemon reads.
    Pure reporting -- writes nothing unless --agg_dir is set; touches no algorithm state."""
    import json
    os.makedirs(agg_dir, exist_ok=True)
    # NaN, not 0.0, when the row carries no success metric: a domain with no binary success
    # rule (ExoRL walker/cheetah -- env/unified.py::evaluate omits the key there) would
    # otherwise be aggregated as a perfect-failure 0.0, which is indistinguishable from a
    # policy that failed every episode. NaN propagates through the daemon's mean and shows
    # up as a gap in the plot instead of a wrong number.
    rec = dict(seed=int(seed), step=int(step),
               success_rate=float(row.get("success", row.get("success_rate", float("nan")))),
               at_goal_ticks=float(row.get("at_goal_ticks", float("nan"))),
               family=_FAMILY.get(str(task).split("-")[0]))
    with open(os.path.join(agg_dir, f"{group}__s{seed}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _drain_per_episode(run_dir, scope, step, row):
    """Pop the `_per_episode` payload out of an eval row and append it to
    <run_dir>/eval_episodes.jsonl, returning the scalar-only row.

    Spec E0.2: E5 re-scores success at a sweep of goal radii, which needs the per-episode
    distances, not their mean. They cannot ride in the eval row -- CsvLogger freezes its
    header on the first row and wandb's scalar log would choke on a list -- so they get their
    own append-only file, keyed by (step, scope) so a re-eval pass adds rows rather than
    overwriting (the eval.csv corruption mode in the spec's gotchas)."""
    per = row.pop("_per_episode", None)
    if per is None:
        return row
    try:
        import json as _json
        with open(os.path.join(run_dir, "eval_episodes.jsonl"), "a") as f:
            f.write(_json.dumps({"step": int(step), "scope": scope, **per}) + "\n")
    except Exception as e:                                  # noqa: BLE001 - reporting only
        warnings.warn(f"per-episode eval log at step {step} failed: {type(e).__name__}: {e}")
    return row


def _write_eval_video(run_dir, scope, step, frames):
    """Tile this eval point's rendered episodes into <run_dir>/videos/<scope>_step<N>.mp4.

    Best-effort by design: a codec or render failure must cost a video, never a 300k-step
    training run, so anything raised here is reported and swallowed."""
    if not frames:
        return None
    try:
        from eval.video import save_video_grid
        name = f"{scope.replace('/', '-')}_step{step}.mp4"
        return save_video_grid(frames, os.path.join(run_dir, "videos", name))
    except Exception as e:                                  # noqa: BLE001 - see docstring
        warnings.warn(f"eval video at step {step} failed: {type(e).__name__}: {e}")
        return None


def _log_video(run, scope, step, video_path):
    """Attach the eval video to the wandb run (works in offline mode; synced later)."""
    if video_path is None:
        return
    try:
        import wandb
        run.log({f"{scope}/video": wandb.Video(video_path, format="mp4")}, step=step)
    except Exception as e:                                  # noqa: BLE001
        warnings.warn(f"wandb video log at step {step} failed: {type(e).__name__}: {e}")


def main(_):
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, FLAGS.override)

    # Fail on the FLAG COMBINATION before anything expensive is built. --eval_on_inferred_cond
    # with no inference to roll would evaluate on eval_cond0 -- a zero z, for which Q(s,a;0)
    # == 0 and the policy is goal-blind -- while the log said "inferred". That number would be
    # indistinguishable from a real zero-shot result.
    if FLAGS.eval_on_inferred_cond and FLAGS.infer_cond_episodes <= 0:
        raise ValueError(
            "--eval_on_inferred_cond needs --infer_cond_episodes>0: it evaluates on the "
            "conditioning inference produced, and with inference off that is eval_cond0 "
            "(FB's zero z => Q == 0, a goal-blind policy), not a zero-shot readout.")

    # GCDataset draws every batch index, goal index and relabel coin from the global NumPy
    # RNG (impls/utils/datasets.py), so --seed has to reach it or two seeds replay the same
    # batch stream and the reported seed-to-seed sd is not a seed estimate. (The other half
    # of that bug lives in env/unified.py::seeded_reset, which restores the global state.)
    np.random.seed(FLAGS.seed)

    exp_name = get_exp_name(FLAGS.seed, FLAGS.critic, FLAGS.actor, cfg["env_name"])
    # Each run gets its own subdirectory so two seeds or two critic x actor pairs sharing a
    # --save_dir don't overwrite each other's train.csv/eval.csv/params_N.pkl.
    # --restore_path stays a glob over run directories (impls/utils/flax_utils.py:139-148);
    # it should now point at save_dir/exp_name for a prior run.
    run_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    # Dump the RESOLVED config next to the results. Without this the only record of what a run
    # actually used is the binary wandb file, so answering "was the mask on for this row?"
    # months later means parsing an offline wandb record -- and --seed never appears there at
    # all, since it is injected after make_config.
    try:
        import json as _json
        with open(os.path.join(run_dir, "config.json"), "w") as _f:
            _json.dump({"seed": FLAGS.seed, "critic": FLAGS.critic, "actor": FLAGS.actor,
                        "domain": FLAGS.domain, "override": list(FLAGS.override or []),
                        "config": cfg}, _f, indent=2, default=str)
    except Exception as e:                                  # noqa: BLE001
        warnings.warn(f"could not write config.json: {type(e).__name__}: {e}")

    # Walltime insurance: slurm's `--signal=USR1@N` (scripts/vista_train.slurm) lands here,
    # as does scancel's SIGTERM. The handler only sets a flag; the training loop below
    # checks it after each step and does checkpoint-then-clean-exit, so a killed run loses
    # at most one step instead of everything since the last save_interval checkpoint.
    signal.signal(signal.SIGUSR1, _request_checkpoint)
    signal.signal(signal.SIGTERM, _request_checkpoint)

    agg_group = FLAGS.agg_group or f"{FLAGS.critic}_{FLAGS.actor}__{FLAGS.domain}"
    run = setup_wandb(cfg, group=f"{FLAGS.critic}-{FLAGS.actor}", name=exp_name)
    csv_train = CsvLogger(os.path.join(run_dir, "train.csv"))
    csv_eval = CsvLogger(os.path.join(run_dir, "eval.csv"))
    # A SEPARATE file, not a second schema squeezed into eval.csv: CsvLogger freezes its
    # header on the first row logged, so the relabel route's extra inferred_cond_* witness
    # columns -- the only positive evidence a run actually rolled a reward-inferred z -- would
    # be silently DROPPED the moment a goal-conditioned row (fewer columns) logs first, and
    # both routes share the same task_id, so their rows would also be indistinguishable in
    # one file. See CsvLogger's own docstring warning.
    csv_eval_relabel = CsvLogger(os.path.join(run_dir, "eval_relabel.csv"))

    try:
        adapter = get_adapter(cfg)
        env, train_ds, val_ds = adapter.make_env_and_datasets(cfg)
        train_ds = make_dataset(train_ds, cfg)
        ex = train_ds.sample(1)
        agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                             jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
        if FLAGS.warm_start_path:
            agent = warm_start_agent(agent, FLAGS.warm_start_path, FLAGS.warm_start_epoch)

        # Resolve WHAT to restore before touching the agent. --auto_resume looks in this
        # run's own directory and wins when a checkpoint exists there (the resubmit-after-
        # walltime-kill case: same flags, same run_dir); otherwise --restore_path (an
        # explicit cross-directory resume) applies as before; otherwise fresh start.
        start_step = 1
        restore_dir, restore_epoch = None, 0
        if FLAGS.auto_resume:
            latest = _latest_checkpoint_epoch(run_dir)
            if latest is not None:
                restore_dir, restore_epoch = run_dir, latest
                print(f"[auto_resume] found {run_dir}/params_{latest}.pkl", flush=True)
            else:
                print(f"[auto_resume] no params_*.pkl under {run_dir}; fresh start", flush=True)
        if restore_dir is None and FLAGS.restore_path:
            restore_dir = resolve_run_dir(FLAGS.restore_path)
            restore_epoch = FLAGS.restore_epoch
        if restore_dir is not None:
            agent = restore_agent(agent, restore_dir, restore_epoch)
            start_step = restore_epoch + 1
            # Re-seed the GLOBAL NumPy RNG (the dataset/batch stream -- see the
            # np.random.seed(FLAGS.seed) above) as a function of (seed, restored step).
            # Bit-exact continuation of the pre-kill batch stream is NOT attempted -- that
            # would require checkpointing np.random's state -- and is not needed; what this
            # buys is (a) the resumed run does not REPLAY the step-1..N batch indices it
            # already trained on, which plain np.random.seed(seed) would, and (b) two seeds
            # stay distinct after both resume at the same step.
            np.random.seed((FLAGS.seed * 1_000_003 + start_step) % (2 ** 32))

        # A resume under --online has to restore the REPLAY BUFFER too. restore_agent brings
        # back params/optimizer/rng only; without the buffer the resumed run starts empty,
        # re-runs prefill, and trains a fully-trained agent on a few thousand fresh warmup
        # transitions -- a silent change of training distribution that looks like a clean
        # resume in every log. --online_restore_buffer=false is the explicit opt-in to that
        # old behaviour; by default a missing online_state_<epoch>.npz is a hard error.
        restore_online = bool(restore_dir is not None and FLAGS.online
                              and FLAGS.online_restore_buffer)

        # ONLINE mode is entirely inside this block: `online` stays None with the flag off,
        # env/online.py is not imported, and the loop below runs the offline statements it
        # always ran. Nothing above this line is conditional on --online.
        online = None
        if FLAGS.online:
            from env.online import OnlineTrainer
            # Seeding from the offline dataset is skipped when a restore is about to replace
            # the buffer's contents wholesale -- the copy would be pure waste.
            online = OnlineTrainer.create(adapter, env, cfg, train_ds, ex, FLAGS,
                                          seed_offline=False if restore_online else None)
            if restore_online:
                n_rows = online.restore_state(restore_dir, restore_epoch)
                print(f"[online] restored {n_rows} buffer rows / "
                      f"{online.buffer.n_episodes} episodes, {online.env_steps} env steps",
                      flush=True)
            elif restore_dir is not None:
                warnings.warn(
                    "--restore_path with --online and --online_restore_buffer=false: the "
                    "replay buffer starts EMPTY and prefill re-runs, so this resume trains a "
                    "restored agent against freshly collected warmup data. That is a "
                    "different training distribution from the run being resumed.")
            n = online.prefill(agent)
            print(f"[online] prefilled {n} env steps, buffer {online.buffer.size} rows / "
                  f"{online.buffer.n_episodes} episodes", flush=True)

        # Two-phase schedule (RLDP official parity; docs/specs/2026-08-28-exorl-official-
        # parity-validation.md item A.1). `rldp_two_phase` is off for every other config, in
        # which case `phase_of` returns "joint" for all i and nothing below ever fires.
        two_phase = bool(cfg.get("rldp_two_phase", False))
        repr_steps = int(cfg.get("rldp_phase1_steps", 0))
        hard_b_every = int(cfg.get("rldp_hard_b_target", 0))

        def phase_of(step):
            if not two_phase:
                return "joint"
            return "mr" if step < repr_steps else "fb"

        def set_phase(ag, ph):
            """Swap the critic's static `phase`. The critic is a nonpytree_field, so this
            changes the jit key and retraces `Agent.update` ONCE, at the boundary -- which
            is the point: the phase decides which optimizer groups exist in the traced
            update, and that cannot be a traced value (see RLDPCritic.phase)."""
            return ag.replace(critic=dataclasses.replace(ag.critic, phase=ph))

        cur_phase = phase_of(start_step)
        if two_phase:
            agent = set_phase(agent, cur_phase)
            print(f"[two-phase] start_step={start_step} phase={cur_phase} "
                  f"(phase1_steps={repr_steps}, hard_b_target={hard_b_every})", flush=True)

        for i in range(start_step, FLAGS.train_steps + 1):
            if two_phase:
                ph = phase_of(i)
                if ph != cur_phase:
                    # Entering phase 2. The official code carries the phase-1 backward map
                    # straight through (it is simply never stepped again); the target
                    # forward map begins polyaking from here.
                    agent, cur_phase = set_phase(agent, ph), ph
                    print(f"[two-phase] step {i}: -> phase {ph}", flush=True)
            if online is None:
                batch = train_ds.sample(cfg["batch_size"])
            else:
                online.collect(agent)
                batch = online.sample(cfg["batch_size"])
            agent, info = agent.update(batch)
            if two_phase and cur_phase == "mr" and hard_b_every and i % hard_b_every == 0:
                # Official phase-1 target refresh: a HARD COPY of the backward map every
                # N steps (agent.py:196-197), not a polyak. RLDPCritic.TARGETS reports
                # tau=0 for `backward` in this phase, so the two never fight.
                agent = agent.replace(
                    target={**agent.target, "backward": agent.net["backward"].params})
            if i % FLAGS.log_interval == 0:
                row = {k: float(jnp.asarray(v).mean()) for k, v in info.items()}
                if online is not None:
                    row.update(online.stats())
                csv_train.log(row, step=i)
                if run is not None:
                    run.log({f"train/{k}": v for k, v in row.items()}, step=i)
                print(i, row, flush=True)
            if i == 1 or i % FLAGS.eval_interval == 0:
                eval_agent = agent
                if FLAGS.infer_cond_episodes > 0:
                    rb = train_ds.sample(FLAGS.infer_cond_episodes)
                    # WHICH ROWS is the critic's call, not ours: the FB family's readout is
                    # solved against the NEXT-state marginal its backward/index tower is
                    # trained on, and this used to hard-code rb["observations"] for every
                    # critic. See impls/critics/base.py::INFER_COND_MARGINALS.
                    rows = infer_cond_rows(agent.critic)
                    pre_cond = agent.cond_eval
                    eval_agent = agent.infer_eval_cond(
                        jnp.asarray(rb[rows]), jnp.asarray(rb["rewards"]))
                    if FLAGS.eval_on_inferred_cond:
                        # THE route that makes the inferred conditioning reachable: eval is
                        # driven through a view that never forwards a goal, so nothing can
                        # overwrite what inference just solved for.
                        eval_agent = InferredCondAgent(eval_agent)
                    reason = infer_cond_noop_reason(pre_cond, eval_agent.cond_eval,
                                                    rb["rewards"], agent=eval_agent)
                    if reason is not None:
                        warnings.warn(
                            f"--infer_cond_episodes={FLAGS.infer_cond_episodes} requested but "
                            f"z-inference had no effect this eval: {reason}. Evaluation is "
                            f"using eval_cond0 (or the env task goal), not an inferred "
                            f"conditioning vector -- do not interpret these eval numbers as a "
                            f"zero-shot z-inference result."
                        )
                    if FLAGS.eval_on_inferred_cond:
                        print(f"[infer-cond] step {i}: evaluating on the inferred "
                              f"conditioning from {FLAGS.infer_cond_episodes} rb[{rows!r}] "
                              f"rows; the env goal is NOT supplied", flush=True)
                for task_id in adapter.eval_tasks(env, cfg):
                    if isinstance(eval_agent, InferredCondAgent):
                        eval_agent.reset_counts()      # per-task counters, not cumulative
                    row, frames = adapter.evaluate(eval_agent, env, task_id, cfg,
                                                   FLAGS.eval_episodes, FLAGS.num_video)
                    if isinstance(eval_agent, InferredCondAgent):
                        # The observability half of the requirement: these columns are what a
                        # reader of eval.csv/wandb uses to PROVE the run rolled an inferred
                        # index rather than an env goal. goals_dropped > 0 means the goal path
                        # was live and this wrapper intercepted it.
                        row = {**row, **eval_agent.witness()}
                    scope = "eval" if task_id is None else f"eval/task{task_id}"
                    # Non-scalar eval payloads (per-episode goal distances, spec E0.2) go to
                    # their own jsonl: eval.csv fixes its header on the first row and wandb
                    # takes scalars, so anything array-shaped has to leave the row first.
                    row = _drain_per_episode(run_dir, scope, i, row)
                    csv_eval.log({**row, "task_id": -1 if task_id is None else task_id}, step=i)
                    video_path = _write_eval_video(run_dir, scope, i, frames)
                    if run is not None:
                        run.log({f"{scope}/{k}": v for k, v in row.items()}, step=i)
                        _log_video(run, scope, i, video_path)
                    if FLAGS.agg_dir:
                        _append_agg(FLAGS.agg_dir, agg_group, FLAGS.seed, i, row,
                                    cfg.get("task", cfg.get("env_name")))
                    print("eval", i, "task", task_id, row, flush=True)

                # --- OGBench reward-relabel zero-shot eval (fb_tdj parity arm) ---------------
                # A SECOND route, alongside -- not instead of -- the goal-conditioned loop
                # above. Off (cfg['eval_relabel_size'] unset/0) by default, so every existing
                # run is byte-for-byte unaffected: nothing above this comment changed.
                # docs/notes/2026-09-03-fb-vs-tdjepa.md #7.
                if cfg.get("eval_relabel_size"):
                    relabel_fn = getattr(adapter, "relabel_infer_batch", None)
                    if relabel_fn is None:
                        raise NotImplementedError(
                            f"eval_relabel_size={cfg['eval_relabel_size']} is set but adapter "
                            f"{type(adapter).__name__} has no relabel_infer_batch -- "
                            f"OGBenchAdapter and UnifiedAdapter implement the td_jepa "
                            f"reward-relabel protocol (docs/notes/2026-09-03-fb-vs-tdjepa.md #7)")
                    # OGBench: one relabel task per task_info. Single-task adapters (unified:
                    # MetaWorld / ManiSkill / Box2D) expose no task_infos and evaluate one
                    # task, so the loop runs once with tid=1 (their evaluate ignores it).
                    _task_infos = getattr(getattr(env, "unwrapped", env), "task_infos", None)
                    n_relabel_tasks = (len(_task_infos) if _task_infos is not None
                                       else len(adapter.eval_tasks(env, cfg)))
                    n_relabel_eps = int(cfg.get("eval_relabel_episodes", 10))
                    for tid in range(1, n_relabel_tasks + 1):
                        relabel_obs, relabel_rewards = relabel_fn(
                            train_ds, env, tid, cfg["eval_relabel_size"],
                            shift_reward=cfg.get("eval_relabel_shift", 1.0))
                        pre_cond = agent.cond_eval
                        relabel_agent = agent.infer_eval_cond(
                            jnp.asarray(relabel_obs), jnp.asarray(relabel_rewards))
                        # ROLLS_INFERRED_COND=True: the goal frame is never forwarded, so the
                        # inferred z from the relabelled rewards is what actually gets rolled.
                        relabel_agent = InferredCondAgent(relabel_agent)
                        # NO-OP GUARD. A critic whose infer_cond is a placeholder (density_fb
                        # with the index off returns the zero goal) would otherwise be rolled
                        # on goal=0 and SCORED here as a zero-shot result. Such a row is
                        # written with success=NaN and an invalid_reason, is not rolled, and
                        # never reaches the agg jsonl. Both columns are on EVERY row so the
                        # frozen CSV header can carry them.
                        invalid = route_b_invalid_reason(pre_cond, relabel_agent,
                                                         relabel_rewards)
                        if invalid is not None:
                            warnings.warn(
                                f"eval_relabel task {tid} at step {i}: reward inference had "
                                f"no effect ({invalid}); the row is written with success=NaN "
                                f"and invalid_reason set, and is NOT a zero-shot result.")
                            row = {"success": float("nan"), **relabel_agent.witness()}
                        else:
                            row, _frames = adapter.evaluate(
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
                        if FLAGS.agg_dir and invalid is None:
                            _append_agg(FLAGS.agg_dir, f"{agg_group}_relabel", FLAGS.seed, i,
                                        row, cfg.get("task", cfg.get("env_name")))
                        print("eval_relabel", i, "task", tid, row, flush=True)
            if i % FLAGS.save_interval == 0:
                save_agent(agent, run_dir, i)
                if online is not None:
                    # The other half of the checkpoint: without it --restore_path under
                    # --online cannot resume, and this run's own checkpoints would be the
                    # ones that fail the resume.
                    print(f"[online] saved {online.save_state(run_dir, i)}", flush=True)
            if _CHECKPOINT_SIGNAL["num"] is not None:
                # Signal-triggered checkpoint at an ARBITRARY step, then a clean exit so a
                # resubmission with --auto_resume picks it up. Skip the save if this very
                # step already wrote one above.
                if i % FLAGS.save_interval != 0:
                    save_agent(agent, run_dir, i)
                    if online is not None:
                        print(f"[online] saved {online.save_state(run_dir, i)}", flush=True)
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
