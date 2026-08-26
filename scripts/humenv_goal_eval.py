"""scripts/humenv_goal_eval.py — score a checkpoint on HumEnv's goal-reaching benchmark.

    # our jax density_fb checkpoint (run in .venv-jax-fb)
    PYTHONPATH=$PWD $SCRATCH/.venv-jax-fb/bin/python scripts/humenv_goal_eval.py \
        --agent jax --critic density_fb --actor ddpgbc_density --domain hum-humanoid \
        --restore_path <run dir> --restore_epoch 500000 \
        --override 'measure_hit_spec="0:214:2.548"' --override measure_density_z=true \
        --override measure_density_z_eval_pool_m=1024 --override measure_density_dim=128 \
        --goals buffer --num_goals 8 --episodes_per_goal 5

    # the released FB-CPR model as an ORACLE (run in .venv-humenv, which has torch)
    PYTHONPATH=$PWD $SCRATCH/.venv-humenv/bin/python scripts/humenv_goal_eval.py \
        --agent motivo --goals buffer --num_goals 8 --episodes_per_goal 5

TWO BACKENDS, ONE ROLLOUT AND ONE METRIC. The heavy imports live inside each branch, so the
jax branch runs in the venv that has jax and the motivo branch in the venv that has torch,
while `rollout_goal` and `eval.humenv_goal` are shared. That is what makes the oracle
comparable to us: identical env construction, identical episode budget, identical metric.

THE ORACLE IS NOT A BASELINE (spec 23 §6.4). FB-CPR is online-trained on 30M interactive
steps with AMASS regularisation; we train offline on its replay. Its number is the
distillation ceiling, not a competitor's score.

CONDITIONING FIDELITY. The jax branch conditions exactly the way training-time eval does --
`agent.sample_actions(obs, goals=goal)`, letting the critic's `map_eval_cond` turn the goal
into its own index (for density_fb: M reward samples over the frozen pool, then the centered
solve). It does NOT shortcut to some other goal->z path, because then this script and the
run's own eval.csv would be measuring different agents.
"""
import argparse
import glob
import json
import os

import numpy as np

from eval.humenv_goal import aggregate, episode_goal_stats


def load_goals(args):
    """-> {name: goal_observation}. `--goals json` is the real benchmark set once the AMASS
    chain has produced goal_poses.json; `--goals buffer` takes final observations from the
    converted replay so the pipeline is exercisable before that exists. The two are NOT
    interchangeable as results -- buffer goals are states the behaviour policy actually
    reached, which is an easier set than the benchmark's curated poses."""
    if args.goals == "json":
        with open(args.goals_json) as f:
            raw = json.load(f)
        goals = {k: np.asarray(v, np.float32) for k, v in raw.items()}
    else:
        root = args.buffer_root or os.path.join(os.environ["SCRATCH"], "fb-data/humenv-S1")
        eps = sorted(glob.glob(os.path.join(root, "buffer", "episode_*.npz")))
        if not eps:
            raise FileNotFoundError(f"no episodes under {root}/buffer")
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(eps), size=min(args.num_goals, len(eps)), replace=False)
        goals = {}
        for i in pick:
            d = np.load(eps[i])
            goals[os.path.basename(eps[i])] = d["observation"][-1].astype(np.float32)
    names = list(goals)[: args.num_goals]
    return {k: goals[k] for k in names}


def make_env(seed):
    import gymnasium
    from humenv import make_humenv
    # FlattenObservation and state_init="Fall" are GoalEvaluation's OWN settings: the
    # humanoid starts fallen and must get up into the goal pose. Deviating here would make
    # our numbers incomparable to the published ones for a reason invisible in the output.
    env, _ = make_humenv(num_envs=1, state_init="Fall",
                         wrappers=[gymnasium.wrappers.FlattenObservation])
    return env


def rollout_goal(env, act_fn, goal, n_episodes, horizon, seed0, frames_out=None):
    """-> [per-episode stats]. `act_fn(obs) -> action`; the goal is already bound into it."""
    out = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        obs = np.asarray(obs, np.float32).reshape(-1)
        seq = []
        for t in range(horizon):
            obs, _r, term, trunc, _ = env.step(act_fn(obs))
            obs = np.asarray(obs, np.float32).reshape(-1)
            seq.append(obs)
            if frames_out is not None and ep == 0:
                frames_out.append(env.render())
            if term or trunc:
                break
        out.append(episode_goal_stats(np.asarray(seq, np.float32), goal))
    return out


def jax_agent(args):
    """-> (act_for_goal, horizon). Conditioning goes through the SAME call the run's own
    eval uses, so this script cannot silently score a differently-conditioned agent."""
    import jax
    import jax.numpy as jnp
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset
    from impls.utils.flax_utils import restore_agent
    from env import get_adapter, make_config

    cfg = make_config(args.critic, args.actor, args.domain, args.override)
    np.random.seed(args.seed)
    adapter = get_adapter(cfg)
    _env, train_ds, _val = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(args.critic, args.actor, args.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    key = jax.random.PRNGKey(args.seed)

    def for_goal(goal):
        g = jnp.asarray(goal)
        def act(obs):
            nonlocal key
            key, sub = jax.random.split(key)
            a = agent.sample_actions(observations=jnp.asarray(obs), goals=g,
                                     seed=sub, temperature=0.0)
            return np.asarray(a, np.float32)
        return act
    return for_goal


def motivo_agent(args):
    import torch
    from metamotivo.fb_cpr.huggingface import FBcprModel
    model = FBcprModel.from_pretrained(args.motivo_model)
    model.eval()

    def for_goal(goal):
        # goal_inference(next_obs) is POSITIONAL and returns B(goal) on the sqrt(d) sphere.
        z = model.goal_inference(torch.tensor(goal, dtype=torch.float32).unsqueeze(0))
        def act(obs):
            with torch.no_grad():
                a = model.act(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), z, mean=True)
            return a.squeeze(0).cpu().numpy().astype(np.float32)
        return act
    return for_goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["jax", "motivo"], required=True)
    ap.add_argument("--critic", default="density_fb")
    ap.add_argument("--actor", default="ddpgbc_density")
    ap.add_argument("--domain", default="hum-humanoid")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--restore_path", default="")
    ap.add_argument("--restore_epoch", type=int, default=500000)
    ap.add_argument("--motivo_model", default="facebook/metamotivo-S-1")
    ap.add_argument("--goals", choices=["buffer", "json"], default="buffer")
    ap.add_argument("--goals_json", default="")
    ap.add_argument("--buffer_root", default="")
    ap.add_argument("--num_goals", type=int, default=8)
    ap.add_argument("--episodes_per_goal", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=300)   # HumEnv: 30Hz x 300 steps
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_jsonl", default="")
    ap.add_argument("--video_dir", default="")
    args = ap.parse_args()

    goals = load_goals(args)
    env = make_env(args.seed)
    for_goal = jax_agent(args) if args.agent == "jax" else motivo_agent(args)
    print(f"[humenv-goal] agent={args.agent} goals={len(goals)} "
          f"episodes/goal={args.episodes_per_goal} horizon={args.horizon}", flush=True)

    per_goal, all_eps = {}, []
    for gi, (name, goal) in enumerate(sorted(goals.items())):
        frames = [] if args.video_dir else None
        stats = rollout_goal(env, for_goal(goal), goal, args.episodes_per_goal,
                             args.horizon, seed0=1000 * (gi + 1) + args.seed,
                             frames_out=frames)
        per_goal[name] = aggregate(stats)
        all_eps += stats
        print(f"  {name:34s} success={per_goal[name]['success']:.3f} "
              f"proximity={per_goal[name]['proximity']:.3f} "
              f"distance={per_goal[name]['distance']:.2f}", flush=True)
        if frames:
            os.makedirs(args.video_dir, exist_ok=True)
            _write_mp4(os.path.join(args.video_dir, f"{args.agent}-{gi}.mp4"), frames)

    overall = aggregate(all_eps)
    print(f"[humenv-goal] OVERALL success={overall['success']:.3f} "
          f"proximity={overall['proximity']:.3f} distance={overall['distance']:.2f} "
          f"(n={overall['n_episodes']} episodes)", flush=True)
    if args.out_jsonl:
        with open(args.out_jsonl, "a") as f:
            f.write(json.dumps({"agent": args.agent, "restore_path": args.restore_path,
                                "restore_epoch": args.restore_epoch, "goals": args.goals,
                                "overall": overall, "per_goal": per_goal}) + "\n")


def _write_mp4(path, frames, fps=30):
    """The gh nodes' /usr/bin/ffmpeg is broken (missing libunwind.so.8); imageio-ffmpeg's
    static binary in the jax venv works everywhere."""
    import subprocess
    ff = os.environ.get("FFMPEG_BIN", os.environ["SCRATCH"] +
                        "/.venv-jax-fb/lib/python3.11/site-packages/imageio_ffmpeg/"
                        "binaries/ffmpeg-linux-aarch64-v7.0.2")
    h, w, _ = np.asarray(frames[0]).shape
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an",
                          "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
                         stdin=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() == 0:
        print(f"  WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
