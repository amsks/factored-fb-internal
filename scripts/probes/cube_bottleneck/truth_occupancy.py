"""Ground truth for the sharpness figure: where the cube ACTUALLY ends up.

fig_sharpness_map.py shows that Density-FB's predicted successor distribution is ~13.6x
more concentrated at the goal than the dataset prior, while FB is 1.4x. That is a claim
about SHARPNESS. It is not a claim about correctness, and it cannot be: the prediction is
conditioned on the goal, so a critic that merely echoed "the cube ends up where you asked"
would draw the identical panel.

This module measures the thing that settles it -- the empirical discounted occupancy of the
cube, by rollout, from the same pre-grasp states:

    mu(u) proportional to sum_t gamma^t * 1[cube at u]

which is exactly the object M(s,a,.) and w(s,a,.) estimate. Binned into the same table
frame, it becomes a fourth panel and the predictions become checkable.

POLICY CONDITIONING -- the trap this file exists to avoid. The occupancy measure is
policy-dependent: M^pi. FB's measure estimates FB's policy's occupancy, Density-FB's
estimates its own. There is therefore NO single ground truth; rolling one policy and
scoring both against it measures the wrong thing for one of them, which is precisely the
error the 2026-09-04 cube-bottleneck audit caught ("FB discriminates securing actions
better" was a policy-conditioning artefact). So every arm is rolled under ITS OWN actor and
the output is labelled with the arm that produced it. The comparison is within-arm
calibration: does each method predict what its own policy actually does?

Episodes are stepped in LOCKSTEP over an env pool (measure_probe.make_env_pool's trick):
the flow actor costs about the same for a batch as for a single row, so a per-episode loop
would pay n_episodes times more actor calls for the same rollouts.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import numpy as np

from scripts.probes.cube_bottleneck import agents as A
from scripts.probes.cube_bottleneck import restore as R
from scripts.probes.cube_bottleneck.fig_outcome_map import CM, task_goals

ANALYSIS = Path("analysis/cube_bottleneck")
CUBE_OBS = slice(19, 22)


def rollout_occupancy(arm, qpos, qvel, goals, n_episodes, horizon, discount):
    """[T, n_episodes, 2] cube xy per step, and the matching gamma^t weights.

    Each episode is conditioned on one of the `goals` draws, cycling, so the rollouts
    marginalise the goal's nuisance arm pose exactly the way the prediction does.
    """
    import jax

    from impls.utils.evaluation import supply_rng
    from scripts.probes.cube_bottleneck.measure_probe import make_env_pool

    actor_fn = supply_rng(arm.agent.sample_actions, rng=jax.random.PRNGKey(0))
    envs = getattr(rollout_occupancy, "_pool", None)
    if envs is None or len(envs) < n_episodes:
        envs = make_env_pool(arm.env_name, n_episodes)
        rollout_occupancy._pool = envs

    goal_b = np.stack([goals[k % len(goals)] for k in range(n_episodes)]).astype(np.float32)
    obs = []
    for k in range(n_episodes):
        e = envs[k]
        e.reset(options=dict(task_id=1, render_goal=False))
        R.set_state(e, qpos, qvel)
        R.settle(e)
        obs.append(R.observation(e))
    obs = np.stack(obs).astype(np.float32)

    xy = np.zeros((horizon, n_episodes, 2), np.float32)
    alive = np.ones(n_episodes, bool)
    for t in range(horizon):
        acts = np.clip(np.asarray(actor_fn(observations=obs, goals=goal_b,
                                           temperature=0.0)), -1, 1)
        nxt = []
        for k in range(n_episodes):
            if alive[k]:
                o, _r, term, trunc, _i = envs[k].step(np.asarray(acts[k], np.float32))
                # OGBench ends an episode the instant success fires; a finished rollout
                # keeps occupying its final state rather than dropping out of the average,
                # which is what the discounted occupancy of an absorbing success means.
                if term or trunc:
                    alive[k] = False
                nxt.append(np.asarray(o, np.float32))
            else:
                nxt.append(obs[k])
        obs = np.stack(nxt)
        xy[t] = obs[:, CUBE_OBS][:, :2]
    w = discount ** np.arange(horizon, dtype=np.float64)
    return xy, w / w.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cube-single-play-v0")
    ap.add_argument("--critics", nargs="+", default=["fb_ref", "density_fb_state"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--phase", default="P1")
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--goal-draws", type=int, default=8)
    ap.add_argument("--goal-seed", type=int, default=20260904)
    args = ap.parse_args()

    root = ANALYSIS / args.env
    banks = dict(np.load(root / "phase_banks.npz"))
    out_dir = root / "truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    for a in args.critics:
        for s in args.seeds:
            run = Path("exp") / f"{args.env}_{a}_flowbc_seed_{s}"
            if not (run / "params_500000.pkl").exists():
                print(f"[skip] no checkpoint {run}")
                continue
            out = out_dir / f"truth_{args.phase}_{a}_s{s}.npz"
            if out.exists():
                print(f"[skip] {out.name} exists")
                continue
            arm = A.load_arm(run)
            disc = float(arm.config.get("discount", 0.99))
            goals = task_goals(arm, args.task, args.goal_draws, args.goal_seed)
            n_states = min(args.states, len(banks[f"{args.phase}_qpos"]))
            XY = np.zeros((n_states, args.horizon, args.episodes, 2), np.float32)
            for i in range(n_states):
                xy, w = rollout_occupancy(arm, banks[f"{args.phase}_qpos"][i],
                                          banks[f"{args.phase}_qvel"][i], goals,
                                          args.episodes, args.horizon, disc)
                XY[i] = xy
                if (i + 1) % 4 == 0:
                    print(f"  [{a} s{s}] {i + 1}/{n_states} states", flush=True)
            np.savez(out, cube_xy=XY, weights=w, critic=np.asarray(a), seed=np.asarray(s),
                     discount=np.asarray(disc), phase=np.asarray(args.phase),
                     goal_cube_xy=np.mean([g[CUBE_OBS][:2] for g in goals], axis=0),
                     state_cube_xy=banks[f"{args.phase}_obs"][:n_states][:, CUBE_OBS][:, :2])
            # Headline: did the cube reach the goal, and did it move at all?
            gxy = np.mean([g[CUBE_OBS][:2] for g in goals], axis=0)
            d = np.linalg.norm(XY - gxy, axis=-1)                     # [S, T, E]
            occ_at_goal = float((w[None, :, None] * (d < 0.4)).sum(1).mean())
            moved = float((np.linalg.norm(XY[:, -1] - XY[:, 0], axis=-1) > 0.3).mean())
            print(f"[{a} s{s}] discounted occupancy within 4cm of goal {occ_at_goal:.4f} | "
                  f"cube moved >3cm in {100 * moved:.0f}% of episodes -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
