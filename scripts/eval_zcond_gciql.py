#!/usr/bin/env python
"""scripts/eval_zcond_gciql.py — G4 success-level objective isolation.

Rolls a trained z-conditioned GCIQL-Flow checkpoint (actor_cond='z',
value_arch='bilinear') two ways per task and reports SUCCESS:

  exact-z     : z = value.psi(goal_obs)          (G3-analog with z-conditioning)
  inferred-z  : z = sum_i r_i * value.psi(s'_i)  (FB's L0 rule, full FB pipeline)

The inference relabel batch is gathered by rolling the exact-z policy and scoring
states with the FAITHFUL goal-reaching reward from privileged cube qpos[14:17]
(dist<0.04 -- the same reward the FB evaluator and infer_ablation_gciql use; the
obs cube xy has unrecoverable scale/offset, so privileged qpos is required and
the offline dataset, which lacks qpos, cannot supply it). Same-policy state dist
caveat noted in the spec.

Run: JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
       scripts/eval_zcond_gciql.py --checkpoint-dir <dir> --checkpoint-step 500000 \
       --seed 0 --value-latent-dim 512 --episodes 50 \
       --out analysis/eval_zcond_gciql/s0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))

EVAL_TASKS = [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)]
CUBE_QPOS = slice(14, 17)   # privileged cube xyz (matches infer_ablation_gciql)
THRESH = 0.04


def rollout(agent, env, cond, goal_xyz, episodes, mode, max_steps=200, seed0=0):
    """Roll the z-cond actor. mode='exact' => cond is the RAW goal obs and the
    actor maps it through value.psi internally (sample_actions); mode='inferred'
    => cond is a PRECOMPUTED d-dim z fed straight to the actor
    (sample_actions_with_z). Return per-step next_obs, cube_xyz, and per-EPISODE
    success (cube within THRESH of goal at any step)."""
    import jax
    cond = np.asarray(cond, np.float32)
    rng = jax.random.PRNGKey(seed0)
    obss, cubes, succ = [], [], []
    for ep in range(episodes):
        obs, _info = env.reset(seed=seed0 + ep)
        ep_hit = False
        for _t in range(max_steps):
            rng, sub = jax.random.split(rng)
            o = np.asarray(obs, np.float32)
            if mode == "exact":
                a = np.asarray(agent.sample_actions(observations=o, goals=cond,
                                                    seed=sub, temperature=0.0))
            else:
                a = np.asarray(agent.sample_actions_with_z(observations=o, z=cond,
                                                           seed=sub, temperature=0.0))
            obs, _r, term, trunc, _info = env.step(np.clip(a, -1, 1))
            cube = env.unwrapped.data.qpos[CUBE_QPOS].copy()
            obss.append(np.asarray(obs, np.float32)); cubes.append(cube)
            if np.linalg.norm(cube - goal_xyz) < THRESH:
                ep_hit = True
            if term or trunc:
                break
        succ.append(float(ep_hit))
    return np.asarray(obss), np.asarray(cubes), float(np.mean(succ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--value-latent-dim", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--infer-episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax.numpy as jnp
    import ogbench
    from phase_probe_gciql import load_agent

    agent = load_agent(Path(args.checkpoint_dir).resolve(), args.checkpoint_step,
                       args.seed, value_arch="bilinear",
                       value_latent_dim=args.value_latent_dim, actor_cond="z")
    print(f"[eval_zcond] agent restored (d={args.value_latent_dim}, actor_cond=z)")

    def psi(goal):
        """B := value net's psi(goal). Returns (d,) for a single goal / (N,d)."""
        g = jnp.asarray(goal, jnp.float32)
        _v, _phi, ps = agent.network.select("value")(g, g, info=True)
        return np.asarray(ps)

    rows = {}
    for task in EVAL_TASKS:
        env = ogbench.make_env_and_datasets(task, env_only=True)
        _obs, info = env.reset(seed=0)
        goal_obs = np.asarray(info["goal"], np.float32)
        goal_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]

        # exact-z: actor maps the raw goal through value.psi internally.
        _nobs, _cube, succ_exact = rollout(
            agent, env, goal_obs, goal_xyz, args.episodes, mode="exact")

        # inferred-z: gather a fresh relabel batch (states + privileged cube xyz)
        # under the exact-z policy, score with the faithful reward, z = sum r*psi(s'),
        # then roll the actor on that precomputed z.
        rb_obs, rb_cube, _ = rollout(
            agent, env, goal_obs, goal_xyz, args.infer_episodes, mode="exact")
        rb_r = (np.linalg.norm(rb_cube - goal_xyz[None, :], axis=-1) < THRESH).astype(np.float32)
        z_inf = (rb_r[:, None] * psi(rb_obs)).sum(0)        # L0: z = E[r.B]
        _nobs2, _cube2, succ_inf = rollout(
            agent, env, z_inf, goal_xyz, args.episodes, mode="inferred")

        if hasattr(env, "close"):
            env.close()
        rows[task] = {"success_exact": succ_exact, "success_inferred": succ_inf,
                      "r_pos_relabel": float(rb_r.mean())}
        print(f"  {task}: success exact={succ_exact:.3f}  inferred={succ_inf:.3f}  "
              f"(relabel r_pos={rb_r.mean():.3f})")

    summ = {
        "seed": args.seed, "value_latent_dim": args.value_latent_dim,
        "mean_success_exact": float(np.mean([v["success_exact"] for v in rows.values()])),
        "mean_success_inferred": float(np.mean([v["success_inferred"] for v in rows.values()])),
        "per_task": rows,
    }
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summ, indent=2))
    print(f"[eval_zcond] mean success  exact={summ['mean_success_exact']:.3f}  "
          f"inferred={summ['mean_success_inferred']:.3f}  -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
