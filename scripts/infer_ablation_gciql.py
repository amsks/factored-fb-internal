#!/usr/bin/env python
"""scripts/infer_ablation_gciql.py — conditioning-axis ablation for GCIQL-bilinear (G3).

Question: is G3's win (vs FB/PSM) because it is handed the EXACT goal, or would
FB-style zero-shot inference also work on its value? This is the VALUE-level arm
(no actor rebuild): does the bilinear value V(s,g)=phi(s).psi(g)/sqrt(d) still
rank goal-reaching states when conditioned on an INFERRED task vector instead of
the exact goal embedding?

For one G3 checkpoint and task we collect states by rolling the policy (so the
batch has a natural mix of reaching / not-yet-reaching states), with the FAITHFUL
goal-reaching reward computed from the privileged cube qpos[14:17] (cube_reward_fn,
thresh 0.04 — the SAME reward the FB evaluator uses), not from the ambiguous obs
layout. Then, with the value net's phi/psi (GCBilinearValue info=True):

  z_exact    = psi(goal_obs)               # exact goal embedding (G3's eval path)
  z_inferred = sum_i r_i * psi(next_obs_i) # FB's L0 rule z = E[r.B], B := psi
  V_z(s)     = phi(s) . z                  # bilinear value (action-free value net)
  AUC = rank-AUC of V_z over r=1 vs r=0 states

AUC is invariant to positive scaling of z, so no sphere-projection ambiguity.
Read: AUC_exact high & AUC_inferred high  => value is inference-robust; G3's edge
is the OBJECTIVE, not exact-goal conditioning. AUC_inferred collapses to ~0.5 =>
exact-goal conditioning was doing the work (FB's inference would break it too).

Run: JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
       scripts/infer_ablation_gciql.py --checkpoint-dir <dir> --seed 0 \
       --value-latent-dim 512 --out analysis/infer_ablation_gciql/s0.json
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
CUBE_QPOS = slice(14, 17)   # privileged cube xyz (matches envs.ogbench.cube_reward_fn)
THRESH = 0.04


def value_auc(V: np.ndarray, positive: np.ndarray) -> float:
    """Rank-AUC of V for positive (r>0) vs negative states. 0.5=chance.
    Mirrors evals.z_inference.value_auc (Mann-Whitney U)."""
    V = V.reshape(-1)
    pos = positive.reshape(-1).astype(bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    vp, vn = V[pos], V[~pos]
    gt = (vp[:, None] > vn[None, :]).mean()
    eq = (vp[:, None] == vn[None, :]).mean()
    return float(gt + 0.5 * eq)


def collect_task(agent, env, task_goal_obs, goal_xyz, episodes, max_steps=200):
    """Roll the policy; return next_obs[N,28], cube_xyz[N,3] (privileged), r[N]."""
    import jax
    obss, cubes = [], []
    rng = jax.random.PRNGKey(0)
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        for _t in range(max_steps):
            rng, sub = jax.random.split(rng)
            a = np.asarray(agent.sample_actions(observations=np.asarray(obs, np.float32),
                                                goals=task_goal_obs, seed=sub, temperature=0.0))
            obs, _r, term, trunc, info = env.step(np.clip(a, -1, 1))
            obss.append(np.asarray(obs, np.float32))
            cubes.append(env.unwrapped.data.qpos[CUBE_QPOS].copy())
            if term or trunc:
                break
    obss = np.asarray(obss); cubes = np.asarray(cubes)
    r = (np.linalg.norm(cubes - goal_xyz[None, :], axis=-1) < THRESH).astype(np.float32)
    return obss, cubes, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=1000000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--value-latent-dim", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax.numpy as jnp
    import ogbench
    from phase_probe_gciql import load_agent

    agent = load_agent(Path(args.checkpoint_dir).resolve(), args.checkpoint_step,
                       args.seed, value_arch="bilinear", value_latent_dim=args.value_latent_dim)
    print(f"[infer_ablation] agent restored (d={args.value_latent_dim})")

    def phi_psi(obs, goal):
        """value net (GCBilinearValue, info=True) -> (phi(obs), psi(goal))."""
        _v, phi, psi = agent.network.select("value")(
            jnp.asarray(obs, jnp.float32), jnp.asarray(goal, jnp.float32), info=True)
        return np.asarray(phi), np.asarray(psi)

    rows = {}
    for task in EVAL_TASKS:
        env = ogbench.make_env_and_datasets(task, env_only=True)
        _obs, info = env.reset(seed=0)
        goal_obs = np.asarray(info["goal"], np.float32)                   # 28-dim goal observation
        goal_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        nobs, _cube, r = collect_task(agent, env, goal_obs, goal_xyz, args.episodes)
        if hasattr(env, "close"):
            env.close()

        phi_s, _ = phi_psi(nobs, nobs)            # phi(next_obs)        [N,d]
        _, psi_s = phi_psi(nobs, nobs)            # psi(next_obs) = B    [N,d]
        _, psi_g = phi_psi(nobs[:1], goal_obs[None])  # psi(goal_obs)    [1,d]
        z_exact = psi_g[0]
        z_inf = (r[:, None] * psi_s).sum(0)       # L0: z = E[r.B]
        V_exact = phi_s @ z_exact
        V_inf = phi_s @ z_inf
        auc_e, auc_i = value_auc(V_exact, r), value_auc(V_inf, r)
        rows[task] = {"n": int(len(r)), "r_pos": float(r.mean()),
                      "auc_exact": auc_e, "auc_inferred": auc_i}
        print(f"  {task}: n={len(r)} r_pos={r.mean():.3f}  "
              f"AUC exact={auc_e:.3f}  inferred={auc_i:.3f}")

    finite = [v for v in rows.values() if v["auc_exact"] == v["auc_exact"]]
    summ = {
        "seed": args.seed, "value_latent_dim": args.value_latent_dim,
        "mean_auc_exact": float(np.mean([v["auc_exact"] for v in finite])),
        "mean_auc_inferred": float(np.mean([v["auc_inferred"] for v in finite])),
        "per_task": rows,
    }
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summ, indent=2))
    print(f"[infer_ablation] mean AUC  exact={summ['mean_auc_exact']:.3f}  "
          f"inferred={summ['mean_auc_inferred']:.3f}  -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
