#!/usr/bin/env python
"""scripts/eval_zeroshot_inference.py — E1a: zero-shot GCIQL inference sweep.

Unifies + extends eval_zcond_gciql.py / infer_ablation_gciql.py into the THREE
task-inference rules from the zero-shot spec
(docs/superpowers/specs/2026-07-06-zeroshot-and-interpolation-design.md), evaluated
at the ACTOR / SUCCESS level (roll the z-actor and measure goal-reaching success,
not just value rank-AUC):

  v0 (moment)   z = E_rho[r . psi(s')]                    -> must reproduce ~0.07 (baseline)
  v1 (whitened) z = (E[psi psi^T] + eps I)^-1 E[r . psi]  -> the fix candidate (GLS / least-sq)
  v2 (exact)    z = psi(g)                                -> must reproduce ~0.93 (upper anchor)

psi := the VALUE net's backward map B (GCBilinearValue, info=True) -- the same B the
z-actor conditions on (actor_cond='z' maps the raw goal through value.psi internally;
_value_psi in gciql_flow.py). eps = ridge * trace(E[psi psi^T]) / d (spec: ~1e-3).

REWARD SAMPLES = the offline dataset (spec-faithful "over dataset states"). The
`singletask` OGBench dataset carries the relabeled indicator reward in
`train_dataset['rewards']` (relabel_dataset: rewards = success - 1, computed from the
privileged cube qpos at load time), so r = rewards + 1 is the exact FB-evaluator
indicator over *dataset* states -- no on-policy gather, no state-distribution caveat.
A random subsample of `--relabel-size` states is embedded through psi.

NORMALIZATION (why two columns): the actor was trained on RAW psi(g). The value net
sphere-projects psi to sqrt(d) only when project_z=True, which G4-zcond is NOT, so
||psi(g)|| is task-dependent and generally != sqrt(d). Rescaling an inferred z to
sqrt(d) (the spec's choice) therefore need not match the scale the actor consumed at
exact-z eval. We report each inferred z under BOTH norms:
  'match_g' : rescaled to ||psi(g)|| (the faithful eval scale; makes v2==exact)
  'sqrtd'   : sphere-projected to sqrt(d) (the spec's choice)
cos(z, psi(g)) is scale-invariant and is the load-bearing diagnostic either way.

Decision rule (spec): v1 >= 0.6 -> the inference FORMULA was the dominant break, E1b
becomes polish. v1 ~ v0 ~ 0.07 with cos ~ 1 -> pure actor conditioning-distribution
shift, E1b is the load-bearing fix. cos low for v0 but high for v1 -> whitening
story confirmed.

Run: JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
       scripts/eval_zeroshot_inference.py --checkpoint-dir <dir> \
       --checkpoint-step 1000000 --seed 3 --value-latent-dim 512 --episodes 25 \
       --relabel-size 50000 --out analysis/eval_zeroshot_inference/s3.json
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
CUBE_QPOS = slice(14, 17)   # privileged cube xyz (matches eval_zcond / infer_ablation)
THRESH = 0.04


def rollout(agent, env, cond, goal_xyz, episodes, mode, max_steps=200, seed0=0):
    """Roll the z-cond actor. mode='exact' => cond is the RAW goal obs, the actor maps
    it through value.psi internally (sample_actions). mode='inferred' => cond is a
    PRECOMPUTED d-dim z fed straight to the actor (sample_actions_with_z). Returns
    mean per-EPISODE success (cube within THRESH of goal at any step)."""
    import jax
    cond = np.asarray(cond, np.float32)
    rng = jax.random.PRNGKey(seed0)
    succ = []
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
            if np.linalg.norm(cube - goal_xyz) < THRESH:
                ep_hit = True
            if term or trunc:
                break
        succ.append(float(ep_hit))
    return float(np.mean(succ))


def infer_z(Psi: np.ndarray, r: np.ndarray, ridge: float):
    """Return (z_v0, z_v1) in float64. Psi [N,d] = psi(states); r [N] indicator reward.
      v0 moment   : z = sum_i r_i psi_i            (= N * E[r psi], scale absorbed by norm)
      v1 whitened : z = (G + eps I)^-1 E[r psi], G = E[psi psi^T], eps = ridge*tr(G)/d
    v1 is the generalized-least-squares / dual vector correcting for a non-orthonormal
    psi (regression of r onto psi), which the plain moment v0 ignores."""
    Psi = np.asarray(Psi, np.float64)
    r = np.asarray(r, np.float64).reshape(-1)
    N, d = Psi.shape
    b = (r[:, None] * Psi).sum(0)                 # r-moment  (v0, unnormalized)
    G = (Psi.T @ Psi) / N                         # E[psi psi^T]  [d,d]
    eps = ridge * (np.trace(G) / d)
    z_v1 = np.linalg.solve(G + eps * np.eye(d), b / N)
    return b, z_v1


def rescale(z: np.ndarray, target_norm: float) -> np.ndarray:
    return z * target_norm / (np.linalg.norm(z) + 1e-8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=1000000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--value-latent-dim", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=25, help="rollout episodes per (task, z)")
    ap.add_argument("--relabel-size", type=int, default=50000,
                    help="dataset states subsampled to infer z (must exceed d for a stable v1)")
    ap.add_argument("--ridge", type=float, default=1e-3, help="v1 ridge as fraction of tr(G)/d")
    ap.add_argument("--tasks", default="", help="comma-sep task indices 1..5 (default all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax.numpy as jnp
    import ogbench
    from phase_probe_gciql import load_agent

    tasks = EVAL_TASKS if not args.tasks else \
        [f"cube-single-play-singletask-task{i}-v0" for i in args.tasks.split(",")]

    agent = load_agent(Path(args.checkpoint_dir).resolve(), args.checkpoint_step,
                       args.seed, value_arch="bilinear",
                       value_latent_dim=args.value_latent_dim, actor_cond="z")
    d = args.value_latent_dim
    sqrtd = float(np.sqrt(d))
    print(f"[zeroshot] agent restored (d={d}, actor_cond=z, step={args.checkpoint_step})")

    def psi(obs, chunk=8192):
        """B := value net's psi(obs). obs [.,obs_dim] -> [.,d], chunked to bound memory."""
        obs = np.asarray(obs, np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None]
        out = []
        for i in range(0, len(obs), chunk):
            g = jnp.asarray(obs[i:i + chunk])
            _v, _phi, ps = agent.network.select("value")(g, g, info=True)
            out.append(np.asarray(ps))
        out = np.concatenate(out, 0)
        return out[0] if single else out

    rng = np.random.default_rng(args.seed)
    rows = {}
    for task in tasks:
        env, train_ds, _val = ogbench.make_env_and_datasets(task)
        S_all = np.asarray(train_ds["observations"], np.float32)
        r_all = np.asarray(train_ds["rewards"], np.float64) + 1.0     # indicator 0/1
        m = min(args.relabel_size, len(S_all))
        idx = rng.choice(len(S_all), size=m, replace=False)
        S, r = S_all[idx], r_all[idx]

        _obs, info = env.reset(seed=0)
        goal_obs = np.asarray(info["goal"], np.float32)
        goal_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"]).reshape(-1)[:3]
        psi_g = psi(goal_obs)                     # [d]
        norm_g = float(np.linalg.norm(psi_g))

        succ_exact = rollout(agent, env, goal_obs, goal_xyz, args.episodes, "exact")
        rec = {"n_samples": int(m), "r_pos": float(r.mean()),
               "psi_g_norm": norm_g, "sqrt_d": sqrtd,
               "success_v2_exact": succ_exact, "z_rules": {}}

        if r.sum() == 0:
            print(f"  {task}: r_pos=0 in subsample -> inference undefined; exact={succ_exact:.3f}")
            rows[task] = rec
            if hasattr(env, "close"):
                env.close()
            continue

        z_v0, z_v1 = infer_z(psi(S), r, args.ridge)
        for name, z in (("v0_moment", z_v0), ("v1_whitened", z_v1)):
            entry = {"cos_to_psi_g": cosine(z, psi_g), "raw_norm": float(np.linalg.norm(z))}
            for norm_mode, tgt in (("match_g", norm_g), ("sqrtd", sqrtd)):
                entry[f"success_{norm_mode}"] = rollout(
                    agent, env, rescale(z, tgt).astype(np.float32), goal_xyz,
                    args.episodes, "inferred")
            rec["z_rules"][name] = entry
            print(f"  {task} {name}: cos={entry['cos_to_psi_g']:.3f}  "
                  f"succ match_g={entry['success_match_g']:.3f} sqrtd={entry['success_sqrtd']:.3f}")

        print(f"  {task}: exact(v2)={succ_exact:.3f}  r_pos={r.mean():.4f}  "
              f"||psi_g||={norm_g:.2f} (sqrt_d={sqrtd:.2f})")
        rows[task] = rec
        if hasattr(env, "close"):
            env.close()

    def mean_over(getter):
        vals = [getter(v) for v in rows.values() if getter(v) is not None]
        return float(np.mean(vals)) if vals else None

    def rule(rulename, key):
        return lambda v: v["z_rules"].get(rulename, {}).get(key) if v.get("z_rules") else None

    summ = {
        "seed": args.seed, "checkpoint_step": args.checkpoint_step,
        "value_latent_dim": d, "ridge": args.ridge, "episodes": args.episodes,
        "relabel_size": args.relabel_size,
        "mean_success_v2_exact": mean_over(lambda v: v.get("success_v2_exact")),
        "mean_success_v0_match_g": mean_over(rule("v0_moment", "success_match_g")),
        "mean_success_v0_sqrtd": mean_over(rule("v0_moment", "success_sqrtd")),
        "mean_success_v1_match_g": mean_over(rule("v1_whitened", "success_match_g")),
        "mean_success_v1_sqrtd": mean_over(rule("v1_whitened", "success_sqrtd")),
        "mean_cos_v0": mean_over(rule("v0_moment", "cos_to_psi_g")),
        "mean_cos_v1": mean_over(rule("v1_whitened", "cos_to_psi_g")),
        "per_task": rows,
    }
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summ, indent=2))
    print(f"[zeroshot] mean success  v2_exact={summ['mean_success_v2_exact']}  "
          f"v0={summ['mean_success_v0_match_g']}/{summ['mean_success_v0_sqrtd']} (match_g/sqrtd)  "
          f"v1={summ['mean_success_v1_match_g']}/{summ['mean_success_v1_sqrtd']}  "
          f"cos v0={summ['mean_cos_v0']} v1={summ['mean_cos_v1']}  -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
