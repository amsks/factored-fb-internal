#!/usr/bin/env python
"""scripts/e2_measure_psi_probe.py — E2 prize cross-eval (value-level, option (a)).

Question: did the scale-controlled successor-measure aux loss (MeasureMix cells,
measure_legs='critic') change the CRITIC's backward-map geometry psi(g) in a way
that helps FB-style zero-shot task inference — even though rollout success stayed
flat at ~0.80–0.93 vs the 0.90 plain-IQL base?

Protocol (E1a-faithful, no z-actor needed — E2 cells trained actor_cond='goal'):
  r      = train_dataset['rewards'] + 1  (relabeled indicator over DATASET states;
           the exact FB-evaluator reward, no on-policy sampling caveat — the E1a fix)
  z_v0   = sum_i r_i . psi(s_i)          (dataset moment z = E[r.psi], E1a's winning rule)
  cos    = cos(z_v0, psi(goal_obs))      (scale-free geometry diagnostic)
  AUC    = rank-AUC over r=1 vs r=0 dataset states of the bilinear readout
             critic legs: Q_z(s,a) = mean_heads phi(s,a) . z   (a = dataset action)
             value  legs: V_z(s)   = phi(s) . z
           for z in {psi(g) exact, z_v0 inferred}.

LEGS: the measure trained the CRITIC's phi/psi (measure_legs='critic'); the value
net feels it only indirectly (via the Q-target). We report BOTH: critic = the
treated leg, value = control. Comparison = 5 MeasureMix cells vs the plain
GCIQL-State-Flow-Bilinear-d50 base, all @500k, same protocol.

Run (one invocation, all checkpoints; datasets load once per task):
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
    scripts/e2_measure_psi_probe.py --checkpoint-dirs <dir1> <dir2> ... \
    --checkpoint-step 500000 --value-latent-dim 50 --out analysis/e2_psi_probe/all.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "scripts"),
          str(REPO / "third_party" / "ogbench" / "impls"),
          str(REPO / "tools" / "wandb_mode_shim")):
    if p not in sys.path:
        sys.path.insert(0, p)

from infer_ablation_gciql import value_auc            # Mann-Whitney rank-AUC
from eval_zeroshot_inference import cosine            # scale-free cos(a,b)

EVAL_TASKS = [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)]


def phi_psi(agent, legs, obs, goals, actions, chunk=8192):
    """(phi, psi) for one leg family, chunked. critic outputs are ensembled
    [2,B,d] -> head-mean (mirrors gciql_flow._critic_psi)."""
    import jax.numpy as jnp
    phis, psis = [], []
    for i in range(0, len(obs), chunk):
        o = jnp.asarray(obs[i:i + chunk], jnp.float32)
        g = jnp.asarray(goals[i:i + chunk], jnp.float32)
        if legs == "value":
            _v, ph, ps = agent.network.select("value")(o, g, info=True)
        else:
            a = jnp.asarray(actions[i:i + chunk], jnp.float32)
            _q, ph, ps = agent.network.select("critic")(o, g, a, info=True)
            ph, ps = ph.mean(axis=0), ps.mean(axis=0)
        phis.append(np.asarray(ph)); psis.append(np.asarray(ps))
    return np.concatenate(phis, 0), np.concatenate(psis, 0)


def probe_agent_task(agent, S, A, r, goal_obs):
    """One (agent, task): per-leg {cos_v0, auc_exact, auc_v0, psi_g_norm}."""
    out = {}
    g1 = goal_obs[None]
    a1 = np.zeros((1, A.shape[-1]), np.float32)
    for legs in ("critic", "value"):
        phi_s, psi_s = phi_psi(agent, legs, S, S, A)          # phi(s[,a]), psi(s)
        _, psi_g = phi_psi(agent, legs, g1, g1, a1)
        psi_g = psi_g[0]
        z_v0 = (r[:, None] * psi_s).sum(0)
        out[legs] = {
            "psi_g_norm": float(np.linalg.norm(psi_g)),
            "cos_v0": cosine(z_v0, psi_g),
            "auc_exact": value_auc(phi_s @ psi_g, r),
            "auc_v0": value_auc(phi_s @ z_v0, r),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dirs", nargs="+", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--value-latent-dim", type=int, default=50)
    ap.add_argument("--relabel-size", type=int, default=50000)
    ap.add_argument("--tasks", default="", help="comma-sep task indices 1..5 (default all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import ogbench
    from phase_probe_gciql import load_agent

    tasks = EVAL_TASKS if not args.tasks else \
        [f"cube-single-play-singletask-task{i}-v0" for i in args.tasks.split(",")]

    agents = {}
    for cd in args.checkpoint_dirs:
        cd = Path(cd).resolve()
        label = f"{cd.parent.name}/{cd.name}"
        agents[label] = load_agent(cd, args.checkpoint_step, 0,
                                   value_arch="bilinear",
                                   value_latent_dim=args.value_latent_dim)
        print(f"[e2probe] restored {label}")

    rng = np.random.default_rng(0)
    rows = {label: {} for label in agents}
    for task in tasks:
        env, train_ds, _val = ogbench.make_env_and_datasets(task)
        S_all = np.asarray(train_ds["observations"], np.float32)
        A_all = np.asarray(train_ds["actions"], np.float32)
        r_all = np.asarray(train_ds["rewards"], np.float64) + 1.0
        m = min(args.relabel_size, len(S_all))
        idx = rng.choice(len(S_all), size=m, replace=False)
        S, A, r = S_all[idx], A_all[idx], r_all[idx]
        _obs, info = env.reset(seed=0)
        goal_obs = np.asarray(info["goal"], np.float32)
        if hasattr(env, "close"):
            env.close()
        print(f"[e2probe] {task}: n={m} r_pos={r.mean():.4f}")
        for label, agent in agents.items():
            rec = probe_agent_task(agent, S, A, r, goal_obs)
            rows[label][task] = rec
            c = rec["critic"]
            print(f"  {label:70s} critic: cos={c['cos_v0']:.3f} "
                  f"aucE={c['auc_exact']:.3f} auc0={c['auc_v0']:.3f}")

    summ = {"checkpoint_step": args.checkpoint_step, "relabel_size": args.relabel_size,
            "per_checkpoint": {}}
    for label in rows:
        entry = {}
        for legs in ("critic", "value"):
            for k in ("cos_v0", "auc_exact", "auc_v0"):
                vals = [rows[label][t][legs][k] for t in rows[label]
                        if rows[label][t][legs][k] == rows[label][t][legs][k]]
                entry[f"{legs}.{k}"] = float(np.mean(vals)) if vals else None
        summ["per_checkpoint"][label] = {"means": entry, "per_task": rows[label]}
        print(f"[e2probe] MEAN {label}: " +
              " ".join(f"{k}={v:.3f}" for k, v in entry.items() if v is not None))

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summ, indent=2))
    print(f"[e2probe] -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
