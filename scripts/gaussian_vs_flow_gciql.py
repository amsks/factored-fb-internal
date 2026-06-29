#!/usr/bin/env python
"""scripts/gaussian_vs_flow_gciql.py — does the gaussian (DDPG+BC) actor pick worse
actions than the flow actor on the SAME bilinear value, and/or does it fail to reach
the post-grasp states?

Loads the matched d=50 runs:
  flow   = GCIQL-State-Flow-Bilinear-d50          (90% success)
  gauss  = GCIQL-State-Flow-Bilinear-d50-ddpgbc   (48% success)

Two questions, separating ACTION-QUALITY from STATE-COVERAGE:
  (1) Action quality @ matched states: roll the FLOW policy to get on-good-distribution
      states; at each, score the flow actor's action and the gaussian actor's action
      under the SAME critic (the flow run's, i.e. the good value). dQ = Q_flow(a_flow_best)
      - Q_flow(a_gauss). Large positive dQ at grasp/transport => the gaussian leaves value
      on the table where it matters.
  (2) State coverage: roll EACH policy and compare the phase distribution (approach/contact
      /grasp). If the gaussian rarely reaches 'grasp', the 90->48 gap is largely a
      can't-grasp (state-coverage) effect, not an at-state action-quality effect.

Run: JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \
       scripts/gaussian_vs_flow_gciql.py --flow-dir <d50_sd000> --gauss-dir <d50_ddpgbc_sd000> \
       --seed 0 --out-dir analysis/q_landscape/gauss_vs_flow_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "scripts", REPO / "third_party" / "ogbench" / "impls",
           REPO / "tools" / "wandb_mode_shim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from q_landscape_gciql import (collect_states, q_batch, flow_samples,  # noqa: E402
                               PHASES, PHASE_COLOR, EVAL_TASKS)


def load_agent_kind(checkpoint_dir, step, seed, d, actor_kind):
    """phase_probe_gciql.load_agent, but with actor_kind in the config so the
    DDPG+BC (gaussian) param tree aligns."""
    import jax.numpy as jnp
    import ogbench
    from gciql_flow import GCIQLFlowAgent, get_config
    from utils.flax_utils import restore_agent
    tmp = ogbench.make_env_and_datasets(EVAL_TASKS[0], env_only=True)
    ex_obs = jnp.asarray(tmp.reset()[0], jnp.float32)[None]
    ex_act = jnp.zeros((1, *tmp.action_space.shape), jnp.float32)
    cfg = dict(get_config())
    cfg.update(value_arch="bilinear", value_latent_dim=d, actor_kind=actor_kind,
               actor_cond="goal", encoder=None, frame_stack=None)
    if hasattr(tmp, "close"):
        tmp.close()
    agent = GCIQLFlowAgent.create(seed, ex_obs, ex_act, cfg)
    return restore_agent(agent, str(Path(checkpoint_dir).resolve()), step)


def gauss_action(agent, obs, goal):
    """Deterministic gaussian-actor action (temperature=0 => mean)."""
    import jax
    a = agent.sample_actions(observations=np.asarray(obs, np.float32),
                             goals=np.asarray(goal, np.float32),
                             seed=jax.random.PRNGKey(0), temperature=0.0)
    return np.asarray(a, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-dir", required=True)
    ap.add_argument("--gauss-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-latent-dim", type=int, default=50)
    ap.add_argument("--task", default="cube-single-play-singletask-task1-v0")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--kbest", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ogbench

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = args.value_latent_dim
    flow = load_agent_kind(args.flow_dir, args.checkpoint_step, args.seed, d, "flow")
    gauss = load_agent_kind(args.gauss_dir, args.checkpoint_step, args.seed, d, "ddpgbc")
    print("[gvf] both agents restored")

    # ---- (2) STATE COVERAGE: roll each policy, compare phase distribution -------
    env = ogbench.make_env_and_datasets(args.task, env_only=True)
    Dflow = collect_states(flow, env, args.episodes)
    Dg = collect_states(gauss, env, args.episodes)
    cov = {"flow": {p: int((Dflow["phase"] == p).sum()) for p in PHASES},
           "gauss": {p: int((Dg["phase"] == p).sum()) for p in PHASES}}
    covfrac = {k: {p: cov[k][p] / max(1, sum(cov[k].values())) for p in PHASES} for k in cov}
    print("[gvf] phase fractions  flow :", {p: round(covfrac['flow'][p], 2) for p in PHASES})
    print("[gvf] phase fractions  gauss:", {p: round(covfrac['gauss'][p], 2) for p in PHASES})

    # ---- (1) ACTION QUALITY @ matched (flow-visited) states, scored by C_flow ---
    rows = {p: {"flow_best": [], "flow1": [], "gauss": []} for p in PHASES}
    rng = np.random.default_rng(0)
    for p in PHASES:
        idx = np.where(Dflow["phase"] == p)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(80, len(idx)), replace=False)
        for i in pick:
            o, g = Dflow["obs"][i], Dflow["goal_obs"][i]
            A = flow_samples(flow, o, g, args.kbest)            # flow actor samples
            qA = q_batch(flow, o, g, A)                          # scored by GOOD critic
            ag = gauss_action(gauss, o, g)                       # gaussian deterministic action
            qg = q_batch(flow, o, g, ag[None])[0]               # scored by GOOD critic
            rows[p]["flow_best"].append(float(qA.max()))
            rows[p]["flow1"].append(float(qA[0]))
            rows[p]["gauss"].append(float(qg))

    def stat(xs):
        return {"mean": float(np.mean(xs)), "median": float(np.median(xs)), "n": len(xs)} if xs else None
    dq = {}
    for p in PHASES:
        fb, gz = rows[p]["flow_best"], rows[p]["gauss"]
        if fb and gz:
            d_arr = np.array(fb) - np.array(gz)
            dq[p] = {"dQ_mean": float(d_arr.mean()), "dQ_median": float(np.median(d_arr)),
                     "Q_flow_best": stat(fb), "Q_gauss": stat(gz), "n": len(fb)}
            print(f"[gvf] {p}: Q_flow_best={np.mean(fb):.1f}  Q_gauss={np.mean(gz):.1f}  "
                  f"dQ(flow-gauss)={d_arr.mean():.2f}")

    summary = {"seed": args.seed, "d": d, "task": args.task,
               "coverage": cov, "coverage_frac": covfrac, "action_quality": dq}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- plots -----------------------------------------------------------------
    # phase coverage
    fig, ax = plt.subplots(figsize=(5.4, 4))
    x = np.arange(len(PHASES)); w = 0.38
    ax.bar(x - w/2, [covfrac["flow"][p] for p in PHASES], w, label="flow (90%)", color="#2ca02c")
    ax.bar(x + w/2, [covfrac["gauss"][p] for p in PHASES], w, label="gauss (48%)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(PHASES); ax.set_ylabel("fraction of visited states")
    ax.set_title(f"State coverage by phase (d={d})"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "phase_coverage.png", dpi=130); plt.close(fig)

    # action quality: Q under the good critic, flow_best vs gauss, by phase
    fig, ax = plt.subplots(figsize=(6, 4.2))
    pos = np.arange(len(PHASES))
    have = [p for p in PHASES if p in dq]
    fb_means = [dq[p]["Q_flow_best"]["mean"] for p in have]
    gz_means = [dq[p]["Q_gauss"]["mean"] for p in have]
    xb = np.arange(len(have)); w = 0.38
    ax.bar(xb - w/2, fb_means, w, label="flow action (best of K)", color="#2ca02c")
    ax.bar(xb + w/2, gz_means, w, label="gaussian action", color="#d62728")
    ax.set_xticks(xb); ax.set_xticklabels(have)
    ax.set_ylabel("Q under flow-run critic (good value)")
    ax.set_title(f"Action value: flow vs gaussian @ matched states (d={d})")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "action_quality.png", dpi=130); plt.close(fig)

    print(f"[gvf] wrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
