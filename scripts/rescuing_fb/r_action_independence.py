#!/usr/bin/env python
"""Spec 08 E5: is the recurrence factor R = M/Q action-(and state-)independent?

The two-phase curriculum rests on M(s,a,g) = Q(s,a,g) * R(g) with R a function of the
goal ALONE (the note's Eq. factor). renewal_head.py already validated the MAGNITUDE claim
(R carries occupancy, Spearman 0.60-0.63). This script tests the ACTION-INDEPENDENCE claim
directly, the two diagnostics the note lists as "not yet run":

  Q(s,a,g) = reaching value  = min-head phi_FP(s,a) . psi(g) / sqrt(d)   [FP-NoOrtho ckpt]
  M(s,a,g) = visitation value = min-head phi_vis(s,a) . psi(g) / sqrt(d) [BallVisit ckpt]
  (BallVisit froze psi warm-started from FP, so both readouts use the SAME psi(g).)

  (1) conditional variance of R = M/Q given g: ICC = 1 - E_g[Var(R|g)] / Var(R). ICC->1
      means R is (almost) a function of g alone -- the factorization is sound. Low ICC means
      R carries (s,a) information the decomposition discards.
  (2) argmax_a Q vs argmax_a M agreement over a candidate-action set, and the per-(s,g)
      Spearman rank correlation of Q vs M across actions. If R is action-independent then
      M = Q*R(g) is monotone in Q at fixed g, so the argmaxes coincide and rank corr -> 1.

Runs in the jax venv (restores two checkpoints). No environment rollouts.

  JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/r_action_independence.py \
    --fp-run  exp_spec06/.../GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s0 \
    --vis-run exp_spec07/.../GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50/vis_s0 \
    --goals 150 --states 24 --actions 16 --out analysis/rescuing-fb/e5_action_indep_s0.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))


def spearman(a, b):
    """Rank correlation of two 1-D arrays (ties -> average rank)."""
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x))
        # average ranks for ties
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        avg = {i: (csum[i] - 1 - (cnt[i] - 1) / 2.0) for i in range(len(cnt))}
        return np.array([avg[i] for i in inv])
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-run", required=True, help="FP-NoOrtho run dir (reaching Q)")
    ap.add_argument("--vis-run", required=True, help="BallVisit run dir (visitation M)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--goals", type=int, default=150)
    ap.add_argument("--states", type=int, default=24, help="states per goal")
    ap.add_argument("--actions", type=int, default=16, help="candidate actions per (s,g)")
    ap.add_argument("--q-thresh", type=float, default=0.05,
                    help="keep (s,a,g) with Q above this (ball reachable => M/Q meaningful)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-task", default="cube-single-play-singletask-task1-v0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import jax  # noqa: F401
    import ogbench
    from t6_gpi_eval import load_agent_flexible, make_psi, make_q_at_z

    rng = np.random.default_rng(args.seed)

    # --- dataset states ---
    _env, train_ds, _val = ogbench.make_env_and_datasets(args.dataset_task)
    obs_all = np.asarray(train_ds["observations"], np.float32)
    act_dim = int(np.asarray(train_ds["actions"]).shape[-1])
    d_obs = obs_all.shape[1]

    # --- restore both critics (arch from flags.json) ---
    fp_agent, d, _ = load_agent_flexible(Path(args.fp_run), args.checkpoint_step,
                                         False, 50, "z", "none")
    vis_agent, dv, _ = load_agent_flexible(Path(args.vis_run), args.checkpoint_step,
                                           False, 50, "z", "none")
    assert d == dv, (d, dv)
    psi_fp = make_psi(fp_agent, src="critic", act_dim=act_dim, fbz=False)
    q_fp = make_q_at_z(fp_agent, d, fbz=False)   # Q(s,a,z)
    q_vis = make_q_at_z(vis_agent, d, fbz=False)  # M(s,a,z)

    # --- build the (goal, state, action) grid ---
    G, S, A = args.goals, args.states, args.actions
    gi = rng.integers(0, len(obs_all), size=G)
    goals = obs_all[gi]                                  # [G, d_obs]
    z_goals = np.asarray(psi_fp(goals), np.float64)      # [G, d]  (shared psi)

    # per goal: S states x A actions
    Qs = np.empty((G, S, A)); Ms = np.empty((G, S, A))
    for g in range(G):
        si = rng.integers(0, len(obs_all), size=S)
        s = obs_all[si]                                  # [S, d_obs]
        a = rng.uniform(-1.0, 1.0, size=(S, A, act_dim)).astype(np.float32)
        obs_rep = np.repeat(s[:, None, :], A, axis=1).reshape(S * A, d_obs)
        act_rep = a.reshape(S * A, act_dim)
        zg = z_goals[g].astype(np.float32)              # single task vector [d], broadcast
        Qs[g] = np.asarray(q_fp(obs_rep, act_rep, zg)).reshape(S, A)
        Ms[g] = np.asarray(q_vis(obs_rep, act_rep, zg)).reshape(S, A)

    # ---------- diagnostic 2: argmax agreement + per-(s,g) rank corr ----------
    argmax_agree, rankcorrs = [], []
    for g in range(G):
        for si in range(S):
            q, m = Qs[g, si], Ms[g, si]
            if q.max() < args.q_thresh:
                continue
            argmax_agree.append(int(np.argmax(q) == np.argmax(m)))
            rc = spearman(q, m)
            if not np.isnan(rc):
                rankcorrs.append(rc)

    # ---------- diagnostic 1: ICC of R = M/Q given g ----------
    # pool (s,a) samples per goal; R meaningful only where Q is above threshold.
    R_by_g, R_all = [], []
    for g in range(G):
        q, m = Qs[g].ravel(), Ms[g].ravel()
        keep = q > args.q_thresh
        if keep.sum() < 2:
            continue
        r = m[keep] / q[keep]
        R_by_g.append(r)
        R_all.append(r)
    R_flat = np.concatenate(R_all) if R_all else np.array([])
    within = np.mean([r.var() for r in R_by_g]) if R_by_g else np.nan
    total = R_flat.var() if len(R_flat) else np.nan
    icc = float(1.0 - within / total) if total and total > 0 else np.nan
    # goal-level coefficient of variation of R (dispersion of R within a goal, relative to mean)
    cv_within = float(np.median([r.std() / (abs(r.mean()) + 1e-9) for r in R_by_g])) if R_by_g else np.nan

    res = {
        "fp_run": args.fp_run, "vis_run": args.vis_run, "d": d,
        "grid": {"goals": G, "states": S, "actions": A, "q_thresh": args.q_thresh},
        "n_reachable_sg": len(argmax_agree),
        "argmax_agreement": float(np.mean(argmax_agree)) if argmax_agree else np.nan,
        "rankcorr_QM_mean": float(np.mean(rankcorrs)) if rankcorrs else np.nan,
        "rankcorr_QM_median": float(np.median(rankcorrs)) if rankcorrs else np.nan,
        "R_icc_given_g": icc,                    # ->1 : R is a function of g alone
        "R_cv_within_goal_median": cv_within,    # ->0 : R action/state-independent at fixed g
        "R_mean": float(R_flat.mean()) if len(R_flat) else np.nan,
        "R_median": float(np.median(R_flat)) if len(R_flat) else np.nan,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[e5] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
