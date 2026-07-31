"""Spec 10 L1 on the TORCH phase-2 checkpoint: is the phase-2 object actually a
successor MEASURE, or just a critic that happens to drive a good actor?

Task success (0.89) scores the ACTOR. It says nothing about whether
    M-tilde(s,a,z,g) ~= M^{pi_z}(s,a,B_eps(g))
is calibrated. This script tests the measure directly, with no task and no
success metric, by comparing the model's own set-mass prediction against Monte
Carlo occupancy of the SAME policy rolled out in the env.

Two independent questions, deliberately separated:

  A. DENOMINATOR.  Phase 2b trains N-hat(g) = softplus(MLP(B(g))) to regress the
     hit-column mean. Ground truth is a pure counting problem: rho(B_eps(g)) =
     (# dataset cube positions within eps of g) / n, computed exactly by KDTree.
     So the head can be scored against truth with no rollouts at all.

  B. NUMERATOR / calibration.  For a start row (s0, qpos, qvel), z on-manifold,
     a0 ~ pi(s0,z):
        model : M-hat(X) = (1/n) sum_{i in X} M-tilde(c(x_i)) / N-hat(c(x_i))
        truth : MC(X)    = E[ sum_k gamma^k 1(cube_{1+k} in X) ]
     over random boxes X of several half-widths. Reported with the EXACT KDTree
     denominator (isolates whether M-tilde itself is calibrated) and again with
     the trained head (scores the phase-2 object end to end).

Index convention, matched to training (`diff = Ms - hit - discount*target_M`,
hit evaluated at next_obs): the MC sum starts at the state AFTER a0 with weight
gamma^0. Truncation at H steps leaves gamma^H of the mass unmeasured; that
residual is reported, not hidden.

Usage:
  .venv/bin/python scripts/rescuing_fb/torchfb_full_measure.py \
      --run /home/adityamohan/exp_spec10/torchfb/p2_s42 [--starts 24] [--out X.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_measure import FullMeasure  # noqa: E402
from env_geometry import (geometry_for, resolve_dataset,  # noqa: E402
                          episode_bounds, hindsight_rows, MULTS)


def probe_rows(policy: str, bounds, start_row: int, n: int,
               n_rows: int, rng) -> np.ndarray:
    """Rows whose coordinates become query centers for one start.

    'random'    — uniform over the dataset (the archived cube-single protocol).
    'hindsight' — strictly after start_row within its own episode, so the MC
                  occupancy has something to count (spec 14 §2.1/§3.3).
    """
    if policy == "random":
        return rng.choice(n_rows, min(n, n_rows), replace=False)
    if policy == "hindsight":
        return hindsight_rows(bounds, start_row, rng, n)
    raise ValueError(f"unknown probe-center policy {policy!r}")


def degenerate_frac(recs, mult) -> float:
    """G0: fraction of probes at this multiplier whose MC truth was exactly 0."""
    R = [r for r in recs if r["mult"] == mult]
    if not R:
        return float("nan")
    return float(np.mean([r["mc"] == 0.0 for r in R]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="hydra run dir of a phase-2 run")
    ap.add_argument("--ckpt", default="checkpoints/final.pt")
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--mc", type=int, default=2, help="rollouts per start")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--boxes", type=int, default=12, help="query boxes per half-width")
    ap.add_argument("--rho-samples", type=int, default=200_000,
                    help="dataset rows defining rho / the N-hat kernel")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    from evals.analysis import load_cfg, load_checkpoint
    from envs.ogbench import create_ogbench_env
    from train import make_agent
    import mujoco

    cfg = load_cfg(f"{a.run}/.hydra/config.yaml", device=a.device)
    discount = float(cfg.discount)
    print(f"[cfg] agent={cfg.agent} domain={cfg.domain} discount={discount} "
          f"measure_mode={cfg.get('measure_mode')} nhat_head={cfg.get('nhat_head')}")
    assert cfg.get("measure_mode") == "visitation", (
        "L1 calibration is defined against the VISITATION fixed point "
        "(discounted occupancy). first_passage has an absorbing recursion and a "
        "different ground truth; re-derive before running it here.")

    geo = geometry_for(cfg.domain, cfg.get("hit_spec") or "")
    spec = geo.hit_spec
    print(f"[geo] env={geo.env_name} blocks={spec.blocks} "
          f"probe_centers={geo.probe_centers}")

    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type)
    obs_space = env.observation_space
    action_dim = int(env.action_space.shape[0])
    agent = make_agent(cfg, obs_space, action_dim)
    load_checkpoint(agent, f"{a.run}/{a.ckpt}", map_location=a.device)
    agent.model.eval()

    # ---- dataset: rho (measure coordinates) + resettable states --------------
    npz = np.load(resolve_dataset(cfg.data_path, cfg.domain))
    obs_all, qpos_all, qvel_all = npz["observations"], npz["qpos"], npz["qvel"]
    bounds = episode_bounds(npz["terminals"])
    n_rho = min(a.rho_samples, len(obs_all))
    rho_idx = rng.choice(len(obs_all), n_rho, replace=False)
    rho_obs = obs_all[rho_idx]
    cube_ds = spec.coords(rho_obs)
    print(f"[data] rho = {n_rho:,} rows; coord bbox "
          f"{cube_ds.min(0).round(3)} .. {cube_ds.max(0).round(3)}")

    dev = a.device
    t_rho_obs = torch.as_tensor(rho_obs, device=dev)

    @torch.no_grad()
    def B_of(idx: np.ndarray) -> torch.Tensor:
        return agent.model.backward_map(t_rho_obs[torch.as_tensor(idx, device=dev)])

    # ---------------- A. denominator: trained head vs exact kernel count ------
    fm_exact = FullMeasure(cube_ds, spec=spec)
    probe = rng.choice(n_rho, min(4000, n_rho), replace=False)
    nhat_true = fm_exact.nhat(cube_ds[probe])
    head = getattr(agent, "_nhat_head", None)
    res_A = {"has_head": head is not None}
    if head is not None:
        with torch.no_grad():
            nhat_pred = head(B_of(probe)).squeeze(-1).float().cpu().numpy().astype(np.float64)
        rel = np.abs(nhat_pred - nhat_true) / np.maximum(nhat_true, 1e-9)
        res_A.update(
            corr=float(np.corrcoef(nhat_pred, nhat_true)[0, 1]),
            median_rel_err=float(np.median(rel)),
            mean_rel_err=float(rel.mean()),
            pred_mean=float(nhat_pred.mean()), true_mean=float(nhat_true.mean()),
        )
        print(f"[A] N-hat head vs exact kernel count over {len(probe)} goals: "
              f"corr={res_A['corr']:.3f}  median rel.err={res_A['median_rel_err']:.1%}  "
              f"mean pred={res_A['pred_mean']:.5f} vs true={res_A['true_mean']:.5f}")
    else:
        print("[A] checkpoint has no nhat_head — denominator test skipped")

    # trained-head denominator, looked up by nearest dataset row (centers passed
    # by FullMeasure are dataset cube positions, so the lookup is exact up to
    # rows that share a cube position but differ elsewhere in the observation).
    fm_head = None
    if head is not None:
        def nhat_fn(centers):
            nn = fm_exact.nearest(centers)
            with torch.no_grad():
                return head(B_of(nn)).squeeze(-1).float().cpu().numpy()
        fm_head = FullMeasure(cube_ds, spec=spec, nhat_fn=nhat_fn)

    # ---------------- B. calibration: model set-mass vs MC occupancy ----------
    # Query centers follow the env's probe-center policy (spec 14 §3.3).
    #
    # 'random' reproduces the ARCHIVED cube-single protocol bit-for-bit: centers
    # are drawn ONCE, independently per multiplier, from the rho subsample, and
    # BEFORE start/goal rows -- the rng draw order is part of the protocol, so
    # reordering it silently re-rolls which starts get evaluated (caught by the
    # §5 regression gate on 2026-07-31).
    #
    # 'hindsight' must draw per start, since the centers depend on the start's
    # own episode tail; its draws therefore come after start_rows.
    fixed_queries = None
    if geo.probe_centers == "random":
        fixed_queries = []
        for mult in MULTS:
            rows = probe_rows("random", bounds, 0, a.boxes, n_rho, rng)
            for c in cube_ds[rows]:
                fixed_queries.append({"kind": "box", "center": c, "mult": mult})

    start_rows = rng.choice(len(obs_all), a.starts, replace=False)
    goal_rows = rng.choice(len(obs_all), a.starts, replace=False)
    u = env.unwrapped
    recs = []
    n_skipped = 0

    for si, (s_row, g_row) in enumerate(zip(start_rows, goal_rows)):
        if fixed_queries is not None:
            queries = fixed_queries
        else:
            centers_rows = probe_rows(geo.probe_centers, bounds, int(s_row),
                                      a.boxes, len(obs_all), rng)
            if len(centers_rows) == 0:      # start sits at its episode's end
                n_skipped += 1
                continue
            centers = spec.coords(obs_all[centers_rows])
            queries = [{"kind": "box", "center": c, "mult": mult}
                       for mult in MULTS for c in centers]

        s0 = torch.as_tensor(obs_all[s_row][None], device=dev)
        # on-manifold z: B(g) of a dataset state, projected (the training z law)
        with torch.no_grad():
            z = agent.model.project_z(
                agent.model.backward_map(torch.as_tensor(obs_all[g_row][None], device=dev)))
            a0 = agent.act(s0, z, eval_mode=True)
            F0 = agent.model.forward_map(s0, z, a0).mean(0)      # [1,z_dim] over P heads

        def ball_value_fn(idx):
            with torch.no_grad():
                return (B_of(idx) @ F0.squeeze(0)).float().cpu().numpy().astype(np.float64)

        # ---- MC: roll pi_z from the state reached by a0 ----------------------
        occ = np.zeros(len(queries))
        for _ in range(a.mc):
            # reset FIRST: without it the TimeLimit step counter carries over
            # between starts and every rollout after the first truncates at
            # step 1 (audit 2026-07-28 — this alone produced the 65-215x
            # "miscalibration"). The teleport below overwrites the reset state.
            env.reset()
            u._data.qpos[:] = qpos_all[s_row]
            u._data.qvel[:] = qvel_all[s_row]
            mujoco.mj_forward(u._model, u._data)
            ob, *_ = env.step(np.clip(a0.squeeze(0).cpu().numpy(), -1, 1))
            g = 1.0
            for t in range(a.horizon):
                c = spec.coords(ob[None])[0]
                for qi, q in enumerate(queries):
                    if spec.box_contains(c[None], q["center"],
                                         mult=q["mult"])[0]:
                        occ[qi] += g
                g *= discount
                with torch.no_grad():
                    act = agent.act(torch.as_tensor(ob[None], device=dev), z,
                                    eval_mode=False)
                ob, _, term, trunc, _ = env.step(np.clip(act.squeeze(0).cpu().numpy(), -1, 1))
                if term or trunc:
                    break
        occ /= a.mc

        for qi, q in enumerate(queries):
            mem = fm_exact.member_indices(q)
            r = {"start": int(s_row), "mult": q["mult"], "mc": float(occ[qi]),
                 "n_members": int(len(mem))}
            r["model_exact"] = fm_exact.set_mass(ball_value_fn, mem)["mass"]
            if fm_head is not None:
                r["model_head"] = fm_head.set_mass(ball_value_fn, mem)["mass"]
            recs.append(r)
        if (si + 1) % 4 == 0:
            print(f"[B] {si+1}/{a.starts} starts done", flush=True)

    # ---------------- report --------------------------------------------------
    trunc_residual = discount ** a.horizon
    print(f"\n[B] horizon={a.horizon}, unmeasured tail gamma^H = {trunc_residual:.3f}")
    out = {"run": a.run, "env": cfg.domain, "A_denominator": res_A,
           "mults": list(MULTS), "probe_centers": geo.probe_centers,
           "hit_spec": ",".join(f"{a_}:{b_}:{e_}" for a_, b_, e_ in spec.blocks),
           "starts_skipped": n_skipped,
           "horizon": a.horizon, "gamma_H": trunc_residual, "records": recs}
    print(f"\n{'mult':>6} {'n':>5} {'MC mean':>10} {'model(exact)':>13} {'ratio':>7} "
          f"{'model(head)':>12} {'ratio':>7} {'degen':>7}")
    for mult in MULTS:
        R = [r for r in recs if r["mult"] == mult]
        if not R:
            continue
        mc = np.array([r["mc"] for r in R]); me = np.array([r["model_exact"] for r in R])
        dg = degenerate_frac(recs, mult)
        row = f"{mult:>6} {len(R):>5} {mc.mean():>10.4f} {me.mean():>13.4f} "
        row += f"{(me.mean()/max(mc.mean(),1e-9)):>7.2f} "
        if fm_head is not None:
            mh = np.array([r["model_head"] for r in R])
            row += f"{mh.mean():>12.4f} {(mh.mean()/max(mc.mean(),1e-9)):>7.2f} "
        row += f"{dg:>7.2f}"
        print(row)
        out.setdefault("summary", {})[str(mult)] = {
            "n": len(R), "mc_mean": float(mc.mean()),
            "model_exact_mean": float(me.mean()),
            "ratio_exact": float(me.mean() / max(mc.mean(), 1e-9)),
            "spearman_exact": float(_spearman(me, mc)),
            "degenerate_frac": dg,
            "abs_half_widths": (mult * spec.radii).round(3).tolist(),
        }
        if fm_head is not None:
            mh = np.array([r["model_head"] for r in R])
            out["summary"][str(mult)].update(
                model_head_mean=float(mh.mean()),
                ratio_head=float(mh.mean() / max(mc.mean(), 1e-9)),
                spearman_head=float(_spearman(mh, mc)))
    print("\n[rank] Spearman(model, MC) per multiplier — calibration of ORDERING:")
    for mult in MULTS:
        s = out.get("summary", {}).get(str(mult))
        if s is None:
            continue
        print(f"  mult={mult}: exact={s['spearman_exact']:+.3f}" +
              (f"  head={s['spearman_head']:+.3f}" if "spearman_head" in s else "") +
              f"  degenerate_frac={s['degenerate_frac']:.2f}")
    if n_skipped:
        print(f"[B] {n_skipped}/{a.starts} starts skipped (no episode tail)")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2, default=float)
        print(f"\n[out] {a.out}")
    return 0


def _spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


if __name__ == "__main__":
    raise SystemExit(main())
