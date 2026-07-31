"""Spec 12 — L2 zero-shot reward test on the torch lambda=0 phase-2 checkpoints.

Approach 1 (minimal pipeline): ONE readout — FB's standard reward_inference
(z_r = project(sum r_i * B(s_i)) over dataset rows) — no ridge, no N-hat
correction, no GPI. Two stages:

  A. PREDICTION: does <F(s0,z_r,a0), z> match the MC discounted return of
     pi(.,z_r) under the new reward?  pred_proj uses the projected z_r (the
     actor's z; correlation meaningful, scale not), pred_raw uses the
     unprojected mean (the implied ∫r dM-tilde; carries the pre-registered
     no-1/N-hat density bias).
  B. CONTROL: does pi(.,z_r) EARN the reward, read against z_goal (ceiling)
     and z_rand (floor) on the same checkpoint?

Reward families (cube xyz, obs units; trained kernel eps=0.4): ball0.4 (ref),
ball0.8, ball1.6, smooth0.4, smooth0.8 (gaussian, no threshold), twozone
(two disjoint eps=0.4 balls, centers >1.6 apart).

MC/teleport protocol per audit 2026-07-28: env.reset() BEFORE every rollout
(TimeLimit counter carries over otherwise), then overwrite qpos/qvel.

Usage:
  .venv/bin/python scripts/rescuing_fb/torchfb_zeroshot_l2.py \
      --run /home/adityamohan/exp_spec10/torchfb/p2_s42 --stage A \
      [--out results/spec10_torchfb/L2_A_p2_s42.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BALL_MULTS = (1.0, 2.0, 4.0)     # region radii as multiples of the trained ones
SMOOTH_MULTS = (1.0, 2.0)


def make_rewards(cube_ds: np.ndarray, rng: np.random.Generator, per_family: int,
                 spec=None):
    """Held-out reward families over the measure coordinate (spec 14 §3.4).

    Radii are MULTIPLIERS on the trained per-block radii, so a family means the
    same relative region size on every environment. On cube-single (a single
    block at eps=0.4) the multipliers reproduce the archived families exactly:
    ball1.0/2.0/4.0 == the old ball0.4/0.8/1.6, smooth1.0/2.0 == the old
    smooth0.4/0.8, and twozone keeps its 4x-radius separation rule.
    """
    from env_geometry import HitSpec
    if spec is None:
        spec = HitSpec.single(0, np.shape(cube_ds)[1], 0.4)
    scale = float(np.mean(spec.radii))

    def ball(center, mult):
        return lambda c: spec.contains(c, center, mult=mult).astype(np.float64)

    def smooth(center, mult):
        sig = mult * scale
        return lambda c: np.exp(
            -np.sum((np.asarray(c, np.float64) - center) ** 2, axis=-1)
            / (2 * sig ** 2))

    def twozone(a, b, mult):
        fa, fb = ball(a, mult), ball(b, mult)
        return lambda c: np.maximum(fa(c), fb(c))

    rewards = []
    centers = cube_ds[rng.choice(len(cube_ds), 4 * per_family, replace=False)]
    for k in range(per_family):
        c = centers[k]
        for m in BALL_MULTS:
            rewards.append(dict(family=f"ball{m}", name=f"ball{m}_{k}",
                                fn=ball(c, m), center=c,
                                abs_radius=(m * spec.radii).round(3).tolist()))
        for m in SMOOTH_MULTS:
            rewards.append(dict(family=f"smooth{m}", name=f"smooth{m}_{k}",
                                fn=smooth(c, m), center=c,
                                abs_sigma=round(m * scale, 3)))
    made = 0
    for k in range(per_family, 4 * per_family - 1):
        a, b = centers[k], centers[k + 1]
        if np.linalg.norm(a - b) > 4.0 * scale:
            rewards.append(dict(family="twozone", name=f"twozone_{made}",
                                fn=twozone(a, b, 1.0), center=a, center_b=b,
                                abs_radius=spec.radii.round(3).tolist()))
            made += 1
            if made == per_family:
                break
    return rewards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoints/final.pt")
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--starts", type=int, default=12, help="stage A teleport starts")
    ap.add_argument("--mc", type=int, default=2, help="stage A rollouts per start")
    ap.add_argument("--episodes", type=int, default=10, help="stage B episodes/drive")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--relabel-size", type=int, default=10_000)
    ap.add_argument("--nhat-weight", action="store_true",
                    help="density-corrected inference: weight each relabel "
                         "sample by 1/N-hat(cube) (ledger 9.10 next cell)")
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
          f"measure_mode={cfg.get('measure_mode')} hit_identity={cfg.get('hit_identity')}")
    # Any agent exposing the readout API is admissible: this harness scores a
    # READOUT, and the cross-method arms exist precisely to run the reaching
    # value (first_passage) and FB's own loss (hitdirac) through the identical
    # protocol as the visitation measure. The capability is checked below, after
    # the agent is built; measure_mode is recorded in the output, not gated on.

    env, _ = create_ogbench_env(cfg.domain, obs_type=cfg.obs_type, seed=a.seed)
    agent = make_agent(cfg, env.observation_space, int(env.action_space.shape[0]))
    load_checkpoint(agent, f"{a.run}/{a.ckpt}", map_location=a.device)
    for api in ("reward_inference", "backward_map", "project_z", "sample_z"):
        assert hasattr(agent.model, api), (
            f"{cfg.agent} lacks model.{api}; this harness needs the readout API")
    assert hasattr(agent, "act"), f"{cfg.agent} lacks .act"
    agent.model.eval()
    dev = a.device

    from env_geometry import geometry_for, resolve_dataset
    geo = geometry_for(cfg.domain, cfg.get("hit_spec") or "")
    spec = geo.hit_spec
    print(f"[geo] env={geo.env_name} blocks={spec.blocks}")

    npz = np.load(resolve_dataset(cfg.data_path, cfg.domain))
    obs_all, qpos_all, qvel_all = npz["observations"], npz["qpos"], npz["qvel"]
    cube_all = spec.coords(obs_all)

    rewards = make_rewards(cube_all, rng, a.per_family, spec=spec)
    print(f"[rewards] {len(rewards)} instances "
          f"({sorted(set(r['family'] for r in rewards))})")

    # ---- z inference: the readout under test ---------------------------------
    inf_rows = rng.choice(len(obs_all), a.relabel_size, replace=False)
    inf_obs = torch.as_tensor(obs_all[inf_rows], device=dev)
    inf_cube = cube_all[inf_rows]

    weight_t = None
    if a.nhat_weight:
        from full_measure import FullMeasure
        ref_rows = rng.choice(len(obs_all), 200_000, replace=False)
        fm = FullMeasure(cube_all[ref_rows], spec=spec)
        w = 1.0 / fm.nhat(inf_cube)
        weight_t = torch.as_tensor(w[:, None], dtype=torch.float32, device=dev)
        print(f"[infer] 1/N-hat weights: mean={w.mean():.1f} "
              f"min={w.min():.1f} max={w.max():.1f}")

    for r in rewards:
        rw = r["fn"](inf_cube)
        rw_t = torch.as_tensor(rw[:, None], dtype=torch.float32, device=dev)
        wrw_t = rw_t if weight_t is None else rw_t * weight_t
        with torch.no_grad():
            r["z"] = agent.model.reward_inference(
                next_obs=inf_obs, reward=rw_t, weight=weight_t)[None]
            # unprojected mean for the scale-carrying prediction (spec 12)
            zr = 0.0
            bs = 4096
            for i in range(0, len(inf_rows), bs):
                B = agent.model.backward_map(inf_obs[i:i + bs])
                zr = zr + (wrw_t[i:i + bs].T @ B)
            r["z_raw"] = (zr / len(inf_rows)).float()
        r["r_mean"] = float(rw.mean())
    print("[infer] z done for all rewards "
          f"({'1/N-hat weighted' if a.nhat_weight else 'naive'})")

    u = env.unwrapped
    out = {"run": a.run, "stage": a.stage, "discount": discount,
           "env": cfg.domain,
           "hit_spec": ",".join(f"{a_}:{b_}:{e_}" for a_, b_, e_ in spec.blocks),
           "horizon": a.horizon, "gamma_H": discount ** a.horizon,
           "per_family": a.per_family, "nhat_weight": bool(a.nhat_weight),
           "records": []}

    def cube_of(ob):
        return spec.coords(np.asarray(ob)[None])[0]

    if a.stage == "A":
        start_rows = rng.choice(len(obs_all), a.starts, replace=False)
        for ri, r in enumerate(rewards):
            z = r["z"]
            for s_row in start_rows:
                s0 = torch.as_tensor(obs_all[s_row][None], device=dev)
                with torch.no_grad():
                    a0 = agent.act(s0, z, eval_mode=True)
                    F0 = agent.model.forward_map(s0, z, a0).mean(0)
                    pred_proj = float((F0 * z).sum())
                    pred_raw = float((F0 * r["z_raw"]).sum())
                mc_vals = []
                for _ in range(a.mc):
                    env.reset()  # MUST precede teleport (audit 2026-07-28)
                    u._data.qpos[:] = qpos_all[s_row]
                    u._data.qvel[:] = qvel_all[s_row]
                    mujoco.mj_forward(u._model, u._data)
                    ob, *_ = env.step(np.clip(a0.squeeze(0).cpu().numpy(), -1, 1))
                    g, ret = 1.0, 0.0
                    for _t in range(a.horizon):
                        ret += g * float(r["fn"](cube_of(ob)[None])[0])
                        g *= discount
                        with torch.no_grad():
                            act = agent.act(torch.as_tensor(ob[None], device=dev), z,
                                            eval_mode=False)
                        ob, _, term, trunc, _ = env.step(
                            np.clip(act.squeeze(0).cpu().numpy(), -1, 1))
                        if term or trunc:
                            break
                    mc_vals.append(ret)
                out["records"].append(
                    {"family": r["family"], "name": r["name"], "start": int(s_row),
                     "pred_proj": pred_proj, "pred_raw": pred_raw,
                     "mc": float(np.mean(mc_vals))})
            if (ri + 1) % 8 == 0:
                print(f"[A] {ri + 1}/{len(rewards)} rewards done", flush=True)

    else:  # ---- stage B: control ---------------------------------------------
        tree_rows = rng.choice(len(obs_all), 100_000, replace=False)
        from scipy.spatial import cKDTree
        tree = cKDTree(cube_all[tree_rows])

        def goal_z(center):
            _, j = tree.query(np.asarray(center, np.float64))
            g_obs = torch.as_tensor(obs_all[tree_rows[j]][None], device=dev)
            with torch.no_grad():
                return agent.model.project_z(agent.model.backward_map(g_obs))

        with torch.no_grad():
            z_rand = agent.model.sample_z(1, device=dev)

        for ri, r in enumerate(rewards):
            drives = {"zr": r["z"], "zgoal": goal_z(r["center"]), "zrand": z_rand}
            if "center_b" in r:  # twozone ceiling: better single center
                drives["zgoal_b"] = goal_z(r["center_b"])
            for drive, z in drives.items():
                rets, times_in = [], []
                for _ep in range(a.episodes):
                    ob, _ = env.reset()
                    g, ret, tin = 1.0, 0.0, 0
                    for _t in range(a.horizon):
                        with torch.no_grad():
                            act = agent.act(torch.as_tensor(ob[None], device=dev), z,
                                            eval_mode=True)
                        ob, _, term, trunc, _ = env.step(
                            np.clip(act.squeeze(0).cpu().numpy(), -1, 1))
                        rv = float(r["fn"](cube_of(ob)[None])[0])
                        ret += g * rv
                        tin += int(rv > 0.5)
                        g *= discount
                        if term or trunc:
                            break
                    rets.append(ret)
                    times_in.append(tin)
                out["records"].append(
                    {"family": r["family"], "name": r["name"], "drive": drive,
                     "ret_mean": float(np.mean(rets)), "ret_std": float(np.std(rets)),
                     "time_in": float(np.mean(times_in))})
            if (ri + 1) % 8 == 0:
                print(f"[B] {ri + 1}/{len(rewards)} rewards done", flush=True)

    # ---- report --------------------------------------------------------------
    recs = out["records"]
    fams = sorted(set(x["family"] for x in recs))
    if a.stage == "A":
        from scipy.stats import pearsonr, spearmanr
        for fam in fams:
            R = [x for x in recs if x["family"] == fam]
            mc = np.array([x["mc"] for x in R])
            pr = np.array([x["pred_raw"] for x in R])
            pp = np.array([x["pred_proj"] for x in R])
            scale = pr.mean() / max(mc.mean(), 1e-9)
            summ = {"n": len(R),
                    "spearman_raw": float(spearmanr(mc, pr).statistic),
                    "pearson_raw": float(pearsonr(mc, pr).statistic),
                    "spearman_proj": float(spearmanr(mc, pp).statistic),
                    "scale_raw": float(scale), "mc_mean": float(mc.mean())}
            out.setdefault("summary", {})[fam] = summ
            print(f"[A] {fam:>10}: spearman_raw={summ['spearman_raw']:+.3f} "
                  f"(proj {summ['spearman_proj']:+.3f}) scale={scale:8.2f} "
                  f"MC={mc.mean():.3f}")
    else:
        for fam in fams:
            R = [x for x in recs if x["family"] == fam]
            by = {}
            for d in ("zr", "zgoal", "zgoal_b", "zrand"):
                v = [x["ret_mean"] for x in R if x["drive"] == d]
                if v:
                    by[d] = float(np.mean(v))
            ceil = max(by.get("zgoal", -1e9), by.get("zgoal_b", -1e9))
            floor = by.get("zrand", 0.0)
            frac = (by["zr"] - floor) / max(ceil - floor, 1e-9)
            out.setdefault("summary", {})[fam] = {**by, "gap_frac": float(frac)}
            print(f"[B] {fam:>10}: zr={by['zr']:8.3f} ceil={ceil:8.3f} "
                  f"floor={floor:8.3f} gap_frac={frac:+.3f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
