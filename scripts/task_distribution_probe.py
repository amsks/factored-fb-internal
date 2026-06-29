"""scripts/task_distribution_probe.py — FB vs PSM training task-distribution probe.

Driver for the 3-probe analysis (see PAPER/specs/2026-06-07-fb-vs-psm-task-
distribution-analysis.md). Pure logic lives in evals/task_distribution.py
(unit-tested); this file does checkpoint embedding + IO/figures (TDD-exempt).

Run (CPU is fine):
  PYTHONPATH=. .venv/bin/python scripts/task_distribution_probe.py [--m 50000] [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.ogbench import load_ogbench_dataset
from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
from evals.ogbench import OGBenchEvaluator
from evals.task_distribution import (
    INTERACTION_BUCKETS,
    EFF_SLICE,
    CUBE_OBS,
    CONTACT_IDX,
    CUBE_PHYSICS,
    bucket_histogram,
    coverage_score,
    estimate_table_z,
    label_interaction,
    participation_ratio,
    separability_auc,
)

RUNS = "/dev/shm/factored-fb/runs"
DATA = "/dev/shm/factored-fb/datasets"
DOMAIN = "cube-single-play-v0"
OUT = Path("analysis/task_distribution")

# method -> (run-dir glob prefix, embed kind). Plain FB = canonical FB (see
# [[fb-state-run-name-map]]); both have 10 seeds.
METHODS = {
    "fb_state_plain": ("20260607_070225__fb_state_plain", "fb"),
    "psm_flowbc_state": ("20260603_071856__psm_flowbc_state", "psm"),
    "psm_o1000": ("psm_o1000", "psm"),  # tuned PSM ortho_coef=1000 (symlinked into RUNS)
    "fb_z128": ("fb_z128", "fb"),       # 2x2 capacity control: FB at z_dim=128 (symlinked)
    "fb_state_iwr": ("fb_state_iwr", "fb"),        # IWR rho' tilt (3 seeds @500k)
    "fb_state_iwr_wz": ("fb_state_iwr_wz", "fb"),  # IWR + weight_z
}


def embed_pool(model, next_obs: torch.Tensor, kind: str) -> np.ndarray:
    """Pool of next-states -> the method's training-z cloud.
    FB : project_z(backward_map(s'))   PSM: project_z(phi(s'))."""
    with torch.no_grad():
        if kind == "fb":
            z = model.backward_map(next_obs)
        elif kind == "psm":
            z = model.phi(next_obs)
        else:
            raise ValueError(kind)
        z = model.project_z(z)
    return z.cpu().numpy()


def build_pool(m_samples: int):
    """Shared, method-agnostic goal pool: next-states + interaction labels."""
    torch.manual_seed(0)
    buf = load_ogbench_dataset(domain=DOMAIN, data_path=DATA, obs_type="state",
                               load_n_episodes=1000)
    batch = buf.sample(min(m_samples, len(buf)))
    next_obs = torch.as_tensor(batch["next"]["observation"], dtype=torch.float32)
    obs_np = next_obs.numpy()
    cube_phys_z = np.asarray(batch["next"]["physics"])[:, CUBE_PHYSICS][:, 2]
    table_z = estimate_table_z(cube_phys_z)        # real metres
    lift = cube_phys_z - table_z
    # proximity within the (normalised) obs frame: effector vs cube, same frame
    labels = label_interaction(
        obs_np[:, EFF_SLICE], obs_np[:, CUBE_OBS], lift, obs_np[:, CONTACT_IDX])
    return buf, next_obs, labels, table_z


def run_seed(rundir: Path, kind: str, next_obs, labels, buf, relabel_size, n_eval_ep):
    cfg = load_cfg(str(rundir / ".hydra/config.yaml"), device="cpu")
    _, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, str(rundir / "checkpoints/final.pt"))
    agent.model.train(False)

    Z = embed_pool(agent.model, next_obs, kind)
    norms = np.linalg.norm(Z, axis=1)

    # interaction binary: cube-involved (grasped|lifted) vs free (no_contact|reaching)
    involved = labels >= 2

    # --- probe 3: eval-task z coverage against the training z cloud ---
    ev = OGBenchEvaluator(domain=DOMAIN, agent=agent, offline_buffer=buf,
                          relabel_size=relabel_size, n_episodes=n_eval_ep,
                          obs_type="state", seed=int(cfg.seed), device="cpu",
                          save_videos=False, use_wandb=False)
    eval_cov = {}
    for task in ev.tasks:
        z_eval, _ = ev._infer_z(task)
        z_eval = np.atleast_2d(z_eval.cpu().numpy())
        cov = coverage_score(Z, z_eval)
        eval_cov[task] = {"cos_nn": float(cov["cos_nn"][0]),
                          "mahalanobis": float(cov["mahalanobis"][0])}

    return {
        "z_dim": int(Z.shape[1]),
        # probe 1 (shared across seeds, recorded per seed for convenience)
        "bucket_fractions": bucket_histogram(labels),
        # probe 2: task-embedding geometry
        "participation_ratio": participation_ratio(Z),
        "z_norm_mean": float(norms.mean()),
        "z_norm_std": float(norms.std()),
        "separability_auc_involved": separability_auc(Z, involved),
        # probe 3
        "eval_coverage": eval_cov,
    }


def aggregate(per_seed: list[dict]) -> dict:
    def ms(key):
        v = np.array([d[key] for d in per_seed], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std())}
    tasks = list(per_seed[0]["eval_coverage"].keys())
    cov_cos = {t: ms_list([d["eval_coverage"][t]["cos_nn"] for d in per_seed]) for t in tasks}
    cov_mah = {t: ms_list([d["eval_coverage"][t]["mahalanobis"] for d in per_seed]) for t in tasks}
    return {
        "n_seeds": len(per_seed),
        "z_dim": per_seed[0]["z_dim"],
        "bucket_fractions": per_seed[0]["bucket_fractions"],
        "participation_ratio": ms("participation_ratio"),
        "z_norm_mean": ms("z_norm_mean"),
        "separability_auc_involved": ms("separability_auc_involved"),
        "eval_coverage_cos_nn": cov_cos,
        "eval_coverage_mahalanobis": cov_mah,
    }


def ms_list(vals):
    v = np.array(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std())}


def write_summary(agg: dict):
    lines = ["# FB vs PSM — training task-distribution probe\n"]
    lines.append("## Probe 1 — goal pool by interaction bucket (shared)\n")
    bf = next(iter(agg.values()))["bucket_fractions"]
    for b in INTERACTION_BUCKETS:
        lines.append(f"- {b}: {bf[b]:.3f}")
    lines.append(f"- **cube-involved (grasped+lifted): "
                 f"{bf['grasped'] + bf['lifted_or_transport']:.3f}**\n")
    lines.append("## Probe 2 — task-embedding geometry (mean±std over seeds)\n")
    lines.append("| method | z_dim | participation ratio | z-norm | separability AUC (involved vs free) |")
    lines.append("|---|---|---|---|---|")
    for method, a in agg.items():
        lines.append(
            f"| {method} | {a['z_dim']} | "
            f"{a['participation_ratio']['mean']:.2f}±{a['participation_ratio']['std']:.2f} | "
            f"{a['z_norm_mean']['mean']:.2f} | "
            f"{a['separability_auc_involved']['mean']:.3f}±{a['separability_auc_involved']['std']:.3f} |")
    lines.append("\n## Probe 3 — eval-task z coverage vs training cloud (mean over seeds)\n")
    for method, a in agg.items():
        lines.append(f"### {method}")
        lines.append("| task | cos-to-nearest | Mahalanobis |")
        lines.append("|---|---|---|")
        for t in a["eval_coverage_cos_nn"]:
            lines.append(f"| {t} | {a['eval_coverage_cos_nn'][t]['mean']:.3f} | "
                         f"{a['eval_coverage_mahalanobis'][t]['mean']:.2f} |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=50000, help="pool size (next-states)")
    ap.add_argument("--quick", action="store_true", help="2 seeds, small pool, fast")
    args = ap.parse_args()

    m = 4000 if args.quick else args.m
    relabel_size = 2000 if args.quick else 10000
    n_eval_ep = 1  # rollouts not used here; _infer_z ignores n_episodes
    max_seeds = 2 if args.quick else 10

    buf, next_obs, labels, table_z = build_pool(m)
    print(f"pool: {len(next_obs)} next-states, table_z={table_z:.4f}")
    print("buckets:", {k: round(v, 3) for k, v in bucket_histogram(labels).items()})

    agg_all = {}
    for method, (prefix, kind) in METHODS.items():
        seed_dirs = sorted(Path(RUNS).glob(f"{prefix}__s*"))[:max_seeds]
        outdir = OUT / method
        outdir.mkdir(parents=True, exist_ok=True)
        per_seed = []
        for d in seed_dirs:
            seed = d.name.split("__s")[-1]
            res = run_seed(d, kind, next_obs, labels, buf, relabel_size, n_eval_ep)
            (outdir / f"s{seed}.json").write_text(json.dumps(res, indent=2))
            per_seed.append(res)
            print(f"  {method} s{seed}: PR={res['participation_ratio']:.2f} "
                  f"AUC={res['separability_auc_involved']:.3f}")
        agg = aggregate(per_seed)
        (outdir / "aggregate.json").write_text(json.dumps(agg, indent=2))
        agg_all[method] = agg

    write_summary(agg_all)
    print(f"\nwrote {OUT}/summary.md")


if __name__ == "__main__":
    main()
