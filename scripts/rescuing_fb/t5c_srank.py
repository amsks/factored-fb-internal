#!/usr/bin/env python
"""scripts/rescuing_fb/t5c_srank.py -- ogbench-sota implementations.md item 3.

srank / effective-rank trajectories of the critic representation across training,
on EXISTING checkpoints. Bilinear archs: the forward feature phi(s,a) (head-mean
over the 2-head critic ensemble). MLP archs: the penultimate representation of the
critic value_net (input to the final scalar head), pulled via flax
capture_intermediates. Same FIXED 4096-transition batch (rng(0)) for every
checkpoint so the curves are directly comparable.

Metrics (documented, both reported):
  srank(delta=0.01)  -- Kumar et al. 2021 (arXiv:2010.14498) effective rank:
      sort singular values sigma_1>=...>=sigma_n of the CENTERED feature matrix,
      srank_delta = min{ k : (sum_{i<=k} sigma_i) / (sum_i sigma_i) >= 1 - delta }.
  participation_ratio -- PR = (sum lambda_i)^2 / sum lambda_i^2 with lambda_i =
      sigma_i^2 the covariance eigenvalues (the canonical PR; a smooth rank proxy).

Pre-registered prediction (phenomenon): the MLP penultimate srank COLLAPSES over
training (implicit under-parameterization) while the bilinear phi srank stays
>= the task-family rank (~28); the collapse should time-align with where the MLP's
frozen-critic rerank ordering goes flat. Either outcome sharpens Layer 1's
"absent, not noisy" clause.

Run (CPU-only):
  JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/t5c_srank.py \\
    --checkpoints '~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50/sd000_*' \\
                  '~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow/sd000_*'
  # richer trajectory (the T2/exp5 set with 50k->500k intermediate saves), if present:
  #   --checkpoints '<exp5-bilinear-glob>' '<exp5-mlp-glob>'  --steps 50000 100000 200000 300000 400000 500000
Each --checkpoints entry is a run-dir glob; arch (value_arch/latent) is read from the
dir's flags.json, and the available steps are auto-discovered from params_<step>.pkl
unless --steps is given. JSON -> analysis/rescuing-fb/t5c_srank.json, PNG alongside.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))

EVAL_TASK = "cube-single-play-singletask-task1-v0"
BILINEAR_ARCHES = {"bilinear", "bilinear_sg", "bilinear_sg_bounded", "latefusion",
                   "dueling", "simplesf", "bilinear_split"}


def load_run(run_dir: Path, step: int):
    """Restore a GCIQLFlowAgent using the arch recorded in the run's flags.json."""
    import jax.numpy as jnp
    import ogbench
    from gciql_flow import GCIQLFlowAgent, get_config
    from utils.flax_utils import restore_agent

    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(get_config()); cfg.update(flags["agent"])
    cfg["encoder"] = flags["agent"].get("encoder"); cfg["frame_stack"] = flags["agent"].get("frame_stack")
    env = ogbench.make_env_and_datasets(EVAL_TASK, env_only=True)
    ex_obs = jnp.asarray(env.reset()[0], jnp.float32)[None]
    ex_act = jnp.zeros((1, *env.action_space.shape), jnp.float32)
    if hasattr(env, "close"):
        env.close()
    agent = GCIQLFlowAgent.create(int(flags["seed"]), ex_obs, ex_act, cfg)
    agent = restore_agent(agent, str(run_dir), step)
    return agent, cfg


def fixed_batch(n=4096, seed=0):
    """A fixed transition batch from the task-1 dataset (same across checkpoints)."""
    import ogbench
    _e, ds, _ = ogbench.make_env_and_datasets(EVAL_TASK)
    obs = np.asarray(ds["observations"], np.float32)
    acts = np.asarray(ds["actions"], np.float32)
    idx = np.random.default_rng(seed).choice(len(obs), size=min(n, len(obs)), replace=False)
    return obs[idx], acts[idx]


def bilinear_phi(agent, obs, acts):
    import jax.numpy as jnp
    _v, phi, _psi = agent.network.select("critic")(
        jnp.asarray(obs), jnp.asarray(obs), jnp.asarray(acts), info=True)
    phi = np.asarray(phi)                       # [2,B,d] ensembled, or [B,d]
    if phi.ndim == 3:
        phi = phi.mean(axis=0)                  # head-mean -> [B,d]
    return phi


def mlp_penultimate(agent, obs, acts):
    """Penultimate representation of the MLP critic value_net (the input to the
    final scalar head). Primary path: a manual forward through the ensembled
    value_net params up to (but not including) the final Dense -- deterministic,
    and robust to the ensemble vmap that hides flax capture_intermediates.
    Returns [B, h_last] (head-mean over the 2-head ensemble). None if the layout
    is unexpected."""
    import jax.numpy as jnp
    B = obs.shape[0]
    x = np.concatenate([obs, obs, acts], axis=-1).astype(np.float32)   # [s, g(=s placeholder), a]
    try:
        vp = agent.network.params["modules_critic"]["value_net"]
    except Exception:
        return None
    # collect ensembled Dense params in numeric order (Dense_0, Dense_1, ... Dense_10).
    denses = sorted((k for k in vp if k.startswith("Dense_")),
                    key=lambda k: int(k.split("_")[1]))
    norms = {k: vp[k] for k in vp if k.startswith("LayerNorm")}
    norm_order = sorted(norms, key=lambda k: int(k.split("_")[1])) if norms else []
    if len(denses) < 2:
        return None
    h = np.broadcast_to(x[None], (2, B, x.shape[-1])).copy()           # [2, B, D]
    n_hidden = len(denses) - 1                                         # final Dense -> scalar
    for i in range(n_hidden):                                         # stop BEFORE the final head
        W = np.asarray(vp[denses[i]]["kernel"]); b = np.asarray(vp[denses[i]]["bias"])
        h = np.einsum("ebd,edh->ebh", h, W) + b[:, None, :]
        h = np.asarray(jax_gelu(h))
        if i < len(norm_order):
            g = norms[norm_order[i]]
            sc = np.asarray(g["scale"]); bi = np.asarray(g["bias"])
            mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
            h = (h - mu) / np.sqrt(var + 1e-6) * sc[:, None, :] + bi[:, None, :]
    return h.mean(axis=0)                                             # head-mean -> [B, h_last]


def jax_gelu(x):
    import jax.nn as jnn
    import jax.numpy as jnp
    return jnn.gelu(jnp.asarray(x))


def srank_delta(feats, delta=0.01):
    f = feats - feats.mean(axis=0, keepdims=True)
    s = np.linalg.svd(f, compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return 0
    frac = np.cumsum(s) / s.sum()
    return int(np.searchsorted(frac, 1.0 - delta) + 1)


def participation_ratio(feats):
    f = feats - feats.mean(axis=0, keepdims=True)
    s = np.linalg.svd(f, compute_uv=False)
    lam = s ** 2
    denom = (lam ** 2).sum()
    return float((lam.sum() ** 2) / denom) if denom > 0 else 0.0


def discover_steps(run_dir: Path):
    steps = []
    for p in glob.glob(str(run_dir / "params_*.pkl")):
        m = re.search(r"params_(\d+)\.pkl", p)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="run-dir glob(s); arch read from each dir's flags.json")
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    help="checkpoint steps (default: auto-discover per dir)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--out", default=str(REPO / "analysis/rescuing-fb/t5c_srank.json"))
    args = ap.parse_args()

    obs, acts = fixed_batch(args.batch)
    print(f"[t5c] fixed batch {obs.shape} acts {acts.shape}", flush=True)

    run_dirs = sorted({Path(p).resolve() for pat in args.checkpoints
                       for p in glob.glob(str(Path(pat).expanduser()))})
    run_dirs = [d for d in run_dirs if (d / "flags.json").exists()]
    assert run_dirs, f"no run dirs with flags.json matched {args.checkpoints}"

    results = {"meta": {"batch": int(obs.shape[0]), "delta": args.delta,
                        "srank_def": "Kumar2021 min k: cumsum(sigma)/sum(sigma) >= 1-delta",
                        "pr_def": "(sum lambda)^2 / sum lambda^2, lambda=sigma^2 (covariance eig)",
                        "eval_task": EVAL_TASK}, "runs": []}
    for rd in run_dirs:
        flags = json.loads((rd / "flags.json").read_text())
        arch = flags["agent"].get("value_arch", "mlp")
        group = flags.get("run_group", rd.parent.name)
        is_bil = arch in BILINEAR_ARCHES
        steps = args.steps or discover_steps(rd)
        for step in steps:
            if not (rd / f"params_{step}.pkl").exists():
                continue
            try:
                agent, _cfg = load_run(rd, step)
                feats = bilinear_phi(agent, obs, acts) if is_bil else mlp_penultimate(agent, obs, acts)
                if feats is None:
                    print(f"[t5c] {group}/{rd.name}@{step}: no penultimate feats", flush=True)
                    continue
                rec = {"group": group, "run": rd.name, "step": int(step), "arch": arch,
                       "feat_kind": "phi" if is_bil else "mlp_penultimate",
                       "feat_dim": int(feats.shape[-1]),
                       "srank": srank_delta(feats, args.delta),
                       "participation_ratio": participation_ratio(feats)}
                results["runs"].append(rec)
                print(f"[t5c] {group:38s} {rd.name} @{step//1000:>4d}k  arch={arch:10s} "
                      f"srank={rec['srank']:>4d}  PR={rec['participation_ratio']:.1f} "
                      f"(dim={rec['feat_dim']})", flush=True)
            except Exception as e:
                print(f"[t5c] FAILED {group}/{rd.name}@{step}: {type(e).__name__}: {e}", flush=True)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))

    # PNG: srank vs step, one line per (group, feat_kind), seeds averaged.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import defaultdict
        curves = defaultdict(lambda: defaultdict(list))   # label -> step -> [srank]
        for r in results["runs"]:
            curves[f"{r['arch']} ({r['feat_kind']})"][r["step"]].append(r["srank"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, byz in sorted(curves.items()):
            xs = sorted(byz)
            ys = [float(np.mean(byz[s])) for s in xs]
            ax.plot([x / 1000 for x in xs], ys, marker="o", label=label)
        ax.axhline(28, ls="--", c="gray", lw=1, label="task-family rank ~28")
        ax.set_xlabel("training step (k)"); ax.set_ylabel(f"srank(δ={args.delta})")
        ax.set_title("Critic representation srank over training")
        ax.legend(fontsize=8); fig.tight_layout()
        png = outp.with_suffix(".png"); fig.savefig(png, dpi=130)
        print(f"[t5c] wrote {outp} + {png}", flush=True)
    except Exception as e:
        print(f"[t5c] plot skipped ({type(e).__name__}: {e}); JSON at {outp}", flush=True)
    return 0


if __name__ == "__main__":
    import jax  # noqa: F401  (ensure jax present for the feature helpers)
    sys.exit(main())
