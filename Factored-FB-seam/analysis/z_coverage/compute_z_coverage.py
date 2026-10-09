"""Where do the actor's TRAINING task vectors z lie, and where do the EVALUATION z's land?

Per env: load a density_fb_z checkpoint the way impls/main.py does (make_config from the
run's recorded config, Agent.create, restore_agent params_only), then build

  (a) goal z    -- 4096 hindsight-goal indices EXACTLY as training builds them: the critic's
                   own `sample_cond` on real GCDataset batches (actor_goals = same-trajectory
                   futures), with z_rand_frac forced to 0 so every row keeps the encoded index;
  (b) sphere z  -- 4096 uniform sqrt(d)-sphere draws (the other half of the training mix);
  (c) eval z    -- the 5 Route-B indices: 10k relabelled (next_obs, reward) rows from
                   OGBenchAdapter.relabel_infer_batch (shift 1.0) through critic.infer_cond,
                   i.e. the same call impls/main.py's eval_relabel block makes;
  (d) k-hot z   -- 5 indices for r = 1[all blocks in a random non-empty subset of the hit-spec
                   blocks are within eps of a random dataset state's block coords] else 0;
      dense z   -- 5 indices for r = -clip(d_eps / median_pool(d_eps), 0, 1), d_eps the
                   eps-scaled Euclidean distance of the block coords to a random goal state.
      Both go through the identical reward-sample index (z_from_reward_samples) on one 10k
      next-state pool drawn like Route B's.

Writes analysis/z_coverage/z_<env>[_exact].npz and appends rows to coverage.csv.

Eval-only: forward passes, no parameter writes. Nothing in impls/ or env/ is patched.

Usage (GPU, tmux `zcov`):
  CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    /dev/shm/.venv-jax/bin/python -m analysis.z_coverage.compute_z_coverage --set campaign
  ... --set exact      # the 500k zr=0 exact-reward checkpoints under exp/kz_conf
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import jax
import jax.numpy as jnp

from env import get_adapter, make_config
from impls.agents import Agent
from impls.critics.base import infer_cond_rows
from impls.critics.density_fb import _raw_block_cols, _sphere
from impls.utils.datasets import make_dataset
from impls.utils.flax_utils import restore_agent
from scripts.reeval_checkpoint import _recorded_config_overrides

HUB = ("/tmp/claude-1001/-home-adityamohan-git-Factored-FB-seam/"
       "678b32e6-6347-4028-b719-b8c7b248a0ea/scratchpad/zcov_ckpt")
OUT = os.path.join(ROOT, "analysis", "z_coverage")

# env key -> (domain, run_dir, step, seed). Campaign = 1M density_fb_z latent_centered zr=.5
# (HF amsks/factored-fb-checkpoints exp/<run>); antmaze has only seed 2 on the hub.
CAMPAIGN = {
    "cube-single": ("cube_single", f"{HUB}/cube-single-play-v0_density_fb_z_flowbc_seed_0", 1_000_000, 0),
    "cube-double": ("cube_double", f"{HUB}/cube-double-play-v0_density_fb_z_flowbc_seed_0", 1_000_000, 0),
    "scene":       ("scene_play", f"{HUB}/scene-play-v0_density_fb_z_flowbc_seed_0", 1_000_000, 0),
    "puzzle-3x3":  ("puzzle_3x3", f"{HUB}/puzzle-3x3-play-v0_density_fb_z_flowbc_seed_0", 1_000_000, 0),
    "antmaze-medium": ("antmaze_medium_navigate",
                       f"{HUB}/antmaze-medium-navigate-v0_density_fb_z_flowbc_seed_2", 1_000_000, 2),
}
# Exact-reward (kernel actor reward, z_rand_frac=0) 500k checkpoints: the actor trained on
# goal tasks ONLY. cube seeds are 1 there (seed 0 was not run under kz_conf).
EXACT = {
    "cube-single": ("cube_single", f"{ROOT}/exp/kz_conf/cs_s1/cube-single-play-v0_density_fb_z_flowbc_seed_1", 500_000, 1),
    "cube-double": ("cube_double", f"{ROOT}/exp/kz_conf/cd_s1/cube-double-play-v0_density_fb_z_flowbc_seed_1", 500_000, 1),
    "scene":       ("scene_play", f"{ROOT}/exp/kz_conf/scene_s0/scene-play-v0_density_fb_z_flowbc_seed_0", 500_000, 0),
    "puzzle-3x3":  ("puzzle_3x3", f"{ROOT}/exp/kz_conf/puzzle_s0/puzzle-3x3-play-v0_density_fb_z_flowbc_seed_0", 500_000, 0),
}

N_GOAL, N_SPHERE, BATCH, N_POOL, N_FAM = 4096, 4096, 1024, 10_000, 5


def load(domain, run_dir, step, seed):
    base = _recorded_config_overrides(run_dir)
    cfg = make_config("density_fb_z", "flowbc", domain, base)
    assert cfg.get("eval_relabel_size"), "the recorded config must carry eval_relabel_size (qpos loading)"
    adapter = get_adapter(cfg)
    env, train_ds, _ = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create("density_fb_z", "flowbc", seed, jnp.asarray(ex["observations"]),
                         jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, run_dir, step, params_only=True)
    return agent, adapter, env, train_ds, cfg


def goal_zs(agent, train_ds, n, rng):
    """(a): the critic's own sample_cond, z_rand_frac forced to 0 -> every actor row keeps
    the encoded hindsight-goal index. Also returns the actor goals' block coords."""
    critic0 = dataclasses.replace(agent.critic, z_rand_frac=0.0)
    zs, hs = [], []
    while sum(len(z) for z in zs) < n:
        batch = {k: jnp.asarray(v) for k, v in train_ds.sample(BATCH).items()}
        rng, r = jax.random.split(rng)
        cond = critic0.sample_cond(agent.net, batch, r, target=agent.target)
        zs.append(np.asarray(cond["z"], np.float64))
        hs.append(np.asarray(cond["_z_hindsight"], np.float64).ravel())
    z, h = np.concatenate(zs)[:n], np.concatenate(hs)[:n]
    assert np.all(h > 0.5), "z_rand_frac=0 must keep every encoded index"
    return z


def sphere_zs(d, n, rng):
    return np.asarray(_sphere(jax.random.normal(rng, (n, d)), d), np.float64)


def eval_zs(agent, adapter, env, train_ds, cfg):
    """(c): Route B exactly -- relabel_infer_batch(10k, shift 1.0) -> infer_cond."""
    out, meta = [], []
    n_tasks = len(env.unwrapped.task_infos)
    for tid in range(1, n_tasks + 1):
        obs, rew = adapter.relabel_infer_batch(train_ds, env, tid, int(cfg["eval_relabel_size"]),
                                               shift_reward=float(cfg.get("eval_relabel_shift", 1.0)))
        z = agent.critic.infer_cond(agent.net, jnp.asarray(obs), jnp.asarray(rew))["z"]
        out.append(np.asarray(z, np.float64))
        meta.append({"task": tid, "n_pos": int((rew >= 1.0 - 1e-6).sum()),
                     "r_min": float(rew.min()), "r_max": float(rew.max()),
                     "r_mean": float(rew.mean()), "n_distinct": int(len(np.unique(rew)))})
    return np.stack(out), meta


def family_zs(agent, train_ds, blocks, rng):
    """(d): k-hot and dense-bounded rewards on ONE 10k next-state pool drawn like Route B."""
    base = adapter_base(train_ds)
    idxs = base.get_random_idxs(N_POOL)
    nidx = np.minimum(idxs + 1, base.size - 1)
    pool = np.asarray(base["observations"][nidx], np.float32)                 # [M, obs]
    pool_cols = np.asarray(_raw_block_cols(pool, blocks))                     # [M, d_c]
    widths = [b - a for (a, b, e) in blocks]
    eps = np.concatenate([[e] * w for (a, b, e), w in zip(blocks, widths)])
    offs = np.cumsum([0] + widths)
    goals_all = np.asarray(base["observations"], np.float32)

    def solve(r):
        u_sc = agent.critic._b_query(agent.net, jnp.asarray(pool), jnp.asarray(pool))
        z = agent.critic.z_from_reward_samples(agent.net, u_sc, jnp.asarray(r, jnp.float32))
        return np.asarray(z, np.float64)[0]

    khot, kmeta = [], []
    n_blk = len(blocks)
    for _ in range(N_FAM):
        for attempt in range(200):
            g = goals_all[rng.integers(base.size)]
            gcols = np.asarray(_raw_block_cols(g[None], blocks))[0]
            k = int(rng.integers(1, n_blk + 1))
            sub = sorted(rng.choice(n_blk, size=k, replace=False).tolist())
            hit = np.ones(N_POOL, bool)
            for j in sub:
                a, b = offs[j], offs[j + 1]
                hit &= np.linalg.norm(pool_cols[:, a:b] - gcols[a:b], axis=-1) < blocks[j][2]
            if hit.sum() >= 5:
                break
        r = hit.astype(np.float32)
        khot.append(solve(r))
        kmeta.append({"k": k, "blocks": sub, "n_pos": int(hit.sum()), "attempts": attempt + 1})

    dense, dmeta = [], []
    for _ in range(N_FAM):
        g = goals_all[rng.integers(base.size)]
        gcols = np.asarray(_raw_block_cols(g[None], blocks))[0]
        dist = np.linalg.norm((pool_cols - gcols) / eps, axis=-1)
        r = -np.clip(dist / np.median(dist), 0.0, 1.0).astype(np.float32)
        dense.append(solve(r))
        dmeta.append({"median_d_eps": float(np.median(dist)), "frac_unclipped": float((dist < np.median(dist)).mean())})
    return np.stack(khot), kmeta, np.stack(dense), dmeta


def adapter_base(ds):
    cur = ds
    for _ in range(8):
        try:
            if "observations" in cur:
                return cur
        except TypeError:
            pass
        cur = getattr(cur, "dataset", cur)
    return cur


def metrics(zg, zs, ze):
    """Per eval z: nearest-goal distance in goal-cloud RMS-radius units, cosine to that
    nearest goal z; the same to the sphere cloud; plus the goal cloud's own median NN
    distance (leave-one-out) as the 'inside' yardstick."""
    def nn(cloud, q):
        d2 = (q ** 2).sum(-1)[:, None] + (cloud ** 2).sum(-1)[None] - 2 * q @ cloud.T
        j = np.argmin(d2, axis=1)
        d = np.sqrt(np.maximum(d2[np.arange(len(q)), j], 0))
        cos = (q * cloud[j]).sum(-1) / (np.linalg.norm(q, axis=-1) * np.linalg.norm(cloud[j], axis=-1))
        return d, cos
    rms = np.sqrt(((zg - zg.mean(0)) ** 2).sum(-1).mean())
    # goal cloud's own NN distance (leave-one-out) on a 2048 subsample
    sub = zg[:2048]
    d2 = (sub ** 2).sum(-1)[:, None] + (zg ** 2).sum(-1)[None] - 2 * sub @ zg.T
    d2[np.arange(len(sub)), np.arange(len(sub))] = np.inf
    goal_nn = np.sqrt(np.maximum(d2.min(1), 0))
    goal_nn_med = float(np.median(goal_nn))
    dg, cg = nn(zg, ze)
    ds_, cs = nn(zs, ze)
    # Density-style yardsticks, both as PERCENTILES within the goal cloud's own distribution
    # (0 = deeper inside than every goal z, 100 = farther out than every goal z):
    #   centroid_pct: distance to the goal-cloud mean;
    #   knn10_pct:    mean distance to the 10 nearest goal z's (leave-one-out for the cloud).
    dc_goal = np.sqrt(((zg - zg.mean(0)) ** 2).sum(-1))
    dc_eval = np.sqrt(((ze - zg.mean(0)) ** 2).sum(-1))
    centroid_pct = np.array([(dc_goal < x).mean() * 100 for x in dc_eval])
    d2s = np.sort(d2, axis=1)[:, :10]
    knn_goal = np.sqrt(np.maximum(d2s, 0)).mean(1)
    d2e = (ze ** 2).sum(-1)[:, None] + (zg ** 2).sum(-1)[None] - 2 * ze @ zg.T
    knn_eval = np.sqrt(np.maximum(np.sort(d2e, axis=1)[:, :10], 0)).mean(1)
    knn10_pct = np.array([(knn_goal < x).mean() * 100 for x in knn_eval])
    return {"rms": float(rms), "goal_nn_median_abs": goal_nn_med, "goal_nn_median_rms": goal_nn_med / rms,
            "d_goal_rms": dg / rms, "cos_goal": cg, "d_sphere_rms": ds_ / rms, "cos_sphere": cs,
            "inside": (dg < goal_nn_med).astype(float),
            "centroid_pct": centroid_pct, "knn10_pct": knn10_pct}


def run_env(key, spec, tag, csv_rows):
    domain, run_dir, step, seed = spec
    t0 = time.time()
    print(f"\n=== {key} [{tag}] {run_dir} @ {step} seed {seed}", flush=True)
    agent, adapter, env, train_ds, cfg = load(domain, run_dir, step, seed)
    critic = agent.critic
    d, blocks = int(critic.z_dim), critic.blocks
    assert critic.z_on and getattr(critic, "INFER_COND_MARGINAL", None) == "next_observations"
    print(f"    hit_spec={cfg['measure_hit_spec']} d={d} z_rand_frac={cfg['measure_density_z_rand_frac']} "
          f"actor_reward={cfg['measure_density_z_actor_reward']} rows={infer_cond_rows(critic)}", flush=True)
    np.random.seed(seed)                          # GCDataset's global-RNG stream, as main.py seeds it
    jr = jax.random.PRNGKey(seed)
    jr_goal, jr_sphere = jax.random.split(jr)
    zg = goal_zs(agent, train_ds, N_GOAL, jr_goal)
    zs = sphere_zs(d, N_SPHERE, jr_sphere)
    np.random.seed(seed)                          # Route B draws from the same global stream
    ze, emeta = eval_zs(agent, adapter, env, train_ds, cfg)
    zk, kmeta, zd, dmeta = family_zs(agent, train_ds, blocks, np.random.default_rng(seed + 1000))
    m = {fam: metrics(zg, zs, z) for fam, z in (("eval", ze), ("khot", zk), ("dense", zd))}
    for fam, z, metas in (("eval", ze, emeta), ("khot", zk, kmeta), ("dense", zd, dmeta)):
        mm = m[fam]
        for i in range(len(z)):
            csv_rows.append({
                "env": key, "set": tag, "family": fam, "task": i + 1, "step": step, "seed": seed,
                "d_goal_rms": round(float(mm["d_goal_rms"][i]), 4), "cos_goal": round(float(mm["cos_goal"][i]), 4),
                "d_sphere_rms": round(float(mm["d_sphere_rms"][i]), 4), "cos_sphere": round(float(mm["cos_sphere"][i]), 4),
                "inside_goal_region": int(mm["inside"][i]), "goal_nn_median_rms": round(mm["goal_nn_median_rms"], 4),
                "centroid_pct": round(float(mm["centroid_pct"][i]), 1), "knn10_pct": round(float(mm["knn10_pct"][i]), 1),
                "goal_rms_radius": round(mm["rms"], 4), "z_norm": round(float(np.linalg.norm(z[i])), 3),
                "is_neutral": int(np.allclose(z[i], 1.0, atol=1e-6)),
                "meta": json.dumps(metas[i])})
    frac = {fam: float(m[fam]["inside"].mean()) for fam in m}
    med = {fam: float(np.median(m[fam]["d_goal_rms"])) for fam in m}
    print(f"    eval d_goal/rms per task: {np.round(m['eval']['d_goal_rms'], 3).tolist()} median {med['eval']:.3f} "
          f"inside {frac['eval']:.2f} | khot median {med['khot']:.3f} inside {frac['khot']:.2f} | "
          f"dense median {med['dense']:.3f} inside {frac['dense']:.2f} | goal NN median/rms {m['eval']['goal_nn_median_rms']:.3f}",
          flush=True)
    print(f"    eval centroid_pct {np.round(m['eval']['centroid_pct'], 1).tolist()} knn10_pct {np.round(m['eval']['knn10_pct'], 1).tolist()}",
          flush=True)
    print(f"    eval meta: {emeta}", flush=True)
    print(f"    khot meta: {kmeta}\n    dense meta: {dmeta}", flush=True)
    suffix = "" if tag == "campaign" else f"_{tag}"
    np.savez(os.path.join(OUT, f"z_{key}{suffix}.npz"), z_goal=zg, z_sphere=zs, z_eval=ze, z_khot=zk, z_dense=zd,
             labels_eval=np.array([f"task{i+1}" for i in range(len(ze))]),
             labels_khot=np.array([f"khot{i+1}" for i in range(len(zk))]),
             labels_dense=np.array([f"dense{i+1}" for i in range(len(zd))]),
             meta=json.dumps({"env": key, "set": tag, "run_dir": run_dir, "step": step, "seed": seed,
                              "hit_spec": cfg["measure_hit_spec"], "d": d,
                              "z_rand_frac": cfg["measure_density_z_rand_frac"],
                              "actor_reward": cfg["measure_density_z_actor_reward"],
                              "eval_meta": emeta, "khot_meta": kmeta, "dense_meta": dmeta,
                              "goal_rms": m["eval"]["rms"], "goal_nn_median_rms": m["eval"]["goal_nn_median_rms"],
                              "median_d_goal_rms": med, "frac_inside": frac,
                              "eval_centroid_pct": m["eval"]["centroid_pct"].tolist(),
                              "eval_knn10_pct": m["eval"]["knn10_pct"].tolist()}))
    print(f"    done in {time.time() - t0:.0f}s", flush=True)
    del agent, env, train_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="campaign", choices=["campaign", "exact"])
    ap.add_argument("--envs", default="")
    a = ap.parse_args()
    table = CAMPAIGN if a.set == "campaign" else EXACT
    envs = a.envs.split(",") if a.envs else list(table)
    rows = []
    for k in envs:
        run_env(k, table[k], a.set, rows)
        write_csv(rows, a.set)
    print("ALL DONE", flush=True)


def write_csv(rows, tag):
    """coverage.csv holds every set; rewrite this set's rows, keep the others."""
    path = os.path.join(OUT, "coverage.csv")
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r["set"] != tag]
    fields = list(rows[0].keys()) if rows else (list(old[0].keys()) if old else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in old + rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
