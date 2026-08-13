#!/usr/bin/env python
"""scripts/rescuing_fb/eval_density_z_rewards.py -- spec 24 sec.3 E2:
reward-to-index assignment for `measure_density_z` checkpoints on
point-mass-maze (PAPER/Multistep-FB/specs/24-z-indexed-density.md).

Context: spec 24 swaps the density measure's (spec 18) policy-family index
from goals to an FB-style latent z (`measure_density_z=True`); E1 (direct
goal-encoding, z* = z(phi(g_eval))) already ran and landed at the BC floor
(0.200 on point-mass-maze). E2 asks a narrower question: even if DIRECT goal
encoding fails, does the trained density estimator under z still support
REWARD READOUT -- i.e. is there SOME z whose predicted occupancy scores well
against an arbitrary reward, reached either analytically (E2a) or by search
(E2b)? Per spec 24 G3: E2a =~ E2b => the linear z_r embedding survives the
estimator; E2b >> E2a => the index is fine but the analytic assignment isn't;
both ~= the E1 floor => reward-to-index does not work under z at all.

This script GENERALIZES eval_density_zeroshot.py (spec 18 sec.6's g-indexed
zero-shot harness -- read first) to z-indexing. Per the task brief, it
IMPORTS AND REUSES that script's readout algebra verbatim
(`density_weights`, `compute_density_primary`, `compute_density_secondary`,
`build_pool`, `density_query`, `check_density_consistency`) rather than
reimplementing any of it -- the only new pieces are (a) the z-construction
functions spec 24 sec.3 defines (E2a analytic, E2b search) and (b) a
point-mass-maze rollout loop, since eval_density_zeroshot.py's own rollout
(`batched_goal_rollout_density`) is manipspace-specific (reads mujoco
`data.qpos[14:17]` for cube xyz and goal-conditions via `sample_actions`,
neither of which applies to a 2D point-mass whose actor is z-conditioned via
`sample_actions_with_z`).

TOWER CONVENTION (spec 24 sec.1, matched exactly): the w/kappa READOUT
(density_weights, compute_density_primary/secondary, kernel term) always
uses the ONLINE `density_b` tower -- the SAME tower `build_pool` (reused
from eval_density_zeroshot) already uses, unchanged. Any REWARD-EMBEDDING
use of B (the z_r construction itself, and -- internally, via the agent's
own `_density_z_encode` -- the E2b candidate z's) uses the TARGET
`target_density_b` tower, stop-gradded at training time, mirroring
`_density_actor_q_z`'s `tb24`/`rz24` split in gciql_flow.py exactly. This
script therefore builds TWO B-pools from the SAME pool observations: the
reused `build_pool` (online) for readout, and `build_target_b_pool` (new,
below) for E2a's reward embedding.

TWO ARMS (spec 24 sec.3), one JSON per run (`--arm` selects which):
  E2a analytic: z_r = sqrt(d) * normalize( E_pool[ r(x) * B_bar(x_tilde) ] )
    over the pool -- FB's own linear reward embedding, using the SAME
    sqrt(d)/eps-normalize convention as `gciql_flow._sphere`
    (x*sqrt(d)/(||x||+1e-6)) and the agent's own `target_density_b` tower
    (no reimplementation of the transform -- see `build_target_b_pool`).
  E2b search: candidates {z(u_j)} for pool samples u_j via the agent's OWN
    `_density_z_encode` (E1's transform, called live -- not rewritten here),
    scored by the pool PRIMARY readout Q_r(s0, .) (the SAME
    `compute_density_primary`/`density_weights` this script imports), argmax
    selects z*. Candidates are drawn from a SEPARATE, smaller sample
    (`--e2b_candidate_size`, default 4096) than the P=16384 readout pool --
    a deliberate, documented reduction from the g-indexed retrieval decode's
    default relabel_size=50000 (see `decode_goal_retrieval_density` in
    eval_density_zeroshot.py), because scoring N candidates against a
    P=16384-query pool is an [N,P] matrix (2 heads, float64): N=50000 would
    be ~13GB, N=4096 is ~1GB. Reported in JSON meta so the reduction is
    auditable, not silent.

Both arms then run a PLAIN rollout for the whole episode conditioned on the
selected z (`sample_actions_with_z`, no per-step re-scoring -- same
"retrieval decode, then fixed rollout" shape as eval_density_zeroshot.py's
`--decode retrieval` path).

REWARD FAMILIES (spec 24 sec.3: "sparse/dense/held-out-region ... the eval
script generalizes; add maze hit_spec"): the SAME three families as
eval_density_zeroshot.py (FAMILIES = dirac/dense/region), renamed in the
spec's own prose as sparse=dirac, held-out-region=region:
  dirac  -- goal-conditioned indicator at one of the published 20 maze goals
            (`MAZE_20_GOALS`, url_benchmark/goals.py's MazeMultiGoal, the
            SAME goal set + GOAL_EPS=0.03=hit_spec eps that
            point_mass_maze_env.py already documents and z24's own hit_spec
            "0:2:0.03" matches) -- `reward_conj_dirac`, imported unchanged.
  dense  -- `reward_conj_dense` at the same goal centers, imported unchanged.
  region -- hindsight dataset centers via `draw_region_centers_with_obs`
            (imported unchanged; fully domain-generic, no manipspace
            assumption anywhere in it).
SUCCESS for all three families is `region_success_from_obs` (imported
unchanged) -- also fully generic (obs_traj/lengths/center/spec only), so
ALL family success numbers here are computed the SAME way, unlike
eval_density_zeroshot.py which uses the env's native info['success'] for
dirac/dense (point-mass-maze's own env success is keyed to the ENV's
internal task_id goal, not this script's reward-family center, so that
shortcut does not apply here -- using region_success_from_obs uniformly is
the honest generalization).

ENV LOADING: point-mass-maze is NOT an OGBench env -- eval_density_zeroshot.py's
`ogbench.make_env_and_datasets(...)` (the ogbench PACKAGE's own loader) does
not know about it. Training routes through `utils.env_utils.make_env_and_datasets`,
monkeypatched by `tools/wandb_mode_shim/exorl_env.py` (installed automatically
inside a real training subprocess via sitecustomize.py, which this standalone
script does NOT go through) -- so this script calls `exorl_env.install()`
explicitly before importing `utils.env_utils`, then builds per-episode env
instances via `exorl_env._ExoRLPointMassMazeEnv` directly (the same class
`make_env_and_datasets` builds, avoiding a redundant dataset reload per env).

CONTROL ARM (`--arm control`, added 2026-08-13 on coordinator request): the
task's original anchors (density .315, GCIQL .242, BC .235) are all
GOAL-conditioned NATIVE ENV SUCCESS numbers -- comparing E2a/E2b's
`region_success_from_obs` reads against them is a protocol mismatch (this
project has been bitten by exactly this kind of mismatch before). `--arm
control` runs a G-INDEXED checkpoint (`measure_density_z` absent/False)
through the SAME families/pool/episode-count/metric as E2a/E2b, using
eval_density_zeroshot.py's own retrieval decode (`decode_goal_retrieval_density`,
imported unchanged) to pick the deployment goal instead of any z. This makes
E2a/E2b directly comparable to a same-estimator, same-metric g-indexed
number, and is the correct baseline for the G3 verdict.

CONSISTENCY REFUSAL: `check_density_z_consistency` wraps the reused
`check_density_consistency` (measure_density=True required) AND additionally
refuses a checkpoint without `measure_density_z=True` (this script's
E2a/E2b readouts assume a z-wide F/actor architecture; scoring a g-indexed
checkpoint here would crash on a shape mismatch or silently score the wrong
architecture) -- `--force` overrides with a loud warning, mirroring the
base function's two-branch shape exactly. `check_control_consistency` is the
INVERSE for `--arm control` (refuses a checkpoint WITH `measure_density_z=True`),
so neither arm can accidentally run the other's checkpoint type.

Run (CPU-only; GPU training jobs are live on the box -- do not touch them):
  taskset -c 24-39 env OMP_NUM_THREADS=4 \\
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_density_z_rewards.py \\
    --run-dir '/home/adityamohan/exp_spec24/factored-fb/GCIQL-PointMassMaze-RND-Flow-Measure-DensityZ-d50/z24_pmm_z_s0' \\
    --checkpoint-step 500000 --arm e2a --pool_size 16384 --episodes 25 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "rescuing_fb"))

# Reused, not reimplemented (module docstring "TOWER CONVENTION" / "REWARD
# FAMILIES" above spell out exactly what each import is for).
from eval_density_zeroshot import (  # noqa: E402
    DensityMismatchError,
    build_pool,
    check_density_consistency,
    compute_density_primary,
    compute_density_secondary,
    decode_goal_retrieval_density,
    density_query,
    density_weights,
)
from eval_partition_zeroshot import draw_region_centers_with_obs  # noqa: E402
from eval_zeroshot_rewards import (  # noqa: E402
    reward_conj_dense,
    reward_conj_dirac,
    region_success_from_obs,
)
from env_geometry import HitSpec, episode_bounds  # noqa: E402

FAMILIES = ("dirac", "dense", "region")
# "control" (added on coordinator request, 2026-08-13): a G-INDEXED readout on the
# SAME env/families/pool/episode-count as e2a/e2b, using eval_density_zeroshot.py's
# own retrieval decode (decode_goal_retrieval_density, imported unchanged) instead of
# any z-construction. Exists because the E2a/E2b anchors quoted in the task brief
# (density .315, GCIQL .242, BC .235) are all GOAL-conditioned numbers from a
# DIFFERENT metric (native env success) -- comparing E2's region_success_from_obs
# reads against them is a protocol mismatch. This arm makes the comparison
# apples-to-apples: same metric, same families, same pool, on a measure_density_z=False
# checkpoint of the SAME estimator family.
ARMS = ("e2a", "e2b", "control")
EXORL_ENV_NAME = "exorl:point_mass_maze_rnd_seed0"


# ======================= consistency refusal =======================================

class DensityZMismatchError(DensityMismatchError):
    """The checkpoint's recorded `measure_density_z` config disagrees with
    what this script needs (spec 24: z-wide F/actor architecture). Subclasses
    `DensityMismatchError` so callers that already catch the base class (or
    `RuntimeError`) still catch this."""


def check_density_z_consistency(ckpt_is_density: bool, ckpt_is_density_z: bool,
                                force: bool = False, run_dir: str = "<run>") -> list:
    """`check_density_consistency` (measure_density=True required, reused
    unchanged) PLUS an additional refusal if `measure_density_z` is not set
    -- mirrors that function's exact warn-vs-raise / force shape, extended
    with one more branch. Returns accumulated warning strings; raises
    DensityZMismatchError on an un-forced mismatch in EITHER check."""
    warnings = list(check_density_consistency(ckpt_is_density, True, force=force,
                                               run_dir=run_dir))
    if not ckpt_is_density_z:
        msg = (f"checkpoint {run_dir} was NOT trained with measure_density_z=True "
              "(spec 24) -- this script's E2a/E2b readouts assume the z-conditioned "
              "density_f/actor architecture (F/actor input width = "
              "measure_density_dim, not obs/goal width); evaluating a g-indexed "
              "(spec 18) checkpoint here would crash on a shape mismatch or "
              "silently score against the wrong architecture.")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are LIKELY GARBAGE): " + msg)
        else:
            raise DensityZMismatchError(msg + " Pass --force to override (not recommended).")
    return warnings


def check_control_consistency(ckpt_is_density: bool, ckpt_is_density_z: bool,
                              force: bool = False, run_dir: str = "<run>") -> list:
    """The control arm's mirror of `check_density_z_consistency`: same base
    `measure_density=True` requirement (reused), but INVERTED on the z flag --
    refuses a checkpoint that WAS trained with `measure_density_z=True`,
    rather than one that wasn't. This is deliberate, not `not
    check_density_z_consistency(...)`: the two functions must independently
    refuse in BOTH directions (a z-indexed checkpoint cannot be run through
    `--arm control`, and -- by `check_density_z_consistency`, unchanged -- a
    g-indexed checkpoint cannot be run through `--arm e2a`/`e2b`), so a
    checkpoint/arm mismatch is caught no matter which side is wrong. Mirrors
    the exact warn-vs-raise / force shape of every consistency check in this
    harness."""
    warnings = list(check_density_consistency(ckpt_is_density, True, force=force,
                                               run_dir=run_dir))
    if ckpt_is_density_z:
        msg = (f"checkpoint {run_dir} WAS trained with measure_density_z=True -- "
              "the control arm exists to give E2a/E2b a G-INDEXED comparison point "
              "(spec 24 G3, coordinator request 2026-08-13) and requires "
              "measure_density_z absent/False; running a z-indexed checkpoint "
              "through --arm control would not be a control at all.")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are LIKELY GARBAGE): " + msg)
        else:
            raise DensityZMismatchError(msg + " Pass --force to override (not recommended).")
    return warnings


# ======================= z construction (pure numpy where possible) ================

def analytic_zr(r_pool: np.ndarray, b_pool_target: np.ndarray, d: int) -> np.ndarray:
    """E2a (spec 24 sec.3): z_r = sqrt(d) * normalize( E_pool[ r(x) B_bar(x_tilde) ] ).
    `b_pool_target` = B_bar(x_tilde) for every pool point (the TARGET tower --
    see `build_target_b_pool`); `r_pool` = the family reward evaluated at the
    SAME pool points (prefix-stable, same convention as everywhere else in
    this harness). Uses the EXACT sqrt(d)/eps-normalize convention
    `gciql_flow._sphere` uses (x*sqrt(d)/(||x||+1e-6)), not a bare unit
    normalize, so E2a's z_r has the SAME norm convention E2b's candidates
    (built by the agent's own `_density_z_encode`, which calls `_sphere`)
    do -- an apples-to-apples comparison is the entire point of G3. Pure
    numpy, jax-free, unit-tested directly (tests/test_eval_density_z_rewards.py)."""
    r_pool = np.asarray(r_pool, np.float64)
    b = np.asarray(b_pool_target, np.float64)
    assert r_pool.ndim == 1 and b.ndim == 2 and len(r_pool) == len(b), \
        f"r_pool {r_pool.shape} / b_pool_target {b.shape} shape mismatch"
    num = np.mean(r_pool[:, None] * b, axis=0)                 # E_pool[r(x) B_bar(x)]  [d]
    n = float(np.linalg.norm(num))
    return num * np.sqrt(float(d)) / (n + 1e-6)


def z_search_scores(w: np.ndarray, kappa_x0: np.ndarray, r_pool: np.ndarray) -> np.ndarray:
    """E2b's per-candidate ranking term (pure numpy, jax-free): the
    candidate-independent r(x0)/(1-gamma) offset drops out of an argmax, so
    ranking by the full PRIMARY Q_r reduces to ranking by this dot product --
    EXACTLY the reduction `decode_goal_retrieval_density` in
    eval_density_zeroshot.py already uses for the g-indexed retrieval decode
    (see that function's docstring). w: [N,P] (N candidates); kappa_x0: [P]
    (candidate-independent, one x0); r_pool: [P]. Returns scores [N]."""
    w = np.asarray(w, np.float64)
    kappa_x0 = np.asarray(kappa_x0, np.float64).reshape(-1)
    r_pool = np.asarray(r_pool, np.float64)
    assert w.ndim == 2 and w.shape[1] == len(kappa_x0) == len(r_pool), \
        f"w {w.shape} / kappa_x0 {kappa_x0.shape} / r_pool {r_pool.shape} mismatch"
    return np.sum((w - kappa_x0[None, :]) * r_pool[None, :], axis=-1)


# ======================= live-agent integration (jax-dependent; lazy import) =======

def build_target_b_pool(agent, blocks, pool_obs: np.ndarray) -> np.ndarray:
    """B_bar(x_tilde) for every pool observation via the agent's TARGET tower
    (`target_density_b`) -- the z-construction convention spec 24 sec.1
    pins (B_bar, frozen), NOT the ONLINE `density_b` tower `build_pool`
    (reused unchanged) uses for the w/kappa readout. Same eps-scaled input
    (`_density_queries`, called live -- not reimplemented) `build_pool` uses
    for the online tower. Returns [P, d] float64."""
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    u_sc = np.asarray(GCIQLFlowAgent._density_queries(
        jnp.asarray(pool_obs, jnp.float32), blocks))                       # [P, d_c]
    b_target = np.asarray(agent.network.select("target_density_b")(jnp.asarray(u_sc)),
                          np.float64)                                       # [P, d]
    return b_target


def z_candidates_from_pool(agent, cand_obs: np.ndarray) -> np.ndarray:
    """{z(u_j)} for candidate pool samples u_j (spec 24 sec.3 E2b) via the
    agent's OWN `_density_z_encode` -- E1's deterministic transform, called
    live (no rng, no reimplementation): z(x) = sqrt(d)*B_bar(x_tilde)/||.||.
    cand_obs: [N, obs_dim] full observations (NOT just coords --
    `_density_z_encode` slices hit-spec blocks internally). Returns [N, d]
    float64."""
    import jax.numpy as jnp

    z = agent._density_z_encode(jnp.asarray(cand_obs, jnp.float32))
    return np.asarray(z, np.float64)


def decode_z_search(agent, x0_obs: np.ndarray, z_cands: np.ndarray, u_raw: np.ndarray,
                    b_pool: np.ndarray, r_pool: np.ndarray, blocks, temp: float,
                    double_q: bool, rng_seed: int = 0):
    """E2b (spec 24 sec.3): score every candidate z_j against the pool
    PRIMARY readout at x0, argmax selects z*. ACTION FED (mirrors
    `decode_goal_retrieval_density`'s own justification exactly): the
    trained flow actor's one-shot deterministic proposal
    a_j ~ actor(x0, z_j) via `sample_actions_with_z`, temperature=0, a
    SINGLE shared rng key across all candidates -- "what the deployed
    z-conditioned policy would do immediately if it committed to z_j".
    Returns (z_star [d], info dict)."""
    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    n = len(z_cands)
    x0_b = jnp.tile(jnp.asarray(x0_obs, jnp.float32)[None], (n, 1))
    z_j = jnp.asarray(z_cands, jnp.float32)
    rng = jax.random.PRNGKey(rng_seed)
    a_cand = np.clip(np.asarray(agent.sample_actions_with_z(
        observations=x0_b, z=z_j, seed=rng, temperature=0.0)), -1, 1)
    f_emb = np.asarray(agent.network.select("density_f")(
        jnp.concatenate([x0_b, jnp.asarray(a_cand), z_j], -1)), np.float64)   # [2,n,d]
    w = density_weights(f_emb, b_pool, temp, double_q)                        # [n,P]
    kappa_x0 = np.asarray(GCIQLFlowAgent._delta_phi(
        jnp.asarray(x0_obs, jnp.float32)[None], u_raw, blocks), np.float64)   # [1,P]
    scores = z_search_scores(w, kappa_x0[0], r_pool)                          # [n]
    chosen = int(np.argmax(scores))
    info = {"n_candidates": n, "chosen_idx": chosen,
           "score_max": float(scores[chosen]), "score_mean": float(scores.mean()),
           "score_std": float(scores.std()),
           "action_fed": "trained flow actor's one-shot deterministic proposal "
                         "a_j ~ actor(x0, z_j) via sample_actions_with_z, "
                         "temperature=0, single shared rng key across all candidates"}
    return z_cands[chosen], info


def batched_goal_rollout_pmm(agent, u_raw, b_pool, r_pool, blocks, gamma, temp, double_q,
                             spec, reward_fn, envs, g_hat_obs, obs_dim: int, act_dim: int,
                             ridge: float = 1e-4, H: int = 1000):
    """Control arm's rollout: the g-indexed twin of `batched_z_rollout_pmm`,
    conditioning the actor on a DECODED GOAL OBSERVATION (`agent.sample_actions`,
    the standard g-indexed path -- no z anywhere) instead of a precomputed z.
    Reuses `density_query` (imported from eval_density_zeroshot.py, unchanged)
    for the t=0 w/kappa readout -- the SAME function `eval_density_zeroshot.py`'s
    own rollout uses for the g-indexed w/kappa computation, so the readout math
    is identical to what E2a/E2b use, only the conditioning differs.

    Returns (obs_traj [n,H,obs_dim], lengths [n], t0 dict)."""
    import jax
    import jax.numpy as jnp

    n = len(envs)
    g_hat_b = jnp.tile(jnp.asarray(g_hat_obs, jnp.float32)[None], (n, 1))
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _info = e.reset(options=dict(task_id=1, render_goal=False))
        obs[i] = o
    active = np.ones(n, bool)
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    t0 = {}
    rng = jax.random.PRNGKey(0)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        obs_j = jnp.asarray(obs)
        acts = np.clip(np.asarray(agent.sample_actions(
            observations=obs_j, goals=g_hat_b, seed=sub, temperature=0.0)), -1, 1)
        if t == 0:
            w, kappa_x0 = density_query(agent, obs_j, jnp.asarray(acts), g_hat_b,
                                        u_raw, b_pool, blocks, temp, double_q)
            r_x0 = reward_fn(spec.coords(obs))
            q_primary = compute_density_primary(r_x0, w, kappa_x0, r_pool[None, :], gamma)
            q_secondary, _w_r, _mu, _mu0 = compute_density_secondary(
                r_x0, w, kappa_x0, b_pool, r_pool, gamma, ridge=ridge)
            t0["qr_primary"] = np.asarray(q_primary)
            t0["qr_secondary"] = np.asarray(q_secondary)
        for i, e in enumerate(envs):
            if not active[i]:
                obs_traj[i, t] = obs_traj[i, t - 1]
                continue
            o, _r, term, trunc, _inf = e.step(acts[i])
            obs[i] = o
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if term or trunc:
                active[i] = False
        if not active.any():
            obs_traj[:, t + 1:] = obs_traj[:, t:t + 1]
            break
    return obs_traj, lengths, t0


def make_pmm_envs(n: int, seed_offset: int = 0, task: str = "reach_top_left"):
    """`n` independent `_ExoRLPointMassMazeEnv` instances (spec 23 P0's
    dm_control adapter, imported via `exorl_env`'s own mujoco-overlay-safe
    loader -- not reimplemented). Each instance's `reset()` reconstructs the
    inner dm_control env from its OWN fixed construction seed
    (`PointMassMazeEnv.reset`), so per-env rollouts are deterministic across
    repeat calls -- this IS the harness's determinism convention for pmm
    (there is no per-call `reset(seed=...)` re-randomization available)."""
    import exorl_env

    pmm_module = exorl_env._import_point_mass_maze_env()
    return [exorl_env._ExoRLPointMassMazeEnv(pmm_module, task=task, seed=seed_offset + i)
           for i in range(n)]


def batched_z_rollout_pmm(agent, u_raw, b_pool, r_pool, blocks, gamma, temp, double_q,
                          spec, reward_fn, envs, z_fixed, obs_dim: int, act_dim: int,
                          ridge: float = 1e-4, H: int = 1000):
    """Plain fixed-z rollout for the whole episode (no per-step re-scoring --
    same "retrieval decode, then fixed rollout" shape as
    `batched_goal_rollout_density`, adapted for point-mass-maze: actions come
    from `sample_actions_with_z` (a precomputed z, no goal-obs mapping) and
    the tracked trajectory is the raw 4-dim obs (position+velocity), not
    mujoco qpos cube xyz -- there is no cube here. Reports, at t=0, PRIMARY
    and SECONDARY Q_r at the z-conditioned action, via the SAME imported
    readout functions eval_density_zeroshot.py's own rollout uses.

    Returns (obs_traj [n,H,obs_dim], lengths [n], t0 dict)."""
    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    n = len(envs)
    z_b = jnp.tile(jnp.asarray(z_fixed, jnp.float32)[None], (n, 1))
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _info = e.reset(options=dict(task_id=1, render_goal=False))
        obs[i] = o
    active = np.ones(n, bool)
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    t0 = {}
    rng = jax.random.PRNGKey(0)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        obs_j = jnp.asarray(obs)
        acts = np.clip(np.asarray(agent.sample_actions_with_z(
            observations=obs_j, z=z_b, seed=sub, temperature=0.0)), -1, 1)
        if t == 0:
            f_emb = np.asarray(agent.network.select("density_f")(
                jnp.concatenate([obs_j, jnp.asarray(acts), z_b], -1)), np.float64)
            w = density_weights(f_emb, b_pool, temp, double_q)
            kappa_x0 = np.asarray(GCIQLFlowAgent._delta_phi(obs_j, u_raw, blocks), np.float64)
            r_x0 = reward_fn(spec.coords(obs))
            q_primary = compute_density_primary(r_x0, w, kappa_x0, r_pool[None, :], gamma)
            q_secondary, _w_r, _mu, _mu0 = compute_density_secondary(
                r_x0, w, kappa_x0, b_pool, r_pool, gamma, ridge=ridge)
            t0["qr_primary"] = np.asarray(q_primary)
            t0["qr_secondary"] = np.asarray(q_secondary)
        for i, e in enumerate(envs):
            if not active[i]:
                obs_traj[i, t] = obs_traj[i, t - 1]
                continue
            o, _r, term, trunc, _inf = e.step(acts[i])
            obs[i] = o
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if term or trunc:
                active[i] = False
        if not active.any():
            obs_traj[:, t + 1:] = obs_traj[:, t:t + 1]
            break
    return obs_traj, lengths, t0


# ======================= CLI ======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json; first match "
                         "used; must be a measure_density_z=True checkpoint)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--arm", required=True, choices=ARMS,
                    help="e2a (analytic z_r) or e2b (search over pool candidates) -- "
                         "z-indexed checkpoints only; control (g-indexed retrieval "
                         "decode, eval_density_zeroshot.py's own decode_goal_retrieval_"
                         "density) -- g-indexed checkpoints only. One JSON per "
                         "(checkpoint, arm)")
    ap.add_argument("--tasks", default="1,2,3,4,5",
                    help="comma-sep task indices 1..20 into MAZE_20_GOALS (dirac/dense "
                         "families)")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help=f"comma-sep subset of {FAMILIES}")
    ap.add_argument("--regions", type=int, default=4,
                    help="region family: number of hindsight centers")
    ap.add_argument("--region-min-pos", type=int, default=5,
                    help="region family: min positive relabel samples per center")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=0,
                    help="0 = point_mass_maze_env.EPISODE_LEN (1000)")
    ap.add_argument("--pool_size", type=int, default=16384,
                    help="P: query pool size for the readout (spec 24 protocol: 16384)")
    ap.add_argument("--e2b_candidate_size", type=int, default=4096,
                    help="e2b: number of pool samples u_j whose z(u_j) become search "
                         "candidates. control: number of dataset samples scored as "
                         "retrieval-decode candidate goals. Both deliberately smaller "
                         "than --pool_size (see module docstring 'TWO ARMS')")
    ap.add_argument("--retrieval-k", type=int, default=8,
                    help="control arm only: top-k candidates averaged before the "
                         "nearest-dataset-state decode (decode_goal_retrieval_density's "
                         "own default)")
    ap.add_argument("--ridge", type=float, default=1e-4,
                    help="SECONDARY readout ridge regularization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="override the measure_density_z/control consistency refusal "
                         "(loud warning, numbers likely garbage)")
    ap.add_argument("--out", default="",
                    help="output JSON (default results/spec24_eval/<arm>_<run>.json, or "
                         "results/spec24_eval/control_gindexed_pmm_s<seed>.json for "
                         "--arm control)")
    return ap


# ======================= main =====================================================

def main() -> int:
    args = build_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    assert not unknown, f"unknown families {unknown}; choose from {FAMILIES}"
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])

    ckpt_is_density = bool(cfg.get("measure_density", False))
    ckpt_is_density_z = bool(cfg.get("measure_density_z", False))
    _consistency_check = (check_control_consistency if args.arm == "control"
                          else check_density_z_consistency)
    for _w in _consistency_check(ckpt_is_density, ckpt_is_density_z,
                                 force=args.force, run_dir=str(run_dir)):
        print(f"[density-z] {_w}", flush=True)

    env_name = flags["env_name"]
    assert env_name == EXORL_ENV_NAME, \
        f"this script is scoped to {EXORL_ENV_NAME!r} (spec 24 round 1); got {env_name!r}"
    gamma = float(cfg["discount"])
    double_q = bool(cfg.get("double_q", True))
    temp = float(cfg.get("measure_density_temp", 1.0))
    d = int(cfg["measure_density_dim"])

    spec = HitSpec.parse(cfg["measure_hit_spec"])   # spec 24: read from flags.json, not a CLI default
    blocks = spec.blocks
    task_ids = sorted(set(task_ids))
    assert all(1 <= t <= 20 for t in task_ids), \
        f"task ids must be in 1..20 (MAZE_20_GOALS), got {task_ids}"

    t00 = time.time()

    import exorl_env
    exorl_env.install()
    import utils.env_utils as env_utils

    _env0, train, _val = env_utils.make_env_and_datasets(EXORL_ENV_NAME, frame_stack=None)
    obs_dim = int(np.asarray(train["observations"]).shape[-1])
    act_dim = int(np.asarray(train["actions"]).shape[-1])
    H = args.horizon if args.horizon > 0 else 1000

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)

    print(f"[density-z] restored {run_dir.name} step={args.checkpoint_step} "
          f"arm={args.arm} env={env_name} d={d} gamma={gamma} "
          f"hit_spec={cfg.get('measure_hit_spec')!r} double_q={double_q} temp={temp} "
          f"pool_size={args.pool_size} e2b_candidate_size={args.e2b_candidate_size} "
          f"H={H}", flush=True)

    pmm_module = exorl_env._import_point_mass_maze_env()
    MAZE_20_GOALS = pmm_module.MAZE_20_GOALS

    rng_np = np.random.default_rng(args.seed)
    obs_all = np.asarray(train["observations"], np.float32)
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(train["terminals"]))

    # readout pool (ONLINE tower, reused build_pool -- unchanged) + relabel sample for
    # region hindsight centers and E2b candidates.
    n_ds = len(obs_all)
    pool_idx = rng_np.choice(n_ds, size=min(args.pool_size, n_ds),
                             replace=(args.pool_size > n_ds))
    pool_obs = obs_all[pool_idx]
    u_raw, b_pool = build_pool(agent, blocks, pool_obs, spec)
    b_pool_target = build_target_b_pool(agent, blocks, pool_obs)   # E2a only
    print(f"[density-z] pool drawn: P={len(pool_obs)} ({time.time()-t00:.0f}s)", flush=True)

    cand_idx = rng_np.choice(n_ds, size=min(args.e2b_candidate_size, n_ds),
                             replace=(args.e2b_candidate_size > n_ds))
    relabel_coords = coords_all[cand_idx]

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "arm": args.arm, "env": env_name, "d": d, "gamma": gamma, "double_q": double_q,
        "temp": temp, "hit_spec": cfg.get("measure_hit_spec"),
        "pool_size": args.pool_size, "e2b_candidate_size": args.e2b_candidate_size,
        "ridge": args.ridge, "seed": args.seed, "episodes": args.episodes, "horizon": H,
        "tasks": task_ids, "families": families,
        "tower_convention": "w/kappa readout = ONLINE density_b (build_pool, unchanged "
                            "from eval_density_zeroshot.py); z_r reward embedding "
                            "(E2a) and _density_z_encode candidates (E2b) = TARGET "
                            "target_density_b -- spec 24 sec.1",
        "readout": {
            "e2a": "z_r = sqrt(d) * normalize( E_pool[ r(x) B_bar(x_tilde) ] ) -- "
                  "spec 24 sec.3 E2a, gciql_flow._sphere norm convention",
            "e2b": "candidates z(u_j) = agent._density_z_encode(u_j) for u_j ~ "
                  "e2b_candidate_size pool samples; scored by pool PRIMARY Q_r(s0,.) "
                  "(compute_density_primary/density_weights, imported unchanged); "
                  "argmax selects z* -- spec 24 sec.3 E2b",
            "rollout": "plain fixed-z rollout (sample_actions_with_z), no per-step "
                      "re-scoring -- retrieval-decode-then-rollout shape shared with "
                      "eval_density_zeroshot.py",
            "control": "g-indexed retrieval decode (decode_goal_retrieval_density, "
                      "imported unchanged from eval_density_zeroshot.py) over "
                      "e2b_candidate_size dataset candidates -- NOT a z-construction "
                      "at all; requires a measure_density_z=False checkpoint "
                      "(check_control_consistency). Added 2026-08-13 so E2a/E2b's "
                      "region_success_from_obs reads have a same-metric, "
                      "same-families, same-pool g-indexed comparison point (the "
                      "task's original .315/.242/.235 anchors are native env "
                      "success, a DIFFERENT metric -- comparing E2 against them is a "
                      "protocol mismatch).",
            "success": "region_success_from_obs (imported unchanged) at the family's "
                      "own reward center, uniformly across dirac/dense/region -- "
                      "point-mass-maze's native env success is keyed to a DIFFERENT "
                      "(env-internal task_id) goal, so the manipspace shortcut of "
                      "using info['success'] does not apply here",
            "spec": "PAPER/Multistep-FB/specs/24-z-indexed-density.md sec.3",
        },
        "role": "control arm (g-indexed)" if args.arm == "control" else "E2 arm (z-indexed)",
    }, "families": {}}

    def envs_for_cell():
        return make_pmm_envs(args.episodes, seed_offset=0)

    def select_z(r_pool_full, x0_obs, envs0):
        if args.arm == "e2a":
            z, info = analytic_zr(r_pool_full, b_pool_target, d), {"arm": "e2a"}
            return z, info
        z_cands = z_candidates_from_pool(agent, obs_all[cand_idx])
        z_star, info = decode_z_search(agent, x0_obs, z_cands, u_raw, b_pool,
                                       r_pool_full, blocks, temp, double_q,
                                       rng_seed=args.seed)
        info["arm"] = "e2b"
        return z_star, info

    def select_g_hat(r_pool_full, x0_obs):
        """Control arm: eval_density_zeroshot.py's OWN retrieval decode
        (decode_goal_retrieval_density, imported unchanged), candidates drawn
        from the SAME cand_idx sample E2b's candidates use (parity, not a new
        sample)."""
        g_hat, info = decode_goal_retrieval_density(
            agent, x0_obs, u_raw, b_pool, r_pool_full, blocks, temp, double_q,
            coords_all[cand_idx], obs_all[cand_idx],
            retrieval_k=args.retrieval_k, rng_seed=args.seed)
        info["arm"] = "control"
        return g_hat, info

    def run_cell(r_pool_full, rfn, envs, x0_obs):
        if args.arm == "control":
            g_hat, decode_info = select_g_hat(r_pool_full, x0_obs)
            obs_traj, lengths, t0 = batched_goal_rollout_pmm(
                agent, u_raw, b_pool, r_pool_full, blocks, gamma, temp, double_q, spec,
                rfn, envs, g_hat, obs_dim, act_dim, ridge=args.ridge, H=H)
            return obs_traj, lengths, t0, decode_info
        z, decode_info = select_z(r_pool_full, x0_obs, envs)
        obs_traj, lengths, t0 = batched_z_rollout_pmm(
            agent, u_raw, b_pool, r_pool_full, blocks, gamma, temp, double_q, spec,
            rfn, envs, z, obs_dim, act_dim, ridge=args.ridge, H=H)
        return obs_traj, lengths, t0, decode_info

    def build_rec(fin, t0, decode_info, extra=None):
        rec = {"decode": decode_info, "success_final": float(np.mean(fin)),
              "qr_primary_t0_mean": float(np.mean(t0["qr_primary"])),
              "qr_secondary_t0_mean": float(np.mean(t0["qr_secondary"]))}
        if extra:
            rec.update(extra)
        return rec

    # ================= dirac / dense (goal families) =================
    for fam_name, rfn_builder in (("dirac", reward_conj_dirac), ("dense", reward_conj_dense)):
        if fam_name not in families:
            continue
        fam = {}
        for i in task_ids:
            center = MAZE_20_GOALS[i - 1]

            def rfn(coords, c=center, f=rfn_builder):
                return f(coords, c, spec)

            r_pool_full = rfn(coords_all[pool_idx])
            envs = envs_for_cell()
            x0_obs, _ = envs[0].reset(options=dict(task_id=1, render_goal=False))
            obs_traj, lengths, t0, decode_info = run_cell(r_pool_full, rfn, envs, x0_obs)
            fin, anys = region_success_from_obs(obs_traj, lengths, center, spec)
            rec = build_rec(fin, t0, decode_info,
                            extra={"success_any": float(anys.mean()),
                                  "goal_coord": np.asarray(center).tolist()})
            fam[f"t{i}"] = rec
            print(f"[density-z][{fam_name}] t{i}: succ_final={rec['success_final']} "
                  f"succ_any={rec['success_any']} qr0={rec['qr_primary_t0_mean']:.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success_final"] for v in fam.values()]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"][fam_name] = fam

    # ================= region (hindsight family) =================
    if "region" in families:
        rng_reg = np.random.default_rng(args.seed + 2)
        centers, _centers_obs, npos_c, rejected = draw_region_centers_with_obs(
            obs_all, coords_all, bounds, spec, relabel_coords, rng_reg, args.regions,
            min_pos=args.region_min_pos)
        fam = {"n_centers": int(len(centers)), "centers_rejected": int(rejected),
              "center_min_pos": int(args.region_min_pos)}
        print(f"[density-z][region] {len(centers)}/{args.regions} hindsight centers "
              f"({rejected} rejected at min_pos={args.region_min_pos})", flush=True)
        for k_i in range(len(centers)):
            c = centers[k_i]

            def rfn(coords, c=c):
                return reward_conj_dirac(coords, c, spec)

            r_pool_full = rfn(coords_all[pool_idx])
            envs = envs_for_cell()
            x0_obs, _ = envs[0].reset(options=dict(task_id=1, render_goal=False))
            obs_traj, lengths, t0, decode_info = run_cell(r_pool_full, rfn, envs, x0_obs)
            fin, anys = region_success_from_obs(obs_traj, lengths, c, spec)
            rec = build_rec(fin, t0, decode_info,
                            extra={"success_any": float(anys.mean()),
                                  "center_coord": np.asarray(c).tolist(),
                                  "center_n_pos": int(npos_c[k_i])})
            fam[f"c{k_i}"] = rec
            print(f"[density-z][region] c{k_i}: succ_final={rec['success_final']} "
                  f"succ_any={rec['success_any']} ({time.time()-t00:.0f}s)", flush=True)
        cells = [v for k, v in fam.items() if k.startswith("c") and isinstance(v, dict)]
        vals = [v["success_final"] for v in cells]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["region"] = fam

    fam_means = {k: v.get("mean") for k, v in results["families"].items()}
    results["meta"]["family_means"] = fam_means

    if args.out:
        outp = Path(args.out)
    elif args.arm == "control":
        outp = REPO / "results" / "spec24_eval" / f"control_gindexed_pmm_s{args.seed}.json"
    else:
        outp = REPO / "results" / "spec24_eval" / f"{args.arm}_{run_dir.name}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[density-z] DONE ({time.time()-t00:.0f}s) -> {outp}\n"
          f"[density-z] family means: {fam_means}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
