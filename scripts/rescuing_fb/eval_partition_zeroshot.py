#!/usr/bin/env python
"""scripts/rescuing_fb/eval_partition_zeroshot.py -- spec 17b (variant B) zero-shot
multi-reward readout for `measure_delta_partition` checkpoints.

Background: `PAPER/rescuing-fb/specs/17b-delta-partition-impl.md` sec.0/3,
`PAPER/rescuing-fb/specs/17-delta-measure.md` sec.1 variant B. 17b trains a
one-stage simplex successor measure over a FIXED partition of K dataset-drawn
prototypes of the hit-spec object-configuration blocks:

    phi_k(x)   = softmax_k( -0.5 * sum_blocks ||(x_blk - proto_k,blk)/eps_blk||^2 )  in Delta^K
    p(s,a,g)   = softmax( delta_head([s, a, g]) )                                     in Delta^K
    fixed point p*(s,a,g) = (1-gamma) * E^{pi_g}[ sum_t gamma^t phi(x_{t+1}) | s,a ]

Because p is a K-vector, ANY reward function r(x) can be evaluated zero-shot at
inference time with NO relabel-sample z-inference step (contrast the bilinear
`eval_zeroshot_rewards.py` harness, which must estimate z_r from a moment/
retrieval/ridge fit over a large relabeled dataset sample):

    z_r,k  = r(proto_k)                       (evaluate r directly at each prototype)
    F^Δ    = p(s, a, g) - phi(x0)
    Q_r    = ( r(x0) + F^Δ . z_r ) / (1 - gamma)

This script loads a 17b checkpoint, builds z_r for a family of rewards (reusing
the reward primitives `reward_conj_dirac` / `reward_conj_dense` and the region-
center / success helpers already defined in `eval_zeroshot_rewards.py` and
`env_geometry.py` -- NOT reinventing them), computes Q_r via the exact formula
above using `GCIQLFlowAgent._delta_phi` (the SAME code path training uses, not
a reimplementation), and rolls out a Q_r-greedy policy.

CLI shape mirrors `eval_zeroshot_rewards.py` / `run_spec14_jax_table.sh`
(--run-dir, --checkpoint-step, --families, --episodes, --horizon, --seed,
--out) and the "dirac"/"region" output JSON keys are structured identically
(`families["dirac"][f"t{i}"]["success"]`, `families["region"]["c{k}"]...`,
`...["mean"]` / `...["mean_final"]`) so `scripts/rescuing_fb/spec14_jax_table.py`
can be pointed at this script's `--out` directory too.

Families supported: dirac, dense, region (the same restricted set
`eval_zeroshot_rewards.py` uses off cube-single -- spec 14 sec.7). mixture /
linear / reachavoid are NOT ported: mixture and reachavoid are trivially
r(x) = max/difference of `reward_conj_dirac` calls and would compose the same
way if ever needed, but are left out of round 1 to keep the (unvalidated,
no-checkpoint-yet) rollout mechanism's scope small; linear needs a psi(s,a)
feature-space z-manifold that has no equivalent in a K-dim simplex partition
and is out of scope entirely.

CRITICAL -- prototype provenance (spec 17b sec.2.7/2.2): the K prototypes are
NOT stored in the checkpoint (`GCIQLFlowAgent.delta_protos` is a static/
nonpytree field, deliberately, so old checkpoints keep restoring -- see
`tests/test_gciql_measure_delta_partition.py` test 12). The agent rebuilds
`delta_protos` at `create()` time by `np.load`-ing the path recorded in the
checkpoint's OWN config (`measure_delta_protos`, from flags.json). A silently
regenerated or permuted prototype file changes every downstream number with NO
error from the agent itself -- this script independently re-loads the same
path, computes its sha256, prints it, and (when the digest is known -- the
canonical registry below, or --expected-protos-sha256) fails loudly on a
mismatch. It also cross-checks the independently-loaded array against
`agent.delta_protos` (the array actually baked into the restored agent) so a
path-resolution bug in this script cannot silently diverge from what the
agent is really using.

Two DECODE MODES (--decode; default "retrieval"):

* "retrieval" (default) -- comparability with the spec-14 numbers, which were
  produced with `--z-inference retrieval --retrieval-k 1`
  (run_spec14_jax_table.sh). Mirrors eval_zeroshot_rewards.py's own
  actor_cond='goal' deployment convention EXACTLY: decode ONE goal-proxy
  observation g_hat from a fixed dataset candidate pool (same pool, same
  --seed rng handling as the harness's relabel sample `S`), ONCE per
  family/task, state-independently -- then roll out the goal-conditioned flow
  actor at that single g_hat for the whole episode (`batched_goal_rollout`,
  the same "one decoded goal, no per-step re-scoring" shape as the harness's
  `batched_rollout_record`). The only departure from the harness: candidates
  are scored by THIS readout's own Q_r algebra (`select_retrieval_candidate` /
  `decode_goal_retrieval`), not by raw reward -- since scoring needs an action
  argument (p(s,a,g) takes s,a,g), the action fed is the trained flow actor's
  own one-shot deterministic proposal a_j ~ actor(x0, g_j) at a canonical
  start state x0 (the family/task's own env, seed-0 reset) -- see
  `decode_goal_retrieval`'s docstring for the full justification.
* "qr_greedy" -- the mechanism this script originally shipped with: no
  retrieval/decode step at all; deployment conditions on the "natural" goal
  (task goal_obs for dirac/dense, the hindsight-drawn region center's own
  observation for region) and re-scores `--n-candidates` action proposals by
  Q_r AT EVERY STEP (`batched_qr_rollout`). Since r(x0) and phi(x0).z_r are
  candidate-independent at a fixed state, ranking by Q_r reduces to ranking
  by p(s,a,g_hat).z_r (min over the 2 delta_head heads when double_q,
  matching gciql_flow.py's own actor_loss q-term reduction, spec 17b
  sec.2.6) -- the argmax candidate is executed.

Both are NEW DESIGN CHOICES by this script, not specified verbatim by spec
17/17b (which explicitly deferred "the zero-shot readout script" to post-wave
-- spec 17b sec.7); "retrieval" is now the default because it is the
comparable one (17b's actor is goal-conditioned by lattice, which the
retrieval decode's goal-observation deployment matches structurally). Both
have no golden reference and are UNVALIDATED until a real 17b checkpoint
exists (the running wave was at ~30k of 500k steps when this script was
written; first save at 100k). The READOUT ALGEBRA (`compute_Qr` /
`compute_Qr_from_dots`) and the retrieval SELECTION algorithm
(`select_retrieval_candidate`) are the tested, load-bearing pieces -- see
tests/test_partition_zeroshot.py. The decode mode used is always recorded in
the output JSON (`meta.decode`) so a number can never be read without it.

Run (CPU-only rollouts; GPUs 0-5 carry training jobs -- do not touch them):
  taskset -c 32-39 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_partition_zeroshot.py \\
    --run-dir '/home/adityamohan/exp_spec17/factored-fb/GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50/*s0*' \\
    --checkpoint-step 500000 --families dirac,dense,region --episodes 25 --horizon 0
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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

# Reused, not reimplemented: reward primitives + region-center/success helpers
# from the proven spec 07/14 harness (eval_zeroshot_rewards.py), and the single
# geometry definition (env_geometry.py, spec 14 sec.3.1).
from eval_zeroshot_rewards import (  # noqa: E402
    CUBE_QPOS,
    eval_tasks_for,
    print_table,
    region_success_from_obs,
    reward_conj_dense,
    reward_conj_dirac,
)
from env_geometry import episode_bounds, geometry_for, hindsight_rows  # noqa: E402

FAMILIES = ("dirac", "dense", "region")
DECODES = ("retrieval", "qr_greedy")

# Spec 17b's own canonical prototype file for the running wave (see the task
# brief this script was built against): repo-relative path -> sha256. Any
# `measure_delta_protos` path that resolves to one of these (after making it
# repo-relative) is checked against this digest; unknown paths are only
# printed, not gated (round-1 covers cube-single K=256 only -- other envs/K's
# get added here as their own waves land, per spec 17b sec.7).
CANONICAL_PROTOS_SHA256 = {
    "results/delta_protos/cube_single_K256_s0.npy":
        "b4e13e9d9e0a0b0b18deedf5fdce66c1d3f9a17e02175608dd1679064494cbf5",
}


# ======================= prototype loading + verification =========================

def file_sha256(path) -> str:
    """Streaming sha256 (same pattern as scripts/rescuing_fb/measure_readout_head.py
    file_sha256 -- not imported from there to avoid pulling in its jax-lazy module)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_protos_file(path, expected_sha256: str = "") -> dict:
    """Loud, fail-fast verification of the prototype .npy file recorded in the
    checkpoint's own config (spec 17b sec.2.7: delta_protos is NOT a serialized
    checkpoint field -- see tests/test_gciql_measure_delta_partition.py test 12
    -- so a silently regenerated or permuted prototype file changes the readout
    with NO error from the agent itself; that is the exact failure mode this
    function defends against).

    Raises FileNotFoundError if the file is missing (never silently regenerates
    or substitutes). Raises AssertionError if a digest is known for this path
    (via `expected_sha256` or the CANONICAL_PROTOS_SHA256 registry, keyed by
    the path's repo-relative form) and does not match what was loaded. Always
    prints the digest it loaded.

    Returns {'path', 'sha256', 'matched' (True/False/None -- None = no known
    digest to check against), 'expected_sha256'}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"measure_delta_protos file not found: {p}\n"
            "The checkpoint's flags.json records this path but the file is "
            "missing on this machine. Refusing to proceed: regenerating a "
            "prototype file draws a DIFFERENT (and differently-ordered) set of "
            "dataset rows even with the same --k/--seed, since it depends on "
            "the dataset build; a regenerated file silently changes every "
            "downstream Q_r number with no error.")
    digest = file_sha256(p)
    try:
        rel = str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        rel = str(p)
    expected = expected_sha256 or CANONICAL_PROTOS_SHA256.get(rel, "")
    matched = None
    if expected:
        matched = digest == expected
    print(f"[partition-zs] protos={p} sha256={digest} "
          f"expected={expected or '(none known)'} "
          f"matched={matched}", flush=True)
    if expected:
        assert matched, (
            f"prototype file digest MISMATCH for {p}:\n"
            f"  loaded   sha256={digest}\n"
            f"  expected sha256={expected}\n"
            "This is the exact failure mode spec 17b sec.2.7 flags: a silently "
            "regenerated or permuted prototype file. Refusing to evaluate this "
            "checkpoint against prototypes that do not match what training used."
        )
    return {"path": str(p), "sha256": digest, "matched": matched,
            "expected_sha256": expected or None}


def load_protos_verified(path, expected_sha256: str = ""):
    """np.load the prototype file at `path` after `verify_protos_file`. Returns
    (protos [K, d_c] float32, verify_info dict)."""
    info = verify_protos_file(path, expected_sha256)
    protos = np.load(path).astype(np.float32)
    assert protos.ndim == 2 and protos.shape[0] >= 2, \
        f"measure_delta_protos must be [K>=2, d_c], got {protos.shape}"
    return protos, info


# ======================= readout algebra (pure numpy, unit-tested) =================

def compute_Qr_from_dots(r_x0, p_dot_zr, phi_dot_zr, gamma):
    """Q_r = (r(x0) + p.z_r - phi(x0).z_r) / (1 - gamma), given PRE-DOTTED
    scalars p.z_r and phi(x0).z_r (the rollout loop already has these -- the
    argmax candidate's winning score IS p.z_r, computed once per candidate
    anyway; this avoids re-dotting a full K-vector after the fact). Algebraic
    shortcut of `compute_Qr`; test_partition_zeroshot.py pins the two
    equivalent on random inputs."""
    r_x0 = np.asarray(r_x0, np.float64)
    p_dot_zr = np.asarray(p_dot_zr, np.float64)
    phi_dot_zr = np.asarray(phi_dot_zr, np.float64)
    return (r_x0 + p_dot_zr - phi_dot_zr) / (1.0 - gamma)


def compute_Qr(r_x0, p, phi_x0, z_r, gamma):
    """Spec 17b sec.3 / spec 17 sec.1 variant B zero-shot readout:

        F^Delta = p(s, a, g) - phi(x0)
        Q_r     = ( r(x0) + F^Delta . z_r ) / (1 - gamma)

    r_x0: scalar or [...] reward at the CURRENT state x0 (action-independent
      at fixed s -- it does not vary with the candidate action `a`).
    p, phi_x0: [..., K] (or [K]) vectors on the simplex (p = the learned
      occupancy readout at (s, a, g); phi_x0 = the fixed partition of x0).
    z_r: [K] reward evaluated at each of the K prototypes, z_r_k = r(proto_k).
    gamma: the CHECKPOINT's own recorded discount (config['discount']), never
      a CLI default (spec 17b sec.2.4 uses `self.config['discount']` the same
      way).

    Returns Q_r broadcast over the batch shape shared by r_x0/p/phi_x0.
    """
    p = np.asarray(p, np.float64)
    phi_x0 = np.asarray(phi_x0, np.float64)
    z_r = np.asarray(z_r, np.float64)
    p_dot_zr = np.sum(p * z_r, axis=-1)
    phi_dot_zr = np.sum(phi_x0 * z_r, axis=-1)
    return compute_Qr_from_dots(r_x0, p_dot_zr, phi_dot_zr, gamma)


# ======================= retrieval-decode candidate selection (pure numpy) =========

def select_retrieval_candidate(scores, cand_coords, cand_obs, retrieval_k=8):
    """Pure-numpy retrieval selection (spec-14 harness parity, `--retrieval-k`
    in eval_zeroshot_rewards.py / run_spec14_jax_table.sh's `--retrieval-k 1`
    gate protocol): given a precomputed score per candidate (higher = better;
    `decode_goal_retrieval` supplies this readout's own Q_r-based ranking),
    take the top `retrieval_k` candidates.

    k=1 (the gate protocol default this script's numbers are meant to be
    comparable with) returns that single top-scoring candidate directly. k>1
    averages the top-k candidates' COORDINATES and returns the dataset
    candidate nearest (L2, in hit-spec coordinate space) to that average --
    the discrete analogue of the harness's own "average psi embeddings, then
    nearest-neighbor decode" (eval_zeroshot_rewards.z_to_goal_proxy_idx),
    adapted because 17b has no continuous per-state embedding to average a z
    in (only the fixed K-simplex phi(x), which is not what's being averaged
    here).

    Factored out of `decode_goal_retrieval` (which needs jax to compute
    `scores` via the actor + delta_head) so the SELECTION algorithm itself is
    unit-testable jax-free -- mirrors eval_zeroshot_rewards.gpi_select's own
    factoring rationale, and is a genuinely DIFFERENT code path from
    `batched_qr_rollout`'s per-step argmax-over-ACTIONS (this selects among
    candidate GOALS from precomputed scores, once, not among actions at a
    fixed goal, every step).

    Returns (chosen_idx: int, info dict).
    """
    scores = np.asarray(scores, np.float64)
    cand_coords = np.asarray(cand_coords, np.float64)
    n = len(scores)
    assert n > 0, "no retrieval candidates"
    assert len(cand_coords) == n == len(cand_obs), \
        f"scores/cand_coords/cand_obs length mismatch: {n}/{len(cand_coords)}/{len(cand_obs)}"
    k = max(1, min(int(retrieval_k), n))
    order = np.argsort(scores)
    top_idx = order[-k:]
    if k == 1:
        chosen = int(top_idx[0])
    else:
        avg_coord = cand_coords[top_idx].mean(axis=0)
        d = np.linalg.norm(cand_coords - avg_coord[None, :], axis=1)
        chosen = int(np.argmin(d))
    info = {"retrieval_k": k, "n_candidates": n,
           "score_chosen": float(scores[chosen]), "score_max": float(scores.max()),
           "score_mean": float(scores.mean()), "top1_idx": int(top_idx[-1])}
    return chosen, info


# ======================= region centers WITH the source observation ================

def draw_region_centers_with_obs(obs_all, coords_all, bounds, spec, relabel_coords,
                                 rng, k, min_pos=5, max_tries=400):
    """Fork of `eval_zeroshot_rewards.draw_region_centers` (spec 14 sec.2.1/3.3
    hindsight-center policy: a region center = the coordinate of a dataset row
    VISITED later in a random start row's own episode, rejecting centers with
    fewer than `min_pos` positive relabel samples at the trained radii). Not
    reusable as-is: 17b's actor/delta_head need a full goal OBSERVATION to
    condition on (g_hat), not just the region coordinate -- the existing
    helper discards the row index once it has the coordinate, because
    eval_zeroshot_rewards.py's z-inference deployment path decodes a
    goal-proxy separately (`z_to_goal_proxy_idx`), which 17b has no need of
    (z_r is evaluated directly at the prototypes, no relabel-moment fit).

    Returns (centers [m<=k, D] coords, centers_obs [m, obs_dim], n_pos [m],
    n_rejected) -- identical semantics to draw_region_centers plus centers_obs.
    """
    centers, obs_rows, n_pos, rejected = [], [], [], 0
    n_rows = len(coords_all)
    for _ in range(int(max_tries)):
        if len(centers) == k:
            break
        start = int(rng.integers(0, n_rows))
        rows = hindsight_rows(bounds, start, rng, 1)
        if len(rows) == 0:                      # start at its episode's end
            rejected += 1
            continue
        ridx = int(rows[0])
        c = coords_all[ridx]
        npos = int(spec.contains(relabel_coords, c).sum())
        if npos < min_pos:
            rejected += 1
            continue
        centers.append(c)
        obs_rows.append(obs_all[ridx])
        n_pos.append(npos)
    d = coords_all.shape[1] if len(coords_all) else 0
    od = obs_all.shape[1] if len(obs_all) else 0
    centers_arr = (np.asarray(centers, np.float64) if centers
                   else np.zeros((0, d), np.float64))
    obs_arr = (np.asarray(obs_rows, np.float32) if obs_rows
              else np.zeros((0, od), np.float32))
    return centers_arr, obs_arr, np.asarray(n_pos, int), rejected


# ======================= rollout (jax-dependent; imported lazily) ==================

def batched_qr_rollout(agent, protos, blocks, gamma, spec, reward_fn, envs, g_hat_obs,
                       z_r, H=200, n_candidates=8, double_q=True, rng_seed=0):
    """Q_r-greedy lockstep rollout (see module docstring "Rollout mechanism").
    UNVALIDATED absent a real checkpoint -- see the module docstring's warning.

    reward_fn: coords [N, d_c] -> reward [N] (evaluated at the trajectory's OWN
      state, for the r(x0) term and diagnostics -- built by the caller from the
      same reward this family used to build z_r, so r(x0) and z_r are always
      the SAME reward function).
    g_hat_obs: [obs_dim] the fixed per-episode goal-conditioning observation
      (module docstring: the actor/delta_head's `g` input; real observation).
    z_r: [K] reward evaluated at each prototype.

    Returns (traj [n,H,3] cube xyz meters -- cube-single qpos semantics only,
    obs_traj [n,H,obs_dim], lengths [n], succ_env [n] native env info['success']
    any-time flag, qr_pred_t0 [n] this readout's OWN Q_r estimate, via
    compute_Qr_from_dots, for the executed action at t=0).
    """
    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    n = len(envs)
    obs_dim = envs[0].observation_space.shape[0]
    g_hat_b = jnp.tile(jnp.asarray(g_hat_obs, jnp.float32)[None], (n, 1))
    z_r_j = jnp.asarray(z_r, jnp.float32)
    protos_j = jnp.asarray(protos, jnp.float32)
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    succ_env = np.zeros(n)
    qr_pred_t0 = np.zeros(n)
    rng = jax.random.PRNGKey(rng_seed)
    for t in range(H):
        obs_j = jnp.asarray(obs)
        phi_x0 = np.asarray(GCIQLFlowAgent._delta_phi(obs_j, protos_j, blocks))  # [n,K]
        cand_actions = []
        scores = np.zeros((n_candidates, n))
        for c in range(n_candidates):
            rng, sub = jax.random.split(rng)
            a_c = np.clip(np.asarray(agent.sample_actions(
                observations=obs_j, goals=g_hat_b, seed=sub, temperature=0.0)), -1, 1)
            cand_actions.append(a_c)
            logits = agent.network.select("delta_head")(
                jnp.concatenate([obs_j, jnp.asarray(a_c), g_hat_b], -1))          # [2,n,K]
            p = jax.nn.softmax(logits, -1)
            qh = jnp.einsum("hnk,k->hn", p, z_r_j)                               # [2,n]
            q = jnp.min(qh, axis=0) if double_q else qh[0]
            scores[c] = np.asarray(q)
        cand_actions = np.stack(cand_actions, 0)          # [C,n,act]
        jstar = scores.argmax(axis=0)                      # [n]
        acts_exec = cand_actions[jstar, np.arange(n)]
        if t == 0:
            r_x0 = reward_fn(spec.coords(obs))
            p_dot_zr_exec = scores[jstar, np.arange(n)]
            phi_dot_zr = phi_x0 @ np.asarray(z_r, np.float64)
            qr_pred_t0 = compute_Qr_from_dots(r_x0, p_dot_zr_exec, phi_dot_zr, gamma)
        for i, e in enumerate(envs):
            if not active[i]:
                traj[i, t] = traj[i, t - 1]
                continue
            o, _r, term, trunc, inf = e.step(acts_exec[i])
            obs[i] = o
            traj[i, t] = e.unwrapped.data.qpos[CUBE_QPOS]
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if inf.get("success"):
                succ_env[i] = 1.0
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    return traj, obs_traj, lengths, succ_env, qr_pred_t0


def decode_goal_retrieval(agent, x0_obs, z_r, cand_coords, cand_obs,
                          retrieval_k=8, double_q=True, rng_seed=0):
    """Retrieval decode (spec-14 harness parity: `--z-inference retrieval`).
    Scores each candidate dataset state as a PROPOSED deployment goal by this
    readout's own Q_r algebra: p(x0, a_j, g_j).z_r (min over heads if
    double_q), where g_j = cand_obs[j].

    ACTION FED (per the task brief's explicit ask -- "if scoring a candidate
    goal requires an action argument, state which action you feed and why"):
    a_j = the trained flow actor's own ONE-SHOT DETERMINISTIC proposal,
    a_j ~ actor(x0, g_j), temperature=0, drawn from a SINGLE shared rng key
    (jax.random.PRNGKey(rng_seed)) applied identically to every candidate in
    one batched call -- so the ranking is reproducible and the noise draw
    cannot itself bias one candidate over another. This is "what the deployed
    policy would do immediately if it committed to goal g_j" -- there is no
    dataset behavior action available at x0 (x0 is a fresh env reset, not a
    dataset row), so the actor's own proposal is the only defensible
    per-candidate action; using the SAME action the eventual rollout would
    take at that goal keeps the score consistent with what deployment means.

    r(x0) and phi(x0).z_r are candidate-independent additive constants (same
    reduction `batched_qr_rollout` / gciql_flow.py's actor_loss use), so
    ranking by the full Q_r is exactly ranking by p(x0,a_j,g_j).z_r -- x0's
    reward/phi terms are never computed here (this function only needs to
    RANK candidates, not report an absolute Q_r; `select_retrieval_candidate`
    does the ranking, in a separate jax-free function so it is unit-testable).

    x0_obs: [obs_dim] the canonical start state (this script's callers use a
      seed-0 reset of the family/task's own env -- see main()).
    z_r: [K] reward evaluated at each prototype (the SAME z_r used to build
      Q_r everywhere else in this script -- family-specific).
    cand_coords, cand_obs: the SAME relabel dataset sample used throughout
      this script (main()'s `S_coords`/`S_obs` -- same pool, same
      `np.random.default_rng(args.seed)` handling as
      eval_zeroshot_rewards.py's own relabel sample `S`).

    Returns (g_hat_obs [obs_dim], info dict -- selection diagnostics plus
    x0/action provenance).
    """
    import jax
    import jax.numpy as jnp

    n = len(cand_obs)
    x0_b = jnp.tile(jnp.asarray(x0_obs, jnp.float32)[None], (n, 1))
    cand_obs_j = jnp.asarray(cand_obs, jnp.float32)
    z_r_j = jnp.asarray(z_r, jnp.float32)
    rng = jax.random.PRNGKey(rng_seed)
    a_cand = np.clip(np.asarray(agent.sample_actions(
        observations=x0_b, goals=cand_obs_j, seed=rng, temperature=0.0)), -1, 1)
    logits = agent.network.select("delta_head")(
        jnp.concatenate([x0_b, jnp.asarray(a_cand), cand_obs_j], -1))     # [2,n,K]
    p = jax.nn.softmax(logits, -1)
    qh = jnp.einsum("hnk,k->hn", p, z_r_j)                                # [2,n]
    q = jnp.min(qh, axis=0) if double_q else qh[0]
    scores = np.asarray(q)

    chosen, info = select_retrieval_candidate(scores, cand_coords, cand_obs, retrieval_k)
    info["x0_source"] = "canonical seed-0 env reset (main()'s per-task/region pool env)"
    info["action_fed"] = ("trained flow actor's one-shot deterministic proposal "
                          "a_j ~ actor(x0, g_j), temperature=0, single shared rng "
                          "key across all candidates")
    return cand_obs[chosen], info


def batched_goal_rollout(agent, protos, blocks, gamma, spec, reward_fn, envs,
                         g_hat_obs, z_r, H=200, double_q=True):
    """Plain fixed-goal rollout (retrieval decode's deployment step): g_hat_obs
    is FIXED for the whole episode (chosen ONCE, state-independently, by
    `decode_goal_retrieval` before this is called) -- exactly
    eval_zeroshot_rewards.py's convention for actor_cond='goal' cells (one
    decoded goal proxy per family/task, `batched_rollout_record`): no per-step
    candidate re-scoring, unlike `batched_qr_rollout`. Same return shape as
    `batched_qr_rollout` (traj, obs_traj, lengths, succ_env, qr_pred_t0) so
    both decode modes share a call site in main().
    """
    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    n = len(envs)
    obs_dim = envs[0].observation_space.shape[0]
    g_hat_b = jnp.tile(jnp.asarray(g_hat_obs, jnp.float32)[None], (n, 1))
    z_r_j = jnp.asarray(z_r, jnp.float32)
    protos_j = jnp.asarray(protos, jnp.float32)
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    succ_env = np.zeros(n)
    qr_pred_t0 = np.zeros(n)
    rng = jax.random.PRNGKey(0)
    for t in range(H):
        rng, sub = jax.random.split(rng)
        obs_j = jnp.asarray(obs)
        acts = np.clip(np.asarray(agent.sample_actions(
            observations=obs_j, goals=g_hat_b, seed=sub, temperature=0.0)), -1, 1)
        if t == 0:
            phi_x0 = np.asarray(GCIQLFlowAgent._delta_phi(obs_j, protos_j, blocks))
            logits = agent.network.select("delta_head")(
                jnp.concatenate([obs_j, jnp.asarray(acts), g_hat_b], -1))
            p = jax.nn.softmax(logits, -1)
            qh = jnp.einsum("hnk,k->hn", p, z_r_j)
            p_dot_zr = np.asarray(jnp.min(qh, axis=0) if double_q else qh[0])
            phi_dot_zr = phi_x0 @ np.asarray(z_r, np.float64)
            r_x0 = reward_fn(spec.coords(obs))
            qr_pred_t0 = compute_Qr_from_dots(r_x0, p_dot_zr, phi_dot_zr, gamma)
        for i, e in enumerate(envs):
            if not active[i]:
                traj[i, t] = traj[i, t - 1]
                continue
            o, _r, term, trunc, inf = e.step(acts[i])
            obs[i] = o
            traj[i, t] = e.unwrapped.data.qpos[CUBE_QPOS]
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if inf.get("success"):
                succ_env[i] = 1.0
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    return traj, obs_traj, lengths, succ_env, qr_pred_t0


# ======================= CLI ======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json; first match "
                         "used; must be a measure_delta_partition=True checkpoint)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--env", default="",
                    help="override env (default: flags.json env_name)")
    ap.add_argument("--tasks", default="1,2,3,4,5",
                    help="comma-sep task indices 1..5 (dirac/dense families)")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help=f"comma-sep subset of {FAMILIES}")
    ap.add_argument("--regions", type=int, default=4,
                    help="region family: number of hindsight centers")
    ap.add_argument("--region-min-pos", type=int, default=5,
                    help="region family: min positive relabel samples per center")
    ap.add_argument("--decode", choices=DECODES, default="retrieval",
                    help="'retrieval' (default, spec-14-comparable): decode ONE "
                         "goal-proxy dataset observation per family/task via "
                         "Q_r-scored retrieval, then a plain fixed-goal rollout. "
                         "'qr_greedy': re-score --n-candidates action proposals by "
                         "Q_r at EVERY rollout step, no retrieval decode.")
    ap.add_argument("--retrieval-k", type=int, default=8,
                    help="--decode retrieval: top-k candidates to retrieve before "
                         "collapsing to one goal (run_spec14_jax_table.sh's gate "
                         "protocol uses 1 -- pass --retrieval-k 1 to match it)")
    ap.add_argument("--n-candidates", type=int, default=8,
                    help="--decode qr_greedy: candidate pool size per rollout step "
                         "(flow-actor noise draws at the fixed episode goal)")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=0,
                    help="0 = the env's own max_episode_steps")
    ap.add_argument("--relabel-size", type=int, default=50000,
                    help="dataset sample size for the region family's hindsight draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protos-path", default="",
                    help="override the .npy path (default: checkpoint config "
                         "measure_delta_protos)")
    ap.add_argument("--expected-protos-sha256", default="",
                    help="override the built-in canonical-registry digest check")
    ap.add_argument("--out", default="",
                    help="output JSON (default results/spec17_delta/pzs_<run>.json)")
    return ap


# ======================= main =====================================================

def main() -> int:
    args = build_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    assert not unknown, f"unknown families {unknown}; choose from {FAMILIES}"
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    assert all(1 <= t <= 5 for t in task_ids), f"task ids must be in 1..5, got {task_ids}"

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])
    assert cfg.get("measure_delta_partition", False), (
        f"{run_dir} is not a measure_delta_partition (spec 17b) checkpoint "
        f"(measure_delta_partition={cfg.get('measure_delta_partition')!r}); "
        "this script's readout formula only applies to that path.")
    env_name = args.env or flags["env_name"]
    gamma = float(cfg["discount"])
    double_q = bool(cfg.get("double_q", True))
    protos_path = args.protos_path or cfg.get("measure_delta_protos", "")
    assert protos_path, f"{run_dir} flags.json has no measure_delta_protos recorded"
    protos, protos_info = load_protos_verified(protos_path, args.expected_protos_sha256)

    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    blocks = spec.blocks
    d_c = spec.dim
    assert protos.shape[1] == d_c, (
        f"prototype width {protos.shape[1]} != hit-spec width {d_c} "
        f"(measure_hit_spec={cfg.get('measure_hit_spec')!r}) -- protos file does "
        "not match this checkpoint's hit spec")
    K = int(protos.shape[0])
    eval_tasks = eval_tasks_for(env_name)

    t00 = time.time()
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    _obs_dim = int(np.asarray(ds1["observations"]).shape[-1])
    _act_dim = int(np.asarray(ds1["actions"]).shape[-1])
    H = args.horizon if args.horizon > 0 else int(_e0.spec.max_episode_steps)

    # checkpoint restore: t3_offpeak_multienv.load_agent_dims (domain-correct
    # example dims; same sys.path repair eval_zeroshot_rewards.py needs).
    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, _obs_dim, _act_dim)

    # Cross-check: the agent's OWN baked-in prototypes (set at create() time
    # from this same cfg['measure_delta_protos'] path -- spec 17b sec.2.7) must
    # equal what this script independently loaded+verified above. Catches a
    # path-resolution divergence between this script and the agent.
    agent_protos = np.asarray(agent.delta_protos, np.float32)
    assert agent_protos.shape == protos.shape and np.allclose(agent_protos, protos), (
        "agent.delta_protos (baked in at create() from the checkpoint's own "
        "config) does not match the independently-loaded+verified prototype "
        "file -- path resolution or environment mismatch between this script "
        "and the restored agent.")

    print(f"[partition-zs] restored {run_dir.name} step={args.checkpoint_step} "
          f"env={env_name} K={K} d_c={d_c} gamma={gamma} "
          f"hit_spec={cfg.get('measure_hit_spec')!r} double_q={double_q} "
          f"decode={args.decode} retrieval_k={args.retrieval_k} "
          f"n_candidates={args.n_candidates} H={H}", flush=True)

    rng_np = np.random.default_rng(args.seed)
    obs_all = np.asarray(ds1["observations"], np.float32)
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(ds1["terminals"]))
    idx = rng_np.choice(len(coords_all), size=min(args.relabel_size, len(coords_all)),
                        replace=False)
    relabel_coords = coords_all[idx]
    # S_obs/S_coords: the retrieval-decode candidate pool -- SAME pool, SAME
    # rng handling (np.random.default_rng(args.seed).choice over ds1, the
    # task-1 relabel sample) as eval_zeroshot_rewards.py's own `S`, so the
    # --decode retrieval numbers are comparable to the spec-14 harness.
    S_obs, S_coords = obs_all[idx], relabel_coords

    goal_obs, goal_coord = {}, {}
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        goal_coord[i] = spec.coords(goal_obs[i][None])[0]
        env.close()

    pools = {}

    def pool(i):
        if i not in pools:
            pools[i] = [ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
                        for _ in range(args.episodes)]
        return pools[i]

    def close_all():
        for i in list(pools):
            for e in pools.pop(i):
                if hasattr(e, "close"):
                    e.close()

    def rollout_for(z_r, rfn, envs, g_hat_natural):
        """Dispatch on args.decode -- the ONLY place the two decode modes
        branch, so a mode string can never silently fall through to the
        wrong implementation (DECODES is also enforced by argparse `choices`
        at parse time). 'qr_greedy' calls `batched_qr_rollout` directly on
        the natural goal (no decode step). 'retrieval' first calls
        `decode_goal_retrieval` (a genuinely separate function, over the
        shared S_obs/S_coords candidate pool, scored at a canonical x0 =
        envs[0] reset at seed=0) to pick g_hat, then `batched_goal_rollout`
        (a separate rollout function -- no per-step re-scoring) at that
        single g_hat for the whole episode."""
        if args.decode == "qr_greedy":
            out = batched_qr_rollout(agent, protos, blocks, gamma, spec, rfn, envs,
                                     g_hat_natural, z_r, H=H,
                                     n_candidates=args.n_candidates,
                                     double_q=double_q, rng_seed=args.seed)
            return out, {"mode": "qr_greedy",
                        "note": "no retrieval decode; conditions directly on the "
                                "natural goal observation"}
        assert args.decode == "retrieval", args.decode
        x0_obs, _ = envs[0].reset(seed=0)
        g_hat, decode_info = decode_goal_retrieval(
            agent, x0_obs, z_r, S_coords, S_obs, retrieval_k=args.retrieval_k,
            double_q=double_q, rng_seed=args.seed)
        decode_info["mode"] = "retrieval"
        out = batched_goal_rollout(agent, protos, blocks, gamma, spec, rfn, envs,
                                   g_hat, z_r, H=H, double_q=double_q)
        return out, decode_info

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "K": K, "d_c": d_c, "gamma": gamma, "double_q": double_q,
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
        "prototypes": protos_info,
        "readout": {
            "z_r_k": "r(proto_k) -- reward evaluated directly at each of the K "
                     "prototypes; NO relabel-sample z-inference (contrast the "
                     "bilinear eval_zeroshot_rewards.py harness)",
            "F_delta": "p(s, a, g_hat) - phi(x0)",
            "Q_r": "( r(x0) + F_delta . z_r ) / (1 - gamma)",
            "phi": "GCIQLFlowAgent._delta_phi (same code path as training)",
            "p": "softmax(delta_head([s, a, g_hat])), min over the 2 heads "
                 "(double_q)",
            "policy": ("'retrieval' (default, spec-14-comparable): ONE goal-proxy "
                      "observation g_hat decoded per family/task via Q_r-scored "
                      "retrieval over the shared relabel dataset pool "
                      "(decode_goal_retrieval + select_retrieval_candidate), then "
                      "a plain fixed-goal rollout (batched_goal_rollout, no "
                      "per-step re-scoring) -- mirrors eval_zeroshot_rewards.py's "
                      "actor_cond='goal' deployment convention."
                      if args.decode == "retrieval" else
                      "'qr_greedy': Q_r-greedy over n_candidates flow-actor noise "
                      "draws at a FIXED per-episode natural-goal observation "
                      "g_hat, re-scored at EVERY rollout step "
                      "(batched_qr_rollout); no retrieval decode."),
            "decode": args.decode,
            "unvalidated": "NEW design choice by this script, no golden reference "
                           "-- UNVALIDATED absent a real checkpoint (module docstring)",
            "spec": "PAPER/rescuing-fb/specs/17b-delta-partition-impl.md sec.3 + "
                    "specs/17-delta-measure.md sec.1 variant B",
        },
        "decode": args.decode, "retrieval_k": args.retrieval_k,
        "n_candidates": args.n_candidates, "tasks": task_ids, "families": families,
        "episodes": args.episodes, "horizon": H, "seed": args.seed,
        "relabel_size": int(len(relabel_coords)),
        "goal_coords": {f"t{i}": goal_coord[i].tolist() for i in task_ids},
    }, "families": {}}
    table = []

    # ================= 1. dirac =================
    if "dirac" in families:
        fam = {}
        for i in task_ids:
            z_r = reward_conj_dirac(protos, goal_coord[i], spec)
            n_pos = int((z_r > 0).sum())

            def rfn(coords, c=goal_coord[i]):
                return reward_conj_dirac(coords, c, spec)

            (traj, ot, ln, senv, qr0), decode_info = rollout_for(
                z_r, rfn, pool(i), goal_obs[i])
            rec = {"diag": {"n_pos": n_pos, "n_neg": K - n_pos,
                            "z_r_mean": float(z_r.mean()), "z_r_max": float(z_r.max())},
                  "decode": decode_info,
                  "g_hat": "task goal_obs (exact)" if args.decode == "qr_greedy"
                           else "retrieval-decoded dataset proxy (see decode)",
                  "success_env": float(senv.mean()),
                  "success": float(senv.mean()),
                  "qr_pred_t0_mean": float(qr0.mean())}
            fam[f"t{i}"] = rec
            table.append(("dirac", f"t{i}", "success", f"{rec['success']}"))
            print(f"[partition-zs][dirac] t{i}: succ={rec['success']} "
                  f"qr0={rec['qr_pred_t0_mean']:.3f} n_pos={n_pos}/{K} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success"] for v in fam.values() if v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["dirac"] = fam
        table.append(("dirac", "MEAN", "success", f"{fam['mean']}"))

    # ================= 2. dense =================
    if "dense" in families:
        fam = {}
        for i in task_ids:
            z_r = reward_conj_dense(protos, goal_coord[i], spec)
            n_pos = int((z_r > 0.5).sum())

            def rfn(coords, c=goal_coord[i]):
                return reward_conj_dense(coords, c, spec)

            (traj, ot, ln, senv, qr0), decode_info = rollout_for(
                z_r, rfn, pool(i), goal_obs[i])
            rec = {"diag": {"n_pos": n_pos, "n_neg": K - n_pos,
                            "z_r_mean": float(z_r.mean()), "z_r_max": float(z_r.max())},
                  "decode": decode_info,
                  "g_hat": "task goal_obs (exact)" if args.decode == "qr_greedy"
                           else "retrieval-decoded dataset proxy (see decode)",
                  "success_env": float(senv.mean()),
                  "success": float(senv.mean()),
                  "qr_pred_t0_mean": float(qr0.mean())}
            fam[f"t{i}"] = rec
            table.append(("dense", f"t{i}", "success", f"{rec['success']}"))
            print(f"[partition-zs][dense] t{i}: succ={rec['success']} "
                  f"qr0={rec['qr_pred_t0_mean']:.3f} n_pos={n_pos}/{K} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success"] for v in fam.values() if v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["dense"] = fam
        table.append(("dense", "MEAN", "success", f"{fam['mean']}"))

    # ================= 3. region =================
    if "region" in families:
        rng_reg = np.random.default_rng(args.seed + 2)
        centers, centers_obs, npos_c, rejected = draw_region_centers_with_obs(
            obs_all, coords_all, bounds, spec, relabel_coords, rng_reg, args.regions,
            min_pos=args.region_min_pos)
        v0 = task_ids[0]
        fam = {"probe_policy": geo.probe_centers, "region_variant": v0,
              "n_centers": int(len(centers)), "centers_rejected": int(rejected),
              "center_min_pos": int(args.region_min_pos)}
        print(f"[partition-zs][region] {len(centers)}/{args.regions} hindsight "
              f"centers ({rejected} rejected at min_pos={args.region_min_pos})",
              flush=True)
        for k_i in range(len(centers)):
            c, g_hat = centers[k_i], centers_obs[k_i]
            z_r = reward_conj_dirac(protos, c, spec)
            n_pos = int((z_r > 0).sum())

            def rfn(coords, c=c):
                return reward_conj_dirac(coords, c, spec)

            (_tj, ot, ln, _se, qr0), decode_info = rollout_for(z_r, rfn, pool(v0), g_hat)
            fin, anys = region_success_from_obs(ot, ln, c, spec)
            rec = {"center_coord": np.asarray(c).tolist(), "center_n_pos": int(npos_c[k_i]),
                  "diag": {"n_pos": n_pos, "n_neg": K - n_pos},
                  "decode": decode_info,
                  "success_final": float(fin.mean()), "success_any": float(anys.mean()),
                  "qr_pred_t0_mean": float(qr0.mean())}
            fam[f"c{k_i}"] = rec
            table.append(("region", f"c{k_i}", "success_final", f"{rec['success_final']}"))
            print(f"[partition-zs][region] c{k_i}: fin={rec['success_final']} "
                  f"any={rec['success_any']} n_pos={n_pos}/{K} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        cells = [v for k, v in fam.items() if k.startswith("c") and isinstance(v, dict)]
        vals = [v["success_final"] for v in cells]
        anyv = [v["success_any"] for v in cells]
        fam["mean_final"] = float(np.mean(vals)) if vals else None
        fam["mean_any"] = float(np.mean(anyv)) if anyv else None
        fam["degenerate_frac"] = 0.0
        results["families"]["region"] = fam
        table.append(("region", "MEAN", "success_final", f"{fam['mean_final']}"))

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec17_delta" / f"pzs_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\n[partition-zs] DONE ({time.time()-t00:.0f}s) -> {outp}\n", flush=True)
    print_table(table)
    close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
