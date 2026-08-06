#!/usr/bin/env python
"""scripts/rescuing_fb/ors_calibration_probe.py -- spec 19b sec.1, the P1.2 HARD
GATE (PAPER/Multistep-FB/specs/19-ors-program.md sec.1; contract:
specs/19b-ors-extractor-impl.md sec.1; conventions verified against the paper +
released code: specs/19b-ors-transcription.md).

WHAT THIS CHECKS: whether a frozen spec-19a `measure_density_behavior`
checkpoint's r_W ORDERING agrees with the dataset's own empirical discounted
future distance -- the shaping signal is only as good as the occupancy
ordering (spec 19 sec.1 P1.2), so this eval-only probe gates every later
extractor launch (spec 19b sec.4). Ground truth is the DATASET ITSELF, no
rollouts: for a sampled (s_i, a_i) and goal g, r_W(s_i, a_i, g) is compared
against the trajectory's own realized future
    Sigma_{k>=1} gamma^{k-1} * (-d^2(x_{i+k}, g))
truncated at a horizon and renormalized by the truncated discount mass -- an
unbiased-in-truncation empirical stand-in for the true occupancy expectation
E_{s+~d^piD(.|s,a)}[-d^2(s+,g)], which is exactly what r_W is supposed to
estimate (spec 19b sec.0 point 2; transcription note sec (f)).

FORMULA (spec 19b sec.0 point 2 / sec.2.3 -- the extractor's own r_W, reused
here unmodified so the probe tests the SAME quantity the extractor will
consume, not a different approximation of it):
    d^2(u, g)   = Sigma_b || (u_b - g_b) / eps_b ||^2       (per-block eps-scaled)
    w(s,a)      = mean_h softmax_j( F_D(s,a) . B_D(u~tilde_j) / tau )   (h in {0,1}
                  F-heads; MEAN of the two softmaxes -- spec 19b sec.2.3 explicitly:
                  "r_W is a reward, not a value estimate -- no min", the OPPOSITE
                  convention from eval_density_zeroshot.density_weights' min, which
                  is why this script does NOT import that function -- see below)
    r_W(s,a,g)  = ors_reward_scale * (1-gamma) * Sigma_j w_j(s,a) * (-d^2(u_j, g))
over a FIXED pool U_P drawn once from the dataset's object-block coords.
`ors_reward_scale` is a positive multiplicative constant and (1-gamma) is a
positive constant for gamma in (0,1): NEITHER changes a Spearman rank
correlation (Spearman is invariant to any strictly-increasing transform), so
the GATE STATISTIC in this script does not depend on --ors-reward-scale at
all -- it is exposed on the CLI purely so the reported r_W values match
whatever config the extractor will actually use, for readability of the
scatter JSON, not because it changes pass/fail.

REUSE, NOT REIMPLEMENTATION (task brief's explicit instruction): the towers
and their exact input conventions are called through
`GCIQLFlowAgent._raw_block_cols` / `_density_queries` / the agent's own
`density_f`/`density_b` module selection -- the SAME methods
`eval_density_zeroshot.py` uses for the (goal-conditioned) density readout
and training itself uses for the loss (spec 18/19a). The eps-scaled squared
distance reuses `env_geometry.HitSpec.scaled` (the repo's single hit-spec
definition) rather than re-deriving per-block division. The softmax
normalization reuses `eval_density_zeroshot.softmax_np` (the repo's one
numerically-stable softmax routine). The Spearman statistic reuses
`evals._profile_core._spearman` (the repo's one rank-correlation helper,
already used by 6+ other probe scripts -- grep "from evals._profile_core
import _spearman"). What is NOT reused: `eval_density_zeroshot.density_weights`
(that function's elementwise-MIN-over-heads is the DENSITY READOUT's own
pessimism convention for actor/eval Q-values -- spec 19b sec.2.3 explicitly
overrides this to MEAN for r_W, a different algorithm on the same towers, not
a duplicate of the same algorithm).

TWO AXES REPORTED (spec sec.1):
  STATE axis (the GATE): Spearman(r_W, ground truth) pooled over the sampled
    (s_i,a_i) rows and the 5 task goals + 20 random dataset goals, per env.
    GATE: Spearman >= 0.65. This is what P1.3 (the extractor wave) is gated
    on -- spec 19 sec.1 "else STOP and report".
  ACTION axis (diagnostic, NOT the gate): among the sampled rows, find
    near-duplicate-STATE groups (eps-scaled L2 within --action-state-tol)
    whose ACTIONS differ by at least --action-gap, and report Spearman
    restricted to those rows, plus how many such rows/groups exist (coverage
    reported honestly -- the spec's own expectation is that this is rare in a
    continuous manipulation action space; an empty or tiny action-axis sample
    is an EXPECTED, correctly-reported outcome, not a bug).

CONSISTENCY REFUSAL: `check_behavior_density_consistency` refuses, loudly, a
checkpoint whose recorded config is not `measure_density=True AND
measure_density_behavior=True` (both directions), mirroring
`eval_density_zeroshot.check_density_consistency` /
`eval_zeroshot_rewards.check_delta_consistency`'s two-branch precedent.
`--force` overrides with a loud warning, never silently.

UNVALIDATED END-TO-END: no `measure_density_behavior` checkpoint exists yet
at the time this script was written (spec 19a is landing in
tools/wandb_mode_shim/gciql_flow.py on a parallel track -- this script imports
`GCIQLFlowAgent._raw_block_cols`/`_density_queries`/`density_f`/`density_b`
LIVE, the same audited pattern `eval_density_zeroshot.py` used for spec 18
before its own checkpoint existed, so this script automatically tracks
whatever that agent lands rather than reimplementing it). Everything past
checkpoint restore (`main`, `build_ors_pool`, `compute_f_emb`) is therefore
untested against a REAL trained checkpoint and will fail loudly (missing
modules / config keys) until 19a lands. What IS tested now, per the task
brief: (1) the full pure-numpy layer (ground truth, index sampling, eps
distance, r_W algebra, action-axis grouping, Spearman gate) against
hand-computable references, with no jax/agent/dataset involved; (2) the
jax-dependent tower-calling layer against a STRUCTURALLY VALID but UNTRAINED
`measure_density_behavior=True` agent built in-process via
`GCIQLFlowAgent.create(...)` (the exact harness `tests/test_gciql_measure_
density.py` already uses for the sibling spec) -- this proves the plumbing
(shapes, tower selection, F-tower's no-goal-leg convention, determinism) is
correct even though an untrained agent's r_W is meaningless as a signal (its
own Spearman will NOT be expected to pass the 0.65 gate -- that is not
tested, and would not mean anything if it did).

CLI (CPU-only; GPUs carry training -- do not touch them):
  taskset -c 40-47 env OMP_NUM_THREADS=4 \\
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/ors_calibration_probe.py \\
    --run-dir '/home/adityamohan/exp_spec19a/.../GCIQL-CubeDouble-Flow-Measure-Density-Behavior-d50/*s0*' \\
    --checkpoint-step 500000 --n-samples 2000 --n-random-goals 20 \\
    --min-remaining 500 --horizon 500 --pool-size 4096
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

# Reused, not reimplemented (module-level imports below are jax-FREE -- verified:
# t6_gpi_eval.py, which eval_zeroshot_rewards.py imports, keeps every `import jax`
# inside function bodies; env_geometry.py and evals/_profile_core.py never import
# jax at all -- this script stays importable/testable under the plain .venv, same
# discipline as eval_density_zeroshot.py / eval_zeroshot_rewards.py):
from env_geometry import HitSpec, episode_bounds, geometry_for  # noqa: E402
from eval_zeroshot_rewards import eval_tasks_for  # noqa: E402
from evals._profile_core import _spearman  # noqa: E402

GATE_SPEARMAN = 0.65
DEFAULT_HORIZON = 500
DEFAULT_MIN_REMAINING = 500
DEFAULT_N_SAMPLES = 2000
DEFAULT_N_RANDOM_GOALS = 20
DEFAULT_POOL_SIZE = 4096


# ============================= consistency refusal =================================

class OrsCalibrationMismatchError(RuntimeError):
    """The checkpoint's recorded config is not a spec-19a `measure_density_behavior`
    run -- refusing to score r_W against untrained/wrong-mode density_f/density_b
    towers. Mirrors eval_density_zeroshot.DensityMismatchError /
    eval_zeroshot_rewards.DeltaMismatchError's two-branch discipline exactly."""


def check_behavior_density_consistency(ckpt_cfg: dict, force: bool = False,
                                       run_dir: str = "<run>") -> list:
    """Refuse a measure_density_behavior mismatch (checkpoint not trained with
    it, requested but absent) unless `force` (loud warning, never silent).
    Returns a list of warning strings to print (empty if nothing to warn
    about); raises OrsCalibrationMismatchError on an un-forced mismatch. Pure
    logic, no jax -- directly unit-testable, matching the mirrored
    precedents' own signature/test shape."""
    is_density = bool(ckpt_cfg.get("measure_density", False))
    is_behavior = bool(ckpt_cfg.get("measure_density_behavior", False))
    warnings = []
    if not (is_density and is_behavior):
        msg = (f"checkpoint {run_dir} was NOT trained with measure_density=True "
               f"AND measure_density_behavior=True (recorded "
               f"measure_density={is_density!r}, "
               f"measure_density_behavior={is_behavior!r}) -- this probe's r_W "
               "requires the FROZEN (F_D, B_D) BEHAVIOR-policy density towers "
               "(spec 19a). Scoring against a goal-conditioned measure_density "
               "checkpoint (or a non-density checkpoint) would either crash on "
               "missing modules or silently use the wrong F-tower input "
               "convention -- behavior F takes [s,a] only, goal-conditioned F "
               "takes [s,a,g] (spec 19a vs spec 18).")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are LIKELY GARBAGE): " + msg)
        else:
            raise OrsCalibrationMismatchError(msg + " Pass --force to override (not recommended).")
    return warnings


# ============================= pure numpy: geometry / ground truth =================

def eps_scaled_sq_dist(u_raw: np.ndarray, g_coords: np.ndarray, blocks) -> np.ndarray:
    """d^2(u, g) = Sigma_b ||(u_b - g_b)/eps_b||^2 (spec 19b sec.0 point 2), reusing
    `env_geometry.HitSpec.scaled` for the per-block division rather than
    re-deriving it. u_raw: [..., d_c] RAW (unscaled) block-column coords (same
    convention `GCIQLFlowAgent._raw_block_cols` produces); g_coords: [d_c] a
    single goal's block-column coords, broadcast over u_raw's leading axes.
    Returns [...] squared eps-scaled distance."""
    hs = HitSpec(tuple(blocks))
    u = np.asarray(u_raw, np.float64)
    g = np.asarray(g_coords, np.float64).reshape(-1)
    assert u.shape[-1] == g.shape[-1] == hs.dim, \
        f"coord dim mismatch: u={u.shape[-1]} g={g.shape[-1]} hit_spec dim={hs.dim}"
    diff = u - g
    scaled = hs.scaled(diff.reshape(-1, hs.dim)).reshape(diff.shape)
    return np.sum(scaled ** 2, axis=-1)


def sample_valid_indices(bounds: np.ndarray, seed: int, n_samples: int,
                         min_remaining: int = DEFAULT_MIN_REMAINING):
    """Seeded uniform sample (without replacement) from the UNION, over every
    trajectory, of rows with >= `min_remaining` SAME-TRAJECTORY steps
    remaining (spec 19b sec.1: "spread across trajectories, each with >=500
    same-trajectory steps remaining"). Uniform sampling over the union spreads
    draws across trajectories in proportion to each trajectory's own count of
    eligible rows (a trajectory shorter than `min_remaining` contributes zero
    rows) -- the explicit reading of "spread across trajectories" this
    function implements; NOT a stratified equal-count-per-episode scheme
    (recorded here so the choice is auditable, not silently assumed).

    bounds: [n_ep, 2] half-open [start, stop) row ranges (env_geometry.
    episode_bounds). Row `i` is eligible iff its OWN episode's last row
    (stop-1) satisfies `stop - 1 - i >= min_remaining`.

    Returns (idxs [n<=n_samples] int64 sorted ascending,
             stops [same len] int64 -- each row's own episode's `stop`,
             so the ground-truth window i+1..stop-1 is available for free
             without a second searchsorted pass).
    """
    bounds = np.asarray(bounds, np.int64)
    rows, stop_of_row = [], []
    for start, stop in bounds:
        last_valid = stop - 1 - min_remaining
        if last_valid >= start:
            ep_rows = np.arange(start, last_valid + 1)
            rows.append(ep_rows)
            stop_of_row.append(np.full(len(ep_rows), stop, np.int64))
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pool = np.concatenate(rows)
    pool_stops = np.concatenate(stop_of_row)
    rng = np.random.default_rng(seed)
    if len(pool) <= n_samples:
        order = np.arange(len(pool))
    else:
        order = rng.choice(len(pool), size=n_samples, replace=False)
    order = np.sort(order)
    return pool[order], pool_stops[order]


def ground_truth_discounted_distance(coords_all: np.ndarray, blocks, goal_coords: np.ndarray,
                                     idx: int, ep_stop: int, gamma: float,
                                     horizon: int = DEFAULT_HORIZON) -> float:
    """Scalar reference path (spec 19b sec.1): the empirical discounted future
    distance from row `idx`, TRUNCATED at `horizon` steps or the trajectory's
    own end (whichever binds first), RENORMALIZED by the truncated discount
    mass:
        K   = min(horizon, ep_stop - 1 - idx)
        val = ( Sigma_{k=1}^{K} gamma^(k-1) * (-d^2(x_{idx+k}, g)) )
              / ( Sigma_{k=1}^{K} gamma^(k-1) )                    [= (1-gamma^K)/(1-gamma)]

    Under `sample_valid_indices`'s own >=500-remaining contract, K == horizon
    for every sampled row (the renormalizing denominator is then a CONSTANT
    across rows and does not affect a Spearman rank correlation) -- this
    function still implements the general (possibly-truncated, K<horizon)
    formula correctly and unconditionally, both because it is the literal
    spec and because `ground_truth_batch` below dispatches any row that
    genuinely has K<horizon (e.g. if called with a looser sampler) through
    this exact function. `idx` must satisfy `idx < ep_stop - 1` (there is at
    least one future row in its own trajectory) -- the sampler's own
    min_remaining>=1 contract guarantees this; asserted here regardless."""
    remaining = int(ep_stop) - 1 - int(idx)
    assert remaining >= 1, f"row {idx} has no future row in its own episode (stop={ep_stop})"
    K = min(int(horizon), remaining)
    ks = np.arange(1, K + 1)
    future_coords = coords_all[idx + ks]                      # [K, d_c]
    d2 = eps_scaled_sq_dist(future_coords, goal_coords, blocks)  # [K]
    discounts = np.asarray(gamma, np.float64) ** (ks - 1)
    total = float(np.sum(discounts * (-d2)))
    mass = float(np.sum(discounts))
    assert mass > 0.0
    return total / mass


def ground_truth_batch(coords_all: np.ndarray, blocks, goal_coords: np.ndarray,
                       idxs: np.ndarray, stops: np.ndarray, gamma: float,
                       horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    """Vectorized ground truth for every (idx, goal) row, one goal at a time
    (goal_coords is a SINGLE goal here; call once per goal -- mirrors how
    `eps_scaled_sq_dist`/r_W are also computed per goal against a shared
    pool). Rows with the full `horizon` of remaining steps (the sampler's own
    contract -- the common/production case) go through one vectorized batch;
    any row with fewer remaining steps (K<horizon, e.g. a looser sampler) is
    spliced in via the scalar reference path -- exact agreement with
    `ground_truth_discounted_distance` is a unit test, not merely asserted."""
    idxs = np.asarray(idxs, np.int64)
    stops = np.asarray(stops, np.int64)
    n = len(idxs)
    out = np.empty(n, np.float64)
    if n == 0:
        return out
    remaining = stops - 1 - idxs
    full = remaining >= horizon
    if np.any(full):
        offs = np.arange(1, horizon + 1)
        fut_idx = idxs[full][:, None] + offs[None, :]              # [Nf, horizon]
        fut_coords = coords_all[fut_idx]                            # [Nf, horizon, d_c]
        d2 = eps_scaled_sq_dist(fut_coords, goal_coords, blocks)     # [Nf, horizon]
        discounts = np.asarray(gamma, np.float64) ** (offs - 1)      # [horizon]
        mass = float(np.sum(discounts))
        out[full] = np.sum(discounts[None, :] * (-d2), axis=1) / mass
    for i in np.flatnonzero(~full):
        out[i] = ground_truth_discounted_distance(
            coords_all, blocks, goal_coords, idxs[i], stops[i], gamma, horizon)
    return out


def rw_from_weights(w: np.ndarray, d2_pool: np.ndarray, gamma: float,
                    scale: float = 1.0) -> np.ndarray:
    """r_W(s,a,g) = scale * (1-gamma) * Sigma_j w_j * (-d2_pool_j) (spec 19b
    sec.0 point 2 / sec.2.3). w: [..., P]; d2_pool: [P] (one goal, broadcasts
    over w's leading axes)."""
    w = np.asarray(w, np.float64)
    d2 = np.asarray(d2_pool, np.float64)
    return float(scale) * (1.0 - float(gamma)) * np.sum(w * (-d2), axis=-1)


def ors_weights(f_emb: np.ndarray, b_pool: np.ndarray, tau: float) -> np.ndarray:
    """w(s,a) = mean_h softmax_j(F_D(s,a)_h . B_D(u_j) / tau) -- spec 19b
    sec.2.3's explicit MEAN-of-two-heads convention (NOT
    eval_density_zeroshot.density_weights' elementwise-MIN, a different
    algorithm's pessimism convention on the same towers; see module
    docstring). Reuses `eval_density_zeroshot.softmax_np` for the
    normalization itself (imported lazily -- see module docstring "REUSE" for
    why eval_density_zeroshot.py stays import-safe at module level; loading
    it here would trigger its own jax-free import chain, harmless, but the
    import is deferred anyway to keep this function's own dependency surface
    minimal and mirror the "jax-dependent-adjacent code stays lazy" style).

    f_emb: [H, ..., d] -- H (F-head count, always 2) LEADING and EXPLICIT,
    same no-guessing discipline as density_weights (a caller must pass
    f_emb[None] for a genuinely single-head net rather than relying on shape
    inference). b_pool: [P, d]. Returns w [..., P]."""
    from eval_density_zeroshot import softmax_np
    f_emb = np.asarray(f_emb, np.float64)
    assert f_emb.ndim >= 2, \
        "f_emb must carry an explicit leading head axis [H, ..., d] -- wrap a " \
        "single-head embedding as f_emb[None] rather than relying on shape guessing"
    b_pool = np.asarray(b_pool, np.float64)
    logits = np.einsum("h...d,pd->h...p", f_emb, b_pool) / float(tau)   # [H,...,P]
    w_h = softmax_np(logits, axis=-1)                                   # [H,...,P]
    return w_h.mean(axis=0)


def spearman_gate(rw: np.ndarray, gt: np.ndarray, threshold: float = GATE_SPEARMAN):
    """Reuses `evals._profile_core._spearman` (the repo's one rank-correlation
    helper). Returns (rho: float|None [None iff either side is degenerate --
    _spearman's own nan convention], passed: bool)."""
    rho = _spearman(np.asarray(gt, np.float64), np.asarray(rw, np.float64))
    passed = bool((not np.isnan(rho)) and rho >= threshold)
    return (None if np.isnan(rho) else float(rho)), passed


def find_action_axis_groups(coords_raw: np.ndarray, actions: np.ndarray, blocks,
                            state_tol: float = 0.05, action_gap: float = 0.1):
    """Near-duplicate-(s,.)-pair groups for the ACTION-axis diagnostic (spec
    19b sec.1: "states with >=2 near-duplicate (s,.) pairs are rare -- report
    coverage honestly"). Groups rows whose eps-SCALED state coords (unit-
    radius, so the tolerance means the same fraction-of-hit-radius across
    heterogeneous per-block eps -- e.g. scene's 0.4/0.5/0.72/0.6) are within
    `state_tol` of each other (union-find over a KD-tree radius query), then
    KEEPS only groups whose max pairwise ACTION L2 distance is >= `action_gap`
    (drops near-identical-action duplicates, which carry no action-axis
    ordering information).

    coords_raw: [N, d_c] RAW block coords of the sampled rows (spec.coords
    output); actions: [N, act_dim]. Returns a list of index arrays (each
    indexing INTO coords_raw/actions, len>=2, groups non-overlapping) --
    empty list is an EXPECTED, correctly-reported outcome on a typical
    continuous-action manipulation dataset, not a bug."""
    from scipy.spatial import cKDTree

    hs = HitSpec(tuple(blocks))
    coords = hs.scaled(np.asarray(coords_raw, np.float64))
    actions = np.asarray(actions, np.float64)
    n = len(coords)
    if n < 2:
        return []
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=state_tol)
    if not pairs:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)
    members_by_root = {}
    for i in range(n):
        members_by_root.setdefault(find(i), []).append(i)
    out = []
    for members in members_by_root.values():
        if len(members) < 2:
            continue
        members = np.asarray(members, np.int64)
        acts = actions[members]
        # max pairwise action distance within the group (small groups -- O(m^2) is fine)
        diffs = acts[:, None, :] - acts[None, :, :]
        max_gap = float(np.sqrt(np.sum(diffs ** 2, axis=-1)).max()) if len(members) > 1 else 0.0
        if max_gap >= action_gap:
            out.append(members)
    return out


# ============================= jax-dependent (lazy import) ==========================
# Calls the agent's OWN _raw_block_cols/_density_queries/density_f/density_b -- the
# SAME code path training uses (task brief). None of this is reachable/testable
# against a REAL checkpoint until spec 19a lands -- see module docstring
# "UNVALIDATED"; the plumbing is proven now against an in-process untrained agent
# (tests/test_ors_calibration_probe.py), same precedent as eval_density_zeroshot.py.

def build_ors_pool(agent, blocks, pool_obs: np.ndarray):
    """One forward pass: raw block coords U_raw [P,d_c] (RAW, unscaled --
    `GCIQLFlowAgent._raw_block_cols`, the twin `_delta_phi`/`eps_scaled_sq_dist`
    both expect) and B_D(U~P) [P,d] via the agent's OWN `density_b` tower
    (eps-scaled input, `_density_queries` -- do NOT double-scale, spec 18
    sec.4.2's called-out trap, same discipline as eval_density_zeroshot.
    build_pool)."""
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    pool_obs_j = jnp.asarray(pool_obs, jnp.float32)
    u_raw = np.asarray(GCIQLFlowAgent._raw_block_cols(pool_obs_j, blocks), np.float64)
    u_sc = GCIQLFlowAgent._density_queries(pool_obs_j, blocks)
    b_emb = np.asarray(agent.network.select("density_b")(u_sc), np.float64)
    return u_raw, b_emb


def compute_f_emb(agent, obs: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """F_D(s,a) -- the BEHAVIOR tower's own input convention (spec 19a: no
    goal leg, [s,a] only -- opposite of spec 18's goal-conditioned [s,a,g]).
    Returns [2, N, d] (2 F-heads, leading axis, matching `ors_weights`'s
    explicit-head-axis contract)."""
    import jax.numpy as jnp
    f_emb = np.asarray(agent.network.select("density_f")(
        jnp.concatenate([jnp.asarray(obs, jnp.float32), jnp.asarray(acts, jnp.float32)], -1)),
        np.float64)
    return f_emb


# ============================= CLI ==================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json; first match "
                         "used; must be a measure_density=True, "
                         "measure_density_behavior=True checkpoint)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--env", default="", help="override env (default: flags.json env_name)")
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    ap.add_argument("--n-random-goals", type=int, default=DEFAULT_N_RANDOM_GOALS)
    ap.add_argument("--min-remaining", type=int, default=DEFAULT_MIN_REMAINING,
                    help="sampler's own-trajectory-steps-remaining floor (spec sec.1)")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    help="ground-truth truncation horizon (spec sec.1)")
    ap.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--ors-reward-scale", type=float, default=1.0,
                    help="cosmetic only -- does not change the Spearman gate "
                         "(rank-invariant to positive scale); see module docstring")
    ap.add_argument("--action-state-tol", type=float, default=0.05,
                    help="eps-scaled L2 radius for the action-axis near-duplicate-"
                         "state search")
    ap.add_argument("--action-gap", type=float, default=0.1,
                    help="minimum pairwise action L2 distance to keep an action-axis group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="override the measure_density_behavior consistency refusal "
                         "(loud warning, numbers likely garbage)")
    ap.add_argument("--out", default="",
                    help="output JSON (default results/spec19_ors/calib_<run>.json)")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])

    for w in check_behavior_density_consistency(cfg, force=args.force, run_dir=str(run_dir)):
        print(f"[ors-calib] {w}", flush=True)

    env_name = args.env or flags["env_name"]
    gamma = float(cfg["discount"])
    tau = float(cfg.get("measure_density_temp", 1.0))
    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    blocks = spec.blocks

    import ogbench

    t0 = time.time()
    eval_tasks = eval_tasks_for(env_name)
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    obs_all = np.asarray(ds1["observations"], np.float32)
    acts_all = np.asarray(ds1["actions"], np.float32)
    terminals = np.asarray(ds1["terminals"])
    bounds = episode_bounds(terminals)
    coords_all = spec.coords(obs_all)
    obs_dim, act_dim = obs_all.shape[-1], acts_all.shape[-1]
    print(f"[ors-calib] dataset loaded: {len(obs_all)} rows, {len(bounds)} episodes "
          f"({time.time()-t0:.0f}s)", flush=True)

    idxs, stops = sample_valid_indices(bounds, args.seed, args.n_samples,
                                       min_remaining=args.min_remaining)
    assert len(idxs) > 0, \
        f"no dataset row has >={args.min_remaining} same-trajectory steps remaining"
    print(f"[ors-calib] sampled {len(idxs)}/{args.n_samples} valid rows "
          f"(min_remaining={args.min_remaining})", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    pool_idx = rng.choice(len(obs_all), size=min(args.pool_size, len(obs_all)),
                          replace=(args.pool_size > len(obs_all)))
    pool_obs = obs_all[pool_idx]

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)
    print(f"[ors-calib] restored {run_dir.name} step={args.checkpoint_step} "
          f"env={env_name} gamma={gamma} tau={tau} "
          f"hit_spec={cfg.get('measure_hit_spec')!r} "
          f"({time.time()-t0:.0f}s)", flush=True)

    u_raw, b_pool = build_ors_pool(agent, blocks, pool_obs)
    f_emb = compute_f_emb(agent, obs_all[idxs], acts_all[idxs])       # [2, N, d]
    w = ors_weights(f_emb, b_pool, tau)                                # [N, P]
    print(f"[ors-calib] pool + weights built ({time.time()-t0:.0f}s)", flush=True)

    goal_coords_by_name = {}
    for i in range(1, 6):
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_coords_by_name[f"task{i}"] = spec.coords(np.asarray(info["goal"])[None])[0]
        env.close()
    rng_g = np.random.default_rng(args.seed + 2)
    rand_goal_idx = rng_g.choice(len(obs_all), size=args.n_random_goals, replace=False)
    for j, gi in enumerate(rand_goal_idx):
        goal_coords_by_name[f"rand{j}"] = coords_all[gi]

    per_goal, all_rw, all_gt = {}, [], []
    for gname, gcoords in goal_coords_by_name.items():
        d2_pool = eps_scaled_sq_dist(u_raw, gcoords, blocks)                  # [P]
        rw = rw_from_weights(w, d2_pool, gamma, scale=args.ors_reward_scale)  # [N]
        gt = ground_truth_batch(coords_all, blocks, gcoords, idxs, stops, gamma,
                                horizon=args.horizon)                        # [N]
        rho, passed = spearman_gate(rw, gt)
        per_goal[gname] = {"spearman": rho, "n": int(len(idxs)), "gate_passed": passed}
        all_rw.append(rw)
        all_gt.append(gt)

    all_rw_arr, all_gt_arr = np.concatenate(all_rw), np.concatenate(all_gt)
    pooled_rho, pooled_pass = spearman_gate(all_rw_arr, all_gt_arr)
    print(f"[ors-calib] STATE-axis (GATE) pooled Spearman = {pooled_rho} "
          f"(n={len(all_rw_arr)}) -> {'PASS' if pooled_pass else 'FAIL'} "
          f"(threshold {GATE_SPEARMAN})", flush=True)

    groups = find_action_axis_groups(coords_all[idxs], acts_all[idxs], blocks,
                                     state_tol=args.action_state_tol,
                                     action_gap=args.action_gap)
    n_action_rows = int(sum(len(g) for g in groups))
    action_result = {
        "n_groups": len(groups), "n_rows_covered": n_action_rows,
        "coverage_fraction": n_action_rows / max(1, len(idxs)),
    }
    if groups:
        group_rows = np.concatenate(groups)
        arw, agt = [], []
        for gcoords in goal_coords_by_name.values():
            d2_pool = eps_scaled_sq_dist(u_raw, gcoords, blocks)
            rw_full = rw_from_weights(w, d2_pool, gamma, scale=args.ors_reward_scale)
            gt_full = ground_truth_batch(coords_all, blocks, gcoords, idxs, stops, gamma,
                                         horizon=args.horizon)
            arw.append(rw_full[group_rows])
            agt.append(gt_full[group_rows])
        arw, agt = np.concatenate(arw), np.concatenate(agt)
        a_rho, a_pass = spearman_gate(arw, agt)
        action_result["spearman"] = a_rho
        action_result["gate_passed_informational_only"] = a_pass
    else:
        action_result["spearman"] = None
    print(f"[ors-calib] ACTION-axis (diagnostic, NOT the gate): "
          f"{action_result['n_groups']} groups, "
          f"{action_result['coverage_fraction']:.4f} coverage, "
          f"Spearman={action_result['spearman']}", flush=True)

    results = {
        "meta": {
            "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
            "env": env_name, "gamma": gamma, "tau": tau,
            "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
            "n_samples_requested": args.n_samples, "n_samples_used": int(len(idxs)),
            "min_remaining": args.min_remaining, "horizon": args.horizon,
            "pool_size": int(len(u_raw)), "ors_reward_scale": args.ors_reward_scale,
            "n_goals": len(goal_coords_by_name), "n_task_goals": 5,
            "n_random_goals": args.n_random_goals, "seed": args.seed,
            "formula": "r_W = scale*(1-gamma)*sum_j mean_h softmax_j(F_D(s,a)_h.B_D(u_j)/tau) "
                      "* (-d2_eps(u_j,g)); ground truth = truncated/renormalized "
                      "discounted future d2_eps under the dataset trajectory "
                      "(spec 19b sec.0/1; PAPER/Multistep-FB/specs/19b-ors-transcription.md)",
            "gate": f"Spearman(STATE axis, pooled over goals) >= {GATE_SPEARMAN}",
        },
        "state_axis": {"pooled": {"spearman": pooled_rho, "n": int(len(all_rw_arr)),
                                  "gate_passed": pooled_pass},
                      "per_goal": per_goal},
        "action_axis": action_result,
    }
    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec19_ors" / f"calib_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[ors-calib] DONE ({time.time()-t0:.0f}s) -> {outp}\n", flush=True)
    print(f"[ors-calib] {'GATE PASSED' if pooled_pass else 'GATE FAILED'}: "
          f"state-axis Spearman={pooled_rho} vs threshold {GATE_SPEARMAN}", flush=True)
    return 0 if pooled_pass else 1


if __name__ == "__main__":
    sys.exit(main())
