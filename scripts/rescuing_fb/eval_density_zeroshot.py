#!/usr/bin/env python
"""scripts/rescuing_fb/eval_density_zeroshot.py -- spec 18 sec.6 zero-shot
multi-reward readout for `measure_density` checkpoints.

Background: `PAPER/Multistep-FB/specs/18-density-measure.md` sec.0/1/6. Spec
18 replaces 17b's FIXED K-prototype simplex with a two-tower score
`f(s,a,g,u) = F(s,a,g).B(u)/tau` over ARBITRARY object-coordinate queries `u`,
softmax-normalized over a query set resampled from the dataset object
marginal rho_x every training step. There is no fixed partition, no .npy
prototype file, no checkpoint<->file coupling (17b's audit finding #2 -- see
eval_partition_zeroshot.py's module docstring -- disappears structurally).
At eval time the "partition" becomes a POOL of P draws from rho_x (this
script's --pool_size), and the pool count M is a Monte-Carlo sample size, not
a spatial resolution (spec sec.0): queries change every draw, eval can use a
much larger pool than training ever saw, and growing P only reduces
self-normalization/integration error (spec sec.1: O(1/M)) -- it does not
change the represented space.

This script GENERALIZES `eval_partition_zeroshot.py` (read first; it is the
audited precedent this shares its harness conventions with -- CLI shape,
sys.path setup, retrieval-decode mechanism, checkpoint-config-not-CLI-default
discipline, output JSON shape) for the density object. It reuses, not
reimplements, `draw_region_centers_with_obs` / `select_retrieval_candidate`
from that module (both are generic w.r.t. the measure parameterization) and
`reward_conj_dirac` / `reward_conj_dense` / `region_success_from_obs` /
`eval_tasks_for` / `print_table` from `eval_zeroshot_rewards.py`, and
`geometry_for` / `episode_bounds` from `env_geometry.py` -- the single hit-
spec definition every eval script in the repo shares.

SCOPE NARROWING vs the partition harness (deliberate, stated up front): the
task brief for this script asks for "the spec-14-comparable retrieval policy
decode, same as the partition harness" -- singular. Unlike
eval_partition_zeroshot.py's `--decode {retrieval,qr_greedy}`, this script
ONLY implements retrieval decode: one goal-proxy observation g_hat is decoded
ONCE per family/task via this readout's own PRIMARY score, then a plain
fixed-goal rollout runs for the whole episode (no per-step re-scoring). If a
qr_greedy analogue is wanted later it is a straightforward addition mirroring
`batched_qr_rollout`, deliberately deferred to keep this (unvalidated, no
checkpoint yet) script's scope matched to what was asked.

TWO READOUTS (spec sec.1/6 -- report BOTH, quote SEPARATELY, never conflated):

  Pool: U_P = {u_1..u_P} ~ rho_x (dataset object-block coords, seeded rng);
  B-embeddings B(u~tilde_j) precomputed ONCE per pool. For a query (s,a,g):
    f(s,a,g,.) -> F-tower embedding, 2 heads (spec sec.4.6: "F 2-headed, B
      single"); w_h = softmax_j( f_h . B(u~tilde_j) / tau ) per head SEPARATELY,
      normalized over j in {1..P}, then w = elementwise MIN over the two
      heads (spec sec.3 "double_q = min over the two w's at actor/eval" --
      literal: min is applied to the two PROBABILITY VECTORS, not to a
      downstream dot-product scalar). This is what makes `w` a fixed,
      r-INDEPENDENT object -- see "why the linearity split is real" below.
    kappa(x0|U_P) = GCIQLFlowAgent._delta_phi(x0_obs, U_raw, blocks) -- the
      SAME kernel training's immediate term uses, evaluated at the CURRENT
      state x0 (not the successor) against the raw (un-scaled) pool coords;
      _delta_phi does its own eps-scaling internally -- the spec's called-out
      double-scaling trap this script avoids by always passing RAW block
      rows to `_delta_phi` and only ever eps-scaling via the agent's own
      `_density_queries` when building the B-tower's input.

  PRIMARY   Qhat_r = ( r(x0) + (w - kappa(x0|U_P))^T r(U_P) ) / (1 - gamma)
            -- exactly part's (17b) equation with prototypes -> pool.
  SECONDARY mu   = sum_j w_j B(u~tilde_j);  mu0 = kappa(x0|U_P)^T B(U~P)
            w_r ~= ridge-fit of r on B(U~P)   (--ridge, default 1e-4)
            Qhat_r ~= ( r(x0) + (mu - mu0)^T w_r ) / (1 - gamma)
            -- the "compact reward code" repair of FB's dot-product theorem,
               lost to the partition function (spec sec.1).

WHY THE LINEARITY SPLIT IS REAL (not just spec-asserted): `w`, `kappa(x0|.)`,
and `B(U~P)` never depend on the reward `r` -- only on (s,a,g), x0, and the
pool. PRIMARY is therefore a PURE DOT PRODUCT of a fixed vector against
`r(U_P)`: exactly, structurally, linear in r for ANY r, ANY pool, in EXACT
arithmetic (float round-off aside, and even that is ~1e-14 relative -- see
tests/test_eval_density_zeroshot.py). SECONDARY routes r through a
regularized least-squares SOLVE (`w_r = (B^T B + ridge*I)^-1 B^T r(U_P)`);
for a FIXED pool and FIXED ridge this solve is *also* a linear map of
r(U_P) in exact arithmetic, but it is not a bare dot product -- it is a
matrix inversion, and its floating-point error grows with B's conditioning
(demonstrated, not asserted, in the test suite: ill-conditioned B gives a
combined-vs-separate discrepancy ~1e3x larger than PRIMARY's for the same
inputs). This is the concrete, measurable sense in which spec sec.1 writes
PRIMARY's formula with "=" and SECONDARY's with "~=": SECONDARY is an
approximation of the true integral by construction (r is only approximately
representable as w_r^T B) even though, as an ESTIMATOR PIPELINE, it is
linear in the SAME sense PRIMARY is for a fixed pool/ridge. Both facts are
recorded, honestly, in the test suite rather than one being silently assumed.

--pool_sweep (spec sec.6/7 G3): recomputes the PRIMARY readout at each pool
size using PREFIXES of ONE shared pool draw (not fresh re-draws per size).
This is deliberate and exact, not a shortcut: a softmax vector restricted to
a leading subset and renormalized by THAT SUBSET's own sum is mathematically
IDENTICAL to computing the softmax over just that subset from scratch (the
defining property of softmax), so every swept P's `w`/`kappa` come for free
from ONE forward pass over the largest pool -- no extra tower calls. Report
is the t=0 PRIMARY Q-ESTIMATE drift across pool sizes for every cell already
being evaluated (cheap: no extra rollouts). This is the "query count is a
Monte-Carlo sample size" evidence (module docstring above) -- it is NOT the
same thing as full-rollout SUCCESS-RATE stability, which is what spec sec.7
gate G3 ultimately needs (re-decoding + re-rolling at each P) -- pass
--pool_sweep_full_rollout to additionally get that (expensive: re-runs
retrieval decode + a full episode batch at every swept size; off by default,
per the CPU-only / no-full-eval-sweep hard rule this script was built under).

CONSISTENCY REFUSAL: this script ONLY implements the density readout.
`check_density_consistency` (mirrors `eval_zeroshot_rewards.check_delta_
consistency`'s two-branch shape exactly) refuses, loudly, to evaluate a
checkpoint whose recorded config is not `measure_density=True`, in EITHER
direction (requested-but-absent / present-but-not-requested -- the latter is
unreachable from this script's own CLI, which always requests density=True,
but is kept as an independently unit-testable branch of the SAME function,
matching the mirrored precedent's shape) -- `--force` overrides with a loud
warning, never a silent one.

UNVALIDATED: no `measure_density` checkpoint exists yet (spec 18 sec.4 is
being implemented in tools/wandb_mode_shim/gciql_flow.py by a parallel
agent, not this one -- this script imports `GCIQLFlowAgent._density_queries`
/ the `density_f` / `density_b` network modules LIVE, the same audited
pattern eval_partition_zeroshot.py uses for `_delta_phi` / `delta_head`, so
this script automatically tracks whatever that agent lands, rather than
reimplementing it). Everything past checkpoint restore (`main`,
`decode_goal_retrieval_density`, `batched_goal_rollout_density`,
`density_weights_live`) is therefore untested end-to-end and WILL fail
loudly (AttributeError / KeyError on the missing modules) until that lands.
The readout ALGEBRA (`compute_density_primary`, `ridge_fit_wr`,
`compute_density_secondary`, `pool_sweep_primary`) is pure numpy, jax-free,
and IS tested now -- see tests/test_eval_density_zeroshot.py.

Run (CPU-only rollouts; GPU 0 carries a training job -- do not touch it):
  taskset -c 32-39 env OMP_NUM_THREADS=4 \\
    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_density_zeroshot.py \\
    --run-dir '/home/adityamohan/exp_spec18/.../GCIQL-CubeSingle-Flow-Measure-Density-d50/*s0*' \\
    --checkpoint-step 500000 --families dirac,dense,region --episodes 25 --horizon 0 \\
    --pool_size 4096 --pool_sweep 256,1024,4096,16384
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

# Reused, not reimplemented: reward primitives + region-center/success helpers
# from the proven spec 07/14 harness, the single geometry definition, and the
# generic (measure-parameterization-independent) region/retrieval helpers
# audited into the 17b partition harness.
from eval_zeroshot_rewards import (  # noqa: E402
    eval_tasks_for,
    print_table,
    region_success_from_obs,
    reward_conj_dense,
    reward_conj_dirac,
)
from env_geometry import episode_bounds, geometry_for  # noqa: E402
from eval_partition_zeroshot import (  # noqa: E402
    draw_region_centers_with_obs,
    select_retrieval_candidate,
)

FAMILIES = ("dirac", "dense", "region")


# ======================= consistency refusal =======================================

class DensityMismatchError(RuntimeError):
    """The checkpoint's recorded `measure_density` config disagrees with what
    this script needs. Spec 18's density_f/density_b towers are a DIFFERENT
    set of trained parameters than 17/17a/17b's measure/delta_head/delta
    modules -- silently scoring against a checkpoint that never trained them
    would either crash on missing params or, far worse, silently read
    untrained/absent weights as a real signal. Mirrors
    eval_zeroshot_rewards.DeltaMismatchError."""


def check_density_consistency(ckpt_is_density: bool, requested_density: bool = True,
                              force: bool = False, run_dir: str = "<run>") -> list:
    """Refuse a density/checkpoint-config mismatch in EITHER direction unless
    `force` is set (a loud warning, never a silent override) -- mirrors
    `eval_zeroshot_rewards.check_delta_consistency`'s exact two-branch shape.
    `requested_density` is always True from this script's own CLI (there is
    no --density flag to get wrong, unlike 17's shared --delta harness), but
    is kept as a parameter -- not hardcoded inline -- so BOTH directions are
    independently unit-testable, matching the mirrored precedent's own
    signature and test coverage. Returns a list of warning strings to print
    (empty if nothing to warn about); raises DensityMismatchError on an
    un-forced mismatch. Pure logic, no jax/ogbench -- directly unit-testable.
    """
    warnings = []
    if requested_density and not ckpt_is_density:
        msg = (f"checkpoint {run_dir} was NOT trained with measure_density=True "
               f"(recorded config measure_density={ckpt_is_density!r}) but this "
               "script's PRIMARY/SECONDARY readouts (spec 18 sec.1/6) require the "
               "trained density_f/density_b towers -- evaluating them against a "
               "checkpoint that never trained those modules (e.g. plain "
               "measure_only, measure_delta, or measure_delta_partition) would "
               "either crash on missing params or silently score against "
               "untrained/randomly-initialized weights.")
        if force:
            warnings.append("FORCE OVERRIDE (numbers below are LIKELY GARBAGE): " + msg)
        else:
            raise DensityMismatchError(msg + " Pass --force to override (not recommended).")
    if ckpt_is_density and not requested_density:
        msg = (f"checkpoint {run_dir} WAS trained with measure_density=True but "
               "this call requested requested_density=False -- the trained "
               "density_f/density_b towers exist and would be silently ignored.")
        if force:
            warnings.append("FORCE OVERRIDE: " + msg)
        else:
            raise DensityMismatchError(msg + " Pass --force to override (not recommended).")
    return warnings


# ======================= readout algebra (pure numpy, unit-tested) =================

def softmax_np(x, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax, float64. The single normalization routine
    every readout function below shares -- so a normalization bug can only
    exist in one place."""
    x = np.asarray(x, np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def density_weights(f_emb: np.ndarray, b_pool: np.ndarray, temp: float,
                    double_q: bool = True) -> np.ndarray:
    """w (spec sec.1/6): softmax_j( f_h . B(u_j) / temp ) per F-head h, THEN
    (spec sec.3 "double_q = min over the two w's") elementwise MIN over heads
    -- min is applied to the two PROBABILITY VECTORS, not to a downstream
    dot-product scalar (that distinction is what makes `w` reward-
    independent -- see module docstring "why the linearity split is real").

    f_emb: [H, ..., d] -- H is ALWAYS the LEADING axis, supplied EXPLICITLY by
      the caller (F is always the 2-headed ensemble per spec sec.4.6; a
      single-head/ablation net still passes H=1 explicitly, e.g. `f_emb[None]`).
      There is deliberately no ndim-based auto-detection of "did the caller
      include a head axis": a single query's [H, d] (H=2 heads, no separate
      batch axis) is INDISTINGUISHABLE by ndim alone from a headless batch
      [N, d] whenever N happens to equal H (e.g. exactly 2 queries) --
      auto-detecting silently mis-shapes that case, so this function refuses
      to guess. "..." may be empty (a single query: f_emb [H,d] -> w [P]) or
      any number of batch axes (f_emb [H,N,d] -> w [N,P]).
    b_pool: [P, d] B-tower embeddings of the query pool (precomputed once,
      shared across every query this pool is scored against). Returns w
      [..., P] (the "..." batch shape from f_emb, head axis consumed by the
      min/select below).
    """
    f_emb = np.asarray(f_emb, np.float64)
    assert f_emb.ndim >= 2, \
        "f_emb must carry an explicit leading head axis [H, ..., d] -- wrap a " \
        "single-head embedding as f_emb[None] rather than relying on shape guessing"
    b_pool = np.asarray(b_pool, np.float64)
    logits = np.einsum("h...d,pd->h...p", f_emb, b_pool) / float(temp)   # [H,...,P]
    w_h = softmax_np(logits, axis=-1)                   # [H,...,P], each head sums to 1
    if double_q and w_h.shape[0] > 1:
        return w_h.min(axis=0)
    return w_h[0]


def compute_density_primary(r_x0, w: np.ndarray, kappa_x0: np.ndarray,
                            r_pool: np.ndarray, gamma: float) -> np.ndarray:
    """PRIMARY (spec sec.1/6, "=", part's eq. with protos -> pool):
        Qhat_r = ( r(x0) + (w - kappa(x0|U_P))^T r(U_P) ) / (1 - gamma)
    w, kappa_x0, r_pool: [..., P] (or [P]) broadcastable over the last (pool)
    axis; r_x0: [...] (or scalar). EXACTLY linear in r_pool AND r_x0 for
    FIXED w/kappa_x0 (both are r-independent by construction) -- a plain
    weighted dot product, no matrix solve anywhere in this function.
    """
    r_x0 = np.asarray(r_x0, np.float64)
    w = np.asarray(w, np.float64)
    kappa_x0 = np.asarray(kappa_x0, np.float64)
    r_pool = np.asarray(r_pool, np.float64)
    score = np.sum((w - kappa_x0) * r_pool, axis=-1)
    return (r_x0 + score) / (1.0 - gamma)


def ridge_fit_wr(b_pool: np.ndarray, r_pool: np.ndarray, ridge: float) -> np.ndarray:
    """w_r = argmin_w || B w - r ||^2 + ridge*||w||^2 over the pool, closed
    form: w_r = (B^T B + ridge*I)^-1 B^T r. A FIXED linear map of r_pool for
    fixed b_pool/ridge (the matrix (B^T B + ridge*I)^-1 B^T never depends on
    r) -- see module docstring for why this still does not make SECONDARY's
    numerical behaviour equivalent to PRIMARY's plain dot product.
    b_pool: [P, d]; r_pool: [P]. Returns w_r [d]."""
    b_pool = np.asarray(b_pool, np.float64)
    r_pool = np.asarray(r_pool, np.float64)
    d = b_pool.shape[-1]
    a = b_pool.T @ b_pool + float(ridge) * np.eye(d)
    return np.linalg.solve(a, b_pool.T @ r_pool)


def compute_density_secondary(r_x0, w: np.ndarray, kappa_x0: np.ndarray,
                              b_pool: np.ndarray, r_pool: np.ndarray, gamma: float,
                              ridge: float = 1e-4, w_r: np.ndarray | None = None):
    """SECONDARY (spec sec.1/6, "~=", the compact-reward-code repair of FB's
    dot-product theorem):
        mu  = sum_j w_j B(u~tilde_j) = w^T B(U~P)
        mu0 = kappa(x0|U_P)^T B(U~P)
        w_r ~= ridge fit of r on B(U~P)                          (ridge_fit_wr)
        Qhat_r ~= ( r(x0) + (mu - mu0)^T w_r ) / (1 - gamma)
    `w_r` may be supplied precomputed (a caller scoring many (s,a,g) rows
    against the SAME pool/reward should fit it once, not once per row --
    `mu`/`mu0` are the only per-row-dependent pieces); otherwise it is fit
    here from `b_pool`/`r_pool`/`ridge`. Returns (Qhat_r, w_r, mu, mu0)."""
    w = np.asarray(w, np.float64)
    kappa_x0 = np.asarray(kappa_x0, np.float64)
    b_pool = np.asarray(b_pool, np.float64)
    r_pool = np.asarray(r_pool, np.float64)
    r_x0 = np.asarray(r_x0, np.float64)
    if w_r is None:
        w_r = ridge_fit_wr(b_pool, r_pool, ridge)
    mu = w @ b_pool
    mu0 = kappa_x0 @ b_pool
    score = np.sum((mu - mu0) * w_r, axis=-1)
    return (r_x0 + score) / (1.0 - gamma), w_r, mu, mu0


def pool_sweep_primary(pool_sizes, r_x0, w_h_full: np.ndarray, kappa_full: np.ndarray,
                       r_pool_full: np.ndarray, gamma: float, double_q: bool = True) -> dict:
    """--pool_sweep (spec sec.6/7 G3): the PRIMARY readout at every P in
    `pool_sizes`, derived EXACTLY (not approximately) from ONE P_max-sized
    forward pass -- no extra tower calls, no re-drawing. A softmax vector
    restricted to a leading subset and renormalized by that subset's own sum
    is IDENTICAL to computing the softmax over just that subset from scratch
    (the defining property of softmax): so each smaller P's `w`/`kappa` are
    obtained by prefixing `w_h_full`/`kappa_full` (BEFORE the double_q min --
    each head is its own valid softmax and gets its own renormalization) and
    renormalizing, then combining/dotting exactly as `density_weights` /
    `compute_density_primary` do.

    w_h_full: [H, P_max] (or [P_max], H=1) per-head UN-combined softmax
      weights over the FULL (largest) pool draw for ONE fixed query (s,a,g).
    kappa_full: [P_max] softmax kernel weights at the SAME pool draw for the
      SAME query's x0.
    r_pool_full: [P_max] reward evaluated at every pool point (prefix-stable:
      r(u) does not depend on pool size, only the SUM does).
    r_x0: scalar reward at x0 (pool-size-independent, added identically at
      every P).

    Returns {"Q_by_pool": {P: Qhat_r}, "drift_vs_maxP": {P: |Q(P)-Q(P_max)|},
    "max_pairwise_drift": float, "P_max": int}."""
    pool_sizes = sorted(int(p) for p in pool_sizes)
    assert pool_sizes, "pool_sweep_primary needs at least one pool size"
    P_max = pool_sizes[-1]
    w_h_full = np.asarray(w_h_full, np.float64)
    if w_h_full.ndim == 1:
        w_h_full = w_h_full[None]
    kappa_full = np.asarray(kappa_full, np.float64)
    r_pool_full = np.asarray(r_pool_full, np.float64)
    assert w_h_full.shape[-1] >= P_max and kappa_full.shape[-1] >= P_max \
        and len(r_pool_full) >= P_max, \
        f"pool draws must cover the largest swept size {P_max}"
    q_by_pool = {}
    for P in pool_sizes:
        wh_p = w_h_full[..., :P]
        wh_p = wh_p / wh_p.sum(axis=-1, keepdims=True)         # per-head renorm (exact)
        w_p = wh_p.min(axis=0) if (double_q and wh_p.shape[0] > 1) else wh_p[0]
        k_p = kappa_full[..., :P]
        k_p = k_p / k_p.sum(axis=-1, keepdims=True)
        q_by_pool[P] = float(compute_density_primary(r_x0, w_p, k_p, r_pool_full[:P], gamma))
    q_max = q_by_pool[P_max]
    drift_vs_max = {P: abs(q_by_pool[P] - q_max) for P in pool_sizes}
    ordered = [q_by_pool[P] for P in pool_sizes]
    max_pairwise = max((abs(a - b) for a, b in zip(ordered, ordered[1:])), default=0.0)
    return {"Q_by_pool": q_by_pool, "drift_vs_maxP": drift_vs_max,
           "max_pairwise_drift": max_pairwise, "P_max": P_max}


# ======================= live-agent integration (jax-dependent; lazy import) =======
# Everything below calls the agent's OWN `density_f`/`density_b`/`_delta_phi`/
# `_density_queries` -- the SAME code path training uses, not a reimplementation
# (task brief). None of this is reachable/testable until spec 18 sec.4 lands in
# gciql_flow.py -- see module docstring "UNVALIDATED".

def build_pool(agent, blocks, pool_obs: np.ndarray, spec):
    """One forward pass: raw block coords U_raw [P,d_c] (spec.coords, the
    generic env_geometry helper -- NOT `_density_queries`, which only returns
    the eps-SCALED twin; the kernel path needs RAW rows, spec sec.4.2's
    called-out double-scaling trap), eps-scaled U_sc via the agent's OWN
    `_density_queries` (live import), and B(U~P) via the agent's OWN
    `density_b` tower. Returns (U_raw [P,d_c] np.float32, B_pool [P,d] np.float64).
    """
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    u_raw = spec.coords(pool_obs).astype(np.float32)                       # [P, d_c]
    u_sc = np.asarray(GCIQLFlowAgent._density_queries(
        jnp.asarray(pool_obs, jnp.float32), blocks))                       # [P, d_c]
    b_pool = np.asarray(agent.network.select("density_b")(jnp.asarray(u_sc)),
                        np.float64)                                        # [P, d]
    return u_raw, b_pool


def density_query(agent, obs, acts, g, u_raw: np.ndarray, b_pool: np.ndarray,
                  blocks, temp: float, double_q: bool):
    """One (s,a,g) batch -> (w [n,P], kappa_x0 [n,P]) against a given pool
    (u_raw/b_pool already sliced to the desired P). `obs` doubles as x0 for
    kappa (the CURRENT state, per spec sec.1 -- not the successor)."""
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    f_emb = np.asarray(agent.network.select("density_f")(
        jnp.concatenate([jnp.asarray(obs, jnp.float32), jnp.asarray(acts, jnp.float32),
                        jnp.asarray(g, jnp.float32)], -1)), np.float64)     # [2,n,d]
    w = density_weights(f_emb, b_pool, temp, double_q)                      # [n,P]
    kappa_x0 = np.asarray(GCIQLFlowAgent._delta_phi(
        jnp.asarray(obs, jnp.float32), u_raw, blocks), np.float64)          # [n,P]
    return w, kappa_x0


def decode_goal_retrieval_density(agent, x0_obs, u_raw, b_pool, r_pool, blocks,
                                  temp, double_q, cand_coords, cand_obs,
                                  retrieval_k=8, rng_seed=0):
    """Retrieval decode (spec-14-comparable, fork of
    eval_partition_zeroshot.decode_goal_retrieval): scores each candidate
    dataset state as a proposed deployment goal g_j by this readout's own
    PRIMARY dot-product term (w - kappa(x0)).r(U_P), min over F-heads if
    double_q -- r(x0) and the constant offset are candidate-independent
    (same reduction the partition script and gciql_flow.py's actor_loss
    q-term use), so ranking by full Qhat_r reduces to ranking by this term.

    ACTION FED: the trained flow actor's own one-shot deterministic proposal
    a_j ~ actor(x0, g_j), temperature=0, a SINGLE shared rng key across all
    candidates -- identical justification to the partition script's
    `decode_goal_retrieval` (see that function's docstring): "what the
    deployed policy would do immediately if it committed to goal g_j."

    x0_obs: [obs_dim] canonical start state. u_raw/b_pool/r_pool: the SAME
    pool/reward used everywhere else for this family/task (u_raw RAW block
    rows, b_pool [P,d], r_pool [P]). cand_coords/cand_obs: the shared
    relabel-dataset retrieval candidate pool (main()'s S_coords/S_obs, same
    rng handling as eval_zeroshot_rewards.py's own `S`).

    Returns (g_hat_obs [obs_dim], info dict).
    """
    import jax
    import jax.numpy as jnp

    n = len(cand_obs)
    x0_b = jnp.tile(jnp.asarray(x0_obs, jnp.float32)[None], (n, 1))
    cand_obs_j = jnp.asarray(cand_obs, jnp.float32)
    rng = jax.random.PRNGKey(rng_seed)
    a_cand = np.clip(np.asarray(agent.sample_actions(
        observations=x0_b, goals=cand_obs_j, seed=rng, temperature=0.0)), -1, 1)
    w, kappa_x0 = density_query(agent, x0_b, jnp.asarray(a_cand), cand_obs_j,
                                u_raw, b_pool, blocks, temp, double_q)       # [n,P] each
    scores = np.sum((w - kappa_x0) * r_pool[None, :], axis=-1)              # [n]
    chosen, info = select_retrieval_candidate(scores, cand_coords, cand_obs, retrieval_k)
    info["x0_source"] = "canonical seed-0 env reset (main()'s per-task/region pool env)"
    info["action_fed"] = ("trained flow actor's one-shot deterministic proposal "
                          "a_j ~ actor(x0, g_j), temperature=0, single shared rng "
                          "key across all candidates")
    return cand_obs[chosen], info


def batched_goal_rollout_density(agent, u_raw, b_pool, r_pool, blocks, gamma, temp,
                                 double_q, spec, reward_fn, envs, g_hat_obs,
                                 ridge=1e-4, H=200, u_raw_full=None, b_pool_full=None,
                                 r_pool_full=None, pool_sizes=None):
    """Plain fixed-goal rollout at a SINGLE decoded g_hat_obs for the whole
    episode (retrieval decode's deployment step -- no per-step re-scoring,
    mirrors eval_partition_zeroshot.batched_goal_rollout exactly). Reports,
    at t=0: PRIMARY Qhat_r (this pool size), SECONDARY Qhat_r (ridge-fit at
    this pool size), and -- if `u_raw_full`/`b_pool_full`/`r_pool_full`/
    `pool_sizes` are given (main()'s --pool_sweep path) -- the PRIMARY
    pool-sweep drift at the SAME t=0 state/action/goal (no extra rollouts).

    Returns (traj [n,H,3] cube xyz meters, obs_traj [n,H,obs_dim],
    lengths [n], succ_env [n], t0 dict with qr_primary/qr_secondary/
    pool_sweep/decode-independent diagnostics)."""
    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent

    n = len(envs)
    obs_dim = envs[0].observation_space.shape[0]
    g_hat_b = jnp.tile(jnp.asarray(g_hat_obs, jnp.float32)[None], (n, 1))
    obs = np.zeros((n, obs_dim), np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=i)
        obs[i] = o
    active = np.ones(n, bool)
    traj = np.zeros((n, H, 3))
    obs_traj = np.zeros((n, H, obs_dim), np.float32)
    lengths = np.zeros(n, int)
    succ_env = np.zeros(n)
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
            q_secondary, w_r, mu, mu0 = compute_density_secondary(
                r_x0, w, kappa_x0, b_pool, r_pool, gamma, ridge=ridge)
            t0["qr_primary"] = np.asarray(q_primary)
            t0["qr_secondary"] = np.asarray(q_secondary)
            t0["obs"] = np.asarray(obs_j)
            t0["acts"] = np.asarray(acts)
            if pool_sizes and u_raw_full is not None:
                f_emb_full = np.asarray(agent.network.select("density_f")(
                    jnp.concatenate([obs_j, jnp.asarray(acts), g_hat_b], -1)), np.float64)
                kappa_full = np.asarray(GCIQLFlowAgent._delta_phi(
                    obs_j, u_raw_full, blocks), np.float64)                  # [n,P_max]
                sweeps = []
                for i in range(n):
                    logits_full = np.einsum("hd,pd->hp", f_emb_full[:, i, :], b_pool_full) / temp
                    w_h_full_i = softmax_np(logits_full, axis=-1)
                    sweeps.append(pool_sweep_primary(
                        pool_sizes, float(r_x0[i]), w_h_full_i, kappa_full[i],
                        r_pool_full, gamma, double_q=double_q))
                t0["pool_sweep"] = sweeps
        for i, e in enumerate(envs):
            if not active[i]:
                traj[i, t] = traj[i, t - 1]
                continue
            o, _r, term, trunc, inf = e.step(acts[i])
            obs[i] = o
            traj[i, t] = e.unwrapped.data.qpos[14:17]
            obs_traj[i, t] = o
            lengths[i] = t + 1
            if inf.get("success"):
                succ_env[i] = 1.0
            if term or trunc:
                active[i] = False
        if not active.any():
            traj[:, t + 1:] = traj[:, t:t + 1]
            break
    return traj, obs_traj, lengths, succ_env, t0


# ======================= CLI ======================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run dir or glob (params_<step>.pkl + flags.json; first match "
                         "used; must be a measure_density=True checkpoint)")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--env", default="", help="override env (default: flags.json env_name)")
    ap.add_argument("--tasks", default="1,2,3,4,5",
                    help="comma-sep task indices 1..5 (dirac/dense families)")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help=f"comma-sep subset of {FAMILIES}")
    ap.add_argument("--regions", type=int, default=4,
                    help="region family: number of hindsight centers")
    ap.add_argument("--region-min-pos", type=int, default=5,
                    help="region family: min positive relabel samples per center")
    ap.add_argument("--retrieval-k", type=int, default=8,
                    help="top-k candidates to retrieve before collapsing to one goal "
                         "(run_spec14_jax_table.sh's gate protocol uses 1)")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=0,
                    help="0 = the env's own max_episode_steps")
    ap.add_argument("--relabel-size", type=int, default=50000,
                    help="dataset sample size for the region family's hindsight draw "
                         "and the retrieval-decode candidate pool")
    ap.add_argument("--pool_size", type=int, default=4096,
                    help="P: query pool size used for the readout/decode/rollout "
                         "(spec sec.6 default)")
    ap.add_argument("--pool_sweep", default="256,1024,4096,16384",
                    help="comma-sep pool sizes: recompute the t=0 PRIMARY Q-estimate "
                         "at each size (prefixes of one shared draw -- see module "
                         "docstring pool_sweep_primary) and report drift. Empty string "
                         "disables the sweep entirely.")
    ap.add_argument("--pool_sweep_full_rollout", action="store_true",
                    help="ALSO re-run retrieval decode + a full episode batch at every "
                         "--pool_sweep size (the literal spec sec.7 G3 success-rate "
                         "check) -- expensive, off by default.")
    ap.add_argument("--ridge", type=float, default=1e-4,
                    help="SECONDARY readout ridge regularization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="override the measure_density consistency refusal (loud "
                         "warning, numbers likely garbage)")
    ap.add_argument("--out", default="",
                    help="output JSON (default results/spec18_density/dzs_<run>.json)")
    return ap


# ======================= main =====================================================

def main() -> int:
    args = build_parser().parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in FAMILIES]
    assert not unknown, f"unknown families {unknown}; choose from {FAMILIES}"
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    assert all(1 <= t <= 5 for t in task_ids), f"task ids must be in 1..5, got {task_ids}"
    sweep_sizes = sorted({int(p) for p in args.pool_sweep.split(",") if p.strip()}
                         | {args.pool_size}) if args.pool_sweep else [args.pool_size]

    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    assert (run_dir / "flags.json").exists(), \
        f"{run_dir} has no flags.json (the arch would silently mis-build)"
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])

    ckpt_is_density = bool(cfg.get("measure_density", False))
    for _w in check_density_consistency(ckpt_is_density, True, force=args.force,
                                        run_dir=str(run_dir)):
        print(f"[density-zs] {_w}", flush=True)

    env_name = args.env or flags["env_name"]
    gamma = float(cfg["discount"])
    double_q = bool(cfg.get("double_q", True))
    temp = float(cfg.get("measure_density_temp", 1.0))

    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    blocks = spec.blocks
    d_c = spec.dim
    eval_tasks = eval_tasks_for(env_name)

    t00 = time.time()
    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks[1])
    _obs_dim = int(np.asarray(ds1["observations"]).shape[-1])
    _act_dim = int(np.asarray(ds1["actions"]).shape[-1])
    H = args.horizon if args.horizon > 0 else int(_e0.spec.max_episode_steps)

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, _obs_dim, _act_dim)

    print(f"[density-zs] restored {run_dir.name} step={args.checkpoint_step} "
          f"env={env_name} d_c={d_c} gamma={gamma} "
          f"hit_spec={cfg.get('measure_hit_spec')!r} double_q={double_q} temp={temp} "
          f"pool_size={args.pool_size} pool_sweep={sweep_sizes} ridge={args.ridge} "
          f"H={H}", flush=True)

    rng_np = np.random.default_rng(args.seed)
    obs_all = np.asarray(ds1["observations"], np.float32)
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(ds1["terminals"]))
    idx = rng_np.choice(len(coords_all), size=min(args.relabel_size, len(coords_all)),
                        replace=False)
    relabel_coords = coords_all[idx]
    S_obs, S_coords = obs_all[idx], relabel_coords

    # pool draw: ONE draw of size max(pool_size, every swept size) -- every P
    # this run needs is a PREFIX of it (pool_sweep_primary's exact-renorm trick;
    # module docstring).
    pool_cap = max([args.pool_size] + sweep_sizes)
    n_ds = len(obs_all)
    pool_idx = rng_np.choice(n_ds, size=min(pool_cap, n_ds), replace=(pool_cap > n_ds))
    pool_obs_full = obs_all[pool_idx]
    u_raw_full, b_pool_full = build_pool(agent, blocks, pool_obs_full, spec)
    u_raw, b_pool = u_raw_full[:args.pool_size], b_pool_full[:args.pool_size]
    print(f"[density-zs] pool drawn: cap={len(pool_obs_full)} "
          f"(={time.time()-t00:.0f}s)", flush=True)

    goal_obs, goal_coord = {}, {}
    for i in task_ids:
        env = ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
        _o, info = env.reset(seed=0)
        goal_obs[i] = np.asarray(info["goal"], np.float32)
        goal_coord[i] = spec.coords(goal_obs[i][None])[0]
        env.close()

    pools = {}

    def pool_envs(i):
        if i not in pools:
            pools[i] = [ogbench.make_env_and_datasets(eval_tasks[i], env_only=True)
                       for _ in range(args.episodes)]
        return pools[i]

    def close_all():
        for i in list(pools):
            for e in pools.pop(i):
                if hasattr(e, "close"):
                    e.close()

    def run_cell(z_r_full, rfn, envs, want_sweep, pool_size=None):
        """z_r_full: reward evaluated at pool_obs_full's coords [pool_cap]
        (prefix-stable). Decodes g_hat by retrieval at `pool_size` (default
        args.pool_size), then a fixed-goal rollout; want_sweep=True threads
        the full pool draw through for the (cheap) t=0 pool_sweep -- only
        meaningful when pool_size == args.pool_size (the "production" cell);
        --pool_sweep_full_rollout's per-P reruns pass want_sweep=False."""
        P = pool_size if pool_size is not None else args.pool_size
        u_raw_p, b_pool_p = u_raw_full[:P], b_pool_full[:P]
        r_pool_full = z_r_full
        r_pool = r_pool_full[:P]
        x0_obs, _ = envs[0].reset(seed=0)
        g_hat, decode_info = decode_goal_retrieval_density(
            agent, x0_obs, u_raw_p, b_pool_p, r_pool, blocks, temp, double_q,
            S_coords, S_obs, retrieval_k=args.retrieval_k, rng_seed=args.seed)
        traj, ot, ln, senv, t0 = batched_goal_rollout_density(
            agent, u_raw_p, b_pool_p, r_pool, blocks, gamma, temp, double_q, spec, rfn,
            envs, g_hat, ridge=args.ridge, H=H,
            u_raw_full=u_raw_full if want_sweep else None,
            b_pool_full=b_pool_full if want_sweep else None,
            r_pool_full=r_pool_full if want_sweep else None,
            pool_sizes=sweep_sizes if want_sweep else None)
        return traj, ot, ln, senv, t0, decode_info

    def full_rollout_sweep_for(z_r_full, rfn, envs, success_fn=None):
        """--pool_sweep_full_rollout (spec sec.7 G3, literal): re-runs
        retrieval decode + a full episode batch at EVERY swept pool size (not
        just the t=0 Q-estimate `run_cell`'s want_sweep already gives for
        free) -- the actual success-RATE stability check G3 asks for. Off by
        default (expensive: `len(sweep_sizes)` extra full rollouts per cell).
        success_fn(obs_traj, lengths) -> per-episode success array; defaults
        to the raw env success flag (dirac/dense); region passes its own
        conjunctive-membership reduction (region_success_from_obs)."""
        out = {}
        for P in sweep_sizes:
            if P == args.pool_size:
                continue     # the production cell already covers this size
            _tj, ot_p, ln_p, senv_p, t0_p, _di = run_cell(
                z_r_full, rfn, envs, want_sweep=False, pool_size=P)
            succ_p = senv_p if success_fn is None else success_fn(ot_p, ln_p)
            out[P] = {"success": float(np.mean(succ_p)),
                     "qr_primary_t0_mean": float(np.mean(t0_p["qr_primary"]))}
        return out

    results = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "d_c": d_c, "gamma": gamma, "double_q": double_q,
        "temp": temp, "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
        "pool_size": args.pool_size, "pool_sweep_sizes": sweep_sizes,
        "pool_cap": int(len(pool_obs_full)), "ridge": args.ridge, "seed": args.seed,
        "readout": {
            "primary": "( r(x0) + (w - kappa(x0|U_P))^T r(U_P) ) / (1-gamma) -- "
                       "part's (17b) eq. with prototypes->pool; exactly linear in r",
            "secondary": "( r(x0) + (mu-mu0)^T w_r ) / (1-gamma), mu=w^T B(U~P), "
                        "mu0=kappa(x0)^T B(U~P), w_r=ridge fit of r on B(U~P) -- "
                        "the compact-reward-code repair; linear in r for a fixed "
                        "pool/ridge but not a bare dot product (see script docstring)",
            "w": "softmax_j(F(s,a,g).B(u_j)/temp) per F-head, elementwise min over "
                "the 2 heads if double_q (spec sec.3)",
            "kappa": "GCIQLFlowAgent._delta_phi(x0, U_raw, blocks) -- SAME code path "
                    "training uses, evaluated at the CURRENT state (not successor)",
            "decode": "retrieval only (spec-14-comparable): ONE goal-proxy observation "
                     "g_hat decoded per family/task via this readout's own PRIMARY "
                     "score, then a plain fixed-goal rollout, no per-step re-scoring "
                     "(qr_greedy not implemented -- see module docstring 'SCOPE "
                     "NARROWING')",
            "pool_sweep": "prefix+renormalize of ONE P_max forward pass (exact, no "
                         "re-draws) -- t=0 PRIMARY Q-estimate drift only, NOT full "
                         "rollout success-rate stability unless "
                         "--pool_sweep_full_rollout was passed",
            "spec": "PAPER/Multistep-FB/specs/18-density-measure.md sec.1/6",
        },
        "tasks": task_ids, "families": families, "episodes": args.episodes,
        "horizon": H, "relabel_size": int(len(relabel_coords)),
        "goal_coords": {f"t{i}": goal_coord[i].tolist() for i in task_ids},
    }, "families": {}}
    table = []

    def build_rec(senv, t0, decode_info, extra=None):
        rec = {"decode": decode_info,
              "success": float(senv.mean()),
              "qr_primary_t0_mean": float(np.mean(t0["qr_primary"])),
              "qr_secondary_t0_mean": float(np.mean(t0["qr_secondary"]))}
        if "pool_sweep" in t0:
            rec["pool_sweep"] = t0["pool_sweep"]
        if extra:
            rec.update(extra)
        return rec

    # ================= 1. dirac =================
    if "dirac" in families:
        fam = {}
        for i in task_ids:
            z_r_full = reward_conj_dirac(coords_all[pool_idx], goal_coord[i], spec)

            def rfn(coords, c=goal_coord[i]):
                return reward_conj_dirac(coords, c, spec)

            _tj, _ot, _ln, senv, t0, decode_info = run_cell(
                z_r_full, rfn, pool_envs(i), want_sweep=True)
            extra = {}
            if args.pool_sweep_full_rollout:
                extra["pool_sweep_full_rollout"] = full_rollout_sweep_for(
                    z_r_full, rfn, pool_envs(i))
            rec = build_rec(senv, t0, decode_info, extra=extra)
            fam[f"t{i}"] = rec
            table.append(("dirac", f"t{i}", "success", f"{rec['success']}"))
            print(f"[density-zs][dirac] t{i}: succ={rec['success']} "
                  f"qr0_primary={rec['qr_primary_t0_mean']:.3f} "
                  f"qr0_secondary={rec['qr_secondary_t0_mean']:.3f} "
                  f"({time.time()-t00:.0f}s)", flush=True)
        vals = [v["success"] for v in fam.values() if v["success"] is not None]
        fam["mean"] = float(np.mean(vals)) if vals else None
        results["families"]["dirac"] = fam
        table.append(("dirac", "MEAN", "success", f"{fam['mean']}"))

    # ================= 2. dense =================
    if "dense" in families:
        fam = {}
        for i in task_ids:
            z_r_full = reward_conj_dense(coords_all[pool_idx], goal_coord[i], spec)

            def rfn(coords, c=goal_coord[i]):
                return reward_conj_dense(coords, c, spec)

            _tj, _ot, _ln, senv, t0, decode_info = run_cell(
                z_r_full, rfn, pool_envs(i), want_sweep=True)
            extra = {}
            if args.pool_sweep_full_rollout:
                extra["pool_sweep_full_rollout"] = full_rollout_sweep_for(
                    z_r_full, rfn, pool_envs(i))
            rec = build_rec(senv, t0, decode_info, extra=extra)
            fam[f"t{i}"] = rec
            table.append(("dense", f"t{i}", "success", f"{rec['success']}"))
            print(f"[density-zs][dense] t{i}: succ={rec['success']} "
                  f"qr0_primary={rec['qr_primary_t0_mean']:.3f} "
                  f"qr0_secondary={rec['qr_secondary_t0_mean']:.3f} "
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
        fam = {"n_centers": int(len(centers)), "centers_rejected": int(rejected),
              "center_min_pos": int(args.region_min_pos)}
        print(f"[density-zs][region] {len(centers)}/{args.regions} hindsight "
              f"centers ({rejected} rejected at min_pos={args.region_min_pos})",
              flush=True)
        for k_i in range(len(centers)):
            c = centers[k_i]
            z_r_full = reward_conj_dirac(coords_all[pool_idx], c, spec)

            def rfn(coords, c=c):
                return reward_conj_dirac(coords, c, spec)

            _tj, ot, ln, _se, t0, decode_info = run_cell(
                z_r_full, rfn, pool_envs(v0), want_sweep=True)
            fin, anys = region_success_from_obs(ot, ln, c, spec)
            extra = {}
            if args.pool_sweep_full_rollout:
                extra["pool_sweep_full_rollout"] = full_rollout_sweep_for(
                    z_r_full, rfn, pool_envs(v0),
                    success_fn=lambda ot_p, ln_p, c=c: region_success_from_obs(
                        ot_p, ln_p, c, spec)[0])
            extra.update({"center_coord": np.asarray(c).tolist(),
                         "center_n_pos": int(npos_c[k_i]),
                         "success_final": float(fin.mean()),
                         "success_any": float(anys.mean())})
            rec = build_rec(np.asarray(fin), t0, decode_info, extra=extra)
            fam[f"c{k_i}"] = rec
            table.append(("region", f"c{k_i}", "success_final", f"{rec['success_final']}"))
            print(f"[density-zs][region] c{k_i}: fin={rec['success_final']} "
                  f"any={rec['success_any']} ({time.time()-t00:.0f}s)", flush=True)
        cells = [v for k, v in fam.items() if k.startswith("c") and isinstance(v, dict)]
        vals = [v["success_final"] for v in cells]
        anyv = [v["success_any"] for v in cells]
        fam["mean_final"] = float(np.mean(vals)) if vals else None
        fam["mean_any"] = float(np.mean(anyv)) if anyv else None
        results["families"]["region"] = fam
        table.append(("region", "MEAN", "success_final", f"{fam['mean_final']}"))

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec18_density" / f"dzs_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=lambda o: float(o)))
    print(f"\n[density-zs] DONE ({time.time()-t00:.0f}s) -> {outp}\n", flush=True)
    print_table(table)
    close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
