#!/usr/bin/env python
"""scripts/rescuing_fb/eval_z_action_ranking.py -- spec 28 P0a: the
action-ranking decomposition (PAPER/Multistep-FB/specs/28-z-interface-
canonicalization.md sec.0 + P0a). EVAL ONLY: nothing here trains, and no
training code path is touched.

WHY. Spec 24 changed TWO things at once -- the policy's CONDITIONING
(goal obs -> latent z) and the actor's REWARD READOUT (the exact,
query-set-normalized kernel kg -> B~(u)^T z / sqrt(d)). The .736 -> .017
cube-double collapse is therefore not yet attributable to either. P0a
decomposes them on EXISTING checkpoints by computing BOTH readouts under
the SAME weights w_j(s, a, cond) and comparing how they RANK actions:

    q_g(s,a) = sum_j w_j(s,a,cond) * kg_j            (exact kernel readout)
    q_z(s,a) = sum_j w_j(s,a,cond) * B~(u_j)^T z/sqrt(d)   (latent reward)

AUDIT CORRECTION HONOURED (spec 28 sec.0, first bullet): the g-path
"exact kernel" is `GCIQLFlowAgent._delta_phi(goals, u_raw, blocks)` --
QUERY-SET SOFTMAX-NORMALIZED, rows summing to 1 over the M pool queries.
That exact object is what this script uses for kg (never a raw kappa), so
q_g here is bit-for-bit the quantity `_density_actor_q` maximizes and the
two readouts are directly comparable.

FAITHFULNESS TO THE TRAINED OBJECTS (all live agent code, never
reimplemented):
  * w  : softmax_j( F_h(s,a,cond).B(u_j)/temp ) at the ONLINE density_f /
         density_b towers -- the same objects `_density_actor_q` /
         `_density_actor_q_z` build. When the checkpoint carries spec 25
         C1's structural tilt (measure_density_z_tilt != 0) the SAME tilt
         term beta*z.B(u)/sqrt(d) is added to the logits, because the
         trained actor reads the tilted family (gciql_flow
         `_density_actor_q_z`); omitting it would score an occupancy the
         measure never predicts.
  * kg : `_delta_phi(goal_obs, u_raw, blocks)` (RAW block rows in, the
         script never double-scales -- spec 18 sec.4.2 trap).
  * rz : `target_density_b(_density_queries(u))` . z / sqrt(d), the TARGET
         tower exactly as `_density_actor_q_z` uses for B~.
  * z  : `agent._density_z_encode(goal_obs)` (reads the checkpoint's own
         config for target_density_b vs density_b_zenc, spec 25 C3).
  * head reduction: the TRAINING convention q = min_h q^h (gciql_flow
         actor_loss `jnp.minimum(q1,q2)`) is PRIMARY; the spec-18 eval
         harness convention (elementwise min over the two PROBABILITY
         vectors, then one dot product) is reported alongside as `_wmin`
         so a convention difference can never masquerade as a finding.

METRICS per (s,g) pair, over the K-action set:
  spearman(q_g, q_z); span (max-min) and IQR of each; hit-region AUC of
  each against dataset "good" (on-trajectory) actions with random-uniform
  and off-trajectory-dataset negatives (CHANCE = 0.500 -- always printed
  next to the number); Delta_actor = max_k q(s,a_k) - q(s,pi(s,cond)) for
  both readouts, raw and as a fraction of the span; and ACTOR SENSITIVITY
  E||pi(s,cond) - pi(s,cond')|| over resampled cond' (sphere z' and
  other-goal z'), with its two mandatory reference scales -- the actor's
  own NOISE sensitivity E||pi(s,cond,xi) - pi(s,cond,xi')|| at fixed cond
  (a floor: below this, cond does nothing the sampling noise does not) and
  the uniform-action scale E||u-u'|| (a ceiling). The existing
  `density_w_ztv` instrument measures the MEASURE's index sensitivity and
  never the ACTOR's; this closes that gap.

SPAN >> 0 IS MEASURED AGAINST AN MC FLOOR, NOT ASSERTED. Both readouts are
Monte-Carlo integrals over the query pool, so a "span" can be pure sampling
noise. The pool is split into halves; each half's w and kg are the exact
softmax over that half (restrict + renormalize == softmax over the subset),
giving two independent estimates of the SAME half-pool quantity per action.
  snr          = span(q_half) / median_a |(q_h1-mean q_h1)-(q_h2-mean q_h2)|
  selfcons     = spearman(q_h1, q_h2)   (1.0 = the ranking is real,
                 ~0.0 = the ranking IS the MC noise)

The centering in `snr` is load-bearing and was corrected AFTER a smoke run on
the g anchor exposed the naive version: the half a pool point lands in shifts
every action's q by the SAME common-mode offset, which the ranking is blind
to, so the uncentered ratio reported snr 0.27 on a checkpoint whose
selfcons was 0.95 -- an instrument disagreeing with itself. Both are recorded
(`snr` / `snr_uncentered`); `selfcons`, a rank statistic and therefore
common-mode-free, is the primary "the span is not noise" evidence.

CONTROL GATE (spec 28 P0: "every new probe passes a control gate on the
g-indexed anchor before any z cell is read" -- the three-times-burned
rule). PRE-REGISTERED, in `gate_p0a`, BEFORE any z checkpoint is run:
  CG1 instrument sensitivity : median AUC_g(traj vs uniform) >= 0.65
                               (chance 0.500)
  CG2 ranking is not MC noise: median selfcons_g >= 0.90 AND median
                               snr_g >= 3.0
  CG3 Delta_actor calibrated : median Delta_actor_g / span_g <= 0.50
                               (on a checkpoint whose extraction WORKS the
                               actor's own action must sit in the upper
                               half of the action set, else the
                               Delta_actor branch of the readout table is
                               an uncalibrated instrument)
CG1+CG2 are BLOCKING (a failure means the probe cannot see anything and no
z number may be read); CG3 is REPORTED and gates only the "actor extractor
indicted" branch.

Run (CPU-only jax; long runs in tmux):
  taskset -c 24-39 env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OMP_NUM_THREADS=4 \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_z_action_ranking.py \\
    --run-dir ~/exp_spec18/factored-fb/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s0 \\
    --checkpoint-step 500000 --pool_size 16384 --pairs 64 --actions 64 \\
    --out results/spec28_p0a/p0a_cs_g_anchor.json
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

# Reused, never reimplemented: the audited spec-18 pool builder / consistency
# refusal, the spec-24 target-tower + z-index adapter, the single geometry
# definition.
from eval_density_zeroshot import (  # noqa: E402
    build_pool,
    check_density_consistency,
)
from eval_density_z_rewards import _ZIndexView, target_b_pool  # noqa: E402
from eval_zeroshot_rewards import eval_tasks_for  # noqa: E402
from env_geometry import HitSpec, episode_bounds, geometry_for  # noqa: E402

ACTION_SOURCES = ("traj", "dsrand", "actor", "unif")


# ======================= pure logic (jax-free, unit-tested) ========================

def spearman(x, y) -> float:
    """Spearman rank correlation, average ranks for ties. Returns nan when
    either input is constant (undefined, never silently 0.0)."""
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    assert x.shape == y.shape and x.size > 1, f"spearman shape {x.shape}/{y.shape}"
    rx, ry = _avg_rank(x), _avg_rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _avg_rank(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged (the Spearman convention)."""
    v = np.asarray(v, np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), np.float64)
    ranks[order] = np.arange(1, len(v) + 1, dtype=np.float64)
    # average tied ranks
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return ranks


def auc_mw(scores, labels) -> float:
    """Mann-Whitney AUC = P(score(pos) > score(neg)) + 0.5 P(tie). CHANCE =
    0.5 by construction. Returns nan if either class is empty."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _avg_rank(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def span_stats(q) -> dict:
    """(max-min), IQR, mean, std of an action-value vector."""
    q = np.asarray(q, np.float64).ravel()
    q1, q3 = np.percentile(q, [25.0, 75.0])
    return {"span": float(q.max() - q.min()), "iqr": float(q3 - q1),
            "mean": float(q.mean()), "std": float(q.std())}


def softmax_np(x, axis=-1) -> np.ndarray:
    x = np.asarray(x, np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def readouts_from_logits(logits_h: np.ndarray, kg: np.ndarray, rz: np.ndarray,
                         sl: slice | None = None) -> dict:
    """Both readouts from ONE set of per-head logits, under the SAME weights.

    logits_h [H, K, P] = F_h(s,a,cond).B(u)/temp (+ tilt if the checkpoint has
    one); kg [P] the query-set-normalized kernel row of THIS pair's goal;
    rz [P] = B~(u).z/sqrt(d). `sl` restricts to a pool subset -- restricting a
    softmax to a subset and renormalizing by that subset's own sum is exactly
    the softmax over the subset (used for the MC half-pool noise floor); kg is
    itself a softmax row over the pool and is renormalized the same way.

    Returns q_g/q_z under BOTH head conventions:
      *_minhead : per-head dot product, then min over heads  (TRAINING, primary)
      *_wmin    : elementwise min of the two probability vectors, one dot
                  product (the spec-18 eval harness convention)
    """
    logits_h = np.asarray(logits_h, np.float64)
    assert logits_h.ndim == 3, f"logits_h must be [H,K,P], got {logits_h.shape}"
    kg = np.asarray(kg, np.float64).ravel()
    rz = np.asarray(rz, np.float64).ravel()
    if sl is not None:
        logits_h, kg, rz = logits_h[..., sl], kg[sl], rz[sl]
    assert logits_h.shape[-1] == kg.size == rz.size
    w_h = softmax_np(logits_h, axis=-1)                       # [H,K,P]
    kg = kg / kg.sum()
    q_g_h = w_h @ kg                                          # [H,K]
    q_z_h = w_h @ rz
    w_min = w_h.min(axis=0) if w_h.shape[0] > 1 else w_h[0]   # no renorm (harness)
    return {"q_g_minhead": q_g_h.min(axis=0) if q_g_h.shape[0] > 1 else q_g_h[0],
            "q_z_minhead": q_z_h.min(axis=0) if q_z_h.shape[0] > 1 else q_z_h[0],
            "q_g_wmin": w_min @ kg, "q_z_wmin": w_min @ rz}


def mc_floor(q_h1, q_h2) -> dict:
    """MC noise floor from two independent half-pool estimates of the SAME
    quantity.

    The noise must be measured on the CENTERED estimates. Which pool points
    landed in which half shifts EVERY action's q by the same common-mode
    offset (a half that happens to hold more mass near the goal raises the
    whole vector), and that offset says nothing about whether the readout
    separates actions -- the only thing P0a asks of it. `snr` therefore uses
    the per-action discrepancy AFTER removing each half's mean;
    `snr_uncentered` (the naive version, common mode included) is kept
    alongside so the correction is visible and auditable rather than silent.

      snr        = mean half-pool span / median_a |(q1-mean q1)-(q2-mean q2)|
      selfcons   = spearman(q_h1, q_h2): 1 = the ranking is real, ~0 = the
                   "ranking" IS the sampling noise. Common-mode-free by
                   construction (a rank statistic), which is why it is the
                   primary "span is not noise" instrument.
    """
    q_h1 = np.asarray(q_h1, np.float64).ravel()
    q_h2 = np.asarray(q_h2, np.float64).ravel()
    raw = float(np.median(np.abs(q_h1 - q_h2)))
    cen = float(np.median(np.abs((q_h1 - q_h1.mean()) - (q_h2 - q_h2.mean()))))
    span = 0.5 * ((q_h1.max() - q_h1.min()) + (q_h2.max() - q_h2.min()))
    return {"snr": float(span / cen) if cen > 0 else float("inf"),
            "snr_uncentered": float(span / raw) if raw > 0 else float("inf"),
            "selfcons": spearman(q_h1, q_h2), "noise": cen,
            "noise_uncentered": raw, "span_half": span}


# -- the PRE-REGISTERED control gate (written before any checkpoint was run) --

GATE_THRESHOLDS = {"auc_g_vs_unif": 0.65, "selfcons_g": 0.90, "snr_g": 3.0,
                   "delta_over_span_g": 0.50}


def gate_p0a(summary: dict, thresholds: dict | None = None) -> tuple:
    """Spec 28 P0's control gate on a G-INDEXED anchor. `summary` is this
    script's own results['summary'] block (medians over pairs). Returns
    (blocking_ok, rows). CG1/CG2 are BLOCKING -- if they fail the probe
    cannot resolve anything and NO z cell may be read. CG3 is reported and
    gates only the 'actor extractor indicted' branch of the readout table."""
    t = dict(GATE_THRESHOLDS if thresholds is None else thresholds)
    rows = []

    def _add(name, val, thr, cmp_ge, blocking):
        v = float("nan") if val is None else float(val)
        ok = bool(v >= thr) if cmp_ge else bool(v <= thr)
        if np.isnan(v):
            ok = False
        rows.append({"check": name, "value": v, "threshold": thr,
                     "direction": "ge" if cmp_ge else "le", "ok": ok,
                     "blocking": blocking})

    _add("CG1_auc_g_traj_vs_unif", summary.get("auc_g_vs_unif_median"),
         t["auc_g_vs_unif"], True, True)
    _add("CG2a_selfcons_g", summary.get("selfcons_g_median"),
         t["selfcons_g"], True, True)
    _add("CG2b_snr_g", summary.get("snr_g_median"), t["snr_g"], True, True)
    _add("CG3_delta_over_span_g", summary.get("delta_over_span_g_median"),
         t["delta_over_span_g"], False, False)
    blocking_ok = all(r["ok"] for r in rows if r["blocking"])
    return blocking_ok, rows


def sample_pairs(bounds: np.ndarray, n_traj: int, n_rand: int, n_ds: int,
                 rng: np.random.Generator, min_gap: int = 10, max_gap: int = 60,
                 min_len: int = 40) -> list:
    """The (s, g) pairs. `n_traj` of them are ON-TRAJECTORY: s = obs[t] and
    g = obs[t + gap] from the SAME episode, so the dataset demonstrably
    reaches g from s and the dataset actions at t..t+k are "good" by
    construction. `n_rand` are RANDOM-STATE controls: g from one episode, s a
    uniformly drawn dataset row (usually a different episode) -- no
    on-trajectory label, so they contribute to spearman/span/Delta/sensitivity
    but never to the AUC. Pure index arithmetic; returns dicts with
    s_idx / g_idx / traj_lo / traj_hi / on_traj."""
    bounds = np.asarray(bounds, np.int64)
    usable = [tuple(b) for b in bounds if (b[1] - b[0]) >= min_len]
    assert usable, f"no episode of length >= {min_len}"
    out = []
    for k in range(n_traj + n_rand):
        lo, hi = usable[int(rng.integers(len(usable)))]
        L = hi - lo
        t = int(lo + rng.integers(0, max(1, L - min_gap - 1)))
        gap = int(rng.integers(min_gap, max_gap + 1))
        g_idx = min(t + gap, hi - 1)
        on_traj = k < n_traj
        s_idx = t if on_traj else int(rng.integers(n_ds))
        out.append({"s_idx": int(s_idx), "g_idx": int(g_idx),
                    "traj_lo": int(t), "traj_hi": int(min(g_idx, hi - 1)),
                    "on_traj": bool(on_traj)})
    return out


def z_geometry(z: np.ndarray, rng: np.random.Generator, n_pairs: int = 20000) -> dict:
    """Is the INDEX itself spread out, or collapsed?

    Without this, an actor that does not move when the goal changes is
    ambiguous between two very different diagnoses: the actor ignores z, or
    z(g) is the same vector for every g. Both z(g) and the training-time task
    prior live on the sqrt(d) sphere, so a random-sphere sample of the same
    size and dimension is the exact reference: mean pairwise cosine ~ 0 and
    mean pairwise distance ~ sqrt(2d). Reported side by side, never alone."""
    z = np.asarray(z, np.float64)
    n, d = z.shape
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    cos = np.sum(zn[i] * zn[j], axis=1)
    dist = np.linalg.norm(z[i] - z[j], axis=1)
    ref = rng.normal(size=(min(n, 2048), d))
    ref = ref * np.sqrt(d) / np.linalg.norm(ref, axis=1, keepdims=True)
    ri = rng.integers(0, len(ref), 4000)
    rj = rng.integers(0, len(ref), 4000)
    rcos = np.sum((ref[ri] / np.sqrt(d)) * (ref[rj] / np.sqrt(d)), axis=1)
    return {"n_goals": int(n), "d": int(d),
            "z_norm_median": float(np.median(np.linalg.norm(z, axis=1))),
            "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
            "dist_median": float(np.median(dist)),
            "sphere_ref_cos_mean": float(rcos.mean()),
            "sphere_ref_dist_expected": float(np.sqrt(2.0 * d)),
            "coord_std_mean": float(z.std(axis=0).mean())}


def action_labels(sources: list) -> np.ndarray:
    """AUC labelling: dataset ON-TRAJECTORY actions are the positives; the
    negatives are uniform draws and off-trajectory dataset actions. Flow-actor
    samples are UNLABELLED (they are what Delta_actor measures, not what the
    AUC scores) -- returned as -1 and dropped by the AUC caller."""
    m = {"traj": 1, "unif": 0, "dsrand": 0, "actor": -1}
    return np.asarray([m[s] for s in sources], np.int64)


# ======================= live agent integration (jax; lazy) ========================

def _cond_for(agent, cfg, goal_obs: np.ndarray):
    """The conditioning vector the trained F/actor consumes for these goals:
    z(g) = agent._density_z_encode(g) for a measure_density_z checkpoint,
    the RAW goal observation for the g-indexed anchor."""
    import jax.numpy as jnp

    if bool(cfg.get("measure_density_z", False)):
        return np.asarray(agent._density_z_encode(
            jnp.asarray(goal_obs, jnp.float32)), np.float32)
    return np.asarray(goal_obs, np.float32)


def _z_for(agent, goal_obs: np.ndarray) -> np.ndarray:
    """z(g) ALWAYS (both checkpoint kinds) -- the latent whose linear reward
    r_z(u) = B~(u).z/sqrt(d) the q_z readout integrates. On the g-indexed
    anchor this is the readout-only comparison (spec 28 cell iii preview):
    the weights stay g-conditioned, only the reward vector changes."""
    import jax.numpy as jnp

    return np.asarray(agent._density_z_encode(jnp.asarray(goal_obs, jnp.float32)),
                      np.float32)


def _actor(agent, cfg):
    """A single sampling entry point: `sample_actions_with_z` (through the
    audited `_ZIndexView`) for a z checkpoint -- plain `sample_actions` would
    re-encode a precomputed z as if it were a goal observation -- and plain
    `sample_actions` for the g-indexed anchor."""
    return _ZIndexView(agent) if bool(cfg.get("measure_density_z", False)) else agent


def logits_for(agent, obs_b, act_b, cond_b, b_pool, temp, tilt_beta, z_row):
    """[H, K, P] density logits F_h(s,a,cond).B(u)/temp at the ONLINE towers,
    plus spec 25 C1's structural tilt beta*z.B(u)/sqrt(d) when the checkpoint
    carries one (the trained actor reads the TILTED family; omitting it would
    score an occupancy the measure never predicts)."""
    import jax.numpy as jnp

    f_emb = np.asarray(agent.network.select("density_f")(
        jnp.concatenate([jnp.asarray(obs_b, jnp.float32),
                         jnp.asarray(act_b, jnp.float32),
                         jnp.asarray(cond_b, jnp.float32)], -1)), np.float64)  # [H,K,d]
    logits = np.einsum("hkd,pd->hkp", f_emb, b_pool) / float(temp)
    if tilt_beta:
        d = b_pool.shape[-1]
        logits = logits + (float(tilt_beta) *
                           (b_pool @ np.asarray(z_row, np.float64)) / np.sqrt(d))[None, None]
    return logits


def hit_spec_for(env_name: str, cfg_hit_spec: str):
    """The measure coordinate. `env_geometry.geometry_for` is the registry
    every eval script shares, but it only knows the three OGBench manipspace
    domains -- ExoRL point-mass-maze (spec 23) was never registered (the v2
    E2 harness dropped v1's exorl branch entirely, which is why no v2 pmm
    number exists). This probe needs nothing from the registry except the
    HitSpec, and spec 14 sec.3.4's rule is that the geometry TRAVELS WITH THE
    WEIGHTS: for an unregistered domain, parse the checkpoint's own recorded
    `measure_hit_spec` (an unrecorded one is a hard error, never a default)."""
    try:
        return geometry_for(env_name, str(cfg_hit_spec or "")).hit_spec
    except KeyError:
        assert str(cfg_hit_spec or "").strip(), (
            f"env {env_name!r} is not in the env_geometry registry AND the "
            "checkpoint records no measure_hit_spec -- refusing to guess a "
            "measure coordinate")
        return HitSpec.parse(str(cfg_hit_spec))


def load_dataset(env_name: str):
    """The offline dataset this probe draws its states/actions/pool from.
    OGBench domains go through `ogbench.make_env_and_datasets` on the task-1
    singletask id (the convention every audited eval here uses); ExoRL
    domains through `exorl_env`'s OWN registry loader. No env is rolled out
    anywhere in P0a -- the returned env is discarded."""
    if env_name.startswith("exorl:"):
        import exorl_env

        _env, train, _val = exorl_env.make_exorl_env_and_datasets(env_name)
        return train
    import ogbench

    _e0, ds1, _ = ogbench.make_env_and_datasets(eval_tasks_for(env_name)[1])
    return ds1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--pool_size", type=int, default=16384)
    ap.add_argument("--pairs", type=int, default=64, help="total (s,g) pairs")
    ap.add_argument("--rand-state-pairs", type=int, default=16,
                    help="of --pairs, how many use a RANDOM state (no AUC label)")
    ap.add_argument("--actions", type=int, default=64,
                    help="K, split evenly over traj/dsrand/actor/unif")
    ap.add_argument("--sens-draws", type=int, default=8,
                    help="resampled cond' / noise draws for the sensitivity metric")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="this is a G-INDEXED ANCHOR run: evaluate the "
                         "pre-registered control gate (gate_p0a) and record the "
                         "PASS/FAIL verdict in the output JSON")
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    t00 = time.time()

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])
    for _w in check_density_consistency(bool(cfg.get("measure_density", False)), True,
                                        force=args.force, run_dir=str(run_dir)):
        print(f"[p0a] {_w}", flush=True)

    env_name = flags["env_name"]
    is_z = bool(cfg.get("measure_density_z", False))
    temp = float(cfg.get("measure_density_temp", 1.0))
    d = int(cfg["measure_density_dim"])
    tilt_beta = float(cfg.get("measure_density_z_tilt") or 0.0)
    spec = hit_spec_for(env_name, cfg.get("measure_hit_spec"))
    blocks = spec.blocks

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]

    ds1 = load_dataset(env_name)
    obs_all = np.asarray(ds1["observations"], np.float32)
    act_all = np.asarray(ds1["actions"], np.float32)
    obs_dim, act_dim = obs_all.shape[-1], act_all.shape[-1]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)

    import jax
    import jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent, _sphere

    print(f"[p0a] restored {run_dir.name} step={args.checkpoint_step} env={env_name} "
          f"z={is_z} d={d} temp={temp} tilt={tilt_beta} "
          f"hit_spec={cfg.get('measure_hit_spec')!r} P={args.pool_size}", flush=True)

    # ---- pool (audited builder; rng order matches the spec-18 harness) ----
    rng = np.random.default_rng(args.seed)
    bounds = episode_bounds(np.asarray(ds1["terminals"]))
    n_ds = len(obs_all)
    pool_idx = rng.choice(n_ds, size=min(args.pool_size, n_ds),
                          replace=(args.pool_size > n_ds))
    pool_obs = obs_all[pool_idx]
    u_raw, b_pool = build_pool(agent, blocks, pool_obs, spec)          # online B
    b_pool_t = target_b_pool(agent, blocks, pool_obs)                  # target B~
    P = len(pool_obs)
    half = P // 2
    print(f"[p0a] pool P={P} ({time.time()-t00:.0f}s)", flush=True)

    # ---- (s,g) pairs ----
    n_rand = min(args.rand_state_pairs, args.pairs)
    pairs = sample_pairs(bounds, args.pairs - n_rand, n_rand, n_ds, rng)
    g_obs = obs_all[[p["g_idx"] for p in pairs]]                       # [N, obs]
    kg_all = np.asarray(GCIQLFlowAgent._delta_phi(
        jnp.asarray(g_obs, jnp.float32), u_raw, blocks), np.float64)   # [N, P]
    z_all = _z_for(agent, g_obs)                                       # [N, d]
    cond_all = _cond_for(agent, cfg, g_obs)                            # [N, d or obs]
    rz_all = (b_pool_t @ z_all.T).T / np.sqrt(float(d))                # [N, P]
    actor = _actor(agent, cfg)

    K4 = max(1, args.actions // 4)
    K = 4 * K4
    unif_scale = float(np.sqrt(act_dim * 2.0 / 3.0))   # E||u-u'|| for u~U[-1,1]^A

    rows = []
    for i, pr in enumerate(pairs):
        s = obs_all[pr["s_idx"]]
        cond = cond_all[i]
        obs_b = np.tile(s[None], (K, 1))
        cond_b = np.tile(cond[None], (K, 1))

        # --- action set: dataset on-trajectory / dataset elsewhere / actor / uniform
        lo, hi = pr["traj_lo"], max(pr["traj_lo"] + 1, pr["traj_hi"])
        t_idx = np.clip(np.arange(lo, lo + K4), 0, min(hi, n_ds - 1))
        a_traj = act_all[t_idx]
        a_dsrand = act_all[rng.integers(0, n_ds, size=K4)]
        a_actor = np.zeros((K4, act_dim), np.float32)
        for k in range(K4):
            a_actor[k] = np.clip(np.asarray(actor.sample_actions(
                observations=jnp.asarray(obs_b[:1]), goals=jnp.asarray(cond_b[:1]),
                seed=jax.random.PRNGKey(1000 + k), temperature=0.0))[0], -1, 1)
        a_unif = rng.uniform(-1.0, 1.0, size=(K4, act_dim)).astype(np.float32)
        acts = np.concatenate([a_traj, a_dsrand, a_actor, a_unif], 0).astype(np.float32)
        sources = (["traj"] * K4 + ["dsrand"] * K4 + ["actor"] * K4 + ["unif"] * K4)
        a_ref_row = 2 * K4          # the reference pi(s,cond): actor sample, seed 1000

        logits = logits_for(agent, obs_b, acts, cond_b, b_pool, temp, tilt_beta, z_all[i])
        full = readouts_from_logits(logits, kg_all[i], rz_all[i])
        h1 = readouts_from_logits(logits, kg_all[i], rz_all[i], slice(0, half))
        h2 = readouts_from_logits(logits, kg_all[i], rz_all[i], slice(half, 2 * half))

        rec = {"pair": i, "on_traj": pr["on_traj"], "s_idx": pr["s_idx"],
               "g_idx": pr["g_idx"]}
        for tag in ("minhead", "wmin"):
            qg, qz = full[f"q_g_{tag}"], full[f"q_z_{tag}"]
            rec[f"spearman_{tag}"] = spearman(qg, qz)
            rec[f"span_g_{tag}"] = span_stats(qg)
            rec[f"span_z_{tag}"] = span_stats(qz)
        qg, qz = full["q_g_minhead"], full["q_z_minhead"]
        rec["mc_g"] = mc_floor(h1["q_g_minhead"], h2["q_g_minhead"])
        rec["mc_z"] = mc_floor(h1["q_z_minhead"], h2["q_z_minhead"])

        lab = action_labels(sources)
        src = np.asarray(sources)
        if pr["on_traj"]:
            m_all = lab >= 0
            m_unif = (src == "traj") | (src == "unif")
            m_ds = (src == "traj") | (src == "dsrand")
            rec["auc_g"] = auc_mw(qg[m_all], lab[m_all] == 1)
            rec["auc_z"] = auc_mw(qz[m_all], lab[m_all] == 1)
            rec["auc_g_vs_unif"] = auc_mw(qg[m_unif], src[m_unif] == "traj")
            rec["auc_z_vs_unif"] = auc_mw(qz[m_unif], src[m_unif] == "traj")
            rec["auc_g_vs_dsrand"] = auc_mw(qg[m_ds], src[m_ds] == "traj")
            rec["auc_z_vs_dsrand"] = auc_mw(qz[m_ds], src[m_ds] == "traj")
        for tag, q in (("g", qg), ("z", qz)):
            sp = float(q.max() - q.min())
            dl = float(q.max() - q[a_ref_row])
            rec[f"delta_actor_{tag}"] = dl
            rec[f"delta_over_span_{tag}"] = float(dl / sp) if sp > 0 else float("nan")

        # --- ACTOR sensitivity (the gap density_w_ztv never closed) ---
        xi = jax.random.PRNGKey(7)
        a0 = np.asarray(actor.sample_actions(observations=jnp.asarray(obs_b[:1]),
                                             goals=jnp.asarray(cond_b[:1]),
                                             seed=xi, temperature=0.0))[0]
        n_s = args.sens_draws
        alt_goal = obs_all[rng.integers(0, n_ds, size=n_s)]
        cond_alt = _cond_for(agent, cfg, alt_goal)                       # [n_s, cond]
        s_goal, s_sphere, s_noise = [], [], []
        for k in range(n_s):
            ak = np.asarray(actor.sample_actions(
                observations=jnp.asarray(obs_b[:1]),
                goals=jnp.asarray(cond_alt[k][None]), seed=xi, temperature=0.0))[0]
            s_goal.append(float(np.linalg.norm(ak - a0)))
            an = np.asarray(actor.sample_actions(
                observations=jnp.asarray(obs_b[:1]), goals=jnp.asarray(cond_b[:1]),
                seed=jax.random.PRNGKey(500 + k), temperature=0.0))[0]
            s_noise.append(float(np.linalg.norm(an - a0)))
            if is_z:
                zr = np.asarray(_sphere(jax.random.normal(
                    jax.random.PRNGKey(900 + k), (1, d)), float(d)), np.float32)
                az = np.asarray(actor.sample_actions(
                    observations=jnp.asarray(obs_b[:1]), goals=jnp.asarray(zr),
                    seed=xi, temperature=0.0))[0]
                s_sphere.append(float(np.linalg.norm(az - a0)))
        rec["sens_cond_goal"] = float(np.mean(s_goal))
        rec["sens_cond_sphere"] = float(np.mean(s_sphere)) if s_sphere else None
        rec["sens_noise_floor"] = float(np.mean(s_noise))
        rec["sens_unif_ceiling"] = unif_scale
        rows.append(rec)
        if (i + 1) % 8 == 0:
            print(f"[p0a] pair {i+1}/{len(pairs)} ({time.time()-t00:.0f}s)", flush=True)

    def med(key, filt=None):
        v = [r[key] for r in rows if key in r and r[key] is not None
             and not (isinstance(r[key], float) and np.isnan(r[key]))
             and (filt is None or filt(r))]
        return float(np.median(v)) if v else None

    summary = {
        "n_pairs": len(rows), "n_actions": K, "P": P,
        "spearman_median": med("spearman_minhead"),
        "spearman_wmin_median": med("spearman_wmin"),
        "spearman_iqr": [float(np.percentile([r["spearman_minhead"] for r in rows
                                              if not np.isnan(r["spearman_minhead"])], p))
                         for p in (25.0, 75.0)],
        "span_g_median": float(np.median([r["span_g_minhead"]["span"] for r in rows])),
        "span_z_median": float(np.median([r["span_z_minhead"]["span"] for r in rows])),
        "iqr_g_median": float(np.median([r["span_g_minhead"]["iqr"] for r in rows])),
        "iqr_z_median": float(np.median([r["span_z_minhead"]["iqr"] for r in rows])),
        "relspan_g_median": float(np.median(
            [r["span_g_minhead"]["span"] / abs(r["span_g_minhead"]["mean"])
             for r in rows if r["span_g_minhead"]["mean"] != 0])),
        "relspan_z_median": float(np.median(
            [r["span_z_minhead"]["span"] / abs(r["span_z_minhead"]["mean"])
             for r in rows if r["span_z_minhead"]["mean"] != 0])),
        "snr_g_median": float(np.median([r["mc_g"]["snr"] for r in rows])),
        "snr_z_median": float(np.median([r["mc_z"]["snr"] for r in rows])),
        "snr_g_uncentered_median": float(np.median(
            [r["mc_g"]["snr_uncentered"] for r in rows])),
        "snr_z_uncentered_median": float(np.median(
            [r["mc_z"]["snr_uncentered"] for r in rows])),
        "selfcons_g_median": float(np.median([r["mc_g"]["selfcons"] for r in rows])),
        "selfcons_z_median": float(np.median([r["mc_z"]["selfcons"] for r in rows])),
        "auc_g_median": med("auc_g"), "auc_z_median": med("auc_z"),
        "auc_g_vs_unif_median": med("auc_g_vs_unif"),
        "auc_z_vs_unif_median": med("auc_z_vs_unif"),
        "auc_g_vs_dsrand_median": med("auc_g_vs_dsrand"),
        "auc_z_vs_dsrand_median": med("auc_z_vs_dsrand"),
        "auc_chance": 0.5,
        "delta_actor_g_median": med("delta_actor_g"),
        "delta_actor_z_median": med("delta_actor_z"),
        "delta_over_span_g_median": med("delta_over_span_g"),
        "delta_over_span_z_median": med("delta_over_span_z"),
        "sens_cond_goal_median": med("sens_cond_goal"),
        "sens_cond_sphere_median": med("sens_cond_sphere"),
        "sens_noise_floor_median": med("sens_noise_floor"),
        "sens_unif_ceiling": unif_scale,
        "z_geometry": z_geometry(_z_for(agent, obs_all[
            rng.choice(n_ds, size=min(4096, n_ds), replace=False)]), rng),
    }

    out = {"meta": {
        "run_dir": str(run_dir), "checkpoint_step": args.checkpoint_step,
        "env": env_name, "is_density_z": is_z, "d": d, "temp": temp,
        "tilt_beta": tilt_beta, "pool_size": P, "seed": args.seed,
        "hit_spec": ",".join(f"{a}:{b}:{e}" for a, b, e in blocks),
        "spec": "PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md P0a",
        "readouts": {
            "q_g": "sum_j w_j(s,a,cond) * kg_j, kg = _delta_phi(goal_obs, u_raw, "
                   "blocks) -- the QUERY-SET SOFTMAX-NORMALIZED kernel (spec 28 "
                   "sec.0 audit correction), not a raw kappa",
            "q_z": "sum_j w_j(s,a,cond) * B~(u_j).z(g)/sqrt(d), B~ = target_density_b",
            "w": "softmax_j(F_h(s,a,cond).B(u_j)/temp) at the ONLINE towers"
                 + (" + beta*z.B(u)/sqrt(d) (spec 25 C1 tilt)" if tilt_beta else ""),
            "head_reduction": "PRIMARY = per-head dot then min over heads (the "
                              "TRAINING convention); *_wmin = min over the two "
                              "probability vectors (spec-18 eval convention)",
            "cond": "z(g)=_density_z_encode(g)" if is_z else "raw goal observation",
        },
        "chance_refs": {"auc": 0.5,
                        "spearman_null": 0.0,
                        "sens_floor": "sens_noise_floor (same cond, resampled actor "
                                      "noise) -- a cond-sensitivity at or below this "
                                      "is indistinguishable from sampling noise",
                        "sens_ceiling": "E||u-u'|| for uniform actions"},
    }, "summary": summary, "pairs": rows}

    if args.gate:
        assert not is_z, "--gate is for the G-INDEXED anchor (control) only"
        ok, grows = gate_p0a(summary)
        out["control_gate"] = {"pre_registered_thresholds": GATE_THRESHOLDS,
                               "blocking_pass": bool(ok), "rows": grows}
        print(f"[p0a] CONTROL GATE blocking_pass={ok}")
        for r in grows:
            print(f"    {r['check']}: {r['value']:.4f} "
                  f"{'>=' if r['direction']=='ge' else '<='} {r['threshold']} "
                  f"-> {'PASS' if r['ok'] else 'FAIL'}"
                  f"{'' if r['blocking'] else '  (non-blocking)'}")

    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec28_p0a" / f"p0a_{run_dir.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=lambda o: float(o)))
    print(f"[p0a] DONE ({time.time()-t00:.0f}s) -> {outp}", flush=True)
    print(f"[p0a] spearman(q_g,q_z) med={summary['spearman_median']}  "
          f"span_g={summary['span_g_median']:.3e} span_z={summary['span_z_median']:.3e}  "
          f"snr_g={summary['snr_g_median']:.2f} snr_z={summary['snr_z_median']:.2f}  "
          f"AUC_g={summary['auc_g_median']} AUC_z={summary['auc_z_median']} "
          f"(chance .500)  d/span_g={summary['delta_over_span_g_median']} "
          f"d/span_z={summary['delta_over_span_z_median']}  "
          f"sens_goal={summary['sens_cond_goal_median']} "
          f"(noise floor {summary['sens_noise_floor_median']}, "
          f"unif ceiling {unif_scale:.2f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
