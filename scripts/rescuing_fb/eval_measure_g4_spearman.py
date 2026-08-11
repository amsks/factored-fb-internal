#!/usr/bin/env python
"""scripts/rescuing_fb/eval_measure_g4_spearman.py -- spec 17 sec.6 / spec 18
sec.7 / spec 19 sec.1 gate G4: Spearman of MC region masses vs the model's
own read-out masses, on the JAX gciql_flow stack (measure_density AND
measure_delta_partition checkpoints).

BACKGROUND -- this gate has been WRITTEN three times and never RUN once on
these checkpoints: `PAPER/Multistep-FB/specs/17-delta-measure.md` sec.6 G4
("Spearman of MC region/cell masses >= 0.65 on the post-hoc calibration
read"), `specs/18-density-measure.md` sec.7 G4 ("Run it for part K256 too --
it is still owed."), `specs/19-ors-program.md` sec.1 P0.1 (restates the debt
as a P0 blocker). This script closes it.

REFERENCE PROTOCOL (reused, not reinvented) -- the valid-measure campaign's
G2/G4 Spearman machinery on the TORCH FB stack,
`scripts/rescuing_fb/torchfb_full_measure.py` part B ("NUMERATOR /
calibration"): for `--starts` dataset rows, teleport the env to that row's
EXACT (qpos, qvel), execute one action a0 from the trained actor conditioned
on a fixed goal g, then MC-roll the SAME goal-conditioned policy for
`--horizon` steps, accumulating discounted box-membership `sum_k gamma^k
1[x_{1+k} in box]` for `--boxes` query boxes at each of `env_geometry.MULTS`
half-widths (drawn ONCE, matching cube-single's "random" probe-center
protocol byte-for-byte -- see `env_geometry.geometry_for(...).probe_centers`
and `torchfb_full_measure.py`'s own `fixed_queries` comment). That MC number
is compared, by Spearman rank correlation, against the model's OWN read-out
mass of the SAME box, computed from the SAME (s0, a0, g) via the checkpoint's
OWN zero-shot readout algebra:

  density   (spec 18 sec.1/6, `eval_density_zeroshot.py`): `w^T 1_box(U_P) /
            (1-gamma)` -- `eval_density_zeroshot.compute_density_primary`
            called with r_x0=0 AND kappa_x0=0, r_pool=the box indicator over
            the SAME query pool the checkpoint's density_f/density_b towers
            were scored against (`build_pool` / `density_query`, imported
            live, NOT reimplemented).
  partition (spec 17b, `eval_partition_zeroshot.py`): the analogous term,
            `p^T 1_box(protos) / (1-gamma)` --
            `eval_partition_zeroshot.compute_Qr_from_dots` called with
            r_x0=0 AND the phi(x0) term=0, z_r=the box indicator over the
            checkpoint's OWN (sha256-verified) K prototypes.

BUGFIX (2026-08-06, `.superpowers/sdd/spec19-ors/g4-verification.md`,
adversarial re-verification): the FIRST version of this script called these
same two functions with r_x0=0 but kept the paired `-kappa(x0|U_P))^T` /
`-phi(x0))^T` subtraction. Spec 18 sec.1's own Q_r formula, `(r(x0) + (w -
kappa(x0))^T r) / (1-gamma)`, defines r(x0) and -kappa(x0)^T r as a
DE-BIASING PAIR that CANCELS ("exact-r at x0 replaces the smoothed
estimate") -- both terms describe the SAME point x0 through two different
estimators, and the intent is `r(x0)` (exact) MINUS its own kernel-smoothed
self-estimate `kappa(x0)^T r` (~0, a small mollifier residual), leaving
`w^T r / (1-gamma)` as the actual successor-mass estimand -- exactly the MC
object `sum_k gamma^k 1[x_{1+k} in box]` (both count mass from the state
AFTER (s,a), never x0 itself). Zeroing r(x0) alone left `-kappa(x0)^T
1_box(U_P) / (1-gamma) = -1[x0 in box]/(1-gamma) = -100*1[x0 in box]`
injected into the readout -- and boxes containing x0 are exactly the
high-MC boxes, so the injected term is ANTI-correlated with MC truth. This
capped the pooled Spearman at ~0.17 across all 6 checkpoints (verified: the
old output equalled `w.r_pool/(1-gamma) - kappa(x0).r_pool/(1-gamma)` to
3e-14). Fixed by reading `w^T r_pool / (1-gamma)` (`p^T z_r / (1-gamma)`)
directly -- pass kappa_x0/phi_x0=0, not r_x0=0.

Both are literally the FIRST term of each script's own Q_r formula (the
r(x0)/-kappa(x0) pair cancelled per the derivation above) with r_pool/z_r
replaced by a box-membership indicator -- exactly the "successor MASS of a
set", not a reward value -- so this file reuses `compute_density_primary` /
`compute_Qr_from_dots` unmodified (zeroing the SECOND operand, not r_x0)
rather than re-deriving the readout. `env_geometry.HitSpec.box_contains`
(L-inf, already used by `full_measure.FullMeasure.member_indices` for 'box'
queries) is the SAME membership test on both the MC side and the pool/proto
side.

Spearman: the GATING statistic is `spearman_tie` (this file, average-rank
tie correction via `scipy.stats.rankdata`). BUGFIX (same verification,
item 3): `measure_readout_head.spearman` (kept alongside for reference --
the exact function the valid-measure campaign's G2 table,
`PAPER/rescuing-fb/PHASE2-LEDGER.md` sec.9.15, quotes throughout) uses
ORDINAL ranks (`np.argsort(np.argsort(x))`) with no tie averaging; ~60% of
this probe's MC values are exactly 0.0 (box never entered), and ordinal
ranking breaks that tie block by array order instead of assigning the tied
average rank -- measurably suppressing the correlation on this data (a
model with a perfect ordering on the non-degenerate points and an arbitrary
one within the zero-tie block scores ~0.75 under ordinal ranking vs. ~0.88
under tie-corrected -- both bounded below 1.0, since y genuinely varies
within the x-tied block and a correctly tie-corrected correlation reports
that as unexplained variance rather than hiding it; the gap between the two
is the artifact this fixes, not a claim that either reaches 1.0).

DETERMINISM NOTE (this session's own finding): every jax call below is seeded
explicitly (`jax.random.PRNGKey` built from `--seed`/start/mc-round indices),
so a re-run with the same flags is bit-identical -- verified by running the
readout twice and diffing the output JSON (see task report).

POOL SIZE NOTE (this session's own finding): the density PRIMARY readout is
under-converged at P=4096 and converged at P>=16384 (drift 0.000 16384->
32768 in the pool-sweep smoke); this script's `--pool-size` therefore
defaults to 16384, not `eval_density_zeroshot.py`'s own P=4096 default. Stay
at or below 32768 (P=65536 OOM-killed).

Run (CPU-only; GPUs are busy):
  taskset -c 24-39 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu \\
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_measure_g4_spearman.py \\
    --run-dir '/home/adityamohan/exp_spec18/factored-fb/GCIQL-CubeSingle-Flow-Measure-Density-d50/d18_cs_dens_s0' \\
    --checkpoint-step 500000 --starts 24 --mc 2 --horizon 200 --boxes 12
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

from scipy.stats import rankdata  # noqa: E402

from env_geometry import episode_bounds, geometry_for, resolve_dataset  # noqa: E402
from eval_density_zeroshot import (  # noqa: E402
    build_pool,
    compute_density_primary,
    density_query,
)
from eval_partition_zeroshot import compute_Qr_from_dots, load_protos_verified  # noqa: E402
from measure_readout_head import spearman  # noqa: E402 -- ordinal-rank reference (kept for
                                            # comparison; NOT used for the gate -- see spearman_tie)

MULTS = (1.0, 2.0, 4.0)   # env_geometry.MULTS -- box half-width multipliers


def partition_box_mass(p_h, z_r, double_q):
    """Partition box mass under the DEPLOYED head aggregation: dot each head's
    softmax with z_r, then min the SCALARS -- exactly eval_partition_zeroshot.py's
    `qh = einsum('hnk,k->hn', p, z_r); q = min(qh, axis=0)`. RESCORING FIX
    2026-08-11 (code-review finding 1): the first version of this probe instead
    took the ELEMENTWISE min of the two softmax vectors and dotted once --
    density's convention, a sub-normalized vector whose box masses are
    systematically deflated relative to the estimator the method actually
    deploys. The G4 statistic must be computed on the deployed estimator; g4
    JSONs written before this fix carry no `head_agg` meta key and quote the
    mismatched one."""
    q_h = np.einsum("hnk,k->hn", np.asarray(p_h, np.float64), np.asarray(z_r, np.float64))
    return q_h.min(axis=0) if (double_q and q_h.shape[0] > 1) else q_h[0]


def spearman_tie(x, y) -> float:
    """Tie-corrected Spearman (average ranks via scipy.stats.rankdata), used for
    the gate. BUGFIX (g4-verification.md item 3): `measure_readout_head.spearman`
    uses `np.argsort(np.argsort(x))` -- ORDINAL ranks, no tie averaging -- which
    silently breaks ties by array order. On this probe ~60% of MC values are
    exactly 0.0 (the box was never entered): ordinal ranking assigns those points
    an arbitrary permutation of low ranks instead of the tied average rank, which
    measurably suppresses the correlation (a perfect-on-nonzero, arbitrary-on-ties
    model scores ~0.75 under ordinal ranking vs. ~0.88 under this tie-corrected
    one; see tests/test_eval_measure_g4_spearman.py). Mirrors
    `measure_readout_head.spearman`'s signature/nan-guard exactly, differing only
    in the ranking function."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


# ============================ query-box construction (torch precedent) =============

def probe_rows(policy: str, bounds, start_row: int, n: int, n_rows: int, rng) -> np.ndarray:
    """Mirror of `torchfb_full_measure.probe_rows` (pure numpy, not imported --
    that module is torch-only at its top level; this is 5 lines of rng logic,
    reproduced not reinvented). 'random' = the archived cube-single protocol
    (uniform dataset draw, independent of the start row)."""
    if policy == "random":
        return rng.choice(n_rows, min(n, n_rows), replace=False)
    from env_geometry import hindsight_rows
    if policy == "hindsight":
        return hindsight_rows(bounds, start_row, rng, n)
    raise ValueError(f"unknown probe-center policy {policy!r}")


def build_fixed_queries(geo, coords_all, bounds, n_boxes, rng):
    """'random' probe policy (cube-single): draw n_boxes centers PER
    multiplier, ONCE, before any start/goal row is touched -- byte-for-byte
    the `torchfb_full_measure.py` `fixed_queries` protocol (the rng draw
    ORDER is part of the archived protocol). Returns a flat list of
    {'center','mult'} dicts, or None if this env's probe policy is
    'hindsight' (per-start draw; not needed for cube-single, kept for
    completeness against future envs)."""
    if geo.probe_centers != "random":
        return None
    n_rows = len(coords_all)
    queries = []
    for mult in MULTS:
        rows = probe_rows("random", bounds, 0, n_boxes, n_rows, rng)
        for c in coords_all[rows]:
            queries.append({"center": c, "mult": float(mult)})
    return queries


# ============================ live-agent MC + readout (jax-dependent) ===============

def run_one_checkpoint(args) -> dict:
    import jax
    import jax.numpy as jnp
    import mujoco
    import ogbench

    cands = sorted(glob.glob(str(Path(args.run_dir).expanduser())))
    assert cands, f"no run dir matched {args.run_dir}"
    run_dir = Path(cands[0]).resolve()
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(flags["agent"])

    is_density = bool(cfg.get("measure_density", False))
    is_partition = bool(cfg.get("measure_delta_partition", False))
    assert is_density or is_partition, (
        f"{run_dir} is neither measure_density nor measure_delta_partition "
        f"(measure_density={cfg.get('measure_density')!r}, "
        f"measure_delta_partition={cfg.get('measure_delta_partition')!r}) -- "
        "this script's readout only covers those two spec 17/18 objects.")
    obj = "density" if is_density else "partition"
    if args.object != "auto":
        assert args.object == obj, (
            f"--object {args.object} but checkpoint's recorded config says {obj}")

    env_name = args.env or flags["env_name"]
    gamma = float(cfg["discount"])
    double_q = bool(cfg.get("double_q", True))
    geo = geometry_for(env_name, str(cfg.get("measure_hit_spec") or ""))
    spec = geo.hit_spec
    blocks = spec.blocks

    # ---- dataset with qpos/qvel (env_geometry.resolve_dataset -- the SAME
    # npz `torchfb_full_measure.py` teleports from; the jax-side
    # `ogbench.make_env_and_datasets` dataset has NO qpos/qvel, so it cannot
    # supply exact start states for this protocol). ----
    npz_path = resolve_dataset(str(cfg.get("data_path") or ""), geo.dataset_name)
    npz = np.load(npz_path)
    obs_all = np.asarray(npz["observations"], np.float32)
    qpos_all, qvel_all = np.asarray(npz["qpos"]), np.asarray(npz["qvel"])
    coords_all = spec.coords(obs_all)
    bounds = episode_bounds(np.asarray(npz["terminals"]))
    n_rows = len(obs_all)

    eval_tasks_env = f"{env_name[:-3]}-singletask-task1-v0"
    env = ogbench.make_env_and_datasets(eval_tasks_env, env_only=True)
    env.reset(seed=0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    from t3_offpeak_multienv import load_agent_dims
    _impls = str(REPO / "third_party" / "ogbench" / "impls")
    while _impls in sys.path:
        sys.path.remove(_impls)
    sys.path.insert(0, _impls)
    _u = sys.modules.get("utils")
    if _u is not None and not hasattr(_u, "__path__"):
        del sys.modules["utils"]
    agent = load_agent_dims(str(run_dir), args.checkpoint_step, obs_dim, act_dim)

    # ---- object-specific setup ----
    if obj == "density":
        temp = float(cfg.get("measure_density_temp", 1.0))
        rng_np0 = np.random.default_rng(args.seed)
        pool_idx = rng_np0.choice(n_rows, size=min(args.pool_size, n_rows), replace=False)
        pool_obs = obs_all[pool_idx]
        u_raw, b_pool = build_pool(agent, blocks, pool_obs, spec)
        pool_coords = spec.coords(pool_obs)
        print(f"[g4] density pool: P={len(pool_obs)}", flush=True)
    else:
        protos_path = cfg.get("measure_delta_protos", "")
        assert protos_path, f"{run_dir} flags.json has no measure_delta_protos recorded"
        protos, protos_info = load_protos_verified(protos_path)
        agent_protos = np.asarray(agent.delta_protos, np.float32)
        assert agent_protos.shape == protos.shape and np.allclose(agent_protos, protos), (
            "agent.delta_protos != independently-loaded+verified prototype file")
        K = int(protos.shape[0])
        print(f"[g4] partition protos: K={K} sha256={protos_info['sha256'][:12]}...", flush=True)

    print(f"[g4] restored {run_dir.name} object={obj} step={args.checkpoint_step} "
          f"env={env_name} gamma={gamma} double_q={double_q} "
          f"hit_spec={cfg.get('measure_hit_spec')!r}", flush=True)

    rng = np.random.default_rng(args.seed)
    queries = build_fixed_queries(geo, coords_all, bounds, args.boxes, rng)
    assert queries is not None, f"probe policy {geo.probe_centers!r} not yet supported here"
    n_q = len(queries)
    q_centers = np.stack([q["center"] for q in queries])          # [n_q, d_c]
    q_mults = np.array([q["mult"] for q in queries])

    start_rows = rng.choice(n_rows, args.starts, replace=False)
    goal_rows = rng.choice(n_rows, args.starts, replace=False)

    recs = []
    t0 = time.time()
    u = env.unwrapped
    for si, (s_row, g_row) in enumerate(zip(start_rows, goal_rows)):
        s0_obs = obs_all[s_row]
        g_obs = obs_all[g_row]
        s0_j = jnp.asarray(s0_obs, jnp.float32)[None]
        g_j = jnp.asarray(g_obs, jnp.float32)[None]
        key = jax.random.PRNGKey(args.seed * 1_000_003 + int(s_row))
        key, sub = jax.random.split(key)
        a0 = np.clip(np.asarray(agent.sample_actions(
            observations=s0_j, goals=g_j, seed=sub, temperature=0.0)), -1, 1)      # [1, act]

        # ---- model readout mass: fixed (s0, a0, g) scored once, dotted per box ----
        #
        # BUGFIX (g4-verification.md, adversarial review): spec 18 sec.1 / 17b define
        # r(x0) and -kappa(x0)/-phi(x0) as a DE-BIASING PAIR that CANCELS ("exact-r at
        # x0 replaces the smoothed estimate"), leaving w^T r / (1-gamma) as the object
        # that matches the MC ground truth (both count mass from the NEXT state, x_1
        # onward -- x0 itself is never in either sum). The previous version zeroed
        # r(x0) but kept the paired -kappa(x0)/-phi(x0) subtraction, injecting a spurious
        # -1[x0 in box]/(1-gamma) = -100*1[x0 in box] term into a 0-100 quantity --
        # anti-correlated with MC truth because boxes containing x0 are exactly the
        # high-MC boxes. Verified: the old code's output == w.r_pool/(1-gamma) -
        # kappa(x0).r_pool/(1-gamma) to 3e-14. Fix: read w^T r_pool / (1-gamma) directly
        # (equivalently: call the SAME formula with kappa_x0/phi_x0 zeroed, not r_x0).
        if obj == "density":
            w, _kappa_x0_unused = density_query(agent, s0_j, jnp.asarray(a0), g_j,
                                                u_raw, b_pool, blocks, temp, double_q)  # [1,P] each
            model_mass = np.empty(n_q)
            for qi in range(n_q):
                r_pool = spec.box_contains(pool_coords, q_centers[qi], mult=q_mults[qi]).astype(np.float64)
                model_mass[qi] = float(compute_density_primary(
                    0.0, w[0], np.zeros_like(r_pool), r_pool, gamma))          # w^T r_pool / (1-gamma)
        else:
            logits = np.asarray(agent.network.select("delta_head")(
                jnp.concatenate([s0_j, jnp.asarray(a0), g_j], -1)))                 # [2,1,K]
            p_h = np.exp(logits - logits.max(-1, keepdims=True))
            p_h = p_h / p_h.sum(-1, keepdims=True)                                 # [2,1,K]
            model_mass = np.empty(n_q)
            for qi in range(n_q):
                z_r = spec.box_contains(protos, q_centers[qi], mult=q_mults[qi]).astype(np.float64)
                q = partition_box_mass(p_h, z_r, double_q)                     # [1], dot-then-min
                model_mass[qi] = float(compute_Qr_from_dots(
                    0.0, float(q[0]), 0.0, gamma))                             # min_h p_h^T z_r / (1-gamma)

        # ---- MC ground truth: teleport to (qpos,qvel)[s_row], step a0, roll g-cond policy ----
        occ = np.zeros(n_q)
        for r in range(args.mc):
            env.reset()
            u.data.qpos[:] = qpos_all[s_row]
            u.data.qvel[:] = qvel_all[s_row]
            mujoco.mj_forward(u.model, u.data)
            ob, *_ = env.step(a0[0])
            g_disc = 1.0
            for t in range(args.horizon):
                c = spec.coords(ob[None])[0]
                for qi in range(n_q):
                    if spec.box_contains(c[None], q_centers[qi], mult=q_mults[qi])[0]:
                        occ[qi] += g_disc
                g_disc *= gamma
                key, sub = jax.random.split(key)
                act = np.clip(np.asarray(agent.sample_actions(
                    observations=jnp.asarray(ob)[None], goals=g_j, seed=sub,
                    temperature=0.0)), -1, 1)
                ob, _rw, term, trunc, _info = env.step(act[0])
                if term or trunc:
                    break
        occ /= args.mc

        for qi in range(n_q):
            recs.append({"start": int(s_row), "mult": float(q_mults[qi]),
                        "mc": float(occ[qi]), "model_mass": float(model_mass[qi])})
        if (si + 1) % 4 == 0:
            print(f"[g4] {si+1}/{args.starts} starts done ({time.time()-t0:.0f}s)", flush=True)

    if hasattr(env, "close"):
        env.close()

    # ---- report: per-mult (PRIMARY diagnostic -- see note below) + pooled (ALL
    # mults combined -- the gate's own, unqualified "Spearman of MC region masses
    # vs read-out masses") + the non-degenerate (mc>0) subset. spearman_tie
    # (average-rank, tie-corrected) is the GATING statistic; the ordinal
    # `spearman` is kept alongside for reference/backward comparison only
    # (g4-verification.md item 3). ----
    mc_all = np.array([r["mc"] for r in recs])
    model_all = np.array([r["model_mass"] for r in recs])
    nondegen = mc_all > 0.0
    pooled = {
        "n": len(recs),
        "spearman": spearman_tie(mc_all, model_all),
        "spearman_ordinal_uncorrected": spearman(mc_all, model_all),
        "degenerate_frac": float(np.mean(mc_all == 0.0)),
    }
    nondegenerate = {
        "n": int(nondegen.sum()),
        "spearman": spearman_tie(mc_all[nondegen], model_all[nondegen]),
    }
    per_mult = {}
    for mult in MULTS:
        R = [r for r in recs if r["mult"] == mult]
        mc = np.array([r["mc"] for r in R]); mm = np.array([r["model_mass"] for r in R])
        per_mult[str(mult)] = {
            "n": len(R), "mc_mean": float(mc.mean()), "model_mass_mean": float(mm.mean()),
            "spearman": spearman_tie(mc, mm),
            "spearman_ordinal_uncorrected": spearman(mc, mm),
            "degenerate_frac": float(np.mean(mc == 0.0)),
        }

    gate_pass = bool(pooled["spearman"] >= 0.65)
    print(f"\n[g4] {run_dir.name} ({obj}) POOLED spearman(tie-corr)={pooled['spearman']:.3f} "
          f"(ordinal={pooled['spearman_ordinal_uncorrected']:.3f}) n={pooled['n']} "
          f"degenerate_frac={pooled['degenerate_frac']:.2f} "
          f"-> {'PASS' if gate_pass else 'FAIL'} (gate >= 0.65)", flush=True)
    print(f"  non-degenerate subset (mc>0): spearman(tie-corr)={nondegenerate['spearman']:.3f} "
          f"n={nondegenerate['n']}", flush=True)
    print("  PER-MULTIPLIER (pooling is NOT innocent -- box size drives both variables, "
          "so the pooled figure sits above every column except mult=4; read this row, "
          "not just the pooled number):", flush=True)
    for mult in MULTS:
        s = per_mult[str(mult)]
        print(f"    mult={mult} (half-width {mult*0.4:.1f}): spearman(tie-corr)={s['spearman']:.3f} "
              f"(ordinal={s['spearman_ordinal_uncorrected']:.3f}) n={s['n']} "
              f"degenerate_frac={s['degenerate_frac']:.2f}", flush=True)

    return {
        "run_dir": str(run_dir), "object": obj, "checkpoint_step": args.checkpoint_step,
        "env": env_name, "gamma": gamma, "double_q": double_q,
        "hit_spec": cfg.get("measure_hit_spec"), "seed": args.seed,
        "protocol": {
            "reference": "scripts/rescuing_fb/torchfb_full_measure.py part B "
                        "(valid-measure campaign G2/G4 Spearman machinery); "
                        "PAPER/rescuing-fb/PHASE2-LEDGER.md sec.9.15",
            "starts": args.starts, "mc": args.mc, "horizon": args.horizon,
            "boxes_per_mult": args.boxes, "mults": list(MULTS),
            "pool_size": args.pool_size if obj == "density" else None,
            "readout": ("w^T box_indicator(U_P) / (1-gamma)  [compute_density_primary(r_x0=0, "
                       "w, kappa_x0=0, box_indicator, gamma) -- see g4-verification.md: r(x0) "
                       "and kappa(x0) are a de-biasing PAIR that cancels; dropping only r(x0) "
                       "while keeping -kappa(x0) was the bug fixed 2026-08-06]"
                       if obj == "density" else
                       "min_h(p_h^T box_indicator(protos)) / (1-gamma)  [compute_Qr_from_dots(r_x0=0, "
                       "partition_box_mass, phi(x0)_term=0, gamma) -- phi(x0) pair dropped (2026-08-06 "
                       "fix) AND dot-then-min head aggregation matching eval_partition_zeroshot "
                       "(2026-08-11 rescore fix; pre-rescore JSONs used vector-min-then-dot)]"),
            "head_agg": ("elementwise-vector-min of softmax heads (deployed density convention)"
                         if obj == "density" else
                         "dot-then-min of scalars (deployed partition convention, 2026-08-11)"),
            "mc_ground_truth": "sum_k gamma^k 1[x_{1+k} in box], box=env_geometry.HitSpec.box_contains "
                              "(L-inf), teleport to dataset (qpos,qvel)[s_row], a0 then g-conditioned "
                              "policy rollout (exact torchfb_full_measure.py part B translation)",
            "spearman_fn": "spearman_tie (average-rank, tie-corrected) -- see g4-verification.md "
                          "item 3; spearman_ordinal_uncorrected kept alongside for reference only",
        },
        "gate": {"threshold": 0.65, "statistic": "pooled Spearman (tie-corrected), all mults combined",
                 "value": pooled["spearman"], "pass": gate_pass,
                 "caveat": "pooling is not innocent -- box size drives both MC and model mass, "
                          "so the pooled figure sits above every individual mult column except "
                          "mult=4.0; see per_mult for the honest per-scale read"},
        "pooled": pooled, "nondegenerate_mc_gt_0": nondegenerate, "per_mult": per_mult,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--object", choices=["auto", "density", "partition"], default="auto")
    ap.add_argument("--env", default="")
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--mc", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--boxes", type=int, default=12)
    ap.add_argument("--pool-size", type=int, default=16384,
                    help="density only: query pool P (this session's finding: "
                         "converged at P>=16384, OOMs at 65536)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    res = run_one_checkpoint(args)
    outp = Path(args.out) if args.out else \
        (REPO / "results" / "spec19_g4" / f"g4_{Path(args.run_dir).name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, default=lambda o: float(o)))
    print(f"\n[g4] DONE -> {outp}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
