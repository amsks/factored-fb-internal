#!/usr/bin/env python
"""scripts/rescuing_fb/renewal_head.py -- Spec 07 item 2: the renewal head.

The successor measure, reconstructed on top of the FirstPassage-NoOrtho win at
provably zero control risk (nothing retrained; the towers are structurally
frozen -- sg is numpy). Renewal identity (spec 07 item 0 fact 2):

    M^pi(s, a, B_eps(g)) = E[gamma^{T_g - 1}] . R^pi(g) = Q_FP(s,a,g) . R(g)

All action-dependence lives in the (learned, working) FP factor; the
mixing-dominated gamma-mass lives in the action-free renewal factor R(g) =
expected discounted revisit mass of the eps-ball after first entry.

Deliverables (spec 07 item 2):
  2a  MC renewal targets (pure offline, no TD, units pinned by construction):
      for test goals g from dataset next-states, every trajectory ENTRY into
      B_eps(cube(g)) (entry = first passage into the ball; consecutive in-ball
      steps are ONE entry) contributes
          R_tgt = sum_{t >= t_entry} gamma^(t - t_entry) 1[cube(s_t) in ball]
      averaged over entries per goal. Head = scalar MLP 2x256 GELU on either
      sg[psi(g)] (--input psi, the deployable one) or cube(g) (--input cube,
      the position-only control). psi := the CRITIC net's psi, head-mean
      (gciql_flow._critic_psi) -- under measure_only the value net is never
      trained, so the critic psi IS the trained backward map.
  2b  --fit-fp-mu: the honest single-policy FP_mu(s,a,g) by within-trajectory
      first-hit MC regression, target gamma^(T-1) (T = first k>=1 with
      cube(s_{t+k}) in ball), censored-as-zero at episode end; the censoring
      fraction is REPORTED and claims are restricted to the uncensored-
      dominated region. Input = raw [s, a, cube(g)] (37-dim): this is an
      appendix object -- raw input keeps it independent of the learned towers
      so FP_mu x R is a tower-free behavior-policy measure; documented choice.
  2c  infer_z: FB's moment readout, renewal-weighted --
          z = matchnorm( mean_j r_j . R(g_j) . psi(g_j) )
      (R_vals=None => the plain moment; the mandatory ablation).
  Calibration eval (--eval, the paper figure): held-out episodes (last
  --holdout), empirical discounted-occupancy targets on an (s,a) x g grid
  (same protocol/buckets as measure_readout_head, spec 06 item 8, so the two
  heads are directly comparable); Spearman + calibration slope for the
  reconstruction sg[Q_FP(s,a,g)] . R(g), bucketed K<=5 / 5-20 / >20.

Units: cube xyz = obs[19:22] in OBS units (1 obs unit = 10 cm); eps = 0.4 obs
units = 4 cm (same hit ball as gciql_flow's measure_first_passage). gamma=0.99.

Pinned interface (spec 07 orchestration -- two agents build against this):
  CLI: --run-dir --dataset-path --input {psi,cube} --epsilon 0.4 --gamma 0.99
       --max-goals N --out PATH   (writes a .npz bundle: head params + metadata
       incl. the input kind)
  load_renewal_head(path) -> callable R(goal_obs [N, obs_dim]) -> [N]
       (psi-vs-cube handled internally from metadata; psi input lazily restores
       the recorded checkpoint and therefore needs the jax venv; cube input is
       pure numpy)
  infer_z(psi, rewards, R_vals, match_norm) -> [d]   (pure numpy)

Run (jax venv; CPU is fine):
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \\
    /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/renewal_head.py \\
    --run-dir '/home/adityamohan/exp_spec06/factored-fb/GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s0' \\
    --input psi --max-goals 2000 --eval --fit-fp-mu \\
    --out analysis/rescuing-fb/renewal_head_psi_s0.npz

Pre-registered reads (spec 07 item 2): control untouched (checkpoint hash
asserted unchanged); FPxR fidelity >= occupancy-refit fidelity per bucket;
flat-R diagnostic var[R]/mean[R]^2 reported; position-only R within noise of
psi-input R => the renewal factor is a cube-position functional.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# numpy-only helpers shared with the spec-06 item-8 head (same fidelity protocol
# => the two heads are directly comparable). measure_readout_head's module top
# level is numpy-only; jax stays inside its functions.
from measure_readout_head import (  # noqa: E402
    build_fidelity_grid, calibration_slope, fidelity_report, file_sha256,
    spearman, split_episodes,
)

CUBE_OBS = slice(19, 22)     # obs cube xyz, OBS units ((m - center) * 10)
EPS_OBS = 0.4                # 4 cm in obs units -- gciql_flow's first-passage ball
TASK_NAME = "cube-single-play-singletask-task1-v0"
_BUNDLE_KEYS = ("W0", "b0", "W1", "b1", "W2", "b2",
                "x_mean", "x_std", "y_mean", "y_std")


# ====================== pure-numpy math (unit-tested, no jax) ======================

def _gelu(x, xp=np):
    return 0.5 * x * (1.0 + xp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def mlp_forward(params, x, xp=np):
    """Scalar MLP 2x256 GELU. Works with numpy (inference) or jax.numpy (fit,
    pass xp=jnp -- same tanh GELU so the two forwards agree bit-for-float)."""
    h = _gelu(x @ params["W0"] + params["b0"], xp)
    h = _gelu(h @ params["W1"] + params["b1"], xp)
    return (h @ params["W2"] + params["b2"])[..., 0]


def predict_from_bundle(bundle, x):
    """Un-standardized head prediction from a saved bundle (pure numpy)."""
    xs = (np.asarray(x, np.float64) - bundle["x_mean"]) / bundle["x_std"]
    params = {k: bundle[k] for k in ("W0", "b0", "W1", "b1", "W2", "b2")}
    return mlp_forward(params, xs) * float(bundle["y_std"]) + float(bundle["y_mean"])


def _ball_entries(inball_ep):
    """First-passage entry indices within ONE episode's in-ball flag [L].
    Consecutive in-ball steps are one sojourn => ONE entry (its first step);
    an episode that STARTS in the ball counts as an entry at t=0."""
    idx = np.nonzero(np.asarray(inball_ep))[0]
    if len(idx) == 0:
        return idx
    first = np.ones(len(idx), bool)
    first[1:] = (idx[1:] - idx[:-1]) > 1
    return idx[first]


def renewal_targets(cube, spans, goals_xyz, gamma, eps):
    """MC renewal targets (spec 07 2a). cube [T, D], spans = [start, end) episode
    row spans, goals_xyz [N, D]. Per goal: over every episode entry e into the
    eps-ball, R_tgt(e) = sum_{t >= e, t < ep_end} gamma^(t-e) 1[cube_t in ball];
    return (mean over entries [N] (nan if never entered), n_entries [N])."""
    cube = np.asarray(cube, np.float64)
    goals = np.asarray(goals_xyz, np.float64)
    n = len(goals)
    tgts = np.full(n, np.nan)
    nent = np.zeros(n, np.int64)
    g_chunk, t_chunk = 64, 100_000
    for i0 in range(0, n, g_chunk):
        gg = goals[i0:i0 + g_chunk]
        inball = np.empty((len(gg), len(cube)), bool)
        for t0 in range(0, len(cube), t_chunk):
            c = cube[t0:t0 + t_chunk]
            d2 = ((gg[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            inball[:, t0:t0 + t_chunk] = d2 <= eps * eps
        for k in range(len(gg)):
            vals = []
            for a, b in spans:
                idx = np.nonzero(inball[k, a:b])[0]
                if len(idx) == 0:
                    continue
                first = np.ones(len(idx), bool)
                first[1:] = (idx[1:] - idx[:-1]) > 1
                for e in idx[first]:
                    later = idx[idx >= e]
                    vals.append(float(np.sum(gamma ** (later - e))))
            if vals:
                tgts[i0 + k] = float(np.mean(vals))
                nent[i0 + k] = len(vals)
    return tgts, nent


def fp_targets(cube, spans, q_rows, goals_xyz, gamma, eps):
    """MC first-passage targets under mu (spec 07 2b), pairwise: for each pair
    (query row t = q_rows[p], goal = goals_xyz[p]) the within-episode first hit
    T = min{k >= 1 : cube(s_{t+k}) in B_eps(goal)}; target gamma^(T-1);
    censored-as-zero when the episode ends without a hit (T >= 1 convention:
    being in the ball AT t does not count). Returns (targets [P], censored [P])."""
    cube = np.asarray(cube, np.float64)
    goals = np.asarray(goals_xyz, np.float64)
    q_rows = np.asarray(q_rows, np.int64)
    span_of = np.full(len(cube), -1, np.int64)
    ends = np.zeros(len(spans), np.int64)
    for i, (a, b) in enumerate(spans):
        span_of[a:b] = i
        ends[i] = b
    tgts = np.zeros(len(q_rows))
    cens = np.zeros(len(q_rows), bool)
    for p, t in enumerate(q_rows):
        si = span_of[t]
        assert si >= 0, f"query row {t} outside every span"
        tail = cube[t + 1: ends[si]]
        if len(tail) == 0:
            cens[p] = True
            continue
        hit = np.nonzero(((tail - goals[p]) ** 2).sum(-1) <= eps * eps)[0]
        if len(hit) == 0:
            cens[p] = True
        else:
            tgts[p] = gamma ** hit[0]      # T = hit[0]+1  =>  gamma^(T-1)
    return tgts, cens


def infer_z(psi, rewards, R_vals, match_norm):
    """Spec 07 2c, pinned interface (pure numpy). FB moment readout, renewal-
    weighted: z = mean_j rewards[j] * (R_vals[j] if R_vals is not None else 1.0)
    * psi[j], then renormalized to L2 norm match_norm. R_vals=None => the plain
    moment (the mandatory ablation)."""
    psi = np.asarray(psi, np.float64)
    r = np.asarray(rewards, np.float64)
    w = r if R_vals is None else r * np.asarray(R_vals, np.float64)
    z = (w[:, None] * psi).mean(axis=0)
    nrm = float(np.linalg.norm(z))
    if nrm == 0.0:
        return z                            # degenerate readout: all-zero moment
    return z * (float(match_norm) / nrm)


# ====================== bundle save / load (pinned interface) ======================

def save_head_bundle(path, bundle, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(bundle[k]) for k in _BUNDLE_KEYS},
             meta_json=json.dumps(meta))
    return path


def _load_bundle(path):
    data = np.load(Path(path), allow_pickle=False)
    bundle = {k: np.asarray(data[k]) for k in _BUNDLE_KEYS}
    meta = json.loads(str(data["meta_json"]))
    return bundle, meta


def load_renewal_head(path):
    """Pinned interface: -> callable R(goal_obs [N, obs_dim]) -> [N]. Input kind
    (psi vs cube) is read from the bundle metadata. cube: pure numpy (also
    accepts already-sliced [N,3] cube xyz). psi: lazily restores the recorded
    checkpoint (jax venv required). Predictions are clamped at 0 (R is a
    discounted revisit mass >= 1 at entry; a negative prediction is off-manifold
    and would flip infer_z weights)."""
    bundle, meta = _load_bundle(path)
    assert meta.get("kind", "renewal") == "renewal", \
        f"load_renewal_head expects a renewal bundle, got kind={meta.get('kind')}"
    kind = meta["input"]
    cache = {}

    def R(goal_obs):
        x = np.asarray(goal_obs, np.float32)
        single = x.ndim == 1
        if single:
            x = x[None]
        if kind == "cube":
            feats = x if x.shape[-1] == 3 else x[..., CUBE_OBS]
        else:
            if "psi" not in cache:
                from eval_rgfb_zeroshot import load_run
                agent, _flags = load_run(Path(meta["run_dir"]),
                                         int(meta["checkpoint_step"]))
                cache["psi"] = make_critic_psi(agent)
            feats = cache["psi"](x)
        out = np.maximum(predict_from_bundle(bundle, feats), 0.0)
        return out[0] if single else out

    return R


def load_fp_mu_head(path):
    """-> callable FP(s_obs [N, obs_dim], actions [N, act_dim], goal_obs) -> [N].
    goal_obs may be full obs or [N,3] cube xyz. Clamped to [0, 1] (gamma^(T-1))."""
    bundle, meta = _load_bundle(path)
    assert meta.get("kind") == "fp_mu", f"expects fp_mu bundle, got {meta.get('kind')}"

    def FP(s_obs, actions, goal_obs):
        s = np.asarray(s_obs, np.float64)
        a = np.asarray(actions, np.float64)
        g = np.asarray(goal_obs, np.float64)
        gc = g if g.shape[-1] == 3 else g[..., CUBE_OBS]
        x = np.concatenate([s, a, gc], axis=-1)
        return np.clip(predict_from_bundle(bundle, x), 0.0, 1.0)

    return FP


# ====================== jax-side plumbing (checkpoint-frozen) ======================

def make_critic_psi(agent):
    """psi := CRITIC net's psi, head-mean (gciql_flow._critic_psi) -- the trained
    backward map of the measure-only FP cell. Structurally sg: numpy out."""
    import jax.numpy as jnp

    def psi(goal_obs, chunk=4096):
        goal_obs = np.asarray(goal_obs, np.float32)
        out = []
        for i in range(0, len(goal_obs), chunk):
            g = jnp.asarray(goal_obs[i:i + chunk])
            out.append(np.asarray(agent._critic_psi(g)))
        return np.concatenate(out, 0)

    return psi


def make_qfp(agent):
    """sg[Q_FP(s,a,g)] = min over the two critic heads of phi(s,a).psi(g)/sqrt(d)
    (the trained first-passage matrix, evaluated at the Dirac column g)."""
    import jax.numpy as jnp
    fbz = bool(agent.config.get("fb_forward_z", False))

    def qfp(s_obs, actions, goal_obs, chunk=4096):
        s_obs = np.asarray(s_obs, np.float32)
        actions = np.asarray(actions, np.float32)
        goal_obs = np.asarray(goal_obs, np.float32)
        out = []
        for i in range(0, len(s_obs), chunk):
            o = jnp.asarray(s_obs[i:i + chunk])
            g = jnp.asarray(goal_obs[i:i + chunk])
            a = jnp.asarray(actions[i:i + chunk])
            zarg = (g,) if fbz else ()
            v = agent.network.select("critic")(o, g, a, *zarg)      # [2, B]
            out.append(np.asarray(jnp.min(v, axis=0)))
        return np.concatenate(out, 0)

    return qfp


def fit_head(X, Y, iters, batch, lr, seed, tag):
    """MSE fit of the 2x256 GELU scalar MLP (jax + optax -- the fit runs in the
    jax venv anyway for the psi/eval plumbing; inference is pure numpy via
    predict_from_bundle). Inputs/targets standardized; stats stored in bundle."""
    import jax
    import jax.numpy as jnp
    import optax

    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    y_mean, y_std = float(Y.mean()), float(Y.std()) + 1e-6
    Xs = ((X - x_mean) / x_std).astype(np.float32)
    Ys = ((Y - y_mean) / y_std).astype(np.float32)

    dims = (X.shape[1], 256, 256, 1)
    key = jax.random.PRNGKey(seed)
    params = {}
    for li in range(3):
        key, k = jax.random.split(key)
        params[f"W{li}"] = jax.random.normal(k, (dims[li], dims[li + 1])) / np.sqrt(dims[li])
        params[f"b{li}"] = jnp.zeros((dims[li + 1],))
    tx = optax.adam(lr)
    opt = tx.init(params)

    def loss_fn(p, x, y):
        return jnp.mean((mlp_forward(p, x, jnp) - y) ** 2)

    @jax.jit
    def step(p, o, x, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, y)
        upd, o = tx.update(g, o)
        return optax.apply_updates(p, upd), o, l

    rng = np.random.default_rng(seed)
    for it in range(iters):
        idx = rng.integers(0, len(Xs), size=min(batch, len(Xs)))
        params, opt, l = step(params, opt, jnp.asarray(Xs[idx]), jnp.asarray(Ys[idx]))
        if it % 500 == 0:
            print(f"[renewal/{tag}] it={it} mse_std={float(l):.5f}", flush=True)
    bundle = {k: np.asarray(v, np.float64) for k, v in params.items()}
    bundle.update(x_mean=x_mean, x_std=x_std,
                  y_mean=np.float64(y_mean), y_std=np.float64(y_std))
    train_mse = float(np.mean((predict_from_bundle(bundle, X) - Y) ** 2))
    print(f"[renewal/{tag}] done: train MSE={train_mse:.5f} "
          f"(target mean={y_mean:.3f} std={y_std:.3f})", flush=True)
    return bundle, train_mse


# ====================== fit + eval driver ======================

def load_dataset(dataset_path, n_transitions):
    import ogbench
    _env, ds, _val = ogbench.make_env_and_datasets(
        TASK_NAME, dataset_dir=str(Path(dataset_path).expanduser()))
    obs = np.asarray(ds["observations"], np.float32)
    nobs = np.asarray(ds["next_observations"], np.float32)
    acts = np.asarray(ds["actions"], np.float32)
    term = np.asarray(ds.get("terminals", np.zeros(len(obs)))).astype(float)
    if term.sum() == 0:      # ogbench single-play: fixed 1000-step episodes
        term = np.zeros(len(obs))
        term[999::1000] = 1.0
    n = min(int(n_transitions), len(obs))
    return obs[:n], nobs[:n], acts[:n], term[:n]


def print_fid_table(name, fid):
    print(f"[renewal/eval] {name}")
    print(f"  {'bucket':<10}{'n':>7}{'spearman':>11}{'slope':>10}")
    for b in ("overall", "K<=5", "5<K<=20", "K>20"):
        row = fid[b]
        sp = row.get("spearman")
        sl = row.get("calib_slope")
        print(f"  {b:<10}{row['n']:>7}"
              f"{(f'{sp:.3f}' if sp is not None and np.isfinite(sp) else '  --'):>11}"
              f"{(f'{sl:.3f}' if sl is not None and np.isfinite(sl) else '  --'):>10}",
              flush=True)


def run(args):
    import glob as _glob
    from eval_rgfb_zeroshot import load_run

    t0 = time.time()
    gamma, eps = args.gamma, args.epsilon
    rng = np.random.default_rng(args.seed)

    obs, nobs, acts, term = load_dataset(args.dataset_path, args.n_transitions)
    cube = obs[:, CUBE_OBS].astype(np.float64)          # OBS units throughout
    spans = split_episodes(term)
    holdout = spans[-args.holdout:]
    train_spans = spans[:-args.holdout]
    print(f"[renewal] {len(spans)} episodes -> {len(train_spans)} train / "
          f"{len(holdout)} holdout; eps={eps} obs units, gamma={gamma}", flush=True)

    need_agent = args.input == "psi" or args.eval
    agent, ckpt_file, hash_before = None, None, None
    if need_agent:
        run_dir = Path(sorted(_glob.glob(str(Path(args.run_dir).expanduser())))[0]).resolve()
        ckpt_file = run_dir / f"params_{args.checkpoint_step}.pkl"
        hash_before = file_sha256(ckpt_file) if ckpt_file.exists() else None
        agent, flags = load_run(run_dir, args.checkpoint_step)
        print(f"[renewal] restored {run_dir.name} "
              f"d={flags['agent']['value_latent_dim']} step={args.checkpoint_step}", flush=True)

    train_rows = np.concatenate([np.arange(a, b) for a, b in train_spans])

    # ---- 2a: renewal head ----
    bundle_R, meta_R = None, None
    if args.head_path:
        bundle_R, meta_R = _load_bundle(args.head_path)
        assert meta_R.get("kind", "renewal") == "renewal"
        print(f"[renewal] loaded head bundle {args.head_path} (input={meta_R['input']})",
              flush=True)
    else:
        g_rows = rng.choice(train_rows, size=min(args.max_goals, len(train_rows)),
                            replace=False)
        goal_obs = nobs[g_rows]
        goal_cube = goal_obs[:, CUBE_OBS].astype(np.float64)
        print(f"[renewal] computing MC renewal targets for {len(g_rows)} goals "
              f"over {len(train_spans)} train episodes ...", flush=True)
        tgts, nent = renewal_targets(cube, train_spans, goal_cube, gamma, eps)
        keep = nent > 0
        coverage = float(keep.mean())
        Y = tgts[keep]
        flat_r_diag = float(np.var(Y) / (np.mean(Y) ** 2 + 1e-12))
        print(f"[renewal] goal coverage={coverage:.3f} ({keep.sum()}/{len(keep)}); "
              f"entries/goal mean={nent[keep].mean():.1f}; "
              f"R_tgt mean={Y.mean():.3f} var={Y.var():.3f} "
              f"var/mean^2={flat_r_diag:.3f}  <- flat-R side-prediction", flush=True)
        if args.input == "psi":
            X = make_critic_psi(agent)(goal_obs[keep])
        else:
            X = goal_cube[keep]
        bundle_R, train_mse = fit_head(X, Y, args.iters, args.batch, args.lr,
                                       args.seed, tag=f"fit:{args.input}")
        meta_R = {
            "kind": "renewal", "input": args.input,
            "run_dir": str(Path(args.run_dir).expanduser()) if args.run_dir else "",
            "checkpoint_step": args.checkpoint_step,
            "epsilon": eps, "gamma": gamma, "obs_cube_slice": [19, 22],
            "n_goals": int(keep.sum()), "coverage": coverage,
            "entries_per_goal_mean": float(nent[keep].mean()),
            "target_mean": float(Y.mean()), "target_var": float(Y.var()),
            "flat_R_var_over_mean_sq": flat_r_diag, "train_mse": train_mse,
            "n_transitions": int(len(obs)), "holdout": args.holdout,
            "seed": args.seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spec": "specs/07-fp-capability-and-renewal-measure.md item 2a",
        }
        out = Path(args.out) if args.out else \
            REPO / "analysis" / "rescuing-fb" / f"renewal_head_{args.input}.npz"
        save_head_bundle(out, bundle_R, meta_R)
        print(f"[renewal] head bundle -> {out}", flush=True)

    # ---- 2b: FP_mu head (appendix object) ----
    bundle_fp, meta_fp = None, None
    if args.fit_fp_mu:
        elig = np.array([r for a, b in train_spans for r in range(a, b - 1)])
        q = rng.choice(elig, size=min(args.fp_pairs, len(elig)), replace=True)
        gp_rows = rng.choice(train_rows, size=len(q), replace=True)
        gp_cube = nobs[gp_rows][:, CUBE_OBS].astype(np.float64)
        print(f"[renewal] computing MC FP_mu targets for {len(q)} (s,a)xg pairs ...",
              flush=True)
        tg, cens = fp_targets(cube, train_spans, q, gp_cube, gamma, eps)
        censor_frac = float(cens.mean())
        print(f"[renewal] FP_mu censoring fraction={censor_frac:.3f} "
              f"(claims restricted to the uncensored-dominated region); "
              f"uncensored target mean={tg[~cens].mean() if (~cens).any() else float('nan'):.3f}",
              flush=True)
        Xf = np.concatenate([obs[q].astype(np.float64), acts[q].astype(np.float64),
                             gp_cube], axis=-1)
        bundle_fp, fp_mse = fit_head(Xf, tg, args.iters, args.batch, args.lr,
                                     args.seed + 1, tag="fit:fp_mu")
        meta_fp = {
            "kind": "fp_mu", "input": "raw_s_a_cubeg",
            "input_note": "raw [s, a, cube(g)] (37-dim): tower-free appendix object; "
                          "documented choice per spec 07 2b",
            "epsilon": eps, "gamma": gamma, "n_pairs": int(len(q)),
            "censor_fraction": censor_frac, "train_mse": fp_mse,
            "seed": args.seed + 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spec": "specs/07-fp-capability-and-renewal-measure.md item 2b",
        }
        base = Path(args.out) if args.out else \
            REPO / "analysis" / "rescuing-fb" / f"renewal_head_{args.input}.npz"
        out_fp = base.with_name(base.stem + "_fpmu.npz")
        save_head_bundle(out_fp, bundle_fp, meta_fp)
        print(f"[renewal] FP_mu bundle -> {out_fp}", flush=True)

    # ---- calibration eval (the deliverable figure) ----
    result = {"meta_renewal": meta_R, "meta_fp_mu": meta_fp}
    if args.eval:
        erng = np.random.default_rng(1)
        q_ep, q_row, tp_row, Ks, tgt = build_fidelity_grid(
            cube, holdout, erng, gamma, args.fid_queries,
            offsets=(1, 2, 4, 8, 16, 32, 64), eps=eps)
        goal_obs_e = obs[tp_row]
        if meta_R["input"] == "psi":
            Xe = make_critic_psi(agent)(goal_obs_e)
        else:
            Xe = cube[tp_row]
        Rg = np.maximum(predict_from_bundle(bundle_R, Xe), 0.0)
        qfp = make_qfp(agent)(obs[q_row], acts[q_row], goal_obs_e)   # sg[Q_FP]
        m_hat = qfp * Rg
        fid = fidelity_report(m_hat, tgt, Ks)
        fid_fp_only = fidelity_report(qfp, tgt, Ks)   # control: what R adds to ranking
        print_fid_table("reconstruction sg[Q_FP] x R vs MC occupancy (holdout)", fid)
        print_fid_table("control: sg[Q_FP] alone", fid_fp_only)
        result.update({
            "fidelity_fp_x_R": fid, "fidelity_fp_only": fid_fp_only,
            "R_pred_mean": float(Rg.mean()), "R_pred_var": float(Rg.var()),
            "R_pred_var_over_mean_sq": float(Rg.var() / (Rg.mean() ** 2 + 1e-12)),
            "fid_queries": args.fid_queries,
        })
        if bundle_fp is not None:
            Xf = np.concatenate([obs[q_row].astype(np.float64),
                                 acts[q_row].astype(np.float64),
                                 cube[tp_row]], axis=-1)
            fpmu = np.clip(predict_from_bundle(bundle_fp, Xf), 0.0, 1.0)
            fid_mu = fidelity_report(fpmu * Rg, tgt, Ks)
            print_fid_table("single-policy M^mu = FP_mu x R (appendix)", fid_mu)
            result["fidelity_fpmu_x_R"] = fid_mu

    if need_agent:
        hash_after = file_sha256(ckpt_file) if ckpt_file.exists() else None
        assert hash_before == hash_after, \
            "control checkpoint changed! (must be structurally frozen)"
        result["checkpoint_sha256"] = hash_after
        result["checkpoint_unchanged"] = True

    eval_out = Path(args.eval_out) if args.eval_out else \
        REPO / "analysis" / "rescuing-fb" / f"renewal_head_eval_{args.input}.json"
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.write_text(json.dumps(result, indent=2))
    print(f"[renewal] done in {time.time() - t0:.0f}s -> {eval_out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="",
                    help="gciql_flow run dir (glob ok); required for --input psi and --eval")
    ap.add_argument("--dataset-path", default="~/.ogbench/data",
                    help="ogbench dataset_dir containing cube-single-play-v0.npz")
    ap.add_argument("--input", choices=["psi", "cube"], default="psi",
                    help="renewal-head input: sg[psi(g)] (deployable) or cube(g) (control)")
    ap.add_argument("--epsilon", type=float, default=EPS_OBS,
                    help="ball radius in OBS units (0.4 = 4 cm)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max-goals", type=int, default=2000,
                    help="test goals sampled from dataset next-states")
    ap.add_argument("--out", default="", help="output .npz bundle path")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--head-path", default="",
                    help="load an existing renewal bundle instead of fitting")
    ap.add_argument("--fit-fp-mu", action="store_true",
                    help="also fit the single-policy FP_mu head (spec 07 2b)")
    ap.add_argument("--fp-pairs", type=int, default=20000)
    ap.add_argument("--eval", action="store_true",
                    help="calibration eval on held-out episodes (spec 07 2a deliverable)")
    ap.add_argument("--eval-out", default="", help="eval json path")
    ap.add_argument("--n-transitions", type=int, default=1_000_000)
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--fid-queries", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.input == "psi" or args.eval:
        assert args.run_dir, "--run-dir required for --input psi / --eval"
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
