#!/usr/bin/env python
"""scripts/rescuing_fb/measure_readout_head.py -- Spec 06 item 8 Phase 1.

A stop-grad measure readout head, fit POST-HOC on FROZEN control features, at
provably zero cost to control (nothing in the control towers is retrained; the
towers never enter the head's graph -- sg is structural). It recovers the
occupancy extractables (density queries, OPE, temporal distance) and produces the
honest fidelity number the phenomenon draft needs.

Two backends (the venvs are disjoint -- .venv has torch, /dev/shm/.venv-jax has
jax -- so each backend is its own process; `--mode compare` ingests both JSONs):

  --backend jax    RG-FB / bilinear-GCIQL run dir. Feature dump:
                     feat_phi = concat of the twin critic towers [phi_1(s,a); phi_2(s,a)]  (2*d)
                     feat_psi = psi(s')                                                      (d)
                   both stop-gradded structurally (numpy arrays; towers never in the graph).
                   Head:  M_hat(s,a,s+) = phi_M(feat_phi) . psi_M(feat_psi) / sqrt(d_M)
                          phi_M, psi_M = 2-layer 256 GELU MLPs, d_M=50.
                   --estimator mc (DEFAULT): within-trajectory discounted-visitation
                     contrastive regression (CRL/density-ratio; future offset k~Geom(1-gamma),
                     in-batch negatives). No TD, no scale question.
                   --estimator td_sarsa: batch measure TD, dataset-action bootstrap, with the
                     T10-b2 anchors REQUIRED (sphere-projected psi_M, ortho lambda=1, RAW base
                     leg) -- there is no scalar term to pin scale (phenomenon SS6 rule).

  --backend torch  Plain-FB checkpoint. Its native F.B IS its measure readout -- evaluated
                   directly, NO refit: M_native(s,a,s+) = <F(s,a, project_z(B(s+))), B(s+)>.

Fidelity (the deliverable): held-out episodes (last --holdout, excluded from head
training); empirical MC discounted-occupancy targets on a (s,a) x s+ grid; report
Spearman rank corr + calibration slope, broken down by horizon bucket K<=5 / 5-20
/ >20 (makes the misalignment bound visible in the calibration plot).
Extractable demos: (i) OPE of a random dense reward r(s')=sg[psi(s')].z~/sqrt(d)
vs MC ground truth; (ii) temporal distance T_hat=log_gamma(1+(1-gamma)Q) from the
CONTROL head directly (first-passage identity -- needs no measure head) vs realized
hitting times.

Pre-registered reads: control untouched (structural; the checkpoint hash is
asserted unchanged). Fidelity(RG-FB head) >= Fidelity(plain-FB native F.B) =>
"scalar grounding is measure-free but not measure-blind"; < => print the honest
trade (the head is kept either way for the extractables). No success-rate read
exists for this item by design.

Run:
  # RG-FB feature head (jax venv):
  JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python \\
    scripts/rescuing_fb/measure_readout_head.py --backend jax \\
    --run-dir '/home/adityamohan/exp_rgfb/factored-fb/RGFB-Cube-Single-State-d50-v0/rgfb_v0_s0' \\
    --checkpoint-step 500000 --estimator mc
  # plain-FB native (torch venv):
  PYTHONPATH=. .venv/bin/python scripts/rescuing_fb/measure_readout_head.py --backend torch \\
    --run-dir /dev/shm/factored-fb/runs/<fb_state_plain>__s0 --checkpoint final
  # compare:
  .venv/bin/python scripts/rescuing_fb/measure_readout_head.py --mode compare \\
    --a analysis/rescuing-fb/measure_head_rgfb_mc.json \\
    --b analysis/rescuing-fb/measure_head_plainfb.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CUBE_OBS = slice(19, 22)           # obs cube xyz = (m - CEN)*10  (jax obs frame)
CEN = np.array([0.425, 0.0, 0.0])
CUBE_PHYS = slice(14, 17)          # physics qpos cube xyz in meters (torch datasets)
EPS_M = 0.04                       # occupancy ball radius, meters (= t6 THRESH)


# ============================ shared numpy machinery ============================

def decode_cube_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs)[..., CUBE_OBS] / 10.0 + CEN


def split_episodes(terminals: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) row spans per episode from a 0/1 terminal flag."""
    ends = np.nonzero(np.asarray(terminals) > 0)[0]
    spans, s = [], 0
    for e in ends:
        spans.append((s, int(e) + 1))
        s = int(e) + 1
    if s < len(terminals):
        spans.append((s, len(terminals)))
    return spans


def build_fidelity_grid(cube, spans, rng, gamma, n_queries, offsets, eps=EPS_M):
    """MC discounted-occupancy targets. For each query (episode, t) and each test
    point s+ = s_{t+K} (K in `offsets`, clipped to the episode), the ground-truth
    discounted occupancy of the eps-ball around cube(s+) along the SAME trajectory:
        M*(s_t, s_{t+K}) = sum_{j>=0} gamma^j 1[||cube(s_{t+j}) - cube(s_{t+K})|| <= eps]
    Returns arrays (q_ep, q_row, tp_row, K, target) over all (query, offset) pairs
    that fit inside their episode. q_row/tp_row index into the global `cube` array."""
    q_ep, q_row, tp_row, Ks, tgt = [], [], [], [], []
    long_spans = [(a, b) for (a, b) in spans if b - a >= 8]
    if not long_spans:
        long_spans = spans
    for _ in range(n_queries):
        a, b = long_spans[rng.integers(len(long_spans))]
        L = b - a
        t = int(rng.integers(0, max(L - 1, 1)))
        for K in offsets:
            if t + K >= L:
                continue
            gp = a + t + K
            c_tp = cube[gp]
            fut = cube[a + t: b]                                   # cube from t to episode end
            d = np.linalg.norm(fut - c_tp, axis=-1) <= eps
            disc = gamma ** np.arange(len(fut))
            m_star = float((disc * d).sum())
            q_ep.append((a, b)); q_row.append(a + t); tp_row.append(gp)
            Ks.append(int(K)); tgt.append(m_star)
    return (q_ep, np.asarray(q_row), np.asarray(tp_row),
            np.asarray(Ks), np.asarray(tgt, np.float64))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def calibration_slope(m_hat, m_star):
    m_hat = np.asarray(m_hat, float); m_star = np.asarray(m_star, float)
    if len(m_hat) < 3 or np.std(m_hat) == 0:
        return float("nan")
    A = np.vstack([m_hat, np.ones_like(m_hat)]).T
    slope, _ = np.linalg.lstsq(A, m_star, rcond=None)[0]
    return float(slope)


def fidelity_report(m_hat, m_star, Ks):
    """Overall + per horizon-bucket Spearman / calibration slope."""
    buckets = {"K<=5": Ks <= 5, "5<K<=20": (Ks > 5) & (Ks <= 20), "K>20": Ks > 20}
    rep = {"overall": {"n": int(len(m_hat)), "spearman": spearman(m_hat, m_star),
                       "calib_slope": calibration_slope(m_hat, m_star)}}
    for name, mask in buckets.items():
        rep[name] = {"n": int(mask.sum()),
                     "spearman": spearman(m_hat[mask], m_star[mask]) if mask.sum() >= 3 else None,
                     "calib_slope": calibration_slope(m_hat[mask], m_star[mask]) if mask.sum() >= 3 else None}
    return rep


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================ jax backend (RG-FB head) ============================

def run_jax(args) -> dict:
    import jax
    import jax.numpy as jnp
    import optax
    import ogbench
    sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
    sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
    from gciql_flow import GCIQLFlowAgent, get_config
    from utils.flax_utils import restore_agent

    import glob
    run_dir = Path(sorted(glob.glob(str(Path(args.run_dir).expanduser())))[0]).resolve()
    flags = json.loads((run_dir / "flags.json").read_text())
    cfg = dict(get_config()); cfg.update(flags["agent"])
    cfg["encoder"] = flags["agent"].get("encoder"); cfg["frame_stack"] = flags["agent"].get("frame_stack")
    d = int(cfg["value_latent_dim"]); sqd = float(np.sqrt(d))

    env = ogbench.make_env_and_datasets("cube-single-play-singletask-task1-v0", env_only=True)
    ex_obs = jnp.asarray(env.reset()[0], jnp.float32)[None]
    ex_act = jnp.zeros((1, *env.action_space.shape), jnp.float32)
    if hasattr(env, "close"):
        env.close()
    ckpt_file = run_dir / f"params_{args.checkpoint_step}.pkl"
    hash_before = file_sha256(ckpt_file) if ckpt_file.exists() else None
    agent = GCIQLFlowAgent.create(int(flags["seed"]), ex_obs, ex_act, cfg)
    agent = restore_agent(agent, str(run_dir), args.checkpoint_step)
    print(f"[head/jax] restored {run_dir.name} d={d} step={args.checkpoint_step}", flush=True)

    # ---- dataset -> episodes ----
    _e, ds, _ = ogbench.make_env_and_datasets("cube-single-play-singletask-task1-v0")
    obs = np.asarray(ds["observations"], np.float32)
    nobs = np.asarray(ds["next_observations"], np.float32)
    acts = np.asarray(ds["actions"], np.float32)
    term = np.asarray(ds.get("terminals", np.zeros(len(obs)))).astype(float)
    if term.sum() == 0:  # ogbench single-play: fixed 1000-step episodes
        term = np.zeros(len(obs)); term[999::1000] = 1.0
    n_tr = min(args.n_transitions, len(obs))
    obs, nobs, acts, term = obs[:n_tr], nobs[:n_tr], acts[:n_tr], term[:n_tr]
    cube = decode_cube_obs(obs)
    spans = split_episodes(term)
    holdout = spans[-args.holdout:]
    train_spans = spans[:-args.holdout]
    print(f"[head/jax] {len(spans)} episodes, {len(train_spans)} train / {len(holdout)} holdout", flush=True)

    # ---- frozen feature dump: feat_phi=[phi1;phi2](s,a) [N,2d], feat_psi=psi(s') [N,d] ----
    def dump_features(o, a, no, chunk=8192):
        fp, fs = [], []
        for i in range(0, len(o), chunk):
            oo = jnp.asarray(o[i:i + chunk]); aa = jnp.asarray(a[i:i + chunk])
            nn = jnp.asarray(no[i:i + chunk])
            _, phi, _ = agent.network.select("critic")(oo, oo, aa, info=True)   # phi [2,B,d]
            _, _, psi = agent.network.select("value")(nn, nn, info=True)         # psi [B,d]
            fp.append(np.asarray(jnp.concatenate([phi[0], phi[1]], axis=-1)))    # [B,2d]
            fs.append(np.asarray(psi))
        return np.concatenate(fp, 0), np.concatenate(fs, 0)

    feat_phi, feat_psi_next = dump_features(obs, acts, nobs)
    # psi over ALL states (for test points / OPE): value.psi(s)
    def psi_states(o, chunk=8192):
        out = []
        for i in range(0, len(o), chunk):
            oo = jnp.asarray(o[i:i + chunk])
            _, _, ps = agent.network.select("value")(oo, oo, info=True)
            out.append(np.asarray(ps))
        return np.concatenate(out, 0)
    feat_psi_state = psi_states(obs)                                            # psi(s) [N,d]

    # ---- head modules (flax) ----
    import flax.linen as fnn

    class MLPHead(fnn.Module):
        out: int
        @fnn.compact
        def __call__(self, x):
            x = fnn.gelu(fnn.Dense(256)(x)); x = fnn.gelu(fnn.Dense(256)(x))
            return fnn.Dense(self.out)(x)

    dM = args.d_head
    phi_M = MLPHead(dM); psi_M = MLPHead(dM)
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    p_phi = phi_M.init(k1, jnp.zeros((1, feat_phi.shape[1])))
    p_psi = psi_M.init(k2, jnp.zeros((1, feat_psi_state.shape[1])))
    params = {"phi": p_phi, "psi": p_psi}
    gamma = args.gamma
    anchors = args.estimator == "td_sarsa"

    def m_hat(params, fp, gp, sphere=False):
        u = phi_M.apply(params["phi"], fp)                                     # [B,dM]
        v = psi_M.apply(params["psi"], gp)                                     # [B,dM]
        if sphere:
            v = np.sqrt(dM) * v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        return u, v

    # ---- training samplers over TRAIN episodes ----
    train_rows = np.concatenate([np.arange(a, b) for a, b in train_spans])
    row_ep = {}
    for a, b in train_spans:
        for r in range(a, b):
            row_ep[r] = (a, b)

    def sample_future_pairs(bs, rng_np):
        """(query rows, future rows) with future offset k~Geom(1-gamma) inside episode."""
        qi = rng_np.choice(train_rows, size=bs)
        fr = np.empty(bs, int)
        for n, r in enumerate(qi):
            a, b = row_ep[int(r)]
            k = rng_np.geometric(1 - gamma)
            fr[n] = min(int(r) + k, b - 1)
        return qi, fr

    tx = optax.adam(args.lr)
    opt = tx.init(params)
    rng_np = np.random.default_rng(0)

    if args.estimator == "mc":
        # CRL contrastive: logits[i,j]=phi_M(feat_phi_i).psi_M(psi(future_j))/sqrt(dM);
        # positive = diagonal (i's own geometric future). Softmax-CE with in-batch negs.
        def loss_fn(params, fp, gp):
            u, v = m_hat(params, fp, gp)
            logits = (u @ v.T) / jnp.sqrt(dM)
            labels = jnp.arange(u.shape[0])
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        @jax.jit
        def step(params, opt, fp, gp):
            l, g = jax.value_and_grad(loss_fn)(params, fp, gp)
            upd, opt = tx.update(g, opt)
            return optax.apply_updates(params, upd), opt, l

        for it in range(args.iters):
            qi, fr = sample_future_pairs(args.batch, rng_np)
            fp = jnp.asarray(feat_phi[qi]); gp = jnp.asarray(feat_psi_state[fr])
            params, opt, l = step(params, opt, fp, gp)
            if it % 500 == 0:
                print(f"[head/jax][mc] it={it} ce={float(l):.4f}", flush=True)
    else:
        # td_sarsa: batch measure TD on the head, dataset-action bootstrap, T10-b2 anchors.
        # M(s,a,s+) = phi_M([phi1;phi2](s,a)) . sphere[psi_M(psi(s+))] / sqrt(dM).
        # Test points s+ = the batch's next-states (rho). Target uses the SAME head on
        # (s', a'=dataset next action) via the frozen next-feature dump. RAW base leg =
        # no scale-norm on the Dirac; ortho lambda=1 on the sphered psi_M.
        # Precompute frozen next-features [phi1;phi2](s', a') for SARSA bootstrap.
        nn_rows = train_rows
        # next-state's dataset next action: approximate a' by the action at row+1 in-episode.
        def next_action_row(r):
            a, b = row_ep[int(r)]
            return min(int(r) + 1, b - 1)
        feat_phi_next = None  # built lazily per batch to save memory

        def loss_fn(params, fp, gp, fp_next, gp_all):
            u, v = m_hat(params, fp, gp, sphere=True)                          # phi_M(s,a), sphere psi_M(s+)
            M = (u @ v.T) / jnp.sqrt(dM)                                       # [B,B]
            un, _ = m_hat(params, fp_next, gp, sphere=True)
            tM = jax.lax.stop_gradient((un @ v.T) / jnp.sqrt(dM))
            B = M.shape[0]; off = 1.0 - jnp.eye(B)
            diff = M - gamma * tM
            offdiag = 0.5 * jnp.sum((diff * off) ** 2) / jnp.sum(off)          # FB off-diag TD
            diagt = -jnp.mean(jnp.diagonal(diff))                             # Dirac attraction (RAW)
            # ortho lambda=1 on sphered psi_M rows (v): ||E[vv^T]-I||
            Cov = v @ v.T
            orth = 0.5 * jnp.sum((Cov * off) ** 2) / jnp.sum(off) - jnp.mean(jnp.diag(Cov))
            return offdiag + diagt + 1.0 * orth

        @jax.jit
        def step(params, opt, fp, gp, fp_next):
            l, g = jax.value_and_grad(lambda p: loss_fn(p, fp, gp, fp_next, None))(params)
            upd, opt = tx.update(g, opt)
            return optax.apply_updates(params, upd), opt, l

        for it in range(args.iters):
            qi = rng_np.choice(nn_rows, size=args.batch)
            na = np.array([next_action_row(r) for r in qi])
            fp = jnp.asarray(feat_phi[qi])
            gp = jnp.asarray(feat_psi_next[qi])              # psi(s') = test points (rho)
            fp_next = jnp.asarray(feat_phi[na])              # [phi;phi](s', a'_data)
            params, opt, l = step(params, opt, fp, gp, fp_next)
            if it % 500 == 0:
                print(f"[head/jax][td] it={it} loss={float(l):.4f}", flush=True)

    # ---- fidelity on HOLDOUT ----
    rng = np.random.default_rng(1)
    q_ep, q_row, tp_row, Ks, tgt = build_fidelity_grid(
        cube, holdout, rng, gamma, args.fid_queries, offsets=(1, 2, 4, 8, 16, 32, 64))
    u, v = m_hat(params, jnp.asarray(feat_phi[q_row]),
                 jnp.asarray(feat_psi_state[tp_row]), sphere=anchors)
    M_pred = np.asarray((u * v).sum(-1)) / np.sqrt(dM)
    fid = fidelity_report(M_pred, tgt, Ks)
    print(f"[head/jax] fidelity overall spearman={fid['overall']['spearman']:.3f} "
          f"slope={fid['overall']['calib_slope']:.3f}", flush=True)

    # ---- extractable (i): OPE of random dense reward r(s')=sg[psi(s')].z~/sqrt(d) ----
    z_tilde = rng.normal(size=d); z_tilde *= np.sqrt(d) / (np.linalg.norm(z_tilde) + 1e-8)
    r_all = (feat_psi_state @ z_tilde) / sqd                                    # r(s) over all rows
    ope_pred, ope_star = [], []
    long_ho = [(a, b) for a, b in holdout if b - a >= 8] or holdout
    for _ in range(args.fid_queries):
        a, b = long_ho[rng.integers(len(long_ho))]
        t = int(rng.integers(0, max(b - a - 1, 1)))
        tail = np.arange(a + t, b)
        disc = gamma ** np.arange(len(tail))
        v_star = float((disc * r_all[tail]).sum())
        # OPE via the head measure: V_hat(s,a)=sum_s+ M_hat(s,a,s+) r(s+) over rho test points
        tp = rng.choice(train_rows, size=min(256, len(train_rows)), replace=False)
        uu, vv = m_hat(params, jnp.asarray(feat_phi[a + t][None]),
                       jnp.asarray(feat_psi_state[tp]), sphere=anchors)
        m_row = np.asarray((uu @ vv.T)[0]) / np.sqrt(dM)
        v_hat = float((m_row * r_all[tp]).mean() / (1 - gamma))
        ope_pred.append(v_hat); ope_star.append(v_star)
    ope = {"spearman": spearman(ope_pred, ope_star),
           "calib_slope": calibration_slope(ope_pred, ope_star), "n": len(ope_pred)}

    # ---- extractable (ii): temporal distance T_hat from the CONTROL head ----
    # Q(s,a,z=psi(g)) = phi(s,a).psi(g)/sqrt(d); T_hat = log_gamma(1+(1-gamma)Q); vs realized K.
    td_pred, td_real = [], []
    for _ in range(args.fid_queries):
        a, b = long_ho[rng.integers(len(long_ho))]
        L = b - a; t = int(rng.integers(0, max(L - 1, 1)))
        K = int(rng.integers(1, L - t))
        g_row = a + t + K
        # scalar control Q at z=psi(goal): min over BOTH critic heads (frozen features).
        u2 = feat_phi[a + t].reshape(2, d)
        q = np.minimum((u2[0] * feat_psi_state[g_row]).sum(),
                       (u2[1] * feat_psi_state[g_row]).sum()) / sqd
        arg = 1.0 + (1.0 - gamma) * q
        if arg <= 0:
            continue
        t_hat = np.log(arg) / np.log(gamma)
        td_pred.append(float(t_hat)); td_real.append(K)
    tdist = {"spearman": spearman(td_pred, td_real), "n": len(td_pred)}

    hash_after = file_sha256(ckpt_file) if ckpt_file.exists() else None
    assert hash_before == hash_after, "control checkpoint changed! (must be structurally frozen)"

    return {
        "backend": "jax", "estimator": args.estimator, "run_dir": str(run_dir),
        "d": d, "d_head": dM, "gamma": gamma, "iters": args.iters,
        "checkpoint_sha256": hash_after, "checkpoint_unchanged": True,
        "fidelity": fid, "ope": ope, "temporal_distance": tdist,
        "spec": "specs/06-phenomenon-implementations.md item 8 Phase 1",
    }


# ============================ torch backend (plain-FB native) ============================

def run_torch(args) -> dict:
    import torch
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from data.ogbench import load_ogbench_dataset

    run_dir = Path(args.run_dir).expanduser()
    cfg = load_cfg(str(run_dir / ".hydra/config.yaml"), device="cpu")
    _env, agent = build_env_and_agent(cfg, device="cpu")
    ckpt = run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    hash_before = file_sha256(ckpt)
    load_checkpoint(agent, str(ckpt))
    model = agent.model

    data_path = (str(REPO / "datasets") if (REPO / "datasets" / "cube-single-play-v0").exists()
                 else "/dev/shm/factored-fb/datasets")
    buf = load_ogbench_dataset(domain="cube-single-play-v0", data_path=data_path,
                               obs_type="state", load_n_episodes=args.holdout + 200)
    # Reconstruct arrays from the DictBuffer storage (attribute name varies by version).
    store = next((getattr(buf, a) for a in ("_storage", "storage", "_data", "data")
                  if hasattr(buf, a)), None)
    assert store is not None and "observation" in store, \
        f"could not read DictBuffer storage; attrs={[a for a in dir(buf) if not a.startswith('__')]}"
    obs = np.asarray(store["observation"])
    acts = np.asarray(store["action"])
    phys = np.asarray(store["physics"]) if "physics" in store else None
    disc = np.asarray(store.get("discount", np.ones(len(obs))))
    term = (disc <= 0).astype(float)
    if term.sum() == 0:
        term = np.zeros(len(obs)); term[999::1000] = 1.0
    cube = phys[:, CUBE_PHYS] if phys is not None else decode_cube_obs(obs)
    spans = split_episodes(term)
    holdout = spans[-args.holdout:]
    print(f"[head/torch] plain-FB {run_dir.name}: {len(spans)} eps, {len(holdout)} holdout", flush=True)

    @torch.no_grad()
    def m_native(s_rows, sp_rows):
        s = torch.as_tensor(obs[s_rows], dtype=torch.float32)
        a = torch.as_tensor(acts[s_rows], dtype=torch.float32)
        sp = torch.as_tensor(obs[sp_rows], dtype=torch.float32)
        z = model.project_z(model.backward_map(sp))                       # z = B(s+)
        F = model.forward_map(s, z, a)                                    # F(s,a,z)
        return (F * z).sum(-1).cpu().numpy()                             # <F, B(s+)> (scale-free; Spearman-invariant)

    rng = np.random.default_rng(1)
    q_ep, q_row, tp_row, Ks, tgt = build_fidelity_grid(
        cube, holdout, rng, args.gamma, args.fid_queries, offsets=(1, 2, 4, 8, 16, 32, 64))
    M_pred = m_native(q_row, tp_row)
    fid = fidelity_report(M_pred, tgt, Ks)
    print(f"[head/torch] fidelity overall spearman={fid['overall']['spearman']:.3f} "
          f"slope={fid['overall']['calib_slope']:.3f}", flush=True)

    hash_after = file_sha256(ckpt)
    assert hash_before == hash_after, "control checkpoint changed!"
    return {
        "backend": "torch", "estimator": "native_FB", "run_dir": str(run_dir),
        "gamma": args.gamma, "checkpoint_sha256": hash_after, "checkpoint_unchanged": True,
        "fidelity": fid,
        "note": "M_native = <F(s,a,project_z(B(s+))), B(s+)>; scale-free (Spearman is the headline).",
        "spec": "specs/06-phenomenon-implementations.md item 8 Phase 1 (comparison arm)",
    }


def run_compare(args) -> dict:
    a = json.loads(Path(args.a).read_text()); b = json.loads(Path(args.b).read_text())
    sa = a["fidelity"]["overall"]["spearman"]; sb = b["fidelity"]["overall"]["spearman"]
    verdict = ("scalar grounding is measure-free but not measure-blind"
               if sa >= sb else
               "honest trade: plain-FB native F.B ranks occupancy better; head kept for the extractables")
    print(f"[compare] A({a['backend']}/{a.get('estimator')}) spearman={sa:.3f}  "
          f"B({b['backend']}/{b.get('estimator')}) spearman={sb:.3f}  -> {verdict}", flush=True)
    return {"a": args.a, "b": args.b, "spearman_a": sa, "spearman_b": sb,
            "verdict_preregistered": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fit", "compare"], default="fit")
    ap.add_argument("--backend", choices=["jax", "torch"], default="jax")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--checkpoint-step", type=int, default=500000)
    ap.add_argument("--checkpoint", default="final", help="torch: checkpoint stem under checkpoints/")
    ap.add_argument("--estimator", choices=["mc", "td_sarsa"], default="mc")
    ap.add_argument("--n-transitions", type=int, default=1_000_000)
    ap.add_argument("--holdout", type=int, default=100, help="held-out episodes for fidelity")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--d-head", type=int, default=50)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fid-queries", type=int, default=2000)
    ap.add_argument("--a", default=""); ap.add_argument("--b", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    if args.mode == "compare":
        res = run_compare(args)
        tag = "compare"
    elif args.backend == "jax":
        res = run_jax(args); tag = f"rgfb_{args.estimator}"
    else:
        res = run_torch(args); tag = "plainfb"
    out = Path(args.out) if args.out else REPO / "analysis" / "rescuing-fb" / f"measure_head_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[head] done in {time.time()-t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
