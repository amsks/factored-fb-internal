"""Spec 25 -- porting FB's three index-coupling mechanisms to the z-indexed density
measure. Contract: `PAPER/Multistep-FB/specs/25-z-coupling-port.md` sections 0, 2, 4.

Delta on spec 24's `measure_density_z` (tests/test_gciql_measure_density_z.py). Four
knobs, all under the `measure_density_z` lattice, all default-off and bit-identical to
the spec-24 arm:

  C1 `measure_density_z_tilt` (beta)   structural logit tilt, applied to the ONLINE
        l_ij = F(s_i,a_i,z_i)^T B(u_j)/tau      logits, the TARGET logits and the
             + beta * z_i^T B(u_j)/sqrt(d)      actor's q-path logits (same
                                                exponential family on all three).
  C2 `measure_density_z_rdecode_coef` reward-decodability auxiliary: one kernel reward
        r_c(u) = exp(-0.5||(u-c)/eps||^2) at a random batch query center, closed-form
        ridge v* = (B^T B + 1e-4 I)^-1 B^T r_c through B, L_dec = ||B v* - r_c||^2/n.
  C3 `measure_density_z_bc_hindsight`  mask the imitation (distillation) term to
        hindsight-z rows, renormalized by the surviving count.
     `measure_density_z_enc_tau`       slow-EMA index encoder `density_b_zenc`.
  INSTRUMENT `density_w_ztv` = E[TV(w(s,a,z), w(s,a,z'))], logged UNCONDITIONALLY in
        z mode -- the pre-registered gate of the whole spec.

The suite deliberately drives the REAL entrypoints -- `update()` and `sample_actions()`
-- and not just `create()`: spec 24's own launch bug (a z build whose `sample_actions`
had no z branch at all) passed eight construct-only unit tests.

Same subprocess-jax harness as tests/test_gciql_measure_density_z.py: OBS_D=40,
action dim 5, d=8, hit_spec '19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_density_z_coupling.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import subprocess
import textwrap

import numpy as np
import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8    # value_latent_dim AND measure_density_dim, matches the spec-18/24 harness
ACT_D = 5

# The spec-24 IMPLEMENTATION commit ("feat(spec24): z-indexed Density-FB, with the
# eval-path width fix"). It has measure_density_z and none of the spec-25 knobs, so it
# permanently marks the arm this spec must reproduce bit-for-bit at defaults -- pinned,
# not HEAD, because HEAD moves forward on this shared branch as other sessions land
# commits (see tests/test_gciql_measure_density.py's DENSITY_PRE_SPEC18_SHA).
SPEC24_SHA = "67fc08e8b4bda5d1d8a78d564e6fa47d97d507a9"


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                       cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0


COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import (GCIQLFlowAgent, get_config, _parse_hit_spec, _sphere,
                            _Z_RDECODE_RIDGE_LAMBDA, _Z_ZTV_FOLD, _Z_RDECODE_FOLD)

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def base_cfg(**kw):
        # spec 24's base cell (the arm spec 25 is a delta on): measure-only, bilinear,
        # density, Z ON, every spec-25 knob at its default.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["actor_q_center"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg["measure_density_z"] = True
        cfg.update(kw)
        return cfg

    def full_stack_cfg(**kw):
        # C1 + C2 + C3 at their round-1 doses (spec 25 §0/§3 `z25_*_full`).
        return base_cfg(measure_density_z_tilt=1.0,
                        measure_density_z_rdecode_coef=1.0,
                        measure_density_z_bc_hindsight=True,
                        measure_density_z_enc_tau=0.0005, **kw)

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        return dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, ACT_D)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def jb(b):
        return {{k: jnp.asarray(v) for k, v in b.items()}}

    def z_ref(agent, goal_obs, blocks, d, rand_frac, rng, enc_module='target_density_b'):
        # direct-module reference for _density_z (the full mix), SAME rng-split order as
        # the agent's own method. enc_module names the tower the index is encoded with
        # (spec 25 C3: 'density_b_zenc' when measure_density_z_enc_tau > 0).
        x_sc = GCIQLFlowAgent._density_queries(goal_obs, blocks)
        z_g = _sphere(agent.network.select(enc_module)(x_sc), float(d))
        rng_sphere, rng_mix = jax.random.split(rng)
        B = goal_obs.shape[0]
        z_rand = _sphere(jax.random.normal(rng_sphere, (B, d)), float(d))
        use_rand = jax.random.bernoulli(rng_mix, rand_frac, (B, 1))
        return jnp.where(use_rand, z_rand, z_g), (1.0 - use_rand[:, 0].astype(jnp.float32))

    def tilt_ref(beta, z, b_emb, d):
        # spec 25 C1: beta * z^T B(u) / sqrt(d) -> [B, M]
        return beta * np.einsum('bd,md->bm', np.asarray(z), np.asarray(b_emb)) / np.sqrt(d)

    def rdecode_ref(u_sc, b_emb, rng):
        # spec 25 C2 numpy reference of the closed-form ridge.
        u_sc = np.asarray(u_sc); b_emb = np.asarray(b_emb)
        M, d = u_sc.shape[0], b_emb.shape[-1]
        c_idx = int(jax.random.randint(rng, (), 0, M))
        c = u_sc[c_idx]
        r_c = np.exp(-0.5 * np.sum((u_sc - c[None]) ** 2, axis=-1))
        gram = b_emb.T @ b_emb + _Z_RDECODE_RIDGE_LAMBDA * np.eye(d)
        v = np.linalg.solve(gram, b_emb.T @ r_c)
        resid = b_emb @ v - r_c
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((r_c - r_c.mean()) ** 2))
        return sse / M, float(np.clip(1.0 - sse / (sst + 1e-8), 0.0, 1.0)), r_c

    def density_ref(agent, cfg, bj, rng, hard=False):
        # direct-module reference for density_loss WITH the spec-25 tilt on both towers
        # and the spec-25 rdecode auxiliary. Same rng discipline as the implementation:
        # rng1, z_rng = split(incoming); the two spec-25 streams are fold_in-DERIVED
        # from z_rng, so they consume nothing.
        obs, nobs = bj['observations'], bj['next_observations']
        acts, g = bj['actions'], bj['value_goals']
        gamma, tau = cfg['discount'], cfg['measure_density_temp']
        d = cfg['measure_density_dim']
        beta = float(cfg.get('measure_density_z_tilt', 0.0))
        dec_coef = float(cfg.get('measure_density_z_rdecode_coef', 0.0))
        enc_mod = ('density_b_zenc'
                   if float(cfg.get('measure_density_z_enc_tau', 0.0)) > 0
                   else 'target_density_b')
        blocks = _parse_hit_spec(cfg['measure_hit_spec'])
        rand_frac = cfg.get('measure_density_z_rand_frac', 0.5)

        rng1, z_rng = jax.random.split(rng)
        z, _ = z_ref(agent, g, blocks, d, rand_frac, z_rng, enc_mod)

        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)
        b_emb = agent.network.select('density_b')(u_sc)
        f_emb = agent.network.select('density_f')(jnp.concatenate([obs, acts, z], -1))
        logits = np.asarray(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau)
        logits = logits + tilt_ref(beta, z, b_emb, d)[None]
        logp = np.asarray(jax.nn.log_softmax(jnp.asarray(logits), -1))
        w = np.asarray(jax.nn.softmax(jnp.asarray(logits), -1))

        noise = jax.random.normal(rng1, acts.shape)
        a_plus = jax.lax.stop_gradient(agent.network.select('actor')(nobs, z, noise))
        tb = agent.network.select('target_density_b')(u_sc)
        tf = agent.network.select('target_density_f')(jnp.concatenate([nobs, a_plus, z], -1))
        tgt_logits = np.asarray(jnp.einsum('hbd,md->hbm', tf, tb) / tau)
        tgt_logits = tgt_logits + tilt_ref(beta, z, tb, d)[None]
        w_bar = np.asarray(jax.nn.softmax(jnp.asarray(tgt_logits), -1))

        k = (np.eye(nobs.shape[0]) if hard
             else np.asarray(GCIQLFlowAgent._delta_phi(nobs, u_raw, blocks)))
        t = (1.0 - gamma) * k[None] + gamma * w_bar
        ce = -np.mean(np.sum(t * logp, axis=-1))

        # density_w_ztv: z' resampled off a fold_in-derived key.
        z_alt, _ = z_ref(agent, g, blocks, d, rand_frac,
                         jax.random.fold_in(z_rng, _Z_ZTV_FOLD), enc_mod)
        f_alt = agent.network.select('density_f')(jnp.concatenate([obs, acts, z_alt], -1))
        la = np.asarray(jnp.einsum('hbd,md->hbm', f_alt, b_emb) / tau)
        la = la + tilt_ref(beta, z_alt, b_emb, d)[None]
        w_alt = np.asarray(jax.nn.softmax(jnp.asarray(la), -1))
        ztv = 0.5 * float(np.mean(np.sum(np.abs(w - w_alt), axis=-1)))

        loss = ce
        dec = None
        if dec_coef > 0:
            dec_loss, dec_r2, _ = rdecode_ref(
                u_sc, b_emb, jax.random.fold_in(z_rng, _Z_RDECODE_FOLD))
            loss = ce + dec_coef * dec_loss
            dec = (dec_loss, dec_r2)
        return dict(ce=ce, loss=loss, t=t, w=w, ztv=ztv, dec=dec, z=np.asarray(z))

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


# --------------------------------------------------------------------------
# 1. all-defaults bit-identity to the spec-24 arm (param tree + loss)
# --------------------------------------------------------------------------
def test_default_off_bit_identical_to_spec24():
    """Every spec-25 knob at its default, measure_density_z ON: the param tree AND
    total_loss must be bit-identical to the spec-24 arm loaded from the PINNED commit
    SPEC24_SHA. This is the load-bearing regression guard for the whole spec -- an
    accidentally-live tilt, a consumed (rather than fold_in-DERIVED) rng stream for the
    always-on density_w_ztv instrument, a stray registered module, or a changed default
    would all show up here because the two sides run genuinely different code.

    Also checks the z-OFF (plain spec-18 density) and no-knob configs, so the spec-25
    edits cannot have leaked into the g-indexed paths (spec 25 §6 anti-scope: "no
    changes to the g-indexed paths (bit-identity is a test)")."""
    if not _baseline_reachable(SPEC24_SHA):
        pytest.skip(f"pinned spec-24 baseline {SPEC24_SHA} is not reachable in this "
                    "checkout (e.g. a shallow clone) -- cannot run the bit-identity diff")
    _run(f"""
    SPEC24_SHA = {SPEC24_SHA!r}
    REPO_DIR = {str(REPO)!r}
    """ + """
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", SPEC24_SHA + ":tools/wandb_mode_shim/gciql_flow.py"],
        cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    assert "measure_density_z" in baseline_src, \\
        "pinned baseline predates spec 24 entirely -- re-pin SPEC24_SHA"
    for k25 in ("measure_density_z_tilt", "measure_density_z_rdecode_coef",
                "measure_density_z_bc_hindsight", "measure_density_z_enc_tau",
                "density_w_ztv"):
        assert k25 not in baseline_src, (
            "pinned baseline already contains " + k25 + " -- the pin has gone stale; "
            "re-pin SPEC24_SHA to the spec-24 implementation commit")
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_spec24.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_spec24", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def cfg_of(get_cfg_fn, **kw):
        cfg = dict(get_cfg_fn())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["actor_q_center"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg.update(kw)
        return cfg

    bj = jb(make_batch())

    def same(a_base, a_new, label):
        t_base = jax.tree_util.tree_leaves(a_base.network.params)
        t_new = jax.tree_util.tree_leaves(a_new.network.params)
        assert len(t_base) == len(t_new), (label, len(t_base), len(t_new))
        assert all((x == y).all() for x, y in zip(t_base, t_new)), label + ": param tree"
        assert (set(a_base.network.params.keys()) == set(a_new.network.params.keys())), \\
            (label, sorted(a_base.network.params), sorted(a_new.network.params))
        l_b, _ = a_base.total_loss(bj, a_base.network.params, jax.random.PRNGKey(7))
        l_n, _ = a_new.total_loss(bj, a_new.network.params, jax.random.PRNGKey(7))
        assert float(l_b) == float(l_n), (label, float(l_b), float(l_n))

    # (a) THE case: measure_density_z ON, every spec-25 knob at its default.
    same(baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                        cfg_of(baseline.get_config, measure_density_z=True)),
         GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                               cfg_of(get_config, measure_density_z=True)),
         "spec-24 z arm at spec-25 defaults")

    # (b) the g-indexed density arm (spec 18/19a) -- must be untouched.
    same(baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_of(baseline.get_config)),
         GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_of(get_config)),
         "g-indexed density arm")

    # (c) no measure knobs at all.
    cnb = dict(baseline.get_config()); cnb["encoder"] = None; cnb["frame_stack"] = None
    cnn = dict(get_config()); cnn["encoder"] = None; cnn["frame_stack"] = None
    same(baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cnb),
         GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cnn), "no-knob config")

    # and a real UPDATE step on the z arm must agree too (the loss check above is a
    # single trace; this exercises apply_gradients + the target soft-updates).
    a_b = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                         cfg_of(baseline.get_config, measure_density_z=True))
    a_n = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_of(get_config, measure_density_z=True))
    for step in range(3):
        bb = jb(make_batch(seed=step))
        a_b, i_b = a_b.update(bb)
        a_n, i_n = a_n.update(bb)
        assert float(i_b['measure/density_ce']) == float(i_n['measure/density_ce']), step
    lb = jax.tree_util.tree_leaves(a_b.network.params)
    ln = jax.tree_util.tree_leaves(a_n.network.params)
    assert all((x == y).all() for x, y in zip(lb, ln)), \\
        "param tree diverged from the spec-24 arm after 3 real update() steps"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. tilt numpy reference (both towers + actor path; rows on simplex)
# --------------------------------------------------------------------------
def test_tilt_matches_reference_on_all_three_paths():
    """C1's tilt is applied in the ONLINE logits, the TARGET logits and the actor's
    q-path logits. Each is checked against an independent direct-module/numpy
    reconstruction that adds beta*z^T B(u)/sqrt(d) with the tower that produced those
    logits (online density_b online-side, target_density_b target-side).

    The target leg is the one that burns programs: if it were left untilted, the CE
    would fit the tilted family while the bootstrap target lived in the untilted one.
    The reference builds `t` from the TILTED target, so a missing target tilt fails
    here even though the online tilt is present. Rows of `t` stay on the simplex."""
    _run("""
    for beta in (1.0, 2.5):
        cfg = base_cfg(measure_density_z_tilt=beta)
        a = agent_for(cfg)
        bj = jb(make_batch(B=6, seed=11))
        rng = jax.random.PRNGKey(7)

        loss, info = a.density_loss(bj, a.network.params, rng)
        ref = density_ref(a, cfg, bj, rng)
        assert np.allclose(ref['loss'], float(loss), atol=1e-5), (beta, ref['loss'], float(loss))
        assert np.allclose(ref['t'].sum(-1), 1.0, atol=1e-5), "target rows off the simplex"
        assert np.allclose(ref['w'].sum(-1), 1.0, atol=1e-5)
        assert np.allclose(ref['ztv'], float(info['density_w_ztv']), atol=1e-5)

        # a beta=0 sibling at the SAME params must give a DIFFERENT loss -- otherwise
        # the "reference matches" above would be vacuous.
        a0 = agent_for(base_cfg(measure_density_z_tilt=0.0))
        loss0, _ = a0.density_loss(bj, a0.network.params, rng)
        assert abs(float(loss0) - float(loss)) > 1e-6, (beta, float(loss0), float(loss))

        # TARGET-leg isolation: reconstruct the CE with the target tilt DELETED (the
        # bug this test exists to catch) and confirm it does NOT match the shipped loss.
        obs, nobs = bj['observations'], bj['next_observations']
        acts, g = bj['actions'], bj['value_goals']
        tau = cfg['measure_density_temp']; gamma = cfg['discount']
        blocks = _parse_hit_spec(cfg['measure_hit_spec'])
        rng1, z_rng = jax.random.split(rng)
        z, _ = z_ref(a, g, blocks, D, cfg['measure_density_z_rand_frac'], z_rng)
        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)
        b_emb = a.network.select('density_b')(u_sc)
        f_emb = a.network.select('density_f')(jnp.concatenate([obs, acts, z], -1))
        lg = np.asarray(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau) \\
            + tilt_ref(beta, z, b_emb, D)[None]
        logp = np.asarray(jax.nn.log_softmax(jnp.asarray(lg), -1))
        noise = jax.random.normal(rng1, acts.shape)
        a_plus = a.network.select('actor')(nobs, z, noise)
        tb = a.network.select('target_density_b')(u_sc)
        tf = a.network.select('target_density_f')(jnp.concatenate([nobs, a_plus, z], -1))
        wbar_untilted = np.asarray(jax.nn.softmax(
            jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1))     # THE MISTAKE
        k = np.asarray(GCIQLFlowAgent._delta_phi(nobs, u_raw, blocks))
        t_bad = (1.0 - gamma) * k[None] + gamma * wbar_untilted
        ce_bad = -np.mean(np.sum(t_bad * logp, axis=-1))
        assert abs(ce_bad - float(loss)) > 1e-6, \\
            (beta, "an UNTILTED target reproduces the shipped loss -- the target leg is "
                   "not actually tilted")

        # actor q path: q = sum_j w_j(s, pi(s,z), z) * (Btilde(u_j)^T z)/sqrt(d) with the
        # SAME tilt inside w.
        ba = jb(make_batch(B=5, seed=6))
        arng = jax.random.PRNGKey(4)
        _, ainfo = a.actor_loss(ba, a.network.params, arng)
        obs_a, goals_a, acts_a = ba['observations'], ba['actor_goals'], ba['actions']
        _, x0_rng, t_rng, n_rng, az_rng = jax.random.split(arng, 5)
        za, _ = z_ref(a, goals_a, blocks, D, cfg['measure_density_z_rand_frac'], az_rng)
        an = jax.random.normal(n_rng, acts_a.shape)
        aa = a.network.select('actor')(obs_a, za, an, params=a.network.params)
        u_sc_a = GCIQLFlowAgent._density_queries(ba['next_observations'], blocks)
        b_a = a.network.select('density_b')(u_sc_a)
        f_a = a.network.select('density_f')(jnp.concatenate([obs_a, aa, za], -1))
        lq = np.asarray(jnp.einsum('hbd,md->hbm', f_a, b_a) / tau) \\
            + tilt_ref(beta, za, b_a, D)[None]
        wq = jax.nn.softmax(jnp.asarray(lq), -1)
        tb_a = a.network.select('target_density_b')(u_sc_a)
        rz = jnp.einsum('md,bd->bm', tb_a, za) / jnp.sqrt(D)
        q1h, q2h = (wq * rz[None]).sum(-1)
        qh = jnp.minimum(q1h, q2h) if cfg['double_q'] else q1h
        assert np.allclose(float(jnp.mean(qh)), float(ainfo['q_mean']), atol=1e-4), \\
            (beta, float(jnp.mean(qh)), float(ainfo['q_mean']))
        # and the untilted actor q must NOT match (the actor really reads the tilt).
        wq0 = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_a, b_a) / tau, -1)
        q1b, q2b = (wq0 * rz[None]).sum(-1)
        qb = jnp.minimum(q1b, q2b) if cfg['double_q'] else q1b
        assert abs(float(jnp.mean(qb)) - float(ainfo['q_mean'])) > 1e-7, \\
            (beta, "the actor q path is NOT tilted")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. tilt non-ignorability
# --------------------------------------------------------------------------
def test_tilt_non_ignorability():
    """The point of C1: "the ignore-z fixed point stops being reachable". Simulate that
    fixed point exactly -- F frozen at random init AND evaluated at a SINGLE z, so the
    base scores are literally z-blind -- and measure TV(w(z1), w(z2)):

      beta = 0  ->  TV is EXACTLY 0 (a z-blind F gives a z-blind w: the failure mode)
      beta > 0  ->  TV bounded away from 0, by construction, at step 0.

    Then the same statement through the shipped instrument on the real path:
    density_w_ztv is finite, > 0, and moves when beta moves."""
    _run("""
    cfg1 = base_cfg(measure_density_z_tilt=1.0)
    a = agent_for(cfg1)
    bj = jb(make_batch(B=12, seed=3))
    blocks = _parse_hit_spec(cfg1['measure_hit_spec'])
    tau = cfg1['measure_density_temp']

    obs, nobs, acts = bj['observations'], bj['next_observations'], bj['actions']
    u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)
    b_emb = a.network.select('density_b')(u_sc)

    # two DIFFERENT index draws
    z1 = _sphere(jax.random.normal(jax.random.PRNGKey(11), (obs.shape[0], D)), float(D))
    z2 = _sphere(jax.random.normal(jax.random.PRNGKey(12), (obs.shape[0], D)), float(D))
    # F frozen at random init AND z-blind (evaluated once, at z1, for BOTH sides).
    f_blind = a.network.select('density_f')(jnp.concatenate([obs, acts, z1], -1))
    base = jnp.einsum('hbd,md->hbm', f_blind, b_emb) / tau

    def tv(beta):
        w1 = jax.nn.softmax(base + jnp.asarray(tilt_ref(beta, z1, b_emb, D))[None], -1)
        w2 = jax.nn.softmax(base + jnp.asarray(tilt_ref(beta, z2, b_emb, D))[None], -1)
        return 0.5 * float(jnp.mean(jnp.sum(jnp.abs(w1 - w2), -1)))

    assert tv(0.0) == 0.0, (tv(0.0), "z-blind F + no tilt should give TV exactly 0")
    assert tv(1.0) > 1e-3, (tv(1.0), "beta=1 failed to make a z-blind w z-sensitive")
    assert tv(2.0) > tv(1.0), (tv(1.0), tv(2.0), "TV is not monotone in beta")

    # ...and through the shipped instrument, on the real density_loss path.
    rng = jax.random.PRNGKey(5)
    _, i1 = a.density_loss(bj, a.network.params, rng)
    a0 = agent_for(base_cfg(measure_density_z_tilt=0.0))
    _, i0 = a0.density_loss(bj, a0.network.params, rng)
    for nm, iv in (('beta=1', i1), ('beta=0', i0)):
        v = float(iv['density_w_ztv'])
        assert np.isfinite(v) and 0.0 <= v <= 1.0 + 1e-6, (nm, v)
    assert float(i1['density_w_ztv']) > 0.0
    assert abs(float(i1['density_w_ztv']) - float(i0['density_w_ztv'])) > 1e-8, \\
        "density_w_ztv did not move with beta -- the instrument does not see the tilt"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. EMA index encoder (init, tau_z mix, invariance, eval-path readback)
# --------------------------------------------------------------------------
def test_ema_index_encoder():
    """C3's `density_b_zenc`:
      (a) registered only when enc_tau > 0, and initialized EQUAL to density_b;
      (b) the soft-update is exactly p*tau_z + ep*(1-tau_z) off the PRE-step params
          (checked at step 2 -- at step 1 the two towers are still equal and any
          formula would pass);
      (c) z is invariant to the ONLINE density_b AND to target_density_b (spec 24's
          encoder), and DEPENDS on density_b_zenc;
      (d) THE eval-path contract (rz-law lesson): the params survive a real
          flax.serialization state-dict round trip, and `sample_actions` on the
          RESTORED agent reads density_b_zenc -- perturbing it in the restored
          checkpoint changes the sampled actions, perturbing target_density_b does not.
    """
    _run("""
    TAUZ = 0.01
    cfg = base_cfg(measure_density_z_enc_tau=TAUZ)
    a = agent_for(cfg)

    # (a) registration + init
    assert 'modules_density_b_zenc' in a.network.params
    assert 'modules_density_b_zenc' not in agent_for(base_cfg()).network.params, \\
        "density_b_zenc registered with enc_tau=0 -- default param tree is not clean"
    l_b = jax.tree_util.tree_leaves(a.network.params['modules_density_b'])
    l_e = jax.tree_util.tree_leaves(a.network.params['modules_density_b_zenc'])
    assert len(l_b) == len(l_e) and all((x == y).all() for x, y in zip(l_b, l_e)), \\
        "density_b_zenc did not initialize equal to density_b"

    # (b) tau_z mix formula, at step 2
    a1, _ = a.update(jb(make_batch(seed=0)))
    b1 = a1.network.params['modules_density_b']
    e1 = a1.network.params['modules_density_b_zenc']
    a2, _ = a1.update(jb(make_batch(seed=1)))
    e2 = a2.network.params['modules_density_b_zenc']
    want = jax.tree_util.tree_map(lambda p, ep: p * TAUZ + ep * (1.0 - TAUZ), b1, e1)
    for x, y in zip(jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(e2)):
        assert np.allclose(np.asarray(x), np.asarray(y), atol=1e-6), "tau_z mix formula"
    # discriminating: the encoder must have MOVED and must LAG density_b.
    b2 = a2.network.params['modules_density_b']
    assert any(not np.allclose(np.asarray(x), np.asarray(y), atol=0)
               for x, y in zip(jax.tree_util.tree_leaves(e1),
                               jax.tree_util.tree_leaves(e2))), "encoder never moved"
    assert any(not np.allclose(np.asarray(x), np.asarray(y), atol=1e-7)
               for x, y in zip(jax.tree_util.tree_leaves(b2),
                               jax.tree_util.tree_leaves(e2))), \\
        "encoder tracks density_b exactly -- tau_z is not slowing anything down"

    # (c) which tower z reads
    g = jnp.asarray(make_batch(B=7, seed=9)['value_goals'])
    z0 = a2._density_z_encode(g)
    def bump(agent, key):
        p = dict(agent.network.params)
        p[key] = jax.tree_util.tree_map(lambda x: x + 7.0, p[key])
        return agent.replace(network=agent.network.replace(params=p))
    for key in ('modules_density_b', 'modules_target_density_b'):
        assert np.array_equal(np.asarray(z0), np.asarray(bump(a2, key)._density_z_encode(g))), \\
            "z moved when perturbing " + key + " -- enc_tau>0 must read density_b_zenc ONLY"
    assert not np.allclose(np.asarray(z0),
                           np.asarray(bump(a2, 'modules_density_b_zenc')._density_z_encode(g))), \\
        "z did NOT move when perturbing density_b_zenc -- the EMA tower is not being read"
    # ...and with enc_tau=0 the spec-24 convention is intact (target_density_b).
    a_off = agent_for(base_cfg())
    z_off = a_off._density_z_encode(g)
    assert not np.allclose(np.asarray(z_off),
                           np.asarray(bump(a_off, 'modules_target_density_b')._density_z_encode(g))), \\
        "enc_tau=0 no longer reads target_density_b"

    # (d) real serialization round trip + the EVAL entrypoint
    # exactly what utils/flax_utils.py's save_agent/restore_agent do.
    sd = flax.serialization.to_state_dict(a2)
    a_fresh = agent_for(cfg, seed=123)   # a genuinely COLD agent: different init
    assert not np.allclose(
        np.asarray(jax.tree_util.tree_leaves(
            a_fresh.network.params['modules_density_b_zenc'])[0]),
        np.asarray(jax.tree_util.tree_leaves(
            a2.network.params['modules_density_b_zenc'])[0]), atol=1e-8), \\
        "test setup bug: the cold agent already has the trained encoder"
    a_rest = flax.serialization.from_state_dict(a_fresh, sd)
    for x, y in zip(jax.tree_util.tree_leaves(a2.network.params['modules_density_b_zenc']),
                    jax.tree_util.tree_leaves(a_rest.network.params['modules_density_b_zenc'])):
        assert np.allclose(np.asarray(x), np.asarray(y), atol=0), \\
            "density_b_zenc did not survive the checkpoint round trip"

    obs_e = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    g_e = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))
    act_ref = np.asarray(a_rest.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
    assert act_ref.shape == (4, ACT_D) and np.isfinite(act_ref).all()
    assert np.allclose(act_ref,
                       np.asarray(a2.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0))),
                       atol=1e-6), "restored agent does not reproduce sample_actions"
    act_enc = np.asarray(bump(a_rest, 'modules_density_b_zenc').sample_actions(
        obs_e, g_e, seed=jax.random.PRNGKey(0)))
    act_tgt = np.asarray(bump(a_rest, 'modules_target_density_b').sample_actions(
        obs_e, g_e, seed=jax.random.PRNGKey(0)))
    assert not np.allclose(act_ref, act_enc, atol=1e-6), \\
        "sample_actions ignored density_b_zenc in the RESTORED checkpoint -- the eval " \\
        "path is not reading the same index encoder training used"
    assert np.allclose(act_ref, act_tgt, atol=1e-6), \\
        "sample_actions still reads target_density_b under enc_tau>0 (train/eval mismatch)"
    # unbatched single-observation eval shape too (the per-episode-step call).
    a1e = a_rest.sample_actions(obs_e[0], g_e[0], seed=jax.random.PRNGKey(1))
    assert a1e.shape == (ACT_D,) and np.isfinite(np.asarray(a1e)).all()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. BC mask
# --------------------------------------------------------------------------
def test_bc_hindsight_mask():
    """C3's imitation mask, checked on the REAL actor_loss path:
      - the reported distill_loss equals sum(per_row * m)/sum(m) (renormalized by the
        SURVIVING count, so an all-hindsight batch reproduces the plain mean) and
        differs from the unmasked mean on a mixed batch;
      - d(distill_loss)/d(observations) is EXACTLY zero on sphere-z rows and nonzero on
        hindsight rows -- the literal "sphere-z rows contribute zero grad" claim;
      - the q term is NOT masked (q_mean is unchanged by the knob), and bc_hindsight_frac
        is logged.
    """
    _run("""
    cfg_on = base_cfg(measure_density_z_bc_hindsight=True)
    cfg_off = base_cfg(measure_density_z_bc_hindsight=False)
    a_on, a_off = agent_for(cfg_on), agent_for(cfg_off)
    for x, y in zip(jax.tree_util.tree_leaves(a_on.network.params),
                    jax.tree_util.tree_leaves(a_off.network.params)):
        assert (x == y).all(), "the knob changed the param tree"

    b = jb(make_batch(B=32, seed=21))
    rng = jax.random.PRNGKey(4)
    _, i_on = a_on.actor_loss(b, a_on.network.params, rng)
    _, i_off = a_off.actor_loss(b, a_off.network.params, rng)

    # hand reconstruction with actor_loss's own rng-split order
    blocks = _parse_hit_spec(cfg_on['measure_hit_spec'])
    obs, goals, acts = b['observations'], b['actor_goals'], b['actions']
    _, x0_rng, t_rng, n_rng, z_rng = jax.random.split(rng, 5)
    z, m = z_ref(a_on, goals, blocks, D, cfg_on['measure_density_z_rand_frac'], z_rng)
    m = np.asarray(m)
    assert 0 < m.sum() < m.size, ("test batch is not mixed", m.sum(), m.size)
    noise = jax.random.normal(n_rng, acts.shape)
    aa = a_on.network.select('actor')(obs, z, noise, params=a_on.network.params)
    target = a_on._flow_rollout(obs, noise, a_on.network.params)
    per_row = np.asarray(jnp.mean((aa - target) ** 2, axis=-1))

    want_on = float((per_row * m).sum() / m.sum())
    want_off = float(per_row.mean())
    assert np.allclose(float(i_on['distill_loss']), want_on, atol=1e-5), \\
        (float(i_on['distill_loss']), want_on)
    assert np.allclose(float(i_off['distill_loss']), want_off, atol=1e-5), \\
        (float(i_off['distill_loss']), want_off)
    assert abs(want_on - want_off) > 1e-7, "masked and unmasked coincide -- not discriminating"
    assert np.allclose(float(i_on['bc_hindsight_frac']), float(m.mean()), atol=1e-6)
    assert 'bc_hindsight_frac' not in i_off, "the frac instrument leaked into the off arm"
    # the q term is UNMASKED (all rows, sphere ones included)
    assert np.allclose(float(i_on['q_mean']), float(i_off['q_mean']), atol=1e-6), \\
        "the knob touched the q term -- only the imitation term may be masked"

    # per-row zero grad: d(distill_loss)/d(obs_i) == 0 exactly on sphere rows.
    def dl(obs_in, agent):
        bb = dict(b); bb['observations'] = obs_in
        _, i = agent.actor_loss(bb, agent.network.params, rng)
        return i['distill_loss']
    g_on = np.abs(np.asarray(jax.grad(dl)(obs, a_on))).sum(-1)
    g_off = np.abs(np.asarray(jax.grad(dl)(obs, a_off))).sum(-1)
    sphere = m == 0.0
    hind = m == 1.0
    assert (g_on[sphere] == 0.0).all(), (g_on[sphere].max(),
        "sphere-z rows still push gradient into the masked imitation term")
    assert (g_on[hind] > 0.0).all(), (g_on[hind].min(), "hindsight rows lost their gradient")
    assert (g_off[sphere] > 0.0).all(), \\
        "test setup bug: without the mask, sphere rows must have nonzero grad"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. rdecode auxiliary
# --------------------------------------------------------------------------
def test_rdecode_ridge():
    """C2: the closed-form ridge matches a numpy reference; the auxiliary is genuinely
    ADDED to the density CE (loss_with - loss_without == coef * L_dec, and no other
    coefficient on this path can zero it); gradients reach B (and ONLY B -- the target
    towers and the actor stay clean); z_rdecode_r2 lands in [0, 1] and is > 0 on a
    kernel target a random B can partly span."""
    _run("""
    COEF = 1.0
    cfg = base_cfg(measure_density_z_rdecode_coef=COEF)
    a = agent_for(cfg)
    a0 = agent_for(base_cfg())          # same seed -> same params, coef off
    bj = jb(make_batch(B=24, seed=13))
    rng = jax.random.PRNGKey(7)

    loss, info = a.density_loss(bj, a.network.params, rng)
    loss0, info0 = a0.density_loss(bj, a0.network.params, rng)
    assert 'z_rdecode_loss' in info and 'z_rdecode_r2' in info
    assert 'z_rdecode_loss' not in info0, "the aux leaked into the coef=0 arm"

    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    b_emb = a.network.select('density_b')(u_sc)
    _, z_rng = jax.random.split(rng)
    dec_ref, r2_ref, r_c = rdecode_ref(u_sc, b_emb, jax.random.fold_in(z_rng, _Z_RDECODE_FOLD))
    assert np.allclose(dec_ref, float(info['z_rdecode_loss']), atol=1e-6), \\
        (dec_ref, float(info['z_rdecode_loss']))
    assert np.allclose(r2_ref, float(info['z_rdecode_r2']), atol=1e-6)
    assert 0.0 <= float(info['z_rdecode_r2']) <= 1.0
    assert float(info['z_rdecode_r2']) > 0.0, "the ridge explains nothing at all -- suspect"
    assert r_c.max() > 0.99, "the kernel reward has no peak at its own center"

    # the aux is genuinely ADDED at weight coef, on top of the UNCHANGED CE.
    assert np.allclose(float(loss) - float(loss0), COEF * dec_ref, atol=1e-5), \\
        (float(loss), float(loss0), dec_ref)
    ref = density_ref(a, cfg, bj, rng)
    assert np.allclose(ref['loss'], float(loss), atol=1e-5)
    assert np.allclose(ref['ce'], float(loss0), atol=1e-5), \\
        "the CE itself moved when the auxiliary was switched on"
    # dose responds (no silent clamp)
    a2 = agent_for(base_cfg(measure_density_z_rdecode_coef=3.0))
    loss2, _ = a2.density_loss(bj, a2.network.params, rng)
    assert np.allclose(float(loss2) - float(loss0), 3.0 * dec_ref, atol=1e-5)

    # grads reach B, and only B.
    def aux_only(params):
        bb = a.network.select('density_b')(u_sc, params=params)
        l, _ = a._density_rdecode_loss(u_sc, bb, jax.random.fold_in(z_rng, _Z_RDECODE_FOLD))
        return l
    ga = jax.grad(aux_only)(a.network.params)
    assert gnorm(ga, 'modules_density_b') > 1e-8, gnorm(ga, 'modules_density_b')
    for key in ('modules_density_f', 'modules_target_density_b',
                'modules_target_density_f', 'modules_actor'):
        assert gnorm(ga, key) < 1e-12, (key, gnorm(ga, key))
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. density_w_ztv logged in z mode, absent otherwise
# --------------------------------------------------------------------------
def test_ztv_logged_in_z_mode_only():
    """The instrument is the spec's own pre-registered gate, so it is logged in EVERY z
    arm (knobs on or off) and in NO other arm. Checked through total_loss's prefixed
    info (what wandb actually records) and through a real update() step, not just
    density_loss in isolation. Value is a valid TV: in [0, 1], and 0 exactly when the
    two index draws coincide (rand_frac=0 + no tilt makes z' == z)."""
    _run("""
    bj = jb(make_batch(B=8, seed=31))
    rng = jax.random.PRNGKey(2)

    for label, cfg in (('z defaults', base_cfg()),
                       ('z full stack', full_stack_cfg())):
        ag = agent_for(cfg)
        _, info = ag.total_loss(bj, ag.network.params, rng)
        assert 'measure/density_w_ztv' in info, (label, sorted(info))
        v = float(info['measure/density_w_ztv'])
        assert np.isfinite(v) and -1e-6 <= v <= 1.0 + 1e-6, (label, v)
        ag2, uinfo = ag.update(bj)
        assert 'measure/density_w_ztv' in uinfo, (label, "missing from update() info")

    for label, cfg in (('g-indexed density', base_cfg(measure_density_z=False)),
                       ('no measure knobs', dict(get_config(), encoder=None, frame_stack=None))):
        ag = agent_for(cfg)
        _, info = ag.total_loss(bj, ag.network.params, rng)
        assert not any('density_w_ztv' in k for k in info), (label, sorted(info))

    # degenerate control: identical index draws -> TV exactly 0.
    ag = agent_for(base_cfg(measure_density_z_rand_frac=0.0))
    _, info = ag.density_loss(bj, ag.network.params, rng)
    assert abs(float(info['density_w_ztv'])) < 1e-6, \\
        (float(info['density_w_ztv']), "z' == z must give TV 0 -- the meter is miscalibrated")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. lattice refusals + full-stack end-to-end
# --------------------------------------------------------------------------
def test_lattice_and_full_stack_end_to_end():
    """Every spec-25 knob is refused outside the measure_density_z lattice; ranges are
    enforced (beta >= 0, coef >= 0, tau_z in [0, 0.1]); and the two CONTRADICTORY
    bc_hindsight combinations -- rand_frac=1 (the mask would zero the imitation term
    outright: a silently-disabled loss term, the 17b fake-NULL species) and rand_frac=0
    / bc_coeff=0 (provably inert) -- are refused loudly.

    Then 30 real update() steps with C1+C2+C3 all on: finite info every step, w on the
    simplex at the end, the EMA encoder and both towers actually moved, and
    sample_actions still works from the trained agent."""
    _run("""
    def refuses(cfg):
        try:
            agent_for(cfg)
        except AssertionError:
            return True
        return False

    # each knob outside the z lattice
    for k, v in (('measure_density_z_tilt', 1.0),
                 ('measure_density_z_rdecode_coef', 1.0),
                 ('measure_density_z_bc_hindsight', True),
                 ('measure_density_z_enc_tau', 0.0005)):
        assert refuses(base_cfg(measure_density_z=False, **{k: v})), k
        cfg_nd = dict(get_config(), encoder=None, frame_stack=None)
        cfg_nd[k] = v
        assert refuses(cfg_nd), (k, 'no measure_density at all')
    # ranges
    assert refuses(base_cfg(measure_density_z_tilt=-0.5))
    assert refuses(base_cfg(measure_density_z_rdecode_coef=-1e-3))
    assert refuses(base_cfg(measure_density_z_enc_tau=-1e-4))
    assert refuses(base_cfg(measure_density_z_enc_tau=0.2))
    # contradictory bc_hindsight combinations
    assert refuses(base_cfg(measure_density_z_bc_hindsight=True,
                            measure_density_z_rand_frac=1.0))
    assert refuses(base_cfg(measure_density_z_bc_hindsight=True,
                            measure_density_z_rand_frac=0.0))
    assert refuses(base_cfg(measure_density_z_bc_hindsight=True, bc_coeff=0.0))
    # accepted
    assert not refuses(base_cfg())
    assert not refuses(full_stack_cfg())
    assert not refuses(base_cfg(measure_density_z_enc_tau=0.1))    # boundary

    # --- full stack, 30 real update() steps ---
    cfg = full_stack_cfg()
    a = agent_for(cfg)
    p0 = a.network.params
    for step in range(30):
        b = jb(make_batch(seed=step))
        a, info = a.update(b)
        for key in ('measure/density_ce', 'measure/density_ess', 'measure/z_cos_spread',
                    'measure/density_w_ztv', 'measure/z_rdecode_loss',
                    'measure/z_rdecode_r2', 'actor/actor_loss', 'actor/q_mean',
                    'actor/distill_loss', 'actor/bc_hindsight_frac'):
            v = float(info[key])
            assert np.isfinite(v), (step, key, v)
        assert 0.0 <= float(info['measure/z_rdecode_r2']) <= 1.0, step
        assert 0.0 <= float(info['measure/density_w_ztv']) <= 1.0 + 1e-6, step
    assert all(np.isfinite(np.asarray(l)).all()
               for l in jax.tree_util.tree_leaves(a.network.params)), "NaN after 30 updates"
    for tower in ('density_f', 'density_b', 'density_b_zenc'):
        assert any(not np.allclose(np.asarray(x), np.asarray(y), atol=0)
                   for x, y in zip(jax.tree_util.tree_leaves(p0['modules_' + tower]),
                                   jax.tree_util.tree_leaves(a.network.params['modules_' + tower]))), \\
            tower + " did not move after 30 updates"

    # w still on the simplex under the tilted family
    b = jb(make_batch(seed=99))
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    z = a._density_z(b['value_goals'], jax.random.PRNGKey(0))
    u_sc = GCIQLFlowAgent._density_queries(b['next_observations'], blocks)
    b_emb = a.network.select('density_b')(u_sc)
    f_emb = a.network.select('density_f')(jnp.concatenate([b['observations'], b['actions'], z], -1))
    lg = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp'] \\
        + jnp.asarray(tilt_ref(cfg['measure_density_z_tilt'], z, b_emb, D))[None]
    w = np.asarray(jax.nn.softmax(lg, -1))
    assert np.allclose(w.sum(-1), 1.0, atol=1e-4) and (w >= -1e-6).all()

    # and the EVAL entrypoint still runs on the trained full-stack agent
    obs_e = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    g_e = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))
    ac = np.asarray(a.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
    assert ac.shape == (4, ACT_D) and np.isfinite(ac).all()
    print("OK")
    """)
