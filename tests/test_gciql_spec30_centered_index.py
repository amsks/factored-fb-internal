"""Spec 30 P1 (Rev 1) -- the translation-gauge-invariant CENTERED dual encoder.

Contract: `PAPER/Multistep-FB/specs/30-review-response.md` P1 + `specs/30a-impl-wave1.md`
Item 1 (and Item 4's duplicate instrument, tested at the bottom).

Spec 30 re-measured the geometry spec 28 diagnosed and found that the "rank-1 collapse"
was largely a SOFTMAX-GAUGE MEAN OFFSET: the density is exactly invariant to B -> B + b,
||mu_B|| is 33-103x the centered signal, the CENTERED covariance has effective rank 6-21
of 50, and centering ALONE lifts the point-index reconstruction AUC from .43-.86 to .99+.
It also found the trained spec-29 (uncentered) encoder is SHIFT-UNSTABLE --
cos(enc(r), enc(r+1)) ~ 0, i.e. a policy-preserving constant reward shift rotates the task
index to near-orthogonal -- while the centered form is exactly invariant. One default-off
enum value conditions on the centered object instead:

  `measure_density_z_encode = 'dual_centered'`, TWO SEPARATE APIs (there is NO M=1 limit
  of the centered sample encoder: m_c = (1/M) sum_j (B(u_j)-mu)(r_j-rbar) is identically
  zero at M=1, since r_1 - rbar = 0):

    point   (measure_density_z_prompt='point')   z = sqrt(d)*norm(B̄(phi(g)) - mu_hat)
            -- the ANALYTIC CENTERED POINT PROMPT. Never a "Dirac/M=1 limit".
    sample  (measure_density_z_prompt='sample')  zhat = sqrt(d)*norm((C_hat+lam I)^-1 m_c)
            -- the general reward interface, M > 1 (M=1 raises), with the
            constant-reward fallback ||c|| < eps_z -> z_neutral.

The invariance tests (3, 4) are the point of the whole redesign, so they are pinned at
float32 round-off (atol 1e-5 on components of a norm-sqrt(d) vector), against controls
that move by O(1) -- not at a tolerance loose enough to hide a broken centering.

The other scars, each with a test:
  1. mu_hat and C_hat are EMA'd, stored as ORDINARY PARAM LEAVES (the density_b_zenc /
     ZSigmaState pattern) on the SAME schedule as spec 29's uncentered Sigma_hat, centered
     and 1/M, symmetrized every update (test 7 -- the centered sibling of spec 29's
     `test_sigma_is_ema_uncentered_and_matches_spec28_pool`). Batch statistics are
     forbidden, so this suite drives update() AND sample_actions throughout.
  2. `sample_actions` on the reward-sample arm goes through the REWARD-SAMPLE API over the
     frozen M-row pool and NEVER silently substitutes the point prompt (test 8) -- that
     substitution is what made spec 29's .812 uninterpretable (spec 30 sec 4.8).
  3. Every knob is refused where it would be inert (test 9).

Same subprocess-jax harness as tests/test_gciql_spec29_dual_index.py: OBS_D=40, action
dim 5, d=8, hit_spec '19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_spec30_centered_index.py -x -q`
"""
import subprocess
import textwrap

import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8    # value_latent_dim AND measure_density_dim, matches the spec-18/24/25/28/29 harness
ACT_D = 5
POOL_M = 8   # measure_density_z_eval_pool_m in the tests (256 in the shipped arms)

# The last commit BEFORE spec 30's implementation. Pinned, not HEAD: HEAD moves forward on
# this shared branch as other sessions land commits (spec-28/29 discipline).
PRE_SPEC30_SHA = "59f50725e401659c6ae6a522fb008f6bf9d73541"

NEW_KNOBS = ("measure_density_z_prompt", "measure_density_z_eval_pool_m",
             "measure_density_z_cov_eig_floor", "measure_density_z_const_eps",
             "measure_density_dup_instrument", "measure_density_dup_eps")


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                       cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0


COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec, _sphere

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    POOL_M = {POOL_M}
    REPO_DIR = {str(REPO)!r}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def g_cfg(**kw):
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["actor_q_center"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg.update(kw)
        return cfg

    def z_cfg(**kw):
        return g_cfg(measure_density_z=True, **kw)

    def dual_cfg(**kw):
        # spec 29's UNCENTERED arm -- the control this spec is measured against.
        kw.setdefault("measure_density_z_enc_tau", 0.0005)
        return z_cfg(measure_density_z_encode="dual", **kw)

    def cen_cfg(**kw):
        # spec 30's centered arm. prompt defaults to 'sample' (the general API).
        kw.setdefault("measure_density_z_enc_tau", 0.0005)
        if str(kw.get("measure_density_z_prompt", "sample")) == "sample":
            kw.setdefault("measure_density_z_eval_pool_m", POOL_M)
        return z_cfg(measure_density_z_encode="dual_centered", **kw)

    def point_cfg(**kw):
        return cen_cfg(measure_density_z_prompt="point", **kw)

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

    def blocks_of(cfg):
        return _parse_hit_spec(cfg["measure_hit_spec"])

    def b_of(agent, x_scaled, mod='density_b_zenc'):
        return np.asarray(agent.network.select(mod)(x_scaled), np.float64)

    def moments_of(agent):
        p = agent.network.params['modules_density_zmoments']
        return np.asarray(p['mu'], np.float64), np.asarray(p['cov'], np.float64)

    def pool_of(agent):
        p = agent.network.params['modules_density_zpool']
        return np.asarray(p['pool'], np.float64), float(p['filled'])

    def kg_ref(goals, protos_obs, blocks):
        u_raw = GCIQLFlowAgent._raw_block_cols(protos_obs, blocks)
        return np.asarray(GCIQLFlowAgent._delta_phi(goals, u_raw, blocks), np.float64)

    def cen_ref(b_pool, r_rows, mu, cov, lam_mult=0.01, floor_mult=1e-6):
        \"\"\"The reference centered encoder, in numpy float64 -- the same convention as
        scripts/rescuing_fb/eval_spec30_geom.py::ridge_batch(centered=True) (which produced
        the spec-30 preflight evidence), plus the eigenvalue floor the agent applies.\"\"\"
        b_pool = np.asarray(b_pool, np.float64); r_rows = np.atleast_2d(np.asarray(r_rows, np.float64))
        M, d = b_pool.shape
        db = b_pool - np.asarray(mu, np.float64)[None]
        dr = r_rows - r_rows.mean(axis=1, keepdims=True)
        m_c = dr @ db / M                                              # [N, d]
        cov = 0.5 * (np.asarray(cov, np.float64) + np.asarray(cov, np.float64).T)
        tr = np.trace(cov)
        ev, vec = np.linalg.eigh(cov)
        ev = np.maximum(ev, floor_mult * tr / d)
        lam = lam_mult * tr / d
        c = ((m_c @ vec) / (ev + lam)[None, :]) @ vec.T
        return np.sqrt(float(d)) * c / np.linalg.norm(c, axis=-1, keepdims=True), c

    def bumped_b(agent, shift):
        \"\"\"B̄(x) -> B̄(x) + shift for EVERY x, by moving the index tower's output bias.\"\"\"
        p = dict(agent.network.params)
        zenc = jax.tree_util.tree_map(lambda x: x, agent.network.params['modules_density_b_zenc'])
        last = sorted(k for k in zenc if k.startswith('Dense_'))[-1]
        zenc = dict(zenc); zenc[last] = dict(zenc[last])
        zenc[last]['bias'] = zenc[last]['bias'] + jnp.asarray(shift, jnp.float32)
        p['modules_density_b_zenc'] = zenc
        return agent.replace(network=agent.network.replace(params=p))

    def with_mu(agent, mu_new):
        p = dict(agent.network.params)
        p['modules_density_zmoments'] = dict(p['modules_density_zmoments'])
        p['modules_density_zmoments']['mu'] = jnp.asarray(mu_new, jnp.float32)
        return agent.replace(network=agent.network.replace(params=p))
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


# --------------------------------------------------------------------------
# 1. all-defaults bit-identity to the pre-spec-30 agent
# --------------------------------------------------------------------------
def test_default_off_bit_identical_to_pre_spec30():
    """Every spec-30 knob at its default: param tree, total_loss, THREE REAL update()
    steps and sample_actions must be bit-identical to the agent loaded from the pinned
    pre-spec-30 commit -- on the spec-24 z arm, the spec-25 C3 arm, the spec-28 cell-(ii)
    arm, the spec-29 DUAL arm (uncentered, the one this spec must not disturb), the
    g-indexed arm and a config with no measure knobs at all.

    Real update steps plus the EVAL entrypoint, not a construct-only check: a previous z
    build passed 8/8 unit tests while its eval path was broken because nothing drove
    sample_actions."""
    if not _baseline_reachable(PRE_SPEC30_SHA):
        pytest.skip(f"pinned baseline {PRE_SPEC30_SHA} is not reachable in this checkout")
    _run(f"""
    PRE_SPEC30_SHA = {PRE_SPEC30_SHA!r}
    NEW_KNOBS = {NEW_KNOBS!r}
    """ + """
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", PRE_SPEC30_SHA + ":tools/wandb_mode_shim/gciql_flow.py"],
        cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    assert "measure_density_z_encode" in baseline_src, \\
        "pinned baseline predates spec 29 -- re-pin PRE_SPEC30_SHA"
    for k30 in NEW_KNOBS:
        assert k30 not in baseline_src, (
            "pinned baseline already contains " + k30 + " -- the pin has gone stale; "
            "re-pin PRE_SPEC30_SHA to the last commit before the spec-30 implementation")
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_pre30.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_pre30", baseline_path)
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
    obs_e = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    g_e = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))

    def same_after_updates(cfg_kw, label, n=3):
        a_b = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                             cfg_of(baseline.get_config, **cfg_kw))
        a_n = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_of(get_config, **cfg_kw))
        assert set(a_b.network.params) == set(a_n.network.params), \\
            (label, sorted(a_b.network.params), sorted(a_n.network.params))
        assert all((x == y).all() for x, y in zip(
            jax.tree_util.tree_leaves(a_b.network.params),
            jax.tree_util.tree_leaves(a_n.network.params))), label + ": param tree"
        l_b, i_b = a_b.total_loss(bj, a_b.network.params, jax.random.PRNGKey(7))
        l_n, i_n = a_n.total_loss(bj, a_n.network.params, jax.random.PRNGKey(7))
        assert float(l_b) == float(l_n), (label, float(l_b), float(l_n))
        assert sorted(i_b) == sorted(i_n), (label, "info keys drifted", set(i_b) ^ set(i_n))
        for step in range(n):
            bb = jb(make_batch(seed=step))
            a_b, ib = a_b.update(bb)
            a_n, inn = a_n.update(bb)
            assert sorted(ib) == sorted(inn), (label, step, set(ib) ^ set(inn))
            for k in ('measure/density_ce', 'actor/actor_loss', 'actor/q_mean'):
                assert float(ib[k]) == float(inn[k]), (label, step, k)
        assert all((x == y).all() for x, y in zip(
            jax.tree_util.tree_leaves(a_b.network.params),
            jax.tree_util.tree_leaves(a_n.network.params))), \\
            label + ": param tree diverged after " + str(n) + " real update() steps"
        ac_b = np.asarray(a_b.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        ac_n = np.asarray(a_n.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        assert (ac_b == ac_n).all(), label + ": sample_actions diverged"

    same_after_updates(dict(measure_density_z=True), "spec-24 z arm")
    same_after_updates(dict(measure_density_z=True, measure_density_z_enc_tau=0.0005),
                       "spec-25 C3 arm")
    same_after_updates(dict(measure_density_z=True,
                            measure_density_z_actor_reward='kernel'),
                       "spec-28 cell (ii)")
    same_after_updates(dict(measure_density_z=True, measure_density_z_encode='dual',
                            measure_density_z_enc_tau=0.0005,
                            measure_density_z_actor_reward='kernel'),
                       "spec-29 R-A/kernel (uncentered dual)")
    same_after_updates(dict(), "g-indexed density arm")

    cnb = dict(baseline.get_config()); cnb["encoder"] = None; cnb["frame_stack"] = None
    cnn = dict(get_config()); cnn["encoder"] = None; cnn["frame_stack"] = None
    ab = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cnb)
    an = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cnn)
    assert set(ab.network.params) == set(an.network.params)
    assert all((x == y).all() for x, y in zip(
        jax.tree_util.tree_leaves(ab.network.params),
        jax.tree_util.tree_leaves(an.network.params))), "no-knob config"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. the two APIs are the two formulas (and they are DIFFERENT objects)
# --------------------------------------------------------------------------
def test_two_apis_match_their_closed_forms():
    """point: z = sqrt(d)*norm(B̄(phi(g)) - mu_hat) -- mean subtraction ALONE, no inverse
    covariance (spec 30 P2.3: centering and dualization must not be bundled).
    sample: zhat = sqrt(d)*norm((C_hat+lam I)^-1 m_c), m_c = (1/M) sum (B̄(u_j)-mu)(r_j-rbar),
    checked against the numpy reference that reproduces eval_spec30_geom.py's
    ridge_batch(centered=True) -- the code that produced the spec-30 preflight evidence.
    Both carry ||z|| = sqrt(d), and they are NOT the same vector (otherwise the P1 screen's
    two arms would be one arm)."""
    _run("""
    cfg = cen_cfg()
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    for s in range(2):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    mu, cov = moments_of(a)
    assert not np.allclose(mu, 0.0) and not np.allclose(cov, np.eye(D)), \\
        "moments never moved off their init -- the checks below would be vacuous"

    bj = jb(make_batch(B=32, seed=7))
    goals = bj['actor_goals']
    x_sc = GCIQLFlowAgent._density_queries(goals, blocks)
    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    kg = kg_ref(goals, bj['next_observations'], blocks)
    assert np.allclose(kg.sum(-1), 1.0, atol=1e-5), \\
        "reward samples must be the QUERY-SET NORMALIZED kernel (spec 28 parity rule)"

    # ---- point prompt ----
    bg = b_of(a, x_sc)
    ref_p = np.sqrt(float(D)) * (bg - mu[None]) / np.linalg.norm(bg - mu[None], axis=-1, keepdims=True)
    got_p = np.asarray(a._density_z_point_prompt(x_sc), np.float64)
    assert np.allclose(got_p, ref_p, atol=1e-5), (got_p[0], ref_p[0])
    assert np.allclose(np.asarray(a.density_z_point_prompt(goals), np.float64), ref_p, atol=1e-5)

    # ---- reward-sample encoder ----
    ref_s, c_ref = cen_ref(b_of(a, u_sc), kg, mu, cov)
    got_s = np.asarray(a._density_z_encode(goals, u_sc, jnp.asarray(kg)), np.float64)
    assert np.allclose(got_s, ref_s, atol=1e-4), (got_s[0], ref_s[0])
    got_h = np.asarray(a.density_z_from_reward_samples(u_sc, jnp.asarray(kg)), np.float64)
    assert np.allclose(got_h, ref_s, atol=1e-4), "harness entrypoint disagrees with training"

    # the eigenvalue floor is a no-op at this conditioning: the plain solve agrees
    lam = 0.01 * np.trace(cov) / D
    db = b_of(a, u_sc) - mu[None]
    m_c = (kg - kg.mean(1, keepdims=True)) @ db / kg.shape[-1]
    c_plain = np.linalg.solve(0.5 * (cov + cov.T) + lam * np.eye(D), m_c.T).T
    z_plain = np.sqrt(float(D)) * c_plain / np.linalg.norm(c_plain, axis=-1, keepdims=True)
    assert np.allclose(got_s, z_plain, atol=1e-3), (got_s[0], z_plain[0])

    # ---- both carry the standing norm invariant, and they are different objects ----
    for z_ in (got_p, got_s):
        assert np.allclose(np.linalg.norm(z_, axis=-1), np.sqrt(float(D)), atol=1e-4)
    cos = (got_p * got_s).sum(-1) / (np.linalg.norm(got_p, -1) * np.linalg.norm(got_s, -1))
    assert (cos < 0.999).any(), \\
        ("point prompt and reward-sample index coincide -- the P1 arms would not separate", cos)

    # ... and the centered point prompt is NOT the (uncentered) sphere index
    z_sphere = np.asarray(_sphere(jnp.asarray(bg), float(D)), np.float64)
    cos_s = (got_p * z_sphere).sum(-1) / (np.linalg.norm(got_p, -1) * np.linalg.norm(z_sphere, -1))
    assert (cos_s < 0.999).any(), "centering changed nothing -- mu_hat is not being subtracted"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. EXACT invariance to the additive gauge B -> B + b
# --------------------------------------------------------------------------
def test_exact_invariance_under_additive_b_gauge():
    """The density softmax is exactly invariant to B(x) -> B(x) + b, so a task encoder that
    moves under it is conditioning the policy on an arbitrary gauge (spec 30, our stated
    novelty). Both APIs must be invariant:
      * point prompt: invariant once mu_hat carries the same shift (it is E_rho[B]);
      * reward-sample: invariant EVEN IF mu_hat is left stale, because sum_j (r_j - rbar) = 0
        makes m_c literally independent of mu -- tested separately, because it is a
        stronger property and it is what makes the API robust to a lagging EMA.
    Control: the spec-29 UNCENTERED encoder moves by O(1) under the same shift."""
    _run("""
    cfg = cen_cfg()
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    mu, cov = moments_of(a)

    bj = jb(make_batch(B=32, seed=11))
    goals = bj['actor_goals']
    x_sc = GCIQLFlowAgent._density_queries(goals, blocks)
    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    kg = jnp.asarray(kg_ref(goals, bj['next_observations'], blocks))

    shift = np.linspace(-1.5, 2.0, D).astype(np.float32)   # a real gauge move, not epsilon
    a_sh = bumped_b(a, shift)
    assert np.allclose(b_of(a_sh, x_sc), b_of(a, x_sc) + shift[None], atol=1e-5), \\
        "the bias bump did not implement B -> B + b"

    # -- point prompt: shift the tower AND mu_hat (mu_hat = E_rho[B] moves with the gauge)
    z_p0 = np.asarray(a._density_z_point_prompt(x_sc), np.float64)
    z_p1 = np.asarray(with_mu(a_sh, mu + shift)._density_z_point_prompt(x_sc), np.float64)
    dp = np.abs(z_p0 - z_p1).max()
    assert dp < 1e-5, ("centered POINT prompt is not gauge-invariant", dp, z_p0[0], z_p1[0])

    # -- reward-sample: invariant with the shifted mu ...
    z_s0 = np.asarray(a._density_z_encode(goals, u_sc, kg), np.float64)
    z_s1 = np.asarray(with_mu(a_sh, mu + shift)._density_z_encode(goals, u_sc, kg), np.float64)
    ds = np.abs(z_s0 - z_s1).max()
    assert ds < 1e-5, ("centered SAMPLE encoder is not gauge-invariant", ds)
    # ... and ALSO with a stale mu (m_c does not depend on mu at all)
    z_s2 = np.asarray(a_sh._density_z_encode(goals, u_sc, kg), np.float64)
    ds2 = np.abs(z_s0 - z_s2).max()
    assert ds2 < 1e-5, ("m_c must be independent of mu_hat (sum_j (r_j - rbar) = 0)", ds2)

    # -- CONTROL: the point prompt with a STALE mu does move (the test can discriminate)
    z_p2 = np.asarray(a_sh._density_z_point_prompt(x_sc), np.float64)
    assert np.abs(z_p0 - z_p2).max() > 1e-2, \\
        "shifting B without shifting mu left the point prompt unchanged -- vacuous test"

    # -- CONTROL: spec 29's UNCENTERED encoder is NOT gauge-invariant
    au = agent_for(dual_cfg())
    for s in range(3):
        au, _ = au.update(jb(make_batch(B=32, seed=s)))
    zu0 = np.asarray(au._density_z_encode(goals, u_sc, kg), np.float64)
    zu1 = np.asarray(bumped_b(au, shift)._density_z_encode(goals, u_sc, kg), np.float64)
    assert np.abs(zu0 - zu1).max() > 1e-2, \\
        "the uncentered control did not move under the gauge shift -- control is broken"
    print("OK point_dz=%.3g sample_dz=%.3g stale_mu_dz=%.3g" % (dp, ds, ds2))
    """)


# --------------------------------------------------------------------------
# 4. EXACT invariance to reward shift r -> r + beta and rescale r -> alpha r
# --------------------------------------------------------------------------
def test_exact_invariance_under_reward_shift_and_scale():
    """Spec 30 sec 4.5, the severe finding: the TRAINED uncentered encoder reads
    cos(enc(kg), enc(kg+1)) = .035/-.000/-.044 -- a constant reward shift, which changes no
    policy at all, rotates the task index to near-orthogonal. The centered encoder reads
    1.000000. Here: r -> r + beta and r -> alpha r (alpha > 0) leave zhat unchanged at
    float32 round-off, for several beta including a large one, with the uncentered arm as
    the control that moves. A NEGATIVE alpha must flip the index (else the check would pass
    for a degenerate encoder that ignores r)."""
    _run("""
    cfg = cen_cfg()
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))

    bj = jb(make_batch(B=32, seed=13))
    goals = bj['actor_goals']
    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    kg = kg_ref(goals, bj['next_observations'], blocks).astype(np.float32)

    def enc(r):
        return np.asarray(a.density_z_from_reward_samples(u_sc, jnp.asarray(r)), np.float64)

    def cos_of(x, y):
        return (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1))

    z0 = enc(kg)
    worst_shift = 0.0
    for beta, tol, cos_tol in ((1.0, 1e-5, 1e-8), (-5.0, 1e-4, 1e-8), (100.0, 2e-3, 1e-6)):
        # The algebraic identity is EXACT. The float32 EVALUATION of (r+beta) - mean(r+beta)
        # loses digits by cancellation in proportion to beta/|r| (kg rows sum to 1 over M,
        # so r ~ 1/M): the per-component bound therefore scales with beta, while the
        # SCALE-FREE statement -- the direction, which is all the index is -- stays pinned
        # at cos > 1 - 1e-8 for every beta. Spec 30 sec 4.5's own probe is r+1, where the
        # UNCENTERED encoder rotates to cos ~ .035/-.000/-.044.
        z_b = enc(kg + beta)
        d_ = np.abs(z0 - z_b).max()
        worst_shift = max(worst_shift, d_)
        assert d_ < tol, ("centered encoder moved under r -> r + %g" % beta, d_)
        cos = cos_of(z0, z_b)
        assert (cos > 1 - cos_tol).all(), ("cos(enc(r), enc(r+beta)) < 1", beta, cos.min())

    worst_scale = 0.0
    for alpha in (3.7, 0.01):
        d_ = np.abs(z0 - enc(alpha * kg)).max()
        worst_scale = max(worst_scale, d_)
        assert d_ < 1e-4, ("centered encoder moved under r -> %g r" % alpha, d_)

    # a NEGATIVE rescale must flip it (the encoder is not ignoring r)
    z_neg = enc(-1.0 * kg)
    assert np.abs(z0 + z_neg).max() < 1e-3, "r -> -r must flip the index exactly"
    assert np.abs(z0 - z_neg).max() > 1e-2, "vacuous: the index is ~0"

    # composed affine map: alpha r + beta
    assert np.abs(z0 - enc(2.5 * kg - 7.0)).max() < 1e-4

    # ---- CONTROL: spec 29's uncentered encoder is shift-UNSTABLE (sec 4.5) ----
    au = agent_for(dual_cfg())
    for s in range(3):
        au, _ = au.update(jb(make_batch(B=32, seed=s)))
    zu0 = np.asarray(au.density_z_from_reward_samples(u_sc, jnp.asarray(kg)), np.float64)
    zu1 = np.asarray(au.density_z_from_reward_samples(u_sc, jnp.asarray(kg + 1.0)), np.float64)
    cos_u = (zu0 * zu1).sum(-1) / D
    assert np.abs(zu0 - zu1).max() > 1e-2, \\
        "the uncentered control did not move under r -> r+1 -- control is broken"
    print("OK shift<=%.3g scale<=%.3g uncentered_cos_mean=%.4f"
          % (worst_shift, worst_scale, float(cos_u.mean())))
    """)


# --------------------------------------------------------------------------
# 5. M = 1 raises (no silent degradation to a zero index)
# --------------------------------------------------------------------------
def test_m_equals_one_raises():
    """The centered moment m_c = (1/M) sum_j (B(u_j)-mu)(r_j-rbar) is IDENTICALLY ZERO at
    M=1 (r_1 - rbar = 0). Silently returning it would hand the actor the constant-reward
    fallback index while the caller believed it had encoded a task, so a one-sample call is
    an ERROR -- on the harness entrypoint AND on the training encode -- and the message must
    point at the point-prompt API. The uncentered spec-29 encoder, which does have an M=1
    member, keeps working (this is not a global M>1 rule)."""
    _run("""
    cfg = cen_cfg()
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    a, _ = a.update(jb(make_batch(B=32, seed=1)))
    bj = jb(make_batch(B=32, seed=2))
    goals = bj['actor_goals']
    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    kg = jnp.asarray(kg_ref(goals, bj['next_observations'], blocks))

    def raises(fn):
        try:
            fn()
        except ValueError as e:
            assert "M > 1" in str(e) and "point" in str(e), str(e)
            return True
        return False

    assert raises(lambda: a.density_z_from_reward_samples(u_sc[:1], kg[:, :1])), \\
        "M=1 must RAISE on the general reward interface"
    assert raises(lambda: a._density_z_encode(goals, u_sc[:1], kg[:, :1])), \\
        "M=1 must RAISE on the training encode too"
    assert raises(lambda: a.density_z_from_reward_samples(u_sc[:1], kg[0, :1])), \\
        "M=1 must RAISE for an unbatched reward vector as well"
    # M=2 is accepted (the boundary is exactly at 1)
    z2 = np.asarray(a.density_z_from_reward_samples(u_sc[:2], kg[:, :2]), np.float64)
    assert np.isfinite(z2).all()

    # the UNCENTERED encoder is unaffected -- it does have an M=1 member
    au = agent_for(dual_cfg())
    au, _ = au.update(jb(make_batch(B=32, seed=1)))
    z1u = np.asarray(au.density_z_from_reward_samples(u_sc[:1], kg[:, :1]), np.float64)
    assert np.isfinite(z1u).all(), "spec 29's uncentered M=1 path must keep working"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. the constant-reward fallback
# --------------------------------------------------------------------------
def test_constant_reward_fallback():
    """A (near-)constant reward has m_c ~ 0; normalizing it would amplify numerical noise
    into a confident-looking task index. ||c|| < eps_z returns z_neutral = 1_d (norm
    sqrt(d), so the standing invariant holds), the fallback rate is logged, and a
    genuinely informative reward does NOT trigger it."""
    _run("""
    cfg = cen_cfg(measure_density_z_actor_reward='kernel')
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    bj = jb(make_batch(B=32, seed=5))
    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    M = u_sc.shape[0]

    for const in (0.0, 1.0, -3.25):
        z = np.asarray(a.density_z_from_reward_samples(
            u_sc, jnp.full((3, M), const, jnp.float32)), np.float64)
        assert np.allclose(z, 1.0, atol=1e-6), ("constant reward must give z_neutral", const, z[0])
        assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(float(D)), atol=1e-5), \\
            "z_neutral must carry ||z|| = sqrt(d)"

    # an informative reward must NOT hit the fallback
    kg = kg_ref(bj['actor_goals'], bj['next_observations'], blocks)
    z_ok = np.asarray(a.density_z_from_reward_samples(u_sc, jnp.asarray(kg)), np.float64)
    assert not np.allclose(z_ok, 1.0, atol=1e-3), "an informative reward hit the fallback"

    # an informative reward riding on a LARGE constant offset is still encoded, and
    # encoded identically to the un-offset one (the fallback keys on ||c||, i.e. on the
    # centered signal, not on the level of r) ...
    z_off = np.asarray(a.density_z_from_reward_samples(
        u_sc, jnp.asarray((5.0 + kg).astype(np.float32))), np.float64)
    assert np.allclose(z_off, z_ok, atol=1e-3), \\
        "a constant offset on an informative reward changed the index"
    # ... and a large eps_z pushes even that into the fallback, which is what makes eps_z a
    # real knob rather than decoration.
    a_big = agent_for(cen_cfg(measure_density_z_actor_reward='kernel',
                              measure_density_z_const_eps=1e3))
    a_big = a_big.replace(network=a_big.network.replace(params=a.network.params))
    z_big = np.asarray(a_big.density_z_from_reward_samples(u_sc, jnp.asarray(kg)), np.float64)
    assert np.allclose(z_big, 1.0, atol=1e-6), "eps_z is not being read"

    # the diagnostics the spec requires are logged every step on the sample arm
    _, info = a.update(bj)
    for k in ('measure/density_zc_mc_norm', 'measure/density_zc_c_norm',
              'measure/density_zc_r_var', 'measure/density_zc_fallback_frac'):
        assert k in info and np.isfinite(float(info[k])), k
    assert float(info['measure/density_zc_r_var']) > 0.0
    assert float(info['measure/density_zc_fallback_frac']) == 0.0
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. mu_hat / C_hat: EMA'd, CENTERED, 1/M, symmetric  (the spec-29 sibling)
# --------------------------------------------------------------------------
def test_moments_are_ema_centered_and_symmetric():
    """The centered sibling of spec 29's
    `test_sigma_is_ema_uncentered_and_matches_spec28_pool`, on the same three axes.

    (a) EMA at the configured rate (the SAME knob/schedule as spec 29's Sigma_hat): after
        one update, mu = (1-tau)*0 + tau*mean(B) and C = (1-tau)*I + tau*Cov(B), with B the
        PRE-STEP index tower on THIS batch's queries (the pre-step source discipline the
        target/zenc soft-updates use).
    (b) CENTERED and 1/M: at tau=1 the stored C equals np.cov(B.T, bias=True) -- and NOT
        the uncentered second moment spec 29 stores, and not the 1/(M-1) version.
    (c) symmetric exactly (symmetrized every update), and the whole index reproduces the
        eval_spec30_geom.py centered-ridge convention end to end."""
    _run("""
    blocks = blocks_of(cen_cfg())

    # ---- (a) EMA at the configured rate ----
    tau = 0.05
    a = agent_for(cen_cfg(measure_density_z_sigma_tau=tau))
    bj1 = jb(make_batch(B=32, seed=3))
    u1 = GCIQLFlowAgent._density_queries(bj1['next_observations'], blocks)
    B_pre = b_of(a, u1)
    mu0, cov0 = moments_of(a)
    assert np.allclose(mu0, 0.0) and np.allclose(cov0, np.eye(D)), \\
        "mu must init to 0 and C to I (no cold-start blow-up)"
    a1, _ = a.update(bj1)
    mu1, cov1 = moments_of(a1)
    mu_b = B_pre.mean(0)
    cov_b = (B_pre - mu_b[None]).T @ (B_pre - mu_b[None]) / len(B_pre)
    assert np.allclose(mu1, tau * mu_b, atol=1e-5), (mu1[:3], (tau * mu_b)[:3])
    assert np.allclose(cov1, (1 - tau) * np.eye(D) + tau * cov_b, atol=1e-5)

    # ---- (b) centered, 1/M, and NOT spec 29's uncentered object ----
    a2 = agent_for(cen_cfg(measure_density_z_sigma_tau=1.0))
    a2, _ = a2.update(bj1)
    mu2, cov2 = moments_of(a2)
    assert np.allclose(mu2, mu_b, atol=1e-5)
    assert np.allclose(cov2, np.cov(B_pre.T, bias=True), atol=1e-5)
    uncentered = B_pre.T @ B_pre / len(B_pre)
    assert not np.allclose(cov2, uncentered, atol=1e-3), \\
        "centered and uncentered coincide here -- the check cannot discriminate"
    assert not np.allclose(cov2, np.cov(B_pre.T, bias=False), atol=1e-6), \\
        "1/M and 1/(M-1) coincide here -- the check cannot discriminate"

    # ---- (c) symmetry + the end-to-end index convention ----
    assert np.abs(cov2 - cov2.T).max() == 0.0, "C_hat must be symmetrized every update"
    goals = bj1['actor_goals']
    kg = kg_ref(goals, bj1['next_observations'], blocks)
    ref, _ = cen_ref(b_of(a2, u1), kg, mu2, cov2)
    got = np.asarray(a2._density_z_encode(goals, u1, jnp.asarray(kg)), np.float64)
    assert np.allclose(got, ref, atol=1e-4), (got[0], ref[0])

    # the geometry + sync instruments, and the gauge ratio the spec-30 claim rests on
    a3, info = agent_for(cen_cfg(measure_density_z_sigma_tau=1.0)).update(bj1)
    ev = np.clip(np.linalg.eigvalsh(moments_of(a3)[1]), 0.0, None)
    assert np.isclose(float(info['measure/density_zcov_eff_rank']),
                      ev.sum() ** 2 / (ev ** 2).sum(), rtol=1e-3)
    assert np.isclose(float(info['measure/density_zcov_cond']),
                      ev[-1] / max(ev[0], 1e-12), rtol=1e-2)
    assert np.isclose(float(info['measure/density_zcov_min_eig']), ev[0], rtol=1e-3)
    assert np.isclose(float(info['measure/density_zmu_norm']),
                      np.linalg.norm(moments_of(a3)[0]), rtol=1e-4)
    # at tau=1 the EMA IS the batch, so both sync residuals are ~0 (they are the
    # EMA-vs-current-batch lag the contract asks to watch)
    assert float(info['measure/density_zc_sync_mu']) < 1e-4
    assert float(info['measure/density_zc_sync_cov']) < 1e-3
    a4, info4 = agent_for(cen_cfg(measure_density_z_sigma_tau=0.005)).update(bj1)
    assert float(info4['measure/density_zc_sync_cov']) > 1e-2, \\
        "a slow EMA must show a nonzero sync residual on the first step"

    # the sphere and dual arms log NONE of the spec-30 keys (no silent key drift)
    for c_ in (z_cfg(), dual_cfg()):
        _, i_ = agent_for(c_).update(bj1)
        assert not any('zcov' in k or 'zmoments' in k or '_zc_' in k for k in i_), \\
            sorted(k for k in i_ if 'zc' in k)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. train/eval identity + the NO-SILENT-SHORTCUT eval rule
# --------------------------------------------------------------------------
def test_eval_goes_through_the_reward_sample_api():
    """The rule spec 30a makes a hard requirement: on the reward-sample arm the native goal
    eval MUST go through the reward-sample API over an M-row kernel pool. Spec 29's .812
    came from the analytic point prompt at eval while training encoded through the sample
    path, which is exactly why it could not be read as general-encoder evidence.

    (a) sample_actions' index == the reward-sample encoder over (pool, kg(goal, pool)) with
        M = measure_density_z_eval_pool_m, and it is NOT the point prompt;
    (b) the pool is FROZEN after the first update and stored/serialized -- perturbing it
        moves the actions, a checkpoint round-trip reproduces them bit-for-bit, and further
        updates on other batches do not change the eval index at fixed params;
    (c) perturbing mu_hat / C_hat moves the eval actions (the eval path reads the stored
        moments, not batch statistics);
    (d) the point-prompt arm evaluates through the point prompt, and
        `sample_actions_point` exposes that shortcut as a SEPARATE diagnostic on the
        sample arm."""
    _run("""
    import pickle
    cfg = cen_cfg(measure_density_z_actor_reward='kernel')
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))

    obs = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    gl = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))
    key = jax.random.PRNGKey(0)

    # (a) the eval index IS the reward-sample encoder over the M-row pool
    pool_raw, filled = pool_of(a)
    assert filled == 1.0 and pool_raw.shape == (POOL_M, 3), (filled, pool_raw.shape)
    u_pool = GCIQLFlowAgent._scale_block_cols(jnp.asarray(pool_raw, jnp.float32), blocks)
    kg_pool = np.asarray(GCIQLFlowAgent._delta_phi(gl, jnp.asarray(pool_raw, jnp.float32),
                                                   blocks), np.float64)
    z_ref = np.asarray(a.density_z_from_reward_samples(u_pool, jnp.asarray(kg_pool, jnp.float32)),
                       np.float64)
    z_eval = np.asarray(a._density_z_encode(gl), np.float64)
    assert np.allclose(z_eval, z_ref, atol=1e-5), \\
        "sample_actions' index is not the reward-sample encoder over the eval pool"
    z_point = np.asarray(a.density_z_point_prompt(gl), np.float64)
    assert not np.allclose(z_eval, z_point, atol=1e-2), \\
        "the eval index equals the POINT prompt -- the silent shortcut spec 30 forbids"
    noise = jax.random.normal(key, (4, ACT_D))
    ref_ac = np.asarray(jnp.clip(a.network.select('actor')(obs, jnp.asarray(z_eval, jnp.float32),
                                                          noise), -1, 1))
    ac = np.asarray(a.sample_actions(obs, gl, seed=key))
    assert np.allclose(ac, ref_ac, atol=1e-6), "sample_actions is not conditioning on zhat"

    # (b) the pool is frozen, load-bearing and serialized
    for s in range(5, 9):
        a_more, _ = a.update(jb(make_batch(B=32, seed=s)))
        assert np.allclose(pool_of(a_more)[0], pool_raw, atol=0), \\
            "the eval pool was rewritten after the first update (it must be frozen)"
    p2 = dict(a.network.params)
    p2['modules_density_zpool'] = dict(p2['modules_density_zpool'])
    p2['modules_density_zpool']['pool'] = jnp.asarray(
        np.asarray(pool_raw, np.float32) + 1.0)
    a2 = a.replace(network=a.network.replace(params=p2))
    assert not np.allclose(ac, np.asarray(a2.sample_actions(obs, gl, seed=key)), atol=1e-6), \\
        "perturbing the stored eval pool did not move the actions"
    blob = pickle.dumps(jax.device_get(a.network.params))
    a3 = a.replace(network=a.network.replace(params=pickle.loads(blob)))
    assert (np.asarray(a3.sample_actions(obs, gl, seed=key)) == ac).all(), \\
        "actions changed across a checkpoint round-trip"
    a4 = a_more.replace(network=a_more.network.replace(params=a.network.params))
    assert (np.asarray(a4.sample_actions(obs, gl, seed=key)) == ac).all(), \\
        "the eval path depends on the last batch, not on the stored params"

    # (c) the stored moments are load-bearing at eval -- with the asymmetry the two APIs
    # are supposed to have: C_hat drives the reward-sample index, while mu_hat does NOT
    # (m_c is independent of mu because sum_j (r_j - rbar) = 0), and on the point arm it is
    # the other way round.
    def with_moment(agent, leaf, val):
        p = dict(agent.network.params)
        p['modules_density_zmoments'] = dict(p['modules_density_zmoments'])
        p['modules_density_zmoments'][leaf] = jnp.asarray(val, jnp.float32)
        return agent.replace(network=agent.network.replace(params=p))

    assert not np.allclose(ac, np.asarray(with_moment(
        a, 'cov', np.diag(np.linspace(1.0, 50.0, D))).sample_actions(obs, gl, seed=key)),
        atol=1e-6), "perturbing C_hat did not move sample_actions (batch stats?)"
    assert np.allclose(ac, np.asarray(with_moment(
        a, 'mu', np.linspace(1.0, 2.0, D)).sample_actions(obs, gl, seed=key)), atol=1e-5), \\
        "the reward-sample eval index must be independent of mu_hat"

    # unbatched single-eval-step goal (the in-training eval loop's shape)
    a1 = np.asarray(a.sample_actions(obs[0], gl[0], seed=jax.random.PRNGKey(1)))
    assert a1.shape == (ACT_D,) and np.isfinite(a1).all()

    # (d) the point arm evaluates through the point prompt; the diagnostic entrypoint works
    ap = agent_for(point_cfg(measure_density_z_actor_reward='kernel'))
    for s in range(3):
        ap, _ = ap.update(jb(make_batch(B=32, seed=s)))
    assert 'modules_density_zpool' not in ap.network.params, \\
        "the point arm must not register an eval pool"
    zp = np.asarray(ap._density_z_encode(gl), np.float64)
    assert np.allclose(zp, np.asarray(ap.density_z_point_prompt(gl), np.float64), atol=1e-6)
    ac_pt = np.asarray(ap.sample_actions(obs, gl, seed=key))
    assert not np.allclose(ac_pt, np.asarray(with_moment(
        ap, 'mu', np.linspace(1.0, 2.0, D)).sample_actions(obs, gl, seed=key)), atol=1e-6), \\
        "on the POINT arm mu_hat must be load-bearing at eval"
    ac_diag = np.asarray(a.sample_actions_point(obs, gl, seed=key))
    ref_diag = np.asarray(jnp.clip(a.network.select('actor')(
        obs, jnp.asarray(z_point, jnp.float32), noise), -1, 1))
    assert np.allclose(ac_diag, ref_diag, atol=1e-6)
    assert not np.allclose(ac_diag, ac, atol=1e-6), \\
        "the diagnostic point-shortcut eval coincides with the reward-sample eval -- it " \\
        "would not be a separate column"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 9. lattice: every inert or unsupported combination refused
# --------------------------------------------------------------------------
def test_lattice_refuses_inert_and_unsupported_combinations():
    """Same fake-NULL discipline as specs 25/28/29: no knob may be set where it is read
    nowhere, and no dose may silently turn the centered arm back into another arm while it
    logs as centered."""
    _run("""
    def refuses(cfg):
        try:
            agent_for(cfg)
        except AssertionError:
            return True
        return False

    # the slow-EMA index tower is mandatory (ridge through a drifting B)
    assert refuses(z_cfg(measure_density_z_encode='dual_centered'))
    assert refuses(z_cfg(measure_density_z_encode='dual_centered',
                         measure_density_z_enc_tau=0.0))

    # scope: read only on the z path
    assert refuses(g_cfg(measure_density_z_encode='dual_centered'))
    assert refuses(g_cfg(measure_density_z_prompt='point'))
    assert refuses(g_cfg(measure_density_z_eval_pool_m=64))
    assert refuses(g_cfg(measure_density_z_cov_eig_floor=1e-3))
    assert refuses(g_cfg(measure_density_z_const_eps=1e-3))
    assert refuses(z_cfg(measure_density_z_prompt='point')), \\
        "prompt is read only under the centered encode"
    assert refuses(dual_cfg(measure_density_z_prompt='point')), \\
        "prompt is inert under the UNCENTERED dual encode"
    assert refuses(dual_cfg(measure_density_z_cov_eig_floor=1e-3))
    assert refuses(dual_cfg(measure_density_z_const_eps=1e-3))
    assert refuses(cen_cfg(measure_density_z_prompt='point',
                           measure_density_z_eval_pool_m=64)), \\
        "the eval pool is inert on the point arm"

    # no dose may zero the arm
    assert refuses(cen_cfg(measure_density_z_sigma_tau=0.0)), \\
        "tau=0 freezes mu=0, C=I -- 'centered' would BE the sphere index"
    assert refuses(cen_cfg(measure_density_z_ridge_lambda_mult=0.0))
    assert refuses(cen_cfg(measure_density_z_cov_eig_floor=0.0))
    assert refuses(cen_cfg(measure_density_z_const_eps=0.0))
    assert refuses(cen_cfg(measure_density_z_eval_pool_m=1)), \\
        "an M=1 eval pool would hit the raise on every eval step"
    assert refuses(cen_cfg(measure_density_z_eval_pool_m=4096)), \\
        "a pool larger than the batch cannot be drawn from a batch"

    # enum hygiene
    assert refuses(z_cfg(measure_density_z_encode='centered'))
    assert refuses(cen_cfg(measure_density_z_prompt='dirac'))
    assert refuses(cen_cfg(measure_density_z_prompt=True))

    # the duplicate instrument is scoped to the density path, and its eps to the instrument
    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_density_dup_instrument=True))
    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_density_dup_eps=0.5))
    assert refuses(g_cfg(measure_density_dup_eps=0.5)), "eps without the instrument"
    assert refuses(g_cfg(measure_density_dup_instrument=True, measure_density_dup_eps=0.0))

    # accepted: both wave-1 arms and the neighbours they must not disturb
    assert not refuses(cen_cfg(measure_density_z_actor_reward='kernel')), "ra2_cs_centdual"
    assert not refuses(point_cfg(measure_density_z_actor_reward='kernel')), "ra2_cs_centpoint"
    assert not refuses(cen_cfg(measure_density_z_actor_reward='kernel',
                               measure_density_dup_instrument=True))
    assert not refuses(cen_cfg()), "centered + latent readout"
    assert not refuses(dual_cfg()), "spec 29 arm untouched"
    assert not refuses(z_cfg()), "spec 24 arm untouched"
    assert not refuses(g_cfg()), "g-indexed arm untouched"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 10. both wave-1 arms: real training, live gradients, the eval entrypoint
# --------------------------------------------------------------------------
def test_both_arms_end_to_end():
    """30 real update() steps for ra2_cs_centdual and ra2_cs_centpoint: finite metrics,
    no NaNs, towers and actor moved, mu_hat/C_hat moved off their init, z_cos_spread (the
    mid-run instrument the kill gate reads) logged and finite, and sample_actions returns
    finite in-range actions. Plus: switching the encode CHANGES the actor gradient (the
    index swap is not inert), and no gradient reaches the moments or the pool -- they are
    statistics, written by their EMA alone."""
    _run("""
    obs_e = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    g_e = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))

    cfg0 = dict(get_config())
    assert cfg0['measure_density_z_encode'] == 'sphere'
    assert cfg0['measure_density_z_prompt'] == 'sample'
    assert isinstance(cfg0['measure_density_z_prompt'], str), "must be an enum, not a dose"
    assert cfg0['measure_density_dup_instrument'] is False

    for label, c, keys in (
        ('ra2_cs_centdual', cen_cfg(measure_density_z_actor_reward='kernel',
                                    measure_density_dup_instrument=True),
         ('measure/density_ce', 'measure/z_cos_spread', 'measure/density_w_ztv',
          'measure/density_zcov_eff_rank', 'measure/density_zcov_cond',
          'measure/density_zcov_min_eig', 'measure/density_zmu_norm',
          'measure/density_zc_sync_mu', 'measure/density_zc_sync_cov',
          'measure/density_zc_mc_norm', 'measure/density_zc_c_norm',
          'measure/density_zc_r_var', 'measure/density_zc_fallback_frac',
          'measure/density_dup_exact_frac', 'measure/density_dup_eps_frac',
          'measure/density_dup_irreducible_ce',
          'actor/actor_loss', 'actor/q_mask_mean', 'actor/q_mask_frac')),
        ('ra2_cs_centpoint', point_cfg(measure_density_z_actor_reward='kernel',
                                       measure_density_dup_instrument=True),
         ('measure/density_ce', 'measure/z_cos_spread',
          'measure/density_zcov_eff_rank', 'measure/density_zc_point_norm',
          'measure/density_dup_eps_frac',
          'actor/actor_loss', 'actor/q_mask_mean', 'actor/q_mask_frac')),
    ):
        a = agent_for(c)
        p0 = a.network.params
        for step in range(30):
            a, info = a.update(jb(make_batch(B=32, seed=step)))
            for k in keys:
                assert np.isfinite(float(info[k])), (label, step, k, float(info[k]))
        assert all(np.isfinite(np.asarray(l)).all()
                   for l in jax.tree_util.tree_leaves(a.network.params)), \\
            label + ": NaN after 30 updates"
        mu, cov = moments_of(a)
        assert not np.allclose(mu, 0.0) and not np.allclose(cov, np.eye(D)), \\
            label + ": the centered moments never moved"
        for mod in ('density_f', 'density_b', 'actor'):
            assert any(not np.allclose(np.asarray(x), np.asarray(y), atol=0)
                       for x, y in zip(
                           jax.tree_util.tree_leaves(p0['modules_' + mod]),
                           jax.tree_util.tree_leaves(a.network.params['modules_' + mod]))), \\
                label + ": " + mod + " did not move after 30 updates"
        ac = np.asarray(a.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        assert ac.shape == (4, ACT_D) and np.isfinite(ac).all() \\
            and (np.abs(ac) <= 1.0 + 1e-6).all(), (label, ac.shape)

    # the encode/prompt knobs move the actor gradient (not inert)
    bj = jb(make_batch(B=32, seed=21))
    rng = jax.random.PRNGKey(9)

    def actor_grad(cfg_d):
        a = agent_for(cfg_d)
        g = jax.grad(lambda p: a.actor_loss(bj, p, rng)[0])(a.network.params)
        return np.concatenate([np.asarray(x).ravel()
                               for x in jax.tree_util.tree_leaves(g['modules_actor'])])

    g_dual = actor_grad(dual_cfg(measure_density_z_actor_reward='kernel'))
    g_cen = actor_grad(cen_cfg(measure_density_z_actor_reward='kernel'))
    g_pt = actor_grad(point_cfg(measure_density_z_actor_reward='kernel'))
    assert np.abs(g_cen).max() > 0 and np.abs(g_pt).max() > 0
    assert not np.allclose(g_dual, g_cen, atol=1e-7), "centering did not change the actor gradient"
    assert not np.allclose(g_cen, g_pt, atol=1e-7), "the two APIs give the same actor gradient"

    # ... and no gradient flows INTO the moments or the pool
    a = agent_for(cen_cfg(measure_density_z_actor_reward='kernel'))
    gr = jax.grad(lambda p: a.total_loss(bj, p, rng)[0])(a.network.params)
    for mod, leaves in (('modules_density_zmoments', ('mu', 'cov')),
                        ('modules_density_zpool', ('pool', 'filled'))):
        for lf in leaves:
            assert np.abs(np.asarray(gr[mod][lf])).max() == 0.0, \\
                (mod, lf, "received a gradient -- it must be written by its EMA alone")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 11. spec 30a Item 4: the duplicate-query instrument
# --------------------------------------------------------------------------
def test_duplicate_query_instrument():
    """Instrumentation only (no fix): per-batch exact- and eps-duplicate query fractions
    plus the irreducible-CE estimate E[log |class|] the hard immediate label would incur.
    Planted duplicates must be counted exactly; a batch with none must read 0; the loss
    itself must be UNCHANGED by the instrument (it only adds info keys); and with the knob
    off no key appears at all."""
    _run("""
    cfg = g_cfg(measure_density_dup_instrument=True)
    a = agent_for(cfg)
    b = make_batch(B=16, seed=4)

    # -- no duplicates: both fractions 0, irreducible CE 0 --
    _, i0 = a.update(jb(b))
    assert float(i0['measure/density_dup_exact_frac']) == 0.0
    assert float(i0['measure/density_dup_eps_frac']) == 0.0
    assert float(i0['measure/density_dup_irreducible_ce']) == 0.0

    # -- plant EXACT duplicates: rows 1,2,3 = row 0 and row 5 = row 4 --
    b2 = {k: v.copy() for k, v in b.items()}
    for i in (1, 2, 3):
        b2['next_observations'][i] = b2['next_observations'][0]
    b2['next_observations'][5] = b2['next_observations'][4]
    _, i2 = a.update(jb(b2))
    # class sizes: four rows of size 4, two rows of size 2, ten singletons
    exp_frac = 6.0 / 16.0
    exp_ce = (4 * np.log(4) + 2 * np.log(2)) / 16.0
    assert abs(float(i2['measure/density_dup_exact_frac']) - exp_frac) < 1e-6, \\
        float(i2['measure/density_dup_exact_frac'])
    assert abs(float(i2['measure/density_dup_eps_frac']) - exp_frac) < 1e-6
    assert abs(float(i2['measure/density_dup_irreducible_ce']) - exp_ce) < 1e-5
    assert abs(float(i2['measure/density_dup_irreducible_ce_exact']) - exp_ce) < 1e-5
    gamma = float(dict(get_config())['discount'])
    assert abs(float(i2['measure/density_dup_irreducible_ce_weighted'])
               - (1 - gamma) * exp_ce) < 1e-5
    assert abs(float(i2['measure/density_dup_eps_class_mean'])
               - (4 * 4 + 2 * 2 + 10) / 16.0) < 1e-5

    # -- eps-duplicates that are NOT exact: nudge a row inside the radius --
    b3 = {k: v.copy() for k, v in b.items()}
    b3['next_observations'][1, 19:22] = b3['next_observations'][0, 19:22] + 0.4 * 0.01
    _, i3 = a.update(jb(b3))
    assert float(i3['measure/density_dup_exact_frac']) == 0.0, "not an exact duplicate"
    assert abs(float(i3['measure/density_dup_eps_frac']) - 2.0 / 16.0) < 1e-6, \\
        float(i3['measure/density_dup_eps_frac'])
    a_tight = agent_for(g_cfg(measure_density_dup_instrument=True,
                              measure_density_dup_eps=1e-4))
    _, i4 = a_tight.update(jb(b3))
    assert float(i4['measure/density_dup_eps_frac']) == 0.0, \\
        "measure_density_dup_eps is not being read"

    # -- instrumentation ONLY: the loss is identical with and without the knob --
    bj = jb(b2)
    a_on = agent_for(g_cfg(measure_density_dup_instrument=True))
    a_off = agent_for(g_cfg())
    l_on, i_on = a_on.total_loss(bj, a_on.network.params, jax.random.PRNGKey(3))
    l_off, i_off = a_off.total_loss(bj, a_off.network.params, jax.random.PRNGKey(3))
    assert float(l_on) == float(l_off), (float(l_on), float(l_off))
    assert not any('dup' in k for k in i_off), sorted(k for k in i_off if 'dup' in k)
    assert sum('dup' in k for k in i_on) == 6, sorted(k for k in i_on if 'dup' in k)

    # -- it also runs in HARD mode, where the irreducible CE is the operative number --
    a_h = agent_for(g_cfg(measure_density_hard=True, measure_density_dup_instrument=True))
    _, ih = a_h.update(jb(b2))
    assert abs(float(ih['measure/density_dup_irreducible_ce']) - exp_ce) < 1e-5
    print("OK")
    """)
