"""Spec 29 R-A -- the dual-corrected index.

Contract: `PAPER/Multistep-FB/specs/29-ra-dual-index.md` section 1 + its implementation
contract.

Density-FB's softmax normalization destroys FB's linear identity Q_r = F^T E[rB], and the
CE pins only the PRODUCT F^T B, so B's Euclidean geometry is gauge-arbitrary: spec 28 P0
measured Sigma_B at eff rank 1.0002/50 and cond 1e6-1e7, which makes the shipped index
z(g) = sqrt(d)*normalize(B̄(g)) a collapsed PRIMAL feature vector used where the algebra
needs a DUAL coefficient. One default-off enum knob conditions on the dual instead:

  `measure_density_z_encode = 'dual'`
      zhat = sqrt(d) * normalize( (Sigma_hat_B + lambda I)^-1 m_hat ),
      m_hat = (1/M) sum_j B̄(u_j) r_j  over the row's reward samples
              (hindsight rows: r_j = kg_j = _delta_phi(goals, u_raw), the QUERY-SET
               NORMALIZED kernel -- matched exactly to the g path, standing spec-28 rule),
      lambda = measure_density_z_ridge_lambda_mult * tr(Sigma_hat)/d.

The five contract requirements are each a scar, and each has a test here:
  1. Sigma_hat is EMA-TRACKED and stored as an ORDINARY PARAM LEAF (the density_b_zenc
     pattern), uncentered second moment, matching `spec28_encodings.py`'s pool
     computation -- asserted against that module directly (test 3). Batch statistics are
     forbidden: they reintroduce the train/eval mismatch that broke the first z build,
     where 8/8 unit tests passed while the eval path was wrong because nothing drove
     sample_actions. So this suite drives update() AND sample_actions throughout, and
     test 4 proves sample_actions reads the stored matrix itself.
  2. B̄ for the encode is the slow-EMA tower: dual REQUIRES enc_tau > 0 (test 6).
  3. lambda = 0.01*tr(Sigma)/d default, knob over {.001,.1}; eff_rank/cond logged (test 7).
  4. Eval-time zhat comes from the SAME stored object (test 4), and the general
     reward-sample interface is the harness path (test 8).
  5. Task mixture unchanged; sphere rows keep being masked by the existing z_hindsight
     machinery (test 9).

Same subprocess-jax harness as tests/test_gciql_spec28_actor_reward.py: OBS_D=40, action
dim 5, d=8, hit_spec '19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_spec29_dual_index.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import subprocess
import textwrap

import numpy as np
import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8    # value_latent_dim AND measure_density_dim, matches the spec-18/24/25/28 harness
ACT_D = 5

# The last commit BEFORE spec 29's implementation. It has specs 24/25/28 and none of the
# spec-29 knobs, so it permanently marks the arm this spec must reproduce bit-for-bit at
# defaults -- pinned, not HEAD, because HEAD moves forward on this shared branch as other
# sessions land commits (same discipline as tests/test_gciql_spec28_actor_reward.py).
PRE_SPEC29_SHA = "6e640065ee3bd363576df3c8f18597efbc4916f9"

NEW_KNOBS = ("measure_density_z_encode", "measure_density_z_ridge_lambda_mult",
             "measure_density_z_sigma_tau")


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
        # spec 24's z-indexed arm at its shipped defaults (sphere encode).
        return g_cfg(measure_density_z=True, **kw)

    def dual_cfg(**kw):
        # the spec-29 arm: dual encode + the REQUIRED slow-EMA index tower.
        kw.setdefault("measure_density_z_enc_tau", 0.0005)
        return z_cfg(measure_density_z_encode="dual", **kw)

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

    def spec28():
        # the JAX-FREE numpy core the P0 probes scored the encodings with -- the whole
        # point of asserting against IT is that the training-time index must be the same
        # object those probes measured, not a re-derivation that happens to look similar.
        import importlib.util, os
        p = os.path.join(REPO_DIR, "scripts", "rescuing_fb", "spec28_encodings.py")
        spec = importlib.util.spec_from_file_location("spec28_encodings", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def b_of(agent, x_scaled, mod='density_b_zenc'):
        return np.asarray(agent.network.select(mod)(x_scaled), np.float64)

    def sigma_of(agent):
        return np.asarray(agent.network.params['modules_density_zsigma']['sigma'], np.float64)

    def kg_ref(goals, nobs, blocks):
        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        return np.asarray(GCIQLFlowAgent._delta_phi(goals, u_raw, blocks), np.float64)
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


# --------------------------------------------------------------------------
# 1. all-defaults bit-identity to the pre-spec-29 agent
# --------------------------------------------------------------------------
def test_default_off_bit_identical_to_pre_spec29():
    """measure_density_z_encode at its default 'sphere': param tree, total_loss, THREE
    REAL update() steps and sample_actions must all be bit-identical to the agent loaded
    from the pinned pre-spec-29 commit -- on the spec-24 z arm, the spec-25 C3 arm (which
    already registers density_b_zenc, the pattern spec 29 copies), the spec-28 cell-(ii)
    arm, the g-indexed arm and a config with no measure knobs at all.

    Real update steps plus the EVAL entrypoint, not a construct-only check: a previous z
    build passed 8/8 unit tests while its eval path was broken because nothing drove
    sample_actions."""
    if not _baseline_reachable(PRE_SPEC29_SHA):
        pytest.skip(f"pinned baseline {PRE_SPEC29_SHA} is not reachable in this "
                    "checkout (e.g. a shallow clone) -- cannot run the bit-identity diff")
    _run(f"""
    PRE_SPEC29_SHA = {PRE_SPEC29_SHA!r}
    NEW_KNOBS = {NEW_KNOBS!r}
    """ + """
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", PRE_SPEC29_SHA + ":tools/wandb_mode_shim/gciql_flow.py"],
        cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    assert "measure_density_z_actor_reward" in baseline_src, \\
        "pinned baseline predates spec 28 -- re-pin PRE_SPEC29_SHA"
    for k29 in NEW_KNOBS:
        assert k29 not in baseline_src, (
            "pinned baseline already contains " + k29 + " -- the pin has gone stale; "
            "re-pin PRE_SPEC29_SHA to the last commit before the spec-29 implementation")
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_pre29.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_pre29", baseline_path)
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
                       "spec-25 C3 arm (density_b_zenc registered)")
    same_after_updates(dict(measure_density_z=True,
                            measure_density_z_actor_reward='kernel'),
                       "spec-28 cell (ii)")
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
# 2. the index IS the ridge coefficient (and NOT the sphere index)
# --------------------------------------------------------------------------
def test_dual_index_is_the_ridge_coefficient():
    """zhat = sqrt(d)*normalize((Sigma_hat+lambda I)^-1 m_hat), checked against an
    independent numpy solve; ||zhat|| = sqrt(d) exactly (the standing invariant the whole
    conditioning distribution assumes); and the two structural degeneracies:
      * at Sigma_hat = I (its init) the dual reduces to the SPHERE direction -- so the
        arm cannot be silently different for reasons other than the geometry;
      * at a NON-identity Sigma_hat it genuinely differs from the sphere index (otherwise
        the knob would be an expensive no-op)."""
    _run("""
    cfg = dual_cfg()
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    b = make_batch(B=12, seed=5); bj = jb(b)
    goals = bj['actor_goals']

    d = float(D)
    lam_mult = float(cfg['measure_density_z_ridge_lambda_mult'])

    # --- Sigma_hat = I at init => dual == sphere direction ---
    sig0 = sigma_of(a)
    assert np.allclose(sig0, np.eye(D)), ("Sigma_hat must init to I", sig0)
    x_sc = GCIQLFlowAgent._density_queries(goals, blocks)
    b_g = b_of(a, x_sc)
    z_dual0 = np.asarray(a._density_z_encode(goals), np.float64)
    z_sphere = np.asarray(_sphere(jnp.asarray(b_g), d), np.float64)
    assert np.allclose(z_dual0, z_sphere, atol=1e-4), \\
        "at Sigma=I the ridge solve must reduce to the sphere direction"

    # --- after a real update Sigma_hat is a real second moment; recompute by hand ---
    a, _ = a.update(jb(make_batch(B=32, seed=1)))
    sig = sigma_of(a)
    assert not np.allclose(sig, np.eye(D)), "Sigma_hat did not move off its init"
    lam = lam_mult * np.trace(sig) / D
    m_hat = b_of(a, x_sc)                       # no reward samples => Dirac moment B̄(x~_g)
    c = np.linalg.solve(sig + lam * np.eye(D), m_hat.T).T
    ref = np.sqrt(d) * c / np.linalg.norm(c, axis=-1, keepdims=True)
    got = np.asarray(a._density_z_encode(goals), np.float64)
    assert np.allclose(got, ref, atol=1e-4), (got[0], ref[0])
    assert np.allclose(np.linalg.norm(got, axis=-1), np.sqrt(d), atol=1e-4), \\
        ("||zhat|| = sqrt(d) must be re-imposed after the solve",
         np.linalg.norm(got, axis=-1))

    # --- and it is NOT the sphere index any more (no-op guard) ---
    z_sph = np.asarray(_sphere(jnp.asarray(m_hat), d), np.float64)
    cos = (got * z_sph).sum(-1) / (np.linalg.norm(got, -1) * np.linalg.norm(z_sph, -1))
    assert (cos < 0.999).any(), \\
        ("dual and sphere indices coincide -- the knob is not doing anything", cos)

    # --- with reward samples: m_hat = (1/M) sum_j B̄(u_j) kg_j ---
    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    kg = kg_ref(goals, bj['next_observations'], blocks)
    assert np.allclose(kg.sum(-1), 1.0, atol=1e-5), \\
        "reward samples must be the QUERY-SET NORMALIZED kernel (spec 28 parity rule)"
    m_k = kg @ b_of(a, u_sc) / kg.shape[-1]
    ck = np.linalg.solve(sig + lam * np.eye(D), m_k.T).T
    refk = np.sqrt(d) * ck / np.linalg.norm(ck, axis=-1, keepdims=True)
    gotk = np.asarray(a._density_z_encode(goals, u_sc, jnp.asarray(kg)), np.float64)
    assert np.allclose(gotk, refk, atol=1e-4), (gotk[0], refk[0])
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. Sigma_hat: EMA'd, uncentered, and EQUAL to spec28_encodings' pool computation
# --------------------------------------------------------------------------
def test_sigma_is_ema_uncentered_and_matches_spec28_pool():
    """Contract requirement 1, three ways.

    (a) It is an EMA with the configured rate: after one update at tau_sigma, Sigma_hat ==
        (1-tau)*I + tau * B^T B / M with B the PRE-STEP index tower on THIS batch's
        queries -- the pre-step source discipline the target/zenc soft-updates use.
    (b) UNCENTERED, 1/M: at tau_sigma = 1.0 the stored matrix equals spec28_encodings'
        own `sigma = b.T @ b / P` bit-for-bit in float64 tolerance. A centered or 1/(M-1)
        moment fails here, which is the point -- the training index must be the object
        the P0 probes scored.
    (c) The whole index agrees with spec28_encodings.ridge_index end to end: at
        tau_sigma = 1.0, zhat == sqrt(d)*normalize(ridge_index(b_pool, r_pool, lam)) with
        lam = ridge_lambdas(b_pool)[0.01]."""
    _run("""
    s28 = spec28()
    blocks = blocks_of(dual_cfg())

    # ---- (a) EMA at the configured rate ----
    tau = 0.05
    a = agent_for(dual_cfg(measure_density_z_sigma_tau=tau))
    b1 = make_batch(B=32, seed=3); bj1 = jb(b1)
    u1 = GCIQLFlowAgent._density_queries(bj1['next_observations'], blocks)
    B_pre = b_of(a, u1)                                   # PRE-step index tower
    a1, _ = a.update(bj1)
    exp = (1.0 - tau) * np.eye(D) + tau * (B_pre.T @ B_pre / B_pre.shape[0])
    assert np.allclose(sigma_of(a1), exp, atol=1e-5), (sigma_of(a1)[0, :3], exp[0, :3])

    # ---- (b) uncentered, 1/M, identical to spec28_encodings' pool sigma ----
    a2 = agent_for(dual_cfg(measure_density_z_sigma_tau=1.0))
    a2, _ = a2.update(bj1)
    sig = sigma_of(a2)
    assert np.allclose(sig, B_pre.T @ B_pre / len(B_pre), atol=1e-5)
    centered = np.cov(B_pre.T, bias=True)
    assert not np.allclose(sig, centered, atol=1e-3), \\
        "uncentered and centered moments coincide here -- the check cannot discriminate"
    unbiased = B_pre.T @ B_pre / (len(B_pre) - 1)
    assert not np.allclose(sig, unbiased, atol=1e-5), \\
        "1/M and 1/(M-1) coincide here -- the check cannot discriminate"

    # ---- (c) the full index equals spec28_encodings.ridge_index ----
    goals = bj1['actor_goals']
    kg = kg_ref(goals, bj1['next_observations'], blocks)
    b_pool = b_of(a2, u1)
    lam = s28.ridge_lambdas(b_pool, mults=(1e-2,))[0.01]
    got = np.asarray(a2._density_z_encode(goals, u1, jnp.asarray(kg)), np.float64)
    for i in range(len(goals)):
        c = s28.ridge_index(b_pool, kg[i], lam)
        ref = np.sqrt(float(D)) * c / np.linalg.norm(c)
        assert np.allclose(got[i], ref, atol=1e-4), (i, got[i], ref)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. train/eval identity: sample_actions reads the STORED Sigma_hat
# --------------------------------------------------------------------------
def test_eval_path_reads_the_stored_serialized_sigma():
    """The scar this exists for: the first z build passed 8/8 unit tests while
    `sample_actions` was broken, because no test drove it. So:

    (a) sample_actions' own index equals _density_z_encode's -- the object the training
        rows condition on -- computed from the restored params ALONE (no rng, no batch);
    (b) perturbing the stored Sigma_hat CHANGES the actions, i.e. the eval path really
        reads that leaf and not a recomputed batch statistic;
    (c) a serialization round-trip of the param tree (the checkpoint path) reproduces the
        actions bit-for-bit, and Sigma_hat survives it;
    (d) sample_actions is deterministic given params + seed: two calls with DIFFERENT
        preceding batches give identical actions (a batch statistic could not)."""
    _run("""
    import pickle
    cfg = dual_cfg()
    a = agent_for(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(seed=s)))

    obs = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    gl = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))
    key = jax.random.PRNGKey(0)

    # (a) the eval index is the training encode, from params alone
    z_eval = np.asarray(a._density_z_encode(gl), np.float64)
    noise = jax.random.normal(key, (4, ACT_D))
    ref = np.asarray(jnp.clip(a.network.select('actor')(obs, jnp.asarray(z_eval), noise),
                              -1, 1))
    ac = np.asarray(a.sample_actions(obs, gl, seed=key))
    assert np.allclose(ac, ref, atol=1e-6), "sample_actions is not conditioning on zhat"

    # (b) the stored Sigma_hat is load-bearing at eval
    p2 = dict(a.network.params)
    p2['modules_density_zsigma'] = {'sigma': jnp.asarray(
        np.diag(np.linspace(1.0, 50.0, D)).astype(np.float32))}
    a2 = a.replace(network=a.network.replace(params=p2))
    ac2 = np.asarray(a2.sample_actions(obs, gl, seed=key))
    assert not np.allclose(ac, ac2, atol=1e-6), \\
        "perturbing the stored Sigma_hat did not move sample_actions -- the eval path is " \\
        "not reading the stored object (the train/eval mismatch the contract forbids)"

    # (c) checkpoint round-trip
    blob = pickle.dumps(jax.device_get(a.network.params))
    a3 = a.replace(network=a.network.replace(params=pickle.loads(blob)))
    assert np.allclose(sigma_of(a3), sigma_of(a), atol=0), "Sigma_hat did not serialize"
    assert (np.asarray(a3.sample_actions(obs, gl, seed=key)) == ac).all(), \\
        "actions changed across a checkpoint round-trip"

    # (d) no batch dependence
    a4 = a
    for s in range(5, 9):
        a4, _ = a4.update(jb(make_batch(seed=s)))
    a4 = a4.replace(network=a4.network.replace(params=a.network.params))
    assert (np.asarray(a4.sample_actions(obs, gl, seed=key)) == ac).all()

    # unbatched single-eval-step goal (the in-training eval loop's shape)
    a1 = np.asarray(a.sample_actions(obs[0], gl[0], seed=jax.random.PRNGKey(1)))
    assert a1.shape == (ACT_D,) and np.isfinite(a1).all()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. the sample-free encode is the M=1 (Dirac) member of the SAME family
# --------------------------------------------------------------------------
def test_dirac_limit_matches_the_reward_sample_encode():
    """`sample_actions` has a goal and no pool, so its moment is B̄(x~_g). That is not a
    second construction: it is the reward-sample moment at r = a Dirac on the goal. Pin
    it -- hand the encode a query set that CONTAINS each goal's own coords (so the
    query-set-normalized kernel is essentially one-hot there) and the two encodes must
    agree, while a kernel over an unrelated query set must NOT (else the check is
    vacuous)."""
    _run("""
    cfg = dual_cfg()
    a = agent_for(cfg)
    a, _ = a.update(jb(make_batch(B=32, seed=2)))
    blocks = blocks_of(cfg)

    goals = jnp.asarray(np.random.default_rng(17).normal(
        size=(6, OBS_D)).astype(np.float32) * 3.0)     # far apart in eps-scaled units
    # query set = the goals themselves -> kg is (near) the identity, i.e. a Dirac per row
    u_sc = GCIQLFlowAgent._density_queries(goals, blocks)
    kg = kg_ref(goals, goals, blocks)
    assert np.diag(kg).min() > 0.99, ("kernel is not sharp enough for a limit test",
                                      np.diag(kg))

    z_dirac = np.asarray(a._density_z_encode(goals), np.float64)
    z_samp = np.asarray(a._density_z_encode(goals, u_sc, jnp.asarray(kg)), np.float64)
    assert np.allclose(z_dirac, z_samp, atol=1e-3), (z_dirac[0], z_samp[0])

    # ... and a genuinely different reward set moves it (the check discriminates)
    other = jnp.asarray(np.random.default_rng(18).normal(size=(20, OBS_D)).astype(np.float32))
    u_o = GCIQLFlowAgent._density_queries(other, blocks)
    kg_o = kg_ref(goals, other, blocks)
    z_o = np.asarray(a._density_z_encode(goals, u_o, jnp.asarray(kg_o)), np.float64)
    assert not np.allclose(z_dirac, z_o, atol=1e-2), \\
        "an unrelated reward sample set gave the same index -- vacuous limit test"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5b. the encode runs through the SLOW-EMA tower, not the fast target tower
# --------------------------------------------------------------------------
def test_encode_runs_through_the_slow_ema_tower():
    """Contract requirement 2 as BEHAVIOR, not just a create() assert. `enc_tau > 0` is
    mandatory because a ridge solve through a fast-drifting B is the known
    non-stationarity risk -- but requiring the knob is worthless if the code then reads
    `target_density_b` anyway. Early in training the two towers are numerically close
    (both start at density_b), so a value comparison cannot separate them; a PERTURBATION
    can:
      * perturbing `density_b_zenc` MUST move the index (it is the encoder), and
      * perturbing `target_density_b` MUST NOT (it is not).
    This is the test that kills the 'encode through the fast tower' mutant, which the
    value-level checks do not catch.

    The Sigma_hat EMA is checked the same way, since it is fed by the same tower."""
    _run("""
    a = agent_for(dual_cfg())
    for s in range(3):
        a, _ = a.update(jb(make_batch(seed=s)))
    blocks = blocks_of(dual_cfg())
    goals = jnp.asarray(np.random.default_rng(31).normal(size=(8, OBS_D)).astype(np.float32))
    z0 = np.asarray(a._density_z_encode(goals), np.float64)

    def bumped(mod, scale=1.5):
        p = dict(a.network.params)
        p[mod] = jax.tree_util.tree_map(lambda x: x * scale, a.network.params[mod])
        return a.replace(network=a.network.replace(params=p))

    z_zenc = np.asarray(bumped('modules_density_b_zenc')._density_z_encode(goals), np.float64)
    z_tgt = np.asarray(bumped('modules_target_density_b')._density_z_encode(goals), np.float64)
    assert not np.allclose(z0, z_zenc, atol=1e-4), \\
        "perturbing density_b_zenc left the index unchanged -- the encode is NOT reading " \\
        "the slow-EMA tower the contract requires"
    assert np.allclose(z0, z_tgt, atol=1e-6), \\
        "perturbing target_density_b moved the index -- the encode is reading the FAST " \\
        "target tower (contract requirement 2 violated)"

    # the same tower feeds the Sigma_hat EMA
    s_zenc = sigma_of(bumped('modules_density_b_zenc').update(jb(make_batch(seed=9)))[0])
    s_tgt = sigma_of(bumped('modules_target_density_b').update(jb(make_batch(seed=9)))[0])
    s_ref = sigma_of(a.update(jb(make_batch(seed=9)))[0])
    assert not np.allclose(s_zenc, s_ref, atol=1e-5), \\
        "Sigma_hat is not accumulated from the slow-EMA tower"
    assert np.allclose(s_tgt, s_ref, atol=1e-6), \\
        "Sigma_hat is accumulated from the fast target tower"

    # and the eval entrypoint inherits it (the same object, end to end)
    obs = jnp.asarray(np.random.default_rng(32).normal(size=(8, OBS_D)).astype(np.float32))
    k = jax.random.PRNGKey(0)
    ac0 = np.asarray(a.sample_actions(obs, goals, seed=k))
    assert not np.allclose(ac0, np.asarray(
        bumped('modules_density_b_zenc').sample_actions(obs, goals, seed=k)), atol=1e-6)
    assert np.allclose(ac0, np.asarray(
        bumped('modules_target_density_b').sample_actions(obs, goals, seed=k)), atol=1e-6)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. lattice: dual requires the slow-EMA tower; every inert combination refused
# --------------------------------------------------------------------------
def test_lattice_refuses_inert_and_unsupported_combinations():
    """Contract requirement 2 is an ASSERT, not a comment: a ridge solve through the
    fast-drifting target tower is the known non-stationarity risk, so dual without
    enc_tau > 0 is refused at create(). Plus the standing fake-NULL guards: no knob may
    be set where it would be read nowhere, and no dose may silently turn the arm back
    into the sphere arm while it logs as dual."""
    _run("""
    def refuses(cfg):
        try:
            agent_for(cfg)
        except AssertionError:
            return True
        return False

    # -- requirement 2: the slow-EMA index tower is mandatory --
    assert refuses(z_cfg(measure_density_z_encode='dual')), \\
        "dual without enc_tau>0 must be refused (ridge through a drifting B)"
    assert refuses(z_cfg(measure_density_z_encode='dual',
                         measure_density_z_enc_tau=0.0)), "enc_tau=0 is the same refusal"

    # -- scope: read only on the z path --
    assert refuses(g_cfg(measure_density_z_encode='dual')), \\
        "dual outside the measure_density_z lattice would be inert"
    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_density_z_encode='dual')), "no measure_density at all"
    assert refuses(g_cfg(measure_density_z_ridge_lambda_mult=0.1)), \\
        "a dual-only knob set off the z lattice"
    assert refuses(g_cfg(measure_density_z_sigma_tau=0.05)), \\
        "a dual-only knob set off the z lattice"
    assert refuses(z_cfg(measure_density_z_ridge_lambda_mult=0.1)), \\
        "lambda is read only under the dual encode -- inert under sphere"
    assert refuses(z_cfg(measure_density_z_sigma_tau=0.05)), \\
        "tau_sigma is read only under the dual encode -- inert under sphere"

    # -- no dose may zero the arm --
    assert refuses(dual_cfg(measure_density_z_ridge_lambda_mult=0.0)), \\
        "lambda=0 is an unregularized solve against a rank-1-dominated Sigma"
    assert refuses(dual_cfg(measure_density_z_ridge_lambda_mult=-1.0))
    assert refuses(dual_cfg(measure_density_z_sigma_tau=0.0)), \\
        "tau_sigma=0 freezes Sigma at I -- dual would BE sphere while logging as dual"
    assert refuses(dual_cfg(measure_density_z_sigma_tau=1.5))

    # -- enum hygiene --
    assert refuses(z_cfg(measure_density_z_encode='ridge')), "unknown enum value"
    assert refuses(z_cfg(measure_density_z_encode=True)), "non-enum value"

    # -- accepted: both R-A arms and the ablation ends --
    assert not refuses(dual_cfg()), "R-A/latent"
    assert not refuses(dual_cfg(measure_density_z_actor_reward='kernel')), "R-A/kernel"
    for m in (0.001, 0.01, 0.1):
        assert not refuses(dual_cfg(measure_density_z_ridge_lambda_mult=m)), m
    assert not refuses(z_cfg()), "spec-24 sphere arm untouched"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. lambda scaling + the geometry instruments
# --------------------------------------------------------------------------
def test_lambda_scale_and_sigma_geometry_instruments():
    """lambda = mult * tr(Sigma_hat)/d -- a multiple of the MEAN EIGENVALUE, the exact
    convention of spec28_encodings.ridge_lambdas, so it is dimensionless and comparable
    across checkpoints. And eff_rank/cond -- the two statistics that DIAGNOSED the
    original failure -- are logged every step, with the same definitions the P0b probe
    uses, so the training curve and the probe are the same numbers."""
    _run("""
    s28 = spec28()
    blocks = blocks_of(dual_cfg())
    bj = jb(make_batch(B=32, seed=4))
    goals = bj['actor_goals']

    zs = {}
    for m in (0.001, 0.01, 0.1):
        a = agent_for(dual_cfg(measure_density_z_ridge_lambda_mult=m,
                               measure_density_z_sigma_tau=1.0))
        a, info = a.update(bj)
        sig = sigma_of(a)
        lam = m * np.trace(sig) / D
        x_sc = GCIQLFlowAgent._density_queries(goals, blocks)
        mh = b_of(a, x_sc)
        c = np.linalg.solve(sig + lam * np.eye(D), mh.T).T
        ref = np.sqrt(float(D)) * c / np.linalg.norm(c, axis=-1, keepdims=True)
        got = np.asarray(a._density_z_encode(goals), np.float64)
        assert np.allclose(got, ref, atol=1e-4), (m, got[0], ref[0])
        zs[m] = got

        # the instruments, against the P0b definitions
        ev = np.clip(np.linalg.eigvalsh(sig), 0.0, None)
        assert np.isclose(float(info['measure/density_zsigma_eff_rank']),
                          ev.sum() ** 2 / (ev ** 2).sum(), rtol=1e-3), m
        assert np.isclose(float(info['measure/density_zsigma_cond']),
                          ev[-1] / max(ev[0], 1e-12), rtol=1e-2), m
        assert np.isclose(float(info['measure/density_zsigma_trace_over_d']),
                          np.trace(sig) / D, rtol=1e-4), m

    assert not np.allclose(zs[0.001], zs[0.1], atol=1e-3), \\
        "lambda has no effect -- the sweep would be vacuous"

    # the sphere arm logs NONE of the spec-29 instruments (no silent key drift)
    a_s = agent_for(z_cfg())
    _, i_s = a_s.update(bj)
    assert not any('zsigma' in k for k in i_s), sorted(k for k in i_s if 'zsigma' in k)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. the general reward interface (G29-3's entrypoint)
# --------------------------------------------------------------------------
def test_general_reward_sample_interface():
    """The binding user requirement: rewards specified as SAMPLES {(u_j, r_j)}, not
    goals. `density_z_from_reward_samples(u_scaled, r)` is the harness path -- the same
    stored Sigma_hat, the same solve, an arbitrary reward family -- and its output feeds
    `sample_actions_with_z` directly. Checked against spec28_encodings.ridge_index on a
    reward that is NOT a goal kernel (a signed dense reward), plus the invariants:
    ||zhat|| = sqrt(d), different rewards give different indices, and a positive rescale
    of the reward leaves the index unchanged (a linear solve + a normalization)."""
    _run("""
    s28 = spec28()
    cfg = dual_cfg()
    a = agent_for(cfg)
    a, _ = a.update(jb(make_batch(B=32, seed=6)))
    blocks = blocks_of(cfg)

    pool_obs = jnp.asarray(np.random.default_rng(21).normal(size=(64, OBS_D)).astype(np.float32))
    u_sc = GCIQLFlowAgent._density_queries(pool_obs, blocks)
    r = np.random.default_rng(22).normal(size=(3, 64)).astype(np.float32)   # dense, signed

    z = np.asarray(a.density_z_from_reward_samples(u_sc, jnp.asarray(r)), np.float64)
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(float(D)), atol=1e-4)

    sig = sigma_of(a)
    b_pool = b_of(a, u_sc)
    lam = float(cfg['measure_density_z_ridge_lambda_mult']) * np.trace(sig) / D
    for i in range(3):
        m = (r[i][:, None] * b_pool).mean(0)
        c = np.linalg.solve(sig + lam * np.eye(D), m)
        ref = np.sqrt(float(D)) * c / np.linalg.norm(c)
        assert np.allclose(z[i], ref, atol=1e-4), (i, z[i], ref)

    assert not np.allclose(z[0], z[1], atol=1e-3), "different rewards gave the same index"
    z_scaled = np.asarray(a.density_z_from_reward_samples(u_sc, jnp.asarray(3.7 * r)),
                          np.float64)
    assert np.allclose(z, z_scaled, atol=1e-4), \\
        "a positive rescale of the reward moved the index -- the solve is not linear+norm"

    # and it drives the policy through the precomputed-z entrypoint
    obs = jnp.asarray(np.random.default_rng(23).normal(size=(3, OBS_D)).astype(np.float32))
    ac = np.asarray(a.sample_actions_with_z(obs, jnp.asarray(z, np.float32),
                                            seed=jax.random.PRNGKey(0)))
    assert ac.shape == (3, ACT_D) and np.isfinite(ac).all() and (np.abs(ac) <= 1 + 1e-6).all()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 9. task mixture unchanged; sphere rows still masked in the kernel arm
# --------------------------------------------------------------------------
def test_task_mixture_and_hindsight_mask_unchanged():
    """Contract requirement 5. The mixture is NOT a spec-29 variable: sphere rows still
    arrive at measure_density_z_rand_frac, they still carry ||z|| = sqrt(d), and the
    hindsight indicator still comes from the SAME single bernoulli draw as the mix (so
    mask and index can never disagree). In R-A/kernel those sphere rows -- which have no
    reward samples at all -- are masked out of the actor-q term and the term renormalized
    by the survivors, through the machinery spec 28 cell (ii) already built and tested."""
    _run("""
    cfg = dual_cfg(measure_density_z_actor_reward='kernel', bc_coeff=0.0)
    a = agent_for(cfg)
    blocks = blocks_of(cfg)
    bj = jb(make_batch(B=64, seed=8))
    rngA = jax.random.PRNGKey(4)
    z_rng = jax.random.split(rngA, 5)[4]     # actor_loss's 5-way split, z stream last

    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    kg = jnp.asarray(kg_ref(bj['actor_goals'], bj['next_observations'], blocks))
    z, hs = a._density_z_masked(bj['actor_goals'], z_rng, u_sc, kg)
    z = np.asarray(z); hs = np.asarray(hs)
    assert np.allclose(np.linalg.norm(z, axis=-1), np.sqrt(float(D)), atol=1e-4), \\
        "sphere rows and dual rows must BOTH carry ||z||=sqrt(d)"
    assert 0.0 < hs.mean() < 1.0, "degenerate draw -- the mask check would be vacuous"

    # the mix is exactly the shipped one: hindsight rows == the dual encode, and the
    # indicator selects precisely those rows.
    z_h = np.asarray(a._density_z_encode(bj['actor_goals'], u_sc, kg))
    keep = hs > 0.5
    assert np.allclose(z[keep], z_h[keep], atol=1e-5), "hindsight rows are not the encode"
    assert not np.allclose(z[~keep], z_h[~keep], atol=1e-3), \\
        "sphere rows equal the encode -- the mixture is not drawing spheres"

    # actor_loss reduces over the SURVIVING rows only, and logs the fraction
    l, info = a.actor_loss(bj, a.network.params, rngA)
    noise = jax.random.normal(jax.random.split(rngA, 5)[3], bj['actions'].shape)
    acts = a.network.select('actor')(bj['observations'], jnp.asarray(z), noise)
    q1, q2 = a._density_actor_q_z(bj, bj['observations'], acts, jnp.asarray(z),
                                  goals=bj['actor_goals'])
    q = np.minimum(np.asarray(q1), np.asarray(q2))
    masked = float((q * hs).sum() / hs.sum())
    assert np.allclose(float(l) - float(info['bc_flow_loss']), -masked, atol=1e-5), \\
        (float(l), float(info['bc_flow_loss']), -masked, -q.mean())
    assert not np.allclose(masked, q.mean(), atol=1e-6), "vacuous mask check"
    assert abs(float(info['q_mask_frac']) - hs.mean()) < 1e-6

    # rand_frac itself is untouched by the encode knob
    for rf in (0.0, 0.5):
        a_rf = agent_for(dual_cfg(measure_density_z_rand_frac=rf))
        _, h_rf = a_rf._density_z_masked(bj['actor_goals'], z_rng, u_sc, kg)
        assert abs(float(np.asarray(h_rf).mean()) - (1.0 - rf)) < 0.25, rf
    print("OK")
    """)


# --------------------------------------------------------------------------
# 10. both arms: real training, live gradients, the eval entrypoint
# --------------------------------------------------------------------------
def test_both_arms_end_to_end():
    """30 real update() steps for R-A/kernel and R-A/latent: finite metrics throughout,
    no NaNs, both density towers and the actor moved, Sigma_hat moved off its identity
    init, z_cos_spread (the train-time pairwise-cosine instrument G29-1 reads) is logged
    and finite, and sample_actions still returns finite in-range actions -- the
    entrypoint a construct-only suite would miss. Plus: the encode knob is an ENUM (no
    dose can silently zero it) and switching it CHANGES the actor gradient, so the dual
    index is actually driving the extraction rather than sitting inert beside it."""
    _run("""
    obs_e = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    g_e = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))

    cfg = dict(get_config())
    assert cfg['measure_density_z_encode'] == 'sphere', cfg['measure_density_z_encode']
    assert isinstance(cfg['measure_density_z_encode'], str), "must be an enum, not a dose"

    for label, c, keys in (
        ('R-A/latent', dual_cfg(),
         ('measure/density_ce', 'measure/z_cos_spread', 'measure/density_w_ztv',
          'measure/density_zsigma_eff_rank', 'measure/density_zsigma_cond',
          'actor/actor_loss', 'actor/q_mean')),
        ('R-A/kernel', dual_cfg(measure_density_z_actor_reward='kernel'),
         ('measure/density_ce', 'measure/z_cos_spread',
          'measure/density_zsigma_eff_rank', 'measure/density_zsigma_cond',
          'actor/actor_loss', 'actor/q_mask_mean', 'actor/q_mask_frac')),
    ):
        a = agent_for(c)
        p0 = a.network.params
        for step in range(30):
            a, info = a.update(jb(make_batch(seed=step)))
            for k in keys:
                assert np.isfinite(float(info[k])), (label, step, k, float(info[k]))
        assert all(np.isfinite(np.asarray(l)).all()
                   for l in jax.tree_util.tree_leaves(a.network.params)), \\
            label + ": NaN after 30 updates"
        assert not np.allclose(sigma_of(a), np.eye(D)), label + ": Sigma_hat never moved"
        for mod in ('density_f', 'density_b', 'actor'):
            assert any(not np.allclose(np.asarray(x), np.asarray(y), atol=0)
                       for x, y in zip(
                           jax.tree_util.tree_leaves(p0['modules_' + mod]),
                           jax.tree_util.tree_leaves(a.network.params['modules_' + mod]))), \\
                label + ": " + mod + " did not move after 30 updates"
        ac = np.asarray(a.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        assert ac.shape == (4, ACT_D) and np.isfinite(ac).all() \\
            and (np.abs(ac) <= 1.0 + 1e-6).all(), (label, ac.shape)

    # the knob moves the actor gradient (not inert)
    bj = jb(make_batch(B=16, seed=21))
    rng = jax.random.PRNGKey(9)

    def actor_grad(cfg_d):
        a = agent_for(cfg_d)
        g = jax.grad(lambda p: a.actor_loss(bj, p, rng)[0])(a.network.params)
        return np.concatenate([np.asarray(x).ravel()
                               for x in jax.tree_util.tree_leaves(g['modules_actor'])])

    g_sphere = actor_grad(z_cfg(measure_density_z_enc_tau=0.0005))
    g_dual = actor_grad(dual_cfg())
    assert np.abs(g_dual).max() > 0, "dual actor gradient is identically zero"
    assert not np.allclose(g_sphere, g_dual, atol=1e-7), \\
        "the dual encode did not change the actor gradient -- the index swap is inert"

    # ... and no gradient flows INTO Sigma_hat (it is a statistic, not a weight)
    a = agent_for(dual_cfg())
    gr = jax.grad(lambda p: a.total_loss(bj, p, rng)[0])(a.network.params)
    assert np.abs(np.asarray(gr['modules_density_zsigma']['sigma'])).max() == 0.0, \\
        "Sigma_hat received a gradient -- it must be written by its EMA alone"
    print("OK")
    """)
