"""`measure_density_z` -- spec 24: z-indexed Density-FB. Contract:
`PAPER/Multistep-FB/specs/24-z-indexed-density.md` sections 1/2/5.

Swaps `measure_density`'s (spec 18) policy-family index from goals to an FB-style
latent z, and nothing else:

    z(x)      = sqrt(d) * B̄(x̃) / ||B̄(x̃)||     (x̃ = eps-scaled block coords; B̄ =
                                                 target_density_b, stop-gradded)
    z_i       = z(x_{g,i})  w.p. 1 - z_rand_frac  (x_{g,i} = value_goals row i)
                ~ sqrt(d) * Uniform(S^{d-1})  otherwise (fresh per row per step)

    f_ij   = F(s_i, a_i, z_i)^T B(u_j)/tau        (z replaces g in F's input)
    a+_i   ~ pi(.|s'_i, z_i)                       (same z, one-shot flow actor)
    t_ij   = sg[(1-gamma)*k_ij + gamma*wbar_ij]    (k UNCHANGED -- never read g/z)
    actor q_i = sum_j w_j(s_i, pi(s_i,z_i), z_i) * (B̃(u_j)^T z_i / sqrt(d))
                (B̃ = target_density_b, stop-gradded; the z-analog of the g-indexed
                 kernel term kg -- FB's own linear reward model r_z(u) = B(u)^T z)

Scoped to actor_kind='flow' (round 1): the architecture change (F's [obs,actions,z]
input width, the actor's z-conditioned goal leg) is only implemented on that path.
Mutually exclusive with measure_density_behavior (which drops the actor's q-term
instead of z-indexing it).

Same subprocess-jax harness as tests/test_gciql_measure_density.py: OBS_D=40,
action dim 5, d=8 (value_latent_dim AND measure_density_dim), B=16, hit_spec
'19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_density_z.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import subprocess
import textwrap

import numpy as np
import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8   # value_latent_dim AND measure_density_dim, matches the spec-18 harness
ACT_D = 5

# Pinned pre-measure_density_z baseline commit for test_default_off_bit_identical --
# NOT "HEAD" (see tests/test_gciql_measure_density.py's DENSITY_PRE_SPEC18_SHA for the
# full rationale: HEAD moves forward on this shared branch as other sessions land
# commits). This is the spec-24 DOCS commit -- it has measure_density/
# measure_density_behavior but not measure_density_z, so it permanently predates this
# feature regardless of where HEAD drifts later.
DENSITY_Z_PRE_SPEC24_SHA = "602008f1cdaa28cc0b4150e8812ec90e3f6be482"


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                        cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0


COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec, _sphere

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def base_cfg(**kw):
        # spec 24's base cell: measure-only, bilinear, density, Z ON.
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

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        batch = dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, ACT_D)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )
        return batch

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def total_loss_of(agent, batch, seed=7):
        loss, info = agent.total_loss(
            {{k: jnp.asarray(v) for k, v in batch.items()}},
            agent.network.params, jax.random.PRNGKey(seed))
        return float(loss), info

    def z_encode_ref(agent, goal_obs, blocks, d):
        # numpy/direct-module reference for _density_z_encode: deterministic
        # sqrt(d)-normalized target_density_b(x_tilde), no rng.
        x_sc = GCIQLFlowAgent._density_queries(goal_obs, blocks)
        b_emb = agent.network.select('target_density_b')(x_sc)
        return _sphere(b_emb, float(d))

    def z_ref(agent, goal_obs, blocks, d, rand_frac, rng):
        # numpy/direct-module reference for _density_z (the full mix), SAME rng-split
        # order as the agent's own method: rng_sphere, rng_mix = split(rng).
        z_g = z_encode_ref(agent, goal_obs, blocks, d)
        rng_sphere, rng_mix = jax.random.split(rng)
        B = goal_obs.shape[0]
        z_rand = _sphere(jax.random.normal(rng_sphere, (B, d)), float(d))
        use_rand = jax.random.bernoulli(rng_mix, rand_frac, (B, 1))
        return jnp.where(use_rand, z_rand, z_g)

    def density_z_ref(agent, cfg, bj, rng, hard):
        # numpy/direct-module reference for density_loss's z-indexed computation,
        # SAME rng-split order as density_loss: rng, z_rng = split(incoming rng).
        obs, nobs = bj['observations'], bj['next_observations']
        acts, g = bj['actions'], bj['value_goals']
        gamma, tau = cfg['discount'], cfg['measure_density_temp']
        d = cfg['measure_density_dim']
        blocks = _parse_hit_spec(cfg['measure_hit_spec'])
        rand_frac = cfg.get('measure_density_z_rand_frac', 0.5)

        rng1, z_rng = jax.random.split(rng)
        z = z_ref(agent, g, blocks, d, rand_frac, z_rng)

        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)
        b_emb = agent.network.select('density_b')(u_sc, params=agent.network.params)
        f_emb = agent.network.select('density_f')(
            jnp.concatenate([obs, acts, z], -1), params=agent.network.params)
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau
        logp = np.asarray(jax.nn.log_softmax(logits, -1))

        noise = jax.random.normal(rng1, acts.shape)
        a_plus = jax.lax.stop_gradient(agent.network.select('actor')(nobs, z, noise))
        tb = agent.network.select('target_density_b')(u_sc)
        tf = agent.network.select('target_density_f')(jnp.concatenate([nobs, a_plus, z], -1))
        w_bar = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1))
        if hard:
            k = np.eye(nobs.shape[0])
        else:
            k = np.asarray(GCIQLFlowAgent._delta_phi(nobs, u_raw, blocks))
        t = (1.0 - gamma) * k[None] + gamma * w_bar
        ce_ref = -np.mean(np.sum(t * logp, axis=-1))
        return ce_ref, t, np.asarray(z)

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
# 1. default-off bit-identical
# --------------------------------------------------------------------------
def test_default_off_bit_identical():
    """measure_density_z=False (the default, whether unset or explicit) on TOP of
    measure_density=True: param tree AND total_loss bit-identical to the shipped
    spec-18/19a arm loaded from the PINNED commit DENSITY_Z_PRE_SPEC24_SHA (a genuine
    off-path regression -- reordered rng splits, an accidentally-live branch, a changed
    default -- would show up here because the two sides run different code). No
    density-z-specific leaves (there are none registered: measure_density_z changes
    example-input WIDTHS of existing modules, not new modules) and the param COUNT
    per module matches exactly.

    See tests/test_gciql_measure_density.py::test_default_off_bit_identical's
    docstring for why this pins a SHA and not HEAD -- same reasoning, one feature
    boundary later."""
    if not _baseline_reachable(DENSITY_Z_PRE_SPEC24_SHA):
        pytest.skip(f"pinned baseline commit {DENSITY_Z_PRE_SPEC24_SHA} is not reachable "
                    "in this checkout (e.g. a shallow clone) -- cannot run the off-path "
                    "regression diff against it")
    _run("""
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", "%s:tools/wandb_mode_shim/gciql_flow.py"],
        cwd=%r, capture_output=True, text=True, check=True).stdout
    assert "measure_density_z" not in baseline_src, (
        "pinned baseline %s no longer predates measure_density_z -- the pin has gone "
        "stale (e.g. history was rewritten); re-pin DENSITY_Z_PRE_SPEC24_SHA to a commit "
        "that genuinely predates spec 24")
    assert "measure_density_behavior" in baseline_src, \\
        "pinned baseline is missing measure_density_behavior entirely -- spec 19a must " \\
        "already be shipped there"
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_baseline.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def cfg18(get_cfg_fn, **kw):
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

    b = make_batch()
    bj = {kk: jnp.asarray(vv) for kk, vv in b.items()}

    # (a) z knob OFF, measure_density ON: bit-identical to the shipped spec-18/19a arm.
    a_base = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg18(baseline.get_config))
    a_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                   cfg18(get_config, measure_density_z=False))
    t_base = jax.tree_util.tree_leaves(a_base.network.params)
    t_off = jax.tree_util.tree_leaves(a_off.network.params)
    assert len(t_base) == len(t_off), (len(t_base), len(t_off))
    assert all((x == y).all() for x, y in zip(t_base, t_off)), \\
        "param tree differs from the shipped spec-18/19a HEAD baseline with the z knob off"
    l_base, _ = a_base.total_loss(bj, a_base.network.params, jax.random.PRNGKey(7))
    l_off, _ = a_off.total_loss(bj, a_off.network.params, jax.random.PRNGKey(7))
    assert np.allclose(float(l_base), float(l_off), atol=1e-6), (float(l_base), float(l_off))

    # (b) BOTH off (no measure_density knob at all): bit-identical to a no-knob config.
    cfg_nk_base = dict(baseline.get_config())
    cfg_nk_base["encoder"] = None; cfg_nk_base["frame_stack"] = None
    a_nk_base = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_nk_base)
    cfg_nk = dict(get_config())
    cfg_nk["encoder"] = None; cfg_nk["frame_stack"] = None
    a_nk = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_nk)
    tn_base = jax.tree_util.tree_leaves(a_nk_base.network.params)
    tn = jax.tree_util.tree_leaves(a_nk.network.params)
    assert len(tn_base) == len(tn)
    assert all((x == y).all() for x, y in zip(tn_base, tn))
    ln_base, _ = a_nk_base.total_loss(bj, a_nk_base.network.params, jax.random.PRNGKey(7))
    ln, _ = a_nk.total_loss(bj, a_nk.network.params, jax.random.PRNGKey(7))
    assert np.allclose(float(ln_base), float(ln), atol=1e-6)
    print("OK")
    """ % (DENSITY_Z_PRE_SPEC24_SHA, str(REPO), DENSITY_Z_PRE_SPEC24_SHA))


# --------------------------------------------------------------------------
# 2. z construction
# --------------------------------------------------------------------------
def test_z_construction():
    """||z_i|| = sqrt(d) in BOTH mix branches; rand_frac=0 -> every row is the
    relabeled-goal embedding (z == _density_z_encode(g) exactly); rand_frac=1 ->
    every row is a fresh uniform-sphere draw, DIFFERENT across two rng keys (proving
    it is actually random, not a relabeled constant); z is built from the TARGET
    tower -- perturbing the ONLINE density_b tower's params leaves z byte-identical."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    b = make_batch(B=10, seed=2)
    g = jnp.asarray(b['value_goals'])

    # rand_frac=0: z == the deterministic goal embedding exactly.
    a0 = agent_for(base_cfg(measure_density_z_rand_frac=0.0))
    z_g = a0._density_z_encode(g)
    z0 = a0._density_z(g, jax.random.PRNGKey(1))
    assert np.allclose(np.asarray(z0), np.asarray(z_g), atol=1e-6)
    norms0 = np.linalg.norm(np.asarray(z0), axis=-1)
    assert np.allclose(norms0, np.sqrt(D), atol=1e-4), norms0

    # rand_frac=1: every row is a FRESH uniform-sphere draw -- norm sqrt(d), and two
    # different rng keys give DIFFERENT draws (not a relabeled constant).
    a1 = agent_for(base_cfg(measure_density_z_rand_frac=1.0))
    z1a = a1._density_z(g, jax.random.PRNGKey(2))
    z1b = a1._density_z(g, jax.random.PRNGKey(3))
    norms1a = np.linalg.norm(np.asarray(z1a), axis=-1)
    assert np.allclose(norms1a, np.sqrt(D), atol=1e-4), norms1a
    assert not np.allclose(np.asarray(z1a), np.asarray(z1b), atol=1e-3), \\
        "rand_frac=1 z did not vary across rng keys -- not actually random"
    assert not np.allclose(np.asarray(z1a), np.asarray(z_g), atol=1e-2), \\
        "rand_frac=1 z coincides with the goal embedding -- mixing had no effect"

    # default mix (rand_frac=0.5): also norm sqrt(d).
    zmix = a._density_z(g, jax.random.PRNGKey(4))
    normsmix = np.linalg.norm(np.asarray(zmix), axis=-1)
    assert np.allclose(normsmix, np.sqrt(D), atol=1e-4), normsmix

    # TARGET-tower invariance: perturb the ONLINE density_b params; z (built from
    # target_density_b only) must be byte-identical.
    z_before = a._density_z_encode(g)
    p2 = dict(a.network.params)
    p2['modules_density_b'] = jax.tree_util.tree_map(lambda x: x + 7.0, p2['modules_density_b'])
    a2 = a.replace(network=a.network.replace(params=p2))
    z_after = a2._density_z_encode(g)
    assert np.array_equal(np.asarray(z_before), np.asarray(z_after)), \\
        "z changed after perturbing the ONLINE density_b tower -- it must be built from TARGET B"
    # sanity: perturbing the TARGET tower DOES move z (the test can distinguish the two).
    p3 = dict(a.network.params)
    p3['modules_target_density_b'] = jax.tree_util.tree_map(
        lambda x: x + 7.0, p3['modules_target_density_b'])
    a3 = a.replace(network=a.network.replace(params=p3))
    z_target_moved = a3._density_z_encode(g)
    assert not np.allclose(np.asarray(z_before), np.asarray(z_target_moved), atol=1e-4), \\
        "test setup bug: perturbing target_density_b should move z"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. CE loss matches a numpy/direct-module reference (soft AND hard)
# --------------------------------------------------------------------------
def test_ce_matches_numpy_reference():
    """density_loss's CE (z replacing g throughout: F's input, the bootstrap actor's
    conditioning, target_density_f's input) matches density_z_ref atol 1e-5, for both
    the soft (eps-kernel) and hard (one-hot) immediate term k -- which is UNCHANGED by
    the z knob (it never read g/z, only nobs vs the resampled query support). Rows of
    t sum to 1 in both cases."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch(B=6, seed=11)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(7)

    loss, info = a.density_loss(bj, a.network.params, rng)
    ce_ref, t, z = density_z_ref(a, cfg, bj, rng, hard=False)
    assert np.allclose(ce_ref, float(loss), atol=1e-5), (ce_ref, float(loss))
    assert np.allclose(t.sum(-1), 1.0, atol=1e-5)
    assert 'z_cos_spread' in info

    cfg_hard = base_cfg(measure_density_hard=True)
    a_hard = agent_for(cfg_hard)
    loss_h, info_h = a_hard.density_loss(bj, a_hard.network.params, rng)
    ce_ref_h, t_h, _ = density_z_ref(a_hard, cfg_hard, bj, rng, hard=True)
    assert np.allclose(ce_ref_h, float(loss_h), atol=1e-5), (ce_ref_h, float(loss_h))
    assert np.allclose(t_h.sum(-1), 1.0, atol=1e-5)

    # the immediate term k itself must be UNCHANGED by the z knob -- cross-check
    # against a g-indexed (z off) sibling agent's own k on the SAME batch.
    a_g = agent_for(base_cfg(measure_density_z=False))
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    u_raw = GCIQLFlowAgent._raw_block_cols(bj['next_observations'], blocks)
    k_shared = np.asarray(GCIQLFlowAgent._delta_phi(bj['next_observations'], u_raw, blocks))
    # k_shared is architecture/param-free (pure function of nobs + queries) -- both
    # the z-on and z-off agents must read the exact same immediate term.
    assert np.allclose(k_shared.sum(-1), 1.0, atol=1e-5)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. actor q-term matches a numpy/direct-module reference
# --------------------------------------------------------------------------
def test_actor_q_matches_numpy():
    """actor_loss's q_mean under measure_density_z equals a hand-reconstructed
    q = sum_j w_j(s, pi(s,z), z) * (B̃(u_j)^T z)/sqrt(d), B̃=target_density_b, built
    from DIRECT module calls at the agent's own params with the SAME rng-split order
    actor_loss itself uses (rng, x0_rng, t_rng, n_rng, z_rng = split(rng, 5); z =
    _density_z(actor_goals, z_rng); noise off n_rng)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    b = make_batch(B=5, seed=6)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(4)

    _, info = a.actor_loss(bj, a.network.params, rng)

    obs, goals, actions = bj['observations'], bj['actor_goals'], bj['actions']
    _, x0_rng, t_rng, n_rng, z_rng = jax.random.split(rng, 5)
    z = a._density_z(goals, z_rng)
    noise = jax.random.normal(n_rng, actions.shape)
    actor_actions = a.network.select('actor')(obs, z, noise, params=a.network.params)

    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    b_emb = a.network.select('density_b')(u_sc)
    f_emb = a.network.select('density_f')(jnp.concatenate([obs, actor_actions, z], -1))
    w_hand = jax.nn.softmax(
        jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp'], -1)
    tb = a.network.select('target_density_b')(u_sc)
    rz = jnp.einsum('md,bd->bm', tb, z) / jnp.sqrt(cfg['measure_density_dim'])
    q1_hand, q2_hand = (w_hand * rz[None]).sum(-1)
    q_hand = jnp.minimum(q1_hand, q2_hand) if cfg['double_q'] else q1_hand
    assert np.isfinite(np.asarray(q_hand)).all()
    assert np.allclose(float(jnp.mean(q_hand)), float(info['q_mean']), atol=1e-4), \\
        (float(jnp.mean(q_hand)), float(info['q_mean']))
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4b. grad hygiene, both directions, incl. z's stop-grad being load-bearing
# --------------------------------------------------------------------------
def test_grad_hygiene():
    """jax.grad(actor_loss): ~0 over density_f/density_b/target_density_f/
    target_density_b/critic (the q-term reads frozen towers); >0 over modules_actor.
    jax.grad(density_loss): ~0 over modules_actor (bootstrap stop-gradded); >0 over
    modules_density_f/density_b (both trained); ~0 over both target towers
    (soft-updated outside the loss, and z's own target_density_b read is never
    passed grad_params either).

    z stop-grad load-bearing (mirrors test_target_stopgrad_is_load_bearing in the
    spec-18 suite): reconstructing density_loss's z-conditioned F/logits computation
    with target_density_b DELIBERATELY called WITH params=grad_params (the live path
    the shipped stop_gradient(z) guards against) shows the grad into
    modules_target_density_b is NONZERO without the wrapper and ZERO with it --
    proving the wrapper is load-bearing on the path it exists to guard, using
    rand_frac=0 so the reconstruction isolates the z_encode path (no bernoulli mixing
    noise in the way)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = {k: jnp.asarray(v) for k, v in make_batch().items()}
    rng = jax.random.PRNGKey(3)

    def actor_loss_fn(params):
        loss, _ = a.actor_loss(b, params, rng)
        return loss
    grads_a = jax.grad(actor_loss_fn)(a.network.params)
    for key in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b", "modules_critic"):
        assert gnorm(grads_a, key) < 1e-9, (key, gnorm(grads_a, key))
    assert gnorm(grads_a, 'modules_actor') > 1e-8, gnorm(grads_a, 'modules_actor')

    def density_loss_fn(params):
        loss, _ = a.density_loss(b, params, rng)
        return loss
    grads_d = jax.grad(density_loss_fn)(a.network.params)
    assert gnorm(grads_d, 'modules_actor') < 1e-9, gnorm(grads_d, 'modules_actor')
    assert gnorm(grads_d, 'modules_density_f') > 1e-8, gnorm(grads_d, 'modules_density_f')
    assert gnorm(grads_d, 'modules_density_b') > 1e-8, gnorm(grads_d, 'modules_density_b')
    assert gnorm(grads_d, 'modules_target_density_f') < 1e-9, gnorm(grads_d, 'modules_target_density_f')
    assert gnorm(grads_d, 'modules_target_density_b') < 1e-9, gnorm(grads_d, 'modules_target_density_b')

    # z stop-grad load-bearing.
    cfg0 = base_cfg(measure_density_z_rand_frac=0.0)
    a0 = agent_for(cfg0)
    obs, nobs = b['observations'], b['next_observations']
    acts, g = b['actions'], b['value_goals']
    tau = cfg0['measure_density_temp']
    d = cfg0['measure_density_dim']
    blocks = _parse_hit_spec(cfg0['measure_hit_spec'])
    x_sc = GCIQLFlowAgent._density_queries(g, blocks)
    u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)

    def loss_with_live_z(params, apply_sg):
        b_emb_z = a0.network.select('target_density_b')(x_sc, params=params)  # THE MISTAKE
        z_raw = _sphere(b_emb_z, float(d))
        z = jax.lax.stop_gradient(z_raw) if apply_sg else z_raw
        b_emb = a0.network.select('density_b')(u_sc, params=params)
        f_emb = a0.network.select('density_f')(
            jnp.concatenate([obs, acts, z], -1), params=params)
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau
        return jnp.mean(logits ** 2)   # any differentiable scalar of logits suffices

    grads_sg = jax.grad(lambda p: loss_with_live_z(p, True))(a0.network.params)
    grads_nosg = jax.grad(lambda p: loss_with_live_z(p, False))(a0.network.params)
    assert gnorm(grads_sg, 'modules_target_density_b') < 1e-9, \\
        (gnorm(grads_sg, 'modules_target_density_b'),
         "stop_gradient(z) failed to zero a LIVE target-tower path")
    assert gnorm(grads_nosg, 'modules_target_density_b') > 1e-8, \\
        "test setup bug: target_density_b grad should be NONZERO without stop_gradient(z)"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. E1 goal-encoding determinism
# --------------------------------------------------------------------------
def test_goal_encoding_determinism():
    """_density_z_encode(g) -- the transform E1's direct goal conditioning
    (z* = z(phi(g_eval))) reads -- takes no rng and is bit-for-bit reproducible: two
    independent calls on the same agent agree exactly, and rebuilding a FRESH agent
    instance from the SAME params tree (simulating a checkpoint restore -- a new
    TrainState wrapping a plain dict copy of the params, not the same live object)
    reproduces it exactly too."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch(B=7, seed=9)
    g = jnp.asarray(b['value_goals'])

    z1 = a._density_z_encode(g)
    z2 = a._density_z_encode(g)
    assert np.array_equal(np.asarray(z1), np.asarray(z2))

    # simulate a checkpoint restore: a fresh agent wrapping a plain-dict COPY of the
    # trained params (not the same python object).
    import copy as _copy
    restored_params = jax.tree_util.tree_map(lambda x: np.array(x).copy(), a.network.params)
    a_restored = a.replace(network=a.network.replace(
        params=jax.tree_util.tree_map(jnp.asarray, restored_params)))
    z_restored = a_restored._density_z_encode(g)
    assert np.array_equal(np.asarray(z1), np.asarray(z_restored)), \\
        "z*(g) not reproducible across a simulated checkpoint restore"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. assert lattice
# --------------------------------------------------------------------------
def test_lattice():
    """Refuses: measure_density_z without measure_density; with
    measure_density_behavior; measure_density_z_rand_frac out of [0,1];
    actor_kind='ddpgbc'/'awr_flow' (spec 24 round 1 is flow-only). Accepts the base
    config (measure_density_z=True on top of the spec-18 base cell)."""
    _run("""
    def refuses(**kw):
        try:
            agent_for(base_cfg(**kw))
        except AssertionError:
            return True
        return False
    assert refuses(measure_density=False)
    assert refuses(measure_density_behavior=True)
    assert refuses(measure_density_z_rand_frac=-0.1)
    assert refuses(measure_density_z_rand_frac=1.1)
    assert refuses(actor_kind='ddpgbc')
    assert refuses(actor_kind='awr_flow')
    assert not refuses()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. real launch crash regression (2026-08-12): sample_actions had no
#    measure_density_z branch -- ScopeParamShapeError under run_gciql.py
# --------------------------------------------------------------------------
def test_sample_actions_matches_registered_z_width():
    """Reproduces the crash spec 24's first real launch hit: `sample_actions` (the
    eval-time entrypoint the real training loop's periodic/startup evaluation calls)
    had no `measure_density_z` branch, so it passed the RAW goal obs straight into an
    actor module registered z-wide -> `ScopeParamShapeError`. The unit suite's own
    harness (OBS_D=40, ACT_D=5, D=8: g-path width 85, z-path width 53) never drove
    `sample_actions` at all (only `actor_loss`/`density_loss` via `update()`), so a
    width slip there could not have shown up regardless of dims.

    This test uses OBS_D2, ACT_D2, DZ2 ALL mutually distinct (7, 3, 11) so a width
    slip cannot pass silently by shape coincidence, and exercises BOTH of the real
    entrypoints: a few real `update()` steps (actor_loss/density_loss's z branches),
    then `sample_actions` on RAW env-shaped goals in BOTH the batched shape (the
    vectorized eval harness) and the UNBATCHED single-observation shape (the
    documented per-episode-step eval call, sample_actions's own docstring) -- the
    second of which caught a SECOND bug in the same fix (`_density_z_encode`'s
    `_density_queries` call slices with an explicit batch axis, unlike `_actor_z`'s
    plain Dense calls which broadcast over any leading shape)."""
    _run("""
    OBS_D2, ACT_D2, DZ2 = 7, 3, 11
    ex_obs2 = jnp.zeros((1, OBS_D2), jnp.float32)
    ex_act2 = jnp.zeros((1, ACT_D2), jnp.float32)
    cfg2 = dict(get_config())
    cfg2["encoder"] = None; cfg2["frame_stack"] = None
    cfg2["value_arch"] = "bilinear"; cfg2["value_latent_dim"] = DZ2
    cfg2["measure_only"] = True
    cfg2["actor_q_center"] = True
    cfg2["measure_density"] = True
    cfg2["measure_density_dim"] = DZ2
    cfg2["measure_hit_spec"] = "0:2:0.4"   # a 2-dim block within the 7-dim obs
    cfg2["measure_density_z"] = True

    a2 = GCIQLFlowAgent.create(0, ex_obs2, ex_act2, cfg2)

    def batch2(seed, B=9):
        r = np.random.default_rng(seed)
        return dict(
            observations=r.normal(size=(B, OBS_D2)).astype(np.float32),
            next_observations=r.normal(size=(B, OBS_D2)).astype(np.float32),
            actions=np.clip(r.normal(size=(B, ACT_D2)), -1, 1).astype(np.float32),
            value_goals=r.normal(size=(B, OBS_D2)).astype(np.float32),
            actor_goals=r.normal(size=(B, OBS_D2)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )
    for step in range(3):
        b = {k: jnp.asarray(v) for k, v in batch2(step).items()}
        a2, info = a2.update(b)
        assert np.isfinite(float(info['measure/density_ce'])), (step, info['measure/density_ce'])

    # THE crashing call site: sample_actions with RAW env-shaped (obs-width, NOT
    # z-width) goals -- exactly what run_gciql.py's real eval loop passes.
    rB = np.random.default_rng(1)
    obs_eval = jnp.asarray(rB.normal(size=(4, OBS_D2)).astype(np.float32))
    goals_eval = jnp.asarray(rB.normal(size=(4, OBS_D2)).astype(np.float32))
    actions = a2.sample_actions(obs_eval, goals_eval, seed=jax.random.PRNGKey(0))
    assert actions.shape == (4, ACT_D2), actions.shape
    assert np.isfinite(np.asarray(actions)).all()

    # the UNBATCHED single-observation shape (per-episode-step eval, documented in
    # sample_actions's own docstring) -- caught a second, distinct shape bug.
    obs1, goal1 = obs_eval[0], goals_eval[0]
    a1 = a2.sample_actions(obs1, goal1, seed=jax.random.PRNGKey(1))
    assert a1.shape == (ACT_D2,), a1.shape
    assert np.isfinite(np.asarray(a1)).all()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. end-to-end updates
# --------------------------------------------------------------------------
def test_end_to_end_updates():
    """30 update() steps on synthetic data: finite density_ce/entropy/z_cos_spread
    info each step, w stays on the simplex, z_cos_spread < 1 (no index collapse) and
    in [0,1], no NaN in params at the end. Load-bearing check: the actor's q_mean
    (unlike measure_density_behavior, z-indexing does NOT drop the q-term) is
    NONZERO and moves the towers -- density_f/density_b params differ from init after
    30 steps."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    p0 = a.network.params
    for step in range(30):
        b = {k: jnp.asarray(v) for k, v in make_batch(seed=step).items()}
        a, info = a.update(b)
        assert np.isfinite(float(info['measure/density_ce'])), (step, info['measure/density_ce'])
        for key in ('measure/density_w_entropy', 'measure/density_tgt_entropy',
                    'measure/density_ess', 'measure/density_ess_median',
                    'measure/density_ess_h0', 'measure/density_ess_h1',
                    'measure/density_logit_max', 'measure/density_k_self',
                    'measure/z_cos_spread', 'actor/actor_loss', 'actor/q_mean'):
            assert np.isfinite(float(info[key])), (step, key, info[key])
        zcs = float(info['measure/z_cos_spread'])
        assert 0.0 <= zcs < 1.0 - 1e-6, (step, zcs)
    leaves = jax.tree_util.tree_leaves(a.network.params)
    assert all(np.isfinite(np.asarray(l)).all() for l in leaves), "NaN in params after 30 updates"

    for tower in ('density_f', 'density_b'):
        onl0 = jax.tree_util.tree_leaves(p0[f'modules_{tower}'])
        onl1 = jax.tree_util.tree_leaves(a.network.params[f'modules_{tower}'])
        assert any(not np.allclose(np.asarray(x), np.asarray(y), atol=0)
                   for x, y in zip(onl0, onl1)), f"{tower} did not move after 30 updates"

    b = {k: jnp.asarray(v) for k, v in make_batch(seed=99).items()}
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    z = a._density_z(b['value_goals'], jax.random.PRNGKey(0))
    u_sc = GCIQLFlowAgent._density_queries(b['next_observations'], blocks)
    b_emb = a.network.select('density_b')(u_sc)
    f_emb = a.network.select('density_f')(jnp.concatenate([b['observations'], b['actions'], z], -1))
    logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp']
    w = np.asarray(jax.nn.softmax(logits, -1))
    assert np.allclose(w.sum(-1), 1.0, atol=1e-4)
    assert (w >= -1e-6).all()
    print("OK")
    """)
