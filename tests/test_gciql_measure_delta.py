"""`measure_delta` -- spec 17a temporally-centered (delta / net-flux) region successor
measure TD on gciql_flow.py's existing first-passage hit path.

Every failed occupancy cell regresses the raw region measure M(s,a,g), whose targets
are dominated by the action-independent dwell mass (PHASE2-LEDGER 9.16 / 9.15c). The
delta measure replaces the target reward hit(x') with the SIGNED EVENT indicator
hit(x')-hit(x) (outside->outside 0, enter +1, dwell 0, exit -1), non-absorbing:

    stock (ball-visitation):  diff = M - hit_n           - gamma*tM
    delta:                    diff = M - (hit_n - hit_c) - gamma*tM

Fixed point identity (spec 17a section 2): J = (1-gamma)*M_visitation - h(x0), bounded
in [-1, 1], same action argmax as the visitation measure at fixed (s, columns).

Knob (default-off, bit-identical): measure_delta.

Discipline (mirrors tests/test_gciql_measure_nstep.py, the spec-16 sibling): defaults
byte-identical in the param tree AND bit-identical in measure_loss; the counting
convention (hit_c from `observations`, hit_n from `next_observations`) is pinned by a
regression test against the off-by-one; the exact fixed-point identity is pinned by a
pure-numpy tabular Bellman check.

Runs in the jax venv subprocess (test_identity_tabular is pure numpy and needs neither).
"""
import textwrap

import numpy as np

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

OBS_D = 40
COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config

    OBS_D = {OBS_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)
    BLOCKS = ((19, 22, 0.4),)

    def base_cfg(**kw):
        # the phase-1 winner's cell: measure-only, bilinear, first-passage, no ortho.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["measure_legs"] = "critic"
        cfg["measure_ortho_coef"] = 0.0; cfg["measure_scale_norm"] = True
        cfg["measure_first_passage"] = True
        cfg.update(kw)
        return cfg

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs[:, 19:] *= 0.25   # so the 0.4-ball fires sometimes
        return dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def measure_loss_of(agent, batch, seed=7):
        loss, info = agent.measure_loss(
            {{k: jnp.asarray(v) for k, v in batch.items()}},
            agent.network.params, jax.random.PRNGKey(seed))
        return float(loss), info

    def flat_sd(agent):
        sd = flax.serialization.to_state_dict(agent.network.params)
        leaves = jax.tree_util.tree_flatten_with_path(sd)[0]
        return [(jax.tree_util.keystr(p), np.asarray(l)) for p, l in leaves]

    def four_transition_batch(seed=1):
        '''Plants all four hit(x')-hit(x) transition types (against goal column j=0,
        col_states=nobs since tpd defaults to 'batch') in rows 1-4: outside->outside,
        outside->inside (enter), inside->inside (dwell), inside->outside (exit).'''
        rng = np.random.default_rng(seed)
        B = 16
        obs = (rng.normal(size=(B, OBS_D)) * 0.1).astype(np.float32)
        nobs = (rng.normal(size=(B, OBS_D)) * 0.1).astype(np.float32)
        obs[:, 19:22] *= 20.0
        nobs[:, 19:22] *= 20.0
        C = np.array([3.0, -2.0, 1.0], np.float32)     # goal-column center = nobs[0]
        FAR = C + np.array([50.0, 0.0, 0.0], np.float32)
        nobs[0, 19:22] = C; obs[0, 19:22] = C           # column j=0's own test point
        obs[1, 19:22] = FAR; nobs[1, 19:22] = FAR       # outside -> outside
        obs[2, 19:22] = FAR; nobs[2, 19:22] = C         # outside -> inside (enter)
        obs[3, 19:22] = C;   nobs[3, 19:22] = C          # inside -> inside (dwell)
        obs[4, 19:22] = C;   nobs[4, 19:22] = FAR        # inside -> outside (exit)
        return dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=(rng.normal(size=(B, OBS_D)) * 0.1).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )

    def capture_terms(agent, batch, seed=7):
        '''Monkeypatch _measure_terms to capture the (M, tM, hit, absorbing, scale_norm)
        actually fed to it by measure_loss. M/tM/gamma/scale_norm are UNCHANGED by
        measure_delta (it only edits the hit argument after they are built), so this is
        a faithful capture of the TD inputs without hand-replicating the flow actor's
        bootstrap sampling.'''
        cap = {{}}
        orig = GCIQLFlowAgent._measure_terms
        def spy(M, tM, gamma, scale_norm, eps=1e-6, hit=None, absorbing=True):
            cap['M'] = np.asarray(M); cap['tM'] = np.asarray(tM)
            cap['hit'] = None if hit is None else np.asarray(hit)
            cap['gamma'] = gamma; cap['scale_norm'] = scale_norm; cap['absorbing'] = absorbing
            return orig(M, tM, gamma, scale_norm, eps=eps, hit=hit, absorbing=absorbing)
        GCIQLFlowAgent._measure_terms = staticmethod(spy)
        try:
            loss, info = agent.measure_loss(
                {{k: jnp.asarray(v) for k, v in batch.items()}},
                agent.network.params, jax.random.PRNGKey(seed))
        finally:
            GCIQLFlowAgent._measure_terms = staticmethod(orig)
        return float(loss), info, cap

    def numpy_loss_ref(M, tM, hit_ref, gamma, eps=1e-6):
        '''Reference for the non-absorbing (cont=1), scale_norm=True, first-passage
        residual: diff = M - hit_ref - gamma*tM, squared diag, offdiag, /msq, mean heads.'''
        B = M.shape[-1]
        off = 1.0 - np.eye(B)
        diff = M - hit_ref[None] - gamma * tM
        diagt = np.mean(np.diagonal(diff, axis1=-2, axis2=-1) ** 2, axis=-1)
        offdiag = 0.5 * np.sum((diff * off[None]) ** 2, axis=(-2, -1)) / np.sum(off)
        msq = np.mean(M ** 2)
        return float(np.mean(offdiag / (msq + eps) + diagt / (msq + eps)))
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, cwd=str(OGBENCH_IMPLS), **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_default_off_bit_identical():
    """measure_delta=False (and a config dict that PREDATES the knob) => identical
    param tree AND bit-identical measure_loss, for tpd batch AND hindsight."""
    _run("""
    for tpd in ("batch", "hindsight"):
        cfg_old = base_cfg(measure_fp_absorbing=False, measure_testpoint_dist=tpd)
        cfg_old.pop("measure_delta", None)          # a config dict that PREDATES the knob
        cfg_new = base_cfg(measure_fp_absorbing=False, measure_testpoint_dist=tpd,
                            measure_delta=False)
        assert cfg_new["measure_delta"] is False
        a_old, a_new = agent_for(cfg_old), agent_for(cfg_new)
        f_old, f_new = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in f_old] == [k for k, _ in f_new], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(f_old, f_new):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs ({tpd})"
        b = make_batch()
        l_old, _ = measure_loss_of(a_old, b)
        l_new, _ = measure_loss_of(a_new, b)
        assert l_old == l_new, (tpd, l_old, l_new)
    print("OK")
    """)


def test_delta_target_matches_numpy_reference():
    """Constructed batch realizing all four transition types; the code's captured hit
    matrix equals hit_n-hit_c (regression on the whole B x B matrix, not just the
    planted rows) and the returned loss matches a from-scratch numpy reference, for tpd
    batch AND hindsight. Also re-asserts the actor's Q-term never backprops into the
    critic (crl-actor-loss-critic-gradient lesson, spec 17a section 4) on a delta config."""
    _run("""
    for tpd in ("batch", "hindsight"):
        cfg = base_cfg(measure_fp_absorbing=False, measure_delta=True,
                        measure_testpoint_dist=tpd)
        agent = agent_for(cfg)
        b = four_transition_batch()
        loss, info, cap = capture_terms(agent, b)
        assert cap['hit'] is not None and cap['absorbing'] is False

        col_states = b['next_observations'] if tpd == 'batch' else b['value_goals']
        hit_n = np.asarray(GCIQLFlowAgent._hit_matrix(
            jnp.asarray(b['next_observations']), jnp.asarray(col_states), BLOCKS, jnp.float32))
        hit_c = np.asarray(GCIQLFlowAgent._hit_matrix(
            jnp.asarray(b['observations']), jnp.asarray(col_states), BLOCKS, jnp.float32))
        hit_ref = hit_n - hit_c
        assert np.array_equal(cap['hit'], hit_ref), (tpd, np.abs(cap['hit'] - hit_ref).max())

        if tpd == 'batch':
            # column j=0 IS nobs[0] here -- spot-check the transition table.
            assert hit_ref[1, 0] == 0.0, "outside->outside must be 0"
            assert hit_ref[2, 0] == 1.0, "outside->inside (enter) must be +1"
            assert hit_ref[3, 0] == 0.0, "inside->inside (dwell) must be 0"
            assert hit_ref[4, 0] == -1.0, "inside->outside (exit) must be -1"

        loss_ref = numpy_loss_ref(cap['M'], cap['tM'], hit_ref, cfg['discount'])
        assert abs(loss_ref - loss) < 1e-5, (tpd, loss_ref, loss)

    # grad-hygiene (spec 17a section 4): the flow actor's q-term reads the critic WITHOUT
    # params=grad_params, so d(actor_loss)/d(critic params) must be exactly zero even
    # though delta changed what the critic's fixed point IS.
    cfg = base_cfg(measure_fp_absorbing=False, measure_delta=True)
    agent = agent_for(cfg)
    b = four_transition_batch()
    g = jax.grad(lambda p: agent.actor_loss(b, p, rng=jax.random.PRNGKey(0))[0])(
        agent.network.params)
    crit_g = jax.tree_util.tree_leaves(g["modules_critic"])
    assert all(bool(jnp.all(x == 0)) for x in crit_g), \
        "delta actor loss must not backprop into the critic"
    print("OK")
    """)


def test_hit_c_reads_observations():
    """Regression against the off-by-one: hit_c MUST come from `observations`, hit_n
    from `next_observations`. A pure-exit row (obs inside the column ball, nobs outside)
    must show up as -1, and measure_delta_event_rate must be > 0 there; a batch whose
    obs/nobs are both far from every column ball (and dwell on the diagonal) must show
    measure_delta_event_rate == 0."""
    _run("""
    b = four_transition_batch()
    cfg = base_cfg(measure_fp_absorbing=False, measure_delta=True)
    agent = agent_for(cfg)
    loss, info, cap = capture_terms(agent, b)
    hit_n = np.asarray(GCIQLFlowAgent._hit_matrix(
        jnp.asarray(b['next_observations']), jnp.asarray(b['next_observations']),
        BLOCKS, jnp.float32))
    hit_c = np.asarray(GCIQLFlowAgent._hit_matrix(
        jnp.asarray(b['observations']), jnp.asarray(b['next_observations']),
        BLOCKS, jnp.float32))
    hit_ref = hit_n - hit_c
    assert hit_ref[4, 0] == -1.0, "row4 (obs inside, nobs outside) must be a pure exit"
    assert np.array_equal(cap['hit'], hit_ref)
    assert float(info['measure_delta_event_rate']) > 0.0

    # a batch dwelling far from every column ball -- zero net-flux everywhere (diagonal
    # dwells because obs==nobs; off-diagonal never enters because the object coords are
    # separated by 1000 obs units per row, far outside the 0.4 radius).
    B = 12
    rng = np.random.default_rng(9)
    far = rng.normal(size=(B, OBS_D)).astype(np.float32)
    far[:, 19:22] = (np.arange(B)[:, None] * 1000.0).astype(np.float32)
    b_far = dict(
        observations=far.copy(), next_observations=far.copy(),
        actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
        value_goals=far.copy(), actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
        rewards=np.zeros(B, np.float32), masks=np.ones(B, np.float32))
    l2, info2 = measure_loss_of(agent, b_far)
    assert float(info2['measure_delta_event_rate']) == 0.0, info2['measure_delta_event_rate']
    print("OK")
    """)


def test_identity_tabular():
    """Pure numpy, no agent: a 3-state deterministic chain (0->1->2->0) with a 1-state
    region (state 1), under the code's Bellman convention (reward at the NEXT state):
    J(s) = (h(T(s))-h(s)) + gamma*J(T(s)), M(s) = h(T(s)) + gamma*M(T(s)). Assert
    J = (1-gamma)*M - h(x0) to 1e-6, for two discounts. Pins the eval-side conversion
    Mhat = (h + Jhat)/(1-gamma) used later by the L1fix/L2 harnesses."""
    T = np.array([1, 2, 0])          # deterministic 3-cycle: 0->1->2->0
    h = np.array([0.0, 1.0, 0.0])    # region = state 1
    for gamma in (0.9, 0.99):
        M = np.zeros(3)
        for _ in range(20000):
            M = h[T] + gamma * M[T]
        J = np.zeros(3)
        for _ in range(20000):
            J = (h[T] - h) + gamma * J[T]
        for s in range(3):
            lhs = J[s]
            rhs = (1 - gamma) * M[s] - h[s]
            assert abs(lhs - rhs) < 1e-6, (gamma, s, lhs, rhs)


def test_bounded_target():
    """With measure_delta, the immediate target entries lie in {-1, 0, +1} (no jnp.clip
    anywhere in the path -- the boundedness is structural, from hit_n, hit_c in {0,1});
    the loss re-derived from the captured bounded target matches the returned loss."""
    _run("""
    b = four_transition_batch()
    cfg = base_cfg(measure_fp_absorbing=False, measure_delta=True)
    agent = agent_for(cfg)
    loss, info, cap = capture_terms(agent, b)
    vals = set(np.round(np.unique(cap['hit']), 6).tolist())
    assert vals <= {-1.0, 0.0, 1.0}, vals
    loss_ref = numpy_loss_ref(cap['M'], cap['tM'], cap['hit'], cfg['discount'])
    assert abs(loss_ref - loss) < 1e-5, (loss_ref, loss)
    print("OK")
    """)


def test_bounded_target_no_clip_in_source():
    """Static check: the measure_delta insertion in gciql_flow.py contains no jnp.clip
    (the boundedness must be structural -- hit_n, hit_c in {0,1} -- not a bolt-on clamp)."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "tools/wandb_mode_shim/gciql_flow.py"
    text = src.read_text()
    start = text.index("if self.config.get('measure_delta', False):")
    end = text.index("if self.config.get('measure_nhat_head', False):", start)
    block = text[start:end]
    assert "clip" not in block, "measure_delta target must be bounded structurally, not clamped"


def test_info_keys():
    """measure_delta_event_rate/measure_delta_hit_level/measure_delta_msq present iff
    the knob is on; absent on both the delta-off ball-visitation path and the plain
    (absorbing) first-passage path -- regression against info-key leakage.
    measure_delta_msq (round-1 review MEDIUM finding) is spec 17 §6 gate G2's
    instrument ("no zero-flux collapse (msq bounded away from 0)"): finite and
    non-negative (a mean-square)."""
    _run("""
    b = make_batch()
    a_on = agent_for(base_cfg(measure_fp_absorbing=False, measure_delta=True))
    _, info_on = measure_loss_of(a_on, b)
    assert "measure_delta_event_rate" in info_on and "measure_delta_hit_level" in info_on
    assert "measure_delta_msq" in info_on
    assert np.isfinite(float(info_on["measure_delta_msq"]))
    assert float(info_on["measure_delta_msq"]) >= 0.0

    a_off = agent_for(base_cfg(measure_fp_absorbing=False, measure_delta=False))
    _, info_off = measure_loss_of(a_off, b)
    assert "measure_delta_event_rate" not in info_off
    assert "measure_delta_hit_level" not in info_off
    assert "measure_delta_msq" not in info_off

    a_stock = agent_for(base_cfg())   # plain (absorbing) first-passage, no delta
    _, info_stock = measure_loss_of(a_stock, b)
    assert "measure_delta_event_rate" not in info_stock
    assert "measure_delta_hit_level" not in info_stock
    assert "measure_delta_msq" not in info_stock
    print("OK")
    """)


def test_assert_lattice():
    """delta requires the hit-region + non-absorbing path; refuses the exotic axes and
    the 17b knob; accepts hindsight, stopgrad_psi, and a warm-start-shaped config."""
    _run("""
    def refuses(**kw):
        cfg_kw = dict(measure_fp_absorbing=False, measure_delta=True)
        cfg_kw.update(kw)
        try:
            agent_for(base_cfg(**cfg_kw))
        except AssertionError:
            return True
        return False
    assert refuses(measure_first_passage=False)
    assert refuses(measure_fp_absorbing=True)
    assert refuses(measure_nstep=4)
    assert refuses(measure_expectile=True)
    assert refuses(measure_nhat_head=True)
    assert refuses(measure_eps_multiscale='0.2,0.4')
    assert refuses(fb_forward_z=True)
    assert refuses(measure_delta_partition=True)
    assert not refuses()
    assert not refuses(measure_testpoint_dist='hindsight')
    assert not refuses(measure_stopgrad_psi=True)
    assert not refuses(warm_start_ckpt='')   # warm-start-shaped config, no ckpt loaded
    print("OK")
    """)
