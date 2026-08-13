"""Campaign v2 (2026-07-27): two default-off training knobs on gciql_flow.

  measure_eps_multiscale  per-column hit-ball radius sampled from a level set +
                          a learned per-level psi gain (EpsGainTable, init ones).
  actor_z_greedy_frac     per-sample random sqrt(d)-sphere actor z + FB's native
                          greedy readout q = F(s,a,z).z/sqrt(d) on those samples.

Discipline (mirrors tests/test_gciql_measure_hit_spec.py): both knobs OFF must be
byte-identical in the param tree AND bit-identical in measure_loss/actor_loss, so
every existing checkpoint keeps loading and reproducing. Runs in the jax venv
subprocess.
"""
import textwrap

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

OBS_D = 29
COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import (GCIQLFlowAgent, get_config, _parse_eps_levels,
                            _parse_hit_spec, _sphere)

    OBS_D = {OBS_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def zx_cfg(**kw):
        # the visz/zx cell: measure-only ballvisit ZX (fb_forward_z + z-actor) --
        # the cell BOTH campaign-v2 knobs target.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["measure_legs"] = "critic"
        cfg["measure_coef"] = 1.0   # get_config defaults 0.0, which ZEROES the
        # measure loss inside total_loss even under measure_only (the real yamls
        # set it explicitly) -- without this the update() path trains nothing
        cfg["measure_ortho_coef"] = 0.0; cfg["measure_scale_norm"] = True
        cfg["measure_first_passage"] = True; cfg["measure_fp_absorbing"] = False
        cfg["fb_forward_z"] = True
        cfg["actor_cond"] = "z"; cfg["actor_z_from_critic"] = True
        cfg.update(kw)
        return cfg

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs[:, 19:] *= 0.25          # so the 0.4-ball fires sometimes
        return dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
            masks=rng.integers(0, 2, size=B).astype(np.float32),
        )

    def flat_sd(tree):
        sd = flax.serialization.to_state_dict(tree)
        leaves = jax.tree_util.tree_flatten_with_path(sd)[0]
        return [(jax.tree_util.keystr(p), np.asarray(l)) for p, l in leaves]
"""


def _run(code):
    code = textwrap.dedent(COMMON) + textwrap.dedent(code)
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_parse_eps_levels():
    """Comma-string of positive radii; malformed specs rejected loudly."""
    _run("""
        assert _parse_eps_levels('0.2,0.4,0.8') == (0.2, 0.4, 0.8)
        assert _parse_eps_levels(' 0.4 ') == (0.4,)
        for bad in ('', '0.2,-0.4', '0.0', 'a,b'):
            try:
                _parse_eps_levels(bad)
            except (ValueError, AssertionError):
                pass
            else:
                raise AssertionError(f'accepted malformed levels {bad!r}')
        print("OK")
    """)


def test_both_knobs_off_bit_identical():
    """Default knob values (and a config that PREDATES the knobs) => identical
    param tree and bit-identical measure_loss AND actor_loss."""
    _run("""
        cfg_new = zx_cfg()
        assert cfg_new["measure_eps_multiscale"] == '' and cfg_new["actor_z_greedy_frac"] == 0.0
        cfg_old = zx_cfg()
        cfg_old.pop("measure_eps_multiscale"); cfg_old.pop("actor_z_greedy_frac")
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old.network.params), flat_sd(a_new.network.params)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        b = make_batch(); key = jax.random.PRNGKey(3)
        lo, _ = a_old.measure_loss(b, a_old.network.params, key)
        ln, _ = a_new.measure_loss(b, a_new.network.params, key)
        assert float(lo) == float(ln), (float(lo), float(ln))
        ao, io = a_old.actor_loss(b, a_old.network.params, rng=key)
        an, iN = a_new.actor_loss(b, a_new.network.params, rng=key)
        assert float(ao) == float(an), (float(ao), float(an))
        assert float(io['q_mean']) == float(iN['q_mean'])
        print("OK")
    """)


def test_hit_matrix_per_column_eps_matches_numpy():
    """_hit_matrix with eps_scale: column j's radius = block_eps * eps_scale[j],
    verified against a numpy reference; eps_scale=None = the scalar path."""
    _run("""
        rng = np.random.default_rng(0)
        R = jnp.asarray(rng.normal(size=(7, OBS_D)).astype(np.float32) * 0.3)
        C = jnp.asarray(rng.normal(size=(9, OBS_D)).astype(np.float32) * 0.3)
        blocks = _parse_hit_spec('19:22:0.4')
        scale = jnp.asarray(rng.choice([0.5, 1.0, 2.0], size=9).astype(np.float32))
        got = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, blocks, jnp.float32,
                                                    eps_scale=scale))
        Rn, Cn, sn = np.asarray(R), np.asarray(C), np.asarray(scale)
        dist = np.linalg.norm(Rn[:, None, 19:22] - Cn[None, :, 19:22], axis=-1)
        exp = (dist <= 0.4 * sn[None, :]).astype(np.float32)
        assert np.array_equal(got, exp)
        # multi-block: every block scales by the SAME per-column factor
        blocks2 = _parse_hit_spec('19:22:0.4,24:26:0.2')
        got2 = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, blocks2, jnp.float32,
                                                     eps_scale=scale))
        d2 = np.linalg.norm(Rn[:, None, 24:26] - Cn[None, :, 24:26], axis=-1)
        exp2 = exp * (d2 <= 0.2 * sn[None, :]).astype(np.float32)
        assert np.array_equal(got2, exp2)
        # eps_scale=None == the old scalar expression, bit-identical
        g0 = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, blocks, jnp.float32))
        g1 = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, blocks, jnp.float32,
                                                   eps_scale=None))
        assert np.array_equal(g0, g1)
        print("OK")
    """)


def test_multiscale_identity_level_bit_identical_loss():
    """levels='0.4' (the default radius) at init: every column's radius is the
    base one and eps_gain is ONES, so measure_loss must be bit-identical to the
    knob-off cell -- the only param-tree change is the eps_gain module itself."""
    _run("""
        a0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg())
        a1 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg(measure_eps_multiscale='0.4'))
        k0 = {k for k, _ in flat_sd(a0.network.params)}
        k1 = {k for k, _ in flat_sd(a1.network.params)}
        extra = {k.split("'")[1] for k in (k1 - k0)}
        assert extra == {'modules_eps_gain'}, extra
        tab = a1.network.params['modules_eps_gain']['table']
        assert np.asarray(tab).shape == (1, 8) and (np.asarray(tab) == 1.0).all()
        b = make_batch(); key = jax.random.PRNGKey(5)
        l0, _ = a0.measure_loss(b, a0.network.params, key)
        l1, _ = a1.measure_loss(b, a1.network.params, key)
        assert float(l0) == float(l1), (float(l0), float(l1))
        print("OK")
    """)


def test_multiscale_trains_and_gains_get_gradient():
    """levels='0.2,0.4,0.8': loss finite and different from base (radii moved),
    the eps_gain table receives gradient, and update() runs end-to-end."""
    _run("""
        a0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg())
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                   zx_cfg(measure_eps_multiscale='0.2,0.4,0.8'))
        b = make_batch(); key = jax.random.PRNGKey(0)
        l0, _ = a0.measure_loss(b, a0.network.params, key)
        l1, _ = ag.measure_loss(b, ag.network.params, key)
        assert np.isfinite(float(l1))
        assert float(l1) != float(l0), "multi-level radii must move the hit matrix"
        g = jax.grad(lambda p: ag.measure_loss(b, p, key)[0])(ag.network.params)
        gt = np.asarray(g['modules_eps_gain']['table'])
        assert gt.shape == (3, 8)
        assert np.abs(gt).max() > 0, "eps_gain got no gradient"
        for _ in range(2):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_actor_greedy_frac1_q_is_phi_dot_z():
    """actor_z_greedy_frac=1.0: every sample's conditioning is the random sphere
    z and the q term is EXACTLY min-head F(s,a,z).z/sqrt(d) -- verified by
    replicating the rng chain outside the loss."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg(actor_z_greedy_frac=1.0))
        b = make_batch(); key = jax.random.PRNGKey(7)
        _, info = ag.actor_loss(b, ag.network.params, rng=key)
        # replicate: rng, x0, t, n, z = split(key, 5); gz, gp = split(z)
        _, _, _, n_rng, z_rng = jax.random.split(key, 5)
        gz_rng, gp_rng = jax.random.split(z_rng)
        obs = jnp.asarray(b['observations']); goals = jnp.asarray(b['actor_goals'])
        acts = jnp.asarray(b['actions'])
        d = 8
        z_rand = _sphere(jax.random.normal(gz_rng, (obs.shape[0], d)), d)
        assert bool(jax.random.bernoulli(gp_rng, 1.0, (obs.shape[0], 1)).all())
        noise = jax.random.normal(n_rng, acts.shape)
        a_act = ag.network.select('actor')(obs, z_rand, noise)
        _, phi, _ = ag.network.select('critic')(obs, goals, a_act, goals,
                                                info=True, z_vec=z_rand)
        qg = jnp.einsum('hid,id->hi', phi, z_rand) / jnp.sqrt(d)
        q_exp = float(jnp.minimum(qg[0], qg[1]).mean())
        assert float(info['q_mean']) == q_exp, (float(info['q_mean']), q_exp)
        print("OK")
    """)


def test_actor_greedy_frac0_and_training_step():
    """frac=0.0 == base bit-identically (static guard); frac=0.3 trains: finite
    losses through update(), and the actor loss differs from base."""
    _run("""
        a0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg())
        aF = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg(actor_z_greedy_frac=0.0))
        b = make_batch(); key = jax.random.PRNGKey(1)
        l0, _ = a0.actor_loss(b, a0.network.params, rng=key)
        lf, _ = aF.actor_loss(b, aF.network.params, rng=key)
        assert float(l0) == float(lf)
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg(actor_z_greedy_frac=0.3))
        lg, ig = ag.actor_loss(b, ag.network.params, rng=key)
        assert np.isfinite(float(lg)) and float(lg) != float(l0)
        for _ in range(2):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_lattice_refusals():
    """Mis-compositions refused at construction, mirroring measure_hit_spec's
    discipline (silent-wrong training is worse than a loud assert)."""
    _run("""
        cases = [
            # multiscale without first passage
            zx_cfg(measure_eps_multiscale='0.2,0.4', measure_first_passage=False),
            # multiscale on the expectile tower
            zx_cfg(measure_eps_multiscale='0.2,0.4', measure_expectile=True),
            # multiscale under hindsight columns
            zx_cfg(measure_eps_multiscale='0.2,0.4', measure_testpoint_dist='hindsight'),
            # multiscale with a task_prior arm
            zx_cfg(measure_eps_multiscale='0.2,0.4', task_prior_actor='mix',
                   fb_forward_z=False, actor_cond='goal', actor_z_from_critic=False),
            # actor knob without fb_forward_z
            zx_cfg(actor_z_greedy_frac=0.3, fb_forward_z=False),
            # actor knob without the z-actor
            zx_cfg(actor_z_greedy_frac=0.3, actor_cond='goal', actor_z_from_critic=False),
            # actor knob out of range
            zx_cfg(actor_z_greedy_frac=1.5),
        ]
        for cfg in cases:
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
            except AssertionError:
                pass
            else:
                raise AssertionError(f'accepted a refused composition: '
                                     f'ms={cfg.get("measure_eps_multiscale")!r} '
                                     f'azg={cfg.get("actor_z_greedy_frac")!r}')
        print("OK")
    """)


def test_nhat_head_off_bit_identical():
    """measure_nhat_head absent/default => identical param tree + bit-identical
    measure_loss (the phase-2b upgrade must not disturb existing cells)."""
    _run("""
        cfg_new = zx_cfg()
        assert cfg_new["measure_nhat_head"] is False
        cfg_old = zx_cfg(); cfg_old.pop("measure_nhat_head")
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old.network.params), flat_sd(a_new.network.params)
        assert [k for k, _ in fo] == [k for k, _ in fn]
        for (k, lo), (_, ln) in zip(fo, fn):
            assert (lo == ln).all(), f"param {k} differs"
        b = make_batch(); key = jax.random.PRNGKey(3)
        lo, _ = a_old.measure_loss(b, a_old.network.params, key)
        ln, _ = a_new.measure_loss(b, a_new.network.params, key)
        assert float(lo) == float(ln)
        print("OK")
    """)


def test_nhat_head_on_isolated_and_positive():
    """Knob on: every PRE-EXISTING measure term is bit-identical to the base
    cell; total loss = base + nhat_loss exactly; softplus keeps predictions > 0;
    gradient reaches ONLY the head (all other modules' measure-grads match the
    base cell bit for bit)."""
    _run("""
        a0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg())
        aN = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg(measure_nhat_head=True))
        extra = {k.split("'")[1] for k in
                 ({k for k, _ in flat_sd(aN.network.params)}
                  - {k for k, _ in flat_sd(a0.network.params)})}
        assert extra == {'modules_nhat_head'}, extra
        b = make_batch(); key = jax.random.PRNGKey(4)
        l0, i0 = a0.measure_loss(b, a0.network.params, key)
        lN, iN = aN.measure_loss(b, aN.network.params, key)
        for term in ('measure_offdiag', 'measure_diag', 'measure_orth'):
            assert float(i0[term]) == float(iN[term]), term
        nh = np.float32(iN['measure_nhat_loss'])
        assert float(lN) == float(np.float32(l0) + nh), (float(lN), float(l0), float(nh))
        assert float(iN['measure_nhat_pred_mean']) > 0.0        # softplus
        g0 = jax.grad(lambda p: a0.measure_loss(b, p, key)[0])(a0.network.params)
        gN = jax.grad(lambda p: aN.measure_loss(b, p, key)[0])(aN.network.params)
        for mod in g0.keys():
            f0, fN = flat_sd(g0[mod]), flat_sd(gN[mod])
            for (k, x), (_, y) in zip(f0, fN):
                assert (x == y).all(), f"grad leaked into {mod}{k}"
        gh = flat_sd(gN['modules_nhat_head'])
        assert any(np.abs(x).max() > 0 for _, x in gh), "nhat head got no gradient"
        print("OK")
    """)


def test_nhat_loss_decreases_on_planted_pattern():
    """A batch with a PLANTED column-mass pattern (a dense cube cluster => some
    columns' hit mass >> others): repeated update() steps must reduce the nhat
    regression loss."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg(measure_nhat_head=True))
        rng = np.random.default_rng(0)
        B = 64
        b = make_batch(B=B, seed=0)
        nobs = b['next_observations']
        nobs[: B // 2, 19:22] = rng.normal(0, 0.1, (B // 2, 3))   # dense cluster
        nobs[B // 2:, 19:22] = rng.uniform(-3, 3, (B - B // 2, 3))  # sparse spread
        first = None
        for step in range(60):
            ag, info = ag.update(b)
            if first is None:
                first = float(info['measure/measure_nhat_loss'])
        last = float(info['measure/measure_nhat_loss'])
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first, (first, last)
        assert float(info['measure/measure_nhat_pred_mean']) > 0.0
        print("OK")
    """)


def test_nhat_lattice_and_azg_composition():
    """The lattice refuses nhat without first-passage / with multiscale, and
    PERMITS the adopted-arm composition measure_nhat_head + actor_z_greedy_frac."""
    _run("""
        for cfg in (zx_cfg(measure_nhat_head=True, measure_first_passage=False),
                    zx_cfg(measure_nhat_head=True, measure_eps_multiscale='0.2,0.4'),
                    zx_cfg(measure_nhat_head=True, measure_testpoint_dist='hindsight')):
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
            except AssertionError:
                pass
            else:
                raise AssertionError('accepted a refused nhat composition')
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                   zx_cfg(measure_nhat_head=True,
                                          actor_z_greedy_frac=0.3))
        b = make_batch(); key = jax.random.PRNGKey(6)
        l, info = ag.measure_loss(b, ag.network.params, key)
        assert np.isfinite(float(l)) and 'measure_nhat_loss' in info
        for _ in range(2):
            ag, up = ag.update(b)
        for k, v in up.items():
            assert np.isfinite(float(v)), f"{k} not finite"
        print("OK")
    """)


def test_warm_start_fresh_inits_eps_gain_only():
    """warm_start from a checkpoint that PREDATES the knob: every shared module is
    spliced, eps_gain alone is fresh-init (ones) -- the phase-2 upgrade path."""
    _run("""
        import os, pickle, tempfile
        a0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, zx_cfg())
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, 'ckpt_run'); os.makedirs(run_dir)
            params = flax.serialization.to_state_dict(a0.network.params)
            with open(os.path.join(run_dir, 'params_0.pkl'), 'wb') as f:
                pickle.dump({'agent': {'network': {'params': params}}}, f)
            ag = GCIQLFlowAgent.create(
                1, EX_OBS, EX_ACT,
                zx_cfg(measure_eps_multiscale='0.2,0.4,0.8',
                       warm_start_ckpt=run_dir, warm_start_epoch=0))
        # spliced: critic params equal the checkpoint's, bit for bit
        ck = flat_sd(a0.network.params['modules_critic'])
        nw = flat_sd(ag.network.params['modules_critic'])
        for (k, lo), (_, ln) in zip(ck, nw):
            assert (lo == ln).all(), f"critic {k} not spliced"
        # fresh: eps_gain at its ones init
        tab = np.asarray(ag.network.params['modules_eps_gain']['table'])
        assert tab.shape == (3, 8) and (tab == 1.0).all()
        b = make_batch(); key = jax.random.PRNGKey(2)
        l, _ = ag.measure_loss(b, ag.network.params, key)
        assert np.isfinite(float(l))
        print("OK")
    """)
