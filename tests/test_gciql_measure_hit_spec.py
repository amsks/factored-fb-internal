"""`measure_hit_spec` — the env-portable goal-region kernel (spec 08 E7).

The two-phase curriculum's hit indicator h(s', g) was hardcoded to cube-single's
cube xyz at obs[19:22] with eps=0.4 obs units (=4cm). Porting phase 1 / phase 2 to
`cube-double-play-v0` (two cubes: obs[19:22] + obs[28:31]) and `scene-play-v0`
(cube + 2 button one-hots + drawer + window) needs the region to be a CONJUNCTION
over several observation blocks, each with its own radius.

`measure_hit_spec` is a STRING (ml_collections rejects list overrides -- see the
spec06 launch gotcha) of comma-separated `start:stop:eps` blocks:

    '19:22:0.4'                       cube-single  (the DEFAULT, bit-identical)
    '19:22:0.4,28:31:0.4'             cube-double  (both cubes within 4cm)
    '19:22:0.4,28:30:0.5,...'         scene        (cube + buttons + drawer + window)

Discipline (mirrors tests/test_gciql_measure_axes.py): the default must be
byte-identical in the param tree AND bit-identical in measure_loss, so every
existing cube-single checkpoint keeps loading and reproducing.

Runs in the jax venv subprocess.
"""
import textwrap

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

OBS_D = 40  # scene-sized, so every block in the specs below is in range
COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec

    OBS_D = {OBS_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)
    CUBE_SINGLE = '19:22:0.4'
    CUBE_DOUBLE = '19:22:0.4,28:31:0.4'
    SCENE = '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'

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
        # scale the object dims down so a 0.4-radius ball actually fires sometimes.
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs[:, 19:] *= 0.25
        return dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
            masks=rng.integers(0, 2, size=B).astype(np.float32),
        )

    def flat_sd(agent):
        sd = flax.serialization.to_state_dict(agent.network.params)
        leaves = jax.tree_util.tree_flatten_with_path(sd)[0]
        return [(jax.tree_util.keystr(p), np.asarray(l)) for p, l in leaves]
"""


def _run(code):
    code = textwrap.dedent(COMMON) + textwrap.dedent(code)
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_parser():
    """`start:stop:eps` blocks; whitespace tolerated; malformed specs rejected loudly."""
    _run("""
        assert _parse_hit_spec(CUBE_SINGLE) == ((19, 22, 0.4),)
        assert _parse_hit_spec(CUBE_DOUBLE) == ((19, 22, 0.4), (28, 31, 0.4))
        assert _parse_hit_spec(' 19:22:0.4 , 28:31:0.4 ') == ((19, 22, 0.4), (28, 31, 0.4))
        assert len(_parse_hit_spec(SCENE)) == 5
        for bad in ('', '19:22', '19:22:0.4:9', 'a:b:c', '22:19:0.4', '19:22:-1'):
            try:
                _parse_hit_spec(bad)
            except (ValueError, AssertionError):
                pass
            else:
                raise AssertionError(f'accepted malformed spec {bad!r}')
        print("OK")
    """)


def test_default_bit_identical_and_checkpoint_compatible():
    """Default spec (and a config dict that PREDATES the knob) => identical param tree
    and bit-identical measure_loss. Old cube-single checkpoints must be unaffected."""
    _run("""
        cfg_new = base_cfg()
        assert cfg_new["measure_hit_spec"] == CUBE_SINGLE, cfg_new["measure_hit_spec"]
        cfg_old = base_cfg(); cfg_old.pop("measure_hit_spec")      # pre-E7 config
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        b = make_batch(); key = jax.random.PRNGKey(3)
        lo, _ = a_old.measure_loss(b, a_old.network.params, key)
        ln, _ = a_new.measure_loss(b, a_new.network.params, key)
        assert float(lo) == float(ln), (float(lo), float(ln))
        print("OK")
    """)


def test_hit_matrix_is_the_conjunction():
    """_hit_matrix = AND over blocks of (||row[a:b] - col[a:b]|| <= eps), matching numpy."""
    _run("""
        rng = np.random.default_rng(0)
        R = jnp.asarray(rng.normal(size=(7, OBS_D)).astype(np.float32) * 0.3)
        C = jnp.asarray(rng.normal(size=(9, OBS_D)).astype(np.float32) * 0.3)
        for spec in (CUBE_SINGLE, CUBE_DOUBLE, SCENE):
            blocks = _parse_hit_spec(spec)
            got = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, blocks, jnp.float32))
            Rn, Cn = np.asarray(R), np.asarray(C)
            exp = np.ones((7, 9), bool)
            for (a, bb, e) in blocks:
                exp &= np.linalg.norm(Rn[:, None, a:bb] - Cn[None, :, a:bb], axis=-1) <= e
            assert got.shape == (7, 9), got.shape
            assert np.array_equal(got, exp.astype(np.float32)), spec
        # a block that always fires is a no-op; one that never fires zeroes the matrix.
        one = _parse_hit_spec(CUBE_SINGLE)
        wide = _parse_hit_spec(CUBE_SINGLE + ',28:31:1e6')
        never = _parse_hit_spec(CUBE_SINGLE + ',28:31:0.0')
        h1 = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, one, jnp.float32))
        hw = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, wide, jnp.float32))
        hn = np.asarray(GCIQLFlowAgent._hit_matrix(R, C, never, jnp.float32))
        assert np.array_equal(h1, hw), "an always-true block must not change the region"
        assert (hn == 0).all(), "a never-true block must empty the region"
        print("OK")
    """)


def test_multiblock_specs_train():
    """cube-double and scene specs: measure_loss finite, param tree unchanged, the loss
    genuinely differs from cube-single's, and update() runs end-to-end."""
    _run("""
        a_ref = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg())
        b = make_batch(); key = jax.random.PRNGKey(0)
        l_ref, _ = a_ref.measure_loss(b, a_ref.network.params, key)
        for spec in (CUBE_DOUBLE, SCENE):
            ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(measure_hit_spec=spec))
            assert [k for k, _ in flat_sd(a_ref)] == [k for k, _ in flat_sd(ag)], spec
            l, _ = ag.measure_loss(b, ag.network.params, key)
            assert np.isfinite(float(l)), (spec, l)
            assert float(l) != float(l_ref), f"{spec} must narrow the region"
            for _ in range(2):
                ag, info = ag.update(b)
            for k, v in info.items():
                assert np.isfinite(float(v)), f"{spec} {k} not finite: {v}"
        print("OK")
    """)


def test_ballvisit_phase2_composes_with_multiblock():
    """Phase 2 (measure_fp_absorbing=False) is the same region under a non-absorbing
    backup -- the multi-block spec must compose with it and with frozen psi."""
    _run("""
        cfg = base_cfg(measure_hit_spec=SCENE, measure_fp_absorbing=False,
                       measure_stopgrad_psi=True)
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
        b = make_batch(); key = jax.random.PRNGKey(2)
        l, _ = ag.measure_loss(b, ag.network.params, key)
        assert np.isfinite(float(l)), l
        # frozen psi: no gradient into the psi (goal) tower of the critic.
        g = jax.grad(lambda p: ag.measure_loss(b, p, key)[0])(ag.network.params)
        leaves = jax.tree_util.tree_flatten_with_path(
            flax.serialization.to_state_dict(g))[0]
        psi_g = [np.asarray(v) for p, v in leaves
                 if "critic" in jax.tree_util.keystr(p) and "psi" in jax.tree_util.keystr(p)]
        assert psi_g, "no psi leaves found -- the grad-tree key convention changed"
        assert all(np.abs(x).max() == 0 for x in psi_g), "measure_stopgrad_psi leaked gradient"
        for _ in range(2):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_nondefault_spec_refuses_cube_single_task_priors():
    """The task_prior / tpactor mixture arms still read cube-single's obs[19:22] with a
    hardcoded 0.4. They are OFF in the two-phase cells; the assert-lattice must refuse
    the silent-wrong-env composition rather than train a mis-specified reward."""
    _run("""
        for kw in (dict(task_prior='mix'), dict(task_prior_actor='mix')):
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                      base_cfg(measure_hit_spec=CUBE_DOUBLE, **kw))
            except AssertionError:
                pass
            else:
                raise AssertionError(f'accepted {kw} under a non-cube-single hit spec')
        print("OK")
    """)
