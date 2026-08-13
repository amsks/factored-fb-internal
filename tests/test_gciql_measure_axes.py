"""Spec 06 measure-axis knobs (items 5/6/7) on gciql_flow.py's measure path:
  measure_first_passage  (item 5) -- absorbing first-passage residual, cube-xyz hit.
  measure_expectile      (item 6) -- state-only measure value + in-sample backup (+ tower).
  measure_testpoint_dist (item 7) -- hindsight (value_goals) test-point columns.

Discipline (mirrors tests/test_gciql_rgfb.py): all default-OFF => byte-identical param
tree AND bit-identical measure_loss on a fixed batch (old checkpoints keep loading);
each knob ON has the specified effect; the assert-lattice refuses mis-compositions.

obs_dim = 29 here (NOT the 11 of the RG-FB test): the first-passage hit reads the cube
xyz at obs slice [19:22], so the batch must actually have those dims.

Runs in the jax venv subprocess.
"""
import textwrap

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

OBS_D = 29
COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config

    OBS_D = {OBS_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def base_cfg(**kw):
        # plain measure-ONLY cell: bilinear critic, state-action SM on the critic legs.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["measure_legs"] = "critic"
        cfg.update(kw)
        return cfg

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        return dict(
            observations=rng.normal(size=(B, OBS_D)).astype(np.float32),
            next_observations=rng.normal(size=(B, OBS_D)).astype(np.float32),
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


def test_all_off_bit_identical_and_checkpoint_compatible():
    """Default OFF (and a config that PREDATES all three knobs) => identical param tree
    and bit-identical measure_loss on a fixed batch."""
    _run("""
        cfg_old = base_cfg()
        for k in ("measure_first_passage", "measure_expectile",
                  "measure_expectile_tau", "measure_testpoint_dist"):
            cfg_old.pop(k, None)                        # a pre-spec06 config dict
        cfg_new = base_cfg()
        assert cfg_new["measure_first_passage"] is False
        assert cfg_new["measure_expectile"] is False
        assert cfg_new["measure_testpoint_dist"] == "batch"
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        # no measure_v_phi tower in the default tree.
        assert not any("measure_v_phi" in k for k, _ in fn), "measure_v_phi leaked into default tree"
        b = make_batch(); key = jax.random.PRNGKey(3)
        lo, _ = a_old.measure_loss(b, a_old.network.params, key)
        ln, _ = a_new.measure_loss(b, a_new.network.params, key)
        assert float(lo) == float(ln), (float(lo), float(ln))
        print("OK")
    """)


def test_item5_first_passage_effect():
    """measure_first_passage=True: finite measure_loss, param tree UNCHANGED (no new
    module), and the loss differs from the plain residual on the same batch."""
    _run("""
        base = base_cfg()
        fp = base_cfg(measure_first_passage=True)
        a0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base)
        a1 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp)
        # same param tree (first-passage adds no module).
        assert [k for k, _ in flat_sd(a0)] == [k for k, _ in flat_sd(a1)]
        b = make_batch(); key = jax.random.PRNGKey(0)
        l0, _ = a0.measure_loss(b, a0.network.params, key)
        l1, i1 = a1.measure_loss(b, a1.network.params, key)
        assert np.isfinite(float(l1)), l1
        assert float(l0) != float(l1), "first-passage must change the residual"
        # Review finding 3: with hit supplied, the diagonal is a SQUARED residual (hit is a
        # representable reward -- no linear attraction), so diagt >= 0 and equals
        # mean(diag(M - hit - gamma*(1-hit)*tM)^2).
        rng = np.random.default_rng(0)
        Mm = jnp.asarray(rng.normal(size=(2, 6, 6)).astype(np.float32))
        tMm = jnp.asarray(rng.normal(size=(2, 6, 6)).astype(np.float32))
        hit1 = jnp.ones((6, 6), jnp.float32)
        _, dg = GCIQLFlowAgent._measure_terms(Mm, tMm, 0.99, False, hit=hit1)
        exp = np.mean(np.diagonal(np.asarray(Mm) - 1.0, axis1=-2, axis2=-1) ** 2, axis=-1)
        assert np.allclose(np.asarray(dg), exp, atol=1e-5), (np.asarray(dg), exp)
        assert (np.asarray(dg) >= 0).all(), "first-passage diagonal must be a squared residual"
        # update() runs end-to-end (finite) under measure_only + first-passage.
        ag = a1
        for _ in range(2):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_item6_expectile_tower_grad_and_compose():
    """measure_expectile=True: adds the measure_v_phi tower; measure_loss finite and
    sends gradient into measure_v_phi; composes with first-passage; update() runs."""
    _run("""
        # composed cell (default per spec: first-passage supplies r_delta + mask, bounded).
        cfg = base_cfg(measure_expectile=True, measure_first_passage=True)
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
        assert any("measure_v_phi" in k for k, _ in flat_sd(ag)), "measure_v_phi tower missing"
        b = make_batch(); key = jax.random.PRNGKey(1)
        l, info = ag.measure_loss(b, ag.network.params, key)
        assert np.isfinite(float(l)), l
        # gradient reaches measure_v_phi (the V_M expectile term trains it).
        g = jax.grad(lambda p: ag.measure_loss(b, p, key)[0])(ag.network.params)
        mv = [np.asarray(x) for kk, x in
              jax.tree_util.tree_flatten_with_path(g["modules_measure_v_phi"])[0]]
        mx = max(float(np.abs(x).max()) for x in mv)
        assert mx > 0.0, "measure_expectile sends no gradient into measure_v_phi"
        for _ in range(2):
            ag, i2 = ag.update(b)
        for k, v in i2.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_item7_hindsight_testpoints_effect():
    """measure_testpoint_dist='hindsight': finite, differs from 'batch' (columns moved
    to psi(value_goals)); tree unchanged."""
    _run("""
        a0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg())
        a1 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(measure_testpoint_dist="hindsight"))
        assert [k for k, _ in flat_sd(a0)] == [k for k, _ in flat_sd(a1)]
        b = make_batch(); key = jax.random.PRNGKey(0)
        l0, _ = a0.measure_loss(b, a0.network.params, key)
        l1, _ = a1.measure_loss(b, a1.network.params, key)
        assert np.isfinite(float(l1)), l1
        assert float(l0) != float(l1), "hindsight test points must change the loss"
        print("OK")
    """)


def test_cellw_knobs_and_affine_identity():
    """Cell W (spec 06 re-center): (a) knob defaults are bit-identical to the pre-knob
    corner cell (guards the RUNNING s06corner arm); (b) w_off=0 / target_min change the
    loss, stay finite, update() runs on the full Cell W combo; (c) the affine identity
    delta_M = (1-gamma)*delta_Q holds numerically (the theorem behind the win cell);
    (d) refusals."""
    _run("""
        corner = dict(measure_first_passage=True, measure_expectile=True,
                      measure_ortho_coef=1.0, measure_testpoint_dist="hindsight")
        # (a) defaults bit-identical to a config that PREDATES the Cell W knobs.
        cfg_old = base_cfg(**corner)
        for k in ("measure_offdiag_weight", "measure_expectile_source"):
            cfg_old.pop(k, None)
        cfg_new = base_cfg(**corner)
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        b = make_batch(); key = jax.random.PRNGKey(7)
        lo, _ = a_old.measure_loss(b, a_old.network.params, key)
        ln, _ = a_new.measure_loss(b, a_new.network.params, key)
        assert float(lo) == float(ln), (float(lo), float(ln))

        # (b) Cell W combo: w=0 and target_min each change the loss; update() finite.
        aw0 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(
            **corner, measure_offdiag_weight=0.0, measure_expectile_source="target_min"))
        lw0, _ = aw0.measure_loss(b, aw0.network.params, key)
        assert np.isfinite(float(lw0)) and float(lw0) != float(ln)
        ag = aw0
        for _ in range(2):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"

        # (c) affine identity: delta_M = (1-g)*delta_Q with M=1+(1-g)Q, VM=1+(1-g)VQ.
        rng = np.random.default_rng(0); g = 0.99
        Q = rng.normal(size=(2, 5, 5)); VQ = rng.normal(size=(2, 5, 5))
        hit = (rng.random((5, 5)) < 0.3).astype(np.float64)
        M = 1.0 + (1.0 - g) * Q; VM = 1.0 + (1.0 - g) * VQ
        dM = M - hit - g * (1.0 - hit) * VM
        dQ = Q - (hit - 1.0) - g * (1.0 - hit) * VQ
        assert np.allclose(dM, (1.0 - g) * dQ, atol=1e-12), "affine identity violated"

        # (d) refusals.
        def refuses(msg_frag, **kw):
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(**kw))
            except AssertionError as e:
                assert msg_frag in str(e), (msg_frag, str(e))
                return
            raise SystemExit(f"expected AssertionError ({msg_frag}) for {kw}")
        refuses("require measure_expectile=True", measure_offdiag_weight=0.3)
        refuses("unknown measure_expectile_source", measure_expectile=True,
                measure_ortho_coef=1.0, measure_expectile_source="bogus")
        print("OK")
    """)


def test_assert_lattice():
    """Mis-compositions refuse at construction."""
    _run("""
        def refuses(msg_frag, **kw):
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(**kw))
            except AssertionError as e:
                assert msg_frag in str(e), (msg_frag, str(e))
                return
            raise SystemExit(f"expected AssertionError ({msg_frag}) for {kw}")

        # spec06 axis needs the state-action (critic) legs.
        refuses("measure_legs='critic'", measure_first_passage=True, measure_legs="value")
        # spec06 axis needs the rho/full single-matrix test set (not task-norm).
        refuses("single-matrix rho/full", measure_first_passage=True,
                measure_test_states="goals", measure_test_rank="task")
        # unmasked expectile needs the FB ortho anchor.
        refuses("ortho anchor", measure_expectile=True, measure_ortho_coef=0.0)
        # unmasked expectile + hindsight test points has no coherent reward (finding 5/6).
        refuses("no coherent reward", measure_expectile=True, measure_ortho_coef=1.0,
                measure_testpoint_dist="hindsight")
        # (z,x)-separation 2026-07-23: fb_forward_z now COMPOSES with first-passage
        # (rows carry a policy index z, columns the query x); the guard remains for the
        # non-fp axes and for the expectile-measure branch.
        GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(measure_first_passage=True, fb_forward_z=True))
        refuses("incompatible with fb_forward_z", measure_testpoint_dist="hindsight", fb_forward_z=True)
        refuses("plain first-passage/visitation residual only", measure_first_passage=True,
                measure_expectile=True, fb_forward_z=True)
        # unknown testpoint dist.
        refuses("unknown measure_testpoint_dist", measure_testpoint_dist="bogus")
        # needs a measure objective at all.
        cfg = base_cfg(measure_first_passage=True); cfg["measure_only"] = False
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
        except AssertionError as e:
            assert "measure_only=True or measure_coef>0" in str(e), str(e)
        else:
            raise SystemExit("expected refusal for spec06 axis without a measure objective")
        print("OK")
    """)
