"""Phase-2 occupancy knobs (2026-07-23) on gciql_flow.py:
  measure_stopgrad_psi -- freeze the backward map: stop-grad every ONLINE psi column in
                          measure_loss so the measure TD trains only phi (phase-2 cell:
                          warm-start from the FP-NoOrtho ckpt, occupancy TD on frozen psi).
  actor_q_center       -- flow actor q-loss normalized by the CENTERED magnitude
                          sg[mean|q - mean(q)|] (affine-offset invariant; the Cell W
                          M = 1+(1-gamma)Q dilution fix from the wave-2 forensics).

Discipline (mirrors tests/test_gciql_measure_axes.py): both default-OFF => byte-identical
param tree AND bit-identical losses on a fixed batch; each knob ON has the specified
gradient/normalizer effect.

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


def test_knobs_off_bit_identical():
    """Defaults (and a config that PREDATES both knobs) => identical param tree and
    bit-identical measure_loss + actor_loss on a fixed batch."""
    _run("""
        cfg_old = base_cfg()
        for k in ("measure_stopgrad_psi", "actor_q_center"):
            cfg_old.pop(k, None)                        # a pre-knob config dict
        cfg_new = base_cfg()
        assert cfg_new["measure_stopgrad_psi"] is False
        assert cfg_new["actor_q_center"] is False
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        mo = a_old.measure_loss(batch, a_old.network.params, rng)[0]
        mn = a_new.measure_loss(batch, a_new.network.params, rng)[0]
        assert float(mo) == float(mn), (float(mo), float(mn))
        ao = a_old.actor_loss(batch, a_old.network.params, rng=rng)[0]
        an = a_new.actor_loss(batch, a_new.network.params, rng=rng)[0]
        assert float(ao) == float(an), (float(ao), float(an))
        print("OK")
    """)


def test_stopgrad_psi_freezes_backward_map():
    """measure_stopgrad_psi=True: measure_loss gradient w.r.t. every critic psi-tower
    param is EXACTLY zero while some phi-tower grads stay nonzero; loss VALUE unchanged."""
    _run("""
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        a_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg())
        a_on = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(measure_stopgrad_psi=True))
        lo = a_off.measure_loss(batch, a_off.network.params, rng)[0]
        ln = a_on.measure_loss(batch, a_on.network.params, rng)[0]
        assert float(lo) == float(ln), "stop_gradient must not change the loss VALUE"
        g = jax.grad(lambda p: a_on.measure_loss(batch, p, rng)[0])(a_on.network.params)
        leaves = jax.tree_util.tree_flatten_with_path(
            flax.serialization.to_state_dict(g))[0]
        psi = [(jax.tree_util.keystr(p), np.asarray(l)) for p, l in leaves
               if 'critic' in jax.tree_util.keystr(p) and 'psi' in jax.tree_util.keystr(p)]
        phi = [(jax.tree_util.keystr(p), np.asarray(l)) for p, l in leaves
               if 'critic' in jax.tree_util.keystr(p) and 'phi' in jax.tree_util.keystr(p)]
        assert psi and phi, "expected critic phi/psi tower params in the grad tree"
        for k, l in psi:
            assert (l == 0).all(), f"psi grad leaked through: {k}"
        assert any((l != 0).any() for _, l in phi), "phi got NO gradient - loss is dead"
        print("OK")
    """)


def test_actor_q_center_changes_only_the_normalizer():
    """actor_q_center=True changes the actor loss (different normalizer) but leaves
    the measure_loss and the param tree untouched."""
    _run("""
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        a_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg())
        a_on = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(actor_q_center=True))
        fo, fn = flat_sd(a_off), flat_sd(a_on)
        assert [k for k, _ in fo] == [k for k, _ in fn]
        mo = a_off.measure_loss(batch, a_off.network.params, rng)[0]
        mn = a_on.measure_loss(batch, a_on.network.params, rng)[0]
        assert float(mo) == float(mn)
        lo, io = a_off.actor_loss(batch, a_off.network.params, rng=rng)
        ln, in_ = a_on.actor_loss(batch, a_on.network.params, rng=rng)
        assert np.isfinite(float(lo)) and np.isfinite(float(ln))
        # same q (identical params/rng) => the two losses differ EXACTLY by the q-term
        # renormalization: (lo - bc) * |q|_mean = (ln - bc) * |q - mean(q)|_mean is hard
        # to extract; assert the losses differ and q diagnostics are identical.
        assert float(io['q_mean']) == float(in_['q_mean'])
        assert float(lo) != float(ln), "centered normalizer had no effect on this batch"
        print("OK")
    """)


def test_fp_absorbing_knob():
    """measure_fp_absorbing default True is bit-identical FP; False (ball-visitation)
    changes the loss and its per-column fixed point exceeds the FP bound of 1."""
    _run("""
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        fp = base_cfg(measure_first_passage=True, measure_scale_norm=True)
        a_fp = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp)
        a_def = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                      base_cfg(measure_first_passage=True,
                                               measure_scale_norm=True,
                                               measure_fp_absorbing=True))
        cfg_old = dict(fp); cfg_old.pop("measure_fp_absorbing", None)
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        lf = a_fp.measure_loss(batch, a_fp.network.params, rng)[0]
        ld = a_def.measure_loss(batch, a_def.network.params, rng)[0]
        lo = a_old.measure_loss(batch, a_old.network.params, rng)[0]
        assert float(lf) == float(ld) == float(lo), "default/absent knob must be bit-identical"
        a_vis = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                      base_cfg(measure_first_passage=True,
                                               measure_scale_norm=True,
                                               measure_fp_absorbing=False))
        lv = a_vis.measure_loss(batch, a_vis.network.params, rng)[0]
        assert np.isfinite(float(lv)) and float(lv) != float(lf), \
            "visitation form must change the loss"
        # residual algebra check on the static helper: with hit=1 everywhere the
        # absorbing residual ignores tM entirely; the visitation residual keeps it.
        M = jnp.ones((4, 4)); tM = jnp.full((4, 4), 5.0); hit = jnp.ones((4, 4))
        offa, _ = GCIQLFlowAgent._measure_terms(M, tM, 0.99, False, hit=hit, absorbing=True)
        offv, _ = GCIQLFlowAgent._measure_terms(M, tM, 0.99, False, hit=hit, absorbing=False)
        assert float(offa) == 0.0, "absorbing: M=hit=1 is the exact fixed point"
        assert float(offv) > 0.0, "visitation: bootstrap must still flow through tM"
        print("OK")
    """)


# --- spec 08 E2: freeze_actor (frozen-B / frozen-actor phase-2 ablation) ---------------
# Mechanism mirrors freeze_critic_ckpt (create(): optax.multi_transform + set_to_zero),
# but masks the ACTOR modules (modules_actor + modules_actor_vf) so phase 2 becomes
# fixed-policy visitation evaluation: pi stays at the warm-started phase-1 policy while phi
# (the forward tower) still trains against the frozen backward map.

def _fp_helper():
    # a flat {keystr: ndarray} view of a params tree, shared by the E2 tests.
    return """
        def fp(params):
            leaves = jax.tree_util.tree_flatten_with_path(
                flax.serialization.to_state_dict(params))[0]
            return {jax.tree_util.keystr(p): np.asarray(l) for p, l in leaves}
    """


def test_freeze_actor_off_bit_identical():
    """Default (and a pre-knob config) => freeze_actor False, identical param tree and
    bit-identical total_loss on a fixed batch."""
    _run("""
        cfg_old = base_cfg(); cfg_old.pop("freeze_actor", None)
        cfg_new = base_cfg()
        assert cfg_new.get("freeze_actor", False) is False
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        lo = a_old.total_loss(batch, a_old.network.params, rng)[0]
        ln = a_new.total_loss(batch, a_new.network.params, rng)[0]
        assert float(lo) == float(ln), (float(lo), float(ln))
        print("OK")
    """)


def test_freeze_actor_off_actor_still_trains():
    """Control for the ON test: with freeze_actor off, one optimizer step MOVES the
    actor modules."""
    _run(_fp_helper() + """
        a = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg())
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        g = jax.grad(lambda p: a.total_loss(batch, p, rng)[0])(a.network.params)
        stepped = a.network.apply_gradients(grads=g)
        b, aft = fp(a.network.params), fp(stepped.params)
        amod = [k for k in b if "actor" in k]
        assert amod, "no actor params found"
        assert any((b[k] != aft[k]).any() for k in amod), "actor must move when not frozen"
        print("OK")
    """)


def test_freeze_actor_freezes_policy_trains_critic():
    """freeze_actor=True: after one optimizer step EVERY actor param is byte-unchanged
    (pi frozen at the warm-started policy) while critic phi still moves (phi trains)."""
    _run(_fp_helper() + """
        # measure_coef=1.0 so total_loss actually feeds phi a measure signal (the phase-2
        # cell value); with the get_config default 0.0 phi has no signal to train from.
        a = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, base_cfg(freeze_actor=True, measure_coef=1.0))
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        g = jax.grad(lambda p: a.total_loss(batch, p, rng)[0])(a.network.params)
        stepped = a.network.apply_gradients(grads=g)
        b, aft = fp(a.network.params), fp(stepped.params)
        amod = [k for k in b if "actor" in k]
        cphi = [k for k in b if "critic" in k and "phi" in k]
        assert amod and cphi, (len(amod), len(cphi))
        for k in amod:
            assert (b[k] == aft[k]).all(), f"actor param moved despite freeze_actor: {k}"
        assert any((b[k] != aft[k]).any() for k in cphi), "critic phi must still train"
        print("OK")
    """)


# --- spec 08 E6: measure_z_random_frac_anneal (ramp random-z prob 0 -> frac) ------------
# Only the bernoulli PROBABILITY is annealed (a traced scalar from network.step); the branch
# stays static on the config max so knob-off is bit-identical. Needs the ZX form (fb_forward_z)
# for the random-z path to be active at all.

def test_zfrac_anneal_off_is_step_independent_and_bit_identical():
    """No anneal knob (or =False): measure_loss does not depend on the training step, and a
    pre-knob config is bit-identical."""
    _run("""
        cfg = base_cfg(fb_forward_z=True, actor_cond="z", actor_z_from_critic=True,
                       measure_first_passage=True, measure_z_random_frac=0.5)
        a = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        def loss_at(agent, step):
            ag = agent.replace(network=agent.network.replace(step=jnp.asarray(step, jnp.int32)))
            return float(ag.measure_loss(batch, ag.network.params, rng)[0])
        assert loss_at(a, 0) == loss_at(a, 400000), "off: measure_loss must be step-independent"
        cfg_old = dict(cfg); cfg_old.pop("measure_z_random_frac_anneal", None)
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        assert float(a_old.measure_loss(batch, a_old.network.params, rng)[0]) == loss_at(a, 0)
        print("OK")
    """)


def test_zfrac_anneal_makes_measure_loss_step_dependent():
    """anneal=True: at step 0 the random-z probability is ~0, at step >= anneal_ramp it is the
    config max -> the z-mix (and hence measure_loss) depends on the training step."""
    _run("""
        cfg = base_cfg(fb_forward_z=True, actor_cond="z", actor_z_from_critic=True,
                       measure_first_passage=True, measure_z_random_frac=0.5,
                       measure_z_random_frac_anneal=True, anneal_ramp=300000)
        a = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        def loss_at(step):
            ag = a.replace(network=a.network.replace(step=jnp.asarray(step, jnp.int32)))
            return float(ag.measure_loss(batch, ag.network.params, rng)[0])
        assert loss_at(0) != loss_at(400000), "anneal: measure_loss must depend on the step"
        print("OK")
    """)


# --- §9.16j: actor_freeze_steps (delayed actor through the phase-2 objective migration) --
# Mechanism: optax.multi_transform routes actor modules through
# chain(scale_by_schedule(0 while count<N else 1), base_adam) -- gradients are zeroed
# BEFORE adam for the first N optimizer steps (m=v=0 => update exactly 0, and the actor's
# adam state is effectively fresh at unfreeze), then trains normally. Default 0 is
# byte/bit-identical.


def test_actor_freeze_steps_off_bit_identical():
    """Default (and a pre-knob config) => actor_freeze_steps 0, identical param tree and
    bit-identical total_loss on a fixed batch."""
    _run("""
        cfg_old = base_cfg(); cfg_old.pop("actor_freeze_steps", None)
        cfg_new = base_cfg()
        assert cfg_new.get("actor_freeze_steps", 0) == 0
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        lo = a_old.total_loss(batch, a_old.network.params, rng)[0]
        ln = a_new.total_loss(batch, a_new.network.params, rng)[0]
        assert float(lo) == float(ln), (float(lo), float(ln))
        print("OK")
    """)


def test_actor_freeze_steps_freezes_then_unfreezes():
    """actor_freeze_steps=2: optimizer steps 1-2 leave EVERY actor param byte-unchanged
    while critic phi moves; step 3 moves the actor."""
    _run(_fp_helper() + """
        a = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                  base_cfg(actor_freeze_steps=2, measure_coef=1.0))
        batch = {k: jnp.asarray(v) for k, v in make_batch().items()}
        rng = jax.random.PRNGKey(7)
        st = a.network
        snaps = [fp(st.params)]
        for i in range(3):
            g = jax.grad(lambda p: a.total_loss(batch, p, rng)[0])(st.params)
            st = st.apply_gradients(grads=g)
            snaps.append(fp(st.params))
        amod = [k for k in snaps[0] if "actor" in k]
        cmod = [k for k in snaps[0] if "modules_critic" in k]
        assert amod and cmod
        for step in (1, 2):
            assert all((snaps[0][k] == snaps[step][k]).all() for k in amod), \
                f"actor moved during frozen step {step}"
            assert any((snaps[step - 1][k] != snaps[step][k]).any() for k in cmod), \
                f"critic did not train during frozen step {step}"
        assert any((snaps[2][k] != snaps[3][k]).any() for k in amod), \
            "actor did not unfreeze at step 3"
        print("OK")
    """)


def test_actor_freeze_steps_incompatible_with_freeze_actor():
    """Combining the permanent mask and the delayed mask is ambiguous => create() asserts."""
    _run("""
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                  base_cfg(actor_freeze_steps=5, freeze_actor=True))
        except AssertionError as e:
            print("OK")
        else:
            raise SystemExit("create() must refuse actor_freeze_steps + freeze_actor")
    """)
