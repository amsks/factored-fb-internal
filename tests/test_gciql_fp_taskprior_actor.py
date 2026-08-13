"""Spec 07 items 1+4 (task_prior_actor knob family) on gciql_flow.py's FP corner:
  task_prior_actor       (item 1) -- per-sample ACTOR-ONLY z-prior (Dirac/mixture/random),
                                     fully sg; the FP matrix critic loss is untouched.
  task_prior_critic_mix  (item 4) -- the linearity-conflict probe: scalar FP TD at the
                                     mixture z ON TOP of the untouched matrix loss.

Discipline (mirrors tests/test_gciql_measure_axes.py): default OFF => byte-identical
param tree AND bitwise-identical update() vs the FP-NoOrtho cell on a fixed batch;
knob ON leaves the measure loss exactly unchanged (actor-only knob); the actor z is
fully stop-gradded in every arm; the mixture renorm hits the match-data norm; arm
frequencies match probs; the assert-lattice refuses mis-compositions.

obs_dim = 29: the mixture goals / critic-mix hit read cube xyz at obs slice [19:22].
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

    def fp_cfg(**kw):
        # the FP-NoOrtho winning cell (cube_single_state_measure_firstpassage_noortho_d50):
        # bilinear critic, measure-ONLY, first-passage residual, scale-norm, NO ortho.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["measure_legs"] = "critic"
        cfg["measure_coef"] = 1.0; cfg["measure_ortho_coef"] = 0.0
        cfg["measure_scale_norm"] = True; cfg["measure_first_passage"] = True
        cfg["bc_coeff"] = 3.0
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


def test_knob_off_bitwise_identical_to_fp_cell():
    """(a) task_prior_actor='dirac' default (and a config that PREDATES the knob family)
    => identical param tree and bitwise-identical update() -- same params + same rng =>
    identical NEW params -- vs the FP-NoOrtho cell on a fixed batch."""
    _run("""
        cfg_old = fp_cfg()
        for k in ("task_prior_actor", "task_prior_actor_probs",
                  "task_prior_actor_mix_k", "task_prior_critic_mix"):
            cfg_old.pop(k, None)                       # a pre-spec07 config dict
        cfg_new = fp_cfg()
        assert cfg_new["task_prior_actor"] == "dirac"
        assert cfg_new["task_prior_critic_mix"] is False
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        b = make_batch()
        for _ in range(3):
            a_old, io = a_old.update(b)
            a_new, inw = a_new.update(b)
            for k in io:
                assert float(io[k]) == float(inw[k]), (k, float(io[k]), float(inw[k]))
        # bitwise-identical NEW params after the updates (the strongest knob-off gate).
        for (k, lo), (_, ln) in zip(flat_sd(a_old), flat_sd(a_new)):
            assert (lo == ln).all(), f"post-update param {k} diverged"
        print("OK")
    """)


def test_mix_is_actor_only_critic_loss_unchanged():
    """(b) knob-on: the measure (critic) loss value on a fixed batch is EXACTLY unchanged
    vs the knob-off z-actor FP cell (task_prior_actor forces actor_cond='z' +
    actor_z_from_critic, so THAT cell is the same-param-tree comparator); update() runs
    finite end-to-end; eval sampling works."""
    _run("""
        a_on = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp_cfg(task_prior_actor="mix"))
        a_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp_cfg(
            actor_cond="z", actor_z_from_critic=True))
        # the forced conditioning reproduces the D1 z-actor tree exactly (same seed).
        fo, fn = flat_sd(a_off), flat_sd(a_on)
        assert [k for k, _ in fo] == [k for k, _ in fn], "knob-on param tree differs from the z-actor FP cell"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert (lo == ln).all(), f"param {k} differs"
        assert a_on.config["actor_cond"] == "z" and a_on.config["actor_z_from_critic"]
        # actor-only knob: measure_loss is bitwise identical (same params, same rng).
        b = make_batch(); key = jax.random.PRNGKey(3)
        l_on, _ = a_on.measure_loss(b, a_on.network.params, key)
        l_off, _ = a_off.measure_loss(b, a_off.network.params, key)
        assert float(l_on) == float(l_off), (float(l_on), float(l_off))
        # end-to-end: update() finite; the info stream carries the arm fractions.
        ag = a_on
        for _ in range(2):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        assert "tpactor/frac_mix" in info
        fr = info["tpactor/frac_dirac"] + info["tpactor/frac_mix"] + info["tpactor/frac_rand"]
        assert abs(float(fr) - 1.0) < 1e-6
        # eval path: sample_actions maps the raw goal through the critic psi (z-actor).
        a = ag.sample_actions(observations=b["observations"], goals=b["actor_goals"],
                              seed=jax.random.PRNGKey(0))
        assert np.asarray(a).shape == b["actions"].shape and np.all(np.isfinite(np.asarray(a)))
        print("OK")
    """)


def test_actor_z_fully_stop_gradded_all_arms():
    """(c) the actor's z is FULLY sg in every arm: jax.grad of the sampled z AND of the
    whole actor loss w.r.t. the psi/B towers (critic + value modules) is zero, for each
    arm forced via degenerate probs (pattern: test_gciql_rgfb_v1)."""
    _run("""
        b = {k: jnp.asarray(v) for k, v in make_batch(B=64).items()}
        key = jax.random.PRNGKey(0); akey = jax.random.PRNGKey(1)
        for probs in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
            ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp_cfg(
                task_prior_actor="mix", task_prior_actor_probs=probs))
            # (i) the sampled z (and the full-batch z_mix) carry NO psi-gradient.
            for out_ix in (0, 2):
                g = jax.grad(lambda p: ag._sample_actor_tasks(b, p, key)[out_ix].sum())(
                    ag.network.params)
                for mod in ("modules_critic", "modules_value"):
                    mx = max(float(jnp.abs(l).max()) for l in jax.tree_util.tree_leaves(g[mod]))
                    assert mx == 0.0, f"probs={probs}: z output {out_ix} leaks grad into {mod} ({mx})"
            # (ii) the whole actor loss (conditioned on that z) sends NO gradient into the
            # psi/B towers -- psi trains ONLY through the FP matrix loss.
            def aloss(p):
                z = ag._sample_actor_tasks(b, p, key)[0]
                return ag.actor_loss(b, p, akey, task_z=z)[0]
            ga = jax.grad(aloss)(ag.network.params)
            for mod in ("modules_critic", "modules_value"):
                mx = max(float(jnp.abs(l).max()) for l in jax.tree_util.tree_leaves(ga[mod]))
                assert mx == 0.0, f"probs={probs}: actor loss leaks grad into {mod} ({mx})"
            # sanity: the actor itself DOES train.
            mxa = max(float(jnp.abs(l).max())
                      for l in jax.tree_util.tree_leaves(ga["modules_actor"]))
            assert mxa > 0.0, "actor loss sends no gradient into the actor"
        print("OK")
    """)


def test_mixture_renorm_and_arm_frequencies():
    """(d) mixture (and random) z renormed to the match-data norm mean||psi(s')||;
    arm frequencies over a large draw match task_prior_actor_probs."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp_cfg(task_prior_actor="mix"))
        b = {k: jnp.asarray(v) for k, v in make_batch(B=4096, seed=2).items()}
        key = jax.random.PRNGKey(7)
        z, ttype, z_mix, cube_K = ag._sample_actor_tasks(b, ag.network.params, key)
        z, ttype, z_mix, cube_K = map(np.asarray, (z, ttype, z_mix, cube_K))
        assert z.shape == (4096, 8) and np.all(np.isfinite(z))
        assert cube_K.shape == (4096, 2, 3)

        # psi here = the measure-trained critic B (actor_z_from_critic forced).
        psi_next = np.asarray(ag._actor_z(b["next_observations"]))
        mean_norm = np.linalg.norm(psi_next, axis=-1).mean()
        # mixture (1) and random (2) rows sit exactly at the match-data norm.
        for t in (1, 2):
            zn = np.linalg.norm(z[ttype == t], axis=-1)
            assert np.allclose(zn, mean_norm, rtol=1e-3), (t, zn[:3], mean_norm)
        # the FULL-batch mixture arm (item 4's input) is renormed on every row.
        assert np.allclose(np.linalg.norm(z_mix, axis=-1), mean_norm, rtol=1e-3)
        # Dirac rows = sg[psi(actor_goals)] -- the existing actor-z path (D8 split).
        psi_ag = np.asarray(ag._actor_z(b["actor_goals"]))
        assert np.allclose(z[ttype == 0], psi_ag[ttype == 0], atol=1e-5)

        # arm frequencies ~ probs = [0.6, 0.2, 0.2] (defaults; NOT overridden in yaml).
        for t, p in enumerate([0.6, 0.2, 0.2]):
            f = float((ttype == t).mean())
            assert abs(f - p) < 0.03, (t, f, p)
        print("OK")
    """)


def test_critic_mix_probe_grounds_phi_only_and_matrix_untouched():
    """Item 4 (task_prior_critic_mix): the scalar FP TD at z_mix trains the critic's phi
    (grad reaches modules_critic) while the matrix measure loss stays exactly unchanged;
    the reward is [0,1] (not gc_negative); update() runs finite."""
    _run("""
        a_cm = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp_cfg(
            task_prior_actor="mix", task_prior_critic_mix=True))
        a_on = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp_cfg(task_prior_actor="mix"))
        b = {k: jnp.asarray(v) for k, v in make_batch(B=64).items()}
        key = jax.random.PRNGKey(5)
        # the matrix loss is untouched by the added grounding (same params, same rng).
        lm_cm, _ = a_cm.measure_loss(b, a_cm.network.params, key)
        lm_on, _ = a_on.measure_loss(b, a_on.network.params, key)
        assert float(lm_cm) == float(lm_on), (float(lm_cm), float(lm_on))

        z, ttype, z_mix, cube_K = a_cm._sample_actor_tasks(b, a_cm.network.params, key)
        l, info = a_cm._tpactor_critic_mix_loss(b, a_cm.network.params, z_mix, cube_K,
                                                jax.random.PRNGKey(9))
        assert np.isfinite(float(l))
        # [0,1] reward convention (matching the FP cell, NOT gc_negative).
        assert 0.0 <= float(info["criticmix_r_mean"]) <= 1.0
        # grads reach the critic (phi) but NOT the value net; z_mix is sg (no psi path).
        g = jax.grad(lambda p: a_cm._tpactor_critic_mix_loss(
            b, p, z_mix, cube_K, jax.random.PRNGKey(9))[0])(a_cm.network.params)
        mxc = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g["modules_critic"]))
        mxv = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g["modules_value"]))
        assert mxc > 0.0, "critic-mix grounding sends no gradient into the critic phi"
        assert mxv == 0.0, f"critic-mix grounding leaks into the value net ({mxv})"

        ag = a_cm
        for _ in range(2):
            ag, i2 = ag.update(b)
        for k, v in i2.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        assert "tpactor/criticmix_loss" in i2
        print("OK")
    """)


def test_assert_lattice():
    """(e) mis-compositions refuse at construction."""
    _run("""
        def refuses(msg_frag, **kw):
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, fp_cfg(**kw))
            except AssertionError as e:
                assert msg_frag in str(e), (msg_frag, str(e))
                return
            raise SystemExit(f"expected AssertionError ({msg_frag}) for {kw}")

        refuses("unknown task_prior_actor", task_prior_actor="bogus")
        # rides the FP matrix critic: needs measure_only + first-passage + bilinear.
        refuses("requires measure_only=True", task_prior_actor="mix", measure_only=False)
        refuses("requires measure_first_passage=True", task_prior_actor="mix",
                measure_first_passage=False)
        refuses("requires value_arch='bilinear'", task_prior_actor="mix", value_arch="mlp")
        # disjoint namespaces with the scalar-corner v1 knob.
        refuses("mutually exclusive with task_prior='mix'", task_prior_actor="mix",
                task_prior="mix")
        # the actor's z has one owner.
        refuses("actor_z_infer would conflict", task_prior_actor="mix", actor_z_infer=True)
        refuses("task_prior_actor_probs must be 3", task_prior_actor="mix",
                task_prior_actor_probs=[0.5, 0.5])
        refuses("task_prior_actor_probs must be 3", task_prior_actor="mix",
                task_prior_actor_probs=[0.5, 0.2, 0.2])
        refuses("task_prior_actor_mix_k must be >= 1", task_prior_actor="mix",
                task_prior_actor_mix_k=0)
        # item 4 is guarded on item 1.
        refuses("requires task_prior_actor='mix'", task_prior_critic_mix=True)
        print("OK")
    """)
