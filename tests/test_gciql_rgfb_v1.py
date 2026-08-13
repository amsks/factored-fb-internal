"""RG-FB v1 (spec 06 item 4): task-prior scalar grounding on gciql_flow.py.

Knobs task_prior in {'dirac' (default), 'mix'}, task_prior_probs, task_prior_mix_k.
Discipline (mirrors tests/test_gciql_rgfb.py): task_prior='dirac' is BIT-IDENTICAL to
v0 (param tree + update losses); the 'mix' arm builds the sampled tasks correctly
(z renorm, ball reward, stop-grad); the per-task-type normalization keeps the dense
linear arm from dominating the sparse arms; the assert-lattice refuses mis-compositions.

obs_dim = 29 (the mixture/Dirac ball reward reads cube xyz at obs slice [19:22]).
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

    def v0_cfg(**kw):
        # the RG-FB v0 recipe: bilinear-d, z-conditioned actor + vf, flow actor.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["actor_cond"] = "z"; cfg["actor_vf_cond"] = "z"; cfg["bc_coeff"] = 3.0
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


def test_dirac_default_bit_identical_to_v0():
    """(a) task_prior='dirac' (and a config that PREDATES the knob) => identical param
    tree and bit-identical update() losses vs v0."""
    _run("""
        cfg_old = v0_cfg()
        for k in ("task_prior", "task_prior_probs", "task_prior_mix_k"):
            cfg_old.pop(k, None)
        cfg_new = v0_cfg()
        assert cfg_new["task_prior"] == "dirac"
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)
        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"
        b = make_batch()
        for _ in range(3):
            a_old, io = a_old.update(b)
            a_new, inw = a_new.update(b)
            for k in io:
                assert float(io[k]) == float(inw[k]), (k, float(io[k]), float(inw[k]))
        print("OK")
    """)


def test_mix_sample_tasks_renorm_reward_and_grad_discipline():
    """(b, CORRECTED per spec06 review finding 1) mixture/linear z renormed to the
    match-data norm; ball reward is gc_negative {-1,0} (finding 2); gradient discipline:
    z CARRIES psi-gradients (arms 0/1 train psi through the vc TD), while the actor's
    z_actor and every reward are stop-gradded; z_actor Dirac rows come from actor_goals
    (finding 4)."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, v0_cfg(
            task_prior="mix", task_prior_probs=[0.34, 0.33, 0.33]))
        b = {k: jnp.asarray(v) for k, v in make_batch(B=256).items()}
        key = jax.random.PRNGKey(0)
        z, z_actor, r, mask, ttype = ag._sample_tasks(b, ag.network.params, key)
        z, z_actor, r, ttype = map(np.asarray, (z, z_actor, r, ttype))
        assert z.shape == (256, 8) and np.all(np.isfinite(z))

        psi_next = np.asarray(ag._actor_z(b["next_observations"]))
        mean_norm = np.linalg.norm(psi_next, axis=-1).mean()
        # mixture (type 1) and linear (type 2) z are renormed to the match-data norm.
        for t in (1, 2):
            zn = np.linalg.norm(z[ttype == t], axis=-1)
            assert np.allclose(zn, mean_norm, rtol=1e-3), (t, zn[:3], mean_norm)
        # mixture reward is gc_negative {-1,0} (matches the Dirac arm's convention).
        assert set(np.unique(r[ttype == 1]).tolist()) <= {-1.0, 0.0}, np.unique(r[ttype == 1])
        assert np.all(np.isfinite(r))
        # mask semantics: mixture mask = 1[no hit] = -reward (gc_negative); linear mask = 1.
        assert np.allclose(np.asarray(mask)[ttype == 1], -r[ttype == 1])
        assert np.allclose(np.asarray(mask)[ttype == 2], 1.0)

        # z_actor: Dirac rows = psi(actor_goals) (the D8 split); others = the sampled z.
        psi_ag = np.asarray(ag._actor_z(b["actor_goals"]))
        assert np.allclose(z_actor[ttype == 0], psi_ag[ttype == 0], atol=1e-5)
        assert np.allclose(z_actor[ttype != 0], z[ttype != 0], atol=1e-5)

        # GRADIENT DISCIPLINE. (i) z DOES carry gradient into psi (the value net's goal
        # tower -- _actor_z default): fully-sg z was the frozen-random-psi 0.061 bug.
        gz = jax.grad(lambda p: ag._sample_tasks(b, p, key)[0].sum())(ag.network.params)
        mxv = max(float(jnp.abs(l).max()) for l in jax.tree_util.tree_leaves(gz["modules_value"]))
        assert mxv > 0.0, "z carries NO gradient into psi -- the frozen-psi bug is back"
        # (ii) z_actor is fully stop-gradded (the actor never trains psi).
        ga = jax.grad(lambda p: ag._sample_tasks(b, p, key)[1].sum())(ag.network.params)
        # (iii) rewards are stop-gradded (the linear arm's reward is not gameable).
        gr = jax.grad(lambda p: ag._sample_tasks(b, p, key)[2].sum())(ag.network.params)
        for g, name in ((ga, "z_actor"), (gr, "reward")):
            for mod in ("modules_value", "modules_critic"):
                mx = max(float(jnp.abs(l).max()) for l in jax.tree_util.tree_leaves(g[mod]))
                assert mx == 0.0, f"{name} leaks gradient into {mod} (max |g|={mx})"
        print("OK")
    """)


def test_mix_pertype_normalization_and_update():
    """(c) the per-task-type normalization pulls the dense linear arm and the sparse
    Dirac arm to the same order (raw scales differ; normalized losses within ~2x);
    update() runs finite end-to-end."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, v0_cfg(
            task_prior="mix", task_prior_probs=[0.5, 0.0, 0.5]))
        b = {k: jnp.asarray(v) for k, v in make_batch(B=256).items()}
        key = jax.random.PRNGKey(1)
        z, z_actor, r, mask, ttype = ag._sample_tasks(b, ag.network.params, key)
        _, info = ag._taskprior_vc_loss(b, ag.network.params, z, r, mask, ttype)
        s_dirac = float(info["taskprior_scale_dirac"]); s_lin = float(info["taskprior_scale_lin"])
        n_dirac = float(info["taskprior_norm_ct_dirac"]); n_lin = float(info["taskprior_norm_ct_lin"])
        # raw target scales genuinely differ (else normalization would be a no-op).
        assert max(s_dirac, s_lin) / (min(s_dirac, s_lin) + 1e-9) > 1.3, (s_dirac, s_lin)
        # after normalization the two arms' critic losses are the same order (<= 3x).
        assert 0.0 < n_dirac and 0.0 < n_lin
        ratio = max(n_dirac, n_lin) / (min(n_dirac, n_lin) + 1e-9)
        assert ratio <= 3.0, (n_dirac, n_lin, ratio)

        ag2 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, v0_cfg(task_prior="mix"))
        bb = make_batch(B=64)
        for _ in range(3):
            ag2, i2 = ag2.update(bb)
        for k, v in i2.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        # the actor conditioned on the sampled z produces finite actions.
        a = ag2.sample_actions_with_z(observations=bb["observations"],
                                      z=np.zeros((64, 8), np.float32),
                                      seed=jax.random.PRNGKey(0), temperature=0.0)
        assert np.all(np.isfinite(np.asarray(a)))
        print("OK")
    """)


def test_assert_lattice():
    """(d) mis-compositions refuse at construction."""
    _run("""
        def refuses(msg_frag, **kw):
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, v0_cfg(**kw))
            except AssertionError as e:
                assert msg_frag in str(e), (msg_frag, str(e))
                return
            raise SystemExit(f"expected AssertionError ({msg_frag}) for {kw}")

        refuses("requires value_arch='bilinear'", task_prior="mix", value_arch="mlp",
                actor_cond="goal", actor_vf_cond="none")
        refuses("actor_cond='z' and actor_vf_cond='z'", task_prior="mix", actor_vf_cond="none")
        refuses("incompatible with measure_only", task_prior="mix", measure_only=True,
                measure_legs="critic")
        refuses("task_prior_probs must be 3", task_prior="mix", task_prior_probs=[0.5, 0.5])
        refuses("task_prior_probs must be 3", task_prior="mix", task_prior_probs=[0.5, 0.2, 0.2])
        print("OK")
    """)
