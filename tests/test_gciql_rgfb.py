"""RG-FB v0 (PAPER/rescuing-fb/02-rgfb-spec.md §2.1-2.2): actor_vf_cond knob.

The BC flow vector field (and the distillation rollout) can be conditioned on the
goal ('goal': raw goal concatenated, awr_flow parity) or on the task vector
('z': sg(psi(goal)) -- the SAME z the one-shot z-actor consumes). Default 'none'
keeps the stock goal-agnostic vf bit-identically (param tree AND losses), so every
existing checkpoint keeps loading.

Runs in the jax venv subprocess (mirrors tests/test_gciql_psi_sg.py).
"""
import textwrap

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

COMMON = """
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config

    EX_OBS = jnp.zeros((1, 11), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def base_cfg(**kw):
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg.update(kw)
        return cfg

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        return dict(
            observations=rng.normal(size=(B, 11)).astype(np.float32),
            next_observations=rng.normal(size=(B, 11)).astype(np.float32),
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, 11)).astype(np.float32),
            actor_goals=rng.normal(size=(B, 11)).astype(np.float32),
            rewards=rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
            masks=rng.integers(0, 2, size=B).astype(np.float32),
        )

    def flat_sd(agent):
        sd = flax.serialization.to_state_dict(agent.network.params)
        leaves = jax.tree_util.tree_flatten_with_path(sd)[0]
        return [(jax.tree_util.keystr(p), np.asarray(l)) for p, l in leaves]
"""


def _run(code):
    # Dedent the two blocks SEPARATELY: COMMON is indented 4, tests 8 -- a joint
    # dedent would leave the test body indented (dead code inside flat_sd).
    code = textwrap.dedent(COMMON) + textwrap.dedent(code)
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_default_none_is_bit_identical_and_checkpoint_compatible():
    """(a) actor_vf_cond='none' (and a config that PREDATES the knob entirely) must
    produce the identical param tree and bitwise-identical update() losses on a
    fixed batch -- the old-checkpoint-loadability + bit-identity guarantee."""
    _run("""
        cfg_old = base_cfg(value_arch="bilinear", value_latent_dim=8)
        cfg_old.pop("actor_vf_cond")          # a pre-RGFB config dict (old checkpoints)
        cfg_new = base_cfg(value_arch="bilinear", value_latent_dim=8)
        assert cfg_new["actor_vf_cond"] == "none", cfg_new["actor_vf_cond"]
        a_old = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_old)
        a_new = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_new)

        fo, fn = flat_sd(a_old), flat_sd(a_new)
        assert [k for k, _ in fo] == [k for k, _ in fn], "param tree structure changed!"
        for (k, lo), (_, ln) in zip(fo, fn):
            assert lo.shape == ln.shape and (lo == ln).all(), f"param {k} differs"

        # _vf_obs must be a passthrough (the SAME array object) under 'none'.
        b = make_batch()
        o = jnp.asarray(b["observations"])
        assert a_new._vf_obs(o, jnp.asarray(b["actor_goals"])) is o, \
            "_vf_obs must return obs unchanged under actor_vf_cond='none'"

        # Bitwise-identical training trajectories on a fixed batch.
        for i in range(3):
            a_old, io = a_old.update(b)
            a_new, inw = a_new.update(b)
            for k in io:
                assert float(io[k]) == float(inw[k]), (i, k, float(io[k]), float(inw[k]))

        # Under 'none' the vf is goal-agnostic: bc_flow_loss must NOT change when
        # actor_goals change (rng held fixed).
        params = a_new.network.params
        key = jax.random.PRNGKey(3)
        b2 = dict(b); b2["actor_goals"] = make_batch(seed=99)["actor_goals"]
        _, i1 = a_new.actor_loss(b, params, rng=key)
        _, i2 = a_new.actor_loss(b2, params, rng=key)
        assert float(i1["bc_flow_loss"]) == float(i2["bc_flow_loss"]), \
            "stock vf must be goal-agnostic"
        print("OK")
    """)


def test_z_cond_shapes_stopgrad_and_training():
    """(b) actor_vf_cond='z': vf input widened by d; the condition is sg(psi(goal))
    so NO gradient reaches the value net (or the critic) through actor_loss; the
    vf becomes goal-dependent; update() trains end-to-end and eval sampling works."""
    _run("""
        d = 8
        cfg = base_cfg(value_arch="bilinear", value_latent_dim=d,
                       actor_cond="z", actor_vf_cond="z")
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)

        # vf first-layer width: obs(11) + z(d) + action(5) + t(1).
        kerns = [l for k, l in flat_sd(ag) if "modules_actor_vf" in k and k.endswith("kernel']")]
        in_dims = sorted(int(k.shape[0]) for k in kerns)
        assert 11 + d + 5 + 1 in in_dims, f"no vf kernel with input dim {11+d+5+1}: {in_dims}"

        b = make_batch()
        params = ag.network.params
        key = jax.random.PRNGKey(3)

        # stop-gradient: actor_loss must send ZERO gradient into the value net
        # (psi) and the critic (frozen-critic rule) -- the vf condition is sg'd.
        g = jax.grad(lambda p: ag.actor_loss(b, p, rng=key)[0])(params)
        for mod in ("modules_value", "modules_critic"):
            leaves = jax.tree_util.tree_leaves(g[mod])
            mx = max(float(jnp.abs(l).max()) for l in leaves)
            assert mx == 0.0, f"actor_loss leaks gradient into {mod} (max |g|={mx})"

        # the vf must now be goal-DEPENDENT (through z=psi(goal)).
        b2 = dict(b); b2["actor_goals"] = make_batch(seed=99)["actor_goals"]
        _, i1 = ag.actor_loss(b, params, rng=key)
        _, i2 = ag.actor_loss(b2, params, rng=key)
        assert float(i1["bc_flow_loss"]) != float(i2["bc_flow_loss"]), \
            "z-conditioned vf must depend on the goal"

        for _ in range(3):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        a = ag.sample_actions(observations=b["observations"], goals=b["actor_goals"],
                              seed=jax.random.PRNGKey(0), temperature=0.0)
        assert np.all(np.isfinite(np.asarray(a))), "actions not finite"
        print("OK")
    """)


def test_goal_cond_works_on_mlp_arch():
    """(b') actor_vf_cond='goal' needs no psi -- it must work on the plain MLP
    value_arch too (the spec's 'unless goal' clause), widening the vf by obs_dim."""
    _run("""
        cfg = base_cfg(value_arch="mlp", actor_vf_cond="goal")
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
        kerns = [l for k, l in flat_sd(ag) if "modules_actor_vf" in k and k.endswith("kernel']")]
        in_dims = sorted(int(k.shape[0]) for k in kerns)
        assert 11 + 11 + 5 + 1 in in_dims, f"no vf kernel with input dim 28: {in_dims}"

        b = make_batch()
        key = jax.random.PRNGKey(3)
        b2 = dict(b); b2["actor_goals"] = make_batch(seed=99)["actor_goals"]
        _, i1 = ag.actor_loss(b, ag.network.params, rng=key)
        _, i2 = ag.actor_loss(b2, ag.network.params, rng=key)
        assert float(i1["bc_flow_loss"]) != float(i2["bc_flow_loss"]), \
            "goal-conditioned vf must depend on the goal"
        for _ in range(3):
            ag, info = ag.update(b)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_assert_lattice():
    """(c) invalid combinations must refuse at construction."""
    _run("""
        def refuses(msg_frag, **kw):
            cfg = base_cfg(**kw)
            try:
                GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg)
            except AssertionError as e:
                assert msg_frag in str(e), (msg_frag, str(e))
                return
            raise SystemExit(f"expected AssertionError ({msg_frag}) for {kw}")

        # 'z' without actor_cond='z'
        refuses("requires actor_cond='z'", value_arch="bilinear", value_latent_dim=8,
                actor_vf_cond="z")
        # 'z' with actor_z_infer
        refuses("actor_z_infer", value_arch="bilinear", value_latent_dim=8,
                actor_cond="z", actor_vf_cond="z", actor_z_infer=True)
        # conditioning without a flow actor
        refuses("actor_kind='flow'", value_arch="bilinear", value_latent_dim=8,
                actor_kind="ddpgbc", actor_vf_cond="goal")
        # unknown mode
        refuses("unknown actor_vf_cond", value_arch="bilinear", value_latent_dim=8,
                actor_vf_cond="bogus")
        print("OK")
    """)
