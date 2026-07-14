"""psi_base: the readout-vs-TD crux (sf-design-space.md item 1b).

Bilinear GCIQL trained with its NORMAL scalar IQL TD (value/critic expectile backup, AWR-free
flow actor -- everything identical to the 0.90 `GCIQL-State-Flow-Bilinear-d50` control), but
with the goal leg psi(g) crippled the same way the SF cells cripple it:

    psi_base='learned'  (default) -> GCBilinearValue, psi trained          [the 0.90 control]
    psi_base='frozen'            -> GCFrozenBValue,  psi = frozen random MLP (d free)
    psi_base='identity'          -> GCIdentityBValue, psi(g) = g           (d == goal dim)

This disentangles the two things the SF cells changed at once. If scalar-TD + frozen psi still
reaches ~0.9 while SF + frozen psi sits at ~0.1 on the SAME crippled leg, the SF failure is
pinned on the VECTOR TD itself. If scalar-TD + frozen psi also collapses, the frozen readout was
part of SF's handicap and the SF cells were never a clean test of the vector TD.

Previously these classes were reachable ONLY under sf_feature_td; `psi_base` lifts the dispatch
out of that branch and applies it SYMMETRICALLY to the value net and the critic. Runs in the jax
venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_psi_base_default_is_learned_and_trains_the_goal_leg():
    """CONTROL: the default (psi_base='learned') must be the untouched bilinear GCIQL --
    GCBilinearValue with a TRAINED psi. Guarantees the new knob is opt-in and that the
    'frozen' assertions below are measuring a real difference, not a no-op."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        from utils.networks import GCBilinearValue
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        assert cfg["psi_base"] == "learned", cfg["psi_base"]      # opt-in default
        assert not cfg["sf_feature_td"]
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        before = jax.tree_util.tree_leaves(agent.network.params["modules_critic"]["psi"])
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(3):
            agent, info = agent.update(batch)
        after = jax.tree_util.tree_leaves(agent.network.params["modules_critic"]["psi"])
        moved = any(not jnp.allclose(a, b) for a, b in zip(before, after))
        assert moved, "default psi_base='learned' must TRAIN the goal leg"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_psi_base_frozen_freezes_goal_leg_on_both_value_and_critic():
    """DECISIVE for the frozen cell: under the NORMAL scalar IQL TD (sf_feature_td=False), a
    psi_base='frozen' run must leave the psi params of BOTH the value net and the critic exactly
    at their random init after many updates, while phi keeps moving. Symmetry matters -- freezing
    only the critic would leave V(s,g) a fully-trained bilinear head and confound the readout."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["psi_base"] = "frozen"
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        snap = lambda p, m: jax.tree_util.tree_leaves(p["modules_" + m]["psi"])
        snap_phi = lambda p, m: jax.tree_util.tree_leaves(p["modules_" + m]["phi"])
        p0 = agent.network.params
        psi_v0, psi_c0 = snap(p0, "value"), snap(p0, "critic")
        phi_c0 = snap_phi(p0, "critic")
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        p1 = agent.network.params
        for a, b in zip(psi_v0, snap(p1, "value")):
            assert jnp.allclose(a, b), "value net's psi must stay FROZEN at init"
        for a, b in zip(psi_c0, snap(p1, "critic")):
            assert jnp.allclose(a, b), "critic's psi must stay FROZEN at init"
        assert any(not jnp.allclose(a, b) for a, b in zip(phi_c0, snap_phi(p1, "critic"))), \\
            "phi must still train under the scalar IQL TD"
        assert all(jnp.isfinite(v) for v in info.values() if jnp.ndim(v) == 0), info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_psi_base_identity_readout_is_the_raw_goal_on_both_nets():
    """DECISIVE for the identity cell: Q(s,a,g)=<phi(s,a), g>/sqrt(d) and V(s,g)=<phi(s), g>/sqrt(d)
    -- the goal leg IS the raw goal obs, param-free (no 'psi' params exist at all), on BOTH nets.
    This is the scalar-TD analogue of torch's idB_decond_dnorm (which collapsed to 2.7)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11   # == goal dim
        cfg["psi_base"] = "identity"
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        goals = jax.random.normal(jax.random.PRNGKey(3), (6, 11))
        p = agent.network.params
        assert "psi" not in p["modules_critic"], "identity goal leg must be param-free"
        assert "psi" not in p["modules_value"], "identity goal leg must be param-free (value net)"
        qv, phi, psi = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        assert jnp.allclose(psi[0], goals) and jnp.allclose(psi[1], goals), "critic psi(g) must be g"
        assert jnp.allclose(qv, (phi * psi).sum(-1) / jnp.sqrt(11.0), atol=1e-5)
        vv, vphi, vpsi = agent.network.select("value")(ex_obs, goals, info=True, params=p)
        assert jnp.allclose(vpsi, goals), "value psi(g) must be g"
        assert jnp.allclose(vv, (vphi * vpsi).sum(-1) / jnp.sqrt(11.0), atol=1e-5)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((6,)), masks=jnp.ones((6,)),
                     value_goals=goals, actor_goals=goals)
        for _ in range(3):
            agent, info = agent.update(batch)
        assert all(jnp.isfinite(v) for v in info.values() if jnp.ndim(v) == 0), info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_psi_base_guards_reject_invalid_combinations():
    """The knob is only meaningful for the scalar-TD bilinear critic. Refuse: (a) value_arch='mlp'
    (no goal leg to cripple); (b) sf_feature_td (sf_base already owns the base class -- silently
    letting both through would make the cell ambiguous); (c) identity with d != goal dim (the
    bilinear form would be malformed); (d) fb_forward_z / fwd_two_tower, which pick their own
    critic class and would silently override psi_base."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (4, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
        def build(**over):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg.update(over)
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        cases = [
            dict(psi_base="frozen", value_arch="mlp"),           # (a)
            dict(psi_base="frozen", sf_feature_td=True),         # (b)
            dict(psi_base="identity", value_latent_dim=50),      # (c) d != goal dim 11
            dict(psi_base="frozen", fb_forward_z=True),          # (d)
            dict(psi_base="frozen", fwd_two_tower=True),         # (d)
        ]
        for over in cases:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("psi_base guard missing for " + repr(over))
        build(psi_base="identity", value_latent_dim=11)          # the valid identity cell
        build(psi_base="frozen")                                 # the valid frozen cell
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
