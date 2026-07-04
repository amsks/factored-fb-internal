"""sf_feature_td: successor-FEATURE vector TD on phi with an identity base leg b(s)=s.

Q(s,a,g)=<phi(s,a), g>/sqrt(d), psi=b=identity (param-free, NOT trained). phi (the
successor-feature vector) is trained ONLY by a per-sample vector TD:

    phi(s,a) <- b(s) + gamma * mask * phi_target(s',a'),   a' ~ pi(s',g),   b(s)=s

This is the SF objective (raymondchua/simple_successor_features), NOT the successor-MEASURE
loss (the B x B density matrix M[i,j]=<phi_i, psi_j>). The decisive test is gamma=0: the
target collapses to b(s)=obs, so L_SF == mean((phi - obs)^2) -- a per-sample regression,
impossible for a measure matrix. Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_sf_identity_b_leg_and_phi_dim():
    """The critic's base/goal leg is identity(g) and phi's output dim == obs/goal dim so
    Q=<phi(s,a), g> is well-formed and finite."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        goals = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
        v, phi, psi = agent.network.select("critic")(ex_obs, goals, ex_act, info=True)
        assert psi.shape == (2, 4, 11), psi.shape
        assert jnp.allclose(psi[0], goals) and jnp.allclose(psi[1], goals), "base leg must be identity(g)"
        assert phi.shape[-1] == 11, phi.shape        # phi dim == obs dim
        assert v.shape == (2, 4) and jnp.all(jnp.isfinite(v)), v
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_loss_is_feature_td_not_measure_gamma0():
    """DECISIVE: with gamma=0 the SF target is b(s)=obs, so L_SF == mean((phi - obs)^2)
    (a per-sample vector regression). A measure loss could never equal this."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True; cfg["discount"] = 0.0
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((6,)), masks=jnp.ones((6,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        _, phi, _ = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True,
                                                   params=agent.network.params)
        expected = jnp.mean((phi - ex_obs[None]) ** 2)     # target = b(s)=obs at gamma=0
        assert jnp.allclose(loss, expected, atol=1e-5), (float(loss), float(expected))
        assert "sf_loss" in info, list(info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_base_leg_has_no_params():
    """The identity base leg carries no learnable params; only phi is learned."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        flat = jax.tree_util.tree_flatten_with_path(agent.network.params["modules_critic"])[0]
        paths = [".".join(str(getattr(k, "key", k)) for k in p) for p, _ in flat]
        assert any("phi" in s for s in paths), paths[:8]
        assert not any("psi" in s for s in paths), [p for p in paths if "psi" in p]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_frozen_base_d_free_and_psi_stays_frozen():
    """Frozen-random base (sf_base='frozen'): d can differ from obs dim (e.g. 50), and the
    psi leg's params are UNCHANGED after training (frozen random encoder -> no collapse)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50   # != obs dim 11
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "frozen"
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        def psi_params(a):
            flat = jax.tree_util.tree_flatten_with_path(a.network.params["modules_critic"])[0]
            return {".".join(str(getattr(k,'key',k)) for k in p): v for p, v in flat if "psi" in
                    ".".join(str(getattr(k,'key',k)) for k in p)}
        before = psi_params(agent)
        assert before, "expected a learnable-shaped psi leg to freeze"
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        after = psi_params(agent)
        for k in before:
            assert jnp.allclose(before[k], after[k]), f"psi param {k} changed -> not frozen"
        assert "sf/sf_loss" in info and jnp.isfinite(info["sf/sf_loss"]), list(info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_frozen_base_feature_is_psi_of_state_gamma0():
    """Frozen base semantic: with gamma=0 the SF target is b(s)=psi(s) (the frozen features
    of the current state), so L_SF == mean((phi - psi(obs))^2) at d=50."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "frozen"; cfg["discount"] = 0.0
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((6,)), masks=jnp.ones((6,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        _, phi, _ = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True,
                                                   params=agent.network.params)
        _, _, psi_s = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True,
                                                    params=agent.network.params)   # b(s)=psi(obs)
        expected = jnp.mean((phi - psi_s) ** 2)
        assert jnp.allclose(loss, expected, atol=1e-5), (float(loss), float(expected))
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_trains_finite():
    """Full agent with sf_feature_td trains finite; total_loss reports sf/sf_loss (the SF
    vector TD) and NOT measure/measure_loss, and the flow actor loss stays finite."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert "sf/sf_loss" in info, list(info)
        assert not any(k.startswith("measure/") for k in info), list(info)
        assert jnp.isfinite(info["sf/sf_loss"]), info["sf/sf_loss"]
        assert jnp.isfinite(info["actor/actor_loss"]), info["actor/actor_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
