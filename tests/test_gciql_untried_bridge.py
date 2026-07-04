"""Untried bridge cells (B1, C2) + the inferred-z actor control (exp 4).

These pin three cells the fb-gciql-bridge audit found empty:
  * C2  -- measure objective on the FB two-tower forward map (measure_only + fwd_two_tower)
  * B1  -- a genuine z-conditioned forward map F(s,a,z) trained under the IQL backup
           (fb_forward_z WITHOUT measure_only; the `assert measure_only` must be relaxed
           and z_goals threaded through the IQL value/critic losses)
  * exp4 -- the flow actor trained across the *inferred* task distribution z=E[r*psi],
           not only the exact goal embedding z=psi(g) (the G4 "next control").

Runs in the jax venv subprocess (mirrors tests/test_gciql_two_tower_critic.py).
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_measure_two_tower_trains_finite():
    """C2: measure_only=True + fwd_two_tower=True routes to GCTwoTowerCritic; the
    successor-measure loss must run finite on the two-tower forward legs."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["fwd_two_tower"] = True
        cfg["measure_only"] = True; cfg["measure_coef"] = 1.0; cfg["measure_ortho_coef"] = 1.0
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        flat = jax.tree_util.tree_flatten_with_path(agent.network.params['modules_critic'])[0]
        paths = [".".join(str(getattr(k, 'key', k)) for k in p) for p, _ in flat]
        assert any("embed_sz" in s for s in paths), paths[:8]
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((4,)), masks=jnp.ones((4,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert "measure/measure_loss" in info, list(info)
        assert jnp.isfinite(info["measure/measure_loss"]), info["measure/measure_loss"]
        assert jnp.isfinite(info["actor/actor_loss"]), info["actor/actor_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_fb_forward_z_under_iql_trains_finite():
    """B1: fb_forward_z=True with measure_only=False must build (assert relaxed) and
    the IQL value/critic/actor losses must run finite -- Q(s,a,g)=<F(s,a,psi(g)),psi(g)>."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["fb_forward_z"] = True
        cfg["measure_only"] = False; cfg["measure_coef"] = 0.0   # pure IQL backup
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)   # must NOT assert
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((4,)), masks=jnp.ones((4,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        for k in ("value/value_loss", "critic/critic_loss", "actor/actor_loss"):
            assert k in info, list(info)
            assert jnp.isfinite(info[k]), (k, info[k])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_infer_actor_z_is_sphere_normed_and_differs_from_exact():
    """exp4 helper: _infer_actor_z maps batch goal embeddings psi [B,d] to an inferred
    task vector z=E[r*psi] per sample -- shape [B,d], sphere-normed to sqrt(d), and NOT
    equal to the exact per-sample embedding."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["actor_cond"] = "z"; cfg["actor_z_infer"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        psi = jax.random.normal(jax.random.PRNGKey(1), (6, 8))
        z = agent._infer_actor_z(psi, jax.random.PRNGKey(2))
        assert z.shape == (6, 8), z.shape
        norms = jnp.linalg.norm(z, axis=-1)
        assert jnp.allclose(norms, jnp.sqrt(8.0), atol=1e-3), norms
        assert not jnp.allclose(z, psi), "inferred z must differ from the exact embeddings"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_actor_z_infer_trains_finite():
    """exp4: full agent with actor_z_infer=True (IQL bilinear + z-cond flow actor)
    trains finite -- the actor's Q-term uses <phi(s,a), z_infer>/sqrt(d)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["actor_cond"] = "z"; cfg["actor_z_infer"] = True
        cfg["measure_only"] = False; cfg["measure_coef"] = 0.0
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((4,)), masks=jnp.ones((4,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["actor/actor_loss"]), info["actor/actor_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
