"""Tests for the D1 knob `actor_z_from_critic`: source the flow actor's task vector z
from the CRITIC's psi, not the value net's psi.

Under measure_only the IQL value net is never trained, so actor_cond='z' (which today pulls
z=psi from the *value* net, `_value_psi`) feeds the actor a garbage task vector. Real FB
conditions the actor and the critic on the SAME z. `actor_z_from_critic` routes the actor's z
through the critic's psi (=B, trained by the measure loss), closing that mismatch (audit #2).

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_actor_z_sourced_from_critic_not_value():
    """`_critic_psi(goals)` must equal the critic's own psi (head-mean) and differ from
    `_value_psi(goals)` (a distinct, untrained-under-measure network)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["fb_forward_z"] = True; cfg["measure_coef"] = 1.0
        cfg["actor_cond"] = "z"; cfg["actor_z_from_critic"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        g = jnp.linspace(-1, 1, 44).reshape(4, 11)
        params = agent.network.params

        zc = agent._critic_psi(g, params)                 # [B, d]
        # the critic's own psi (ensembled [2,B,d]) -> head-mean
        _, _, psi_c = agent.network.select('critic')(g, g, ex_act, g, info=True)
        assert zc.shape == (4, 8), zc.shape
        assert jnp.allclose(zc, psi_c.mean(axis=0), atol=1e-4), "critic-z must be the critic psi head-mean"

        zv = agent._value_psi(g, params)                  # value net psi
        assert not jnp.allclose(zc, zv), "critic-z must differ from value-net-z (distinct nets)"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_critic_psi_handles_unbatched_goal():
    """At eval the harness hands a SINGLE unbatched goal (obs_dim,); _critic_psi must
    return a (d,) task vector -- not mis-read the obs dim as a batch dim."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["fb_forward_z"] = True; cfg["measure_coef"] = 1.0
        cfg["actor_cond"] = "z"; cfg["actor_z_from_critic"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        g1 = jnp.linspace(-1, 1, 11)                 # single unbatched goal (11,)
        z = agent._critic_psi(g1, agent.network.params)
        assert z.shape == (8,), z.shape
        gb = jnp.ones((6, 11))                        # batched still works
        zb = agent._critic_psi(gb, agent.network.params)
        assert zb.shape == (6, 8), zb.shape
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_actor_z_from_critic_trains_finite():
    """The D1 cell (measure_only + fb_forward_z + actor_cond='z' + actor_z_from_critic)
    trains finite for a few steps -- the actor now steers on the trained critic psi."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["fb_forward_z"] = True; cfg["measure_coef"] = 1.0
        cfg["measure_ortho_coef"] = 1000.0
        cfg["actor_cond"] = "z"; cfg["actor_z_from_critic"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        # distinct goals so the measure ortho on B=psi is well-posed (see project_z test).
        k = jax.random.PRNGKey(0)
        r = lambda s, sh: jax.random.normal(jax.random.fold_in(k, s), sh)
        batch = dict(observations=r(1, (8, 11)), next_observations=r(2, (8, 11)),
                     actions=jnp.tanh(r(3, (8, 5))), rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=r(4, (8, 11)), actor_goals=r(5, (8, 11)))
        for _ in range(5):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["actor/actor_loss"]), info["actor/actor_loss"]
        assert jnp.isfinite(info["measure/measure_loss"]), info["measure/measure_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
