"""Tests for the z-conditioned FB forward map on GCIQL's bilinear critic.

The audit found GCIQL's bilinear critic factorizes Q(s,a,g)=phi(s,a).psi(g) with NO
task/z on the forward leg, so the grafted FB "successor measure" is the measure of a
single goal-randomized policy rather than a z-indexed family (the defining FB
structure F(s,a,z).B(s')). `fb_forward_z=True` makes the forward map consume a
stop-gradded task vector z=psi(goal); these tests pin that wiring.

Runs in the jax venv subprocess (mirrors tests/test_gciql_flow_injection.py).
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_fb_value_forward_map_depends_on_z():
    """GCFBValue.phi (the forward map F) must change with the task goal z_goals;
    psi (the backward map B) must NOT, since it reads only b_goals."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCFBValue
        m = GCFBValue(hidden_dims=(32, 32), latent_dim=8, layer_norm=True, ensemble=True)
        obs = jnp.ones((4, 11)); b_goals = jnp.ones((4, 11)); acts = jnp.ones((4, 5))
        zg1 = jnp.zeros((4, 11)); zg2 = jnp.ones((4, 11))   # different task goals -> different z
        params = m.init(jax.random.PRNGKey(0), obs, b_goals, acts, zg1)['params']
        _, phi1, psi1 = m.apply({'params': params}, obs, b_goals, acts, zg1, info=True)
        _, phi2, psi2 = m.apply({'params': params}, obs, b_goals, acts, zg2, info=True)
        assert phi1.shape == (2, 4, 8), phi1.shape
        assert not jnp.allclose(phi1, phi2), "forward map F must depend on z (task-conditioned)"
        assert jnp.allclose(psi1, psi2), "backward map B must depend only on b_goals, not z"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_gciql_fb_forward_z_measure_only_trains_finite():
    """Agent with fb_forward_z + measure_only trains a few steps with finite losses,
    and the z-conditioned critic is actually built (forward map sees value_goals)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["measure_coef"] = 1.0
        cfg["measure_legs"] = "critic"; cfg["measure_ortho_coef"] = 1.0
        cfg["fb_forward_z"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((4,)), masks=jnp.ones((4,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        for k in ("measure/measure_loss", "measure/measure_orth", "actor/actor_loss"):
            assert k in info, list(info)
            assert jnp.isfinite(info[k]), (k, info[k])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
