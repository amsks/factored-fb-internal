"""Smoke/behavior tests for the HIQL + FlowBC actor graft (hiql_flow).

HIQL's IVL value + state-dependent subgoal rep are inherited unchanged; the
Gaussian-AWR actors are replaced by AWR-weighted flow-matching BC actors, behind
a `flow_levels` knob ('low' | 'high' | 'both') selecting which of HIQL's two
actors is flow. Mirrors tests/test_gciql_flow_injection.py; runs in the jax venv.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_hiql_flow_registered_in_agents():
    code = """
        import agents
        assert 'hiql_flow' in agents.agents, list(agents.agents)
        from hiql_flow import HIQLFlowAgent
        assert agents.agents['hiql_flow'] is HIQLFlowAgent
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_hiql_flow_builds_and_samples_state():
    code = f"""
        import jax, jax.numpy as jnp
        from hiql_flow import HIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["flow_levels"] = "low"
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = HIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        a = agent.sample_actions(ex_obs, ex_obs, seed=jax.random.PRNGKey(0))
        assert a.shape == (4, 5), a.shape
        assert jnp.all(jnp.abs(a) <= 1.0 + 1e-5), "actions must be clipped to [-1,1]"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_hiql_flow_trains_finite_each_level():
    """Each of flow_levels in {low, high, both} builds and trains with finite losses.

    Uses a non-degenerate (random) batch: HIQL's goal_rep ends in LengthNormalize,
    so an all-zero observation gives a zero subgoal rep -> 0/0 NaN (stock HIQL NaNs
    on all-zeros too). Real training data is never degenerate this way.
    """
    code = """
        import jax, jax.numpy as jnp
        from hiql_flow import HIQLFlowAgent, get_config
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        k = jax.random.PRNGKey(0)
        obs = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
        nobs = jax.random.normal(jax.random.PRNGKey(2), (4, 11))
        act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(3), (4, 5)))
        batch = dict(observations=obs, next_observations=nobs, actions=act,
                     rewards=jnp.zeros((4,)), masks=jnp.ones((4,)), value_goals=nobs,
                     low_actor_goals=nobs, high_actor_goals=nobs, high_actor_targets=nobs)
        for lvl in ("low", "high", "both"):
            cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["flow_levels"] = lvl
            agent = HIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            for _ in range(3):
                agent, info = agent.update(batch)
            assert jnp.isfinite(info["value/value_loss"]), (lvl, info["value/value_loss"])
            assert jnp.isfinite(info["low_actor/actor_loss"]), (lvl, info["low_actor/actor_loss"])
            assert jnp.isfinite(info["high_actor/actor_loss"]), (lvl, info["high_actor/actor_loss"])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_hiql_flow_sample_actions_unbatched_state():
    code = """
        import jax, jax.numpy as jnp
        from hiql_flow import HIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["flow_levels"] = "both"
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = HIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        ob = jnp.zeros((11,))  # single, unbatched
        a = agent.sample_actions(ob, ob, seed=jax.random.PRNGKey(0))
        assert a.shape == (5,), a.shape
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
