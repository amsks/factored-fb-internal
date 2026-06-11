"""Smoke tests for the GCIQL + FlowBC-actor graft (gciql_flow).

GCIQL's IQL critic/value are inherited unchanged; only the actor is swapped for
FQL's flow actor (bc-flow + one-step distillation + Q-max on the frozen critic).
Mirrors tests/test_crl_flowbc_injection.py. Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def _full_batch(obs="ex_obs", act="ex_act"):
    # GCIQL needs value+critic targets, not just actor goals.
    return (
        f"dict(observations={obs}, next_observations={obs}, actions={act}, "
        f"rewards=jnp.zeros((2,)), masks=jnp.ones((2,)), "
        f"value_goals={obs}, actor_goals={obs})"
    )


def test_gciql_flow_agent_builds_state():
    code = f"""
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = get_config()
        cfg = dict(cfg); cfg["encoder"] = None; cfg["frame_stack"] = None
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        a = agent.sample_actions(ex_obs, ex_obs, seed=jax.random.PRNGKey(0))
        assert a.shape == (2, 5), a.shape
        assert jnp.all(jnp.abs(a) <= 1.0 + 1e-5), "actions must be clipped to [-1,1]"
        batch = {_full_batch()}
        agent2, info = agent.update(batch)
        for k in ("actor/bc_flow_loss", "actor/q_mean", "value/value_loss", "critic/critic_loss"):
            assert k in info, list(info)
            assert jnp.isfinite(info[k]), (k, info[k])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_gciql_flow_trains_a_few_steps_finite():
    """Several updates keep every loss finite (no NaN blowups from the flow rollout)."""
    code = f"""
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = get_config()
        cfg = dict(cfg); cfg["encoder"] = None; cfg["frame_stack"] = None
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = {_full_batch()}
        for _ in range(5):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["actor/actor_loss"]), info["actor/actor_loss"]
        assert jnp.isfinite(info["critic/critic_loss"]), info["critic/critic_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_gciql_flow_registered_in_agents():
    code = """
        import agents
        assert 'gciql_flow' in agents.agents, list(agents.agents)
        from gciql_flow import GCIQLFlowAgent
        assert agents.agents['gciql_flow'] is GCIQLFlowAgent
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_gciql_flow_sample_actions_unbatched_state():
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = get_config()
        cfg = dict(cfg); cfg["encoder"] = None; cfg["frame_stack"] = None
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        ob = jnp.zeros((11,))  # single, unbatched
        a = agent.sample_actions(ob, ob, seed=jax.random.PRNGKey(0))
        assert a.shape == (5,), a.shape
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_gciql_flow_sample_actions_unbatched_pixel():
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = get_config()
        cfg = dict(cfg); cfg["encoder"] = "impala"; cfg["frame_stack"] = None
        ex_obs = jnp.zeros((2, 64, 64, 3)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        ob = jnp.zeros((64, 64, 3))  # single, unbatched image
        a = agent.sample_actions(ob, ob, seed=jax.random.PRNGKey(0))
        assert a.shape == (5,), a.shape
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
