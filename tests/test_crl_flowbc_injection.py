from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_crl_flowbc_agent_builds_state():
    code = """
        import jax, jax.numpy as jnp
        from crl_flowbc import CRLFlowBCAgent, get_config
        cfg = get_config()
        cfg = dict(cfg); cfg["encoder"] = None; cfg["frame_stack"] = None
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = CRLFlowBCAgent.create(0, ex_obs, ex_act, cfg)
        a = agent.sample_actions(ex_obs, ex_obs, seed=jax.random.PRNGKey(0))
        assert a.shape == (2, 5), a.shape
        batch = dict(observations=ex_obs, actions=ex_act, value_goals=ex_obs, actor_goals=ex_obs)
        agent2, info = agent.update(batch)
        assert "actor/bc_flow_loss" in info, list(info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_crl_flowbc_registered_in_agents():
    code = """
        import agents
        assert 'crl_flowbc' in agents.agents, list(agents.agents)
        from crl_flowbc import CRLFlowBCAgent
        assert agents.agents['crl_flowbc'] is CRLFlowBCAgent
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
