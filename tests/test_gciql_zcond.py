"""Tests for gciql_flow actor_cond='z' (G4: zero-shot z-conditioned actor on the
IQL-value rep). Default actor_cond='goal' must reproduce the stock gciql_flow.
Runs in the jax venv subprocess. See PAPER/specs/2026-06-16-g4-*.md."""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def _full_batch(obs="ex_obs", act="ex_act"):
    return (
        f"dict(observations={obs}, next_observations={obs}, actions={act}, "
        f"rewards=jnp.zeros((2,)), masks=jnp.ones((2,)), "
        f"value_goals={obs}, actor_goals={obs})"
    )


def test_actor_cond_defaults_to_goal():
    code = """
        from gciql_flow import get_config
        cfg = get_config()
        assert cfg.actor_cond == 'goal', cfg.actor_cond
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_trains_finite():
    code = f"""
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'bilinear'; cfg['value_latent_dim'] = 16
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = {_full_batch()}
        for _ in range(5):
            agent, info = agent.update(batch)
        for k in ('actor/actor_loss', 'actor/q_mean', 'value/value_loss', 'critic/critic_loss'):
            assert jnp.isfinite(info[k]), (k, info[k])
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_actor_loss_no_grad_into_value():
    code = f"""
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'bilinear'; cfg['value_latent_dim'] = 16
        ex_obs = jnp.ones((2, 11)); ex_act = jnp.ones((2, 5)) * 0.3
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = {_full_batch()}
        def aloss(p):
            l, _ = agent.actor_loss(batch, p, rng=jax.random.PRNGKey(0))
            return l
        g = jax.grad(aloss)(agent.network.params)
        vnorm = sum(float(jnp.sum(x ** 2)) for x in jax.tree_util.tree_leaves(g['modules_value']))
        assert vnorm == 0.0, vnorm  # stop_grad keeps the rep purely IQL-trained
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_eval_script_imports():
    # The eval script must import cleanly (path wiring + helpers resolve) without
    # a checkpoint. Full eval is a manual GPU/CPU run, not a unit test.
    code = """
        import importlib.util, sys
        from pathlib import Path
        REPO = Path.cwd().parents[2]  # third_party/ogbench/impls -> repo root
        path = REPO / 'scripts' / 'eval_zcond_gciql.py'
        spec = importlib.util.spec_from_file_location('eval_zcond_gciql', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert hasattr(m, 'rollout') and hasattr(m, 'main')
        assert m.THRESH == 0.04 and m.CUBE_QPOS == slice(14, 17)
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_load_agent_accepts_actor_cond():
    # Signature-level guard: load_agent must expose an actor_cond param so the
    # eval script can rebuild a z-cond template before restore.
    code = """
        import inspect, sys
        from pathlib import Path
        REPO = Path.cwd().parents[2]  # third_party/ogbench/impls -> repo root
        sys.path.insert(0, str(REPO))
        from scripts.phase_probe_gciql import load_agent
        params = inspect.signature(load_agent).parameters
        assert 'actor_cond' in params, list(params)
        assert params['actor_cond'].default == 'goal', params['actor_cond'].default
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_requires_bilinear():
    code = """
        import jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'mlp'
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        try:
            GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        except AssertionError:
            print('OK')
        else:
            print('NO-RAISE')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_sample_actions_maps_raw_goal_to_z():
    # The eval harness hands sample_actions the RAW goal obs (11-dim); the z-actor
    # maps it through value.psi internally, so the call must succeed end to end.
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'bilinear'; cfg['value_latent_dim'] = 16
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        goal = jnp.zeros((2, 11))  # raw goal obs, NOT a z
        a = agent.sample_actions(ex_obs, goal, seed=jax.random.PRNGKey(0))
        assert a.shape == (2, 5), a.shape
        assert jnp.all(jnp.abs(a) <= 1.0 + 1e-5)
        # unbatched (single obs/goal), the eval-loop calling convention
        a1 = agent.sample_actions(ex_obs[0], goal[0], seed=jax.random.PRNGKey(1))
        assert a1.shape == (5,), a1.shape
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_zcond_sample_actions_with_z_takes_d_dim_z():
    # The inferred-z arm feeds a precomputed d-dim z straight to the actor.
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config()); cfg['encoder'] = None; cfg['frame_stack'] = None
        cfg['actor_cond'] = 'z'; cfg['value_arch'] = 'bilinear'; cfg['value_latent_dim'] = 16
        ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        z = jnp.zeros((16,))  # precomputed task vector, single
        a = agent.sample_actions_with_z(ex_obs[0], z, seed=jax.random.PRNGKey(0))
        assert a.shape == (5,), a.shape
        assert jnp.all(jnp.abs(a) <= 1.0 + 1e-5)
        print('OK')
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
