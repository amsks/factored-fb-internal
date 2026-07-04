"""Test: FB orthonormality penalty on the IQL bilinear critic, with NO measure loss.

The ortho term ||E[psi psi^T] - I|| currently lives only inside measure_loss, so it
can only run alongside the FB successor-measure objective. This isolates it: apply the
SAME ortho penalty on the IQL critic's backward map psi(value_goals) while keeping the
pure IQL value/critic objective (the ~90 bilinear-GCIQL) -- "bilinear GCIQL + ortho
only, nothing else from FB". `iql_ortho_coef` gates it.

Runs in the jax venv subprocess (mirrors tests/test_gciql_fb_forward_z.py).
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_iql_ortho_only_adds_finite_penalty_under_iql():
    """value_arch=bilinear, pure IQL (measure_only=False, measure_coef=0) + iql_ortho_coef>0:
    total_loss must expose a finite ortho penalty AND keep the IQL value/critic/actor losses
    (i.e. the ortho rides on the IQL objective, not the measure one)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = False; cfg["measure_coef"] = 0.0   # pure IQL (no measure loss)
        cfg["iql_ortho_coef"] = 1.0                               # ONLY the FB orthonormality term
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((4,)), masks=jnp.ones((4,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert "ortho/ortho_loss" in info, list(info)
        for k in ("ortho/ortho_loss", "value/value_loss", "critic/critic_loss", "actor/actor_loss"):
            assert k in info, list(info)
            assert jnp.isfinite(info[k]), (k, info[k])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_iql_ortho_coef_zero_adds_no_penalty():
    """With iql_ortho_coef=0 (default) the ortho key must be ABSENT -- the penalty is
    strictly opt-in and stock IQL bilinear is unchanged."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = False; cfg["measure_coef"] = 0.0
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((4,)), masks=jnp.ones((4,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        agent, info = agent.update(batch)
        assert "ortho/ortho_loss" not in info, list(info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
