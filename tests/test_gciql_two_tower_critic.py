"""Tests for the FB-style two-tower forward map on GCIQL's bilinear critic (#4).

Stock GCBilinearValue factorizes Q(s,a,g)=phi(s,a).psi(g): the forward leg phi(s,a)
is goal-INDEPENDENT. FB's forward map instead embeds (s,a) and (s,z) in two towers
and combines them: F = trunk(MLP[s,a], MLP[s,z]). GCTwoTowerCritic ports that shape
into the bilinear critic with z=stop_grad(psi(g)) (the goal embedding), trained under
the *IQL* objective (NOT the measure loss) -- the architecture-only ablation of the
~90 bilinear-GCIQL. These tests pin (i) the forward leg now depends on the goal via z,
and (ii) it is a drop-in for the IQL value/critic losses (finite training).

Runs in the jax venv subprocess (mirrors tests/test_gciql_fb_forward_z.py).
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_two_tower_forward_leg_depends_on_goal():
    """GCTwoTowerCritic.phi (forward map) must change with the goal (its second tower
    consumes z=psi(g)); stock GCBilinearValue.phi=phi(s,a) would NOT. Same drop-in
    (obs, goals, actions) signature as GCBilinearValue."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCTwoTowerCritic
        m = GCTwoTowerCritic(hidden_dims=(32, 32), latent_dim=8, layer_norm=True, ensemble=True)
        obs = jnp.ones((4, 11)); acts = jnp.ones((4, 5))
        g1 = jnp.zeros((4, 11)); g2 = jnp.ones((4, 11))      # different goals -> different z=psi(g)
        params = m.init(jax.random.PRNGKey(0), obs, g1, acts)['params']
        v1, phi1, psi1 = m.apply({'params': params}, obs, g1, acts, info=True)
        v2, phi2, psi2 = m.apply({'params': params}, obs, g2, acts, info=True)
        assert phi1.shape == (2, 4, 8), phi1.shape
        assert v1.shape == (2, 4), v1.shape
        assert not jnp.allclose(phi1, phi2), "two-tower forward leg must depend on goal (via z=psi(g))"
        assert not jnp.allclose(psi1, psi2), "backward map psi must depend on the goal"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_two_tower_critic_iql_trains_finite():
    """Agent with value_arch=bilinear + fwd_two_tower (NO measure loss; pure IQL)
    is a drop-in: value/critic/actor losses train finite for a few steps."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["fwd_two_tower"] = True
        cfg["measure_only"] = False; cfg["measure_coef"] = 0.0   # pure IQL (the 90-performer)
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        # the two-tower critic must actually be wired (stock bilinear has no embed_sz tower)
        flat = jax.tree_util.tree_flatten_with_path(agent.network.params['modules_critic'])[0]
        paths = [".".join(str(getattr(k, 'key', k)) for k in p) for p, _ in flat]
        assert any("embed_sz" in s for s in paths), paths[:8]
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
