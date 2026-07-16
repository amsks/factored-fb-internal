"""value_arch='bilinear_sg': state-conditioned backward leg, Q(s,a,g)=phi(s,a).psi(s,g)/sqrt(d).

Stock GCBilinearValue factorizes Q(s,a,g)=phi(s,a).psi(g): the backward leg psi sees
ONLY the goal, which is what makes it an FB-style rank-d goal embedding (and what makes
z=psi(g) a well-defined task vector). GCStatePsiValue instead feeds psi the pair (s,g),
so the "goal embedding" becomes state-dependent. This is the natural expressivity
ablation of the bilinear bridge cell: it asks whether the bilinear win comes from the
low-rank GOAL bottleneck specifically, or merely from the inner-product head.

Note this deliberately breaks the zero-shot task-inference API (there is no longer a
state-independent z=psi(g)); that is the point of the ablation, not a bug.

Runs in the jax venv subprocess (mirrors tests/test_gciql_two_tower_critic.py).
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_psi_depends_on_both_state_and_goal():
    """psi must change with BOTH the state and the goal; phi keeps its phi(s,a) form."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCStatePsiValue
        m = GCStatePsiValue(hidden_dims=(32, 32), latent_dim=8, layer_norm=True, ensemble=True)
        acts = jnp.ones((4, 5))
        s1 = jnp.zeros((4, 11)); s2 = jnp.ones((4, 11))
        g1 = jnp.zeros((4, 11)); g2 = jnp.ones((4, 11))
        params = m.init(jax.random.PRNGKey(0), s1, g1, acts)['params']
        ap = lambda s, g: m.apply({'params': params}, s, g, acts, info=True)
        v11, phi11, psi11 = ap(s1, g1)
        v21, phi21, psi21 = ap(s2, g1)      # state changed, goal fixed
        v12, phi12, psi12 = ap(s1, g2)      # goal changed, state fixed
        assert psi11.shape == (2, 4, 8), psi11.shape
        assert v11.shape == (2, 4), v11.shape
        assert not jnp.allclose(psi11, psi21), "psi(s,g) must depend on the STATE"
        assert not jnp.allclose(psi11, psi12), "psi(s,g) must depend on the GOAL"
        # phi is still the goal-independent forward leg phi(s,a)
        assert jnp.allclose(phi11, phi12), "phi(s,a) must NOT depend on the goal"
        assert not jnp.allclose(phi11, phi21), "phi(s,a) must depend on the state"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_q_is_the_scaled_inner_product():
    """Q must be exactly (phi * psi).sum(-1)/sqrt(d) -- same readout as GCBilinearValue."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCStatePsiValue
        d = 8
        m = GCStatePsiValue(hidden_dims=(32,), latent_dim=d, layer_norm=True, ensemble=True)
        s = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
        g = jax.random.normal(jax.random.PRNGKey(2), (4, 11))
        a = jax.random.normal(jax.random.PRNGKey(3), (4, 5))
        params = m.init(jax.random.PRNGKey(0), s, g, a)['params']
        v, phi, psi = m.apply({'params': params}, s, g, a, info=True)
        expected = (phi * psi / jnp.sqrt(d)).sum(-1)
        assert jnp.allclose(v, expected, atol=1e-6), (v[:2], expected[:2])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_stock_bilinear_psi_still_goal_only():
    """REGRESSION GUARD: the existing value_arch='bilinear' path must be untouched --
    its psi must remain a function of the goal ALONE. Every bridge checkpoint on disk
    depends on this."""
    code = """
        import jax, jax.numpy as jnp
        from utils.networks import GCBilinearValue
        m = GCBilinearValue(hidden_dims=(32,), latent_dim=8, layer_norm=True, ensemble=True)
        acts = jnp.ones((4, 5))
        s1 = jnp.zeros((4, 11)); s2 = jnp.ones((4, 11)); g = jnp.ones((4, 11))
        params = m.init(jax.random.PRNGKey(0), s1, g, acts)['params']
        _, _, psi1 = m.apply({'params': params}, s1, g, acts, info=True)
        _, _, psi2 = m.apply({'params': params}, s2, g, acts, info=True)
        assert jnp.allclose(psi1, psi2), "stock bilinear psi must depend on the goal ONLY"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_agent_trains_end_to_end():
    """The real entry point: build the agent with value_arch='bilinear_sg' and run actual
    update() steps. Component tests that never call update() have burned this repo before."""
    code = """
        import jax, jax.numpy as jnp, numpy as np
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear_sg"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = False; cfg["measure_coef"] = 0.0     # pure IQL
        ex_obs = jnp.zeros((1, 11), jnp.float32); ex_act = jnp.zeros((1, 5), jnp.float32)
        ag = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        rng = np.random.default_rng(0)
        B = 16
        batch = dict(
            observations=rng.normal(size=(B, 11)).astype(np.float32),
            next_observations=rng.normal(size=(B, 11)).astype(np.float32),
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, 11)).astype(np.float32),
            actor_goals=rng.normal(size=(B, 11)).astype(np.float32),
            rewards=rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
            masks=rng.integers(0, 2, size=B).astype(np.float32),
        )
        for i in range(5):
            ag, info = ag.update(batch)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        a = ag.sample_actions(observations=batch["observations"],
                              goals=batch["actor_goals"],
                              seed=jax.random.PRNGKey(0), temperature=0.0)
        assert np.all(np.isfinite(np.asarray(a))), "actions not finite"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
