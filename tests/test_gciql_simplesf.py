"""GC-SimpleSF (spec: docs/superpowers/specs/2026-07-10-gc-simple-sf-design.md).

Chua et al.'s Simple SF ported onto the GCIQL trunk. Q = <Psi(s,a,z), z>/sqrt(d) with z = psi(g),
trained by an ordinary scalar TD; the task vector psi(g) is optionally grounded by a
reward-prediction loss and by NOTHING else. The reference's trunk split is the load-bearing part:
b = h/||h|| and Psi = head(h) are different vectors off one trunk, so the reward model <b, psi(g)>
is ASYMMETRIC and has no psi=0 saddle (unlike E3's symmetric <psi(s), psi(g)>).

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_simplesf_readout_shapes_and_bilinear_form():
    """Q_i = <Psi_i, z>/sqrt(d) with twin heads on the critic and one head on the value net.
    info=True returns (v, Psi, z, b)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        B = 7
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        p = agent.network.params
        q, Psi, z, b = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        assert Psi.shape == (2, B, 50), Psi.shape          # twin heads
        assert z.shape == (B, 50) and b.shape == (B, 50), (z.shape, b.shape)
        assert jnp.allclose(q, (Psi * z[None]).sum(-1) / jnp.sqrt(50.0), atol=1e-5)
        v, PsiV, zV, bV = agent.network.select("value")(ex_obs, goals, info=True, params=p)
        assert PsiV.shape == (B, 50), PsiV.shape           # value net: one head
        assert jnp.allclose(v, (PsiV * zV).sum(-1) / jnp.sqrt(50.0), atol=1e-5)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_basis_is_unit_norm_and_not_collinear_with_psi_features():
    """DECISIVE for the no-saddle property. b = h/||h|| has unit norm and is a DIFFERENT vector
    from Psi = head(h). If b were collinear with Psi, the reward loss <sg[b], psi(g)> would be a
    stop-gradded copy of Q and would demand Q(s,a,g) ~ r_g(s) -- circular and wrong."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        B = 7
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        p = agent.network.params
        _, Psi, _, b = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        assert jnp.allclose(jnp.linalg.norm(b, axis=-1), 1.0, atol=1e-4)
        bn = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        pn = Psi[0] / (jnp.linalg.norm(Psi[0], axis=-1, keepdims=True) + 1e-8)
        cos = jnp.abs((bn * pn).sum(-1))
        assert jnp.max(cos) < 0.99, ("b must not be collinear with Psi", float(jnp.max(cos)))
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
