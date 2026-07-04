"""Tests for the FB sphere-normalization knob `project_z` on the bilinear critics.

FB pins the backward map and task vector to the sphere of radius sqrt(d): ||B||=||z||=sqrt(d)
(a Norm layer in agents/fb/model.py). GCIQL's bilinear critics let psi (=B) and z float. The
`project_z` knob hard-projects both z (the forward map's task input) and psi (=B) onto that
sphere inside the critic forward -- FB's actual normalization regime, needed by the B4/C4
"full FB structure" endpoints. These pin (i) the projection actually fixes ||.||=sqrt(d), and
(ii) it is a drop-in for both the IQL two-tower critic (B) and the measure fb_forward_z critic (C).

Runs in the jax venv subprocess (mirrors tests/test_gciql_two_tower_critic.py).
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_project_z_normalizes_psi_to_sqrt_d():
    """With project_z=True the critic's backward map psi has L2 norm sqrt(d) per row
    (for both GCFBValue and GCTwoTowerCritic); without it, norms float off sqrt(d)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCFBValue, GCTwoTowerCritic
        d = 8; sqd = jnp.sqrt(d * 1.0)
        obs = jnp.ones((4, 11)); acts = jnp.ones((4, 5)); g = jnp.linspace(-1, 1, 44).reshape(4, 11)

        # GCFBValue (measure/C path): signature (obs, b_goals, actions, z_goals)
        m = GCFBValue(hidden_dims=(32, 32), latent_dim=d, layer_norm=True, ensemble=True, project_z=True)
        p = m.init(jax.random.PRNGKey(0), obs, g, acts, g)['params']
        _, phi, psi = m.apply({'params': p}, obs, g, acts, g, info=True)
        n = jnp.linalg.norm(psi, axis=-1)
        assert jnp.allclose(n, sqd, atol=1e-3), ('GCFBValue psi norm', n)

        # GCTwoTowerCritic (IQL/B path): signature (obs, goals, actions)
        t = GCTwoTowerCritic(hidden_dims=(32, 32), latent_dim=d, layer_norm=True, ensemble=True, project_z=True)
        pt = t.init(jax.random.PRNGKey(1), obs, g, acts)['params']
        _, tphi, tpsi = t.apply({'params': pt}, obs, g, acts, info=True)
        nt = jnp.linalg.norm(tpsi, axis=-1)
        assert jnp.allclose(nt, sqd, atol=1e-3), ('TwoTower psi norm', nt)

        # off by default: norms should NOT all equal sqrt(d)
        m0 = GCFBValue(hidden_dims=(32, 32), latent_dim=d, layer_norm=True, ensemble=True)
        p0 = m0.init(jax.random.PRNGKey(0), obs, g, acts, g)['params']
        _, _, psi0 = m0.apply({'params': p0}, obs, g, acts, g, info=True)
        assert not jnp.allclose(jnp.linalg.norm(psi0, axis=-1), sqd, atol=1e-3), "should float when off"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_project_z_trains_finite_iql_and_measure():
    """project_z is a drop-in: both the B4 cell (IQL two-tower + project_z) and the C4
    cell (measure fb_forward_z + project_z) train finite for a few steps."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        # distinct goals/obs so the FB ortho penalty on sqrt(d)-projected psi is well-posed
        # (an all-identical batch makes every psi equal -> degenerate ortho, not a real regime).
        k = jax.random.PRNGKey(0)
        r = lambda s, sh: jax.random.normal(jax.random.fold_in(k, s), sh)
        batch = dict(observations=r(1, (8, 11)), next_observations=r(2, (8, 11)),
                     actions=jnp.tanh(r(3, (8, 5))), rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=r(4, (8, 11)), actor_goals=r(5, (8, 11)))

        def run(extra):
            cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8; cfg["project_z"] = True
            cfg.update(extra)
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            for _ in range(5):
                agent, info = agent.update(batch)
            return info

        b4 = run(dict(fwd_two_tower=True, measure_only=False, measure_coef=0.0, iql_ortho_coef=1.0))
        assert jnp.isfinite(b4["actor/actor_loss"]), b4["actor/actor_loss"]
        c4 = run(dict(fb_forward_z=True, measure_only=True, measure_coef=1.0, measure_ortho_coef=1.0))
        assert jnp.isfinite(c4["measure/measure_loss"]), c4["measure/measure_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
