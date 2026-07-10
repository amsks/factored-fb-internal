"""Joint-psi SF cells (spec: docs/superpowers/specs/2026-07-10-joint-sf-ortho-design.md).

psi is trained by the SF vector TD itself (the base b(.) leaves the stop_gradient), and held
non-degenerate by FB's orthonormality penalty + sqrt(d)-sphere normalization -- instead of being
frozen, identity, or reward-regressed. Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_project_z_puts_psi_on_the_sqrt_d_sphere():
    """FB pins ||B|| = ||z|| = sqrt(d) via a Norm layer. With project_z the learned base leg has
    unit-sqrt(d) rows for EVERY input, so the ortho penalty's diagonal term is the constant -d
    (tr(C) = d) and only the off-diagonal decorrelation is active. Radial collapse (psi -> 0)
    becomes impossible by construction."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
        cfg["sf_reward_regress"] = True          # keeps guard 5 satisfied for now
        cfg["fb_forward_z"] = True
        cfg["project_z"] = True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        goals = jax.random.normal(jax.random.PRNGKey(3), (8, 11))
        p = agent.network.params
        _, _, psi = agent.network.select("critic")(ex_obs, goals, ex_act, goals, info=True, params=p)
        norms = jnp.linalg.norm(psi, axis=-1)               # [2,8]
        assert jnp.allclose(norms, jnp.sqrt(50.0), atol=1e-4), norms
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_psi_ortho_matches_the_exact_finite_batch_fb_formula():
    """DECISIVE. sf_psi_ortho_coef applies FB's orthonormality penalty to psi(next_observations)
    -- the rho marginal, exactly as measure_loss does (B := psi(s')), NOT to the HER value_goals.

    Implemented form (per head, then averaged):
        o_off  = 0.5 * mean_{i!=j} <psi_i, psi_j>^2
        o_diag = - mean_i ||psi_i||^2
    With G = Psi Psi^T and C = Psi^T Psi / B, ||G||_F^2 = tr((Psi^T Psi)^2) = B^2 ||C||_F^2, so

        mean_{i!=j} <psi_i,psi_j>^2 = ( B^2 ||C||_F^2 - sum_i ||psi_i||^4 ) / (B^2 - B)
        o_diag                      = - tr(C)

    exactly. (Asymptotically o_off + o_diag -> 0.5||C - I||_F^2 - d/2, which is why minimizing it
    whitens psi; the finite-batch form above is what the code actually computes.)"""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
        cfg["sf_reward_regress"] = True; cfg["sf_reward_coef"] = 0.0
        cfg["fb_forward_z"] = True
        cfg["sf_psi_ortho_coef"] = 3.0
        cfg["discount"] = 0.0
        B = 12
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                     rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                     value_goals=goals, actor_goals=goals)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        p = agent.network.params
        _, _, psi_o = agent.network.select("critic")(ex_obs, ex_nobs, ex_act, goals, info=True, params=p)
        # exact finite-batch identity, per head then averaged
        G = jnp.einsum('hid,hjd->hij', psi_o, psi_o)                  # [2,B,B]
        sq_norms = jnp.diagonal(G, axis1=1, axis2=2)                  # [2,B] = ||psi_i||^2
        C = jnp.einsum('hid,hie->hde', psi_o, psi_o) / B              # [2,d,d]
        c_fro2 = jnp.sum(C ** 2, axis=(1, 2))                         # [2] = ||C||_F^2
        mean_off = (B * B * c_fro2 - jnp.sum(sq_norms ** 2, axis=1)) / (B * B - B)
        exp_orth = jnp.mean(0.5 * mean_off - jnp.mean(sq_norms, axis=1))
        assert jnp.allclose(info["psi_ortho"], exp_orth, rtol=1e-4), (info["psi_ortho"], exp_orth)
        # and it is the term added to the loss (discount=0, reward_coef=0 -> loss = td + 3*orth)
        _, phi, _ = agent.network.select("critic")(ex_obs, goals, ex_act, goals, info=True, params=p)
        _, _, b = agent.network.select("critic")(ex_obs, ex_obs, ex_act, goals, info=True, params=p)
        exp_td = jnp.mean((phi - b) ** 2)
        assert jnp.allclose(loss, exp_td + 3.0 * exp_orth, rtol=1e-4), (loss, exp_td, exp_orth)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_psi_ortho_diag_is_constant_minus_d_under_project_z():
    """Under project_z every psi row has norm sqrt(d), so tr(C) = d and the ortho DIAGONAL term is
    the constant -d for any batch. The sphere kills the radial collapse; only the off-diagonal
    (angular) decorrelation term still has a gradient."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def build_info(seed):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["sf_reward_regress"] = True; cfg["sf_reward_coef"] = 0.0
            cfg["fb_forward_z"] = True; cfg["project_z"] = True
            cfg["sf_psi_ortho_coef"] = 1.0
            B = 10
            ex_obs = jax.random.normal(jax.random.PRNGKey(seed), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(seed + 100), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            _, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
            return info
        i0, i1 = build_info(0), build_info(7)
        for i in (i0, i1):
            assert jnp.allclose(i["psi_ortho_diag"], -50.0, atol=1e-3), i["psi_ortho_diag"]
            assert jnp.allclose(i["psi_norm_mean"], jnp.sqrt(50.0), atol=1e-3), i["psi_norm_mean"]
        assert not jnp.allclose(i0["psi_ortho_off"], i1["psi_ortho_off"]), "off-diag must vary"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
