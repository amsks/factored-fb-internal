"""E2 scale-controlled measure interpolation: the two default-OFF knobs that let the FB
successor-measure loss be grafted onto a WORKING IQL critic without the Q-scale blowup
(audit-20260706 §1: 18/18 mc* runs crashed via unbounded <F,z> inflation).

  measure_scale_norm : divide the measure TD terms by a stop-gradded measure scale so the
                       gradient is dimensionless (the trick the actor q_loss already uses:
                       q_loss = -q.mean()/sg[|q|.mean()+eps]). offdiag/sg[mean(M^2)+eps],
                       diag/sg[sqrt(mean(M^2))+eps]. Ortho term unchanged.
  measure_stopgrad_phi: stop-grad phi inside measure_loss (both online legs) so the measure
                       gradient flows ONLY into psi (the goal/readout leg), quarantining it
                       from the value-critical forward leg phi.

Both default OFF: with them off measure_loss is bit-exact the canonical FB reduction.
Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_measure_terms_default_matches_canonical_formula():
    """REFACTOR GUARD: _measure_terms(scale_norm=False) reproduces the exact FB reduction
    offdiag = 0.5*sum((diff*off)^2)/sum(off), diag = -mean(diag(diff)), diff = M-gamma*tM,
    for both a [B,B] (value leg) and a [2,B,B] (critic per-head) measure matrix."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent
        gamma = 0.99
        for shape in [(5, 5), (2, 5, 5)]:
            M = jax.random.normal(jax.random.PRNGKey(0), shape)
            tM = jax.random.normal(jax.random.PRNGKey(1), shape)
            offdiag, diagt = GCIQLFlowAgent._measure_terms(M, tM, gamma, False)
            B = shape[-1]; off = 1.0 - jnp.eye(B)
            diff = M - gamma * tM
            exp_off = 0.5 * jnp.sum((diff * off) ** 2, axis=(-2, -1)) / jnp.sum(off)
            exp_dia = -jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1), axis=-1)
            assert jnp.allclose(offdiag, exp_off, atol=1e-6), (shape, offdiag, exp_off)
            assert jnp.allclose(diagt, exp_dia, atol=1e-6), (shape, diagt, exp_dia)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_measure_scale_norm_is_scale_invariant():
    """scale_norm=True => (offdiag+diag) is invariant (<=1e-4 rel) to multiplying the measure
    matrices by 10 (== scaling phi,psi features); scale_norm=False is NOT (it scales ~100x).
    This is the mechanism that stops the <F,z> inflation from blowing up the shared critic."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent
        gamma = 0.99
        M = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 6)) * 3.0
        tM = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 6)) * 3.0
        def total(scale, c):
            o, dgt = GCIQLFlowAgent._measure_terms(M * c, tM * c, gamma, scale)
            return float(jnp.mean(o + dgt))
        base_n = total(True, 1.0); scl_n = total(True, 10.0)
        rel = abs(scl_n - base_n) / (abs(base_n) + 1e-9)
        assert rel <= 1e-4, ("scale_norm not invariant", base_n, scl_n, rel)
        base_r = total(False, 1.0); scl_r = total(False, 10.0)
        assert abs(scl_r - base_r) / (abs(base_r) + 1e-9) > 1.0, ("raw should scale", base_r, scl_r)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_measure_stopgrad_phi_quarantines_gradient_to_psi():
    """measure_stopgrad_phi=True => d(measure_loss)/d(critic phi params) == 0 while
    d/d(psi params) != 0 (the measure gradient shapes only the backward/readout leg).
    With it OFF, phi receives gradient (control)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def make(sg):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 12
            cfg["measure_coef"] = 1.0; cfg["measure_legs"] = "critic"
            cfg["measure_stopgrad_phi"] = sg
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (16, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (16, 5))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                         rewards=jnp.zeros((16,)), masks=jnp.ones((16,)),
                         value_goals=ex_obs, actor_goals=ex_obs)
            g = jax.grad(lambda p: agent.measure_loss(batch, p, jax.random.PRNGKey(2))[0])(
                agent.network.params)
            def absnorm(needle):
                flat = jax.tree_util.tree_flatten_with_path(g["modules_critic"])[0]
                tot = 0.0
                for p, v in flat:
                    s = ".".join(str(getattr(k, "key", k)) for k in p)
                    if needle in s: tot += float(jnp.sum(jnp.abs(v)))
                return tot
            return absnorm("phi"), absnorm("psi")
        phi_sg, psi_sg = make(True)
        phi_no, psi_no = make(False)
        assert phi_sg < 1e-8, ("phi grad must vanish under stopgrad", phi_sg)
        assert psi_sg > 1e-6, ("psi grad must survive under stopgrad", psi_sg)
        assert phi_no > 1e-6, ("phi grad must exist without stopgrad", phi_no)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_measure_knobs_compose_with_iql_finite_and_bounded():
    """Both knobs ON compose with measure_coef>0 on top of the full IQL loss: 5 updates stay
    finite and critic q_mean stays within [-100, 10] (the audit's scale-escape guardrail)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 12
        cfg["measure_coef"] = 1.0; cfg["measure_legs"] = "critic"
        cfg["measure_scale_norm"] = True; cfg["measure_stopgrad_phi"] = True
        # defaults must be OFF
        assert get_config()["measure_scale_norm"] is False
        assert get_config()["measure_stopgrad_phi"] is False
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (16, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (16, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((16,)), masks=jnp.ones((16,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["measure/measure_loss"]), info.get("measure/measure_loss")
        qm = float(info["critic/q_mean"])
        assert jnp.isfinite(qm) and -100.0 <= qm <= 10.0, ("q_mean escaped", qm)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
