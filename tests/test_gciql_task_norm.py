"""Task-weighted successor moments (spec: docs/superpowers/specs/2026-07-10-task-weighted-moments-design.md).

FB's measure loss tests its residual against psi(s'_j), s' ~ rho (the replay marginal). The research
note argues control needs the residual weighted by the TASK norm instead. For goal indicators the
moment map collapses (z_g is proportional to psi(g)), so the task norm is exactly: draw the B x B
test states from value_goals (nu) rather than next_observations (rho).

Two invariants make the cell non-degenerate, and both are asserted here:
  * the ORTHO stays on psi(next_observations). If it followed the test set, C_nu -> I at the ortho
    optimum and the objective would silently become canonical SF (measured: 0.120).
  * the DIRAC term stays on psi(s'_i). It is the reward. If it followed the test set, the diagonal
    becomes Q - gamma*Qbar and the reward vanishes.

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

# Shared config + batch preamble for a measure_only FB cell (the C-fbz-projz-ortho1000 shape).
# NOTE the 8-space indent: run_jax() applies textwrap.dedent to PREAMBLE + body, so both halves
# must carry the SAME leading indent or the combined snippet raises IndentationError.
PREAMBLE = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config

        def build(test_states="rho", test_rank="full", lam=2.0, gamma=0.99, B=9, gseed=3):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["measure_only"] = True; cfg["measure_coef"] = 1.0; cfg["measure_legs"] = "critic"
            cfg["fb_forward_z"] = True; cfg["project_z"] = True
            cfg["measure_ortho_coef"] = lam
            cfg["measure_test_states"] = test_states; cfg["measure_test_rank"] = test_rank
            cfg["discount"] = gamma
            ex_obs  = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act  = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals   = jax.random.normal(jax.random.PRNGKey(gseed), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            return agent, batch, ex_obs, ex_nobs, ex_act, goals

        def legs(agent, obs, b_goals, acts, zg, params):
            # phi depends on (obs, acts, z_goals); b_goals feeds psi alone.
            _, phi, psi = agent.network.select("critic")(obs, b_goals, acts, zg, info=True, params=params)
            return phi, psi
"""


def test_rho_full_is_bit_identical_to_the_legacy_measure_loss():
    """BACKWARD COMPAT, decisive. The default ('rho','full') must reproduce the single-matrix
    _measure_terms formula exactly: offdiag and diag are the two halves of ONE B x B matrix built
    from psi(next_observations)."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("rho", "full")
        p = agent.network.params
        loss, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        phi, psi = legs(agent, obs, nobs, acts, goals, p)
        noise = jax.random.normal(jax.random.PRNGKey(2), acts.shape)
        a_next = jax.lax.stop_gradient(agent.network.select("actor")(nobs, goals, noise))
        _, tphi, tpsi = agent.network.select("target_critic")(nobs, nobs, a_next, goals, info=True)
        sqd = jnp.sqrt(50.0); B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        M  = jnp.einsum('hid,hjd->hij', phi, psi) / sqd
        tM = jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd
        diff = M - 0.99 * tM
        exp_off = 0.5 * jnp.sum((diff * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        exp_dia = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2), axis=1)
        assert jnp.allclose(info["measure_offdiag"], jnp.mean(exp_off), rtol=1e-5), (info["measure_offdiag"], jnp.mean(exp_off))
        assert jnp.allclose(info["measure_diag"],    jnp.mean(exp_dia), rtol=1e-5)
        assert "measure_qz_mean" not in info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_goals_full_tests_against_psi_of_value_goals():
    """DECISIVE for N1. offdiag is built from psi(value_goals) (the task distribution nu), while the
    Dirac term stays on the psi(next_observations) diagonal and the ortho stays on
    psi(next_observations)."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("goals", "full")
        p = agent.network.params
        loss, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        phi, psi_o = legs(agent, obs, nobs,  acts, goals, p)      # rho  leg
        _,   psi_t = legs(agent, obs, goals, acts, goals, p)      # task leg
        noise = jax.random.normal(jax.random.PRNGKey(2), acts.shape)
        a_next = jax.lax.stop_gradient(agent.network.select("actor")(nobs, goals, noise))
        _, tphi, tpsi_o = agent.network.select("target_critic")(nobs, nobs,  a_next, goals, info=True)
        _, _,    tpsi_t = agent.network.select("target_critic")(nobs, goals, a_next, goals, info=True)
        sqd = jnp.sqrt(50.0); B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        diff_t = (jnp.einsum('hid,hjd->hij', phi, psi_t) - 0.99 * jnp.einsum('hid,hjd->hij', tphi, tpsi_t)) / sqd
        exp_off = 0.5 * jnp.sum((diff_t * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        dir_i = (jnp.einsum('hid,hid->hi', phi, psi_o) - 0.99 * jnp.einsum('hid,hid->hi', tphi, tpsi_o)) / sqd
        exp_dia = -jnp.mean(dir_i, axis=1)
        Cov = jnp.einsum('hid,hjd->hij', psi_o, psi_o)
        o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_dia = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
        assert jnp.allclose(info["measure_offdiag"], jnp.mean(exp_off), rtol=1e-4), (info["measure_offdiag"], jnp.mean(exp_off))
        assert jnp.allclose(info["measure_diag"],    jnp.mean(exp_dia), rtol=1e-4)
        assert jnp.allclose(info["measure_orth"],    jnp.mean(o_off + o_dia), rtol=1e-4)
        assert not jnp.allclose(psi_t, psi_o)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_perturbing_goals_moves_offdiag_but_never_the_ortho():
    """THE INVARIANT THAT MAKES THE CELL NON-DEGENERATE. We whiten under rho and test under nu.
    If the ortho followed the test set, C_nu -> I at its optimum and the objective would silently
    collapse to canonical SF (0.120).

    NOTE: we do NOT assert the Dirac term is goal-invariant. Under fb_forward_z the forward map is
    phi(s,a,z) with z = psi(g), so perturbing the goals legitimately changes phi and hence
    dir_i = <phi_i, psi(s'_i)>. The ortho reads psi(next_observations) alone and is the only term
    that must be goal-independent. That the Dirac uses psi(s'_i) rather than psi(g_i) is asserted
    separately, by test_dirac_term_never_follows_the_test_set."""
    code = PREAMBLE + """
        a1, b1, *_ = build("goals", "full", gseed=3)
        _,  b2, *_ = build("goals", "full", gseed=77)   # different value_goals, same obs/nobs/acts
        i1 = a1.measure_loss(b1, a1.network.params, jax.random.PRNGKey(2))[1]
        i2 = a1.measure_loss(dict(b1, value_goals=b2["value_goals"], actor_goals=b2["actor_goals"]),
                             a1.network.params, jax.random.PRNGKey(2))[1]
        assert not jnp.allclose(i1["measure_offdiag"], i2["measure_offdiag"]), "offdiag must depend on the goals"
        assert jnp.allclose(i1["measure_orth"], i2["measure_orth"], rtol=1e-6), "ortho must NOT follow the test set"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_goals_task_gamma0_is_the_rank_one_projection():
    """DECISIVE for N2. At gamma=0 the rank-1 cell is
        L = 0.5*mean_i (Q_i)^2  -  mean_i <phi_i, psi(s'_i)>/sqrt(d)  +  lam*ortho
    with Q_i = <phi_i, psi(g_i)>/sqrt(d): each residual is projected onto its OWN task vector."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("goals", "task", lam=2.0, gamma=0.0)
        p = agent.network.params
        loss, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        phi, psi_o = legs(agent, obs, nobs,  acts, goals, p)
        _,   psi_t = legs(agent, obs, goals, acts, goals, p)
        sqd = jnp.sqrt(50.0); B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        q_i   = jnp.einsum('hid,hid->hi', phi, psi_t) / sqd        # [2,B]
        dir_i = jnp.einsum('hid,hid->hi', phi, psi_o) / sqd        # [2,B]
        exp_off = 0.5 * jnp.mean(q_i ** 2, axis=1)
        exp_dia = -jnp.mean(dir_i, axis=1)
        Cov = jnp.einsum('hid,hjd->hij', psi_o, psi_o)
        o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_dia = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
        assert jnp.allclose(info["measure_offdiag"], jnp.mean(exp_off), rtol=1e-4), (info["measure_offdiag"], jnp.mean(exp_off))
        assert jnp.allclose(info["measure_diag"],    jnp.mean(exp_dia), rtol=1e-4)
        assert jnp.allclose(loss, jnp.mean(exp_off + exp_dia + 2.0 * (o_off + o_dia)), rtol=1e-4)
        assert jnp.allclose(info["measure_qz_mean"], jnp.mean(q_i), rtol=1e-4)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_dirac_term_never_follows_the_test_set():
    """The Dirac term IS the reward. If it were computed from psi(g_i) instead of psi(s'_i), the
    diagonal would be Q_i - gamma*Qbar_i and the reward would vanish -- silently, and the cell's
    number would be uninterpretable. Assert the two candidates differ and that we use psi(s')."""
    code = PREAMBLE + """
        agent, batch, obs, nobs, acts, goals = build("goals", "task", gamma=0.0)
        p = agent.network.params
        _, info = agent.measure_loss(batch, p, jax.random.PRNGKey(2))
        phi, psi_o = legs(agent, obs, nobs,  acts, goals, p)
        _,   psi_t = legs(agent, obs, goals, acts, goals, p)
        sqd = jnp.sqrt(50.0)
        right = -jnp.mean(jnp.einsum('hid,hid->hi', phi, psi_o) / sqd)     # psi(s') -- correct
        wrong = -jnp.mean(jnp.einsum('hid,hid->hi', phi, psi_t) / sqd)     # psi(g)  -- the bug
        assert not jnp.allclose(right, wrong), "test vacuous: psi(s') == psi(g)"
        assert jnp.allclose(info["measure_diag"], right, rtol=1e-4), (info["measure_diag"], right, wrong)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_task_norm_guards_reject_invalid_combinations():
    """The guards encode the design so a misconfigured cell fails loudly instead of producing an
    uninterpretable number.

    1. measure_test_rank='task' => measure_test_states='goals'. The rho diagonal IS the Dirac term;
       projecting there would double-count the reward.
    2. either knob off-default => measure_only or measure_coef>0 (they only affect measure_loss).
    3. either knob off-default => measure_legs='critic' (the value-legs branch has no goal leg).
    4. either knob off-default => not sf_feature_td, and value_arch != 'simplesf' (each replaces
       the loss entirely)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        def build(**over):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg.update(over)
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        M = dict(measure_only=True, measure_coef=1.0, measure_legs="critic",
                 fb_forward_z=True, project_z=True, measure_ortho_coef=1000.0)
        bad = [
            dict(M, measure_test_rank="task", measure_test_states="rho"),   # 1
            dict(M, measure_test_states="goals", measure_only=False, measure_coef=0.0),  # 2
            dict(M, measure_test_states="goals", measure_legs="value"),     # 3
            dict(M, measure_test_states="goals", sf_feature_td=True, sf_base="frozen"),  # 4
        ]
        for over in bad:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("guard missing for " + repr(over))
        build(**M, measure_test_states="goals", measure_test_rank="full")   # N1
        build(**M, measure_test_states="goals", measure_test_rank="task")   # N2
        build(**M)                                                          # the 0.231 anchor
        build()                                                             # stock GCIQL
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
