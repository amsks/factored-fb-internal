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


def test_cell_A_gamma0_target_is_psi_of_next_state():
    """DECISIVE for cell A. sf_base_index='next' makes the immediate feature b = psi(s'), FB's
    arriving-state convention (its measure counts arrival at s'). At gamma=0 the bootstrap
    vanishes, so the TD term must equal mean((phi - psi(s'))^2) exactly -- NOT psi(s)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
        cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 2.0
        cfg["sf_base_index"] = "next"; cfg["sf_mask"] = "none"
        cfg["fb_forward_z"] = True; cfg["project_z"] = True
        cfg["discount"] = 0.0
        B = 9
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
        _, phi, _ = agent.network.select("critic")(ex_obs, goals, ex_act, goals, info=True, params=p)
        _, _, psi_ns = agent.network.select("critic")(ex_obs, ex_nobs, ex_act, goals, info=True, params=p)
        _, _, psi_s  = agent.network.select("critic")(ex_obs, ex_obs,  ex_act, goals, info=True, params=p)
        exp_td = jnp.mean((phi - psi_ns) ** 2)
        assert jnp.allclose(info["sf_loss"], exp_td, rtol=1e-5), (info["sf_loss"], exp_td)
        wrong = jnp.mean((phi - psi_s) ** 2)
        assert not jnp.allclose(exp_td, wrong), "test vacuous: psi(s) == psi(s')"
        assert jnp.allclose(loss, exp_td + 2.0 * info["psi_ortho"], rtol=1e-4)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_mask_none_ignores_the_goal_mask():
    """Cell A computes a visitation density Q = sum_t gamma^t 1[s_t=g], so the goal-dependent
    absorbing mask must be GONE (m == 1). FB's measure_loss has zero references to batch['masks'].
    Randomizing masks must not move the loss under sf_mask='none'."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def loss_for(masks):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 1.0
            cfg["sf_base_index"] = "next"; cfg["sf_mask"] = "none"
            cfg["fb_forward_z"] = True; cfg["discount"] = 0.99
            B = 9
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=masks,
                         value_goals=goals, actor_goals=goals)
            _, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
            return info["sf_loss"]
        assert jnp.allclose(loss_for(jnp.ones((9,))), loss_for(jnp.zeros((9,))), rtol=1e-6)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_cell_C_goal_mask_is_live_and_absorbs_at_the_goal():
    """Cell C keeps the absorbing mask (Q = gamma^T, a first-hitting value) with a z-conditioned
    phi. Flipping masks 1->0 must change the loss: with m=0 the bootstrap is cut."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def loss_for(masks):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 1.0
            cfg["sf_base_index"] = "current"; cfg["sf_mask"] = "goal"
            cfg["fb_forward_z"] = True; cfg["discount"] = 0.99
            B = 9
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=masks,
                         value_goals=goals, actor_goals=goals)
            _, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
            return info["sf_loss"]
        assert not jnp.allclose(loss_for(jnp.ones((9,))), loss_for(jnp.zeros((9,)))), \\
            "sf_mask='goal' must use batch['masks']"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_joint_psi_sends_td_gradient_into_psi_and_off_does_not():
    """sf_joint_psi un-stop-grads the base, so the TD residual reaches psi. Control: with
    sf_joint_psi=False and sf_reward_coef=0 (reward regression weighted to zero), psi has NO
    gradient path at all through sf_loss -- the base is stop-gradded and the actor Q-term takes
    no critic grads. Tiny ortho coef in the joint arm so the TD dominates the psi gradient."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def psi_grad_norm(joint):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
            cfg["fb_forward_z"] = True; cfg["discount"] = 0.99
            if joint:
                cfg["sf_joint_psi"] = True; cfg["sf_psi_ortho_coef"] = 1e-9
                cfg["sf_base_index"] = "next"; cfg["sf_mask"] = "none"
            else:
                cfg["sf_reward_regress"] = True; cfg["sf_reward_coef"] = 0.0
            B = 9
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            g = jax.grad(lambda p: agent.sf_loss(batch, p, jax.random.PRNGKey(2))[0])(agent.network.params)
            leaves = jax.tree_util.tree_leaves(g["modules_critic"]["psi"])
            return sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        gj, gn = psi_grad_norm(True), psi_grad_norm(False)
        assert gj > 1e-3, ("joint psi must receive TD gradient", gj)
        assert gn < 1e-8, ("non-joint psi must receive NO sf_loss gradient", gn)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_defaults_reproduce_todays_sf_loss():
    """BACKWARDS COMPAT. With every new knob at its default (joint off, base 'current', mask
    'goal', no psi ortho) sf_loss must equal the pre-existing expression. gamma=0 pins the target
    to b(s)=psi(s), the same decisive form the legacy suite uses."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        assert cfg["sf_joint_psi"] is False and cfg["sf_base_index"] == "current"
        assert cfg["sf_mask"] == "goal" and cfg["sf_psi_ortho_coef"] == 0.0
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "frozen"
        cfg["discount"] = 0.0
        B = 7
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((B,)), masks=jnp.ones((B,)),
                     value_goals=goals, actor_goals=goals)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        p = agent.network.params
        _, phi, _ = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        _, _, b = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True, params=p)
        assert jnp.allclose(loss, jnp.mean((phi - b) ** 2), rtol=1e-5)
        assert "psi_ortho" not in info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_joint_psi_guards_reject_invalid_combinations():
    """The guards encode the algebra, so a misconfigured cell fails loudly instead of producing an
    uninterpretable number -- the failure mode this program keeps repeating.

    1. joint => sf_base='learned' and NOT sf_reward_regress  (exactly one grad path into psi)
    2. joint => sf_psi_ortho_coef > 0   (else phi=0, psi=0 is a GLOBAL minimum of the TD)
    3. joint and sf_mask='goal' => fb_forward_z  (a goal-dependent absorption coefficient cannot
       sit in a goal-agnostic recursion). Scoped to joint: every legacy SF cell is agnostic+masked.
    4. NOT (sf_base_index='next' and sf_mask='goal')  (mixes the arriving-state and current-index
       conventions -- the exact class of bug fixed in E3 on 2026-07-09)
    5. sf_base='learned' => sf_reward_regress OR sf_joint_psi  (a learned psi with no signal)"""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        def build(**over):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["sf_feature_td"] = True
            cfg.update(over)
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        J = dict(sf_base="learned", sf_joint_psi=True, sf_psi_ortho_coef=1000.0,
                 fb_forward_z=True, project_z=True)
        bad = [
            dict(J, sf_base="frozen"),                                   # 1: joint on a fixed base
            dict(J, sf_reward_regress=True),                             # 1: two grad paths
            dict(J, sf_psi_ortho_coef=0.0),                              # 2: guaranteed collapse
            dict(J, sf_mask="goal", sf_base_index="current", fb_forward_z=False),  # 3
            dict(J, sf_base_index="next", sf_mask="goal"),               # 4: mixed conventions
            dict(sf_base="learned"),                                     # 5: no signal at all
        ]
        for over in bad:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("guard missing for " + repr(over))
        build(**J, sf_base_index="next", sf_mask="none")       # cell A
        build(**J, sf_base_index="current", sf_mask="goal")    # cell C
        build(sf_base="frozen")                                # legacy cell unaffected
        build(sf_base="learned", sf_reward_regress=True)       # E3 cell unaffected
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
