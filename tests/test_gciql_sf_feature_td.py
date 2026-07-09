"""sf_feature_td: successor-FEATURE vector TD on phi with an identity base leg b(s)=s.

Q(s,a,g)=<phi(s,a), g>/sqrt(d), psi=b=identity (param-free, NOT trained). phi (the
successor-feature vector) is trained ONLY by a per-sample vector TD:

    phi(s,a) <- b(s) + gamma * mask * phi_target(s',a'),   a' ~ pi(s',g),   b(s)=s

This is the SF objective (raymondchua/simple_successor_features), NOT the successor-MEASURE
loss (the B x B density matrix M[i,j]=<phi_i, psi_j>). The decisive test is gamma=0: the
target collapses to b(s)=obs, so L_SF == mean((phi - obs)^2) -- a per-sample regression,
impossible for a measure matrix. Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_sf_identity_b_leg_and_phi_dim():
    """The critic's base/goal leg is identity(g) and phi's output dim == obs/goal dim so
    Q=<phi(s,a), g> is well-formed and finite."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        goals = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
        v, phi, psi = agent.network.select("critic")(ex_obs, goals, ex_act, info=True)
        assert psi.shape == (2, 4, 11), psi.shape
        assert jnp.allclose(psi[0], goals) and jnp.allclose(psi[1], goals), "base leg must be identity(g)"
        assert phi.shape[-1] == 11, phi.shape        # phi dim == obs dim
        assert v.shape == (2, 4) and jnp.all(jnp.isfinite(v)), v
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_loss_is_feature_td_not_measure_gamma0():
    """DECISIVE: with gamma=0 the SF target is b(s)=obs, so L_SF == mean((phi - obs)^2)
    (a per-sample vector regression). A measure loss could never equal this."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True; cfg["discount"] = 0.0
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((6,)), masks=jnp.ones((6,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        _, phi, _ = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True,
                                                   params=agent.network.params)
        expected = jnp.mean((phi - ex_obs[None]) ** 2)     # target = b(s)=obs at gamma=0
        assert jnp.allclose(loss, expected, atol=1e-5), (float(loss), float(expected))
        assert "sf_loss" in info, list(info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_base_leg_has_no_params():
    """The identity base leg carries no learnable params; only phi is learned."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        flat = jax.tree_util.tree_flatten_with_path(agent.network.params["modules_critic"])[0]
        paths = [".".join(str(getattr(k, "key", k)) for k in p) for p, _ in flat]
        assert any("phi" in s for s in paths), paths[:8]
        assert not any("psi" in s for s in paths), [p for p in paths if "psi" in p]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_frozen_base_d_free_and_psi_stays_frozen():
    """Frozen-random base (sf_base='frozen'): d can differ from obs dim (e.g. 50), and the
    psi leg's params are UNCHANGED after training (frozen random encoder -> no collapse)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50   # != obs dim 11
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "frozen"
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        def psi_params(a):
            flat = jax.tree_util.tree_flatten_with_path(a.network.params["modules_critic"])[0]
            return {".".join(str(getattr(k,'key',k)) for k in p): v for p, v in flat if "psi" in
                    ".".join(str(getattr(k,'key',k)) for k in p)}
        before = psi_params(agent)
        assert before, "expected a learnable-shaped psi leg to freeze"
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        after = psi_params(agent)
        for k in before:
            assert jnp.allclose(before[k], after[k]), f"psi param {k} changed -> not frozen"
        assert "sf/sf_loss" in info and jnp.isfinite(info["sf/sf_loss"]), list(info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_frozen_base_feature_is_psi_of_state_gamma0():
    """Frozen base semantic: with gamma=0 the SF target is b(s)=psi(s) (the frozen features
    of the current state), so L_SF == mean((phi - psi(obs))^2) at d=50."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "frozen"; cfg["discount"] = 0.0
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((6,)), masks=jnp.ones((6,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        _, phi, _ = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True,
                                                   params=agent.network.params)
        _, _, psi_s = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True,
                                                    params=agent.network.params)   # b(s)=psi(obs)
        expected = jnp.mean((phi - psi_s) ** 2)
        assert jnp.allclose(loss, expected, atol=1e-5), (float(loss), float(expected))
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_task_conditioned_phi_depends_on_goal():
    """With fb_forward_z, the SF forward map is task-conditioned phi([s,a,z=psi(g)]) (à la the
    reference SF net input cat([obs, task])): phi must DIFFER for different goals (same s,a)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "identity"; cfg["fb_forward_z"] = True
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        gfix = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
        gA = jax.random.normal(jax.random.PRNGKey(2), (4, 11))
        gB = jax.random.normal(jax.random.PRNGKey(3), (4, 11))
        # critic(obs, b_goals, actions, z_goals, info): b_goals fixed, vary z_goals -> phi must change
        _, phiA, _ = agent.network.select("critic")(ex_obs, gfix, ex_act, gA, info=True)
        _, phiB, _ = agent.network.select("critic")(ex_obs, gfix, ex_act, gB, info=True)
        assert not jnp.allclose(phiA, phiB), "phi must depend on z_goals (task-conditioned)"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_task_conditioned_trains_finite():
    """sf_feature_td + fb_forward_z (task-conditioned phi) trains finite for both bases."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        for base, d in [("identity", 11), ("frozen", 50)]:
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = d
            cfg["sf_feature_td"] = True; cfg["sf_base"] = base; cfg["fb_forward_z"] = True
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                         rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                         value_goals=ex_obs, actor_goals=ex_obs)
            for _ in range(5):
                agent, info = agent.update(batch)
            assert jnp.isfinite(info["sf/sf_loss"]), (base, info["sf/sf_loss"])
            assert jnp.isfinite(info["actor/actor_loss"]), (base, info["actor/actor_loss"])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_ddpgbc_actor_with_conditioned_critic_trains_finite():
    """actor_kind='ddpgbc' (stock GCIQL gaussian actor, no-flow-actor control) must compose with
    the TASK-CONDITIONED SF critic (fb_forward_z=True): the ddpgbc Q-term has to pass the z=goal
    leg to the critic (which expects (obs, goals, actions, z_goals)), else the 3-arg stock call
    crashes. Training stays finite for both bases."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        for base, d in [("identity", 11), ("frozen", 50)]:
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = d
            cfg["sf_feature_td"] = True; cfg["sf_base"] = base; cfg["fb_forward_z"] = True
            cfg["actor_kind"] = "ddpgbc"
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                         rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                         value_goals=ex_obs, actor_goals=ex_obs)
            for _ in range(5):
                agent, info = agent.update(batch)
            assert jnp.isfinite(info["sf/sf_loss"]), (base, info["sf/sf_loss"])
            assert jnp.isfinite(info["actor/actor_loss"]), (base, info["actor/actor_loss"])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_ortho_regularizer_adds_finite_term():
    """sf_ortho_coef>0 adds an orthonormality penalty ||E[phi phi^T]-I|| on the SF vector phi
    (the sf_laplacian-style anti-collapse aux reg) to the task-conditioned SF loss: it must be
    reported (sf/orth) and training stays finite. With coef=0 (default) no orth term appears."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        # default: no ortho term
        cfg0 = dict(get_config()); cfg0["encoder"]=None; cfg0["frame_stack"]=None
        cfg0["value_arch"]="bilinear"; cfg0["value_latent_dim"]=11
        cfg0["sf_feature_td"]=True; cfg0["sf_base"]="identity"; cfg0["fb_forward_z"]=True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        a0 = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg0)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        a0, i0 = a0.update(batch)
        assert "sf/orth" not in i0, list(i0)
        # coef=10000: ortho term present + finite training
        cfg = dict(cfg0); cfg["sf_ortho_coef"]=10000.0
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert "sf/orth" in info, list(info)
        assert jnp.isfinite(info["sf/orth"]) and jnp.isfinite(info["sf/sf_loss"]), info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_ortho_without_conditioning_trains_finite():
    """The ortho aux reg composes with the TASK-AGNOSTIC SF (fb_forward_z=False) too -- both
    bases train finite with sf_ortho_coef=10000 and no task-conditioning."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        for base, d in [("identity", 11), ("frozen", 50)]:
            cfg = dict(get_config()); cfg["encoder"]=None; cfg["frame_stack"]=None
            cfg["value_arch"]="bilinear"; cfg["value_latent_dim"]=d
            cfg["sf_feature_td"]=True; cfg["sf_base"]=base
            cfg["fb_forward_z"]=False; cfg["sf_ortho_coef"]=10000.0   # AGNOSTIC + ortho
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                         rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                         value_goals=ex_obs, actor_goals=ex_obs)
            for _ in range(5):
                agent, info = agent.update(batch)
            assert "sf/orth" in info, (base, list(info))
            assert jnp.isfinite(info["sf/orth"]) and jnp.isfinite(info["sf/sf_loss"]), (base, info)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_learned_base_reward_regression_semantics():
    """E3 (corrected SF cell): sf_base='learned' + sf_reward_regress grounds the base in the
    ACTUAL reward -- a bilinear reward model r_hat(s',g)=<psi(s'),psi(g)>/sqrt(d) trained by
    regression on batch rewards. DECISIVE at gamma=0: total SF loss must equal
    mean((phi - psi(s))^2) + coef * mean((r_hat - r)^2), both terms computed by hand."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"
        cfg["sf_reward_regress"] = True; cfg["sf_reward_coef"] = 2.0
        cfg["discount"] = 0.0
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_nobs = jax.random.normal(jax.random.PRNGKey(3), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        goals = jax.random.normal(jax.random.PRNGKey(4), (6, 11))
        rewards = jax.random.normal(jax.random.PRNGKey(5), (6,))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                     rewards=rewards, masks=jnp.ones((6,)),
                     value_goals=goals, actor_goals=goals)
        loss, info = agent.sf_loss(batch, agent.network.params, jax.random.PRNGKey(2))
        p = agent.network.params
        _, phi, _ = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        _, _, psi_s = agent.network.select("critic")(ex_obs, ex_obs, ex_act, info=True, params=p)
        _, _, psi_ns = agent.network.select("critic")(ex_nobs, ex_nobs, ex_act, info=True, params=p)
        _, _, psi_g = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        exp_td = jnp.mean((phi - psi_s) ** 2)                       # target = b(s)=psi(s) at gamma=0
        r_hat = (psi_ns * psi_g).sum(-1) / jnp.sqrt(50.0)
        exp_rew = jnp.mean((r_hat - rewards[None]) ** 2)
        assert jnp.allclose(loss, exp_td + 2.0 * exp_rew, atol=1e-5), \\
            (float(loss), float(exp_td), float(exp_rew))
        assert "sf_reward_loss" in info and jnp.allclose(info["sf_reward_loss"], exp_rew, atol=1e-5)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_learned_base_psi_trains():
    """With sf_reward_regress the psi leg is LEARNED: its params must CHANGE under training
    (the reward regression is its grad path) -- the exact opposite of the frozen-base test --
    and training stays finite (sf/sf_loss, sf/sf_reward_loss, actor)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"; cfg["sf_reward_regress"] = True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        def psi_params(a):
            flat = jax.tree_util.tree_flatten_with_path(a.network.params["modules_critic"])[0]
            return {".".join(str(getattr(k,'key',k)) for k in p): v for p, v in flat if "psi" in
                    ".".join(str(getattr(k,'key',k)) for k in p)}
        before = psi_params(agent)
        assert before, "expected a learnable psi leg"
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=-jnp.ones((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        after = psi_params(agent)
        changed = any(not jnp.allclose(before[k], after[k]) for k in before)
        assert changed, "psi params never changed -> reward regression not training the base"
        assert jnp.isfinite(info["sf/sf_loss"]) and jnp.isfinite(info["sf/sf_reward_loss"]), info
        assert jnp.isfinite(info["actor/actor_loss"]), info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_reward_regress_config_guards():
    """sf_reward_regress on a FIXED base is the known confound (a linear-in-s reward proxy
    cannot express the goal indicator) -> create() must refuse it. Symmetrically,
    sf_base='learned' without sf_reward_regress leaves psi with no training signal -> refuse."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        def make(base, regress, d):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = d
            cfg["sf_feature_td"] = True; cfg["sf_base"] = base; cfg["sf_reward_regress"] = regress
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        for base, d in [("frozen", 50), ("identity", 11)]:
            try:
                make(base, True, d); raise SystemExit(f"regress on {base} base must be refused")
            except AssertionError:
                pass
        try:
            make("learned", False, 50); raise SystemExit("learned base without regress must be refused")
        except AssertionError:
            pass
        make("learned", True, 50)   # the valid E3 combination builds
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_learned_base_task_conditioned_trains_finite():
    """sf_base='learned' composes with fb_forward_z (task-conditioned phi([s,a,z=psi(g)]));
    the z fed to phi is stop-gradded so the TD cannot train psi through phi's input."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
        cfg["sf_feature_td"] = True; cfg["sf_base"] = "learned"; cfg["sf_reward_regress"] = True
        cfg["fb_forward_z"] = True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=-jnp.ones((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["sf/sf_loss"]) and jnp.isfinite(info["sf/sf_reward_loss"]), info
        assert jnp.isfinite(info["actor/actor_loss"]), info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_sf_trains_finite():
    """Full agent with sf_feature_td trains finite; total_loss reports sf/sf_loss (the SF
    vector TD) and NOT measure/measure_loss, and the flow actor loss stays finite."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 11
        cfg["sf_feature_td"] = True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (8, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (8, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert "sf/sf_loss" in info, list(info)
        assert not any(k.startswith("measure/") for k in info), list(info)
        assert jnp.isfinite(info["sf/sf_loss"]), info["sf/sf_loss"]
        assert jnp.isfinite(info["actor/actor_loss"]), info["actor/actor_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
