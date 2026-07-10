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


def test_grounded_psi_gets_gradient_only_from_the_reward_loss():
    """DECISIVE for rung 3. With simplesf_ground_psi, psi's ONLY gradient path is L_rew: the value
    and critic losses see a stop-gradded z, and the trunk input is stop-gradded in both rungs."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def psi_grad(ground, which):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
            cfg["simplesf_ground_psi"] = ground
            B = 8
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            def f(p):
                if which == "td":
                    return agent.value_loss(batch, p)[0] + agent.critic_loss(batch, p)[0]
                return agent.simplesf_reward_loss(batch, p)[0]
            g = jax.grad(f)(agent.network.params)
            leaves = jax.tree_util.tree_leaves(g["modules_critic"]["psi"])
            return sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        assert psi_grad(True, "td")  < 1e-8, "grounded: value+critic must NOT train psi"
        assert psi_grad(True, "rew") > 1e-3, "grounded: L_rew must train psi"
        assert psi_grad(False, "td") > 1e-3, "ungrounded: value+critic must train psi"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_trunk_receives_td_gradient_in_both_rungs():
    """b = h/||h|| has no parameters of its own; its ONLY training signal is the TD gradient
    flowing into the trunk. If the trunk froze, b would be a fixed random basis and we would have
    rebuilt the frozen-basis confound (psi_base frozen = 0.036)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        def trunk_grad(ground):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
            cfg["simplesf_ground_psi"] = ground
            B = 8
            ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                         rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                         value_goals=goals, actor_goals=goals)
            g = jax.grad(lambda p: agent.critic_loss(batch, p)[0])(agent.network.params)
            leaves = jax.tree_util.tree_leaves(g["modules_critic"]["trunk"])
            return sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        assert trunk_grad(True)  > 1e-3, "grounded: trunk must still get TD gradient"
        assert trunk_grad(False) > 1e-3, "ungrounded: trunk must get TD gradient"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_reward_loss_is_exact_and_uses_the_current_index():
    """L_rew = mean((<sg[b], psi(g)>/sqrt(d) - batch['rewards'])^2), summed over the value net and
    the critic. CURRENT index: GCDataset ties rewards AND masks to idxs == value_goal_idxs
    (datasets.py:242-244; rewards = -masks exactly). The reference's (s',a') is its own convention;
    mixing the two is the class of bug fixed in E3 on 2026-07-09."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        cfg["simplesf_ground_psi"] = True
        B = 9
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        rew = jnp.where(jax.random.uniform(jax.random.PRNGKey(5), (B,)) < 0.2, 0.0, -1.0)
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_nobs, actions=ex_act,
                     rewards=rew, masks=1.0 + rew, value_goals=goals, actor_goals=goals)
        loss, info = agent.simplesf_reward_loss(batch, agent.network.params)
        p = agent.network.params
        _, _, zq, bq = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        _, _, zv, bv = agent.network.select("value")(ex_obs, goals, info=True, params=p)
        rq = (bq * zq).sum(-1) / jnp.sqrt(50.0)
        rv = (bv * zv).sum(-1) / jnp.sqrt(50.0)
        exp = jnp.mean((rq - rew) ** 2) + jnp.mean((rv - rew) ** 2)
        assert jnp.allclose(loss, exp, rtol=1e-5), (loss, exp)
        assert jnp.allclose(info["b_norm_mean"], 1.0, atol=1e-4), info["b_norm_mean"]
        # b is evaluated at the CURRENT observation, not next_observations
        _, _, _, b_next = agent.network.select("critic")(ex_nobs, goals, ex_act, info=True, params=p)
        wrong = jnp.mean(((b_next * zq).sum(-1) / jnp.sqrt(50.0) - rew) ** 2) + jnp.mean((rv - rew) ** 2)
        assert not jnp.allclose(loss, wrong), "test vacuous: b(s) == b(s')"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_reward_model_has_no_psi_zero_saddle():
    """The asymmetric model <sg[b], psi(g)> has a NONZERO gradient at psi = 0, because b is a
    separate object. Contrast E3's symmetric <psi(s), psi(g)>, whose gradient is proportional to
    psi and therefore vanishes there -- the saddle that collapsed E3 v1 and forced
    sf_reward_shift. project_z is OFF here: the sphere projection is ill-conditioned at z = 0."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "simplesf"; cfg["value_latent_dim"] = 50
        cfg["simplesf_ground_psi"] = True; cfg["project_z"] = False
        B = 8
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
        goals = jax.random.normal(jax.random.PRNGKey(3), (B, 11))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=-jnp.ones((B,)), masks=jnp.ones((B,)),
                     value_goals=goals, actor_goals=goals)
        # Zero the critic's psi OUTPUT Dense layer -> z == 0 exactly. Zero ONLY the final Dense:
        # zeroing the whole subtree would also zero that layer's INPUT, making dz/dW vanish and
        # the test vacuous. And note sorted(keys)[-1] is 'LayerNorm_2', not 'Dense_3' --
        # alphabetical order puts LayerNorm after Dense. Select the highest-numbered Dense.
        p = jax.tree_util.tree_map(lambda x: x, agent.network.params)   # rebuilds dicts: safe to mutate
        psi_tree = p["modules_critic"]["psi"]
        denses = [k for k in psi_tree if k.startswith("Dense_")]
        last = max(denses, key=lambda k: int(k.split("_")[1]))
        assert last == "Dense_3", (last, sorted(psi_tree.keys()))
        psi_tree[last] = jax.tree_util.tree_map(jnp.zeros_like, psi_tree[last])
        _, _, z0, _ = agent.network.select("critic")(ex_obs, goals, ex_act, info=True, params=p)
        assert jnp.allclose(z0, 0.0), "setup failed: z is not zero"
        g = jax.grad(lambda q: agent.simplesf_reward_loss(batch, q)[0])(p)
        leaves = jax.tree_util.tree_leaves(g["modules_critic"]["psi"])
        gn = sum(float(jnp.sum(jnp.abs(l))) for l in leaves)
        assert gn > 1e-3, ("psi=0 must NOT be a stationary point of the asymmetric model", gn)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_simplesf_guards_reject_invalid_combinations():
    """Guards encode the design so a misconfigured cell fails loudly instead of producing an
    uninterpretable number.

    1. simplesf_ground_psi => value_arch == 'simplesf'
    2. value_arch=='simplesf' excludes every knob that selects its own critic class or loss
       (sf_feature_td, measure_only, measure_coef>0, fb_forward_z, fwd_two_tower, isfb,
       sf_joint_psi) -- they would silently override the architecture.
    3. value_arch=='simplesf' => psi_base == 'learned' (psi_base dispatches GCBilinearValue).
    4. value_arch=='simplesf' => actor_cond == 'goal' (the flow actor sees the raw goal)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        def build(**over):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_latent_dim"] = 50
            cfg.update(over)
            return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        S = dict(value_arch="simplesf")
        bad = [
            dict(simplesf_ground_psi=True, value_arch="bilinear"),   # 1
            dict(S, sf_feature_td=True, sf_base="frozen"),           # 2
            dict(S, measure_only=True),                              # 2
            dict(S, measure_coef=1.0),                               # 2
            dict(S, fb_forward_z=True),                              # 2
            dict(S, fwd_two_tower=True),                             # 2
            dict(S, sf_joint_psi=True),                              # 2
            dict(S, psi_base="frozen"),                              # 3
            dict(S, actor_cond="z"),                                 # 4
        ]
        for over in bad:
            try:
                build(**over)
            except AssertionError:
                continue
            raise SystemExit("guard missing for " + repr(over))
        build(**S)                                       # rung 2: simplesf-arch
        build(**S, simplesf_ground_psi=True)             # rung 3: simplesf-grounded
        build(value_arch="bilinear")                     # legacy cells unaffected
        build()                                          # mlp default unaffected
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
