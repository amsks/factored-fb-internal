"""FB-fidelity knob: measure_z_random_frac — FB's mixed z-sampling for the measure graft.

The audited fidelity gap between the JAX measure graft (C1 25.5 / C4 16.5) and torch
plain-FB (48): the z-indexed forward map F(s,a,z) is only ever trained at z=psi(goal-mix),
never at RANDOM task directions. FB (metamotivo sample_mixed_z) trains with
z ~ 50% N(0,I)->sqrt(d)-sphere, 50% B(goals)->sqrt(d)-sphere — covering z-space so that
inference-time z (a weighted psi combination, generally OFF the psi(single-goal) manifold)
lands on trained ground.

  measure_z_random_frac (default 0.0): per-sample Bernoulli fraction of the measure-loss
  task vectors z_i replaced by sqrt(d)-sphere gaussian directions. Same z on the online
  and target side of the TD (FB backup is at fixed z). Under actor_cond='z' the bootstrap
  action a+ ~ pi(s+, z_i) uses the SAME mixed z (z-consistent policy), not psi(vgoals).

  GCFBValue gains an optional z_vec kwarg: use the given task vector directly instead of
  mapping z_goals through psi (needed to inject non-goal z). z_vec=None => bit-exact
  current behavior.

(The audit's OTHER caveat — termination masking in the measure TD — was traced and
dropped: tM is stop-gradded, so absorbing-goal masking only shifts the diagonal term by
a gradient-constant; off-diagonal absorption is measure-zero in continuous state.)

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_gcfbvalue_z_vec_overrides_z_goals():
    """GCFBValue(..., z_vec=x) must use x as the task vector directly: output equals the
    stock z_goals path when x is the manually-computed stop-gradded (head-mean, projected)
    psi(z_goals), and differs when x is a different vector."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 7
        cfg["fb_forward_z"] = True; cfg["measure_only"] = True; cfg["measure_coef"] = 1.0
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        zg = ex_obs[::-1]
        # manual z = head-mean psi(z_goals) (module: stop-grad; project_z off here)
        _v, _phi, psi = agent.network.select("critic")(zg, zg, ex_act, zg, info=True)
        z_manual = psi.mean(axis=0)
        q_stock = agent.network.select("critic")(ex_obs, ex_obs, ex_act, zg)
        q_zvec = agent.network.select("critic")(ex_obs, ex_obs, ex_act, zg, z_vec=z_manual)
        assert jnp.allclose(q_stock, q_zvec, atol=1e-5), (q_stock, q_zvec)
        q_other = agent.network.select("critic")(ex_obs, ex_obs, ex_act, zg,
                                                 z_vec=z_manual + 1.0)
        assert not jnp.allclose(q_stock, q_other, atol=1e-3)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_measure_z_random_frac_changes_loss_and_stays_finite():
    """measure_z_random_frac=1.0 must actually change the measure loss vs 0.0 (same params,
    same batch, same rng) — the knob is load-bearing, not silently ignored — and training
    stays finite. frac requires fb_forward_z (z-indexed forward map)."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 7
        cfg["fb_forward_z"] = True; cfg["measure_only"] = True; cfg["measure_coef"] = 1.0
        cfg["measure_z_random_frac"] = 0.0
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent0 = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((6,)), masks=jnp.ones((6,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        rng = jax.random.PRNGKey(7)
        l0, _ = agent0.measure_loss(batch, agent0.network.params, rng)
        cfg1 = dict(cfg); cfg1["measure_z_random_frac"] = 1.0
        agent1 = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg1)  # same seed => same params
        l1, _ = agent1.measure_loss(batch, agent1.network.params, rng)
        assert jnp.isfinite(l0) and jnp.isfinite(l1), (l0, l1)
        assert not jnp.allclose(l0, l1, atol=1e-6), (l0, l1)
        for _ in range(3):
            agent1, info = agent1.update(batch)
        assert jnp.isfinite(info["measure/measure_loss"]), info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_measure_zmix_actor_cond_z_uses_mixed_z_and_trains_finite():
    """The full-fidelity FB-rebuild stack (measure_only + fb_forward_z + project_z +
    actor_cond='z' + actor_z_from_critic + z_random_frac=0.5) must train finite: under
    actor_cond='z' the bootstrap action is conditioned on the SAME mixed z as the TD
    (z-consistent policy), which must compose with the flow actor's d-dim goal leg."""
    code = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 7
        cfg["fb_forward_z"] = True; cfg["project_z"] = True
        cfg["measure_only"] = True; cfg["measure_coef"] = 1.0
        cfg["measure_ortho_coef"] = 1.0; cfg["measure_z_random_frac"] = 0.5
        cfg["actor_cond"] = "z"; cfg["actor_z_from_critic"] = True
        ex_obs = jax.random.normal(jax.random.PRNGKey(0), (6, 11))
        ex_act = jax.random.normal(jax.random.PRNGKey(1), (6, 5))
        agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = dict(observations=ex_obs, next_observations=ex_obs, actions=ex_act,
                     rewards=jnp.zeros((6,)), masks=jnp.ones((6,)),
                     value_goals=ex_obs, actor_goals=ex_obs)
        for _ in range(3):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["measure/measure_loss"]), info
        assert jnp.isfinite(info["actor/actor_loss"]), info
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
