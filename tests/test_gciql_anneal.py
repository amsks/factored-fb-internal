"""Annealed SM finetuning (spec: docs/superpowers/specs/2026-07-10-annealed-sm-finetune-design.md).

Warm-start from converged bilinear GCIQL and grow the successor-measure weight lambda(t) slowly:

    lam(t) = measure_coef * clip(t / anneal_ramp, 0, 1)      # ramp, then HOLD
    L      = L_IQL + lam(t) * L_SM                            # L_IQL untouched

t is network.step, which the prep script resets to 0. Two facts make that necessary:
  * restore_agent restores network.step (the anchor's is 1_000_001), so a schedule keyed on it
    would start fully ramped and the run would silently be a constant-lambda cold start;
  * optax.adam(learning_rate=<callable>) changes the opt_state tree, so a constant-LR checkpoint
    cannot restore into a scheduled optimizer at all.

Runs in the jax venv subprocess.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

# Builds a measure-graft agent on the bilinear-d50 architecture (the E2 mix shape).
# NOTE the 8-space indent: run_jax() applies textwrap.dedent to PREAMBLE + body, so both halves
# must carry the SAME leading indent or the combined snippet raises IndentationError.
PREAMBLE = """
        import jax, jax.numpy as jnp
        from gciql_flow import GCIQLFlowAgent, get_config

        def build(step=0, lam_max=10.0, ramp=100, anneal=True, warm=0, sgphi=True):
            cfg = dict(get_config())
            cfg["encoder"] = None; cfg["frame_stack"] = None
            cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 50
            cfg["measure_only"] = False; cfg["measure_coef"] = lam_max
            cfg["measure_legs"] = "critic"; cfg["measure_ortho_coef"] = 0.0
            cfg["measure_scale_norm"] = True; cfg["measure_stopgrad_phi"] = sgphi
            cfg["anneal_measure"] = anneal; cfg["anneal_ramp"] = ramp
            cfg["anneal_lr_warmup"] = warm
            B = 12
            obs  = jax.random.normal(jax.random.PRNGKey(0), (B, 11))
            nobs = jax.random.normal(jax.random.PRNGKey(4), (B, 11))
            act  = jax.random.normal(jax.random.PRNGKey(1), (B, 5))
            goals= jax.random.normal(jax.random.PRNGKey(3), (B, 11))
            rew  = jnp.where(jax.random.uniform(jax.random.PRNGKey(5), (B,)) < 0.2, 0.0, -1.0)
            ag = GCIQLFlowAgent.create(0, obs, act, cfg)
            ag = ag.replace(network=ag.network.replace(step=step))   # TrainState is a flax struct
            batch = dict(observations=obs, next_observations=nobs, actions=act,
                         rewards=rew, masks=1.0 + rew, value_goals=goals, actor_goals=goals)
            return ag, batch
"""


def test_lambda_ramps_linearly_then_holds():
    """DECISIVE for the schedule. lam(t) = lam_max * clip(t/ramp, 0, 1): zero at t=0, half at the
    midpoint, exactly lam_max at t=ramp, and HELD at lam_max beyond it. Without the hold we would
    only ever see the transient, never the steady state at full dose."""
    code = PREAMBLE + """
        for t, expect in [(0, 0.0), (50, 5.0), (100, 10.0), (200, 10.0), (10_000, 10.0)]:
            ag, batch = build(step=t, lam_max=10.0, ramp=100)
            _, info = ag.total_loss(batch, ag.network.params, jax.random.PRNGKey(7))
            got = float(info["anneal/lambda"])
            assert abs(got - expect) < 1e-5, (t, got, expect)
            assert abs(float(info["anneal/t"]) - t) < 1e-5
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_lambda_zero_contributes_no_gradient():
    """At t=0 the SM term is multiplied by exactly 0, so the gradient of total_loss must equal the
    pure-IQL gradient. This is what makes the very first finetune step a no-op on the restored
    geometry -- and what makes the first eval a free restore check (it must read ~0.90)."""
    code = PREAMBLE + """
        ag0, batch = build(step=0, lam_max=10.0, ramp=100, anneal=True)
        agN, _     = build(step=0, lam_max=0.0,  ramp=100, anneal=False)   # pure IQL
        g0 = jax.grad(lambda p: ag0.total_loss(batch, p, jax.random.PRNGKey(7))[0])(ag0.network.params)
        gN = jax.grad(lambda p: agN.total_loss(batch, p, jax.random.PRNGKey(7))[0])(agN.network.params)
        d = jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), g0["modules_critic"], gN["modules_critic"])
        worst = max(float(x) for x in jax.tree_util.tree_leaves(d))
        assert worst < 1e-8, ("lambda(0) must contribute no gradient", worst)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_measure_loss_is_still_computed_at_lambda_zero():
    """Static shapes under jit: the measure loss must still be COMPUTED at t=0 (and logged), just
    scaled to zero. If it were skipped, the jitted graph would change shape mid-run the moment
    lambda became positive."""
    code = PREAMBLE + """
        ag, batch = build(step=0, lam_max=10.0, ramp=100)
        _, info = ag.total_loss(batch, ag.network.params, jax.random.PRNGKey(7))
        assert "measure/measure_loss" in info, sorted(info)
        assert bool(jnp.isfinite(info["measure/measure_loss"]))
        assert float(info["anneal/lambda"]) == 0.0
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_lr_warmup_schedule_is_linear_then_constant():
    """optax.linear_schedule(0, lr, N) rises linearly and HOLDS at lr past N. The warmup exists to
    protect the restored weights from a large first step. It cannot be faked by scaling the loss:
    Adam is approximately scale-invariant per parameter (m/sqrt(v)), so a global loss factor is a
    no-op. lambda works only because it changes the RELATIVE weight of L_SM against L_IQL."""
    code = """
        import optax, jax.numpy as jnp
        lr, N = 3e-4, 10
        s = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=N)
        # tolerance 1e-9, not 1e-12: optax computes in float32, so s(5) = 1.500000071e-4 rather
        # than exactly lr/2 = 1.5e-4 -- an absolute error of ~7e-12, above a 1e-12 threshold.
        for c, expect in [(0, 0.0), (5, lr/2), (10, lr), (20, lr), (10_000, lr)]:
            got = float(s(c))
            assert abs(got - expect) < 1e-9, (c, got, expect)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_scheduled_optimizer_changes_the_opt_state_tree():
    """THE FACT THAT FORCES THE PREP SCRIPT. optax.adam(constant lr) ends in EmptyState;
    optax.adam(callable lr) ends in ScaleByScheduleState(count). restore_agent deserializes the
    whole TrainState, so a checkpoint saved with a constant LR CANNOT restore into a scheduled
    optimizer. Asserted explicitly so a future optax that unifies them is caught here, not by a
    five-hour run that dies on restore."""
    code = """
        import jax, jax.numpy as jnp, optax
        p = {'w': jnp.ones((3,))}
        const = optax.adam(learning_rate=3e-4)
        sched = optax.adam(learning_rate=optax.linear_schedule(0.0, 3e-4, 10))
        ts_c = jax.tree_util.tree_structure(const.init(p))
        ts_s = jax.tree_util.tree_structure(sched.init(p))
        assert ts_c != ts_s, ("optax unified the trees; prepare_anneal_ckpt.py may be simplifiable", ts_c)
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_agent_builds_a_scheduled_tx_only_when_warmup_is_set():
    """anneal_lr_warmup=0 must reproduce today's optimizer exactly (EmptyState); >0 must produce the
    scheduled one. This is the switch prepare_anneal_ckpt.py has to mirror."""
    code = PREAMBLE + """
        import optax
        a0, _ = build(warm=0)
        a1, _ = build(warm=10)
        t0 = jax.tree_util.tree_structure(a0.network.opt_state)
        t1 = jax.tree_util.tree_structure(a1.network.opt_state)
        assert t0 != t1, "anneal_lr_warmup did not change the optimizer"
        ref = jax.tree_util.tree_structure(optax.adam(learning_rate=3e-4).init(a0.network.params))
        assert t0 == ref, "warmup=0 must reproduce today's constant-LR adam exactly"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
