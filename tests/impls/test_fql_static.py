"""FQL actor: no in-tree numerical reference (see Task 2 Step 1 investigation).

Vendored OGBench (`third_party/ogbench/impls/agents/`) has no `fql` agent — only crl,
gcbc, gciql, gcivl, hiql, qrl, sac. The only in-tree candidate reference is
`tools/wandb_mode_shim/gciql_flow.py`'s `GCIQLFlowAgent.actor_loss`, which grafts a flow
actor onto the vendored GCIQL critic. Term-by-term comparison against
`impls/actors/fql.py:37-55` found it DIVERGES from `FQLActor`, not corresponds:

- BC-flow conditioning: `fql.py` always concatenates the critic's `cvec` into the vector
  field's obs (`obs_c = concat([obs, cvec], -1)`). This is only a DEFAULT difference from
  the graft, not a structural one: the graft's default config (`actor_vf_cond='none'`,
  gciql_flow.py:1837) leaves the BC flow's obs unconditioned, but setting
  `actor_vf_cond='goal'` builds `ex_vf_obs = concat([ex_observations, ex_goals], -1)`
  (gciql_flow.py:1757), which is textually identical to our `obs_c` when the critic is
  `GCIQLCritic` (`cond_vec` returns `cond["goal"]` — impls/critics/gciql.py:85-86).
- One-step actor network: `fql.py` uses the same `ActorVectorField` module for both the
  bc-flow and the one-step actor, taking a single concatenated `obs_c` input. The graft's
  one-step actor is NOT `GCActor`-style — `GCActor` is wired only for the
  `actor_kind='ddpgbc'` fallback branch (gciql_flow.py:1712-1714). On the flow branch that
  is actually comparable to `FQLActor` (the default `actor_kind='flow'`), the one-step
  actor is `GCNoiseActor` (gciql_flow.py:1743-1749, wired into `network_info['actor']` at
  gciql_flow.py:1763), which wraps `NoiseConditionedActor`
  (imported gciql_flow.py:24 from `ogbench_flow.py`; defined ogbench_flow.py:31-56) — a
  dual-embedding (`embed_s`, `embed_z`) tanh policy over `(obs, goal, noise)`, structurally
  unlike `ActorVectorField`'s single-MLP-over-concatenated-input design.
- Distill weighting/name: `fql.py` weights the distillation term by `alpha` (multiplying
  `distill_loss`, gated by nothing) and gates a *multiplicative* `normalize_q_loss` on the
  raw `q_loss` term (off by default -> `q_loss = -q.mean()` unscaled). The graft's knob is
  `bc_coeff` (default 3.0, not `alpha` — `alpha` is a different hyperparameter used only in
  the `ddpgbc` branch), and it *unconditionally* divides the q-term by
  `stop_grad(|q|.mean() + eps)` whenever `bc_coeff > 0`, with no flag gate.
- Critic evaluation: `fql.py` evaluates Q at the *live* `params` dict passed into `loss`
  (only `stop_encoder_grad=True` is set) and clips the one-step action to [-1, 1] first.
  The graft evaluates Q at the actor's own frozen network params (no `params=grad_params`
  at all -- the whole critic is stop-gradded, not just its encoder) and does not clip
  `actor_actions` before the Q call.

Two of the above are configurable away (BC-flow conditioning, via `actor_vf_cond='goal'`)
or superficial (naming). The two ACTUALLY dispositive reasons a parity lock is impossible,
under ANY graft config, are structural:

(a) Q-loss normalisation is inseparable from distillation. gciql_flow.py:698-704 gates the
    `stop_grad(|q|.mean() + eps)` division of the q-term unconditionally on `bc_coeff > 0`
    (there is no separate `normalize_q_loss`-style flag anywhere in the file — confirmed by
    its absence from the module). So "distillation on, q-loss unnormalised" —
    `FQLActor`'s default combination (`normalize_q_loss=False` with `alpha=10.0` in the test
    config below) — is unreachable by any setting of the graft's config.
(b) The param trees cannot be transplanted between modules. The graft's `GCVectorField`
    (wrapping `VectorField`, ogbench_flow.py:14-28) and `GCNoiseActor` (wrapping
    `NoiseConditionedActor`, ogbench_flow.py:31-56) use a uniform-width `hidden_dim` repeated
    `hidden_layers` times, the exact (erf) GELU (`nn.gelu(..., approximate=False)`,
    ogbench_flow.py:25,27), and explicit Dense names (`l0`..`l{N}`, `out`,
    `{prefix}_l0`..`{prefix}_lout`, `policy_l{i}`, `policy_out`). Our `ActorVectorField`
    (impls/utils/networks.py:317-333) is an `MLP` over a `Sequence[int]` of (possibly
    non-uniform) `hidden_dims`, uses Flax's default approximate GELU (`nn.gelu` with no
    `approximate=False` override, impls/utils/networks.py:210-225), and auto-names its Dense
    layers (`Dense_0`, `Dense_1`, ...). The two param pytrees have different shapes, names,
    and nonlinearities and cannot be swapped in either direction.

None of the three loss terms (bc-flow loss, distill loss, q-loss) are computed the same way,
and reasons (a) and (b) above make the mismatch structural rather than a matter of config, so
a `1e-10` parity lock against the graft is not possible and would misrepresent `FQLActor`'s
provenance, which is the external PSMFlows `fql.py` (not in this repo).

Per the Task 2 brief's Step 5 fallback, this file locks the loss *decomposition* instead:
actor_loss == bc_flow_loss + alpha * distill_loss + q_loss, self-consistently, without an
external reference. It also independently recomputes the bc-flow term from the
flow-matching definition (see `test_fql_bc_flow_loss_matches_first_principles` below), so
a sign error in the velocity target is caught even though it would not perturb the
info-dict's self-consistency.
"""
import jax.numpy as jnp
import numpy as np
from impls.actors.fql import FQLActor
from impls.utils.networks import ActorVectorField


def test_fql_loss_decomposes_as_documented():
    """No in-tree reference exists for FQL (see plan Task 2 Step 1). Lock the loss
    decomposition instead: actor_loss == bc_flow_loss + alpha*distill_loss + q_loss."""
    from impls.agents import Agent
    cfg = dict(value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9,
               tau=0.005, lr=3e-4, lr_actor=3e-4, gc_negative=True,
               actor=dict(hidden_dims=[32, 32], alpha=10.0, flow_steps=5,
                          normalize_q_loss=False, layer_norm=False))
    agent = Agent.create("gciql", "fql", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), cfg)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((16, 8)).astype(np.float32)
    batch = {"observations": rng.standard_normal((16, 8)).astype(np.float32),
             "actions": np.tanh(rng.standard_normal((16, 5))).astype(np.float32),
             "next_observations": rng.standard_normal((16, 8)).astype(np.float32),
             "value_goals": g, "actor_goals": g,
             "rewards": rng.integers(0, 2, size=16).astype(np.float32) - 1.0,
             "masks": rng.integers(0, 2, size=16).astype(np.float32)}
    _, info = agent.update(batch)
    recon = info["bc_flow_loss"] + 10.0 * info["distill_loss"] + info["q_loss"]
    np.testing.assert_allclose(np.asarray(info["actor_loss"]), np.asarray(recon), rtol=1e-6)


def test_fql_bc_flow_loss_matches_first_principles():
    """Independent check of the bc-flow term, with verification value the decomposition
    test above lacks: it hand-computes `xt`, `vel`, and the network's prediction itself from
    the flow-matching definition (xt = (1-t)*x0 + t*action; vel = action - x0;
    loss = mean((vf(obs_c, xt, t) - vel)**2)), calling the `actor_bc_flow` network module
    directly, and never reads `vel`/`xt` back out of FQLActor's own `info` dict. A sign flip
    in `fql.py`'s `vel = action - x0` (e.g. `action + x0`) changes the network's *target*
    without changing its *prediction* (xt is unaffected by the flip), so this test's
    independently-computed `bc_flow_loss` would diverge from FQLActor's reported one --
    unlike the decomposition test, which only re-sums `info["bc_flow_loss"]` and would still
    pass because both sides read the same (corrupted) info-dict entry.

    Uses `Agent.compute_static` (cond/boot passed in explicitly, no internal RNG draw) so the
    test controls x0/t/action/goal exactly and can reproduce the same `obs_c`/`xt` used inside
    FQLActor.loss without guessing what the agent's internal `sample_bootstrap` RNG drew.
    """
    from impls.agents import Agent
    cfg = dict(value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9,
               tau=0.005, lr=3e-4, lr_actor=3e-4, gc_negative=True,
               actor=dict(hidden_dims=[32, 32], alpha=10.0, flow_steps=5,
                          normalize_q_loss=False, layer_norm=False))
    agent = Agent.create("gciql", "fql", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), cfg)
    assert isinstance(agent.actor, FQLActor)
    assert isinstance(agent.actor.actor_bc_flow, ActorVectorField)

    rng = np.random.default_rng(1)
    B = 16
    obs = rng.standard_normal((B, 8)).astype(np.float32)
    action = np.tanh(rng.standard_normal((B, 5))).astype(np.float32)
    goal = rng.standard_normal((B, 8)).astype(np.float32)
    x0 = rng.standard_normal((B, 5)).astype(np.float32)
    t = rng.uniform(0.0, 1.0, size=(B, 1)).astype(np.float32)
    noise = rng.standard_normal((B, 5)).astype(np.float32)

    batch = {"observations": obs, "actions": action,
             "next_observations": rng.standard_normal((B, 8)).astype(np.float32),
             "rewards": rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
             "masks": rng.integers(0, 2, size=B).astype(np.float32)}
    cond = {"value_goal": jnp.asarray(goal), "goal": jnp.asarray(goal)}
    boot = {"flow_x0": jnp.asarray(x0), "flow_t": jnp.asarray(t), "actor_noise": jnp.asarray(noise)}

    info = agent.compute_static(batch, cond, boot)

    # First-principles recomputation (flow-matching definition), independent of fql.py:
    # GCIQLCritic.cond_vec(cond) == cond["goal"], so obs_c below matches FQLActor.loss's own.
    obs_c = jnp.asarray(np.concatenate([obs, goal], axis=-1))
    xt_expected = (1 - t) * x0 + t * action
    vel_expected = action - x0
    bc_flow_params = agent.net["actor_bc_flow"].params
    pred_expected = agent.actor.actor_bc_flow.apply(
        {"params": bc_flow_params}, obs_c, jnp.asarray(xt_expected), jnp.asarray(t))
    bc_flow_loss_expected = jnp.mean((pred_expected - jnp.asarray(vel_expected)) ** 2)

    np.testing.assert_allclose(np.asarray(info["bc_flow_loss"]),
                                np.asarray(bc_flow_loss_expected), rtol=1e-6)
