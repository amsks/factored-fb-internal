"""FQL actor: no in-tree numerical reference (see Task 2 Step 1 investigation).

Vendored OGBench (`third_party/ogbench/impls/agents/`) has no `fql` agent — only crl,
gcbc, gciql, gcivl, hiql, qrl, sac. The only in-tree candidate reference is
`tools/wandb_mode_shim/gciql_flow.py`'s `GCIQLFlowAgent.actor_loss`, which grafts a flow
actor onto the vendored GCIQL critic. Term-by-term comparison against
`impls/actors/fql.py:37-55` found it DIVERGES from `FQLActor`, not corresponds:

- BC-flow conditioning: `fql.py` always concatenates the critic's `cvec` into the vector
  field's obs (`obs_c = concat([obs, cvec], -1)`). The graft's default config
  (`actor_vf_cond='none'`) leaves the BC flow's obs completely unconditioned (goal-agnostic).
- One-step actor network: `fql.py` uses the same `ActorVectorField` module for both the
  bc-flow and the one-step actor, taking a single concatenated `obs_c` input. The graft's
  one-step actor is `network.select('actor')`, a `GCActor`-style module taking `(obs, goal,
  noise)` as three separate arguments, conditioned on the raw goal (`actor_cond='goal'`
  default) rather than a `cvec`.
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

None of the three terms (bc-flow loss, distill loss, q-loss) are computed the same way, so
a `1e-10` parity lock against the graft is not possible and would misrepresent
`FQLActor`'s provenance, which is the external PSMFlows `fql.py` (not in this repo).

Per the Task 2 brief's Step 5 fallback, this file locks the loss *decomposition* instead:
actor_loss == bc_flow_loss + alpha * distill_loss + q_loss, self-consistently, without an
external reference.
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
