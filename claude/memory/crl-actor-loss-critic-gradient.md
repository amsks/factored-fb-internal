---
name: crl-actor-loss-critic-gradient
description: "OGBench single-optimizer agents — the actor's Q-term must NOT pass params=grad_params to the critic"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 52c7de9a-dc0d-44e3-9c0a-6711de243819
---

In OGBench-style agents the whole network (critic+actor+value) is ONE ModuleDict
under ONE optimizer, and `total_loss = critic_loss + actor_loss`. So in the actor
loss, the DDPG/Q-max term `-Q(s, π(s))` MUST evaluate the critic at its frozen
params — `self.network.select('critic')(...)` WITHOUT `params=grad_params`. If you
pass `params=grad_params`, `-q.mean()` backprops into the critic and the optimizer
minimizes it by INFLATING Q rather than improving the actor → critic collapses.

Symptoms of this bug (seen in crl_flowbc, fixed f7b623d 2026-05-25): q_mean explodes
(→1e6+), contrastive `logits_pos == logits_neg`, `categorical_accuracy` pinned at
1/batch (chance), value head 0 or inf, 0% eval success despite "training" losses
looking plausible. Reference: `third_party/ogbench/impls/agents/crl.py` ddpgbc actor
loss calls `critic(...)` with no grad_params; `select('actor')(..., params=grad_params)`
keeps the actor trainable.

Fast diagnosis without a full run: `jax.grad(actor_loss)(params)` and sum the
grad-norm over leaves whose path contains 'critic' — should be ~0; nonzero means the
actor loss is corrupting the critic. See [[crl-flowbc-campaign]].
