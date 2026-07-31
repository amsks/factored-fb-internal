"""CRLCritic — Contrastive RL critic on the factored seam.

Contrastive (InfoNCE / sigmoid-BCE) value learning on a bilinear head:
    V(s, g)   = <phi(s),    psi(g)> / sqrt(d)
    Q(s,a,g)  = <phi(s,a),  psi(g)> / sqrt(d)
trained so the positive future-goal pair (diagonal of the B×B logit matrix) scores high vs
the batch negatives. Math transcribed from OGBench crl.py::contrastive_loss (continuous).

PROVIDES_V (so AWR composes). NO target networks — the contrastive objective is not
bootstrapped, so TARGETS is empty. Same GCBilinearValue head as iql_bilinear; only the loss
differs (contrastive vs expectile-TD).
"""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import optax

from impls.utils.networks import GCBilinearValue


@dataclasses.dataclass(frozen=True)
class CRLCritic:
    value: Any
    critic: Any
    obs_dim: int

    GROUPS = ("value", "critic")
    STAGES = ("crl",)
    PROVIDES_V = True
    PROVIDES_COND = frozenset({"value_goal", "goal"})

    @property
    def TARGETS(self):
        return {}                       # contrastive loss is not bootstrapped -> no targets

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "crl"
        return self.GROUPS

    def _contrastive(self, net_def, p, obs, goal, actions):
        _, phi, psi = net_def.apply({"params": p}, obs, goal, actions=actions, info=True)
        if phi.ndim == 2:               # non-ensemble -> add a singleton head axis
            phi, psi = phi[None], psi[None]
        B = obs.shape[0]
        logits = jnp.einsum("eik,ejk->ije", phi, psi) / jnp.sqrt(phi.shape[-1])  # (B, B, e)
        eye = jnp.eye(B)
        loss = jax.vmap(
            lambda lg: optax.sigmoid_binary_cross_entropy(logits=lg, labels=eye),
            in_axes=-1, out_axes=-1)(logits).mean()
        lm = jnp.mean(logits, axis=-1)
        acc = jnp.mean(jnp.argmax(lm, axis=1) == jnp.argmax(eye, axis=1))
        return loss, acc

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "crl"
        obs, action, vg = batch["observations"], batch["actions"], cond["value_goal"]
        c_loss, c_acc = self._contrastive(self.critic, params["critic"], obs, vg, action)
        v_loss, v_acc = self._contrastive(self.value, params["value"], obs, vg, None)
        loss = c_loss + v_loss
        return loss, {"critic_loss": c_loss, "value_loss": v_loss,
                      "critic_acc": c_acc, "value_acc": v_acc}

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        q = self.critic.apply({"params": params["critic"]}, obs, cond["goal"], actions=actions)  # (2, B)
        q1, q2 = q[0], q[1]
        return jnp.minimum(q1, q2), jnp.stack([q1, q2], 0)

    def v_fn(self, params, obs, cond):
        return self.value.apply({"params": params["value"]}, obs, cond["goal"])                 # (B,)

    def cond_vec(self, cond):
        return cond["goal"]

    def sample_cond(self, net, batch, rng):
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        hd = tuple(config["value_hidden_dims"])
        d = int(config["latent_dim"])
        ln = bool(config["layer_norm"])
        head = dict(hidden_dims=hd, latent_dim=d, layer_norm=ln, value_exp=False)
        value = GCBilinearValue(ensemble=False, **head)
        critic = GCBilinearValue(ensemble=True, **head)
        ex_goal = jnp.zeros_like(ex_obs)
        rv, rc = jr.split(jr.PRNGKey(config["seed"] + 7000), 2)
        params = {
            "value": value.init(rv, ex_obs, ex_goal)["params"],
            "critic": critic.init(rc, ex_obs, ex_goal, ex_actions)["params"],
        }
        return cls(value=value, critic=critic, obs_dim=int(ex_obs.shape[-1])), params, {}
