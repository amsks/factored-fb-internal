"""QRLCritic — offline Quasimetric RL (Wang et al., arXiv:2304.01203) on the factored seam.

Learns d_theta(s, g) with an architecturally-quasimetric IQE head over an MLP encoder:
maximize a damped E[d(s, g)] over (state, value_goal) pairs subject to not overestimating
the one-step cost, E[relu(d(s, s') - step_cost)^2] <= eps^2, via a learnable Lagrange
multiplier (dual ascent through a sign-flipped gradient term).

Representation-only: no latent dynamics model and no critic-driven actor extraction — pair
with the bcflow actor (pure BC). V(s,g) = -d(s,g) is provided so probes/AWR can read it;
q_fn ignores actions and returns V (logged by flowbc-style actors, never optimized when
q_coeff=0). No target networks.

Simplification vs. the QRL repo, on purpose (documented in the metric-probe spec): the
global term uses the bounded concave transform d/(d+T) instead of QRL's affine-softplus
damping — same role (vanishing gradient on already-large distances), simpler.
"""
import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from impls.utils.networks import MLP
from impls.utils.quasimetrics import iqe


class QRLNet(nn.Module):
    hidden_dims: tuple
    components: int          # k
    dim_per_component: int   # l
    layer_norm: bool = True

    def setup(self):
        self.encoder = MLP((*self.hidden_dims, self.components * self.dim_per_component),
                           activate_final=False, layer_norm=self.layer_norm)
        self.alpha_raw = self.param("alpha_raw", nn.initializers.zeros, ())     # sigmoid -> 0.5
        # softplus(lam_raw) = 1.0 at init
        self.lam_raw = self.param("lam_raw", nn.initializers.constant(math.log(math.e - 1.0)), ())

    def embed(self, obs):
        return self.encoder(obs)

    def lam(self):
        return nn.softplus(self.lam_raw)

    def __call__(self, obs, goals):
        u, v = self.encoder(obs), self.encoder(goals)
        return iqe(u, v, self.components, nn.sigmoid(self.alpha_raw))


@dataclasses.dataclass(frozen=True)
class QRLCritic:
    value: Any
    step_cost: float
    eps: float
    transform_t: float
    obs_dim: int

    GROUPS = ("value",)
    STAGES = ("qrl",)
    PROVIDES_V = True
    PROVIDES_COND = frozenset({"value_goal", "goal"})

    @property
    def TARGETS(self):
        return {}                       # no bootstrapping -> no targets

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "qrl"
        return self.GROUPS

    def _d(self, p, obs, goals):
        return self.value.apply({"params": p}, obs, goals)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "qrl"
        p = params["value"]
        obs, next_obs, vg = batch["observations"], batch["next_observations"], cond["value_goal"]

        d_global = self._d(p, obs, vg)
        d_local = self._d(p, obs, next_obs)
        lam = self.value.apply({"params": p}, method="lam")

        # maximize bounded transform of global distances
        global_term = (d_global / (d_global + self.transform_t)).mean()
        # one-sided local constraint: only OVERestimating the step cost is penalized
        viol = (jnp.maximum(d_local - self.step_cost, 0.0) ** 2).mean() - self.eps ** 2
        # theta descends -global + lam*viol; lam_raw ascends on viol via the sign-flipped term
        loss = (-global_term
                + jax.lax.stop_gradient(lam) * viol
                - lam * jax.lax.stop_gradient(viol))
        return loss, {"qrl_loss": loss, "d_global_mean": d_global.mean(),
                      "d_local_mean": d_local.mean(), "constraint_viol": viol, "lambda": lam}

    def v_fn(self, params, obs, cond):
        return -self._d(params["value"], obs, cond["goal"])

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # No action model: Q := V(s,g), ignoring actions. Only ever consumed as a logged
        # scalar by bcflow (q_coeff=0); do not pair with a critic-maximizing actor.
        v = self.v_fn(params, obs, cond)
        return v, jnp.stack([v, v], 0)

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
        net = QRLNet(hidden_dims=tuple(config["value_hidden_dims"]),
                     components=int(config["qrl_components"]),
                     dim_per_component=int(config["qrl_dim_per_component"]),
                     layer_norm=bool(config["layer_norm"]))
        ex_goal = jnp.zeros_like(ex_obs)
        params = {"value": net.init(jr.PRNGKey(config["seed"] + 8000), ex_obs, ex_goal)["params"]}
        obj = cls(value=net, step_cost=float(config["qrl_step_cost"]),
                  eps=float(config["qrl_eps"]), transform_t=float(config["qrl_transform_t"]),
                  obs_dim=int(ex_obs.shape[-1]))
        return obj, params, {}
