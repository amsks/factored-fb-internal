"""Factored agent: composes a Critic and an Actor over per-group TrainStates and drives
them with a generic staged update. Structure is OGBench-idiomatic (TrainState + nonpytree
static aux); FB math/RNG order are byte-identical to PSMFlows agents/fb.py."""
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from impls.utils.flax_utils import TrainState, soft_update, nonpytree_field
from impls.critics.base import check_combo
from impls.critics.fb import FBCritic
from impls.actors.flowbc import FlowBCActor
from impls.actors.ddpgbc import DDPGBCActor
from impls.actors.awr import AWRActor
from impls.actors.fql import FQLActor

CRITICS = {"fb": FBCritic}
ACTORS = {"flowbc": FlowBCActor, "ddpgbc": DDPGBCActor, "awr": AWRActor, "fql": FQLActor}

# per-group learning rates come from config; this maps group -> config key.
_LR_KEY = {"forward": "lr_f", "left_encoder": "lr_f", "backward": "lr_b",
           "actor": "lr_actor", "actor_vf": None,          # actor_vf uses actor.lr_actor_vf
           "actor_bc_flow": "lr_actor", "actor_onestep": "lr_actor"}  # fql groups


class Agent(flax.struct.PyTreeNode):
    rng: Any
    net: Any            # dict[group -> TrainState]   (online groups)
    target: Any         # dict[group -> params]        (target groups)
    cond_eval: Any      # eval-time conditioning (FB: {"z": ...})
    critic: Any = nonpytree_field()
    actor: Any = nonpytree_field()
    config: Any = nonpytree_field()

    # ---------------- core staged update (no drawing) ----------------
    def apply_update(self, batch, cond, boot):
        net = dict(self.net)
        target = dict(self.target)
        info = {}
        online = lambda: {g: net[g].params for g in net}

        for stage in self.critic.STAGES:
            groups = self.critic.stage_groups(stage)

            def stage_loss_fn(gp):
                params = {**online(), **gp}
                return self.critic.stage_loss(stage, params, target, batch, cond, boot)

            (_, sinfo), grads = jax.value_and_grad(stage_loss_fn, has_aux=True)(
                {g: net[g].params for g in groups})
            for g in groups:
                net[g] = net[g].apply_gradients(grads=grads[g])
            info.update(sinfo)

        def actor_loss_fn(gp):
            params = {**online(), **gp}
            return self.actor.loss(params, target, batch, self.critic, cond, boot)

        (_, ainfo), agrads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            {g: net[g].params for g in self.actor.GROUPS})
        for g in self.actor.GROUPS:
            net[g] = net[g].apply_gradients(grads=agrads[g])
        info.update(ainfo)

        for g, tau in self.critic.TARGETS.items():
            target[g] = soft_update(net[g].params, target[g], tau)

        return self.replace(net=net, target=target), info

    def compute_static(self, batch, cond, boot):
        info = {}
        online = {g: self.net[g].params for g in self.net}
        for stage in self.critic.STAGES:
            _, sinfo = self.critic.stage_loss(stage, online, self.target, batch, cond, boot)
            info.update(sinfo)
        _, ainfo = self.actor.loss(online, self.target, batch, self.critic, cond, boot)
        info.update(ainfo)
        return info

    # ---------------- live update (draws randomness) ----------------
    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        r_cond, r_boot = jax.random.split(rng)
        cond = self.critic.sample_cond(self.net, batch, r_cond)
        boot = self.actor.sample_bootstrap(self.net, batch, cond, r_boot)
        new_agent, info = self.apply_update(batch, cond, boot)
        return new_agent.replace(rng=new_rng), info

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        # `goals` is accepted (and ignored) so the ogbench eval loop can call uniformly;
        # FB conditions on cond_eval (an inferred task latent), not a per-episode goal.
        seed = self.rng if seed is None else seed
        return self.actor.sample_actions(self.net, observations, self.cond_eval, seed, temperature)

    def infer_eval_cond(self, observations, rewards):
        return self.replace(cond_eval=self.critic.infer_cond(self.net, observations, rewards))

    # ---------------- factory ----------------
    @classmethod
    def create(cls, critic_name, actor_name, seed, ex_obs, ex_actions, config):
        config = {**config, "seed": seed}
        critic_cls = CRITICS[critic_name]
        actor_cls = ACTORS[actor_name]
        critic, cparams, targets = critic_cls.build(ex_obs, ex_actions, config)
        actor, aparams = actor_cls.build(ex_obs, ex_actions, config)
        check_combo(critic, actor)

        params = {**cparams, **aparams}
        # Each group's nn.Module is an attribute named after the group on its owner.
        modules = {g: getattr(critic, g) for g in critic.GROUPS}
        modules.update({g: getattr(actor, g) for g in actor.GROUPS})
        net = {}
        for g, p in params.items():
            # actor_vf (flowbc) has its own lr; other groups map to a config lr key.
            lr = float(config["actor"]["lr_actor_vf"]) if g == "actor_vf" else float(config[_LR_KEY[g]])
            net[g] = TrainState.create(modules[g], p, tx=optax.adam(lr))

        z_dim = config["z_dim"]
        cond_eval = {"z": jnp.zeros((z_dim,), jnp.float64)}
        return cls(rng=jax.random.PRNGKey(seed), net=net, target=targets,
                   cond_eval=cond_eval, critic=critic, actor=actor,
                   config=flax.core.FrozenDict(config))
