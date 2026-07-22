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
from impls.critics.gciql import GCIQLCritic
from impls.critics.isfb import ISFBCritic
from impls.critics.iql_bilinear import IqlBilinearCritic
from impls.critics.simple_sf import SimpleSfCritic
from impls.critics.sf_feature import SfFeatureCritic
from impls.critics.centered_bilinear import CenteredBilinearCritic
from impls.critics.fb_forward_bilinear import FbForwardBilinearCritic
from impls.actors.flowbc import FlowBCActor
from impls.actors.ddpgbc import DDPGBCActor
from impls.actors.ddpg import DDPGActor
from impls.actors.awr import AWRActor
from impls.actors.fql import FQLActor

CRITICS = {"fb": FBCritic, "gciql": GCIQLCritic, "isfb": ISFBCritic,
           "iql_bilinear": IqlBilinearCritic, "simple_sf": SimpleSfCritic,
           "sf_feature": SfFeatureCritic, "centered_bilinear": CenteredBilinearCritic,
           "fb_forward_bilinear": FbForwardBilinearCritic}
ACTORS = {"flowbc": FlowBCActor, "ddpgbc": DDPGBCActor, "ddpg": DDPGActor, "awr": AWRActor, "fql": FQLActor}

# per-group learning rates come from config; this maps group -> config key.
_LR_KEY = {"forward": "lr_f", "left_encoder": "lr_f", "backward": "lr_b",
           "value_forward": "lr_f",                       # isfb F_V rides the forward group
           "value": "lr", "critic": "lr", "center_b": "lr",  # gciql / centered_bilinear groups
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
        # `goals` is used by goal-conditioned critics (gciql: cond["goal"]); FB ignores it
        # (it conditions on cond_eval["z"]). When goals is None, use the stored eval cond.
        seed = self.rng if seed is None else seed
        cond = self.cond_eval if goals is None else {**self.cond_eval, "goal": goals}
        # Optional critic hook: some critics condition the actor on a FUNCTION of the goal
        # (iql_bilinear with actor_cond='z' feeds the actor psi(goal), width latent_dim),
        # which cannot be recovered from `cond` alone — it needs the critic's params.
        # Critics without the hook are unaffected.
        if hasattr(self.critic, "map_eval_cond"):
            cond = self.critic.map_eval_cond(self.net, cond)
        return self.actor.sample_actions(self.net, observations, cond, seed, temperature)

    def infer_eval_cond(self, observations, rewards):
        return self.replace(cond_eval=self.critic.infer_cond(self.net, observations, rewards))

    # ---------------- factory ----------------
    @classmethod
    def create(cls, critic_name, actor_name, seed, ex_obs, ex_actions, config):
        config = {**config, "seed": seed}
        critic_cls = CRITICS[critic_name]
        actor_cls = ACTORS[actor_name]
        critic, cparams, targets = critic_cls.build(ex_obs, ex_actions, config)
        # Validate before building the actor: check_combo reads only class attributes, so
        # an invalid pair raises ComboError naming the real reason rather than dying on
        # whatever config key the actor's build() indexes first.
        check_combo(critic, actor_cls)
        config = {**config, "cvec_dim": int(critic.cvec_dim)}  # actors size conditioning on this
        actor, aparams = actor_cls.build(ex_obs, ex_actions, config)

        params = {**cparams, **aparams}
        # Each group's nn.Module is an attribute named after the group on its owner.
        modules = {g: getattr(critic, g) for g in critic.GROUPS}
        modules.update({g: getattr(actor, g) for g in actor.GROUPS})
        net = {}
        for g, p in params.items():
            # actor_vf (flowbc) has its own lr; other groups map to a config lr key.
            lr = float(config["actor"]["lr_actor_vf"]) if g == "actor_vf" else float(config[_LR_KEY[g]])
            net[g] = TrainState.create(modules[g], p, tx=optax.adam(lr))

        cond_eval = critic.eval_cond0(ex_obs, config)
        return cls(rng=jax.random.PRNGKey(seed), net=net, target=targets,
                   cond_eval=cond_eval, critic=critic, actor=actor,
                   config=flax.core.FrozenDict(config))
