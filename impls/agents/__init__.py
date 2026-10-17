"""The agent: a critic and an actor composed over per-group TrainStates, trained by a
staged update (critic stages, then the actor, then the target networks)."""
from typing import Any

import flax
import jax
import optax

from impls.utils.flax_utils import TrainState, soft_update, nonpytree_field
from impls.critics.base import check_combo
from impls.utils.config_utils import upgrade_config
from impls.critics.fb import FBCritic
from impls.critics.density_fb import DensityFBCritic
from impls.critics.hilp import HILPCritic
from impls.critics.rldp import RLDPCritic
from impls.actors.flowbc import FlowBCActor
from impls.actors.ddpgbc import DDPGBCActor

CRITICS = {"fb": FBCritic, "density_fb": DensityFBCritic, "hilp": HILPCritic, "rldp": RLDPCritic}

# Config-only variants: a --critic name that selects its own yaml but the same critic class.
#   density_fb_z      Density-FB whose task vector z is inferred from reward samples
#                     (the yaml sets infer_z: true); used for every ExoRL cell and
#                     the reward-inference OGBench numbers.
#   density_fb_state  goal-conditioned Density-FB whose backward network reads the full state
#                     instead of the object coordinates (backward_input: state).
#   fb_tdj, fb_ref    FB at the TD-JEPA reference settings (ortho_coef 1000). The two yamls
#                     are identical; both names are kept because run directories use them.
#   hilp_tdjc         HILP matched to TD-JEPA's OGBench configuration.
CRITIC_ALIASES = {"density_fb_z": "density_fb", "density_fb_state": "density_fb",
                  "fb_tdj": "fb", "fb_ref": "fb", "hilp_tdjc": "hilp"}

ACTORS = {"flowbc": FlowBCActor, "ddpgbc": DDPGBCActor}

# Per-group learning rates: optimizer group -> config key.
_LR_KEY = {"forward": "lr_f", "left_encoder": "lr_f", "backward": "lr_b",
           # RLDP's multi-step predictor rides the backward optimizer, as in the reference.
           "predictor": "lr_b",
           # Density-FB's two towers. The index state (slow encoder copy, running moments)
           # never receives a gradient; it lives in `net` so every consumer can read it.
           "density_f": "lr", "density_b": "lr",
           "density_b_zenc": "lr", "density_zstats": "lr",
           # HILP: phi towers ride lr_b, the successor-feature net and left encoder lr_f.
           "hilp_phi1": "lr_b", "hilp_phi2": "lr_b", "hilp_sf": "lr_f", "hilp_stats": "lr_b",
           "hilp_le": "lr_f",
           "actor": "lr_actor", "actor_vf": None}   # actor_vf uses actor.lr_actor_vf


def _group_lr(group, config):
    """The learning rate one optimizer group is built with. `group_lr: {group: lr}` in the
    config overrides the _LR_KEY lookup."""
    if group in (config.get("group_lr") or {}):
        return float(config["group_lr"][group])
    if group == "actor_vf":                       # flowbc's value-flow head owns its own lr
        return float(config["actor"]["lr_actor_vf"])
    return float(config[_LR_KEY[group]])


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

        # Stage parameter source. By default each stage re-reads `net`, so a later stage sees
        # the parameters an earlier one just stepped. Official HILP computes every stage's
        # loss from the same pre-step parameters and steps them together afterwards;
        # `stages_from_pre_step_params` (set by hilp_tdjc) reproduces that.
        pre_step = {g: net[g].params for g in net}
        from_pre = bool(self.config.get("stages_from_pre_step_params", False))
        pending = {}
        for stage in self.critic.STAGES:
            groups = self.critic.stage_groups(stage)
            base = pre_step if from_pre else None

            def stage_loss_fn(gp, _base=base):
                params = {**(_base if _base is not None else online()), **gp}
                return self.critic.stage_loss(stage, params, target, batch, cond, boot)

            (_, sinfo), grads = jax.value_and_grad(stage_loss_fn, has_aux=True)(
                {g: (pre_step[g] if from_pre else net[g].params) for g in groups})
            if from_pre:
                pending.update({g: grads[g] for g in groups})
            else:
                for g in groups:
                    net[g] = net[g].apply_gradients(grads=grads[g])
            info.update(sinfo)
        for g, gr in pending.items():          # all stages stepped together, after the fact
            net[g] = net[g].apply_gradients(grads=gr)

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

        # Optional critic hook for state that is neither a gradient target nor a polyak
        # target: Density-FB's running feature moments, slow encoder copy and frozen eval
        # pool. It reads the pre-step online params.
        if hasattr(self.critic, "update_stats"):
            net = self.critic.update_stats(net, self.net, batch)

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
        """Act for `observations`. With `goals` the critic maps the goal to its conditioning
        (map_eval_cond); without, the stored eval conditioning (e.g. an inferred z) is used."""
        seed = self.rng if seed is None else seed
        cond = self.cond_eval if goals is None else {**self.cond_eval, "goal": goals}
        cond = self.critic.map_eval_cond(self.net, cond, observations=observations)
        return self.actor.sample_actions(self.net, observations, cond, seed, temperature)

    def infer_eval_cond(self, observations, rewards):
        return self.replace(cond_eval=self.critic.infer_cond(self.net, observations, rewards))

    # ---------------- factory ----------------
    @classmethod
    def create(cls, critic_name, actor_name, seed, ex_obs, ex_actions, config):
        config = {**upgrade_config(config), "seed": seed}
        critic_cls = CRITICS[CRITIC_ALIASES.get(critic_name, critic_name)]
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
            net[g] = TrainState.create(modules[g], p, tx=optax.adam(_group_lr(g, config)))

        cond_eval = critic.eval_cond0(ex_obs, config)
        return cls(rng=jax.random.PRNGKey(seed), net=net, target=targets,
                   cond_eval=cond_eval, critic=critic, actor=actor,
                   config=flax.core.FrozenDict(config))
