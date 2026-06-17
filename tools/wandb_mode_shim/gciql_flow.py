"""GCIQL + FB FlowBC actor (JAX). Subclasses the vendored GCIQLAgent: keeps the
IQL expectile value + double critic UNCHANGED, and replaces the Gaussian DDPG+BC
actor with FQL's flow actor -- a flow-matching velocity field + a one-shot
noise-conditioned policy distilled from it, with Q-guidance from GCIQL's IQL
min(q1, q2). Registered into agents.agents['gciql_flow'] by sitecustomize.
Never edits OGBench.

This is the actor-only ablation of GCIQL (everything else identical), so it is
directly comparable to the GCIQL DDPG+BC runs. It differs from canonical FQL only
in the critic: GCIQL's IQL (expectile V + advantage) instead of FQL's plain TD.
"""
import copy

import flax
import jax
import jax.numpy as jnp
import optax
from agents.gciql import GCIQLAgent, get_config as _gciql_get_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCValue, GCBilinearValue, GCActor

from ogbench_flow import GCNoiseActor, GCVectorField


class GCIQLFlowAgent(GCIQLAgent):
    """GCIQL with FB's FlowBC actor. Inherits value_loss/critic_loss/update."""

    def _flow_rollout(self, obs, noise, params):
        """Euler-integrate the BC velocity field from noise -> action (flow_steps)."""
        fs = self.config['flow_steps']
        actions = noise
        for i in range(fs):
            t = jnp.ones((noise.shape[0], 1)) * (i / fs)
            vels = self.network.select('actor_vf')(obs, actions, t, params=params)
            actions = actions + vels / fs
        return jnp.clip(actions, -1.0, 1.0)

    def _value_psi(self, goals, params):
        """B := the value net's psi (action-free GCBilinearValue). psi depends
        only on the goal arg; the obs arg is a placeholder for phi(s)."""
        _v, _phi, psi = self.network.select('value')(goals, goals, info=True, params=params)
        return psi

    def value_loss(self, batch, grad_params):
        """IQL expectile value loss; ablate clipped double-Q (double_q=False => q1)."""
        q1, q2 = self.network.select('target_critic')(
            batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2) if self.config['double_q'] else q1
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        return value_loss, {'value_loss': value_loss, 'v_mean': v.mean(),
                            'v_max': v.max(), 'v_min': v.min()}

    def actor_loss(self, batch, grad_params, rng=None):
        # actor_kind='ddpgbc' => fall back to GCIQL's stock gaussian DDPG+BC actor
        # (bilinear critic kept); only the flow-actor path is custom below.
        if self.config['actor_kind'] == 'ddpgbc':
            return GCIQLAgent.actor_loss(self, batch, grad_params, rng)
        obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
        rng = rng if rng is not None else self.rng
        rng, x0_rng, t_rng, n_rng = jax.random.split(rng, 4)

        # 1) flow-matching loss trains actor_vf (the BC flow policy).
        x_1 = actions
        x_0 = jax.random.normal(x0_rng, actions.shape)
        t = jax.random.uniform(t_rng, (actions.shape[0], 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_vf')(obs, x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # 2) Q-max on the one-shot actor's actions, using GCIQL's IQL min(q1,q2).
        # The critic is evaluated at FROZEN params (no params=grad_params): otherwise
        # -q.mean() backprops into the critic and the single optimizer inflates Q
        # instead of improving the actor (see crl-actor-loss-critic-gradient).
        noise = jax.random.normal(n_rng, actions.shape)
        # z-conditioning (G4): the actor conditions on z=stop_grad(psi(goal))
        # (B := value net's psi). stop_grad keeps the representation purely
        # IQL-trained. The critic Q-term keeps the raw goal obs -- it computes
        # its own psi internally.
        if self.config['actor_cond'] == 'z':
            actor_cond_input = jax.lax.stop_gradient(self._value_psi(goals, grad_params))
        else:
            actor_cond_input = goals
        actor_actions = self.network.select('actor')(obs, actor_cond_input, noise, params=grad_params)
        q1, q2 = self.network.select('critic')(obs, goals, actor_actions)
        q = jnp.minimum(q1, q2) if self.config['double_q'] else q1
        actor_loss = -q.mean()

        # 3) BC-distillation: one-shot actor -> frozen multi-step flow rollout (FQL
        # one-step distillation). Q-loss is normalized by |q| mean to be scale
        # invariant, exactly as in GCIQL's DDPG+BC actor loss.
        distill_error = 0.0
        if self.config['bc_coeff'] > 0:
            target = self._flow_rollout(obs, noise, jax.lax.stop_gradient(grad_params))
            distill_error = jnp.mean((actor_actions - target) ** 2)
            bc_loss = self.config['bc_coeff'] * distill_error
            actor_loss = actor_loss / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6) + bc_loss
        actor_loss = actor_loss + bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_error,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """IQL value + critic (inherited) + flow actor loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        # actor_kind='ddpgbc' => stock GCIQL gaussian actor sampling.
        if self.config['actor_kind'] == 'ddpgbc':
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
            return jnp.clip(dist.sample(seed=seed), -1.0, 1.0)
        del temperature  # flow actor is noise-driven
        # Noise must share the obs batch shape. Eval passes a single, unbatched
        # observation; training/tests pass batches. Event rank is 1 for state and
        # 3 (H,W,C) for pixels (encoder present), so strip the event dims.
        event_rank = 1 if self.config['encoder'] is None else 3
        batch_shape = observations.shape[:-event_rank]
        noise = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
        # z-conditioning (G4): the eval harness (and the in-training eval loop)
        # hands the RAW goal obs; map it through B := value.psi to the task vector
        # the z-actor expects. This makes in-training eval report exact-z success.
        # For an arbitrary precomputed z, use sample_actions_with_z (no mapping).
        if self.config['actor_cond'] == 'z':
            goals = self.network.select('value')(goals, goals, info=True)[2]
        actions = self.network.select('actor')(observations, goals, noise)
        return jnp.clip(actions, -1.0, 1.0)

    @jax.jit
    def sample_actions_with_z(self, observations, z, seed=None, temperature=1.0):
        """Roll the z-actor from a PRECOMPUTED task vector z (no psi mapping).
        Used by the G4 eval script's inferred-z arm (z = E[r.psi(s')])."""
        del temperature
        event_rank = 1 if self.config['encoder'] is None else 3
        batch_shape = observations.shape[:-event_rank]
        noise = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
        actions = self.network.select('actor')(observations, z, noise)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        config = dict(config)
        config['action_dim'] = action_dim

        # Actor goal-input example. In z-conditioning (G4) the actor consumes a
        # d-dim task vector z=psi(goal) (B := value net's psi), not the raw goal
        # observation, so its goal-leg params must be sized to value_latent_dim.
        if config.get('actor_cond', 'goal') == 'z':
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "actor_cond='z' requires value_arch='bilinear' (B := value net's psi)"
            ex_actor_goals = jnp.zeros((ex_observations.shape[0], config['value_latent_dim']))
        else:
            ex_actor_goals = ex_goals

        # Encoders: IQL value/critic use the GCEncoder wrapper (as in gciql.py);
        # the flow nets take the raw encoder module (as in crl_flowbc.py).
        encoders = {}
        if config['encoder'] is not None:
            em = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=em())
            encoders['critic'] = GCEncoder(concat_encoder=em())
            encoders['actor_obs'] = em()
            encoders['actor_goal'] = em()

        # value_arch: 'mlp' = canonical GCIQL GCValue (concat MLP over [s,g,a]);
        # 'bilinear' = inner-product value V=phi(s).psi(g)/sqrt(d), Q=phi(s,a).psi(g)/sqrt(d)
        # (Direction A: GCIQL's IQL training poured through an FB-style rank-d head).
        if config.get('value_arch', 'mlp') == 'bilinear':
            d = config['value_latent_dim']
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'], ensemble=False,
                state_encoder=encoders.get('value'), goal_encoder=encoders.get('value'),
            )
            critic_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'], ensemble=True,
                state_encoder=encoders.get('critic'), goal_encoder=encoders.get('critic'),
            )
        else:
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                gc_encoder=encoders.get('value'),
            )
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )
        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
        )
        if config.get('actor_kind', 'flow') == 'ddpgbc':
            # Stock GCIQL gaussian DDPG+BC actor on the (possibly bilinear) critic.
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor_obs'),
            )
            network_info['actor'] = (actor_def, (ex_observations, ex_goals))
        else:
            actor_vf_def = GCVectorField(
                hidden_dim=config['actor_vf_hidden_dim'],
                hidden_layers=config['actor_vf_hidden_layers'],
                obs_encoder=encoders.get('actor_obs'),
            )
            actor_def = GCNoiseActor(
                hidden_dim=config['actor_hidden_dim'],
                hidden_layers=config['actor_hidden_layers'],
                embedding_layers=config['actor_embedding_layers'],
                obs_encoder=encoders.get('actor_obs'),
                goal_encoder=encoders.get('actor_goal'),
            )
            ex_noise = jnp.zeros_like(ex_actions)
            ex_t = jnp.zeros((ex_actions.shape[0], 1))
            network_info['actor_vf'] = (actor_vf_def, (ex_observations, ex_actions, ex_t))
            network_info['actor'] = (actor_def, (ex_observations, ex_actor_goals, ex_noise))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = _gciql_get_config()
    with config.unlocked():
        config.agent_name = 'gciql_flow'
        # Value architecture: 'mlp' (canonical GCIQL) or 'bilinear' (inner-product, Direction A).
        config.value_arch = 'mlp'
        config.value_latent_dim = 512
        # Actor conditioning identity (G4 objective-isolation arm):
        # 'goal' = raw goal obs (stock); 'z' = z=stop_grad(value.psi(goal)),
        # FB-style single backward map. 'z' requires value_arch='bilinear'.
        config.actor_cond = 'goal'
        # Clipped double-Q: True = min(q1,q2) (stock); False = q1 only (pessimism ablation).
        config.double_q = True
        # Actor family: 'flow' (FlowBC, the custom actor) or 'ddpgbc' (stock GCIQL
        # gaussian actor, for the no-flow-actor control). 'ddpgbc' needs actor_loss=ddpgbc.
        config.actor_kind = 'flow'
        # Flow actor hyperparameters.
        config.flow_steps = 10  # Euler steps for the BC flow rollout / distillation.
        config.bc_coeff = 3.0  # Distillation (BC) coefficient; the key actor knob.
        config.actor_vf_hidden_dim = 512
        config.actor_vf_hidden_layers = 4
        config.actor_hidden_dim = 512
        config.actor_hidden_layers = 2
        config.actor_embedding_layers = 2
    return config
