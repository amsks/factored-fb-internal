"""GCIVL with a goal-conditioned FlowBC actor. GCIVL is V-only (no Q), so the policy
cannot get its goal-conditioning from FQL's Q-max. Instead: a GOAL-CONDITIONED flow
velocity field (the goal is concatenated into the velocity net's obs input) trained by
AWR-weighted flow-matching BC -- weight exp(alpha*(V(s',g)-V(s,g))). Actions are sampled
by Euler-integrating the velocity field (no separate one-shot actor). Value (IVL expectile,
mlp|bilinear) inherited from GCIVLAgent. Registered as agent_name='gcivl_flow'."""
import copy

import flax
import jax
import jax.numpy as jnp
import optax

from agents.gcivl import GCIVLAgent, get_config as _gcivl_get_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCBilinearValue, GCValue
from ogbench_flow import GCVectorField


class GCIVLFlowAgent(GCIVLAgent):
    """GCIVL IVL value + goal-conditioned, advantage-weighted flow-matching BC actor."""

    def _rollout(self, obs_goal, noise, params=None):
        """Euler-integrate the goal-conditioned velocity field noise -> action."""
        fs = self.config['flow_steps']
        actions = noise
        batch_shape = noise.shape[:-1]
        sel = self.network.select('actor_vf')
        for i in range(fs):
            t = jnp.full((*batch_shape, 1), i / fs)
            vels = sel(obs_goal, actions, t, params=params) if params is not None else sel(obs_goal, actions, t)
            actions = actions + vels / fs
        return jnp.clip(actions, -1.0, 1.0)

    def actor_loss(self, batch, grad_params, rng=None):
        obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
        # AWR advantage from the IVL value (V-only): adv = V(s',g) - V(s,g).
        v1, v2 = self.network.select('value')(obs, goals)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], goals)
        adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
        exp_a = jnp.minimum(jnp.exp(adv * self.config['alpha']), 100.0)

        # Goal-conditioned flow-matching BC, weighted per-sample by exp_a.
        rng = rng if rng is not None else self.rng
        rng, x0_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x0_rng, actions.shape)
        t = jax.random.uniform(t_rng, (actions.shape[0], 1))
        x_t = (1 - t) * x_0 + t * actions
        vel = actions - x_0
        obs_goal = jnp.concatenate([obs, goals], axis=-1)
        pred = self.network.select('actor_vf')(obs_goal, x_t, t, params=grad_params)
        fm = jnp.mean((pred - vel) ** 2, axis=-1)  # [B]
        actor_loss = jnp.mean(exp_a * fm)
        return actor_loss, {'actor_loss': actor_loss, 'adv': adv.mean(),
                            'flow_bc': fm.mean(), 'exp_a': exp_a.mean()}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        return value_loss + actor_loss, info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        del temperature  # flow actor is noise-driven
        event_rank = 1 if self.config['encoder'] is None else 3
        batch_shape = observations.shape[:-event_rank]
        noise = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
        obs_goal = jnp.concatenate([observations, goals], axis=-1)
        return self._rollout(obs_goal, noise)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        config = dict(config)
        config['action_dim'] = action_dim

        encoders = {}
        if config['encoder'] is not None:
            em = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=em())
            encoders['actor_obs'] = em()

        if config.get('value_arch', 'mlp') == 'bilinear':
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'], latent_dim=config['value_latent_dim'],
                layer_norm=config['layer_norm'], ensemble=True,
                state_encoder=encoders.get('value'), goal_encoder=encoders.get('value'))
        else:
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                ensemble=True, gc_encoder=encoders.get('value'))

        # Goal-conditioned velocity field: goal is concatenated into its obs input.
        actor_vf_def = GCVectorField(
            hidden_dim=config['actor_vf_hidden_dim'], hidden_layers=config['actor_vf_hidden_layers'],
            obs_encoder=encoders.get('actor_obs'))
        ex_obsgoal = jnp.concatenate([ex_observations, ex_goals], axis=-1)
        ex_t = jnp.zeros((ex_actions.shape[0], 1))

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor_vf=(actor_vf_def, (ex_obsgoal, ex_actions, ex_t)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network_params
        params['modules_target_value'] = params['modules_value']
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = _gcivl_get_config()
    with config.unlocked():
        config.agent_name = 'gcivl_flow'
        config.value_arch = 'mlp'
        config.value_latent_dim = 512
        config.flow_steps = 10
        config.actor_vf_hidden_dim = 512
        config.actor_vf_hidden_layers = 4
    return config
