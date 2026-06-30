"""HIQL with FlowBC actor(s). HIQL's IVL expectile value + state-dependent subgoal
representation goal_rep([s;g]) are inherited UNCHANGED; one or both of HIQL's two
Gaussian-AWR actors are replaced by AWR-weighted flow-matching BC actors, selected
by `flow_levels` ('low' | 'high' | 'both'):

  - low  : low_actor (action policy) -> AWR-weighted flow BC over actions,
           conditioned on concat(obs, goal_reps) (goal_reps = goal_rep([obs;low_g])).
  - high : high_actor (subgoal proposer) -> AWR-weighted flow BC over the rep_dim
           subgoal-rep target goal_rep([obs;high_target]), conditioned on
           concat(obs, goal); sampled rep is length-normalized (x sqrt(rep_dim)).

The non-flow level keeps HIQL's stock Gaussian-AWR actor. Mirrors gcivl_flow's
AWR-weighted flow-matching BC recipe (reuses GCVectorField). Registered as
agent_name='hiql_flow' by sitecustomize. Never edits OGBench.
"""
import copy

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.hiql import HIQLAgent, _BilinearRepValue, get_config as _hiql_get_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, MLP
from ogbench_flow import GCVectorField


class HIQLFlowAgent(HIQLAgent):
    """HIQL IVL value + subgoal rep, with AWR-weighted flow-matching BC actor(s)."""

    def _low_is_flow(self):
        return self.config['flow_levels'] in ('low', 'both')

    def _high_is_flow(self):
        return self.config['flow_levels'] in ('high', 'both')

    def _rollout(self, vf_name, obs_cond, noise, params=None):
        """Euler-integrate the named velocity field: noise -> sample."""
        fs = self.config['flow_steps']
        x = noise
        bshape = noise.shape[:-1]
        sel = self.network.select(vf_name)
        for i in range(fs):
            t = jnp.full((*bshape, 1), i / fs)
            v = sel(obs_cond, x, t, params=params) if params is not None else sel(obs_cond, x, t)
            x = x + v / fs
        return x

    def low_actor_loss(self, batch, grad_params, rng=None):
        if not self._low_is_flow():
            return HIQLAgent.low_actor_loss(self, batch, grad_params)
        # AWR advantage/weight identical to stock HIQL low_actor_loss.
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
        exp_a = jnp.minimum(jnp.exp(adv * self.config['low_alpha']), 100.0)

        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1), params=grad_params)
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        actions = batch['actions']
        rng = rng if rng is not None else self.rng
        rng, x0_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x0_rng, actions.shape)
        t = jax.random.uniform(t_rng, (actions.shape[0], 1))
        x_t = (1 - t) * x_0 + t * actions
        vel = actions - x_0
        obs_cond = jnp.concatenate([batch['observations'], goal_reps], axis=-1)
        pred = self.network.select('low_actor_vf')(obs_cond, x_t, t, params=grad_params)
        fm = jnp.mean((pred - vel) ** 2, axis=-1)
        actor_loss = jnp.mean(exp_a * fm)
        return actor_loss, {'actor_loss': actor_loss, 'adv': adv.mean(),
                            'flow_bc': fm.mean(), 'exp_a': exp_a.mean()}

    def high_actor_loss(self, batch, grad_params, rng=None):
        if not self._high_is_flow():
            return HIQLAgent.high_actor_loss(self, batch, grad_params)
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
        exp_a = jnp.minimum(jnp.exp(adv * self.config['high_alpha']), 100.0)

        # Subgoal-rep target (stop-grad, as in stock high_actor_loss which reads goal_rep
        # without grad_params). Flow BC matches the velocity toward this rep.
        target = jax.lax.stop_gradient(self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)))
        rng = rng if rng is not None else self.rng
        rng, x0_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x0_rng, target.shape)
        t = jax.random.uniform(t_rng, (target.shape[0], 1))
        x_t = (1 - t) * x_0 + t * target
        vel = target - x_0
        obs_cond = jnp.concatenate([batch['observations'], batch['high_actor_goals']], axis=-1)
        pred = self.network.select('high_actor_vf')(obs_cond, x_t, t, params=grad_params)
        fm = jnp.mean((pred - vel) ** 2, axis=-1)
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
        rng, low_rng, high_rng = jax.random.split(rng, 3)
        low_loss, low_info = self.low_actor_loss(batch, grad_params, low_rng)
        for k, v in low_info.items():
            info[f'low_actor/{k}'] = v
        high_loss, high_info = self.high_actor_loss(batch, grad_params, high_rng)
        for k, v in high_info.items():
            info[f'high_actor/{k}'] = v
        return value_loss + low_loss + high_loss, info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        high_seed, low_seed = jax.random.split(seed)
        event_rank = 1 if self.config['encoder'] is None else 3
        bshape = observations.shape[:-event_rank]

        # High stage: produce subgoal rep, then length-normalize (x sqrt(rep_dim)).
        if self._high_is_flow():
            hnoise = jax.random.normal(high_seed, (*bshape, self.config['rep_dim']))
            obs_goal = jnp.concatenate([observations, goals], axis=-1)
            goal_reps = self._rollout('high_actor_vf', obs_goal, hnoise)
        else:
            goal_reps = self.network.select('high_actor')(
                observations, goals, temperature=temperature).sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        # Low stage: produce the action conditioned on the subgoal rep.
        if self._low_is_flow():
            lnoise = jax.random.normal(low_seed, (*bshape, self.config['action_dim']))
            obs_cond = jnp.concatenate([observations, goal_reps], axis=-1)
            actions = self._rollout('low_actor_vf', obs_cond, lnoise)
        else:
            actions = self.network.select('low_actor')(
                observations, goal_reps, goal_encoded=True, temperature=temperature).sample(seed=low_seed)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        config = dict(config)
        action_dim = ex_actions.shape[-1]
        config['action_dim'] = action_dim
        rep_dim = config['rep_dim']

        # Subgoal representation goal_rep([s; g]) -> length-normalized rep_dim vector.
        if config['encoder'] is not None:
            em = encoder_modules[config['encoder']]
            goal_rep_seq = [em()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(MLP(hidden_dims=(*config['value_hidden_dims'], rep_dim),
                                activate_final=False, layer_norm=config['layer_norm']))
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if config['encoder'] is not None:
            em = encoder_modules[config['encoder']]
            value_encoder_def = GCEncoder(state_encoder=em(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=em(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=em(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=em())
            vf_obs_encoder = em()
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None
            vf_obs_encoder = None

        if config.get('value_arch', 'mlp') == 'bilinear':
            value_def = _BilinearRepValue(goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                                          latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'], ensemble=True)
            target_value_def = _BilinearRepValue(goal_rep=goal_rep_def, hidden_dims=config['value_hidden_dims'],
                                                 latent_dim=config['value_latent_dim'], layer_norm=config['layer_norm'], ensemble=True)
        else:
            value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                                ensemble=True, gc_encoder=value_encoder_def)
            target_value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                                       ensemble=True, gc_encoder=target_value_encoder_def)

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1),)),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
        )

        low_is_flow = config['flow_levels'] in ('low', 'both')
        high_is_flow = config['flow_levels'] in ('high', 'both')
        ex_rep = jnp.zeros((ex_observations.shape[0], rep_dim))
        ex_t = jnp.zeros((ex_actions.shape[0], 1))
        ex_obs_goal = jnp.concatenate([ex_observations, ex_goals], axis=-1)

        if low_is_flow:
            network_info['low_actor_vf'] = (
                GCVectorField(hidden_dim=config['actor_vf_hidden_dim'],
                              hidden_layers=config['actor_vf_hidden_layers'], obs_encoder=vf_obs_encoder),
                (jnp.concatenate([ex_observations, ex_rep], axis=-1), ex_actions, ex_t))
        else:
            low_actor_def = (GCDiscreteActor if config['discrete'] else GCActor)(
                hidden_dims=config['actor_hidden_dims'], action_dim=action_dim,
                **({} if config['discrete'] else dict(state_dependent_std=False, const_std=config['const_std'])),
                gc_encoder=low_actor_encoder_def)
            network_info['low_actor'] = (low_actor_def, (ex_observations, ex_goals))

        if high_is_flow:
            network_info['high_actor_vf'] = (
                GCVectorField(hidden_dim=config['actor_vf_hidden_dim'],
                              hidden_layers=config['actor_vf_hidden_layers'], obs_encoder=vf_obs_encoder),
                (ex_obs_goal, ex_rep, ex_t))
        else:
            high_actor_def = GCActor(hidden_dims=config['actor_hidden_dims'], action_dim=rep_dim,
                                     state_dependent_std=False, const_std=config['const_std'],
                                     gc_encoder=high_actor_encoder_def)
            network_info['high_actor'] = (high_actor_def, (ex_observations, ex_goals))

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
    config = _hiql_get_config()
    with config.unlocked():
        config.agent_name = 'hiql_flow'
        # Which actor(s) become flow: 'low' (action policy), 'high' (subgoal proposer), 'both'.
        config.flow_levels = 'low'
        config.flow_steps = 10
        config.actor_vf_hidden_dim = 512
        config.actor_vf_hidden_layers = 4
        # value_arch inherited from HIQL ('mlp' default; 'bilinear' for the rank-d head).
        config.value_arch = 'mlp'
        config.value_latent_dim = 512
    return config
