"""campaign/jax_agents/flowbc.py — pure Flow-matching Behavior Cloning baseline.

The flow-BC component of FB+FlowBC / CRL+FlowBC in ISOLATION: just the velocity field
(actor_vf), trained by conditional flow matching on (obs, action). No critic, no Q,
no FB representation, no goal/z conditioning (the obs carries the goal for these tasks).
Eval samples an action by integrating the flow ODE from noise.

This ablates the effect of behavior cloning vs the RL critics: compare FB (no BC) /
FB+FlowBC / CRL (gaussian BC) / CRL+FlowBC against pure Flow-BC (imitation only).

Registered as agents.agents['flowbc'] -> drives via jax_train_unified / eval.
"""
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogbench_flow import GCVectorField


class FlowBCAgent(flax.struct.PyTreeNode):
    rng: jnp.ndarray
    network: TrainState
    config: ml_collections.ConfigDict = nonpytree_field()

    def _flow_rollout(self, net, obs, noise, params):
        fs = self.config["flow_steps"]
        a = noise
        for i in range(fs):
            t = jnp.ones((noise.shape[0], 1)) * (i / fs)
            a = a + net.select("actor_vf")(obs, a, t, params=params) / fs
        return jnp.clip(a, -1.0, 1.0)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        obs, action = batch["observations"], batch["actions"]
        rng = rng if rng is not None else self.rng
        rng, x0r, tr = jax.random.split(rng, 3)
        x0 = jax.random.normal(x0r, action.shape)
        t = jax.random.uniform(tr, (action.shape[0], 1))
        x_t = (1 - t) * x0 + t * action
        pred = self.network.select("actor_vf")(obs, x_t, t, params=grad_params)
        bc_flow = jnp.mean((pred - (action - x0)) ** 2)
        return bc_flow, dict(bc_flow_loss=bc_flow)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, z=None, seed=None, temperature=1.0):
        del goals, z, temperature
        obs = jnp.asarray(observations, jnp.float32)
        single = obs.ndim == 1
        obs2 = obs[None] if single else obs
        seed = seed if seed is not None else jax.random.PRNGKey(0)
        noise = jax.random.normal(seed, (obs2.shape[0], self.config["action_dim"]))
        a = self._flow_rollout(self.network, obs2, noise, self.network.params)
        return a[0] if single else a

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        act_dim = ex_actions.shape[-1]
        config = dict(config)
        config["action_dim"] = act_dim
        config = ml_collections.FrozenConfigDict(config)
        ex_obs = jnp.asarray(ex_observations, jnp.float32)
        ex_act = jnp.asarray(ex_actions, jnp.float32)
        ex_t = jnp.zeros((ex_obs.shape[0], 1), jnp.float32)

        vf = GCVectorField(hidden_dim=config["hidden_dim"], hidden_layers=config["hidden_layers"])
        net_def = ModuleDict({"actor_vf": vf})
        tx = optax.adam(learning_rate=config["lr"])
        params = net_def.init(init_rng, actor_vf=(ex_obs, ex_act, ex_t))["params"]
        network = TrainState.create(net_def, params, tx=tx)
        return cls(rng=rng, network=network, config=config)


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name="flowbc", lr=3e-4, batch_size=256, flow_steps=10,
        discount=0.99,                                 # unused by flowbc; GCDataset needs it for goal sampling
        hidden_dim=512, hidden_layers=4,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class="GCDataset",
        value_p_curgoal=0.0, value_p_trajgoal=1.0, value_p_randomgoal=0.0, value_geom_sample=1,
        actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0, actor_geom_sample=0,
        gc_negative=1, p_aug=ml_collections.config_dict.placeholder(float),
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))


import agents as _ogbench_agents  # noqa: E402
_ogbench_agents.agents["flowbc"] = FlowBCAgent
