"""campaign/jax_agents/fb.py — JAX/Flax port of the torch FB agent (agents/fb/), for the
unified jax pipeline. Supports plain FB (TD3 Gaussian actor) and FB+FlowBC (flow-matching
actor) via the config flag `flow`. Faithful to the torch logic:

  goal = next_obs; z = half random (sqrt(d)*normalize(gaussian)) / half project(B(next_obs[perm])).
  Ms = F(left_enc(obs), z, action) . B(next_obs).T   ; diff = Ms - gamma*min_P target_M
  fb = 0.5*(diff*offdiag)^2/sum - diag(diff).mean*P  + ortho_coef*[0.5*(BB^T*offdiag)^2/sum - diag(BB^T).mean]
  TD3 actor: -min_P(F(left_enc,z,a_pi).z).mean (+bc);  forward FROZEN in actor loss.
  FlowBC actor: bc_flow_loss (actor_vf) + -Q(one-shot actor) + bc distillation to the flow rollout.
  soft targets tau on forward/backward/left_encoder.

Registered into agents.agents['fb'] (fb_flowbc.py registers 'fb_flowbc' with flow=True).
"""
import copy

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ensemblize
from ogbench_flow import GCVectorField, GCNoiseActor


# --------------------------------------------------------------------- networks
class SimpleEmbedding(nn.Module):
    hidden_dim: int
    embedding_layers: int

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        for _ in range(self.embedding_layers - 1):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.relu(nn.Dense(self.hidden_dim // 2)(x))


class TrunkNet(nn.Module):
    hidden_dim: int
    hidden_layers: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        for _ in range(self.hidden_layers - 1):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.out_dim)(x)


class ForwardMap(nn.Module):
    hidden_dim: int
    hidden_layers: int
    embedding_layers: int
    z_dim: int

    @nn.compact
    def __call__(self, obs, z, action):
        ez = SimpleEmbedding(self.hidden_dim, self.embedding_layers)(jnp.concatenate([obs, z], -1))
        esa = SimpleEmbedding(self.hidden_dim, self.embedding_layers)(jnp.concatenate([obs, action], -1))
        x = jnp.concatenate([ez, esa], -1)
        for _ in range(self.hidden_layers):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.z_dim)(x)


class Actor(nn.Module):
    """Plain-FB TD3 actor mean (unbounded)."""
    hidden_dim: int
    hidden_layers: int
    embedding_layers: int
    action_dim: int

    @nn.compact
    def __call__(self, obs, z):
        ez = SimpleEmbedding(self.hidden_dim, self.embedding_layers)(jnp.concatenate([obs, z], -1))
        es = SimpleEmbedding(self.hidden_dim, self.embedding_layers)(obs)
        x = jnp.concatenate([ez, es], -1)
        for _ in range(self.hidden_layers):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.action_dim)(x)


def _fwd_module(cfg, zc):
    return ensemblize(ForwardMap, cfg["num_parallel"])(
        hidden_dim=cfg["hidden_dim"], hidden_layers=cfg["fwd_layers"],
        embedding_layers=cfg["embedding_layers"], z_dim=zc)


# ------------------------------------------------------------------------ agent
class FBAgent(flax.struct.PyTreeNode):
    rng: jnp.ndarray
    network: TrainState
    config: ml_collections.ConfigDict = nonpytree_field()

    def project_z(self, z):
        return jnp.sqrt(z.shape[-1]) * z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def sample_z(self, rng, n):
        return self.project_z(jax.random.normal(rng, (n, self.config["z_dim"])))

    def infer_z(self, next_obs, reward):
        """Zero-shot FB task vector z = project(sum_i reward_i * B(next_obs_i)).
        Not jitted (called once at eval setup over a subsample of the offline data)."""
        B = self.network.select("backward")(jnp.asarray(next_obs, jnp.float32))   # [N, z_dim]
        z = jnp.matmul(jnp.asarray(reward, jnp.float32).reshape(1, -1), B)         # [1, z_dim]
        return self.project_z(z)[0]                                                # [z_dim]

    def _flow_rollout(self, net, obs, noise, params):
        fs = self.config["flow_steps"]
        a = noise
        for i in range(fs):
            t = jnp.ones((noise.shape[0], 1)) * (i / fs)
            a = a + net.select("actor_vf")(obs, a, t, params=params) / fs
        return jnp.clip(a, -1.0, 1.0)

    def _actor_loss(self, net, grad_params, obs, left, z, action, rng):
        cfg = self.config
        if not cfg["flow"]:
            a_pi = jnp.clip(net.select("actor")(obs, z, params=grad_params), -1.0, 1.0)
            Fpi = net.select("forward")(jax.lax.stop_gradient(left), z, a_pi)   # FROZEN forward
            Qs = (Fpi * z[None]).sum(-1)
            Q = jnp.min(Qs, axis=0)
            aloss = -Q.mean()
            bc = 0.0
            if cfg["bc_coeff"] > 0:
                bc = jnp.mean((a_pi - action) ** 2)
                aloss = aloss / (jnp.abs(Qs).mean() + 1e-6) + cfg["bc_coeff"] * bc
            return aloss, dict(actor_loss=aloss, q=Q.mean(), bc=bc, bc_flow=0.0)
        # FlowBC actor
        rng, x0r, tr, nr = jax.random.split(rng, 4)
        x0 = jax.random.normal(x0r, action.shape)
        t = jax.random.uniform(tr, (action.shape[0], 1))
        x_t = (1 - t) * x0 + t * action
        pred = net.select("actor_vf")(obs, x_t, t, params=grad_params)
        bc_flow = jnp.mean((pred - (action - x0)) ** 2)
        noise = jax.random.normal(nr, action.shape)
        a_pi = net.select("actor")(obs, z, noise, params=grad_params)           # tanh-bounded
        Fpi = net.select("forward")(jax.lax.stop_gradient(left), z, a_pi)        # FROZEN forward
        Qs = (Fpi * z[None]).sum(-1)
        Q = jnp.min(Qs, axis=0)
        aloss = -Q.mean()
        bc = 0.0
        if cfg["bc_coeff"] > 0:
            target = self._flow_rollout(net, obs, noise, jax.lax.stop_gradient(grad_params))
            bc = jnp.mean((a_pi - target) ** 2)
            aloss = aloss / (jnp.abs(Qs).mean() + 1e-6) + cfg["bc_coeff"] * bc
        aloss = aloss + bc_flow
        return aloss, dict(actor_loss=aloss, q=Q.mean(), bc=bc, bc_flow=bc_flow)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        cfg = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        term = batch["terminals"] if "terminals" in batch else jnp.zeros(obs.shape[0])
        B = obs.shape[0]
        discount = (cfg["discount"] * (1.0 - term)).reshape(-1, 1)
        rng = rng if rng is not None else self.rng
        rng, zr, pr, mr, ar = jax.random.split(rng, 5)
        net = self.network

        z_rand = self.sample_z(zr, B)
        perm = jax.random.permutation(pr, B)
        z_goal = self.project_z(net.select("backward")(next_obs[perm], params=grad_params))
        mask = jax.random.uniform(mr, (B, 1)) < cfg["train_goal_ratio"]
        z = jnp.where(mask, z_goal, z_rand)

        left = net.select("left_encoder")(obs, params=grad_params)
        Fs = net.select("forward")(left, z, action, params=grad_params)
        Bg = net.select("backward")(next_obs, params=grad_params)
        Ms = jnp.einsum("pbz,cz->pbc", Fs, Bg)

        nleft = net.select("target_left_encoder")(next_obs)
        a_next = self.sample_actions_train(net, next_obs, z, ar)
        tFs = net.select("target_forward")(nleft, z, a_next)
        tBg = net.select("target_backward")(next_obs)
        target_M = jax.lax.stop_gradient(jnp.min(jnp.einsum("pbz,cz->pbc", tFs, tBg), axis=0))

        diff = Ms - discount[None] * target_M[None]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()
        fb_offdiag = 0.5 * ((diff * off[None]) ** 2).sum() / off_sum
        fb_diag = -jax.vmap(jnp.diag)(diff).mean() * cfg["num_parallel"]
        Cov = Bg @ Bg.T
        orth = 0.5 * ((Cov * off) ** 2).sum() / off_sum - jnp.diag(Cov).mean()
        fb_loss = fb_offdiag + fb_diag + cfg["ortho_coef"] * orth

        aloss, ainfo = self._actor_loss(net, grad_params, obs, left, z, action, rng)
        loss = fb_loss + aloss
        info = dict(fb_loss=fb_loss, fb_offdiag=fb_offdiag, fb_diag=fb_diag, orth=orth, **ainfo)
        return loss, info

    def sample_actions_train(self, net, obs, z, rng):
        """next-action for the FB target (frozen actor params)."""
        if self.config["flow"]:
            noise = jax.random.normal(rng, (obs.shape[0], self.config["action_dim"]))
            return net.select("actor")(obs, z, noise)
        return jnp.clip(net.select("actor")(obs, z), -1.0, 1.0)

    def target_update(self, network, name):
        tau = self.config["tau"]
        new_t = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            self.network.params[f"modules_{name}"], self.network.params[f"modules_target_{name}"])
        network.params[f"modules_target_{name}"] = new_t

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        for name in ("forward", "backward", "left_encoder"):
            self.target_update(new_network, name)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, z=None, seed=None, temperature=1.0):
        del goals, temperature
        obs = jnp.asarray(observations, jnp.float32)
        single = obs.ndim == 1
        obs2 = obs[None] if single else obs
        z2 = jnp.broadcast_to(jnp.atleast_2d(z), (obs2.shape[0], self.config["z_dim"]))
        if self.config["flow"]:
            # FlowBC action head uses a FIXED seed (flow_seed=128) so the policy is
            # deterministic at eval -> reproducible success rate, no per-episode RNG luck.
            del seed
            noise = jax.random.normal(jax.random.PRNGKey(self.config["flow_seed"]),
                                      (obs2.shape[0], self.config["action_dim"]))
            a = self.network.select("actor")(obs2, z2, noise)
        else:
            a = self.network.select("actor")(obs2, z2)
        a = jnp.clip(a, -1.0, 1.0)
        return a[0] if single else a

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        act_dim = ex_actions.shape[-1]
        zc = config["z_dim"]
        config = dict(config)
        config["action_dim"] = act_dim
        config = ml_collections.FrozenConfigDict(config)

        ex_obs = jnp.asarray(ex_observations, jnp.float32)
        ex_act = jnp.asarray(ex_actions, jnp.float32)
        ex_left = jnp.zeros((ex_obs.shape[0], config["L_dim"]), jnp.float32)
        ex_z = jnp.zeros((ex_obs.shape[0], zc), jnp.float32)
        ex_t = jnp.zeros((ex_obs.shape[0], 1), jnp.float32)

        left_def = TrunkNet(config["hidden_dim"], config["enc_layers"], config["L_dim"])
        back_def = TrunkNet(config["hidden_dim"], config["enc_layers"], zc)
        net_info = dict(
            left_encoder=(left_def, (ex_obs,)),
            backward=(back_def, (ex_obs,)),
            forward=(_fwd_module(config, zc), (ex_left, ex_z, ex_act)),
            target_left_encoder=(copy.deepcopy(left_def), (ex_obs,)),
            target_backward=(copy.deepcopy(back_def), (ex_obs,)),
            target_forward=(_fwd_module(config, zc), (ex_left, ex_z, ex_act)),
        )
        if config["flow"]:
            net_info["actor"] = (GCNoiseActor(hidden_dim=config["hidden_dim"], hidden_layers=config["actor_layers"],
                                              embedding_layers=config["embedding_layers"]), (ex_obs, ex_z, ex_act))
            net_info["actor_vf"] = (GCVectorField(hidden_dim=config["hidden_dim"], hidden_layers=config["enc_layers"]),
                                    (ex_obs, ex_act, ex_t))
        else:
            net_info["actor"] = (Actor(config["hidden_dim"], config["actor_layers"],
                                       config["embedding_layers"], act_dim), (ex_obs, ex_z))

        networks = {k: v[0] for k, v in net_info.items()}
        args = {k: v[1] for k, v in net_info.items()}
        net_def = ModuleDict(networks)
        tx = optax.adam(learning_rate=config["lr"])
        params = net_def.init(init_rng, **args)["params"]
        for name in ("forward", "backward", "left_encoder"):
            params[f"modules_target_{name}"] = params[f"modules_{name}"]
        network = TrainState.create(net_def, params, tx=tx)
        return cls(rng=rng, network=network, config=config)


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name="fb",
        lr=1e-4, batch_size=256,
        z_dim=50, L_dim=50, discount=0.99, tau=0.005,
        train_goal_ratio=0.5, ortho_coef=1.0, num_parallel=2, bc_coeff=0.0,
        flow=0, flow_steps=10, flow_seed=128,   # fixed action-head noise seed (FlowBC)
        hidden_dim=512, enc_layers=4, fwd_layers=2, actor_layers=2, embedding_layers=2,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class="GCDataset",
        value_p_curgoal=0.0, value_p_trajgoal=1.0, value_p_randomgoal=0.0, value_geom_sample=1,
        actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0, actor_geom_sample=0,
        gc_negative=1, p_aug=ml_collections.config_dict.placeholder(float),
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))


import agents as _ogbench_agents  # noqa: E402
_ogbench_agents.agents["fb"] = FBAgent
