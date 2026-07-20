"""campaign/jax_agents/fb.py — JAX/Flax port of the torch FB agent (agents/fb/), for the
unified jax pipeline. Faithful to the torch logic:

  goal = next_obs; z = half random (sqrt(d)*normalize(gaussian)) / half project(B(next_obs[perm])).
  Ms = F(left_enc(obs), z, action) . B(next_obs).T           [P, B, B]
  target_M = min_P F_tgt(left_enc_tgt(next_obs), z, a') . B_tgt(next_obs).T   (a' = actor(next_obs,z))
  diff = Ms - gamma*target_M
  fb = 0.5*(diff*offdiag)^2/offdiag_sum  -  diag(diff).mean*P
  ortho = 0.5*(BB^T*offdiag)^2/offdiag_sum - diag(BB^T).mean
  actor (TD3): -min_P (F(left_enc,z,a_pi).z).mean  (+ bc_coeff*mse when bc_coeff>0)
  soft targets tau on forward/backward/left_encoder (target modules, gciql-style).

Registered into agents.agents['fb'] so jax_train_unified.py / eval_jax.py drive it exactly
like the OGBench agents. Reads observations/actions/next_observations/terminals from the
(GC)Dataset batch and ignores the goal-conditioned fields.
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


# --------------------------------------------------------------------- networks
class SimpleEmbedding(nn.Module):
    """torch simple_embedding: Dense+LN+Tanh, (emb_layers-1)x[Dense+ReLU], Dense(hidden//2)+ReLU."""
    hidden_dim: int
    embedding_layers: int

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(nn.LayerNorm()(nn.Dense(self.hidden_dim)(x)))
        for _ in range(self.embedding_layers - 1):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.relu(nn.Dense(self.hidden_dim // 2)(x))


class TrunkNet(nn.Module):
    """torch backward/left_encoder: Dense+LN+Tanh, (layers-1)x[Dense+ReLU], Dense(out)."""
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
    """F(obs=left_enc, z, action) -> z_dim. Split embeddings embed_z(obs,z)/embed_sa(obs,action)."""
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
    """TD3 actor mean: embed_z(obs,z)/embed_s(obs) -> trunk -> action_dim (unbounded mean)."""
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        cfg = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        term = batch["terminals"] if "terminals" in batch else jnp.zeros(obs.shape[0])
        B = obs.shape[0]
        discount = (cfg["discount"] * (1.0 - term)).reshape(-1, 1)
        rng = rng if rng is not None else self.rng
        rng, zr, pr, mr = jax.random.split(rng, 4)
        net = self.network

        # z: half random, half project(B(next_obs[perm]))  (backward has grad, matches torch)
        z_rand = self.sample_z(zr, B)
        perm = jax.random.permutation(pr, B)
        z_goal = self.project_z(net.select("backward")(next_obs[perm], params=grad_params))
        mask = jax.random.uniform(mr, (B, 1)) < cfg["train_goal_ratio"]
        z = jnp.where(mask, z_goal, z_rand)

        # online successor Ms = F(left_enc(obs), z, a) . B(next_obs).T
        left = net.select("left_encoder")(obs, params=grad_params)
        Fs = net.select("forward")(left, z, action, params=grad_params)          # [P, B, z]
        Bg = net.select("backward")(next_obs, params=grad_params)                # [B, z]
        Ms = jnp.einsum("pbz,cz->pbc", Fs, Bg)                                    # [P, B, B]

        # target (detached): a' = online actor(next_obs, z) [no grad via stored params]
        nleft = net.select("target_left_encoder")(next_obs)
        a_next = jnp.clip(net.select("actor")(next_obs, z), -1.0, 1.0)
        tFs = net.select("target_forward")(nleft, z, a_next)
        tBg = net.select("target_backward")(next_obs)
        target_M = jax.lax.stop_gradient(jnp.min(jnp.einsum("pbz,cz->pbc", tFs, tBg), axis=0))

        diff = Ms - discount[None] * target_M[None]                              # [P, B, B]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()
        fb_offdiag = 0.5 * ((diff * off[None]) ** 2).sum() / off_sum
        fb_diag = -jax.vmap(jnp.diag)(diff).mean() * cfg["num_parallel"]
        Cov = Bg @ Bg.T
        orth = 0.5 * ((Cov * off) ** 2).sum() / off_sum - jnp.diag(Cov).mean()
        fb_loss = fb_offdiag + fb_diag + cfg["ortho_coef"] * orth

        # TD3 actor: maximize min_P (F(left_enc, z, a_pi) . z). The forward map is
        # evaluated at FROZEN params (no params=grad_params) so -Q updates ONLY the actor
        # (torch uses a separate actor optimizer); otherwise -Q inflates F -> divergence.
        a_pi = jnp.clip(net.select("actor")(obs, z, params=grad_params), -1.0, 1.0)
        Fpi = net.select("forward")(jax.lax.stop_gradient(left), z, a_pi)
        Qs = (Fpi * z[None]).sum(-1)                                             # [P, B]
        Q = jnp.min(Qs, axis=0)
        actor_loss = -Q.mean()
        bc = 0.0
        if cfg["bc_coeff"] > 0:
            bc = jnp.mean((a_pi - action) ** 2)
            actor_loss = actor_loss / (jnp.abs(Qs).mean() + 1e-6) + cfg["bc_coeff"] * bc

        loss = fb_loss + actor_loss
        return loss, dict(fb_loss=fb_loss, fb_offdiag=fb_offdiag, fb_diag=fb_diag, orth=orth,
                          actor_loss=actor_loss, q=Q.mean(), bc=bc)

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
        del goals, seed, temperature
        obs = jnp.asarray(observations, jnp.float32)
        single = obs.ndim == 1
        obs2 = obs[None] if single else obs
        z2 = jnp.atleast_2d(z)
        z2 = jnp.broadcast_to(z2, (obs2.shape[0], z2.shape[-1]))
        a = jnp.clip(self.network.select("actor")(obs2, z2), -1.0, 1.0)
        return a[0] if single else a

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        act_dim = ex_actions.shape[-1]
        zc = config["z_dim"]
        ex_obs = jnp.asarray(ex_observations, jnp.float32)
        ex_act = jnp.asarray(ex_actions, jnp.float32)
        ex_left = jnp.zeros((ex_obs.shape[0], config["L_dim"]), jnp.float32)
        ex_z = jnp.zeros((ex_obs.shape[0], zc), jnp.float32)

        left_def = TrunkNet(config["hidden_dim"], config["enc_layers"], config["L_dim"])
        back_def = TrunkNet(config["hidden_dim"], config["enc_layers"], zc)
        actor_def = Actor(config["hidden_dim"], config["actor_layers"], config["embedding_layers"], act_dim)

        net_info = dict(
            left_encoder=(left_def, (ex_obs,)),
            backward=(back_def, (ex_obs,)),
            forward=(_fwd_module(config, zc), (ex_left, ex_z, ex_act)),
            actor=(actor_def, (ex_obs, ex_z)),
            target_left_encoder=(copy.deepcopy(left_def), (ex_obs,)),
            target_backward=(copy.deepcopy(back_def), (ex_obs,)),
            target_forward=(_fwd_module(config, zc), (ex_left, ex_z, ex_act)),
        )
        networks = {k: v[0] for k, v in net_info.items()}
        args = {k: v[1] for k, v in net_info.items()}
        net_def = ModuleDict(networks)
        tx = optax.adam(learning_rate=config["lr"])
        params = net_def.init(init_rng, **args)["params"]
        for name in ("forward", "backward", "left_encoder"):
            params[f"modules_target_{name}"] = params[f"modules_{name}"]
        network = TrainState.create(net_def, params, tx=tx)
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name="fb",
        lr=1e-4, batch_size=256,
        z_dim=50, L_dim=50, discount=0.99, tau=0.005,
        train_goal_ratio=0.5, ortho_coef=1.0, num_parallel=2, bc_coeff=0.0,
        hidden_dim=512, enc_layers=4, fwd_layers=2, actor_layers=2, embedding_layers=2,
        encoder=ml_collections.config_dict.placeholder(str),
        dataset_class="GCDataset",
        value_p_curgoal=0.0, value_p_trajgoal=1.0, value_p_randomgoal=0.0, value_geom_sample=1,
        actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0, actor_geom_sample=0,
        gc_negative=1, p_aug=ml_collections.config_dict.placeholder(float),
        frame_stack=ml_collections.config_dict.placeholder(int),
    ))


# self-register so agents.agents['fb'] resolves (jax_train_unified / eval_jax use it).
import agents as _ogbench_agents  # noqa: E402
_ogbench_agents.agents["fb"] = FBAgent
