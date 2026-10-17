"""Forward-backward successor-measure critic (Touati & Ollivier).

M(s, a, s'; z) = F(s, a, z)^T B(s'); the actor maximises Q(s, a, z) = F(s, a, z)^T z.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z


@dataclasses.dataclass(frozen=True)
class FBCritic:
    forward: Any
    backward: Any
    left_encoder: Any
    z_dim: int
    num_parallel: int
    discount: float
    ortho_coef: float
    train_goal_ratio: float
    norm_z: bool
    f_tau: float
    b_tau: float

    GROUPS = ("forward", "backward", "left_encoder")
    STAGES = ("fb",)
    PROVIDES_COND = frozenset({"z"})
    INFER_COND_MARGINAL = "next_observations"

    @property
    def TARGETS(self):
        return {"forward": self.f_tau, "left_encoder": self.f_tau, "backward": self.b_tau}

    @property
    def cvec_dim(self):
        return self.z_dim

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,), jnp.float64)}

    def map_eval_cond(self, net, cond, observations=None):
        """Map an evaluation goal to z = B(goal)."""
        if "goal" not in cond:
            return cond
        z = project_z(self._B(net["backward"].params, cond["goal"]), self.norm_z)
        return {**cond, "z": z}

    def stage_groups(self, stage):
        assert stage == "fb"
        return self.GROUPS

    def _F(self, p, obs_feat, z, action):
        return self.forward.apply({"params": p}, obs_feat, z, action)

    def _B(self, p, x):
        return self.backward.apply({"params": p}, x)

    def _le(self, p, x):
        return self.left_encoder.apply({"params": p}, x)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "fb"
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs
        disc, P = self.discount, self.num_parallel
        z, next_action = cond["z"], boot["next_action"]
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()

        # TD target
        nle = self._le(target["left_encoder"], next_obs)
        tFs = self._F(target["forward"], nle, z, next_action)
        tB = self._B(target["backward"], goal)
        target_M = jax.lax.stop_gradient(jnp.einsum("pbz,cz->pbc", tFs, tB).mean(axis=0))

        le = self._le(params["left_encoder"], obs)
        Fs = self._F(params["forward"], le, z, action)
        Bm = self._B(params["backward"], goal)
        M = jnp.einsum("pbz,cz->pbc", Fs, Bm)
        diff = M - disc * target_M
        fb_off = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        fb_diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * P
        cov = Bm @ Bm.T
        orth_off = 0.5 * jnp.sum((cov * off) ** 2) / off_sum
        orth_diag = -jnp.mean(jnp.diagonal(cov))
        orth = orth_off + orth_diag
        loss = fb_off + fb_diag + self.ortho_coef * orth
        return loss, {"fb_loss": loss, "fb_offdiag": fb_off, "fb_diag": fb_diag, "ortho_loss": orth}

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """Q(s, a, z) = F(s, a, z)^T z, averaged over the F heads."""
        z = cond["z"]
        le = self._le(params["left_encoder"], obs)
        if stop_encoder_grad:
            le = jax.lax.stop_gradient(le)
        Fs = self._F(params["forward"], le, z, actions)
        Qs = (Fs * z).sum(-1)
        return Qs.mean(axis=0), Qs

    def cond_vec(self, cond):
        return cond["z"]

    def sample_cond(self, net, batch, rng):
        """Mix B(s') at shuffled next states with random sphere directions."""
        next_obs = batch["next_observations"]
        B = next_obs.shape[0]
        r_z, r_mask, r_perm = jax.random.split(rng, 3)
        z_gauss = project_z(jax.random.normal(r_z, (B, self.z_dim)), self.norm_z)
        goals = project_z(
            self._B(net["backward"].params, next_obs)[jax.random.permutation(r_perm, B)],
            self.norm_z,
        )
        mix_mask = (jax.random.uniform(r_mask, (B,)) < self.train_goal_ratio)[:, None]
        z = jnp.where(mix_mask, goals, z_gauss)
        return {"z": z}

    def infer_cond(self, net, obs, rewards):
        """z = sum_i r_i B(s_i), normalised."""
        Bm = self._B(net["backward"].params, obs)
        z = (rewards.reshape(1, -1) @ Bm).reshape(-1)
        return {"z": project_z(z, self.norm_z)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        z_dim, L_dim, P = config["z_dim"], config["L_dim"], config["num_parallel"]
        forward = ForwardMap(z_dim=z_dim, hidden_dim=config["forward"]["hidden_dim"],
                             num_parallel=P, hidden_layers=config["forward"]["hidden_layers"],
                             embedding_layers=config["forward"]["embedding_layers"])
        backward = PhiMap(z_dim=z_dim, hidden_dim=config["backward"]["hidden_dim"],
                          hidden_layers=config["backward"]["hidden_layers"], norm=config["backward"]["norm"])
        left_encoder = PhiMap(z_dim=L_dim, hidden_dim=config["left_encoder"]["hidden_dim"],
                              hidden_layers=config["left_encoder"]["hidden_layers"],
                              norm=config["left_encoder"]["norm"])
        rf, rb, rle = jr.split(jr.PRNGKey(config["seed"]), 3)
        ex_feat = jnp.zeros((ex_obs.shape[0], L_dim))
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        params = {
            "left_encoder": left_encoder.init(rle, ex_obs)["params"],
            "backward": backward.init(rb, ex_obs)["params"],
            "forward": forward.init(rf, ex_feat, ex_z, ex_actions)["params"],
        }
        targets = {g: copy.deepcopy(params[g]) for g in cls.GROUPS}
        critic = cls(
            forward=forward, backward=backward, left_encoder=left_encoder,
            z_dim=z_dim, num_parallel=P, discount=config["discount"],
            ortho_coef=config["ortho_coef"],
            train_goal_ratio=config["train_goal_ratio"], norm_z=config["norm_z"],
            f_tau=config["f_target_tau"], b_tau=config["b_target_tau"],
        )
        return critic, params, targets
