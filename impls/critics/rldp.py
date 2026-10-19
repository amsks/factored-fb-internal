"""RLDP critic: forward-backward with a self-predictive backward representation.

F is trained by the FB loss with B frozen; B by multi-step latent prediction.
"""

import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.fb_math import project_z
from impls.utils.networks import ForwardMap, PhiMap, PredictorMap


@dataclasses.dataclass(frozen=True)
class RLDPCritic:
    forward: Any
    backward: Any
    left_encoder: Any
    predictor: Any
    z_dim: int
    num_parallel: int
    horizon: int
    discount: float
    ortho_coef: float
    train_goal_ratio: float
    norm_z: bool
    f_tau: float
    b_tau: float

    GROUPS = ("forward", "backward", "left_encoder", "predictor")
    STAGES = ("fb",)
    PROVIDES_COND = frozenset({"z"})
    INFER_COND_MARGINAL = "next_observations"

    @property
    def TARGETS(self):
        # The predictor has no target network.
        return {
            "forward": self.f_tau,
            "left_encoder": self.f_tau,
            "backward": self.b_tau,
        }

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

    def _P(self, p, x, action):
        return self.predictor.apply({"params": p}, x, action)

    # Self-predictive loss
    def _sp_pairs(self, batch):
        """Actions a_t..a_{t+h-1} and reached states s_{t+1}..s_{t+h}."""
        fo, fa = batch["future_observations"], batch["future_actions"]
        assert fo.shape[1] == fa.shape[1] == self.horizon - 1, (
            f"windowed batch has {fo.shape[1]} future columns but the critic was built "
            f"with horizon={self.horizon} (expects {self.horizon - 1})"
        )
        actions = [batch["actions"]] + [fa[:, i] for i in range(fa.shape[1])]
        targets = [batch["next_observations"]] + [fo[:, i] for i in range(fo.shape[1])]
        return actions, targets

    def _sp_loss(self, params, target, batch):
        """Self-predictive loss and B(s_t)."""
        actions, targets = self._sp_pairs(batch)
        P = self.num_parallel
        ys = [jax.lax.stop_gradient(self._B(target["backward"], s)) for s in targets]

        # Rollout from the online B(s_t), broadcast over predictor heads
        B_pred = self._B(params["backward"], batch["observations"])
        curr = jnp.broadcast_to(B_pred[None], (P,) + B_pred.shape)

        sp = 0.0
        for a_t, y_t in zip(actions, ys):
            curr = self._P(params["predictor"], curr, a_t)
            sp = sp + jnp.mean(jnp.sum((curr - y_t[None]) ** 2, axis=-1))
        return sp / len(actions), B_pred

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "fb"
        obs, action, next_obs = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
        )
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
        target_M = jax.lax.stop_gradient(
            jnp.einsum("pbz,cz->pbc", tFs, tB).mean(axis=0)
        )

        # Successor loss trains F only
        le = self._le(params["left_encoder"], obs)
        Fs = self._F(params["forward"], le, z, action)
        Bg = jax.lax.stop_gradient(self._B(params["backward"], goal))
        M = jnp.einsum("pbz,cz->pbc", Fs, Bg)
        diff = M - disc * target_M
        fb_off = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        fb_diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * P
        fb = fb_off + fb_diag

        sp, B_pred = self._sp_loss(params, target, batch)

        # Orthonormality on B(s_t)
        cov = B_pred @ B_pred.T
        orth_off = 0.5 * jnp.sum((cov * off) ** 2) / off_sum
        orth_diag = -jnp.mean(jnp.diagonal(cov))
        orth = orth_off + orth_diag

        loss = fb + sp + self.ortho_coef * orth
        return loss, {
            "fb_loss": loss,
            "fb_offdiag": fb_off,
            "fb_diag": fb_diag,
            "ortho_loss": orth,
            "ortho_offdiag": orth_off,
            "ortho_diag": orth_diag,
            "sp_loss": sp,
        }

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
            self._B(net["backward"].params, next_obs)[
                jax.random.permutation(r_perm, B)
            ],
            self.norm_z,
        )
        mix_mask = (jax.random.uniform(r_mask, (B,)) < self.train_goal_ratio)[:, None]
        z = jnp.where(mix_mask, goals, z_gauss)
        return {"z": z}

    def infer_cond(self, net, obs, rewards):
        """z = mean_i r_i B(s_i), normalised."""
        Bm = self._B(net["backward"].params, obs)
        z = (rewards.reshape(1, -1) @ Bm).reshape(-1) / Bm.shape[0]
        return {"z": project_z(z, self.norm_z)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr

        z_dim, L_dim, P = config["z_dim"], config["L_dim"], config["num_parallel"]
        horizon = int(config["horizon"])
        assert horizon >= 2, (
            f"RLDP needs a window of at least 2 steps, got horizon={horizon}"
        )
        forward = ForwardMap(
            z_dim=z_dim,
            hidden_dim=config["forward"]["hidden_dim"],
            num_parallel=P,
            hidden_layers=config["forward"]["hidden_layers"],
            embedding_layers=config["forward"]["embedding_layers"],
        )
        backward = PhiMap(
            z_dim=z_dim,
            hidden_dim=config["backward"]["hidden_dim"],
            hidden_layers=config["backward"]["hidden_layers"],
            norm=config["backward"]["norm"],
        )
        left_encoder = PhiMap(
            z_dim=L_dim,
            hidden_dim=config["left_encoder"]["hidden_dim"],
            hidden_layers=config["left_encoder"]["hidden_layers"],
            norm=config["left_encoder"]["norm"],
        )
        predictor = PredictorMap(
            out_dim=z_dim,
            hidden_dim=config["predictor"]["hidden_dim"],
            num_parallel=P,
            hidden_layers=config["predictor"]["hidden_layers"],
            embedding_layers=config["predictor"]["embedding_layers"],
        )
        rf, rb, rle, rp = jr.split(jr.PRNGKey(config["seed"]), 4)
        ex_feat = jnp.zeros((ex_obs.shape[0], L_dim))
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        ex_pred_in = jnp.zeros((P, ex_obs.shape[0], z_dim))
        params = {
            "left_encoder": left_encoder.init(rle, ex_obs)["params"],
            "backward": backward.init(rb, ex_obs)["params"],
            "forward": forward.init(rf, ex_feat, ex_z, ex_actions)["params"],
            "predictor": predictor.init(rp, ex_pred_in, ex_actions)["params"],
        }
        critic = cls(
            forward=forward,
            backward=backward,
            left_encoder=left_encoder,
            predictor=predictor,
            z_dim=z_dim,
            num_parallel=P,
            horizon=horizon,
            discount=config["discount"],
            ortho_coef=config["ortho_coef"],
            train_goal_ratio=config["train_goal_ratio"],
            norm_z=config["norm_z"],
            f_tau=config["f_target_tau"],
            b_tau=config["b_target_tau"],
        )
        targets = {g: copy.deepcopy(params[g]) for g in critic.TARGETS}
        return critic, params, targets
