"""HILP critic: Hilbert-representation successor features (Park et al., 2024).

Stage phi fits V(s, g) = -||phi(s) - phi(g)||; stage sf fits successor features of phi.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from impls.utils.fb_math import project_z
from impls.utils.networks import ForwardMap, PhiMap


def hilp_value(phi_a, phi_b):
    """v = -||phi(a) - phi(b)||, clamped to keep the gradient finite at zero distance."""
    sq = jnp.sum((phi_a - phi_b) ** 2, axis=-1)
    return -jnp.sqrt(jnp.clip(sq, 1e-6, None))


def expectile_loss(adv, diff, expectile):
    """Expectile-weighted squared error."""
    weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * diff**2


def hilp_phi_loss(
    phi1_p,
    phi2_p,
    tphi1_p,
    tphi2_p,
    apply_phi,
    obs,
    goals,
    next_obs,
    discount,
    expectile,
):
    """Phi value loss; each online head regresses its own target head's TD target."""
    same = (jnp.linalg.norm(obs - goals, axis=-1) < 1e-6).astype(obs.dtype)
    masks = 1.0 - same
    rewards = same - 1.0

    tn1 = jax.lax.stop_gradient(
        hilp_value(apply_phi(tphi1_p, next_obs), apply_phi(tphi1_p, goals))
    )
    tn2 = jax.lax.stop_gradient(
        hilp_value(apply_phi(tphi2_p, next_obs), apply_phi(tphi2_p, goals))
    )
    tv1 = jax.lax.stop_gradient(
        hilp_value(apply_phi(tphi1_p, obs), apply_phi(tphi1_p, goals))
    )
    tv2 = jax.lax.stop_gradient(
        hilp_value(apply_phi(tphi2_p, obs), apply_phi(tphi2_p, goals))
    )

    next_v = jnp.minimum(tn1, tn2)
    q = rewards + discount * masks * next_v
    v_t = (tv1 + tv2) / 2.0
    adv = q - v_t

    q1 = rewards + discount * masks * tn1
    q2 = rewards + discount * masks * tn2
    v1 = hilp_value(apply_phi(phi1_p, obs), apply_phi(phi1_p, goals))
    v2 = hilp_value(apply_phi(phi2_p, obs), apply_phi(phi2_p, goals))

    loss = jnp.mean(expectile_loss(adv, q1 - v1, expectile)) + jnp.mean(
        expectile_loss(adv, q2 - v2, expectile)
    )
    info = {
        "hilp_value_loss": loss,
        "hilp_v_mean": jnp.mean((v1 + v2) / 2.0),
        "hilp_adv_mean": jnp.mean(adv),
        "hilp_abs_adv_mean": jnp.mean(jnp.abs(adv)),
        "hilp_accept_prob": jnp.mean((adv >= 0).astype(loss.dtype)),
    }
    return loss, info


@dataclasses.dataclass(frozen=True)
class HILPCritic:
    # Field names are the parameter group names
    hilp_phi1: Any
    hilp_phi2: Any
    hilp_sf: Any
    hilp_stats: Any
    hilp_le: Any
    z_dim: int
    num_parallel: int
    discount: float
    phi_discount: float
    expectile: float
    phi_tau: float
    sf_tau: float
    norm_z: bool
    stats_momentum: float
    mix_ratio: float
    sf_loss_scale: float

    GROUPS = ("hilp_phi1", "hilp_phi2", "hilp_sf", "hilp_stats", "hilp_le")
    STAGES = ("phi", "sf")
    PROVIDES_COND = frozenset({"z"})
    INFER_COND_MARGINAL = "next_observations"
    BOOT_COND = "actor_goal"

    @property
    def TARGETS(self):
        # hilp_stats is written by update_stats, not polyak-averaged
        return {
            "hilp_phi1": self.phi_tau,
            "hilp_phi2": self.phi_tau,
            "hilp_sf": self.sf_tau,
            "hilp_stats": 0.0,
            "hilp_le": self.sf_tau,
        }

    @property
    def cvec_dim(self):
        return self.z_dim

    def stage_groups(self, stage):
        if stage == "phi":
            return ("hilp_phi1", "hilp_phi2")
        assert stage == "sf", stage
        return ("hilp_sf", "hilp_le")

    # Towers
    def _phi(self, p, x):
        return self.hilp_phi1.apply({"params": p}, x)

    def _le(self, container, x):
        return self.hilp_le.apply({"params": container["hilp_le"]}, x)

    def _sf(self, p, obs, z, action):
        return self.hilp_sf.apply({"params": p}, obs, z, action)

    def _feature_net(self, net_or_params, stats_params, x):
        """Centered feature phi1(x) - running mean."""
        return self._phi(net_or_params, x) - stats_params["mean"]

    # Losses
    def stage_loss(self, stage, params, target, batch, cond, boot):
        if stage == "phi":
            return hilp_phi_loss(
                params["hilp_phi1"],
                params["hilp_phi2"],
                target["hilp_phi1"],
                target["hilp_phi2"],
                lambda p, x: self._phi(p, x),
                batch["observations"],
                batch["hilp_future_observations"],
                batch["next_observations"],
                self.phi_discount,
                self.expectile,
            )

        assert stage == "sf", stage
        obs, action = batch["observations"], batch["actions"]
        nobs, z = batch["next_observations"], cond["z"]

        tF = self._sf(
            target["hilp_sf"], self._le(target, nobs), z, boot["next_action"]
        )  # [P, B, d]
        # Immediate term uses the online phi, stop-gradient
        target_phi = jax.lax.stop_gradient(
            self._feature_net(params["hilp_phi1"], params["hilp_stats"], nobs)
        )  # [B, d]
        next_F = jnp.mean(tF, axis=0)  # [B, d]
        target_F = jax.lax.stop_gradient(target_phi + self.discount * next_F)

        F = self._sf(params["hilp_sf"], self._le(params, obs), z, action)  # [P, B, d]
        sf_loss = jnp.sum(jnp.mean((F - target_F[None]) ** 2, axis=(-1, -2)))
        sf_loss = self.sf_loss_scale * sf_loss
        return sf_loss, {
            "sf_loss": sf_loss,
            "target_F": jnp.mean(target_F),
            "phi_norm": jnp.mean(jnp.linalg.norm(target_phi, axis=-1)),
            "z_norm": jnp.mean(jnp.linalg.norm(z, axis=-1)),
        }

    # Control
    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """Q(s, a, z) = F(s, a, z)^T z, averaged over heads."""
        z = cond["z"]
        Qs = jnp.einsum(
            "pbd,bd->pb",
            self._sf(params["hilp_sf"], self._le(params, obs), z, actions),
            z,
        )
        return jnp.mean(Qs, axis=0), Qs

    def cond_vec(self, cond):
        return cond["z"]

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,))}

    def map_eval_cond(self, net, cond, observations=None):
        """Map an evaluation goal to z = normalise(phi(g) - phi(s))."""
        if "goal" not in cond:
            return cond
        if observations is None:
            raise ValueError(
                "HILP's goal->z map is joint in goal AND current observation "
                "(z = normalize(phi(g) - phi(s))); pass observations=."
            )
        p, st = net["hilp_phi1"].params, net["hilp_stats"].params
        z = self._feature_net(p, st, cond["goal"]) - self._feature_net(
            p, st, observations
        )
        return {**cond, "z": project_z(z, self.norm_z)}

    def sample_cond(self, net, batch, rng):
        """Mix random sphere directions with whitened features phi(s') Cov(phi)^-1."""
        B = batch["observations"].shape[0]
        r_g, r_perm, r_mask = jax.random.split(rng, 3)
        z = project_z(jax.random.normal(r_g, (B, self.z_dim)), self.norm_z)
        nobs = batch["next_observations"]
        phi = jax.lax.stop_gradient(
            self._feature_net(
                net["hilp_phi1"].params,
                net["hilp_stats"].params,
                nobs[jax.random.permutation(r_perm, B)],
            )
        )  # [B, d]
        cov = (phi.T @ phi) / phi.shape[0]
        inv_cov = jnp.linalg.pinv(cov)
        new_z = project_z(phi @ inv_cov, self.norm_z)
        mix = (jax.random.uniform(r_mask, (B,)) < self.mix_ratio)[:, None]
        return {"z": jnp.where(mix, new_z, z)}

    def infer_cond(self, net, obs, rewards):
        """z = argmin_z ||phi(s') z - r||^2, normalised."""
        phi = self._feature_net(
            net["hilp_phi1"].params, net["hilp_stats"].params, jnp.asarray(obs)
        )
        r = jnp.asarray(rewards).reshape(-1, 1)
        z = jnp.linalg.lstsq(phi, r, rcond=None)[0].reshape(-1)
        return {"z": project_z(z, self.norm_z)}

    def update_stats(self, net, pre_net, batch):
        """EMA of the phi mean and std from the pre-step online tower."""
        m = self.stats_momentum
        p = net["hilp_stats"].params
        phi1 = self._phi(pre_net["hilp_phi1"].params, batch["observations"])
        net = dict(net)
        net["hilp_stats"] = net["hilp_stats"].replace(
            params={
                "mean": p["mean"] * (1.0 - m) + jnp.mean(phi1, axis=0) * m,
                "std": p["std"] * (1.0 - m) + jnp.std(phi1, axis=0) * m,
            }
        )
        return net


class _HILPStats(nn.Module):
    """Running phi mean and std, stored as parameters; written only by `update_stats`."""

    dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self):
        mean = self.param("mean", lambda _k: jnp.zeros((self.dim,), self.dtype))
        std = self.param("std", lambda _k: jnp.ones((self.dim,), self.dtype))
        return mean, std


def _hilp_build(cls, ex_obs, ex_actions, config):
    import jax.random as jr

    z_dim = int(config["z_dim"])
    P = int(config["num_parallel"])
    L_dim = int(config["L_dim"])

    mk_phi = lambda: PhiMap(
        z_dim=z_dim,
        hidden_dim=config["phi"]["hidden_dim"],
        hidden_layers=config["phi"]["hidden_layers"],
        norm=False,
    )
    phi1, phi2 = mk_phi(), mk_phi()
    sf = ForwardMap(
        z_dim=z_dim,
        hidden_dim=config["sf"]["hidden_dim"],
        num_parallel=P,
        hidden_layers=config["sf"]["hidden_layers"],
        embedding_layers=config["sf"]["embedding_layers"],
    )
    le_cfg = config["left_encoder"]
    le = PhiMap(
        z_dim=L_dim,
        hidden_dim=int(le_cfg["hidden_dim"]),
        hidden_layers=int(le_cfg["hidden_layers"]),
        norm=bool(le_cfg["norm"]),
    )
    stats_mod = _HILPStats(dim=z_dim, dtype=jnp.zeros((), ex_obs.dtype).dtype)

    r1, r2, rs, rst = jr.split(jr.PRNGKey(config["seed"]), 4)
    ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
    ex_sf_in = jnp.zeros((ex_obs.shape[0], L_dim), ex_obs.dtype)
    params = {
        "hilp_phi1": phi1.init(r1, ex_obs)["params"],
        "hilp_phi2": phi2.init(r2, ex_obs)["params"],
        "hilp_sf": sf.init(rs, ex_sf_in, ex_z, ex_actions)["params"],
        "hilp_stats": stats_mod.init(rst)["params"],
        "hilp_le": le.init(jr.PRNGKey(config["seed"] + 913), ex_obs)["params"],
    }
    critic = cls(
        hilp_phi1=phi1,
        hilp_phi2=phi2,
        hilp_sf=sf,
        hilp_stats=stats_mod,
        hilp_le=le,
        z_dim=z_dim,
        num_parallel=P,
        discount=float(config["discount"]),
        phi_discount=float(config["phi_discount"]),
        expectile=float(config["expectile"]),
        phi_tau=float(config["phi_target_tau"]),
        sf_tau=float(config["sf_target_tau"]),
        norm_z=bool(config["norm_z"]),
        stats_momentum=float(config["phi_stats_tau"]),
        mix_ratio=float(config["train_goal_ratio"]),
        sf_loss_scale=float(config["sf_loss_scale"]),
    )
    targets = {g: copy.deepcopy(params[g]) for g in critic.TARGETS}
    return critic, params, targets


HILPCritic.build = classmethod(_hilp_build)
