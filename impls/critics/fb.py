"""FBCritic — Forward-Backward successor-measure critic on the factored seam.

Math transcribed verbatim from PSMFlows agents/fb.py (_fb_loss_fn, _draw_injection z-mix,
_actor_loss_fn q_of, infer_z); restructured onto per-group TrainState params.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z, targets_uncertainty


@dataclasses.dataclass(frozen=True)
class FBCritic:
    forward: Any
    backward: Any
    left_encoder: Any
    z_dim: int
    num_parallel: int
    discount: float
    ortho_coef: float
    fb_pessimism_penalty: float
    train_goal_ratio: float
    norm_z: bool
    f_tau: float
    b_tau: float

    GROUPS = ("forward", "backward", "left_encoder")
    STAGES = ("fb",)
    PROVIDES_V = False
    PROVIDES_COND = frozenset({"z"})  # sample_cond() produces {"z": ...}
    # infer_cond's base measure: the SUCCESSOR marginal. This critic's backward tower is
    # trained on `goal = next_obs` (stage_loss) and sample_cond mixes in B(next_observations),
    # so the reward->z readout must be solved on the same rows, paired with the reward of
    # that same transition. See impls/critics/base.py::INFER_COND_MARGINALS.
    INFER_COND_MARGINAL = "next_observations"

    @property
    def TARGETS(self):
        return {"forward": self.f_tau, "left_encoder": self.f_tau, "backward": self.b_tau}

    @property
    def cvec_dim(self):
        return self.z_dim

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,), jnp.float64)}

    def map_eval_cond(self, net, cond):
        """Eval-time hook: embed the env's goal frame into the z the actor was TRAINED on,
        z = project_z(B(goal)) — the exact conditioning sample_cond mixes in via
        train_goal_ratio. Without this hook eval runs on eval_cond0's zero z, for which
        Q(s,a;0) = <F,0> ≡ 0 and the policy is goal-blind (2026-08-17 FB-zero bug)."""
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
        goal = next_obs  # bw_encoder = Identity for state
        disc, P = self.discount, self.num_parallel
        z, next_action = cond["z"], boot["next_action"]
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()

        # target measure (stop-grad): target nets on next_obs.
        nle = self._le(target["left_encoder"], next_obs)
        tFs = self._F(target["forward"], nle, z, next_action)
        tB = self._B(target["backward"], goal)
        tM = jnp.einsum("pbz,cz->pbc", tFs, tB)
        tmean, tunc = targets_uncertainty(tM, P)
        target_M = jax.lax.stop_gradient(tmean - self.fb_pessimism_penalty * tunc)

        # online measure.
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
        z = cond["z"]
        le = self._le(params["left_encoder"], obs)
        if stop_encoder_grad:
            le = jax.lax.stop_gradient(le)
        Fs = self._F(params["forward"], le, z, actions)
        Qs = (Fs * z).sum(-1)
        qmean, _ = targets_uncertainty(Qs, self.num_parallel)
        # actor_pessimism_penalty is 0 in scope; keep the mean-of-heads reduction.
        return qmean, Qs

    def cond_vec(self, cond):
        """The vector actors concatenate onto obs to condition. FB: the latent z."""
        return cond["z"]

    def sample_cond(self, net, batch, rng):
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
        Bm = self._B(net["backward"].params, obs)
        z = (rewards.reshape(1, -1) @ Bm).reshape(-1) / Bm.shape[0]
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
            ortho_coef=config["ortho_coef"], fb_pessimism_penalty=config["fb_pessimism_penalty"],
            train_goal_ratio=config["train_goal_ratio"], norm_z=config["norm_z"],
            f_tau=config["f_target_tau"], b_tau=config["b_target_tau"],
        )
        return critic, params, targets
