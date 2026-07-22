"""PSMCritic — Proto Successor Measure critic on the factored seam.

PSM is FB with a DUAL-psi decomposition of the successor measure M(s,a,z) = <psi(s,z,a), phi(g)>:
  * psm_psi  — the PROTO successor features (bootstrapped from a behavior next-action),
  * sf_psi   — the SF successor features (bootstrapped from the LEARNED policy's next-action),
  * phi      — the shared basis / backward embedding (orthonormalized in the PSM stage).
Each psi is trained with the SAME FB successor-measure loss as impls/critics/fb.py
(offdiag 0.5(M-γM_t)^2, diag -diagonal.mean * P), the psm stage additionally orthonormalizes
phi; the sf stage trains sf_psi against phi.detach() (phi is a proto stage's responsibility).
The actor's Q uses the SF features: Q(s,a,z) = mean_P <sf_psi(s,z,a), z>.

Math transcribed from agents/psm (torch, feat/psm-integration) _update_psm / _update_sf, on
the FB seam. SIMPLIFICATION (documented): the torch psm stage draws its next-action from a
separate `proto_sampler` behavior policy; here BOTH stages use the actor's bootstrap
next-action (boot["next_action"]) -- the seam's single behavior hook. z-critic like FB.

VERIFICATION: unlike crl/qrl (ported from JAX sources), this is a torch->JAX transcription
with no JAX golden reference; it is structurally faithful to FB (which IS 1e-10 parity-locked)
but the PSM-specific dual-psi coupling is validated only by train/eval behavior, not a fixture.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z, targets_uncertainty


@dataclasses.dataclass(frozen=True)
class PSMCritic:
    psm_psi: Any
    sf_psi: Any
    phi: Any
    z_dim: int
    num_parallel: int
    discount: float
    ortho_coef: float
    pessimism: float
    train_goal_ratio: float
    norm_z: bool
    tau: float

    GROUPS = ("phi", "psm_psi", "sf_psi")
    STAGES = ("psm", "sf")
    PROVIDES_V = False
    PROVIDES_COND = frozenset({"z"})

    @property
    def TARGETS(self):
        return {"phi": self.tau, "psm_psi": self.tau, "sf_psi": self.tau}

    @property
    def cvec_dim(self):
        return self.z_dim

    def stage_groups(self, stage):
        return ("psm_psi", "phi") if stage == "psm" else ("sf_psi",)

    # --- module apply helpers ---
    def _psi(self, net, p, obs, z, action):
        return net.apply({"params": p}, obs, z, action)      # (P, B, z_dim)

    def _phi(self, p, x):
        return self.phi.apply({"params": p}, x)              # (B, z_dim)

    def _measure_loss(self, M, target_M, off, off_sum):
        diff = M - self.discount * target_M
        off_loss = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * self.num_parallel
        return off_loss, diag

    def stage_loss(self, stage, params, target, batch, cond, boot):
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs
        z, next_action = cond["z"], boot["next_action"]
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()

        if stage == "psm":
            t_psi = self._psi(self.psm_psi, target["psm_psi"], next_obs, z, next_action)
            t_phi = self._phi(target["phi"], goal)
            tM = jnp.einsum("pbz,cz->pbc", t_psi, t_phi)
            tmean, tunc = targets_uncertainty(tM, self.num_parallel)
            target_M = jax.lax.stop_gradient(tmean - self.pessimism * tunc)

            psi = self._psi(self.psm_psi, params["psm_psi"], obs, z, action)
            phi = self._phi(params["phi"], goal)
            M = jnp.einsum("pbz,cz->pbc", psi, phi)
            off_loss, diag = self._measure_loss(M, target_M, off, off_sum)
            cov = phi @ phi.T
            orth = 0.5 * jnp.sum((cov * off) ** 2) / off_sum - jnp.mean(jnp.diagonal(cov))
            loss = off_loss + diag + self.ortho_coef * orth
            return loss, {"psm_loss": loss, "psm_offdiag": off_loss, "psm_diag": diag, "ortho_loss": orth}

        # sf stage: train sf_psi against phi (online phi per the reference), phi detached.
        t_psi = self._psi(self.sf_psi, target["sf_psi"], next_obs, z, next_action)
        t_phi = self._phi(params["phi"], goal)                       # online phi (ref: not target_phi)
        tM = jnp.einsum("pbz,cz->pbc", t_psi, t_phi)
        tmean, tunc = targets_uncertainty(tM, self.num_parallel)
        target_M = jax.lax.stop_gradient(tmean - self.pessimism * tunc)

        psi = self._psi(self.sf_psi, params["sf_psi"], obs, z, action)
        phi = jax.lax.stop_gradient(self._phi(params["phi"], goal))  # detached (ref line 506)
        M = jnp.einsum("pbz,cz->pbc", psi, phi)
        off_loss, diag = self._measure_loss(M, target_M, off, off_sum)
        loss = off_loss + diag
        return loss, {"sf_loss": loss, "sf_offdiag": off_loss, "sf_diag": diag}

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        z = cond["z"]
        Ss = self._psi(self.sf_psi, params["sf_psi"], obs, z, actions)   # (P, B, z)
        Qs = (Ss * z).sum(-1)                                            # (P, B)
        qmean, _ = targets_uncertainty(Qs, self.num_parallel)
        return qmean, Qs

    def cond_vec(self, cond):
        return cond["z"]

    def sample_cond(self, net, batch, rng):
        next_obs = batch["next_observations"]
        B = next_obs.shape[0]
        r_z, r_mask, r_perm = jax.random.split(rng, 3)
        z_gauss = project_z(jax.random.normal(r_z, (B, self.z_dim)), self.norm_z)
        goals = project_z(self._phi(net["phi"].params, next_obs)[jax.random.permutation(r_perm, B)], self.norm_z)
        mix = (jax.random.uniform(r_mask, (B,)) < self.train_goal_ratio)[:, None]
        return {"z": jnp.where(mix, goals, z_gauss)}

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        phi = self._phi(net["phi"].params, obs)
        z = (rewards.reshape(1, -1) @ phi).reshape(-1) / phi.shape[0]
        return {"z": project_z(z, self.norm_z)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        z_dim, P = int(config["z_dim"]), int(config["num_parallel"])
        fc = config["forward"]
        mk_psi = lambda: ForwardMap(z_dim=z_dim, hidden_dim=fc["hidden_dim"], num_parallel=P,
                                    hidden_layers=fc["hidden_layers"], embedding_layers=fc["embedding_layers"])
        bc = config["backward"]
        phi = PhiMap(z_dim=z_dim, hidden_dim=bc["hidden_dim"], hidden_layers=bc["hidden_layers"], norm=bc["norm"])
        psm_psi, sf_psi = mk_psi(), mk_psi()
        rp, rs, rphi = jr.split(jr.PRNGKey(config["seed"]), 3)
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        params = {
            "phi": phi.init(rphi, ex_obs)["params"],
            "psm_psi": psm_psi.init(rp, ex_obs, ex_z, ex_actions)["params"],
            "sf_psi": sf_psi.init(rs, ex_obs, ex_z, ex_actions)["params"],
        }
        targets = {g: copy.deepcopy(params[g]) for g in cls.GROUPS}
        obj = cls(psm_psi=psm_psi, sf_psi=sf_psi, phi=phi, z_dim=z_dim, num_parallel=P,
                  discount=float(config["discount"]), ortho_coef=float(config["ortho_coef"]),
                  pessimism=float(config["fb_pessimism_penalty"]),
                  train_goal_ratio=float(config["train_goal_ratio"]), norm_z=bool(config["norm_z"]),
                  tau=float(config["f_target_tau"]))
        return obj, params, targets
