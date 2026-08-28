"""DDPGBCDensityCPRActor — ddpgbc_density plus a hindsight-z-conditioned BC term.

WHY (2026-08-26, humenv diagnosis). FB-CPR's actor carries no BC term because its
discriminator regularizes pi(.|s, z) toward expert-like trajectories, z-conditionally. The
motivo-parity humenv arm copied the "no BC term" half (actor.alpha=0) while removing the
discriminator, leaving the actor with no grounding at all — and every conditioned policy it
produced scored ~0 on goal reaching under BOTH prompting routes. This actor restores the
offline core of that anchoring.

THE TERM. density_fb's training z for the actor leg is, per row, either (a) the encoded
index of that row's `actor_goal` — a same-trajectory FUTURE state when the dataset runs
actor_p_trajgoal=1 — or (b) a fresh uniform sphere draw (`_z_mix`, probability
z_rand_frac). On (a) rows the dataset action IS a demonstration of pursuing that z, so we
clone it: hindsight-conditioned BC, the construction the bridge carried as "hindsight
indicator feeds a BC mask" and this tree never ported. On (b) rows the action has no
relation to z and cloning it would be the marginal-BC failure documented on ExoRL (flowbc
cloning the RND explorer), so the mask excludes them.

    cpr_bc = -cpr_alpha * sum(m * log pi(a|s, z)) / (sum(m) + eps),   m = cond["_z_hindsight"]

KNOBS (actor/ddpgbc_density_cpr.yaml):
  * `cpr_alpha` — weight of the masked BC term.
  * `q_coeff`   — weight of the inherited critic-max term. 1.0 = full CPR-style arm;
    0.0 = the pure hindsight-BC floor (the bcflow precedent: same actor, value term off).
  * inherited `alpha` stays the UNCONDITIONAL BC coefficient and defaults to 0 here.

Requires density_fb with measure_density_z=true — the only critic that publishes
`_z_hindsight`. Any other pairing fails loudly at the first loss call, mirroring the
`actor_sample_rng` idiom, because a silently-empty mask would train the parent actor while
reporting a CPR arm.
"""
import dataclasses

import jax
import jax.numpy as jnp

from impls.actors.ddpgbc import _obs_feat, _smooth
from impls.actors.ddpgbc_density import DDPGBCDensityActor


@dataclasses.dataclass(frozen=True)
class DDPGBCDensityCPRActor(DDPGBCDensityActor):
    cpr_alpha: float = 1.0
    q_coeff: float = 1.0

    def loss(self, params, target, batch, critic, cond, boot):
        if "_z_hindsight" not in cond:
            raise KeyError(
                "ddpgbc_density_cpr needs cond['_z_hindsight'], which only density_fb "
                "under measure_density_z=true publishes (its sample_cond marks the rows "
                "whose z kept the encoded index of their own actor_goal). Without it the "
                "CPR term has no mask; use ddpgbc_density instead.")
        obs, action = batch["observations"], batch["actions"]
        cvec = critic.cond_vec(cond)
        dist = self._dist(params["actor"], _obs_feat(cond, obs), cvec)
        q_actions = self._q_actions(dist, boot)
        if self.q_noise_std > 0.0:
            q_actions = _smooth(q_actions, boot["actor_q_rng"],
                                self.q_noise_std, self.q_noise_clip)
        q, qs = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
        q_scale = jnp.abs(qs).mean()
        q_loss = -q.mean() / jax.lax.stop_gradient(q_scale + 1e-6)
        log_prob = dist.log_prob(action)
        bc_loss = -(self.alpha * log_prob).mean()
        m = cond["_z_hindsight"][:, 0]
        cpr_bc_loss = -self.cpr_alpha * (m * log_prob).sum() / (m.sum() + 1e-6)
        loss = self.q_coeff * q_loss + bc_loss + cpr_bc_loss
        return loss, {
            "actor_loss": loss, "q_loss": q_loss, "bc_loss": bc_loss,
            "cpr_bc_loss": cpr_bc_loss, "hindsight_frac": m.mean(),
            "q_mean": q.mean(), "q_scale": q_scale, "bc_log_prob": log_prob.mean(),
            "cpr_log_prob": (m * log_prob).sum() / (m.sum() + 1e-6),
            "mse": jnp.mean((dist.mode() - action) ** 2),
            "action_gap_l2": jnp.linalg.norm(q_actions - action, axis=-1).mean(),
        }

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        base, params = DDPGBCDensityActor.build(ex_obs, ex_actions, config)
        a_cfg = config["actor"]
        return cls(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                   cpr_alpha=float(a_cfg.get("cpr_alpha", 1.0)),
                   q_coeff=float(a_cfg.get("q_coeff", 1.0))), params
