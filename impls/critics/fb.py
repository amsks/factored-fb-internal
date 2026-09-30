"""FBCritic — Forward-Backward successor-measure critic on the factored seam.

Math transcribed verbatim from PSMFlows agents/fb.py (_fb_loss_fn, _draw_injection z-mix,
_actor_loss_fn q_of, infer_z); restructured onto per-group TrainState params.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

import flax.linen as nn

from impls.utils.networks import ForwardMap, PhiMap
from impls.critics._measure_common import parse_hit_spec, scaled_hit_cols
from impls.utils.fb_math import project_z, targets_uncertainty


class _IdentityMap(nn.Module):
    """Parameterless stand-in for the left encoder (`left_encoder.identity: true`): the
    reference FB feeds F the raw observation. init() yields empty params, so the group's
    optimizer and polyak target are no-ops without any special-casing in the seam."""

    @nn.compact
    def __call__(self, x):
        return x


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
    # Touati et al. parity knobs (2026-08-27), both default 0.0 = every prior cell identical.
    # q_loss_coef: the reference's auxiliary Q loss (fb_ddpg.py `q_loss`; paper eq. 42) —
    #   MSE(F_k^T z, implicit_reward + gamma * min_l F_l^-T z) with the implicit reward read
    #   off the ONLINE B covariance inverse. The paper's Listing 1 divides by z_dim where the
    #   repo uses a 0.01 coefficient; pass the coefficient explicitly (1/z_dim for the
    #   paper's reading).
    # actor_pessimism_penalty: q_fn returns mean - c * unc over the F heads. With 2 heads
    #   unc = |Q1 - Q2|, so c = 0.5 IS the reference's min(Q1, Q2) exactly; 0.0 keeps the
    #   mean-of-heads every earlier cell ran.
    q_loss_coef: float = 0.0
    actor_pessimism_penalty: float = 0.0
    target_scale: float = 1.0     # scales the implicit immediate term: F^T B represents target_scale * M (ablation; 1-gamma = normalized measure under the squared loss)
    # `backward_input` ("state" | "hit_spec"): what the backward tower B consumes as its
    # RAW input, the same knob rldp.py / psm.py / hilp.py carry. "state" (default) is the
    # identity and keeps every prior FB cell bit-identical; "hit_spec" feeds the
    # eps-scaled object-coordinate block slice instead.
    #
    # This is NOT the `fixed_b=cube_xyz` arm from feat/goal-conditioned-fb. That one
    # REPLACED B with a fixed non-learned map and collapsed z_dim to 3 (cube-single 0.06
    # against plain FB's 0.50). Here B stays a learned PhiMap emitting the full z_dim, and
    # only its input narrows -- so the two are not the same experiment, and the latent-width
    # confound that sinks `-d5` arms (9.6 vs 90.7 on the GCIQL morph) is absent.
    backward_input: str = "state"
    blocks: tuple = ()

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

    def map_eval_cond(self, net, cond, observations=None):
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

    def _b_in(self, x):
        """B's raw input under `backward_input`. THE choke point: every `_B` call routes
        through here -- the loss, the frozen target, sample_cond's mix and map_eval_cond's
        z = project_z(B(goal)) alike -- so no caller can hand B the wrong view. In "state"
        mode this is the identity; in "hit_spec" mode the eps-scaled block slice."""
        if self.backward_input == "state":
            return x
        return scaled_hit_cols(x, self.blocks)

    def _B(self, p, x):
        return self.backward.apply({"params": p}, self._b_in(x))

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
        # Per-row critic loss weight (persist_mode="loss": the rebalance lever applied to the
        # CRITIC objective only). Weighted mean over the ROW (state-action) axis b; candidate
        # columns c and the ortho term are untouched. Absent key = plain mean, and all-ones
        # weights reproduce the plain mean exactly (x*1.0 and /1.0 are exact).
        rw = batch.get("row_weight")
        if rw is None:
            fb_off = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
            fb_diag = -self.target_scale * jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * P
        else:
            rw = jnp.asarray(rw, diff.dtype).reshape(-1)
            rw_mean = jnp.mean(rw)
            fb_off = 0.5 * jnp.sum((diff * off) ** 2 * rw[None, :, None]) / (off_sum * rw_mean)
            fb_diag = (-self.target_scale
                       * jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2) * rw[None, :]) / rw_mean * P)
        cov = Bm @ Bm.T
        orth_off = 0.5 * jnp.sum((cov * off) ** 2) / off_sum
        orth_diag = -jnp.mean(jnp.diagonal(cov))
        orth = orth_off + orth_diag
        loss = fb_off + fb_diag + self.ortho_coef * orth
        info = {"fb_loss": loss, "fb_offdiag": fb_off, "fb_diag": fb_diag, "ortho_loss": orth}
        if self.q_loss_coef > 0.0:
            # Reference auxiliary Q loss (fb_ddpg.py, paper eq. 42), transcribed exactly:
            # min over TARGET heads for next_Q, ONLINE B covariance inverse (torch.inverse,
            # not pinv) for the implicit reward, both heads regressed to one target.
            next_Q = jnp.min(jnp.einsum("pbz,bz->pb", tFs, z), axis=0)          # [B]
            # pinv, not inv: the paper's Listing 1 choice (the repo's torch.inverse blew up
            # here — maze z100 NaN'd by 100k with aux loss ~5e3 at step 200; pinv is equal
            # where inverse is sane and bounded where it is not).
            cov = Bm.T @ Bm / Bm.shape[0]                                        # [z, z]
            implicit_reward = ((Bm @ jnp.linalg.pinv(cov)) * z).sum(-1)          # [B]
            target_Q = jax.lax.stop_gradient(implicit_reward + disc * next_Q)
            Qs = jnp.einsum("pbz,bz->pb", Fs, z)                                 # [P, B]
            q_loss = ((Qs - target_Q[None]) ** 2).mean(axis=1).sum()
            loss = loss + self.q_loss_coef * q_loss
            info.update({"fb_loss": loss, "fb_q_loss": q_loss,
                         "fb_implicit_reward": implicit_reward.mean()})
        return loss, info

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        z = cond["z"]
        le = self._le(params["left_encoder"], obs)
        if stop_encoder_grad:
            le = jax.lax.stop_gradient(le)
        Fs = self._F(params["forward"], le, z, actions)
        Qs = (Fs * z).sum(-1)
        qmean, qunc = targets_uncertainty(Qs, self.num_parallel)
        # c = 0.5 at 2 heads is the reference's min(Q1, Q2); 0.0 is the mean every earlier
        # cell ran (see the field comment).
        return qmean - self.actor_pessimism_penalty * qunc, Qs

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
        # No 1/N: the reference accumulates z += wr.T @ B and projects. Masked while
        # norm_z=true because project_z is scale-invariant.
        z = (rewards.reshape(1, -1) @ Bm).reshape(-1)
        return {"z": project_z(z, self.norm_z)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        z_dim, L_dim, P = config["z_dim"], config["L_dim"], config["num_parallel"]
        forward = ForwardMap(z_dim=z_dim, hidden_dim=config["forward"]["hidden_dim"],
                             num_parallel=P, hidden_layers=config["forward"]["hidden_layers"],
                             embedding_layers=config["forward"]["embedding_layers"])
        backward_input = str(config.get("backward_input", "state"))
        assert backward_input in ("state", "hit_spec"), (
            f"backward_input must be 'state' or 'hit_spec', got {backward_input!r}")
        blocks = ()
        ex_b_in = ex_obs
        if backward_input == "hit_spec":
            # B's input becomes a literal column slice, so the spec must fit inside the
            # observation -- jax would otherwise clip short rows into a silently narrower
            # input at trace time. Same check as rldp.py / psm.py / hilp.py.
            blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
            need = max(b for (_a, b, _e) in blocks)
            obs_dim = int(ex_obs.shape[-1])
            assert obs_dim >= need, (
                f"backward_input='hit_spec' needs observations wide enough for every "
                f"measure_hit_spec block: the spec reaches column {need} but the "
                f"observations are only {obs_dim} wide.")
            ex_b_in = scaled_hit_cols(ex_obs, blocks)
        backward = PhiMap(z_dim=z_dim, hidden_dim=config["backward"]["hidden_dim"],
                          hidden_layers=config["backward"]["hidden_layers"], norm=config["backward"]["norm"])
        if config["left_encoder"].get("identity", False):
            # Touati parity: the reference FB has NO encoder in front of F — its
            # preprocessors read the raw observation (controllable_agent fb_modules.py).
            # An identity module keeps the group/target plumbing untouched (empty params,
            # zero-cost soft_update) so nothing else in the seam changes shape.
            left_encoder = _IdentityMap()
            ex_feat = jnp.zeros_like(ex_obs)
        else:
            left_encoder = PhiMap(z_dim=L_dim, hidden_dim=config["left_encoder"]["hidden_dim"],
                                  hidden_layers=config["left_encoder"]["hidden_layers"],
                                  norm=config["left_encoder"]["norm"])
            ex_feat = jnp.zeros((ex_obs.shape[0], L_dim))
        rf, rb, rle = jr.split(jr.PRNGKey(config["seed"]), 3)
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        params = {
            # .get: _IdentityMap creates no variables, so init() has no "params" collection.
            "left_encoder": left_encoder.init(rle, ex_obs).get("params", {}),
            "backward": backward.init(rb, ex_b_in)["params"],
            "forward": forward.init(rf, ex_feat, ex_z, ex_actions)["params"],
        }
        targets = {g: copy.deepcopy(params[g]) for g in cls.GROUPS}
        critic = cls(
            forward=forward, backward=backward, left_encoder=left_encoder,
            z_dim=z_dim, num_parallel=P, discount=config["discount"],
            ortho_coef=config["ortho_coef"], fb_pessimism_penalty=config["fb_pessimism_penalty"],
            train_goal_ratio=config["train_goal_ratio"], norm_z=config["norm_z"],
            f_tau=config["f_target_tau"], b_tau=config["b_target_tau"],
            q_loss_coef=float(config.get("q_loss_coef", 0.0)),
            actor_pessimism_penalty=float(config.get("actor_pessimism_penalty", 0.0)),
            target_scale=float(config.get("fb_target_scale", 1.0)),
            backward_input=backward_input, blocks=blocks,
        )
        return critic, params, targets
