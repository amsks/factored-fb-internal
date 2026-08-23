"""PSMCritic — Proto Successor Measure critic on the factored critic x actor seam.

Transcribed (NOT imported) from the PyTorch reference in this repo's torch tree,
`agents/psm/`: `agent.py` (`_update_psm`, `_update_sf`, `sample_mixed_z`, the update
orchestration), `model.py` (`PSMModel.sample_z_psm` / `project_z` /
`get_targets_uncertainty` / `reward_inference`) and `proto_sampler.py`
(`ProtoBehaviorSampler`).

What PSM is
-----------
Two successor-measure heads over one shared basis phi(goal):

  * `psm_psi(s, z_bin, a)` -- the PROTO successor net. `z_bin` is a *binary* code of
    width `max_log_seed`, and the TD target's next action comes from a deterministic
    pseudo-random behaviour policy keyed by (dataset row, z_bin). Each binary code
    therefore indexes a different fixed policy, and psm_psi learns the successor
    measure of a whole family of them at once -- the "proto" measures the paper spans
    task-specific measures with.
  * `sf_psi(s, z, a)` -- the ordinary successor-feature head at a continuous z, whose
    readout Q(s,a,z) = <sf_psi(s,a,z), z> is what the actor ascends.

Both are trained with FB's contrastive measure loss against M = psi phi^T; only the
proto branch carries the orthonormality term on phi (the reference multiplies the SF
branch's copy by a literal `0`, so it is omitted here rather than added as a no-op).

Seam mapping
------------
GROUPS      = (phi, sf_psi, psm_psi)   -- one nn.Module per group, as the seam requires.
STAGES      = ("psm", "sf")            -- the reference's two separate .backward() calls,
                                          in the same order. Stage "psm" steps psm_psi+phi;
                                          stage "sf" steps sf_psi only and READS the phi
                                          that "psm" just moved (the generic update's
                                          `online()` is re-read per stage), matching the
                                          reference exactly.
TARGETS     = all three groups at `target_tau`.
cond["z"]   = the reference `sample_mixed_z` draw (Gaussian, `mix_ratio` of the rows
              replaced by project_z(phi(shuffled next_obs))). This is `cond_vec`, so it
              is what every seam actor conditions on and what `boot["next_action"]` --
              the SF branch's target action, drawn from the LEARNED policy in the
              reference -- is drawn at.
cond["z_psm"] / cond["psm_next_action"] = the proto branch's binary code and its
              behaviour-policy action. Critic-private; no actor reads them.

Deviations from the reference, all deliberate and load-bearing to state exactly
--------------------------------------------------------------------------------
1. The proto behaviour sampler's action table. The reference materialises
   `2**max_log_seed + 20000` actions by looping `torch.random.manual_seed(i)` and
   drawing `(torch.rand(action_dim) - 1) * 2`. That is deterministic, but it is
   deterministic *in torch's MT19937*, which JAX cannot reproduce. We keep the exact
   structure -- a pure function seed -> action, the same seed arithmetic
   ((z_bin . powers) + row_hash) mod max_seed, and the same [-2, 0] action range the
   reference actually produces -- but draw with `jax.random.fold_in` on threefry. The
   sampler is therefore deterministic and reproducible, just not bit-equal to torch.
   Nothing in the parity test depends on it (the reference exposes a `next_action=`
   injection hook that bypasses it, and the fixture uses that hook).
2. The reference has no encoder on the state path (Identity normalizer + Identity
   fw/bw encoders), so `goal = next_obs` and obs is used raw. Same here. There is no
   `left_encoder`; PSM's psi towers take the observation directly.
3. The reference's TD3 actor is not part of this critic -- the seam supplies the actor.

Row discount
------------
The reference uses a per-row discount, `gamma * (1 - terminated)`, shaped [B, 1] so it
scales the target measure row-wise. FBDataset emits `terminals`, GCDataset emits
`masks`; either is honoured, and a batch with neither falls back to the scalar gamma
(which is what FBDataset's always-zero `terminals` amounts to anyway).
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.utils.networks import ForwardMap, PhiMap
from impls.utils.fb_math import project_z, targets_uncertainty, contrastive_loss, ortho_loss


class PsmConfigError(ValueError):
    """Raised at build time for a structurally invalid PSM config."""


def proto_actions(final_seed, action_dim, base_key):
    """Reference `ProtoBehaviorSampler.forward`'s table lookup, as a pure function.

    The reference precomputes `seed_to_action[i] = (torch.rand(action_dim) - 1) * 2`
    under `torch.random.manual_seed(i)` and gathers row `final_seed`. We evaluate the
    same map lazily with threefry (see deviation 1 in the module docstring): still a
    pure, deterministic function of the integer seed, and it costs B draws per update
    instead of an 85k-row constant baked into the jaxpr.
    """
    def one(s):
        return (jax.random.uniform(jax.random.fold_in(base_key, s), (action_dim,)) - 1.0) * 2.0
    return jax.vmap(one)(final_seed.astype(jnp.uint32))


@dataclasses.dataclass(frozen=True)
class PSMCritic:
    phi: Any
    sf_psi: Any
    psm_psi: Any
    z_dim: int
    max_log_seed: int
    action_dim: int
    num_parallel: int
    discount: float
    ortho_coef: float
    pessimism_penalty: float
    actor_pessimism_penalty: float
    mix_ratio: float
    norm_z: bool
    target_tau: float
    proto_seed: int

    GROUPS = ("phi", "sf_psi", "psm_psi")
    STAGES = ("psm", "sf")
    PROVIDES_V = False
    PROVIDES_COND = frozenset({"z"})   # z_psm/psm_next_action are critic-private
    # infer_cond's base measure: the SUCCESSOR marginal. This critic's backward tower is
    # trained on `goal = next_obs` (stage_loss) and sample_cond mixes in B(next_observations),
    # so the reward->z readout must be solved on the same rows, paired with the reward of
    # that same transition. See impls/critics/base.py::INFER_COND_MARGINALS.
    INFER_COND_MARGINAL = "next_observations"
    BOOT_COND = "actor_goal"           # the SF target action is pi(.|s', z) -- cond_vec IS z

    @property
    def TARGETS(self):
        return {g: self.target_tau for g in self.GROUPS}

    @property
    def cvec_dim(self):
        return self.z_dim

    @property
    def max_seed(self):
        """Reference ProtoBehaviorSampler.max_seed (proto_sampler.py)."""
        return 2 ** self.max_log_seed + 20000

    def stage_groups(self, stage):
        # "psm" steps psm_psi AND phi (reference: one backward, two optimizers);
        # "sf" steps sf_psi only -- the reference detaches phi there.
        if stage == "psm":
            return ("psm_psi", "phi")
        assert stage == "sf", stage
        return ("sf_psi",)

    # --- module apply helpers ---
    def _phi(self, p, x):
        return self.phi.apply({"params": p}, x)

    def _sf(self, p, obs, z, action):
        return self.sf_psi.apply({"params": p}, obs, z, action)

    def _psm(self, p, obs, z, action):
        return self.psm_psi.apply({"params": p}, obs, z, action)

    def _row_discount(self, batch, B):
        """gamma * (1 - terminated), shaped [B, 1] (reference agent.py update())."""
        if "terminals" in batch:
            return self.discount * (1.0 - jnp.asarray(batch["terminals"]).reshape(B, 1))
        if "masks" in batch:
            return self.discount * jnp.asarray(batch["masks"]).reshape(B, 1)
        return self.discount

    def _measure(self, psis, phi):
        return jnp.einsum("pbz,cz->pbc", psis, phi)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        goal = next_obs                     # phi_input="s"; bw encoder is Identity for state
        B, P = obs.shape[0], self.num_parallel
        off = 1.0 - jnp.eye(B)
        off_sum = off.sum()
        disc = self._row_discount(batch, B)

        if stage == "psm":
            z, next_action = cond["z_psm"], cond["psm_next_action"]
            # target measure: BOTH target nets (reference _update_psm uses target_phi).
            tM = self._measure(self._psm(target["psm_psi"], next_obs, z, next_action),
                               self._phi(target["phi"], goal))
            tmean, tunc = targets_uncertainty(tM, P)
            target_M = jax.lax.stop_gradient(tmean - self.pessimism_penalty * tunc)

            phi = self._phi(params["phi"], goal)
            M = self._measure(self._psm(params["psm_psi"], obs, z, action), phi)
            meas, diag, offdiag = contrastive_loss(M, target_M, disc, off, off_sum)
            orth, orth_diag, orth_offdiag = ortho_loss(phi, off, off_sum)
            loss = meas + self.ortho_coef * orth
            return loss, {"psm_loss": loss, "psm_diag": diag, "psm_offdiag": offdiag,
                          "orth_loss": orth, "orth_loss_diag": orth_diag,
                          "orth_loss_offdiag": orth_offdiag,
                          "phi_norm": jnp.linalg.norm(phi, axis=-1).mean(),
                          "psm_M_diag": jnp.diagonal(M, axis1=1, axis2=2).mean(),
                          "psm_targetM_diag": jnp.diagonal(target_M).mean()}

        assert stage == "sf", stage
        z, next_action = cond["z"], boot["next_action"]
        # Reference _update_sf line 499: the TARGET measure uses the ONLINE phi, not
        # target_phi -- and the online phi it uses is the one the "psm" stage just
        # stepped. Only sf_psi is a target net here.
        phi = jax.lax.stop_gradient(self._phi(params["phi"], goal))
        tM = self._measure(self._sf(target["sf_psi"], next_obs, z, next_action), phi)
        tmean, tunc = targets_uncertainty(tM, P)
        target_M = jax.lax.stop_gradient(tmean - self.pessimism_penalty * tunc)

        M = self._measure(self._sf(params["sf_psi"], obs, z, action), phi)
        loss, diag, offdiag = contrastive_loss(M, target_M, disc, off, off_sum)
        # The reference adds `0 * ortho_coef * orth_loss` to the SF loss -- a literal
        # zero. Omitted rather than reproduced: it contributes neither value nor grad.
        return loss, {"sf_loss": loss, "sf_diag": diag, "sf_offdiag": offdiag,
                      "sf_phi_norm": jnp.linalg.norm(phi, axis=-1).mean(),
                      "sf_M_diag": jnp.diagonal(M, axis1=1, axis2=2).mean(),
                      "sf_targetM_diag": jnp.diagonal(target_M).mean()}

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        # stop_encoder_grad is accepted for seam uniformity; PSM's psi towers consume the
        # observation directly (state path => Identity encoder), so there is nothing to cut.
        z = cond["z"]
        Qs = (self._sf(params["sf_psi"], obs, z, actions) * z).sum(-1)   # [P, B]
        qmean, qunc = targets_uncertainty(Qs, self.num_parallel)
        # Reference _update_td3_actor: Q = mean - actor_pessimism_penalty * uncertainty.
        return qmean - self.actor_pessimism_penalty * qunc, Qs

    def cond_vec(self, cond):
        return cond["z"]

    def sample_cond(self, net, batch, rng):
        next_obs = batch["next_observations"]
        B = next_obs.shape[0]
        # The proto action draws no randomness here: it is a pure function of
        # (binary code, row) at the FIXED `proto_seed` key -- see proto_actions.
        r_bin, r_z, r_mask, r_perm = jax.random.split(rng, 4)

        # --- SF/actor branch: reference PSMAgent.sample_mixed_z ---
        z_gauss = project_z(jax.random.normal(r_z, (B, self.z_dim)), self.norm_z)
        mixed = project_z(
            self._phi(net["phi"].params, next_obs[jax.random.permutation(r_perm, B)]),
            self.norm_z,
        )
        mask = (jax.random.uniform(r_mask, (B,)) < self.mix_ratio)[:, None]
        z = jnp.where(mask, mixed, z_gauss)

        # --- proto branch: reference PSMModel.sample_z_psm (LSB-first bits of a
        # uniform integer in [0, 2**max_log_seed)) ---
        codes = jax.random.randint(r_bin, (B,), 0, 2 ** self.max_log_seed)
        z_psm = ((codes[:, None] >> jnp.arange(self.max_log_seed)) & 1).astype(z.dtype)

        # --- proto behaviour action (reference ProtoBehaviorSampler.forward) ---
        # `powers` is the reversed power list, so the dot product reads the bit array
        # MSB-first. row_hash is the dataset ROW index (reference: batch["index"], which
        # train.py enables with_index=True for psm specifically). Never batch position.
        powers = 2 ** jnp.arange(self.max_log_seed)[::-1]
        if "index" not in batch:
            raise KeyError(
                "psm needs batch['index'] -- the dataset ROW index. Without it the proto "
                "policy is keyed on batch POSITION, so a transition draws a different "
                "policy every resample and psm_psi has no fixed policy to converge to. "
                "Dataset.get_subset emits it; a hand-built batch must too.")
        row_hash = jnp.asarray(batch["index"]).reshape(-1)
        seed = ((z_psm.astype(jnp.int32) * powers).sum(1) + row_hash) % self.max_seed
        psm_next_action = proto_actions(seed, self.action_dim,
                                        jax.random.PRNGKey(self.proto_seed))

        return {"z": z, "z_psm": z_psm, "psm_next_action": psm_next_action}

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,), jnp.float64)}

    def map_eval_cond(self, net, cond):
        """Eval-time hook (mirrors FBCritic.map_eval_cond): embed the env's goal frame
        into the latent the actor was TRAINED on, z = project_z(phi(goal)) -- the exact
        conditioning `sample_cond` mixes in at rate `mix_ratio`. Without it a goal-driven
        eval runs on eval_cond0's zero z, for which Q(s,a;0) = <psi,0> == 0 and the policy
        is goal-blind."""
        if "goal" not in cond:
            return cond
        return {**cond, "z": project_z(self._phi(net["phi"].params, cond["goal"]), self.norm_z)}

    def infer_cond(self, net, obs, rewards):
        """Reference PSMModel.reward_inference: z = (r^T phi(s')) / N, projected."""
        p = self._phi(net["phi"].params, obs)
        z = (jnp.asarray(rewards).reshape(1, -1) @ p).reshape(-1) / p.shape[0]
        return {"z": project_z(z, self.norm_z)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr

        z_dim = int(config["z_dim"])
        max_log_seed = int(config["max_log_seed"])
        P = int(config["num_parallel"])
        if P < 2:
            # targets_uncertainty divides by P**2 - P.
            raise PsmConfigError(f"num_parallel must be >= 2 (got {P}); the uncertainty "
                                 f"term divides by num_parallel**2 - num_parallel")
        if not 1 <= max_log_seed <= 30:
            # sample_cond draws the code as an int32 in [0, 2**max_log_seed) and shifts it.
            raise PsmConfigError(f"max_log_seed must be in [1, 30] (got {max_log_seed})")

        phi = PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                     hidden_layers=config["phi"]["hidden_layers"], norm=config["phi"]["norm"])
        psi_kw = dict(z_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"], num_parallel=P,
                      hidden_layers=config["sf"]["hidden_layers"],
                      embedding_layers=config["sf"]["embedding_layers"])
        sf_psi = ForwardMap(**psi_kw)
        psm_psi = ForwardMap(**psi_kw)          # same shape; z leg is max_log_seed wide

        rp, rs, rq = jr.split(jr.PRNGKey(config["seed"] + 7000), 3)
        n = ex_obs.shape[0]
        params = {
            "phi": phi.init(rp, ex_obs)["params"],
            "sf_psi": sf_psi.init(rs, ex_obs, jnp.zeros((n, z_dim)), ex_actions)["params"],
            "psm_psi": psm_psi.init(rq, ex_obs, jnp.zeros((n, max_log_seed)), ex_actions)["params"],
        }
        targets = {g: copy.deepcopy(params[g]) for g in cls.GROUPS}
        critic = cls(
            phi=phi, sf_psi=sf_psi, psm_psi=psm_psi,
            z_dim=z_dim, max_log_seed=max_log_seed, action_dim=int(ex_actions.shape[-1]),
            num_parallel=P, discount=float(config["discount"]),
            ortho_coef=float(config["ortho_coef"]),
            pessimism_penalty=float(config["pessimism_penalty"]),
            actor_pessimism_penalty=float(config["actor_pessimism_penalty"]),
            mix_ratio=float(config["mix_ratio"]), norm_z=bool(config["norm_z"]),
            target_tau=float(config["target_tau"]),
            # NOT derived from config["seed"]: the reference's seed_to_action table is a
            # fixed constant of the code, identical in every run, so the proto policy
            # family must not vary with the run seed either.
            proto_seed=int(config["proto_seed"]),
        )
        return critic, params, targets
