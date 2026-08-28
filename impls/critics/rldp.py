"""RLDPCritic — Forward-Backward plus a self-predictive (SP) head, on the factored seam.

Transcribed from the PyTorch reference `agents/rldp/{agent,model}.py` (branch
gciql-fb-bridge; `RLDPAgent.update_fb` and `RLDPModel.__init__`), which is itself an
adaptation of td_jepa/metamotivo's rldp. Nothing is imported from `third_party/` or from
that tree — the math below is a line-by-line transcription, restructured onto the seam's
per-group TrainState params. The FB half is `impls/critics/fb.py` with the three
divergences listed under "What RLDP changes" below.

The 'DP' in RLDP
----------------
`_predictor` maps a backward embedding + an action to a *predicted next* backward
embedding. Starting at B(s_t) it is rolled forward h steps, one step per action in the
window, and each step is L2-matched against the TARGET backward map at the corresponding
future state:

    curr_0 = B(s_t)
    curr_k = P(curr_{k-1}, a_{t+k-1})      for k = 1..h
    sp     = mean_k  || curr_k - sg[B_target(s_{t+k})] ||^2  (sum over z, mean over P x B)

i.e. step k applies the action taken at s_{t+k-1} and predicts the embedding of the
state it reaches, exactly td_jepa's alignment (see `_sp_pairs`; the torch port
agents/rldp/agent.py has an off-by-one here that this transcription CORRECTS —
history in the `_sp_pairs` docstring).

What RLDP changes relative to `impls/critics/fb.py`
---------------------------------------------------
1. **B is detached inside the successor term.** The reference writes
   `B = self.model._backward_map(goal).detach()` where plain FB writes
   `B = self.model._backward_map(goal)`. So the FB TD residual trains the FORWARD map
   only; the backward map is learned entirely by the SP loss and the orthonormality term.
   This is the structural point of RLDP and is transcribed as-is.
2. **The orthonormality term sits at B(obs), not B(goal).** The reference builds
   `Cov = B_pred @ B_pred.T` from `B_pred = self.model._backward_map(obs_bw)` — the same
   tensor the SP rollout starts from — whereas FB whitens `B(goal)`.
3. **A fourth module group, `predictor`.** It rides the backward learning rate (`lr_b`),
   matching `RLDPAgent.setup_training`, which appends `_predictor.parameters()` to the
   backward optimizer. It has no target network (the reference polyak-updates only
   forward / backward / left_encoder), so it is absent from `TARGETS`.

State-based only, as everywhere in this tree: `_fw_encoder` / `_bw_encoder` are Identity,
so the reference's `goal = _bw_encoder(next_obs)` is `next_obs` and its
`next_obs_fw = _fw_encoder(next_obs)` is `next_obs` as well.

The reference's optional Q-MSE auxiliary (`q_loss_coef`) is NOT carried, exactly as
`impls/critics/fb.py` does not carry it: no config in this tree sets it, and a
never-exercised code path in a loss is a liability.
"""
import copy
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from impls.critics._measure_common import parse_hit_spec, scaled_hit_cols
from impls.utils.networks import ForwardMap, PhiMap, PredictorMap
from impls.utils.fb_math import project_z, targets_uncertainty


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
    fb_pessimism_penalty: float
    train_goal_ratio: float
    norm_z: bool
    f_tau: float
    b_tau: float
    # `backward_input` ("state" | "hit_spec"): what the backward tower B consumes as its
    # RAW input. "state" (default) = the full observation, bit-identical shipped behavior.
    # "hit_spec" = the eps-scaled hit-spec block slice of the observation — exactly
    # density_fb.py's `_density_queries` transform (see _measure_common.scaled_hit_cols),
    # applied inside `_B` itself so EVERY consumer (stage_loss's B(goal), the SP rollout's
    # start/targets, sample_cond, map_eval_cond, infer_cond, and every subclass) inherits
    # it through the one choke point. The SP predictor lives in B's EMBEDDING space and is
    # untouched. `blocks` is the parsed measure_hit_spec, only read in "hit_spec" mode.
    backward_input: str = "state"
    blocks: tuple = ()

    GROUPS = ("forward", "backward", "left_encoder", "predictor")
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
        # No entry for `predictor`: the reference soft-updates only these three
        # (RLDPAgent.update -> _soft_update_params on forward / backward / left_encoder).
        return {"forward": self.f_tau, "left_encoder": self.f_tau, "backward": self.b_tau}

    @property
    def cvec_dim(self):
        return self.z_dim

    def eval_cond0(self, ex_obs, config):
        return {"z": jnp.zeros((self.z_dim,), jnp.float64)}

    def map_eval_cond(self, net, cond):
        """Eval-time hook, identical to FBCritic's: embed the env's goal frame into the z
        the actor was TRAINED on, z = project_z(B(goal)). Without it eval runs on
        eval_cond0's zero z, for which Q(s,a;0) = <F,0> == 0 and the policy is goal-blind
        (the 2026-08-17 FB-zero bug)."""
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
        through here, so no caller (parent or subclass) can hand B the wrong view. In
        "state" mode this is the identity; in "hit_spec" mode it is the eps-scaled
        hit-spec block slice (density_fb's `_density_queries` semantics). No caller
        pre-slices — `_B` always receives full observation rows."""
        if self.backward_input == "state":
            return x
        return scaled_hit_cols(x, self.blocks)

    def _B(self, p, x):
        return self.backward.apply({"params": p}, self._b_in(x))

    def _le(self, p, x):
        return self.left_encoder.apply({"params": p}, x)

    def _P(self, p, x, action):
        return self.predictor.apply({"params": p}, x, action)

    # --- the self-predictive head ---
    def _sp_pairs(self, batch):
        """The (action, target-state) sequence the rollout is trained on — the
        reference's h-step alignment: rollout step k applies a_{t+k-1} and is matched
        against B_target(s_{t+k}), for k = 1..h.

            actions = [a_t]      + [a_{t+1}, ..., a_{t+h-1}]   ([action]   + future_act columns)
            targets = [s_{t+1}]  + [s_{t+2}, ..., s_{t+h}]     ([next_obs] + future_obs columns)

        The window arrives PRE-ALIGNED from WindowedDataset (impls/utils/datasets.py):
        future_actions column i is a_{t+1+i} and future_observations column i is
        s_{t+2+i} — the state that action reaches — mirroring td_jepa's sampler
        (metamotivo/buffers/parallel.py: `traj_action` = a_{t+1}..a_{t+H-1},
        `traj_observation` = s_{t+2}..s_{t+H}). So the plain per-column zip below is
        exactly the reference's zip in `update_fb`.

        HISTORY (2026-08-25 fix). This port originally copied the torch port
        agents/rldp/agent.py, whose window slices `batch["observation"][:, 1:]` where
        td_jepa reads the next-observation sequence: targets came out as
        [s_{t+1}, s_{t+1}, s_{t+2}, ..., s_{t+h-1}] — step 2 re-targeted s_{t+1}
        (training the predictor that a_{t+1} does not move the embedding) and the
        rollout never reached s_{t+h}. That off-by-one was the TORCH PORT's, not the
        td_jepa reference's, and is corrected here; the correction lives in
        WindowedDataset's column offsets, so this enumeration matches the reference
        literally.

        With h == 1 the batch carries no window and this degenerates to the single
        one-step pair, matching the reference's `if self.horizon > 1` guard.
        """
        actions = [batch["actions"]]
        targets = [batch["next_observations"]]
        if "future_observations" in batch:
            fo, fa = batch["future_observations"], batch["future_actions"]
            assert fo.shape[1] == fa.shape[1] == self.horizon - 1, (
                f"windowed batch has {fo.shape[1]} future columns but the critic was built "
                f"with horizon={self.horizon} (expects {self.horizon - 1}); the critic and "
                f"impls/utils/datasets.py:WindowedDataset are reading different configs")
            targets += [fo[:, i] for i in range(fo.shape[1])]
            actions += [fa[:, i] for i in range(fa.shape[1])]
        else:
            assert self.horizon == 1, (
                f"critic built with horizon={self.horizon} but the batch carries no "
                f"'future_observations'; rldp needs dataset_class: WindowedDataset")
        return actions, targets

    def _sp_loss(self, params, target, batch):
        """The RLDP self-predictive term. Split out of `stage_loss` so it can be pinned
        against an independent reference (tests/impls/test_rldp_seam.py)."""
        actions, targets = self._sp_pairs(batch)
        P = self.num_parallel
        # Targets come from the TARGET backward map under no_grad. The reference applies
        # it once to the flattened window; that is elementwise over the batch axis, so
        # per-step application is the same tensor.
        ys = [jax.lax.stop_gradient(self._B(target["backward"], s)) for s in targets]

        # `curr` starts at B(obs) -- the ONLINE backward map, which is what carries the
        # gradient into B. Broadcast to the ensemble axis: the reference's
        # `obs.expand(num_parallel, -1, -1)` inside VForwardMap.forward, which is a no-op
        # on every later step because `curr` is already [P, B, z_dim].
        B_pred = self._B(params["backward"], batch["observations"])
        curr = jnp.broadcast_to(B_pred[None], (P,) + B_pred.shape)

        sp = 0.0
        for a_t, y_t in zip(actions, ys):
            curr = self._P(params["predictor"], curr, a_t)
            sp = sp + jnp.mean(jnp.sum((curr - y_t[None]) ** 2, axis=-1))
        return sp / max(len(actions), 1), B_pred

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

        # online measure. Divergence 1: B(goal) is DETACHED here (reference:
        # `B = self.model._backward_map(goal).detach()`), so the successor residual
        # trains the forward map only.
        le = self._le(params["left_encoder"], obs)
        Fs = self._F(params["forward"], le, z, action)
        Bg = jax.lax.stop_gradient(self._B(params["backward"], goal))
        M = jnp.einsum("pbz,cz->pbc", Fs, Bg)
        diff = M - disc * target_M
        fb_off = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        fb_diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * P
        fb = fb_off + fb_diag

        # self-predictive head (also returns the B(obs) the ortho term whitens).
        sp, B_pred = self._sp_loss(params, target, batch)

        # Divergence 2: orthonormality at B(obs), not B(goal).
        cov = B_pred @ B_pred.T
        orth_off = 0.5 * jnp.sum((cov * off) ** 2) / off_sum
        orth_diag = -jnp.mean(jnp.diagonal(cov))
        orth = orth_off + orth_diag

        loss = fb + sp + self.ortho_coef * orth
        return loss, {"fb_loss": loss, "fb_offdiag": fb_off, "fb_diag": fb_diag,
                      "ortho_loss": orth, "sp_loss": sp}

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
        """The vector actors concatenate onto obs to condition. RLDP, like FB: the latent z."""
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
        horizon = int(config["horizon"])
        assert horizon >= 1, f"horizon must be >= 1, got {horizon}"
        backward_input = str(config.get("backward_input", "state"))
        assert backward_input in ("state", "hit_spec"), (
            f"backward_input must be 'state' or 'hit_spec', got {backward_input!r}")
        blocks = ()
        ex_b_in = ex_obs
        if backward_input == "hit_spec":
            # The B tower's input is a literal column slice of the observation, so the
            # spec must fit inside it — jax would otherwise clip short rows into a
            # silently different (narrower) input at trace time.
            blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
            need = max(b for (_a, b, _e) in blocks)
            obs_dim = int(ex_obs.shape[-1])
            assert obs_dim >= need, (
                f"backward_input='hit_spec' needs observations wide enough for every "
                f"measure_hit_spec block: the spec reaches column {need} but the "
                f"observations are only {obs_dim} wide.")
            ex_b_in = scaled_hit_cols(ex_obs, blocks)
        forward = ForwardMap(z_dim=z_dim, hidden_dim=config["forward"]["hidden_dim"],
                             num_parallel=P, hidden_layers=config["forward"]["hidden_layers"],
                             embedding_layers=config["forward"]["embedding_layers"])
        backward = PhiMap(z_dim=z_dim, hidden_dim=config["backward"]["hidden_dim"],
                          hidden_layers=config["backward"]["hidden_layers"], norm=config["backward"]["norm"])
        left_encoder = PhiMap(z_dim=L_dim, hidden_dim=config["left_encoder"]["hidden_dim"],
                              hidden_layers=config["left_encoder"]["hidden_layers"],
                              norm=config["left_encoder"]["norm"])
        # RLDPModel builds the predictor as VForwardMap(obs=z_space[z_dim], z=action_dim)
        # -> z_dim, i.e. it lives entirely in backward-embedding space.
        predictor = PredictorMap(out_dim=z_dim, hidden_dim=config["predictor"]["hidden_dim"],
                                 num_parallel=P,
                                 hidden_layers=config["predictor"]["hidden_layers"],
                                 embedding_layers=config["predictor"]["embedding_layers"])
        rf, rb, rle, rp = jr.split(jr.PRNGKey(config["seed"]), 4)
        ex_feat = jnp.zeros((ex_obs.shape[0], L_dim))
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))
        ex_pred_in = jnp.zeros((P, ex_obs.shape[0], z_dim))
        params = {
            "left_encoder": left_encoder.init(rle, ex_obs)["params"],
            # PhiMap infers its input width from the example: the SLICED example in
            # hit_spec mode, so the first Dense is exactly d_c wide.
            "backward": backward.init(rb, ex_b_in)["params"],
            "forward": forward.init(rf, ex_feat, ex_z, ex_actions)["params"],
            "predictor": predictor.init(rp, ex_pred_in, ex_actions)["params"],
        }
        critic = cls(
            forward=forward, backward=backward, left_encoder=left_encoder,
            predictor=predictor,
            z_dim=z_dim, num_parallel=P, horizon=horizon, discount=config["discount"],
            ortho_coef=config["ortho_coef"], fb_pessimism_penalty=config["fb_pessimism_penalty"],
            train_goal_ratio=config["train_goal_ratio"], norm_z=config["norm_z"],
            f_tau=config["f_target_tau"], b_tau=config["b_target_tau"],
            backward_input=backward_input, blocks=blocks,
        )
        targets = {g: copy.deepcopy(params[g]) for g in critic.TARGETS}
        return critic, params, targets
