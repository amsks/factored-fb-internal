"""agents/fb/agent.py — Literal port of td_jepa/metamotivo/agents/fb/agent.py.

Goal: byte-equivalent algorithm to td_jepa, only differing where their config
infrastructure is replaced by our Hydra/kwarg-driven instance attributes.

Substitutions vs upstream:
    self.cfg.train.X      -> self.X         (hyperparams live on the agent)
    self.cfg.model.X      -> self.X         (model-side hyperparams too)
    self._model.X         -> self.model.X   (single-underscore convention)
    cudagraphs / compile  -> dropped (eager mode)
    safetensors / pickle  -> dropped (use torch.save state_dict)

Everything else — math, gradient paths, autocast wraps, optimizer parameter
lists, off-diagonal mask precomputation, soft-update wrapping — is preserved
line for line.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from agents.base import BaseAgent
from agents.fb.model import FBModel
from nn_models import _soft_update_params, eval_mode, weight_init


def coverage_weights(cube_xyz: Tensor, rho, alpha: float, clip: float) -> Tensor:
    """Goal-agnostic coverage-balancing weights w(g) from the offline cube density.

    rho = (H, [ex, ey, ez]) — a 3D histogram over cube xyz + bin edges, built from the
    offline data only (NO eval/task goals). Rare cells get weight > 1, common < 1; raw
    weight is capped at `clip` (rare-outlier guard) then mean-normalized so E[w]=1 (the
    loss scales are preserved; only the distribution over goals shifts).
    """
    H, (ex, ey, ez) = rho
    ix = (torch.bucketize(cube_xyz[:, 0].contiguous(), ex) - 1).clamp(0, H.shape[0] - 1)
    iy = (torch.bucketize(cube_xyz[:, 1].contiguous(), ey) - 1).clamp(0, H.shape[1] - 1)
    iz = (torch.bucketize(cube_xyz[:, 2].contiguous(), ez) - 1).clamp(0, H.shape[2] - 1)
    dens = H[ix, iy, iz] + 1e-6
    w = (dens.median() / dens) ** alpha  # rare -> >1, common -> <1
    w = w.clamp(min=1.0 / clip, max=clip)  # also lower-clip: dense cells must keep mass
    return (w / w.mean().clamp_min(1e-8)).detach()


def interaction_weights(eff_xyz: Tensor, cube_xyz: Tensor,
                        c: float, sigma: float, eps: float) -> Tensor:
    """IWR-style interaction-proximity weights w(g) (goal-agnostic, data-only).

    Ports Interaction-Weighted Resampling (Factored_URL_in_Manipulation) to the FB
    goal measure: w peaks where the effector-cube distance d(g) is at the contact
    threshold c (the interaction boundary), with window sigma and floor eps>0 so
    rho' keeps full support. Mean-normalized (E[w]=1) so loss scales are preserved;
    only the measure over goals tilts. Both inputs in the SAME (obs) frame.
    See PAPER/specs/2026-06-11-interaction-weighted-fb.md.
    """
    d = (eff_xyz - cube_xyz).norm(dim=-1)
    w = eps + torch.exp(-((d - c).pow(2)) / (2 * sigma ** 2))
    return (w / w.mean().clamp_min(1e-8)).detach()


def fb_successor_terms(diff: Tensor, off_diag: Tensor, off_diag_sum,
                       w: Tensor | None = None, weight_diag: bool = False):
    """FB successor loss split into (off-diagonal, diagonal).

    The off-diagonal (quadratic, goals g~rho) is weighted by w_j => a clean change of
    reference measure rho -> rho'. The diagonal (the -E[F^T B(s')] Bellman/Dirac term)
    is NOT an integral over an independently sampled goal: under L2(rho') its density
    ratio cancels (the immediate successor is delta_{s'}), so a PURE rho'-FB leaves it
    UNWEIGHTED (weight_diag=False, default). weight_diag=True instead applies w_i to it
    = extra TD pressure on rare achieved next-states (coverage-PRIORITIZED FB, a hybrid,
    not a pure measure change). w=None reproduces the original unweighted terms exactly.
    diff: [n_parallel, batch, batch]; w: [batch], mean-normalized.
    """
    n_parallel = diff.shape[0]
    if w is None:
        fb_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * n_parallel
        return fb_offdiag, fb_diag
    wcol = w.view(1, 1, -1)  # weight goal/column j ~ rho'
    fb_offdiag = 0.5 * ((diff * off_diag).pow(2) * wcol).sum() / off_diag_sum
    diag = torch.diagonal(diff, dim1=1, dim2=2)
    if weight_diag:
        fb_diag = -(diag * w.view(1, -1)).mean() * n_parallel   # prioritized
    else:
        fb_diag = -diag.mean() * n_parallel                     # pure rho' (Dirac cancels)
    return fb_offdiag, fb_diag


def ortho_cov(B: Tensor, w: Tensor | None = None) -> Tensor:
    """Backward-embedding covariance under rho': whiten B~=sqrt(w)*B so the ortho loss
    minimum is E_rho'[B B^T]=I. This is the whitening term the diagnosis blames (it
    isotropizes B over the bulk); reweighting it is the centerpiece of coverage-balancing.
    w=None reproduces the original B B^T.
    """
    if w is None:
        return torch.matmul(B, B.T)
    Bw = B * w.sqrt().unsqueeze(-1)
    return torch.matmul(Bw, Bw.T)


def expectile_loss(diff: Tensor, expectile: float) -> Tensor:
    """Asymmetric L2 (IQL value loss, https://arxiv.org/abs/2110.06169). `diff` is the
    residual target - V; positive residuals (target above V) are weighted by `expectile`,
    negative by (1 - expectile). expectile>0.5 pushes V toward an upper expectile of the
    target distribution; expectile=0.5 reduces to 0.5*MSE."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def iql_goal_target(success: Tensor, Vn: Tensor, discount: float) -> Tensor:
    """Goal-conditioned IQL Bellman target, byte-faithful to OGBench GCIQL
    (utils/datasets.py: masks = 1 - success, rewards = success - 1 with gc_negative=True;
    gciql.py: q = rewards + discount * masks * V_next). The bootstrap is MASKED OFF once
    the relabeled goal is reached (mask = 1 - success), anchoring V to 0 at the goal so the
    value is a bounded negative distance-to-go. Using the env `terminated` flag as the mask
    (never set in cube offline data) and a {0,+1} reward instead lets the goal state collect
    reward + gamma*V forever -> fixed point 1/(1-gamma) -> V/Q inflate and the actor collapses."""
    reward = success - 1.0          # {-1, 0}  (gc_negative)
    mask = 1.0 - success            # cut the bootstrap at the goal
    return reward + discount * mask * Vn


class FBAgent(BaseAgent):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        batch_size: int = 256,
        z_dim: int = 50,
        L_dim: int = 50,
        actor_encode_obs: bool = False,
        forward_cfg=None,
        backward_cfg=None,
        left_encoder_cfg=None,
        actor_cfg=None,
        obs_normalizer_cfg=None,
        rgb_encoder_cfg=None,
        augmentator_cfg=None,
        discount: float = 0.99,
        lr_f: float = 1e-4,
        lr_b: float = 1e-4,
        lr_actor: float = 1e-4,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 0.0,
        ortho_coef: float = 1.0,
        train_goal_ratio: float = 0.5,
        fb_pessimism_penalty: float = 0.0,
        actor_pessimism_penalty: float = 0.0,
        actor_std: float = 0.2,
        stddev_clip: float = 0.3,
        f_target_tau: float = 0.005,
        b_target_tau: float = 0.005,
        bc_coeff: float = 0.0,
        q_loss_coef: float = 0.0,
        reweight_alpha: float = 0.0,
        reweight_clip: float = 10.0,
        reweight_density_path: str | None = None,
        reweight_mode: str = "coverage",
        iwr_c: float = 0.03,
        iwr_sigma: float = 0.05,
        iwr_eps: float = 0.1,
        weight_diag: bool = False,
        weight_z: bool = False,
        train_goal_mode: str = "perm",
        decouple_actor_z: bool = False,
        critic_mode: str = "measure",
        iql_expectile: float = 0.9,
        iql_thresh: float = 0.04,
        onestep: bool = False,
        decondition_F: bool = False,
        d_norm: bool = False,
        bilinear_v: bool = False,
        critic_goal_mix: tuple = (1.0, 0.0, 0.0),
        actor_goal_mix: tuple = (1.0, 0.0, 0.0),
        iql_reward: str = "cube_xyz",
        goal_cond: bool = False,
        fixed_b: str = "none",
        amp: bool = False,
        device: str = "cpu",
    ):
        # Store every hyperparam as an instance attribute — direct stand-in
        # for td_jepa's self.cfg.train.X / self.cfg.model.X.
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.L_dim = L_dim
        self.actor_encode_obs = actor_encode_obs
        self.discount = discount
        self.lr_f = lr_f
        self.lr_b = lr_b
        self.lr_actor = lr_actor
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.ortho_coef = ortho_coef
        self.train_goal_ratio = train_goal_ratio
        self.fb_pessimism_penalty = fb_pessimism_penalty
        self.actor_pessimism_penalty = actor_pessimism_penalty
        self.actor_std = actor_std
        self.stddev_clip = stddev_clip
        self.f_target_tau = f_target_tau
        self.b_target_tau = b_target_tau
        self.bc_coeff = bc_coeff
        self.q_loss_coef = q_loss_coef
        self.amp = amp
        self.device = device
        self.device_type = str(device).split(":")[0]  # "cuda:0" -> "cuda" for autocast

        # Coverage-balanced (goal-agnostic) reweight under measure rho'. Default-off
        # (alpha=0) => byte-identical FB. Density built from offline cube positions only
        # (no eval/task goals). weight_diag: pure rho' (False, diagonal unweighted) vs
        # coverage-PRIORITIZED FB (True). weight_z: resample goal-derived z's by w so F
        # and the actor also see rare-goal z=B(g). See
        # docs/superpowers/specs/2026-05-25-coverage-balanced-fb.md.
        self.reweight_alpha = float(reweight_alpha)
        self.reweight_clip = float(reweight_clip)
        # reweight_mode: "coverage" = inverse cube-position density (original covbal);
        # "interaction" = IWR-style contact-proximity bump (interaction_weights).
        # alpha>0 remains the master switch for BOTH modes.
        self.reweight_mode = str(reweight_mode)
        self.iwr_c = float(iwr_c)
        self.iwr_sigma = float(iwr_sigma)
        self.iwr_eps = float(iwr_eps)
        self.weight_diag = bool(weight_diag)
        self.weight_z = bool(weight_z)
        # train_goal_mode: "perm" = goal-derived z's from batch-permuted next-states
        # (uniform buffer goals, original); "traj" = zHER: z = B(s_{t+k}) for the SAME
        # trajectory's discounted future (requires batch["future"], i.e. the buffer
        # sampled with future_gamma). The actor/F then practice tasks actually
        # reachable from their states. Spec: 2026-06-11-interaction-weighted-fb.md §1.4.
        self.train_goal_mode = str(train_goal_mode)
        # GCIQL-style decoupling: train the actor ONLY on reachable on-trajectory goals
        # z_actor=B(s_future) while the critic keeps the broad sample_mixed_z. Needs
        # batch["future"] (future_gamma). Measure-mode only; default off => byte-identical.
        self.decouple_actor_z = bool(decouple_actor_z)
        # FB-IQL: replace the measure loss with a goal-conditioned IQL critic on the
        # bilinear value Q=F.z (spec 2026-06-11-interaction-weighted-fb.md). 'measure'
        # = stock FB (byte-identical). Requires train_goal_mode='traj' (the goals).
        self.critic_mode = str(critic_mode)
        self.iql_expectile = float(iql_expectile)
        self.iql_thresh = float(iql_thresh)
        try:
            self._obs_dim = int(obs_space.shape[-1])
        except Exception:
            self._obs_dim = None
        # One-step FB (https://github.com/chongyi-zheng/onestep-fb): learn the BEHAVIOR
        # successor features — F is NOT conditioned on z (see _fb_z) and the FB loss uses
        # the dataset's own next action (SARSA, batch["next"]["action"]) instead of the
        # policy's. The actor then does one improvement step on <F(s,a), z>. Default off.
        self.onestep = bool(onestep)
        # decondition_F: feed zeros to F's task slot => phi(s,a) task-agnostic (bilinear-GCIQL).
        self.decondition_F = bool(decondition_F)
        # d_norm: scale the bilinear value F.z by 1/sqrt(z_dim) (GCIQL's phi.psi/sqrt(d)).
        self.d_norm = bool(d_norm)
        # bilinear_v: FB-IQL value head V=<phi_V(s),z>/sqrt(d) instead of MLP([s,z]) (matches Q's form).
        self.bilinear_v = bool(bilinear_v)
        # OGBench GCDataset goal mix [p_traj, p_cur, p_random] for the IQL critic / actor,
        # and iql_reward: 'cube_xyz' (privileged threshold) or 'goal_match' (GCIQL: success=goal is cur).
        self.critic_goal_mix = tuple(float(x) for x in critic_goal_mix)
        self.actor_goal_mix = tuple(float(x) for x in actor_goal_mix)
        self.iql_reward = str(iql_reward)
        self.goal_cond = bool(goal_cond)
        self.fixed_b = str(fixed_b)
        # goal_dim: the slice of obs fed to F/actor in V1 (full obs by default).
        self.goal_dim = int(obs_space.shape[0]) if self.goal_cond else 0
        # Only the goal variants use eval_context; baseline keeps legacy reward-inference.
        self._use_eval_context = self.goal_cond or self.fixed_b in ("cube_xyz", "identity", "identity_norm")
        self._eval_goal = None
        self._rho = None
        if self.reweight_alpha > 0:
            if not reweight_density_path:
                raise ValueError(
                    "reweight_alpha > 0 requires reweight_density_path "
                    "(silent fallback would make ablations unreadable)."
                )
            zd = np.load(reweight_density_path)
            self._rho = (
                torch.tensor(zd["H"], dtype=torch.float32, device=self.device),
                [torch.tensor(zd[k], dtype=torch.float32, device=self.device) for k in ("ex", "ey", "ez")],
            )

        # Build the model via a hook so subclasses (FBFlowBCAgent) can return
        # FBFlowBCModel directly — matches td_jepa's `cfg.model.build(...)`
        # one-shot construction. Building twice (parent first, child second)
        # would consume RNG and shift weight_init's offset.
        self.model = self._make_model(
            obs_space=obs_space,
            action_dim=action_dim,
            z_dim=z_dim,
            L_dim=L_dim,
            actor_encode_obs=actor_encode_obs,
            amp=amp,
            forward_cfg=forward_cfg,
            backward_cfg=backward_cfg,
            left_encoder_cfg=left_encoder_cfg,
            actor_cfg=actor_cfg,
            obs_normalizer_cfg=obs_normalizer_cfg,
            rgb_encoder_cfg=rgb_encoder_cfg,
            augmentator_cfg=augmentator_cfg,
            fixed_b=self.fixed_b,
            goal_dim=self.goal_dim,
            device=device,
        )
        # fixed_b in {cube_xyz, identity} overrides z_dim inside the model
        # (3 / obs_dim); sync so downstream (e.g. the FB-IQL value MLP) sizes match.
        self.z_dim = self.model.z_dim
        self.setup_training()
        self.model.to(self.device)

    def _make_model(self, **kwargs):
        """Construct the model. Subclasses override to return their variant."""
        return FBModel(**kwargs)

    @property
    def amp_dtype(self):
        return self.model.amp_dtype

    def setup_training(self) -> None:
        self.model.train(True)
        self.model.requires_grad_(True)
        self.model.apply(weight_init)
        self.model._prepare_for_train()  # ensure target nets are initialized AFTER applying the weights

        bw_params = list(self.model._backward_map.parameters()) + list(self.model._bw_encoder.parameters())
        self.backward_optimizer = (
            torch.optim.Adam(bw_params, lr=self.lr_b, weight_decay=self.weight_decay)
            if bw_params else None
        )
        self.forward_optimizer = torch.optim.Adam(
            list(self.model._forward_map.parameters())
            + list(self.model._left_encoder.parameters())
            + list(self.model._fw_encoder.parameters()),
            lr=self.lr_f,
            weight_decay=self.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model._actor.parameters(),
            lr=self.lr_actor,
            weight_decay=self.weight_decay,
        )

        # prepare parameter list (used by soft_update_params via torch._foreach_*)
        self._forward_map_paramlist = tuple(x for x in self.model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self.model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self.model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self.model._target_backward_map.parameters())
        self._left_encoder_paramlist = tuple(x for x in self.model._left_encoder.parameters())
        self._target_left_encoder_paramlist = tuple(x for x in self.model._target_left_encoder.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.batch_size, self.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        # FB-IQL critic: a goal-conditioned expectile value V(s, z) trained by IQL,
        # replacing the measure loss (critic_mode='iql'). The bilinear Q=F(s,a,z).z is
        # then a goal-reaching value (GCIQL's signal through FB's inner-product form).
        if self.critic_mode == "iql":
            obs_dim = self._obs_dim
            if self.bilinear_v:   # V(s,z) = <phi_V(s), z>/sqrt(d) -- matches the bilinear Q's form
                self._value_phi = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 512), torch.nn.LayerNorm(512), torch.nn.GELU(),
                    torch.nn.Linear(512, 512), torch.nn.GELU(),
                    torch.nn.Linear(512, self.z_dim),
                ).to(self.device)
                self.value_optimizer = torch.optim.Adam(self._value_phi.parameters(), lr=self.lr_f)
            else:
                self._value_mlp = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim + self.z_dim, 512), torch.nn.LayerNorm(512), torch.nn.GELU(),
                    torch.nn.Linear(512, 512), torch.nn.GELU(),
                    torch.nn.Linear(512, 1),
                ).to(self.device)
                self.value_optimizer = torch.optim.Adam(self._value_mlp.parameters(), lr=self.lr_f)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor | None = None,
                       goal_weights: torch.Tensor | None = None,
                       permute: bool = True, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks.
        # goal_weights (mean-normalized w over batch goals): when given, the goal-derived
        # z's are resampled ~ w (rare goals over-represented) so F and the actor also see
        # rare-goal z=B(g); None => uniform permutation (behavior-marginal, original).
        # permute=False (zHER/traj goals): keep train_goal ALIGNED with the batch — each
        # element's task is its own trajectory future. Only honored when goal_weights is
        # None (weights imply resampling).
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            z = self.model.sample_z(self.batch_size, device=self.device)
            g_used = torch.zeros((self.batch_size, train_goal.shape[-1]), device=self.device,
                                 dtype=train_goal.dtype) \
                if train_goal is not None else None
            if train_goal is not None:
                if goal_weights is None:
                    # permute=False (zHER/traj goals): keep train_goal aligned with the batch.
                    perm = torch.randperm(self.batch_size, device=self.device) if permute else None
                else:
                    probs = goal_weights.float() / goal_weights.float().sum().clamp_min(1e-8)
                    perm = torch.multinomial(probs, self.batch_size, replacement=True)
                tg = train_goal[perm] if perm is not None else train_goal
                goals = self.model._backward_map(tg)
                goals = self.model.project_z(goals)
                mask = torch.rand((self.batch_size, 1), device=self.device) < self.train_goal_ratio
                z = torch.where(mask, goals, z)
                g_used = torch.where(mask, tg, g_used)   # zero placeholder where z is random
        return z, g_used

    @torch.no_grad()
    def aug(self, obs, next_obs):
        """Augments observations when training from pixels, does nothing otherwise."""
        return self.model._augmentator(obs), self.model._augmentator(next_obs)

    def enc(self, obs, next_obs):
        """Encodes observations when training from pixels, does nothing otherwise."""
        obs = self.model._fw_encoder(obs)
        goal = self.model._bw_encoder(next_obs)
        with torch.no_grad():
            next_obs = self.model._fw_encoder(next_obs)
        return obs, next_obs, goal

    def _coverage_weights(self, batch) -> torch.Tensor | None:
        """Goal-measure weight w(g) over the next-state (the rho_G variable), per
        reweight_mode: "coverage" = inverse cube-xyz density; "interaction" = IWR-style
        contact-proximity bump (effector-cube distance in the obs frame, slices 12:15 /
        19:22). Goal-agnostic in both modes (offline data only, never eval/task goals).
        Returns None when reweighting is off => byte-identical."""
        if self.reweight_alpha <= 0:
            return None
        if self.reweight_mode == "interaction":
            nobs = batch["next"]["observation"].to(self.device).float()
            return interaction_weights(nobs[:, 12:15], nobs[:, 19:22],
                                       self.iwr_c, self.iwr_sigma, self.iwr_eps)
        if self._rho is None:
            return None
        cube = batch["next"]["physics"][:, 14:17].to(self.device).float()  # s+ cube xyz
        return coverage_weights(cube, self._rho, self.reweight_alpha, self.reweight_clip)

    @property
    def _value_scale(self) -> float:
        """Bilinear-value normalizer: 1/sqrt(d) when d_norm (GCIQL's phi.psi/sqrt(d)),
        else 1 (stock FB's unscaled <F,z>). d = z_dim = the inner-product dimension."""
        return float(self.z_dim) ** -0.5 if self.d_norm else 1.0

    def _fb_z(self, z: torch.Tensor) -> torch.Tensor:
        """z fed to the forward map. F is made z-independent (=> phi(s,a), task-agnostic)
        for one-step FB (behavior successor features) or decondition_F (bilinear-GCIQL) =>
        feed zeros. Otherwise pass z through. The <F, z> dot products keep the real z."""
        return torch.zeros_like(z) if (self.onestep or self.decondition_F) else z

    def _fb_Q_heads(self, left, z, action, fmap):
        """Per-head bilinear value <F(s,a,_fb_z(z)), z> * value_scale -> [num_parallel, batch].
        Centralizes decondition_F (via _fb_z) and d_norm (via _value_scale) for critic + actor."""
        return (fmap(left, self._fb_z(z), action) * z).sum(-1) * self._value_scale

    def _V(self, obs, z):
        """IQL expectile value V(s,z). bilinear_v: <phi_V(s), z>/sqrt(d) (matches the bilinear
        Q's form); else the legacy MLP([s, z])."""
        if self.bilinear_v:
            return (self._value_phi(obs) * z).sum(-1) * self._value_scale
        return self._value_mlp(torch.cat([obs, z], dim=-1)).squeeze(-1)

    def _sample_goal_mix(self, cur_obs, traj_obs, mix):
        """OGBench GCDataset goal sampling. mix=[p_traj, p_cur, p_random]; returns
        (goal_obs, is_cur). is_cur (goal IS the current state) is the GCIQL success/reach
        flag (datasets.py: successes = idxs==goal_idxs). random goal = a batch permutation
        (a random buffer state). cur/traj/random selected per-row from the categorical mix."""
        B = cur_obs.shape[0]
        p = torch.as_tensor(mix, device=cur_obs.device, dtype=torch.float32)  # [traj, cur, rand]
        types = torch.multinomial(p.expand(B, -1), 1).squeeze(-1)             # 0=traj,1=cur,2=rand
        rand_obs = cur_obs[torch.randperm(B, device=cur_obs.device)]
        is_cur = types == 1
        goal = torch.where(is_cur.unsqueeze(-1), cur_obs,
                           torch.where((types == 2).unsqueeze(-1), rand_obs, traj_obs))
        return goal, is_cur

    def _update_fb_iql(self, obs, action, next_obs, discount, goal_enc, z, batch, success_override=None):
        """FB-IQL critic: train the bilinear value Q=F(s,a,z).z as a goal-reaching value
        by IQL (expectile V) on trajectory-future goals, replacing the measure loss.
        z is goal-conditioned (= B(future)), so grad flows into B end-to-end. The FB
        identity is kept usable by the same ortho whitening on B. obs/next_obs are the
        fw-encoded normalized states (Identity for state). discount already carries
        (1-terminated). goal_enc is the bw-encoded future goal (z's source)."""
        # mean over the num_parallel forward heads (matches actor's Q reduction at penalty 0)
        def _Q(left, z_, a, fmap):
            return self._fb_Q_heads(left, z_, a, fmap).mean(0)

        # (1) IQL value V(s,z) <- expectile of the TARGET Q (frozen). V owns its optimizer.
        with torch.no_grad():
            tQ = _Q(self.model._target_left_encoder(obs), z, action, self.model._target_forward_map)
        V = self._V(obs, z.detach())
        value_loss = expectile_loss(tQ - V, self.iql_expectile)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        # (2) critic Q(s,a,z) <- GCIQL goal-conditioned Bellman target. success = reached the
        # relabeled goal (cube xyz within iql_thresh); the bootstrap is masked OFF at the goal
        # (see iql_goal_target) so V stays a bounded negative distance-to-go. NOTE: the goal
        # mask, NOT the env `terminated` flag (which is never set in cube offline data), is what
        # cuts the bootstrap.
        if success_override is not None:                # GCIQL goal_match: success = goal is cur
            success = success_override.to(self.device).float()  # [B]
        else:                                           # legacy cube_xyz threshold
            nphys = batch["next"]["physics"][:, 14:17].to(self.device).float()
            gphys = batch["future"]["physics"][:, 14:17].to(self.device).float()
            success = (nphys - gphys).norm(dim=-1).lt(self.iql_thresh).float()  # [B]
        with torch.no_grad():
            Vn = self._V(next_obs, z.detach())
            target_Q = iql_goal_target(success, Vn, self.discount)
        Q = _Q(self.model._left_encoder(obs), z, action, self.model._forward_map)
        q_loss = (Q - target_Q).pow(2).mean()

        # ortho whitening on B keeps it a valid goal embedding for zero-shot inference.
        B = self.model._backward_map(goal_enc)
        Cov = ortho_cov(B, None)
        orth_loss = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum - Cov.diag().mean()
        fb_loss = q_loss + self.ortho_coef * orth_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        if self.backward_optimizer is not None:        # None for fixed_b=identity (param-free B)
            self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.forward_optimizer.step()
        if self.backward_optimizer is not None:
            self.backward_optimizer.step()
        return {"q_loss": q_loss.detach(), "value_loss": value_loss.detach(),
                "orth_loss": orth_loss.detach(), "iql_r_pos": success.mean(),
                "Q_mean": Q.mean().detach(), "V_mean": V.mean().detach(),
                "B_norm": B.norm(dim=-1).mean().detach()}

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:
        obs, action, next_obs, terminated = (
            batch["observation"].to(self.device).float(),
            batch["action"].to(self.device).float(),
            batch["next"]["observation"].to(self.device).float(),
            batch["next"]["terminated"].to(self.device),
        )
        if terminated.dtype == torch.bool:
            discount = self.discount * ~terminated
        else:
            discount = self.discount * (1.0 - terminated)
        discount = discount.reshape(-1, 1)  # [B,1] => row-wise discount in Ms - discount*target_M

        self.model._obs_normalizer(obs)
        self.model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self.model._obs_normalizer):
            obs, next_obs = self.model._obs_normalizer(obs), self.model._obs_normalizer(next_obs)

        obs, next_obs = self.aug(obs, next_obs)
        obs, next_obs, goal = self.enc(obs, next_obs)

        # Coverage-balanced reweight (goal-agnostic): same measure rho' for every
        # goal-side term. None when alpha=0 => byte-identical. Computed BEFORE z so the
        # goal-derived z's can be resampled by w when weight_z is set.
        w = self._coverage_weights(batch)
        train_goal, permute = goal, True
        if self.train_goal_mode == "traj" and "future" in batch:
            # zHER: each element's task z = B(its own discounted trajectory future).
            fut = batch["future"]["observation"].to(self.device).float()
            with torch.no_grad(), eval_mode(self.model._obs_normalizer):
                fut = self.model._obs_normalizer(fut)
            train_goal, permute = self.model._bw_encoder(fut), False
        clip_grad_norm = self.clip_grad_norm if self.clip_grad_norm > 0 else None

        # --- FB-IQL: goal-conditioned expectile critic on the bilinear value, replacing
        # the measure loss (z = B(trajectory future), all goal-conditioned). ---
        if self.critic_mode == "iql":
            assert "future" in batch, "critic_mode='iql' requires train_goal_mode='traj'"
            if self.iql_reward == "goal_match":
                # OGBench GCDataset: value/actor goals from {cur, traj, random} mix; success =
                # goal IS the current state (cur-goal anchor). z = B(goal); grad => trains B.
                cur_raw = batch["observation"].to(self.device).float()
                traj_raw = batch["future"]["observation"].to(self.device).float()
                vgoal_raw, is_cur = self._sample_goal_mix(cur_raw, traj_raw, self.critic_goal_mix)
                agoal_raw, _ = self._sample_goal_mix(cur_raw, traj_raw, self.actor_goal_mix)
                with torch.no_grad(), eval_mode(self.model._obs_normalizer):
                    vgoal_n = self.model._obs_normalizer(vgoal_raw)
                    agoal_n = self.model._obs_normalizer(agoal_raw)
                goal_enc = self.model._bw_encoder(vgoal_n)            # grad -> trains B end-to-end
                z = self.model.project_z(self.model._backward_map(goal_enc))
                z_actor = self.model.project_z(self.model._backward_map(self.model._bw_encoder(agoal_n)))
                success_override = is_cur
            else:
                goal_enc = train_goal
                z = self.model.project_z(self.model._backward_map(train_goal))  # grad => trains B
                z_actor, success_override = z, None
            metrics = self._update_fb_iql(obs, action, next_obs, discount, goal_enc, z, batch,
                                          success_override=success_override)
            metrics.update(self.update_actor(obs=obs.detach(), action=action,
                                             z=z_actor.detach(), clip_grad_norm=clip_grad_norm))
            with torch.no_grad():
                _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.f_target_tau)
                if len(self._backward_map_paramlist):   # empty for fixed_b=identity (param-free B)
                    _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.b_target_tau)
                if len(self._left_encoder_paramlist):
                    _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.f_target_tau)
            return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}

        z, g_used = self.sample_mixed_z(train_goal=train_goal, permute=permute,
                                        goal_weights=(w if self.weight_z else None))
        z = z.clone()
        self._g_used = g_used if self.goal_cond else None   # stashed for update_fb/update_actor

        q_loss_coef = self.q_loss_coef if self.q_loss_coef > 0 else None

        # One-step FB: bootstrap with the dataset's next action (SARSA), not the policy's.
        next_action_override = None
        if self.onestep:
            next_action_override = batch["next"]["action"].to(self.device).float()

        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=goal,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
            w=w,
            weight_diag=self.weight_diag,
            next_action_override=next_action_override,
            goal_obs=self._g_used,
        )
        # GCIQL-style value/actor decoupling: the critic (measure) keeps the broad z above
        # (random dirs + B(buffer goals) + off-diag negatives); the actor instead practices
        # ONLY reachable on-trajectory goals z_actor = B(s_future), like GCIQL's
        # actor_p_trajgoal=1.0. Default off => z_actor is z (byte-identical).
        z_actor = z
        if self.decouple_actor_z and "future" in batch:
            fut = batch["future"]["observation"].to(self.device).float()
            with torch.no_grad(), eval_mode(self.model._obs_normalizer):
                fut = self.model._obs_normalizer(fut)
            z_actor = self.model.project_z(
                self.model._backward_map(self.model._bw_encoder(fut))).detach()
        metrics.update(
            self.update_actor(
                obs=obs.detach(),
                action=action,
                z=z_actor,
                clip_grad_norm=clip_grad_norm,
                goal=self._g_used,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.f_target_tau)
            if len(self._backward_map_paramlist):
                _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.b_target_tau)
            if len(self._left_encoder_paramlist):
                _soft_update_params(self._left_encoder_paramlist, self._target_left_encoder_paramlist, self.f_target_tau)

        # convert tensor metrics to floats for downstream logging (train.py expects floats)
        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        gk = {} if goal is None else {"goal": goal}
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            dist = self.model._actor(obs, z, self.actor_std, **gk)
            action = dist.sample(clip=self.stddev_clip)
        return action

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
        w: torch.Tensor | None = None,
        weight_diag: bool = False,
        next_action_override: torch.Tensor | None = None,
        goal_obs: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        zf = self._fb_z(z)  # one-step FB: zeros (F is z-independent); else z
        # V1 goal-conditioning: the forward map AND the policy that picks next_action
        # also see the (raw) goal observation. goal_obs=None => byte-identical to baseline.
        fwd_kw = {} if goal_obs is None else {"goal": goal_obs}
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                next_left_enc = self.model._target_left_encoder(next_obs)  # batch x L_dim
                actor_in = next_left_enc if self.actor_encode_obs else next_obs
                # One-step FB uses the dataset's next action (SARSA); else sample from policy.
                next_action = (next_action_override if next_action_override is not None
                               else self.sample_action_from_norm_obs(actor_in, z, goal_obs))
                target_Fs = self.model._target_forward_map(next_left_enc, zf, next_action, **fwd_kw)  # num_parallel x batch x z_dim
                target_B = self.model._target_backward_map(goal)  # batch x z_dim
                target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
                _, _, target_M = self.get_targets_uncertainty(target_Ms, self.fb_pessimism_penalty)  # batch x batch

            # compute FB loss
            left_enc = self.model._left_encoder(obs)  # batch x L_dim
            Fs = self.model._forward_map(left_enc, zf, action, **fwd_kw)  # num_parallel x batch x z_dim
            B = self.model._backward_map(goal)  # batch x z_dim
            Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

            diff = Ms - discount * target_M  # num_parallel x batch x batch
            # Coverage-balanced reweight: the SAME goal measure rho' (w) is applied to
            # every goal-side term — successor off-diag + diag here, orthonormality below,
            # q-cov below — so the trained model is a valid FB in L2(rho'). w=None =>
            # byte-identical to current FB.
            fb_offdiag, fb_diag = fb_successor_terms(diff, self.off_diag, self.off_diag_sum, w, weight_diag)
            fb_loss = fb_offdiag + fb_diag

            # orthonormality loss for backward embedding (whitened under rho').
            # Disabled for any fixed-B (B has no params: privileged cube xyz, or identity).
            if self.fixed_b == "none":
                Cov = ortho_cov(B, w)
                orth_loss_diag = -Cov.diag().mean()
                orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
                orth_loss = orth_loss_offdiag + orth_loss_diag
                fb_loss += self.ortho_coef * orth_loss
            else:
                orth_loss = orth_loss_diag = orth_loss_offdiag = torch.zeros(1, device=z.device, dtype=z.dtype)

            q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
            if q_loss_coef is not None:
                with torch.no_grad():
                    next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                    _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.fb_pessimism_penalty)  # batch
                    # FP32 + ridge for a stable solve (rho'-weighting can lower the
                    # effective sample size and ill-condition the covariance).
                    with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=False):
                        B32 = B.float()
                        if w is None:
                            cov = B32.T @ B32 / B32.shape[0]  # z_dim x z_dim
                        else:
                            cov = (B32 * w.float().unsqueeze(-1)).T @ B32 / B32.shape[0]  # E_rho'[B B^T]
                        cov = cov + 1e-4 * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
                    B_inv_conv = torch.linalg.solve(cov, B32, left=False)
                    implicit_reward = (B_inv_conv * z.float()).sum(dim=-1).to(z.dtype)  # batch
                    target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                    expanded_targets = target_Q.expand(Fs.shape[0], -1)
                Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
                q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
                fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        if self.backward_optimizer is not None:
            self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._backward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model._left_encoder.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        if self.backward_optimizer is not None:
            self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
            if w is not None:
                ess = (w.sum() ** 2) / w.pow(2).sum().clamp_min(1e-8)
                output_metrics.update({
                    "reweight_w_mean": w.mean(),
                    "reweight_w_min": w.min(),
                    "reweight_w_max": w.max(),
                    "reweight_w_std": w.std(),
                    "reweight_ess_frac": ess / w.numel(),  # 1.0 = uniform, ->0 = few dominate
                })
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
        goal: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        # V1 goal-conditioning lives in the FlowBC actor override; the TD3 path ignores
        # `goal` (accepted only so the goal-aware update() call doesn't break non-FlowBC FB).
        return self.update_td3_actor(obs=obs, action=action, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor, clip_grad_norm: float | None
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp):
            with torch.no_grad():
                left_enc = self.model._left_encoder(obs)
            actor_in = left_enc if self.actor_encode_obs else obs
            dist = self.model._actor(actor_in, z, self.actor_std)
            actor_action = dist.sample(clip=self.stddev_clip)
            Qs = self._fb_Q_heads(left_enc, z, actor_action, self.model._forward_map)  # num_parallel x batch
            _, _, Q = self.get_targets_uncertainty(Qs, self.actor_pessimism_penalty)  # batch
            actor_loss = -Q.mean()

            # compute bc loss
            bc_error = torch.tensor([0.0], device=action.device)
            if self.bc_coeff > 0:
                bc_error = F.mse_loss(actor_action, action)
                bc_loss = self.bc_coeff * bc_error
                actor_loss = (actor_loss / Qs.abs().mean().detach()) + bc_loss

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "bc_error": bc_error.detach() if isinstance(bc_error, torch.Tensor) else torch.tensor(bc_error),
            "q": Q.mean().detach(),
        }

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    # ── Rollout / eval API (kept from previous impl; not in td_jepa's agent.py
    #     because their rollout calls model.act directly via extract_model) ── #

    @torch.no_grad()
    def act(self, obs: Tensor, z: Tensor, *, eval_mode: bool = True) -> Tensor:
        obs = obs.to(self.device).float()
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.model.act(obs, z, mean=True)

    @torch.no_grad()
    def eval_context(self, *, env, domain, task, goal_obs=None):
        """Goal-conditioned eval context. Returns (z, metrics) and, for V1, stashes the
        NORMALIZED goal observation on self._eval_goal for goal-cond act()."""
        if not self._use_eval_context:
            raise AttributeError("baseline uses reward-inference _infer_z")
        if self.fixed_b == "cube_xyz":
            from agents.fb.fixed_backward import scale_goal_xyz
            goal_xyzs = env.unwrapped.cur_task_info["goal_xyzs"]      # [num_cubes,3], metres
            g = torch.as_tensor(goal_xyzs[0], dtype=torch.float32, device=self.device)
            z = self.model.project_z(scale_goal_xyz(g).unsqueeze(0)).squeeze(0)
            self._eval_goal = None
            return z, {"goal_source": 0.0}
        if self.fixed_b in ("identity", "identity_norm"):
            # z = the goal observation via B=identity. 'identity' => z=g (norm_z off);
            # 'identity_norm' => z=sqrt(d)*g/||g|| (project_z normalizes). No actor goal-cond.
            if goal_obs is None:
                raise ValueError(f"fixed_b='{self.fixed_b}' eval_context requires goal_obs (g*) from the evaluator")
            g_obs = goal_obs.to(self.device).float()
            if g_obs.dim() == 1:
                g_obs = g_obs.unsqueeze(0)
            z = self.model.project_z(self.model.backward_map(g_obs)).squeeze(0)
            self._eval_goal = None
            return z, {"goal_source": 1.0}
        # V1: z = B(g*) from a goal OBSERVATION sourced by the evaluator; stash NORMALIZED g*.
        if goal_obs is None:
            raise ValueError("goal_cond eval_context requires goal_obs (g*) from the evaluator")
        g_obs = goal_obs.to(self.device).float()
        if g_obs.dim() == 1:
            g_obs = g_obs.unsqueeze(0)
        z = self.model.project_z(self.model.backward_map(g_obs)).squeeze(0)   # backward_map normalizes raw obs
        self._eval_goal = self.model._normalize(g_obs)                         # actor goal in normalized space (matches training)
        return z, {"goal_source": 1.0}

    @torch.no_grad()
    def infer_z(self, obs: Tensor, reward: Tensor) -> Tensor:
        return self.model.reward_inference(obs.to(self.device), reward.to(self.device))

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
            # fixed-B (e.g. fixed_b=cube_xyz) has no backward params => no optimizer.
            "backward_optimizer": (
                self.backward_optimizer.state_dict() if self.backward_optimizer is not None else None
            ),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.forward_optimizer.load_state_dict(state["forward_optimizer"])
        if self.backward_optimizer is not None and state.get("backward_optimizer") is not None:
            self.backward_optimizer.load_state_dict(state["backward_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
