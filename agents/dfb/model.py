"""agents/dfb/model.py — Density-FB networks, torch port of the jax reference.

Ground truth: tools/wandb_mode_shim/gciql_flow.py network registration
(measure_density branch, `create`, ~line 5386):

    density_f = ensemblize(MLP, 2)((*value_hidden_dims, d), activate_final=False,
                                   layer_norm=True)     # 2 heads, matches double_q
    density_b = MLP((*value_hidden_dims, d), ...)        # single tower, NO output norm
    target twins of both (frozen copies).

The ogbench MLP convention is Dense -> GELU -> LayerNorm per hidden layer
(activation BEFORE the norm — third_party/ogbench/impls/utils/networks.py:52-60),
final Dense bare. DensityMLP reproduces that ordering; the 2-head F tower is the
same stack built from DenseParallel/ParallelLayerNorm (the torch stack's batch-
ensemble idiom, agents/fb ForwardMap).

Actors follow the torch FB stack: NoiseConditionedActor + unconditioned
VectorField for the flow arm (agents/fb/flow_bc — the jax flagship recipe has
actor_vf_cond='none' and actor_cond='goal', so the one-shot actor's "z" slot
carries the RAW GOAL OBSERVATION and the BC flow is goal-agnostic), or the stock
gaussian Actor for the ddpgbc arm.

Extension points deliberately left open (spec 24/29/30 z interfaces are OUT of
scope for this port): the F-tower input is assembled by DFBModel.f_input(), and
the actor conditioning by the agent — a latent-index variant overrides only
those two seams.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from nn_models import (
    Actor,
    NoiseConditionedActor,
    NoiseConditionedActorArchiConfig,
    ParallelLayerNorm,
    SimpleActorArchiConfig,
    SimpleVectorFieldArchiConfig,
    DenseParallel,
    linear,
)


class DensityMLP(nn.Module):
    """ogbench-convention MLP: (Linear -> GELU -> LayerNorm) x len(hidden) -> Linear(out).

    n_parallel=1 -> [B, out]; n_parallel=2 -> input [B, in] is expanded to
    [2, B, in] and the stack runs batched heads -> [2, B, out] (the jax
    `ensemblize(MLP, 2)` twin).
    """

    def __init__(self, in_dim: int, hidden_dims, out_dim: int, n_parallel: int = 1):
        super().__init__()
        self.n_parallel = n_parallel
        seq = []
        d = in_dim
        for h in hidden_dims:
            seq += [linear(d, h, n_parallel), nn.GELU(),
                    ParallelLayerNorm(h, n_parallel) if n_parallel > 1 else nn.LayerNorm(h)]
            d = h
        seq += [linear(d, out_dim, n_parallel)]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_parallel > 1 and x.dim() == 2:
            x = x.expand(self.n_parallel, -1, -1)
        return self.net(x)


class DFBModel(nn.Module):
    """Two-tower normalized successor density + actor, torch idiom of agents/fb.

    density_f : [obs, action, goal_obs] -> [2, B, d]   (2 heads)
    density_b : eps-scaled query coords [M, d_c] -> [M, d]   (single tower)
    actor     : flow  -> NoiseConditionedActor(obs, goal, noise) + VectorField(obs)
                ddpgbc -> gaussian Actor(obs, goal, std)
    """

    def __init__(
        self,
        obs_space,
        action_dim: int,
        query_dim: int,
        density_dim: int = 50,
        hidden_dims=(512, 512, 512),
        actor_kind: str = "flow",
        actor_cfg: NoiseConditionedActorArchiConfig | SimpleActorArchiConfig | None = None,
        actor_vf_cfg: SimpleVectorFieldArchiConfig | None = None,
        actor_std: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        assert len(obs_space.shape) == 1, "state observations only (pixel path not ported)"
        obs_dim = int(obs_space.shape[0])
        self.obs_space = obs_space
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.query_dim = query_dim
        self.density_dim = density_dim
        self.actor_kind = actor_kind
        self.actor_std = actor_std
        self.device = device

        # F conditions on the FULL goal observation (jax: concat([obs, acts, g])).
        self.density_f = DensityMLP(obs_dim + action_dim + obs_dim, hidden_dims,
                                    density_dim, n_parallel=2)
        self.density_b = DensityMLP(query_dim, hidden_dims, density_dim, n_parallel=1)

        if actor_kind == "flow":
            if actor_cfg is None:
                actor_cfg = NoiseConditionedActorArchiConfig(
                    hidden_dim=512, hidden_layers=2, embedding_layers=2)
            if actor_vf_cfg is None:
                actor_vf_cfg = SimpleVectorFieldArchiConfig(hidden_dim=512, hidden_layers=4)
            # z slot carries the raw goal obs (actor_cond='goal').
            self._actor = actor_cfg.build(obs_space, obs_dim, action_dim)
            # BC flow is goal-agnostic (actor_vf_cond='none').
            self._actor_vf = actor_vf_cfg.build(obs_space, action_dim)
        elif actor_kind == "ddpgbc":
            if actor_cfg is None:
                actor_cfg = SimpleActorArchiConfig(hidden_dim=512, hidden_layers=2,
                                                   embedding_layers=2)
            self._actor = actor_cfg.build(obs_space, obs_dim, action_dim)
            self._actor_vf = None
        else:
            raise ValueError(f"actor_kind must be 'flow' or 'ddpgbc', got {actor_kind!r}")

        self.to(device)

    # -- seams (latent-index variants override exactly these) -----------------
    def f_input(self, obs: torch.Tensor, action: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, action, cond], dim=-1)

    def _prepare_for_train(self) -> None:
        self.target_density_f = copy.deepcopy(self.density_f)
        self.target_density_b = copy.deepcopy(self.density_b)
        for p in self.target_density_f.parameters():
            p.requires_grad_(False)
        for p in self.target_density_b.parameters():
            p.requires_grad_(False)

    # -- actor forward helpers -------------------------------------------------
    def one_shot_action(self, obs: torch.Tensor, goal: torch.Tensor,
                        noise: torch.Tensor) -> torch.Tensor:
        assert self.actor_kind == "flow"
        return self._actor(obs, goal, noise)

    def gaussian_dist(self, obs: torch.Tensor, goal: torch.Tensor,
                      std: float | None = None):
        assert self.actor_kind == "ddpgbc"
        return self._actor(obs, goal, self.actor_std if std is None else std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if self.actor_kind == "flow":
            noise = torch.randn((obs.shape[0], self.action_dim),
                                device=obs.device, dtype=obs.dtype)
            return torch.clamp(self.one_shot_action(obs, goal, noise), -1.0, 1.0)
        dist = self.gaussian_dist(obs, goal)
        return torch.clamp(dist.mean, -1.0, 1.0)
