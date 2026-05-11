"""
agents/fb/flow_bc/model.py — FBFlowBCModel.

Extends FBModel with an unconditional flow-matching velocity field
`_actor_vf(obs, x_t, t) -> velocity`. The standard noise-conditioned
`_actor` is kept from FBModel; the velocity field is used during
training to define the BC distillation target.
"""

from __future__ import annotations

from typing import Optional

from agents.fb.model import FBModel
from nn_models import SimpleVectorFieldArchiConfig, weight_init


class FBFlowBCModel(FBModel):
    def __init__(
        self,
        obs_space,
        action_dim: int,
        actor_vf_cfg: Optional[SimpleVectorFieldArchiConfig] = None,
        **kwargs,
    ):
        super().__init__(obs_space=obs_space, action_dim=action_dim, **kwargs)
        if actor_vf_cfg is None:
            actor_vf_cfg = SimpleVectorFieldArchiConfig(hidden_dim=512, hidden_layers=4)

        # Build the velocity field on the same input space as the noise-conditioned
        # actor (i.e. _fw_encoder.output_space when actor_encode_obs=False).
        actor_in_space = (
            self._left_encoder.output_space
            if self.actor_encode_obs
            else self._fw_encoder.output_space
        )
        self._actor_vf = actor_vf_cfg.build(actor_in_space, action_dim)
        self._actor_vf.apply(weight_init)
        self.to(self.device)
