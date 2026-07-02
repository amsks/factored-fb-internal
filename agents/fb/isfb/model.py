"""agents/fb/isfb/model.py — SG-ISFB model. Subclasses FBFlowBCModel (flow actor)
and adds ONE module: a value-forward map F_V(s,z), a second ForwardMap fed a zeroed
action. Everything else (F, B, actor, project_z, normalizer) is standard FB."""
from __future__ import annotations

import copy
from typing import Optional

from agents.fb.flow_bc.model import FBFlowBCModel
from nn_models import ForwardArchiConfig


class ISFBModel(FBFlowBCModel):
    def __init__(self, obs_space, action_dim: int, forward_cfg: Optional[ForwardArchiConfig] = None, **kwargs):
        # forward_cfg is consumed by FBModel to build _forward_map; capture it so we
        # can build F_V with the IDENTICAL architecture.
        self._isfb_forward_cfg = forward_cfg if forward_cfg is not None else ForwardArchiConfig()
        super().__init__(obs_space=obs_space, action_dim=action_dim, forward_cfg=forward_cfg, **kwargs)
        self._value_forward_map = self._isfb_forward_cfg.build(
            self._left_encoder.output_space, self.z_dim, action_dim, goal_dim=self.goal_dim
        )
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_value_forward_map = copy.deepcopy(self._value_forward_map)
