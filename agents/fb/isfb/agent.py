"""agents/fb/isfb/agent.py — SG-ISFB agent (PAPER/fb-modified/isfb.tex).

Subclasses FBFlowBCAgent (flow actor + z-cond inherited UNCHANGED). The ONLY
addition over standard FB is the critic update: the successor measure M is trained
with IQL's in-sample improvement operator (scalar expectile L_V on q=<F,z> vs
v=<F_V,z>; advantage-weighted vector regression L_N fitting N=<F_V,B> to sg M;
measure TD L_M bootstrapping through N^- at the next state, NO a'~pi)."""
from __future__ import annotations

from agents.fb.flow_bc.agent import FBFlowBCAgent
from agents.fb.isfb.model import ISFBModel


class ISFBAgent(FBFlowBCAgent):
    def __init__(self, *args, measure_insample: str = "scalar_gate_hard",
                 alpha_n: float = 1.0, alpha_m: float = 1.0,
                 awr_beta: float = 3.0, awr_wmax: float = 100.0, **kwargs):
        # Stash ISFB knobs BEFORE super().__init__ (which calls setup_training()).
        self.measure_insample = str(measure_insample)
        self.alpha_n = float(alpha_n)
        self.alpha_m = float(alpha_m)
        self.awr_beta = float(awr_beta)
        self.awr_wmax = float(awr_wmax)
        super().__init__(*args, **kwargs)

    def _make_model(self, **kwargs):
        return ISFBModel(actor_vf_cfg=self._actor_vf_cfg, **kwargs)

    def setup_training(self) -> None:
        super().setup_training()
        # F_V rides the forward optimizer (lr_f), matching FB's optimizer grouping.
        self.forward_optimizer.add_param_group(
            {"params": list(self.model._value_forward_map.parameters())}
        )
        self._value_forward_map_paramlist = tuple(self.model._value_forward_map.parameters())
        self._target_value_forward_map_paramlist = tuple(self.model._target_value_forward_map.parameters())
