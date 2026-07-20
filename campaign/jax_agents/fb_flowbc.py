"""campaign/jax_agents/fb_flowbc.py — FB + FlowBC (flow-matching actor), same FBAgent
class with config flow=1. Registers agents.agents['fb_flowbc'].
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fb import FBAgent, get_config as _fb_config


def get_config():
    c = _fb_config()
    with c.unlocked():
        c.agent_name = "fb_flowbc"
        c.flow = 1
        c.bc_coeff = 3.0        # manipulation default (matches torch fb_flowbc)
        c.flow_steps = 10
    return c


import agents as _ogbench_agents  # noqa: E402
_ogbench_agents.agents["fb_flowbc"] = FBAgent
