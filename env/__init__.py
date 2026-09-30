"""env/ — the environment layer: one API (get_adapter) serving OGBench, ExoRL,
MetaWorld and ManiSkill behind the DomainAdapter protocol. The critics and actors never
import from here; impls/main.py does. Adding an environment is a registry entry
(env/registry.py) plus a domain config.
"""
from env.base import DomainAdapter, get_adapter
from env.config import make_config
from env import registry

__all__ = ["DomainAdapter", "get_adapter", "make_config", "registry"]
