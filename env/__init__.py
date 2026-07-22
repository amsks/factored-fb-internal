"""env/ — the unified environment layer for the factored algorithm tree.

A single generic API (get_adapter) that serves EVERY environment: OGBench and our collected
Box2D / Metaworld / ManiSkill datasets, behind one DomainAdapter seam. The algorithm in
impls/ never imports from here — impls/main.py (the harness entrance) does. Adding an
environment is a registry entry (env/registry.py) + a thin domain config, no algorithm change.
"""
from env.base import DomainAdapter, get_adapter
from env import registry

__all__ = ["DomainAdapter", "get_adapter", "registry"]
