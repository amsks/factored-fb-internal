from __future__ import annotations

__all__ = ["CRLIWSAgentTorch", "CRLIWSAgent"]


def __getattr__(name: str):
    if name in ("CRLIWSAgentTorch", "CRLIWSAgent"):
        from agents.crl_iws.agent_torch import CRLIWSAgentTorch
        return CRLIWSAgentTorch
    if name == "CRLIWSAgentJax":
        from agents.crl_iws.agent_jax import CRLIWSAgent as CRLIWSAgentJax
        return CRLIWSAgentJax
    raise AttributeError(name)
