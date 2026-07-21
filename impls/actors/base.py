"""Actor protocol for the factored critic×actor seam."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Actor(Protocol):
    GROUPS: tuple           # module groups this actor owns
    NEEDS_V: bool           # awr=True; ddpgbc/flowbc/fql=False
    REQUIRES_COND: frozenset  # cond keys this actor hard-requires (flowbc: {"z"}; others: {})

    @classmethod
    def build(cls, ex_obs, ex_actions, config): ...
    def sample_bootstrap(self, net, batch, cond, rng): ...
    def loss(self, params, target, batch, critic, cond, boot): ...
    def sample_actions(self, net, obs, cond, rng, temperature): ...
