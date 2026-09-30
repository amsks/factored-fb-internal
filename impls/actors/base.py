"""Actor protocol, and the conditioning-vector helper both actors share."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Actor(Protocol):
    GROUPS: tuple             # module groups this actor owns
    REQUIRES_COND: frozenset  # cond keys this actor hard-requires

    @classmethod
    def build(cls, ex_obs, ex_actions, config): ...
    def sample_bootstrap(self, net, batch, cond, rng): ...
    def loss(self, params, target, batch, critic, cond, boot): ...
    def sample_actions(self, net, obs, cond, rng, temperature): ...


def cond_vector(cond):
    """The vector the policy is conditioned on, for the paths that do not see the critic
    (bootstrap and acting): the task latent z when the critic has one, else the goal."""
    return cond["z"] if "z" in cond else cond["goal"]
