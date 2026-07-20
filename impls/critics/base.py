"""Critic protocol + combo validation for the factored critic×actor seam."""
from typing import Protocol, runtime_checkable


class ComboError(ValueError):
    """Raised when a critic×actor pair is structurally invalid."""


@runtime_checkable
class Critic(Protocol):
    GROUPS: tuple           # module groups this critic owns
    STAGES: tuple           # ordered; run sequentially by the generic update
    TARGETS: dict           # group -> polyak tau  (FB: forward/left_encoder=f_tau, backward=b_tau)
    PROVIDES_V: bool        # whether v_fn is available (AWR requires it)
    cvec_dim: int           # width of cond_vec(cond) — the actor's conditioning input

    def eval_cond0(self, ex_obs, config): ...   # initial eval-time cond (FB: {"z":..}, GC: {"goal":..})

    @classmethod
    def build(cls, ex_obs, ex_actions, config): ...
    def stage_groups(self, stage) -> tuple: ...
    def stage_loss(self, stage, params, target, batch, cond, boot): ...
    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad: bool = False): ...
    def cond_vec(self, cond): ...            # conditioning vector actors concat onto obs
    def sample_cond(self, net, batch, rng): ...
    def infer_cond(self, net, obs, rewards): ...


def check_combo(critic, actor):
    """Validate a critic×actor pair at construction. Raises ComboError with a named
    reason if the actor needs a V function the critic does not provide."""
    if getattr(actor, "NEEDS_V", False) and not getattr(critic, "PROVIDES_V", False):
        raise ComboError(
            f"{type(actor).__name__} requires a V function but "
            f"{type(critic).__name__}.PROVIDES_V is False"
        )
