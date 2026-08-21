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
    PROVIDES_COND: frozenset  # cond keys sample_cond() produces (FB: {"z"}, GCIQL: {"value_goal","goal"})
    BOOT_COND: str          # which cond the TD bootstrap action must be drawn at:
                            # "actor_goal" (default) -> boot["next_action"]
                            # "value_goal"           -> boot["next_action_vgoal"]
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


def _name(obj):
    """Display name for either a class or an instance of one."""
    return obj.__name__ if isinstance(obj, type) else type(obj).__name__


# A critic's TD target is only valid at one bootstrap action, and which one is a
# property of the critic, not of the actor drawing it. Actors emit BOTH keys (guarded
# on "value_goal" in cond); this maps a critic's declaration to the key it reads.
# Deliberately NOT consulted by any actor: keeping it out of the draw is what makes
# this change unable to move an existing number.
BOOT_KEY = {"actor_goal": "next_action", "value_goal": "next_action_vgoal"}


def boot_key(critic):
    """The boot dict key `critic`'s TD target reads. Default "actor_goal"."""
    cond = getattr(critic, "BOOT_COND", "actor_goal")
    if cond not in BOOT_KEY:
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is {cond!r}; expected one of {sorted(BOOT_KEY)}"
        )
    return BOOT_KEY[cond]


def check_combo(critic, actor):
    """Validate a critic×actor pair. Raises ComboError with a named reason if the actor
    needs a V function the critic does not provide, or if the actor requires a cond key
    (e.g. FlowBCActor's "z") the critic's sample_cond does not produce.

    Both arguments may be a class or an instance: every attribute consulted here is a
    class attribute. Agent.create relies on that to validate BEFORE building the actor,
    so an invalid pair reports its real reason instead of whatever missing config key
    the actor's build() happens to index first (e.g. gciql×flowbc dying on z_dim)."""
    boot_cond = getattr(critic, "BOOT_COND", "actor_goal")
    if boot_cond not in BOOT_KEY:
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is {boot_cond!r}; expected one of {sorted(BOOT_KEY)}"
        )
    if boot_cond == "value_goal" and "value_goal" not in set(getattr(critic, "PROVIDES_COND", ())):
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is 'value_goal' but its sample_cond does not "
            f"produce 'value_goal', so no actor will emit boot['next_action_vgoal'] "
            f"(every actor guards that draw on \"value_goal\" in cond)"
        )
    if getattr(actor, "NEEDS_V", False) and not getattr(critic, "PROVIDES_V", False):
        raise ComboError(
            f"{_name(actor)} requires a V function but {_name(critic)}.PROVIDES_V is False"
        )
    missing = set(getattr(actor, "REQUIRES_COND", ())) - set(getattr(critic, "PROVIDES_COND", ()))
    if missing:
        raise ComboError(
            f"{_name(actor)} requires cond key(s) {sorted(missing)} but "
            f"{_name(critic)}.sample_cond only produces "
            f"{sorted(getattr(critic, 'PROVIDES_COND', ()))}"
        )
