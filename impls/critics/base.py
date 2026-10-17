"""Critic protocol and critic x actor combo validation."""

from typing import Protocol, runtime_checkable


class ComboError(ValueError):
    """Invalid critic and actor pair."""


@runtime_checkable
class Critic(Protocol):
    GROUPS: tuple  # parameter groups
    STAGES: tuple  # loss stages, run in order
    TARGETS: dict  # group -> polyak tau
    PROVIDES_COND: frozenset  # cond keys produced by sample_cond
    BOOT_COND: str  # "actor_goal" or "value_goal": cond of the TD bootstrap action
    INFER_COND_MARGINAL: str  # batch key of the rows passed to infer_cond
    cvec_dim: int  # width of the actor's conditioning vector

    def eval_cond0(self, ex_obs, config): ...  # initial evaluation cond

    @classmethod
    def build(cls, ex_obs, ex_actions, config): ...
    def stage_groups(self, stage) -> tuple: ...
    def stage_loss(self, stage, params, target, batch, cond, boot): ...
    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad: bool = False): ...
    def cond_vec(self, cond): ...  # actor conditioning vector
    def sample_cond(self, net, batch, rng): ...
    def infer_cond(self, net, obs, rewards): ...


# State distribution that infer_cond's reward samples are drawn from
INFER_COND_MARGINALS = ("observations", "next_observations")


def infer_cond_marginal(critic):
    """Return the critic's validated INFER_COND_MARGINAL."""
    m = getattr(critic, "INFER_COND_MARGINAL", None)
    if m not in INFER_COND_MARGINALS:
        raise ComboError(
            f"{_name(critic)}.INFER_COND_MARGINAL is {m!r}, "
            f"must be one of {list(INFER_COND_MARGINALS)}."
        )
    return m


def infer_cond_rows(critic):
    """Batch key of the rows passed to infer_cond."""
    return infer_cond_marginal(critic)


def _name(obj):
    """Class name of a class or instance."""
    return obj.__name__ if isinstance(obj, type) else type(obj).__name__


# BOOT_COND -> boot dict key of the TD bootstrap action
BOOT_KEY = {"actor_goal": "next_action", "value_goal": "next_action_vgoal"}


def boot_key(critic):
    """Boot dict key the critic's TD target reads."""
    cond = getattr(critic, "BOOT_COND", "actor_goal")
    if cond not in BOOT_KEY:
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is {cond!r}; expected one of {sorted(BOOT_KEY)}"
        )
    return BOOT_KEY[cond]


def check_combo(critic, actor):
    """Validate a critic and actor pair (classes or instances); raise ComboError if invalid."""
    boot_cond = getattr(critic, "BOOT_COND", "actor_goal")
    if boot_cond not in BOOT_KEY:
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is {boot_cond!r}; expected one of {sorted(BOOT_KEY)}"
        )
    if boot_cond == "value_goal" and "value_goal" not in set(
        getattr(critic, "PROVIDES_COND", ())
    ):
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is 'value_goal' but its sample_cond "
            f"does not produce 'value_goal'."
        )
    missing = set(getattr(actor, "REQUIRES_COND", ())) - set(
        getattr(critic, "PROVIDES_COND", ())
    )
    if missing:
        raise ComboError(
            f"{_name(actor)} requires cond key(s) {sorted(missing)} but "
            f"{_name(critic)}.sample_cond only produces "
            f"{sorted(getattr(critic, 'PROVIDES_COND', ()))}"
        )
