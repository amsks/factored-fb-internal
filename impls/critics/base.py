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
    INFER_COND_MARGINAL: str  # which batch column infer_cond()'s `obs` rows come from;
                            # see INFER_COND_MARGINALS below. No default -- a critic that
                            # does not declare it is rejected by infer_cond_rows().
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


# ── which marginal `infer_cond` is solved against ──────────────────────────────────────
# The base measure a zero-shot reward->conditioning readout integrates against belongs to
# the critic's math, not the caller's, and picking the wrong column is a different marginal
# that nothing downstream complains about. So each critic declares its own:
#   "next_observations" -- the successor / query marginal (the whole FB family, whose
#                          backward/index tower is built on successor states)
#   "observations"      -- the current-state marginal
#   "unused"            -- infer_cond ignores `obs` entirely (every goal-conditioned critic
#                          here); `infer_cond_rows` then hands it _UNUSED_ROWS.
INFER_COND_MARGINALS = ("observations", "next_observations", "unused")
_UNUSED_ROWS = "next_observations"


def infer_cond_marginal(critic):
    """`critic`'s declared INFER_COND_MARGINAL, validated. Raises rather than defaulting:
    silently guessing a marginal is exactly the bug this declaration exists to prevent."""
    m = getattr(critic, "INFER_COND_MARGINAL", None)
    if m not in INFER_COND_MARGINALS:
        raise ComboError(
            f"{_name(critic)}.INFER_COND_MARGINAL is {m!r}; every critic must declare one of "
            f"{list(INFER_COND_MARGINALS)} so the --infer_cond_episodes caller knows which "
            f"batch column its infer_cond() expects (the FB family is solved against the "
            f"NEXT-state marginal its backward tower is trained on)."
        )
    return m


def infer_cond_rows(critic):
    """-> the BATCH KEY whose rows feed `critic.infer_cond(net, obs, rewards)`."""
    m = infer_cond_marginal(critic)
    return _UNUSED_ROWS if m == "unused" else m


def _name(obj):
    """Display name for either a class or an instance of one."""
    return obj.__name__ if isinstance(obj, type) else type(obj).__name__


# A critic's TD target is valid at one bootstrap action, and which one is a property of the
# critic, not of the actor drawing it. Actors emit BOTH keys (guarded on "value_goal" in
# cond); this maps a critic's declaration to the key it reads.
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
    needs a V function the critic does not provide, or a cond key its sample_cond does not
    produce.

    Both arguments may be a class or an instance -- every attribute consulted here is a
    class attribute -- so Agent.create can validate BEFORE building the actor, and an
    invalid pair reports its real reason rather than whatever missing config key the
    actor's build() indexes first."""
    boot_cond = getattr(critic, "BOOT_COND", "actor_goal")
    if boot_cond not in BOOT_KEY:
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is {boot_cond!r}; expected one of {sorted(BOOT_KEY)}"
        )
    if boot_cond == "value_goal" and "value_goal" not in set(getattr(critic, "PROVIDES_COND", ())):
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is 'value_goal' but its sample_cond does not "
            f"produce 'value_goal', so it cannot publish cond['_cvec_vgoal'] and no actor "
            f"will emit boot['next_action_vgoal'] (every actor guards that draw on "
            f"'_cvec_vgoal' in cond)"
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
