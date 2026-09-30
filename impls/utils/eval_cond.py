"""The eval-time conditioning view for reward-inference evaluation. Lives outside
impls/main.py so a second entry point (scripts/reeval_checkpoint.py) can import it without
re-registering main's absl flags; impls/main.py re-exports it.
"""
import jax.numpy as jnp


class InferredCondAgent:
    """An eval-only view of an `Agent` that rolls the conditioning `infer_eval_cond`
    produced and refuses the env's goal.

    `Agent.sample_actions(obs, goals=...)` merges a supplied goal over cond_eval, and every
    critic with a `map_eval_cond` hook (fb, rldp, hilp, density_fb) re-encodes that goal into
    the conditioning, overwriting the task vector inferred from reward samples. This wrapper
    only ever forwards `goals=None`, so it works through both evaluators unchanged.

    `witness()` is logged into eval.csv every eval point: `inferred_cond_goals_dropped`
    counts the goal frames intercepted (nonzero proves the goal path was live and was
    overridden), and `inferred_cond_norm` fingerprints the vector that was rolled.
    """

    # Read by env/base.py::needs_goal_frame, which then builds no goal frame for this agent.
    ROLLS_INFERRED_COND = True

    def __init__(self, agent):
        self.agent = agent
        self.critic = agent.critic       # evaluators introspect this
        self.cond_eval = agent.cond_eval
        self.reset_counts()

    def reset_counts(self):
        self.action_calls = 0
        self.goals_dropped = 0

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        self.action_calls += 1
        if goals is not None:
            self.goals_dropped += 1
        return self.agent.sample_actions(observations, goals=None, seed=seed,
                                         temperature=temperature)

    def witness(self):
        """Scalars proving what this eval point actually conditioned on."""
        norms = {k: float(jnp.linalg.norm(jnp.asarray(v).ravel()))
                 for k, v in self.cond_eval.items() if not k.startswith("_")}
        primary = "z" if "z" in norms else (sorted(norms)[0] if norms else None)
        return {"inferred_cond": 1.0,
                "inferred_cond_action_calls": float(self.action_calls),
                "inferred_cond_goals_dropped": float(self.goals_dropped),
                "inferred_cond_norm": norms.get(primary, float("nan")),
                "inferred_cond_cos_neutral": self._cos_neutral(primary)}

    def _cos_neutral(self, primary):
        """cos(cond, 1_d): makes density_fb's constant-reward fallback visible.

        Every task vector this critic family returns is projected onto the sqrt(d) sphere,
        and so is the fallback (z_neutral = 1_d when the reward-weighted average is ~0), so
        the norm cannot tell them apart. The fallback is parallel to 1_d, so this reads 1.0
        for it and well below 1.0 for a vector the reward samples actually informed."""
        if primary is None:
            return float("nan")
        v = jnp.asarray(self.cond_eval[primary]).ravel()
        n = jnp.linalg.norm(v)
        ones = jnp.ones_like(v)
        return float(jnp.where(n > 0, v.sum() / (n * jnp.linalg.norm(ones)), jnp.nan))


