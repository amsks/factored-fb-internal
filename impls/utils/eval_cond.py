"""Agent wrapper for reward-inference evaluation."""
import jax.numpy as jnp


class InferredCondAgent:
    """Eval-only agent that acts on the inferred conditioning and drops the env's goal.

    Forwards `goals=None` so the critic does not re-encode the goal over the inferred z.
    """

    ROLLS_INFERRED_COND = True

    def __init__(self, agent):
        self.agent = agent
        self.critic = agent.critic
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
        """Logging scalars describing the conditioning used."""
        norms = {k: float(jnp.linalg.norm(jnp.asarray(v).ravel()))
                 for k, v in self.cond_eval.items() if not k.startswith("_")}
        primary = "z" if "z" in norms else (sorted(norms)[0] if norms else None)
        return {"inferred_cond": 1.0,
                "inferred_cond_action_calls": float(self.action_calls),
                "inferred_cond_goals_dropped": float(self.goals_dropped),
                "inferred_cond_norm": norms.get(primary, float("nan")),
                "inferred_cond_cos_neutral": self._cos_neutral(primary)}

    def _cos_neutral(self, primary):
        """cos(cond, 1_d); 1.0 means the constant-reward fallback z was used."""
        if primary is None:
            return float("nan")
        v = jnp.asarray(self.cond_eval[primary]).ravel()
        n = jnp.linalg.norm(v)
        ones = jnp.ones_like(v)
        return float(jnp.where(n > 0, v.sum() / (n * jnp.linalg.norm(ones)), jnp.nan))


