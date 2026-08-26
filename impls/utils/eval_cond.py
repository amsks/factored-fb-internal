"""impls/utils/eval_cond.py — the eval-time conditioning view.

Lives here rather than in impls/main.py because that module defines absl flags at import
time, so any second entrypoint that imported the class from it (scripts/reeval_checkpoint.py
does, to score one checkpoint on several reward tasks) died with a DuplicateFlagError before
reaching its own main. impls/main.py re-exports the name, so `impls.main.InferredCondAgent`
still resolves for everything that referenced it there.
"""
import jax.numpy as jnp


class InferredCondAgent:
    """An EVAL-ONLY view of an `Agent` that rolls the conditioning `infer_eval_cond`
    produced and REFUSES the env's goal. The --eval_on_inferred_cond route.

    The problem it solves. `Agent.sample_actions(obs, goals=...)` does
    `cond = self.cond_eval if goals is None else {**self.cond_eval, "goal": goals}`, and both
    evaluators always pass a goal for any goal-capable critic (env/unified.py::evaluate builds
    a goal frame; impls/utils/evaluation.py reads info['goal']). For every critic that owns a
    `map_eval_cond` hook -- fb, psm, rldp, density_fb -- that goal is then re-encoded INTO the
    conditioning, overwriting whatever `infer_cond` had just solved for. So the reward-sample
    interface existed, was tested in isolation, and could not be exercised by a run:
    `--infer_cond_episodes` set a `cond_eval` the very next `sample_actions` discarded.

    Why a wrapper and not a change to `Agent.sample_actions`. The precedence rule is correct
    for goal-conditioned evaluation and matches the bridge's own split (`sample_actions(obs,
    goal)` maps the goal, `sample_actions_with_z` rolls a precomputed z). Nothing about a
    critic changes here; what changes is which of the two entrypoints THIS RUN drives. Because
    the wrapper only ever forwards `goals=None`, it works identically through both evaluators
    without either of them knowing about it.

    OBSERVABILITY, which is the other half of the requirement: a run must be able to prove it
    evaluated on an inferred index rather than an env goal. `witness()` is logged into
    eval.csv and wandb every eval point -- `inferred_cond_goals_dropped` counts the goal
    frames this wrapper actually intercepted and threw away, so a nonzero value is positive
    evidence that the goal path was live and was overridden, and `inferred_cond_norm`
    fingerprints the vector that was rolled instead.
    """

    # Read by env/base.py::needs_goal_frame, which then does not even BUILD a goal frame
    # for this agent (so `eval_goal_mode=synth` on a task with no goal-in-obs layout is not
    # an error on this route -- there is no goal to synthesise).
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
        """cos(cond, 1_d) -- the scalar that makes density_fb's CONSTANT-REWARD FALLBACK
        visible.

        `inferred_cond_norm` cannot do it: every index this critic family returns is
        projected onto the sqrt(d) sphere, and so is the fallback (density_fb's
        `_sphere_or_neutral` returns z_neutral = 1_d at ||c|| < z_const_eps, norm sqrt(d)
        exactly like a real index). The two are therefore indistinguishable by norm, which
        is how a whole eval can silently roll a reward-free direction. The fallback is
        parallel to 1_d, so this reads 1.0 for it and well below 1.0 for anything the
        reward samples actually informed. On point_mass_maze's sparser targets that is not
        a hypothetical: only 0.31% of RND rows carry nonzero `reach_bottom_right` reward, so
        a small sample pool hits the fallback outright."""
        if primary is None:
            return float("nan")
        v = jnp.asarray(self.cond_eval[primary]).ravel()
        n = jnp.linalg.norm(v)
        ones = jnp.ones_like(v)
        return float(jnp.where(n > 0, v.sum() / (n * jnp.linalg.norm(ones)), jnp.nan))


