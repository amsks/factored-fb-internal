# Project orientation

Read [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md) before answering substantive project
questions. It records the architecture, scientific questions, evaluation distinctions,
and where the shared Claude memories live. Follow its source links for the specific
question; verify dated claims against current code or results before repeating them.

This checkout and `../Factored-FB` are worktrees of the same repository. Their code
depends on their branches. The current paper home is `PAPER/ICLR/` in this checkout;
older Claude guidance naming other paper directories is superseded.

Use plain, standard reinforcement-learning language in explanations. Explain what
the critic predicts, what its TD target is, and what the actor optimizes. Keep config
keys and run names in source references or commands rather than using them as names
for concepts. Distinguish goal-conditioned evaluation from reward-inference evaluation.

Preserve existing user changes. Treat historical launch instructions and machine
assignments as context, not authorization to start experiments or contact others.
