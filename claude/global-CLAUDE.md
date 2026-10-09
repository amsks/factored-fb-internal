# graphify
- **graphify** (`~/.claude/skills/graphify/SKILL.md`) - any input to knowledge graph. Trigger: `/graphify`
When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

# Explanations: simple RL language (user rule, 2026-09-07)
Whenever you explain anything to the user — a result, a mechanism, a bug, a design — use plain,
standard reinforcement-learning vocabulary: critic, actor, value, reward, TD target, successor
measure, goal sampling, policy, rollout. No invented or session-local jargon (no "dwell",
"absorbing mass", "point-mass domination", "band pairs", "rule change"), no non-RL analogies,
no measure-theoretic phrasing unless the user asks for the math. Short sentences. Say what the
learner regresses, what the target is, and what the policy does with it. If a technical term
is unavoidable, define it in one clause the first time. This applies to every agent and every
session under this account; when handing off to another agent or machine, copy this rule into
that machine's ~/.claude/CLAUDE.md.

# No config-key jargon, no project-manager register (user rule, 2026-09-08, emphatic)
Never use a config key, flag, or run-directory name as if it were the name of a concept.
Say the thing itself. `measure_hit_spec` -> "which observation columns are the task and how
close counts as reaching it (cube xyz within 0.4)". Identifiers belong only in a file:line
citation, never in the sentence that does the explaining. Answer technical questions in
direct prose: no status tables, no must-lists, no "what survives / what this means for you",
no progress-report framing.
Also banned: repo-internal shorthand the user never coined ("the bridge", "ported", "the seam",
"the arm", "the cell"). Name the actual code path or run. Answer in as few sentences as the
question needs; do not restate, do not add sections.
