---
name: simple-means-plain-rl-language
description: "When the user asks for a 'simple' explanation, use plain standard RL vocabulary (critic, actor, TD target, goal sampling, value estimate) — NOT session-invented jargon (band pairs, restoring force, dwell constant, rule change) and NOT non-RL analogies (coach/player)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f238489d-9b6b-4dab-8e89-37a81b457275
  modified: 2026-08-04T07:04:31.431Z
---

When the user says **"simple"** or "in simple language", they mean: explain
using **standard RL terminology** — critic, actor, Q-value, TD update, goal
sampling, bootstrapping, off-policy, overestimation — at the level of an RL
practitioner, without decoration.

**Why:** After a long probe campaign I explained a mechanism with a
coach/player analogy plus campaign-invented terms ("band pairs", "restoring
force", "the shock", "hits"). The user is an RL researcher; analogies and
invented shorthand add translation overhead instead of removing it.

**Extends to PAPER TEXT (2026-08-04):** the user rejected sentences like
"This structure is precisely what a pointwise density objective cannot
absorb" as "unnecessarily vague". In any paper/writeup: no metaphorical verbs
(absorb, pay, live in, sit at, turn on a knob, chase), no vague rhetorical
topic sentences, no "recipe/ingredient/knob/story/gates" framing. Every
sentence states a concrete claim; every topic sentence says what the section
shows, not what something "cannot absorb".

**How to apply:** In "simple" explanations, replace invented terms with their
standard RL referents (e.g. "band pairs" → "(state, future-goal-of-same-
trajectory) pairs the actor loss uses"; "restoring force" → "informative
critic gradient for the actor"; "hit" → "the indicator reward/grounding term
of the TD target"). No metaphors, no personification. Precision at normal RL
vocabulary level IS the simplicity being requested.


**2026-09-07 promoted to a STANDING RULE for ALL explanations (not only when the user says 'simple'), written into ~/.claude/CLAUDE.md so every session on this machine inherits it. Other machines (e.g. the Vista cluster agent) need the same lines in their ~/.claude/CLAUDE.md.**

**2026-09-08 EMPHATIC (user, all caps): STOP USING CONFIG-KEY NAMES AS CONCEPTS.**
Never write `measure_hit_spec` (or any other config key / run-dir / flag name) as if it
were the name of an idea. The user called it "absolutely fucking disgusting" and "makes no
sense at all". Same message banned project-manager register: no status tables, no
must-lists, no "what survives / what this means for you" framing when the user asked a
technical question.

**How to apply:** say the THING, not the key. `measure_hit_spec` = "which observation
columns count as the task and how close counts as reaching it" — e.g. on cube it is the
cube's xyz within 0.4. `_delta_phi` = "the soft success score: how close the next state's
object position is to the target, softmaxed into weights over the batch". `measure_density_hard`
= "using a one-hot at the state's own successor as the immediate target instead of that soft
score". If a file:line receipt is needed, put the identifier in the citation, not in the
sentence doing the explaining. Answer directly, in prose, no dashboard formatting.
